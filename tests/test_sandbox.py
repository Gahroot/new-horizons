import unittest
from unittest import mock

from horizons.template import Runtime
from horizons.tools.sandbox import DockerSandbox, SandboxUnavailable, make_sandbox, parse_metrics
from tests.helpers import TINY_EVALUATOR, TempDirCase, requires_docker


class ParseMetricsTests(unittest.TestCase):
    M = "@@M@@"

    def test_last_marked_line_wins(self):
        out = f"noise\n{self.M}{{\"a\": 1}}\n{self.M}{{\"a\": 2, \"ok\": true}}\n"
        self.assertEqual(parse_metrics(out, self.M), ({"a": 2.0, "ok": 1.0}, ""))

    def test_rejects_bad_output(self):
        for out in ("", f"{self.M}not json", f"{self.M}[1]", f'{self.M}{{"a": NaN}}', f'{self.M}{{"a": "x"}}',
                    f'{self.M}{{"a": Infinity}}', f"{self.M}{{}}", '@@OTHER@@{"a": 1}'):
            with self.subTest(out=out):
                metrics, why = parse_metrics(out, self.M)
                self.assertIsNone(metrics)
                self.assertTrue(why)


class UnsafeLocalGateTests(TempDirCase):
    def test_unsafe_local_requires_flag(self):
        with self.assertRaises(SandboxUnavailable):
            make_sandbox(Runtime(sandbox="unsafe-local"), self.tmp, None, accept_unsafe_local=False)

    def test_unreadable_topic_fails_closed_on_linux(self):
        self.tmp.chmod(0o700)
        with mock.patch("horizons.tools.sandbox.docker_available", return_value=(True, "")), \
                mock.patch("horizons.tools.sandbox.sys.platform", "linux"):
            with self.assertRaisesRegex(SandboxUnavailable, "chmod o\\+rx"):
                DockerSandbox(Runtime(), self.tmp)

    def test_argv_has_hardening_flags(self):
        ev = self.write("evaluate.py", TINY_EVALUATOR)
        sb = DockerSandbox.__new__(DockerSandbox)  # build argv without needing a daemon
        sb.docker, sb.rt, sb.root, sb.workspace = "docker", Runtime(memory="256m", cpus=0.5), self.tmp, self.tmp / ".horizons"
        sb.auth_dir = self.tmp / "secrets" / "login"  # a login folder that happens to sit inside the topic
        argv = sb.argv("n", self.tmp / "att", ev, {"K": "V"})
        joined = " ".join(argv)
        for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                     "--pids-limit 256", "--memory 256m", "--memory-swap 256m", "--cpus 0.5", "--user 65534:65534",
                     f"source={self.tmp},target=/topic,readonly", "--tmpfs /topic/.horizons:ro",
                     "--tmpfs /topic/secrets/login:ro"):
            self.assertIn(flag, joined)
        self.assertNotIn("--privileged", joined)
        self.assertEqual(argv[-3:], ["python", "-I", "/topic/evaluate.py"])


@requires_docker
class LoginFileExposureTests(TempDirCase):
    def test_refuses_topic_folder_that_is_the_login_folder(self):
        with mock.patch.dict("os.environ", {"HORIZONS_AUTH_FILE": str(self.tmp / "auth.json")}):
            with self.assertRaisesRegex(SandboxUnavailable, "login file"):
                DockerSandbox(Runtime(), self.tmp)

    def test_login_folder_inside_topic_is_hidden_in_container(self):
        (self.tmp / "login").mkdir()
        (self.tmp / "login" / "auth.json").write_text('{"claude": "secret"}')
        ev = self.write("evaluate.py", TINY_EVALUATOR)
        att = self.tmp / "att"
        att.mkdir()
        (att / "candidate.py").write_text("import os\ndef value():\n    return len(os.listdir('/topic/login'))\n")
        with mock.patch.dict("os.environ", {"HORIZONS_AUTH_FILE": str(self.tmp / "login" / "auth.json")}):
            r = DockerSandbox(Runtime(timeout_s=20, memory="128m"), self.tmp).run(att, ev)
        self.assertEqual(r.status, "ok", r.error)
        self.assertEqual(r.metrics["score"], 0.0)


@requires_docker
class DockerSandboxTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.ev = self.write("evaluate.py", TINY_EVALUATOR)
        self.sb = DockerSandbox(Runtime(timeout_s=8, memory="128m"), self.tmp, self.tmp / ".horizons")
        (self.tmp / ".horizons").mkdir()
        (self.tmp / ".horizons" / "secret.txt").write_text("kb contents")

    def run_candidate(self, code: str):
        att = self.tmp / "att"
        att.mkdir(exist_ok=True)
        (att / "candidate.py").write_text(code)
        return self.sb.run(att, self.ev)

    def test_ok_as_nobody(self):
        r = self.run_candidate("import os\ndef value():\n    return os.getuid()\n")
        self.assertEqual(r.status, "ok", r.error)
        self.assertEqual(r.metrics["score"], 65534.0)

    def test_network_is_denied(self):
        r = self.run_candidate(
            "import socket\ndef value():\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n    return 1\n")
        self.assertEqual(r.status, "errored")
        self.assertRegex(r.error, "unreachable|OSError|Errno")

    def test_evaluator_and_topic_are_read_only(self):
        r = self.run_candidate("def value():\n    open('/topic/evaluate.py', 'a').write('x')\n    return 1\n")
        self.assertEqual(r.status, "errored")
        self.assertIn("Read-only", r.error)
        self.assertNotIn("x", self.ev.read_text()[-2:])

    def test_workspace_hidden(self):
        r = self.run_candidate("import os\ndef value():\n    return len(os.listdir('/topic/.horizons'))\n")
        self.assertEqual(r.status, "ok", r.error)
        self.assertEqual(r.metrics["score"], 0.0)

    def test_timeout_is_killed(self):
        r = self.run_candidate("import time\ndef value():\n    time.sleep(60)\n    return 1\n")
        self.assertEqual(r.status, "timeout")
        self.assertLess(r.duration_s, 30)
        self.assertIn("8s timeout (+5s container start-up allowance)", r.error)

    def test_memory_limit_kills(self):
        r = self.run_candidate("def value():\n    x = bytearray(600 * 1024 * 1024)\n    return len(x)\n")
        self.assertEqual(r.status, "errored")
        self.assertRegex(r.error, "137|MemoryError")

    def test_candidate_cannot_read_marker_from_env(self):
        r = self.run_candidate("import os, json\n"
                               "m = os.environ.get('HORIZONS_RESULT_MARKER')\n"
                               "if m: print(m + json.dumps({'score': 999, 'correct': 1}))\n"
                               "def value():\n    return 1\n")
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.metrics["score"], 1.0)

    def test_marker_is_secret_per_run(self):
        # A candidate that guesses the marker format can't forge a result line.
        r = self.run_candidate("import os\nprint('@@HORIZONS_RESULT_deadbeef@@{\"score\": 999}')\n"
                               "def value():\n    return 1\n")
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.metrics["score"], 1.0)
        self.assertNotIn("HORIZONS_RESULT_", r.stdout.replace("@@HORIZONS_RESULT_deadbeef@@", ""))


if __name__ == "__main__":
    unittest.main()
