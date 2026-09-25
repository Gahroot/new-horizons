import unittest

from horizons.agents import evaluator as ev
from horizons.agents.executor import verify_quote
from horizons.agents.hypothesizer import sanitize
from horizons.agents.reflector import classify_error, decide
from horizons.report import esc, fence
from horizons.template import load_topic, parse_topic
from tests.helpers import EXAMPLES

SPEC = load_topic(EXAMPLES / "memory_reduction" / "topic.toml")  # minimize, -50% vs baseline, guard correct == 1
OK = {"peak_memory_mb": 4.0, "correct": 1}


class ScreenTests(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(ev.screen(SPEC, "timeout", {}, "killed", 10)[0], "errored")
        self.assertEqual(ev.screen(SPEC, "ok", {"peak_memory_mb": 1, "correct": 0}, "", 10)[0], "errored")
        self.assertEqual(ev.screen(SPEC, "ok", {"correct": 1}, "", 10)[0], "errored")
        self.assertEqual(ev.screen(SPEC, "ok", {"peak_memory_mb": 6, "correct": 1}, "", 10)[0], "refuted")
        self.assertEqual(ev.screen(SPEC, "ok", OK, "", 10)[0], "replicate")


class JudgeTests(unittest.TestCase):
    base = [10.0, 10.1, 9.9]

    def test_supported_when_all_replications_pass(self):
        v = ev.judge(SPEC, [("ok", OK)] * 3, self.base, 10.0, [])
        self.assertEqual(v.outcome, "supported")
        self.assertTrue(v.improved)
        self.assertAlmostEqual(v.p_value, 0.05)

    def test_inconclusive_when_replication_fails(self):
        v = ev.judge(SPEC, [("ok", OK), ("ok", {"peak_memory_mb": 7.0, "correct": 1}), ("ok", OK)], self.base, 10, [])
        self.assertEqual(v.outcome, "inconclusive")
        self.assertIn("did not replicate", v.reason)

    def test_inconclusive_when_replication_crashes(self):
        v = ev.judge(SPEC, [("ok", OK), ("errored", {}), ("ok", OK)], self.base, 10, [])
        self.assertEqual(v.outcome, "inconclusive")

    def test_guard_failure_on_replication_blocks_success(self):
        v = ev.judge(SPEC, [("ok", OK), ("ok", {"peak_memory_mb": 4.0, "correct": 0}), ("ok", OK)], self.base, 10, [])
        self.assertNotEqual(v.outcome, "supported")

    def test_significance_with_holm(self):
        raw = {
            "topic": {"name": "s", "goal": "g"},
            "tools": {"allowed": ["python_sandbox"],
                      "python_sandbox": {"baseline": "baseline.py", "evaluator": "evaluate.py"}},
            "validation": {"metric": "peak_memory_mb", "direction": "minimize", "kind": "significance",
                           "replications": 4},
        }
        spec = parse_topic(raw, EXAMPLES / "memory_reduction")
        base = [10.0, 10.2, 9.9, 10.1]
        good = [("ok", {"peak_memory_mb": x}) for x in (5.0, 5.1, 4.9, 5.2)]
        v = ev.judge(spec, good, base, 10.0, [])
        self.assertEqual(v.outcome, "supported")  # p = 1/70
        # p = 1/70 ~ 0.0143 is the smallest of 4 tests, so Holm multiplies it by 4 -> 0.057 > alpha
        v2 = ev.judge(spec, good, base, 10.0, [0.2, 0.3, 0.4])
        self.assertEqual(v2.outcome, "inconclusive")
        self.assertIn("Holm", v2.reason)


class ReflectDecisionTests(unittest.TestCase):
    def test_decide(self):
        self.assertEqual(decide("supported", 0, 2), "STOP")
        self.assertEqual(decide("errored", 0, 2), "REFINE")
        self.assertEqual(decide("errored", 2, 2), "PIVOT")
        self.assertEqual(decide("refuted", 0, 2), "PIVOT")
        self.assertEqual(decide("inconclusive", 0, 2), "PIVOT")

    def test_classify_error(self):
        self.assertEqual(classify_error("killed after 60s timeout"), "system")
        self.assertEqual(classify_error("SyntaxError: invalid syntax Traceback"), "experiment")


class HypothesisSanitizeTests(unittest.TestCase):
    def test_schema_enforced(self):
        raw = {"hypotheses": [
            {"statement": "Use a streaming minimum to save memory", "test_kind": "python_sandbox",
             "citations": ["demo:a", "fake:1"], "evil_field": "rm -rf /"},
            {"statement": "short"},
            {"statement": "Run a shell command to download a faster library", "test_kind": "shell"},
            "not a dict",
        ]}
        out = sanitize(raw, ("python_sandbox",), {"demo:a"})
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["citations"], ["demo:a"])
        self.assertNotIn("evil_field", out[0])
        self.assertEqual(out[1]["readiness"], "needs_resources")
        self.assertIn("not allowed", out[1]["missing"])
        self.assertEqual(out[1]["test_kind"], "unknown")


class EvidenceAndReportTests(unittest.TestCase):
    def test_quote_must_be_in_abstract(self):
        paper = {"title": "T", "abstract": "Sparse attention  matches dense quality at 32k tokens. Other text."}
        self.assertTrue(verify_quote("sparse attention matches dense quality at 32k tokens", paper))
        self.assertFalse(verify_quote("sparse attention beats dense quality at 32k tokens", paper))
        self.assertFalse(verify_quote("Sparse", paper))  # too short to count as evidence

    def test_markdown_escaping(self):
        s = esc("<script>alert(1)</script> | [x](javascript:y) `code`")
        self.assertNotIn("<script>", s)
        self.assertNotIn("[x](", s)  # brackets escaped, so no link can form
        self.assertIn("\\[x\\]", s)
        self.assertIn("\\|", s)
        f = fence("a = '```'\n", "python")
        self.assertTrue(f.startswith("````python"))


if __name__ == "__main__":
    unittest.main()
