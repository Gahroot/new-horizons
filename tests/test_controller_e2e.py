import io
import json
import shutil
import sqlite3
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from horizons.cli import main
from horizons.controller import Controller
from horizons.kb.store import KB
from horizons.llm import ScriptedClient
from horizons.template import load_topic
from tests.helpers import EXAMPLES, TINY_EVALUATOR, TINY_TOPIC, TempDirCase, hyp, requires_docker

BMAP = {"summary": "tiny map", "known_approaches": [], "gaps": [], "promising_directions": []}
FIDELITY_OK = {"implements": True, "missing": ""}

# Scales value() by a factor drawn from HORIZONS_SEED: absolute scores are noisy, paired ratios are exact.
SEEDED_EVALUATOR = """
import json, os, random, runpy
MARKER = os.environ.pop("HORIZONS_RESULT_MARKER")
scale = random.Random(int(os.environ["HORIZONS_SEED"])).uniform(0.5, 2.0)
ns = runpy.run_path(os.environ["HORIZONS_CANDIDATE"])
v = ns["value"]()
print(MARKER + json.dumps({"score": v * scale, "correct": int(v > 0)}))
"""


def code(v: int) -> str:
    return f"```python\ndef value():\n    return {v}\n```"


@requires_docker
class ControllerE2E(TempDirCase):
    def setUp(self):
        super().setUp()
        self.topic = self.tmp / "topic"
        self.ws = self.tmp / "ws"

    def make_tiny(self, script: dict, topic_text: str = TINY_TOPIC):
        self.write("topic/baseline.py", "def value():\n    return 1\n")
        self.write("topic/evaluate.py", TINY_EVALUATOR)
        self.write_json("topic/script.json", {"check_fidelity": [FIDELITY_OK], **script})
        self.write("topic/topic.toml", topic_text)
        return load_topic(self.topic / "topic.toml")

    def run_ctl(self, spec, client, resume=None):
        kb = KB(self.ws / "kb.sqlite")
        self.addCleanup(kb.close)
        ctl = Controller(spec, kb, client, self.ws, offline=True, out=lambda m: None)
        run_id = ctl.resume(resume) if resume else ctl.start()
        return run_id, ctl.run(), kb

    def test_memory_example_via_cli(self):
        shutil.copytree(EXAMPLES / "memory_reduction", self.topic)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["--workspace", str(self.ws), "run", str(self.topic / "topic.toml"), "--llm", "scripted"])
        text = out.getvalue()
        self.assertEqual(rc, 0, text)
        self.assertIn("REFUTED", text)
        self.assertIn("REFINE", text)
        self.assertIn("SUPPORTED", text)
        kb = KB(self.ws / "kb.sqlite")
        self.addCleanup(kb.close)
        run = kb.list_runs()[0]
        self.assertEqual(run["status"], "success")
        statuses = {h["status"] for h in kb.hypotheses(run["id"])}
        self.assertTrue({"refuted", "supported", "needs_resources"} <= statuses)
        self.assertEqual(len(kb.findings(run["id"], "breakthrough")), 1)
        reps = [e for e in kb.experiments(run["id"]) if e["purpose"] == "replication"]
        self.assertEqual(len(reps), 2)
        report = (self.ws / "runs" / run["id"] / "report.md").read_text()
        self.assertIn("Target met and replicated", report)
        self.assertIn("arxiv:0000.00000", report)  # the stripped hallucinated citation is reported as a lesson
        self.assertNotIn("(`arxiv:0000.00000`)", report)  # ...and not cited in the boundary map

    def test_budget_exhaustion_stops_and_labels_best(self):
        spec = self.make_tiny({"boundary_map": BMAP,
                               "hypothesize": [{"hypotheses": [hyp(s)]} for s in (
                                   "Returning three instead of one raises the score",
                                   "Caching a precomputed five doubles throughput somehow",
                                   "Loop unrolling toward seven improves arithmetic results",
                                   "Bit shifting yields eight through clever tricks")],
                               "implement": [code(3), code(5), code(7), code(8)],
                               "reflect": {"explanation": "too small", "lessons": []}})
        run_id, status, kb = self.run_ctl(spec, ScriptedClient.from_file(self.topic / "script.json"))
        self.assertEqual(status, "stopped")
        run = kb.get_run(run_id)
        self.assertRegex(run["stop_reason"], "budget exhausted: iterations|stalled")
        self.assertEqual(run["state"]["best"]["value"], 8.0)
        report = (self.ws / "runs" / run_id / "report.md").read_text()
        self.assertIn("Target NOT met", report)

    def test_disallowed_tool_is_never_executed(self):
        spec = self.make_tiny({"boundary_map": BMAP,
                               "hypothesize": {"hypotheses": [hyp("Search the web for faster constants",
                                                                  kind="literature")]},
                               "reflect": {"lessons": []}})
        run_id, status, kb = self.run_ctl(spec, ScriptedClient.from_file(self.topic / "script.json"))
        self.assertEqual(status, "stopped")
        hyps = kb.hypotheses(run_id)
        self.assertTrue(hyps)
        self.assertTrue(all(h["status"] == "needs_resources" for h in hyps))
        self.assertEqual([e["purpose"] for e in kb.experiments(run_id)], ["baseline", "baseline"])

    def test_crash_then_resume(self):
        script = {"boundary_map": BMAP,
                  "hypothesize": {"hypotheses": [hyp("Return the constant 12 to reach the target")]},
                  "implement": code(12)}
        spec = self.make_tiny({**script, "implement": "no code block and not python ("})
        run_id, status, kb = self.run_ctl(spec, ScriptedClient.from_file(self.topic / "script.json"))
        # The syntax error is caught before running and classed errored -> REFINE -> ... reflect is not
        # needed for errored outcomes, so the run keeps refining until pivots/patience/budget end it.
        self.assertIn(status, ("stopped",))
        # Now simulate a crash mid-run: missing 'reflect' raises inside execute for a refuted candidate.
        spec2 = self.make_tiny({**script, "implement": code(4)})
        run2, status2, kb2 = self.run_ctl(spec2, ScriptedClient.from_file(self.topic / "script.json"))
        self.assertEqual(status2, "failed")
        self.assertEqual(kb2.get_run(run2)["state"]["phase"], "execute")
        spent = kb2.get_run(run2)["budget"]["sandbox_runs"]
        self.assertGreater(spent, 0)
        # Fix the script and resume: the same run continues and succeeds.
        spec3 = self.make_tiny({**script, "reflect": {"lessons": []}})
        _, status3, kb3 = self.run_ctl(spec3, ScriptedClient(json.loads((self.topic / "script.json").read_text())
                                                             | {"implement": [code(12)]}), resume=run2)
        self.assertEqual(status3, "success")
        self.assertGreater(kb3.get_run(run2)["budget"]["sandbox_runs"], spent)


    def test_code_that_does_not_match_the_hypothesis_is_sent_back(self):
        spec = self.make_tiny({"boundary_map": BMAP,
                               "hypothesize": {"hypotheses": [hyp("Caching the answer in a lookup table reaches 12")]},
                               "implement": [code(12), code(12)],
                               "check_fidelity": [{"implements": False, "missing": "no lookup table is added"},
                                                  FIDELITY_OK],
                               "reflect": {"lessons": []}})
        client = ScriptedClient.from_file(self.topic / "script.json")
        run_id, status, kb = self.run_ctl(spec, client)
        self.assertEqual(status, "success")
        cands = [e for e in kb.experiments(run_id) if e["purpose"] == "candidate"]
        self.assertEqual([e["status"] for e in cands], ["errored", "ok"])
        self.assertIn("does not implement the hypothesis: no lookup table is added", cands[0]["error"])
        self.assertIsNone(cands[0]["artifacts_dir"])  # rejected before anything ran in the sandbox
        self.assertEqual(client.calls["check_fidelity"], 2)
        h = kb.hypotheses(run_id)[0]
        self.assertIs(h["outcome"]["fidelity"], True)
        report = (self.ws / "runs" / run_id / "report.md").read_text()
        self.assertIn("checked to implement the hypothesis", report)

    def test_paired_mode_compares_each_run_with_the_baseline_on_the_same_seed(self):
        spec = self.make_tiny({"boundary_map": BMAP,
                               "hypothesize": {"hypotheses": [hyp("Returning three triples the score")]},
                               "implement": code(3), "reflect": {"lessons": []}},
                              TINY_TOPIC.replace("target = { absolute = 10 }", "target = { relative_to_baseline = 1.5 }")
                              .replace("replications = 2", "replications = 3\npaired = true"))
        self.write("topic/evaluate.py", SEEDED_EVALUATOR)
        run_id, status, kb = self.run_ctl(spec, ScriptedClient.from_file(self.topic / "script.json"))
        self.assertEqual(status, "success")
        exps = kb.experiments(run_id)
        purposes = [e["purpose"] for e in exps]
        self.assertEqual(purposes.count("paired-baseline"), 3)
        seeds = {}
        for e in exps:
            seed = Path(e["artifacts_dir"], "seed.txt").read_text()
            seeds.setdefault(seed, []).append((e["purpose"], e["metrics"]["score"]))
        pairs = [v for v in seeds.values() if len(v) == 2]
        self.assertEqual(len(pairs), 3)  # each candidate/replication shares its seed with one baseline run
        for pair in pairs:
            got = dict(pair)
            cand = got.get("candidate", got.get("replication"))
            self.assertAlmostEqual(cand, 3 * got["paired-baseline"])
        report = (self.ws / "runs" / run_id / "report.md").read_text()
        self.assertIn("Paired comparison", report)

    def test_paired_mode_skips_the_baseline_run_when_a_guard_fails(self):
        spec = self.make_tiny({"boundary_map": BMAP,
                               "hypothesize": {"hypotheses": [hyp("Returning three triples the score")]},
                               "implement": [code(-1), code(3)], "reflect": {"lessons": []}},
                              TINY_TOPIC.replace("target = { absolute = 10 }", "target = { relative_to_baseline = 1.5 }")
                              .replace("replications = 2", "replications = 2\npaired = true"))
        self.write("topic/evaluate.py", SEEDED_EVALUATOR)
        run_id, status, kb = self.run_ctl(spec, ScriptedClient.from_file(self.topic / "script.json"))
        self.assertEqual(status, "success")
        purposes = [e["purpose"] for e in kb.experiments(run_id)]
        # guard-failing candidate: no paired run; the fixed candidate and its replication: one each
        self.assertEqual(purposes, ["baseline", "baseline", "candidate", "candidate", "paired-baseline",
                                    "replication", "paired-baseline"])


class DebugFlagTests(TempDirCase):
    """--debug must print the traceback of a failed run; without it only a one-line error is shown."""

    def run_cli(self, *extra):
        self.write("topic.toml", '''
            [topic]
            name = "dbg"
            goal = "Exercise the failure path of a literature-only run"
            [tools]
            allowed = ["literature"]
            [tools.literature]
            sources = []
            fixture = "papers.json"
            [validation]
            metric = "evidence_balance"
            direction = "maximize"
            target = { absolute = 0.5 }
            [llm]
            scripted_file = "script.json"
        ''')
        self.write_json("papers.json", [])
        self.write_json("script.json", {})  # no responses -> the first LLM call fails
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = main(["--workspace", str(self.tmp / "ws"), "run", str(self.tmp / "topic.toml"),
                       "--llm", "scripted", *extra])
        return rc, out.getvalue(), err.getvalue()

    def test_debug_prints_traceback(self):
        rc, out, err = self.run_cli("--debug")
        self.assertEqual(rc, 1)
        self.assertIn("has no response for task", out)
        self.assertIn("Traceback (most recent call last)", err)

    def test_no_traceback_without_debug(self):
        rc, out, err = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertIn("has no response for task", out)
        self.assertNotIn("Traceback", err)


class LiteratureE2E(TempDirCase):
    """Literature-only topic, offline with a local fixture: no Docker needed."""

    def test_evidence_quotes_are_verified(self):
        papers = [{"id": f"fx:{i}", "title": f"Paper {i}",
                   "abstract": f"Streaming minima reduce memory use in experiment number {i} considerably."}
                  for i in range(4)]
        papers.append({"id": "fx:against", "title": "Counter",
                       "abstract": "We find streaming minima do not reduce memory for small inputs."})
        self.write_json("papers.json", papers)
        self.write("topic.toml", '''
            [topic]
            name = "lit"
            goal = "Find well-supported claims about streaming minima and memory"
            [tools]
            allowed = ["literature"]
            [tools.literature]
            sources = []
            fixture = "papers.json"
            [validation]
            metric = "evidence_balance"
            direction = "maximize"
            target = { absolute = 0.5 }
            guards = [{ metric = "supporting_papers", op = ">=", value = 3 }]
            replications = 2
            [budget]
            max_iterations = 2
            [llm]
            scripted_file = "script.json"
        ''')
        judgements = [{"paper_id": f"fx:{i}", "relation": "supports", "confidence": 0.9,
                       "evidence_quote": f"Streaming minima reduce memory use in experiment number {i} considerably."}
                      for i in range(4)]
        judgements += [
            {"paper_id": "fx:against", "relation": "challenges",
             "evidence_quote": "streaming minima do not reduce memory for small inputs"},
            {"paper_id": "fx:0", "relation": "challenges", "evidence_quote": "This sentence is not in the abstract at all."},
            {"paper_id": "fx:invented", "relation": "supports", "evidence_quote": "made up paper id here ok"},
        ]
        self.write_json("script.json", {
            "deconstruct_queries": {"queries": ["streaming minima memory"]},
            "boundary_map": BMAP,
            "hypothesize": {"hypotheses": [hyp("Streaming minima reduce memory use", kind="literature",
                                               citations=["fx:1", "fx:nope"])]},
            "classify_evidence": {"judgements": judgements},
        })
        ws = self.tmp / ".horizons"
        with redirect_stdout(io.StringIO()):
            rc = main(["--workspace", str(ws), "run", str(self.tmp / "topic.toml"), "--llm", "scripted"])
        self.assertEqual(rc, 0)
        kb = KB(ws / "kb.sqlite")
        self.addCleanup(kb.close)
        run = kb.list_runs()[0]
        h = kb.hypotheses(run["id"])[0]
        self.assertEqual(h["status"], "supported")
        self.assertEqual(h["citations"], ["fx:1"])  # unknown citation dropped
        exp = kb.experiments(run["id"], h["id"])[0]
        self.assertEqual(exp["metrics"]["supporting_papers"], 4)
        self.assertEqual(exp["metrics"]["challenging_papers"], 1)
        self.assertEqual(exp["metrics"]["unverified_quotes"], 1)  # fake quote dropped; fx:0 still supports once
        edges = [e for e in kb.edges(run["id"]) if e["dst"] == h["id"] and e["src"].startswith("fx:")]
        self.assertEqual(sorted(e["type"] for e in edges), ["challenges"] + ["supports"] * 4)
        report = (ws / "runs" / run["id"] / "report.md").read_text()
        self.assertIn("quotes verified against abstracts", report)


@requires_docker
class DataQueryE2E(TempDirCase):
    def test_data_query_topic(self):
        con = sqlite3.connect(self.tmp / "data.sqlite")
        con.execute("CREATE TABLE sites (name TEXT, elevation REAL, finds INTEGER)")
        con.executemany("INSERT INTO sites VALUES (?, ?, ?)",
                        [(f"s{i}", float(i * 10), i % 7 + (5 if i > 30 else 0)) for i in range(60)])
        con.commit()
        con.close()
        self.write("evaluate_data.py", '''
            """Reads rows.json (columns elevation, finds) and reports the mean finds as `mean_finds`."""
            import json, os
            d = json.load(open(os.path.join(os.environ["HORIZONS_WORK"], "rows.json")))
            i = d["columns"].index("finds")
            vals = [r[i] for r in d["rows"]]
            print(os.environ["HORIZONS_RESULT_MARKER"] + json.dumps({"mean_finds": sum(vals) / max(len(vals), 1),
                                                                     "n": len(vals)}))
        ''')
        self.write("topic.toml", '''
            [topic]
            name = "sites"
            goal = "Find a subset of sites whose mean finds is at least 7"
            [tools]
            allowed = ["data_query"]
            [tools.data_query]
            database = "data.sqlite"
            evaluator = "evaluate_data.py"
            timeout_s = 20
            memory = "256m"
            [validation]
            metric = "mean_finds"
            direction = "maximize"
            target = { absolute = 7 }
            guards = [{ metric = "n", op = ">=", value = 10 }]
            replications = 2
            [budget]
            max_iterations = 3
            [llm]
            scripted_file = "script.json"
        ''')
        self.write_json("script.json", {
            "boundary_map": BMAP,
            "hypothesize": [{"hypotheses": [hyp("Low elevation sites have more finds", kind="data_query")]},
                            {"hypotheses": [hyp("High elevation sites above 300m have more finds", kind="data_query")]}],
            "data_query_sql": [{"sql": "DELETE FROM sites"}, {"sql": "SELECT elevation, finds FROM sites WHERE elevation < 100"},
                               {"sql": "SELECT elevation, finds FROM sites WHERE elevation > 300"}],
            "reflect": {"lessons": [{"category": "analysis", "text": "Low sites are not richer."}]},
        })
        ws = self.tmp / ".horizons"
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            rc = main(["--workspace", str(ws), "run", str(self.tmp / "topic.toml"), "--llm", "scripted"])
        self.assertEqual(rc, 0, err.getvalue())
        kb = KB(ws / "kb.sqlite")
        self.addCleanup(kb.close)
        run = kb.list_runs()[0]
        outcomes = [h["status"] for h in kb.hypotheses(run["id"])]
        self.assertEqual(outcomes, ["refuted", "supported"])
        exps = kb.experiments(run["id"])
        self.assertEqual(exps[0]["status"], "errored")  # the DELETE was refused
        self.assertIn("only SELECT", exps[0]["error"])
        con = sqlite3.connect(self.tmp / "data.sqlite")
        self.assertEqual(con.execute("SELECT count(*) FROM sites").fetchone()[0], 60)
        con.close()


if __name__ == "__main__":
    unittest.main()
