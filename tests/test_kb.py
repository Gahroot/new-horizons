import unittest

from horizons.kb import graph
from horizons.kb.recall import max_similarity, recall, rrf
from horizons.kb.store import KB
from horizons.kb.vectors import cosine, embed
from tests.helpers import TempDirCase, hyp


class KBTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.kb = KB(self.tmp / "kb.sqlite")
        self.addCleanup(self.kb.close)
        self.run_id = self.kb.create_run("topic-a", "p", "goal", "scripted:x")

    def test_rrf_formula_and_order(self):
        a = [{"id": 1, "kind": "lesson", "ref_id": "a", "text": "A"}, {"id": 2, "kind": "lesson", "ref_id": "b", "text": "B"}]
        b = [{"id": 2, "kind": "lesson", "ref_id": "b", "text": "B"}]
        hits = rrf({"vector": a, "bm25": b}, k=60)
        self.assertEqual(hits[0].ref_id, "b")
        self.assertAlmostEqual(hits[0].score, 1 / 62 + 1 / 61)
        self.assertEqual(hits[0].arms, {"vector": 2, "bm25": 1})
        self.assertAlmostEqual(hits[1].score, 1 / 61)

    def test_recall_finds_relevant_lesson_and_filters_topic(self):
        self.kb.add_lesson(self.run_id, "topic-a", "experiment", "Chunking the distance matrix lowers peak memory")
        self.kb.add_lesson(self.run_id, "topic-a", "system", "Docker needs to be running before the sandbox starts")
        other = self.kb.create_run("topic-b", "p", "g", "x")
        self.kb.add_lesson(other, "topic-b", "experiment", "Chunking the distance matrix lowers peak memory in topic b")
        hits = recall(self.kb, "reduce peak memory with chunking", kinds=("lesson",), topic="topic-a")
        self.assertTrue(hits)
        self.assertIn("Chunking", hits[0].text)
        self.assertNotIn("topic b", " ".join(h.text for h in hits))
        self.assertEqual(set(hits[0].arms), {"vector", "bm25"})

    def test_recall_is_safe_with_fts_syntax_in_query(self):
        self.kb.add_lesson(self.run_id, "topic-a", "analysis", "quoted text matters")
        for q in ['"unbalanced', "NEAR(a b) OR * AND", "text) OR (1=1", "memory_fts MATCH '*'", ""]:
            with self.subTest(q=q):
                recall(self.kb, q)  # must not raise

    def test_hash_embedding_similarity(self):
        a = embed("reduce peak memory of the distance matrix")
        self.assertAlmostEqual(cosine(a, a), 1.0, places=5)
        self.assertGreater(cosine(a, embed("lower peak memory for the distance matrix")),
                           cosine(a, embed("literature about protein folding")))

    def test_duplicate_detection(self):
        h1 = self.kb.add_hypothesis(self.run_id, 1, hyp("Stream rows and keep a running minimum to cut memory"))
        sim, sid = max_similarity(self.kb, "Stream rows and keep a running minimum to cut memory", [h1])
        self.assertGreater(sim, 0.99)
        self.assertEqual(sid, h1)
        sim2, _ = max_similarity(self.kb, "Use a GPU kernel", [h1])
        self.assertLess(sim2, 0.5)

    def test_graph_whitelist_dedupe_and_cap(self):
        with self.assertRaises(graph.EdgeError):
            graph.link(self.kb, self.run_id, "a", "b", "proves")
        with self.assertRaises(graph.EdgeError):
            graph.link(self.kb, self.run_id, "a", "a", "supports")
        self.assertTrue(graph.link(self.kb, self.run_id, "a", "b", "supports", 7))
        self.assertFalse(graph.link(self.kb, self.run_id, "a", "b", "supports"))
        self.assertEqual(self.kb.edges(self.run_id)[0]["confidence"], 1.0)  # clamped
        for i in range(graph.MAX_LINKS_PER_NODE + 5):
            graph.link(self.kb, self.run_id, "a", f"n{i}", "related_to")
        self.assertEqual(self.kb.count_edges_from("a"), graph.MAX_LINKS_PER_NODE)

    def test_run_state_roundtrip(self):
        self.kb.save_run(self.run_id, state={"phase": "execute", "best": {"value": 1.5}}, budget={"llm_calls": 3})
        run = self.kb.get_run(self.run_id)
        self.assertEqual(run["state"]["best"]["value"], 1.5)
        self.assertEqual(run["budget"]["llm_calls"], 3)

    def test_reopen_keeps_data(self):
        self.kb.add_lesson(self.run_id, "topic-a", "analysis", "persist me")
        self.kb.close()
        self.kb = KB(self.tmp / "kb.sqlite")
        self.assertEqual(self.kb.lessons(run_id=self.run_id)[0]["text"], "persist me")


if __name__ == "__main__":
    unittest.main()
