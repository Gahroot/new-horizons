import json
import os
import sqlite3
import unittest
from unittest import mock

from horizons.kb.store import KB
from horizons.template import DataQueryCfg, LiteratureCfg, Runtime, load_topic
from horizons.tools.data_query import DataQueryTool, QueryError
from horizons.tools import literature as litmod
from horizons.tools.literature import (LiteratureError, LiteratureTool, arxiv_query, openalex_abstract,
                                       parse_arxiv, parse_openalex, parse_s2)
from horizons.tools.registry import ToolNotAllowed, ToolRegistry
from tests.helpers import EXAMPLES, TempDirCase

ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <published>2024-01-03T00:00:00Z</published>
    <title>  Linear   attention at scale </title>
    <summary>We study linear attention.</summary>
    <author><name>A. Person</name></author>
  </entry>
  <entry><id>garbage</id><title>skip me</title></entry>
</feed>"""


class RegistryTests(unittest.TestCase):
    def test_disallowed_tool_refused(self):
        spec = load_topic(EXAMPLES / "literature_question" / "topic.toml")
        reg = ToolRegistry(spec, {"literature": lambda: "lit", "python_sandbox": lambda: "sandbox"})
        self.assertEqual(reg.get("literature"), "lit")
        with self.assertRaises(ToolNotAllowed):
            reg.get("python_sandbox")
        with self.assertRaises(ToolNotAllowed):
            reg.get("shell")


class LiteratureParsingTests(TempDirCase):
    def test_parse_s2(self):
        payload = {"data": [{"paperId": "abc", "title": "T", "abstract": "A  b", "year": 2020, "citationCount": 3,
                             "authors": [{"name": "X"}], "externalIds": {"DOI": "10.1/x"}, "url": None},
                            {"paperId": None, "title": "no id"}]}
        papers = parse_s2(payload)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["id"], "s2:abc")
        self.assertEqual(papers[0]["abstract"], "A b")
        self.assertEqual(papers[0]["url"], "https://doi.org/10.1/x")

    def test_parse_arxiv(self):
        papers = parse_arxiv(ATOM)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["id"], "arxiv:2401.01234")
        self.assertEqual(papers[0]["title"], "Linear attention at scale")
        self.assertEqual(papers[0]["year"], 2024)

    def test_arxiv_query_strips_operators(self):
        q = arxiv_query('ti:"x" OR (sparse) ANDNOT attention')
        self.assertNotIn('"', q)
        self.assertNotIn("(", q)
        self.assertIn("all:sparse", q)

    def test_offline_uses_fixture_and_cache_only(self):
        kb = KB(self.tmp / "kb.sqlite")
        self.addCleanup(kb.close)
        fixture = LiteratureTool.load_fixture(EXAMPLES / "memory_reduction" / "papers.json")
        tool = LiteratureTool(kb, LiteratureCfg(max_papers=10), offline=True, fixture=fixture)
        found = tool.search("streaming nearest neighbour memory")
        self.assertTrue(found)
        self.assertEqual(found[0]["id"], "demo:streaming-nn")


EMPTY_ATOM = b"""<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


class LiteratureSourceTests(TempDirCase):
    """Online sources, with HTTP replaced by a fake so nothing leaves the machine."""

    def tool(self, *sources):
        kb = KB(self.tmp / "kb.sqlite")
        self.addCleanup(kb.close)
        logs: list[str] = []
        return LiteratureTool(kb, LiteratureCfg(sources=sources, max_papers=10), log=logs.append), logs

    def fake_http(self, handler):
        calls = []

        def get(url, headers=None, **kw):
            calls.append((url, headers or {}))
            return handler(url)

        patcher = mock.patch.object(litmod, "_http_get", side_effect=get)
        patcher.start()
        self.addCleanup(patcher.stop)
        sleep = mock.patch.object(litmod.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)
        return calls

    def test_refused_source_is_switched_off_with_one_clear_message(self):
        def handler(url):
            if "semanticscholar" in url:
                raise LiteratureError("HTTP 429")
            return ATOM

        calls = self.fake_http(handler)
        tool, logs = self.tool("semantic_scholar", "arxiv")
        self.assertEqual(len(tool.search("linear attention scale")), 1)
        self.assertEqual(len(tool.search("sparse attention long context")), 1)
        self.assertEqual(sum("semanticscholar" in u for u, _ in calls), 1)  # not retried on the second query
        self.assertEqual(len(logs), 1)
        self.assertIn("SEMANTIC_SCHOLAR_API_KEY", logs[0])
        self.assertIn("semantic_scholar: OFF (HTTP 429", tool.coverage())
        self.assertIn("arxiv: ok", tool.coverage())

    def test_transient_failure_does_not_switch_a_source_off(self):
        def handler(url):
            raise LiteratureError("network error: timed out")

        calls = self.fake_http(handler)
        tool, logs = self.tool("openalex")
        tool.search("first query here")
        tool.search("second query here")
        self.assertEqual(len(calls), 2)
        self.assertEqual(tool.coverage(), "openalex: 2/2 searches failed")

    def test_arxiv_relaxes_a_query_that_matches_nothing(self):
        calls = self.fake_http(lambda url: EMPTY_ATOM if url.count("AND") > 2 else ATOM)
        tool, _ = self.tool("arxiv")
        found = tool.search("sub one bit binary factorisation mixture experts quantization compression")
        self.assertEqual([p["id"] for p in found], ["arxiv:2401.01234"])
        self.assertEqual(len(calls), 3)  # 6 terms -> 4 terms -> 2 terms

    def test_openalex_key_goes_in_a_header_never_the_url(self):
        payload = {"results": [{"id": "https://openalex.org/W123", "display_name": "Binary  experts",
                                "publication_year": 2025, "abstract_inverted_index": {"Experts": [0], "compress": [1]},
                                "authorships": [{"author": {"display_name": "B. Author"}}],
                                "primary_location": {"source": {"display_name": "NeurIPS"}},
                                "doi": "https://doi.org/10.1/b", "cited_by_count": 4}]}
        calls = self.fake_http(lambda url: json.dumps(payload).encode())
        tool, _ = self.tool("openalex")
        with mock.patch.dict(os.environ, {"OPENALEX_API_KEY": "sekrit-key"}):
            found = tool.search("binary experts")
        self.assertEqual(found[0]["id"], "openalex:W123")
        self.assertEqual(found[0]["abstract"], "Experts compress")
        url, headers = calls[0]
        self.assertNotIn("sekrit", url)
        self.assertEqual(headers, {"authorization": "Bearer sekrit-key"})
        cache = tool.kb.conn.execute("SELECT group_concat(key || response) FROM lit_cache").fetchone()[0]
        self.assertNotIn("sekrit", cache)

    def test_parse_openalex_rejects_malformed_items(self):
        self.assertEqual(openalex_abstract({"b": [1], "a": [0], "bad": ["x"]}), "a b")
        self.assertEqual(openalex_abstract(None), "")
        papers = parse_openalex({"results": [{"id": "https://openalex.org/X1", "display_name": "bad id"},
                                             {"id": "https://openalex.org/W9", "display_name": "ok",
                                              "doi": "javascript:alert(1)"}]})
        self.assertEqual([p["id"] for p in papers], ["openalex:W9"])
        self.assertEqual(papers[0]["url"], "https://openalex.org/W9")


class DataQueryTests(TempDirCase):
    def setUp(self):
        super().setUp()
        db = self.tmp / "data.sqlite"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE obs (site TEXT, year INTEGER, value REAL)")
        con.executemany("INSERT INTO obs VALUES (?, ?, ?)", [("a", 2000 + i, i * 1.5) for i in range(50)])
        con.commit()
        con.close()
        self.db = db
        self.tool = DataQueryTool(DataQueryCfg(db, self.tmp / "e.py", max_rows=10, runtime=Runtime()))

    def test_select_with_row_cap(self):
        r = self.tool.query("SELECT site, year FROM obs ORDER BY year")
        self.assertEqual(r.columns, ["site", "year"])
        self.assertEqual(len(r.rows), 10)
        self.assertTrue(r.truncated)
        self.assertIn("CREATE TABLE obs", self.tool.schema())

    def test_writes_and_tricks_refused(self):
        for sql in ("DELETE FROM obs", "DROP TABLE obs", "UPDATE obs SET value = 0",
                    "INSERT INTO obs VALUES ('x', 1, 1)", "SELECT 1; DROP TABLE obs",
                    "WITH x AS (SELECT 1) DELETE FROM obs", "ATTACH DATABASE 'other.db' AS o",
                    "PRAGMA writable_schema = 1", "SELECT load_extension('evil')",
                    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"):
            with self.subTest(sql=sql), self.assertRaises(QueryError):
                self.tool.query(sql)
        con = sqlite3.connect(self.db)
        self.assertEqual(con.execute("SELECT count(*) FROM obs").fetchone()[0], 50)
        con.close()


if __name__ == "__main__":
    unittest.main()
