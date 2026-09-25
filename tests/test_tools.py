import json
import sqlite3
import unittest

from horizons.kb.store import KB
from horizons.template import DataQueryCfg, LiteratureCfg, Runtime, load_topic
from horizons.tools.data_query import DataQueryTool, QueryError
from horizons.tools.literature import LiteratureTool, arxiv_query, parse_arxiv, parse_s2
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
