"""SQLite knowledge base: one file per workspace (``.horizons/kb.sqlite``).

Holds runs, papers, hypotheses, experiments, findings, finding edges, lessons,
LLM call audit, events, a literature response cache, and a ``memory`` table
indexed by FTS5 (BM25 arm) and hashed vectors (vector arm) for recall.

All SQL uses bound parameters. Schema changes are applied by version.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from horizons.kb import vectors

SCHEMA_VERSION = 1

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    topic_name TEXT NOT NULL,
    topic_path TEXT NOT NULL,
    goal TEXT NOT NULL,
    llm TEXT NOT NULL,
    status TEXT NOT NULL,          -- running | success | stopped | failed
    stage TEXT NOT NULL,
    stop_reason TEXT,
    state_json TEXT NOT NULL DEFAULT '{}',
    budget_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,           -- source:external_id
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,
    year INTEGER,
    venue TEXT,
    url TEXT,
    citations INTEGER,
    added_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS run_papers (
    run_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    query TEXT,
    PRIMARY KEY (run_id, paper_id)
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    statement TEXT NOT NULL,
    rationale TEXT,
    test_kind TEXT NOT NULL,
    falsification TEXT,
    expected_effect TEXT,
    citations_json TEXT NOT NULL DEFAULT '[]',
    alternatives_json TEXT NOT NULL DEFAULT '[]',
    readiness TEXT NOT NULL DEFAULT 'testable',   -- testable | needs_resources
    missing TEXT,
    status TEXT NOT NULL DEFAULT 'proposed',       -- proposed | selected | supported | refuted | errored | inconclusive | needs_resources | skipped
    parent_id TEXT,
    lineage_root TEXT,
    score REAL,
    outcome_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    hypothesis_id TEXT,
    purpose TEXT NOT NULL,          -- baseline | candidate | replication | data | literature
    attempt INTEGER NOT NULL DEFAULT 0,
    code_hash TEXT,
    parent_experiment_id TEXT,
    artifacts_dir TEXT,
    status TEXT NOT NULL,           -- ok | errored | invalid | timeout
    metrics_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    duration_s REAL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,             -- boundary_map | result | evidence | breakthrough
    hypothesis_id TEXT,
    experiment_id TEXT,
    summary TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL NOT NULL,
    rationale TEXT,
    created_at REAL NOT NULL,
    UNIQUE (src, dst, type)
);
CREATE TABLE IF NOT EXISTS lessons (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    hypothesis_id TEXT,
    category TEXT NOT NULL,         -- system | experiment | literature | analysis
    outcome TEXT,
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,             -- lesson | hypothesis | finding | paper
    ref_id TEXT NOT NULL,
    run_id TEXT,
    topic_name TEXT,
    text TEXT NOT NULL,
    vec BLOB NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (kind, ref_id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(text, content='memory', content_rowid='id');
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    task TEXT NOT NULL,
    prompt_chars INTEGER NOT NULL,
    output TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS lit_cache (
    key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_hyp_run ON hypotheses(run_id);
CREATE INDEX IF NOT EXISTS ix_exp_run ON experiments(run_id, hypothesis_id);
CREATE INDEX IF NOT EXISTS ix_find_run ON findings(run_id);
CREATE INDEX IF NOT EXISTS ix_less_topic ON lessons(topic_name);
CREATE INDEX IF NOT EXISTS ix_mem_kind ON memory(kind, topic_name);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(4)}"


def _j(v: Any) -> str:
    return json.dumps(v, default=str)


class KB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        self.conn.close()

    def _migrate(self) -> None:
        v = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if v > SCHEMA_VERSION:
            raise RuntimeError(f"knowledge base {self.path} is from a newer version (schema {v})")
        if v < 1:
            with self.conn:
                self.conn.executescript(_SCHEMA_V1)
                self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # -- generic -------------------------------------------------------------
    def _insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self.conn:
            self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", tuple(row.values()))

    def _update(self, table: str, key: str, fields: dict[str, Any]) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.conn:
            self.conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", (*fields.values(), key))

    def _rows(self, sql: str, args: Iterable[Any] = ()) -> list[dict[str, Any]]:
        out = []
        for r in self.conn.execute(sql, tuple(args)).fetchall():
            d = dict(r)
            for k in list(d):
                if k.endswith("_json") and isinstance(d[k], str):
                    d[k[:-5]] = json.loads(d.pop(k))
            out.append(d)
        return out

    def _row(self, sql: str, args: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self._rows(sql, args)
        return rows[0] if rows else None

    # -- runs ------------------------------------------------------------------
    def create_run(self, topic_name: str, topic_path: str, goal: str, llm: str) -> str:
        rid = new_id("run")
        now = time.time()
        self._insert("runs", dict(id=rid, topic_name=topic_name, topic_path=topic_path, goal=goal, llm=llm,
                                  status="running", stage="bootstrap", created_at=now, updated_at=now))
        return rid

    def save_run(self, run_id: str, *, status: str | None = None, stage: str | None = None,
                 state: dict | None = None, budget: dict | None = None, stop_reason: str | None = None) -> None:
        f: dict[str, Any] = {"updated_at": time.time()}
        if status is not None:
            f["status"] = status
        if stage is not None:
            f["stage"] = stage
        if state is not None:
            f["state_json"] = _j(state)
        if budget is not None:
            f["budget_json"] = _j(budget)
        if stop_reason is not None:
            f["stop_reason"] = stop_reason
        self._update("runs", run_id, f)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM runs WHERE id = ?", (run_id,))

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))

    def event(self, run_id: str, stage: str, message: str) -> None:
        self._insert("events", dict(run_id=run_id, stage=stage, message=message, created_at=time.time()))

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM events WHERE run_id = ? ORDER BY id", (run_id,))

    def log_llm(self, run_id: str | None, task: str, prompt_chars: int, output: str) -> None:
        self._insert("llm_calls", dict(run_id=run_id, task=task, prompt_chars=prompt_chars,
                                       output=output[:20000], created_at=time.time()))

    # -- papers ----------------------------------------------------------------
    def add_paper(self, run_id: str, paper: dict[str, Any], query: str = "") -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO papers (id, source, title, abstract, authors, year, venue, url, citations, added_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (paper["id"], paper["source"], paper["title"], paper.get("abstract") or "",
                 ", ".join(paper.get("authors") or [])[:1000], paper.get("year"), paper.get("venue") or "",
                 paper.get("url") or "", paper.get("citations"), time.time()),
            )
            self.conn.execute("INSERT OR IGNORE INTO run_papers (run_id, paper_id, query) VALUES (?, ?, ?)",
                              (run_id, paper["id"], query))
        self.remember("paper", paper["id"], f"{paper['title']}. {paper.get('abstract') or ''}", run_id, None)

    def run_papers(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT p.* FROM papers p JOIN run_papers r ON r.paper_id = p.id WHERE r.run_id = ?"
                          " ORDER BY COALESCE(p.citations, 0) DESC", (run_id,))

    def paper_ids(self, run_id: str) -> set[str]:
        return {r["paper_id"] for r in self._rows("SELECT paper_id FROM run_papers WHERE run_id = ?", (run_id,))}

    # -- hypotheses --------------------------------------------------------------
    def add_hypothesis(self, run_id: str, iteration: int, h: dict[str, Any], parent_id: str | None = None,
                       lineage_root: str | None = None) -> str:
        hid = new_id("H")
        self._insert("hypotheses", dict(
            id=hid, run_id=run_id, iteration=iteration, statement=h["statement"], rationale=h.get("rationale", ""),
            test_kind=h["test_kind"], falsification=h.get("falsification", ""),
            expected_effect=h.get("expected_effect", ""), citations_json=_j(h.get("citations", [])),
            alternatives_json=_j(h.get("alternatives", [])), readiness=h.get("readiness", "testable"),
            missing=h.get("missing"), status=h.get("status", "proposed"), parent_id=parent_id,
            lineage_root=lineage_root or hid, score=h.get("score"), created_at=time.time()))
        run = self.conn.execute("SELECT topic_name FROM runs WHERE id = ?", (run_id,)).fetchone()
        self.remember("hypothesis", hid, h["statement"], run_id, run["topic_name"] if run else None)
        return hid

    def update_hypothesis(self, hid: str, **fields: Any) -> None:
        if "outcome" in fields:
            fields["outcome_json"] = _j(fields.pop("outcome"))
        self._update("hypotheses", hid, fields)

    def get_hypothesis(self, hid: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM hypotheses WHERE id = ?", (hid,))

    def hypotheses(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM hypotheses WHERE run_id = ? ORDER BY created_at", (run_id,))

    # -- experiments -------------------------------------------------------------
    def add_experiment(self, run_id: str, purpose: str, status: str, metrics: dict[str, Any], *,
                       hypothesis_id: str | None = None, attempt: int = 0, code_hash: str | None = None,
                       parent_experiment_id: str | None = None, artifacts_dir: str | None = None,
                       error: str | None = None, duration_s: float | None = None) -> str:
        eid = new_id("E")
        self._insert("experiments", dict(
            id=eid, run_id=run_id, hypothesis_id=hypothesis_id, purpose=purpose, attempt=attempt,
            code_hash=code_hash, parent_experiment_id=parent_experiment_id, artifacts_dir=artifacts_dir,
            status=status, metrics_json=_j(metrics), error=(error or "")[:4000] or None, duration_s=duration_s,
            created_at=time.time()))
        return eid

    def experiments(self, run_id: str, hypothesis_id: str | None = None) -> list[dict[str, Any]]:
        if hypothesis_id:
            return self._rows("SELECT * FROM experiments WHERE run_id = ? AND hypothesis_id = ? ORDER BY created_at",
                              (run_id, hypothesis_id))
        return self._rows("SELECT * FROM experiments WHERE run_id = ? ORDER BY created_at", (run_id,))

    # -- findings / edges ----------------------------------------------------------
    def add_finding(self, run_id: str, kind: str, summary: str, data: dict[str, Any] | None = None, *,
                    hypothesis_id: str | None = None, experiment_id: str | None = None,
                    topic_name: str | None = None) -> str:
        fid = new_id("F")
        self._insert("findings", dict(id=fid, run_id=run_id, kind=kind, hypothesis_id=hypothesis_id,
                                      experiment_id=experiment_id, summary=summary, data_json=_j(data or {}),
                                      created_at=time.time()))
        self.remember("finding", fid, summary, run_id, topic_name)
        return fid

    def findings(self, run_id: str, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            return self._rows("SELECT * FROM findings WHERE run_id = ? AND kind = ? ORDER BY created_at",
                              (run_id, kind))
        return self._rows("SELECT * FROM findings WHERE run_id = ? ORDER BY created_at", (run_id,))

    def insert_edge(self, run_id: str, src: str, dst: str, type_: str, confidence: float, rationale: str) -> bool:
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO edges (run_id, src, dst, type, confidence, rationale, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)", (run_id, src, dst, type_, confidence, rationale, time.time()))
        return cur.rowcount == 1

    def edges(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM edges WHERE run_id = ? ORDER BY id", (run_id,))

    def count_edges_from(self, src: str) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM edges WHERE src = ?", (src,)).fetchone()[0]

    # -- lessons -------------------------------------------------------------------
    def add_lesson(self, run_id: str, topic_name: str, category: str, text: str, *,
                   hypothesis_id: str | None = None, outcome: str | None = None) -> str:
        lid = new_id("L")
        self._insert("lessons", dict(id=lid, run_id=run_id, topic_name=topic_name, hypothesis_id=hypothesis_id,
                                     category=category, outcome=outcome, text=text, created_at=time.time()))
        self.remember("lesson", lid, f"[{category}] {text}", run_id, topic_name)
        return lid

    def lessons(self, run_id: str | None = None, topic_name: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            return self._rows("SELECT * FROM lessons WHERE run_id = ? ORDER BY created_at", (run_id,))
        if topic_name:
            return self._rows("SELECT * FROM lessons WHERE topic_name = ? ORDER BY created_at", (topic_name,))
        return self._rows("SELECT * FROM lessons ORDER BY created_at")

    # -- memory index ----------------------------------------------------------------
    def remember(self, kind: str, ref_id: str, text: str, run_id: str | None, topic_name: str | None) -> None:
        text = text[:8000]
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO memory (kind, ref_id, run_id, topic_name, text, vec, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (kind, ref_id, run_id, topic_name, text, vectors.to_blob(vectors.embed(text)), time.time()))
            if cur.rowcount == 1:
                self.conn.execute("INSERT INTO memory_fts (rowid, text) VALUES (?, ?)", (cur.lastrowid, text))

    # -- literature cache --------------------------------------------------------------
    def cache_get(self, key: str, max_age_s: float) -> str | None:
        r = self.conn.execute("SELECT response, fetched_at FROM lit_cache WHERE key = ?", (key,)).fetchone()
        if r and time.time() - r["fetched_at"] <= max_age_s:
            return r["response"]
        return None

    def cache_put(self, key: str, response: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO lit_cache (key, response, fetched_at) VALUES (?, ?, ?)",
                              (key, response, time.time()))
