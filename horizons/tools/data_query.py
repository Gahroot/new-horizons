"""Read-only SQL access to a user-supplied SQLite database.

Defence in depth: the file is opened with ``mode=ro`` (URI), an authorizer
allows only read operations, Python's sqlite3 refuses multiple statements
per ``execute``, a progress handler caps query work, and rows are capped.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from horizons.template import DataQueryCfg

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _ALLOWED_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
MAX_VM_STEPS = 50_000_000  # progress handler ticks every 1000 VM ops


class QueryError(ValueError):
    pass


def _authorizer(action: int, arg1: str | None, arg2: str | None, db: str | None, trigger: str | None) -> int:
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool


class DataQueryTool:
    def __init__(self, cfg: DataQueryCfg):
        self.cfg = cfg

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{quote(str(Path(self.cfg.database).resolve()))}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.set_authorizer(_authorizer)
        ticks = [0]

        def progress() -> int:
            ticks[0] += 1
            return 1 if ticks[0] * 1000 > MAX_VM_STEPS else 0

        conn.set_progress_handler(progress, 1000)
        return conn

    def schema(self) -> str:
        """CREATE statements for the model to plan queries against."""
        conn = sqlite3.connect(f"file:{quote(str(Path(self.cfg.database).resolve()))}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL"
                                " AND name NOT LIKE 'sqlite_%'").fetchall()
        finally:
            conn.close()
        return "\n".join(r[0] for r in rows)

    def query(self, sql: str) -> QueryResult:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise QueryError("empty query")
        if not sql.split(None, 1)[0].lower() in ("select", "with"):
            raise QueryError("only SELECT queries are allowed")
        conn = self._connect()
        try:
            cur = conn.execute(sql)
            rows = cur.fetchmany(self.cfg.max_rows + 1)
            cols = [d[0] for d in cur.description or []]
        except (sqlite3.DatabaseError, sqlite3.Warning, sqlite3.ProgrammingError) as e:
            raise QueryError(f"query rejected: {e}") from None
        finally:
            conn.close()
        truncated = len(rows) > self.cfg.max_rows
        return QueryResult(cols, [list(r) for r in rows[: self.cfg.max_rows]], truncated)
