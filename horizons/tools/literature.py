"""Literature search over Semantic Scholar and arXiv (stdlib HTTP).

Adapted from openags/paper-search-mcp: explicit timeouts, 429/Retry-After
backoff with exponential fallback, arXiv's 1-request-per-3-seconds policy,
and conservative query building. Responses are cached in the KB so reruns
and resumes don't hammer the APIs. In offline mode only the cache and an
optional local fixture are used.

Returned text (titles/abstracts) is untrusted data: it is stored and shown,
never executed, and cannot change permissions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from horizons.kb.store import KB
from horizons.kb.vectors import tokens
from horizons.template import LiteratureCfg

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,citationCount,authors,url,venue,externalIds"
ARXIV_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "new-horizons/0.1 (research engine; stdlib urllib)"
CACHE_MAX_AGE_S = 7 * 24 * 3600
_ATOM = {"a": "http://www.w3.org/2005/Atom"}


class LiteratureError(RuntimeError):
    pass


def _clean(s: Any, cap: int = 4000) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:cap]


def _http_get(url: str, headers: dict[str, str] | None = None, timeout: float = 20, retries: int = 3,
              sleep: Callable[[float], None] = time.sleep) -> bytes:
    last = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"user-agent": USER_AGENT, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(5_000_000)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = float(ra) if ra and ra.isdigit() else 2.0 * 2 ** attempt
                sleep(min(wait, 60.0))
                continue
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"network error: {getattr(e, 'reason', e)}"
            if attempt < retries:
                sleep(2.0 * 2 ** attempt)
                continue
    raise LiteratureError(last)


def parse_s2(payload: dict) -> list[dict[str, Any]]:
    out = []
    for it in payload.get("data") or []:
        if not isinstance(it, dict) or not it.get("paperId") or not it.get("title"):
            continue
        ext = it.get("externalIds") or {}
        out.append({
            "id": f"s2:{it['paperId']}",
            "source": "semantic_scholar",
            "title": _clean(it["title"], 500),
            "abstract": _clean(it.get("abstract")),
            "authors": [_clean(a.get("name"), 100) for a in (it.get("authors") or [])[:10] if isinstance(a, dict)],
            "year": it.get("year") if isinstance(it.get("year"), int) else None,
            "venue": _clean(it.get("venue"), 200),
            "url": it.get("url") or (f"https://doi.org/{ext['DOI']}" if ext.get("DOI") else ""),
            "citations": it.get("citationCount") if isinstance(it.get("citationCount"), int) else None,
        })
    return out


def parse_arxiv(xml_bytes: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_bytes)  # Atom from a fixed host; no DTDs/entities are expanded by ElementTree
    out = []
    for e in root.findall("a:entry", _ATOM):
        raw_id = (e.findtext("a:id", "", _ATOM) or "").strip()
        m = re.search(r"arxiv\.org/abs/([^\s/]+(?:/[^\s/]+)?)$", raw_id)
        title = _clean(e.findtext("a:title", "", _ATOM), 500)
        if not m or not title:
            continue
        aid = re.sub(r"v\d+$", "", m.group(1))
        pub = e.findtext("a:published", "", _ATOM) or ""
        out.append({
            "id": f"arxiv:{aid}",
            "source": "arxiv",
            "title": title,
            "abstract": _clean(e.findtext("a:summary", "", _ATOM)),
            "authors": [_clean(a.findtext("a:name", "", _ATOM), 100) for a in e.findall("a:author", _ATOM)[:10]],
            "year": int(pub[:4]) if pub[:4].isdigit() else None,
            "venue": "arXiv",
            "url": f"https://arxiv.org/abs/{aid}",
            "citations": None,
        })
    return out


def arxiv_query(q: str) -> str:
    """AND of plain terms in all fields; strips arXiv operators/quotes from model-written text."""
    toks = [t for t in tokens(q) if t not in ("and", "or", "andnot")][:8]
    return " AND ".join(f"all:{t}" for t in toks) if toks else "all:research"


@dataclass
class LiteratureTool:
    kb: KB
    cfg: LiteratureCfg
    offline: bool = False
    fixture: list[dict[str, Any]] = field(default_factory=list)
    log: Callable[[str], None] = lambda msg: None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_arxiv: float = 0.0

    @staticmethod
    def load_fixture(path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        papers = []
        for p in data if isinstance(data, list) else []:
            if isinstance(p, dict) and p.get("id") and p.get("title"):
                papers.append({"id": str(p["id"]), "source": str(p.get("source", "fixture")),
                               "title": _clean(p["title"], 500), "abstract": _clean(p.get("abstract")),
                               "authors": [str(a) for a in p.get("authors", [])][:10], "year": p.get("year"),
                               "venue": _clean(p.get("venue"), 200), "url": str(p.get("url", "")),
                               "citations": p.get("citations")})
        return papers

    def _cached(self, key: str, fetch: Callable[[], str]) -> str | None:
        k = hashlib.sha256(key.encode()).hexdigest()
        hit = self.kb.cache_get(k, CACHE_MAX_AGE_S)
        if hit is not None or self.offline:
            return hit
        body = fetch()
        self.kb.cache_put(k, body)
        return body

    def _s2(self, query: str, limit: int) -> list[dict[str, Any]]:
        url = S2_URL + "?" + urllib.parse.urlencode({"query": query[:300], "limit": limit, "fields": S2_FIELDS})
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        body = self._cached("s2|" + url, lambda: _http_get(url, {"x-api-key": key} if key else None).decode("utf-8"))
        return parse_s2(json.loads(body)) if body else []

    def _arxiv(self, query: str, limit: int) -> list[dict[str, Any]]:
        url = ARXIV_URL + "?" + urllib.parse.urlencode(
            {"search_query": arxiv_query(query), "max_results": limit, "sortBy": "relevance"})

        def fetch() -> str:
            with self._lock:  # arXiv TOU: at most one request every 3 seconds
                wait = 3.0 - (time.monotonic() - self._last_arxiv)
                if wait > 0:
                    time.sleep(wait)
                try:
                    return _http_get(url).decode("utf-8")
                finally:
                    self._last_arxiv = time.monotonic()

        body = self._cached("arxiv|" + url, fetch)
        return parse_arxiv(body.encode("utf-8")) if body else []

    def _fixture_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        q = set(tokens(query))
        scored = [(len(q & set(tokens(p["title"] + " " + p["abstract"]))), p) for p in self.fixture]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: -s[0])
        return [p for _, p in scored[:limit]]

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search every configured source; errors on one source don't stop the others."""
        limit = max(1, min(limit, self.cfg.max_papers))
        results: list[dict[str, Any]] = []
        if self.fixture:
            results.extend(self._fixture_search(query, limit))
        for src in self.cfg.sources:
            try:
                results.extend(self._s2(query, limit) if src == "semantic_scholar" else self._arxiv(query, limit))
            except (LiteratureError, ValueError, ET.ParseError) as e:
                self.log(f"{src} search failed for {query[:80]!r}: {e}")
        seen, out = set(), []
        for p in results:
            key = re.sub(r"\W+", "", p["title"].lower())
            if p["id"] in seen or key in seen:
                continue
            seen.update({p["id"], key})
            out.append(p)
        return out
