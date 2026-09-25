"""Pluggable LLM clients: Anthropic Messages API, OpenAI-compatible, ChatGPT (Codex), scripted.

Stdlib only (urllib). API keys are read from the environment; subscription
tokens come from ``horizons.auth`` (``horizons login claude|chatgpt``). Neither
is ever logged, stored in the knowledge base, or put into exception messages.
Every call is charged to the budget guard *before* the request is sent.

JSON extraction follows AutoResearchClaw's 3-tier parser (direct → fenced
block → first balanced object), using ``raw_decode`` so braces inside
strings don't confuse it.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Iterator, Protocol

from horizons.auth import AuthError, AuthStore, Credentials, TokenProvider, claude_cli_user_agent
from horizons.budget import BudgetGuard

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_CHATGPT_MODEL = "gpt-6-sol"
USER_AGENT = "new-horizons/0.1 (+local research engine)"
# Subscription (OAuth) requests must identify as Claude Code; same as ezcoder.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
CLAUDE_OAUTH_BETAS = "claude-code-20250219,oauth-2025-04-20"
CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
# The ChatGPT backend gates newer models on a minimum Codex client version.
CODEX_CLIENT_VERSION = "0.155.1"
_RETRY_STATUSES = (429, 500, 502, 503, 504, 529)


class LLMError(RuntimeError):
    pass


class HTTPStatusError(LLMError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _retry_wait(e: urllib.error.HTTPError, attempt: int) -> float:
    ra = e.headers.get("retry-after") if e.headers else None
    try:
        return min(60.0, float(ra)) if ra else 2.0 * 2 ** attempt
    except ValueError:
        return 2.0 * 2 ** attempt


class LLMClient(Protocol):
    name: str
    model: str

    def complete(self, task: str, system: str, user: str) -> str: ...


# ----------------------------------------------------------------------------
# JSON extraction


def extract_json(text: str) -> Any:
    """Return the first JSON object/array found in ``text``; raise LLMError otherwise."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(text, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise LLMError(f"no JSON found in model output ({len(text)} chars)")


def extract_code(text: str) -> str:
    """Return the first ```python fenced block, else the whole text."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip() + "\n"


# ----------------------------------------------------------------------------
# HTTP


def _post_json(url: str, headers: dict[str, str], body: dict, timeout: float, retries: int = 3) -> dict:
    data = json.dumps(body).encode()
    last = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json",
                                              "user-agent": USER_AGENT, **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Log status + a short body excerpt, never request headers (they hold the key).
            with e:
                try:
                    excerpt = e.read(500).decode("utf-8", "replace")
                except OSError:
                    excerpt = ""
            last = f"HTTP {e.code}: {excerpt}"
            if e.code in _RETRY_STATUSES and attempt < retries:
                time.sleep(_retry_wait(e, attempt))
                continue
            raise HTTPStatusError(e.code, last) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"network error: {getattr(e, 'reason', e)}"
            if attempt < retries:
                time.sleep(2.0 * 2 ** attempt)
                continue
            raise LLMError(last) from None
    raise LLMError(last)


@dataclass
class AnthropicClient:
    """Anthropic Messages API with either ANTHROPIC_API_KEY or a Claude subscription login."""

    model: str = DEFAULT_ANTHROPIC_MODEL
    max_tokens: int = 4096
    base_url: str = "https://api.anthropic.com"
    timeout: float = 300
    name: str = "anthropic"
    oauth: TokenProvider | None = None
    _user_agent: str | None = field(default=None, repr=False)

    def _oauth_post(self, url: str, body: dict) -> dict:
        assert self.oauth is not None
        if self._user_agent is None:
            self._user_agent = claude_cli_user_agent()
        for forced in (False, True):  # one retry with a freshly refreshed token after a 401
            creds = self._token(forced)
            headers = {"authorization": f"Bearer {creds.access_token}", "anthropic-version": "2023-06-01",
                       "anthropic-beta": CLAUDE_OAUTH_BETAS, "user-agent": self._user_agent, "x-app": "cli"}
            try:
                return _post_json(url, headers, body, self.timeout)
            except HTTPStatusError as e:
                if e.status != 401 or forced:
                    raise
        raise AssertionError("unreachable")

    def _token(self, force: bool) -> Credentials:
        return _oauth_token(self.oauth, force)

    def complete(self, task: str, system: str, user: str) -> str:
        url = self.base_url.rstrip("/") + "/v1/messages"
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": user}]}
        if self.oauth is not None:
            body["system"] = [{"type": "text", "text": CLAUDE_CODE_IDENTITY}, {"type": "text", "text": system}]
            resp = self._oauth_post(url, body)
        else:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("ANTHROPIC_API_KEY is not set (or run `horizons login claude`)")
            body["system"] = system
            resp = _post_json(url, {"x-api-key": key, "anthropic-version": "2023-06-01"}, body, self.timeout)
        parts = [b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text"]
        if not parts:
            raise LLMError(f"empty response (stop_reason={resp.get('stop_reason')})")
        return "".join(parts)


@dataclass
class OpenAIClient:
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, Ollama, vLLM, ...)."""

    model: str = DEFAULT_OPENAI_MODEL
    max_tokens: int = 4096
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 300
    name: str = "openai"

    def complete(self, task: str, system: str, user: str) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        local = self.base_url.startswith(("http://localhost", "http://127.0.0.1"))
        if not key and not local:
            raise LLMError("OPENAI_API_KEY is not set")
        body: dict[str, Any] = {"model": self.model,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]}
        # api.openai.com's newer models only accept max_completion_tokens.
        body["max_completion_tokens" if "api.openai.com" in self.base_url else "max_tokens"] = self.max_tokens
        headers = {"authorization": f"Bearer {key}"} if key else {}
        resp = _post_json(self.base_url.rstrip("/") + "/chat/completions", headers, body, self.timeout)
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("malformed chat completion response") from None
        if not content:
            raise LLMError("empty response")
        return content


def _oauth_token(provider: TokenProvider | None, force: bool) -> Credentials:
    if provider is None:
        raise LLMError("no subscription login configured")
    try:
        return provider.get(force_refresh=force)
    except AuthError as e:
        raise LLMError(str(e)) from None


def iter_sse(stream: IO[bytes]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a text/event-stream body (multi-line data fields joined)."""
    data: list[str] = []
    for raw in stream:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            payload = "\n".join(data).strip()
            data = []
            if payload and payload != "[DONE]":
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    if data:
        try:
            obj = json.loads("\n".join(data))
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            pass


def _codex_usage_limit(text: str) -> str | None:
    """A friendly message if a 429 body is a ChatGPT plan usage-window stop, else None."""
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return None
    err = d.get("error") if isinstance(d, dict) and isinstance(d.get("error"), dict) else d
    if not isinstance(err, dict):
        return None
    code = str(err.get("code") or err.get("type") or "")
    if not re.search(r"usage_limit_reached|usage_not_included", code):
        return None
    resets = err.get("resets_at")
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(resets)) if isinstance(resets, (int, float)) else None
    return "ChatGPT plan usage limit reached" + (f"; it resets at {when}" if when else "")


def codex_text(events: Iterator[dict[str, Any]]) -> str:
    """Visible assistant text from a Codex Responses event stream.

    Text is taken only from items positively identified as assistant messages
    (never reasoning), preferring each item's final ``output_item.done`` copy.
    """
    item_types: dict[str, str] = {}
    deltas: dict[str, str] = {}
    order: list[str] = []
    final: dict[str, str] = {}
    for ev in events:
        t = ev.get("type")
        if t == "error" or t == "response.failed":
            err = ev.get("error") if isinstance(ev.get("error"), dict) else ev
            if isinstance(ev.get("response"), dict) and isinstance(ev["response"].get("error"), dict):
                err = ev["response"]["error"]
            msg = str(err.get("message") or err.get("code") or "Codex stream error")[:300]
            limit = _codex_usage_limit(json.dumps(err))
            raise LLMError(limit or f"ChatGPT error: {msg}")
        item = ev.get("item") if isinstance(ev.get("item"), dict) else None
        if t == "response.output_item.added" and item and item.get("id"):
            item_types[item["id"]] = str(item.get("type"))
            if item["id"] not in order:
                order.append(item["id"])
        elif t == "response.output_text.delta" and isinstance(ev.get("item_id"), str):
            iid = ev["item_id"]
            deltas[iid] = deltas.get(iid, "") + str(ev.get("delta") or "")
            if iid not in order:
                order.append(iid)
        elif t == "response.output_item.done" and item and item.get("type") == "message":
            iid = str(item.get("id") or f"_anon{len(order)}")
            item_types[iid] = "message"
            if iid not in order:
                order.append(iid)
            final[iid] = "".join(str(c.get("text") or "") for c in item.get("content") or []
                                 if isinstance(c, dict) and c.get("type") == "output_text")
    return "".join(final.get(i, deltas.get(i, "")) for i in order if item_types.get(i) == "message")


@dataclass
class CodexClient:
    """ChatGPT subscription models through the Codex Responses endpoint (``horizons login chatgpt``)."""

    oauth: TokenProvider | None = None
    model: str = DEFAULT_CHATGPT_MODEL
    url: str = CODEX_URL
    reasoning_effort: str = "medium"
    timeout: float = 600
    retries: int = 3
    name: str = "chatgpt"
    session_id: str = field(default_factory=lambda: secrets.token_hex(16))

    def _responses_lite(self) -> bool:
        return self.model.startswith(("gpt-5.6-", "gpt-6-"))

    def _body(self, system: str, user: str) -> dict[str, Any]:
        lite = self._responses_lite()
        return {
            "model": self.model, "store": False, "stream": True, "instructions": system,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": user}]}],
            "tool_choice": "auto", "parallel_tool_calls": False,
            "reasoning": {"effort": self.reasoning_effort, "summary": "auto",
                          **({"context": "all_turns"} if lite else {})},
            "prompt_cache_key": "new-horizons",
        }

    def _headers(self, creds: Credentials) -> dict[str, str]:
        h = {"content-type": "application/json", "accept": "text/event-stream",
             "authorization": f"Bearer {creds.access_token}", "openai-beta": "responses=experimental",
             "originator": "codex_cli_rs", "user-agent": f"codex_cli_rs/{CODEX_CLIENT_VERSION}",
             "session_id": self.session_id}
        if creds.account_id:
            h["chatgpt-account-id"] = creds.account_id
        if self._responses_lite():
            h["version"] = CODEX_CLIENT_VERSION
            h["x-openai-internal-codex-responses-lite"] = "true"
        return h

    def complete(self, task: str, system: str, user: str) -> str:
        data = json.dumps(self._body(system, user)).encode()
        forced = False
        attempt = 0
        while True:
            creds = _oauth_token(self.oauth, forced)
            req = urllib.request.Request(self.url, data=data, method="POST", headers=self._headers(creds))
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    text = codex_text(iter_sse(r))
                if not text:
                    raise LLMError("empty response from ChatGPT")
                return text
            except urllib.error.HTTPError as e:
                with e:
                    try:
                        body = e.read(2000).decode("utf-8", "replace")
                    except OSError:
                        body = ""
                if e.code == 401 and not forced:
                    forced = True
                    continue
                if e.code == 429 and (limit := _codex_usage_limit(body)):
                    raise LLMError(limit) from None
                if e.code in _RETRY_STATUSES and attempt < self.retries:
                    time.sleep(_retry_wait(e, attempt))
                    attempt += 1
                    continue
                hint = ""
                if e.code in (400, 404) and "model" in body.lower():
                    hint = f" (is {self.model!r} available on your ChatGPT plan? try --model gpt-6-luna)"
                raise HTTPStatusError(e.code, f"ChatGPT HTTP {e.code}: {body[:300]}{hint}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.retries:
                    time.sleep(2.0 * 2 ** attempt)
                    attempt += 1
                    continue
                raise LLMError(f"network error: {getattr(e, 'reason', e)}") from None


@dataclass
class ScriptedClient:
    """Deterministic offline client.

    ``script`` maps a task name to a list of responses consumed in order; the
    last response repeats once the list is exhausted. A response may be a
    string or any JSON value (serialised before returning). Used by the
    offline demo and tests - it proves the plumbing, not research quality.
    """

    script: dict[str, list[Any]]
    model: str = "scripted"
    name: str = "scripted"
    calls: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "ScriptedClient":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise LLMError("scripted LLM file must be a JSON object of task -> [responses]")
        return cls({k: (v if isinstance(v, list) else [v]) for k, v in data.items() if not k.startswith("_")})

    def complete(self, task: str, system: str, user: str) -> str:
        responses = self.script.get(task)
        if not responses:
            raise LLMError(f"scripted LLM has no response for task {task!r}")
        i = self.calls.get(task, 0)
        self.calls[task] = i + 1
        r = responses[min(i, len(responses) - 1)]
        return r if isinstance(r, str) else json.dumps(r)


# ----------------------------------------------------------------------------


@dataclass
class LLM:
    """Budgeted, logged wrapper around a client."""

    client: LLMClient
    budget: BudgetGuard
    log: Callable[[str, str, str, str], None] | None = None  # (task, system, user, output)

    @property
    def label(self) -> str:
        return f"{self.client.name}:{self.client.model}"

    def text(self, task: str, system: str, user: str) -> str:
        self.budget.charge_llm()
        out = self.client.complete(task, system, user)
        if self.log:
            self.log(task, system, user, out)
        return out

    def json(self, task: str, system: str, user: str, retries: int = 1) -> Any:
        prompt = user
        for attempt in range(retries + 1):
            out = self.text(task, system, prompt)
            try:
                return extract_json(out)
            except LLMError:
                if attempt == retries:
                    raise
                prompt = user + "\n\nYour previous reply was not valid JSON. Reply with JSON only."
        raise AssertionError("unreachable")


def make_client(provider: str | None, model: str | None = None, base_url: str | None = None,
                max_tokens: int = 4096, scripted_file: Path | None = None,
                auth_store: AuthStore | None = None) -> LLMClient:
    """Pick a client.

    ``provider=None`` auto-detects: a subscription login (Claude, then ChatGPT)
    wins over API keys, as in ezcoder, then ANTHROPIC_API_KEY, then OpenAI.
    """
    model = model or os.environ.get("HORIZONS_MODEL") or None
    env_openai_url = os.environ.get("HORIZONS_OPENAI_BASE_URL")
    if env_openai_url and not env_openai_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise LLMError("HORIZONS_OPENAI_BASE_URL must be https:// (plain http only for localhost)")
    store = auth_store or AuthStore()
    logged_in: list[str] = []
    if provider in (None, "claude", "chatgpt"):
        try:
            logged_in = store.providers()
        except AuthError as e:
            raise LLMError(str(e)) from None
    if provider is None:
        if "claude" in logged_in:
            provider = "claude"
        elif "chatgpt" in logged_in:
            provider = "chatgpt"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY") or env_openai_url:
            provider = "openai"
        else:
            raise LLMError("no LLM configured: run `horizons login claude` or `horizons login chatgpt` "
                           "(subscription), or set ANTHROPIC_API_KEY / OPENAI_API_KEY, "
                           "or use --llm scripted for the offline demo")
    if provider in ("claude", "chatgpt") and provider not in logged_in:
        raise LLMError(f"not logged in to {provider}; run `horizons login {provider}`")
    if provider == "claude":
        return AnthropicClient(model=model or DEFAULT_ANTHROPIC_MODEL, max_tokens=max_tokens, name="claude",
                               oauth=TokenProvider("claude", store))
    if provider == "chatgpt":
        return CodexClient(oauth=TokenProvider("chatgpt", store), model=model or DEFAULT_CHATGPT_MODEL)
    if provider == "anthropic":
        return AnthropicClient(model=model or DEFAULT_ANTHROPIC_MODEL, max_tokens=max_tokens,
                               **({"base_url": base_url} if base_url else {}))
    if provider == "openai":
        url = base_url or env_openai_url
        return OpenAIClient(model=model or DEFAULT_OPENAI_MODEL, max_tokens=max_tokens,
                            **({"base_url": url} if url else {}))
    if provider == "scripted":
        if scripted_file is None:
            raise LLMError("scripted LLM needs [llm] scripted_file in the topic template")
        return ScriptedClient.from_file(scripted_file)
    raise LLMError(f"unknown LLM provider {provider!r}")
