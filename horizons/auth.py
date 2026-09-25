"""Subscription logins: Claude (Pro/Max) and ChatGPT (Plus/Pro) via OAuth + PKCE.

Ported from ezcoder's ``packages/core/src/oauth/{anthropic,openai,pkce}.ts``:

- Claude: browser authorize on claude.ai, the user pastes ``code#state`` back,
  tokens come from platform.claude.com (console.anthropic.com as fallback).
- ChatGPT: browser authorize on auth.openai.com with a loopback callback on
  127.0.0.1:1455; a pasted callback URL works too (SSH / headless), and
  whichever arrives first wins. The ChatGPT account id is read from the token.

Tokens live in one JSON file outside any project (``~/.horizons/auth.json``,
override with ``HORIZONS_AUTH_FILE``), written atomically with mode 0600.
Tokens are never printed or logged; errors carry status codes and short
provider messages only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import re
import secrets
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable

PROVIDERS = ("claude", "chatgpt")
REFRESH_MARGIN_S = 300  # refresh when a token has less than 5 minutes left
HTTP_TIMEOUT_S = 30

# -- Claude (Anthropic) --------------------------------------------------------------
CLAUDE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
CLAUDE_TOKEN_URLS = ("https://platform.claude.com/v1/oauth/token", "https://console.anthropic.com/v1/oauth/token")
CLAUDE_REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
CLAUDE_SCOPES = ("org:create_api_key user:profile user:inference user:sessions:claude_code "
                 "user:mcp_servers user:file_upload")
# Anthropic's OAuth edge checks for a recent claude-cli user agent. Resolved from npm
# (cached a day); this is the offline fallback.
CLAUDE_CLI_FALLBACK_VERSION = "2.1.280"
CLAUDE_CLI_NPM_URL = "https://registry.npmjs.org/@anthropic-ai/claude-code/latest"

# -- ChatGPT (OpenAI Codex) ------------------------------------------------------------
CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CHATGPT_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CHATGPT_TOKEN_URL = "https://auth.openai.com/oauth/token"
CHATGPT_CALLBACK_PORT = 1455
CHATGPT_REDIRECT_URI = f"http://localhost:{CHATGPT_CALLBACK_PORT}/auth/callback"
CHATGPT_SCOPE = "openid profile email offline_access api.connectors.read api.connectors.invoke"
CHATGPT_JWT_CLAIM = "https://api.openai.com/auth"
CALLBACK_TIMEOUT_S = 300
MAX_PASTE_ATTEMPTS = 3
MIN_RAW_CODE_LENGTH = 10


class AuthError(RuntimeError):
    pass


class InvalidGrant(AuthError):
    """The provider rejected the refresh token (4xx): the user must log in again."""


@dataclass
class Credentials:
    access_token: str
    refresh_token: str
    expires_at: float  # unix seconds
    account_id: str | None = None

    def expires_soon(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) >= self.expires_at - REFRESH_MARGIN_S

    @classmethod
    def from_dict(cls, d: object) -> "Credentials":
        if not isinstance(d, dict):
            raise AuthError("stored credentials are malformed")
        try:
            return cls(str(d["access_token"]), str(d["refresh_token"]), float(d["expires_at"]),
                       str(d["account_id"]) if d.get("account_id") else None)
        except (KeyError, TypeError, ValueError):
            raise AuthError("stored credentials are malformed") from None


# ----------------------------------------------------------------------------------------
# storage


def default_auth_file() -> Path:
    env = os.environ.get("HORIZONS_AUTH_FILE")
    return Path(env).expanduser() if env else Path.home() / ".horizons" / "auth.json"


class AuthStore:
    """One JSON file mapping provider -> credentials. Private (0600) and replaced atomically."""

    def __init__(self, path: Path | None = None):
        self.path = (path or default_auth_file()).expanduser().resolve()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as e:
            raise AuthError(f"cannot read {self.path}: {e}") from None
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(prefix=".auth-", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def get(self, provider: str) -> Credentials | None:
        d = self._read().get(provider)
        return Credentials.from_dict(d) if d else None

    def put(self, provider: str, creds: Credentials) -> None:
        data = self._read()
        data[provider] = asdict(creds)
        self._write(data)

    def delete(self, provider: str) -> bool:
        data = self._read()
        if provider not in data:
            return False
        del data[provider]
        self._write(data)
        return True

    def providers(self) -> list[str]:
        return sorted(p for p in self._read() if p in PROVIDERS)


# ----------------------------------------------------------------------------------------
# helpers


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def pkce_pair() -> tuple[str, str]:
    """(verifier, S256 challenge) per RFC 7636."""
    verifier = _b64url(secrets.token_bytes(32))
    return verifier, _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def jwt_payload(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    try:
        data = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def chatgpt_account_id(access_token: str) -> str | None:
    claim = jwt_payload(access_token).get(CHATGPT_JWT_CLAIM)
    acct = claim.get("chatgpt_account_id") if isinstance(claim, dict) else None
    return acct if isinstance(acct, str) and acct else None


def _post(url: str, body: bytes, headers: dict[str, str]) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
            return r.status, r.read(1_000_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        with e:
            try:
                text = e.read(2000).decode("utf-8", "replace")
            except OSError:
                text = ""
        return e.code, text


def _error_summary(text: str) -> str:
    """Short provider error (e.g. 'invalid_grant: ...') without echoing arbitrary bodies."""
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return re.sub(r"\s+", " ", text)[:200]
    if isinstance(d, dict):
        err = d.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("type") or "")[:200]
        desc = d.get("error_description") or d.get("message") or ""
        return f"{err or ''} {desc}".strip()[:200]
    return ""


def _token_response(status: int, text: str, label: str) -> dict:
    if status >= 400:
        cls = InvalidGrant if 400 <= status < 500 else AuthError
        raise cls(f"{label} failed (HTTP {status}): {_error_summary(text)}")
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        raise AuthError(f"{label} returned non-JSON") from None
    if not isinstance(d, dict) or not d.get("access_token"):
        raise AuthError(f"{label} returned no access token")
    return d


def _creds(d: dict, previous_refresh: str | None = None) -> Credentials:
    try:
        expires_in = float(d.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600.0
    refresh = d.get("refresh_token") or previous_refresh
    if not refresh:
        raise AuthError("provider returned no refresh token")
    return Credentials(str(d["access_token"]), str(refresh), time.time() + expires_in)


# ----------------------------------------------------------------------------------------
# Claude


def claude_cli_user_agent(cache_dir: Path | None = None) -> str:
    """``claude-cli/<version> (external, cli)`` with the latest published Claude Code version."""
    override = os.environ.get("HORIZONS_CLAUDE_CLI_VERSION")
    if override and re.fullmatch(r"\d+\.\d+\.\d+", override):
        return f"claude-cli/{override} (external, cli)"
    cache = (cache_dir or default_auth_file().parent) / "claude-cli-version.json"
    version = None
    try:
        c = json.loads(cache.read_text(encoding="utf-8"))
        if time.time() - float(c.get("fetched_at", 0)) < 86400:
            version = c.get("version")
    except (OSError, ValueError, AttributeError):
        pass
    if not version:
        try:
            req = urllib.request.Request(CLAUDE_CLI_NPM_URL, headers={"accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as r:
                version = json.loads(r.read(200_000)).get("version")
            if isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version):
                cache.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                cache.write_text(json.dumps({"version": version, "fetched_at": time.time()}), encoding="utf-8")
        except (OSError, ValueError, AttributeError):
            version = None
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        version = CLAUDE_CLI_FALLBACK_VERSION
    return f"claude-cli/{version} (external, cli)"


def _claude_token_request(body: dict, label: str, token_urls: tuple[str, ...]) -> dict:
    headers = {"content-type": "application/json", "user-agent": claude_cli_user_agent(),
               "anthropic-beta": "oauth-2025-04-20"}
    data = json.dumps(body).encode()
    last: Exception | None = None
    for url in token_urls:
        try:
            status, text = _post(url, data, headers)
        except (urllib.error.URLError, OSError) as e:
            last = AuthError(f"{label}: network error ({getattr(e, 'reason', e)})")
            continue
        if status < 500:  # success or an authoritative 4xx; don't mask it with the fallback host
            return _token_response(status, text, f"Claude {label}")
        last = AuthError(f"Claude {label} failed (HTTP {status})")
    raise last or AuthError(f"Claude {label}: no token endpoint reachable")


def claude_authorize_url(challenge: str, state: str) -> str:
    q = urllib.parse.urlencode({"code": "true", "client_id": CLAUDE_CLIENT_ID, "response_type": "code",
                                "redirect_uri": CLAUDE_REDIRECT_URI, "scope": CLAUDE_SCOPES,
                                "code_challenge": challenge, "code_challenge_method": "S256", "state": state})
    return f"{CLAUDE_AUTHORIZE_URL}?{q}"


def parse_claude_paste(raw: str, expected_state: str) -> str:
    code, sep, state = raw.strip().partition("#")
    if not sep or not code or not secrets.compare_digest(state, expected_state):
        raise AuthError("that is not the code for this login (expected code#state). Run the login again.")
    return code


def login_claude(open_url: Callable[[str], None], prompt: Callable[[str], str],
                 token_urls: tuple[str, ...] = CLAUDE_TOKEN_URLS) -> Credentials:
    verifier, challenge = pkce_pair()
    state = secrets.token_hex(16)
    open_url(claude_authorize_url(challenge, state))
    code = parse_claude_paste(prompt("Paste the code shown after you approve (looks like code#state): "), state)
    d = _claude_token_request({"grant_type": "authorization_code", "client_id": CLAUDE_CLIENT_ID, "code": code,
                               "state": state, "redirect_uri": CLAUDE_REDIRECT_URI, "code_verifier": verifier},
                              "token exchange", token_urls)
    return _creds(d)


def refresh_claude(refresh_token: str, token_urls: tuple[str, ...] = CLAUDE_TOKEN_URLS) -> Credentials:
    d = _claude_token_request({"grant_type": "refresh_token", "client_id": CLAUDE_CLIENT_ID,
                               "refresh_token": refresh_token}, "token refresh", token_urls)
    return _creds(d, refresh_token)


# ----------------------------------------------------------------------------------------
# ChatGPT


def chatgpt_authorize_url(challenge: str, state: str, redirect_uri: str = CHATGPT_REDIRECT_URI) -> str:
    q = urllib.parse.urlencode({
        "response_type": "code", "client_id": CHATGPT_CLIENT_ID, "redirect_uri": redirect_uri,
        "scope": CHATGPT_SCOPE, "code_challenge": challenge, "code_challenge_method": "S256", "state": state,
        "prompt": "login", "id_token_add_organizations": "true", "codex_cli_simplified_flow": "true",
        "originator": "codex_cli_rs",
    })
    return f"{CHATGPT_AUTHORIZE_URL}?{q}"


def parse_chatgpt_paste(raw: str) -> tuple[str | None, str | None]:
    """Accept a full callback URL, ``code#state``, a ``code=..&state=..`` query, or a bare code."""
    value = raw.strip()
    if not value:
        return None, None
    if value.startswith(("http://", "https://")):
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(value).query)
        return (q.get("code") or [None])[0], (q.get("state") or [None])[0]
    if "#" in value:
        code, _, state = value.partition("#")
        return code or None, state or None
    if "code=" in value:
        q = urllib.parse.parse_qs(value.lstrip("?"))
        return (q.get("code") or [None])[0], (q.get("state") or [None])[0]
    if re.search(r"\s", value) or len(value) < MIN_RAW_CODE_LENGTH:
        return None, None
    return value, None


def _callback_server(expected_state: str, port: int, results: "queue.Queue[tuple[str, str]]") -> HTTPServer | None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            url = urllib.parse.urlsplit(self.path)
            q = urllib.parse.parse_qs(url.query)
            if url.path != "/auth/callback":
                self.send_error(404)
                return
            state, code = (q.get("state") or [""])[0], (q.get("code") or [""])[0]
            if not code or not secrets.compare_digest(state, expected_state):
                self.send_error(400, "State mismatch")
                return
            body = b"<html><body><h1>Login successful</h1><p>You can close this tab.</p></body></html>"
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            results.put(("code", code))

        def log_message(self, format: str, *args: object) -> None:  # silence access logs (they hold the code)
            pass

    try:
        server = HTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        return None  # port busy: the paste route still works
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
    return server


def login_chatgpt(open_url: Callable[[str], None], prompt: Callable[[str], str],
                  status: Callable[[str], None] = print, token_url: str = CHATGPT_TOKEN_URL,
                  port: int = CHATGPT_CALLBACK_PORT, timeout_s: float = CALLBACK_TIMEOUT_S) -> Credentials:
    verifier, challenge = pkce_pair()
    state = secrets.token_hex(16)
    redirect_uri = f"http://localhost:{port}/auth/callback"
    results: "queue.Queue[tuple[str, str]]" = queue.Queue()
    server = _callback_server(state, port, results)

    def paste_route() -> None:
        msg = "Or, if the browser is on another machine, paste the final URL here: "
        for _ in range(MAX_PASTE_ATTEMPTS):
            try:
                raw = prompt(msg)
            except (EOFError, OSError):
                break
            code, pasted_state = parse_chatgpt_paste(raw)
            if code and pasted_state and not secrets.compare_digest(pasted_state, state):
                results.put(("error", "state mismatch: that URL belongs to a different login. Run the login again."))
                return
            if code:
                results.put(("code", code))
                return
            msg = "That didn't contain an authorization code. Paste the full URL from the address bar: "
        results.put(("paste_failed", ""))

    try:
        open_url(chatgpt_authorize_url(challenge, state, redirect_uri))
        status("Waiting for the browser to finish the login...")
        threading.Thread(target=paste_route, daemon=True).start()
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AuthError("login timed out; run `horizons login chatgpt` again")
            try:
                kind, value = results.get(timeout=remaining)
            except queue.Empty:
                continue
            if kind == "code":
                code = value
                break
            if kind == "error":
                raise AuthError(value)
            # The paste route gave up; keep waiting only if the browser callback can still arrive.
            if server is None:
                raise AuthError("no authorization code received")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()

    body = urllib.parse.urlencode({"grant_type": "authorization_code", "client_id": CHATGPT_CLIENT_ID, "code": code,
                                   "redirect_uri": redirect_uri, "code_verifier": verifier}).encode()
    try:
        st, text = _post(token_url, body, {"content-type": "application/x-www-form-urlencoded"})
    except (urllib.error.URLError, OSError) as e:
        raise AuthError(f"ChatGPT token exchange: network error ({getattr(e, 'reason', e)})") from None
    creds = _creds(_token_response(st, text, "ChatGPT token exchange"))
    creds.account_id = chatgpt_account_id(creds.access_token)
    if not creds.account_id:
        raise AuthError("the ChatGPT token has no account id; is this account on a ChatGPT plan?")
    return creds


def refresh_chatgpt(refresh_token: str, account_id: str | None = None,
                    token_url: str = CHATGPT_TOKEN_URL) -> Credentials:
    body = urllib.parse.urlencode({"grant_type": "refresh_token", "refresh_token": refresh_token,
                                   "client_id": CHATGPT_CLIENT_ID}).encode()
    try:
        st, text = _post(token_url, body, {"content-type": "application/x-www-form-urlencoded"})
    except (urllib.error.URLError, OSError) as e:
        raise AuthError(f"ChatGPT token refresh: network error ({getattr(e, 'reason', e)})") from None
    creds = _creds(_token_response(st, text, "ChatGPT token refresh"), refresh_token)
    creds.account_id = chatgpt_account_id(creds.access_token) or account_id
    return creds


# ----------------------------------------------------------------------------------------
# token provider used by the LLM clients


class TokenProvider:
    """Hands out a valid access token for one provider, refreshing and persisting as needed."""

    def __init__(self, provider: str, store: AuthStore,
                 refresher: Callable[[Credentials], Credentials] | None = None):
        if provider not in PROVIDERS:
            raise ValueError(provider)
        self.provider = provider
        self.store = store
        self._refresher = refresher or (
            (lambda c: refresh_claude(c.refresh_token)) if provider == "claude"
            else (lambda c: refresh_chatgpt(c.refresh_token, c.account_id)))
        self._lock = threading.Lock()

    def _login_hint(self) -> str:
        return f"run `horizons login {self.provider}`"

    def get(self, force_refresh: bool = False) -> Credentials:
        with self._lock:
            creds = self.store.get(self.provider)
            if creds is None:
                raise AuthError(f"not logged in to {self.provider}; {self._login_hint()}")
            if not force_refresh and not creds.expires_soon():
                return creds
            try:
                fresh = self._refresher(creds)
            except InvalidGrant as e:
                # Another process may have rotated the refresh token first; use theirs if so.
                latest = self.store.get(self.provider)
                if latest and latest.refresh_token != creds.refresh_token and not latest.expires_soon():
                    return latest
                raise AuthError(f"{self.provider} login expired or was revoked ({e}); {self._login_hint()}") from None
            self.store.put(self.provider, fresh)
            return fresh
