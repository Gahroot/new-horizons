"""Subscription login + client tests against local fake OAuth / model servers (no real accounts)."""

import base64
import hashlib
import io
import json
import os
import socket
import stat
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from horizons import auth
from horizons.auth import AuthError, AuthStore, Credentials, InvalidGrant, TokenProvider
from horizons.cli import main
from horizons.llm import (CLAUDE_CODE_IDENTITY, AnthropicClient, CodexClient, LLMError, codex_text, iter_sse,
                          make_client)
from tests.helpers import TempDirCase


def fake_jwt(account_id: str = "acct-123") -> str:
    def enc(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{enc({'alg': 'none'})}.{enc({auth.CHATGPT_JWT_CLAIM: {'chatgpt_account_id': account_id}})}.sig"


class FakeServer:
    """Local HTTP server replaying scripted (status, headers, body) responses and recording requests."""

    def __init__(self, responses: list[tuple[int, dict, bytes]]):
        self.responses = list(responses)
        self.requests: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = self.rfile.read(int(self.headers.get("content-length") or 0))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": body})
                status, headers, payload = outer.responses.pop(0) if outer.responses else (500, {}, b"no more")
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def token_json(access: str = "at-1", refresh: str | None = "rt-1", expires_in: int = 3600) -> tuple[int, dict, bytes]:
    d = {"access_token": access, "expires_in": expires_in}
    if refresh:
        d["refresh_token"] = refresh
    return 200, {"content-type": "application/json"}, json.dumps(d).encode()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerCase(TempDirCase):
    def serve(self, responses: list) -> FakeServer:
        s = FakeServer(responses)
        self.addCleanup(s.close)
        return s


class PrimitivesTests(unittest.TestCase):
    def test_pkce_s256(self):
        v, c = auth.pkce_pair()
        self.assertEqual(len(v), 43)
        expect = base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()
        self.assertEqual(c, expect)
        self.assertNotEqual(auth.pkce_pair()[0], v)

    def test_account_id_from_jwt(self):
        self.assertEqual(auth.chatgpt_account_id(fake_jwt("abc")), "abc")
        self.assertIsNone(auth.chatgpt_account_id("not-a-jwt"))
        self.assertIsNone(auth.chatgpt_account_id("a.!!!.c"))

    def test_chatgpt_paste_forms(self):
        p = auth.parse_chatgpt_paste
        self.assertEqual(p("http://localhost:1455/auth/callback?code=abc&state=xyz"), ("abc", "xyz"))
        self.assertEqual(p("abc#xyz"), ("abc", "xyz"))
        self.assertEqual(p("code=abc&state=xyz"), ("abc", "xyz"))
        self.assertEqual(p("a-long-raw-code-123"), ("a-long-raw-code-123", None))
        for bad in ("", "short", "two words here ok"):
            self.assertEqual(p(bad), (None, None))

    def test_claude_paste_requires_matching_state(self):
        self.assertEqual(auth.parse_claude_paste(" code1#st \n", "st"), "code1")
        for bad in ("code1#other", "code1", "#st"):
            with self.subTest(bad=bad), self.assertRaises(AuthError):
                auth.parse_claude_paste(bad, "st")


class StoreTests(TempDirCase):
    def test_private_roundtrip_and_delete(self):
        store = AuthStore(self.tmp / "sub" / "auth.json")
        self.assertIsNone(store.get("claude"))
        store.put("claude", Credentials("a", "r", 123.0))
        store.put("chatgpt", Credentials("b", "s", 456.0, "acct"))
        self.assertEqual(stat.S_IMODE(os.stat(store.path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(store.path.parent).st_mode), 0o700)
        self.assertEqual(store.get("chatgpt").account_id, "acct")
        self.assertEqual(store.providers(), ["chatgpt", "claude"])
        self.assertTrue(store.delete("claude"))
        self.assertFalse(store.delete("claude"))
        self.assertEqual(store.providers(), ["chatgpt"])
        self.assertEqual([p.name for p in store.path.parent.iterdir()], ["auth.json"])  # no temp files left

    def test_malformed_file(self):
        (self.tmp / "auth.json").write_text("{not json")
        with self.assertRaises(AuthError):
            AuthStore(self.tmp / "auth.json").get("claude")
        (self.tmp / "auth.json").write_text('{"claude": {"access_token": "x"}}')
        with self.assertRaises(AuthError):
            AuthStore(self.tmp / "auth.json").get("claude")


class TokenProviderTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.store = AuthStore(self.tmp / "auth.json")
        self.calls = []

    def provider(self, result=None, error=None) -> TokenProvider:
        def refresher(c: Credentials) -> Credentials:
            self.calls.append(c.refresh_token)
            if error:
                raise error
            return result
        return TokenProvider("claude", self.store, refresher)

    def test_valid_token_is_not_refreshed(self):
        self.store.put("claude", Credentials("a", "r", time.time() + 3600))
        self.assertEqual(self.provider().get().access_token, "a")
        self.assertEqual(self.calls, [])

    def test_expiring_token_is_refreshed_and_saved(self):
        self.store.put("claude", Credentials("old", "r1", time.time() + 60))
        got = self.provider(Credentials("new", "r2", time.time() + 3600)).get()
        self.assertEqual((got.access_token, self.calls), ("new", ["r1"]))
        self.assertEqual(self.store.get("claude").refresh_token, "r2")

    def test_force_refresh(self):
        self.store.put("claude", Credentials("a", "r1", time.time() + 3600))
        self.provider(Credentials("b", "r2", time.time() + 3600)).get(force_refresh=True)
        self.assertEqual(self.calls, ["r1"])

    def test_revoked_login_asks_to_log_in_again(self):
        self.store.put("claude", Credentials("a", "r1", 0))
        with self.assertRaisesRegex(AuthError, "horizons login claude"):
            self.provider(error=InvalidGrant("invalid_grant")).get()

    def test_rotation_by_another_process_is_picked_up(self):
        self.store.put("claude", Credentials("a", "r1", 0))

        def refresher(c):
            self.store.put("claude", Credentials("theirs", "r9", time.time() + 3600))  # the other process won
            raise InvalidGrant("refresh token already used")
        self.assertEqual(TokenProvider("claude", self.store, refresher).get().access_token, "theirs")

    def test_not_logged_in(self):
        with self.assertRaisesRegex(AuthError, "horizons login chatgpt"):
            TokenProvider("chatgpt", self.store).get()


class ClaudeLoginTests(ServerCase):
    def test_login_exchanges_code_with_pkce(self):
        srv = self.serve([token_json("sk-ant-oat-1", "rt-1", 28800)])
        seen = {}

        def open_url(u):
            seen["q"] = urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)

        creds = auth.login_claude(open_url, lambda msg: f"the-code#{seen['q']['state'][0]}",
                                  token_urls=(srv.url + "/v1/oauth/token",))
        q = seen["q"]
        self.assertEqual(q["client_id"], [auth.CLAUDE_CLIENT_ID])
        self.assertEqual(q["code_challenge_method"], ["S256"])
        self.assertIn("user:inference", q["scope"][0])
        req = srv.requests[0]
        body = json.loads(req["body"])
        self.assertEqual((body["grant_type"], body["code"], body["state"]),
                         ("authorization_code", "the-code", q["state"][0]))
        challenge = base64.urlsafe_b64encode(hashlib.sha256(body["code_verifier"].encode()).digest()).rstrip(b"=")
        self.assertEqual(challenge.decode(), q["code_challenge"][0])
        self.assertEqual(req["headers"]["anthropic-beta"], "oauth-2025-04-20")
        self.assertEqual(req["headers"]["user-agent"], "claude-cli/2.1.999 (external, cli)")
        self.assertEqual((creds.access_token, creds.refresh_token), ("sk-ant-oat-1", "rt-1"))
        self.assertAlmostEqual(creds.expires_at, time.time() + 28800, delta=5)

    def test_wrong_state_never_hits_token_endpoint(self):
        srv = self.serve([token_json()])
        with self.assertRaises(AuthError):
            auth.login_claude(lambda u: None, lambda m: "code#wrong", token_urls=(srv.url,))
        self.assertEqual(srv.requests, [])

    def test_refresh_falls_back_on_5xx_but_not_on_4xx(self):
        down = self.serve([(503, {}, b"down")])
        up = self.serve([token_json("new", None)])
        c = auth.refresh_claude("rt-old", (down.url, up.url))
        self.assertEqual((c.access_token, c.refresh_token), ("new", "rt-old"))  # keeps refresh token if not rotated
        bad = self.serve([(400, {}, b'{"error": "invalid_grant", "error_description": "revoked"}')])
        never = self.serve([token_json()])
        with self.assertRaisesRegex(InvalidGrant, "invalid_grant revoked"):
            auth.refresh_claude("rt", (bad.url, never.url))
        self.assertEqual(never.requests, [])


class ChatGPTLoginTests(ServerCase):
    def test_browser_callback_route(self):
        srv = self.serve([token_json(fake_jwt("acct-9"), "rt")])
        port = free_port()
        blocked = threading.Event()

        def open_url(u):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)
            self.assertEqual(q["redirect_uri"], [f"http://localhost:{port}/auth/callback"])
            state = q["state"][0]

            def browser():
                time.sleep(0.2)
                base = f"http://127.0.0.1:{port}/auth/callback"
                with self.assertRaises(urllib.error.HTTPError) as cm:  # a forged callback with the wrong state is refused
                    urllib.request.urlopen(f"{base}?code=evil&state=nope", timeout=5)
                cm.exception.close()
                urllib.request.urlopen(f"{base}?code=good-code&state={state}", timeout=5).read()
            threading.Thread(target=browser, daemon=True).start()

        creds = auth.login_chatgpt(open_url, lambda m: blocked.wait(30) or "", status=lambda m: None,
                                   token_url=srv.url + "/oauth/token", port=port, timeout_s=20)
        self.assertEqual(creds.account_id, "acct-9")
        form = urllib.parse.parse_qs(srv.requests[0]["body"].decode())
        self.assertEqual((form["grant_type"], form["code"]), (["authorization_code"], ["good-code"]))
        self.assertEqual(form["redirect_uri"], [f"http://localhost:{port}/auth/callback"])
        with self.assertRaises(OSError):  # listener is shut down afterwards
            urllib.request.urlopen(f"http://127.0.0.1:{port}/auth/callback", timeout=2)

    def test_paste_route_when_port_is_busy(self):
        srv = self.serve([token_json(fake_jwt(), "rt")])
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            port = busy.getsockname()[1]
            state = {}

            def open_url(u):
                state["s"] = urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["state"][0]
            answers = iter(["oops", None])

            def prompt(msg):
                a = next(answers)
                return a if a is not None else f"http://localhost:{port}/auth/callback?code=pasted&state={state['s']}"
            creds = auth.login_chatgpt(open_url, prompt, status=lambda m: None, token_url=srv.url, port=port,
                                       timeout_s=10)
        self.assertEqual(creds.account_id, "acct-123")
        self.assertEqual(urllib.parse.parse_qs(srv.requests[0]["body"].decode())["code"], ["pasted"])

    def test_pasted_url_from_another_login_is_rejected(self):
        srv = self.serve([token_json(fake_jwt())])
        with self.assertRaisesRegex(AuthError, "state mismatch"):
            auth.login_chatgpt(lambda u: None, lambda m: "http://localhost/auth/callback?code=c&state=other",
                               status=lambda m: None, token_url=srv.url, port=free_port(), timeout_s=10)
        self.assertEqual(srv.requests, [])

    def test_token_without_account_is_rejected(self):
        srv = self.serve([token_json("plain-token")])
        with self.assertRaisesRegex(AuthError, "account id"):
            auth.login_chatgpt(lambda u: None, lambda m: "raw-code-1234567", status=lambda m: None,
                               token_url=srv.url, port=free_port(), timeout_s=10)


def sse(*events: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"


class SubscriptionClientTests(ServerCase):
    def setUp(self):
        super().setUp()
        self.store = AuthStore(self.tmp / "auth.json")

    def test_claude_oauth_request_shape_and_401_refresh(self):
        ok = (200, {"content-type": "application/json"},
              json.dumps({"content": [{"type": "text", "text": "hello"}]}).encode())
        srv = self.serve([(401, {}, b'{"error": {"message": "expired"}}'), ok])
        self.store.put("claude", Credentials("tok-old", "rt", time.time() + 3600))
        prov = TokenProvider("claude", self.store, lambda c: Credentials("tok-new", "rt2", time.time() + 3600))
        client = AnthropicClient(model="claude-sonnet-5", base_url=srv.url, name="claude", oauth=prov)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-should-not-be-used"}):
            self.assertEqual(client.complete("t", "SYSTEM", "USER"), "hello")
        first, second = srv.requests
        self.assertEqual(first["headers"]["authorization"], "Bearer tok-old")
        self.assertEqual(second["headers"]["authorization"], "Bearer tok-new")
        for r in (first, second):
            self.assertNotIn("x-api-key", r["headers"])
            self.assertIn("oauth-2025-04-20", r["headers"]["anthropic-beta"])
            self.assertEqual(r["headers"]["x-app"], "cli")
            self.assertTrue(r["headers"]["user-agent"].startswith("claude-cli/"))
        body = json.loads(second["body"])
        self.assertEqual([b["text"] for b in body["system"]], [CLAUDE_CODE_IDENTITY, "SYSTEM"])
        self.assertEqual(body["messages"], [{"role": "user", "content": "USER"}])

    def test_codex_request_and_stream_parsing(self):
        events = sse(
            {"type": "response.output_item.added", "item": {"id": "rs_1", "type": "reasoning"}},
            {"type": "response.output_text.delta", "item_id": "rs_1", "delta": "SECRET THOUGHT"},
            {"type": "response.output_item.added", "item": {"id": "msg_1", "type": "message"}},
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": '{"a": '},
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "1}"},
            {"type": "response.output_item.done",
             "item": {"id": "msg_1", "type": "message", "content": [{"type": "output_text", "text": '{"a": 1}'}]}},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 3}}},
        )
        srv = self.serve([(200, {"content-type": "text/event-stream"}, events)])
        self.store.put("chatgpt", Credentials("cg-tok", "rt", time.time() + 3600, "acct-7"))
        client = CodexClient(oauth=TokenProvider("chatgpt", self.store), model="gpt-6-sol",
                             url=srv.url + "/backend-api/codex/responses")
        self.assertEqual(client.complete("t", "SYS", "USER"), '{"a": 1}')
        req = srv.requests[0]
        h = req["headers"]
        self.assertEqual((h["authorization"], h["chatgpt-account-id"]), ("Bearer cg-tok", "acct-7"))
        self.assertEqual((h["originator"], h["version"], h["x-openai-internal-codex-responses-lite"]),
                         ("codex_cli_rs", "0.155.1", "true"))
        body = json.loads(req["body"])
        self.assertEqual((body["model"], body["instructions"], body["stream"], body["store"]),
                         ("gpt-6-sol", "SYS", True, False))
        self.assertEqual(body["input"], [{"role": "user", "content": [{"type": "input_text", "text": "USER"}]}])

    def test_codex_usage_limit_is_not_retried(self):
        limit = json.dumps({"error": {"code": "usage_limit_reached", "resets_at": 2000000000}}).encode()
        srv = self.serve([(429, {}, limit), (200, {}, sse())])
        self.store.put("chatgpt", Credentials("t", "r", time.time() + 3600, "a"))
        client = CodexClient(oauth=TokenProvider("chatgpt", self.store), url=srv.url)
        with self.assertRaisesRegex(LLMError, "usage limit reached; it resets at"):
            client.complete("t", "s", "u")
        self.assertEqual(len(srv.requests), 1)

    def test_codex_401_refreshes_once(self):
        ok = sse({"type": "response.output_item.done",
                  "item": {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}})
        srv = self.serve([(401, {}, b"{}"), (200, {}, ok)])
        self.store.put("chatgpt", Credentials("old", "r", time.time() + 3600, "a"))
        prov = TokenProvider("chatgpt", self.store, lambda c: Credentials("new", "r2", time.time() + 3600, "a"))
        self.assertEqual(CodexClient(oauth=prov, url=srv.url).complete("t", "s", "u"), "hi")
        self.assertEqual(srv.requests[1]["headers"]["authorization"], "Bearer new")

    def test_codex_stream_error_event(self):
        with self.assertRaisesRegex(LLMError, "ChatGPT error: boom"):
            codex_text(iter_sse(io.BytesIO(sse({"type": "error", "error": {"message": "boom"}}))))

    def test_make_client_prefers_subscription_login(self):
        env = {"ANTHROPIC_API_KEY": "sk-x", "OPENAI_API_KEY": "sk-y"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(make_client(None, auth_store=self.store).name, "anthropic")
            self.store.put("chatgpt", Credentials("t", "r", 0, "a"))
            c = make_client(None, auth_store=self.store)
            self.assertEqual((c.name, c.model), ("chatgpt", "gpt-6-sol"))
            self.store.put("claude", Credentials("t", "r", 0))
            c = make_client(None, auth_store=self.store)
            self.assertEqual((c.name, c.model), ("claude", "claude-sonnet-5"))
            self.assertEqual(make_client("anthropic", auth_store=self.store).name, "anthropic")  # explicit wins
        self.store.delete("claude")
        with self.assertRaisesRegex(LLMError, "horizons login claude"):
            make_client("claude", auth_store=self.store)


class CliTests(TempDirCase):
    def test_login_status_logout_never_print_tokens(self):
        fake = Credentials("SECRET-ACCESS", "SECRET-REFRESH", time.time() + 3600)
        out = io.StringIO()
        with mock.patch("horizons.cli.auth.login_claude", return_value=fake), redirect_stdout(out):
            self.assertEqual(main(["login", "claude", "--no-browser"]), 0)
            main(["doctor"])
            self.assertEqual(main(["logout", "all"]), 0)
        text = out.getvalue()
        self.assertIn("Logged in to claude", text)
        self.assertIn("claude: logged in (token valid until", text)
        self.assertIn("claude: logged out", text)
        self.assertIn("chatgpt: was not logged in", text)
        self.assertNotIn("SECRET", text)
        self.assertIsNone(AuthStore().get("claude"))

    def test_failed_login_exit_code(self):
        err = io.StringIO()
        with mock.patch("horizons.cli.auth.login_chatgpt", side_effect=AuthError("state mismatch")), \
                redirect_stderr(err):
            self.assertEqual(main(["login", "chatgpt", "--no-browser"]), 1)
        self.assertIn("login failed: state mismatch", err.getvalue())


if __name__ == "__main__":
    unittest.main()
