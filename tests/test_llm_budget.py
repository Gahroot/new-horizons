import os
import unittest
from unittest import mock

import tests.helpers  # noqa: F401 - isolates HORIZONS_AUTH_FILE from the user's real logins

from horizons.budget import BudgetExceeded, BudgetGuard
from horizons.llm import LLM, LLMError, ScriptedClient, extract_code, extract_json, make_client
from horizons.template import Budget


class ExtractTests(unittest.TestCase):
    def test_json_variants(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('Sure!\n```json\n{"a": [1, 2]}\n```\nthanks'), {"a": [1, 2]})
        self.assertEqual(extract_json('prefix {"a": "}{"} suffix'), {"a": "}{"})
        self.assertEqual(extract_json("[1, 2]"), [1, 2])
        with self.assertRaises(LLMError):
            extract_json("no json here")

    def test_code(self):
        self.assertEqual(extract_code("text\n```python\nx = 1\n```\nmore"), "x = 1\n")
        self.assertEqual(extract_code("x = 2"), "x = 2\n")


class BudgetTests(unittest.TestCase):
    def test_fails_closed(self):
        g = BudgetGuard(Budget(max_llm_calls=2, max_sandbox_runs=1, max_iterations=1))
        g.charge_llm()
        g.charge_llm()
        with self.assertRaises(BudgetExceeded):
            g.charge_llm()
        g.charge_sandbox()
        with self.assertRaises(BudgetExceeded):
            g.charge_sandbox()
        self.assertEqual(g.llm_calls, 2)

    def test_clock(self):
        g = BudgetGuard(Budget(max_wall_clock_min=1), elapsed_before_s=61)
        with self.assertRaises(BudgetExceeded):
            g.charge_llm()
        self.assertEqual(g.llm_calls, 0)

    def test_roundtrip(self):
        g = BudgetGuard(Budget())
        g.charge_llm()
        g2 = BudgetGuard.from_dict(Budget(), g.to_dict())
        self.assertEqual(g2.llm_calls, 1)


class ClientTests(unittest.TestCase):
    def test_scripted_sequence_and_repeat(self):
        c = ScriptedClient({"t": [{"a": 1}, "second"]})
        self.assertEqual(c.complete("t", "", ""), '{"a": 1}')
        self.assertEqual(c.complete("t", "", ""), "second")
        self.assertEqual(c.complete("t", "", ""), "second")
        with self.assertRaises(LLMError):
            c.complete("missing", "", "")

    def test_llm_wrapper_charges_before_call_and_retries_json(self):
        c = ScriptedClient({"t": ["not json", '{"ok": true}']})
        g = BudgetGuard(Budget(max_llm_calls=5))
        seen = []
        llm = LLM(c, g, lambda task, s, u, out: seen.append(out))
        self.assertEqual(llm.json("t", "sys", "user"), {"ok": True})
        self.assertEqual(g.llm_calls, 2)
        self.assertEqual(len(seen), 2)
        g2 = BudgetGuard(Budget(max_llm_calls=1))
        with self.assertRaises(BudgetExceeded):
            LLM(ScriptedClient({"t": ["x"]}), g2).json("t", "s", "u")

    def test_auto_client_needs_key(self):
        empty = {"HORIZONS_AUTH_FILE": os.environ["HORIZONS_AUTH_FILE"] + ".none"}
        with mock.patch.dict("os.environ", empty, clear=True):
            with self.assertRaisesRegex(LLMError, "horizons login claude"):
                make_client(None)
        with mock.patch.dict("os.environ", {**empty, "ANTHROPIC_API_KEY": "sk-test"}, clear=True):
            self.assertEqual(make_client(None).name, "anthropic")
        with mock.patch.dict("os.environ", {**empty, "HORIZONS_OPENAI_BASE_URL": "http://evil.example/v1"},
                             clear=True):
            with self.assertRaises(LLMError):
                make_client(None)

    def test_missing_key_error_does_not_leak(self):
        with mock.patch.dict("os.environ", {"HORIZONS_AUTH_FILE": os.environ["HORIZONS_AUTH_FILE"]}, clear=True):
            c = make_client("anthropic")
            with self.assertRaisesRegex(LLMError, "ANTHROPIC_API_KEY is not set"):
                c.complete("t", "s", "u")


if __name__ == "__main__":
    unittest.main()
