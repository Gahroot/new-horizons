import os
import unittest

from horizons.template import TemplateError, load_topic
from tests.helpers import EXAMPLES, TINY_EVALUATOR, TINY_TOPIC, TempDirCase


class TemplateTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.write("baseline.py", "def value():\n    return 1\n")
        self.write("evaluate.py", TINY_EVALUATOR)
        self.write("script.json", "{}")

    def load(self, text: str):
        self.write("topic.toml", text)
        return load_topic(self.tmp / "topic.toml")

    def test_examples_load(self):
        spec = load_topic(EXAMPLES / "memory_reduction" / "topic.toml")
        self.assertEqual(spec.validation.metric, "peak_memory_mb")
        self.assertEqual(spec.validation.target_relative, -0.5)
        self.assertTrue(spec.allows("python_sandbox"))
        self.assertFalse(spec.allows("data_query"))
        lit = load_topic(EXAMPLES / "literature_question" / "topic.toml")
        self.assertEqual(lit.allowed_tools, ("literature",))

    def test_tiny_topic_loads(self):
        spec = self.load(TINY_TOPIC)
        self.assertEqual(spec.python_sandbox.evaluator, (self.tmp / "evaluate.py").resolve())
        self.assertIn("score", spec.validation.describe())

    def test_unknown_key_rejected(self):
        with self.assertRaisesRegex(TemplateError, "unknown key"):
            self.load(TINY_TOPIC.replace("replications = 2", "replications = 2\nreplicatoins = 3"))

    def test_parent_path_rejected(self):
        with self.assertRaisesRegex(TemplateError, "inside the topic directory"):
            self.load(TINY_TOPIC.replace('evaluator = "evaluate.py"', 'evaluator = "../evaluate.py"'))

    def test_absolute_path_rejected(self):
        with self.assertRaisesRegex(TemplateError, "inside the topic directory"):
            self.load(TINY_TOPIC.replace('evaluator = "evaluate.py"', 'evaluator = "/etc/passwd"'))

    def test_symlink_escape_rejected(self):
        outside = self.tmp.parent / (self.tmp.name + "-outside.py")
        outside.write_text("print(1)\n")
        self.addCleanup(outside.unlink)
        os.symlink(outside, self.tmp / "link.py")
        with self.assertRaisesRegex(TemplateError, "outside the topic directory"):
            self.load(TINY_TOPIC.replace('evaluator = "evaluate.py"', 'evaluator = "link.py"'))

    def test_unknown_tool_rejected(self):
        with self.assertRaisesRegex(TemplateError, "unknown tool"):
            self.load(TINY_TOPIC.replace('allowed = ["python_sandbox"]', 'allowed = ["python_sandbox", "shell"]'))

    def test_allowed_tool_needs_config(self):
        with self.assertRaises(TemplateError):
            self.load(TINY_TOPIC.replace('allowed = ["python_sandbox"]', 'allowed = ["python_sandbox", "data_query"]'))

    def test_relative_target_sign_must_match_direction(self):
        with self.assertRaisesRegex(TemplateError, "sign"):
            self.load(TINY_TOPIC.replace("target = { absolute = 10 }", "target = { relative_to_baseline = -0.5 }"))

    def test_significance_needs_enough_replications(self):
        text = TINY_TOPIC.replace('direction = "maximize"', 'direction = "maximize"\nkind = "significance"')
        with self.assertRaisesRegex(TemplateError, "replications >= 4"):
            self.load(text)
        spec = self.load(text.replace("replications = 2", "replications = 4"))
        self.assertEqual(spec.validation.kind, "significance")

    def test_bad_runtime_values_rejected(self):
        for old, new in (('memory = "256m"', 'memory = "lots"'), ('memory = "256m"', 'memory = "1m"'),
                         ("timeout_s = 20", "timeout_s = 0"),
                         ('evaluator = "evaluate.py"', 'evaluator = "evaluate.py"\nimage = "--privileged"')):
            with self.subTest(new=new), self.assertRaises(TemplateError):
                self.load(TINY_TOPIC.replace(old, new))

    def test_llm_base_url_must_be_https(self):
        with self.assertRaisesRegex(TemplateError, "https"):
            self.load(TINY_TOPIC.replace('scripted_file = "script.json"',
                                         'scripted_file = "script.json"\nbase_url = "http://evil.example.com/v1"'))


if __name__ == "__main__":
    unittest.main()
