"""No model, GPU, network, Gradio, or third-party test dependency is required."""
import ast
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery import (LONG_WAIT_SECONDS, SnippetInputError, classify_error, code_html,
                      local_commands, sdk_example)


class ClassificationTests(unittest.TestCase):
    def test_explicit_zerogpu_failures(self):
        cases = {
            "You have exceeded your GPU quota (60s requested vs. 10s left).": "quota",
            "ZeroGPU quota exceeded: Space app has reached its GPU limit.": "quota",
            "You have exceeded your ZeroGPU runs limit.": "quota",
            "The queue is full. Please try again.": "queue",
            "ZeroGPU queue timeout: No GPU was available after 60s": "queue",
            "You have too many ZeroGPU credits allocated to running tasks.": "queue",
            "GPU task is exceeding its requested duration and might be aborted": "timeout",
            "Gateway timeout": "timeout",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(classify_error(RuntimeError(message)).code, expected)

    def test_unknown_is_not_a_quota_guess(self):
        for message in ("HTTP 429 Too Many Requests", "HTTP 503", "Disk quota exceeded", "CUDA out of memory",
                        "GPU worker failed", "ZeroGPU illegal duration", "GPU quota is available"):
            with self.subTest(message=message):
                self.assertEqual(classify_error(RuntimeError(message)).code, "unknown")

    def test_validation_phase_overrides_reflected_user_text(self):
        error = ValueError("Invalid option: 'You have exceeded your GPU quota'")
        self.assertEqual(classify_error(error, input_error=True).code, "input")
        self.assertEqual(classify_error(ValueError("backend bug")).code, "unknown")
        self.assertEqual(classify_error(json.JSONDecodeError("bad", "x", 0)).code, "input")

    def test_typed_timeout_and_observed_queue_state(self):
        self.assertEqual(classify_error(TimeoutError("hidden details")).code, "timeout")
        self.assertEqual(classify_error(queue_full=True).code, "queue")
        self.assertEqual(classify_error(queue_wait_seconds=LONG_WAIT_SECONDS).code, "queue")
        self.assertEqual(classify_error(queue_wait_seconds=LONG_WAIT_SECONDS-1).code, "unknown")
        self.assertEqual(classify_error(queue_wait_seconds=float("nan")).code, "unknown")
        self.assertEqual(classify_error(RuntimeError("bug"), queue_wait_seconds=100).code, "unknown")

    def test_ui_never_contains_raw_exception_payload(self):
        secret = "hf_secret-token alice@example.org Traceback /tmp/private.py <script>alert(1)</script>"
        for prefix in ("You have exceeded your GPU quota. ", "queue is full ", "unexpected "):
            recovery = classify_error(RuntimeError(prefix + secret))
            text = recovery.as_markdown()
            for marker in ("hf_secret", "alice@", "Traceback", "/tmp", "<script>"):
                self.assertNotIn(marker, text)

    def test_broken_exception_str_is_unknown(self):
        class BrokenError(Exception):
            def __str__(self):
                raise RuntimeError("secret")
        self.assertEqual(classify_error(BrokenError()).code, "unknown")


class ExportTests(unittest.TestCase):
    def run_snippet(self, source):
        """Execute only generated source against a recording SDK, not a model."""
        ast.parse(source)
        captured = {}
        sdk = types.ModuleType("jev_style")

        class Client:
            def __init__(self, **kwargs): captured["client"] = kwargs
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def decide(self, state, questions, **kwargs):
                captured.update(state=state, questions=questions, **kwargs)
                return {"answers": {}}

        sdk.JevStyle = Client
        sdk.noul = lambda instructions, criteria=None: {"type": "noul", "instructions": instructions, "criteria": criteria}
        sdk.choice = lambda instructions, criteria: {"type": "choice", "instructions": instructions, "criteria": criteria}
        sdk.score = lambda instructions, criteria: {"type": "score", "instructions": instructions, "criteria": criteria}
        sdk.image = lambda path: {"saved_path": path}
        with patch.dict(sys.modules, {"jev_style": sdk}), patch("builtins.print"):
            exec(compile(source, "<generated export>", "exec"), {})
        return captured

    def test_three_question_types_preserve_values(self):
        state = {"message": "退款在哪？", "paid": True, "balance": None, "history": [3, 1.5, False]}
        questions = [
            {"t": "noul", "ins": "Is this about a refund?", "crit": None},
            {"t": "choice", "ins": "Team?", "crit": {"billing": "payments", "support": None}},
            {"t": "score", "ins": "Urgency?", "crit": ["low", "medium", "high"]},
        ]
        out = self.run_snippet(sdk_example(state, questions))
        self.assertEqual(out["state"], state)
        self.assertEqual(out["questions"]["question_2"]["criteria"], questions[1]["crit"])
        self.assertEqual(out["questions"]["question_3"]["criteria"], questions[2]["crit"])
        self.assertEqual(out["client"]["base_url"], "http://127.0.0.1:8765")
        self.assertNotIn("assets", out)

    def test_sdk_mapping_and_optional_noul_criteria(self):
        questions = {"ok": {"type": "noul", "instructions": {"rule": "check"},
                            "criteria": {"true": "yes", "false": "no"}}}
        out = self.run_snippet(sdk_example("x", questions, backend="mlx"))
        self.assertEqual(out["questions"], questions)

    def test_malicious_text_stays_data_and_is_escaped_for_display(self):
        hostile = "'); __import__('os').system('echo SHOULD_NOT_RUN'); #\n</code><script>alert(1)</script>\n```"
        questions = {hostile: {"t": "choice", "ins": hostile, "crit": {hostile: "x", "other": None}}}
        source = sdk_example(hostile, questions)
        out = self.run_snippet(source)
        self.assertEqual(out["state"], hostile)
        self.assertIn(hostile, out["questions"])
        self.assertNotIn("<script>", source)
        self.assertNotIn("```", source)
        self.assertNotIn("<script>", code_html(hostile))
        self.assertIn("&lt;script&gt;", code_html(hostile))

    def test_image_anchor_and_collision_free_asset(self):
        state = {"receipt": "<|image_1|>", "note": "jev_uploaded_image_1 already occurs"}
        questions = [{"t": "noul", "ins": "Review?", "crit": None}]
        out = self.run_snippet(sdk_example(state, questions, has_image=True))
        key = next(iter(out["assets"]))
        self.assertNotEqual(key, "jev_uploaded_image_1")
        self.assertEqual(out["state"]["receipt"], key)
        self.assertEqual(out["state"]["note"], state["note"])
        self.assertEqual(out["assets"][key], {"saved_path": "image.png"})

    def test_no_anchor_preserves_state_in_explicit_wrapper(self):
        original = {"policy": "Review meals above 50", "total": 60}
        out = self.run_snippet(sdk_example(original, [{"t": "noul", "ins": "Review?"}], has_image=True))
        self.assertEqual(out["state"]["input"], original)
        self.assertEqual(out["state"]["image"], next(iter(out["assets"])))
        self.assertEqual(original, {"policy": "Review meals above 50", "total": 60})

    def test_invalid_export_is_rejected_with_static_messages(self):
        valid = [{"t": "noul", "ins": "ok?"}]
        bad = [
            lambda: sdk_example("x", [], has_image=False),
            lambda: sdk_example(float("nan"), valid),
            lambda: sdk_example(True, valid),
            lambda: sdk_example("x", [{"t": "noul", "ins": {}}]),
            lambda: sdk_example("x", [{"t": "noul", "ins": "q", "crit": {"maybe": "yes"}}]),
            lambda: sdk_example("x", [{"t": "choice", "ins": "q", "crit": {"a": True, "b": None}}]),
            lambda: sdk_example("x", [{"t": "score", "ins": "q", "crit": [1, 2]}]),
            lambda: sdk_example({1: "bad key"}, valid),
            lambda: sdk_example("x", [{"t": "choice", "ins": "q", "crit": ["a", "a"]}]),
            lambda: sdk_example("x", [{"t": "score", "ins": "q", "crit": ["only one"]}]),
            lambda: sdk_example("x", [{"t": "unsupported", "ins": "q"}]),
            lambda: sdk_example("<|image_1|>", valid),
            lambda: sdk_example("<|image_2|>", valid, has_image=True),
            lambda: sdk_example("x", [{"t": "noul", "ins": "<|image_1|>"}], has_image=True),
            lambda: sdk_example("x", valid, has_image=True, backend="mlx"),
        ]
        for call in bad:
            with self.subTest(call=call), self.assertRaises(SnippetInputError): call()

    def test_local_setup_uses_correct_build(self):
        command = local_commands(has_image=True)
        self.assertIn('"jev-style[torch]>=0.6.0"', command)
        self.assertIn("--release 2b-v4 --backend torch", command)
        self.assertIn("Python 3.10+", command)
        self.assertIn("text only", local_commands(backend="mlx"))
        with self.assertRaises(SnippetInputError): local_commands(backend="mlx", has_image=True)


if __name__ == "__main__":
    unittest.main()
