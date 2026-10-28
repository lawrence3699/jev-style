"""Pure-Python recovery messages and local SDK exports for the JevStyle demo.

This module imports no Gradio, Torch, model code, or network clients. Error text is
used only for conservative classification and is never copied into UI messages.
The caller may log the original exception server-side; never display it directly.
Render generated source with gr.Code or code_html(), not unescaped gr.HTML.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
import math
import re
from typing import Any, Mapping

MAX_QUESTIONS = 12
LONG_WAIT_SECONDS = 45
IMAGE_MARKER = "<|image_1|>"


@dataclass(frozen=True)
class Recovery:
    code: str
    title: str
    message: str
    action: str
    retryable: bool

    def as_markdown(self) -> str:
        """All fields come from fixed text below, never the exception message."""
        return f"**{self.title}**\n\n{self.message}\n\n{self.action}"


_RECOVERY = {
    "quota": Recovery(
        "quota", "GPU quota unavailable",
        "Hugging Face could not allocate enough GPU quota for this request.",
        "Wait for quota to become available, try a shorter request, or copy the local setup below. "
        "Signing in may change your allowance; it does not guarantee this run.", False),
    "queue": Recovery(
        "queue", "The demo is busy",
        "The shared GPU queue is full or taking longer than usual.",
        "Keep one request open, or cancel it before retrying later. You can also copy the local setup below.", True),
    "timeout": Recovery(
        "timeout", "The request timed out",
        "This request did not finish within the available time.",
        "Try less text or fewer questions, or copy the local setup below. A timeout does not tell us your remaining quota.", True),
    "input": Recovery(
        "input", "Check the request",
        "The request could not be accepted in its current form.",
        "Check the question type and options, shorten long inputs, and use a supported image. "
        "Your text and selected fields are still available to edit.", False),
    "unknown": Recovery(
        "unknown", "The run could not finish",
        "The demo encountered an unexpected error. We could not identify the cause reliably.",
        "Try again later or copy the local setup below. Your inputs have not been replaced with a result.", True),
}


def _error_text(error: BaseException | str | None) -> str:
    """Bound inspection work; no inspected bytes leave this module as UI text."""
    if error is None:
        return ""
    try:
        return str(error)[:16_384].lower()
    except Exception:
        return ""


def classify_error(error: BaseException | str | None = None, *, input_error: bool = False,
                   queue_full: bool = False, queue_wait_seconds: float | None = None) -> Recovery:
    """Classify a caught failure or an observed queue state, without inventing quota facts.

    ``input_error`` must be set only around the app's own validation/token-budget
    phase. Generic ValueError/HTTP 400/429/503 are not sufficient evidence by
    themselves. ``queue_wait_seconds`` must be an observed wait, not an ETA.
    Framework queue rejections can happen before a callback: expose a static local
    fallback too; a try/except inside inference cannot intercept every rejection.
    """
    names = {cls.__name__ for cls in type(error).__mro__} if isinstance(error, BaseException) else set()
    if input_error or names & {"JSONDecodeError", "InputBudgetError", "QuestionError", "SnippetInputError"}:
        return _RECOVERY["input"]
    if queue_full:
        return _RECOVERY["queue"]
    text = _error_text(error)
    quota_phrases = (
        "zerogpu quota exceeded", "you have exceeded your gpu quota", "insufficient gpu quota",
        "not enough gpu quota", "gpu quota exhausted", "space app has reached its gpu limit",
        "you have exceeded your zerogpu runs limit",
    )
    if any(phrase in text for phrase in quota_phrases):
        return _RECOVERY["quota"]
    queue_phrases = (
        "queue is full", "queue full", "queue capacity exceeded", "maximum queue size reached",
        "zerogpu queue timeout", "no gpu was available after", "zerogpu pending credits exceeded",
        "too many zerogpu credits allocated to running tasks",
    )
    if any(phrase in text for phrase in queue_phrases):
        return _RECOVERY["queue"]
    if isinstance(error, TimeoutError) or names & {"ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout"}:
        return _RECOVERY["timeout"]
    timeout_phrases = ("request timed out", "request timeout", "gateway timeout", "read timed out",
                       "gpu task is exceeding its requested duration", "gpu task exceeded its requested duration")
    if any(phrase in text for phrase in timeout_phrases):
        return _RECOVERY["timeout"]
    if error is None and isinstance(queue_wait_seconds, (int, float)) and not isinstance(queue_wait_seconds, bool):
        if math.isfinite(queue_wait_seconds) and queue_wait_seconds >= LONG_WAIT_SECONDS:
            return _RECOVERY["queue"]
    return _RECOVERY["unknown"]


class SnippetInputError(ValueError):
    """An invalid export request; text contains no copied user data."""


def _backend(backend: str, has_image: bool) -> str:
    if backend not in ("torch", "mlx"):
        raise SnippetInputError("Choose PyTorch or Apple silicon MLX.")
    if has_image and backend == "mlx":
        raise SnippetInputError("Images require the PyTorch build. MLX supports text only.")
    return backend


def local_commands(*, backend: str = "torch", has_image: bool = False) -> str:
    """Cross-platform CLI commands; the caller offers activation instructions separately."""
    backend = _backend(backend, has_image)
    support = ("PyTorch supports text and images on CUDA, Apple MPS, or CPU."
               if backend == "torch" else "Apple silicon MLX supports text only; use PyTorch for images.")
    return ("# Python 3.10+; run inside an activated virtual environment.\n"
            f"# {support}\n"
            f'python -m pip install -U "jev-style[{backend}]>=0.6.0"\n'
            f"jev-style serve --release 2b-v4 --backend {backend}\n"
            "# Keep this terminal open. Local Playground: http://127.0.0.1:8765\n")


def _json_value(value: Any, depth: int = 0) -> Any:
    """Copy only supported JSON values; reject NaN, objects and non-string keys."""
    if depth > 64:
        raise SnippetInputError("The input is nested too deeply to export.")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(v, depth + 1) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _json_value(v, depth + 1) for k, v in value.items()}
    raise SnippetInputError("Use finite JSON values and string object keys.")


def _literal(value: Any) -> str:
    """Valid Python literals, with markup/control characters escaped inside strings."""
    if isinstance(value, str):
        result = json.dumps(value, ensure_ascii=True)
        for char, code in (("<", "003c"), (">", "003e"), ("&", "0026"), ("`", "0060")):
            result = result.replace(char, "\\u" + code)
        return result
    if value is None or isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    return "{" + ", ".join(_literal(k) + ": " + _literal(v) for k, v in value.items()) + "}"


def normalize_questions(questions: list | Mapping[str, dict]) -> list[tuple[str, str, Any, Any]]:
    """Accept the demo's [{t,ins,crit}] or SDK {id:{type,instructions,criteria}} form."""
    if isinstance(questions, list):
        entries = [(f"question_{i}", q) for i, q in enumerate(questions, 1)]
    elif isinstance(questions, Mapping):
        entries = list(questions.items())
    else:
        raise SnippetInputError("Questions must be a list or an object keyed by question ID.")
    if not 1 <= len(entries) <= MAX_QUESTIONS:
        raise SnippetInputError("Export between 1 and 12 questions at a time.")
    result = []
    for key, q in entries:
        if not isinstance(key, str) or not key or not isinstance(q, dict):
            raise SnippetInputError("Use nonempty question IDs and question objects.")
        kind = q.get("t", q.get("type"))
        instructions = _json_value(q.get("ins", q.get("instructions")))
        criteria = _json_value(q.get("crit", q.get("criteria")))
        if (not isinstance(instructions, (str, dict, list)) or not instructions
                or isinstance(instructions, str) and not instructions.strip()):
            raise SnippetInputError("Every question needs nonempty text, object, or array instructions.")
        if kind == "choice":
            if isinstance(criteria, list) and all(isinstance(x, str) for x in criteria):
                if len(set(criteria)) != len(criteria):
                    raise SnippetInputError("Choice options must have unique names.")
                criteria = dict.fromkeys(criteria)
            if (not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255
                    or any(not k.strip() for k in criteria)):
                raise SnippetInputError("Choice questions need 1 to 255 named options.")
            if any(v is not None and not isinstance(v, (str, dict, list)) for v in criteria.values()):
                raise SnippetInputError("Option descriptions must be text, objects, arrays, or null.")
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise SnippetInputError("Score questions need 2 to 10 levels, lowest first.")
            for level in criteria:
                if not isinstance(level, (str, dict, list)) or not level:
                    raise SnippetInputError("Score levels must be nonempty text, objects, or arrays.")
                if isinstance(level, str) and not level.strip():
                    raise SnippetInputError("Score level text must not be empty.")
                if isinstance(level, dict) and "label" in level:
                    if not isinstance(level["label"], str) or not level["label"].strip():
                        raise SnippetInputError("Score labels must be nonempty text.")
                    if level.get("description") is not None and not isinstance(level["description"], str):
                        raise SnippetInputError("Score descriptions must be text.")
        elif kind == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                    raise SnippetInputError("Yes/no criteria may contain only true and false descriptions.")
                if any(v is not None and not isinstance(v, (str, dict, list)) for v in criteria.values()):
                    raise SnippetInputError("Yes/no descriptions must be text, objects, arrays, or null.")
        else:
            raise SnippetInputError("Use noul, choice, or score question types.")
        result.append((key, kind, instructions, criteria))
    return result


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)


def _image_state(state: Any, questions: list) -> tuple[Any, str]:
    all_text = "\n".join(_strings([state, _json_value(questions)]))
    asset = "jev_uploaded_image_1"
    while asset in all_text:
        asset += "_"
    question_text = "\n".join(_strings(_json_value(questions)))
    if re.search(r"<\|image_\d+\|>", question_text):
        raise SnippetInputError("Put image references in the state, not in question instructions or options.")
    if any(marker != IMAGE_MARKER for marker in re.findall(r"<\|image_\d+\|>", all_text)):
        raise SnippetInputError("This demo exports one image per request.")
    found = False

    def replace(value):
        nonlocal found
        if isinstance(value, str):
            found = found or IMAGE_MARKER in value
            return value.replace(IMAGE_MARKER, asset)
        if isinstance(value, list):
            return [replace(x) for x in value]
        if isinstance(value, dict):
            if any(IMAGE_MARKER in key for key in value):
                raise SnippetInputError("Put image references in state values, not object keys.")
            return {key: replace(item) for key, item in value.items()}
        return value

    state = replace(state)
    if not found:
        # Preserve all supplied data and attach the image explicitly for the SDK.
        state = {"image": asset, "input": state}
    return state, asset


def sdk_example(state: Any, questions: list | Mapping[str, dict], *, has_image: bool = False,
                backend: str = "torch") -> str:
    """Generate source for the current request, calling the local server started above.

    ``state`` is the parsed demo state (use the app's state_of for its text box).
    Uploaded images are referenced as a user-saved ``image.png`` file, never by the
    server's temporary path. No files are read here and no supplied code is run.
    If the state has no image marker, export wraps it as {image, input}; all input
    data is preserved but exact numerical parity with direct-runtime rendering is
    not promised. Gradio should render this as code, not as Markdown/HTML.
    """
    _backend(backend, has_image)
    state = _json_value(state)
    if not isinstance(state, (str, dict, list)):
        raise SnippetInputError("The state must be text, a JSON object, or a JSON array.")
    normalized = normalize_questions(questions)
    asset = None
    if has_image:
        state, asset = _image_state(state, normalized)
    elif any(re.search(r"<\|image_\d+\|>", s) for s in _strings(state)):
        raise SnippetInputError("Add the referenced image or remove its placeholder before exporting.")
    imports = sorted({kind for _, kind, _, _ in normalized} | ({"image"} if has_image else set()))
    lines = ["# Run the local setup first, then run this in a second terminal.",
             "from jev_style import JevStyle, " + ", ".join(imports), ""]
    if has_image:
        lines.extend(["# Save the selected image as image.png next to this script.",
                      "# Image requests require the 2B v4 PyTorch server; MLX is text-only."])
    lines.extend(["state = " + _literal(state), "questions = {"])
    for key, kind, instructions, criteria in normalized:
        args = _literal(instructions)
        if kind != "noul" or criteria is not None:
            args += ", " + _literal(criteria)
        lines.append("    " + _literal(key) + ": " + kind + "(" + args + "),")
    lines.extend(["}", "", 'with JevStyle(base_url="http://127.0.0.1:8765", api_key="") as client:'])
    extra = ", assets={" + _literal(asset) + ': image("image.png")}' if has_image else ""
    lines.append("    result = client.decide(state, questions" + extra + ")")
    lines.append('print(result["answers"])')
    return "\n".join(lines) + "\n"


def code_html(source: str) -> str:
    """Safe optional HTML display. Prefer a Gradio Code component for copying."""
    return '<pre><code class="language-python">' + html.escape(source, quote=True) + "</code></pre>"
