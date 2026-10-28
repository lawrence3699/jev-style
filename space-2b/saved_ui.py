"""Render the three recorded demo examples as static, escaped HTML.

No Gradio import, network, JavaScript, model load, or inference. Once the HTML is
loaded, its native <details> controls keep working without a server connection.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES_PATH = HERE / "saved_examples.json"
RECEIPT_PATH = HERE / "examples" / "receipt.png"
RECEIPT_SHA256 = "413476173ea3a6e8d514bba7d2c4317e225b6b12a610c66fe02e29e8c4963c05"
TASKS = {
    "support_ticket": ("Route a support ticket", {"billing", "technical", "sales"}),
    "agent_command": ("Review an agent command", {"allow", "ask", "deny"}),
    "receipt_approval": ("Check a receipt", {"Yes", "No"}),
}


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Saved example {field} must be text")
    return html.escape(value, quote=True)


def _revision(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Saved example revision must be a full commit hash")
    return value


def _retrieved(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Saved example retrieval date must be text")
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid saved example retrieval date") from exc
    if date.tzinfo is None:
        raise ValueError("Saved example retrieval date must include its timezone")
    return date.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _probabilities(value: object, example_id: str) -> list[tuple[str, float]]:
    if not isinstance(value, dict) or set(value) != TASKS[example_id][1]:
        raise ValueError("Unexpected probability labels in saved example")
    for probability in value.values():
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not 0 <= probability <= 1 or not math.isfinite(probability)):
            raise ValueError("Saved probabilities must be finite numbers between 0 and 1")
    if not math.isclose(sum(value.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Saved probabilities must sum to one")
    return sorted(value.items(), key=lambda item: (-item[1], item[0]))


def _receipt(image_record: object) -> str:
    if not isinstance(image_record, dict) or image_record.get("sha256") != RECEIPT_SHA256:
        raise ValueError("Saved receipt metadata has an unexpected image hash")
    # Never trust a path or URL from JSON: only the fixed packaged image is read.
    try:
        data = RECEIPT_PATH.read_bytes()
    except OSError:
        return '<p class="saved-image-note">The recorded receipt image is unavailable.</p>'
    if (len(data) > 1_000_000 or not data.startswith(b"\x89PNG\r\n\x1a\n")
            or hashlib.sha256(data).hexdigest() != RECEIPT_SHA256):
        return ('<p class="saved-image-note">The recorded receipt image could not be '
                'verified and is not displayed.</p>')
    encoded = base64.b64encode(data).decode("ascii")
    return ('<figure class="saved-receipt"><img loading="lazy" '
            'alt="Synthetic receipt used for this saved example" '
            f'src="data:image/png;base64,{encoded}">'
            '<figcaption>Original synthetic receipt for this recorded result.</figcaption></figure>')


def _load_examples() -> dict[str, dict]:
    with EXAMPLES_PATH.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported saved example schema")
    records = document.get("examples")
    if not isinstance(records, list) or len(records) != len(TASKS):
        raise ValueError("Exactly three saved examples are required")
    found = {}
    for record in records:
        if not isinstance(record, dict) or record.get("id") not in TASKS:
            raise ValueError("Unknown saved example ID")
        if record["id"] in found:
            raise ValueError("Duplicate saved example ID")
        found[record["id"]] = record
    return found


def _render_card(example_id: str, record: dict, opened: bool) -> str:
    if example_id not in TASKS:
        raise ValueError("Unknown saved example ID")
    source, output = record.get("input"), record.get("output")
    model, evidence = record.get("model"), record.get("evidence")
    if not all(isinstance(part, dict) for part in (source, output, model, evidence)):
        raise ValueError("Incomplete saved example record")
    probabilities = _probabilities(output.get("probabilities"), example_id)
    label = output.get("label")
    if label != probabilities[0][0]:
        raise ValueError("Saved result label does not match its probabilities")
    state = _text(source.get("state_text"), "input")
    question = _text(source.get("question"), "question")
    kind = _text(source.get("answer_kind"), "answer kind")
    options = _text(source.get("options_text"), "options")
    if not options:
        options = "Yes / No"  # The fixed receipt fixture uses implicit boolean choices.
    repo = _text(model.get("repo_id"), "model repo")
    revision = _revision(model.get("revision"))
    space_revision = _revision(evidence.get("space_commit"))
    retrieved = _retrieved(evidence.get("retrieved_at_utc"))
    recorded = _text(evidence.get("cache_timestamp_raw"), "recorded time")
    historical = _text(output.get("metadata_from_recorded_run"), "historical metadata")
    rows = []
    for name, probability in probabilities:
        safe_name = _text(name, "option name")
        percent = probability * 100
        rows.append(
            '<li class="saved-probability">'
            f'<div class="saved-probability-label"><span>{safe_name}</span>'
            f'<strong>{percent:.2f}%</strong></div>'
            f'<meter class="saved-bar" min="0" max="1" value="{probability:.17g}" '
            f'aria-label="{safe_name}: {percent:.2f}%">{percent:.2f}%</meter>'
            f'<code class="saved-probability-exact">{probability!r}</code></li>'
        )
    picture = _receipt(source.get("image")) if example_id == "receipt_approval" else ""
    title = _text(TASKS[example_id][0], "task title")
    agent_note = ('<p class="saved-scope-note">This is an option-probability prediction, '
                  'not permission to execute a command or a safety guarantee.</p>'
                  if example_id == "agent_command" else "")
    open_attr = " open" if opened else ""
    return (
        f'<details class="saved-card" id="saved-{example_id}"{open_attr}>'
        f'<summary><span class="saved-card-title">{title}</span> '
        '<span class="saved-badge">Saved example · Not a live response</span></summary>'
        '<div class="saved-card-body"><div class="saved-input">'
        f'<h3>Recorded input</h3><pre>{state}</pre>{picture}'
        f'<h4>Question</h4><p>{question}</p><h4>Options ({kind})</h4><pre>{options}</pre>'
        '</div><div class="saved-output"><h3>Recorded probabilities</h3>'
        f'<ul class="saved-probabilities">{"".join(rows)}</ul>{agent_note}</div></div>'
        '<details class="saved-provenance"><summary>Recording details</summary><p><strong>Recorded model:</strong> '
        f'<code>{repo}</code><br><strong>Model revision:</strong> <code>{revision}</code></p>'
        f'<p><strong>Cache retrieved:</strong> {retrieved}<br>'
        f'<strong>Source cache timestamp:</strong> {recorded} (timezone not recorded)</p>'
        f'<p><strong>Source Space revision:</strong> <code>{space_revision}</code></p>'
        f'<p class="saved-historical"><strong>Historical run only:</strong> {historical}. '
        'These timings do not measure a current request.</p></details></details>'
    )


def render_saved_examples(selected_id: str = "support_ticket", expand: bool = False) -> str:
    """Return static HTML; expand only the selected card when explicitly requested.

    Invalid IDs, malformed probabilities or incomplete provenance fail closed
    with ValueError. Missing/unverified image bytes are never embedded.
    """
    if not isinstance(selected_id, str) or selected_id not in TASKS:
        raise ValueError("Unknown saved example ID")
    if not isinstance(expand, bool):
        raise ValueError("expand must be boolean")
    examples = _load_examples()
    cards = "".join(_render_card(key, examples[key], expand and key == selected_id) for key in TASKS)
    return (
        '<section class="saved-examples" aria-label="Saved example results">'
        '<h2>Explore saved examples</h2>'
        '<p class="saved-disclaimer"><strong>Saved examples · Not live responses.</strong> '
        'These recorded results are separate from the input above. '
        'Open them instantly without a GPU request.</p>'
        f'{cards}</section>'
    )
