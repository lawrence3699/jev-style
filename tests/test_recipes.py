"""Workflow behavior, using real client transport but deterministic responses (no quality claims)."""
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from jev_style import JevStyle
from examples.recipes.common import QUESTIONS, read_tickets, summarize, DATA
from examples.recipes.ticket_router import route as route_ticket
from examples.recipes.fallback_router import route as route_with_fallback
from examples.recipes.receipt_review import SAMPLE, review

ROOT = Path(__file__).resolve().parents[1]
TICKET = {"id": "t1", "text": "Please remove the duplicate charge.", "expected_team": "billing"}


def client_with(probabilities=None, status=200, image_support=False):
    calls = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "test-model", "images": image_support}]})
        calls.append({"body": json.loads(request.content), "headers": request.headers})
        if status != 200:
            return httpx.Response(status, json={"error": {"code": "unavailable", "message": "test"}})
        if image_support:
            answers = {"review": {"type": "noul", "noul": 0.9}}
        else:
            values = probabilities or {"billing": 0.9, "shipping": 0.06, "returns": 0.04}
            answers = {"team": {"type": "choice", "choice": max(values, key=values.get),
                                "probabilities": values, "confidence": 0.0}}
        return httpx.Response(200, json={"model": "test-model", "answers": answers})

    http = httpx.Client(base_url="http://local.test", transport=httpx.MockTransport(handle))
    return JevStyle(base_url="http://local.test", api_key="", http_client=http), calls


def test_confident_ticket_uses_probability_not_confidence_and_hides_ground_truth():
    js, calls = client_with()
    result = route_ticket(js, TICKET)
    assert result["decision"] == "billing" and result["top_probability"] == 0.9
    assert calls[0]["body"] == {"state": TICKET["text"], "questions": QUESTIONS}
    assert "expected_team" not in calls[0]["body"]


def test_confident_primary_skips_fallback():
    primary, first_calls = client_with()
    fallback, second_calls = client_with()
    result = route_with_fallback(primary, fallback, TICKET)
    assert result["source"] == "primary" and result["fallback_called"] is False
    assert len(first_calls) == 1 and second_calls == []


@pytest.mark.parametrize("status", [200, 503])
def test_uncertain_or_failed_primary_calls_fallback(status):
    primary, _ = client_with({"billing": 0.4, "shipping": 0.35, "returns": 0.25}, status=status)
    fallback, calls = client_with()
    result = route_with_fallback(primary, fallback, TICKET)
    assert result["source"] == "fallback" and result["decision"] == "billing"
    assert result["fallback_called"] is True and len(calls) == 1


def test_missing_or_uncertain_fallback_keeps_human_review():
    primary, _ = client_with({"billing": 0.4, "shipping": 0.35, "returns": 0.25})
    fallback, _ = client_with({"billing": 0.4, "shipping": 0.35, "returns": 0.25})
    assert route_with_fallback(primary, None, TICKET)["decision"] == "human_review"
    assert route_with_fallback(primary, fallback, TICKET)["decision"] == "human_review"


def test_failures_and_invalid_probabilities_do_not_auto_route():
    for values, status in [({"billing": 1.4, "shipping": -0.2, "returns": -0.2}, 200), (None, 503)]:
        js, _ = client_with(values, status=status)
        result = route_ticket(js, TICKET)
        assert result["decision"] == "human_review" and "error" in result


def test_metrics_count_accepted_labels_and_do_not_report_fake_accuracy():
    rows = [{"decision": "billing", "expected_team": "billing", "elapsed_ms": 10},
            {"decision": "shipping", "expected_team": "billing", "elapsed_ms": 30},
            {"decision": "human_review", "expected_team": "billing", "elapsed_ms": 20}]
    metrics = summarize(rows)
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["accuracy_on_accepted"] == 0.5 and metrics["labelled_accepted"] == 2
    assert summarize(rows, fake=True)["accuracy_on_accepted"] is None


def test_receipt_sends_image_reference_assets_and_header():
    js, calls = client_with(image_support=True)
    result = review(js, SAMPLE)
    assert result["recommendation"] == "manager_review"
    request = calls[0]
    assert request["body"]["state"]["receipt"] == "receipt_asset_1"
    assert request["body"]["assets"]["receipt_asset_1"]["format"] == "png"
    assert request["headers"]["TypeSafe-Danger-Zone-Images"] == "true"
    assert "Authorization" not in request["headers"]


def test_receipt_refuses_text_only_server_before_upload():
    js, calls = client_with(image_support=False)
    with pytest.raises(ValueError, match="PyTorch"):
        review(js, SAMPLE)
    assert calls == []


def test_synthetic_fixture_is_balanced_and_labels_never_become_questions():
    rows = read_tickets(DATA)
    assert len(rows) == 12
    assert {team: sum(r["expected_team"] == team for r in rows) for team in QUESTIONS["team"]["criteria"]} == {
        "billing": 4, "shipping": 4, "returns": 4}


@pytest.mark.parametrize("script,args", [
    ("ticket_router.py", ["--fake"]),
    ("fallback_router.py", ["--fake"]),
    ("receipt_review.py", ["--check"]),
])
def test_documented_commands_run_without_weights(script, args):
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / "recipes" / script), *args],
                          cwd=ROOT, text=True, capture_output=True, check=True)
    records = [json.loads(line) for line in proc.stdout.splitlines()]
    assert records
    if script == "receipt_review.py":
        assert records[-1]["inference_run"] is False
    else:
        assert records[-1]["mode"] == "fake_plumbing_only" and records[-1]["accuracy_on_accepted"] is None
