"""Shared request and reporting helpers for the runnable routing recipes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from jev_style import JevStyle, choice

DATA = Path(__file__).with_name("data") / "tickets.jsonl"
TEAMS = {
    "billing": "Charges, payment failures, invoices, or duplicate payments.",
    "shipping": "Delivery tracking, delays, or a parcel that has not arrived.",
    "returns": "Return, refund, or exchange of an item that arrived damaged or in the wrong size.",
}
QUESTIONS = {"team": choice("Which team should handle the main request?", TEAMS)}


def probability_arg(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("use a finite probability between 0 and 1")
    return number


def add_arguments(parser):
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="primary systemone server")
    parser.add_argument("--input", type=Path, default=DATA, help="JSONL with id, text and optional expected_team")
    parser.add_argument("--min-probability", type=probability_arg, default=0.8,
                        help="illustrative top-option probability threshold (not the API confidence field)")
    parser.add_argument("--fake", action="store_true", help="exercise plumbing without a model; scores are meaningless")


def read_tickets(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"line {number}: expected a nonempty text string")
        if "expected_team" in row and row["expected_team"] not in TEAMS:
            raise ValueError(f"line {number}: expected_team must be one of {list(TEAMS)}")
        rows.append({**row, "id": row.get("id", str(number))})
    if not rows:
        raise ValueError("input contains no tickets")
    return rows


def client(url, fake=False):
    # No API key or remote endpoint is inferred for the fallback server.
    return JevStyle(fake=True) if fake else JevStyle(base_url=url, api_key="")


def pick(response):
    answer = response["answers"]["team"]
    probabilities = answer["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(TEAMS):
        raise ValueError("server returned unexpected team options")
    values = list(probabilities.values())
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("server returned invalid probabilities")
    if not math.isclose(sum(values), 1.0, abs_tol=0.01):
        raise ValueError("server probabilities do not sum to 1")
    team = max(probabilities, key=probabilities.get)
    return team, probabilities[team]


def summarize(rows, fake=False):
    accepted = [r for r in rows if r["decision"] != "human_review"]
    labelled = [r for r in accepted if r.get("expected_team") is not None]
    times = [r["elapsed_ms"] for r in rows]
    return {
        "kind": "summary", "mode": "fake_plumbing_only" if fake else "model_run",
        "tickets": len(rows), "automatically_routed": len(accepted),
        "coverage": len(accepted) / len(rows) if rows else 0,
        "labelled_accepted": len(labelled),
        "accuracy_on_accepted": (sum(r["decision"] == r["expected_team"] for r in labelled) / len(labelled)
                                 if labelled and not fake else None),
        "mean_elapsed_ms": mean(times) if times else None,
        "median_elapsed_ms": median(times) if times else None,
        "timing_scope": "client request(s), including first request; excludes server startup and model download",
        "note": "Bundled data is synthetic; this is a workflow check, not a general accuracy or savings claim.",
    }


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
