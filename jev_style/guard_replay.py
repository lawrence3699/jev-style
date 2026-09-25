#!/usr/bin/env python3
"""Replay labelled tool calls through the Jev-Style guard and report allow / ask / deny rates.

    jev-style guard-replay --url http://127.0.0.1:8765     # a running server
    jev-style guard-replay --model                         # load the model in this process
    jev-style guard-replay --fake                          # fake engine (numbers mean nothing)
    jev-style guard-replay --examples my_calls.jsonl       # your own labelled calls

--fake runs the fake engine in-process (no model; answers are hash-based, so the agreement numbers
mean nothing). --url uses a running server. --model loads the real model in this process.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

from . import guard

HERE = Path(__file__).resolve().parent

LABELS = ("allow", "ask", "deny")
DEFAULT_EXAMPLES = HERE / "data" / "guard_examples" / "labelled_commands.jsonl"


def load_examples(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def make_caller(args) -> tuple:
    """Return (call(body)->response, engine description)."""
    if args.url:
        cfg = {"server_url": args.url, "api_key_env": "JEV_STYLE_API_KEY", "timeout_s": args.timeout,
               "allow_remote_server": args.allow_remote}
        return guard.http_caller(cfg), f"http {args.url}"
    from .client import JevStyle
    if args.model:
        mj = JevStyle(backend=args.backend)
        desc = f"in-process model ({args.backend})"
    else:
        mj = JevStyle(fake=True)
        desc = "in-process FAKE engine"
    return (lambda body: mj.decide(body["state"], body["questions"], model=body.get("model"))), desc


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def summarise(rows: List[Dict[str, Any]], results: List[Dict[str, Any]], engine: str) -> Dict[str, Any]:
    n = len(rows)
    final = Counter(r["decision"] for r in results)
    model = Counter(r.get("model_decision", r["decision"]) for r in results)
    confusion = {lab: {p: 0 for p in LABELS} for lab in LABELS}
    for row, res in zip(rows, results):
        if row.get("label") in confusion:
            confusion[row["label"]][res["decision"]] += 1
    labelled = sum(sum(v.values()) for v in confusion.values())
    correct = sum(confusion[l][l] for l in LABELS)
    unsafe_allowed = confusion["ask"]["allow"] + confusion["deny"]["allow"]
    lat = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    means: Dict[str, Dict[str, float]] = {}
    for lab in LABELS:
        vals: Dict[str, List[float]] = {}
        for row, res in zip(rows, results):
            if row.get("label") == lab:
                for q, v in (res.get("values") or {}).items():
                    vals.setdefault(q, []).append(v)
        means[lab] = {q: round(statistics.mean(v), 3) for q, v in vals.items()}
    return {
        "engine": engine,
        "n": n,
        "rates_final": {k: _pct(final.get(k, 0), n) for k in LABELS},
        "counts_final": {k: final.get(k, 0) for k in LABELS},
        "rates_model_only": {k: _pct(model.get(k, 0), n) for k in LABELS},
        "sources": dict(Counter(r["source"] for r in results)),
        "errors": sum(1 for r in results if r["source"] == "error"),
        "agreement": {"labelled": labelled, "correct": correct, "accuracy_pct": _pct(correct, labelled),
                      "unsafe_allowed": unsafe_allowed,
                      "denied_but_labelled_allow": confusion["allow"]["deny"]},
        "confusion_label_by_verdict": confusion,
        "mean_values_by_label": means,
        "latency_ms": {"mean": round(statistics.mean(lat), 2) if lat else None,
                       "p50": round(statistics.median(lat), 2) if lat else None,
                       "max": round(max(lat), 2) if lat else None},
    }


def render(summary: Dict[str, Any], rows, results, verbose: bool) -> str:
    s = summary
    out = [f"Jev-Style guard replay: {s['n']} tool calls, engine = {s['engine']}"]
    if "FAKE" in s["engine"]:
        out.append("NOTE: fake engine - verdicts come from hash-based probabilities, not from the model.")
    out.append("")
    out.append(f"{'':16}{'allow':>8}{'ask':>8}{'deny':>8}")
    out.append(f"{'final %':16}" + "".join(f"{s['rates_final'][k]:>8}" for k in LABELS))
    out.append(f"{'model only %':16}" + "".join(f"{s['rates_model_only'][k]:>8}" for k in LABELS))
    out.append("")
    out.append("label \\ verdict " + "".join(f"{k:>8}" for k in LABELS))
    for lab in LABELS:
        out.append(f"{lab:16}" + "".join(f"{s['confusion_label_by_verdict'][lab][k]:>8}" for k in LABELS))
    a = s["agreement"]
    out.append("")
    out.append(f"agreement {a['correct']}/{a['labelled']} = {a['accuracy_pct']}%   "
               f"unsafe allowed: {a['unsafe_allowed']}   denied-but-safe: {a['denied_but_labelled_allow']}   "
               f"errors: {s['errors']}")
    out.append(f"latency ms: mean {s['latency_ms']['mean']}  p50 {s['latency_ms']['p50']}  max {s['latency_ms']['max']}")
    if verbose:
        out.append("")
        for row, res in zip(rows, results):
            mark = "ok " if res["decision"] == row.get("label") else "-- "
            ti = row["hook_input"].get("tool_input", {})
            what = ti.get("command") or ti.get("file_path") or ti.get("url") or ti.get("notebook_path") or ""
            out.append(f"{mark}{row['id']:<5}{row.get('label', '?'):>6} -> {res['decision']:<6}"
                       f"{row['hook_input']['tool_name']:<12}{str(what)[:70]}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--fake", action="store_true", help="in-process fake engine (default)")
    src.add_argument("--url", help="running Jev-Style server, e.g. http://127.0.0.1:8765")
    src.add_argument("--model", action="store_true", help="load the real model in-process")
    ap.add_argument("--backend", default="auto", choices=("auto", "torch", "mlx", "gguf"))
    ap.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    ap.add_argument("--config", help="guard config (default: guard_config.json)")
    ap.add_argument("--no-rules", action="store_true", help="ignore hard_rules")
    ap.add_argument("--json", metavar="PATH", help="also write the summary and per-call verdicts as JSON")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every call")
    args = ap.parse_args(argv)

    cfg = guard.load_config(args.config)
    rows = load_examples(Path(args.examples))
    call, engine = make_caller(args)
    results = [guard.evaluate(r["hook_input"], cfg, call=call, use_rules=not args.no_rules,
                              home=r["hook_input"].get("home")) for r in rows]
    summary = summarise(rows, results, engine)
    print(render(summary, rows, results, args.verbose))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "results": [
            {"id": r["id"], "label": r.get("label"), **res} for r, res in zip(rows, results)]},
            ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
