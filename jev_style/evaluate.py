"""``jev-style eval``: measure the model on your own labelled examples before you trust it.

Input: JSONL, one request per line, the same shape as ``POST /v1/systemone`` plus a ``label`` on each
question you want scored::

    {"state": "I was charged twice", "questions": {
        "team":  {"type": "choice", "instructions": "Which team?", "criteria": {"billing": null, "tech": null},
                  "label": "billing"},
        "angry": {"type": "noul", "instructions": "The customer is angry.", "label": false},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"],
                    "label": "high"}}}

Labels: noul true/false (or "true"/"false", 1/0); choice an option name; score a level label, or a 0-based level
index when no level is named like that (so levels "1".."5" take labels "1".."5").
Questions whose label cannot be read are skipped with a warning (``--strict`` stops instead).
Questions without a label are still sent (they share the state) but not scored.

Output, per question id and overall:

* accuracy (top answer == label), Brier score, log loss, ECE (15 equal-width bins on the top probability);
* automation at an error budget: the share of decisions you can let the model take on its own if you only
  act when its confidence is above a threshold, such that the error rate among those is <= the budget
  (1 %, 5 %, 10 %), with the threshold to use;
* a refitted temperature: probabilities are re-sharpened or softened as p_i^(1/T) / sum_j p_j^(1/T), with T
  fitted by log loss on a calibration split (``--cal-fraction``, default 0.5) and reported on the other half,
  so the before/after numbers are honest. Apply it in your code with ``recalibrate(probs, T)``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

BUDGETS = (0.01, 0.05, 0.10)
ECE_BINS = 15
T_GRID = [round(0.1 * 1.05 ** i, 4) for i in range(0, 91)]     # 0.1 .. ~8.1


# ----------------------------------------------------------------------------- labels
def _truth(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "yes", "1", "y", "t"):
        return True
    if s in ("false", "no", "0", "n", "f"):
        return False
    raise ValueError(f"noul label must be true/false, got {v!r}")


def label_key(q: dict, label: Any) -> str:
    """The option key (as in ``probabilities``) that the label names."""
    t = q.get("type")
    if t == "noul":
        return "true" if _truth(label) else "false"
    if t == "choice":
        crit = q.get("criteria")
        names = list(crit) if isinstance(crit, dict) else [str(c) for c in crit or []]
        if str(label) not in names:
            raise ValueError(f"choice label {label!r} is not one of {names}")
        return str(label)
    levels = q.get("criteria") or []
    labels = [str(lv["label"]) if isinstance(lv, dict) and "label" in lv else str(lv) for lv in levels]
    if str(label).strip() in labels:                 # a level name wins, so levels "1".."5" work as expected
        i = labels.index(str(label).strip())
    elif isinstance(label, int) or (isinstance(label, str) and label.strip().isdigit()):
        i = int(label)                               # otherwise a 0-based level index
    else:
        raise ValueError(f"score label {label!r} is neither one of {labels} nor a level index")
    if not 0 <= i < len(levels):
        raise ValueError(f"score label index {i} out of range")
    return str(i)


def answer_probs(ans: dict) -> dict[str, float]:
    if ans["type"] == "noul":
        return {"false": 1.0 - float(ans["noul"]), "true": float(ans["noul"])}
    return {k: float(v) for k, v in ans["probabilities"].items()}


# ----------------------------------------------------------------------------- metrics
def recalibrate(probs: dict[str, float], t: float) -> dict[str, float]:
    """p_i^(1/T) renormalised: the same as dividing the model's logits by an extra factor T."""
    logs = {k: math.log(max(p, 1e-12)) / t for k, p in probs.items()}
    m = max(logs.values())
    ex = {k: math.exp(v - m) for k, v in logs.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


def _conf(probs: dict[str, float]) -> float:
    return max(probs.values())


def metrics(items: list[dict]) -> dict:
    """items: [{"probs": {k: p}, "label": key}] -> summary."""
    n = len(items)
    if not n:
        return {"n": 0}
    correct, brier, nll = 0, 0.0, 0.0
    for it in items:
        p, y = it["probs"], it["label"]
        top = max(p, key=p.get)
        correct += top == y
        brier += sum((v - (1.0 if k == y else 0.0)) ** 2 for k, v in p.items())
        nll += -math.log(max(p.get(y, 0.0), 1e-12))
    return {"n": n, "accuracy": correct / n, "brier": brier / n, "log_loss": nll / n, "ece": ece(items),
            "automation": automation(items)}


def ece(items: list[dict], bins: int = ECE_BINS) -> float:
    acc = [[0, 0.0, 0] for _ in range(bins)]         # count, sum conf, correct
    for it in items:
        p = it["probs"]
        c = _conf(p)
        b = min(bins - 1, int(c * bins))
        acc[b][0] += 1
        acc[b][1] += c
        acc[b][2] += max(p, key=p.get) == it["label"]
    n = len(items)
    return sum(abs(s / cnt - ok / cnt) * cnt / n for cnt, s, ok in acc if cnt)


def automation(items: list[dict], budgets: Iterable[float] = BUDGETS) -> dict:
    """Largest share of decisions that can be automated (act only when top probability >= threshold)
    with an error rate <= budget among the automated ones."""
    ranked = sorted(((_conf(it["probs"]), max(it["probs"], key=it["probs"].get) == it["label"]) for it in items),
                    key=lambda x: -x[0])
    n = len(ranked)
    out = {}
    for b in budgets:
        best, thr, errs = 0, None, 0
        for i, (c, ok) in enumerate(ranked, 1):
            errs += not ok
            nxt = ranked[i][0] if i < n else -1.0
            if nxt == c:                              # a threshold cannot split tied confidences
                continue
            if errs / i <= b:
                best, thr = i, c
        out[f"{b:.0%}"] = {"coverage": best / n, "threshold": thr}
    return out


def fit_temperature(items: list[dict]) -> float:
    def loss(t: float) -> float:
        return sum(-math.log(max(recalibrate(it["probs"], t).get(it["label"], 0.0), 1e-12)) for it in items)
    return min(T_GRID, key=loss) if items else 1.0


# ----------------------------------------------------------------------------- running
def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "state" not in row or not isinstance(row.get("questions"), dict):
                raise ValueError(f"{path}:{ln}: expected {{'state', 'questions'}}")
            rows.append(row)
    return rows


def run(rows: list[dict], decide, strict: bool = False) -> list[dict]:
    """-> [{"row", "qid", "type", "probs", "label"}] for every labelled question."""
    out = []
    for ri, row in enumerate(rows):
        qs, labels = {}, {}
        for qid, q in row["questions"].items():
            q = dict(q)
            if "label" in q:
                raw = q.pop("label")
                try:
                    labels[qid] = label_key(q, raw)
                except ValueError as e:
                    if strict:
                        raise
                    print(f"warning: row {ri + 1}, question {qid!r}: {e}; not scored", file=sys.stderr)
            qs[qid] = q
        if not labels:
            continue
        resp = decide(row["state"], qs)
        for qid, key in labels.items():
            ans = resp["answers"][qid]
            out.append({"row": ri, "qid": qid, "type": ans["type"], "probs": answer_probs(ans), "label": key})
        if (ri + 1) % 25 == 0:
            print(f"  {ri + 1}/{len(rows)} rows", file=sys.stderr, flush=True)
    return out


def report(results: list[dict], cal_fraction: float = 0.5, seed: int = 0) -> dict:
    by_q: dict[str, list] = defaultdict(list)
    for r in results:
        by_q[r["qid"]].append(r)
    rows = sorted({r["row"] for r in results})
    random.Random(seed).shuffle(rows)
    cal_rows = set(rows[:int(len(rows) * cal_fraction)]) if 0 < cal_fraction < 1 else set()
    summary = {"overall": metrics(results), "questions": {}}
    for qid, items in by_q.items():
        entry = metrics(items)
        cal = [x for x in items if x["row"] in cal_rows]
        test = [x for x in items if x["row"] not in cal_rows]
        if cal and test:
            t = fit_temperature(cal)
            entry["temperature"] = {"fitted_T": t, "fit_on": len(cal), "report_on": len(test),
                                    "before": _brief(metrics(test)),
                                    "after": _brief(metrics([{**x, "probs": recalibrate(x["probs"], t)}
                                                             for x in test]))}
        summary["questions"][qid] = entry
    return summary


def _brief(m: dict) -> dict:
    return {k: m[k] for k in ("accuracy", "log_loss", "ece")} | {"automation_5%": m["automation"]["5%"]["coverage"]}


def render(summary: dict) -> str:
    def pct(x: float | None) -> str:
        return "   -  " if x is None else f"{100 * x:5.1f}%"
    lines = [f"{'question':<22}{'n':>6}{'acc':>8}{'brier':>8}{'ece':>7}   automate @1% / 5% / 10% error"]
    for name, m in [*summary["questions"].items(), ("OVERALL", summary["overall"])]:
        if not m.get("n"):
            continue
        a = m["automation"]
        lines.append(f"{name:<22}{m['n']:>6}{pct(m['accuracy']):>8}{m['brier']:>8.3f}{m['ece']:>7.3f}   "
                     + " / ".join(f"{pct(a[k]['coverage'])} (p>={a[k]['threshold']:.2f})" if a[k]['threshold']
                                  is not None else f"{pct(0.0)}" for k in ("1%", "5%", "10%")))
    temps = [(q, m["temperature"]) for q, m in summary["questions"].items() if "temperature" in m]
    if temps:
        lines += ["", "refitted temperature (fit on one half of the rows, reported on the other half):"]
        for q, t in temps:
            b, a = t["before"], t["after"]
            lines.append(f"  {q:<20} T={t['fitted_T']:<6}  ece {b['ece']:.3f} -> {a['ece']:.3f}   "
                         f"log loss {b['log_loss']:.3f} -> {a['log_loss']:.3f}   "
                         f"automate@5% {pct(b['automation_5%'])} -> {pct(a['automation_5%'])}")
        lines.append("  apply in code: jev_style.evaluate.recalibrate(probabilities, T)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jev-style eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="labelled JSONL")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--url", help="running server (default $JEV_STYLE_URL, else load the model in-process)")
    src.add_argument("--fake", action="store_true", help="fake engine (plumbing test only)")
    ap.add_argument("--backend", default="auto", choices=("auto", "torch", "mlx", "gguf"))
    ap.add_argument("--limit", type=int, help="only the first N rows")
    ap.add_argument("--cal-fraction", type=float, default=0.5, help="rows used to fit the temperature (0 = skip)")
    ap.add_argument("--json", metavar="PATH", help="also write the summary (and per-question results) as JSON")
    ap.add_argument("--strict", action="store_true", help="stop on a label that names no option (default: skip it)")
    args = ap.parse_args(argv)

    import os

    from .client import JevStyle
    rows = load_rows(Path(args.data))[: args.limit]
    url = args.url or os.environ.get("JEV_STYLE_URL")
    js = JevStyle(base_url=url) if url else JevStyle(backend=args.backend, fake=args.fake)
    print(f"{len(rows)} rows, engine: {url or ('fake' if args.fake else args.backend)}", file=sys.stderr)
    results = run(rows, js.decide, strict=args.strict)
    if not results:
        print("error: no labelled questions found (add \"label\" to questions)", file=sys.stderr)
        return 2
    summary = report(results, args.cal_fraction)
    print(render(summary))
    if args.fake:
        print("\nNOTE: fake engine - these numbers say nothing about the model.")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False,
                                              indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
