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

Comparing engines: repeat ``--server`` to run the same file through several systemone-compatible engines,
e.g. this package's local model, ``jev-style serve`` on another machine, or any other server that implements
``POST /v1/systemone``::

    jev-style eval data.jsonl --server local --server other=http://127.0.0.1:8000 --key other=OTHER_API_KEY

Each spec is ``NAME=URL`` (http/https), ``NAME=local[:release][:backend]`` such as ``local:mlx`` or ``local:2b:mlx``
(in-process),
``NAME=hf:<repo id>`` (in-process, a Jev-Style release by repo id), ``NAME=jevk5:<repo id or folder>`` (JevK5
in-process, answering as ``jevk5-serve`` does, see ``jev_style.jevk5_engine``),
``NAME=cascade:<cascade.json>`` (a confidence cascade, see ``jev_style.cascade``) or ``NAME=fake``; a bare
``local`` or ``fake`` is its own name. The first
engine is the reference. Every engine is scored on the same questions: the ones every
engine answered (a row an engine rejects, e.g. too long or too many options, is dropped for all and counted).
Differences to the reference come with a 95 % paired bootstrap interval over rows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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


def run(rows: list[dict], decide, strict: bool = False, errors: list | None = None) -> list[dict]:
    """-> [{"row", "qid", "type", "probs", "label"}] for every labelled question. With ``errors`` (a list), a row
    whose request fails is recorded there as {"row", "error"} and skipped instead of raising."""
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
        if errors is None:
            resp = decide(row["state"], qs)
        else:
            try:
                resp = decide(row["state"], qs)
            except Exception as e:  # noqa: BLE001 - any engine failure: record it, keep going
                errors.append({"row": ri, "error": f"{e.__class__.__name__}: {e}"[:300]})
                continue
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


# ----------------------------------------------------------------------------- comparing engines
BOOTSTRAP = 2000


def _row_stats(items: list[dict]) -> dict[int, list[float]]:
    """row -> [n, correct, brier sum]"""
    out: dict[int, list[float]] = defaultdict(lambda: [0, 0, 0.0])
    for it in items:
        p, y = it["probs"], it["label"]
        s = out[it["row"]]
        s[0] += 1
        s[1] += max(p, key=p.get) == y
        s[2] += sum((v - (1.0 if k == y else 0.0)) ** 2 for k, v in p.items())
    return out


def paired_diff(ref: list[dict], other: list[dict], seed: int = 0, n_boot: int = BOOTSTRAP) -> dict:
    """Accuracy and Brier of ``other`` minus ``ref`` on the same questions, with a 95 % bootstrap interval that
    resamples rows (questions of one row stay together)."""
    a, b = _row_stats(ref), _row_stats(other)
    rows = sorted(a)
    rng = random.Random(seed)

    def diff(sample: list[int]) -> tuple[float, float]:
        n = sum(a[r][0] for r in sample)
        return ((sum(b[r][1] for r in sample) - sum(a[r][1] for r in sample)) / n,
                (sum(b[r][2] for r in sample) - sum(a[r][2] for r in sample)) / n)

    acc, brier = diff(rows)
    boots = sorted(diff([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(n_boot)) if rows else []
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1

    def ci(i: int) -> list[float]:
        vals = sorted(x[i] for x in boots)
        return [vals[lo], vals[hi]] if vals else [0.0, 0.0]
    return {"accuracy": acc, "accuracy_ci95": ci(0), "brier": brier, "brier_ci95": ci(1), "rows": len(rows)}


def compare(rows: list[dict], engines: dict, strict: bool = False) -> dict:
    """engines: {name: decide}. Scores every engine on the questions all of them answered."""
    raw, errs = {}, {}
    for name, decide in engines.items():
        print(f"engine {name} ...", file=sys.stderr, flush=True)
        errs[name] = []
        raw[name] = run(rows, decide, strict=strict, errors=errs[name])
    keys = set.intersection(*({(r["row"], r["qid"]) for r in res} for res in raw.values()))
    common = {n: [r for r in res if (r["row"], r["qid"]) in keys] for n, res in raw.items()}
    names = list(engines)
    ref = names[0]
    return {"reference": ref, "common_questions": len(keys),
            "engines": {n: {"answered": len(raw[n]), "failed_rows": len(errs[n]), "errors": errs[n][:20],
                            "summary": metrics(common[n])} for n in names},
            "vs_reference": {n: paired_diff(common[ref], common[n]) for n in names[1:]},
            "results": common}


def render_compare(c: dict) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:5.1f}%"
    lines = [f"{c['common_questions']} questions answered by every engine; the table and the differences use only "
             "those.", "",
             f"{'engine':<18}{'answered':>9}{'failed rows':>12}{'acc':>8}{'brier':>8}{'log loss':>9}{'ece':>7}"
             "   automate @5% error"]
    for name, e in c["engines"].items():
        m = e["summary"]
        if not m.get("n"):
            lines.append(f"{name:<18}{e['answered']:>9}{e['failed_rows']:>12}   (nothing in common)")
            continue
        a5 = m["automation"]["5%"]
        lines.append(f"{name:<18}{e['answered']:>9}{e['failed_rows']:>12}{pct(m['accuracy']):>8}{m['brier']:>8.3f}"
                     f"{m['log_loss']:>9.3f}{m['ece']:>7.3f}   {pct(a5['coverage'])}"
                     + (f" (p>={a5['threshold']:.2f})" if a5["threshold"] is not None else ""))
    if c["vs_reference"]:
        lines += ["", f"difference to {c['reference']} (95 % paired bootstrap over rows; brier: lower is better):"]
        for name, d in c["vs_reference"].items():
            lo, hi = d["accuracy_ci95"]
            blo, bhi = d["brier_ci95"]
            lines.append(f"  {name:<16} accuracy {100 * d['accuracy']:+5.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                         f"   brier {d['brier']:+.3f} [{blo:+.3f}, {bhi:+.3f}]")
    for name, e in c["engines"].items():
        if e["errors"]:
            lines.append(f"\n{name}: {e['failed_rows']} rows failed, first: row {e['errors'][0]['row'] + 1}: "
                         f"{e['errors'][0]['error']}")
    return "\n".join(lines)


def parse_server(spec: str) -> tuple[str, str]:
    """'NAME=TARGET' or bare 'local' / 'fake' -> (name, target)."""
    name, sep, target = spec.partition("=")
    if not sep:
        if spec.split(":", 1)[0] in ("local", "fake"):
            return spec, spec
        raise ValueError(f"--server {spec!r}: expected NAME=URL, NAME=local[:release][:backend], NAME=hf:<repo>, "
                         "NAME=jevk5:<repo or folder>, NAME=cascade:<path> or NAME=fake")
    if not name or not target:
        raise ValueError(f"--server {spec!r}: empty name or target")
    return name, target


def _keyed(pairs: list[str] | None, flag: str) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        k, sep, v = p.partition("=")
        if not sep or not k or not v:
            raise ValueError(f"{flag} {p!r}: expected NAME=VALUE")
        out[k] = v
    return out


def build_engine(target: str, backend: str, api_key: str | None = None, model: str | None = None):
    """A ``decide(state, questions)`` callable for an engine target (``jev_style.client.parse_target``)."""
    from .client import from_target
    js = from_target(target, backend=backend, api_key=api_key)
    return lambda state, qs: js.decide(state, qs, model=model)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jev-style eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="labelled JSONL")
    ap.add_argument("--server", action="append", metavar="NAME=TARGET",
                    help="compare engines (repeatable; the first is the reference), see above")
    ap.add_argument("--key", action="append", metavar="NAME=ENV",
                    help="bearer token for server NAME, read from environment variable ENV")
    ap.add_argument("--model", action="append", metavar="NAME=MODEL", help="'model' field to send to server NAME")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--url", help="running server (default $JEV_STYLE_URL, else load the model in-process)")
    src.add_argument("--fake", action="store_true", help="fake engine (plumbing test only)")
    ap.add_argument("--backend", default="auto", choices=("auto", "torch", "mlx", "gguf"))
    ap.add_argument("--limit", type=int, help="only the first N rows")
    ap.add_argument("--cal-fraction", type=float, default=0.5, help="rows used to fit the temperature (0 = skip)")
    ap.add_argument("--json", metavar="PATH", help="also write the summary (and per-question results) as JSON")
    ap.add_argument("--strict", action="store_true", help="stop on a label that names no option (default: skip it)")
    args = ap.parse_args(argv)

    from .client import JevStyle
    rows = load_rows(Path(args.data))[: args.limit]
    if args.server:
        if args.url or args.fake:
            ap.error("--server cannot be combined with --url or --fake (use NAME=URL or NAME=fake)")
        try:
            specs = [parse_server(s) for s in args.server]
            keys, models = _keyed(args.key, "--key"), _keyed(args.model, "--model")
        except ValueError as e:
            ap.error(str(e))
        names = [n for n, _ in specs]
        if len(set(names)) != len(names):
            ap.error(f"engine names must be unique, got {names}")
        for flag, d in (("--key", keys), ("--model", models)):
            for k in d:
                if k not in names:
                    ap.error(f"{flag} names {k!r}, which is not a --server name")
        missing = [keys[n] for n in keys if not os.environ.get(keys[n])]
        if missing:
            print(f"error: environment variable(s) {', '.join(missing)} empty or unset", file=sys.stderr)
            return 2
        engines = {n: build_engine(t, args.backend, os.environ.get(keys[n]) if n in keys else None, models.get(n))
                   for n, t in specs}
        print(f"{len(rows)} rows, engines: {', '.join(f'{n}={t}' for n, t in specs)}", file=sys.stderr)
        c = compare(rows, engines, strict=args.strict)
        if not c["common_questions"]:
            print(render_compare(c))
            print("error: no labelled question was answered by every engine", file=sys.stderr)
            return 2
        print(render_compare(c))
        if any(t == "fake" for _, t in specs):
            print("\nNOTE: a fake engine is included - its numbers say nothing about any model.")
        if args.json:
            Path(args.json).write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
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
