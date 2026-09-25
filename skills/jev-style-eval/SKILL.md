---
name: jev-style-eval
description: Measure the Jev-Style local decision model on the user's own labelled examples before relying on it - accuracy, Brier, ECE, how many decisions can be automated at a 1/5/10 % error budget and at which confidence threshold, plus a refitted calibration temperature. Use when the user asks whether the local model is good enough for their task, wants thresholds for automating a classification/routing/gating step, or wants to compare it with the labels an existing LLM or human process produces.
---

# Evaluate Jev-Style on your own data

`jev-style eval data.jsonl` sends every labelled example to the model and reports, per question and
overall: accuracy, Brier, ECE, and **automation at an error budget** (the share of decisions the model
can take alone if you only act above a confidence threshold, with the threshold to use). It also refits
a temperature on half the rows and reports the effect on the other half. No training involved.

## 1. Build the labelled file

One JSON object per line: the same shape as a `/v1/systemone` request, plus `"label"` on each question
to score. Lines starting with `#` are ignored.

```json
{"state": "I was charged twice for order #4411.",
 "questions": {
   "team":     {"type": "choice", "instructions": "Which team should handle this ticket?",
                "criteria": {"billing": "charges, refunds", "shipping": "delays, lost parcels"}, "label": "billing"},
   "escalate": {"type": "noul", "instructions": "The customer asks for a person to handle this urgently.",
                "label": false},
   "mood":     {"type": "score", "instructions": "How upset is the customer?",
                "criteria": ["calm", "annoyed", "angry"], "label": "annoyed"}}}
```

Labels: `noul` true/false; `choice` an option name; `score` a level label, or a 0-based index when no level
has that name (so with levels `"1"`..`"5"` the label `"3"` means the level named 3).

Where labels come from, best first:
1. **Real, human-labelled examples** the user already has (support tickets with their final team,
   moderated posts, reviewed PRs...). Write a short converter script; keep the question wording exactly
   what production will send.
2. **Existing LLM outputs** in logs (e.g. the label a GPT/Claude prompt returned): then the eval measures
   agreement with that LLM, which is the right number when replacing it (see `jev-style-adopt`).
3. **Hand-written examples**: only for a smoke test; say clearly that numbers on them are not evidence.

Aim for at least 100 labelled decisions per question (the automation numbers are noisy below that),
drawn the way production traffic is. Never put the same example in twice.

## 2. Run it

```sh
curl -s http://127.0.0.1:8765/healthz    # if nothing answers, start `jev-style serve` in another terminal
                                         # and wait until healthz answers before running eval
jev-style eval data.jsonl --url http://127.0.0.1:8765 --json eval.json
```

Without `--url` (or `JEV_STYLE_URL`), eval loads its own copy of the model in-process. Labels that name
no option are skipped with a warning (`--strict` stops instead).

A bundled example (30 hand-written tickets, format demo only):
`jev-style eval examples/support_tickets.jsonl --url http://127.0.0.1:8765` from a clone of https://github.com/lawrence3699/jev-style.

## 3. Read the report

```
question                   n     acc   brier    ece   automate @1% / 5% / 10% error
team                      30   93.3%   0.077  0.123    90.0% (p>=0.65) /  93.3% (p>=0.57) / 100.0% (p>=0.44)
```

- `acc`: top answer equals the label. `brier` (lower is better) and `ece` (0 = perfectly calibrated)
  judge the probabilities, which matter because the code acts on them.
- `automate @5%: 93.3% (p>=0.57)`: acting only when the top probability is >= 0.57 covers 93.3 % of
  cases with at most 5 % errors among them; the rest go to a person or an LLM. This is the number to
  pick thresholds with. The threshold is on the top probability (`max(probabilities)`, or max(p, 1-p) for
  `noul`), not on the `confidence` field.
- `refitted temperature`: if ECE and log loss drop on the held-out half, apply it in code with
  `jev_style.evaluate.recalibrate(probabilities, T)` before thresholding. If they do not improve (small
  data), keep T = 1.

## 4. Report honestly

Tell the user the n, which data it was, and that numbers on hand-written or tiny sets are indicative
only. If accuracy is low on a question, try rewording it (a `noul` phrased as a statement, clearer
`criteria` descriptions, an `other` option) and re-run on the same file; do not tune wording on the
final test rows you report.
