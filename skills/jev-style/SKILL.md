---
name: jev-style
description: Call the local Jev-Style decision model (systemone-compatible API, default http://127.0.0.1:8765) to answer typed questions about text or JSON - yes/no probabilities (noul), one-of-N choices, ordered scores - for routing, classification, gating, triage, guardrails and scoring. Use when writing code that classifies, routes, scores or checks text with a small local model, when replacing an LLM call that only returns a label, or when the user mentions jev-style, /v1/systemone, noul, choice or score questions.
---

# Jev-Style: local typed decisions

Jev-Style-0.8B-Decision-v3 is a small decision model that runs on the user's machine. It does not
generate text. It reads a **state** (text or JSON, up to 25,600 tokens) and answers **typed questions**
about it with calibrated probabilities, in one forward pass. Use it where code needs a fast, cheap,
structured judgement; keep using an LLM for writing.

| Type | Answer | Use for |
|---|---|---|
| `noul` | `noul` = P(statement is true), 0..1 | checks, gates, "is this X?" |
| `choice` | `choice` + `probabilities` per option + `confidence` | routing, labels, picking a candidate |
| `score` | `score` = expected level index + `probabilities` + `legend` | urgency, quality, risk on 2..10 ordered levels |

## 1. Make sure a server is running

```sh
curl -s http://127.0.0.1:8765/healthz        # {"status": "ok", "model": "jev-style-0.8b-decision-v3", ...}
```

If not, use the `jev-style-serve` skill, or ask the user to run `jev-style serve` (it loads the model;
do not start a second copy). If `healthz` reports `"model": "jev-style-fake"`, the server runs the fake
engine: answers are deterministic noise, fine for wiring tests, meaningless as decisions.

## 2. Call it

One request = one state + any number of named questions. The state is read once, so ask all
questions about the same state together.

```sh
curl -s http://127.0.0.1:8765/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Hi, I was billed twice for my March plan. Please refund one of them today.",
  "questions": {
    "refund":  {"type": "noul", "instructions": "The customer asks for money back."},
    "team":    {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "charges, refunds, invoices", "tech": "bugs, outages", "sales": null}},
    "urgency": {"type": "score", "instructions": "How urgent is it?",
                "criteria": ["can wait", "this week", {"label": "today", "description": "customer is losing money"}]}
  }}'
```

Python (`pip install jev-style` is enough for this client; running the model yourself needs `"jev-style[torch]"`, or `"jev-style[mlx]"` on Apple silicon; or just `httpx`/`requests`):

```python
from jev_style import JevStyle, noul, choice, score
js = JevStyle(base_url="http://127.0.0.1:8765")
out = js.decide(ticket_text, {
    "refund": noul("The customer asks for money back."),
    "team": choice("Which team should handle this?", {"billing": "charges, refunds", "tech": "bugs", "sales": None}),
    "urgency": score("How urgent is it?", ["can wait", "this week", "today"]),
})
out["answers"]["team"]["choice"], out["answers"]["team"]["confidence"]
```

Any systemone-compatible SDK also works: point its base URL at `http://127.0.0.1:8765` with any API
key string. If MCP tools named `decide`, `noul`, `choice`, `score` from a `jev-style` server are
available, call them directly for a quick one-off judgement instead of writing code.

## 3. Read the answer

- `noul`: `answers.<id>.noul` is P(true).
- `choice`: `choice` is the argmax; `probabilities` keep the option order and sum to 1;
  `confidence = (k * p_max - 1) / (k - 1)` for k options (0 = uniform, 1 = certain).
- `score`: `score = sum_i i * p_i` with level 0 = the first level, so it can land between levels;
  `legend` maps `"0"`, `"1"`, ... to labels and `probabilities` is keyed the same way.

## 4. Turn probabilities into actions in code

Never hand the raw number to the user as the decision. Map it with explicit thresholds:

```python
p = out["answers"]["refund"]["noul"]
action = "auto" if p >= 0.9 else "review" if p >= 0.5 else "skip"
```

Pick thresholds from the user's own labelled examples with the `jev-style-eval` skill
(`jev-style eval data.jsonl` prints the threshold for a 1 % / 5 % / 10 % error budget).

## 5. Limits (do not work around them by trimming silently)

- State + question must fit in **25,600 tokens**; each question with its options in **2,048 tokens**.
  Too big -> HTTP 422 `input_budget_exceeded`. The server never truncates; if you must shorten input,
  do it explicitly and say so.
- `choice`: 1..255 options. `score`: 2..10 levels, lowest first.
- Other errors: 422 `invalid_request` / `invalid_question` (names the question), 401 without the right key.
- It is a 0.8B model: good at clear-cut judgements, weaker on multi-step arithmetic or facts not in the
  state. Compute amounts, dates and counts in code and put the results into the state.

See `reference.md` next to this file for the full request/response format and pattern code.
