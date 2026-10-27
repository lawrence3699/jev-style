---
name: jev-style
description: Call the local Jev-Style decision model (systemone-compatible API, default http://127.0.0.1:8765) to answer typed questions about text or JSON - yes/no probabilities (noul), one-of-N choices, ordered scores - for routing, classification, gating, triage, guardrails and scoring. Use when writing code that classifies, routes, scores or checks text with a small local model, when replacing an LLM call that only returns a label, or when the user mentions jev-style, /v1/systemone, noul, choice or score questions.
---

# Jev-Style: local typed decisions

Jev-Style is a family of small decision models that run on the user's machine (default release
Jev-Style-0.8B-Decision-v3; `--release 2b` = Jev-Style-2B-Decision-v4). They do not generate text. A model
reads a **state** (text or JSON, up to 25,600 tokens; the 2B v4: up to 32,768 tokens, and images) and
answers **typed questions** about it with calibrated probabilities, in one forward pass. Use it where code needs a fast, cheap,
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

Images (only when `GET /v1/models` reports `"images": true` in `data[0]`, i.e. the 2B v4 on PyTorch): put
a reference key in the state and send the image under `assets`, as TypeSafe's API takes it, with the header
`TypeSafe-Danger-Zone-Images: true`. The Python client does both:

```python
from jev_style import JevStyle, image, noul
js = JevStyle(base_url="http://127.0.0.1:8765")
js.decide({"receipt": "r1", "policy": "Meals over 50 EUR need a manager approval."},
          {"approval": noul("This receipt needs a manager approval.")},
          assets={"r1": image("receipt.jpg")})        # PNG or JPEG, 280..1568 px per side
```

Any systemone-compatible SDK also works: point its base URL at `http://127.0.0.1:8765` with any API
key string. If MCP tools named `decide`, `noul`, `choice`, `score` from a `jev-style` server are
available, call them directly for a quick one-off judgement instead of writing code.

## 3. Read the answer

- `noul`: `answers.<id>.noul` is P(true).
- `choice`: `choice` is the argmax; `probabilities` keep the option order and sum to 1;
  `confidence = (k * p_max - 1) / (k - 1)` for k options (0 = uniform, 1 = certain).
- `score`: `score = sum_i i * p_i` with level 0 = the first level, so it can land between levels;
  `legend` maps `"0"`, `"1"`, ... to labels and `probabilities` is keyed the same way. Its `confidence`
  is TypeSafe's ordinal formula: probability on a level next to the peak costs less than probability far away.

## 4. Turn probabilities into actions in code

Never hand the raw number to the user as the decision. Map it with explicit thresholds:

```python
p = out["answers"]["refund"]["noul"]
action = "auto" if p >= 0.9 else "review" if p >= 0.5 else "skip"
```

Pick thresholds from the user's own labelled examples with the `jev-style-eval` skill
(`jev-style eval data.jsonl` prints the threshold for a 1 % / 5 % / 10 % error budget).

## 5. Limits (do not work around them by trimming silently)

- State + question must fit in **25,600 tokens** (2B v4: **32,768**, image tokens included); with the
  0.8B each question with its options must also fit in **2,048 tokens**. Too big -> HTTP 400 `max_tokens_exceeded`. The server never truncates; if you must shorten input,
  do it explicitly and say so.
- `choice`: 1..255 options. `score`: 2..10 levels, lowest first.
- Other errors: 422 `invalid_request` / `invalid_question` (names the question), 401 without the right key,
  400 `images_not_supported` for images on a text-only model (0.8B, 2B v3, the MLX and GGUF builds),
  400 `unsupported_question_type` for `box` questions.
- It is a small model (0.8B or 2B): good at clear-cut judgements, weaker on multi-step arithmetic or facts not in the
  state. Compute amounts, dates and counts in code and put the results into the state.

See `reference.md` next to this file for the full request/response format and pattern code.
