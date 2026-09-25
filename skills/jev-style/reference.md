# Jev-Style reference

Source and full docs: https://github.com/lawrence3699/jev-style

## Request: `POST /v1/systemone`

| Field | |
|---|---|
| `model` | optional, ignored (one model per server) |
| `state` | string, or JSON object/array (serialised for the model) |
| `questions` | `{id: question}`, at least one |
| `question.type` | `noul`, `choice` or `score` |
| `question.instructions` | string, or JSON object/array (sent as compact JSON) |
| `criteria` (noul) | optional `{"true": "...", "false": "..."}` describing each side |
| `criteria` (choice) | `{option: description or null}`, 1..255 options, order kept |
| `criteria` (score) | list of 2..10 levels, lowest first: `"label"` or `{"label", "description"}` |

## Response

```json
{"model": "jev-style-0.8b-decision-v3",
 "answers": {
   "refund":  {"type": "noul", "noul": 0.97},
   "team":    {"type": "choice", "choice": "billing",
               "probabilities": {"billing": 0.91, "tech": 0.06, "sales": 0.03}, "confidence": 0.865},
   "urgency": {"type": "score", "score": 1.72, "legend": {"0": "can wait", "1": "this week", "2": "today"},
               "probabilities": {"0": 0.04, "1": 0.20, "2": 0.76}, "confidence": 0.64}},
 "usage": {"input_tokens": 212, "state_tokens": 31, "output_tokens": 0},
 "latency_ms": 61.2, "timing": {"total_ms": 61.2},
 "backend": "mlx"}
```

(The numbers above are illustrative.)

## Errors

`{"error": {"code", "message", "question"?}}`

| Status | code | Meaning |
|---|---|---|
| 422 | `invalid_json`, `invalid_request` | body or top-level fields wrong |
| 422 | `invalid_question` | one question is malformed; `question` names it |
| 422 | `input_budget_exceeded` | a question with its options is over 2,048 tokens, or state + question is over 25,600; `message` says which (nothing is truncated) |
| 401 | `unauthorized` | server has a key and the bearer is missing or wrong |
| 404 / 500 | `not_found` / `internal_error` | unknown route / server-side failure |

## Writing good questions

- Phrase a `noul` as a statement that is either true or false of the state
  ("The email asks for a meeting."), not as an open question.
- Use `criteria.true` / `criteria.false` when the boundary is fuzzy.
- Give `choice` options short names plus a description; add an `other` / `none` option when the
  state may fit none of them.
- Order `score` levels from lowest to highest and describe what each level means.
- Put facts computed in code (totals, day counts, matches) into the state as fields.

## Patterns

Thresholds from `jev-style eval` (`p>=...`) are on the **top probability** (`max(probabilities)`, or
`max(p, 1 - p)` for a noul), not on `confidence`; compare like with like.


Fan-out (one read, many answers):

```python
checks = {f"c{i}": noul(rule) for i, rule in enumerate(policy_rules)}
ans = js.decide(document, checks)["answers"]
failed = [policy_rules[int(k[1:])] for k, a in ans.items() if a["noul"] < 0.5]
```

Confidence routing:

```python
a = js.decide(ticket, {"team": choice("Which team?", teams)})["answers"]["team"]
if a["confidence"] >= 0.7:
    route(a["choice"])
else:
    send_to_human(ticket, suggestions=sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:2])
```

Composite score:

```python
ans = js.decide(lead, {
    "fit": score("How well does the company fit our product?", ["poor", "some", "good", "great"]),
    "intent": score("How strong is the buying intent?", ["none", "weak", "clear", "urgent"]),
    "budget": noul("The lead mentions an approved budget."),
})["answers"]
total = 0.4 * ans["fit"]["score"] / 3 + 0.4 * ans["intent"]["score"] / 3 + 0.2 * ans["budget"]["noul"]
```

Intent routing with a floor:

```python
handlers = {"order_status": "where is my order", "returns": "send something back",
            "billing": "charges and invoices", "other": "anything else"}
a = js.decide(message, {"intent": choice("What does the customer want?", handlers)})["answers"]["intent"]
handler = a["choice"] if a["probabilities"][a["choice"]] >= 0.6 else "other"
```

## MCP tools (`jev-style mcp`)

| Tool | Arguments | Returns |
|---|---|---|
| `decide` | `state`, `questions` | full response |
| `noul` | `state`, `question`, `true_means?`, `false_means?` | `p_true`, `answer` |
| `choice` | `state`, `question`, `options` (list or map) | `choice`, `probabilities`, `confidence` |
| `score` | `state`, `question`, `levels` | `score`, `level`, `legend`, `probabilities`, `confidence` |
| `model_info` | none | model id, budgets, backend |
