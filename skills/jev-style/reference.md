# Jev-Style reference

Source and full docs: https://github.com/lawrence3699/jev-style

## Request: `POST /v1/systemone`

| Field | |
|---|---|
| `model` | optional, ignored (one model per server) |
| `state` | string, or JSON object/array (serialised for the model); string values may name an image in `assets` |
| `assets` | optional, models that read images only (2B v4 on PyTorch): `{key: {"type": "image", "format": "png" \| "jpg", "data_base64": "..."}}`; needs the header `TypeSafe-Danger-Zone-Images: true` |
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
| 400 | `max_tokens_exceeded` | the input is over the model's budget: state + question over 25,600 tokens (2B v4: 32,768, image tokens included), or with the 0.8B a question with its options over 2,048; `message` says which (nothing is truncated). Servers before 0.5.0 answered 422 `input_budget_exceeded` |
| 400 | `invalid_encoding`, `nesting_too_deep`, `non_finite_number`, `invalid_compressed_body` | a request rule is broken: not UTF-8 or a byte order mark, over 64 nested levels in `state` / `instructions` / `criteria`, NaN / Infinity / 1e400, a malformed compressed body |
| 400 | `images_not_supported` | `assets` or multipart bodies for a model that reads text only, or an image reference in a question's instructions / criteria (images belong in `state`) |
| 400 | `unsupported_question_type` | `box` questions |
| 400 | `images_header_required`, `invalid_image`, `invalid_multipart` | images without `TypeSafe-Danger-Zone-Images: true`; an image that is not a still PNG / JPEG of 280..1568 px per side (78,400..2,458,624 px), an unused or missing asset; a malformed multipart body |
| 413 | `image_too_large`, `reference_too_long` | an image over 16 MiB; reference keys over 1,024 bytes each or 65,536 together |
| 413 | `payload_too_large` | body over 32 MiB, compressed or decompressed |
| 415 | `unsupported_content_encoding` | a `Content-Encoding` other than one layer of `gzip`, `x-gzip` or `zstd` |
| 401 | `unauthorized` | server has a key and the bearer is missing or wrong |
| 404 / 500 | `not_found` / `internal_error` | unknown route / server-side failure |

## Images (2B v4)

A reference is an asset key matched as plain text inside a string value of `state`: a value equal to the key is
the image; in a longer string every occurrence is replaced (left to right) and each use adds the image again.
Object keys are never replaced. Multipart works as in TypeSafe's API: a first part `request` (the JSON body with
`{"part": "<file part name>"}` in place of `format` + `data_base64`) and one file part per image. Image tokens count
toward the 32,768-token budget (about one token per 32 x 32 px).

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
