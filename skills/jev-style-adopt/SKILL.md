---
name: jev-style-adopt
description: Find LLM API calls in a codebase that only return a decision (a category, yes/no, or a 1-5 rating) and move them to the local Jev-Style decision model safely - scan, pick candidates with the user, rewrite each as a typed noul/choice/score question with a confidence cascade back to the LLM, run in shadow mode, measure agreement with jev-style eval, then switch. Use when the user wants to cut LLM cost or latency for classification, routing, moderation, gating or scoring calls, or asks which LLM calls a small local model could replace.
---

# Replace decision-only LLM calls with Jev-Style

Many codebases send a hosted LLM a prompt and then parse one word, a boolean or a number out of the
reply. Those calls pay for text generation to get a label. Jev-Style answers the same question
locally, in one forward pass, with a probability per option. This skill moves them over without
changing behaviour blindly: **scan -> choose -> rewrite with a cascade -> shadow -> measure -> switch**.

Never delete the LLM path in the first change. Every step below is reversible.

## 1. Scan

```sh
python3 <this-skill-dir>/scripts/find_decisions.py path/to/src --json /tmp/decision_calls.json
```

It lists LLM call sites whose prompt or parsing looks decision-shaped (classify / "yes or no" /
"respond with only" / rate 1-5 / enum or bool schemas / tiny `max_tokens` / `.strip().lower() == "yes"`)
with a guessed type. It is a heuristic: read every hit. Drop calls that need generated text, world
knowledge not in the input, multi-step math, or tool use.

## 2. Choose with the user

Show a short table: file:line, what it decides, guessed type, how often it runs (ask, or look for
loops/queues), and what a wrong answer costs. Recommend starting with one high-volume, low-risk call.
Confirm the server is running (`curl -s http://127.0.0.1:8765/healthz`; else the `jev-style-serve`
skill).

## 3. Translate the prompt into a typed question

| LLM prompt pattern | Jev-Style question |
|---|---|
| "Classify into one of A, B, C. Respond with only the name." | `choice`, criteria `{A: desc, B: desc, C: desc}` (add `other` if the prompt allowed none) |
| "Answer yes or no: does X ...?" / returns bool | `noul`, instructions = X as a **statement** ("The ticket asks for a refund."), optional `criteria.true/false` |
| "Rate 1 to 5" / "low, medium, high" | `score`, criteria = the levels, **lowest first**, with short descriptions |
| JSON with several such fields | one request, several questions about the same state |

Move instructions about *what to judge* into `instructions`/`criteria`; move the *input* into `state`
(text or a JSON object with named fields). Put facts the prompt made the LLM compute (totals, dates,
counts) into the state, computed in code.

## 4. Rewrite with a cascade and shadow logging

Keep the old function as `*_llm`. The new one asks Jev-Style first and falls back to the LLM when the
model is unsure or unreachable. While shadowing, it also calls the LLM and logs both answers in the
`jev-style eval` format. Python sketch (adapt to the codebase's language and HTTP client):

```python
import json, os, time, httpx

JEV_URL = os.environ.get("JEV_STYLE_URL", "http://127.0.0.1:8765")
SHADOW = os.environ.get("JEV_SHADOW_LOG")          # e.g. logs/route_shadow.jsonl while measuring
THRESHOLD = 0.60                                  # replace with the value jev-style eval recommends
TEAMS = {"billing": "charges, refunds, invoices", "shipping": "delivery, delays, lost parcels",
         "returns": "exchanges, wrong item", "account": "login, password, privacy"}

def route(ticket: str) -> str:
    q = {"type": "choice", "instructions": "Which team should handle this support ticket?", "criteria": TEAMS}
    try:
        a = httpx.post(f"{JEV_URL}/v1/systemone", json={"state": ticket, "questions": {"team": q}},
                       timeout=5).raise_for_status().json()["answers"]["team"]
    except Exception:
        return route_llm(ticket)                   # server down: old behaviour
    confident = max(a["probabilities"].values()) >= THRESHOLD
    if SHADOW:
        llm = route_llm(ticket)
        with open(SHADOW, "a") as f:
            f.write(json.dumps({"state": ticket, "questions": {"team": {**q, "label": llm}},
                                "jev": a, "ts": time.time()}) + "\n")
        return llm                                 # shadow mode: production keeps the LLM answer
    return a["choice"] if confident else route_llm(ticket)
```

Make sure the LLM label is one of the option names (normalise case/whitespace; map unknown replies to
`other` or skip the row; `jev-style eval` skips unreadable labels with a warning). Add a test that the new function returns the LLM result when the server is
unreachable.

## 5. Shadow, then measure

Turn shadow mode on (`JEV_SHADOW_LOG=...`) in staging or production for enough traffic: at least a few
hundred decisions per question. Then:

```sh
jev-style eval logs/route_shadow.jsonl --url http://127.0.0.1:8765
```

(`eval` ignores the extra `jev`/`ts` keys and re-asks the model, so the numbers reflect the current
server.) Read **agreement with the LLM** (`acc`) and `automate @1% / 5% / 10%`: the share of calls the
model can answer alone at that disagreement budget, and the confidence threshold for it. Agreement
with the LLM is not accuracy: if the user can label a sample by hand, evaluate on that too.

## 6. Switch, gradually

Set `THRESHOLD` to the threshold for the error budget the user accepts, turn shadow mode off, and ship.
Only calls above the threshold skip the LLM; the rest still use it. Report the expected share of LLM
calls saved (= the automation coverage) and keep the shadow log option for re-checking after prompt or
model changes. Never tune the threshold on the same rows you report as the final number.
