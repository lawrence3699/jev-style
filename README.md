# Jev-Style

Small, calibrated decision models you run on your own machine, plus the tooling to put them to work in AI agents.

<p>
  <a href="https://pypi.org/project/jev-style/"><img alt="PyPI" src="https://img.shields.io/pypi/v/jev-style?style=for-the-badge&labelColor=000000&color=0a0a0a" height="28"></a>
  <a href="https://github.com/lawrence3699/jev-style/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/lawrence3699/jev-style/ci.yml?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/collections/chaoliangUNSW/jev-style-08b-decision-v3-6ab58abb90ae4b7b55578b3e"><img alt="Weights: 0.8B · torch · MLX · GGUF" src="https://img.shields.io/badge/WEIGHTS-0.8B%20%C2%B7%20torch%20%C2%B7%20MLX%20%C2%B7%20GGUF-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/spaces/chaoliangUNSW/jev-style-v3"><img alt="Demo on Hugging Face Spaces" src="https://img.shields.io/badge/DEMO-HF%20Spaces-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="#agent-skills"><img alt="Agent skills: 6" src="https://img.shields.io/badge/AGENT%20SKILLS-6-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

![The Playground answering a support-ticket request, then the agent-approval demo allowing, asking about and denying tool calls](https://raw.githubusercontent.com/lawrence3699/jev-style/main/docs/demo.gif)

Jev-Style is a family of small decision models built on Qwen3.5. The current releases are [Jev-Style-2B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3) and [Jev-Style-0.8B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) (0.53 GB at 4-bit). You give a model text or JSON and some typed questions, and it returns a calibrated probability for every option in one forward pass. The server's API follows the public systemone request shape, so clients written for Jev-compatible servers can call your laptop instead.

This repository is the part that makes the model useful day to day:

- **`jev-style serve`**: a local API with a Playground and demos. It picks MLX on Apple silicon and PyTorch on CUDA or CPU; llama.cpp is optional.
- **Six agent skills**: install with one `npx skills add`. They serve the model, call it, evaluate it on your own labels, replace LLM calls that only return a label, add a guard to Claude Code, and add the MCP tools.
- **A Claude Code guard**: a `PreToolUse` hook where the local model checks every tool call before it runs and answers allow, ask or deny.
- **An MCP server**: tools `decide`, `noul`, `choice` and `score` for Claude Code, Cursor, Codex and any other MCP client.
- **`jev-style eval`**: measures accuracy and calibration on your own labelled data, and reports how many decisions you can automate at a 1, 5 or 10 % error budget.

No GPU, no API key and no training needed.

## Highlights

- **Three question types in one request.** Yes/no (`noul`), multiple choice (`choice`, up to 255 options) and ordered ratings (`score`, 2 to 10 levels). The model reads the text once and answers every question about it.
- **Probabilities, not just labels.** Each release ships temperatures fitted on held-out data, so your code can act on confident answers and send the rest to a person or an LLM.
- **Long inputs.** Up to 25,600 tokens per call. Nothing is truncated: an input that is too long is rejected with an error that says so.
- **Runs locally.** About 0.15 to 0.2 s for a short request to the 0.8B with MLX on an M1 Max, after the first call. Nothing leaves your machine.
- **51 languages evaluated.** Training covers 19 languages. On MASSIVE intent the 0.8B beats Laya's multilingual checkpoint in all 51 evaluated languages.
- **Built for agents.** The skills, the MCP tools and the guard all work with Claude Code, Codex, Cursor and other agents.

## Models

| Release (`--release`) | Parameters · smallest build | JevBench v1.4.1 public (231) | tweet_topic, zero-shot (1,693) | Context |
|---|---|:---:|:---:|:---:|
| [Jev-Style-2B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3) (`2b`) | 1.9B · 1.27 GB (Q4_K_M) | **73.6 %** | **82.2 %** | 25,600 tokens |
| [Jev-Style-0.8B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) (`0.8b`, default) | 0.8B · 0.53 GB (Q4_K_M) | 64.1 % | 75.5 % | 25,600 tokens |
| Hosted Jev 1.13, for reference | – | 86.6 % | 79.3 % | – |

The 2B numbers are single pre-declared runs with its GGUF F16 build; its [model card](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3) gives the protocols and confidence intervals. Hosted Jev is well ahead on JevBench. On tweet_topic the 2B's accuracy is above Jev's published number, but its macro-F1 is below (0.678 vs 0.694). The 2B was trained on a reduced data pool (60M tokens) and has no separate limit for the question and its options; everything counts toward the 25,600 tokens.

The 0.8B against Laya, on sets neither was trained on:

| Model | Banking77 (77 intents, never trained) | MASSIVE intent, 37 held-out languages | tweet_topic, zero-shot | JevBench v1.4.1 public (231) |
|---|:---:|:---:|:---:|:---:|
| [Jev-Style-0.8B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) | **68.2 %** | **65.5 %** | **75.5 %** | **64.1 %** |
| Best official Laya checkpoint (0.8B, 1,024 tokens by default) | 49.2 % | 36.1 % | 63.2 % | 58.4 % |

These numbers are from the [0.8B model card](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3#results), which gives the full protocol and confidence intervals. The Laya rows are its official checkpoints re-run on the same rows, except tweet_topic and JevBench, which use published numbers. The hosted Jev API has higher accuracy than the 0.8B on every one of these sets where its accuracy is published. Treat both releases as small local options, not replacements for the hosted model.

| Build | 0.8B | 2B | Used by |
|---|---:|---:|---|
| safetensors: [0.8B](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3), [2B](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3) | 1.50 GB | 3.76 GB | `--backend torch` (CUDA, Apple MPS, CPU) |
| MLX bf16 / 8-bit: [0.8B](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX), [2B](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX) | 1.50 / 0.80 GB | 3.76 / 2.00 GB | `--backend mlx` (Apple silicon; `auto` picks it there) |
| GGUF F16 / Q8_0 / Q4_K_M: [0.8B](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF), [2B](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF) | 1.52 / 0.81 / 0.53 GB | 3.78 / 2.01 / 1.27 GB | `--backend gguf` (llama.cpp through the release's scorer: `jev-score` for the 0.8B, `jev-score-v2` for the 2B) |

Each build carries its own runtime file next to the weights. The server downloads a pinned revision and uses that file, so the answers here match what the model card documents. Stock llama.cpp, Ollama, LM Studio or `mlx_lm.generate` can load the weights but cannot produce the decision scores. The 2B MLX runtime needs mlx-lm 0.31.3 exactly, which `jev-style[mlx]` installs. For long documents on the 2B, use Q8_0 (the default) or F16 rather than Q4_K_M. Earlier 2B generations, for use in LM Studio or Ollama without this server: [v1 GGUF](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF) (LM Studio, llama.cpp) and [v2 GGUF](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2-GGUF) (Ollama).

## Quick Start

### Try it in the browser

The [Hugging Face Space](https://huggingface.co/spaces/chaoliangUNSW/jev-style-v3) runs v3. There is nothing to install.

### Run it locally

You'll need Python 3.10 or newer.

```bash
pip install "jev-style[mlx]"      # Apple silicon (MLX)
pip install "jev-style[torch]"    # Linux, Windows or an Intel Mac (PyTorch on CUDA or CPU)

jev-style serve          # downloads the model once (~1.5 GB), then serves http://127.0.0.1:8765
jev-style serve --release 2b    # the 2B instead (a 2 to 3.8 GB download, depending on the build)
```

To install the command line tool on its own, with the MCP server, use [uv](https://docs.astral.sh/uv/): `uv tool install "jev-style[all]"` on Apple silicon, `uv tool install "jev-style[torch,mcp]"` elsewhere.

Open http://127.0.0.1:8765 for the Playground and the demos: agent action approval, Snake, and Chinese and 51 languages. In another terminal, send a ticket:

```bash
curl -s localhost:8765/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns":  "Exchanges, refunds, wrong or damaged items",
                                 "shipping": "Delivery status, delays, lost packages",
                                 "billing":  "Charges, invoices, payment problems"}},
    "escalate":    {"type": "noul",  "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }}'
```

This is the actual response from v3 with MLX on an Apple M1 Max, with probabilities rounded:

```json
{
  "model": "jev-style-0.8b-decision-v3",
  "answers": {
    "department":  { "type": "choice", "choice": "billing", "confidence": 0.46,
                     "probabilities": { "returns": 0.07, "shipping": 0.29, "billing": 0.64 } },
    "escalate":    { "type": "noul", "noul": 0.45 },
    "frustration": { "type": "score", "score": 1.28, "confidence": 0.36,
                     "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
                     "probabilities": { "0": 0.07, "1": 0.57, "2": 0.35 } }
  },
  "usage": { "input_tokens": 250, "state_tokens": 25, "output_tokens": 0 },
  "latency_ms": 194.0,
  "backend": "mlx"
}
```

The ticket raises both a billing problem and a late delivery, and the probabilities show it: billing 0.64, shipping 0.29. That is why the model returns probabilities rather than a single label. Your code can act on the confident answers and hand the rest to a person.

### Use it from Python

For a script, one line is enough. The first call loads the model in-process, or uses the server in `JEV_STYLE_URL` if you set it:

```python
import jev_style

jev_style.classify("Where is my parcel? It was due Monday.", ["billing", "shipping", "tech"])
# {'label': 'shipping', 'confidence': ..., 'probabilities': {'billing': ..., 'shipping': ..., 'tech': ...}}
```

For several questions about one text, and to choose the engine yourself:

```python
from jev_style import JevStyle, choice, noul, score

js = JevStyle.from_pretrained("chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX")   # in-process; the repo picks the backend
# js = JevStyle(release="2b")                                                    # the 2B, backend chosen for this machine
# js = JevStyle(base_url="http://127.0.0.1:8765")                                # or a running server
out = js.decide("I was charged twice. Please fix this ASAP.", {
    "billing": noul("This ticket is about billing."),
    "tone":    choice("What is the customer's tone?", ["calm", "frustrated", "angry"]),
    "urgency": score("How urgent is this ticket?", ["can wait", "this week", "today"]),
})
print(out["answers"]["billing"]["noul"], out["answers"]["tone"]["choice"])
```

`from_pretrained` takes the main repo (PyTorch), `-MLX` (`precision="8bit"` for the 0.8 GB weights) or `-GGUF` (`quant="Q4_K_M"`, needs the `jev-score` scorer, see [Backends](#backends)). The client is not tied to this model: `JevStyle(base_url=...)` works with any server that implements `POST /v1/systemone`, and a client written for another systemone-compatible server can call `http://127.0.0.1:8765` with any API key string.

### From the shell

```bash
jev-style decide "Refund still missing after 3 weeks" --url http://127.0.0.1:8765 \
  --choice "Which team?::billing,shipping,returns" --noul "The customer is angry"
# without --url (or JEV_STYLE_URL) it loads its own copy of the model
```

## Agent Skills

The skills live in [`skills/`](https://github.com/lawrence3699/jev-style/tree/main/skills). Each one tells a coding agent how to do one job from start to finish, and ends with a check that the job worked.

```bash
npx skills add lawrence3699/jev-style                    # pick skills interactively
npx skills add lawrence3699/jev-style --skill '*' -y     # all six
```

| Skill | What your agent does with it |
|---|---|
| [`jev-style-serve`](https://github.com/lawrence3699/jev-style/blob/main/skills/jev-style-serve/SKILL.md) | Installs the CLI, picks the backend for your machine, starts the server, checks it with a real request, and can optionally start it at login. |
| [`jev-style`](https://github.com/lawrence3699/jev-style/blob/main/skills/jev-style/SKILL.md) | Writes code that calls the model: request format, reading the probabilities, turning them into actions with thresholds, limits. |
| [`jev-style-eval`](https://github.com/lawrence3699/jev-style/blob/main/skills/jev-style-eval/SKILL.md) | Builds a labelled JSONL from your data and runs `jev-style eval`. It reports accuracy, Brier, ECE, how much you can automate at a 1, 5 or 10 % error budget, and a refitted temperature. |
| [`jev-style-adopt`](https://github.com/lawrence3699/jev-style/blob/main/skills/jev-style-adopt/SKILL.md) | Scans your code for LLM calls that only return a label, yes/no or a rating, and rewrites them as typed questions. The LLM stays as the fallback below a confidence threshold. It runs in shadow mode, measures agreement, then switches. |
| [`jev-style-guard`](https://github.com/lawrence3699/jev-style/blob/main/skills/jev-style-guard/SKILL.md) | Installs the Claude Code guard hook. It recommends a dry run first, then shows how to tune thresholds and measure them on labelled calls. |
| [`jev-style-mcp`](https://github.com/lawrence3699/jev-style/blob/main/skills/jev-style-mcp/SKILL.md) | Registers the MCP tools with Claude Code, Codex, Cursor, Claude Desktop or Windsurf, and checks that a tool call works. |

In Claude Code you can also install everything as plugins:

```text
/plugin marketplace add lawrence3699/jev-style
/plugin install jev-style@jev-style          # the six skills + MCP tools
/plugin install jev-style-guard@jev-style    # the PreToolUse guard
```

## Claude Code Guard

Before Claude Code runs a `Bash`, `Write`, `Edit`, `Read`, `WebFetch` or MCP call, `jev-style guard` sends the call to the local model. It asks whether the call is destructive, exfiltrates data, touches secrets or goes outside the project, plus a 0 to 4 risk score. It turns the answers into **allow / ask / deny**. Regex hard rules in code can only make a verdict stricter. If the server is down, times out or rejects the request, the verdict is **ask**, never a silent allow.

```bash
jev-style guard --check "git push --force origin main"
# decision: DENY   (source: rule, 443 ms)  destructive 0.79 ...
jev-style guard-replay --url http://127.0.0.1:8765     # the 49 bundled labelled tool calls
```

On the 49 bundled tool calls, which were written and labelled by hand, the default config agrees with the labels on **77.6 %**. **No call labelled deny was allowed.** 2 of the 16 calls labelled ask were allowed. The model alone, with no hard rules, agrees on 61.2 %. Use it as a second line of defence, not as a sandbox. The [skill](https://github.com/lawrence3699/jev-style/blob/main/skills/jev-style-guard/SKILL.md) covers installing, dry runs and tuning.

## MCP Server

```bash
claude mcp add jev-style --scope user -- jev-style mcp       # Claude Code
```

`jev-style mcp` is a thin stdio server that forwards to the running `jev-style serve`, so one copy of the model serves every client. It provides `decide` (several questions about one input), `noul`, `choice`, `score` and `model_info`. Setup for Codex, Cursor and Claude Desktop is in [the skill](https://github.com/lawrence3699/jev-style/blob/main/skills/jev-style-mcp/SKILL.md).

## Evaluate on Your Own Data

The best evidence is your own labelled examples. `jev-style eval` takes JSONL in the request format, with a `label` on each question:

```bash
jev-style eval examples/support_tickets.jsonl --url http://127.0.0.1:8765
```

```
question                   n     acc   brier    ece   automate @1% / 5% / 10% error
team                      30   93.3%   0.077  0.123    90.0% (p>=0.65) /  93.3% (p>=0.57) / 100.0% (p>=0.44)
escalate                  30   96.7%   0.107  0.125    80.0% (p>=0.71) / 100.0% (p>=0.54) / 100.0% (p>=0.54)
mood                      30   73.3%   0.433  0.174    40.0% (p>=0.56) /  40.0% (p>=0.56) /  46.7% (p>=0.54)
```

The 30 example tickets were written by hand for this repository. They demonstrate the format; they are not a benchmark. Read `automate @5%: 93.3% (p>=0.57)` like this: if you act only when the top probability is at least 0.57, the model handles 93.3 % of tickets with at most 5 % errors among them, and everything else goes to a person or an LLM. The report also refits a temperature on half the rows and shows its effect on the other half. See the [skill](https://github.com/lawrence3699/jev-style/blob/main/skills/jev-style-eval/SKILL.md) for building a proper evaluation set.

### Compare engines

Repeat `--server` to run the same file through several engines: this package's local model, another build or quantisation, or any other server that implements `POST /v1/systemone`. The first one is the reference.

```bash
jev-style eval examples/support_tickets.jsonl --server bf16=local:mlx --server q8=http://127.0.0.1:8799
```

```
90 questions answered by every engine; the table and the differences use only those.

engine             answered failed rows     acc   brier log loss    ece   automate @5% error
bf16                     90           0   87.8%   0.206    0.369  0.125    80.0% (p>=0.54)
q8                       90           0   87.8%   0.206    0.370  0.117    78.9% (p>=0.54)

difference to bf16 (95 % paired bootstrap over rows; brier: lower is better):
  q8               accuracy  +0.0 pts [+0.0, +0.0]   brier +0.001 [-0.001, +0.002]
```

That run compares the MLX bf16 weights in-process with `jev-style serve --precision 8bit` on an Apple M1 Max. Engine specs are `NAME=URL`, `NAME=local[:backend]`, `NAME=hf:<repo id>` or `NAME=fake`; `--key NAME=ENV_VAR` sends a bearer token from an environment variable and `--model NAME=MODEL` sets the request's `model` field. Every engine is scored on the same questions: if one engine rejects a row (too long, too many options), that row is dropped for all of them and counted under `failed rows`. `--json PATH` writes every answer.

## API

`POST /v1/systemone` with `{"state": text | object | array, "questions": {id: question}}`:

| `type` | `criteria` | Answer |
|---|---|---|
| `noul` | optional `{"true": "...", "false": "..."}` | `noul` = P(true) |
| `choice` | `{option: description or null}`, 1 to 255 options | `choice`, `probabilities`, `confidence` |
| `score` | 2 to 10 levels, lowest first: `"label"` or `{"label", "description"}` | `score` = expected level index, `legend`, `probabilities`, `confidence` |

`confidence = (k · p_max − 1) / (k − 1)` for k options: 0 when the probabilities are uniform, 1 when one option takes all of them. The other routes are `GET /v1/models`, `GET /healthz` and the Playground at `/`. Errors look like `{"error": {"code", "message", "question"?}}` and use HTTP 422 (`invalid_json`, `invalid_request`, `invalid_question`, `input_budget_exceeded`), 401 (`unauthorized`), 404 or 500. Start the server with `--api-key-env NAME` to require `Authorization: Bearer <key>`. The full reference is [skills/jev-style/reference.md](https://github.com/lawrence3699/jev-style/blob/main/skills/jev-style/reference.md).

## Backends

| Machine | Command |
|---|---|
| Apple silicon | `jev-style serve` (MLX bf16; add `--precision 8bit` for 0.8 GB) |
| NVIDIA GPU | `jev-style serve --backend torch` |
| CPU only | `jev-style serve --backend torch --device cpu` |
| llama.cpp | build the release's scorer once with `sh build_jev_score.sh /path/to/llama.cpp` from its GGUF repo (`jev-score` for the 0.8B, `jev-score-v2` for the 2B), then `jev-style serve --backend gguf --scorer /absolute/path/printed/by/the/script --quant Q4_K_M` (add `--release 2b` for the 2B) |
| Offline | `jev-style serve --model-dir /path/to/a/downloaded/model/repo` |
| No model (UI or plumbing work) | `jev-style serve --fake` (deterministic, meaningless answers) |

## Repository Layout

```
jev_style/           server, client, CLI, guard, MCP server, eval; web/ = Playground + demos
skills/              six agent skills (npx skills add lawrence3699/jev-style)
plugins/             Claude Code guard plugin (the root .claude-plugin/ is the marketplace)
examples/            labelled example data for jev-style eval
space/               source of the Hugging Face Space
tests/               pytest suite, runs on the fake engine (no model download)
```

## Development

```bash
git clone https://github.com/lawrence3699/jev-style && cd jev-style
uv venv && uv pip install -e ".[dev]"
uv run pytest -q                # fake engine: no weights, no GPU
uv run jev-style serve --fake   # UI work without the model
```

## Acknowledgements and License

Code: Apache-2.0 ([LICENSE](https://github.com/lawrence3699/jev-style/blob/main/LICENSE)). The weights are Apache-2.0 fine-tunes of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B); the NOTICE in each model repository lists the changes. The typed-question convention follows [Laya](https://github.com/NandhaKishorM/laya).

Not affiliated with, endorsed by or connected to TypeSafe or Jev. "Jev-Style" describes the kind of model: a small typed-decision model in a similar style. No Jev weights, code or outputs are included. Not affiliated with Alibaba Cloud or the Qwen team or the Laya authors.
