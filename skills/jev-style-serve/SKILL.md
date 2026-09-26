---
name: jev-style-serve
description: Install and start the Jev-Style local decision server (systemone-compatible API + Playground at http://127.0.0.1:8765) on macOS, Linux or Windows/WSL, pick the right backend (MLX on Apple silicon, PyTorch CUDA/CPU elsewhere, llama.cpp GGUF optional), verify it answers, and optionally keep it running at login. Use when the user wants to run Jev-Style, set up a local decision model / systemone endpoint, or when another jev-style skill finds no server on :8765.
---

# Run Jev-Style locally

Goal: a working `http://127.0.0.1:8765/v1/systemone` with the real model, verified by a request.
Work through the steps in order and report what you did. Ask before anything that changes the user's
login items or installs system packages.

## 1. Check what is already there

```sh
curl -s http://127.0.0.1:8765/healthz || echo "no server"
command -v jev-style && jev-style --version
command -v uv || echo "no uv"
uname -sm; python3 --version
```

If `healthz` already answers with `"model": "jev-style-0.8b-decision-v3"`, stop here: it is running.

## 2. Install the CLI

Prefer `uv` (installs into its own environment, puts `jev-style` on PATH):

```sh
# Apple silicon (MLX backend, fastest on a Mac) + PyTorch fallback + MCP server:
uv tool install "jev-style[all]"
# Linux / Windows / Intel Mac (PyTorch on CUDA or CPU) + MCP server:
uv tool install "jev-style[torch,mcp]"
```

No `uv`? Install it (`curl -LsSf https://astral.sh/uv/install.sh | sh`, ask first) or use
`pip install "jev-style[all]"` inside a venv.
Needs Python 3.10+.

## 3. Pick the backend

| Machine | Backend | Flag |
|---|---|---|
| Apple silicon with `mlx` installed | MLX, bf16 weights (default `auto`) | `--backend mlx` (`--precision 8bit` for less memory) |
| NVIDIA GPU | PyTorch CUDA | `--backend torch` |
| anything else | PyTorch CPU (slower, still works) | `--backend torch --device cpu` |
| llama.cpp users | GGUF via the `jev-score` scorer | `--backend gguf --scorer /path/to/jev-score` |

`auto` picks MLX on Apple silicon when installed, else PyTorch. The GGUF backend needs the `jev-score`
binary built once against llama.cpp (`sh build_jev_score.sh /path/to/llama.cpp`, shipped in the
`chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF` repo); only set it up if the user asks for llama.cpp.

## 4. Download and start

```sh
jev-style download                  # ~1.5 GB once, into the Hugging Face cache (optional; serve does it too)
jev-style serve                     # http://127.0.0.1:8765  (--port N, --backend ..., --api-key-env NAME)
```

Run `serve` in the background or a separate terminal, since it keeps running. Startup takes a few
seconds after the first download. Bind to `127.0.0.1` (the default); only use `--host 0.0.0.0` together
with `--api-key-env` if the user explicitly wants other machines to reach it.

## 5. Verify with a real request

```sh
curl -s http://127.0.0.1:8765/healthz
curl -s http://127.0.0.1:8765/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for my subscription this month.",
  "questions": {"billing": {"type": "noul", "instructions": "This ticket is about billing."}}}'
```

Expect `"model": "jev-style-0.8b-decision-v3"` and a `noul` close to 1. Then point the user at the
Playground: http://127.0.0.1:8765/ (also demos: agent action approval, Snake, 51 languages).

## 6. Optional: start at login

Only if the user asks. macOS: write `~/Library/LaunchAgents/com.jev-style.serve.plist` running
`$(command -v jev-style) serve` with `RunAtLoad` and `KeepAlive`, then
`launchctl load ~/Library/LaunchAgents/com.jev-style.serve.plist`. Linux: write `~/.config/systemd/user/jev-style.service`
(`[Service] ExecStart=%h/.local/bin/jev-style serve`, `Restart=on-failure`; `[Install] WantedBy=default.target`),
then `systemctl --user enable --now jev-style`.

Settings can also come from the environment: `JEV_STYLE_BACKEND`, `JEV_STYLE_PORT`, `JEV_STYLE_DEVICE`,
`JEV_STYLE_MODEL_DIR`.

## Troubleshooting

- `There is no Stream(gpu, ...)` or MLX errors: update (`uv tool upgrade jev-style`), or use `--backend torch`.
- `transformers.models.qwen3_5` missing: the torch backend needs `transformers>=5.0`.
- Port in use: `--port 8766` (or `JEV_STYLE_PORT`), then point the other tools at it: `JEV_STYLE_URL=http://127.0.0.1:8766`
  for `jev-style decide/eval/mcp` and the guard (or `server_url` in `~/.config/jev-style/guard_config.json`),
  `--url` for `jev-style guard-replay`, and `JevStyle(base_url=...)` in Python.
- No network for the download: copy a model folder and use `--model-dir /path/to/Jev-Style-0.8B-Decision-v3`.
- Just testing plumbing without the 1.5 GB download: `jev-style serve --fake` (answers are meaningless).
