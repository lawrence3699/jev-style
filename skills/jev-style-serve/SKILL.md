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

If `healthz` already answers with a `"model"` that starts with `jev-style-` (`jev-style-0.8b-decision-v3` is the
default release, `jev-style-2b-decision-v4` the 2B), stop here: it is running. If it says `jev-style-fake`, the
server runs the fake engine (meaningless answers): restart it without `--fake`.

## 2. Install the CLI

Prefer `uv` (installs into its own environment, puts `jev-style` on PATH):

```sh
# Apple silicon (MLX backend, fastest on a Mac; text only):
uv tool install "jev-style[mlx]"
# Linux / Windows / Intel Mac (PyTorch on CUDA or CPU), and for images on any machine:
uv tool install "jev-style[torch]"
```

The MCP server is part of every install. The bare `pip install jev-style` is only the client. No `uv`? Ask first,
then install it with a package manager (`brew install uv` or `pipx install uv`; other options:
https://docs.astral.sh/uv/getting-started/installation/), or use `pip install "jev-style[mlx]"` /
`"jev-style[torch]"` inside a venv of its own (`[mlx]` pins mlx-lm to 0.31.3, which the 2B MLX runtimes require).
Needs Python 3.10+. Native Windows runs the PyTorch backend; the Claude Code hook commands are POSIX shell, so
the guard needs Git Bash or WSL there.

## 3. Pick the backend

| Machine | Backend | Flag |
|---|---|---|
| Apple silicon with `mlx` installed | MLX, bf16 weights (default `auto`) | `--backend mlx` (`--precision 8bit` for less memory) |
| NVIDIA GPU | PyTorch CUDA | `--backend torch` (`--dtype bfloat16` for the 2B v4, as it was measured) |
| anything else | PyTorch CPU (slower, still works) | `--backend torch --device cpu` |
| llama.cpp users | GGUF via the release's scorer | `--backend gguf --scorer /path/to/<scorer>` |

`auto` picks MLX on Apple silicon when installed, else PyTorch. The model is chosen with `--release`
(`JEV_STYLE_RELEASE` sets it for every command): the default `0.8b` (= `0.8b-v3`, 1.5 GB), or `2b`
(= `2b-v4`, 2.2B parameters, 32,768 tokens; reads images with `--backend torch` only, 4.4 GB; MLX 3.8 GB, 2 GB
at `--precision 8bit`), or `2b-v3`. Users who ask for "the 2B", "the newest", or "the one that reads
images" want `--release 2b` (images: `--release 2b --backend torch`, also on a Mac).

The GGUF backend needs a scorer binary built once against llama.cpp: `jev-score` for the 0.8B (sources in
`chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF`), `jev-score-v2` for both 2Bs (sources in the
`Jev-Style-2B-Decision-v3-GGUF` / `-v4-GGUF` repos; one binary serves both). `jev-style download --backend gguf
[--release 2b]` prints the model folder; run `sh build_jev_score.sh /path/to/llama.cpp` there and pass the
path it prints as `--scorer` (or set `JEV_SCORE_BIN` / `JEV_SCORE_V2_BIN`). `serve` refuses to start the download
while no scorer can be found. Only set this up if the user asks for llama.cpp.

## 4. Download and start

```sh
jev-style download                  # ~1.5 GB once, into the Hugging Face cache (optional; serve does it too)
jev-style serve                     # http://127.0.0.1:8765  (--port N, --backend ..., --api-key-env NAME)
jev-style download --release 2b     # the 2B v4 instead (then: jev-style serve --release 2b [--backend torch])
```

`download` refuses to fetch anything the installed backend cannot run (`error: ... pip install
"jev-style[torch]"`), checks every file against the repo's manifest, and prints a one-line `error:` when
the download fails (network, `HF_HUB_OFFLINE`); there is no traceback to parse.

Run `serve` in the background or a separate terminal, since it keeps running. Startup takes a few
seconds after the first download. Bind to `127.0.0.1` (the default); only use `--host 0.0.0.0` together
with `--api-key-env` if the user explicitly wants other machines to reach it. The server answers only
requests addressed to loopback names (or its `--host`) and refuses browser requests from other sites, so a
web page elsewhere cannot use it; `--allow-any-host` lifts that if the user has a reason.

## 5. Verify with a real request

```sh
curl -s http://127.0.0.1:8765/healthz
curl -s http://127.0.0.1:8765/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for my subscription this month.",
  "questions": {"billing": {"type": "noul", "instructions": "This ticket is about billing."}}}'
```

Expect a `"model"` starting with `jev-style-` (`jev-style-0.8b-decision-v3` by default,
`jev-style-2b-decision-v4` with `--release 2b`) and a `noul` close to 1. Then point the user at the
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
- `needs mlx-lm 0.31.3 exactly`: another tool changed mlx-lm in the same environment; give jev-style its own
  (`uv tool install`) rather than downgrading the shared one.
- `the mlx backend ... only runs on Apple silicon`: Intel Mac, Linux or Windows: use `--backend torch`.
- `transformers.models.qwen3_5` missing: the torch backend needs `transformers>=5.0`.
- Port in use: `--port 8766` (or `JEV_STYLE_PORT`), then point the other tools at it: `JEV_STYLE_URL=http://127.0.0.1:8766`
  for `jev-style decide/eval/mcp` and the guard (or `server_url` in `~/.config/jev-style/guard_config.json`),
  `--url` for `jev-style guard-replay`, and `JevStyle(base_url=...)` in Python.
- No network for the download: copy a model folder and use `--model-dir /path/to/Jev-Style-0.8B-Decision-v3`
  (`--verify` checks it against the folder's manifest at load time).
- Just testing plumbing without the 1.5 GB download: `jev-style serve --fake` (answers are meaningless).
