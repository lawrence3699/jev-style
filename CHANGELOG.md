# Changelog

Versions follow [semantic versioning](https://semver.org/). Each release pins the model revisions it loads
(`jev_style/models.py`), so upgrading the package is what changes the weights you get.

## 0.4.0 (unreleased)

<!-- TODO(0.4.0): the release date, and the two new torch revisions below -->

- **CUDA graphs for the PyTorch backend, on by default.** On CUDA the torch runtime records CUDA graphs at start-up
  and replays them. Measured on an RTX 5090 in float32 on 4,992 calibration questions against the ordinary path:
  0 top-1 answers changed, largest probability difference 0.00086 (0.8B) and 0.0024 (2B), median latency 41.6 ->
  11.3 ms (0.8B) and 86.2 -> 13.9 ms (2B). Recording the graphs adds about 7-11 s to start-up; inputs longer than
  4,096 tokens and several questions over a long shared state use the ordinary path. `--cuda-graphs auto|on|off`
  (auto = on when the model runs on CUDA), `$JEV_STYLE_CUDA_GRAPHS`, `cuda_graphs=` in `JevStyle` and
  `models.load_release`. A runtime without the CUDA-graph path loads as before under `auto`; `on` is refused there.
- **New torch pins.** The PyTorch builds of `0.8b-v3` and `2b-v3` are pinned to revisions that ship the CUDA-graph
  runtime (TODO). The MLX and GGUF pins are unchanged.
- **Confidence cascades** (`jev_style.cascade`). A `cascade.json` lists tiers (smallest first: an in-process release,
  a Hub repo, or any `POST /v1/systemone` server) and one threshold per tier below the top. Tier 1 answers every
  question in one call; questions below its threshold (normalized p_max computed from the probabilities, never the
  tier's own `confidence`), or that it failed on, go to the next tier in one call with the same state. The top tier
  is final; if it fails, the last successful answer is used; if every tier fails, the error is returned as a single
  model would return it (plus `tier`, `tier_model`, `tier_errors`). Optional per-tier `max_options`. Answers add
  `tier` and `tier_model`, `timing.tiers` reports each tier; `model` is the cascade id, `backend` is `cascade`.
  `select()` replays recorded per-tier answers offline with the same rules.
- `jev-style serve --cascade FILE`, `jev-style decide --cascade FILE`, `JevStyle(cascade=FILE)`, and the eval engine
  `NAME=cascade:FILE`.
- **JevK5 tiers.** `"protocol": "jevk5"` talks to allebee/jevk5's `jevk5-serve` (0.3.x): structured option
  descriptions are sent as text, single-option choices are not sent (it rejects them), its `{"error": "<message>"}`
  bodies and its answer shapes (no `legend`, `confidence` = p_max) are read correctly. `examples/cascade/` holds a
  2B -> JevK5-9B and a 0.8B -> 2B -> JevK5-9B file (placeholder thresholds); `scripts/cascade_smoke.py` runs one on
  fake engines.
- **In-process JevK5 tiers**: the target `jevk5:<Hub repo or local folder>` runs allebee/jevk5 in this process, with
  the code `jevk5-serve` runs (`jevk5.server.normalized`, then `model.decide`; one bad question fails the request as
  its HTTP 400 does), so a tier answers the same in-process and over HTTP. A Hub repo is downloaded at its revision
  (pinned for JevK5-9B), checked against its `SHA256SUMS` (by default for the pinned release; tier key `verify`), and
  refused without its `jevk5_config.json`, without which JevK5 would run uncalibrated. All JevK5 calls run on one
  worker thread. jevk5 is not on PyPI and is not a dependency: a missing install fails before any download and
  names the command, `pip install "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.3.3"`. `jevk5:` also works
  as an eval engine.
- **Named cascades** (`jev_style.cascades.CASCADES`): `--cascade`, `JevStyle(cascade=...)` and `cascade:` eval
  engines take a name as well as a file. **`cascade-9b`** (`jev-style-cascade-9b`): Jev-Style 2B Decision v3
  in-process (PyTorch float32, CUDA graphs) -> JevK5-9B v0.3.3 by alibiserikbay (third party, Apache-2.0) in-process,
  one threshold on normalized confidence, calibration set sha256 `580962e4...`. A named cascade whose threshold is
  still a placeholder is refused. `/v1/models` reports the cascade id and its tiers.
- `jev-style releases [--json]` lists the model releases (with their pinned revisions) and the named cascades.
  `jev-style download --cascade NAME|FILE` downloads every tier ahead of time (the JevK5 snapshot is checked as a
  load checks it; jevk5 itself is not needed for that).
- `--cuda-graphs` now also applies to a cascade's local tiers, like `--device` and `--dtype`.
- `jev_style.client.parse_target` / `from_target`: one resolver for engine targets, shared by `eval --server` and the
  cascade tiers. The HTTP client now reports `{"error": "<message>"}` bodies by their message.

## 0.3.0 (2026-09-27)

- **Jev-Style-2B-Decision-v3.** `--release 2b` (or `JevStyle(release="2b")`, or `from_pretrained` with any of its
  three repos) loads the new 2B at pinned revisions: PyTorch, MLX (bf16 / 8-bit; needs mlx-lm 0.31.3) and GGUF
  (F16 / Q8_0 / Q4_K_M; scorer `jev-score-v2`, found through `--scorer` or `$JEV_SCORE_V2_BIN`). The default stays
  the 0.8B.
- `[mlx]` now pins mlx-lm 0.31.3 (the 2B MLX runtime checks it; the 0.8B was tested on it). `auto` skips an MLX
  build whose requirements are not met.
- **Several model releases.** `jev_style.models.RELEASES` lists each release with its three builds (PyTorch main repo,
  `-MLX`, `-GGUF`). Pick one with `--release` (`serve`, `decide`, `download`, `mcp`), `JevStyle(release=...)`,
  `$JEV_STYLE_RELEASE`, an eval engine `local:<release>[:backend]`, or `JevStyle.from_pretrained(<any of its repos>)`.
  Only releases whose builds are all pinned are offered by the CLI.
- The server reports the loaded release: `model` in answers, `/v1/models` and `/healthz` (id, release date,
  description) come from the release instead of being fixed to the 0.8B model.
- MLX prefix sharing is enabled per build, only where it was shown to give identical scores (0.8B v3 MLX).
- A release that is not pinned yet can still be loaded from a local folder (`model_dir`), or from its main branch
  with `trust_remote_code=True`.
- `models.load_release()` returns the build as well; `models.load()` keeps its 0.2 return value.
- **Fail fast without a backend.** With only `pip install jev-style` (the client), loading the model used to download
  1-2 GB and then stop at `No module named 'torch'`. It now stops before the download with `MissingBackendError`,
  which names the extra to install (`[mlx]` on Apple silicon, `[torch]` elsewhere); the CLI prints it as one line.
- **MCP Registry.** Listed as `io.github.lawrence3699/jev-style` (`server.json` at the repo root). `mcp` is now a core
  dependency, so `uvx jev-style mcp` works without extras; `jev-style[mcp]` still resolves. The README carries the
  registry's `mcp-name` ownership line, and `release.yml` publishes `server.json` to the registry after the PyPI job
  (GitHub OIDC, no token; the version is taken from the tag).
- The MCP server reports the package version (it said 0.1.0).

## 0.2.0 (2026-09-26)

- **On PyPI.** `pip install "jev-style[mlx]"` on Apple silicon, `pip install "jev-style[torch]"` elsewhere; the
  README and the skills no longer install from a git URL.
- **`JevStyle.from_pretrained(repo_id)`**: load a release by its Hub repo id; the repo picks the backend (main repo =
  PyTorch, `-MLX`, `-GGUF`). The three v3 repos load at their pinned revisions. Any other repo runs the runtime file it
  ships, so it is refused unless you pass `trust_remote_code=True`.
- **Module-level shortcuts**: `jev_style.decide(...)`, `jev_style.classify(text, labels)` and `jev_style.configure(...)`
  share one lazily created client (`$JEV_STYLE_URL`, else the local model).
- **`jev-style eval --server ...` compares engines**: the local model, other builds, or any server that implements
  `POST /v1/systemone`, on the same questions, with 95 % paired bootstrap intervals for the accuracy and Brier
  differences. `--key NAME=ENV_VAR` and `--model NAME=MODEL` per engine.
- MLX: the state is read once for many questions (bit-identical answers, up to 8x faster).
- Packaging: version read from `jev_style/__init__.py`; the source distribution now holds only the package, tests
  and examples (about 210 KB instead of 6.6 MB); README links are absolute so they work on PyPI.

No change to answers: the pinned model revisions are the same as in 0.1.0.

## 0.1.0 (2026-09-26)

First release: `jev-style serve` (systemone-compatible API, Playground and demos; MLX, PyTorch and GGUF backends),
the Python client, six agent skills, the Claude Code guard, the MCP server and `jev-style eval`.
