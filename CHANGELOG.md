# Changelog

Versions follow [semantic versioning](https://semver.org/). Each release pins the model revisions it loads
(`jev_style/models.py`), so upgrading the package is what changes the weights you get.

## 0.6.0 (2026-10-28)

- **Jev-Style-2B-Decision-v4** (`--release 2b-v4`), all three builds pinned: torch @ c9504148, MLX @ a14bbe4d,
  GGUF @ 9111d5d2. The first Jev-Style model that reads images (PyTorch build; the MLX and GGUF builds read text
  only), 32,768 tokens per call. Its PyTorch runtime ships the reference scorer it was evaluated with (`src/macjev/`)
  and needs the image processor, so the `torch` extra now also installs `pillow` and `torchvision`. Its MLX build
  needs mlx-lm 0.31.3 exactly and its GGUF build the `jev-score-v2` scorer, as for `2b-v3` (one binary serves both).
  Trained in part on soft labels distilled from TypeSafe Jev 1.14 via its public API (see its model card).
- **`--release 2b` now means `2b-v4`** (`JevStyle(release="2b")`, `JEV_STYLE_RELEASE=2b`, `local:2b` targets). Use
  `2b-v3` to keep the v3 weights; `cascade-9b` and the example cascades name `2b-v3` explicitly and are unchanged.
  The default release is still `0.8b-v3`.
- **Images in requests, the way TypeSafe's API takes them** (Jev 1.13 / 1.14): `assets` with `data_base64`, or
  multipart bodies with one file part per image, plus the header `TypeSafe-Danger-Zone-Images: true`. A string value
  in `state` that names an asset becomes the image (`<|image_k|>` in reading order); a reference in a question's
  instructions or criteria is refused (this model reads images as part of the state). TypeSafe's limits: still PNG
  or JPEG, 16 MiB per image, 280 to 1,568 px per side, 32 MiB in total. Models that read text only (0.8B, 2B v3,
  every MLX and GGUF build) keep answering 400 `images_not_supported`. `GET /v1/models` reports `images` and `box`.
- **Python client:** `JevStyle.decide(..., assets={...})` and `jev_style.decide(..., assets=...)` send images
  (in-process, or over HTTP with the header), and `jev_style.image(path_or_bytes)` builds an `assets` entry.
- **CUDA graphs under `auto`:** a runtime whose own `cuda_graphs` default is auto (the 2B v4) now makes the choice
  itself: graphs with bfloat16 on CUDA, none in float32, where its forward is compute-bound and recording costs about
  a minute for little gain. The v3 runtimes keep 0.4.0's behaviour (on whenever they run on CUDA).
  `--cuda-graphs on|off`, `JEV_STYLE_CUDA_GRAPHS` and `cuda_graphs=` still override it; `JEV_STYLE_CUDA_GRAPHS=auto`
  now means auto.
- **Downloads:** a cached snapshot of the pinned revision is used without a network call only when it holds every
  file the build needs (checked against its `manifest.json`). With huggingface_hub before 2.x, a snapshot holding
  only some of them (another MLX precision or GGUF quant, or one file fetched with `hf_hub_download`) was used as
  is and loading failed with `jev_style_decision*.py not found` or a missing weights file.
- `jev-style releases` no longer prints an empty alias field for a release without an alias.

## 0.5.0 (2026-10-17)

Compatibility with TypeSafe's API as of Jev 1.14 and the official SDKs (Python 0.7.5, JavaScript 0.7.0).

- **Compressed request bodies.** The official SDKs now compress bodies of 1 KiB or more with zstd by default, which
  older servers rejected with 422 `invalid_json`. `POST /v1/systemone` accepts one layer of
  `Content-Encoding: gzip`, `x-gzip` or `zstd` (window up to 8 MiB). Checked end to end with `typesafe-sdk` 0.7.5
  and its default compression. New core dependency: `zstandard`.
- **Request rules as in TypeSafe's API.** Body over 32 MiB (compressed or decompressed): 413 `payload_too_large`.
  Unsupported or multi-layer `Content-Encoding`: 415 `unsupported_content_encoding`. 400 for a malformed compressed
  body (`invalid_compressed_body`), a byte order mark or invalid UTF-8 (`invalid_encoding`), more than 64 nested
  levels in `state`, `instructions` or `criteria` (`nesting_too_deep`; very deep bodies used to fail with 500), and
  NaN, Infinity or numbers too large to represent (`non_finite_number`). Validation errors stay 422.
- **Over-long inputs: 400 `max_tokens_exceeded`** (was 422 `input_budget_exceeded`), the code TypeSafe returns.
  The Playground understands both.
- **Score `confidence` uses TypeSafe's ordinal formula** (docs.typesafe.ai/confidence):
  `max(0, 1 - sum_i p_i |i - m| / MAD_unif)`, so probability on a neighbouring level costs less than probability
  far away. It was the Choice formula. Probabilities, `score` and Choice confidence are unchanged; cascade routing
  still uses normalized p_max, so frozen thresholds keep their meaning.
- **Text only, said clearly.** `assets` (images), multipart bodies and `box` questions get 400
  `images_not_supported` / `unsupported_question_type` instead of a validation error.
- `GET /v1/models` always gives `release_date` as a string (the official SDKs refused `null` from the fake engine
  and cascades); a cascade reports the date its routing was frozen.
- Official model names (`jev-latest`, `jev-preview`, `jev-1.14.0`, ...) keep working in the `model` field.

## 0.4.0 (2026-10-03)

- **CUDA graphs for the PyTorch backend, on by default.** On CUDA the torch runtime records CUDA graphs at start-up
  and replays them. Measured on an RTX 5090 in float32 on 4,992 calibration questions against the ordinary path:
  0 top-1 answers changed, largest probability difference 0.00086 (0.8B) and 0.0024 (2B), median latency 41.6 ->
  11.3 ms (0.8B) and 86.2 -> 13.9 ms (2B). Recording the graphs adds about 7-11 s to start-up; inputs longer than
  4,096 tokens and several questions over a long shared state use the ordinary path. `--cuda-graphs auto|on|off`
  (auto = on when the model runs on CUDA), `$JEV_STYLE_CUDA_GRAPHS`, `cuda_graphs=` in `JevStyle` and
  `models.load_release`. A runtime without the CUDA-graph path loads as before under `auto`; `on` is refused there.
- **New torch pins.** The PyTorch builds of `0.8b-v3` and `2b-v3` are pinned to revisions that ship the CUDA-graph
  runtime: `0.8b-v3` torch @ b023d1f9, `2b-v3` torch @ 5bad2d53 (the 0.8B pin before that, d53c8f82, failed
  `verify=True` because of a stray manifest entry). The MLX and GGUF pins are unchanged.
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
