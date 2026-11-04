# Changelog

Versions follow [semantic versioning](https://semver.org/). Each release pins the model revisions it loads
(`jev_style/models.py`), so upgrading the package is what changes the weights you get.

## 0.6.4 (unreleased)

Fixes from the 2026-11-05 review of the package. Model revisions and weights are unchanged from 0.6.0.

- **Error bodies of both shapes.** The hosted `api.jevstyle.com` answers `{"error": "<code>", "message": "..."}`
  and reports over-long input as `413 input_too_large`, while this package's server answers the nested
  `{"error": {"code", "message"}}` and `400 max_tokens_exceeded`. The Python client (`parse_error_body`), the
  guard and the MCP server now read both; the client used to report `http_error: unauthorized` and the guard and
  MCP server crashed with `AttributeError` on a flat body. `JevStyleError.input_too_long` is true for
  `max_tokens_exceeded`, `input_too_large` and the pre-0.5.0 `input_budget_exceeded`.
- **Guard: a 200 without usable answers is "ask".** A response with no answer, a wrong answer type or no number
  for any of the five questions was treated as "no concern" and, with `emit_allow`, auto-approved. It is now an
  error (`on_error`, "ask"), like a server failure. The request no longer names a `model` (servers ignore it).
  `apply_thresholds` returns the `missing` questions. New tests cover thresholds, `emit_allow`, HTTP error
  bodies of both shapes and malformed 200 responses, through a real local stub server.
- **Guard and MCP server: no key or state to the wrong place.** `JEV_STYLE_API_KEY` is sent only to loopback
  servers and to the server named in `JEV_STYLE_URL`; any other `base_url` (eval `--server`, a cascade http
  tier, `--url`) needs an explicit key (`eval --key`, a tier's `api_key_env`, and the new `decide --api-key-env`). The MCP server's instructions are built per backend: the served model's
  context and per-question budgets come from `GET /v1/models` (the text used to say 25,600 / 2,048 for every
  model), and with `--allow-remote --url` they say that every state goes to that server instead of
  "nothing is sent elsewhere".
- **Local server: cross-site requests refused.** `jev-style serve` answers only requests whose `Host` is a
  loopback name or its `--host`, and refuses browser requests whose `Origin` is another site (403
  `forbidden_host` / `forbidden_origin`), so a web page elsewhere cannot use the local model (also under DNS
  rebinding). `--allow-any-host` and `create_app(allowed_hosts=None)` switch it off; with `--host 0.0.0.0` any
  Host is accepted but the Origin must still be the same host or loopback. Tests that build a `TestClient`
  now address it as `http://127.0.0.1`.
- **Downloads fail early and clearly.** `jev-style download` (and `models.download`) checks the backend's
  packages first, so a bare install no longer fetches 1.5-4.4 GB of weights it cannot run
  (`--no-backend-check` to download anyway); the GGUF backend refuses to start a download while no scorer
  binary can be found (`--scorer`, `$JEV_SCORE_BIN` / `$JEV_SCORE_V2_BIN`, `build/` in a cached snapshot, or
  PATH) and says how to get one; a failed download is one `error:` line (`DownloadError`) naming the repo,
  the reason and what to do (`HF_HUB_OFFLINE`, network, `--model-dir`) instead of a traceback; a missing
  `--model-dir` is `error: file not found`. `JEV_STYLE_RELEASE` is checked before argument parsing (one line
  instead of a traceback; `2B` is accepted for `2b`; the `guard` hook ignores it and still exits 0). `--backend mlx` on a machine that is not Apple silicon
  says so and points at `[torch]` instead of suggesting the `[mlx]` extra that installs nothing there.
- **Integrity check from the CLI.** `jev-style download` verifies every file against the repo's
  `manifest.json` (sizes and sha256; `--no-verify` skips it), and `serve` / `decide --verify` pass `verify=True`
  to the runtime; `models.verify_folder()` does the check on any model folder.
- **`trust_remote_code` repos get a generic file list.** A repo this package does not pin used to be
  downloaded with the 0.8B v3's file names (one `model.safetensors`, `bf16/*`, or
  `Jev-Style-0.8B-Decision-v3-<quant>.gguf`), so a 2B fork or an experiment repo failed after the metadata
  download; custom repos now take every runtime file, config, safetensors shard, `src/**`, MLX precision folder
  or `*-<quant>.gguf` they contain.
- **CLI help.** `jev-style --help` lists `eval`, `guard`, `guard-replay` and `mcp` with the other commands, and
  `jev-style guard --help` / `guard-replay --help` show their own name in the usage line.
- **Release workflow.** A manual run can publish to TestPyPI only; PyPI releases come from a `vX.Y.Z` tag, which is
  checked against `__version__` first. `server.json` describes the current models (2B v4 and 0.8B, text and
  images). CI runs the test suite and the fake-engine CLI on Windows as well, and the package declares Windows.
- **Metadata and documentation.** `NOTICE` lists the 2B v3 and 2B v4 sources (Qwen3.5-2B derivatives; v4 with
  Jev 1.14 soft labels) next to the 0.8B. The `Models` link points at the Hub profile (the v4 collection has no
  0.8B). The README gives the Decision Index 0.3 board entry (2B v4 listed 2026-11-04: Public 44.37, Full 34.53,
  Breadth 41.75; Vision 57.55 / 51.94; 0.8B v3 17.0 / 12.5) instead of "self-run, v3 34.05 on the board", the
  hosted API's context (8,192 tokens, 8,000-token inputs) and 429 / `Retry-After` behaviour, the `[mlx]`
  pin, the key rule, the error shapes, and `noul` / `choice` examples in place of the urgency `score` ones. The
  serve, mcp and jev-style skills cover `--release 2b` (2B v4), the per-release GGUF scorers, a `model` field that
  starts with `jev-style-`, and drop the pre-0.3.0 "mcp extra" advice; the guard numbers are marked as measured
  with the 0.8B v3. New `SECURITY.md` (private reports), `CODE_OF_CONDUCT.md`, a pull request template, and
  GitHub Discussions as the place for questions.

## 0.6.3 (2026-11-02)

Documentation-only release: clarify the official local installation, separate local inference from hosted API and third-party Jev services, place the 2B v4 teacher-label provenance alongside the independence statement, and link the dated/live service-status source and reproducible routing comparison. PyPI now serves the same corrected README as the repository. Model revisions, dependencies, inference behavior and weights are unchanged from 0.6.2.

## 0.6.2 (2026-10-29)

- Synchronize the PyPI description with the project’s local classification, routing,
  and scoring positioning, and link directly to model selection, benchmark protocols,
  runnable application guides, and project information on jevstyle.com.
- Documentation and package metadata only. Runtime behavior, dependencies, model
  revisions, and weights are unchanged from 0.6.1.

## 0.6.1 (2026-10-28)

- Start with the live demo or a short local setup. The README now puts installation,
  model selection and runnable recipes ahead of benchmark details, and fixes a broken
  code fence. Package metadata links to the current v4 model collection.
- Add complete support-ticket, confidence-based fallback and receipt-review recipes,
  with synthetic sample inputs and transport/routing tests.
- Add a Semantic Router 1.1.1 integration example using a local systemone endpoint.
- Add contribution guidance and issue templates for reproducible quality reports and
  real application use cases.
- Preserve the 0.8B Space source and include the current 2B v4 Space source separately.
  The 2B demo offers task-based examples and visible copyable local code.
- Model revisions and inference behavior match 0.6.0.

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
