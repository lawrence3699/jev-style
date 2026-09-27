# Releasing

## A package release

1. Update `__version__` in `jev_style/__init__.py`, both `version` fields in `server.json`, and the top of
   `CHANGELOG.md` (move "unreleased" to a date). `tests/test_server_json.py` checks that `server.json` matches.
2. `uv run pytest -q`, then merge to `main`.
3. `git tag -a vX.Y.Z -m "jev-style X.Y.Z" && git push origin vX.Y.Z`. The `release` workflow checks that the tag
   equals `__version__`, runs the tests, builds, checks the metadata, installs the wheel in a clean venv and
   publishes to PyPI through Trusted Publishing (environment `pypi`; no token).
   After the PyPI job, the `mcp-registry` job waits until PyPI serves the version, sets it in `server.json` from the
   tag and publishes to the MCP Registry (GitHub OIDC). The registry checks that the PyPI README contains
   `<!-- mcp-name: io.github.lawrence3699/jev-style -->`, so keep that line in `README.md`.
4. `gh release create vX.Y.Z` with the CHANGELOG section.

## Adding a model release (done for 2b-v3 in 0.3.0)

The package pins every build to a Hub revision, so a new model reaches `pip` users only with a package release.

1. The three repos exist on the Hub: `chaoliangUNSW/<Name>`, `<Name>-MLX`, `<Name>-GGUF`. Each ships its runtime
   file with the interface `jev_style/models.py` calls:
   - torch `jev_style_decision.JevStyleDecision(folder, device=, dtype=, verify=)`,
     MLX `jev_style_decision_mlx.JevStyleDecisionMLX(folder, precision=, verify=)`,
     GGUF `jev_style_decision_gguf.JevStyleDecisionGGUF(folder, quant=, binary=, verify=)`;
   - `decide_many(state, questions, category=...)` accepts any category and returns, per question,
     `probabilities` (noul: `"false"`, `"true"`), `input_tokens`, `head_tokens`;
   - the module raises `InputBudgetError` and `QuestionError`; `runtime.renderer.max_len` / `head_max` exist.
2. In `RELEASES`: fill the three revisions and `release_date`, and check the file patterns against the repos
   (safetensors shards, MLX precision folders, GGUF file names).
3. MLX prefix sharing stays off (`mlx_prefix_sharing=False`) unless you have shown identical scores with it. Name any
   extra root files the runtime's integrity check wants (`mlx_extra`), an exact mlx-lm version it checks (`mlx_lm`),
   and the GGUF scorer source / binary / environment variable (`scorer_src`, `scorer_bin`, `scorer_env`).
4. Load each build through the package (`jev-style decide ... --release <key> --backend <b>`) and compare with the
   model card's parity numbers.
5. README: add the release to the Model table and mention `--release`. Then do a package release (above).
6. After the package is on PyPI: add the `pip install` / `from_pretrained` snippet to the three model cards
   (update the README hash in `manifest.json` in the same commit) and resync the GitHub mirrors.
