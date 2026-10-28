# Contributing to Jev-Style

Useful contributions include a reproducible failure case, a working integration, clearer setup instructions, and evaluations on data you can share. Open an [issue](https://github.com/lawrence3699/jev-style/issues/new/choose) before a large change so we can agree on scope.

## Report a problem or share a workflow

Include the package version, model release, backend, operating system, and the smallest input that reproduces the behavior. For model-quality reports, explain the expected answer and share probabilities, not only the winning label. Remove credentials and private data. Use the workflow template to share an integration or an idea without reporting a bug.

## Development setup

Python 3.10 or newer is required. The default tests use a fake engine and do not download weights.

```bash
git clone https://github.com/lawrence3699/jev-style
cd jev-style
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

On Windows, activate with `.venv\Scripts\activate`. For real-model work, also install `.[torch]` or `.[mlx]` as appropriate. Record the pinned model revision, hardware, dtype, input length, warmup and sample count when reporting speed or quality.

## Good first contributions

- Reproduce one of the [recipes](examples/recipes) on your machine and improve a setup step that was unclear.
- Add a small, shareable failure case with a label and an explanation of ambiguity.
- Add a focused integration recipe that runs from a documented command and produces inspectable output.

These are contribution suggestions, not claims that corresponding issues already exist. Choose a small change and include a focused test when behavior changes. For documentation, check the rendered Markdown and run any commands you change. Do not commit downloaded model weights, credentials, private datasets, or generated benchmark claims.

## Pull requests

Describe the user-visible problem, your change, and how you verified it. Preserve documented API compatibility and benchmark caveats. Keep fake-engine results clearly separate from real-model measurements. New examples should fail to human review on uncertain decisions rather than silently approving an action.

Contributions are provided under this repository's Apache-2.0 license.
