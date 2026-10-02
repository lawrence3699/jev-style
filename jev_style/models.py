"""Download and load a Jev-Style release from the Hugging Face Hub.

Every build ships its own self-contained runtime next to the weights (``jev_style_decision*.py``):
input rendering, verdict readout, calibration temperatures and token budgets all live there. This
module only picks a build, downloads a pinned revision and imports that runtime, so the server gives
exactly the answers the model card documents.

Releases (``--release`` / ``JevStyle(release=...)``; default ``0.8b-v3``), each with three builds:

* ``0.8b-v3``  chaoliangUNSW/Jev-Style-0.8B-Decision-v3[-MLX|-GGUF]
* ``2b-v3``    chaoliangUNSW/Jev-Style-2B-Decision-v3[-MLX|-GGUF]

Backends:

* ``torch``  the main repo                 CUDA, Apple MPS or CPU (float32 by default)
* ``mlx``    the ``-MLX`` repo             Apple silicon, bf16 or 8-bit weights
* ``gguf``   the ``-GGUF`` repo            llama.cpp via the ``jev-score`` scorer (build it once, see the GGUF card)

``auto`` = mlx on Apple silicon when ``mlx`` + ``mlx-lm`` are installed, otherwise torch (or whatever the release
ships). ``build_for_repo`` maps a Hub repo id to its build (``JevStyle.from_pretrained``). Known repos load at their
pinned revisions. Any other repo runs the ``jev_style_decision*.py`` file it ships, so it is refused unless the
caller passes ``trust_remote_code=True``; the same flag lets a release that is not pinned yet load from ``main``.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import platform
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any

BACKENDS = ("auto", "torch", "mlx", "gguf")


@dataclass(frozen=True)
class Build:
    backend: str
    repo: str
    revision: str | None
    module: str
    cls: str
    patterns: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)
    release: str = "0.8b-v3"


@dataclass(frozen=True)
class Release:
    key: str                    # "0.8b-v3"
    model_id: str               # what the server reports: "jev-style-0.8b-decision-v3"
    name: str                   # Hub name: "Jev-Style-0.8B-Decision-v3"
    title: str                  # for people: "Jev-Style 0.8B Decision v3"
    release_date: str | None
    builds: dict[str, Build]

    @property
    def published(self) -> bool:
        """Every build pinned to a revision: only then do the CLI and ``auto`` offer it."""
        return all(b.revision for b in self.builds.values())


_COMMON = ("LICENSE", "NOTICE", "manifest.json", "readout_config.json", "release_config.json", "requirements.txt")
_TORCH_FILES = ("jev_style_decision.py", "config.json", "generation_config.json", "chat_template.jinja",
                "tokenizer.json", "tokenizer_config.json")


def _release(key: str, name: str, title: str, model_id: str, release_date: str | None,
             revisions: dict[str, str | None], torch_weights: tuple[str, ...], mlx_prefix_sharing: bool,
             mlx_extra: tuple[str, ...] = (), mlx_lm: str | None = None, scorer_src: str = "jev_score.cpp",
             scorer_bin: str = "jev-score", scorer_env: str = "JEV_SCORE_BIN") -> Release:
    repo = "chaoliangUNSW/" + name
    builds = {
        "torch": Build("torch", repo, revisions["torch"], "jev_style_decision", "JevStyleDecision",
                       _COMMON + _TORCH_FILES + torch_weights, release=key),
        "mlx": Build("mlx", repo + "-MLX", revisions["mlx"], "jev_style_decision_mlx", "JevStyleDecisionMLX",
                     _COMMON + ("jev_style_decision_mlx.py",) + mlx_extra,
                     {"precision_patterns": "{precision}/*", "prefix_sharing": mlx_prefix_sharing,
                      "mlx_lm": mlx_lm}, release=key),
        "gguf": Build("gguf", repo + "-GGUF", revisions["gguf"], "jev_style_decision_gguf", "JevStyleDecisionGGUF",
                      _COMMON + ("jev_style_decision_gguf.py", scorer_src, "build_jev_score.sh", "tokenizer/*"),
                      {"quant_file": name + "-{quant}.gguf", "scorer_bin": scorer_bin, "scorer_env": scorer_env},
                      release=key),
    }
    return Release(key, model_id, name, title, release_date, builds)


# The torch builds pin the main repos' CUDA-graph runtimes (0.4.0; JevStyleDecision takes cuda_graphs=). The 0.8B
# pin before that (d53c8f82) had a manifest bug that made verify=True fail.
RELEASES: dict[str, Release] = {
    "0.8b-v3": _release(
        "0.8b-v3", "Jev-Style-0.8B-Decision-v3", "Jev-Style 0.8B Decision v3", "jev-style-0.8b-decision-v3",
        "2026-09-24",
        {"torch": "b023d1f9c7858fbf01504577a3bfc349ea5c7385",     # 0.4.0: CUDA-graph runtime (cuda_graphs=)
         "mlx": "7f14c9fa1491d168a7f70b16acf68baf9d4f7353", "gguf": "b8356a83beb560cf34cbe6f9a20c2076e1b532d3"},
        ("model.safetensors",), mlx_prefix_sharing=True),
    # 2B: block attention. Its MLX runtime patches mlx-lm and checks the patched source, so it needs exactly
    # mlx-lm 0.31.3; its GGUF runtime needs the jev-score-v2 scorer (the 0.8B jev-score is refused). Prefix sharing
    # stays off: the runtime already reads the state once per call, and the 0.8B patch was never checked on it.
    "2b-v3": _release(
        "2b-v3", "Jev-Style-2B-Decision-v3", "Jev-Style 2B Decision v3", "jev-style-2b-decision-v3",
        "2026-09-27",
        {"torch": "5bad2d53ae04e832e2b30aa2a89bec76ec03deec",     # 0.4.0: CUDA-graph runtime (cuda_graphs=)
         "mlx": "11ce5d718e22412b38624bb863a1eee73c0a5934", "gguf": "283b9eb7903aeb184b16d49ab8f38b256f4afcb3"},
        ("*.safetensors", "model.safetensors.index.json"), mlx_prefix_sharing=False,
        mlx_extra=("config.json", "THIRD_PARTY_NOTICES.md"), mlx_lm="0.31.3",
        scorer_src="jev_score_v2.cpp", scorer_bin="jev-score-v2", scorer_env="JEV_SCORE_V2_BIN"),
}
ALIASES = {"0.8b": "0.8b-v3", "2b": "2b-v3"}
DEFAULT_RELEASE = "0.8b-v3"

# the default release, under the names 0.2.x exported
BUILDS: dict[str, Build] = RELEASES[DEFAULT_RELEASE].builds
MODEL_ID = RELEASES[DEFAULT_RELEASE].model_id
MODEL_NAME = RELEASES[DEFAULT_RELEASE].name


def release_keys(published_only: bool = True) -> list[str]:
    """Release keys and their short aliases, e.g. for ``--release`` choices."""
    keys = [k for k, r in RELEASES.items() if r.published or not published_only]
    return keys + [a for a, k in ALIASES.items() if k in keys]


def get_release(key: str | None = None) -> Release:
    key = key or os.environ.get("JEV_STYLE_RELEASE") or DEFAULT_RELEASE
    key = ALIASES.get(key.lower(), key.lower())
    if key not in RELEASES:
        raise ValueError(f"unknown release {key!r}; known: {', '.join(release_keys(published_only=False))}")
    return RELEASES[key]


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def resolve_backend(backend: str = "auto", release: str | None = None) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    builds = get_release(release).builds
    if backend != "auto":
        if backend not in builds:
            raise ValueError(f"release {get_release(release).key} has no {backend} build")
        return backend
    if "mlx" in builds and is_apple_silicon() and build_problem(builds["mlx"]) is None:
        return "mlx"
    return "torch" if "torch" in builds else next(iter(builds))


class MissingBackendError(ImportError):
    """The chosen backend's Python packages are not installed (raised before any download)."""


_BACKEND_MODULES = {"torch": ("torch", "transformers"), "mlx": ("mlx", "mlx_lm"), "gguf": ()}


def _installed_version(dist: str) -> str | None:
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def build_problem(build: Build) -> str | None:
    """Why this build cannot run with the installed packages, or None."""
    missing = [m for m in _BACKEND_MODULES.get(build.backend, ()) if not has_module(m)]
    if missing:
        return f"needs {', '.join(missing)}, which {'is' if len(missing) == 1 else 'are'} not installed"
    want = build.extra.get("mlx_lm") if build.backend == "mlx" else None
    have = _installed_version("mlx-lm") if want else None
    if want and have != want:
        return f"needs mlx-lm {want} exactly (its runtime patches mlx-lm and checks the source), found {have}"
    return None


def require_backend(backend: str | Build) -> None:
    """Fail fast, before a 1-4 GB download, when the backend's packages are missing or the wrong version."""
    build = backend if isinstance(backend, Build) else BUILDS[backend]
    problem = build_problem(build)
    if not problem:
        return
    b = build.backend
    if b == "mlx":
        hint = 'pip install "jev-style[mlx]"'
    elif is_apple_silicon():
        hint = 'pip install "jev-style[mlx]" (Apple silicon), or "jev-style[torch]" for PyTorch'
    else:
        hint = 'pip install "jev-style[torch]"'
    raise MissingBackendError(
        f"the {b} backend of {build.repo} {problem}. The bare `pip install jev-style` is only the client; to run the "
        f"model locally: {hint}. The gguf backend needs no extra, only the release's scorer (see the GGUF card).")


def build_for_repo(repo: str, *, trust_remote_code: bool = False) -> Build:
    """The build a Hub repo id names. Known repos keep their pinned revision; others need trust_remote_code."""
    for r in RELEASES.values():
        for b in r.builds.values():
            if b.repo.lower() == repo.lower():
                return b
    if not trust_remote_code:
        known = ", ".join(b.repo for r in RELEASES.values() if r.published for b in r.builds.values())
        raise ValueError(f"{repo!r} is not a Jev-Style release this package knows ({known}). Loading it runs the "
                         "Python runtime file it ships; pass trust_remote_code=True if you trust it.")
    tail = repo.rsplit("/", 1)[-1].lower()
    backend = "gguf" if "gguf" in tail else "mlx" if "mlx" in tail else "torch"
    return replace(BUILDS[backend], repo=repo, revision=None, release="custom")


def patterns_for(build: Build, precision: str = "bf16", quant: str = "Q8_0") -> list[str]:
    pats = list(build.patterns)
    if build.backend == "mlx":
        pats.append(build.extra["precision_patterns"].format(precision=precision))
    if build.backend == "gguf":
        pats.append(build.extra["quant_file"].format(quant=quant.upper()))
    return pats


def download(backend: str = "auto", *, precision: str = "bf16", quant: str = "Q8_0",
             revision: str | None = None, build: Build | None = None, release: str | None = None,
             trust_remote_code: bool = False) -> Path:
    """Download (or reuse from the HF cache) the files one backend needs. Returns the local folder."""
    from huggingface_hub import snapshot_download

    build = build or get_release(release).builds[resolve_backend(backend, release)]
    if not (revision or build.revision) and not trust_remote_code:
        raise ValueError(f"{build.repo} has no pinned revision in jev-style {_version()} (release "
                         f"{build.release} is not published in this version). Upgrade jev-style, pass a local "
                         "model_dir, or pass trust_remote_code=True to load the repo's main branch.")
    kw = dict(revision=revision or build.revision, allow_patterns=patterns_for(build, precision, quant))
    try:                                   # pinned revision already cached: no network round trip
        return Path(snapshot_download(build.repo, local_files_only=True, **kw))
    except Exception:  # noqa: BLE001 - not cached (or partly): download
        return Path(snapshot_download(build.repo, **kw))


def import_runtime(model_dir: Path, module: str) -> ModuleType:
    """Import the runtime file that ships with the weights (``<model_dir>/<module>.py``)."""
    f = Path(model_dir) / f"{module}.py"
    if not f.is_file():
        raise FileNotFoundError(f"{f} not found: is {model_dir} a Jev-Style model folder?")
    name = f"_jev_style_rt_{module}"
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_release(backend: str = "auto", *, release: str | None = None, model_dir: str | Path | None = None,
                 device: str | None = None, dtype: str = "float32", precision: str = "bf16", quant: str = "Q8_0",
                 scorer: str | None = None, revision: str | None = None, verify: bool = False,
                 repo: str | None = None, trust_remote_code: bool = False,
                 cuda_graphs: bool | None = None) -> tuple[Any, ModuleType, Build]:
    """-> (runtime object, runtime module, build). ``model_dir`` skips the download; ``repo`` picks the build (and
    so the release and backend) from a Hub repo id instead of ``release`` and ``backend``.

    ``cuda_graphs`` (torch backend): True / False, or None = $JEV_STYLE_CUDA_GRAPHS if set, else on whenever the
    model runs on CUDA. Runtimes older than the CUDA-graph releases do not take the option: then None / False load
    them as before and True is refused."""
    if repo:
        build = build_for_repo(repo, trust_remote_code=trust_remote_code)
    else:
        build = get_release(release).builds[resolve_backend(backend, release)]
    backend = build.backend
    require_backend(build)
    folder = Path(model_dir).expanduser() if model_dir else download(
        backend, precision=precision, quant=quant, revision=revision, build=build,
        trust_remote_code=trust_remote_code)
    rt = import_runtime(folder, build.module)
    cls = getattr(rt, build.cls)
    if backend == "torch":
        device = device or os.environ.get("JEV_STYLE_DEVICE") or os.environ.get("JEV_DEVICE") or None
        kw = {}
        if "cuda_graphs" in inspect.signature(cls).parameters:
            kw["cuda_graphs"] = _want_cuda_graphs(cuda_graphs, device)
        elif cuda_graphs:
            raise ValueError(f"the {build.repo} runtime at this revision has no CUDA-graph path (cuda_graphs=True)")
        runtime = cls(folder, device=device, dtype=dtype, verify=verify, **kw)
    elif backend == "mlx":
        runtime = cls(folder, precision=precision, verify=verify)
    else:
        runtime = cls(folder, quant=quant.upper(), binary=scorer, verify=verify)  # None: the runtime's own lookup
    return runtime, rt, build


def _want_cuda_graphs(flag: bool | None, device: str | None) -> bool:
    if flag is None:
        env = os.environ.get("JEV_STYLE_CUDA_GRAPHS", "").strip().lower()
        if env:
            flag = env not in ("0", "false", "no", "off")
    if flag is not None:
        return bool(flag)
    if device:
        return str(device).split(":", 1)[0] == "cuda"
    import torch
    return torch.cuda.is_available()                # the runtime's own default: cuda > mps > cpu


def load(backend: str = "auto", **kw: Any) -> tuple[Any, ModuleType, str]:
    """-> (runtime object, runtime module, backend name); see ``load_release``."""
    runtime, rt, build = load_release(backend, **kw)
    return runtime, rt, build.backend


def _version() -> str:
    from . import __version__
    return __version__
