"""Download and load a Jev-Style release from the Hugging Face Hub.

Every build ships its own self-contained runtime next to the weights (``jev_style_decision*.py``):
input rendering, verdict readout, calibration temperatures and token budgets all live there. This
module only picks a build, downloads a pinned revision and imports that runtime, so the server gives
exactly the answers the model card documents.

Releases (``--release`` / ``JevStyle(release=...)``; default ``0.8b-v3``; ``0.8b`` / ``2b`` = the newest of each size):

* ``0.8b-v3``  chaoliangUNSW/Jev-Style-0.8B-Decision-v3[-MLX|-GGUF]
* ``2b-v3``    chaoliangUNSW/Jev-Style-2B-Decision-v3[-MLX|-GGUF]
* ``2b-v4``    chaoliangUNSW/Jev-Style-2B-Decision-v4[-MLX|-GGUF] (images: the torch build; MLX and GGUF read text only)

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


# 2B v4 (multimodal Qwen3.5 export): the runtime calls the reference scorer it ships under src/macjev/ and reads the
# image processor configs and macjev_readout.json next to the weights.
_TORCH_FILES_V4 = _TORCH_FILES + ("preprocessor_config.json", "video_preprocessor_config.json", "macjev_readout.json",
                                  "SHA256SUMS.json", "src/macjev/*")


def _release(key: str, name: str, title: str, model_id: str, release_date: str | None,
             revisions: dict[str, str | None], torch_weights: tuple[str, ...], mlx_prefix_sharing: bool = False,
             mlx_extra: tuple[str, ...] = (), mlx_lm: str | None = None, scorer_src: str = "jev_score.cpp",
             scorer_bin: str = "jev-score", scorer_env: str = "JEV_SCORE_BIN",
             backends: tuple[str, ...] = ("torch", "mlx", "gguf"),
             torch_files: tuple[str, ...] = _TORCH_FILES) -> Release:
    repo = "chaoliangUNSW/" + name
    builds = {
        "torch": Build("torch", repo, revisions["torch"], "jev_style_decision", "JevStyleDecision",
                       _COMMON + torch_files + torch_weights, release=key),
        "mlx": Build("mlx", repo + "-MLX", revisions.get("mlx"), "jev_style_decision_mlx", "JevStyleDecisionMLX",
                     _COMMON + ("jev_style_decision_mlx.py",) + mlx_extra,
                     {"precision_patterns": "{precision}/*", "prefix_sharing": mlx_prefix_sharing,
                      "mlx_lm": mlx_lm}, release=key),
        "gguf": Build("gguf", repo + "-GGUF", revisions.get("gguf"), "jev_style_decision_gguf", "JevStyleDecisionGGUF",
                      _COMMON + ("jev_style_decision_gguf.py", scorer_src, "build_jev_score.sh", "tokenizer/*"),
                      {"quant_file": name + "-{quant}.gguf", "scorer_bin": scorer_bin, "scorer_env": scorer_env},
                      release=key),
    }
    return Release(key, model_id, name, title, release_date, {b: builds[b] for b in backends})


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
    # 2B v4: text and images (Jev 1.14 ``assets``) in the PyTorch build, text only in the MLX and GGUF builds. Its
    # PyTorch runtime runs the reference scorer the model was evaluated with and picks its own speed paths (CUDA graphs
    # with bfloat16 on CUDA). MLX and GGUF as for 2b-v3: mlx-lm 0.31.3 exactly, and the jev-score-v2 scorer (the GGUF
    # repo ships the 2b-v3 scorer source unchanged, so one jev-score-v2 binary serves both).
    "2b-v4": _release(
        "2b-v4", "Jev-Style-2B-Decision-v4", "Jev-Style 2B Decision v4", "jev-style-2b-decision-v4",
        "2026-10-27",
        {"torch": "c9504148dae7f7a43d67a5782f12a015a4b4022c",
         "mlx": "a14bbe4dafb154f0b05884314c8e227b1c6a5cd6", "gguf": "9111d5d232d21d52593c4c10f1e8cfa6561669f6"},
        ("*.safetensors", "model.safetensors.index.json"), mlx_prefix_sharing=False,
        mlx_extra=("config.json", "THIRD_PARTY_NOTICES.md"), mlx_lm="0.31.3",
        scorer_src="jev_score_v2.cpp", scorer_bin="jev-score-v2", scorer_env="JEV_SCORE_V2_BIN",
        torch_files=_TORCH_FILES_V4),
}
ALIASES = {"0.8b": "0.8b-v3", "2b": "2b-v4"}
DEFAULT_RELEASE = "0.8b-v3"

# the default release, under the names 0.2.x exported
BUILDS: dict[str, Build] = RELEASES[DEFAULT_RELEASE].builds
MODEL_ID = RELEASES[DEFAULT_RELEASE].model_id
MODEL_NAME = RELEASES[DEFAULT_RELEASE].name


def release_keys(published_only: bool = True) -> list[str]:
    """Release keys and their short aliases, e.g. for ``--release`` choices."""
    keys = [k for k, r in RELEASES.items() if r.published or not published_only]
    return keys + [a for a, k in ALIASES.items() if k in keys]


def env_release_problem() -> str | None:
    """Why ``$JEV_STYLE_RELEASE`` cannot be used, or None (unset, or a release this version offers). Checked
    before argument parsing, so a typo in the variable is one clear line instead of a traceback later."""
    raw = os.environ.get("JEV_STYLE_RELEASE")
    if not raw:
        return None
    key = ALIASES.get(raw.strip().lower(), raw.strip().lower())
    if key not in RELEASES:
        return (f"JEV_STYLE_RELEASE={raw!r} is not a release this version of jev-style knows; "
                f"use one of {', '.join(release_keys())} (or unset it)")
    if not RELEASES[key].published:
        return f"JEV_STYLE_RELEASE={raw!r} names a release that is not published in this version of jev-style"
    return None


def env_release(default: str = DEFAULT_RELEASE) -> str:
    """``$JEV_STYLE_RELEASE`` as a ``--release`` choice (case and alias normalised), else ``default``."""
    raw = (os.environ.get("JEV_STYLE_RELEASE") or "").strip().lower()
    return raw if raw and env_release_problem() is None else default


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


class MissingScorerError(MissingBackendError):
    """The GGUF backend's scorer binary cannot be found and the weights are not downloaded yet (raised before
    the download, so the first run does not fetch 1-4 GB only to stop on this)."""


class DownloadError(RuntimeError):
    """The Hub download failed (network, offline mode, a missing or private repo): the message says what to do."""


class VerifyError(RuntimeError):
    """A downloaded file does not match the repo's ``manifest.json`` (size or sha256)."""


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
    if b == "mlx" and not is_apple_silicon():
        raise MissingBackendError(
            f"the mlx backend of {build.repo} only runs on Apple silicon (the [mlx] extra installs nothing on "
            f"other machines). Use --backend torch: pip install \"jev-style[torch]\" (CUDA, CPU or Apple MPS).")
    if b == "mlx":
        hint = 'pip install "jev-style[mlx]"'
    elif is_apple_silicon():
        hint = 'pip install "jev-style[mlx]" (Apple silicon), or "jev-style[torch]" for PyTorch'
    else:
        hint = 'pip install "jev-style[torch]"'
    raise MissingBackendError(
        f"the {b} backend of {build.repo} {problem}. The bare `pip install jev-style` is only the client; to run the "
        f"model locally: {hint}. The gguf backend needs no extra, only the release's scorer (see the GGUF card).")


def find_scorer(build: Build, scorer: str | None = None, folder: Path | None = None) -> str | None:
    """Where the GGUF build's scorer binary is, or None: ``scorer`` (``--scorer``), the release's environment
    variable (``$JEV_SCORE_BIN`` / ``$JEV_SCORE_V2_BIN``), ``<folder>/build/<binary>`` (where
    ``build_jev_score.sh`` puts it) or the binary on PATH. The same places the runtime looks, checked before any
    download."""
    import shutil
    if scorer:
        return scorer
    env, name = build.extra.get("scorer_env"), build.extra.get("scorer_bin")
    if env and os.environ.get(env):
        return os.environ[env]
    if folder is not None and name:
        for candidate in (Path(folder) / "build" / name, Path(folder) / "build" / f"{name}.exe"):
            if candidate.is_file():
                return str(candidate)
    return shutil.which(name) if name else None


def cached_folder(build: Build, precision: str = "bf16", quant: str = "Q8_0",
                  revision: str | None = None) -> Path | None:
    """The cached snapshot folder with every file the build needs, or None (no network)."""
    from huggingface_hub import snapshot_download
    rev = revision or build.revision
    if not rev:
        return None
    pats = patterns_for(build, precision, quant)
    try:
        folder = Path(snapshot_download(build.repo, revision=rev, allow_patterns=pats, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached (or partly)
        return None
    return folder if cached_complete(folder, pats) else None


def require_scorer(build: Build, scorer: str | None = None, *, precision: str = "bf16", quant: str = "Q8_0",
                   revision: str | None = None) -> None:
    """GGUF: refuse to start the download when no scorer binary can be found anywhere and the pinned files are
    not cached yet. The runtime looks the binary up itself once the files are there (a scorer built inside the
    model folder is found then), so a cached snapshot passes."""
    if build.backend != "gguf" or build.release == "custom":
        return
    folder = cached_folder(build, precision, quant, revision)
    if folder is not None or find_scorer(build, scorer, folder) is not None:
        return
    name, env, src = build.extra.get("scorer_bin"), build.extra.get("scorer_env"), None
    for pat in build.patterns:
        if pat.startswith("jev_score") and pat.endswith(".cpp"):
            src = pat
    rel = build.release
    raise MissingScorerError(
        f"the gguf backend of {build.repo} needs the {name} scorer binary, built once from {src} against llama.cpp, "
        f"and none was found (--scorer PATH, ${env}, or {name} on PATH). Download the files first: "
        f"`jev-style download --backend gguf --release {rel}` prints the model folder; run "
        f"`sh build_jev_score.sh /path/to/llama.cpp` there, then start again with --scorer <the path it prints> "
        f"(see the GGUF model card).")


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
    # Not one of the pinned repos, so its file layout is unknown: take every runtime file, config and weights file
    # the three layouts use (sharded safetensors, an MLX precision folder, a GGUF quant), not the 0.8B's names.
    base = BUILDS[backend]
    return replace(base, repo=repo, revision=None, release="custom", patterns=_CUSTOM_PATTERNS[backend],
                   extra={**base.extra, **_CUSTOM_EXTRA.get(backend, {})})


_CUSTOM_PATTERNS: dict[str, tuple[str, ...]] = {
    "torch": _COMMON + ("*.py", "*.json", "*.jinja", "*.txt", "*.safetensors", "src/**"),
    "mlx": _COMMON + ("*.py", "*.json", "*.md", "*.txt"),                    # + "{precision}/*"
    "gguf": _COMMON + ("*.py", "*.cpp", "*.sh", "*.txt", "tokenizer/*"),      # + "*-{quant}.gguf"
}
_CUSTOM_EXTRA: dict[str, dict[str, Any]] = {
    "gguf": {"quant_file": "*-{quant}.gguf", "scorer_bin": None, "scorer_env": None},
}


def patterns_for(build: Build, precision: str = "bf16", quant: str = "Q8_0") -> list[str]:
    pats = list(build.patterns)
    if build.backend == "mlx":
        pats.append(build.extra["precision_patterns"].format(precision=precision))
    if build.backend == "gguf":
        pats.append(build.extra["quant_file"].format(quant=quant.upper()))
    return pats


def download(backend: str = "auto", *, precision: str = "bf16", quant: str = "Q8_0",
             revision: str | None = None, build: Build | None = None, release: str | None = None,
             trust_remote_code: bool = False, check_backend: bool = True, verify: bool = False) -> Path:
    """Download (or reuse from the HF cache) the files one backend needs. Returns the local folder.

    ``check_backend`` (default on) refuses, before anything is fetched, a backend whose Python packages are not
    installed (``MissingBackendError``): the bare install cannot run the weights it would download. ``verify``
    checks every downloaded file's size and sha256 against the repo's ``manifest.json`` (``VerifyError``). A
    failed download raises ``DownloadError`` with what to do instead of the Hub client's traceback."""
    from huggingface_hub import snapshot_download

    build = build or get_release(release).builds[resolve_backend(backend, release)]
    if check_backend:
        require_backend(build)
    if not (revision or build.revision) and not trust_remote_code:
        raise ValueError(f"{build.repo} has no pinned revision in jev-style {_version()} (release "
                         f"{build.release} is not published in this version). Upgrade jev-style, pass a local "
                         "model_dir, or pass trust_remote_code=True to load the repo's main branch.")
    kw = dict(revision=revision or build.revision, allow_patterns=patterns_for(build, precision, quant))
    folder = None
    try:                                   # pinned revision already cached: no network round trip
        cached = Path(snapshot_download(build.repo, local_files_only=True, **kw))
        if cached_complete(cached, kw["allow_patterns"]):
            folder = cached
    except Exception:  # noqa: BLE001 - not cached (or partly): download
        pass
    if folder is None:
        try:
            folder = Path(snapshot_download(build.repo, **kw))
        except (OSError, ValueError) as e:     # HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled ...
            raise DownloadError(_download_problem(build, kw["revision"], e)) from None
    if verify:
        verify_folder(folder, kw["allow_patterns"])
    return folder


def _download_problem(build: Build, revision: str | None, exc: BaseException) -> str:
    reason = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    if len(reason) > 300:
        reason = reason[:300] + "..."
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")
    hint = ("HF_HUB_OFFLINE is set, so nothing is fetched: unset it, or use a local copy with --model-dir." if offline
            else "Check the network connection and any proxy (HF_ENDPOINT / HTTPS_PROXY); if the repo is private, "
                 "log in with `hf auth login`; a local copy can be used with --model-dir.")
    rev = f" (revision {revision[:12]})" if revision else ""
    return f"could not download {build.repo}{rev}: {reason}. {hint}"


def manifest_entries(folder: Path) -> dict[str, dict]:
    """``manifest.json`` of a model folder as {relative path: {"sha256", "bytes"}} (the jev-style-manifest-v1
    format every release ships), or {} when the folder has none."""
    import json
    try:
        files = json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8")).get("files")
    except (OSError, ValueError, AttributeError):
        return {}
    if isinstance(files, dict):
        return {str(k): v for k, v in files.items() if isinstance(v, dict)}
    if isinstance(files, list):            # older manifests: a list of names, nothing to compare against
        return {str(n): {} for n in files}
    return {}


def verify_folder(folder: str | Path, patterns: list[str] | None = None) -> list[str]:
    """Check the files ``patterns`` select (default: all) against the folder's ``manifest.json``: present,
    the right size and the right sha256. Returns the names checked; raises ``VerifyError`` on the first
    mismatch, or when the folder has no manifest to check against."""
    import fnmatch
    import hashlib
    folder = Path(folder)
    entries = manifest_entries(folder)
    if not entries:
        raise VerifyError(f"{folder} has no manifest.json to verify against")
    names = [n for n in entries if patterns is None or any(fnmatch.fnmatchcase(n, p) for p in patterns)]
    names = [n for n in names if n != "manifest.json"]
    if not names:
        raise VerifyError(f"{folder}: the manifest lists none of the files this build uses")
    checked = []
    for name in names:
        want, path = entries[name], folder / name
        if not path.is_file():
            raise VerifyError(f"{folder}: {name} is missing")
        if want.get("bytes") is not None and path.stat().st_size != int(want["bytes"]):
            raise VerifyError(f"{folder}: {name} is {path.stat().st_size} bytes, the manifest says {want['bytes']}")
        if want.get("sha256"):
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != str(want["sha256"]).lower():
                raise VerifyError(f"{folder}: {name} does not match the manifest's sha256")
        checked.append(name)
    return checked


def cached_complete(folder: Path, patterns: list[str]) -> bool:
    """Every file of the folder's manifest.json that ``patterns`` select is present. A cached snapshot of the pinned
    revision can hold only some of them: another precision or quant, or one file fetched with ``hf_hub_download``;
    huggingface_hub before 2.x returns such a folder from ``local_files_only=True`` without checking."""
    import fnmatch
    import json
    try:
        names = json.loads((folder / "manifest.json").read_text())["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    want = [n for n in names if any(fnmatch.fnmatchcase(n, pat) for pat in patterns)]
    return bool(want) and all((folder / n).is_file() for n in want)


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

    ``cuda_graphs`` (torch backend): True / False, or None = $JEV_STYLE_CUDA_GRAPHS if set, else the runtime's own
    default when it has one (2B v4: on for bfloat16 on CUDA), else on whenever the model runs on CUDA (the v3
    runtimes, whose default is off). Runtimes older than the CUDA-graph releases do not take the option: then None /
    False load them as before and True is refused."""
    if repo:
        build = build_for_repo(repo, trust_remote_code=trust_remote_code)
    else:
        build = get_release(release).builds[resolve_backend(backend, release)]
    backend = build.backend
    require_backend(build)
    if model_dir:
        folder = Path(model_dir).expanduser()
        if not folder.is_dir():
            raise FileNotFoundError(f"model_dir {folder} is not a directory")
    else:
        require_scorer(build, scorer, precision=precision, quant=quant, revision=revision)
        folder = download(backend, precision=precision, quant=quant, revision=revision, build=build,
                          trust_remote_code=trust_remote_code, check_backend=False)
    rt = import_runtime(folder, build.module)
    cls = getattr(rt, build.cls)
    if backend == "torch":
        device = device or os.environ.get("JEV_STYLE_DEVICE") or os.environ.get("JEV_DEVICE") or None
        kw = {}
        param = inspect.signature(cls).parameters.get("cuda_graphs")
        if param is not None:
            want = _want_cuda_graphs(cuda_graphs, device, runtime_decides=param.default is None)
            if want is not None:
                kw["cuda_graphs"] = want
        elif cuda_graphs:
            raise ValueError(f"the {build.repo} runtime at this revision has no CUDA-graph path (cuda_graphs=True)")
        runtime = cls(folder, device=device, dtype=dtype, verify=verify, **kw)
    elif backend == "mlx":
        runtime = cls(folder, precision=precision, verify=verify)
    else:
        runtime = cls(folder, quant=quant.upper(), binary=scorer, verify=verify)  # None: the runtime's own lookup
    return runtime, rt, build


def _want_cuda_graphs(flag: bool | None, device: str | None, runtime_decides: bool = False) -> bool | None:
    """cuda_graphs= for the runtime; None = leave it to the runtime's own default (``runtime_decides``)."""
    if flag is None:
        env = os.environ.get("JEV_STYLE_CUDA_GRAPHS", "").strip().lower()
        if env and env != "auto":
            flag = env not in ("0", "false", "no", "off")
    if flag is not None:
        return bool(flag)
    if runtime_decides:                             # 2B v4: on for bfloat16 on CUDA; in float32 they gain little
        return None
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
