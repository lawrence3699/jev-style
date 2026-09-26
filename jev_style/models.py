"""Download and load a Jev-Style release from the Hugging Face Hub.

Every build ships its own self-contained runtime next to the weights (``jev_style_decision*.py``):
input rendering, verdict readout, calibration temperatures and token budgets all live there. This
module only picks a build, downloads a pinned revision and imports that runtime, so the server gives
exactly the answers the model card documents.

Backends:

* ``torch``  chaoliangUNSW/Jev-Style-0.8B-Decision-v3        CUDA, Apple MPS or CPU (float32 by default)
* ``mlx``    chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX    Apple silicon, bf16 or 8-bit weights
* ``gguf``   chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF   llama.cpp via the ``jev-score`` scorer
                                                             (build it once, see the GGUF model card)

``auto`` = mlx on Apple silicon when ``mlx`` + ``mlx-lm`` are installed, otherwise torch.

``build_for_repo`` maps a Hub repo id to its build (``JevStyle.from_pretrained``). The three repos above load at
their pinned revisions. Any other repo runs the ``jev_style_decision*.py`` file it ships, so it is refused
unless the caller passes ``trust_remote_code=True``.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any

MODEL_ID = "jev-style-0.8b-decision-v3"
MODEL_NAME = "Jev-Style-0.8B-Decision-v3"
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


_COMMON = ("LICENSE", "NOTICE", "manifest.json", "readout_config.json", "release_config.json", "requirements.txt")

BUILDS: dict[str, Build] = {
    "torch": Build("torch", "chaoliangUNSW/Jev-Style-0.8B-Decision-v3", "d53c8f826f35e811d06529f5c1066dfd60eee00c",
                   "jev_style_decision", "JevStyleDecision",
                   _COMMON + ("jev_style_decision.py", "config.json", "generation_config.json", "chat_template.jinja",
                              "model.safetensors", "tokenizer.json", "tokenizer_config.json")),
    "mlx": Build("mlx", "chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX", "7f14c9fa1491d168a7f70b16acf68baf9d4f7353",
                 "jev_style_decision_mlx", "JevStyleDecisionMLX",
                 _COMMON + ("jev_style_decision_mlx.py",), {"precision_patterns": "{precision}/*"}),
    "gguf": Build("gguf", "chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF", "b8356a83beb560cf34cbe6f9a20c2076e1b532d3",
                  "jev_style_decision_gguf", "JevStyleDecisionGGUF",
                  _COMMON + ("jev_style_decision_gguf.py", "jev_score.cpp", "build_jev_score.sh", "tokenizer/*"),
                  {"quant_file": MODEL_NAME + "-{quant}.gguf"}),
}


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def resolve_backend(backend: str = "auto") -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend != "auto":
        return backend
    if is_apple_silicon() and has_module("mlx") and has_module("mlx_lm"):
        return "mlx"
    return "torch"


def build_for_repo(repo: str, *, trust_remote_code: bool = False) -> Build:
    """The build a Hub repo id names. Known repos keep their pinned revision; others need trust_remote_code."""
    for b in BUILDS.values():
        if b.repo.lower() == repo.lower():
            return b
    if not trust_remote_code:
        known = ", ".join(b.repo for b in BUILDS.values())
        raise ValueError(f"{repo!r} is not a Jev-Style release this package knows ({known}). Loading it runs the "
                         "Python runtime file it ships; pass trust_remote_code=True if you trust it.")
    tail = repo.rsplit("/", 1)[-1].lower()
    backend = "gguf" if "gguf" in tail else "mlx" if "mlx" in tail else "torch"
    return replace(BUILDS[backend], repo=repo, revision=None)


def patterns_for(build: Build, precision: str = "bf16", quant: str = "Q8_0") -> list[str]:
    pats = list(build.patterns)
    if build.backend == "mlx":
        pats.append(build.extra["precision_patterns"].format(precision=precision))
    if build.backend == "gguf":
        pats.append(build.extra["quant_file"].format(quant=quant.upper()))
    return pats


def download(backend: str = "auto", *, precision: str = "bf16", quant: str = "Q8_0",
             revision: str | None = None, build: Build | None = None) -> Path:
    """Download (or reuse from the HF cache) the files one backend needs. Returns the local folder."""
    from huggingface_hub import snapshot_download

    build = build or BUILDS[resolve_backend(backend)]
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


def load(backend: str = "auto", *, model_dir: str | Path | None = None, device: str | None = None,
         dtype: str = "float32", precision: str = "bf16", quant: str = "Q8_0", scorer: str | None = None,
         revision: str | None = None, verify: bool = False, repo: str | None = None,
         trust_remote_code: bool = False) -> tuple[Any, ModuleType, str]:
    """-> (runtime object, runtime module, backend name). ``model_dir`` skips the download; ``repo`` picks the
    build (and so the backend) from a Hub repo id instead of ``backend``."""
    if repo:
        build = build_for_repo(repo, trust_remote_code=trust_remote_code)
        backend = build.backend
    else:
        backend = resolve_backend(backend)
        build = BUILDS[backend]
    folder = Path(model_dir).expanduser() if model_dir else download(backend, precision=precision, quant=quant,
                                                                     revision=revision, build=build)
    rt = import_runtime(folder, build.module)
    cls = getattr(rt, build.cls)
    if backend == "torch":
        runtime = cls(folder, device=device or os.environ.get("JEV_STYLE_DEVICE") or os.environ.get("JEV_DEVICE") or None, dtype=dtype, verify=verify)
    elif backend == "mlx":
        runtime = cls(folder, precision=precision, verify=verify)
    else:
        runtime = cls(folder, quant=quant.upper(), binary=scorer, verify=verify)
    return runtime, rt, backend
