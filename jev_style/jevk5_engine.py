"""In-process JevK5: the ``jevk5:<Hub repo or local folder>`` engine target.

JevK5 is a third-party typed-decision model family by alibiserikbay: weights on the Hugging Face Hub (e.g.
``alibiserikbay/JevK5-9B``), runtime ``jevk5`` from github.com/allebee/jevk5, both Apache-2.0. A cascade can reach
it over HTTP through ``jevk5-serve`` (an ``http(s)://`` tier with ``"protocol": "jevk5"``) or, with this module, in
this process. Both run the same code, so they give identical answers:

* one request = ``jevk5.server.normalized(q)`` then ``model.decide(state, q)`` for each question, in order, on the
  request as JSON carries it (the body is JSON round-tripped the way the HTTP client encodes it);
* a ``ValueError`` / ``KeyError`` / ``TypeError`` for any question fails the whole request: jevk5-serve answers 400
  ``{"error": "<message>"}``; here the ``JevStyleError`` the HTTP client raises for that answer is raised;
* the response is jevk5-serve's: ``model`` (the request's, else the source), ``answers`` (their ``input_tokens``
  moved to ``usage``), ``usage`` = {input_tokens, output_tokens}, ``latency_ms``.

The questions are shaped for JevK5 by the cascade tier (``jev_style.cascade.jevk5_questions``), for both transports;
a ``jevk5:`` tier always uses the ``jevk5`` protocol.

Loading (``load_jevk5``):

1. ``import jevk5``, before any download. It is not on PyPI: ``pip install "jevk5[fast] @
   git+https://github.com/allebee/jevk5@v0.3.3"`` (``MissingBackendError`` says so);
2. a Hub repo is downloaded with ``huggingface_hub.snapshot_download(repo, revision=...)`` (the whole snapshot, as
   JevK5's README does); a pinned repo (``PINNED``) defaults to its pinned revision. A local folder is used as is;
3. ``SHA256SUMS`` (``<sha256>  <file>`` lines, as ``sha256sum`` writes them) is checked when ``verify`` is on, which
   is the default for the pinned revision of a pinned repo (it hashes every listed file, ~19 GB for the 9B);
4. ``jevk5_config.json`` must be in the folder with a positive ``temperature``: without it ``jevk5.JevK5`` silently
   runs uncalibrated (temperature 1.0), so loading is refused (``JevK5LoadError``);
5. ``jevk5.JevK5(<folder>)`` is built with jevk5-serve's defaults (CUDA, bfloat16, CUDA graphs) on the engine's one
   worker thread, which then runs every model call, one request at a time (as jevk5-serve's lock does).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

from .models import MissingBackendError

log = logging.getLogger("jev_style.jevk5")

JEVK5_INSTALL = 'pip install "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.3.3"'
CONFIG_FILE = "jevk5_config.json"
SUMS_FILE = "SHA256SUMS"


@dataclass(frozen=True)
class JevK5Pin:
    repo: str           # Hub repo id
    revision: str       # commit sha
    version: str        # the model version at that commit
    runtime: str        # the jevk5 package version it is served with


JEVK5_9B = JevK5Pin("alibiserikbay/JevK5-9B", "d6521a18a86999190e9d775c915af3d6d6772fc4", "v0.3.3", "0.3.3")
PINNED: dict[str, JevK5Pin] = {JEVK5_9B.repo.lower(): JEVK5_9B}


class JevK5LoadError(RuntimeError):
    """A JevK5 folder that must not be loaded: no calibration config, or files that do not match SHA256SUMS."""


def jevk5_thread() -> ThreadPoolExecutor:
    """One worker thread that loads JevK5 (its CUDA graphs are recorded there) and runs every call."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-style-jevk5")


# ----------------------------------------------------------------------------- the jevk5 package
def import_jevk5() -> tuple[ModuleType, ModuleType]:
    """-> (``jevk5``, ``jevk5.server``), or MissingBackendError with the install line."""
    try:
        pkg = importlib.import_module("jevk5")
        server = importlib.import_module("jevk5.server")
    except ImportError as e:
        if getattr(e, "name", None) == "jevk5":
            raise MissingBackendError(
                "an in-process JevK5 tier (jevk5:<repo or folder>) needs the jevk5 package (allebee/jevk5), which is "
                f"not on PyPI: {JEVK5_INSTALL}. Or run jevk5-serve and use an http(s):// tier with "
                '"protocol": "jevk5".') from None
        raise MissingBackendError(f"jevk5 is installed but cannot be imported ({e}); install it with its "
                                  f"dependencies: {JEVK5_INSTALL}") from None
    if not hasattr(pkg, "JevK5"):                   # e.g. a checkout of the jevk5 repo in the working directory
        where = list(getattr(pkg, "__path__", [])) or getattr(pkg, "__file__", "?")
        raise MissingBackendError(f"'import jevk5' found {where}, which is not the jevk5 package (no JevK5 class); "
                                  "a folder named jevk5 in the working directory can shadow the installed package. "
                                  f"Run from another directory, or install it: {JEVK5_INSTALL}")
    return pkg, server


# ----------------------------------------------------------------------------- files
def pin_for(source: str) -> JevK5Pin | None:
    return PINNED.get(source.lower())


def is_folder(source: str | Path) -> bool:
    return Path(source).expanduser().is_dir()


def read_sha256sums(path: Path) -> dict[str, str]:
    """``<sha256>  <file>`` (or ``<sha256> *<file>``) lines -> {file: sha256}. Blank and ``#`` lines are skipped."""
    out: dict[str, str] = {}
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not m:
            raise JevK5LoadError(f"{path}:{ln}: expected '<sha256>  <file>', got {line[:80]!r}")
        name = m.group(2)
        out[name[2:] if name.startswith("./") else name] = m.group(1).lower()
    if not out:
        raise JevK5LoadError(f"{path} lists no files")
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256sums(folder: Path) -> int:
    """Check every file SHA256SUMS lists (each must exist and match). Returns the number of files checked."""
    folder = Path(folder)
    sums = folder / SUMS_FILE
    if not sums.is_file():
        raise JevK5LoadError(f"{folder} has no {SUMS_FILE} to verify against (pass verify=False, or "
                             '"verify": false on the tier, to load it unchecked)')
    expected = read_sha256sums(sums)
    log.info("JevK5: checking %d file(s) in %s against %s ...", len(expected), folder, SUMS_FILE)
    bad = []
    for name, digest in expected.items():
        if name == SUMS_FILE:
            continue
        f = folder / name
        # The listed NAME must stay inside the folder (no absolute paths, no ".."). Where the file finally lives is
        # not checked: a Hugging Face cache snapshot is a folder of symlinks into the cache's blobs/ directory.
        parts = PurePosixPath(name).parts
        if not parts or PurePosixPath(name).is_absolute() or ".." in parts or "\\" in name:
            bad.append(f"{name}: outside the folder")
        elif not f.is_file():
            bad.append(f"{name}: missing")
        elif _sha256(f) != digest:
            bad.append(f"{name}: sha256 does not match")
    if bad:
        raise JevK5LoadError(f"{folder} does not match its {SUMS_FILE}: " + "; ".join(bad[:10])
                             + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else ""))
    return len(expected)


def check_config(folder: Path) -> dict:
    """JevK5's calibration config, which must be next to the weights (see the module docstring)."""
    path = Path(folder) / CONFIG_FILE
    if not path.is_file():
        raise JevK5LoadError(f"{folder} has no {CONFIG_FILE}: JevK5 would run uncalibrated (temperature 1.0) "
                             "without it, so it is not loaded. Is this a JevK5 release folder?")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise JevK5LoadError(f"{path} is not valid JSON: {e}") from None
    for key, required in (("temperature", True), ("knockout_temperature", False)):
        v = cfg.get(key) if isinstance(cfg, dict) else None
        if v is None and not required:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise JevK5LoadError(f"{path}: {key} must be a positive number, got {v!r} (JevK5 would fall back to an "
                                 "uncalibrated default), so it is not loaded")
    return cfg


def _complete(folder: Path) -> bool:
    """A cached snapshot that has the config and every file its SHA256SUMS lists (else download the rest)."""
    if not (folder / CONFIG_FILE).is_file():
        return False
    sums = folder / SUMS_FILE
    try:
        return not sums.is_file() or all((folder / n).is_file() for n in read_sha256sums(sums))
    except (OSError, JevK5LoadError):
        return False


def download(source: str, revision: str | None = None, verify: bool | None = None) -> Path:
    """The local folder of a JevK5 source, checked: a local folder as is; a Hub repo downloaded (or reused from the
    HF cache) at ``revision``, default the pinned one for a pinned repo, else its main branch. ``verify``:
    check SHA256SUMS (None = on for the pinned revision of a pinned repo). The calibration config is always
    required. Needs only huggingface_hub, not jevk5."""
    if is_folder(source):
        folder, pinned = Path(source).expanduser(), False
        if revision:
            log.info("JevK5: %s is a local folder; revision %s is recorded only", source, revision)
    else:
        from huggingface_hub import snapshot_download
        pin = pin_for(source)
        rev = revision or (pin.revision if pin else None)
        if rev is None:
            log.warning("JevK5: %s has no pinned revision in jev-style; loading its main branch", source)
        pinned = pin is not None and rev == pin.revision
        try:                                    # pinned revision already cached: no network round trip
            folder = Path(snapshot_download(source, revision=rev, local_files_only=True))
            if not _complete(folder):
                raise FileNotFoundError(folder)
        except Exception:  # noqa: BLE001 - not cached (or partly): download
            folder = Path(snapshot_download(source, revision=rev))
    if verify if verify is not None else pinned:
        verify_sha256sums(folder)
    check_config(folder)
    return folder


# ----------------------------------------------------------------------------- the engine
class JevK5Engine:
    """The ``Adapter`` surface (``systemone``, ``models``, ``release_info``, ``model_id``, ``backend``, ``max_len``,
    ``head_max``) over an in-process ``jevk5.JevK5``, answering as jevk5-serve does (see the module docstring).
    Thread-safe: every model call runs on ``worker``, one request at a time."""

    def __init__(self, model: Any, normalized: Any, name: str, *, worker: ThreadPoolExecutor | None = None,
                 revision: str | None = None, folder: Path | None = None, config: dict | None = None,
                 jevk5_version: str | None = None):
        self.model, self.normalized, self.name = model, normalized, name
        self.worker = worker or jevk5_thread()
        self.revision, self.folder, self.config = revision, folder, dict(config or {})
        self.jevk5_version = jevk5_version
        self.release_date = None
        self.description = f"JevK5 {name} (third party, allebee/jevk5 {jevk5_version or '?'}), in-process"

    @classmethod
    def load(cls, source: str, *, revision: str | None = None, verify: bool | None = None) -> "JevK5Engine":
        pkg, server = import_jevk5()                  # fail fast, before a ~19 GB download
        version = getattr(pkg, "__version__", None)
        pin = None if is_folder(source) else pin_for(source)
        if pin and version != pin.runtime:
            log.warning("JevK5: %s %s is served with jevk5 %s; jevk5 %s is installed", pin.repo, pin.version,
                        pin.runtime, version)
        folder = download(source, revision=revision, verify=verify)
        config = check_config(folder)
        log.info("JevK5: loading %s from %s (temperature %s, knockout_temperature %s)", source, folder,
                 config.get("temperature"), config.get("knockout_temperature", "jevk5 default"))
        worker = jevk5_thread()
        model = worker.submit(pkg.JevK5, str(folder)).result()      # jevk5-serve's defaults: cuda, bf16, graphs
        if pin and not revision:
            revision = pin.revision
        return cls(model, server.normalized, source, worker=worker, revision=revision, folder=folder,
                   config=config, jevk5_version=version)

    # -- Adapter surface -----------------------------------------------------------------------
    @property
    def model_id(self) -> str:
        return self.name

    @property
    def backend(self) -> str:
        dev = getattr(self.model, "device", None)
        return f"jevk5-{dev}" if dev else "jevk5"

    max_len = None                      # JevK5 has no input budget of its own (long inputs skip its graphs)
    head_max = None

    def systemone(self, body: Any) -> dict:
        """jevk5-serve's ``POST /v1/systemone`` (allebee/jevk5 v0.3.3 ``jevk5/server.py``), in-process."""
        started = time.perf_counter()
        body = json.loads(json.dumps(body, ensure_ascii=False, allow_nan=False))     # what the HTTP client sends
        try:
            answers = self.worker.submit(self._decide_all, body).result()
        except (ValueError, KeyError, TypeError) as error:
            from .client import JevStyleError             # = what the HTTP client raises for jevk5-serve's 400
            raise JevStyleError(400, "http_error", str(error), body={"error": str(error)}) from None
        tokens = sum(a.pop("input_tokens") for a in answers.values())
        payload = {"model": body.get("model") or self.name, "answers": answers,
                   "usage": {"input_tokens": tokens, "output_tokens": 0},
                   "latency_ms": round((time.perf_counter() - started) * 1e3, 2)}
        return json.loads(json.dumps(payload))          # the body the HTTP client would read back

    def _decide_all(self, body: dict) -> dict:
        return {qid: self.model.decide(body["state"], self.normalized(q)) for qid, q in body["questions"].items()}

    def models(self) -> dict:
        entry = {"id": self.name, "backend": self.backend, "revision": self.revision,
                 "jevk5_version": self.jevk5_version, "temperature": self.config.get("temperature"),
                 "knockout_temperature": self.config.get("knockout_temperature")}
        return {"object": "list", "data": [entry],
                "models": [{"name": self.name, "release_date": None, "description": self.description}]}

    def release_info(self) -> dict:
        return {"model_name": self.name, "release_kind": "jevk5", "untrained": False, "acceptance_status": None}

    def close(self) -> None:
        self.worker.shutdown(wait=False)


def load_jevk5(source: str, *, revision: str | None = None, verify: bool | None = None) -> JevK5Engine:
    """An in-process JevK5 engine for a Hub repo id or a local folder (see the module docstring)."""
    return JevK5Engine.load(source, revision=revision, verify=verify)
