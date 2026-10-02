import hashlib
import json
import math
import sys
import threading
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jev_style.server import build_app


@pytest.fixture(scope="session")
def app():
    return build_app(fake=True)


@pytest.fixture()
def client(app):
    return TestClient(app)


# ----------------------------------------------------------------------------- a fake jevk5 package (no torch, no GPU)
def jevk5_normalized(question: dict) -> dict:
    """Mirrors ``jevk5.server.normalized`` (allebee/jevk5 v0.3.3, Apache-2.0)."""
    if question.get("type") not in ("noul", "choice", "score"):
        raise ValueError(f"unknown question type {question.get('type')!r}")
    if "instructions" not in question:
        raise ValueError("question is missing instructions")
    criteria = question.get("criteria")
    if question["type"] == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice criteria must name at least two options")
    if question["type"] == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
        raise ValueError("score criteria must list at least two levels")
    return {**question, "criteria": criteria}


class FakeJevK5:
    """``jevk5.JevK5``'s interface: reads jevk5_config.json from the folder it is given, answers in
    ``jevk5.prompt.answer``'s shape with hash-based probabilities, records the threads it runs on."""
    device = "cuda"
    instances: list = []

    def __init__(self, source):
        self.source = source
        self.config = json.loads((Path(source) / "jevk5_config.json").read_text())
        self.load_thread = threading.current_thread().name
        self.threads = []
        FakeJevK5.instances.append(self)

    def decide(self, state, question):
        self.threads.append(threading.current_thread().name)
        crit = question["criteria"]
        keys = (["true", "false"] if question["type"] == "noul" else list(crit) if question["type"] == "choice"
                else [str(i) for i in range(len(crit))])
        raw = [int(hashlib.sha256(json.dumps([state, question["instructions"], k, crit]).encode()).hexdigest()[:8], 16)
               / 2**32 * 4 / self.config["temperature"] for k in keys]
        e = [math.exp(x - max(raw)) for x in raw]
        probs = {k: v / sum(e) for k, v in zip(keys, e)}
        out = {"type": question["type"], "confidence": max(probs.values()), "input_tokens": 20 + 5 * len(keys)}
        if question["type"] == "noul":
            out["noul"] = probs["true"]
        elif question["type"] == "choice":
            out.update(choice=max(probs, key=probs.get), probabilities=probs)
        else:
            out.update(score=sum(int(k) * v for k, v in probs.items()), probabilities=probs)
        return out


@pytest.fixture()
def fake_jevk5(monkeypatch):
    """Injects a fake ``jevk5`` / ``jevk5.server`` into sys.modules; returns the package module."""
    pkg = types.ModuleType("jevk5")
    server = types.ModuleType("jevk5.server")
    server.normalized = jevk5_normalized
    FakeJevK5.instances = []
    pkg.__version__, pkg.JevK5, pkg.server = "0.3.3", FakeJevK5, server
    monkeypatch.setitem(sys.modules, "jevk5", pkg)
    monkeypatch.setitem(sys.modules, "jevk5.server", server)
    return pkg


JEVK5_CONFIG = {"temperature": 1.316, "knockout_temperature": 1.05}


def make_jevk5_folder(folder: Path, *, config=JEVK5_CONFIG, sums: bool = True) -> Path:
    """A JevK5 release folder: config, weights stand-in, SHA256SUMS (``sha256  file`` lines)."""
    folder.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (folder / "jevk5_config.json").write_text(json.dumps(config))
    (folder / "config.json").write_text('{"model_type": "qwen3_5"}')
    (folder / "model-00001-of-00002.safetensors").write_bytes(b"\x00weights-1" * 100)
    (folder / "model-00002-of-00002.safetensors").write_bytes(b"\x00weights-2" * 100)
    if sums:
        names = sorted(p.name for p in folder.iterdir() if p.name != "SHA256SUMS")
        (folder / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256((folder / n).read_bytes()).hexdigest()}  {n}\n" for n in names))
    return folder


class FakeHub:
    """Stands in for huggingface_hub.snapshot_download: records calls; ``cached`` = what local_files_only finds
    (None: nothing cached), ``remote`` = the folder a download returns."""

    def __init__(self, remote: Path, cached: Path | None = None):
        self.remote, self.cached, self.calls = remote, cached, []

    def __call__(self, repo_id, revision=None, local_files_only=False, **kw):
        self.calls.append({"repo": repo_id, "revision": revision, "local_files_only": local_files_only, **kw})
        if local_files_only:
            if self.cached is None:
                raise FileNotFoundError("not in the cache")
            return str(self.cached)
        return str(self.remote)


@pytest.fixture()
def fake_hub(monkeypatch, tmp_path):
    import huggingface_hub
    hub = FakeHub(make_jevk5_folder(tmp_path / "hub-snapshot"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    return hub
