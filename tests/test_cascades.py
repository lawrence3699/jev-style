"""Named cascade releases (jev_style.cascades): cascade-9b, --cascade NAME, download --cascade, jev-style releases."""
import json
import re

import pytest
from conftest import FakeJevK5
from fastapi.testclient import TestClient

from jev_style import JevStyle, cascades, models
from jev_style.cascade import CascadeConfigError, download_tiers, resolve_config, resolve_tiers
from jev_style.jevk5_engine import JEVK5_9B
from jev_style.models import RELEASES

STATE = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."


@pytest.fixture()
def frozen(monkeypatch):
    """cascade-9b with a test threshold (whatever the placeholders in this checkout hold)."""
    monkeypatch.setitem(cascades.CASCADES, "cascade-9b", {**cascades.CASCADES["cascade-9b"], "thresholds": [0.42],
                                                          "frozen_utc": "2026-10-03T00:00:00Z"})


@pytest.fixture()
def unfrozen(monkeypatch):
    monkeypatch.setitem(cascades.CASCADES, "cascade-9b", {**cascades.CASCADES["cascade-9b"], "thresholds": [None],
                                                          "frozen_utc": None})


@pytest.fixture()
def fake_2b(monkeypatch):
    """The 2B tier without weights: load_release returns the fake runtime; records its keyword arguments."""
    from jev_style import fake
    seen = []

    def load_release(backend="auto", **kw):
        seen.append({"backend": backend, **kw})
        return fake.FakeRuntime(), fake, RELEASES["2b-v3"].builds["torch"]
    monkeypatch.setattr(models, "load_release", load_release)
    return seen


def test_cascade_9b_is_the_planned_release():
    d = cascades.CASCADES["cascade-9b"]
    assert d["id"] == "jev-style-cascade-9b" and d["confidence"] == "normalized_pmax"
    assert d["tiers"] == [
        {"name": "2b", "model_id": "jev-style-2b-decision-v3", "target": "local:2b-v3:torch", "dtype": "float32"},
        {"name": "jevk5-9b", "model_id": "alibiserikbay/JevK5-9B", "target": "jevk5:alibiserikbay/JevK5-9B",
         "revision": "d6521a18a86999190e9d775c915af3d6d6772fc4"}]
    assert d["tiers"][1]["revision"] == JEVK5_9B.revision and d["thresholds"] == [0.75]
    assert d["frozen_utc"] == "2026-10-02T21:53:31Z"
    assert d["calibration_sha256"] == "580962e4b2bc86b32b42731eda42536f57321c7d1ca9ac1e3b970bf7dbd7e2d2"
    tiers = resolve_tiers("cascade-9b")                         # the tiers parse whether or not tau is frozen
    assert [t.protocol for t in tiers] == ["systemone", "jevk5"] and tiers[0].dtype == "float32"


def test_every_named_cascade_is_frozen_or_refused():
    for name in cascades.names():
        if cascades.placeholders(name):
            with pytest.raises(CascadeConfigError, match="placeholder"):
                resolve_config(name)
        else:
            cfg = resolve_config(name)
            assert cfg.frozen_utc and cfg.calibration_sha256 and all(0 <= t <= 1 for t in cfg.thresholds)


def test_placeholders_are_named(unfrozen, capsys):
    assert cascades.placeholders("cascade-9b") == ["thresholds[0]", "frozen_utc"]
    with pytest.raises(CascadeConfigError, match=re.escape("thresholds[0], frozen_utc are still a placeholder")):
        JevStyle(cascade="cascade-9b")
    from jev_style.cli import main
    assert main(["decide", STATE, "--cascade", "cascade-9b", "--noul", "x"]) == 2
    assert "not frozen" in capsys.readouterr().err


def test_a_name_resolves_to_a_frozen_config(frozen):
    cfg = resolve_config("Cascade-9B")
    assert cfg.id == "jev-style-cascade-9b" and cfg.thresholds == (0.42,)
    assert cfg.frozen_utc == "2026-10-03T00:00:00Z" and cfg.calibration_sha256.startswith("580962e4")
    assert cfg.sha256 == resolve_config("cascade-9b").sha256 and len(cfg.sha256) == 64
    assert cfg.tiers[1].protocol == "jevk5" and cfg.tiers[1].revision == JEVK5_9B.revision


def test_name_or_file(frozen, tmp_path, monkeypatch):
    other = {"id": "file-cascade", "description": "d", "thresholds": [0.5],
             "tiers": [{"name": "a", "model_id": "a", "target": "fake"},
                       {"name": "b", "model_id": "b", "target": "fake"}]}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(other))
    assert resolve_config(str(p)).id == "file-cascade"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cascade-9b").write_text(json.dumps(other))
    assert resolve_config("cascade-9b").id == "jev-style-cascade-9b"          # the name wins ...
    assert resolve_config("./cascade-9b").id == "file-cascade"                # ... ./ for the file
    with pytest.raises(CascadeConfigError, match="not a named cascade either; known: cascade-9b"):
        resolve_config("cascade-4b")


def test_cascade_9b_end_to_end_in_process(frozen, fake_2b, fake_jevk5, fake_hub):
    js = JevStyle(cascade="cascade-9b", backend="torch", device="cuda")
    assert fake_2b == [{"backend": "torch", "release": "2b-v3", "device": "cuda", "dtype": "float32"}]
    assert [(c["repo"], c["revision"]) for c in fake_hub.calls][-1] == (JEVK5_9B.repo, JEVK5_9B.revision)
    assert FakeJevK5.instances[0].source == str(fake_hub.remote)
    m = js.models()
    entry = m["data"][0]
    assert entry["id"] == "jev-style-cascade-9b" and m["models"][0]["name"] == "jev-style-cascade-9b"
    assert entry["calibration_sha256"].startswith("580962e4") and entry["frozen_utc"] == "2026-10-03T00:00:00Z"
    keys = ("name", "model", "target", "revision", "protocol", "threshold")
    assert [tuple(t[k] for k in keys) for t in entry["tiers"]] == [
        ("2b", "jev-style-2b-decision-v3", "local:2b-v3:torch", None, "systemone", 0.42),
        ("jevk5-9b", "alibiserikbay/JevK5-9B", "jevk5:alibiserikbay/JevK5-9B", JEVK5_9B.revision, "jevk5", None)]
    assert entry["tiers"][1]["backend"] == "jevk5-cuda"
    qs = {f"q{i}": {"type": "noul", "instructions": f"statement {i}"} for i in range(12)}
    out = js.decide(STATE, qs)
    assert out["model"] == "jev-style-cascade-9b" and out["backend"] == "cascade"
    tiers = {a["tier"] for a in out["answers"].values()}
    assert tiers == {1, 2}                                       # fake confidences straddle 0.42 on this sample
    for a in out["answers"].values():
        assert a["tier_model"] == ("jev-style-2b-decision-v3" if a["tier"] == 1 else "alibiserikbay/JevK5-9B")
        assert a["tier"] == 2 or abs(2 * a["noul"] - 1) >= 0.42
    k5_calls = out["timing"]["tiers"][1]["questions"]
    assert k5_calls == sum(a["tier"] == 2 for a in out["answers"].values())


def test_serve_reports_the_cascade(frozen, fake_2b, fake_jevk5, fake_hub):
    from jev_style.server import build_app
    c = TestClient(build_app("torch", cascade="cascade-9b"))
    m = c.get("/v1/models").json()
    assert m["data"][0]["id"] == "jev-style-cascade-9b"
    assert [t["name"] for t in m["data"][0]["tiers"]] == ["2b", "jevk5-9b"]
    h = c.get("/healthz").json()
    assert h["model"] == "jev-style-cascade-9b" and h["release_kind"] == "cascade" and not h["untrained"]
    r = c.post("/v1/systemone", json={"state": STATE, "questions": {"b": {"type": "noul", "instructions": "x"}}})
    assert r.status_code == 200 and r.json()["answers"]["b"]["tier"] in (1, 2)


def test_download_every_tier_even_before_the_freeze(unfrozen, fake_hub, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(models, "download", lambda *a, **k: seen.append((a, k)) or "/hf/2b-torch")
    from jev_style.cli import main
    assert main(["download", "--cascade", "cascade-9b", "--backend", "torch"]) == 0
    assert seen == [(("torch",), {"precision": "bf16", "quant": "Q8_0", "revision": None, "release": "2b-v3"})]
    assert fake_hub.calls[-1]["revision"] == JEVK5_9B.revision
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["2b\t/hf/2b-torch", f"jevk5-9b\t{fake_hub.remote}"]
    (fake_hub.remote / "jevk5_config.json").unlink()            # a download is checked as a load is
    assert main(["download", "--cascade", "cascade-9b"]) == 2


def test_download_tiers_of_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "download", lambda *a, **k: f"/hf/{k['release']}")
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"id": "c", "description": "d", "thresholds": [0.5, 0.5], "tiers": [
        {"name": "a", "model_id": "a", "target": "local:0.8b"}, {"name": "b", "model_id": "b", "target": "fake"},
        {"name": "c", "model_id": "c", "target": "http://127.0.0.1:1"}]}))
    assert [(t.name, f) for t, f in download_tiers(str(p))] == [("a", "/hf/0.8b"), ("b", None), ("c", None)]


def test_releases_lists_releases_and_cascades(capsys):
    from jev_style.cli import main
    assert main(["releases"]) == 0
    out = capsys.readouterr().out
    for s in ("2b-v3", "jev-style-0.8b-decision-v3", "cascade-9b", "jev-style-cascade-9b",
              "jevk5:alibiserikbay/JevK5-9B @ d6521a18", "580962e4"):
        assert s in out
    assert main(["releases", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["releases"]["2b-v3"]["builds"]["torch"]["revision"] == RELEASES["2b-v3"].builds["torch"].revision
    c = d["cascades"]["cascade-9b"]
    assert c["id"] == "jev-style-cascade-9b" and c["tiers"][1]["revision"] == JEVK5_9B.revision
    assert c["placeholders"] == cascades.placeholders("cascade-9b")


def test_cuda_graphs_flag_reaches_cascade_tiers(fake_2b, tmp_path):
    from jev_style.cli import main
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"id": "c", "description": "d", "thresholds": [0.0], "tiers": [
        {"name": "a", "model_id": "a", "target": "local:2b"}, {"name": "b", "model_id": "b", "target": "fake"}]}))
    assert main(["decide", STATE, "--cascade", str(p), "--backend", "torch", "--cuda-graphs", "off",
                 "--noul", "x"]) == 0
    assert fake_2b[-1]["cuda_graphs"] is False and fake_2b[-1]["release"] == "2b"
    assert main(["decide", STATE, "--cascade", str(p), "--noul", "x"]) == 0
    assert fake_2b[-1]["cuda_graphs"] is None                       # auto: on when the model runs on CUDA
