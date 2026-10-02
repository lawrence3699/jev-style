"""In-process JevK5 tiers (``jevk5:<repo or folder>``) on a fake jevk5 package: no GPU, no network."""
import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import pytest
from conftest import FakeJevK5, jevk5_normalized, make_jevk5_folder

from jev_style import JevStyle, JevStyleError, choice, noul, score
from jev_style import jevk5_engine as k5
from jev_style.cascade import CascadeAdapter, CascadeConfig, CascadeConfigError, Tier, TierError, build_tier
from jev_style.client import parse_target
from jev_style.models import MissingBackendError

PIN = k5.JEVK5_9B
STATE = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."
QS = {"n": noul("Does this need urgent human attention?"),
      "c": choice("Which team?", {"returns": "exchanges", "shipping": None, "billing": {"means": "charges"}}),
      "s": score("How frustrated?", ["Calm", {"label": "Frustrated", "description": "clearly unhappy"}, "Very angry"]),
      "one": choice("Only option?", ["only"])}


def jevk5_serve(model, name):
    """An httpx transport doing what jevk5-serve 0.3.3's do_POST does (jevk5/server.py), over ``model``."""
    def handler(request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content)
            answers = {qid: model.decide(body["state"], jevk5_normalized(q)) for qid, q in body["questions"].items()}
        except (ValueError, KeyError, TypeError) as error:
            return httpx.Response(400, json={"error": str(error)})
        tokens = sum(a.pop("input_tokens") for a in answers.values())
        return httpx.Response(200, json={"model": body.get("model") or name, "answers": answers,
                                         "usage": {"input_tokens": tokens, "output_tokens": 0}, "latency_ms": 1.0})
    return httpx.MockTransport(handler)


def http_engine(model, name="k5"):
    return JevStyle(base_url="http://k5", http_client=httpx.Client(base_url="http://k5",
                                                                    transport=jevk5_serve(model, name)))


class Unsure:
    """A tier 1 that is never confident: everything it can answer goes on to tier 2."""

    def decide(self, state, questions, model=None):
        out = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                out[qid] = {"type": "noul", "noul": 0.5}
            else:
                keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
                out[qid] = {"type": q["type"], "probabilities": {k: 1 / len(keys) for k in keys}}
        return {"answers": out, "usage": {"input_tokens": 1}}


def two_tier(engine):
    return CascadeAdapter([Tier("small", "jev-style-2b-decision-v3", Unsure()),
                           Tier("jevk5-9b", PIN.repo, engine, protocol="jevk5")], [0.5], cascade_id="t")


# ----------------------------------------------------------------------------- targets and config
def test_target_and_tier_config():
    assert parse_target("jevk5:alibiserikbay/JevK5-9B") == ("jevk5", {"jevk5": "alibiserikbay/JevK5-9B"})
    assert parse_target("jevk5:/models/k5") == ("jevk5", {"jevk5": "/models/k5"})
    with pytest.raises(ValueError):
        parse_target("jevk5:")
    base = {"id": "c", "description": "d", "thresholds": [0.5],
            "tiers": [{"name": "a", "model_id": "a", "target": "fake"},
                      {"name": "k5", "model_id": PIN.repo, "target": "jevk5:" + PIN.repo, "revision": PIN.revision}]}
    cfg = CascadeConfig.from_dict(base)
    assert cfg.tiers[1].protocol == "jevk5" and cfg.tiers[1].verify is None       # protocol follows the target
    assert CascadeConfig.from_dict(cfg.to_dict()) == cfg
    for change, msg in (({"protocol": "systemone"}, "protocol"), ({"dtype": "float32"}, "dtype"),
                        ({"verify": "yes"}, "verify"), ({"api_key_env": "K"}, "only apply to http")):
        bad = json.loads(json.dumps(base))
        bad["tiers"][1].update(change)
        with pytest.raises(CascadeConfigError, match=msg):
            CascadeConfig.from_dict(bad)
    ok = json.loads(json.dumps(base))
    ok["tiers"][1]["verify"] = False
    assert CascadeConfig.from_dict(ok).to_dict()["tiers"][1]["verify"] is False
    http = json.loads(json.dumps(base))
    http["tiers"][1] = {"name": "k5", "model_id": "m", "target": "http://127.0.0.1:1", "verify": True}
    with pytest.raises(CascadeConfigError, match="verify only applies"):
        CascadeConfig.from_dict(http)


def test_pyproject_has_no_direct_url_dependency():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "git+" not in text and "jevk5" not in text           # PyPI rejects direct URLs; jevk5 is installed apart


# ----------------------------------------------------------------------------- loading
def test_missing_jevk5_fails_before_any_download(monkeypatch):
    import huggingface_hub
    monkeypatch.setitem(sys.modules, "jevk5", None)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("downloaded first"))
    with pytest.raises(MissingBackendError) as ei:
        k5.load_jevk5(PIN.repo)
    assert 'pip install "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.3.3"' in str(ei.value)
    assert "not on PyPI" in str(ei.value)
    cfg = {"id": "c", "description": "d", "thresholds": [0.5],
           "tiers": [{"name": "a", "model_id": "a", "target": "fake"},
                     {"name": "k5", "model_id": "k5", "target": "jevk5:" + PIN.repo}]}
    with pytest.raises(MissingBackendError):
        CascadeAdapter.from_config(cfg)


def test_jevk5_that_cannot_be_imported(monkeypatch, fake_jevk5):
    monkeypatch.setitem(sys.modules, "jevk5.server", None)           # e.g. torch missing under jevk5.runtime
    with pytest.raises(MissingBackendError, match="cannot be imported"):
        k5.import_jevk5()


def test_pinned_repo_downloads_its_revision_verifies_and_loads_locally(fake_jevk5, fake_hub):
    eng = k5.load_jevk5(PIN.repo)
    assert [(c["repo"], c["revision"], c["local_files_only"]) for c in fake_hub.calls] == [
        (PIN.repo, PIN.revision, True), (PIN.repo, PIN.revision, False)]
    assert "allow_patterns" not in fake_hub.calls[1]                     # the whole snapshot
    (model,) = FakeJevK5.instances
    assert model.source == str(fake_hub.remote) and isinstance(model.source, str)   # constructed from the folder
    assert model.load_thread.startswith("jev-style-jevk5")
    assert eng.revision == PIN.revision and eng.backend == "jevk5-cuda" and eng.model_id == PIN.repo
    entry = eng.models()["data"][0]
    assert (entry["temperature"], entry["knockout_temperature"], entry["jevk5_version"]) == (1.316, 1.05, "0.3.3")
    assert eng.release_info()["release_kind"] == "jevk5" and eng.max_len is None


def test_a_complete_cached_snapshot_needs_no_network(fake_jevk5, fake_hub):
    fake_hub.cached = fake_hub.remote
    k5.load_jevk5(PIN.repo)
    assert [c["local_files_only"] for c in fake_hub.calls] == [True]


def test_a_partial_cached_snapshot_is_completed(fake_jevk5, fake_hub, tmp_path):
    partial = make_jevk5_folder(tmp_path / "partial")
    (partial / "model-00002-of-00002.safetensors").unlink()
    fake_hub.cached = partial
    k5.load_jevk5(PIN.repo)
    assert [c["local_files_only"] for c in fake_hub.calls] == [True, False]
    assert FakeJevK5.instances[0].source == str(fake_hub.remote)


@pytest.mark.parametrize("damage,msg", [
    (lambda f: (f / "model-00001-of-00002.safetensors").write_bytes(b"other"), "does not match"),
    (lambda f: (f / "model-00002-of-00002.safetensors").unlink(), "missing"),
    (lambda f: (f / "SHA256SUMS").unlink(), "no SHA256SUMS"),
    (lambda f: (f / "SHA256SUMS").write_text("not a checksum line\n"), "expected '<sha256>  <file>'"),
    (lambda f: (f / "SHA256SUMS").write_text(f"{'0' * 64}  ../outside\n"), "outside the folder"),
])
def test_sha256sums_is_checked_for_the_pinned_release(fake_jevk5, fake_hub, damage, msg):
    damage(fake_hub.remote)
    with pytest.raises(k5.JevK5LoadError, match=re.escape(msg)):
        k5.load_jevk5(PIN.repo)
    assert FakeJevK5.instances == []                                     # never constructed
    if msg != "missing":
        k5.load_jevk5(PIN.repo, verify=False)                           # the caller may skip the check
        assert len(FakeJevK5.instances) == 1


def test_sha256sums_binary_mode_lines_and_comments(tmp_path):
    f = make_jevk5_folder(tmp_path / "r", sums=False)
    lines = ["# sha256sum -b *"] + [f"{hashlib.sha256((f / n).read_bytes()).hexdigest().upper()} *./{n}"
                                    for n in ("jevk5_config.json", "config.json")]
    (f / "SHA256SUMS").write_text("\n".join(lines) + "\n\n")
    assert k5.verify_sha256sums(f) == 2


@pytest.mark.parametrize("config,msg", [(None, "no jevk5_config.json"), ({"knockout_temperature": 1.05}, "temperature"),
                                        ({"temperature": 0}, "temperature"), ({"temperature": True}, "temperature"),
                                        ({"temperature": 1.3, "knockout_temperature": -1}, "knockout_temperature")])
def test_no_calibration_config_no_load(fake_jevk5, tmp_path, config, msg):
    folder = make_jevk5_folder(tmp_path / "r", config=config)
    with pytest.raises(k5.JevK5LoadError, match=msg):
        k5.load_jevk5(str(folder), verify=False)                       # refused even unverified
    assert FakeJevK5.instances == []


def test_config_that_is_not_json(fake_jevk5, tmp_path):
    folder = make_jevk5_folder(tmp_path / "r")
    (folder / "jevk5_config.json").write_text("{temperature: 1.3")
    with pytest.raises(k5.JevK5LoadError, match="not valid JSON"):
        k5.load_jevk5(str(folder))


def test_local_folder_is_used_as_is(fake_jevk5, monkeypatch, tmp_path):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("no download"))
    folder = make_jevk5_folder(tmp_path / "r", sums=False)
    eng = k5.load_jevk5(str(folder))                                    # no SHA256SUMS: not checked by default
    assert FakeJevK5.instances[0].source == str(folder) and eng.name == str(folder)
    with pytest.raises(k5.JevK5LoadError, match="no SHA256SUMS"):
        k5.load_jevk5(str(folder), verify=True)


def test_other_repos_load_main_or_their_revision_unverified(fake_jevk5, fake_hub):
    (fake_hub.remote / "SHA256SUMS").write_text(f"{'0' * 64}  config.json\n")    # would fail if checked
    k5.load_jevk5("someone/JevK5-Other")
    assert fake_hub.calls[-1]["revision"] is None
    k5.load_jevk5(PIN.repo, revision="v0.3")                           # not the pinned revision: not checked
    assert fake_hub.calls[-1]["revision"] == "v0.3"
    with pytest.raises(k5.JevK5LoadError):
        k5.load_jevk5(PIN.repo, revision="v0.3", verify=True)


# ----------------------------------------------------------------------------- answering as jevk5-serve
def test_answers_are_jevk5_serves(fake_jevk5, tmp_path):
    eng = k5.load_jevk5(str(make_jevk5_folder(tmp_path / "r")))
    qs = {"a": {"type": "noul", "instructions": "x"},
          "b": {"type": "choice", "instructions": "y", "criteria": ["p", "q"]},
          "c": {"type": "score", "instructions": "z", "criteria": ["lo", "hi"]}}
    out = eng.systemone({"state": {"k": [1, 2]}, "questions": qs, "model": "asked-for"})
    assert eng.systemone({"state": "s", "questions": qs})["model"] == eng.name and out["model"] == "asked-for"
    model = FakeJevK5.instances[0]
    assert len(model.threads) == 6 and set(model.threads) == {model.load_thread}    # every call on the loading thread
    want = {qid: model.decide({"k": [1, 2]}, jevk5_normalized(q)) for qid, q in qs.items()}
    tokens = sum(a.pop("input_tokens") for a in want.values())
    assert out["answers"] == want and out["usage"] == {"input_tokens": tokens, "output_tokens": 0}


def test_one_bad_question_fails_the_request_as_over_http(fake_jevk5, tmp_path):
    eng = k5.load_jevk5(str(make_jevk5_folder(tmp_path / "r")))
    over_http = http_engine(FakeJevK5(tmp_path / "r"))
    for qs in ({"ok": {"type": "noul", "instructions": "x"}, "bad": {"type": "choice", "instructions": "x",
                                                                      "criteria": {"only": None}}},
               {"bad": {"type": "score", "criteria": ["a", "b"]}}, {"bad": {"type": "maybe", "instructions": "x"}}):
        errors = []
        for engine in (JevStyle(adapter=eng), over_http):
            with pytest.raises(JevStyleError) as ei:
                engine.decide("s", qs)
            errors.append(ei.value)
        a, b = errors
        assert (a.status, a.code, a.message, a.question, a.body) == (b.status, b.code, b.message, b.question, b.body)
        assert a.status == 400 and TierError.from_exception(a) == TierError.from_exception(b)
        assert TierError.from_exception(a).code == "tier_error"


def test_cascade_in_process_and_over_http_give_identical_answers(fake_jevk5, tmp_path):
    folder = make_jevk5_folder(tmp_path / "r")
    eng = k5.load_jevk5(str(folder))
    live, via_http = two_tier(JevStyle(adapter=eng)), two_tier(http_engine(FakeJevK5(folder)))
    for body in ({"state": STATE, "questions": QS},
                 {"state": {"ticket": 7, "text": "火事です"}, "questions": {
                     "i": noul({"ask": "Is this an emergency?"}),
                     "m": choice("Which intent?", [f"intent_{i}" for i in range(20)]),
                     "l": score("Severity", [{"label": "1"}, {"label": "2", "description": ""}, ["three", 3]])}}):
        a, b = live.systemone(body), via_http.systemone(body)
        assert a["answers"] == b["answers"] and a["usage"] == b["usage"]
        tiers = {k: v["tier"] for k, v in a["answers"].items()}
        assert set(tiers.values()) == {2} or tiers == {"n": 2, "c": 2, "s": 2, "one": 1}   # 'one' is not sent


def test_build_tier_passes_revision_and_verify(fake_jevk5, fake_hub):
    cfg = CascadeConfig.from_dict({"id": "c", "description": "d", "thresholds": [0.5], "tiers": [
        {"name": "a", "model_id": "a", "target": "fake"},
        {"name": "k5", "model_id": PIN.repo, "target": "jevk5:" + PIN.repo, "revision": PIN.revision}]})
    tier = build_tier(cfg.tiers[1], backend="torch", device="cuda", dtype="bfloat16", cuda_graphs=True)
    assert tier.protocol == "jevk5" and tier.backend == "jevk5-cuda" and tier.max_len is None
    assert fake_hub.calls[-1]["revision"] == PIN.revision and not tier.untrained
    (fake_hub.remote / "SHA256SUMS").write_text(f"{'0' * 64}  config.json\n")
    with pytest.raises(k5.JevK5LoadError):
        build_tier(cfg.tiers[1])                                          # verified by default
    unchecked = CascadeConfig.from_dict({**cfg.to_dict(), "tiers": [
        cfg.to_dict()["tiers"][0], {**cfg.to_dict()["tiers"][1], "verify": False}]})
    assert build_tier(unchecked.tiers[1]).backend == "jevk5-cuda"


def test_eval_engine_and_cli_take_a_jevk5_target(fake_jevk5, tmp_path, capsys):
    from jev_style.cli import main
    from jev_style.evaluate import build_engine
    folder = make_jevk5_folder(tmp_path / "r")
    out = build_engine(f"jevk5:{folder}", "auto")("s", {"x": {"type": "noul", "instructions": "y"}})
    assert out["answers"]["x"]["type"] == "noul" and "state_tokens" not in out["usage"]
    ex = str(Path(__file__).resolve().parents[1] / "examples" / "support_tickets.jsonl")
    assert main(["eval", ex, "--limit", "3", "--server", "fake", "--server", f"k5=jevk5:{folder}"]) == 0
    assert "k5" in capsys.readouterr().out
    bad = make_jevk5_folder(tmp_path / "nocal", config=None)
    assert main(["eval", ex, "--server", "fake", "--server", f"k5=jevk5:{bad}"]) == 2
    assert "jevk5_config.json" in capsys.readouterr().err


def test_sha256sums_accepts_a_hub_cache_snapshot_of_symlinks(tmp_path):
    # A Hugging Face cache snapshot is a folder of symlinks into ../../blobs/: the check follows them.
    from jev_style.jevk5_engine import verify_sha256sums
    blobs, snap = tmp_path / "blobs", tmp_path / "snapshots" / "rev"
    blobs.mkdir()
    snap.mkdir(parents=True)
    lines = []
    for name, data in (("config.json", b"{}"), ("model.safetensors", b"weights")):
        blob = blobs / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        (snap / name).symlink_to(blob)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    (snap / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    assert verify_sha256sums(snap) == 2
    (blobs / hashlib.sha256(b"weights").hexdigest()).write_bytes(b"tampered")
    with pytest.raises(Exception, match="sha256 does not match"):
        verify_sha256sums(snap)


def test_a_shadowing_jevk5_folder_gets_a_clear_error(monkeypatch, tmp_path):
    # `import jevk5` from a directory holding a checkout of the jevk5 repo finds a namespace package without JevK5.
    import types
    from jev_style.jevk5_engine import import_jevk5
    from jev_style.models import MissingBackendError
    shadow = types.ModuleType("jevk5")
    shadow.__path__ = [str(tmp_path / "jevk5")]
    monkeypatch.setitem(sys.modules, "jevk5", shadow)
    monkeypatch.setitem(sys.modules, "jevk5.server", types.ModuleType("jevk5.server"))
    with pytest.raises(MissingBackendError, match="shadow"):
        import_jevk5()
