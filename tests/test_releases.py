import re

import pytest

from jev_style import models
from jev_style.models import ALIASES, DEFAULT_RELEASE, RELEASES, build_for_repo, get_release, release_keys


def test_published_releases_are_pinned_and_named_consistently():
    assert RELEASES[DEFAULT_RELEASE].published
    for key, r in RELEASES.items():
        assert r.model_id == r.name.lower() and set(r.builds) == {"torch", "mlx", "gguf"}
        suffix = {"torch": "", "mlx": "-MLX", "gguf": "-GGUF"}
        for backend, b in r.builds.items():
            assert (b.backend, b.release, b.repo) == (backend, key, "chaoliangUNSW/" + r.name + suffix[backend])
            if r.published:
                assert re.fullmatch(r"[0-9a-f]{40}", b.revision)
        if r.published:
            assert r.release_date


def test_release_keys_and_aliases():
    published = release_keys()
    assert DEFAULT_RELEASE in published and "0.8b" in published
    for alias, key in ALIASES.items():
        assert get_release(alias.upper()) is RELEASES[key]
        assert (alias in published) == RELEASES[key].published
    assert set(release_keys(published_only=False)) == set(RELEASES) | set(ALIASES)
    with pytest.raises(ValueError, match="unknown release"):
        get_release("9b")


def test_env_picks_the_release(monkeypatch):
    monkeypatch.setenv("JEV_STYLE_RELEASE", "2b")
    assert get_release().key == "2b-v4"                              # "2b" = the newest 2B (0.6.0: v4)
    assert get_release("2b-v3").key == "2b-v3"
    assert get_release("0.8b").key == "0.8b-v3"


def test_unpinned_release_needs_trust_remote_code_to_download():
    unpinned = [b for r in RELEASES.values() if not r.published for b in r.builds.values()]
    for b in unpinned:
        with pytest.raises(ValueError, match="trust_remote_code"):
            models.download(build=b)


def test_resolve_backend_per_release(monkeypatch):
    monkeypatch.setattr(models, "is_apple_silicon", lambda: False)
    assert models.resolve_backend("auto", "2b") == "torch"
    assert models.resolve_backend("gguf", "2b-v3") == "gguf"
    monkeypatch.setattr(models, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(models, "has_module", lambda name: True)
    assert models.resolve_backend("auto", "0.8b") == "mlx"


def test_build_for_repo_covers_every_release():
    for r in RELEASES.values():
        for b in r.builds.values():
            assert build_for_repo(b.repo.upper()) is b
    assert build_for_repo("someone/x-GGUF", trust_remote_code=True).release == "custom"


class _Runtime:
    def __init__(self, folder, **kw):
        self.folder, self.kw = folder, kw


class _Module:
    JevStyleDecision = JevStyleDecisionMLX = JevStyleDecisionGGUF = _Runtime


def test_load_release_from_a_local_folder_needs_no_pin(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "import_runtime", lambda folder, module: _Module)
    monkeypatch.setattr(models, "require_backend", lambda build: None)   # runs without torch / mlx installed (CI: [dev])
    runtime, rt, build = models.load_release("torch", release="2b", model_dir=tmp_path, device="cpu")
    assert (build.release, build.backend, runtime.folder) == ("2b-v4", "torch", tmp_path)
    assert runtime.kw == {"device": "cpu", "dtype": "float32", "verify": False}
    runtime, _, build = models.load_release("gguf", release="2b", model_dir=tmp_path, quant="q4_k_m", scorer="/s")
    assert runtime.kw == {"quant": "Q4_K_M", "binary": "/s", "verify": False}
    assert models.load("mlx", release="2b", model_dir=tmp_path)[2] == "mlx"


@pytest.mark.parametrize("key,backend,shares", [("0.8b-v3", "mlx", True), ("2b-v3", "mlx", False),
                                                ("0.8b-v3", "torch", False)])
def test_adapter_reports_the_release_and_gates_prefix_sharing(monkeypatch, key, backend, shares):
    from jev_style import adapter, fake, fastpath
    calls = []
    build = RELEASES[key].builds[backend]
    monkeypatch.setattr(models, "load_release", lambda *a, **k: (fake.FakeRuntime(), fake, build))
    monkeypatch.setattr(fastpath, "enable_prefix_sharing", lambda runtime, rt: calls.append(1))
    a = adapter.build_adapter(backend, release=key)
    assert a.model_id == RELEASES[key].model_id and a.backend == backend
    info = a.models()["models"][0]
    assert info["name"] == a.model_id and info["release_date"] == RELEASES[key].release_date
    assert RELEASES[key].title in info["description"]
    assert bool(calls) == shares
    body = a.systemone({"state": "I was charged twice.", "questions": {"b": {"type": "noul", "instructions": "x"}}})
    assert body["model"] == a.model_id


def test_adapter_for_a_custom_repo_reports_the_repo(monkeypatch):
    from jev_style import adapter, fake
    build = build_for_repo("someone/Other-Decision", trust_remote_code=True)
    monkeypatch.setattr(models, "load_release", lambda *a, **k: (fake.FakeRuntime(), fake, build))
    a = adapter.build_adapter(repo=build.repo, trust_remote_code=True)
    assert a.model_id == "someone/Other-Decision" and a.release_date is None


def test_eval_local_spec_parses_release_and_backend(monkeypatch):
    from jev_style import client, evaluate
    seen = []

    class Recorder:
        def __init__(self, **kw):
            seen.append(kw)
    monkeypatch.setattr(client, "JevStyle", Recorder)
    evaluate.build_engine("local:2b:mlx", "auto")
    evaluate.build_engine("local:gguf", "auto")
    evaluate.build_engine("local", "torch")
    assert seen == [{"backend": "mlx", "release": "2b"}, {"backend": "gguf"}, {"backend": "torch"}]


def test_cli_offers_only_published_releases():
    from jev_style.cli import main
    unpublished = [a for a, k in ALIASES.items() if not RELEASES[k].published]
    for alias in unpublished:
        with pytest.raises(SystemExit):
            main(["download", "--release", alias])


@pytest.mark.parametrize("apple,backend,hint", [(False, "torch", "jev-style[torch]"), (True, "torch", "jev-style[mlx]"),
                                                (True, "mlx", "jev-style[mlx]")])
def test_missing_backend_fails_before_download(monkeypatch, apple, backend, hint):
    monkeypatch.setattr(models, "is_apple_silicon", lambda: apple)
    monkeypatch.setattr(models, "has_module", lambda name: False)
    monkeypatch.setattr(models, "download", lambda *a, **k: pytest.fail("downloaded before the backend check"))
    with pytest.raises(models.MissingBackendError, match=re.escape(hint)):
        models.load_release(backend)


def test_gguf_needs_no_python_extra(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "has_module", lambda name: False)
    monkeypatch.setattr(models, "import_runtime", lambda folder, module: _Module)
    runtime, _, build = models.load_release("gguf", model_dir=tmp_path)
    assert build.backend == "gguf"


# the files of the 2B repos at their pinned revisions (Hub tree, 2026-09-27), without figures/ and validation/
HUB_2B = {
    "torch": ["LICENSE", "NOTICE", "README.md", "chat_template.jinja", "config.json", "eval_results.json",
              "generation_config.json", "jev_style_decision.py", "manifest.json", "model-00001-of-00002.safetensors",
              "model-00002-of-00002.safetensors", "model.safetensors.index.json", "readout_config.json",
              "release_config.json", "requirements.txt", "tokenizer.json", "tokenizer_config.json"],
    "gguf": ["Jev-Style-2B-Decision-v3-F16.gguf", "Jev-Style-2B-Decision-v3-Q4_K_M.gguf",
             "Jev-Style-2B-Decision-v3-Q8_0.gguf", "LICENSE", "NOTICE", "README.md", "build_jev_score.sh",
             "jev_score_v2.cpp", "jev_style_decision_gguf.py", "manifest.json", "readout_config.json",
             "release_config.json", "requirements.txt", "tokenizer/tokenizer.json"],
    "mlx": ["8bit/model.safetensors", "8bit/config.json", "8bit/macjev_norms_fp32.safetensors", "LICENSE", "NOTICE",
            "README.md", "THIRD_PARTY_NOTICES.md", "bf16/model.safetensors", "bf16/config.json",
            "bf16/macjev_norms_fp32.safetensors", "config.json", "jev_style_decision_mlx.py", "manifest.json",
            "readout_config.json", "release_config.json", "requirements.txt"],
}


def test_2b_patterns_select_what_the_runtimes_need():
    from huggingface_hub.utils import filter_repo_objects
    builds = RELEASES["2b-v3"].builds

    def pick(backend, **kw):
        return set(filter_repo_objects(HUB_2B[backend], allow_patterns=models.patterns_for(builds[backend], **kw)))
    torch = pick("torch")
    assert {"jev_style_decision.py", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            "model.safetensors.index.json", "config.json", "manifest.json"} <= torch
    gguf = pick("gguf", quant="q4_k_m")
    assert {"jev_score_v2.cpp", "build_jev_score.sh", "jev_style_decision_gguf.py", "tokenizer/tokenizer.json"} <= gguf
    assert [f for f in gguf if f.endswith(".gguf")] == ["Jev-Style-2B-Decision-v3-Q4_K_M.gguf"]
    mlx = pick("mlx", precision="8bit")
    assert {"config.json", "THIRD_PARTY_NOTICES.md", "8bit/macjev_norms_fp32.safetensors"} <= mlx
    assert not any(f.startswith("bf16/") for f in mlx)


def test_2b_mlx_needs_the_pinned_mlx_lm(monkeypatch):
    b = RELEASES["2b-v3"].builds["mlx"]
    monkeypatch.setattr(models, "has_module", lambda name: True)
    monkeypatch.setattr(models, "_installed_version", lambda dist: "0.31.2")
    assert "mlx-lm 0.31.3" in models.build_problem(b)
    assert models.build_problem(RELEASES["0.8b-v3"].builds["mlx"]) is None
    with pytest.raises(models.MissingBackendError, match="0.31.3"):
        models.require_backend(b)
    monkeypatch.setattr(models, "is_apple_silicon", lambda: True)
    assert models.resolve_backend("auto", "2b") == "torch"          # auto skips an MLX build that cannot run
    assert models.resolve_backend("auto", "0.8b") == "mlx"
    monkeypatch.setattr(models, "_installed_version", lambda dist: "0.31.3")
    assert models.resolve_backend("auto", "2b") == "mlx"


class _GraphRuntime:                    # a runtime that has the CUDA-graph option (the torch builds since 0.4.0)
    def __init__(self, folder, device=None, dtype="float32", verify=False, cuda_graphs=False):
        self.kw = {"device": device, "dtype": dtype, "verify": verify, "cuda_graphs": cuda_graphs}


class _GraphModule:
    JevStyleDecision = _GraphRuntime


def test_cuda_graphs_follow_the_device_unless_told(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "import_runtime", lambda folder, module: _GraphModule)
    monkeypatch.setattr(models, "require_backend", lambda build: None)
    monkeypatch.delenv("JEV_STYLE_CUDA_GRAPHS", raising=False)

    def graphs(**kw):
        return models.load_release("torch", release="2b", model_dir=tmp_path, **kw)[0].kw["cuda_graphs"]

    assert graphs(device="cpu") is False
    assert graphs(device="cuda") is True and graphs(device="cuda:1") is True
    assert graphs(device="cuda", cuda_graphs=False) is False
    monkeypatch.setenv("JEV_STYLE_CUDA_GRAPHS", "0")
    assert graphs(device="cuda") is False
    assert graphs(device="cuda", cuda_graphs=True) is True          # an explicit choice beats the environment


class _AutoGraphRuntime(_GraphRuntime):  # cuda_graphs=None: the runtime picks (2B v4: bfloat16 on CUDA)
    def __init__(self, folder, device=None, dtype="float32", verify=False, cuda_graphs=None):
        super().__init__(folder, device=device, dtype=dtype, verify=verify, cuda_graphs=cuda_graphs)


class _AutoGraphModule:
    JevStyleDecision = _AutoGraphRuntime


def test_cuda_graphs_left_to_a_runtime_that_picks_them(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "import_runtime", lambda folder, module: _AutoGraphModule)
    monkeypatch.setattr(models, "require_backend", lambda build: None)
    monkeypatch.delenv("JEV_STYLE_CUDA_GRAPHS", raising=False)

    def graphs(**kw):
        return models.load_release("torch", release="2b-v4", model_dir=tmp_path, **kw)[0].kw["cuda_graphs"]

    assert graphs(device="cuda") is None and graphs(device="cuda", dtype="bfloat16") is None    # auto: not passed
    assert graphs(device="cuda", cuda_graphs=True) is True and graphs(device="cuda", cuda_graphs=False) is False
    monkeypatch.setenv("JEV_STYLE_CUDA_GRAPHS", "auto")
    assert graphs(device="cuda") is None
    monkeypatch.setenv("JEV_STYLE_CUDA_GRAPHS", "1")
    assert graphs(device="cuda") is True
    monkeypatch.setenv("JEV_STYLE_CUDA_GRAPHS", "off")
    assert graphs(device="cuda") is False


def test_cuda_graphs_on_an_old_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "import_runtime", lambda folder, module: _Module)
    monkeypatch.setattr(models, "require_backend", lambda build: None)
    runtime, _, _ = models.load_release("torch", release="2b", model_dir=tmp_path, device="cuda")
    assert "cuda_graphs" not in runtime.kw                           # auto: an old runtime loads as before
    with pytest.raises(ValueError, match="no CUDA-graph path"):
        models.load_release("torch", release="2b", model_dir=tmp_path, device="cuda", cuda_graphs=True)


# the files of the 2B v4 repos at their pinned revisions (Hub tree, 2026-10-28), without validation/ and assets/
HUB_2B_V4 = {
    "torch": ["LICENSE", "NOTICE", "README.md", "SHA256SUMS.json", "chat_template.jinja", "config.json",
              "generation_config.json", "jev_style_decision.py", "macjev_readout.json", "manifest.json",
              "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
              "model-00003-of-00003.safetensors", "model.safetensors.index.json", "preprocessor_config.json",
              "readout_config.json", "release_config.json", "requirements.txt", "tokenizer.json",
              "tokenizer_config.json", "video_preprocessor_config.json", "src/macjev/__init__.py",
              "src/macjev/backbone.py", "src/macjev/eval/__init__.py", "src/macjev/runtime/v4_engine.py"],
    "gguf": ["Jev-Style-2B-Decision-v4-F16.gguf", "Jev-Style-2B-Decision-v4-Q4_K_M.gguf",
             "Jev-Style-2B-Decision-v4-Q8_0.gguf", "LICENSE", "NOTICE", "README.md", "build_jev_score.sh",
             "jev_score_v2.cpp", "jev_style_decision_gguf.py", "manifest.json", "readout_config.json",
             "release_config.json", "requirements.txt", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"],
    "mlx": ["LICENSE", "NOTICE", "README.md", "THIRD_PARTY_NOTICES.md", "config.json", "jev_style_decision_mlx.py",
            "manifest.json", "readout_config.json", "release_config.json", "requirements.txt"]
           + [f"{p}/{f}" for p in ("bf16", "8bit") for f in (
               "chat_template.jinja", "config.json", "generation_config.json", "macjev_norms_fp32.safetensors",
               "model.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")],
}


def test_2b_v4_patterns_select_what_the_runtimes_need():
    from huggingface_hub.utils import filter_repo_objects
    r = RELEASES["2b-v4"]
    assert r.published and ALIASES["2b"] == "2b-v4"
    assert {b: r.builds[b].repo for b in r.builds} == {
        "torch": "chaoliangUNSW/Jev-Style-2B-Decision-v4", "mlx": "chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX",
        "gguf": "chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF"}

    def pick(backend, **kw):
        return set(filter_repo_objects(HUB_2B_V4[backend], allow_patterns=models.patterns_for(r.builds[backend], **kw)))
    torch = pick("torch")
    assert set(HUB_2B_V4["torch"]) - torch == {"README.md"}           # the reference scorer under src/ included
    gguf = pick("gguf", quant="q4_k_m")
    assert {"jev_score_v2.cpp", "build_jev_score.sh", "jev_style_decision_gguf.py", "tokenizer/tokenizer.json"} <= gguf
    assert [f for f in gguf if f.endswith(".gguf")] == ["Jev-Style-2B-Decision-v4-Q4_K_M.gguf"]
    assert r.builds["gguf"].extra["scorer_bin"] == "jev-score-v2"
    mlx = pick("mlx", precision="bf16")
    assert {"config.json", "THIRD_PARTY_NOTICES.md", "bf16/model.safetensors", "bf16/macjev_norms_fp32.safetensors"} <= mlx
    assert not any(f.startswith("8bit/") for f in mlx)
    assert r.builds["mlx"].extra["mlx_lm"] == "0.31.3" and not r.builds["mlx"].extra["prefix_sharing"]


def test_download_skips_a_partly_cached_snapshot(monkeypatch, tmp_path):
    """A cached snapshot of the pinned revision that lacks a selected file (another quant, or one file fetched with
    hf_hub_download) must not be used as is: download() goes to the network (0.6.0 fix)."""
    import json as _json
    import huggingface_hub
    build = RELEASES["2b-v4"].builds["gguf"]
    names = ["manifest.json", "jev_style_decision_gguf.py", "jev_score_v2.cpp", "build_jev_score.sh",
             "tokenizer/tokenizer.json", "Jev-Style-2B-Decision-v4-Q8_0.gguf", "Jev-Style-2B-Decision-v4-F16.gguf"]
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "manifest.json").write_text(_json.dumps({"files": {n: {} for n in names if n != "manifest.json"}}))
    for n in names[1:6]:                                              # Q8_0 cached, F16 not
        (tmp_path / n).write_text("x")
    calls = []

    def fake_snapshot(repo, local_files_only=False, **kw):
        calls.append(local_files_only)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    assert models.download(build=build, quant="Q8_0") == tmp_path and calls == [True]
    calls.clear()
    models.download(build=build, quant="F16")
    assert calls == [True, False]                                    # incomplete -> network download
    assert not models.cached_complete(tmp_path / "missing", ["*"])


# ----------------------------------------------------------------------------- 0.6.4: custom repos, checks before downloads
def test_custom_repo_patterns_are_not_the_0_8b_file_names():
    from jev_style.models import patterns_for
    gguf = build_for_repo("someone/My-2B-Fork-GGUF", trust_remote_code=True)
    pats = patterns_for(gguf, quant="Q4_K_M")
    assert pats[-1] == "*-Q4_K_M.gguf" and not any("0.8B" in p for p in pats)
    assert {"*.py", "*.cpp", "*.sh", "tokenizer/*"} <= set(pats) and gguf.extra["scorer_bin"] is None
    torch = build_for_repo("someone/My-2B-Fork", trust_remote_code=True)
    assert {"*.safetensors", "*.json", "*.py", "src/**"} <= set(patterns_for(torch))
    assert "model.safetensors" not in torch.patterns
    mlx = build_for_repo("someone/My-2B-Fork-MLX", trust_remote_code=True)
    assert patterns_for(mlx, precision="8bit")[-1] == "8bit/*" and "*.py" in mlx.patterns
    for b in (gguf, torch, mlx):
        assert (b.release, b.revision) == ("custom", None)


def test_download_refuses_a_backend_that_cannot_run(monkeypatch):
    monkeypatch.setattr(models, "has_module", lambda name: False)
    monkeypatch.setattr(models, "is_apple_silicon", lambda: False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("downloaded first"))
    with pytest.raises(models.MissingBackendError, match=r"jev-style\[torch\]"):
        models.download("torch")
    with pytest.raises(models.MissingBackendError, match="Apple silicon"):
        models.require_backend(RELEASES["0.8b-v3"].builds["mlx"])
    monkeypatch.setattr(models, "is_apple_silicon", lambda: True)
    with pytest.raises(models.MissingBackendError, match=r"jev-style\[mlx\]"):
        models.require_backend(RELEASES["2b-v4"].builds["mlx"])


def test_gguf_refuses_before_downloading_when_no_scorer_can_be_found(monkeypatch, tmp_path):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, local_files_only=False, **k: (_ for _ in ()).throw(FileNotFoundError("no cache"))
                        if local_files_only else pytest.fail("downloaded before the scorer check"))
    monkeypatch.delenv("JEV_SCORE_BIN", raising=False)
    monkeypatch.delenv("JEV_SCORE_V2_BIN", raising=False)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(models.MissingScorerError) as ei:
        models.load_release("gguf")
    msg = str(ei.value)
    assert "jev-score scorer" in msg and "jev-style download --backend gguf --release 0.8b-v3" in msg
    with pytest.raises(models.MissingScorerError, match="jev-score-v2"):
        models.load_release("gguf", release="2b")
    # any of the usual places satisfies the check, so the download goes ahead
    monkeypatch.setattr(models, "download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("download reached")))
    with pytest.raises(RuntimeError, match="download reached"):
        models.load_release("gguf", scorer="/opt/jev-score")
    monkeypatch.setenv("JEV_SCORE_V2_BIN", "/opt/jev-score-v2")
    with pytest.raises(RuntimeError, match="download reached"):
        models.load_release("gguf", release="2b-v4")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/" + name)
    with pytest.raises(RuntimeError, match="download reached"):
        models.load_release("gguf")
    # a scorer built inside a cached snapshot (build/jev-score) counts too
    folder = tmp_path / "snap"
    (folder / "build").mkdir(parents=True)
    (folder / "build" / "jev-score").write_bytes(b"\x7fELF")
    assert models.find_scorer(RELEASES["0.8b-v3"].builds["gguf"], folder=folder) == str(folder / "build" / "jev-score")


def test_download_failures_name_the_repo_and_what_to_do(monkeypatch):
    import huggingface_hub

    def fail(repo_id, *a, local_files_only=False, **k):
        raise (FileNotFoundError if local_files_only else OSError)("Cannot reach https://huggingface.co")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(models.DownloadError) as ei:
        models.download(build=RELEASES["0.8b-v3"].builds["gguf"])
    assert str(ei.value).startswith("could not download chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF (revision b8356a83")
    assert "OSError: Cannot reach" in str(ei.value) and "--model-dir" in str(ei.value)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(models.DownloadError, match="HF_HUB_OFFLINE is set"):
        models.download(build=RELEASES["0.8b-v3"].builds["gguf"])


def _manifest_folder(tmp_path, bad=False):
    import hashlib
    import json
    folder = tmp_path / "model"
    folder.mkdir(parents=True)
    (folder / "jev_style_decision_gguf.py").write_text("# runtime\n")
    (folder / "Jev-Style-0.8B-Decision-v3-Q8_0.gguf").write_bytes(b"\x00weights" * 50)
    files = {}
    for p in folder.iterdir():
        files[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    if bad:                                              # same size, other bytes: only the hash can tell
        (folder / "Jev-Style-0.8B-Decision-v3-Q8_0.gguf").write_bytes(b"\x00tampere" * 50)
    (folder / "manifest.json").write_text(json.dumps({"format": "jev-style-manifest-v1", "files": files}))
    return folder


def test_verify_folder_checks_sizes_and_hashes(tmp_path, monkeypatch):
    ok = _manifest_folder(tmp_path / "ok")
    assert set(models.verify_folder(ok)) == {"jev_style_decision_gguf.py", "Jev-Style-0.8B-Decision-v3-Q8_0.gguf"}
    assert models.verify_folder(ok, ["*.gguf"]) == ["Jev-Style-0.8B-Decision-v3-Q8_0.gguf"]
    with pytest.raises(models.VerifyError, match="does not match"):
        models.verify_folder(_manifest_folder(tmp_path / "bad", bad=True))
    (ok / "jev_style_decision_gguf.py").write_text("# runtime, longer\n")
    with pytest.raises(models.VerifyError, match="the manifest says"):
        models.verify_folder(ok)
    (ok / "jev_style_decision_gguf.py").unlink()
    with pytest.raises(models.VerifyError, match="is missing"):
        models.verify_folder(ok)
    with pytest.raises(models.VerifyError, match="no manifest.json"):
        models.verify_folder(tmp_path)
    # download(verify=True) runs it on what the Hub returned
    import huggingface_hub
    good = _manifest_folder(tmp_path / "hub")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(good))
    build = RELEASES["0.8b-v3"].builds["gguf"]
    assert models.download(build=build, quant="Q8_0", verify=True) == good
    tampered = _manifest_folder(tmp_path / "hub2", bad=True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tampered))
    with pytest.raises(models.VerifyError):
        models.download(build=build, quant="Q8_0", verify=True)


def test_env_release_problem(monkeypatch):
    monkeypatch.delenv("JEV_STYLE_RELEASE", raising=False)
    assert models.env_release_problem() is None and models.env_release() == DEFAULT_RELEASE
    monkeypatch.setenv("JEV_STYLE_RELEASE", "2B")
    assert models.env_release_problem() is None and models.env_release() == "2b"
    monkeypatch.setenv("JEV_STYLE_RELEASE", "9b")
    assert "JEV_STYLE_RELEASE='9b'" in models.env_release_problem()
    assert models.env_release() == DEFAULT_RELEASE
