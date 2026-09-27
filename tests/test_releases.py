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
    assert get_release().key == "2b-v3"
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
    runtime, rt, build = models.load_release("torch", release="2b", model_dir=tmp_path, device="cpu")
    assert (build.release, build.backend, runtime.folder) == ("2b-v3", "torch", tmp_path)
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
