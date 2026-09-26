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
