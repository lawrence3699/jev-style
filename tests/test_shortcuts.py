import pytest

import jev_style
from jev_style import JevStyle, noul
from jev_style.models import BUILDS, build_for_repo


def test_build_for_known_repos_keeps_pinned_revision():
    for backend, b in BUILDS.items():
        assert build_for_repo(b.repo) is b
        assert build_for_repo(b.repo.lower()).backend == backend


def test_unknown_repo_needs_trust_remote_code():
    with pytest.raises(ValueError, match="trust_remote_code"):
        build_for_repo("someone/Other-Decision-GGUF")
    b = build_for_repo("someone/Other-Decision-GGUF", trust_remote_code=True)
    assert (b.backend, b.repo, b.revision) == ("gguf", "someone/Other-Decision-GGUF", None)
    assert build_for_repo("someone/x-MLX", trust_remote_code=True).backend == "mlx"
    assert build_for_repo("someone/x", trust_remote_code=True).backend == "torch"


def test_from_pretrained_passes_repo_to_loader(monkeypatch):
    from jev_style.adapter import build_adapter
    seen = {}

    def fake_build(backend, fake=False, **kw):
        seen.update(kw)
        return build_adapter(fake=True)
    monkeypatch.setattr("jev_style.adapter.build_adapter", fake_build)
    js = JevStyle.from_pretrained("chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF", quant="Q4_K_M")
    assert seen == {"repo": "chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF", "trust_remote_code": False,
                    "quant": "Q4_K_M"}
    assert js.decide("x", {"q": noul("y")})["answers"]["q"]["type"] == "noul"


def test_shortcuts_use_the_configured_client():
    try:
        js = jev_style.configure(fake=True)
        assert jev_style.client.default_client() is js
        out = jev_style.decide("I was charged twice.", {"b": noul("About billing?")})
        assert 0.0 <= out["answers"]["b"]["noul"] <= 1.0
        c = jev_style.classify("Where is my parcel?", {"billing": None, "shipping": "delivery, tracking"})
        assert c["label"] in ("billing", "shipping") and set(c["probabilities"]) == {"billing", "shipping"}
        assert c["label"] == max(c["probabilities"], key=c["probabilities"].get)
    finally:
        jev_style.client._default = None


def test_default_client_uses_env_url(monkeypatch):
    monkeypatch.setenv("JEV_STYLE_URL", "http://127.0.0.1:9")
    try:
        jev_style.client._default = None
        assert jev_style.client.default_client().base_url == "http://127.0.0.1:9"
    finally:
        jev_style.client._default = None
