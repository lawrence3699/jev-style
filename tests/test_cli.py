import json

from jev_style.cli import main


def test_decide_fake(capsys):
    assert main(["decide", "I was charged twice", "--fake", "--noul", "About billing?",
                 "--choice", "Team?::billing,tech", "--score", "Urgency?::low,mid,high"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["answers"]) == {"noul_1", "choice_1", "score_1"}


def test_decide_needs_a_question(capsys):
    assert main(["decide", "x", "--fake"]) == 2


def test_backend_resolution():
    from jev_style.models import resolve_backend
    assert resolve_backend("torch") == "torch"
    assert resolve_backend("auto") in ("torch", "mlx")


def test_top_level_help_lists_every_command(capsys):
    import pytest
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "{serve,download,releases,decide,eval,guard,guard-replay,mcp}" in out
    for cmd in ("eval", "guard-replay", "mcp"):
        assert f"\n    {cmd} " in out


def test_download_checks_the_backend_before_fetching(monkeypatch, capsys):
    from jev_style import models
    monkeypatch.setattr(models, "has_module", lambda name: False)
    monkeypatch.setattr(models, "is_apple_silicon", lambda: False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("downloaded before the backend check")))
    assert main(["download", "--backend", "torch"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: the torch backend") and "jev-style[torch]" in err
    assert main(["download", "--backend", "mlx"]) == 2
    assert "Apple silicon" in capsys.readouterr().err


def test_download_failures_are_one_line(monkeypatch, capsys):
    import huggingface_hub

    def fail(repo_id, *a, local_files_only=False, **k):
        raise (FileNotFoundError if local_files_only else OSError)("Cannot reach https://huggingface.co")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert main(["download", "--backend", "gguf", "--no-backend-check"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: could not download chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF")
    assert "Cannot reach" in err and "network" in err and "Traceback" not in err
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert main(["download", "--backend", "gguf"]) == 2
    assert "HF_HUB_OFFLINE is set" in capsys.readouterr().err


def test_missing_model_dir_is_one_line(tmp_path, capsys):
    assert main(["decide", "x", "--noul", "y", "--backend", "gguf", "--model-dir", str(tmp_path / "nope")]) == 2
    assert capsys.readouterr().err.startswith("error: file not found: model_dir ")


def test_env_release_is_checked_first(monkeypatch, capsys):
    monkeypatch.setenv("JEV_STYLE_RELEASE", "9b")
    assert main(["releases"]) == 2
    assert "JEV_STYLE_RELEASE='9b'" in capsys.readouterr().err
    monkeypatch.setenv("JEV_STYLE_RELEASE", "2B")
    from jev_style.models import env_release
    assert env_release() == "2b"
    assert main(["releases"]) == 0


def test_guard_hook_ignores_a_bad_env_release(monkeypatch, capsys):
    monkeypatch.setenv("JEV_STYLE_RELEASE", "9b")
    monkeypatch.setenv("JEV_STYLE_URL", "http://127.0.0.1:9")      # nothing listens there: the verdict is "ask"
    assert main(["guard", "--check", "ls"]) == 0                   # a hook: never a non-zero exit for the env
    out = capsys.readouterr()
    assert "ASK" in out.out and "JEV_STYLE_RELEASE" not in out.err
