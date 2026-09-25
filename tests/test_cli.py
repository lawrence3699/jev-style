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
