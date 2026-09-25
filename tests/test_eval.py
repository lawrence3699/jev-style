import math
from pathlib import Path

from jev_style.evaluate import automation, label_key, load_rows, main, metrics, recalibrate, report, run

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "support_tickets.jsonl"


def test_label_keys():
    assert label_key({"type": "noul"}, "yes") == "true"
    assert label_key({"type": "noul"}, False) == "false"
    assert label_key({"type": "choice", "criteria": {"a": None, "b": None}}, "b") == "b"
    assert label_key({"type": "score", "criteria": ["lo", {"label": "hi"}]}, "hi") == "1"
    assert label_key({"type": "score", "criteria": ["lo", "hi"]}, 0) == "0"


def test_recalibrate_is_temperature():
    p = {"a": 0.7, "b": 0.2, "c": 0.1}
    assert recalibrate(p, 1.0) == pytest_approx(p)
    sharp = recalibrate(p, 0.5)
    assert sharp["a"] > p["a"] and math.isclose(sum(sharp.values()), 1.0)


def pytest_approx(d):
    import pytest
    return pytest.approx(d)


def test_metrics_perfect_and_automation():
    items = [{"probs": {"x": 0.9, "y": 0.1}, "label": "x"}] * 8 + [{"probs": {"x": 0.6, "y": 0.4}, "label": "y"}] * 2
    m = metrics(items)
    assert m["accuracy"] == 0.8 and m["n"] == 10
    a = automation(items)
    assert a["1%"]["coverage"] == 0.8 and a["1%"]["threshold"] == 0.9


def test_run_on_examples_fake():
    from jev_style import JevStyle
    rows = load_rows(EXAMPLES)
    res = run(rows, JevStyle(fake=True).decide)
    assert len(res) == 3 * len(rows)
    s = report(res)
    assert set(s["questions"]) == {"team", "escalate", "mood"} and "temperature" in s["questions"]["team"]


def test_cli(tmp_path, capsys):
    out = tmp_path / "s.json"
    assert main([str(EXAMPLES), "--fake", "--json", str(out)]) == 0
    assert "OVERALL" in capsys.readouterr().out and out.exists()


def test_numeric_level_names_are_names_not_indices():
    q = {"type": "score", "criteria": ["1", "2", "3", "4", "5"]}
    assert label_key(q, "3") == "2" and label_key(q, "5") == "4" and label_key(q, 5) == "4"


def test_unreadable_label_is_skipped():
    from jev_style import JevStyle
    rows = [{"state": "x", "questions": {"t": {"type": "choice", "instructions": "a",
                                               "criteria": {"a": None, "b": None}, "label": "zzz"}}}]
    assert run(rows, JevStyle(fake=True).decide) == []
