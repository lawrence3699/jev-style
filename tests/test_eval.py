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


def test_compare_same_questions_and_failures():
    from jev_style import JevStyle
    from jev_style.evaluate import compare, render_compare
    rows = load_rows(EXAMPLES)
    fake = JevStyle(fake=True).decide

    def flaky(state, qs):                      # fails on the first row, otherwise the same answers
        if state == rows[0]["state"]:
            raise RuntimeError("too long")
        return fake(state, qs)
    c = compare(rows, {"a": fake, "b": flaky})
    assert c["engines"]["b"]["failed_rows"] == 1
    assert c["common_questions"] == 3 * (len(rows) - 1) == c["engines"]["a"]["summary"]["n"]
    d = c["vs_reference"]["b"]
    assert d["accuracy"] == 0 and d["accuracy_ci95"] == [0, 0] and d["brier"] == 0
    assert "difference to a" in render_compare(c)


def test_paired_diff_detects_a_better_engine():
    from jev_style.evaluate import paired_diff
    ref = [{"row": i, "qid": "q", "probs": {"x": 0.6, "y": 0.4}, "label": "x" if i % 2 else "y"} for i in range(40)]
    better = [{**r, "probs": {r["label"]: 0.9, ("y" if r["label"] == "x" else "x"): 0.1}} for r in ref]
    d = paired_diff(ref, better)
    assert d["accuracy"] == 0.5 and d["accuracy_ci95"][0] > 0 and d["brier_ci95"][1] < 0


def test_cli_compare(tmp_path, capsys):
    out = tmp_path / "c.json"
    assert main([str(EXAMPLES), "--server", "fake", "--server", "again=fake", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "difference to fake" in text and "again" in text and out.exists()


def test_parse_server():
    import pytest
    from jev_style.evaluate import parse_server
    assert parse_server("local") == ("local", "local")
    assert parse_server("local:mlx") == ("local:mlx", "local:mlx")
    assert parse_server("x=http://h:1/v") == ("x", "http://h:1/v")
    with pytest.raises(ValueError):
        parse_server("http://h:1")
