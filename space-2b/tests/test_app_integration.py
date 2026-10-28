"""Offline Gradio 6.28 integration: real component graph, stubbed model boundary.

Run: JEV_DEMO_PREVIEW=1 venv/bin/python -m pytest tests/test_app_integration.py -q
No server is launched, and network access/model downloads are blocked.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT
TEXT = '{"message":"My duplicate payment is still pending.","paid":true}'
QUESTION = "Which team should help?"
OPTIONS = "billing: duplicate payments\ntechnical: software errors"
TASK = "Route a support ticket"


@pytest.fixture(scope="session")
def app():
    mp = pytest.MonkeyPatch()
    for key, value in {"JEV_DEMO_PREVIEW": "1", "GRADIO_ANALYTICS_ENABLED": "False",
                       "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}.items():
        mp.setenv(key, value)
    attempted = []

    def no_network(*args, **kwargs):
        attempted.append("network")
        raise AssertionError("Network access is forbidden in these tests")

    mp.setattr(socket.socket, "connect", no_network)
    mp.setattr(socket.socket, "connect_ex", no_network)
    mp.setattr(socket, "create_connection", no_network)
    mp.setattr(socket, "getaddrinfo", no_network)
    import huggingface_hub
    download = Mock(side_effect=AssertionError("Model download is forbidden in preview tests"))
    mp.setattr(huggingface_hub, "snapshot_download", download)
    mp.syspath_prepend(str(APP))
    spec = importlib.util.spec_from_file_location("jev_demo_integration_app", APP / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._test_download = download
    module._test_network_attempts = attempted
    yield module
    assert not attempted, "The preview/test path attempted a network request"
    mp.undo()


@pytest.fixture
def live(app, monkeypatch):
    """The only live-model boundary is replaced; all app logic is real."""
    monkeypatch.setattr(app, "PREVIEW_MODE", False)
    monkeypatch.setattr(app, "MODEL", object())
    monkeypatch.setattr(app, "DEVICE", "test-no-gpu")
    calls = {"count": [], "score": []}

    def count(state, qs, images):
        calls["count"].append(copy.deepcopy((state, qs, images)))
        return {"state": 40, "heads": 10 * len(qs), "questions": len(qs),
                "images": len(images), "image_tokens": 0, "overflow": 0}

    def score(state, qs, images):
        calls["score"].append(copy.deepcopy((state, qs, images)))
        results = []
        for q in qs:
            names = (["false", "true"] if q["t"] == "noul" else list(q["crit"])
                     if q["t"] == "choice" else [str(i) for i in range(len(q["crit"]))])
            probs = {n: (0.85 if i == 0 else 0.15 / (len(names)-1)) for i, n in enumerate(names)}
            results.append({"answer": names[0], "probabilities": probs, "state_tokens": 40, "head_tokens": 10})
        return results, 12.5

    monkeypatch.setattr(app, "count", count)
    monkeypatch.setattr(app, "score", score)
    return app, calls


def make_job(app, **changes):
    values = dict(text=TEXT, question=QUESTION, kind="Choice", options=OPTIONS, image=None, task=TASK)
    values.update(changes)
    return app.prepare_one(**values)


def exported_state(source):
    tree = ast.parse(source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "state" for target in node.targets))
    return ast.literal_eval(assignment.value)


def assert_error_result(result, *, code=True, multi=False, title=None):
    assert len(result) == 9
    assert result[0] == ("" if multi else {})
    assert result[1] == "", "Historical timing must not appear in live metadata"
    if code:
        assert exported_state(result[2]) == json.loads(TEXT)
    assert 'data-state="error"' in result[3]
    if title:
        assert title in result[3]
    assert result[4]["visible"] is True
    assert "Input submitted for this run" in result[5]
    assert "saved-" in result[6], "Saved examples belong only in the separate HTML region"
    assert result[7]["interactive"] is True
    assert result[8]["interactive"] is False


def test_preview_import_is_offline_and_never_builds_gpu_example_cache(app):
    assert app.gr.__version__ == "6.28.0"
    assert app.PREVIEW_MODE and app.MODEL is None and app.rt is None
    app._test_download.assert_not_called()
    assert "torch" not in sys.modules
    assert not app._test_network_attempts
    # Gradio Examples may load inputs, but must not invoke/copy inference results.
    example_functions = [fn for fn in app.demo.fns.values() if getattr(fn.fn, "__name__", "") == "load_example"]
    assert len(example_functions) == 2
    assert sorted(len(fn.outputs) for fn in example_functions) == [3, 5]
    assert not any("cache" in getattr(fn.fn, "__name__", "") for fn in app.demo.fns.values())


def test_prepare_is_cpu_only_clears_old_results_and_preserves_current_export(live):
    app, calls = live
    prepared = make_job(app)
    assert len(prepared) == 10
    job, *result = prepared
    assert job["text"] == TEXT and job["question"] == QUESTION and job["options"] == OPTIONS
    assert result[0] == {} and result[1] == ""
    assert exported_state(result[2]) == json.loads(TEXT)
    assert 'data-state="queued"' in result[3]
    assert result[7]["interactive"] is False and result[8]["interactive"] is True
    assert calls == {"count": [], "score": []}


def test_run_success_is_generator_and_uses_frozen_submitted_input(live):
    app, calls = live
    job = make_job(app)[0]
    before = copy.deepcopy(job)
    generator = app.run_job(job)
    assert inspect.isgenerator(generator)
    results = list(generator)
    assert len(results) == 2
    assert 'data-state="running"' in results[0][3]
    assert results[0][0] == {}
    final = results[-1]
    assert final[0] == {"billing": 0.85, "technical": 0.15}
    assert 'data-state="live"' in final[3] and "Live result for your submitted input" in final[3]
    assert "12 ms" in final[1] and "recorded" not in final[1].lower()
    assert exported_state(final[2]) == json.loads(TEXT)
    assert final[6] == app.gr.skip(), "Live success must not relabel a saved result"
    assert job == before
    assert calls["score"][0][0] == json.loads(TEXT)
    assert calls["score"][0][1][0]["ins"] == QUESTION


@pytest.mark.parametrize("message,title", [
    ("You have exceeded your GPU quota (20s requested vs. 0s left)", "GPU quota unavailable"),
    ("The queue is full", "The demo is busy"),
    ("The request timed out", "The request timed out"),
    ("CUDA unexpected failure", "The run could not finish"),
])
def test_live_failure_preserves_input_and_code_never_substitutes_saved_prediction(live, monkeypatch, message, title):
    app, _ = live
    secret = "hf_private_token /tmp/private-model.py Traceback private@example.org"
    monkeypatch.setattr(app, "score", Mock(side_effect=RuntimeError(message + " " + secret)))
    job = make_job(app)[0]
    before = copy.deepcopy(job)
    final = list(app.run_job(job))[-1]
    assert_error_result(final, title=title)
    assert job == before
    rendered = json.dumps(final, default=str)
    for hidden in ("hf_private_token", "/tmp/private-model.py", "Traceback", "private@example.org"):
        assert hidden not in rendered


def test_validation_failure_never_requests_gpu_and_keeps_inputs(live, monkeypatch):
    app, calls = live
    monkeypatch.setattr(app, "count", Mock(side_effect=ValueError("secret internal details")))
    job = make_job(app)[0]
    final = list(app.run_job(job))[-1]
    assert_error_result(final, title="Check the request")
    assert not calls["score"]
    assert job["text"] == TEXT
    assert "secret internal" not in json.dumps(final, default=str)


def test_incomplete_input_is_rejected_without_count_or_score(live):
    app, calls = live
    prepared = make_job(app, options="only one option")
    assert prepared[0]["options"] == "only one option"
    assert 'data-state="error"' in prepared[4]
    final = list(app.run_job(prepared[0]))[-1]
    assert_error_result(final, code=False, title="Check the request")
    assert calls == {"count": [], "score": []}


def test_preview_rejects_live_without_count_and_keeps_current_sdk_export(app, monkeypatch):
    monkeypatch.setattr(app, "count", Mock(side_effect=AssertionError("No preview inference")))
    job = make_job(app)[0]
    results = list(app.run_job(job))
    assert len(results) == 1
    assert_error_result(results[0], title="The run could not finish")
    app.count.assert_not_called()


def test_multiple_questions_success_and_failure_use_separate_saved_region(live, monkeypatch):
    app, calls = live
    qs = [{"t": "noul", "ins": "Refund?", "crit": None},
          {"t": "score", "ins": "Urgency?", "crit": ["low", "high"]}]
    prepared = app.prepare_many(TEXT, json.dumps(qs), None)
    job = prepared[0]
    final = list(app.run_job(job))[-1]
    assert "| Question | Top option | Probability |" in final[0]
    assert "Refund?" in final[0] and "Urgency?" in final[0]
    assert 'data-state="live"' in final[3]
    assert exported_state(final[2]) == json.loads(TEXT)
    monkeypatch.setattr(app, "score", Mock(side_effect=TimeoutError("private timeout detail")))
    error = list(app.run_job(job))[-1]
    assert_error_result(error, multi=True, title="The request timed out")
    assert 'id="saved-many-' in error[6] and 'id="saved-' not in error[6].replace('id="saved-many-', '')


def test_submitted_input_html_is_escaped(live):
    app, _ = live
    raw = "<script>alert('x')</script> & keep this text"
    prepared = make_job(app, text=raw, question="<img src=x onerror=alert(1)>")
    assert "<script>" not in prepared[6]
    assert "<img " not in prepared[6]
    assert "&lt;script&gt;" in prepared[6]
    assert exported_state(prepared[3]) == raw


def test_task_switch_clears_live_outputs_but_does_not_run_model(live):
    app, calls = live
    changed = app.select_task("Check a receipt")
    assert len(changed) == 16
    assert changed[0] == app.RECEIPT and changed[2] == "Yes / No"
    assert changed[4] == app.RECEIPT_IMG
    assert changed[5] == {} and changed[6] == "" and changed[10] == ""
    assert "image(\"image.png\")" in changed[7]
    assert "receipt_approval" in changed[11]
    assert changed[12]["interactive"] is True and changed[13]["interactive"] is False
    assert changed[15]["open"] is True
    custom = app.select_task("Use my own input")
    assert all(value == app.gr.skip() for value in custom[:5]), "Custom mode must keep current input fields"
    assert calls == {"count": [], "score": []}


def test_legacy_api_still_returns_three_outputs(live):
    app, _ = live
    one = app.decide(TEXT, QUESTION, "Choice", OPTIONS, None)
    many = app.decide_several(TEXT, json.dumps([{"t": "noul", "ins": "Refund?", "crit": None}]), None)
    assert len(one) == len(many) == 3
    assert isinstance(one[0], dict) and isinstance(many[0], str)
    assert "model run" in one[1] and "model run" in many[1]
    ast.parse(one[2]); ast.parse(many[2])


@pytest.mark.parametrize("api", ["one", "many"])
@pytest.mark.parametrize("phase", ["validation", "inference"])
def test_legacy_api_errors_do_not_expose_raw_exception(live, monkeypatch, api, phase):
    app, _ = live
    hidden = "hf_do_not_expose /private/model-path.py"
    monkeypatch.setattr(app, "count" if phase == "validation" else "score",
                        Mock(side_effect=ValueError(hidden) if phase == "validation" else RuntimeError(hidden)))
    with pytest.raises(app.gr.Error) as error:
        if api == "one":
            app.decide(TEXT, QUESTION, "Choice", OPTIONS, None)
        else:
            app.decide_several(TEXT, json.dumps([{"t": "noul", "ins": "Refund?", "crit": None}]), None)
    assert hidden not in str(error.value)
    assert "Check the request" in str(error.value) if phase == "validation" else "The run could not finish" in str(error.value)


def test_queue_cancel_and_api_component_graph(app):
    assert app.demo._queue.max_size == 30
    functions = app.demo.fns
    runs = {key: fn for key, fn in functions.items() if fn.fn is app.run_job}
    assert len(runs) == 2
    for key, fn in runs.items():
        assert fn.queue and fn.types_generator
        assert fn.concurrency_id == "live-inference" and fn.concurrency_limit == 1
        assert len(fn.outputs) == 9
        # The inference event cannot write back into text/question/options/image inputs.
        inputs = {app.text._id, app.question._id, app.options._id, app.image._id,
                  app.text2._id, app.questions._id, app.image2._id}
        assert not inputs & {component._id for component in fn.outputs}
        cancellations = [other for other in functions.values() if key in other.cancels]
        assert cancellations and all(not other.queue for other in cancellations)
    prepares = [fn for fn in functions.values() if fn.fn in (app.prepare_one, app.prepare_many)]
    assert len(prepares) == 2 and all(not fn.queue for fn in prepares)
    apis = [fn for fn in functions.values() if fn.fn in (app.decide, app.decide_several)]
    assert len(apis) == 2
    assert all(len(fn.outputs) == 3 and fn.concurrency_id == "live-inference" and fn.concurrency_limit == 1 for fn in apis)
    cancel_ids = {app.cancel._id, app.cancel2._id}
    cancel_dependencies = [d for d in app.demo.config["dependencies"]
                           if any(target[0] in cancel_ids for target in d["targets"])]
    assert cancel_dependencies and all(not d["queue"] for d in cancel_dependencies)
    for multiple in (False, True):
        stopped = app.stopped(multiple)
        assert stopped[0] == ("" if multiple else {}) and stopped[1] == ""
        assert 'data-state="stale"' in stopped[2]
        assert stopped[4]["interactive"] is True and stopped[5]["interactive"] is False


@pytest.mark.parametrize("failure", [False, True])
def test_real_gradio_preprocess_state_and_generator_postprocess(live, monkeypatch, failure):
    """Exercise actual Gradio State + Label/HTML/Code serialization, without HTTP."""
    import asyncio
    from gradio.state_holder import SessionState
    app, _ = live
    if failure:
        monkeypatch.setattr(app, "score", Mock(side_effect=RuntimeError("You have exceeded your GPU quota")))
    prepare_id = next(key for key, fn in app.demo.fns.items() if fn.fn is app.prepare_one)
    run_id = next(key for key, fn in app.demo.fns.items() if fn.fn is app.run_job and fn.inputs[0]._id == app.job._id)

    async def invoke():
        session = SessionState(app.demo)
        prepared = await app.demo.process_api(
            prepare_id, [TEXT, QUESTION, "Choice", OPTIONS, None, TASK], state=session,
            session_hash="offline-integration", event_id="prepare-offline", simple_format=True)
        submitted = copy.deepcopy(session[app.job._id])
        first = await app.demo.process_api(
            run_id, [None], state=session, session_hash="offline-integration", event_id="run-offline",
            simple_format=True)
        second = await app.demo.process_api(
            run_id, [], state=session, iterator=first["iterator"], session_hash="offline-integration",
            event_id="run-offline", simple_format=True)
        # Exhaust the generator so Gradio releases its in-process iterator state.
        await app.demo.process_api(
            run_id, [], state=session, iterator=second["iterator"], session_hash="offline-integration",
            event_id="run-offline", simple_format=True)
        return prepared, first, second, submitted, session[app.job._id]

    prepared, first, second, before, after = asyncio.run(invoke())
    assert len(prepared["data"]) == 10 and len(first["data"]) == len(second["data"]) == 9
    assert before == after and after["text"] == TEXT
    assert exported_state(second["data"][2]) == json.loads(TEXT)
    if failure:
        assert second["data"][0] == {}, "An error must serialize an empty live prediction"
        assert 'data-state="error"' in second["data"][3]
        assert "Saved example" in second["data"][6]
    else:
        assert second["data"][0]["label"] == "billing"
        assert second["data"][0]["confidences"][0] == {"label": "billing", "confidence": 0.85}
        assert 'data-state="live"' in second["data"][3]
