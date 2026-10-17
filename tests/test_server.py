"""The HTTP server (anyintent.server, extra 'server'): FastAPI's TestClient (in process, no socket) against stand-in
classifiers, the tiny random model of conftest, and tiny models with the settings of R2Intent v1 with and without a
calibration file (check and scores: equal to the library, the 409 without the calibration). Skipped when fastapi or
httpx is not installed."""
import json
import logging
import re
import warnings

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient   # noqa: E402

from anyintent import AnyIntent, __version__   # noqa: E402
from anyintent import server   # noqa: E402
from anyintent.service import Limits, Service   # noqa: E402
from conftest import LABELS, MESSAGES   # noqa: E402
from serving_fakes import (INTENTS, SECRET, FakeBroken, FakeClassifier, FakeWithExtras,   # noqa: E402
                           assert_scores_answer, calibrated_models)

CHECK_KEYS = {"intent", "yes", "probability", "score", "threshold"}
SCORES_KEYS = {"label", "intents"}


@pytest.fixture(scope="module")
def cal_models(tmp_path_factory, tok):
    return calibrated_models(str(tmp_path_factory.mktemp("serving_cal")), tok)


def load(folder):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AnyIntent.from_pretrained(folder, device="cpu")


def client(clf=None, api_key=None, cors=(), allowed_hosts=None, max_concurrency=64, base_url="http://testserver",
           **kw):
    s = Service(clf or FakeClassifier(), model_id="fake/model", **kw)
    return TestClient(server.create_app(s, api_key=api_key, cors_origins=cors, allowed_hosts=allowed_hosts,
                                        max_concurrency=max_concurrency), base_url=base_url)


def err(r, status, code, needle=""):
    assert r.status_code == status, (r.status_code, r.text)
    body = r.json()
    assert set(body) == {"error"} and body["error"]["code"] == code, body
    assert needle in body["error"]["message"], body
    assert "Traceback" not in r.text and 'File "' not in r.text
    return body["error"]


def test_health_and_docs():
    c = client()
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["ready"] is True and h["model"] == "fake/model"
    assert h["revision"] == "fake-revision-0001" and h["config"]["k"] == 8 and "root" not in h["config"]
    assert h["features"] == {"classify": True, "check": False, "scores": False}
    assert h["supported"] == ["classify"] and h["calibration"] is None
    assert c.get("/openapi.json").status_code == 200 and c.get("/docs").status_code == 200
    paths = c.get("/openapi.json").json()["paths"]
    assert {"/health", "/classify", "/check", "/scores", "/lists", "/lists/{name}"} <= set(paths)


def test_classify_text_texts_and_list():
    c = client()
    r = c.post("/classify", json={"text": "I lost my bank card", "intents": INTENTS})
    assert r.status_code == 200 and r.json()["results"][0]["label"] == "lost card"
    assert set(r.json()["results"][0]) == {"label", "index", "score", "margin", "runner_up"}
    r = c.post("/classify", json={"texts": ["when will my refund arrive", "close account now"], "intents": INTENTS,
                                  "k": 2})
    assert [x["label"] for x in r.json()["results"]] == ["refund status", "close account"]
    assert c.put("/lists/support", json={"intents": INTENTS}).status_code == 201
    assert c.put("/lists/support", json={"intents": INTENTS}).status_code == 200
    r = c.post("/classify", json={"text": "refund please", "list": "support"})
    assert r.status_code == 200 and r.json()["results"][0]["label"] == "refund status"


def test_validation_errors_are_clear():
    c = client()
    err(c.post("/classify", json={"intents": INTENTS}), 422, "invalid_request", "exactly one of 'text'")
    err(c.post("/classify", json={"text": "x", "texts": ["y"], "intents": INTENTS}), 422, "invalid_request")
    err(c.post("/classify", json={"text": "x"}), 422, "invalid_request", "'list' over HTTP")
    err(c.post("/classify", json={"text": "x", "intents": ["a", "a"]}), 422, "invalid_request", "unique")
    err(c.post("/classify", json={"text": "x", "list": "nope"}), 404, "not_found", "no saved intent list")
    e = err(c.post("/classify", json={"text": 5, "intents": INTENTS}), 422, "invalid_request", "expected JSON body")
    assert e["details"] and e["details"][0]["loc"][:2] == ["body", "text"]
    err(c.post("/classify", json={"text": "x", "intents": [{"name": "a", "colour": "red"}]}), 422, "invalid_request")
    err(c.post("/classify", json={"text": "x", "intents": INTENTS, "unknown": 1}), 422, "invalid_request")
    err(c.post("/classify", json={"text": "x", "intents": INTENTS, "k": "3"}), 422, "invalid_request")
    err(c.post("/classify", content=b"{not json", headers={"content-type": "application/json"}), 422,
        "invalid_request")
    err(c.get("/nowhere"), 404, "not_found")
    err(c.delete("/classify"), 405, "method_not_allowed")


def test_limits():
    lim = Limits(max_batch=2, max_text_chars=30, max_intents=3, max_name_chars=20, max_description_chars=100)
    c = client(limits=lim)
    err(c.post("/classify", json={"texts": ["a", "b", "c"], "intents": INTENTS}), 413, "limit_exceeded", "--max-batch")
    err(c.post("/classify", json={"text": "x" * 31, "intents": INTENTS}), 413, "limit_exceeded", "--max-text-chars")
    err(c.post("/classify", json={"text": "x", "intents": ["a", "b", "c", "d"]}), 413, "limit_exceeded",
        "--max-intents")
    err(c.put("/lists/big", json={"intents": ["a", "b", "c", "d"]}), 413, "limit_exceeded")
    big = b'{"text": "' + b"x" * (lim.max_body_bytes + 10) + b'"}'
    err(c.post("/classify", content=big, headers={"content-type": "application/json"}), 413, "limit_exceeded",
        "request body")

    def chunks():                                        # no Content-Length: the streamed bytes are counted
        for _ in range(lim.max_body_bytes // 1000 + 2):
            yield b" " * 1000

    err(c.post("/classify", content=chunks(), headers={"content-type": "application/json"}), 413, "limit_exceeded")
    assert c.post("/classify", json={"texts": ["a", "b"], "intents": INTENTS}).status_code == 200


def test_api_key():
    c = client(api_key="s3cret-key")
    e = err(c.post("/classify", json={"text": "x", "intents": INTENTS}), 401, "unauthorized", "Bearer")
    r = c.post("/classify", json={"text": "x", "intents": INTENTS})
    assert r.headers.get("www-authenticate") == "Bearer"
    err(c.post("/classify", json={"text": "x", "intents": INTENTS}, headers={"Authorization": "Bearer wrong"}), 401,
        "unauthorized")
    err(c.post("/classify", json={"text": "x", "intents": INTENTS}, headers={"Authorization": "Basic s3cret-key"}),
        401, "unauthorized")
    ok = {"Authorization": "Bearer s3cret-key"}
    assert c.post("/classify", json={"text": "x", "intents": INTENTS}, headers=ok).status_code == 200
    for method, path in (("get", "/lists"), ("get", "/lists/a"), ("delete", "/lists/a")):
        assert getattr(c, method)(path).status_code == 401, path
    assert c.put("/lists/a", json={"intents": ["x"]}).status_code == 401
    assert c.get("/health").json() == {"status": "ok", "ready": True}          # liveness without the key
    assert c.get("/health", headers=ok).json()["model"] == "fake/model"
    assert c.get("/openapi.json").status_code == 200
    assert e["message"].startswith("missing or wrong API key")


def test_lists_endpoints_and_persistence(tmp_path):
    d = str(tmp_path / "lists")
    c = client(lists_dir=d)
    r = c.put("/lists/billing", json={"intents": INTENTS})
    assert r.status_code == 201 and r.json() == {"name": "billing", "n_intents": 3, "created": True,
                                                 "persistent": True}
    assert c.get("/lists").json() == {"lists": [{"name": "billing", "n_intents": 3}], "persistent": True}
    assert c.get("/lists/billing").json()["intents"][2] == {"name": "close account"}
    err(c.put("/lists/Bad", json={"intents": INTENTS}), 422, "invalid_request", "list name")
    err(c.put("/lists/con", json={"intents": INTENTS}), 422, "invalid_request")
    err(c.put("/lists/a.b", json={"intents": INTENTS}), 422, "invalid_request")
    err(c.put("/lists/" + "x" * 65, json={"intents": INTENTS}), 422, "invalid_request")
    err(c.put("/lists/ok", json={"intents": []}), 422, "invalid_request", "empty")
    err(c.get("/lists/missing"), 404, "not_found")
    c2 = client(lists_dir=d)                                    # a restarted server reads the folder
    assert c2.get("/lists/billing").json() == c.get("/lists/billing").json()
    assert c2.delete("/lists/billing").json() == {"name": "billing", "deleted": True}
    err(c2.get("/lists/billing"), 404, "not_found")
    assert client(lists_dir=d).get("/lists").json()["lists"] == []


def test_check_and_scores_501_without_the_library_methods():
    c = client(FakeClassifier())
    err(c.post("/check", json={"text": "x", "intent": "a"}), 501, "not_available", "AnyIntent.check")
    err(c.post("/scores", json={"text": "x", "intents": ["a"]}), 501, "not_available", "AnyIntent.scores")


def test_check_and_scores_with_the_library_methods():
    c = client(FakeWithExtras())
    h = c.get("/health").json()
    assert h["features"] == {"classify": True, "check": True, "scores": True}
    assert h["supported"] == ["classify", "check", "scores"]
    assert h["calibration"] == {"file": True, "functions": ["check", "scores"]}
    r = c.post("/check", json={"texts": ["I lost my bank card", "hello"], "intent": INTENTS[0]})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert [set(x) for x in res] == [CHECK_KEYS] * 2
    assert [x["yes"] for x in res] == [True, False] and res[0]["intent"] == "lost card"
    assert c.post("/check", json={"text": "card", "intent": "lost card", "threshold": 0.99}).json()["results"][0][
        "threshold"] == 0.99
    pair = c.post("/check", json={"text": "card", "intent": [INTENTS[0]["name"], INTENTS[0]["description"]]})
    assert pair.status_code == 200 and pair.json() == c.post("/check", json={"text": "card",
                                                                             "intent": INTENTS[0]}).json()
    r = c.post("/scores", json={"text": "I lost my bank card", "intents": INTENTS})
    res = r.json()["results"][0]
    assert r.status_code == 200 and set(res) == SCORES_KEYS and res["label"] == "lost card"
    assert sorted(x["intent"] for x in res["intents"]) == ["close account", "lost card", "refund status"]
    assert_scores_answer(res, FakeWithExtras().set_labels(INTENTS).scores("I lost my bank card"))
    c.put("/lists/s", json={"intents": INTENTS})
    assert c.post("/scores", json={"text": "I lost my bank card", "list": "s"}).json()["results"][0] == res


def test_check_and_scores_validation():
    lim = Limits(max_batch=2, max_text_chars=30, max_intents=3, max_name_chars=20, max_description_chars=100)
    c = client(FakeWithExtras(), limits=lim)
    for body, needle in (({"intent": "a"}, "exactly one of 'text'"),
                         ({"text": "x", "texts": ["y"], "intent": "a"}, "exactly one of 'text'"),
                         ({"text": "x", "intent": ["a", "b", "c"]}, "check takes one intent"),
                         ({"text": "x", "intent": ["a"]}, "check takes one intent"),
                         ({"text": "x", "intent": ""}, "empty name"),
                         ({"text": "x", "intent": {"name": " "}}, "empty name"),
                         ({"text": "x", "intent": "a", "threshold": 1.0}, "'threshold'"),
                         ({"text": "x", "intent": "a", "threshold": 0.0}, "'threshold'")):
        err(c.post("/check", json=body), 422, "invalid_request", needle)
    for body in ({"text": "x"}, {"text": "x", "intent": 3}, {"text": "x", "intent": {"name": "a", "colour": "red"}},
                 {"text": "x", "intent": [1, 2]}, {"text": "x", "intent": "a", "threshold": "0.5"},
                 {"text": "x", "intent": "a", "unknown": 1}, {"text": 5, "intent": "a"}):
        err(c.post("/check", json=body), 422, "invalid_request", "expected JSON body")
    err(c.post("/check", json={"texts": ["a", "b", "c"], "intent": "a"}), 413, "limit_exceeded", "--max-batch")
    err(c.post("/check", json={"text": "x" * 31, "intent": "a"}), 413, "limit_exceeded", "--max-text-chars")
    err(c.post("/check", json={"text": "x", "intent": "n" * 21}), 413, "limit_exceeded", "the limit is 20")
    err(c.post("/check", json={"text": "x", "intent": ["a", "d" * 101]}), 413, "limit_exceeded", "the limit is 100")
    for body, needle in (({"text": "x"}, "'list' over HTTP"), ({"text": "x", "intents": ["a"], "list": "s"},
                                                                 "exactly one of 'intents'"),
                         ({"text": "x", "intents": ["a", "a"]}, "unique"), ({"text": "x", "intents": []}, "empty"),
                         ({"texts": [], "intents": ["a"]}, "empty")):
        err(c.post("/scores", json=body), 422, "invalid_request", needle)
    err(c.post("/scores", json={"text": "x", "intents": ["a", "b", "c", "d"]}), 413, "limit_exceeded",
        "--max-intents")
    err(c.post("/scores", json={"texts": ["a", "b", "c"], "intents": ["a"]}), 413, "limit_exceeded", "--max-batch")
    err(c.post("/scores", json={"text": "x", "list": "nope"}), 404, "not_found", "no saved intent list")
    err(c.post("/scores", json={"text": "x", "list": "Bad Name"}), 422, "invalid_request", "list name")
    err(c.post("/scores", json={"text": "x", "intents": [{"name": "a", "description": 3}]}), 422, "invalid_request")


def test_real_library_without_calibration_is_a_409(one_file):
    """The tiny model of conftest has no calibration file: /check and /scores answer 409 calibration_missing (the
    library's refusal), /classify works, and /health lists classify only."""
    clf = load(one_file)
    c = client(clf)
    h = c.get("/health").json()
    assert h["features"] == {"classify": True, "check": True, "scores": True}
    assert h["supported"] == ["classify"] and h["calibration"] == {"file": False, "functions": []}
    e = err(c.post("/check", json={"text": "where is my parcel", "intent": "order status"}), 409, "calibration_missing",
            "'check' is not available with this model")
    assert "does not ship" in e["message"] and one_file not in e["message"]
    e = err(c.post("/scores", json={"text": "where is my parcel", "intents": LABELS}), 409, "calibration_missing",
            "'scores' is not available with this model")
    assert "classify works" in e["message"]
    assert c.post("/classify", json={"text": "where is my parcel", "intents": LABELS}).status_code == 200


def test_check_and_scores_through_the_server_equal_the_library(cal_models):
    """With a calibration file next to a tiny model at the settings of R2Intent v1: every field of every result equals
    the library's for the request's batch (padding in a batch changes the last float digits, so the library is called
    with the same batch)."""
    clf = load(cal_models["full"])
    c = client(clf)
    h = c.get("/health").json()
    assert h["supported"] == ["classify", "check", "scores"]
    assert h["calibration"] == {"file": True, "functions": ["check", "scores"]}
    intent = LABELS[7]
    for extra in ({}, {"threshold": 0.3}):
        r = c.post("/check", json={"texts": MESSAGES, "intent": intent, **extra})
        assert r.status_code == 200, r.text
        want = clf.check(MESSAGES, intent, **extra)
        got = r.json()["results"]
        assert len(got) == len(want) and all(set(g) == CHECK_KEYS for g in got)
        assert [[g[k] for k in sorted(CHECK_KEYS)] for g in got] == \
            [[getattr(w, k) for k in sorted(CHECK_KEYS)] for w in want]
    r = c.post("/check", json={"text": MESSAGES[7], "intent": [intent["name"], intent["description"]],
                               "threshold": 0.9})
    w = clf.check(MESSAGES[7], (intent["name"], intent["description"]), threshold=0.9)
    assert r.json()["results"] == [{k: getattr(w, k) for k in CHECK_KEYS}]
    r = c.post("/scores", json={"texts": MESSAGES, "intents": LABELS})
    assert r.status_code == 200, r.text
    clf.set_labels(LABELS)
    want = clf.scores(MESSAGES)
    got = r.json()["results"]
    assert len(got) == len(want) == len(MESSAGES)
    for g, w in zip(got, want):
        assert_scores_answer(g, w)
        assert len(g["intents"]) == 5 and len(w.values) == len(LABELS) == 20 and "remainder" not in r.text
    assert [g["label"] for g in got] == [p.label for p in clf.classify(MESSAGES, cascade=False)]
    # 20 intents: the right intent can be one the model did not read, so the listed values can sum to less than 1
    assert any(w.remainder > 0 for w in want) and any(sum(x["value"] for x in g["intents"]) < 1 - 1e-6 for g in got)
    assert c.put("/lists/tiny", json={"intents": LABELS}).status_code == 201
    assert c.post("/scores", json={"texts": MESSAGES, "list": "tiny"}).json()["results"] == got
    r = c.post("/scores", json={"text": MESSAGES[4], "list": "tiny"}).json()["results"]
    assert len(r) == 1
    assert_scores_answer(r[0], clf.scores(MESSAGES[4]))
    few = LABELS[:3]
    r = c.post("/scores", json={"texts": MESSAGES[:4], "intents": few})
    clf.set_labels(few)
    for g, w in zip(r.json()["results"], clf.scores(MESSAGES[:4])):
        assert_scores_answer(g, w)                         # 3 intents: every one is read and listed, sum 1
        assert len(g["intents"]) == 3 and abs(sum(x["value"] for x in g["intents"]) - 1.0) < 1e-9


def test_check_and_scores_409_without_calibration(cal_models, caplog):
    caplog.set_level(logging.DEBUG)
    none = client(load(cal_models["none"]))
    assert none.get("/health").json()["supported"] == ["classify"]
    for path, body in (("/check", {"text": SECRET, "intent": LABELS[0]}),
                       ("/scores", {"texts": [SECRET, "x"], "intents": LABELS})):
        e = err(none.post(path, json=body), 409, "calibration_missing",
                f"'{path[1:]}' is not available with this model: {path[1:]} needs the calibration file "
                "r2intent_calibration.json")
        assert "does not ship one" in e["message"] and cal_models["none"] not in e["message"]
    one = client(load(cal_models["check_only"]))
    h = one.get("/health").json()
    assert h["supported"] == ["classify", "check"] and h["calibration"] == {"file": True, "functions": ["check"]}
    assert one.post("/check", json={"text": "x", "intent": LABELS[0]}).status_code == 200
    e = err(one.post("/scores", json={"text": "x", "intents": LABELS}), 409, "calibration_missing",
            "its release ships no calibration for scores")
    assert "names only check" in e["message"]
    assert SECRET not in caplog.text and "no calibration file" in caplog.text


def test_classify_through_the_server_equals_the_library(one_file):
    clf = AnyIntent.from_pretrained(one_file, device="cpu")
    c = client(clf)
    for extra in ({}, {"k": 3}):
        r = c.post("/classify", json={"texts": MESSAGES, "intents": LABELS, **extra})
        assert r.status_code == 200, r.text
        clf.set_labels(LABELS)
        want = clf.classify(MESSAGES, **extra)
        assert [(g["label"], g["index"], g["score"], g["margin"]) for g in r.json()["results"]] == \
            [(p.label, p.index, p.score, p.margin) for p in want]
    h = c.get("/health").json()
    assert h["config"]["layout"] == "one_file" and h["revision"] == clf.revision


def test_no_text_in_the_log(caplog):
    caplog.set_level(logging.DEBUG)
    c = client(FakeWithExtras())
    c.post("/classify", json={"texts": [SECRET + " lost card"], "intents": INTENTS})
    c.put("/lists/l", json={"intents": INTENTS})
    c.post("/classify", json={"text": SECRET, "list": "l"})
    c.post("/classify", json={"text": SECRET * 1000, "intents": INTENTS})          # refused: too long
    assert c.post("/check", json={"texts": [SECRET, "x"], "intent": INTENTS[0]}).status_code == 200
    assert c.post("/scores", json={"text": SECRET, "list": "l"}).status_code == 200
    refused = client(FakeWithExtras(functions=None)).post("/check", json={"text": SECRET, "intent": "a"})
    assert refused.status_code == 409
    assert SECRET not in caplog.text and "classify: 1 message(s)" in caplog.text
    assert "check: 2 message(s)" in caplog.text and "scores: 1 message(s)" in caplog.text
    caplog.clear()
    client(log_texts=True).post("/classify", json={"text": SECRET, "intents": INTENTS})
    assert SECRET in caplog.text


def test_internal_error_is_a_json_500_without_text(caplog):
    caplog.set_level(logging.DEBUG)
    c = client(FakeBroken())
    r = c.post("/classify", json={"text": SECRET, "intents": INTENTS})
    err(r, 500, "internal_error", "server log")
    assert SECRET not in r.text and SECRET not in caplog.text and "RuntimeError" in caplog.text


def test_no_cors_by_default():
    origin = {"Origin": "https://example.org"}
    r = client().post("/classify", json={"text": "x", "intents": INTENTS}, headers=origin)
    assert r.status_code == 200 and "access-control-allow-origin" not in r.headers
    r = client(cors=["https://example.org"]).post("/classify", json={"text": "x", "intents": INTENTS}, headers=origin)
    assert r.headers.get("access-control-allow-origin") == "https://example.org"


def test_main_builds_the_server(monkeypatch, tmp_path):
    import uvicorn
    seen = {}
    monkeypatch.setattr(server, "_load_model", lambda args: FakeClassifier())
    monkeypatch.delenv("ANYINTENT_LISTS_DIR", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    monkeypatch.setenv("ANYINTENT_API_KEY", "env-key")
    rc = server.main(["--model", "fake/model", "--port", "8123", "--lists-dir", str(tmp_path / "l"), "--max-batch",
                      "5"])
    assert rc == 0 and seen["host"] == "127.0.0.1" and seen["port"] == 8123
    err(TestClient(seen["app"]).get("/health"), 400, "host_not_allowed", "--allowed-host testserver")
    c = TestClient(seen["app"], base_url="http://127.0.0.1:8123")
    assert c.get("/health").json() == {"status": "ok", "ready": True}
    h = c.get("/health", headers={"Authorization": "Bearer env-key"}).json()
    assert h["limits"]["max_batch"] == 5 and h["lists"]["persistent"] is True and h["model"] == "fake/model"
    monkeypatch.delenv("ANYINTENT_API_KEY")
    monkeypatch.delenv("ANYINTENT_MODEL", raising=False)
    assert server.main([]) == 2                                     # no model
    assert server.main(["--model", "m", "--max-batch", "0"]) == 2

    from anyintent import AccessError

    def refused(args):
        raise AccessError("No access to the model repository 'm'. It is gated")

    monkeypatch.setattr(server, "_load_model", refused)
    assert server.main(["--model", "m"]) == 1

    def no_gpu(args):
        raise RuntimeError("Found no NVIDIA driver on your system")

    monkeypatch.setattr(server, "_load_model", no_gpu)
    assert server.main(["--model", "m", "--device", "cuda"]) == 1
    monkeypatch.setattr(server, "_load_model", lambda args: FakeClassifier())
    assert server.main(["--model", "m", "--host", "0.0.0.0"]) == 0          # as in Docker, without an API key
    err(TestClient(seen["app"], base_url="http://evil.example").get("/lists"), 400, "host_not_allowed")
    assert TestClient(seen["app"], base_url="http://127.0.0.1:8000").get("/lists").status_code == 200
    monkeypatch.setenv("ANYINTENT_API_KEY", "env-key")
    assert server.main(["--model", "m", "--host", "0.0.0.0"]) == 0          # with a key: any Host
    assert TestClient(seen["app"], base_url="http://intent.example.org").get("/health").status_code == 200
    monkeypatch.delenv("ANYINTENT_API_KEY")
    assert server.main(["--model", "m", "--allowed-host", "intent.example.org"]) == 0
    assert TestClient(seen["app"], base_url="http://intent.example.org").get("/health").status_code == 200
    assert TestClient(seen["app"], base_url="http://localhost").get("/health").status_code == 200   # always
    err(TestClient(seen["app"], base_url="http://other.example").get("/health"), 400, "host_not_allowed")
    assert server.main(["--model", "m", "--label-text-mode", "v2", "--doubt-margin", "0.5"]) == 0
    c = TestClient(seen["app"], base_url="http://127.0.0.1:8000")
    h = c.get("/health").json()
    assert h["label_text_mode"] == "v2" and h["doubt_margin"] == 0.5
    assert "doubtful" in c.post("/classify", json={"text": "x", "intents": INTENTS}).json()["results"][0]
    assert server.main(["--model", "m", "--doubt-margin", "-1"]) == 2
    assert server.main(["--model", "m", "--max-concurrency", "0"]) == 2
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    assert server.main(["--model", "m", "--lists-dir", str(blocker / "lists")]) == 2      # a clear error, no traceback


def test_args_defaults(monkeypatch):
    for v in ("ANYINTENT_MODEL", "ANYINTENT_REVISION", "ANYINTENT_LISTS_DIR", "ANYINTENT_LABEL_TEXT_MODE"):
        monkeypatch.setenv(v, "")                        # empty, as an unset ARG of the Docker image leaves it
    a = server.build_parser().parse_args(["--model", "m"])
    assert (a.host, a.port, a.max_batch, a.max_text_chars, a.max_intents, a.log_texts, a.cors_origin, a.allowed_host) \
        == ("127.0.0.1", 8000, 64, 2000, 200, False, [], [])
    assert (a.revision, a.lists_dir, a.label_text_mode, a.doubt_margin, a.max_concurrency) == (None, None, None, None,
                                                                                               64)
    assert json.dumps(vars(a))


class _Args:
    def __init__(self, host="127.0.0.1", allowed_host=()):
        self.host, self.allowed_host = host, list(allowed_host)


def test_allowed_hosts_rule():
    loop = ["127.0.0.1", "localhost", "[::1]"]
    assert server.allowed_hosts_for(_Args(), None) == loop
    assert server.allowed_hosts_for(_Args(), "key") == loop
    assert server.allowed_hosts_for(_Args("0.0.0.0"), None) == loop           # Docker without a key: the guard stays
    assert server.allowed_hosts_for(_Args("0.0.0.0"), "key") is None          # a key protects every endpoint
    assert server.allowed_hosts_for(_Args("0.0.0.0", ["a.example"]), None) == loop + ["a.example"]
    assert server.allowed_hosts_for(_Args("0.0.0.0", ["*"]), None) is None
    assert server.allowed_hosts_for(_Args("::1"), None) == loop


def test_host_guard():
    loop = ["127.0.0.1", "localhost", "[::1]"]
    c = client(allowed_hosts=loop)
    for host in ("127.0.0.1:8000", "localhost", "[::1]:8000", "[::1]", "LOCALHOST:1"):
        assert c.get("/lists", headers={"Host": host}).status_code == 200, host
    for host in ("evil.example", "127.0.0.1.evil.example", "[::2]:8000", "::1", "[::1"):
        e = err(c.get("/lists", headers={"Host": host}), 400, "host_not_allowed", "--allowed-host")
        assert "DNS rebinding" in e["message"], host
    c = client(allowed_hosts=["::1", "intent.example.org:8000", "*.corp.example"])
    assert c.get("/health", headers={"Host": "[::1]:9"}).status_code == 200
    for base in ("http://intent.example.org", "http://a.corp.example:80"):
        assert client(allowed_hosts=["intent.example.org:8000", "*.corp.example"], base_url=base).get(
            "/health").status_code == 200, base
    r = client(allowed_hosts=loop, base_url="http://127.0.0.1").get("/health", headers={"Host": ""})
    err(r, 400, "host_not_allowed")
    origin = {"Origin": "https://example.org", "Access-Control-Request-Method": "POST"}
    r = client(allowed_hosts=loop, cors=["https://example.org"], base_url="http://evil.example").options(
        "/classify", headers=origin)
    err(r, 400, "host_not_allowed")                    # the Host check runs before CORS


def test_concurrency_cap_and_health_while_busy():
    import asyncio
    import threading
    import time

    gate, inside = threading.Event(), threading.Event()

    class Slow(FakeClassifier):
        def classify(self, text, k=None, return_scores=False):
            inside.set()
            gate.wait(10)
            return super().classify(text, k=k, return_scores=return_scores)

    c = client(Slow(), max_concurrency=1)
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post("/classify", json={"text": "x", "intents": INTENTS})))
    t.start()
    try:
        assert inside.wait(10)
        e = err(c.post("/classify", json={"text": "y", "intents": INTENTS}), 503, "busy", "--max-concurrency")
        assert "in progress" in e["message"]
        t0 = time.perf_counter()
        assert c.get("/health").json()["status"] == "ok" and time.perf_counter() - t0 < 5     # not counted, no wait
        assert c.get("/openapi.json").status_code == 200
    finally:
        gate.set()
        t.join(10)
    assert out["r"].status_code == 200
    assert c.post("/classify", json={"text": "y", "intents": INTENTS}).status_code == 200      # the slot is free again
    route = {r.path: r for r in c.app.routes}["/health"]
    assert asyncio.iscoroutinefunction(route.endpoint)        # /health needs no worker thread


def test_list_names_and_texts_that_are_refused():
    c = client()
    err(c.put("/lists/abc%0A", json={"intents": INTENTS}), 422, "invalid_request", "list name")
    err(c.put("/lists/sur%ED%A0%80", json={"intents": INTENTS}), 422, "invalid_request", "list name")

    def raw_json(obj):
        """The body as a client sends it: JSON allows a lone surrogate as the escape \\ud800 (json.dumps writes it so,
        in ASCII bytes). Given as content=, since newer httpx versions refuse to encode a lone surrogate for json=."""
        body = json.dumps(obj).encode("ascii")
        assert b"\\ud800" in body
        return {"content": body, "headers": {"content-type": "application/json"}}

    err(c.put("/lists/sur", **raw_json({"intents": ["a\ud800b"]})), 422, "invalid_request", "not valid Unicode")
    err(c.post("/classify", **raw_json({"text": "a\ud800b", "intents": INTENTS})), 422, "invalid_request",
        "not valid Unicode")
    assert c.get("/lists").json()["lists"] == []
    e = err(c.get("/lists/missing"), 404, "not_found")
    assert "GET /lists" in e["message"]


def test_openapi_has_response_schemas():
    paths = client().get("/openapi.json").json()["paths"]
    for path, method in (("/health", "get"), ("/classify", "post"), ("/check", "post"), ("/scores", "post"),
                         ("/lists", "get"), ("/lists/{name}", "get"), ("/lists/{name}", "put"),
                         ("/lists/{name}", "delete")):
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        assert "$ref" in schema, (path, method, schema)
    assert "201" in paths["/lists/{name}"]["put"]["responses"]
    c = client(lists_dir=None)
    c.put("/lists/a", json={"intents": ["x", {"name": "y", "description": "d"}]})
    assert c.get("/lists/a").json() == {"name": "a", "intents": [{"name": "x"}, {"name": "y", "description": "d"}]}
    assert c.get("/lists").json() == {"lists": [{"name": "a", "n_intents": 2}], "persistent": False}
    assert c.delete("/lists/a").json() == {"name": "a", "deleted": True}


def test_check_and_scores_are_described():
    """The OpenAPI page says what yes and probability mean; for scores, that the listed intents are the ones the model
    read, that an intent not listed was not read (value 0, still possible), that the values sum to 1 with 5 intents or
    fewer and can sum to less than 1 with more, and that the part missing to 1 is not a 'none of these' answer (no
    remainder in the answer or in its description); that the numbers are calibrated on intents written with a
    description, and the 409."""
    api = client(FakeWithExtras()).get("/openapi.json").json()
    check, scores = api["paths"]["/check"]["post"], api["paths"]["/scores"]["post"]
    for needle in ("`yes`", "`probability`", "equally likely", "written with a description", "409",
                   "calibration_missing", "501"):
        assert needle in check["description"], needle
    for needle in ("`intents`", '`{"intent": name, "value": number}`', "`label`", "the intents the model read",
                   "at most 5 intents", "An intent not listed was not read", "its value is 0",
                   "not read, not impossible", "can still be the right intent",
                   "every intent is listed and the values sum to 1", "sum to less than 1", "did not read",
                   "The part missing to reach 1 is not a 'none of these intents' answer", "no such mode",
                   "assume that one of the given intents is right", "written with a description", "409",
                   "calibration_missing", "/classify returns without `k`"):
        assert needle in scores["description"], needle
    assert "This is" not in scores["description"]
    assert "model read" in scores["summary"] and "remainder" not in json.dumps(api)
    assert {"409", "501"} <= set(check["responses"]) and {"409", "501"} <= set(scores["responses"])
    for op, model in ((check, "CheckOut"), (scores, "ScoresOut")):
        assert op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(model)
    schemas = api["components"]["schemas"]
    props = schemas["ScoresResultOut"]["properties"]
    assert set(props) == {"label", "intents"} and set(schemas["IntentValueOut"]["properties"]) == {"intent", "value"}
    for needle in ("the intents the model read", "at most 5", "highest value to the lowest", "not listed was not read",
                   "value is 0", "can still be the right intent", "the values sum to 1", "sum to less than 1",
                   "The part missing to reach 1 is not a 'none of these intents' answer",
                   "assume that one of the given intents is right"):
        assert needle in props["intents"]["description"], needle
    assert "first one listed" in props["label"]["description"]
    assert "written with a description" in schemas["IntentValueOut"]["properties"]["value"]["description"]
    assert "equally likely" in schemas["CheckResultOut"]["properties"]["probability"]["description"]


def test_check_is_a_beta_and_nothing_says_zero_shot():
    """The OpenAPI page marks /check as a beta (it shipped in R2Intent v1 after it missed one pre-registered DEV
    calibration limit) in its summary, its description and the API description; /scores is not marked; no text calls
    the models zero-shot, and the API description says that no training examples of the intents are needed."""
    api = client(FakeWithExtras()).get("/openapi.json").json()
    check, scores = api["paths"]["/check"]["post"], api["paths"]["/scores"]["post"]
    assert check["summary"].endswith("(beta)")
    for needle in ("Beta: check shipped in R2Intent v1", "missed one pre-registered calibration limit on the DEV data",
                   "treat its probability as approximate"):
        assert needle in check["description"], needle
    assert "beta" not in (scores["summary"] + scores["description"]).lower()
    info = api["info"]["description"]
    assert "/check (yes or no for one intent, with a probability; a beta)" in info
    assert "no training examples of your intents are needed" in info
    text = json.dumps(api).lower()
    assert "zero-shot" not in text and "zero shot" not in text


def test_serve_cli_without_the_extra(monkeypatch, capsys):
    import sys
    from anyintent import serve_cli
    monkeypatch.delitem(sys.modules, "anyintent.server")
    monkeypatch.setitem(sys.modules, "fastapi", None)          # import fastapi now fails, as without the extra
    assert serve_cli.main(["--help"]) == 2
    assert 'anyintent[server]' in capsys.readouterr().err


def test_install_hint_pins_a_release_tag():
    """The install hint of the server extra names a release tag (a bare repository URL would install the default
    branch over the installed release), and in a release it is this release's own tag."""
    tag = re.search(r'"anyintent\[server\] @ git\+https://github\.com/lukamaga/anyintent@v(\d+\.\d+\.\d+)"$',
                    server.INSTALL_HINT)
    assert tag, server.INSTALL_HINT
    if re.fullmatch(r"\d+\.\d+\.\d+", __version__):
        assert tag.group(1) == __version__
