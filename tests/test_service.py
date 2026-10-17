"""The serving core of the HTTP and MCP servers (anyintent.service): needs no extra package. Validation and limits,
named lists and their files, check and scores by attribute, loading states, privacy of the log, classify through the
service equal to the library called directly (tiny random model of conftest), and check and scores through the service
equal to the library on a tiny model with the settings of R2Intent v1 and a calibration file (tests/calib_support.py),
with the 409 of a model without it."""
import json
import logging
import os
import threading
import time
import warnings

import pytest

from anyintent import AnyIntent
from anyintent.service import (LABEL_TEXT_MODES, CalibrationMissing, InvalidRequest, LimitExceeded, Limits,
                               ModelRefused, NotAvailable, NotFound, NotReady, Service, check_list_name, check_result,
                               jsonable, scores_result)
from conftest import LABELS, MESSAGES
from serving_fakes import (INTENTS, SECRET, FakeBroken, FakeClassifier, FakeWithExtras, assert_scores_answer,
                           calibrated_models)

CHECK_KEYS = {"intent", "yes", "probability", "score", "threshold"}
SCORES_KEYS = {"label", "intents"}


@pytest.fixture(scope="module")
def cal_models(tmp_path_factory, tok):
    return calibrated_models(str(tmp_path_factory.mktemp("serving_cal")), tok)


def load(folder):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AnyIntent.from_pretrained(folder, device="cpu")


def svc(clf=None, **kw):
    return Service(clf or FakeClassifier(), model_id="fake/model", **kw)


def test_classify_shapes_and_label_memo():
    clf = FakeClassifier()
    s = svc(clf)
    one = s.classify("I lost my bank card", intents=INTENTS)
    assert len(one) == 1 and one[0]["label"] == "lost card" and one[0]["index"] == 0
    assert set(one[0]) == {"label", "index", "score", "margin", "runner_up"} and isinstance(one[0]["margin"], float)
    many = s.classify(["I lost my bank card", "when will my refund arrive", "close account please"], intents=INTENTS)
    assert [r["label"] for r in many] == ["lost card", "refund status", "close account"]
    assert clf.set_labels_calls == 1                  # the same label set is applied once
    s.classify("x", intents=INTENTS[:2])
    assert clf.set_labels_calls == 2
    s.classify("x", intents=INTENTS[:2], k=1)
    assert clf.set_labels_calls == 2 and clf.classify_calls[-1][1] == 1
    clf.set_labels(["other"])                         # someone changed the instance: applied again
    s.classify("x", intents=INTENTS[:2])
    assert clf.set_labels_calls == 4


def test_validation_messages():
    s = svc(limits=Limits(max_batch=3, max_text_chars=20, max_intents=3, max_name_chars=10, max_description_chars=30))
    bad = [((), dict(intents=INTENTS[:1]), "texts"),
           (([],), dict(intents=INTENTS[:1]), "empty"),
           (([1],), dict(intents=INTENTS[:1]), "message 0: expected a string, got a number"),
           (("x",), dict(), "exactly one"),
           (("x",), dict(intents=INTENTS[:1], list_name="a"), "exactly one"),
           (("x",), dict(intents=[]), "empty"),
           (("x",), dict(intents="lost card"), "must be a list"),
           (("x",), dict(intents=["a", "a"]), "unique"),
           (("x",), dict(intents=["a", " a "]), "unique"),
           (("x",), dict(intents=[""]), "empty name"),
           (("x",), dict(intents=[{"description": "d"}]), "needs a 'name'"),
           (("x",), dict(intents=[{"name": "a", "desc": "d"}]), "unknown key"),
           (("x",), dict(intents=[{"name": 3}]), "'name' must be a string"),
           (("x",), dict(intents=[3]), "got a number"),
           (("x",), dict(intents=["a"], k=0), "'k' must be between"),
           (("x",), dict(intents=["a"], k=True), "'k' must be an integer"),
           (("x",), dict(intents=["a"], k=1.5), "'k' must be an integer")]
    for args, kw, needle in bad:
        with pytest.raises(InvalidRequest) as e:
            s.classify(*args, **kw) if args else s.classify(None, **kw)
        assert needle in e.value.message and e.value.status == 422, (args, kw, e.value.message)
    limits = [((["a", "b", "c", "d"],), dict(intents=["a"]), "the limit is 3 (--max-batch)"),
              (("x" * 21,), dict(intents=["a"]), "the limit is 20 (--max-text-chars)"),
              (("x",), dict(intents=["a", "b", "c", "d"]), "the limit is 3 (--max-intents)"),
              (("x",), dict(intents=["n" * 11]), "the limit is 10"),
              (("x",), dict(intents=[{"name": "a", "description": "d" * 31}]), "the limit is 30")]
    for args, kw, needle in limits:
        with pytest.raises(LimitExceeded) as e:
            s.classify(*args, **kw)
        assert needle in e.value.message and e.value.status == 413, e.value.message
    with pytest.raises(NotFound):
        s.classify("x", list_name="missing")
    assert s.normalize_intents([" a ", {"name": " b ", "description": " d "}, {"name": "c", "description": None}]) == \
        [{"name": "a"}, {"name": "b", "description": "d"}, {"name": "c"}]


def test_limits_are_checked():
    with pytest.raises(ValueError):
        Limits(max_batch=0)
    with pytest.raises(ValueError):
        Limits(max_intents=True)
    assert Limits().max_body_bytes > 6 * 64 * 2000


def test_lists_in_memory_and_on_disk(tmp_path):
    s = svc()
    r = s.save_list("support", INTENTS)
    assert r == {"name": "support", "n_intents": 3, "created": True, "persistent": False}
    assert s.save_list("support", INTENTS[:2])["created"] is False
    assert s.get_list("support")["intents"] == [{"name": "lost card", "description": INTENTS[0]["description"]},
                                                {"name": "refund status", "description": INTENTS[1]["description"]}]
    assert s.classify("my refund", list_name="support")[0]["label"] == "refund status"
    for bad in ("Support", "a.b", "../x", "-x", "", "x" * 65, "con", "nul", "com1", "a/b", 3, "abc\n", "abc\r",
                "abc ", "\nabc"):
        with pytest.raises(InvalidRequest):
            s.save_list(bad, INTENTS)
        with pytest.raises(InvalidRequest):
            check_list_name(bad)
    assert s.delete_list("support") == {"name": "support", "deleted": True}
    with pytest.raises(NotFound):
        s.get_list("support")
    with pytest.raises(NotFound):
        s.delete_list("support")

    d = tmp_path / "lists"
    a = svc(lists_dir=str(d))
    assert a.save_list("billing", INTENTS)["persistent"] is True
    a.save_list("other-1", ["x", "y"])
    stored = json.load(open(d / "billing.json", encoding="utf-8"))
    assert stored["format"] == "anyintent-intent-list" and stored["name"] == "billing" and len(stored["intents"]) == 3
    assert sorted(os.listdir(d)) == ["billing.json", "other-1.json"]       # no temporary file left
    if os.name != "nt":      # Windows ignores chmod bits apart from read-only (st_mode is 0o666 for a writable file)
        assert (os.stat(d / "billing.json").st_mode & 0o777) == 0o600     # readable by the owner only
    with pytest.raises(InvalidRequest):                                    # the file path is checked again
        a.lists._path("abc\n")
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    (d / "wrongname.json").write_text(json.dumps({"format": "anyintent-intent-list", "name": "other", "intents": ["a"]}), encoding="utf-8")
    (d / "Upper.json").write_text(json.dumps({"format": "anyintent-intent-list", "name": "Upper", "intents": ["a"]}), encoding="utf-8")
    (d / "dupes.json").write_text(json.dumps({"format": "anyintent-intent-list", "name": "dupes", "intents": ["a", "a"]}), encoding="utf-8")
    (d / "notes.txt").write_text("ignored", encoding="utf-8")
    b = svc(lists_dir=str(d))                                          # read again at start
    assert b.list_lists() == {"lists": [{"name": "billing", "n_intents": 3}, {"name": "other-1", "n_intents": 2}],
                              "persistent": True}
    assert b.get_list("billing")["intents"] == a.get_list("billing")["intents"]
    b.delete_list("other-1")
    assert not (d / "other-1.json").exists()
    small = svc(lists_dir=str(tmp_path / "small"), limits=Limits(max_lists=1))
    small.save_list("a", ["x"])
    with pytest.raises(LimitExceeded):
        small.save_list("b", ["x"])
    small.save_list("a", ["y"])                                         # replacing is not a new list


def test_list_file_with_a_bom_is_read(tmp_path):
    """A list file saved again by a Windows editor or by Windows PowerShell 5.1 (Set-Content -Encoding UTF8) starts with
    a UTF-8 BOM and may have CRLF line ends: it is read like the server's own file."""
    d = tmp_path / "lists"
    intents = [{"name": "zwrot pieniędzy", "description": "Klient prosi o zwrot pieniędzy za towar."},
               {"name": "siuntos būsena", "description": "Klientas klausia, kur yra jo užsakymas."}, "close account"]
    a = svc(lists_dir=str(d))
    a.save_list("bom", intents)
    p = d / "bom.json"
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and "zwrot pieniędzy".encode("utf-8") in raw   # UTF-8, no BOM
    p.write_bytes(b"\xef\xbb\xbf" + raw.replace(b"\n", b"\r\n"))
    b = svc(lists_dir=str(d))
    assert b.list_lists() == {"lists": [{"name": "bom", "n_intents": 3}], "persistent": True}
    assert b.get_list("bom")["intents"] == a.get_list("bom")["intents"]


def test_check_and_scores_follow_the_classifier():
    plain = svc(FakeClassifier())
    for call in (lambda: plain.check("x", "a"), lambda: plain.scores("x", intents=["a"])):
        with pytest.raises(NotAvailable) as e:
            call()
        assert e.value.status == 501 and "is used automatically once" in e.value.message
    h = plain.health()
    assert h["features"] == {"classify": True, "check": False, "scores": False}
    assert h["supported"] == ["classify"] and h["calibration"] is None
    full = svc(FakeWithExtras())
    h = full.health()
    assert h["features"] == {"classify": True, "check": True, "scores": True}
    assert h["supported"] == ["classify", "check", "scores"]
    assert h["calibration"] == {"file": True, "functions": ["check", "scores"]}
    r = full.check(["I lost my bank card", "hello"], {"name": "lost card", "description": "a card was lost"})
    assert [set(x) for x in r] == [CHECK_KEYS] * 2 and r[0]["intent"] == "lost card" and r[0]["yes"] is True
    assert isinstance(r[0]["probability"], float) and r[1]["yes"] is False and r[0]["threshold"] == 0.5
    assert json.dumps(r)
    assert full.check("lost card", "lost card", threshold=0.9)[0]["threshold"] == 0.9
    assert full.check("card", ["lost card", "a card was lost"]) == full.check("card", {"name": "lost card",
                                                                                      "description": "a card was lost"})
    assert full.check("card", ["lost card", None]) == full.check("card", "lost card")
    for t in (0, 1, 1.5, -0.1, True, "0.5", float("nan")):
        with pytest.raises(InvalidRequest, match="'threshold'"):
            full.check("x", "a", threshold=t)
    for bad in (["a", "b", "c"], ["a"], [], [1, "d"], ["a", 2], [["a", "b"]], 3, None, {"description": "d"}):
        with pytest.raises(InvalidRequest, match="intent"):
            full.check("x", bad)
    sc = full.scores(["I lost my bank card"], intents=INTENTS)
    assert set(sc[0]) == SCORES_KEYS and sc[0]["label"] == "lost card" and json.dumps(sc)
    assert sorted(x["intent"] for x in sc[0]["intents"]) == ["close account", "lost card", "refund status"]
    assert_scores_answer(sc[0], FakeWithExtras().set_labels(INTENTS).scores("I lost my bank card"))


def test_check_and_scores_without_calibration_with_stand_ins():
    """The 409 is decided per call from the library's refusal and the calibration status read after it."""
    none = svc(FakeWithExtras(functions=None))
    assert none.health()["supported"] == ["classify"]
    assert none.health()["calibration"] == {"file": False, "functions": []}
    for feature, call in (("check", lambda: none.check("x", "a")), ("scores", lambda: none.scores("x", intents=["a"]))):
        with pytest.raises(CalibrationMissing) as e:
            call()
        assert e.value.status == 409 and e.value.code == "calibration_missing"
        m = e.value.message
        assert m.startswith(f"'{feature}' is not available with this model") and "r2intent_calibration.json" in m
        assert "does not ship" in m and "classify works" in m and "/private/model" not in m      # no local path
    clf = FakeWithExtras(functions=("check",))
    one = svc(clf)
    assert one.health()["supported"] == ["classify", "check"] and one.check("x", "a")[0]["intent"] == "a"
    with pytest.raises(CalibrationMissing, match=r"its release ships no calibration for scores .*names only check"):
        one.scores("x", intents=["a"])
    assert svc(FakeWithExtras(functions=("scores",))).health()["supported"] == ["classify", "scores"]
    clf.functions = ("check", "scores")                    # a calibration file put next to the model later
    assert one.scores("x", intents=["a"])[0]["label"] == "a"
    assert one.health()["supported"] == ["classify", "check", "scores"]

    class OtherRefusal(FakeWithExtras):
        def check(self, text, intent, threshold=0.5):
            from anyintent import AnyIntentError
            raise AnyIntentError("check: the model runs at w 10.0 (a w= override at load) but was calibrated at w 60.0")

    with pytest.raises(ModelRefused, match="w= override") as e:          # calibrated, refused for another reason
        svc(OtherRefusal()).check("x", "a")
    assert e.value.status == 422


def test_check_and_scores_equal_the_library(cal_models):
    """Every field of the service's check and scores equals the library's result for the same messages, intent and
    label set (the same batch: padding in a batch changes the last float digits), one message too, and a saved list."""
    clf = load(cal_models["full"])
    s = Service(clf)
    intent = LABELS[7]                                     # lost card, with a description
    for kwargs in ({}, {"threshold": 0.25}):
        got = s.check(MESSAGES, intent, **kwargs)
        want = clf.check(MESSAGES, intent, **kwargs)
        assert got == [check_result(r) for r in want] and len(got) == len(MESSAGES)
        assert all(g[k] == getattr(w, k) for g, w in zip(got, want) for k in CHECK_KEYS)
    assert s.check(MESSAGES[3], [intent["name"], intent["description"]]) == \
        [check_result(clf.check(MESSAGES[3], (intent["name"], intent["description"])))]
    assert s.check(MESSAGES[0], "refund request") == [check_result(clf.check(MESSAGES[0], "refund request"))]
    got = s.scores(MESSAGES, intents=LABELS)
    clf.set_labels(LABELS)
    want = clf.scores(MESSAGES)
    assert got == [scores_result(r) for r in want] and len(got) == len(MESSAGES)
    for g, w in zip(got, want):
        assert_scores_answer(g, w)
        assert len(g["intents"]) == 5 and len(w.values) == len(LABELS) == 20
    assert [g["label"] for g in got] == [p.label for p in clf.classify(MESSAGES, cascade=False)]
    # 20 intents: the right intent can be one the model did not read, so the listed values can sum to less than 1
    assert any(w.remainder > 0 for w in want) and any(sum(x["value"] for x in g["intents"]) < 1 - 1e-6 for g in got)
    assert s.scores(MESSAGES[2], intents=LABELS) == [scores_result(clf.scores(MESSAGES[2]))]
    s.save_list("tiny", LABELS)
    assert s.scores(MESSAGES, list_name="tiny") == got
    few = LABELS[:4]
    got = s.scores(MESSAGES[:5], intents=few)
    clf.set_labels(few)
    want = clf.scores(MESSAGES[:5])
    assert got == [scores_result(r) for r in want] and all(w.remainder == 0.0 for w in want)
    for g, w in zip(got, want):
        assert_scores_answer(g, w)                         # 4 intents: every one is read and listed, sum 1
        assert len(g["intents"]) == 4 and abs(sum(x["value"] for x in g["intents"]) - 1.0) < 1e-9
    h = s.health()
    assert h["supported"] == ["classify", "check", "scores"]
    assert h["calibration"] == {"file": True, "functions": ["check", "scores"]}


def test_missing_calibration_with_the_library(cal_models, caplog):
    caplog.set_level(logging.DEBUG)
    s = Service(load(cal_models["none"]))
    assert s.health()["supported"] == ["classify"] and s.health()["calibration"] == {"file": False, "functions": []}
    assert s.classify(MESSAGES[:2], intents=LABELS)[0]["label"]               # classify works
    for feature, call in (("check", lambda: s.check(SECRET, LABELS[0])),
                          ("scores", lambda: s.scores(SECRET, intents=LABELS))):
        with pytest.raises(CalibrationMissing) as e:
            call()
        assert e.value.message.startswith(f"'{feature}' is not available with this model")
        assert cal_models["none"] not in e.value.message and "does not ship one" in e.value.message
    assert "no calibration file" in caplog.text and SECRET not in caplog.text      # the library's reason, in the log
    one = Service(load(cal_models["check_only"]))
    assert one.health()["supported"] == ["classify", "check"]
    assert one.health()["calibration"] == {"file": True, "functions": ["check"]}
    assert set(one.check(MESSAGES[0], LABELS[0])[0]) == CHECK_KEYS
    with pytest.raises(CalibrationMissing, match="ships no calibration for scores"):
        one.scores(MESSAGES[0], intents=LABELS)
    named = Service(load(cal_models["full"]), label_text_mode="name")
    with pytest.raises(ModelRefused, match="--label-text-mode"):              # calibrated, but another label text mode
        named.scores(MESSAGES[0], intents=LABELS)
    assert named.check(MESSAGES[0], LABELS[0])[0]["intent"] == LABELS[0]["name"]       # check uses the model's own mode


def test_refusal_holds_no_local_folder(cal_models, tmp_path, caplog):
    """A refusal of check or scores (422) whose library message names the model folder, here a local folder that
    changed after the load, reaches the client without the path; the operator's log keeps it."""
    from calib_support import model_copy
    caplog.set_level(logging.DEBUG)
    folder = model_copy(cal_models["full"], str(tmp_path / "m"))
    s = Service(load(folder))
    assert s.health()["supported"] == ["classify", "check", "scores"]
    p = os.path.join(folder, "config.json")
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["anyintent_test_changed"] = True
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    for call in (lambda: s.check(SECRET, LABELS[0]), lambda: s.scores(SECRET, intents=LABELS)):
        with pytest.raises(ModelRefused, match="changed after it was loaded") as e:
            call()
        assert e.value.status == 422 and "the model folder 'm' changed" in e.value.message
        assert folder not in e.value.message and str(tmp_path) not in e.value.message
    assert folder in caplog.text and SECRET not in caplog.text
    assert s.without_model_folders(object(), f"x {folder} y") == f"x {folder} y"        # no library configuration


def test_calibration_status_is_read_when_the_model_loads(cal_models):
    lazy = Service(loader=lambda: load(cal_models["check_only"]), load_wait=30)
    assert lazy.health()["supported"] is None and "calibration" not in lazy.health()
    lazy.start_loading()
    lazy.classifier()
    assert lazy.health()["supported"] == ["classify", "check"]
    now = Service(loader=lambda: FakeWithExtras(functions=("scores",)))
    now.load_now()
    assert now.health()["supported"] == ["classify", "scores"]


def test_real_library_features_are_detected(one_file):
    clf = AnyIntent.from_pretrained(one_file, device="cpu")
    s = Service(clf)
    assert s.has("check") == hasattr(AnyIntent, "check") and s.has("scores") == hasattr(AnyIntent, "scores")
    lazy = Service(loader=lambda: clf)
    assert lazy.has("check") == hasattr(AnyIntent, "check")             # before loading: the class decides


def test_classify_equals_the_library(one_file):
    """The service's classify returns exactly what AnyIntent.classify returns for the same labels and messages."""
    clf = AnyIntent.from_pretrained(one_file, device="cpu")
    s = Service(clf)
    for k in (None, 3):
        got = s.classify(MESSAGES, intents=LABELS, k=k)
        clf.set_labels(LABELS)
        want = clf.classify(MESSAGES) if k is None else clf.classify(MESSAGES, k=k)
        assert [(g["label"], g["index"], g["score"], g["margin"]) for g in got] == \
            [(p.label, p.index, p.score, p.margin) for p in want]
    one = s.classify(MESSAGES[0], intents=LABELS)[0]
    clf.set_labels(LABELS)
    p = clf.classify(MESSAGES[0])
    assert (one["label"], one["score"], one["margin"]) == (p.label, p.score, p.margin)
    s.save_list("tiny", LABELS)
    assert s.classify(MESSAGES, list_name="tiny") == s.classify(MESSAGES, intents=LABELS)


def test_library_refusal_is_a_clear_error(one_file):
    from anyintent import AnyIntentError

    class Refusing(FakeClassifier):
        def classify(self, text, k=None, **kwargs):
            raise AnyIntentError("this model refuses the request, for a reason the library states")

    with pytest.raises(ModelRefused) as e:
        svc(Refusing()).classify("x", intents=["a"])
    assert e.value.status == 422 and "for a reason the library states" in e.value.message
    clf = AnyIntent.from_pretrained(one_file, device="cpu")
    clf.set_labels(LABELS)
    with pytest.raises(ModelRefused) as e:          # the tiny v1.1 model has no cascade threshold
        Service(clf)._call(clf.classify, "x", cascade=True)
    assert "cascade" in e.value.message


def test_loading_states():
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return FakeClassifier()

    s = Service(loader=slow, model_id="m")
    assert s.health(detail=False) == {"status": "loading", "ready": False}
    s.start_loading()
    with pytest.raises(NotReady) as e:
        s.classify("x", intents=["a"])
    assert "still loading" in e.value.message and e.value.status == 503
    gate.set()
    for _ in range(100):
        if s.ready:
            break
        time.sleep(0.02)
    assert s.ready and s.classify("x", intents=["a"])[0]["label"] == "a"
    assert s.health()["revision"] == FakeClassifier.revision

    def broken():
        raise RuntimeError("cannot reach the model repository")

    b = Service(loader=broken, load_wait=5)
    b.start_loading()
    with pytest.raises(NotReady) as e:
        b.classify("x", intents=["a"])
    assert "could not be loaded" in e.value.message and "RuntimeError" in e.value.message
    assert b.health()["status"] == "error" and "RuntimeError" in b.health()["error"]
    assert "cannot reach" not in b.health()["error"]          # not a library message: its text stays in the log
    w = Service(loader=lambda: (time.sleep(0.05), FakeClassifier())[1], load_wait=5)
    w.start_loading()
    assert w.classify("x", intents=["a"])[0]["label"] == "a"            # waits for the model
    assert Service(loader=FakeClassifier).load_now() is not None


def test_health_shows_settings_but_no_paths():
    h = svc().health()
    assert h["status"] == "ok" and h["ready"] and h["model"] == "fake/model" and h["revision"] == "fake-revision-0001"
    assert h["config"]["k"] == 8 and "root" not in h["config"] and "cross_dir" not in h["config"]
    assert h["config"]["active"] == {"w": 1.0} and h["limits"]["max_batch"] == 64 and h["log_texts"] is False
    assert h["label_text_mode"] is None and h["doubt_margin"] is None and h["lists"] == {"persistent": False,
                                                                                         "count": 0}
    assert json.dumps(h)


def test_texts_are_not_logged(caplog):
    caplog.set_level(logging.DEBUG)
    s = svc(FakeWithExtras())
    s.classify([SECRET + " lost card"], intents=INTENTS)
    s.save_list("x", INTENTS)
    s.classify(SECRET, list_name="x")
    s.check([SECRET, SECRET + " card"], INTENTS[0])
    s.scores(SECRET, list_name="x")
    for refused in (lambda: svc(FakeWithExtras(functions=None)).check(SECRET, SECRET),
                    lambda: svc(FakeWithExtras(functions=None)).scores(SECRET, intents=[SECRET])):
        with pytest.raises(CalibrationMissing):
            refused()
    assert SECRET not in caplog.text and "classify: 1 message(s), 3 intent(s)" in caplog.text
    assert "check: 2 message(s), 1 intent(s)" in caplog.text and "scores: 1 message(s), 3 intent(s)" in caplog.text
    caplog.clear()
    loud = svc(log_texts=True)
    loud.classify([SECRET], intents=INTENTS)
    assert SECRET in caplog.text and "--log-texts is on" in caplog.text


def test_internal_error_log_has_no_message(caplog):
    caplog.set_level(logging.DEBUG)
    s = svc(FakeBroken())
    with pytest.raises(RuntimeError) as e:
        s.classify(SECRET, intents=INTENTS)
    s.log_internal_error("POST /classify", e.value)
    assert "RuntimeError" in caplog.text and "Traceback, frames only" in caplog.text and SECRET not in caplog.text
    caplog.clear()
    svc(FakeBroken(), log_texts=True).log_internal_error("POST /classify", e.value)
    assert SECRET in caplog.text


def test_jsonable_and_result_shapes():
    import numpy as np
    from anyintent import CheckResult, ScoresResult
    assert jsonable({"a": np.float32(0.5), "b": np.int64(3), "c": np.bool_(True), "d": float("nan"),
                     "e": np.array([1.0, 2.0]), "f": (1, 2)}) == {"a": 0.5, "b": 3, "c": True, "d": None,
                                                                    "e": [1.0, 2.0], "f": [1, 2]}
    c = CheckResult(yes=True, probability=0.75, intent="a", label_text="a: d", score=61.5, threshold=0.5)
    assert check_result(c) == {"intent": "a", "yes": True, "probability": 0.75, "score": 61.5, "threshold": 0.5}
    # only the read intents (5 of 7), highest value first (ties: retrieval order); no remainder, no intent not read
    r = ScoresResult(values={"b": 0.0, "a": 0.25, "c": 0.35, "d": 0.1, "e": 0.05, "f": 0.0, "g": 0.05}, remainder=0.2,
                     read=["a", "c", "d", "g", "e"], label="c")
    assert scores_result(r) == {"label": "c", "intents": [{"intent": "c", "value": 0.35}, {"intent": "a", "value": 0.25},
                                                          {"intent": "d", "value": 0.1}, {"intent": "g", "value": 0.05},
                                                          {"intent": "e", "value": 0.05}]}
    assert_scores_answer(scores_result(r), r)
    # ties: the label first, then retrieval order
    r = ScoresResult(values={"w": 0.0, "x": 0.3, "y": 0.3, "z": 0.3}, remainder=0.1, read=["z", "x", "y"], label="y")
    assert [x["intent"] for x in scores_result(r)["intents"]] == ["y", "z", "x"]
    r = ScoresResult(values={"x": 0.5, "y": 0.5}, remainder=0.0, read=["y", "x"], label="y")
    assert scores_result(r) == {"label": "y", "intents": [{"intent": "y", "value": 0.5}, {"intent": "x", "value": 0.5}]}
    # a value of a numpy type becomes a Python float; a read intent with the value 0.0 is still listed
    r = ScoresResult(values={"a": np.float64(1.0), "b": 0.0}, remainder=0.0, read=["b", "a"], label="a")
    got = scores_result(r)
    assert got == {"label": "a", "intents": [{"intent": "a", "value": 1.0}, {"intent": "b", "value": 0.0}]}
    assert type(got["intents"][0]["value"]) is float and json.dumps(got)


def test_label_text_mode_reaches_set_labels():
    from anyintent.labels import MODES
    assert LABEL_TEXT_MODES == MODES
    clf = FakeClassifier()
    svc(clf).classify("x", intents=INTENTS)
    assert clf.modes == [None]                         # no mode: set_labels(labels), the model's own mode
    clf2 = FakeClassifier()
    s = svc(clf2, label_text_mode="v2")
    s.classify("x", intents=INTENTS)
    s.classify("y", intents=INTENTS)
    assert clf2.modes == ["v2"] and s.health()["label_text_mode"] == "v2"
    s.label_text_mode = "name"                         # another mode is another label set
    s.classify("x", intents=INTENTS)
    assert clf2.modes == ["v2", "name"]
    with pytest.raises(ValueError, match="label text mode"):
        svc(label_text_mode="plain")


def test_label_text_mode_with_the_library(one_file):
    """With label_text_mode the service equals set_labels(labels, label_text_mode=...) plus classify."""
    clf = AnyIntent.from_pretrained(one_file, device="cpu")
    s = Service(clf, label_text_mode="name")
    got = s.classify(MESSAGES, intents=LABELS)
    clf.set_labels(LABELS, label_text_mode="name")
    want = clf.classify(MESSAGES)
    assert [(g["label"], g["score"], g["margin"]) for g in got] == [(p.label, p.score, p.margin) for p in want]
    assert s.health()["config"]["active"]["mode"] == "name"
    assert set(s.health()["config"]["active"]) == {"w", "k", "tau", "mode"}


def test_runner_up_and_doubtful():
    s = svc()
    r = s.classify(["I lost my bank card", "close my account"], intents=INTENTS)
    assert r[0]["runner_up"] in ("refund status", "close account") and r[0]["runner_up"] != r[0]["label"]
    assert "doubtful" not in r[0]                       # only with a doubt margin
    one = s.classify("x", intents=["only"])[0]
    assert one["margin"] is None and one["runner_up"] is None
    d = svc(doubt_margin=1.5)
    sure, unsure = d.classify(["I lost my bank card stolen", "card refund"], intents=INTENTS)
    assert sure["margin"] >= 1.5 and sure["doubtful"] is False
    assert unsure["margin"] < 1.5 and unsure["doubtful"] is True
    assert d.classify("x", intents=["only"])[0]["doubtful"] is False      # one intent: nothing to doubt between
    assert d.health()["doubt_margin"] == 1.5
    for bad in (0, -1, float("nan"), float("inf"), True, "1"):
        with pytest.raises(ValueError):
            svc(doubt_margin=bad)


def test_runner_up_with_the_library(one_file):
    clf = AnyIntent.from_pretrained(one_file, device="cpu")
    got = Service(clf).classify(MESSAGES, intents=LABELS, k=4)
    clf.set_labels(LABELS)
    for g, p in zip(got, clf.classify(MESSAGES, k=4, return_scores=True)):
        others = sorted((c for c in p.candidates if c["index"] != p.index), key=lambda c: -c["fused"])
        assert g["runner_up"] == others[0]["label"]
        assert abs((p.score - others[0]["fused"]) - p.margin) < 1e-5


def test_lone_surrogates_are_refused(tmp_path):
    s = svc(lists_dir=str(tmp_path / "l"))
    for call in (lambda: s.classify("a\ud800b", intents=INTENTS),
                 lambda: s.classify(["ok", "\udfff"], intents=INTENTS),
                 lambda: s.classify("x", intents=["a\ud800"]),
                 lambda: s.save_list("sur", ["a\ud800b"]),
                 lambda: s.save_list("sur", [{"name": "a", "description": "d\ud800"}])):
        with pytest.raises(InvalidRequest, match="not valid Unicode"):
            call()
    assert s.list_lists()["lists"] == [] and os.listdir(tmp_path / "l") == []


def test_store_size_cap(tmp_path):
    lim = Limits(max_store_bytes=300)
    s = svc(lists_dir=str(tmp_path / "l"), limits=lim)
    s.save_list("a", ["x" * 100])
    s.save_list("a", ["y" * 150])                        # replacing counts the new size only
    with pytest.raises(LimitExceeded, match="300"):
        s.save_list("b", ["z" * 200])
    s.delete_list("a")
    s.save_list("b", ["z" * 200])
    small = svc(lists_dir=str(tmp_path / "l"), limits=Limits(max_store_bytes=50))      # read again under a lower cap
    assert small.list_lists()["lists"] == []


def test_lists_dir_tilde_and_local_model_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))   # on Windows, os.path.expanduser reads USERPROFILE, not HOME
    s = svc(lists_dir="~/anyintent-lists")
    assert s.lists.directory == str(tmp_path / "anyintent-lists") and (tmp_path / "anyintent-lists").is_dir()
    folder = tmp_path / "private" / "models" / "r2intent"
    folder.mkdir(parents=True)
    h = Service(FakeClassifier(), model_id=str(folder)).health()
    assert h["model"] == "r2intent (local folder)" and str(tmp_path) not in json.dumps(h)
    assert Service(FakeClassifier(), model_id="lukashm/r2intent-v1").health()["model"] == "lukashm/r2intent-v1"


def test_load_errors_shown_to_clients(caplog):
    from anyintent import AccessError
    caplog.set_level(logging.DEBUG)

    def gated():
        raise AccessError("No access to the model repository 'm'. It is gated: accept its terms")

    def broken():
        raise OSError("/home/someone/private/models/x: " + SECRET)

    a = Service(loader=gated, load_wait=5)
    a.start_loading()
    with pytest.raises(NotReady, match="It is gated"):     # the library's own message says what to do
        a.classify("x", intents=["a"])
    b = Service(loader=broken, load_wait=5)
    b.start_loading()
    with pytest.raises(NotReady) as e:
        b.classify("x", intents=["a"])
    assert "OSError" in e.value.message and "server log" in e.value.message
    assert SECRET not in e.value.message and "/home/someone" not in json.dumps(b.health())
    assert SECRET in caplog.text                          # the operator's log has the details
