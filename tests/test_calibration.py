"""check and scores (PROTOCOL entry of 2026-10-13 00:06, (1) and (10); design revision 4, 1.1 to 1.5, 2.6 and 4.1):
the math of anyintent.calibration against hand values and at extreme inputs, the copied block, the validation of the
calibration file, the refusals at call time, lazy loading and the Hugging Face fetch of the optional file (mocked), the
model sha256 binding, the warnings, check against the reference runtime's cross logit and cosine (CPU fp32), scores
against classify's own candidates, batch against single, the instance lock. Tiny random models, no network."""
import ast
import dataclasses
import json
import math
import os
import shutil
import threading
import time
import warnings

import numpy as np
import pytest
import torch

import anyintent
import anyintent.hub as hub
from anyintent import AnyIntent, AnyIntentError, CheckResult, ScoresResult
from anyintent import calibration as calib
from calib_support import (CHECK_AB, FEWSHOT_P, P_K, P_W, SCORES_PARAMS, calibration_doc, make_p_like, model_copy,
                           write_calibration)
from conftest import LABELS, MESSAGES, make_one_file
from reference.v11_runtime_ref import R2IntentV11

TOL = 1e-4
COMMIT = "89abcdef0123456789abcdef0123456789abcdef"
NAMES = [l if isinstance(l, str) else l["name"] for l in LABELS]
MAPPED = pytest.mark.skipif(os.name == "nt", reason="Windows refuses to replace a memory-mapped file, and this test "
                            "overwrites the weights of a loaded model")


@pytest.fixture(scope="module")
def p_model(tmp_path_factory, tok):
    return make_p_like(str(tmp_path_factory.mktemp("newfx") / "p_like"), tok)[1]


@pytest.fixture(scope="module")
def p_cal(tmp_path_factory, p_model):
    """the R2Intent copy with a calibration file holding both functions"""
    return model_copy(p_model, tmp_path_factory.mktemp("newfx_cal") / "p_cal")


@pytest.fixture(scope="module")
def clf(p_cal):
    c = AnyIntent.from_pretrained(p_cal, device="cpu")
    c.set_labels(LABELS)
    return c


def sigmoid_ref(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def scores_ref(f, L, T, gamma, eta, kappa):
    """the definition of entry (1) written out with math, one row: (values of the read intents, remainder)"""
    n = min(5, L)
    f = [float(x) for x in f[:n]]
    if L <= 5:
        z = [x / T for x in f]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        return [x / sum(e) for x in e], 0.0
    lo, hi = min(f), max(f)
    z = [(x - lo) / T for x in f] + [gamma + eta * (hi - lo) + kappa * math.log(L - 5)]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    return [x / sum(e) for x in e[:5]], e[5] / sum(e)


# ---- the math ------------------------------------------------------------------------------------------------------
def test_check_probability_hand_values():
    assert calib.check_probability(0.0, 1.0, 0.0) == 0.5
    assert isinstance(calib.check_probability(1.0, 2.0, -1.0), float)
    assert abs(calib.check_probability(1.0, 2.0, -1.0) - 0.7310585786300049) < 1e-15        # 1 / (1 + e^-1)
    assert abs(calib.check_probability(-3.0, 0.5, 0.25) - 0.22270013882530884) < 1e-15      # e^-1.25 / (1 + e^-1.25)
    assert abs(calib.check_probability(60.0, 0.3, -18.0) - 0.5) < 1e-15
    s = np.array([[-2.0, 0.0], [3.5, 61.25]])
    p = calib.check_probability(s, 0.3, -18.0)
    assert p.shape == s.shape and p.dtype == np.float64
    for x, q in zip(s.ravel(), p.ravel()):
        assert abs(q - sigmoid_ref(0.3 * x - 18.0)) < 1e-15
    xs = np.linspace(-40, 40, 801)
    p = calib.check_probability(xs, 1.0, 0.0)
    assert np.all(np.diff(p) >= 0) and np.max(np.abs(p + p[::-1] - 1.0)) < 1e-14


def test_check_probability_extremes_and_refusals():
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        p = calib.check_probability(np.array([1e308, -1e308, np.inf, -np.inf, 800.0, -800.0]), 2.0, 0.5)
    assert p.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    for x in (-30.0, -300.0, -700.0):
        q = calib.check_probability(x, 1.0, 0.0)
        assert q > 0 and abs(q / (math.exp(x) / (1 + math.exp(x))) - 1) < 1e-12, x      # relative accuracy far out
    for args in ((np.nan, 1.0, 0.0), (np.array([0.0, np.nan]), 1.0, 0.0), (0.0, np.inf, 0.0), (0.0, 1.0, np.nan),
                 (np.inf, 0.0, 0.0)):
        with pytest.raises(calib.CalibrationValueError):
            calib.check_probability(*args)
    assert issubclass(calib.CalibrationValueError, ValueError)


def test_scores_values_hand_values():
    l3 = math.log(3.0)
    F = np.array([[l3, 0, 0, 0, 0],      # L 6, T 1, no remainder terms: e = 3, 1, 1, 1, 1 and 1 -> 3/8, 1/8 each
                  [l3, 0, 0, 0, 0],      # L 7, kappa 1: virtual e = L - 5 = 2 -> 3/9, 1/9 each, 2/9
                  [l3, 0, 0, 0, 0]])     # L 6, eta 1: virtual e = e^(f1 - f5) = 3 -> 3/10, 1/10 each, 3/10
    v, r = calib.scores_values(F[:1], [6], 1.0, 0.0, 0.0, 0.0)
    assert np.allclose(v, [[3 / 8] + [1 / 8] * 4], atol=1e-15, rtol=0) and abs(r[0] - 1 / 8) < 1e-15
    v, r = calib.scores_values(F[1:2], [7], 1.0, 0.0, 0.0, 1.0)
    assert np.allclose(v, [[3 / 9] + [1 / 9] * 4], atol=1e-15, rtol=0) and abs(r[0] - 2 / 9) < 1e-15
    v, r = calib.scores_values(F[2:], [6], 1.0, 0.0, 1.0, 0.0)
    assert np.allclose(v, [[3 / 10] + [1 / 10] * 4], atol=1e-15, rtol=0) and abs(r[0] - 3 / 10) < 1e-15
    # every intent read (L <= 5): the softmax of f / T, remainder exactly 0, nothing beyond the read intents
    v, r = calib.scores_values(np.array([[2 * l3, 0.0, 0.0]]), [3], 2.0, 5.0, 5.0, 5.0)
    assert np.allclose(v[0, :3], [3 / 5, 1 / 5, 1 / 5], atol=1e-15, rtol=0) and v[0, 3:].tolist() == [0.0, 0.0]
    assert r.tolist() == [0.0]
    # the full formula with T, gamma, eta and kappa against the written-out definition; a set of 4 (4 columns)
    rows = [([3.0, 1.0, 2.0, 0.0, 0.5], 6), ([3.0, 1.0, 2.0, 0.0, 0.5], 10), ([7.5, -2.0, 0.25, 4.0, 1.0], 63),
            ([3.0, 1.0, 2.0, 0.0, 0.5], 5), ([1.0, -1.0, 0.5, 2.0, np.nan], 4)]
    for f, L in rows:
        v, r = calib.scores_values(np.array([f]), [L], 2.0, 0.1, 0.2, 0.3)
        want_v, want_r = scores_ref(f, L, 2.0, 0.1, 0.2, 0.3)
        assert np.allclose(v[0, :len(want_v)], want_v, atol=1e-15, rtol=0) and abs(r[0] - want_r) < 1e-15, (f, L)
    v4, r4 = calib.scores_values(np.array([[1.0, -1.0, 0.5, 2.0]]), [4], 2.0, 0.1, 0.2, 0.3)   # 4 columns, padded
    assert np.array_equal(v4, v) and r4.tolist() == [0.0] and v4[0, 4] == 0.0
    # revision 3 as the restriction eta = kappa = 0: a virtual score f_(5) + delta, delta = gamma T
    f = np.array([[3.0, 1.0, 2.0, 0.0, 0.5]])
    v, r = calib.scores_values(f, [20], 2.0, 0.7, 0.0, 0.0)
    e = np.exp(np.append(f[0] - 0.0, 0.0 + 0.7 * 2.0) / 2.0)
    assert np.allclose(v[0], e[:5] / e.sum(), atol=1e-15, rtol=0) and abs(r[0] - e[5] / e.sum()) < 1e-15


def test_scores_values_properties():
    rng = np.random.default_rng(0)
    n = 4000
    L = rng.integers(1, 80, size=n)
    F = rng.normal(0, 4, size=(n, 5)) + rng.normal(0, 50, size=(n, 1))
    F[np.arange(5)[None, :] >= np.minimum(L, 5)[:, None]] = np.nan          # cells beyond the read intents
    for T, g, e, k in ((0.05, 0.0, 0.0, 0.0), (2.7, -1.2, 0.3, 0.5), (19.9, 3.0, -0.4, -0.8)):
        v, r = calib.scores_values(F, L, T, g, e, k)
        total = v.sum(axis=1) + r
        assert np.max(np.abs(total - 1.0)) < 1e-12
        assert (v >= 0).all() and (v <= 1).all() and (r >= 0).all() and (r <= 1).all()
        assert (r[L <= 5] == 0.0).all() and (r[L > 5] > 0).all()
        beyond = np.arange(5)[None, :] >= np.minimum(L, 5)[:, None]
        assert (v[beyond] == 0.0).all()
        # the top value's position is the fused argmax among the read intents (first maximum)
        Fm = np.where(beyond, -np.inf, F)
        assert np.array_equal(v.argmax(axis=1), Fm.argmax(axis=1))
        # only the gaps enter: a constant added to a row changes nothing beyond rounding
        v2, r2 = calib.scores_values(F + 1234.5, L, T, g, e, k)
        assert np.max(np.abs(v2 - v)) < 1e-9 and np.max(np.abs(r2 - r)) < 1e-9


def test_scores_values_extremes_and_refusals():
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        v, r = calib.scores_values(np.array([[1e6, 0, 0, 0, 0], [-1e9, -1e9 - 3, -1e9 - 1e6, -1e9, -1e9 - 2],
                                             [1e15, 1e15, 1e15, 1e15, 1e15], [5.0, 4.0, 3.0, 2.0, 1.0]]),
                                   [6, 64, 7, 200], 0.05, 0.0, 0.0, 0.0)
    assert np.max(np.abs(v.sum(axis=1) + r - 1)) < 1e-12
    assert v[0, 0] == 1.0 and r[0] == 0.0 and v[0, 1:].tolist() == [0.0] * 4
    assert np.allclose(v[2], 1 / 6, atol=1e-15, rtol=0) and abs(r[2] - 1 / 6) < 1e-15
    v, r = calib.scores_values(np.array([[3.0]]), [1], 2.0, 0.0, 0.0, 0.0)        # one intent: value 1
    assert v[0].tolist() == [1.0, 0, 0, 0, 0] and r.tolist() == [0.0]
    good = np.array([[1.0, 0.5, 0.0, -0.5, -1.0]])
    bad_calls = [
        (np.array([[np.nan, 0.5, 0.0, -0.5, -1.0]]), [6], 1.0, 0, 0, 0),       # a read score not finite
        (np.array([[np.inf, 0.5, 0.0, -0.5, -1.0]]), [6], 1.0, 0, 0, 0),
        (np.array([[1.0, 0.5, 0.0, np.nan, 2.0]]), [4], 1.0, 0, 0, 0),         # the 4th of 4 read intents
        (good, [6], 0.0, 0, 0, 0), (good, [6], -1.0, 0, 0, 0), (good, [6], np.nan, 0, 0, 0),
        (good, [6], np.inf, 0, 0, 0), (good, [6], 1.0, np.inf, 0, 0), (good, [6], 1.0, 0, np.nan, 0),
        (good, [6], 1.0, 0, 0, -np.inf),
        (good, [6], 1e-310, 0, 0, 0),                                             # logits overflow float64
        (good, [6], 1.0, 1e308, 1e308, 0),
        (good, [0], 1.0, 0, 0, 0),                                                # no intent
        (good, [6, 7], 1.0, 0, 0, 0),                                             # one count per row
        (good[0], [6], 1.0, 0, 0, 0),                                             # not 2-D
        (good[:, :4], [6], 1.0, 0, 0, 0),                                         # reads 5, 4 columns given
    ]
    for args in bad_calls:
        with pytest.raises(calib.CalibrationValueError):
            calib.scores_values(*args)
    v, r = calib.scores_values(np.array([[1.0, 0.5, np.nan, np.nan, np.nan]]), [2], 1.0, 0, 0, 0)
    assert v[0, 2:].tolist() == [0.0, 0.0, 0.0] and r[0] == 0.0


def _copied_block():
    with open(calib.__file__, encoding="utf-8") as f:
        src = f.read()
    start = src.index("# ---- copied block:")
    end = src.index("# ---- end of the copied block ----")
    assert src.count("# ---- copied block:") == 1 and start < end
    return src[src.index("\n", start) + 1:end]


def test_copied_block_is_self_contained():
    block = _copied_block()
    tree = ast.parse(block)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            raise AssertionError(f"from-import inside the copied block: {ast.dump(node)}")
        if isinstance(node, ast.Import):
            assert all(a.name in ("math", "numpy") for a in node.names), ast.dump(node)
    assert "anyintent" not in block and "AnyIntentError" not in block
    assert not any(d in block for d in (chr(0x2013), chr(0x2014), chr(0x2015)))
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert {"check_probability", "scores_values", "CalibrationValueError"} <= names
    ns = {}
    exec(compile(block, "copied_block", "exec"), ns)
    rng = np.random.default_rng(1)
    F, L = rng.normal(0, 3, size=(50, 5)), rng.integers(2, 70, size=50)
    a = ns["scores_values"](F, L, 2.0, -0.5, 0.1, 0.2)
    b = calib.scores_values(F, L, 2.0, -0.5, 0.1, 0.2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(ns["check_probability"](F, 0.3, -1.0), calib.check_probability(F, 0.3, -1.0))
    assert ns["K_SCORES"] == calib.K_SCORES == 5 and ns["SCORES_METHOD"] == calib.SCORES_METHOD


# ---- the calibration file ------------------------------------------------------------------------------------------
def test_read_calibration_valid(p_model, tmp_path):
    for funcs in (("check", "scores"), ("scores", "check"), ("check",), ("scores",)):
        doc = calibration_doc(p_model, functions=funcs)
        p = tmp_path / "c.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        got = calib.read_calibration(str(p))
        assert got == doc and sorted(got["functions"]) == sorted(funcs)
    assert calib.CALIBRATION_FILE in hub.OPTIONAL_FILES and calib.CALIBRATION_FILE not in hub.MODEL_FILES
    assert calib.CALIBRATION_FILE not in hub.FINGERPRINT_FILES


def _mut(path, value):
    def f(d):
        cur = d
        for key in path[:-1]:
            cur = cur[key]
        if value is KeyError:
            del cur[path[-1]]
        else:
            cur[path[-1]] = value
    return f


MALFORMED = {
    "format": _mut(["format"], "r2intent-calibration-2"),
    "format_missing": _mut(["format"], KeyError),
    "sha_short": _mut(["model_sha256"], "ab" * 10),
    "sha_upper": _mut(["model_sha256"], "AB" * 32),
    "fusion_missing": _mut(["fusion"], KeyError),
    "w_string": _mut(["fusion", "w"], "60"),
    "w_bool": _mut(["fusion", "w"], True),
    "w_nan": _mut(["fusion", "w"], float("nan")),
    "k_zero": _mut(["fusion", "k"], 0),
    "k_float": _mut(["fusion", "k"], 5.0),
    "mode_unknown": _mut(["fusion", "label_text_mode"], "v9"),
    "functions_missing": _mut(["functions"], KeyError),
    "functions_string": _mut(["functions"], "check"),
    "functions_empty": lambda d: (d.update(functions=[]), d.pop("check"), d.pop("scores")),    # no function passed
    "functions_twice": _mut(["functions"], ["check", "check", "scores"]),
    "functions_unknown": _mut(["functions"], ["check", "scores", "classify"]),
    "block_not_listed": _mut(["functions"], ["check"]),
    "listed_without_block": lambda d: d.pop("scores"),
    "check_not_object": _mut(["check"], [0.3, -18.0]),
    "check_method": _mut(["check", "method"], "isotonic"),
    "a_zero": _mut(["check", "a"], 0.0),
    "a_negative": _mut(["check", "a"], -0.3),
    "a_nan": _mut(["check", "a"], float("nan")),
    "a_inf": _mut(["check", "a"], float("inf")),
    "a_string": _mut(["check", "a"], "0.3"),
    "a_bool": _mut(["check", "a"], True),
    "a_missing": _mut(["check", "a"], KeyError),
    "b_inf": _mut(["check", "b"], float("-inf")),
    "b_missing": _mut(["check", "b"], KeyError),
    "scores_not_object": _mut(["scores"], None),
    "scores_method": _mut(["scores", "method"], "temperature_softmax_top_k"),
    "scores_method_missing": _mut(["scores", "method"], KeyError),
    "temperature_zero": _mut(["scores", "temperature"], 0.0),
    "temperature_negative": _mut(["scores", "temperature"], -2.5),
    "temperature_nan": _mut(["scores", "temperature"], float("nan")),
    "temperature_missing": _mut(["scores", "temperature"], KeyError),
    "remainder_missing": _mut(["scores", "remainder"], KeyError),
    "remainder_not_object": _mut(["scores", "remainder"], [0.0, 0.0, 0.0]),
    "remainder_key_missing": _mut(["scores", "remainder", "kappa"], KeyError),
    "remainder_key_extra": _mut(["scores", "remainder", "delta"], 0.0),
    "remainder_nan": _mut(["scores", "remainder", "eta"], float("nan")),
    "remainder_inf": _mut(["scores", "remainder", "gamma"], float("inf")),
    "remainder_string": _mut(["scores", "remainder", "gamma"], "0"),
    "remainder_null": _mut(["scores", "remainder", "kappa"], None),
    "scores_k_other": _mut(["scores", "k"], 8),
    "scores_k_missing": _mut(["scores", "k"], KeyError),
    "fusion_k_other": _mut(["fusion", "k"], 8),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_read_calibration_malformed(case, p_model, tmp_path):
    doc = calibration_doc(p_model)
    MALFORMED[case](doc)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(doc), encoding="utf-8")                     # NaN and Infinity are written as JSON's extension literals
    with pytest.raises(AnyIntentError) as e:
        calib.read_calibration(str(p))
    assert "malformed calibration file" in str(e.value) and str(p) in str(e.value)


@pytest.mark.parametrize("text", ["", "{", "[1, 2]", "null", "\"r2intent-calibration-1\"", "\xff\xfe"])
def test_read_calibration_not_json_or_not_an_object(text, tmp_path):
    p = tmp_path / "c.json"
    p.write_bytes(text.encode("latin-1"))
    with pytest.raises(AnyIntentError):
        calib.read_calibration(str(p))


def test_hf_snapshot_paths():
    assert calib.hf_snapshot(f"/x/hub/models--user--r2intent-v1/snapshots/{COMMIT}") == ("user/r2intent-v1", COMMIT)
    assert calib.hf_snapshot(f"/x/hub/models--gpt2/snapshots/{COMMIT}/") == ("gpt2", COMMIT)
    for p in (f"/x/models--user--r/snapshots/{COMMIT[:-1]}", f"/x/datasets--u--r/snapshots/{COMMIT}",
              f"/x/models--/snapshots/{COMMIT}", "/x/local/model", f"/x/models--u--r/refs/{COMMIT}"):
        assert calib.hf_snapshot(p) is None, p


def test_weights_sha256_from_the_blob_name(p_model, tmp_path, monkeypatch):
    real = calib.weights_sha256(p_model)
    assert real == hub.file_sha256(os.path.join(p_model, "model.safetensors"))
    snap = tmp_path / "models--user--r" / "snapshots" / COMMIT
    blobs = tmp_path / "models--user--r" / "blobs"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    shutil.copy(os.path.join(p_model, "model.safetensors"), blobs / real)
    try:
        os.symlink(os.path.join("..", "..", "blobs", real), snap / "model.safetensors")
    except (OSError, NotImplementedError) as e:
        if os.name != "nt":         # on Linux and macOS a failed symlink is a broken test setup, not a skip
            raise
        pytest.skip(f"os.symlink is not permitted here ({type(e).__name__}); on Windows it needs Developer Mode or "
                    "administrator rights")
    monkeypatch.setattr(calib, "file_sha256", lambda p: (_ for _ in ()).throw(AssertionError("hashed")))
    assert calib.weights_sha256(str(snap)) == real
    with pytest.raises(AnyIntentError):
        calib.weights_sha256(str(tmp_path))


# ---- the functions on a tiny model ---------------------------------------------------------------------------------
def test_exports_and_result_types(clf):
    assert "CheckResult" in anyintent.__all__ and "ScoresResult" in anyintent.__all__
    r = clf.check(MESSAGES[0], LABELS[0])
    s = clf.scores(MESSAGES[0])
    assert isinstance(r, CheckResult) and isinstance(s, ScoresResult)
    assert [f.name for f in dataclasses.fields(CheckResult)] == ["yes", "probability", "intent", "label_text", "score",
                                                                 "threshold"]
    assert [f.name for f in dataclasses.fields(ScoresResult)] == ["values", "remainder", "read", "label"]
    for obj, field in ((r, "yes"), (s, "label")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, None)
    assert type(r.yes) is bool and type(r.probability) is float and type(r.score) is float
    assert r.intent == "order status" and r.label_text == clf.texts[0] and r.threshold == 0.5
    assert type(s.remainder) is float and all(type(x) is float for x in s.values.values())


def test_missing_file_never_breaks_load_or_classify(p_model, tmp_path):
    d = model_copy(p_model, tmp_path / "plain", calibrated=False)
    c = AnyIntent.from_pretrained(d, device="cpu")
    c.set_labels(LABELS)
    p = c.classify(MESSAGES[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert c.calibration is None
    for call in (lambda: c.check(MESSAGES[0], "order status"), lambda: c.scores(MESSAGES[0])):
        with pytest.raises(AnyIntentError) as e:
            call()
        assert "no calibration file" in str(e.value) and calib.CALIBRATION_FILE in str(e.value)
    assert c.classify(MESSAGES[0]) == p


def test_malformed_file_raises_only_at_call_time(p_model, tmp_path):
    d = model_copy(p_model, tmp_path / "malformed", calibrated=False)
    write_calibration(d, text="{\"format\": \"r2intent-calibration-1\"")
    c = AnyIntent.from_pretrained(d, device="cpu")          # load: the file is not read
    c.set_labels(LABELS)
    p = c.classify(MESSAGES[:3])
    with pytest.warns(UserWarning, match="calibration file not usable"):
        assert c.calibration is None
    for call in (lambda: c.check(MESSAGES[0], "order status"), lambda: c.scores(MESSAGES[0])):
        with pytest.raises(AnyIntentError, match="cannot be read as JSON"):
            call()
    doc = calibration_doc(d)
    doc["scores"]["remainder"].pop("eta")
    write_calibration(d, doc)
    with pytest.raises(AnyIntentError, match="malformed calibration file"):
        c.scores(MESSAGES[0])
    write_calibration(d)                                      # a valid file is read at the next call
    assert isinstance(c.check(MESSAGES[0], "order status"), CheckResult)
    assert isinstance(c.scores(MESSAGES[0]), ScoresResult)
    assert c.classify(MESSAGES[:3]) == p


def test_file_is_read_lazily_and_cached(p_model, tmp_path):
    d = model_copy(p_model, tmp_path / "lazy", calibrated=False)
    c = AnyIntent.from_pretrained(d, device="cpu")
    c.set_labels(LABELS)
    assert "_calib_doc" not in c.__dict__
    write_calibration(d)                                      # written after the load
    r = c.check(MESSAGES[0], "order status")
    assert c.__dict__["_calib_doc"]["functions"] == ["check", "scores"]
    os.remove(os.path.join(d, calib.CALIBRATION_FILE))       # parsed once, cached on the instance
    assert c.check(MESSAGES[0], "order status") == r
    cal = c.calibration
    cal["check"]["a"] = 99.0                                  # a copy
    assert c.calibration["check"]["a"] == CHECK_AB["a"] and c.check(MESSAGES[0], "order status") == r


def test_no_block_for_the_function(p_model, tmp_path):
    for funcs, works, refused in ((("check",), "check", "scores"), (("scores",), "scores", "check")):
        d = model_copy(p_model, tmp_path / f"only_{works}", functions=funcs)
        c = AnyIntent.from_pretrained(d, device="cpu")
        c.set_labels(LABELS)
        call = {"check": lambda: c.check(MESSAGES[0], LABELS[0]), "scores": lambda: c.scores(MESSAGES[0])}
        call[works]()
        with pytest.raises(AnyIntentError, match=f"{refused} is not available for this model"):
            call[refused]()


def test_w_override_at_load_refused(p_cal):
    with pytest.warns(UserWarning, match="overriding the frozen"):
        c = AnyIntent.from_pretrained(p_cal, device="cpu", w=10.0)
    c.set_labels(LABELS)
    for call in (lambda: c.check(MESSAGES[0], LABELS[0]), lambda: c.scores(MESSAGES[0])):
        with pytest.raises(AnyIntentError, match="w= override"):
            call()
    c.classify(MESSAGES[0])


def test_k_override_refuses_scores_only(p_cal):
    with pytest.warns(UserWarning, match="overriding the frozen"):
        c = AnyIntent.from_pretrained(p_cal, device="cpu", k=8)
    c.set_labels(LABELS)
    assert isinstance(c.check(MESSAGES[0], LABELS[0]), CheckResult)
    with pytest.raises(AnyIntentError, match="k 8 but scores was calibrated at w 60.0, k 5"):
        c.scores(MESSAGES[0])


def test_label_text_mode_override_at_load_refused(p_cal):
    c = AnyIntent.from_pretrained(p_cal, device="cpu", label_text_mode="name")
    c.set_labels(LABELS)
    with pytest.raises(AnyIntentError, match="label_text_mode= override at load"):
        c.check(MESSAGES[0], LABELS[0])
    with pytest.raises(AnyIntentError, match="label text mode 'name'"):
        c.scores(MESSAGES[0])


def test_label_set_mode_refuses_scores_only(clf):
    c = clf
    try:
        c.set_labels(LABELS, label_text_mode="name")
        assert isinstance(c.check(MESSAGES[0], LABELS[0]), CheckResult)
        with pytest.raises(AnyIntentError, match="call set_labels without label_text_mode="):
            c.scores(MESSAGES[0])
    finally:
        c.set_labels(LABELS)


def test_scores_refusals_of_the_label_set(p_cal, tmp_path):
    c = AnyIntent.from_pretrained(p_cal, device="cpu")
    with pytest.raises(AnyIntentError, match="no labels"):
        c.scores(MESSAGES[0])
    assert isinstance(c.check(MESSAGES[0], LABELS[0]), CheckResult)       # check needs no label set
    with pytest.warns(UserWarning, match="identical label texts"):
        c.set_labels(LABELS + [LABELS[0]])
    with pytest.raises(AnyIntentError, match="duplicate names"):
        c.scores(MESSAGES[0])
    c.set_labels([LABELS[0], {"name": LABELS[0]["name"], "description": "Another text."}] + LABELS[1:5])
    with pytest.raises(AnyIntentError, match="duplicate names"):
        c.scores(MESSAGES[0])
    d = tmp_path / "fewshot"
    shutil.copytree(p_cal, d)
    (d / "anyintent.json").write_text(json.dumps({"format": "anyintent-one-file", "fewshot": FEWSHOT_P}), encoding="utf-8")
    f = AnyIntent.from_pretrained(str(d), device="cpu")
    f.set_labels(LABELS, examples={"order status": ["where is my order", "parcel late"]})
    with pytest.raises(AnyIntentError, match="few-shot"):
        f.scores(MESSAGES[0])
    assert isinstance(f.check(MESSAGES[0], LABELS[0]), CheckResult)
    f.set_labels(LABELS)
    assert isinstance(f.scores(MESSAGES[0]), ScoresResult)


def test_onnx_int8_refused(clf, monkeypatch):
    monkeypatch.setattr(clf.backend, "variant", "int8", raising=False)    # the attribute of the ONNX backend
    for call in (lambda: clf.check(MESSAGES[0], LABELS[0]), lambda: clf.scores(MESSAGES[0])):
        with pytest.raises(AnyIntentError, match="ONNX int8"):
            call()


def test_onnx_int8_refused_real(p_cal, tmp_path):
    pytest.importorskip("onnxruntime")
    from anyintent import onnx as aonnx
    aonnx.export(p_cal, str(tmp_path / "onnx"), quantize_int8=True)
    q = AnyIntent.from_pretrained(p_cal, engine="onnx", onnx_dir=str(tmp_path / "onnx"), onnx_variant="int8")
    q.set_labels(LABELS)
    with pytest.raises(AnyIntentError, match="ONNX int8"):
        q.check(MESSAGES[0], LABELS[0])
    with pytest.raises(AnyIntentError, match="ONNX int8"):
        q.scores(MESSAGES[0])


def test_model_sha256_binding(p_model, tok, tmp_path):
    d = model_copy(p_model, tmp_path / "other_sha", calibrated=False)
    write_calibration(d, calibration_doc(d, model_sha256="0" * 64))
    c = AnyIntent.from_pretrained(d, device="cpu")
    c.set_labels(LABELS)
    for call in (lambda: c.check(MESSAGES[0], LABELS[0]), lambda: c.scores(MESSAGES[0])):
        with pytest.raises(AnyIntentError, match="belongs to another model"):
            call()
    # a re-saved model in the same folder (other weights) no longer matches the file written for the first weights
    d2 = model_copy(p_model, tmp_path / "resaved")
    other = make_one_file(str(tmp_path / "other_weights"), tok, seed=11, w=P_W, k=P_K, final=True, tau=None, mode="v2")
    shutil.copy(os.path.join(other, "model.safetensors"), os.path.join(d2, "model.safetensors"))
    c2 = AnyIntent.from_pretrained(d2, device="cpu")
    c2.set_labels(LABELS)
    with pytest.raises(AnyIntentError, match="belongs to another model"):
        c2.check(MESSAGES[0], LABELS[0])
    write_calibration(d2)                                    # bound to the new weights: accepted by a new instance
    c3 = AnyIntent.from_pretrained(d2, device="cpu")
    c3.set_labels(LABELS)
    assert isinstance(c3.scores(MESSAGES[0]), ScoresResult)


def _fake_cache(tmp_path, src, repo="user/r2intent-test", commit=COMMIT):
    d = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / commit
    shutil.copytree(src, d)
    return str(d)


def _hub_mock(monkeypatch, snap, cal_doc=None, fail=False):
    calls = []

    def fake(repo_id, **kw):
        calls.append((repo_id, kw))
        if kw.get("allow_patterns") == [calib.CALIBRATION_FILE]:
            if fail:
                raise OSError("connection refused")
            if cal_doc is not None:
                write_calibration(snap, cal_doc)
        return snap

    monkeypatch.setattr(hub, "_snapshot_download", fake)
    return calls


def test_hf_fetch_of_the_optional_file_online(p_model, tmp_path, monkeypatch):
    snap = _fake_cache(tmp_path, p_model)
    calls = _hub_mock(monkeypatch, snap, cal_doc=calibration_doc(snap))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    c = AnyIntent.from_pretrained("user/r2intent-test", revision="v1.0", device="cpu")
    assert len(calls) == 1 and calls[0][1]["allow_patterns"] == hub.MODEL_FILES     # load: the model files only
    assert not os.path.exists(os.path.join(snap, calib.CALIBRATION_FILE))
    c.set_labels(LABELS)
    assert isinstance(c.check(MESSAGES[0], LABELS[0]), CheckResult)
    assert len(calls) == 2
    repo, kw = calls[1]
    assert repo == "user/r2intent-test" and kw["revision"] == COMMIT and kw["allow_patterns"] == [calib.CALIBRATION_FILE]
    assert isinstance(c.scores(MESSAGES[0]), ScoresResult) and c.calibration is not None
    assert len(calls) == 2                                                             # fetched once


@pytest.mark.parametrize("mode", ["offline", "error", "absent"])
def test_hf_fetch_of_the_optional_file_unavailable(mode, p_model, tmp_path, monkeypatch):
    snap = _fake_cache(tmp_path, p_model)
    calls = _hub_mock(monkeypatch, snap, fail=(mode == "error"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1" if mode == "offline" else "0")
    c = AnyIntent.from_pretrained("user/r2intent-test", device="cpu")
    c.set_labels(LABELS)
    p = c.classify(MESSAGES[0])
    for call in (lambda: c.check(MESSAGES[0], LABELS[0]), lambda: c.scores(MESSAGES[0])):
        with pytest.raises(AnyIntentError) as e:
            call()
        msg = str(e.value)
        assert f"hf download user/r2intent-test {calib.CALIBRATION_FILE} --revision {COMMIT}" in msg
        assert "anyintent.hub.OPTIONAL_FILES" in msg
    assert len(calls) == (1 if mode == "offline" else 2)        # offline: no fetch; otherwise one attempt
    assert c.calibration is None and c.classify(MESSAGES[0]) == p
    if mode != "offline":
        assert hub.fetch_optional("user/r2intent-test", COMMIT, calib.CALIBRATION_FILE) is None
    with pytest.raises(AnyIntentError):
        hub.fetch_optional("user/r2intent-test", COMMIT, "model.safetensors")
    write_calibration(snap)                                      # after `hf download`: found in the snapshot
    assert isinstance(c.check(MESSAGES[0], LABELS[0]), CheckResult)


def test_intent_forms_and_list_refused(clf):
    lab = LABELS[0]
    a = clf.check(MESSAGES[0], lab)
    b = clf.check(MESSAGES[0], (lab["name"], lab["description"]))
    assert a == b
    with pytest.raises(AnyIntentError, match="check takes one intent; call it once per intent"):
        clf.check(MESSAGES[0], [lab["name"], lab["description"]])
    with pytest.raises(AnyIntentError, match="check takes one intent"):
        clf.check(MESSAGES[0], [lab])
    with pytest.raises(TypeError, match="intent"):
        clf.check(MESSAGES[0], ("a", "b", "c"))
    with pytest.raises(ValueError, match="intent"):
        clf.check(MESSAGES[0], {"description": "no name"})
    with pytest.raises(TypeError):
        clf.check([MESSAGES[0], 3], lab)
    for thr in (float("nan"), float("inf"), "0.5", True, None):
        with pytest.raises(ValueError, match="threshold"):
            clf.check(MESSAGES[0], lab, threshold=thr)
    assert clf.check([], lab) == []
    assert clf.scores([]) == []
    hi = clf.check(MESSAGES[0], lab, threshold=0.0)
    lo = clf.check(MESSAGES[0], lab, threshold=1.0 + 1e-9)
    assert hi.yes and not lo.yes and hi.probability == lo.probability == a.probability and hi.threshold == 0.0


def test_warnings_once_per_instance(p_cal):
    c = AnyIntent.from_pretrained(p_cal, device="cpu")
    c.set_labels(LABELS[:5])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c.check(MESSAGES[0], LABELS[0])
        c.scores(MESSAGES[0])
    assert not [x for x in w if issubclass(x.category, UserWarning)]
    with pytest.warns(UserWarning, match="calibrated on intents written with a description") as w:
        c.check(MESSAGES[0], "refund request")
    assert len([x for x in w if "check scored the name" in str(x.message)]) == 1    # a current label with a description
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c.check(MESSAGES[0], "refund request")
        c.check(MESSAGES[0], {"name": "lost card"})
    assert not [x for x in w if issubclass(x.category, UserWarning)]                # once per instance each
    c.set_labels(LABELS)                                                              # holds intents without descriptions
    with pytest.warns(UserWarning, match="some intents of the label set have no description"):
        c.scores(MESSAGES[0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c.scores(MESSAGES[1])
    assert not [x for x in w if issubclass(x.category, UserWarning)]


def test_scores_are_the_values_of_classify_candidates(clf):
    T, r = SCORES_PARAMS["temperature"], SCORES_PARAMS["remainder"]
    res = clf.scores(MESSAGES)
    preds = clf.classify(MESSAGES, cascade=False, return_scores=True)
    F = np.array([[c["fused"] for c in p.candidates] for p in preds])
    V, R = calib.scores_values(F, [len(LABELS)] * len(preds), T, r["gamma"], r["eta"], r["kappa"])
    for i, (s, p) in enumerate(zip(res, preds)):
        assert s.label == p.label and s.read == [c["label"] for c in p.candidates] and len(s.read) == P_K
        assert list(s.values) == NAMES
        assert [s.values[c["label"]] for c in p.candidates] == V[i].tolist() and s.remainder == R[i]
        assert all(s.values[n] == 0.0 for n in NAMES if n not in s.read)
        assert abs(sum(s.values.values()) + s.remainder - 1.0) < 1e-9
        assert s.values[s.label] == max(s.values.values()) and 0 < s.remainder < 1
        want_v, want_r = scores_ref([c["fused"] for c in p.candidates], len(LABELS), T, r["gamma"], r["eta"],
                                    r["kappa"])
        assert np.allclose(V[i], want_v, atol=1e-15, rtol=0) and abs(R[i] - want_r) < 1e-15
    one = clf.scores(MESSAGES[3])                                                     # single against batch (CPU)
    assert one.label == res[3].label and max(abs(one.values[n] - res[3].values[n]) for n in NAMES) < 1e-4


@pytest.mark.parametrize("n", [5, 4, 3, 2, 1])
def test_scores_remainder_zero_for_five_intents_or_fewer(p_cal, n):
    c = AnyIntent.from_pretrained(p_cal, device="cpu")
    c.set_labels(LABELS[:n])
    T = SCORES_PARAMS["temperature"]
    for s, p in zip(c.scores(MESSAGES), c.classify(MESSAGES, return_scores=True)):
        assert s.remainder == 0.0 and sorted(s.read) == sorted(NAMES[:n]) and s.label == p.label
        f = np.array([c_["fused"] for c_ in p.candidates]) / T
        e = np.exp(f - f.max())
        assert np.allclose([s.values[c_["label"]] for c_ in p.candidates], e / e.sum(), atol=1e-12, rtol=0)
        assert abs(sum(s.values.values()) - 1.0) < 1e-9


def test_scores_label_is_classify_label_on_many_sets(p_cal):
    c = AnyIntent.from_pretrained(p_cal, device="cpu", engine="eager")
    rng = np.random.default_rng(3)
    for size in (6, 7, 9, 12, 20):
        idx = rng.choice(len(LABELS), size=size, replace=False)
        c.set_labels([LABELS[int(j)] for j in idx])
        for s, p in zip(c.scores(MESSAGES), c.classify(MESSAGES)):
            assert s.label == p.label and abs(sum(s.values.values()) + s.remainder - 1) < 1e-9 and s.remainder > 0


def _ref_runtime(path):
    rt = R2IntentV11(path, device="cpu")
    return rt, int(rt.meta["msg_max_len"]), int(rt.meta["label_max_len"])


@pytest.mark.parametrize("engine", ["eager", "fast"])
def test_check_against_reference_runtime(p_model, tmp_path, engine):
    kdir = model_copy(p_model, tmp_path / "p_ref")                  # the reference runtime reads the same folder
    c = AnyIntent.from_pretrained(kdir, device="cpu", engine=engine)
    assert c.engine == engine
    rt, mlen, llen = _ref_runtime(kdir)
    a, b = CHECK_AB["a"], CHECK_AB["b"]
    dmax = dpmax = 0.0
    for lab in LABELS[:6] + [LABELS[14], LABELS[17]]:
        name = lab if isinstance(lab, str) else lab["name"]
        desc = "" if isinstance(lab, str) else (lab.get("description") or "")
        text = f"{name}: {desc}" if desc else name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = c.check(MESSAGES, lab)
        cos_l, cross_l, fused_l = c._pair_raw(MESSAGES, text)
        q = rt.embed(MESSAGES, mlen)
        v = rt.embed([text], llen)
        cos_r = (q @ v.T)[:, 0].numpy()
        cross_r = rt.cross(MESSAGES, [text] * len(MESSAGES)).numpy()
        assert np.max(np.abs(cos_l - cos_r)) < TOL and np.max(np.abs(cross_l - cross_r)) < TOL
        dmax = max(dmax, float(np.max(np.abs(cos_l - cos_r))), float(np.max(np.abs(cross_l - cross_r))))
        fused_r = cross_r.astype(np.float64) + P_W * cos_r.astype(np.float64)
        p_r = calib.check_probability(fused_r, a, b)
        for i, r in enumerate(res):
            assert r.label_text == text and r.score == float(fused_l[i])
            assert r.score == float(np.float64(cross_l[i]) + P_W * np.float64(cos_l[i]))
            assert r.probability == calib.check_probability(r.score, a, b) and r.yes == (r.probability >= 0.5)
            assert abs(r.probability - p_r[i]) <= a / 4 * (1 + P_W) * TOL
            dpmax = max(dpmax, abs(r.probability - p_r[i]))
    print(f"CHECK vs reference {engine}: max abs cos/cross diff {dmax:.2e}, max abs probability diff {dpmax:.2e}")


def test_onnx_fp32_check_against_torch(p_cal, tmp_path):
    pytest.importorskip("onnxruntime")
    from anyintent import onnx as aonnx
    aonnx.export(p_cal, str(tmp_path / "onnx"))
    t = AnyIntent.from_pretrained(p_cal, device="cpu", engine="eager")
    o = AnyIntent.from_pretrained(p_cal, engine="onnx", onnx_dir=str(tmp_path / "onnx"))
    for lab in LABELS[:4]:
        a, b = t._pair_raw(MESSAGES, f"{lab['name']}: {lab['description']}"), \
            o._pair_raw(MESSAGES, f"{lab['name']}: {lab['description']}")
        assert np.max(np.abs(a[0] - b[0])) < TOL and np.max(np.abs(a[1] - b[1])) < TOL
        pa, pb = t.check(MESSAGES, lab), o.check(MESSAGES, lab)
        assert max(abs(x.probability - y.probability) for x, y in zip(pa, pb)) <= CHECK_AB["a"] / 4 * (1 + P_W) * TOL


def test_check_orders_classify_candidates_as_fused_scores(clf):
    preds = clf.classify(MESSAGES, cascade=False, return_scores=True)
    by_name = {n: l for n, l in zip(NAMES, LABELS)}
    for p, msg in zip(preds, MESSAGES):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs = [clf.check(msg, by_name[c["label"]]) for c in p.candidates]
        for c, r in zip(p.candidates, rs):
            assert abs(r.score - c["fused"]) < (1 + P_W) * 1e-5           # the same pair; batch effects only
        f = np.array([c["fused"] for c in p.candidates])
        pr = np.array([r.probability for r in rs])
        for i in range(len(f)):
            for j in range(len(f)):
                if f[i] - f[j] > (1 + P_W) * 2e-5:
                    assert pr[i] >= pr[j]


def test_batch_equals_single(clf):
    lab = LABELS[7]
    batch = clf.check(MESSAGES, lab, batch_size=7)
    single = [clf.check(m, lab) for m in MESSAGES]
    for x, y in zip(batch, single):
        assert x.intent == y.intent and x.label_text == y.label_text and x.yes == y.yes
        assert abs(x.probability - y.probability) < 1e-4 and abs(x.score - y.score) < (1 + P_W) * 1e-5
    sb = clf.scores(MESSAGES, batch_size=5)
    ss = [clf.scores(m) for m in MESSAGES]
    for x, y in zip(sb, ss):
        assert x.label == y.label and x.read == y.read and abs(x.remainder - y.remainder) < 1e-4
        assert max(abs(x.values[n] - y.values[n]) for n in NAMES) < 1e-4


def test_two_file_model_check(two_file, tmp_path):
    d = tmp_path / "two"
    shutil.copytree(two_file, d)
    c = AnyIntent.from_pretrained(str(d), device="cpu")
    doc = calibration_doc(c.cfg.cross_dir, functions=("check",))
    doc["fusion"] = {"w": c.cfg.w, "k": c.cfg.k, "label_text_mode": "v2"}
    write_calibration(str(d), doc)                                        # at the release root; weights in cross/
    r = c.check(MESSAGES[:4], LABELS[1])
    assert len(r) == 4 and all(isinstance(x, CheckResult) for x in r)
    c.set_labels(LABELS)
    with pytest.raises(AnyIntentError, match="scores is not available"):
        c.scores(MESSAGES[0])


@pytest.mark.parametrize("fn", ["check", "scores"])
def test_the_instance_lock_is_held(clf, fn):
    done = threading.Event()
    out = {}

    def run():
        try:
            out["r"] = clf.check(MESSAGES[0], LABELS[0]) if fn == "check" else clf.scores(MESSAGES[0])
        finally:
            done.set()

    with clf._lock:
        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.3)
        assert not done.is_set(), f"{fn} ran while another caller held the instance lock"
    t.join(timeout=60)
    assert done.is_set() and "r" in out


# ---- non-finite scores, the binding to the loaded weights, the messages, the lock, no fetch for a local folder -------
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_check_refuses_non_finite_cross_logits(clf, monkeypatch, bad):
    monkeypatch.setattr(clf.backend, "cross_texts", lambda qs, ls: np.full(len(qs), bad, dtype=np.float32))
    for text in (MESSAGES[0], MESSAGES[:3]):
        with pytest.raises(AnyIntentError, match="not finite") as e:
            clf.check(text, LABELS[0])
        assert "no probability is given" in str(e.value)


def test_check_refuses_non_finite_cosines(p_cal, monkeypatch):
    c = AnyIntent.from_pretrained(p_cal, device="cpu")           # its own private intent cache
    real = c.backend.embed
    monkeypatch.setattr(c.backend, "embed", lambda texts, max_len: np.full_like(real(texts, max_len), np.nan))
    with pytest.raises(AnyIntentError, match="the pair score of 2 of 2 messages is not finite"):
        c.check(MESSAGES[:2], LABELS[0])


def test_scores_refuses_non_finite_read_scores(clf, monkeypatch):
    real = clf.classify

    def nan_first(*a, **k):
        out = real(*a, **k)
        (out if isinstance(out, list) else [out])[0].candidates[1]["fused"] = float("nan")
        return out

    monkeypatch.setattr(clf, "classify", nan_first)
    with pytest.raises(AnyIntentError, match="scores: 1 read fused scores are not finite"):
        clf.scores(MESSAGES[:3])


@MAPPED
def test_binding_is_to_the_loaded_weights(p_model, tok, tmp_path):
    d = model_copy(p_model, tmp_path / "swapped", calibrated=False)
    c = AnyIntent.from_pretrained(d, device="cpu")                 # the first weights are loaded
    c.set_labels(LABELS)
    other = make_one_file(str(tmp_path / "other"), tok, seed=11, w=P_W, k=P_K, final=True, tau=None, mode="v2")
    shutil.copy(os.path.join(other, "model.safetensors"), os.path.join(d, "model.safetensors"))
    write_calibration(d)                                           # bound to the weights on disk, not the loaded ones
    assert calib.weights_sha256(d) == calibration_doc(d)["model_sha256"]
    for call in (lambda: c.check(MESSAGES[0], LABELS[0]), lambda: c.scores(MESSAGES[0]),
                 lambda: c.check(MESSAGES[1], LABELS[2])):                # refused at every call, not only the first
        with pytest.raises(AnyIntentError, match="changed after it was loaded"):
            call()
    c2 = AnyIntent.from_pretrained(d, device="cpu")                # loaded again: the file is bound to these weights
    c2.set_labels(LABELS)
    assert isinstance(c2.check(MESSAGES[0], LABELS[0]), CheckResult)
    assert isinstance(c2.scores(MESSAGES[0]), ScoresResult)


def test_w_message_names_an_override_only_when_there_is_one(p_model, tmp_path):
    d = model_copy(p_model, tmp_path / "w10", calibrated=False)
    doc = calibration_doc(d)
    doc["fusion"]["w"] = 10.0                                      # this model's file, another w, no override at load
    write_calibration(d, doc)
    c = AnyIntent.from_pretrained(d, device="cpu")
    c.set_labels(LABELS)
    for call in (lambda: c.check(MESSAGES[0], LABELS[0]), lambda: c.scores(MESSAGES[0])):
        with pytest.raises(AnyIntentError, match="runs at w 60.0 but was calibrated at w 10.0") as e:
            call()
        assert "override" not in str(e.value)
    doc = calibration_doc(d, model_sha256="0" * 64)                # another model's file with another w
    doc["fusion"]["w"] = 10.0
    write_calibration(d, doc)
    c2 = AnyIntent.from_pretrained(d, device="cpu")
    c2.set_labels(LABELS)
    for call in (lambda: c2.check(MESSAGES[0], LABELS[0]), lambda: c2.scores(MESSAGES[0])):
        with pytest.raises(AnyIntentError, match="belongs to another model"):
            call()


def test_hf_message_names_every_reason(p_model, tmp_path, monkeypatch):
    snap = _fake_cache(tmp_path, p_model)
    _hub_mock(monkeypatch, snap)                                   # the release has no such file at that commit
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    c = AnyIntent.from_pretrained("user/r2intent-test", device="cpu")
    with pytest.raises(AnyIntentError) as e:
        c.check(MESSAGES[0], LABELS[0])
    msg = str(e.value)
    for part in ("offline mode", "no network", "HF_TOKEN", "a token= given at load is not used",
                 "a release without this file at that commit", "if the release ships it"):
        assert part in msg, part


@pytest.mark.parametrize("fn", ["check", "scores"])
def test_the_lock_is_held_for_the_whole_call(clf, fn, monkeypatch):
    owned = []
    for obj, name in ((clf, "_calibration_for"), (clf.backend, "embed"), (clf.backend, "cross_texts"),
                      (clf.backend, "cross_pairs"), (calib, "check_probability"), (calib, "scores_values")):
        real = getattr(obj, name, None)
        if real is None:
            continue

        def wrapped(*a, _r=real, _n=name, **k):
            owned.append((_n, clf._lock._is_owned()))
            return _r(*a, **k)

        monkeypatch.setattr(obj, name, wrapped)
    if fn == "check":
        clf.check(MESSAGES[:3], LABELS[0])
        need = {"_calibration_for", "embed", "cross_texts", "check_probability"}
    else:
        clf.scores(MESSAGES[:3])
        need = {"_calibration_for", "scores_values"}
    assert need <= {n for n, _ in owned} and all(o for _, o in owned), owned


def test_local_folder_never_fetches(p_model, tmp_path, monkeypatch):
    d = model_copy(p_model, tmp_path / "plain_nofetch", calibrated=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(hub, "_snapshot_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    c = AnyIntent.from_pretrained(d, device="cpu")
    c.set_labels(LABELS)
    for call in (lambda: c.check(MESSAGES[0], LABELS[0]), lambda: c.scores(MESSAGES[0])):
        with pytest.raises(AnyIntentError, match="no calibration file") as e:
            call()
        assert "hf download" not in str(e.value)
    assert c.calibration is None
