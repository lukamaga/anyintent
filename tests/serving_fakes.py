"""Stand-in classifiers for the server tests (no model): the public surface the servers use (set_labels, classify,
config, revision, label_key and the instance lock), with a word-overlap score. FakeWithExtras adds check, scores and the
calibration property in the shapes of AnyIntent.check, AnyIntent.scores and AnyIntent.calibration (CheckResult,
ScoresResult), refused with the library's messages when its stand-in calibration does not name the function;
FakeBroken raises an error that holds the message. calibrated_models builds tiny real models with the settings of
R2Intent v1 with and without a calibration file (tests/calib_support.py). assert_scores_answer checks one scores answer
of the servers against the library's ScoresResult."""
import math
import os
import threading

import numpy as np

from anyintent import AnyIntentError, CheckResult, Prediction, ScoresResult
from anyintent.labels import label_texts, parse_labels

SECRET = "SECRET-MESSAGE-7f3a"


def _words(s):
    return set(str(s).lower().replace(":", " ").replace(".", " ").replace(",", " ").split())


class FakeClassifier:
    revision = "fake-revision-0001"

    def __init__(self):
        self._lock = threading.RLock()
        self.names, self.texts, self.label_key = [], [], None
        self.set_labels_calls = 0
        self.modes = []
        self.classify_calls = []

    @property
    def config(self):
        return {"layout": "one_file", "w": 1.0, "k": 8, "tau": None, "label_text_mode": "v2", "engine": "fake",
                "device": "cpu", "dtype": "float32", "root": "/private/model/folder", "cross_dir": "/private/x",
                "revision": self.revision, "active": {"w": 1.0}}

    def set_labels(self, labels, label_text_mode=None):
        with self._lock:
            names, descs = parse_labels(labels)
            self.names, self.texts = names, label_texts(names, descs, label_text_mode or "v2")
            self.label_key = f"key-{label_text_mode}-" + "|".join(self.texts)
            self.set_labels_calls += 1
            self.modes.append(label_text_mode)
        return self

    def _scores(self, msg):
        w = _words(msg)
        return [len(w & _words(t)) + 0.001 * (len(self.texts) - j) for j, t in enumerate(self.texts)]

    def classify(self, text, k=None, return_scores=False):
        single = isinstance(text, str)
        msgs = [text] if single else list(text)
        with self._lock:
            self.classify_calls.append((list(msgs), k))
            out = []
            for m in msgs:
                sc = self._scores(m)
                order = sorted(range(len(sc)), key=lambda j: -sc[j])[:k or len(sc)]
                j = order[0]
                margin = sc[order[0]] - sc[order[1]] if len(order) > 1 else None
                p = Prediction(label=self.names[j], index=j, score=float(sc[j]), label_text=self.texts[j],
                               n_scored=len(order), margin=margin)
                if return_scores:                 # as the library: retrieval order, not score order
                    p.candidates = [{"index": c, "label": self.names[c], "fused": float(sc[c])}
                                    for c in sorted(order)]
                out.append(p)
        return out[0] if single else out


class FakeWithExtras(FakeClassifier):
    """functions: the functions its stand-in calibration file names (None: no calibration file)."""

    def __init__(self, functions=("check", "scores")):
        super().__init__()
        self.functions = functions

    @property
    def calibration(self):
        with self._lock:
            if self.functions is None:
                return None
            return {"format": "r2intent-calibration-1", "functions": list(self.functions)}

    def _calibration_for(self, kind):
        """the library's refusals of a missing calibration (AnyIntent._calibration_for), the same wording"""
        if self.functions is None:
            raise AnyIntentError(f"{kind}: this model has no calibration file (r2intent_calibration.json in "
                                 "/private/model/folder). check and scores are available only for a release that ships "
                                 "it (the README says which functions a released model supports)")
        if kind not in self.functions:
            raise AnyIntentError(f"{kind} is not available for this model: its calibration file has no '{kind}' block "
                                 f"(functions {list(self.functions)}): the function did not pass its DEV rule and is "
                                 "not in this release")

    def check(self, text, intent, threshold=0.5):
        single = isinstance(text, str)
        msgs = [text] if single else list(text)
        if isinstance(intent, list):
            raise AnyIntentError("check takes one intent; call it once per intent")
        names, descs = parse_labels([intent])
        lt = label_texts(names, descs, "v2")[0]
        out = []
        with self._lock:
            self._calibration_for("check")
            for m in msgs:
                s = float(len(_words(m) & _words(lt)))
                p = float(np.float32(1.0 / (1.0 + math.exp(-(s - 0.5)))))
                out.append(CheckResult(yes=bool(p >= threshold), probability=p, intent=names[0], label_text=lt, score=s,
                                       threshold=float(threshold)))
        return out[0] if single else out

    def scores(self, text):
        single = isinstance(text, str)
        msgs = [text] if single else list(text)
        out = []
        with self._lock:
            self._calibration_for("scores")
            for m in msgs:
                sc = np.array(self._scores(m), dtype=np.float64)
                e = np.exp(sc - sc.max())
                v = e / e.sum()
                values = {n: float(x) for n, x in zip(self.names, v)}
                out.append(ScoresResult(values=values, remainder=0.0, read=list(self.names),
                                        label=self.names[int(np.argmax(v))]))
        return out[0] if single else out


class FakeBroken(FakeClassifier):
    def classify(self, text, k=None, return_scores=False):
        msgs = [text] if isinstance(text, str) else list(text)
        raise RuntimeError(f"simulated failure while reading {msgs!r}")


def assert_scores_answer(got, want):
    """One result of POST /scores or score_intents (got, after JSON or as the tool returns it) against the library's
    ScoresResult for the same call (want): only the keys label and intents (no remainder, no value for every intent);
    intents lists exactly the intents the model read (want.read, at most 5), each as {intent, value}, sorted by value
    from highest to lowest (ties: the label first, then retrieval order), each value exactly want.values[name] (the same
    float); label is want.label and the first one listed; the listed values plus the library's remainder sum to 1
    (with 5 intents or fewer every intent is listed, the remainder is 0 and the values sum to 1)."""
    assert isinstance(got, dict) and set(got) == {"label", "intents"}, got
    items = got["intents"]
    assert isinstance(items, list) and all(isinstance(x, dict) and set(x) == {"intent", "value"} for x in items), got
    names, vals = [x["intent"] for x in items], [x["value"] for x in items]
    assert 1 <= len(items) <= 5 and len(set(names)) == len(names) and sorted(names) == sorted(want.read), (got, want)
    assert all(type(v) is float and v == want.values[n] for n, v in zip(names, vals)), (got, want)
    assert vals == sorted(vals, reverse=True), got
    pos = {n: i for i, n in enumerate(want.read)}
    for (n1, v1), (n2, v2) in zip(zip(names, vals), zip(names[1:], vals[1:])):
        if v1 == v2 and want.label not in (n1, n2):
            assert pos[n1] < pos[n2], got
    assert got["label"] == want.label == names[0], (got, want)
    assert abs(sum(vals) + want.remainder - 1.0) < 1e-9, (got, want)
    if len(want.values) <= 5:
        assert want.remainder == 0.0 and len(items) == len(want.values), (got, want)


INTENTS = [{"name": "lost card", "description": "The customer reports that a bank card was lost or stolen."},
           {"name": "refund status", "description": "The customer asks when a refund will arrive."},
           "close account"]


def calibrated_models(root, tok):
    """Tiny models with the settings of R2Intent v1 (w 60, k 5, label text mode v2, R2Intent names; the same weights):
    'full' with a calibration file naming check and scores, 'check_only' naming check only, 'none' without the file."""
    from calib_support import make_p_like, model_copy
    _, r2 = make_p_like(os.path.join(root, "p_like"), tok)
    return {"full": model_copy(r2, os.path.join(root, "full")),
            "check_only": model_copy(r2, os.path.join(root, "check_only"), functions=("check",)),
            "none": model_copy(r2, os.path.join(root, "none"), calibrated=False)}
