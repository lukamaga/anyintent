"""The calibration file of a release and the math of AnyIntent.check and AnyIntent.scores (PROTOCOL entry of
2026-10-13 00:06, (1) and (10); design revision 4, sections 1.1, 1.2, 1.5 and 2.6).

r2intent_calibration.json sits at the release root (next to r2intent_meta.json) when at least one function passed its
DEV rule. It is read at the first call of check or scores (a model from the Hugging Face cache fetches only that file,
at the commit it was loaded from, with anyintent.hub.fetch_optional), never at load; a malformed file raises only in
those two functions. Format r2intent-calibration-1:

    {"format": "r2intent-calibration-1", "r2intent": ..., "model": "p_full", "model_sha256": "<64 hex>",
     "fusion": {"w": 60.0, "k": 5, "label_text_mode": "v2"}, "functions": ["check", "scores"],
     "check": {"method": "platt_fused_pair", "a": ..., "b": ..., "population": ...},
     "scores": {"method": "temperature_softmax_top_k_remainder_gap_size", "temperature": ...,
                "remainder": {"gamma": ..., "eta": ..., "kappa": ...}, "k": 5},
     "dev_record": {...}, "protocol": ..., "license": ...}

`functions` lists exactly the blocks present (the functions that passed); there is no threshold in the file.

The block between the markers below is the one source of the math: the newfx tools copy it byte for byte (design 4.2
and 4.5), so it imports nothing from anyintent and raises its own CalibrationValueError (a ValueError).
"""
import json
import os
import re

from .hub import AnyIntentError, file_sha256
from .labels import MODES

CALIBRATION_FILE = "r2intent_calibration.json"
FORMAT = "r2intent-calibration-1"
FUNCTIONS = ("check", "scores")
REMAINDER_KEYS = ("gamma", "eta", "kappa")
WEIGHTS_FILE = "model.safetensors"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


# ---- copied block: the math of check and scores (one source; copied byte for byte by the newfx tools) ----
import math

import numpy as np

K_SCORES = 5   # the frozen k of classify that scores reads with the message
CHECK_METHOD = "platt_fused_pair"
SCORES_METHOD = "temperature_softmax_top_k_remainder_gap_size"


class CalibrationValueError(ValueError):
    """An input the calibration math refuses: a score or a parameter that is not finite, a temperature that is not
    positive, a logit that is not finite in float64, an array of the wrong shape."""


def check_probability(fused, a, b):
    """check (design 1.1): probability = 1 / (1 + exp(-(a s + b))), s the fused pair score cross + w x cosine. Float64;
    the exponential only ever sees a number <= 0, so nothing overflows (an underflow to 0 is the right limit): a huge
    positive or negative a s + b gives 1.0 or 0.0 (an infinite fused score too). A scalar gives a float, an array an array of the same shape. Refused: a or b not
    finite, a fused score that is NaN. (The rule a > 0 belongs to the calibration file and its fit, not to the map.)"""
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise CalibrationValueError(f"check parameters not usable: a {a}, b {b}")
    s = np.asarray(fused, dtype=np.float64)
    if np.isnan(s).any():
        raise CalibrationValueError(f"{int(np.isnan(s).sum())} fused scores are NaN")
    with np.errstate(over="ignore", invalid="ignore"):
        x = a * s + b
    if np.isnan(x).any():
        raise CalibrationValueError(f"a s + b is not defined for {int(np.isnan(x).sum())} scores (a {a}, s infinite)")
    with np.errstate(under="ignore"):
        e = np.exp(-np.abs(x))
        p = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return float(p) if p.ndim == 0 else p


def n_read(n_labels):
    """the number of intents classify reads with the message: min(k, labels)"""
    return np.minimum(K_SCORES, np.asarray(n_labels, dtype=np.int64))


def _fused_rows(F, n_labels):
    """the fused scores of the read intents (float64, the first 5 columns, retrieval order) minus the row's lowest read
    one, the mask of read cells and the rows with a virtual candidate (label sets over 5 intents). An array of fewer
    than 5 columns (the 4 candidates of a set of 4 intents) is padded with NaN when no row reads more intents than it
    has columns. Refused: not a 2-D array, a length that differs from n_labels, a label set without an intent, a row
    that reads more intents than the columns given, a read score that is not finite (cells beyond the read intents are
    ignored and may be NaN)"""
    F = np.asarray(F, dtype=np.float64)
    L = np.asarray(n_labels, dtype=np.int64)
    if F.ndim != 2 or L.ndim != 1 or len(L) != len(F):
        raise CalibrationValueError("fused scores must be (rows, columns) with one label count per row")
    if (L < 1).any():
        raise CalibrationValueError(f"{int((L < 1).sum())} rows with a label set without an intent")
    k = n_read(L)
    if len(F) and F.shape[1] < int(k.max()):
        raise CalibrationValueError(f"{int((k > F.shape[1]).sum())} rows read more intents than the {F.shape[1]} "
                                    "columns given")
    if F.shape[1] < K_SCORES:
        F = np.concatenate([F, np.full((len(F), K_SCORES - F.shape[1]), np.nan)], axis=1)
    F = F[:, :K_SCORES]
    live = np.arange(K_SCORES)[None, :] < k[:, None]
    if not np.isfinite(F[live]).all():
        raise CalibrationValueError(f"{int((~np.isfinite(F[live])).sum())} read fused scores are not finite")
    lo = np.where(live, F, np.inf).min(axis=1)
    X = np.where(live, F - lo[:, None], 0.0)
    return X, live, L > K_SCORES


def _remainder_features(X, live, virt, n_labels):
    """the features of the virtual candidate's logit on rows of label sets over 5 intents: 1, the top gap f_(1) - f_(5)
    and log(L - 5) (0 for a set of 6); zeros on the other rows, which have no virtual candidate"""
    L = np.asarray(n_labels, dtype=np.int64)
    top = np.where(live, X, -np.inf).max(axis=1) if len(X) else np.zeros(0)
    V = np.stack([np.ones(len(X)), top, np.log(np.maximum(L - K_SCORES, 1))], axis=1)
    return np.where(virt[:, None], V, 0.0)


def scores_values(F, n_labels, T, gamma, eta, kappa):
    """scores (design 1.2, revision 4): (values (rows, 5), remainder (rows,)). F: the fused scores of the intents
    classify reads with the message (retrieval order, rows by columns, at least min(5, L) columns); n_labels: the number
    of intents L of each row's label set. On a row of a label set of L > 5 intents: the softmax over the 5 read intents,
    logit (f_j - f_(5)) / T, and a virtual sixth candidate, logit gamma + eta (f_(1) - f_(5)) + kappa log(L - 5), f_(1)
    and f_(5) the highest and the lowest of the 5 fused scores; a read intent's value is its share, the remainder the
    virtual candidate's share. On a row whose every intent is read (5 or fewer): the softmax of f_j / T over them and
    the remainder exactly 0.0. Float64, logits shifted by the row maximum; 0.0 beyond the read intents; the values plus
    the remainder sum to 1. Only gaps between read scores and L enter. With eta = kappa = 0 this is revision 3's
    remainder with delta = gamma T. Refused: T not finite or not positive, gamma, eta or kappa not finite, a logit that
    is not finite in float64 (parameters far outside any fitted range)"""
    T, gamma, eta, kappa = float(T), float(gamma), float(eta), float(kappa)
    if not (math.isfinite(T) and T > 0 and math.isfinite(gamma) and math.isfinite(eta) and math.isfinite(kappa)):
        raise CalibrationValueError(f"scores parameters not usable: T {T}, gamma {gamma}, eta {eta}, kappa {kappa}")
    X, live, virt = _fused_rows(F, n_labels)
    V = _remainder_features(X, live, virt, n_labels)
    Z = np.full((len(X), K_SCORES + 1), -np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        Z[:, :K_SCORES] = np.where(live, X / T, -np.inf)
        Z[:, K_SCORES] = np.where(virt, V @ np.array([gamma, eta, kappa]), -np.inf)
    used = np.concatenate([live, virt[:, None]], axis=1)
    if not np.isfinite(Z[used]).all():
        raise CalibrationValueError(f"{int((~np.isfinite(Z[used])).sum())} logits are not finite in float64 (T {T}, "
                                    f"gamma {gamma}, eta {eta}, kappa {kappa})")
    with np.errstate(under="ignore"):
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        P = E / E.sum(axis=1, keepdims=True)
    return P[:, :K_SCORES], P[:, K_SCORES]
# ---- end of the copied block ----


def _malformed(where, msg):
    return AnyIntentError(f"{where}: malformed calibration file ({FORMAT}): {msg}")


def _number(x):
    """a JSON number (not a bool) that is finite"""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        return math.isfinite(float(x))
    except OverflowError:
        return False


def validate_calibration(doc, where=CALIBRATION_FILE):
    """The checks of design 2.6 that need no model: the format, `functions` not empty (with no function passed no
    file is written) and equal to the blocks present, the method names, finite values, a > 0, T > 0, `remainder` with
    exactly the keys gamma, eta and kappa, each finite, k of `scores` equal to the k of `fusion` and to the k the math
    reads (5), a model sha256 of 64 hex characters. The fusion settings and the model sha256 are compared with the
    loaded model at each call (AnyIntent._calibration_for). Returns the document; raises AnyIntentError naming the file
    and the problem."""
    if not isinstance(doc, dict):
        raise _malformed(where, f"expected a JSON object, got {type(doc).__name__}")
    if doc.get("format") != FORMAT:
        raise _malformed(where, f"format {doc.get('format')!r}, expected {FORMAT!r}")
    sha = doc.get("model_sha256")
    if not (isinstance(sha, str) and _HEX64.fullmatch(sha)):
        raise _malformed(where, f"model_sha256 {sha!r} is not 64 lowercase hex characters")
    fus = doc.get("fusion")
    if not isinstance(fus, dict):
        raise _malformed(where, "no fusion block {w, k, label_text_mode}")
    if not _number(fus.get("w")):
        raise _malformed(where, f"fusion.w {fus.get('w')!r} is not a finite number")
    if isinstance(fus.get("k"), bool) or not isinstance(fus.get("k"), int) or fus["k"] < 1:
        raise _malformed(where, f"fusion.k {fus.get('k')!r} is not a positive integer")
    if fus.get("label_text_mode") not in MODES:
        raise _malformed(where, f"fusion.label_text_mode {fus.get('label_text_mode')!r}, expected one of {MODES}")
    funcs = doc.get("functions")
    if not isinstance(funcs, list) or not all(isinstance(f, str) for f in funcs):
        raise _malformed(where, "functions must be a list of function names")
    if not funcs:
        raise _malformed(where, "functions is empty; the file is written only when at least one function passed its "
                                "DEV rule")
    if len(set(funcs)) != len(funcs):
        raise _malformed(where, f"functions {funcs} lists a function twice")
    unknown = [f for f in funcs if f not in FUNCTIONS]
    if unknown:
        raise _malformed(where, f"functions {unknown} unknown (expected names among {list(FUNCTIONS)})")
    blocks = [f for f in FUNCTIONS if f in doc]
    if sorted(funcs) != sorted(blocks):
        raise _malformed(where, f"functions {funcs} differ from the blocks present {blocks}")
    if "check" in doc:
        c = doc["check"]
        if not isinstance(c, dict):
            raise _malformed(where, "the check block is not an object")
        if c.get("method") != CHECK_METHOD:
            raise _malformed(where, f"check.method {c.get('method')!r}, expected {CHECK_METHOD!r}")
        if not _number(c.get("a")) or not c["a"] > 0:
            raise _malformed(where, f"check.a {c.get('a')!r} is not a finite number > 0")
        if not _number(c.get("b")):
            raise _malformed(where, f"check.b {c.get('b')!r} is not a finite number")
    if "scores" in doc:
        s = doc["scores"]
        if not isinstance(s, dict):
            raise _malformed(where, "the scores block is not an object")
        if s.get("method") != SCORES_METHOD:
            raise _malformed(where, f"scores.method {s.get('method')!r}, expected {SCORES_METHOD!r}")
        if not _number(s.get("temperature")) or not s["temperature"] > 0:
            raise _malformed(where, f"scores.temperature {s.get('temperature')!r} is not a finite number > 0")
        r = s.get("remainder")
        if not isinstance(r, dict) or sorted(r) != sorted(REMAINDER_KEYS):
            raise _malformed(where, f"scores.remainder must hold exactly the keys {list(REMAINDER_KEYS)}, got "
                                    f"{None if not isinstance(r, dict) else sorted(r)}")
        bad = [kk for kk in REMAINDER_KEYS if not _number(r[kk])]
        if bad:
            raise _malformed(where, f"scores.remainder {bad} not finite numbers")
        if isinstance(s.get("k"), bool) or s.get("k") != fus["k"] or s.get("k") != K_SCORES:
            raise _malformed(where, f"scores.k {s.get('k')!r} must equal fusion.k {fus['k']!r} and {K_SCORES} (the k "
                                    "of the method)")
    return doc


def read_calibration(path):
    """Reads and validates a calibration file (validate_calibration); AnyIntentError when it is not readable JSON or
    is malformed."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise AnyIntentError(f"{path}: the calibration file cannot be read as JSON ({type(e).__name__}: "
                             f"{str(e)[:200]})") from e
    return validate_calibration(doc, path)


def weights_sha256(model_dir, name=WEIGHTS_FILE):
    """sha256 of the model's weight file: a Hugging Face cache stores it as a symlink to blobs/<sha256>, so the name
    gives it without reading; otherwise hub.file_sha256 (memoised in the process: a local folder's file was already
    hashed when its revision id was computed)."""
    p = os.path.join(model_dir, name)
    if not os.path.isfile(p):
        raise AnyIntentError(f"{model_dir}: no {name}; the calibration file is bound to the sha256 of the model's "
                             f"{name}")
    if os.path.islink(p):
        target = os.readlink(p)
        blob = os.path.basename(target)
        if os.path.basename(os.path.dirname(target)) == "blobs" and _HEX64.fullmatch(blob):
            return blob
    return file_sha256(p)


def hf_snapshot(root):
    """(repository id, commit) of a Hugging Face cache snapshot folder .../models--<user>--<repo>/snapshots/<commit>,
    else None"""
    parts = os.path.normpath(str(root)).split(os.sep)
    if len(parts) >= 3 and parts[-2] == "snapshots" and _COMMIT.fullmatch(parts[-1]) and \
            parts[-3].startswith("models--") and len(parts[-3]) > len("models--"):
        return parts[-3][len("models--"):].replace("--", "/"), parts[-1]
    return None
