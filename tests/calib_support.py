"""Shared helpers of test_calibration.py and test_classify_identity.py: a tiny one-file model with the settings of
R2Intent v1 (w 60, k 5, label text mode v2, R2Intent names) and calibration files for it (format
r2intent-calibration-1 of the PROTOCOL entry of 2026-10-13 00:06, (10); the parameter values are arbitrary)."""
import copy
import json
import os
import shutil

from anyintent import calibration as calib
from conftest import make_one_file

P_W, P_K = 60.0, 5
CHECK_AB = {"a": 0.3, "b": -18.0}
SCORES_PARAMS = {"temperature": 2.5, "remainder": {"gamma": -1.5, "eta": 0.2, "kappa": 0.4}}
FEWSHOT_P = {"variant": "P", "agg": None, "a": None, "w": P_W, "k": P_K}


def make_p_like(folder, tok, seed=7):
    """the one-file layout of R2Intent v1 at its settings, with the R2Intent names. Returns (folder, folder), the same
    folder twice: callers unpack two values (the first one was once a copy under the earlier file names)"""
    make_one_file(folder, tok, seed=seed, w=P_W, k=P_K, final=True, tau=None, mode="v2")
    return folder, folder


def calibration_doc(model_dir, functions=("check", "scores"), **over):
    """a valid calibration file for the model in model_dir (its model.safetensors sha256, w 60, k 5, v2)"""
    doc = {"format": calib.FORMAT, "r2intent": "R2Intent smoke model", "model": "p_full",
           "model_sha256": calib.weights_sha256(model_dir),
           "fusion": {"w": P_W, "k": P_K, "label_text_mode": "v2"}, "functions": list(functions),
           "dev_record": {"file": "newfx_calibration_smoke.json", "sha256": "0" * 64},
           "protocol": "PROTOCOL 2026-10-13 00:06 (test stand-in)", "license": "cc-by-nc-4.0"}
    if "check" in functions:
        doc["check"] = dict({"method": calib.CHECK_METHOD, "population": "test stand-in"}, **CHECK_AB)
    if "scores" in functions:
        doc["scores"] = dict(copy.deepcopy(SCORES_PARAMS), method=calib.SCORES_METHOD, k=P_K)
    doc.update(over)
    return doc


def write_calibration(model_dir, doc=None, text=None):
    p = os.path.join(model_dir, calib.CALIBRATION_FILE)
    with open(p, "w", encoding="utf-8") as f:
        if text is not None:
            f.write(text)
        else:
            json.dump(calibration_doc(model_dir) if doc is None else doc, f, indent=1)
    return p


def model_copy(src, dst, calibrated=True, functions=("check", "scores")):
    """a copy of a model folder, with a valid calibration file (or none)"""
    shutil.copytree(src, dst)
    if calibrated:
        write_calibration(str(dst), calibration_doc(str(dst), functions=functions))
    return str(dst)
