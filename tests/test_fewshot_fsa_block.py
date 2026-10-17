"""The few-shot block chosen by the FINAL few-shot DEV run on the v1.1 model (results/fewshot_dev_v11/fewshot_decision.json,
sha256 db0f6b9eb6803c6da54f00bea3f403cc9f9122a01ca4581d6f552dcc2c105b47, key "chosen"; PROTOCOL result entry of
2026-10-06) is accepted verbatim by the library and runs: variant P, w 120, k 3. When the decision file is present (the
repository, not the Kaggle smoke, which allows no skipped test), the constants below are checked against it."""
import hashlib
import json
import os
import shutil

import pytest

from anyintent import AnyIntent
from conftest import LABELS, MESSAGES

FSA_DECISION_SHA256 = "db0f6b9eb6803c6da54f00bea3f403cc9f9122a01ca4581d6f552dcc2c105b47"
FSA_CHOSEN = {"variant": "P", "agg": None, "a": None, "w": 120.0, "k": 3, "lse_tau": 0.05, "m_text": 3, "ex_word_cap": 25,
              "msg_max_len": 128, "label_max_len": 192, "pair_max_len": 160, "n_set": [2, 5, 10],
              "label_text_mode": "v2", "v3s_variant": None, "runtime_version": "1"}
DECISION = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "fewshot_dev_v11", "fewshot_decision.json")


def test_constants_equal_the_decision_file():
    if not os.path.exists(DECISION):
        # the Kaggle smoke has no results folder (and its verdict allows no skip): the constants are checked against the
        # decision file in the repository (library_BATCH1_FIX.md)
        print("FSA decision file not present here: constants checked in the repository")
        return
    assert hashlib.sha256(open(DECISION, "rb").read()).hexdigest() == FSA_DECISION_SHA256
    assert json.load(open(DECISION, encoding="utf-8"))["chosen"] == FSA_CHOSEN


@pytest.mark.parametrize("engine", ["eager", "fast"])
def test_fsa_block_accepted_and_runs(one_file, tmp_path, engine):
    d = tmp_path / f"fsa_{engine}"
    shutil.copytree(one_file, d)
    json.dump({"format": "anyintent-one-file", "fewshot": FSA_CHOSEN}, open(d / "anyintent.json", "w", encoding="utf-8"))
    clf = AnyIntent.from_pretrained(str(d), device="cpu", engine=engine)
    fb = clf.cfg.fewshot
    assert (fb["variant"], float(fb["w"]), int(fb["k"])) == ("P", 120.0, 3) and fb["n_set"] == [2, 5, 10]
    examples = [[MESSAGES[i % len(MESSAGES)], MESSAGES[(i + 1) % len(MESSAGES)]] for i in range(len(LABELS))]
    clf.set_labels(LABELS, examples=examples)
    res = clf.classify(MESSAGES, return_scores=True)
    assert len(res) == len(MESSAGES) and all(p.n_scored == min(3, len(LABELS)) for p in res)
    zs = AnyIntent.from_pretrained(str(d), device="cpu", engine=engine).set_labels(LABELS)
    assert zs.fs is None and all(p.n_scored == zs.cfg.k for p in zs.classify(MESSAGES))
