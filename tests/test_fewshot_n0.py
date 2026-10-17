"""Audit item P11 (c) and F3 (docs/work/stage12/AUDIT_FIXES_FOR_BUILD.md) at the library level: a model with a few-shot
configuration but no example for any label (N = 0) gives exactly the zero-shot pipeline of the registered runtime at the
few-shot block's (w, k): retrieval by the label-text cosine, the cross-encoder over the top k, fused = cross + w x cos;
the cross pass reads the label texts for P and K and the label texts without generated examples for KX. And
set_labels(labels) equals set_labels(labels, examples=None). CPU fp32, tiny random one-file model, both engines;
reference = tests/reference/v11_runtime_ref.py (the registered v1.1 runtime, with the R2Intent names). Written by the few-shot
part (docs/work/extras/fewshot_library.patch)."""
import json
import shutil

import numpy as np
import pytest
import torch

from anyintent import AnyIntent
from conftest import LABELS, MESSAGES, label_texts_ref
from reference.v11_runtime_ref import R2IntentV11

TOL = 1e-4
NEAR_TIE = 1e-5
BLOCKS = {"P": {"variant": "P", "agg": None, "a": None, "w": 10.0, "k": 5},
          "K": {"variant": "K", "agg": "max", "a": 0.5, "w": 10.0, "k": 5},
          "KX": {"variant": "KX", "agg": "lse", "a": 0.75, "w": 15.0, "k": 3}}


def with_block(one_file, tmp_path, block, name):
    d = tmp_path / name
    shutil.copytree(one_file, d)
    json.dump({"format": "anyintent-one-file", "fewshot": block}, open(d / "anyintent.json", "w", encoding="utf-8"))
    return str(d)


@pytest.mark.parametrize("engine", ["eager", "fast"])
@pytest.mark.parametrize("variant", ["P", "K", "KX"])
def test_fewshot_n0_equals_zero_shot_runtime(one_file, tmp_path, engine, variant):
    block = BLOCKS[variant]
    path = with_block(one_file, tmp_path, block, f"n0_{variant}_{engine}")
    clf = AnyIntent.from_pretrained(path, device="cpu", engine=engine)
    clf.set_labels(LABELS, examples=[[] for _ in LABELS])            # few-shot mode, N = 0 for every label
    assert clf.fs is not None and len(clf.fs["ex_texts"]) == 0
    res = clf.classify(MESSAGES, return_scores=True)
    texts = label_texts_ref()
    cross_texts = label_texts_ref("noex") if variant == "KX" else texts
    rt = R2IntentV11(one_file, device="cpu")
    rt.set_labels(texts)
    w, k = block["w"], block["k"]
    n_near, dr, dx = 0, 0.0, 0.0
    for msg, p in zip(MESSAGES, res):
        q = rt.embed([msg], int(rt.meta["msg_max_len"]))
        cos, top = (q @ rt.L.T)[0].topk(min(k, len(texts)))
        c = rt.cross([msg] * len(top), [cross_texts[i] for i in top.tolist()])
        f = (c + w * cos).double()
        s = torch.sort(f, descending=True).values
        near = len(s) > 1 and float(s[0] - s[1]) < NEAR_TIE
        n_near += int(near)
        want = int(top[int(f.argmax())])
        if not near:
            assert p.index == want, f"{variant}/{engine}: N = 0 differs from the zero-shot runtime on {msg[:40]!r}"
        lib = {d["index"]: d for d in p.candidates}
        for t, j in enumerate(top.tolist()):
            if j in lib:
                dr = max(dr, abs(lib[j]["retrieval"] - float(cos[t])))
                dx = max(dx, abs(lib[j]["cross"] - float(c[t])))
                assert lib[j]["label_text"] == cross_texts[j]
    assert dr < TOL and dx < TOL, f"{variant}/{engine}: retrieval diff {dr}, cross diff {dx}"
    print(f"EQUIV fewshot N0 {variant}/{engine}: n {len(res)} near_ties {n_near} max_abs_retrieval {dr:.2e} "
          f"max_abs_cross {dx:.2e}")


def test_examples_none_is_zero_shot(one_file, tmp_path):
    path = with_block(one_file, tmp_path, BLOCKS["K"], "none")
    a = AnyIntent.from_pretrained(path, device="cpu", engine="eager").set_labels(LABELS)
    b = AnyIntent.from_pretrained(path, device="cpu", engine="eager").set_labels(LABELS, examples=None)
    assert a.fs is None and b.fs is None
    ra, rb = a.classify(MESSAGES, return_scores=True), b.classify(MESSAGES, return_scores=True)
    assert [p.index for p in ra] == [p.index for p in rb]
    assert max(abs(x.score - y.score) for x, y in zip(ra, rb)) == 0.0
    plain = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager").set_labels(LABELS)
    rp = plain.classify(MESSAGES, return_scores=True)
    assert [p.index for p in ra] == [p.index for p in rp]
    assert max(abs(x.score - y.score) for x, y in zip(ra, rp)) < TOL
