"""Few-shot mode against the registered few-shot runtime (tests/reference/fewshot_runtime_ref.py, a copy with the
R2Intent names), CPU fp32 on the tiny one-file model: variants P, K (every aggregation) and KX, N = 2, 5 and 10
examples per label, some labels without examples, k = 3, 5, 8. Candidates, retrieval scores and cross logits within
1e-4, top-1 identical outside fused near ties. Plus the API rules of the mode."""
import json
import os
import shutil

import numpy as np
import pytest

from anyintent import AnyIntent, AnyIntentError, fewshot
from conftest import LABELS, MESSAGES, label_texts_ref
from reference import fewshot_runtime_ref as RF

TOL = 1e-4
NEAR_TIE = 1e-5
NAMES = [l if isinstance(l, str) else l["name"] for l in LABELS]
CONFIGS = {
    "P": {"variant": "P", "agg": None, "a": None, "w": 10.0, "k": 8},
    "K_mean": {"variant": "K", "agg": "mean", "a": 0.5, "w": 10.0, "k": 8},
    "K_lse": {"variant": "K", "agg": "lse", "a": 0.75, "w": 10.0, "k": 8, "lse_tau": 0.05},
    "K_max": {"variant": "K", "agg": "max", "a": 1.0, "w": 10.0, "k": 8},
    "KX": {"variant": "KX", "agg": "lse", "a": 0.5, "w": 10.0, "k": 8, "m_text": 3, "ex_word_cap": 25},
}
EXTRA = ["please help me with this {n} thing", "I have a question about {n}", "{n}?", "Klausimas apie {n}",
         "Pytanie o {n}, proszę", "urgent: {n} \"right now\"", "hello, {n} again", "{n} {n} {n}",
         "can you do {n} for me today", "is {n} possible on sunday"]


def examples_for(n_per_label, without=(14, 15, 16)):
    """Generated examples: N per label, built from the conftest messages and templates; some labels get none."""
    out = []
    for j, name in enumerate(NAMES):
        if j in without:
            out.append([])
            continue
        pool = [MESSAGES[(j + i) % len(MESSAGES)] for i in range(3)] + [t.format(n=name) for t in EXTRA]
        out.append(pool[:n_per_label])
    return out


def fewshot_copy(one_file, tmp_path, block, name="fs"):
    d = tmp_path / name
    if not d.exists():
        shutil.copytree(one_file, d)
    json.dump({"format": "anyintent-one-file", "fewshot": block}, open(d / "anyintent.json", "w", encoding="utf-8"))
    return str(d)


def reference(path, block, examples, msgs, k):
    rt = RF.FewShotRuntime(path, device="cpu")
    rt.set_examples(label_texts_ref(), examples, names=NAMES)
    Q = rt.embed(msgs, int(rt.meta["msg_max_len"])).cpu().numpy().astype(np.float32)

    def cross_fn(pairs):
        if not pairs:
            return np.zeros(0, np.float32)
        return rt.cross([msgs[i] for i, _ in pairs], [t for _, t in pairs]).cpu().numpy()
    return RF.score_batch(Q, rt.Tn, dict(block, k=k), rt.En, rt.ex_label, rt.ex_texts, rt.noex, rt.labels, cross_fn,
                          P=rt.Pn), rt


def compare(res, ref, w, k, tag):
    cand, retr, cross = ref["cand"], ref["retr"], ref["cross"]
    f = cross.astype(np.float64) + w * retr.astype(np.float64)
    s = -np.sort(-f, axis=1)
    near = (s[:, 0] - s[:, 1] < NEAR_TIE) if f.shape[1] > 1 else np.zeros(len(f), bool)
    dr = dx = 0.0
    for i, p in enumerate(res):
        lib = {c["index"]: c for c in p.candidates}
        assert p.n_scored == min(k, len(NAMES))
        for t, j in enumerate(cand[i]):
            j = int(j)
            if j not in lib:      # only at a retrieval near tie at the k-th place
                assert abs(retr[i, t] - retr[i, -1]) < TOL, f"{tag}: candidate set differs on row {i}"
                continue
            dr = max(dr, abs(lib[j]["retrieval"] - float(retr[i, t])))
            dx = max(dx, abs(lib[j]["cross"] - float(cross[i, t])))
            if lib[j]["label_text"] != ref["texts"][i][t]:
                raise AssertionError(f"{tag}: row {i} label text differs: {lib[j]['label_text']!r} vs {ref['texts'][i][t]!r}")
        if not near[i]:
            assert p.index == int(ref["pred"][i]), f"{tag}: top-1 differs on row {i}"
    assert dr < TOL and dx < TOL, f"{tag}: max abs retrieval diff {dr}, cross diff {dx}"
    print(f"EQUIV fewshot {tag}: n {len(res)} near_ties {int(near.sum())} max_abs_retr {dr:.2e} max_abs_cross {dx:.2e}")


@pytest.mark.parametrize("name,n", [("P", 2), ("P", 5), ("P", 10), ("K_mean", 2), ("K_lse", 5), ("K_max", 10),
                                    ("KX", 2), ("KX", 5), ("KX", 10)])
def test_fewshot_vs_runtime(one_file, tmp_path, name, n):
    block = CONFIGS[name]
    path = fewshot_copy(one_file, tmp_path, block)
    ex = examples_for(n)
    clf = AnyIntent.from_pretrained(path, device="cpu", engine="eager")
    assert clf.config["fewshot"]["variant"] == block["variant"]
    clf.set_labels(LABELS, examples=ex)
    assert clf.config["active"]["fewshot"] and clf.config["active"]["n_examples"] == sum(len(x) for x in ex)
    for k in (3, 5, 8):
        ref, _ = reference(path, block, ex, MESSAGES, k)
        res = clf.classify(MESSAGES, k=k, return_scores=True)
        compare(res, ref, block["w"], k, f"{name}/N{n}/k{k}")
    p = clf.classify("I lost my card", return_scores=True)
    assert p.n_scored == block["k"]                # the block's k by default


@pytest.mark.parametrize("name", ["KX", "K_lse", "P"])
def test_fewshot_fast_engine(one_file, tmp_path, name):
    block = CONFIGS[name]
    path = fewshot_copy(one_file, tmp_path, block)
    ex = examples_for(5)
    clf = AnyIntent.from_pretrained(path, device="cpu", engine="fast")
    assert clf.engine == "fast"
    clf.set_labels(LABELS, examples=ex)
    ref, _ = reference(path, block, ex, MESSAGES, 5)
    compare(clf.classify(MESSAGES, k=5, return_scores=True), ref, block["w"], 5, f"fast/{name}/N5/k5")


def test_fewshot_block_from_meta_and_dict_examples(one_file, tmp_path):
    d = tmp_path / "meta_block"
    shutil.copytree(one_file, d)
    meta = json.load(open(d / "r2intent_meta.json", encoding="utf-8"))
    meta["fewshot"] = dict(CONFIGS["K_lse"], msg_max_len=128, label_max_len=192, pair_max_len=160, n_set=[2, 5, 10])
    json.dump(meta, open(d / "r2intent_meta.json", "w", encoding="utf-8"))
    clf = AnyIntent.from_pretrained(str(d), device="cpu", engine="eager")
    assert clf.cfg.fewshot["source"] == "r2intent_meta.json" and clf.cfg.fewshot["n_set"] == [2, 5, 10]
    ex = examples_for(2)
    a = [p.index for p in clf.set_labels(LABELS, examples=ex).classify(MESSAGES)]
    b = [p.index for p in clf.set_labels(LABELS, examples={n: e for n, e in zip(NAMES, ex) if e}).classify(MESSAGES)]
    assert a == b


def test_zero_shot_unchanged_without_examples(one_file, tmp_path):
    path = fewshot_copy(one_file, tmp_path, CONFIGS["KX"])
    plain = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager").set_labels(LABELS)
    clf = AnyIntent.from_pretrained(path, device="cpu", engine="eager")
    clf.set_labels(LABELS, examples=examples_for(5))
    clf.set_labels(LABELS)                          # no examples: zero-shot again
    assert clf.fs is None and not clf.config["active"]["fewshot"]
    assert [p.index for p in clf.classify(MESSAGES)] == [p.index for p in plain.classify(MESSAGES)]


def test_fewshot_rules(one_file, tmp_path):
    with pytest.raises(AnyIntentError, match="few-shot configuration"):
        AnyIntent.from_pretrained(one_file, device="cpu").set_labels(LABELS, examples=examples_for(2))
    path = fewshot_copy(one_file, tmp_path, CONFIGS["K_mean"])
    clf = AnyIntent.from_pretrained(path, device="cpu", engine="eager")
    with pytest.raises(ValueError, match="one list per label"):
        clf.set_labels(LABELS, examples=[["x"]])
    with pytest.raises(TypeError, match="not a single string"):
        clf.set_labels(LABELS, examples=["one string"] * len(LABELS))
    with pytest.raises(ValueError, match="unknown labels"):
        clf.set_labels(LABELS, examples={"no such label": ["x"]})
    clf.set_labels(LABELS, examples=examples_for(2))
    with pytest.raises(AnyIntentError, match="no cascade"):
        clf.classify("x", cascade=True)
    with pytest.raises(AnyIntentError, match="examples= needs labels="):
        clf.predict("x", examples=examples_for(2))
    assert clf.predict("I lost my card", labels=LABELS, examples=examples_for(2)) in NAMES
    for bad, needle in ((dict(CONFIGS["K_mean"], agg="median"), "agg"), (dict(CONFIGS["P"], variant="Z"), "variant Z"),
                        (dict(CONFIGS["P"], variant="Q"), "variant"), (dict(CONFIGS["P"], msg_max_len=64), "msg_max_len")):
        p = fewshot_copy(one_file, tmp_path, bad, name="bad")
        with pytest.raises(AnyIntentError, match=needle):
            AnyIntent.from_pretrained(p, device="cpu", engine="eager")


def test_fewshot_example_vectors_cached(one_file, tmp_path):
    path = fewshot_copy(one_file, tmp_path, CONFIGS["P"])
    cache = str(tmp_path / "cache")
    ex = examples_for(5)
    a = AnyIntent.from_pretrained(path, device="cpu", engine="eager", cache_dir=cache).set_labels(LABELS, examples=ex)
    b = AnyIntent.from_pretrained(path, device="cpu", engine="eager", cache_dir=cache).set_labels(LABELS, examples=ex)
    assert b.cache.hits["disk"] == 2 and np.array_equal(a.fs["E"], b.fs["E"]) and a.label_key == b.label_key
    sides = [json.load(open(os.path.join(cache, f), encoding="utf-8")) for f in os.listdir(cache) if f.endswith(".json")]
    assert sorted(s["fields"]["role"] for s in sides) == ["embed", "examples"]
    c = AnyIntent.from_pretrained(path, device="cpu", engine="eager", cache_dir=cache)
    c.set_labels(LABELS, examples=examples_for(2))
    assert c.label_key != a.label_key


def test_fewshot_module_is_the_runtime_code():
    import inspect
    for n in ("clean_example", "kx_label_text", "example_scores", "prototypes", "mix_scores", "retrieval_scores",
              "topk_stable", "fused_pred", "nearest_examples", "kx_texts_for", "validate_config", "score_batch",
              "noex_label_text", "check_ex_label", "example_counts"):
        assert inspect.getsource(getattr(fewshot, n)) == inspect.getsource(getattr(RF, n)), n
    assert (fewshot.VARIANTS, fewshot.AGGS, fewshot.LSE_TAU, fewshot.M_TEXT, fewshot.EX_WORD_CAP) == \
        (RF.VARIANTS, RF.AGGS, RF.LSE_TAU, RF.M_TEXT, RF.EX_WORD_CAP)
