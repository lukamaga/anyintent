"""Numeric equivalence with the registered v1.1 code (verbatim copies in tests/reference, with the R2Intent names), CPU
fp32: top-1 identical and max abs difference below 1e-4 for cosines and cross logits, at k = 3, 5, 8, cascade on and
off, one-file (v1.1 and stage-1.2 FINAL metas) and two-file layouts, English, Lithuanian and Polish, empty and long
messages, batch equal to single."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from anyintent import AnyIntent
from conftest import LABELS, MESSAGES, label_texts_ref
from reference import v11_latency_ref as LR
from reference import v11_lib_ref as R
from reference.v11_runtime_ref import R2IntentV11

TOL = 1e-4
NEAR_TIE = 1e-5     # a reference fused margin below this is reported, not asserted (float order of tiny models)
KS = (3, 5, 8)
REPORT = {}


def dev_scorer_one_file(path, msgs, texts, kmax):
    m, lora, meta = R.load_one_file(path, torch.float32)
    m.eval()
    with R.adapter_on(lora, True):
        Q = R.embed_texts(m, m_tok(path), msgs, meta["msg_max_len"], "cpu", amp=False)
        L = R.embed_texts(m, m_tok(path), texts, meta["label_max_len"], "cpu", amp=False)
    cand, vals, S = R.retrieve(Q, L, kmax)
    with R.adapter_on(lora, False):
        X = R.rerank(m, m_tok(path), msgs, texts, cand, kmax, meta["pair_max_len"], "cpu", amp=False)
    return cand, vals, X, S, meta


_TOKS = {}


def m_tok(path):
    from transformers import AutoTokenizer
    if path not in _TOKS:
        _TOKS[path] = AutoTokenizer.from_pretrained(path)
    return _TOKS[path]


def fused_margin(X, vals, w, k):
    f = X[:, :k].astype(np.float64) + w * vals[:, :k].astype(np.float64)
    s = -np.sort(-f, axis=1)
    return s[:, 0] - s[:, 1] if k > 1 else np.full(len(f), np.inf)


def compare(res, cand, vals, X, S, w, k, tag):
    n = len(res)
    near = fused_margin(X, vals, w, k) < NEAR_TIE
    ref_pred = R.fused_pred(cand, vals, X, k, w)
    lib_pred = np.array([p.index for p in res])
    mism = np.flatnonzero((lib_pred != ref_pred) & ~near)
    assert len(mism) == 0, f"{tag}: top-1 differs on rows {mism.tolist()}"
    dc, dx = 0.0, 0.0
    for i, p in enumerate(res):
        lib = {c["index"]: c for c in p.candidates}
        for t in range(k):
            j = int(cand[i, t])
            if j not in lib:   # only allowed at a cosine near tie at the k-th position
                assert abs(S[i, j] - S[i, cand[i, k - 1]]) < TOL, f"{tag}: candidate set differs on row {i}"
                continue
            dc = max(dc, abs(lib[j]["cosine"] - float(vals[i, t])))
            if lib[j]["cross"] is not None:
                dx = max(dx, abs(lib[j]["cross"] - float(X[i, t])))
    assert dc < TOL and dx < TOL, f"{tag}: max abs cos diff {dc}, cross diff {dx}"
    REPORT[tag] = {"n": n, "near_ties": int(near.sum()), "max_abs_cos": dc, "max_abs_cross": dx}
    print(f"EQUIV {tag}: n {n} near_ties {int(near.sum())} max_abs_cos {dc:.2e} max_abs_cross {dx:.2e}")


@pytest.mark.parametrize("engine", ["eager", "fast"])
@pytest.mark.parametrize("k", KS)
def test_one_file_vs_dev_scorer(one_file, engine, k):
    texts = label_texts_ref()
    cand, vals, X, S, meta = dev_scorer_one_file(one_file, MESSAGES, texts, 8)
    clf = AnyIntent.from_pretrained(one_file, device="cpu", engine=engine)
    assert clf.engine == engine
    clf.set_labels(LABELS)
    assert clf.texts == texts
    res = clf.classify(MESSAGES, k=k, cascade=False, return_scores=True)
    compare(res, cand, vals, X, S, meta["fusion"]["w"], k, f"one_file/{engine}/k{k}")
    spread = float(X.std())
    assert spread > 0.05, f"cross logits of the tiny model are too flat to test anything ({spread})"
    if engine == "fast":
        assert clf.backend.pairs.n_fallback > 0   # the long message went through the tokenizer's truncation


@pytest.mark.parametrize("k", KS)
def test_one_file_vs_runtime(one_file, k):
    texts = label_texts_ref()
    rt = R2IntentV11(one_file, device="cpu")
    rt.set_labels(texts)
    w = rt.meta["fusion"]["w"]
    clf = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager")
    clf.set_labels(LABELS)
    res = clf.classify(MESSAGES, k=k, cascade=False, return_scores=True)
    dc = float(np.abs(rt.L.numpy() - clf.L).max())
    assert dc < TOL, f"label vectors differ from the runtime: {dc}"
    n_near = 0
    for msg, p in zip(MESSAGES, res):
        q = rt.embed([msg], int(rt.meta["msg_max_len"]))
        cos, top = (q @ rt.L.T)[0].topk(min(k, len(texts)))
        c = rt.cross([msg] * len(top), [texts[i] for i in top.tolist()])
        f = (c + w * cos).double()
        s = torch.sort(f, descending=True).values
        near = len(s) > 1 and float(s[0] - s[1]) < NEAR_TIE
        n_near += int(near)
        if not near:
            assert rt.predict(msg, k=k, return_index=True) == p.index, f"runtime differs on {msg[:40]!r}"
        lib = {d["index"]: d for d in p.candidates}
        for t, j in enumerate(top.tolist()):
            if j in lib:
                assert abs(lib[j]["cosine"] - float(cos[t])) < TOL
                assert abs(lib[j]["cross"] - float(c[t])) < TOL
    print(f"EQUIV runtime/k{k}: n {len(MESSAGES)} near_ties {n_near}")


def test_fast_vs_latency_pipe(one_file):
    texts = label_texts_ref()
    m, lora, meta = R.load_one_file(one_file, torch.float32)
    m.eval()
    lora.detach()
    LR.PAIR_MAX_LEN, LR.MSG_MAX_LEN, LR.LABEL_MAX_LEN = meta["pair_max_len"], meta["msg_max_len"], meta["label_max_len"]
    emb_body = LR.FastXLMRLoRA(m.roberta, lora.lora_state(), lora.scale).eval()
    cross_body, head = LR.FastXLMR(m.roberta).eval(), LR.Head(m.classifier).eval()
    eng = LR.Engine("eager_fastxlmr", lambda ids, mm: F.normalize(emb_body(ids, mm).float(), dim=-1),
                    lambda ids, mm: head(cross_body(ids, mm)).float())
    with torch.no_grad():
        pipe = LR.PipeFused(eng, m_tok(one_file), texts, "cpu", w=meta["fusion"]["w"])
        cand_p, sc_p, vals_p = pipe.run(MESSAGES, 8, full=True)
    clf = AnyIntent.from_pretrained(one_file, device="cpu", engine="fast")
    clf.set_labels(LABELS)
    res = clf.classify(MESSAGES, k=8, cascade=False, return_scores=True)
    dx = dc = 0.0
    for i, p in enumerate(res):
        lib = {d["index"]: d for d in p.candidates}
        for t, j in enumerate(cand_p[i].tolist()):
            if j in lib:
                dx = max(dx, abs(lib[j]["cross"] - float(sc_p[i, t])))
                dc = max(dc, abs(lib[j]["cosine"] - float(vals_p[i, t])))
    assert dx < TOL and dc < TOL, (dx, dc)
    print(f"EQUIV latency_pipe: max_abs_cos {dc:.2e} max_abs_cross {dx:.2e} pipe_fallbacks {pipe.n_pair_fallback}")


@pytest.mark.parametrize("engine", ["eager", "fast"])
def test_cascade_vs_offline_reference(one_file, engine):
    texts = label_texts_ref()
    cand, vals, X, S, meta = dev_scorer_one_file(one_file, MESSAGES, texts, 8)
    w = meta["fusion"]["w"]
    f3 = X[:, :3].astype(np.float64) + w * vals[:, :3].astype(np.float64)
    s3 = -np.sort(-f3, axis=1)
    margin = s3[:, 0] - s3[:, 1]
    tau = float(np.median(margin))
    stop = margin > tau
    assert 0 < stop.sum() < len(stop), "both cascade branches must occur"
    want = np.where(stop, R.fused_pred(cand, vals, X, 3, w), R.fused_pred(cand, vals, X, 8, w))
    with pytest.warns(UserWarning):
        clf = AnyIntent.from_pretrained(one_file, device="cpu", engine=engine, tau=tau)
    clf.set_labels(LABELS)
    res = clf.classify(MESSAGES, k=8, cascade=True, return_scores=True)
    ok = np.abs(margin - tau) > NEAR_TIE
    got = np.array([p.index for p in res])
    assert np.array_equal(got[ok], want[ok])
    assert [p.n_scored for p in res] == [3 if s else 8 for s in stop]
    off = clf.classify(MESSAGES, k=8, cascade=False)
    plain = R.fused_pred(cand, vals, X, 8, w)
    clear = fused_margin(X, vals, w, 8) >= NEAR_TIE          # rows without a near tie of the full-k fusion
    assert clear.sum() > 0
    assert np.array_equal(np.array([p.index for p in off])[clear], plain[clear])
    assert all(p.n_scored == 8 for p in off)
    print(f"EQUIV cascade/{engine}: tau {tau:.4f} stopped {int(stop.sum())} of {len(stop)}")


@pytest.mark.parametrize("engine", ["eager", "fast"])
def test_batch_equals_single(one_file, engine):
    clf = AnyIntent.from_pretrained(one_file, device="cpu", engine=engine)
    clf.set_labels(LABELS)
    batch = clf.classify(MESSAGES, k=5, cascade=False, return_scores=True, batch_size=7)
    for msg, b in zip(MESSAGES, batch):
        s = clf.classify(msg, k=5, cascade=False, return_scores=True)
        assert s.index == b.index
        # the equivalence tolerance (fp32 sums depend on the padded batch shape: about 1e-5 seen on Kaggle CPUs)
        assert abs(s.score - b.score) < TOL
        for cs, cb in zip(s.candidates, b.candidates):
            assert cs["index"] == cb["index"] and abs(cs["cosine"] - cb["cosine"]) < TOL


@pytest.mark.parametrize("engine", ["eager", "fast"])
@pytest.mark.parametrize("k", KS)
def test_final_meta_noex(final_file, engine, k):
    texts = label_texts_ref("noex")
    assert texts != label_texts_ref("v2")   # the example phrasings were removed somewhere
    cand, vals, X, S, meta = dev_scorer_one_file(final_file, MESSAGES, texts, 8)
    clf = AnyIntent.from_pretrained(final_file, device="cpu", engine=engine)
    assert clf.cfg.tau == 0.5 and clf.cfg.label_text_mode == "noex" and clf.cfg.version == "1.2"
    clf.set_labels(LABELS)
    assert clf.texts == texts
    res = clf.classify(MESSAGES, k=k, cascade=False, return_scores=True)
    compare(res, cand, vals, X, S, meta["fusion"]["w"], k, f"final/{engine}/k{k}")
    default = clf.classify(MESSAGES)     # frozen k = 8; the cascade is off by default (DECISIONS 2.6, B-2)
    assert all(p.n_scored == min(8, len(LABELS)) for p in default)
    casc = clf.classify(MESSAGES, cascade=True)     # the frozen cascade on request
    assert all(p.n_scored in (3, min(8, len(LABELS))) for p in casc)


def dev_scorer_two_file(root, msgs, texts, kmax):
    emb = R.load_m3_encoder(root + "/embedder", torch.float32).eval()
    cross = R.load_seqcls_exact(root + "/cross", torch.float32).eval()
    tok_e, tok_c = m_tok(root + "/embedder"), m_tok(root + "/cross")
    Q = R.embed_texts(emb, tok_e, msgs, 512, "cpu", amp=False)
    L = R.embed_texts(emb, tok_e, texts, 512, "cpu", amp=False)
    cand, vals, S = R.retrieve(Q, L, kmax)
    X = R.rerank(cross, tok_c, msgs, texts, cand, kmax, 160, "cpu", amp=False)
    return cand, vals, X, S


@pytest.mark.parametrize("engine", ["eager", "fast"])
@pytest.mark.parametrize("k", KS)
def test_two_file_vs_dev_scorer(two_file, engine, k):
    texts = label_texts_ref()
    cand, vals, X, S = dev_scorer_two_file(two_file, MESSAGES, texts, 8)
    clf = AnyIntent.from_pretrained(two_file, device="cpu", engine=engine)
    assert clf.cfg.layout == "two_file" and clf.cfg.w == 5.0 and clf.cfg.msg_max_len == 512
    clf.set_labels(LABELS)
    res = clf.classify(MESSAGES, k=k, cascade=False, return_scores=True)
    compare(res, cand, vals, X, S, 5.0, k, f"two_file/{engine}/k{k}")


def test_two_file_by_arguments(two_file):
    texts = label_texts_ref()
    cand, vals, X, S = dev_scorer_two_file(two_file, MESSAGES, texts, 8)
    clf = AnyIntent.from_pretrained(two_file + "/cross", device="cpu", embedder=two_file + "/embedder", w=5.0)
    assert clf.cfg.layout == "two_file" and clf.cfg.k == 8
    clf.set_labels(LABELS)
    res = clf.classify(MESSAGES, cascade=False, return_scores=True)
    compare(res, cand, vals, X, S, 5.0, 8, "two_file/arguments/k8")


def test_ties_follow_the_dev_scorer(one_file):
    labels = ["same intent: identical text", "other: a different text", "same intent: identical text",
              "third: yet another text"]
    clf = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager")
    with pytest.warns(UserWarning):
        clf.set_labels(labels)
    m, lora, meta = R.load_one_file(one_file, torch.float32)
    with R.adapter_on(lora, True):
        Q = R.embed_texts(m, m_tok(one_file), MESSAGES, 128, "cpu", amp=False)
        L = R.embed_texts(m, m_tok(one_file), labels, 192, "cpu", amp=False)
    cand, vals, S = R.retrieve(Q, L, 4)
    with R.adapter_on(lora, False):
        X = R.rerank(m, m_tok(one_file), MESSAGES, labels, cand, 4, 160, "cpu", amp=False)
    want = R.fused_pred(cand, vals, X, 4, meta["fusion"]["w"])
    got = [p.index for p in clf.classify(MESSAGES, k=4, cascade=False)]
    assert got == want.tolist()


def test_exact_ties_follow_the_dev_scorer():
    # exact ties built by hand (duplicate label texts are not bitwise ties in floating point, so the rule is checked
    # on constructed arrays): retrieval ties go to the lower label index, fusion ties to the first in retrieval order
    from anyintent import fusion
    rng = np.random.default_rng(0)
    S = np.round(rng.random((50, 12)) * 4).astype(np.float32) / 4      # many exact cosine ties
    cand, vals = fusion.topk_stable(S, 8)
    rcand, rvals, _ = R.retrieve(S, np.eye(12, dtype=np.float32), 8)     # Q @ I.T = S
    assert np.array_equal(cand, rcand) and np.array_equal(vals, rvals)
    X = np.round(rng.random((50, 8)) * 3).astype(np.float32)          # many exact fused ties
    for w in (0.0, 1.0, 10.0):
        mine = cand[np.arange(50), fusion.fused_argmax(X, vals, w)]
        assert np.array_equal(mine, R.fused_pred(cand, vals, X, 8, w))
