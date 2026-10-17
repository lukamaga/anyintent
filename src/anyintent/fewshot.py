"""Few-shot mode: N real example messages per label given at inference time, nothing trained (PROTOCOL 2026-10-05
23:33 and 23:37). The scoring functions below are a verbatim copy of the numpy part of the registered few-shot runtime
(the code the DEV selection and the final look use; tests/reference/fewshot_runtime_ref.py is a copy of it); the build
of the library checks every block byte for byte. Variants (the model's "fewshot" block names one):
  P  (prototype)       r = cos(q, p_l), p_l = normalise(t_l + sum_i e_li)
  K  (example kNN)     r = (1 - a) cos(q, t_l) + a agg_i cos(q, e_li), agg in max | mean | lse
  KX (K + examples)    K's r; the cross-encoder reads the label text without its generated examples followed by the m
                       examples of that label nearest to the message, each in double quotes
final = cross logit + w x r over the top k by r.
"""
import re

import numpy as np

from .labels import label_text_of, strip_examples   # verbatim copies of the same data-notebook code (checked)

# ---- verbatim copy (the few-shot runtime) ----
FEWSHOT_RUNTIME_VERSION = "1"
VARIANTS = ("Z", "P", "K", "KX")
AGGS = ("max", "mean", "lse")
LSE_TAU = 0.05
M_TEXT = 3
EX_WORD_CAP = 25


def noex_label_text(name, label_text):
    # The label text without its quoted example phrasings: "name: description" -> "name: strip_examples(description)"
    name, label_text = str(name), str(label_text)
    pre = f"{name}: "
    if label_text.startswith(pre):
        d = label_text[len(pre):]
        return label_text_of(name, strip_examples(d, name) if d else "")
    return label_text


def clean_example(text, word_cap=EX_WORD_CAP):
    # One example as it is quoted in a KX label text: whitespace collapsed, inner double quotes made single, at most
    # word_cap words
    t = re.sub(r"\s+", " ", str(text)).strip().replace('"', "'").replace("“", "'").replace("”", "'")
    w = t.split(" ")
    return " ".join(w[:int(word_cap)]) if t else ""


def kx_label_text(noex_text, examples, word_cap=EX_WORD_CAP):
    # name: <description without generated examples> "ex1" "ex2" "ex3" (the v2 description layout, real examples)
    ex = [clean_example(e, word_cap) for e in examples]
    ex = [e for e in ex if e]
    return str(noex_text) + "".join(f' "{e}"' for e in ex)


def check_ex_label(ex_label, n_labels):
    ex_label = np.asarray(ex_label, dtype=np.int64)
    if len(ex_label) and (np.any(np.diff(ex_label) < 0) or ex_label.min() < 0 or ex_label.max() >= n_labels):
        raise ValueError("ex_label must be sorted label indices in [0, n_labels)")
    return ex_label


def example_counts(ex_label, n_labels):
    return np.bincount(check_ex_label(ex_label, n_labels), minlength=n_labels)


def example_scores(sim, ex_label, n_labels, agg, tau=LSE_TAU):
    """sim [n, m] cosines of the messages to every example; returns s_ex [n, L] float64 (NaN for a label without
    examples). agg: max | mean | lse, lse = tau x log(mean_i exp(c_i / tau)) computed stably."""
    if agg not in AGGS:
        raise ValueError(f"agg {agg!r}")
    ex_label = check_ex_label(ex_label, n_labels)
    sim = np.asarray(sim, dtype=np.float64)
    n = sim.shape[0]
    out = np.full((n, n_labels), np.nan, dtype=np.float64)
    cnt = np.bincount(ex_label, minlength=n_labels)
    has = np.flatnonzero(cnt > 0)
    if not len(has) or n == 0:
        return out
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])[has]
    mx = np.maximum.reduceat(sim, starts, axis=1)
    if agg == "max":
        out[:, has] = mx
    elif agg == "mean":
        out[:, has] = np.add.reduceat(sim, starts, axis=1) / cnt[has][None, :]
    else:
        m_full = np.zeros((n, n_labels), dtype=np.float64)
        m_full[:, has] = mx
        z = np.exp((sim - m_full[:, ex_label]) / float(tau))
        out[:, has] = mx + float(tau) * np.log(np.add.reduceat(z, starts, axis=1) / cnt[has][None, :])
    return out


def prototypes(T, E, ex_label):
    """p_l = normalise(t_l + sum of the label's example vectors) (the mean of the label-text vector and its examples,
    normalised); a label without examples keeps t_l. float32 [L, d]."""
    T = np.asarray(T, dtype=np.float32)
    P = T.copy()
    ex_label = check_ex_label(ex_label, T.shape[0])
    if len(ex_label):
        np.add.at(P, ex_label, np.asarray(E, dtype=np.float32))
    nrm = np.linalg.norm(P, axis=1, keepdims=True)
    return (P / np.maximum(nrm, 1e-12)).astype(np.float32)


def mix_scores(cos_t, s_ex, a):
    """K's retrieval score r = (1 - a) cos_t + a s_ex (float64); a label without examples uses s_ex = cos_t."""
    c = np.asarray(cos_t, dtype=np.float64)
    s = np.where(np.isnan(s_ex), c, s_ex)
    return (1.0 - float(a)) * c + float(a) * s


def retrieval_scores(variant, Q, T, E=None, ex_label=None, agg=None, a=None, tau=LSE_TAU, P=None):
    """r [n, L] of a variant for messages Q against the label vectors T and the example vectors E."""
    Q = np.asarray(Q, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    if variant == "Z":
        return Q @ T.T
    if variant == "P":
        if P is None:
            P = prototypes(T, E if E is not None else np.zeros((0, T.shape[1]), np.float32),
                           ex_label if ex_label is not None else np.zeros(0, np.int64))
        return Q @ np.asarray(P, dtype=np.float32).T
    if variant in ("K", "KX"):
        cos_t = Q @ T.T
        if E is None or not len(E):
            return mix_scores(cos_t, np.full(cos_t.shape, np.nan), a)
        sim = Q @ np.asarray(E, dtype=np.float32).T
        return mix_scores(cos_t, example_scores(sim, ex_label, T.shape[0], agg, tau), a)
    raise ValueError(f"variant {variant!r}")


def topk_stable(S, k):
    """Top-k label indices by score (descending; ties to the lower label index, as the v1.1 retrieve) and the values."""
    k = min(int(k), S.shape[1])
    cand = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return cand, np.take_along_axis(S, cand, 1)


def fused_pred(cand, retr, cross, k, w):
    # final score = cross logit + w x r over the first k candidates; ties: retrieval order (the v1.1 fused_pred)
    kk = min(k, cross.shape[1])
    s = cross[:, :kk].astype(np.float64) + float(w) * retr[:, :kk].astype(np.float64)
    s = np.where(np.isnan(s), -np.inf, s)
    return cand[np.arange(len(cand)), s.argmax(1)]


def nearest_examples(sim_row, ex_label, label, m=M_TEXT):
    """Positions (into the example list) of the m examples of `label` nearest to one message, nearest first (ties:
    the earlier example)."""
    ex_label = np.asarray(ex_label)
    pos = np.flatnonzero(ex_label == int(label))
    if not len(pos):
        return pos
    s = np.asarray(sim_row, dtype=np.float64)[pos]
    order = np.argsort(-s, kind="stable")[:int(m)]
    return pos[order]


def kx_texts_for(sim, cand, ex_label, ex_texts, noex_texts, m=M_TEXT, word_cap=EX_WORD_CAP):
    """KX cross texts [n][k] for the candidates of every message (sim [n, m_ex] message-example cosines)."""
    out = []
    for i in range(len(cand)):
        row = []
        for j in cand[i]:
            pos = nearest_examples(sim[i], ex_label, int(j), m)
            row.append(kx_label_text(noex_texts[int(j)], [ex_texts[p] for p in pos], word_cap))
        out.append(row)
    return out


def validate_config(cfg):
    """A few-shot configuration (the `chosen` block of a fewshot decision file)."""
    v = cfg.get("variant")
    if v not in VARIANTS:
        raise ValueError(f"fewshot variant {v!r}")
    if v in ("K", "KX"):
        if cfg.get("agg") not in AGGS:
            raise ValueError(f"fewshot agg {cfg.get('agg')!r}")
        a = float(cfg.get("a"))
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"fewshot a {a}")
    if int(cfg.get("k")) < 1:
        raise ValueError("fewshot k")
    float(cfg.get("w"))
    return {"variant": v, "agg": cfg.get("agg"), "a": None if cfg.get("a") is None else float(cfg["a"]),
            "w": float(cfg["w"]), "k": int(cfg["k"]), "lse_tau": float(cfg.get("lse_tau", LSE_TAU)),
            "m_text": int(cfg.get("m_text", M_TEXT)), "ex_word_cap": int(cfg.get("ex_word_cap", EX_WORD_CAP))}


def score_batch(Q, T, cfg, E=None, ex_label=None, ex_texts=None, noex_texts=None, label_texts=None, cross_fn=None,
                P=None):
    """The whole few-shot decision for a batch of messages from precomputed vectors. cross_fn(list of (i, text)) ->
    logits for message i against that label text. Returns cand [n, k], retr [n, k], cross [n, k], pred [n], texts."""
    c = validate_config(cfg)
    k = min(c["k"], np.asarray(T).shape[0])
    r = retrieval_scores(c["variant"], Q, T, E, ex_label, c["agg"], c["a"], c["lse_tau"], P=P)
    cand, retr = topk_stable(r, k)
    if c["variant"] == "KX":
        sim = np.asarray(Q, dtype=np.float32) @ np.asarray(E, dtype=np.float32).T if E is not None and len(E) else \
            np.zeros((len(cand), 0), np.float32)
        texts = kx_texts_for(sim, cand, ex_label if ex_label is not None else np.zeros(0, np.int64),
                             ex_texts or [], noex_texts, c["m_text"], c["ex_word_cap"])
    else:
        texts = [[label_texts[int(j)] for j in row] for row in cand]
    pairs = [(i, t) for i in range(len(cand)) for t in texts[i]]
    cross = np.asarray(cross_fn(pairs), dtype=np.float32).reshape(len(cand), k) if pairs else np.zeros((0, k), np.float32)
    pred = fused_pred(cand, retr, cross, k, c["w"])
    return {"cand": cand, "retr": retr, "cross": cross, "pred": pred, "texts": texts}
# ---- end of the verbatim copy ----
