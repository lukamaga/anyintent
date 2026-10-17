"""Retrieval order, score fusion and the cascade, on numpy arrays.

Tie rules are the ones of the DEV scorer, which computed every registered DEV number: retrieval sorts by
cosine descending with ties to the lower label index (stable sort), and the fused argmax takes the first maximum in
retrieval order, computed in float64 from the float32 scores.
"""
import numpy as np


def topk_stable(S, k):
    """S: [n, L] cosines. Returns (cand [n, k] label indices, vals [n, k] cosines), k clipped to L."""
    k = min(int(k), S.shape[1])
    cand = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return cand, np.take_along_axis(S, cand, 1)


def fused_scores(cross, cos, w):
    """cross logit + w x cosine in float64; NaN (not scored) becomes -inf."""
    s = np.asarray(cross, dtype=np.float64) + float(w) * np.asarray(cos, dtype=np.float64)
    return np.where(np.isnan(s), -np.inf, s)


def fused_argmax(cross, cos, w):
    """Position (within the candidate list) of the best fused score; ties: the first in retrieval order."""
    return fused_scores(cross, cos, w).argmax(axis=-1)


def fused_margin(fused):
    """Best minus second-best fused score per row, over the scored candidates (-inf = not scored); NaN when fewer than
    two candidates were scored."""
    f = np.asarray(fused, dtype=np.float64)
    if f.shape[1] < 2:
        return np.full(f.shape[0], np.nan)
    top2 = -np.sort(-f, axis=1)[:, :2]
    out = top2[:, 0] - top2[:, 1]
    return np.where(np.isfinite(out), out, np.nan)


def cascade_stop(cross_first, cos_first, w, tau):
    """First stage of the cascade: True where the fused margin between the best and the second best of the first
    candidates is strictly above tau (one candidate: always True)."""
    f = fused_scores(cross_first, cos_first, w)
    if f.shape[1] < 2:
        return np.ones(f.shape[0], dtype=bool)
    top2 = -np.sort(-f, axis=1)[:, :2]
    return (top2[:, 0] - top2[:, 1]) > float(tau)


def cascade_offline(cross_full, cos_full, w, tau, k, first=3):
    """The cascade computed from full k scores (the reference of the tests and of a DEV tau rule): returns
    (positions of the prediction, stop mask)."""
    k = min(int(k), cross_full.shape[1])
    m = min(int(first), k)
    stop = cascade_stop(cross_full[:, :m], cos_full[:, :m], w, tau) if m < k else np.ones(len(cross_full), dtype=bool)
    p_first = fused_argmax(cross_full[:, :m], cos_full[:, :m], w)
    p_full = fused_argmax(cross_full[:, :k], cos_full[:, :k], w)
    return np.where(stop, p_first, p_full), stop
