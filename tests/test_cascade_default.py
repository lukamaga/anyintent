"""DECISIONS 2.6 (B-2): the cascade is off by default (the published numbers are without it); cascade=True uses only a
threshold checked for the k in use (the frozen tau at the frozen k, or a tau_by_k entry), else it stops with an error."""
import json
import shutil

import pytest

from anyintent import AnyIntent, AnyIntentError
from conftest import LABELS, MESSAGES


def _copy(final_file, tmp_path, name, edit=None):
    d = tmp_path / name
    shutil.copytree(final_file, d)
    if edit is not None:
        meta = json.load(open(d / "r2intent_meta.json", encoding="utf-8"))
        edit(meta)
        json.dump(meta, open(d / "r2intent_meta.json", "w", encoding="utf-8"))
    return str(d)


def test_cascade_off_by_default(final_file):
    c = AnyIntent.from_pretrained(final_file, device="cpu", engine="eager").set_labels(LABELS)
    assert c.cfg.tau == 0.5
    k = min(c.cfg.k, len(LABELS))
    default = c.classify(MESSAGES, return_scores=True)
    off = c.classify(MESSAGES, cascade=False, return_scores=True)
    assert all(p.n_scored == k for p in default)
    assert [p.index for p in default] == [p.index for p in off]
    assert max(abs(a.score - b.score) for a, b in zip(default, off)) == 0.0
    on = c.classify(MESSAGES, cascade=True)
    assert all(p.n_scored in (min(c.cfg.cascade_first, k), k) for p in on)


def test_cascade_needs_a_checked_tau_for_the_k(final_file, tmp_path):
    c = AnyIntent.from_pretrained(final_file, device="cpu", engine="eager").set_labels(LABELS)
    with pytest.raises(AnyIntentError, match="no cascade threshold was checked for this k"):
        c.classify("x", k=5, cascade=True)

    def tbk(m):
        m["fusion"]["tau_by_k"] = {"5": 0.25, "8": m["fusion"]["tau"]}

    c2 = AnyIntent.from_pretrained(_copy(final_file, tmp_path, "tbk", tbk), device="cpu", engine="eager")
    c2.set_labels(LABELS)
    assert c2.cfg.tau_by_k == {5: 0.25, 8: 0.5}
    assert c2.cfg.tau_for_k(c2.mode, 5) == 0.25 and c2.cfg.tau_for_k(c2.mode, 8) == 0.5
    assert c2.cfg.tau_for_k(c2.mode, 3) is None
    p = c2.classify(MESSAGES, k=5, cascade=True, return_scores=True)
    assert all(x.n_scored in (min(c2.cfg.cascade_first, 5), 5) for x in p)
    with pytest.raises(AnyIntentError, match=r"checked: k in \[5, 8\]"):
        c2.classify("x", k=3, cascade=True)

    def bad(m):
        m["fusion"]["tau_by_k"] = {"8": 0.1}

    with pytest.raises(AnyIntentError, match="differs from the frozen tau"):
        AnyIntent.from_pretrained(_copy(final_file, tmp_path, "bad", bad), device="cpu", engine="eager")


def test_k_override_takes_the_threshold_of_that_k(final_file, tmp_path):
    """B-2: a k override at load uses tau_by_k[k] (none if not checked), never the frozen k's threshold"""
    def tbk(m):
        m["fusion"]["tau_by_k"] = {"5": 0.25, "8": m["fusion"]["tau"]}

    d = _copy(final_file, tmp_path, "kover", tbk)
    with pytest.warns(UserWarning, match="k overridden to 5"):
        c = AnyIntent.from_pretrained(d, device="cpu", engine="eager", k=5)
    assert c.cfg.k == 5 and c.cfg.tau == 0.25
    with pytest.warns(UserWarning, match="k overridden to 3"):
        c3 = AnyIntent.from_pretrained(d, device="cpu", engine="eager", k=3)
    c3.set_labels(LABELS)
    assert c3.cfg.tau is None
    with pytest.raises(AnyIntentError, match="no frozen tau"):
        c3.classify("x", cascade=True)
