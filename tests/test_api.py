import json
import os
import shutil

import numpy as np
import pytest

from anyintent import AnyIntent, AnyIntentError, Prediction
from anyintent.labels import label_texts, parse_labels
from conftest import LABELS, MESSAGES, label_texts_ref

MAPPED = pytest.mark.skipif(os.name == "nt", reason="Windows refuses to replace a memory-mapped file, and this test "
                            "overwrites the weights of a loaded model")


@pytest.fixture(scope="module")
def clf(one_file):
    c = AnyIntent.from_pretrained(one_file, device="cpu")
    c.set_labels(LABELS)
    return c


def test_engine_auto_picks_fast_on_cpu(clf):
    assert clf.engine == "fast"
    cfg = clf.config
    for key in ("layout", "w", "k", "tau", "msg_max_len", "label_max_len", "pair_max_len", "label_text_mode",
                "revision", "engine", "device", "dtype"):
        assert key in cfg, key
    assert cfg["layout"] == "one_file" and cfg["k"] == 8 and cfg["w"] == 10.0 and cfg["tau"] is None
    assert cfg["revision"].startswith("local-") and cfg["dtype"] == "float32"


def test_single_and_list_types(clf):
    p = clf.classify("where is my parcel")
    assert isinstance(p, Prediction) and isinstance(p.label, str) and 0 <= p.index < len(LABELS)
    assert p.candidates is None and p.n_scored == 8 and str(p) == p.label
    ps = clf.classify(["a", "b"])
    assert isinstance(ps, list) and len(ps) == 2
    assert clf.classify([]) == []
    assert clf.classify_batch(MESSAGES[:3])[0].index == clf.classify(MESSAGES[0]).index
    with pytest.raises(TypeError):
        clf.classify([1, 2])
    with pytest.raises(TypeError):
        clf.classify_batch("one string")


def test_k_values_and_clipping(clf):
    for k in (1, 3, 5, 8, 50):
        p = clf.classify(MESSAGES[0], k=k, return_scores=True)
        assert len(p.candidates) == min(k, len(LABELS))
    with pytest.raises(AnyIntentError):
        clf.classify("x", k=0)


def test_cascade_switch(clf):
    with pytest.raises(AnyIntentError):
        clf.classify("x", cascade=True)          # no frozen tau in the v1.1 meta
    p = clf.classify("x", cascade=False, return_scores=True)
    assert all(c["cross"] is not None for c in p.candidates)


def test_label_inputs_and_modes():
    names, descs = parse_labels(["a", {"name": " b ", "description": " d. "}, ("c", None), {"name": "e"}])
    assert names == ["a", "b", "c", "e"] and descs == ["", "d.", "", ""]
    assert label_texts(names, descs, "v2") == ["a", "b: d.", "c", "e"]
    assert label_texts(names, descs, "name") == names
    for bad in ([], [{"description": "x"}], [""], "abc", [3]):
        with pytest.raises((ValueError, TypeError)):
            parse_labels(bad)
    with pytest.raises(ValueError):
        label_texts(names, descs, "unknown")
    assert label_texts(*parse_labels(LABELS), mode="noex") == label_texts_ref("noex")


def test_names_only_and_mode_override(one_file):
    c = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager")
    c.set_labels([l if isinstance(l, str) else l["name"] for l in LABELS])
    assert c.texts == label_texts_ref("name")
    c.set_labels(LABELS, label_text_mode="name")
    assert c.texts == label_texts_ref("name")
    assert c.predict("where is my parcel") in c.labels
    assert isinstance(c.predict(["a", "b"]), list)


def test_errors(one_file, tmp_path):
    with pytest.raises(AnyIntentError):
        AnyIntent.from_pretrained("")
    c = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager")
    with pytest.raises(AnyIntentError):
        c.classify("no labels yet")
    with pytest.raises(AnyIntentError):
        AnyIntent.from_pretrained(one_file, device="cpu", engine="nonsense")
    bad = tmp_path / "bad"
    shutil.copytree(one_file, bad)
    meta = json.load(open(bad / "r2intent_meta.json", encoding="utf-8"))
    meta["passes"]["cross"]["pair_truncation"] = "only_second"
    json.dump(meta, open(bad / "r2intent_meta.json", "w", encoding="utf-8"))
    with pytest.raises(AnyIntentError):
        AnyIntent.from_pretrained(str(bad), device="cpu")


def test_override_warns(one_file):
    with pytest.warns(UserWarning):
        c = AnyIntent.from_pretrained(one_file, device="cpu", k=5, w=0.0)
    assert c.cfg.k == 5 and c.cfg.w == 0.0 and sorted(c.cfg.overridden) == ["k", "w"]


def test_anyintent_json_overrides_one_file_fusion(one_file, tmp_path):
    d = tmp_path / "pinned_k"
    shutil.copytree(one_file, d)
    json.dump({"format": "anyintent-one-file", "fusion": {"k": 5, "tau": 2.0}}, open(d / "anyintent.json", "w", encoding="utf-8"))
    c = AnyIntent.from_pretrained(str(d), device="cpu")
    assert c.cfg.k == 5 and c.cfg.tau == 2.0 and c.cfg.w == 10.0


def test_cache_memory_and_disk(one_file, tmp_path):
    c = AnyIntent.from_pretrained(one_file, device="cpu", cache_dir=str(tmp_path / "cache"))
    c.set_labels(LABELS)
    L1 = c.L.copy()
    c.set_labels(LABELS[:5])
    c.set_labels(LABELS)
    assert c.cache.hits["memory"] == 1
    files = os.listdir(tmp_path / "cache")
    assert sum(f.endswith(".npy") for f in files) == 2 and sum(f.endswith(".json") for f in files) == 2
    c2 = AnyIntent.from_pretrained(one_file, device="cpu", cache_dir=str(tmp_path / "cache"))
    c2.set_labels(LABELS)
    assert c2.cache.hits["disk"] == 1 and np.array_equal(c2.L, L1)
    c3 = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager", cache_dir=str(tmp_path / "cache"))
    c3.set_labels(LABELS, label_text_mode="noex")   # other texts: other key
    assert c3.cache.hits["miss"] == 1


def test_cache_key_depends_on_revision(one_file, tmp_path):
    d = tmp_path / "other_rev"
    shutil.copytree(one_file, d)
    meta = json.load(open(d / "r2intent_meta.json", encoding="utf-8"))
    meta["run_id"] = "another run"
    json.dump(meta, open(d / "r2intent_meta.json", "w", encoding="utf-8"))
    a = AnyIntent.from_pretrained(one_file, device="cpu")
    b = AnyIntent.from_pretrained(str(d), device="cpu")
    a.set_labels(LABELS)
    b.set_labels(LABELS)
    assert a.revision != b.revision and a.label_key != b.label_key


def test_warm_up_restores_labels(clf):
    before = (list(clf.texts), clf.label_key)
    s = clf.warm_up(batch_sizes=(1, 4))
    assert s > 0 and (list(clf.texts), clf.label_key) == before
    fresh = AnyIntent.from_pretrained(clf.cfg.root, device="cpu")
    fresh.warm_up(batch_sizes=(1,))
    assert fresh.L is None


def test_single_label_and_short_sets(one_file):
    c = AnyIntent.from_pretrained(one_file, device="cpu")
    c.set_labels(["only label"])
    p = c.classify("anything", return_scores=True)
    assert p.index == 0 and p.n_scored == 1 and len(p.candidates) == 1
    c.set_labels(["a", "b"])
    assert c.classify("x", k=8).n_scored == 2


def test_cache_key_depends_on_engine(one_file, tmp_path):
    cache = str(tmp_path / "cache")
    a = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager", cache_dir=cache)
    a.set_labels(LABELS)
    b = AnyIntent.from_pretrained(one_file, device="cpu", engine="fast", cache_dir=cache)
    b.set_labels(LABELS)
    assert b.cache.hits == {"memory": 0, "disk": 0, "miss": 1} and a.label_key != b.label_key


def test_margin(clf):
    for k in (3, 8):
        p = clf.classify("where is my parcel", k=k, cascade=False, return_scores=True)
        fused = sorted((c["fused"] for c in p.candidates), reverse=True)
        assert p.margin is not None and abs(p.margin - (fused[0] - fused[1])) < 1e-9 and p.margin >= 0
    c1 = AnyIntent.from_pretrained(clf.cfg.root, device="cpu", engine="eager")
    c1.set_labels(["only label"])
    assert c1.classify("anything").margin is None
    assert clf.classify("x", k=1).margin is None


def test_margin_with_the_cascade_uses_the_scored_candidates(final_file):
    c = AnyIntent.from_pretrained(final_file, device="cpu", engine="eager")
    c.set_labels(LABELS)
    for p in c.classify(MESSAGES, cascade=True, return_scores=True):     # the cascade is off by default (B-2)
        fused = sorted((x["fused"] for x in p.candidates if x["fused"] is not None), reverse=True)
        assert len(fused) == p.n_scored and abs(p.margin - (fused[0] - fused[1])) < 1e-9
        if p.n_scored == 3:
            assert p.margin > c.cfg.tau


def test_threads_share_one_instance(one_file):
    import threading
    other = [{"name": f"topic {i}", "description": f"messages about subject number {i}"} for i in range(9)]
    c = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager")
    msgs = MESSAGES[:6]
    want = {}
    for name, labels in (("a", LABELS), ("b", other)):
        c.set_labels(labels)
        want[name] = [(p.label, p.score) for p in c.classify(msgs)]
    got, errors = {"a": [], "b": []}, []

    def work(name, labels):
        try:
            for _ in range(20):
                r = c.predict(msgs, labels=labels)
                got[name].append(r)
        except Exception as e:      # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=work, args=("a", LABELS)), threading.Thread(target=work, args=("b", other))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    for name in ("a", "b"):
        assert len(got[name]) == 20 and all(r == [x[0] for x in want[name]] for r in got[name]), name
    # classify under the lock: scores identical to the single-thread ones
    c.set_labels(LABELS)
    again = [(p.label, p.score) for p in c.classify(msgs)]
    assert [x[0] for x in again] == [x[0] for x in want["a"]]
    assert max(abs(x[1] - y[1]) for x, y in zip(again, want["a"])) < 1e-5


def test_two_file_needs_an_explicit_layout(one_file, tmp_path):
    d = tmp_path / "bare_cross"
    shutil.copytree(one_file, d)
    meta = json.load(open(d / "r2intent_meta.json", encoding="utf-8"))
    meta.pop("adapter")
    json.dump(meta, open(d / "r2intent_meta.json", "w", encoding="utf-8"))
    with pytest.raises(AnyIntentError) as e:
        AnyIntent.from_pretrained(str(d), device="cpu")
    assert "anyintent-two-file" in str(e.value) and "embedder=" in str(e.value) and "variant" in str(e.value)


def _final_copy(final_file, tmp_path, edit):
    d = tmp_path / "final_alt"
    shutil.copytree(final_file, d)
    meta = json.load(open(d / "r2intent_meta.json", encoding="utf-8"))
    edit(meta)
    json.dump(meta, open(d / "r2intent_meta.json", "w", encoding="utf-8"))
    return str(d)


def test_alternative_meta_key_names(final_file, tmp_path):
    def cascade_block(m):
        m["cascade"] = {"tau": m["fusion"].pop("tau"), "first": 2}
        m["fusion"]["label_text_mode"] = m.pop("label_text_mode")

    c = AnyIntent.from_pretrained(_final_copy(final_file, tmp_path, cascade_block), device="cpu", engine="eager")
    assert (c.cfg.tau, c.cfg.cascade_first, c.cfg.label_text_mode) == (0.5, 2, "noex")

    def top_level(m):
        m["tau"] = m["fusion"].pop("tau")

    c = AnyIntent.from_pretrained(_final_copy(final_file, tmp_path / "t", top_level), device="cpu", engine="eager")
    assert c.cfg.tau == 0.5 and c.cfg.label_text_mode == "noex"


def test_final_meta_without_tau_or_mode_warns(final_file, tmp_path):
    def strip(m):
        m["fusion"].pop("tau")
        m.pop("label_text_mode")

    with pytest.warns(UserWarning, match="cascade is OFF"):
        c = AnyIntent.from_pretrained(_final_copy(final_file, tmp_path, strip), device="cpu", engine="eager")
    assert c.cfg.tau is None and c.cfg.label_text_mode == "v2"

    def null_tau(m):
        m["fusion"]["tau"] = None       # an explicit null: no cascade on purpose, no warning about tau

    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c = AnyIntent.from_pretrained(_final_copy(final_file, tmp_path / "n", null_tau), device="cpu", engine="eager")
    assert c.cfg.tau is None and not [w for w in rec if "cascade is OFF" in str(w.message)]


def test_v3s_mode_warns_once(final_file, tmp_path):
    def v3s(m):
        m["label_text_mode"] = "v3s"
        m["v3s_variant"] = "A"

    c = AnyIntent.from_pretrained(_final_copy(final_file, tmp_path, v3s), device="cpu", engine="eager")
    assert c.config["v3s_variant"] == "A"
    with pytest.warns(UserWarning, match="v3s layout"):
        c.set_labels(LABELS)
    assert c.texts == label_texts_ref("v2")
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.set_labels(LABELS[:5])
    assert not [w for w in rec if "v3s" in str(w.message)]


def test_fusion_by_mode_switches_the_settings(final_file, tmp_path):
    def v3s(m):
        m["label_text_mode"] = "v3s"
        m["v3s_variant"] = "A"

    d = _final_copy(final_file, tmp_path, v3s)
    json.dump({"format": "anyintent-one-file", "fusion_by_mode": {"v2": {"w": 7.0, "k": 5, "tau": None}}},
              open(os.path.join(d, "anyintent.json"), "w", encoding="utf-8"))
    c = AnyIntent.from_pretrained(d, device="cpu", engine="eager")
    assert c.cfg.fusion_by_mode == {"v2": {"w": 7.0, "k": 5, "tau": None}} and c.cfg.fusion_for("v3s") == (15.0, 8, 0.5)
    with pytest.warns(UserWarning, match="label_text_mode='v2'"):
        c.set_labels(LABELS)
    assert c.config["active"] == {"w": 15.0, "k": 8, "tau": 0.5, "mode": "v3s", "fewshot": False, "n_examples": 0}
    assert c.classify("where is my parcel").n_scored in (3, 8)
    c.set_labels(LABELS, label_text_mode="v2")
    assert c.config["active"]["w"] == 7.0 and c.config["active"]["k"] == 5
    p = c.classify("where is my parcel", return_scores=True)
    assert p.n_scored == 5 and len(p.candidates) == 5          # tau None in v2: no cascade
    f = [x["cross"] + 7.0 * x["cosine"] for x in p.candidates]
    assert abs(max(f) - p.score) < 1e-5
    with pytest.raises(AnyIntentError, match="no frozen tau"):
        c.classify("x", cascade=True)
    json.dump({"format": "anyintent-one-file", "fusion_by_mode": {"v3s": {"w": 1.0, "k": 5}}},
              open(os.path.join(d, "anyintent.json"), "w", encoding="utf-8"))
    with pytest.raises(AnyIntentError, match="differs from the main fusion values"):
        AnyIntent.from_pretrained(d, device="cpu", engine="eager")
    json.dump({"format": "anyintent-one-file", "fusion_by_mode": {"v9": {"w": 1.0, "k": 5}}},
              open(os.path.join(d, "anyintent.json"), "w", encoding="utf-8"))
    with pytest.raises(AnyIntentError, match="unknown label_text_mode"):
        AnyIntent.from_pretrained(d, device="cpu", engine="eager")


def test_one_file_without_w_is_an_error(final_file, tmp_path):
    def no_w(m):
        m["fusion"].pop("w")

    with pytest.raises(AnyIntentError, match="no fusion weight w"):
        AnyIntent.from_pretrained(_final_copy(final_file, tmp_path, no_w), device="cpu")
    c = AnyIntent.from_pretrained(_final_copy(final_file, tmp_path / "w", no_w), device="cpu", w=3.0)
    assert c.cfg.w == 3.0


def test_embedder_revision_is_pinned(two_file):
    from anyintent.config import build_config
    from anyintent.hub import BGE_M3_REVISION
    cross = os.path.join(two_file, "cross")
    assert build_config(cross, embedder="BAAI/bge-m3").embedder["revision"] == BGE_M3_REVISION
    with pytest.warns(UserWarning, match="without embedder_revision"):
        cfg = build_config(cross, embedder="someone/other-encoder")
    assert cfg.embedder["revision"] is None
    assert build_config(cross, embedder="someone/other-encoder", embedder_revision="abc").embedder["revision"] == "abc"


def test_warm_up_is_exception_safe_and_never_caches(one_file, tmp_path):
    cache = str(tmp_path / "cache")
    c = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager", cache_dir=cache)
    c.warm_up(batch_sizes=(1,))
    assert c.L is None and not os.path.exists(cache)          # dummy labels never reach the disk cache
    c.set_labels(LABELS)
    before = (list(c.texts), c.label_key)
    real = c.classify

    def boom(*a, **kw):
        raise RuntimeError("simulated out of memory")
    c.classify = boom
    with pytest.raises(RuntimeError, match="simulated"):
        c.warm_up(batch_sizes=(1,))
    c.classify = real
    assert (list(c.texts), c.label_key) == before and c.classify("where is my parcel").label


def test_cache_key_has_the_versions(one_file, tmp_path):
    import anyintent
    cache = str(tmp_path / "cache")
    AnyIntent.from_pretrained(one_file, device="cpu", engine="eager", cache_dir=cache).set_labels(LABELS)
    side = [json.load(open(os.path.join(cache, f), encoding="utf-8")) for f in os.listdir(cache) if f.endswith(".json")][0]
    assert side["fields"]["lib"] == anyintent.__version__ and side["fields"]["torch"] and side["fields"]["transformers"]


@MAPPED
def test_weights_in_a_subfolder_are_part_of_the_revision(two_file, tmp_path):
    d = tmp_path / "two_copy"
    shutil.copytree(two_file, d)
    a = AnyIntent.from_pretrained(str(d), device="cpu", engine="eager")
    st = d / "cross" / "model.safetensors"
    raw = bytearray(st.read_bytes())
    raw[-1] ^= 0x01
    st.write_bytes(bytes(raw))
    b = AnyIntent.from_pretrained(str(d), device="cpu", engine="eager")
    assert a.revision != b.revision and a.revision.startswith("local-")
