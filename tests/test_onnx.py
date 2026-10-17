"""Optional ONNX Runtime path (skipped when onnxruntime is not installed)."""
import numpy as np
import pytest

from anyintent import AnyIntent, AnyIntentError
from conftest import LABELS, MESSAGES

ort = pytest.importorskip("onnxruntime")


def _compare(a, b, tol):
    n_diff, dmax = 0, 0.0
    for p, q in zip(a, b):
        pc = {c["index"]: c for c in p.candidates}
        for c in q.candidates:
            if c["index"] in pc and c["cross"] is not None:
                dmax = max(dmax, abs(pc[c["index"]]["cross"] - c["cross"]), abs(pc[c["index"]]["cosine"] - c["cosine"]))
        n_diff += int(p.index != q.index)
    return n_diff, dmax


@pytest.mark.parametrize("layout", ["one_file", "two_file"])
def test_onnx_fp32_matches_torch(layout, one_file, two_file, tmp_path):
    from anyintent import onnx as aonnx
    path = one_file if layout == "one_file" else two_file
    meta = aonnx.export(path, str(tmp_path / "onnx"), quantize_int8=True)
    print(f"ONNX export {layout}: {meta['files']}")
    t = AnyIntent.from_pretrained(path, device="cpu", engine="fast")
    o = AnyIntent.from_pretrained(path, engine="onnx", onnx_dir=str(tmp_path / "onnx"))
    t.set_labels(LABELS)
    o.set_labels(LABELS)
    a = t.classify(MESSAGES, cascade=False, return_scores=True)
    b = o.classify(MESSAGES, cascade=False, return_scores=True)
    n_diff, dmax = _compare(a, b, 1e-3)
    print(f"ONNX fp32 {layout}: top-1 differences {n_diff}, max abs score diff {dmax:.2e}")
    assert n_diff == 0 and dmax < 1e-3
    q = AnyIntent.from_pretrained(path, engine="onnx", onnx_dir=str(tmp_path / "onnx"), onnx_variant="int8")
    q.set_labels(LABELS)
    c = q.classify(MESSAGES, cascade=False, return_scores=True)
    assert all(np.isfinite(p.score) for p in c)
    n8, d8 = _compare(a, c, 1.0)
    print(f"ONNX int8 {layout} (information): top-1 differences {n8} of {len(MESSAGES)}, max abs diff {d8:.3f}")


def test_onnx_refuses_other_revision(one_file, final_file, tmp_path):
    from anyintent import onnx as aonnx
    aonnx.export(one_file, str(tmp_path / "o"))
    with pytest.raises(AnyIntentError):
        AnyIntent.from_pretrained(final_file, engine="onnx", onnx_dir=str(tmp_path / "o"))


def test_onnx_dynamo_exporter(one_file, tmp_path):
    # the fallback exporter (used when the TorchScript exporter is gone or fails), forced here
    pytest.importorskip("onnxscript")
    from anyintent import onnx as aonnx
    meta = aonnx.export(one_file, str(tmp_path / "dyn"), exporter="dynamo")
    assert {f["exporter"] for f in meta["files"].values()} == {"dynamo_opset18"}, meta["files"]
    t = AnyIntent.from_pretrained(one_file, device="cpu", engine="fast")
    o = AnyIntent.from_pretrained(one_file, engine="onnx", onnx_dir=str(tmp_path / "dyn"))
    t.set_labels(LABELS)
    o.set_labels(LABELS)
    n_diff, dmax = _compare(t.classify(MESSAGES, cascade=False, return_scores=True),
                            o.classify(MESSAGES, cascade=False, return_scores=True), 1e-3)
    print(f"ONNX dynamo one_file: top-1 differences {n_diff}, max abs score diff {dmax:.2e}")
    assert n_diff == 0 and dmax < 1e-3
    with pytest.raises(AnyIntentError):
        aonnx.export(one_file, str(tmp_path / "bad"), exporter="nonsense")
