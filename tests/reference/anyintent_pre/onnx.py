"""Optional ONNX Runtime path for CPUs (extra "onnx": pip install anyintent[onnx]).

export(repo_or_path, out_dir) writes embed.onnx (the embedding body: FastXLMRLoRA for a one-file model, FastXLMR of
bge-m3 for a two-file model; output the <s> vector before normalisation) and cross.onnx (FastXLMR + Head of the
cross-encoder; output the logit), both fp32, with dynamic batch and length axes, and onnx_meta.json with the model
revision. exporter: "auto" (the TorchScript exporter, then the dynamo exporter, which needs onnxscript), "torchscript"
or "dynamo". quantize_int8=True also writes cross_int8.onnx (dynamic int8 quantisation of the cross body), whose accuracy
must be measured on DEV data before it is relied on. Load with AnyIntent.from_pretrained(path, engine="onnx",
onnx_dir=out_dir, onnx_variant="fp32" or "int8").
"""
import json
import os

import numpy as np
import torch

from .fast import PairAssembler
from .hub import AnyIntentError

EMB_CHUNK, PAIR_CHUNK = 64, 128


class _Cross(torch.nn.Module):
    def __init__(self, body, head):
        super().__init__()
        self.body, self.head = body, head

    def forward(self, input_ids, attention_mask):
        return self.head(self.body(input_ids, attention_mask))


def _export(module, path, out_name, pad_id, exporter="auto"):
    # exporter: "auto" (TorchScript, then the dynamo exporter if that fails), "torchscript" or "dynamo"
    if exporter not in ("auto", "torchscript", "dynamo"):
        raise AnyIntentError(f"exporter {exporter!r}: expected auto, torchscript or dynamo")
    ids = torch.randint(5, 100, (4, 40))
    m = torch.ones_like(ids)
    m[0, 30:] = 0
    ids[0, 30:] = pad_id
    errs = []
    with torch.no_grad():
        if exporter in ("auto", "torchscript"):
            try:
                torch.onnx.export(module, (ids, m), path, input_names=["input_ids", "attention_mask"],
                                  output_names=[out_name], dynamic_axes={"input_ids": {0: "B", 1: "L"},
                                                                         "attention_mask": {0: "B", 1: "L"},
                                                                         out_name: {0: "B"}},
                                  opset_version=17, do_constant_folding=True, dynamo=False)
                return "torchscript_opset17"
            except Exception as e:
                errs.append(f"torchscript: {type(e).__name__}: {str(e)[:300]}")
        if exporter == "torchscript":
            raise AnyIntentError(f"ONNX export failed: {errs}")
        try:   # the dynamo exporter needs onnxscript (extra "onnx")
            from torch.export import Dim
            B = getattr(Dim, "DYNAMIC", None) or Dim("B", min=1, max=4096)
            L = getattr(Dim, "DYNAMIC", None) or Dim("L", min=2, max=512)
            torch.onnx.export(module, (ids, m), path, input_names=["input_ids", "attention_mask"], output_names=[out_name],
                              dynamic_shapes={"input_ids": {0: B, 1: L}, "attention_mask": {0: B, 1: L}},
                              opset_version=18, dynamo=True)
            return "dynamo_opset18"
        except Exception as e:
            errs.append(f"dynamo: {type(e).__name__}: {str(e)[:300]}")
    raise AnyIntentError(f"ONNX export failed: {errs}")


def export(repo_or_path, out_dir, revision=None, quantize_int8=False, exporter="auto", **load_kwargs):
    from .model import AnyIntent
    clf = AnyIntent.from_pretrained(repo_or_path, revision=revision, device="cpu", dtype="float32", engine="fast",
                                    **load_kwargs)
    be = clf.backend
    os.makedirs(out_dir, exist_ok=True)
    meta = {"format": "anyintent-onnx", "version": 1, "model_revision": clf.revision,
            "embedder_revision": clf.embedder_revision, "layout": clf.cfg.layout, "files": {}}
    # each graph in its own folder: a model over 2 GB is written with external tensor files, whose names could
    # collide between the two graphs
    for sub in ("embed", "cross", "cross_int8"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    meta["files"]["embed"] = {"file": "embed/embed.onnx",
                              "exporter": _export(be.f_emb, os.path.join(out_dir, "embed", "embed.onnx"), "cls",
                                                  be.tok_emb.pad_token_id, exporter)}
    meta["files"]["cross"] = {"file": "cross/cross.onnx",
                              "exporter": _export(_Cross(be.f_cross, be.f_head), os.path.join(out_dir, "cross", "cross.onnx"),
                                                  "logit", be.tok.pad_token_id, exporter)}
    if quantize_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        src = os.path.join(out_dir, "cross", "cross.onnx")
        big = sum(os.path.getsize(os.path.join(out_dir, "cross", f)) for f in os.listdir(os.path.join(out_dir, "cross")))
        quantize_dynamic(src, os.path.join(out_dir, "cross_int8", "cross_int8.onnx"), weight_type=QuantType.QInt8,
                         use_external_data_format=big > 1_900_000_000)
        meta["files"]["cross_int8"] = {"file": "cross_int8/cross_int8.onnx",
                                       "exporter": "onnxruntime quantize_dynamic QInt8"}
    with open(os.path.join(out_dir, "onnx_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def _pad(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), int(pad_id), dtype=np.int64)
    m = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        m[i, :len(s)] = 1
    return ids, m


class OrtBackend:
    family = "onnx"
    engine = "onnx"
    dtype = "float32"

    def __init__(self, cfg, onnx_dir, revision, variant="fp32", embedder_dir=None, threads=None):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        meta = json.load(open(os.path.join(onnx_dir, "onnx_meta.json"), encoding="utf-8"))
        if meta.get("model_revision") != revision:
            raise AnyIntentError(f"{onnx_dir} was exported from model revision {meta.get('model_revision')}, "
                                 f"not {revision}: export again")
        key = {"fp32": "cross", "int8": "cross_int8"}.get(variant)
        if key is None or key not in meta["files"]:
            raise AnyIntentError(f"onnx_variant {variant!r} not available in {onnx_dir}")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if threads:
            so.intra_op_num_threads = int(threads)
        prov = ["CPUExecutionProvider"]
        self.s_emb = ort.InferenceSession(os.path.join(onnx_dir, meta["files"]["embed"]["file"]), so, providers=prov)
        self.s_cross = ort.InferenceSession(os.path.join(onnx_dir, meta["files"][key]["file"]), so, providers=prov)
        self.cfg, self.variant = cfg, variant
        self.dev = torch.device("cpu")
        self.tok = AutoTokenizer.from_pretrained(cfg.cross_dir)
        self.tok_emb = AutoTokenizer.from_pretrained(embedder_dir) if cfg.layout == "two_file" else self.tok
        self.pairs = PairAssembler(self.tok, cfg.pair_max_len)

    def embed(self, texts, max_len):
        texts = list(texts)
        seqs = [list(x) for x in self.tok_emb(texts, truncation=True, max_length=int(max_len))["input_ids"]]
        out = None
        order = np.argsort([len(s) for s in seqs], kind="stable")
        for i in range(0, len(seqs), EMB_CHUNK):
            idx = order[i:i + EMB_CHUNK]
            ids, m = _pad([seqs[j] for j in idx], self.tok_emb.pad_token_id)
            h = self.s_emb.run(None, {"input_ids": ids, "attention_mask": m})[0].astype(np.float32)
            h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
            if out is None:
                out = np.zeros((len(seqs), h.shape[1]), dtype=np.float32)
            out[idx] = h
        return out

    def prepare_labels(self, label_texts):
        self.pairs.set_labels(label_texts)
        self.pairs.check(["where is my parcel", "x"])

    def cross_pairs(self, msgs, mi, lj):
        mi, lj = list(mi), list(lj)
        if not mi:
            return np.zeros(0, dtype=np.float32)
        return self._run_cross(self.pairs.pairs(msgs, mi, lj))

    def cross_texts(self, qs, ls):
        """Logits of arbitrary (message, label text) pairs (few-shot KX), tokenised with longest_first truncation."""
        qs, ls = list(qs), list(ls)
        if not qs:
            return np.zeros(0, dtype=np.float32)
        return self._run_cross([list(x) for x in self.tok(qs, ls, truncation="longest_first",
                                                           max_length=int(self.cfg.pair_max_len))["input_ids"]])

    def _run_cross(self, seqs):
        out = np.empty(len(seqs), dtype=np.float32)
        order = np.argsort([len(s) for s in seqs], kind="stable")
        for i in range(0, len(seqs), PAIR_CHUNK):
            idx = order[i:i + PAIR_CHUNK]
            ids, m = _pad([seqs[j] for j in idx], self.tok.pad_token_id)
            out[idx] = self.s_cross.run(None, {"input_ids": ids, "attention_mask": m})[0].reshape(-1)
        return out
