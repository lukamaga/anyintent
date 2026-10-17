"""PyTorch backends: the embedding pass (retrieval vectors) and the cross pass (pair logits) of a one-file or a two-file
model, with two engines.

eager: the Hugging Face forward. For a one-file model the adapter is applied by forward hooks on the backbone's linear
       layers that are active only in the embedding pass, exactly as the v1.1 reference runtime (the cross pass is the
       shipped cross-encoder, bit for bit).
fast:  the latency bench bodies (FastXLMR, FastXLMRLoRA, Head) on the same weights, pairs assembled from ids tokenised
       once, optional torch.compile with 16-token buckets on CUDA.
On CUDA both run fp16 weights under torch.autocast(float16) (the precision of the registered DEV scores); on the CPU
everything is fp32.
"""
import contextlib
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from .fast import FastXLMR, FastXLMRLoRA, Head, PairAssembler, V11_LORA_TARGETS, pad_batch
from .hub import AnyIntentError

log = logging.getLogger("anyintent")
EMB_CHUNK = 64     # texts per embedding forward
PAIR_CHUNK = 128   # pairs per cross forward (pairs sorted by length)


def _dtype_kw(dtype):
    # transformers >= 4.56 takes `dtype`, older versions `torch_dtype`
    import transformers
    major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    return {("dtype" if major_minor >= (4, 56) else "torch_dtype"): dtype}


def _strip(keys):
    return {k[len("roberta."):] if k.startswith("roberta.") else k for k in keys}


def load_cross(path, dtype, adapter_tag=None):
    """XLMRobertaForSequenceClassification with one logit. Nothing may be missing; only adapter keys (one-file) may be
    unexpected."""
    from transformers import AutoModelForSequenceClassification
    m, info = AutoModelForSequenceClassification.from_pretrained(path, output_loading_info=True, **_dtype_kw(dtype))
    unexpected = set(info.get("unexpected_keys") or [])
    other = sorted(k for k in unexpected if not (adapter_tag and adapter_tag in k))
    bad = {k: sorted(v) for k, v in info.items() if v and k != "unexpected_keys"}
    if bad or other:
        raise AnyIntentError(f"{path}: unexpected loading info {bad} / unexpected keys {other[:5]}")
    if m.config.num_labels != 1:
        raise AnyIntentError(f"{path}: num_labels {m.config.num_labels}, expected 1")
    if not hasattr(m, "roberta") or not hasattr(m, "classifier"):
        raise AnyIntentError(f"{path}: {type(m).__name__} is not an XLM-RoBERTa sequence classifier")
    return m


def load_encoder(path, dtype):
    """bge-m3 dense encoder: XLMRobertaModel without the pooler (only pooler.* may be unexpected)."""
    from transformers import XLMRobertaModel
    m, info = XLMRobertaModel.from_pretrained(path, add_pooling_layer=False, output_loading_info=True, **_dtype_kw(dtype))
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs") or \
            not _strip(info["unexpected_keys"]) <= {"pooler.dense.weight", "pooler.dense.bias"}:
        raise AnyIntentError(f"{path}: unexpected loading info { {k: sorted(v) for k, v in info.items() if v} }")
    return m


# the name key of the safetensors metadata of a one-file model, by its adapter prefix: the R2Intent releases
# (r2intent_embed_lora.*, key r2intent; PROTOCOL E5 (10f), decision D6)
METADATA_KEY_OF_PREFIX = {"r2intent_embed_lora.": "r2intent"}


def check_metadata(path, md, prefix):
    """The safetensors metadata of a one-file model agrees with its meta file: the name key of the adapter prefix
    (required for the R2Intent layout), and lora_prefix (when written) equal to the prefix."""
    names = [k for k in METADATA_KEY_OF_PREFIX.values() if k in md]
    want = METADATA_KEY_OF_PREFIX.get(prefix)
    if want == "r2intent" and not names:
        raise AnyIntentError(f"{path}: an R2Intent one-file model carries the safetensors metadata key 'r2intent'")
    if md.get("lora_prefix") not in (None, prefix):
        raise AnyIntentError(f"{path}: the safetensors metadata lora_prefix {md.get('lora_prefix')!r} is not the meta "
                             f"file's adapter prefix {prefix!r}")
    return names[0] if names else None


def read_adapter(path, adapter, n_layers):
    """The LoRA matrices of a one-file model (fp32), keyed by (layer, target)."""
    from safetensors import safe_open
    pre = adapter["prefix"] + "roberta."
    out, sd = {}, {}
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
        check_metadata(path, f.metadata() or {}, adapter["prefix"])
        keys = set(f.keys())
        for li in range(n_layers):
            for t in adapter["targets"]:
                ka, kb = f"{pre}encoder.layer.{li}.{t}.lora_A", f"{pre}encoder.layer.{li}.{t}.lora_B"
                if ka not in keys or kb not in keys:
                    raise AnyIntentError(f"{path}: adapter tensor {ka} or {kb} missing")
                a, b = f.get_tensor(ka).to(torch.float32), f.get_tensor(kb).to(torch.float32)
                out[(li, t)] = (a, b)
                sd[ka], sd[kb] = a, b
        extra = sorted(k for k in keys if k.startswith(adapter["prefix"]) and k not in sd)
    if extra:
        raise AnyIntentError(f"{path}: adapter tensors outside the declared layers/targets: {extra[:3]}")
    return out, sd


class TorchBackend:
    family = "torch"

    def __init__(self, cfg, device, dtype=None, engine="auto", compile=False, embedder_dir=None):
        from transformers import AutoTokenizer
        self.cfg = cfg
        self.dev = torch.device(device)
        self.cuda = self.dev.type == "cuda"
        if dtype is None:
            dtype = torch.float16 if self.cuda else torch.float32
        if isinstance(dtype, str):
            dtype = {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32,
                     "fp32": torch.float32}[dtype]
        if dtype not in (torch.float16, torch.float32):
            raise AnyIntentError("dtype must be float16 or float32 (bf16 is not supported)")
        if dtype == torch.float16 and not self.cuda:
            log.warning("float16 is not used on the CPU; computing in float32")
            dtype = torch.float32
        self.dtype = dtype
        self.amp = self.cuda and dtype == torch.float16
        self.compute_dtype = torch.float16 if self.amp else torch.float32
        self.tok = AutoTokenizer.from_pretrained(cfg.cross_dir)
        tag = cfg.adapter["prefix"].rstrip(".") if cfg.adapter else None
        self.model = load_cross(cfg.cross_dir, dtype, tag).to(self.dev).eval()
        self.lora, self._handles, self._lora_on = {}, [], False
        if cfg.layout == "one_file":
            ad = cfg.adapter
            self.scale = float(ad["alpha"]) / int(ad["rank"])
            layers = self.model.roberta.encoder.layer
            mats, self.lora_sd = read_adapter(cfg.cross_dir, ad, len(layers))
            for (li, t), (a, b) in mats.items():
                mod = layers[li].get_submodule(t)
                if not isinstance(mod, torch.nn.Linear):
                    raise AnyIntentError(f"adapter target encoder.layer.{li}.{t} is {type(mod).__name__}, not nn.Linear")
                # cast once to the compute dtype: identical to the runtime's per-call cast under autocast
                self.lora[mod] = (a.to(self.dev, self.compute_dtype), b.to(self.dev, self.compute_dtype))
            self.enc = self.model.roberta
            self.tok_emb = self.tok
        else:
            if embedder_dir is None:
                raise AnyIntentError("a two-file model needs its embedder folder")
            self.tok_emb = AutoTokenizer.from_pretrained(embedder_dir)
            self.enc = load_encoder(embedder_dir, dtype).to(self.dev).eval()
        self.pairs = PairAssembler(self.tok, cfg.pair_max_len)
        self.compile = bool(compile) and self.cuda
        if compile and not self.cuda:
            log.warning("compile=True is used on CUDA only; ignored on the CPU")
        self.engine = None
        self._build_engine(engine)

    # ---- engines -----------------------------------------------------------------------------------------------
    def _attach(self):
        self._detach()
        for mod in self.lora:
            self._handles.append(mod.register_forward_hook(self._hook))

    def _detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, mod, inp, out):
        if not self._lora_on:
            return None
        a, b = self.lora[mod]
        x = inp[0]
        return out + (F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) * self.scale).to(out.dtype)

    def _amp(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.amp else contextlib.nullcontext()

    def _make_fast(self):
        if self.cfg.layout == "one_file":
            if tuple(self.cfg.adapter["targets"]) != V11_LORA_TARGETS:
                raise AnyIntentError(f"fast engine: adapter targets {self.cfg.adapter['targets']} not supported")
            emb = FastXLMRLoRA(self.model.roberta, self.lora_sd, self.scale,
                               prefix=self.cfg.adapter["prefix"] + "roberta.").to(self.dev).eval()
        else:
            emb = FastXLMR(self.enc).eval()
        cross = FastXLMR(self.model.roberta).eval()
        head = Head(self.model.classifier).eval()
        return emb, cross, head

    def _build_engine(self, engine):
        if engine not in ("auto", "eager", "fast"):
            raise AnyIntentError(f"engine {engine!r}: expected auto, eager or fast")
        self._attach()
        self.engine = "eager"
        if engine == "eager":
            return
        try:
            self.f_emb, self.f_cross, self.f_head = self._make_fast()
        except Exception as e:
            if engine == "fast":
                raise
            log.warning(f"fast engine unavailable ({type(e).__name__}: {e}); using eager")
            return
        if engine == "auto":
            ok, info = self.self_check()
            if not ok:
                log.warning(f"fast engine self-check failed ({info}); using eager")
                return
        self._detach()
        self.engine = "fast"
        if self.compile:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 64
            if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
                torch._dynamo.config.accumulated_cache_size_limit = 512
            self.f_emb_c = torch.compile(self.f_emb, mode="reduce-overhead", dynamic=True)
            self.f_cross_c = torch.compile(self.f_cross, mode="reduce-overhead", dynamic=True)

    @torch.no_grad()
    def self_check(self):
        """Fast vs eager on a few probe texts (the engines must agree before fast is used by default)."""
        probe = ["where is my parcel", "zablokowana karta, co robić?", "kaip pakeisti slaptažodį"]
        labs = ["delivery status: the customer asks where an order is", "card blocked", "password change"]
        e_eager = self._embed_eager(probe, self.cfg.msg_max_len)
        e_fast = self._embed_fast(probe, self.cfg.msg_max_len)
        self.pairs.set_labels(labs)
        c_eager = self._cross_eager(probe, labs)
        c_fast = self._cross_fast(probe, [0, 1, 2], [0, 1, 2])
        dc = float(np.abs(e_eager - e_fast).max())
        dx = float(np.abs(c_eager - c_fast).max())
        tol_c, tol_x = (1e-4, 1e-4) if not self.amp else (5e-3, max(0.05, 0.01 * float(np.abs(c_eager).max())))
        return (dc <= tol_c and dx <= tol_x), {"max_abs_cos": dc, "max_abs_cross": dx}

    # ---- embedding pass ------------------------------------------------------------------------------------------
    def _embed_eager(self, texts, max_len):
        enc = self.tok_emb(list(texts), padding=True, truncation=True, max_length=int(max_len),
                           return_tensors="pt").to(self.dev)
        self._lora_on = self.cfg.layout == "one_file"
        try:
            with self._amp():
                h = self.enc(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, 0]
        finally:
            self._lora_on = False
        return F.normalize(h.float(), dim=-1).cpu().numpy()

    def _run(self, fn, compiled, ids, m):
        if compiled is not None and self.compile:
            torch.compiler.cudagraph_mark_step_begin()
            return compiled(ids, m)
        return fn(ids, m)

    def _embed_fast(self, texts, max_len):
        seqs = [list(x) for x in self.tok_emb(list(texts), truncation=True, max_length=int(max_len))["input_ids"]]
        ids, m = pad_batch(seqs, self.tok_emb.pad_token_id, self.dev, bucket=self.compile, min_b=2 if self.compile else 1)
        with self._amp():
            h = self._run(self.f_emb, getattr(self, "f_emb_c", None), ids, m)
        return F.normalize(h.float(), dim=-1)[:len(seqs)].cpu().numpy()

    @torch.no_grad()
    def embed(self, texts, max_len):
        texts = list(texts)
        out = None
        order = np.argsort([len(t) for t in texts], kind="stable")
        for i in range(0, len(texts), EMB_CHUNK):
            idx = order[i:i + EMB_CHUNK]
            chunk = [texts[j] for j in idx]
            v = self._embed_eager(chunk, max_len) if self.engine == "eager" else self._embed_fast(chunk, max_len)
            if out is None:
                out = np.zeros((len(texts), v.shape[1]), dtype=np.float32)
            out[idx] = v
        return out

    # ---- cross pass ----------------------------------------------------------------------------------------------
    def prepare_labels(self, label_texts):
        self.lab_texts = list(label_texts)
        self.pairs.set_labels(self.lab_texts)
        if self.engine == "fast":
            self.pairs.check(["where is my parcel", "x"])

    def _cross_eager(self, msgs, labs):
        enc = self.tok(list(msgs), list(labs), padding=True, truncation="longest_first",
                       max_length=int(self.cfg.pair_max_len), return_tensors="pt").to(self.dev)
        with self._amp():
            return self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.view(-1).float().cpu().numpy()

    def _cross_fast(self, msgs, mi, lj, msg_ids=None):
        return self._cross_fast_seqs(self.pairs.pairs(msgs, mi, lj, msg_ids))

    def _cross_fast_seqs(self, seqs):
        out = np.empty(len(seqs), dtype=np.float32)
        order = np.argsort([len(s) for s in seqs], kind="stable")
        for i in range(0, len(seqs), PAIR_CHUNK):
            idx = order[i:i + PAIR_CHUNK]
            ids, m = pad_batch([seqs[j] for j in idx], self.tok.pad_token_id, self.dev, bucket=self.compile)
            with self._amp():
                h = self._run(self.f_cross, getattr(self, "f_cross_c", None), ids, m)
                out[idx] = self.f_head(h).float()[:len(idx)].cpu().numpy()
        return out

    @torch.no_grad()
    def cross_pairs(self, msgs, mi, lj):
        """Logits of the pairs (msgs[mi[t]], label lj[t]) for the current label set."""
        mi, lj = list(mi), list(lj)
        if not mi:
            return np.zeros(0, dtype=np.float32)
        if self.engine == "fast":
            used = sorted(set(mi))
            pos = {i: p for p, i in enumerate(used)}
            ids = self.pairs.ids([msgs[i] for i in used])
            msg_ids = [None] * len(msgs)
            for i in used:
                msg_ids[i] = ids[pos[i]]
            return self._cross_fast(msgs, mi, lj, msg_ids)
        return self._cross_eager_chunked([msgs[i] for i in mi], [self.lab_texts[j] for j in lj])

    def _cross_eager_chunked(self, qs, ls):
        out = np.empty(len(qs), dtype=np.float32)
        order = np.argsort([len(a) + len(b) for a, b in zip(qs, ls)], kind="stable")
        for i in range(0, len(qs), PAIR_CHUNK):
            idx = order[i:i + PAIR_CHUNK]
            out[idx] = self._cross_eager([qs[j] for j in idx], [ls[j] for j in idx])
        return out

    @torch.no_grad()
    def cross_texts(self, qs, ls):
        """Logits of arbitrary (message, label text) pairs (the few-shot KX texts change per message): each pair is
        tokenised by the tokenizer with longest_first truncation, as in the cross pass of the runtime."""
        qs, ls = list(qs), list(ls)
        if not qs:
            return np.zeros(0, dtype=np.float32)
        if self.engine == "fast":
            seqs = [list(x) for x in self.tok(qs, ls, truncation="longest_first",
                                              max_length=int(self.cfg.pair_max_len))["input_ids"]]
            return self._cross_fast_seqs(seqs)
        return self._cross_eager_chunked(qs, ls)
