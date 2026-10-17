# GENERATED from the project's registered v1.1 training code (not in this repository): verbatim copies, with the
# R2Intent names in code and comments; used only as the reference of the equivalence tests. Do not edit.
import os, re, gc, sys, json, math, time, glob, random, shutil, hashlib, contextlib
import numpy as np, torch
import torch.nn.functional as F


PAIR_TRUNCATION = "longest_first"   # one truncation rule for every (message, label text) pair: train, DEV, test, card


WARNINGS = []   # every loud warning, printed again in the final summary


def loud_warning(msg):
    bar = "!" * 80
    print(f"{bar}\nWARNING: {msg}\n{bar}", flush=True)
    WARNINGS.append(str(msg))


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tokens(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def norm_text(t):
    return " ".join(tokens(t))


QUOTE_RE = re.compile(r'["“]([^"“”]{3,})["”]')


LEAD = (r"(?:for example|for instance|e\.g\.|i\.e\.|such as|like|including|"
        r"(?:example )?(?:phrasings?|examples?|phrases?)(?: include| are| like| could be| might be)?|"
        r"(?:users? )?(?:might|may|could|would) (?:say|ask|write|type)|as in)")


def strip_examples(d, label=""):
    # Remove quoted example phrasings (not the quoted label name) and the glue words around them
    lab = norm_text(label)
    def sub(m):
        return m.group(0) if lab and norm_text(m.group(1)) == lab else " \x00 "
    s = re.sub(r'["“]([^"“”]{1,})["”]', sub, str(d))
    s = re.sub(r"\x00(?:\s*[,;.]?\s*(?:(?:or|and)\s+)?\x00)*", "\x00", s)
    s = re.sub(r"[,;:]?\s*\(?\s*\b" + LEAD + r"\s*[:,]?\s*\x00\s*\)?", "\x00", s, flags=re.I)
    s = re.sub(r"\s*\(\s*\x00\s*\)", "", s)
    s = re.sub(r"[,;:]?\s*\x00\s*[,;:]?\s*(?=[A-Z])", ". ", s)
    s = re.sub(r"[,;:]?\s*\x00\s*", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])\s*([.!?])", r"\2", s)
    s = re.sub(r"([.!?])\s*\.", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:")
    if s and s[-1] not in '.!?"':
        s += "."
    return s


def label_text_of(name, desc):
    # Label text format of step 0 and the step-1 data: "name: description", name only without a description
    return f"{name}: {desc}" if desc else name


def dtype_kw(dtype):
    import transformers
    key = "dtype" if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"
    return {key: dtype} if dtype is not None else {}


def _strip(keys):
    return {k[len("roberta."):] if k.startswith("roberta.") else k for k in keys}


def load_seqcls_exact(path, dtype=None):
    # A cross-encoder that must load with no missing or unexpected key (reranker, saved checkpoints)
    from transformers import AutoModelForSequenceClassification
    m, info = AutoModelForSequenceClassification.from_pretrained(path, output_loading_info=True, **dtype_kw(dtype))
    bad = {k: sorted(v) for k, v in info.items() if v}
    if bad:
        raise RuntimeError(f"{path}: unexpected loading info {bad}")
    if m.config.num_labels != 1:
        raise RuntimeError(f"{path}: num_labels {m.config.num_labels}, expected 1")
    return m


def load_m3_encoder(path, dtype=None):
    # Frozen bge-m3 dense encoder: XLMRobertaModel without the pooler (only pooler.* may be unexpected)
    from transformers import XLMRobertaModel
    m, info = XLMRobertaModel.from_pretrained(path, add_pooling_layer=False, output_loading_info=True, **dtype_kw(dtype))
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs") or \
            not _strip(info["unexpected_keys"]) <= {"pooler.dense.weight", "pooler.dense.bias"}:
        raise RuntimeError(f"bge-m3 encoder: unexpected loading info { {k: sorted(v) for k, v in info.items() if v} }")
    return m


def amp_ctx(device, enabled=True):
    if enabled and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def cls_embed(roberta, enc):
    # Embedding mode = bge-m3 dense: <s> vector of the last layer, L2-normalised, fp32
    return F.normalize(roberta(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                       .last_hidden_state[:, 0].float(), dim=-1)


def cross_logits(model, enc):
    return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.view(-1).float()


def backbone(model):
    return getattr(model, "roberta", model)


def weight_shas(d):
    return {os.path.basename(p): sha256_file(p) for p in sorted(glob.glob(os.path.join(d, "*.safetensors")))}


EVAL_STATE = {"pair_batch": 512, "emb_batch": 128, "ooms": 0}


@torch.no_grad()
def embed_texts(model, tok, texts, max_len, device, batch=None, amp=True):
    # Normalised CLS vectors [n, d] fp32 (numpy); texts sorted by length for less padding
    rob = backbone(model)
    was = rob.training
    rob.eval()
    texts = list(texts)
    out = np.zeros((len(texts), rob.config.hidden_size), dtype=np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    i = 0
    while i < len(texts):
        b = batch or EVAL_STATE["emb_batch"]
        idx = order[i:i + b]
        try:
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            with amp_ctx(device, amp):
                out[idx] = cls_embed(rob, enc).cpu().numpy()
            i += len(idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if b <= 4:
                raise
            EVAL_STATE["emb_batch"] = b // 2
            EVAL_STATE["ooms"] += 1
            loud_warning(f"embedding OOM at batch {b}; continuing with {b // 2}")
    rob.train(was)
    return out


@torch.no_grad()
def cross_scores(model, tok, qs, ls, max_len, device, amp=True):
    # Cross-encoder logits for (message, label text) pairs; pairs sorted by length, OOM halves the batch
    was = model.training
    model.eval()
    n = len(qs)
    out = np.empty(n, dtype=np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(qs, ls)], kind="stable")
    i = 0
    while i < n:
        pb = EVAL_STATE["pair_batch"]
        idx = order[i:i + pb]
        try:
            enc = tok([qs[j] for j in idx], [ls[j] for j in idx], padding=True, truncation=PAIR_TRUNCATION,
                      max_length=max_len, return_tensors="pt").to(device)
            with amp_ctx(device, amp):
                out[idx] = cross_logits(model, enc).cpu().numpy()
            i += len(idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if pb <= 8:
                raise
            EVAL_STATE["pair_batch"] = pb // 2
            EVAL_STATE["ooms"] += 1
            loud_warning(f"cross-encoder OOM at {pb} pairs; continuing with {pb // 2}")
    model.train(was)
    return out


def retrieve(Q, Lm, kmax):
    # Top-k labels by cosine (stable order: ties to the lower label index); returns cand [n, k], vals, full S
    S = Q @ Lm.T
    k = min(kmax, Lm.shape[0])
    cand = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return cand, np.take_along_axis(S, cand, 1), S


def rerank(model, tok, msgs, label_texts, cand, k, max_len, device, amp=True):
    # Cross scores for the first min(k, kmax) candidates of every message: [n, kk]
    kk = min(k, cand.shape[1])
    qs = [m for m in msgs for _ in range(kk)]
    ls = [label_texts[j] for row in cand[:, :kk] for j in row]
    return cross_scores(model, tok, qs, ls, max_len, device, amp).reshape(len(msgs), kk)


def predict(cand, cross, k):
    # Prediction at k: the best cross score among the first k candidates (ties: retrieval order)
    kk = min(k, cross.shape[1])
    c = np.where(np.isnan(cross[:, :kk]), -np.inf, cross[:, :kk])
    return cand[np.arange(len(cand)), c.argmax(1)]


LORA_TARGETS = ("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense",
                "intermediate.dense", "output.dense")   # q, k, v, o, MLP fc1, MLP fc2 of every layer


LORA_PREFIX = "r2intent_embed_lora."


class EmbedLoRA(torch.nn.Module):
    """LoRA delta scale * B(A(x)) added to the outputs of the target linear layers of every encoder layer, through
    forward hooks. With on = False the hooks return None, so the backbone computes exactly what it computes without the
    adapter (the cross-encoder pass is the frozen cross model, bit for bit). A: rank x in (uniform +-1/sqrt(in), the
    kaiming-uniform default of nn.Linear and peft), B: out x rank (zeros), so a new adapter changes nothing."""

    def __init__(self, rob, rank, alpha, targets=LORA_TARGETS, seed=0, attach=True):
        super().__init__()
        self.rank, self.alpha = int(rank), float(alpha)
        self.scale = self.alpha / self.rank
        self.targets = tuple(targets)
        g = torch.Generator().manual_seed(int(seed))
        A, B, names, mods = [], [], [], []
        for li, lay in enumerate(rob.encoder.layer):
            for t in self.targets:
                m = lay.get_submodule(t)
                if not isinstance(m, torch.nn.Linear):
                    raise TypeError(f"LoRA target encoder.layer.{li}.{t} is {type(m).__name__}, not nn.Linear")
                a = torch.empty(self.rank, m.in_features, dtype=torch.float32)
                bound = 1.0 / math.sqrt(m.in_features)
                a.uniform_(-bound, bound, generator=g)
                A.append(torch.nn.Parameter(a))
                B.append(torch.nn.Parameter(torch.zeros(m.out_features, self.rank, dtype=torch.float32)))
                names.append(f"encoder.layer.{li}.{t}")
                mods.append(m)
        self.A, self.B = torch.nn.ParameterList(A), torch.nn.ParameterList(B)
        self.names = names
        self._mods = mods          # the backbone's modules (a plain list: not submodules of the adapter)
        self.on = False
        self._handles = []
        if attach:
            self.attach()

    def attach(self):
        self.detach()
        for i, m in enumerate(self._mods):
            self._handles.append(m.register_forward_hook(self._make_hook(i)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, i):
        def hook(mod, inp, out):
            if not self.on:
                return None
            x = inp[0]
            d = F.linear(F.linear(x, self.A[i].to(x.dtype)), self.B[i].to(x.dtype))
            return out + (d * self.scale).to(out.dtype)
        return hook

    def n_params(self):
        return int(sum(p.numel() for p in self.parameters()))

    def spec(self):
        return {"rank": self.rank, "alpha": self.alpha, "scale": self.scale, "targets": list(self.targets),
                "n_layers": len(self.names) // max(1, len(self.targets)), "n_matrices": 2 * len(self.names),
                "n_params": self.n_params(), "prefix": LORA_PREFIX}

    def lora_state(self, prefix="roberta."):
        sd = {}
        for n, a, b in zip(self.names, self.A, self.B):
            sd[f"{LORA_PREFIX}{prefix}{n}.lora_A"] = a.detach().float().cpu().contiguous()
            sd[f"{LORA_PREFIX}{prefix}{n}.lora_B"] = b.detach().float().cpu().contiguous()
        return sd

    @torch.no_grad()
    def load_lora_state(self, sd, prefix="roberta."):
        want = set(self.lora_state(prefix))
        have = {k for k in sd if k.startswith(LORA_PREFIX)}
        if want != have:
            raise RuntimeError(f"LoRA keys differ: missing {sorted(want - have)[:3]}, unexpected {sorted(have - want)[:3]}")
        for i, n in enumerate(self.names):
            self.A[i].copy_(sd[f"{LORA_PREFIX}{prefix}{n}.lora_A"].to(self.A[i].device, torch.float32))
            self.B[i].copy_(sd[f"{LORA_PREFIX}{prefix}{n}.lora_B"].to(self.B[i].device, torch.float32))

    @torch.no_grad()
    def merged_backbone_state(self, rob):
        # The backbone with W + scale * B @ A in every target (fp32 math, cast back to the stored dtype), on the CPU
        sd = {k: v.detach().cpu().clone() for k, v in rob.state_dict().items()}
        for n, a, b in zip(self.names, self.A, self.B):
            k = f"{n}.weight"
            w = sd[k]
            sd[k] = (w.float() + self.scale * (b.detach().float().cpu() @ a.detach().float().cpu())).to(w.dtype)
        return sd


@contextlib.contextmanager
def adapter_on(lora, on=True):
    # lora None: no adapter (a plain cross model); the previous state is restored on exit
    if lora is None:
        yield
        return
    prev = lora.on
    lora.on = bool(on)
    try:
        yield
    finally:
        lora.on = prev


def fused_pred(cand, retr, cross, k, w):
    # final score = cross logit + w x cosine of the retriever over the first k candidates; ties: retrieval order
    kk = min(k, cross.shape[1])
    s = cross[:, :kk].astype(np.float64) + float(w) * retr[:, :kk].astype(np.float64)
    s = np.where(np.isnan(s), -np.inf, s)
    return cand[np.arange(len(cand)), s.argmax(1)]


def save_one_file(model, lora, tok, out_dir, meta):
    from safetensors.torch import save_file
    tmp, old = out_dir + ".tmp", out_dir + ".old"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    sd = {k: v.detach().to("cpu", torch.float16).contiguous() for k, v in model.state_dict().items()}
    lsd = lora.lora_state()
    if set(sd) & set(lsd):
        raise RuntimeError("LoRA keys collide with backbone keys")
    sd.update(lsd)
    sp = lora.spec()
    save_file(sd, os.path.join(tmp, "model.safetensors"),
              metadata={"format": "pt", "r2intent": "v1.1", "lora_prefix": LORA_PREFIX, "lora_rank": str(sp["rank"]),
                        "lora_alpha": str(sp["alpha"]), "lora_targets": ",".join(sp["targets"]),
                        "embedding_pass": "adapter on", "cross_pass": "adapter off"})
    del sd
    model.config.save_pretrained(tmp)
    cp = os.path.join(tmp, "config.json")
    c = json.load(open(cp))
    c["dtype"] = "float16"
    if "torch_dtype" in c:
        c["torch_dtype"] = "float16"
    json.dump(c, open(cp, "w"), indent=2)
    tok.save_pretrained(tmp)
    json.dump(meta, open(os.path.join(tmp, "r2intent_meta.json"), "w"), indent=1, default=str)
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(out_dir):
        os.replace(out_dir, old)
    os.replace(tmp, out_dir)
    shutil.rmtree(old, ignore_errors=True)
    gc.collect()
    return weight_shas(out_dir)


def load_one_file(path_dir, dtype=None):
    # Returns (cross model, adapter, meta). The model must load with only the adapter keys unexpected.
    from transformers import AutoModelForSequenceClassification
    from safetensors import safe_open
    m, info = AutoModelForSequenceClassification.from_pretrained(path_dir, output_loading_info=True, **dtype_kw(dtype))
    unexpected = set(info.get("unexpected_keys") or [])
    tag = LORA_PREFIX.rstrip(".")
    bad = {k: sorted(v) for k, v in info.items() if v and k != "unexpected_keys"}
    other = sorted(k for k in unexpected if tag not in k)
    if bad or other:
        raise RuntimeError(f"{path_dir}: unexpected loading info {bad} / unexpected non-adapter keys {other[:5]}")
    meta = json.load(open(os.path.join(path_dir, "r2intent_meta.json")))
    sp = meta["adapter"]
    lora = EmbedLoRA(backbone(m), sp["rank"], sp["alpha"], tuple(sp["targets"]), attach=True)
    with safe_open(os.path.join(path_dir, "model.safetensors"), framework="pt") as f:
        lsd = {k: f.get_tensor(k) for k in f.keys() if k.startswith(LORA_PREFIX)}
    lora.load_lora_state(lsd)
    return m, lora, meta
