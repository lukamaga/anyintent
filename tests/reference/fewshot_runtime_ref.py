# GENERATED from the project's registered few-shot runtime (not in this repository): byte for byte but for the
# R2Intent names in code and comments; used only as a reference of the tests. Do not edit.
# R2Intent few-shot runtime: training-free few-shot scoring for the one-file model (PROTOCOL 2026-10-05 23:33 and 23:37).
# A user gives N real example messages per label; nothing is trained. Three registered variants (docs/work/extras/
# fewshot_DESIGN.md section 4), each a change of the retrieval score r(q, l) and, for KX, of the label text that the
# cross-encoder reads:
#   Z  (zero-shot reference) r = cos(q, t_l)
#   P  (prototype)           r = cos(q, p_l), p_l = normalise(t_l + sum_i e_li)
#   K  (example kNN)         r = (1 - a) cos(q, t_l) + a agg_i cos(q, e_li), agg in max | mean | lse (tau 0.05)
#   KX (K + examples)        K's r; the cross text is the label text without generated examples followed by the m
#                            examples of that label nearest to the query, each in double quotes
# final = cross logit + w x r over the top k by r (ties: the lower label index in retrieval, the first maximum in
# retrieval order in the fusion, float64), exactly the v1.1 fusion with r in place of the label-text cosine.
#
# The numpy functions below are the only scoring code: the DEV notebook (FEWSHOT_DEV) calls them on precomputed
# matrices, the runtime class calls them on the vectors of one request, the look carries byte-identical copies of this
# file and the library's tests this copy with the R2Intent names. The model code is a copy (not an
# import) of the v1.1 runtime (v11_runtime_ref.py): the class
# R2IntentV11 below is verbatim except __init__, which also accepts the stage-1.2 layout (cross folder + adapter file).
import contextlib, json, os, re, zlib
import numpy as np

FEWSHOT_RUNTIME_VERSION = "1"
VARIANTS = ("Z", "P", "K", "KX")
AGGS = ("max", "mean", "lse")
LSE_TAU = 0.05
M_TEXT = 3
EX_WORD_CAP = 25
DEFAULT_LENGTHS = {"msg": 128, "label": 192, "pair": 160}


# ---- text helpers: verbatim copy from notebooks/step1_build_training_data.ipynb (sections 2 and 9) ----
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
# ---- end of the verbatim copy ----


def label_text_of(name, desc):
    # Label text format of step 0 and the step-1 data: "name: description", name only without a description
    return f"{name}: {desc}" if desc else name


def noex_label_text(name, label_text):
    # The label text without its quoted example phrasings: "name: description" -> "name: strip_examples(description)"
    name, label_text = str(name), str(label_text)
    pre = f"{name}: "
    if label_text.startswith(pre):
        d = label_text[len(pre):]
        return label_text_of(name, strip_examples(d, name) if d else "")
    return label_text


# ---------------------------------------------------------------------------------------------------------------
# Examples: cleaning, the KX label text, deterministic sampling
# ---------------------------------------------------------------------------------------------------------------
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


def sample_order(n, seed, key):
    # The registered example order of one label: a permutation of its n rows (rows sorted by id before) drawn with
    # numpy default_rng([seed, crc32(key)]); the first POOL_MAX rows are its example pool, N uses the first N
    return np.random.default_rng([int(seed), zlib.crc32(str(key).encode("utf-8"))]).permutation(int(n))


# ---------------------------------------------------------------------------------------------------------------
# Scoring (numpy). Shapes: Q [n, d], T [L, d], E [m, d] (unit vectors, float32); ex_label [m] label index of each
# example, sorted ascending (examples of one label are contiguous).
# ---------------------------------------------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------------------------------------------
# The model (copy of the v1.1 runtime; __init__ also loads the stage-1.2 layout cross folder + adapter file)
# ---------------------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LORA_PREFIX = "r2intent_embed_lora."
LORA_TARGETS = ("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense",
                "intermediate.dense", "output.dense")


def _dtype_kw(dtype):
    # transformers >= 4.56 takes `dtype`, older versions `torch_dtype` (an unknown keyword is not rejected there, it
    # would be passed to the config and the weights would load in fp32)
    major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    return {("dtype" if major_minor >= (4, 56) else "torch_dtype"): dtype}


class R2IntentV11:
    def __init__(self, path=None, device=None, dtype=None, cross_dir=None, adapter_path=None, meta=None):
        # path: a one-file model folder (v1.1 or stage-1.2 FINAL: model.safetensors with r2intent_embed_lora.* and
        # r2intent_meta.json). cross_dir + adapter_path: the stage-1.2 ARM_DEV layout (a cross-encoder folder and an
        # adapter file with the same tensor names); meta then gives w, k and the lengths (defaults 128 / 192 / 160).
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda = str(self.dev).startswith("cuda")
        dt = dtype or (torch.float16 if self.cuda else torch.float32)
        src = path if path is not None else cross_dir
        if path is not None:
            self.meta = json.load(open(os.path.join(path, "r2intent_meta.json"), encoding="utf-8"))
            lora_file = os.path.join(path, "model.safetensors")
        else:
            self.meta = {"msg_max_len": DEFAULT_LENGTHS["msg"], "label_max_len": DEFAULT_LENGTHS["label"],
                         "pair_max_len": DEFAULT_LENGTHS["pair"], **(meta or {})}
            lora_file = adapter_path
        self.tok = AutoTokenizer.from_pretrained(src)
        m = AutoModelForSequenceClassification.from_pretrained(src, **_dtype_kw(dt))
        self.model = m.to(self.dev).eval()
        if path is None and "adapter" not in self.meta:
            with safe_open(lora_file, framework="pt") as f:
                md = f.metadata() or {}
                rank = int(f.get_tensor(sorted(k for k in f.keys() if k.endswith("lora_A"))[0]).shape[0])
            self.meta["adapter"] = {"rank": rank, "alpha": float(md.get("lora_alpha") or 2.0 * rank),
                                    "targets": (md.get("lora_targets") or ",".join(LORA_TARGETS)).split(","),
                                    "prefix": LORA_PREFIX}
        ad = self.meta["adapter"]
        self.scale = float(ad["alpha"]) / int(ad["rank"])
        pre = ad.get("prefix", LORA_PREFIX) + "roberta."
        self.lora = {}
        with safe_open(lora_file, framework="pt") as f:
            for li, layer in enumerate(self.model.roberta.encoder.layer):
                for t in ad["targets"]:
                    mod = layer.get_submodule(t)
                    a = f.get_tensor(f"{pre}encoder.layer.{li}.{t}.lora_A").to(self.dev, torch.float32)
                    b = f.get_tensor(f"{pre}encoder.layer.{li}.{t}.lora_B").to(self.dev, torch.float32)
                    self.lora[mod] = (a, b)
        self.on = False
        for mod in self.lora:
            mod.register_forward_hook(self._hook)
        self.labels, self.L = [], None

    def _amp(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.cuda else contextlib.nullcontext()

    def _hook(self, mod, inp, out):
        if not self.on:
            return None
        a, b = self.lora[mod]
        x = inp[0]
        return out + (F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) * self.scale).to(out.dtype)

    @torch.no_grad()
    def embed(self, texts, max_len):
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.dev)
        self.on = True
        try:
            with self._amp():
                h = self.model.roberta(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, 0]
        finally:
            self.on = False
        return F.normalize(h.float(), dim=-1)

    @torch.no_grad()
    def cross(self, msgs, labels):
        enc = self.tok(list(msgs), list(labels), padding=True, truncation="longest_first",
                       max_length=int(self.meta["pair_max_len"]), return_tensors="pt").to(self.dev)
        with self._amp():
            return self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.view(-1).float()

    def set_labels(self, label_texts):
        self.labels = list(label_texts)
        self.L = self.embed(self.labels, int(self.meta["label_max_len"]))

    @torch.no_grad()
    def predict(self, msg, k=None, w=None, return_index=False):
        k = int(k or self.meta["fusion"]["k"])
        w = float(self.meta["fusion"]["w"] if w is None else w)
        q = self.embed([msg], int(self.meta["msg_max_len"]))
        cos, top = (q @ self.L.T)[0].topk(min(k, len(self.labels)))
        c = self.cross([msg] * len(top), [self.labels[i] for i in top.tolist()])
        j = int(top[int((c + w * cos).argmax())])
        return j if return_index else self.labels[j]


class FewShotRuntime(R2IntentV11):
    """The v1.1 runtime plus examples: set_examples(...) then classify(msgs, cfg). Vectors come from the model's own
    embedding pass (adapter on; examples at the message length); scoring is the numpy code above."""

    def set_examples(self, label_texts, examples=None, noex_texts=None, names=None):
        # examples: one list of strings per label (empty allowed). noex_texts: the label texts without generated
        # examples (KX); default: derived with strip_examples from "name: description" when names are given.
        self.set_labels(label_texts)
        n_lab = len(self.labels)
        examples = examples or [[] for _ in range(n_lab)]
        if len(examples) != n_lab:
            raise ValueError("one example list per label")
        self.ex_texts = [str(e) for ex in examples for e in ex]
        self.ex_label = np.array([j for j, ex in enumerate(examples) for _ in ex], dtype=np.int64)
        if noex_texts is None:
            noex_texts = [noex_label_text(n, t) for n, t in zip(names, self.labels)] if names else list(self.labels)
        self.noex = list(noex_texts)
        self.Tn = self.L.cpu().numpy().astype(np.float32)
        self.En = (self.embed(self.ex_texts, int(self.meta["msg_max_len"])).cpu().numpy().astype(np.float32)
                   if self.ex_texts else np.zeros((0, self.Tn.shape[1]), np.float32))
        self.Pn = prototypes(self.Tn, self.En, self.ex_label)
        return self

    def classify(self, msgs, cfg, batch=64):
        out = []
        msgs = list(msgs)
        for s in range(0, len(msgs), batch):
            ms = msgs[s:s + batch]
            Q = self.embed(ms, int(self.meta["msg_max_len"])).cpu().numpy().astype(np.float32)

            def cross_fn(pairs):
                if not pairs:
                    return np.zeros(0, np.float32)
                return self.cross([ms[i] for i, _ in pairs], [t for _, t in pairs]).cpu().numpy()
            r = score_batch(Q, self.Tn, cfg, self.En, self.ex_label, self.ex_texts, self.noex, self.labels, cross_fn,
                            P=self.Pn)
            out += [int(x) for x in r["pred"]]
        return out
