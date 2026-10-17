"""AnyIntent: zero-shot intent classification over a label set given at run time.

Two passes over the label set:
1. retrieval: the message and every label text are embedded (the model's embedding pass: the one-file model's
   backbone with its adapter on, or bge-m3 for a two-file model) and the k labels with the highest cosine are kept;
2. rerank: the cross-encoder scores the (message, label text) pairs of those k labels; the final score is
   cross logit + w x cosine and the prediction is its argmax (w, k and the cascade threshold tau are frozen in the
   model's configuration).

Few-shot mode (optional, only for a model whose configuration carries a "fewshot" block): set_labels(...,
examples=[[...], ...]) gives N real example messages per label; retrieval and, for the variant KX, the label texts the
cross-encoder reads then use them (anyintent.fewshot, a verbatim copy of the registered few-shot scoring code). Nothing
is trained. Without examples the model is zero-shot, exactly as before.

One instance serialises its calls (a lock is held for the whole of set_labels, classify, predict and warm_up), so it
can be shared by the threads of a web server; for parallel throughput use one instance per worker process.
"""
import hashlib
import logging
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from . import fewshot as fs
from . import fusion
from ._version import __version__
from .cache import LabelCache, cache_key
from .config import build_config
from .hub import AnyIntentError, BGE_M3_IGNORE, local_fingerprint, resolve
from .labels import label_texts, parse_labels

log = logging.getLogger("anyintent")


@dataclass
class Prediction:
    label: str                       # the label name
    index: int                       # position of the label in the list given to set_labels
    score: float                     # fused score: cross logit + w x retrieval score (the cosine in zero-shot mode)
    label_text: str                  # the text the cross-encoder read for this label
    n_scored: int                    # (message, label) pairs scored by the cross-encoder for this message
    margin: Optional[float] = None   # fused score of the best minus the second-best scored candidate (None: one
    #                                  candidate scored); a small margin means a doubtful prediction. Not calibrated:
    #                                  fused-score units (cross logit + w x cosine), its scale depends on w, k and on
    #                                  whether the cascade ran; a threshold does not carry over between settings
    candidates: Optional[List[dict]] = field(default=None)   # with return_scores: every retrieved candidate

    def __str__(self):
        return self.label


class AnyIntent:
    """Load with AnyIntent.from_pretrained(repo_or_path) (or AnyIntent(repo_or_path)), then set_labels(...) and
    classify(...)."""

    def __init__(self, repo_or_path, revision=None, device=None, dtype=None, engine="auto", compile=False,
                 cache_dir=None, token=None, local_files_only=False, w=None, k=None, tau=None, label_text_mode=None,
                 embedder=None, embedder_revision=None, onnx_dir=None, onnx_variant="fp32"):
        if repo_or_path is None or str(repo_or_path).strip() == "":
            raise AnyIntentError("a model id or a local model folder is required, for example "
                                 "AnyIntent.from_pretrained('<user>/r2intent-v1')")
        root, rev = resolve(repo_or_path, revision, token, local_files_only)
        self.cfg = build_config(root, embedder=embedder, embedder_revision=embedder_revision, w=w, k=k, tau=tau,
                                label_text_mode=label_text_mode)
        if rev.startswith("local-") and os.path.normpath(self.cfg.cross_dir) != os.path.normpath(root):
            # the weights live in a sub-folder (anyintent.json cross.path): its files are part of the revision
            rev = "local-" + hashlib.sha256((rev + local_fingerprint(self.cfg.cross_dir)).encode()).hexdigest()
        if self.cfg.overridden:
            warnings.warn(f"overriding the frozen {self.cfg.overridden}: published numbers hold only at the frozen values",
                          stacklevel=2)
        emb_dir, emb_rev = None, None
        if self.cfg.layout == "two_file":
            e = self.cfg.embedder
            if e.get("path"):
                emb_dir, emb_rev = resolve(e["path"])
            else:
                emb_dir, emb_rev = resolve(e["repo"], e.get("revision"), token, local_files_only,
                                           ignore_patterns=BGE_M3_IGNORE)
        self.revision, self.embedder_revision = rev, emb_rev
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if engine == "onnx":
            from .onnx import OrtBackend
            if onnx_dir is None:
                raise AnyIntentError("engine='onnx' needs onnx_dir= (written by anyintent.onnx.export)")
            self.backend = OrtBackend(self.cfg, onnx_dir, rev, variant=onnx_variant, embedder_dir=emb_dir)
        else:
            from .backends import TorchBackend
            self.backend = TorchBackend(self.cfg, device, dtype=dtype, engine=engine, compile=compile,
                                        embedder_dir=emb_dir)
        self.cache = LabelCache(cache_dir=cache_dir)
        self.names, self.texts, self.L, self.label_key = [], [], None, None
        self.mode, self.fs, self.active = None, None, None
        self._lock = threading.RLock()
        self._v3s_warned = False
        if self.cfg.model_license:
            log.info(f"model license: {self.cfg.model_license} (see the LICENSE file of the model repository)")

    @classmethod
    def from_pretrained(cls, repo_or_path, revision=None, device=None, **kwargs):
        return cls(repo_or_path, revision=revision, device=device, **kwargs)

    # ---- settings ------------------------------------------------------------------------------------------------
    @property
    def config(self):
        d = self.cfg.public()
        d.update(revision=self.revision, embedder_revision=self.embedder_revision, engine=self.engine,
                 family=self.backend.family, device=str(getattr(self.backend, "dev", "cpu")),
                 dtype=str(getattr(self.backend, "dtype", "float32")).replace("torch.", ""),
                 active=None if self.active is None else dict(self.active, mode=self.mode,
                                                                 fewshot=self.fs is not None,
                                                                 n_examples=0 if self.fs is None else
                                                                 len(self.fs["ex_texts"])))
        return d

    @property
    def engine(self):
        return getattr(self.backend, "engine", None)

    @property
    def labels(self):
        return list(self.names)

    # ---- labels --------------------------------------------------------------------------------------------------
    def _key_fields(self, role, max_len, texts, **extra):
        return {"revision": self.revision, "embedder": self.embedder_revision, "role": role, "max_len": max_len,
                "family": self.backend.family, "engine": self.engine,
                "compile": bool(getattr(self.backend, "compile", False)),
                "variant": getattr(self.backend, "variant", None),
                "dtype": str(getattr(self.backend, "dtype", "float32")),
                "device": getattr(getattr(self.backend, "dev", None), "type", "cpu"),
                "lib": __version__, "torch": torch.__version__, "transformers": _transformers_version(),
                **extra, "texts": texts}

    def _vectors(self, role, texts, max_len, **extra):
        fields = self._key_fields(role, max_len, texts, **extra)
        key = cache_key(**fields)
        V = self.cache.get(key, texts)
        if V is None:
            V = self.backend.embed(texts, max_len)
            self.cache.put(key, texts, V, {k: v for k, v in fields.items() if k != "texts"})
        return V, key

    def set_labels(self, labels, label_text_mode=None, examples=None):
        """labels: names, or {"name", "description"} dicts. The label vectors are computed once per label set and
        model and cached. label_text_mode: None for the model's frozen mode; another mode uses its registered
        (w, k, tau) when the model carries them (anyintent.json fusion_by_mode). examples (few-shot mode, only for a
        model with a "fewshot" block): one list of example messages per label, aligned with labels (an empty list for a
        label without examples), or a dict {label name: [messages]}."""
        names, descs = parse_labels(labels)
        mode = label_text_mode or self.cfg.label_text_mode
        texts = label_texts(names, descs, mode)
        ex = _parse_examples(examples, names)
        if ex is not None and self.cfg.fewshot is None:
            raise AnyIntentError("examples= needs a model released with a few-shot configuration (a 'fewshot' block in "
                                 "anyintent.json or in the meta file, r2intent_meta.json); this "
                                 "model is zero-shot only")
        if mode == "v3s" and not self._v3s_warned:
            self._v3s_warned = True
            var = self.cfg.v3s_variant
            plain = ("pass label_text_mode='v2', which uses this model's registered v2 settings "
                     f"(w, k, tau = {self.cfg.fusion_for('v2')})" if "v2" in self.cfg.fusion_by_mode else
                     "they are used as given ('name: description') at the v3s settings")
            warnings.warn("label_text_mode 'v3s': this model was released with descriptions in the v3s layout"
                          + (f" (variant {var})" if var else "") + ". The library does not write them: pass "
                          "descriptions that follow that layout (see the README section 'Writing labels'). For plain "
                          "descriptions " + plain + ".", stacklevel=2)
        if len(set(texts)) < len(texts):
            warnings.warn("some labels have identical label texts; they get (almost) the same scores and either may win",
                          stacklevel=2)
        w, k, tau = self.cfg.fusion_for(mode)
        with self._lock:
            L, key = self._vectors("embed", texts, self.cfg.label_max_len)
            state = None
            if ex is not None:
                block = {kk: v for kk, v in self.cfg.fewshot.items() if kk not in ("source", "n_set")}
                ex_texts = [str(e) for row in ex for e in row]
                ex_label = np.array([j for j, row in enumerate(ex) for _ in row], dtype=np.int64)
                if ex_texts:
                    E, ekey = self._vectors("examples", ex_texts, self.cfg.msg_max_len, fewshot=block)
                else:
                    E, ekey = np.zeros((0, L.shape[1]), dtype=np.float32), None
                state = {"block": block, "ex_texts": ex_texts, "ex_label": ex_label, "E": E,
                         "P": fs.prototypes(L, E, ex_label),
                         "noex": [fs.noex_label_text(n, t_) for n, t_ in zip(names, texts)],
                         "index": {t_: j for j, t_ in reversed(list(enumerate(texts)))}, "key": ekey}
                key = cache_key(labels=key, examples=ekey, fewshot=block)
            self.names, self.texts, self.L, self.label_key = names, texts, L, key
            self.mode, self.fs = mode, state
            self.active = {"w": w, "k": k, "tau": tau}
            self.backend.prepare_labels(texts)
        return self

    # ---- classification ------------------------------------------------------------------------------------------
    def _resolve(self, k, cascade):
        if self.L is None:
            raise AnyIntentError("no labels: call set_labels(...) first")
        if self.fs is not None:
            k = self.fs["block"]["k"] if k is None else int(k)
            if cascade:
                raise AnyIntentError("the few-shot mode has no cascade threshold (cascade=True is not available)")
            cascade, tau = False, None
        else:
            k = self.active["k"] if k is None else int(k)
            tau = None
        if k < 1:
            raise AnyIntentError(f"k must be >= 1, got {k}")
        if cascade is None:
            cascade = False   # DECISIONS 2.6 (B-2): the cascade is off by default; the published numbers are without it
        elif cascade:
            # a threshold checked for this k: the frozen (k, tau) of the label text mode, else tau_by_k (B-2)
            tau = self.cfg.tau_for_k(self.mode, k)
            if tau is None and k == self.active["k"]:
                raise AnyIntentError("cascade=True needs a threshold: this model has no frozen tau for this label text "
                                     "mode (pass tau= at load)")
            if tau is None:
                checked = sorted(kk for kk in {self.active["k"]} | set(self.cfg.tau_by_k)
                                 if self.cfg.tau_for_k(self.mode, kk) is not None)
                raise AnyIntentError(f"cascade=True at k = {k}: no cascade threshold was checked for this k (checked: "
                                     f"k in {checked}); use the frozen k or cascade=False")
        k = min(k, len(self.names))
        first = min(self.cfg.cascade_first, k)
        return k, bool(cascade) and first < k, first, tau

    def _classify_fewshot(self, msgs, k, return_scores):
        """Few-shot decision of one chunk: anyintent.fewshot.score_batch (the registered scoring code) on this
        model's vectors; the cross pass reads the label texts (P, K) or the per-message KX texts."""
        st = self.fs
        block = dict(st["block"], k=k)
        Q = self.backend.embed(msgs, self.cfg.msg_max_len)

        def cross_fn(pairs):
            if not pairs:
                return np.zeros(0, dtype=np.float32)
            mi = [int(i) for i, _ in pairs]
            lj = [st["index"].get(t_) for _, t_ in pairs]
            if all(j is not None for j in lj):
                return self.backend.cross_pairs(msgs, mi, lj)
            return self.backend.cross_texts([msgs[i] for i in mi], [t_ for _, t_ in pairs])

        r = fs.score_batch(Q, self.L, block, st["E"], st["ex_label"], st["ex_texts"], st["noex"], self.texts,
                           cross_fn, P=st["P"])
        cand, retr, cross, pred, ptexts = r["cand"], r["retr"], r["cross"], r["pred"], r["texts"]
        fused = fusion.fused_scores(cross, retr, block["w"])
        pos = fused.argmax(1)
        margins = fusion.fused_margin(fused)
        cos = Q @ self.L.T
        out = []
        for i in range(len(msgs)):
            j = int(cand[i, pos[i]])
            if j != int(pred[i]):
                raise AnyIntentError("internal: few-shot argmax differs from score_batch")
            p = Prediction(label=self.names[j], index=j, score=float(fused[i, pos[i]]), label_text=ptexts[i][pos[i]],
                           n_scored=int(cand.shape[1]), margin=None if np.isnan(margins[i]) else float(margins[i]))
            if return_scores:
                p.candidates = [{"index": int(c), "label": self.names[int(c)], "retrieval": float(retr[i, t_]),
                                 "cosine": float(cos[i, int(c)]), "cross": float(cross[i, t_]),
                                 "fused": float(fused[i, t_]), "label_text": ptexts[i][t_]}
                                for t_, c in enumerate(cand[i])]
            out.append(p)
        return out

    def _classify_chunk(self, msgs, k, cascade, first, return_scores, tau=None):
        if self.fs is not None:
            return self._classify_fewshot(msgs, k, return_scores)
        w = self.active["w"]
        Q = self.backend.embed(msgs, self.cfg.msg_max_len)
        S = Q @ self.L.T
        cand, cos = fusion.topk_stable(S, k)
        n = len(msgs)
        cross = np.full((n, k), np.nan, dtype=np.float32)
        m = first if cascade else k
        mi = [i for i in range(n) for _ in range(m)]
        lj = [int(j) for i in range(n) for j in cand[i, :m]]
        cross[:, :m] = self.backend.cross_pairs(msgs, mi, lj).reshape(n, m)
        if cascade:
            stop = fusion.cascade_stop(cross[:, :m], cos[:, :m], w, tau)
            rest = np.flatnonzero(~stop)
            if len(rest):
                mi = [int(i) for i in rest for _ in range(k - m)]
                lj = [int(j) for i in rest for j in cand[i, m:k]]
                cross[rest, m:] = self.backend.cross_pairs(msgs, mi, lj).reshape(len(rest), k - m)
            n_scored = np.where(stop, m, k)
        else:
            n_scored = np.full(n, k)
        fused = fusion.fused_scores(cross, cos, w)
        pos = fused.argmax(1)
        margins = fusion.fused_margin(fused)
        out = []
        for i in range(n):
            j = int(cand[i, pos[i]])
            p = Prediction(label=self.names[j], index=j, score=float(fused[i, pos[i]]), label_text=self.texts[j],
                           n_scored=int(n_scored[i]), margin=None if np.isnan(margins[i]) else float(margins[i]))
            if return_scores:
                p.candidates = [{"index": int(c), "label": self.names[int(c)], "cosine": float(cos[i, t]),
                                 "retrieval": float(cos[i, t]),
                                 "cross": None if np.isnan(cross[i, t]) else float(cross[i, t]),
                                 "fused": None if np.isnan(cross[i, t]) else float(fused[i, t])}
                                for t, c in enumerate(cand[i])]
            out.append(p)
        return out

    def classify(self, text, k=None, cascade=None, return_scores=False, batch_size=64):
        """text: a string (returns a Prediction) or a list of strings (returns a list). k None: the frozen k of the
        label text mode (few-shot mode: the k of the fewshot block). cascade None or False: off (the default; the
        published numbers are without the cascade). cascade=True: the faster cascade, only at a k with a checked
        threshold (the frozen k of the mode, or a k of tau_by_k); never in few-shot mode."""
        single = isinstance(text, str)
        msgs = [text] if single else list(text)
        for i, t in enumerate(msgs):
            if not isinstance(t, str):
                raise TypeError(f"message {i}: expected str, got {type(t).__name__}")
        if not msgs:
            return []
        out = []
        bs = max(1, int(batch_size))
        with self._lock, torch.inference_mode():
            k, cascade, first, tau = self._resolve(k, cascade)
            for s in range(0, len(msgs), bs):
                out += self._classify_chunk(msgs[s:s + bs], k, cascade, first, return_scores, tau=tau)
        return out[0] if single else out

    def classify_batch(self, texts, batch_size=64, **kwargs):
        if isinstance(texts, str):
            raise TypeError("classify_batch takes a list of messages")
        return self.classify(list(texts), batch_size=batch_size, **kwargs)

    def predict(self, text, labels=None, examples=None, **kwargs):
        """One call: set the labels (and examples) if given, return the label name (or a list of names for a list of
        messages)."""
        with self._lock:
            if labels is not None:
                self.set_labels(labels, examples=examples)
            elif examples is not None:
                raise AnyIntentError("examples= needs labels= in predict (or call set_labels)")
            r = self.classify(text, **kwargs)
        return r.label if isinstance(r, Prediction) else [p.label for p in r]

    def warm_up(self, batch_sizes=(1, 64), ks=None):
        """Runs dummy messages through the shapes this session will use. Returns the seconds spent. Without labels it
        uses 16 dummy labels that never enter the label-vector cache; the label state is restored even on an error."""
        with self._lock:
            t0 = time.perf_counter()
            saved = (self.names, self.texts, self.L, self.label_key, self.mode, self.fs, self.active)
            try:
                if self.L is None:
                    texts = [f"warm up label {i}: a short description of request type {i}" for i in range(16)]
                    self.names, self.texts = list(texts), list(texts)
                    self.L = self.backend.embed(texts, self.cfg.label_max_len)
                    self.label_key, self.mode, self.fs = None, self.cfg.label_text_mode, None
                    self.active = {"w": self.cfg.w, "k": self.cfg.k, "tau": self.cfg.tau}
                    self.backend.prepare_labels(texts)
                ks = sorted({min(x, len(self.names)) for x in (ks or (3, 5, 8, self.cfg.k))})
                for b in batch_sizes:
                    msgs = [f"warm up message number {i} about an order" for i in range(int(b))]
                    for kk in ks:
                        self.classify(msgs, k=kk, cascade=False, batch_size=int(b))
                    if self.fs is None and self.active["tau"] is not None:
                        self.classify(msgs, cascade=True, batch_size=int(b))
            finally:
                self.names, self.texts, self.L, self.label_key, self.mode, self.fs, self.active = saved
                if self.L is not None:
                    self.backend.prepare_labels(self.texts)
            return time.perf_counter() - t0


def _transformers_version():
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return None


def _parse_examples(examples, names):
    """None, or one list of strings per label (aligned with names)."""
    if examples is None:
        return None
    if isinstance(examples, dict):
        unknown = sorted(set(examples) - set(names))
        if unknown:
            raise ValueError(f"examples for unknown labels: {unknown[:5]}")
        examples = [examples.get(n) or [] for n in names]
    if isinstance(examples, (str, bytes)) or len(examples) != len(names):
        raise ValueError(f"examples: one list per label expected ({len(names)} labels)")
    out = []
    for j, row in enumerate(examples):
        row = [] if row is None else row
        if isinstance(row, (str, bytes)):
            raise TypeError(f"examples of label {j}: a list of messages, not a single string")
        for e in row:
            if not isinstance(e, str):
                raise TypeError(f"examples of label {j}: expected str, got {type(e).__name__}")
        out.append(list(row))
    return out
