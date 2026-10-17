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


@dataclass(frozen=True)
class CheckResult:
    """check(): yes or no for one intent, with a probability (PROTOCOL entry of 2026-10-13 00:06, (1)). probability is
    the chance that the message expresses the intent when, before reading it, a match and a non-match are taken as
    equally likely (calibrated on DEV at that balance; the README gives the share of right "yes" answers at other match
    rates). Only the threshold 0.5 was measured."""
    yes: bool                        # probability >= threshold
    probability: float               # in [0, 1]
    intent: str                      # the intent's name
    label_text: str                  # the text the model read for the intent (as Prediction.label_text)
    score: float                     # the fused pair score, cross logit + w x cosine (the units of Prediction.score)
    threshold: float                 # the threshold used (0.5 unless given)


@dataclass(frozen=True)
class ScoresResult:
    """scores(): a value from 0 to 1 for every intent of the label set and the remainder (PROTOCOL entry of 2026-10-13
    00:06, (1)). A read intent's value is the chance that it is the right intent; the remainder is the chance that the
    right intent is one of the given intents the model did not read with the message; both assume that one of the given
    intents is right (the remainder is not a "none of these" answer). The values sum to 1 minus the remainder. 0.0 means
    "not read", not "impossible": the intents not read share the remainder in proportions the model does not know."""
    values: dict                     # {name: value} for every intent of the set, in set_labels order; 0.0: not read
    remainder: float                 # the chance that the right intent is one not read; 0.0 when every intent was read
    read: list                       # the names of the read intents (up to 5), in retrieval order
    label: str                       # the intent with the largest value: the label classify returns in the same call


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

    # ---- calibrated outputs: check and scores (PROTOCOL entry of 2026-10-13 00:06, (1); README) -------------------
    @property
    def calibration(self):
        """The release's calibration file (anyintent.calibration.CALIBRATION_FILE, format r2intent-calibration-1),
        parsed and validated at the first access (a model from the Hugging Face cache fetches only that file, at the
        commit of its model files), or None when the release has none or it cannot be fetched or read; a malformed file
        gives a warning here and raises only in check and scores. A copy: changing it changes nothing."""
        import copy
        with self._lock:
            try:
                doc = self._calibration_file()
            except AnyIntentError as e:
                warnings.warn(f"calibration file not usable: {e}", stacklevel=2)
                return None
            return None if doc is None else copy.deepcopy(doc)

    def _calibration_file(self):
        """The parsed calibration file (cached on the instance once read) or None. Looked up in the release root; for a
        model from the Hugging Face cache without it, fetched once per instance with hub.fetch_optional. Raises
        AnyIntentError for a malformed file (not cached: it is read again at the next call). The fetch does not know
        the local_files_only= and token= of the load (AnyIntent.__init__ keeps neither): see hub.fetch_optional."""
        from . import calibration as calib
        from .hub import fetch_optional
        doc = self.__dict__.get("_calib_doc")
        if doc is not None:
            return doc
        path = os.path.join(self.cfg.root, calib.CALIBRATION_FILE)
        if not os.path.isfile(path):
            path = None
            snap = calib.hf_snapshot(self.cfg.root)
            if snap is not None and not self.__dict__.get("_calib_fetched"):
                self._calib_fetched = True
                got = fetch_optional(snap[0], snap[1], calib.CALIBRATION_FILE)
                if got is not None and os.path.isfile(got):
                    path = got
            if path is None:
                return None
        doc = calib.read_calibration(path)
        self._calib_doc, self._calib_path = doc, path
        return doc

    def _calibration_for(self, kind):
        """The block of `kind` ("check" or "scores") of the calibration file, after the refusals of the PROTOCOL entry
        (1) and design 1.5, at call time: no calibration file, no block for the function, a calibration file of another
        model (model_sha256 against the loaded model.safetensors; a local folder must also be unchanged since the load,
        its revision id recomputed), a w different from the file's (cfg.w, a load override), ONNX int8; check also a
        label text mode different from the file's (cfg.label_text_mode); scores also no labels, few-shot mode, duplicate
        names, an active w or k or a label set's text mode different from the file's. Called with the lock held."""
        from . import calibration as calib
        from .hub import OPTIONAL_FILES
        doc = self._calibration_file()
        if doc is None:
            snap = calib.hf_snapshot(self.cfg.root)
            how = ""
            if snap is not None:
                how = (f"; it is not in the local Hugging Face cache and was not fetched (offline mode, no network, no "
                       "access with your stored login or HF_TOKEN, since a token= given at load is not used for this "
                       "file, or a release without this file at that commit): if the release ships it, run `hf "
                       f"download {snap[0]} {calib.CALIBRATION_FILE} --revision {snap[1]}` (the optional files of a "
                       f"release are listed in anyintent.hub.OPTIONAL_FILES {OPTIONAL_FILES})")
            raise AnyIntentError(f"{kind}: this model has no calibration file ({calib.CALIBRATION_FILE} in "
                                 f"{self.cfg.root}){how}. check and scores are available only for a release that ships "
                                 "it (the README says which functions a released model supports)")
        if kind not in doc["functions"]:
            raise AnyIntentError(f"{kind} is not available for this model: its calibration file has no '{kind}' "
                                 f"block (functions {doc['functions']}): the function did not pass its DEV rule and is "
                                 "not in this release")
        sha = self.__dict__.get("_calib_weights_sha256")
        if sha is None:
            if str(self.revision).startswith("local-"):
                # a local folder: the weights hashed below must be the loaded ones, so the folder's revision id
                # (computed at load from the weights and the config, meta and tokenizer files, as in __init__) must not
                # have changed since the load (memoised hashes: cheap when nothing changed)
                rev = local_fingerprint(self.cfg.root)
                if os.path.normpath(self.cfg.cross_dir) != os.path.normpath(self.cfg.root):
                    sub = local_fingerprint(self.cfg.cross_dir)
                    rev = "local-" + hashlib.sha256((rev + sub).encode()).hexdigest()
                if rev != self.revision:
                    raise AnyIntentError(f"{kind}: the model folder {self.cfg.root} changed after it was loaded "
                                         "(its weights, configuration or tokenizer files), so the calibration file "
                                         "cannot be bound to the loaded model.safetensors; load the model again")
            sha = calib.weights_sha256(self.cfg.cross_dir)
        if sha != doc["model_sha256"]:
            raise AnyIntentError(f"{kind}: the calibration file belongs to another model (its model_sha256 "
                                 f"{doc['model_sha256'][:12]}..., the loaded {calib.WEIGHTS_FILE} {sha[:12]}...)")
        self._calib_weights_sha256 = sha
        fus = doc["fusion"]
        if float(self.cfg.w) != float(fus["w"]):
            over = "w" in self.cfg.overridden
            raise AnyIntentError(f"{kind}: the model runs at w {self.cfg.w}"
                                 + (" (a w= override at load)" if over else "") + f" but was calibrated at w {fus['w']}"
                                 + ("; load it without the override" if over else ""))
        if getattr(self.backend, "variant", None) == "int8":
            raise AnyIntentError(f"{kind}: not calibrated for the ONNX int8 variant (its accuracy is not measured); "
                                 "use onnx_variant='fp32' or the PyTorch engines")
        if kind == "check":
            if self.cfg.label_text_mode != fus["label_text_mode"]:
                raise AnyIntentError(f"check: the model uses the label text mode {self.cfg.label_text_mode!r} but was "
                                     f"calibrated with {fus['label_text_mode']!r}; a label_text_mode= override at load "
                                     "changes the mode: load the model without it")
        else:
            if self.L is None:
                raise AnyIntentError("scores: no labels: call set_labels(...) first")
            if self.fs is not None:
                raise AnyIntentError("scores: not calibrated for the few-shot mode; call set_labels without examples")
            if len(set(self.names)) < len(self.names):
                dup = sorted({n for n in self.names if self.names.count(n) > 1})
                raise AnyIntentError(f"scores: the label set has duplicate names {dup[:5]}; scores gives one value per "
                                     "name, so every name must be unique")
            if float(self.active["w"]) != float(fus["w"]) or int(self.active["k"]) != int(fus["k"]):
                raise AnyIntentError(f"scores: the label set runs at w {self.active['w']}, k {self.active['k']} but "
                                     f"scores was calibrated at w {fus['w']}, k {fus['k']}")
            if self.mode != fus["label_text_mode"]:
                raise AnyIntentError(f"scores: the label set uses the label text mode {self.mode!r} but scores was "
                                     f"calibrated with {fus['label_text_mode']!r}; call set_labels without "
                                     "label_text_mode=")
        return doc[kind]

    def _check_vector(self, label_text):
        """The intent vector of check (embedding pass, adapter on, label_max_len), kept in a private in-memory cache:
        its own LabelCache, created at the first call, never written to disk and never shared with the label set's
        cache; it keeps the vectors of up to 256 intents."""
        cache = self.__dict__.get("_check_cache")
        if cache is None:
            cache = self._check_cache = LabelCache(max_items=256, cache_dir=None)
        texts = [label_text]
        key = cache_key(**self._key_fields("check_intent", self.cfg.label_max_len, texts))
        V = cache.get(key, texts)
        if V is None:
            V = self.backend.embed(texts, self.cfg.label_max_len)
            cache.put(key, texts, V)
        return V[0]

    def _pair_raw(self, texts, label_text, batch_size=64):
        """The raw scores of the pairs (texts[i], label_text), as check scores them: the cosine of the message vector
        (embedding pass, adapter on, msg_max_len) and the intent vector (_check_vector), the cross logit of the pair
        (adapter off, pair_max_len, longest_first) through backend.cross_texts, and the fused score cross + w x cosine
        in float64 at the model's w. Returns (cosine float32, cross float32, fused float64), one entry per message.
        Needs no calibration file and leaves the label set, the backend's prepared label texts and the shared label
        cache untouched."""
        msgs = [texts] if isinstance(texts, str) else list(texts)
        for i, t in enumerate(msgs):
            if not isinstance(t, str):
                raise TypeError(f"message {i}: expected str, got {type(t).__name__}")
        if not isinstance(label_text, str):
            raise TypeError(f"label_text: expected str, got {type(label_text).__name__}")
        n = len(msgs)
        cos = np.zeros(n, dtype=np.float32)
        cross = np.zeros(n, dtype=np.float32)
        bs = max(1, int(batch_size))
        with self._lock, torch.inference_mode():
            if n:
                v = self._check_vector(label_text)
            for s in range(0, n, bs):
                chunk = msgs[s:s + bs]
                Q = self.backend.embed(chunk, self.cfg.msg_max_len)
                cos[s:s + len(chunk)] = Q @ v
                cross[s:s + len(chunk)] = self.backend.cross_texts(chunk, [label_text] * len(chunk))
        return cos, cross, fusion.fused_scores(cross, cos, self.cfg.w)

    def _warn_once(self, flag, msg):
        if not self.__dict__.get(flag):
            setattr(self, flag, True)
            warnings.warn(msg, stacklevel=3)

    def check(self, text, intent, threshold=0.5, batch_size=64):
        """Yes or no for one intent, with a probability. text: a string (returns a CheckResult) or a list of strings
        (returns a list, in order, the same intent for every message). intent: a name, a {"name", "description"} dict
        or a (name, description) tuple; a list is refused (call check once per intent; each answer is independent of
        the other intents). probability = 1 / (1 + exp(-(a s + b))), s the fused pair score cross + w x cosine, (a, b)
        from the release's calibration file; yes = probability >= threshold. Meaning: the chance that the message
        expresses the intent when a match and a non-match are taken as equally likely before reading it (only 0.5 was
        measured). The pair is scored directly: the current label set, the backend's prepared label texts and the
        shared label cache are not used or changed. Available only when the release ships the calibration file with a
        check block; refusals as in _calibration_for, and a pair score that is not finite (a cosine or cross logit
        that is NaN or infinite) raises: no probability is given for it."""
        from . import calibration as calib
        single = isinstance(text, str)
        msgs = [text] if single else list(text)
        for i, t in enumerate(msgs):
            if not isinstance(t, str):
                raise TypeError(f"message {i}: expected str, got {type(t).__name__}")
        if isinstance(intent, list):
            raise AnyIntentError("check takes one intent; call it once per intent")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.integer, np.floating)) or \
                not np.isfinite(threshold):
            raise ValueError(f"threshold: expected a finite number, got {threshold!r}")
        threshold = float(threshold)
        try:
            names, descs = parse_labels([intent])
        except (TypeError, ValueError) as e:
            raise type(e)(f"intent: {e}") from e
        name, desc = names[0], descs[0]
        with self._lock:
            block = self._calibration_for("check")
            label_text = label_texts([name], [desc], self.cfg.label_text_mode)[0]
            if not desc:
                self._warn_once("_check_warned_no_description",
                                "check: the probabilities were calibrated on intents written with a description; a "
                                "name alone was not part of the ship rule")
                if any(n == name and t_ != n for n, t_ in zip(self.names, self.texts)):
                    self._warn_once("_check_warned_bare_name",
                                    f"check scored the name {name!r} alone; pass the description as you did in "
                                    "set_labels")
            if not msgs:
                return []
            cos, cross, fused = self._pair_raw(msgs, label_text, batch_size=batch_size)
            bad = ~(np.isfinite(cos) & np.isfinite(cross) & np.isfinite(fused))
            if bad.any():
                raise AnyIntentError(f"check: the pair score of {int(bad.sum())} of {len(fused)} messages is not "
                                     "finite (a cosine or cross logit that is NaN or infinite, for example an fp16 "
                                     "overflow); no probability is given")
            try:
                prob = calib.check_probability(fused, block["a"], block["b"])
            except calib.CalibrationValueError as e:
                raise AnyIntentError(f"check: {e}") from e
        out = [CheckResult(yes=bool(p >= threshold), probability=float(p), intent=name, label_text=label_text,
                           score=float(s), threshold=threshold) for p, s in zip(prob, fused)]
        return out[0] if single else out

    def scores(self, text, batch_size=64):
        """A value from 0 to 1 for every intent of the current label set and the remainder. text: a string (returns a
        ScoresResult) or a list of strings (returns a list, in order). It runs classify's own passes
        (classify(text, cascade=False, return_scores=True)) and applies the release's temperature T and remainder
        parameters (gamma, eta, kappa) to the fused scores f of the intents classify reads (the frozen k, 5) and the
        size L of the label set: on a set of 6 intents or more a read intent j gets the logit (f_j - f_(5)) / T and a
        virtual sixth candidate gamma + eta (f_(1) - f_(5)) + kappa log(L - 5); the values and the remainder are their
        softmax shares; on a set of 5 or fewer every intent is read, the values are the softmax of f_j / T and the
        remainder is 0. Every intent not read gets 0.0 ("not read"). The values sum to 1 minus the remainder; label
        (the intent with the largest value) is the label classify returns in the same call. Available only when the
        release ships the calibration file with a scores block; refusals as in _calibration_for, and a read fused
        score that is not finite raises."""
        from . import calibration as calib
        single = isinstance(text, str)
        with self._lock:
            block = self._calibration_for("scores")
            if any(n == t_ for n, t_ in zip(self.names, self.texts)):
                self._warn_once("_scores_warned_no_description",
                                "scores: some intents of the label set have no description; the values were calibrated "
                                "on intents written with a description, names alone were not part of the ship rule")
            preds = self.classify(text, cascade=False, return_scores=True, batch_size=batch_size)
            preds = [preds] if single else preds
            if not preds:
                return []
            names = list(self.names)
            F = np.array([[c["fused"] for c in p.candidates] for p in preds], dtype=np.float64)
            r = block["remainder"]
            try:
                vals, rem = calib.scores_values(F, np.full(len(preds), len(names)), block["temperature"], r["gamma"],
                                                r["eta"], r["kappa"])
            except calib.CalibrationValueError as e:
                raise AnyIntentError(f"scores: {e}") from e
        out = []
        for i, p in enumerate(preds):
            values = dict.fromkeys(names, 0.0)
            read = []
            for t, c in enumerate(p.candidates):
                values[c["label"]] = float(vals[i, t])
                read.append(c["label"])
            if values[p.label] != max(values.values()):
                raise AnyIntentError("internal: the largest value of scores is not classify's label")
            out.append(ScoresResult(values=values, remainder=float(rem[i]), read=read, label=p.label))
        return out[0] if single else out


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
