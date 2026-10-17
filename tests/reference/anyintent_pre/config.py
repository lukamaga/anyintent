"""Resolved model settings from the model meta file, the optional anyintent.json and keyword overrides.

The meta file is r2intent_meta.json (the R2Intent names of the releases, PROTOCOL E5 (10f), decision D6);
its name key "r2intent" gives the version. A meta file under another name is not read: such a folder is a folder
without a meta file.

Precedence: keyword arguments, then anyintent.json, then the meta file (its "fusion" block, else the v1 key "k"
with w = 0). The reader accepts a few spellings of the cascade settings (fusion.tau, a top-level tau, a "cascade"
block {"tau", "first"}) and of the label text mode (top-level label_text_mode or fusion.label_text_mode); fusion.*
wins when several are present.

Layout. A one-file model has an "adapter" block in its meta file. A two-file model (a cross-encoder plus a separate
retrieval encoder, bge-m3 by default) is accepted only when something says so: anyintent.json with format
"anyintent-two-file", an explicit embedder= argument, or a v1 checkpoint meta (keys "variant" and "k", no "fusion"),
read from r2intent_meta.json like every meta file (a checkpoint written under another meta file name is renamed
first). Anything else is an error, so that a bare cross-encoder folder is never run silently as a configuration it was
not registered with.

Optional blocks (interface additions; files without them load as before):
- "fusion_by_mode" in anyintent.json: {"v2": {"w", "k", "tau"}, ...}: the registered (w, k, tau) of another label text
  mode (for example the v2 settings of a model released with v3s descriptions); set_labels(..., label_text_mode=M)
  then uses that block. The frozen mode keeps the main fusion values.
- "fewshot" in anyintent.json (else the meta file): the few-shot configuration (the "chosen" keys of the few-shot
  DEV decision: variant, agg, a, w, k, lse_tau, m_text, ex_word_cap); without it the model is zero-shot only.
"""
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

from . import fewshot as fs
from .hub import AnyIntentError, BGE_M3_REPO, BGE_M3_REVISION, read_json
from .labels import MODES

log = logging.getLogger("anyintent")
_MISSING = object()
META_FILE = "r2intent_meta.json"          # the meta file of the R2Intent releases
NAME_KEY = "r2intent"                     # the meta's (and the safetensors metadata's) name key
ADAPTER_PREFIX = "r2intent_embed_lora."   # the adapter tensors of a one-file model


def read_meta(cross_dir):
    """(meta dict or None, the meta file name or None): r2intent_meta.json; (None, None) without it"""
    if not os.path.exists(os.path.join(cross_dir, META_FILE)):
        return None, None
    return read_json(cross_dir, META_FILE), META_FILE


@dataclass
class ModelConfig:
    layout: str                       # "one_file" or "two_file"
    root: str
    cross_dir: str
    w: float
    k: int
    tau: Optional[float]
    cascade_first: int
    msg_max_len: int                  # embedding pass, messages
    label_max_len: int                # embedding pass, label texts
    pair_max_len: int                 # cross pass, (message, label text) pairs, truncation longest_first
    label_text_mode: str
    v3s_variant: Optional[str] = None  # information: the v3s description layout the model was released with
    adapter: Optional[dict] = None
    embedder: Optional[dict] = None   # two-file: {"repo", "revision", "path", "max_len"}
    model_license: Optional[str] = None
    version: Optional[str] = None
    meta_file: Optional[str] = None   # r2intent_meta.json (None: no meta file)
    layout_reason: str = ""
    overridden: list = field(default_factory=list)
    fusion_by_mode: dict = field(default_factory=dict)   # {mode: {"w", "k", "tau"}} for modes other than the frozen one
    tau_by_k: dict = field(default_factory=dict)         # {k: tau or None}: cascade thresholds checked per k (B-2)
    fewshot: Optional[dict] = None    # validated few-shot block (None: zero-shot only)
    meta: dict = field(default_factory=dict)

    def fusion_for(self, mode):
        """(w, k, tau) registered for a label text mode: its fusion_by_mode block, else the main values."""
        b = self.fusion_by_mode.get(mode)
        return (self.w, self.k, self.tau) if b is None else (b["w"], b["k"], b["tau"])

    def tau_for_k(self, mode, k):
        """The cascade threshold checked for (label text mode, k): the mode's frozen tau at its frozen k; at another k
        the tau_by_k entry (modes without their own fusion_by_mode block only); None when none was checked
        (DECISIONS 2.6, B-2)."""
        _, k_m, tau_m = self.fusion_for(mode)
        if int(k) == int(k_m):
            return tau_m
        if mode in self.fusion_by_mode:
            return None
        return self.tau_by_k.get(int(k))

    def public(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ("meta", "adapter")}
        if self.adapter:
            d["adapter"] = {k: self.adapter.get(k) for k in ("rank", "alpha", "targets", "prefix")}
        return d


def _int(x, what):
    try:
        v = int(x)
    except (TypeError, ValueError):
        raise AnyIntentError(f"{what}: expected an integer, got {x!r}")
    if v < 1:
        raise AnyIntentError(f"{what}: must be >= 1, got {v}")
    return v


def _version_tuple(v):
    m = re.search(r"(\d+)\.(\d+)", str(v or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _first(*pairs):
    """The value of the first key that is present (even with the value None); _MISSING if none is."""
    for d, key in pairs:
        if isinstance(d, dict) and key in d:
            return d[key]
    return _MISSING


def is_v1_checkpoint(meta):
    return isinstance(meta, dict) and "variant" in meta and "k" in meta and "fusion" not in meta


def _layout(meta, ai, embedder, cross_dir, meta_file="the meta file"):
    adapter = meta.get("adapter") if isinstance(meta.get("adapter"), dict) else None
    fmt = ai.get("format")
    if adapter:
        if fmt == "anyintent-two-file":
            raise AnyIntentError(f"{cross_dir}: anyintent.json says two-file but {meta_file} has an adapter "
                                 "(a one-file model); remove one of the two")
        return "one_file", adapter, f"{meta_file} has an adapter block"
    if fmt == "anyintent-one-file":
        raise AnyIntentError(f"{cross_dir}: anyintent.json says one-file but {meta_file} has no adapter block")
    if fmt == "anyintent-two-file":
        return "two_file", None, "anyintent.json format anyintent-two-file"
    if embedder is not None:
        return "two_file", None, "embedder= argument"
    if is_v1_checkpoint(meta):
        return "two_file", None, "v1 checkpoint meta (variant and k, no fusion): bge-m3 retrieval with w = 0"
    raise AnyIntentError(
        f"{cross_dir}: not a one-file model (no meta file with an adapter block: {META_FILE}) and nothing says it is "
        "a two-file model. Use one of: (1) an anyintent.json next to it with format 'anyintent-two-file' (embedder, "
        "w, k), (2) the embedder= argument (a folder or a Hugging Face id of the retrieval encoder), (3) a v1 "
        f"checkpoint meta in {META_FILE} (keys 'variant' and 'k', no 'fusion').")


def build_config(root, cross_dir=None, embedder=None, embedder_revision=None, w=None, k=None, tau=None,
                 label_text_mode=None):
    root = os.path.abspath(root)
    ai = read_json(root, "anyintent.json") or {}
    if ai and ai.get("format") not in (None, "anyintent-two-file", "anyintent-one-file"):
        raise AnyIntentError(f"anyintent.json: unknown format {ai.get('format')!r}")
    if cross_dir is None:
        cross_dir = os.path.normpath(os.path.join(root, (ai.get("cross") or {}).get("path") or "."))
    meta, meta_file = read_meta(cross_dir)
    if meta is None:
        meta = {}
    mf = meta_file or "the meta file"
    if not os.path.exists(os.path.join(cross_dir, "config.json")):
        raise AnyIntentError(f"{cross_dir}: no config.json (not a model folder)")
    layout, adapter, reason = _layout(meta, ai, embedder, cross_dir, mf)
    if adapter and adapter.get("prefix") not in (None, ADAPTER_PREFIX):
        raise AnyIntentError(f"{cross_dir}: {meta_file} names the adapter prefix {adapter.get('prefix')!r}; the R2Intent "
                             f"layout pairs {META_FILE} with {ADAPTER_PREFIX}*")
    if adapter and not adapter.get("prefix"):
        adapter = dict(adapter, prefix=ADAPTER_PREFIX)

    meta_fusion = meta.get("fusion") if isinstance(meta.get("fusion"), dict) else {}
    fusion = dict(meta_fusion)
    if not fusion and is_v1_checkpoint(meta):            # v1 checkpoint: k only, no fusion
        fusion = {"w": 0.0, "k": meta["k"]}
    elif not fusion and "k" in meta:
        fusion = {"k": meta["k"]}
    casc = meta.get("cascade") if isinstance(meta.get("cascade"), dict) else {}
    tau_src = _first((meta_fusion, "tau"), (casc, "tau"), (meta, "tau"))
    if tau_src is not _MISSING:
        fusion["tau"] = tau_src
    first_src = _first((meta_fusion, "cascade_first"), (casc, "first"), (casc, "cascade_first"), (meta, "cascade_first"))
    if first_src is not _MISSING and first_src is not None:
        fusion["cascade_first"] = first_src
    ai_fusion = dict(ai.get("fusion") or {})
    fusion.update({kk: v for kk, v in ai_fusion.items() if kk != "label_text_mode"})
    overridden = []
    for name, val in (("w", w), ("k", k), ("tau", tau)):
        if val is not None:
            if fusion.get(name) != val:
                overridden.append(name)
            fusion[name] = val
    if "k" not in fusion or fusion.get("k") is None:
        raise AnyIntentError(f"{cross_dir}: no k in {mf} or anyintent.json; pass k=...")
    if fusion.get("w") is None:
        if layout == "one_file":
            raise AnyIntentError(f"{cross_dir}: no fusion weight w in {mf} or anyintent.json (fusion.w); "
                                 "pass w=... (w = 0 is a different configuration)")
        warnings.warn(f"{cross_dir}: no fusion weight w found; using w = 0 (retrieval cosine not fused)", stacklevel=3)
    w_v = float(fusion.get("w") or 0.0)
    k_v = _int(fusion["k"], "k")
    tau_v = fusion.get("tau")
    tau_v = None if tau_v is None else float(tau_v)
    tbk_src = _first((ai_fusion, "tau_by_k"), (meta_fusion, "tau_by_k"), (casc, "tau_by_k"), (meta, "tau_by_k"))
    tau_by_k = _tau_by_k(None if tbk_src is _MISSING else tbk_src, k_v, tau_v,
                         check=not ({"k", "tau"} & set(overridden)))
    if "k" in overridden and tau is None:
        # B-2: a k override at load takes the threshold checked for that k (none if it was not checked), never the
        # threshold of the frozen k
        new_tau = tau_by_k.get(k_v)
        if new_tau != tau_v:
            warnings.warn(f"k overridden to {k_v}: the cascade threshold becomes {new_tau} (tau_by_k), not the frozen "
                          f"k's {tau_v}; cascade=True needs a threshold checked for this k", stacklevel=3)
        tau_v = new_tau
    first = _int(fusion.get("cascade_first", 3), "cascade_first")

    mode_src = _first((ai, "label_text_mode"), (ai_fusion, "label_text_mode"), (meta, "label_text_mode"),
                      (meta_fusion, "label_text_mode"))
    mode = label_text_mode or (None if mode_src is _MISSING else mode_src) or "v2"
    if mode not in MODES:
        raise AnyIntentError(f"label_text_mode {mode!r}: expected one of {MODES}")
    v3s_variant = _first((ai, "v3s_variant"), (meta, "v3s_variant"), (meta_fusion, "v3s_variant"))
    v3s_variant = None if v3s_variant is _MISSING else v3s_variant

    version = str(meta.get(NAME_KEY) or "")
    vt = _version_tuple(version)
    if vt is not None and vt >= (1, 2) and layout == "one_file":
        if tau is None and "tau" not in ai_fusion and tau_src is _MISSING:
            warnings.warn(f"{cross_dir}: {NAME_KEY} {version} model without a cascade threshold (no fusion.tau, "
                          "cascade.tau or tau key): the cascade is OFF. If the registered configuration has a tau, the "
                          "meta was written with another key name; add it to anyintent.json (fusion.tau).", stacklevel=3)
            log.warning(f"{NAME_KEY} >= 1.2 meta without tau: cascade off")
        if label_text_mode is None and mode_src is _MISSING:
            warnings.warn(f"{cross_dir}: {NAME_KEY} {version} model without label_text_mode: using 'v2'. If the "
                          "registered configuration uses another label text mode, add it to anyintent.json.",
                          stacklevel=3)
            log.warning(f"{NAME_KEY} >= 1.2 meta without label_text_mode: v2")

    trunc = ((meta.get("passes") or {}).get("cross") or {}).get("pair_truncation") or meta.get("pair_truncation")
    if trunc not in (None, "longest_first"):
        raise AnyIntentError(f"pair truncation {trunc!r} is not supported (the models use longest_first)")
    cross_cfg = ai.get("cross") or {}
    pair_max_len = cross_cfg.get("pair_max_len") or meta.get("pair_max_len")
    if pair_max_len is None:
        raise AnyIntentError(f"{cross_dir}: pair_max_len missing in {mf} (or anyintent.json cross)")
    emb = None
    if layout == "one_file":
        msg_len, lab_len = meta.get("msg_max_len"), meta.get("label_max_len")
        if msg_len is None or lab_len is None:
            raise AnyIntentError(f"{cross_dir}: msg_max_len / label_max_len missing in {mf}")
        if embedder is not None:
            raise AnyIntentError("embedder= is only used with a two-file model (this is a one-file model)")
    else:
        emb = dict(ai.get("embedder") or {})
        if embedder is not None:
            if os.path.isdir(str(embedder)):
                emb = {"path": os.path.abspath(str(embedder)), "repo": None, "revision": None}
            else:
                rev = embedder_revision
                if rev is None and str(embedder) == BGE_M3_REPO:
                    rev = BGE_M3_REVISION          # the pinned revision of every registered number
                    log.info(f"embedder {BGE_M3_REPO}: using the pinned revision {BGE_M3_REVISION}")
                elif rev is None:
                    warnings.warn(f"embedder {embedder!r} without embedder_revision: the current 'main' is downloaded "
                                  "(pin a commit for reproducible numbers)", stacklevel=3)
                emb = {"repo": str(embedder), "revision": rev, "path": None}
        if not emb:
            emb = {"repo": BGE_M3_REPO, "revision": BGE_M3_REVISION, "path": None}
        if emb.get("path") and not os.path.isabs(emb["path"]):
            emb["path"] = os.path.normpath(os.path.join(root, emb["path"]))
        emb.setdefault("max_len", 512)
        msg_len = lab_len = int(emb.get("max_len") or 512)
    by_mode = _fusion_by_mode(ai.get("fusion_by_mode"), mode, (w_v, k_v, tau_v), {"w": w, "k": k, "tau": tau})
    fs_block = _fewshot_block(ai, meta, layout, msg_len, lab_len, pair_max_len, mf)
    cfg = ModelConfig(layout=layout, root=root, cross_dir=cross_dir, w=w_v, k=k_v, tau=tau_v, cascade_first=first,
                      msg_max_len=_int(msg_len, "msg_max_len"), label_max_len=_int(lab_len, "label_max_len"),
                      pair_max_len=_int(pair_max_len, "pair_max_len"), label_text_mode=mode, v3s_variant=v3s_variant,
                      adapter=adapter, embedder=emb, model_license=meta.get("license"), version=version,
                      meta_file=meta_file,
                      layout_reason=reason, overridden=overridden, fusion_by_mode=by_mode, tau_by_k=tau_by_k,
                      fewshot=fs_block, meta=meta)
    log.info(f"anyintent: {layout} model ({reason}); w {w_v}, k {k_v}, tau {tau_v}, cascade_first {first}, "
             f"label_text_mode {mode}")
    return cfg


def _tau_by_k(raw, k_frozen, tau_frozen, check=True):
    """Cascade thresholds checked per k, {k: tau or null} (stage-1.2 tau_by_k; DECISIONS 2.6, B-2). The entry of the
    frozen k must equal the frozen tau (unless k or tau was overridden at load)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise AnyIntentError("tau_by_k: expected {k: tau or null}")
    out = {_int(kk, "tau_by_k key"): (None if v is None else float(v)) for kk, v in raw.items()}
    if check and k_frozen in out and out[k_frozen] != tau_frozen:
        raise AnyIntentError(f"tau_by_k[{k_frozen}] = {out[k_frozen]} differs from the frozen tau {tau_frozen}")
    return out


def _fusion_by_mode(block, frozen_mode, frozen, overrides):
    """anyintent.json "fusion_by_mode": {mode: {"w", "k", "tau"}}. A block for the frozen mode must equal the main
    values; keyword overrides (w=, k=, tau= at load) apply to every mode."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise AnyIntentError("anyintent.json fusion_by_mode: expected {mode: {w, k, tau}}")
    out = {}
    for m, b in block.items():
        if m not in MODES:
            raise AnyIntentError(f"anyintent.json fusion_by_mode: unknown label_text_mode {m!r} (expected one of {MODES})")
        if not isinstance(b, dict) or b.get("w") is None or b.get("k") is None:
            raise AnyIntentError(f"anyintent.json fusion_by_mode.{m}: needs w and k (and tau, null for no cascade)")
        v = {"w": float(b["w"]), "k": _int(b["k"], f"fusion_by_mode.{m}.k"),
             "tau": None if b.get("tau") is None else float(b["tau"])}
        for key in ("w", "k", "tau"):
            if overrides.get(key) is not None:
                v[key] = float(overrides[key]) if key != "k" else _int(overrides[key], "k")
        if m == frozen_mode:
            if (v["w"], v["k"], v["tau"]) != tuple(frozen):
                raise AnyIntentError(f"anyintent.json fusion_by_mode.{m} {v} differs from the main fusion values of the "
                                     f"frozen mode {frozen}")
            continue
        out[m] = v
    return out


def _fewshot_block(ai, meta, layout, msg_len, lab_len, pair_len, meta_file=META_FILE):
    """The optional few-shot block, validated with the runtime's validate_config; the max lengths it records (if any)
    must equal the model's."""
    src = "anyintent.json" if isinstance(ai.get("fewshot"), dict) else meta_file
    raw = ai.get("fewshot") if isinstance(ai.get("fewshot"), dict) else meta.get("fewshot")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AnyIntentError(f"{src} fewshot: expected a block with variant, agg, a, w, k")
    if layout != "one_file":
        raise AnyIntentError(f"{src} fewshot: the few-shot mode is defined for one-file models only")
    try:
        c = fs.validate_config(raw)
    except (ValueError, TypeError, KeyError) as e:
        raise AnyIntentError(f"{src} fewshot block invalid: {type(e).__name__}: {e}")
    if c["variant"] == "Z":
        raise AnyIntentError(f"{src} fewshot: variant Z is the zero-shot reference, not a few-shot mode")
    for key, own in (("msg_max_len", msg_len), ("label_max_len", lab_len), ("pair_max_len", pair_len)):
        if raw.get(key) is not None and own is not None and int(raw[key]) != int(own):
            raise AnyIntentError(f"{src} fewshot.{key} {raw[key]} differs from the model's {own}")
    c["source"] = src
    c["n_set"] = raw.get("n_set")
    return c
