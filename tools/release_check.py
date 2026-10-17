"""Release check of a packaged R2Intent model with the anyintent library (a tool of the project that trained the
models; not part of the wheel, which packages src/ only).

What it checks, on DEV data only (review item 1 of docs/work/stage12/library_REVIEW.md):
- inputs, fail closed: the FINAL decision file, the released model.safetensors, the data and ML manifests and the
  stored DEV score frame, each found by name exactly once under its staged folder and equal to its pinned sha256; any
  path with "test" in it is refused (the search does not even enter such folders);
- the configuration the library resolves from the released folder (w, k, tau, cascade_first, label_text_mode) equals
  the values of the pre-release PROTOCOL entry (EXPECT_CONFIG), w, tau and the label text mode equal the decision's,
  and k is one of the decision's per_k keys;
- the label-text composition of the library equals the data's label_text / label_text_noex columns;
- for every engine and precision of this device (CUDA: eager, fast and auto in fp16; CPU: eager and fast in fp32 and
  ONNX Runtime fp32, int8 as information), on the first 64 decision rows (not near_test) of clinc_dev, clinc_dev@lt
  and clinc_dev@pl, against the stored DEV scores of the decision: candidate sets equal except cosine near ties
  (2e-3 at the last place), cosines of the stored candidates within 5e-3, cross scores within max(0.05, 1% of the
  largest stored logit) (the step-1 consistency rule), fused top-1 at the frozen (w, k) and with the frozen cascade
  in agreement with the stored scores on at least 97% of the rows (the v1.1 save check);
- integration fix I4 (AUDIT_FIXES L01, the registered bar of the release check: GPU fp16 over clinc_dev, top-1 agreement
  of at least 0.99 with the stored DEV scores at the shipped k, w and cascade): every row of clinc_dev that the stored
  DEV frame holds ("clinc_dev full": the decision rows and the rows flagged near_test, 1,800 for the registered DEV) with
  the same checks and a fused top-1 agreement of at least 99% (AGREE_MIN_FULL) at the resolved (w, k) and with the
  frozen cascade, on every gating engine of a GPU (and of the smoke); a CPU run scores it on its first engine only with
  full_row_on_cpu, and a run in which no gating engine scored the full row is never PASS (INCOMPLETE);
- PIN_BY_SOURCE_TENSORS (setting pin_by_source_tensors, off by default): a released file that the decision does not pin
  and no package record chain pins is accepted when every tensor equals the decision-pinned cross-encoder and adapter
  files (the source tensor check, then mandatory): the honest model Z repackaged under the R2Intent names, whose
  stage-1.2 decision pins the files the look evaluated;
- the R2Intent names (PROTOCOL E5 (10f), decision D6): the released folder's meta file is
  r2intent_meta.json, its adapter prefix r2intent_embed_lora. (the library reads no other names); the source tensor
  check maps the adapter keys of the decision-pinned adapter file (under the released prefix or under one of the
  setting training_prefixes, the prefixes of the project's training files) to the released prefix; the staged folders
  are named <staged_prefix>-<kind> (setting staged_prefix, required); the decision of the P job (FINAL pub, configs
  p_full and p0) and of a WiSE job pins the released file through its package files (configs.<arm>.package.files_sha256,
  package of a WiSE record), so the release folder r2intent-v1 of the PACK build is pinned by the decision;
- information: per-message latency (p50, p90) of each engine at k = 3, 5, 8 and with the cascade, and the GPU names.
Round 1 of the second cycle (library_REVIEW.md of 2026-10-06):
- every file of the released folder is hashed into the result (released_files), and anyintent.json must equal the
  pasted EXPECT_ANYINTENT_JSON_SHA256 (None: the file must be absent);
- per-mode settings (anyintent.json fusion_by_mode, for example the v2 settings of a v3s release) must equal the
  pasted EXPECT_CONFIG["fusion_by_mode"] and the stage-1.2 decision (STAGE12_STAGED, bound by the FINAL decision's
  stage12_decision_sha256), and are compared with the stored DEV scores of that mode when the frame holds them;
- k is checked against the per_k keys of the FINAL decision, else of the stage-1.2 decision, else k in (3, 5, 8);
- v3s label texts are checked for composition like v2 and noex;
- a few-shot block (fewshot) must equal EXPECT_CONFIG["fewshot"] and the few-shot DEV decision (FEWSHOT_STAGED,
  EXPECT_FEWSHOT_DECISION_SHA256); the library's few-shot predictions on clinc_dev are compared with the registered
  few-shot runtime (tests/reference/fewshot_runtime_ref.py) run on the released file, with FEWSHOT_N examples
  per label taken from other clinc_dev decision rows;
- a session deadline cut gives the status INCOMPLETE (never PASS; re-run), and the latency rows are shed when the
  projection after the first engine exceeds 80% of the deadline.

Entry points: run_check(settings) (used by kaggle/library_release/library_release.ipynb), write_pins(...) (the L1 step
that pins k, tau or the label text mode in anyintent.json when the PROTOCOL values differ from the released meta),
smoke_inputs(...) (generated stand-ins for the CPU smoke). Command line:
  python3 tools/release_check.py pins <release folder> [--w W] [--k K] [--tau TAU] [--cascade-first N] [--mode MODE]
      [--mode-settings v2:W,K,TAU] [--fewshot-json '{"variant": ..., "k": ...}']
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
import traceback

import numpy as np

POOLS = ("clinc_dev", "clinc_dev@lt", "clinc_dev@pl")
N_ROWS = 64
NEAR_TIE_COS = 2e-3      # candidate sets equal except a cosine near tie at the last place (step-1 consistency rule)
COS_TOL = 5e-3           # stored vs re-computed cosines of the stored candidates (v1.1 save check reload_rescore_64)
AGREE_MIN = 0.97         # fused top-1 agreement with the stored scores (v1.1 save check runtime_agreement_64)
FULL_POOL = "clinc_dev"  # integration fix I4 (AUDIT_FIXES L01): every decision row of this pool on the first engine
AGREE_MIN_FULL = 0.99    # its fused top-1 agreement (and cascade agreement) threshold
NEAR_FUSED = 1e-3        # stored fused margin below this: a disagreement is counted as a near tie (information)
EXPECT_KEYS = ("w", "k", "tau", "cascade_first", "label_text_mode")
FEWSHOT_KEYS = ("variant", "agg", "a", "w", "k", "lse_tau", "m_text", "ex_word_cap")
K_ALLOWED = (3, 5, 8)      # the registered k grid (used when no per_k table is available)
SHED_AT = 0.8              # latency rows are dropped when the projection exceeds this share of the deadline
# The names of the project that trained the models are settings, not constants of this tool: its staged folders
# (Kaggle datasets and kernel outputs) are named <staged_prefix>-<kind> (setting staged_prefix, required; run_check
# refuses a run without it), and its training files may hold the adapter tensors under other prefixes than the
# released one (setting training_prefixes, used by the source tensor check). The release notebook passes the project's
# values; the smoke stand-ins below use the neutral names SMOKE_STAGED_PREFIX and SMOKE_TRAINING_PREFIX.
SMOKE_STAGED_PREFIX = "project"
SMOKE_TRAINING_PREFIX = "training_embed_lora."
STAGED_PREFIX_RE = re.compile(r"[a-z0-9][a-z0-9_]*")
STAGE12_STAGED_KINDS = ("stage12", "stage12-seeds")
FEWSHOT_STAGED_KINDS = ("fewshot-dev-s12", "fewshot-dev-v11")
FEWSHOT_N = 5
SHA_LEN = 64
# staged folders that may hold the decision (FINAL outputs; the stage-1.2 outputs for the honest model, plan section 6)
RELEASE_STAGED_KINDS = ("final-pub", "final-dom", "stage12", "stage12-seeds")
DECISION_GLOBS = ("final_decision_*.json", "stage12_decision_*.json")
R2_PREFIX = "r2intent_embed_lora."                 # the adapter prefix of a released one-file model (D6)
META_FILE = "r2intent_meta.json"


def staged_name(kind, prefix=SMOKE_STAGED_PREFIX):
    """the name of a staged folder (a Kaggle dataset or kernel output): <prefix>-<kind>, kind for example final-pub,
    stage12, data-pub; prefix is the setting staged_prefix of the run (the smoke stand-ins' by default)"""
    return f"{prefix}-{kind}"


def staged_names(prefix, kinds):
    """the staged folder names of the kinds under the prefix"""
    return tuple(staged_name(k, prefix) for k in kinds)


def staged_prefix(S):
    """the setting staged_prefix (lower-case letters, digits and underscores); Refused when it is missing"""
    pre = S.get("staged_prefix")
    if not isinstance(pre, str) or not STAGED_PREFIX_RE.fullmatch(pre):
        raise Refused(f"STAGED_PREFIX {pre!r}: the settings must name the prefix of the staged folders "
                      "(staged_prefix: lower-case letters, digits and underscores)")
    return pre


def training_prefixes(S):
    """the setting training_prefixes: adapter prefixes (each ending in '.') other than the released one; () when unset"""
    tp = S.get("training_prefixes") or ()
    if isinstance(tp, str) or not all(isinstance(p, str) and p.endswith(".") and p != R2_PREFIX for p in tp):
        raise Refused(f"TRAINING_PREFIXES {tp!r}: a list of adapter prefixes, each ending in '.', other than "
                      f"{R2_PREFIX}")
    return tuple(tp)


def release_meta(folder):
    """(meta dict, meta file name) of a released folder: r2intent_meta.json; ({}, None) without it"""
    p = os.path.join(folder, META_FILE)
    if not os.path.exists(p):
        return {}, None
    with open(p, encoding="utf-8") as f:
        return json.load(f), META_FILE


class Refused(RuntimeError):
    """An input is missing, ambiguous, unpinned, a test file, or differs from its pinned sha256."""


# ---- inputs ---------------------------------------------------------------------------------------------------------
def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _is_sha(x):
    return isinstance(x, str) and len(x) == SHA_LEN and all(c in "0123456789abcdef" for c in x)


def refuse_test(path):
    if "test" in os.path.basename(os.path.normpath(str(path))).lower():
        raise Refused(f"refused: {path} (a name with 'test' in it; this check reads DEV data only)")


def _walk(root, max_depth):
    for dp, dn, fn in os.walk(root):
        depth = 0 if dp == root else os.path.relpath(dp, root).count(os.sep) + 1
        dn[:] = sorted(d for d in dn if "test" not in d.lower() and not d.startswith("."))
        if depth >= max_depth:
            dn[:] = []
        yield dp, dn, fn


def staged_dir(input_root, name):
    """The staged folder `name`: input_root/name, else the one folder of that name at depth <= 3 (Kaggle mounts
    datasets at /kaggle/input/<slug> and kernel outputs at /kaggle/input/notebooks/<user>/<slug>)."""
    refuse_test(name)
    direct = os.path.join(input_root, name)
    if os.path.isdir(direct):
        return direct
    hits = [os.path.join(dp, d) for dp, dn, _ in _walk(input_root, 3) for d in dn if d == name]
    if len(hits) != 1:
        raise Refused(f"staged folder {name!r}: expected exactly one under {input_root}, found {len(hits)}: {hits[:3]}")
    return hits[0]


def find_file(root, name):
    """The one file `name` under root (folders with 'test' in their name are not searched)."""
    refuse_test(name)
    hits = [os.path.join(dp, name) for dp, _, fn in _walk(root, 6) if name in fn]
    if len(hits) != 1:
        raise Refused(f"input {name}: expected exactly one file under {root}, found {len(hits)}: {hits[:3]}")
    return hits[0]


def find_glob(root, pattern):
    hits = sorted(p for dp, _, fn in _walk(root, 6) for p in glob.glob(os.path.join(dp, pattern)))
    hits = [h for h in hits if "test" not in os.path.basename(h).lower()]
    if len(hits) != 1:
        raise Refused(f"input {pattern}: expected exactly one file under {root}, found {len(hits)}: {hits[:3]}")
    return hits[0]


def check_sha(path, want, what):
    got = sha256_file(path)
    if not _is_sha(want):
        raise Refused(f"{what}: no pinned sha256 (got {want!r}); paste it from the PROTOCOL entry")
    if got != want:
        raise Refused(f"{what}: sha256 {got} differs from the pinned {want} ({path})")
    return got


FINAL_CONFIG_OF = {"pub": "P", "dom": "D"}   # stage-1.2 FINAL: the trained config of each final_data (cells_s12_part5)


def read_final_decision(path, config_name=None):
    """(config name, config dict, whole decision). Tolerant: a single config in "configs", the "adopted" one, or the
    config keys at the top level (the forms the testlook reader accepts). With several configs and "adopted" not
    among them (stage-1.2 FINAL pub writes configs P and Z and keeps the stage-1.2 name, for example HN12, in
    "adopted"), the config is the one of "final_data" (pub: P, dom: D); an explicit config_name (setting CONFIG_NAME)
    must name one of the configs. Anything else is refused as ambiguous."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cs = d.get("configs")
    if isinstance(cs, dict) and cs:
        if config_name:
            if config_name not in cs:
                raise Refused(f"{path}: CONFIG_NAME {config_name!r} is not one of the configs {sorted(cs)}")
            name = config_name
        elif len(cs) == 1:
            name = next(iter(cs))
        elif d.get("adopted") in cs:
            name = d["adopted"]
        elif FINAL_CONFIG_OF.get(d.get("final_data")) in cs:
            name = FINAL_CONFIG_OF[d["final_data"]]
        else:
            raise Refused(f"{path}: several configs {sorted(cs)}, adopted {d.get('adopted')!r} is not one of them and "
                          f"final_data {d.get('final_data')!r} names none (set CONFIG_NAME)")
        cfg = cs[name]
        if not isinstance(cfg, dict):
            raise Refused(f"{path}: config {name!r} is not a record")
    else:
        name, cfg = d.get("adopted") or "final", d
    return name, cfg, d


def decision_shas(cfg):
    out = []
    for key in ("cross_sha256", "adapter_sha256", "model_sha256", "sha256"):
        v = cfg.get(key)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, dict):
                    out += [y for y in x.values() if isinstance(y, str)]
    return [x for x in out if _is_sha(x)]


def _pinned(cfg, what):
    """The decision's sha256 of the 'cross' or 'adapter' file (keys <what>_sha256 or sha256.<what>). The stage-1.2
    writer pins the cross-encoder folder as a dict {file name: sha256} (weight_shas): its model.safetensors entry,
    else its only entry."""
    v = cfg.get(f"{what}_sha256")
    if not _is_sha(v) and isinstance(cfg.get("sha256"), dict):
        v = cfg["sha256"].get(what)
    if isinstance(v, dict):
        shas = {k: x for k, x in v.items() if _is_sha(x)}
        v = shas.get("model.safetensors") or (next(iter(shas.values())) if len(shas) == 1 else None)
    return v if _is_sha(v) else None


def find_package_record(rel_root):
    """The stage-1.2 FINAL package record final_package_<run>.json next to the decision: (path, record) when found
    exactly once, (None, None) when absent; several are refused."""
    hits = sorted(p for dp, _, fn in _walk(rel_root, 6) for p in glob.glob(os.path.join(dp, "final_package_*.json")))
    hits = [h for h in hits if "test" not in os.path.basename(h).lower()]
    if len(hits) > 1:
        raise Refused(f"package record final_package_*.json: expected at most one under {rel_root}, found {hits[:3]}")
    if not hits:
        return None, None
    with open(hits[0], encoding="utf-8") as f:
        return hits[0], json.load(f)


def package_chain(rec_path, rec, dec, dec_sha, model_sha, rel, expect_rec_sha=None):
    """Second accepted pin of the released file (final check N1 (ii)), used only when the decision itself does not
    pin it: the chain decision (sha256 pasted from the PROTOCOL entry) -> package record (its final_record_sha256 is
    the decision's sha256) -> the record's files["model.safetensors"] (equal to the released file, whose sha256 also
    equals the pasted EXPECT_MODEL_SHA256) -> the packaged r2intent_meta.json (its final_record_sha256 is the decision's
    sha256). Returns (ok, reasons, info). The caller then requires the source tensor check (mandatory on this path)."""
    if rec is None:
        return False, ["no final_package_*.json next to the decision"], {}
    info = {"package_record": os.path.basename(rec_path), "package_record_sha256": sha256_file(rec_path)}
    bad = []
    if expect_rec_sha is not None and info["package_record_sha256"] != expect_rec_sha:
        bad.append(f"package record sha256 {info['package_record_sha256'][:16]} differs from EXPECT_PACKAGE_RECORD_SHA256")
    if rec.get("final_record_sha256") != dec_sha:
        bad.append(f"package record final_record_sha256 {str(rec.get('final_record_sha256'))[:16]} is not the decision's "
                   f"{dec_sha[:16]}")
    if dec.get("package") is not True:
        bad.append(f"the decision says package {dec.get('package')!r}")
    if rec.get("ok") is not True:
        bad.append(f"the package record says ok {rec.get('ok')!r} ({str(rec.get('error') or rec.get('checks'))[:200]})")
    files = rec.get("files") if isinstance(rec.get("files"), dict) else {}
    if files.get("model.safetensors") != model_sha:
        bad.append(f"the package record lists model.safetensors {str(files.get('model.safetensors'))[:16]}, the released "
                   f"file is {model_sha[:16]}")
    if rec.get("dir") and os.path.basename(os.path.normpath(str(rec["dir"]))) != os.path.basename(os.path.normpath(rel)):
        bad.append(f"the package record names folder {os.path.basename(str(rec['dir']))!r}, the released folder is "
                   f"{os.path.basename(rel)!r}")
    meta, mname = release_meta(rel)
    if meta.get("final_record_sha256") != dec_sha:
        bad.append(f"the packaged {mname or 'meta file'} final_record_sha256 {str(meta.get('final_record_sha256'))[:16]} "
                   f"is not the decision's {dec_sha[:16]}")
    return not bad, bad, info


def _locate(path, roots, is_dir):
    """The staged copy of a file or folder the decision names by a path of the run machine: searched by its base
    name under the staged roots (folders with 'test' are not entered); the literal path only if nothing is staged."""
    if not path:
        return [], None
    base = os.path.basename(os.path.normpath(str(path)))
    refuse_test(base)
    hits, seen = [], set()
    for root in roots:
        for dp, dn, fn in _walk(root, 6):
            names = dn if is_dir else fn
            if base in names:
                p = os.path.realpath(os.path.join(dp, base))
                if p not in seen and (not is_dir or os.path.exists(os.path.join(p, "model.safetensors"))):
                    seen.add(p)
                    hits.append(p)
    if not hits and os.path.exists(str(path)) and (not is_dir or os.path.exists(os.path.join(str(path), "model.safetensors"))):
        hits = [os.path.realpath(str(path))]
    return hits, base


def tensors_equal_sources(packaged, cross_file, adapter_file, prefix=R2_PREFIX, training_prefixes=()):
    """Every tensor of the packaged one-file model.safetensors equals its source: keys outside the adapter prefix the
    cross-encoder file's tensor cast to the packaged dtype (the packaging casts the backbone and head to fp16), adapter
    keys the adapter file's tensor (fp32 in both; its keys under R2_PREFIX or one of training_prefixes); no key missing
    on either side."""
    import torch
    from safetensors import safe_open
    out = {"n_backbone": 0, "n_adapter": 0, "different": [], "missing_in_source": [], "only_in_source": []}
    with safe_open(packaged, framework="pt") as fp, safe_open(cross_file, framework="pt") as fc, \
            safe_open(adapter_file, framework="pt") as fa:
        kc = set(fc.keys())

        def _norm(k):      # the adapter file's keys under the packaged prefix (from R2_PREFIX or a training prefix)
            for pre in (R2_PREFIX,) + tuple(training_prefixes):
                if k.startswith(pre):
                    return prefix + k[len(pre):]
            return prefix + k
        ka = {_norm(k): k for k in fa.keys()}
        kp = set(fp.keys())
        for k in sorted(kp):
            tp = fp.get_tensor(k)
            if k.startswith(prefix):
                if k not in ka:
                    out["missing_in_source"].append(k)
                    continue
                ts = fa.get_tensor(ka[k])
                out["n_adapter"] += 1
            else:
                if k not in kc:
                    out["missing_in_source"].append(k)
                    continue
                ts = fc.get_tensor(k)
                out["n_backbone"] += 1
            if ts.is_floating_point() and tp.is_floating_point():
                ts = ts.to(tp.dtype)
            if ts.shape != tp.shape or ts.dtype != tp.dtype or not torch.equal(ts, tp):
                out["different"].append(k)
        only = sorted((kc - kp) | (set(ka) - kp))
        out["only_in_source"] = only[:10]
        out["n_only_in_source"] = len(only)
    out["different"], out["missing_in_source"] = out["different"][:10], out["missing_in_source"][:10]
    out["equal"] = not out["different"] and not out["missing_in_source"] and not only and out["n_backbone"] > 0
    return out


def source_tensors_check(wfile, mcfg, cfg, roots, training_prefixes=()):
    """Additional identity check of the released file against the decision-pinned cross-encoder and adapter files
    (does not replace the pin of the released file itself). Returns {"checked": bool, "equal": bool, ...}."""
    roots = list(dict.fromkeys(os.path.realpath(r) for r in roots if r))
    if mcfg.layout != "one_file":
        return {"checked": False, "reason": "two-file layout: the released cross-encoder file is pinned directly"}
    want_c, want_a = _pinned(cfg, "cross"), _pinned(cfg, "adapter")
    if not want_c or not want_a:
        return {"checked": False, "reason": "the decision pins no separate cross and adapter sha256"}
    if sha256_file(wfile) in (want_c, want_a):
        return {"checked": False, "reason": "the released file is itself the pinned cross or adapter file"}
    cdirs, cbase = _locate(cfg.get("cross_dir"), roots, True)
    afiles, abase = _locate(cfg.get("adapter_path"), roots, False)
    if len(cdirs) > 1 or len(afiles) > 1:
        raise Refused(f"source files: cross folder {cbase!r} found {len(cdirs)} times, adapter file {abase!r} "
                      f"{len(afiles)} times under the staged folders (expected at most once)")
    if not cdirs or not afiles:
        return {"checked": False, "reason": f"not staged: cross folder {cbase!r} {len(cdirs)}, adapter {abase!r} "
                                            f"{len(afiles)} (information; the pin of the released file still applies)"}
    cfile = os.path.join(cdirs[0], "model.safetensors")
    check_sha(cfile, want_c, "decision-pinned cross-encoder file")
    check_sha(afiles[0], want_a, "decision-pinned adapter file")
    prefix = (mcfg.adapter or {}).get("prefix") or R2_PREFIX
    r = tensors_equal_sources(wfile, cfile, afiles[0], prefix, training_prefixes)
    r.update(checked=True, cross_file=os.path.join(os.path.basename(cdirs[0]), "model.safetensors"),
             adapter_file=os.path.basename(afiles[0]))
    return r


def record_package_shas(dec, cname, cfg):
    """the released file's pins of a FINAL pub record (PROTOCOL E5 (8), (9)): the P job's and a rerun's configs.<arm>
    .package.files_sha256 (written by the worker, model.safetensors), and the package of a WiSE record whose model is
    the config"""
    out = []
    pk = cfg.get("package") if isinstance(cfg.get("package"), dict) else {}
    v = (pk.get("files_sha256") or {}).get("model.safetensors")
    if _is_sha(v):
        out.append(v)
    wp = dec.get("package") if isinstance(dec.get("package"), dict) else {}
    if wp and wp.get("model") == cname:
        v = (wp.get("files_sha256") or ((wp.get("checks") or {}).get("files_sha256")) or {}).get("model.safetensors")
        if _is_sha(v):
            out.append(v)
    return out


def per_k_keys(d, name):
    pk = d.get("per_k")
    if isinstance(pk, dict):
        sub = pk.get(name, pk if all(str(x).isdigit() for x in pk) else None)
        if isinstance(sub, dict):
            return sorted(int(x) for x in sub if str(x).isdigit())
    return None


def manifest_sha(manifest, name):
    v = (manifest.get("files") or {}).get(name)
    return v.get("sha256") if isinstance(v, dict) else (v if isinstance(v, str) else None)


def label_column(LAB, mode, variant):
    if mode == "v2":
        cols = ["label_text"]
    elif mode == "noex":
        cols = ["label_text_noex"]
    elif mode == "v3s":
        if variant not in ("A", "B"):
            raise Refused(f"label_text_mode v3s needs v3s_variant A or B in the decision (got {variant!r})")
        cols = [f"label_text_v3s_{variant}"]
    elif mode == "name":
        cols = ["name", "label_name"]
    else:
        raise Refused(f"label_text_mode {mode!r} unknown")
    for c in cols:
        if c in LAB.columns:
            return c
    raise Refused(f"the label file has none of the columns {cols} (label_text_mode {mode})")


def load_pools(data_dir, ml_dir, SC, sc_variant, mode, variant, n_rows, man, man_ml, pools_wanted=None, all_rows=False):
    """Rows, messages, label texts and stored arrays of the first n_rows decision rows of every pool in POOLS (with
    all_rows: every row of the stored frame, the rows flagged near_test included). Also returns, per English pool, the
    label names and descriptions (from the mode's column) and the other decision rows of the pool (the example pool of
    the few-shot row)."""
    import pandas as pd
    paths = {}
    for name, root, mf in (("step1_dev.parquet", data_dir, man), ("step1_labels.parquet", data_dir, man),
                           ("step1_dev_ml.parquet", ml_dir, man_ml)):
        p = find_file(root, name)
        check_sha(p, manifest_sha(mf, name), f"{name} (pinned by its manifest)")
        paths[name] = p
    DV = pd.read_parquet(paths["step1_dev.parquet"])
    LAB = pd.read_parquet(paths["step1_labels.parquet"]).drop_duplicates("label_id").set_index("label_id")
    DVML = pd.read_parquet(paths["step1_dev_ml.parquet"])
    col = label_column(LAB, mode, variant)
    nt_en = DV.drop_duplicates("row_id").set_index("row_id").near_test.astype(bool).to_dict() \
        if "near_test" in DV.columns else {}
    out, comp = {}, {}
    for pool in pools_wanted or POOLS:
        base, _, lang = pool.partition("@")
        en = DV[DV.pool == base]
        if not len(en):
            raise Refused(f"DEV pool {base} not in step1_dev.parquet")
        ids = sorted(en.label_id.unique())
        texts = [str(LAB.at[l, col]) for l in ids]
        if lang:
            rows = DVML[(DVML.pool == base) & (DVML.lang == lang)].copy()
            src = pd.to_numeric(rows["src_row_id"], errors="coerce") if "src_row_id" in rows else None
            rows["near_test"] = [bool(nt_en.get(int(s), False)) if pd.notna(s) else False for s in src] \
                if src is not None else False
        else:
            rows = en.copy()
            rows["near_test"] = rows.near_test.astype(bool) if "near_test" in rows else False
        rows = rows.drop_duplicates("row_id").set_index("row_id")
        fr = SC[(SC.pool == pool) & (SC.variant == sc_variant)].sort_values("row_id")
        if not len(fr):
            raise Refused(f"the stored DEV scores have no rows of pool {pool} (variant {sc_variant})")
        missing = sorted(set(fr.row_id) - set(rows.index))
        if missing:
            raise Refused(f"pool {pool}: {len(missing)} stored rows are not in the DEV data (e.g. {missing[:3]})")
        dec = ~rows.loc[fr.row_id, "near_test"].to_numpy(dtype=bool)
        n_near = int((~dec).sum())
        fr = fr if all_rows else fr[dec].iloc[:n_rows]
        li = {l: i for i, l in enumerate(ids)}
        try:
            cand = np.array([[li[x] for x in r] for r in fr.cand_label_ids], dtype=np.int64)
        except KeyError as e:
            raise Refused(f"pool {pool}: a stored candidate {e} is not a label of the pool")
        name_col = "label_name" if "label_name" in LAB.columns else ("name" if "name" in LAB.columns else None)
        names = [str(LAB.at[l, name_col]) if name_col else str(t_) for l, t_ in zip(ids, texts)]
        rest = rows[~rows.near_test.astype(bool) & ~rows.index.isin(fr.row_id)].sort_index()
        out[pool] = {"row_ids": fr.row_id.tolist(), "msgs": rows.loc[fr.row_id, "text"].astype(str).tolist(),
                     "names": names, "other_rows": [(str(r_.label_id), str(r_.text)) for r_ in rest.itertuples()]
                     if "label_id" in rest.columns else [],
                     "texts": texts, "label_ids": ids, "cand": cand,
                     "retr": np.array([list(r) for r in fr.retr_scores], dtype=np.float32),
                     "cross": np.array([list(r) for r in fr.cross_scores], dtype=np.float32),
                     "n_stored": int(len(fr)), "n_decision_total": int(dec.sum()), "n_near_test": n_near}
        if lang == "":
            comp[base] = composition(LAB, ids, variant if mode == "v3s" else None)
    return out, comp, {k: sha256_file(v) for k, v in paths.items()}, col


def split_label_text(name, text):
    """(name, description) of a composed label text "name: description" (the name alone: empty description); None when
    the text does not start with the name."""
    name, text = str(name), str(text)
    if text == name:
        return name, ""
    if text.startswith(name + ": "):
        return name, text[len(name) + 2:]
    return None


def composition(LAB, ids, variant=None):
    """The library's label texts from (name, description) against the data's label_text and label_text_noex, and, for
    a v3s variant, label_text_v3s_<V> rebuilt from the name and the text after "name: " (review item 13)."""
    from anyintent.labels import label_texts
    name_col = "label_name" if "label_name" in LAB.columns else ("name" if "name" in LAB.columns else None)
    if name_col is None or "description" not in LAB.columns:
        return {"checked": False, "reason": "no name and description columns"}
    names = [str(LAB.at[l, name_col]) for l in ids]
    descs = ["" if LAB.at[l, "description"] is None or (isinstance(LAB.at[l, "description"], float))
             else str(LAB.at[l, "description"]) for l in ids]
    r = {"checked": True}
    for mode, c in (("v2", "label_text"), ("noex", "label_text_noex")):
        if c in LAB.columns:
            lib = label_texts([n.strip() for n in names], [d.strip() for d in descs], mode)
            r[mode] = int(sum(a != str(LAB.at[l, c]) for a, l in zip(lib, ids)))
    c = f"label_text_v3s_{variant}"
    if variant in ("A", "B") and c in LAB.columns:
        bad = 0
        for n, l in zip(names, ids):
            sp = split_label_text(n.strip(), LAB.at[l, c])
            if sp is None or label_texts([sp[0]], [sp[1].strip()], "v3s")[0] != str(LAB.at[l, c]):
                bad += 1
        r["v3s"] = bad
    return r


# ---- comparison -----------------------------------------------------------------------------------------------------
def compare_pool(clf, P, w, k, tau, first, mode, agree_min=AGREE_MIN):
    from anyintent import fusion
    import warnings
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # the v3s note: the texts below are complete already
        # complete label texts used as names (no composition; every mode gives the name alone back), in the mode whose
        # settings are compared, so that the library's per-mode (w, k, tau) are the ones in use
        clf.set_labels(P["texts"], label_text_mode=mode)
    active = clf.config["active"]
    if (active["w"], active["k"], active["tau"]) != (w, k, tau):
        return {"ok": False, "error": f"the library's settings for mode {mode} are {active}, expected {(w, k, tau)}"}
    if clf.texts != P["texts"]:
        return {"ok": False, "error": "the library changed the label texts (leading or trailing whitespace)"}
    be, msgs = clf.backend, P["msgs"]
    n, K = P["cand"].shape
    if k > K:
        return {"ok": False, "error": f"k {k} above the {K} stored candidates"}
    Q = be.embed(msgs, clf.cfg.msg_max_len)
    S = Q @ clf.L.T
    cand_n, _ = fusion.topk_stable(S, K)
    bad = 0
    for i in range(n):
        dif = set(cand_n[i].tolist()) ^ set(P["cand"][i].tolist())
        if dif and np.abs(S[i, list(dif)] - S[i, cand_n[i][-1]]).max() > NEAR_TIE_COS:
            bad += 1
    cos_diff = float(np.abs(np.take_along_axis(S, P["cand"], 1) - P["retr"]).max())
    mi = [i for i in range(n) for _ in range(K)]
    cross_n = np.asarray(be.cross_pairs(msgs, mi, P["cand"].reshape(-1).tolist()), dtype=np.float32).reshape(n, K)
    fin = np.isfinite(P["cross"])
    xd = float(np.abs(cross_n - P["cross"])[fin].max())
    tol = max(0.05, 0.01 * float(np.abs(P["cross"][fin]).max()))
    rows = np.arange(n)
    fz = fusion.fused_scores(P["cross"][:, :k], P["retr"][:, :k], w)
    p_ref = P["cand"][rows, fz.argmax(1)]
    near = np.nan_to_num(fusion.fused_margin(fz), nan=np.inf) < NEAR_FUSED
    res = clf.classify(msgs, k=k, cascade=False)
    p_lib = np.array([r.index for r in res])
    agree = float((p_lib == p_ref).mean())
    r = {"rows": n, "candidate_mismatch_rows": bad, "max_abs_cos_diff": cos_diff, "cos_tol": COS_TOL,
         "max_abs_cross_diff": xd, "cross_tol": tol, "fused_top1_agreement": agree,
         "flips_outside_near_ties": int(((p_lib != p_ref) & ~near).sum())}
    r["agree_min"] = agree_min
    ok = bad == 0 and cos_diff <= COS_TOL and xd <= tol and agree >= agree_min
    if tau is not None and min(first, k) < k:
        res_c = clf.classify(msgs, k=k, cascade=True)
        pos, stop = fusion.cascade_offline(P["cross"], P["retr"], w, tau, k, first)
        p_ref_c = P["cand"][rows, pos]
        p_lib_c = np.array([x.index for x in res_c])
        lib_stop = np.array([x.n_scored == min(first, k) for x in res_c])
        r.update(cascade_top1_agreement=float((p_lib_c == p_ref_c).mean()),
                 cascade_stop_agreement=float((lib_stop == stop).mean()), cascade_stopped_ref=int(stop.sum()))
        ok = ok and r["cascade_top1_agreement"] >= agree_min
    r["seconds"] = round(time.perf_counter() - t0, 2)
    r["ok"] = bool(ok)
    return r


def timing(clf, msgs, texts, ks=(3, 5, 8), cascade=False, mode=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.set_labels(texts, label_text_mode=mode)
    clf.warm_up(batch_sizes=(1,), ks=ks)
    out = {}
    plan = [(f"k{k}", k, False) for k in ks] + ([("cascade", None, True)] if cascade else [])
    for tag, k, casc in plan:
        t = []
        for m in msgs:
            t0 = time.perf_counter()
            clf.classify(m, k=k, cascade=casc)
            t.append((time.perf_counter() - t0) * 1000)
        out[tag] = {"n": len(t), "p50_ms": round(float(np.percentile(t, 50)), 2),
                    "p90_ms": round(float(np.percentile(t, 90)), 2)}
    return out


# ---- few-shot row ------------------------------------------------------------------------------------------------
def fewshot_reference_module():
    """The registered few-shot runtime (its copy with the R2Intent names in tests/reference of the package, next to
    tools/)."""
    import importlib.util
    p = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "reference",
                                      "fewshot_runtime_ref.py"))
    if not os.path.exists(p):
        raise Refused(f"the few-shot reference runtime {p} is missing from the package")
    spec = importlib.util.spec_from_file_location("fewshot_runtime_ref", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fewshot_inputs(P, n_per_label):
    """Labels (name and the description from the mode's label text), examples (the first n_per_label other decision
    rows of each label, by row_id; none for a label without such rows) and messages of the few-shot row."""
    from anyintent.labels import label_texts
    by = {}
    for lid, text in P["other_rows"]:
        lst = by.setdefault(lid, [])
        if len(lst) < n_per_label:
            lst.append(text)
    names, descs = [], []
    for n, t_ in zip(P["names"], P["texts"]):
        sp = split_label_text(n.strip(), t_)
        names.append(sp[0] if sp else str(t_))
        descs.append(sp[1].strip() if sp else "")
    texts = label_texts(names, descs, "v2")
    return {"labels": [{"name": n, "description": d} for n, d in zip(names, descs)], "names": names, "texts": texts,
            "examples": [by.get(l, []) for l in P["label_ids"]], "msgs": P["msgs"],
            "n_examples": int(sum(len(v) for v in by.values())), "labels_without_examples":
            int(sum(1 for l in P["label_ids"] if not by.get(l)))}


def fewshot_reference(rel, device, FP, block, msg_max_len):
    """cand, retr, cross, pred of the registered few-shot runtime on the released one-file model (fp16 autocast on
    CUDA, fp32 on the CPU, as the runtime decides)."""
    RF = fewshot_reference_module()
    rt = RF.FewShotRuntime(rel, device=device)
    rt.set_examples(FP["texts"], FP["examples"], names=FP["names"])
    msgs = FP["msgs"]
    Q = rt.embed(msgs, int(msg_max_len)).cpu().numpy().astype(np.float32)

    def cross_fn(pairs):
        if not pairs:
            return np.zeros(0, np.float32)
        return rt.cross([msgs[i] for i, _ in pairs], [t_ for _, t_ in pairs]).cpu().numpy()
    return RF.score_batch(Q, rt.Tn, dict(block), rt.En, rt.ex_label, rt.ex_texts, rt.noex, rt.labels, cross_fn, P=rt.Pn)


def compare_fewshot(clf, FP, ref, block):
    t0 = time.perf_counter()
    clf.set_labels(FP["labels"], label_text_mode="v2", examples=FP["examples"])
    if clf.texts != FP["texts"]:
        return {"ok": False, "error": "the library composed other label texts for the few-shot row"}
    res = clf.classify(FP["msgs"], return_scores=True)
    cand, retr, cross, pred = ref["cand"], ref["retr"], ref["cross"], ref["pred"]
    bad, rd, xd = 0, 0.0, 0.0
    for i, p in enumerate(res):
        lib = {c["index"]: c for c in p.candidates}
        row_bad = False
        for t_, j in enumerate(cand[i]):
            j = int(j)
            if j not in lib:
                row_bad = row_bad or abs(float(retr[i, t_]) - float(retr[i, -1])) > NEAR_TIE_COS
                continue
            rd = max(rd, abs(lib[j]["retrieval"] - float(retr[i, t_])))
            xd = max(xd, abs(lib[j]["cross"] - float(cross[i, t_])))
        bad += int(row_bad)
    agree = float(np.mean([p.index == int(pred[i]) for i, p in enumerate(res)]))
    tol = max(0.05, 0.01 * float(np.abs(cross).max())) if cross.size else 0.05
    r = {"rows": len(res), "candidate_mismatch_rows": bad, "max_abs_retrieval_diff": rd, "retrieval_tol": COS_TOL,
         "max_abs_cross_diff": xd, "cross_tol": tol, "fused_top1_agreement": agree, "k": int(block["k"]),
         "variant": block["variant"], "seconds": round(time.perf_counter() - t0, 2)}
    r["ok"] = bool(bad == 0 and rd <= COS_TOL and xd <= tol and agree >= AGREE_MIN)
    return r


def stage12_record(S, inp, dec, dglob, cname):
    """The stage-1.2 decision used for the per_k keys and the per-mode settings: the decision itself on the honest
    path; else the one in STAGE12_STAGED, whose sha256 must equal the FINAL decision's stage12_decision_sha256."""
    if dglob == "stage12_decision_*.json":
        return dec, cname, "the decision itself"
    name = S.get("stage12_staged")
    if not name:
        return None, None, "not staged (STAGE12_STAGED None)"
    allowed = staged_names(staged_prefix(S), STAGE12_STAGED_KINDS)
    if name not in allowed:
        raise Refused(f"STAGE12_STAGED {name!r}: expected one of {allowed}")
    p = find_glob(staged_dir(inp, name), "stage12_decision_*.json")
    want = dec.get("stage12_decision_sha256")
    check_sha(p, want, "stage-1.2 decision (pinned by the FINAL decision's stage12_decision_sha256)")
    with open(p, encoding="utf-8") as f:
        d12 = json.load(f)
    adopted = dec.get("adopted") or d12.get("adopted")
    if adopted not in (d12.get("configs") or {}):
        raise Refused(f"the stage-1.2 decision has no config {adopted!r} (the FINAL decision's adopted)")
    return d12, adopted, os.path.basename(p)


def mode_registered(d12, adopted, mode, variant):
    """(w, k, tau, per_k keys) the stage-1.2 decision registers for a label text mode of its adopted config: v2 from the
    config's "v2" block (when R8 adopted a v3s variant), its "per_mode" or the config itself, the v3s variant from
    r8.variants.<V>; None when the decision registers nothing for the mode."""
    c = (d12.get("configs") or {}).get(adopted) or {}
    r8v = ((d12.get("r8") or {}).get("variants") or {})
    if mode == "v3s":
        e = r8v.get(variant) or {}
        if c.get("label_text_mode") == "v3s" and c.get("v3s_variant") == variant:
            src = {"w": c.get("w"), "k": c.get("k"), "tau": c.get("tau")}
        else:
            src = {"w": e.get("w"), "k": e.get("k"), "tau": e.get("tau")} if e else None
        pk = e.get("per_k")
    elif mode == "v2":
        if isinstance(c.get("v2"), dict):
            src = c["v2"]
        elif isinstance((c.get("per_mode") or {}).get("v2"), dict):
            src = c["per_mode"]["v2"]
        elif c.get("label_text_mode", "v2") == "v2" and c.get("w") is not None:
            src = c
        else:
            src = r8v.get("v2")
        pk = (r8v.get("v2") or {}).get("per_k") or ((d12.get("per_k") or {}).get(adopted))
    else:
        return None
    if not src or src.get("w") is None:
        return None
    keys = sorted(int(x) for x in (pk or {}) if str(x).isdigit()) or None
    return {"w": float(src["w"]), "k": int(src["k"]), "tau": None if src.get("tau") is None else float(src["tau"]),
            "per_k_keys": keys}


def released_files(rel):
    out = {}
    for dp, dn, fn in os.walk(rel):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for f in sorted(fn):
            p = os.path.join(dp, f)
            out[os.path.relpath(p, rel)] = sha256_file(p)
    return out


# ---- the check --------------------------------------------------------------------------------------------------------
def environment():
    import torch
    import transformers
    import anyintent
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "transformers": transformers.__version__, "anyintent": anyintent.__version__,
           "cuda": bool(torch.cuda.is_available()), "gpus": []}
    if torch.cuda.is_available():
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    try:
        import onnxruntime
        env["onnxruntime"] = onnxruntime.__version__
    except Exception:
        env["onnxruntime"] = None
    return env


def default_runs(cuda, have_ort):
    if cuda:
        return [("eager", "float16", True), ("fast", "float16", True), ("auto", "float16", True)]
    runs = [("eager", "float32", True), ("fast", "float32", True)]
    if have_ort:
        runs += [("onnx", "fp32", True), ("onnx", "int8", False)]
    return runs


def run_check(S):
    """S: settings (see kaggle/library_release/library_release.ipynb). Returns the result dict; writes
    library_release_<run>.json and .txt into S["out_dir"]."""
    import pandas as pd
    import torch
    from anyintent import AnyIntent
    from anyintent.config import build_config
    t_start = time.time()
    deadline_s = float(S.get("deadline_h") or 0.75) * 3600
    run = S.get("run_name") or "release"
    R = {"run": run, "smoke": bool(S.get("smoke")), "status": "FAIL", "problems": [], "env": environment(),
         "settings": {k: v for k, v in S.items() if k not in ("expect_config",)},
         "expect_config": S.get("expect_config")}
    print(f"release check {run}: env {json.dumps(R['env'])}")
    print(f"recommended SLURM --time: {float(S.get('deadline_h') or 0.75) + 0.5:.2f} h (session deadline + 0.5 h)")
    try:
        inp = S["input_root"]
        pre, train_prefixes = staged_prefix(S), training_prefixes(S)
        rel_name = S.get("release_staged") or staged_name("final-pub", pre)
        allowed = staged_names(pre, RELEASE_STAGED_KINDS)
        if rel_name not in allowed:
            raise Refused(f"RELEASE_STAGED {rel_name!r}: expected one of {allowed}")
        rel_root = staged_dir(inp, rel_name)
        dglob = S.get("decision_glob") or "final_decision_*.json"
        if dglob not in DECISION_GLOBS:
            raise Refused(f"DECISION_GLOB {dglob!r}: expected one of {DECISION_GLOBS}")
        dec_path = find_glob(rel_root, dglob)
        R["decision_file"] = os.path.basename(dec_path)
        R["decision_sha256"] = check_sha(dec_path, S.get("expect_decision_sha256"), "decision file")
        cname, cfg, dec = read_final_decision(dec_path, S.get("config_name"))
        R["config_name"], R["shipped"] = cname, dec.get("shipped")
        if dec.get("status") not in (None, "FINAL") and not S.get("smoke"):
            raise Refused(f"decision status {dec.get('status')!r}: the release check needs a FINAL decision")
        if dec.get("package") is False:
            R["problems"].append("information: the decision says package false (model D, the honest model packaged "
                                 "elsewhere, or an unpackaged run)")
        # the released folder: under PACKAGE_STAGED when set (the honest model packaged apart from its decision),
        # else under the decision's staged folder; the decision's package_dir (interface addition) or RELEASE_SUBDIR
        # picks one when there are several
        pkg_root = staged_dir(inp, S["package_staged"]) if S.get("package_staged") else rel_root
        cands = []
        for dp, dn, fn in _walk(pkg_root, 4):
            if dp.endswith("_embedding_merged_optional"):
                continue
            meta = release_meta(dp)[0] if META_FILE in fn else {}
            if "anyintent.json" in fn or ("config.json" in fn and isinstance(meta.get("adapter"), dict)):
                cands.append(dp)
        # the stage-1.2 FINAL package record next to the decision (absent for the honest path and for a decision that
        # pins the packaged file itself); it may name the released folder
        pkg_rec_path, pkg_rec = (None, None) if S.get("package_staged") else find_package_record(rel_root)
        sub = S.get("release_subdir") or cfg.get("package_dir") or (
            os.path.basename(os.path.normpath(str((cfg.get("package") or {}).get("dir"))))
            if isinstance(cfg.get("package"), dict) and (cfg.get("package") or {}).get("dir") else None) or (
            pkg_rec or {}).get("dir")
        if sub:
            sub = os.path.normpath(str(sub))
            cands = [c for c in cands if os.path.normpath(c) == os.path.normpath(os.path.join(pkg_root, sub))
                     or (os.path.isabs(sub) and os.path.basename(c) == os.path.basename(sub))]
        if len(cands) != 1:
            raise Refused(f"released model folder: expected exactly one under {pkg_root}, found {cands[:4]} "
                          "(set RELEASE_SUBDIR)")
        rel = cands[0]
        mcfg = build_config(rel)
        wfile = os.path.join(mcfg.cross_dir, "model.safetensors")
        R["model_sha256"] = check_sha(wfile, S.get("expect_model_sha256"), "released model.safetensors")
        dshas = decision_shas(cfg) + record_package_shas(dec, cname, cfg)
        R["meta_file"] = mcfg.meta_file
        if not dshas:
            raise Refused("the decision pins no sha256 for the config (cross_sha256 / adapter_sha256 / sha256)")
        if R["model_sha256"] in dshas:
            R["pin"] = "decision"
        else:
            # second accepted pin (final check N1 (ii)): the sha256-bound chain decision -> package record ->
            # packaged meta; the refusal below stays for every other case, and this path makes the source tensor
            # check mandatory
            ok_chain, why, R["package_chain"] = package_chain(pkg_rec_path, pkg_rec, dec, R["decision_sha256"],
                                                              R["model_sha256"], rel,
                                                              S.get("expect_package_record_sha256"))
            if ok_chain:
                R["pin"] = "package record chain"
            elif S.get("pin_by_source_tensors"):
                # third accepted pin, only on request (the honest model Z repackaged under the R2Intent names, PROTOCOL
                # E3 and D6: its stage-1.2 decision pins the cross-encoder and adapter files the look evaluated, not the
                # repackaged one-file model): every tensor of the released file equals the decision-pinned files, which
                # makes the source tensor check below mandatory
                R["pin"] = "source tensors"
            else:
                raise Refused(f"the released model.safetensors ({R['model_sha256'][:16]}) is not a file the decision pins "
                              f"({[x[:16] for x in dshas]}), and the package record chain does not pin it: {why}")
        R["release_dir"], R["layout"] = os.path.relpath(rel, inp), mcfg.layout
        # additional identity check (on top of the pin above): when the decision-pinned cross-encoder and adapter
        # files are staged, the packaged tensors must equal them (backbone and head after the cast to the packaged
        # dtype, adapter exactly), as the v1.1 save check backbone_equals_source; a mismatch is refused
        R["source_tensors"] = source_tensors_check(wfile, mcfg, cfg, [rel_root, pkg_root] + [
            staged_dir(inp, s) for s in (S.get("source_staged") or [])], train_prefixes)
        print(f"source tensors: {json.dumps(R['source_tensors'])}")
        if R["source_tensors"].get("checked") and not R["source_tensors"].get("equal"):
            raise Refused(f"the released model.safetensors does not equal the decision-pinned cross-encoder and adapter "
                          f"files: {json.dumps(R['source_tensors'])[:600]}")
        if R["pin"] in ("package record chain", "source tensors") and not R["source_tensors"].get("checked"):
            raise Refused(f"the {R['pin']} pin needs the source tensor check, which did not run "
                          f"({R['source_tensors'].get('reason')}); stage the cross folder and adapter file the decision "
                          f"pins (SOURCE_STAGED)")
        # every released file is hashed into the result; anyintent.json is pinned by the PROTOCOL value (review item 7)
        R["released_files"] = released_files(rel)
        aj = os.path.join(rel, "anyintent.json")
        want_aj = S.get("expect_anyintent_json_sha256")
        if os.path.exists(aj):
            R["anyintent_json_sha256"] = check_sha(aj, want_aj, "anyintent.json of the released folder "
                                                                "(EXPECT_ANYINTENT_JSON_SHA256)")
        elif want_aj is not None:
            raise Refused("EXPECT_ANYINTENT_JSON_SHA256 is set but the released folder has no anyintent.json")
        # the stage-1.2 decision (per_k keys and per-mode settings)
        d12, adopted12, R["stage12_decision"] = stage12_record(S, inp, dec, dglob, cname)
        # data
        final_data = dec.get("final_data")
        data_name = S.get("data_staged") or {"pub": staged_name("data-pub", pre),
                                             "dom": staged_name("data-dom", pre)}.get(final_data)
        ml_name = S.get("ml_staged") or {"pub": staged_name("ml-pub", pre),
                                         "dom": staged_name("ml-dom", pre)}.get(final_data)
        if not data_name or not ml_name:
            raise Refused(f"final_data {final_data!r}: set DATA_STAGED and ML_STAGED")
        data_dir, ml_dir = staged_dir(inp, data_name), staged_dir(inp, ml_name)
        man_p, man_ml_p = find_file(data_dir, "step1_manifest.json"), find_file(ml_dir, "step1_ml_manifest.json")
        R["data_manifest_sha256"] = check_sha(man_p, S.get("expect_data_manifest_sha256"), "data manifest")
        R["ml_manifest_sha256"] = check_sha(man_ml_p, S.get("expect_ml_manifest_sha256"), "ML manifest")
        man, man_ml = json.load(open(man_p, encoding="utf-8")), json.load(open(man_ml_p, encoding="utf-8"))
        sc_name = os.path.basename(str(cfg.get("dev_scores") or cfg.get("dev_scores_file") or ""))
        if not sc_name:
            raise Refused("the decision names no dev_scores file for the config")
        refuse_test(sc_name)
        sc_path = None
        for root in (rel_root, inp):
            try:
                sc_path = find_file(root, sc_name)
                break
            except Refused:
                if root == inp:
                    raise
        want_sc = cfg.get("dev_scores_sha256") or S.get("expect_dev_scores_sha256")
        if cfg.get("dev_scores_sha256") and S.get("expect_dev_scores_sha256") and \
                cfg["dev_scores_sha256"] != S["expect_dev_scores_sha256"]:
            raise Refused("dev_scores_sha256 of the decision differs from EXPECT_DEV_SCORES_SHA256")
        R["dev_scores"], R["dev_scores_sha256"] = sc_name, check_sha(sc_path, want_sc, "stored DEV scores")
        SC = pd.read_parquet(sc_path)
        sc_variant = cfg.get("dev_scores_variant")
        if sc_variant is None:
            vs = sorted(SC.variant.unique())
            if len(vs) != 1:
                raise Refused(f"the stored DEV scores hold variants {vs}; the decision names none (dev_scores_variant)")
            sc_variant = vs[0]
        # configuration: library vs PROTOCOL vs decision
        exp = S.get("expect_config")
        if not isinstance(exp, dict) or any(k not in exp for k in EXPECT_KEYS):
            raise Refused(f"EXPECT_CONFIG must hold {EXPECT_KEYS} from the pre-release PROTOCOL entry (got {exp!r})")
        resolved = {"w": mcfg.w, "k": mcfg.k, "tau": mcfg.tau, "cascade_first": mcfg.cascade_first,
                    "label_text_mode": mcfg.label_text_mode}

        def same(a, b):
            if a is None or b is None:
                return a is None and b is None
            if isinstance(a, str) or isinstance(b, str):
                return str(a) == str(b)
            return abs(float(a) - float(b)) <= 1e-9

        cfg_bad = [f"{k}: library {resolved[k]!r} vs PROTOCOL {exp[k]!r}" for k in EXPECT_KEYS if not same(resolved[k], exp[k])]
        for key in ("w", "tau", "label_text_mode"):
            if key in cfg and not same(resolved[key], cfg[key]):
                cfg_bad.append(f"{key}: library {resolved[key]!r} vs decision {cfg[key]!r}")
        variant = cfg.get("v3s_variant") or mcfg.v3s_variant or (dec.get("r8") or {}).get("chosen_variant")
        ks, ks_src = per_k_keys(dec, cname), "the decision"
        if ks is None and d12 is not None:
            reg = mode_registered(d12, adopted12, resolved["label_text_mode"], variant)
            ks, ks_src = (reg or {}).get("per_k_keys"), f"the stage-1.2 decision ({adopted12})"
        if ks is None:
            ks, ks_src = list(K_ALLOWED), "not in the FINAL decision: the registered grid (3, 5, 8)"
        if int(resolved["k"]) not in ks:
            cfg_bad.append(f"k {resolved['k']} is not one of the per_k keys {ks} ({ks_src})")
        # B-2: the shipped model carries only its frozen (k, tau); a tau_by_k threshold at another k is not checked here
        _tbk_other = sorted(kk for kk, v in (getattr(mcfg, "tau_by_k", None) or {}).items()
                            if v is not None and int(kk) != int(mcfg.k))
        if _tbk_other:
            cfg_bad.append(f"tau_by_k holds thresholds at k {_tbk_other} other than the frozen k {mcfg.k}; this release "
                           "check compares the cascade only at the frozen k (remove them or check them first)")
        # per-mode settings (review item 5)
        by_mode = mcfg.fusion_by_mode or {}
        exp_bm = exp.get("fusion_by_mode") or {}
        bm_check = {}
        if by_mode or exp_bm:
            if sorted(by_mode) != sorted(exp_bm):
                cfg_bad.append(f"fusion_by_mode modes: library {sorted(by_mode)} vs PROTOCOL {sorted(exp_bm)}")
            if d12 is None:
                raise Refused(f"the release carries fusion_by_mode: set STAGE12_STAGED ({staged_name('stage12', pre)}) so "
                              "that its blocks are checked against the stage-1.2 decision")
            for m, b in sorted(by_mode.items()):
                e = exp_bm.get(m) or {}
                for key in ("w", "k", "tau"):
                    if not same(b[key], e.get(key)):
                        cfg_bad.append(f"fusion_by_mode.{m}.{key}: library {b[key]!r} vs PROTOCOL {e.get(key)!r}")
                reg = mode_registered(d12, adopted12, m, variant)
                bm_check[m] = {"library": b, "stage12": reg}
                if reg is None:
                    cfg_bad.append(f"fusion_by_mode.{m}: the stage-1.2 decision registers no settings for this mode")
                    continue
                for key in ("w", "tau"):
                    if not same(b[key], reg[key]):
                        cfg_bad.append(f"fusion_by_mode.{m}.{key}: library {b[key]!r} vs stage-1.2 decision {reg[key]!r}")
                if int(b["k"]) not in (reg["per_k_keys"] or list(K_ALLOWED)):
                    cfg_bad.append(f"fusion_by_mode.{m}.k {b['k']} not in {reg['per_k_keys'] or list(K_ALLOWED)}")
        # few-shot block (review item 1)
        fs_block = None
        if mcfg.fewshot or exp.get("fewshot"):
            fs_block = {k: v for k, v in (mcfg.fewshot or {}).items() if k not in ("source", "n_set")}
            e = exp.get("fewshot") or {}
            for key in FEWSHOT_KEYS:
                if not same(fs_block.get(key), e.get(key)):
                    cfg_bad.append(f"fewshot.{key}: library {fs_block.get(key)!r} vs PROTOCOL {e.get(key)!r}")
            if not mcfg.fewshot:
                cfg_bad.append("EXPECT_CONFIG has a fewshot block but the released model carries none")
                fs_block = None
            else:
                fname = S.get("fewshot_staged")
                allowed = staged_names(pre, FEWSHOT_STAGED_KINDS)
                if fname not in allowed:
                    raise Refused(f"the release carries a few-shot block: FEWSHOT_STAGED must be one of "
                                  f"{allowed} (got {fname!r})")
                fp = find_glob(staged_dir(inp, fname), "fewshot_decision_*.json")
                R["fewshot_decision"] = {"file": os.path.basename(fp), "sha256": check_sha(
                    fp, S.get("expect_fewshot_decision_sha256"), "few-shot decision (EXPECT_FEWSHOT_DECISION_SHA256)")}
                fdec = json.load(open(fp, encoding="utf-8"))
                if str(fdec.get("status", "")).split(" ")[0] != "FINAL" and not S.get("smoke"):
                    raise Refused(f"few-shot decision status {fdec.get('status')!r}: FINAL needed")
                ch = fdec.get("chosen") or {}
                for key in FEWSHOT_KEYS:
                    if key != "k" and not same(fs_block.get(key), ch.get(key)):
                        cfg_bad.append(f"fewshot.{key}: library {fs_block.get(key)!r} vs few-shot decision {ch.get(key)!r}")
                fks = sorted(int(x) for x in (fdec.get("per_k") or {}) if str(x).isdigit()) or list(K_ALLOWED)
                if int(fs_block["k"]) not in fks:
                    cfg_bad.append(f"fewshot.k {fs_block['k']} not in the few-shot decision's per_k keys {fks}")
                R["fewshot_decision"].update(chosen=ch, per_k_keys=fks)
        R["config"] = {"resolved": resolved, "decision": {k: cfg.get(k) for k in ("w", "k", "tau", "label_text_mode",
                                                                                    "v3s_variant")},
                       "per_k_keys": ks, "per_k_source": ks_src, "fusion_by_mode": bm_check, "fewshot": fs_block,
                       "problems": cfg_bad, "ok": not cfg_bad,
                       "anyintent_json": os.path.exists(os.path.join(rel, "anyintent.json"))}
        print(f"config: {json.dumps(R['config'])}")
        n_rows = int(S.get("n_rows") or N_ROWS)
        pools, comp, data_shas, col = load_pools(data_dir, ml_dir, SC, sc_variant, resolved["label_text_mode"], variant,
                                                 n_rows, man, man_ml)
        R["data_files_sha256"], R["label_column"], R["composition"] = data_shas, col, comp
        # integration fix I4 (AUDIT_FIXES L01): every row of clinc_dev the stored frame holds
        full = load_pools(data_dir, ml_dir, SC, sc_variant, resolved["label_text_mode"], variant, None, man, man_ml,
                          pools_wanted=[FULL_POOL], all_rows=True)[0][FULL_POOL]
        comp_bad = [f"{p}: {m} {v} label texts differ" for p, c in comp.items() for m, v in c.items()
                    if m in ("v2", "noex", "v3s") and v]
        # stored scores of the other registered modes (the same frame, variant <retriever>|<mode>)
        extra_pools, R["mode_rows"] = {}, {}
        for m, b in sorted(by_mode.items()):
            sv = f"{str(sc_variant).split('|')[0]}|{m}" if "|" in str(sc_variant) else None
            if sv is None or sv not in set(SC.variant):
                R["mode_rows"][m] = f"not compared: the stored DEV scores hold no variant {sv!r}"
                continue
            extra_pools[m] = load_pools(data_dir, ml_dir, SC, sv, m, variant, n_rows, man, man_ml)[0]
            R["mode_rows"][m] = {"variant": sv, "pools": sorted(extra_pools[m])}
        # few-shot row inputs and the reference (computed once on this device)
        FP = fs_ref = None
        if fs_block is not None:
            FP = fewshot_inputs(pools[POOLS[0]], int(S.get("fewshot_n") or FEWSHOT_N))
            if FP["n_examples"] == 0:
                raise Refused(f"the few-shot row has no examples: {POOLS[0]} has no decision rows beyond the {n_rows} "
                              "compared ones (lower N_ROWS)")
            R["fewshot_row"] = {"pool": POOLS[0], "rows": len(FP["msgs"]), "n_per_label": int(S.get("fewshot_n")
                                                                                                or FEWSHOT_N),
                                "examples": FP["n_examples"], "labels_without_examples": FP["labels_without_examples"]}
        R["rows"] = {p: {"n": P["n_stored"], "decision_rows_total": P["n_decision_total"]} for p, P in pools.items()}
        R["rows"][f"{FULL_POOL} full"] = {"n": full["n_stored"], "decision_rows_total": full["n_decision_total"],
                                          "near_test_rows": full["n_near_test"]}
        # engines and precisions
        cuda = torch.cuda.is_available() and not S.get("force_cpu")
        have_ort = R["env"]["onnxruntime"] is not None
        runs = S.get("runs") or default_runs(cuda, have_ort)
        device = "cuda:0" if cuda else "cpu"
        if FP is not None:
            t_ref = time.perf_counter()
            fs_ref = fewshot_reference(rel, device, FP, fs_block, mcfg.msg_max_len)
            R["fewshot_row"]["reference_s"] = round(time.perf_counter() - t_ref, 1)
            if cuda:
                torch.cuda.empty_cache()
        R["runs"], onnx_dir = [], None
        R["skipped_runs"], R["timing_shed"] = [], False
        n_t = int(S.get("n_timing") or 0)
        for ri, (engine, prec, gating) in enumerate(runs):
            # the first engine always runs; a later one is skipped when the session deadline has passed (INCOMPLETE)
            if ri > 0 and time.time() - t_start > deadline_s:
                R["skipped_runs"] = [f"{e}/{p_}" for e, p_, _ in runs[ri:]]
                R["problems"].append(f"session deadline reached before {engine}/{prec}: skipped {R['skipped_runs']}")
                break
            rr = {"engine": engine, "precision": prec, "device": device, "gating": gating}
            t0 = time.perf_counter()
            try:
                if engine == "onnx":
                    if onnx_dir is None:
                        from anyintent import onnx as aonnx
                        onnx_dir = tempfile.mkdtemp(prefix="anyintent_onnx_")
                        try:
                            rr["export"] = aonnx.export(rel, onnx_dir, quantize_int8=True, local_files_only=True)["files"]
                        except Exception as e:      # int8 quantisation is information: retry without it
                            rr["export_int8_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                            rr["export"] = aonnx.export(rel, onnx_dir, quantize_int8=False, local_files_only=True)["files"]
                    clf = AnyIntent.from_pretrained(rel, engine="onnx", onnx_dir=onnx_dir, onnx_variant=prec,
                                                    local_files_only=True)
                else:
                    clf = AnyIntent.from_pretrained(rel, device=device, dtype=prec, engine=engine,
                                                    local_files_only=True)
                rr["engine_used"] = clf.engine
                rr["load_s"] = round(time.perf_counter() - t0, 2)
                mode0 = resolved["label_text_mode"]
                rr["pools"] = {p: compare_pool(clf, P, mcfg.w, mcfg.k, mcfg.tau, mcfg.cascade_first, mode0)
                               for p, P in pools.items()}
                if gating and ((cuda or S.get("smoke")) or (ri == 0 and S.get("full_row_on_cpu"))):
                    # I4 (L01): every gating engine of a GPU (eager, fast and auto in fp16) and of the smoke on every
                    # clinc_dev row of the stored frame; on a CPU only the first engine and only on request (about 9,000
                    # cross pairs per engine would not fit the CPU session deadline next to the other engines)
                    rr["pools"][f"{FULL_POOL} full"] = compare_pool(clf, full, mcfg.w, mcfg.k, mcfg.tau,
                                                                    mcfg.cascade_first, mode0, agree_min=AGREE_MIN_FULL)
                for m, mp in extra_pools.items():          # the other registered modes at their own (w, k, tau)
                    b = by_mode[m]
                    rr["pools"].update({f"{p} [{m}]": compare_pool(clf, P, b["w"], b["k"], b["tau"], mcfg.cascade_first, m)
                                        for p, P in mp.items()})
                if fs_ref is not None:
                    rr["pools"][f"{POOLS[0]} [fewshot {fs_block['variant']}]"] = compare_fewshot(clf, FP, fs_ref, fs_block)
                rr["ok"] = all(x["ok"] for x in rr["pools"].values())
                if not R["runs"]:     # preflight: projection from the first engine on this device
                    per_run = time.perf_counter() - t0
                    proj = time.time() - t_start + per_run * (len(runs) - 1)
                    R["projection"] = {"first_run_s": round(per_run, 1), "runs": len(runs),
                                       "projected_s": round(proj, 1), "deadline_s": deadline_s}
                    print(f"preflight: first run {per_run:.1f} s, projected {proj:.0f} s for {len(runs)} runs "
                          f"(deadline {deadline_s:.0f} s)")
                    if n_t and proj > SHED_AT * deadline_s:
                        R["timing_shed"] = True
                        print(f"projection above {SHED_AT:.0%} of the deadline: latency rows (information) are dropped "
                              "for the remaining engines")
                if n_t and not (R["timing_shed"] and R["runs"]):
                    P0 = pools[POOLS[0]]
                    rr["timing"] = timing(clf, P0["msgs"][:n_t], P0["texts"], cascade=mcfg.tau is not None,
                                          mode=resolved["label_text_mode"])
                del clf
            except Exception as e:
                traceback.print_exc()
                rr["ok"], rr["error"] = False, f"{type(e).__name__}: {str(e)[:400]}"
            if cuda:
                torch.cuda.empty_cache()
            rr["seconds"] = round(time.perf_counter() - t0, 1)
            print(f"run {engine}/{prec}: {json.dumps(rr)}")
            R["runs"].append(rr)
        gating_runs = [r for r in R["runs"] if r["gating"]]
        failed = [r for r in gating_runs if not r.get("ok")]
        R["problems"] += cfg_bad + comp_bad + [f"run {r['engine']}/{r['precision']} failed" for r in failed]
        R["full_row_runs"] = [f"{r['engine']}/{r['precision']}" for r in gating_runs
                              if f"{FULL_POOL} full" in (r.get("pools") or {})]
        if not R["full_row_runs"]:
            R["problems"].append(f"the full {FULL_POOL} row (every row at {AGREE_MIN_FULL}, AUDIT_FIXES L01) was scored "
                                 "by no gating engine: a GPU run, or full_row_on_cpu on a CPU, is needed for PASS")
        # FAIL: a numeric or configuration problem; INCOMPLETE: no problem found but the deadline cut engines (re-run,
        # never PASS); PASS: every engine ran and passed
        if failed or cfg_bad or comp_bad or not gating_runs and not R["skipped_runs"]:
            R["status"] = "FAIL"
        elif R["skipped_runs"] or not R["full_row_runs"]:
            R["status"] = "INCOMPLETE"
        else:
            R["status"] = "PASS"
    except Refused as e:
        R["status"], R["refused"] = "REFUSED", str(e)
        print(f"REFUSED: {e}")
    except Exception as e:
        traceback.print_exc()
        R["status"], R["error"] = "ERROR", f"{type(e).__name__}: {str(e)[:600]}"
    R["seconds"] = round(time.time() - t_start, 1)
    out = S["out_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"library_release_{run}.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=str)
    lines = [f"library release check {run}: {R['status']}", f"model sha256 {R.get('model_sha256')} pin {R.get('pin')}",
             f"anyintent.json sha256 {R.get('anyintent_json_sha256')}; released files {len(R.get('released_files') or {})}",
             f"skipped runs {R.get('skipped_runs')}; timing shed {R.get('timing_shed')}",
             f"decision sha256 {R.get('decision_sha256')}", f"config {json.dumps(R.get('config', {}).get('resolved'))}",
             f"gpus {R['env']['gpus']}", f"problems {R['problems']}", f"refused {R.get('refused')}"]
    for r in R.get("runs", []):
        lines.append(f"  {r['engine']}/{r['precision']} ok {r.get('ok')} " + json.dumps(
            {p: {k: x.get(k) for k in ("candidate_mismatch_rows", "max_abs_cross_diff", "fused_top1_agreement",
                                       "cascade_top1_agreement")} for p, x in (r.get("pools") or {}).items()}))
    with open(os.path.join(out, f"library_release_{run}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return R


# ---- L1 step: pin the PROTOCOL values in anyintent.json ------------------------------------------------------------
def write_pins(release_dir, w=None, k=None, tau="unset", cascade_first=None, label_text_mode=None, v3s_variant=None,
               fusion_by_mode=None, fewshot=None):
    """Writes (or updates) anyintent.json in a one-file release folder so that the library resolves the values of the
    pre-release PROTOCOL entry when they differ from the released meta file (for example the k of the
    T4 k-feasibility rule), the registered settings of another label text mode (fusion_by_mode, for example the v2
    settings of a v3s release) and the few-shot block (the chosen keys of the few-shot DEV decision, with the k of the
    pre-look PROTOCOL entry). Returns (path, sha256). The weights are not touched."""
    p = os.path.join(release_dir, "anyintent.json")
    ai = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"format": "anyintent-one-file", "version": 1}
    fu = dict(ai.get("fusion") or {})
    for key, val in (("w", w), ("k", k), ("cascade_first", cascade_first)):
        if val is not None:
            fu[key] = val
    if tau != "unset":
        fu["tau"] = tau
    ai["fusion"] = fu
    if label_text_mode is not None:
        ai["label_text_mode"] = label_text_mode
    if v3s_variant is not None:
        ai["v3s_variant"] = v3s_variant
    if fusion_by_mode:
        bm = dict(ai.get("fusion_by_mode") or {})
        for m, b in fusion_by_mode.items():
            bm[m] = {"w": float(b["w"]), "k": int(b["k"]), "tau": None if b.get("tau") is None else float(b["tau"])}
        ai["fusion_by_mode"] = bm
    if fewshot is not None:
        ai["fewshot"] = {kk: fewshot.get(kk) for kk in FEWSHOT_KEYS + ("msg_max_len", "label_max_len", "pair_max_len",
                                                                       "n_set") if fewshot.get(kk) is not None
                         or kk in ("agg", "a")}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ai, f, indent=1, sort_keys=True)
        f.write("\n")
    return p, sha256_file(p)


# ---- smoke stand-ins (tiny random models; needs the package's tests folder on sys.path) ------------------------------
def split_one_file(packaged_dir, cross_dir, adapter_file, prefix=R2_PREFIX, out_prefix=SMOKE_TRAINING_PREFIX):
    """Smoke stand-in of the stage-1.2 training outputs a packaged one-file model was made from: the cross-encoder
    folder (backbone and head in fp32, config.json) and the adapter file (the adapter tensors under the training prefix
    out_prefix, SMOKE_TRAINING_PREFIX*, whatever the packaged prefix is). The packaged file casts the backbone to
    fp16, so the three files have three different sha256."""
    import shutil
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(cross_dir, exist_ok=True)
    with safe_open(os.path.join(packaged_dir, "model.safetensors"), framework="pt") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    save_file({k: v.to(torch.float32).contiguous() for k, v in sd.items() if not k.startswith(prefix)},
              os.path.join(cross_dir, "model.safetensors"), metadata={"format": "pt"})
    shutil.copy(os.path.join(packaged_dir, "config.json"), os.path.join(cross_dir, "config.json"))
    save_file({out_prefix + k[len(prefix):]: v.contiguous() for k, v in sd.items() if k.startswith(prefix)}, adapter_file,
              metadata={"format": "pt"})


def _perturb(path, adapter=False, prefix=SMOKE_TRAINING_PREFIX):
    """Adds 1 to the first element of one tensor of a safetensors file (smoke faults)."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    with safe_open(path, framework="pt") as f:
        sd = {k: f.get_tensor(k).clone() for k in f.keys()}
    key = sorted(k for k in sd if k.startswith(prefix) == adapter and sd[k].is_floating_point())[0]
    sd[key].view(-1)[0] += 1.0
    save_file(sd, path, metadata={"format": "pt"})


def smoke_inputs(root, layout="one_file", fault=None, style="configs"):
    """Writes a staged input tree under root with a tiny random released model, the training outputs it was packaged
    from (cross-encoder folder and adapter file, as stage-1.2 FINAL writes them), generated DEV data and stored DEV
    scores computed by the reference DEV scorer (verbatim copy of the registered code, with the R2Intent names), and
    returns the settings for run_check. The released one-file folder carries the R2Intent names (r2intent_meta.json,
    r2intent_embed_lora.*, safetensors metadata key r2intent; PROTOCOL E5 (10f), D6) while the decision-pinned adapter
    file holds its tensors under a training prefix (SMOKE_TRAINING_PREFIX, in the setting training_prefixes), as the
    project's stage-1.2 adapter files do; the staged folders have neutral names (staged_name with SMOKE_STAGED_PREFIX,
    the setting staged_prefix). style: "configs" (final_decision with configs), "top" (config keys
    at the top level), "honest" (plan section 6, the honest model ships: stage12_decision_*.json in the stage12 folder
    with configs B, C12, HN12, the packaged model in the release-honest folder, data in the data-v3 and ml-v3 folders),
    "s12_final" (the stage-1.2 FINAL pub writer's shapes, cells_s12_part5 and part6, under the R2Intent names (the
    writer itself wrote the names before the rename, which the library no longer reads): configs P and Z with
    adopted = the stage-1.2 name, cross_sha256 as a dict, no model_sha256 or package_dir in the decision,
    final_package_<run>.json with final_record_sha256 and files, the packaged r2intent_meta.json with
    final_record_sha256, score variant "adapter|<mode>", no per_k). fault: None, "tamper",
    "test_path", "sha", "config", "pins", "alt_keys", "no_tau", "unpinned_package" (the decision does not pin the
    packaged file), "cross_mismatch", "adapter_mismatch" (a staged source file differs from the packaged tensors);
    with style "s12_final" also "s12_bad_record" (the package record names another decision), "s12_meta_record" (the
    packaged meta names another decision), "s12_no_sources" (the source files are not staged). Round 1 of the second
    cycle: style "v3s" (a v3s-A release with the registered v2 settings in anyintent.json fusion_by_mode, the stage-1.2
    decision staged in the stage12 folder and bound by stage12_decision_sha256, stored scores of both modes, no per_k in
    the FINAL decision; fault "v3s_mode_mismatch": the v2 block differs from the stage-1.2 decision); faults "fewshot"
    (a KX few-shot block pinned in anyintent.json, the few-shot DEV decision staged in the fewshot-dev-s12 folder) and
    "fewshot_mismatch" (the block's w differs from that decision), "anyjson_sha" (a pasted anyintent.json sha256
    without the file). style "pub_record": the record of the P job (FINAL pub with patch P: configs p_full, p0 and Z,
    the package files pinned in configs.p_full.package.files_sha256, no dev_scores_sha256 in the config, CONFIG_NAME
    p_full); with fault "pub_package_mismatch" the record pins another model file."""
    import pandas as pd
    import torch
    import conftest as C
    from reference import v11_lib_ref as R
    from anyintent.labels import label_texts, strip_examples
    rng = np.random.default_rng(0)
    os.makedirs(root, exist_ok=True)
    tok = C.build_tokenizer(tempfile.mkdtemp(prefix="anyintent_tok_"))
    honest, s12, v3s, pubrec = style == "honest", style == "s12_final", style == "v3s", style == "pub_record"
    fin = os.path.join(root, staged_name("stage12") if honest else staged_name("final-pub"))
    rel = os.path.join(root, staged_name("release-honest"), "r2intent_honest_smoke") if honest else \
        os.path.join(fin, "r2intent_final_smoke")
    os.makedirs(fin, exist_ok=True)
    if layout == "one_file":
        mode = "v3s" if v3s else "noex"
        C.make_one_file(rel, tok, seed=5, w=15.0, final=True, tau=0.5, mode=mode)
        w, k, tau = 15.0, 8, 0.5
        meta_p = os.path.join(rel, release_meta(rel)[1])
        meta = json.load(open(meta_p))
        if v3s:
            meta["v3s_variant"] = "A"
        if fault == "alt_keys":
            meta["fusion"].pop("tau")
            meta["cascade"] = {"tau": tau, "first": 3}
            meta["fusion"]["label_text_mode"] = meta.pop("label_text_mode")
        if fault == "no_tau":
            meta["fusion"].pop("tau")
        json.dump(meta, open(meta_p, "w"))
    else:
        C.make_two_file(rel, tok, seed=3, w=5.0, k=8)
        w, k, tau, mode = 5.0, 8, None, "v2"
    # labels: the conftest label set as clinc_dev, plus another pool
    names = [l if isinstance(l, str) else l["name"] for l in C.LABELS] + ["other a", "other b"]
    descs = ["" if isinstance(l, str) else (l.get("description") or "") for l in C.LABELS] + ["first other.", ""]
    ids = [f"L{i:02d}" for i in range(len(names))]
    pool_of = ["clinc_dev"] * len(C.LABELS) + ["other_dev"] * 2
    LAB = pd.DataFrame({"label_id": ids, "label_name": names, "name": names, "description": descs,
                        "description_noex": [strip_examples(d, n) if d else "" for n, d in zip(names, descs)],
                        "label_text": label_texts(names, descs, "v2"), "label_text_noex": label_texts(names, descs, "noex"),
                        "split": "dev", "pool": pool_of})
    LAB["label_text_v3s_A"] = LAB.label_name + ": a short gloss."
    LAB["label_text_v3s_B"] = LAB.label_text
    msgs = list(C.MESSAGES) + [m + " please" for m in C.MESSAGES]
    nC = len(C.LABELS)
    DV = pd.DataFrame({"row_id": np.arange(len(msgs)), "text": msgs,
                       "label_id": [ids[int(x)] for x in rng.integers(0, nC, len(msgs))], "pool": "clinc_dev",
                       "near_test": [i in (1, 5) for i in range(len(msgs))]})
    DV = pd.concat([DV, pd.DataFrame({"row_id": [900, 901], "text": ["other one", "other two"],
                                      "label_id": ["L20", "L21"], "pool": "other_dev", "near_test": False})],
                   ignore_index=True)
    ml = []
    for li, lang in enumerate(("lt", "pl")):
        for j in range(len(C.MESSAGES)):
            ml.append({"row_id": 1000 + 100 * li + j, "text": C.MESSAGES[(j + 3 * li) % len(C.MESSAGES)],
                       "label_id": DV.label_id[j], "pool": "clinc_dev", "lang": lang, "src_row_id": j})
    DVML = pd.DataFrame(ml)
    dname, mname_ = (staged_name("data-v3"), staged_name("ml-v3")) if honest else (staged_name("data-pub"),
                                                                                  staged_name("ml-pub"))
    data_dir, ml_dir = os.path.join(root, dname), os.path.join(root, mname_)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ml_dir, exist_ok=True)
    DV.to_parquet(os.path.join(data_dir, "step1_dev.parquet"), index=False)
    LAB.to_parquet(os.path.join(data_dir, "step1_labels.parquet"), index=False)
    DVML.to_parquet(os.path.join(ml_dir, "step1_dev_ml.parquet"), index=False)
    for d, mname, files in ((data_dir, "step1_manifest.json", ("step1_dev.parquet", "step1_labels.parquet")),
                            (ml_dir, "step1_ml_manifest.json", ("step1_dev_ml.parquet",))):
        json.dump({"recipe": "v3" if honest else "pub", "smoke": True,
                   "files": {f: {"sha256": sha256_file(os.path.join(d, f))} for f in files}},
                  open(os.path.join(d, mname), "w"), indent=1)
    # stored DEV scores: the reference DEV scorer, CPU fp32
    pool_ids = sorted(DV[DV.pool == "clinc_dev"].label_id.unique())

    def tokr(p):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(p)

    frames = []
    variants = [(f"adapter|{mode}" if (s12 or pubrec) else "desc",
                 {"v2": "label_text", "noex": "label_text_noex"}.get(mode))]
    if v3s:
        variants = [("adapter|A", "label_text_v3s_A"), ("adapter|v2", "label_text")]
    for (sc_var, col), (pool, rows) in [(vc, pr) for vc in variants for pr in (
            ("clinc_dev", DV[DV.pool == "clinc_dev"]), ("clinc_dev@lt", DVML[DVML.lang == "lt"]),
            ("clinc_dev@pl", DVML[DVML.lang == "pl"]))]:
        texts = LAB.set_index("label_id").loc[pool_ids, col].tolist()
        rows = rows.sort_values("row_id")
        q = rows.text.tolist()
        if layout == "one_file":
            m, lora, mt = R.load_one_file(rel, torch.float32)
            m.eval()
            with R.adapter_on(lora, True):
                Q = R.embed_texts(m, tokr(rel), q, mt["msg_max_len"], "cpu", amp=False)
                L = R.embed_texts(m, tokr(rel), texts, mt["label_max_len"], "cpu", amp=False)
            cand, vals, _ = R.retrieve(Q, L, 8)
            with R.adapter_on(lora, False):
                X = R.rerank(m, tokr(rel), q, texts, cand, 8, mt["pair_max_len"], "cpu", amp=False)
        else:
            emb = R.load_m3_encoder(os.path.join(rel, "embedder"), torch.float32).eval()
            cr = R.load_seqcls_exact(os.path.join(rel, "cross"), torch.float32).eval()
            Q = R.embed_texts(emb, tokr(os.path.join(rel, "embedder")), q, 512, "cpu", amp=False)
            L = R.embed_texts(emb, tokr(os.path.join(rel, "embedder")), texts, 512, "cpu", amp=False)
            cand, vals, _ = R.retrieve(Q, L, 8)
            X = R.rerank(cr, tokr(os.path.join(rel, "cross")), q, texts, cand, 8, 160, "cpu", amp=False)
        frames.append(pd.DataFrame({"row_id": rows.row_id.to_numpy(), "pool": pool, "variant": sc_var,
                                    "gold_label_id": rows.label_id.to_numpy(),
                                    "cand_label_ids": [[pool_ids[j] for j in r] for r in cand],
                                    "retr_scores": [list(map(float, r)) for r in vals],
                                    "cross_scores": [list(map(float, r)) for r in X], "gold_retr_rank": 0}))
    SCF = pd.concat(frames, ignore_index=True)
    if fault == "tamper":
        SCF.at[0, "cross_scores"] = [x + 1.0 for x in SCF.at[0, "cross_scores"]]
    sc_name = "stage12_dev_scores_final_smoke.parquet" if fault != "test_path" else "stage12_dev_scores_test_smoke.parquet"
    sc_path = os.path.join(fin, sc_name)
    SCF.to_parquet(sc_path, index=False)
    wfile = os.path.join(rel if layout == "one_file" else os.path.join(rel, "cross"), "model.safetensors")
    msha = sha256_file(wfile)
    if layout == "one_file":
        # the training outputs the packaged file was made from (stage-1.2 FINAL writes them next to its decision)
        cdir, afile = os.path.join(fin, "cross_final_smoke"), os.path.join(fin, "adapter_final_smoke.safetensors")
        split_one_file(rel, cdir, afile, prefix=(release_meta(rel)[0].get("adapter") or {}).get("prefix") or R2_PREFIX)
        if fault == "cross_mismatch":
            _perturb(os.path.join(cdir, "model.safetensors"), adapter=False)
        if fault == "adapter_mismatch":
            _perturb(afile, adapter=True)
        csha, asha = sha256_file(os.path.join(cdir, "model.safetensors")), sha256_file(afile)
        assert len({msha, csha, asha}) == 3, "the stand-in files must have three different sha256"
        if fault == "s12_no_sources":
            import shutil
            shutil.rmtree(cdir)
            os.remove(afile)
    if layout == "one_file" and pubrec:
        # the P job's config of an arm (cells_s12_pub_b._cfg_of): base names, the cross folder pinned as {file: sha256},
        # the adapter file, the worker's package record with its files (the released file pinned through it)
        conf = {"cross_dir": os.path.basename(cdir), "cross_sha256": {"model.safetensors": csha},
                "adapter_path": os.path.basename(afile), "adapter_sha256": asha, "retriever": "adapter",
                "package": {"dir": "/run/machine/out/" + os.path.basename(rel),
                            "files_sha256": {"model.safetensors": msha if fault != "pub_package_mismatch" else "4" * 64},
                            "reload_ok": True}}
    elif layout == "one_file" and s12:
        # cfg_record of cells_s12_part5: base names, the cross folder pinned as {file: sha256}, nothing about the package
        conf = {"cross_dir": os.path.basename(cdir), "cross_sha256": {"model.safetensors": csha},
                "adapter_path": os.path.basename(afile), "adapter_sha256": asha, "retriever": "adapter",
                "dev_scores_variant": f"adapter|{mode}"}
    elif layout == "one_file":
        conf = {"cross_dir": "/run/machine/out/cross_final_smoke",
                "adapter_path": "/run/machine/out/adapter_final_smoke.safetensors",
                "sha256": {"cross": csha, "adapter": asha}}
        if fault != "unpinned_package":     # interface addition: FINAL with packaging pins the packaged file
            conf.update(model_sha256=msha, package_dir=os.path.basename(rel))
    else:
        conf = {"cross_dir": "/run/machine/out/cross", "adapter_path": None, "sha256": {"cross": msha, "adapter": None}}
    conf.update({"w": w, "k": k, "tau": tau, "label_text_mode": mode, "dev_scores": sc_name,
                 "dev_scores_sha256": sha256_file(sc_path)})
    pk = {"3": {"pooled": 0.5}, "5": {"pooled": 0.5}, "8": {"pooled": 0.5}}
    v2_block = {"w": 5.0, "k": 5, "tau": None}
    extra = {}
    if v3s:     # the stage-1.2 decision of the adopted config (R8 adopted v3s A), staged in the stage12 folder
        conf.update(v3s_variant="A", dev_scores_variant="adapter|A")
        d12_dir = os.path.join(root, staged_name("stage12"))
        os.makedirs(d12_dir, exist_ok=True)
        d12 = {"status": "FINAL", "adopted": "HN12", "per_k": {"HN12": pk},
               "configs": {"HN12": {"w": w, "k": k, "tau": tau, "label_text_mode": "v3s", "v3s_variant": "A",
                                    "v2": dict(v2_block, dev_scores_variant="adapter|v2"),
                                    "per_mode": {"v2": v2_block, "A": {"w": w, "k": k, "tau": tau}}}},
               "r8": {"chosen_variant": "A", "variants": {"v2": dict(v2_block, per_k=pk),
                                                           "A": {"w": w, "k": k, "tau": tau, "per_k": pk}}}}
        d12_path = os.path.join(d12_dir, "stage12_decision_smoke.json")
        json.dump(d12, open(d12_path, "w"), indent=1)
        extra["stage12_staged"] = staged_name("stage12")
    if pubrec:      # the P job's record (FINAL pub with patch P): configs p_full, p0 and Z; the CONFIG_NAME is p_full
        dec = {"kind": "final_decision", "stage": "1.2", "status": "FINAL", "final_data": "pub", "pub_run": "pjob",
               "pub_kind": "train", "adopted": "B", "smoke": True,
               "configs": {"p_full": conf, "p0": {"w": w, "k": k, "tau": tau},
                           "Z": {"source": "stage-1.2 adopted B seed 0 (configs.B, the v1.1 ONE0)", "w": w, "k": k}},
               "w": w, "k": k, "tau": tau, "label_text_mode": mode, "nwp_pass": {"p_full": True, "p0": True}}
        dec_path = os.path.join(fin, "final_decision_smoke.json")
        extra.update(config_name="p_full", expect_dev_scores_sha256=sha256_file(sc_path))
        conf.pop("dev_scores_sha256", None)
    elif s12:       # FINAL pub record of cells_s12_part5 (two configs, adopted = the stage-1.2 name, no per_k)
        dec = {"kind": "final_decision", "stage": "1.2", "status": "FINAL", "final_data": "pub", "adopted": "HN12",
               "configs": {"P": conf, "Z": {"source": "stage-1.2 adopted HN12 seed 0 (honest model)",
                                            "dev_scores": "stage12_dev_scores_z_smoke.parquet",
                                            "dev_scores_sha256": "1" * 64}},
               "w": w, "k": k, "tau": tau, "label_text_mode": mode, "v3s_variant": None, "package": True,
               "not_worse": {"pass": True}, "decision": "ship the public model"}
        dec_path = os.path.join(fin, "final_decision_smoke.json")
    elif honest:      # stage-1.2 decision of the honest model (plan section 6: the honest model ships)
        other = {kk: v for kk, v in conf.items() if kk not in ("model_sha256", "package_dir")}
        dec = {"status": "FINAL", "branch": "A1", "adopted": "HN12", "shipped": "honest",
               "configs": {"B": dict(other, w=5.0), "C12": dict(other, w=10.0), "HN12": conf},
               "per_k": {c: pk for c in ("B", "C12", "HN12")}}
        dec_path = os.path.join(fin, "stage12_decision_smoke.json")
    elif v3s:       # FINAL record of cells_s12_part5: no per_k, bound to the stage-1.2 decision by its sha256
        dec = {"status": "FINAL", "final_data": "pub", "package": True, "adopted": "HN12", "configs": {"P": conf},
               "stage12_decision_sha256": sha256_file(d12_path), "w": w, "k": k, "tau": tau,
               "label_text_mode": "v3s", "v3s_variant": "A"}
        dec_path = os.path.join(fin, "final_decision_smoke.json")
    else:
        dec = {"status": "FINAL", "final_data": "pub", "package": True, "not_worse": {"ok": True}, "adopted": "HN12",
               "per_k": {"HN12": pk}}
        if style == "configs":
            dec["configs"] = {"HN12": conf}
        else:
            dec.update(conf)
        dec_path = os.path.join(fin, "final_decision_smoke.json")
    json.dump(dec, open(dec_path, "w"), indent=1)
    if s12:     # cells_s12_part6: the packaged meta and the package record carry the decision's sha256
        dsha = sha256_file(dec_path)
        meta_p = os.path.join(rel, release_meta(rel)[1])
        meta = json.load(open(meta_p))
        meta["final_record_sha256"] = dsha if fault != "s12_meta_record" else "2" * 64
        meta["r2intent"] = "stage 1.2 public model (R2Intent)"
        json.dump(meta, open(meta_p, "w"), indent=1)
        json.dump({"dir": "/run/machine/out/" + os.path.basename(rel), "checks": {"reload_ok": True},
                   "files": {"model.safetensors": msha}, "ok": True,
                   "final_record_sha256": dsha if fault != "s12_bad_record" else "3" * 64},
                  open(os.path.join(fin, "final_package_smoke.json"), "w"), indent=1)
    expect = {"w": w, "k": k, "tau": tau, "cascade_first": 3, "label_text_mode": mode}
    if fault == "config":
        expect["k"] = 5
    if fault == "pins":
        expect["k"] = 5
        write_pins(rel, k=5)
    if v3s:
        blk = dict(v2_block, w=6.0) if fault == "v3s_mode_mismatch" else dict(v2_block)
        write_pins(rel, fusion_by_mode={"v2": blk})
        expect["fusion_by_mode"] = {"v2": blk}
    if fault in ("fewshot", "fewshot_mismatch"):
        block = {"variant": "KX", "agg": "lse", "a": 0.5, "w": 10.0, "k": 5, "lse_tau": 0.05, "m_text": 3,
                 "ex_word_cap": 25}
        write_pins(rel, fewshot=dict(block, msg_max_len=128, label_max_len=192, pair_max_len=160, n_set=[2, 5, 10]))
        expect["fewshot"] = dict(block)
        fdir = os.path.join(root, staged_name("fewshot-dev-s12"))
        os.makedirs(fdir, exist_ok=True)
        chosen = dict(block, w=12.0) if fault == "fewshot_mismatch" else dict(block)
        fpath = os.path.join(fdir, "fewshot_decision_smoke.json")
        json.dump({"status": "FINAL", "chosen": dict(chosen, msg_max_len=128, label_max_len=192, pair_max_len=160),
                   "per_k": {"3": {"Q": 0.5}, "5": {"Q": 0.5}, "8": {"Q": 0.5}}}, open(fpath, "w"), indent=1)
        extra.update(fewshot_staged=staged_name("fewshot-dev-s12"), expect_fewshot_decision_sha256=sha256_file(fpath),
                     fewshot_n=2, n_rows=24)
    if honest:
        extra.update({"release_staged": staged_name("stage12"), "decision_glob": "stage12_decision_*.json",
                      "package_staged": staged_name("release-honest"), "data_staged": staged_name("data-v3"),
                      "ml_staged": staged_name("ml-v3")})
    aj = os.path.join(rel, "anyintent.json")
    aj_sha = sha256_file(aj) if os.path.exists(aj) else None
    if fault == "anyjson_sha":
        aj_sha = "0" * 64
    return {"input_root": root, "release_staged": staged_name("final-pub"), "expect_anyintent_json_sha256": aj_sha,
            "expect_decision_sha256": sha256_file(dec_path),
            "expect_model_sha256": msha if fault != "sha" else "0" * 64,
            "expect_data_manifest_sha256": sha256_file(os.path.join(data_dir, "step1_manifest.json")),
            "expect_ml_manifest_sha256": sha256_file(os.path.join(ml_dir, "step1_ml_manifest.json")),
            "expect_dev_scores_sha256": None, "expect_config": expect, "smoke": True, "n_rows": N_ROWS,
            "staged_prefix": SMOKE_STAGED_PREFIX, "training_prefixes": [SMOKE_TRAINING_PREFIX], **extra}


def main():
    ap = argparse.ArgumentParser(description="anyintent release tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pins", help="write the PROTOCOL values into anyintent.json of a one-file release folder")
    p.add_argument("release_dir")
    p.add_argument("--w", type=float)
    p.add_argument("--k", type=int)
    p.add_argument("--tau", help="a number, or 'none' for no cascade")
    p.add_argument("--cascade-first", type=int)
    p.add_argument("--mode", choices=["v2", "v3s", "noex", "name"])
    p.add_argument("--v3s-variant", choices=["A", "B"])
    p.add_argument("--mode-settings", action="append", default=[],
                   help="MODE:W,K,TAU (repeatable), for example v2:20,5,none: the registered settings of another label "
                   "text mode (fusion_by_mode)")
    p.add_argument("--fewshot-json", help="the few-shot block as JSON (variant, agg, a, w, k, lse_tau, m_text, "
                   "ex_word_cap; the chosen keys of the few-shot DEV decision with the PROTOCOL k)")
    a = ap.parse_args()
    tau = "unset" if a.tau is None else (None if a.tau.lower() == "none" else float(a.tau))
    by_mode = {}
    for s in a.mode_settings:
        m, _, vals = s.partition(":")
        w_, k_, t_ = vals.split(",")
        by_mode[m] = {"w": float(w_), "k": int(k_), "tau": None if t_.lower() == "none" else float(t_)}
    path, sha = write_pins(a.release_dir, w=a.w, k=a.k, tau=tau, cascade_first=a.cascade_first, label_text_mode=a.mode,
                           v3s_variant=a.v3s_variant, fusion_by_mode=by_mode or None,
                           fewshot=json.loads(a.fewshot_json) if a.fewshot_json else None)
    print(f"wrote {path}\nsha256 {sha}\n{open(path).read()}")


if __name__ == "__main__":
    main()
