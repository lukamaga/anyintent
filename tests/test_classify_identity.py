"""classify is unchanged by the check and scores work (PROTOCOL entry of 2026-10-13 00:06, (2); design revision 4, 1.6
checks 1 to 3):

1. the vendored pre-change library (tests/reference/anyintent_pre/) is tied to the pre-change commit by its sha256 list:
   its files are the commit's bytes, except the files listed in "renamed", from which the support of the file names
   before the rename to R2Intent (the old meta file, name key and adapter prefix) and the old name in comments were
   removed on 2026-10-17 with the same edit as in the library (check 2 holds them to the library's files); where git
   holds the commit, the diff of each such file from the commit's file must have the sha256 of its list entry
   (rename_diff_sha256), which freezes that edit;
2. source: every definition of every module of the pre-change library is unchanged byte for byte in the new library
   (whole files where nothing was added; in model.py and hub.py every top-level statement and every class body
   statement, by text and by AST, in the same order, and the old lines are kept in order with only lines inserted);
   new names only where the entry allows them; __init__.py changes only its import line of .model and __all__; the
   new code (calibration.py, the new statements of hub.py and model.py) writes no state of the old code: no attribute
   or item of an imported module or of a module-level object, no attribute of the instance that is not its own, no
   global, no process-wide setting;
3. in process: the old and the new library loaded side by side on the same tiny models give equal Predictions (every
   field, floats with ==) for k 3, 5 and 8, the cascade on and off, few-shot on and off, return_scores true and false,
   single and batch, with and without a calibration file in the model folder, before and after calls of check and
   scores on the same new instance, with a repeated run of the old library as the control; every setting but the
   cascade on a model without a tau returns Predictions, so the comparison is never a comparison of refusals only;
4. state: check and scores leave the label state, the backend's prepared label texts and the shared label cache
   (keys, order, hit counters, the disk folder) untouched."""
import ast
import dataclasses
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import warnings

import numpy as np
import pytest

import anyintent as NEW
from anyintent import calibration as calib
from calib_support import (FEWSHOT_P, calibration_doc, make_p_like, model_copy, write_calibration)
from conftest import LABELS, MESSAGES
from reference import anyintent_pre as OLD

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_DIR = os.path.join(HERE, "reference", "anyintent_pre")
SHA_LIST = os.path.join(HERE, "reference", "anyintent_pre_sha256.json")
NEW_DIR = os.path.dirname(os.path.abspath(NEW.__file__))
PRE_COMMIT = "bc424ed11d093c7da808e5568f9962881f51fd43"

UNCHANGED_FILES = ["backends.py", "cache.py", "config.py", "fast.py", "fewshot.py", "fusion.py", "labels.py",
                   "onnx.py"]
# _version.py: the library release v0.1.0 changes only its version string, from the pre-release to the release (the
# pinned library content of PROTOCOL 2026-10-13 03:29 (a) leaves this file out); nothing else in it may change
VERSION_FILE = "_version.py"
PRE_CHANGE_VERSION, RELEASE_VERSION = "0.1.0.dev0", "0.1.0"
ADDITIVE_FILES = ["hub.py", "model.py"]
# new modules: calibration.py (check and scores) and the server and MCP modules of the extras, which only call the
# library (service.py, server.py, serve_cli.py, mcp_server.py); every pre-change module stays as checked below
NEW_FILES = ["calibration.py", "mcp_server.py", "serve_cli.py", "server.py", "service.py"]
# the new top-level statements the entry allows (definitions by name, imports by their text)
ALLOWED_NEW_TOP = {
    "hub.py": {"OPTIONAL_FILES", "fetch_optional"},
    "model.py": {"CheckResult", "ScoresResult"},
}
ALLOWED_NEW_METHODS = {("model.py", "AnyIntent"): {"calibration", "check", "scores", "_calibration_file",
                                                  "_calibration_for", "_check_vector", "_pair_raw", "_warn_once"}}
INIT_CHANGES = {
    "from .model import AnyIntent, Prediction": "from .model import AnyIntent, CheckResult, Prediction, ScoresResult",
    '__all__ = ["AnyIntent", "Prediction", "AnyIntentError", "AccessError", "__version__"]':
        '__all__ = ["AnyIntent", "Prediction", "CheckResult", "ScoresResult", "AnyIntentError", "AccessError", '
        '"__version__"]',
}


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _sha_list():
    return json.loads(_read(SHA_LIST).decode("utf-8"))


# ---- 1. the vendored copy ------------------------------------------------------------------------------------------
def _git_blobs():
    """{file: git blob id} of lib/anyintent/src/anyintent at the pre-change commit, or None without a git checkout
    that holds the commit (the Kaggle smoke has none)"""
    try:
        top = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                             encoding="utf-8", timeout=30)
        if top.returncode != 0:
            return None
        out = subprocess.run(["git", "-C", top.stdout.strip(), "ls-tree", PRE_COMMIT, "lib/anyintent/src/anyintent/"],
                             capture_output=True, text=True, encoding="utf-8", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return {os.path.basename(r[3]): r[2] for r in (line.split() for line in out.stdout.splitlines())}


def _git_blob_bytes(blob):
    """the bytes of a git blob of this checkout"""
    out = subprocess.run(["git", "-C", HERE, "cat-file", "blob", blob], capture_output=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _rename_diff_sha256(old, new, name):
    """sha256 of the unified diff without context lines from the commit's file (old) to the copy (new)"""
    d = difflib.unified_diff(old.decode("utf-8").splitlines(keepends=True), new.decode("utf-8").splitlines(keepends=True),
                             fromfile=f"commit/{name}", tofile=f"copy/{name}", n=0)
    return hashlib.sha256("".join(d).encode("utf-8")).hexdigest()


def test_vendored_copy_matches_its_sha256_list():
    doc = _sha_list()
    assert doc["commit"] == PRE_COMMIT and doc["source"] == "lib/anyintent/src/anyintent"
    have = sorted(f for f in os.listdir(OLD_DIR) if not f.startswith(".") and f != "__pycache__")
    assert have == sorted(doc["files"]), "the vendored copy holds other files than its list"
    assert set(doc["renamed"]) <= set(doc["files"]), doc["renamed"]
    for name, meta in doc["files"].items():
        data = _read(os.path.join(OLD_DIR, name))
        assert hashlib.sha256(data).hexdigest() == meta["sha256"] and len(data) == meta["bytes"], name
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == meta["git_blob"], name
        # the files without the rename edit are the commit's bytes; the edited ones carry the sha256 of their edit
        assert (meta["git_blob"] == meta["commit_git_blob"]) == (name not in doc["renamed"]), name
        assert ("rename_diff_sha256" in meta) == (name in doc["renamed"]), name
    blobs = _git_blobs()                       # in the repository: the list's commit blobs are the files of the commit
    if blobs is not None:
        assert blobs == {n: m["commit_git_blob"] for n, m in doc["files"].items()}
        for name in doc["renamed"]:            # and each edited file is the commit's file with exactly that edit
            meta = doc["files"][name]
            got = _rename_diff_sha256(_git_blob_bytes(meta["commit_git_blob"]), _read(os.path.join(OLD_DIR, name)), name)
            assert got == meta["rename_diff_sha256"], name
    print(f"VENDORED anyintent_pre: {len(have)} files equal to their sha256 list; git commit check "
          f"{'done' if blobs is not None else 'not available here'}")


# ---- 2. source -----------------------------------------------------------------------------------------------------
def _stmts(text, body=None):
    """[(statement, its source text with decorators, its AST dump)] of a module (or of a class body)"""
    lines = text.splitlines(keepends=True)
    out = []
    for node in (ast.parse(text).body if body is None else body):
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        out.append((node, "".join(lines[start - 1:node.end_lineno]), ast.dump(node, include_attributes=False)))
    return out


def _name(node, seg):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if len(names) == 1:
            return names[0]
    return seg.strip()


def _class_header(text, node):
    lines = text.splitlines(keepends=True)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "".join(lines[start - 1:node.body[0].lineno - 1])


def _subsequence(old, new, where, allowed, old_text=None, new_text=None, classes=None):
    """every old statement appears unchanged (text and AST) in new, in the same order; the statements of new that are
    not matched are additions, whose names must be in `allowed`. classes: names of classes that may gain statements in
    their body (their header unchanged, the body checked the same way)"""
    j = 0
    added = []
    for node, seg, dump in old:
        while j < len(new):
            n2, seg2, dump2 = new[j]
            j += 1
            if seg2 == seg and dump2 == dump:
                break
            if classes and isinstance(node, ast.ClassDef) and isinstance(n2, ast.ClassDef) and \
                    n2.name == node.name and node.name in classes:
                assert _class_header(old_text, node) == _class_header(new_text, n2), f"{where}: header of {node.name}"
                assert [ast.dump(d) for d in node.decorator_list] == [ast.dump(d) for d in n2.decorator_list]
                assert [ast.dump(b) for b in node.bases] == [ast.dump(b) for b in n2.bases]
                body_added = _subsequence(_stmts(old_text, node.body), _stmts(new_text, n2.body),
                                          f"{where}: class {node.name}", classes[node.name])
                assert body_added, f"{where}: class {node.name} differs without additions"
                break
            added.append(_name(n2, seg2))
        else:
            raise AssertionError(f"{where}: the pre-change statement {_name(node, seg)!r} is changed, moved or gone")
    added += [_name(n2, seg2) for n2, seg2, _ in new[j:]]
    extra = sorted(set(added) - set(allowed))
    assert not extra, f"{where}: new statements the entry does not allow: {extra}"
    assert len(added) == len(set(added)), f"{where}: a name added twice: {added}"
    return added


def test_every_pre_change_definition_is_unchanged():
    old_files = sorted(f for f in os.listdir(OLD_DIR) if f.endswith(".py"))
    new_files = sorted(f for f in os.listdir(NEW_DIR) if f.endswith(".py"))
    assert new_files == sorted(old_files + NEW_FILES), "modules added or removed"
    assert sorted(UNCHANGED_FILES + ADDITIVE_FILES + [VERSION_FILE, "__init__.py"]) == old_files
    for f in UNCHANGED_FILES:
        assert _read(os.path.join(OLD_DIR, f)) == _read(os.path.join(NEW_DIR, f)), f"{f} changed"
    assert _read(os.path.join(OLD_DIR, VERSION_FILE)) == f'__version__ = "{PRE_CHANGE_VERSION}"\n'.encode()
    assert _read(os.path.join(NEW_DIR, VERSION_FILE)) == f'__version__ = "{RELEASE_VERSION}"\n'.encode(), \
        f"{VERSION_FILE} holds more than the release version string"
    for f in ADDITIVE_FILES:
        old_text = _read(os.path.join(OLD_DIR, f)).decode("utf-8")
        new_text = _read(os.path.join(NEW_DIR, f)).decode("utf-8")
        added = _subsequence(_stmts(old_text), _stmts(new_text), f, ALLOWED_NEW_TOP[f], old_text, new_text,
                             classes={c: m for (ff, c), m in ALLOWED_NEW_METHODS.items() if ff == f})
        assert set(added) == ALLOWED_NEW_TOP[f], f"{f}: additions {sorted(added)}"
        # text: the old lines are all kept, in order; lines were only inserted
        ops = difflib.SequenceMatcher(None, old_text.splitlines(), new_text.splitlines(), autojunk=False).get_opcodes()
        assert {op for op, *_ in ops} <= {"equal", "insert"}, f"{f}: lines changed or deleted: " + str(
            [o for o in ops if o[0] not in ("equal", "insert")][:3])
    # model.py: AnyIntent gained exactly the allowed methods; Prediction and every old method are unchanged
    new_model = ast.parse(_read(os.path.join(NEW_DIR, "model.py")).decode("utf-8"))
    cls = {n.name: n for n in new_model.body if isinstance(n, ast.ClassDef)}
    old_model = ast.parse(_read(os.path.join(OLD_DIR, "model.py")).decode("utf-8"))
    old_cls = {n.name: n for n in old_model.body if isinstance(n, ast.ClassDef)}
    members = lambda c: {_name(n, "") for n in c.body if not isinstance(n, ast.Expr)}   # noqa: E731
    assert members(cls["AnyIntent"]) - members(old_cls["AnyIntent"]) == ALLOWED_NEW_METHODS[("model.py", "AnyIntent")]
    assert ast.dump(cls["Prediction"]) == ast.dump(old_cls["Prediction"])
    # __init__.py: the import line of .model and __all__ change, nothing else
    old_init = _stmts(_read(os.path.join(OLD_DIR, "__init__.py")).decode("utf-8"))
    new_init = _stmts(_read(os.path.join(NEW_DIR, "__init__.py")).decode("utf-8"))
    assert len(old_init) == len(new_init)
    for (n1, s1, d1), (n2, s2, d2) in zip(old_init, new_init):
        if s1.strip() in INIT_CHANGES:
            assert s2.strip() == INIT_CHANGES[s1.strip()]
        else:
            assert s1 == s2 and d1 == d2, s1


# the instance attributes the new code may set (its own, none of them used by the pre-change code)
NEW_SELF_ATTRS = {"_calib_doc", "_calib_path", "_calib_fetched", "_calib_weights_sha256", "_check_cache",
                  "_check_warned_no_description", "_check_warned_bare_name", "_scores_warned_no_description"}
MUTATORS = {"append", "extend", "insert", "pop", "popitem", "remove", "clear", "update", "setdefault", "sort",
            "reverse", "put", "prepare_labels", "set_labels", "__setitem__", "__delitem__", "__setattr__",
            "__delattr__"}
STATE_SETTERS = {"seterr", "set_printoptions", "set_num_threads", "set_num_interop_threads", "set_default_dtype",
                 "manual_seed", "seed", "set_grad_enabled", "use_deterministic_algorithms", "simplefilter",
                 "filterwarnings", "resetwarnings", "basicConfig", "setLevel", "disable", "putenv", "chdir",
                 "__setattr__", "__delattr__"}


def _root(node):
    """the name at the root of an attribute or item chain, and the attribute names from it"""
    chain = []
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute):
            chain.append(node.attr)
        node = node.value
    return (node.id if isinstance(node, ast.Name) else None), chain[::-1]


def _flat(t):
    if isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            yield from _flat(e)
    elif isinstance(t, ast.Starred):
        yield from _flat(t.value)
    else:
        yield t


def _module_names(tree):
    """(imported names, other module-level names) of a module"""
    imported, names = set(), set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imported |= {a.asname or a.name.split(".")[0] for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                names |= {x.id for x in _flat(t) if isinstance(x, ast.Name)}
    return imported, names


def _state_writes(nodes, imported, module_names):
    """the places in `nodes` that write state the old code could see: global or nonlocal; an assignment or deletion of
    any attribute but the instance's own new ones (NEW_SELF_ATTRS), of an item of an imported module, of a module-level
    object or of the instance; setattr or delattr on anything but the instance's own new attributes; a mutating method
    called on an imported module, a module-level object or an attribute of the instance that is not its own; a call that
    changes a process-wide setting"""
    bad = []
    for top in nodes:
        imp = set(imported) | {a.asname or a.name.split(".")[0] for n in ast.walk(top)
                               if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        shared = imp | set(module_names)
        for n in ast.walk(top):
            at = f"line {getattr(n, 'lineno', '?')}"
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                bad.append(f"{at}: {type(n).__name__.lower()} {n.names}")
            targets = []
            if isinstance(n, (ast.Assign, ast.Delete)):
                targets = n.targets
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.comprehension, ast.NamedExpr)):
                targets = [n.target]
            elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                targets = [n.optional_vars]
            for t in (x for tt in targets for x in _flat(tt)):
                if isinstance(t, ast.Attribute):
                    if not (isinstance(t.value, ast.Name) and t.value.id == "self" and t.attr in NEW_SELF_ATTRS):
                        bad.append(f"{at}: writes {ast.unparse(t)}")
                elif isinstance(t, ast.Subscript):
                    root, _ = _root(t)
                    if root is None or root == "self" or root in shared:
                        bad.append(f"{at}: writes {ast.unparse(t)}")
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Name) and f.id in ("setattr", "delattr"):
                obj = n.args[0] if n.args else None
                name = n.args[1] if len(n.args) > 1 else None
                if not (isinstance(obj, ast.Name) and obj.id == "self") or \
                        (isinstance(name, ast.Constant) and name.value not in NEW_SELF_ATTRS):
                    bad.append(f"{at}: {ast.unparse(n)}")
            if isinstance(f, ast.Attribute):
                root, chain = _root(f.value)
                if f.attr in STATE_SETTERS:
                    bad.append(f"{at}: {ast.unparse(n)}")
                elif f.attr in MUTATORS and (root is None or root in shared or
                                             (root == "self" and not (chain and chain[0] in NEW_SELF_ATTRS))):
                    bad.append(f"{at}: {ast.unparse(n)}")
                if f.attr == "_warn_once" and root == "self" and not chain:
                    flag = n.args[0] if n.args else None
                    if not (isinstance(flag, ast.Constant) and flag.value in NEW_SELF_ATTRS):
                        bad.append(f"{at}: {ast.unparse(n)}")
    return bad


def _snippet_writes(src):
    tree = ast.parse(src)
    imported, names = _module_names(tree)
    return _state_writes(tree.body, imported, names)


def test_the_guard_of_state_writes_finds_them():
    for src in ("import numpy as np\ndef f(self):\n    np.pi = 3",
                "from . import fusion\nfusion.fused_scores = None",
                "def f(self):\n    from . import fusion\n    setattr(fusion, 'fused_scores', None)",
                "def f(self):\n    self.names = []", "def f(self):\n    del self.L",
                "def f(self):\n    self.active['w'] = 1", "def f(self):\n    self.cache.mem.clear()",
                "def f(self):\n    self.backend.prepare_labels([])", "def f(self):\n    self.__dict__['names'] = []",
                "import os\ndef f(self):\n    os.environ['X'] = '1'", "def f(self):\n    setattr(self, 'names', [])",
                "def f(self):\n    setattr(self.backend, 'x', 1)",
                "import torch\ndef f(self):\n    torch.set_num_threads(1)",
                "X = []\ndef f(self):\n    X.append(1)", "def f(self):\n    global Y\n    Y = 1",
                "def f(self, p):\n    p.label = 'x'", "def f(self):\n    object.__setattr__(self, 'L', 1)",
                "def f(self):\n    self._warn_once('names', 'x')"):
        assert _snippet_writes(src), src
    assert _snippet_writes("def f(self):\n    self._calib_doc = 1\n    a = {}\n    a['x'] = 1\n    out = []\n"
                           "    out.append(1)\n    self._check_cache.put(1, 2, 3)\n    setattr(self, flag, True)") == []


def test_the_new_code_writes_no_state_of_the_old_code():
    found = {}
    tree = ast.parse(_read(os.path.join(NEW_DIR, "calibration.py")).decode("utf-8"))
    imported, names = _module_names(tree)
    found["calibration.py"] = _state_writes(tree.body, imported, names)
    for f in ADDITIVE_FILES:
        tree = ast.parse(_read(os.path.join(NEW_DIR, f)).decode("utf-8"))
        imported, names = _module_names(tree)
        nodes = [n for n in tree.body if _name(n, "") in ALLOWED_NEW_TOP[f]]
        assert {_name(n, "") for n in nodes} == ALLOWED_NEW_TOP[f], f
        for (ff, c), allowed in ALLOWED_NEW_METHODS.items():
            if ff == f:
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == c)
                meths = [n for n in cls.body if _name(n, "") in allowed]
                assert {_name(n, "") for n in meths} == allowed, (f, c)
                nodes += meths
        found[f] = _state_writes(nodes, imported, names)
    assert not any(found.values()), found
    # the instance attributes the new code sets are not used anywhere in the pre-change library
    old_attrs = set()
    for name in os.listdir(OLD_DIR):
        if name.endswith(".py"):
            for n in ast.walk(ast.parse(_read(os.path.join(OLD_DIR, name)).decode("utf-8"))):
                if isinstance(n, ast.Attribute):
                    old_attrs.add(n.attr)
                elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                    old_attrs.add(n.value)
    assert not (NEW_SELF_ATTRS & old_attrs), NEW_SELF_ATTRS & old_attrs


def test_new_names_resolve():
    for name in ("CheckResult", "ScoresResult"):
        assert getattr(NEW, name) is getattr(NEW.model, name) and name in NEW.__all__
    assert NEW.hub.OPTIONAL_FILES == [calib.CALIBRATION_FILE] and callable(NEW.hub.fetch_optional)
    assert NEW.hub.MODEL_FILES == OLD.hub.MODEL_FILES and NEW.hub.FINGERPRINT_FILES == OLD.hub.FINGERPRINT_FILES
    assert (OLD.__version__, NEW.__version__) == (PRE_CHANGE_VERSION, RELEASE_VERSION)
    assert set(OLD.__all__) < set(NEW.__all__)


# ---- 3. in process: old and new side by side -----------------------------------------------------------------------
def _outcome(fn):
    try:
        r = fn()
    except Exception as e:      # the same refusal must come from both libraries
        return ("raised", type(e).__name__, str(e))
    rs = r if isinstance(r, list) else [r]
    return [(type(p).__name__, [f.name for f in dataclasses.fields(p)], dataclasses.asdict(p)) for p in rs]


def _grid(clf):
    """classify on every setting of the grid: k 3, 5, 8, cascade None, False, True, return_scores false and true;
    the batch of all messages and every message alone"""
    out = {}
    for k in (3, 5, 8):
        for cascade in (None, False, True):
            for rs in (False, True):
                out[(k, cascade, rs, "batch")] = _outcome(
                    lambda: clf.classify(MESSAGES, k=k, cascade=cascade, return_scores=rs, batch_size=8))
                out[(k, cascade, rs, "single")] = [
                    _outcome(lambda: clf.classify(m, k=k, cascade=cascade, return_scores=rs)) for m in MESSAGES]
    out["default"] = _outcome(lambda: clf.classify(MESSAGES))
    try:
        out["predict"] = clf.predict(MESSAGES[:5]) + [clf.predict(MESSAGES[7])]
    except Exception as e:
        out["predict"] = ("raised", type(e).__name__, str(e))
    return out


def _assert_predictions(ref, case):
    """the comparison is not vacuous: every setting of the grid without cascade=True, the default call and predict
    returned Predictions (labels for predict); only the cascade may be refused (a model without a tau)"""
    n = 0
    for kk, v in ref.items():
        if isinstance(kk, tuple) and kk[1] is not True:
            for x in (v if kk[3] == "single" else [v]):
                assert isinstance(x, list) and x and all(t[0] == "Prediction" for t in x), (case, kk, x)
                n += len(x)
    assert isinstance(ref["default"], list) and all(t[0] == "Prediction" for t in ref["default"]), case
    assert isinstance(ref["predict"], list) and len(ref["predict"]) == 6, case
    assert n == 3 * 2 * 2 * 2 * len(MESSAGES), (case, n)   # k; cascade None, False; return_scores; batch, single


def _same(a, b, what):
    assert a.keys() == b.keys()
    bad = [k for k in a if a[k] != b[k]]
    assert not bad, f"{what}: Predictions differ at {bad[:3]}"


def _new_calls(clf):
    """calls of the new functions on the new instance: their results (or refusals) are recorded, not compared"""
    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for call in (lambda: clf.check(MESSAGES, LABELS[0]), lambda: clf.check(MESSAGES[2], "greeting"),
                     lambda: clf.check(MESSAGES[:3], ("lost card", "A card was lost.")),
                     lambda: clf.scores(MESSAGES), lambda: clf.scores(MESSAGES[5]), lambda: clf.calibration):
            try:
                call()
                seen.append("ran")
            except NEW.AnyIntentError:
                seen.append("refused")
    return seen


@pytest.fixture(scope="module")
def models(tmp_path_factory, tok, one_file, two_file, final_file):
    base = tmp_path_factory.mktemp("identity")
    _, p_like = make_p_like(str(base / "p_like"), tok, seed=13)
    out = {"p_cal": model_copy(p_like, base / "p_cal"),
           "p_plain": model_copy(p_like, base / "p_plain", calibrated=False),
           "one_file": one_file, "two_file": two_file, "final": final_file}
    d = model_copy(one_file, base / "one_file_cal", calibrated=False)
    doc = calibration_doc(d, functions=("check",))
    doc["fusion"] = {"w": 10.0, "k": 8, "label_text_mode": "v2"}
    write_calibration(d, doc)
    out["one_file_cal"] = d
    fsd = base / "p_cal_fewshot"
    shutil.copytree(out["p_cal"], fsd)
    (fsd / "anyintent.json").write_text(json.dumps({"format": "anyintent-one-file", "fewshot": FEWSHOT_P}), encoding="utf-8")
    out["p_cal_fewshot"] = str(fsd)
    return out


EXAMPLES = {"order status": ["where is my order", "parcel still not here"], "lost card": ["I lost my card"],
            "refund request": ["money back please", "refund for the broken lamp", "can I get a refund"]}
CASES = {   # model, engine, examples, what the new functions do on it
    "p_cal_fast": ("p_cal", "auto", None, ["ran"] * 6),
    "p_cal_eager": ("p_cal", "eager", None, ["ran"] * 6),
    "p_plain": ("p_plain", "auto", None, ["refused"] * 5 + ["ran"]),
    "one_file": ("one_file", "auto", None, ["refused"] * 5 + ["ran"]),
    "one_file_cal": ("one_file_cal", "auto", None, ["ran"] * 3 + ["refused"] * 2 + ["ran"]),
    "two_file": ("two_file", "auto", None, ["refused"] * 5 + ["ran"]),
    "final_cascade": ("final", "eager", None, ["refused"] * 5 + ["ran"]),
    "p_cal_fewshot": ("p_cal_fewshot", "auto", EXAMPLES, ["ran"] * 3 + ["refused"] * 2 + ["ran"]),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_old_and_new_classify_identical(case, models, monkeypatch):
    # the label cache key holds the library version (a new release gets new cache entries; checked in
    # test_the_label_key_differs_only_by_the_library_version), so the new library runs here under the old version
    # string: the label key is then compared like every other field of the label state
    monkeypatch.setattr(NEW.model, "__version__", OLD.__version__)
    key, engine, examples, expect_calls = CASES[case]
    path = models[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = OLD.AnyIntent.from_pretrained(path, device="cpu", engine=engine)
        new = NEW.AnyIntent.from_pretrained(path, device="cpu", engine=engine)
    assert old.engine == new.engine and old.revision == new.revision and old.config == new.config
    for c in (old, new):
        c.set_labels(LABELS, examples=examples)
    assert (old.names, old.texts, old.label_key, old.mode, old.active) == \
        (new.names, new.texts, new.label_key, new.mode, new.active) and np.array_equal(old.L, new.L)
    ref = _grid(old)
    _assert_predictions(ref, case)
    _same(ref, _grid(old), f"{case}: control (the old library run twice)")
    _same(ref, _grid(new), f"{case}: old against new, before check and scores")
    assert _new_calls(new) == expect_calls
    _same(ref, _grid(new), f"{case}: old against new, after check and scores")
    if examples is not None:                                   # few-shot off again on both
        for c in (old, new):
            c.set_labels(LABELS)
        ref = _grid(old)
        _assert_predictions(ref, f"{case}, zero-shot")
        _same(ref, _grid(new), f"{case}: zero-shot after the few-shot label set")
        assert _new_calls(new)[3:5] == ["ran", "ran"]
        _same(ref, _grid(new), f"{case}: zero-shot, after check and scores")
    n_cascade = sum(1 for kk, v in ref.items() if isinstance(kk, tuple) and kk[1] is True and isinstance(v, list))
    if key == "final" and examples is None:
        assert n_cascade > 0, "the cascade ran nowhere on the model with a tau"


def test_the_label_key_differs_only_by_the_library_version(models):
    """the release changes the version string, which is one field of the label cache key: the key fields of the old and
    the new library are the same except that field, so the release only gets new cache entries"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = OLD.AnyIntent.from_pretrained(models["p_cal"], device="cpu")
        new = NEW.AnyIntent.from_pretrained(models["p_cal"], device="cpu")
    for c in (old, new):
        c.set_labels(LABELS)
    fo = old._key_fields("embed", old.cfg.label_max_len, old.texts)
    fn = new._key_fields("embed", new.cfg.label_max_len, new.texts)
    assert set(fo) == set(fn) and {k for k in fo if fo[k] != fn[k]} == {"lib"}
    assert (fo["lib"], fn["lib"]) == (PRE_CHANGE_VERSION, RELEASE_VERSION)
    assert old.label_key == OLD.cache.cache_key(**fo) and new.label_key == NEW.cache.cache_key(**fn)
    assert old.label_key != new.label_key


# ---- 4. state ------------------------------------------------------------------------------------------------------
def _state(c, cache_dir):
    be = c.backend
    pairs = getattr(be, "pairs", None)
    return {
        "ids": [id(c.names), id(c.texts), id(c.L), id(c.active), id(c.cache), id(c.cache.mem), id(c.cache.hits),
                id(getattr(be, "lab_texts", None)), id(getattr(pairs, "lab_texts", None)),
                id(getattr(pairs, "lab_ids", None))] + [id(v) for v in c.cache.mem.values()],
        "label_state": (list(c.names), list(c.texts), c.label_key, c.mode, c.fs, dict(c.active)),
        "L": c.L.copy(),
        "backend": (list(getattr(be, "lab_texts", [])), list(getattr(pairs, "lab_texts", [])),
                    [list(x) for x in getattr(pairs, "lab_ids", [])], getattr(pairs, "ok", None)),
        "cache": (list(c.cache.mem.keys()), dict(c.cache.hits), c.cache.max_items, c.cache.cache_dir,
                  sorted(os.listdir(cache_dir))),
        "fallback": getattr(pairs, "n_fallback", None),
    }


def _equal_state(a, b, skip=()):
    for k in a:
        if k in skip:
            continue
        if k == "L":
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


@pytest.mark.parametrize("engine", ["eager", "fast"])
def test_check_and_scores_leave_the_label_state_and_the_cache_alone(engine, models, tmp_path):
    cache_dir = str(tmp_path / "cache")
    c = NEW.AnyIntent.from_pretrained(models["p_cal"], device="cpu", engine=engine, cache_dir=cache_dir)
    assert c.engine == engine
    c.set_labels(LABELS[:6])
    c.set_labels(LABELS)                                       # two label sets in the shared cache
    before = _state(c, cache_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lab in (LABELS[0], "greeting", LABELS[3]["name"], ("new intent", "Something no label is about."),
                    {"name": "lost card", "description": "Another description of a current label."}):
            c.check(MESSAGES, lab)
            c.check(MESSAGES[1], lab)
        c._pair_raw(MESSAGES[:4], "lost card: A card was lost.")
    _equal_state(before, _state(c, cache_dir))                 # check: everything, the fallback counter included
    assert c._check_cache is not c.cache and c._check_cache.cache_dir is None
    assert len(c._check_cache.mem) == 6 and not (set(c._check_cache.mem) & set(c.cache.mem))
    x0 = before["fallback"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.scores(MESSAGES)
        c.scores(MESSAGES[4])
    after = _state(c, cache_dir)
    _equal_state(before, after, skip=("fallback",))
    x1 = after["fallback"]
    c.classify(MESSAGES, cascade=False, return_scores=True)
    c.classify(MESSAGES[4], cascade=False, return_scores=True)
    x2 = _state(c, cache_dir)["fallback"]
    assert (x1 is None and x2 is None) or x2 - x1 == x1 - x0   # scores changes only what classify changes
