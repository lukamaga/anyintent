"""Serving core shared by the HTTP server (anyintent.server, extra 'server') and the MCP server (anyintent.mcp_server,
extra 'mcp'). Plain Python on the public library API, no web or MCP package needed:

- one AnyIntent instance (given, or loaded by a loader, optionally in a background thread);
- input validation with limits (message length, messages per request, intents per request, name and description
  length), with clear messages;
- named intent lists, in memory or one JSON file per list under a folder (written atomically, read again at start);
- AnyIntent.check and AnyIntent.scores, used only when the installed library has them (detected by attribute); they
  answer only with a model whose calibration file names them: a refusal of the library for a missing calibration
  becomes a clear CalibrationMissing (HTTP 409), decided per call from the refusal and the model's calibration status
  (AnyIntent.calibration, read without a forward pass), and health lists the functions the loaded model supports;
- privacy: message texts are never logged unless log_texts is on; an unexpected error is logged with its traceback
  frames and its type, not its message (the message may hold a text); a refusal of check or scores reaches clients
  without the model's local folder paths (the log keeps them);
- the Host header check of the HTTP servers (a guard against DNS rebinding), as plain ASGI code (HostGuard).

Thread safety: a label set and the classification that uses it run under one lock (the instance's own lock, the one
AnyIntent.set_labels and AnyIntent.classify hold, so the pair is atomic). The service owns its instance: it calls
set_labels again only when the label set of a request differs from the last one it applied.
"""
import ipaddress
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass

from ._version import __version__
from .calibration import CALIBRATION_FILE
from .hub import AnyIntentError

log = logging.getLogger("anyintent.server")

LIST_FORMAT = "anyintent-intent-list"
NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")      # used with fullmatch (no trailing newline slips through)
NAME_RULE = ("a list name has 1 to 64 characters: lowercase letters, digits, '_' and '-', starting with a letter or a "
             "digit")
_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
OPTIONAL_FEATURES = ("check", "scores")
LABEL_TEXT_MODES = ("v2", "v3s", "noex", "name")       # anyintent.labels.MODES (checked against it in the tests)
# the settings of AnyIntent.config shown by health (no local paths, no label state)
CONFIG_KEYS = ("version", "layout", "w", "k", "tau", "cascade_first", "label_text_mode", "v3s_variant", "msg_max_len",
               "label_max_len", "pair_max_len", "model_license", "revision", "embedder_revision", "engine", "family",
               "device", "dtype")


@dataclass(frozen=True)
class Limits:
    max_batch: int = 64                 # messages per request
    max_text_chars: int = 2000          # characters per message
    max_intents: int = 200              # intents per request or per saved list
    max_name_chars: int = 200           # characters of an intent name
    max_description_chars: int = 2000   # characters of an intent description
    max_lists: int = 1000               # saved intent lists
    max_store_bytes: int = 50_000_000   # all saved intent lists together (their JSON, in bytes)

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"limit {key}: expected an integer >= 1, got {value!r}")

    @property
    def max_body_bytes(self):
        """Request body cap of the HTTP server: the largest valid request, counting 6 bytes per character (a JSON
        \\u escape), plus 64 KiB for the JSON structure."""
        chars = self.max_batch * self.max_text_chars + self.max_intents * (self.max_name_chars +
                                                                           self.max_description_chars + 32)
        return 6 * chars + 65536


class ServiceError(Exception):
    """An error the client caused or can act on; status is the HTTP status, code a stable machine-readable name."""
    status = 400
    code = "bad_request"

    def __init__(self, message):
        super().__init__(message)
        self.message = str(message)


class InvalidRequest(ServiceError):
    status, code = 422, "invalid_request"


class LimitExceeded(ServiceError):
    status, code = 413, "limit_exceeded"


class Unauthorized(ServiceError):
    status, code = 401, "unauthorized"


class NotFound(ServiceError):
    status, code = 404, "not_found"


class NotAvailable(ServiceError):
    status, code = 501, "not_available"


class NotReady(ServiceError):
    status, code = 503, "not_ready"


class ModelRefused(ServiceError):
    """The library refused the request (AnyIntentError); the message is the library's."""
    status, code = 422, "model_refused"


class CalibrationMissing(ServiceError):
    """check or scores on a model whose calibration file is missing or does not name the function: the model release
    does not ship that function (classify still works)."""
    status, code = 409, "calibration_missing"


def jsonable(x):
    """A JSON-safe copy: numpy and torch scalars become Python numbers, NaN and infinities become None."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    item = getattr(x, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return jsonable(tolist())
    return str(x)


def check_result(r):
    """An AnyIntent.check result (CheckResult) as the servers return it: intent, yes, probability, score, threshold."""
    return {"intent": str(r.intent), "yes": bool(r.yes), "probability": jsonable(float(r.probability)),
            "score": jsonable(float(r.score)), "threshold": jsonable(float(r.threshold))}


def scores_result(r):
    """An AnyIntent.scores result (ScoresResult) as the servers return it: {"label", "intents": [{"intent", "value"}]}.
    intents lists only the intents the model read with the message (r.read, at most 5), from the highest value to the
    lowest (ties: the label first, then retrieval order), each value exactly r.values[name]; the label is r.label, the
    first one listed. An intent not listed was not read (its value is 0.0). The remainder is not sent: the listed
    values sum to 1 minus r.remainder, so they can sum to less than 1 (the missing part is not a 'none of these
    intents' answer)."""
    pos = {name: i for i, name in enumerate(r.read)}
    order = sorted(pos, key=lambda name: (-float(r.values[name]), name != r.label, pos[name]))
    return {"label": str(r.label),
            "intents": [{"intent": str(name), "value": jsonable(float(r.values[name]))} for name in order]}


def _kind(x):
    return {str: "a string", dict: "an object", list: "a list", tuple: "a list", bool: "a boolean", int: "a number",
            float: "a number", type(None): "null"}.get(type(x), type(x).__name__)


def check_list_name(name):
    if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name in _RESERVED:
        shown = name if isinstance(name, str) and len(name) <= 80 else _kind(name)
        raise InvalidRequest(f"invalid list name {shown!r}: {NAME_RULE} (and not a reserved device name such as "
                             "'con' or 'nul')")
    return name


def _list_bytes(intents):
    return len(json.dumps(intents, ensure_ascii=False).encode("utf-8"))


def _fsync_dir(directory):
    """Flushes a rename or a removal in the folder to disk (best effort; not every system can open a folder)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class ListStore:
    """Named intent lists in memory; with a folder, also one JSON file per list (read at start, written atomically,
    readable by the owner only). max_bytes caps all lists together."""

    def __init__(self, directory=None, max_lists=1000, validate=None, max_bytes=Limits.max_store_bytes):
        self.directory = os.path.abspath(os.path.expanduser(directory)) if directory else None
        self.max_lists = int(max_lists)
        self.max_bytes = int(max_bytes)
        self._lists = {}
        self._sizes = {}
        self._lock = threading.Lock()
        if self.directory:
            os.makedirs(self.directory, mode=0o700, exist_ok=True)
            self._read_all(validate)

    @property
    def persistent(self):
        return self.directory is not None

    @property
    def count(self):
        """The number of saved lists (no lock: a dict's length is read atomically)."""
        return len(self._lists)

    def _path(self, name):
        return os.path.join(self.directory, check_list_name(name) + ".json")

    def _read_all(self, validate):
        for f in sorted(os.listdir(self.directory)):
            if not f.endswith(".json") or f.startswith("."):
                continue
            name, p = f[:-5], os.path.join(self.directory, f)
            try:
                check_list_name(name)
                with open(p, encoding="utf-8-sig") as fh:      # a BOM, as Windows editors write it, is accepted
                    d = json.load(fh)
                if not isinstance(d, dict) or d.get("format") != LIST_FORMAT or d.get("name") != name:
                    raise ValueError(f"not an intent list file of this server (format {LIST_FORMAT!r}, name {name!r})")
                intents = validate(d.get("intents")) if validate else d.get("intents")
            except (OSError, ValueError, ServiceError) as e:
                log.warning("intent list file %s skipped: %s", p, getattr(e, "message", e))
                continue
            if len(self._lists) >= self.max_lists:
                log.warning("intent list folder %s holds more than %d lists; the rest are not loaded", self.directory,
                            self.max_lists)
                break
            size = _list_bytes(intents)
            if sum(self._sizes.values()) + size > self.max_bytes:
                log.warning("intent list file %s skipped: the saved lists would exceed %d bytes", p, self.max_bytes)
                continue
            self._lists[name], self._sizes[name] = intents, size
        log.info("intent lists: %d loaded from %s", len(self._lists), self.directory)

    def _write(self, name, intents):
        """Atomic: a new file with a random name (created exclusively, mode 0600), written, flushed to disk, then
        renamed over the old one."""
        p = self._path(name)
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix="." + name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"format": LIST_FORMAT, "version": 1, "name": name, "intents": intents}, fh,
                          ensure_ascii=False, indent=1)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        _fsync_dir(self.directory)

    def put(self, name, intents):
        """Saves (or replaces) a list; returns True when it is new."""
        size = _list_bytes(intents)
        with self._lock:
            created = name not in self._lists
            if created and len(self._lists) >= self.max_lists:
                raise LimitExceeded(f"{len(self._lists)} intent lists are saved; the limit is {self.max_lists}: delete "
                                    "one first")
            total = sum(self._sizes.values()) - self._sizes.get(name, 0) + size
            if total > self.max_bytes:
                raise LimitExceeded(f"the saved intent lists would take {total} bytes; the limit is {self.max_bytes}: "
                                    "delete a list first")
            if self.directory:
                self._write(name, intents)
            self._lists[name], self._sizes[name] = [dict(x) for x in intents], size
        return created

    @staticmethod
    def _missing(name):
        return NotFound(f"no saved intent list named {name!r}; list_intent_lists (MCP) or GET /lists (HTTP) shows the "
                        "saved ones")

    def get(self, name):
        with self._lock:
            if name not in self._lists:
                raise self._missing(name)
            return [dict(x) for x in self._lists[name]]

    def delete(self, name):
        with self._lock:
            if name not in self._lists:
                raise self._missing(name)
            if self.directory and os.path.exists(self._path(name)):
                os.remove(self._path(name))
                _fsync_dir(self.directory)
            del self._lists[name]
            self._sizes.pop(name, None)

    def summary(self):
        with self._lock:
            return [{"name": n, "n_intents": len(v)} for n, v in sorted(self._lists.items())]


def public_model_id(model_id):
    """The model id shown to clients: a Hugging Face id as given; for a local folder only its last part (no path)."""
    if model_id is None:
        return None
    m = str(model_id)
    if os.path.isdir(os.path.expanduser(m)):
        return os.path.basename(os.path.normpath(os.path.expanduser(m))) + " (local folder)"
    return m


def check_label_text_mode(mode):
    if mode is not None and mode not in LABEL_TEXT_MODES:
        raise ValueError(f"label text mode {mode!r}: expected one of {', '.join(LABEL_TEXT_MODES)}")
    return mode


def check_doubt_margin(x):
    if x is None:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        raise ValueError(f"doubt margin {x!r}: expected a number > 0 (choose it on your own labelled examples)")
    return float(x)


class Service:
    """classifier: an AnyIntent instance (or an object with its set_labels and classify). Or loader: a callable that
    returns one, run by load_now() or, in a background thread, by start_loading(); until it is done, the calls that
    need the model wait up to load_wait seconds and then raise NotReady.
    label_text_mode: None (the model's own mode) or a mode passed to set_labels (for example "v2" for plain descriptions
    on a model released with "v3s"). doubt_margin: None, or the margin below which a result is marked doubtful (it adds
    "doubtful" to every classify result)."""

    def __init__(self, classifier=None, *, loader=None, model_id=None, lists_dir=None, limits=None, log_texts=False,
                 load_wait=0.0, classifier_class=None, label_text_mode=None, doubt_margin=None):
        if classifier is None and loader is None:
            raise ValueError("Service needs a classifier or a loader")
        self.limits = limits or Limits()
        self.model_id = public_model_id(model_id)
        self.label_text_mode = check_label_text_mode(label_text_mode)
        self.doubt_margin = check_doubt_margin(doubt_margin)
        self.log_texts = bool(log_texts)
        self.load_wait = float(load_wait)
        self._clf = classifier
        self._loader = loader
        self._class = classifier_class
        self._load_error = None
        self._load_started = None
        self._loaded = threading.Event()
        if classifier is not None:
            self._loaded.set()
        self._own_lock = threading.RLock()
        self._applied = None
        self._calib = None              # the model's calibration status (read_calibration), read when it is loaded
        self.lists = ListStore(lists_dir, self.limits.max_lists, self.normalize_intents, self.limits.max_store_bytes)
        if classifier is not None:
            self._calib = self.read_calibration(classifier)

    # ---- model --------------------------------------------------------------------------------------------------
    def load_now(self):
        """Runs the loader in this thread (errors propagate)."""
        if self._clf is None:
            self._load_started = time.time()
            try:
                clf = self._loader()
                self._calib = self.read_calibration(clf)
                self._clf = clf
            finally:
                self._loaded.set()
        return self._clf

    def start_loading(self):
        """Runs the loader in a background thread (a failure is kept and reported by every call that needs it)."""
        if self._clf is not None or self._load_started is not None:
            return
        self._load_started = time.time()
        threading.Thread(target=self._load_in_background, name="anyintent-load", daemon=True).start()

    def _load_in_background(self):
        try:
            clf = self._loader()
            self._calib = self.read_calibration(clf)
        except AnyIntentError as e:
            # the library's own message (access, revision, configuration) says what to do and holds no message text
            self._load_error = f"{type(e).__name__}: {e}"
            log.error("the model could not be loaded: %s", self._load_error)
        except Exception as e:
            # any other error may hold local paths or hub details: clients get its type, the log gets everything
            self._load_error = f"{type(e).__name__} (the server log has the details)"
            log.error("the model could not be loaded", exc_info=(type(e), e, e.__traceback__))
        else:
            self._clf = clf
            log.info("model ready after %.1f s", time.time() - self._load_started)
        finally:
            self._loaded.set()

    @property
    def ready(self):
        return self._clf is not None

    def classifier(self):
        if self._clf is None and self._load_started is not None and self.load_wait > 0:
            self._loaded.wait(self.load_wait)
        if self._clf is not None:
            return self._clf
        if self._load_error:
            raise NotReady(f"the model could not be loaded ({self._load_error}); fix the cause and restart the server")
        if self._load_started is None:
            raise NotReady("the model is not loaded")
        raise NotReady(f"the model is still loading ({time.time() - self._load_started:.0f} s so far; the first start "
                       "downloads it, which can take minutes); try again shortly")

    def _feature_source(self):
        if self._clf is not None:
            return self._clf
        if self._class is None:
            from .model import AnyIntent
            self._class = AnyIntent
        return self._class

    def has(self, feature):
        """True when the classifier (or, before it is loaded, its class) has the method (check, scores)."""
        return callable(getattr(self._feature_source(), feature, None))

    @staticmethod
    def read_calibration(clf):
        """The calibration status of a loaded model, from the library's AnyIntent.calibration property (it reads the
        model's calibration file, no forward pass): {"file": True, "functions": [the functions it names]}, or
        {"file": False, "functions": []} when the model has no calibration file the library can use (none, not
        downloaded, or malformed: the library warns with the reason); None when the classifier has no such property (a
        library without check and scores). Called before the model is shared or with its lock held."""
        try:
            doc = clf.calibration
        except AttributeError:
            return None
        except Exception as e:           # not expected (the property catches the library's own errors)
            log.warning("the calibration status of the model could not be read (%s)", type(e).__name__)
            return {"file": False, "functions": []}
        if not isinstance(doc, dict):
            return {"file": False, "functions": []}
        names = doc.get("functions") or []
        return {"file": True, "functions": [f for f in OPTIONAL_FEATURES if f in names]}

    def supported(self):
        """The functions the loaded model answers (None until it is loaded): classify, and check and scores when the
        library has them and the model's calibration file names them (a library that reads no calibration file: when it
        has them). A call can still be refused with ModelRefused (422) and the library's message, for example with a
        calibration file of another model, a changed model folder or a load override."""
        if not self.ready:
            return None
        calib = self._calib
        return ["classify"] + [f for f in OPTIONAL_FEATURES
                               if self.has(f) and (calib is None or f in calib["functions"])]

    def require(self, feature):
        if not self.has(feature):
            raise NotAvailable(f"'{feature}' is not available: the installed anyintent library (version {__version__}) "
                               f"has no AnyIntent.{feature}. It is used automatically once a library version with it "
                               "is installed; classification works now")

    def _model_lock(self, clf):
        lk = getattr(clf, "_lock", None)
        return lk if lk is not None and hasattr(lk, "__enter__") else self._own_lock

    @staticmethod
    def _call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AnyIntentError as e:
            raise ModelRefused(str(e)) from e

    @staticmethod
    def missing_message(feature, calib):
        """The message of CalibrationMissing: which function, and that the model release does not ship it."""
        if calib["file"]:
            named = " and ".join(calib["functions"]) or "no function"
            return (f"'{feature}' is not available with this model: its release ships no calibration for {feature} "
                    f"(the model's calibration file {CALIBRATION_FILE} names only {named}: {feature} did not meet its "
                    "accuracy rule on the model's development data, so it is not part of this release). classify works "
                    "with this model.")
        return (f"'{feature}' is not available with this model: {feature} needs the calibration file "
                f"{CALIBRATION_FILE} of the model release, and this model has none that the library can use: its "
                "release does not ship one (or the file was not downloaded with the model, or cannot be read; the "
                "server log has the library's message). classify works with this model.")

    @staticmethod
    def without_model_folders(clf, msg):
        """msg with each local folder of the loaded model (its release folder and the folder of its weights, from the
        library's configuration) replaced by its last part in quotes: clients get no local path (health, too, shows a
        local folder only by its name)."""
        cfg = getattr(clf, "cfg", None)
        folders = {str(p) for p in (getattr(cfg, "root", None), getattr(cfg, "cross_dir", None)) if p}
        for p in sorted((p for p in folders if len(p) > 1), key=len, reverse=True):
            msg = msg.replace(p, repr(os.path.basename(os.path.normpath(p))))
        return msg

    def _calibrated_call(self, clf, feature, fn, *args, **kwargs):
        """fn (clf.check or clf.scores), with the model lock held by the caller. A refusal of the library
        (AnyIntentError) is CalibrationMissing (409) when the model's calibration status, read again after the refusal,
        does not name the function (no calibration file, or no block for the function), else ModelRefused (422) with the
        library's message, the model's local folders in it shown by name only (without_model_folders; the log keeps
        them). Decided per call, so a calibration file put next to a local model later is used. The status that health
        shows is updated here."""
        try:
            res = fn(*args, **kwargs)
        except AnyIntentError as e:
            calib = self.read_calibration(clf)
            self._calib = calib
            if calib is not None and feature not in calib["functions"]:
                # the library's message names the file and, for a model from the Hugging Face cache, how to fetch it;
                # it holds no message text (it may hold a local path, so it goes to the operator's log only)
                log.info("%s: not available with this model (the library: %s)", feature, e)
                raise CalibrationMissing(self.missing_message(feature, calib)) from e
            msg = self.without_model_folders(clf, str(e))
            if msg != str(e):
                # the library's message names a local folder (for example a model folder that changed after it was
                # loaded): the client gets it without the path, the operator's log keeps it (it holds no message text)
                log.info("%s refused by the library: %s", feature, e)
            if self.label_text_mode is not None and "label text mode" in msg:
                msg += (f" (this server passes the label text mode {self.label_text_mode!r} to set_labels: "
                        "--label-text-mode)")
            raise ModelRefused(msg) from e
        calib = self._calib
        if calib is not None and feature not in calib["functions"]:
            self._calib = self.read_calibration(clf)        # the file appeared after the status was read
        return res

    def _apply_labels(self, clf, labels):
        """set_labels (with the service's label text mode, when one is set), skipped when this label set is the one
        applied last (and the instance still holds it)."""
        mode = self.label_text_mode
        key = json.dumps([mode, labels], sort_keys=True, ensure_ascii=False)
        if self._applied == (id(clf), key, getattr(clf, "label_key", None)):
            return
        self._applied = None
        if mode is None:
            clf.set_labels(labels)
        else:
            clf.set_labels(labels, label_text_mode=mode)
        self._applied = (id(clf), key, getattr(clf, "label_key", None))

    # ---- validation ---------------------------------------------------------------------------------------------
    @staticmethod
    def _unicode(s, where):
        """A string that can be encoded as UTF-8 (JSON allows a lone surrogate such as \\ud800; files and tokenizers
        do not)."""
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            raise InvalidRequest(f"{where}: not valid Unicode (it holds a lone surrogate code point)") from None
        return s

    def messages(self, text=None, texts=None):
        """Exactly one of text (one message) or texts (a list); returns the list of messages."""
        if (text is None) == (texts is None):
            raise InvalidRequest("give exactly one of 'text' (one message) or 'texts' (a list of messages)")
        if text is not None:
            msgs = [text]
        else:
            if not isinstance(texts, (list, tuple)):
                raise InvalidRequest(f"'texts' must be a list of strings, got {_kind(texts)}")
            msgs = list(texts)
            if not msgs:
                raise InvalidRequest("'texts' is empty: give at least one message")
        L = self.limits
        if len(msgs) > L.max_batch:
            raise LimitExceeded(f"{len(msgs)} messages in one request; the limit is {L.max_batch} (--max-batch)")
        for i, t in enumerate(msgs):
            if not isinstance(t, str):
                raise InvalidRequest(f"message {i}: expected a string, got {_kind(t)}")
            if len(t) > L.max_text_chars:
                raise LimitExceeded(f"message {i} has {len(t)} characters; the limit is {L.max_text_chars} "
                                    "(--max-text-chars)")
            self._unicode(t, f"message {i}")
        return msgs

    def _msgs(self, texts):
        return self.messages(text=texts) if isinstance(texts, str) else self.messages(texts=texts)

    def normalize_intent(self, x, where="intent"):
        """A name, or {"name", "description"}; returns {"name"} or {"name", "description"} with whitespace trimmed."""
        if isinstance(x, str):
            name, desc = x, ""
        elif isinstance(x, dict):
            extra = sorted(str(k) for k in x if k not in ("name", "description"))
            if extra:
                raise InvalidRequest(f"{where}: unknown key(s) {extra}; an intent is a name or an object with 'name' "
                                     "and an optional 'description'")
            if "name" not in x:
                raise InvalidRequest(f"{where}: an intent object needs a 'name'")
            name, desc = x["name"], x.get("description")
            if not isinstance(name, str):
                raise InvalidRequest(f"{where}: 'name' must be a string, got {_kind(name)}")
            desc = "" if desc is None else desc
            if not isinstance(desc, str):
                raise InvalidRequest(f"{where}: 'description' must be a string, got {_kind(desc)}")
        else:
            raise InvalidRequest(f"{where}: expected a name (a string) or an object with 'name' and 'description', got "
                                 f"{_kind(x)}")
        name, desc = name.strip(), desc.strip()
        L = self.limits
        if not name:
            raise InvalidRequest(f"{where}: empty name")
        if len(name) > L.max_name_chars:
            raise LimitExceeded(f"{where}: the name has {len(name)} characters; the limit is {L.max_name_chars}")
        if len(desc) > L.max_description_chars:
            raise LimitExceeded(f"{where}: the description has {len(desc)} characters; the limit is "
                                f"{L.max_description_chars}")
        self._unicode(name, f"{where}: the name")
        self._unicode(desc, f"{where}: the description")
        return {"name": name, "description": desc} if desc else {"name": name}

    def normalize_intents(self, intents):
        if not isinstance(intents, (list, tuple)):
            raise InvalidRequest(f"'intents' must be a list of names or of objects with 'name' and 'description', got "
                                 f"{_kind(intents)}")
        if not intents:
            raise InvalidRequest("'intents' is empty: give at least one intent")
        if len(intents) > self.limits.max_intents:
            raise LimitExceeded(f"{len(intents)} intents; the limit is {self.limits.max_intents} (--max-intents)")
        out, seen = [], {}
        for i, x in enumerate(intents):
            it = self.normalize_intent(x, f"intent {i}")
            if it["name"] in seen:
                raise InvalidRequest(f"intent {i}: the name {it['name']!r} is already intent {seen[it['name']]}; "
                                     "intent names must be unique")
            seen[it["name"]] = i
            out.append(it)
        return out

    def _labels(self, intents, list_name):
        if (intents is None) == (list_name is None):
            raise InvalidRequest("give exactly one of 'intents' (a list of names or of objects with 'name' and "
                                 "'description') or a saved list name ('list' over HTTP, 'list_name' in MCP)")
        if list_name is not None:
            return self.lists.get(check_list_name(list_name))
        return self.normalize_intents(intents)

    def _k(self, k):
        if k is None:
            return None
        if isinstance(k, bool) or not isinstance(k, int):
            raise InvalidRequest(f"'k' must be an integer, got {_kind(k)}")
        if not 1 <= k <= self.limits.max_intents:
            raise InvalidRequest(f"'k' must be between 1 and {self.limits.max_intents}, got {k}")
        return k

    @staticmethod
    def _threshold(t):
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0.0 < float(t) < 1.0:
            raise InvalidRequest(f"'threshold' must be a number between 0 and 1 (exclusive), got {t!r}")
        return float(t)

    # ---- logging ------------------------------------------------------------------------------------------------
    def _log(self, op, msgs, n_intents, t0, labels=None):
        log.info("%s: %d message(s), %d intent(s), %.1f ms", op, len(msgs), n_intents, 1000 * (time.perf_counter() - t0))
        if self.log_texts:
            log.info("%s texts (--log-texts is on): %s -> %s", op, json.dumps(msgs, ensure_ascii=False),
                     json.dumps(labels, ensure_ascii=False))

    def log_internal_error(self, where, exc):
        """An unexpected error: its type and traceback frames; its message only with log_texts (it may hold a text)."""
        if self.log_texts:
            log.error("internal error in %s", where, exc_info=(type(exc), exc, exc.__traceback__))
            return
        frames = "".join(traceback.format_tb(exc.__traceback__)).rstrip()
        log.error("internal error in %s: %s (its message is not logged because it may hold a message text; start "
                  "with --log-texts to log it)\nTraceback, frames only:\n%s", where, type(exc).__name__, frames)

    # ---- operations ---------------------------------------------------------------------------------------------
    @staticmethod
    def _runner_up(p):
        """The label of the second-best scored candidate (None when only one was scored), from return_scores."""
        cands = [c for c in (getattr(p, "candidates", None) or [])
                 if c.get("fused") is not None and int(c.get("index", -1)) != int(p.index)]
        if not cands:
            return None
        return str(max(cands, key=lambda c: float(c["fused"]))["label"])

    def _result(self, p):
        margin = None if p.margin is None else jsonable(float(p.margin))
        r = {"label": str(p.label), "index": int(p.index), "score": jsonable(float(p.score)), "margin": margin,
             "runner_up": self._runner_up(p)}
        if self.doubt_margin is not None:
            r["doubtful"] = margin is not None and margin < self.doubt_margin
        return r

    def classify(self, texts, intents=None, list_name=None, k=None):
        """texts: a string or a list of strings. Returns one {label, index, score, margin, runner_up} per message, in
        order (plus doubtful when the service has a doubt margin). label, index, score and margin are those of
        AnyIntent.classify; runner_up is the second-best scored intent (from return_scores=True, which adds the
        candidates and changes nothing else)."""
        msgs = self._msgs(texts)
        labels = self._labels(intents, list_name)
        k = self._k(k)
        clf = self.classifier()
        t0 = time.perf_counter()
        kwargs = {"return_scores": True} if k is None else {"k": k, "return_scores": True}
        with self._model_lock(clf):
            self._call(self._apply_labels, clf, labels)
            preds = self._call(clf.classify, msgs, **kwargs)
        out = [self._result(p) for p in preds]
        self._log("classify", msgs, len(labels), t0, [o["label"] for o in out])
        return out

    def normalize_check_intent(self, x):
        """The one intent of check: a name, an object with 'name' and 'description', or a [name, description] pair
        (AnyIntent.check takes a (name, description) tuple); returns it as normalize_intent does."""
        if isinstance(x, (list, tuple)):
            if len(x) != 2 or not isinstance(x[0], str) or not (x[1] is None or isinstance(x[1], str)):
                raise InvalidRequest("'intent': check takes one intent: a name, an object with 'name' and "
                                     "'description', or a [name, description] pair of strings (to check several "
                                     "intents, call it once per intent)")
            x = {"name": x[0], "description": x[1]}
        return self.normalize_intent(x, "intent")

    def check(self, texts, intent, threshold=None):
        """AnyIntent.check (when the library has it): one {intent, yes, probability, score, threshold} per message, in
        order, the values of the library's CheckResult. intent: a name, {"name", "description"} or [name, description].
        CalibrationMissing (409) when the model's calibration does not name check."""
        self.require("check")
        msgs = self._msgs(texts)
        it = self.normalize_check_intent(intent)
        kwargs = {} if threshold is None else {"threshold": self._threshold(threshold)}
        clf = self.classifier()
        t0 = time.perf_counter()
        with self._model_lock(clf):
            res = self._calibrated_call(clf, "check", clf.check, msgs,
                                        it["name"] if "description" not in it else dict(it), **kwargs)
        out = [check_result(r) for r in res]
        self._log("check", msgs, 1, t0, [o["yes"] for o in out])
        return out

    def scores(self, texts, intents=None, list_name=None):
        """AnyIntent.scores (when the library has it): one {label, intents} per message, in order (scores_result): the
        intents the model read with the message, each with its value from the library's ScoresResult for this label
        set, highest first. CalibrationMissing (409) when the model's calibration does not name scores."""
        self.require("scores")
        msgs = self._msgs(texts)
        labels = self._labels(intents, list_name)
        clf = self.classifier()
        t0 = time.perf_counter()
        with self._model_lock(clf):
            self._call(self._apply_labels, clf, labels)
            res = self._calibrated_call(clf, "scores", clf.scores, msgs)
        out = [scores_result(r) for r in res]
        self._log("scores", msgs, len(labels), t0, [o["label"] for o in out])
        return out

    def save_list(self, name, intents):
        check_list_name(name)
        norm = self.normalize_intents(intents)
        created = self.lists.put(name, norm)
        return {"name": name, "n_intents": len(norm), "created": created, "persistent": self.lists.persistent}

    def get_list(self, name):
        return {"name": name, "intents": self.lists.get(check_list_name(name))}

    def delete_list(self, name):
        self.lists.delete(check_list_name(name))
        return {"name": name, "deleted": True}

    def list_lists(self):
        return {"lists": self.lists.summary(), "persistent": self.lists.persistent}

    def health(self, detail=True):
        """Takes no lock that a classification holds (it answers while the model is busy)."""
        status = "ok" if self.ready else ("error" if self._load_error else "loading")
        out = {"status": status, "ready": self.ready}
        if not detail:
            return out
        out.update(model=self.model_id, library_version=__version__,
                   features={"classify": True, **{f: self.has(f) for f in OPTIONAL_FEATURES}},
                   supported=self.supported(),
                   limits=asdict(self.limits),
                   lists={"persistent": self.lists.persistent, "count": self.lists.count},
                   label_text_mode=self.label_text_mode, doubt_margin=self.doubt_margin,
                   log_texts=self.log_texts)
        if self._load_error:
            out["error"] = self._load_error
        if self.ready:
            out["calibration"] = None if self._calib is None else {"file": self._calib["file"],
                                                                   "functions": list(self._calib["functions"])}
            out["revision"] = jsonable(getattr(self._clf, "revision", None))
            cfg = getattr(self._clf, "config", None)
            if isinstance(cfg, dict):
                conf = {k: jsonable(cfg[k]) for k in CONFIG_KEYS if k in cfg}
                active = cfg.get("active")
                if isinstance(active, dict):     # the settings in use for the current label set
                    conf["active"] = {k: jsonable(active[k]) for k in ("w", "k", "tau", "mode") if k in active}
                out["config"] = conf
            else:
                out["config"] = None
        return out


# ---- the Host header check of the HTTP servers (plain ASGI, no web package) ----------------------------------------
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")


def is_loopback(host):
    """True for localhost and for a loopback address (127.0.0.0/8, ::1, with or without brackets)."""
    h = str(host or "").strip().lower()
    if h == "localhost":
        return True
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def host_name(value):
    """The name of a Host header value: lowercase, without the port; an IPv6 address keeps its brackets."""
    v = str(value or "").strip().lower()
    if v.startswith("["):
        end = v.find("]")
        return v[:end + 1] if end > 0 else ""
    return v.split(":", 1)[0]


def allowed_name(name):
    """A name for an allow list: a bare IPv6 address gets brackets (as in a Host header), a port is dropped."""
    n = str(name).strip().lower()
    try:
        ipaddress.IPv6Address(n)
        return f"[{n}]"
    except ValueError:
        return n if n.startswith("*") else host_name(n)


def host_allowed(value, allowed):
    """allowed: None (any Host) or names from allowed_name ('*' matches any, '*.example.org' its subdomains)."""
    if allowed is None:
        return True
    name = host_name(value)
    if not name:
        return False
    return any(a == "*" or a == name or (a.startswith("*.") and name.endswith(a[1:])) for a in allowed)


async def send_json_error(send, status, code, message, headers=()):
    """A complete JSON error response {"error": {"code", "message"}} on a plain ASGI send."""
    body = json.dumps({"error": {"code": code, "message": message}}).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
                + [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers]})
    await send({"type": "http.response.body", "body": body})


def scope_header(scope, name):
    for key, value in scope.get("headers") or []:
        if key == name:
            return value.decode("latin-1")
    return None


class HostGuard:
    """Plain ASGI middleware: a request whose Host header names none of the allowed names gets a JSON 400. A guard
    against DNS rebinding: a web page that points its own domain name at 127.0.0.1 still sends that name as Host.
    allowed None: any Host. hint: what the error message tells the user to do to accept a name ({name} is filled)."""

    def __init__(self, app, allowed, hint="start the server with --allowed-host {name} to accept it"):
        self.app = app
        self.allowed = None if allowed is None else sorted({allowed_name(a) for a in allowed})
        self.hint = hint

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and not host_allowed(scope_header(scope, b"host"), self.allowed):
            name = host_name(scope_header(scope, b"host"))[:100]
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            await send_json_error(send, 400, "host_not_allowed",
                                  f"the Host header {name!r} is not one this server accepts (a guard against DNS "
                                  f"rebinding); {self.hint.format(name=name or '<name>')}")
            return
        await self.app(scope, receive, send)
