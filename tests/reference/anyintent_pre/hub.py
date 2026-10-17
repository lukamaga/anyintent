"""Local folders and Hugging Face downloads.

The library carries no token and no license flag: a gated model is downloaded with the user's own Hugging Face login
(`hf auth login`, or `token=`) after the user has accepted the model's terms on its Hugging Face page.
"""
import hashlib
import json
import os

# Root files of a released model; nothing else (no merged-embedding copy, no README) is downloaded
# (the meta file is r2intent_meta.json, the name of the R2Intent releases)
MODEL_FILES = ["config.json", "r2intent_meta.json", "anyintent.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", "LICENSE", "NOTICE"]
# bge-m3 (the two-file retriever): the ignore patterns of the project's prefetch, so the same cache serves it offline
BGE_M3_REPO = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_IGNORE = ["onnx/*", "imgs/*", "colbert_linear.pt", "sparse_linear.pt", "*.jpg", "*.webp", "*.DS_Store"]
# Files of a local folder that determine its scores (hashed into the local revision id)
FINGERPRINT_FILES = ["config.json", "anyintent.json", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "sentencepiece.bpe.model", "r2intent_meta.json"]
WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]
# files of a sub-folder named by anyintent.json (cross.path, embedder.path) in a repository download
SUB_FILES = MODEL_FILES + ["pytorch_model.bin"]


class AnyIntentError(RuntimeError):
    pass


class AccessError(AnyIntentError):
    pass


def _snapshot_download(*args, **kwargs):
    # indirection so that tests can replace the download without network
    from huggingface_hub import snapshot_download
    return snapshot_download(*args, **kwargs)


def _error_kind(e):
    names = {c.__name__ for c in type(e).__mro__}
    status = getattr(getattr(e, "response", None), "status_code", None)
    msg = str(e).lower()
    # GatedRepoError is a subclass of RepositoryNotFoundError in huggingface_hub: test it first
    if "GatedRepoError" in names or (status == 403 and "gated" in msg):
        return "gated"
    if "IncompleteSnapshotError" in names:
        return "incomplete"
    if "RepositoryNotFoundError" in names or "RevisionNotFoundError" in names or status == 404:
        return "missing"
    if status == 401:
        return "unauthorized"
    if status == 403:
        return "forbidden"
    if "LocalEntryNotFoundError" in names or "OfflineModeIsEnabled" in names:
        return "offline"
    return None


def download(repo_id, revision=None, token=None, local_files_only=False, allow_patterns=None, ignore_patterns=None):
    """snapshot_download with readable errors for gated, missing, private, incomplete and offline repositories."""
    try:
        return _snapshot_download(repo_id, revision=revision, token=token, local_files_only=local_files_only,
                                  allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
    except Exception as e:
        kind = _error_kind(e)
        detail = f"({type(e).__name__}: {str(e)[:300]})"
        if kind == "gated":
            raise AccessError(
                f"No access to the model repository '{repo_id}'. It is gated: open https://huggingface.co/{repo_id}, "
                "accept its terms with your Hugging Face account, then log in on this machine with `hf auth login` "
                "(or pass token=...). The library license gives no rights to the model weights; their license is in "
                f"the repository. {detail}") from e
        if kind in ("missing", "unauthorized"):
            raise AccessError(
                f"Model repository '{repo_id}' (revision {revision}) was not found, is private, or you are not logged in. "
                "Check the id and the revision, and run `hf auth login` (or pass token=...) if the repository is "
                f"private or gated. {detail}") from e
        if kind == "forbidden":
            raise AccessError(
                f"Access to '{repo_id}' was refused (HTTP 403). If the repository is gated, accept its terms on "
                f"https://huggingface.co/{repo_id}; check that your token has read access. {detail}") from e
        if kind == "incomplete":
            raise AnyIntentError(
                f"The local Hugging Face cache holds only part of '{repo_id}' (revision {revision}) and downloads are "
                "disabled (offline). Download it once with network access, using the same file patterns as this "
                f"library (anyintent.hub.MODEL_FILES, or BGE_M3_IGNORE for bge-m3). Missing files: {detail}") from e
        if kind == "offline":
            raise AnyIntentError(f"'{repo_id}' (revision {revision}) is not in the local Hugging Face cache and downloads "
                                 f"are disabled (offline). {detail}") from e
        raise


def _commit_of(path):
    # .../snapshots/<commit>/ in the Hugging Face cache
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 2 and parts[-2] == "snapshots" and len(parts[-1]) == 40:
        return parts[-1]
    return None


_SHA_MEMO = {}


def file_sha256(path, chunk=8 << 20):
    """sha256 of a file, memoised in this process by (absolute path, size, mtime in ns)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_MEMO:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        _SHA_MEMO[key] = h.hexdigest()
    return _SHA_MEMO[key]


def local_fingerprint(folder):
    """A revision id for a local folder: sha256 over the meta, config, anyintent and tokenizer files and the sha256 of
    the weight file (all bytes, so a re-saved or fine-tuned model in the same folder gets a new id; the hash of a 1.2 GB
    file takes a few seconds once per process)."""
    h = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(name.encode() + b"\0" + f.read() + b"\0")
    for name in WEIGHT_FILES:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            h.update(name.encode() + b"\0" + file_sha256(p).encode() + b"\0")
    return "local-" + h.hexdigest()


def subfolders(folder):
    """Sub-folders a release's anyintent.json names (cross.path, embedder.path), relative and below the root."""
    ai = read_json(folder, "anyintent.json") or {}
    out = []
    for key in ("cross", "embedder"):
        sub = (ai.get(key) or {}).get("path")
        if not sub or os.path.normpath(str(sub)) == ".":
            continue
        sub = os.path.normpath(str(sub))
        if os.path.isabs(sub) or sub.startswith(".."):
            raise AnyIntentError(f"anyintent.json {key}.path {sub!r}: a repository may only name a sub-folder of itself")
        out.append(sub.replace(os.sep, "/"))
    return out


def resolve(repo_or_path, revision=None, token=None, local_files_only=False, files=None, ignore_patterns=None):
    """Returns (local folder, revision id). An existing local folder is used as it is. A repository download takes the
    root files first; when its anyintent.json names sub-folders (a two-file release with cross/ and embedder/), their
    files are downloaded in a second call at the same commit."""
    if os.path.isdir(str(repo_or_path)):
        folder = os.path.abspath(str(repo_or_path))
        return folder, _commit_of(folder) or local_fingerprint(folder)
    if ignore_patterns is not None:
        folder = download(str(repo_or_path), revision, token, local_files_only, ignore_patterns=ignore_patterns)
    else:
        root_files = list(files or MODEL_FILES)
        folder = download(str(repo_or_path), revision, token, local_files_only, allow_patterns=root_files)
        subs = subfolders(folder) if files is None else []
        if subs:
            pats = root_files + [f"{s}/{f}" for s in subs for f in SUB_FILES]
            folder = download(str(repo_or_path), _commit_of(folder) or revision, token, local_files_only,
                              allow_patterns=pats)
    return folder, _commit_of(folder) or local_fingerprint(folder)


def read_json(folder, name):
    p = os.path.join(folder, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)
