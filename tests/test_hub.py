"""The Hugging Face download is mocked: no network."""
import json
import os
import shutil

import pytest

import anyintent.hub as hub
from anyintent import AccessError, AnyIntent, AnyIntentError
from conftest import LABELS

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def fake_cache(tmp_path, src, repo="user/r2intent-test", commit=COMMIT):
    d = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / commit
    shutil.copytree(src, d)
    return str(d)


def test_download_uses_explicit_files_and_revision(one_file, tmp_path, monkeypatch):
    snap = fake_cache(tmp_path, one_file)
    calls = []

    def fake(repo_id, **kw):
        calls.append((repo_id, kw))
        return snap

    monkeypatch.setattr(hub, "_snapshot_download", fake)
    c = AnyIntent.from_pretrained("user/r2intent-test", revision="v1.0", device="cpu")
    assert c.revision == COMMIT
    (repo, kw), = calls
    assert repo == "user/r2intent-test" and kw["revision"] == "v1.0"
    assert kw["allow_patterns"] == hub.MODEL_FILES and kw["ignore_patterns"] is None
    assert not any("*" in p for p in kw["allow_patterns"])
    c.set_labels(LABELS)
    assert c.classify("where is my parcel").label


def test_gated_repo_gives_access_error(monkeypatch):
    class GatedRepoError(Exception):
        pass

    def fake(repo_id, **kw):
        raise GatedRepoError("403 gated")

    monkeypatch.setattr(hub, "_snapshot_download", fake)
    with pytest.raises(AccessError) as e:
        AnyIntent.from_pretrained("user/r2intent-test", device="cpu")
    assert "hf auth login" in str(e.value) and "user/r2intent-test" in str(e.value)


def test_http_403_gives_access_error(monkeypatch):
    class Resp:
        status_code = 403

    class HfHubHTTPError(Exception):
        response = Resp()

    monkeypatch.setattr(hub, "_snapshot_download", lambda repo_id, **kw: (_ for _ in ()).throw(HfHubHTTPError("x")))
    with pytest.raises(AccessError):
        AnyIntent.from_pretrained("user/r2intent-test", device="cpu")


@pytest.mark.parametrize("name", ["RepositoryNotFoundError", "RevisionNotFoundError"])
def test_missing_repo_or_revision(monkeypatch, name):
    err = type(name, (Exception,), {})
    monkeypatch.setattr(hub, "_snapshot_download", lambda repo_id, **kw: (_ for _ in ()).throw(err("x")))
    with pytest.raises(AccessError) as e:
        AnyIntent.from_pretrained("user/nothing", revision="v9", device="cpu")
    assert "user/nothing" in str(e.value) and "not found, is private" in str(e.value) and "It is gated" not in str(e.value)


def _http_error(status, msg):
    class Resp:
        status_code = status

    class HfHubHTTPError(Exception):
        response = Resp()

    return HfHubHTTPError(msg)


@pytest.mark.parametrize("status,msg,needle", [
    (401, "401 Client Error: Unauthorized", "not found, is private, or you are not logged in"),
    (403, "403 Forbidden: access to model user/r2intent-test is restricted, it is a gated repo", "It is gated"),
    (403, "403 Forbidden", "HTTP 403")])
def test_http_status_messages(monkeypatch, status, msg, needle):
    monkeypatch.setattr(hub, "_snapshot_download", lambda repo_id, **kw: (_ for _ in ()).throw(_http_error(status, msg)))
    with pytest.raises(AccessError) as e:
        AnyIntent.from_pretrained("user/r2intent-test", device="cpu")
    assert needle in str(e.value), str(e.value)
    if status == 401:
        assert "It is gated" not in str(e.value)


@pytest.mark.parametrize("name,needle", [("LocalEntryNotFoundError", "not in the local Hugging Face cache"),
                                         ("IncompleteSnapshotError", "holds only part of")])
def test_offline_and_incomplete(monkeypatch, name, needle):
    err = type(name, (Exception,), {})
    monkeypatch.setattr(hub, "_snapshot_download",
                        lambda repo_id, **kw: (_ for _ in ()).throw(err("missing: model.safetensors")))
    with pytest.raises(AnyIntentError) as e:
        AnyIntent.from_pretrained("user/nothing", revision="v9", device="cpu", local_files_only=True)
    assert not isinstance(e.value, AccessError) and needle in str(e.value) and "user/nothing" in str(e.value)


def test_two_file_embedder_from_hub(two_file, tmp_path, monkeypatch):
    root = tmp_path / "release"
    shutil.copytree(os.path.join(two_file, "cross"), root)
    json.dump({"format": "anyintent-two-file", "embedder": {"repo": hub.BGE_M3_REPO, "revision": hub.BGE_M3_REVISION},
               "cross": {"path": "."}, "fusion": {"w": 5.0, "k": 8}}, open(root / "anyintent.json", "w", encoding="utf-8"))
    emb_snap = fake_cache(tmp_path, os.path.join(two_file, "embedder"), repo=hub.BGE_M3_REPO,
                          commit=hub.BGE_M3_REVISION)
    calls = []

    def fake(repo_id, **kw):
        calls.append((repo_id, kw))
        return emb_snap

    monkeypatch.setattr(hub, "_snapshot_download", fake)
    c = AnyIntent.from_pretrained(str(root), device="cpu")
    (repo, kw), = calls
    assert repo == hub.BGE_M3_REPO and kw["revision"] == hub.BGE_M3_REVISION
    assert kw["ignore_patterns"] == hub.BGE_M3_IGNORE and kw["allow_patterns"] is None
    assert c.embedder_revision == hub.BGE_M3_REVISION and c.cfg.msg_max_len == 512
    c.set_labels(LABELS)
    assert c.classify("refund please").label


def test_local_fingerprint_stable(one_file):
    assert hub.local_fingerprint(one_file) == hub.local_fingerprint(one_file)
    assert hub.resolve(one_file)[1] == hub.local_fingerprint(one_file)


def test_weight_bytes_change_the_revision_and_the_cache_key(one_file, tmp_path):
    d = tmp_path / "same_shapes"
    shutil.copytree(one_file, d)
    st = d / "model.safetensors"
    raw = bytearray(st.read_bytes())
    raw[-1] ^= 0x01                                    # one byte of the last tensor's data (the header is unchanged)
    st.write_bytes(bytes(raw))
    assert hub.local_fingerprint(one_file) != hub.local_fingerprint(str(d))
    cache = str(tmp_path / "cache")
    a = AnyIntent.from_pretrained(one_file, device="cpu", engine="eager", cache_dir=cache)
    a.set_labels(LABELS)
    b = AnyIntent.from_pretrained(str(d), device="cpu", engine="eager", cache_dir=cache)
    b.set_labels(LABELS)
    assert b.cache.hits == {"memory": 0, "disk": 0, "miss": 1} and a.label_key != b.label_key


def test_two_file_release_with_subfolders_downloads_them(two_file, tmp_path, monkeypatch):
    snap = fake_cache(tmp_path, two_file)
    calls = []

    def fake(repo_id, **kw):
        calls.append((repo_id, kw))
        return snap

    monkeypatch.setattr(hub, "_snapshot_download", fake)
    c = AnyIntent.from_pretrained("user/r2intent-test", revision="v1.0", device="cpu", engine="eager")
    assert len(calls) == 2 and c.revision == COMMIT
    assert calls[0][1]["allow_patterns"] == hub.MODEL_FILES and calls[0][1]["revision"] == "v1.0"
    pats = calls[1][1]["allow_patterns"]
    assert calls[1][1]["revision"] == COMMIT and "cross/model.safetensors" in pats and "embedder/config.json" in pats
    assert c.cfg.layout == "two_file"
    c.set_labels(LABELS)
    assert c.classify("refund please").label


def test_subfolder_outside_the_repository_is_refused(two_file, tmp_path, monkeypatch):
    snap = fake_cache(tmp_path, two_file)
    json.dump({"format": "anyintent-two-file", "cross": {"path": "../elsewhere"}, "fusion": {"w": 1.0, "k": 8}},
              open(os.path.join(snap, "anyintent.json"), "w", encoding="utf-8"))
    monkeypatch.setattr(hub, "_snapshot_download", lambda repo_id, **kw: snap)
    with pytest.raises(AnyIntentError, match="sub-folder"):
        AnyIntent.from_pretrained("user/r2intent-test", device="cpu")
