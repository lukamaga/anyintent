"""The R2Intent names (PROTOCOL E5 (10f), decision D6): a one-file package with r2intent_meta.json, the
adapter under r2intent_embed_lora.* and the safetensors metadata key r2intent loads; inconsistent names are refused,
and a meta file under any other name is not read (the library reads the R2Intent names only; the support of the names
before the rename was removed on 2026-10-17). CPU fp32, tiny random models, no network."""
import json
import os
import shutil

import pytest

import anyintent.hub as hub
from anyintent import AnyIntent, AnyIntentError
from anyintent.config import read_meta


def _clf(path, engine="eager"):
    return AnyIntent.from_pretrained(path, device="cpu", engine=engine, local_files_only=True)


def test_one_file_model_under_r2intent_names(one_file, final_file):
    a, b = _clf(one_file), _clf(final_file)
    for c in (a, b):
        assert c.cfg.meta_file == "r2intent_meta.json" and c.cfg.adapter["prefix"] == "r2intent_embed_lora."
    assert a.cfg.version == "v1.1" and b.cfg.version == "1.2"
    assert (b.cfg.tau, b.cfg.label_text_mode) == (0.5, "noex")


def test_a_meta_file_under_another_name_is_not_read(tmp_path, one_file):
    """a folder whose only meta file has another name than r2intent_meta.json (for example the name of an older
    package) is a folder without a meta file: a one-file model folder is then refused by the layout check, whose
    message names r2intent_meta.json"""
    for i, name in enumerate(("model_meta.json", "meta.json")):
        d = tmp_path / f"meta_{i}"
        shutil.copytree(one_file, d)
        os.replace(d / "r2intent_meta.json", d / name)
        assert read_meta(str(d)) == (None, None)
        with pytest.raises(AnyIntentError, match=r"not a one-file model \(no meta file with an adapter block: "
                                                 r"r2intent_meta\.json\)"):
            _clf(str(d))


def test_r2intent_meta_with_another_adapter_prefix_is_refused(tmp_path, one_file):
    d = tmp_path / "otherprefix"
    shutil.copytree(one_file, d)
    m = json.load(open(d / "r2intent_meta.json", encoding="utf-8"))
    m["adapter"]["prefix"] = "other_embed_lora."
    json.dump(m, open(d / "r2intent_meta.json", "w", encoding="utf-8"))
    with pytest.raises(AnyIntentError, match="names the adapter prefix 'other_embed_lora.'"):
        _clf(str(d))


@pytest.mark.parametrize("change,needle", [({"drop": "r2intent"}, "carries the safetensors metadata key 'r2intent'"),
                                           ({"lora_prefix": "other_embed_lora."}, "lora_prefix")])
def test_safetensors_metadata_must_match_the_names(tmp_path, one_file, change, needle):
    from safetensors import safe_open
    from safetensors.torch import save_file
    d = tmp_path / "md"
    shutil.copytree(one_file, d)
    wp = str(d / "model.safetensors")
    with safe_open(wp, framework="pt") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
        md = dict(f.metadata() or {})
    if "drop" in change:
        md.pop(change["drop"])
    if "lora_prefix" in change:
        md["lora_prefix"] = change["lora_prefix"]
    save_file(sd, wp, metadata=md)
    with pytest.raises(AnyIntentError, match=needle):
        _clf(str(d))


def test_hub_lists_the_meta_file_and_the_fingerprint_sees_it(tmp_path, one_file):
    assert "r2intent_meta.json" in hub.MODEL_FILES and "r2intent_meta.json" in hub.FINGERPRINT_FILES
    assert not any(n.endswith("_meta.json") and n != "r2intent_meta.json"
                   for n in hub.MODEL_FILES + hub.FINGERPRINT_FILES)
    d = tmp_path / "meta_changed"
    shutil.copytree(one_file, d)
    m = json.load(open(d / "r2intent_meta.json", encoding="utf-8"))
    m["run_id"] = "another run"
    json.dump(m, open(d / "r2intent_meta.json", "w", encoding="utf-8"))
    fa, fb = hub.local_fingerprint(one_file), hub.local_fingerprint(str(d))
    assert fa != fb and fa.startswith("local-") and fb.startswith("local-")
