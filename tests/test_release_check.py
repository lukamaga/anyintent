"""The release check (tools/release_check.py, not part of the wheel) on generated stand-ins: a tiny random released
model, generated DEV data and stored DEV scores of the reference DEV scorer. Every branch: pass (one-file with the
cascade, two-file, config keys at the top level of the decision, alternative meta key names, a pins file), a tampered
score, a test-named input, a wrong sha256, a k different from the PROTOCOL value, a meta without tau; the packaged
file against the decision-pinned cross-encoder and adapter files (three different sha256, as stage-1.2 FINAL writes
them), a decision that does not pin the packaged file, the honest-model path of plan section 6, and the shapes the
stage-1.2 FINAL pub writer produces (configs P and Z, the package record chain, final check N1)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import release_check as RC   # noqa: E402

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
CPU_TORCH = [("eager", "float32", True), ("fast", "float32", True)]
# the stand-in package record of the style s12_final names the released folder by a Linux path (/run/machine/out/...);
# from Python 3.13 on, Windows does not treat a path that starts with a single slash as absolute, so the check finds no
# folder there (release_check is a tool for Linux and Kaggle)
LINUX_RECORD_PATH = pytest.mark.skipif(
    os.name == "nt" and not os.path.isabs("/run/machine/out"),
    reason="a test record names the released folder by a Linux path, which Python 3.13 and later on Windows does not "
           "treat as absolute; the release check tool expects Linux paths")


@pytest.fixture(autouse=True)
def fixed_stand_in(monkeypatch):
    """One stand-in for every run. The Unigram trainer of tokenizers returns the same pieces in another order on each
    call, so the pieces got other ids and each run built another random model. The retrieval scores of these models lie
    close together, and on a rare one a near tie, most likely two labels at the top-k cut, broke one way in the
    reference and the other way in an engine: the top-1 of one row changed and 23 of 24 rows agreed. With the pieces
    sorted, every run gives each piece the same id and builds the same weights. (The release notebook's own call of
    release_check.smoke_inputs, outside pytest, is not covered by this fixture.)"""
    import conftest as C
    build = C.build_tokenizer

    def build_sorted(d):
        build(d)
        p = os.path.join(d, "tokenizer.json")
        with open(p, encoding="utf-8") as f:
            tj = json.load(f)
        # ids 0 to 4 stay the special tokens
        tj["model"]["vocab"] = tj["model"]["vocab"][:5] + sorted(tj["model"]["vocab"][5:])
        with open(p, "w", encoding="utf-8") as f:
            json.dump(tj, f, ensure_ascii=False)
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(d)
    monkeypatch.setattr(C, "build_tokenizer", build_sorted)


def run(tmp_path, name, layout="one_file", fault=None, style="configs", runs=CPU_TORCH):
    S = RC.smoke_inputs(str(tmp_path / name / "in"), layout=layout, fault=fault, style=style)
    S.update(out_dir=str(tmp_path / name / "out"), run_name=name, runs=runs, n_timing=3, force_cpu=True)
    R = RC.run_check(S)
    on_disk = json.load(open(tmp_path / name / "out" / f"library_release_{name}.json", encoding="utf-8"))
    assert on_disk["status"] == R["status"]
    print(f"RELEASE {name}: {R['status']} problems {R['problems'][:3]} refused {R.get('refused')}")
    return R


def test_pass_one_file_with_cascade(tmp_path):
    R = run(tmp_path, "one_file")
    assert R["status"] == "PASS", R
    assert R["config"]["resolved"] == {"w": 15.0, "k": 8, "tau": 0.5, "cascade_first": 3, "label_text_mode": "noex"}
    assert R["label_column"] == "label_text_noex" and R["composition"]["clinc_dev"] == {"checked": True, "v2": 0,
                                                                                         "noex": 0}
    for r in R["runs"]:
        for p, x in r["pools"].items():
            assert x["candidate_mismatch_rows"] == 0 and x["fused_top1_agreement"] == 1.0, (p, x)
            assert x["cascade_top1_agreement"] == 1.0 and x["max_abs_cross_diff"] < 1e-4
        assert set(r["timing"]) == {"k3", "k5", "k8", "cascade"}
    assert R["rows"]["clinc_dev"]["n"] == 42          # 44 rows, 2 near_test rows are not decision rows
    # I4 (L01): every clinc_dev row of the stored frame (44, the 2 near_test rows included) on every gating engine
    assert R["rows"]["clinc_dev full"] == {"n": 44, "decision_rows_total": 42, "near_test_rows": 2}
    for r in R["runs"]:
        full = r["pools"]["clinc_dev full"]
        assert full["ok"] and full["agree_min"] == RC.AGREE_MIN_FULL == 0.99 and full["rows"] == 44
    assert R["full_row_runs"] == ["eager/float32", "fast/float32"]
    st = R["source_tensors"]
    assert st["checked"] and st["equal"] and st["n_backbone"] > 0 and st["n_adapter"] > 0, st
    assert R["decision_file"] == "final_decision_smoke.json"


def test_pass_two_file_and_top_level_decision(tmp_path):
    R = run(tmp_path, "two_file", layout="two_file", style="top")
    assert R["status"] == "PASS", R
    assert R["layout"] == "two_file" and R["config"]["resolved"]["tau"] is None
    assert R["source_tensors"]["checked"] is False


def test_pass_honest_model_from_the_stage12_decision(tmp_path):
    R = run(tmp_path, "honest", style="honest")
    assert R["status"] == "PASS", R
    assert R["decision_file"] == "stage12_decision_smoke.json" and R["config_name"] == "HN12"
    assert R["shipped"] == "honest" and R["release_dir"].startswith(RC.staged_name("release-honest"))
    assert R["source_tensors"]["equal"] is True


def test_refuses_a_release_staged_folder_outside_the_list(tmp_path):
    S = RC.smoke_inputs(str(tmp_path / "in"))
    S.update(out_dir=str(tmp_path / "out"), run_name="badstaged", runs=CPU_TORCH[:1], force_cpu=True,
             release_staged=RC.staged_name("look"))
    R = RC.run_check(S)
    assert R["status"] == "REFUSED" and "RELEASE_STAGED" in R["refused"]


def test_the_project_names_are_settings(tmp_path):
    """the staged folder prefix and the training prefixes are settings of the run (the tool holds no project name): a
    run without staged_prefix is refused, and with an adapter file whose prefix is not in training_prefixes the source
    tensor check fails closed (the adapter keys are not found under the released prefix)"""
    S = RC.smoke_inputs(str(tmp_path / "in"), style="honest")
    assert S["staged_prefix"] == RC.SMOKE_STAGED_PREFIX and S["training_prefixes"] == [RC.SMOKE_TRAINING_PREFIX]
    S.update(out_dir=str(tmp_path / "out"), run_name="noprefix", runs=CPU_TORCH[:1], n_timing=0, force_cpu=True)
    del S["staged_prefix"]
    R = RC.run_check(S)
    assert R["status"] == "REFUSED" and "staged_prefix" in R["refused"], R.get("refused")
    S.update(staged_prefix=RC.SMOKE_STAGED_PREFIX, training_prefixes=[], run_name="notraining")
    R = RC.run_check(S)
    assert R["status"] == "REFUSED" and "does not equal the decision-pinned" in R["refused"], R.get("refused")
    assert R["source_tensors"]["checked"] and R["source_tensors"]["n_adapter"] == 0, R["source_tensors"]


def test_pass_alternative_meta_keys(tmp_path):
    assert run(tmp_path, "alt_keys", fault="alt_keys")["status"] == "PASS"


def test_pins_file_sets_the_protocol_k(tmp_path):
    R = run(tmp_path, "pins", fault="pins")
    assert R["status"] == "PASS" and R["config"]["resolved"]["k"] == 5 and R["config"]["anyintent_json"]


@pytest.mark.parametrize("fault,status,needle", [
    ("tamper", "FAIL", "failed"),
    ("config", "FAIL", "k: library 8 vs PROTOCOL 5"),
    ("no_tau", "FAIL", "tau: library None vs PROTOCOL 0.5"),
    ("test_path", "REFUSED", "test"),
    ("sha", "REFUSED", "released model.safetensors"),
    ("unpinned_package", "REFUSED", "is not a file the decision pins"),
    ("anyjson_sha", "REFUSED", "EXPECT_ANYINTENT_JSON_SHA256"),
    ("fewshot_mismatch", "FAIL", "vs few-shot decision"),
    ("cross_mismatch", "REFUSED", "does not equal the decision-pinned"),
    ("adapter_mismatch", "REFUSED", "does not equal the decision-pinned")])
def test_faults(tmp_path, fault, status, needle):
    R = run(tmp_path, fault, fault=fault, runs=CPU_TORCH[:1])
    assert R["status"] == status, R
    text = json.dumps(R["problems"]) + str(R.get("refused"))
    assert needle in text, text


def test_refuses_unpinned_and_missing_expectations(tmp_path):
    S = RC.smoke_inputs(str(tmp_path / "in"))
    S.update(out_dir=str(tmp_path / "out"), run_name="unpinned", runs=CPU_TORCH[:1], force_cpu=True)
    S["expect_decision_sha256"] = None
    assert RC.run_check(dict(S))["status"] == "REFUSED"
    S2 = RC.smoke_inputs(str(tmp_path / "in2"))
    S2.update(out_dir=str(tmp_path / "out2"), run_name="noexpect", runs=CPU_TORCH[:1], force_cpu=True,
              expect_config=None)
    R = RC.run_check(S2)
    assert R["status"] == "REFUSED" and "EXPECT_CONFIG" in R["refused"]


@LINUX_RECORD_PATH
def test_pass_stage12_final_writer_shapes(tmp_path):
    R = run(tmp_path, "s12_final", style="s12_final")
    assert R["status"] == "PASS", R
    assert R["config_name"] == "P" and R["pin"] == "package record chain", R
    assert R["config"]["per_k_source"].startswith("not in the FINAL decision") and R["config"]["per_k_keys"] == [3, 5, 8]
    assert R["package_chain"]["package_record"] == "final_package_smoke.json"
    st = R["source_tensors"]
    assert st["checked"] and st["equal"] and st["n_backbone"] > 0 and st["n_adapter"] > 0, st


@LINUX_RECORD_PATH
@pytest.mark.parametrize("fault,needle", [
    ("s12_bad_record", "package record final_record_sha256"),
    ("s12_meta_record", "r2intent_meta.json final_record_sha256"),
    ("s12_no_sources", "needs the source tensor check"),
    ("cross_mismatch", "does not equal the decision-pinned")])
def test_stage12_final_faults(tmp_path, fault, needle):
    R = run(tmp_path, "s12_" + fault, fault=fault, style="s12_final", runs=CPU_TORCH[:1])
    assert R["status"] == "REFUSED", R
    assert needle in str(R.get("refused")), R.get("refused")


def test_decision_reader_and_dict_pins(tmp_path):
    p = tmp_path / "final_decision_x.json"
    sha_a, sha_b = "a" * 64, "b" * 64
    p.write_text(json.dumps({"adopted": "HN12", "configs": {"Q": {}, "Y": {}}}), encoding="utf-8")
    with pytest.raises(RC.Refused, match="CONFIG_NAME"):
        RC.read_final_decision(str(p))
    assert RC.read_final_decision(str(p), "Y")[0] == "Y"
    with pytest.raises(RC.Refused, match="is not one of the configs"):
        RC.read_final_decision(str(p), "P")
    p.write_text(json.dumps({"adopted": "HN12", "final_data": "dom", "configs": {"D": {}, "Z": {}}}), encoding="utf-8")
    assert RC.read_final_decision(str(p))[0] == "D"
    p.write_text(json.dumps({"adopted": "HN12", "final_data": "pub", "configs": {"P": {}, "HN12": {}}}), encoding="utf-8")
    assert RC.read_final_decision(str(p))[0] == "HN12"      # adopted wins when it is one of the configs
    assert RC._pinned({"cross_sha256": {"model.safetensors": sha_a, "other.safetensors": sha_b}}, "cross") == sha_a
    assert RC._pinned({"cross_sha256": {"x.safetensors": sha_b}}, "cross") == sha_b
    assert RC._pinned({"cross_sha256": {"x.safetensors": sha_a, "y.safetensors": sha_b}}, "cross") is None
    assert RC._pinned({"sha256": {"adapter": sha_b}}, "adapter") == sha_b


def test_released_files_and_anyintent_json_pin(tmp_path):
    R = run(tmp_path, "pins_files", fault="pins", runs=CPU_TORCH[:1])
    assert R["status"] == "PASS", R
    assert "model.safetensors" in R["released_files"] and "anyintent.json" in R["released_files"]
    assert R["anyintent_json_sha256"] == R["released_files"]["anyintent.json"]


def test_pass_v3s_release_with_the_registered_v2_settings(tmp_path):
    R = run(tmp_path, "v3s", style="v3s")
    assert R["status"] == "PASS", R
    assert R["config"]["resolved"]["label_text_mode"] == "v3s" and R["label_column"] == "label_text_v3s_A"
    assert R["composition"]["clinc_dev"]["v3s"] == 0
    assert R["config"]["per_k_source"].startswith("the stage-1.2 decision")
    assert R["config"]["fusion_by_mode"]["v2"]["stage12"]["w"] == 5.0
    assert R["mode_rows"]["v2"]["variant"] == "adapter|v2"
    for r in R["runs"]:
        assert r["pools"]["clinc_dev [v2]"]["ok"] and r["pools"]["clinc_dev [v2]"]["fused_top1_agreement"] == 1.0
        assert r["pools"]["clinc_dev"]["fused_top1_agreement"] == 1.0


def test_v3s_release_faults(tmp_path):
    R = run(tmp_path, "v3s_bad", style="v3s", fault="v3s_mode_mismatch", runs=CPU_TORCH[:1])
    assert R["status"] == "FAIL" and "vs stage-1.2 decision" in json.dumps(R["problems"]), R["problems"]
    S = RC.smoke_inputs(str(tmp_path / "in"), style="v3s")
    S.update(out_dir=str(tmp_path / "out"), run_name="v3s_nostage", runs=CPU_TORCH[:1], force_cpu=True,
             stage12_staged=None)
    R = RC.run_check(S)
    assert R["status"] == "REFUSED" and "STAGE12_STAGED" in R["refused"], R.get("refused")


def test_pass_fewshot_release(tmp_path):
    R = run(tmp_path, "fewshot", fault="fewshot")
    assert R["status"] == "PASS", R
    key = "clinc_dev [fewshot KX]"
    assert R["fewshot_row"]["examples"] > 0 and R["fewshot_row"]["labels_without_examples"] > 0, R["fewshot_row"]
    for r in R["runs"]:
        x = r["pools"][key]
        assert x["ok"] and x["fused_top1_agreement"] == 1.0 and x["max_abs_cross_diff"] < 1e-4, x
    assert R["fewshot_decision"]["chosen"]["variant"] == "KX"
    S = RC.smoke_inputs(str(tmp_path / "in"), fault="fewshot")
    S.update(out_dir=str(tmp_path / "out"), run_name="fs_nostage", runs=CPU_TORCH[:1], force_cpu=True,
             fewshot_staged=None)
    R = RC.run_check(S)
    assert R["status"] == "REFUSED" and "FEWSHOT_STAGED" in R["refused"], R.get("refused")


def test_deadline_cut_is_incomplete(tmp_path):
    S = RC.smoke_inputs(str(tmp_path / "in"))
    S.update(out_dir=str(tmp_path / "out"), run_name="deadline", runs=CPU_TORCH, force_cpu=True, n_timing=3,
             deadline_h=1e-7)
    R = RC.run_check(S)
    assert R["status"] == "INCOMPLETE", R["status"]
    assert R["skipped_runs"] == ["fast/float32"] and R["timing_shed"] is True and len(R["runs"]) == 1
    assert "timing" not in R["runs"][0] or R["runs"][0]["timing"] is not None


def test_full_row_counts_every_clinc_dev_row(tmp_path):
    """I4 (L01): with 10 rows per pool for the sampled rows, the full row still scores every clinc_dev row of the stored
    frame (44: 42 decision rows and 2 near_test rows)"""
    S = RC.smoke_inputs(str(tmp_path / "nrows" / "in"))
    S.update(out_dir=str(tmp_path / "nrows" / "out"), run_name="nrows", runs=CPU_TORCH[:1], n_timing=0, force_cpu=True,
             n_rows=10)
    R = RC.run_check(S)
    assert R["status"] == "PASS", R
    pools = R["runs"][0]["pools"]
    assert pools["clinc_dev"]["rows"] == 10 and pools["clinc_dev full"]["rows"] == 44
    assert R["rows"]["clinc_dev"]["n"] == 10 and R["rows"]["clinc_dev full"]["n"] == 44


def test_cpu_run_without_the_full_row_is_never_pass(tmp_path):
    """a CPU run outside the smoke scores the full clinc_dev row only with full_row_on_cpu; without it the status is
    INCOMPLETE, never PASS (the registered bar is GPU fp16 over clinc_dev at 0.99)"""
    out = {}
    for name, extra in (("cpu_nofull", {}), ("cpu_full", {"full_row_on_cpu": True})):
        S = RC.smoke_inputs(str(tmp_path / name / "in"))
        S.update(out_dir=str(tmp_path / name / "out"), run_name=name, runs=CPU_TORCH, n_timing=0, force_cpu=True,
                 smoke=False, **extra)
        out[name] = RC.run_check(S)
    assert out["cpu_nofull"]["status"] == "INCOMPLETE" and not out["cpu_nofull"]["full_row_runs"], out["cpu_nofull"]
    assert any("full clinc_dev row" in p for p in out["cpu_nofull"]["problems"])
    assert out["cpu_full"]["status"] == "PASS" and out["cpu_full"]["full_row_runs"] == ["eager/float32"], out["cpu_full"]


def test_pass_r2intent_names_honest_and_pub_record(tmp_path):
    """the R2Intent names (D6): the honest model Z repackaged (r2intent names, the stage-1.2 adapter file holds its
    tensors under a training prefix, SMOKE_TRAINING_PREFIX) and the P job's record (package files pinned in configs.p_full.package); every
    stand-in release carries the R2Intent names (the library reads no other)"""
    S = RC.smoke_inputs(str(tmp_path / "zr2" / "in"), style="honest")
    S.update(out_dir=str(tmp_path / "zr2" / "out"), run_name="zr2", runs=CPU_TORCH[:1], n_timing=0, force_cpu=True)
    R = RC.run_check(S)
    assert R["status"] == "PASS", R
    assert R["meta_file"] == "r2intent_meta.json" and R["source_tensors"]["equal"] and R["source_tensors"]["n_adapter"] > 0
    assert "r2intent_meta.json" in R["released_files"]
    assert not any(n.endswith("_meta.json") and n != "r2intent_meta.json" for n in R["released_files"])
    R2 = run(tmp_path, "pubrec", style="pub_record", runs=CPU_TORCH[:1])
    assert R2["status"] == "PASS" and R2["pin"] == "decision" and R2["config_name"] == "p_full", R2
    assert R2["source_tensors"]["equal"] and R2["meta_file"] == "r2intent_meta.json"
    for name, pin_req, want in (("zr2_nopin", False, "REFUSED"), ("zr2_srcpin", True, "PASS")):
        Sx = RC.smoke_inputs(str(tmp_path / name / "in"), style="honest", fault="unpinned_package")
        Sx.update(out_dir=str(tmp_path / name / "out"), run_name=name, runs=CPU_TORCH[:1], n_timing=0, force_cpu=True,
                  pin_by_source_tensors=pin_req)
        Rx = RC.run_check(Sx)
        assert Rx["status"] == want, (name, Rx["status"], Rx.get("refused"))
        if want == "PASS":
            assert Rx["pin"] == "source tensors" and Rx["source_tensors"]["equal"], Rx
    S4 = RC.smoke_inputs(str(tmp_path / "pr4" / "in"), style="pub_record", fault="pub_package_mismatch")
    S4.update(out_dir=str(tmp_path / "pr4" / "out"), run_name="pr4", runs=CPU_TORCH[:1], n_timing=0, force_cpu=True)
    R4 = RC.run_check(S4)
    assert R4["status"] == "REFUSED" and "is not a file the decision pins" in R4["refused"], R4.get("refused")
