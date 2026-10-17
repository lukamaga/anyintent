"""The example of the README section "Test it on your own messages" runs as printed: its only code block, with the model
id replaced by a tiny model at the settings of R2Intent v1 (tests/calib_support.py make_p_like; revision="main" stays,
since a local folder is used as it is), on a messages.csv with a UTF-8 BOM (as Excel's CSV UTF-8 writes it) and English,
Lithuanian and Polish rows, in a fresh interpreter with the folder of the CSV file as its working directory. No
network: the model is a local folder and the hub is offline."""
import csv
import os
import re
import site
import subprocess
import sys

import pytest
import torch

import anyintent
from anyintent import AnyIntent
from calib_support import make_p_like

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
SECTION = "## Test it on your own messages"
MODEL_ID = '"lukashm/r2intent-v1"'
MISSING = "every intent in the file needs a description"
ROWS = [  # (text, intent): 4 per intent, in English, Lithuanian and Polish
    ("I lost my card yesterday, please block it", "lost card"),
    ("Someone stole my wallet with my debit card in it", "lost card"),
    ("Pamečiau savo banko kortelę", "lost card"),
    ("Zgubiłem kartę, co mam zrobić?", "lost card"),
    ("When will I get my money back for the returned shoes?", "refund status"),
    ("I returned the item two weeks ago, where is my refund?", "refund status"),
    ("Kada grąžinsite pinigus už grąžintą prekę?", "refund status"),
    ("Kiedy dostanę zwrot pieniędzy?", "refund status"),
    ("I want to close my account", "close account"),
    ("Please close my account and delete my data", "close account"),
    ("Noriu uždaryti savo sąskaitą", "close account"),
    ("Chcę zamknąć konto", "close account"),
]
ACCURACY_RE = re.compile(r"accuracy: (\d\.\d{3}) on (\d+) messages")
THRESHOLD_RE = re.compile(r"threshold (-?\d+\.\d{2}): (\d+)% doubtful, accuracy of the rest (\d\.\d{3})")
WRONG_RE = re.compile(r"wrong: (.+) is '([^']+)', the model said '([^']+)' \(margin (\d+\.\d{2})\)")


def readme_example():
    """the only code block of the section, which must be a python block"""
    with open(README, encoding="utf-8") as f:
        lines = f.read().split("\n")
    start = lines.index(SECTION)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    fences = [i for i in range(start + 1, end) if lines[i].startswith("```")]
    assert len(fences) == 2, f"the section must hold exactly one code block (fence lines {fences})"
    assert lines[fences[0]] == "```python" and lines[fences[1]] == "```", (lines[fences[0]], lines[fences[1]])
    return "\n".join(lines[fences[0] + 1:fences[1]]) + "\n"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory, tok):
    return make_p_like(str(tmp_path_factory.mktemp("docs_example") / "p_like"), tok)[1]


def write_csv(folder, rows):
    with open(os.path.join(folder, "messages.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["text", "intent"])
        w.writerows(rows)


def run_example(folder, model):
    code = readme_example()
    assert code.count(MODEL_ID) == 1, "the example must load the model by its id exactly once"
    with open(os.path.join(folder, "example.py"), "w", encoding="utf-8") as f:
        f.write(code.replace(MODEL_ID, repr(str(model))))
    threads = str(torch.get_num_threads())
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    # the anyintent under test, also when it is found through a relative PYTHONPATH (such as src) and not installed
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(anyintent.__file__)))
    if pkg not in site.getsitepackages() + [site.getusersitepackages()]:
        env["PYTHONPATH"] = os.pathsep.join([pkg] + [x for x in [os.environ.get("PYTHONPATH")] if x])
    r = subprocess.run([sys.executable, "example.py"], cwd=folder, env=env, capture_output=True, timeout=900)
    return r.returncode, r.stdout.decode("utf-8"), r.stderr.decode("utf-8", "replace")


def test_readme_example_runs_on_a_tiny_model(tiny_model, tmp_path):
    write_csv(str(tmp_path), ROWS)
    with open(tmp_path / "messages.csv", "rb") as f:
        assert f.read(3) == b"\xef\xbb\xbf"
    rc, out, err = run_example(str(tmp_path), tiny_model)
    assert rc == 0, err[-3000:]
    lines = out.splitlines()
    assert len(lines) >= 4, out

    m = ACCURACY_RE.fullmatch(lines[0])
    assert m and int(m.group(2)) == len(ROWS), lines[0]
    accuracy = float(m.group(1))

    shown = [THRESHOLD_RE.fullmatch(x) for x in lines[1:4]]
    assert all(shown), lines[1:4]
    thresholds = [float(x.group(1)) for x in shown]
    doubtful = [int(x.group(2)) for x in shown]
    assert thresholds == sorted(thresholds) and doubtful == sorted(doubtful) and doubtful[-1] <= 100, lines[1:4]
    assert all(0.0 <= float(x.group(3)) <= 1.0 for x in shown), lines[1:4]

    wrong = [WRONG_RE.fullmatch(x) for x in lines[4:]]
    assert all(wrong), lines[4:]
    assert len(wrong) == round((1 - accuracy) * len(ROWS)) and wrong, out
    gold = dict(ROWS)
    names = set(gold.values())
    for x in wrong:
        text = x.group(1)
        assert text[0] == text[-1] == "'" and text[1:-1] in gold, text          # the message, as repr() prints it
        assert gold[text[1:-1]] == x.group(2) and x.group(3) in names and x.group(3) != x.group(2), x.group(0)

    # the same answers as the library called here (the labels; floats can differ in the last digits between processes)
    descriptions = re.findall(r'^    "([^"]+)": "([^"]+)",$', readme_example(), re.M)
    assert [n for n, _ in descriptions] == ["lost card", "refund status", "close account"], descriptions
    clf = AnyIntent.from_pretrained(tiny_model, device="cpu")
    clf.set_labels([{"name": n, "description": d} for n, d in descriptions])
    preds = clf.classify([t for t, _ in ROWS])
    want = {(repr(t), g, p.label) for (t, g), p in zip(ROWS, preds) if p.label != g}
    assert {(x.group(1), x.group(2), x.group(3)) for x in wrong} == want
    assert accuracy == round(sum(p.label == g for (_, g), p in zip(ROWS, preds)) / len(ROWS), 3)


def test_readme_example_stops_on_an_intent_without_a_description(tiny_model, tmp_path):
    assert MISSING in readme_example()
    write_csv(str(tmp_path), ROWS[:3] + [("My card was blocked at the shop", "card blocked")])
    rc, out, err = run_example(str(tmp_path), tiny_model)
    assert rc != 0 and out == "", (rc, out)
    assert f"AssertionError: {MISSING}" in err, err[-3000:]
