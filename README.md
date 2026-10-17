<p align="center">
  <img src="docs/assets/banner.png" alt="AnyIntent: decisions in milliseconds. The logo is a blue bird drawn with white lines in a blue circle." width="100%">
</p>

<p align="center">
  <b>A multilingual decision engine for intents you describe in plain words.</b> Send a message and get the matching intent, with a margin that flags doubtful answers: no training examples, <a href="#results">about 26 ms per message on a GPU</a> (compiled), on your own server. With R2Intent v1 you can also ask yes or no for one intent (beta) or get a probability for every intent.
</p>

<div align="center">

[![R2Intent v1 on Hugging Face](https://img.shields.io/badge/Hugging%20Face-R2Intent%20v1-FFD21E?logo=huggingface)](https://huggingface.co/lukashm/r2intent-v1)
[![r2intent-zs-v1 on Hugging Face](https://img.shields.io/badge/Hugging%20Face-r2intent--zs--v1-FFD21E?logo=huggingface)](https://huggingface.co/lukashm/r2intent-zs-v1)
[![Library license: Apache 2.0](https://img.shields.io/badge/library%20license-Apache--2.0-blue)](LICENSE)
[![Model license: CC BY-NC 4.0, non-commercial](https://img.shields.io/badge/model%20license-CC%20BY--NC%204.0-blue)](#license)
[![Version 0.1.0](https://img.shields.io/badge/version-v0.1.0-blue)](https://github.com/lukamaga/anyintent/tree/v0.1.0)
[![Python 3.9 or later](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](pyproject.toml)

</div>

<p align="center">
  <a href="#quick-start">Quick start</a> &middot;
  <a href="#results">Results</a> &middot;
  <a href="#where-to-go-next">Documentation</a> &middot;
  <a href="https://huggingface.co/lukashm/r2intent-v1">Full report</a> &middot;
  <a href="#contact">Contact</a>
</p>

AnyIntent is a Python library that picks the matching intent for a customer message from a list of intents you give at run time. Each intent is a name with an optional short description, and no training examples of your intents are needed. Besides `classify`, it answers yes or no for one intent with a probability (`check`, a beta) and gives a value to every intent (`scores`), and it can run as a local HTTP server or MCP server. It runs the R2Intent models, published on Hugging Face:

- [R2Intent v1](https://huggingface.co/lukashm/r2intent-v1), the product model. It was trained on public intent data, including parts of the training splits of Banking77, HWU64, MASSIVE and CLINC150 banking and credit_cards, so it is not a zero-shot model and its results on those benchmarks are in-domain results.
- [r2intent-zs-v1](https://huggingface.co/lukashm/r2intent-zs-v1), the zero-shot research model (zs: trained without any training data of the registered test benchmarks and without banking data), for research and comparison only.

R2Intent v1 was trained on English and 50 other languages. It was checked on DEV, the project's development messages, in English, Lithuanian, Polish, German and Slovak; the non-English DEV messages are machine translations.

License: the library code is under the Apache License 2.0 ([LICENSE](LICENSE) and [NOTICE](NOTICE)). The models are not part of this library, and its license gives no rights to them. Both models are under CC BY-NC 4.0, non-commercial use only. A commercial license for R2Intent v1 is available on request. For commercial use, contact lukashcode@gmail.com or info@liriti.com. A company can also ask there for a free 30-day evaluation, to test R2Intent v1 on its own messages before buying. No commercial license is offered for r2intent-zs-v1.

## Results

Accuracy on the test sets of public benchmarks. R2Intent v1 was measured once, on the frozen files of its release; the other models on 2026-10-08, before R2Intent v1 was trained, in the project's single look at the test sets and in the baselines registered before it. The best number in each row is in bold.

| Benchmark | R2Intent v1 | r2intent-zs-v1 | Qwen3-Embedding-8B | EmbeddingGemma 2 | Laya |
|---|---|---|---|---|---|
| Banking77 | 0.810 | 0.711 | 0.766 | **0.933** | 0.452 |
| HWU64 | 0.860 | 0.705 | 0.845 | **0.865** | 0.569 |
| MASSIVE, 17 languages | 0.774 | 0.587 | **0.832** | 0.660 | 0.311 |
| MTOP, German, French and Spanish | 0.633 | 0.547 | **0.728** | 0.677 | 0.301 |
| xSID, Lithuanian | **0.888** | 0.864 | 0.840 | 0.310 | 0.418 |
| MInDS-14, 9 languages | **0.896** | 0.895 | 0.737 | 0.814 | 0.615 |
| SIB-200, 9 languages | 0.715 | 0.714 | **0.795** | 0.722 | 0.761 |
| Mean of the 7 benchmarks above | **0.796** | 0.717 | 0.792 | 0.712 | 0.489 |
| CLINC150 banking and credit cards | **0.949** | 0.909 | 0.784 | 0.842 | 0.882 |
| New topics (DEV, 8 topics in 5 languages) | **0.781** | 0.779 | 0.641 | 0.546 | 0.482 |
| Parameters | 582 million | 582 million | 7.57 billion | 744 million | 421 million |

R2Intent v1 was trained on public data that include the training splits of Banking77, HWU64, MASSIVE and CLINC150 banking and credit cards, with every training message close to a test message removed; it was not trained on the data of MTOP, xSID, MInDS-14 and SIB-200, although their topics are partly covered. r2intent-zs-v1 was trained without the training data of any of these benchmarks and without banking data. The training data of Qwen3-Embedding-8B and EmbeddingGemma 2 are not disclosed, and Banking77, MASSIVE and MTOP are part of the public MTEB collection, so whether those models saw them is not known. Qwen3-Embedding-8B and EmbeddingGemma 2 are shown with intent names, their better setting in these runs; Laya with its default settings. New topics are 8 DEV topics that no public dataset contains, in English, German, Lithuanian, Polish and Slovak; they were generated by the same model that generated part of this project's synthetic training data, which favours this project's models. Parameter counts are those of the published model files.

Every number of the evaluation is published, including the rows where other models score higher: the results per language, with intervals and macro-F1, and the test results of `check` and `scores` are in the model card of R2Intent v1 (Test results in its full report) and in the protocol.

Speed on a Kaggle T4 GPU, one message at a time: R2Intent v1 25.86 ms per message (median; 90th percentile 27.41 ms).

This was measured on the English CLINC150 DEV messages at k = 5, in fp16, with torch.compile, CUDA graphs and the intent vectors computed in advance. Only this compiled setup has a published speed; none is given for the library's default (`compile=False`).

Size: 582 million parameters, one model file of 1.19 GB.

Robustness: on DEV messages changed by fixed rules, R2Intent v1 was more accurate than the zero-shot model r2intent-zs-v1 by 0.091 with an e-mail greeting and signature around the message, by 0.037 on Lithuanian and Polish messages without diacritics and by 0.009 with one typo. These are DEV numbers, not a held-out comparison: R2Intent v1 was trained with changes of these kinds (though not with the greetings and signatures of this check), some of these DEV sets are in the domains of its training data, and the variation between training runs under these changes was not measured, so a small difference is not evidence either way. The model card gives the intervals and every caveat.

## Limitations

- The model always picks one of your intents: there is no "none of these fits" answer, and no measurement on messages that fit none of the given intents is reported (see [Doubtful answers](#doubtful-answers)).
- Every non-English DEV number comes from machine-translated messages, not from messages written by customers in those languages, and Lithuanian has no human-written training rows.
- R2Intent v1 was trained once, with a single training seed, so the numbers on this page do not show how much results move between training runs.
- R2Intent v1 was trained on only part of the training splits of Banking77, HWU64, CLINC150 banking and credit_cards and MASSIVE. After the removal of training rows close to test messages (rule E4 of the protocol) and the other data rules, about 23% of the Banking77 and 33% of the HWU64 training rows that were not held out for validation remain, and so do about 53% of the CLINC150 banking and credit_cards training rows and about 12% of the MASSIVE en-US utterances (the full report on the model card, Disclosures, 'What survives E4').
- `check` is a beta: it missed one clause of its calibration rule (see [check (beta) and scores](#check-beta-and-scores)).

The [model card of R2Intent v1](https://huggingface.co/lukashm/r2intent-v1) on Hugging Face has the full list.

## Quick start

You need Python 3.9 or later, git and a Hugging Face account. On macOS or Windows, read [Systems](#systems) first.

1. Install the library. pip also installs what the library needs: PyTorch 2.6 or later, transformers 4.56 or later, safetensors, huggingface_hub and numpy.

   ```bash
   pip install "git+https://github.com/lukamaga/anyintent@v0.1.0"
   python -c "import anyintent; print(anyintent.__version__)"    # prints 0.1.0
   ```

2. Request access to the model. Open the [R2Intent v1 page](https://huggingface.co/lukashm/r2intent-v1) while logged in to Hugging Face, read its license and click **Agree and request access**. When you request access, Hugging Face shares your user name and e-mail address with the owner of the model repository. They are used only to manage access and to answer license requests. Access is granted automatically. r2intent-zs-v1 is gated the same way, with a short form (your name, affiliation, intended use and a box that confirms non-commercial research).

3. Log in once on this machine with an access token from https://huggingface.co/settings/tokens (read access is enough):

   ```bash
   hf auth login
   ```

   Or set the environment variable `HF_TOKEN` to the token.

4. Run this. The first run downloads the model into the Hugging Face cache (`HF_HUB_CACHE` moves it): one model file of 1.19 GB and a few smaller files (its meta, config and tokenizer files, LICENSE and NOTICE).

   ```python
   from anyintent import AnyIntent

   clf = AnyIntent.from_pretrained("lukashm/r2intent-v1", revision="main")  # or a commit hash of the model repository, to pin its files
   clf.set_labels([
       {"name": "lost card", "description": "The customer reports that a bank card was lost or stolen."},
       {"name": "refund status", "description": "The customer asks when a refund will arrive."},
       "close account",  # a name alone also works
   ])

   p = clf.classify("I can't find my card anywhere")
   print(p.label, p.score, p.margin)

   preds = clf.classify(["Kur yra mano pinigai?", "Chcę zamknąć konto"])  # a list in, a list out
   ```

If loading fails with an `AccessError`, its message says which case it is:

- `No access to the model repository ... It is gated`: you are not logged in, or the account you are logged in with has no access yet. Request access (step 2) and log in with that account (step 3).
- `... was not found, is private, or you are not logged in`: check the model id and the revision, then log in (step 3).
- `Access to ... was refused (HTTP 403)`: check that your token has read access.

With transformers 5, loading a model can print a `LOAD REPORT` that lists the adapter's `r2intent_embed_lora.*` keys as `UNEXPECTED`. This is expected: the library reads those keys itself (see [Model folders](docs/advanced.md#model-folders)).

## What you get back

`classify` returns a `Prediction` (a list of them for a list of messages):

- `label`: the chosen intent.
- `score`: its fused score (higher is better; not a probability).
- `margin`: how far the best intent is ahead of the second best, in fused-score units (`None` when only one intent was scored). A small margin means a doubtful answer.

The full signature, the other methods (`set_labels`, `warm_up`, `predict`, `config`) and the rules for threads are in [docs/advanced.md](docs/advanced.md#the-python-api).

### Doubtful answers

```python
p = clf.classify(message)
if p.margin is not None and p.margin < THRESHOLD:     # THRESHOLD: choose it on your own labeled messages (next section)
    route_to_fallback(message)                           # a person, or a larger model
else:
    handle(p.label)
```

The model has no "none of these fits" answer: a message that fits none of your intents still gets one, so a small margin is the signal to route it elsewhere. A fallback is outside the measured speed of the model.

## Test it on your own messages

Before you rely on the model, test it on messages from your own traffic that you have labeled with the right intent. A company that runs this test for its business needs a commercial license or the free evaluation (see [License](#license)). Save them as `messages.csv` with two columns, `text` and `intent`, one message per row. In the script, replace the three example intents in `DESCRIPTIONS` with yours, each with a short, plain description, and spell each intent exactly as in the file. The script prints the accuracy, what three margin thresholds would do and every wrong answer.

```python
import csv
from anyintent import AnyIntent

DESCRIPTIONS = {  # every intent in your file, with a short, plain description
    "lost card": "The customer reports that a bank card was lost or stolen.",
    "refund status": "The customer asks when a refund will arrive.",
    "close account": "The customer wants to close their account.",
}

with open("messages.csv", newline="", encoding="utf-8-sig") as f:  # columns: text, intent
    rows = list(csv.DictReader(f))
assert {r["intent"] for r in rows} <= DESCRIPTIONS.keys(), "every intent in the file needs a description"

clf = AnyIntent.from_pretrained("lukashm/r2intent-v1", revision="main")
clf.set_labels([{"name": n, "description": d} for n, d in DESCRIPTIONS.items()])
preds = clf.classify([r["text"] for r in rows])  # a list in, a list out
right = [p.label == r["intent"] for p, r in zip(preds, rows)]
print(f"accuracy: {sum(right) / len(rows):.3f} on {len(rows)} messages")

# What a margin threshold would do: the answers below it go to a person
margins = sorted(p.margin for p in preds)
for share in (0.05, 0.1, 0.2):
    threshold = round(margins[int(share * len(margins))], 2)
    rest = [ok for ok, p in zip(right, preds) if p.margin >= threshold]
    print(f"threshold {threshold:.2f}: {1 - len(rest) / len(rows):.0%} doubtful, "
          f"accuracy of the rest {sum(rest) / max(len(rest), 1):.3f}")

for r, p, ok in zip(rows, preds, right):
    if not ok:
        print(f"wrong: {r['text']!r} is {r['intent']!r}, the model said {p.label!r} (margin {p.margin:.2f})")
```

- Save the file in UTF-8: a file in another encoding can stop the script with a UnicodeDecodeError. If your spreadsheet program separates the columns with semicolons, change `csv.DictReader(f)` to `csv.DictReader(f, delimiter=";")`.
- On Windows, when you send the output to a file, Python 3.14 and earlier write it in the system's code page, and a letter that this code page lacks stops the script. In cmd, with the script saved as, for example, `check_messages.py`, run `python -X utf8 check_messages.py > out.txt`. For PowerShell, see [docs/windows.md](docs/windows.md#run-the-quick-start).
- Each threshold line shows what share of answers would go to a person and how accurate the rest would be. Pick a threshold that suits you. Use it as `THRESHOLD` in the code of [Doubtful answers](#doubtful-answers), or pass it to the HTTP server as `--doubt-margin`; the server then marks the answers below it as doubtful. The margin is in score units, not a probability. Its scale depends on the model and its settings, so choose the threshold on your own messages.
- With few messages, the three lines can show the same threshold, and a line can leave no answer at or above its threshold: its accuracy of the rest then shows as zero, which means that no answer is left. The HTTP server's `--doubt-margin` takes only a value above zero.
- Tune the descriptions and the threshold on one part of your messages and keep another part, for example one in five, for a single final test. Numbers from the messages you tuned on usually look better than the numbers on real traffic.
- Leave out messages that fit none of your intents, or count them separately: the model always picks one of your intents, so on such messages it is always wrong.
- Do not paste messages from `messages.csv` into the descriptions. That makes the test look too good (see [Writing labels](#writing-labels)).

## check (beta) and scores

Besides `classify`, two functions give numbers from 0 to 1 that were calibrated for R2Intent v1: `check` and `scores`.

```python
r = clf.check("I can't find my card anywhere",
              {"name": "lost card", "description": "The customer reports that a bank card was lost or stolen."})
r.yes, r.probability     # yes or no for this one intent, and the probability of a match

s = clf.scores("I can't find my card anywhere")    # every intent of the current label set
s.label, s.values        # classify's label, and {name: value} in set_labels order
```

- Scope: R2Intent v1 only, as measured on held-out DEV data, for intents written with a description, at the threshold 0.5 of `check`, with the library's GPU engines `auto` and `eager` in fp16 and on label sets of 4 to 63 intents. r2intent-zs-v1 ships no calibration file, so both functions refuse it; `classify` works the same either way.
- `scores` gives a value from 0 to 1 to every intent of the label set: for an intent the model read, the chance that it is the right one; an intent it did not read gets 0.0, which means "not read", not "impossible". It passed every clause of its rule (S1, the calibration error of the largest value on all held-out rows: 0.019 [0.012, 0.035]; limit 0.030, and 0.050 for the upper end of the interval).
- `check` answers yes or no for one intent, with a probability. It missed one clause of its rule, C4: on the unseen English DEV sets, its probabilities were off by 0.0566 on average, above the limit of 0.050; every other clause passed. It ships as a beta, with its parameters exactly as fitted, by a decision taken after this result, so treat its probability as approximate. A yes does not mean that the intent is the best one. To choose one intent, use `classify`.

If the intent is rare in your traffic, a yes is right less often. The share of yes answers that are right, from the held-out rates of right yes and right no answers:

| messages that express the intent | share of yes answers that are right [95% interval] |
|---|---|
| 1 in 2 | 0.888 [0.862, 0.909] |
| 1 in 5 | 0.665 [0.610, 0.715] |
| 1 in 20 | 0.295 [0.248, 0.346] |

[docs/calibration.md](docs/calibration.md) has the rest: what the probability means, how the unseen DEV sets were chosen, every clause with its result, the formula of `scores` and its remainder, the refusals, and when the calibration file is downloaded.

## How it works

1. Retrieval. One XLM-RoBERTa large network turns the message and every intent text into vectors, with a small LoRA adapter switched on, and keeps the k intents closest to the message. An intent this step misses cannot be chosen.
2. Reranking. With the adapter off, the same network reads the message together with each of those k intent texts (the cross-encoder pass) and scores each pair.
3. Fusion. The final score of an intent is `cross-encoder score + w x cosine`, and the highest score wins, so the model always picks one of your intents. `w` and `k` are fixed when the model is trained and stored with it (w = 60 and k = 5 for both released models).

## Writing labels

The model reads each intent as the text `name: description` (the name alone when there is no description). This is the label text format of both released models, and their numbers were measured with it. Write a name and a short, plain description of what the customer wants, as in the quick start. Long texts are cut: R2Intent v1 reads at most 160 tokens of the message and the intent text together, and when a pair is longer, the longer of the two is cut first. Do not paste example messages from your own test data into the descriptions; that makes the test numbers look better than the numbers on real traffic.

The library also supports models released with another description layout (`config["label_text_mode"]` is `"v3s"`). Neither released model uses it. With such a model and plain descriptions, pass `label_text_mode="v2"`: when the model carries registered settings for plain descriptions (`config["fusion_by_mode"]["v2"]`), they are used for that label set, and the library warns once and says which settings apply.

## Run it as a local service

Two optional ways to use the library from other programs: an HTTP server (extra `server`: FastAPI, uvicorn and pydantic) and an MCP server for agents and editors that speak the Model Context Protocol (extra `mcp`: the official MCP Python SDK 1.x, which needs Python 3.10 or later). Both load one model at start, only call the library and run on your machine.

```bash
pip install "anyintent[server] @ git+https://github.com/lukamaga/anyintent@v0.1.0"
anyintent-serve --model lukashm/r2intent-v1 --lists-dir ./intent-lists
```

```bash
curl -s http://127.0.0.1:8000/classify -H "Content-Type: application/json" -d '{
  "texts": ["I cannot find my card anywhere", "Chcę zamknąć konto"],
  "intents": [
    {"name": "lost card", "description": "The customer reports that a bank card was lost or stolen."},
    {"name": "refund status", "description": "The customer asks when a refund will arrive."},
    "close account"
  ]
}'
```

On Windows, use the PowerShell version in [docs/windows.md](docs/windows.md#the-http-server).

- It listens on `127.0.0.1:8000`, this machine only, and the OpenAPI page with every request and response schema is at `http://127.0.0.1:8000/docs`.
- Message texts are not stored, and they are not logged unless the server is started with `--log-texts`.
- No speed was measured for the servers.
- Docker files for the HTTP server are in `docker/`, but they were not built or tested for this release.

[docs/server.md](docs/server.md) gives every endpoint, answer, error, option and limit of the HTTP server, and [docs/mcp.md](docs/mcp.md) covers the MCP server (`anyintent-mcp`), its tools and a client configuration.

## Hardware

The models are GPU-first. On a CUDA GPU the library runs the fp16 weights with autocast; on a CPU it runs fp32. CPU fp32 results can differ from the GPU fp16 numbers on near ties. No CPU speed was measured. For CPUs there is also an ONNX path (extra `onnx`), checked against the reference on tiny models only. [docs/advanced.md](docs/advanced.md#hardware-and-engines) describes it and the engines.

## Systems

The library and both released models were tested on the CPU, on GitHub Actions runners with Ubuntu 24.04, macOS 26 on Apple Silicon and Windows Server 2025. Windows 10 and 11 themselves and a GPU on Windows were not part of these runs.

- Linux: the released models were measured with NVIDIA GPUs. Their speed was measured on a Kaggle T4. PyTorch from PyPI includes CUDA on Linux. From version 2.11 on, the PyPI builds for Linux use CUDA 13, which needs a GPU of the Turing generation or later (GTX 16 and RTX 20 series and newer) and NVIDIA driver 580 or newer. For an older GPU, install PyTorch from `https://download.pytorch.org/whl/cu126` before the library. With the Python of a Linux distribution, install the library in a virtual environment (`python3 -m venv .venv`, then `source .venv/bin/activate`).
- macOS: Apple Silicon only, since PyPI has no PyTorch 2.6 or later for Intel Macs. The library uses a CUDA GPU when there is one, and the CPU otherwise, so on a Mac it uses the CPU.
- Windows 10 and 11: use 64-bit x64 Python (3.12 and 3.13 are the safe choices) and Git for Windows. PyTorch from PyPI is CPU only on Windows. For an NVIDIA GPU, install PyTorch from the pytorch.org index before the library. Leave `compile` off (the default): it needs Triton, which has no official Windows build. [docs/windows.md](docs/windows.md) has the steps and the PowerShell commands. In WSL 2 (Linux inside Windows), the Linux notes apply.

## Where to go next

- [docs/calibration.md](docs/calibration.md): `check` and `scores` in full, with every clause of their rules and its result.
- [docs/server.md](docs/server.md): the HTTP server, its endpoints, errors, API key, limits and privacy.
- [docs/mcp.md](docs/mcp.md): the MCP server, its tools and a client configuration.
- [docs/advanced.md](docs/advanced.md): install details and extras, the full Python API, the files of a release, model folders, engines and ONNX, the few-shot mode, tests.
- [docs/windows.md](docs/windows.md): step by step on Windows 10 and 11, with PowerShell and cmd commands.
- [docker/README.md](docker/README.md): Docker images of the HTTP server.
- The model cards of [R2Intent v1](https://huggingface.co/lukashm/r2intent-v1) and [r2intent-zs-v1](https://huggingface.co/lukashm/r2intent-zs-v1): training data, results with their caveats, limitations and, folded at the end, the full report (see [Protocol](#protocol)).

## Protocol

The models were built and tested under a registered protocol. The full report that it requires for each release, each statement with the record it comes from, is folded at the end of each model card, under "Full report (PROTOCOL)": [R2Intent v1](https://huggingface.co/lukashm/r2intent-v1), [r2intent-zs-v1](https://huggingface.co/lukashm/r2intent-zs-v1). A public copy of the protocol, with every rule and result in the order they were logged, is in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Contact

Questions, feedback and collaboration: lukashcode@gmail.com or info@liriti.com. LinkedIn: https://www.linkedin.com/in/lukashm/

To report a bug or propose a change, see [CONTRIBUTING.md](CONTRIBUTING.md). To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Citation

```bibtex
@misc{magalinski2026r2intent,
  author       = {Lukaš Patrik Magalinski},
  title        = {R2Intent v1: intent classification with your own list of intents},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/lukashm/r2intent-v1}},
  url          = {https://huggingface.co/lukashm/r2intent-v1}
}
```

## License

The code of this library is under the Apache License 2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE). The R2Intent models are not part of the library. Both are under CC BY-NC 4.0, non-commercial use only, and each comes with its own LICENSE and NOTICE files on its Hugging Face page. A commercial license for R2Intent v1 is available on request (lukashcode@gmail.com or info@liriti.com); none is offered for r2intent-zs-v1. A company can also write to the same addresses for a free 30-day evaluation, to test R2Intent v1 on its own messages before buying a commercial license. When you share a model, share LICENSE, NOTICE, the three license texts and hint3_alterations.json with it (see [Files of a release](docs/advanced.md#files-of-a-release)).
