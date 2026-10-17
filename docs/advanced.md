# Install details, Python API and engines

[Back to the README](../README.md)

## Install details

```bash
pip install "git+https://github.com/lukamaga/anyintent@v0.1.0"
```

pip also installs what the library needs: PyTorch 2.6 or later, transformers 4.56 or later, safetensors, huggingface_hub and numpy. Optional extras, each with the same pinned form:

```bash
pip install "anyintent[onnx] @ git+https://github.com/lukamaga/anyintent@v0.1.0"     # ONNX Runtime path for CPUs
pip install "anyintent[server] @ git+https://github.com/lukamaga/anyintent@v0.1.0"   # HTTP server
pip install "anyintent[mcp] @ git+https://github.com/lukamaga/anyintent@v0.1.0"      # MCP server
```

The `mcp` extra needs Python 3.10 or later, as every mcp 1.x release does.

On Windows, [windows.md](windows.md) gives the steps in PowerShell and cmd, including how to install PyTorch for an NVIDIA GPU.

Building from source without pip's build isolation (`--no-build-isolation`) needs setuptools 77 or later and packaging 24.2 or later in your environment.

## Files of a release

`from_pretrained` downloads the files in `anyintent.hub.MODEL_FILES` (the model, its meta, config and tokenizer files, LICENSE and NOTICE), and the calibration file at the first call of `check` or `scores` (see Loading in [calibration.md](calibration.md#notes-for-check-and-scores)). The other files of a release (the model card, r2intent_runtime.py, LICENSE-APACHE-2.0.txt, LICENSE-MIT-FlagEmbedding.txt, LICENSE-MIT-fairseq.txt and hint3_alterations.json) stay in its Hugging Face repository: `hf download lukashm/r2intent-v1` downloads all of them. When you share a model, share LICENSE, NOTICE, the three license texts and hint3_alterations.json with it.

## The Python API

- `classify(text, k=None, cascade=None, return_scores=False)` returns a `Prediction` (a list of them for a list of messages). `label` is the chosen intent and `score` its fused score (higher is better; not a probability). `k=None` uses the model's k (5 for both released models, the k of the published numbers); a smaller k gives the cross-encoder fewer intents to read. `return_scores=True` adds every candidate with its cosine, cross-encoder score and fused score.
- `Prediction.margin` is the fused score of the best intent minus that of the second best among the scored intents (`None` when only one was scored). A small margin means a doubtful answer (see [Doubtful answers](../README.md#doubtful-answers) in the README). The margin is not a calibrated probability: it is in fused-score units (cross-encoder logit + w x cosine), so its scale depends on w, on k and on whether the cascade ran, and a threshold chosen for one model and setting does not carry over to another.
- `cascade=True` turns on an early-stop cascade (the first few intents by retrieval are scored first, the rest only when the top two are close). It needs an early-stop threshold stored with the model; the released models have none, so with them it is refused. It is off by default, and the published numbers are without it.
- `set_labels` computes the intent vectors once; they are cached in memory, and on disk with `cache_dir=...`.
- `warm_up()` runs a few dummy messages, so that one-time start-up work is done before the first real request.
- `predict(text, labels=[...])` does `set_labels` and `classify` in one call and returns the label name.
- `config` shows the resolved settings, including the ones in use for the current label set (`config["active"]`).
- Pin a revision (a commit hash of the model repository) when you need the published numbers to apply exactly.
- Threads: one instance serializes its calls, so it can be shared by the threads of a web server. Threads that use different label sets should call `predict(text, labels=...)` (labels and classification under one lock) or use one instance each. For parallel work use one instance per worker process.

`check` and `scores` are in [calibration.md](calibration.md).

## Model folders

A released model folder holds `model.safetensors` (the backbone and head in fp16, the adapter in fp32 under `r2intent_embed_lora.*`), the config and tokenizer files and the meta file `r2intent_meta.json` with the frozen settings and the label text mode. Model folders from the project's earlier development packages use an older name for the meta file and for the adapter prefix; the library loads them the same way. A folder with both meta files, or with names of the two kinds mixed, is refused. The older two-file layout (a bge-m3 encoder plus a separate cross-encoder) loads too; it needs an `anyintent.json` next to the model or the `embedder=` argument.

With transformers 5, loading a model can print a `LOAD REPORT` that lists the `r2intent_embed_lora.*` keys as `UNEXPECTED` (seen with transformers 5.5.0 on a tiny model). This is expected: those keys are the adapter's tensors, which transformers does not know and the library reads itself. The library refuses a model with any other unexpected or missing key.

## Hardware and engines

The models are GPU-first. On a CUDA GPU the library runs the fp16 weights with autocast; on a CPU it runs fp32. CPU fp32 results can differ from the GPU fp16 numbers on near ties. No CPU speed was measured.

Engines: `engine="eager"` is the plain Hugging Face forward (the reference), and `engine="fast"` uses equivalent hand-written bodies that skip unneeded work. The default `auto` uses the fast engine after checking at load time that it agrees with the reference within a fixed tolerance (1e-4 in fp32; in fp16 5e-3 for cosines and the larger of 0.05 and 1% of the logits). Use `engine="eager"` on a GPU in fp16 for the computation closest to the one the published numbers were made with; across GPUs, drivers and batch sizes a few predictions on near ties can differ. `compile=True` adds `torch.compile` with CUDA graphs on CUDA; it is experimental.

ONNX (CPU, extra `onnx`):

```python
from anyintent import AnyIntent, onnx
onnx.export("lukashm/r2intent-v1", "r2intent-onnx", revision="main")
clf = AnyIntent.from_pretrained("lukashm/r2intent-v1", revision="main", engine="onnx", onnx_dir="r2intent-onnx")
```

The ONNX path was checked against the reference on tiny models only. `onnx.export(..., quantize_int8=True)` also writes an int8 variant (`onnx_variant="int8"`), whose accuracy has not been measured; `check` and `scores` refuse it.

## Few-shot mode (not used by the released models)

The library also has a few-shot mode for model versions that carry a few-shot configuration (`config["fewshot"]` is not `None`): `set_labels(labels, examples=...)` then takes a few real example messages per label, and nothing is trained. Neither released model carries such a configuration, so with them `examples=` raises an error, and their behavior never changes silently.

That error message calls such a model "zero-shot only", and the docstrings of the package use the same term: in the library's code, docstrings and messages, zero-shot means without example messages at run time. It does not mean that a model never saw benchmark training data; R2Intent v1 did (see the top of the [README](../README.md)).

## What is not here

Training, data building and evaluation code are not part of this package.

## Tests of the library (not of the model's accuracy)

```bash
pip install -e ".[test]"
pytest
```

The tests build tiny random models and check that the library gives the same predictions and scores as the reference implementation of the models (within 1e-4 in fp32). They need no network. The tests of the HTTP and MCP servers need their extras (`pip install -e ".[test,server,mcp]"`); without them, the tests that need those packages are skipped. These tests do not measure how accurate a model is; for that, see [Test it on your own messages](../README.md#test-it-on-your-own-messages) in the README. On Windows, see [windows.md](windows.md#run-the-library-tests).
