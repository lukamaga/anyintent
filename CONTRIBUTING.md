# Contributing to AnyIntent

Thank you for your interest in AnyIntent. This page explains how to report a bug, propose a change and run the tests. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Report a bug

Open an issue on GitHub. Please include:

- the library version (`python -c "import anyintent; print(anyintent.__version__)"`),
- your Python version and your operating system,
- the command or the code you ran,
- the full error message, with its traceback.

Do not put private data in an issue: no access tokens (such as a Hugging Face token or the server's API key), no passwords and no messages from your customers. If an example needs messages, write made-up ones.

To report a security problem, do not open an issue. See [SECURITY.md](SECURITY.md).

## Propose a change

Fork the repository on GitHub, make your change on a new branch and open a pull request. For a larger change, please open an issue first, so we can discuss it before you write the code.

- Keep the style of the code around your change.
- Add or update tests for what you change.
- Keep one change in one pull request.
- Run the tests before you open it (next section).

## Run the tests

The tests build tiny random models. They need no model download and no network. They check that the library gives the same predictions and raw scores as the reference implementation of the models; they do not measure how accurate a model is.

Run them in a virtual environment, from a clone of the repository:

```bash
python3 -m venv anyintent-env
source anyintent-env/bin/activate
git clone https://github.com/lukamaga/anyintent
cd anyintent
pip install -e ".[test]"
python -m pytest
```

On Linux, run `pip install torch --index-url https://download.pytorch.org/whl/cpu` before `pip install -e`. PyTorch from PyPI brings the large CUDA packages, and the tests need only the CPU build. The CI does the same.

Tests that need a package you did not install are skipped. To run those tests too, as the CI does:

```bash
pip install -e ".[test,server,mcp,onnx]"
pip install pandas pyarrow
python -m pytest -q -rs
python tools/server_smoke.py --out smoke_out
```

- The HTTP and MCP server tests need the `server` and `mcp` extras, and the ONNX tests need the `onnx` extra. The `mcp` extra needs Python 3.10 or later.
- The release check tool's tests (`tests/test_release_check.py`) need pandas and pyarrow.
- `-rs` lists each skipped test with its reason.
- `tools/server_smoke.py` starts the HTTP and MCP servers on a tiny random model and checks their answers against the library, with no network beyond `127.0.0.1`. Without the `mcp` extra, add `--skip-mcp`.

On Windows, use the PowerShell commands in [docs/windows.md](docs/windows.md#run-the-library-tests), which also lists the tests that can be skipped there. [docs/advanced.md](docs/advanced.md#tests-of-the-library-not-of-the-models-accuracy) describes the tests.

## What the CI checks

The workflow [.github/workflows/tests.yml](.github/workflows/tests.yml) runs on GitHub Actions for every pull request, every push to `main` and every version tag. A run for a pull request from a fork can wait until a maintainer approves it. The workflow installs the library with its extras and runs the tests and the server smoke on:

- Linux: Ubuntu, Python 3.9, 3.12 and 3.13 (the Python 3.9 job runs without the `mcp` extra),
- macOS: Python 3.12,
- Windows: Python 3.10, 3.12 and 3.13, and once more with Python 3.12 on a checkout with `core.autocrlf true`.

It downloads no model and uses no secret. Please make sure that it passes on your pull request.

## Changes to how the library answers

The code that computes the answers is everything in `src/anyintent` apart from the HTTP and MCP servers. It is frozen for this release. The calibration of `check` and `scores` was fitted on DEV, the project's development messages, with this code before the release removed its support for an older file layout; that removal changed no computation, as the protocol records. The tests check that its predictions and raw scores match the reference implementation of the models.

So any change to how the library classifies or scores, however small, can only be released in a new version of the library. If your pull request changes a prediction or a score, please say so in its description.

## License of contributions

The library is under the Apache License 2.0 ([LICENSE](LICENSE) and [NOTICE](NOTICE)). Contributions are accepted only under the same license, without any additional terms (section 5 of the license). Submit only work that you have the right to license this way.

## The R2Intent models

The R2Intent models are not in this repository, and they are not open for contributions. They are published on Hugging Face under CC BY-NC 4.0, non-commercial use only (see [License](README.md#license) in the README). Feedback on a model is welcome by e-mail to lukashcode@gmail.com.

## Questions

For questions, see [Contact](README.md#contact) in the README.
