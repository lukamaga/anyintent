# Windows

[Back to the README](../README.md)

These are the steps for Windows 10 and 11 in PowerShell. Where the Command Prompt (cmd) needs another form, the page gives it. To open PowerShell, type PowerShell in the Start menu. Most commands of this page ran on GitHub Actions runners with Windows Server 2025, in Windows PowerShell 5.1, PowerShell 7 and cmd. The Quick start, the HTTP server and the MCP server ran there with both released models. Reports of what works and what does not are welcome (see [Contact](../README.md#contact)).

## What you need

- 64-bit x64 Python 3.12 or 3.13. x64 is the build for Intel and AMD processors. PyTorch on PyPI has no builds for 32-bit Python and none for Windows on Arm. Install Python with the Python install manager from python.org/downloads or the Microsoft Store (`pymanager install 3.13`; `py install 3.13` works when no older Python launcher is installed), or with the Windows installer from python.org. The commands on this page use 3.13. If you installed 3.12, type 3.12 instead. Then check that this prints `win-amd64` (a 32-bit Python prints `win32`, and an Arm one `win-arm64`):

  ```powershell
  py -V:3.13 -c "import sys, sysconfig; print(sys.version, sysconfig.get_platform())"
  ```

- Git for Windows, which pip needs for the `git+https` install. Install it:

  ```powershell
  winget install --id Git.Git -e --source winget
  ```

  Then open a new window, so that it finds git, and check it:

  ```powershell
  git --version
  ```

## A virtual environment

A virtual environment keeps the library and the packages it needs apart from the rest of your system. Create one in a work folder, here `C:\work`:

```powershell
mkdir C:\work
cd C:\work
py -V:3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Both the Python install manager and the older `py` launcher of the python.org installer accept `py -V:3.13`; `py -3.13` is the short form.

Windows PowerShell 5.1, the one that comes with Windows, does not run scripts by default, so it refuses `Activate.ps1`. To allow scripts in this window only (nothing is saved), run this and then `.\.venv\Scripts\Activate.ps1` again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Or use cmd instead:

```bat
cd C:\work
.venv\Scripts\activate.bat
```

Every new window starts without the environment. In each one, run `cd C:\work` and the activate line again. In Windows PowerShell 5.1, run the `Set-ExecutionPolicy` line first, since it holds for one window only.

The commands below also work in cmd, except a few PowerShell lines in [Run the Quick start](#run-the-quick-start), [Log in to Hugging Face](#log-in-to-hugging-face), [The HTTP server](#the-http-server) and [The MCP server](#the-mcp-server). There the text gives the cmd form.

## NVIDIA GPU (optional)

PyTorch from PyPI is CPU only on Windows. For an NVIDIA GPU, install PyTorch from the pytorch.org index before the library. The library needs PyTorch 2.6 or later, and the build you install there meets that, so pip keeps it when it installs the library. The CUDA 12.6 build below needs NVIDIA driver 527.41 or newer on Windows (`nvidia-smi` shows the driver version):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

With a GPU of the Turing generation or later (compute capability 7.5 or higher, for example the GTX 16 and RTX 20 series and newer) and an NVIDIA driver new enough for CUDA 13.2 (NVIDIA's CUDA release notes give the minimum), you can use `https://download.pytorch.org/whl/cu132` instead. Older GPUs need cu126. The second line should print a version that ends in `+cu126` (or `+cu132`) and `True`.

- If that line prints `False` later, look at the version. If it has no `+cu`, the CPU build from PyPI has replaced yours, for example after a `pip install -U`: run `pip uninstall -y torch`, then the `pip install torch --index-url ...` line again. If it has `+cu`, check that the NVIDIA driver is new enough for that build.
- On a GPU older than Turing, use cu126 even if the check prints `True` with cu132: the CUDA 13 builds of PyTorch have no code for those GPUs, so the first model call fails.
- If `import torch` fails with `DLL load failed`, install the Microsoft Visual C++ Redistributable, as the PyTorch notes for Windows advise.

## Install the library

```powershell
pip install "git+https://github.com/lukamaga/anyintent@v0.1.0"
python -c "import anyintent; print(anyintent.__version__)"
```

The second line prints 0.1.0.

Double quotes work in PowerShell and in cmd; single quotes do not work in cmd. The extras install the same way, for example `pip install "anyintent[server,mcp] @ git+https://github.com/lukamaga/anyintent@v0.1.0"`.

## Log in to Hugging Face

Request access to the model first (step 2 of the [Quick start](../README.md#quick-start)). Then log in once with an access token from https://huggingface.co/settings/tokens (read access is enough):

```powershell
hf auth login
```

The login is stored in `%USERPROFILE%\.cache\huggingface\token`, and the models go to `%USERPROFILE%\.cache\huggingface\hub`. To keep the models elsewhere, set `HF_HUB_CACHE` before you start Python. For this window only:

```powershell
$env:HF_HUB_CACHE = "D:\hf\hub"
```

In cmd, `set HF_HUB_CACHE=D:\hf\hub` does the same. `setx HF_HUB_CACHE "D:\hf\hub"` keeps it for every window you open later, in both shells, but not for the current one.

## Run the Quick start

Save the code of step 4 of the [Quick start](../README.md#quick-start) as `try_anyintent.py` in `C:\work`, in UTF-8, and run it there with the environment active:

```powershell
python try_anyintent.py
```

Python reads its source files as UTF-8, so the Lithuanian and Polish letters in the file are read correctly. With transformers 5, loading a model can print a `LOAD REPORT` that lists the adapter's keys as `UNEXPECTED`. This is expected: the library reads those keys itself (see [Model folders](advanced.md#model-folders)).

To run the script of [Test it on your own messages](../README.md#test-it-on-your-own-messages), save it as `check_messages.py` in the same folder as `messages.csv` and run `python check_messages.py`. In Excel, save `messages.csv` as CSV UTF-8: the script accepts the byte order mark (BOM) that Excel writes. With Polish, Lithuanian and many other regional settings, Excel separates the columns with semicolons: then use `csv.DictReader(f, delimiter=";")`, as the README says. The wrong answers that the script prints include your messages, so an output file such as `out.txt` below contains them too.

In a console window every letter shows as it is. But when the output goes to a file or a pipe, Python 3.14 and earlier write it in the system's code page (the ANSI code page) and stop with `UnicodeEncodeError` at a letter that this code page lacks. In cmd, this line writes UTF-8 instead:

```bat
python -X utf8 check_messages.py > out.txt
```

Run that line in cmd: PowerShell before 7.4, Windows PowerShell 5.1 among them, does not keep the bytes of a program's output when `>` writes it to a file, so the Lithuanian and Polish letters can change, with no error. In Windows PowerShell 5.1, `>` also writes UTF-16. In PowerShell, pass the line to cmd:

```powershell
cmd /c "python -X utf8 check_messages.py > out.txt"
```

In PowerShell 7.4 and later, the plain `>` line works too.

## The HTTP server

```powershell
pip install "anyintent[server] @ git+https://github.com/lukamaga/anyintent@v0.1.0"
anyintent-serve --model lukashm/r2intent-v1 --lists-dir .\intent-lists
```

The first start downloads the model, which can take minutes. Leave this window open while you use the server.

Open a second PowerShell window and send the request as UTF-8 bytes. The lines below are written for both Windows PowerShell 5.1 and PowerShell 7:

```powershell
$body = @{
  texts   = @("I cannot find my card anywhere", "Chcę zamknąć konto")
  intents = @(
    @{ name = "lost card";     description = "The customer reports that a bank card was lost or stolen." },
    @{ name = "refund status"; description = "The customer asks when a refund will arrive." },
    "close account"
  )
}
$json  = $body | ConvertTo-Json -Depth 5
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
$r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/classify" -ContentType "application/json; charset=utf-8" -Body $bytes
$r.results | Format-Table label, index, score, margin, runner_up
```

Without a charset in -ContentType, Windows PowerShell 5.1 and PowerShell 7 before 7.4 send a string body as ISO-8859-1, so Polish and Lithuanian letters arrive changed (as question marks or as letters without their marks), with no error. The lines above give the charset and also send UTF-8 bytes, which keep the request right even if the charset is left out.

- If you keep these lines in a `.ps1` file for Windows PowerShell 5.1, save the file as UTF-8 with BOM. Without the BOM, 5.1 reads the script in the system's code page.
- Windows PowerShell 5.1 does not read the answer as UTF-8, so intent names with letters outside the English alphabet look wrong in `$r`. In 5.1, read the answer this way instead:

  ```powershell
  $w = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "http://127.0.0.1:8000/classify" -ContentType "application/json; charset=utf-8" -Body $bytes
  [System.Text.Encoding]::UTF8.GetString($w.RawContentStream.ToArray()) | ConvertFrom-Json
  ```

With curl, write the body to a UTF-8 file and send the file. Type `curl.exe`, not `curl`: in Windows PowerShell 5.1, `curl` is another name for `Invoke-WebRequest`. Keep `"@body.json"` in quotes, since PowerShell treats a leading `@` as its own syntax. The first line below writes `$json`, made by the block with `$body` above, to `body.json`.

```powershell
$json | Set-Content -Path body.json -Encoding UTF8      # in 5.1 this adds a BOM, which the server accepts
curl.exe -s http://127.0.0.1:8000/classify -H "Content-Type: application/json" --data-binary "@body.json"
```

In cmd, `curl` is `curl.exe` (part of Windows 10 version 1803 and later), and it sends the same `body.json`. To make that file without PowerShell, save the JSON body of the curl example in [Run it as a local service](../README.md#run-it-as-a-local-service) in Notepad as `body.json`, with Encoding set to UTF-8 and Save as type set to All files. Then send it:

```bat
curl -s http://127.0.0.1:8000/classify -H "Content-Type: application/json" --data-binary "@body.json"
```

The answer has one result per message with `label`, `index`, `score`, `margin` and `runner_up` (and `doubtful` with `--doubt-margin`), as described in [server.md](server.md#classify). The page `http://127.0.0.1:8000/docs` in a browser shows every request and response schema.

## The MCP server

```powershell
pip install "anyintent[mcp] @ git+https://github.com/lukamaga/anyintent@v0.1.0"
(Get-Command anyintent-mcp).Source
```

The last line prints the full path of `anyintent-mcp.exe`. `where.exe anyintent-mcp` also works in PowerShell; type `where.exe`, since `where` alone means `Where-Object` there. In cmd, use `where anyintent-mcp`. Put that path into the client configuration, with every backslash doubled, as JSON requires:

```json
{
  "mcpServers": {
    "anyintent": {
      "command": "C:\\work\\.venv\\Scripts\\anyintent-mcp.exe",
      "args": ["--model", "lukashm/r2intent-v1", "--lists-dir", "C:\\work\\intent-lists"]
    }
  }
}
```

- For example, Claude Desktop reads this configuration from `%APPDATA%\Claude\claude_desktop_config.json` and writes the server's log to `%APPDATA%\Claude\logs\mcp-server-anyintent.log`.
- The server uses your Hugging Face login (`hf auth login`). Clients built on the official MCP SDKs pass the server only a few system variables, so other variables, such as `HF_HUB_CACHE`, may not reach it from your shell or from `setx`. Give them in an `"env"` block next to `"args"`, for example `"env": {"HF_HUB_CACHE": "D:\\hf\\hub"}`.
- On GitHub Actions runners with Windows Server 2025, the client of the MCP Python SDK started the server from this configuration with both released models, and all checks of its tools passed. That SDK has open issues with stdio on Windows, so if a client waits forever when it starts the server, please report it (see [Contact](../README.md#contact)).

## What is different on Windows

- `compile=True` on an NVIDIA GPU needs Triton, which has no official Windows build, so PyTorch raises `TritonMissing` at the first `set_labels`, `classify` or `warm_up` call. Leave `compile` off (the default). On the CPU, `compile=True` is ignored with a warning.
- Without Developer Mode or administrator rights, Windows does not let huggingface_hub create symbolic links (symlinks) in its cache, and huggingface_hub prints a warning about that. The warning is harmless: the files are stored without links, which takes more disk space only when you download several revisions of one model. To stop the warning, turn on Developer Mode, which lets huggingface_hub create the links, or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, which only hides the warning. Without the links, the first `check` or `scores` call of a process reads the whole model file (1.19 GB) once to compute its sha256 (the time was not measured).
- Who can read the lists folder: on Linux and macOS, the server creates the `--lists-dir` folder and its files so that only your account can read them (modes 0700 and 0600). Windows ignores these modes, apart from read-only. With Python 3.12.4 or later, a folder that the server creates is limited to your account and the administrators. Otherwise, and for a folder that already exists, the folder's own access list decides. To see it, run `icacls <folder>`.
- JSON files you edit by hand, such as an `anyintent.json` next to a model, must be saved as UTF-8 without a BOM, or loading the model fails. In Windows PowerShell 5.1, `Set-Content -Encoding UTF8` adds a BOM (fine for `body.json` above, which the server accepts, but not for these files), and `>` writes UTF-16. The server's own list files are read with or without a BOM.

## WSL 2

WSL 2 runs Linux inside Windows. There, follow the README as on Linux, with its bash examples.

- Run `wsl --install` in a PowerShell window opened as administrator. It needs Windows 10 version 2004 (build 19041) or later, or Windows 11, and installs Ubuntu by default.
- In Ubuntu, first make a virtual environment: run `sudo apt update` and `sudo apt install python3-venv`, then `python3 -m venv ~/.venv` and `source ~/.venv/bin/activate`. After that, follow the README.
- For an NVIDIA GPU, install the NVIDIA driver on Windows only, not inside Linux. PyTorch from PyPI includes CUDA on Linux.
- Keep the model cache and your files in the Linux home folder (`/home/<name>`), not under `/mnt/c`, for speed.

## Run the library tests

The [tests of the library](advanced.md#tests-of-the-library-not-of-the-models-accuracy) build tiny random models and need no network. They run from a clone of the repository:

```powershell
git clone https://github.com/lukamaga/anyintent
cd anyintent
pip install -e ".[test,server,mcp]"
python -m pytest -q -rs
python tools\server_smoke.py --out smoke_out
```

If `server_smoke.py` hangs at the MCP part, press Ctrl+C, run it again with `--skip-mcp`, and please report it.

The same pytest and `server_smoke.py` commands passed on GitHub Actions runners with Windows Server 2025 and Python 3.12 and 3.13. They ran in Git Bash, from a checkout of the library's code, with onnxruntime, pandas and pyarrow also installed. `-rs` lists each skip as a SKIPPED line, with its file, line and reason. On Windows, these can appear:

- In `test_api.py` and `test_calibration.py`, the two tests that overwrite the model file of a loaded model: "Windows refuses to replace a memory-mapped file". The library keeps `model.safetensors` mapped after loading. The runners skipped these two.
- In `test_calibration.py`, the test that reads the sha256 of the weights from the name of their file in the cache, when your account may not create symlinks (no Developer Mode, no administrator rights): "os.symlink is not permitted here".
- Without onnxruntime or mcp installed, their tests are skipped, as on every system. The commands above install mcp but not onnxruntime.
- The commands above do not install pandas and pyarrow, so all of `test_release_check.py`, the tests of the release check tool, is skipped. With both installed and Python 3.13 or later, some of its tests are skipped instead: "a test record names the released folder by a Linux path". The release check tool expects Linux paths, and from Python 3.13 on, Windows does not treat a path that starts with one slash as absolute. The runner with Python 3.13 skipped these.

On Windows, `test_service.py` and the server smoke do not check the file mode of a saved list; the smoke reports that check as `SKIP`. The repository's `.gitattributes` keeps LF line endings and never converts the byte-exact files under `tests/reference`, so they match their sha256 on Windows too: the tests passed on a runner with `core.autocrlf true`.
