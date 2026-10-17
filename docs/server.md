# HTTP server

[Back to the README](../README.md)

Two optional ways to use the library from other programs: an HTTP server (extra `server`: FastAPI, uvicorn and pydantic) and an MCP server for agents and editors that speak the Model Context Protocol (extra `mcp`: the official MCP Python SDK 1.x, which needs Python 3.10 or later). Both load one model at start, only call the library and run on your machine. This page covers the HTTP server; the MCP server is in [mcp.md](mcp.md).

```bash
pip install "anyintent[server] @ git+https://github.com/lukamaga/anyintent@v0.1.0"
pip install "anyintent[mcp] @ git+https://github.com/lukamaga/anyintent@v0.1.0"
```

The model is gated (see the [Quick start](../README.md#quick-start) of the README): accept its terms on its Hugging Face page and run `hf auth login` once, or set `HF_TOKEN`. The first start downloads the model, which can take minutes; to do that beforehand, run `hf download lukashm/r2intent-v1` (add `--revision` with the commit you pin).

The shell examples on this page are for bash. On Windows, [windows.md](windows.md#the-http-server) shows how to send a request (POST /classify) from PowerShell or cmd; for the other endpoints, change the method, the address and the body (if any) in the same way.

No speed was measured for the servers. Docker files for the HTTP server (CPU, and CUDA on an NVIDIA runtime base) are provided in `docker/` (see [docker/README.md](../docker/README.md)), but they were not built or tested for this release.

## Start

```bash
anyintent-serve --model lukashm/r2intent-v1 --lists-dir ./intent-lists
```

Add `--revision` with a commit hash to pin the model, and `--warm-up` to run a few dummy messages before serving. It listens on `127.0.0.1:8000`, this machine only (`--host 0.0.0.0` listens on every interface). The OpenAPI page with every request and response schema is at `http://127.0.0.1:8000/docs`. `python -m anyintent.server` works as well.

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | state (`ready`), model id, revision, settings, limits, and `supported`: `classify`, and `check` and `scores` when the library has them and the model's calibration file names them (a call can still be refused with 422, for example with a calibration file of another model or a changed model folder) |
| `POST /classify` | the matching intent for each message |
| `POST /check` | yes or no for one intent, with a probability, for each message (`AnyIntent.check`, a beta) |
| `POST /scores` | a value from 0 to 1 for each intent the model read with the message, for each message (`AnyIntent.scores`) |
| `GET /lists` | the saved intent lists |
| `PUT`, `GET`, `DELETE /lists/{name}` | save (or replace), read, delete a named intent list |

## Classify

A request names the messages (`text` for one, `texts` for a list) and the intents: `intents`, a list of names or of `{"name", "description"}` objects, or `list`, the name of a saved list. `k` is optional, as in `classify`:

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

The answer has one result per message, in order: the intent's name, its position in the list, its score and margin (as in `Prediction`), and the runner-up, the second best intent:

```text
{"results": [{"label": "<intent name>", "index": <position>, "score": <number>, "margin": <number or null>, "runner_up": <intent name or null>}, ...]}
```

Started with `--doubt-margin X` (a threshold chosen on your own labeled messages, see [Test it on your own messages](../README.md#test-it-on-your-own-messages) in the README), the server adds `"doubtful": true` to every result whose margin is below X, so that clients and agents need not know the model's units.

## Named lists

Named lists save sending the intents with every request:

```bash
curl -s -X PUT http://127.0.0.1:8000/lists/support -H "Content-Type: application/json" -d '{"intents": [
  {"name": "lost card", "description": "The customer reports that a bank card was lost or stolen."},
  {"name": "refund status", "description": "The customer asks when a refund will arrive."},
  "close account"
]}'
curl -s http://127.0.0.1:8000/classify -H "Content-Type: application/json" \
     -d '{"text": "Where is my refund?", "list": "support"}'
```

From Python, with the standard library only:

```python
import json
import urllib.request

req = urllib.request.Request("http://127.0.0.1:8000/classify",
                             data=json.dumps({"text": "Where is my refund?", "list": "support"}).encode("utf-8"),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as r:  # an HTTP error raises urllib.error.HTTPError
    print(json.load(r)["results"][0]["label"])
```

A list name has 1 to 64 characters: lowercase letters, digits, `_` and `-`, starting with a letter or a digit. With `--lists-dir`, every list is a JSON file in that folder, and the server reads the lists again at start. On Linux and macOS, the server creates the folder and its files so that only your account can read them. On Windows, who can read them depends on the Python version and on whether the folder already existed (see [windows.md](windows.md#what-is-different-on-windows)). Without `--lists-dir`, lists live in memory until the server stops.

## Check and scores

`/check` (a beta) takes the messages (`text` or `texts`), one `intent` (a name, a `{"name", "description"}` object or a `[name, description]` pair) and an optional `threshold` (default 0.5, the only threshold measured):

```bash
curl -s http://127.0.0.1:8000/check -H "Content-Type: application/json" -d '{
  "texts": ["I cannot find my card anywhere", "Where is my refund?"],
  "intent": {"name": "lost card", "description": "The customer reports that a bank card was lost or stolen."}
}'
```

```text
{"results": [{"intent": "lost card", "yes": <true or false>, "probability": <0 to 1>, "score": <number>, "threshold": 0.5}, ...]}
```

`yes` means `probability` >= `threshold`, and `probability` and `score` mean what they mean for `AnyIntent.check`. `/scores` takes the messages and the intents (`intents` or `list`, as `/classify`):

```bash
curl -s http://127.0.0.1:8000/scores -H "Content-Type: application/json" \
     -d '{"text": "Where is my refund?", "list": "support"}'
```

```text
{"results": [{"label": "<intent name>", "intents": [{"intent": "<intent name>", "value": <0 to 1>}, ...]}]}
```

`intents` lists the intents the model read with the message (at most 5; every intent when there are 5 or fewer), from the highest value to the lowest. An intent not listed was not read: its value is 0, which means not read, not impossible, and it can still be the right intent. With more than 5 intents the listed values can sum to less than 1; the part missing to reach 1 is not a "none of these intents" answer. `label` is the intent with the largest value, the one `/classify` returns without `k`. Both `/check` and `/scores` were calibrated on intents written with a description: give one.

Both answer only with a model release that ships the calibration of the function. On any other model they answer 409 with the code `calibration_missing` and a message that names the function; `/classify` works as before. The servers read the calibration file when the model loads, not at the first request: for a model from the Hugging Face cache without the file, this asks the Hub for it once at start (see Loading in [calibration.md](calibration.md#notes-for-check-and-scores)), unless `HF_HUB_OFFLINE=1` is set or the file was downloaded beforehand. A server with a library version without `check` or `scores` answers 501. Other refusals of the library are a 422 with the library's message; a local model folder in that message is shown by its name only (the server log has the full message).

## Errors

Errors come as `{"error": {"code": "...", "message": "..."}}`: 400 (a request addressed to a host name the server does not accept), 401 (API key), 404 (unknown list), 409 (`calibration_missing`), 413 (a limit), 422 (an invalid request, or a request the library refuses), 501 (a function the installed library does not have), 503 (model not ready, or `busy`: too many requests in progress), 500 (an unexpected error; no stack trace is sent).

## Label texts, API key and CORS

Label texts: the server passes your intents to `set_labels` as they are. `--label-text-mode` (for a model released with another description layout, see [Writing labels](../README.md#writing-labels) in the README) applies to `/classify` and `/scores`; `/scores` refuses a mode other than the one its calibration was made with (422), and `/check` always uses the model's own mode.

API key: start the server with the environment variable `ANYINTENT_API_KEY` set (or `--api-key`, which other accounts on the machine can see in the process list); clients then send `Authorization: Bearer <key>`. `GET /health` answers without the key, with `status` and `ready` only. The server sends no CORS headers, so browsers do not let other sites read its answers, unless a site is allowed with `--cors-origin https://your.site`. Without an API key, and always on `127.0.0.1`, it accepts only requests addressed to `127.0.0.1` or `localhost` (a guard against DNS rebinding from web pages), wherever it listens; `--allowed-host` names more (`'*'` accepts any), and a key on a non-loopback address accepts any.

## Limits

Limits (options of `anyintent-serve`):

| Option | Default | Limit |
|---|---|---|
| `--max-batch` | 64 | messages per request |
| `--max-text-chars` | 2000 | characters per message (the model reads only the first part of a long message, see [Writing labels](../README.md#writing-labels) in the README) |
| `--max-intents` | 200 | intents per request or per saved list |
| `--max-concurrency` | 64 | requests in progress at once; more get 503 `busy` (`/health` is not counted) |

Intent names are limited to 200 characters, descriptions to 2000, saved lists to 1000 and to 50 MB together, and the request body to the size of the largest valid request. One server runs one model in one process and answers requests one after another (the library serializes its calls); for parallel work run several servers.

## Privacy

The servers and the model run on your machine. Message texts are not stored, and they are not logged unless the server is started with `--log-texts` (the log has counts and times). Saved intent lists are stored in the `--lists-dir` folder. The `/docs` page loads the Swagger UI files from a public CDN into your browser. At start the library contacts Hugging Face to download or check the model files; after the first download, `HF_HUB_OFFLINE=1` starts it without network access. The MCP server logs to stderr, which an MCP client may keep in its own log files. An unexpected error reaches the client as a short message; its details go to the server log. If the model fails to load, the error that `/health` shows can include the path of the model folder.
