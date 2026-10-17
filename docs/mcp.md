# MCP server

[Back to the README](../README.md)

The MCP server serves the library to agents and editors that speak the Model Context Protocol (extra `mcp`: the official MCP Python SDK 1.x). Like the HTTP server ([server.md](server.md)), it loads one model at start, only calls the library and runs on your machine.

```bash
pip install "anyintent[mcp] @ git+https://github.com/lukamaga/anyintent@v0.1.0"
```

The MCP extra installs `mcp>=1.8,<2`. mcp 2 renamed the server class this server uses and changed other APIs, and it is not supported yet; if another package brought in mcp 2, `anyintent-mcp` stops at the start with a message saying so, and `pip install "mcp<2"` fixes it. Every mcp 1.x release needs Python 3.10 or later, so the MCP server does too.

The model is gated (see the [Quick start](../README.md#quick-start) of the README): accept its terms on its Hugging Face page and run `hf auth login` once, or give `HF_TOKEN` in the client configuration (see [Client configuration](#client-configuration)). The first start downloads the model, which can take minutes; to do that beforehand, run `hf download lukashm/r2intent-v1` (add `--revision` with the commit you pin).

## Start

```bash
anyintent-mcp --model lukashm/r2intent-v1 --lists-dir ~/.anyintent/lists
```

It speaks MCP over stdio (the default). The model loads in the background after the start (with its calibration file, as for the HTTP server); a tool call waits for it for up to `--load-timeout` seconds (default 45), then answers that the model is still loading. `--label-text-mode` and `--doubt-margin` work as for the HTTP server. `python -m anyintent.mcp_server` works as well.

`--transport streamable-http` or `--transport sse` serve it over HTTP on `127.0.0.1:8001`, without authentication; clients connect to `http://127.0.0.1:8001/mcp` (streamable-http) or `http://127.0.0.1:8001/sse` (sse). These transports accept only requests addressed to `127.0.0.1` or `localhost`; another `--host` needs `--allow-remote`, and then anyone who can reach the port can use the model and the saved lists.

## Tools

- `classify_intent(message, intents or list_name)`: the best matching intent for one message, with `label`, `index`, `score`, `margin` and `runner_up` (and `doubtful` with `--doubt-margin`).
- `check_intent(message, intent)` (a beta): yes or no for one intent, with a probability, as `POST /check` answers it for one message (`intent`, `yes`, `probability`, `score`, `threshold`; the threshold is 0.5).
- `score_intents(message, intents or list_name)`: a value from 0 to 1 for each intent the model read with the message, as `POST /scores` answers it for one message (`label` and `intents`, highest value first).
- `save_intent_list(name, intents)`, `get_intent_list(name)`, `list_intent_lists()`, `delete_intent_list(name)`: named intent lists, so that an agent can pass `list_name` instead of the whole list.

The tool descriptions tell the agent what the answers mean (for `score_intents`: an intent not listed was not read, and the part missing to reach 1 is not a "none of these intents" answer), that the numbers are calibrated only for intents written with a description, and that `check_intent` is a beta. Both calibrated tools are offered when the installed library has `AnyIntent.check` and `AnyIntent.scores`; on a model whose release does not ship the calibration of a function, a call returns an MCP error result that says so (`classify_intent` still works).

## Client configuration

An MCP client that uses a JSON configuration with an `mcpServers` block takes an entry like this one:

```json
{
  "mcpServers": {
    "anyintent": {
      "command": "/full/path/to/anyintent-mcp",
      "args": ["--model", "lukashm/r2intent-v1", "--lists-dir", "/full/path/to/intent-lists"]
    }
  }
}
```

Use the full path that `which anyintent-mcp` prints, because desktop apps often do not see the PATH of your shell. On Windows, get it in PowerShell with `(Get-Command anyintent-mcp).Source` and double every backslash in the JSON (see [windows.md](windows.md#the-mcp-server)). The server uses your Hugging Face login (`hf auth login`), so the configuration needs no token. A token can be given with `"env": {"HF_TOKEN": "..."}` next to `"args"`, but then it is stored in that configuration file. Clients built on the official MCP SDKs pass the server only a few system variables, so other variables, such as `HF_HUB_CACHE`, go in the same `"env"` block.

## Privacy

The privacy notes of both servers, including the MCP server's log on stderr, are in [server.md](server.md#privacy).
