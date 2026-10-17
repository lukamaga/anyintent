# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Report a vulnerability

Please do not report a vulnerability in a public issue, pull request or discussion. Instead, send an e-mail to lukashcode@gmail.com with "AnyIntent security" in the subject.

Please include:

- the library version, and the commit if you run it from a clone,
- your Python version and your operating system,
- the part that is affected (the library, the HTTP server or the MCP server) and, for a server, the options it was started with (leave out the API key and any token),
- what the problem is and what an attacker could do with it,
- the shortest steps to reproduce it,
- a fix or an idea for one, if you have it.

Do not send real customer messages, tokens or passwords. Use made-up ones.

Please keep the details private until a fix is released, or until we agree on a date to publish them.

## Scope

In scope: the code in this repository, that is the library, its HTTP server (`anyintent-serve`) and its MCP server (`anyintent-mcp`).

Out of scope: the R2Intent models, which are not in this repository, and the packages that the library uses, such as PyTorch, transformers, FastAPI and the MCP SDK. Please report a problem in those to their own projects. A wrong answer from a model is not a vulnerability.

## Safe use of the servers

- By default, the HTTP server listens on `127.0.0.1`, so only this machine can reach it.
- To serve other machines, start it with `--host 0.0.0.0` and an API key in the environment variable `ANYINTENT_API_KEY`. Without a key, by default it accepts only requests addressed to `127.0.0.1` or `localhost`.
- The HTTP server speaks plain HTTP, without TLS. Across a network, the API key and the messages are not encrypted.
- The MCP server speaks over stdio by default. Its HTTP transports have no authentication. With `--allow-remote`, anyone who can reach the port can use the model and the saved lists.

[docs/server.md](docs/server.md#label-texts-api-key-and-cors) and [docs/mcp.md](docs/mcp.md#start) have the details.

## What to expect

We try to answer every report. There is no fixed reply time and no bug bounty.
