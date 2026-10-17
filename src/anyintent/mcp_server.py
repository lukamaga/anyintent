"""MCP server for AnyIntent (the 'mcp' extra: the official Python MCP SDK 1.x, FastMCP; mcp 2 is not supported
yet), stdio transport by default.

    anyintent-mcp --model lukashm/r2intent-v1 --lists-dir ~/.anyintent/lists

Tools: classify_intent, save_intent_list, get_intent_list, list_intent_lists, delete_intent_list; check_intent and
score_intents when the installed library has AnyIntent.check and AnyIntent.scores (registered before the model loads;
on a model whose release does not ship their calibration a call returns an error result saying so, and classify_intent
still works). The model loads in a background thread at start, so the client's handshake is not held up; a tool call
that needs the model waits for it (up to --load-timeout seconds). Message texts are never logged unless --log-texts. The
log goes to stderr (stdout carries the MCP protocol). An unexpected error reaches the client as a short message; its
details go to the log.

The sse and streamable-http transports listen on 127.0.0.1 and accept only requests addressed to 127.0.0.1 or
localhost (a guard against DNS rebinding); another address needs --allow-remote (they have no authentication).

The tool functions are built by make_tools(service) as plain async functions, so they can be called and tested
without the MCP package; build_server(service) registers them on a FastMCP server.
"""
import argparse
import asyncio
import inspect
import logging
import os
import re
import sys
from typing import List, Optional, Union

from typing_extensions import NotRequired, TypedDict

from .service import (LABEL_TEXT_MODES, LOOPBACK_NAMES, HostGuard, Limits, Service, ServiceError, check_doubt_margin,
                      is_loopback)

INSTALL_HINT = ('the MCP server needs the mcp extra (mcp 1.x): pip install '
                '"anyintent[mcp] @ git+https://github.com/lukamaga/anyintent@v0.1.0"')
# the MCP SDK versions this server is written for (the mcp extra pins them); mcp 2 renamed FastMCP, changed other APIs
SUPPORTED_MCP = 'this MCP server needs mcp 1.x (1.8 or later); mcp 2 is not supported yet: pip install "mcp<2"'

log = logging.getLogger("anyintent.server")

INSTRUCTIONS = (
    "AnyIntent routes a customer message to one of the intents the user defines; no training examples of those intents "
    "are needed. Call classify_intent with the message and the intents (names, better with a one or two sentence "
    "description), or with the name of an intent list saved earlier with save_intent_list (list_intent_lists shows the "
    "saved ones). The answer has the chosen label, its score, its margin and the runner-up intent; when it also has "
    "`doubtful`, follow it. check_intent (yes or no for one intent, with a probability; a beta) and score_intents (a "
    "value for each intent the model read with the message) give calibrated numbers only with a model release that "
    "ships their calibration; otherwise they return an error saying so, and classify_intent still works. The model "
    "runs on the user's machine.")

CLASSIFY_DESCRIPTION = (
    "Route one customer message to the best matching intent from the user's own list of intents (no training examples "
    "of those intents are needed). Use it to sort or triage a message, for example a support ticket or a chat request. "
    "Give either `intents` (a list of names, or of {name, description} objects; a one or two sentence description "
    "usually helps) or `list_name` (a list saved earlier with save_intent_list; list_intent_lists shows them), not "
    "both. Returns `label` (the chosen intent's name), `index` (its position in the list), `score`, `margin` (the "
    "score of the best intent minus that of the second best; a small margin means the answer is doubtful) and "
    "`runner_up` (the second-best intent). If the result has `doubtful`, follow it: when it is true, treat the answer "
    "as uncertain and ask the user, for example whether the message is about `label` or `runner_up`. Without "
    "`doubtful`, score and margin are raw model units with no fixed scale, not probabilities: do not judge a single "
    "answer from them alone; compare margins only between calls with the same intents. The message is not stored.")

SAVE_DESCRIPTION = (
    "Save a named list of intents, so that later classify_intent calls can pass `list_name` instead of the whole list. "
    "A list with the same name is replaced. `name`: 1 to 64 characters, lowercase letters, digits, '_' and '-'. "
    "`intents`: names, or {name, description} objects; names must be unique. Returns whether the list is kept on disk "
    "(the server was started with --lists-dir) or only in memory until the server stops.")

GET_DESCRIPTION = "Return the intents of a saved intent list (names and descriptions, in their order)."

LIST_DESCRIPTION = ("List the saved intent lists (names and how many intents each holds). Call it before passing "
                    "`list_name` when the name is not known.")

DELETE_DESCRIPTION = "Delete a saved intent list. Call it only when the user asks for that list to be deleted."

CHECK_DESCRIPTION = (
    "Ask whether one customer message expresses one given intent, with a calibrated probability. Use it when there is "
    "a single candidate intent to confirm; to choose among several intents use classify_intent. `intent`: a name, or a "
    "{name, description} object; give the description (one or two sentences): the probabilities were calibrated only "
    "on intents written with a description. Returns `yes` (true when `probability` is at least 0.5), `probability` "
    "(the chance that the message expresses this intent, taking a match and a non-match as equally likely before "
    "reading the message; for an intent that is rare in the user's traffic the real chance is lower), `score` (the "
    "model's raw pair score, not a probability), `intent` and `threshold`. The answer does not depend on any other "
    "intent. This tool is a beta: check shipped in R2Intent v1 after it missed one pre-registered calibration limit on "
    "the DEV data, so treat the probability as approximate. Works only with a model release that ships the calibration "
    "of check: otherwise the call returns an error saying so (then use classify_intent). The message is not stored.")

SCORES_DESCRIPTION = (
    "For one customer message, give a value from 0 to 1 (the chance that it is the right intent) to each intent the "
    "model reads with it: at most 5 intents of the list, every intent when there are 5 or fewer. Use it when you need "
    "to weigh several intents rather than take only the best one; to route a message use classify_intent. Give "
    "`intents` or `list_name`, as for classify_intent, with a description for each intent: the values were calibrated "
    "only on intents written with a description. Returns `intents`, which lists only the intents the model read (a "
    'list of objects `{"intent": name, "value": number}`, from the highest value to the lowest), and `label` (the '
    "intent with the largest value, the first one listed: the one classify_intent returns). An intent not listed was "
    "not read: its value is 0, which means not read, not impossible; it can still be the right intent. With 5 intents "
    "or fewer every intent is listed and the values sum to 1. With more, the values can sum to less than 1, because "
    "the right intent can be one of the intents the model did not read. The part missing to reach 1 is NOT a 'none of "
    "these intents' answer: there is no such mode, and the values assume that one of the given intents is right. "
    "Works only with a model release that ships the calibration of scores: otherwise the call returns an error saying "
    "so (then use classify_intent). The message is not stored.")

_READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
_CHANGES_LISTS = {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": False}
# MCP tool annotations, given when the SDK's tool() takes them (save replaces a list of the same name)
TOOL_HINTS = {"classify_intent": _READ_ONLY, "get_intent_list": _READ_ONLY, "list_intent_lists": _READ_ONLY,
              "check_intent": _READ_ONLY, "score_intents": _READ_ONLY, "save_intent_list": _CHANGES_LISTS,
              "delete_intent_list": _CHANGES_LISTS}


class IntentSpec(TypedDict):
    name: str
    description: NotRequired[str]


def _doc(tp, text):
    """tp with a description for the tool's input schema (when pydantic is installed; it comes with the mcp extra)."""
    try:
        from pydantic import Field
        from typing_extensions import Annotated
    except ImportError:   # pragma: no cover
        return tp
    return Annotated[tp, Field(description=text)]


MESSAGE_DOC = "the customer message to classify, as written"
INTENTS_DOC = "the intents to choose from: names, or objects with name and description (or give list_name)"
LIST_NAME_DOC = "the name of an intent list saved with save_intent_list (or give intents)"
SCORE_INTENTS_DOC = ("the intents to weigh (the model reads at most 5 of them): names, or (better) objects with name "
                     "and description (or give list_name)")


INTERNAL_ERROR = "internal error; the MCP server log (stderr) has the details"


async def run_tool(service, where, fn, *args):
    """fn(*args) in a worker thread. A ServiceError (the caller's mistake, a limit, a missing list, the model not
    ready, a library refusal, a model without the calibration of check or scores) reaches the client with its clear
    message (FastMCP sends an error raised by a tool as an error result, isError true, with that text); any other error
    is logged (type and traceback frames, its message only with --log-texts) and the client gets only INTERNAL_ERROR
    (no local paths, no message text; raised outside the except block, so it carries no chained original either,
    should the SDK log it with its traceback)."""
    try:
        return await asyncio.to_thread(fn, *args)
    except ServiceError:
        raise
    except Exception as e:
        service.log_internal_error(where, e)
    raise RuntimeError(INTERNAL_ERROR)


def make_tools(service):
    """[(name, async function, description)]: the tools for this service; check_intent and score_intents only when
    the library has AnyIntent.check and AnyIntent.scores. They return one result (one message per call): check_intent
    {intent, yes, probability, score, threshold}, score_intents {label, intents: [{intent, value}]} (only the intents
    the model read, highest value first)."""

    async def classify_intent(message: _doc(str, MESSAGE_DOC),
                              intents: _doc(Optional[List[Union[str, IntentSpec]]], INTENTS_DOC) = None,
                              list_name: _doc(Optional[str], LIST_NAME_DOC) = None):
        return (await run_tool(service, "classify_intent", service.classify, message, intents, list_name))[0]

    async def save_intent_list(name: _doc(str, "the list's name: lowercase letters, digits, '_' and '-'"),
                               intents: _doc(List[Union[str, IntentSpec]], "names, or objects with name and "
                                                                           "description")):
        r = await run_tool(service, "save_intent_list", service.save_list, name, intents)
        if not r["persistent"]:
            r["note"] = ("kept in memory until the server stops; start the server with --lists-dir to keep lists on "
                         "disk")
        return r

    async def get_intent_list(name: _doc(str, "the name of a saved intent list")):
        return await run_tool(service, "get_intent_list", service.get_list, name)

    async def list_intent_lists():
        return await run_tool(service, "list_intent_lists", service.list_lists)

    async def delete_intent_list(name: _doc(str, "the name of the saved intent list to delete")):
        return await run_tool(service, "delete_intent_list", service.delete_list, name)

    async def check_intent(message: _doc(str, "the customer message, as written"),
                           intent: _doc(Union[str, IntentSpec], "the one intent to check: a name, or (better) an "
                                                                "object with name and description")):
        return (await run_tool(service, "check_intent", service.check, message, intent))[0]

    async def score_intents(message: _doc(str, "the customer message, as written"),
                            intents: _doc(Optional[List[Union[str, IntentSpec]]], SCORE_INTENTS_DOC) = None,
                            list_name: _doc(Optional[str], LIST_NAME_DOC) = None):
        return (await run_tool(service, "score_intents", service.scores, message, intents, list_name))[0]

    tools = [("classify_intent", classify_intent, CLASSIFY_DESCRIPTION),
             ("save_intent_list", save_intent_list, SAVE_DESCRIPTION),
             ("get_intent_list", get_intent_list, GET_DESCRIPTION),
             ("list_intent_lists", list_intent_lists, LIST_DESCRIPTION),
             ("delete_intent_list", delete_intent_list, DELETE_DESCRIPTION)]
    if service.has("check"):
        tools.append(("check_intent", check_intent, CHECK_DESCRIPTION))
    if service.has("scores"):
        tools.append(("score_intents", score_intents, SCORES_DESCRIPTION))
    return tools


def _takes(method, param):
    try:
        return param in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


class UnsupportedMCP(ImportError):
    """The installed MCP SDK is a version this server does not support (so far: mcp 2 and later). An ImportError, so
    that main() reports it as it reports a missing extra (a log line and exit code 2)."""


def installed_mcp_version():
    """The version of the installed mcp distribution, or None when it is not installed."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("mcp")
    except PackageNotFoundError:
        return None


def _installed():
    v = installed_mcp_version()
    return f"the installed mcp {v}" if v else "the installed mcp package"


def mcp_major(version):
    """The major version number of a version string such as '1.27.0' (None when there is none)."""
    m = re.match(r"\s*(\d+)", version or "")
    return int(m.group(1)) if m else None


def _sent(annotations):
    """Tool annotations as the SDK sends them to a client: a pydantic model serialised by alias and without the fields
    that are None (as the SDK does), a dict as it is."""
    dump = getattr(annotations, "model_dump", None)
    if callable(dump):
        return dump(by_alias=True, exclude_none=True, mode="json")
    return dict(annotations)


def _annotations(hints):
    """mcp.types.ToolAnnotations with these hints (a plain dict when the SDK has no such type). The hints use the MCP
    field names (readOnlyHint, destructiveHint, idempotentHint, openWorldHint); a ToolAnnotations that does not take
    them or does not send them back under these names (another field naming, as in a future SDK) raises UnsupportedMCP
    with what to install, instead of registering tools whose hints are lost."""
    try:
        from mcp.types import ToolAnnotations
    except ImportError:
        return dict(hints)
    try:
        ann = ToolAnnotations(**hints)
        sent = _sent(ann)
    except Exception as e:
        raise UnsupportedMCP(f"{_installed()} does not take the MCP tool annotations {sorted(hints)} "
                             f"({type(e).__name__}); {SUPPORTED_MCP}") from e
    lost = sorted(k for k, v in hints.items() if k not in sent or sent[k] != v)
    if lost:
        raise UnsupportedMCP(f"{_installed()} does not send the MCP tool annotations {lost} as given (its "
                             f"ToolAnnotations sends {sorted(sent)}); {SUPPORTED_MCP}")
    return ann


def register(app, tools):
    """Registers the tools on a FastMCP server (or any object with FastMCP's tool(name=, description=) decorator);
    the TOOL_HINTS annotations too when its tool() takes annotations (MCP SDK versions from about 1.9 on). Raises
    UnsupportedMCP when the SDK's ToolAnnotations does not keep these hints."""
    hints = _takes(app.tool, "annotations")
    for name, fn, description in tools:
        kwargs = {"name": name, "description": description}
        if hints and name in TOOL_HINTS:
            kwargs["annotations"] = _annotations(TOOL_HINTS[name])
        app.tool(**kwargs)(fn)
    return app


def build_server(service, name="anyintent", **settings):
    """A FastMCP server with the tools of this service. settings: FastMCP settings such as host and port, given to its
    constructor (newer SDK versions set up their own DNS rebinding protection there, from the host). Without the mcp
    package: ImportError with the install hint; with mcp 2 or later (no FastMCP there): UnsupportedMCP."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        version = installed_mcp_version()
        if (mcp_major(version) or 0) >= 2:
            raise UnsupportedMCP(f"the installed mcp {version} has no FastMCP (mcp.server.fastmcp); "
                                 f"{SUPPORTED_MCP}") from e
        raise ImportError(INSTALL_HINT) from e
    return register(FastMCP(name, instructions=INSTRUCTIONS, **settings), make_tools(service))


def _env(name):
    """An environment variable; empty counts as not set."""
    return os.environ.get(name) or None


def build_parser():
    p = argparse.ArgumentParser(prog="anyintent-mcp", description="MCP server for AnyIntent intent classification "
                                "(stdio transport by default, for Claude Desktop, Cursor and other MCP clients)")
    p.add_argument("--model", default=_env("ANYINTENT_MODEL"),
                   help="Hugging Face model id or local folder, for example lukashm/r2intent-v1 (default: "
                        "ANYINTENT_MODEL)")
    p.add_argument("--revision", default=_env("ANYINTENT_REVISION"), help="tag or commit of the model")
    p.add_argument("--device", default=None, help="cpu, cuda or cuda:N (default: cuda when available, else cpu)")
    p.add_argument("--lists-dir", default=_env("ANYINTENT_LISTS_DIR"),
                   help="folder for saved intent lists ('~' is expanded; default: ANYINTENT_LISTS_DIR; without it, "
                        "lists live in memory until the server stops)")
    p.add_argument("--max-text-chars", type=int, default=Limits.max_text_chars,
                   help="characters per message (default 2000)")
    p.add_argument("--max-intents", type=int, default=Limits.max_intents,
                   help="intents per call or saved list (default 200)")
    p.add_argument("--label-text-mode", default=_env("ANYINTENT_LABEL_TEXT_MODE"), choices=LABEL_TEXT_MODES,
                   help="the label text mode passed to set_labels (default: the model's own). v2 for plain "
                        "descriptions on a model released with v3s (see the README section 'Writing labels')")
    p.add_argument("--doubt-margin", type=float, default=None,
                   help="add 'doubtful' to every classify_intent result: true when its margin is below this value. "
                        "Choose it on your own labelled examples for this model and these settings (no default)")
    p.add_argument("--load-timeout", type=float, default=45.0,
                   help="seconds a tool call waits for the model while it loads (default 45, below the 60 s request "
                        "timeout common in MCP clients)")
    p.add_argument("--log-texts", action="store_true", help="log message texts and labels to stderr (off by default)")
    p.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    p.add_argument("--host", default="127.0.0.1", help="address for the sse and streamable-http transports (default "
                                                       "127.0.0.1; another address needs --allow-remote)")
    p.add_argument("--port", type=int, default=8001, help="port for the sse and streamable-http transports")
    p.add_argument("--allow-remote", action="store_true",
                   help="allow the sse and streamable-http transports on a non-loopback --host. They have no "
                        "authentication: anyone who can reach the port can use the model and the saved lists")
    p.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    return p


class ProtocolStdout:
    """sys.stdout for the stdio transport: the MCP SDK writes the protocol to sys.stdout.buffer (the real standard
    output); text that any library prints goes to stderr instead of into the protocol stream."""

    def __init__(self, real, err):
        self.buffer = real.buffer
        self._err = err

    def write(self, text):
        return self._err.write(text)

    def flush(self):
        self._err.flush()

    def __getattr__(self, name):
        return getattr(self._err, name)


def _loader(args):
    def load():
        from .model import AnyIntent
        return AnyIntent.from_pretrained(args.model, revision=args.revision, device=args.device)
    return load


# loggers that log what they receive or send (tool arguments and results included) at debug level
PROTOCOL_LOGGERS = ("mcp", "sse_starlette")


def transport_app(app, transport):
    """The ASGI app of an HTTP transport (FastMCP.streamable_http_app or FastMCP.sse_app), or None when the SDK has no
    such method."""
    make = getattr(app, "streamable_http_app" if transport == "streamable-http" else "sse_app", None)
    return make() if callable(make) else None


def serve_http(app, args):
    """Runs an HTTP transport with uvicorn (as FastMCP.run does) behind this server's Host check: on a loopback
    address only requests addressed to 127.0.0.1 or localhost are accepted, whatever the SDK version does itself."""
    inner = transport_app(app, args.transport)
    if inner is None:
        log.warning("this MCP SDK has no app for %s to wrap: it runs without this server's Host check",
                    args.transport)
        app.run(transport=args.transport)
        return 0
    import uvicorn
    allowed = list(LOOPBACK_NAMES) if is_loopback(args.host) else None
    guarded = HostGuard(inner, allowed, hint="the MCP server accepts only 127.0.0.1 and localhost when it listens on "
                                             "a loopback address")
    path = getattr(getattr(app, "settings", None), "streamable_http_path" if args.transport == "streamable-http" else
                   "sse_path", None) or ("/mcp" if args.transport == "streamable-http" else "/sse")
    shown = f"[{args.host}]" if ":" in args.host else args.host
    log.info("MCP over %s at http://%s:%d%s", args.transport, shown, args.port, path)
    uvicorn.run(guarded, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.transport == "stdio" and hasattr(sys.stdout, "buffer") and not isinstance(sys.stdout, ProtocolStdout):
        # before anything is imported or printed: text on stdout would corrupt the protocol stream
        sys.stdout = ProtocolStdout(sys.stdout, sys.stderr)
    logging.basicConfig(stream=sys.stderr, level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.log_texts and logging.getLogger().getEffectiveLevel() < logging.INFO:
        for name in PROTOCOL_LOGGERS:
            logging.getLogger(name).setLevel(logging.INFO)
    if not args.model:
        log.error("no model: pass --model (for example lukashm/r2intent-v1) or set ANYINTENT_MODEL")
        return 2
    try:
        limits = Limits(max_text_chars=args.max_text_chars, max_intents=args.max_intents)
        check_doubt_margin(args.doubt_margin)
    except ValueError as e:
        log.error("%s", e)
        return 2
    http = args.transport != "stdio"
    if http and not is_loopback(args.host) and not args.allow_remote:
        log.error("MCP over %s has no authentication: listening on %s needs --allow-remote (then anyone who can reach "
                  "the port can use the model and the saved lists); the default 127.0.0.1 serves this machine only",
                  args.transport, args.host)
        return 2
    try:
        service = Service(loader=_loader(args), model_id=args.model, lists_dir=args.lists_dir, limits=limits,
                          log_texts=args.log_texts, load_wait=args.load_timeout,
                          label_text_mode=args.label_text_mode, doubt_margin=args.doubt_margin)
    except OSError as e:
        log.error("the intent list folder %s cannot be used: %s", args.lists_dir, e)
        return 2
    try:
        app = build_server(service, **({"host": args.host, "port": args.port} if http else {}))
    except ImportError as e:
        log.error("%s", e)
        return 2
    service.start_loading()
    if args.log_texts:
        log.warning("--log-texts is on: message texts and labels are written to the log")
    if not http:
        app.run(transport="stdio")
        return 0
    if not is_loopback(args.host):
        log.warning("MCP over %s on %s has no authentication: anyone who can reach the port can use the model and the "
                    "saved lists", args.transport, args.host)
    return serve_http(app, args)


if __name__ == "__main__":
    sys.exit(main())
