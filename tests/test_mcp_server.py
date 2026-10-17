"""The MCP server (anyintent.mcp_server, extra 'mcp'). The tool functions are plain async functions: they are called
directly here, without the MCP package. Their input schema is built with pydantic as FastMCP builds it (skipped without
pydantic); the registration on a real FastMCP server and tool calls through it are skipped without the mcp package
(and with mcp 2, which is not supported yet: there the clear error is checked). check_intent and score_intents run on
tiny models with the settings of R2Intent v1, with a calibration file (equal to the library, one message per call) and
without it (an error with the clear message)."""
import asyncio
import importlib.util
import json
import logging
import re
import sys
import types
import warnings
from typing import Optional

import pytest

from anyintent import AnyIntent, __version__, mcp_server
from anyintent.service import CalibrationMissing, InvalidRequest, NotFound, NotReady, Service, ServiceError
from conftest import LABELS, MESSAGES
from serving_fakes import (INTENTS, SECRET, FakeBroken, FakeClassifier, FakeWithExtras, assert_scores_answer,
                           calibrated_models)

BASE = {"classify_intent", "save_intent_list", "get_intent_list", "list_intent_lists", "delete_intent_list"}
RESULT_KEYS = {"label", "index", "score", "margin", "runner_up"}
CHECK_KEYS = {"intent", "yes", "probability", "score", "threshold"}
SCORES_KEYS = {"label", "intents"}


@pytest.fixture(scope="module")
def cal_models(tmp_path_factory, tok):
    return calibrated_models(str(tmp_path_factory.mktemp("serving_cal")), tok)


def load(folder):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AnyIntent.from_pretrained(folder, device="cpu")


def tools_of(clf, **kw):
    return {name: (fn, desc) for name, fn, desc in mcp_server.make_tools(Service(clf, model_id="m", **kw))}


def run(coro):
    return asyncio.run(coro)


def test_tool_set_follows_the_library():
    assert set(tools_of(FakeClassifier())) == BASE
    assert set(tools_of(FakeWithExtras())) == BASE | {"check_intent", "score_intents"}
    real = set(dict.fromkeys(n for n, _, _ in mcp_server.make_tools(Service(loader=lambda: None))))
    assert ("check_intent" in real) == hasattr(AnyIntent, "check")
    assert ("score_intents" in real) == hasattr(AnyIntent, "scores")


def test_descriptions_tell_an_agent_when_to_call():
    t = tools_of(FakeWithExtras())
    d = t["classify_intent"][1]
    for needle in ("Route one customer message", "`intents`", "`list_name`", "`label`", "`score`", "`margin`",
                   "`runner_up`", "small margin means the answer is doubtful", "not probabilities",
                   "If the result has `doubtful`, follow it", "ask the user", "list_intent_lists"):
        assert needle in d, needle
    assert "save_intent_list" in d and "list_name" in t["save_intent_list"][1]
    assert "before passing `list_name`" in t["list_intent_lists"][1]
    assert "only when the user asks" in t["delete_intent_list"][1]
    assert "speed" not in " ".join(x[1] for x in t.values()).lower()


def test_check_and_score_descriptions():
    """check_intent and score_intents say when to call them, what yes and probability mean; for score_intents, that
    the listed intents are the ones the model read, that an intent not listed was not read (value 0, still possible),
    that the values sum to 1 with 5 intents or fewer and can sum to less than 1 with more, and that the part missing to
    1 is not a 'none of these' answer (the description names no remainder); that the numbers are calibrated only for
    intents written with a description, and that a model without their calibration answers with an error."""
    t = tools_of(FakeWithExtras())
    check, scores = t["check_intent"][1], t["score_intents"][1]
    for needle in ("classify_intent", "single candidate intent", "`intent`", "`yes`", "`probability`",
                   "equally likely", "calibrated only on intents written with a description", "`score`",
                   "not a probability", "returns an error saying so", "not stored"):
        assert needle in check, needle
    for needle in ("`intents`", "`list_name`", '`{"intent": name, "value": number}`', "at most 5 intents",
                   "each intent the model reads", "lists only the intents the model read",
                   "An intent not listed was not read", "its value is 0", "not read, not impossible",
                   "can still be the right intent", "highest value to the lowest",
                   "every intent is listed and the values sum to 1", "sum to less than 1", "did not read",
                   "The part missing to reach 1 is NOT a 'none of these intents' answer", "no such mode",
                   "assume that one of the given intents is right",
                   "calibrated only on intents written with a description", "`label`", "the first one listed",
                   "returns an error saying so", "not stored"):
        assert needle in scores, needle
    assert "remainder" not in scores and "`values`" not in scores and "`read`" not in scores
    assert "This is" not in scores and scores.startswith("For one customer message")
    assert "check_intent" in mcp_server.INSTRUCTIONS and "score_intents" in mcp_server.INSTRUCTIONS
    assert "model read" in mcp_server.INSTRUCTIONS and "remainder" not in mcp_server.INSTRUCTIONS


def test_check_is_a_beta_and_nothing_says_zero_shot():
    """check_intent is marked a beta (it shipped in R2Intent v1 after it missed one pre-registered DEV calibration
    limit) in its description and in the instructions; score_intents is not; no text calls the models zero-shot, and
    the instructions and classify_intent say that no training examples of the intents are needed."""
    t = tools_of(FakeWithExtras())
    check, scores, classify = t["check_intent"][1], t["score_intents"][1], t["classify_intent"][1]
    for needle in ("This tool is a beta", "R2Intent v1", "missed one pre-registered calibration limit on the DEV data",
                   "treat the probability as approximate"):
        assert needle in check, needle
    assert "check_intent (yes or no for one intent, with a probability; a beta)" in mcp_server.INSTRUCTIONS
    assert "beta" not in scores.lower()
    assert "no training examples of those intents are needed" in mcp_server.INSTRUCTIONS
    assert "no training examples of those intents are needed" in classify
    every = [mcp_server.INSTRUCTIONS] + [x[1] for x in t.values()]
    assert not any("zero-shot" in s.lower() or "zero shot" in s.lower() for s in every)


def test_classify_and_lists_through_the_tools(tmp_path):
    t = tools_of(FakeClassifier(), lists_dir=str(tmp_path / "lists"))
    classify, save, get, names = (t[n][0] for n in ("classify_intent", "save_intent_list", "get_intent_list",
                                                    "list_intent_lists"))
    r = run(classify("I lost my bank card", intents=INTENTS))
    assert r["label"] == "lost card" and set(r) == RESULT_KEYS and json.dumps(r)
    assert r["runner_up"] in ("refund status", "close account")
    s = run(save("support", INTENTS))
    assert s["created"] and s["persistent"] and "note" not in s
    assert run(get("support"))["intents"][0]["name"] == "lost card"
    assert run(names()) == {"lists": [{"name": "support", "n_intents": 3}], "persistent": True}
    assert run(classify("when will my refund arrive", list_name="support"))["label"] == "refund status"
    with pytest.raises(InvalidRequest):
        run(classify("x"))
    with pytest.raises(InvalidRequest):
        run(classify("x", intents=INTENTS, list_name="support"))
    with pytest.raises(NotFound):
        run(classify("x", list_name="other"))
    with pytest.raises(InvalidRequest):
        run(save("Bad Name", INTENTS))
    with pytest.raises(NotFound, match="list_intent_lists"):
        run(get("other"))
    assert run(t["delete_intent_list"][0]("support")) == {"name": "support", "deleted": True}
    assert run(names())["lists"] == [] and not (tmp_path / "lists" / "support.json").exists()
    mem = tools_of(FakeClassifier())
    assert "in memory" in run(mem["save_intent_list"][0]("a", ["x"]))["note"]
    d = tools_of(FakeClassifier(), doubt_margin=0.5)["classify_intent"][0]
    assert set(run(d("I lost my bank card", intents=INTENTS))) == RESULT_KEYS | {"doubtful"}


def test_unexpected_errors_reach_the_client_without_details(caplog):
    caplog.set_level(logging.DEBUG)
    classify = tools_of(FakeBroken())["classify_intent"][0]
    with pytest.raises(RuntimeError) as e:
        run(classify(SECRET, intents=INTENTS))
    assert str(e.value) == mcp_server.INTERNAL_ERROR and SECRET not in str(e.value)
    assert e.value.__cause__ is None and e.value.__context__ is None
    assert "internal error in classify_intent" in caplog.text and "RuntimeError" in caplog.text
    assert "Traceback, frames only" in caplog.text and SECRET not in caplog.text


def test_check_and_score_tools():
    t = tools_of(FakeWithExtras())
    r = run(t["check_intent"][0]("I lost my bank card", INTENTS[0]))
    assert set(r) == CHECK_KEYS and r["yes"] is True and r["intent"] == "lost card" and r["threshold"] == 0.5
    assert json.dumps(r)
    r = run(t["score_intents"][0]("I lost my bank card", intents=INTENTS))
    assert set(r) == SCORES_KEYS and r["label"] == "lost card" and r["intents"][0]["intent"] == "lost card"
    assert_scores_answer(r, FakeWithExtras().set_labels(INTENTS).scores("I lost my bank card"))
    assert json.dumps(r)
    run(t["save_intent_list"][0]("s", INTENTS))
    assert run(t["score_intents"][0]("I lost my bank card", list_name="s")) == r
    with pytest.raises(InvalidRequest):
        run(t["score_intents"][0]("x", intents=INTENTS, list_name="s"))
    with pytest.raises(InvalidRequest, match="empty name"):
        run(t["check_intent"][0]("x", ""))


def test_check_and_score_tools_equal_the_library(cal_models):
    """One message per call: every field equals the library's result for that message alone."""
    clf = load(cal_models["full"])
    t = tools_of(clf)
    intent = LABELS[7]
    for m in MESSAGES:
        got = run(t["check_intent"][0](m, intent))
        w = clf.check(m, intent)
        assert got == {k: getattr(w, k) for k in CHECK_KEYS}, m
    run(t["save_intent_list"][0]("tiny", LABELS))
    sums = []
    for m in MESSAGES[:8]:
        got = run(t["score_intents"][0](m, intents=LABELS))
        clf.set_labels(LABELS)
        w = clf.scores(m)
        assert_scores_answer(got, w)
        assert len(got["intents"]) == 5 and w.remainder > 0 and got["label"] == clf.classify(m, cascade=False).label
        assert run(t["score_intents"][0](m, list_name="tiny")) == got
        sums.append(sum(x["value"] for x in got["intents"]))
    assert any(x < 1 - 1e-6 for x in sums)        # 20 intents: the values sum to 1 minus the remainder, less than 1


def test_check_and_score_tools_without_calibration(cal_models, caplog):
    """A model whose release does not ship the calibration: the tool raises CalibrationMissing, a ServiceError that
    FastMCP sends as an error result with its clear message (not the internal error), and nothing is logged as an
    internal error."""
    caplog.set_level(logging.DEBUG)
    t = tools_of(load(cal_models["none"]))
    for name, call in (("check", lambda: t["check_intent"][0](SECRET, LABELS[0])),
                       ("scores", lambda: t["score_intents"][0](SECRET, intents=LABELS))):
        with pytest.raises(CalibrationMissing) as e:
            run(call())
        assert isinstance(e.value, ServiceError) and str(e.value) == e.value.message
        assert e.value.message.startswith(f"'{name}' is not available with this model") and \
            "does not ship one" in e.value.message and mcp_server.INTERNAL_ERROR not in e.value.message
    one = tools_of(load(cal_models["check_only"]))
    assert set(run(one["check_intent"][0]("x", LABELS[0]))) == CHECK_KEYS
    with pytest.raises(CalibrationMissing, match="ships no calibration for scores"):
        run(one["score_intents"][0]("x", intents=LABELS))
    assert "internal error" not in caplog.text and SECRET not in caplog.text


def test_classify_tool_equals_the_library(one_file):
    clf = AnyIntent.from_pretrained(one_file, device="cpu")
    classify = tools_of(clf)["classify_intent"][0]
    got = [run(classify(m, intents=LABELS)) for m in MESSAGES[:6]]
    clf.set_labels(LABELS)
    want = [clf.classify(m) for m in MESSAGES[:6]]
    assert [(g["label"], g["score"], g["margin"]) for g in got] == [(p.label, p.score, p.margin) for p in want]


def test_tools_wait_for_the_model_and_report_a_failed_load():
    import threading
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return FakeClassifier()

    s = Service(loader=slow, load_wait=0.05)
    classify = dict((n, f) for n, f, _ in mcp_server.make_tools(s))["classify_intent"]
    s.start_loading()
    with pytest.raises(NotReady):
        run(classify("x", intents=["a"]))
    gate.set()
    s.load_wait = 5
    assert run(classify("x", intents=["a"]))["label"] == "a"


def test_input_schema_as_fastmcp_builds_it():
    """FastMCP turns each tool's signature into a pydantic model and its JSON schema; the same is done here."""
    pydantic = pytest.importorskip("pydantic")
    import inspect
    for name, (fn, _) in tools_of(FakeWithExtras()).items():
        fields = {p.name: (p.annotation, ... if p.default is inspect.Parameter.empty else p.default)
                  for p in inspect.signature(fn).parameters.values()}
        model = pydantic.create_model(f"{name}_args", **fields)
        schema = model.model_json_schema()
        for p in fields:
            assert schema["properties"][p].get("description"), (name, p)
        if name == "classify_intent":
            assert schema["required"] == ["message"]
            ok = model.model_validate({"message": "m", "intents": ["a", {"name": "b", "description": "d"}]})
            assert ok.intents[1] == {"name": "b", "description": "d"}
            with pytest.raises(pydantic.ValidationError):
                model.model_validate({"message": "m", "intents": [{"description": "no name"}]})


def test_registration_on_a_fastmcp_like_object():
    class FakeApp:
        def __init__(self):
            self.tools = {}

        def tool(self, name=None, description=None):
            def deco(fn):
                self.tools[name] = (fn, description)
                return fn
            return deco

    app = mcp_server.register(FakeApp(), mcp_server.make_tools(Service(FakeClassifier())))
    assert set(app.tools) == BASE and all(d for _, d in app.tools.values())

    class FakeAppWithHints(FakeApp):
        def tool(self, name=None, title=None, description=None, annotations=None):
            def deco(fn):
                self.tools[name] = (fn, description, annotations)
                return fn
            return deco

    app = mcp_server.register(FakeAppWithHints(), mcp_server.make_tools(Service(FakeWithExtras())))
    # as the SDK sends them: by alias (newer SDKs name the pydantic fields differently, readOnlyHint is the alias)
    hints = {n: dict(a) if isinstance(a, dict) else a.model_dump(by_alias=True, exclude_none=True, mode="json")
             for n, (_, _, a) in app.tools.items()}
    assert set(hints) == BASE | {"check_intent", "score_intents"}
    for n in ("classify_intent", "get_intent_list", "list_intent_lists", "check_intent", "score_intents"):
        assert hints[n]["readOnlyHint"] is True and hints[n]["destructiveHint"] is False, n
    for n in ("save_intent_list", "delete_intent_list"):
        assert hints[n]["readOnlyHint"] is False and hints[n]["destructiveHint"] is True, n
    assert all(h["openWorldHint"] is False for h in hints.values())


class AppTakingHints:
    """A FastMCP-like object whose tool() takes annotations; keeps them by tool name."""

    def __init__(self):
        self.tools = {}

    def tool(self, name=None, title=None, description=None, annotations=None):
        def deco(fn):
            self.tools[name] = annotations
            return fn
        return deco


def _fake_mcp_types(monkeypatch, tool_annotations):
    mod = types.ModuleType("mcp.types")
    mod.ToolAnnotations = tool_annotations
    for name, obj in (("mcp", types.ModuleType("mcp")), ("mcp.types", mod)):
        monkeypatch.setitem(sys.modules, name, obj)


def test_tool_annotations_of_another_sdk_fail_clearly(monkeypatch):
    """The tool hints use the MCP field names (readOnlyHint ...). A ToolAnnotations that names its fields in snake_case
    with these names as aliases (a pydantic alias generator) sends them as they are: registered. One that drops or
    refuses them: UnsupportedMCP (an ImportError, so main() exits with 2) saying that mcp 2 is not supported yet,
    instead of a KeyError or tools whose hints are silently lost."""
    pydantic = pytest.importorskip("pydantic")
    from pydantic import BaseModel, ConfigDict
    from pydantic.alias_generators import to_camel
    tools = mcp_server.make_tools(Service(FakeWithExtras()))

    class Fields(BaseModel):
        title: Optional[str] = None
        read_only_hint: Optional[bool] = None
        destructive_hint: Optional[bool] = None
        idempotent_hint: Optional[bool] = None
        open_world_hint: Optional[bool] = None

    class Aliased(Fields):
        model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")

    class Dropping(Fields):
        model_config = ConfigDict(extra="ignore")

    class Refusing(Fields):
        model_config = ConfigDict(extra="forbid")

    _fake_mcp_types(monkeypatch, Aliased)
    app = mcp_server.register(AppTakingHints(), tools)
    ann = app.tools["classify_intent"]
    assert "readOnlyHint" not in ann.model_dump(exclude_none=True)       # the field names are not the MCP names
    assert ann.model_dump(by_alias=True, exclude_none=True) == mcp_server.TOOL_HINTS["classify_intent"]
    for bad in (Dropping, Refusing):
        _fake_mcp_types(monkeypatch, bad)
        with pytest.raises(mcp_server.UnsupportedMCP, match=r'mcp 2 is not supported yet: pip install "mcp<2"') as e:
            mcp_server.register(AppTakingHints(), tools)
        assert isinstance(e.value, ImportError) and "readOnlyHint" in str(e.value), bad.__name__
    with pytest.raises(pydantic.ValidationError):
        Refusing(readOnlyHint=True)


def test_mcp_2_is_refused_with_what_to_install(monkeypatch, caplog):
    """Without FastMCP: with mcp 2 or later installed, the error says that mcp 2 is not supported yet and how to
    install mcp 1.x (anyintent-mcp logs it and exits with 2); without the package (or with mcp 1.x that cannot be
    imported), the install hint of the extra."""
    for name in ("mcp", "mcp.server", "mcp.server.fastmcp"):
        monkeypatch.setitem(sys.modules, name, None)                  # the import of mcp.server.fastmcp fails
    for version, unsupported, want in (
            ("2.3.0", True, r'mcp 2\.3\.0 has no FastMCP.*mcp 2 is not supported yet: pip install "mcp<2"'),
            ("10.0", True, r'mcp 10\.0 has no FastMCP.*pip install "mcp<2"'),
            ("1.27.0", False, r"anyintent\[mcp\]"), (None, False, r"anyintent\[mcp\]")):
        monkeypatch.setattr(mcp_server, "installed_mcp_version", lambda v=version: v)
        with pytest.raises(ImportError, match=want) as e:
            mcp_server.build_server(Service(FakeClassifier()))
        assert isinstance(e.value, mcp_server.UnsupportedMCP) == unsupported, version
    monkeypatch.setattr(mcp_server, "installed_mcp_version", lambda: "2.3.0")
    monkeypatch.setattr(sys, "stdout", sys.stdout)         # main() swaps it for the stdio transport
    assert mcp_server.main(["--model", "m"]) == 2 and 'mcp 2 is not supported yet: pip install "mcp<2"' in caplog.text
    assert [mcp_server.mcp_major(v) for v in ("1.27.0", "2.3.0", "2", "", None, "x")] == [1, 2, 2, None, None, None]


def test_build_server_gives_host_and_port_to_the_constructor(monkeypatch):
    seen = {}

    class FakeFastMCP:
        def __init__(self, name, instructions=None, **settings):
            seen.update(name=name, instructions=instructions, **settings)

        def tool(self, name=None, description=None):
            return lambda fn: fn

    fastmcp = types.ModuleType("mcp.server.fastmcp")
    fastmcp.FastMCP = FakeFastMCP
    for mod, obj in (("mcp", types.ModuleType("mcp")), ("mcp.server", types.ModuleType("mcp.server")),
                     ("mcp.server.fastmcp", fastmcp)):
        monkeypatch.setitem(sys.modules, mod, obj)
    mcp_server.build_server(Service(FakeClassifier()), host="127.0.0.1", port=8123)
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8123 and seen["name"] == "anyintent"
    assert seen["instructions"] == mcp_server.INSTRUCTIONS


def test_main_wires_the_server(monkeypatch, tmp_path, caplog):
    seen = {}

    class FakeServer:
        def run(self, transport="stdio"):
            seen["transport"] = transport

    def fake_build(service, name="anyintent", **settings):
        seen.update(service=service, settings=settings, stdout_at_build=sys.stdout)
        return FakeServer()

    def fake_serve_http(app, args):
        seen.update(transport=args.transport, http=(args.host, args.port))
        return 0

    monkeypatch.setattr(mcp_server, "build_server", fake_build)
    monkeypatch.setattr(mcp_server, "serve_http", fake_serve_http)
    monkeypatch.setattr(mcp_server, "_loader", lambda args: FakeClassifier)
    import io
    real_out, err_out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8"), io.StringIO()
    monkeypatch.setattr(sys, "stdout", real_out)
    monkeypatch.setattr(sys, "stderr", err_out)
    assert mcp_server.main(["--model", "m", "--lists-dir", str(tmp_path / "l")]) == 0
    assert isinstance(sys.stdout, mcp_server.ProtocolStdout) and sys.stdout.buffer is real_out.buffer
    assert seen["stdout_at_build"] is sys.stdout                  # swapped before the tools import the library
    print("a library prints this")                                 # stdio: printed text stays out of the protocol
    assert "a library prints this" in err_out.getvalue() and real_out.buffer.getvalue() == b""
    monkeypatch.setattr(sys, "stdout", real_out)
    s = seen["service"]
    assert seen["transport"] == "stdio" and s.lists.persistent and s.model_id == "m" and s.load_wait == 45.0
    assert seen["settings"] == {}
    for _ in range(100):
        if s.ready:
            break
        asyncio.run(asyncio.sleep(0.02))
    assert s.ready and s.classify("x", intents=["a"])[0]["label"] == "a"
    assert mcp_server.main(["--model", "m", "--transport", "streamable-http", "--port", "9009"]) == 0
    assert seen["transport"] == "streamable-http" and seen["settings"] == {"host": "127.0.0.1", "port": 9009}
    assert seen["http"] == ("127.0.0.1", 9009) and sys.stdout is real_out        # no swap for HTTP transports
    assert mcp_server.main(["--model", "m", "--transport", "sse", "--host", "0.0.0.0"]) == 2   # no authentication
    assert mcp_server.main(["--model", "m", "--transport", "sse", "--host", "0.0.0.0", "--allow-remote"]) == 0
    assert seen["settings"]["host"] == "0.0.0.0"
    assert mcp_server.main(["--model", "m", "--label-text-mode", "v2", "--doubt-margin", "0.25"]) == 0
    assert seen["service"].label_text_mode == "v2" and seen["service"].doubt_margin == 0.25
    assert mcp_server.main(["--model", "m", "--doubt-margin", "0"]) == 2
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    assert mcp_server.main(["--model", "m", "--lists-dir", str(blocker / "l")]) == 2    # a clear error, no traceback
    monkeypatch.setenv("ANYINTENT_REVISION", "")
    monkeypatch.setenv("ANYINTENT_LISTS_DIR", "")
    a = mcp_server.build_parser().parse_args(["--model", "m"])
    assert a.revision is None and a.lists_dir is None and a.load_timeout == 45.0
    monkeypatch.delenv("ANYINTENT_MODEL", raising=False)
    assert mcp_server.main([]) == 2
    caplog.set_level(logging.DEBUG)                                 # as --log-level debug sets the root logger
    assert mcp_server.main(["--model", "m", "--log-level", "debug"]) == 0
    for name in ("mcp", "sse_starlette"):                          # their debug logs hold tool arguments and results
        assert logging.getLogger(name).level == logging.INFO, name
        logging.getLogger(name).setLevel(logging.NOTSET)
    assert mcp_server.main(["--model", "m", "--log-level", "debug", "--log-texts"]) == 0
    assert logging.getLogger("mcp").level == logging.NOTSET


def _asgi_get(app, host, path="/mcp"):
    """One GET through a plain ASGI app; returns (status, body)."""
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": path, "headers": [(b"host", host.encode())]}
    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


def test_serve_http_puts_the_host_check_in_front(monkeypatch):
    import argparse
    calls = []

    async def inner(scope, receive, send):
        calls.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    class App:
        settings = types.SimpleNamespace(streamable_http_path="/mcp", sse_path="/sse")

        def streamable_http_app(self):
            return inner

        def sse_app(self):
            return inner

    seen = {}
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kw: seen.update(app=app, **kw)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    args = argparse.Namespace(transport="streamable-http", host="127.0.0.1", port=8001, log_level="info")
    assert mcp_server.serve_http(App(), args) == 0
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8001
    assert _asgi_get(seen["app"], "127.0.0.1:8001") == (200, b"ok")
    assert _asgi_get(seen["app"], "localhost:8001")[0] == 200
    status, body = _asgi_get(seen["app"], "evil.example:8001")
    assert status == 400 and json.loads(body)["error"]["code"] == "host_not_allowed" and calls == ["/mcp", "/mcp"]
    args.host = "0.0.0.0"
    mcp_server.serve_http(App(), args)                     # --allow-remote: no Host check (and no authentication)
    assert _asgi_get(seen["app"], "evil.example:8001")[0] == 200

    class OldApp:
        def run(self, transport):
            seen["ran"] = transport

    assert mcp_server.serve_http(OldApp(), argparse.Namespace(transport="sse", host="127.0.0.1", port=1,
                                                               log_level="info")) == 0
    assert seen["ran"] == "sse"


def test_build_server_with_or_without_the_mcp_package(monkeypatch, caplog):
    monkeypatch.setattr(sys, "stdout", sys.stdout)         # main() may swap it for the stdio transport
    if importlib.util.find_spec("mcp") is None:          # without the extra: a clear install hint, exit code 2
        with pytest.raises(ImportError, match=r"anyintent\[mcp\]"):
            mcp_server.build_server(Service(FakeClassifier()))
        assert mcp_server.main(["--model", "m"]) == 2 and "anyintent[mcp]" in caplog.text
    elif (mcp_server.mcp_major(mcp_server.installed_mcp_version()) or 0) >= 2:     # not supported yet, exit code 2
        with pytest.raises(mcp_server.UnsupportedMCP, match=r'mcp 2 is not supported yet: pip install "mcp<2"'):
            mcp_server.build_server(Service(FakeClassifier()))
        assert mcp_server.main(["--model", "m"]) == 2 and 'pip install "mcp<2"' in caplog.text
    else:
        app = mcp_server.build_server(Service(FakeClassifier()))
        assert callable(getattr(app, "run", None)) and hasattr(app, "settings")


def test_install_hint_pins_a_release_tag():
    """The install hint of the mcp extra names a release tag (a bare repository URL would install the default branch
    over the installed release), and in a release it is this release's own tag."""
    tag = re.search(r'"anyintent\[mcp\] @ git\+https://github\.com/lukamaga/anyintent@v(\d+\.\d+\.\d+)"$',
                    mcp_server.INSTALL_HINT)
    assert tag, mcp_server.INSTALL_HINT
    if re.fullmatch(r"\d+\.\d+\.\d+", __version__):
        assert tag.group(1) == __version__


def _text_of(result):
    """FastMCP.call_tool returns content blocks (older SDK) or (content blocks, structured result) (newer SDK)."""
    content = result[0] if isinstance(result, tuple) else result
    return "".join(getattr(c, "text", "") for c in content)


def test_real_fastmcp_server(one_file, cal_models, caplog):
    pytest.importorskip("mcp")
    if (mcp_server.mcp_major(mcp_server.installed_mcp_version()) or 0) >= 2:
        pytest.skip("mcp 2 is not supported yet (test_build_server_with_or_without_the_mcp_package checks its error)")
    import logging
    caplog.set_level(logging.INFO)          # the default level of anyintent-mcp
    clf = AnyIntent.from_pretrained(one_file, device="cpu")
    app = mcp_server.build_server(Service(clf, model_id="tiny"))

    async def go():
        tools = {t.name: t for t in await app.list_tools()}
        assert BASE | {"check_intent", "score_intents"} == set(tools)
        assert "NOT a 'none of these intents' answer" in tools["score_intents"].description
        assert "remainder" not in tools["score_intents"].description
        assert "This tool is a beta" in tools["check_intent"].description
        assert {"message", "intent"} <= set(tools["check_intent"].inputSchema["properties"])
        assert "margin" in tools["classify_intent"].description
        schema = tools["classify_intent"].inputSchema
        assert "message" in schema.get("required", []) and {"intents", "list_name"} <= set(schema["properties"])
        ann = getattr(tools["classify_intent"], "annotations", None)
        if ann is not None:                                 # SDK versions with tool annotations, as they are sent
            sent = {n: tools[n].annotations.model_dump(by_alias=True, exclude_none=True)
                    for n in ("classify_intent", "delete_intent_list")}
            assert sent["classify_intent"]["readOnlyHint"] is True
            assert sent["delete_intent_list"]["destructiveHint"] is True
        r = await app.call_tool("classify_intent", {"message": MESSAGES[0] + " " + SECRET, "intents": LABELS})
        # one_file has no calibration file: FastMCP raises the tool's error in process (its protocol layer sends it to
        # a client as an error result with this text)
        with pytest.raises(Exception, match="'check' is not available with this model") as e:
            await app.call_tool("check_intent", {"message": SECRET, "intent": LABELS[0]})
        assert mcp_server.INTERNAL_ERROR not in str(e.value)
        return json.loads(_text_of(r))

    got = asyncio.run(go())
    clf.set_labels(LABELS)
    p = clf.classify(MESSAGES[0] + " " + SECRET)
    assert (got["label"], got["score"], got["margin"]) == (p.label, p.score, p.margin) and "runner_up" in got
    cal = load(cal_models["full"])
    app = mcp_server.build_server(Service(cal, model_id="tiny calibrated"))

    async def calibrated():
        c = json.loads(_text_of(await app.call_tool("check_intent", {"message": MESSAGES[7], "intent": LABELS[7]})))
        s = json.loads(_text_of(await app.call_tool("score_intents", {"message": MESSAGES[7], "intents": LABELS})))
        return c, s

    c, s = asyncio.run(calibrated())
    w = cal.check(MESSAGES[7], LABELS[7])
    assert c == {k: getattr(w, k) for k in CHECK_KEYS}
    cal.set_labels(LABELS)
    w = cal.scores(MESSAGES[7])
    assert_scores_answer(s, w)
    assert len(s["intents"]) == 5
    assert SECRET not in caplog.text
