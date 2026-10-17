"""Live smoke of the HTTP and MCP servers on a tiny random model (no download, no network beyond 127.0.0.1).

    python tools/server_smoke.py --out <folder> [--skip-mcp]

Needs the package installed with the server and mcp extras (pip install ".[server,mcp,test]"). It builds two tiny
models from the test fixtures: the one-file model of tests/conftest.py (no calibration file) and a model at the settings
of R2Intent v1 (w 60, k 5, label text mode v2) with a calibration file naming check and scores (tests/calib_support.py;
arbitrary parameters). Each result is compared with the library called in a separate process with the servers'
environment, with the same shape: the request's batch for HTTP, one message per call for MCP (padding in a batch
changes the last float digits). Then:
  HTTP: starts anyintent-serve (the console script) on the calibrated model with an API key and a lists folder; checks
        /health (short without the key, full with it, the supported functions), 401 without the key, PUT /lists (201,
        a JSON file in the folder, mode 0600), POST /classify with the list equal to AnyIntent.classify (label, index,
        score, margin exactly, and a runner_up), POST /check (a batch; a [name, description] pair with a threshold)
        equal to AnyIntent.check field by field, POST /scores (with intents, and with a saved list of 7 intents) equal
        to AnyIntent.scores in the servers' answer shape (same_scores: the label and the intents the model read with
        their values, highest first, no remainder), the Host guard (a JSON 400), the body cap with Content-Length and
        with a chunked body (413), a restart that reads the saved list again; then, on the model without a calibration
        file, without an API key on 0.0.0.0 (as the Docker image runs): a foreign Host is refused, 127.0.0.1 is served,
        /health lists classify only, /check and /scores answer 409 calibration_missing; and that the server log holds
        no message text;
  MCP:  starts anyintent-mcp over stdio on the calibrated model with the official MCP client; checks initialize, the
        tool list (five tools, plus check_intent and score_intents when the library has check and scores), save, list,
        get, classify with intents and with list_name, check_intent and score_intents (with intents and with
        list_name; score_intents as same_scores) equal to the library, an error result for intents and list_name
        together, delete, and that its stderr log holds no message text; then over streamable-http on 127.0.0.1 on the
        model without a calibration file: a client on /mcp lists the tools and classifies, check_intent and
        score_intents return error results that say the calibration is missing, and a request with a foreign Host is
        refused.
On Windows the file mode of the saved list is not checked (Windows ignores chmod bits apart from read-only); the run
prints it as SKIP and lists it under "skipped" in server_smoke.json.
Writes server_smoke.json in --out and exits 1 when a check fails.
"""
import argparse
import asyncio
import http.client
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(os.path.dirname(HERE), "tests")
SECRET = "SMOKE-SECRET-TEXT-41c9"
KEY = "smoke-key-5d1e"
THREADS = "2"          # the thread count of the servers and of the reference process: float sums depend on it
INTENTS = [{"name": "lost card", "description": "The customer reports that a bank card was lost or stolen."},
           {"name": "refund status", "description": "The customer asks when a refund will arrive."},
           "close account"]
# 7 intents: scores reads 5 of them, so the remainder (the intents not read) is not 0
WIDE = INTENTS + [{"name": "card blocked", "description": "A payment card was blocked or declined."},
                  {"name": "change password", "description": "The customer wants to change or reset the password."},
                  {"name": "opening hours", "description": "Questions about when a store or office is open."},
                  {"name": "invoice request", "description": "The customer asks for an invoice or a receipt."}]
MESSAGES = [SECRET + " I cannot find my card anywhere", "Where is my refund?", "Chcę zamknąć konto", "Kur mano pinigai?"]
PAIR = [INTENTS[1]["name"], INTENTS[1]["description"]]       # check's [name, description] form, with threshold 0.9
CHECK_FIELDS = ("intent", "yes", "probability", "score", "threshold")
CHECKS = {}
SKIPPED = {}           # parts of a check that cannot be made on this system (name: reason); not counted as checks


def check(name, ok, info=None):
    CHECKS[name] = {"ok": bool(ok), "info": info}
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {info}"), flush=True)
    return ok


def skipped(name, reason):
    SKIPPED[name] = reason
    print(f"SKIP {name}: {reason}", flush=True)


def build_tiny(out):
    """(the tiny one-file model without a calibration file, the tiny model at the settings of R2Intent v1 with a
    calibration file naming check and scores)"""
    sys.path.insert(0, TESTS)
    import conftest as C      # the test fixtures: offline tokenizer and tiny random XLM-R models
    import calib_support as S
    for d in ("tiny_one_file", "p_like", "p_like_r2", "tiny_calibrated"):
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)
    tok = C.build_tokenizer(os.path.join(out, "tok"))
    plain = C.make_one_file(os.path.join(out, "tiny_one_file"), tok)
    _, r2 = S.make_p_like(os.path.join(out, "p_like"), tok)
    return plain, S.model_copy(r2, os.path.join(out, "tiny_calibrated"))


def server_env(**extra):
    return dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS=THREADS, **extra)


REFERENCE = """
import json, sys, warnings
warnings.simplefilter("ignore")
from anyintent import AnyIntent
model, intents, wide, pair, fields, messages, calibrated = json.loads(sys.stdin.read())
clf = AnyIntent.from_pretrained(model, device="cpu")
clf.set_labels(intents)
row = lambda p: {"label": p.label, "index": p.index, "score": p.score, "margin": p.margin}
# a batch as the HTTP server gets it, and one message per call as the MCP tool gets it: padding to the longest message
# of a batch changes the float sums in the last digits, so each path is compared with the reference of its own shape
out = {"batch": [row(p) for p in clf.classify(messages)], "single": [row(clf.classify(m)) for m in messages]}
if calibrated:
    chk = lambda r: {k: getattr(r, k) for k in fields}
    sco = lambda r: {"label": r.label, "values": r.values, "remainder": r.remainder, "read": r.read}
    out["check_batch"] = [chk(r) for r in clf.check(messages, intents[0])]
    out["check_single"] = [chk(clf.check(m, intents[0])) for m in messages]
    out["check_pair"] = chk(clf.check(messages[1], tuple(pair), threshold=0.9))
    for name, labels in (("intents", intents), ("wide", wide)):
        clf.set_labels(labels)
        out["scores_batch_" + name] = [sco(r) for r in clf.scores(messages)]
        out["scores_single_" + name] = [sco(clf.scores(m)) for m in messages]
print(json.dumps(out))
"""


def library_reference(model, calibrated=False):
    """AnyIntent.classify (and, with calibrated, AnyIntent.check and AnyIntent.scores) in a fresh process with the
    servers' environment (thread counts change float sums, so the reference is computed as the servers compute)."""
    r = subprocess.run([sys.executable, "-c", REFERENCE], text=True, capture_output=True, env=server_env(), check=True,
                       input=json.dumps([model, INTENTS, WIDE, PAIR, CHECK_FIELDS, MESSAGES, calibrated]))
    return json.loads(r.stdout.strip().splitlines()[-1])


FOUR = ("label", "index", "score", "margin")


def same_as_library(results, ref):
    """label, index, score and margin exactly as AnyIntent.classify; a runner_up next to them."""
    return (isinstance(results, list) and len(results) == len(ref) and
            all(r is not None and {k: r.get(k) for k in FOUR} == want and "runner_up" in r
                for r, want in zip(results, ref)))


def same_results(results, ref):
    """every field of every check result exactly as the library's (intent, yes, probability, score, threshold), and no
    other field"""
    return isinstance(results, list) and len(results) == len(ref) and all(r == want for r, want in zip(results, ref))


def same_scores(results, ref):
    """every scores answer of the servers against the library's ScoresResult (ref: its label, values, remainder and
    read): only the keys label and intents (no remainder, no value for every intent); intents lists exactly the intents
    the model read (at most 5), each {intent, value} with the library's value (the same float), from the highest value
    to the lowest (ties: the label first, then retrieval order); label is the library's and the first one listed; the
    listed values plus the library's remainder sum to 1"""
    def one(r, w):
        if not isinstance(r, dict) or set(r) != {"label", "intents"} or not isinstance(r["intents"], list):
            return False
        items = r["intents"]
        if not items or len(items) > 5 or not all(isinstance(x, dict) and set(x) == {"intent", "value"} for x in items):
            return False
        names, vals = [x["intent"] for x in items], [x["value"] for x in items]
        pos = {n: i for i, n in enumerate(w["read"])}
        return (sorted(names) == sorted(w["read"]) and len(set(names)) == len(names) and
                all(type(v) is float and v == w["values"][n] for n, v in zip(names, vals)) and
                names == sorted(names, key=lambda n: (-w["values"][n], n != w["label"], pos[n])) and
                r["label"] == w["label"] == names[0] and abs(sum(vals) + w["remainder"] - 1.0) < 1e-9)
    return isinstance(results, list) and len(results) == len(ref) and all(one(r, w) for r, w in zip(results, ref))


def missing_calibration(status, body, feature):
    """the 409 of a model whose release does not ship the calibration of the function"""
    return (status == 409 and isinstance(body, dict) and body.get("error", {}).get("code") == "calibration_missing"
            and body["error"]["message"].startswith(f"'{feature}' is not available with this model")
            and "does not ship" in body["error"]["message"])


def request(port, method, path, body=None, headers=None, raw=None, chunks=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    h = {"Content-Type": "application/json", **(headers or {})}
    if chunks is not None:
        c.putrequest(method, path)
        for k, v in dict(h, **{"Transfer-Encoding": "chunked"}).items():
            c.putheader(k, v)
        c.endheaders()
        try:
            for part in chunks:
                c.send(b"%x\r\n%s\r\n" % (len(part), part))
            c.send(b"0\r\n\r\n")
        except OSError:
            pass                              # the server may answer 413 and close before the last chunk
    else:
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    text = r.read().decode("utf-8", "replace")
    c.close()
    try:
        return r.status, json.loads(text), dict(r.getheaders())
    except ValueError:
        return r.status, text, dict(r.getheaders())


def start_http(model, lists, port, log_path, key=KEY, extra=()):
    exe = shutil.which("anyintent-serve")
    cmd = ([exe] if exe else [sys.executable, "-m", "anyintent.server"]) + [
        "--model", model, "--device", "cpu", "--port", str(port), "--lists-dir", lists] + list(extra)
    env = server_env()
    env.pop("ANYINTENT_API_KEY", None)
    if key:
        env["ANYINTENT_API_KEY"] = key
    proc = subprocess.Popen(cmd, stdout=open(log_path, "a", encoding="utf-8"), stderr=subprocess.STDOUT, env=env)
    for _ in range(240):
        try:
            if request(port, "GET", "/health", headers={"Host": f"127.0.0.1:{port}"})[0] == 200:
                return proc, exe
        except OSError:
            pass
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    raise RuntimeError(f"anyintent-serve did not start (exit {proc.poll()}); see {log_path}")


def stop(proc):
    proc.terminate()
    try:
        return proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(30)


def http_smoke(model, plain, out, ref, ref_plain):
    """model: the calibrated model; plain: the model without a calibration file"""
    lists, log_path, port = os.path.join(out, "http_lists"), os.path.join(out, "http_server.log"), 8765
    shutil.rmtree(lists, ignore_errors=True)
    proc, exe = start_http(model, lists, port, log_path)
    check("http console script installed", exe is not None, exe)
    auth = {"Authorization": "Bearer " + KEY}
    try:
        s, b, _ = request(port, "GET", "/health")
        check("http health without the key is short", s == 200 and b == {"status": "ok", "ready": True}, b)
        s, b, _ = request(port, "GET", "/health", headers=auth)
        check("http health with the key", s == 200 and b.get("ready") and b.get("config", {}).get("layout") ==
              "one_file", b)
        check("http health lists the supported functions", s == 200 and b.get("supported") ==
              ["classify", "check", "scores"] and b.get("calibration") == {"file": True,
                                                                           "functions": ["check", "scores"]}, b)
        s, b, h = request(port, "POST", "/classify", {"text": "x", "intents": INTENTS})
        check("http 401 without the key", s == 401 and h.get("www-authenticate") == "Bearer", (s, b))
        s, b, _ = request(port, "PUT", "/lists/support", {"intents": INTENTS}, headers=auth)
        f = os.path.join(lists, "support.json")
        # Windows ignores chmod bits apart from read-only (st_mode of a writable file is always 0o666 there)
        mode_ok = os.name == "nt" or (os.path.exists(f) and (os.stat(f).st_mode & 0o777) == 0o600)
        check("http put list", s == 201 and os.path.exists(f) and mode_ok, (s, b))
        if os.name == "nt":
            skipped("http put list: file mode 0600", "Windows ignores chmod bits apart from read-only")
        s, b, _ = request(port, "POST", "/classify", {"texts": MESSAGES, "list": "support"}, headers=auth)
        check("http classify equals the library", s == 200 and same_as_library(b.get("results"), ref["batch"]),
              {"server": b, "library": ref["batch"]})
        s, b, _ = request(port, "POST", "/classify", {"text": MESSAGES[1], "intents": INTENTS, "k": 2}, headers=auth)
        check("http classify with k", s == 200 and len(b["results"]) == 1, b)
        s, b, _ = request(port, "POST", "/check", {"texts": MESSAGES, "intent": INTENTS[0]}, headers=auth)
        check("http check equals the library", s == 200 and same_results(b.get("results"), ref["check_batch"]),
              {"server": b, "library": ref["check_batch"]})
        s, b, _ = request(port, "POST", "/check", {"text": MESSAGES[1], "intent": PAIR, "threshold": 0.9}, headers=auth)
        check("http check with a [name, description] pair and a threshold equals the library",
              s == 200 and same_results(b.get("results"), [ref["check_pair"]]), {"server": b,
                                                                                 "library": ref["check_pair"]})
        s, b, _ = request(port, "POST", "/scores", {"texts": MESSAGES, "intents": INTENTS}, headers=auth)
        check("http scores equals the library",
              s == 200 and same_scores(b.get("results"), ref["scores_batch_intents"])
              and all(w["remainder"] == 0.0 for w in ref["scores_batch_intents"])
              and all(len(r["intents"]) == len(INTENTS) for r in b["results"]),
              {"server": b, "library": ref["scores_batch_intents"]})
        request(port, "PUT", "/lists/wide", {"intents": WIDE}, headers=auth)
        s, b, _ = request(port, "POST", "/scores", {"texts": MESSAGES, "list": "wide"}, headers=auth)
        check("http scores with a saved list of 7 intents equals the library",
              s == 200 and same_scores(b.get("results"), ref["scores_batch_wide"])
              and all(w["remainder"] > 0 and len(w["read"]) == 5 for w in ref["scores_batch_wide"])
              and all(len(r["intents"]) == 5 for r in b["results"]),
              {"server": b, "library": ref["scores_batch_wide"]})
        s, b, _ = request(port, "GET", "/health", headers={"Host": "evil.example"})
        check("http host guard", s == 400 and isinstance(b, dict) and b["error"]["code"] == "host_not_allowed",
              (s, b))
        big = b'{"text": "' + b"x" * 4_000_000 + b'"}'
        s, b, _ = request(port, "POST", "/classify", raw=big, headers=auth)
        check("http body cap (Content-Length)", s == 413, (s, str(b)[:200]))
        s, b, _ = request(port, "POST", "/classify", headers=auth, chunks=(b" " * 65536 for _ in range(64)))
        check("http body cap (chunked)", s == 413, (s, str(b)[:200]))
        s, b, _ = request(port, "POST", "/classify", {"text": "x", "intents": ["a", "a"]}, headers=auth)
        check("http validation error", s == 422 and b["error"]["code"] == "invalid_request", b)
    finally:
        rc = stop(proc)
    proc, _ = start_http(model, lists, port, log_path)
    try:
        s, b, _ = request(port, "GET", "/lists/support", headers=auth)
        check("http lists read again after a restart", s == 200 and len(b.get("intents", [])) == 3, (s, b))
    finally:
        stop(proc)
    # as the Docker image runs it: on 0.0.0.0, without an API key; only 127.0.0.1 and localhost are served. The model
    # here has no calibration file: check and scores answer 409, classify works
    proc, _ = start_http(plain, lists, port, log_path, key=None, extra=("--host", "0.0.0.0"))
    loop = {"Host": f"127.0.0.1:{port}"}
    try:
        s, b, _ = request(port, "GET", "/lists", headers={"Host": "evil.example"})
        check("http 0.0.0.0 without a key refuses a foreign Host", s == 400 and b["error"]["code"] ==
              "host_not_allowed", (s, b))
        s, b, _ = request(port, "GET", "/lists", headers=loop)
        check("http 0.0.0.0 without a key serves 127.0.0.1", s == 200 and "lists" in b, (s, b))
        s, b, _ = request(port, "GET", "/health", headers=loop)
        check("http health lists only classify without a calibration file", s == 200 and b.get("supported") ==
              ["classify"] and b.get("calibration") == {"file": False, "functions": []}, b)
        s, b, _ = request(port, "POST", "/check", {"text": MESSAGES[0], "intent": INTENTS[0]}, headers=loop)
        check("http check without a calibration file is a 409 calibration_missing",
              missing_calibration(s, b, "check"), (s, b))
        s, b, _ = request(port, "POST", "/scores", {"texts": MESSAGES, "list": "wide"}, headers=loop)
        check("http scores without a calibration file is a 409 calibration_missing",
              missing_calibration(s, b, "scores"), (s, b))
        s, b, _ = request(port, "POST", "/classify", {"texts": MESSAGES, "list": "support"}, headers=loop)
        check("http classify without a calibration file equals the library",
              s == 200 and same_as_library(b.get("results"), ref_plain["batch"]),
              {"server": b, "library": ref_plain["batch"]})
    finally:
        stop(proc)
    text = open(log_path, encoding="utf-8", errors="replace").read()
    check("http log holds no message text", SECRET not in text and all(
        f"{op}: 4 message(s)" in text for op in ("classify", "check", "scores")), text[-1500:])
    CHECKS["http_exit_code"] = {"ok": True, "info": rc}


async def mcp_session(model, out, ref):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from anyintent import AnyIntent
    exe = shutil.which("anyintent-mcp")
    check("mcp console script installed", exe is not None, exe)
    lists = os.path.join(out, "mcp_lists")
    shutil.rmtree(lists, ignore_errors=True)
    params = StdioServerParameters(command=exe or sys.executable,
                                   args=([] if exe else ["-m", "anyintent.mcp_server"]) +
                                   ["--model", model, "--device", "cpu", "--lists-dir", lists],
                                   env=server_env())
    errlog_path = os.path.join(out, "mcp_server.log")
    with open(errlog_path, "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                t0 = time.time()
                await session.initialize()
                CHECKS["mcp initialize seconds (information)"] = {"ok": True, "info": round(time.time() - t0, 2)}
                names = {t.name for t in (await session.list_tools()).tools}
                want = {"classify_intent", "save_intent_list", "get_intent_list", "list_intent_lists",
                        "delete_intent_list"}
                if hasattr(AnyIntent, "check"):
                    want.add("check_intent")
                if hasattr(AnyIntent, "scores"):
                    want.add("score_intents")
                check("mcp tool list", names == want, sorted(names))

                async def call(name, args):
                    r = await session.call_tool(name, args)
                    text = "".join(getattr(c, "text", "") for c in r.content)
                    return r.isError, text

                err, text = await call("save_intent_list", {"name": "support", "intents": INTENTS})
                check("mcp save list", not err and json.loads(text)["persistent"] is True and
                      os.path.exists(os.path.join(lists, "support.json")), text)
                err, text = await call("list_intent_lists", {})
                check("mcp list lists", not err and json.loads(text)["lists"] == [{"name": "support", "n_intents": 3}],
                      text)
                err, text = await call("get_intent_list", {"name": "support"})
                check("mcp get list", not err and len(json.loads(text)["intents"]) == 3, text)
                got_i, got_l = [], []
                for m in MESSAGES:
                    err, text = await call("classify_intent", {"message": m, "intents": INTENTS})
                    got_i.append(None if err else json.loads(text))
                    err, text = await call("classify_intent", {"message": m, "list_name": "support"})
                    got_l.append(None if err else json.loads(text))
                check("mcp classify equals the library", same_as_library(got_i, ref["single"]) and
                      same_as_library(got_l, ref["single"]), {"intents": got_i, "list": got_l, "library": ref["single"]})
                got_c, got_w, got_s = [], [], []
                for m in MESSAGES:
                    err, text = await call("check_intent", {"message": m, "intent": INTENTS[0]})
                    got_c.append(None if err else json.loads(text))
                    err, text = await call("score_intents", {"message": m, "intents": WIDE})
                    got_w.append(None if err else json.loads(text))
                    err, text = await call("score_intents", {"message": m, "list_name": "support"})
                    got_s.append(None if err else json.loads(text))
                check("mcp check_intent equals the library", same_results(got_c, ref["check_single"]),
                      {"server": got_c, "library": ref["check_single"]})
                check("mcp score_intents equals the library", same_scores(got_w, ref["scores_single_wide"]) and
                      same_scores(got_s, ref["scores_single_intents"]),
                      {"intents": got_w, "list": got_s, "library": [ref["scores_single_wide"],
                                                                    ref["scores_single_intents"]]})
                err, text = await call("classify_intent", {"message": "x", "intents": INTENTS, "list_name": "support"})
                check("mcp error result for intents and list_name together", err and "exactly one" in text, text)
                err, text = await call("delete_intent_list", {"name": "support"})
                check("mcp delete list", not err and not os.path.exists(os.path.join(lists, "support.json")), text)
    text = open(errlog_path, encoding="utf-8", errors="replace").read()
    check("mcp log holds no message text", SECRET not in text, text[-1500:])


async def mcp_http_session(model, out, ref):
    """anyintent-mcp --transport streamable-http on 127.0.0.1 on the model without a calibration file: a client on
    /mcp (classify equal to the library; check_intent and score_intents are error results), and the Host check."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    exe = shutil.which("anyintent-mcp")
    port = 8766
    cmd = ([exe] if exe else [sys.executable, "-m", "anyintent.mcp_server"]) + [
        "--model", model, "--device", "cpu", "--transport", "streamable-http", "--port", str(port)]
    log_path = os.path.join(out, "mcp_http_server.log")
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                            env=server_env())
    try:
        for _ in range(120):
            try:
                request(port, "GET", "/mcp", headers={"Host": f"127.0.0.1:{port}"})
                break
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError(f"anyintent-mcp over streamable-http stopped (exit {proc.poll()}); see "
                                       f"{log_path}")
                time.sleep(0.5)
        s, b, _ = request(port, "POST", "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                          headers={"Host": "evil.example", "Accept": "application/json, text/event-stream"})
        check("mcp http refuses a foreign Host", s == 400 and isinstance(b, dict) and b["error"]["code"] ==
              "host_not_allowed", (s, b))
        async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                check("mcp http tool list", "classify_intent" in names and "delete_intent_list" in names,
                      sorted(names))
                r = await session.call_tool("classify_intent", {"message": MESSAGES[0], "intents": INTENTS})
                got = None if r.isError else json.loads("".join(getattr(c, "text", "") for c in r.content))
                check("mcp http classify equals the library", same_as_library([got], ref["single"][:1]),
                      {"server": got, "library": ref["single"][:1]})
                errs = {}
                for name, feature, args in (("check_intent", "check", {"message": MESSAGES[0], "intent": INTENTS[0]}),
                                            ("score_intents", "scores", {"message": MESSAGES[0], "intents": WIDE})):
                    r = await session.call_tool(name, args)
                    errs[name] = (r.isError, "".join(getattr(c, "text", "") for c in r.content))
                    errs[name] += (f"'{feature}' is not available with this model" in errs[name][1],)
                check("mcp http check_intent and score_intents without a calibration file are error results",
                      all(e and found for e, _, found in errs.values()), errs)
    finally:
        stop(proc)
    text = open(log_path, encoding="utf-8", errors="replace").read()
    check("mcp http log holds no message text", SECRET not in text, text[-1500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-mcp", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    plain, model = build_tiny(a.out)
    ref = library_reference(model, calibrated=True)
    ref_plain = library_reference(plain)
    versions = {}
    import importlib.metadata as md
    for m in ("anyintent", "fastapi", "starlette", "pydantic", "uvicorn", "httpx", "mcp", "torch", "transformers"):
        try:
            versions[m] = md.version(m)
        except Exception:
            versions[m] = "missing"
    print("versions", json.dumps(versions), flush=True)
    try:
        http_smoke(model, plain, a.out, ref, ref_plain)
    except Exception as e:
        check("http smoke ran", False, f"{type(e).__name__}: {e}")
    if a.skip_mcp:
        CHECKS["mcp"] = {"ok": True, "info": "skipped by --skip-mcp"}
    else:
        try:
            asyncio.run(mcp_session(model, a.out, ref))
        except Exception as e:
            check("mcp smoke ran", False, f"{type(e).__name__}: {e}")
        try:
            asyncio.run(mcp_http_session(plain, a.out, ref_plain))
        except Exception as e:
            check("mcp http smoke ran", False, f"{type(e).__name__}: {e}")
    status = "PASS" if all(v["ok"] for v in CHECKS.values()) else "FAIL"
    doc = {"status": status, "versions": versions, "checks": CHECKS}
    if SKIPPED:
        doc["skipped"] = SKIPPED
    with open(os.path.join(a.out, "server_smoke.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=str)
    print(f"SERVER SMOKE {status}: {sum(v['ok'] for v in CHECKS.values())}/{len(CHECKS)} checks"
          + (f" ({len(SKIPPED)} skipped)" if SKIPPED else ""), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
