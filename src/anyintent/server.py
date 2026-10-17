"""HTTP server for AnyIntent (the 'server' extra: FastAPI and uvicorn).

    anyintent-serve --model lukashm/r2intent-v1 --lists-dir ./intent-lists      (add --revision TAG to pin the model)

One AnyIntent instance is loaded at start and shared by all requests (the library serialises its calls; a label set
and the classification that uses it run under the instance's lock). Endpoints, JSON in and out (the OpenAPI page is at
/docs):

    GET    /health          state, model id, revision, settings, limits, the functions the model supports (details
                            only with the API key, if one is set)
    POST   /classify        {"text" or "texts", "intents" or "list", "k"?} -> one {label, index, score, margin,
                            runner_up} per text (and doubtful, with --doubt-margin)
    POST   /check           {"text" or "texts", "intent", "threshold"?} -> one {intent, yes, probability, score,
                            threshold} per text (AnyIntent.check)
    POST   /scores          {"text" or "texts", "intents" or "list"} -> one {label, intents: [{intent, value}]} per
                            text: the intents the model read, highest value first (AnyIntent.scores)
    GET    /lists           the saved intent lists
    PUT    /lists/{name}    save a named intent list {"intents": [...]}
    GET    /lists/{name}    read one
    DELETE /lists/{name}    delete one

Security and privacy defaults: it listens on 127.0.0.1; an optional API key (--api-key or the environment variable
ANYINTENT_API_KEY) is checked as a bearer token with a constant-time compare; without a key, only requests addressed to
127.0.0.1 or localhost are accepted, wherever it listens (a guard against DNS rebinding); no CORS unless --cors-origin
is given; request bodies and the requests in progress are capped; message texts are never logged unless --log-texts;
an unexpected error returns a JSON 500 without a stack trace. /check and /scores answer 409 'calibration_missing' on a
model whose release does not ship the calibration of the function (classify still works), and 501 with a library
without the method.
"""
import argparse
import hmac
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

INSTALL_HINT = ('the HTTP server needs the server extra: pip install '
                '"anyintent[server] @ git+https://github.com/lukamaga/anyintent@v0.1.0"')

try:
    from fastapi import Depends, FastAPI, Response
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr
    from starlette.exceptions import HTTPException as StarletteHTTPException
except ImportError as e:   # pragma: no cover (exercised only without the extra)
    raise ImportError(INSTALL_HINT) from e

from ._version import __version__
from .service import (LABEL_TEXT_MODES, LOOPBACK_NAMES, Limits, Service, ServiceError, Unauthorized, allowed_name,
                      check_doubt_margin, host_allowed, host_name, is_loopback, scope_header, send_json_error)

log = logging.getLogger("anyintent.server")

API_DESCRIPTION = (
    "Intent classification with intents you choose at request time; no training examples of your intents are needed. "
    "Send messages and a list of intents (names, better with a one or two sentence description) or the name of a saved "
    "list; get the matching intent per message with its score, its margin and the runner-up intent. The margin is the "
    "score of the best intent minus that of the second best: a small margin means a doubtful answer. Scores and "
    "margins are in the model's raw units, not probabilities; when the server was started with --doubt-margin, each "
    "result also says whether it is doubtful. /check (yes or no for one intent, with a probability; a beta) and "
    "/scores (a value from 0 to 1 for each intent the model read with the message) give calibrated numbers, only with "
    "a model release that ships their calibration (409 'calibration_missing' otherwise; GET /health lists the "
    "functions the model supports). Runs on your machine; message texts are not stored and not logged unless the "
    "server was started with --log-texts.")

_STATUS_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
                 405: "method_not_allowed", 409: "conflict", 413: "limit_exceeded", 415: "unsupported_media_type",
                 422: "invalid_request", 500: "internal_error", 501: "not_available", 503: "not_ready"}
# answered even when --max-concurrency requests are in progress (liveness probes, the API description)
_UNLIMITED_PATHS = ("/health", "/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc")


# ---- request and response bodies --------------------------------------------------------------------------------
class IntentIn(BaseModel):
    """An intent: its name (returned as the label) and, better, a one or two sentence description."""
    model_config = ConfigDict(extra="forbid")
    name: StrictStr = Field(..., description="the intent's name; it is returned as the label",
                            examples=["lost card"])
    description: Optional[StrictStr] = Field(None, description="one or two sentences saying what the customer wants",
                                             examples=["The customer reports that a bank card was lost or stolen."])


Intent = Union[StrictStr, IntentIn]


class _Messages(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    text: Optional[StrictStr] = Field(None, description="one message (give text or texts)",
                                      examples=["I can't find my card anywhere"])
    texts: Optional[List[StrictStr]] = Field(None, description="several messages, answered in order (at most "
                                                               "--max-batch)")


class ClassifyIn(_Messages):
    intents: Optional[List[Intent]] = Field(None, description="the intents to choose from: names, or objects with "
                                                              "name and description (give intents or list)")
    list_name: Optional[StrictStr] = Field(None, alias="list", description="the name of a saved intent list "
                                                                           "(PUT /lists/{name})")
    k: Optional[StrictInt] = Field(None, description="how many intents the reranker reads per message; default: "
                                                     "the model's frozen k (the published numbers use it)")


class CheckIn(_Messages):
    intent: Union[StrictStr, IntentIn, List[StrictStr]] = Field(
        ..., description="the one intent to check: a name, an object with name and description, or a [name, "
                         "description] pair; write the description: the probabilities were calibrated on intents "
                         "written with one",
        examples=[{"name": "lost card", "description": "The customer reports that a bank card was lost or stolen."}])
    threshold: Optional[StrictFloat] = Field(None, description="yes means probability >= threshold, a number between 0 "
                                                               "and 1 (default 0.5, the only threshold measured)")


class ScoresIn(_Messages):
    intents: Optional[List[Intent]] = Field(None, description="the intents to score (give intents or list)")
    list_name: Optional[StrictStr] = Field(None, alias="list", description="the name of a saved intent list")


class ListIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intents: List[Intent] = Field(..., description="names, or objects with name and description")


class PredictionOut(BaseModel):
    label: str = Field(..., description="the chosen intent's name")
    index: int = Field(..., description="its position in the intent list")
    score: float = Field(..., description="fused score of the chosen intent (cross-encoder logit + w x cosine); not "
                                          "a probability")
    margin: Optional[float] = Field(None, description="score of the best minus the second-best scored intent (null "
                                                      "when only one was scored); small means doubtful")
    runner_up: Optional[str] = Field(None, description="the second-best scored intent (null when only one was "
                                                       "scored): the alternative to offer when the answer is doubtful")
    doubtful: Optional[bool] = Field(None, description="only when the server was started with --doubt-margin: true "
                                                       "when the margin is below it (false when there is no margin, "
                                                       "that is when only one intent was scored)")


class ClassifyOut(BaseModel):
    results: List[PredictionOut]


class CheckResultOut(BaseModel):
    intent: str = Field(..., description="the intent's name")
    yes: bool = Field(..., description="probability >= threshold")
    probability: float = Field(..., description="the chance that the message expresses this intent when a match and a "
                                                "non-match are taken as equally likely before reading it (calibrated "
                                                "on intents written with a description); lower in reality for an "
                                                "intent that is rare in your traffic")
    score: float = Field(..., description="the fused pair score (cross-encoder logit + w x cosine), the units of "
                                          "/classify's score; not a probability")
    threshold: float = Field(..., description="the threshold used")


class CheckOut(BaseModel):
    results: List[CheckResultOut]


class IntentValueOut(BaseModel):
    intent: str = Field(..., description="the intent's name")
    value: float = Field(..., description="the chance, from 0 to 1, that this intent is the right one (calibrated on "
                                          "intents written with a description)")


class ScoresResultOut(BaseModel):
    label: str = Field(..., description="the intent with the largest value (the first one listed): the label /classify "
                                        "returns without k")
    intents: List[IntentValueOut] = Field(..., description="the intents the model read with the message (at most 5; "
                                                           "every intent when there are 5 or fewer), from the highest "
                                                           "value to the lowest. An intent not listed was not read: its "
                                                           "value is 0, which means not read, not impossible; it can "
                                                           "still be the right intent. With 5 intents or fewer the "
                                                           "values sum to 1; with more they can sum to less than 1, "
                                                           "because the right intent can be one of the intents the "
                                                           "model did not read. The part missing to reach 1 is not a "
                                                           "'none of these intents' answer (there is no such mode): "
                                                           "the values assume that one of the given intents is right")


class ScoresOut(BaseModel):
    results: List[ScoresResultOut]


class HealthOut(BaseModel):
    """Without the API key (when one is set) only status and ready."""
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    status: str = Field(..., description="ok, loading or error")
    ready: bool
    model: Optional[str] = Field(None, description="the model id (for a local folder only its name)")
    revision: Optional[str] = None
    library_version: Optional[str] = None
    features: Optional[Dict[str, bool]] = Field(None, description="classify, and whether the installed library has "
                                                                  "check and scores")
    supported: Optional[List[str]] = Field(None, description="the functions the loaded model answers (null while the "
                                                             "model loads): classify, and check and scores when the "
                                                             "library has them and the model's calibration file names "
                                                             "them; a call can still be refused with 422 (for example "
                                                             "a calibration file of another model or a changed model "
                                                             "folder)")
    calibration: Optional[Dict[str, Any]] = Field(None, description="the model's calibration file: whether there is "
                                                                    "one the library can use, and the functions it "
                                                                    "names")
    limits: Optional[Dict[str, int]] = None
    lists: Optional[Dict[str, Any]] = None
    label_text_mode: Optional[str] = Field(None, description="the label text mode set with --label-text-mode (null: "
                                                             "the model's own)")
    doubt_margin: Optional[float] = None
    log_texts: Optional[bool] = None
    config: Optional[Dict[str, Any]] = Field(None, description="the model's settings (config['active']: those in use "
                                                               "for the current label set)")
    error: Optional[str] = None


class IntentOut(BaseModel):
    name: str
    description: Optional[str] = None


class ListSavedOut(BaseModel):
    name: str
    n_intents: int
    created: bool = Field(..., description="true for a new list, false when it replaced one")
    persistent: bool = Field(..., description="true when it is kept on disk (--lists-dir)")


class ListOut(BaseModel):
    name: str
    intents: List[IntentOut]


class ListSummaryOut(BaseModel):
    name: str
    n_intents: int


class ListsOut(BaseModel):
    lists: List[ListSummaryOut]
    persistent: bool


class DeletedOut(BaseModel):
    name: str
    deleted: bool


CHECK_DESCRIPTION = (
    "AnyIntent.check: whether each message expresses one given intent. One result per message, in order: `yes` "
    "(`probability` >= `threshold`, default 0.5, the only threshold measured), `probability` (the chance that the "
    "message expresses this intent when a match and a non-match are taken as equally likely before reading it; if the "
    "intent is rare in your traffic, the real chance is lower), `score` (the fused pair score, the units of "
    "/classify's score, not a probability), `intent` and `threshold`. The intent is scored on its own: the answer does "
    "not depend on other intents. The probabilities were calibrated on intents written with a description: give one. "
    "Beta: check shipped in R2Intent v1 after it missed one pre-registered calibration limit on the DEV data, so treat "
    "its probability as approximate. Only with a model release that ships the calibration of check: otherwise 409 "
    "`calibration_missing` (classify still works); 501 when the installed library has no check.")
SCORES_DESCRIPTION = (
    "AnyIntent.scores: for each message, the intents the model read with it, each with a value from 0 to 1: the chance "
    "that it is the right intent. One result per message, in order: `intents` (a list of objects "
    '`{"intent": name, "value": number}`, from the highest value to the lowest) and `label` (the intent with the '
    "largest value, the first one listed: the one /classify returns without `k`). The model reads at most 5 intents "
    "with a message (every intent when there are 5 or fewer), and `intents` lists only those. An intent not listed "
    "was not read: its value is 0, which means not read, not impossible; it can still be the right intent. With 5 "
    "intents or fewer every intent is listed and the values sum to 1. With more, the values can sum to less than 1, "
    "because the right intent can be one of the intents the model did not read. The part missing to reach 1 is not a "
    "'none of these intents' answer (there is no such mode): the values assume that one of the given intents is "
    "right. The values were calibrated on intents written with a description: give one for each intent. Only with a "
    "model release that ships the calibration of scores: otherwise 409 `calibration_missing` (classify still works); "
    "501 when the installed library has no scores.")


def _plain(item):
    if isinstance(item, (str, list)):
        return item
    return item.model_dump(exclude_none=True)


def _plain_list(items):
    return None if items is None else [_plain(x) for x in items]


def _error(status, code, message, headers=None, details=None):
    err = {"code": code, "message": message}
    if details:
        err["details"] = details
    return JSONResponse({"error": err}, status_code=status, headers=headers)


_send_error = send_json_error


class _Guard:
    """The outermost safety net (pure ASGI), in this order: the Host check (allowed_hosts None: any Host; a JSON 400
    otherwise), a cap on the requests in progress (a JSON 503 'busy' beyond max_concurrency; /health and the API
    description are not counted), a request-body cap (on Content-Length and on the bytes received), and any
    unexpected error becomes a JSON 500 without a stack trace; the log gets the error's type and traceback frames but
    not its message (it may hold a message text) unless --log-texts."""

    def __init__(self, app, service, max_body, allowed_hosts=None, max_concurrency=None):
        self.app, self.service, self.max_body = app, service, int(max_body)
        self.allowed = None if allowed_hosts is None else sorted({allowed_name(a) for a in allowed_hosts})
        self.max_concurrency = None if max_concurrency is None else int(max_concurrency)
        self.in_progress = 0      # changed only on the event loop's thread

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if not host_allowed(scope_header(scope, b"host"), self.allowed):
            name = host_name(scope_header(scope, b"host"))[:100] or "<name>"
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            await _send_error(send, 400, "host_not_allowed",
                              f"the Host header {name!r} is not one this server accepts (a guard against DNS "
                              f"rebinding from web pages); start the server with --allowed-host {name} to accept it "
                              "(to serve other machines, set an API key too: ANYINTENT_API_KEY)")
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        counted = self.max_concurrency is not None and scope.get("path") not in _UNLIMITED_PATHS
        if counted and self.in_progress >= self.max_concurrency:
            await _send_error(send, 503, "busy", f"{self.in_progress} requests are in progress, the limit of this "
                                                 "server (--max-concurrency); try again shortly",
                              headers=[("retry-after", "1")])
            return
        if counted:
            self.in_progress += 1
        try:
            await self._handle(scope, receive, send)
        finally:
            if counted:
                self.in_progress -= 1

    async def _handle(self, scope, receive, send):
        started = False

        async def send_tracked(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        for key, value in scope.get("headers") or []:
            if key == b"content-length":
                try:
                    n = int(value)
                except ValueError:
                    n = -1
                if n < 0:
                    await _send_error(send, 400, "bad_request", "invalid Content-Length header")
                    return
                if n > self.max_body:
                    await _send_error(send, 413, "limit_exceeded", f"the request body has {n} bytes; the limit is "
                                                                   f"{self.max_body} bytes")
                    return
        seen = 0

        async def receive_capped():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body") or b"")
                if seen > self.max_body:
                    raise StarletteHTTPException(413, f"the request body is larger than the limit of {self.max_body} "
                                                      "bytes")
            return message

        try:
            await self.app(scope, receive_capped, send_tracked)
        except StarletteHTTPException as e:
            if not started:
                await _send_error(send, e.status_code, _STATUS_CODES.get(e.status_code, "http_error"), str(e.detail))
        except Exception as e:
            self.service.log_internal_error(f"{scope.get('method')} {scope.get('path')}", e)
            if not started:
                await _send_error(send, 500, "internal_error", "internal error; the server log has the details")


def create_app(service, api_key=None, cors_origins=(), allowed_hosts=None, max_concurrency=64):
    """The FastAPI application around a Service (anyintent.service.Service). allowed_hosts: the Host header names
    accepted (None: any; the command line decides with allowed_hosts_for, a guard against DNS rebinding from web
    pages). max_concurrency: requests in progress at once (None: no cap); more get a JSON 503."""
    if not isinstance(service, Service):
        raise TypeError("create_app needs an anyintent.service.Service")
    app = FastAPI(title="AnyIntent", version=__version__, description=API_DESCRIPTION)
    bearer = HTTPBearer(auto_error=False, description="the server's API key, only when the server was started with "
                                                      "one (--api-key or ANYINTENT_API_KEY)")
    secret = api_key.encode("utf-8") if api_key else None

    def key_ok(cred):
        if secret is None:
            return True
        given = cred.credentials.encode("utf-8") if cred is not None else b""
        return hmac.compare_digest(given, secret)

    def require_key(cred: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
        if not key_ok(cred):
            raise Unauthorized("missing or wrong API key: send the header 'Authorization: Bearer <key>'")

    protected = [Depends(require_key)]

    @app.exception_handler(ServiceError)
    async def _service_error(request, exc):
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        return _error(exc.status, exc.code, exc.message, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):
        details = [{"loc": [str(x) for x in e.get("loc", ())], "msg": str(e.get("msg", ""))} for e in exc.errors()]
        return _error(422, "invalid_request", "the request does not match the expected JSON body (see details and "
                                              "/docs)", details=details[:20])

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc):
        return _error(exc.status_code, _STATUS_CODES.get(exc.status_code, "http_error"), str(exc.detail),
                      headers=getattr(exc, "headers", None))

    @app.get("/health", response_model=HealthOut, response_model_exclude_unset=True, tags=["service"],
             summary="State, model, settings and limits")
    async def health(cred: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
        """Always answers, also while the model is busy (for liveness probes). With an API key set, the details need
        the key."""
        return service.health(detail=key_ok(cred))

    @app.post("/classify", response_model=ClassifyOut, response_model_exclude_unset=True, dependencies=protected,
              tags=["intents"], summary="Choose the matching intent for each message")
    def classify(req: ClassifyIn):
        msgs = service.messages(req.text, req.texts)
        return {"results": service.classify(msgs, intents=_plain_list(req.intents), list_name=req.list_name, k=req.k)}

    missing = {"description": "the model's release does not ship the calibration of this function (error code "
                              "calibration_missing; GET /health lists the functions the model supports)"}

    @app.post("/check", response_model=CheckOut, dependencies=protected, tags=["intents"],
              summary="Does each message express this one intent? Yes or no, with a probability (beta)",
              description=CHECK_DESCRIPTION,
              responses={409: missing, 501: {"description": "the installed library has no AnyIntent.check"}})
    def check(req: CheckIn):
        service.require("check")
        msgs = service.messages(req.text, req.texts)
        return {"results": service.check(msgs, _plain(req.intent), threshold=req.threshold)}

    @app.post("/scores", response_model=ScoresOut, dependencies=protected, tags=["intents"],
              summary="A value from 0 to 1 for each intent the model read with the message",
              description=SCORES_DESCRIPTION,
              responses={409: missing, 501: {"description": "the installed library has no AnyIntent.scores"}})
    def scores(req: ScoresIn):
        service.require("scores")
        msgs = service.messages(req.text, req.texts)
        return {"results": service.scores(msgs, intents=_plain_list(req.intents), list_name=req.list_name)}

    @app.get("/lists", response_model=ListsOut, dependencies=protected, tags=["lists"],
             summary="The saved intent lists")
    def lists():
        return service.list_lists()

    @app.put("/lists/{name}", response_model=ListSavedOut, dependencies=protected, tags=["lists"],
             summary="Save (or replace) a named intent list", responses={201: {"model": ListSavedOut,
                                                                               "description": "a new list"}})
    def put_list(name: str, body: ListIn, response: Response):
        r = service.save_list(name, _plain_list(body.intents))
        response.status_code = 201 if r["created"] else 200
        return r

    @app.get("/lists/{name}", response_model=ListOut, response_model_exclude_unset=True, dependencies=protected,
             tags=["lists"], summary="Read a saved intent list")
    def get_list(name: str):
        return service.get_list(name)

    @app.delete("/lists/{name}", response_model=DeletedOut, dependencies=protected, tags=["lists"],
                summary="Delete a saved intent list")
    def delete_list(name: str):
        return service.delete_list(name)

    if cors_origins:
        from starlette.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=list(cors_origins), allow_methods=["GET", "POST", "PUT",
                                                                                            "DELETE"],
                           allow_headers=["Authorization", "Content-Type"])
    # added last, so it runs first: the Host check comes before CORS and before everything else
    app.add_middleware(_Guard, service=service, max_body=service.limits.max_body_bytes, allowed_hosts=allowed_hosts,
                       max_concurrency=max_concurrency)
    return app


# ---- command line -----------------------------------------------------------------------------------------------
def env(name):
    """An environment variable; empty counts as not set."""
    return os.environ.get(name) or None


def build_parser():
    p = argparse.ArgumentParser(prog="anyintent-serve", description="HTTP server for AnyIntent intent classification "
                                "(one model, loaded at start). The OpenAPI page is at /docs.")
    p.add_argument("--model", default=env("ANYINTENT_MODEL"),
                   help="Hugging Face model id or local folder, for example lukashm/r2intent-v1 (default: the "
                        "environment variable ANYINTENT_MODEL)")
    p.add_argument("--revision", default=env("ANYINTENT_REVISION"),
                   help="tag or commit of the model (pin one for reproducible results; default: ANYINTENT_REVISION)")
    p.add_argument("--device", default=None, help="cpu, cuda or cuda:N (default: cuda when available, else cpu)")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1, this machine only)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None,
                   help="require 'Authorization: Bearer <key>' (prefer the environment variable ANYINTENT_API_KEY: a "
                        "command-line value is visible in the process list)")
    p.add_argument("--lists-dir", default=env("ANYINTENT_LISTS_DIR"),
                   help="folder for saved intent lists, one JSON file each, read at start (default: "
                        "ANYINTENT_LISTS_DIR; without it, lists live in memory until the server stops)")
    p.add_argument("--max-batch", type=int, default=Limits.max_batch, help="messages per request (default 64)")
    p.add_argument("--max-text-chars", type=int, default=Limits.max_text_chars,
                   help="characters per message (default 2000)")
    p.add_argument("--max-intents", type=int, default=Limits.max_intents,
                   help="intents per request or saved list (default 200)")
    p.add_argument("--max-concurrency", type=int, default=64,
                   help="requests in progress at once; more get 503 'busy' (default 64; /health is not counted)")
    p.add_argument("--label-text-mode", default=env("ANYINTENT_LABEL_TEXT_MODE"), choices=LABEL_TEXT_MODES,
                   help="the label text mode passed to set_labels (default: ANYINTENT_LABEL_TEXT_MODE, else the "
                        "model's own, shown in /health as config.label_text_mode). v2 for plain descriptions on a "
                        "model released with v3s (see the README section 'Writing labels')")
    p.add_argument("--doubt-margin", type=float, default=None,
                   help="add 'doubtful' to every result: true when its margin is below this value. Choose it on your "
                        "own labelled examples for this model and these settings (no default)")
    p.add_argument("--log-texts", action="store_true",
                   help="log message texts and labels (off by default, for privacy)")
    p.add_argument("--cors-origin", action="append", default=[], metavar="ORIGIN",
                   help="allow browser requests from this origin (repeatable; no CORS by default)")
    p.add_argument("--allowed-host", action="append", default=[], metavar="NAME",
                   help="also accept requests addressed to this Host name (repeatable; '*' accepts any). Without an "
                        "API key, only 127.0.0.1 and localhost are accepted by default, a guard against DNS rebinding; "
                        "with a key, any Host on a non-loopback address")
    p.add_argument("--warm-up", action="store_true",
                   help="run a few dummy messages before serving, so the first request is not the slow one")
    p.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    return p


def _load_model(args):
    from .model import AnyIntent
    return AnyIntent.from_pretrained(args.model, revision=args.revision, device=args.device)


def allowed_hosts_for(args, api_key):
    """The Host names the server accepts (None: any). The loopback names are always accepted (a page that rebinds its
    own name sends that name, never 127.0.0.1 or localhost; the image's health check calls 127.0.0.1). '*' accepts any
    Host. Without --allowed-host: only the loopback names when listening on a loopback address or when there is no API
    key (a server on 0.0.0.0, as in Docker, without a key); any Host when there is a key on a non-loopback address."""
    names = list(args.allowed_host or [])
    if "*" in names:
        return None
    if names:
        return list(LOOPBACK_NAMES) + names
    if is_loopback(args.host) or not api_key:
        return list(LOOPBACK_NAMES)
    return None


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.model:
        log.error("no model: pass --model (for example lukashm/r2intent-v1) or set ANYINTENT_MODEL")
        return 2
    try:
        limits = Limits(max_batch=args.max_batch, max_text_chars=args.max_text_chars, max_intents=args.max_intents)
        check_doubt_margin(args.doubt_margin)
        if args.max_concurrency < 1:
            raise ValueError(f"--max-concurrency: expected an integer >= 1, got {args.max_concurrency}")
    except ValueError as e:
        log.error("%s", e)
        return 2
    try:
        import uvicorn
    except ImportError:
        log.error("%s", INSTALL_HINT)
        return 2
    if args.api_key:
        log.warning("--api-key on the command line is visible in the process list; prefer ANYINTENT_API_KEY")
    api_key = args.api_key or os.environ.get("ANYINTENT_API_KEY") or None
    log.info("loading %s (revision %s)", args.model, args.revision or "default")
    from .hub import AnyIntentError
    try:
        clf = _load_model(args)
    except AnyIntentError as e:          # access, revision, configuration: the library's message says what to do
        log.error("the model could not be loaded: %s", e)
        return 1
    except Exception:                    # for example --device cuda without a visible GPU
        log.exception("the model could not be loaded")
        return 1
    if args.warm_up:
        log.info("warm-up: %.1f s", clf.warm_up())
    try:
        service = Service(clf, model_id=args.model, lists_dir=args.lists_dir, limits=limits, log_texts=args.log_texts,
                          label_text_mode=args.label_text_mode, doubt_margin=args.doubt_margin)
    except OSError as e:
        log.error("the intent list folder %s cannot be used: %s", args.lists_dir, e)
        return 2
    allowed = allowed_hosts_for(args, api_key)
    app = create_app(service, api_key=api_key, cors_origins=args.cors_origin, allowed_hosts=allowed,
                     max_concurrency=args.max_concurrency)
    if not is_loopback(args.host) and not api_key:
        if allowed is None:
            log.warning("listening on %s without an API key and with --allowed-host '*': anyone who can reach this "
                        "port can use the model", args.host)
        else:
            log.info("listening on %s without an API key: requests are accepted only when addressed to %s (to serve "
                     "other machines set ANYINTENT_API_KEY, or name the host with --allowed-host)", args.host,
                     ", ".join(allowed))
    if args.log_texts:
        log.warning("--log-texts is on: message texts and labels are written to the log")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
