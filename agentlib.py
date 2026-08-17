"""agentlib.py - minimal agent library for OpenAI-compatible APIs.

Single-file module. Lets an agent writer register tools, run an agent loop
against any OpenAI-compatible chat/completions endpoint (DeepSeek, OpenAI,
Fireworks, local, ...), and capture the full trajectory of messages - exactly
in the format exchanged with the API - into a SQLite database.

Config is loaded from the existing .nearai.* JSON files:

    {"host": "https://api.deepseek.com/chat/completions",
     "model": "deepseek-v4-flash",
     "apikey": "sk-...",
     "context_window": 160000,
     "merge_reasoning": false}

Quick start:

    import agentlib

    cfg  = agentlib.load_config("~/.nearai.deepseek.flash")
    reg  = agentlib.ToolRegistry()

    @reg.tool("add", "Add two numbers.", {"type":"object",
              "properties":{"a":{"type":"number"},"b":{"type":"number"}},
              "required":["a","b"]})
    def add(a, b): return a + b

    agent = agentlib.Agent(cfg, reg, db_path="traces.db", agent_name="calc")
    result = agent.run("What is 2+2?")
    print(result.finish_reason, agent.dump_trajectory(result.trajectory_id))

# Safe bash

The library also provides a `register_safe_bash()` that registers a `bash`
tool whose every invocation is sandboxed by a cgroup with hard limits on
memory, tasks (forks), CPU, and wall-clock time. A single runaway command
cannot OOM the host. Hosts with cgroup v2 + systemd-run --user use the
systemd path; everything else falls back to `prlimit` + `unshare`. Use this
in place of the per-harness `_make_bash_tool` definitions:

    reg = agentlib.ToolRegistry()
    agentlib.register_safe_bash(reg, workdir="/path/to/run")

For whole-of-agent sandboxing, call `enter_agent_cgroup()` as the FIRST
statement in each harness's main() — before any print or file open, since it
re-execs the process into its cgroup. It puts the agent in a per-agent slice
under one fleet-wide slice, and routes every later bash scope into the same
per-agent slice. The fleet slice is the actual host guarantee: all agents
together are capped at ~75% of RAM (override with CURRIC_FLEET_RAM_MB), so no
number of concurrent agents can OOM the box.

    agentlib.enter_agent_cgroup("lean-b1", max_ram_mb=1024)
    agentlib.refuse_if_low_ram(need_mb=1024)  # pre-flight, optional
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Union

import requests


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class AgentConfig:
    """Connection details for an OpenAI-compatible chat/completions endpoint."""
    host: str                       # full chat/completions URL
    model: str
    apikey: str
    context_window: int = 128000
    merge_reasoning: bool = False
    # Context compaction: the fraction of context_window at which the whole
    # history is replaced by one summary of it. 0 disables it.
    compact_at: float = 0.8


def load_config(path: str | os.PathLike) -> AgentConfig:
    """Read a .nearai.* JSON file into an AgentConfig. ~ is expanded."""
    path = os.path.expanduser(str(path))
    with open(path, "r") as f:
        data = json.load(f)
    return AgentConfig(
        host=data["host"],
        model=data["model"],
        apikey=data["apikey"],
        context_window=data.get("context_window", 128000),
        merge_reasoning=data.get("merge_reasoning", False),
        compact_at=data.get("compact_at", 0.8),
    )


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # JSON schema describing arguments
    fn: Callable[..., Any]


class ToolRegistry:
    """Holds tools and renders them in the OpenAI `tools` request format."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # -- registration ------------------------------------------------------- #
    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        self._tools[name] = Tool(name, description, parameters, fn)
        return fn

    def tool(self, name: str, description: str, parameters: dict):
        """Decorator equivalent of register()."""
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name, description, parameters, fn)
            return fn
        return deco

    # -- rendering / dispatch ---------------------------------------------- #
    def to_openai(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def invoke(self, name: str, arguments: str):
        """Execute a tool by name; arguments is the JSON string from the model.

        Returns a string for ordinary tools. A tool may instead return a LIST
        of OpenAI content parts (e.g. text + image_url) to produce a
        multimodal tool result; the list is passed through untouched and
        becomes the tool message's `content`. Only meaningful with a
        multimodal model -- a text-only model will reject the request.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name!r}")
        args = json.loads(arguments) if arguments else {}
        result = tool.fn(**args)
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            return result          # content parts, verbatim
        # Anything else: JSON-encode so the content is still a string.
        return json.dumps(result, default=str)


# --------------------------------------------------------------------------- #
# SQLite trajectory store
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    id            TEXT PRIMARY KEY,
    agent_name    TEXT,
    model         TEXT,
    host          TEXT,
    context_window INTEGER,
    merge_reasoning INTEGER,
    system_prompt TEXT,
    user_input    TEXT,
    final_result  TEXT,
    finish_reason TEXT,
    num_steps     INTEGER,
    created_at    TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id            TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    step_idx      INTEGER NOT NULL,
    seq           INTEGER NOT NULL,
    role          TEXT NOT NULL,
    payload       TEXT NOT NULL,   -- exact OpenAI-format message dict, as JSON
    created_at    TEXT NOT NULL,
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_traj ON messages(trajectory_id);
"""


class TrajectoryStore:
    """SQLite-backed store for trajectories and their messages."""

    def __init__(self, db_path: str | os.PathLike) -> None:
        self.db_path = os.path.expanduser(str(db_path))
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- low-level --------------------------------------------------------- #
    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _insert_message(
        self, trajectory_id: str, step_idx: int, seq: int, message: dict
    ) -> str:
        msg_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO messages(id, trajectory_id, step_idx, seq, role, payload, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (
                msg_id,
                trajectory_id,
                step_idx,
                seq,
                message.get("role", "unknown"),
                json.dumps(message, default=str, ensure_ascii=False),
                self._now(),
            ),
        )
        return msg_id

    # -- public ------------------------------------------------------------ #
    def create_trajectory(
        self, agent_name: str, config: AgentConfig,
        system_prompt: Optional[str], user_input: Union[str, list[dict]],
    ) -> str:
        # Normalize user_input to a JSON string if it's a multimodal content
        # list, so the trajectories.user_input TEXT column always holds a str.
        user_input_str = (
            user_input if isinstance(user_input, str)
            else json.dumps(user_input, ensure_ascii=False, default=str)
        )
        tid = str(uuid.uuid4())
        cfg_snapshot = json.dumps(
            {
                "agent_name": agent_name,
                "model": config.model,
                "host": config.host,
                "context_window": config.context_window,
                "merge_reasoning": config.merge_reasoning,
            },
            ensure_ascii=False,
        )
        self._conn.execute(
            "INSERT INTO trajectories"
            "(id, agent_name, model, host, context_window, merge_reasoning,"
            " system_prompt, user_input, final_result, finish_reason, num_steps, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,NULL,NULL,0,?)",
            (
                tid, agent_name, config.model, config.host,
                config.context_window, int(config.merge_reasoning),
                system_prompt, user_input_str, self._now(),
            ),
        )
        # store the unused-but-handy full snapshot in a column? keep lean; ignore.
        _ = cfg_snapshot
        self._conn.commit()
        return tid

    def log_message(
        self, trajectory_id: str, step_idx: int, seq: int, message: dict
    ) -> str:
        mid = self._insert_message(trajectory_id, step_idx, seq, message)
        self._conn.commit()
        return mid

    def finalize_trajectory(
        self, trajectory_id: str, final_result: Optional[str],
        finish_reason: str, num_steps: int,
    ) -> None:
        self._conn.execute(
            "UPDATE trajectories SET final_result=?, finish_reason=?, num_steps=?"
            " WHERE id=?",
            (final_result, finish_reason, num_steps, trajectory_id),
        )
        self._conn.commit()

    # -- retrieval --------------------------------------------------------- #
    def load_messages(self, trajectory_id: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT payload FROM messages WHERE trajectory_id=? ORDER BY seq ASC",
            (trajectory_id,),
        )
        return [json.loads(row[0]) for row in cur.fetchall()]

    def load_trajectory(self, trajectory_id: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT id, agent_name, model, host, context_window, merge_reasoning,"
            " system_prompt, user_input, final_result, finish_reason, num_steps, created_at"
            " FROM trajectories WHERE id=?",
            (trajectory_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = (
            "id", "agent_name", "model", "host", "context_window", "merge_reasoning",
            "system_prompt", "user_input", "final_result", "finish_reason",
            "num_steps", "created_at",
        )
        return dict(zip(keys, row))

    def list_trajectories(self, limit: int = 100) -> list[dict]:
        cur = self._conn.execute(
            "SELECT id, agent_name, model, finish_reason, num_steps, created_at"
            " FROM trajectories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        keys = ("id", "agent_name", "model", "finish_reason", "num_steps", "created_at")
        return [dict(zip(keys, row)) for row in cur.fetchall()]

    # -- lifecycle --------------------------------------------------------- #
    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------- #
# Streaming reassembly
# --------------------------------------------------------------------------- #


# How much of a failed response body to keep. Providers put the actual reason
# in there -- "image file is truncated", "context length exceeded", "invalid
# tool schema" -- and recording only the status code turns a five-second
# diagnosis into an afternoon of reproducing the request by hand.
_ERR_BODY_CHARS = 2000


def _error_detail(resp) -> str:
    """The provider's own error text, best effort, never raising.

    The response is streamed, so the body has not been consumed on an error
    path and can still be read. A provider that answers with JSON gets its
    `error.message` lifted out; anything else is kept verbatim and truncated.
    """
    if resp is None:
        return ""
    try:
        body = resp.text
    except Exception:                                  # noqa: BLE001
        return ""
    if not body:
        return ""
    try:
        d = json.loads(body)
        err = d.get("error", d)
        if isinstance(err, dict):
            msg = err.get("message") or err.get("detail") or ""
            if msg:
                extra = err.get("code") or err.get("type") or ""
                return f"{msg}" + (f" (code={extra})" if extra else "")
    except Exception:                                  # noqa: BLE001
        pass
    return body[:_ERR_BODY_CHARS].strip()


def _stream_chat(
    config: AgentConfig, messages: list[dict], tools: list[dict],
    request_kwargs: Optional[dict] = None,
    connect_timeout: float = 10.0,
    read_timeout: float = 120.0,
    max_retries: int = 3,
) -> tuple[dict, str]:
    """POST a streaming chat completion and reassemble a single assistant
    message dict (OpenAI format). Returns (message, finish_reason).

    Tool-call fragments are merged by their `index`. `reasoning_content`
    (DeepSeek) is folded into `content` when merge_reasoning is True, else
    preserved as a separate `reasoning_content` field.

    Timeouts: `connect_timeout` seconds to establish the TCP connection,
    `read_timeout` seconds of silence between any two bytes on the stream.
    On a connect/read timeout or transient connection error, the call is
    retried up to `max_retries` times with exponential backoff; a raised
    exception after exhausting retries is returned as an assistant message
    with finish_reason="error" so the loop can continue or stop cleanly.
    """
    import time as _time

    headers = {
        "Authorization": f"Bearer {config.apikey}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    body: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    if request_kwargs:
        body.update(request_kwargs)

    last_err: Optional[str] = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = "stop"

    for attempt in range(max_retries + 1):
        content_parts.clear()
        reasoning_parts.clear()
        tool_calls.clear()
        finish_reason = "stop"
        try:
            resp = requests.post(
                config.host, headers=headers, json=body, stream=True,
                timeout=(connect_timeout, read_timeout),
            )
            resp.raise_for_status()

            # Stream processing is inside the try so that a mid-stream
            # ChunkedEncodingError or ConnectionError triggers a retry.
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].lstrip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {}) or {}

                c = delta.get("content")
                if c:
                    content_parts.append(c)
                r = delta.get("reasoning_content")
                if r:
                    reasoning_parts.append(r)

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(
                        idx,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    if tc.get("type"):
                        slot["type"] = tc["type"]
                    fn = tc.get("function", {}) or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]

                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
            # Stream completed successfully — break out of retry loop.
            break
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                backoff = 2 ** attempt
                _time.sleep(backoff)
                continue
            return (
                {"role": "assistant", "content": "",
                 "error": f"[llm call failed after {max_retries+1} attempts: {last_err}]",
                 "error_body": _error_detail(getattr(e, "response", None))},
                "error",
            )
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            detail = _error_detail(getattr(e, "response", None))
            if status == 429 and attempt < max_retries:
                backoff = 2 ** attempt
                _time.sleep(backoff)
                continue
            return (
                {"role": "assistant", "content": "",
                 "error": f"[HTTP {status}: {e}]"
                          + (f" -- {detail}" if detail else ""),
                 "error_status": status,
                 "error_body": detail},
                "error",
            )
    else:
        return (
            {"role": "assistant", "content": "",
             "error": f"[exhausted retries: {last_err}]"},
            "error",
        )

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        if config.merge_reasoning:
            message["content"] = (reasoning + "\n" + content).strip()
        else:
            message["reasoning_content"] = reasoning

    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

    return message, finish_reason



# --------------------------------------------------------------------------- #
# Context compaction
#
# Nothing trimmed the transcript before, so `messages` grew without bound: a
# 403-step Lean proof reached a 810k-token prompt, and each of those 403 calls
# re-sent the whole prefix. A long run eventually hits the provider's context
# ceiling mid-proof -- an API rejection, not a graceful stop.
#
# There is exactly ONE mechanism. When the buffer crosses its budget, the
# ENTIRE history is sent in a single request to be summarized into one message,
# and the run continues with only that message. Nothing is elided, truncated or
# selectively dropped: everything that is not in the summary is gone from the
# model's view, which is precisely why the summary is written by a model that
# has just read all of it -- including the reasoning.
#
# The result is bounded by what a model can emit in one reply, so a compaction
# always lands the buffer near-empty rather than just under the trigger. That
# matters: an implementation that evicted only enough to clear the threshold
# ran 9 summarization calls on a real trajectory where 1 suffices.
#
# On the record: compaction rewrites the WIRE BUFFER, while every original
# message is already in SQLite. To keep the two reconcilable, Agent.run logs
# one `role="compaction"` row per event holding the stats, the summary, and the
# full resulting buffer -- so "what did the model actually see at step N" is
# answerable exactly, by replaying from the last compaction row rather than by
# re-deriving it (which would silently change meaning whenever this code did).
# --------------------------------------------------------------------------- #

# Chars per token. Measured 2.57 on a real Lean trajectory (unicode maths and
# JSON-escaped tool arguments both fragment badly); 2.4 rounds that DOWN so the
# estimate errs high and compaction fires early rather than late.
_CHARS_PER_TOKEN = 2.4
_MSG_OVERHEAD_TOKENS = 4

COMPACTION_MARKER = "[compaction]"

_SUMMARY_INSTRUCTION = (
    "STOP working on the task. Your context is full, and everything above is "
    "about to be deleted and replaced by your reply to this message.\n\n"
    "You have EXACTLY ONE attempt, and you have NO tools for it. Do not call a "
    "tool. Do not answer with a plan to check something first -- there is no "
    "next turn: whatever is not in this single reply is gone for good, and a "
    "reply like 'let me look at the file before summarizing' becomes your "
    "entire memory of this run. Everything you need is already above. Reply "
    "with the summary text and nothing else.\n\n"
    "Compress everything above into the one message you will continue from. "
    "Write a dense factual brief, no preamble. Cover, in this order and only "
    "where they apply:\n"
    "1. the goal, and every hard constraint or rule you were given;\n"
    "2. what you have ESTABLISHED -- results proved, files written and their "
    "current contents or state, facts confirmed, with exact names and paths;\n"
    "3. what you TRIED AND FAILED, each with the reason it failed, so that "
    "you do not try it again;\n"
    "4. the current state and your immediate next step.\n"
    "Preserve exact identifiers, file paths, lemma names, command lines and "
    "error text. Do not speculate and do not add advice."
)


def summary_from_reply(msg: Optional[dict]) -> Optional[str]:
    """The summary text, or None if the reply is not a usable summary.

    A reply carrying `tool_calls` is REJECTED outright, content and all.
    DeepSeek in particular likes to answer a summarize request with "I should
    check the current state first" plus a tool call -- and that sentence would
    otherwise become the agent's entire memory of the run.
    """
    if not msg:
        return None
    if msg.get("tool_calls"):
        return None
    text = msg.get("content")
    return text if text else None


def request_summary(cfg, buffer, tools, request_kwargs, *, connect_timeout,
                    read_timeout, max_retries, chat=None):
    """Ask for the summary; if the model reaches for a tool, ask once more
    with the tools withheld so that it cannot.

    The first attempt passes `tools` unchanged so the prefix matches the call
    just made and the provider serves it from cache. The retry drops them,
    which costs a full-price prompt, but only happens when the model has
    already misbehaved -- and it makes a tool call physically impossible.
    """
    chat = chat or _stream_chat
    ask = buffer + [{"role": "user", "content": _SUMMARY_INSTRUCTION}]
    msg, _fr = chat(cfg, ask, tools, request_kwargs,
                    connect_timeout=connect_timeout, read_timeout=read_timeout,
                    max_retries=max_retries)
    text = summary_from_reply(msg)
    if text is not None:
        return text
    msg, _fr = chat(cfg, ask, None, request_kwargs,
                    connect_timeout=connect_timeout, read_timeout=read_timeout,
                    max_retries=max_retries)
    return summary_from_reply(msg)


def estimate_message_tokens(m: dict) -> int:
    """Cheap, deliberately high estimate of one message's cost on the wire."""
    n = 0
    c = m.get("content")
    if isinstance(c, str):
        n += len(c)
    elif c is not None:
        n += len(json.dumps(c, ensure_ascii=False))
    r = m.get("reasoning_content")
    if isinstance(r, str):
        n += len(r)
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function") or {}
        n += len(fn.get("arguments") or "") + len(fn.get("name") or "")
    return int(n / _CHARS_PER_TOKEN) + _MSG_OVERHEAD_TOKENS


def estimate_tokens(messages: list[dict]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def compact_messages(
    messages: list[dict],
    *,
    budget_tokens: int,
    summarize,
) -> tuple[list[dict], dict]:
    """Replace the entire history with a single summary of it.

    Returns (messages, stats); the input is not mutated. The ORIGINAL PROMPT is
    kept -- the system message and the first user message, which carry the task
    and the rules the agent is judged against. Everything after it, including
    any previous summary, becomes the material for the new one, so the buffer is
    always exactly [system, task, summary] immediately after a compaction.

    `summarize` is handed the WHOLE current buffer, not just the part being
    replaced, so that the request it builds can reuse this exact prefix and hit
    the provider's cache -- re-rendering the history into a fresh prompt would
    bill every token at full rate. It returns the summary text. If it fails or
    returns nothing, compaction DOES NOT HAPPEN: the buffer is handed back
    untouched and the next step tries again. Substituting some other reduction
    would defeat the point of summarizing in the first place.
    """
    before = estimate_tokens(messages)
    stats = {"before": before, "after": before, "messages_summarized": 0,
             "summary": None, "error": None}
    if before <= budget_tokens:
        return list(messages), stats

    # The original prompt: the system message plus the FIRST user message. Any
    # later user message is either a nudge or a previous compaction summary,
    # and both belong in the material to be re-summarized.
    keep, history, seen_user = [], [], False
    for m in messages:
        if m.get("role") == "system":
            keep.append(m)
        elif m.get("role") == "user" and not seen_user:
            seen_user = True
            keep.append(m)
        else:
            history.append(m)
    if not history:
        return list(messages), stats

    try:
        text = summarize(list(messages))
    except Exception as e:                                # noqa: BLE001
        stats["error"] = f"{type(e).__name__}: {e}"
        return list(messages), stats
    if not text:
        stats["error"] = "summarizer returned nothing"
        return list(messages), stats

    stats["summary"] = text
    stats["messages_summarized"] = len(history)
    out = keep + [{
        "role": "user",
        "content": (COMPACTION_MARKER + " COMPACTION SUMMARY. The work you did "
                    "between the task above and now exceeded the context window, "
                    "so the full transcript of it was replaced by this summary. "
                    "This is a summary, not a transcript: treat it as your own "
                    "record of what happened, continue from it, and re-read any "
                    "file you need rather than relying on memory of it.\n\n"
                    + text)}]
    stats["after"] = estimate_tokens(out)
    return out, stats

# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #

@dataclass
class AgentResult:
    trajectory_id: str
    messages: list[dict]
    finish_reason: str
    num_steps: int
    final_result: Optional[str] = None


class Agent:
    """Runs the tool-calling loop and records every message to SQLite."""

    def __init__(
        self,
        config: AgentConfig,
        registry: ToolRegistry,
        db_path: str | os.PathLike = "trajectories.db",
        agent_name: str = "agent",
    ) -> None:
        self.config = config
        self.registry = registry
        self.agent_name = agent_name
        self.store = TrajectoryStore(db_path)

    # -- public API -------------------------------------------------------- #
    def run(
        self,
        user_input: Union[str, list[dict]],
        system_prompt: Optional[str] = None,
        max_steps: int = 10,
        request_kwargs: Optional[dict] = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        llm_retries: int = 3,
        max_length_continues: int = 10,
        should_continue=None,
    ) -> AgentResult:
        """Run the loop. Stops when the model returns no tool_calls or when
        max_steps is reached. LLM calls use (connect_timeout, read_timeout)
        and retry up to llm_retries times on connection/timeout errors.

        A turn the provider truncated at its output ceiling
        (finish_reason="length") is not a stop: the loop appends a "Continue"
        user message and keeps going, up to max_length_continues consecutive
        times.

        `should_continue(assistant_msg, step_idx) -> str | None` lets the
        CALLER own the stopping rule. It is consulted when the model decides
        it is finished; returning text appends it as a user message and the
        loop carries on with the conversation intact, returning None accepts
        the stop. This is for tasks where the model's own sense of "done" is
        not the real finish line -- an optimization problem is not solved
        because the agent has run out of ideas, it is solved when the judge
        says Accepted."""
        cfg = self.config
        tools = self.registry.to_openai()

        # Start a trajectory row.
        traj_id = self.store.create_trajectory(
            self.agent_name, cfg, system_prompt, user_input
        )

        messages: list[dict] = []
        seq = 0
        step_idx = 0

        if system_prompt:
            m = {"role": "system", "content": system_prompt}
            messages.append(m)
            self.store.log_message(traj_id, step_idx, seq, m)
            seq += 1

        # user_input can be a plain string (text-only model) or a list of
        # OpenAI content parts (for multimodal models, e.g. an image_url part).
        m = {"role": "user", "content": user_input}
        messages.append(m)
        self.store.log_message(traj_id, step_idx, seq, m)
        seq += 1

        finish_reason = "max_steps"
        # Consecutive truncated turns. Bounded so a model that truncates every
        # single time still terminates instead of spinning -- which matters
        # now that callers run with no step cap.
        length_continues = 0
        budget = int(cfg.context_window * cfg.compact_at) if cfg.compact_at else 0

        def _summarize(buffer: list[dict]) -> Optional[str]:
            """One summarization request (plus at most one retry). See
            request_summary: the buffer goes out verbatim so the prefix is
            cached, and a reply that calls a tool is rejected rather than
            mistaken for a summary."""
            return request_summary(
                cfg, buffer, tools, request_kwargs,
                connect_timeout=connect_timeout, read_timeout=read_timeout,
                max_retries=llm_retries)

        for step_idx in range(1, max_steps + 1):
            # Compact before spending a call on the buffer. Every message is
            # already in SQLite; the compaction row logged below records what
            # the model sees from here on.
            if budget and estimate_tokens(messages) > budget:
                cand, cstats = compact_messages(
                    messages, budget_tokens=budget, summarize=_summarize)
                if cstats["error"]:
                    print(f"[agentlib] compaction skipped: {cstats['error']}",
                          file=sys.stderr, flush=True)
                elif cstats["summary"]:
                    messages = cand
                    self.store.log_message(
                        traj_id, step_idx, seq,
                        {"role": "compaction", "stats": cstats,
                         "wire": messages})
                    seq += 1
                    print(f"[agentlib] compacted {cstats['before']:,} -> "
                          f"{cstats['after']:,} est. tokens "
                          f"({cstats['messages_summarized']} messages "
                          f"summarized into one)", flush=True)
            assistant_msg, fr = _stream_chat(
                cfg, messages, tools, request_kwargs,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                max_retries=llm_retries,
            )
            messages.append(assistant_msg)
            self.store.log_message(traj_id, step_idx, seq, assistant_msg)
            seq += 1

            # On LLM call failure, stop the loop cleanly with finish_reason="error".
            if fr == "error":
                finish_reason = "error"
                break

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                # finish_reason="length" means the provider truncated the turn
                # at its own output ceiling -- the model did NOT choose to
                # stop. Breaking here throws the whole run away mid-thought
                # (observed on Lean proofs: a single 43k-token reasoning block
                # with no content and no tool call, dead at step 2). Nudge it
                # to continue instead.
                if fr == "length" and length_continues < max_length_continues:
                    length_continues += 1
                    # The truncated turn goes back on the wire in a form the
                    # API will accept. `reasoning_content` is an output-only
                    # field (DeepSeek does not carry it across turns and the
                    # provider's template renderer strips it from the prior
                    # assistant message on the next call), so keeping it as-is
                    # would silently DISCARD everything the model had reasoned
                    # before the truncation point. Fold it into `content`,
                    # which IS carried across turns, so the model sees its own
                    # prior reasoning when it continues.
                    rc = assistant_msg.pop("reasoning_content", None)
                    cc = assistant_msg.get("content")
                    parts: list[str] = []
                    if cc:
                        parts.append(f"Content was:\n{cc}")
                    if rc:
                        parts.append(f"Reasoning content was:\n{rc}")
                    if parts:
                        assistant_msg["content"] = (
                            "This message exceeded the output limit and was "
                            "truncated.\n\n" + "\n\n".join(parts))
                    else:
                        assistant_msg["content"] = (
                            "This message exceeded the output limit and was "
                            "truncated. No content or reasoning content was "
                            "produced before the limit.")
                    m = {"role": "user", "content": "Continue"}
                    messages.append(m)
                    self.store.log_message(traj_id, step_idx, seq, m)
                    seq += 1
                    continue
                # The model believes it is done. The caller may disagree --
                # and if it does, the conversation continues from here rather
                # than restarting, so everything learned so far is still in
                # context (compaction keeps that affordable).
                if should_continue is not None:
                    nudge = should_continue(assistant_msg, step_idx)
                    if nudge:
                        m = {"role": "user", "content": nudge}
                        messages.append(m)
                        self.store.log_message(traj_id, step_idx, seq, m)
                        seq += 1
                        continue
                finish_reason = fr or "stop"
                break
            length_continues = 0

            # Execute each tool call and feed results back.
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"].get("arguments", "")
                try:
                    result = self.registry.invoke(fn_name, fn_args)
                    err = None
                except Exception as e:  # noqa: BLE001 - record, don't crash
                    result = f"ERROR: {type(e).__name__}: {e}"
                    err = type(e).__name__

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn_name,
                    "content": result,
                }
                messages.append(tool_msg)
                self.store.log_message(traj_id, step_idx, seq, tool_msg)
                seq += 1
            # loop continues; next iteration calls the model again.

        final_result = None
        if finish_reason != "max_steps":
            final_result = messages[-1].get("content") if messages else None

        self.store.finalize_trajectory(
            traj_id, final_result, finish_reason, step_idx
        )

        return AgentResult(
            trajectory_id=traj_id,
            messages=messages,
            finish_reason=finish_reason,
            num_steps=step_idx,
            final_result=final_result,
        )

    # -- convenience ------------------------------------------------------- #
    def dump_trajectory(self, trajectory_id: str) -> list[dict]:
        return self.store.load_messages(trajectory_id)

    def close(self) -> None:
        self.store.close()


# --------------------------------------------------------------------------- #
# Safe bash: cgroup-sandboxed command execution
#
# The goal is exactly one thing: NO AGENT CAN OOM THE HOST. Everything else
# here is either in service of that or deliberately generous so it never
# interferes with honest heavy work (lake builds, brute-force searches).
#
# Three nested cgroups, from outermost in:
#
#   agents.slice               fleet cap = the host guarantee. Bounds every
#     |                        agent TOGETHER at ~75% of RAM, so safety does
#     |                        not depend on anyone counting how many agents
#     |                        are running.
#     +- agents-<agent>.slice  per-agent cap: one runaway agent cannot starve
#          |                   its siblings.
#          +- <agent>.scope    the agent's Python process
#          +- agent-bash-*.scope   one per command
#
# Set by enter_agent_cgroup() (fleet + agent levels) and _safe_bash_run() (the
# command level). Slices nest, scopes do not — a command scope created without
# `--slice=` lands in app.slice as a SIBLING of its agent, outside every
# budget, which is why the `--slice=` argument is load-bearing.
#
# Within a level: MemoryHigh sits below MemoryMax so pressure first throttles
# and reclaims and only then kills, and each command raises its own
# oom_score_adj so that when a cap IS hit the kernel takes the command rather
# than the agent that launched it. On a SWAPLESS host the throttle band is
# collapsed (High = Max) because anon memory there is unreclaimable and the
# band converts a prompt kill into an unbounded stall — see _mem_props().
#
# systemd owns the wall clock too (RuntimeMaxSec), because it stops the whole
# scope at the deadline; the `timeout` binary signals only the bash it started
# and leaves anything backgrounded behind.
#
# NEVER add `ulimit -u` / `prlimit --nproc` to any path. RLIMIT_NPROC is
# counted per-UID ACROSS THE WHOLE MACHINE, not per process tree: setting it
# to N when the user already has >N processes makes the very next fork fail
# with EAGAIN ("bash: fork: Resource temporarily unavailable"), which is
# indistinguishable from a real fork bomb and kills every agent on a busy box.
# TasksMax= on the cgroup is the correct, scoped fork cap.
#
# Likewise never use `ulimit -v` as a memory cap: address space is not RSS, so
# it breaks every mmap-heavy tool while a plain malloc loop sails past it.
# Without cgroups there is no honest memory limit, and _safe_bash_run says so
# out loud rather than pretending.
# --------------------------------------------------------------------------- #

# Detect capabilities once at import. These are evaluated lazily via
# functions defined immediately below (forward references resolved at
# call time, not at module-load time, hence the wrappers).
_CGROUP_V2 = os.path.isfile("/sys/fs/cgroup/cgroup.controllers")
_SYSTEMD_RUN_USER: Optional[bool] = None  # lazily set by _systemd_user_ok()

# Fleet root. Every per-agent slice is named `<_FLEET>-<tag>.slice`, and
# systemd reads a dash as hierarchy, so they are all children of
# `<_FLEET>.slice` and one cap on it bounds the whole fleet.
_FLEET = "agents"
_FLEET_SLICE = f"{_FLEET}.slice"

# Fraction of host RAM the fleet may use in total. The remainder is headroom
# for the OS, page cache, and anything else on the box.
_FLEET_RAM_FRACTION = 0.75
_FLEET_RAM_ENV = "CURRIC_FLEET_RAM_MB"      # operator override

# Set by enter_agent_cgroup(): the per-agent slice every command scope is
# placed into, so the aggregate cap covers them. None => no slice available.
_AGENT_SLICE: Optional[str] = None
_AGENT_SLICE_ENV = "CURRIC_AGENT_SLICE"     # survives the re-exec
_SLICE_OWNER_ENV = "CURRIC_SLICE_OWNER"     # pid allowed to size that slice

_WARNED_UNSANDBOXED = False


def _unit_safe(name: str) -> str:
    """Sanitize a free-form name for use in a systemd unit name."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)[:64]


def _mem_total_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 4096


def _swap_total_mb() -> int:
    """Swap configured on the host, in MB. 0 on a swapless box."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("SwapTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def fleet_ram_mb_default() -> int:
    """Total RAM the whole agent fleet may use, across every agent."""
    env = os.environ.get(_FLEET_RAM_ENV)
    if env:
        try:
            return max(512, int(env))
        except ValueError:
            pass
    return int(_mem_total_mb() * _FLEET_RAM_FRACTION)


def _set_unit_props(unit: str, props: dict) -> bool:
    """Apply resource properties to a (transient) unit. Idempotent, so
    concurrent agents setting the same values on the shared fleet slice do
    not race. The unit must already exist — for a slice that means a unit
    inside it is running."""
    args = [f"{k}={v}" for k, v in props.items()]
    try:
        r = subprocess.run(
            ["systemctl", "--user", "set-property", "--runtime", unit, *args],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _warn_unsandboxed() -> None:
    """Say it once, loudly. Silently degrading to 'no limits' is how a host
    gets OOMed by a harness everyone believed was sandboxed."""
    global _WARNED_UNSANDBOXED
    if not _WARNED_UNSANDBOXED:
        _WARNED_UNSANDBOXED = True
        print("[agentlib] WARNING: systemd user cgroups unavailable — bash "
              "commands run with a wall-clock limit ONLY. Memory and fork "
              "limits are NOT enforced; a runaway command can take the host "
              "down.", file=sys.stderr, flush=True)


def _get_unit_prop(unit: str, prop: str) -> Optional[str]:
    """Current value of one unit property, or None if it cannot be read."""
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", "-p", prop, "--value", unit],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _mem_props(max_mb: int) -> dict:
    """MemoryHigh below MemoryMax: the kernel throttles and reclaims at High
    and only kills at Max, so a build that briefly spikes gets slowed down
    instead of shot. Without this, a legitimate heavy `lake build` dies at the
    first spike over the cap.

    That ramp is only worth having when the spike is in something the kernel
    can actually reclaim. WITH swap, anon pages above High get paged out and
    the workload survives. WITHOUT swap, anon memory is unreclaimable: a
    process whose footprint is anon (a brute-force search holding its state
    in RAM, say) sits above High and takes a forced synchronous reclaim on
    every single allocation, each one scanning and freeing nothing. It never
    reaches Max, so it is never killed — it just stops making progress and
    burns its entire wall-clock budget in reclaim. Observed in the wild: a
    stress test at 95% of an 8 GiB cap, 34,900 `high` events, 6 CPU-seconds
    of actual work in 5 wall-clock minutes.

    So on a swapless host, collapse the gap and let Max do the work. Nothing
    is lost: the kernel still reclaims file-backed pages when a cgroup hits
    MemoryMax, and only OOM-kills if that reclaim fails. A page-cache-heavy
    build keeps its protection; an anon-heavy runaway now dies promptly
    instead of thrashing, which is the honest outcome either way."""
    if _swap_total_mb() > 0:
        return {"MemoryHigh": f"{int(max_mb * 0.9)}M", "MemoryMax": f"{max_mb}M"}
    return {"MemoryHigh": f"{max_mb}M", "MemoryMax": f"{max_mb}M"}


def _systemd_user_ok() -> bool:
    """True iff `systemd-run --user` will work — i.e. we have a user manager
    to talk to AND the controllers we actually use are delegated. Cached
    after first call.

    The probe sets the SAME properties the real invocations do. Probing only
    TasksMax would pass on a host where cpu/memory are not delegated to the
    user manager, and then every real bash call would fail instead."""
    global _SYSTEMD_RUN_USER
    if _SYSTEMD_RUN_USER is not None:
        return _SYSTEMD_RUN_USER
    if shutil.which("systemd-run") is None:
        _SYSTEMD_RUN_USER = False
        return False
    try:
        # Unique unit name: a fixed one collides when several agents start at
        # the same moment, and the loser would wrongly conclude systemd is
        # unavailable and drop to the weaker fallback path.
        probe = f"agentlib-probe-{os.getpid()}-{int(time.monotonic() * 1e6)}"
        r = subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet",
             f"--unit={probe}",
             "--property=TasksMax=16",
             "--property=MemoryMax=64M",
             "--property=CPUQuota=100%",
             "--", "/bin/true"],
            capture_output=True, text=True, timeout=10,
        )
        _SYSTEMD_RUN_USER = (r.returncode == 0)
    except Exception:
        _SYSTEMD_RUN_USER = False
    return _SYSTEMD_RUN_USER


def _wrap(
    workdir: str,
    prelude: str,
    command: str,
    max_file_mb: Optional[int] = None,
) -> str:
    """Build the shell text run inside the sandbox: cd, prelude, oom bias,
    optional file-size cap, then the agent's command.

    One statement per LINE, and the command is NOT `exec`ed. `exec <command>`
    only accepts a simple command, so it made `for i in ...; do ...; done` a
    syntax error — and, far worse, silently swallowed everything after the
    first top-level operator: `exec make && ./run` replaces the shell with
    make, so ./run never ran and the agent saw a clean exit 0. The cost of
    dropping exec is one extra bash process, which the cgroup accounts for."""
    lines = [f"cd {shlex.quote(workdir)} || exit 1"]
    if prelude:
        lines.append(prelude)
    # Bias the OOM killer toward the command and away from the agent that
    # launched it. Both live in the same slice, so when the slice hits its
    # cap the kernel picks by oom_score — and it will happily pick the small
    # Python agent over the 10GB build that caused the problem, losing the
    # trajectory. Raising our own oom_score_adj is allowed unprivileged
    # (lowering it is not), so the bias has to be applied on this side.
    # 2>/dev/null FIRST: bash processes redirections left to right, and if
    # the /proc open is denied (e.g. under Landlock) the error must land in
    # the already-redirected stderr instead of polluting every tool result.
    lines.append("echo 500 2>/dev/null > /proc/self/oom_score_adj")
    if max_file_mb:
        # bash counts 1024-byte blocks for -f (verified: `ulimit -f 1` caps a
        # file at 1024 bytes). Off by default: a build artifact or a big log
        # is not a memory-safety problem, and hitting this cap kills the
        # process with an opaque SIGXFSZ.
        lines.append(f"ulimit -f {max_file_mb * 1024} 2>/dev/null")
    lines.append(command)
    return "\n".join(lines)


def _safe_bash_run(
    command: str,
    *,
    workdir: str,
    max_ram_mb: int,
    max_cpu_pct: Optional[int],
    max_wall_s: int,
    tasks_max: int,
    output_cap: int = 32768,
    prelude: str = "",
    max_file_mb: Optional[int] = None,
    landlock_ro: Optional[list] = None,
    landlock_rw: Optional[list] = None,
) -> str:
    """Run `command` inside a memory/fork/wall-clock-bounded sandbox.
    (See register_safe_bash for the public surface; this is the impl.)

    `prelude` is shell text prepended to the command inside the sandbox
    (after cd, before the command). Use it to set up PATH or other env, e.g.
    prelude='export PATH="$HOME/.elan/bin:$PATH"'.

    `landlock_ro` / `landlock_rw`: when either is set, the command runs
    under a Landlock filesystem allowlist (see apply_landlock): listed
    paths (plus workdir, added to rw automatically) are the ONLY visible
    parts of the filesystem. Fail-closed if the kernel can't enforce it."""
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

    unit: Optional[str] = None
    wrapped = _wrap(workdir, prelude, command, max_file_mb=max_file_mb)

    # The Landlock wrapper execs the real bash after confining itself; it
    # goes INSIDE systemd-run so the whole confined tree is also the scope.
    bash_argv = ["bash", "--noprofile", "--norc", "-c", wrapped]
    if landlock_ro is not None or landlock_rw is not None:
        policy = json.dumps({"ro": landlock_ro or [],
                             "rw": (landlock_rw or []) + [workdir]})
        bash_argv = [sys.executable, os.path.abspath(__file__),
                     "--landlock-exec", policy, "--"] + bash_argv

    if _systemd_user_ok():
        unit = f"agent-bash-{os.getpid()}-{int(time.monotonic()*1e6)}"
        # NOTE: do NOT set `ulimit -u` / prlimit --nproc here. RLIMIT_NPROC is
        # per-UID system-wide (not cgroup-scoped), so when multiple agents run
        # concurrently their bash invocations share one budget and even a
        # simple `curl` (which forks a resolver) hits "Resource temporarily
        # unavailable". TasksMax on the cgroup is the correct, scoped
        # mechanism.
        cmd = [
            "systemd-run", "--user", "--scope", "--quiet",
            f"--unit={unit}",
            f"--property=TasksMax={tasks_max}",
            # systemd owns the wall clock: at the deadline it stops the whole
            # SCOPE, so anything the command backgrounded dies with it. The
            # `timeout` binary only signals the bash it started.
            f"--property=RuntimeMaxSec={max_wall_s}",
        ]
        for k, v in _mem_props(max_ram_mb).items():
            cmd.append(f"--property={k}={v}")
        # CPU is optional: it is a responsiveness knob, not a memory-safety
        # one, and a quota silently halves the throughput of a parallel build.
        if max_cpu_pct:
            cmd.append(f"--property=CPUQuota={max_cpu_pct}%")
        # Nest inside the agent's slice when there is one, so this scope
        # counts against the agent's aggregate budget instead of being a
        # sibling of it under app.slice.
        if _AGENT_SLICE:
            cmd.append(f"--slice={_AGENT_SLICE}")
        cmd += ["--"] + bash_argv
    else:
        # No cgroups: wall-clock is all that can honestly be enforced. The old
        # `ulimit -v` here was worse than nothing — address space is not RSS,
        # so it broke every mmap-heavy tool (JVM, Go, sanitizers) while a
        # plain malloc-and-touch loop still took the box down.
        _warn_unsandboxed()
        cmd = [
            "timeout", "--kill-after=2s", "--signal=KILL",
            f"{max_wall_s}s",
        ] + bash_argv

    # Read stdout/stderr with the cap applied AS WE READ, not afterwards:
    # capture_output=True would pull an unbounded `yes`-style stream into the
    # agent's own address space (which enter_agent_cgroup caps) and OOM the
    # agent instead of the sandbox.
    #
    # Truncation strategy: if a stream exceeds output_cap chars, we keep the
    # first half AND the last half with a marker in between, so the model
    # sees both the head (where errors usually appear) and the tail (where
    # the final result / prompt is). E.g. output_cap=32768: first 16384 chars
    # + "[... N chars truncated ...]" + last 16384 chars. Past the cap we
    # keep only a rolling tail buffer in memory and discard the middle, so a
    # multi-gigabyte `yes` stream cannot OOM the agent.
    keep = output_cap // 2
    state = {"truncated": False, "total": 0, "head_len": 0}
    lock = threading.Lock()

    def _drain(stream, head: list[str], tail: list[str]) -> None:
        """head: first `keep` chars. tail: rolling buffer of last `keep` chars."""
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                with lock:
                    state["total"] += len(chunk)
                    hl = state["head_len"]
                    # Fill the head bucket until it has `keep` chars.
                    if hl < keep:
                        room = keep - hl
                        take = min(len(chunk), room)
                        head.append(chunk[:take])
                        state["head_len"] = hl + take
                        chunk = chunk[take:]
                    # Anything past the head goes into a rolling tail buffer
                    # of at most `keep` chars (discard oldest overflow).
                    if chunk:
                        tail.append(chunk)
                        tl = sum(len(s) for s in tail)
                        if tl > keep:
                            # Drop from the front until we're at most `keep`.
                            while tl > keep and len(tail) > 1:
                                drop = tail[0]
                                tl -= len(drop)
                                tail.pop(0)
                            if tl > keep:
                                tail[0] = tail[0][-keep:]
                    if state["total"] > output_cap:
                        state["truncated"] = True
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    proc = None
    started = time.monotonic()
    try:
        # start_new_session: the sandbox gets its own process group, so a
        # timeout can kill the WHOLE tree. Killing just the direct child
        # leaves grandchildren orphaned, and enough of those accumulating is
        # what pushes a box toward genuine fork exhaustion.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        out_head: list[str] = []
        out_tail: list[str] = []
        err_head: list[str] = []
        err_tail: list[str] = []
        threads = [
            threading.Thread(target=_drain, args=(proc.stdout, out_head, out_tail), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, err_head, err_tail), daemon=True),
        ]
        for t in threads:
            t.start()

        timed_out = False
        reaped = False
        try:
            rc = proc.wait(timeout=max_wall_s + 10)
        except subprocess.TimeoutExpired:
            timed_out = True
            _reap_tree(proc, unit)   # must precede the wait below
            reaped = True
            try:
                rc = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rc = -signal.SIGKILL
        # Reap on the normal path too, and BEFORE joining the drain threads.
        # The inner `timeout` only kills the bash it launched, so `foo &`
        # inside the command outlives it — and such an orphan still holds the
        # inherited stdout pipe, which would otherwise keep the drain threads
        # blocked here for their full join timeout on every such call.
        if not reaped:
            _reap_tree(proc, unit)
        for t in threads:
            t.join(timeout=5)

        def _assemble(head: list[str], tail: list[str]) -> str:
            """Combine head + tail with a truncation marker if needed.
            - total <= keep: head holds everything, tail empty.
            - keep < total <= output_cap: head has `keep`, tail has the
              rest (<= keep); concatenate them, no marker.
            - total > output_cap: head has `keep`, tail has last `keep`;
              show head + marker + tail."""
            head_str = "".join(head)
            tail_str = "".join(tail)
            if not state["truncated"]:
                # Fits within the cap; concatenate head + whatever tail
                # captured (may be empty if total <= keep).
                return head_str + tail_str
            dropped = state["total"] - len(head_str) - len(tail_str)
            return (
                head_str
                + f"\n[... {dropped} chars truncated ...]\n"
                + tail_str
            )

        parts: list[str] = []
        out_str = _assemble(out_head, out_tail)
        err_str = _assemble(err_head, err_tail)
        if out_str:
            parts.append(out_str)
        if err_str:
            parts.append("[stderr]\n" + err_str)
        elapsed = time.monotonic() - started
        if timed_out:
            parts.append(
                f"[timeout after {max_wall_s}s — outer wrapper; process tree killed]"
            )
        elif rc in (124, 137) or (rc < 0 and elapsed >= max_wall_s - 1):
            # RuntimeMaxSec stops the scope with SIGTERM at the deadline, so a
            # wall-clock kill arrives as rc=-15; distinguish it from a memory
            # kill by the elapsed time rather than guessing from the signal.
            parts.append(f"[killed: hit the {max_wall_s}s wall-clock limit]")
        elif rc < 0:
            parts.append(
                f"[killed by signal {-rc}: over the {max_ram_mb}MB memory cap, "
                f"the {tasks_max}-task cap, or the fleet-wide memory budget]"
            )
        parts.append(f"[exit {rc}]")
        return "\n".join(parts)
    except Exception as e:
        if proc is not None:
            _reap_tree(proc, unit)
        return f"[error: {type(e).__name__}: {e}]"


def _reap_tree(proc: "subprocess.Popen", unit: Optional[str]) -> None:
    """Kill anything the sandboxed command left running, at every layer.

    Two layers because neither alone is enough: the process group catches the
    fallback paths and anything the inner `timeout` left behind (it signals
    only the bash it started, so `foo &` survives it), while stopping the
    transient scope catches processes that called setsid() and so escaped the
    group. Safe to call on an already-finished command — both layers no-op.

    start_new_session=True at spawn is what makes the killpg safe: the sandbox
    is its own session leader, so this can never signal the agent itself."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    if unit:
        try:
            subprocess.run(
                ["systemctl", "--user", "stop", f"{unit}.scope"],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        # A transient scope whose payload was killed (OOM, TasksMax, timeout)
        # stays loaded in `failed` state forever. One per killed command adds
        # up fast on a long-running fleet, so clear this unit — by name, never
        # a bare `reset-failed`, which would also clear unrelated user units.
        try:
            subprocess.run(
                ["systemctl", "--user", "reset-failed", f"{unit}.scope"],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass


def enter_agent_cgroup(
    agent_name: str,
    max_ram_mb: int,
    max_cpu_pct: Optional[int] = None,
    tasks_max: int = 8192,
    fleet_ram_mb: Optional[int] = None,
) -> bool:
    """Put this agent, and every command it will run, under one budget.

    Call this as the FIRST thing in main(), before printing or opening any
    file — see the re-exec note below.

    Topology, and why each level exists:

        agents.slice              <- fleet cap: the host guarantee. Bounds ALL
          |                          agents together, so safety does not depend
          |                          on anyone tracking how many are running.
          +- agents-<agent>.slice <- per-agent cap: one runaway agent cannot
               |                     starve its siblings.
               +- <agent>.scope   <- the agent's own Python process
               +- agent-bash-*.scope  <- one per command, created by
                                        _safe_bash_run with --slice= so it
                                        lands HERE and not in app.slice.

    Slices, not scopes, because scopes do not nest: a bash scope created
    without `--slice=` is a SIBLING of the agent under app.slice, outside any
    budget set on it.

    Implemented by RE-EXECING this process under `systemd-run --user --scope`.
    That replaces the old dance of starting a stub `sleep`, discovering its
    cgroup, and writing our own PID into cgroup.procs. The PID is preserved
    across the exec (verified), so a parent that tracks or kills this pid --
    run_orchestrator does -- is unaffected. `CURRIC_AGENT_SLICE` in the
    environment marks the second incarnation so this cannot loop.

    Returns True if the agent is inside its slice (either just re-exec'd into
    it or already there), False if systemd user cgroups are unavailable, in
    which case per-command limits are unavailable too and _safe_bash_run says
    so loudly.
    """
    global _AGENT_SLICE

    already = os.environ.get(_AGENT_SLICE_ENV)
    if already:
        # Second incarnation: we are inside the slice. Apply the budgets now —
        # the slice exists only once a unit is running in it, which is now.
        _AGENT_SLICE = already
        # ...but only the agent that CREATED this slice may size it. A
        # subagent launched through the bash tool inherits this environment,
        # and without this guard it would re-apply its own (larger) numbers to
        # its parent's slice, silently widening the parent's budget. Being
        # counted against the parent's budget is correct; resizing it is not.
        # The pid survives the exec, so it identifies the owner exactly.
        if os.environ.get(_SLICE_OWNER_ENV) != str(os.getpid()):
            return True
        props = dict(_mem_props(max_ram_mb), TasksMax=str(tasks_max))
        if max_cpu_pct:
            props["CPUQuota"] = f"{max_cpu_pct}%"
        _set_unit_props(already, props)
        # The fleet cap is what actually protects the host — but it is SHARED,
        # so only establish it if nobody has yet. Setting it unconditionally
        # means the newest agent's idea of the budget silently overwrites
        # everyone else's, including a deliberate operator override; with a
        # busy orchestrator that happens every few seconds.
        fleet_mb = fleet_ram_mb or fleet_ram_mb_default()
        current = _get_unit_prop(_FLEET_SLICE, "MemoryMax")
        if current in (None, "", "infinity", "0"):
            _set_unit_props(_FLEET_SLICE, dict(_mem_props(fleet_mb),
                                               TasksMax="32768"))
        elif current.isdigit() and abs(int(current) // (1024 * 1024) - fleet_mb) > 1:
            print(f"[agentlib] fleet cap already set to "
                  f"{int(current) // (1024*1024)}MB (this agent would have used "
                  f"{fleet_mb}MB); leaving it alone.", file=sys.stderr, flush=True)
        return True

    if not _systemd_user_ok():
        return False

    tag = _unit_safe(f"{agent_name}-{os.getpid()}")
    slice_name = f"{_FLEET}-{tag}.slice"
    env = dict(os.environ, **{_AGENT_SLICE_ENV: slice_name,
                              _SLICE_OWNER_ENV: str(os.getpid())})
    argv = [
        "systemd-run", "--user", "--scope", "--quiet",
        f"--unit={_FLEET}-{tag}",
        f"--slice={slice_name}",
        "--", sys.executable, *sys.argv,
    ]
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.execvpe("systemd-run", argv, env)
    except Exception:
        return False   # exec failed: carry on uncontained rather than dying
    return False       # unreachable when the exec succeeds


def refuse_if_low_ram(need_mb: int, label: str = "agent") -> None:
    """Pre-flight: exit with a clear message if available RAM < need_mb.
    Prevents oversubscription when many launchers run concurrently. Reads
    MemAvailable from /proc/meminfo (lenient on parse error — never blocks
    a legit run on a misread)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                    avail_mb = avail_kb // 1024
                    if avail_mb < need_mb:
                        print(
                            f"[refuse_if_low_ram] only {avail_mb}MB available, "
                            f"need {need_mb}MB for {label}; refusing to start.",
                            file=sys.stderr,
                        )
                        raise SystemExit(2)
                    return
    except SystemExit:
        raise
    except Exception:
        pass


def register_safe_bash(
    registry: "ToolRegistry",
    *,
    workdir: str,
    max_ram_mb: int = 12288,
    max_cpu_pct: Optional[int] = None,
    max_wall_s: int = 3600,
    tasks_max: int = 4096,
    output_cap: int = 32768,
    description: Optional[str] = None,
    prelude: str = "",
    max_file_mb: Optional[int] = None,
    landlock_ro: Optional[list] = None,
    landlock_rw: Optional[list] = None,
) -> None:
    """Register a `bash` tool on `registry` whose every invocation is
    sandboxed by _safe_bash_run(). All the duplicate `_make_bash_tool`
    definitions across the harnesses should call this instead.

    Defaults are DELIBERATELY GENEROUS. The host guarantee comes from the
    fleet cap in enter_agent_cgroup(), not from these, so a per-command limit
    only has to be low enough to catch a genuine runaway — anything tighter
    just kills honest work. 12GB fits a heavy `lake build` or a brute-force
    with a big table; CPU is unlimited by default (a quota halves a parallel
    build and protects no memory); there is no file-size cap by default
    (a large build artifact is not a memory-safety problem).

    `workdir` is the only directory guaranteed writable under the systemd
    path. `prelude` is shell text inserted after cd and before the user
    command — use it to export PATH for tools like elan.
    """
    schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Bash command. Executed from the work directory with "
                    "hard limits on RAM, CPU, wall-clock, and fork count. "
                    "Output (stdout+stderr) is returned, capped at "
                    f"{output_cap} chars. Long-running or memory-hungry "
                    "commands will be killed — prefer small, fast commands."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    f"Wall-clock seconds. Default {max_wall_s}. Cannot "
                    f"exceed {max_wall_s}."
                ),
            },
        },
        "required": ["command"],
    }

    def bash(command: str, timeout: Optional[int] = None) -> str:
        wall = min(timeout or max_wall_s, max_wall_s)
        return _safe_bash_run(
            command,
            workdir=workdir,
            max_ram_mb=max_ram_mb,
            max_cpu_pct=max_cpu_pct,
            max_wall_s=wall,
            tasks_max=tasks_max,
            output_cap=output_cap,
            prelude=prelude,
            max_file_mb=max_file_mb,
            landlock_ro=landlock_ro,
            landlock_rw=landlock_rw,
        )

    registry.register(
        "bash",
        description or (
            "Execute a bash command under a cgroup sandbox with hard RAM, "
            "fork, and wall-clock limits. Use for everything: fetching "
            "pages, writing/compiling/testing code, running build tools."
        ),
        schema, bash,
    )


def run_sandboxed_shellcommand(
    command: str,
    *,
    workdir: str,
    max_ram_mb: int = 12288,
    max_cpu_pct: Optional[int] = None,
    max_wall_s: int = 3600,
    tasks_max: int = 4096,
    output_cap: int = 32768,
    prelude: str = "",
    max_file_mb: Optional[int] = None,
    landlock_ro: Optional[list] = None,
    landlock_rw: Optional[list] = None,
) -> str:
    """Run an ad-hoc shell command with the same sandboxing that
    `register_safe_bash` applies to per-agent bash tool calls. Use this
    from harness code that needs to invoke a command directly (e.g. a
    `verify()` tool implementation) instead of letting the agent trigger
    it via bash — same caps, same protections.

    Returns the same concatenated stdout/stderr/exit string that the bash
    tool would return.
    """
    return _safe_bash_run(
        command,
        workdir=workdir,
        max_ram_mb=max_ram_mb,
        max_cpu_pct=max_cpu_pct,
        max_wall_s=max_wall_s,
        tasks_max=tasks_max,
        output_cap=output_cap,
        prelude=prelude,
        max_file_mb=max_file_mb,
        landlock_ro=landlock_ro,
        landlock_rw=landlock_rw,
    )


# --------------------------------------------------------------------------- #
# Landlock: per-command filesystem allowlists
# --------------------------------------------------------------------------- #
# Landlock is a kernel LSM that lets an UNPRIVILEGED process irrevocably
# restrict its own filesystem access before exec; every descendant inherits
# the restriction. No root, no user namespaces, no external binaries -- which
# is what lets this live in a single self-contained file.
#
# Usage from a harness:
#     register_safe_bash(reg, workdir=d, landlock_rw=[d], landlock_ro=[...])
# Every bash invocation is then wrapped as
#     python3 agentlib.py --landlock-exec '<policy json>' -- bash -c ...
# The wrapper applies the ruleset to itself and execs the real command, so
# the restriction covers the whole command tree but never the harness.
#
# Semantics: paths in `ro` get read+execute, paths in `rw` get everything.
# Anything not under a listed path is invisible (opens fail with EACCES).
# The policy is fail-CLOSED: if the kernel cannot enforce it, the wrapper
# refuses to run the command rather than running it unconfined.

_LL_SYS_CREATE = 444          # x86_64 syscall numbers
_LL_SYS_ADD_RULE = 445
_LL_SYS_RESTRICT = 446
_LL_RULE_PATH_BENEATH = 1
_LL_CREATE_VERSION = 1        # flag: query highest supported ABI

_LL_EXECUTE = 1 << 0
_LL_WRITE_FILE = 1 << 1
_LL_READ_FILE = 1 << 2
_LL_READ_DIR = 1 << 3
_LL_V1_ALL = (1 << 13) - 1    # EXECUTE .. MAKE_SYM
_LL_REFER = 1 << 13           # ABI >= 2
_LL_TRUNCATE = 1 << 14        # ABI >= 3
_LL_IOCTL_DEV = 1 << 15       # ABI >= 5
# Bits that are valid on a rule whose target is a FILE, not a directory.
_LL_FILE_BITS = (_LL_EXECUTE | _LL_WRITE_FILE | _LL_READ_FILE
                 | _LL_TRUNCATE | _LL_IOCTL_DEV)


def landlock_abi() -> int:
    """Highest Landlock ABI the running kernel supports; 0 if none."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    v = libc.syscall(_LL_SYS_CREATE, None, 0, _LL_CREATE_VERSION)
    return max(0, v)


def apply_landlock(ro: list, rw: list) -> int:
    """Restrict THIS process (and all future children) to the given paths.

    `ro` paths become read+execute only; `rw` paths get full access. Paths
    that do not exist are skipped. Irrevocable once applied. Returns the ABI
    version used. Raises RuntimeError when the kernel cannot enforce it.
    """
    import ctypes

    abi = landlock_abi()
    if abi < 1:
        raise RuntimeError("Landlock unsupported by this kernel")

    handled = _LL_V1_ALL
    if abi >= 2:
        handled |= _LL_REFER
    if abi >= 3:
        handled |= _LL_TRUNCATE
    if abi >= 5:
        handled |= _LL_IOCTL_DEV

    libc = ctypes.CDLL(None, use_errno=True)

    class RulesetAttr(ctypes.Structure):
        # Only the first field; the kernel zero-fills the rest (network
        # access stays unrestricted, which is what we want here).
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathBeneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64),
                    ("parent_fd", ctypes.c_int32)]

    attr = RulesetAttr(handled_access_fs=handled)
    ruleset_fd = libc.syscall(_LL_SYS_CREATE, ctypes.byref(attr),
                              ctypes.sizeof(attr), 0)
    if ruleset_fd < 0:
        raise RuntimeError(
            f"landlock_create_ruleset failed: errno {ctypes.get_errno()}")

    try:
        for paths, access in ((ro, _LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR),
                              (rw, handled)):
            for p in paths:
                p = os.path.abspath(os.path.expanduser(p))
                if not os.path.exists(p):
                    continue
                if not os.path.isdir(p):
                    a = access & _LL_FILE_BITS
                else:
                    a = access
                fd = os.open(p, os.O_PATH | os.O_CLOEXEC)
                try:
                    rule = PathBeneath(allowed_access=a, parent_fd=fd)
                    r = libc.syscall(_LL_SYS_ADD_RULE, ruleset_fd,
                                     _LL_RULE_PATH_BENEATH,
                                     ctypes.byref(rule), 0)
                    if r != 0:
                        raise RuntimeError(
                            f"landlock_add_rule({p!r}) failed: "
                            f"errno {ctypes.get_errno()}")
                finally:
                    os.close(fd)

        # Required before restrict_self for unprivileged processes.
        PR_SET_NO_NEW_PRIVS = 38
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise RuntimeError(
                f"prctl(NO_NEW_PRIVS) failed: errno {ctypes.get_errno()}")
        if libc.syscall(_LL_SYS_RESTRICT, ruleset_fd, 0) != 0:
            raise RuntimeError(
                f"landlock_restrict_self failed: errno {ctypes.get_errno()}")
    finally:
        os.close(ruleset_fd)
    return abi


def _landlock_exec_main(argv: list) -> int:
    """`python3 agentlib.py --landlock-exec '<json>' -- cmd args...`

    Applies the policy to this process, then execs the command. Fail-closed:
    any error means the command does NOT run.
    """
    try:
        sep = argv.index("--")
    except ValueError:
        print("usage: agentlib.py --landlock-exec '<json>' -- cmd args...",
              file=sys.stderr)
        return 2
    policy = json.loads(argv[sep - 1])
    cmd = argv[sep + 1:]
    if not cmd:
        print("landlock-exec: no command given", file=sys.stderr)
        return 2
    try:
        apply_landlock(policy.get("ro") or [], policy.get("rw") or [])
    except Exception as e:  # noqa: BLE001
        print(f"landlock-exec: REFUSING to run unconfined: {e}",
              file=sys.stderr)
        return 125
    os.execvp(cmd[0], cmd)
    return 127   # unreachable


if __name__ == "__main__" and "--landlock-exec" in sys.argv:
    sys.exit(_landlock_exec_main(sys.argv))
