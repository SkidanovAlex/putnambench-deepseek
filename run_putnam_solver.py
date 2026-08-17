#!/usr/bin/env python3
"""run_putnam_solver.py - agent harness for PutnamBench.

Built on agentlib (trajectory capture, per-agent cgroup, sandboxed
bash/verify). The PutnamBench contract: Mathlib IS available, there is no
per-problem lake project, and the check is `lake env lean` on a single file
rather than `lake build`. The harness stages each problem's directory
itself and pulls work from a flock-protected shared queue, so any number of
worker copies can run in parallel without colliding.

Note: expects a PutnamBench checkout under ./putnambench with the queue /
staging helper (pb.py); that benchmark tooling is not part of this repo.

Usage:
    # one specific problem
    python3 run_putnam_solver.py --problem putnam_1985_b2

    # take whatever is next in the queue, then mark the outcome
    python3 run_putnam_solver.py

    # a worker that keeps going until the queue drains
    python3 run_putnam_solver.py --loop 0

Budgets default to "don't stop me": no step cap, 3-day wall clock per
problem. Lean runs are long, and a run that ends at a cap tells you about
the cap rather than about the problem.

A run ends in exactly one of two ways, and the agent is told so repeatedly:

  * `completed()` runs verify.sh and PASSES -- the theorem is proved; or
  * `disproved()` -- the agent has proven IN LEAN that the frozen statement
    is false. Not machine-checked (there is nothing for verify.sh to pass
    when the statement itself is wrong); the Lean file and the agent's
    explanation are archived under putnambench/defects/ for human review,
    and the queue row gets its own `disproved` status.

Anything else is not an ending. If the model stops without one of those --
out of ideas, or believing it has done enough -- `should_continue` feeds it
back into the loop with the conversation intact. A failed `completed()` is
handed back with the verifier's exact words and never accepted as a stop.

Run as many workers as you like; claiming is atomic (see pb.py's queue
section). A worker that dies mid-problem leaves a `running` row, which the
next claim reaps once its pid is gone.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import socket
import sys
import time

import agentlib

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.expanduser(
    os.environ.get("TRAJ_DB", os.path.join(HERE, "db.sql")))
PB_DIR = os.path.join(HERE, "putnambench")
PB_PY = os.path.join(PB_DIR, "harness", "pb.py")
RUNS_DIR = os.path.join(PB_DIR, "runs")
LEAN4_DIR = os.path.join(PB_DIR, "lean4")
# Claimed disproofs land here for human review; see _make_disproved_tool.
DEFECTS_DIR = os.path.join(PB_DIR, "defects")

# JSON: {"host", "model", "apikey", "context_window", "merge_reasoning"}.
DEEPSEEK_CFG = os.path.join(HERE, ".nearai.deepseek.pro")

# These limits catch a runaway; they do not ration. One Putnam file importing Mathlib elaborates in well under a GB,
# but a bad `simp` or a deep `decide` can blow up, so keep real headroom.
LEAN_AGENT_RAM_MB = 16384
LEAN_BASH_RAM_MB = 12288
LEAN_BASH_WALL_S = 3600
LEAN_BASH_TASKS = 4096
LEAN_BASH_CPU_PCT = None
LEAN_VERIFY_RAM_MB = 12288
LEAN_VERIFY_WALL_S = 3600
LEAN_PRELUDE = 'export PATH="$HOME/.elan/bin:$PATH"'

# `--max-steps 0` means "no cap". agentlib wants a number, so hand it one no
# run will ever reach rather than threading an Optional through the loop.
UNLIMITED_STEPS = 10_000_000


def _load_pb():
    spec = importlib.util.spec_from_file_location("pb", PB_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

def _make_bash_tool(registry: agentlib.ToolRegistry, bench_dir: str) -> None:
    agentlib.register_safe_bash(
        registry, workdir=bench_dir,
        max_ram_mb=LEAN_BASH_RAM_MB,
        max_cpu_pct=LEAN_BASH_CPU_PCT,
        max_wall_s=LEAN_BASH_WALL_S,
        tasks_max=LEAN_BASH_TASKS,
        prelude=LEAN_PRELUDE,
        description=(
            "Execute a bash command from the benchmark directory. PATH "
            "includes elan/lake/lean. Use to read and edit the problem's "
            ".lean file and to elaborate it with `lake env lean`. Sandboxed: "
            "hard limits on RAM, CPU, forks, and wall-clock."
        ),
    )


class _Outcome:
    """Terminal state for one problem, set ONLY by the two finishing tools.

    The agent's own sense of "done" is not the finish line: a run ends when
    the verifier passes or when the agent has filed a Lean disproof. This
    object is what `should_continue` consults, so a stop that is not backed
    by one of those two tool calls is turned back into more work.
    """

    def __init__(self) -> None:
        self.kind: str | None = None     # None | "completed" | "disproved"
        self.disproof_file = ""
        self.explanation = ""
        self.nudges = 0
        self.failed_completions = 0


def _run_verify(bench_dir: str) -> str:
    return agentlib.run_sandboxed_shellcommand(
        "./verify.sh",
        workdir=bench_dir,
        max_ram_mb=LEAN_VERIFY_RAM_MB,
        max_cpu_pct=LEAN_BASH_CPU_PCT,
        max_wall_s=LEAN_VERIFY_WALL_S,
        tasks_max=LEAN_BASH_TASKS,
        prelude=LEAN_PRELUDE,
    )


def _verify_passed(out: str) -> bool:
    return any(l.strip().startswith("PASS [") for l in out.splitlines())


def _make_completed_tool(registry: agentlib.ToolRegistry, bench_dir: str,
                         outcome: _Outcome) -> None:
    """Finishing tool (a): run the verifier; finish only if it passes."""
    schema: dict = {"type": "object", "properties": {}}

    def completed() -> str:
        out = _run_verify(bench_dir)
        if _verify_passed(out):
            outcome.kind = "completed"
            return (out + "\n\n"
                    "VERIFY PASSED. The problem is SOLVED and you are DONE.\n"
                    "Write your final report as your next message -- what the "
                    "mathematical idea was, what you tried, what worked -- and "
                    "then stop. Do not call any more tools.")
        outcome.failed_completions += 1
        # Requirement: never accept a stop off the back of a failed check.
        # Hand back the verifier's own words, in full, and put the agent
        # straight back to work.
        return (out + "\n\n"
                "VERIFY FAILED. You are NOT done. The exact verifier output is "
                "above -- read it carefully, it names the precise reason.\n"
                "Keep working. You have no turn limit and no time limit. Fix "
                "what it reported and call completed() again. If instead you "
                "come to believe the goal is FALSE, prove that in Lean and "
                "call disproved().")

    registry.register(
        "completed",
        "Declare the proof finished. Runs verify.sh, the DEFINITIVE pass/fail "
        "check: no sorry/sorryAx, no axiom, no native_decide, no escape "
        "hatches, the theorem statement byte-for-byte unchanged, the file "
        "elaborates, and #print axioms is clean. If it passes you are done. "
        "If it fails you get the exact reason and must keep working -- a "
        "failed check is not an ending. Call it as often as you like.",
        schema, completed,
    )


def _make_disproved_tool(registry: agentlib.ToolRegistry, bench_dir: str,
                         problem: str, outcome: _Outcome) -> None:
    """Finishing tool (b): the agent claims a Lean disproof of the goal.

    Deliberately NOT machine-checked here. A false goal means the frozen
    statement is itself wrong, so there is nothing for verify.sh to pass --
    it would reject any file that proves something other than the frozen
    theorem. These claims are reviewed by hand, which is exactly why the
    Lean artifact is required and archived: an assertion without the file
    is not reviewable.
    """
    schema: dict = {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description":
                    "Why the formalized goal is false, in ordinary "
                    "mathematics: the counterexample or the contradiction, "
                    "and where the formalization diverges from the informal "
                    "problem. Written for a human reviewer.",
            },
            "file": {
                "type": "string",
                "description":
                    "Filename of your Lean disproof inside the benchmark "
                    "directory (default DISPROOF.lean).",
            },
        },
        "required": ["explanation"],
    }

    def disproved(explanation: str, file: str = "DISPROOF.lean") -> str:
        name = os.path.basename(file or "DISPROOF.lean")
        path = os.path.join(bench_dir, name)
        # Not a verification -- just making sure the thing a human is being
        # asked to review actually exists.
        if not os.path.exists(path):
            return (f"No such file: {name} in the benchmark directory. The "
                    "disproof claim was NOT recorded and you are NOT done.\n"
                    "Write your Lean disproof to DISPROOF.lean first. It "
                    "should elaborate cleanly with no sorry and no axiom, and "
                    "derive False from the theorem's hypotheses (or prove the "
                    "negation of its conclusion). Then call disproved() again.")
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError as e:
            return (f"Could not read {name}: {e}. The claim was NOT recorded "
                    "and you are NOT done.")
        if not text.strip():
            return (f"{name} is empty. The claim was NOT recorded and you are "
                    "NOT done. Put your Lean disproof in it, check that it "
                    "elaborates, then call disproved() again.")

        os.makedirs(DEFECTS_DIR, exist_ok=True)
        with open(os.path.join(DEFECTS_DIR, f"{problem}.DISPROOF.lean"), "w",
                  encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(DEFECTS_DIR, f"{problem}.claim.md"), "w",
                  encoding="utf-8") as f:
            f.write(f"# Claimed disproof: {problem}\n\n"
                    f"Filed by the solver; NOT machine-checked. Review by "
                    f"hand against the frozen statement.\n\n"
                    f"Source: {path}\n\n## Explanation\n\n{explanation}\n")

        outcome.kind = "disproved"
        outcome.disproof_file = path
        outcome.explanation = explanation
        return ("Disproof claim recorded, along with " + name + ". It will be "
                "checked by a human, not by this harness. You are DONE.\n"
                "Write your final report as your next message -- the "
                "counterexample or contradiction, and exactly where the "
                "formalization diverges from the informal problem -- and then "
                "stop. Do not call any more tools.")

    registry.register(
        "disproved",
        "Declare that the goal as formalized is FALSE and that you have "
        "PROVEN so in Lean. Use only in the rare case where the frozen "
        "statement is itself defective. Requires a Lean file (default "
        "DISPROOF.lean) that elaborates cleanly with no sorry and no axiom "
        "and derives False from the hypotheses, or proves the negation of "
        "the conclusion. This claim is NOT machine-checked -- it is reviewed "
        "by a human later -- so do not reach for it to escape a hard proof.",
        schema, disproved,
    )


def _nudge(outcome: _Outcome) -> str:
    """What we say when the agent stops without having finished."""
    outcome.nudges += 1
    extra = ""
    if outcome.failed_completions:
        extra = ("\n\nYou have called completed() and had it come back FAILED "
                 f"{outcome.failed_completions} time(s). That is normal -- it "
                 "is a checkpoint, not a verdict on you. Read the last "
                 "verifier output again and fix precisely what it named.")
    return (
        "You are not finished, so do not stop.\n\n"
        "A run ends in exactly one of two ways, and neither has happened yet:\n"
        "  (a) you call completed() and the verifier PASSES, or\n"
        "  (b) you prove in Lean that the goal is false and call disproved().\n\n"
        "You have NO turn limit and NO time limit. There is no budget you are "
        "spending, nothing is running out, and no one is waiting on you. Taking "
        "another hundred turns costs nothing and is completely fine.\n\n"
        "You are doing well -- the work you have already done is still in front "
        "of you and none of it is wasted. Pick up where you left off: try "
        "another angle on the mathematics, look for different Mathlib lemmas, "
        "break the goal into intermediate lemmas and prove them one at a time, "
        "or test a small case to sharpen your intuition. If you have started "
        "to suspect the goal is actually false, chase that: try to build a "
        "counterexample and derive False from the hypotheses in Lean."
        + extra +
        "\n\nContinue now, with a tool call."
    )


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #

def _system_prompt(problem: str, bench_dir: str, entry: dict,
                   variant: str) -> str:
    answer_block = ""
    if variant == "full" and entry.get("has_solution"):
        answer_block = f"""
# This problem also asks for the ANSWER

`{entry['solution_name']}` is declared `:= sorry` at the top of the file. The
theorem is stated in terms of it, so you must supply the correct closed-form
answer there AND prove the theorem with it. Its TYPE is frozen -- you may
change the value, never the type. A wrong answer makes the theorem false and
unprovable, so work the mathematics out before you start proving.
"""

    return f"""\
You are an autonomous Lean 4 proof engineer working on PutnamBench, a
benchmark of William Lowell Putnam competition problems formalized in Lean 4
with Mathlib.

# Problem

Name: {problem}
Directory: {bench_dir}
File: {problem}.lean   (the ONLY file you may edit)
Tags: {', '.join(entry.get('tags') or []) or '(untagged)'}

Informal statement:

{entry.get('informal_statement', '(none)')}

The file contains a single theorem whose proof is `sorry`. Replace the
`sorry` with a real Lean 4 proof.
{answer_block}
# How to build

There is NO lake project in this directory and `lake build` will NOT work
here. The problem file is elaborated against a SHARED Mathlib project that is
already fully built. To check your work:

    cd {LEAN4_DIR} && lake env lean {bench_dir}/{problem}.lean

That prints nothing on success. A cold run takes ~30-60s, almost all of it
`import Mathlib` -- that is normal, do not assume it hung. Run it from a
subshell so your working directory stays put, e.g.

    (cd {LEAN4_DIR} && lake env lean {bench_dir}/{problem}.lean)

Mathlib IS available and you are expected to use it. `import Mathlib` at the
top of the file already imports all of it; do not add other imports.

# Your tools

1. `bash(command)` - run commands from the benchmark directory. Use it to
   read and rewrite `{problem}.lean` and to run the elaboration command
   above. Scratch files inside this directory are fine (`#check`, `#print`,
   `example`), but the final `{problem}.lean` must be clean.

2. `completed()` - run `./verify.sh`. DEFINITIVE pass/fail. It checks:
   - no `sorry` / `sorryAx`, no `axiom`, no `native_decide`, no
     `@[implemented_by]`, no `@[extern]`, no imports besides Mathlib;
   - the theorem statement is byte-for-byte the frozen one (comments
     stripped, whitespace normalized);
   - the file elaborates with no errors;
   - `#print axioms` names only `propext`, `Classical.choice`, `Quot.sound`.
   Exit 0 and a `PASS` line = solved, and you are done.
   If it FAILS you are not done: you get the exact reason and you keep
   working. Call it as often as you like -- a failed check costs nothing and
   is a checkpoint, not a verdict.

3. `disproved(explanation, file)` - see "If the goal is false" below. Rare.

# HOW THIS RUN ENDS

There are exactly TWO ways to finish, and you must reach one of them:

  (a) `completed()` returns PASS -- you proved the theorem; or
  (b) you prove IN LEAN that the goal is false and call `disproved()`.

Nothing else ends the run. If you stop for any other reason -- out of ideas,
feeling stuck, "this seems infeasible", believing you have done enough -- you
will simply be told to continue, and you will have lost nothing but the turn.

# You have NO limits

This is the single most important thing to understand about this run:

- There is NO turn limit. Not a large one -- none. Take ten turns or ten
  thousand.
- There is NO time limit you need to manage. Nothing is running out.
- There is NO token budget you are spending down, and no cost you should be
  economizing on.
- Nobody is waiting on you. There is no deadline, no queue behind you, and no
  penalty whatsoever for taking longer.

So never ration your effort, never rush a proof because it is "taking too
long", and never wrap up early with a summary of what you would have tried.
If an approach needs fifty turns of lemma-by-lemma grinding, spend them. If
you need to test a small case, or restate the problem from scratch, or throw
away an hour of work and start over, do it -- that is a normal and good use
of this run, not a waste.

The ONLY thing that ends this run is (a) or (b) above.

# If the goal is false

Rarely -- it is uncommon, but it does happen -- the formalization is
defective and the goal as stated in Lean is FALSE. The informal problem is
fine; the translation into Lean lost or mangled something. Typical causes: a
hypothesis was dropped, a quantifier landed in the wrong place, a Mathlib
notation means something other than what it looks like (`f⁻¹` is the
pointwise reciprocal, not the inverse function), or an edge case the informal
problem excluded is admitted by the formal one.

A false goal is UNPROVABLE, and no amount of effort will close it. So if the
mathematics keeps refusing to work out, consider seriously that the statement
itself may be wrong -- and then settle the question instead of guessing:

1. Find the specific counterexample, or the contradiction among the
   hypotheses.
2. PROVE IT IN LEAN. Write `DISPROOF.lean` in your directory that derives
   `False` from the theorem's hypotheses, or proves the negation of its
   conclusion. It must elaborate cleanly with NO `sorry` and NO `axiom` --
   the same standard as a real proof. Do not modify `{problem}.lean` for
   this; the frozen statement stays as it is.
3. Call `disproved(explanation=...)` with the mathematics written out for a
   human reader: the counterexample, and exactly where the formal statement
   diverges from the informal one.

That claim is NOT machine-checked. A person reads it later. That cuts both
ways: an honest, fully proven disproof is a genuinely valuable result and
counts as a successful run -- and a disproof filed to escape a hard proof is
worse than useless, because it wastes a reviewer's time and will be seen for
what it is. Do not reach for this because a proof is hard. Reach for it when
you have the counterexample in hand and Lean agrees with you.

# Rules

- Edit ONLY `{problem}.lean`, inside `{bench_dir}`.
- Do NOT modify anything under `{LEAN4_DIR}`. That is the shared Mathlib
  project every other run on this machine depends on; breaking it breaks
  them all.
- Do NOT change the theorem statement -- not a binder, not a hypothesis, not
  the conclusion. verify.sh compares against a frozen signature. Weakening a
  hypothesis to make the proof go through is a failure, not a solution.
- Helper lemmas ABOVE the theorem are encouraged. They are subject to the
  same no-sorry / no-axiom rules.

# CRITICAL: do not look for the answer on disk

The reference statements and, for some problems, reference ANSWERS exist
elsewhere on this machine. Finding and copying them is not solving.

- Stay inside `{bench_dir}`. Do NOT read, list, or grep any path under
  `{PB_DIR}` other than your own directory -- in particular NOT
  `lean4/src/`, NOT `solutions_replaced_new/`, and NOT `harness/`.
- Do NOT read sibling directories under `{RUNS_DIR}`; they are other
  problems, some already solved.
- Do NOT run `find`, `locate`, or `grep -r` outside `{bench_dir}`.
- Do NOT search the web.

If you are stuck, think harder about the mathematics, try a different
tactic, or build the proof through intermediate lemmas.

# Strategy

1. Read the file. Restate the goal to yourself in ordinary mathematics --
   the informal statement above is the same problem in prose.
2. Get the mathematical idea straight before writing tactics -- these are
   competition problems, and formalizing is easy once the argument is right,
   hopeless when it is not. But "the idea" means a sketch you could say out
   loud in a minute, not a complete formal derivation. Sketch, then test the
   first step against Lean.
3. Find the Mathlib lemmas you need. `exact?`, `apply?`, `simp?`, `rw?` and
   `open Nat in #check @...` are the fast way; `grep` in Mathlib source is
   the slow way and is allowed only under {LEAN4_DIR}/.lake/packages/mathlib.
4. Useful general-purpose closers: `norm_num`, `ring`, `field_simp`, `omega`
   (LINEAR integer/nat arithmetic only), `positivity`, `nlinarith`,
   `linarith`, `decide`, `interval_cases`, `fin_cases`, `aesop`.
5. Elaborate often. Read errors carefully -- Lean shows the unsolved goal.
6. When it elaborates clean, call `completed()`.
7. If it will not close no matter what you try, ask whether the goal is
   actually true -- and if you decide it is not, prove that (see above).

# Operating rules

- Work autonomously. Do not stop until `completed()` passes or you have
  filed a proven `disproved()`. Running out of ideas is not a stopping
  condition -- it is a signal to try a different angle.
- A `sorry` while debugging is fine; the final state must have none.
- Your trajectory is saved automatically.
- Write a final report only once one of the two finishing tools has told you
  that you are done. Say plainly which of the two happened.
"""


# --------------------------------------------------------------------------- #
# One problem
# --------------------------------------------------------------------------- #

class _Timeout(Exception):
    pass


def _solve_one(pb, problem: str, variant: str, args, agent_name: str) -> str:
    """Stage, run the agent, verify. Returns a queue status."""
    man = pb.load_manifest()
    if problem not in man:
        print(f"[harness] ERROR: unknown problem {problem}")
        return "failed"
    entry = man[problem]

    suffix = "-full" if variant == "full" else ""
    bench_dir = os.path.join(args.runs_dir, f"{problem}{suffix}")

    from types import SimpleNamespace
    pb.cmd_stage(SimpleNamespace(problem=problem, dest=bench_dir,
                                 variant=variant, force=True))

    print(f"[harness] bench_dir={bench_dir}")
    print(f"[harness] problem={problem} variant={variant}")
    print(f"[harness] agent_name={agent_name} max_steps={args.max_steps}")
    print(f"[harness] db={args.db} timeout={args.timeout}s", flush=True)

    model_cfg = getattr(args, "model_cfg", None) or DEEPSEEK_CFG
    print(f"[harness] model config: {model_cfg}", flush=True)
    cfg = agentlib.load_config(model_cfg)
    reg = agentlib.ToolRegistry()
    outcome = _Outcome()
    _make_bash_tool(reg, bench_dir)
    _make_completed_tool(reg, bench_dir, outcome)
    _make_disproved_tool(reg, bench_dir, problem, outcome)
    agent = agentlib.Agent(cfg, reg, db_path=args.db, agent_name=agent_name)

    user_input = (
        f"Solve the PutnamBench problem '{problem}' in {bench_dir}. "
        f"Replace the `sorry` in {problem}.lean with a real Lean 4 proof, "
        f"then call completed(). You have no turn limit and no time limit. "
        f"The run ends only when completed() reports PASS, or -- in the rare "
        f"case where the formalized goal is false -- when you have proven "
        f"that in Lean and called disproved()."
    )

    def should_continue(assistant_msg, step_idx):
        """Accept a stop only once a finishing tool has fired."""
        if outcome.kind is not None:
            return None
        return _nudge(outcome)

    def _alarm(signum, frame):
        raise _Timeout()

    timed_out = False
    result = None
    if args.timeout:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(args.timeout)
    max_steps = args.max_steps if args.max_steps > 0 else UNLIMITED_STEPS
    t0 = time.time()
    try:
        result = agent.run(
            user_input,
            system_prompt=_system_prompt(problem, bench_dir, entry, variant),
            max_steps=max_steps,
            connect_timeout=10.0, read_timeout=600.0, llm_retries=3,
            should_continue=should_continue,
        )
    except _Timeout:
        timed_out = True
        print(f"[harness] TIMEOUT after {args.timeout}s", flush=True)
    finally:
        if args.timeout:
            signal.alarm(0)
    elapsed = time.time() - t0

    # The definitive check is verify.sh run by US, not the agent's own claim
    # of success and not a "PASS" string echoed inside some file it happened
    # to cat. Re-run it here; it is cheap next to the agent loop.
    out = _run_verify(bench_dir)
    passed = _verify_passed(out)
    # The sandbox wrapper appends "[exit N]", so the last line is never the
    # reason. Pull the FAIL line itself -- that is what makes a queue note
    # worth reading.
    reason = next((l.strip() for l in reversed(out.splitlines())
                   if l.strip().startswith("FAIL [")), "")
    if not reason and out.strip():
        reason = out.strip().splitlines()[-1]

    print("=" * 70)
    print(f"problem        : {problem} ({variant})")
    print(f"agent_name     : {agent_name}")
    if result is not None:
        print(f"finish_reason  : {result.finish_reason}")
        print(f"num_steps      : {result.num_steps}")
        print(f"trajectory_id  : {result.trajectory_id}")
    print(f"elapsed        : {elapsed:.0f}s")
    print(f"passed         : {passed}")
    print(f"finishing tool : {outcome.kind or '(none called)'}")
    print(f"nudges         : {outcome.nudges}")
    if outcome.kind == "disproved":
        print(f"disproof file  : {outcome.disproof_file}")
        print(f"claim archived : {DEFECTS_DIR}/{problem}.claim.md")
    if not passed:
        print(f"verify said    : {reason or '(no output)'}")
    print("=" * 70, flush=True)

    fin = getattr(result, "finish_reason", None)
    _solve_one.last_steps = getattr(result, "num_steps", None)
    _solve_one.last_note = f"{elapsed:.0f}s"
    if passed:
        return "solved"

    # A filed disproof is its own outcome, not a failure: the run ended the
    # way we asked it to. It is NOT solved either -- nothing has checked the
    # claim yet -- so it gets its own status and waits for a human.
    if outcome.kind == "disproved" and not timed_out:
        _solve_one.last_note += " | DISPROOF CLAIMED (unverified) | " + \
            " ".join(outcome.explanation.split())[:70]
        return "disproved"

    # Strip the "FAIL [problem] " prefix; the row already names it.
    why = reason.split("] ", 1)[-1] if reason.startswith("FAIL [") else reason
    _solve_one.last_note += f" | {fin or ''} | {why[:70]}"
    return "timeout" if timed_out else "failed"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", help="solve this problem (skip the queue)")
    p.add_argument("--variant", choices=["wsolution", "full"],
                   default=None,
                   help="default: wsolution for --problem, else the queue row")
    p.add_argument("--loop", type=int, default=1,
                   help="how many problems to take from the queue "
                        "(0 = until it drains)")
    p.add_argument("--owner", default=None,
                   help="worker id recorded in the queue "
                        "(default: <host>-<pid>)")
    p.add_argument("--runs-dir", default=RUNS_DIR)
    p.add_argument("--agent-name", default=None)
    # Lean work is slow and bursty: a single elaboration is a minute, a real
    # proof search is many, and an agent that is making progress can spend
    # dozens of steps on one lemma. Capping either of these buys nothing but
    # false negatives -- a `max_steps` finish tells you about the cap, not the
    # problem. Both defaults are effectively "don't stop me".
    p.add_argument("--max-steps", type=int, default=0,
                   help="agent step cap; 0 = unlimited (default)")
    p.add_argument("--timeout", type=int, default=3 * 24 * 3600,
                   help="wall-clock seconds per problem "
                        "(default 3 days; 0 = none)")
    p.add_argument("--queue", default=None, help="path to problems.tsv")
    p.add_argument("--model-cfg", default=None,
                   help="path to a .nearai.* model config "
                        f"(default: {DEEPSEEK_CFG})")
    p.add_argument("--db", default=DB_PATH)
    args = p.parse_args()

    if args.queue:
        os.environ["PB_QUEUE"] = os.path.abspath(args.queue)

    if not os.path.exists(PB_PY):
        print(f"[harness] ERROR: {PB_PY} not found")
        return 2

    # Must precede any output: enter_agent_cgroup re-execs, so anything
    # printed before it appears twice -- and the pid we record in the queue
    # has to be the post-exec one, or the reaper would think we are dead.
    owner_label = args.owner or f"{socket.gethostname()}-{os.getpid()}"
    agentlib.refuse_if_low_ram(need_mb=2048, label=owner_label)
    entered = agentlib.enter_agent_cgroup(owner_label,
                                          max_ram_mb=LEAN_AGENT_RAM_MB,
                                          tasks_max=LEAN_BASH_TASKS)
    owner = args.owner or f"{socket.gethostname()}-{os.getpid()}"

    if not entered:
        print("[harness] WARNING: systemd user cgroups unavailable; commands "
              "run with a wall-clock limit only.", flush=True)

    pb = _load_pb()
    from types import SimpleNamespace

    # --problem bypasses the queue entirely.
    if args.problem:
        variant = args.variant or "wsolution"
        name = args.agent_name or f"lean-{args.problem}" + (
            "-full" if variant == "full" else "")
        status = _solve_one(pb, args.problem, variant, args, name)
        return 0 if status == "solved" else 1

    n_done = ok = 0
    while args.loop == 0 or n_done < args.loop:
        rc = pb.cmd_claim(SimpleNamespace(
            owner=owner, pid=os.getpid(), variant=args.variant,
            problem=None, stale_seconds=pb.STALE_SECONDS))
        if rc != 0:
            print("[harness] queue drained; nothing left to claim")
            break
        # cmd_claim printed "problem\tvariant"; re-read the row we own rather
        # than capturing stdout, so the two can never disagree.
        with pb._Lock(shared=True):
            mine = [r for r in pb._read_rows()
                    if r["status"] == "running" and r["owner"] == owner
                    and r["pid"] == str(os.getpid())]
        if not mine:
            print("[harness] ERROR: claimed a row but cannot find it")
            return 2
        row = mine[0]
        problem, variant = row["problem"], row["variant"]

        name = args.agent_name or f"lean-{problem}" + (
            "-full" if variant == "full" else "")
        try:
            status = _solve_one(pb, problem, variant, args, name)
        except KeyboardInterrupt:
            pb.cmd_release(SimpleNamespace(
                problem=problem, status="todo", variant=variant,
                steps=None, note="interrupted"))
            raise
        except Exception as e:                       # noqa: BLE001
            pb.cmd_release(SimpleNamespace(
                problem=problem, status="failed", variant=variant,
                steps=None, note=f"harness error: {e}"[:120]))
            raise

        pb.cmd_release(SimpleNamespace(
            problem=problem, status=status, variant=variant,
            steps=getattr(_solve_one, "last_steps", None),
            note=getattr(_solve_one, "last_note", "")))
        n_done += 1
        ok += (status == "solved")

    if n_done:
        print(f"[harness] {ok}/{n_done} solved this worker")
    return 0 if (n_done and ok == n_done) else 1


if __name__ == "__main__":
    sys.exit(main())
