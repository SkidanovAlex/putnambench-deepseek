# putnambench-deepseek

An autonomous DeepSeek-driven agent that solved PutnamBench in Lean 4:
**672 problems solved and machine-validated** — all 670 solvable upstream
problems it attempted, plus corrected variants of the two problems whose
upstream formalizations turned out to be defective (below).

## The agent

Two files:

- **`agentlib.py`** — self-contained agent framework: OpenAI-compatible chat
  loop with tool calling, SQLite trajectory capture, cgroup-sandboxed bash
  (RAM / fork / wall-clock caps, optional Landlock filesystem allowlists),
  and prefix-cache-friendly context compaction (when the conversation
  exceeds 80% of the context window, the entire history is summarized into
  one message and the run continues).
- **`run_putnam_solver.py`** — the PutnamBench harness. Workers pull
  problems from a flock-protected queue, so any number can run in parallel.

The agent gets a bash tool (Lean elaboration against a prebuilt Mathlib via
`lake env lean`) and exactly two ways to finish:

- **`completed()`** — runs the verifier: no `sorry`/`axiom`/`native_decide`,
  theorem statement byte-identical to the frozen one, file elaborates,
  `#print axioms` clean. Only a PASS ends the run; a failure returns the
  verifier's exact output and the run continues.
- **`disproved()`** — for the rare case the formalized statement is FALSE:
  the agent must supply a Lean proof deriving `False` from the statement
  (reviewed by a human, and how one of the two defects below was caught).

There is no step or token budget, and the harness owns the stopping rule:
if the model stops without finishing, it is told to continue. This is the
single most important design choice — several problems were solved after
**thousands of refused stop attempts** (one after 3,423 of them, on step
5,105, hour 26). Solutions were additionally re-validated by a semantic
checker: the solved theorem's fully-elaborated type (`pp.all`, clean
environment) must be character-identical to the pristine statement's, which
defeats statement-rewriting exploits that textual diffs miss.

## Models

DeepSeek V4 flash solved the bulk; V4 pro was used for the hardest tail
(retry pools). Of the 672 solved problems, the successful attempt ran on:

| model | problems solved |
|---|---|
| DeepSeek V4 flash | 653 |
| DeepSeek V4 pro | 19 |

## Cost

Estimated from full trajectory replay with prefix-cache accounting
(flash $0.0028/$0.14/$0.28, pro $0.003625/$0.435/$0.87 per 1M
cached/input/output tokens):

- **Total: ~$112** for 727 attempts across 672 problems
- Per problem: mean **$0.17**, median **$0.04**, max **$11.18**
  (`putnam_2021_a6`, 4 attempts, solved on step 5,411 of a 78-hour run)

Marathon runs are affordable because compaction keeps the growing context
append-only between resets, so ~99.8% of input tokens hit the prefix cache.

## Two defective upstream formalizations

1. **`putnam_1974_b1`** (5 points on a circle maximizing pairwise
   distances): the official solution predicate — "some ordering has equal
   consecutive distances" — is satisfied by degenerate configurations with
   repeated points (an equilateral triangle listed as `[A,B,C,A,B]`), which
   are not maximizers, so the theorem as stated is **false**. The repo
   includes context in the agent's defect artifacts; a machine-checked
   proof of `statement → False` was produced (clean axioms), matching
   upstream issue #347. Fixed by adding `Function.Injective p` to the
   predicate; the corrected theorem was then proved twice independently.
2. **`putnam_2013_a5`** (area-definite lists, R² → R³): the statement pins
   3D area to `μH[2]` on `Fin 3 → ℝ`, whose Pi instance carries the **sup
   metric** — making "area" the plane-dependent Busemann–Hausdorff measure
   rather than Euclidean area. The resulting statement appears true but is
   not the informal problem and requires convex-geometry theory absent from
   Mathlib. Fixed by moving to `EuclideanSpace ℝ (Fin 3)` (the residual
   dimensional constant cancels in the homogeneous inequality); the
   corrected theorem was proved twice independently within ~5 hours each,
   while the original resisted multi-day attempts.
