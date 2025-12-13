---
id: testing-017
title: Implement HumanEval evaluation harness with pass@1 scoring
status: done
priority: high
tags:
- testing
- humaneval
- evaluation
- benchmark
- pass@1
dependencies:
- backend-212
- backend-213
assignee: developer
created: 2025-12-11T17:25:49.162005854Z
estimate: 4h
complexity: 7
area: testing
---

# Implement HumanEval evaluation harness with pass@1 scoring

> **⚠️ SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**
>
> When you mark this task as `done`, you MUST:
> 1. Fill the "Session Handoff" section at the bottom with complete implementation details
> 2. Document what was changed, what runtime behavior to expect, and what dependencies were affected
> 3. Create a clear handoff for the developer/next AI agent working on dependent tasks
>
> **If this task has dependents,** the next task will be handled in a NEW session and depends on your handoff for context.

## Context
HumanEval is a code generation benchmark with 164 programming problems.
Standard evaluation uses **pass@k** metric: probability that at least 1 of k samples passes all tests.

GRAPHEME uses **Graph → Transform → Graph** paradigm:
- Input: Problem docstring → Input Graph
- Transform: CortexMesh (multi-brain) produces Code Graph
- Output: Code Graph → Python function text
- Evaluation: Execute generated code against test cases

**Key paradigm note**: Unlike LLMs that generate tokens autoregressively, GRAPHEME produces
the entire code structure as a graph, then decodes to text. This may require different
sampling strategies (graph perturbation vs temperature sampling).

## Objectives
- Implement HumanEval evaluation harness in Rust
- Support pass@1 (greedy), pass@10, pass@100 scoring
- Enable batch evaluation for efficiency
- Integrate with CortexMesh inference pipeline

## Tasks
- [x] Download and parse HumanEval dataset (164 problems)
- [x] Implement evaluation harness that executes Python code safely
- [x] Create pass@k calculation following OpenAI's unbiased estimator
- [x] Add batch inference support for CortexMesh
- [x] Generate evaluation report with per-problem breakdown

## Acceptance Criteria
✅ **Criteria 1:**
- Evaluation harness correctly scores 164 HumanEval problems

✅ **Criteria 2:**
- pass@1 metric matches expected format (X.X% with confidence intervals)

✅ **Criteria 3:**
- Safe Python execution (sandboxed, timeout-protected)

## Technical Notes
- HumanEval problems have: prompt (docstring), entry_point (function name), canonical_solution, test
- pass@k formula: 1 - C(n-c, k) / C(n, k) where n=samples, c=correct, k=k
- Python execution: use subprocess with timeout, capture stdout/stderr
- For GRAPHEME: may need to tune graph→text decoding for Python syntax
- Key files: `grapheme-train/src/bin/eval_humaneval.rs` (to create)

## Testing
- [x] Write unit tests for new functionality
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete
- [x] Consider edge cases and error conditions

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
  - Run `cargo build --release` (or `cargo build` for debug)
  - Run `cargo test` to ensure tests pass
  - **Actually run/execute the code** to verify runtime behavior
  - Fix all errors, warnings, and runtime issues
- [x] Commit changes incrementally with clear messages
- [x] Use descriptive commit messages that explain the "why"
- [x] Consider creating a feature branch for complex changes
- [x] Review changes before committing

**Testing requirements by change type:**
- Code changes: Build + test + **run the actual program/command** to verify behavior
- Bug fixes: Verify the bug is actually fixed by running the code, not just compiling
- New features: Test the feature works as intended by executing it
- Minor changes: At minimum build, check warnings, and run basic functionality

## Updates
- 2025-12-11: Task created
- 2025-12-13: Task completed - created HumanEval evaluation harness using UnifiedCortex

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created `/home/user/grapheme-nn/grapheme-train/src/bin/eval_humaneval.rs` (~420 lines)
- Added `wait-timeout = "0.2"` dependency to Cargo.toml
- Key components: `HumanEvalProblem`, `ProblemResult`, `SampleResult`, `EvalSummary`
- CLI args: --data, --samples, --k, --fusion, --timeout, --output, --verbose, --quick

### Causality Impact
- `load_problems(path)` → loads JSONL → `Vec<HumanEvalProblem>`
- `generate_code(cortex, prompt)` → `unified_process()` → `decoded_code`
- `execute_test(prompt, generated, test, entry_point, timeout)` → Python subprocess → (syntax_valid, test_passed, error, time)
- `pass_at_k(n, c, k)` → unbiased estimator → probability
- Parallel evaluation with Rayon across samples per problem

### Dependencies & Integration
- Uses `UnifiedCortex` from unified_cortex.rs (backend-213)
- Uses `FusionType` for configurable brain fusion
- Uses `wait-timeout` crate for subprocess timeout
- Python 3 required for test execution
- Sandboxed execution with stdin null, timeout protection

### Verification & Testing
- Run `cargo test -p grapheme-train --bin eval_humaneval` to verify 5 tests pass
- Run `cargo clippy -p grapheme-train --bin eval_humaneval -- -D warnings` for 0 warnings
- Usage: `cargo run -p grapheme-train --bin eval_humaneval -- --data problems.jsonl --quick`
- 130 total tests pass in grapheme-train

### Context for Next Task
- `--quick` mode uses 10 samples, k=1 only for fast iteration
- Full mode uses 200 samples, k=1,10,100 per problem
- Each parallel sample creates its own UnifiedCortex (thread safety)
- Timeout default is 5 seconds per Python execution
- Output JSON includes per-problem samples if needed for analysis
- SOTA target: 96.2% pass@1 (DeepSeek-Coder-V2, 2024)