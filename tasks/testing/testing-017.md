---
id: testing-017
title: Implement HumanEval evaluation harness with pass@1 scoring
status: todo
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
- [ ] Download and parse HumanEval dataset (164 problems)
- [ ] Implement evaluation harness that executes Python code safely
- [ ] Create pass@k calculation following OpenAI's unbiased estimator
- [ ] Add batch inference support for CortexMesh
- [ ] Generate evaluation report with per-problem breakdown

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
- [ ] Write unit tests for new functionality
- [ ] Write integration tests if applicable
- [ ] Ensure all tests pass before marking task complete
- [ ] Consider edge cases and error conditions

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [ ] **BEFORE committing**: Build, test, AND run the code to verify it works
  - Run `cargo build --release` (or `cargo build` for debug)
  - Run `cargo test` to ensure tests pass
  - **Actually run/execute the code** to verify runtime behavior
  - Fix all errors, warnings, and runtime issues
- [ ] Commit changes incrementally with clear messages
- [ ] Use descriptive commit messages that explain the "why"
- [ ] Consider creating a feature branch for complex changes
- [ ] Review changes before committing

**Testing requirements by change type:**
- Code changes: Build + test + **run the actual program/command** to verify behavior
- Bug fixes: Verify the bug is actually fixed by running the code, not just compiling
- New features: Test the feature works as intended by executing it
- Minor changes: At minimum build, check warnings, and run basic functionality

## Updates
- 2025-12-11: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [Document code changes, new files, modified functions]
- [What runtime behavior is new or different]

### Causality Impact
- [What causal chains were created or modified]
- [What events trigger what other events]
- [Any async flows or timing considerations]

### Dependencies & Integration
- [What dependencies were added/changed]
- [How this integrates with existing code]
- [What other tasks/areas are affected]

### Verification & Testing
- [How to verify this works]
- [What to test when building on this]
- [Any known edge cases or limitations]

### Context for Next Task
- [What the next developer/AI should know]
- [Important decisions made and why]
- [Gotchas or non-obvious behavior]