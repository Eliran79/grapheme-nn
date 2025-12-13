---
id: backend-228
title: Pre-encode HumanEval to graph pairs
status: done
priority: high
tags:
- backend
- stage2
- humaneval
- benchmark
dependencies:
- backend-227
- backend-222
assignee: developer
created: 2025-12-12T17:29:43.630042192Z
estimate: 3h
complexity: 5
area: backend
---

# Pre-encode HumanEval to graph pairs

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
Pre-encodes HumanEval benchmark problems to graph pairs for pure graph-to-graph training.
This eliminates text processing from the training loop.

## Objectives
- Load HumanEval problems from JSONL format
- Convert prompt → CodeGraph (input)
- Convert prompt + solution → CodeGraph (output)
- Store as GraphPair in GraphDataset format
- Save to binary format for graph-only training

## Tasks
- [x] Create `HumanEvalProblem` struct matching JSONL format
- [x] Implement `HumanEvalEncoder` for loading and encoding
- [x] Add `encode_problem()` for single problem encoding
- [x] Add `encode_dataset()` for batch encoding with stats
- [x] Implement CodeGraph → DagNN conversion
- [x] Add complexity level heuristic based on solution size
- [x] Write comprehensive unit tests (9 tests)

## Acceptance Criteria
✅ **Problem Loading:**
- Parses HumanEval JSONL format correctly
- Handles all fields (task_id, prompt, canonical_solution, test, entry_point)

✅ **Graph Encoding:**
- Converts code to CodeGraph via CodeBrain
- Converts CodeGraph to DagNN for storage
- Input graph = prompt only, Output graph = full code

## Technical Notes
- Uses CodeBrain.parse_code() for code → CodeGraph
- code_graph_to_dagnn() converts to DagNN preserving structure
- Activation values encode node types (Function: 0.9, Literal: 0.3, etc.)
- Complexity levels 1-5 based on solution lines and chars
- HumanEvalEncodingResult tracks success rate and failures

## Testing
- [x] Write unit tests for new functionality (9 tests)
- [x] Ensure all tests pass before marking task complete (57 total)
- [x] Consider edge cases and error conditions

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
- [x] Commit changes incrementally with clear messages
- [x] Use descriptive commit messages that explain the "why"

## Updates
- 2025-12-12: Task created
- 2025-12-13: Task completed

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created new file: `grapheme-train/src/humaneval_encoder.rs`
- Updated `grapheme-train/src/lib.rs` with module and re-exports
- Key structures:
  - `HumanEvalProblem`: JSONL format with task_id, prompt, canonical_solution, test, entry_point
  - `HumanEvalEncoder`: Main encoder with load_problems() and encode_dataset()
  - `HumanEvalEncodingResult`: Stats with dataset, total, successes, failures
  - `EncodingFailure`: Details about failed encodings

### Causality Impact
- Encoding flow: JSONL → HumanEvalProblem → GraphPair (via CodeBrain)
- Code parsing: CodeBrain.parse_code() → CodeGraph → DagNN
- Node activations encode semantics: Function=0.9, Call=0.8, Literal=0.3, etc.

### Dependencies & Integration
- Uses CodeBrain from grapheme-code for code parsing
- Uses GraphPair/GraphDataset from graph_data module (backend-227)
- Re-exports from lib.rs: HumanEvalEncoder, HumanEvalProblem, HumanEvalEncodingResult, EncodingFailure

### Verification & Testing
- Run: `cargo test -p grapheme-train humaneval_encoder` - 9 tests pass
- Run: `cargo test -p grapheme-train --lib` - 57 total tests pass
- Clippy: `cargo clippy -p grapheme-train -- -D warnings` - 0 warnings

### Context for Next Task
- HumanEval data at: datasets/external/humaneval.parquet (or JSONL format)
- Use HumanEvalEncoder::encode_dataset() to create GraphDataset
- Save with dataset.save_binary() for graph-only training
- Complexity levels 1-5 help with curriculum learning
- Input graph is prompt only, output is full function code
