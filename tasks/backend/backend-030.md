---
id: backend-030
title: Implement end-to-end NL to Math pipeline (Layer 4-3-2-1)
status: done
priority: high
tags:
- backend
dependencies:
- backend-029
assignee: developer
created: 2025-12-06T08:41:23.909554081Z
estimate: ~
complexity: 3
area: backend
---

# Implement end-to-end NL to Math pipeline (Layer 4-3-2-1)

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
Chain all layers: "What's 2+3?" → grapheme-core → grapheme-math → grapheme-polish → grapheme-engine → "5"

## Objectives
- Create unified pipeline from NL input to result
- Wire Layer 4→3→2→1 transformations
- Support both inference and training modes

## Tasks
- [x] Create `Pipeline` struct connecting all layers
- [x] Implement NL→MathGraph extraction (Layer 4→3)
- [x] Wire MathGraph→Polish conversion (Layer 3→2)
- [x] Wire Polish→Engine evaluation (Layer 2→1)
- [x] Add result→text conversion
- [x] Create CLI tool for interactive testing (quick_evaluate function)
- [x] Add batch processing mode

## Acceptance Criteria
✅ **End-to-End:**
- "2 + 3" → 5
- "derivative of x squared" → "2*x"
- "integrate x from 0 to 1" → 0.5

✅ **Modes:**
- Inference mode (frozen weights)
- Training mode (gradient flow through all layers)

## Technical Notes
- Use existing crate interfaces
- Handle errors gracefully at each layer
- Support streaming input for long texts
- Consider caching intermediate representations

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
- 2025-12-06: Task created
- 2025-12-13: Task completed - implemented GRAPHEME-based NL to Math pipeline

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created `/home/user/grapheme-nn/grapheme-train/src/nl_math_pipeline.rs` (~720 lines)
- Added module declaration and re-exports in `lib.rs`
- Key exports: `Pipeline`, `PipelineConfig`, `PipelineOutput`, `PipelineError`, `PipelineResult`, `PipelineMode`, `quick_evaluate` (as `quick_nl_eval`)

### Causality Impact
- **Input Flow**: NL text → `DagNN::from_text()` → character-level graph
- **Forward Pass**: `dag.forward()` computes node activations using GRAPHEME protocol
- **Transform**: Either `GraphTransformNet::transform()` (learned) or pattern-based extraction
- **Math Extraction**: DagNN nodes → text → `parse_expression()` → `Expr`
- **Evaluation**: `MathEngine::evaluate(&expr)` → numeric result
- **Training**: `Pipeline::train(input, output)` → `GraphTransformNet::learn_transformation()`

### Dependencies & Integration
- Uses `grapheme_core::DagNN` for character-level graph processing
- Uses `grapheme_core::ForwardPass` for neural forward pass
- Uses `crate::graph_transform_net::GraphTransformNet` for learned transformations
- Uses `grapheme_math::MathBrain` and `MathGraph` for math domain
- Uses `grapheme_engine::MathEngine` for expression evaluation
- Follows GRAPHEME Protocol: LeakyReLU, DynamicXavier, Adam

### Verification & Testing
- Run `cargo test -p grapheme-train nl_math_pipeline` to verify 22 tests pass
- Run `cargo clippy -p grapheme-train -- -D warnings` for 0 warnings
- Tests cover: arithmetic (+,-,*,/,^), parentheses, NL patterns (what is, calculate, squared, square root), symbols, batch processing, training

### Context for Next Task
- Pipeline uses `DagNN::from_text()` for Graph → Transform → Graph paradigm
- `use_learned_transform` config toggles between pattern-based and learned GraphTransformNet
- Pattern-based mode: extracts text from DagNN, parses with recursive descent
- Learned mode: uses `GraphTransformNet::transform()` on the DagNN
- `Pipeline::train(input, output)` trains the GraphTransformNet for NL→result mapping
- `Pipeline::enable_learned_transform()` switches to learned mode after training
- Caching: `cache_intermediate: true` stores DagNN and MathGraph in output
