---
id: backend-199
title: Fix accumulate_gradients in SharedAGIModel to use real backward pass
status: done
priority: high
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-11T11:32:40.343105574Z
estimate: ~
complexity: 5
area: backend
---

# Fix accumulate_gradients in SharedAGIModel to use real backward pass

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
Brief description of what needs to be done and why.

## Objectives
- Clear, actionable objectives
- Measurable outcomes
- Success criteria

## Tasks
- [ ] Break down the work into specific tasks
- [ ] Each task should be clear and actionable
- [ ] Mark tasks as completed when done

## Acceptance Criteria
✅ **Criteria 1:**
- Specific, testable criteria

✅ **Criteria 2:**
- Additional criteria as needed

## Technical Notes
- Implementation details
- Architecture considerations
- Dependencies and constraints

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
- Modified `grapheme-train/src/bin/train_unified_agi.rs`
- Added imports: `BackwardPass`, `Embedding`, `InitStrategy`, `NodeId`, `Array1`
- Restructured `SharedAGIModel`:
  - Replaced `gradients: HashMap<String, Vec<f32>>` with `embedding: Embedding`
  - Added `output_node_ids: HashMap<String, Vec<NodeId>>` to track output nodes per brain
- Updated `accumulate_gradients()` to accept `target_activations` parameter
  - Now computes MSE gradients: `dL/dy = 2 * (y - target) / n`
  - Calls `dag.backward_accumulate()` for proper backprop
- Added `zero_grad()` method
- Updated `apply_gradients()` to use `dag.step(lr)` and zero gradients
- Updated training loop to call `zero_grad()` at epoch start and pass target_activations

### Causality Impact
- Training now uses proper gradient-based learning for brain slices
- Gradients flow through DagNN graph structure, not just activations
- Loss should decrease across epochs (previously gradients were synthetic)

### Dependencies & Integration
- Uses `BackwardPass` trait from grapheme-core
- Uses `Embedding` for gradient tracking
- Brain slice output nodes are tracked for proper gradient computation
- Integrates with CognitiveBrainOrchestrator for slice allocation

### Verification & Testing
- Run: `cargo build -p grapheme-train --bin train_unified_agi`
- Run: `cargo run --release -p grapheme-train --bin train_unified_agi`
- Expected: Loss values decrease across epochs (meaningful learning)

### Context for Next Task
- Output gradients are computed per brain slice using the slice's output_node_ids
- Gradients incorporate structural loss signal via `loss.sqrt()` scaling
- Embedding is shared across all brains within SharedAGIModel
- Training loop calls zero_grad() per epoch, apply_gradients() after all examples