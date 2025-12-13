---
id: backend-198
title: Integrate backward pass in train_agi.rs
status: done
priority: high
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-11T11:32:40.149077461Z
estimate: ~
complexity: 5
area: backend
---

# Integrate backward pass in train_agi.rs

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
- Modified `grapheme-train/src/bin/train_agi.rs`
- Added imports: `BackwardPass`, `Embedding`, `InitStrategy`, `Array1`
- Replaced stub gradient logic with proper backward pass in `train_batch()`:
  - Creates `Embedding` for gradient tracking
  - Calls `dag.zero_grad()` at batch start
  - Computes MSE output gradients: `dL/dy = 2 * (y - target) / n`
  - Calls `dag.backward_accumulate()` for each example
  - Applies gradients with `dag.step(effective_lr)` after batch

### Causality Impact
- Training now actually updates edge weights via backpropagation
- Loss should decrease during training (previously no learning occurred)
- Gradient accumulation happens per-batch, not per-example

### Dependencies & Integration
- Uses `BackwardPass` trait from grapheme-core
- Uses `Embedding` struct for gradient accumulation context
- Requires `InitStrategy` for Embedding initialization
- Integrates with existing DagNN forward pass

### Verification & Testing
- Run: `cargo build -p grapheme-train --bin train_agi`
- Run: `cargo test -p grapheme-train`
- Test with data: create `data/mixed_agi/train.jsonl` with MixedExample format

### Context for Next Task
- The backward pass computes output gradients and accumulates them into `dag.edge_grads`
- `dag.step(lr)` applies accumulated gradients to edge weights
- Embedding is created per-batch (could be optimized to share across batches)
- Output gradients are computed as MSE gradient scaled by 1/n