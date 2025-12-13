---
id: backend-142
title: Add gradient accumulator and zero_grad/step API to DagNN
status: done
priority: high
tags:
- backend
- core
- training
- gradients
dependencies:
- backend-139
assignee: developer
created: 2025-12-10T19:43:30.632623960Z
estimate: 4h
complexity: 7
area: backend
---

# Add gradient accumulator and zero_grad/step API to DagNN

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
DagNN currently lacks the infrastructure needed for proper gradient-based training. The
`backward_and_update()` method computes gradients and applies them immediately, with no
way to accumulate gradients across mini-batches or separate the backward pass from the
optimization step.

**Current pattern (broken for training):**
```rust
sample → backward_and_update() → immediate update, gradients discarded
```

**Required pattern (standard training):**
```rust
sample → backward() → accumulate grad → step(lr) → zero_grad()
```

This task adds the foundational gradient storage and API that backend-143 will use to
implement the `Learnable` trait, ultimately enabling backend-140 (joint training).

## Objectives
- Add persistent gradient storage to DagNN for edge weights
- Implement `zero_grad()` method to clear accumulated gradients
- Implement `step(lr)` method to apply accumulated gradients
- Modify `backward()` to accumulate gradients instead of returning transient struct

## Tasks
- [x] Add `edge_grads: HashMap<(NodeId, NodeId), f32>` field to DagNN struct
- [x] Add `requires_grad: bool` field to DagNN struct
- [x] Implement `zero_grad(&mut self)` to clear edge_grads HashMap
- [x] Implement `step(&mut self, lr: f32)` to apply edge_grads to Edge.weight
- [x] Add `backward_accumulate()` method to accumulate into edge_grads
- [x] Keep `backward_and_update()` for backwards compatibility (marked deprecated)
- [x] Add `gradient_norm(&self) -> f32` method
- [x] Add `has_gradients(&self) -> bool` method
- [x] Update DagNN::new() to initialize gradient storage
- [x] Add `clip_gradients(&mut self, max_norm: f32)` method
- [x] Add `num_parameters(&self) -> usize` method
- [x] Add `train(mode: bool)` and `is_training()` methods

## Acceptance Criteria
✅ **Criteria 1:**
- `dag.zero_grad()` clears all accumulated gradients

✅ **Criteria 2:**
- `dag.backward(output_grad, embedding)` accumulates gradients without applying them

✅ **Criteria 3:**
- `dag.step(lr)` applies accumulated gradients to edge weights

✅ **Criteria 4:**
- Calling backward() multiple times accumulates (sums) gradients

✅ **Criteria 5:**
- `dag.gradient_norm()` returns L2 norm of accumulated gradients

## Technical Notes
- **Edge gradient storage**: Use `HashMap<EdgeIndex, f32>` to match Edge.weight type
- **Embedding gradients**: Embedding already has proper gradient accumulation - follow that pattern
- **NodeGradients struct**: Keep for internal use, but accumulate into DagNN fields
- **Thread safety**: Not required for this task (single-threaded training)

**Reference: Embedding pattern (lines 4990-5130 in lib.rs):**
```rust
pub struct Embedding {
    pub weights: Array2<f32>,
    pub grad: Option<Array2<f32>>,  // Gradient accumulator
    pub requires_grad: bool,
}

fn zero_grad(&mut self) { self.grad = None; }
fn step(&mut self, lr: f32) {
    if let Some(ref grad) = self.grad {
        self.weights = &self.weights - &(grad * lr);
    }
}
```

## Testing
- [x] Write unit tests for new functionality (10 tests added)
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete
- [x] Consider edge cases and error conditions

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
- 2025-12-10: Task created
- 2025-12-10: Task completed - all gradient accumulation APIs implemented

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
**File:** `grapheme-core/src/lib.rs`

**DagNN struct changes (lines 707-733):**
- Added `edge_grads: HashMap<(NodeId, NodeId), f32>` - gradient accumulator
- Added `requires_grad: bool` - training mode flag

**New methods in DagNN (lines 1078-1205):**
- `zero_grad()` - Clear all accumulated gradients
- `step(lr: f32)` - Apply accumulated gradients to edge weights (SGD update)
- `gradient_norm() -> f32` - L2 norm of accumulated gradients
- `has_gradients() -> bool` - Check if any gradients accumulated
- `num_parameters() -> usize` - Count of trainable edge weights
- `clip_gradients(max_norm: f32) -> f32` - Gradient clipping
- `accumulate_edge_grad(from, to, grad)` - Manually accumulate a gradient
- `get_edge_grad(from, to) -> Option<f32>` - Query accumulated gradient
- `train(mode: bool)` - Set training mode
- `is_training() -> bool` - Check training mode

**BackwardPass trait changes (lines 5968-6097):**
- Added `backward_accumulate()` method to accumulate gradients into DagNN's internal storage
- Deprecated `backward_and_update()` in favor of `backward_accumulate() + step()`

**Tests added (lines 13342-13554):**
- `test_dagnn_zero_grad` - Verifies zero_grad clears gradients
- `test_dagnn_gradient_accumulation` - Verifies gradients sum across calls
- `test_dagnn_gradient_norm` - Verifies L2 norm calculation
- `test_dagnn_clip_gradients` - Verifies gradient clipping
- `test_dagnn_step_updates_weights` - Verifies step applies gradients correctly
- `test_dagnn_num_parameters` - Verifies parameter count
- `test_dagnn_train_mode` - Verifies train/eval mode switching
- `test_dagnn_backward_accumulate_respects_requires_grad` - Verifies eval mode skips accumulation
- `test_dagnn_full_training_loop` - Integration test for complete training pattern

### Causality Impact
**Training loop pattern now supported:**
```rust
for batch in data {
    dag.zero_grad();  // Clear previous gradients
    // ... forward pass ...
    dag.backward_accumulate(&loss_grad, &mut embedding);  // Accumulate gradients
    dag.clip_gradients(1.0);  // Optional: clip gradients
    dag.step(learning_rate);  // Apply updates
}
```

**Inference mode:**
```rust
dag.train(false);  // Disable gradient accumulation
// ... forward pass only, no gradients computed ...
```

### Dependencies & Integration
- No new crate dependencies
- All existing tests pass (900+ tests)
- `backward_and_update()` still works but is deprecated
- Edge gradients stored by `(NodeId, NodeId)` tuple, matching `NodeGradients.edge_grads` format

### Verification & Testing
```bash
# Run all gradient tests
cargo test -p grapheme-core test_dagnn

# Verify full test suite
cargo test
```

All 900+ tests pass including 10 new gradient-specific tests.

### Context for Next Task (backend-143)
**For implementing Learnable trait:**
- DagNN now has all the methods needed: `zero_grad()`, `step()`, `num_parameters()`, `has_gradients()`, `gradient_norm()`
- Only need to add `impl Learnable for DagNN` that delegates to these methods
- `clip_gradients()` is already implemented as a standalone method
- Reference: Embedding's Learnable impl at lines 5094-5130

**Key design decisions:**
1. Edge gradients stored as `(NodeId, NodeId) -> f32` rather than `EdgeIndex -> f32` for consistency with `NodeGradients`
2. `backward_accumulate()` added as new method rather than changing `backward()` signature to avoid breaking changes
3. `requires_grad` defaults to `true` (training mode) for new DagNN instances
4. Fields use `#[serde(skip)]` to avoid serializing transient gradient state