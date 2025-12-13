---
id: backend-092
title: Implement differentiable graph transformation with backpropagation
status: done
priority: critical
tags:
- backend
- training
- neural-network
- differentiation
dependencies:
- backend-091
assignee: developer
created: 2025-12-07T20:00:00Z
estimate: ~
complexity: 5
area: backend
---

# Implement differentiable graph transformation with backpropagation

## Context

The training binary (`train.rs`) currently has a **placeholder training loop** that:
1. Computes GED loss between input and target graphs
2. Records the loss in metrics
3. **Does NOT apply gradients to update model weights**

The comment in `train.rs:305-308` explicitly states:
```rust
// Note: Full backpropagation would require differentiable graph operations
// This is a placeholder for the training loop structure
// The model.encode() method processes graphs, but doesn't produce
// output graphs directly - that requires additional generation logic
```

### Current State
- `GraphTransformNet` has:
  - `encode()` - produces node embeddings
  - `predict_edits()` - predicts edit operations per node
  - `apply_edits()` - applies edits to create output graph
  - `zero_grad()`, `step(lr)` - gradient infrastructure exists
- `Optimizer` trait has `step()` method that updates parameters
- `DagNN` has `backward()` and `backward_and_update()` methods
- Loss is computed via `compute_ged_loss()` but loss gradients aren't backpropagated

### Problem
Training doesn't learn because:
1. The model transforms input → output but this isn't differentiated
2. GED loss is computed on discrete graphs, not differentiable
3. No gradient flows from loss back to model parameters

## Objectives

1. Make the graph transformation differentiable
2. Connect loss computation to gradient flow
3. Update model weights during training

## Approach Options

### Option A: Differentiable GED Approximation
- Use soft matching instead of hard edit operations
- Approximate GED with differentiable operations
- Similar to differentiable optimal transport

### Option B: Supervised Edit Prediction
- Train edit prediction heads directly with cross-entropy loss
- Ground truth: compute actual edits between input/target graphs
- Simpler but requires edit supervision

### Option C: Reinforcement Learning
- Treat edit predictions as actions
- Use policy gradient with GED as reward
- More complex but handles discrete operations

### Option D: Straight-Through Estimator
- Use hard edits in forward pass
- Estimate gradients through discrete operations
- Common in quantization literature

## Recommended Approach

**Option B (Supervised Edit Prediction)** is recommended for initial implementation:
1. Compute ground-truth edit sequence from input→target
2. Train edit prediction heads with cross-entropy loss
3. This gives clear supervision signal and is easier to debug

## Tasks

### Phase 1: Ground Truth Edit Computation
- [x] Implement `compute_edit_sequence(input: &str, target: &str) -> EditSequence` (string-based, O(n*m) DP)
- [x] Handle node insertions, deletions, and modifications via `EditLabel` enum
- [x] Add tests for edit sequence computation (5 tests)

### Phase 2: Edit Prediction Loss
- [x] Add cross-entropy loss for edit operation prediction (`cross_entropy_loss()`)
- [x] Connect `NodeHead::predict_op_probs()` output to loss via `compute_edit_prediction_loss()`
- [x] Implement differentiable loss computation (`cross_entropy_gradient()`)

### Phase 3: Backpropagation Integration
- [x] Add backward pass through `GraphTransformNet`
- [x] Connect optimizer to model parameter updates
- [x] Update `train.rs` to use proper training loop

### Phase 4: Validation
- [x] Verify loss decreases during training
- [x] Verify model weights change
- [x] Add training convergence tests

## Acceptance Criteria

- [x] Loss decreases over epochs (currently stays constant)
- [x] Model weights are updated during training
- [x] Training produces a model that can transform expressions
- [x] All existing tests pass (580+ tests)
- [x] New tests for gradient flow (12 new tests added)

## Dependencies

- backend-091 (persistence infrastructure) - DONE

## Related

- `grapheme-core/src/lib.rs:4324` - `GraphTransformNet::encode()`
- `grapheme-core/src/lib.rs:4355` - `GraphTransformNet::predict_edits()`
- `grapheme-core/src/lib.rs:4376` - `GraphTransformNet::apply_edits()`
- `grapheme-train/src/bin/train.rs:274` - supervised edit prediction training loop
- `grapheme-train/src/lib.rs:1086` - `compute_edit_sequence()` (O(n*m) DP algorithm)
- `grapheme-train/src/lib.rs:1232` - `cross_entropy_loss()`
- `grapheme-train/src/lib.rs:1277` - `compute_edit_prediction_loss()`

## Session Handoff

**Implementation Summary:**
- Implemented supervised edit prediction (Option B from the task)
- Added `compute_edit_sequence()` using O(n*m) Levenshtein-based DP algorithm (NOT NP-hard)
- Added `cross_entropy_loss()` and `cross_entropy_gradient()` for differentiable training
- Updated `train.rs` to use cross-entropy loss on edit predictions instead of GED
- Model now calls `model.zero_grad()` and `model.step(lr)` each batch
- 12 new unit tests for edit prediction functionality

**Key Files Changed:**
- `grapheme-train/src/lib.rs` - Added ~300 lines for edit prediction infrastructure
- `grapheme-train/src/bin/train.rs` - Updated training loop to use supervised learning

**Complexity Analysis:**
- `compute_edit_sequence()`: O(n*m) where n=input length, m=target length
- `cross_entropy_loss()`: O(4) = O(1)
- `compute_edit_prediction_loss()`: O(batch * max(n*m, n*hidden))
- All operations are polynomial time - NO NP-hard algorithms
