---
id: backend-097
title: Replace Cross-Entropy with Structural Loss in Training
status: todo
priority: high
tags:
- backend
- training
- vision-alignment
dependencies:
- backend-096
assignee: developer
created: 2025-12-07T22:00:00Z
estimate: ~
complexity: 2
area: backend
---

# backend-097: Replace Cross-Entropy with Structural Loss in Training

## Problem Statement

Current training loop in `grapheme-train/src/bin/train.rs` uses cross-entropy:

```rust
// train.rs:303
// This uses the differentiable cross-entropy loss, NOT GED
```

This violates GRAPHEME_Vision.md which explicitly states:
```rust
// Not cross-entropy on tokens, but structural alignment
loss = α·node_insertion_cost + β·edge_deletion_cost + γ·clique_mismatch
```

## Objective

Replace cross-entropy training with the differentiable structural loss from backend-096.

## Current State (to be replaced)

```rust
// Current: Token-level classification
let loss = cross_entropy_loss(predicted_probs, target_label);
let grad = cross_entropy_gradient(predicted_probs, target_label);
```

## Target State

```rust
// Target: Structural alignment (per vision)
let loss_result = compute_structural_loss(&predicted_graph, &target_graph, &config);
let loss = loss_result.total_loss;  // α·node + β·edge + γ·clique
let grad = loss_result.gradients;
```

## Implementation Plan

### Phase 1: Training Loop Modification
- [ ] Import `compute_structural_loss` from backend-096
- [ ] Replace `cross_entropy_loss()` call with `compute_structural_loss()`
- [ ] Update gradient computation to use structural gradients
- [ ] Keep GED monitoring for comparison

### Phase 2: Backward Pass Integration
- [ ] Propagate structural gradients through graph operations
- [ ] Update `backward()` to handle graph-level gradients
- [ ] Ensure gradient flow to node features and edge weights

### Phase 3: Configuration Updates
- [ ] Add structural loss config to `ConfigFile`
- [ ] Add CLI flags for α, β, γ weights
- [ ] Add Sinkhorn parameters (iterations, temperature)
- [ ] Update default configs in `configs/`

### Phase 4: Hybrid Mode (Optional)
- [ ] Add `--loss-mode` flag: `structural`, `cross-entropy`, `hybrid`
- [ ] Hybrid: `total = λ·structural + (1-λ)·cross_entropy`
- [ ] Allow gradual transition during training

## Files to Modify

1. `grapheme-train/src/bin/train.rs`
   - Replace loss computation
   - Update gradient handling
   - Add config parsing

2. `grapheme-train/src/lib.rs`
   - Update `TrainingMetrics` struct
   - Modify `compute_edit_prediction_loss()` or deprecate
   - Add structural loss integration

3. `configs/*.yaml`
   - Add structural loss parameters
   - Update defaults

## API Changes

```rust
// New training config additions
pub struct TrainingConfig {
    // ... existing fields ...

    /// Loss mode: "structural", "cross_entropy", or "hybrid"
    pub loss_mode: LossMode,

    /// Structural loss weights (from GRAPHEME vision)
    pub alpha: f32,  // Node cost weight (default: 1.0)
    pub beta: f32,   // Edge cost weight (default: 0.5)
    pub gamma: f32,  // Clique cost weight (default: 2.0)

    /// Sinkhorn parameters
    pub sinkhorn_iterations: usize,  // Default: 20
    pub sinkhorn_temperature: f32,   // Default: 0.1

    /// Hybrid mode mixing (if loss_mode == "hybrid")
    pub hybrid_lambda: f32,  // Default: 0.5
}

pub enum LossMode {
    Structural,
    CrossEntropy,
    Hybrid,
}
```

## Migration Strategy

1. **Phase A**: Add structural loss alongside cross-entropy (both computed)
2. **Phase B**: Default to hybrid mode (50/50)
3. **Phase C**: Default to structural loss
4. **Phase D**: Deprecate cross-entropy path (optional)

## Dependencies
- backend-096 (differentiable structural loss implementation)

## Success Criteria
- [ ] Training converges with structural loss
- [ ] Loss decreases monotonically
- [ ] WL kernel similarity improves over epochs
- [ ] No regression in model quality vs cross-entropy baseline
- [ ] CLI flags work correctly

## Completion Checklist

> 1. All code compiles with no errors or warnings (cargo build, cargo clippy)
> 2. Document what was changed, what runtime behavior to expect, and what dependencies were affected

- [ ] Implementation complete
- [ ] Tests passing
- [ ] No clippy warnings
- [ ] Documentation updated
- [ ] Configs updated

### What Changed
- [To be filled on completion]

### Runtime Behavior
- [To be filled on completion]

### Dependencies Affected
- [To be filled on completion]
