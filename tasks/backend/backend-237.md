---
id: backend-237
title: Fix incomplete Linear backward pass
status: todo
priority: critical
tags:
- backend
- backprop
- dag-optimization
dependencies: []
assignee: developer
created: 2025-12-13T00:00:00Z
estimate: 2h
complexity: 3
area: backend
---

# Fix incomplete Linear backward pass

## Context
The Linear operation backward pass assumes identity weights and ignores actual weight matrix. Weight gradients are never computed, so weights can't learn.

## Problem Location
- `grapheme-train/src/backprop.rs:359-367`

## Current Code (WRONG)
```rust
TapeOp::Linear {
    input_idx,
    output_idx,
    weight_id: _,  // Ignored!
} => {
    // Linear: gradient flows through (simplified - assumes identity weights)
    if let Some(grad) = self.gradients.get(output_idx).cloned() {
        self.accumulate_grad(*input_idx, grad);  // Just passes through!
    }
}
```

## Fix
Multiply gradient by weight matrix and accumulate weight gradients:
```rust
TapeOp::Linear {
    input_idx,
    output_idx,
    weight_id,
} => {
    if let (Some(grad), Some(input), Some(weights)) = (
        self.gradients.get(output_idx).cloned(),
        self.values.get(input_idx).cloned(),
        self.weights.get(weight_id),
    ) {
        // Gradient w.r.t. input: grad @ weights.T
        let input_grad = grad.dot(&weights.t());
        self.accumulate_grad(*input_idx, input_grad);

        // Gradient w.r.t. weights: input.T @ grad
        let weight_grad = input.t().dot(&grad);
        self.accumulate_weight_grad(*weight_id, weight_grad);
    }
}
```

## DAG Impact
- Weights in linear layers can't learn
- Embedding projections don't update
- Critical for any learning to happen

## Acceptance Criteria
- [ ] Store weight matrices in Tape
- [ ] Compute gradient w.r.t. input correctly
- [ ] Compute and accumulate weight gradients
- [ ] Add `get_weight_grads()` method
- [ ] Add gradient check test for Linear
- [ ] Verify training actually updates weights
