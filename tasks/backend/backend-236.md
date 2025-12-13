---
id: backend-236
title: Fix incorrect Max aggregation backward pass
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

# Fix incorrect Max aggregation backward pass

## Context
The Max aggregation backward pass distributes gradients equally instead of routing only to the max element. This breaks gradient flow for max pooling operations.

## Problem Location
- `grapheme-train/src/backprop.rs:346-354`

## Current Code (WRONG)
```rust
AggregationType::Max => {
    // Max: gradient only flows to the max element
    // This requires knowing which element was max during forward
    // For now, distribute equally (simplified)  <-- WRONG!
    let n = input_indices.len() as f32;
    let scaled_grad = &grad / n;
    for idx in input_indices {
        self.accumulate_grad(*idx, scaled_grad.clone());
    }
}
```

## Fix
Store argmax during forward pass, route gradient only to max:
```rust
// In TapeOp::Aggregate, add:
max_idx: Option<usize>,  // Index of max element (for Max aggregation)

// In backward:
AggregationType::Max => {
    if let Some(max_idx) = max_idx {
        // Gradient flows only to the max element
        self.accumulate_grad(input_indices[max_idx], grad);
    }
}
```

## DAG Impact
- Incorrect gradients break optimization
- Max pooling can't learn properly
- Critical for graph pooling operations

## Acceptance Criteria
- [ ] Add `max_idx` field to Aggregate TapeOp
- [ ] Store argmax during forward pass
- [ ] Route gradient only to max element in backward
- [ ] Add test verifying correct gradient flow
- [ ] Verify existing tests still pass
