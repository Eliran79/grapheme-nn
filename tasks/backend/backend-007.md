---
id: backend-007
title: Implement BP2 quadratic GED approximation
status: todo
priority: medium
tags:
- backend
- algorithm
- ged
dependencies:
- backend-006
assignee: developer
created: 2025-12-05T21:39:40.905838228Z
estimate: ~
complexity: 3
area: backend
---

# Implement BP2 quadratic GED approximation

## Context
BP2 (Bipartite Matching with Hausdorff distance) provides a fast O(n²) upper bound for graph edit distance. This is useful for training where WL kernel might be too slow for large batches.

**Research basis**: "Quadratic-time methods perform equally well or, quite surprisingly, in some cases even better than the cubic-time method." - ScienceDirect

## Objectives
- Implement Hausdorff-based node matching
- Implement greedy edge assignment
- Provide fast O(n²) GED upper bound
- Use as alternative to WL for large batch training

## Tasks
- [ ] Implement node-to-node Hausdorff distance
- [ ] Implement greedy assignment algorithm
- [ ] Add `compute_bp2()` method to GraphEditDistance
- [ ] Compare accuracy with WL kernel
- [ ] Add benchmarks for BP2 vs WL performance
- [ ] Write unit tests

## Acceptance Criteria
✅ **Performance:**
- Runs in O(n²) time
- At least 2x faster than WL for graphs > 100 nodes

✅ **Accuracy:**
- Provides valid upper bound (never underestimates)
- Correlation with WL similarity > 0.8

✅ **Integration:**
- Can be selected as alternative loss function
- Works with batch training pipeline

## Technical Notes

### Algorithm Pseudocode
```rust
pub fn compute_bp2(g1: &MathGraph, g2: &MathGraph) -> Self {
    // 1. Compute node-to-node costs (Hausdorff-inspired)
    let node_costs = compute_node_costs(g1, g2);

    // 2. Greedy assignment (instead of Hungarian O(n³))
    let (assignment, unmatched_cost) = greedy_assign(&node_costs);

    // 3. Compute edge edit costs based on assignment
    let edge_cost = compute_edge_cost(g1, g2, &assignment);

    Self {
        node_insertion_cost: unmatched_cost.insertions,
        node_deletion_cost: unmatched_cost.deletions,
        edge_insertion_cost: edge_cost.insertions,
        edge_deletion_cost: edge_cost.deletions,
        node_mismatch_cost: assignment.mismatch_cost,
        ..Default::default()
    }
}
```

### Key Design Decisions
- Use greedy instead of Hungarian for O(n²) vs O(n³)
- Node cost based on label + degree
- Edge cost computed after node assignment

### Files to Modify
- `grapheme-train/src/lib.rs`: Add `compute_bp2()` to GraphEditDistance
- `grapheme-train/benches/train_bench.rs`: Add BP2 benchmarks

## Testing
- [ ] Test on known graph pairs with computed GED
- [ ] Verify upper bound property
- [ ] Benchmark against WL kernel
- [ ] Test scaling behavior

## Updates
- 2025-12-05: Task created from algorithm research

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [Document code changes, new files, modified functions]

### Causality Impact
- BP2 provides fast alternative for large batch training
- Can be mixed with WL for curriculum learning

### Dependencies & Integration
- Depends on WL kernel (backend-006) for correctness validation
- No external crate dependencies

### Verification & Testing
- Compare BP2 results with WL kernel
- Run benchmarks to verify O(n²) scaling
