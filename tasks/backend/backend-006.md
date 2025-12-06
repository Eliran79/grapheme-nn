---
id: backend-006
title: Implement Weisfeiler-Leman kernel for GED loss
status: done
priority: high
tags:
- backend
- algorithm
- ged
dependencies: []
assignee: developer
created: 2025-12-05T21:39:37.483416727Z
estimate: ~
complexity: 4
area: backend
---

# Implement Weisfeiler-Leman kernel for GED loss

## Context
The current GED implementation (grapheme-train/src/lib.rs:822) only compares node/edge counts in O(1) time without actual graph structure comparison. The Weisfeiler-Leman (WL) kernel provides a polynomial-time graph similarity measure that is the theoretical foundation for Graph Neural Networks.

**Research basis**: "GNNs can be viewed as a neural version of the 1-WL algorithm, where continuous feature vectors replace colors, and neural networks are used to aggregate over local node neighborhoods." - JMLR

## Objectives
- Implement 1-WL color refinement algorithm
- Create graph kernel for similarity computation
- Replace current O(1) count-based GED with WL-based similarity
- Complexity: O(n·m·k) where n=nodes, m=edges, k=iterations

## Tasks
- [ ] Implement color/label initialization from node types
- [ ] Implement neighbor aggregation (color refinement step)
- [ ] Implement histogram comparison for graph similarity
- [ ] Add `compute_wl()` method to GraphEditDistance
- [ ] Update loss computation in training pipeline
- [ ] Add benchmarks for WL kernel performance
- [ ] Write unit tests for WL correctness

## Acceptance Criteria
✅ **Algorithm Correctness:**
- WL kernel produces same similarity for isomorphic graphs
- Similarity decreases with structural differences

✅ **Performance:**
- Runs in O(n·m·k) time
- Handles graphs up to 1000 nodes efficiently

✅ **Integration:**
- Works with both MathGraph and GraphemeGraph
- Benchmarks show reasonable training loss convergence

## Technical Notes

### Algorithm Pseudocode
```rust
pub fn compute_wl(g1: &MathGraph, g2: &MathGraph, iterations: usize) -> f32 {
    // 1. Initialize colors from node types
    let mut colors1 = init_colors(g1);
    let mut colors2 = init_colors(g2);

    // 2. Iterate color refinement
    for _ in 0..iterations {
        colors1 = refine_colors(g1, &colors1);
        colors2 = refine_colors(g2, &colors2);
    }

    // 3. Compare color histograms
    histogram_similarity(&colors1, &colors2)
}

fn refine_colors(g: &MathGraph, colors: &[u64]) -> Vec<u64> {
    // For each node, hash its color with sorted neighbor colors
    // This is the WL color refinement step
}
```

### Key Design Decisions
- Use hash-based color compression for efficiency
- Store color histograms at each iteration for multi-scale comparison
- Default to 3-5 iterations (sufficient for most expression graphs)

### Files to Modify
- `grapheme-train/src/lib.rs`: Add `compute_wl()` to GraphEditDistance
- `grapheme-train/benches/train_bench.rs`: Add WL kernel benchmarks

## Testing
- [ ] Test isomorphic graphs have similarity 1.0
- [ ] Test completely different graphs have low similarity
- [ ] Test intermediate cases with known structure
- [ ] Benchmark scaling with graph size

## Updates
- 2025-12-05: Task created from algorithm research

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added `WeisfeilerLehmanKernel` struct in `grapheme-train/src/lib.rs`
- Implements O(n·m·k) graph similarity with color refinement
- Added `compute_wl()` and `compute_wl_math()` to `GraphEditDistance`
- Added `compute_combined()` for hybrid count+WL loss
- Added petgraph dependency to grapheme-train/Cargo.toml
- 12 new tests for WL kernel (27 total tests in grapheme-train)

### Causality Impact
- WL kernel provides training loss signal
- Affects gradient flow in graph-to-graph learning
- Similarity ranges 0.0 (different) to 1.0 (identical)

### Dependencies & Integration
- Pure Rust implementation using std::hash
- Integrates with existing GraphEditDistance struct
- Requires petgraph for graph traversal

### Verification & Testing
- Run `cargo test -p grapheme-train` for unit tests
- All 27 tests passing with 0 warnings

### Context for Next Task
- backend-007 (BP2) can use WL as reference for correctness
- WL kernel becomes primary loss function for training