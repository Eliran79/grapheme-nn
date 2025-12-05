---
id: backend-021
title: Implement parallel graph operations (rayon/tokio)
status: todo
priority: high
tags:
- backend
- infrastructure
- parallelism
dependencies:
- api-010
assignee: developer
created: 2025-12-05T22:24:10.522788694Z
estimate: ~
complexity: 4
area: backend
---

# Implement parallel graph operations (rayon/tokio)

## Context
Implement the parallel-first architecture defined in api-010. Use rayon for CPU-bound data parallelism and tokio for async I/O operations.

## Objectives
- Implement ParallelGraph for all graph types
- Parallelize expensive operations (clique finding, GED, etc.)
- Add rayon/tokio dependencies to workspace
- Benchmark parallel speedup

## Tasks
- [ ] Add rayon dependency to workspace
- [ ] Implement `ParallelGraph` for GraphemeGraph
- [ ] Implement `ParallelGraph` for MathGraph
- [ ] Parallelize `find_cliques()` - parallel subset enumeration
- [ ] Parallelize `compute_ged()` - parallel node comparison
- [ ] Parallelize `forward_pass()` - parallel layer execution
- [ ] Add tokio for async memory operations
- [ ] Create parallel benchmarks
- [ ] Update existing tests for thread safety

## Acceptance Criteria
✅ **Performance:**
- Linear speedup on multi-core systems
- No regression on single-core
- Memory usage scales reasonably

✅ **Correctness:**
- Same results as sequential operations
- No data races (miri clean)
- Deterministic outputs

## Technical Notes

### Parallel Clique Finding
```rust
impl GraphemeGraph {
    pub fn find_cliques_parallel(&self, k: usize) -> Result<Vec<Vec<NodeIndex>>, CliqueError> {
        if k > MAX_CLIQUE_K {
            return Err(CliqueError::KTooLarge { requested: k, max: MAX_CLIQUE_K });
        }

        let ordering = self.degeneracy_ordering();

        // Parallel over starting nodes
        let cliques: Vec<Vec<NodeIndex>> = ordering
            .par_iter()
            .flat_map(|&v| {
                let neighbors: Vec<_> = self.neighbors(v)
                    .filter(|u| ordering.position(u) > ordering.position(&v))
                    .collect();

                neighbors.combinations(k - 1)
                    .filter(|subset| self.is_clique(subset))
                    .map(|mut subset| {
                        subset.push(v);
                        subset
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(cliques)
    }
}
```

### Parallel GED Computation
```rust
impl GraphEditDistance {
    pub fn compute_parallel(g1: &GraphemeGraph, g2: &GraphemeGraph) -> Self {
        // Parallel node comparison
        let node_costs: Vec<f32> = g1.par_nodes()
            .map(|n1| {
                g2.nodes()
                    .map(|n2| Self::node_distance(n1, n2))
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(1.0)
            })
            .collect();

        // Aggregate costs
        Self {
            node_mismatch_cost: node_costs.par_iter().sum(),
            // ... other fields
        }
    }
}
```

### Parallel Batch Training
```rust
impl Trainer {
    pub fn train_batch_parallel(&mut self, batch: &[Example]) {
        // Parallel forward pass
        let outputs: Vec<_> = batch
            .par_iter()
            .map(|example| self.model.forward(&example.input))
            .collect();

        // Parallel loss computation
        let losses: Vec<_> = outputs
            .par_iter()
            .zip(batch.par_iter())
            .map(|(output, example)| self.loss_fn.compute(output, &example.target))
            .collect();

        // Aggregate gradients (requires synchronization)
        let avg_loss = losses.par_iter().sum::<f32>() / losses.len() as f32;

        // Sequential parameter update (atomic)
        self.optimizer.step(avg_loss);
    }
}
```

### Workspace Cargo.toml Updates
```toml
[workspace.dependencies]
rayon = "1.8"
tokio = { version = "1", features = ["sync", "rt-multi-thread", "macros"] }
parking_lot = "0.12"  # Faster mutexes
```

### Files to Modify
- `Cargo.toml`: Add workspace dependencies
- `grapheme-core/Cargo.toml`: Add rayon
- `grapheme-core/src/lib.rs`: Implement ParallelGraph
- `grapheme-train/src/lib.rs`: Parallel training
- `grapheme-math/src/lib.rs`: Parallel MathGraph ops

## Testing
- [ ] Test parallel results match sequential
- [ ] Run with `RUST_TEST_THREADS=1` for determinism
- [ ] Run miri for data race detection
- [ ] Benchmark 1, 2, 4, 8 threads

## Updates
- 2025-12-05: Task created for AGI infrastructure

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### Dependencies & Integration
- Depends on: api-010 (parallel API design)
- Affects: All compute-heavy operations
- Crate changes: rayon, tokio added
