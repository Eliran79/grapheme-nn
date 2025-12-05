---
id: backend-009
title: Implement fixed-k clique enumeration
status: todo
priority: medium
tags:
- backend
- algorithm
- clique
dependencies: []
assignee: developer
created: 2025-12-05T21:39:47.697874221Z
estimate: ~
complexity: 3
area: backend
---

# Implement fixed-k clique enumeration

## Context
For small k (3-6), brute-force clique enumeration is polynomial and practical. This is the foundation for k-Clique Percolation in GRAPHEME's concept detection.

**Research basis**: "If k is a fixed constant, the brute force algorithm... gives a O(n^k · k²) = n^O(1) algorithm." - Math Stack Exchange

## Objectives
- Implement O(n^k) clique enumeration for fixed k
- Optimize for sparse graphs (degeneracy-based)
- Support k=3, 4, 5, 6 for concept detection
- Integrate with GraphemeGraph

## Tasks
- [ ] Implement basic k-clique enumeration
- [ ] Add degeneracy ordering optimization
- [ ] Create `find_cliques(k)` method on GraphemeGraph
- [ ] Add iterator-based lazy enumeration for large graphs
- [ ] Write unit tests for k=3 to k=6
- [ ] Add benchmarks

## Acceptance Criteria
✅ **Correctness:**
- Finds all k-cliques (complete subgraphs of size k)
- No duplicates in output

✅ **Performance:**
- O(n^k) for dense graphs
- Much faster for sparse graphs (degeneracy optimization)
- Handles k=3 on 10000-node sparse graphs

✅ **Integration:**
- Works with GraphemeGraph and MathGraph
- Provides foundation for CPM (backend-008)

## Technical Notes

### Algorithm Pseudocode
```rust
impl GraphemeGraph {
    /// Find all k-cliques in the graph
    pub fn find_cliques(&self, k: usize) -> Vec<Vec<NodeIndex>> {
        let mut cliques = Vec::new();

        // Degeneracy ordering for efficiency
        let ordering = self.degeneracy_ordering();

        // For each node in ordering
        for v in &ordering {
            // Get higher-ordered neighbors
            let neighbors: Vec<_> = self.neighbors(*v)
                .filter(|u| ordering.position(u) > ordering.position(v))
                .collect();

            // Find (k-1)-cliques in neighbor subgraph
            for subset in neighbors.combinations(k - 1) {
                if self.is_clique(&subset) {
                    let mut clique = subset;
                    clique.push(*v);
                    cliques.push(clique);
                }
            }
        }

        cliques
    }

    fn is_clique(&self, nodes: &[NodeIndex]) -> bool {
        for i in 0..nodes.len() {
            for j in i+1..nodes.len() {
                if !self.has_edge(nodes[i], nodes[j]) {
                    return false;
                }
            }
        }
        true
    }
}
```

### Key Design Decisions
- Degeneracy ordering: process low-degree nodes first
- For sparse graphs, degeneracy d << n, so O(n · d^(k-1))
- Lazy iterator for memory efficiency on large graphs

### Files to Modify
- `grapheme-core/src/lib.rs`: Add `find_cliques()`, `degeneracy_ordering()`
- `grapheme-core/benches/core_bench.rs`: Add clique benchmarks

## Testing
- [ ] Test triangle (k=3) finding
- [ ] Test k=4, k=5, k=6
- [ ] Test on empty graph, complete graph, star graph
- [ ] Benchmark on sparse text graphs

## Updates
- 2025-12-05: Task created from algorithm research

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [Document code changes, new files, modified functions]

### Causality Impact
- Foundation for k-Clique Percolation
- Enables concept detection pipeline

### Dependencies & Integration
- Required by backend-008 (CPM)
- Pure Rust, no external dependencies

### Verification & Testing
- Test clique counts match known values
- Verify degeneracy optimization speedup
