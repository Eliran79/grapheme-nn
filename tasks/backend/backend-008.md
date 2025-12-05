---
id: backend-008
title: Implement k-Clique Percolation for concept communities
status: todo
priority: high
tags:
- backend
- algorithm
- clique
dependencies: []
assignee: developer
created: 2025-12-05T21:39:44.337615303Z
estimate: ~
complexity: 4
area: backend
---

# Implement k-Clique Percolation for concept communities

## Context
GRAPHEME's core principle is "cliques = concepts". The k-Clique Percolation Method (CPM) finds overlapping communities by identifying adjacent k-cliques, directly supporting this concept detection.

**Research basis**: "The computational time of the SCP algorithm scales linearly with the number of k-cliques in the network." - ResearchGate

## Objectives
- Implement Sequential Clique Percolation (SCP) algorithm
- Find overlapping concept communities in graphs
- Feed results into `clique_mismatch` loss term
- Complexity: O(m · d^(k-2)) where d = max degree

## Tasks
- [ ] Implement k-clique adjacency detection (share k-1 nodes)
- [ ] Implement community merging via clique adjacency
- [ ] Create `find_concept_communities()` method
- [ ] Add Community struct with member nodes
- [ ] Integrate with GraphemeGraph in grapheme-core
- [ ] Update `clique_mismatch` in GraphEditDistance
- [ ] Add benchmarks and tests

## Acceptance Criteria
✅ **Algorithm Correctness:**
- Finds all k-cliques in graph
- Correctly identifies overlapping communities
- Two cliques are adjacent iff they share k-1 nodes

✅ **Performance:**
- Linear in number of cliques for sparse graphs
- Handles graphs with 1000+ nodes

✅ **Integration:**
- Works with GraphemeGraph (text) and MathGraph (expressions)
- Populates `clique_mismatch` field in GED

## Technical Notes

### Algorithm Pseudocode
```rust
pub struct Community {
    pub nodes: Vec<NodeIndex>,
    pub cliques: Vec<Vec<NodeIndex>>,
}

impl GraphemeGraph {
    /// Find concept communities using k-Clique Percolation
    pub fn find_concept_communities(&self, k: usize) -> Vec<Community> {
        // 1. Find all k-cliques
        let cliques = self.find_cliques(k);

        // 2. Build clique adjacency graph
        let adj = build_clique_adjacency(&cliques, k);

        // 3. Find connected components in clique graph
        let components = connected_components(&adj);

        // 4. Merge cliques into communities
        components.iter().map(|c| merge_cliques(c, &cliques)).collect()
    }
}
```

### Key Design Decisions
- Default k=3 for character-level graphs (trigrams)
- k=4-5 for concept-level compression
- Communities can overlap (node belongs to multiple concepts)

### Files to Modify
- `grapheme-core/src/lib.rs`: Add `find_concept_communities()`
- `grapheme-train/src/lib.rs`: Update `clique_mismatch` computation

## Testing
- [ ] Test on known graph with clear community structure
- [ ] Test overlapping communities
- [ ] Test k=3, k=4, k=5 configurations
- [ ] Benchmark on text graphs

## Updates
- 2025-12-05: Task created from algorithm research

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [Document code changes, new files, modified functions]

### Causality Impact
- Community detection enables concept-level compression
- Feeds into loss function via `clique_mismatch`

### Dependencies & Integration
- Depends on fixed-k enumeration (backend-009) for finding cliques
- Integrates with GraphemeGraph in grapheme-core
- Updates GraphEditDistance in grapheme-train

### Verification & Testing
- Test community detection on known graphs
- Verify overlapping communities work correctly
