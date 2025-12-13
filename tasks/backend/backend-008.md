---
id: backend-008
title: Implement k-Clique Percolation for concept communities
status: done
priority: high
tags:
- backend
- algorithm
- clique
dependencies:
- backend-009
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

### ⚠️ CRITICAL: k-Bounds Enforcement Required

**Inherited from backend-009**: CPM calls `find_cliques(k)` which must validate k.

**Additional Validation in CPM**:
```rust
pub fn find_concept_communities(&self, k: usize) -> Result<Vec<Community>, CliqueError> {
    // Validate k (also validated in find_cliques, but defense in depth)
    if k > MAX_CLIQUE_K {
        return Err(CliqueError::KTooLarge { requested: k, max: MAX_CLIQUE_K });
    }

    // 1. Find all k-cliques (will also validate k)
    let cliques = self.find_cliques(k)?;

    // ... rest of algorithm
}
```

**Must propagate errors** from `find_cliques()` - no silent failures!

### Key Design Decisions
- Default k=3 for character-level graphs (trigrams)
- k=4-5 for concept-level compression
- Communities can overlap (node belongs to multiple concepts)
- **k is bounded to 3-6** via shared constant from backend-014

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
- Added `Community` struct with id, nodes, cliques, and strength fields
- Added `find_concept_communities(k)` method to DagNN for CPM algorithm
- Added `find_triangle_communities()` convenience method (k=3)
- Added helper methods:
  - `build_clique_adjacency()` - builds clique-clique adjacency graph
  - `cliques_share_k_minus_1()` - checks k-1 node overlap
  - `find_clique_components()` - BFS for connected components
  - `merge_into_communities()` - merges cliques into communities
- Added 9 CPM tests (54 total tests in grapheme-core)

### Causality Impact
- Community detection enables concept-level compression
- Supports overlapping communities (node in multiple concepts)
- Default k=3 for character trigrams, k=4-5 for concept-level

### Dependencies & Integration
- Uses fixed-k enumeration from backend-009
- Validates k bounds (3 <= k <= MAX_CLIQUE_K)
- Integrates with DagNN in grapheme-core

### Verification & Testing
- Run `cargo test -p grapheme-core` for unit tests
- All 54 tests passing with 0 warnings
