---
id: testing-002
title: Review Dataset Generation Strategy (GRAPHEME_Math_Dataset)
status: done
priority: medium
tags:
- testing
dependencies:
- backend-001
- backend-002
- backend-003
- backend-004
assignee: developer
created: 2025-12-05T19:55:24.572929956Z
estimate: ~
complexity: 3
area: testing
---

# Review Dataset Generation Strategy (GRAPHEME_Math_Dataset)

## Context
Reviewed the GRAPHEME_Math_Dataset.md specification against the grapheme-train implementation and analyzed polynomial-time alternatives for NP-hard graph operations.

## Objectives
- [x] Review GRAPHEME_Math_Dataset.md specification
- [x] Verify grapheme-train implementation matches spec
- [x] Analyze polynomial-time alternatives for GED and clique detection
- [x] Document implementation gaps and recommendations

## Specification Compliance Review

### LevelSpec Implementation ✅
| Spec Field | Implementation | Status |
|------------|----------------|--------|
| `ops: Vec<MathOp>` | ✅ Implemented | Match |
| `functions: Vec<MathFn>` | ✅ Implemented | Match |
| `max_depth: usize` | ✅ Implemented | Match |
| `allow_symbols: bool` | ✅ Implemented | Match |
| `output: OutputType` | ✅ Numeric/Symbolic/Both | Match |
| `samples: usize` | ✅ Implemented | Match |

### Curriculum Levels
| Level | Spec | Implementation | Status |
|-------|------|----------------|--------|
| 1 | Basic arithmetic | ✅ BasicArithmetic | Complete |
| 2 | Nested ops | ✅ NestedOperations | Complete |
| 3 | Symbol substitution | ✅ SymbolSubstitution | Complete |
| 4 | Basic functions | ✅ BasicFunctions | Complete |
| 5 | Differentiation | ✅ Differentiation | Complete |
| 6 | Integration | ⚠️ Placeholder | Partial |
| 7 | Equation solving | ⚠️ Placeholder | Partial |

### Dataset Features
| Feature | Spec | Implementation | Status |
|---------|------|----------------|--------|
| JSONL save/load | ✅ Required | ✅ save_jsonl/load_jsonl | Complete |
| Train/Val/Test split | ✅ Required | ✅ split(train, val) | Complete |
| Batch iteration | ✅ Required | ✅ batches() | Complete |
| Level filtering | ✅ Required | ✅ filter_by_level() | Complete |
| Validation | ✅ Required | ✅ validate_dataset() | Complete |

## Graph Algorithm Analysis

### Current GED Implementation (grapheme-train/src/lib.rs:822)
```rust
// Current: O(1) - only compares counts, no alignment
node_insertion_cost: node_diff.max(0) as f32,
node_deletion_cost: (-node_diff).max(0) as f32,
clique_mismatch: 0.0,    // TODO: implement clique comparison
```

**Gap**: No actual graph edit distance computation or node alignment.

### Polynomial-Time Alternatives (Research)

#### For Graph Edit Distance:

| Algorithm | Complexity | Use Case |
|-----------|------------|----------|
| **Weisfeiler-Leman Kernel** | O(n·m·k) | GNN-native loss, graph similarity |
| **BP2 (Hausdorff + Greedy)** | O(n²) | Fast training loss, upper bound |
| **Hungarian Algorithm** | O(n³) | Optimal node assignment |
| **Spectral Methods** | O(n³) approx | Eigenvalue comparison |

**Recommendation**: Use Weisfeiler-Leman for loss computation - it's already the theoretical foundation for GNNs and provides polynomial-time graph similarity.

#### For Clique Detection:

| Algorithm | Complexity | Use Case |
|-----------|------------|----------|
| **Fixed-k Enumeration** | O(n^k · k²) | Small cliques k≤6 (tractable) |
| **k-Clique Percolation (CPM)** | O(m · d^(k-2)) | Overlapping communities |
| **Degeneracy-Based** | Fast for sparse | Text graphs (sparse) |
| **k-Clique Densest Subgraph** | Polynomial for fixed k | Concept density |

**Recommendation**: Use k-Clique Percolation for concept detection - linear in clique count, supports overlapping communities (matches "cliques = concepts" design).

### Key Insight
GRAPHEME graphs are **sparse** (characters don't fully connect) with **small cliques** (k=3-6 for concepts). This means:
- Worst-case exponential bounds rarely apply
- Fixed-k algorithms are practical
- Degeneracy-based methods excel

## Implementation Recommendations

### Priority 1: Enhance GED (grapheme-train)
```rust
impl GraphEditDistance {
    /// Weisfeiler-Leman based graph similarity
    pub fn compute_wl(g1: &MathGraph, g2: &MathGraph, iterations: usize) -> f32 {
        // 1-WL color refinement
        // Compare color histograms
        // O(n·m·iterations)
    }

    /// Fast quadratic upper bound
    pub fn compute_bp2(g1: &MathGraph, g2: &MathGraph) -> Self {
        // Hausdorff matching + greedy assignment
        // O(n²)
    }
}
```

### Priority 2: Clique Detection (grapheme-core)
```rust
impl GraphemeGraph {
    /// Find k-cliques using fixed-k enumeration
    pub fn find_cliques(&self, k: usize) -> Vec<Vec<NodeIndex>> {
        // O(n^k) for small k
    }

    /// k-Clique Percolation for concept communities
    pub fn find_concept_communities(&self, k: usize) -> Vec<Community> {
        // Linear in clique count
    }
}
```

### Priority 3: Complete Levels 6-7
- Integration: Symbolic integration rules
- Equation solving: Algebraic manipulation

## Acceptance Criteria
✅ **Specification Review**: LevelSpec, CurriculumLevel, Dataset match spec
✅ **Gap Analysis**: Identified GED and clique detection gaps
✅ **Alternatives**: Documented polynomial-time algorithms
✅ **Recommendations**: Prioritized implementation plan

## Technical Notes
- Weisfeiler-Leman is GNN-native (same theoretical foundation)
- BP2 provides O(n²) upper bound for fast training
- k-Clique Percolation supports overlapping concepts
- Sparse graphs make worst-case bounds irrelevant

## Testing
- [x] Reviewed spec compliance
- [x] Identified implementation gaps
- [x] Documented polynomial alternatives

## Updates
- 2025-12-05: Task created
- 2025-12-05: Completed spec review and algorithm analysis

## Session Handoff

### What Changed
- Reviewed GRAPHEME_Math_Dataset.md against implementation
- Analyzed current GED implementation (O(1) count-based)
- Documented polynomial-time alternatives from research
- Created prioritized recommendations

### Implementation Gaps Identified
1. **GED compute()**: Only compares node/edge counts, no alignment
2. **clique_mismatch**: Set to 0.0 with TODO comment
3. **Clique detection**: Data structure exists, no algorithm
4. **Levels 6-7**: Integration and Solve are placeholders

### Recommended Algorithms
| Operation | Algorithm | Complexity | File |
|-----------|-----------|------------|------|
| Loss function | Weisfeiler-Leman | O(n·m·k) | grapheme-train |
| Fast training | BP2 | O(n²) | grapheme-train |
| Concept detection | k-Clique Percolation | ~Linear | grapheme-core |
| Small cliques | Fixed-k enumeration | O(n^k) | grapheme-core |

### Dependencies & Integration
- WL kernel integrates with existing GNN architecture
- Clique detection feeds into `clique_mismatch` loss term
- No new crate dependencies needed (implement in pure Rust)

### Verification & Testing
```bash
cargo test                    # 106 tests pass
cargo bench --bench train_bench -- --test  # Benchmarks work
```

### Context for Next Task
- GED enhancement should be a new task (backend-level)
- Clique detection should be added to grapheme-core
- Consider adding WL kernel benchmark once implemented
- Levels 6-7 depend on symbolic integration (SymbolicEngine)