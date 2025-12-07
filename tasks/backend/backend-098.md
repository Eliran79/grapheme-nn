---
id: backend-098
title: Implement Differentiable Clique Alignment Loss
status: todo
priority: high
tags:
- backend
- training
- vision-alignment
- cliques
dependencies:
- backend-096
assignee: developer
created: 2025-12-07T22:00:00Z
estimate: ~
complexity: 3
area: backend
---

# backend-098: Implement Differentiable Clique Alignment Loss

## Problem Statement

The GRAPHEME vision formula includes clique mismatch as a key component:

```rust
loss = α·node_insertion_cost + β·edge_deletion_cost + γ·clique_mismatch
```

Cliques represent learned concepts in GRAPHEME (per GRAPHEME_Vision.md:74-78):
> - Clique formation: Densely connected subgraphs = learned concepts
> - Mimics neural plasticity (Reimann et al., 2017: cliques in real brains)

Currently, clique alignment is not differentiable - we need a soft version.

## Background: Why Cliques Matter

From the vision document:
- Cliques are densely connected subgraphs representing **semantic concepts**
- Training should preserve/align cliques between predicted and target graphs
- Clique mismatch penalty (γ weight) is the **highest** in the default config (2.0)

## Solution: Differentiable Clique Alignment

### Approach 1: Soft Clique Membership

Instead of hard clique detection (NP-Hard for max clique), use:
1. **Clique scores**: Continuous measure of "cliqueness" for node subsets
2. **Soft membership**: Probability that a node belongs to a clique
3. **Alignment via transport**: Match cliques using Sinkhorn (from backend-096)

### Approach 2: Spectral Clique Approximation

1. Compute graph Laplacian eigenvalues
2. Use spectral clustering to identify quasi-cliques
3. Align via eigenvalue comparison (differentiable)

## Implementation Plan

### Phase 1: Soft Clique Detection
- [ ] Implement `compute_clique_scores()` - local density measure
- [ ] Implement `soft_clique_membership()` - node-to-clique probabilities
- [ ] Use existing clique detection as initialization

### Phase 2: Clique Alignment
- [ ] Implement `clique_cost_matrix()` - pairwise clique similarity
- [ ] Apply Sinkhorn for soft clique matching
- [ ] Compute alignment loss from transport plan

### Phase 3: Gradient Computation
- [ ] Backprop through Sinkhorn alignment
- [ ] Gradient w.r.t. node features affecting clique membership
- [ ] Gradient w.r.t. edge weights affecting clique structure

## API Design

```rust
/// Soft clique representation
pub struct SoftClique {
    /// Membership probabilities for each node
    pub membership: Vec<f32>,
    /// Clique "center" embedding (weighted mean of members)
    pub embedding: Vec<f32>,
    /// Cliqueness score (0 = sparse, 1 = complete subgraph)
    pub density: f32,
}

/// Detect soft cliques in a graph
pub fn detect_soft_cliques(
    graph: &Graph,
    temperature: f32,
    min_density: f32,
) -> Vec<SoftClique>;

/// Compute differentiable clique alignment cost
pub fn compute_clique_alignment(
    predicted_cliques: &[SoftClique],
    target_cliques: &[SoftClique],
    config: &CliqueAlignmentConfig,
) -> CliqueAlignmentResult;

pub struct CliqueAlignmentConfig {
    /// Sinkhorn iterations for clique matching
    pub sinkhorn_iters: usize,
    /// Temperature for soft assignment
    pub temperature: f32,
    /// Weight for size mismatch penalty
    pub size_weight: f32,
    /// Weight for density mismatch penalty
    pub density_weight: f32,
}

pub struct CliqueAlignmentResult {
    /// Total clique mismatch cost
    pub total_cost: f32,
    /// Number of unmatched predicted cliques
    pub unmatched_predicted: usize,
    /// Number of unmatched target cliques
    pub unmatched_target: usize,
    /// Soft assignment matrix (predicted × target)
    pub assignment: Vec<f32>,
    /// Gradients w.r.t. predicted clique embeddings
    pub gradients: Vec<Vec<f32>>,
}
```

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Soft clique detection | O(n·d²) | O(n·c) |
| Clique cost matrix | O(c₁·c₂·d) | O(c₁·c₂) |
| Sinkhorn matching | O(k·c₁·c₂) | O(c₁·c₂) |
| Gradient computation | O(c₁·c₂·d) | O(c₁·c₂·d) |

Where n = nodes, d = avg degree, c = clique count, k = Sinkhorn iterations

## Integration with backend-096

The clique alignment cost plugs into the structural loss:

```rust
pub fn compute_structural_loss(...) -> StructuralLossResult {
    let node_cost = compute_node_cost(...);
    let edge_cost = compute_edge_cost(...);

    // From this task (backend-098)
    let clique_cost = compute_clique_alignment(
        &detect_soft_cliques(predicted, ...),
        &detect_soft_cliques(target, ...),
        &config.clique_config,
    ).total_cost;

    let total = config.alpha * node_cost
              + config.beta * edge_cost
              + config.gamma * clique_cost;  // γ = 2.0 default

    // ...
}
```

## Dependencies
- backend-096 (Sinkhorn implementation)
- Existing clique detection in grapheme-core

## Success Criteria
- [ ] Soft cliques approximate hard cliques (IoU > 0.8)
- [ ] Gradients pass finite difference check
- [ ] Clique alignment improves during training
- [ ] Performance: <10ms for typical graph sizes

## Completion Checklist

> 1. All code compiles with no errors or warnings (cargo build, cargo clippy)
> 2. Document what was changed, what runtime behavior to expect, and what dependencies were affected

- [ ] Implementation complete
- [ ] Tests passing
- [ ] No clippy warnings
- [ ] Documentation updated

### What Changed
- [To be filled on completion]

### Runtime Behavior
- [To be filled on completion]

### Dependencies Affected
- [To be filled on completion]
