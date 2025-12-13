---
id: backend-118
title: Add activation gradients to LegalGraph (grapheme-law)
status: done
priority: high
tags:
- backend
- gradient
- cognitive
- law
dependencies: []
assignee: developer
created: 2025-12-09T10:07:16.377484358Z
estimate: 2h
complexity: 6
area: backend
---

# Add activation gradients to LegalGraph (grapheme-law)

## Context

The `LegalGraph` brain in `grapheme-law` uses a plain enum `LegalNode` as its node type. Without an `activation: f32` field, there's no learnable parameter for gradients to flow through. **This brain will NEVER learn** from structural loss until this is fixed.

Backend-104 gradient fix pattern - successfully applied to GraphemeGraph and MathGraph.

## Objectives

1. Refactor `LegalNode` enum to struct-with-type pattern
2. Add activation field for gradient flow
3. Implement `compute_structural_loss_legal` in grapheme-train

## Tasks

- [ ] **Step 1: Refactor LegalNode in grapheme-law/src/lib.rs**
  - Rename `LegalNode` enum to `LegalNodeType`
  - Create new `LegalNode` struct with `node_type` and `activation: f32`
  - Add `type_activation()` method with appropriate values per node type

- [ ] **Step 2: Update all LegalNode usages**
  - Update `LegalGraph::add_node()` calls
  - Update pattern matches to use `node.node_type`

- [ ] **Step 3: Add structural loss in grapheme-train/src/lib.rs**
  - `compute_structural_loss_legal()`
  - `compute_soft_node_costs_legal()`
  - `compute_activation_gradients_legal()`

## Acceptance Criteria

✅ `LegalNode` has `activation: f32` field
✅ `compute_structural_loss_legal` returns non-zero `activation_gradients`
✅ All existing tests pass

## Technical Notes

**Suggested activation values:**
- `Citation` → 0.8 (high importance)
- `Statute` → 0.7
- `Holding` → 0.9 (core legal principle)
- `Argument` → 0.6
- `Rule` → 0.7
- `Conclusion` → 0.9

**Reference:** `grapheme-math/src/lib.rs` MathNode pattern

## Testing

- [ ] `cargo build` passes
- [ ] `cargo test` passes
- [ ] Activation gradients are non-zero

## Version Control

- [ ] Commit: `feat(law): Add activation gradients to LegalGraph (backend-118)`