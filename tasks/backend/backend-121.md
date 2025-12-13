---
id: backend-121
title: Add activation gradients to CausalGraph (grapheme-reason)
status: done
priority: high
tags:
- backend
- gradient
- cognitive
- reasoning
dependencies: []
assignee: developer
created: 2025-12-09T10:08:38.490018427Z
estimate: 2h
complexity: 6
area: backend
---

# Add activation gradients to CausalGraph (grapheme-reason)

## Context

The `CausalGraph` brain in `grapheme-reason` uses a plain enum `CausalNode` as its node type. Without an `activation: f32` field, there's no learnable parameter for gradients to flow through. **This brain will NEVER learn** from structural loss until this is fixed.

Backend-104 gradient fix pattern - successfully applied to GraphemeGraph and MathGraph.

## Objectives

1. Refactor `CausalNode` enum to struct-with-type pattern
2. Add activation field for gradient flow
3. Implement `compute_structural_loss_causal` in grapheme-train

## Tasks

- [ ] **Step 1: Refactor CausalNode in grapheme-reason/src/lib.rs**
  - Rename `CausalNode` enum to `CausalNodeType`
  - Create new `CausalNode` struct with `node_type` and `activation: f32`
  - Add `type_activation()` method

- [ ] **Step 2: Update all CausalNode usages**
  - Update `CausalGraph::add_node()` calls
  - Update pattern matches to use `node.node_type`

- [ ] **Step 3: Add structural loss in grapheme-train/src/lib.rs**
  - `compute_structural_loss_causal()`
  - `compute_soft_node_costs_causal()`
  - `compute_activation_gradients_causal()`

## Acceptance Criteria

✅ `CausalNode` has `activation: f32` field
✅ `compute_structural_loss_causal` returns non-zero `activation_gradients`
✅ All existing tests pass

## Technical Notes

**Suggested activation values (based on causal importance):**
- `Cause` → 0.8 (primary causal factor)
- `Effect` → 0.7 (outcome)
- `Condition` → 0.5 (enabling factor)
- `Intervention` → 0.9 (action that changes outcome)
- `Observation` → 0.3 (evidence)
- `Counterfactual` → 0.6 (hypothetical)
- `Mediator` → 0.5 (intermediate variable)
- `Confounder` → 0.7 (important for correct inference)

**Reference:** `grapheme-math/src/lib.rs` MathNode pattern

## Testing

- [ ] `cargo build` passes
- [ ] `cargo test` passes
- [ ] Activation gradients are non-zero

## Version Control

- [ ] Commit: `feat(reason): Add activation gradients to CausalGraph (backend-121)`