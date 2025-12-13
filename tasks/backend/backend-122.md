---
id: backend-122
title: Add activation gradients to ModalGraph (grapheme-multimodal)
status: done
priority: high
tags:
- backend
- gradient
- cognitive
- multimodal
dependencies: []
assignee: developer
created: 2025-12-09T10:09:06.331375184Z
estimate: 2h
complexity: 6
area: backend
---

# Add activation gradients to ModalGraph (grapheme-multimodal)

## Context

The `ModalGraph` brain in `grapheme-multimodal` uses a plain enum `ModalNode` as its node type. Without an `activation: f32` field, there's no learnable parameter for gradients to flow through. **This brain will NEVER learn** from structural loss until this is fixed.

Backend-104 gradient fix pattern - successfully applied to GraphemeGraph and MathGraph.

## Objectives

1. Refactor `ModalNode` enum to struct-with-type pattern
2. Add activation field for gradient flow
3. Implement `compute_structural_loss_modal` in grapheme-train

## Tasks

- [ ] **Step 1: Refactor ModalNode in grapheme-multimodal/src/lib.rs**
  - Rename `ModalNode` enum to `ModalNodeType`
  - Create new `ModalNode` struct with `node_type` and `activation: f32`
  - Add `type_activation()` method

- [ ] **Step 2: Update all ModalNode usages**
  - Update `ModalGraph::add_node()` calls
  - Update pattern matches to use `node.node_type`

- [ ] **Step 3: Add structural loss in grapheme-train/src/lib.rs**
  - `compute_structural_loss_modal()`
  - `compute_soft_node_costs_modal()`
  - `compute_activation_gradients_modal()`

## Acceptance Criteria

✅ `ModalNode` has `activation: f32` field
✅ `compute_structural_loss_modal` returns non-zero `activation_gradients`
✅ All existing tests pass

## Technical Notes

**Suggested activation values (based on modality importance):**
- `Text` → 0.6 (common modality)
- `Image` → 0.7 (rich information)
- `Audio` → 0.6
- `Video` → 0.8 (most complex)
- `Embedding` → 0.5 (learned representation)
- `CrossModal` → 0.9 (fusion point - critical!)
- `Attention` → 0.7

**Reference:** `grapheme-math/src/lib.rs` MathNode pattern

## Testing

- [ ] `cargo build` passes
- [ ] `cargo test` passes
- [ ] Activation gradients are non-zero

## Version Control

- [ ] Commit: `feat(multimodal): Add activation gradients to ModalGraph (backend-122)`