---
id: backend-120
title: Add activation gradients to MolecularGraph (grapheme-chem)
status: done
priority: high
tags:
- backend
- gradient
- cognitive
- chemistry
dependencies: []
assignee: developer
created: 2025-12-09T10:08:02.227172045Z
estimate: 2h
complexity: 6
area: backend
---

# Add activation gradients to MolecularGraph (grapheme-chem)

## Context

The `MolecularGraph` brain in `grapheme-chem` uses a plain enum `Atom` as its node type. Without an `activation: f32` field, there's no learnable parameter for gradients to flow through. **This brain will NEVER learn** from structural loss until this is fixed.

Backend-104 gradient fix pattern - successfully applied to GraphemeGraph and MathGraph.

## Objectives

1. Refactor `Atom` enum to struct-with-type pattern
2. Add activation field for gradient flow
3. Implement `compute_structural_loss_chem` in grapheme-train

## Tasks

- [ ] **Step 1: Refactor Atom in grapheme-chem/src/lib.rs**
  - Rename `Atom` enum to `AtomType` (or keep as Element enum)
  - Create new `Atom` struct with `element` and `activation: f32`
  - Add `type_activation()` method

- [ ] **Step 2: Update all Atom usages**
  - Update `MolecularGraph::add_atom()` calls
  - Update pattern matches

- [ ] **Step 3: Add structural loss in grapheme-train/src/lib.rs**
  - `compute_structural_loss_chem()`
  - `compute_soft_node_costs_chem()`
  - `compute_activation_gradients_chem()`

## Acceptance Criteria

✅ `Atom` has `activation: f32` field
✅ `compute_structural_loss_chem` returns non-zero `activation_gradients`
✅ All existing tests pass

## Technical Notes

**Suggested activation values (based on electronegativity/reactivity):**
- `Carbon` → 0.5 (backbone element)
- `Hydrogen` → 0.2 (common, low weight)
- `Oxygen` → 0.7 (reactive, electronegative)
- `Nitrogen` → 0.6
- `Sulfur` → 0.5
- `Phosphorus` → 0.6
- `Halogen` → 0.8 (highly reactive)
- `Metal` → 0.7

**Reference:** `grapheme-math/src/lib.rs` MathNode pattern

## Testing

- [ ] `cargo build` passes
- [ ] `cargo test` passes
- [ ] Activation gradients are non-zero

## Version Control

- [ ] Commit: `feat(chem): Add activation gradients to MolecularGraph (backend-120)`