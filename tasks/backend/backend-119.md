---
id: backend-119
title: Add activation gradients to MusicGraph (grapheme-music)
status: done
priority: high
tags:
- backend
- gradient
- cognitive
- music
dependencies: []
assignee: developer
created: 2025-12-09T10:07:36.098531132Z
estimate: 2h
complexity: 6
area: backend
---

# Add activation gradients to MusicGraph (grapheme-music)

## Context

The `MusicGraph` brain in `grapheme-music` uses a plain enum `MusicNode` as its node type. Without an `activation: f32` field, there's no learnable parameter for gradients to flow through. **This brain will NEVER learn** from structural loss until this is fixed.

Backend-104 gradient fix pattern - successfully applied to GraphemeGraph and MathGraph.

## Objectives

1. Refactor `MusicNode` enum to struct-with-type pattern
2. Add activation field for gradient flow
3. Implement `compute_structural_loss_music` in grapheme-train

## Tasks

- [ ] **Step 1: Refactor MusicNode in grapheme-music/src/lib.rs**
  - Rename `MusicNode` enum to `MusicNodeType`
  - Create new `MusicNode` struct with `node_type` and `activation: f32`
  - Add `type_activation()` method

- [ ] **Step 2: Update all MusicNode usages**
  - Update `MusicGraph::add_node()` calls
  - Update pattern matches to use `node.node_type`

- [ ] **Step 3: Add structural loss in grapheme-train/src/lib.rs**
  - `compute_structural_loss_music()`
  - `compute_soft_node_costs_music()`
  - `compute_activation_gradients_music()`

## Acceptance Criteria

✅ `MusicNode` has `activation: f32` field
✅ `compute_structural_loss_music` returns non-zero `activation_gradients`
✅ All existing tests pass

## Technical Notes

**Suggested activation values (based on musical importance):**
- `Note` → 0.5 (basic element)
- `Rest` → 0.2 (silence)
- `Chord` → 0.7 (harmonic structure)
- `Measure` → 0.4 (structural)
- `Key` → 0.8 (tonal center)
- `TimeSignature` → 0.6
- `Dynamic` → 0.5

**Reference:** `grapheme-math/src/lib.rs` MathNode pattern

## Testing

- [ ] `cargo build` passes
- [ ] `cargo test` passes
- [ ] Activation gradients are non-zero

## Version Control

- [ ] Commit: `feat(music): Add activation gradients to MusicGraph (backend-119)`