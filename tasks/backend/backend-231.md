---
id: backend-231
title: Enforce LeakyReLU everywhere (replace all ReLU)
status: done
priority: high
tags:
- backend
- activation
- protocol
dependencies: []
assignee: developer
created: 2025-12-13T00:00:00Z
estimate: 1h
complexity: 2
area: backend
---

# Enforce LeakyReLU everywhere (replace all ReLU)

## Context
GRAPHEME protocol requires LeakyReLU (α=0.01) everywhere instead of standard ReLU.
Standard ReLU causes "dying neurons" in dynamic graph architectures.

## Objectives
- [x] Replace all ReLU activations with LeakyReLU
- [x] Update grapheme-core forward pass
- [x] Update documentation comments

## Tasks
- [x] Update `grapheme-core/src/lib.rs` - Forward pass activations
- [x] Update `grapheme-vision/src/lib.rs` - Already uses LeakyReLU
- [x] Update `grapheme-train/src/backprop.rs` - Has LeakyReLU variant

## Acceptance Criteria
✅ All forward pass activations use LeakyReLU
✅ Alpha = 0.01 consistently
✅ Clippy passes with 0 warnings

## Technical Notes
```rust
// LeakyReLU activation (GRAPHEME protocol)
let alpha = 0.01;
let activation = if x > 0.0 { x } else { alpha * x };
```

## Session Handoff
### What Changed
- `grapheme-core/src/lib.rs:520-523` - Changed simple ReLU to LeakyReLU
- `grapheme-core/src/lib.rs:926-932` - Added dynamic √n + LeakyReLU
- Documentation updated in CLAUDE.md

### Context for Next Task
- All new neural components should use LeakyReLU
- The `ActivationType::LeakyReLU(0.01)` variant exists in backprop.rs
