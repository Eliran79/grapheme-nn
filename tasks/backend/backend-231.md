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
// GRAPHEME Protocol: Fixed LeakyReLU (α=0.01)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

fn activation(x: f32) -> f32 {
    if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
}

// ActivationType now uses fixed alpha (not configurable)
pub enum ActivationType {
    #[deprecated] ReLU,  // DEPRECATED
    LeakyReLU,           // Fixed α=0.01
    Tanh,
    Sigmoid,
}

impl Default for ActivationType {
    fn default() -> Self { Self::LeakyReLU }
}
```

## Session Handoff
### What Changed
- `grapheme-core/src/lib.rs:520-523` - Changed simple ReLU to LeakyReLU
- `grapheme-core/src/lib.rs:926-932` - Added dynamic √n + LeakyReLU
- `grapheme-train/src/backprop.rs` - Deprecated ReLU, fixed LeakyReLU α=0.01
- `CLAUDE.md` - Protocol section with summary table
- Adam optimizer set as default (lr=0.001)

### Context for Next Task
- All neural components use fixed LeakyReLU (α=0.01)
- `ActivationType::default()` returns `LeakyReLU`
- ReLU is deprecated with compiler warning
