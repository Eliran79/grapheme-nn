---
id: backend-233
title: Audit and convert remaining ReLU usages to LeakyReLU
status: done
priority: medium
tags:
- backend
- activation
- protocol
- audit
dependencies:
- backend-231
assignee: developer
created: 2025-12-13T00:00:00Z
estimate: 2h
complexity: 3
area: backend
---

# Audit and convert remaining ReLU usages to LeakyReLU

## Context
After initial LeakyReLU enforcement (backend-231), a full audit is needed to ensure
ALL neural components use LeakyReLU consistently.

## Objectives
- [x] Audit grapheme-train/src/backprop.rs - ReLU deprecated, LeakyReLU fixed α=0.01
- [x] Check all training binaries use LeakyReLU activation type
- [x] Update task documentation (backend-106, backend-107, etc.)

## Files Audited
- [x] `grapheme-train/src/backprop.rs` - ReLU deprecated, LeakyReLU now fixed α=0.01
- [x] `grapheme-vision/src/lib.rs` - Xavier deprecated, DynamicXavier added
- [x] `grapheme-train/src/optimizer.rs` - Adam::default() added
- [x] `tasks/backend/backend-106.md` - Updated to reference LeakyReLU
- [x] `tasks/backend/backend-107.md` - Updated to reference LeakyReLU
- [x] `tasks/backend/backend-105.md` - Updated to reference Dynamic Xavier
- [x] `tasks/backend/backend-114.md` - Updated to reference LeakyReLU
- [x] `tasks/backend/backend-214.md` - Updated to reference Dynamic Xavier
- [x] `tasks/api/api-002.md` - Updated to reference LeakyReLU
- [x] `CLAUDE.md` - Protocol section updated with full summary

## Tasks Completed
- [x] Deprecated `ActivationType::ReLU` with `#[deprecated]` attribute
- [x] Changed `LeakyReLU(f32)` to fixed `LeakyReLU` (α=0.01 via constant)
- [x] Added `Default` impl for `ActivationType` returning `LeakyReLU`
- [x] Added `LEAKY_RELU_ALPHA` constant = 0.01
- [x] Deprecated `InitStrategy::Xavier`, added `DynamicXavier`
- [x] Added `Default` impl for `InitStrategy` returning `DynamicXavier`
- [x] Added `Default` impl for `Adam` optimizer (lr=0.001)

## Acceptance Criteria ✅
- ReLU deprecated with compiler warning
- LeakyReLU uses fixed α=0.01 (LEAKY_RELU_ALPHA constant)
- Default activation is LeakyReLU
- Default init is DynamicXavier
- Default optimizer is Adam(0.001)
- All tests pass

## Technical Notes
The `ActivationType` enum in backprop.rs now:
```rust
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

pub enum ActivationType {
    #[deprecated(note = "Use LeakyReLU per GRAPHEME protocol")]
    ReLU,
    Tanh,
    Sigmoid,
    LeakyReLU,  // Fixed α=0.01
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::LeakyReLU
    }
}
```
