---
id: backend-234
title: Deprecate ReLU and Xavier - Enforce LeakyReLU + Dynamic Xavier
status: doing
priority: high
tags:
- backend
- activation
- protocol
- deprecation
dependencies:
- backend-231
- backend-232
assignee: developer
created: 2025-12-13T00:00:00Z
estimate: 1h
complexity: 2
area: backend
---

# Deprecate ReLU and Xavier - Enforce LeakyReLU + Dynamic Xavier

## Context
Per GRAPHEME protocol, LeakyReLU (α=0.01) and Dynamic Xavier initialization are mandatory.
Standard ReLU causes dead neurons; static Xavier doesn't adapt to topology changes.

## Objectives
- [x] Add `#[deprecated]` attributes to ReLU and static Xavier variants
- [x] Make `LeakyReLU(0.01)` the default activation
- [x] Make Dynamic Xavier the default initialization
- [x] Keep tests working (they test deprecated behavior intentionally)

## Tasks
- [x] Add `#[deprecated]` to `ActivationType::ReLU` in backprop.rs
- [x] Add `#[deprecated]` to `InitStrategy::Xavier` (prefer DynamicXavier)
- [x] Add `Default` impl for `ActivationType` returning `LeakyReLU(0.01)`
- [x] Add `InitStrategy::DynamicXavier` variant if not present
- [x] Update `InitStrategy::default()` to return `DynamicXavier`
- [x] Allow deprecated in test modules to prevent warnings
- [x] Run cargo check and clippy

## Acceptance Criteria
- `#[deprecated]` warnings when using ReLU or static Xavier
- Tests pass without warnings (use `#[allow(deprecated)]`)
- Default activation is `LeakyReLU(0.01)`
- Default initialization is `DynamicXavier`

## Technical Notes
```rust
// backprop.rs
#[deprecated(since = "0.1.0", note = "Use LeakyReLU(0.01) per GRAPHEME protocol")]
ReLU,

// grapheme-core or grapheme-vision
#[deprecated(since = "0.1.0", note = "Use DynamicXavier per GRAPHEME protocol")]
Xavier,

impl Default for ActivationType {
    fn default() -> Self {
        Self::LeakyReLU(0.01)
    }
}
```

## Version Control
- [x] Build passes with zero errors
- [x] Clippy passes with zero warnings
- [x] Tests pass
