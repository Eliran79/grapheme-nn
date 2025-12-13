---
id: backend-239
title: Deprecate SGD optimizer, enforce Adam as default
status: done
priority: high
tags:
- backend
- optimizer
- protocol
dependencies:
- backend-234
assignee: developer
created: 2025-12-13T00:00:00Z
estimate: 1h
complexity: 2
area: backend
---

# Deprecate SGD optimizer, enforce Adam as default

## Context
GRAPHEME Protocol requires Adam optimizer (lr=0.001, beta1=0.9, beta2=0.999) as the default.
SGD is deprecated because Adam provides better convergence with LeakyReLU + DynamicXavier networks.

## Problem Locations
- `grapheme-train/src/optimizer.rs` - SGD struct needs deprecation
- Documentation needs to emphasize Adam as default

## Fix
Add `#[deprecated]` to SGD and document Adam as protocol default:
```rust
#[deprecated(
    since = "0.1.0",
    note = "Use Adam per GRAPHEME protocol - Adam provides better convergence with LeakyReLU + DynamicXavier"
)]
pub struct SGD { ... }

/// Adam optimizer (GRAPHEME Protocol default)
pub struct Adam { ... }

impl Default for Adam {
    fn default() -> Self {
        Self::new(0.001)  // lr=0.001, beta1=0.9, beta2=0.999
    }
}
```

## Protocol Summary
| Component | Deprecated | Use Instead |
|-----------|-----------|-------------|
| Activation | `ReLU` | `LeakyReLU` (α=0.01 fixed) |
| Init | `Xavier` | `DynamicXavier` |
| Optimizer | `SGD` | `Adam` (lr=0.001) |

## Acceptance Criteria
- [x] Add `#[deprecated]` to SGD struct
- [x] Add `#[allow(deprecated)]` to SGD impl blocks
- [x] Add `#[allow(deprecated)]` to SGD tests
- [x] Update Adam documentation as GRAPHEME Protocol default
- [x] Update module-level docs emphasizing Adam
- [x] Update CLAUDE.md with SGD deprecation note
- [x] Update PROTOCOL_MIGRATION.md with Step 3
- [x] Verify all tests pass

## Session Handoff

### What Changed
- **grapheme-train/src/optimizer.rs**:
  - Added `#[deprecated]` attribute to `SGD` struct
  - Added `#[allow(deprecated)]` to all SGD impl blocks and tests
  - Updated module-level docs to emphasize Adam as default
  - Enhanced Adam documentation with GRAPHEME Protocol notes

- **grapheme-train/src/semantic_decoder.rs**:
  - Updated comment mentioning SGD to suggest Adam migration

- **CLAUDE.md**:
  - Updated Adam section to state "SGD is DEPRECATED"
  - Updated code comment to emphasize deprecation

- **docs/PROTOCOL_MIGRATION.md**:
  - Updated Step 3 to show SGD deprecation pattern

### API Changes
```rust
// SGD is now deprecated
#[deprecated(note = "Use Adam per GRAPHEME protocol")]
pub struct SGD { ... }

// Adam is the default
impl Default for Adam {
    fn default() -> Self {
        Self::new(0.001)  // GRAPHEME Protocol settings
    }
}
```

### Usage
```rust
// DEPRECATED
#[allow(deprecated)]
let optimizer = SGD::new(0.1);

// CORRECT (GRAPHEME Protocol)
let optimizer = Adam::default();
```

### Testing
All 25 grapheme-train tests pass.
