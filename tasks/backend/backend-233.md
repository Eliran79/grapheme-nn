---
id: backend-233
title: Audit and convert remaining ReLU usages to LeakyReLU
status: todo
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
- [ ] Audit grapheme-train/src/backprop.rs - has ReLU variant that may still be used
- [ ] Check all training binaries use LeakyReLU activation type
- [ ] Update task documentation (backend-106, backend-107, etc.)

## Files to Audit
- [ ] `grapheme-train/src/backprop.rs` - ActivationType::ReLU still exists (deprecate or remove)
- [ ] `grapheme-train/src/online_learner.rs`
- [ ] `grapheme-train/src/bin_disabled/*.rs` - Training binaries
- [x] `tasks/backend/backend-106.md` - Updated to reference LeakyReLU
- [x] `tasks/backend/backend-107.md` - Updated to reference LeakyReLU
- [x] `tasks/backend/backend-105.md` - Updated to reference Dynamic Xavier
- [x] `tasks/backend/backend-114.md` - Updated to reference LeakyReLU
- [x] `tasks/backend/backend-214.md` - Updated to reference Dynamic Xavier
- [x] `tasks/api/api-002.md` - Updated to reference LeakyReLU

## Tasks
- [ ] Search for `ActivationType::ReLU` usages and convert to LeakyReLU
- [ ] Update default activation in ActivationType enum docs
- [ ] Update task documentation to reflect LeakyReLU standard
- [ ] Consider deprecating ActivationType::ReLU variant

## Acceptance Criteria
- No active code uses `ActivationType::ReLU`
- All default activations are LeakyReLU(0.01)
- Documentation reflects LeakyReLU standard

## Technical Notes
The `ActivationType` enum in backprop.rs has both variants:
```rust
pub enum ActivationType {
    ReLU,            // DEPRECATED - don't use
    LeakyReLU(f32),  // Use this with alpha=0.01
}
```

Consider adding:
```rust
impl Default for ActivationType {
    fn default() -> Self {
        Self::LeakyReLU(0.01)
    }
}
```
