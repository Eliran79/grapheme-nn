---
id: backend-095
title: Fix clippy warnings in training binaries
status: todo
priority: low
tags:
- backend
- code-quality
- clippy
dependencies: []
assignee: developer
created: 2025-12-07T20:00:00Z
estimate: ~
complexity: 1
area: backend
---

# Fix clippy warnings in training binaries

## Context

Running `cargo clippy --all` shows 2 warnings in `grapheme-train/src/bin/train.rs`:

```
warning: writing `&PathBuf` instead of `&Path` involves a new object where a slice will do
```

## Tasks

- [ ] Change `&PathBuf` parameters to `&Path` in `save_unified_checkpoint` and `load_unified_checkpoint`
- [ ] Verify all clippy warnings are resolved
- [ ] Run full test suite to ensure no regressions

## Location

`grapheme-train/src/bin/train.rs`:
- `save_unified_checkpoint(path: &PathBuf, ...)` → `save_unified_checkpoint(path: &Path, ...)`
- `load_unified_checkpoint(path: &PathBuf)` → `load_unified_checkpoint(path: &Path)`

## Acceptance Criteria

- [ ] `cargo clippy --all` produces no warnings
- [ ] All tests pass
