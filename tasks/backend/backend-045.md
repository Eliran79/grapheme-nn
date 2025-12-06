---
id: backend-045
title: Fix NaN panic in predict_op float comparison
status: done
priority: high
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T10:41:54.478525863Z
estimate: ~
complexity: 3
area: backend
---

# Fix NaN panic in predict_op float comparison

> **⚠️ SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**
>
> When you mark this task as `done`, you MUST:
> 1. Fill the "Session Handoff" section at the bottom with complete implementation details
> 2. Document what was changed, what runtime behavior to expect, and what dependencies were affected
> 3. Create a clear handoff for the developer/next AI agent working on dependent tasks
>
> **If this task has dependents,** the next task will be handled in a NEW session and depends on your handoff for context.

## Context
**CRITICAL: This will cause training to crash with NaN values.**

The `predict_op` function at grapheme-core/src/lib.rs:3906 uses `.unwrap()` on `partial_cmp()` which panics on NaN values:

```rust
// Current problematic code (line 3906):
.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
```

During training, embedding weights can diverge to produce NaN values, causing immediate panic.

## Objectives
- Replace `.unwrap()` with safe NaN handling
- Ensure training continues gracefully with NaN inputs
- Add NaN detection logging for debugging

## Tasks
- [ ] Replace `partial_cmp().unwrap()` with `unwrap_or(Ordering::Equal)` or `total_cmp()`
- [ ] Add early NaN detection before comparison
- [ ] Log warning when NaN detected (helps debug training issues)
- [ ] Add unit test for NaN input handling

## Acceptance Criteria
✅ **No Panic on NaN:**
- `predict_op()` returns graceful result with NaN inputs
- Training loop continues without crashing

✅ **Debugging Support:**
- NaN occurrences are logged for debugging

## Technical Notes
- File: grapheme-core/src/lib.rs line 3906
- Pattern: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
- Solution: Use `f32::total_cmp()` or `unwrap_or(Ordering::Equal)`
- Consider: `is_nan()` check with warning log before comparison

## Testing
- [ ] Write unit tests for new functionality
- [ ] Write integration tests if applicable
- [ ] Ensure all tests pass before marking task complete
- [ ] Consider edge cases and error conditions

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [ ] **BEFORE committing**: Build, test, AND run the code to verify it works
  - Run `cargo build --release` (or `cargo build` for debug)
  - Run `cargo test` to ensure tests pass
  - **Actually run/execute the code** to verify runtime behavior
  - Fix all errors, warnings, and runtime issues
- [ ] Commit changes incrementally with clear messages
- [ ] Use descriptive commit messages that explain the "why"
- [ ] Consider creating a feature branch for complex changes
- [ ] Review changes before committing

**Testing requirements by change type:**
- Code changes: Build + test + **run the actual program/command** to verify behavior
- Bug fixes: Verify the bug is actually fixed by running the code, not just compiling
- New features: Test the feature works as intended by executing it
- Minor changes: At minimum build, check warnings, and run basic functionality

## Updates
- 2025-12-06: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [Document code changes, new files, modified functions]
- [What runtime behavior is new or different]

### Causality Impact
- [What causal chains were created or modified]
- [What events trigger what other events]
- [Any async flows or timing considerations]

### Dependencies & Integration
- [What dependencies were added/changed]
- [How this integrates with existing code]
- [What other tasks/areas are affected]

### Verification & Testing
- [How to verify this works]
- [What to test when building on this]
- [Any known edge cases or limitations]

### Context for Next Task
- [What the next developer/AI should know]
- [Important decisions made and why]
- [Gotchas or non-obvious behavior]