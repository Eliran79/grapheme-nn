---
id: backend-047
title: Guard against empty slices in data generation choose calls
status: done
priority: high
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T10:42:03.256411154Z
estimate: ~
complexity: 3
area: backend
---

# Guard against empty slices in data generation choose calls

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
**HIGH: Dataset generation will crash when random selection from empty slices.**

Multiple locations in grapheme-train/src/lib.rs use `.choose(&mut rng).unwrap()` which panics on empty slices:

```rust
// Problematic patterns (lines 389, 412, 413, 438, 441):
terms.choose(&mut rng).unwrap()
operands.choose(&mut rng).unwrap()
```

When certain math expression types have no available terms, generation panics.

## Objectives
- Replace `.unwrap()` with graceful empty slice handling
- Return `Option` or use fallback values
- Add validation before random selection

## Tasks
- [ ] Add `if slice.is_empty()` guards before `.choose()` calls at lines 389, 412, 413, 438, 441
- [ ] Return `None` or use fallback for empty slices
- [ ] Propagate errors up to caller with proper Result types
- [ ] Add unit test for edge case with empty input slices

## Acceptance Criteria
✅ **No Panic on Empty:**
- Dataset generation handles empty slices gracefully
- Returns error or skips instead of panicking

✅ **Proper Propagation:**
- Callers are notified of generation failures

## Technical Notes
- File: grapheme-train/src/lib.rs lines 389, 412, 413, 438, 441
- Pattern: `.choose(&mut rng).unwrap()`
- Solution: Check `.is_empty()` first, or use `.choose().ok_or(Error)?`
- Affects: `generate_expression()` and related functions

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