---
id: backend-046
title: Fix NaN handling in GED pair sorting
status: done
priority: high
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T10:41:58.204412217Z
estimate: ~
complexity: 3
area: backend
---

# Fix NaN handling in GED pair sorting

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
**CRITICAL: This will cause GED computation to crash with NaN values.**

The GED pair sorting at grapheme-train/src/lib.rs:1459 uses `partial_cmp()` with only partial fallback:

```rust
// Current problematic code (line 1459):
pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
```

This pattern can still misbehave with NaN - it doesn't panic but produces inconsistent ordering that breaks greedy assignment.

## Objectives
- Use fully robust float comparison for GED sorting
- Add NaN filtering before sorting
- Ensure deterministic behavior with edge cases

## Tasks
- [ ] Replace `partial_cmp` with `f32::total_cmp()` for total ordering
- [ ] Add pre-sort validation to filter/flag NaN pairs
- [ ] Add unit test with NaN in GED costs
- [ ] Document expected behavior with NaN inputs

## Acceptance Criteria
✅ **Robust Sorting:**
- GED sorting produces deterministic results regardless of NaN presence
- No panic possible from float comparison

✅ **Correct Assignment:**
- Greedy assignment works correctly even with degenerate costs

## Technical Notes
- File: grapheme-train/src/lib.rs line 1459
- Pattern: `sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(...))`
- Solution: Use `f32::total_cmp()` which treats NaN as greater than all values
- NaN values indicate broken embeddings - should be logged

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