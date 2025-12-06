---
id: backend-048
title: Add graph structure validation before edge unwrap in grapheme-polish
status: todo
priority: high
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T10:42:07.014659136Z
estimate: ~
complexity: 3
area: backend
---

# Add graph structure validation before edge unwrap in grapheme-polish

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
**HIGH: Graph to expression conversion crashes on malformed graphs.**

The `node_to_expr()` function in grapheme-polish/src/lib.rs uses `.unwrap()` on edge lookups:

```rust
// Problematic patterns (lines 203, 213, 218):
graph.edges(node).next().unwrap()  // Panics if node has no edges
```

Malformed or incomplete graphs cause immediate panic during inference.

## Objectives
- Add graph structure validation before edge access
- Return Result type for graceful error handling
- Add pre-validation function for graph structure

## Tasks
- [ ] Replace `.next().unwrap()` with proper Option handling at lines 203, 213, 218
- [ ] Return `Result<Expr, PolishError>` from `node_to_expr()`
- [ ] Add `validate_graph_structure()` helper
- [ ] Add unit test with malformed graph input

## Acceptance Criteria
✅ **No Panic on Malformed Graph:**
- `node_to_expr()` returns error for invalid graphs
- Clear error message indicates which node is problematic

✅ **Validation Available:**
- Can pre-validate graphs before conversion

## Technical Notes
- File: grapheme-polish/src/lib.rs lines 203, 213, 218
- Pattern: `.edges(node).next().unwrap()`
- Solution: Match on `.next()` and return error, or validate edge count upfront
- Affects: All graph-to-expression conversions

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
