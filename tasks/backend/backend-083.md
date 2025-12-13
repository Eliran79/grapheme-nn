---
id: backend-083
title: Implement actual math validation in grapheme-math validate()
status: done
priority: low
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T14:54:00.216518786Z
estimate: ~
complexity: 3
area: backend
---

# Implement actual math validation in grapheme-math validate()

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
The `MathBrain::validate()` method in grapheme-math has a comment: "(placeholder - actual validation would check operator arity, type consistency, etc.)" indicating incomplete implementation.

## Objectives
- Implement proper mathematical graph validation
- Detect common issues: arity mismatches, type errors, undefined operations
- Return meaningful ValidationIssue diagnostics

## Tasks
- [ ] Validate operator arity (binary ops need 2 inputs, unary need 1)
- [ ] Validate function arity (sin/cos/tan need 1 arg, etc.)
- [ ] Check type consistency (numeric ops on numeric values)
- [ ] Detect undefined symbols
- [ ] Check for division by zero possibilities
- [ ] Validate derivative/integral variable references
- [ ] Add tests for each validation rule

## Acceptance Criteria
✅ **Detects Issues:**
- Invalid graphs return non-empty ValidationIssue list

✅ **Clear Diagnostics:**
- Each issue has meaningful message and severity

## Technical Notes
### Current state (placeholder):
```rust
fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
    let mut issues = Vec::new();
    // (placeholder - actual validation would check operator arity, type consistency, etc.)
    Ok(issues)
}
```

### Validations to implement:
1. **Arity check**: BinOp nodes should have exactly 2 children, Function nodes should have expected arg count
2. **Type consistency**: Operations that expect numbers shouldn't have symbol inputs (unless symbolic mode)
3. **Division by zero**: Warn if divisor could be zero
4. **Undefined variables**: Variables should be defined in context or be explicit unknowns
5. **Derivative variables**: The variable being differentiated should appear in expression

### File to modify:
- `grapheme-math/src/lib.rs` - MathBrain::validate() method around line 722

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