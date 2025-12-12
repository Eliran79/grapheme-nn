---
id: backend-080
title: Implement actual transforms in domain brain plugins
status: todo
priority: medium
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T14:54:00.170100975Z
estimate: ~
complexity: 3
area: backend
---

# Implement actual transforms in domain brain plugins

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
All domain brain plugins (grapheme-code, grapheme-law, grapheme-music, grapheme-chem) have `transform()` methods that are stubs - they just return `graph.clone()` without actually applying any transformation rules. The brains define rules via `get_rules()` but those rules aren't actually implemented.

## Objectives
- Implement actual graph transformations for each domain brain's rules
- Enable domain brains to modify graphs according to domain-specific rules
- Support domain-specific reasoning and manipulation

## Tasks
- [ ] **grapheme-code**: Implement transforms for Dead Code Elimination, Constant Folding, Inline Expansion, Loop Unrolling, Type Inference
- [ ] **grapheme-law**: Implement transforms for Stare Decisis, Distinguish Precedent, IRAC Analysis, Citation Validation, Hierarchy of Authority
- [ ] **grapheme-music**: Implement transforms for Voice Leading, Chord Progression, Key Detection, Rhythm Quantization
- [ ] **grapheme-chem**: Implement transforms for Balance Equation, Valence Check, IUPAC Naming, Molecular Weight, Functional Group Detection
- [ ] **grapheme-math**: Review and enhance existing transforms (may already be implemented via MathBrain)
- [ ] Add unit tests for each transform

## Acceptance Criteria
✅ **Transforms Modify Graphs:**
- Each `transform(graph, rule_id)` returns a modified graph (not just clone) when applicable

✅ **Rule Logic Implemented:**
- Each rule in `get_rules()` has corresponding logic in `transform()`

## Technical Notes
### Current state (stubs):
```rust
fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
    match rule_id {
        0..=4 => Ok(graph.clone()),  // Stub!
        _ => Err(DomainError::InvalidInput(...))
    }
}
```

### Target state:
```rust
fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
    match rule_id {
        0 => self.apply_dead_code_elimination(graph),
        1 => self.apply_constant_folding(graph),
        // ...
    }
}
```

### Files to modify:
- `grapheme-code/src/lib.rs`
- `grapheme-law/src/lib.rs`
- `grapheme-music/src/lib.rs`
- `grapheme-chem/src/lib.rs`

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
