---
id: backend-081
title: Implement domain-specific from_core/to_core conversions
status: done
priority: medium
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T14:54:00.185350294Z
estimate: ~
complexity: 3
area: backend
---

# Implement domain-specific from_core/to_core conversions

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
Domain brains have `from_core()` and `to_core()` methods that should convert between core DagNN graphs and domain-specific representations. Currently these are stubs that just return `graph.clone()` without actual conversion.

## Objectives
- Implement proper conversion between core DagNN and domain-specific graph structures
- Enable domain brains to work with their specialized node types internally
- Support round-trip conversion: `from_core(to_core(domain_graph)) ≈ domain_graph`

## Tasks
- [ ] **grapheme-code**: Convert DagNN ↔ CodeGraph with AST node types
- [ ] **grapheme-law**: Convert DagNN ↔ LegalGraph with Citation, Holding, etc.
- [ ] **grapheme-music**: Convert DagNN ↔ MusicGraph with Note, Chord, Scale nodes
- [ ] **grapheme-chem**: Convert DagNN ↔ MolecularGraph with Atom, Bond nodes
- [ ] Define node type mapping conventions for each domain
- [ ] Add round-trip conversion tests

## Acceptance Criteria
✅ **Proper Conversion:**
- `from_core()` produces domain-specific node types where applicable
- `to_core()` converts domain nodes back to generic NodeType

✅ **Information Preserved:**
- Round-trip conversions preserve essential structure

## Technical Notes
### Current state (stubs):
```rust
fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
    Ok(graph.clone())  // Stub - no actual conversion
}
```

### Target state:
Each domain should define how NodeType maps to domain-specific types:
- `NodeType::Char` in code context → `CodeNode::Identifier` or `CodeNode::Literal`
- `NodeType::Char` in music context → Note name or rest
- Chemical formula parsing → `ChemNode::Atom` nodes with Element type

### Files to modify:
- `grapheme-code/src/lib.rs` (from_core, to_core)
- `grapheme-law/src/lib.rs` (from_core, to_core)
- `grapheme-music/src/lib.rs` (from_core, to_core)
- `grapheme-chem/src/lib.rs` (from_core, to_core)

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