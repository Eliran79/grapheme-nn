---
id: backend-123
title: Create grapheme-brain-common crate with shared abstractions
status: todo
priority: high
tags:
- backend
- refactoring
- deduplication
dependencies:
- backend-092
assignee: developer
created: 2025-12-09T11:43:59.146038185Z
estimate: 4h
complexity: 8
area: backend
---

# Create grapheme-brain-common crate with shared abstractions

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
Code analysis revealed ~3,470 lines of duplicated code across 5 cognitive brain crates (grapheme-math, grapheme-code, grapheme-law, grapheme-music, grapheme-chem) with 70-95% similarity. This task creates a shared `grapheme-brain-common` crate to eliminate duplication and establish reusable abstractions.

### Duplication Summary
| Category | Duplicated Lines | Similarity |
|----------|-----------------|------------|
| Node activation wrappers | ~400 | 5 identical copies |
| Graph wrapper structures | ~600 | 5 partial copies |
| DomainBrain trait impl | ~800 | 70% identical |
| Transformation rules | ~700 | 80%+ similar |
| Normalization/detection | ~400 | 90%+ identical |

## Objectives
- Create `grapheme-brain-common` crate in workspace
- Define reusable generic types and traits
- Reduce maintenance burden for future brain implementations
- Target ~1,850 lines of code savings (54% reduction)

## Tasks
- [ ] Create `grapheme-brain-common/` crate directory structure
- [ ] Add crate to workspace Cargo.toml
- [ ] Define `ActivatedNode<T>` generic struct with activation field
- [ ] Define `TypedGraph<N, E>` generic wrapper for petgraph
- [ ] Define `BaseDomainBrain` trait with default implementations
- [ ] Implement `TextTransformRule` for text-based transformations
- [ ] Implement `KeywordCapabilityDetector` for domain detection
- [ ] Implement `TextNormalizer` for input normalization
- [ ] Add comprehensive tests for all abstractions
- [ ] Document public API with examples

## Acceptance Criteria
✅ **Crate Structure:**
- `grapheme-brain-common` compiles independently
- All public types have documentation
- No circular dependencies introduced

✅ **Generic Types:**
- `ActivatedNode<T>` works with any node type enum
- `TypedGraph<N, E>` provides common graph operations
- Types are `Send + Sync` compatible

✅ **Trait Defaults:**
- `BaseDomainBrain` provides defaults for: version(), parse(), execute()
- Brains only need to implement domain-specific methods
- Backward compatible with existing DomainBrain trait

## Technical Notes
### Architecture
```
grapheme-brain-common/
├── src/
│   ├── lib.rs           # Re-exports
│   ├── node.rs          # ActivatedNode<T>
│   ├── graph.rs         # TypedGraph<N, E>
│   ├── traits.rs        # BaseDomainBrain
│   ├── transform.rs     # TextTransformRule
│   └── utils.rs         # Normalizer, Detector
└── Cargo.toml
```

### Key Generic Definitions
```rust
pub struct ActivatedNode<T> {
    pub node_type: T,
    pub activation: f32,
}

pub struct TypedGraph<N, E = ()> {
    pub graph: DiGraph<N, E>,
    pub root: Option<NodeIndex>,
}

pub trait BaseDomainBrain: DomainBrain {
    fn version(&self) -> &str { "0.1.0" }
    fn parse(&self, input: &str) -> DomainResult<DagNN> { ... }
    // ... more defaults
}
```

### Dependencies
- petgraph (already in workspace)
- grapheme-core (for DagNN, DomainBrain trait)

## Testing
- [ ] Unit tests for ActivatedNode creation and activation access
- [ ] Unit tests for TypedGraph add_node, add_edge, traversal
- [ ] Unit tests for TextTransformRule application
- [ ] Unit tests for KeywordCapabilityDetector matching
- [ ] Integration test: create minimal brain using all abstractions

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
- 2025-12-09: Task created

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
