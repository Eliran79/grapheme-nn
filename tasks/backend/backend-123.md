---
id: backend-123
title: Create grapheme-brain-common crate with shared abstractions
status: done
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
- [x] Create `grapheme-brain-common/` crate directory structure
- [x] Add crate to workspace Cargo.toml
- [x] Define `ActivatedNode<T>` generic struct with activation field
- [x] Define `TypedGraph<N, E>` generic wrapper for petgraph
- [ ] Define `BaseDomainBrain` trait with default implementations (deferred to backend-126)
- [x] Implement `TextTransformRule` for text-based transformations
- [x] Implement `KeywordCapabilityDetector` for domain detection
- [x] Implement `TextNormalizer` for input normalization
- [x] Add comprehensive tests for all abstractions (37 tests)
- [x] Document public API with examples

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
- [x] Unit tests for ActivatedNode creation and activation access (7 tests)
- [x] Unit tests for TypedGraph add_node, add_edge, traversal (9 tests)
- [x] Unit tests for TextTransformRule application (7 tests)
- [x] Unit tests for KeywordCapabilityDetector matching (14 tests)
- [ ] Integration test: create minimal brain using all abstractions (deferred to backend-129+)

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
- 2025-12-09: Completed - created grapheme-brain-common crate with 37 passing tests

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **New Crate**: `grapheme-brain-common/` created with 4 modules:
  - `node.rs`: `ActivatedNode<T>` - generic node wrapper with activation field
  - `graph.rs`: `TypedGraph<N, E>` - generic graph wrapper for petgraph DiGraph
  - `transform.rs`: `TextTransformRule` and `TransformRuleSet` for text transformations
  - `utils.rs`: `KeywordCapabilityDetector`, `TextNormalizer`, and preset normalizers

- **Re-exports**: lib.rs re-exports all public types plus petgraph's `DiGraph`, `NodeIndex`, `EdgeIndex`

### Causality Impact
- No runtime effects yet - this is a new shared library crate
- Brain crates can optionally adopt these abstractions
- No existing behavior changed

### Dependencies & Integration
- **Dependencies added**: serde, petgraph, thiserror, grapheme-core
- **Workspace**: Added to Cargo.toml members list
- **Integration path**: Brain crates (backend-129 through backend-133) will migrate to use these types

### Verification & Testing
- Run `cargo test --package grapheme-brain-common` - 37 tests pass
- Run `cargo build --package grapheme-brain-common` - compiles cleanly
- Doc tests pass for `TextTransformRule`, `KeywordCapabilityDetector`, `TextNormalizer`
- 2 doc tests ignored (require `ignore` attribute due to type visibility)

### Context for Next Task
- **BaseDomainBrain trait**: Deferred to backend-126 - requires careful design to work with existing DomainBrain trait
- **ActivatedNode equality**: Intentionally excludes activation from equality/hash - two nodes with same type are equal regardless of activation level
- **TextNormalizer order**: Operations apply in order: replacements → collapse spaces → trim → lowercase. This means case-sensitive patterns should be defined in the original case.
- **Preset normalizers**: `math_normalizer()`, `code_normalizer()`, `legal_normalizer()` provide common presets