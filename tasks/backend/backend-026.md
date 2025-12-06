---
id: backend-026
title: Implement node embedding layer with learnable weights
status: todo
priority: high
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T08:41:07.938711581Z
estimate: ~
complexity: 3
area: backend
---

# Implement node embedding layer with learnable weights

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
Currently DagNN nodes have fixed activation values. To enable learning, nodes need learnable embedding vectors that can be updated via gradient descent.

## Objectives
- Add learnable weight matrices to node embeddings
- Enable gradient computation for embeddings
- Integrate with existing Node/DagNN structures

## Tasks
- [ ] Create `Embedding` struct with weight matrix (d_model x vocab_size)
- [ ] Add `requires_grad` flag to tensors
- [ ] Implement forward pass: char → embedding vector
- [ ] Store gradients for backward pass
- [ ] Add embedding initialization (Xavier/He)
- [ ] Integrate with grapheme-core Node struct

## Acceptance Criteria
✅ **Learnable Embeddings:**
- Embedding weights can be initialized and stored
- Forward pass produces embedding vectors from characters

✅ **Gradient Ready:**
- Gradients can be accumulated on embeddings
- Weights can be updated after backward pass

## Technical Notes
- Consider using `ndarray` crate for matrix operations
- Embedding dimension: start with d=64 or d=128
- Character vocabulary: all Unicode codepoints (use sparse lookup)
- Store embeddings in grapheme-core or new grapheme-nn crate

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
