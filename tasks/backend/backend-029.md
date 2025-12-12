---
id: backend-029
title: Implement learnable graph transformation network
status: todo
priority: high
tags:
- backend
dependencies:
- backend-027
- backend-028
assignee: developer
created: 2025-12-06T08:41:19.959886720Z
estimate: ~
complexity: 3
area: backend
---

# Implement learnable graph transformation network

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
The "brain" that learns to transform input graphs to output graphs. Replaces hand-coded rules with learned transformations. This is the core innovation.

## Objectives
- Create neural network that predicts graph edits
- Learn node insertions, deletions, edge modifications
- Train on engine-generated (input, output) pairs

## Tasks
- [ ] Design `GraphTransformNet` architecture
- [ ] Implement message passing layers (GCN/GAT style)
- [ ] Add node-level prediction heads (insert/delete/modify)
- [ ] Add edge-level prediction heads
- [ ] Implement graph pooling for global features
- [ ] Connect to existing `GraphTransformer` trait
- [ ] Add attention mechanism for edit localization

## Acceptance Criteria
✅ **Learn Transformations:**
- Network predicts correct graph edits on training data
- Generalizes to unseen expressions (same level)

✅ **Integration:**
- Implements `GraphTransformer` trait
- Compatible with existing graph structures

## Technical Notes
- Start simple: 2-3 message passing layers
- Consider edge features (edge type, weight)
- Output: probability distribution over edit operations
- Use softmax for discrete choices, regression for continuous
- Reference: Graph2Graph, Neural Edit Operations

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