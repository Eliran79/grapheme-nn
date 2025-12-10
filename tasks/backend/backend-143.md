---
id: backend-143
title: Implement Learnable trait for DagNN
status: todo
priority: high
tags:
- backend
- core
- training
- traits
dependencies:
- backend-142
assignee: developer
created: 2025-12-10T19:43:34.271508622Z
estimate: 3h
complexity: 6
area: backend
---

# Implement Learnable trait for DagNN

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
After backend-142 adds gradient accumulator and `zero_grad()`/`step()` methods to DagNN,
this task implements the `Learnable` trait so DagNN can be used with standard training
infrastructure alongside Embedding, MessagePassingLayer, and GraphTransformNet.

**Learnable trait (lines 2529-2547 in lib.rs):**
```rust
pub trait Learnable {
    fn zero_grad(&mut self);
    fn step(&mut self, lr: f32);
    fn num_parameters(&self) -> usize;
    fn gradient_norm(&self) -> f32;
    fn has_gradients(&self) -> bool;
    fn clip_gradients(&mut self, max_norm: f32);
}
```

This enables standard training loops:
```rust
for batch in data {
    model.zero_grad();
    let loss = forward_and_loss(&model, batch);
    model.backward(&loss_grad, &embedding);
    model.step(learning_rate);
}
```

## Objectives
- Implement `Learnable` trait for DagNN
- Enable DagNN to work with standard training infrastructure
- Add gradient clipping support

## Tasks
- [ ] Implement `Learnable` trait for DagNN struct
- [ ] Implement `num_parameters(&self) -> usize` (count trainable edge weights)
- [ ] Implement `clip_gradients(&mut self, max_norm: f32)` for gradient clipping
- [ ] Add tests verifying Learnable methods work correctly
- [ ] Verify DagNN can be used in generic `fn train<T: Learnable>(model: &mut T)` functions

## Acceptance Criteria
✅ **Criteria 1:**
- DagNN implements all 6 methods of Learnable trait

✅ **Criteria 2:**
- `num_parameters()` returns correct count of trainable edge weights

✅ **Criteria 3:**
- `clip_gradients(max_norm)` scales gradients when norm exceeds threshold

✅ **Criteria 4:**
- DagNN can be passed to generic functions expecting `T: Learnable`

## Technical Notes
- **Depends on backend-142**: This task uses the gradient storage and methods added there
- **num_parameters()**: Count edges in the graph (each edge has one trainable weight)
- **clip_gradients()**: Scale all gradients by `max_norm / gradient_norm()` if norm exceeds max

**Reference implementations in lib.rs:**
- Embedding: lines 5094-5130
- MessagePassingLayer: lines 2630-2666
- GraphTransformNet: lines 2668-2706

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
- 2025-12-10: Task created

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
