---
id: backend-099
title: Implement backward pass through structural loss to model parameters
status: todo
priority: high
tags:
- backend
dependencies:
- backend-096
- backend-097
- backend-098
assignee: developer
created: 2025-12-07T17:46:13.581569785Z
estimate: ~
complexity: 3
area: backend
---

# Implement backward pass through structural loss to model parameters

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

Backend-096, 097, 098 implemented the complete structural loss formula with gradients:
- Sinkhorn optimal transport for node alignment
- Edge cost from soft assignments
- DAG-specific clique metric (O(n))

The gradients are **computed** (`node_gradients`, `edge_gradients` in `StructuralLossResult`), but not yet **connected** to the model parameters. This is why the kindergarten training loop shows constant loss (19.89).

## Current State

**What Works:**
```rust
let loss_result = compute_structural_loss(&predicted_graph, &target_graph, &config);
// loss_result.node_gradients: Vec<f32> ✓ (computed)
// loss_result.edge_gradients: Vec<f32> ✓ (computed)
// loss_result.total_loss: f32 ✓ (computed)
```

**What's Missing:**
- Backprop from graph gradients → model layer gradients
- Update `GraphTransformNet` parameters using optimizer
- Gradient flow verification (finite difference check)

## Objectives

1. **Connect structural loss gradients to model parameters**
2. **Implement backward() for GraphTransformNet layers**
3. **Verify gradient flow with tests**
4. **Demonstrate loss decreasing in kindergarten training**

## Tasks

### Phase 1: Gradient Mapping
- [ ] Map `node_gradients` to embedding layer gradients
- [ ] Map `edge_gradients` to message passing layer gradients
- [ ] Implement `backward()` method for each layer

### Phase 2: Backpropagation
- [ ] Implement `GraphTransformNet::backward()`
- [ ] Chain gradients through: output → attention → message passing → embedding
- [ ] Handle batch gradient accumulation

### Phase 3: Optimizer Integration
- [ ] Connect gradients to `Adam::step()`
- [ ] Update all learnable parameters (weights, biases)
- [ ] Zero gradients after each update

### Phase 4: Verification
- [ ] Finite difference gradient check
- [ ] Test on simple synthetic examples
- [ ] Verify loss decreases in kindergarten training

## Acceptance Criteria

✅ **Gradient Flow:**
- `GraphTransformNet::backward()` computes gradients for all parameters
- Gradients pass finite difference check (relative error < 1e-4)
- No NaN or Inf in gradient tensors

✅ **Training Convergence:**
- Loss decreases monotonically in kindergarten training
- After 100 epochs, loss < 50% of initial loss
- Model parameters visibly change (L2 norm delta > 0)

✅ **Code Quality:**
- All tests pass (cargo test)
- No clippy warnings (cargo clippy)
- Backward pass integrated in train.rs

## Technical Notes

**Architecture Flow:**
```
Input Text → GraphemeGraph (fixed structure)
           ↓
GraphTransformNet (learnable):
  - Embedding: char → vector
  - MessagePassing: propagate features
  - Attention: focus on relevant nodes
  - Output: predicted graph features
           ↓
Structural Loss (backend-096, 097, 098)
           ↓
Gradients: node_gradients, edge_gradients
           ↓
Backward (THIS TASK):
  - Map graph grads → layer grads
  - Backprop through network
  - Accumulate parameter gradients
           ↓
Optimizer: Adam updates parameters
```

**Key Challenge:**
The model operates on **fixed graph topology** (from input text), but learns **node features**. Gradients from structural loss (which compares graph structures) need to flow back to the feature-learning layers.

**Implementation Strategy:**
1. Start with embedding layer (simplest)
2. Add message passing layers
3. Add attention layer
4. Test incrementally at each step

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
- 2025-12-07: Task created

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