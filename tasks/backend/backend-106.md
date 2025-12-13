---
id: backend-106
title: Implement per-node activation functions (ReLU, Sigmoid, Tanh, Linear)
status: done
priority: high
tags:
- backend
dependencies:
- backend-092
assignee: developer
created: 2025-12-08T08:38:11.869071685Z
estimate: ~
complexity: 3
area: backend
---

# Implement per-node activation functions (ReLU, Sigmoid, Tanh, Linear)

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
For the neuromorphic forward pass (backend-107), each node needs its own activation function to enable heterogeneous network architectures. Different node types should use different activations (e.g., Linear for inputs, ReLU for hidden, Sigmoid for gates).

## Objectives
- Add per-node activation function support
- Support ReLU, Sigmoid, Tanh, Linear, and LeakyReLU
- Enable efficient forward pass with activation caching
- Support backpropagation with activation derivatives

## Tasks
- [x] Define ActivationFn enum with 5 variants
- [x] Add `activation_fn` field to Node struct
- [x] Add `pre_activation` field to cache input before activation
- [x] Implement `apply()` and `derivative()` methods
- [x] Add `derivative_from_input()` for convenience
- [x] Add vec operations for batch processing
- [x] Update all Node constructors with defaults
- [x] Add `set_pre_activation()` helper for forward pass
- [x] Add `activation_derivative()` helper for backward pass
- [x] Write comprehensive unit tests

## Acceptance Criteria
✅ **Activation Functions:**
- Linear: identity function, derivative = 1
- ReLU: max(0, x), derivative = 1 if x > 0 else 0
- Sigmoid: 1/(1+exp(-x)), uses cached output for derivative
- Tanh: tanh(x), uses cached output for derivative
- LeakyReLU: max(αx, x) where α=0.01

✅ **Node Integration:**
- All nodes have activation_fn and pre_activation fields
- Input/Output nodes default to Linear
- Hidden/Clique/Pattern nodes default to ReLU
- `set_pre_activation()` computes and caches activation

## Technical Notes
### API
```rust
pub enum ActivationFn {
    Linear, ReLU, Sigmoid, Tanh, LeakyReLU
}

impl ActivationFn {
    pub fn apply(&self, x: f32) -> f32;
    pub fn derivative(&self, x: f32, output: f32) -> f32;
    pub fn derivative_from_input(&self, x: f32) -> f32;
    pub fn apply_vec(&self, xs: &[f32]) -> Vec<f32>;
    pub fn derivative_vec(&self, xs: &[f32], outputs: &[f32]) -> Vec<f32>;
}

impl Node {
    pub fn set_pre_activation(&mut self, pre_act: f32);
    pub fn activation_derivative(&self) -> f32;
    pub fn hidden_with_activation(activation_fn: ActivationFn) -> Self;
    pub fn output_with_activation(activation_fn: ActivationFn) -> Self;
    pub fn with_activation_fn(mut self, activation_fn: ActivationFn) -> Self;
}
```

### Location
- File: `grapheme-core/src/lib.rs`
- ActivationFn enum: lines 103-193
- Node struct: lines 212-228
- Node impl: lines 230-348

## Testing
- [x] test_activation_fn_linear - Linear activation and derivative
- [x] test_activation_fn_relu - ReLU activation and derivative
- [x] test_activation_fn_sigmoid - Sigmoid with saturation
- [x] test_activation_fn_tanh - Tanh with saturation
- [x] test_activation_fn_leaky_relu - LeakyReLU with alpha=0.01
- [x] test_activation_fn_derivative_from_input - All functions match
- [x] test_activation_fn_vec_operations - Batch processing
- [x] test_node_with_activation_fn - Node forward/backward
- [x] test_node_activation_fn_defaults - Verify defaults per node type
- [x] test_node_with_activation_fn_builder - Builder pattern
- [x] test_activation_fn_gradient_numerical - Finite difference check

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
- 2025-12-08: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-core/src/lib.rs** (lines 103-348):
  - Added `ActivationFn` enum with 5 variants: Linear, ReLU, Sigmoid, Tanh, LeakyReLU
  - Added `pre_activation: f32` field to `Node` struct for caching input
  - Added `activation_fn: ActivationFn` field to `Node` struct
  - Implemented `apply()`, `derivative()`, `derivative_from_input()` for ActivationFn
  - Implemented `apply_vec()`, `derivative_vec()` for batch processing
  - Added `set_pre_activation()` and `activation_derivative()` to Node
  - Added `hidden_with_activation()`, `output_with_activation()`, `with_activation_fn()` builders
  - Updated all Node constructors (input, hidden, output, clique, pattern, compressed)
  - Updated one direct Node construction in DiffPool (line 3623)

### Causality Impact
- **Forward pass**: Use `node.set_pre_activation(weighted_sum)` to:
  1. Store pre-activation value
  2. Automatically compute post-activation via `activation_fn.apply()`
- **Backward pass**: Use `node.activation_derivative()` to get local gradient
- The neuromorphic forward pass (backend-107) can now iterate in topological order and call `set_pre_activation()` for each node

### Dependencies & Integration
- No new crate dependencies added
- ActivationFn implements: Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default
- Default activation: Linear (for enum default)
- Node defaults by type:
  - Input: Linear (pass-through)
  - Hidden: ReLU (sparse activations)
  - Output: Linear (regression tasks)
  - Clique/Pattern/Compressed: ReLU

### Verification & Testing
```bash
cargo test --package grapheme-core test_activation  # 11 tests
cargo test --package grapheme-core test_node        # 9 tests
cargo test                                          # Full suite passes
```

### Context for Next Task
**backend-107 (Neuromorphic Forward Pass)** can now:
1. Get topological order of nodes
2. For each node in order:
   - Sum weighted activations from predecessors: `sum += edge.weight * pred.activation`
   - Apply activation: `node.set_pre_activation(sum)`
3. For backprop, use `node.activation_derivative()` in reverse topological order

**Key decisions:**
- LeakyReLU added (α=0.01) to prevent dying ReLU problem
- `pre_activation` stored for efficient derivative computation (sigmoid/tanh)
- Input nodes stay at activation=1.0 (identity function)

**Gotchas:**
- ReLU derivative is 0 at exactly x=0 (could cause gradient issues at boundary)
- Sigmoid/Tanh saturate at large values (vanishing gradients)