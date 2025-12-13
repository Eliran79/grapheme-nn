---
id: backend-107
title: Implement neuromorphic forward pass (topological order + activation propagation)
status: done
priority: high
tags:
- backend
dependencies:
- backend-105
- backend-106
assignee: developer
created: 2025-12-08T08:38:17.124241006Z
estimate: ~
complexity: 3
area: backend
---

# Implement neuromorphic forward pass (topological order + activation propagation)

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
The neuromorphic forward pass implements biologically-inspired activation propagation through the DagNN. This is the core computation for the neural network - processing nodes in topological order and applying per-node activation functions with learnable edge weights.

## Objectives
- Implement topological order traversal for forward pass
- Use per-node activation functions (from backend-106)
- Apply learnable edge weights (from backend-105)
- Cache pre-activation values for backpropagation

## Tasks
- [x] Explore existing DagNN forward pass implementation
- [x] Verify topological sort exists (TopologicalOrder::from_graph)
- [x] Update forward() to delegate to neuromorphic_forward()
- [x] Implement neuromorphic_forward() method
- [x] Add forward_with_inputs() for custom input activations
- [x] Add get_pre_activations() helper
- [x] Add get_activation_derivatives() helper
- [x] Add get_output_activations() helper
- [x] Write comprehensive tests (11 tests)

## Acceptance Criteria
✅ **Topological Order:**
- Nodes processed in dependency order
- Input nodes processed first
- Each node sees all predecessor activations

✅ **Activation Functions:**
- Per-node activation function applied via set_pre_activation()
- Pre-activation values cached for backprop
- Different activations supported (LeakyReLU preferred per GRAPHEME protocol, Sigmoid, Tanh, Linear)

✅ **Edge Weights:**
- Weighted sum of predecessor activations
- `pre_activation = Σ(edge_weight * source_activation)`

## Technical Notes
### Algorithm
```rust
for node in topological_order:
    if is_input(node):
        pre_activation = activation  // Keep input value
    else:
        pre_activation = Σ(edge_weight[u,v] * activation[u])
    node.set_pre_activation(pre_activation)  // Applies activation_fn
```

### New Methods (lines 1253-1374)
- `neuromorphic_forward()` - Main forward pass O(V+E)
- `forward_with_inputs(&HashMap<NodeId, f32>)` - Custom inputs
- `get_pre_activations()` - For backprop
- `get_activation_derivatives()` - For backprop
- `get_output_activations(n)` - Last n nodes

## Testing
- [x] test_neuromorphic_forward_basic - Input nodes preserved
- [x] test_neuromorphic_forward_with_hidden_nodes - Weighted sum
- [x] test_neuromorphic_forward_relu_clips_negative - ReLU behavior
- [x] test_neuromorphic_forward_sigmoid_activation - Sigmoid
- [x] test_neuromorphic_forward_tanh_activation - Tanh
- [x] test_neuromorphic_forward_preserves_pre_activation - Caching
- [x] test_neuromorphic_forward_chain - Multi-layer
- [x] test_forward_with_inputs - Custom inputs
- [x] test_get_activation_derivatives - For backprop
- [x] test_get_pre_activations - Pre-activation cache
- [x] test_neuromorphic_forward_edge_weight_scaling - Dynamic Xavier init

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
- **grapheme-core/src/lib.rs** (lines 1218-1374):
  - Updated `ForwardPass::forward()` to delegate to `neuromorphic_forward()`
  - Added `neuromorphic_forward()` - main forward pass using topological order
  - Added `forward_with_inputs()` - for custom input activations
  - Added `get_pre_activations()` - returns cached pre-activation values
  - Added `get_activation_derivatives()` - for backpropagation
  - Added `get_output_activations(n)` - last n node activations
  - Added 11 comprehensive tests for neuromorphic forward pass

### Causality Impact
- **Forward Pass Flow:**
  1. `update_topology()` computes topological order (if needed)
  2. For each node in order: sum weighted inputs → apply activation
  3. Both `pre_activation` and `activation` are cached on each node
- **Backward Pass Dependency:**
  - `pre_activation` cached for computing `d_activation/d_pre_activation`
  - Use `node.activation_derivative()` to get local gradient
  - Use `get_activation_derivatives()` to get all derivatives at once

### Dependencies & Integration
- **Builds on:**
  - backend-105 (learnable edge weights) - `edge.weight` used in forward pass
  - backend-106 (activation functions) - `node.set_pre_activation()` applies activation
- **Enables:**
  - backend-108 (pruning) - can now measure activation flow
  - backend-111 (backward pass) - has cached pre_activation for gradients

### Verification & Testing
```bash
cargo test --package grapheme-core test_neuromorphic  # 8 tests
cargo test --package grapheme-core test_forward       # 2 tests
cargo test --package grapheme-core test_get_          # 3 tests
cargo test                                            # Full suite passes
```

### Context for Next Task
**backend-108 (Edge Weight Pruning)** can now:
1. Run forward pass to populate activations
2. Identify low-activation edges for pruning
3. Measure activation flow through the network

**backend-111 (Backward Pass)** can now:
1. Get pre_activation values via `get_pre_activations()`
2. Get activation derivatives via `get_activation_derivatives()`
3. Compute gradients: `d_loss/d_weight = d_loss/d_activation * d_activation/d_pre_act * source_activation`

**Key Decisions:**
- Input nodes keep their original activation (typically 1.0)
- Non-input nodes compute weighted sum of predecessors
- Pre-activation cached for efficient backprop
- Clone of topology.order used to avoid borrow conflict

**Gotchas:**
- Must call `update_topology()` before forward if graph changed
- Input nodes identified via `input_nodes_set` (O(1) lookup)
- Empty topology triggers automatic recomputation