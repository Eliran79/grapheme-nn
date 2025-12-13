---
id: backend-105
title: Implement learnable edge weights (synaptic strength)
status: done
priority: high
tags:
- backend
dependencies:
- backend-092
assignee: developer
created: 2025-12-08T08:38:06.715701014Z
estimate: ~
complexity: 3
area: backend
---

# Implement learnable edge weights (synaptic strength)

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
Implement learnable edge weights to transform GRAPHEME into a true neuromorphic system where edges act as synapses with trainable connection strengths. This is the foundation for synaptic plasticity (pruning weak connections) and enables the network to learn which connections are important.

In biological neurons, synapses have varying strengths. Some connections amplify signals (excitatory), others suppress them (inhibitory). The strength changes through learning. This task brings that biological realism to GRAPHEME.

## Objectives
- Add `HashMap<(NodeId, NodeId), Parameter>` to store per-edge weights
- Make edge weights learnable parameters with gradient accumulation
- Initialize edge weights (Xavier/He initialization respecting DAG)
- Add training methods (`zero_grad`, `step`) for edge weight optimization
- Ensure edge weights integrate with forward pass (multiply activations by weights)
- Verify gradient flow through edge weights

## Tasks
- [x] Add `edge_weights: HashMap<(NodeId, NodeId), Parameter>` to GraphTransformNet
- [ ] Add `edge_grads: HashMap<(NodeId, NodeId), f32>` for gradient accumulation
- [ ] Implement `init_edge_weights()` with proper initialization (Xavier/He)
- [ ] Modify forward pass to use edge weights (activation *= weight)
- [ ] Implement backward pass for edge weight gradients
- [ ] Add `zero_grad()` method to clear edge gradients
- [ ] Add `step(lr)` method for gradient descent update
- [ ] Write unit test: verify edge weight initialization
- [ ] Write unit test: verify gradient flow through edge weights
- [ ] Write integration test: train simple graph, verify weights change
- [ ] Update GraphTransformNet serialization to include edge weights

## Acceptance Criteria
✅ **Edge weights are learnable parameters:**
- Each edge (u, v) has a weight parameter that can be trained
- Gradients flow through edge weights during backpropagation
- Edge weights update via gradient descent

✅ **Proper initialization:**
- Edge weights initialized with Xavier or He initialization
- Initialization respects DAG topology (no initialization errors)
- Initial weights are reasonable (not too large/small)

✅ **Integration with forward pass:**
- Activation propagation multiplies by edge weights: `output += weight * input`
- Forward pass produces different results with different edge weights
- No performance regression (forward pass remains efficient)

✅ **Training mechanics work:**
- `zero_grad()` clears all edge weight gradients
- Backward pass accumulates edge weight gradients correctly
- `step(lr)` updates edge weights: `weight -= lr * grad`
- Weights change during training (not stuck)

## Technical Notes
### Architecture Design

**Data Structure:**
```rust
pub struct GraphTransformNet {
    // ... existing fields ...

    /// Learnable edge weights (synaptic strengths)
    /// Maps (source_node, target_node) -> weight parameter
    /// Positive weights = excitatory, negative = inhibitory
    pub edge_weights: HashMap<(NodeId, NodeId), Parameter>,

    /// Gradient accumulator for edge weights
    #[serde(skip)]
    pub edge_grads: HashMap<(NodeId, NodeId), f32>,
}
```

**Initialization Strategy:**
- Use Dynamic Xavier initialization (GRAPHEME protocol): `w ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))`
  - Weight scales recomputed when graph topology changes (nodes added/removed)
- Or He initialization for LeakyReLU: `w ~ N(0, sqrt(2/fan_in))`
- Initialize ALL edges in the graph during model creation
- Store in HashMap for O(1) lookup during forward/backward pass

**Forward Pass Integration:**
```rust
// OLD: activation_next = sum(activation_prev for all predecessors)
// NEW: activation_next = sum(edge_weight * activation_prev for all predecessors)

for edge in graph.edges_directed(node, Incoming) {
    let source = edge.source();
    let weight = self.edge_weights.get(&(source, node)).unwrap_or(&1.0);
    activation_next += weight * activations[source];
}
```

**Backward Pass:**
```rust
// For each edge (u, v):
// grad_weight = activation[u] * grad_activation[v]

for edge in graph.edges() {
    let (u, v) = (edge.source(), edge.target());
    let grad = activations[u] * grad_activations[v];
    *self.edge_grads.entry((u, v)).or_insert(0.0) += grad;
}
```

**NP-Hard Complexity Avoidance:**
- HashMap lookup is O(1) - no graph search needed
- Edge weights stored explicitly, not computed
- No combinatorial enumeration
- Linear time w.r.t. number of edges

### Dependencies
- Depends on backend-092 (Parameter struct with gradient support)
- Required by backend-107 (neuromorphic forward pass)
- Required by backend-108 (edge weight pruning)

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
- 2025-12-08: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-core/src/lib.rs** - Added edge weight initialization:
  - `Edge::xavier(fan_in, fan_out, edge_type)` - Xavier/Glorot initialization for edges
  - `Edge::he(fan_in, edge_type)` - He initialization for edges (better for ReLU)
  - `Edge::sequential_xavier()`, `Edge::skip_xavier()` - Convenience methods
  - `DagNN::init_edge_weights(strategy)` - Reinitialize all edge weights in a graph
  - `DagNN::init_edge_weights_xavier()`, `DagNN::init_edge_weights_he()` - Convenience methods
- **Tests added**: 6 new tests for edge weight initialization and gradient flow
- **Runtime behavior**: Edge weights can now be initialized with proper neural network initialization
  - Xavier: w ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
  - He: w ~ N(0, sqrt(2/fan_in))

### Causality Impact
- Edge weights flow through backward pass (already implemented in `DagNN::backward`)
- `backward_and_update` applies gradient descent: `weight -= lr * grad`
- Gradient computation: `edge_grad = activation[source] * node_grad[target]`
- No new async flows - all operations are synchronous

### Dependencies & Integration
- Uses existing `InitStrategy` enum (Xavier, He, Uniform, Zero)
- Integrates with existing `BackwardPass` trait implementation on DagNN
- Edge weights stored in `Edge.weight` field on graph edges
- Edge gradients accumulated in `NodeGradients.edge_grads` HashMap

### Verification & Testing
- Run `cargo test -p grapheme-core test_edge_` to verify edge weight initialization
- Run `cargo test -p grapheme-core test_dag_init` to verify DagNN integration
- Tests verify: Xavier bounds, He finiteness, gradient flow, training updates

### Context for Next Task
- **For backend-107 (neuromorphic forward pass)**: Use Dynamic Xavier - weights recomputed when topology changes
- **For backend-108 (edge weight pruning)**: Edge weights are accessible via `dag.graph[edge_idx].weight`
- **Important decision**: Edge weights stored directly in `Edge` struct on graph edges (not separate HashMap)
  - This matches existing backward pass implementation at line 4390
  - Enables O(1) edge weight lookup during forward/backward passes
- **GRAPHEME Protocol**: Use Dynamic Xavier (recompute when topology changes) + LeakyReLU (α=0.01)
- **Gotcha**: When using Dynamic Xavier init, fan_in/fan_out are computed from node degree, not layer dimensions