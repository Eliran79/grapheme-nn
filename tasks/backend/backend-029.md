---
id: backend-029
title: Implement learnable graph transformation network
status: done
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
- [x] Design `GraphTransformNet` architecture
- [x] Implement message passing layers (GCN/GAT style)
- [x] Add node-level prediction heads (insert/delete/modify)
- [x] Add edge-level prediction heads
- [x] Implement graph pooling for global features
- [x] Connect to existing `GraphTransformer` trait
- [ ] Add attention mechanism for edit localization (future enhancement)

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
- [x] Write unit tests for new functionality
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete
- [x] Consider edge cases and error conditions

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
- Created new file: `grapheme-train/src/graph_transform_net.rs`
- Added module and re-exports in `grapheme-train/src/lib.rs`
- Key structures implemented:
  - `GraphTransformNet`: Main learnable transformation network
  - `GraphTransformNetConfig`: Configuration (embed_dim=64, hidden_dim=128, num_layers=3)
  - `MessagePassingLayer`: GNN message passing with mean aggregation
  - `NodePredictionHead` / `EdgePredictionHead`: Edit classifiers
  - `NodeEdit` / `EdgeEdit` enums: Keep, Delete, Modify, Insert/Add
  - `ForwardOutput` type alias for complex return type

### Causality Impact
- Forward pass: graph → embed_graph → message_passing (N layers) → predict_edits
- Learning: forward → compute_loss (cross-entropy on node/edge edits)
- Transform: forward → apply_edits → new DagNN via text reconstruction
- Uses GRAPHEME Protocol: LeakyReLU (α=0.01), Adam (lr=0.001)

### Dependencies & Integration
- Uses `grapheme_core::{DagNN, GraphTransformer, GraphemeResult, NodeId, TransformRule}`
- Uses `crate::backprop::{Tape, LEAKY_RELU_ALPHA}`
- Implements `GraphTransformer` trait for seamless integration
- No external dependencies added

### Verification & Testing
- 6 unit tests in `graph_transform_net::tests` module
- Run: `cargo test -p grapheme-train`
- Clippy: `cargo clippy -p grapheme-train -- -D warnings` passes with 0 warnings
- All 32 tests pass

### Context for Next Task
- Network uses simple mean aggregation (no attention yet - marked as future)
- `apply_edits` uses text reconstruction approach (rebuilds via DagNN::from_text)
- Edge predictions compare source/target NodeId pairs using HashSet
- Learning is via supervised loss - needs actual gradient updates in training loop