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
- [x] Add attention mechanism for edit localization

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

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
  - Run `cargo build --release` (or `cargo build` for debug)
  - Run `cargo test` to ensure tests pass
  - **Actually run/execute the code** to verify runtime behavior
  - Fix all errors, warnings, and runtime issues
- [x] Commit changes incrementally with clear messages
- [x] Use descriptive commit messages that explain the "why"
- [x] Consider creating a feature branch for complex changes
- [x] Review changes before committing

**Testing requirements by change type:**
- Code changes: Build + test + **run the actual program/command** to verify behavior
- Bug fixes: Verify the bug is actually fixed by running the code, not just compiling
- New features: Test the feature works as intended by executing it
- Minor changes: At minimum build, check warnings, and run basic functionality

## Updates
- 2025-12-06: Task created
- 2025-12-06: Task completed - GraphTransformNet with message passing and attention

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added `EditOp` enum: Keep, Delete, Modify, Insert operations for graph edits
- Added `MessagePassingLayer` struct: Linear transform + neighbor aggregation + ReLU
- Added `AttentionLayer` struct: Query/Key/Value projections with softmax attention
- Added `NodePredictionHead` struct: Predicts edit operation for each node with softmax
- Added `GraphPooling` struct with `PoolingType` enum: Mean, Max, Sum pooling
- Added `GraphTransformNet` struct: Full GNN architecture implementing `GraphTransformer` trait
- Added 21 new tests for all components
- File: grapheme-core/src/lib.rs (lines ~3300-3800)

### Key APIs
```rust
// Create network with vocab_size=256, hidden_dim=64, embed_dim=32, num_layers=2
let mut net = GraphTransformNet::new(256, 64, 32, 2);

// Predict edits: Vec<(NodeIndex, EditOp, Vec<f32>)>
let edits = net.predict_edits(&dag);

// Transform graph (implements GraphTransformer trait)
let result = net.transform(&dag)?;

// Get graph-level embedding
let embedding = net.get_graph_embedding(&dag);

// Zero gradients for training
net.zero_grad();
```

### Causality Impact
- `encode()` embeds nodes → message passing → attention-weighted pooling
- `predict_edits()` uses node embeddings + global context to predict operations
- `apply_edits()` creates new graph based on predicted edits
- Attention weights are normalized via softmax (sum to 1.0)

### Dependencies & Integration
- Uses Embedding from backend-026 for character embeddings
- Uses BackwardPass trait from backend-027 for gradient computation
- Works with TrainingLoop from backend-028 for training
- Implements existing `GraphTransformer` trait from grapheme-core
- No new external dependencies

### Verification & Testing
- 21 new tests: test_message_passing_*, test_attention_*, test_graph_pooling_*, test_node_prediction_*, test_graph_transform_net_*
- Run: `cargo test -p grapheme-core`
- 119 tests in grapheme-core, 382 total across workspace

### Context for Next Task
- For backend-030 (end-to-end pipeline):
  - Use GraphTransformNet as the core transformation network
  - Input: DagNN from natural language via grapheme-core
  - Output: DagNN representing math expression
  - Train with TrainingLoop using GED loss
- Message passing is single-direction (aggregates from incoming edges)
- Default initialization uses Xavier for weights, zeros for biases
- Attention layer projects to head_dim=64 internally