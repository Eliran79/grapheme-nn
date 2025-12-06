---
id: backend-026
title: Implement node embedding layer with learnable weights
status: done
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
- 2025-12-06: Task completed - Embedding layer implemented

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added `Embedding` struct in `grapheme-core/src/lib.rs` (lines 2293-2493)
- Added `InitStrategy` enum: Xavier, He, Uniform, Zero
- Added `EmbeddingExt` trait for DagNN integration
- New types: `Embedding`, `InitStrategy`, `EmbeddingExt`
- Re-exports `ndarray::Array1` and `ndarray::Array2` via workspace

### Key APIs
```rust
// Create embedding layer (256 vocab, 64 dim)
let emb = Embedding::xavier(256, 64);

// Forward pass
let vec: Array1<f32> = emb.forward('a');
let batch: Array2<f32> = emb.forward_batch(&['h', 'i']);

// Backward pass
emb.backward(idx, &grad_output);
emb.step(learning_rate);

// With DagNN
let embeddings = dag.get_node_embeddings(&emb);
let matrix = dag.get_embedding_matrix(&emb);
```

### Dependencies & Integration
- Added `ndarray = "0.15"` and `rand = "0.8"` to workspace
- Added to grapheme-core/Cargo.toml
- Integrates with existing Node/DagNN via EmbeddingExt trait

### Verification & Testing
- 8 new tests: test_embedding_* in grapheme-core
- Run: `cargo test -p grapheme-core`
- 75 tests total, all passing

### Context for Next Task
- Gradients stored in `emb.grad: Option<Array2<f32>>`
- Call `zero_grad()` before each training step
- Call `backward(idx, grad)` for each used embedding
- Call `step(lr)` to apply SGD update
- For backend-027 (backprop): use this as leaf node in computation graph