---
id: backend-032
title: Add learnable memory retrieval and consolidation
status: done
priority: high
tags:
- backend
dependencies:
- backend-031
assignee: developer
created: 2025-12-06T09:49:32.482832523Z
estimate: ~
complexity: 3
area: backend
---

# Add learnable memory retrieval and consolidation

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
Add learnable components to the memory system that can be optimized via gradient descent,
replacing fixed heuristics with neural network-based retrieval and consolidation.

## Objectives
- Create learnable graph encoders for fixed-size embeddings
- Implement learnable similarity functions for retrieval
- Add learnable importance scoring for memory consolidation
- Provide gradient flow for all learnable components

## Tasks
- [x] Implement GraphEncoder with DynamicXavier initialization
- [x] Implement LearnableSimilarity with bilinear attention
- [x] Implement ImportanceScorer for consolidation decisions
- [x] Create LearnableEpisodicMemory with learned retrieval
- [x] Create LearnableSemanticGraph with learned queries
- [x] Create LearnableContinualLearning with experience replay
- [x] Add gradient methods (zero_grad, step, num_parameters)
- [x] Write comprehensive unit tests

## Acceptance Criteria
✅ **Learnable Components:**
- GraphEncoder produces L2-normalized embeddings
- LearnableSimilarity computes [0,1] bounded similarity scores
- ImportanceScorer outputs [0,1] importance scores

✅ **Gradient Flow:**
- All components have zero_grad and step methods
- num_parameters returns correct parameter counts

## Technical Notes
- Uses GRAPHEME Protocol: LeakyReLU (α=0.01), Adam (lr=0.001)
- GraphEncoder: 18 input features → embed_dim → hidden_dim → embed_dim (L2 normalized)
- LearnableSimilarity: bilinear attention (a^T W b) with temperature scaling
- ImportanceScorer: weighted combination of recency, access count, valence, embedding norm

## Testing
- [x] Write unit tests for new functionality (10 new tests)
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete (23 tests pass)
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
- Created new file: `grapheme-memory/src/learnable.rs`
- Updated `grapheme-memory/src/lib.rs` with module and re-exports
- Updated `grapheme-memory/Cargo.toml` with ndarray and rand dependencies
- Key structures:
  - `GraphEncoder`: MLP that encodes graphs to fixed-size L2-normalized embeddings
  - `LearnableSimilarity`: Bilinear attention for computing similarity scores
  - `ImportanceScorer`: Learned importance function for consolidation
  - `LearnableEpisodicMemory`: Episodic memory with learned retrieval
  - `LearnableSemanticGraph`: Semantic graph with learned queries
  - `LearnableContinualLearning`: Experience replay with learned importance

### Causality Impact
- Retrieval flow: graph → GraphEncoder → embedding → LearnableSimilarity → similarity scores
- Consolidation: importance score determines which memories to keep/prune
- Learning: zero_grad → forward → backward → step cycle for gradient updates
- All components use GRAPHEME Protocol (LeakyReLU α=0.01)

### Dependencies & Integration
- Added `ndarray.workspace = true` and `rand.workspace = true` to Cargo.toml
- Re-exports from lib.rs: GraphEncoder, LearnableSimilarity, ImportanceScorer, etc.
- Compatible with existing memory traits (EpisodicMemory, SemanticGraph, etc.)

### Verification & Testing
- Run: `cargo test -p grapheme-memory` - 23 tests pass
- Clippy: `cargo clippy -p grapheme-memory -- -D warnings` - 0 warnings
- 10 new tests in `learnable::tests` module

### Context for Next Task
- GraphFingerprint features: 2 counts + 8 node_types + 8 degree_hist = 18 features
- Embeddings are L2-normalized for cosine similarity via dot product
- ImportanceScorer uses sigmoid for [0,1] bounded output
- Experience buffer uses FIFO eviction when at capacity