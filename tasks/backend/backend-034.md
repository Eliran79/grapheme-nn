---
id: backend-034
title: Add learnable world model with transition dynamics
status: done
priority: high
tags:
- backend
dependencies:
- backend-031
assignee: developer
created: 2025-12-06T09:49:32.507729803Z
estimate: ~
complexity: 3
area: backend
---

# Add learnable world model with transition dynamics

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
Add learnable components to the world model for predicting future states,
enabling the system to learn transition dynamics from observed experiences.

## Objectives
- Create learnable state encoders for fixed-size representations
- Implement learnable transition dynamics for predicting next states
- Add action encoding for modeling interventions
- Enable experience-based learning from observed transitions

## Tasks
- [x] Implement StateEncoder with entity/relation encoding
- [x] Implement ActionEncoder with softmax output
- [x] Create LearnableTransition for state prediction
- [x] Build LearnableWorldModel combining all components
- [x] Add observe_transition for experience collection
- [x] Implement imagine for multi-step prediction
- [x] Write comprehensive unit tests

## Acceptance Criteria
✅ **State Encoding:**
- Encodes entities and relations separately
- Produces L2-normalized fixed-size embeddings

✅ **Transition Dynamics:**
- Predicts next state from current state + action
- Supports multi-step imagination/planning

## Technical Notes
- Uses GRAPHEME Protocol: LeakyReLU (α=0.01), lr=0.001, DynamicXavier
- StateEncoder: 18 features each for entities/relations → combined → embed_dim
- LearnableTransition: (embed_dim + action_dim) → hidden → embed_dim
- ActionEncoder: softmax for action distribution

## Testing
- [x] Write unit tests for new functionality (13 new tests)
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete (26 total)
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
- Created new file: `grapheme-world/src/learnable.rs`
- Updated `grapheme-world/src/lib.rs` with module and re-exports
- Updated `grapheme-world/Cargo.toml` with ndarray, rand, grapheme-memory deps
- Key structures:
  - `StateEncoder`: Encodes WorldState (entities+relations) to embedding
  - `ActionEncoder`: Encodes action graphs to action distribution
  - `LearnableTransition`: Predicts next state embedding
  - `LearnableWorldModel`: Complete world model with learning

### Causality Impact
- Prediction flow: state + action → LearnableTransition → next_state_embed
- Learning: observe_transition → experience_buffer → compute_prediction_loss
- Imagination: initial_state + [actions] → [predicted_state_embeddings]
- All components use GRAPHEME Protocol (LeakyReLU α=0.01)

### Dependencies & Integration
- Added `ndarray.workspace = true`, `rand.workspace = true`, `grapheme-memory`
- Re-exports from lib.rs: LearnableWorldModel, StateEncoder, ActionEncoder, etc.
- Integrates with existing WorldState and Graph types

### Verification & Testing
- Run: `cargo test -p grapheme-world` - 26 tests pass
- Clippy: `cargo clippy -p grapheme-world -- -D warnings` - 0 warnings
- 13 new tests in `learnable::tests` module

### Context for Next Task
- State embeddings are L2-normalized for cosine similarity
- Action embeddings use softmax for probability distribution
- Experience buffer limited to 1000 examples (FIFO)
- imagine() returns list of state embeddings including initial state