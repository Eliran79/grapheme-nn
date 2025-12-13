---
id: backend-038
title: Add learnable grounding with embodied interaction learning
status: done
priority: medium
tags:
- backend
dependencies:
- backend-031
- backend-034
assignee: developer
created: 2025-12-06T09:49:43.496969695Z
estimate: ~
complexity: 3
area: backend
---

# Add learnable grounding with embodied interaction learning

> **SESSION WORKFLOW NOTICE (for AI Agents):**
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
Add learnable components to the grounding module for embodied interaction learning.
This enables adaptive symbol-referent binding through perception encoding, action
encoding, grounding networks, and interaction prediction. Follows GRAPHEME Protocol
(LeakyReLU alpha=0.01, DynamicXavier, Adam lr=0.001).

## Objectives
- Create learnable perception encoders for fixed-size representations
- Implement learnable action encoders for action embeddings
- Add learnable grounding networks for symbol-referent binding strength
- Create learnable interaction predictors for perception-action sequences
- Enable experience-based learning from grounding and interaction outcomes

## Tasks
- [x] Implement PerceptionEncoder with modality-aware encoding weights
- [x] Implement ActionEncoder for graph-based action representations
- [x] Implement GroundingNetwork for symbol-referent binding strength computation
- [x] Implement InteractionPredictor for perception + action -> next perception
- [x] Create LearnableGrounding model combining all components
- [x] Add experience buffers for grounding and interaction learning
- [x] Write comprehensive unit tests (18 new tests)

## Acceptance Criteria
**Perception Encoding:**
- Encodes ModalGraphs to L2-normalized embeddings
- Modality-aware encoding with per-modality learned weights
- Combines graph features with modality embedding

**Action Encoding:**
- Encodes action graphs to fixed-size embeddings
- L2-normalized for cosine similarity

**Grounding Network:**
- Computes binding strength between symbol and perception embeddings
- Uses sigmoid for [0, 1] binding strength output
- Learns from grounding experience (correct/incorrect bindings)

**Interaction Prediction:**
- Predicts next perception embedding from current perception + action
- Learns from interaction sequences

## Technical Notes
- Uses GRAPHEME Protocol: LeakyReLU (alpha=0.01), lr=0.001, DynamicXavier
- PerceptionEncoder: 18 graph features + modality -> embed_dim
- ActionEncoder: 18 graph features -> embed_dim
- GroundingNetwork: symbol_embed + perception_embed -> binding_strength [0, 1]
- InteractionPredictor: perception_embed + action_embed -> next_perception_embed

## Testing
- [x] Write unit tests for new functionality (18 new tests)
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete (32 total in grapheme-ground)
- [x] Consider edge cases and error conditions

## Version Control

**CRITICAL: Always test AND run before committing!**

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
- [x] Commit changes incrementally with clear messages
- [x] Use descriptive commit messages that explain the "why"
- [x] Review changes before committing

## Updates
- 2025-12-06: Task created
- 2025-12-13: Task completed - Learnable grounding with embodied interaction added

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created new file: `grapheme-ground/src/learnable.rs` (~700 lines)
- Updated `grapheme-ground/src/lib.rs` with module declaration and re-exports
- Updated `grapheme-ground/Cargo.toml` with ndarray, rand, grapheme-memory deps
- Key structures:
  - `PerceptionEncoder`: Modality-aware perception encoding
  - `ActionEncoder`: Action graph encoding
  - `GroundingNetwork`: Symbol-referent binding strength computation
  - `InteractionPredictor`: Perception-action sequence prediction
  - `LearnableGrounding`: Complete model with experience buffers
  - `GroundingExperience`: Experience tuple for grounding learning
  - `InteractionExperience`: Experience tuple for interaction learning

### Causality Impact
- Perception flow: ModalGraph -> encoder -> embedding (L2 normalized)
- Action flow: Graph -> encoder -> embedding (L2 normalized)
- Grounding flow: (symbol_embed, perception_embed) -> network -> binding_strength [0, 1]
- Interaction flow: (perception_embed, action_embed) -> predictor -> next_perception_embed
- All components use GRAPHEME Protocol (LeakyReLU alpha=0.01)

### Dependencies & Integration
- Added `ndarray.workspace = true`, `rand.workspace = true`, `grapheme-memory`
- Re-exports from lib.rs: LearnableGrounding, PerceptionEncoder, ActionEncoder, etc.
- Integrates with existing ModalGraph, Modality, Graph, NodeId types
- Uses GraphFingerprint from grapheme-memory for 18-feature encoding

### Verification & Testing
- Run: `cargo test -p grapheme-ground` - 32 tests pass
- Clippy: `cargo clippy -p grapheme-ground -- -D warnings` - 0 warnings
- 18 new tests in `learnable::tests` module

### Context for Next Task
- Perception embeddings are L2-normalized for cosine similarity
- Grounding binding returns sigmoid [0, 1] strength
- Interaction prediction returns L2-normalized embedding
- Two separate experience buffers: grounding (symbol-perception pairs) and interaction (perception-action-next sequences)
- Experience buffers limited to config.buffer_size (default 1000)
- Learning uses TD-like error signals (predicted vs actual)
