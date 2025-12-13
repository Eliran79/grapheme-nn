---
id: testing-005
title: 'Integration test: End-to-end cognitive module training'
status: done
priority: high
tags:
- testing
dependencies:
- backend-038
assignee: developer
created: 2025-12-06T09:49:43.510002297Z
estimate: ~
complexity: 3
area: testing
---

# Integration test: End-to-end cognitive module training

> **SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**

## Context
Integration tests for the learnable cognitive modules (Agency, Multimodal, Grounding)
that verify end-to-end training capability and module interoperability.

## Objectives
- Test learnable module creation and initialization
- Verify forward pass produces valid outputs (embeddings, values, bindings)
- Test gradient flow (zero_grad, step) works correctly
- Verify training loop patterns work end-to-end
- Test integration between cognitive modules

## Tasks
- [x] Create integration test file for cognitive modules
- [x] Add tests for LearnableAgency (goal encoding, value estimation, drives)
- [x] Add tests for LearnableMultiModal (event encoding, fusion, attention)
- [x] Add tests for LearnableGrounding (perception encoding, binding, prediction)
- [x] Add end-to-end integration tests (cognitive loop, gradient flow)
- [x] Add training loop simulation tests
- [x] Fix clippy warnings

## Acceptance Criteria
**Agency Tests (9 tests):**
- Creation, goal encoding, value estimation, learning, priority
- Goal selection, adaptive drives, gradient flow, context

**Multimodal Tests (7 tests):**
- Creation, event encoding, fusion, binding, attention
- Learning, loss computation, gradient flow

**Grounding Tests (8 tests):**
- Creation, perception encoding, binding, prediction
- Interaction learning, loss, gradient flow

**Integration Tests (8 tests):**
- Cognitive loop: Agency → Multimodal → Grounding
- Multimodal + grounding integration
- Gradient flow across modules
- Training loop simulations for all modules
- Parameter count verification

## Technical Notes
- Location: `grapheme-tests/tests/integration_cognitive.rs`
- Dependencies added to `grapheme-tests/Cargo.toml`:
  - grapheme-agent, grapheme-multimodal, grapheme-ground, grapheme-world, ndarray
- All modules follow GRAPHEME Protocol:
  - LeakyReLU (α=0.01), DynamicXavier, Adam (lr=0.001)
  - L2-normalized embeddings
- 32 tests total covering all three cognitive modules

## Testing
- [x] Write unit tests for new functionality (32 tests)
- [x] Ensure all tests pass before marking task complete
- [x] Run clippy with 0 warnings

## Version Control
- [x] Build, test verified working

## Updates
- 2025-12-06: Task created
- 2025-12-13: Task completed - 32 integration tests for cognitive modules

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created `/home/user/grapheme-nn/grapheme-tests/tests/integration_cognitive.rs` (~550 lines)
- Updated `/home/user/grapheme-nn/grapheme-tests/Cargo.toml` with dependencies
- 32 comprehensive tests covering:
  - LearnableAgency (9 tests)
  - LearnableMultiModal (7 tests)
  - LearnableGrounding (8 tests)
  - End-to-end integration (8 tests)

### Causality Impact
- Tests verify embeddings are L2-normalized
- Tests verify binding strengths in [0, 1]
- Tests verify gradient flow (zero_grad, step) works
- Tests verify training loop patterns

### Dependencies & Integration
- Added to Cargo.toml: grapheme-agent, grapheme-multimodal, grapheme-ground, grapheme-world, ndarray
- Uses helper functions for creating test data (make_goal, make_modal_graph, etc.)
- Integrates with all learnable cognitive modules

### Verification & Testing
- Run: `cargo test -p grapheme-tests --test integration_cognitive` - 32 tests pass
- Clippy: `cargo clippy -p grapheme-tests --test integration_cognitive -- -D warnings` - 0 warnings

### Context for Next Task
- All cognitive modules can be created with default configs
- All embeddings are L2-normalized (sum of squares = 1.0)
- Training loops work: forward → record → zero_grad → compute_loss → step(lr)
- Modules can be used together for cognitive processing pipelines
