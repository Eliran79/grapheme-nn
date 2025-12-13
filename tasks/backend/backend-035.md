---
id: backend-035
title: Add learnable meta-cognition with calibrated uncertainty
status: done
priority: medium
tags:
- backend
dependencies:
- backend-031
assignee: developer
created: 2025-12-06T09:49:43.457196970Z
estimate: ~
complexity: 3
area: backend
---

# Add learnable meta-cognition with calibrated uncertainty

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
Add learnable components to the meta-cognition system for uncertainty estimation,
confidence calibration, and cognitive state monitoring.

## Objectives
- Create learnable uncertainty estimator for epistemic/aleatoric decomposition
- Implement confidence calibrator with temperature scaling
- Add introspection monitor for predicting overload/stuck states
- Enable gradient-based learning for all components

## Tasks
- [x] Implement LearnableUncertaintyEstimator with neural network
- [x] Implement ConfidenceCalibrator with temperature scaling
- [x] Create IntrospectionMonitor for state prediction
- [x] Build LearnableMetaCognition combining all components
- [x] Add calibration from prediction/outcome pairs
- [x] Write comprehensive unit tests

## Acceptance Criteria
✅ **Uncertainty Estimation:**
- Separates epistemic and aleatoric uncertainty
- Uses sigmoid for [0,1] bounded outputs

✅ **Confidence Calibration:**
- Temperature scaling adjusts based on ECE
- Tracks prediction history for calibration

## Technical Notes
- Uses GRAPHEME Protocol: LeakyReLU (α=0.01), lr=0.001, DynamicXavier
- LearnableUncertaintyEstimator: 18 features → hidden → (epistemic, aleatoric)
- ConfidenceCalibrator: temperature scaling with ECE-based adjustment
- IntrospectionMonitor: 7 state features → hidden → (overload_prob, stuck_prob)

## Testing
- [x] Write unit tests for new functionality (14 new tests)
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete (31 total)
- [x] Consider edge cases and error conditions

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
- [x] Commit changes incrementally with clear messages
- [x] Use descriptive commit messages that explain the "why"

## Updates
- 2025-12-06: Task created
- 2025-12-13: Task completed

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created new file: `grapheme-meta/src/learnable.rs`
- Updated `grapheme-meta/src/lib.rs` with module and re-exports
- Updated `grapheme-meta/Cargo.toml` with ndarray, rand, grapheme-memory deps
- Key structures:
  - `LearnableUncertaintyEstimator`: Neural network for epistemic/aleatoric
  - `ConfidenceCalibrator`: Temperature scaling with ECE tracking
  - `IntrospectionMonitor`: Predicts overload/stuck from cognitive state
  - `LearnableMetaCognition`: Complete meta-cognitive system

### Causality Impact
- Uncertainty: graph → features → hidden → (epistemic, aleatoric)
- Calibration: raw_confidence → temperature_scale → calibrated
- Introspection: cognitive_state → features → (overload_prob, stuck_prob)
- Calibration updates temperature based on prediction/outcome history

### Dependencies & Integration
- Added `ndarray.workspace = true`, `rand.workspace = true`, `grapheme-memory`
- Re-exports from lib.rs: LearnableMetaCognition, ConfidenceCalibrator, etc.
- Integrates with existing UncertaintyEstimate and CognitiveState types

### Verification & Testing
- Run: `cargo test -p grapheme-meta` - 31 tests pass
- Clippy: `cargo clippy -p grapheme-meta -- -D warnings` - 0 warnings
- 14 new tests in `learnable::tests` module

### Context for Next Task
- Uncertainty uses sigmoid for bounded [0,1] outputs
- Calibrator uses temperature scaling (increases when overconfident)
- IntrospectionMonitor tracks state history (50 samples) for trend detection
- ECE (Expected Calibration Error) computed with 10 bins
