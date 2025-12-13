---
id: backend-033
title: Add learnable reasoning with rule confidence updating
status: done
priority: high
tags:
- backend
dependencies:
- backend-031
assignee: developer
created: 2025-12-06T09:49:32.495698782Z
estimate: ~
complexity: 3
area: backend
---

# Add learnable reasoning with rule confidence updating

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
Add learnable components to the reasoning system that can update rule confidences
based on outcomes, enabling the system to learn which rules are more reliable.

## Objectives
- Create learnable rule confidence with Bayesian updates
- Implement neural rule selection for ranking rules
- Provide gradient flow for learning from reasoning outcomes
- Integrate with the Deduction trait

## Tasks
- [x] Implement LearnableRuleConfidence with Bayesian updates
- [x] Implement NeuralRuleSelector with neural scoring
- [x] Create LearnableDeduction implementing Deduction trait
- [x] Add learn_from_proof for outcome-based learning
- [x] Write comprehensive unit tests

## Acceptance Criteria
✅ **Bayesian Confidence Updates:**
- Confidence increases on success, decreases on failure
- Uses Laplace smoothing: (successes+1)/(applications+2)

✅ **Neural Rule Selection:**
- Rules scored by neural network based on input features
- Combined score = neural_score * confidence

## Technical Notes
- Uses GRAPHEME Protocol: LeakyReLU (α=0.01), lr=0.001
- NeuralRuleSelector: 18 input features → hidden_dim → score [0,1]
- Bayesian update provides natural regularization for confidence
- LearnableDeduction implements Deduction trait for seamless integration

## Testing
- [x] Write unit tests for new functionality (12 new tests)
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete (24 total)
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
- Created new file: `grapheme-reason/src/learnable.rs`
- Updated `grapheme-reason/src/lib.rs` with module and re-exports
- Updated `grapheme-reason/Cargo.toml` with ndarray and rand dependencies
- Key structures:
  - `LearnableRuleConfidence`: Bayesian confidence tracking for rules
  - `NeuralRuleSelector`: Neural network for rule scoring
  - `LearnableDeduction`: Deduction trait implementation with learning
  - `RuleOutcome`: Success/Failure/Uncertain enum

### Causality Impact
- Outcome feedback: apply rule → observe outcome → update confidence
- Bayesian update: (successes+1)/(applications+2) for natural regularization
- Neural scoring: input features → hidden → sigmoid → score [0,1]
- Combined ranking: neural_score * bayesian_confidence

### Dependencies & Integration
- Added `ndarray.workspace = true` and `rand.workspace = true`
- LearnableDeduction implements the Deduction trait
- Uses GraphFingerprint for feature extraction (18 features)
- learn_from_proof() updates confidences from reasoning traces

### Verification & Testing
- Run: `cargo test -p grapheme-reason` - 24 tests pass
- Clippy: `cargo clippy -p grapheme-reason -- -D warnings` - 0 warnings
- 12 new tests in `learnable::tests` module

### Context for Next Task
- Confidence uses Bayesian update, not gradient-based (more stable)
- NeuralRuleSelector uses gradient-based learning for rule scoring
- Combined score provides both learned preference and reliability
- Match threshold is 0.7 for pattern matching