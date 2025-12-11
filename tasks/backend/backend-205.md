---
id: backend-205
title: Implement EWC loss integration for forgetting prevention
status: done
priority: medium
tags:
- backend
- online
- ewc
- regularization
dependencies:
- backend-202
assignee: developer
created: 2025-12-11T12:03:26.563091741Z
estimate: ~
complexity: 6
area: backend
---

# Implement EWC loss integration for forgetting prevention

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
Brief description of what needs to be done and why.

## Objectives
- Clear, actionable objectives
- Measurable outcomes
- Success criteria

## Tasks
- [ ] Break down the work into specific tasks
- [ ] Each task should be clear and actionable
- [ ] Mark tasks as completed when done

## Acceptance Criteria
✅ **Criteria 1:**
- Specific, testable criteria

✅ **Criteria 2:**
- Additional criteria as needed

## Technical Notes
- Implementation details
- Architecture considerations
- Dependencies and constraints

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
- 2025-12-11: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added `EWCState` struct (~140 lines) in `online_learner.rs`
- Added `EWCStats` struct for statistics reporting
- Added `EdgeKey` type alias for edge identification
- Added `ewc_state` field to `MemoryOnlineLearner`
- Added `petgraph::visit::EdgeRef` import for edge iteration
- Modified `train_single()` to add EWC gradient penalty to edge weights
- Modified `consolidate()` to call `update_ewc_fisher_from_buffer()` and `consolidate_ewc()`
- Added helper methods: `get_current_edge_weights()`, `get_edge_gradients()`
- Updated lib.rs exports to include `EWCState`, `EWCStats`, `EdgeKey`
- Added 14 unit tests for EWC functionality

**New Types:**
```rust
pub type EdgeKey = (usize, usize);  // (src_index, tgt_index)

pub struct EWCState {
    pub fisher_diag: HashMap<EdgeKey, f32>,     // Fisher information per edge
    pub optimal_params: HashMap<EdgeKey, f32>,  // Optimal weights at consolidation
    pub task_count: usize,                      // Number of consolidations
    pub samples_used: usize,                    // Samples for Fisher estimation
    pub is_active: bool,                        // Whether EWC is protecting weights
}
```

**Key Methods:**
- `EWCState::compute_penalty(current_weights, lambda)` - EWC loss term
- `EWCState::compute_gradient_penalty(edge_key, current_weight, lambda)` - Per-edge gradient
- `EWCState::update_fisher(edge_grads)` - Online Fisher estimation
- `EWCState::consolidate(current_weights)` - Record optimal weights
- `MemoryOnlineLearner::ewc_active()` - Check if EWC is protecting weights
- `MemoryOnlineLearner::compute_ewc_penalty()` - Current penalty loss

### Causality Impact
- During `train_single()`: After backward pass, EWC gradient penalty is added to edge gradients
- During `consolidate()`: Fisher information is estimated from recent buffer, then optimal weights are recorded
- EWC gradient: `grad_ewc = lambda * F_i * (weight_i - weight_optimal_i)`
- EWC penalty: `L_ewc = (lambda/2) * sum(F_i * (weight_i - weight_optimal_i)^2)`

### Dependencies & Integration
- Uses `petgraph::visit::EdgeRef` for edge iteration
- Uses existing `OnlineLearnerConfig.use_ewc` and `ewc_lambda` fields
- Integrates with `consolidate()` flow - EWC updates happen during consolidation
- Works with any model that has edges (minimal models without edges will have EWC inactive)

### Verification & Testing
- Run: `cargo test -p grapheme-train --lib online_learner` (50 tests pass)
- Build: `cargo build --release -p grapheme-train`
- Test binary: `cargo run --release -p grapheme-train --bin train_online -- --examples 100`
- EWC is enabled by default in `OnlineLearnerConfig::stable()`

### Context for Next Task
- EWC only tracks edge weights (not node activations) - this matches how DagNN stores learnable parameters
- Fisher information is estimated from recent buffer samples before clearing
- `is_active` only becomes true when there are edges to protect
- EWC lambda default is 0.1 in stable config, 0.0 (disabled) in default config
- Online Fisher estimation uses running mean: `F_new = F_old + (grad^2 - F_old) / n`