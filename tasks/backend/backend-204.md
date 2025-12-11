---
id: backend-204
title: Add curriculum progression (Level 1->7) in online mode
status: done
priority: medium
tags:
- backend
- online
- curriculum
dependencies:
- backend-202
assignee: developer
created: 2025-12-11T12:03:26.237806908Z
estimate: ~
complexity: 4
area: backend
---

# Add curriculum progression (Level 1->7) in online mode

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
- Added `CurriculumConfig` struct in `online_learner.rs` (~60 lines)
- Added `CurriculumState` struct in `online_learner.rs` (~90 lines)
- Updated lib.rs exports to include `CurriculumConfig`, `CurriculumState`
- Added 12 unit tests for curriculum functionality

**New Types:**
```rust
pub struct CurriculumConfig {
    pub start_level: u8,       // 1-7
    pub max_level: u8,         // 1-7
    pub examples_per_level: usize,
    pub advance_loss_threshold: Option<f32>,
    pub min_examples_before_advance: usize,
    pub allow_regression: bool,
    pub regression_threshold: f32,
}

pub struct CurriculumState {
    pub current_level: u8,
    pub examples_at_level: usize,
    pub best_loss_at_level: f32,
    pub advancements: usize,
    pub regressions: usize,
}
```

**Methods:**
- `CurriculumConfig::fast()` - Quick curriculum (200 examples/level)
- `CurriculumConfig::thorough()` - Careful curriculum (1000 examples/level)
- `CurriculumConfig::with_levels(start, max)` - Custom level range
- `CurriculumState::record_example(loss)` - Returns true if level changed
- `CurriculumState::level_progress()` - Progress within current level (0.0-1.0)

### Causality Impact
- Curriculum can advance early if loss drops below threshold
- Regression optional (disabled by default) when loss exceeds regression_threshold
- State tracks advancements/regressions for monitoring

### Dependencies & Integration
- Fully serializable with serde for checkpointing
- Works alongside `CurriculumGenerator` in train_online binary
- Can be integrated with `MemoryOnlineLearner.stats.current_level`

### Verification & Testing
- Run: `cargo test -p grapheme-train online_learner::tests::test_curriculum`
- 12 tests verify: config presets, state transitions, early advance, regression, max level, progress

### Context for Next Task
- **backend-205 (EWC)**: Already has `use_ewc` and `ewc_lambda` in `OnlineLearnerConfig`
- EWC needs to add Fisher Information matrix computation after consolidation
- EWC loss: `L_total = L_task + (lambda/2) * sum(F_i * (theta_i - theta_star_i)^2)`
- Consider computing Fisher diagonal from replay buffer gradients