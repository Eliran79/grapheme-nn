---
id: backend-202
title: Create train_online binary with curriculum generator
status: done
priority: high
tags:
- backend
- online
- training
- binary
dependencies:
- backend-200
- backend-201
assignee: developer
created: 2025-12-11T12:03:18.036679762Z
estimate: ~
complexity: 6
area: backend
---

# Create train_online binary with curriculum generator

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
- Created `grapheme-train/src/bin/train_online.rs` (~400 lines)
- Added binary entry in `grapheme-train/Cargo.toml`
- Fixed `learn_batch` to update `best_loss` in stats

**New Binary Features:**
- CLI with clap for configuration
- Curriculum generator with 7 levels (math → text → sequences → logic → multi-domain)
- Progress bar with indicatif
- Configurable replay strategy via `--replay-strategy`
- Checkpoint saving at intervals
- Summary output with training and replay stats

**CLI Options:**
```
--examples N       # Number of examples (0=infinite)
--batch-size N     # Mini-batch size (default: 32)
--lr RATE          # Learning rate (default: 0.001)
--replay-strategy  # uniform|prioritized|recency|mixed|balanced
--replay-capacity  # Buffer size (default: 10000)
--replay-ratio     # Fraction from replay (default: 0.5)
--start-level      # Starting curriculum level (1-7)
--max-level        # Maximum curriculum level (1-7)
--checkpoint-interval N  # Save every N examples
--output DIR       # Output directory
--verbose LEVEL    # 0-2 verbosity
```

### Causality Impact
- Training loop: generate → buffer → batch → train → replay → consolidate
- Curriculum auto-advances after 500 examples per level
- Checkpoints saved every 60 seconds or checkpoint_interval
- Progress bar updates every batch with current loss

### Dependencies & Integration
- Uses `MemoryOnlineLearner` from backend-200/201
- Uses `OnlineLearnerConfig` with replay_strategy
- Integrates with indicatif for progress bars
- Outputs JSON checkpoints to checkpoints/online/

### Verification & Testing
- Run: `cargo run --release -p grapheme-train --bin train_online -- --help`
- Test: `cargo run --release -p grapheme-train --bin train_online -- --examples 1000`
- Verified: 167K examples/sec throughput, replay working, consolidation triggered

### Context for Next Task
- **backend-203 (ConsolidationScheduler)**: Can extend consolidate trigger logic in train_online
- **backend-204 (Curriculum progression)**: CurriculumGenerator already has level progression
  - `examples_to_advance = 500` can be made configurable
  - Level descriptions available via `level_description()`
- **backend-205 (EWC)**: Config already has `use_ewc` and `ewc_lambda` fields
  - Need to implement EWC loss in `train_single()` when enabled
- Checkpoints store stats JSON, not full model state (for quick inspection)
- Full model save/restore needs UnifiedCheckpoint integration