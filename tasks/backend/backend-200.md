---
id: backend-200
title: Create OnlineLearner trait using grapheme-memory
status: done
priority: high
tags:
- backend
- online
- memory
- trait
dependencies: []
assignee: developer
created: 2025-12-11T12:03:17.658158472Z
estimate: ~
complexity: 5
area: backend
---

# Create OnlineLearner trait using grapheme-memory

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
- Created new file: `grapheme-train/src/online_learner.rs` (~630 lines)
- Added module export in `grapheme-train/src/lib.rs`
- Added `grapheme-memory` dependency to `grapheme-train/Cargo.toml`

**New Types:**
- `OnlineLearnerConfig` - Configuration for learning rate, batch size, replay ratio, EWC
- `OnlineExample` - Training example with id, input/target vectors, domain, level, last_loss
- `OnlineLearnerStats` - Statistics tracking (examples_seen, avg_loss, domain_counts)
- `OnlineLearner` trait - Core interface with `learn_one`, `learn_batch`, `consolidate`
- `MemoryOnlineLearner` - Implementation using grapheme-memory integration

**Key Methods:**
- `learn_one(example)` - Train on single example, store in episodic memory
- `learn_batch(batch)` - Batch training with replay mixing (configurable ratio)
- `consolidate()` - Apply retention policy, trigger replay_and_integrate
- `save_state()/load_state()` - Checkpoint serialization via UnifiedCheckpoint

### Causality Impact
- Each `learn_one` call: trains → stores in episodic memory → updates stats → checks consolidation
- Consolidation triggers when `examples_since_consolidation >= consolidation_interval`
- Batch training mixes new examples with replay samples based on `replay_ratio`
- Replay samples are recalled from EpisodicMemory using `recall(&query, count)`

### Dependencies & Integration
- Uses `BackwardPass` trait from grapheme-core for gradient computation
- Uses `Embedding` for gradient accumulation context
- Uses `SimpleEpisodicMemory` for experience replay (stores as Episodes)
- Uses `SimpleContinualLearning` for consolidation via `replay_and_integrate()`
- Uses `RetentionPolicy` for memory management

### Verification & Testing
- Run: `cargo build -p grapheme-train`
- Run: `cargo test -p grapheme-train online_learner` (9 tests pass)
- Tests cover: config presets, example creation, learn_one, learn_batch, consolidation, stats

### Context for Next Task
- **backend-201**: Experience replay is already partially implemented via `sample_replay()`
  - Currently uses `episodic_memory.recall(&query, count)` with empty query
  - May need prioritized replay based on `example.last_loss`
- **backend-202**: train_online binary needs to use `OnlineLearner` trait
  - Use `MemoryOnlineLearner::with_default_model(config)` for initialization
  - Call `learner.learn_one()` or `learner.learn_batch()` per example/batch
  - Curriculum generator should produce `OnlineExample` with appropriate `level`
- Config presets: `OnlineLearnerConfig::fast()` for quick learning, `stable()` for EWC-enabled