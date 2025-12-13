---
id: backend-201
title: Implement experience replay using EpisodicMemory
status: done
priority: high
tags:
- backend
- online
- memory
- replay
dependencies:
- backend-200
assignee: developer
created: 2025-12-11T12:03:17.800796170Z
estimate: ~
complexity: 5
area: backend
---

# Implement experience replay using EpisodicMemory

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
- Enhanced `grapheme-train/src/online_learner.rs` with prioritized experience replay
- Updated `grapheme-train/src/lib.rs` exports to include new types

**New Types Added:**
- `ReplayStrategy` enum: Uniform, PrioritizedLoss, PrioritizedRecency, Mixed, DomainBalanced
- `ReplayMetadata` struct: tracks episode_id, loss, domain, timestamp, replay_count
- `ReplayStats` struct: buffer_size, total_replays, avg_loss, domain_distribution

**New Config Fields:**
- `replay_strategy: ReplayStrategy` - which sampling strategy to use
- `priority_alpha: f32` - exponent for priority weighting (higher = more focus)

**New Config Presets:**
- `OnlineLearnerConfig::prioritized()` - focus on hard examples
- `OnlineLearnerConfig::balanced()` - domain-balanced sampling

**New Methods on MemoryOnlineLearner:**
- `sample_by_priority(&weights, count)` - weighted sampling
- `sample_uniform(total, count)` - uniform random sampling
- `sample_replay(&mut self, count)` - sample using configured strategy
- `update_replay_metadata(episode_id, new_loss)` - update loss after replay
- `add_replay_metadata(example, episode_id)` - track new examples
- `replay_stats()` - get buffer statistics

### Causality Impact
- Replay sampling is now strategy-aware (not just random)
- Loss is updated in metadata after each replay (adaptive learning)
- Buffer eviction follows strategy: PrioritizedLoss removes easiest, Recency removes oldest
- Domain-balanced ensures equal representation across domains
- Mixed strategy combines 50% loss-prioritized + 50% uniform

### Dependencies & Integration
- Builds on backend-200's OnlineLearner foundation
- Uses xorshift64 RNG for sampling (no external dependency)
- Compatible with existing EpisodicMemory storage
- Works with all existing OnlineLearner trait methods

### Verification & Testing
- Run: `cargo test -p grapheme-train online_learner` (18 tests pass)
- New tests: test_replay_strategy_enum, test_config_presets_have_strategies,
  test_replay_metadata_tracking, test_replay_with_prioritized_loss,
  test_replay_domain_balanced, test_replay_capacity_enforcement,
  test_replay_mixed_strategy, test_replay_recency_strategy,
  test_replay_stats_serialization

### Context for Next Task
- **backend-202 (train_online binary)**: Use `MemoryOnlineLearner::with_default_model(config)`
  - Select strategy based on training goals:
    - `fast()` for quick iteration (Uniform)
    - `stable()` for long training (PrioritizedLoss)
    - `prioritized()` for hard example focus
    - `balanced()` for multi-domain training
  - Access stats: `learner.stats()` for learning stats, `learner.replay_stats()` for buffer stats
- `replay_stats()` can be used for monitoring/logging buffer health
- Replay buffer auto-enforces capacity; eviction strategy matches replay strategy