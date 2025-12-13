---
id: backend-036
title: Add learnable agency with adaptive goals and values
status: done
priority: medium
tags:
- backend
dependencies:
- backend-031
- backend-034
assignee: developer
created: 2025-12-06T09:49:43.470374083Z
estimate: ~
complexity: 3
area: backend
---

# Add learnable agency with adaptive goals and values

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
Add learnable components to the agency module for adaptive goal prioritization,
value estimation, and context-aware drive adjustment. Follows GRAPHEME Protocol
(LeakyReLU α=0.01, DynamicXavier, Adam lr=0.001).

## Objectives
- Create learnable goal encoders for fixed-size goal representations
- Implement learnable value networks for goal/state value estimation
- Add learnable drive networks for adaptive motivation
- Create learnable priority networks for context-aware goal prioritization
- Enable experience-based learning from goal outcomes

## Tasks
- [x] Implement GoalEncoder with graph feature extraction and priority encoding
- [x] Implement ValueNetwork with TD error computation
- [x] Implement DriveNetwork for adaptive motivation
- [x] Implement PriorityNetwork for context-aware goal selection
- [x] Create LearnableAgency model combining all components
- [x] Add experience buffer for learning from outcomes
- [x] Write comprehensive unit tests (17 new tests)

## Acceptance Criteria
✅ **Goal Encoding:**
- Encodes goal graphs and priority features to L2-normalized embeddings
- Caches encodings for efficiency

✅ **Value Estimation:**
- Estimates goal value in [0, 1] range using sigmoid output
- Supports TD error computation for temporal difference learning

✅ **Adaptive Drives:**
- Computes context-dependent drive strengths using softmax
- Returns Drive enum variants with adaptive strengths

✅ **Goal Prioritization:**
- Computes priority scores based on goal + context embeddings
- Supports selecting best goal from candidates

## Technical Notes
- Uses GRAPHEME Protocol: LeakyReLU (α=0.01), lr=0.001, DynamicXavier
- GoalEncoder: graph features (18) + priority features (4) → combined → embed_dim
- ValueNetwork: embed_dim → hidden_dim → hidden_dim → 1 (sigmoid)
- DriveNetwork: embed_dim → num_drives (softmax)
- PriorityNetwork: (goal_embed + context_embed) → hidden → 1 (sigmoid)

## Testing
- [x] Write unit tests for new functionality (17 new tests)
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete (31 total in grapheme-agent)
- [x] Consider edge cases and error conditions

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
- [x] Commit changes incrementally with clear messages
- [x] Use descriptive commit messages that explain the "why"
- [x] Review changes before committing

## Updates
- 2025-12-06: Task created
- 2025-12-13: Task completed - Learnable agency infrastructure added

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created new file: `grapheme-agent/src/learnable.rs` (~750 lines)
- Updated `grapheme-agent/src/lib.rs` with module declaration and re-exports
- Updated `grapheme-agent/Cargo.toml` with ndarray, rand, grapheme-memory deps
- Key structures:
  - `GoalEncoder`: Encodes goals (graph + priority) to embeddings
  - `ValueNetwork`: Estimates goal/state value
  - `DriveNetwork`: Computes adaptive drive strengths
  - `PriorityNetwork`: Computes context-aware goal priorities
  - `LearnableAgency`: Complete model combining all components
  - `GoalExperience`: Experience tuple for learning

### Causality Impact
- Goal selection flow: goals → encode → priority × value → select best
- Value learning: record_goal_outcome → experience_buffer → compute_value_loss
- Drive adaptation: set_context → get_adaptive_drives → context-aware motivation
- All components use GRAPHEME Protocol (LeakyReLU α=0.01)

### Dependencies & Integration
- Added `ndarray.workspace = true`, `rand.workspace = true`, `grapheme-memory`
- Re-exports from lib.rs: LearnableAgency, GoalEncoder, ValueNetwork, etc.
- Integrates with existing Goal, Drive, Graph types from grapheme-agent

### Verification & Testing
- Run: `cargo test -p grapheme-agent` - 31 tests pass
- Clippy: `cargo clippy -p grapheme-agent -- -D warnings` - 0 warnings
- 17 new tests in `learnable::tests` module

### Context for Next Task
- Goal embeddings are L2-normalized for cosine similarity
- Value estimates are sigmoid-bounded [0, 1]
- Drive strengths use softmax (sum to 1.0)
- Experience buffer limited to config.buffer_size (default 1000)
- Goal cache can be cleared with clear_cache() for memory management
