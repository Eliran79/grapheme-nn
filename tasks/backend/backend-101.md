---
id: backend-101
title: Full end-to-end training on kindergarten QA dataset
status: done
priority: high
tags:
- backend
dependencies:
- backend-099
- backend-100
assignee: developer
created: 2025-12-07T17:47:12.387755307Z
estimate: ~
complexity: 3
area: backend
---

# Full end-to-end training on kindergarten QA dataset

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
- 2025-12-07: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Extended `TrainingExample` struct to support both math curriculum and text pairs format
- Added `DatasetFormat` enum with `MathCurriculum` and `TextPairs` variants
- Implemented auto-format detection in `Dataset::load_jsonl()` based on JSON field presence
- Added `get_input()` and `get_target()` methods to `TrainingExample` for unified access
- Modified `train.rs` to use unified methods instead of format-specific field access
- Training now works with both math datasets (`input_polish`, `expected_symbolic`) and QA datasets (`input`, `target`)
- Successfully trained on kindergarten QA dataset (10 examples, text pairs format)

### Causality Impact
- `Dataset::load_jsonl()` now auto-detects format from first JSON line
- `TrainingExample.input_expr` changed from `Expr` to `Option<Expr>` (None for text pairs)
- Training loop uses `example.get_input()` and `example.get_target()` for unified access
- Validation code updated to handle both formats gracefully

### Dependencies & Integration
- No new external dependencies
- `TrainingExample` fields now have `#[serde(default)]` for backward compatibility
- Existing math training configs continue to work unchanged
- New `train_config_kindergarten.toml` works with text pairs format

### Verification & Testing
- `cargo run --bin train -- --config train_config_kindergarten.toml --epochs 5 -v` successfully trains on QA data
- All 187 workspace tests pass
- Zero clippy warnings across workspace

### Context for Next Task
- Unified train command now supports both math curriculum and text pairs datasets
- Format auto-detected from first JSON line - no manual configuration needed
- Loss is still structural graph edit distance, regardless of input format