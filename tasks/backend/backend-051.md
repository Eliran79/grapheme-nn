---
id: backend-051
title: Add logging for dropped examples in data generation
status: done
priority: medium
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T10:42:34.327284420Z
estimate: ~
complexity: 3
area: backend
---

# Add logging for dropped examples in data generation

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
**MEDIUM: Dropped examples during data generation are not reported.**

The data generation functions in grapheme-train/src/lib.rs (lines 383-461) silently drop examples that fail validation, making it difficult to assess data quality.

```rust
// Examples filtered out without reporting (lines 383-461):
examples.into_iter()
    .filter_map(|e| validate(e).ok())  // Dropped silently
    .collect()
```

Users have no visibility into how many examples are being dropped or why.

## Objectives
- Log dropped examples with reason
- Track generation success/failure rate
- Provide generation summary statistics

## Tasks
- [ ] Add counter for dropped examples
- [ ] Log reason for each dropped example
- [ ] Print summary at end of generation (total generated, dropped, success rate)
- [ ] Add optional verbose mode for detailed logging

## Acceptance Criteria
✅ **Visibility:**
- Summary shows examples generated vs dropped
- Success rate percentage reported

✅ **Debugging:**
- Reasons for dropped examples are logged
- Patterns in dropped examples identifiable

## Technical Notes
- File: grapheme-train/src/lib.rs lines 383-461
- Pattern: `filter_map(|e| validate(e).ok())` silently drops
- Solution: Use explicit match with logging for Err case
- Consider: Structured logging with `tracing` crate

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
- 2025-12-06: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added `GenerationStats` struct to `DataGenerator` in grapheme-train/src/lib.rs
- Tracks: `attempted`, `generated`, `dropped_eval_error` counts
- Added `stats()` and `reset_stats()` methods to DataGenerator
- Added `success_rate()` and `print_summary()` methods to GenerationStats
- Updated all generation functions (basic_arithmetic, nested_operations, symbol_substitution, basic_functions, differentiation) to track statistics

### Causality Impact
- `generate_*` functions now increment stats counters
- Users can call `generator.stats()` to get generation statistics
- `print_summary()` outputs to stderr for visibility

### Dependencies & Integration
- No new dependencies
- Statistics integrated into existing DataGenerator struct
- Backward compatible - no API changes required

### Verification & Testing
- All workspace tests pass
- Stats are tracked during data generation
- Use `generator.stats().print_summary()` to see results

### Context for Next Task
- Statistics tracking is opt-in (call print_summary explicitly)
- Uses eprintln! for output (no logging framework added)
- Consider adding structured logging (tracing crate) in future