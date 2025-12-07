---
id: backend-102
title: Extend train command to support QA pairs format
status: todo
priority: medium
tags:
- backend
dependencies:
- backend-099
assignee: developer
created: 2025-12-07T17:52:36.463513113Z
estimate: ~
complexity: 3
area: backend
---

# Extend train command to support QA pairs format

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

Currently we have TWO training binaries:
1. **`train`** - Only supports math curriculum format (`input_polish`, `expected_symbolic`)
2. **`train_kindergarten_loop`** - Only supports QA pairs (`input`, `target`)

This creates maintenance overhead and splits the training logic. We should have ONE unified training command that supports both formats.

## Current State

**train.rs (lines 296-305):**
```rust
let input = &example.input_polish;  // ❌ Math-specific
let target = if let Some(ref symbolic) = &example.expected_symbolic {
    expr_to_polish(symbolic)
} else if let Some(result) = example.expected_result {
    result.to_string()
} else {
    continue;
};
```

**train_kindergarten_loop.rs (lines 122-124):**
```rust
let pair: QAPair = ...;
let input = &pair.input;    // ✓ Text pairs
let target = &pair.target;
```

## Objectives

1. **Extend Dataset enum to support both formats**
2. **Auto-detect dataset format from JSON structure**
3. **Unify training loop to handle both math and QA**
4. **Single `train` command for all use cases**

## Tasks

- [ ] Add `DatasetFormat` enum (Math, TextPairs)
- [ ] Implement format auto-detection from first JSON line
- [ ] Extend `Dataset::load_jsonl()` to handle both formats
- [ ] Update training loop to dispatch based on format
- [ ] Test with both math and kindergarten datasets
- [ ] Deprecate `train_kindergarten_loop.rs`

## Acceptance Criteria

✅ **Format Detection:**
- Auto-detects math format (has `input_polish` field)
- Auto-detects QA format (has `input` + `target` fields)
- Clear error message for unrecognized formats

✅ **Unified Training:**
- `train --config train_config.toml` works for math
- `train --config train_config_kindergarten.toml` works for QA
- Same structural loss computation for both

✅ **Backward Compatible:**
- Existing math training configs still work
- No breaking changes to checkpoint format

## Technical Notes

**Dataset Format Detection:**
```rust
pub enum DatasetFormat {
    MathCurriculum,  // input_polish, expected_symbolic
    TextPairs,       // input, target
}

impl Dataset {
    pub fn detect_format(path: &Path) -> Result<DatasetFormat> {
        // Read first line, check for field presence
    }
}
```

**Training Loop Dispatch:**
```rust
match dataset.format() {
    DatasetFormat::MathCurriculum => {
        // Use example.input_polish
    }
    DatasetFormat::TextPairs => {
        // Use example.input, example.target
    }
}
```

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
- [Document code changes, new files, modified functions]
- [What runtime behavior is new or different]

### Causality Impact
- [What causal chains were created or modified]
- [What events trigger what other events]
- [Any async flows or timing considerations]

### Dependencies & Integration
- [What dependencies were added/changed]
- [How this integrates with existing code]
- [What other tasks/areas are affected]

### Verification & Testing
- [How to verify this works]
- [What to test when building on this]
- [Any known edge cases or limitations]

### Context for Next Task
- [What the next developer/AI should know]
- [Important decisions made and why]
- [Gotchas or non-obvious behavior]