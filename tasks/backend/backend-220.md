---
id: backend-220
title: Generalize parallel CPU processing across all mesh trainers
status: todo
priority: medium
tags:
- backend
- parallel
- refactor
- semantic
dependencies:
- backend-217
- backend-218
- backend-219
assignee: developer
created: 2025-12-12T08:37:03.007927852Z
estimate: ~
complexity: 6
area: backend
---

# Generalize parallel CPU processing across all mesh trainers

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
After backend-217/218/219, the parallel processing and SemanticDecoder integration patterns are duplicated across multiple training binaries. This task extracts the common functionality into shared utilities to reduce code duplication and ensure consistency. New trainers should automatically benefit from auto-discovery.

## Objectives
- Extract common helper functions into a shared module
- Create a reusable TrainingUtils struct with parallel batch processing
- Ensure all mesh trainers use the same auto-discovery mechanism
- Reduce code duplication across training binaries

## Tasks
- [ ] Create `grapheme-train/src/training_utils.rs` module
- [ ] Move helper functions: `semantic_accuracy()`, `decode_features_to_graph()`, `prepare_decoder_batch()`
- [ ] Add `ParallelTrainer` or `TrainingContext` struct with common batch logic
- [ ] Refactor train_cortex_mesh.rs to use shared utilities
- [ ] Refactor train_mesh_code.rs to use shared utilities
- [ ] Update lib.rs to export new module

## Acceptance Criteria
✅ **Code Reuse:**
- Helper functions in single shared location
- Trainers use shared utilities instead of duplicated code

✅ **Auto-Discovery Preserved:**
- `SemanticDecoder::build_vocab_from_brains()` still auto-discovers node types
- Adding new brains automatically updates all trainers

✅ **Backward Compatible:**
- No change in training behavior or metrics

## Technical Notes
- New module: `grapheme-train/src/training_utils.rs`
- Consider trait-based design for different training architectures (CortexMesh, CortexMesh+EncoderDecoder, etc.)
- Helper functions should be generic where possible
- Keep SemanticDecoder initialization in individual trainers (they may have different configs)

## Testing
- [ ] Build passes with zero errors
- [ ] All existing tests pass
- [ ] Run both trainers for 2 epochs to verify unchanged behavior
- [ ] Verify semantic accuracy metrics are consistent

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
- 2025-12-12: Task created

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