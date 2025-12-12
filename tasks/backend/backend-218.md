---
id: backend-218
title: Add parallel validation with semantic accuracy metrics
status: done
priority: high
tags:
- backend
- parallel
- validation
- semantic
dependencies:
- backend-217
assignee: developer
created: 2025-12-12T08:37:02.564622771Z
estimate: ~
complexity: 4
area: backend
---

# Add parallel validation with semantic accuracy metrics

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
After backend-217 integrates SemanticDecoder training, the validation loop needs parallel processing for semantic accuracy computation. The current validation loop is sequential and only computes structural loss. This task adds parallel validation with proper semantic metrics.

## Objectives
- Parallelize validation loop using rayon `par_iter()`
- Compute semantic accuracy using `decode_features_to_graph()` during validation
- Report both structural loss and semantic accuracy per epoch
- Use atomic counters for parallel-safe metric accumulation

## Tasks
- [x] Wrap validation samples with `par_iter()` for parallel processing
- [x] Use AtomicUsize/AtomicF32 for loss and accuracy accumulation (used Vec::par_iter().collect() instead - cleaner)
- [x] Compute semantic accuracy using `semantic_accuracy()` helper
- [x] Update epoch logging to show val_loss and val_semantic_acc

## Acceptance Criteria
✅ **Parallel Validation:**
- Uses rayon `par_iter()` for parallel processing of validation samples
- Thread-safe metric accumulation with atomic operations

✅ **Semantic Metrics:**
- Reports semantic accuracy (% matching node types) on validation set
- Reports validation structural loss

## Technical Notes
- File: `grapheme-train/src/bin/train_cortex_mesh.rs`
- Validation loop starts around line 392
- Use `AtomicUsize` for sample counts, accumulate into `AtomicF32` or mutex-protected floats
- Helper `semantic_accuracy()` already defined (compares node types between pred and target)
- Use `decode_features_to_graph()` to convert features to graph for comparison

## Testing
- [x] Build passes with zero errors
- [x] Validation time decreases with parallel processing
- [x] Semantic accuracy is computed and logged per epoch

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
- 2025-12-12: Marked done - implemented as part of backend-217

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **File**: `grapheme-train/src/bin/train_cortex_mesh.rs` (lines 419-459)
- **Implemented**: Parallel validation with `par_iter()` from rayon
- **Note**: This was implemented as part of backend-217 (SemanticDecoder integration)
- **Implementation approach**: Used `par_iter().collect()` pattern instead of atomic counters - cleaner and equally thread-safe

### Causality Impact
- Validation now runs in parallel using all CPU cores
- Metrics (val_loss, sem_acc, dec_acc) computed per sample in parallel, then aggregated
- No synchronization issues since each sample's metrics are independent

### Dependencies & Integration
- Uses `mesh.model.forward()` which takes `&self` (immutable) - thread-safe
- Uses `compute_structural_loss()`, `semantic_accuracy()`, `prepare_decoder_batch()`, `decoder.compute_accuracy()` helper functions
- All helper functions are pure/thread-safe

### Verification & Testing
- Build: `cargo build --release -p grapheme-train --bin train_cortex_mesh` - zero warnings
- Test: `cargo test --release -p grapheme-train` - all pass
- Run: Training shows semantic metrics in epoch output: `val=871.84, sem_acc=1.0%, dec_acc=24.5%`

### Context for Next Task
- **backend-219** (train_mesh_code.rs): Copy same parallel validation pattern
- **backend-220** (generalization): Extract parallel validation helpers to training_utils.rs
- Key insight: `par_iter().collect()` is cleaner than atomic operations for this use case