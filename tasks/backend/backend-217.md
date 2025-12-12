---
id: backend-217
title: Integrate SemanticDecoder into CortexMesh training loop
status: todo
priority: high
tags:
- backend
- parallel
- autodiscovery
- semantic
dependencies:
- backend-214
assignee: developer
created: 2025-12-12T08:37:02.333481609Z
estimate: ~
complexity: 5
area: backend
---

# Integrate SemanticDecoder into CortexMesh training loop

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
The `train_cortex_mesh.rs` training binary has been prepared with SemanticDecoder integration (imports, helper functions) but the training loop has not been updated to use parallel processing with the auto-discovered unified vocabulary. This task completes the integration to allow all CortexMesh brains to train with semantic accuracy metrics.

Key insight: The previous model collapse (0% accuracy) happened because outputs were copied from inputs. SemanticDecoder with auto-discovered vocabulary allows generating NEW semantic node types (Keyword, Variable, Op, etc.) not present in the input.

## Objectives
- Use `SemanticDecoder::build_vocab_from_brains()` for auto-discovered unified vocabulary
- Integrate parallel batch processing using rayon `par_iter()`
- Train decoder alongside CortexMesh model
- Report semantic accuracy (% of matching node types) as primary metric

## Tasks
- [x] Add SemanticDecoder imports and helper functions (DONE in current uncommitted changes)
- [x] Initialize SemanticDecoder with `build_vocab_from_brains()` (DONE)
- [ ] Use helper functions in training loop for decoder backward pass
- [ ] Add decoder loss to total training loss
- [ ] Update epoch logging to show decoder accuracy

## Acceptance Criteria
✅ **Auto-Discovery:**
- Uses `SemanticDecoder::build_vocab_from_brains()` (not hardcoded types)
- Automatically includes new node types when brains are added

✅ **Training Integration:**
- Decoder trains alongside CortexMesh model
- Reports decoder_loss and decoder_accuracy per epoch

✅ **Parallel Processing:**
- Batch processing uses rayon `par_iter()` for CPU parallelization

## Technical Notes
- File: `grapheme-train/src/bin/train_cortex_mesh.rs`
- Helper functions already added: `semantic_accuracy()`, `decode_features_to_graph()`, `prepare_decoder_batch()`
- SemanticDecoder initialized at line 305-323
- Training loop starts at line 337
- Use `prepare_decoder_batch()` to get (hidden, target_idx) pairs
- Call `decoder.backward()` with batch for training

## Testing
- [ ] Build passes with zero errors
- [ ] Run training for 5 epochs to verify semantic accuracy improves
- [ ] Verify auto-discovery works (vocab should have 4301 types from 8 brains)

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
