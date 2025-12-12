---
id: backend-219
title: Integrate SemanticDecoder into train_mesh_code.rs
status: todo
priority: medium
tags:
- backend
- parallel
- autodiscovery
- semantic
dependencies:
- backend-217
assignee: developer
created: 2025-12-12T08:37:02.806598857Z
estimate: ~
complexity: 5
area: backend
---

# Integrate SemanticDecoder into train_mesh_code.rs

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
The `train_mesh_code.rs` binary combines CortexMesh with EncoderDecoder for code generation. It needs the same SemanticDecoder integration and parallel processing as `train_cortex_mesh.rs` to enable semantic accuracy metrics and efficient CPU utilization.

## Objectives
- Apply same pattern from backend-217 to train_mesh_code.rs
- Use auto-discovered vocabulary via `SemanticDecoder::build_vocab_from_brains()`
- Add parallel batch processing for training and validation
- Report semantic accuracy alongside code generation metrics

## Tasks
- [ ] Add SemanticDecoder imports and helper functions (copy from backend-217)
- [ ] Initialize SemanticDecoder with auto-discovered vocabulary
- [ ] Integrate decoder training into batch processing
- [ ] Add parallel validation with semantic accuracy metrics
- [ ] Update epoch logging

## Acceptance Criteria
✅ **Auto-Discovery:**
- Uses `build_vocab_from_brains()` for unified vocabulary (not hardcoded)
- New node types automatically included when brains are added

✅ **Training Integration:**
- Decoder trains alongside CortexMesh + EncoderDecoder
- Reports decoder accuracy per epoch

## Technical Notes
- File: `grapheme-train/src/bin/train_mesh_code.rs`
- This file uses CortexMesh + EncoderDecoder architecture
- Copy helper functions from train_cortex_mesh.rs (semantic_accuracy, decode_features_to_graph, prepare_decoder_batch)
- SemanticDecoder config should match embed_dim of the model

## Testing
- [ ] Build passes with zero errors
- [ ] Run training for 5 epochs to verify metrics
- [ ] Verify EncoderDecoder still works correctly with added decoder

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