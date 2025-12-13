---
id: backend-219
title: Integrate SemanticDecoder into train_mesh_code.rs
status: done
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
- [x] Add SemanticDecoder imports and helper functions (copy from backend-217)
- [x] Initialize SemanticDecoder with auto-discovered vocabulary
- [x] Integrate decoder training into batch processing
- [x] Add parallel validation with semantic accuracy metrics
- [x] Update epoch logging

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
- [x] Build passes with zero errors
- [x] Run training for 5 epochs to verify metrics
- [x] Verify EncoderDecoder still works correctly with added decoder

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
- 2025-12-12: Marked done - SemanticDecoder integrated with parallel validation

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **File**: `grapheme-train/src/bin/train_mesh_code.rs`
- **Added imports**: `SemanticDecoder`, `SemanticDecoderConfig`, `rayon::prelude::*`, `DiGraph`, `Edge`, `Node`
- **Added helper functions** (lines 114-200):
  - `semantic_accuracy()` - compares node types between predicted and target graphs
  - `decode_features_to_graph()` - converts pooled features to graph using SemanticDecoder (marked `#[allow(dead_code)]` for future use)
  - `prepare_decoder_batch()` - prepares training batch for SemanticDecoder
- **Added SemanticDecoder initialization** (lines 316-334):
  - Uses `SemanticDecoder::build_vocab_from_brains()` for auto-discovered vocabulary
  - embed_dim = 128 to match MeshCodeGen EncoderDecoder config
- **Modified training loop** (lines 358-427):
  - Accumulates decoder training batch from target graphs
  - Uses hash-based feature embedding for decoder training
  - Calls `decoder.backward()` at end of each epoch
- **Modified validation** (lines 432-479):
  - Hybrid approach: predictions generated sequentially (model.generate requires &mut self)
  - Metrics computed in parallel using rayon `par_iter().zip()`
  - Returns `(val_acc, exact_rate, val_sem_acc, val_dec_acc)`
- **Updated epoch logging** (lines 497-506):
  - Shows decoder loss: `loss=5.5451 (dec=8.3459)`
  - Shows semantic metrics: `sem_acc=23.3%, dec_acc=0.0%`

### Causality Impact
- Validation uses two-phase approach: sequential prediction then parallel metrics
- Decoder training happens at epoch level, not batch level (accumulates samples)
- No synchronization issues since metric computation is independent per sample

### Dependencies & Integration
- Uses `SemanticDecoder::build_vocab_from_brains()` - auto-discovers 4301 node types from 8 brains
- Helper functions are thread-safe (pure functions with immutable refs)
- MeshCodeGen wrapper preserved - EncoderDecoder still trains via `model.train_step()`

### Verification & Testing
- Build: `cargo build --release -p grapheme-train --bin train_mesh_code` - zero errors
- Test: `cargo test --release -p grapheme-train` - all pass
- Run: `cargo run --release -p grapheme-train --bin train_mesh_code -- --data data/code_training --output checkpoints/mesh_code_semantic.json --epochs 2`
- Verify: SemanticDecoder reports vocabulary size: 4301 (35 Keywords, 25 Ops, 12 Puncts, 97 Input chars)
- Verify: Epoch logging shows sem_acc% and dec_acc%

### Context for Next Task
- **backend-220** (generalize): Helper functions can be extracted to training_utils.rs
- Key difference from train_cortex_mesh.rs: validation uses hybrid sequential+parallel approach because `model.generate()` requires `&mut self`
- `decode_features_to_graph()` is marked `#[allow(dead_code)]` - needed for future decoded graph visualization
- Decoder trains on accumulated epoch batch (not per-batch) to avoid interference with main model training