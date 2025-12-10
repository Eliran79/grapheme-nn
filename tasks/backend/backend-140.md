---
id: backend-140
title: Implement joint training for VisionBrain + ClassificationBrain + GRAPHEME Core
status: doing
priority: critical
tags:
- backend
- training
- joint-learning
- image-classification
dependencies:
- backend-139
- backend-143
assignee: developer
created: 2025-12-09T19:29:33.452991788Z
estimate: 8h
complexity: 9
area: backend
---

# Implement joint training for VisionBrain + ClassificationBrain + GRAPHEME Core

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
Currently, `ImageClassificationModel` creates a fresh DagNN for each sample, which means
gradient updates are applied to temporary graphs that are discarded. This limits accuracy
because learned weights don't persist across samples.

This task implements weight persistence so the model can learn across the training dataset,
enabling the VisionBrain → DagNN → ClassificationBrain pipeline to achieve high accuracy
(target >90% on MNIST).

**Current state (backend-139):**
- Forward pass works end-to-end
- Loss and gradient computation works
- But gradients applied to temp DAG (discarded after each sample)
- Classification templates DO persist in ClassificationBrain

**Needed:**
- Persistent weight matrix separate from graph topology
- Weight accumulation across batches
- Shared model state between samples

## Objectives
- [ ] Implement weight persistence for DagNN across training samples
- [ ] Enable gradient accumulation over mini-batches
- [ ] Achieve >90% accuracy on MNIST validation set
- [ ] Support saving/loading trained models

## Tasks
- [ ] Design persistent weight storage strategy (global weight matrix vs. persistent DagNN)
- [ ] Implement shared weight updates across samples
- [ ] Add mini-batch gradient accumulation
- [ ] Implement model checkpoint save/load for ImageClassificationModel
- [ ] Add weight decay / regularization option
- [ ] Update train_mnist.rs to use persistent weights
- [ ] Add training metrics logging (loss, accuracy per epoch)
- [ ] Benchmark training performance
- [ ] Verify >90% accuracy on MNIST test set

## Acceptance Criteria
✅ **Criteria 1:**
- Weights persist across training samples (not discarded per sample)

✅ **Criteria 2:**
- Model achieves >90% accuracy on MNIST test set after training

✅ **Criteria 3:**
- Trained model can be saved to disk and loaded for inference

✅ **Criteria 4:**
- Training loop shows decreasing loss over epochs

## Technical Notes
- **Weight Persistence Options:**
  1. Global weight store indexed by edge type
  2. Persistent "template DagNN" with shared weights copied to per-sample DAGs
  3. Weight matrix separate from graph topology (like classic NNs)

- **Integration with existing code:**
  - ClassificationBrain templates already persist via momentum updates
  - Use similar pattern for DagNN edge weights
  - May need to refactor DagNN to separate topology from weights

- **Generic API (2025-12-10 refactoring):**
  - Use `ImageClassificationModel`, `ImageClassificationConfig` from grapheme-vision
  - Model works with any image size via `RawImage::grayscale(w, h, &pixels)`
  - Works with any number of classes via `ClassificationConfig`

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
- 2025-12-09: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added `FeatureMode` enum: `GridSampling`, `BlobDetection`, `Hybrid`
- Added `grid_size` field to `FeatureConfig` (default: 10 for 10x10 = 100 nodes)
- Implemented `image_to_graph_grid()` for dense, consistent feature extraction
- Implemented `image_to_graph_hybrid()` combining grid + blob features
- Refactored `image_to_graph_blobs()` (original blob detection as separate function)
- Changed default `FeatureMode` to `GridSampling` (was implicit BlobDetection)
- Updated `ImageClassificationConfig::default()` to use `grid_sampling(10)` with `max_vision_nodes: 101`

**Key runtime behavior changes:**
- VisionGraph now produces 101 nodes (1 root + 100 grid cells) by default
- Non-zero activations increased from 4 (2%) to 66 (38%) per image
- Gradient magnitude increased from 1.0 to 2.24
- All 53 grapheme-vision tests + 13 MNIST integration tests pass

### Causality Impact
- `ImageClassificationModel::forward()` and `train_step()` work unchanged (same API)
- VisionBrain produces more nodes → DagNN has more non-zero input activations
- More non-zero activations → better gradient flow through the network
- No async flows or timing changes

### Dependencies & Integration
- No new external dependencies
- `FeatureConfig` API extended with backward-compatible builders:
  - `.grid_sampling(size)` - use grid mode with specified size
  - `.blob_detection()` - use original blob detection mode
  - `.with_mode(FeatureMode)` and `.with_grid_size(size)` for fine control
- Tests expecting blob behavior must now explicitly use `.blob_detection()`

### Verification & Testing
- Run `cargo test --package grapheme-vision` (53 tests pass)
- Run `cargo test --package grapheme-vision test_weight_persistence_across_samples -- --nocapture` to see weight changes
- Run `cargo test --package grapheme-vision test_gradient_magnitude_and_direction -- --nocapture` to see activation coverage

**Known limitations:**
- Grid sampling produces fixed-size graphs regardless of image content
- For very small images (< grid_size), some grid cells may sample same pixels
- Actual MNIST accuracy testing still pending (requires training loop execution)

### Context for Next Task
**For testing-007 (validate >90% accuracy):**
1. The infrastructure is now in place - VisionGraph produces dense, consistent features
2. Weight persistence works - weights change across training samples (verified by test)
3. To test accuracy, need to run actual MNIST training loop with many samples
4. Consider tuning hyperparameters: learning_rate, gradient_weight, hebbian_weight
5. Grid size 10x10 may need tuning - try 7x7 (49 nodes) or 14x14 (196 nodes) for MNIST

**Important decisions:**
- Chose GridSampling as default because blob detection produced too sparse features
- Grid uses 3x3 pixel averaging (`sample_pixel_region`) for robustness
- Hybrid mode adds blobs on top of grid - useful for semantic features + consistent base