---
id: backend-140
title: Implement joint training for VisionBrain + ClassificationBrain + GRAPHEME Core
status: todo
priority: critical
tags:
- backend
- training
- joint-learning
- image-classification
dependencies:
- backend-139
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
