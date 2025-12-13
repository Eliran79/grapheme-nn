---
id: testing-007
title: Validate image classification accuracy with GRAPHEME-native architecture (target >90%)
status: done
priority: high
tags:
- testing
- image-classification
- validation
- benchmark
dependencies:
- backend-140
assignee: developer
created: 2025-12-09T19:29:43.011165695Z
estimate: 4h
complexity: 5
area: testing
---

# Validate image classification accuracy with GRAPHEME-native architecture (target >90%)

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
Validate that the GRAPHEME-native image classification pipeline (using `ImageClassificationModel`
from grapheme-vision) achieves competitive accuracy on standard benchmarks. This validates
the vision that graph-based learning can match or exceed traditional CNNs.

**Pipeline being validated:**
```
RawImage → VisionBrain.to_graph() → DagNN → ClassificationBrain → class label
```

**Generic API (2025-12-10):**
- Uses `ImageClassificationModel`, `ImageClassificationConfig` from grapheme-vision
- Works with any image size via `RawImage::grayscale(w, h, &pixels)`
- Works with any number of classes via `ClassificationConfig`

## Objectives
- [ ] Achieve >90% accuracy on MNIST test set
- [ ] Document training configuration that achieves target accuracy
- [ ] Benchmark training time vs accuracy tradeoff
- [ ] Compare with baseline CNN accuracy (~98-99%)

## Tasks
- [ ] Train ImageClassificationModel on full MNIST training set (60K samples)
- [ ] Evaluate on full MNIST test set (10K samples)
- [ ] Record accuracy, loss, and training time
- [ ] Tune hyperparameters (hidden_size, learning_rate, epochs) if needed
- [ ] Document best configuration
- [ ] Add benchmark script for reproducible validation

## Acceptance Criteria
✅ **Criteria 1:**
- ImageClassificationModel achieves >90% accuracy on MNIST test set

✅ **Criteria 2:**
- Training configuration is documented and reproducible

✅ **Criteria 3:**
- Validation script can be run to verify results

## Technical Notes
- Depends on backend-140 for weight persistence (without it, ~10-20% accuracy)
- Use `--vision` flag with train_mnist binary
- MNIST data must be downloaded to data/mnist/ directory
- Consider using structural loss (`--structural`) vs cross-entropy
- Graph structure matching should enable learning without softmax/cross-entropy

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