---
id: backend-139
title: 'Implement ImageClassificationModel: VisionBrain + ClassificationBrain + GRAPHEME Core pipeline'
status: done
priority: critical
tags:
- backend
- mnist
- end-to-end
- model
dependencies:
- backend-137
- backend-138
assignee: developer
created: 2025-12-09T19:29:29.370093946Z
estimate: 6h
complexity: 8
area: backend
---

# Implement ImageClassificationModel: VisionBrain + ClassificationBrain + GRAPHEME Core pipeline

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
Create the complete image classification pipeline combining VisionBrain (image-to-graph),
GRAPHEME Core (graph processing), and ClassificationBrain (graph-to-class) into a
unified ImageClassificationModel that can be trained and evaluated on any image dataset.

**Note (2025-12-10):** Types renamed from MNIST-specific to generic:
- `MnistModel` → `ImageClassificationModel`
- `MnistModelConfig` → `ImageClassificationConfig`
- `MnistForwardResult` → `ForwardResult`
- `MnistTrainResult` → `TrainResult`

## Objectives
- [x] Create ImageClassificationModel combining all components
- [x] Implement forward pass through full pipeline
- [x] Implement train_step with loss and gradient computation
- [x] Add --vision flag to train_mnist binary
- [x] Test pipeline on real image data (MNIST used for verification)

## Tasks
- [x] Create ImageClassificationConfig struct (was MnistModelConfig)
- [x] Create ForwardResult struct (was MnistForwardResult)
- [x] Create TrainResult struct (was MnistTrainResult)
- [x] Implement ImageClassificationModel with forward() method
- [x] Implement train_step() returning loss and gradient
- [x] Add unit tests (11 tests)
- [x] Add integration tests on real MNIST (5 tests)
- [x] Update train_mnist.rs with --vision mode
- [x] Verify all modes work (cross-entropy, structural, vision)

## Acceptance Criteria
✅ **Criteria 1:**
- ImageClassificationModel::forward() runs complete pipeline: RawImage → VisionBrain → DagNN → ClassificationBrain → class

✅ **Criteria 2:**
- train_step() returns structural loss and gradient for backpropagation

✅ **Criteria 3:**
- train_mnist --vision runs end-to-end training with VisionBrain pipeline

✅ **Criteria 4:**
- All 58 grapheme-vision tests pass (44 unit + 13 integration + 1 doc)

## Technical Notes
- ImageClassificationModel creates fresh DagNN per sample (stateless inference)
- Weight persistence for training requires backend-140 (joint training)
- VisionBrain extracts blob features deterministically (any image size)
- ClassificationBrain uses structural template matching (any number of classes)
- Pipeline is: RawImage → VisionGraph → DagNN → forward → classify
- **Generic API**: Works with any image dimensions and any number of output classes

## Testing
- [x] test_mnist_model_config_default
- [x] test_mnist_model_config_builder
- [x] test_mnist_model_new
- [x] test_mnist_model_forward_blank
- [x] test_mnist_model_forward_with_digit
- [x] test_mnist_model_forward_with_target
- [x] test_mnist_model_train_step
- [x] test_mnist_model_deterministic
- [x] test_mnist_model_vision_access
- [x] test_mnist_model_classification_access
- [x] test_mnist_model_debug
- [x] test_mnist_model_forward_on_real_data (integration)
- [x] test_mnist_model_train_step_on_real_data (integration)
- [x] test_mnist_model_determinism_on_real_data (integration)
- [x] test_mnist_model_all_digit_classes (integration)
- [x] test_mnist_model_custom_config (integration)

## Version Control
- [x] Build passes (release mode)
- [x] All 58 grapheme-vision tests pass
- [x] train_mnist --vision runs successfully

## Updates
- 2025-12-09: Task created
- 2025-12-09: Implementation complete
- 2025-12-10: **Refactored**: Renamed types to generic names (MnistModel→ImageClassificationModel, etc.). API now works with any image size and number of classes

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-vision/src/lib.rs** (~300 new lines):
  - `ImageClassificationConfig` (was MnistModelConfig): Configuration (vision, classification, hidden_size)
  - `ForwardResult` (was MnistForwardResult): Forward pass result (class, confidence, vision stats)
  - `TrainResult` (was MnistTrainResult): Training result (loss, gradient, correct)
  - `ImageClassificationModel` (was MnistModel): Complete end-to-end pipeline
  - `forward(&RawImage)`: RawImage → VisionBrain → DagNN → ClassificationBrain → class
  - `train_step(&RawImage, label)`: Forward + compute structural loss + get gradient

- **grapheme-vision/tests/mnist_integration.rs** (+175 lines):
  - 5 new MnistModel integration tests on real MNIST data

- **grapheme-train/src/bin/train_mnist.rs** (~160 new lines):
  - Added `--vision` flag for VisionBrain pipeline mode
  - `train_epoch_vision()`: Training loop using MnistModel
  - `evaluate_vision()`: Evaluation using MnistModel

- **grapheme-train/Cargo.toml**: Added grapheme-vision dependency

### Causality Impact
- ImageClassificationModel creates FRESH DagNN for each sample
- Gradient updates are computed but applied to temporary DAG
- For high accuracy (>90%), need backend-140 to persist weights between samples
- Vision graph extraction IS deterministic (same image = same graph)
- Template updates DO persist in ClassificationBrain across samples
- **Generic**: Works with any image size via RawImage::grayscale(w, h, &pixels) or RawImage::rgb(w, h, &pixels)

### Dependencies & Integration
- Uses: VisionBrain, ClassificationBrain from grapheme-vision
- Uses: DagNN, StructuralClassifier from grapheme-core
- Exports: ImageClassificationModel, ImageClassificationConfig, ForwardResult, TrainResult, ModelStats
- Unblocks: backend-140 (joint training with weight persistence)

### Verification & Testing
```bash
# Run all grapheme-vision tests (58 total)
cargo test -p grapheme-vision

# Run MnistModel tests specifically
cargo test -p grapheme-vision test_mnist_model

# Test train_mnist with vision mode
cargo run --release --bin train_mnist -- --vision --epochs 1 --train-samples 100 --test-samples 20

# Test all three modes
cargo run --release --bin train_mnist -- --epochs 1 --train-samples 100 --test-samples 20
cargo run --release --bin train_mnist -- --structural --epochs 1 --train-samples 100 --test-samples 20
cargo run --release --bin train_mnist -- --vision --epochs 1 --train-samples 100 --test-samples 20
```

### Context for Next Task
- **backend-140 (Joint Training)** needs to implement weight persistence:
  - Current: Each sample creates new DagNN, gradients applied to temp graph
  - Needed: Shared weight matrix that persists across samples
  - Options: (1) Global weight store, (2) Persistent DagNN with weight sharing
  - ClassificationBrain templates DO persist - use similar pattern for DagNN weights

- **Current accuracy limitation**: ~10-20% because weights don't persist
- **Expected with backend-140**: Should reach 90%+ with proper weight persistence
- **Pipeline is correct**: The forward pass, loss computation, and gradient calculation all work
- **Determinism verified**: Same image always produces same vision graph
