---
id: backend-113
title: Implement MNIST cognitive module (image classification)
status: done
priority: high
tags:
- backend
dependencies:
- backend-111
assignee: developer
created: 2025-12-08T08:51:45.073854808Z
estimate: ~
complexity: 3
area: backend
---

# Implement MNIST cognitive module (image classification)

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
Implemented a cognitive module for image classification (MNIST digits) to demonstrate GRAPHEME's ability to handle non-NLP tasks. The infrastructure is complete and verified working.

## Objectives
- [x] Design pixel-to-graph encoding for 28×28 images
- [x] Implement 10-way classification head (digits 0-9)
- [x] Create training infrastructure for MNIST
- [x] Verify end-to-end gradient flow (pixels → graph → logits)

## Tasks
- [x] Design image-to-graph encoding strategy (grid topology)
- [x] Add `NodeType::Pixel` and `NodeType::ClassOutput` variants
- [x] Implement `Node::pixel()` and `Node::class_output()` constructors
- [x] Implement `DagNN::from_image()` for arbitrary image dimensions
- [x] Implement `DagNN::from_mnist()` for 28×28 images
- [x] Implement `DagNN::from_mnist_with_classifier()` with hidden layer
- [x] Add `get_classification_logits()` and `predict_class()` methods
- [x] Implement `softmax()` function with log-sum-exp trick
- [x] Implement `cross_entropy_loss()` for classification
- [x] Implement `cross_entropy_loss_with_grad()` with gradient
- [x] Add `compute_accuracy()` helper function
- [x] Create `ClassificationConfig` and `ClassificationStepResult` structs
- [x] Add MNIST and indicatif dependencies to grapheme-train
- [x] Create `train_mnist.rs` training binary
- [x] Add 19 new tests for image encoding and classification
- [x] Fix all compiler warnings
- [x] Verify training runs end-to-end

## Acceptance Criteria
✅ **Image-to-graph encoding works:**
- 28×28 pixels → 784 input nodes in DAG
- Grid topology with 4-neighbor connectivity (right and down edges for DAG)
- Row-major ordering preserves spatial structure
- Pixel intensities normalized to [0.0, 1.0]

✅ **Classification head works:**
- 10 output nodes (one per digit class)
- Softmax normalization over logits (numerically stable)
- Cross-entropy loss for training
- Argmax for prediction via `predict_class()`

✅ **Training infrastructure ready:**
- MNIST data loading via `mnist` crate
- Progress bars via `indicatif`
- Configurable epochs, batch size, learning rate, hidden size
- Optional Hebbian learning integration
- Verified working with real MNIST data

✅ **All tests pass:**
- 266 tests pass in grapheme-core (19 new)
- No compiler warnings
- High code quality maintained

## Technical Notes
### New NodeType Variants (lib.rs)

```rust
pub enum NodeType {
    // ... existing variants ...
    Pixel { row: usize, col: usize },
    ClassOutput(usize),
}
```

### Image Encoding API (Generic - Refactored 2025-12-10)

```rust
// Generic image encoding (any dimensions) - grapheme-core
let dag = DagNN::from_image(&pixels, width, height)?;

// Generic classifier (input → hidden → N outputs) - grapheme-core
let dag = DagNN::with_classifier(num_inputs, hidden_size, num_classes, Some(&activations))?;

// Classification
dag.neuromorphic_forward()?;
let logits = dag.get_classification_logits(); // Vec<f32> len num_classes
let predicted = dag.predict_class(); // 0..num_classes-1

// For image classification use grapheme-vision ImageClassificationModel:
let config = ImageClassificationConfig::default();
let model = ImageClassificationModel::new(config);
let image = RawImage::grayscale(28, 28, &pixels)?;
let result = model.forward(&image);
```

**Note:** MNIST-specific methods (`from_mnist()`, `from_mnist_with_classifier()`) were
removed in refactoring (2025-12-10). Use generic `with_classifier()` and
`ImageClassificationModel` from grapheme-vision instead.

### Training Command

```bash
cargo run --release --bin train_mnist -- \
  --data-dir ./data/mnist \
  --epochs 2 \
  --train-samples 1000 \
  --test-samples 200 \
  --hidden-size 128 \
  --hebbian
```

### Known Limitation
Current architecture creates fresh DAG per image with random weights. No weight sharing between samples limits cross-sample learning. Future work could add:
- Shared weight matrix separate from graph topology
- Weight accumulation across batches
- Persistent model state

## Testing
- [x] 19 new tests added for image encoding and classification
- [x] All 266 tests pass
- [x] Training verified working with real MNIST data

## Version Control
- [x] All changes committed
- [x] Zero compiler warnings
- [x] High code quality maintained

## Updates
- 2025-12-08: Task created
- 2025-12-09: Implementation completed and verified
- 2025-12-10: **Refactored**: Removed MNIST-specific methods from grapheme-core. Now uses generic `with_classifier()` and `ImageClassificationModel` from grapheme-vision

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed

**grapheme-core/src/lib.rs:**
- Added `NodeType::Pixel { row, col }` and `NodeType::ClassOutput(usize)` variants
- Added `GraphemeError::DimensionMismatch` error variant
- Added `Node::pixel()` and `Node::class_output()` constructors
- Added `DagNN::from_image()`, `DagNN::with_classifier()` (MNIST-specific methods removed 2025-12-10)
- Added `DagNN::get_classification_logits()` and `DagNN::predict_class()`
- Added `softmax()`, `cross_entropy_loss()`, `cross_entropy_loss_with_grad()`
- Added `compute_accuracy()` helper
- Added `ClassificationConfig` and `ClassificationStepResult` structs
- Updated `Embedding::embed_node()` to handle Pixel and ClassOutput types
- Fixed snake_case warnings throughout codebase
- Added 19 new tests

**grapheme-train/Cargo.toml:**
- Added `mnist = "0.6"` dependency
- Added `indicatif = "0.17"` for progress bars
- Added `rand.workspace = true`
- Added `[[bin]] name = "train_mnist"`

**grapheme-train/src/bin/train_mnist.rs:**
- New binary for MNIST training
- CLI arguments for data-dir, epochs, batch-size, learning-rate, hidden-size, hebbian
- Automatic MNIST loading (download manually from Google storage if needed)
- Training loop with progress bars
- Evaluation on test set

**grapheme-train/src/lib.rs:**
- Updated `hash_node_type()` to handle Pixel and ClassOutput

### Causality Impact
- Image → DAG conversion creates grid topology
- Forward pass propagates pixel activations through grid edges to hidden nodes to outputs
- Classification uses softmax over output node activations
- Gradient flows backward through cross-entropy → softmax → output → hidden → input

### Dependencies & Integration
- New dependencies: `mnist`, `indicatif`
- Integrates with existing `HebbianLearning` trait for hybrid learning
- Uses existing `neuromorphic_forward()` for inference
- Compatible with all existing DagNN methods

### Verification & Testing
- Run `cargo test --package grapheme-core` - 266 tests should pass
- Run `cargo build --package grapheme-train --bin train_mnist` - should compile with no warnings
- Download MNIST from https://storage.googleapis.com/cvdf-datasets/mnist/ to data/mnist/
- Run `cargo run --release --bin train_mnist -- --epochs 1 --train-samples 500 --test-samples 100`

### Context for Next Task
- Grid topology uses 4-neighbor connectivity (right + down only for DAG property)
- Hidden layer is fully connected to subset of inputs (locality for efficiency)
- Output layer is fully connected to all hidden nodes
- Softmax uses log-sum-exp trick for numerical stability
- Cross-entropy gradient is `prob - one_hot(target)` (elegant formula)
- Training script creates new DAG per sample (stateless - no weight sharing)
- For better accuracy, would need shared weight matrix approach
