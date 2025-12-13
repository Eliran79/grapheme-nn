---
id: backend-141
title: Remove softmax/cross-entropy from MNIST, use graph structure matching
status: done
priority: high
tags:
- backend
- mnist
- structural-loss
dependencies:
- backend-113
assignee: developer
created: 2025-12-09T19:29:38.137694227Z
estimate: 4h
complexity: 6
area: backend
---

# Remove softmax/cross-entropy from MNIST, use graph structure matching

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
GRAPHEME-native classification using graph structure matching instead of softmax/cross-entropy.
Per GRAPHEME Vision: "loss = α·node_insertion_cost + β·edge_deletion_cost + γ·clique_mismatch"

## Objectives
- [x] Replace cross-entropy loss with structural distance loss
- [x] Implement class template graphs (learned activation patterns per class)
- [x] Classification via graph similarity matching
- [x] Provide gradient for backpropagation through structural loss

## Tasks
- [x] Implement ClassTemplate struct for learned activation patterns
- [x] Implement StructuralClassifier with template-based classification
- [x] Add structural_loss() and structural_loss_with_grad() functions
- [x] Add DagNN::structural_classify() and structural_classification_step() methods
- [x] Update train_mnist.rs with --structural flag
- [x] Add 10 unit tests for structural classification
- [x] All tests pass (276 in grapheme-core)

## Acceptance Criteria
✅ **Criteria 1:**
- StructuralClassifier classifies based on nearest template (L2 distance)

✅ **Criteria 2:**
- structural_loss_with_grad() returns differentiable loss and gradient

✅ **Criteria 3:**
- train_mnist --structural uses structural classification instead of cross-entropy

## Technical Notes
- **Loss components**: L2 distance + (1 - cosine similarity) + magnitude difference
- **Template updates**: Exponential moving average with configurable momentum
- **Gradient**: Derived analytically for both L2 and cosine components
- **Initial templates**: One-hot patterns (class i has 1.0 at position i)

## Testing
- [x] test_class_template_new
- [x] test_class_template_distance
- [x] test_class_template_cosine_similarity
- [x] test_structural_classifier_new
- [x] test_structural_classifier_classify
- [x] test_structural_loss
- [x] test_structural_loss_with_grad
- [x] test_structural_classifier_update_template
- [x] test_structural_classifier_all_distances
- [x] test_distance_to_probs

## Version Control
- [x] Build passes
- [x] All 276 grapheme-core tests pass
- [x] All workspace tests pass

## Updates
- 2025-12-09: Task created
- 2025-12-09: Implementation complete

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-core/src/lib.rs**: Added ~350 lines of structural classification code
  - `ClassTemplate` struct: Learned activation pattern per class
  - `StructuralClassifier` struct: Template-based classifier
  - `StructuralLossWeights` struct: Configurable loss component weights
  - `StructuralClassificationResult` struct: Classification result with gradient
  - `DagNN::structural_classify()`: Classify using template matching
  - `DagNN::structural_classification_step()`: Get loss + gradient + prediction

- **grapheme-train/src/bin/train_mnist.rs**: Added structural training mode
  - `--structural` flag: Enable GRAPHEME-native classification
  - `--template_momentum` flag: Control template adaptation rate
  - `train_epoch_structural()`: Training with structural loss
  - `evaluate_structural()`: Evaluation with structural loss

### Causality Impact
- Templates are updated during training (moving average)
- Gradient flows from structural loss → output node activations → edge weights
- Template momentum controls how quickly templates adapt to data

### Dependencies & Integration
- Exports: `StructuralClassifier`, `ClassTemplate`, `StructuralLossWeights`, `StructuralClassificationResult`
- Used by: `train_mnist.rs` (optional --structural mode)
- Unblocks: `backend-138` (ClassificationBrain needs structural matching)

### Verification & Testing
```bash
# Run structural classification tests
cargo test -p grapheme-core test_structural

# Run MNIST with structural mode (requires MNIST data)
cargo run --bin train_mnist -- --structural --epochs 1 --train-samples 100

# Compare with cross-entropy mode
cargo run --bin train_mnist -- --epochs 1 --train-samples 100
```

### Context for Next Task
- **backend-138 (ClassificationBrain)** can now use `StructuralClassifier` for output graph → class conversion
- The structural approach is more aligned with GRAPHEME vision (graph matching, not softmax)
- Current implementation uses activation patterns as "graphs" - true graph structure matching would compare topologies
- Template momentum default is 0.9 (slow adaptation) - may need tuning per task