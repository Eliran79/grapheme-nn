---
id: backend-138
title: Create ClassificationBrain for output graph to class label conversion
status: done
priority: high
tags:
- backend
- classification
- cognitive-brain
dependencies:
- backend-123
- backend-126
- backend-141
assignee: developer
created: 2025-12-09T19:29:25.550476135Z
estimate: 4h
complexity: 5
area: backend
---

# Create ClassificationBrain for output graph to class label conversion

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
Create ClassificationBrain for converting GRAPHEME output graphs to class labels.
Uses StructuralClassifier (from backend-141) for GRAPHEME-native classification.

## Objectives
- [x] Create ClassificationBrain implementing DomainBrain trait
- [x] Support MNIST (10 classes) and custom classification tasks
- [x] Integrate with StructuralClassifier for template-based matching
- [x] Provide loss_and_gradient() for training integration

## Tasks
- [x] Implement ClassificationConfig struct
- [x] Implement ClassificationOutput struct
- [x] Implement ClassificationBrain with StructuralClassifier
- [x] Implement DomainBrain trait
- [x] Add classify() for inference
- [x] Add loss_and_gradient() for training
- [x] Add 10 unit tests
- [x] Update crate documentation

## Acceptance Criteria
✅ **Criteria 1:**
- ClassificationBrain::mnist() creates 10-class classifier for MNIST

✅ **Criteria 2:**
- classify() returns ClassificationOutput with predicted_class, confidence, probabilities

✅ **Criteria 3:**
- loss_and_gradient() returns structural loss and gradient for training

✅ **Criteria 4:**
- Implements DomainBrain trait with execute() returning predicted class

## Technical Notes
- Uses StructuralClassifier from grapheme-core (added in backend-141)
- Template-based matching: no softmax/cross-entropy
- Confidence = exp(-distance) where distance is from structural matching
- Supports custom labels (e.g., ["cat", "dog", "bird"])

## Testing
- [x] test_classification_config_mnist
- [x] test_classification_config_custom
- [x] test_classification_brain_mnist
- [x] test_classification_brain_with_labels
- [x] test_classification_output_new
- [x] test_classification_output_with_label
- [x] test_classification_brain_classify
- [x] test_classification_brain_domain_brain_trait
- [x] test_classification_brain_classifier_access
- [x] test_classification_brain_execute

## Version Control
- [x] Build passes
- [x] All 20 grapheme-vision tests pass (10 VisionBrain + 10 ClassificationBrain)
- [x] All workspace tests pass

## Updates
- 2025-12-09: Task created
- 2025-12-09: Implementation complete

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-vision/src/lib.rs** (~300 new lines): Added ClassificationBrain
  - `ClassificationConfig`: Configuration (num_classes, num_outputs, momentum, use_structural)
  - `ClassificationOutput`: Result struct (predicted_class, confidence, probabilities, label)
  - `ClassificationBrain`: Main brain implementing DomainBrain trait
  - `classify(&DagNN) → ClassificationOutput`: Inference method
  - `loss_and_gradient(&DagNN, target) → StructuralClassificationResult`: Training method
  - `update_templates(activations, class)`: Template update during training
  - `classifier() / classifier_mut()`: Access to underlying StructuralClassifier

- Updated crate documentation with new pipeline diagram

### Causality Impact
- ClassificationBrain wraps StructuralClassifier from grapheme-core
- classify() extracts output activations via get_classification_logits()
- Confidence is computed as exp(-distance) from structural matching
- Template updates happen via classifier_mut().update_template()

### Dependencies & Integration
- Uses: grapheme_core::StructuralClassifier (from backend-141)
- Uses: DomainBrain trait from grapheme-brain-common
- Exports: `ClassificationConfig`, `ClassificationOutput`, `ClassificationBrain`
- Unblocks: backend-139 (MNIST pipeline needs both VisionBrain and ClassificationBrain)

### Verification & Testing
```bash
# Run classification tests
cargo test -p grapheme-vision test_classification

# Test all grapheme-vision tests
cargo test -p grapheme-vision
```

### Context for Next Task
- **backend-139 (MNIST Pipeline)**: Now has both input (VisionBrain) and output (ClassificationBrain) brains
- Pipeline is: `RawImage → VisionBrain → VisionGraph → to_dagnn() → GRAPHEME Core → ClassificationBrain → class`
- ClassificationBrain.classify() requires DagNN with output nodes (use from_mnist_with_classifier)
- For training, use loss_and_gradient() to get structural loss and gradient
- Templates adapt during training via update_templates() - momentum controls adaptation speed