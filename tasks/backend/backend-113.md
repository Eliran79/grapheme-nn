---
id: backend-113
title: Implement MNIST cognitive module (image classification)
status: todo
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
Implement a cognitive module for image classification (MNIST digits) to demonstrate GRAPHEME's ability to handle non-NLP tasks. This proves the neuromorphic architecture is general-purpose and can learn visual patterns, not just text.

MNIST is the "Hello World" of computer vision: 28×28 grayscale images of handwritten digits (0-9). We'll convert pixels to graph nodes and train the neuromorphic network to classify digits. This tests whether the DAG-based architecture can learn hierarchical visual features (edges → shapes → digits).

## Objectives
- Design pixel-to-graph encoding for 28×28 images
- Implement 10-way classification head (digits 0-9)
- Train neuromorphic GRAPHEME on MNIST dataset
- Achieve >95% accuracy (competitive with simple CNNs)
- Verify end-to-end gradient flow (pixels → graph → logits)
- Demonstrate learnability of all parameters (embeddings, edge weights, activations)

## Tasks
- [ ] Design image-to-graph encoding strategy (grid topology, local connections, DAG structure)
- [ ] Implement `ImageEncoder` trait for 28×28 grayscale images
- [ ] Add classification output layer (10 logits for digits 0-9)
- [ ] Implement softmax + cross-entropy loss for classification
- [ ] Download MNIST dataset (60k train, 10k test)
- [ ] Create `grapheme-train/src/bin/train_mnist.rs` training script
- [ ] Implement data loader for MNIST images
- [ ] Train neuromorphic model on MNIST train set
- [ ] Evaluate accuracy on MNIST test set
- [ ] Verify gradient flow through all components
- [ ] Verify loss decreases and accuracy increases
- [ ] Achieve >95% test accuracy
- [ ] Benchmark training time and memory usage
- [ ] Document image-to-graph encoding design

## Acceptance Criteria
✅ **Image-to-graph encoding works:**
- 28×28 pixels → 784 input nodes in DAG
- Local connectivity (4-neighbors or 8-neighbors)
- DAG topology (e.g., row-by-row or hierarchical)
- Encoding preserves spatial structure

✅ **Classification head works:**
- 10 output nodes (one per digit class)
- Softmax normalization over logits
- Cross-entropy loss for training
- Argmax for prediction

✅ **Training converges:**
- Loss decreases consistently over epochs
- Train accuracy increases to >98%
- Test accuracy reaches >95%
- No gradient vanishing/explosion

✅ **End-to-end learnability:**
- Embeddings learn to represent pixel intensities
- Edge weights learn to detect visual features
- Node activations learn hierarchical patterns
- Output layer learns digit classification

✅ **Performance is acceptable:**
- Training time < 10 minutes on CPU for 10 epochs
- Memory usage < 2GB
- Inference time < 10ms per image

## Technical Notes
### Image-to-Graph Encoding

**Option 1: Grid Topology (Simple)**
```
28×28 pixels → 784 nodes in grid layout
Edges: 4-neighbors (up, down, left, right)
Total edges: ~3000 (28×27×2)
DAG order: row-major (top to bottom, left to right)
```

**Option 2: Hierarchical Encoding (Advanced)**
```
Layer 1: 28×28 = 784 pixels
Layer 2: 14×14 = 196 nodes (2×2 pooling)
Layer 3: 7×7 = 49 nodes (2×2 pooling)
Layer 4: 10 output nodes (classification)
Total nodes: ~1000
Edges: local + pooling connections
DAG order: layer-by-layer
```

**Chosen: Option 1 (Grid Topology)** for simplicity and interpretability.

### Architecture Design

**Input Layer:**
- 784 input nodes (28×28 pixels)
- Node activation = pixel intensity / 255.0 (normalize to [0, 1])
- Node type: `NodeType::Input(pixel_idx)`

**Hidden Layers:**
- Neuromorphic forward pass with edge weights
- Per-node activation functions (ReLU recommended)
- Topological order ensures feedforward computation

**Output Layer:**
- 10 output nodes (one per digit)
- Aggregate activations from hidden layer
- Apply softmax: `p_i = exp(logit_i) / sum(exp(logit_j))`

**Loss Function:**
- Cross-entropy: `L = -log(p_y)` where y is true label
- Backward pass computes gradients w.r.t. all parameters

### Training Pipeline

```rust
// grapheme-train/src/bin/train_mnist.rs

fn main() {
    // Load MNIST dataset
    let train_images = load_mnist_train(); // 60,000 images
    let test_images = load_mnist_test();   // 10,000 images

    // Create neuromorphic model
    let mut model = GraphTransformNet::new(
        vocab_size: 256,  // grayscale intensities
        hidden_dim: 128,
        num_layers: 3,
    );

    // Training loop
    for epoch in 1..=10 {
        for (image, label) in train_images.iter() {
            // Convert image to graph
            let graph = encode_image_to_graph(image);

            // Forward pass
            let logits = model.forward(&graph);

            // Compute loss
            let loss = cross_entropy_loss(logits, label);

            // Backward pass
            model.backward(loss);

            // Update parameters
            model.step(lr);
        }

        // Evaluate on test set
        let accuracy = evaluate(&model, &test_images);
        println!("Epoch {}: Loss={:.4}, Acc={:.2}%", epoch, loss, accuracy * 100.0);
    }
}
```

### MNIST Dataset

**Format:**
- Train: 60,000 images, 28×28 grayscale, labels 0-9
- Test: 10,000 images, 28×28 grayscale, labels 0-9
- Source: http://yann.lecun.com/exdb/mnist/

**Loading:**
- Use `mnist` crate: `cargo add mnist`
- Or download raw files and parse manually

### Expected Results

**Baseline (traditional ML):**
- Logistic regression: ~92% accuracy
- Simple 2-layer MLP: ~96% accuracy
- CNN: ~99% accuracy

**Target for neuromorphic GRAPHEME:**
- >95% accuracy (competitive with simple MLP)
- Demonstrates visual learning capability
- Proves architecture generality (beyond NLP)

### Gradient Flow Verification

**Check:**
1. Embedding gradients (pixel intensity → embedding)
2. Edge weight gradients (visual feature detectors)
3. Node activation gradients (hierarchical patterns)
4. Output layer gradients (classification)

**Diagnostic:**
- Print gradient magnitudes per layer
- Verify gradients are not zero or inf/NaN
- Verify gradients have reasonable scale (10^-3 to 10^-1)

### Dependencies
- Depends on backend-111 (full neuromorphic architecture with backward pass)
- Introduces new modality (images) beyond NLP
- Tests generality of GRAPHEME architecture

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
- 2025-12-08: Task created

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
