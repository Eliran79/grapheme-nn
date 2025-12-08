---
id: backend-115
title: Implement multi-task learning (kindergarten + math)
status: todo
priority: high
tags:
- backend
dependencies:
- backend-111
- backend-113
- backend-114
assignee: developer
created: 2025-12-08T08:51:45.078071013Z
estimate: ~
complexity: 3
area: backend
---

# Implement multi-task learning (kindergarten + math)

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
Implement multi-task learning to verify that a SINGLE neuromorphic GRAPHEME model can be trained on MULTIPLE tasks simultaneously (kindergarten reading + math problems) and save/load trained models. This is critical for AGI - the ability to learn multiple skills and retain knowledge.

Traditional deep learning trains separate models per task. AGI requires multi-task learning where one network handles diverse problems. GRAPHEME's neuromorphic architecture should support this through shared representations (edge weights, node activations) that transfer across tasks.

This task verifies:
1. **Multi-task training**: Train one model on both kindergarten dataset and math dataset
2. **Task switching**: Model can handle different input types in same session
3. **Knowledge retention**: Training on task B doesn't catastrophically forget task A
4. **Model persistence**: Save trained model to disk, load it back, verify performance
5. **End-to-end learnability**: All parameters learn from both tasks

## Objectives
- Train single model on both kindergarten and math datasets
- Implement task-specific output heads (or shared output)
- Verify no catastrophic forgetting (both tasks improve)
- Implement model save/load functionality (serialize to disk)
- Verify loaded model has same performance as pre-save
- Demonstrate knowledge transfer between tasks (if any)
- Achieve good performance on both tasks (not just one)

## Tasks
- [ ] Design multi-task training strategy (alternating batches, mixed batches, or curriculum)
- [ ] Implement task-specific output heads (or unified output)
- [ ] Create `grapheme-train/src/bin/train_multitask.rs` training script
- [ ] Load both kindergarten and math datasets
- [ ] Implement alternating training (kindergarten batch → math batch → repeat)
- [ ] Track per-task loss and accuracy during training
- [ ] Verify both tasks improve (no catastrophic forgetting)
- [ ] Implement model serialization (save to .bin or .json)
- [ ] Implement model deserialization (load from disk)
- [ ] Verify loaded model performance matches pre-save performance
- [ ] Test model on held-out test sets for both tasks
- [ ] Measure catastrophic forgetting (task A performance after training task B)
- [ ] Document multi-task learning strategy

## Acceptance Criteria
✅ **Multi-task training works:**
- Single model trains on both kindergarten and math
- Both task losses decrease over training
- Both task accuracies increase over training
- Training is stable (no divergence or collapse)

✅ **No catastrophic forgetting:**
- After training task B, task A performance ≥ 90% of peak
- Model maintains competence on both tasks
- Shared representations benefit both tasks (or at least don't hurt)

✅ **Model persistence works:**
- Model saves to disk (file size reasonable, < 100MB)
- Model loads from disk without errors
- Loaded model has same performance (within 1% accuracy)
- Save/load is deterministic (same weights)

✅ **Task switching works:**
- Model can switch between tasks in same session
- No need to reload model for different tasks
- Inference works for both kindergarten and math inputs

✅ **Performance is good on both tasks:**
- Kindergarten accuracy > 80%
- Math accuracy > 70%
- Performance comparable to single-task models

## Technical Notes
### Multi-Task Learning Strategies

**Option 1: Alternating Batches (Simple)**
```rust
for epoch in 1..=100 {
    // Train on kindergarten batch
    let loss_k = train_batch(&model, &kindergarten_data);

    // Train on math batch
    let loss_m = train_batch(&model, &math_data);

    println!("Epoch {}: K_loss={:.4}, M_loss={:.4}", epoch, loss_k, loss_m);
}
```
- Pros: Simple, both tasks get equal training
- Cons: May have task interference

**Option 2: Mixed Batches**
```rust
// Create unified dataset with task labels
let mixed_data = interleave(kindergarten_data, math_data);

for (input, task_label) in mixed_data {
    let loss = train_example(&model, input, task_label);
}
```
- Pros: Model learns to handle task variety
- Cons: More complex implementation

**Option 3: Curriculum Learning**
```rust
// Phase 1: Pretrain on kindergarten (easier task)
for epoch in 1..=50 {
    train(&model, &kindergarten_data);
}

// Phase 2: Fine-tune on math (harder task)
for epoch in 1..=50 {
    train(&model, &math_data);
}

// Phase 3: Mixed training
for epoch in 1..=50 {
    train(&model, &interleaved_data);
}
```
- Pros: Easier task first, builds foundation
- Cons: More training time

**Chosen: Option 1 (Alternating Batches)** for simplicity and fair comparison.

### Output Head Design

**Option A: Shared Output Head**
```rust
// Single output layer handles both tasks
// Kindergarten: 26 outputs (letters A-Z)
// Math: Variable outputs (e.g., digits 0-9 or equation results)
// Problem: Different output spaces
```

**Option B: Task-Specific Output Heads**
```rust
pub struct MultiTaskModel {
    shared_backbone: GraphTransformNet,  // Shared parameters
    kindergarten_head: OutputLayer,      // 26 outputs (letters)
    math_head: OutputLayer,              // 10 outputs (digits)
}

fn forward(&self, input, task: Task) -> Output {
    let features = self.shared_backbone.forward(input);
    match task {
        Task::Kindergarten => self.kindergarten_head.forward(features),
        Task::Math => self.math_head.forward(features),
    }
}
```
- Pros: Clean separation, appropriate output for each task
- Cons: More parameters (but minimal - just output layers)

**Chosen: Option B (Task-Specific Heads)** - cleaner and more flexible.

### Model Serialization

**Format Options:**
1. **Bincode** (Rust-native binary serialization)
   - Pros: Fast, compact
   - Cons: Not human-readable
2. **JSON** (serde_json)
   - Pros: Human-readable, debugging
   - Cons: Larger file size
3. **MessagePack** (rmp-serde)
   - Pros: Compact, fast, portable
   - Cons: Less common

**Chosen: Bincode** for production, JSON for debugging.

**Implementation:**
```rust
// Save model
pub fn save_model(model: &GraphTransformNet, path: &str) -> Result<()> {
    let bytes = bincode::serialize(model)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

// Load model
pub fn load_model(path: &str) -> Result<GraphTransformNet> {
    let bytes = std::fs::read(path)?;
    let model = bincode::deserialize(&bytes)?;
    Ok(model)
}
```

### Catastrophic Forgetting Measurement

**Metric:**
```
Forgetting = max(0, (Acc_A_peak - Acc_A_after_B) / Acc_A_peak)
```

**Acceptable:**
- Forgetting < 10% (Acc_A_after_B ≥ 90% of Acc_A_peak)

**Mitigation:**
- Alternating training (not sequential)
- Elastic weight consolidation (EWC) - advanced
- Rehearsal (replay old examples) - simple

### Training Pipeline

```rust
// grapheme-train/src/bin/train_multitask.rs

fn main() {
    // Load datasets
    let kindergarten_train = load_kindergarten_train();
    let math_train = load_math_train();
    let kindergarten_test = load_kindergarten_test();
    let math_test = load_math_test();

    // Create shared model with task-specific heads
    let mut model = MultiTaskModel::new(hidden_dim: 128, num_layers: 3);

    // Training loop
    for epoch in 1..=100 {
        // Alternate between tasks
        let loss_k = train_task(&mut model, &kindergarten_train, Task::Kindergarten);
        let loss_m = train_task(&mut model, &math_train, Task::Math);

        // Evaluate both tasks
        if epoch % 10 == 0 {
            let acc_k = evaluate(&model, &kindergarten_test, Task::Kindergarten);
            let acc_m = evaluate(&model, &math_test, Task::Math);
            println!("Epoch {}: K_loss={:.4} K_acc={:.2}% M_loss={:.4} M_acc={:.2}%",
                     epoch, loss_k, acc_k * 100.0, loss_m, acc_m * 100.0);
        }
    }

    // Save model
    save_model(&model, "multitask_model.bin")?;

    // Load model and verify
    let loaded_model = load_model("multitask_model.bin")?;
    let acc_k_loaded = evaluate(&loaded_model, &kindergarten_test, Task::Kindergarten);
    let acc_m_loaded = evaluate(&loaded_model, &math_test, Task::Math);

    println!("Loaded model: K_acc={:.2}% M_acc={:.2}%",
             acc_k_loaded * 100.0, acc_m_loaded * 100.0);

    // Verify performance matches
    assert!((acc_k_loaded - acc_k).abs() < 0.01, "Kindergarten accuracy mismatch!");
    assert!((acc_m_loaded - acc_m).abs() < 0.01, "Math accuracy mismatch!");
}
```

### Expected Results

**Single-task baselines (from previous tasks):**
- Kindergarten: ~90% accuracy
- Math: ~80% accuracy

**Multi-task target:**
- Kindergarten: >80% accuracy (slight drop acceptable)
- Math: >70% accuracy (slight drop acceptable)
- Forgetting: <10% for both tasks
- Training time: ~2x single-task time (trains both)

### Knowledge Transfer Analysis

**Positive transfer:**
- Shared character recognition (letters, digits)
- Shared sequence processing (reading, equation parsing)
- Shared embeddings improve sample efficiency

**Negative transfer (interference):**
- Conflicting patterns (kindergarten vs math syntax)
- Different loss scales (may need task weighting)

**Measurement:**
- Compare multi-task accuracy vs single-task accuracy
- If multi-task > single-task, positive transfer detected!

### Dependencies
- Depends on backend-111 (neuromorphic backward pass)
- Optionally depends on backend-113 (MNIST) and backend-114 (time series) for more diverse multi-task learning
- Tests model persistence and knowledge retention
- Demonstrates AGI-ready multi-task capability

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
