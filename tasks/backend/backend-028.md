---
id: backend-028
title: Implement gradient descent training loop
status: done
priority: high
tags:
- backend
dependencies:
- backend-026
- backend-027
assignee: developer
created: 2025-12-06T08:41:15.881586058Z
estimate: ~
complexity: 3
area: backend
---

# Implement gradient descent training loop

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
The training loop orchestrates: forward pass → loss computation (GED) → backward pass → weight update. Uses existing grapheme-train infrastructure.

## Objectives
- Create main training loop with GED loss
- Implement SGD/Adam optimizers
- Add learning rate scheduling
- Support curriculum learning (levels 1-7)

## Tasks
- [x] Implement `Optimizer` trait (step, zero_grad)
- [x] Implement SGD optimizer with momentum
- [x] Implement Adam optimizer
- [x] Create training loop: batch → forward → loss → backward → step
- [x] Add learning rate scheduler (step, cosine, warmup)
- [x] Integrate with DataGenerator from grapheme-train
- [x] Add training metrics logging

## Acceptance Criteria
✅ **Training Loop:**
- Loss decreases over epochs
- Weights are updated correctly

✅ **Curriculum:**
- Can train on levels 1-7 progressively
- Supports mixed-level batches

## Technical Notes
- Use existing `GraphEditDistance` for loss
- Leverage `Dataset` and `BatchIterator` from grapheme-train
- Consider `TrainingConfig` for hyperparameters
- Add checkpoint save/load functionality

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
- 2025-12-06: Task created
- 2025-12-06: Task completed - Training loop with optimizers and schedulers

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added `Optimizer` trait with step, zero_grad, get_lr, set_lr methods
- Added `SGD` struct with momentum and weight decay support
- Added `Adam` struct with bias correction and AdamW-style weight decay
- Added `LRScheduler` enum: Constant, StepLR, ExponentialLR, CosineAnnealingLR, WarmupLR, WarmupCosineDecay
- Added `TrainingState` and `TrainingMetrics` for tracking training progress
- Added `TrainingLoop` struct for orchestrating training
- Added `compute_ged_loss` function for loss computation
- Added ndarray dependency to grapheme-train
- Added 19 new tests for optimizers, schedulers, and training loop

### Key APIs
```rust
// Optimizers
let mut sgd = SGD::new(0.01).with_momentum(0.9).with_weight_decay(1e-4);
let mut adam = Adam::new(0.001).with_beta1(0.9).with_beta2(0.999);
optimizer.step(&mut weights, &gradients);

// Learning rate schedulers
let scheduler = LRScheduler::CosineAnnealingLR { t_max: 100, eta_min: 1e-5 };
let lr = scheduler.get_lr(base_lr, epoch);

// Training loop
let mut loop_ = TrainingLoop::new(config).with_scheduler(scheduler);
loop_.record_batch(loss);
let avg_loss = loop_.complete_epoch();
let is_best = loop_.record_validation(val_loss, val_acc);
if loop_.should_stop() { break; }  // Early stopping
```

### Causality Impact
- TrainingLoop.record_batch() accumulates loss for epoch averaging
- TrainingLoop.complete_epoch() resets counters, updates LR, records metrics
- Validation tracking enables early stopping via epochs_without_improvement

### Dependencies & Integration
- Added ndarray = "0.15" to grapheme-train/Cargo.toml
- Integrates with existing TrainingConfig, Dataset, BatchIterator
- compute_ged_loss uses GraphEditDistance for loss computation

### Verification & Testing
- 19 new tests: test_sgd_*, test_adam_*, test_lr_scheduler_*, test_training_*
- Run: `cargo test -p grapheme-train`
- 55 tests in grapheme-train, 361 total across workspace

### Context for Next Task
- For backend-029 (learnable graph transform):
  - Use Optimizer trait to update transformation weights
  - Use TrainingLoop to orchestrate training
  - Use compute_ged_loss for loss between predicted and target graphs
- SGD with momentum is often faster for graph neural networks
- Adam with warmup (WarmupCosineDecay) is recommended for transformers
- Early stopping patience is configurable in TrainingConfig