---
id: backend-111
title: Implement backward pass with Hebbian + gradient descent
status: done
priority: critical
tags:
- backend
dependencies:
- backend-107
- backend-108
- backend-110
assignee: developer
created: 2025-12-08T08:38:38.947859906Z
estimate: ~
complexity: 3
area: backend
---

# Implement backward pass with Hebbian + gradient descent

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
Implement a biologically-inspired backward pass that combines traditional gradient descent with Hebbian learning rules. This enables the neural network to learn through both error-driven (supervised) and correlation-driven (unsupervised) mechanisms.

## Objectives
- [x] Implement multiple Hebbian learning rules (Classic, Oja, BCM, Anti-Hebbian)
- [x] Create hybrid learning combining gradient descent + Hebbian
- [x] Add competitive learning (lateral inhibition)
- [x] Ensure weight bounds and stability mechanisms

## Tasks
- [x] Create HebbianConfig struct with learning parameters
- [x] Implement HebbianRule enum (Classic, Oja, BCM, AntiHebbian)
- [x] Create HybridLearningConfig for combined learning
- [x] Implement HebbianLearning trait with backward_hebbian()
- [x] Implement backward_hybrid() combining gradient + Hebbian
- [x] Implement compute_hebbian_delta() for individual edges
- [x] Implement apply_competitive_learning() for lateral inhibition
- [x] Add HebbianResult and HybridResult structs for diagnostics
- [x] Write 23 comprehensive tests

## Acceptance Criteria
✅ **Criteria 1:**
- All four Hebbian rules implemented and tested (Classic, Oja, BCM, AntiHebbian)

✅ **Criteria 2:**
- Hybrid learning combines gradient descent and Hebbian updates correctly

✅ **Criteria 3:**
- Weight bounds, decay, and stability mechanisms work correctly

✅ **Criteria 4:**
- All 237 tests pass in grapheme-core

## Technical Notes

### Hebbian Rules Implemented:

1. **Classic Hebbian**: `Δw = η * pre * post`
   - "Neurons that fire together wire together"
   - Simple correlation-based learning

2. **Oja's Rule**: `Δw = η * post * (pre - w * post)`
   - Adds automatic weight normalization
   - Prevents unbounded weight growth

3. **BCM Rule**: `Δw = η * pre * post * (post - θ)`
   - Bidirectional plasticity with sliding threshold
   - When post > θ: Long-Term Potentiation (strengthening)
   - When post < θ: Long-Term Depression (weakening)

4. **Anti-Hebbian**: `Δw = -η * pre * post`
   - Used for decorrelation and competitive learning
   - Weakens connections between co-active neurons

### Key Structures:

```rust
pub struct HebbianConfig {
    pub learning_rate: f32,      // η
    pub weight_decay: f32,       // Regularization
    pub max_weight: f32,         // Upper bound
    pub min_weight: f32,         // Pruning threshold
    pub rule: HebbianRule,       // Learning rule variant
    pub bcm_threshold: f32,      // θ for BCM
}

pub struct HybridLearningConfig {
    pub gradient_lr: f32,        // Gradient descent LR
    pub hebbian: HebbianConfig,  // Hebbian config
    pub gradient_weight: f32,    // Gradient contribution (0-1)
    pub hebbian_weight: f32,     // Hebbian contribution (0-1)
    pub clip_gradients: bool,    // Gradient clipping
    pub max_grad_norm: f32,      // Max gradient norm
}
```

## Testing
- [x] Write unit tests for all Hebbian rule computations
- [x] Write tests for config builders
- [x] Write tests for backward_hebbian() updates
- [x] Write tests for backward_hybrid() combined updates
- [x] Write tests for competitive learning
- [x] Write integration tests for full training steps
- [x] All 237 tests pass

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
- [x] All tests pass (237 passed, 0 failed)
- [x] No compilation errors

## Updates
- 2025-12-08: Task created
- 2025-12-09: Implementation completed

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added to `grapheme-core/src/lib.rs`:
  - `HebbianConfig` struct (lines ~5652-5722)
  - `HebbianRule` enum (lines ~5724-5738)
  - `HybridLearningConfig` struct (lines ~5740-5786)
  - `HebbianLearning` trait (lines ~5788-5823)
  - `HebbianResult` and `HybridResult` structs (lines ~5825-5847)
  - `impl HebbianLearning for DagNN` (lines ~5849-6038)
  - 23 new tests for Hebbian learning (lines ~11271-11741)

### New Public API:
```rust
// Pure Hebbian learning
let config = HebbianConfig::new(0.01)
    .with_oja_rule()
    .with_weight_decay(0.001);
let result = dag.backward_hebbian(&config);

// Hybrid gradient + Hebbian
let config = HybridLearningConfig::new(0.7, 0.3)
    .with_learning_rates(0.001, 0.01);
let result = dag.backward_hybrid(&output_grad, &mut embedding, &config);

// Competitive learning
dag.apply_competitive_learning(0.5);
```

### Causality Impact
- Hebbian learning updates weights based on pre/post activations after forward pass
- Forward pass must be run first to set activation values
- Weight updates are synchronous (all edges updated in one call)
- Competitive learning modifies activations (not weights)

### Dependencies & Integration
- Integrates with existing `BackwardPass` trait infrastructure
- Uses existing `NodeGradients` for gradient-based updates
- Works with the neuromorphic forward pass from backend-107
- Complements edge pruning (backend-108) and neurogenesis (backend-110)

### Verification & Testing
- Run `cargo test --package grapheme-core` - should show 237 tests passing
- Test individual Hebbian rules with known inputs to verify formulas
- Test hybrid learning by varying gradient_weight/hebbian_weight ratios

### Context for Next Task
- The backward pass now supports three modes:
  1. Pure gradient descent (`backward_and_update`)
  2. Pure Hebbian learning (`backward_hebbian`)
  3. Hybrid learning (`backward_hybrid`)
- Default hybrid config uses 70% gradient, 30% Hebbian
- Edges with zero delta are skipped entirely (including decay)
- Weight bounds prevent unbounded growth (default max: 10.0)
- min_weight > 0 enables automatic pruning of weak connections