# GRAPHEME Protocol Migration Guide

**Date**: 2025-12-13
**Status**: ✅ Complete

## Overview

This document describes the migration from legacy neural network defaults to the GRAPHEME Protocol, which enforces:

| Component | Old (Deprecated) | New (GRAPHEME Protocol) |
|-----------|-----------------|-------------------------|
| Activation | `ReLU` | `LeakyReLU` (fixed α=0.01) |
| Weight Init | `Xavier` | `DynamicXavier` |
| Optimizer | `SGD` | `Adam` (lr=0.001) |

## Why This Migration?

### Problem 1: Dead Neurons (ReLU)
Standard ReLU causes "dying neurons" in dynamic graph architectures:
- When a neuron's input becomes negative, gradient = 0
- In dynamic graphs, topology changes can push neurons negative permanently
- Dead neurons never recover → model capacity loss

**Solution**: LeakyReLU with fixed α=0.01 ensures gradients always flow.

### Problem 2: Static Weight Scales (Xavier)
Static Xavier initialization fails for dynamic DAGs:
- Xavier assumes fixed fan-in/fan-out at initialization
- GRAPHEME graphs morph (nodes added/removed, edges change)
- Weight scales become inappropriate after topology changes

**Solution**: DynamicXavier recomputes weight scales when topology changes.

### Problem 3: Slow Convergence (SGD)
SGD struggles with dynamic graph optimization:
- Graph morphing creates non-stationary loss landscapes
- SGD's fixed learning rate can't adapt

**Solution**: Adam with momentum handles dynamic landscapes better.

## Migration Steps (Correct Approach)

### Step 1: Update ActivationType Enum

**WRONG** (old approach - configurable alpha):
```rust
pub enum ActivationType {
    ReLU,
    LeakyReLU(f32),  // Configurable alpha - BAD
}
```

**CORRECT** (new approach - fixed alpha):
```rust
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

pub enum ActivationType {
    #[deprecated(note = "Use LeakyReLU per GRAPHEME protocol")]
    ReLU,
    LeakyReLU,  // Fixed α=0.01 - GOOD
    Tanh,
    Sigmoid,
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::LeakyReLU
    }
}
```

### Step 2: Update InitStrategy Enum

**WRONG** (using Xavier):
```rust
let embedding = Embedding::new(256, 64, InitStrategy::Xavier);
```

**CORRECT** (using DynamicXavier):
```rust
pub enum InitStrategy {
    DynamicXavier,  // Default - recomputes on topology change
    #[deprecated(note = "Use DynamicXavier per GRAPHEME protocol")]
    Xavier,
    He,
    Uniform,
}

impl Default for InitStrategy {
    fn default() -> Self {
        Self::DynamicXavier
    }
}

// Usage
let embedding = Embedding::new(256, 64, InitStrategy::default());
```

### Step 3: Set Adam as Default Optimizer (SGD Deprecated)

```rust
#[deprecated(note = "Use Adam per GRAPHEME protocol")]
pub struct SGD { ... }

/// Adam optimizer (GRAPHEME Protocol default)
pub struct Adam { ... }

impl Default for Adam {
    fn default() -> Self {
        Self::new(0.001)  // lr=0.001, beta1=0.9, beta2=0.999
    }
}

// Usage
let optimizer = Adam::default();
```

## Files Changed

### Core Protocol Files
- `grapheme-train/src/backprop.rs` - ActivationType enum with deprecation
- `grapheme-vision/src/lib.rs` - InitStrategy enum with DynamicXavier
- `grapheme-train/src/optimizer.rs` - Adam::default()

### Updated Usages
- `grapheme-train/src/online_learner.rs` - Xavier → DynamicXavier
- `grapheme-train/src/bin_disabled/train_agi.rs` - Xavier → DynamicXavier
- `grapheme-train/src/bin_disabled/train_unified_agi.rs` - Xavier → DynamicXavier
- `grapheme-vision/src/lib.rs:3023` - Xavier → DynamicXavier

### Documentation Updated
- `CLAUDE.md` - Protocol Summary table
- `tasks/backend/backend-105.md` - Dynamic Xavier notes
- `tasks/backend/backend-106.md` - LeakyReLU defaults
- `tasks/backend/backend-107.md` - Activation function references
- `tasks/backend/backend-114.md` - Time series activations
- `tasks/backend/backend-214.md` - Semantic decoder init
- `tasks/backend/backend-231.md` - LeakyReLU enforcement
- `tasks/backend/backend-232.md` - DynamicXavier implementation
- `tasks/backend/backend-233.md` - Audit completion
- `tasks/backend/backend-234.md` - Deprecation task
- `tasks/api/api-002.md` - Forward pass context

## Common Migration Mistakes

### Mistake 1: Using Configurable LeakyReLU
```rust
// WRONG - alpha should be fixed
ActivationType::LeakyReLU(0.01)

// CORRECT - use the non-parameterized variant
ActivationType::LeakyReLU  // Uses LEAKY_RELU_ALPHA constant
```

### Mistake 2: Forgetting to Allow Deprecated
```rust
// WRONG - will generate compiler warning
let embedding = Embedding::new(256, 64, InitStrategy::Xavier);

// CORRECT - acknowledge deprecation if you must use it
#[allow(deprecated)]
let embedding = Embedding::new(256, 64, InitStrategy::DynamicXavier);
```

### Mistake 3: Not Using Defaults
```rust
// VERBOSE
let opt = Adam::new(0.001);
let init = InitStrategy::DynamicXavier;
let act = ActivationType::LeakyReLU;

// BETTER - use defaults
let opt = Adam::default();
let init = InitStrategy::default();
let act = ActivationType::default();
```

## Verification

After migration, verify with:

```bash
# Check for compile errors
cargo check

# Check for deprecation warnings (should only be in tests)
cargo clippy -- -D warnings

# Run tests
cargo test --lib

# Verify LeakyReLU constant
grep "LEAKY_RELU_ALPHA" grapheme-train/src/backprop.rs
# Should show: pub const LEAKY_RELU_ALPHA: f32 = 0.01;
```

## Protocol Summary

```rust
// GRAPHEME Protocol in one block
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

fn activation(x: f32) -> f32 {
    if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
}

fn dynamic_xavier_scale(fan_in: usize, fan_out: usize) -> f32 {
    (2.0 / (fan_in + fan_out) as f32).sqrt()
}

fn default_optimizer() -> Adam {
    Adam::new(0.001)  // beta1=0.9, beta2=0.999
}
```

## References

- Backend-231: LeakyReLU enforcement
- Backend-232: Dynamic Xavier implementation
- Backend-233: Protocol audit
- Backend-234: Deprecation enforcement
- CLAUDE.md: Protocol summary table
