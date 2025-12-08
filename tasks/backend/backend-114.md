---
id: backend-114
title: Implement time series cognitive module (forecasting)
status: todo
priority: high
tags:
- backend
dependencies:
- backend-111
assignee: developer
created: 2025-12-08T08:51:45.076052926Z
estimate: ~
complexity: 3
area: backend
---

# Implement time series cognitive module (forecasting)

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
Implement a cognitive module for time series forecasting to demonstrate GRAPHEME's ability to model temporal dependencies and make predictions. This is crucial for AGI-ready systems that need to understand sequences, trends, and temporal causality.

Time series problems (stock prices, weather, sensor data) require modeling dependencies over time. Unlike NLP (discrete tokens) or vision (spatial patterns), time series has continuous values and temporal order. The DAG structure naturally represents time flow, and the neuromorphic architecture can learn temporal patterns through edge weights.

## Objectives
- Design time-series-to-graph encoding (temporal dependencies as DAG edges)
- Implement regression output layer (continuous value prediction)
- Train neuromorphic GRAPHEME on time series dataset (sine wave, stock prices, or weather)
- Achieve low MSE (mean squared error) on forecasting task
- Verify end-to-end gradient flow for regression
- Demonstrate learnability of temporal patterns

## Tasks
- [ ] Design time-series-to-graph encoding (sliding window, temporal edges)
- [ ] Implement `TimeSeriesEncoder` trait
- [ ] Add regression output layer (continuous value prediction)
- [ ] Implement MSE (Mean Squared Error) loss for regression
- [ ] Choose time series dataset (sine wave, stock prices, or weather)
- [ ] Create `grapheme-train/src/bin/train_timeseries.rs` training script
- [ ] Implement data loader for time series (windowed sequences)
- [ ] Train neuromorphic model on time series
- [ ] Evaluate MSE and prediction accuracy on test set
- [ ] Verify gradient flow through all components
- [ ] Verify loss decreases over training
- [ ] Visualize predictions vs ground truth
- [ ] Benchmark training time and memory usage
- [ ] Document time-series-to-graph encoding design

## Acceptance Criteria
✅ **Time-series-to-graph encoding works:**
- Sliding window: past N timesteps → N nodes
- Temporal edges: t → t+1 connections (DAG order)
- Feature encoding: normalize values to [0, 1] or standardize
- DAG topology preserves temporal causality

✅ **Regression head works:**
- Single output node for next-step prediction
- MSE loss: `L = (y_pred - y_true)²`
- Gradient flows through output layer
- Predictions are continuous values (not discrete classes)

✅ **Training converges:**
- MSE loss decreases consistently over epochs
- Predictions improve over training
- No gradient vanishing/explosion
- Model learns temporal patterns (not just mean)

✅ **Temporal learnability:**
- Edge weights learn temporal dependencies (short-term, long-term)
- Node activations capture temporal features (trends, cycles)
- Model predicts better than naive baseline (previous value)
- Model captures nonlinear patterns (not just linear regression)

✅ **Performance is acceptable:**
- Training time < 5 minutes for 1000 epochs
- Memory usage < 1GB
- Inference time < 1ms per prediction

## Technical Notes
### Time-Series-to-Graph Encoding

**Sliding Window Approach:**
```
Input: [x_0, x_1, x_2, ..., x_T]  (T timesteps)
Window size: W = 10 (use past 10 timesteps to predict next)

For each prediction at time t:
  Nodes: [x_{t-W}, x_{t-W+1}, ..., x_{t-1}]  (10 input nodes)
  Edges: Sequential (x_i → x_{i+1}) + skip connections
  Target: x_t (predict next value)
```

**DAG Structure:**
```
x_{t-10} → x_{t-9} → ... → x_{t-1} → [hidden] → x_t (output)
    ↓         ↓                ↓
    └─────────┴────────────────┘  (skip connections for long-term dependencies)
```

**Feature Normalization:**
- Min-max scaling: `x_norm = (x - x_min) / (x_max - x_min)`
- Or z-score: `x_norm = (x - μ) / σ`
- Store normalization params for inference

### Architecture Design

**Input Layer:**
- W nodes (window size, e.g., 10 timesteps)
- Node activation = normalized value
- Node type: `NodeType::Input(timestep_idx)`

**Hidden Layers:**
- Neuromorphic forward pass with edge weights
- Edge weights learn temporal dependencies
- Per-node ReLU or Tanh activations

**Output Layer:**
- Single output node for regression
- Aggregates from hidden layer: `y_pred = sum(w_i * h_i)`
- No activation (linear output for regression)

**Loss Function:**
- MSE: `L = (y_pred - y_true)²`
- Gradient: `∂L/∂y_pred = 2(y_pred - y_true)`

### Training Pipeline

```rust
// grapheme-train/src/bin/train_timeseries.rs

fn main() {
    // Generate or load time series
    let timeseries = generate_sine_wave(1000); // 1000 timesteps
    // Or: load_stock_prices("AAPL.csv");

    // Split into train/test
    let (train, test) = split_timeseries(&timeseries, 0.8);

    // Create neuromorphic model
    let mut model = GraphTransformNet::new(
        vocab_size: 256,  // discretize continuous values
        hidden_dim: 64,
        num_layers: 2,
    );

    // Training loop
    for epoch in 1..=1000 {
        let mut total_loss = 0.0;

        for window in train.windows(WINDOW_SIZE + 1) {
            // window[0..W] = input, window[W] = target
            let input = &window[0..WINDOW_SIZE];
            let target = window[WINDOW_SIZE];

            // Convert to graph
            let graph = encode_timeseries_to_graph(input);

            // Forward pass
            let y_pred = model.forward(&graph);

            // Compute MSE loss
            let loss = (y_pred - target).powi(2);
            total_loss += loss;

            // Backward pass
            model.backward(loss);

            // Update parameters
            model.step(lr);
        }

        if epoch % 100 == 0 {
            let test_mse = evaluate_mse(&model, &test);
            println!("Epoch {}: Train MSE={:.6}, Test MSE={:.6}",
                     epoch, total_loss / train.len() as f32, test_mse);
        }
    }
}
```

### Dataset Options

**Option 1: Synthetic Sine Wave (Simple)**
```rust
fn generate_sine_wave(n: usize) -> Vec<f32> {
    (0..n).map(|t| (t as f32 * 0.1).sin()).collect()
}
```
- Perfectly learnable (deterministic)
- Tests if model can learn periodic patterns
- Baseline MSE: ~0 (should perfectly learn)

**Option 2: Stock Prices (Real-World)**
- Dataset: S&P 500 daily close prices
- Download from Yahoo Finance or Kaggle
- More challenging (noisy, non-stationary)
- Baseline MSE: compare to "predict previous value"

**Option 3: Weather Data (Real-World)**
- Temperature, humidity, etc.
- Multi-variate time series
- Tests multi-input encoding

**Chosen: Option 1 (Sine Wave)** for initial verification, then Option 2 for real-world test.

### Expected Results

**Baseline (traditional ML):**
- Naive (predict previous value): MSE depends on data
- Linear regression: MSE ~0.1 for sine wave
- LSTM: MSE ~0.01 for sine wave

**Target for neuromorphic GRAPHEME:**
- Sine wave: MSE < 0.01 (learn perfect periodicity)
- Stock prices: MSE < naive baseline (beat "predict previous")
- Demonstrates temporal learning capability

### Gradient Flow Verification

**Check:**
1. Embedding gradients (value → embedding)
2. Edge weight gradients (temporal dependencies)
3. Node activation gradients (temporal features)
4. Output layer gradients (prediction error)

**Diagnostic:**
- Print gradient magnitudes per component
- Verify gradients have reasonable scale
- Verify loss decreases (not stuck)

### Dependencies
- Depends on backend-111 (full neuromorphic architecture with backward pass)
- Introduces temporal modality (sequences over time)
- Tests recurrent/temporal learning capability

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
