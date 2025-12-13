---
id: backend-116
title: Implement AGI-ready cognitive router (task selection)
status: done
priority: critical
tags:
- backend
dependencies:
- backend-115
assignee: developer
created: 2025-12-08T08:51:45.080009996Z
estimate: ~
complexity: 3
area: backend
---

# Implement AGI-ready cognitive router (task selection)

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
Implement an AGI-ready cognitive router that automatically selects which cognitive module to use based on input analysis. This is the "brain coordinator" that routes inputs to the right specialist (NLP, vision, time series, math, etc.) - a key component of AGI.

Real AGI systems receive diverse inputs (text, images, sequences) and must intelligently route them to appropriate cognitive modules. The router analyzes input characteristics and activates the correct module(s). This task implements:
1. **Input analysis**: Detect input type (text, image, time series, etc.)
2. **Module selection**: Choose appropriate cognitive module(s)
3. **Multi-module coordination**: Combine outputs from multiple modules if needed
4. **Learnable routing**: Optionally train router to improve selection over time

## Objectives
- Implement automatic input type detection (text, image, time series, etc.)
- Create routing logic to select appropriate cognitive module
- Support single-module and multi-module routing
- Implement confidence scoring for routing decisions
- Optionally: Make router learnable (train on routing accuracy)
- Verify router correctly routes diverse inputs
- Demonstrate end-to-end AGI-style system (input → router → module → output)

## Tasks
- [ ] Design input analysis system (feature extraction for routing)
- [ ] Implement input type classifier (text, image, time series, etc.)
- [ ] Create `CognitiveRouter` struct with module registry
- [ ] Implement routing logic (rule-based or learned)
- [ ] Add confidence scoring for routing decisions
- [ ] Implement multi-module coordination (ensemble outputs)
- [ ] Create `grapheme-train/src/bin/demo_agi_router.rs` demo script
- [ ] Test routing on text inputs → kindergarten/math module
- [ ] Test routing on image inputs → MNIST module
- [ ] Test routing on time series inputs → forecasting module
- [ ] Test routing on ambiguous inputs (should handle gracefully)
- [ ] Implement learnable router (optional: train routing network)
- [ ] Benchmark routing latency (<10ms overhead)
- [ ] Document router architecture and module registration

## Acceptance Criteria
✅ **Input analysis works:**
- Router detects text inputs (character sequences)
- Router detects image inputs (2D arrays of pixels)
- Router detects time series inputs (1D sequences of floats)
- Detection is fast (<1ms per input)

✅ **Routing logic works:**
- Text inputs route to kindergarten/math module
- Image inputs route to MNIST module
- Time series inputs route to forecasting module
- Routing is deterministic (same input → same module)

✅ **Confidence scoring works:**
- Router outputs confidence score (0.0 to 1.0)
- High confidence (>0.9) for clear inputs
- Low confidence (<0.5) for ambiguous inputs
- Can set confidence threshold for rejection

✅ **Multi-module coordination works:**
- Router can activate multiple modules for hybrid inputs
- Outputs are combined (e.g., weighted average, voting)
- Combined output is sensible (not garbled)

✅ **Performance is good:**
- Routing overhead <10ms per input
- Routing accuracy >95% on clear inputs
- Graceful degradation on ambiguous inputs

✅ **AGI-ready demonstration:**
- Single system handles text, images, and time series
- User provides input, system automatically routes and processes
- Output is correct and module selection is explainable

## Technical Notes
### Input Analysis Strategies

**Option 1: Rule-Based Detection (Simple)**
```rust
fn detect_input_type(input: &Input) -> InputType {
    if input.is_2d_array() && input.all_values_in_range(0, 255) {
        InputType::Image
    } else if input.is_1d_sequence() && input.has_numeric_values() {
        InputType::TimeSeries
    } else if input.is_text() {
        // Further classify: kindergarten vs math
        if input.contains_math_symbols() {
            InputType::MathText
        } else {
            InputType::KindergartenText
        }
    } else {
        InputType::Unknown
    }
}
```
- Pros: Fast, deterministic, interpretable
- Cons: Requires manual rules, may miss edge cases

**Option 2: Learned Router (Advanced)**
```rust
// Small neural network that classifies input type
pub struct LearnedRouter {
    feature_extractor: FeatureNet,  // Extract features from raw input
    classifier: OutputLayer,        // Classify to module
}

fn route(&self, input: &Input) -> (ModuleId, f32) {
    let features = self.feature_extractor.forward(input);
    let logits = self.classifier.forward(features);
    let (module_id, confidence) = argmax_with_softmax(logits);
    (module_id, confidence)
}
```
- Pros: Adaptive, learns from mistakes, handles edge cases
- Cons: Requires training data, more complex

**Chosen: Option 1 (Rule-Based)** initially, with Option 2 as future enhancement.

### Router Architecture

```rust
pub enum InputType {
    KindergartenText,
    MathText,
    Image,
    TimeSeries,
    Unknown,
}

pub enum ModuleId {
    Kindergarten,
    Math,
    MNIST,
    TimeSeries,
}

pub struct CognitiveModule {
    id: ModuleId,
    model: GraphTransformNet,
    input_preprocessor: fn(&Input) -> GraphemeGraph,
    output_postprocessor: fn(&Output) -> Result,
}

pub struct CognitiveRouter {
    modules: HashMap<ModuleId, CognitiveModule>,
    routing_confidence_threshold: f32,  // Reject if confidence < threshold
}

impl CognitiveRouter {
    pub fn route(&self, input: &Input) -> RouterResult {
        // Step 1: Analyze input
        let input_type = self.detect_input_type(input);

        // Step 2: Select module(s)
        let (module_ids, confidence) = self.select_modules(input_type);

        // Step 3: Check confidence
        if confidence < self.routing_confidence_threshold {
            return Err("Input type ambiguous, confidence too low");
        }

        // Step 4: Execute module(s)
        let outputs = self.execute_modules(&module_ids, input);

        // Step 5: Combine outputs (if multiple modules)
        let final_output = self.combine_outputs(outputs);

        Ok(RouterResult {
            output: final_output,
            module_used: module_ids,
            confidence,
        })
    }

    fn detect_input_type(&self, input: &Input) -> InputType {
        // Rule-based detection (see above)
        // ...
    }

    fn select_modules(&self, input_type: InputType) -> (Vec<ModuleId>, f32) {
        match input_type {
            InputType::KindergartenText => (vec![ModuleId::Kindergarten], 0.95),
            InputType::MathText => (vec![ModuleId::Math], 0.95),
            InputType::Image => (vec![ModuleId::MNIST], 0.95),
            InputType::TimeSeries => (vec![ModuleId::TimeSeries], 0.95),
            InputType::Unknown => (vec![], 0.0),  // Reject
        }
    }

    fn execute_modules(&self, module_ids: &[ModuleId], input: &Input) -> Vec<Output> {
        module_ids.iter().map(|id| {
            let module = &self.modules[id];
            let graph = (module.input_preprocessor)(input);
            let output = module.model.forward(&graph);
            (module.output_postprocessor)(&output)
        }).collect()
    }

    fn combine_outputs(&self, outputs: Vec<Output>) -> Output {
        if outputs.len() == 1 {
            outputs[0]
        } else {
            // Ensemble: voting, averaging, or max-confidence
            ensemble_voting(outputs)
        }
    }

    pub fn register_module(&mut self, module: CognitiveModule) {
        self.modules.insert(module.id, module);
    }
}
```

### Module Registration

```rust
// In demo_agi_router.rs

fn main() {
    // Create router
    let mut router = CognitiveRouter::new(confidence_threshold: 0.5);

    // Register kindergarten module
    let kindergarten_module = CognitiveModule {
        id: ModuleId::Kindergarten,
        model: load_model("kindergarten_model.bin"),
        input_preprocessor: preprocess_text_to_graph,
        output_postprocessor: decode_letter_output,
    };
    router.register_module(kindergarten_module);

    // Register math module
    let math_module = CognitiveModule {
        id: ModuleId::Math,
        model: load_model("math_model.bin"),
        input_preprocessor: preprocess_math_to_graph,
        output_postprocessor: decode_math_output,
    };
    router.register_module(math_module);

    // Register MNIST module
    let mnist_module = CognitiveModule {
        id: ModuleId::MNIST,
        model: load_model("mnist_model.bin"),
        input_preprocessor: preprocess_image_to_graph,
        output_postprocessor: decode_digit_output,
    };
    router.register_module(mnist_module);

    // Register time series module
    let timeseries_module = CognitiveModule {
        id: ModuleId::TimeSeries,
        model: load_model("timeseries_model.bin"),
        input_preprocessor: preprocess_timeseries_to_graph,
        output_postprocessor: decode_forecast_output,
    };
    router.register_module(timeseries_module);

    // Demo: route diverse inputs
    let text_input = Input::Text("cat");
    let result = router.route(&text_input)?;
    println!("Input: 'cat' -> Module: {:?}, Output: {:?}, Confidence: {:.2}",
             result.module_used, result.output, result.confidence);

    let math_input = Input::Text("2+3");
    let result = router.route(&math_input)?;
    println!("Input: '2+3' -> Module: {:?}, Output: {:?}, Confidence: {:.2}",
             result.module_used, result.output, result.confidence);

    let image_input = Input::Image(mnist_digit_7);
    let result = router.route(&image_input)?;
    println!("Input: [image] -> Module: {:?}, Output: {:?}, Confidence: {:.2}",
             result.module_used, result.output, result.confidence);

    let ts_input = Input::TimeSeries(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = router.route(&ts_input)?;
    println!("Input: [timeseries] -> Module: {:?}, Output: {:?}, Confidence: {:.2}",
             result.module_used, result.output, result.confidence);
}
```

### Confidence Scoring

**Rule-Based Confidence:**
```rust
fn compute_confidence(input: &Input, detected_type: InputType) -> f32 {
    match detected_type {
        InputType::Image if input.shape() == (28, 28) => 0.99,  // Perfect match
        InputType::Image => 0.80,  // Image but wrong size
        InputType::TimeSeries if input.len() >= 10 => 0.95,  // Enough context
        InputType::TimeSeries => 0.70,  // Too short
        InputType::KindergartenText if input.is_alphabetic() => 0.95,
        InputType::MathText if input.contains_digits() => 0.95,
        InputType::Unknown => 0.0,  // Reject
        _ => 0.50,  // Ambiguous
    }
}
```

**Learned Confidence:**
- Train classifier with softmax → use max probability as confidence
- Calibrate confidence with temperature scaling

### Multi-Module Coordination

**Ensemble Methods:**
1. **Voting** (for classification):
   - Each module votes for a class
   - Majority wins
2. **Weighted Average** (for regression):
   - Weight by confidence
   - `output = Σ(confidence_i × output_i) / Σ(confidence_i)`
3. **Max Confidence** (for ambiguous inputs):
   - Use output from most confident module

### Learnable Router (Optional)

**Training Strategy:**
```rust
// Train router to minimize routing error
// Loss = CrossEntropy(predicted_module, ground_truth_module)

for (input, correct_module) in training_data {
    let predicted_module = router.route(input);
    let loss = cross_entropy(predicted_module, correct_module);
    router.backward(loss);
    router.step(lr);
}
```

**Data Generation:**
- Use existing datasets with module labels
- Kindergarten → label "Kindergarten"
- MNIST → label "MNIST"
- Train router to learn input type detection

### Expected Results

**Routing Accuracy:**
- Text inputs: >99% correct routing
- Image inputs: >99% correct routing
- Time series inputs: >99% correct routing
- Ambiguous inputs: Correctly rejected or routed to multiple modules

**Performance:**
- Routing latency: <10ms per input
- End-to-end latency: routing + module inference <100ms
- Memory overhead: <50MB for router

**AGI Demonstration:**
- Single system handles 4+ different modalities
- Automatic routing without manual specification
- Explainable decisions (output includes module used and confidence)

### Dependencies
- Depends on backend-115 (multi-task learning infrastructure)
- Requires trained models for all cognitive modules (kindergarten, math, MNIST, timeseries)
- Demonstrates AGI-ready architecture: diverse inputs, automatic routing, unified system

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
- Created `grapheme-router` crate with AGI-ready cognitive router
- Implemented `CognitiveRouter` struct with module registration and confidence-based routing
- Created `CognitiveModule` trait for pluggable cognitive modules
- Implemented 4 built-in modules: `TextModule`, `MathModule`, `TimeSeriesModule`, `VisionModule`
- Created `Input` enum with constructors: `text()`, `sequence()`, `image()`, `CsvNumeric`
- Created `demo_agi_router.rs` binary demonstrating multi-modal input routing
- Added 15 unit tests for router functionality

### Causality Impact
- Router analyzes input → determines type → selects module → executes → returns result
- Confidence scoring allows threshold-based rejection of ambiguous inputs
- Alternative modules provided when primary has low confidence

### Dependencies & Integration
- New `grapheme-router` crate depends on: grapheme-core, grapheme-time, grapheme-math, grapheme-vision
- Added to workspace members in root Cargo.toml
- grapheme-train depends on grapheme-router for demo binary

### Verification & Testing
- `cargo run --bin demo_agi_router` runs successfully
- 100% routing accuracy on 13 diverse inputs
- Average routing latency: 8µs (well under 10ms target)
- All 15 router tests pass, zero clippy warnings

### Context for Next Task
- Router uses rule-based input type detection (no learned routing yet)
- Modules are stateless - each `process()` call is independent
- Confidence scores based on input pattern matching strength
- Multi-module routing supported via `alternatives` in RouterResult