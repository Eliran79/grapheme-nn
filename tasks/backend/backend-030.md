---
id: backend-030
title: Implement end-to-end NL to Math pipeline (Layer 4-3-2-1)
status: done
priority: high
tags:
- backend
dependencies:
- backend-029
assignee: developer
created: 2025-12-06T08:41:23.909554081Z
estimate: ~
complexity: 3
area: backend
---

# Implement end-to-end NL to Math pipeline (Layer 4-3-2-1)

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
Chain all layers: "What's 2+3?" → grapheme-core → grapheme-math → grapheme-polish → grapheme-engine → "5"

## Objectives
- Create unified pipeline from NL input to result
- Wire Layer 4→3→2→1 transformations
- Support both inference and training modes

## Tasks
- [x] Create `Pipeline` struct connecting all layers
- [x] Implement NL→MathGraph extraction (Layer 4→3)
- [x] Wire MathGraph→Polish conversion (Layer 3→2)
- [x] Wire Polish→Engine evaluation (Layer 2→1)
- [x] Add result→text conversion
- [x] Create CLI tool for interactive testing (quick_eval, quick_symbolic functions)
- [x] Add batch processing mode

## Acceptance Criteria
✅ **End-to-End:**
- "2 + 3" → 5
- "derivative of x squared" → "2*x"
- "integrate x from 0 to 1" → 0.5

✅ **Modes:**
- Inference mode (frozen weights)
- Training mode (gradient flow through all layers)

## Technical Notes
- Use existing crate interfaces
- Handle errors gracefully at each layer
- Support streaming input for long texts
- Consider caching intermediate representations

## Testing
- [x] Write unit tests for new functionality
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete
- [x] Consider edge cases and error conditions

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
  - Run `cargo build --release` (or `cargo build` for debug)
  - Run `cargo test` to ensure tests pass
  - **Actually run/execute the code** to verify runtime behavior
  - Fix all errors, warnings, and runtime issues
- [x] Commit changes incrementally with clear messages
- [x] Use descriptive commit messages that explain the "why"
- [x] Consider creating a feature branch for complex changes
- [x] Review changes before committing

**Testing requirements by change type:**
- Code changes: Build + test + **run the actual program/command** to verify behavior
- Bug fixes: Verify the bug is actually fixed by running the code, not just compiling
- New features: Test the feature works as intended by executing it
- Minor changes: At minimum build, check warnings, and run basic functionality

## Updates
- 2025-12-06: Task created
- 2025-12-06: Task completed - End-to-end Pipeline implemented

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added `PipelineMode` enum: Inference, Training modes
- Added `PipelineResult` struct: Contains all intermediate results (nl_graph, math_graph, optimized_expr, numeric_result, symbolic_result, steps, errors)
- Added `Pipeline` struct: End-to-end processor connecting all 4 layers
- Added natural language expression extraction (supports: "what is", "calculate", "derivative of", "integrate X from A to B", "X squared/cubed")
- Added 22 new pipeline tests
- Added `quick_eval()` and `quick_symbolic()` helper functions
- File: grapheme-train/src/lib.rs (lines ~2238-2735)

### Key APIs
```rust
// Create pipeline
let mut pipeline = Pipeline::new();

// Process input through all layers
let result = pipeline.process("2 + 3");
assert_eq!(result.numeric_result, Some(5.0));

// Check success and get result string
if result.success() {
    println!("{}", result.result_string());
}

// Bind variables
pipeline.bind("x", 5.0);
let result = pipeline.process("x + 1");  // Returns 6.0

// Batch processing
let results = pipeline.process_batch(&["1 + 1", "2 * 3"]);

// Quick evaluation functions
assert_eq!(quick_eval("2 + 3"), Some(5.0));
assert_eq!(quick_symbolic("x + 0"), Some("x".to_string()));
```

### Causality Impact
- Layer 4 (grapheme-core): Text → GraphemeGraph (character-level processing)
- Layer 3 (grapheme-math): Expression → MathGraph
- Layer 2 (grapheme-polish): Expression → Optimized via constant folding and identity elimination
- Layer 1 (grapheme-engine): Expression → Evaluated result (numeric or symbolic)
- Pipeline records all processing steps for debugging

### Dependencies & Integration
- Uses GraphemeGraph::from_text() for NL graph creation
- Uses MathGraph::from_expr() for math representation
- Uses grapheme_polish::Optimizer for expression optimization
- Uses MathEngine for numeric evaluation
- Uses SymbolicEngine for differentiation and integration
- Integrates with TrainingLoop for training mode

### Verification & Testing
- 22 new tests: test_pipeline_*, test_quick_*
- Run: `cargo test -p grapheme-train`
- 77 tests in grapheme-train, 404 total across workspace

### Context for Next Task
- For testing-004 (benchmarks):
  - Use Pipeline for end-to-end inference benchmarks
  - Compare quick_eval() latency vs. transformer inference
  - Measure memory usage per graph vs. per token
- Pipeline supports both infix ("2 + 3") and prefix/polish ("+ 2 3") notation
- Natural language patterns: "what is", "calculate", "compute", "evaluate", "derivative of", "integrate"
- Symbolic expressions remain symbolic unless all variables are bound