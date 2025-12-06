---
id: backend-041
title: Parallelize training loop with Rayon
status: todo
priority: high
tags:
- backend
dependencies:
- backend-030
assignee: developer
created: 2025-12-06T09:57:26.491047738Z
estimate: ~
complexity: 3
area: backend
---

# Parallelize training loop with Rayon

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
The training loop in grapheme-train/src/lib.rs (lines 2647-2693) processes examples sequentially. Modern GPUs/CPUs have 8-128 cores that are completely unused. This is the most CRITICAL performance bottleneck.

**Current sequential code:**
```rust
pub fn train(&mut self, dataset: &Dataset, config: &TrainingConfig) {
    for _epoch in 0..config.epochs {
        for example in &dataset.examples {  // Sequential!
            let result = self.process(input);
            let ged = GraphEditDistance::compute(predicted_graph, &expected_graph);
        }
    }
}
```

## Objectives
- Parallelize training loop using Rayon
- Achieve near-linear speedup with CPU cores
- Maintain gradient accumulation correctness
- Support configurable batch parallelism

## Tasks
- [ ] Add rayon dependency to grapheme-train/Cargo.toml
- [ ] Convert example loop to `par_iter()`
- [ ] Implement thread-safe gradient accumulation
- [ ] Add atomic loss accumulator
- [ ] Benchmark single-threaded vs parallel
- [ ] Add configurable worker count

## Acceptance Criteria
✅ **Parallel Speedup:**
- 4x+ speedup on 8-core CPU
- Linear scaling with cores (up to memory bandwidth)

✅ **Correctness:**
- Gradient accumulation produces same results
- Loss values match sequential version (within floating point tolerance)

## Technical Notes
- Use `rayon::prelude::*` for parallel iterators
- Consider `par_chunks()` for mini-batch processing
- Gradient accumulation needs `Mutex<Vec<f32>>` or atomic operations
- File: grapheme-train/src/lib.rs lines 2647-2693
- **CRITICAL PRIORITY** - this blocks all training performance

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