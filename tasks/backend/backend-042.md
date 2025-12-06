---
id: backend-042
title: Parallelize message passing and forward_batch
status: todo
priority: high
tags:
- backend
dependencies:
- backend-029
assignee: developer
created: 2025-12-06T09:57:29.583986973Z
estimate: ~
complexity: 3
area: backend
---

# Parallelize message passing and forward_batch

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
Two critical loops in grapheme-core process nodes sequentially:

1. **MessagePassingLayer.forward** (lines 2813-2835): Aggregates neighbors sequentially
2. **MessagePassingLayer.forward_batch** (lines 3366-3378): Processes all nodes in a loop

**Current sequential code:**
```rust
// forward_batch - CRITICAL bottleneck
pub fn forward_batch(&self, node_features: &Array2<f32>, adjacency: &[Vec<usize>]) -> Array2<f32> {
    for i in 0..n {  // Sequential!
        let out = self.forward(&node_feat, &neighbor_feats);
        output.row_mut(i).assign(&out);
    }
}
```

## Objectives
- Parallelize node-level operations with Rayon
- Maintain message passing correctness (all reads before writes)
- Enable GPU-style parallel node processing on CPU

## Tasks
- [ ] Add rayon dependency to grapheme-core/Cargo.toml
- [ ] Convert forward_batch node loop to parallel
- [ ] Use `par_iter().map().collect()` pattern for thread-safe output
- [ ] Parallelize neighbor aggregation where beneficial
- [ ] Benchmark with various graph sizes
- [ ] Consider SIMD for inner vector operations (ndarray parallel feature)

## Acceptance Criteria
✅ **Parallel Forward Pass:**
- All nodes processed in parallel
- No data races in adjacency reads

✅ **Correct Output:**
- Identical results to sequential version
- All message passing tests pass

## Technical Notes
- Node operations are embarrassingly parallel (read neighbors, compute, write self)
- Use `into_par_iter()` on node indices
- Consider `ndarray-parallel` feature for BLAS parallelism
- File: grapheme-core/src/lib.rs lines 2813-2835, 3366-3378

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