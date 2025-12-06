---
id: backend-044
title: Parallelize backward pass gradient computation
status: todo
priority: medium
tags:
- backend
dependencies:
- backend-041
- backend-042
assignee: developer
created: 2025-12-06T09:57:36.851115159Z
estimate: ~
complexity: 3
area: backend
---

# Parallelize backward pass gradient computation

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
The backward pass in grapheme-core/src/lib.rs (lines 2889-2898) computes gradients sequentially for each layer and parameter. While less critical than forward pass, this still limits training throughput.

**Current sequential code:**
```rust
impl BackwardPass for MessagePassingLayer {
    fn backward(&mut self, grad_output: &[f32], _learning_rate: f32) {
        for (g, param) in self.weight_grad.iter_mut().zip(grad_output.iter()) {
            *g += param;  // Sequential accumulation
        }
    }
}
```

## Objectives
- Parallelize gradient computation across layers
- Use SIMD/vector operations for element-wise updates
- Maintain gradient correctness

## Tasks
- [ ] Profile backward pass to identify true bottlenecks
- [ ] Parallelize independent layer gradient computation
- [ ] Use ndarray's parallel operations for vector math
- [ ] Consider fused gradient + update step
- [ ] Benchmark vs sequential version

## Acceptance Criteria
✅ **Parallel Gradients:**
- Independent layers computed in parallel
- No gradient corruption from race conditions

✅ **Vector Operations:**
- Use SIMD where possible
- Avoid element-by-element loops for large vectors

## Technical Notes
- Lower priority than forward pass parallelization
- Layer gradients are independent - can parallelize across layers
- Within-layer: use `ndarray::parallel::par_azip!` for element ops
- File: grapheme-core/src/lib.rs lines 2889-2898
- Consider this after backend-041 and backend-042 are complete

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