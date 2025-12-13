---
id: backend-212
title: Implement parallel multi-cortex code training with Rayon
status: todo
priority: critical
tags:
- backend
- parallel
- cortex
- training
- humaneval
dependencies:
- backend-209
assignee: developer
created: 2025-12-11T17:25:48.581475389Z
estimate: 6h
complexity: 8
area: backend
---

# Implement parallel multi-cortex code training with Rayon

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
GRAPHEME uses **Graph → Transform → Graph** paradigm (NOT autoregressive generation).
CortexMesh already supports parallel processing via Rayon. This task extends parallel
training to leverage multiple cortices (domain brains) simultaneously during training.

Currently, `train_cortex_mesh.rs` processes batches sequentially. We need to:
1. Parallelize batch processing across multiple samples
2. Enable multiple brains to process in parallel during `mesh.train_step()`
3. Ensure thread-safe gradient accumulation and model updates

**Key paradigm note**: We're training graph transformation, not character generation.
The parallel training improves throughput for Graph → Transform → Graph operations.

## Objectives
- Enable Rayon-based parallel batch processing during training
- Achieve N× speedup where N = number of CPU cores
- Maintain training quality (loss convergence) with parallel updates
- Support parallel brain activation during forward passes

## Tasks
- [ ] Review existing CortexMesh parallel infrastructure (`process_parallel()`)
- [ ] Implement parallel batch processing in train_step loop
- [ ] Add thread-safe gradient accumulation for model parameters
- [ ] Benchmark: measure training throughput improvement
- [ ] Verify loss convergence matches sequential training

## Acceptance Criteria
✅ **Criteria 1:**
- Training throughput increases by at least 2× on multi-core systems

✅ **Criteria 2:**
- Model quality (val_loss, similarity) matches sequential baseline

✅ **Criteria 3:**
- All existing tests pass, no race conditions or data corruption

## Technical Notes
- CortexMesh already has `config.parallel` flag and uses Rayon for brain processing
- Key file: `grapheme-train/src/cortex_mesh.rs` and `train_cortex_mesh.rs`
- Must ensure GraphTransformNet weight updates are thread-safe
- Consider using `parking_lot::Mutex` or atomic operations for gradient accumulation
- Graph transformation (not autoregressive) - batch graphs can be processed independently

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
- 2025-12-11: Task created

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
