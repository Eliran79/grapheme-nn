---
id: backend-043
title: Parallelize GED computation with Rayon
status: todo
priority: high
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T09:57:32.935572993Z
estimate: ~
complexity: 3
area: backend
---

# Parallelize GED computation with Rayon

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
Graph Edit Distance (GED) computation in grapheme-train has three sequential bottlenecks:

1. **compute_node_costs_grapheme** (lines 1349-1373): O(n1*n2) matrix computed sequentially
2. **Edge cost computation** (lines 1490-1507): Uses O(n) linear search with `position()`
3. **Greedy assignment** (lines 1442-1475): Sequential greedy matching

**Current sequential code:**
```rust
fn compute_node_costs_grapheme(g1: &GraphemeGraph, g2: &GraphemeGraph) -> Vec<Vec<NodeCost>> {
    for (i, &idx1) in indices1.iter().enumerate() {  // Sequential!
        for (j, &idx2) in indices2.iter().enumerate() {
            costs[i][j] = NodeCost { label_cost, degree_cost };
        }
    }
}
```

## Objectives
- Parallelize node cost matrix computation
- Replace linear search with HashMap for O(1) lookup
- Parallelize row-wise computation for large graphs

## Tasks
- [ ] Convert outer loop in compute_node_costs to par_iter
- [ ] Replace `indices.iter().position()` with HashMap lookup
- [ ] Benchmark GED computation on 100, 500, 1000 node graphs
- [ ] Consider parallel Hungarian algorithm for optimal matching (future)

## Acceptance Criteria
✅ **Parallel Cost Matrix:**
- Rows computed in parallel
- 4x+ speedup on 8 cores for large graphs

✅ **O(1) Index Lookup:**
- HashMap instead of linear search
- Edge cost computation is O(edges) not O(edges * nodes)

## Technical Notes
- Use `par_iter().map().collect()` for cost rows
- HashMap<NodeIndex, usize> for index lookup
- File: grapheme-train/src/lib.rs lines 1349-1373, 1442-1475, 1490-1507
- GED is called per training example - high impact optimization

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
