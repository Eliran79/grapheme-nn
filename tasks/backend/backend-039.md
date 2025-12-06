---
id: backend-039
title: Replace exponential clique enumeration with Bron-Kerbosch algorithm
status: todo
priority: high
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T09:57:19.285450907Z
estimate: ~
complexity: 3
area: backend
---

# Replace exponential clique enumeration with Bron-Kerbosch algorithm

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
The current `find_cliques_simple` function in grapheme-core/src/lib.rs (lines 1625-1643) uses O(n^k) enumeration which is exponential in k. The helper `combinations` function (lines 1748-1770) also generates O(2^n) combinations recursively. These are NP-hard in general and will become a training bottleneck.

**Current problematic code:**
```rust
// O(n^k) - exponential in k
fn find_cliques_simple(&self, k: usize) -> Vec<Vec<NodeId>> {
    for combo in Self::combinations(&nodes, k) {
        if self.is_clique(&combo) { cliques.push(combo); }
    }
}
```

## Objectives
- Replace exponential clique enumeration with Bron-Kerbosch algorithm O(3^(n/3))
- Limit k-clique search to reasonable bounds (e.g., k ≤ 10)
- Add early termination when max_results is reached
- Maintain API compatibility with existing code

## Tasks
- [ ] Implement Bron-Kerbosch algorithm with pivoting
- [ ] Replace combinations() with iterative version to avoid stack overflow
- [ ] Add max_cliques parameter to limit results
- [ ] Add max_k parameter to cap clique size searched
- [ ] Benchmark old vs new implementation
- [ ] Update all callers to use bounded parameters

## Acceptance Criteria
✅ **Polynomial in Practice:**
- O(3^(n/3)) worst case instead of O(n^k)
- Handles graphs with 1000+ nodes without explosion

✅ **Bounded Search:**
- Max clique size configurable (default k=10)
- Early termination after max_results found

## Technical Notes
- Bron-Kerbosch reference: https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
- Use pivot vertex selection for O(3^(n/3)) complexity
- Consider degeneracy ordering for sparse graphs (already partially implemented)
- File: grapheme-core/src/lib.rs lines 1625-1770

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
