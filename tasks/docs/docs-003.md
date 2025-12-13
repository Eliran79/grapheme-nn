---
id: docs-003
title: 'Document: Add P-time complexity requirement to all algorithm tasks'
status: done
priority: critical
tags:
- docs
- quality
- process
- complexity
dependencies: []
assignee: developer
created: 2025-12-10T23:05:29.551187736Z
estimate: ~
complexity: 2
area: docs
---

# Document: Add P-time complexity requirement to all algorithm tasks

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
Brief description of what needs to be done and why.

## Objectives
- Clear, actionable objectives
- Measurable outcomes
- Success criteria

## Tasks
- [ ] Break down the work into specific tasks
- [ ] Each task should be clear and actionable
- [ ] Mark tasks as completed when done

## Acceptance Criteria
✅ **Criteria 1:**
- Specific, testable criteria

✅ **Criteria 2:**
- Additional criteria as needed

## Technical Notes
- Implementation details
- Architecture considerations
- Dependencies and constraints

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
- 2025-12-10: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added "P-Time Complexity Only" requirement to CLAUDE.md (lines 15-35)
- Documented ALLOWED algorithms: BFS/DFS O(V+E), Cosine similarity O(n), sorts O(n log n)
- Documented FORBIDDEN algorithms: backtracking, exhaustive clique enumeration, NP-complete solvers
- Added concrete examples for both categories
- Statement: "If unsure about complexity, ask or use a simpler algorithm"

### Causality Impact
- All algorithms must be polynomial time - exponential algorithms rejected
- Prevents GRAPHEME from becoming computationally intractable
- Graph algorithms use BFS/DFS instead of exhaustive search
- Clique operations use fixed-k enumeration, not max-clique finding

### Dependencies & Integration
- CLAUDE.md enforces this for all AI-assisted development
- Existing algorithms reviewed: all confirmed P-time
- backend-009 (fixed-k clique enumeration) specifically implements P-time constraint
- Graph edit distance uses polynomial approximations (BP2)

### Verification & Testing
- Check CLAUDE.md lines 15-35 for complete P-time specification
- Review any new algorithms for complexity bounds
- All graph operations should cite their time complexity in comments

### Context for Next Task
- P-time is non-negotiable for GRAPHEME scalability
- Use O(n³) max for graph operations on large inputs
- When in doubt, use simpler algorithms with better bounds
- Fixed-k clique enumeration is polynomial: O(n^k) for fixed k