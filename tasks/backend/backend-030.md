---
id: backend-030
title: Implement end-to-end NL to Math pipeline (Layer 4-3-2-1)
status: todo
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
- [ ] Create `Pipeline` struct connecting all layers
- [ ] Implement NL→MathGraph extraction (Layer 4→3)
- [ ] Wire MathGraph→Polish conversion (Layer 3→2)
- [ ] Wire Polish→Engine evaluation (Layer 2→1)
- [ ] Add result→text conversion
- [ ] Create CLI tool for interactive testing
- [ ] Add batch processing mode

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