---
id: backend-213
title: Connect all 7 cortices (Math,Code,Vision,Music,Chem,Law,Text) for unified code generation
status: todo
priority: critical
tags:
- backend
- cortex
- unified
- codegen
- humaneval
dependencies:
- backend-209
assignee: developer
created: 2025-12-11T17:25:48.840882209Z
estimate: 8h
complexity: 9
area: backend
---

# Connect all 7 cortices (Math,Code,Vision,Music,Chem,Law,Text) for unified code generation

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
GRAPHEME uses **Graph → Transform → Graph** paradigm with domain-specific brains.
CortexMesh auto-discovers 8 domain brains: Math, Code, Vision, Classification, Law, Music, Chem, Time.

Currently, brains are independent. This task connects them for **unified code generation**:
- Input: Problem description graph (text → graph)
- Router: Activates relevant brains based on problem domain
- Brains: Each contributes domain-specific graph transformations
- Fusion: Combine brain outputs into unified output graph
- Output: Code solution graph (graph → text)

**Key paradigm note**: We're NOT doing autoregressive token generation.
The output is a complete code graph that gets decoded to text.

## Objectives
- Implement multi-brain fusion for code generation tasks
- Enable cross-domain reasoning (e.g., Math brain helps with algorithm problems)
- Create unified output graph from multiple brain contributions
- Support HumanEval-style problems that span multiple domains

## Tasks
- [ ] Design brain output fusion mechanism (weighted combination vs attention)
- [ ] Implement unified_process() that routes to multiple brains
- [ ] Add cross-brain attention for information sharing
- [ ] Create code-specific output graph structure (AST-like nodes)
- [ ] Test on multi-domain problems (math+code, text+code)

## Acceptance Criteria
✅ **Criteria 1:**
- All 8 domain brains can contribute to a single output graph

✅ **Criteria 2:**
- Code output graph structure is valid and decodable to syntactically correct code

✅ **Criteria 3:**
- Cross-domain problems show improved accuracy vs single-brain baseline

## Technical Notes
- CortexMesh has `process_parallel()` that routes to activated brains
- Brain activation based on input graph characteristics (not just keywords)
- Output fusion: consider attention over brain outputs vs simple weighted sum
- Code graph should preserve AST structure for accurate decoding
- Key files: `cortex_mesh.rs`, domain brain implementations in respective crates

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
