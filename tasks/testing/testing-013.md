---
id: testing-013
title: Integration tests for A2A protocol and MCP server/client
status: done
priority: medium
tags:
- testing
- a2a
- mcp
- integration
dependencies:
- api-015
- api-017
- api-018
assignee: developer
created: 2025-12-11T07:46:37.477451051Z
estimate: 5h
complexity: 6
area: testing
---

# Integration tests for A2A protocol and MCP server/client

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
- [x] Create integration test file (tests/integration_a2a_mcp.rs)
- [x] Write 16 integration tests covering:
  - A2A full lifecycle (info, create, get)
  - A2A serialization roundtrip
  - A2A error handling
  - A2A multiple tasks and concurrent operations
  - A2A agent card structure and skills availability
  - MCP full lifecycle (initialize, list tools, call tool)
  - MCP server info and error handling
  - MCP graph operations
  - Cross-protocol text processing
  - JSON-RPC compliance
  - Protocol version compatibility
- [x] All tests pass

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
- 2025-12-11: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created `grapheme-tests/tests/integration_a2a_mcp.rs` with 16 tests
- Added `serde_json` dependency to grapheme-tests/Cargo.toml

### Causality Impact
- No runtime changes - these are test-only additions
- Tests exercise A2A and MCP protocols end-to-end

### Dependencies & Integration
- grapheme-tests now depends on serde_json
- Tests use A2AAgent, MCPServer, and serialization helpers

### Verification & Testing
- Run `cargo test -p grapheme-tests --test integration_a2a_mcp`
- All 16 tests should pass
- Tests cover full lifecycle, error handling, and cross-protocol scenarios

### Context for Next Task
- A2A task IDs use format "task_N" where N increments
- MCP server fields (info, tools) are private - access via methods
- Error handling varies: some methods return error in response.error, others in result