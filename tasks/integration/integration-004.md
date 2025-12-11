---
id: integration-004
title: Implement MCP client for connecting to external tool servers
status: done
priority: medium
tags:
- integration
- mcp
- client
- tools
- external
dependencies: []
assignee: developer
created: 2025-12-11T07:46:04.656062416Z
estimate: 5h
complexity: 6
area: integration
---

# Implement MCP client for connecting to external tool servers

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
- [x] Create MCPClient struct with configuration
- [x] Implement MCPTransport trait for abstraction
- [x] Implement StdioTransport for subprocess servers
- [x] Implement InMemoryTransport for testing
- [x] Add initialize/handshake protocol
- [x] Implement tools/list discovery
- [x] Implement tools/call invocation
- [x] Add MCPClientBuilder for ergonomic construction
- [x] Create MCPServerRegistry for multi-server management
- [x] Write 13 unit tests
- [x] Export module from lib.rs

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
- Created `grapheme-train/src/mcp_client.rs` (~550 lines)
- Key types: MCPClient, MCPClientConfig, MCPTransport trait, StdioTransport, InMemoryTransport, MCPClientBuilder, MCPServerRegistry, MCPClientError
- Added module export in lib.rs

### Causality Impact
- `connect_stdio()` spawns subprocess and performs initialize handshake
- `call_tool()` sends JSON-RPC request and waits for response (blocking)
- MCPServerRegistry.call_tool() routes to correct server automatically
- StdioTransport drop handler kills subprocess

### Dependencies & Integration
- Reuses types from mcp_server module (Tool, JsonRpcRequest, etc.)
- Can connect to any MCP server via subprocess (stdio transport)
- InMemoryTransport allows testing with mock servers

### Verification & Testing
- Run: `cargo test -p grapheme-train mcp_client::`
- Expected: 13 tests pass
- Key tests: connect_with_mock_server, call_tool_with_mock, registry_with_mock_server

### Context for Next Task
- StdioTransport.spawn() creates subprocess with piped stdin/stdout
- InMemoryTransport is useful for unit tests (see test_connect_with_mock_server)
- MCPServerRegistry enables multi-server tool routing
- call_tool() checks available_tools before sending request