---
id: backend-082
title: Add tree-sitter integration for grapheme-code multi-language parsing
status: done
priority: low
tags:
- backend
dependencies: []
assignee: developer
created: 2025-12-06T14:54:00.200331446Z
estimate: ~
complexity: 3
area: backend
---

# Add tree-sitter integration for grapheme-code multi-language parsing

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
The grapheme-code brain currently has a simple hand-written expression parser in `CodeGraph::from_simple_expr()`. For production use, proper AST parsing is needed via tree-sitter, which provides incremental, error-tolerant parsing for many languages.

The crate docstring already notes: "Future enhancements: Tree-sitter integration for multi-language parsing"

## Objectives
- Add tree-sitter as a dependency for robust multi-language parsing
- Parse Rust, Python, JavaScript, and C code into CodeGraph structures
- Support incremental parsing for editor integration

## Tasks
- [ ] Add tree-sitter and language grammar dependencies to Cargo.toml
- [ ] Create `TreeSitterParser` struct wrapping tree-sitter
- [ ] Implement `parse_rust()` converting tree-sitter AST to CodeGraph
- [ ] Implement `parse_python()` for Python code
- [ ] Implement `parse_javascript()` for JS/TS code
- [ ] Implement `parse_c()` for C code
- [ ] Update CodeBrain.parse() to use tree-sitter based on detected language
- [ ] Add tests with real code snippets

## Acceptance Criteria
✅ **Multi-Language Parsing:**
- Successfully parse valid Rust, Python, JS, C code into CodeGraph

✅ **Error Tolerance:**
- Partial results returned for code with syntax errors

## Technical Notes
### Dependencies to add:
```toml
[dependencies]
tree-sitter = "0.20"
tree-sitter-rust = "0.20"
tree-sitter-python = "0.20"
tree-sitter-javascript = "0.20"
tree-sitter-c = "0.20"
```

### Tree-sitter integration pattern:
```rust
use tree_sitter::{Parser, Language};

pub struct TreeSitterParser {
    parser: Parser,
}

impl TreeSitterParser {
    pub fn parse_rust(code: &str) -> CodeGraphResult<CodeGraph> {
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_rust::language())?;
        let tree = parser.parse(code, None)?;
        Self::convert_tree_to_code_graph(tree.root_node(), code)
    }
}
```

### Node mapping examples:
- `function_item` (Rust) → `CodeNode::Function`
- `let_declaration` (Rust) → `CodeNode::Variable`
- `binary_expression` → `CodeNode::BinaryOp`

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