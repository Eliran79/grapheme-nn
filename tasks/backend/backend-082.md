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
- [x] Add tree-sitter and language grammar dependencies to Cargo.toml
- [x] Create `TreeSitterParser` struct wrapping tree-sitter
- [x] Implement `parse_rust()` converting tree-sitter AST to CodeGraph
- [x] Implement `parse_python()` for Python code
- [x] Implement `parse_javascript()` for JS/TS code
- [x] Implement `parse_c()` for C code
- [x] Update CodeBrain.parse() to use tree-sitter based on detected language
- [x] Add tests with real code snippets

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
- [x] Write unit tests for new functionality
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete
- [x] Consider edge cases and error conditions

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
  - Run `cargo build --release` (or `cargo build` for debug)
  - Run `cargo test` to ensure tests pass
  - **Actually run/execute the code** to verify runtime behavior
  - Fix all errors, warnings, and runtime issues
- [x] Commit changes incrementally with clear messages
- [x] Use descriptive commit messages that explain the "why"
- [x] Consider creating a feature branch for complex changes
- [x] Review changes before committing

**Testing requirements by change type:**
- Code changes: Build + test + **run the actual program/command** to verify behavior
- Bug fixes: Verify the bug is actually fixed by running the code, not just compiling
- New features: Test the feature works as intended by executing it
- Minor changes: At minimum build, check warnings, and run basic functionality

## Updates
- 2025-12-06: Task created
- 2025-12-13: Task completed - added tree-sitter multi-language parsing

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created `/home/user/grapheme-nn/grapheme-code/src/tree_sitter_parser.rs` (~600 lines)
- Added module declaration and re-export in lib.rs
- Updated CodeBrain with `parse_code()` using tree-sitter
- Added dependencies: tree-sitter 0.22, tree-sitter-rust/python/javascript/c 0.21
- Key exports: `TreeSitterParser`, methods `parse_rust()`, `parse_python()`, `parse_javascript()`, `parse_c()`

### Causality Impact
- `TreeSitterParser::parse(code, language)` → tree-sitter AST → `CodeGraph`
- `CodeBrain::parse_code(code)` → auto-detects language → uses TreeSitterParser
- `CodeBrain::has_syntax_errors(code)` → uses tree-sitter for error detection
- `CodeBrain::get_syntax_errors(code)` → returns (row, column) positions

### Dependencies & Integration
- Uses tree-sitter crates from crates.io (v0.21-0.22 range)
- Integrates with existing CodeGraph and CodeNode types
- CodeBrain now has language-aware parsing (detect_language → parse)
- Language fallback: Generic uses simple expression parser

### Verification & Testing
- Run `cargo test -p grapheme-code` to verify 21 tests pass (12 new tree-sitter tests)
- Run `cargo clippy -p grapheme-code -- -D warnings` for 0 warnings
- Tests cover: Rust/Python/JS/C parsing, error detection, error positions, arrow functions

### Context for Next Task
- Tree-sitter AST nodes mapped to CodeNode types (function_item → Function, etc.)
- Error-tolerant parsing: partial results returned for invalid code
- `has_errors()` and `get_error_positions()` for syntax validation
- Language detection heuristics in `detect_language()` (fn→Rust, def→Python, etc.)
