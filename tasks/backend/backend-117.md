---
id: backend-117
title: Add activation gradients to CodeGraph (grapheme-code)
status: done
priority: high
tags:
- backend
- gradient
- cognitive
- code
dependencies: []
assignee: developer
created: 2025-12-09T10:07:00.117508104Z
estimate: 2h
complexity: 6
area: backend
---

# Add activation gradients to CodeGraph (grapheme-code)

> **⚠️ SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**

## Context

The `CodeGraph` brain in `grapheme-code` uses a plain enum `CodeNode` as its node type. Without an `activation: f32` field, there's no learnable parameter for gradients to flow through. **This brain will NEVER learn** from structural loss until this is fixed.

This is part of the Backend-104 gradient fix pattern that was successfully applied to:
- `GraphemeGraph` (grapheme-core) - ✅ Working
- `MathGraph` (grapheme-math) - ✅ Working

## Objectives

1. Refactor `CodeNode` enum to struct-with-type pattern
2. Add activation field for gradient flow
3. Implement structural loss functions in grapheme-train
4. Verify gradient descent works for CodeGraph

## Tasks

- [ ] **Step 1: Refactor CodeNode in grapheme-code/src/lib.rs**
  ```rust
  // BEFORE:
  pub enum CodeNode {
      Function { name: String, params: Vec<String>, ... },
      Variable { name: String, var_type: Option<String> },
      Literal(LiteralValue),
      ...
  }

  // AFTER:
  pub enum CodeNodeType {
      Function { name: String, params: Vec<String>, ... },
      Variable { name: String, var_type: Option<String> },
      Literal(LiteralValue),
      ...
  }

  pub struct CodeNode {
      pub node_type: CodeNodeType,
      pub activation: f32,  // Gradient flows here!
  }

  impl CodeNode {
      pub fn new(node_type: CodeNodeType) -> Self {
          let activation = Self::type_activation(&node_type);
          Self { node_type, activation }
      }

      fn type_activation(node_type: &CodeNodeType) -> f32 {
          match node_type {
              CodeNodeType::Function { .. } => 0.8,
              CodeNodeType::Variable { .. } => 0.3,
              CodeNodeType::Literal(_) => 0.1,
              CodeNodeType::BinaryOp(_) => 0.5,
              // ... etc
          }
      }
  }
  ```

- [ ] **Step 2: Update all CodeNode usages in grapheme-code**
  - Update `CodeGraph::add_node()` calls
  - Update pattern matches on `CodeNode`
  - Update `from_simple_expr()` and other constructors

- [ ] **Step 3: Add structural loss functions in grapheme-train/src/lib.rs**
  ```rust
  pub fn compute_structural_loss_code(
      predicted: &CodeGraph,
      target: &CodeGraph,
      config: &StructuralLossConfig,
  ) -> StructuralLossResult { ... }

  fn compute_soft_node_costs_code(...) -> (Vec<f32>, usize, usize) { ... }

  fn compute_activation_gradients_code(...) -> Vec<f32> { ... }
  ```

- [ ] **Step 4: Add training binary (optional)**
  - Create `grapheme-train/src/bin/train_code_sabag.rs`

## Acceptance Criteria

✅ **Criteria 1:** `CodeNode` has `activation: f32` field
✅ **Criteria 2:** `compute_structural_loss_code` returns non-zero `activation_gradients`
✅ **Criteria 3:** All existing tests pass
✅ **Criteria 4:** Gradient descent test shows loss decreasing for CodeGraph

## Technical Notes

**Gradient Chain (Backend-104 pattern):**
```
activation → cost → loss → ∂L/∂activation
```

**Cost formula:**
```
cost = 0.3 * type_cost + 0.5 * activation_cost + 0.2 * degree_cost
```

**Reference implementations:**
- `grapheme-core/src/lib.rs`: GraphemeNode with activation
- `grapheme-math/src/lib.rs`: MathNode with activation
- `grapheme-train/src/lib.rs`: `compute_activation_gradients()`, `compute_activation_gradients_math()`

## Testing

- [ ] Build passes: `cargo build`
- [ ] All tests pass: `cargo test`
- [ ] Verify activation gradients are non-zero in structural loss output

## Version Control

- [ ] Build, test, and verify before committing
- [ ] Commit with message: `feat(code): Add activation gradients to CodeGraph (backend-117)`

## Updates
- 2025-12-09: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-code/src/lib.rs**:
  - Renamed `CodeNode` enum to `CodeNodeType` (lines 746-762)
  - Created new `CodeNode` struct with `node_type: CodeNodeType` and `activation: f32` fields (lines 765-768)
  - Added `CodeNode::new(CodeNodeType) -> Self` constructor with `type_activation()` method (lines 770-797)
  - Updated ALL pattern matches from `CodeNode::Xxx` to `CodeNode { node_type: CodeNodeType::Xxx, .. }`
  - Updated ALL node creation from `CodeNode::Xxx` to `CodeNode::new(CodeNodeType::Xxx)`
  - Updated all 4 language-specific mapping functions: `map_rust_node`, `map_python_node`, `map_javascript_node`, `map_c_node`

- **grapheme-train/Cargo.toml**:
  - Added dependency: `grapheme-code = { path = "../grapheme-code" }`

- **grapheme-train/src/lib.rs**:
  - Added import: `use grapheme_code::{CodeGraph, CodeNode, CodeNodeType};`
  - Added `compute_structural_loss_code()` function (lines 2777-2824)
  - Added `compute_soft_node_costs_code()` function (lines 2826-2871)
  - Added `code_node_type_cost()` helper (lines 2873-2891)
  - Added `code_node_category()` helper (lines 2893-2908)
  - Added `compute_activation_gradients_code()` function (lines 2910-2951)
  - Added `compute_differentiable_edge_cost_code()` function (lines 2953-3014)
  - Added `test_code_graph_activation_gradients` test (lines 6528-6578)

### Verification & Testing
```bash
# All tests pass
cargo test  # 612 tests pass

# Specific gradient test
cargo test --package grapheme-train test_code_graph_activation_gradients -- --nocapture
```

### Activation Values by Node Type
- `Module` → 0.9
- `Function` → 0.8
- `Call` → 0.7
- `If/Loop` → 0.6
- `BinaryOp/Return/Assignment` → 0.5
- `Variable/UnaryOp/Type` → 0.4
- `Identifier/Block/ExprStmt` → 0.3
- `Literal` → 0.2
- `Comment` → 0.1