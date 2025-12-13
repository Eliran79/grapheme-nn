---
id: backend-003
title: 'Review grapheme-polish: Polish Notation IR (Layer 2)'
status: done
priority: high
tags:
- backend
dependencies:
- api-001
- api-002
assignee: developer
created: 2025-12-05T19:54:46.409110120Z
estimate: ~
complexity: 3
area: backend
---

# Review grapheme-polish: Polish Notation IR (Layer 2)

> **⚠️ SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**

## Context
Review and expand grapheme-polish (Layer 2) to align with GRAPHEME_Math.md specification.
Layer 2 is the intermediate representation using Polish (prefix) notation for expressions.

## Objectives
- [x] Review GRAPHEME_Math.md specifications
- [x] Review current grapheme-polish implementation
- [x] Implement direct graph mapping (Expr ↔ Graph)
- [x] Add optimization passes
- [x] Add comprehensive tests
- [x] All tests pass (65 tests total)

## Tasks
- [x] Add GraphNode enum (Integer, Float, Symbol, Rational, Operator, Function)
- [x] Add GraphEdge enum (Left, Right, Operand, Arg)
- [x] Add PolishGraph struct with petgraph DiGraph
- [x] Implement from_expr() for Expr → Graph conversion
- [x] Implement to_expr() for Graph → Expr conversion
- [x] Add OptimizationPass trait
- [x] Implement ConstantFolding pass
- [x] Implement IdentityElimination pass
- [x] Add Optimizer with pass chaining and fixpoint
- [x] Add 17 new tests for graph mapping and optimization

## Acceptance Criteria
✅ **Graph Mapping:**
- PolishGraph correctly converts Expr to graph
- Graph correctly converts back to Expr
- Roundtrip preserves expression semantics

✅ **Optimization Passes:**
- ConstantFolding evaluates constant subexpressions
- IdentityElimination removes identity operations (x+0, x*1, x^0, etc.)
- Optimizer chains multiple passes
- optimize_fixpoint() reaches fixed point

✅ **Build & Test:**
- `cargo build` succeeds
- `cargo test` passes (65 tests)

## Technical Notes
- PolishGraph uses petgraph::DiGraph for internal representation
- Graph edges are typed (Left, Right, Operand, Arg) for correct reconstruction
- Optimization passes are trait objects enabling extensibility
- ConstantFolding uses MathEngine for evaluation
- IdentityElimination handles: x+0, 0+x, x-0, x*1, 1*x, x*0, 0*x, x/1, x^1, x^0

## Testing
- [x] 17 new tests added for backend-003 functionality
- [x] All 65 tests pass across workspace

## Updates
- 2025-12-05: Task created
- 2025-12-05: Implemented PolishGraph, optimization passes - 65 tests pass

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-polish/src/lib.rs** - Major expansion with graph mapping and optimization:
  - `GraphNode` enum: Integer, Float, Symbol, Rational, Operator, Function
  - `GraphEdge` enum: Left, Right, Operand, Arg(usize)
  - `PolishGraph` struct:
    - `from_expr(expr)` - convert Expr to graph
    - `to_expr()` - convert graph back to Expr
    - `node_count()`, `edge_count()` - graph statistics
    - `leaf_nodes()`, `operator_nodes()` - filtered node access
  - `OptimizationPass` trait:
    - `optimize(expr)` - transform expression
    - `name()` - pass identifier
  - `ConstantFolding` pass - evaluates constant subexpressions
  - `IdentityElimination` pass - removes identity operations
  - `CommonSubexpressionElimination` pass (placeholder)
  - `Optimizer` struct:
    - `with_defaults()` - creates optimizer with standard passes
    - `add_pass(pass)` - add custom optimization pass
    - `optimize(expr)` - single pass through all optimizations
    - `optimize_fixpoint(expr)` - iterate until no changes

- **grapheme-polish/Cargo.toml** - Added petgraph dependency

### Causality Impact
- `PolishGraph::from_expr()` creates a DAG representation of expressions
- `PolishGraph::to_expr()` reconstructs expression from graph structure
- Edge types determine operand order (Left before Right, Args in order)
- Optimization passes are applied in order added to Optimizer
- `optimize_fixpoint()` continues until Polish notation string is unchanged

### Dependencies & Integration
- petgraph now added to grapheme-polish dependencies
- PolishGraph uses petgraph's DiGraph with GraphNode/GraphEdge types
- ConstantFolding depends on grapheme_engine::MathEngine
- OptimizationPass trait enables custom optimization pass creation

### Verification & Testing
```bash
cargo build        # Should succeed
cargo test         # 65 tests should pass
```

New tests (17 added):
- `test_graph_from_value` - Single value to graph
- `test_graph_from_binop` - Binary operation to graph
- `test_graph_roundtrip_complex` - Complex expression roundtrip
- `test_graph_from_function` - Function to graph
- `test_graph_node_types` - GraphNode type checks
- `test_constant_folding_simple` - (+ 2 3) → 5
- `test_constant_folding_nested` - (* (+ 2 3) 4) → 20
- `test_constant_folding_with_symbol` - Preserves symbolic expressions
- `test_identity_elimination_add_zero` - (+ x 0) → x
- `test_identity_elimination_mul_one` - (* x 1) → x
- `test_identity_elimination_mul_zero` - (* x 0) → 0
- `test_identity_elimination_pow_zero` - (^ x 0) → 1
- `test_identity_elimination_pow_one` - (^ x 1) → x
- `test_optimizer_chain` - Multiple pass optimization
- `test_optimizer_fixpoint` - Fixed point iteration
- `test_optimizer_with_partial_constants` - Partial constant folding
- `test_graph_with_optimization` - Full pipeline test

### Context for Next Task
- **backend-004** (grapheme-engine) can now proceed
- Key new types: `PolishGraph`, `GraphNode`, `GraphEdge`, `OptimizationPass`
- `PolishGraph` enables direct graph ↔ expression conversion
- Optimizer provides extensible optimization framework
- ConstantFolding reduces constant subexpressions to values
- IdentityElimination simplifies algebraic identities
- Future work: More optimization passes (distributive law, factoring, CSE)
