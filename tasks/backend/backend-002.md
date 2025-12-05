---
id: backend-002
title: 'Review grapheme-math: Math Brain with Typed Nodes (Layer 3)'
status: done
priority: high
tags:
- backend
dependencies:
- api-001
- api-002
assignee: developer
created: 2025-12-05T19:54:41.643365316Z
estimate: ~
complexity: 3
area: backend
---

# Review grapheme-math: Math Brain with Typed Nodes (Layer 3)

> **⚠️ SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**

## Context
Review and expand grapheme-math (Layer 3) to align with GRAPHEME_Math.md specification.
Layer 3 is the learned math brain with typed nodes for mathematical reasoning.

## Objectives
- [x] Review GRAPHEME_Math.md specifications
- [x] Review current grapheme-math implementation
- [x] Implement MathIntent for intent extraction
- [x] Add expression simplification rules
- [x] Add graph transformation capabilities
- [x] Add comprehensive tests
- [x] All tests pass (48 tests total)

## Tasks
- [x] Add MathIntent enum (Compute, Simplify, Solve, Evaluate, Differentiate, Integrate, Factor, Expand)
- [x] Add MathProblem struct for problem representation
- [x] Add SimplificationRule with algebraic identities
- [x] Implement MathTransformer for algebraic transformations
- [x] Add simplify() method with recursive rule application
- [x] Add fold_constants() for constant folding optimization
- [x] Add extract_intent() to MathBrain
- [x] Add create_problem() and solve() to MathBrain
- [x] Add 10 new tests for math transformations

## Acceptance Criteria
✅ **Intent Extraction:**
- MathIntent correctly identifies computation vs simplification
- Differentiates between arithmetic and symbolic expressions

✅ **Simplification Rules:**
- Additive identity: x + 0 = x
- Multiplicative identity: x * 1 = x
- Zero product: x * 0 = 0
- Additive inverse: x - x = 0
- Division identity: x / 1 = x
- Power rules: x^0 = 1, x^1 = x

✅ **Build & Test:**
- `cargo build` succeeds
- `cargo test` passes (48 tests)

## Technical Notes
- MathTransformer applies simplification rules recursively
- Constant folding evaluates pure numeric subexpressions
- MathBrain now includes transformer for algebraic operations
- Intent extraction distinguishes Compute vs Simplify based on symbol presence

## Testing
- [x] 10 new tests added for backend-002 functionality
- [x] All 48 tests pass

## Updates
- 2025-12-05: Task created
- 2025-12-05: Implemented MathIntent, SimplificationRule, MathTransformer - 48 tests pass

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-math/src/lib.rs** - Major expansion with intent and transformation:
  - `MathIntent` enum: Compute, Simplify, Solve, Evaluate, Differentiate, Integrate, Factor, Expand
  - `MathProblem` struct: intent, expression, variable, bounds, expected
  - `SimplificationRule` struct with 7 algebraic identity constants:
    - ADDITIVE_IDENTITY (x + 0 = x)
    - MULTIPLICATIVE_IDENTITY (x * 1 = x)
    - ZERO_PRODUCT (x * 0 = 0)
    - ADDITIVE_INVERSE (x - x = 0)
    - DIVISION_IDENTITY (x / 1 = x)
    - POWER_ONE (x^1 = x)
    - POWER_ZERO (x^0 = 1)
  - `MathTransformer` struct:
    - `simplify(expr)` - recursive algebraic simplification
    - `fold_constants(expr, engine)` - evaluate constant subexpressions
    - `applied_rules()` - get rules used in last simplification
  - `MathBrain` expanded:
    - `extract_intent(expr)` - determine mathematical intent
    - `simplify(expr)` - delegate to transformer
    - `fold_constants(expr)` - delegate to transformer
    - `create_problem(intent, expr)` - construct MathProblem
    - `solve(problem)` - evaluate problem expression

### Causality Impact
- `simplify()` applies rules bottom-up (children simplified before parents)
- Multiple rules can be applied in a single simplification pass
- Intent extraction: expressions with symbols → Simplify, pure numbers → Compute
- Function intents: Derive → Differentiate, Integrate → Integrate, others → Compute

### Dependencies & Integration
- MathTransformer uses MathEngine for constant folding
- MathBrain now owns a MathTransformer instance
- MathIntent integrates with all MathFn variants (Sin, Cos, etc. → Compute)

### Verification & Testing
```bash
cargo build        # Should succeed
cargo test         # 48 tests should pass
```

New tests (10 added):
- `test_math_intent_compute` - Intent for pure arithmetic
- `test_math_intent_simplify` - Intent for symbolic expressions
- `test_simplify_additive_identity` - x + 0 = x
- `test_simplify_multiplicative_identity` - x * 1 = x
- `test_simplify_zero_product` - x * 0 = 0
- `test_simplify_power_zero` - x^0 = 1
- `test_simplify_power_one` - x^1 = x
- `test_fold_constants` - (2+3)*x → 5*x
- `test_math_problem` - Problem creation and solving
- `test_nested_simplification` - (x+0)*1 → x

### Context for Next Task
- **backend-003** (grapheme-polish) can now proceed
- **backend-004** (grapheme-engine) can proceed
- Key new types: `MathIntent`, `MathProblem`, `SimplificationRule`, `MathTransformer`
- The brain now has `simplify()` and `fold_constants()` for optimization
- Intent extraction enables routing to appropriate processing
- Simplification is recursive and handles nested expressions
- Future work: More complex rules (distributive, factoring, etc.)
