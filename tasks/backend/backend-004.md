---
id: backend-004
title: 'Review grapheme-engine: Formal Math Rules Engine (Layer 1)'
status: done
priority: high
tags:
- backend
dependencies:
- api-001
- api-002
assignee: developer
created: 2025-12-05T19:54:51.500120949Z
estimate: ~
complexity: 3
area: backend
---

# Review grapheme-engine: Formal Math Rules Engine (Layer 1)

> **SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**

## Context
Review and expand grapheme-engine (Layer 1) to align with GRAPHEME_Math.md specification.
Layer 1 is the formal math rules engine providing ground-truth symbolic computation.

## Objectives
- [x] Review GRAPHEME_Math.md specifications
- [x] Review current grapheme-engine implementation
- [x] Implement symbolic manipulation (differentiation, substitution)
- [x] Add formal algebraic rules
- [x] Add expression utilities and validation
- [x] Add comprehensive tests
- [x] All tests pass (94 tests total)

## Tasks
- [x] Add expression builder methods (`int()`, `float()`, `symbol()`, `add()`, etc.)
- [x] Add expression analysis methods (`collect_symbols()`, `is_symbolic()`, `depth()`, `node_count()`)
- [x] Implement SymbolicEngine with `substitute()`, `differentiate()`, `evaluate_at()`
- [x] Add FormalRule struct with 14 algebraic rule constants
- [x] Add RuleCategory enum for rule classification
- [x] Implement ExprValidator for expression validation
- [x] Add TrainingDataGenerator for ML training examples
- [x] Extend MathEngine with `unbind()`, `clear_bindings()`, `bound_symbols()`, `is_bound()`
- [x] Add 29 new tests for backend-004 functionality

## Acceptance Criteria
**Expression Utilities:**
- Builder methods for all Expr types
- Symbol collection and analysis
- Expression metrics (depth, node count)

**Symbolic Manipulation:**
- Variable substitution
- Symbolic differentiation with comprehensive rules
- Point evaluation

**Formal Rules:**
- 14 algebraic rules covering identity, inverse, power, and logarithm categories
- Rule categorization via RuleCategory enum

**Validation:**
- Division by zero detection
- Negative square root detection
- Non-positive logarithm argument detection

**Build & Test:**
- `cargo build` succeeds
- `cargo test` passes (94 tests)

## Technical Notes
- SymbolicEngine implements chain rule for nested differentiation
- Differentiation covers: sum, product, quotient, power, trig, exp, ln
- FormalRule uses string-based pattern/replacement for extensibility
- ExprValidator returns ValidationResult with Error/Warning levels
- TrainingDataGenerator produces (input, output) pairs for ML training

## Testing
- [x] 29 new tests added for backend-004 functionality
- [x] All 94 tests pass

## Updates
- 2025-12-05: Task created
- 2025-12-05: Implemented SymbolicEngine, FormalRule, ExprValidator, TrainingDataGenerator - 94 tests pass

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-engine/src/lib.rs** - Major expansion with symbolic engine and utilities:

  **Expression Builders (Expr impl):**
  - `int(n: i64)` - Create integer expression
  - `float(n: f64)` - Create float expression
  - `symbol(name)` - Create symbol expression
  - `add(left, right)` - Create addition
  - `sub(left, right)` - Create subtraction
  - `mul(left, right)` - Create multiplication
  - `div(left, right)` - Create division
  - `pow(base, exp)` - Create power
  - `neg(expr)` - Create negation
  - `func(fn, args)` - Create function call

  **Expression Analysis (Expr impl):**
  - `collect_symbols()` - Returns HashSet<String> of all symbols
  - `is_symbolic()` - True if contains any symbols
  - `is_constant()` - True if only numeric values
  - `depth()` - Maximum nesting depth
  - `node_count()` - Total AST nodes

  **SymbolicEngine:**
  - `substitute(expr, var, replacement)` - Replace variable with expression
  - `differentiate(expr, var)` - Symbolic differentiation
  - `evaluate_at(expr, var, value)` - Evaluate at specific point

  **Differentiation Rules:**
  - Sum rule: d/dx(f + g) = f' + g'
  - Product rule: d/dx(f * g) = f'g + fg'
  - Quotient rule: d/dx(f / g) = (f'g - fg') / g^2
  - Power rule: d/dx(x^n) = n * x^(n-1)
  - Chain rule for nested expressions
  - Trig: sin, cos, tan derivatives
  - Exp/Ln: d/dx(e^x) = e^x, d/dx(ln(x)) = 1/x

  **FormalRule & RuleCategory:**
  - 14 rules: ADDITIVE_IDENTITY, MULTIPLICATIVE_IDENTITY, ZERO_PRODUCT, etc.
  - Categories: Identity, Inverse, Commutative, Associative, Distributive, Power, Logarithm, Trigonometric

  **ExprValidator:**
  - `validate(expr)` - Returns Vec<ValidationResult>
  - Detects: division by zero, negative sqrt, log of non-positive
  - ValidationLevel: Error, Warning

  **TrainingDataGenerator:**
  - `generate_arithmetic(count)` - Random arithmetic expressions
  - `generate_derivatives(count)` - Derivative training pairs
  - `generate_simplifications(count)` - Simplification examples

  **MathEngine Extensions:**
  - `unbind(name)` - Remove single binding
  - `clear_bindings()` - Remove all bindings
  - `bound_symbols()` - Get all bound symbol names
  - `is_bound(name)` - Check if symbol is bound

### Causality Impact
- Differentiation is recursive (chain rule applies automatically)
- Substitution is non-destructive (returns new expression)
- Validation is static (doesn't evaluate, only analyzes structure)
- Training data uses random number generation for variety

### Dependencies & Integration
- No new external dependencies added
- SymbolicEngine is standalone (doesn't need MathEngine)
- FormalRule is data-only (no execution logic yet)
- ExprValidator can be used before evaluation to catch errors
- TrainingDataGenerator uses Expr builders for construction

### Verification & Testing
```bash
cargo build        # Should succeed
cargo test         # 94 tests should pass
```

New tests (29 added):
- Expression builders: `test_expr_builder_int`, `test_expr_builder_float`, `test_expr_builder_symbol`, `test_expr_builder_add`, `test_expr_builder_sub`, `test_expr_builder_mul`, `test_expr_builder_div`, `test_expr_builder_pow`, `test_expr_builder_neg`, `test_expr_builder_func`
- Expression analysis: `test_collect_symbols`, `test_is_symbolic`, `test_is_constant`, `test_expr_depth`, `test_expr_node_count`
- Symbolic engine: `test_substitute_simple`, `test_substitute_nested`, `test_differentiate_constant`, `test_differentiate_variable`, `test_differentiate_sum`, `test_differentiate_product`, `test_differentiate_power`, `test_differentiate_sin`, `test_differentiate_exp`, `test_evaluate_at`
- Validation: `test_validate_division_by_zero`, `test_validate_sqrt_negative`, `test_validate_valid_expression`
- Training: `test_generate_arithmetic`

### Context for Next Task
- **backend-005** (grapheme-train) can now proceed
- Key new types: `SymbolicEngine`, `FormalRule`, `RuleCategory`, `ExprValidator`, `ValidationResult`, `TrainingDataGenerator`
- Expression builders make constructing test expressions much easier
- SymbolicEngine provides foundation for symbolic computation
- FormalRule provides patterns but no auto-application yet (future work)
- Validation should be called before evaluation for safety
- TrainingDataGenerator provides ML training examples
- Future work: Rule auto-application, more trig rules, integration
