---
id: backend-093
title: Implement full Common Subexpression Elimination optimization pass
status: done
priority: medium
tags:
- backend
- optimization
- polish-notation
dependencies: []
assignee: developer
created: 2025-12-07T20:00:00Z
estimate: ~
complexity: 3
area: backend
---

# Implement full Common Subexpression Elimination optimization pass

## Context

The `CommonSubexpressionElimination` optimization pass in `grapheme-polish` is currently a placeholder:

```rust
/// NOTE: This is a placeholder - full CSE requires mutable state and variable introduction
pub struct CommonSubexpressionElimination;

impl OptimizationPass for CommonSubexpressionElimination {
    fn optimize(&self, expr: &Expr) -> Expr {
        // For now, just return the expression - full CSE requires mutable state
        // and variable introduction which is more complex
        match expr {
            Expr::Value(_) => expr.clone(),
            // ... recursively returns same expression
        }
    }
}
```

Location: `grapheme-polish/src/lib.rs:489-647`

## Objectives

Implement proper CSE that:
1. Identifies repeated subexpressions
2. Introduces temporary variables for shared subexpressions
3. Reduces redundant computation

## Example

Input:
```
(* (+ a b) (+ a b))
```

After CSE:
```
(let ((t1 (+ a b))) (* t1 t1))
```

Or in a form compatible with current Expr:
```
(* t1 t1) where bindings = {t1: (+ a b)}
```

## Tasks

- [x] Add expression hashing for subexpression identification
- [x] Track subexpression occurrence counts
- [x] Implement variable introduction for repeated subexpressions
- [x] Update `optimize()` to perform actual CSE
- [x] Add tests for CSE optimization
- [x] Consider expression form changes (let bindings vs external bindings)

## Acceptance Criteria

- [x] CSE identifies repeated subexpressions
- [x] Repeated subexpressions are factored out
- [x] Optimization produces semantically equivalent expressions
- [x] Performance benefit is measurable on complex expressions
- [x] All existing tests pass (588 tests)

## Notes

This is a medium priority task - CSE is useful for optimization but not blocking for basic training functionality.

## Related

- `grapheme-polish/src/lib.rs:489` - full CSE implementation
- Other optimization passes: `ConstantFolding`, `IdentityElimination`

## Implementation Summary

Implemented full CSE with the following features:

1. **Expression Hashing**: Uses `expr_to_polish()` as canonical form for hashing (since `Expr` contains `f64` which can't implement `Hash`)

2. **Occurrence Counting**: `count_subexpressions()` recursively counts all non-trivial subexpressions

3. **Variable Introduction**: `replace_common()` replaces repeated subexpressions with `_cseN` variables

4. **Bindings Storage**: `CommonSubexpressionElimination::bindings()` returns the CSE variable bindings after optimization

5. **Configurable Threshold**: `with_min_occurrences(n)` allows setting minimum occurrences before CSE is applied (default: 2)

**Key Files Changed:**
- `grapheme-polish/src/lib.rs` - Added ~160 lines for full CSE implementation

**New Tests (7 total):**
- `test_cse_simple_duplicate` - Basic CSE with two identical subexpressions
- `test_cse_no_duplicates` - Ensures no change when no duplicates exist
- `test_cse_triple_occurrence` - CSE with 3+ occurrences
- `test_cse_nested_duplicates` - Complex nested structures
- `test_cse_with_functions` - Function call CSE
- `test_cse_min_occurrences` - Custom threshold testing
- `test_cse_clears_bindings` - State management testing