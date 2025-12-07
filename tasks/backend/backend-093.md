---
id: backend-093
title: Implement full Common Subexpression Elimination optimization pass
status: todo
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

Location: `grapheme-polish/src/lib.rs:490-523`

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

- [ ] Add expression hashing for subexpression identification
- [ ] Track subexpression occurrence counts
- [ ] Implement variable introduction for repeated subexpressions
- [ ] Update `optimize()` to perform actual CSE
- [ ] Add tests for CSE optimization
- [ ] Consider expression form changes (let bindings vs external bindings)

## Acceptance Criteria

- [ ] CSE identifies repeated subexpressions
- [ ] Repeated subexpressions are factored out
- [ ] Optimization produces semantically equivalent expressions
- [ ] Performance benefit is measurable on complex expressions
- [ ] All existing tests pass

## Notes

This is a medium priority task - CSE is useful for optimization but not blocking for basic training functionality.

## Related

- `grapheme-polish/src/lib.rs:490` - placeholder implementation
- Other optimization passes: `ConstantFolding`, `IdentityElimination`
