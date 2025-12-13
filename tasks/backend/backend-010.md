---
id: backend-010
title: 'Complete Level 6: Symbolic integration'
status: done
priority: low
tags:
- backend
- curriculum
- symbolic
dependencies:
- backend-004
assignee: developer
created: 2025-12-05T21:39:50.956310279Z
estimate: ~
complexity: 4
area: backend
---

# Complete Level 6: Symbolic integration

## Context
Level 6 of the GRAPHEME curriculum introduces symbolic integration. Currently, the `CurriculumLevel::Integration` exists but generates placeholder data. This task implements proper symbolic integration rules in the SymbolicEngine.

**Current state**: Level 6 returns placeholder/dummy integration results. The DataGenerator has `generate_level()` support but needs actual integration logic.

## Objectives
- Implement symbolic integration rules in SymbolicEngine
- Support basic indefinite integrals (polynomials, trig, exp, log)
- Generate proper input/output pairs for training
- Extend `grapheme-engine` with `integrate()` method

## Tasks
- [ ] Add `integrate()` method to SymbolicEngine
- [ ] Implement power rule: ∫x^n dx = x^(n+1)/(n+1)
- [ ] Implement constant rule: ∫k dx = kx
- [ ] Implement trig integrals: ∫sin(x) = -cos(x), ∫cos(x) = sin(x)
- [ ] Implement exponential: ∫e^x = e^x
- [ ] Implement logarithm: ∫1/x = ln|x|
- [ ] Implement sum rule: ∫(f+g) = ∫f + ∫g
- [ ] Implement constant multiple: ∫kf = k∫f
- [ ] Update DataGenerator to use real integration
- [ ] Add unit tests for each integration rule
- [ ] Add benchmarks

## Acceptance Criteria
✅ **Correctness:**
- Power rule handles all integer exponents (including n=-1 special case)
- Trig integrals produce correct antiderivatives
- Sum/difference of integrals works correctly

✅ **Coverage:**
- Supports polynomials up to degree 5
- Supports sin, cos, tan, exp, ln
- Handles nested expressions (e.g., ∫(x² + sin(x))dx)

✅ **Integration:**
- Works with existing Expr representation
- DataGenerator produces valid Level 6 samples
- MathGraph conversion works for integral expressions

## Technical Notes

### Algorithm Pseudocode
```rust
impl SymbolicEngine {
    /// Compute indefinite integral of expr with respect to var
    pub fn integrate(&self, expr: &Expr, var: &str) -> Result<Expr, IntegrationError> {
        match expr {
            // Constant: ∫k dx = kx
            Expr::Int(n) => Ok(Expr::mul(Expr::int(*n), Expr::symbol(var))),
            Expr::Float(f) => Ok(Expr::mul(Expr::float(*f), Expr::symbol(var))),

            // Variable: ∫x dx = x²/2
            Expr::Symbol(s) if s == var => {
                Ok(Expr::div(
                    Expr::pow(Expr::symbol(var), Expr::int(2)),
                    Expr::int(2)
                ))
            }

            // Power rule: ∫x^n dx = x^(n+1)/(n+1) for n ≠ -1
            Expr::Pow(base, exp) if is_symbol(base, var) => {
                if let Expr::Int(-1) = **exp {
                    // Special case: ∫x^(-1) dx = ln|x|
                    Ok(Expr::func("ln", vec![Expr::func("abs", vec![Expr::symbol(var)])]))
                } else {
                    let new_exp = add_one(exp);
                    Ok(Expr::div(
                        Expr::pow(Expr::symbol(var), new_exp.clone()),
                        new_exp
                    ))
                }
            }

            // Sum rule: ∫(f + g) = ∫f + ∫g
            Expr::Add(left, right) => {
                Ok(Expr::add(
                    self.integrate(left, var)?,
                    self.integrate(right, var)?
                ))
            }

            // Constant multiple: ∫kf = k∫f
            Expr::Mul(left, right) if !contains_var(left, var) => {
                Ok(Expr::mul(
                    (**left).clone(),
                    self.integrate(right, var)?
                ))
            }

            // Trig integrals
            Expr::Func(name, args) => match name.as_str() {
                "sin" if args.len() == 1 && is_symbol(&args[0], var) => {
                    Ok(Expr::neg(Expr::func("cos", vec![Expr::symbol(var)])))
                }
                "cos" if args.len() == 1 && is_symbol(&args[0], var) => {
                    Ok(Expr::func("sin", vec![Expr::symbol(var)]))
                }
                "exp" if args.len() == 1 && is_symbol(&args[0], var) => {
                    Ok(Expr::func("exp", vec![Expr::symbol(var)]))
                }
                _ => Err(IntegrationError::CannotIntegrate)
            }

            _ => Err(IntegrationError::CannotIntegrate)
        }
    }
}
```

### Key Design Decisions
- Return `Result<Expr, IntegrationError>` - not all expressions are integrable
- No constant of integration (C) - training focuses on antiderivative form
- Build on existing `differentiate()` for verification (∫f' = f)
- Start with elementary functions, expand later

### Files to Modify
- `grapheme-engine/src/lib.rs`: Add `integrate()` to SymbolicEngine
- `grapheme-train/src/lib.rs`: Update Level 6 generation to use real integration
- `grapheme-engine/benches/engine_bench.rs`: Add integration benchmarks

## Testing
- [ ] Test power rule for n = 0, 1, 2, 3, -1, -2
- [ ] Test trig integrals: sin, cos
- [ ] Test exponential and logarithm
- [ ] Test sum rule with multiple terms
- [ ] Test constant multiple rule
- [ ] Verify ∫f' = f relationship with differentiation
- [ ] Benchmark integration performance

## Updates
- 2025-12-05: Task created from testing-002 recommendations

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added `IntegrationError` enum in `grapheme-engine/src/lib.rs`
- Added `integrate()` method to SymbolicEngine with ~240 lines
- Implemented: power rule, constant rule, sum/difference rules
- Implemented: sin, cos, exp, tan integrals
- Implemented: 1/x → ln|x| special case
- Added helper methods: `is_var()`, `contains_var()`, `get_int_value()`, `is_int()`
- Added 18 integration tests (50 total tests in grapheme-engine)

### Causality Impact
- Level 6 curriculum becomes functional
- Enables training on integration problems
- Feeds into Level 7 (equation solving)

### Dependencies & Integration
- Depends on backend-004 (SymbolicEngine differentiation) - works with it
- Required by backend-011 (equation solving) - now unblocked
- Integrates with DataGenerator curriculum system

### Verification & Testing
- Run `cargo test -p grapheme-engine` for unit tests
- All 50 tests passing with 0 warnings
- Integration rules match mathematical definitions