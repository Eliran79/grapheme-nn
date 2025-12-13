---
id: backend-011
title: 'Complete Level 7: Equation solving'
status: done
priority: low
tags:
- backend
- curriculum
- symbolic
dependencies:
- backend-010
assignee: developer
created: 2025-12-05T21:39:55.146405643Z
estimate: ~
complexity: 4
area: backend
---

# Complete Level 7: Equation solving

## Context
Level 7 of the GRAPHEME curriculum introduces equation solving - algebraic manipulation to isolate variables. Currently, `CurriculumLevel::Solve` exists but generates placeholder data. This task implements symbolic equation solving in the SymbolicEngine.

**Current state**: Level 7 returns placeholder/dummy solutions. The curriculum structure exists but lacks actual solving logic.

## Objectives
- Implement symbolic equation solving in SymbolicEngine
- Support linear equations (ax + b = c)
- Support quadratic equations (ax² + bx + c = 0)
- Support simple algebraic manipulation (isolation)
- Generate proper equation/solution pairs for training

## Tasks
- [ ] Add `Equation` type to represent equations (lhs = rhs)
- [ ] Add `solve()` method to SymbolicEngine
- [ ] Implement linear equation solver
- [ ] Implement quadratic formula for quadratics
- [ ] Implement algebraic isolation (move terms)
- [ ] Implement simplification after solving
- [ ] Update DataGenerator to use real equation solving
- [ ] Add unit tests for each equation type
- [ ] Add benchmarks

## Acceptance Criteria
✅ **Correctness:**
- Solves linear equations correctly
- Quadratic formula produces correct roots
- Handles no-solution and infinite-solution cases

✅ **Coverage:**
- Linear: ax + b = c, ax + b = dx + e
- Quadratic: ax² + bx + c = 0 (real roots)
- Simple rational: a/x = b

✅ **Integration:**
- Works with existing Expr representation
- DataGenerator produces valid Level 7 samples
- Solutions can be verified by substitution

## Technical Notes

### Algorithm Pseudocode
```rust
/// Represents an equation: lhs = rhs
pub struct Equation {
    pub lhs: Expr,
    pub rhs: Expr,
}

/// Solution to an equation
pub enum Solution {
    /// Single solution: x = value
    Single(Expr),
    /// Multiple solutions (quadratic)
    Multiple(Vec<Expr>),
    /// No real solution
    NoSolution,
    /// Infinitely many solutions (identity)
    Infinite,
}

impl SymbolicEngine {
    /// Solve equation for variable
    pub fn solve(&self, eq: &Equation, var: &str) -> Result<Solution, SolveError> {
        // Move everything to one side: f(x) = 0
        let expr = self.simplify(&Expr::sub(eq.lhs.clone(), eq.rhs.clone()))?;

        // Classify equation type
        let degree = self.polynomial_degree(&expr, var);

        match degree {
            0 => {
                // No variable: constant = 0
                if self.is_zero(&expr) {
                    Ok(Solution::Infinite)
                } else {
                    Ok(Solution::NoSolution)
                }
            }
            1 => self.solve_linear(&expr, var),
            2 => self.solve_quadratic(&expr, var),
            _ => Err(SolveError::DegreeToohigh(degree))
        }
    }

    /// Solve linear equation: ax + b = 0 => x = -b/a
    fn solve_linear(&self, expr: &Expr, var: &str) -> Result<Solution, SolveError> {
        let a = self.coefficient_of(expr, var, 1)?;  // coefficient of x
        let b = self.coefficient_of(expr, var, 0)?;  // constant term

        if self.is_zero(&a) {
            return if self.is_zero(&b) {
                Ok(Solution::Infinite)
            } else {
                Ok(Solution::NoSolution)
            };
        }

        // x = -b/a
        let solution = self.simplify(&Expr::div(
            Expr::neg(b),
            a
        ))?;

        Ok(Solution::Single(solution))
    }

    /// Solve quadratic: ax² + bx + c = 0
    /// Using quadratic formula: x = (-b ± √(b²-4ac)) / 2a
    fn solve_quadratic(&self, expr: &Expr, var: &str) -> Result<Solution, SolveError> {
        let a = self.coefficient_of(expr, var, 2)?;
        let b = self.coefficient_of(expr, var, 1)?;
        let c = self.coefficient_of(expr, var, 0)?;

        // Discriminant: b² - 4ac
        let discriminant = self.simplify(&Expr::sub(
            Expr::pow(b.clone(), Expr::int(2)),
            Expr::mul(Expr::int(4), Expr::mul(a.clone(), c.clone()))
        ))?;

        // Check discriminant sign (simplified case for numeric)
        if let Some(d) = self.evaluate_numeric(&discriminant) {
            if d < 0.0 {
                return Ok(Solution::NoSolution);  // No real roots
            }

            let sqrt_d = Expr::func("sqrt", vec![discriminant]);
            let two_a = Expr::mul(Expr::int(2), a);

            let x1 = self.simplify(&Expr::div(
                Expr::add(Expr::neg(b.clone()), sqrt_d.clone()),
                two_a.clone()
            ))?;

            let x2 = self.simplify(&Expr::div(
                Expr::sub(Expr::neg(b), sqrt_d),
                two_a
            ))?;

            if d == 0.0 {
                Ok(Solution::Single(x1))
            } else {
                Ok(Solution::Multiple(vec![x1, x2]))
            }
        } else {
            // Keep symbolic discriminant
            Err(SolveError::SymbolicDiscriminant)
        }
    }
}
```

### Key Design Decisions
- Equation type separates lhs and rhs (not just expression = 0)
- Solution enum handles all cases (single, multiple, none, infinite)
- Start with polynomial equations, expand to transcendental later
- Use existing `simplify()` for algebraic manipulation
- Verify solutions by substitution back into original

### Files to Modify
- `grapheme-engine/src/lib.rs`: Add `Equation`, `Solution`, `solve()`
- `grapheme-train/src/lib.rs`: Update Level 7 generation
- `grapheme-engine/benches/engine_bench.rs`: Add solving benchmarks

## Testing
- [ ] Test linear: x + 5 = 10 → x = 5
- [ ] Test linear: 2x - 3 = 7 → x = 5
- [ ] Test linear: 3x + 2 = x + 6 → x = 2
- [ ] Test quadratic: x² - 4 = 0 → x = ±2
- [ ] Test quadratic: x² + 2x + 1 = 0 → x = -1 (repeated)
- [ ] Test quadratic: x² + 1 = 0 → no real solution
- [ ] Test identity: x = x → infinite solutions
- [ ] Verify solutions by substitution

## Updates
- 2025-12-05: Task created from testing-002 recommendations

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Added `Equation` struct (lhs, rhs) for representing equations
- Added `Solution` enum (Single, Multiple, NoSolution, Infinite)
- Added `SolveError` enum and `SolveResult` type
- Added `solve()` method to SymbolicEngine
- Added `simplify()` method with constant folding and algebraic rules
- Added `polynomial_degree()` for determining equation type
- Added `solve_linear()` for ax + b = 0
- Added `solve_quadratic()` using quadratic formula
- Added `extract_linear_coefficients()` and `extract_quadratic_coefficients()`
- Added `evaluate_numeric()` for constant evaluation
- Added 13 equation solving tests (63 total tests in grapheme-engine)

### Causality Impact
- Level 7 curriculum becomes functional
- Enables training on equation solving
- Completes the 7-level curriculum system
- Adds simplification capability to SymbolicEngine

### Dependencies & Integration
- Uses backend-010 integration methods (SymbolicEngine)
- Works with existing Expr representation
- Solution can be verified by substitution

### Verification & Testing
- Run `cargo test -p grapheme-engine` for unit tests
- All 63 tests passing with 0 warnings
