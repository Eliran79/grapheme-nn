//! # grapheme-engine
//!
//! Layer 1: The ground truth for GRAPHEME mathematical processing.
//!
//! This crate provides:
//! - Formal algebraic rules
//! - Symbolic manipulation
//! - Numerical evaluation
//! - Training data generation
//! - Output validation
//!
//! All other layers depend on this for correctness verification.

// Allow method names like add, sub, mul, div for expression builders (not implementing std traits)
#![allow(clippy::should_implement_trait)]
// Allow &self in recursive methods for API consistency
#![allow(clippy::only_used_in_recursion)]

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

/// Errors that can occur during engine operations
#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    #[error("Undefined symbol: {0}")]
    UndefinedSymbol(String),
    #[error("Type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },
    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}

/// Result type for engine operations
pub type EngineResult<T> = Result<T, EngineError>;

/// Errors that can occur during symbolic integration
#[derive(Error, Debug)]
pub enum IntegrationError {
    #[error("Cannot integrate expression: {0}")]
    CannotIntegrate(String),
    #[error("Division by zero in antiderivative")]
    DivisionByZero,
    #[error("Integration not supported for this expression type")]
    NotSupported,
}

/// Result type for integration operations
pub type IntegrationResult<T> = Result<T, IntegrationError>;

/// Mathematical operators supported by the engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Neg,
}

/// Mathematical functions supported by the engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathFn {
    Sin,
    Cos,
    Tan,
    Log,
    Ln,
    Exp,
    Sqrt,
    Abs,
    Floor,
    Ceil,
    // Calculus
    Derive,
    Integrate,
}

/// A mathematical value that can be symbolic or numeric
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Integer value
    Integer(i64),
    /// Floating-point value
    Float(f64),
    /// Symbolic value (variable)
    Symbol(String),
    /// Rational number (numerator, denominator)
    Rational(i64, i64),
}

/// A mathematical expression in tree form
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// A literal value
    Value(Value),
    /// Binary operation
    BinOp {
        op: MathOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Unary operation
    UnaryOp { op: MathOp, operand: Box<Expr> },
    /// Function application
    Function {
        func: MathFn,
        args: Vec<Expr>,
    },
}

/// The math engine for symbolic and numeric computation
#[derive(Debug, Default, Clone)]
pub struct MathEngine {
    /// Symbol table for variable bindings
    symbols: std::collections::HashMap<String, Value>,
}

impl MathEngine {
    /// Create a new math engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a symbol to a value
    pub fn bind(&mut self, name: impl Into<String>, value: Value) {
        self.symbols.insert(name.into(), value);
    }

    /// Evaluate an expression to a numeric value
    pub fn evaluate(&self, expr: &Expr) -> EngineResult<f64> {
        match expr {
            Expr::Value(v) => self.value_to_f64(v),
            Expr::BinOp { op, left, right } => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                self.apply_binop(*op, l, r)
            }
            Expr::UnaryOp { op, operand } => {
                let v = self.evaluate(operand)?;
                self.apply_unaryop(*op, v)
            }
            Expr::Function { func, args } => {
                let evaluated: Result<Vec<f64>, _> =
                    args.iter().map(|a| self.evaluate(a)).collect();
                self.apply_function(*func, &evaluated?)
            }
        }
    }

    fn value_to_f64(&self, value: &Value) -> EngineResult<f64> {
        match value {
            Value::Integer(i) => Ok(*i as f64),
            Value::Float(f) => Ok(*f),
            Value::Symbol(s) => {
                let bound = self
                    .symbols
                    .get(s)
                    .ok_or_else(|| EngineError::UndefinedSymbol(s.clone()))?;
                self.value_to_f64(bound)
            }
            Value::Rational(n, d) => {
                if *d == 0 {
                    Err(EngineError::DivisionByZero)
                } else {
                    Ok(*n as f64 / *d as f64)
                }
            }
        }
    }

    fn apply_binop(&self, op: MathOp, left: f64, right: f64) -> EngineResult<f64> {
        match op {
            MathOp::Add => Ok(left + right),
            MathOp::Sub => Ok(left - right),
            MathOp::Mul => Ok(left * right),
            MathOp::Div => {
                if right == 0.0 {
                    Err(EngineError::DivisionByZero)
                } else {
                    Ok(left / right)
                }
            }
            MathOp::Pow => Ok(left.powf(right)),
            MathOp::Mod => {
                if right == 0.0 {
                    Err(EngineError::DivisionByZero)
                } else {
                    Ok(left % right)
                }
            }
            MathOp::Neg => Err(EngineError::InvalidOperation(
                "Neg is a unary operator".into(),
            )),
        }
    }

    fn apply_unaryop(&self, op: MathOp, value: f64) -> EngineResult<f64> {
        match op {
            MathOp::Neg => Ok(-value),
            _ => Err(EngineError::InvalidOperation(format!(
                "{:?} is not a unary operator",
                op
            ))),
        }
    }

    fn apply_function(&self, func: MathFn, args: &[f64]) -> EngineResult<f64> {
        match func {
            MathFn::Sin => Ok(args[0].sin()),
            MathFn::Cos => Ok(args[0].cos()),
            MathFn::Tan => Ok(args[0].tan()),
            MathFn::Log => Ok(args[0].log10()),
            MathFn::Ln => Ok(args[0].ln()),
            MathFn::Exp => Ok(args[0].exp()),
            MathFn::Sqrt => Ok(args[0].sqrt()),
            MathFn::Abs => Ok(args[0].abs()),
            MathFn::Floor => Ok(args[0].floor()),
            MathFn::Ceil => Ok(args[0].ceil()),
            MathFn::Derive | MathFn::Integrate => Err(EngineError::EvaluationError(
                "Symbolic operations require symbolic evaluation".into(),
            )),
        }
    }

    /// Generate a training example: expression -> result
    pub fn generate_training_pair(&self, expr: &Expr) -> EngineResult<(Expr, Value)> {
        let result = self.evaluate(expr)?;
        Ok((expr.clone(), Value::Float(result)))
    }

    /// Unbind a symbol
    pub fn unbind(&mut self, name: &str) -> Option<Value> {
        self.symbols.remove(name)
    }

    /// Clear all symbol bindings
    pub fn clear_bindings(&mut self) {
        self.symbols.clear();
    }

    /// Get all bound symbols
    pub fn bound_symbols(&self) -> Vec<&String> {
        self.symbols.keys().collect()
    }

    /// Check if a symbol is bound
    pub fn is_bound(&self, name: &str) -> bool {
        self.symbols.contains_key(name)
    }
}

// ============================================================================
// Expression Utilities
// ============================================================================

impl Expr {
    /// Create an integer value expression
    pub fn int(n: i64) -> Self {
        Expr::Value(Value::Integer(n))
    }

    /// Create a float value expression
    pub fn float(n: f64) -> Self {
        Expr::Value(Value::Float(n))
    }

    /// Create a symbol expression
    pub fn symbol(name: impl Into<String>) -> Self {
        Expr::Value(Value::Symbol(name.into()))
    }

    /// Create an addition expression
    pub fn add(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a subtraction expression
    pub fn sub(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: MathOp::Sub,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a multiplication expression
    pub fn mul(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a division expression
    pub fn div(left: Expr, right: Expr) -> Self {
        Expr::BinOp {
            op: MathOp::Div,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a power expression
    pub fn pow(base: Expr, exp: Expr) -> Self {
        Expr::BinOp {
            op: MathOp::Pow,
            left: Box::new(base),
            right: Box::new(exp),
        }
    }

    /// Create a negation expression
    pub fn neg(operand: Expr) -> Self {
        Expr::UnaryOp {
            op: MathOp::Neg,
            operand: Box::new(operand),
        }
    }

    /// Create a function application
    pub fn func(f: MathFn, args: Vec<Expr>) -> Self {
        Expr::Function { func: f, args }
    }

    /// Collect all symbols in the expression
    pub fn collect_symbols(&self) -> HashSet<String> {
        let mut symbols = HashSet::new();
        self.collect_symbols_recursive(&mut symbols);
        symbols
    }

    fn collect_symbols_recursive(&self, symbols: &mut HashSet<String>) {
        match self {
            Expr::Value(Value::Symbol(s)) => {
                symbols.insert(s.clone());
            }
            Expr::Value(_) => {}
            Expr::BinOp { left, right, .. } => {
                left.collect_symbols_recursive(symbols);
                right.collect_symbols_recursive(symbols);
            }
            Expr::UnaryOp { operand, .. } => {
                operand.collect_symbols_recursive(symbols);
            }
            Expr::Function { args, .. } => {
                for arg in args {
                    arg.collect_symbols_recursive(symbols);
                }
            }
        }
    }

    /// Check if expression contains any symbols
    pub fn is_symbolic(&self) -> bool {
        !self.collect_symbols().is_empty()
    }

    /// Check if expression is purely numeric (no symbols)
    pub fn is_constant(&self) -> bool {
        self.collect_symbols().is_empty()
    }

    /// Get the depth of the expression tree
    pub fn depth(&self) -> usize {
        match self {
            Expr::Value(_) => 1,
            Expr::BinOp { left, right, .. } => 1 + left.depth().max(right.depth()),
            Expr::UnaryOp { operand, .. } => 1 + operand.depth(),
            Expr::Function { args, .. } => {
                1 + args.iter().map(|a| a.depth()).max().unwrap_or(0)
            }
        }
    }

    /// Count the total number of nodes in the expression tree
    pub fn node_count(&self) -> usize {
        match self {
            Expr::Value(_) => 1,
            Expr::BinOp { left, right, .. } => 1 + left.node_count() + right.node_count(),
            Expr::UnaryOp { operand, .. } => 1 + operand.node_count(),
            Expr::Function { args, .. } => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
        }
    }
}

// ============================================================================
// Symbolic Engine (Symbolic Manipulation)
// ============================================================================

/// Engine for symbolic manipulation of expressions
#[derive(Debug, Default)]
pub struct SymbolicEngine;

impl SymbolicEngine {
    /// Create a new symbolic engine
    pub fn new() -> Self {
        Self
    }

    /// Substitute a symbol with an expression
    pub fn substitute(&self, expr: &Expr, var: &str, replacement: &Expr) -> Expr {
        match expr {
            Expr::Value(Value::Symbol(s)) if s == var => replacement.clone(),
            Expr::Value(_) => expr.clone(),
            Expr::BinOp { op, left, right } => Expr::BinOp {
                op: *op,
                left: Box::new(self.substitute(left, var, replacement)),
                right: Box::new(self.substitute(right, var, replacement)),
            },
            Expr::UnaryOp { op, operand } => Expr::UnaryOp {
                op: *op,
                operand: Box::new(self.substitute(operand, var, replacement)),
            },
            Expr::Function { func, args } => Expr::Function {
                func: *func,
                args: args
                    .iter()
                    .map(|a| self.substitute(a, var, replacement))
                    .collect(),
            },
        }
    }

    /// Symbolic differentiation with respect to a variable
    /// Implements basic differentiation rules:
    /// - d/dx(c) = 0 (constant)
    /// - d/dx(x) = 1
    /// - d/dx(u + v) = du/dx + dv/dx
    /// - d/dx(u - v) = du/dx - dv/dx
    /// - d/dx(u * v) = u * dv/dx + v * du/dx (product rule)
    /// - d/dx(u / v) = (v * du/dx - u * dv/dx) / v² (quotient rule)
    /// - d/dx(u^n) = n * u^(n-1) * du/dx (power rule for constant n)
    /// - d/dx(sin(u)) = cos(u) * du/dx
    /// - d/dx(cos(u)) = -sin(u) * du/dx
    /// - d/dx(exp(u)) = exp(u) * du/dx
    /// - d/dx(ln(u)) = (1/u) * du/dx
    pub fn differentiate(&self, expr: &Expr, var: &str) -> Expr {
        match expr {
            // Constant: d/dx(c) = 0
            Expr::Value(Value::Integer(_))
            | Expr::Value(Value::Float(_))
            | Expr::Value(Value::Rational(_, _)) => Expr::int(0),

            // Variable: d/dx(x) = 1, d/dx(y) = 0
            Expr::Value(Value::Symbol(s)) => {
                if s == var {
                    Expr::int(1)
                } else {
                    Expr::int(0)
                }
            }

            // Binary operations
            Expr::BinOp { op, left, right } => {
                let dl = self.differentiate(left, var);
                let dr = self.differentiate(right, var);

                match op {
                    // d/dx(u + v) = du/dx + dv/dx
                    MathOp::Add => Expr::add(dl, dr),

                    // d/dx(u - v) = du/dx - dv/dx
                    MathOp::Sub => Expr::sub(dl, dr),

                    // d/dx(u * v) = u * dv/dx + v * du/dx (product rule)
                    MathOp::Mul => Expr::add(
                        Expr::mul((**left).clone(), dr),
                        Expr::mul((**right).clone(), dl),
                    ),

                    // d/dx(u / v) = (v * du/dx - u * dv/dx) / v² (quotient rule)
                    MathOp::Div => {
                        let numerator = Expr::sub(
                            Expr::mul((**right).clone(), dl),
                            Expr::mul((**left).clone(), dr),
                        );
                        let denominator = Expr::pow((**right).clone(), Expr::int(2));
                        Expr::div(numerator, denominator)
                    }

                    // d/dx(u^n) = n * u^(n-1) * du/dx (power rule)
                    MathOp::Pow => {
                        // For simplicity, handle constant exponent case
                        if !right.is_symbolic() {
                            // n * u^(n-1) * du/dx
                            Expr::mul(
                                Expr::mul(
                                    (**right).clone(),
                                    Expr::pow((**left).clone(), Expr::sub((**right).clone(), Expr::int(1))),
                                ),
                                dl,
                            )
                        } else {
                            // General case: d/dx(u^v) = u^v * (v' * ln(u) + v * u'/u)
                            // Simplified: assume constant exponent for now
                            Expr::mul(
                                Expr::mul(
                                    (**right).clone(),
                                    Expr::pow((**left).clone(), Expr::sub((**right).clone(), Expr::int(1))),
                                ),
                                dl,
                            )
                        }
                    }

                    // Modulo - not differentiable in general
                    MathOp::Mod => Expr::int(0),

                    // Negation shouldn't appear as binary
                    MathOp::Neg => Expr::int(0),
                }
            }

            // Unary operations
            Expr::UnaryOp { op, operand } => match op {
                MathOp::Neg => Expr::neg(self.differentiate(operand, var)),
                _ => Expr::int(0),
            },

            // Functions
            Expr::Function { func, args } => {
                if args.is_empty() {
                    return Expr::int(0);
                }

                let u = &args[0];
                let du = self.differentiate(u, var);

                match func {
                    // d/dx(sin(u)) = cos(u) * du/dx
                    MathFn::Sin => Expr::mul(Expr::func(MathFn::Cos, vec![u.clone()]), du),

                    // d/dx(cos(u)) = -sin(u) * du/dx
                    MathFn::Cos => Expr::neg(Expr::mul(Expr::func(MathFn::Sin, vec![u.clone()]), du)),

                    // d/dx(tan(u)) = sec²(u) * du/dx = du/dx / cos²(u)
                    MathFn::Tan => Expr::div(
                        du,
                        Expr::pow(Expr::func(MathFn::Cos, vec![u.clone()]), Expr::int(2)),
                    ),

                    // d/dx(exp(u)) = exp(u) * du/dx
                    MathFn::Exp => Expr::mul(Expr::func(MathFn::Exp, vec![u.clone()]), du),

                    // d/dx(ln(u)) = du/dx / u
                    MathFn::Ln => Expr::div(du, u.clone()),

                    // d/dx(log(u)) = du/dx / (u * ln(10))
                    MathFn::Log => Expr::div(
                        du,
                        Expr::mul(u.clone(), Expr::float(10.0_f64.ln())),
                    ),

                    // d/dx(sqrt(u)) = du/dx / (2 * sqrt(u))
                    MathFn::Sqrt => Expr::div(
                        du,
                        Expr::mul(Expr::int(2), Expr::func(MathFn::Sqrt, vec![u.clone()])),
                    ),

                    // d/dx(abs(u)) = sign(u) * du/dx (not properly supported)
                    MathFn::Abs => du,

                    // Floor and ceil are not differentiable
                    MathFn::Floor | MathFn::Ceil => Expr::int(0),

                    // Derive and Integrate are meta-operations
                    MathFn::Derive | MathFn::Integrate => Expr::int(0),
                }
            }
        }
    }

    /// Evaluate a symbolic expression at a specific value
    pub fn evaluate_at(&self, expr: &Expr, var: &str, value: f64) -> Expr {
        self.substitute(expr, var, &Expr::float(value))
    }

    /// Symbolic integration with respect to a variable
    /// Implements basic integration rules:
    /// - ∫k dx = kx (constant)
    /// - ∫x dx = x²/2
    /// - ∫x^n dx = x^(n+1)/(n+1) for n ≠ -1 (power rule)
    /// - ∫x^(-1) dx = ln|x| (special case)
    /// - ∫(f + g) dx = ∫f dx + ∫g dx (sum rule)
    /// - ∫(f - g) dx = ∫f dx - ∫g dx (difference rule)
    /// - ∫kf dx = k∫f dx (constant multiple)
    /// - ∫sin(x) dx = -cos(x)
    /// - ∫cos(x) dx = sin(x)
    /// - ∫exp(x) dx = exp(x)
    /// - ∫1/x dx = ln|x|
    pub fn integrate(&self, expr: &Expr, var: &str) -> IntegrationResult<Expr> {
        match expr {
            // Constant: ∫k dx = kx
            Expr::Value(Value::Integer(n)) => {
                Ok(Expr::mul(Expr::int(*n), Expr::symbol(var)))
            }
            Expr::Value(Value::Float(f)) => {
                Ok(Expr::mul(Expr::float(*f), Expr::symbol(var)))
            }
            Expr::Value(Value::Rational(n, d)) => {
                Ok(Expr::mul(
                    Expr::Value(Value::Rational(*n, *d)),
                    Expr::symbol(var),
                ))
            }

            // Variable: ∫x dx = x²/2, ∫y dx = yx (y is constant wrt x)
            Expr::Value(Value::Symbol(s)) => {
                if s == var {
                    // ∫x dx = x²/2
                    Ok(Expr::div(
                        Expr::pow(Expr::symbol(var), Expr::int(2)),
                        Expr::int(2),
                    ))
                } else {
                    // ∫y dx = yx (y is constant wrt x)
                    Ok(Expr::mul(Expr::symbol(s.clone()), Expr::symbol(var)))
                }
            }

            // Binary operations
            Expr::BinOp { op, left, right } => {
                match op {
                    // Sum rule: ∫(f + g) dx = ∫f dx + ∫g dx
                    MathOp::Add => Ok(Expr::add(
                        self.integrate(left, var)?,
                        self.integrate(right, var)?,
                    )),

                    // Difference rule: ∫(f - g) dx = ∫f dx - ∫g dx
                    MathOp::Sub => Ok(Expr::sub(
                        self.integrate(left, var)?,
                        self.integrate(right, var)?,
                    )),

                    // Constant multiple rule: ∫kf dx = k∫f dx
                    MathOp::Mul => {
                        let left_contains = self.contains_var(left, var);
                        let right_contains = self.contains_var(right, var);

                        match (left_contains, right_contains) {
                            (false, true) => {
                                // k * f where k is constant
                                Ok(Expr::mul(
                                    (**left).clone(),
                                    self.integrate(right, var)?,
                                ))
                            }
                            (true, false) => {
                                // f * k where k is constant
                                Ok(Expr::mul(
                                    self.integrate(left, var)?,
                                    (**right).clone(),
                                ))
                            }
                            (false, false) => {
                                // Both constants: ∫(k1 * k2) dx = k1 * k2 * x
                                Ok(Expr::mul(
                                    Expr::mul((**left).clone(), (**right).clone()),
                                    Expr::symbol(var),
                                ))
                            }
                            (true, true) => {
                                // Both contain variable - not generally integrable by simple rules
                                // Check if it's x * x^n (can be simplified to x^(n+1))
                                if self.is_var(left, var) {
                                    if let Expr::BinOp {
                                        op: MathOp::Pow,
                                        left: base,
                                        right: exp,
                                    } = right.as_ref()
                                    {
                                        if self.is_var(base, var) && !self.contains_var(exp, var) {
                                            // x * x^n = x^(n+1)
                                            let new_exp = Expr::add((**exp).clone(), Expr::int(1));
                                            return self.integrate(
                                                &Expr::pow(Expr::symbol(var), new_exp),
                                                var,
                                            );
                                        }
                                    }
                                }
                                Err(IntegrationError::CannotIntegrate(
                                    "Product of two expressions containing the variable".to_string(),
                                ))
                            }
                        }
                    }

                    // Division: ∫(f/g) - only handle constant divisor
                    MathOp::Div => {
                        let right_contains = self.contains_var(right, var);

                        if !right_contains {
                            // ∫(f/k) dx = (1/k) * ∫f dx
                            Ok(Expr::div(
                                self.integrate(left, var)?,
                                (**right).clone(),
                            ))
                        } else if !self.contains_var(left, var) && self.is_var(right, var) {
                            // ∫(k/x) dx = k * ln|x|
                            Ok(Expr::mul(
                                (**left).clone(),
                                Expr::func(MathFn::Ln, vec![Expr::func(MathFn::Abs, vec![Expr::symbol(var)])]),
                            ))
                        } else if self.is_int(left, 1) && self.is_var(right, var) {
                            // ∫(1/x) dx = ln|x|
                            Ok(Expr::func(MathFn::Ln, vec![Expr::func(MathFn::Abs, vec![Expr::symbol(var)])]))
                        } else {
                            Err(IntegrationError::CannotIntegrate(
                                "Division with variable in denominator".to_string(),
                            ))
                        }
                    }

                    // Power rule: ∫x^n dx = x^(n+1)/(n+1) for n ≠ -1
                    MathOp::Pow => {
                        let is_base_var = self.is_var(left, var);
                        let exp_contains_var = self.contains_var(right, var);

                        if is_base_var && !exp_contains_var {
                            // ∫x^n dx
                            if let Some(n) = self.get_int_value(right) {
                                if n == -1 {
                                    // Special case: ∫x^(-1) dx = ln|x|
                                    Ok(Expr::func(MathFn::Ln, vec![Expr::func(MathFn::Abs, vec![Expr::symbol(var)])]))
                                } else {
                                    // Power rule: ∫x^n dx = x^(n+1)/(n+1)
                                    let new_exp = n + 1;
                                    Ok(Expr::div(
                                        Expr::pow(Expr::symbol(var), Expr::int(new_exp)),
                                        Expr::int(new_exp),
                                    ))
                                }
                            } else {
                                // Non-integer exponent: still apply power rule
                                let new_exp = Expr::add((**right).clone(), Expr::int(1));
                                Ok(Expr::div(
                                    Expr::pow(Expr::symbol(var), new_exp.clone()),
                                    new_exp,
                                ))
                            }
                        } else if !is_base_var && !exp_contains_var && !self.contains_var(left, var) {
                            // Constant: ∫k dx = kx
                            Ok(Expr::mul(expr.clone(), Expr::symbol(var)))
                        } else {
                            Err(IntegrationError::CannotIntegrate(
                                "Power with variable in exponent".to_string(),
                            ))
                        }
                    }

                    MathOp::Mod => Err(IntegrationError::NotSupported),
                    MathOp::Neg => Err(IntegrationError::NotSupported),
                }
            }

            // Unary negation: ∫(-f) dx = -∫f dx
            Expr::UnaryOp { op: MathOp::Neg, operand } => {
                Ok(Expr::neg(self.integrate(operand, var)?))
            }
            Expr::UnaryOp { .. } => Err(IntegrationError::NotSupported),

            // Functions
            Expr::Function { func, args } => {
                if args.is_empty() {
                    return Err(IntegrationError::CannotIntegrate(
                        "Function with no arguments".to_string(),
                    ));
                }

                let arg = &args[0];

                // Check if argument is just the variable
                if self.is_var(arg, var) {
                    match func {
                        // ∫sin(x) dx = -cos(x)
                        MathFn::Sin => Ok(Expr::neg(Expr::func(MathFn::Cos, vec![Expr::symbol(var)]))),

                        // ∫cos(x) dx = sin(x)
                        MathFn::Cos => Ok(Expr::func(MathFn::Sin, vec![Expr::symbol(var)])),

                        // ∫exp(x) dx = exp(x)
                        MathFn::Exp => Ok(Expr::func(MathFn::Exp, vec![Expr::symbol(var)])),

                        // ∫tan(x) dx = -ln|cos(x)|
                        MathFn::Tan => Ok(Expr::neg(Expr::func(
                            MathFn::Ln,
                            vec![Expr::func(MathFn::Abs, vec![Expr::func(MathFn::Cos, vec![Expr::symbol(var)])])],
                        ))),

                        // ∫1/x dx = ln|x| (handled via Ln function of x)
                        MathFn::Ln | MathFn::Log | MathFn::Sqrt | MathFn::Abs | MathFn::Floor | MathFn::Ceil => {
                            Err(IntegrationError::CannotIntegrate(
                                format!("Integration of {:?} not supported", func),
                            ))
                        }

                        MathFn::Derive | MathFn::Integrate => {
                            Err(IntegrationError::CannotIntegrate(
                                "Meta-operations cannot be integrated directly".to_string(),
                            ))
                        }
                    }
                } else if !self.contains_var(arg, var) {
                    // Function of constant: ∫f(c) dx = f(c) * x
                    Ok(Expr::mul(expr.clone(), Expr::symbol(var)))
                } else {
                    Err(IntegrationError::CannotIntegrate(
                        "Chain rule integration not supported".to_string(),
                    ))
                }
            }
        }
    }

    /// Check if expression is the specified variable
    fn is_var(&self, expr: &Expr, var: &str) -> bool {
        matches!(expr, Expr::Value(Value::Symbol(s)) if s == var)
    }

    /// Check if expression contains the specified variable
    fn contains_var(&self, expr: &Expr, var: &str) -> bool {
        match expr {
            Expr::Value(Value::Symbol(s)) => s == var,
            Expr::Value(_) => false,
            Expr::BinOp { left, right, .. } => {
                self.contains_var(left, var) || self.contains_var(right, var)
            }
            Expr::UnaryOp { operand, .. } => self.contains_var(operand, var),
            Expr::Function { args, .. } => args.iter().any(|a| self.contains_var(a, var)),
        }
    }

    /// Try to get integer value from expression
    fn get_int_value(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Value(Value::Integer(n)) => Some(*n),
            _ => None,
        }
    }

    /// Check if expression is a specific integer
    fn is_int(&self, expr: &Expr, n: i64) -> bool {
        matches!(expr, Expr::Value(Value::Integer(i)) if *i == n)
    }

    /// Simplify an expression (basic algebraic simplification)
    ///
    /// Handles:
    /// - x + 0 = x, x * 1 = x, x * 0 = 0
    /// - x - x = 0, x / x = 1
    /// - Constant folding: 2 + 3 = 5
    fn simplify(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::BinOp { op, left, right } => {
                let left_s = self.simplify(left);
                let right_s = self.simplify(right);

                // Try constant folding
                if let (Some(l), Some(r)) = (self.evaluate_numeric(&left_s), self.evaluate_numeric(&right_s)) {
                    let result = match op {
                        MathOp::Add => Some(l + r),
                        MathOp::Sub => Some(l - r),
                        MathOp::Mul => Some(l * r),
                        MathOp::Div if r != 0.0 => Some(l / r),
                        MathOp::Pow => Some(l.powf(r)),
                        _ => None,
                    };
                    if let Some(v) = result {
                        if v == v.floor() && v.abs() < i64::MAX as f64 {
                            return Expr::int(v as i64);
                        } else {
                            return Expr::float(v);
                        }
                    }
                }

                // Algebraic simplifications
                match op {
                    MathOp::Add => {
                        // x + 0 = x
                        if self.is_zero(&right_s) {
                            return left_s;
                        }
                        if self.is_zero(&left_s) {
                            return right_s;
                        }
                    }
                    MathOp::Sub => {
                        // x - 0 = x
                        if self.is_zero(&right_s) {
                            return left_s;
                        }
                        // 0 - x = -x
                        if self.is_zero(&left_s) {
                            return Expr::neg(right_s);
                        }
                        // x - x = 0
                        if left_s == right_s {
                            return Expr::int(0);
                        }
                    }
                    MathOp::Mul => {
                        // x * 0 = 0
                        if self.is_zero(&left_s) || self.is_zero(&right_s) {
                            return Expr::int(0);
                        }
                        // x * 1 = x
                        if self.is_int(&right_s, 1) {
                            return left_s;
                        }
                        if self.is_int(&left_s, 1) {
                            return right_s;
                        }
                    }
                    MathOp::Div => {
                        // x / 1 = x
                        if self.is_int(&right_s, 1) {
                            return left_s;
                        }
                        // 0 / x = 0 (x != 0)
                        if self.is_zero(&left_s) && !self.is_zero(&right_s) {
                            return Expr::int(0);
                        }
                    }
                    MathOp::Pow => {
                        // x^0 = 1
                        if self.is_zero(&right_s) {
                            return Expr::int(1);
                        }
                        // x^1 = x
                        if self.is_int(&right_s, 1) {
                            return left_s;
                        }
                    }
                    _ => {}
                }

                Expr::BinOp {
                    op: *op,
                    left: Box::new(left_s),
                    right: Box::new(right_s),
                }
            }
            Expr::UnaryOp { op: MathOp::Neg, operand } => {
                let operand_s = self.simplify(operand);
                // -(-x) = x
                if let Expr::UnaryOp { op: MathOp::Neg, operand: inner } = operand_s {
                    return *inner;
                }
                // -0 = 0
                if self.is_zero(&operand_s) {
                    return Expr::int(0);
                }
                // -n for integer n
                if let Some(n) = self.get_int_value(&operand_s) {
                    return Expr::int(-n);
                }
                Expr::UnaryOp {
                    op: MathOp::Neg,
                    operand: Box::new(operand_s),
                }
            }
            Expr::Function { func, args } => {
                let args_s: Vec<Expr> = args.iter().map(|a| self.simplify(a)).collect();
                Expr::Function {
                    func: *func,
                    args: args_s,
                }
            }
            _ => expr.clone(),
        }
    }

    // ========================================================================
    // Equation Solving (backend-011)
    // ========================================================================

    /// Solve an equation for a variable
    ///
    /// Supports:
    /// - Linear equations: ax + b = c
    /// - Quadratic equations: ax² + bx + c = 0
    /// - Simple rational: a/x = b
    pub fn solve(&self, equation: &Equation, var: &str) -> SolveResult<Solution> {
        // Move everything to one side: lhs - rhs = 0
        let combined = Expr::sub(equation.lhs.clone(), equation.rhs.clone());
        let simplified = self.simplify(&combined);

        // Determine polynomial degree
        let degree = self.polynomial_degree(&simplified, var);

        match degree {
            0 => {
                // No variable: check if equation is identity or contradiction
                if self.is_zero(&simplified) {
                    Ok(Solution::Infinite)
                } else {
                    Ok(Solution::NoSolution)
                }
            }
            1 => self.solve_linear(&simplified, var),
            2 => self.solve_quadratic(&simplified, var),
            _ => Err(SolveError::DegreeTooHigh(degree)),
        }
    }

    /// Determine the polynomial degree with respect to a variable
    fn polynomial_degree(&self, expr: &Expr, var: &str) -> usize {
        match expr {
            Expr::Value(Value::Symbol(s)) if s == var => 1,
            Expr::Value(_) => 0,
            Expr::BinOp { op: MathOp::Add | MathOp::Sub, left, right } => {
                let left_deg = self.polynomial_degree(left, var);
                let right_deg = self.polynomial_degree(right, var);
                left_deg.max(right_deg)
            }
            Expr::BinOp { op: MathOp::Mul, left, right } => {
                let left_deg = self.polynomial_degree(left, var);
                let right_deg = self.polynomial_degree(right, var);
                left_deg + right_deg
            }
            Expr::BinOp { op: MathOp::Pow, left, right } => {
                if self.is_var(left, var) {
                    if let Some(n) = self.get_int_value(right) {
                        n as usize
                    } else {
                        usize::MAX // Non-integer exponent
                    }
                } else {
                    0
                }
            }
            Expr::BinOp { op: MathOp::Div, left, right } => {
                if self.contains_var(right, var) {
                    usize::MAX // Variable in denominator
                } else {
                    self.polynomial_degree(left, var)
                }
            }
            Expr::UnaryOp { op: MathOp::Neg, operand } => self.polynomial_degree(operand, var),
            _ => 0,
        }
    }

    /// Check if expression evaluates to zero
    fn is_zero(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Value(Value::Integer(0)))
            || matches!(expr, Expr::Value(Value::Float(f)) if *f == 0.0)
    }

    /// Solve linear equation: ax + b = 0 => x = -b/a
    fn solve_linear(&self, expr: &Expr, var: &str) -> SolveResult<Solution> {
        // Extract coefficient of x (a) and constant term (b)
        let (coeff, constant) = self.extract_linear_coefficients(expr, var);

        // Check if coefficient is zero
        if self.is_zero(&coeff) {
            return if self.is_zero(&constant) {
                Ok(Solution::Infinite)
            } else {
                Ok(Solution::NoSolution)
            };
        }

        // x = -b/a
        let neg_constant = Expr::neg(constant);
        let solution = self.simplify(&Expr::div(neg_constant, coeff));

        Ok(Solution::Single(solution))
    }

    /// Extract coefficients from linear expression: ax + b
    fn extract_linear_coefficients(&self, expr: &Expr, var: &str) -> (Expr, Expr) {
        match expr {
            Expr::Value(Value::Symbol(s)) if s == var => (Expr::int(1), Expr::int(0)),
            Expr::Value(_) => (Expr::int(0), expr.clone()),
            Expr::BinOp { op: MathOp::Add, left, right } => {
                let (lc, lt) = self.extract_linear_coefficients(left, var);
                let (rc, rt) = self.extract_linear_coefficients(right, var);
                (self.simplify(&Expr::add(lc, rc)), self.simplify(&Expr::add(lt, rt)))
            }
            Expr::BinOp { op: MathOp::Sub, left, right } => {
                let (lc, lt) = self.extract_linear_coefficients(left, var);
                let (rc, rt) = self.extract_linear_coefficients(right, var);
                (self.simplify(&Expr::sub(lc, rc)), self.simplify(&Expr::sub(lt, rt)))
            }
            Expr::BinOp { op: MathOp::Mul, left, right } => {
                if self.is_var(left, var) && !self.contains_var(right, var) {
                    (right.as_ref().clone(), Expr::int(0))
                } else if self.is_var(right, var) && !self.contains_var(left, var) {
                    (left.as_ref().clone(), Expr::int(0))
                } else if !self.contains_var(left, var) && !self.contains_var(right, var) {
                    (Expr::int(0), expr.clone())
                } else {
                    // Complex case - treat as coefficient 0
                    (Expr::int(0), expr.clone())
                }
            }
            Expr::UnaryOp { op: MathOp::Neg, operand } => {
                let (c, t) = self.extract_linear_coefficients(operand, var);
                (self.simplify(&Expr::neg(c)), self.simplify(&Expr::neg(t)))
            }
            _ => (Expr::int(0), expr.clone()),
        }
    }

    /// Solve quadratic equation: ax² + bx + c = 0
    fn solve_quadratic(&self, expr: &Expr, var: &str) -> SolveResult<Solution> {
        // Extract coefficients a, b, c
        let (a, b, c) = self.extract_quadratic_coefficients(expr, var);

        // Check if a is zero (not really quadratic)
        if self.is_zero(&a) {
            // Reduce to linear: bx + c = 0
            let linear_expr = Expr::add(Expr::mul(b, Expr::symbol(var)), c);
            return self.solve_linear(&linear_expr, var);
        }

        // Compute discriminant: b² - 4ac
        let b_squared = Expr::pow(b.clone(), Expr::int(2));
        let four_ac = Expr::mul(Expr::int(4), Expr::mul(a.clone(), c));
        let discriminant = self.simplify(&Expr::sub(b_squared, four_ac));

        // Try to evaluate discriminant numerically
        if let Some(d) = self.evaluate_numeric(&discriminant) {
            if d < 0.0 {
                return Ok(Solution::NoSolution);
            }

            // sqrt(discriminant)
            let sqrt_d = Expr::Function {
                func: MathFn::Sqrt,
                args: vec![discriminant],
            };

            // 2a
            let two_a = Expr::mul(Expr::int(2), a);

            // x1 = (-b + sqrt(d)) / 2a
            let neg_b = Expr::neg(b.clone());
            let x1 = self.simplify(&Expr::div(Expr::add(neg_b.clone(), sqrt_d.clone()), two_a.clone()));

            // x2 = (-b - sqrt(d)) / 2a
            let x2 = self.simplify(&Expr::div(Expr::sub(neg_b, sqrt_d), two_a));

            if d == 0.0 {
                Ok(Solution::Single(x1))
            } else {
                Ok(Solution::Multiple(vec![x1, x2]))
            }
        } else {
            Err(SolveError::SymbolicDiscriminant)
        }
    }

    /// Extract coefficients from quadratic expression: ax² + bx + c
    fn extract_quadratic_coefficients(&self, expr: &Expr, var: &str) -> (Expr, Expr, Expr) {
        // Simplified extraction - handles basic cases
        match expr {
            Expr::BinOp { op: MathOp::Add | MathOp::Sub, left, right } => {
                let (la, lb, lc) = self.extract_quadratic_coefficients(left, var);
                let (ra, rb, rc) = self.extract_quadratic_coefficients(right, var);
                if matches!(expr, Expr::BinOp { op: MathOp::Add, .. }) {
                    (
                        self.simplify(&Expr::add(la, ra)),
                        self.simplify(&Expr::add(lb, rb)),
                        self.simplify(&Expr::add(lc, rc)),
                    )
                } else {
                    (
                        self.simplify(&Expr::sub(la, ra)),
                        self.simplify(&Expr::sub(lb, rb)),
                        self.simplify(&Expr::sub(lc, rc)),
                    )
                }
            }
            Expr::BinOp { op: MathOp::Mul, left, right } => {
                // Check for x² or coefficient * x² or coefficient * x
                if self.is_var(left, var) && self.is_var(right, var) {
                    // x * x = x²
                    (Expr::int(1), Expr::int(0), Expr::int(0))
                } else if self.is_var(left, var) && !self.contains_var(right, var) {
                    // c * x
                    (Expr::int(0), right.as_ref().clone(), Expr::int(0))
                } else if !self.contains_var(left, var) && self.is_var(right, var) {
                    // c * x
                    (Expr::int(0), left.as_ref().clone(), Expr::int(0))
                } else if !self.contains_var(left, var) && !self.contains_var(right, var) {
                    // constant
                    (Expr::int(0), Expr::int(0), expr.clone())
                } else {
                    // Check if one side is x² pattern
                    let (la, lb, lc) = self.extract_quadratic_coefficients(left, var);
                    if !self.is_zero(&la) && !self.contains_var(right, var) {
                        // coeff * x²
                        (Expr::mul(right.as_ref().clone(), la), Expr::mul(right.as_ref().clone(), lb), Expr::mul(right.as_ref().clone(), lc))
                    } else {
                        (Expr::int(0), Expr::int(0), expr.clone())
                    }
                }
            }
            Expr::BinOp { op: MathOp::Pow, left, right } => {
                if self.is_var(left, var) && self.is_int(right, 2) {
                    // x²
                    (Expr::int(1), Expr::int(0), Expr::int(0))
                } else if self.is_var(left, var) && self.is_int(right, 1) {
                    // x
                    (Expr::int(0), Expr::int(1), Expr::int(0))
                } else {
                    (Expr::int(0), Expr::int(0), expr.clone())
                }
            }
            Expr::Value(Value::Symbol(s)) if s == var => {
                // x
                (Expr::int(0), Expr::int(1), Expr::int(0))
            }
            Expr::Value(_) => {
                // constant
                (Expr::int(0), Expr::int(0), expr.clone())
            }
            Expr::UnaryOp { op: MathOp::Neg, operand } => {
                let (a, b, c) = self.extract_quadratic_coefficients(operand, var);
                (
                    self.simplify(&Expr::neg(a)),
                    self.simplify(&Expr::neg(b)),
                    self.simplify(&Expr::neg(c)),
                )
            }
            _ => (Expr::int(0), Expr::int(0), expr.clone()),
        }
    }

    /// Try to evaluate expression numerically
    fn evaluate_numeric(&self, expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Value(Value::Integer(n)) => Some(*n as f64),
            Expr::Value(Value::Float(f)) => Some(*f),
            Expr::Value(Value::Rational(n, d)) if *d != 0 => Some(*n as f64 / *d as f64),
            Expr::Value(Value::Symbol(_)) => None,
            Expr::BinOp { op, left, right } => {
                let l = self.evaluate_numeric(left)?;
                let r = self.evaluate_numeric(right)?;
                match op {
                    MathOp::Add => Some(l + r),
                    MathOp::Sub => Some(l - r),
                    MathOp::Mul => Some(l * r),
                    MathOp::Div if r != 0.0 => Some(l / r),
                    MathOp::Pow => Some(l.powf(r)),
                    _ => None,
                }
            }
            Expr::UnaryOp { op: MathOp::Neg, operand } => {
                self.evaluate_numeric(operand).map(|v| -v)
            }
            Expr::Function { func, args } => {
                if args.len() != 1 {
                    return None;
                }
                let arg = self.evaluate_numeric(&args[0])?;
                match func {
                    MathFn::Sqrt if arg >= 0.0 => Some(arg.sqrt()),
                    MathFn::Sin => Some(arg.sin()),
                    MathFn::Cos => Some(arg.cos()),
                    MathFn::Exp => Some(arg.exp()),
                    MathFn::Ln if arg > 0.0 => Some(arg.ln()),
                    MathFn::Abs => Some(arg.abs()),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

// ============================================================================
// Equation Types (backend-011)
// ============================================================================

/// Represents an equation: lhs = rhs
#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    /// Left-hand side of the equation
    pub lhs: Expr,
    /// Right-hand side of the equation
    pub rhs: Expr,
}

impl Equation {
    /// Create a new equation
    pub fn new(lhs: Expr, rhs: Expr) -> Self {
        Self { lhs, rhs }
    }

    /// Create equation from expression equal to zero
    pub fn equals_zero(expr: Expr) -> Self {
        Self {
            lhs: expr,
            rhs: Expr::int(0),
        }
    }
}

/// Solution to an equation
#[derive(Debug, Clone, PartialEq)]
pub enum Solution {
    /// Single solution: x = value
    Single(Expr),
    /// Multiple solutions (e.g., quadratic with two roots)
    Multiple(Vec<Expr>),
    /// No real solution
    NoSolution,
    /// Infinitely many solutions (identity)
    Infinite,
}

impl Solution {
    /// Check if there is at least one solution
    pub fn has_solution(&self) -> bool {
        !matches!(self, Solution::NoSolution)
    }

    /// Get solutions as a vector (empty if no solution)
    pub fn solutions(&self) -> Vec<&Expr> {
        match self {
            Solution::Single(e) => vec![e],
            Solution::Multiple(es) => es.iter().collect(),
            Solution::NoSolution | Solution::Infinite => vec![],
        }
    }
}

/// Errors that can occur during equation solving
#[derive(Error, Debug)]
pub enum SolveError {
    #[error("Polynomial degree {0} is too high (max: 2)")]
    DegreeTooHigh(usize),
    #[error("Cannot solve: symbolic discriminant")]
    SymbolicDiscriminant,
    #[error("Cannot solve: expression too complex")]
    TooComplex,
    #[error("Variable not found in equation")]
    VariableNotFound,
}

/// Result type for solve operations
pub type SolveResult<T> = Result<T, SolveError>;

// ============================================================================
// Formal Algebraic Rules
// ============================================================================

/// A formal algebraic rule
#[derive(Debug, Clone)]
pub struct FormalRule {
    /// Rule name
    pub name: &'static str,
    /// Rule description
    pub description: &'static str,
    /// Rule category
    pub category: RuleCategory,
}

/// Categories of algebraic rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleCategory {
    /// Identity rules (x + 0 = x)
    Identity,
    /// Inverse rules (x - x = 0)
    Inverse,
    /// Commutative rules (a + b = b + a)
    Commutative,
    /// Associative rules ((a + b) + c = a + (b + c))
    Associative,
    /// Distributive rules (a * (b + c) = a*b + a*c)
    Distributive,
    /// Power rules (x^a * x^b = x^(a+b))
    Power,
    /// Logarithm rules (log(a*b) = log(a) + log(b))
    Logarithm,
    /// Trigonometric rules (sin²(x) + cos²(x) = 1)
    Trigonometric,
}

impl FormalRule {
    // Identity rules
    pub const ADDITIVE_IDENTITY: Self = FormalRule {
        name: "additive_identity",
        description: "x + 0 = x",
        category: RuleCategory::Identity,
    };

    pub const MULTIPLICATIVE_IDENTITY: Self = FormalRule {
        name: "multiplicative_identity",
        description: "x * 1 = x",
        category: RuleCategory::Identity,
    };

    pub const ZERO_PRODUCT: Self = FormalRule {
        name: "zero_product",
        description: "x * 0 = 0",
        category: RuleCategory::Identity,
    };

    // Inverse rules
    pub const ADDITIVE_INVERSE: Self = FormalRule {
        name: "additive_inverse",
        description: "x - x = 0",
        category: RuleCategory::Inverse,
    };

    pub const MULTIPLICATIVE_INVERSE: Self = FormalRule {
        name: "multiplicative_inverse",
        description: "x / x = 1 (x ≠ 0)",
        category: RuleCategory::Inverse,
    };

    // Power rules
    pub const POWER_ZERO: Self = FormalRule {
        name: "power_zero",
        description: "x^0 = 1",
        category: RuleCategory::Power,
    };

    pub const POWER_ONE: Self = FormalRule {
        name: "power_one",
        description: "x^1 = x",
        category: RuleCategory::Power,
    };

    pub const POWER_PRODUCT: Self = FormalRule {
        name: "power_product",
        description: "x^a * x^b = x^(a+b)",
        category: RuleCategory::Power,
    };

    pub const POWER_QUOTIENT: Self = FormalRule {
        name: "power_quotient",
        description: "x^a / x^b = x^(a-b)",
        category: RuleCategory::Power,
    };

    pub const POWER_POWER: Self = FormalRule {
        name: "power_power",
        description: "(x^a)^b = x^(a*b)",
        category: RuleCategory::Power,
    };

    // Logarithm rules
    pub const LOG_PRODUCT: Self = FormalRule {
        name: "log_product",
        description: "log(a*b) = log(a) + log(b)",
        category: RuleCategory::Logarithm,
    };

    pub const LOG_QUOTIENT: Self = FormalRule {
        name: "log_quotient",
        description: "log(a/b) = log(a) - log(b)",
        category: RuleCategory::Logarithm,
    };

    pub const LOG_POWER: Self = FormalRule {
        name: "log_power",
        description: "log(a^b) = b * log(a)",
        category: RuleCategory::Logarithm,
    };

    // Trigonometric rules
    pub const PYTHAGOREAN_IDENTITY: Self = FormalRule {
        name: "pythagorean_identity",
        description: "sin²(x) + cos²(x) = 1",
        category: RuleCategory::Trigonometric,
    };

    /// Get all defined rules
    pub fn all_rules() -> Vec<Self> {
        vec![
            Self::ADDITIVE_IDENTITY,
            Self::MULTIPLICATIVE_IDENTITY,
            Self::ZERO_PRODUCT,
            Self::ADDITIVE_INVERSE,
            Self::MULTIPLICATIVE_INVERSE,
            Self::POWER_ZERO,
            Self::POWER_ONE,
            Self::POWER_PRODUCT,
            Self::POWER_QUOTIENT,
            Self::POWER_POWER,
            Self::LOG_PRODUCT,
            Self::LOG_QUOTIENT,
            Self::LOG_POWER,
            Self::PYTHAGOREAN_IDENTITY,
        ]
    }

    /// Get rules by category
    pub fn rules_by_category(category: RuleCategory) -> Vec<Self> {
        Self::all_rules()
            .into_iter()
            .filter(|r| r.category == category)
            .collect()
    }
}

// ============================================================================
// Expression Validator
// ============================================================================

/// Result of expression validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the expression is valid
    pub is_valid: bool,
    /// Validation errors (if any)
    pub errors: Vec<String>,
    /// Validation warnings (if any)
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create an invalid result with errors
    pub fn invalid(errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Add a warning
    pub fn with_warning(mut self, warning: String) -> Self {
        self.warnings.push(warning);
        self
    }
}

/// Validator for mathematical expressions
#[derive(Debug, Default)]
pub struct ExprValidator;

impl ExprValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self
    }

    /// Validate an expression
    pub fn validate(&self, expr: &Expr) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        self.validate_recursive(expr, &mut errors, &mut warnings);

        if errors.is_empty() {
            ValidationResult {
                is_valid: true,
                errors,
                warnings,
            }
        } else {
            ValidationResult {
                is_valid: false,
                errors,
                warnings,
            }
        }
    }

    fn validate_recursive(&self, expr: &Expr, errors: &mut Vec<String>, warnings: &mut Vec<String>) {
        match expr {
            Expr::Value(Value::Rational(_, d)) if *d == 0 => {
                errors.push("Division by zero in rational number".to_string());
            }
            Expr::Value(Value::Float(f)) if f.is_nan() => {
                errors.push("NaN value in expression".to_string());
            }
            Expr::Value(Value::Float(f)) if f.is_infinite() => {
                warnings.push("Infinite value in expression".to_string());
            }
            Expr::Value(_) => {}

            Expr::BinOp { op, left, right } => {
                // Check for division by zero
                if *op == MathOp::Div || *op == MathOp::Mod {
                    if let Expr::Value(Value::Integer(0)) = **right {
                        errors.push("Division by zero".to_string());
                    }
                    if let Expr::Value(Value::Float(f)) = **right {
                        if f == 0.0 {
                            errors.push("Division by zero".to_string());
                        }
                    }
                }

                self.validate_recursive(left, errors, warnings);
                self.validate_recursive(right, errors, warnings);
            }

            Expr::UnaryOp { operand, .. } => {
                self.validate_recursive(operand, errors, warnings);
            }

            Expr::Function { func, args } => {
                // Check argument counts
                match func {
                    MathFn::Sin | MathFn::Cos | MathFn::Tan | MathFn::Log | MathFn::Ln
                    | MathFn::Exp | MathFn::Sqrt | MathFn::Abs | MathFn::Floor | MathFn::Ceil => {
                        if args.len() != 1 {
                            errors.push(format!(
                                "Function {:?} expects 1 argument, got {}",
                                func,
                                args.len()
                            ));
                        }
                    }
                    MathFn::Derive | MathFn::Integrate => {
                        if args.len() < 2 {
                            warnings.push(format!(
                                "Function {:?} typically needs at least 2 arguments",
                                func
                            ));
                        }
                    }
                }

                // Check for negative sqrt argument (warning only, as it could be symbolic)
                if *func == MathFn::Sqrt {
                    if let Some(Expr::Value(Value::Integer(n))) = args.first() {
                        if *n < 0 {
                            warnings.push("Square root of negative number".to_string());
                        }
                    }
                    if let Some(Expr::Value(Value::Float(f))) = args.first() {
                        if *f < 0.0 {
                            warnings.push("Square root of negative number".to_string());
                        }
                    }
                }

                // Check for log of non-positive (warning only)
                if *func == MathFn::Log || *func == MathFn::Ln {
                    if let Some(Expr::Value(Value::Integer(n))) = args.first() {
                        if *n <= 0 {
                            warnings.push("Logarithm of non-positive number".to_string());
                        }
                    }
                    if let Some(Expr::Value(Value::Float(f))) = args.first() {
                        if *f <= 0.0 {
                            warnings.push("Logarithm of non-positive number".to_string());
                        }
                    }
                }

                for arg in args {
                    self.validate_recursive(arg, errors, warnings);
                }
            }
        }
    }

    /// Check if two expressions are structurally equal
    pub fn expr_equal(&self, a: &Expr, b: &Expr) -> bool {
        match (a, b) {
            (Expr::Value(va), Expr::Value(vb)) => self.value_equal(va, vb),
            (
                Expr::BinOp {
                    op: opa,
                    left: la,
                    right: ra,
                },
                Expr::BinOp {
                    op: opb,
                    left: lb,
                    right: rb,
                },
            ) => opa == opb && self.expr_equal(la, lb) && self.expr_equal(ra, rb),
            (
                Expr::UnaryOp {
                    op: opa,
                    operand: oa,
                },
                Expr::UnaryOp {
                    op: opb,
                    operand: ob,
                },
            ) => opa == opb && self.expr_equal(oa, ob),
            (
                Expr::Function {
                    func: fa,
                    args: argsa,
                },
                Expr::Function {
                    func: fb,
                    args: argsb,
                },
            ) => {
                fa == fb
                    && argsa.len() == argsb.len()
                    && argsa.iter().zip(argsb.iter()).all(|(a, b)| self.expr_equal(a, b))
            }
            _ => false,
        }
    }

    fn value_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Integer(ia), Value::Integer(ib)) => ia == ib,
            (Value::Float(fa), Value::Float(fb)) => (fa - fb).abs() < 1e-10,
            (Value::Symbol(sa), Value::Symbol(sb)) => sa == sb,
            (Value::Rational(na, da), Value::Rational(nb, db)) => na * db == nb * da,
            // Cross-type comparisons
            (Value::Integer(i), Value::Float(f)) | (Value::Float(f), Value::Integer(i)) => {
                (*i as f64 - f).abs() < 1e-10
            }
            _ => false,
        }
    }

    /// Verify a computed result against expected value
    pub fn verify_result(&self, computed: f64, expected: f64, tolerance: f64) -> bool {
        (computed - expected).abs() < tolerance
    }
}

// ============================================================================
// Training Data Generator
// ============================================================================

/// Generator for training data
#[derive(Debug)]
pub struct TrainingDataGenerator {
    engine: MathEngine,
    symbolic: SymbolicEngine,
}

impl Default for TrainingDataGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingDataGenerator {
    /// Create a new generator
    pub fn new() -> Self {
        Self {
            engine: MathEngine::new(),
            symbolic: SymbolicEngine::new(),
        }
    }

    /// Generate arithmetic training examples
    /// Returns pairs of (expression, result)
    pub fn generate_arithmetic(&self, count: usize, max_depth: usize) -> Vec<(Expr, f64)> {
        let mut examples = Vec::with_capacity(count);
        let ops = [MathOp::Add, MathOp::Sub, MathOp::Mul, MathOp::Div];

        for i in 0..count {
            // Generate random-ish integers based on index
            let a = ((i * 7 + 3) % 20) as i64 + 1;
            let b = ((i * 11 + 5) % 20) as i64 + 1;
            let c = ((i * 13 + 7) % 20) as i64 + 1;

            let op1 = ops[i % 4];
            let op2 = ops[(i + 1) % 4];

            let expr = if max_depth <= 1 {
                Expr::BinOp {
                    op: op1,
                    left: Box::new(Expr::int(a)),
                    right: Box::new(Expr::int(b)),
                }
            } else {
                Expr::BinOp {
                    op: op2,
                    left: Box::new(Expr::BinOp {
                        op: op1,
                        left: Box::new(Expr::int(a)),
                        right: Box::new(Expr::int(b)),
                    }),
                    right: Box::new(Expr::int(c)),
                }
            };

            if let Ok(result) = self.engine.evaluate(&expr) {
                if result.is_finite() {
                    examples.push((expr, result));
                }
            }
        }

        examples
    }

    /// Generate differentiation training examples
    /// Returns pairs of (expression, derivative)
    pub fn generate_derivatives(&self, count: usize) -> Vec<(Expr, Expr)> {
        let mut examples = Vec::with_capacity(count);
        let var = "x";

        // Polynomial terms: x^n
        for n in 1..=count.min(5) {
            let expr = Expr::pow(Expr::symbol(var), Expr::int(n as i64));
            let deriv = self.symbolic.differentiate(&expr, var);
            examples.push((expr, deriv));
        }

        // Trigonometric functions
        if examples.len() < count {
            let sin_x = Expr::func(MathFn::Sin, vec![Expr::symbol(var)]);
            let deriv = self.symbolic.differentiate(&sin_x, var);
            examples.push((sin_x, deriv));
        }

        if examples.len() < count {
            let cos_x = Expr::func(MathFn::Cos, vec![Expr::symbol(var)]);
            let deriv = self.symbolic.differentiate(&cos_x, var);
            examples.push((cos_x, deriv));
        }

        // Exponential
        if examples.len() < count {
            let exp_x = Expr::func(MathFn::Exp, vec![Expr::symbol(var)]);
            let deriv = self.symbolic.differentiate(&exp_x, var);
            examples.push((exp_x, deriv));
        }

        // Natural log
        if examples.len() < count {
            let ln_x = Expr::func(MathFn::Ln, vec![Expr::symbol(var)]);
            let deriv = self.symbolic.differentiate(&ln_x, var);
            examples.push((ln_x, deriv));
        }

        examples
    }

    /// Generate simplification training examples
    /// Returns pairs of (unsimplified, simplified)
    pub fn generate_simplifications(&self, count: usize) -> Vec<(Expr, Expr)> {
        let mut examples = Vec::new();
        let x = Expr::symbol("x");

        // x + 0 = x
        if examples.len() < count {
            examples.push((Expr::add(x.clone(), Expr::int(0)), x.clone()));
        }

        // 0 + x = x
        if examples.len() < count {
            examples.push((Expr::add(Expr::int(0), x.clone()), x.clone()));
        }

        // x * 1 = x
        if examples.len() < count {
            examples.push((Expr::mul(x.clone(), Expr::int(1)), x.clone()));
        }

        // 1 * x = x
        if examples.len() < count {
            examples.push((Expr::mul(Expr::int(1), x.clone()), x.clone()));
        }

        // x * 0 = 0
        if examples.len() < count {
            examples.push((Expr::mul(x.clone(), Expr::int(0)), Expr::int(0)));
        }

        // x - 0 = x
        if examples.len() < count {
            examples.push((Expr::sub(x.clone(), Expr::int(0)), x.clone()));
        }

        // x / 1 = x
        if examples.len() < count {
            examples.push((Expr::div(x.clone(), Expr::int(1)), x.clone()));
        }

        // x ^ 0 = 1
        if examples.len() < count {
            examples.push((Expr::pow(x.clone(), Expr::int(0)), Expr::int(1)));
        }

        // x ^ 1 = x
        if examples.len() < count {
            examples.push((Expr::pow(x.clone(), Expr::int(1)), x.clone()));
        }

        // (x + 0) * 1 = x (nested)
        if examples.len() < count {
            examples.push((
                Expr::mul(Expr::add(x.clone(), Expr::int(0)), Expr::int(1)),
                x.clone(),
            ));
        }

        examples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =====================================================
    // Original Tests
    // =====================================================

    #[test]
    fn test_basic_arithmetic() {
        let engine = MathEngine::new();

        // (2 + 3) * 4 = 20
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Integer(2))),
                right: Box::new(Expr::Value(Value::Integer(3))),
            }),
            right: Box::new(Expr::Value(Value::Integer(4))),
        };

        let result = engine.evaluate(&expr).unwrap();
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_symbol_binding() {
        let mut engine = MathEngine::new();
        engine.bind("x", Value::Integer(5));

        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::Value(Value::Symbol("x".into()))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };

        let result = engine.evaluate(&expr).unwrap();
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_functions() {
        let engine = MathEngine::new();

        let expr = Expr::Function {
            func: MathFn::Sqrt,
            args: vec![Expr::Value(Value::Integer(16))],
        };

        let result = engine.evaluate(&expr).unwrap();
        assert_eq!(result, 4.0);
    }

    // =====================================================
    // Expression Utility Tests
    // =====================================================

    #[test]
    fn test_expr_builders() {
        // Test int, float, symbol builders
        assert_eq!(Expr::int(42), Expr::Value(Value::Integer(42)));
        assert_eq!(Expr::float(2.5), Expr::Value(Value::Float(2.5)));
        assert_eq!(
            Expr::symbol("x"),
            Expr::Value(Value::Symbol("x".to_string()))
        );

        // Test operator builders
        let add = Expr::add(Expr::int(1), Expr::int(2));
        assert!(matches!(add, Expr::BinOp { op: MathOp::Add, .. }));

        let mul = Expr::mul(Expr::int(3), Expr::int(4));
        assert!(matches!(mul, Expr::BinOp { op: MathOp::Mul, .. }));
    }

    #[test]
    fn test_collect_symbols() {
        // Expression with no symbols
        let expr = Expr::add(Expr::int(1), Expr::int(2));
        assert!(expr.collect_symbols().is_empty());

        // Expression with one symbol
        let expr = Expr::add(Expr::symbol("x"), Expr::int(1));
        let symbols = expr.collect_symbols();
        assert_eq!(symbols.len(), 1);
        assert!(symbols.contains("x"));

        // Expression with multiple symbols
        let expr = Expr::mul(Expr::symbol("x"), Expr::symbol("y"));
        let symbols = expr.collect_symbols();
        assert_eq!(symbols.len(), 2);
        assert!(symbols.contains("x"));
        assert!(symbols.contains("y"));
    }

    #[test]
    fn test_is_symbolic_and_constant() {
        let constant = Expr::add(Expr::int(1), Expr::int(2));
        assert!(!constant.is_symbolic());
        assert!(constant.is_constant());

        let symbolic = Expr::add(Expr::symbol("x"), Expr::int(1));
        assert!(symbolic.is_symbolic());
        assert!(!symbolic.is_constant());
    }

    #[test]
    fn test_expr_depth() {
        // Depth 1: single value
        assert_eq!(Expr::int(1).depth(), 1);

        // Depth 2: simple binary op
        assert_eq!(Expr::add(Expr::int(1), Expr::int(2)).depth(), 2);

        // Depth 3: nested
        let nested = Expr::mul(Expr::add(Expr::int(1), Expr::int(2)), Expr::int(3));
        assert_eq!(nested.depth(), 3);
    }

    #[test]
    fn test_expr_node_count() {
        // 1 node
        assert_eq!(Expr::int(1).node_count(), 1);

        // 3 nodes: op + 2 values
        assert_eq!(Expr::add(Expr::int(1), Expr::int(2)).node_count(), 3);

        // 5 nodes: outer op + inner op + 3 values
        let nested = Expr::mul(Expr::add(Expr::int(1), Expr::int(2)), Expr::int(3));
        assert_eq!(nested.node_count(), 5);
    }

    // =====================================================
    // Symbolic Engine Tests
    // =====================================================

    #[test]
    fn test_substitute() {
        let symbolic = SymbolicEngine::new();

        // Substitute x with 5 in (x + 1)
        let expr = Expr::add(Expr::symbol("x"), Expr::int(1));
        let result = symbolic.substitute(&expr, "x", &Expr::int(5));

        // Should be (5 + 1)
        assert_eq!(result, Expr::add(Expr::int(5), Expr::int(1)));
    }

    #[test]
    fn test_differentiate_constant() {
        let symbolic = SymbolicEngine::new();

        // d/dx(5) = 0
        let deriv = symbolic.differentiate(&Expr::int(5), "x");
        assert_eq!(deriv, Expr::int(0));
    }

    #[test]
    fn test_differentiate_variable() {
        let symbolic = SymbolicEngine::new();

        // d/dx(x) = 1
        let deriv = symbolic.differentiate(&Expr::symbol("x"), "x");
        assert_eq!(deriv, Expr::int(1));

        // d/dx(y) = 0
        let deriv = symbolic.differentiate(&Expr::symbol("y"), "x");
        assert_eq!(deriv, Expr::int(0));
    }

    #[test]
    fn test_differentiate_sum() {
        let symbolic = SymbolicEngine::new();

        // d/dx(x + 1) = 1 + 0 = 1
        let expr = Expr::add(Expr::symbol("x"), Expr::int(1));
        let deriv = symbolic.differentiate(&expr, "x");

        // Result is (1 + 0)
        assert_eq!(deriv, Expr::add(Expr::int(1), Expr::int(0)));
    }

    #[test]
    fn test_differentiate_product() {
        let symbolic = SymbolicEngine::new();

        // d/dx(x * x) = x * 1 + x * 1 = 2x (using product rule)
        let expr = Expr::mul(Expr::symbol("x"), Expr::symbol("x"));
        let deriv = symbolic.differentiate(&expr, "x");

        // Should be (x * 1) + (x * 1)
        let expected = Expr::add(
            Expr::mul(Expr::symbol("x"), Expr::int(1)),
            Expr::mul(Expr::symbol("x"), Expr::int(1)),
        );
        assert_eq!(deriv, expected);
    }

    #[test]
    fn test_differentiate_power() {
        let symbolic = SymbolicEngine::new();

        // d/dx(x^2) = 2 * x^1 * 1 (power rule)
        let expr = Expr::pow(Expr::symbol("x"), Expr::int(2));
        let deriv = symbolic.differentiate(&expr, "x");

        // Result should involve 2 * x^1 * 1
        // The structure is (2 * (x ^ (2-1))) * 1
        assert!(matches!(deriv, Expr::BinOp { op: MathOp::Mul, .. }));
    }

    #[test]
    fn test_differentiate_sin() {
        let symbolic = SymbolicEngine::new();

        // d/dx(sin(x)) = cos(x) * 1
        let expr = Expr::func(MathFn::Sin, vec![Expr::symbol("x")]);
        let deriv = symbolic.differentiate(&expr, "x");

        // Result should be cos(x) * 1
        assert!(matches!(deriv, Expr::BinOp { op: MathOp::Mul, .. }));
    }

    #[test]
    fn test_differentiate_exp() {
        let symbolic = SymbolicEngine::new();

        // d/dx(exp(x)) = exp(x) * 1
        let expr = Expr::func(MathFn::Exp, vec![Expr::symbol("x")]);
        let deriv = symbolic.differentiate(&expr, "x");

        // Result should be exp(x) * 1
        assert!(matches!(deriv, Expr::BinOp { op: MathOp::Mul, .. }));
    }

    #[test]
    fn test_evaluate_at() {
        let symbolic = SymbolicEngine::new();

        // Evaluate x^2 at x=3 should give (3.0)^2
        let expr = Expr::pow(Expr::symbol("x"), Expr::int(2));
        let result = symbolic.evaluate_at(&expr, "x", 3.0);

        // Result should be (3.0)^2
        assert_eq!(result, Expr::pow(Expr::float(3.0), Expr::int(2)));
    }

    // =====================================================
    // Formal Rules Tests
    // =====================================================

    #[test]
    fn test_formal_rules_exist() {
        let rules = FormalRule::all_rules();
        assert!(!rules.is_empty());
        assert!(rules.len() >= 10);
    }

    #[test]
    fn test_rules_by_category() {
        let identity_rules = FormalRule::rules_by_category(RuleCategory::Identity);
        assert!(!identity_rules.is_empty());

        let power_rules = FormalRule::rules_by_category(RuleCategory::Power);
        assert!(!power_rules.is_empty());
    }

    // =====================================================
    // Validator Tests
    // =====================================================

    #[test]
    fn test_validate_valid_expression() {
        let validator = ExprValidator::new();

        let expr = Expr::add(Expr::int(1), Expr::int(2));
        let result = validator.validate(&expr);

        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_division_by_zero() {
        let validator = ExprValidator::new();

        let expr = Expr::div(Expr::int(1), Expr::int(0));
        let result = validator.validate(&expr);

        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_validate_negative_sqrt_warning() {
        let validator = ExprValidator::new();

        let expr = Expr::func(MathFn::Sqrt, vec![Expr::int(-1)]);
        let result = validator.validate(&expr);

        // Negative sqrt is a warning, not an error
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_expr_equal() {
        let validator = ExprValidator::new();

        let a = Expr::add(Expr::int(1), Expr::int(2));
        let b = Expr::add(Expr::int(1), Expr::int(2));
        let c = Expr::add(Expr::int(1), Expr::int(3));

        assert!(validator.expr_equal(&a, &b));
        assert!(!validator.expr_equal(&a, &c));
    }

    #[test]
    fn test_verify_result() {
        let validator = ExprValidator::new();

        assert!(validator.verify_result(3.0, 3.0, 1e-10));
        assert!(validator.verify_result(3.0, 3.0000000001, 1e-9));
        assert!(!validator.verify_result(3.0, 4.0, 1e-10));
    }

    // =====================================================
    // Training Data Generator Tests
    // =====================================================

    #[test]
    fn test_generate_arithmetic() {
        let gen = TrainingDataGenerator::new();

        let examples = gen.generate_arithmetic(10, 2);
        assert_eq!(examples.len(), 10);

        // Each example should have a valid result
        for (expr, result) in &examples {
            assert!(result.is_finite());
            assert!(!expr.is_symbolic());
        }
    }

    #[test]
    fn test_generate_derivatives() {
        let gen = TrainingDataGenerator::new();

        let examples = gen.generate_derivatives(5);
        assert!(!examples.is_empty());

        // Each example should be symbolic
        for (expr, _deriv) in &examples {
            assert!(expr.is_symbolic());
        }
    }

    #[test]
    fn test_generate_simplifications() {
        let gen = TrainingDataGenerator::new();

        let examples = gen.generate_simplifications(5);
        assert!(!examples.is_empty());

        // First example should be (x + 0, x)
        let (unsimplified, simplified) = &examples[0];
        assert_eq!(*unsimplified, Expr::add(Expr::symbol("x"), Expr::int(0)));
        assert_eq!(*simplified, Expr::symbol("x"));
    }

    // =====================================================
    // MathEngine Extended Tests
    // =====================================================

    #[test]
    fn test_unbind_symbol() {
        let mut engine = MathEngine::new();
        engine.bind("x", Value::Integer(5));

        assert!(engine.is_bound("x"));
        assert!(!engine.is_bound("y"));

        engine.unbind("x");
        assert!(!engine.is_bound("x"));
    }

    #[test]
    fn test_clear_bindings() {
        let mut engine = MathEngine::new();
        engine.bind("x", Value::Integer(5));
        engine.bind("y", Value::Integer(10));

        assert_eq!(engine.bound_symbols().len(), 2);

        engine.clear_bindings();
        assert!(engine.bound_symbols().is_empty());
    }

    #[test]
    fn test_rational_evaluation() {
        let engine = MathEngine::new();

        // 1/2 = 0.5
        let expr = Expr::Value(Value::Rational(1, 2));
        let result = engine.evaluate(&expr).unwrap();
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_trig_functions() {
        let engine = MathEngine::new();

        // sin(0) = 0
        let sin_zero = Expr::func(MathFn::Sin, vec![Expr::int(0)]);
        let result = engine.evaluate(&sin_zero).unwrap();
        assert!(result.abs() < 1e-10);

        // cos(0) = 1
        let cos_zero = Expr::func(MathFn::Cos, vec![Expr::int(0)]);
        let result = engine.evaluate(&cos_zero).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_and_ln() {
        let engine = MathEngine::new();

        // exp(0) = 1
        let exp_zero = Expr::func(MathFn::Exp, vec![Expr::int(0)]);
        let result = engine.evaluate(&exp_zero).unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // ln(e) ≈ 1
        let ln_e = Expr::func(MathFn::Ln, vec![Expr::float(std::f64::consts::E)]);
        let result = engine.evaluate(&ln_e).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    // =====================================================
    // Integration Tests (backend-010)
    // =====================================================

    #[test]
    fn test_integrate_constant() {
        let symbolic = SymbolicEngine::new();

        // ∫5 dx = 5x
        let integral = symbolic.integrate(&Expr::int(5), "x").unwrap();
        assert_eq!(integral, Expr::mul(Expr::int(5), Expr::symbol("x")));
    }

    #[test]
    fn test_integrate_variable() {
        let symbolic = SymbolicEngine::new();

        // ∫x dx = x²/2
        let integral = symbolic.integrate(&Expr::symbol("x"), "x").unwrap();
        assert_eq!(
            integral,
            Expr::div(
                Expr::pow(Expr::symbol("x"), Expr::int(2)),
                Expr::int(2)
            )
        );
    }

    #[test]
    fn test_integrate_other_variable() {
        let symbolic = SymbolicEngine::new();

        // ∫y dx = yx (y is constant wrt x)
        let integral = symbolic.integrate(&Expr::symbol("y"), "x").unwrap();
        assert_eq!(integral, Expr::mul(Expr::symbol("y"), Expr::symbol("x")));
    }

    #[test]
    fn test_integrate_power_rule() {
        let symbolic = SymbolicEngine::new();

        // ∫x² dx = x³/3
        let expr = Expr::pow(Expr::symbol("x"), Expr::int(2));
        let integral = symbolic.integrate(&expr, "x").unwrap();
        assert_eq!(
            integral,
            Expr::div(
                Expr::pow(Expr::symbol("x"), Expr::int(3)),
                Expr::int(3)
            )
        );

        // ∫x³ dx = x⁴/4
        let expr = Expr::pow(Expr::symbol("x"), Expr::int(3));
        let integral = symbolic.integrate(&expr, "x").unwrap();
        assert_eq!(
            integral,
            Expr::div(
                Expr::pow(Expr::symbol("x"), Expr::int(4)),
                Expr::int(4)
            )
        );
    }

    #[test]
    fn test_integrate_power_rule_negative() {
        let symbolic = SymbolicEngine::new();

        // ∫x^(-1) dx = ln|x|
        let expr = Expr::pow(Expr::symbol("x"), Expr::int(-1));
        let integral = symbolic.integrate(&expr, "x").unwrap();
        assert_eq!(
            integral,
            Expr::func(MathFn::Ln, vec![Expr::func(MathFn::Abs, vec![Expr::symbol("x")])])
        );

        // ∫x^(-2) dx = x^(-1)/(-1)
        let expr = Expr::pow(Expr::symbol("x"), Expr::int(-2));
        let integral = symbolic.integrate(&expr, "x").unwrap();
        assert_eq!(
            integral,
            Expr::div(
                Expr::pow(Expr::symbol("x"), Expr::int(-1)),
                Expr::int(-1)
            )
        );
    }

    #[test]
    fn test_integrate_sum_rule() {
        let symbolic = SymbolicEngine::new();

        // ∫(x + 1) dx = x²/2 + x
        let expr = Expr::add(Expr::symbol("x"), Expr::int(1));
        let integral = symbolic.integrate(&expr, "x").unwrap();

        // Should be (x²/2) + (1*x)
        assert!(matches!(integral, Expr::BinOp { op: MathOp::Add, .. }));
    }

    #[test]
    fn test_integrate_difference_rule() {
        let symbolic = SymbolicEngine::new();

        // ∫(x - 1) dx = x²/2 - x
        let expr = Expr::sub(Expr::symbol("x"), Expr::int(1));
        let integral = symbolic.integrate(&expr, "x").unwrap();

        // Should be (x²/2) - (1*x)
        assert!(matches!(integral, Expr::BinOp { op: MathOp::Sub, .. }));
    }

    #[test]
    fn test_integrate_constant_multiple() {
        let symbolic = SymbolicEngine::new();

        // ∫3x dx = 3 * (x²/2)
        let expr = Expr::mul(Expr::int(3), Expr::symbol("x"));
        let integral = symbolic.integrate(&expr, "x").unwrap();

        // Should be 3 * (x²/2)
        assert!(matches!(integral, Expr::BinOp { op: MathOp::Mul, .. }));
    }

    #[test]
    fn test_integrate_sin() {
        let symbolic = SymbolicEngine::new();

        // ∫sin(x) dx = -cos(x)
        let expr = Expr::func(MathFn::Sin, vec![Expr::symbol("x")]);
        let integral = symbolic.integrate(&expr, "x").unwrap();

        // Should be -cos(x)
        assert_eq!(
            integral,
            Expr::neg(Expr::func(MathFn::Cos, vec![Expr::symbol("x")]))
        );
    }

    #[test]
    fn test_integrate_cos() {
        let symbolic = SymbolicEngine::new();

        // ∫cos(x) dx = sin(x)
        let expr = Expr::func(MathFn::Cos, vec![Expr::symbol("x")]);
        let integral = symbolic.integrate(&expr, "x").unwrap();

        assert_eq!(
            integral,
            Expr::func(MathFn::Sin, vec![Expr::symbol("x")])
        );
    }

    #[test]
    fn test_integrate_exp() {
        let symbolic = SymbolicEngine::new();

        // ∫exp(x) dx = exp(x)
        let expr = Expr::func(MathFn::Exp, vec![Expr::symbol("x")]);
        let integral = symbolic.integrate(&expr, "x").unwrap();

        assert_eq!(
            integral,
            Expr::func(MathFn::Exp, vec![Expr::symbol("x")])
        );
    }

    #[test]
    fn test_integrate_one_over_x() {
        let symbolic = SymbolicEngine::new();

        // ∫(1/x) dx = 1 * ln|x| (or simplified: ln|x|)
        let expr = Expr::div(Expr::int(1), Expr::symbol("x"));
        let integral = symbolic.integrate(&expr, "x").unwrap();

        // Should be 1 * ln|x| or just ln|x|
        // The implementation produces 1 * ln|x| which is mathematically equivalent
        let ln_abs_x = Expr::func(MathFn::Ln, vec![Expr::func(MathFn::Abs, vec![Expr::symbol("x")])]);
        let expected = Expr::mul(Expr::int(1), ln_abs_x.clone());
        assert!(integral == expected || integral == ln_abs_x);
    }

    #[test]
    fn test_integrate_k_over_x() {
        let symbolic = SymbolicEngine::new();

        // ∫(5/x) dx = 5 * ln|x|
        let expr = Expr::div(Expr::int(5), Expr::symbol("x"));
        let integral = symbolic.integrate(&expr, "x").unwrap();

        assert!(matches!(integral, Expr::BinOp { op: MathOp::Mul, .. }));
    }

    #[test]
    fn test_integrate_polynomial() {
        let symbolic = SymbolicEngine::new();

        // ∫(x² + 2x + 1) dx should succeed with sum rule
        let expr = Expr::add(
            Expr::add(
                Expr::pow(Expr::symbol("x"), Expr::int(2)),
                Expr::mul(Expr::int(2), Expr::symbol("x")),
            ),
            Expr::int(1),
        );
        let integral = symbolic.integrate(&expr, "x");

        assert!(integral.is_ok());
    }

    #[test]
    fn test_integrate_negation() {
        let symbolic = SymbolicEngine::new();

        // ∫(-x) dx = -(x²/2)
        let expr = Expr::neg(Expr::symbol("x"));
        let integral = symbolic.integrate(&expr, "x").unwrap();

        assert!(matches!(integral, Expr::UnaryOp { op: MathOp::Neg, .. }));
    }

    #[test]
    fn test_integrate_cannot_integrate() {
        let symbolic = SymbolicEngine::new();

        // ∫(x * x) dx - product of two expressions containing variable
        // This should fail for simple product (not recognized as x^2)
        let expr = Expr::mul(Expr::symbol("x"), Expr::symbol("x"));
        let result = symbolic.integrate(&expr, "x");

        // This is a product of two expressions containing variable
        // Our simple integrator doesn't handle this
        assert!(result.is_err());
    }

    #[test]
    fn test_integrate_tan() {
        let symbolic = SymbolicEngine::new();

        // ∫tan(x) dx = -ln|cos(x)|
        let expr = Expr::func(MathFn::Tan, vec![Expr::symbol("x")]);
        let integral = symbolic.integrate(&expr, "x").unwrap();

        // Should be -ln|cos(x)|
        assert!(matches!(integral, Expr::UnaryOp { op: MathOp::Neg, .. }));
    }

    #[test]
    fn test_integrate_function_of_constant() {
        let symbolic = SymbolicEngine::new();

        // ∫sin(2) dx = sin(2) * x
        let expr = Expr::func(MathFn::Sin, vec![Expr::int(2)]);
        let integral = symbolic.integrate(&expr, "x").unwrap();

        assert!(matches!(integral, Expr::BinOp { op: MathOp::Mul, .. }));
    }

    // ========================================================================
    // Equation Solving Tests (backend-011)
    // ========================================================================

    #[test]
    fn test_equation_new() {
        let eq = Equation::new(Expr::symbol("x"), Expr::int(5));
        assert!(matches!(eq.lhs, Expr::Value(Value::Symbol(_))));
        assert!(matches!(eq.rhs, Expr::Value(Value::Integer(5))));
    }

    #[test]
    fn test_equation_equals_zero() {
        let eq = Equation::equals_zero(Expr::symbol("x"));
        assert!(matches!(eq.rhs, Expr::Value(Value::Integer(0))));
    }

    #[test]
    fn test_solve_linear_simple() {
        let symbolic = SymbolicEngine::new();

        // x + 5 = 10 => x = 5
        let eq = Equation::new(
            Expr::add(Expr::symbol("x"), Expr::int(5)),
            Expr::int(10),
        );
        let solution = symbolic.solve(&eq, "x").unwrap();

        // Should be Single(5)
        assert!(matches!(solution, Solution::Single(_)));
    }

    #[test]
    fn test_solve_linear_coefficient() {
        let symbolic = SymbolicEngine::new();

        // 2x - 3 = 7 => 2x = 10 => x = 5
        let eq = Equation::new(
            Expr::sub(Expr::mul(Expr::int(2), Expr::symbol("x")), Expr::int(3)),
            Expr::int(7),
        );
        let solution = symbolic.solve(&eq, "x").unwrap();

        assert!(matches!(solution, Solution::Single(_)));
    }

    #[test]
    fn test_solve_identity() {
        let symbolic = SymbolicEngine::new();

        // x = x => infinite solutions
        let eq = Equation::new(Expr::symbol("x"), Expr::symbol("x"));
        let solution = symbolic.solve(&eq, "x").unwrap();

        assert!(matches!(solution, Solution::Infinite));
    }

    #[test]
    fn test_solve_contradiction() {
        let symbolic = SymbolicEngine::new();

        // 0 = 5 => no solution
        let eq = Equation::new(Expr::int(0), Expr::int(5));
        let solution = symbolic.solve(&eq, "x").unwrap();

        assert!(matches!(solution, Solution::NoSolution));
    }

    #[test]
    fn test_solve_quadratic_two_roots() {
        let symbolic = SymbolicEngine::new();

        // x² - 4 = 0 => x = ±2
        let eq = Equation::equals_zero(
            Expr::sub(Expr::pow(Expr::symbol("x"), Expr::int(2)), Expr::int(4)),
        );
        let solution = symbolic.solve(&eq, "x").unwrap();

        assert!(matches!(solution, Solution::Multiple(ref v) if v.len() == 2));
    }

    #[test]
    fn test_solve_quadratic_one_root() {
        let symbolic = SymbolicEngine::new();

        // x² + 2x + 1 = 0 => (x+1)² = 0 => x = -1 (repeated)
        let expr = Expr::add(
            Expr::add(
                Expr::pow(Expr::symbol("x"), Expr::int(2)),
                Expr::mul(Expr::int(2), Expr::symbol("x")),
            ),
            Expr::int(1),
        );
        let eq = Equation::equals_zero(expr);
        let solution = symbolic.solve(&eq, "x").unwrap();

        // Should be single solution (discriminant = 0)
        assert!(matches!(solution, Solution::Single(_)));
    }

    #[test]
    fn test_solve_quadratic_no_real_roots() {
        let symbolic = SymbolicEngine::new();

        // x² + 1 = 0 => no real solution
        let eq = Equation::equals_zero(
            Expr::add(Expr::pow(Expr::symbol("x"), Expr::int(2)), Expr::int(1)),
        );
        let solution = symbolic.solve(&eq, "x").unwrap();

        assert!(matches!(solution, Solution::NoSolution));
    }

    #[test]
    fn test_polynomial_degree() {
        let symbolic = SymbolicEngine::new();

        // constant: degree 0
        assert_eq!(symbolic.polynomial_degree(&Expr::int(5), "x"), 0);

        // x: degree 1
        assert_eq!(symbolic.polynomial_degree(&Expr::symbol("x"), "x"), 1);

        // x²: degree 2
        assert_eq!(
            symbolic.polynomial_degree(&Expr::pow(Expr::symbol("x"), Expr::int(2)), "x"),
            2
        );

        // x + 5: degree 1
        assert_eq!(
            symbolic.polynomial_degree(&Expr::add(Expr::symbol("x"), Expr::int(5)), "x"),
            1
        );
    }

    #[test]
    fn test_evaluate_numeric() {
        let symbolic = SymbolicEngine::new();

        // Integer
        assert_eq!(symbolic.evaluate_numeric(&Expr::int(5)), Some(5.0));

        // Float
        assert_eq!(symbolic.evaluate_numeric(&Expr::float(2.5)), Some(2.5));

        // Addition
        assert_eq!(
            symbolic.evaluate_numeric(&Expr::add(Expr::int(2), Expr::int(3))),
            Some(5.0)
        );

        // Symbol (not evaluable)
        assert_eq!(symbolic.evaluate_numeric(&Expr::symbol("x")), None);
    }

    #[test]
    fn test_solution_has_solution() {
        assert!(Solution::Single(Expr::int(5)).has_solution());
        assert!(Solution::Multiple(vec![Expr::int(1), Expr::int(2)]).has_solution());
        assert!(Solution::Infinite.has_solution());
        assert!(!Solution::NoSolution.has_solution());
    }

    #[test]
    fn test_solution_solutions() {
        let single = Solution::Single(Expr::int(5));
        assert_eq!(single.solutions().len(), 1);

        let multiple = Solution::Multiple(vec![Expr::int(1), Expr::int(2)]);
        assert_eq!(multiple.solutions().len(), 2);

        let none = Solution::NoSolution;
        assert!(none.solutions().is_empty());

        let infinite = Solution::Infinite;
        assert!(infinite.solutions().is_empty());
    }
}
