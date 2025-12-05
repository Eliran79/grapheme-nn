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
}

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
        assert_eq!(Expr::float(3.14), Expr::Value(Value::Float(3.14)));
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
}
