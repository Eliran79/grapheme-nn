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
#[derive(Debug, Default)]
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
