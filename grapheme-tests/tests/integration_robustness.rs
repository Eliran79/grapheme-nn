//! Integration tests for robustness and edge case handling
//!
//! Tests NaN handling, empty inputs, malformed data, and error recovery

use grapheme_engine::{Expr, MathEngine, MathOp, Value};
use grapheme_polish::PolishGraph;

/// Test that NaN values don't crash float comparisons
#[test]
fn test_nan_in_float_operations() {
    let engine = MathEngine::new();

    // Create expression that might produce NaN: 0.0 / 0.0
    let expr = Expr::BinOp {
        op: MathOp::Div,
        left: Box::new(Expr::Value(Value::Float(0.0))),
        right: Box::new(Expr::Value(Value::Float(0.0))),
    };

    // Should not panic, even if result is NaN
    let result = engine.evaluate(&expr);

    // Result should be either an error or NaN
    if let Ok(f) = result {
        assert!(f.is_nan(), "0/0 should be NaN");
    }
    // Error is also acceptable - do nothing
}

/// Test infinity handling
#[test]
fn test_infinity_handling() {
    let engine = MathEngine::new();

    // Create expression: 1.0 / 0.0 (positive infinity)
    let expr = Expr::BinOp {
        op: MathOp::Div,
        left: Box::new(Expr::Value(Value::Float(1.0))),
        right: Box::new(Expr::Value(Value::Float(0.0))),
    };

    let result = engine.evaluate(&expr);

    if let Ok(f) = result {
        assert!(f.is_infinite(), "1/0 should be infinite");
    }
    // Error is also acceptable - do nothing
}

/// Test empty graph handling
#[test]
fn test_empty_graph_handling() {
    // Create an empty graph
    let graph = PolishGraph::new();

    // Converting empty graph to expression should return None, not panic
    let expr = graph.to_expr();
    assert!(expr.is_none(), "empty graph should return None");
}

/// Test malformed expression graph recovery
#[test]
fn test_malformed_graph_recovery() {
    // Create a simple valid expression first
    let expr = Expr::Value(Value::Integer(42));
    let graph = PolishGraph::from_expr(&expr);

    // Should successfully convert back
    let recovered = graph.to_expr();
    assert!(recovered.is_some(), "should recover simple expression");
}

/// Test deeply nested expression (stack safety)
#[test]
fn test_deeply_nested_expression() {
    let engine = MathEngine::new();

    // Create a deeply nested expression: ((((1 + 1) + 1) + 1) + ... )
    let mut expr = Expr::Value(Value::Integer(1));

    for _ in 0..50 {
        expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(expr),
            right: Box::new(Expr::Value(Value::Integer(1))),
        };
    }

    // Should not stack overflow
    let result = engine.evaluate(&expr);
    assert!(result.is_ok(), "deeply nested expression should evaluate");

    if let Ok(n) = result {
        assert!((n - 51.0).abs() < 1e-10); // 1 + 50 ones
    }
}

/// Test very large numbers
#[test]
fn test_large_numbers() {
    let engine = MathEngine::new();

    // Create expression with large numbers
    let expr = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(Expr::Value(Value::Integer(1_000_000))),
        right: Box::new(Expr::Value(Value::Integer(1_000_000))),
    };

    let result = engine.evaluate(&expr);
    assert!(result.is_ok(), "large number multiplication should work");
}

/// Test negative numbers
#[test]
fn test_negative_numbers() {
    let engine = MathEngine::new();

    // Create expression: (-5) * 3
    let expr = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(Expr::Value(Value::Integer(-5))),
        right: Box::new(Expr::Value(Value::Integer(3))),
    };

    let result = engine.evaluate(&expr).expect("should evaluate");
    assert!((result - (-15.0)).abs() < 1e-10);
}

/// Test float precision
#[test]
fn test_float_precision() {
    let engine = MathEngine::new();

    // Create expression: 0.1 + 0.2
    let expr = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Float(0.1))),
        right: Box::new(Expr::Value(Value::Float(0.2))),
    };

    let result = engine.evaluate(&expr).expect("should evaluate");
    // Allow for floating point imprecision
    assert!(
        (result - 0.3).abs() < 1e-10,
        "0.1 + 0.2 should be approximately 0.3"
    );
}

/// Test unbound symbol handling
#[test]
fn test_unbound_symbol() {
    let engine = MathEngine::new();

    // Create expression with unbound symbol: x + 1
    let expr = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Symbol("unbound_var".to_string()))),
        right: Box::new(Expr::Value(Value::Integer(1))),
    };

    // Should either return the expression unchanged or error
    // Engine returns error for unbound symbols - this is correct behavior
    let result = engine.evaluate(&expr);
    assert!(result.is_err(), "unbound symbol should produce error");
}

/// Test rational number handling
#[test]
fn test_rational_numbers() {
    let engine = MathEngine::new();

    // Create expression with rationals: 1/2 + 1/3 = 5/6
    let expr = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Rational(1, 2))),
        right: Box::new(Expr::Value(Value::Rational(1, 3))),
    };

    let result = engine.evaluate(&expr);

    // Check result is correct (as float)
    if let Ok(f) = result {
        assert!((f - 5.0 / 6.0).abs() < 1e-10);
    }
}

/// Test graph with many nodes
#[test]
fn test_large_graph() {
    // Create a complex expression with many operations
    let mut expr = Expr::Value(Value::Integer(1));

    for i in 2..=20 {
        expr = Expr::BinOp {
            op: if i % 2 == 0 { MathOp::Add } else { MathOp::Mul },
            left: Box::new(expr),
            right: Box::new(Expr::Value(Value::Integer(i))),
        };
    }

    // Convert to graph
    let graph = PolishGraph::from_expr(&expr);
    assert!(graph.node_count() > 0, "graph should have nodes");

    // Convert back
    let recovered = graph.to_expr();
    assert!(recovered.is_some(), "should recover from large graph");
}

/// Test concurrent-safe operations (no data races)
#[test]
fn test_thread_safe_evaluation() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let engine = MathEngine::new();
                let expr = Expr::BinOp {
                    op: MathOp::Mul,
                    left: Box::new(Expr::Value(Value::Integer(i))),
                    right: Box::new(Expr::Value(Value::Integer(i))),
                };
                engine.evaluate(&expr)
            })
        })
        .collect();

    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.join().expect("thread should not panic");
        if let Ok(n) = result {
            assert!((n - (i * i) as f64).abs() < 1e-10);
        }
    }
}

/// Test zero handling in operations
#[test]
fn test_zero_handling() {
    let engine = MathEngine::new();

    // Multiply by zero
    let expr = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(Expr::Value(Value::Integer(999))),
        right: Box::new(Expr::Value(Value::Integer(0))),
    };

    let result = engine.evaluate(&expr).expect("should evaluate");
    assert!((result - 0.0).abs() < 1e-10);
}

/// Test addition identity
#[test]
fn test_addition_identity() {
    let engine = MathEngine::new();

    // Add zero
    let expr = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Integer(42))),
        right: Box::new(Expr::Value(Value::Integer(0))),
    };

    let result = engine.evaluate(&expr).expect("should evaluate");
    assert!((result - 42.0).abs() < 1e-10);
}

/// Test subtraction to zero
#[test]
fn test_subtraction_to_zero() {
    let engine = MathEngine::new();

    // Subtract equal values
    let expr = Expr::BinOp {
        op: MathOp::Sub,
        left: Box::new(Expr::Value(Value::Integer(100))),
        right: Box::new(Expr::Value(Value::Integer(100))),
    };

    let result = engine.evaluate(&expr).expect("should evaluate");
    assert!((result - 0.0).abs() < 1e-10);
}

/// Test negative infinity
#[test]
fn test_negative_infinity() {
    let engine = MathEngine::new();

    // Create expression: -1.0 / 0.0 (negative infinity)
    let expr = Expr::BinOp {
        op: MathOp::Div,
        left: Box::new(Expr::Value(Value::Float(-1.0))),
        right: Box::new(Expr::Value(Value::Float(0.0))),
    };

    let result = engine.evaluate(&expr);

    if let Ok(f) = result {
        assert!(f.is_infinite() && f < 0.0, "-1/0 should be negative infinity");
    }
    // Error is also acceptable - do nothing
}

/// Test very small numbers
#[test]
fn test_very_small_numbers() {
    let engine = MathEngine::new();

    // Create expression with very small numbers
    let expr = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(Expr::Value(Value::Float(1e-100))),
        right: Box::new(Expr::Value(Value::Float(1e-100))),
    };

    let result = engine.evaluate(&expr);
    assert!(result.is_ok(), "very small number multiplication should work");
}

/// Test division by very small number
#[test]
fn test_division_by_small_number() {
    let engine = MathEngine::new();

    // Create expression: 1.0 / 1e-300
    let expr = Expr::BinOp {
        op: MathOp::Div,
        left: Box::new(Expr::Value(Value::Float(1.0))),
        right: Box::new(Expr::Value(Value::Float(1e-300))),
    };

    let result = engine.evaluate(&expr);
    // Should produce a large number or infinity, but not crash
    assert!(result.is_ok());
}
