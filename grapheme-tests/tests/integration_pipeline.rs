//! Integration tests for end-to-end pipeline functionality
//!
//! Tests the full pipeline from expression creation through evaluation

use grapheme_engine::{Expr, MathEngine, MathOp, Value};
use grapheme_polish::PolishGraph;

/// Test basic expression evaluation through the engine
#[test]
fn test_basic_expression_evaluation() {
    let engine = MathEngine::new();

    // Test: 2 + 3 = 5
    let expr = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Integer(2))),
        right: Box::new(Expr::Value(Value::Integer(3))),
    };

    let result = engine.evaluate(&expr).expect("evaluation should succeed");
    assert!((result - 5.0).abs() < 1e-10, "2 + 3 should equal 5");
}

/// Test nested expression evaluation
#[test]
fn test_nested_expression_evaluation() {
    let engine = MathEngine::new();

    // Test: (2 + 3) * 4 = 20
    let inner = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Integer(2))),
        right: Box::new(Expr::Value(Value::Integer(3))),
    };

    let expr = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(inner),
        right: Box::new(Expr::Value(Value::Integer(4))),
    };

    let result = engine.evaluate(&expr).expect("evaluation should succeed");
    assert!((result - 20.0).abs() < 1e-10, "(2 + 3) * 4 should equal 20");
}

/// Test expression to Polish graph conversion and back
#[test]
fn test_expression_to_polish_roundtrip() {
    // Create expression: 2 + 3
    let expr = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Integer(2))),
        right: Box::new(Expr::Value(Value::Integer(3))),
    };

    // Convert to Polish graph
    let graph = PolishGraph::from_expr(&expr);
    assert!(graph.node_count() > 0, "graph should have nodes");

    // Convert back to expression
    let recovered = graph.to_expr();
    assert!(recovered.is_some(), "should recover expression from graph");
}

/// Test complex expression graph conversion
#[test]
fn test_complex_expression_graph() {
    // Create: ((1 + 2) * 3) - 4
    let add = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Integer(1))),
        right: Box::new(Expr::Value(Value::Integer(2))),
    };

    let mul = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(add),
        right: Box::new(Expr::Value(Value::Integer(3))),
    };

    let expr = Expr::BinOp {
        op: MathOp::Sub,
        left: Box::new(mul),
        right: Box::new(Expr::Value(Value::Integer(4))),
    };

    // Convert to graph and back
    let graph = PolishGraph::from_expr(&expr);
    let recovered = graph.to_expr();

    assert!(recovered.is_some(), "should recover complex expression");

    // Evaluate both and compare results
    let engine = MathEngine::new();
    let original_result = engine.evaluate(&expr).expect("original evaluation");
    let recovered_result = engine
        .evaluate(&recovered.unwrap())
        .expect("recovered evaluation");

    assert!(
        (original_result - recovered_result).abs() < 1e-10,
        "original and recovered should produce same result"
    );
}

/// Test float expression pipeline
#[test]
fn test_float_expression_pipeline() {
    let engine = MathEngine::new();

    // Test: 3.15 * 2.0 = 6.30 (using non-PI value to avoid clippy approx_constant warning)
    let expr = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(Expr::Value(Value::Float(3.15))),
        right: Box::new(Expr::Value(Value::Float(2.0))),
    };

    let result = engine.evaluate(&expr).expect("should evaluate");
    assert!((result - 6.30).abs() < 1e-10);
}

/// Test division operations
#[test]
fn test_division_operations() {
    let engine = MathEngine::new();

    // Test: 10 / 2 = 5
    let expr = Expr::BinOp {
        op: MathOp::Div,
        left: Box::new(Expr::Value(Value::Integer(10))),
        right: Box::new(Expr::Value(Value::Integer(2))),
    };

    let result = engine.evaluate(&expr).expect("should evaluate");
    assert!((result - 5.0).abs() < 1e-10, "10 / 2 should equal 5");
}

/// Test power operations
#[test]
fn test_power_operations() {
    let engine = MathEngine::new();

    // Test: 2 ^ 3 = 8
    let expr = Expr::BinOp {
        op: MathOp::Pow,
        left: Box::new(Expr::Value(Value::Integer(2))),
        right: Box::new(Expr::Value(Value::Integer(3))),
    };

    let result = engine.evaluate(&expr).expect("should evaluate");
    assert!((result - 8.0).abs() < 1e-10, "2 ^ 3 should equal 8");
}

/// Test symbolic expression handling
#[test]
fn test_symbolic_expression() {
    // Create expression with symbol: x + 1
    let expr = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
        right: Box::new(Expr::Value(Value::Integer(1))),
    };

    // Should convert to graph without panic
    let graph = PolishGraph::from_expr(&expr);
    assert!(graph.node_count() > 0);
}

/// Test multiple sequential evaluations
#[test]
fn test_multiple_evaluations() {
    let engine = MathEngine::new();

    for i in 1..=10 {
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::Value(Value::Integer(i))),
            right: Box::new(Expr::Value(Value::Integer(i))),
        };

        let result = engine.evaluate(&expr).expect("should evaluate");
        assert!(
            (result - (i * i) as f64).abs() < 1e-10,
            "i*i should equal {}",
            i * i
        );
    }
}

/// Test graph node counting consistency
#[test]
fn test_graph_node_consistency() {
    // Simple value: 1 node
    let simple = Expr::Value(Value::Integer(42));
    let simple_graph = PolishGraph::from_expr(&simple);
    assert_eq!(simple_graph.node_count(), 1);

    // Binary op: 3 nodes (op + 2 values)
    let binary = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Integer(1))),
        right: Box::new(Expr::Value(Value::Integer(2))),
    };
    let binary_graph = PolishGraph::from_expr(&binary);
    assert!(binary_graph.node_count() >= 3);
}

/// Test subtraction operations
#[test]
fn test_subtraction_operations() {
    let engine = MathEngine::new();

    // Test: 10 - 3 = 7
    let expr = Expr::BinOp {
        op: MathOp::Sub,
        left: Box::new(Expr::Value(Value::Integer(10))),
        right: Box::new(Expr::Value(Value::Integer(3))),
    };

    let result = engine.evaluate(&expr).expect("should evaluate");
    assert!((result - 7.0).abs() < 1e-10, "10 - 3 should equal 7");
}

/// Test chained operations
#[test]
fn test_chained_operations() {
    let engine = MathEngine::new();

    // Test: 1 + 2 + 3 + 4 + 5 = 15
    let mut expr = Expr::Value(Value::Integer(1));
    for i in 2..=5 {
        expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(expr),
            right: Box::new(Expr::Value(Value::Integer(i))),
        };
    }

    let result = engine.evaluate(&expr).expect("should evaluate");
    assert!((result - 15.0).abs() < 1e-10, "1+2+3+4+5 should equal 15");
}
