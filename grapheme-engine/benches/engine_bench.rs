//! Benchmarks for grapheme-engine

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_engine::{Expr, MathEngine, MathOp, Value};

fn bench_evaluate_simple(c: &mut Criterion) {
    let engine = MathEngine::new();
    let expr = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Integer(2))),
        right: Box::new(Expr::Value(Value::Integer(3))),
    };

    c.bench_function("evaluate_simple", |b| {
        b.iter(|| engine.evaluate(black_box(&expr)))
    });
}

fn bench_evaluate_nested(c: &mut Criterion) {
    let engine = MathEngine::new();
    let expr = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        }),
        right: Box::new(Expr::BinOp {
            op: MathOp::Sub,
            left: Box::new(Expr::Value(Value::Integer(10))),
            right: Box::new(Expr::Value(Value::Integer(4))),
        }),
    };

    c.bench_function("evaluate_nested", |b| {
        b.iter(|| engine.evaluate(black_box(&expr)))
    });
}

criterion_group!(benches, bench_evaluate_simple, bench_evaluate_nested);
criterion_main!(benches);
