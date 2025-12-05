//! Benchmarks for grapheme-math

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_engine::{Expr, MathOp, Value};
use grapheme_math::MathGraph;

fn bench_graph_from_expr(c: &mut Criterion) {
    let expr = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        }),
        right: Box::new(Expr::Value(Value::Integer(4))),
    };

    c.bench_function("graph_from_expr", |b| {
        b.iter(|| MathGraph::from_expr(black_box(&expr)))
    });
}

fn bench_graph_to_expr(c: &mut Criterion) {
    let expr = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        }),
        right: Box::new(Expr::Value(Value::Integer(4))),
    };
    let graph = MathGraph::from_expr(&expr);

    c.bench_function("graph_to_expr", |b| {
        b.iter(|| black_box(&graph).to_expr())
    });
}

criterion_group!(benches, bench_graph_from_expr, bench_graph_to_expr);
criterion_main!(benches);
