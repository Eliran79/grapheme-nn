//! Benchmarks for grapheme-math
//!
//! Comprehensive benchmarks covering:
//! - Graph construction from expressions
//! - Graph to expression conversion
//! - Graph equality comparison
//! - Large graph operations
//! - Scaling with expression complexity

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grapheme_engine::{Expr, MathFn};
use grapheme_math::MathGraph;

// ============================================================================
// Graph Construction Benchmarks
// ============================================================================

fn bench_graph_from_expr_simple(c: &mut Criterion) {
    let expr = Expr::add(Expr::int(2), Expr::int(3));

    c.bench_function("graph_from_expr_simple", |b| {
        b.iter(|| MathGraph::from_expr(black_box(&expr)))
    });
}

fn bench_graph_from_expr_nested(c: &mut Criterion) {
    let expr = Expr::mul(
        Expr::add(Expr::int(2), Expr::int(3)),
        Expr::int(4),
    );

    c.bench_function("graph_from_expr_nested", |b| {
        b.iter(|| MathGraph::from_expr(black_box(&expr)))
    });
}

fn bench_graph_from_expr_deeply_nested(c: &mut Criterion) {
    // Create a deeply nested expression: (((1 + 2) + 3) + 4) + 5
    let mut expr = Expr::int(1);
    for i in 2..=10 {
        expr = Expr::add(expr, Expr::int(i));
    }

    c.bench_function("graph_from_expr_deeply_nested", |b| {
        b.iter(|| MathGraph::from_expr(black_box(&expr)))
    });
}

fn bench_graph_from_expr_with_functions(c: &mut Criterion) {
    // sin(x)^2 + cos(x)^2
    let expr = Expr::add(
        Expr::pow(
            Expr::func(MathFn::Sin, vec![Expr::symbol("x")]),
            Expr::int(2),
        ),
        Expr::pow(
            Expr::func(MathFn::Cos, vec![Expr::symbol("x")]),
            Expr::int(2),
        ),
    );

    c.bench_function("graph_from_expr_with_functions", |b| {
        b.iter(|| MathGraph::from_expr(black_box(&expr)))
    });
}

fn bench_graph_from_expr_polynomial(c: &mut Criterion) {
    // x^3 + 2x^2 + 3x + 4
    let expr = Expr::add(
        Expr::add(
            Expr::add(
                Expr::pow(Expr::symbol("x"), Expr::int(3)),
                Expr::mul(Expr::int(2), Expr::pow(Expr::symbol("x"), Expr::int(2))),
            ),
            Expr::mul(Expr::int(3), Expr::symbol("x")),
        ),
        Expr::int(4),
    );

    c.bench_function("graph_from_expr_polynomial", |b| {
        b.iter(|| MathGraph::from_expr(black_box(&expr)))
    });
}

// ============================================================================
// Graph to Expression Conversion Benchmarks
// ============================================================================

fn bench_graph_to_expr_simple(c: &mut Criterion) {
    let expr = Expr::add(Expr::int(2), Expr::int(3));
    let graph = MathGraph::from_expr(&expr);

    c.bench_function("graph_to_expr_simple", |b| {
        b.iter(|| black_box(&graph).to_expr())
    });
}

fn bench_graph_to_expr_nested(c: &mut Criterion) {
    let expr = Expr::mul(
        Expr::add(Expr::int(2), Expr::int(3)),
        Expr::int(4),
    );
    let graph = MathGraph::from_expr(&expr);

    c.bench_function("graph_to_expr_nested", |b| {
        b.iter(|| black_box(&graph).to_expr())
    });
}

fn bench_graph_to_expr_deeply_nested(c: &mut Criterion) {
    let mut expr = Expr::int(1);
    for i in 2..=10 {
        expr = Expr::add(expr, Expr::int(i));
    }
    let graph = MathGraph::from_expr(&expr);

    c.bench_function("graph_to_expr_deeply_nested", |b| {
        b.iter(|| black_box(&graph).to_expr())
    });
}

// ============================================================================
// Round-trip Benchmarks
// ============================================================================

fn bench_roundtrip(c: &mut Criterion) {
    let expr = Expr::mul(
        Expr::add(Expr::int(2), Expr::int(3)),
        Expr::sub(Expr::int(10), Expr::int(4)),
    );

    c.bench_function("roundtrip_expr_graph_expr", |b| {
        b.iter(|| {
            let graph = MathGraph::from_expr(black_box(&expr));
            graph.to_expr()
        })
    });
}

// ============================================================================
// Graph Operations Benchmarks
// ============================================================================

fn bench_graph_node_count(c: &mut Criterion) {
    let mut expr = Expr::int(1);
    for i in 2..=10 {
        expr = Expr::add(expr, Expr::int(i));
    }
    let graph = MathGraph::from_expr(&expr);

    c.bench_function("graph_node_count", |b| {
        b.iter(|| black_box(&graph).node_count())
    });
}

fn bench_graph_edge_count(c: &mut Criterion) {
    let mut expr = Expr::int(1);
    for i in 2..=10 {
        expr = Expr::add(expr, Expr::int(i));
    }
    let graph = MathGraph::from_expr(&expr);

    c.bench_function("graph_edge_count", |b| {
        b.iter(|| black_box(&graph).edge_count())
    });
}

// ============================================================================
// Scaling Benchmarks
// ============================================================================

fn bench_graph_construction_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_construction_scaling");

    for depth in [5, 10, 20, 50].iter() {
        let mut expr = Expr::int(1);
        for i in 2..=*depth {
            expr = Expr::add(expr, Expr::int(i as i64));
        }

        group.bench_with_input(BenchmarkId::new("depth", depth), &expr, |b, expr| {
            b.iter(|| MathGraph::from_expr(black_box(expr)))
        });
    }

    group.finish();
}

fn bench_graph_conversion_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_conversion_scaling");

    for depth in [5, 10, 20, 50].iter() {
        let mut expr = Expr::int(1);
        for i in 2..=*depth {
            expr = Expr::add(expr, Expr::int(i as i64));
        }
        let graph = MathGraph::from_expr(&expr);

        group.bench_with_input(BenchmarkId::new("depth", depth), &graph, |b, graph| {
            b.iter(|| black_box(graph).to_expr())
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    // Graph construction
    bench_graph_from_expr_simple,
    bench_graph_from_expr_nested,
    bench_graph_from_expr_deeply_nested,
    bench_graph_from_expr_with_functions,
    bench_graph_from_expr_polynomial,
    // Graph to expression
    bench_graph_to_expr_simple,
    bench_graph_to_expr_nested,
    bench_graph_to_expr_deeply_nested,
    // Round-trip
    bench_roundtrip,
    // Graph operations
    bench_graph_node_count,
    bench_graph_edge_count,
    // Scaling
    bench_graph_construction_scaling,
    bench_graph_conversion_scaling,
);
criterion_main!(benches);
