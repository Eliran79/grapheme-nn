//! Benchmarks for grapheme-engine
//!
//! Comprehensive benchmarks covering:
//! - Basic expression evaluation
//! - Nested expression evaluation
//! - Symbolic differentiation
//! - Expression substitution
//! - Function evaluation
//! - Expression building and analysis

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grapheme_engine::{Expr, MathEngine, MathFn, SymbolicEngine};

// ============================================================================
// Basic Evaluation Benchmarks
// ============================================================================

fn bench_evaluate_simple(c: &mut Criterion) {
    let engine = MathEngine::new();
    let expr = Expr::add(Expr::int(2), Expr::int(3));

    c.bench_function("evaluate_simple", |b| {
        b.iter(|| engine.evaluate(black_box(&expr)))
    });
}

fn bench_evaluate_nested(c: &mut Criterion) {
    let engine = MathEngine::new();
    let expr = Expr::mul(
        Expr::add(Expr::int(2), Expr::int(3)),
        Expr::sub(Expr::int(10), Expr::int(4)),
    );

    c.bench_function("evaluate_nested", |b| {
        b.iter(|| engine.evaluate(black_box(&expr)))
    });
}

fn bench_evaluate_deeply_nested(c: &mut Criterion) {
    let engine = MathEngine::new();
    // Create a deeply nested expression: (((1 + 2) + 3) + 4) + 5
    let mut expr = Expr::int(1);
    for i in 2..=10 {
        expr = Expr::add(expr, Expr::int(i));
    }

    c.bench_function("evaluate_deeply_nested_10", |b| {
        b.iter(|| engine.evaluate(black_box(&expr)))
    });
}

// ============================================================================
// Function Evaluation Benchmarks
// ============================================================================

fn bench_evaluate_sin(c: &mut Criterion) {
    let engine = MathEngine::new();
    let expr = Expr::func(MathFn::Sin, vec![Expr::float(0.5)]);

    c.bench_function("evaluate_sin", |b| {
        b.iter(|| engine.evaluate(black_box(&expr)))
    });
}

fn bench_evaluate_compound_func(c: &mut Criterion) {
    let engine = MathEngine::new();
    // sin(x)^2 + cos(x)^2 should equal 1
    let expr = Expr::add(
        Expr::pow(
            Expr::func(MathFn::Sin, vec![Expr::float(1.0)]),
            Expr::int(2),
        ),
        Expr::pow(
            Expr::func(MathFn::Cos, vec![Expr::float(1.0)]),
            Expr::int(2),
        ),
    );

    c.bench_function("evaluate_compound_trig", |b| {
        b.iter(|| engine.evaluate(black_box(&expr)))
    });
}

// ============================================================================
// Symbolic Engine Benchmarks
// ============================================================================

fn bench_differentiate_simple(c: &mut Criterion) {
    let engine = SymbolicEngine::new();
    // d/dx(x^2) = 2x
    let expr = Expr::pow(Expr::symbol("x"), Expr::int(2));

    c.bench_function("differentiate_x_squared", |b| {
        b.iter(|| engine.differentiate(black_box(&expr), "x"))
    });
}

fn bench_differentiate_polynomial(c: &mut Criterion) {
    let engine = SymbolicEngine::new();
    // d/dx(x^3 + 2x^2 + 3x + 4)
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

    c.bench_function("differentiate_polynomial", |b| {
        b.iter(|| engine.differentiate(black_box(&expr), "x"))
    });
}

fn bench_differentiate_trig(c: &mut Criterion) {
    let engine = SymbolicEngine::new();
    // d/dx(sin(x))
    let expr = Expr::func(MathFn::Sin, vec![Expr::symbol("x")]);

    c.bench_function("differentiate_sin", |b| {
        b.iter(|| engine.differentiate(black_box(&expr), "x"))
    });
}

fn bench_substitute(c: &mut Criterion) {
    let engine = SymbolicEngine::new();
    // Substitute y for x in (x^2 + 2x + 1)
    let expr = Expr::add(
        Expr::add(
            Expr::pow(Expr::symbol("x"), Expr::int(2)),
            Expr::mul(Expr::int(2), Expr::symbol("x")),
        ),
        Expr::int(1),
    );
    let replacement = Expr::symbol("y");

    c.bench_function("substitute_variable", |b| {
        b.iter(|| engine.substitute(black_box(&expr), "x", black_box(&replacement)))
    });
}

fn bench_evaluate_at(c: &mut Criterion) {
    let engine = SymbolicEngine::new();
    // Evaluate x^2 + 2x + 1 at x=3
    let expr = Expr::add(
        Expr::add(
            Expr::pow(Expr::symbol("x"), Expr::int(2)),
            Expr::mul(Expr::int(2), Expr::symbol("x")),
        ),
        Expr::int(1),
    );

    c.bench_function("evaluate_at_point", |b| {
        b.iter(|| engine.evaluate_at(black_box(&expr), "x", 3.0))
    });
}

// ============================================================================
// Expression Building and Analysis Benchmarks
// ============================================================================

fn bench_expr_builders(c: &mut Criterion) {
    c.bench_function("expr_build_complex", |b| {
        b.iter(|| {
            // Build: (x + 2) * (y - 3)
            Expr::mul(
                Expr::add(Expr::symbol("x"), Expr::int(2)),
                Expr::sub(Expr::symbol("y"), Expr::int(3)),
            )
        })
    });
}

fn bench_collect_symbols(c: &mut Criterion) {
    let expr = Expr::add(
        Expr::mul(Expr::symbol("x"), Expr::symbol("y")),
        Expr::mul(Expr::symbol("z"), Expr::symbol("x")),
    );

    c.bench_function("collect_symbols", |b| {
        b.iter(|| black_box(&expr).collect_symbols())
    });
}

fn bench_expr_depth(c: &mut Criterion) {
    // Create a deeply nested expression
    let mut expr = Expr::int(1);
    for i in 2..=10 {
        expr = Expr::add(expr, Expr::int(i));
    }

    c.bench_function("expr_depth", |b| b.iter(|| black_box(&expr).depth()));
}

fn bench_expr_node_count(c: &mut Criterion) {
    // Create a moderately complex expression
    let expr = Expr::add(
        Expr::mul(Expr::symbol("x"), Expr::symbol("y")),
        Expr::pow(Expr::symbol("z"), Expr::int(2)),
    );

    c.bench_function("expr_node_count", |b| {
        b.iter(|| black_box(&expr).node_count())
    });
}

// ============================================================================
// Scaling Benchmarks
// ============================================================================

fn bench_evaluation_scaling(c: &mut Criterion) {
    let engine = MathEngine::new();
    let mut group = c.benchmark_group("evaluation_scaling");

    for depth in [5, 10, 20, 50].iter() {
        let mut expr = Expr::int(1);
        for i in 2..=*depth {
            expr = Expr::add(expr, Expr::int(i as i64));
        }

        group.bench_with_input(BenchmarkId::new("depth", depth), &expr, |b, expr| {
            b.iter(|| engine.evaluate(black_box(expr)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    // Basic evaluation
    bench_evaluate_simple,
    bench_evaluate_nested,
    bench_evaluate_deeply_nested,
    // Function evaluation
    bench_evaluate_sin,
    bench_evaluate_compound_func,
    // Symbolic engine
    bench_differentiate_simple,
    bench_differentiate_polynomial,
    bench_differentiate_trig,
    bench_substitute,
    bench_evaluate_at,
    // Expression building and analysis
    bench_expr_builders,
    bench_collect_symbols,
    bench_expr_depth,
    bench_expr_node_count,
    // Scaling
    bench_evaluation_scaling,
);
criterion_main!(benches);
