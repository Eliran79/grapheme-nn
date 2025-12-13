//! Benchmarks for grapheme-polish
//!
//! Comprehensive benchmarks covering:
//! - Simple expression parsing
//! - Nested expression parsing
//! - Function call parsing
//! - Complex expression parsing
//! - Expression to polish conversion
//! - Scaling with expression complexity

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grapheme_engine::{Expr, MathFn};
use grapheme_polish::{expr_to_polish, PolishParser};

// ============================================================================
// Parsing Benchmarks
// ============================================================================

fn bench_parse_simple(c: &mut Criterion) {
    let mut parser = PolishParser::new();

    c.bench_function("parse_simple", |b| {
        b.iter(|| parser.parse(black_box("(+ 2 3)")))
    });
}

fn bench_parse_nested(c: &mut Criterion) {
    let mut parser = PolishParser::new();

    c.bench_function("parse_nested", |b| {
        b.iter(|| parser.parse(black_box("(* (+ 2 3) (- 10 4))")))
    });
}

fn bench_parse_deeply_nested(c: &mut Criterion) {
    let mut parser = PolishParser::new();
    // (+ (+ (+ (+ (+ 1 2) 3) 4) 5) 6)
    let input = "(+ (+ (+ (+ (+ 1 2) 3) 4) 5) 6)";

    c.bench_function("parse_deeply_nested", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

fn bench_parse_function_simple(c: &mut Criterion) {
    let mut parser = PolishParser::new();

    c.bench_function("parse_sin", |b| {
        b.iter(|| parser.parse(black_box("(sin 0.5)")))
    });
}

fn bench_parse_function_nested(c: &mut Criterion) {
    let mut parser = PolishParser::new();
    // sin(cos(x))
    let input = "(sin (cos 1.0))";

    c.bench_function("parse_sin_cos", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

fn bench_parse_compound_expression(c: &mut Criterion) {
    let mut parser = PolishParser::new();
    // (sin x)^2 + (cos x)^2
    let input = "(+ (^ (sin 1.0) 2) (^ (cos 1.0) 2))";

    c.bench_function("parse_compound_trig", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

fn bench_parse_polynomial(c: &mut Criterion) {
    let mut parser = PolishParser::new();
    // x^3 + 2x^2 + 3x + 4
    let input = "(+ (+ (+ (^ x 3) (* 2 (^ x 2))) (* 3 x)) 4)";

    c.bench_function("parse_polynomial", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

fn bench_parse_with_symbols(c: &mut Criterion) {
    let mut parser = PolishParser::new();
    // (x + y) * (z - w)
    let input = "(* (+ x y) (- z w))";

    c.bench_function("parse_with_symbols", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

fn bench_parse_floats(c: &mut Criterion) {
    let mut parser = PolishParser::new();
    let input = "(+ 3.14159 2.71828)";

    c.bench_function("parse_floats", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

fn bench_parse_negative(c: &mut Criterion) {
    let mut parser = PolishParser::new();
    let input = "(+ (- 5) 3)";

    c.bench_function("parse_negative", |b| {
        b.iter(|| parser.parse(black_box(input)))
    });
}

// ============================================================================
// Conversion Benchmarks (Expr to Polish)
// ============================================================================

fn bench_expr_to_polish_simple(c: &mut Criterion) {
    let expr = Expr::add(Expr::int(2), Expr::int(3));

    c.bench_function("expr_to_polish_simple", |b| {
        b.iter(|| expr_to_polish(black_box(&expr)))
    });
}

fn bench_expr_to_polish_nested(c: &mut Criterion) {
    let expr = Expr::mul(
        Expr::add(Expr::int(2), Expr::int(3)),
        Expr::sub(Expr::int(10), Expr::int(4)),
    );

    c.bench_function("expr_to_polish_nested", |b| {
        b.iter(|| expr_to_polish(black_box(&expr)))
    });
}

fn bench_expr_to_polish_function(c: &mut Criterion) {
    let expr = Expr::func(MathFn::Sin, vec![Expr::float(0.5)]);

    c.bench_function("expr_to_polish_function", |b| {
        b.iter(|| expr_to_polish(black_box(&expr)))
    });
}

fn bench_expr_to_polish_complex(c: &mut Criterion) {
    // (x + 2) * (y - 3) + sin(z)
    let expr = Expr::add(
        Expr::mul(
            Expr::add(Expr::symbol("x"), Expr::int(2)),
            Expr::sub(Expr::symbol("y"), Expr::int(3)),
        ),
        Expr::func(MathFn::Sin, vec![Expr::symbol("z")]),
    );

    c.bench_function("expr_to_polish_complex", |b| {
        b.iter(|| expr_to_polish(black_box(&expr)))
    });
}

// ============================================================================
// Round-trip Benchmarks
// ============================================================================

fn bench_roundtrip(c: &mut Criterion) {
    let mut parser = PolishParser::new();
    let input = "(* (+ 2 3) (- 10 4))";

    c.bench_function("roundtrip_parse_and_convert", |b| {
        b.iter(|| {
            let expr = parser.parse(black_box(input)).unwrap();
            expr_to_polish(&expr)
        })
    });
}

// ============================================================================
// Scaling Benchmarks
// ============================================================================

fn bench_parse_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_scaling");

    // Generate expressions of increasing depth
    let depths = [2, 4, 6, 8, 10];
    for depth in depths.iter() {
        // Build nested expression: (+ (+ (+ ... 1 2) 3) 4)
        let mut expr_str = "1".to_string();
        for i in 2..=*depth {
            expr_str = format!("(+ {} {})", expr_str, i);
        }

        group.bench_with_input(BenchmarkId::new("depth", depth), &expr_str, |b, input| {
            let mut parser = PolishParser::new();
            b.iter(|| parser.parse(black_box(input.as_str())))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    // Parsing
    bench_parse_simple,
    bench_parse_nested,
    bench_parse_deeply_nested,
    bench_parse_function_simple,
    bench_parse_function_nested,
    bench_parse_compound_expression,
    bench_parse_polynomial,
    bench_parse_with_symbols,
    bench_parse_floats,
    bench_parse_negative,
    // Conversion
    bench_expr_to_polish_simple,
    bench_expr_to_polish_nested,
    bench_expr_to_polish_function,
    bench_expr_to_polish_complex,
    // Round-trip
    bench_roundtrip,
    // Scaling
    bench_parse_scaling,
);
criterion_main!(benches);
