//! Benchmarks for grapheme-code

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_code::{CodeBrain, CodeGraph};
use grapheme_core::DomainBrain;

fn bench_parse_literal(c: &mut Criterion) {
    c.bench_function("code_parse_literal", |b| {
        b.iter(|| CodeGraph::from_simple_expr(black_box("42")))
    });
}

fn bench_parse_binary_op(c: &mut Criterion) {
    c.bench_function("code_parse_binary_op", |b| {
        b.iter(|| CodeGraph::from_simple_expr(black_box("1 + 2")))
    });
}

fn bench_detect_language(c: &mut Criterion) {
    let brain = CodeBrain::new();
    let code_samples = [
        "fn main() -> i32 { 0 }",
        "def foo(): pass",
        "function bar() {}",
    ];

    c.bench_function("code_detect_language", |b| {
        b.iter(|| {
            for code in &code_samples {
                black_box(brain.detect_language(code));
            }
        })
    });
}

fn bench_can_process(c: &mut Criterion) {
    let brain = CodeBrain::new();
    let samples = [
        "fn main() {}",
        "let x = 5",
        "hello world",
        "if condition { }",
    ];

    c.bench_function("code_can_process", |b| {
        b.iter(|| {
            for sample in &samples {
                black_box(brain.can_process(sample));
            }
        })
    });
}

criterion_group!(
    benches,
    bench_parse_literal,
    bench_parse_binary_op,
    bench_detect_language,
    bench_can_process,
);

criterion_main!(benches);
