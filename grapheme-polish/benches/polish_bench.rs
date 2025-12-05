//! Benchmarks for grapheme-polish

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_polish::PolishParser;

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

criterion_group!(benches, bench_parse_simple, bench_parse_nested);
criterion_main!(benches);
