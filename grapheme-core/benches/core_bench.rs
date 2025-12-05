//! Benchmarks for grapheme-core

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_core::GraphemeGraph;

fn bench_text_to_graph_short(c: &mut Criterion) {
    c.bench_function("text_to_graph_short", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box("Hello, World!")))
    });
}

fn bench_text_to_graph_long(c: &mut Criterion) {
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(10);

    c.bench_function("text_to_graph_long", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(&text)))
    });
}

fn bench_unicode_text(c: &mut Criterion) {
    let text = "Hello你好مرحبا🚀∫x²dx";

    c.bench_function("text_to_graph_unicode", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

criterion_group!(
    benches,
    bench_text_to_graph_short,
    bench_text_to_graph_long,
    bench_unicode_text
);
criterion_main!(benches);
