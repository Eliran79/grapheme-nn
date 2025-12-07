//! Benchmarks for grapheme-meta

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_meta::{create_default_metacognition, ComputeBudget, Graph, MetaCognition};

fn make_graph(text: &str) -> Graph {
    grapheme_core::DagNN::from_text(text).unwrap()
}

fn bench_uncertainty(c: &mut Criterion) {
    let meta = create_default_metacognition();
    let query = make_graph("test query for uncertainty estimation");

    c.bench_function("meta_estimate_uncertainty", |b| {
        b.iter(|| meta.estimate_uncertainty(black_box(&query)))
    });
}

fn bench_introspect(c: &mut Criterion) {
    let meta = create_default_metacognition();

    c.bench_function("meta_introspect", |b| b.iter(|| meta.introspect()));
}

fn bench_allocate(c: &mut Criterion) {
    let meta = create_default_metacognition();
    let task = make_graph("complex task requiring computation allocation");
    let budget = ComputeBudget::standard();

    c.bench_function("meta_allocate_computation", |b| {
        b.iter(|| meta.allocate_computation(black_box(&task), black_box(&budget)))
    });
}

criterion_group!(benches, bench_uncertainty, bench_introspect, bench_allocate);

criterion_main!(benches);
