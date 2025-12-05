//! Benchmarks for grapheme-multimodal

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_multimodal::{
    create_default_multimodal, Graph, ModalGraph, Modality, MultiModalGraph,
};

fn make_graph(text: &str) -> Graph {
    grapheme_core::DagNN::from_text(text).unwrap()
}

fn bench_fuse(c: &mut Criterion) {
    let visual = ModalGraph::new(make_graph("visual content"), Modality::Visual);
    let linguistic = ModalGraph::new(make_graph("text content"), Modality::Linguistic);

    c.bench_function("multimodal_fuse", |b| {
        b.iter(|| {
            let mut mm = create_default_multimodal();
            let v = ModalGraph::new(make_graph("visual"), Modality::Visual);
            let l = ModalGraph::new(make_graph("text"), Modality::Linguistic);
            mm.fuse(Some(v), None, Some(l), None)
        })
    });
}

fn bench_translate(c: &mut Criterion) {
    let mm = create_default_multimodal();
    let source = ModalGraph::new(make_graph("source content"), Modality::Linguistic);

    c.bench_function("multimodal_translate", |b| {
        b.iter(|| mm.translate_modality(black_box(&source), Modality::Visual))
    });
}

criterion_group!(benches, bench_fuse, bench_translate);

criterion_main!(benches);
