//! Benchmarks for grapheme-memory

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_memory::{
    Episode, EpisodicMemory, Graph, GraphFingerprint, SimpleEpisodicMemory,
    SimpleSemanticGraph, SemanticGraph, SimpleWorkingMemory, WorkingMemory,
};

fn make_graph(text: &str) -> Graph {
    grapheme_core::DagNN::from_text(text).unwrap()
}

fn bench_fingerprint(c: &mut Criterion) {
    let graph = make_graph("The quick brown fox jumps over the lazy dog");

    c.bench_function("fingerprint_compute", |b| {
        b.iter(|| GraphFingerprint::from_graph(black_box(&graph)))
    });
}

fn bench_fingerprint_similarity(c: &mut Criterion) {
    let g1 = make_graph("Hello world");
    let g2 = make_graph("Hello there");

    let fp1 = GraphFingerprint::from_graph(&g1);
    let fp2 = GraphFingerprint::from_graph(&g2);

    c.bench_function("fingerprint_similarity", |b| {
        b.iter(|| fp1.similarity(black_box(&fp2)))
    });
}

fn bench_episodic_store(c: &mut Criterion) {
    c.bench_function("episodic_store", |b| {
        b.iter(|| {
            let mut memory = SimpleEpisodicMemory::new(None);
            for i in 0..100 {
                let episode = Episode::new(
                    0,
                    i * 100,
                    make_graph("context"),
                    make_graph(&format!("content{}", i)),
                );
                memory.store(episode);
            }
            black_box(memory.len())
        })
    });
}

fn bench_episodic_recall(c: &mut Criterion) {
    let mut memory = SimpleEpisodicMemory::new(None);
    for i in 0..1000 {
        let episode = Episode::new(
            0,
            i * 100,
            make_graph("context"),
            make_graph(&format!("content{}", i)),
        );
        memory.store(episode);
    }

    let query = make_graph("content500");

    c.bench_function("episodic_recall_1000", |b| {
        b.iter(|| memory.recall(black_box(&query), 10))
    });
}

fn bench_semantic_query(c: &mut Criterion) {
    let mut graph = SimpleSemanticGraph::new();
    for i in 0..1000 {
        graph.assert(make_graph(&format!("fact number {}", i)));
    }

    let query = make_graph("fact number 500");

    c.bench_function("semantic_query_1000", |b| {
        b.iter(|| graph.query(black_box(&query), 10))
    });
}

fn bench_working_memory(c: &mut Criterion) {
    c.bench_function("working_memory_churn", |b| {
        b.iter(|| {
            let mut wm = SimpleWorkingMemory::new(7);
            for i in 0..100 {
                wm.attend(make_graph(&format!("item{}", i)));
            }
            black_box(wm.len())
        })
    });
}

criterion_group!(
    benches,
    bench_fingerprint,
    bench_fingerprint_similarity,
    bench_episodic_store,
    bench_episodic_recall,
    bench_semantic_query,
    bench_working_memory,
);

criterion_main!(benches);
