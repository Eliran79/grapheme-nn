//! Benchmarks for grapheme-parallel

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_parallel::{
    make_parallel, BatchProcessor, Graph, GraphBatchProcessor, ParallelGraph, ParallelGraphExt,
    ShardedGraph,
};

fn make_graph(text: &str) -> Graph {
    grapheme_core::DagNN::from_text(text).unwrap()
}

fn make_large_graph() -> Graph {
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    make_graph(&text)
}

fn bench_parallel_map(c: &mut Criterion) {
    let graph = make_large_graph();
    let parallel = make_parallel(graph);

    c.bench_function("parallel_map_nodes", |b| {
        b.iter(|| parallel.par_map(|i| black_box(i * 2)))
    });
}

fn bench_parallel_filter(c: &mut Criterion) {
    let graph = make_large_graph();
    let parallel = make_parallel(graph);

    c.bench_function("parallel_filter_nodes", |b| {
        b.iter(|| parallel.par_filter(|i| black_box(i % 2 == 0)))
    });
}

fn bench_parallel_any(c: &mut Criterion) {
    let graph = make_large_graph();
    let parallel = make_parallel(graph);

    c.bench_function("parallel_any", |b| {
        b.iter(|| parallel.par_any(|i| black_box(i > 100)))
    });
}

fn bench_batch_processing(c: &mut Criterion) {
    let graphs: Vec<Graph> = (0..100)
        .map(|i| make_graph(&format!("graph number {}", i)))
        .collect();

    let processor = GraphBatchProcessor::new(|g: &Graph| g.node_count());

    c.bench_function("batch_process_100", |b| {
        b.iter(|| processor.process_batch(black_box(&graphs)))
    });
}

fn bench_sharded_parallel(c: &mut Criterion) {
    let graphs: Vec<Graph> = (0..8).map(|_| make_large_graph()).collect();
    let sharded = ShardedGraph::from_shards(graphs);

    c.bench_function("sharded_par_map", |b| {
        b.iter(|| sharded.par_map_shards(|g| black_box(g.node_count())))
    });
}

criterion_group!(
    benches,
    bench_parallel_map,
    bench_parallel_filter,
    bench_parallel_any,
    bench_batch_processing,
    bench_sharded_parallel
);

criterion_main!(benches);
