//! Benchmarks for grapheme-core
//!
//! Comprehensive benchmarks covering:
//! - Text to graph conversion
//! - Unicode handling
//! - Text scaling performance
//! - Pattern extraction
//! - Graph operations
//! - Complexity verification (backend-112)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grapheme_core::{
    ActivationFn, DagNN, Edge, EdgeType, Embedding, GraphemeGraph, HebbianConfig,
    HebbianLearning, HybridLearningConfig, InitStrategy,
};
use ndarray::Array1;
use std::collections::HashMap;

// ============================================================================
// Basic Text to Graph Benchmarks
// ============================================================================

fn bench_text_to_graph_short(c: &mut Criterion) {
    c.bench_function("text_to_graph_short", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box("Hello, World!")))
    });
}

fn bench_text_to_graph_medium(c: &mut Criterion) {
    let text = "The quick brown fox jumps over the lazy dog.";

    c.bench_function("text_to_graph_medium", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

fn bench_text_to_graph_long(c: &mut Criterion) {
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(10);

    c.bench_function("text_to_graph_long", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(&text)))
    });
}

// ============================================================================
// Unicode Benchmarks
// ============================================================================

fn bench_unicode_text(c: &mut Criterion) {
    let text = "Hello你好مرحبا🚀∫x²dx";

    c.bench_function("text_to_graph_unicode", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

fn bench_unicode_cjk(c: &mut Criterion) {
    let text = "日本語中文한국어";

    c.bench_function("text_to_graph_cjk", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

fn bench_unicode_arabic(c: &mut Criterion) {
    let text = "مرحبا بالعالم";

    c.bench_function("text_to_graph_arabic", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

fn bench_unicode_emoji(c: &mut Criterion) {
    let text = "🚀🌟💻🎉🔥🌈⭐️🎯";

    c.bench_function("text_to_graph_emoji", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

fn bench_unicode_math_symbols(c: &mut Criterion) {
    let text = "∫∑∏∂∇∆∞≈≠≤≥±×÷√π";

    c.bench_function("text_to_graph_math_symbols", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

fn bench_unicode_mixed(c: &mut Criterion) {
    let text = "Hello 你好 مرحبا 🚀 ∫x²dx π≈3.14 日本語";

    c.bench_function("text_to_graph_mixed_unicode", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

// ============================================================================
// Mathematical Text Benchmarks
// ============================================================================

fn bench_math_expression_text(c: &mut Criterion) {
    let text = "(+ (* 2 3) (- 10 4))";

    c.bench_function("text_to_graph_math_expr", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

fn bench_math_equation_text(c: &mut Criterion) {
    let text = "f(x) = x² + 2x + 1";

    c.bench_function("text_to_graph_equation", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

fn bench_calculus_text(c: &mut Criterion) {
    let text = "∫₀¹ x² dx = 1/3";

    c.bench_function("text_to_graph_calculus", |b| {
        b.iter(|| GraphemeGraph::from_text(black_box(text)))
    });
}

// ============================================================================
// Graph Operation Benchmarks
// ============================================================================

fn bench_graph_node_count(c: &mut Criterion) {
    let text = "The quick brown fox jumps over the lazy dog.";
    let graph = GraphemeGraph::from_text(text);

    c.bench_function("graph_node_count", |b| {
        b.iter(|| black_box(&graph).node_count())
    });
}

fn bench_graph_edge_count(c: &mut Criterion) {
    let text = "The quick brown fox jumps over the lazy dog.";
    let graph = GraphemeGraph::from_text(text);

    c.bench_function("graph_edge_count", |b| {
        b.iter(|| black_box(&graph).edge_count())
    });
}

// ============================================================================
// Scaling Benchmarks
// ============================================================================

fn bench_text_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_scaling");

    let base = "The quick brown fox. ";
    for repeat in [1, 5, 10, 20, 50].iter() {
        let text = base.repeat(*repeat);
        group.bench_with_input(BenchmarkId::new("repeats", repeat), &text, |b, text| {
            b.iter(|| GraphemeGraph::from_text(black_box(text.as_str())))
        });
    }

    group.finish();
}

fn bench_unicode_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("unicode_scaling");

    let base = "Hello 你好 🚀 ";
    for repeat in [1, 5, 10, 20, 50].iter() {
        let text = base.repeat(*repeat);
        group.bench_with_input(BenchmarkId::new("repeats", repeat), &text, |b, text| {
            b.iter(|| GraphemeGraph::from_text(black_box(text.as_str())))
        });
    }

    group.finish();
}

fn bench_word_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("word_count_scaling");

    let words = [
        "one",
        "one two",
        "one two three",
        "one two three four",
        "one two three four five",
    ];

    for (i, text) in words.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("words", i + 1), text, |b, text| {
            b.iter(|| GraphemeGraph::from_text(black_box(*text)))
        });
    }

    group.finish();
}

// ============================================================================
// Complexity Verification Benchmarks (Backend-112)
// ============================================================================
// These benchmarks verify O(V+E), O(E), or O(V) complexity claims.
// We test with different graph sizes and verify linear scaling.

/// Helper to create a DAG with specified number of nodes
fn create_dag_with_nodes(n: usize) -> DagNN {
    // Create a linear chain of n characters
    let text: String = (0..n).map(|i| ((i % 26) as u8 + b'a') as char).collect();
    DagNN::from_text(&text).unwrap()
}

/// Helper to create a DAG with additional hidden nodes and edges
fn create_dag_with_hidden(n: usize, hidden_ratio: f32) -> DagNN {
    let mut dag = create_dag_with_nodes(n);
    let num_hidden = (n as f32 * hidden_ratio) as usize;

    let inputs = dag.input_nodes().to_vec();

    // Add hidden nodes with connections
    for i in 0..num_hidden {
        let hidden = dag.add_hidden_with_activation(ActivationFn::ReLU);
        // Connect to random input
        let src_idx = i % inputs.len();
        dag.add_edge(inputs[src_idx], hidden, Edge::new(0.5, EdgeType::Sequential));
    }

    let _ = dag.update_topology();
    dag
}

fn bench_forward_pass_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_forward_pass");
    group.sample_size(50);

    // Test with increasing graph sizes: 10, 50, 100, 500, 1000
    for size in [10, 50, 100, 500, 1000].iter() {
        let mut dag = create_dag_with_hidden(*size, 0.5);

        group.bench_with_input(BenchmarkId::new("nodes", size), size, |b, _| {
            b.iter(|| {
                dag.neuromorphic_forward().unwrap();
            })
        });
    }

    group.finish();
}

fn bench_edge_pruning_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_edge_pruning");
    group.sample_size(50);

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("nodes", size), size, |b, _| {
            b.iter_batched(
                || create_dag_with_hidden(*size, 0.5),
                |mut dag| {
                    dag.prune_edges_by_threshold(black_box(0.1));
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_orphan_removal_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_orphan_removal");
    group.sample_size(50);

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("nodes", size), size, |b, _| {
            b.iter_batched(
                || {
                    let mut dag = create_dag_with_hidden(*size, 0.5);
                    // Add some orphan nodes
                    for _ in 0..(*size / 10) {
                        dag.add_hidden();
                    }
                    dag
                },
                |mut dag| {
                    dag.remove_orphaned_nodes();
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_neurogenesis_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_neurogenesis");
    group.sample_size(30);

    for size in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::new("nodes", size), size, |b, _| {
            b.iter_batched(
                || {
                    let dag = create_dag_with_hidden(*size, 0.3);
                    // Create edge gradients for neurogenesis
                    let mut edge_grads = HashMap::new();
                    for edge_idx in dag.graph.edge_indices() {
                        if let Some((src, tgt)) = dag.graph.edge_endpoints(edge_idx) {
                            edge_grads.insert((src, tgt), 0.5);
                        }
                    }
                    (dag, edge_grads)
                },
                |(mut dag, edge_grads)| {
                    dag.neurogenesis_from_gradient(&edge_grads, 0.3, 5);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_hebbian_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_hebbian");
    group.sample_size(50);

    let config = HebbianConfig::new(0.01);

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("nodes", size), size, |b, _| {
            b.iter_batched(
                || {
                    let mut dag = create_dag_with_hidden(*size, 0.5);
                    dag.neuromorphic_forward().unwrap();
                    dag
                },
                |mut dag| {
                    dag.backward_hebbian(&config);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_hybrid_backward_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_hybrid_backward");
    group.sample_size(30);

    let config = HybridLearningConfig::default();

    for size in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::new("nodes", size), size, |b, _| {
            b.iter_batched(
                || {
                    let mut dag = create_dag_with_hidden(*size, 0.3);
                    dag.neuromorphic_forward().unwrap();
                    let mut embedding = Embedding::new(256, 8, InitStrategy::Zero);

                    // Create output gradient
                    let mut output_grad = HashMap::new();
                    if let Some(&node) = dag.input_nodes().last() {
                        output_grad.insert(node, Array1::from_vec(vec![0.1]));
                    }

                    (dag, embedding, output_grad)
                },
                |(mut dag, mut embedding, output_grad)| {
                    dag.backward_hybrid(&output_grad, &mut embedding, &config);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_topological_sort_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_topological_sort");
    group.sample_size(50);

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("nodes", size), size, |b, _| {
            b.iter_batched(
                || create_dag_with_hidden(*size, 0.5),
                |mut dag| {
                    dag.update_topology().unwrap();
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_cleanup_disconnected_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_cleanup_disconnected");
    group.sample_size(50);

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("nodes", size), size, |b, _| {
            b.iter_batched(
                || {
                    let mut dag = create_dag_with_hidden(*size, 0.5);
                    // Add disconnected subgraphs
                    for _ in 0..(*size / 5) {
                        let h1 = dag.add_hidden();
                        let h2 = dag.add_hidden();
                        dag.add_edge(h1, h2, Edge::sequential());
                    }
                    // Set first input as output for cleanup test
                    if !dag.input_nodes().is_empty() {
                        let out = dag.input_nodes()[0];
                        dag.set_output_nodes(vec![out]);
                    }
                    dag
                },
                |mut dag| {
                    dag.cleanup_disconnected();
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    // Basic text to graph
    bench_text_to_graph_short,
    bench_text_to_graph_medium,
    bench_text_to_graph_long,
    // Unicode
    bench_unicode_text,
    bench_unicode_cjk,
    bench_unicode_arabic,
    bench_unicode_emoji,
    bench_unicode_math_symbols,
    bench_unicode_mixed,
    // Mathematical text
    bench_math_expression_text,
    bench_math_equation_text,
    bench_calculus_text,
    // Graph operations
    bench_graph_node_count,
    bench_graph_edge_count,
    // Scaling
    bench_text_scaling,
    bench_unicode_scaling,
    bench_word_count_scaling,
);

// Separate group for complexity verification (backend-112)
criterion_group!(
    complexity_benches,
    bench_forward_pass_complexity,
    bench_edge_pruning_complexity,
    bench_orphan_removal_complexity,
    bench_neurogenesis_complexity,
    bench_hebbian_complexity,
    bench_hybrid_backward_complexity,
    bench_topological_sort_complexity,
    bench_cleanup_disconnected_complexity,
);

criterion_main!(benches, complexity_benches);
