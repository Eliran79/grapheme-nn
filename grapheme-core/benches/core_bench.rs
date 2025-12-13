//! Benchmarks for grapheme-core
//!
//! Comprehensive benchmarks covering:
//! - Text to graph conversion
//! - Unicode handling
//! - Text scaling performance
//! - Pattern extraction
//! - Graph operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grapheme_core::GraphemeGraph;

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
criterion_main!(benches);
