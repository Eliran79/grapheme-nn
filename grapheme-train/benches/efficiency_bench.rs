//! Efficiency Benchmark: GRAPHEME vs Transformer
//!
//! Compares GRAPHEME's O(n) graph operations against transformer O(n²) attention.
//! Per testing-004: Benchmark GRAPHEME vs transformer efficiency.
//!
//! Run with: cargo bench --bench efficiency_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grapheme_core::DagNN;
use ndarray::{Array2, Axis};

// ============================================================================
// FLOP and Memory Counters
// ============================================================================

/// Count approximate FLOPs for GRAPHEME graph operations
/// GRAPHEME: O(n) where n = characters
pub fn count_grapheme_flops(n: usize) -> usize {
    // Per character: ~10 ops (node creation, edge creation, activation)
    // Forward pass: ~5 ops per edge
    // Total: ~15 ops per character (O(n))
    n * 15
}

/// Count approximate FLOPs for transformer self-attention
/// Transformer: O(n²) for attention matrix computation
pub fn count_transformer_flops(n: usize, d_model: usize) -> usize {
    // Q, K, V projections: 3 * n * d_model * d_model
    let projections = 3 * n * d_model * d_model;
    // Attention scores: n * n * d_model (Q @ K^T)
    let attention_scores = n * n * d_model;
    // Softmax: n * n (approximate)
    let softmax = n * n;
    // Weighted sum: n * n * d_model
    let weighted_sum = n * n * d_model;
    // Output projection: n * d_model * d_model
    let output_proj = n * d_model * d_model;

    projections + attention_scores + softmax + weighted_sum + output_proj
}

/// Estimate memory usage for GRAPHEME
/// GRAPHEME: O(n) memory (nodes + edges)
pub fn estimate_grapheme_memory(n: usize) -> usize {
    // Per node: ~32 bytes (node struct)
    // Per edge: ~24 bytes (edge struct + weight)
    // Edges ≈ nodes (sequential connections)
    n * 32 + n * 24
}

/// Estimate memory usage for transformer attention
/// Transformer: O(n²) memory for attention matrix
pub fn estimate_transformer_memory(n: usize, d_model: usize) -> usize {
    // Q, K, V: 3 * n * d_model * 4 bytes (f32)
    let qkv = 3 * n * d_model * 4;
    // Attention matrix: n * n * 4 bytes
    let attention = n * n * 4;
    // Output: n * d_model * 4 bytes
    let output = n * d_model * 4;

    qkv + attention + output
}

// ============================================================================
// Simple Transformer Attention Baseline
// ============================================================================

/// Simple scaled dot-product attention for baseline comparison
/// This is a minimal implementation to demonstrate O(n²) scaling
pub fn simple_attention(
    query: &Array2<f32>,  // [n, d]
    key: &Array2<f32>,    // [n, d]
    value: &Array2<f32>,  // [n, d]
) -> Array2<f32> {
    let d_k = key.shape()[1] as f32;

    // Q @ K^T -> [n, n]
    let scores = query.dot(&key.t());

    // Scale
    let scaled = scores / d_k.sqrt();

    // Softmax (simplified: exp / sum)
    let exp_scores = scaled.mapv(|x| x.exp());
    let sum = exp_scores.sum_axis(Axis(1)).insert_axis(Axis(1));
    let attention_weights = &exp_scores / &sum;

    // Attention @ V -> [n, d]
    attention_weights.dot(value)
}

/// Create random matrices for attention benchmark
fn create_attention_inputs(n: usize, d_model: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let q = Array2::from_shape_fn((n, d_model), |_| rng.gen_range(-1.0..1.0));
    let k = Array2::from_shape_fn((n, d_model), |_| rng.gen_range(-1.0..1.0));
    let v = Array2::from_shape_fn((n, d_model), |_| rng.gen_range(-1.0..1.0));

    (q, k, v)
}

// ============================================================================
// GRAPHEME Forward Pass
// ============================================================================

/// Run GRAPHEME forward pass on text
fn grapheme_forward(text: &str) -> usize {
    use grapheme_core::ForwardPass;

    let mut dag = DagNN::from_text(text).unwrap();
    let _ = dag.forward();
    dag.graph.node_count()
}

// ============================================================================
// Benchmarks
// ============================================================================

/// Benchmark GRAPHEME graph creation and forward pass
fn bench_grapheme_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("grapheme_scaling");
    group.sample_size(50);

    let base = "Hello world. ";

    // Test various input sizes
    for &multiplier in &[8, 80, 800] {
        let text = base.repeat(multiplier);
        let n = text.len();

        group.bench_with_input(
            BenchmarkId::new("chars", n),
            &text,
            |b, text| {
                b.iter(|| grapheme_forward(black_box(text)))
            },
        );
    }

    group.finish();
}

/// Benchmark transformer attention scaling
fn bench_transformer_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformer_scaling");
    group.sample_size(30);

    let d_model = 64; // Small model dimension for benchmarking

    // Test various sequence lengths
    for &n in &[100, 500, 1000] {
        let (q, k, v) = create_attention_inputs(n, d_model);

        group.bench_with_input(
            BenchmarkId::new("seq_len", n),
            &(q, k, v),
            |b, (q, k, v)| {
                b.iter(|| simple_attention(black_box(q), black_box(k), black_box(v)))
            },
        );
    }

    group.finish();
}

/// Compare GRAPHEME vs Transformer for same effective input size
fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("grapheme_vs_transformer");
    group.sample_size(30);

    let d_model = 64;

    // Test at different scales
    for &n in &[100, 500, 1000] {
        // GRAPHEME: process n characters
        let text = "a".repeat(n);
        group.bench_with_input(
            BenchmarkId::new("grapheme", n),
            &text,
            |b, text| {
                b.iter(|| grapheme_forward(black_box(text)))
            },
        );

        // Transformer: process n tokens with attention
        let (q, k, v) = create_attention_inputs(n, d_model);
        group.bench_with_input(
            BenchmarkId::new("transformer", n),
            &(q.clone(), k.clone(), v.clone()),
            |b, (q, k, v)| {
                b.iter(|| simple_attention(black_box(q), black_box(k), black_box(v)))
            },
        );
    }

    group.finish();
}

/// Measure FLOP counts for theoretical comparison
fn bench_flop_comparison(c: &mut Criterion) {
    c.bench_function("flop_report", |b| {
        b.iter(|| {
            let d_model = 64;

            // Report FLOPs at different scales
            for &n in &[100, 1000, 10000, 100000] {
                let grapheme_flops = count_grapheme_flops(n);
                let transformer_flops = count_transformer_flops(n, d_model);
                let ratio = transformer_flops as f64 / grapheme_flops as f64;

                black_box((n, grapheme_flops, transformer_flops, ratio));
            }
        })
    });
}

/// Memory comparison benchmark
fn bench_memory_comparison(c: &mut Criterion) {
    c.bench_function("memory_report", |b| {
        b.iter(|| {
            let d_model = 64;

            for &n in &[100, 1000, 10000, 100000] {
                let grapheme_mem = estimate_grapheme_memory(n);
                let transformer_mem = estimate_transformer_memory(n, d_model);
                let ratio = transformer_mem as f64 / grapheme_mem as f64;

                black_box((n, grapheme_mem, transformer_mem, ratio));
            }
        })
    });
}

/// Print efficiency report (run with --nocapture)
fn print_efficiency_report() {
    println!("\n=== GRAPHEME vs Transformer Efficiency Report ===\n");

    let d_model = 64;

    println!("| Input Size | GRAPHEME FLOPs | Transformer FLOPs | Ratio |");
    println!("|------------|----------------|-------------------|-------|");

    for &n in &[100, 1000, 10000, 100000] {
        let g_flops = count_grapheme_flops(n);
        let t_flops = count_transformer_flops(n, d_model);
        let ratio = t_flops as f64 / g_flops as f64;

        println!("| {:>10} | {:>14} | {:>17} | {:>5.1}x |", n, g_flops, t_flops, ratio);
    }

    println!("\n| Input Size | GRAPHEME Memory | Transformer Memory | Ratio |");
    println!("|------------|-----------------|--------------------| ------|");

    for &n in &[100, 1000, 10000, 100000] {
        let g_mem = estimate_grapheme_memory(n);
        let t_mem = estimate_transformer_memory(n, d_model);
        let ratio = t_mem as f64 / g_mem as f64;

        println!("| {:>10} | {:>15} | {:>18} | {:>5.1}x |", n, g_mem, t_mem, ratio);
    }

    println!("\nScaling Analysis:");
    println!("- GRAPHEME: O(n) FLOPs, O(n) memory");
    println!("- Transformer: O(n²) attention FLOPs, O(n²) attention memory");
    println!("- At 100K tokens: Transformer uses ~{:.0}x more FLOPs",
             count_transformer_flops(100000, d_model) as f64 / count_grapheme_flops(100000) as f64);
}

/// Benchmark that also prints report
fn bench_with_report(c: &mut Criterion) {
    // Print report once
    static PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !PRINTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        print_efficiency_report();
    }

    c.bench_function("efficiency_summary", |b| {
        b.iter(|| {
            // Quick summary computation
            let ratio_100k = count_transformer_flops(100000, 64) as f64
                           / count_grapheme_flops(100000) as f64;
            black_box(ratio_100k)
        })
    });
}

criterion_group!(
    benches,
    bench_grapheme_scaling,
    bench_transformer_scaling,
    bench_comparison,
    bench_flop_comparison,
    bench_memory_comparison,
    bench_with_report,
);
criterion_main!(benches);
