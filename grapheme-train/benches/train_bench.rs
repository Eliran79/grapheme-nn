//! Benchmarks for grapheme-train
//!
//! Comprehensive benchmarks covering:
//! - Data generation for all curriculum levels
//! - Dataset operations (creation, splitting, batching)
//! - Graph edit distance computation
//! - Validation utilities

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grapheme_engine::{Expr, MathOp, Value};
use grapheme_math::MathGraph;
use grapheme_train::{CurriculumLevel, DataGenerator, Dataset, GraphEditDistance, LevelSpec};

// ============================================================================
// Data Generation Benchmarks
// ============================================================================

fn bench_generate_level1(c: &mut Criterion) {
    c.bench_function("generate_level1_100", |b| {
        b.iter(|| {
            let mut gen = DataGenerator::new(42);
            gen.generate_level(black_box(CurriculumLevel::BasicArithmetic), 100)
        })
    });
}

fn bench_generate_level2(c: &mut Criterion) {
    c.bench_function("generate_level2_100", |b| {
        b.iter(|| {
            let mut gen = DataGenerator::new(42);
            gen.generate_level(black_box(CurriculumLevel::NestedOperations), 100)
        })
    });
}

fn bench_generate_level3(c: &mut Criterion) {
    c.bench_function("generate_level3_symbols_50", |b| {
        b.iter(|| {
            let mut gen = DataGenerator::new(42);
            gen.generate_level(black_box(CurriculumLevel::SymbolSubstitution), 50)
        })
    });
}

fn bench_generate_level5(c: &mut Criterion) {
    c.bench_function("generate_level5_differentiation_50", |b| {
        b.iter(|| {
            let mut gen = DataGenerator::new(42);
            gen.generate_level(black_box(CurriculumLevel::Differentiation), 50)
        })
    });
}

fn bench_generate_curriculum(c: &mut Criterion) {
    c.bench_function("generate_curriculum_50_per_level", |b| {
        b.iter(|| {
            let mut gen = DataGenerator::new(42);
            gen.generate_curriculum(black_box(50))
        })
    });
}

fn bench_generate_from_spec(c: &mut Criterion) {
    let spec = LevelSpec::level_2();

    c.bench_function("generate_from_spec_level2_100", |b| {
        b.iter(|| {
            let mut gen = DataGenerator::new(42);
            gen.generate_from_spec(black_box(&spec))
        })
    });
}

// ============================================================================
// Dataset Operation Benchmarks
// ============================================================================

fn bench_dataset_creation(c: &mut Criterion) {
    let mut gen = DataGenerator::new(42);
    let examples = gen.generate_level(CurriculumLevel::BasicArithmetic, 1000);

    c.bench_function("dataset_from_1000_examples", |b| {
        b.iter(|| Dataset::from_examples("bench", black_box(examples.clone())))
    });
}

fn bench_dataset_split(c: &mut Criterion) {
    let mut gen = DataGenerator::new(42);
    let examples = gen.generate_level(CurriculumLevel::BasicArithmetic, 1000);
    let dataset = Dataset::from_examples("bench", examples);

    c.bench_function("dataset_split_1000", |b| {
        b.iter(|| black_box(&dataset).split(0.8, 0.1))
    });
}

fn bench_batch_iterator(c: &mut Criterion) {
    let mut gen = DataGenerator::new(42);
    let examples = gen.generate_level(CurriculumLevel::BasicArithmetic, 1000);
    let dataset = Dataset::from_examples("bench", examples);

    c.bench_function("batch_iterate_1000_batch32", |b| {
        b.iter(|| {
            let batches = dataset.batches(32);
            for batch in batches {
                black_box(batch);
            }
        })
    });
}

fn bench_filter_by_level(c: &mut Criterion) {
    let mut gen = DataGenerator::new(42);
    let examples = gen.generate_curriculum(100);
    let dataset = Dataset::from_examples("bench", examples);

    c.bench_function("filter_by_level_2", |b| {
        b.iter(|| black_box(&dataset).filter_by_level(2))
    });
}

// ============================================================================
// Graph Edit Distance Benchmarks
// ============================================================================

fn bench_ged_simple(c: &mut Criterion) {
    // Create two simple expression graphs
    let expr1 = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Integer(2))),
        right: Box::new(Expr::Value(Value::Integer(3))),
    };
    let expr2 = Expr::BinOp {
        op: MathOp::Add,
        left: Box::new(Expr::Value(Value::Integer(2))),
        right: Box::new(Expr::Value(Value::Integer(4))),
    };

    let graph1 = MathGraph::from_expr(&expr1);
    let graph2 = MathGraph::from_expr(&expr2);

    c.bench_function("ged_simple_graphs", |b| {
        b.iter(|| GraphEditDistance::compute_math(black_box(&graph1), black_box(&graph2)))
    });
}

fn bench_ged_nested(c: &mut Criterion) {
    // Create nested expression graphs
    let expr1 = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        }),
        right: Box::new(Expr::BinOp {
            op: MathOp::Sub,
            left: Box::new(Expr::Value(Value::Integer(10))),
            right: Box::new(Expr::Value(Value::Integer(4))),
        }),
    };
    let expr2 = Expr::BinOp {
        op: MathOp::Mul,
        left: Box::new(Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(5))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        }),
        right: Box::new(Expr::Value(Value::Integer(6))),
    };

    let graph1 = MathGraph::from_expr(&expr1);
    let graph2 = MathGraph::from_expr(&expr2);

    c.bench_function("ged_nested_graphs", |b| {
        b.iter(|| GraphEditDistance::compute_math(black_box(&graph1), black_box(&graph2)))
    });
}

// ============================================================================
// Scaling Benchmarks
// ============================================================================

fn bench_generation_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("generation_scaling");

    for size in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::new("level1", size), size, |b, &size| {
            b.iter(|| {
                let mut gen = DataGenerator::new(42);
                gen.generate_level(CurriculumLevel::BasicArithmetic, size)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    // Data generation
    bench_generate_level1,
    bench_generate_level2,
    bench_generate_level3,
    bench_generate_level5,
    bench_generate_curriculum,
    bench_generate_from_spec,
    // Dataset operations
    bench_dataset_creation,
    bench_dataset_split,
    bench_batch_iterator,
    bench_filter_by_level,
    // Graph edit distance
    bench_ged_simple,
    bench_ged_nested,
    // Scaling
    bench_generation_scaling,
);
criterion_main!(benches);
