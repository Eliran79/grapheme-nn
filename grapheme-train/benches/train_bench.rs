//! Benchmarks for grapheme-train

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_train::{CurriculumLevel, DataGenerator};

fn bench_generate_level1(c: &mut Criterion) {
    let generator = DataGenerator::new(42);

    c.bench_function("generate_level1", |b| {
        b.iter(|| generator.generate_level(black_box(CurriculumLevel::BasicArithmetic), 100))
    });
}

fn bench_generate_curriculum(c: &mut Criterion) {
    let generator = DataGenerator::new(42);

    c.bench_function("generate_curriculum", |b| {
        b.iter(|| generator.generate_curriculum(black_box(50)))
    });
}

criterion_group!(benches, bench_generate_level1, bench_generate_curriculum);
criterion_main!(benches);
