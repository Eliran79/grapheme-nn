//! Benchmarks for grapheme-chem

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_chem::{ChemBrain, MolecularGraph};
use grapheme_core::DomainBrain;

fn bench_parse_formula(c: &mut Criterion) {
    c.bench_function("chem_parse_formula", |b| {
        b.iter(|| MolecularGraph::from_formula(black_box("H2O")))
    });
}

fn bench_can_process(c: &mut Criterion) {
    let brain = ChemBrain::new();
    let samples = ["H2O molecule", "hello world", "chemical reaction"];

    c.bench_function("chem_can_process", |b| {
        b.iter(|| {
            for sample in &samples {
                black_box(brain.can_process(sample));
            }
        })
    });
}

criterion_group!(benches, bench_parse_formula, bench_can_process);
criterion_main!(benches);
