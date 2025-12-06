//! Benchmarks for grapheme-law

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_core::DomainBrain;
use grapheme_law::{LawBrain, LegalGraph};

fn bench_parse_citation(c: &mut Criterion) {
    c.bench_function("law_parse_citation", |b| {
        b.iter(|| LegalGraph::parse_citation(black_box("Brown v. Board (1954)")))
    });
}

fn bench_can_process(c: &mut Criterion) {
    let brain = LawBrain::new();
    let samples = [
        "plaintiff v. defendant",
        "hello world",
        "pursuant to statute",
        "simple text",
    ];

    c.bench_function("law_can_process", |b| {
        b.iter(|| {
            for sample in &samples {
                black_box(brain.can_process(sample));
            }
        })
    });
}

criterion_group!(benches, bench_parse_citation, bench_can_process,);

criterion_main!(benches);
