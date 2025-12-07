//! Benchmarks for grapheme-reason

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_reason::{
    Analogy, ComplexityBounds, Deduction, Graph, Induction, LogicRules, SimpleAnalogy,
    SimpleDeduction, SimpleInduction,
};

fn make_graph(text: &str) -> Graph {
    grapheme_core::DagNN::from_text(text).unwrap()
}

fn bench_deduction(c: &mut Criterion) {
    let deduction = SimpleDeduction::new();
    let bounds = ComplexityBounds::default();
    let rules = LogicRules::new();
    let premises = vec![make_graph("premise1"), make_graph("premise2")];

    c.bench_function("deduction_forward", |b| {
        b.iter(|| deduction.deduce(black_box(premises.clone()), &rules, &bounds))
    });
}

fn bench_induction(c: &mut Criterion) {
    let induction = SimpleInduction::new();
    let bounds = ComplexityBounds::default();
    let examples = vec![
        (make_graph("in1"), make_graph("out1")),
        (make_graph("in2"), make_graph("out2")),
        (make_graph("in3"), make_graph("out3")),
    ];

    c.bench_function("induction_induce", |b| {
        b.iter(|| induction.induce(black_box(examples.clone()), &bounds))
    });
}

fn bench_analogy(c: &mut Criterion) {
    let analogy = SimpleAnalogy::new();
    let bounds = ComplexityBounds::default();
    let source = make_graph("The quick brown fox");
    let target = make_graph("A lazy sleeping dog");

    c.bench_function("analogy_mapping", |b| {
        b.iter(|| analogy.analogize(black_box(&source), black_box(&target), &bounds))
    });
}

criterion_group!(benches, bench_deduction, bench_induction, bench_analogy,);

criterion_main!(benches);
