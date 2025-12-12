//! Benchmarks for grapheme-world

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_world::{
    create_default_world_model, Graph, WorldModeling,
};

fn make_graph(text: &str) -> Graph {
    grapheme_core::DagNN::from_text(text).unwrap()
}

fn bench_predict(c: &mut Criterion) {
    let model = create_default_world_model();
    let state = make_graph("test state");
    let action = make_graph("action");

    c.bench_function("world_predict_horizon_5", |b| {
        b.iter(|| model.predict(black_box(&state), black_box(&action), 5))
    });
}

fn bench_simulate(c: &mut Criterion) {
    let model = create_default_world_model();
    let initial = make_graph("initial state");
    let actions: Vec<_> = (0..10)
        .map(|i| make_graph(&format!("action_{}", i)))
        .collect();

    c.bench_function("world_simulate_10_steps", |b| {
        b.iter(|| model.simulate(black_box(&initial), black_box(&actions)))
    });
}

criterion_group!(
    benches,
    bench_predict,
    bench_simulate,
);

criterion_main!(benches);
