use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_time::{generate_sine_wave, create_training_pairs, TimeBrain, TimeSeriesConfig};

fn bench_to_graph(c: &mut Criterion) {
    let config = TimeSeriesConfig::default().with_window_size(10);
    let brain = TimeBrain::new(config);
    let window: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();

    c.bench_function("time_to_graph_w10", |b| {
        b.iter(|| brain.to_graph(black_box(&window)))
    });
}

fn bench_to_graph_large(c: &mut Criterion) {
    let config = TimeSeriesConfig::default().with_window_size(50);
    let brain = TimeBrain::new(config);
    let window: Vec<f32> = (0..50).map(|i| i as f32 * 0.02).collect();

    c.bench_function("time_to_graph_w50", |b| {
        b.iter(|| brain.to_graph(black_box(&window)))
    });
}

fn bench_predict(c: &mut Criterion) {
    let config = TimeSeriesConfig::default().with_window_size(10);
    let brain = TimeBrain::new(config);
    let window: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
    let dag = brain.to_graph(&window).unwrap();

    c.bench_function("time_predict", |b| {
        b.iter(|| {
            let mut dag_clone = dag.clone();
            brain.predict(black_box(&mut dag_clone))
        })
    });
}

fn bench_generate_sine(c: &mut Criterion) {
    c.bench_function("generate_sine_1000", |b| {
        b.iter(|| generate_sine_wave(black_box(1000), 0.1, 1.0, 0.0))
    });
}

fn bench_create_pairs(c: &mut Criterion) {
    let series = generate_sine_wave(1000, 0.1, 1.0, 0.0);

    c.bench_function("create_pairs_w10", |b| {
        b.iter(|| create_training_pairs(black_box(&series), 10))
    });
}

criterion_group!(
    benches,
    bench_to_graph,
    bench_to_graph_large,
    bench_predict,
    bench_generate_sine,
    bench_create_pairs
);
criterion_main!(benches);
