use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_vision::{extract_blobs, image_to_graph, FeatureConfig, RawImage, VisionBrain};

fn bench_blob_extraction(c: &mut Criterion) {
    // Create a test image with multiple blobs
    let mut pixels = vec![0.0f32; 784];
    for i in 0..5 {
        let start = i * 150;
        for j in 0..50 {
            if start + j < 784 {
                pixels[start + j] = 0.8;
            }
        }
    }
    let image = RawImage::from_mnist(&pixels).unwrap();
    let config = FeatureConfig::mnist();

    c.bench_function("extract_blobs_mnist", |b| {
        b.iter(|| extract_blobs(black_box(&image), black_box(&config)))
    });
}

fn bench_image_to_graph(c: &mut Criterion) {
    let mut pixels = vec![0.0f32; 784];
    for i in 100..300 {
        pixels[i] = 0.7;
    }
    let image = RawImage::from_mnist(&pixels).unwrap();
    let config = FeatureConfig::mnist();

    c.bench_function("image_to_graph_mnist", |b| {
        b.iter(|| image_to_graph(black_box(&image), black_box(&config)))
    });
}

fn bench_vision_brain(c: &mut Criterion) {
    let brain = VisionBrain::mnist();
    let mut pixels = vec![0.0f32; 784];
    for i in 100..300 {
        pixels[i] = 0.7;
    }

    c.bench_function("vision_brain_to_graph", |b| {
        b.iter(|| brain.mnist_to_graph(black_box(&pixels)))
    });
}

criterion_group!(
    benches,
    bench_blob_extraction,
    bench_image_to_graph,
    bench_vision_brain
);
criterion_main!(benches);
