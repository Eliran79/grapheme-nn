//! Benchmarks for grapheme-music

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_core::DomainBrain;
use grapheme_music::{MusicBrain, MusicGraph};

fn bench_parse_note(c: &mut Criterion) {
    c.bench_function("music_parse_note", |b| {
        b.iter(|| MusicGraph::parse_note(black_box("C4")))
    });
}

fn bench_can_process(c: &mut Criterion) {
    let brain = MusicBrain::new();
    let samples = ["C major chord", "hello world", "tempo 120 bpm"];

    c.bench_function("music_can_process", |b| {
        b.iter(|| {
            for sample in &samples {
                black_box(brain.can_process(sample));
            }
        })
    });
}

criterion_group!(benches, bench_parse_note, bench_can_process);
criterion_main!(benches);
