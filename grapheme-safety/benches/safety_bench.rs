//! Benchmarks for safety validation

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_safety::{Action, ActionTarget, ActionType, SafetyGuard};

fn bench_safe_action_validation(c: &mut Criterion) {
    let guard = SafetyGuard::new();
    let action = Action::new(
        ActionType::Query,
        ActionTarget::Data,
        "Query weather information",
    );

    c.bench_function("validate_safe_action", |b| {
        b.iter(|| guard.validate(black_box(&action)))
    });
}

fn bench_harmful_action_detection(c: &mut Criterion) {
    let guard = SafetyGuard::new();
    let action = Action::new(ActionType::Physical, ActionTarget::Human, "Harmful action")
        .with_harm_estimate(0.9);

    c.bench_function("detect_harmful_action", |b| {
        b.iter(|| guard.validate(black_box(&action)))
    });
}

fn bench_multiple_actions(c: &mut Criterion) {
    let guard = SafetyGuard::new();
    let actions: Vec<Action> = (0..100)
        .map(|i| {
            Action::new(
                ActionType::Generate,
                ActionTarget::Human,
                format!("Generated response {}", i),
            )
        })
        .collect();

    c.bench_function("validate_100_actions", |b| {
        b.iter(|| {
            for action in &actions {
                guard.validate(black_box(action));
            }
        })
    });
}

criterion_group!(
    benches,
    bench_safe_action_validation,
    bench_harmful_action_detection,
    bench_multiple_actions,
);
criterion_main!(benches);
