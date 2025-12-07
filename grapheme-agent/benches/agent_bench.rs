//! Benchmarks for grapheme-agent

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_agent::{create_default_agent, Agency, Goal, GoalHierarchy, Graph, Plan};
use grapheme_meta::UncertaintyEstimate;

fn make_graph(text: &str) -> Graph {
    grapheme_core::DagNN::from_text(text).unwrap()
}

fn bench_formulate_goal(c: &mut Criterion) {
    let agent = create_default_agent();
    let situation = make_graph("current situation");

    c.bench_function("agent_formulate_goal", |b| {
        b.iter(|| agent.formulate_goal(black_box(&situation)))
    });
}

fn bench_explore_exploit(c: &mut Criterion) {
    let agent = create_default_agent();
    let uncertainty = UncertaintyEstimate::new(0.5, 0.3);

    c.bench_function("agent_explore_exploit", |b| {
        b.iter(|| agent.explore_or_exploit(black_box(&uncertainty)))
    });
}

fn bench_goal_hierarchy(c: &mut Criterion) {
    c.bench_function("agent_goal_hierarchy_add", |b| {
        b.iter(|| {
            let mut hierarchy = GoalHierarchy::new();
            for i in 0..10 {
                let goal = Goal::new(0, &format!("goal_{}", i), make_graph("goal"));
                hierarchy.add_root(goal);
            }
            hierarchy
        })
    });
}

fn bench_plan_advance(c: &mut Criterion) {
    c.bench_function("agent_plan_advance", |b| {
        b.iter(|| {
            let mut plan = Plan::new(1);
            for i in 0..10 {
                plan.add_action(
                    grapheme_agent::Action::new(i, make_graph("action")),
                    make_graph("state"),
                );
            }
            while !plan.is_complete() {
                plan.advance();
            }
            plan
        })
    });
}

criterion_group!(
    benches,
    bench_formulate_goal,
    bench_explore_exploit,
    bench_goal_hierarchy,
    bench_plan_advance
);

criterion_main!(benches);
