//! Benchmarks for grapheme-ground

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grapheme_ground::{
    create_default_embodied_agent, create_default_grounded_graph, create_visual_sensor,
    EmbodiedAgent, ExternalRef, Graph, GroundedGraph, Referent, Sensor, SimpleActuator,
    SimpleSensor, WorldInterface,
};
use grapheme_multimodal::Modality;

fn make_graph(text: &str) -> Graph {
    grapheme_core::DagNN::from_text(text).unwrap()
}

fn bench_grounding(c: &mut Criterion) {
    c.bench_function("ground_bind_referent", |b| {
        b.iter(|| {
            let mut gg = create_default_grounded_graph();
            for i in 0..10 {
                let referent = Referent::External(ExternalRef::new("test", &format!("id_{}", i)));
                gg.bind_referent(black_box(i as u64), referent);
            }
            gg
        })
    });
}

fn bench_perceive(c: &mut Criterion) {
    c.bench_function("ground_sensor_perceive", |b| {
        let mut sensor = create_visual_sensor();
        b.iter(|| sensor.perceive())
    });
}

fn bench_sense_think_act(c: &mut Criterion) {
    c.bench_function("ground_sense_think_act", |b| {
        let mut agent = create_default_embodied_agent();
        let mut world = WorldInterface::new();
        world.add_sensor(Box::new(SimpleSensor::new(Modality::Visual)));
        world.add_actuator(Box::new(SimpleActuator::new("motor")));

        b.iter(|| agent.sense_think_act(black_box(&mut world)))
    });
}

fn bench_simulate_consequence(c: &mut Criterion) {
    let gg = create_default_grounded_graph();
    let action = make_graph("pick up object");

    c.bench_function("ground_simulate_consequence", |b| {
        b.iter(|| gg.simulate_consequence(black_box(&action)))
    });
}

criterion_group!(
    benches,
    bench_grounding,
    bench_perceive,
    bench_sense_think_act,
    bench_simulate_consequence
);

criterion_main!(benches);
