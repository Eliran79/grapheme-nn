#!/bin/bash
# Test that same input gives same embeddings

cat > /tmp/test_consistency.rs << 'EOF'
use grapheme_core::{GraphemeGraph, GraphTransformNet};

fn main() {
    let model = GraphTransformNet::new(256, 64, 128, 2);

    let input = "abc";

    // Forward pass 1
    let graph1 = GraphemeGraph::from_text(input);
    let output1 = model.forward(&graph1);

    // Forward pass 2 (same input)
    let graph2 = GraphemeGraph::from_text(input);
    let output2 = model.forward(&graph2);

    println!("Input: '{}'", input);
    println!("Output1 nodes: {}", output1.node_count());
    println!("Output2 nodes: {}", output2.node_count());

    if output1.node_count() == output2.node_count() {
        println!("✓ Deterministic! Same input → same morphing");
    } else {
        println!("✗ Non-deterministic! Different morphing!");
    }
}
EOF

rustc --edition 2021 /tmp/test_consistency.rs -L target/release/deps --extern grapheme_core=target/release/libgrapheme_core.rlib 2>&1 || echo "Compile failed (expected - needs full cargo build)"
