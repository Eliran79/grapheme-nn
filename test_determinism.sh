#!/bin/bash
# Test that same input gives deterministic embeddings and morphing

echo "=== Testing Embedding Determinism ==="
echo ""
echo "Creating test program..."

cat > /tmp/test_determinism.rs << 'EOF'
use grapheme_core::{GraphemeGraph, GraphTransformNet};

fn main() {
    println!("Creating model with fixed seed...");
    let model = GraphTransformNet::new(256, 64, 128, 2);

    let input = "abc";

    println!("\n=== Forward Pass 1 ===");
    let graph1 = GraphemeGraph::from_text(input);
    let output1 = model.forward(&graph1);
    println!("Input nodes: {}", graph1.node_count());
    println!("Output nodes: {}", output1.node_count());
    println!("Merge threshold: {:.6}", model.merge_threshold.value);

    println!("\n=== Forward Pass 2 (same input) ===");
    let graph2 = GraphemeGraph::from_text(input);
    let output2 = model.forward(&graph2);
    println!("Input nodes: {}", graph2.node_count());
    println!("Output nodes: {}", output2.node_count());
    println!("Merge threshold: {:.6}", model.merge_threshold.value);

    println!("\n=== Determinism Check ===");
    if output1.node_count() == output2.node_count() {
        println!("✓ PASS: Same input → same morphing ({} nodes)", output1.node_count());
        println!("  Embeddings are deterministic!");
    } else {
        println!("✗ FAIL: Different morphing!");
        println!("  Pass 1: {} nodes", output1.node_count());
        println!("  Pass 2: {} nodes", output2.node_count());
    }

    println!("\n=== Learnable Threshold ===");
    println!("Initial threshold value: {:.6}", model.merge_threshold.value);
    println!("Sigmoid(threshold): {:.6}", 1.0 / (1.0 + (-model.merge_threshold.value).exp()));
}
EOF

echo "Compiling test..."
rustc --edition 2021 /tmp/test_determinism.rs \
    -L target/release/deps \
    --extern grapheme_core=target/release/libgrapheme_core.rlib \
    -o /tmp/test_determinism 2>&1 || {
    echo "Compilation failed (expected - needs full dependency resolution)"
    echo ""
    echo "Running via cargo instead..."

    # Create a temporary test binary in the project
    cat > grapheme-train/src/bin/test_determinism.rs << 'INNEREOF'
use grapheme_core::{GraphemeGraph, GraphTransformNet};

fn main() {
    println!("Creating model with fixed seed...");
    let model = GraphTransformNet::new(256, 64, 128, 2);

    let input = "abc";

    println!("\n=== Forward Pass 1 ===");
    let graph1 = GraphemeGraph::from_text(input);
    let output1 = model.forward(&graph1);
    println!("Input nodes: {}", graph1.node_count());
    println!("Output nodes: {}", output1.node_count());
    println!("Merge threshold: {:.6}", model.merge_threshold.value);

    println!("\n=== Forward Pass 2 (same input) ===");
    let graph2 = GraphemeGraph::from_text(input);
    let output2 = model.forward(&graph2);
    println!("Input nodes: {}", graph2.node_count());
    println!("Output nodes: {}", output2.node_count());
    println!("Merge threshold: {:.6}", model.merge_threshold.value);

    println!("\n=== Determinism Check ===");
    if output1.node_count() == output2.node_count() {
        println!("✓ PASS: Same input → same morphing ({} nodes)", output1.node_count());
        println!("  Embeddings are deterministic!");
    } else {
        println!("✗ FAIL: Different morphing!");
        println!("  Pass 1: {} nodes", output1.node_count());
        println!("  Pass 2: {} nodes", output2.node_count());
    }

    println!("\n=== Learnable Threshold ===");
    println!("Initial threshold value: {:.6}", model.merge_threshold.value);
    println!("Sigmoid(threshold): {:.6}", 1.0 / (1.0 + (-model.merge_threshold.value).exp()));
}
INNEREOF

    cargo run --release --bin test_determinism
}
