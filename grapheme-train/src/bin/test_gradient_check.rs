//! Finite difference gradient checking for structural loss
//!
//! Verifies analytical gradients match numerical approximation:
//! ∂L/∂θ ≈ (L(θ+ε) - L(θ-ε)) / 2ε
//!
//! Uses DAG structure for efficiency - no NP-hard operations!
//!
//! NOTE: This tests the COMPLETE gradient path:
//! embedding → forward(morph graph) → structural loss → backward
//! The graph structure changes, so gradients must account for morphing!

use grapheme_core::{GraphemeGraph, GraphTransformNet};
use grapheme_train::{compute_structural_loss, SinkhornConfig, StructuralLossConfig};

const EMBED_DIM: usize = 64;
const EPSILON: f32 = 1e-4;  // Small perturbation for finite difference

fn main() {
    println!("🔬 Finite Difference Gradient Check");
    println!("====================================\n");

    // Create simple test case
    let input = "ab";
    let target = "a";  // Target should merge 'b' into 'a'

    let input_graph = GraphemeGraph::from_text(input);
    let target_graph = GraphemeGraph::from_text(target);

    let structural_config = StructuralLossConfig {
        alpha: 1.0,
        beta: 0.5,
        gamma: 2.0,
        sinkhorn: SinkhornConfig {
            iterations: 20,
            temperature: 0.1,
            epsilon: 1e-8,
        },
    };

    println!("📊 Test Case:");
    println!("   Input:  '{}' ({} nodes)", input, input_graph.node_count());
    println!("   Target: '{}' ({} nodes)", target, target_graph.node_count());
    println!();

    // Test 1: Threshold gradient
    println!("🎯 Test 1: Merge Threshold Gradient");
    println!("-----------------------------------");
    test_threshold_gradient(&input_graph, &target_graph, &structural_config);
    println!();

    // Test 2: Embedding gradients (sample a few characters)
    println!("📝 Test 2: Embedding Gradients");
    println!("-------------------------------");
    test_embedding_gradients(&input_graph, &target_graph, &structural_config);
    println!();

    println!("✅ Gradient check complete!");
}

/// Test threshold gradient using finite difference
fn test_threshold_gradient(
    input_graph: &GraphemeGraph,
    target_graph: &GraphemeGraph,
    config: &StructuralLossConfig,
) {
    let mut model = GraphTransformNet::new(256, EMBED_DIM, 128, 2);

    // Get analytical gradient
    model.zero_grad();
    let (predicted, pooling_result) = model.forward(input_graph);
    let loss_result = compute_structural_loss(&predicted, target_graph, config);
    // Backend-104: Use activation_gradients for proper gradient chain
    model.backward(input_graph, &pooling_result, &loss_result.activation_gradients, EMBED_DIM);

    let analytical_grad = model.merge_threshold.grad;
    let original_threshold = model.merge_threshold.value;

    // Compute numerical gradient: (L(θ+ε) - L(θ-ε)) / 2ε
    model.merge_threshold.value = original_threshold + EPSILON;
    let (predicted_plus, _) = model.forward(input_graph);
    let loss_plus = compute_structural_loss(&predicted_plus, target_graph, config).total_loss;

    model.merge_threshold.value = original_threshold - EPSILON;
    let (predicted_minus, _) = model.forward(input_graph);
    let loss_minus = compute_structural_loss(&predicted_minus, target_graph, config).total_loss;

    let numerical_grad = (loss_plus - loss_minus) / (2.0 * EPSILON);

    // Restore original value
    model.merge_threshold.value = original_threshold;

    // Compare gradients
    let abs_error = (analytical_grad - numerical_grad).abs();
    let rel_error = if numerical_grad.abs() > 1e-8 {
        abs_error / numerical_grad.abs()
    } else {
        abs_error
    };

    println!("   Analytical gradient: {:.6}", analytical_grad);
    println!("   Numerical gradient:  {:.6}", numerical_grad);
    println!("   Absolute error:      {:.6}", abs_error);
    println!("   Relative error:      {:.6}", rel_error);

    if rel_error < 0.01 {
        println!("   ✓ PASS: Gradients match within 1%");
    } else if rel_error < 0.1 {
        println!("   ⚠ WARNING: Gradients match within 10% (acceptable for heuristic)");
    } else {
        println!("   ✗ FAIL: Gradients don't match (error > 10%)");
    }
}

/// Test embedding gradients for a sample of characters
fn test_embedding_gradients(
    input_graph: &GraphemeGraph,
    target_graph: &GraphemeGraph,
    config: &StructuralLossConfig,
) {
    let test_chars = vec!['a', 'b'];

    for &ch in &test_chars {
        println!("\n   Testing character '{}':", ch);
        test_single_embedding_gradient(ch, input_graph, target_graph, config);
    }
}

/// Test gradient for a single embedding dimension of a character
fn test_single_embedding_gradient(
    ch: char,
    input_graph: &GraphemeGraph,
    target_graph: &GraphemeGraph,
    config: &StructuralLossConfig,
) {
    let mut model = GraphTransformNet::new(256, EMBED_DIM, 128, 2);
    let char_idx = ch as usize;

    // Only test first dimension to keep output manageable
    let dim = 0;

    // Get analytical gradient
    model.zero_grad();
    let (predicted, pooling_result) = model.forward(input_graph);
    let loss_result = compute_structural_loss(&predicted, target_graph, config);
    // Backend-104: Use activation_gradients for proper gradient chain
    model.backward(input_graph, &pooling_result, &loss_result.activation_gradients, EMBED_DIM);

    let analytical_grad = model.embedding.grad.as_ref()
        .map(|g| g[[char_idx, dim]])
        .unwrap_or(0.0);

    // Get original weight value
    let original_weight = model.embedding.weights[[char_idx, dim]];

    // Compute numerical gradient
    model.embedding.weights[[char_idx, dim]] = original_weight + EPSILON;
    let (predicted_plus, _) = model.forward(input_graph);
    let loss_plus = compute_structural_loss(&predicted_plus, target_graph, config).total_loss;

    model.embedding.weights[[char_idx, dim]] = original_weight - EPSILON;
    let (predicted_minus, _) = model.forward(input_graph);
    let loss_minus = compute_structural_loss(&predicted_minus, target_graph, config).total_loss;

    let numerical_grad = (loss_plus - loss_minus) / (2.0 * EPSILON);

    // Restore original weight
    model.embedding.weights[[char_idx, dim]] = original_weight;

    // Compare gradients
    let abs_error = (analytical_grad - numerical_grad).abs();
    let rel_error = if numerical_grad.abs() > 1e-8 {
        abs_error / numerical_grad.abs()
    } else {
        abs_error
    };

    println!("      Dimension {}: analytical={:.6}, numerical={:.6}",
             dim, analytical_grad, numerical_grad);
    println!("      Error: abs={:.6}, rel={:.6}", abs_error, rel_error);

    if rel_error < 0.01 {
        println!("      ✓ PASS");
    } else if rel_error < 0.1 {
        println!("      ⚠ WARNING: Error within 10%");
    } else {
        println!("      ✗ FAIL: Error > 10%");
    }
}
