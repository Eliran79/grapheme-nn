//! Gradient descent direction test
//!
//! Instead of finite difference (which fails for discrete graph morphing),
//! verify that gradients point in descent direction:
//! - Loss should DECREASE when we move in negative gradient direction
//! - This is the fundamental requirement for optimization
//!
//! Avoids NP-hard problems: uses existing O(n²) morphing operations

use grapheme_core::{GraphemeGraph, GraphTransformNet};
use grapheme_train::{compute_structural_loss, SinkhornConfig, StructuralLossConfig};

const EMBED_DIM: usize = 64;

fn main() {
    println!("⬇️  Gradient Descent Direction Test");
    println!("===================================\n");

    println!("Testing fundamental optimization property:");
    println!("  If gradients are correct → loss decreases when we update parameters\n");

    // Create test case: learn to merge similar nodes
    let training_examples = vec![
        ("abc", "ab"),   // Learn to merge 'c' into 'b'
        ("xyz", "xy"),   // Learn to merge 'z' into 'y'
        ("123", "12"),   // Learn to merge '3' into '2'
    ];

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

    println!("📊 Test Cases:");
    for (input, target) in &training_examples {
        println!("   '{}' → '{}' (merge last char)", input, target);
    }
    println!();

    // Test 1: Single gradient descent step
    println!("🎯 Test 1: Single Update Step");
    println!("------------------------------");
    test_single_step(&training_examples, &structural_config);
    println!();

    // Test 2: Multiple steps show monotonic decrease
    println!("📈 Test 2: Multi-Step Convergence");
    println!("----------------------------------");
    test_convergence(&training_examples, &structural_config, 20);
    println!();

    println!("✅ Gradient descent test complete!");
}

/// Test that a single gradient step reduces loss
fn test_single_step(
    training_examples: &[(&str, &str)],
    config: &StructuralLossConfig,
) {
    let mut model = GraphTransformNet::new(256, EMBED_DIM, 128, 2);

    // Compute loss before update
    let loss_before = compute_average_loss(&model, training_examples, config);

    // Do backward pass to accumulate gradients
    model.zero_grad();
    for (input, target) in training_examples {
        let input_graph = GraphemeGraph::from_text(input);
        let target_graph = GraphemeGraph::from_text(target);

        let (predicted, pooling_result) = model.forward(&input_graph);
        let loss_result = compute_structural_loss(&predicted, &target_graph, config);

        // Backend-104: Use activation_gradients for proper gradient chain
        model.backward(&input_graph, &pooling_result, &loss_result.activation_gradients, EMBED_DIM);
    }

    // Update parameters
    let lr = 0.01;
    model.step(lr);

    // Compute loss after update
    let loss_after = compute_average_loss(&model, training_examples, config);

    // Check if loss decreased
    let delta = loss_after - loss_before;
    let percent_change = (delta / loss_before) * 100.0;

    println!("   Loss before: {:.6}", loss_before);
    println!("   Loss after:  {:.6}", loss_after);
    println!("   Change:      {:.6} ({:.2}%)", delta, percent_change);

    if delta < 0.0 {
        println!("   ✓ PASS: Loss decreased (gradients correct!)");
    } else if delta.abs() < 0.001 {
        println!("   ⚠ WARNING: Loss barely changed");
    } else {
        println!("   ✗ FAIL: Loss increased (gradients wrong!)");
    }

    // Show threshold change
    println!("\n   Threshold: {:.6} (sigmoid: {:.6})",
             model.merge_threshold.value,
             1.0 / (1.0 + (-model.merge_threshold.value).exp()));
}

/// Test convergence over multiple steps
fn test_convergence(
    training_examples: &[(&str, &str)],
    config: &StructuralLossConfig,
    epochs: usize,
) {
    let mut model = GraphTransformNet::new(256, EMBED_DIM, 128, 2);
    let lr = 0.01;

    let initial_loss = compute_average_loss(&model, training_examples, config);
    println!("   Initial loss: {:.6}\n", initial_loss);

    let mut prev_loss = initial_loss;
    let mut decreases = 0;
    let mut increases = 0;

    for epoch in 0..epochs {
        model.zero_grad();

        // Accumulate gradients
        for (input, target) in training_examples {
            let input_graph = GraphemeGraph::from_text(input);
            let target_graph = GraphemeGraph::from_text(target);

            let (predicted, pooling_result) = model.forward(&input_graph);
            let loss_result = compute_structural_loss(&predicted, &target_graph, config);

            // Backend-104: Use activation_gradients for proper gradient chain
        model.backward(&input_graph, &pooling_result, &loss_result.activation_gradients, EMBED_DIM);
        }

        // Update
        model.step(lr);

        // Check loss
        let curr_loss = compute_average_loss(&model, training_examples, config);

        if epoch % 5 == 0 {
            let delta = curr_loss - prev_loss;
            println!("   Epoch {:2}: loss={:.6} (Δ={:+.6})", epoch, curr_loss, delta);
        }

        if curr_loss < prev_loss {
            decreases += 1;
        } else {
            increases += 1;
        }

        prev_loss = curr_loss;
    }

    let final_loss = prev_loss;
    let total_change = final_loss - initial_loss;
    let percent_change = (total_change / initial_loss) * 100.0;

    println!("\n   Final loss: {:.6}", final_loss);
    println!("   Total change: {:.6} ({:.2}%)", total_change, percent_change);
    println!("   Decreases: {}, Increases: {}", decreases, increases);

    if decreases > increases && total_change < 0.0 {
        println!("   ✓ PASS: Mostly decreasing, net improvement");
    } else if decreases > increases / 2 {
        println!("   ⚠ WARNING: Some oscillation but trending down");
    } else {
        println!("   ✗ FAIL: Not converging");
    }
}

/// Compute average loss across training examples
fn compute_average_loss(
    model: &GraphTransformNet,
    examples: &[(&str, &str)],
    config: &StructuralLossConfig,
) -> f32 {
    let total_loss: f32 = examples
        .iter()
        .map(|(input, target)| {
            let input_graph = GraphemeGraph::from_text(input);
            let target_graph = GraphemeGraph::from_text(target);

            let (predicted, _pooling_result) = model.forward(&input_graph);
            compute_structural_loss(&predicted, &target_graph, config).total_loss
        })
        .sum();

    total_loss / examples.len() as f32
}
