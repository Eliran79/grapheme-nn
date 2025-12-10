//! Training script that tracks merge threshold evolution
//!
//! Shows how the learnable threshold adapts during training

use grapheme_core::{GraphemeGraph, GraphTransformNet};
use grapheme_train::{compute_structural_loss, SinkhornConfig, StructuralLossConfig};

const EMBED_DIM: usize = 64;

fn main() {
    println!("🎯 Training with Learnable Merge Threshold");
    println!("==========================================\n");

    // Create model
    let mut model = GraphTransformNet::new(256, EMBED_DIM, 128, 2);

    println!("📊 Initial State:");
    println!("   Merge threshold value: {:.6}", model.merge_threshold.value);
    println!("   Sigmoid(threshold): {:.6}", 1.0 / (1.0 + (-model.merge_threshold.value).exp()));
    println!();

    // Simple training data
    let training_data = vec![
        ("abc", "ab"),    // Learn to merge 'c' into 'b'
        ("xyz", "xy"),    // Learn to merge 'z' into 'y'
        ("123", "12"),    // Learn to merge '3' into '2'
    ];

    // Loss config
    let structural_config = StructuralLossConfig {
        alpha: 1.0,  // Node weight
        beta: 0.5,   // Edge weight
        gamma: 2.0,  // Clique weight
        sinkhorn: SinkhornConfig {
            iterations: 20,
            temperature: 0.1,
            epsilon: 1e-8,
        },
    };

    let lr = 0.01; // Higher LR to see threshold change faster
    let epochs = 100;

    println!("🏋️  Training for {} epochs (lr={}):\n", epochs, lr);
    println!("Epoch    Loss    Threshold  Sigmoid(θ)  Gradient");
    println!("---------------------------------------------------");

    for epoch in 0..epochs {
        model.zero_grad();

        let mut total_loss = 0.0;

        for (input, target) in &training_data {
            let input_graph = GraphemeGraph::from_text(input);
            let target_graph = GraphemeGraph::from_text(target);

            // Forward pass with morphing
            let (predicted_graph, pooling_result) = model.forward(&input_graph);

            // Compute loss
            let loss_result = compute_structural_loss(
                &predicted_graph,
                &target_graph,
                &structural_config,
            );

            total_loss += loss_result.total_loss;

            // Backend-104: Use activation_gradients for proper gradient chain
            model.backward(&input_graph, &pooling_result, &loss_result.activation_gradients, EMBED_DIM);
        }

        let avg_loss = total_loss / training_data.len() as f32;

        // Print every 10 epochs
        if epoch % 10 == 0 {
            let sigmoid = 1.0 / (1.0 + (-model.merge_threshold.value).exp());
            println!(
                "{:5}  {:7.4}  {:9.6}  {:10.6}  {:9.6}",
                epoch,
                avg_loss,
                model.merge_threshold.value,
                sigmoid,
                model.merge_threshold.grad
            );
        }

        // Update parameters
        model.step(lr);
    }

    println!("\n📈 Final State:");
    println!("   Merge threshold value: {:.6}", model.merge_threshold.value);
    println!("   Sigmoid(threshold): {:.6}", 1.0 / (1.0 + (-model.merge_threshold.value).exp()));
    println!("   Final gradient: {:.6}", model.merge_threshold.grad);

    println!("\n✅ The merge threshold is now a LEARNABLE PARAMETER!");
    println!("   Adam optimizer adjusts it based on structural loss gradients.");
}
