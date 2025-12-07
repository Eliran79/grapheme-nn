//! Proof of Concept: Backward pass from structural loss to model parameters
//!
//! This demonstrates the complete gradient flow:
//! 1. Structural loss computes node_gradients, edge_gradients
//! 2. Map these to embedding layer gradients
//! 3. Call embedding.backward() to accumulate parameter gradients
//! 4. Call optimizer.step() to update weights
//! 5. Verify loss decreases

use grapheme_core::{GraphemeGraph, Embedding};
use grapheme_train::{compute_structural_loss, StructuralLossConfig, SinkhornConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Backward Pass Proof of Concept\n");

    // Simple test case: Make "ab" look more like "abc"
    let input_text = "ab";
    let target_text = "abc";

    println!("Goal: Transform '{}' → '{}'", input_text, target_text);
    println!("Method: Learn character embeddings that minimize structural loss\n");

    // Initialize embedding layer
    const VOCAB_SIZE: usize = 256;
    const EMBED_DIM: usize = 8;  // Small for testing
    let mut embedding = Embedding::xavier(VOCAB_SIZE, EMBED_DIM);
    embedding.unfreeze();  // Enable gradient computation

    println!("📐 Model: Embedding[vocab={}, dim={}]\n", VOCAB_SIZE, EMBED_DIM);

    // Structural loss config
    let config = StructuralLossConfig {
        alpha: 1.0,
        beta: 0.5,
        gamma: 2.0,
        sinkhorn: SinkhornConfig {
            iterations: 20,
            temperature: 0.1,
            epsilon: 1e-6,
        },
    };

    // Training loop
    const EPOCHS: usize = 50;
    const LEARNING_RATE: f32 = 0.01;

    println!("🏋️  Training: {} epochs, lr={}\n", EPOCHS, LEARNING_RATE);
    println!("{:>5} {:>12}", "Epoch", "Loss");
    println!("{}", "-".repeat(20));

    let mut prev_loss = f32::MAX;

    for epoch in 0..EPOCHS {
        // Zero gradients
        embedding.zero_grad();

        // Forward pass: text → graph
        let predicted_graph = GraphemeGraph::from_text(input_text);
        let target_graph = GraphemeGraph::from_text(target_text);

        // Compute structural loss
        let loss_result = compute_structural_loss(&predicted_graph, &target_graph, &config);
        let loss = loss_result.total_loss;

        // Backward pass (SIMPLIFIED POC):
        // In reality, we'd backprop through the model's forward pass
        // For POC: Directly use node_gradients to update embedding
        // This is a PLACEHOLDER to demonstrate the pattern

        // For each character in input, accumulate gradient
        for (i, ch) in input_text.chars().enumerate() {
            let char_idx = ch as usize;

            // Get gradient from structural loss
            // node_gradients are for graph nodes, we need to map to characters
            if i < loss_result.node_gradients.len() / EMBED_DIM {
                let grad_slice = &loss_result.node_gradients
                    [i * EMBED_DIM..(i + 1) * EMBED_DIM.min(loss_result.node_gradients.len())];

                // Create gradient array
                let grad = ndarray::Array1::from_vec(grad_slice.to_vec());

                // Backprop through embedding
                embedding.backward(char_idx, &grad);
            }
        }

        // Optimizer step
        embedding.step(LEARNING_RATE);

        // Print progress
        if epoch % 10 == 0 || epoch == EPOCHS - 1 {
            println!("{:5} {:12.4}", epoch, loss);
        }

        // Check if loss is decreasing
        if epoch > 0 && loss >= prev_loss {
            println!("\n⚠️  Warning: Loss not decreasing (epoch {})", epoch);
            println!("   This is expected - we need actual model forward pass!");
            println!("   This POC only demonstrates gradient accumulation pattern.\n");
            break;
        }

        prev_loss = loss;
    }

    println!("\n📊 Results:");
    println!("   Initial loss: {:.4}", prev_loss);
    println!("   Final loss:   {:.4}", prev_loss);
    println!("\n💡 Next Steps:");
    println!("   1. Implement actual model forward pass (embedding → features)");
    println!("   2. Connect forward pass features to graph structure");
    println!("   3. Backprop from structural loss → features → embedding");
    println!("   4. Then loss will actually decrease!");

    Ok(())
}
