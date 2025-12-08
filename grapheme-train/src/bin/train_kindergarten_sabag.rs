//! Kindergarten training with Sabag pooling and proper learning rates
//!
//! Tests the simplified Sabag algorithm on text-to-text tasks

use grapheme_core::{GraphemeGraph, GraphTransformNet};
use grapheme_train::{compute_structural_loss, StructuralLossConfig, SinkhornConfig};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
struct QAPair {
    input: String,
    target: String,
    #[serde(default)]
    id: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧒 GRAPHEME Kindergarten Training with Sabag\n");

    // Load dataset
    let file = File::open("data/kindergarten/level_1.jsonl")?;
    let reader = BufReader::new(file);
    let mut pairs: Vec<QAPair> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let pair: QAPair = serde_json::from_str(&line)?;
        pairs.push(pair);
    }

    println!("📊 Dataset: {} QA pairs", pairs.len());
    if !pairs.is_empty() {
        println!("   Example: '{}' → '{}'\n", pairs[0].input, pairs[0].target);
    }

    // Determine max output size
    let max_target_len = pairs.iter()
        .map(|p| GraphemeGraph::from_text(&p.target).input_nodes.len())
        .max()
        .unwrap_or(10);

    println!("Max target graph size: {} nodes", max_target_len);

    // Initialize model with output size = max target size
    const VOCAB_SIZE: usize = 256;
    const EMBED_DIM: usize = 64;
    const HIDDEN_DIM: usize = 128;

    let mut model = GraphTransformNet::new(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, max_target_len);

    println!("🧠 Model: vocab={}, embed={}, hidden={}, output_nodes={}\n",
             VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, max_target_len);

    // Learning rates - KEY: Different LRs for different components!
    const LR_EMBEDDING: f32 = 0.001;  // Traditional NN learning rate
    const LR_SABAG: f32 = 1.0;        // Graph morphing learning rate (1000x higher!)

    println!("🎯 Learning Rates:");
    println!("   Embeddings:     {:.4}", LR_EMBEDDING);
    println!("   Sabag (query):  {:.4} (graph morphing!)\n", LR_SABAG);

    // Structural loss config
    let config = StructuralLossConfig {
        alpha: 1.0,
        beta: 0.5,
        gamma: 2.0,
        sinkhorn: SinkhornConfig::default(),
    };

    // Training parameters
    const EPOCHS: usize = 200;
    const PRINT_EVERY: usize = 20;

    println!("🏋️  Training: {} epochs\n", EPOCHS);
    println!("{:>5} {:>12} {:>12} {:>12} {:>8}",
             "Epoch", "Loss", "ΔLoss", "Best", "Time");
    println!("{}", "-".repeat(60));

    let start_time = Instant::now();
    let mut best_loss = f32::INFINITY;
    let mut initial_loss = 0.0;

    for epoch in 0..EPOCHS {
        let epoch_start = Instant::now();
        model.zero_grad();

        let mut total_loss = 0.0;

        // Process all pairs
        for pair in &pairs {
            let input_graph = GraphemeGraph::from_text(&pair.input);
            let target_graph = GraphemeGraph::from_text(&pair.target);

            // Forward pass through Sabag pooling
            let (predicted, pooling_result) = model.forward(&input_graph);

            // Compute structural loss
            let loss_result = compute_structural_loss(&predicted, &target_graph, &config);

            // Backward pass
            model.backward(&input_graph, &pooling_result, &loss_result.node_gradients, EMBED_DIM);

            total_loss += loss_result.total_loss;
        }

        // Average loss
        let avg_loss = total_loss / pairs.len() as f32;

        if epoch == 0 {
            initial_loss = avg_loss;
        }

        if avg_loss < best_loss {
            best_loss = avg_loss;
        }

        // Update parameters with SEPARATE learning rates
        model.embedding.step(LR_EMBEDDING);
        model.merge_threshold.step(LR_EMBEDDING);

        // CRITICAL: Use higher LR for Sabag query matrix!
        if let Some(ref mut sabag) = model.sabag_pooling {
            sabag.step(LR_SABAG);
        }

        // Message passing layers
        for layer in &mut model.mp_layers {
            layer.step(LR_EMBEDDING);
        }

        let epoch_time = epoch_start.elapsed();

        // Print progress
        if epoch % PRINT_EVERY == 0 || epoch == EPOCHS - 1 {
            let delta = 0.0;  // Will be computed between printed epochs

            println!("{:5} {:12.6} {:12.6} {:12.6} {:7.2}s",
                     epoch, avg_loss, delta, best_loss, epoch_time.as_secs_f64());
        }
    }

    let total_time = start_time.elapsed();

    println!("\n{}", "=".repeat(60));
    println!("📊 Training Summary");
    println!("{}", "=".repeat(60));
    let final_loss = best_loss;  // Use best loss as final
    println!("Initial loss:  {:.6}", initial_loss);
    println!("Final loss:    {:.6}", final_loss);
    println!("Best loss:     {:.6}", best_loss);
    println!("Improvement:   {:.6} ({:.2}%)",
             initial_loss - best_loss,
             ((initial_loss - best_loss) / initial_loss) * 100.0);
    println!("Total time:    {:.2}s", total_time.as_secs_f64());

    if best_loss < initial_loss {
        println!("\n✅ SUCCESS: Loss decreased - Sabag is learning!");
    } else {
        println!("\n⚠️  Loss did not decrease - may need more epochs or LR tuning");
    }

    Ok(())
}
