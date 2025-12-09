//! MNIST Training Script for GRAPHEME Neural Network
//!
//! Backend-113: Implements image classification using the neuromorphic
//! graph-based architecture. Converts MNIST images to DAGs and trains
//! using hybrid gradient descent + Hebbian learning.
//!
//! Usage:
//!   cargo run --bin train_mnist -- --data-dir ./data/mnist --epochs 10
//!
//! The MNIST dataset will be automatically downloaded if not present.

use anyhow::{Context, Result};
use clap::Parser;
use grapheme_core::{
    cross_entropy_loss_with_grad, ClassificationConfig, DagNN,
    HebbianConfig, HebbianLearning,
};
use indicatif::{ProgressBar, ProgressStyle};
use mnist::{Mnist, MnistBuilder};
use ndarray::Array1;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use std::path::PathBuf;

/// MNIST Training CLI
#[derive(Parser, Debug)]
#[command(name = "train_mnist")]
#[command(about = "Train GRAPHEME on MNIST digit classification")]
struct Args {
    /// Directory to store/load MNIST data
    #[arg(short, long, default_value = "./data/mnist")]
    data_dir: PathBuf,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 10)]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    /// Learning rate
    #[arg(short, long, default_value_t = 0.01)]
    learning_rate: f32,

    /// Number of hidden nodes
    #[arg(long, default_value_t = 128)]
    hidden_size: usize,

    /// Enable Hebbian learning
    #[arg(long)]
    hebbian: bool,

    /// Hebbian weight (0.0-1.0)
    #[arg(long, default_value_t = 0.3)]
    hebbian_weight: f32,

    /// Number of training samples (0 = all)
    #[arg(long, default_value_t = 0)]
    train_samples: usize,

    /// Number of test samples (0 = all)
    #[arg(long, default_value_t = 0)]
    test_samples: usize,

    /// Log interval (batches)
    #[arg(long, default_value_t = 100)]
    log_interval: usize,
}

/// Load and preprocess MNIST dataset
fn load_mnist(data_dir: &PathBuf, train_samples: usize, test_samples: usize) -> Result<Mnist> {
    // Create data directory if it doesn't exist
    std::fs::create_dir_all(data_dir)
        .context("Failed to create data directory")?;

    let train_len = if train_samples == 0 { 60_000 } else { train_samples.min(60_000) };
    let test_len = if test_samples == 0 { 10_000 } else { test_samples.min(10_000) };

    println!("Loading MNIST dataset from {:?}...", data_dir);
    println!("  Training samples: {}", train_len);
    println!("  Test samples: {}", test_len);

    let mnist = MnistBuilder::new()
        .base_path(data_dir.to_str().unwrap_or("./data/mnist"))
        .label_format_digit()
        .training_set_length(train_len as u32)
        .test_set_length(test_len as u32)
        .finalize();

    Ok(mnist)
}

/// Normalize pixel values from u8 [0, 255] to f32 [0.0, 1.0]
fn normalize_image(pixels: &[u8]) -> Vec<f32> {
    pixels.iter().map(|&p| p as f32 / 255.0).collect()
}

/// Train one epoch
fn train_epoch(
    mnist: &Mnist,
    config: &ClassificationConfig,
    epoch: usize,
) -> (f32, f32) {
    let num_samples = mnist.trn_lbl.len();
    let num_batches = (num_samples + config.batch_size - 1) / config.batch_size;

    let pb = ProgressBar::new(num_batches as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut total_loss = 0.0;
    let mut correct = 0;
    let mut total = 0;

    // Shuffle indices for this epoch
    let mut indices: Vec<usize> = (0..num_samples).collect();
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    for batch_idx in 0..num_batches {
        let start = batch_idx * config.batch_size;
        let end = (start + config.batch_size).min(num_samples);
        let batch_indices = &indices[start..end];

        let mut batch_loss = 0.0;
        let mut batch_correct = 0;

        for &sample_idx in batch_indices {
            // Get image and label
            let img_start = sample_idx * 784;
            let img_end = img_start + 784;
            let pixels = normalize_image(&mnist.trn_img[img_start..img_end]);
            let label = mnist.trn_lbl[sample_idx] as usize;

            // Create DAG from image
            let mut dag = match DagNN::from_mnist_with_classifier(&pixels, config.hidden_size) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Failed to create DAG: {}", e);
                    continue;
                }
            };

            // Forward pass
            if let Err(e) = dag.neuromorphic_forward() {
                eprintln!("Forward pass failed: {}", e);
                continue;
            }

            // Get logits and compute loss
            let logits = dag.get_classification_logits();
            if logits.is_empty() {
                continue;
            }

            let (loss, grad) = cross_entropy_loss_with_grad(&logits, label);
            batch_loss += loss;

            // Check prediction
            let predicted = dag.predict_class();
            if predicted == label {
                batch_correct += 1;
            }

            // Backward pass - update edge weights
            // Convert gradient to output node gradients
            let mut output_grad: HashMap<petgraph::graph::NodeIndex, Array1<f32>> = HashMap::new();
            for (i, &node_id) in dag.output_nodes().iter().enumerate() {
                if i < grad.len() {
                    output_grad.insert(node_id, Array1::from_vec(vec![grad[i]]));
                }
            }

            // Apply gradient updates to edges
            apply_gradient_update(&mut dag, &output_grad, config.learning_rate);

            // Optional: Hebbian learning
            if config.use_hebbian {
                let hebbian_config = HebbianConfig::new(config.learning_rate * config.hebbian_weight);
                dag.backward_hebbian(&hebbian_config);
            }

            total += 1;
        }

        total_loss += batch_loss;
        correct += batch_correct;

        if batch_idx % config.log_interval == 0 {
            let avg_loss = batch_loss / batch_indices.len() as f32;
            let batch_acc = batch_correct as f32 / batch_indices.len() as f32 * 100.0;
            pb.set_message(format!("loss: {:.4}, acc: {:.1}%", avg_loss, batch_acc));
        }

        pb.inc(1);
    }

    pb.finish_with_message(format!(
        "Epoch {} complete - loss: {:.4}, acc: {:.2}%",
        epoch + 1,
        total_loss / total as f32,
        correct as f32 / total as f32 * 100.0
    ));

    (total_loss / total as f32, correct as f32 / total as f32)
}

/// Apply gradient updates to edge weights
fn apply_gradient_update(
    dag: &mut DagNN,
    output_grad: &HashMap<petgraph::graph::NodeIndex, Array1<f32>>,
    learning_rate: f32,
) {
    use petgraph::Direction;

    // Simple gradient descent on edges connected to output nodes
    for (&node_id, grad) in output_grad {
        let grad_val = grad.iter().sum::<f32>() / grad.len() as f32;

        // Update incoming edges to this output node
        let incoming: Vec<_> = dag.graph.edges_directed(node_id, Direction::Incoming)
            .map(|e| e.id())
            .collect();

        for edge_id in incoming {
            if let Some(edge) = dag.graph.edge_weight_mut(edge_id) {
                // Gradient descent: w -= lr * grad
                edge.weight -= learning_rate * grad_val;
                // Clamp weights
                edge.weight = edge.weight.clamp(-10.0, 10.0);
            }
        }
    }
}

/// Evaluate on test set
fn evaluate(mnist: &Mnist, config: &ClassificationConfig) -> (f32, f32) {
    let num_samples = mnist.tst_lbl.len();

    println!("Evaluating on {} test samples...", num_samples);

    let pb = ProgressBar::new(num_samples as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.green/white} {pos}/{len}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut total_loss = 0.0;
    let mut correct = 0;

    for sample_idx in 0..num_samples {
        let img_start = sample_idx * 784;
        let img_end = img_start + 784;
        let pixels = normalize_image(&mnist.tst_img[img_start..img_end]);
        let label = mnist.tst_lbl[sample_idx] as usize;

        // Create DAG and run forward pass
        let mut dag = match DagNN::from_mnist_with_classifier(&pixels, config.hidden_size) {
            Ok(d) => d,
            Err(_) => continue,
        };

        if dag.neuromorphic_forward().is_err() {
            continue;
        }

        let logits = dag.get_classification_logits();
        if logits.is_empty() {
            continue;
        }

        let (loss, _) = cross_entropy_loss_with_grad(&logits, label);
        total_loss += loss;

        if dag.predict_class() == label {
            correct += 1;
        }

        pb.inc(1);
    }

    pb.finish();

    let avg_loss = total_loss / num_samples as f32;
    let accuracy = correct as f32 / num_samples as f32;

    println!("Test Results:");
    println!("  Loss: {:.4}", avg_loss);
    println!("  Accuracy: {:.2}% ({}/{})", accuracy * 100.0, correct, num_samples);

    (avg_loss, accuracy)
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("===========================================");
    println!("  GRAPHEME MNIST Training (Backend-113)");
    println!("===========================================");
    println!();

    // Load dataset
    let mnist = load_mnist(&args.data_dir, args.train_samples, args.test_samples)?;

    // Create configuration
    let config = ClassificationConfig {
        learning_rate: args.learning_rate,
        hidden_size: args.hidden_size,
        batch_size: args.batch_size,
        epochs: args.epochs,
        use_hebbian: args.hebbian,
        hebbian_weight: args.hebbian_weight,
        log_interval: args.log_interval,
    };

    println!("Configuration:");
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Batch size: {}", config.batch_size);
    println!("  Epochs: {}", config.epochs);
    println!("  Hebbian: {} (weight: {})", config.use_hebbian, config.hebbian_weight);
    println!();

    // Training loop
    println!("Starting training...");
    println!();

    let mut best_accuracy = 0.0;

    for epoch in 0..config.epochs {
        println!("Epoch {}/{}", epoch + 1, config.epochs);

        let (_train_loss, _train_acc) = train_epoch(&mnist, &config, epoch);

        // Evaluate every epoch
        let (_test_loss, test_acc) = evaluate(&mnist, &config);

        if test_acc > best_accuracy {
            best_accuracy = test_acc;
            println!("  New best accuracy: {:.2}%", best_accuracy * 100.0);
        }

        println!();
    }

    println!("===========================================");
    println!("  Training Complete!");
    println!("  Best Test Accuracy: {:.2}%", best_accuracy * 100.0);
    println!("===========================================");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_image() {
        let pixels = vec![0u8, 127, 255];
        let normalized = normalize_image(&pixels);

        assert!((normalized[0] - 0.0).abs() < 1e-6);
        assert!((normalized[1] - 0.498).abs() < 0.01);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }
}
