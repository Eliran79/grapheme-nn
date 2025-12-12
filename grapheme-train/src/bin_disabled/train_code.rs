//! Train DAG-NN for Code Generation
//!
//! Trains the model to generate code from natural language prompts.
//! Uses HumanEval/MBPP training data format.
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin train_code -- \
//!     --data data/code_training --output checkpoints/code_model.json

use clap::Parser;
use grapheme_core::{GraphTransformNet, GraphemeGraph, UnifiedCheckpoint};
use grapheme_train::{
    compute_structural_loss, Adam, LRScheduler, StructuralLossConfig,
    TrainingLoop, TrainingMetrics, TrainingState,
};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_code")]
#[command(about = "Train DAG-NN for code generation")]
struct Args {
    /// Path to code training data directory
    #[arg(short, long)]
    data: PathBuf,

    /// Output path for trained model
    #[arg(short, long)]
    output: PathBuf,

    /// Number of training epochs
    #[arg(short, long, default_value = "50")]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value = "16")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.0003")]
    lr: f64,

    /// Resume from checkpoint
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Training example from JSONL
#[derive(Debug, Deserialize, Serialize)]
struct CodeExample {
    id: String,
    input: String,
    target: String,
    #[serde(default)]
    level: u32,
}

/// Load code examples from JSONL file
fn load_examples(path: &PathBuf) -> anyhow::Result<Vec<CodeExample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut examples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let example: CodeExample = serde_json::from_str(&line)?;
        examples.push(example);
    }

    Ok(examples)
}

/// Save unified checkpoint
fn save_checkpoint(
    path: &PathBuf,
    model: &GraphTransformNet,
    state: &TrainingState,
    metrics: &TrainingMetrics,
    optimizer: &Adam,
) -> anyhow::Result<()> {
    let mut checkpoint = UnifiedCheckpoint::new();
    checkpoint.add_module(model)?;
    checkpoint.add_module(state)?;
    checkpoint.add_module(metrics)?;
    checkpoint.add_module(optimizer)?;
    checkpoint.save_to_file(path)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("DAG-NN Code Generation Trainer");
    println!("==============================\n");

    // Model architecture - larger for code
    const VOCAB_SIZE: usize = 256;  // ASCII
    const EMBED_DIM: usize = 128;   // 2x math
    const HIDDEN_DIM: usize = 256;  // 2x math
    const NUM_LAYERS: usize = 4;    // Deeper

    // Load training data
    let train_path = args.data.join("code_train.jsonl");
    let val_path = args.data.join("code_val.jsonl");

    println!("Loading training data from {:?}", train_path);
    let train_examples = load_examples(&train_path)?;
    println!("Loaded {} training examples", train_examples.len());

    let val_examples = if val_path.exists() {
        let examples = load_examples(&val_path)?;
        println!("Loaded {} validation examples", examples.len());
        Some(examples)
    } else {
        None
    };

    // Create output directory
    let output_dir = args.output.parent().unwrap_or(&args.output);
    fs::create_dir_all(output_dir)?;

    // Initialize or resume
    let mut model = if let Some(resume_path) = &args.resume {
        println!("\nResuming from {:?}", resume_path);
        let checkpoint = UnifiedCheckpoint::load_from_file(resume_path)?;
        checkpoint.load_module()?
    } else {
        println!("\nInitializing new model...");
        println!("  Vocab: {}", VOCAB_SIZE);
        println!("  Embed dim: {}", EMBED_DIM);
        println!("  Hidden dim: {}", HIDDEN_DIM);
        println!("  Layers: {}", NUM_LAYERS);
        GraphTransformNet::new(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)
    };

    // Training config
    let config = grapheme_train::TrainingConfig {
        learning_rate: args.lr as f32,
        batch_size: args.batch_size,
        epochs: args.epochs,
        alpha: 1.0,    // Node cost weight
        beta: 0.5,     // Edge cost weight
        gamma: 0.3,    // Mismatch weight
        val_frequency: 1,
        patience: 10,
    };

    let optimizer = Adam::new(config.learning_rate)
        .with_beta1(0.9)
        .with_beta2(0.999)
        .with_weight_decay(0.01);

    let mut training_loop = TrainingLoop::new(config.clone())
        .with_scheduler(LRScheduler::CosineAnnealingLR {
            t_max: args.epochs,
            eta_min: (args.lr * 0.01) as f32,
        });

    // Structural loss config
    let loss_config = StructuralLossConfig {
        alpha: 1.0,  // Node cost
        beta: 0.5,   // Edge cost
        gamma: 0.3,  // Clique weight
        sinkhorn: grapheme_train::SinkhornConfig {
            iterations: 20,
            temperature: 0.1,
            epsilon: 1e-6,
        },
    };

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.lr);
    println!("  Patience: {} epochs", config.patience);

    // Training loop
    let start = Instant::now();
    let mut best_val_loss = f32::INFINITY;

    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut epoch_accuracy = 0.0f32;
        let mut batch_count = 0;

        // Shuffle and batch training data
        let mut indices: Vec<usize> = (0..train_examples.len()).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        for batch_start in (0..indices.len()).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];

            model.zero_grad();

            let mut batch_loss = 0.0f32;
            let mut batch_accuracy = 0.0f32;

            for &idx in batch_indices {
                let example = &train_examples[idx];

                // Convert prompt → graph, target code → graph
                let input_graph = GraphemeGraph::from_text(&example.input);
                let target_graph = GraphemeGraph::from_text(&example.target);

                // Forward pass
                let (predicted_graph, pooling_result) = model.forward(&input_graph);

                // Structural loss
                let loss_result = compute_structural_loss(
                    &predicted_graph,
                    &target_graph,
                    &loss_config,
                );

                batch_loss += loss_result.total_loss;

                // Accuracy as graph similarity
                let max_nodes = predicted_graph.node_count().max(target_graph.node_count());
                let max_edges = predicted_graph.edge_count().max(target_graph.edge_count());
                let max_cost = (max_nodes + max_edges) as f32;
                let accuracy = if max_cost > 0.0 {
                    1.0 - (loss_result.total_loss / max_cost).min(1.0)
                } else {
                    1.0
                };
                batch_accuracy += accuracy;

                // Backward pass
                model.backward(&input_graph, &pooling_result, &loss_result.activation_gradients, EMBED_DIM);
            }

            let n = batch_indices.len() as f32;
            batch_loss /= n;
            batch_accuracy /= n;

            // Update weights
            let lr = training_loop.state.current_lr;
            model.step(lr);

            epoch_loss += batch_loss;
            epoch_accuracy += batch_accuracy;
            batch_count += 1;

            training_loop.record_batch(batch_loss);

            if args.verbose && batch_count % 10 == 0 {
                println!("    Batch {}: loss={:.4}, similarity={:.1}%",
                    batch_count, batch_loss, batch_accuracy * 100.0);
            }
        }

        let avg_loss = epoch_loss / batch_count as f32;
        let avg_accuracy = epoch_accuracy / batch_count as f32;

        // Validation
        let val_loss = if let Some(ref val) = val_examples {
            let mut vl = 0.0f32;
            let mut va = 0.0f32;

            for example in val {
                let input_graph = GraphemeGraph::from_text(&example.input);
                let target_graph = GraphemeGraph::from_text(&example.target);
                let (predicted_graph, _) = model.forward(&input_graph);

                let loss_result = compute_structural_loss(
                    &predicted_graph,
                    &target_graph,
                    &loss_config,
                );

                vl += loss_result.total_loss;

                let max_nodes = predicted_graph.node_count().max(target_graph.node_count());
                let max_edges = predicted_graph.edge_count().max(target_graph.edge_count());
                let max_cost = (max_nodes + max_edges) as f32;
                let acc = if max_cost > 0.0 {
                    1.0 - (loss_result.total_loss / max_cost).min(1.0)
                } else {
                    1.0
                };
                va += acc;
            }

            let n = val.len() as f32;
            let val_loss = vl / n;
            let val_acc = va / n;

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                // Save best model
                let best_path = args.output.with_file_name("code_model_best.json");
                model.save_to_file(&best_path)?;
                println!("  [NEW BEST] val_loss={:.4}, saved to {:?}", val_loss, best_path);
            }

            training_loop.record_validation(val_loss, val_acc);
            Some((val_loss, val_acc))
        } else {
            None
        };

        training_loop.complete_epoch();

        let epoch_time = epoch_start.elapsed();

        // Print epoch summary
        print!("Epoch {}/{}: train_loss={:.4}, similarity={:.1}%",
            epoch + 1, args.epochs, avg_loss, avg_accuracy * 100.0);

        if let Some((vl, va)) = val_loss {
            print!(", val_loss={:.4}, val_sim={:.1}%", vl, va * 100.0);
        }

        println!(", lr={:.6}, time={:.1}s", training_loop.state.current_lr, epoch_time.as_secs_f64());

        // Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 {
            let ckpt_path = args.output.with_file_name(format!("code_model_epoch{}.json", epoch + 1));
            save_checkpoint(&ckpt_path, &model, &training_loop.state, &training_loop.metrics, &optimizer)?;
            println!("  Checkpoint saved: {:?}", ckpt_path);
        }

        // Early stopping
        if training_loop.should_stop() {
            println!("\nEarly stopping after {} epochs without improvement",
                training_loop.state.epochs_without_improvement);
            break;
        }
    }

    let total_time = start.elapsed();

    // Save final model
    model.save_to_file(&args.output)?;
    println!("\nTraining complete in {:.1}s", total_time.as_secs_f64());
    println!("Final model saved to: {:?}", args.output);
    println!("Best validation loss: {:.4}", best_val_loss);

    Ok(())
}
