//! GRAPHEME Semantic Code Training - TRUE Graph-to-Graph
//!
//! This is the GRAPHEME vision for code generation:
//! - Input: Text graph (prompt/docstring)
//! - Output: Semantic code graph (keywords, variables, operators, NOT characters!)
//!
//! Example transformation:
//!   "write a function that prints Hi if x>2"
//!     →
//!   Graph: [Keyword(def), Variable(f), Punct('('), Variable(x), Punct(')'), Punct(':'),
//!           Space(Newline), Space(Indent), Keyword(if), Variable(x), Op(>), Int(2),
//!           Punct(':'), Space(Newline), Space(Indent), Call(print), Punct('('),
//!           Str("Hi"), Punct(')'), EndSeq]
//!
//! This trains on SEMANTIC NODES not characters!

use anyhow::Result;
use clap::Parser;
use grapheme_core::GraphemeGraph;
use grapheme_train::cortex_mesh::{CortexMesh, MeshConfig};
use grapheme_train::{compute_structural_loss, StructuralLossConfig};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_semantic_code")]
#[command(about = "Train GRAPHEME on semantic code graphs (NOT characters!)")]
struct Args {
    /// Path to training data directory
    #[arg(short, long)]
    data: PathBuf,

    /// Output checkpoint path
    #[arg(short, long)]
    output: PathBuf,

    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value = "16")]
    batch_size: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    lr: f32,

    /// Resume from checkpoint
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Verbose output level (0-2)
    #[arg(short, long, default_value = "1")]
    verbose: usize,
}

/// Training sample
#[derive(Debug, Clone, Deserialize)]
struct TrainingSample {
    input: String,
    #[serde(alias = "target")]
    output: String,
    #[serde(default)]
    domain: Option<String>,
}

/// Load training data
fn load_training_data(data_dir: &PathBuf) -> Result<Vec<TrainingSample>> {
    let mut samples = Vec::new();

    // Look for JSONL files
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
            let file = File::open(&path)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(sample) = serde_json::from_str::<TrainingSample>(&line) {
                    samples.push(sample);
                }
            }
        }
    }

    // Also try loading from subdirectories
    for subdir in ["code", "humaneval"] {
        let subpath = data_dir.join(subdir);
        if subpath.exists() {
            for entry in std::fs::read_dir(&subpath)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
                    let file = File::open(&path)?;
                    let reader = BufReader::new(file);

                    for line in reader.lines() {
                        let line = line?;
                        if line.trim().is_empty() {
                            continue;
                        }
                        if let Ok(sample) = serde_json::from_str::<TrainingSample>(&line) {
                            samples.push(sample);
                        }
                    }
                }
            }
        }
    }

    Ok(samples)
}

/// Compute semantic node type accuracy
fn semantic_accuracy(pred_graph: &GraphemeGraph, target_graph: &GraphemeGraph) -> f32 {
    let pred_nodes: Vec<_> = pred_graph.graph.node_indices()
        .map(|idx| &pred_graph.graph[idx])
        .collect();
    let target_nodes: Vec<_> = target_graph.graph.node_indices()
        .map(|idx| &target_graph.graph[idx])
        .collect();

    if target_nodes.is_empty() {
        return if pred_nodes.is_empty() { 1.0 } else { 0.0 };
    }

    let mut matches = 0;
    let check_len = pred_nodes.len().min(target_nodes.len());

    for i in 0..check_len {
        // Compare node types
        let pred_type = format!("{:?}", pred_nodes[i].node_type);
        let target_type = format!("{:?}", target_nodes[i].node_type);

        // Extract the type name (before any parameters)
        let pred_name = pred_type.split('(').next().unwrap_or(&pred_type);
        let target_name = target_type.split('(').next().unwrap_or(&target_type);

        if pred_name == target_name {
            matches += 1;
        }
    }

    matches as f32 / target_nodes.len() as f32
}

/// Compute exact code match
fn exact_code_match(pred_graph: &GraphemeGraph, target_code: &str) -> bool {
    let predicted_code = pred_graph.to_code();
    predicted_code.trim() == target_code.trim()
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     GRAPHEME Semantic Code Training - TRUE Graph-to-Graph   ║");
    println!("║   Nodes are: Keyword, Variable, Int, Op, etc. NOT chars!    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load training data
    println!("Loading training data from {:?}...", args.data);
    let samples = load_training_data(&args.data)?;
    println!("Loaded {} training samples", samples.len());

    if samples.is_empty() {
        println!("WARNING: No training samples found!");
        println!("Expected JSONL files with {{\"input\": ..., \"output\": ...}} format");
        return Ok(());
    }

    // Show example semantic graph conversion
    println!("\n--- Semantic Graph Example ---");
    let example_code = "if x > 2:\n    print('Hi')";
    let semantic_graph = GraphemeGraph::from_code(example_code);
    println!("Code: {}", example_code);
    println!("Semantic nodes ({} nodes):", semantic_graph.node_count());
    for (i, idx) in semantic_graph.graph.node_indices().take(10).enumerate() {
        let node = &semantic_graph.graph[idx];
        println!("  [{}] {:?}", i, node.node_type);
    }
    if semantic_graph.node_count() > 10 {
        println!("  ... ({} more nodes)", semantic_graph.node_count() - 10);
    }
    let reconstructed = semantic_graph.to_code();
    println!("Reconstructed: {}", reconstructed);
    println!();

    // Create mesh configuration
    let config = MeshConfig {
        activation_threshold: 0.2,
        max_active_brains: usize::MAX,
        parallel: true,
        hidden_dim: 256,
        num_layers: 6,
        vocab_size: 256,  // For character-level input
        embed_dim: 64,
    };

    // Create or resume mesh
    let mut mesh = if let Some(resume_path) = &args.resume {
        println!("Resuming from {:?}...", resume_path);
        let mut mesh = CortexMesh::discover_with_config(config);
        mesh.load(resume_path)?;
        mesh
    } else {
        println!("Initializing CortexMesh...");
        CortexMesh::discover_with_config(config)
    };

    println!("\nMesh Ready:");
    println!("  Brains: {}", mesh.brain_count());
    println!("  Modules: {}", mesh.module_count());

    // Training loop
    let mut best_loss = f32::MAX;
    let train_size = (samples.len() as f32 * 0.9) as usize;
    let (train_samples, val_samples) = samples.split_at(train_size);

    println!("\nTraining: {} samples, Validation: {} samples", train_samples.len(), val_samples.len());
    println!("Epochs: {}, Batch size: {}, LR: {}\n", args.epochs, args.batch_size, args.lr);

    let loss_config = StructuralLossConfig::default();

    // Learning rate schedule
    let warmup_epochs = 5;
    let min_lr = args.lr * 0.01;

    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();

        // Calculate learning rate with warmup and cosine decay
        let current_lr = if epoch < warmup_epochs {
            args.lr * (epoch + 1) as f32 / warmup_epochs as f32
        } else {
            let decay_epoch = epoch - warmup_epochs;
            let decay_total = args.epochs - warmup_epochs;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * decay_epoch as f32 / decay_total as f32).cos());
            min_lr + (args.lr - min_lr) * cosine
        };

        // Training
        let mut train_loss = 0.0;
        let pb = ProgressBar::new(train_samples.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

        for batch in train_samples.chunks(args.batch_size) {
            mesh.zero_grad();

            for sample in batch {
                // KEY CHANGE: Use from_code() for target to get SEMANTIC graph!
                let input_graph = GraphemeGraph::from_text(&sample.input);
                let target_graph = GraphemeGraph::from_code(&sample.output);

                // Forward pass
                let (output_graph, pooling) = mesh.model.forward(&input_graph);

                // Compute structural loss against SEMANTIC target
                let loss_result = compute_structural_loss(&output_graph, &target_graph, &loss_config);
                train_loss += loss_result.total_loss;

                // Backward pass
                mesh.model.backward(&input_graph, &pooling, &loss_result.activation_gradients, mesh.config.embed_dim);
            }

            mesh.model.step(current_lr / batch.len() as f32);
            pb.inc(batch.len() as u64);
        }
        pb.finish_and_clear();

        train_loss /= train_samples.len() as f32;

        // Validation
        let mut val_loss = 0.0;
        let mut val_semantic_acc = 0.0;
        let mut val_exact_matches = 0;

        for sample in val_samples {
            let input_graph = GraphemeGraph::from_text(&sample.input);
            let target_graph = GraphemeGraph::from_code(&sample.output);

            let (output_graph, _) = mesh.model.forward(&input_graph);

            let loss_result = compute_structural_loss(&output_graph, &target_graph, &loss_config);
            val_loss += loss_result.total_loss;

            // Semantic accuracy
            val_semantic_acc += semantic_accuracy(&output_graph, &target_graph);

            // Exact code match
            if exact_code_match(&output_graph, &sample.output) {
                val_exact_matches += 1;
            }
        }

        val_loss /= val_samples.len() as f32;
        val_semantic_acc /= val_samples.len() as f32;
        let exact_match_rate = val_exact_matches as f32 / val_samples.len() as f32;

        let epoch_time = epoch_start.elapsed();

        // Save best model
        let is_best = val_loss < best_loss;
        if is_best {
            best_loss = val_loss;
            let best_path = args.output.with_file_name("semantic_code_best.json");
            mesh.save(&best_path)?;
        }

        // Print progress
        print!("Epoch {}/{}: train_loss={:.4}, val_loss={:.4}, semantic_acc={:.1}%, exact={:.1}%",
            epoch + 1, args.epochs, train_loss, val_loss,
            val_semantic_acc * 100.0, exact_match_rate * 100.0);

        if is_best {
            print!(" [BEST]");
        }
        println!(", lr={:.6}, time={:.1}s", current_lr, epoch_time.as_secs_f64());

        // Demo inference every 10 epochs
        if args.verbose >= 1 && (epoch + 1) % 10 == 0 {
            println!("\n--- Demo Inference (Epoch {}) ---", epoch + 1);
            let demo_input = &val_samples[0].input;
            let demo_target = &val_samples[0].output;

            let input_graph = GraphemeGraph::from_text(demo_input);
            let (output_graph, _) = mesh.model.forward(&input_graph);

            println!("Input: {}...", demo_input.chars().take(60).collect::<String>());
            println!("Target code: {}", demo_target.chars().take(80).collect::<String>());
            println!("Predicted graph ({} nodes):", output_graph.node_count());

            // Show predicted semantic nodes
            for (i, idx) in output_graph.graph.node_indices().take(5).enumerate() {
                let node = &output_graph.graph[idx];
                println!("  [{}] {:?}", i, node.node_type);
            }
            let predicted_code = output_graph.to_code();
            println!("Decoded code: {}", predicted_code.chars().take(80).collect::<String>());
            println!();
        }

        // Periodic checkpoint
        if (epoch + 1) % 20 == 0 {
            let checkpoint_path = args.output.with_file_name(format!("semantic_code_epoch{}.json", epoch + 1));
            mesh.save(&checkpoint_path)?;
        }
    }

    // Save final model
    mesh.save(&args.output)?;
    println!("\nFinal model saved to {:?}", args.output);
    println!("Best validation loss: {:.4}", best_loss);

    Ok(())
}
