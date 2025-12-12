//! CortexMesh Training - Full AGI Architecture with AUTO-DISCOVERY
//!
//! This trainer uses compile-time auto-discovery to mesh ALL cognitive components.
//! Just add new brains to BRAIN_FACTORIES in cortex_mesh.rs - they'll be used automatically.
//!
//! ## Auto-Discovered Components
//! - Domain Brains: Auto-discovered from BRAIN_FACTORIES
//! - Router Modules: Auto-discovered from MODULE_FACTORIES
//!
//! All components run in PARALLEL by default using Rayon.
//!
//! ## Parallel Processing
//! - Batch graph creation: parallel pre-computation
//! - Validation: parallel structural loss computation
//! - SemanticDecoder: unified vocabulary for semantic node generation

use anyhow::Result;
use clap::Parser;
use grapheme_core::GraphemeGraph;
use grapheme_train::cortex_mesh::{CortexMesh, MeshConfig, list_all_brains, list_all_modules};
use grapheme_train::semantic_decoder::{SemanticDecoder, SemanticDecoderConfig};
use grapheme_train::training_utils::{semantic_accuracy, prepare_decoder_batch};
use grapheme_train::{compute_structural_loss, StructuralLossConfig};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_cortex_mesh")]
#[command(about = "Train GRAPHEME with ALL cognitive components meshed in parallel (auto-discovered)")]
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

    /// Number of parallel workers (0 = auto)
    #[arg(short, long, default_value = "0")]
    workers: usize,

    /// Resume from checkpoint
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Hidden dimension for model
    #[arg(long, default_value = "256")]
    hidden_dim: usize,

    /// Number of transformer layers
    #[arg(long, default_value = "6")]
    num_layers: usize,

    /// Brain activation threshold (lower = more brains active)
    #[arg(long, default_value = "0.2")]
    activation_threshold: f32,

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
    #[allow(dead_code)]  // Reserved for future domain-specific routing
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
    for subdir in ["code", "math", "vision", "text", "law", "music", "chem", "time"] {
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

// NOTE: Character-level similarity REMOVED - semantic node accuracy is the proper metric

fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        GRAPHEME CortexMesh Training (AUTO-DISCOVERY)         ║");
    println!("║   Full AGI Architecture - ALL Components Meshed in Parallel  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Show auto-discovered components
    let brains = list_all_brains();
    let modules = list_all_modules();

    println!("Auto-Discovered Components:");
    println!("  Brains ({}): {:?}", brains.len(), brains);
    println!("  Modules ({}): {:?}", modules.len(), modules);
    println!();

    // Configure parallelism
    if args.workers > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.workers)
            .build_global()
            .unwrap();
        println!("Parallel workers: {}", args.workers);
    } else {
        println!("Parallel workers: {} (auto)", rayon::current_num_threads());
    }

    // Load training data
    println!("\nLoading training data from {:?}...", args.data);
    let samples = load_training_data(&args.data)?;
    println!("Loaded {} training samples", samples.len());

    if samples.is_empty() {
        println!("WARNING: No training samples found!");
        println!("Expected JSONL files with {{\"input\": ..., \"output\": ...}} format");
        return Ok(());
    }

    // Create mesh configuration
    let config = MeshConfig {
        activation_threshold: args.activation_threshold,
        max_active_brains: usize::MAX, // All brains
        parallel: true,
        hidden_dim: args.hidden_dim,
        num_layers: args.num_layers,
        vocab_size: 256,
        embed_dim: 64,
    };

    // Create or resume mesh using AUTO-DISCOVERY
    let mut mesh = if let Some(resume_path) = &args.resume {
        println!("\nResuming from {:?}...", resume_path);
        let mut mesh = CortexMesh::discover_with_config(config);
        mesh.load(resume_path)?;
        mesh
    } else {
        println!("\nInitializing CortexMesh with AUTO-DISCOVERY...");
        CortexMesh::discover_with_config(config)
    };

    println!("\nMesh Ready:");
    println!("  Brains: {}", mesh.brain_count());
    println!("  Modules: {}", mesh.module_count());
    println!("  Model: {} hidden dim, {} layers", mesh.model.hidden_dim, mesh.model.mp_layers.len());
    println!("  Parallel: {}", mesh.config.parallel);

    // Create SemanticDecoder with unified vocabulary from all brains
    println!("\nBuilding unified semantic vocabulary...");
    let vocab = SemanticDecoder::build_vocab_from_brains();
    let embed_dim = 64;  // Match mesh config
    let decoder_config = SemanticDecoderConfig {
        hidden_dim: embed_dim,
        learning_rate: args.lr,
        temperature: 1.0,
        label_smoothing: 0.1,
    };
    let mut decoder = SemanticDecoder::new(vocab, decoder_config);
    let vocab_stats = decoder.vocab_stats();
    println!("SemanticDecoder ready:");
    println!("  Vocabulary size: {}", decoder.vocab_size());
    println!("  Node types: {} Keywords, {} Ops, {} Puncts, {} Input chars",
        vocab_stats.by_type.get("Keyword").unwrap_or(&0),
        vocab_stats.by_type.get("Op").unwrap_or(&0),
        vocab_stats.by_type.get("Punct").unwrap_or(&0),
        vocab_stats.by_type.get("Input").unwrap_or(&0));

    // Training loop
    let mut best_loss = f32::MAX;
    let train_size = (samples.len() as f32 * 0.9) as usize;
    let (train_samples, val_samples) = samples.split_at(train_size);

    println!("\nTraining: {} samples, Validation: {} samples", train_samples.len(), val_samples.len());
    println!("Epochs: {}, Batch size: {}, LR: {}\n", args.epochs, args.batch_size, args.lr);

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

        // Process batches with PROPER BATCH TRAINING + SemanticDecoder
        // Critical: zero_grad → accumulate gradients → step (once per batch)
        let mut train_struct_loss = 0.0;
        let mut train_decoder_loss = 0.0;
        let loss_config = StructuralLossConfig::default();

        for batch in train_samples.chunks(args.batch_size) {
            // 1. Zero gradients at start of batch
            mesh.model.zero_grad();

            // PARALLEL: Pre-compute all graphs for the batch
            let graphs: Vec<(GraphemeGraph, GraphemeGraph)> = batch
                .par_iter()
                .map(|sample| {
                    let input_graph = GraphemeGraph::from_text(&sample.input);
                    let target_graph = GraphemeGraph::from_text(&sample.output);
                    (input_graph, target_graph)
                })
                .collect();

            // Collect decoder training batch across all samples
            let mut decoder_batch = Vec::new();

            // 2. Accumulate gradients across all samples in batch
            for (input_graph, target_graph) in &graphs {
                // Forward pass - get pooled features
                let (output_graph, pooling) = mesh.model.forward(input_graph);

                // Build decoder training batch from features and target
                let sample_batch = prepare_decoder_batch(&pooling.features, target_graph, &decoder);
                decoder_batch.extend(sample_batch);

                // Compute structural loss
                let loss_result = compute_structural_loss(&output_graph, target_graph, &loss_config);
                train_struct_loss += loss_result.total_loss;
                train_loss += loss_result.total_loss;

                // Backward pass for mesh model
                mesh.model.backward(input_graph, &pooling, &loss_result.activation_gradients, mesh.config.embed_dim);
            }

            // 3. Apply accumulated gradients ONCE per batch (with batch-averaged LR)
            mesh.model.step(current_lr / batch.len() as f32);

            // 4. Train SemanticDecoder on accumulated batch
            if !decoder_batch.is_empty() {
                let dec_loss = decoder.backward(&decoder_batch);
                train_decoder_loss += dec_loss * decoder_batch.len() as f32;
            }

            // Update stats
            mesh.stats.total_processed += batch.len();

            pb.inc(batch.len() as u64);
        }
        pb.finish_and_clear();

        let n = train_samples.len() as f32;
        train_loss /= n;
        train_struct_loss /= n;
        train_decoder_loss /= n;

        // Validation - PARALLEL: Process validation samples in parallel
        // Step 1: Parallel pre-compute all graphs
        let val_graphs: Vec<(GraphemeGraph, GraphemeGraph)> = val_samples
            .par_iter()
            .map(|sample| {
                let input_graph = GraphemeGraph::from_text(&sample.input);
                let target_graph = GraphemeGraph::from_text(&sample.output);
                (input_graph, target_graph)
            })
            .collect();

        // Step 2: Parallel forward pass and metrics computation
        // Note: mesh.model.forward() takes &self (immutable), so it's thread-safe
        let val_results: Vec<(f32, f32, f32)> = val_graphs
            .par_iter()
            .map(|(input_graph, target_graph)| {
                let (output_graph, pooling) = mesh.model.forward(input_graph);

                let loss_result = compute_structural_loss(&output_graph, target_graph, &loss_config);
                let loss = loss_result.total_loss;

                // Semantic accuracy on raw output (before decoder)
                let sem_acc = semantic_accuracy(&output_graph, target_graph);

                // Decoder accuracy
                let dec_batch = prepare_decoder_batch(&pooling.features, target_graph, &decoder);
                let dec_acc = if !dec_batch.is_empty() {
                    decoder.compute_accuracy(&dec_batch)
                } else {
                    0.0
                };

                (loss, sem_acc, dec_acc)
            })
            .collect();

        // Aggregate validation metrics
        let val_count = val_results.len() as f32;
        let val_loss: f32 = val_results.iter().map(|(l, _, _)| l).sum::<f32>() / val_count;
        let val_semantic_acc: f32 = val_results.iter().map(|(_, s, _)| s).sum::<f32>() / val_count;
        let val_decoder_acc: f32 = val_results.iter().map(|(_, _, d)| d).sum::<f32>() / val_count;

        let epoch_time = epoch_start.elapsed();

        // Save best model
        let is_best = val_loss < best_loss;
        if is_best {
            best_loss = val_loss;
            let best_path = args.output.with_file_name("cortex_mesh_best.json");
            mesh.save(&best_path)?;
        }

        // Print progress with semantic metrics
        print!("Epoch {}/{}: loss={:.4} (struct={:.2}, dec={:.2}), val={:.4}, sem_acc={:.1}%, dec_acc={:.1}%",
            epoch + 1, args.epochs, train_loss, train_struct_loss, train_decoder_loss,
            val_loss, val_semantic_acc * 100.0, val_decoder_acc * 100.0);

        if is_best {
            print!(" [BEST]");
        }

        // Show brain activation stats
        let active_count = mesh.stats.brain_activations.len();
        println!(", lr={:.6}, brains={}, time={:.1}s",
            current_lr, active_count, epoch_time.as_secs_f64());

        // Verbose stats
        if args.verbose >= 2 && (epoch + 1) % 10 == 0 {
            println!("\n{}", mesh.stats_summary());
        }

        // Periodic checkpoint
        if (epoch + 1) % 20 == 0 {
            let checkpoint_path = args.output.with_file_name(format!("cortex_mesh_epoch{}.json", epoch + 1));
            mesh.save(&checkpoint_path)?;
        }
    }

    // Save final model
    mesh.save(&args.output)?;
    println!("\nFinal model saved to {:?}", args.output);

    // Print final statistics
    println!("\n{}", mesh.stats_summary());

    // Summary of auto-discovered components used
    println!("Auto-Discovered Components Used:");
    println!("  Brains: {} / {}", mesh.stats.brain_activations.len(), brains.len());
    for (brain, count) in &mesh.stats.brain_activations {
        let pct = *count as f32 / mesh.stats.total_processed.max(1) as f32 * 100.0;
        println!("    {}: {} activations ({:.1}%)", brain, count, pct);
    }

    Ok(())
}
