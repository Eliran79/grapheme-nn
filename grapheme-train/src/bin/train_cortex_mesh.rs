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

use anyhow::Result;
use clap::Parser;
use grapheme_core::GraphemeGraph;
use grapheme_train::cortex_mesh::{CortexMesh, MeshConfig, list_all_brains, list_all_modules};
use grapheme_train::{compute_structural_loss, StructuralLossConfig};
use indicatif::{ProgressBar, ProgressStyle};
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

/// Compute character-level similarity (Jaccard-like)
fn similarity(a: &str, b: &str) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let a_chars: std::collections::HashSet<char> = a.chars().collect();
    let b_chars: std::collections::HashSet<char> = b.chars().collect();

    let intersection = a_chars.intersection(&b_chars).count();
    let union = a_chars.union(&b_chars).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

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

        // Process batches
        for batch in train_samples.chunks(args.batch_size) {
            let batch_loss: f32 = batch.iter()
                .map(|sample| mesh.train_step(&sample.input, &sample.output, current_lr))
                .sum();

            train_loss += batch_loss;
            pb.inc(batch.len() as u64);
        }
        pb.finish_and_clear();

        train_loss /= train_samples.len() as f32;

        // Validation
        let mut val_loss = 0.0;
        let mut val_similarity = 0.0;
        let loss_config = StructuralLossConfig::default();

        for sample in val_samples {
            let result = mesh.process_parallel(&sample.input);
            let target_graph = GraphemeGraph::from_text(&sample.output);

            let loss_result = compute_structural_loss(&result.output_graph, &target_graph, &loss_config);
            val_loss += loss_result.total_loss;

            // Character similarity
            let sim = similarity(&result.decoded, &sample.output);
            val_similarity += sim;
        }

        val_loss /= val_samples.len() as f32;
        val_similarity /= val_samples.len() as f32;

        let epoch_time = epoch_start.elapsed();

        // Save best model
        let is_best = val_loss < best_loss;
        if is_best {
            best_loss = val_loss;
            let best_path = args.output.with_file_name("cortex_mesh_best.json");
            mesh.save(&best_path)?;
        }

        // Print progress
        print!("Epoch {}/{}: train_loss={:.4}, val_loss={:.4}, sim={:.1}%",
            epoch + 1, args.epochs, train_loss, val_loss, val_similarity * 100.0);

        if is_best {
            print!(" [BEST]");
        }

        // Show brain activation stats
        let active_count = mesh.stats.brain_activations.len();
        println!(", lr={:.6}, active_brains={}, time={:.1}s",
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
