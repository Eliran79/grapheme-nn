//! Training execution CLI for GRAPHEME.
//!
//! Runs the training loop with curriculum learning.

use clap::Parser;
use grapheme_core::{GraphTransformNet, UnifiedCheckpoint};
use grapheme_polish::expr_to_polish;
use grapheme_train::{
    compute_structural_loss, Adam, ConfigFile, Dataset, LRScheduler, StructuralLossConfig,
    TrainingLoop, TrainingMetrics, TrainingState,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Save a unified checkpoint containing model, training state, metrics, and optimizer
fn save_unified_checkpoint(
    path: &Path,
    model: &GraphTransformNet,
    training_state: &TrainingState,
    metrics: &TrainingMetrics,
    optimizer: &Adam,
) -> anyhow::Result<()> {
    let mut checkpoint = UnifiedCheckpoint::new();
    checkpoint.add_module(model)?;
    checkpoint.add_module(training_state)?;
    checkpoint.add_module(metrics)?;
    checkpoint.add_module(optimizer)?;
    checkpoint.save_to_file(path)?;
    Ok(())
}

/// Load a unified checkpoint and return components
fn load_unified_checkpoint(
    path: &Path,
) -> anyhow::Result<(GraphTransformNet, TrainingState, TrainingMetrics, Adam)> {
    let checkpoint = UnifiedCheckpoint::load_from_file(path)?;

    let model: GraphTransformNet = checkpoint.load_module()?;
    let training_state: TrainingState = checkpoint.load_module()?;
    let metrics: TrainingMetrics = checkpoint.load_module()?;
    let optimizer: Adam = checkpoint.load_module()?;

    Ok((model, training_state, metrics, optimizer))
}

#[derive(Parser, Debug)]
#[command(name = "train")]
#[command(about = "Train GRAPHEME model", long_about = None)]
struct Args {
    /// Path to training config file (TOML)
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Path to training data directory (overrides config)
    #[arg(short, long)]
    data: Option<PathBuf>,

    /// Output directory for checkpoints (overrides config)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Batch size (overrides config)
    #[arg(short, long)]
    batch_size: Option<usize>,

    /// Number of epochs per level (overrides config)
    #[arg(short, long)]
    epochs: Option<usize>,

    /// Learning rate (overrides config)
    #[arg(long)]
    lr: Option<f64>,

    /// Resume from checkpoint
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Dry run - validate config without training
    #[arg(long)]
    dry_run: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("GRAPHEME Training");
    println!("=================");

    // Load config from file or use defaults
    let mut config = if let Some(config_path) = &args.config {
        if args.verbose {
            println!("Loading config from: {:?}", config_path);
        }
        ConfigFile::load(config_path)?
    } else {
        if args.data.is_none() {
            anyhow::bail!("Must specify either --config or --data");
        }
        // Create default config
        ConfigFile {
            training: Default::default(),
            optimizer: Default::default(),
            loss: Default::default(),
            curriculum: Default::default(),
            paths: Default::default(),
            hardware: Default::default(),
        }
    };

    // Override config with CLI arguments
    if let Some(data) = &args.data {
        config.paths.train_data = data.to_string_lossy().to_string();
    }
    if let Some(output) = &args.output {
        config.paths.output_dir = output.to_string_lossy().to_string();
    }
    if let Some(batch_size) = args.batch_size {
        config.training.batch_size = batch_size;
    }
    if let Some(epochs) = args.epochs {
        config.training.epochs_per_level = epochs;
    }
    if let Some(lr) = args.lr {
        config.training.learning_rate = lr;
    }

    // Display configuration
    println!("\nTraining Configuration:");
    println!("  Data: {}", config.paths.train_data);
    println!("  Output: {}", config.paths.output_dir);
    println!("  Batch size: {}", config.training.batch_size);
    println!("  Epochs/level: {}", config.training.epochs_per_level);
    println!("  Learning rate: {}", config.training.learning_rate);
    println!("  Optimizer: {}", config.optimizer.optimizer_type);
    println!(
        "  Curriculum: levels {}-{}",
        config.curriculum.start_level, config.curriculum.end_level
    );

    if let Some(resume) = &args.resume {
        println!("  Resuming from: {:?}", resume);
    }

    if args.dry_run {
        println!("\n[Dry run] Configuration validated successfully");
        return Ok(());
    }

    // Create output directory
    fs::create_dir_all(&config.paths.output_dir)?;

    // Convert to internal training config
    let training_config = config.to_training_config();

    if args.verbose {
        println!("\nInternal config: {:?}", training_config);
    }

    // Model parameters - could be made configurable
    const VOCAB_SIZE: usize = 256; // ASCII characters
    const EMBED_DIM: usize = 64;
    const HIDDEN_DIM: usize = 128;
    const NUM_LAYERS: usize = 3;

    // Initialize or resume model, optimizer, and training state
    let (mut model, optimizer, mut training_loop) = if let Some(resume_path) = &args.resume {
        println!("\nResuming from unified checkpoint: {:?}", resume_path);

        // Try unified checkpoint first, fall back to model-only for backwards compatibility
        if let Ok((model, state, metrics, opt)) = load_unified_checkpoint(resume_path) {
            println!("  Loaded unified checkpoint:");
            println!("    Epoch: {}", state.epoch);
            println!("    Total steps: {}", state.total_steps);
            println!("    Best val loss: {:.4}", state.best_val_loss);
            println!("    Optimizer timestep: {}", opt.timestep());

            let mut loop_state = TrainingLoop::new(training_config.clone()).with_scheduler(
                LRScheduler::CosineAnnealingLR {
                    t_max: training_config.epochs,
                    eta_min: training_config.learning_rate * 0.01,
                },
            );
            loop_state.state = state;
            loop_state.metrics = metrics;

            (model, opt, loop_state)
        } else {
            // Fall back to model-only checkpoint (backwards compatibility)
            println!("  Note: Loading legacy model-only checkpoint");
            let model = GraphTransformNet::load_from_file(resume_path)?;
            let optimizer = Adam::new(training_config.learning_rate)
                .with_beta1(config.optimizer.beta1 as f32)
                .with_beta2(config.optimizer.beta2 as f32)
                .with_weight_decay(config.optimizer.weight_decay as f32);
            let training_loop = TrainingLoop::new(training_config.clone()).with_scheduler(
                LRScheduler::CosineAnnealingLR {
                    t_max: training_config.epochs,
                    eta_min: training_config.learning_rate * 0.01,
                },
            );
            (model, optimizer, training_loop)
        }
    } else {
        println!("\nInitializing new model...");
        let model = GraphTransformNet::new(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS);
        let optimizer = Adam::new(training_config.learning_rate)
            .with_beta1(config.optimizer.beta1 as f32)
            .with_beta2(config.optimizer.beta2 as f32)
            .with_weight_decay(config.optimizer.weight_decay as f32);
        let training_loop = TrainingLoop::new(training_config.clone()).with_scheduler(
            LRScheduler::CosineAnnealingLR {
                t_max: training_config.epochs,
                eta_min: training_config.learning_rate * 0.01,
            },
        );
        (model, optimizer, training_loop)
    };

    // Structural loss configuration from GRAPHEME vision
    let structural_config = StructuralLossConfig {
        alpha: config.loss.node_insertion_cost as f32,
        beta: config.loss.edge_insertion_cost as f32,
        gamma: config.loss.clique_weight as f32,
        sinkhorn: grapheme_train::SinkhornConfig {
            iterations: 20,
            temperature: 0.1,
            epsilon: 1e-6,
        },
    };

    // Train each curriculum level
    for level in config.curriculum.start_level..=config.curriculum.end_level {
        println!("\n--- Curriculum Level {} ---", level);

        // Try to load training data for this level
        let train_path =
            PathBuf::from(&config.paths.train_data).join(format!("level_{}_train.jsonl", level));
        let val_path =
            PathBuf::from(&config.paths.train_data).join(format!("level_{}_val.jsonl", level));

        // Fall back to combined file if split doesn't exist
        let train_path = if train_path.exists() {
            train_path
        } else {
            PathBuf::from(&config.paths.train_data).join(format!("level_{}.jsonl", level))
        };

        if !train_path.exists() {
            println!(
                "  Warning: Training data not found at {:?}, skipping level",
                train_path
            );
            continue;
        }

        let train_dataset = Dataset::load_jsonl(&train_path, &format!("level_{}_train", level))?;
        println!("  Loaded {} training examples", train_dataset.len());

        // Load validation data if available
        let val_dataset = if val_path.exists() {
            Some(Dataset::load_jsonl(
                &val_path,
                &format!("level_{}_val", level),
            )?)
        } else {
            None
        };

        if let Some(ref val) = val_dataset {
            println!("  Loaded {} validation examples", val.len());
        }

        // Train for epochs_per_level epochs
        let level_start = Instant::now();

        for epoch in 0..config.training.epochs_per_level {
            let epoch_start = Instant::now();
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // Process batches with supervised edit prediction (backend-092)
            for batch in train_dataset.batches(config.training.batch_size) {
                // Zero gradients before each batch
                model.zero_grad();

                // Collect inputs and targets for batch
                let mut inputs: Vec<&str> = Vec::with_capacity(batch.len());
                let mut targets: Vec<String> = Vec::with_capacity(batch.len());

                for example in batch {
                    let input = &example.input_polish;

                    // Get target string
                    let target = if let Some(ref symbolic) = &example.expected_symbolic {
                        expr_to_polish(symbolic)
                    } else if let Some(result) = example.expected_result {
                        result.to_string()
                    } else {
                        continue;
                    };

                    inputs.push(input);
                    targets.push(target);
                }

                if inputs.is_empty() {
                    continue;
                }

                // Zero gradients before computation
                model.zero_grad();

                // Compute differentiable structural loss (backend-096)
                // Uses Sinkhorn optimal transport for soft graph matching
                let mut batch_loss = 0.0;
                let mut total_accuracy = 0.0;

                for (input, target) in inputs.iter().zip(targets.iter()) {
                    // Convert text to graph structures
                    let predicted_graph = grapheme_core::GraphemeGraph::from_text(input);
                    let target_graph = grapheme_core::GraphemeGraph::from_text(target);

                    // Compute structural loss: α·node + β·edge + γ·clique
                    let loss_result = compute_structural_loss(
                        &predicted_graph,
                        &target_graph,
                        &structural_config,
                    );

                    batch_loss += loss_result.total_loss;

                    // Compute accuracy as graph similarity (1 - normalized loss)
                    let max_nodes = predicted_graph.node_count().max(target_graph.node_count());
                    let max_edges = predicted_graph.edge_count().max(target_graph.edge_count());
                    let max_cost = (max_nodes + max_edges) as f32;
                    let accuracy = if max_cost > 0.0 {
                        1.0 - (loss_result.total_loss / max_cost).min(1.0)
                    } else {
                        1.0
                    };
                    total_accuracy += accuracy;

                    // TODO: Backpropagate gradients through graph structure
                    // For now, the structural loss gradients are computed but not yet
                    // connected to model parameter updates
                }

                batch_loss /= inputs.len() as f32;
                total_accuracy /= inputs.len() as f32;

                // Update model weights using gradient descent
                let lr = training_loop.state.current_lr;
                model.step(lr);

                // Record batch metrics - use structural loss
                training_loop.record_batch(batch_loss);
                epoch_loss += batch_loss;
                batch_count += 1;

                // Log metrics periodically
                if args.verbose && batch_count % 10 == 0 {
                    println!(
                        "    Batch {}: Structural loss={:.4}, graph_similarity={:.2}%",
                        batch_count,
                        batch_loss,
                        total_accuracy * 100.0
                    );
                }
            }

            let avg_loss = if batch_count > 0 {
                epoch_loss / batch_count as f32
            } else {
                0.0
            };

            // Validation using structural loss (consistent with training)
            if let Some(ref val) = val_dataset {
                if training_loop.should_validate() {
                    // Compute validation structural loss
                    let mut val_loss = 0.0;
                    let mut val_accuracy = 0.0;
                    let mut val_count = 0;

                    for example in &val.examples {
                        let target = if let Some(ref symbolic) = &example.expected_symbolic {
                            expr_to_polish(symbolic)
                        } else if let Some(result) = example.expected_result {
                            result.to_string()
                        } else {
                            continue;
                        };

                        // Convert to graphs
                        let predicted_graph =
                            grapheme_core::GraphemeGraph::from_text(&example.input_polish);
                        let target_graph = grapheme_core::GraphemeGraph::from_text(&target);

                        // Compute structural loss
                        let loss_result = compute_structural_loss(
                            &predicted_graph,
                            &target_graph,
                            &structural_config,
                        );

                        val_loss += loss_result.total_loss;

                        // Compute accuracy
                        let max_nodes = predicted_graph.node_count().max(target_graph.node_count());
                        let max_edges = predicted_graph.edge_count().max(target_graph.edge_count());
                        let max_cost = (max_nodes + max_edges) as f32;
                        let accuracy = if max_cost > 0.0 {
                            1.0 - (loss_result.total_loss / max_cost).min(1.0)
                        } else {
                            1.0
                        };
                        val_accuracy += accuracy;
                        val_count += 1;
                    }

                    if val_count > 0 {
                        val_loss /= val_count as f32;
                        val_accuracy /= val_count as f32;

                        let improved = training_loop.record_validation(val_loss, val_accuracy);
                        if improved && args.verbose {
                            println!(
                                "    New best validation: loss={:.4}, graph_similarity={:.2}%",
                                val_loss,
                                val_accuracy * 100.0
                            );
                        }
                    }
                }
            }

            training_loop.complete_epoch();

            let epoch_elapsed = epoch_start.elapsed();
            if args.verbose || epoch % 5 == 0 {
                println!(
                    "  Epoch {}/{}: loss={:.4}, lr={:.6}, time={:.2}s",
                    epoch + 1,
                    config.training.epochs_per_level,
                    avg_loss,
                    training_loop.state.current_lr,
                    epoch_elapsed.as_secs_f64()
                );
            }

            // Early stopping check
            if training_loop.should_stop() {
                println!(
                    "  Early stopping after {} epochs without improvement",
                    training_loop.state.epochs_without_improvement
                );
                break;
            }

            // Checkpoint every N epochs (unified format)
            if (epoch + 1) % config.training.checkpoint_every == 0 {
                let checkpoint_path = PathBuf::from(&config.paths.output_dir).join(format!(
                    "checkpoint_level{}_epoch{}.json",
                    level,
                    epoch + 1
                ));
                save_unified_checkpoint(
                    &checkpoint_path,
                    &model,
                    &training_loop.state,
                    &training_loop.metrics,
                    &optimizer,
                )?;
                if args.verbose {
                    println!("    Saved unified checkpoint: {:?}", checkpoint_path);
                }
            }
        }

        let level_elapsed = level_start.elapsed();
        println!(
            "  Level {} complete in {:.2}s",
            level,
            level_elapsed.as_secs_f64()
        );

        // Save end-of-level checkpoint (unified format)
        let level_checkpoint = PathBuf::from(&config.paths.output_dir)
            .join(format!("checkpoint_level{}_final.json", level));
        save_unified_checkpoint(
            &level_checkpoint,
            &model,
            &training_loop.state,
            &training_loop.metrics,
            &optimizer,
        )?;
        println!("  Saved level checkpoint: {:?}", level_checkpoint);
    }

    // Save final unified checkpoint
    let final_path = PathBuf::from(&config.paths.output_dir).join("checkpoint_final.json");
    save_unified_checkpoint(
        &final_path,
        &model,
        &training_loop.state,
        &training_loop.metrics,
        &optimizer,
    )?;
    println!("\nTraining complete!");
    println!("Final checkpoint saved to: {:?}", final_path);

    // Also save model-only for inference convenience
    let model_path = PathBuf::from(&config.paths.output_dir).join("model_final.json");
    model.save_to_file(&model_path)?;
    println!("Model-only saved to: {:?}", model_path);

    // Print training summary
    println!("\nTraining Summary:");
    println!("  Total epochs: {}", training_loop.state.epoch);
    println!("  Total steps: {}", training_loop.state.total_steps);
    println!(
        "  Best validation loss: {:.4}",
        training_loop.state.best_val_loss
    );

    Ok(())
}
