//! Training execution CLI for GRAPHEME.
//!
//! Runs the training loop with curriculum learning.

use clap::Parser;
use grapheme_train::ConfigFile;
use std::path::PathBuf;

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
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.verbose {
        println!("GRAPHEME Training");
        println!("=================");
    }

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
    println!("  Curriculum: levels {}-{}", config.curriculum.start_level, config.curriculum.end_level);

    if let Some(resume) = &args.resume {
        println!("  Resuming from: {:?}", resume);
    }

    // Convert to internal training config
    let training_config = config.to_training_config();

    if args.verbose {
        println!("\nInternal config: {:?}", training_config);
    }

    println!("\n[TODO] Training not yet implemented");
    println!("This binary provides the CLI interface for backend-087");

    Ok(())
}
