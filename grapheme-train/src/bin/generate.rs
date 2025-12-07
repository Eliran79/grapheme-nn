//! Training data generation CLI for GRAPHEME.
//!
//! Generates curriculum-based training data using the math engine.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate")]
#[command(about = "Generate GRAPHEME training data", long_about = None)]
struct Args {
    /// Generate all curriculum levels (1-7)
    #[arg(long)]
    all_levels: bool,

    /// Specific level to generate (1-7)
    #[arg(short, long, value_parser = clap::value_parser!(u8).range(1..=7))]
    level: Option<u8>,

    /// Number of samples to generate per level
    #[arg(short, long, default_value = "10000")]
    samples: usize,

    /// Output directory for generated data
    #[arg(short, long, default_value = "data/generated")]
    output: PathBuf,

    /// Output format
    #[arg(short, long, default_value = "jsonl")]
    format: String,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.verbose {
        println!("GRAPHEME Training Data Generator");
        println!("================================");
    }

    if !args.all_levels && args.level.is_none() {
        anyhow::bail!("Must specify either --all-levels or --level <N>");
    }

    // TODO: Implement actual data generation using grapheme_train::DataGenerator
    println!("Output directory: {:?}", args.output);
    println!("Format: {}", args.format);

    if args.all_levels {
        println!("Generating all levels (1-7), {} samples each", args.samples);
    } else if let Some(level) = args.level {
        println!("Generating level {}, {} samples", level, args.samples);
    }

    println!("\n[TODO] Data generation not yet implemented");
    println!("This binary provides the CLI interface for backend-086");

    Ok(())
}
