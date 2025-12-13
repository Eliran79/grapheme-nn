//! Training data generation CLI for GRAPHEME.
//!
//! Generates curriculum-based training data using the math engine.

use clap::Parser;
use grapheme_train::{CurriculumLevel, DataGenerator, Dataset};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

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

    /// Random seed for reproducible generation
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Split into train/val/test (ratios: 0.8/0.1/0.1)
    #[arg(long)]
    split: bool,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn level_to_curriculum(level: u8) -> CurriculumLevel {
    match level {
        1 => CurriculumLevel::BasicArithmetic,
        2 => CurriculumLevel::NestedOperations,
        3 => CurriculumLevel::SymbolSubstitution,
        4 => CurriculumLevel::BasicFunctions,
        5 => CurriculumLevel::Differentiation,
        6 => CurriculumLevel::Integration,
        7 => CurriculumLevel::EquationSolving,
        _ => CurriculumLevel::BasicArithmetic,
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("GRAPHEME Training Data Generator");
    println!("================================");

    if !args.all_levels && args.level.is_none() {
        anyhow::bail!("Must specify either --all-levels or --level <N>");
    }

    // Create output directory
    fs::create_dir_all(&args.output)?;

    // Initialize generator
    let mut generator = DataGenerator::new(args.seed);

    // Determine which levels to generate
    let levels: Vec<u8> = if args.all_levels {
        (1..=7).collect()
    } else {
        vec![args.level.unwrap()]
    };

    println!("Output directory: {:?}", args.output);
    println!("Seed: {}", args.seed);
    println!("Samples per level: {}", args.samples);
    println!("Levels to generate: {:?}", levels);
    println!();

    let total_start = Instant::now();

    for level in &levels {
        let start = Instant::now();
        let curriculum = level_to_curriculum(*level);

        if args.verbose {
            println!("Generating Level {} ({:?})...", level, curriculum);
        }

        generator.reset_stats();
        let examples = generator.generate_level(curriculum, args.samples);
        let stats = generator.stats();

        let elapsed = start.elapsed();
        println!(
            "  Level {}: {} examples in {:.2}s ({:.0} ex/s)",
            level,
            examples.len(),
            elapsed.as_secs_f64(),
            examples.len() as f64 / elapsed.as_secs_f64()
        );

        if args.verbose {
            println!(
                "    Stats: {} attempted, {} generated, {} dropped ({:.1}% success)",
                stats.attempted,
                stats.generated,
                stats.dropped_eval_error,
                stats.success_rate()
            );
        }

        // Create dataset
        let dataset = Dataset::from_examples(&format!("level_{}", level), examples);

        if args.split {
            // Split into train/val/test (80/10/10)
            let (train, val, test) = dataset.split(0.8, 0.1);

            let train_path = args.output.join(format!("level_{}_train.jsonl", level));
            let val_path = args.output.join(format!("level_{}_val.jsonl", level));
            let test_path = args.output.join(format!("level_{}_test.jsonl", level));

            train.save_jsonl(&train_path)?;
            val.save_jsonl(&val_path)?;
            test.save_jsonl(&test_path)?;

            if args.verbose {
                println!(
                    "    Saved: train={}, val={}, test={}",
                    train.len(),
                    val.len(),
                    test.len()
                );
            }
        } else {
            let path = args.output.join(format!("level_{}.jsonl", level));
            dataset.save_jsonl(&path)?;
            if args.verbose {
                println!("    Saved to: {:?}", path);
            }
        }
    }

    let total_elapsed = total_start.elapsed();
    println!();
    println!(
        "Generation complete: {} levels in {:.2}s",
        levels.len(),
        total_elapsed.as_secs_f64()
    );

    Ok(())
}
