//! Generate Code Training Data from Knowledge Base
//!
//! Converts HumanEval/MBPP entries in the knowledge base to the JSONL training
//! format that the DAG-NN trainer expects.
//!
//! Usage:
//!   generate_code_training --kb checkpoints/cortex_mesh_kb.json --output data/code_training

use clap::Parser;
use grapheme_train::GraphKnowledgeBase;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate_code_training")]
#[command(about = "Generate code training data from knowledge base")]
struct Args {
    /// Path to knowledge base
    #[arg(short, long)]
    kb: PathBuf,

    /// Output directory for training data
    #[arg(short, long)]
    output: PathBuf,

    /// Maximum entries to convert
    #[arg(long, default_value = "1000")]
    max_entries: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Training example format for DAG-NN
#[derive(Debug, Serialize, Deserialize)]
struct TrainingExample {
    id: String,
    input: String,
    target: String,
    input_polish: Option<String>,
    expected_result: Option<f64>,
    level: u32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Code Training Data Generator");
    println!("============================\n");

    // Load knowledge base
    println!("Loading knowledge base from {:?}...", args.kb);
    let kb = GraphKnowledgeBase::load(&args.kb)?;
    let stats = kb.stats();
    println!("Loaded {} entries across {} topics\n", stats.total_entries, stats.entries_by_topic.len());

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Filter for code entries
    let code_entries: Vec<_> = kb.entries()
        .iter()
        .filter(|e| e.topic == "code_generation" || e.topic == "synthetic_code")
        .take(args.max_entries)
        .collect();

    println!("Found {} code entries", code_entries.len());

    // Split into train/val/test (80/10/10)
    let total = code_entries.len();
    let train_end = (total * 80) / 100;
    let val_end = train_end + (total * 10) / 100;

    let train_entries = &code_entries[..train_end];
    let val_entries = &code_entries[train_end..val_end];
    let test_entries = &code_entries[val_end..];

    println!("Split: {} train, {} val, {} test\n", train_entries.len(), val_entries.len(), test_entries.len());

    // Write training data
    let train_path = args.output.join("code_train.jsonl");
    let val_path = args.output.join("code_val.jsonl");
    let test_path = args.output.join("code_test.jsonl");

    write_examples(&train_path, train_entries, args.verbose)?;
    write_examples(&val_path, val_entries, args.verbose)?;
    write_examples(&test_path, test_entries, args.verbose)?;

    println!("\nTraining data written to:");
    println!("  Train: {:?}", train_path);
    println!("  Val: {:?}", val_path);
    println!("  Test: {:?}", test_path);

    println!("\nTo train:");
    println!("  cargo run --release -p grapheme-train --bin train_code -- \\");
    println!("    --data {:?} --output checkpoints/code_model.json", args.output);

    Ok(())
}

fn write_examples(
    path: &PathBuf,
    entries: &[&grapheme_train::KnowledgeEntry],
    verbose: bool,
) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for (i, entry) in entries.iter().enumerate() {
        let example = TrainingExample {
            id: format!("CODE-{:05}", i),
            input: entry.question.clone(),
            target: entry.answer.clone(),
            input_polish: None,
            expected_result: None,
            level: 8, // Code level
        };

        if verbose && i < 3 {
            println!("  Example {}: {} -> {}...",
                i,
                &entry.question[..entry.question.len().min(40)],
                &entry.answer[..entry.answer.len().min(30)]);
        }

        let json = serde_json::to_string(&example)?;
        writeln!(writer, "{}", json)?;
    }

    Ok(())
}
