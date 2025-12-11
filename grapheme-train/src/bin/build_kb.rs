//! Build Knowledge Base from Training Data
//!
//! Reads training data JSONL files and creates a GraphKnowledgeBase.
//!
//! Usage:
//!   build_kb --data data/generated --output checkpoints/math_kb.json --levels 1,2,3

use clap::Parser;
use grapheme_train::{GraphKnowledgeBase, KnowledgeEntry};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "build_kb")]
#[command(about = "Build knowledge base from training data")]
struct Args {
    /// Path to data directory
    #[arg(short, long)]
    data: PathBuf,

    /// Output knowledge base file
    #[arg(short, long)]
    output: PathBuf,

    /// Levels to include (comma-separated, e.g., "1,2,3")
    #[arg(short, long, default_value = "1,2,3,4,5,6,7")]
    levels: String,

    /// Maximum entries per level
    #[arg(long, default_value = "1000")]
    max_per_level: usize,

    /// Include validation set
    #[arg(long)]
    include_val: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Parse a JSONL line into Q/A pair
fn parse_jsonl_line(line: &str) -> Option<(String, String, String)> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;

    let id = v.get("id")?.as_str()?.to_string();
    let input_polish = v.get("input_polish")?.as_str()?.to_string();
    let expected_result = v.get("expected_result")?.as_f64()?;

    // Format answer based on whether it's an integer
    let answer = if expected_result.fract() == 0.0 {
        format!("{}", expected_result as i64)
    } else {
        format!("{:.4}", expected_result)
    };

    Some((id, input_polish, answer))
}

/// Get level topic name
fn level_topic(level: u8) -> &'static str {
    match level {
        1 => "arithmetic",
        2 => "fractions",
        3 => "algebra",
        4 => "functions",
        5 => "calculus",
        6 => "trigonometry",
        7 => "equations",
        _ => "math",
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Building Knowledge Base");
    println!("=======================\n");

    // Parse levels
    let levels: Vec<u8> = args.levels
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("Levels: {:?}", levels);
    println!("Data: {:?}", args.data);
    println!("Max per level: {}", args.max_per_level);

    let mut kb = GraphKnowledgeBase::new();
    let mut level_counts: HashMap<u8, usize> = HashMap::new();

    for level in &levels {
        let train_file = args.data.join(format!("level_{}_train.jsonl", level));
        let val_file = args.data.join(format!("level_{}_val.jsonl", level));

        let mut files = vec![];
        if train_file.exists() {
            files.push(train_file);
        }
        if args.include_val && val_file.exists() {
            files.push(val_file);
        }

        let mut count = 0;
        for file_path in files {
            if args.verbose {
                println!("Reading: {:?}", file_path);
            }

            let file = File::open(&file_path)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                if count >= args.max_per_level {
                    break;
                }

                let line = line?;
                if let Some((id, question, answer)) = parse_jsonl_line(&line) {
                    let topic = level_topic(*level);

                    let entry = KnowledgeEntry::new(
                        format!("{}_{}", topic, id),
                        topic,
                        &question,
                        &answer,
                    )
                    .with_confidence(0.0) // Will be updated during training
                    .with_epoch(*level as usize);

                    kb.add(entry);
                    count += 1;
                }
            }
        }

        level_counts.insert(*level, count);
        println!("Level {}: {} entries", level, count);
    }

    // Save knowledge base
    println!("\nSaving to: {:?}", args.output);

    // Create parent directory if needed
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }

    kb.save(&args.output)?;

    // Print stats
    let stats = kb.stats();
    println!("\nKnowledge Base Statistics:");
    println!("  Total entries: {}", stats.total_entries);
    println!("  Average question nodes: {:.1}", stats.avg_question_nodes);
    println!("  Average answer nodes: {:.1}", stats.avg_answer_nodes);
    println!("\n  By topic:");
    for (topic, count) in &stats.entries_by_topic {
        println!("    - {}: {}", topic, count);
    }

    println!("\nDone! Use with:");
    println!("  cargo run -p grapheme-train --bin query -- --model checkpoints/checkpoint_final.json --kb {:?}", args.output);

    Ok(())
}
