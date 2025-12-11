//! Import External Datasets for GRAPHEME AGI Training
//!
//! Converts GSM8K, SQuAD, and other datasets into GRAPHEME knowledge base format.
//!
//! Usage:
//!   import_datasets --gsm8k data/external/gsm8k --output checkpoints/real_knowledge.json
//!   import_datasets --squad data/external/squad --output checkpoints/qa_knowledge.json

use clap::Parser;
use grapheme_train::{GraphKnowledgeBase, KnowledgeEntry};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "import_datasets")]
#[command(about = "Import external datasets into GRAPHEME knowledge base")]
struct Args {
    /// Path to GSM8K directory
    #[arg(long)]
    gsm8k: Option<PathBuf>,

    /// Path to SQuAD directory
    #[arg(long)]
    squad: Option<PathBuf>,

    /// Output knowledge base file
    #[arg(short, long)]
    output: PathBuf,

    /// Maximum entries per dataset
    #[arg(long, default_value = "5000")]
    max_entries: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

// GSM8K format
#[derive(Deserialize)]
struct Gsm8kEntry {
    question: String,
    answer: String,
}

// SQuAD format
#[derive(Deserialize)]
struct SquadData {
    data: Vec<SquadArticle>,
}

#[derive(Deserialize)]
struct SquadArticle {
    title: String,
    paragraphs: Vec<SquadParagraph>,
}

#[derive(Deserialize)]
struct SquadParagraph {
    context: String,
    qas: Vec<SquadQA>,
}

#[derive(Deserialize)]
struct SquadQA {
    question: String,
    answers: Vec<SquadAnswer>,
    is_impossible: Option<bool>,
}

#[derive(Deserialize)]
struct SquadAnswer {
    text: String,
}

/// Extract final answer from GSM8K format (after ####)
fn extract_gsm8k_answer(answer: &str) -> String {
    if let Some(idx) = answer.rfind("####") {
        answer[idx + 4..].trim().to_string()
    } else {
        // Try to find the last number
        answer.split_whitespace()
            .filter(|s| s.parse::<f64>().is_ok() || s.chars().all(|c| c.is_numeric() || c == '.' || c == '-'))
            .last()
            .unwrap_or("unknown")
            .to_string()
    }
}

/// Import GSM8K dataset
fn import_gsm8k(path: &PathBuf, kb: &mut GraphKnowledgeBase, max: usize, verbose: bool) -> usize {
    let train_path = path.join("train.jsonl");
    if !train_path.exists() {
        println!("GSM8K train.jsonl not found at {:?}", train_path);
        return 0;
    }

    let file = File::open(&train_path).expect("Failed to open GSM8K");
    let reader = BufReader::new(file);
    let mut count = 0;

    for (i, line) in reader.lines().enumerate() {
        if count >= max {
            break;
        }

        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        let entry: Gsm8kEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let answer = extract_gsm8k_answer(&entry.answer);

        if verbose && count < 3 {
            println!("  GSM8K #{}: Q: {}...", i, &entry.question[..entry.question.len().min(50)]);
            println!("          A: {}", answer);
        }

        let kb_entry = KnowledgeEntry::new(
            format!("gsm8k_{}", i),
            "math_word_problem",
            &entry.question,
            &answer,
        )
        .with_confidence(0.9)
        .with_epoch(1);

        kb.add(kb_entry);
        count += 1;
    }

    println!("Imported {} GSM8K entries", count);
    count
}

/// Import SQuAD dataset
fn import_squad(path: &PathBuf, kb: &mut GraphKnowledgeBase, max: usize, verbose: bool) -> usize {
    let train_path = path.join("train-v2.0.json");
    if !train_path.exists() {
        println!("SQuAD train-v2.0.json not found at {:?}", train_path);
        return 0;
    }

    let file = File::open(&train_path).expect("Failed to open SQuAD");
    let data: SquadData = serde_json::from_reader(file).expect("Failed to parse SQuAD");

    let mut count = 0;
    let mut qa_id = 0;

    'outer: for article in &data.data {
        for para in &article.paragraphs {
            for qa in &para.qas {
                if count >= max {
                    break 'outer;
                }

                // Skip impossible questions
                if qa.is_impossible.unwrap_or(false) || qa.answers.is_empty() {
                    continue;
                }

                let answer = &qa.answers[0].text;

                if verbose && count < 3 {
                    println!("  SQuAD #{}: Q: {}...", qa_id, &qa.question[..qa.question.len().min(50)]);
                    println!("          A: {}", answer);
                }

                // Create shorter context-based question
                let question = format!("{}", qa.question);

                let kb_entry = KnowledgeEntry::new(
                    format!("squad_{}", qa_id),
                    "reading_comprehension",
                    &question,
                    answer,
                )
                .with_confidence(0.95)
                .with_epoch(1);

                kb.add(kb_entry);
                count += 1;
                qa_id += 1;
            }
        }
    }

    println!("Imported {} SQuAD entries", count);
    count
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("GRAPHEME Dataset Importer");
    println!("=========================\n");

    let mut kb = GraphKnowledgeBase::new();
    let mut total = 0;

    // Import GSM8K
    if let Some(ref gsm8k_path) = args.gsm8k {
        println!("Importing GSM8K from {:?}...", gsm8k_path);
        total += import_gsm8k(gsm8k_path, &mut kb, args.max_entries, args.verbose);
    }

    // Import SQuAD
    if let Some(ref squad_path) = args.squad {
        println!("\nImporting SQuAD from {:?}...", squad_path);
        total += import_squad(squad_path, &mut kb, args.max_entries, args.verbose);
    }

    if total == 0 {
        println!("No datasets imported. Use --gsm8k or --squad to specify data paths.");
        return Ok(());
    }

    // Save knowledge base
    println!("\nSaving to {:?}...", args.output);
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    kb.save(&args.output)?;

    // Print stats
    let stats = kb.stats();
    println!("\nKnowledge Base Statistics:");
    println!("  Total entries: {}", stats.total_entries);
    println!("  Avg question nodes: {:.1}", stats.avg_question_nodes);
    println!("  Avg answer nodes: {:.1}", stats.avg_answer_nodes);
    println!("\n  By topic:");
    for (topic, count) in &stats.entries_by_topic {
        println!("    - {}: {}", topic, count);
    }

    println!("\nDone! Use with:");
    println!("  cargo run -p grapheme-train --bin agi_infer -- \\");
    println!("    --model checkpoints/checkpoint_level1_final.json \\");
    println!("    --kb {:?}", args.output);

    Ok(())
}
