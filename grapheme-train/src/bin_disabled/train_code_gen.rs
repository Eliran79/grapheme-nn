//! Train Unified Code Generation with EncoderDecoder + Cortex Mesh
//!
//! This uses the EncoderDecoder architecture for TRUE sequence generation:
//! - Encoder: Understands the prompt/docstring
//! - Decoder: Generates the code character-by-character
//! - CodeBrain: Validates syntax and language patterns
//! - KnowledgeBase: Retrieval augmentation for similar problems
//!
//! Architecture:
//! ```
//!     ┌─────────────────────────────────────────────────┐
//!     │              CORTEX MESH                         │
//!     │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
//!     │  │ Encoder  │→ │ Decoder  │→ │ CodeBrain    │  │
//!     │  │ (Graph)  │  │ (Seq)    │  │ (Validate)   │  │
//!     │  └──────────┘  └──────────┘  └──────────────┘  │
//!     │        ↑                            ↓          │
//!     │  ┌──────────────────────────────────────────┐  │
//!     │  │         Knowledge Base (RAG)             │  │
//!     │  └──────────────────────────────────────────┘  │
//!     └─────────────────────────────────────────────────┘
//! ```
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin train_code_gen -- \
//!     --data data/code_training --output checkpoints/code_gen.json

use clap::Parser;
use grapheme_code::CodeBrain;
use grapheme_core::{DomainBrain, EncoderDecoder, GraphemeGraph, UnifiedCheckpoint};
use grapheme_train::GraphKnowledgeBase;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_code_gen")]
#[command(about = "Train unified code generation with EncoderDecoder + Cortex Mesh")]
struct Args {
    /// Path to code training data directory
    #[arg(short, long)]
    data: PathBuf,

    /// Output path for trained model
    #[arg(short, long)]
    output: PathBuf,

    /// Path to knowledge base for retrieval augmentation
    #[arg(short, long)]
    kb: Option<PathBuf>,

    /// Number of training epochs
    #[arg(short, long, default_value = "50")]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value = "4")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// Maximum code length to generate
    #[arg(long, default_value = "512")]
    max_len: usize,

    /// Resume from checkpoint
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Training example from JSONL
#[derive(Debug, Deserialize, Serialize)]
struct CodeExample {
    id: String,
    input: String,   // Prompt with docstring
    target: String,  // Solution code
    #[serde(default)]
    level: u32,
}

/// Load code examples from JSONL
fn load_examples(path: &PathBuf) -> anyhow::Result<Vec<CodeExample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut examples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let example: CodeExample = serde_json::from_str(&line)?;
        examples.push(example);
    }

    Ok(examples)
}

/// Compute character-level accuracy between predicted and target
fn char_accuracy(predicted: &str, target: &str) -> f32 {
    let pred_chars: Vec<char> = predicted.chars().collect();
    let tgt_chars: Vec<char> = target.chars().collect();

    let min_len = pred_chars.len().min(tgt_chars.len());
    let max_len = pred_chars.len().max(tgt_chars.len());

    if max_len == 0 {
        return 1.0;
    }

    let matches = pred_chars.iter()
        .zip(tgt_chars.iter())
        .filter(|(p, t)| p == t)
        .count();

    matches as f32 / max_len as f32
}

/// Compute cross-entropy loss between embeddings and target text (for validation only)
fn compute_loss(predicted: &Array2<f32>, target: &str, vocab_size: usize) -> f32 {
    let target_chars: Vec<usize> = target.chars()
        .map(|c| (c as usize).min(vocab_size - 1))
        .collect();

    let mut total_loss = 0.0;
    let seq_len = predicted.nrows().min(target_chars.len());

    for i in 0..seq_len {
        let target_idx = target_chars[i];
        // Softmax loss: -log(p[target])
        let logits = predicted.row(i);
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_softmax = logits[target_idx] - max_logit - exp_sum.ln();
        total_loss -= log_softmax;
    }

    if seq_len > 0 {
        total_loss / seq_len as f32
    } else {
        0.0
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Unified Code Generation Trainer");
    println!("================================");
    println!("Using EncoderDecoder + Cortex Mesh Architecture\n");

    // Model architecture
    const VOCAB_SIZE: usize = 256;  // ASCII
    const EMBED_DIM: usize = 128;
    const HIDDEN_DIM: usize = 256;
    const NUM_ENCODER_LAYERS: usize = 3;

    // Load training data
    let train_path = args.data.join("code_train.jsonl");
    let val_path = args.data.join("code_val.jsonl");

    println!("Loading training data from {:?}", train_path);
    let train_examples = load_examples(&train_path)?;
    println!("Loaded {} training examples", train_examples.len());

    let val_examples = if val_path.exists() {
        let examples = load_examples(&val_path)?;
        println!("Loaded {} validation examples", examples.len());
        Some(examples)
    } else {
        None
    };

    // Load knowledge base for RAG
    let kb = if let Some(kb_path) = &args.kb {
        println!("Loading knowledge base from {:?}", kb_path);
        let kb = GraphKnowledgeBase::load(kb_path)?;
        let stats = kb.stats();
        println!("Loaded {} KB entries", stats.total_entries);
        Some(kb)
    } else {
        None
    };

    // Initialize CodeBrain for validation
    let code_brain = CodeBrain::new();
    println!("CodeBrain initialized for syntax validation");

    // Create output directory
    let output_dir = args.output.parent().unwrap_or(&args.output);
    std::fs::create_dir_all(output_dir)?;

    // Initialize or resume model
    let mut model = if let Some(resume_path) = &args.resume {
        println!("\nResuming from {:?}", resume_path);
        let checkpoint = UnifiedCheckpoint::load_from_file(resume_path)?;
        checkpoint.load_module()?
    } else {
        println!("\nInitializing EncoderDecoder model...");
        println!("  Vocab: {}", VOCAB_SIZE);
        println!("  Embed dim: {}", EMBED_DIM);
        println!("  Hidden dim: {}", HIDDEN_DIM);
        println!("  Encoder layers: {}", NUM_ENCODER_LAYERS);
        println!("  Max output len: {}", args.max_len);
        EncoderDecoder::new(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            args.max_len,
            NUM_ENCODER_LAYERS,
        )
    };

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.lr);
    if kb.is_some() {
        println!("  RAG: Enabled");
    }

    // Training loop
    let start = Instant::now();
    let mut best_val_acc = 0.0f32;

    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut epoch_acc = 0.0f32;
        let mut batch_count = 0;

        // Shuffle training data
        let mut indices: Vec<usize> = (0..train_examples.len()).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        for batch_start in (0..indices.len()).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];

            let mut batch_loss = 0.0f32;
            let mut batch_acc = 0.0f32;

            // Zero gradients at start of batch
            model.zero_grad();

            for &idx in batch_indices {
                let example = &train_examples[idx];
                let input_graph = GraphemeGraph::from_text(&example.input);

                // Training step: forward + backward + accumulate gradients
                let loss = model.train_step(&input_graph, &example.target, args.lr / batch_indices.len() as f32);
                batch_loss += loss;

                // Get prediction for accuracy calculation
                let (_, predicted) = model.forward(&input_graph);
                let acc = char_accuracy(&predicted, &example.target);
                batch_acc += acc;

                // Validate with CodeBrain (gives us syntax feedback)
                if args.verbose && batch_count % 20 == 0 {
                    let lang = code_brain.detect_language(&predicted);
                    let valid = code_brain.can_process(&predicted);
                    println!("    Sample: {:?} lang, valid={}", lang, valid);
                }
            }

            let n = batch_indices.len() as f32;
            batch_loss /= n;
            batch_acc /= n;

            epoch_loss += batch_loss;
            epoch_acc += batch_acc;
            batch_count += 1;

            if args.verbose && batch_count % 10 == 0 {
                println!("    Batch {}: loss={:.4}, char_acc={:.1}%",
                    batch_count, batch_loss, batch_acc * 100.0);
            }
        }

        let avg_loss = epoch_loss / batch_count as f32;
        let avg_acc = epoch_acc / batch_count as f32;

        // Validation
        let val_result = if let Some(ref val) = val_examples {
            let mut val_loss = 0.0f32;
            let mut val_acc = 0.0f32;
            let mut exact_matches = 0;

            for example in val {
                let (embeddings, predicted) = model.forward(&GraphemeGraph::from_text(&example.input));
                val_loss += compute_loss(&embeddings, &example.target, VOCAB_SIZE);

                let acc = char_accuracy(&predicted, &example.target);
                val_acc += acc;

                // Exact match (pass@1 proxy)
                if predicted.trim() == example.target.trim() {
                    exact_matches += 1;
                }
            }

            let n = val.len() as f32;
            val_loss /= n;
            val_acc /= n;
            let exact_rate = exact_matches as f32 / val.len() as f32;

            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                // Save best model
                let best_path = args.output.with_file_name("code_gen_best.json");
                let mut checkpoint = UnifiedCheckpoint::new();
                checkpoint.add_module(&model)?;
                checkpoint.save_to_file(&best_path)?;
                println!("  [NEW BEST] val_acc={:.1}%, exact={:.1}%, saved", val_acc * 100.0, exact_rate * 100.0);
            }

            Some((val_loss, val_acc, exact_rate))
        } else {
            None
        };

        let epoch_time = epoch_start.elapsed();

        // Print epoch summary
        print!("Epoch {}/{}: train_loss={:.4}, train_acc={:.1}%",
            epoch + 1, args.epochs, avg_loss, avg_acc * 100.0);

        if let Some((vl, va, er)) = val_result {
            print!(", val_loss={:.4}, val_acc={:.1}%, exact={:.1}%", vl, va * 100.0, er * 100.0);
        }

        println!(", time={:.1}s", epoch_time.as_secs_f64());

        // Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 {
            let ckpt_path = args.output.with_file_name(format!("code_gen_epoch{}.json", epoch + 1));
            let mut checkpoint = UnifiedCheckpoint::new();
            checkpoint.add_module(&model)?;
            checkpoint.save_to_file(&ckpt_path)?;
            println!("  Checkpoint saved: {:?}", ckpt_path);
        }
    }

    let total_time = start.elapsed();

    // Save final model
    let mut final_checkpoint = UnifiedCheckpoint::new();
    final_checkpoint.add_module(&model)?;
    final_checkpoint.save_to_file(&args.output)?;

    println!("\nTraining complete in {:.1}s", total_time.as_secs_f64());
    println!("Final model saved to: {:?}", args.output);
    println!("Best validation accuracy: {:.1}%", best_val_acc * 100.0);

    // Demo inference
    println!("\n--- Demo Inference ---");
    let demo_prompt = r#"def add_two_numbers(a: int, b: int) -> int:
    """ Add two integers and return the result.
    >>> add_two_numbers(2, 3)
    5
    """
"#;
    println!("Prompt:\n{}", demo_prompt);
    let (_, generated) = model.forward(&GraphemeGraph::from_text(demo_prompt));
    println!("Generated:\n{}", generated);

    Ok(())
}
