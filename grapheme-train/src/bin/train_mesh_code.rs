//! Train CortexMesh + EncoderDecoder for Code Generation
//!
//! This combines:
//! - CortexMesh: All 8 domain brains for understanding problems
//! - EncoderDecoder: Proper sequence generation with cross-entropy loss
//!
//! Architecture:
//! ```
//!     Input (docstring) → CortexMesh (brain activation)
//!                              ↓
//!                    Brain-enhanced features
//!                              ↓
//!                    EncoderDecoder (seq2seq)
//!                              ↓
//!                    Output (code text)
//! ```

use anyhow::Result;
use clap::Parser;
use grapheme_core::{EncoderDecoder, GraphemeGraph, DomainBrain};
use grapheme_train::cortex_mesh::{CortexMesh, MeshConfig};
use ndarray::Array2;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_mesh_code")]
#[command(about = "Train CortexMesh + EncoderDecoder for code generation")]
struct Args {
    /// Path to training data directory
    #[arg(short, long)]
    data: PathBuf,

    /// Output checkpoint path
    #[arg(short, long)]
    output: PathBuf,

    /// Number of epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value = "8")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// Max output length
    #[arg(long, default_value = "512")]
    max_len: usize,

    /// Resume from checkpoint
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Training example
#[derive(Debug, Deserialize)]
struct CodeExample {
    id: String,
    input: String,
    target: String,
    #[serde(default)]
    level: u32,
}

/// Load examples from JSONL
fn load_examples(path: &PathBuf) -> Result<Vec<CodeExample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut examples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(ex) = serde_json::from_str::<CodeExample>(&line) {
            examples.push(ex);
        }
    }
    Ok(examples)
}

/// Character-level accuracy
fn char_accuracy(pred: &str, target: &str) -> f32 {
    let p: Vec<char> = pred.chars().collect();
    let t: Vec<char> = target.chars().collect();
    let max_len = p.len().max(t.len());
    if max_len == 0 {
        return 1.0;
    }
    let matches = p.iter().zip(t.iter()).filter(|(a, b)| a == b).count();
    matches as f32 / max_len as f32
}

/// Exact match check (trimmed)
fn exact_match(pred: &str, target: &str) -> bool {
    pred.trim() == target.trim()
}

/// Mesh-Enhanced EncoderDecoder
struct MeshCodeGen {
    mesh: CortexMesh,
    encoder_decoder: EncoderDecoder,
}

impl MeshCodeGen {
    fn new(max_len: usize) -> Self {
        // Create CortexMesh with all brains
        let config = MeshConfig {
            activation_threshold: 0.2,
            max_active_brains: usize::MAX,
            parallel: true,
            hidden_dim: 256,
            num_layers: 6,
            vocab_size: 256,
            embed_dim: 64,
        };
        let mesh = CortexMesh::discover_with_config(config);

        // Create EncoderDecoder for sequence generation
        let encoder_decoder = EncoderDecoder::new(
            256,  // vocab_size (ASCII)
            128,  // embed_dim
            256,  // hidden_dim
            max_len,
            3,    // num_layers
        );

        Self { mesh, encoder_decoder }
    }

    /// Train step with cross-entropy loss
    fn train_step(&mut self, input: &str, target: &str, lr: f32) -> f32 {
        // 1. Process through CortexMesh to get brain activations
        let mesh_result = self.mesh.process_parallel(input);

        // 2. Get active brain features (enhance the input)
        let enhanced_input = self.enhance_with_brains(input, &mesh_result.active_brains);

        // 3. Train EncoderDecoder with enhanced input
        let input_graph = GraphemeGraph::from_text(&enhanced_input);
        let loss = self.encoder_decoder.train_step(&input_graph, target, lr);

        loss
    }

    /// Enhance input with brain-specific preprocessing
    fn enhance_with_brains(&self, input: &str, active_brains: &[String]) -> String {
        // For now, prepend active brain tags to help the model understand context
        let brain_tags: Vec<&str> = active_brains.iter()
            .map(|b| match b.as_str() {
                "code" => "<code>",
                "math" => "<math>",
                "text" => "<text>",
                _ => "",
            })
            .filter(|s| !s.is_empty())
            .collect();

        if brain_tags.is_empty() {
            input.to_string()
        } else {
            format!("{} {}", brain_tags.join(" "), input)
        }
    }

    /// Generate code from input
    fn generate(&mut self, input: &str) -> String {
        let mesh_result = self.mesh.process_parallel(input);
        let enhanced_input = self.enhance_with_brains(input, &mesh_result.active_brains);
        let input_graph = GraphemeGraph::from_text(&enhanced_input);
        let (_, output) = self.encoder_decoder.forward(&input_graph);
        output
    }

    fn zero_grad(&mut self) {
        self.encoder_decoder.zero_grad();
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     CortexMesh + EncoderDecoder Code Generation Training     ║");
    println!("║        8 Domain Brains + Cross-Entropy Seq2Seq Loss          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load training data
    let train_path = args.data.join("code_train.jsonl");
    let val_path = args.data.join("code_val.jsonl");

    println!("Loading training data from {:?}", train_path);
    let train_examples = load_examples(&train_path)?;
    println!("Loaded {} training examples", train_examples.len());

    let val_examples = if val_path.exists() {
        let ex = load_examples(&val_path)?;
        println!("Loaded {} validation examples", ex.len());
        Some(ex)
    } else {
        None
    };

    // Create output directory
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Initialize model
    println!("\nInitializing MeshCodeGen...");
    let mut model = MeshCodeGen::new(args.max_len);

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.lr);
    println!("  Max output length: {}", args.max_len);
    println!();

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

            model.zero_grad();

            for &idx in batch_indices {
                let example = &train_examples[idx];

                // Train step
                let loss = model.train_step(
                    &example.input,
                    &example.target,
                    args.lr / batch_indices.len() as f32
                );
                batch_loss += loss;

                // Calculate accuracy
                let predicted = model.generate(&example.input);
                let acc = char_accuracy(&predicted, &example.target);
                batch_acc += acc;
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
            let mut val_acc = 0.0f32;
            let mut exact_matches = 0;

            for example in val {
                let predicted = model.generate(&example.input);
                let acc = char_accuracy(&predicted, &example.target);
                val_acc += acc;

                if exact_match(&predicted, &example.target) {
                    exact_matches += 1;
                }
            }

            let n = val.len() as f32;
            val_acc /= n;
            let exact_rate = exact_matches as f32 / val.len() as f32;

            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                // Save best model
                let best_path = args.output.with_file_name("mesh_code_best.json");
                // Note: We'd need to implement checkpoint saving for MeshCodeGen
                println!("  [NEW BEST] val_acc={:.1}%, exact={:.1}%",
                    val_acc * 100.0, exact_rate * 100.0);
            }

            Some((val_acc, exact_rate))
        } else {
            None
        };

        let epoch_time = epoch_start.elapsed();

        // Print epoch summary
        print!("Epoch {}/{}: loss={:.4}, train_acc={:.1}%",
            epoch + 1, args.epochs, avg_loss, avg_acc * 100.0);

        if let Some((va, er)) = val_result {
            print!(", val_acc={:.1}%, exact={:.1}%", va * 100.0, er * 100.0);
        }

        println!(", time={:.1}s", epoch_time.as_secs_f64());

        // Early stopping check
        if avg_acc > 0.95 && val_result.map(|(_, er)| er > 0.90).unwrap_or(false) {
            println!("\nEarly stopping: reached high accuracy!");
            break;
        }
    }

    let total_time = start.elapsed();
    println!("\nTraining complete in {:.1}s", total_time.as_secs_f64());
    println!("Best validation accuracy: {:.1}%", best_val_acc * 100.0);

    // Demo inference
    println!("\n--- Demo Inference ---");
    let demo_prompt = r#"from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
"#;
    println!("Prompt:\n{}", demo_prompt);
    let generated = model.generate(demo_prompt);
    println!("Generated:\n{}", generated);

    Ok(())
}
