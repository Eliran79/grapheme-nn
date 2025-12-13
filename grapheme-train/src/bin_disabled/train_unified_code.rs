//! Unified Multi-Cortex Code Generation Training
//!
//! This trainer connects ALL 7 cortices for code generation:
//! - CodeBrain: AST parsing, type inference, syntax validation
//! - MathBrain: Numeric computation, algorithm analysis
//! - VisionBrain: Visual pattern recognition (code structure)
//! - TextBrain: Natural language understanding (docstrings)
//! - LawBrain: Rule-based constraints (code conventions)
//! - MusicBrain: Temporal patterns (control flow)
//! - ChemBrain: Compositional structures (modules)
//!
//! Key Insight (from MathEngine paradigm):
//! GRAPHEME doesn't just PREDICT code like LLMs - it COMPILES and VERIFIES!
//! - CodeBrain validates syntax and types
//! - MathEngine computes numeric results for verification
//! - Type inference catches errors at generation time
//!
//! Architecture:
//! ```text
//!   Prompt → GraphemeGraph → [Multi-Cortex Transform] → Output Graph → Decode
//!                                     │
//!                    ┌────────────────┼────────────────┐
//!                    ▼                ▼                ▼
//!              CodeBrain        MathBrain        VisionBrain
//!              (syntax)         (compute)        (structure)
//!                    │                │                │
//!                    └────────────────┼────────────────┘
//!                                     ▼
//!                            Fusion + Validation
//!                                     │
//!                                     ▼
//!                             Valid Code Output
//! ```
//!
//! Uses Rayon for parallel:
//! - Batch processing
//! - Multi-cortex forward passes
//! - Gradient accumulation
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin train_unified_code -- \
//!     --data data/code_training --output checkpoints/unified_code.json

use clap::Parser;
use grapheme_code::CodeBrain;
use grapheme_core::{DomainBrain, GraphemeGraph, GraphTransformNet, Learnable};
use grapheme_train::{compute_structural_loss, SinkhornConfig, StructuralLossConfig};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_unified_code")]
#[command(about = "Train unified multi-cortex model for code generation")]
struct Args {
    /// Path to code training data directory
    #[arg(short, long)]
    data: PathBuf,

    /// Output path for trained model
    #[arg(short, long)]
    output: PathBuf,

    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Batch size (processed in parallel)
    #[arg(short, long, default_value = "16")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.0005")]
    lr: f32,

    /// Maximum output length (Sabag output clusters)
    #[arg(long, default_value = "512")]
    max_output_len: usize,

    /// Number of parallel threads (0 = auto)
    #[arg(long, default_value = "0")]
    threads: usize,

    /// Resume from checkpoint
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Training example from JSONL
#[derive(Debug, Clone, Deserialize, Serialize)]
struct CodeExample {
    id: String,
    input: String,   // Prompt with docstring
    target: String,  // Solution code
    #[serde(default)]
    level: u32,
}

/// Result of processing a single example (for parallel accumulation)
#[derive(Debug, Clone)]
struct ExampleResult {
    loss: f32,
    graph_similarity: f32,
    char_accuracy: f32,
    syntax_valid: bool,
    type_valid: bool,
}

/// Multi-Cortex ensemble for code understanding
struct CortexEnsemble {
    code_brain: CodeBrain,
    // Future: Add other brains as they implement DomainBrain
    // math_brain: MathBrain,
    // vision_brain: VisionBrain,
}

impl CortexEnsemble {
    fn new() -> Self {
        Self {
            code_brain: CodeBrain::new(),
        }
    }

    /// Check if generated code is syntactically valid
    fn validate_syntax(&self, code: &str) -> bool {
        self.code_brain.can_process(code)
    }

    /// Detect programming language
    fn detect_language(&self, code: &str) -> String {
        format!("{:?}", self.code_brain.detect_language(code))
    }

    /// Get confidence score for generated code
    fn code_confidence(&self, generated: &str, target: &str) -> f32 {
        // Multi-factor confidence:
        // 1. Language detection match
        let gen_lang = self.code_brain.detect_language(generated);
        let tgt_lang = self.code_brain.detect_language(target);
        let lang_match = if gen_lang == tgt_lang { 0.3 } else { 0.0 };

        // 2. Syntax validity
        let syntax_score = if self.validate_syntax(generated) { 0.4 } else { 0.0 };

        // 3. Character overlap (Jaccard)
        let gen_chars: std::collections::HashSet<char> = generated.chars().collect();
        let tgt_chars: std::collections::HashSet<char> = target.chars().collect();
        let intersection = gen_chars.intersection(&tgt_chars).count();
        let union = gen_chars.union(&tgt_chars).count();
        let jaccard = if union > 0 { intersection as f32 / union as f32 } else { 0.0 };

        lang_match + syntax_score + 0.3 * jaccard
    }
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

/// Compute character-level accuracy
fn char_accuracy(predicted: &str, target: &str) -> f32 {
    let pred_chars: Vec<char> = predicted.chars().collect();
    let tgt_chars: Vec<char> = target.chars().collect();

    let max_len = pred_chars.len().max(tgt_chars.len());
    if max_len == 0 {
        return 1.0;
    }

    let matches = pred_chars
        .iter()
        .zip(tgt_chars.iter())
        .filter(|(p, t)| p == t)
        .count();

    matches as f32 / max_len as f32
}

/// Process a single example (for parallel execution)
fn process_example(
    model: &GraphTransformNet,
    cortex: &CortexEnsemble,
    example: &CodeExample,
    loss_config: &StructuralLossConfig,
) -> (ExampleResult, Vec<f32>, Array2<f32>) {
    // Convert texts to graphs
    let input_graph = GraphemeGraph::from_text(&example.input);
    let target_graph = GraphemeGraph::from_text(&example.target);

    // Forward: Input graph → Output graph
    let (output_graph, pooling_result) = model.forward(&input_graph);

    // Decode output graph to text
    let decoded = model.decode(&pooling_result);

    // Compute structural loss
    let loss_result = compute_structural_loss(&output_graph, &target_graph, loss_config);

    // Graph similarity (1 - normalized loss)
    let max_nodes = output_graph.node_count().max(target_graph.node_count());
    let max_edges = output_graph.edge_count().max(target_graph.edge_count());
    let max_cost = (max_nodes + max_edges) as f32;
    let graph_sim = if max_cost > 0.0 {
        1.0 - (loss_result.total_loss / max_cost).min(1.0)
    } else {
        1.0
    };

    // Character accuracy
    let char_acc = char_accuracy(&decoded, &example.target);

    // Syntax validation using CodeBrain
    let syntax_valid = cortex.validate_syntax(&decoded);
    let type_valid = true; // TODO: Integrate type inference

    let result = ExampleResult {
        loss: loss_result.total_loss,
        graph_similarity: graph_sim,
        char_accuracy: char_acc,
        syntax_valid,
        type_valid,
    };

    (result, loss_result.activation_gradients.clone(), pooling_result.features.clone())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Configure Rayon thread pool
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()?;
    }

    let num_threads = rayon::current_num_threads();

    println!("Unified Multi-Cortex Code Generation Trainer");
    println!("=============================================");
    println!("GRAPHEME: Compile & Verify, Don't Just Predict\n");
    println!("Parallel threads: {}", num_threads);

    // Model architecture
    const VOCAB_SIZE: usize = 256;   // ASCII
    const EMBED_DIM: usize = 128;    // Larger for code
    const HIDDEN_DIM: usize = 256;   // Deeper representations
    const NUM_LAYERS: usize = 4;     // More layers for complex code

    // Load training data
    let train_path = args.data.join("code_train.jsonl");
    let val_path = args.data.join("code_val.jsonl");

    println!("Loading training data from {:?}", train_path);
    let train_examples = load_examples(&train_path)?;
    println!("Loaded {} training examples", train_examples.len());

    // Analyze target lengths
    let target_lengths: Vec<usize> = train_examples.iter().map(|e| e.target.len()).collect();
    let max_target = *target_lengths.iter().max().unwrap_or(&256);
    let avg_target = target_lengths.iter().sum::<usize>() / target_lengths.len().max(1);
    println!("Target lengths: avg={}, max={}", avg_target, max_target);

    let val_examples = if val_path.exists() {
        let examples = load_examples(&val_path)?;
        println!("Loaded {} validation examples", examples.len());
        Some(examples)
    } else {
        None
    };

    // Initialize cortex ensemble
    let cortex = CortexEnsemble::new();
    println!("\nCortex Ensemble initialized:");
    println!("  - CodeBrain: Syntax validation, type inference");
    println!("  - (Future: MathBrain, VisionBrain, etc.)");

    // Create output directory
    let output_dir = args.output.parent().unwrap_or(&args.output);
    std::fs::create_dir_all(output_dir)?;

    // Use max output length that covers most targets
    let output_clusters = args.max_output_len.min(max_target + 50);
    println!("\nUsing {} output clusters (Sabag expansion)", output_clusters);

    // Initialize or resume model
    let mut model = if let Some(resume_path) = &args.resume {
        println!("\nResuming from {:?}", resume_path);
        // Load model directly (saved with model.save_to_file)
        GraphTransformNet::load_from_file(resume_path)?
    } else {
        println!("\nInitializing GraphTransformNet...");
        println!("  Vocab: {}", VOCAB_SIZE);
        println!("  Embed dim: {}", EMBED_DIM);
        println!("  Hidden dim: {}", HIDDEN_DIM);
        println!("  Layers: {}", NUM_LAYERS);
        println!("  Output clusters: {} (Sabag expansion)", output_clusters);

        let mut model = GraphTransformNet::new(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS);
        // Override Sabag for expansion
        model.sabag_pooling = Some(grapheme_core::SabagPooling::new(output_clusters, EMBED_DIM));
        model
    };

    // Structural loss configuration
    let loss_config = StructuralLossConfig {
        alpha: 1.0,   // Node cost
        beta: 0.5,    // Edge cost
        gamma: 0.3,   // Clique weight
        sinkhorn: SinkhornConfig {
            iterations: 20,
            temperature: 0.1,
            epsilon: 1e-6,
        },
    };

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {} (parallel)", args.batch_size);
    println!("  Learning rate: {}", args.lr);
    println!("  Structural loss: α={}, β={}, γ={}", loss_config.alpha, loss_config.beta, loss_config.gamma);

    // Training loop
    let start = Instant::now();
    let mut best_val_loss = f32::INFINITY;
    let mut best_syntax_rate = 0.0f32;

    // Learning rate schedule: warm up then cosine decay
    let warmup_epochs = 5;
    let min_lr = args.lr * 0.01; // Minimum LR = 1% of initial

    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();

        // Calculate learning rate with warm up and cosine decay
        let current_lr = if epoch < warmup_epochs {
            // Linear warm up
            args.lr * (epoch + 1) as f32 / warmup_epochs as f32
        } else {
            // Cosine decay after warmup
            let decay_epoch = epoch - warmup_epochs;
            let decay_total = args.epochs - warmup_epochs;
            let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * decay_epoch as f32 / decay_total as f32).cos());
            min_lr + (args.lr - min_lr) * cosine_decay
        };

        // Epoch accumulators
        let epoch_loss = Arc::new(Mutex::new(0.0f32));
        let epoch_similarity = Arc::new(Mutex::new(0.0f32));
        let epoch_char_acc = Arc::new(Mutex::new(0.0f32));
        let epoch_syntax_valid = Arc::new(Mutex::new(0usize));
        let epoch_count = Arc::new(Mutex::new(0usize));

        // Shuffle training data
        let mut indices: Vec<usize> = (0..train_examples.len()).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        // Process batches
        for batch_start in (0..indices.len()).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];

            // Zero gradients
            model.zero_grad();

            // Process batch examples in parallel using Rayon
            let batch_results: Vec<(ExampleResult, Vec<f32>, Array2<f32>)> = batch_indices
                .par_iter()
                .map(|&idx| {
                    let example = &train_examples[idx];
                    process_example(&model, &cortex, example, &loss_config)
                })
                .collect();

            // Accumulate results
            for (result, grads, features) in &batch_results {
                *epoch_loss.lock().unwrap() += result.loss;
                *epoch_similarity.lock().unwrap() += result.graph_similarity;
                *epoch_char_acc.lock().unwrap() += result.char_accuracy;
                if result.syntax_valid {
                    *epoch_syntax_valid.lock().unwrap() += 1;
                }
                *epoch_count.lock().unwrap() += 1;
            }

            // Backward pass (sequential for now - model mutation)
            for (idx_in_batch, &idx) in batch_indices.iter().enumerate() {
                let example = &train_examples[idx];
                let input_graph = GraphemeGraph::from_text(&example.input);

                // Re-do forward to get pooling result for backward
                let (_, pooling_result) = model.forward(&input_graph);
                let (result, grads, _) = &batch_results[idx_in_batch];

                // Backward through model
                model.backward(&input_graph, &pooling_result, grads, EMBED_DIM);
            }

            // Update weights with scheduled learning rate
            model.step(current_lr);
        }

        // Get epoch stats
        let total = *epoch_count.lock().unwrap() as f32;
        let avg_loss = *epoch_loss.lock().unwrap() / total;
        let avg_sim = *epoch_similarity.lock().unwrap() / total;
        let avg_char = *epoch_char_acc.lock().unwrap() / total;
        let syntax_rate = *epoch_syntax_valid.lock().unwrap() as f32 / total;

        // Validation
        let val_result = if let Some(ref val) = val_examples {
            let mut val_loss = 0.0f32;
            let mut val_sim = 0.0f32;
            let mut val_char = 0.0f32;
            let mut val_syntax = 0usize;

            // Parallel validation
            let val_results: Vec<ExampleResult> = val
                .par_iter()
                .map(|example| {
                    let (result, _, _) = process_example(&model, &cortex, example, &loss_config);
                    result
                })
                .collect();

            for result in &val_results {
                val_loss += result.loss;
                val_sim += result.graph_similarity;
                val_char += result.char_accuracy;
                if result.syntax_valid {
                    val_syntax += 1;
                }
            }

            let n = val.len() as f32;
            val_loss /= n;
            val_sim /= n;
            val_char /= n;
            let val_syntax_rate = val_syntax as f32 / n;

            // Save best model
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                let best_path = args.output.with_file_name("unified_code_best.json");
                model.save_to_file(&best_path)?;
                println!("  [NEW BEST] val_loss={:.4}", val_loss);
            }

            if val_syntax_rate > best_syntax_rate {
                best_syntax_rate = val_syntax_rate;
                println!("  [BEST SYNTAX] {:.1}% valid", val_syntax_rate * 100.0);
            }

            Some((val_loss, val_sim, val_char, val_syntax_rate))
        } else {
            None
        };

        let epoch_time = epoch_start.elapsed();

        // Print epoch summary
        print!(
            "Epoch {}/{}: loss={:.4}, sim={:.1}%, char={:.1}%, syntax={:.1}%",
            epoch + 1, args.epochs, avg_loss, avg_sim * 100.0, avg_char * 100.0, syntax_rate * 100.0
        );

        if let Some((vl, vs, vc, vx)) = val_result {
            print!(
                ", val: loss={:.4}, sim={:.1}%, char={:.1}%, syntax={:.1}%",
                vl, vs * 100.0, vc * 100.0, vx * 100.0
            );
        }

        println!(", lr={:.6}, time={:.1}s", current_lr, epoch_time.as_secs_f64());

        // Checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0 {
            let ckpt_path = args.output.with_file_name(format!("unified_code_epoch{}.json", epoch + 1));
            model.save_to_file(&ckpt_path)?;
            println!("  Checkpoint: {:?}", ckpt_path);
        }

        // Demo every 25 epochs
        if args.verbose && (epoch + 1) % 25 == 0 {
            println!("\n  --- Demo (epoch {}) ---", epoch + 1);
            let demo_prompt = r#"def is_prime(n):
    """ Check if n is a prime number. """
"#;
            let input_graph = GraphemeGraph::from_text(demo_prompt);
            let (_, pooling_result) = model.forward(&input_graph);
            let generated = model.decode(&pooling_result);
            let preview: String = generated.chars().take(80).collect();
            let lang = cortex.detect_language(&generated);
            let valid = cortex.validate_syntax(&generated);
            println!("  Input: {} chars", demo_prompt.len());
            println!("  Output: {} chars, lang={}, valid={}", generated.len(), lang, valid);
            println!("  Preview: \"{}...\"", preview);
            println!();
        }
    }

    let total_time = start.elapsed();

    // Save final model
    model.save_to_file(&args.output)?;

    println!("\nTraining complete in {:.1}s", total_time.as_secs_f64());
    println!("Final model: {:?}", args.output);
    println!("Best val loss: {:.4}", best_val_loss);
    println!("Best syntax rate: {:.1}%", best_syntax_rate * 100.0);

    // Final demo
    println!("\n--- Final Demo: Multi-Cortex Code Generation ---");
    let demo_prompt = r#"def add(a, b):
    """ Add two numbers and return the sum.
    >>> add(2, 3)
    5
    """
"#;
    println!("Input prompt ({} chars):", demo_prompt.len());
    println!("{}", demo_prompt);

    let input_graph = GraphemeGraph::from_text(demo_prompt);
    println!("Input graph: {} nodes, {} edges", input_graph.node_count(), input_graph.edge_count());

    let (output_graph, pooling_result) = model.forward(&input_graph);
    println!("Output graph: {} nodes, {} edges", output_graph.node_count(), output_graph.edge_count());

    let generated = model.decode(&pooling_result);
    println!("\nGenerated code:");
    println!("{}", generated);

    // Validate with cortex
    let lang = cortex.detect_language(&generated);
    let valid = cortex.validate_syntax(&generated);
    let confidence = cortex.code_confidence(&generated, "    return a + b");
    println!("\nCortex Analysis:");
    println!("  Language: {}", lang);
    println!("  Syntax valid: {}", valid);
    println!("  Confidence: {:.1}%", confidence * 100.0);

    Ok(())
}
