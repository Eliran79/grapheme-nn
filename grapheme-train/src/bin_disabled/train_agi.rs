//! Unified AGI Training Binary
//!
//! Trains on mixed multi-modal datasets (math, text, timeseries, vision)
//! using unified graph-to-graph transformation learning.
//!
//! Auto-discovers external datasets (MATH, SQuAD, GSM8K) when present.
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin train_agi -- \
//!     --data data/mixed_agi --epochs 20 --batch-size 64 --lr 0.001
//!
//!   # With external datasets auto-discovery:
//!   cargo run --release -p grapheme-train --bin train_agi -- \
//!     --data data/mixed_agi --external data/external --epochs 20

use clap::Parser;
use grapheme_core::{BackwardPass, DagNN, Embedding, InitStrategy, NodeId, UnifiedCheckpoint};
use ndarray::Array1;
use grapheme_train::datasets::{Gsm8kLoader, MathLoader, SquadLoader};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_agi")]
#[command(about = "Train GRAPHEME on mixed AGI datasets")]
struct Args {
    /// Data directory containing train.jsonl, val.jsonl, test.jsonl
    #[arg(short, long, default_value = "data/mixed_agi")]
    data: PathBuf,

    /// External datasets directory (auto-discovers MATH, SQuAD, GSM8K)
    #[arg(long)]
    external: Option<PathBuf>,

    /// Output directory for checkpoints
    #[arg(short, long, default_value = "checkpoints/agi")]
    output: PathBuf,

    /// Number of epochs
    #[arg(short, long, default_value = "20")]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value = "64")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// Validation frequency (epochs)
    #[arg(long, default_value = "2")]
    val_freq: usize,

    /// Early stopping patience
    #[arg(long, default_value = "5")]
    patience: usize,

    /// Maximum examples per external dataset (0 = unlimited)
    #[arg(long, default_value = "1000")]
    max_external: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Mixed AGI example format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MixedExample {
    id: String,
    domain: String,
    input_type: String,
    input: serde_json::Value,
    expected_output: String,
    metadata: serde_json::Value,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("==========================================");
    println!(" GRAPHEME Unified AGI Training");
    println!("==========================================\n");

    println!("Configuration:");
    println!("  Data: {:?}", args.data);
    if let Some(ref ext) = args.external {
        println!("  External: {:?}", ext);
    }
    println!("  Output: {:?}", args.output);
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.lr);
    println!("  Val frequency: {} epochs", args.val_freq);
    println!("  Patience: {}", args.patience);
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Auto-discover and load external datasets if specified
    if let Some(ref external_dir) = args.external {
        println!("Discovering external datasets in {:?}...", external_dir);
        let discovered = discover_external_datasets(external_dir, args.max_external)?;
        if !discovered.is_empty() {
            // Merge into training data
            let merged_path = args.data.join("train.jsonl");
            merge_external_datasets(&merged_path, &discovered)?;
        }
    }

    // Load datasets
    let train_path = args.data.join("train.jsonl");
    let val_path = args.data.join("val.jsonl");

    println!("Loading training data from {:?}...", train_path);
    let train_examples = load_jsonl(&train_path)?;
    println!("  Loaded {} training examples", train_examples.len());

    let val_examples = if val_path.exists() {
        println!("Loading validation data from {:?}...", val_path);
        let examples = load_jsonl(&val_path)?;
        println!("  Loaded {} validation examples", examples.len());
        Some(examples)
    } else {
        None
    };

    // Count by domain
    let mut domain_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for ex in &train_examples {
        *domain_counts.entry(ex.domain.clone()).or_insert(0) += 1;
    }
    println!("\nDomain distribution:");
    for (domain, count) in &domain_counts {
        println!("  {}: {} examples", domain, count);
    }

    // Initialize model - create from text to get proper structure
    println!("\nInitializing DagNN model...");
    let mut dag = DagNN::from_text("init").unwrap_or_else(|_| DagNN::new());

    // Training loop
    let mut best_val_loss = f32::MAX;
    let mut epochs_without_improvement = 0;
    let mut total_steps = 0;

    let start_time = Instant::now();

    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        // Shuffle examples (simple Fisher-Yates with deterministic seed)
        let mut examples: Vec<_> = train_examples.clone();
        let seed = epoch as u64 * 12345;
        shuffle(&mut examples, seed);

        // Process batches
        for batch in examples.chunks(args.batch_size) {
            let batch_loss = train_batch(&mut dag, batch, args.lr);
            epoch_loss += batch_loss;
            batch_count += 1;
            total_steps += 1;

            if args.verbose && batch_count % 10 == 0 {
                println!(
                    "    Batch {}: loss={:.4}",
                    batch_count,
                    batch_loss / batch.len() as f32
                );
            }
        }

        let avg_loss = epoch_loss / train_examples.len() as f32;
        let epoch_time = epoch_start.elapsed();

        // Validation
        let val_info = if epoch % args.val_freq == 0 {
            if let Some(ref val) = val_examples {
                let val_loss = validate(&dag, val);

                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    epochs_without_improvement = 0;

                    // Save best checkpoint using UnifiedCheckpoint
                    let checkpoint_path = args.output.join("checkpoint_best.json");
                    save_unified_checkpoint(&dag, &checkpoint_path)?;
                } else {
                    epochs_without_improvement += 1;
                }

                format!(", val_loss={:.4}", val_loss)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        println!(
            "Epoch {}/{}: loss={:.4}{}, time={:.2}s",
            epoch + 1,
            args.epochs,
            avg_loss,
            val_info,
            epoch_time.as_secs_f64()
        );

        // Early stopping
        if epochs_without_improvement >= args.patience {
            println!("\nEarly stopping after {} epochs without improvement", args.patience);
            break;
        }

        // Periodic checkpoint
        if (epoch + 1) % 5 == 0 {
            let checkpoint_path = args.output.join(format!("checkpoint_epoch{}.json", epoch + 1));
            save_unified_checkpoint(&dag, &checkpoint_path)?;
            println!("  Saved checkpoint: {:?}", checkpoint_path);
        }
    }

    // Save final checkpoint
    let final_path = args.output.join("checkpoint_final.json");
    save_unified_checkpoint(&dag, &final_path)?;

    let total_time = start_time.elapsed();

    println!("\n==========================================");
    println!(" Training Complete");
    println!("==========================================");
    println!("  Total epochs: {}", args.epochs.min(args.patience + epochs_without_improvement));
    println!("  Total steps: {}", total_steps);
    println!("  Best validation loss: {:.4}", best_val_loss);
    println!("  Total time: {:.2}s", total_time.as_secs_f64());
    println!("  Final checkpoint: {:?}", final_path);

    Ok(())
}

fn load_jsonl(path: &PathBuf) -> anyhow::Result<Vec<MixedExample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut examples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let example: MixedExample = serde_json::from_str(&line)?;
        examples.push(example);
    }

    Ok(examples)
}

fn shuffle(examples: &mut [MixedExample], seed: u64) {
    let mut state = seed.wrapping_add(1);
    for i in (1..examples.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state as usize) % (i + 1);
        examples.swap(i, j);
    }
}

fn train_batch(
    dag: &mut DagNN,
    batch: &[MixedExample],
    lr: f32,
) -> f32 {
    let mut batch_loss = 0.0;

    // Create embedding for gradient accumulation
    #[allow(deprecated)]
    let mut embedding = Embedding::new(256, 64, InitStrategy::DynamicXavier);  // vocab_size=256, embed_dim=64 (GRAPHEME protocol)

    // Zero gradients at start of batch
    dag.zero_grad();
    embedding.zero_grad();

    for example in batch {
        // Convert input to graph embedding based on domain
        let input_embedding = example_to_embedding(example);

        // Forward pass using forward_with_inputs
        let input_nodes = dag.input_nodes();
        let mut input_map: HashMap<NodeId, f32> = HashMap::new();
        for (i, &node) in input_nodes.iter().enumerate() {
            if i < input_embedding.len() {
                input_map.insert(node, input_embedding[i]);
            }
        }
        let _ = dag.forward_with_inputs(&input_map);

        // Get output activations
        let output = dag.get_output_activations(32);

        // Convert expected output to embedding
        let target_embedding = output_to_embedding(&example.expected_output, example.domain.as_str());

        // Compute loss (MSE)
        let loss = compute_embedding_loss(&output, &target_embedding);
        batch_loss += loss;

        // Compute output gradients (dL/d_output = 2 * (output - target) / n)
        let output_nodes = dag.output_nodes();
        let mut output_grad: HashMap<NodeId, Array1<f32>> = HashMap::new();
        for (i, &node) in output_nodes.iter().enumerate() {
            if i < output.len() && i < target_embedding.len() {
                let grad = 2.0 * (output[i] - target_embedding[i]) / output.len() as f32;
                output_grad.insert(node, Array1::from_vec(vec![grad]));
            }
        }

        // Backward pass: accumulate gradients (proper backprop through graph structure)
        dag.backward_accumulate(&output_grad, &mut embedding);
    }

    // Apply accumulated gradients with learning rate (scaled by batch size)
    let effective_lr = lr / batch.len() as f32;
    dag.step(effective_lr);

    batch_loss
}

fn validate(dag: &DagNN, examples: &[MixedExample]) -> f32 {
    let mut total_loss = 0.0;

    for example in examples {
        let input_embedding = example_to_embedding(example);

        let mut dag_copy = dag.clone();
        let input_nodes = dag_copy.input_nodes();
        let mut input_map: HashMap<NodeId, f32> = HashMap::new();
        for (i, &node) in input_nodes.iter().enumerate() {
            if i < input_embedding.len() {
                input_map.insert(node, input_embedding[i]);
            }
        }
        let _ = dag_copy.forward_with_inputs(&input_map);
        let output = dag_copy.get_output_activations(32);

        let target_embedding = output_to_embedding(&example.expected_output, example.domain.as_str());
        total_loss += compute_embedding_loss(&output, &target_embedding);
    }

    total_loss / examples.len() as f32
}

fn example_to_embedding(example: &MixedExample) -> Vec<f32> {
    let mut embedding = vec![0.0; 64];

    match example.input_type.as_str() {
        "text" => {
            if let Some(text) = example.input.get("text").and_then(|v| v.as_str()) {
                // Character-level embedding
                for (i, ch) in text.chars().take(60).enumerate() {
                    embedding[i] = (ch as u32 % 256) as f32 / 255.0;
                }
            }
        }
        "sequence" => {
            if let Some(values) = example.input.get("values").and_then(|v| v.as_array()) {
                for (i, v) in values.iter().take(60).enumerate() {
                    if let Some(f) = v.as_f64() {
                        embedding[i] = f as f32 / 100.0; // Normalize
                    }
                }
            }
        }
        "image" => {
            if let Some(pixels) = example.input.get("pixels").and_then(|v| v.as_array()) {
                for (i, p) in pixels.iter().take(60).enumerate() {
                    if let Some(f) = p.as_f64() {
                        embedding[i] = f as f32;
                    }
                }
            }
        }
        _ => {}
    }

    // Add domain indicator
    match example.domain.as_str() {
        "math" => embedding[60] = 1.0,
        "text" => embedding[61] = 1.0,
        "timeseries" => embedding[62] = 1.0,
        "vision" => embedding[63] = 1.0,
        _ => {}
    }

    embedding
}

fn output_to_embedding(output: &str, domain: &str) -> Vec<f32> {
    let mut embedding = vec![0.0; 32];

    match domain {
        "math" => {
            // Parse numeric result
            if let Ok(num) = output.parse::<f32>() {
                embedding[0] = num.signum();
                embedding[1] = num.abs().log10().max(-10.0).min(10.0) / 10.0;
                for (i, digit) in output.chars().filter(|c| c.is_ascii_digit()).take(10).enumerate() {
                    embedding[2 + i] = (digit as u32 - '0' as u32) as f32 / 9.0;
                }
            }
        }
        "text" => {
            // Character embedding of answer
            for (i, ch) in output.chars().take(16).enumerate() {
                embedding[i] = (ch as u32 % 256) as f32 / 255.0;
            }
        }
        "timeseries" => {
            // Numeric prediction
            if let Ok(num) = output.parse::<f32>() {
                embedding[0] = num / 100.0;
            }
        }
        "vision" => {
            // Class label embedding
            let labels = ["vertical", "horizontal", "cross", "square", "diagonal", "dot", "corner", "T"];
            for (i, label) in labels.iter().enumerate() {
                if output == *label {
                    embedding[i] = 1.0;
                }
            }
        }
        _ => {}
    }

    embedding
}

fn compute_embedding_loss(output: &[f32], target: &[f32]) -> f32 {
    // MSE loss
    let min_len = output.len().min(target.len());
    let mut loss = 0.0;
    for i in 0..min_len {
        let diff = output[i] - target[i];
        loss += diff * diff;
    }
    loss / min_len as f32
}

/// Discovered external dataset info
#[allow(dead_code)]
struct DiscoveredDataset {
    name: String,
    examples: Vec<MixedExample>,
}

/// Auto-discover external datasets (MATH, SQuAD, GSM8K) in the given directory
fn discover_external_datasets(
    external_dir: &PathBuf,
    max_per_dataset: usize,
) -> anyhow::Result<Vec<DiscoveredDataset>> {
    let mut discovered = Vec::new();

    // Check for GSM8K dataset
    let gsm8k_dir = external_dir.join("gsm8k");
    if gsm8k_dir.exists() {
        println!("  Found GSM8K dataset at {:?}", gsm8k_dir);
        match load_gsm8k(&gsm8k_dir, max_per_dataset) {
            Ok(examples) => {
                println!("    Loaded {} examples", examples.len());
                discovered.push(DiscoveredDataset {
                    name: "gsm8k".to_string(),
                    examples,
                });
            }
            Err(e) => println!("    Warning: Failed to load GSM8K: {}", e),
        }
    }

    // Check for MATH competition dataset
    let math_dir = external_dir.join("math");
    if math_dir.exists() {
        println!("  Found MATH dataset at {:?}", math_dir);
        match load_math(&math_dir, max_per_dataset) {
            Ok(examples) => {
                println!("    Loaded {} examples", examples.len());
                discovered.push(DiscoveredDataset {
                    name: "math".to_string(),
                    examples,
                });
            }
            Err(e) => println!("    Warning: Failed to load MATH: {}", e),
        }
    }

    // Check for SQuAD dataset
    let squad_dir = external_dir.join("squad");
    if squad_dir.exists() {
        println!("  Found SQuAD dataset at {:?}", squad_dir);
        match load_squad(&squad_dir, max_per_dataset) {
            Ok(examples) => {
                println!("    Loaded {} examples", examples.len());
                discovered.push(DiscoveredDataset {
                    name: "squad".to_string(),
                    examples,
                });
            }
            Err(e) => println!("    Warning: Failed to load SQuAD: {}", e),
        }
    }

    if discovered.is_empty() {
        println!("  No external datasets found");
    } else {
        let total: usize = discovered.iter().map(|d| d.examples.len()).sum();
        println!("  Total: {} examples from {} datasets", total, discovered.len());
    }

    Ok(discovered)
}

/// Load GSM8K dataset and convert to MixedExample format
fn load_gsm8k(dir: &PathBuf, max_examples: usize) -> anyhow::Result<Vec<MixedExample>> {
    let loader = Gsm8kLoader::new(dir);
    let raw_examples = loader.load("train")
        .map_err(|e| anyhow::anyhow!("GSM8K load error: {}", e))?;

    let limit = if max_examples == 0 { raw_examples.len() } else { max_examples.min(raw_examples.len()) };

    Ok(raw_examples.into_iter().take(limit).map(|ex| {
        MixedExample {
            id: ex.id,
            domain: "math".to_string(),
            input_type: "text".to_string(),
            input: serde_json::json!({ "text": ex.question }),
            expected_output: ex.final_answer,
            metadata: serde_json::json!({
                "source": "gsm8k",
                "reasoning_steps": ex.reasoning_steps.len()
            }),
        }
    }).collect())
}

/// Load MATH competition dataset and convert to MixedExample format
fn load_math(dir: &PathBuf, max_examples: usize) -> anyhow::Result<Vec<MixedExample>> {
    let loader = MathLoader::new(dir);
    let raw_examples = loader.load("train")
        .map_err(|e| anyhow::anyhow!("MATH load error: {}", e))?;

    let limit = if max_examples == 0 { raw_examples.len() } else { max_examples.min(raw_examples.len()) };

    Ok(raw_examples.into_iter().take(limit).map(|ex| {
        MixedExample {
            id: ex.id,
            domain: "math".to_string(),
            input_type: "text".to_string(),
            input: serde_json::json!({ "text": ex.problem }),
            expected_output: ex.final_answer,
            metadata: serde_json::json!({
                "source": "math_competition",
                "category": ex.category,
                "level": ex.level,
                "latex": ex.latex_content
            }),
        }
    }).collect())
}

/// Load SQuAD dataset and convert to MixedExample format
fn load_squad(dir: &PathBuf, max_examples: usize) -> anyhow::Result<Vec<MixedExample>> {
    let loader = SquadLoader::new(dir);
    let raw_examples = loader.load("train")
        .map_err(|e| anyhow::anyhow!("SQuAD load error: {}", e))?;

    // Filter to answerable questions only
    let answerable: Vec<_> = raw_examples.into_iter()
        .filter(|ex| !ex.is_impossible && !ex.answer.is_empty())
        .collect();

    let limit = if max_examples == 0 { answerable.len() } else { max_examples.min(answerable.len()) };

    Ok(answerable.into_iter().take(limit).map(|ex| {
        // Truncate context for embedding
        let context = if ex.context.len() > 500 {
            format!("{}...", &ex.context[..500])
        } else {
            ex.context.clone()
        };

        MixedExample {
            id: ex.id,
            domain: "text".to_string(),
            input_type: "text".to_string(),
            input: serde_json::json!({
                "text": format!("Context: {}\n\nQuestion: {}", context, ex.question)
            }),
            expected_output: ex.answer,
            metadata: serde_json::json!({
                "source": "squad",
                "title": ex.title,
                "context_length": ex.context.len()
            }),
        }
    }).collect())
}

/// Merge discovered datasets into existing training file
fn merge_external_datasets(
    train_path: &PathBuf,
    discovered: &[DiscoveredDataset],
) -> anyhow::Result<()> {
    // Load existing examples
    let mut all_examples = if train_path.exists() {
        load_jsonl(train_path)?
    } else {
        Vec::new()
    };

    let original_count = all_examples.len();

    // Add discovered examples
    for dataset in discovered {
        all_examples.extend(dataset.examples.clone());
    }

    // Write back
    let file = File::create(train_path)?;
    let mut writer = BufWriter::new(file);
    for example in &all_examples {
        writeln!(writer, "{}", serde_json::to_string(example)?)?;
    }
    writer.flush()?;

    println!(
        "  Merged {} external examples into {} (total: {})",
        all_examples.len() - original_count,
        train_path.display(),
        all_examples.len()
    );

    Ok(())
}

/// Save DagNN using UnifiedCheckpoint for consistency with other training binaries
fn save_unified_checkpoint(dag: &DagNN, path: &PathBuf) -> anyhow::Result<()> {
    let mut checkpoint = UnifiedCheckpoint::new();
    checkpoint.add_module(dag)?;
    checkpoint.save_to_file(path)?;
    Ok(())
}

/// Load DagNN from UnifiedCheckpoint
#[allow(dead_code)]
fn load_unified_checkpoint(path: &PathBuf) -> anyhow::Result<DagNN> {
    let checkpoint = UnifiedCheckpoint::load_from_file(path)?;
    let dag: DagNN = checkpoint.load_module()?;
    Ok(dag)
}
