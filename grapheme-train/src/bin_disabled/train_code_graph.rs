//! Train DAG-NN for Code Generation using Graph-to-Graph Transformation
//!
//! This follows the TRUE GRAPHEME vision:
//!   Text → Input Graph → [DAG-NN Transform] → Output Graph → Decode to Code
//!
//! Key insight: Use Sabag pooling with k=target_length for EXPANSION
//! - Input prompt has n characters → n nodes
//! - Target code has m characters → Need k=m output nodes
//! - Sabag EXPANDS from n → m (not compresses!)
//!
//! Training:
//! 1. Input prompt → GraphemeGraph (n nodes)
//! 2. Target code → GraphemeGraph (m nodes)
//! 3. Transform with GraphTransformNet (Sabag expands n → m)
//! 4. Structural loss between output graph and target graph
//! 5. Backprop through Sabag assignment matrix
//! 6. Decode output graph → text for evaluation
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin train_code_graph -- \
//!     --data data/code_training --output checkpoints/code_graph.json

use clap::Parser;
use grapheme_core::{GraphemeGraph, GraphTransformNet, Learnable, UnifiedCheckpoint};
use grapheme_train::{compute_structural_loss, StructuralLossConfig, SinkhornConfig};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_code_graph")]
#[command(about = "Train DAG-NN for code generation using graph-to-graph transformation")]
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

    /// Batch size
    #[arg(short, long, default_value = "8")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// Maximum output length (Sabag output clusters)
    #[arg(long, default_value = "512")]
    max_output_len: usize,

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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("DAG-NN Code Generation Trainer (Graph-to-Graph)");
    println!("================================================");
    println!("Text → Graph → Transform → Graph → Code\n");

    // Model architecture
    const VOCAB_SIZE: usize = 256;   // ASCII
    const EMBED_DIM: usize = 64;     // Embedding dimension
    const HIDDEN_DIM: usize = 128;   // Hidden layer dimension
    const NUM_LAYERS: usize = 3;     // Message passing layers

    // Load training data
    let train_path = args.data.join("code_train.jsonl");
    let val_path = args.data.join("code_val.jsonl");

    println!("Loading training data from {:?}", train_path);
    let train_examples = load_examples(&train_path)?;
    println!("Loaded {} training examples", train_examples.len());

    // Analyze target lengths for Sabag sizing
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

    // Create output directory
    let output_dir = args.output.parent().unwrap_or(&args.output);
    std::fs::create_dir_all(output_dir)?;

    // Use max output length that covers most targets
    let output_clusters = args.max_output_len.min(max_target + 50);
    println!("Using {} output clusters (Sabag expansion)", output_clusters);

    // Initialize or resume model
    let mut model = if let Some(resume_path) = &args.resume {
        println!("\nResuming from {:?}", resume_path);
        let checkpoint = UnifiedCheckpoint::load_from_file(resume_path)?;
        checkpoint.load_module()?
    } else {
        println!("\nInitializing GraphTransformNet...");
        println!("  Vocab: {}", VOCAB_SIZE);
        println!("  Embed dim: {}", EMBED_DIM);
        println!("  Hidden dim: {}", HIDDEN_DIM);
        println!("  Layers: {}", NUM_LAYERS);
        println!("  Output clusters: {} (Sabag expansion)", output_clusters);

        // Create model with Sabag pooling for expansion
        let mut model = GraphTransformNet::new(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS);

        // Override Sabag to expand to output_clusters
        model.sabag_pooling = Some(grapheme_core::SabagPooling::new(output_clusters, EMBED_DIM));

        model
    };

    // Structural loss configuration
    let loss_config = StructuralLossConfig {
        alpha: 1.0,   // Node insertion/deletion cost
        beta: 0.5,    // Edge insertion/deletion cost
        gamma: 0.3,   // Clique weight
        sinkhorn: SinkhornConfig {
            iterations: 20,
            temperature: 0.1,
            epsilon: 1e-6,
        },
    };

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.lr);
    println!("  Structural loss: α={}, β={}, γ={}", loss_config.alpha, loss_config.beta, loss_config.gamma);

    // Training loop
    let start = Instant::now();
    let mut best_val_loss = f32::INFINITY;

    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut epoch_similarity = 0.0f32;
        let mut batch_count = 0;

        // Shuffle training data
        let mut indices: Vec<usize> = (0..train_examples.len()).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        for batch_start in (0..indices.len()).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];

            // Zero gradients
            model.zero_grad();

            let mut batch_loss = 0.0f32;
            let mut batch_similarity = 0.0f32;

            for &idx in batch_indices {
                let example = &train_examples[idx];

                // Convert texts to graphs
                let input_graph = GraphemeGraph::from_text(&example.input);
                let target_graph = GraphemeGraph::from_text(&example.target);

                // Forward: Input graph → Output graph (via Sabag expansion)
                let (output_graph, pooling_result) = model.forward(&input_graph);

                // Compute structural loss between output and target
                let loss_result = compute_structural_loss(
                    &output_graph,
                    &target_graph,
                    &loss_config,
                );

                batch_loss += loss_result.total_loss;

                // Graph similarity (1 - normalized loss)
                let max_nodes = output_graph.node_count().max(target_graph.node_count());
                let max_edges = output_graph.edge_count().max(target_graph.edge_count());
                let max_cost = (max_nodes + max_edges) as f32;
                let similarity = if max_cost > 0.0 {
                    1.0 - (loss_result.total_loss / max_cost).min(1.0)
                } else {
                    1.0
                };
                batch_similarity += similarity;

                // Backward pass through model
                model.backward(&input_graph, &pooling_result, &loss_result.activation_gradients, EMBED_DIM);
            }

            let n = batch_indices.len() as f32;
            batch_loss /= n;
            batch_similarity /= n;

            // Update weights
            model.step(args.lr);

            epoch_loss += batch_loss;
            epoch_similarity += batch_similarity;
            batch_count += 1;

            if args.verbose && batch_count % 10 == 0 {
                // Decode a sample to see output
                let sample = &train_examples[batch_indices[0]];
                let sample_graph = GraphemeGraph::from_text(&sample.input);
                let (_, pool_res) = model.forward(&sample_graph);
                let decoded = model.decode(&pool_res);
                let preview: String = decoded.chars().take(40).collect();

                println!("    Batch {}: loss={:.4}, sim={:.1}%, preview=\"{}...\"",
                    batch_count, batch_loss, batch_similarity * 100.0, preview);
            }
        }

        let avg_loss = epoch_loss / batch_count as f32;
        let avg_sim = epoch_similarity / batch_count as f32;

        // Validation
        let val_result = if let Some(ref val) = val_examples {
            let mut val_loss = 0.0f32;
            let mut val_sim = 0.0f32;
            let mut char_accuracy = 0.0f32;

            for example in val {
                let input_graph = GraphemeGraph::from_text(&example.input);
                let target_graph = GraphemeGraph::from_text(&example.target);

                let (output_graph, pooling_result) = model.forward(&input_graph);

                // Structural loss
                let loss_result = compute_structural_loss(&output_graph, &target_graph, &loss_config);
                val_loss += loss_result.total_loss;

                // Graph similarity
                let max_nodes = output_graph.node_count().max(target_graph.node_count());
                let max_edges = output_graph.edge_count().max(target_graph.edge_count());
                let max_cost = (max_nodes + max_edges) as f32;
                let sim = if max_cost > 0.0 { 1.0 - (loss_result.total_loss / max_cost).min(1.0) } else { 1.0 };
                val_sim += sim;

                // Decode and compute character accuracy
                let decoded = model.decode(&pooling_result);
                let target_chars: Vec<char> = example.target.chars().collect();
                let decoded_chars: Vec<char> = decoded.chars().collect();
                let min_len = target_chars.len().min(decoded_chars.len());
                let matches = target_chars.iter().zip(decoded_chars.iter())
                    .filter(|(t, d)| t == d).count();
                let acc = if !target_chars.is_empty() {
                    matches as f32 / target_chars.len() as f32
                } else {
                    1.0
                };
                char_accuracy += acc;
            }

            let n = val.len() as f32;
            val_loss /= n;
            val_sim /= n;
            char_accuracy /= n;

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                let best_path = args.output.with_file_name("code_graph_best.json");
                model.save_to_file(&best_path)?;
                println!("  [NEW BEST] val_loss={:.4}, char_acc={:.1}%", val_loss, char_accuracy * 100.0);
            }

            Some((val_loss, val_sim, char_accuracy))
        } else {
            None
        };

        let epoch_time = epoch_start.elapsed();

        // Print epoch summary
        print!("Epoch {}/{}: train_loss={:.4}, graph_sim={:.1}%",
            epoch + 1, args.epochs, avg_loss, avg_sim * 100.0);

        if let Some((vl, vs, ca)) = val_result {
            print!(", val_loss={:.4}, val_sim={:.1}%, char_acc={:.1}%", vl, vs * 100.0, ca * 100.0);
        }

        println!(", time={:.1}s", epoch_time.as_secs_f64());

        // Checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0 {
            let ckpt_path = args.output.with_file_name(format!("code_graph_epoch{}.json", epoch + 1));
            model.save_to_file(&ckpt_path)?;
            println!("  Checkpoint: {:?}", ckpt_path);
        }
    }

    let total_time = start.elapsed();

    // Save final model
    model.save_to_file(&args.output)?;

    println!("\nTraining complete in {:.1}s", total_time.as_secs_f64());
    println!("Final model: {:?}", args.output);
    println!("Best val loss: {:.4}", best_val_loss);

    // Demo inference
    println!("\n--- Demo: Graph-to-Graph Code Generation ---");
    let demo_prompt = r#"def add(a, b):
    """ Add two numbers. """
"#;
    println!("Input prompt ({} chars):", demo_prompt.len());
    println!("{}", demo_prompt);

    let input_graph = GraphemeGraph::from_text(demo_prompt);
    println!("Input graph: {} nodes, {} edges", input_graph.node_count(), input_graph.edge_count());

    let (output_graph, pooling_result) = model.forward(&input_graph);
    println!("Output graph: {} nodes, {} edges", output_graph.node_count(), output_graph.edge_count());

    let generated = model.decode(&pooling_result);
    println!("Generated code:");
    println!("{}", generated);

    Ok(())
}
