//! GRAPHEME Semantic Code Training - TRUE Graph-to-Graph with SemanticDecoder
//!
//! This is the GRAPHEME vision for code generation:
//! - Input: Text graph (prompt/docstring)
//! - Output: Semantic code graph (keywords, variables, operators, NOT characters!)
//!
//! Example transformation:
//!   "write a function that prints Hi if x>2"
//!     →
//!   Graph: [Keyword(def), Variable(f), Punct('('), Variable(x), Punct(')'), Punct(':'),
//!           Space(Newline), Space(Indent), Keyword(if), Variable(x), Op(>), Int(2),
//!           Punct(':'), Space(Newline), Space(Indent), Call(print), Punct('('),
//!           Str("Hi"), Punct(')'), EndSeq]
//!
//! This trains on SEMANTIC NODES not characters!
//!
//! Key innovation: Uses SemanticDecoder with unified vocabulary from all domain brains
//! to enable generation of ANY semantic node type, not just the types in the input.

use anyhow::Result;
use clap::Parser;
use grapheme_core::{ActivationFn, GraphemeGraph, Node};
use grapheme_train::cortex_mesh::{CortexMesh, MeshConfig};
use grapheme_train::semantic_decoder::{SemanticDecoder, SemanticDecoderConfig};
use grapheme_train::{compute_structural_loss, StructuralLossConfig};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_semantic_code")]
#[command(about = "Train GRAPHEME on semantic code graphs (NOT characters!)")]
struct Args {
    /// Path to training data directory
    #[arg(short, long)]
    data: PathBuf,

    /// Output checkpoint path
    #[arg(short, long)]
    output: PathBuf,

    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value = "16")]
    batch_size: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    lr: f32,

    /// Resume from checkpoint
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Verbose output level (0-2)
    #[arg(short, long, default_value = "1")]
    verbose: usize,
}

/// Training sample
#[derive(Debug, Clone, Deserialize)]
struct TrainingSample {
    input: String,
    #[serde(alias = "target")]
    output: String,
    #[serde(default)]
    #[allow(dead_code)]  // Reserved for future domain-specific training
    domain: Option<String>,
}

/// Load training data
fn load_training_data(data_dir: &PathBuf) -> Result<Vec<TrainingSample>> {
    let mut samples = Vec::new();

    // Look for JSONL files
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
            let file = File::open(&path)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(sample) = serde_json::from_str::<TrainingSample>(&line) {
                    samples.push(sample);
                }
            }
        }
    }

    // Also try loading from subdirectories
    for subdir in ["code", "humaneval"] {
        let subpath = data_dir.join(subdir);
        if subpath.exists() {
            for entry in std::fs::read_dir(&subpath)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
                    let file = File::open(&path)?;
                    let reader = BufReader::new(file);

                    for line in reader.lines() {
                        let line = line?;
                        if line.trim().is_empty() {
                            continue;
                        }
                        if let Ok(sample) = serde_json::from_str::<TrainingSample>(&line) {
                            samples.push(sample);
                        }
                    }
                }
            }
        }
    }

    Ok(samples)
}

/// Compute semantic node type accuracy
fn semantic_accuracy(pred_graph: &GraphemeGraph, target_graph: &GraphemeGraph) -> f32 {
    let pred_nodes: Vec<_> = pred_graph.graph.node_indices()
        .map(|idx| &pred_graph.graph[idx])
        .collect();
    let target_nodes: Vec<_> = target_graph.graph.node_indices()
        .map(|idx| &target_graph.graph[idx])
        .collect();

    if target_nodes.is_empty() {
        return if pred_nodes.is_empty() { 1.0 } else { 0.0 };
    }

    let mut matches = 0;
    let check_len = pred_nodes.len().min(target_nodes.len());

    for i in 0..check_len {
        // Compare node types
        let pred_type = format!("{:?}", pred_nodes[i].node_type);
        let target_type = format!("{:?}", target_nodes[i].node_type);

        // Extract the type name (before any parameters)
        let pred_name = pred_type.split('(').next().unwrap_or(&pred_type);
        let target_name = target_type.split('(').next().unwrap_or(&target_type);

        if pred_name == target_name {
            matches += 1;
        }
    }

    matches as f32 / target_nodes.len() as f32
}

/// Compute exact code match
fn exact_code_match(pred_graph: &GraphemeGraph, target_code: &str) -> bool {
    let predicted_code = pred_graph.to_code();
    predicted_code.trim() == target_code.trim()
}

/// Decode pooled features to a semantic graph using SemanticDecoder
fn decode_features_to_graph(
    features: &ndarray::Array2<f32>,
    decoder: &SemanticDecoder,
) -> GraphemeGraph {
    use petgraph::graph::DiGraph;
    use grapheme_core::Edge;

    let mut graph: DiGraph<Node, Edge> = DiGraph::new();
    let mut prev_idx = None;
    let mut input_nodes = Vec::new();

    // Decode each feature vector to a semantic node type
    for i in 0..features.nrows() {
        let hidden: Vec<f32> = features.row(i).to_vec();
        let (node_type, confidence) = decoder.decode(&hidden);

        let node = Node {
            value: None,
            activation: confidence,
            pre_activation: confidence,
            node_type,
            position: Some(i),
            activation_fn: ActivationFn::Linear,
        };

        let idx = graph.add_node(node);
        input_nodes.push(idx);

        // Add sequential edge
        if let Some(prev) = prev_idx {
            graph.add_edge(prev, idx, Edge::sequential());
        }
        prev_idx = Some(idx);
    }

    GraphemeGraph {
        graph,
        input_nodes,
        cliques: Vec::new(),
    }
}

/// Prepare training batch for SemanticDecoder
/// Returns (hidden_vectors, target_indices) for decoder training
fn prepare_decoder_batch(
    features: &ndarray::Array2<f32>,
    target_graph: &GraphemeGraph,
    decoder: &SemanticDecoder,
) -> Vec<(Vec<f32>, usize)> {
    let mut batch = Vec::new();
    let target_nodes: Vec<_> = target_graph.graph.node_indices()
        .map(|idx| &target_graph.graph[idx])
        .collect();

    // Match features to target nodes (min of both lengths)
    let n = features.nrows().min(target_nodes.len());

    for i in 0..n {
        let hidden: Vec<f32> = features.row(i).to_vec();
        let target_type = &target_nodes[i].node_type;

        // Get target index in vocabulary
        if let Some(target_idx) = decoder.get_index(target_type) {
            batch.push((hidden, target_idx));
        }
    }

    batch
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     GRAPHEME Semantic Code Training - TRUE Graph-to-Graph   ║");
    println!("║   Nodes are: Keyword, Variable, Int, Op, etc. NOT chars!    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load training data
    println!("Loading training data from {:?}...", args.data);
    let samples = load_training_data(&args.data)?;
    println!("Loaded {} training samples", samples.len());

    if samples.is_empty() {
        println!("WARNING: No training samples found!");
        println!("Expected JSONL files with {{\"input\": ..., \"output\": ...}} format");
        return Ok(());
    }

    // Show example semantic graph conversion
    println!("\n--- Semantic Graph Example ---");
    let example_code = "if x > 2:\n    print('Hi')";
    let semantic_graph = GraphemeGraph::from_code(example_code);
    println!("Code: {}", example_code);
    println!("Semantic nodes ({} nodes):", semantic_graph.node_count());
    for (i, idx) in semantic_graph.graph.node_indices().take(10).enumerate() {
        let node = &semantic_graph.graph[idx];
        println!("  [{}] {:?}", i, node.node_type);
    }
    if semantic_graph.node_count() > 10 {
        println!("  ... ({} more nodes)", semantic_graph.node_count() - 10);
    }
    let reconstructed = semantic_graph.to_code();
    println!("Reconstructed: {}", reconstructed);
    println!();

    // Create mesh configuration
    let embed_dim = 64;  // Save for decoder config
    let config = MeshConfig {
        activation_threshold: 0.2,
        max_active_brains: usize::MAX,
        parallel: true,
        hidden_dim: 256,
        num_layers: 6,
        vocab_size: 256,  // For character-level input
        embed_dim,
    };

    // Create or resume mesh
    let mut mesh = if let Some(resume_path) = &args.resume {
        println!("Resuming from {:?}...", resume_path);
        let mut mesh = CortexMesh::discover_with_config(config);
        mesh.load(resume_path)?;
        mesh
    } else {
        println!("Initializing CortexMesh...");
        CortexMesh::discover_with_config(config)
    };

    println!("\nMesh Ready:");
    println!("  Brains: {}", mesh.brain_count());
    println!("  Modules: {}", mesh.module_count());

    // Create SemanticDecoder with unified vocabulary from all brains
    println!("\nBuilding unified semantic vocabulary...");
    let vocab = SemanticDecoder::build_vocab_from_brains();
    let decoder_config = SemanticDecoderConfig {
        hidden_dim: embed_dim,  // Match embedding dimension
        learning_rate: args.lr,
        temperature: 1.0,
        label_smoothing: 0.1,
    };
    let mut decoder = SemanticDecoder::new(vocab, decoder_config);
    let vocab_stats = decoder.vocab_stats();
    println!("SemanticDecoder ready:");
    println!("  Vocabulary size: {}", decoder.vocab_size());
    println!("  Node types: {} Keywords, {} Ops, {} Puncts, {} Input chars",
        vocab_stats.by_type.get("Keyword").unwrap_or(&0),
        vocab_stats.by_type.get("Op").unwrap_or(&0),
        vocab_stats.by_type.get("Punct").unwrap_or(&0),
        vocab_stats.by_type.get("Input").unwrap_or(&0));

    // Training loop
    let mut best_loss = f32::MAX;
    let train_size = (samples.len() as f32 * 0.9) as usize;
    let (train_samples, val_samples) = samples.split_at(train_size);

    println!("\nTraining: {} samples, Validation: {} samples", train_samples.len(), val_samples.len());
    println!("Epochs: {}, Batch size: {}, LR: {}\n", args.epochs, args.batch_size, args.lr);

    let loss_config = StructuralLossConfig::default();

    // Learning rate schedule
    let warmup_epochs = 5;
    let min_lr = args.lr * 0.01;

    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();

        // Calculate learning rate with warmup and cosine decay
        let current_lr = if epoch < warmup_epochs {
            args.lr * (epoch + 1) as f32 / warmup_epochs as f32
        } else {
            let decay_epoch = epoch - warmup_epochs;
            let decay_total = args.epochs - warmup_epochs;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * decay_epoch as f32 / decay_total as f32).cos());
            min_lr + (args.lr - min_lr) * cosine
        };

        // Training
        let mut train_loss = 0.0;
        let pb = ProgressBar::new(train_samples.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

        let mut decoder_loss = 0.0;
        let progress = AtomicUsize::new(0);

        for batch in train_samples.chunks(args.batch_size) {
            mesh.zero_grad();

            // PARALLEL: Pre-compute all graphs for the batch
            let graphs: Vec<(GraphemeGraph, GraphemeGraph)> = batch
                .par_iter()
                .map(|sample| {
                    let input_graph = GraphemeGraph::from_text(&sample.input);
                    let target_graph = GraphemeGraph::from_code(&sample.output);
                    (input_graph, target_graph)
                })
                .collect();

            // Collect decoder training batch across all samples in batch
            let mut decoder_batch = Vec::new();

            for (input_graph, target_graph) in &graphs {
                // Forward pass - get pooled features (sequential due to mutable model state)
                let (output_graph, pooling) = mesh.model.forward(input_graph);

                // Build decoder training batch from features and target
                let sample_batch = prepare_decoder_batch(&pooling.features, target_graph, &decoder);
                decoder_batch.extend(sample_batch);

                // Compute structural loss against SEMANTIC target
                let loss_result = compute_structural_loss(&output_graph, target_graph, &loss_config);
                train_loss += loss_result.total_loss;

                // Backward pass for mesh
                mesh.model.backward(input_graph, &pooling, &loss_result.activation_gradients, mesh.config.embed_dim);
            }

            mesh.model.step(current_lr / batch.len() as f32);

            // Train SemanticDecoder on accumulated batch
            if !decoder_batch.is_empty() {
                let loss = decoder.backward(&decoder_batch);
                decoder_loss += loss * decoder_batch.len() as f32;
            }

            progress.fetch_add(batch.len(), Ordering::Relaxed);
            pb.set_position(progress.load(Ordering::Relaxed) as u64);
        }
        pb.finish_and_clear();

        train_loss /= train_samples.len() as f32;
        decoder_loss /= train_samples.len() as f32;

        // Validation - PARALLEL: Process validation samples in parallel
        // Step 1: Parallel pre-compute all graphs
        let val_graphs: Vec<(GraphemeGraph, GraphemeGraph, &str)> = val_samples
            .par_iter()
            .map(|sample| {
                let input_graph = GraphemeGraph::from_text(&sample.input);
                let target_graph = GraphemeGraph::from_code(&sample.output);
                (input_graph, target_graph, sample.output.as_str())
            })
            .collect();

        // Step 2: Parallel forward pass and metrics computation
        // Note: mesh.model.forward() takes &self (immutable), so it's thread-safe
        let val_results: Vec<(f32, f32, f32, bool)> = val_graphs
            .par_iter()
            .map(|(input_graph, target_graph, output_code)| {
                let (output_graph, pooling) = mesh.model.forward(input_graph);

                let loss_result = compute_structural_loss(&output_graph, target_graph, &loss_config);
                let loss = loss_result.total_loss;

                // Semantic accuracy on raw output (before decoder)
                let sem_acc = semantic_accuracy(&output_graph, target_graph);

                // Decoder accuracy: decode features and compare to target
                let decoded_graph = decode_features_to_graph(&pooling.features, &decoder);
                let dec_batch = prepare_decoder_batch(&pooling.features, target_graph, &decoder);
                let dec_acc = if !dec_batch.is_empty() {
                    decoder.compute_accuracy(&dec_batch)
                } else {
                    0.0
                };

                // Exact code match (using decoded graph)
                let exact_match = exact_code_match(&decoded_graph, output_code);

                (loss, sem_acc, dec_acc, exact_match)
            })
            .collect();

        // Step 3: Aggregate results
        let val_loss: f32 = val_results.iter().map(|(l, _, _, _)| l).sum::<f32>() / val_samples.len() as f32;
        let _val_semantic_acc: f32 = val_results.iter().map(|(_, s, _, _)| s).sum::<f32>() / val_samples.len() as f32;
        let val_decoder_acc: f32 = val_results.iter().map(|(_, _, d, _)| d).sum::<f32>() / val_samples.len() as f32;
        let val_exact_matches = val_results.iter().filter(|(_, _, _, e)| *e).count();
        let exact_match_rate = val_exact_matches as f32 / val_samples.len() as f32;

        let epoch_time = epoch_start.elapsed();

        // Save best model
        let is_best = val_loss < best_loss;
        if is_best {
            best_loss = val_loss;
            let best_path = args.output.with_file_name("semantic_code_best.json");
            mesh.save(&best_path)?;
            // Also save decoder
            let decoder_path = args.output.with_file_name("semantic_decoder_best.json");
            decoder.save(decoder_path.to_str().unwrap())?;
        }

        // Print progress - now includes decoder_acc which should improve!
        print!("Epoch {}/{}: train={:.1}, dec_loss={:.2}, dec_acc={:.1}%, exact={:.1}%",
            epoch + 1, args.epochs, train_loss, decoder_loss,
            val_decoder_acc * 100.0, exact_match_rate * 100.0);

        if is_best {
            print!(" [BEST]");
        }
        println!(", lr={:.6}, time={:.1}s", current_lr, epoch_time.as_secs_f64());

        // Demo inference every 10 epochs
        if args.verbose >= 1 && (epoch + 1) % 10 == 0 {
            println!("\n--- Demo Inference (Epoch {}) ---", epoch + 1);
            let demo_input = &val_samples[0].input;
            let demo_target = &val_samples[0].output;

            let input_graph = GraphemeGraph::from_text(demo_input);
            let (_, pooling) = mesh.model.forward(&input_graph);

            // Use SemanticDecoder to decode features to semantic graph
            let decoded_graph = decode_features_to_graph(&pooling.features, &decoder);

            println!("Input: {}...", demo_input.chars().take(60).collect::<String>());
            println!("Target code: {}", demo_target.chars().take(80).collect::<String>());
            println!("Decoded graph ({} nodes from {} features):",
                decoded_graph.node_count(), pooling.features.nrows());

            // Show decoded semantic nodes (now should show diverse types!)
            for (i, idx) in decoded_graph.graph.node_indices().take(8).enumerate() {
                let node = &decoded_graph.graph[idx];
                println!("  [{}] {:?}", i, node.node_type);
            }
            let predicted_code = decoded_graph.to_code();
            println!("Decoded code: {}", predicted_code.chars().take(80).collect::<String>());
            println!();
        }

        // Periodic checkpoint
        if (epoch + 1) % 20 == 0 {
            let checkpoint_path = args.output.with_file_name(format!("semantic_code_epoch{}.json", epoch + 1));
            mesh.save(&checkpoint_path)?;
            let decoder_checkpoint = args.output.with_file_name(format!("semantic_decoder_epoch{}.json", epoch + 1));
            decoder.save(decoder_checkpoint.to_str().unwrap())?;
        }
    }

    // Save final model and decoder
    mesh.save(&args.output)?;
    let decoder_final_path = args.output.with_file_name("semantic_decoder_final.json");
    decoder.save(decoder_final_path.to_str().unwrap())?;
    println!("\nFinal model saved to {:?}", args.output);
    println!("Final decoder saved to {:?}", decoder_final_path);
    println!("Best validation loss: {:.4}", best_loss);

    Ok(())
}
