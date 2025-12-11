//! GRAPHEME Inference Binary - Graph-to-Text Decoding
//!
//! This binary performs actual inference using trained models:
//! 1. Takes a question as input
//! 2. Converts to graph
//! 3. Runs forward pass through trained model (GraphTransformNet or EncoderDecoder)
//! 4. Decodes output embeddings back to text
//!
//! Usage:
//!   infer --model checkpoints/llm_final.checkpoint --question "What is a derivative?"
//!   infer --model checkpoints/enc_dec.checkpoint --encoder-decoder --question "What is 2+2?"

use clap::Parser;
use grapheme_core::{EncoderDecoder, GraphemeGraph, GraphTransformNet, UnifiedCheckpoint};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "infer")]
#[command(about = "Run inference on trained GRAPHEME model")]
struct Args {
    /// Path to trained model checkpoint
    #[arg(short, long)]
    model: PathBuf,

    /// Question to ask the model
    #[arg(short, long)]
    question: Option<String>,

    /// Use encoder-decoder architecture (backend-207)
    #[arg(long)]
    encoder_decoder: bool,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Run inference with GraphTransformNet
fn infer_transform_net(model: &GraphTransformNet, question: &str, verbose: bool) {
    let input_graph = GraphemeGraph::from_text(question);

    if verbose {
        println!("  Input graph: {} nodes, {} edges",
            input_graph.node_count(),
            input_graph.edge_count());
    }

    // Use model.infer() for end-to-end inference
    let (output_graph, decoded_text) = model.infer(&input_graph);

    if verbose {
        println!("  Output graph: {} nodes, {} edges",
            output_graph.node_count(),
            output_graph.edge_count());
    }

    // Show decoded output
    println!("\nModel output (decoded):");
    let clean: String = decoded_text.chars()
        .filter(|c| *c >= ' ' && *c != '\0')
        .collect::<String>()
        .trim()
        .to_string();

    if clean.is_empty() {
        println!("  [empty output - model needs more training]");
    } else {
        println!("  \"{}\"", clean);
    }

    // Also show the direct graph text if different
    let direct_text = output_graph.to_text();
    if !direct_text.is_empty() && direct_text != decoded_text {
        println!("\nOutput graph text (direct):");
        println!("  \"{}\"", direct_text);
    }

    // Explain what's happening
    if decoded_text == question {
        println!("\nNote: Output matches input - model is in identity/autoencoder mode.");
        println!("   This means Sabag pooling is preserving the input structure.");
        println!("   For Q->A generation, use --encoder-decoder with an EncoderDecoder model.");
    }
}

/// Run inference with EncoderDecoder
fn infer_encoder_decoder(model: &mut EncoderDecoder, question: &str, verbose: bool) {
    let input_graph = GraphemeGraph::from_text(question);

    if verbose {
        println!("  Input graph: {} nodes, {} edges",
            input_graph.node_count(),
            input_graph.edge_count());
        println!("  Encoder hidden dim: {}", model.encoder.hidden_dim);
        println!("  Decoder max length: {}", model.decoder.max_length);
    }

    // Use encoder-decoder forward pass
    let (output_embeddings, decoded_text) = model.forward(&input_graph);

    if verbose {
        println!("  Output embeddings: {} x {}",
            output_embeddings.nrows(),
            output_embeddings.ncols());
    }

    // Show decoded output
    println!("\nModel output (encoder-decoder):");
    let clean: String = decoded_text.chars()
        .filter(|c| *c >= ' ' && *c != '\0')
        .collect::<String>()
        .trim()
        .to_string();

    if clean.is_empty() {
        println!("  [empty output - model needs more training]");
    } else {
        println!("  \"{}\"", clean);
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("GRAPHEME Inference");
    println!("====================\n");

    // Load model
    println!("Loading model from: {:?}", args.model);

    let checkpoint = UnifiedCheckpoint::load_from_file(&args.model)
        .map_err(|e| anyhow::anyhow!("Failed to parse checkpoint: {}", e))?;

    // Get question
    let question = if let Some(q) = args.question {
        q
    } else {
        // Interactive mode
        println!("\nEnter your question (or 'quit' to exit):");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        input.trim().to_string()
    };

    if question.to_lowercase() == "quit" {
        return Ok(());
    }

    println!("\nQuestion: \"{}\"", question);
    println!("{}", "-".repeat(50));

    if args.encoder_decoder {
        // Try to load EncoderDecoder
        match checkpoint.load_module::<EncoderDecoder>() {
            Ok(mut model) => {
                println!("Loaded EncoderDecoder model");
                infer_encoder_decoder(&mut model, &question, args.verbose);
            }
            Err(e) => {
                println!("Could not load EncoderDecoder: {}", e);
                println!("Falling back to GraphTransformNet...");

                // Fallback to GraphTransformNet
                match checkpoint.load_module::<GraphTransformNet>() {
                    Ok(model) => {
                        println!("Loaded GraphTransformNet (fallback)");
                        println!("Model: {} hidden dim, {} layers",
                            model.hidden_dim,
                            model.mp_layers.len());
                        infer_transform_net(&model, &question, args.verbose);
                    }
                    Err(e2) => {
                        return Err(anyhow::anyhow!(
                            "Failed to load any model. EncoderDecoder: {}, GraphTransformNet: {}",
                            e, e2
                        ));
                    }
                }
            }
        }
    } else {
        // Load GraphTransformNet (default)
        match checkpoint.load_module::<GraphTransformNet>() {
            Ok(model) => {
                println!("Loaded GraphTransformNet");
                println!("Model: {} hidden dim, {} layers\n",
                    model.hidden_dim,
                    model.mp_layers.len());
                infer_transform_net(&model, &question, args.verbose);
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to load GraphTransformNet: {}", e));
            }
        }
    }

    Ok(())
}
