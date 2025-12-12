//! Query GRAPHEME model for knowledge retrieval.
//!
//! This binary allows querying a trained GRAPHEME model to retrieve
//! learned knowledge using TRUE graph-based retrieval.
//!
//! Usage:
//!   query --model checkpoints/llm_final.checkpoint --kb knowledge.json --query "What is a derivative?"
//!   query --model checkpoints/llm_final.checkpoint --kb knowledge.json  # Interactive mode

use clap::Parser;
use grapheme_core::{GraphemeGraph, GraphTransformNet, UnifiedCheckpoint};
use grapheme_train::GraphKnowledgeBase;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "query")]
#[command(about = "Query GRAPHEME model for learned knowledge", long_about = None)]
struct Args {
    /// Path to trained model checkpoint
    #[arg(short, long)]
    model: PathBuf,

    /// Path to knowledge base JSON file
    #[arg(short = 'k', long)]
    kb: Option<PathBuf>,

    /// Query text (if not provided, enters interactive mode)
    #[arg(short, long)]
    query: Option<String>,

    /// Number of top results to show
    #[arg(short = 'n', long, default_value = "5")]
    top_n: usize,

    /// Use TRUE inference (graph transformation) instead of KB retrieval
    #[arg(long)]
    infer: bool,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Run TRUE GRAPHEME inference (graph-to-graph transformation)
fn run_inference(question: &str, model: &GraphTransformNet, verbose: bool) -> String {
    let input_graph = GraphemeGraph::from_text(question);

    if verbose {
        println!("  Input graph: {} nodes, {} edges",
            input_graph.node_count(),
            input_graph.edge_count());
    }

    // TRUE graph-to-graph transformation
    let (output_graph, decoded) = model.infer(&input_graph);

    if verbose {
        println!("  Output graph: {} nodes, {} edges",
            output_graph.node_count(),
            output_graph.edge_count());
    }

    decoded
}

/// Query knowledge base using graph similarity
fn query_kb(
    query: &str,
    model: &GraphTransformNet,
    kb: &mut GraphKnowledgeBase,
    top_n: usize,
    verbose: bool,
) {
    println!("\nQuery: \"{}\"", query);
    println!("{}", "-".repeat(50));

    if verbose {
        let query_graph = GraphemeGraph::from_text(query);
        println!("Query graph: {} nodes, {} edges",
            query_graph.node_count(),
            query_graph.edge_count());
    }

    // Query knowledge base using trained model embeddings
    let results = kb.query(query, model, top_n);

    if results.is_empty() {
        println!("No matching knowledge found.");
        println!("Tip: Use --infer flag for TRUE graph-to-graph inference.");
        return;
    }

    println!("\nResults:");
    for result in &results {
        let icon = if result.similarity > 0.8 {
            "+++"
        } else if result.similarity > 0.5 {
            "++"
        } else {
            "+"
        };
        println!("  [{}] ({} sim: {:.3}) Q: {}",
            result.rank,
            icon,
            result.similarity,
            truncate(&result.entry.question, 50));
        println!("      A: {}", truncate(&result.entry.answer, 70));
    }
}

/// Run TRUE inference and display result
fn run_inference_mode(query: &str, model: &GraphTransformNet, verbose: bool) {
    println!("\nQuery: \"{}\"", query);
    println!("{}", "-".repeat(50));
    println!("Mode: TRUE Graph-to-Graph Inference\n");

    let decoded = run_inference(query, model, verbose);

    // Clean up the decoded text (remove null chars, trim)
    let clean: String = decoded.chars()
        .filter(|c| *c >= ' ' && *c != '\0')
        .collect::<String>()
        .trim()
        .to_string();

    println!("Model output:");
    if clean.is_empty() {
        println!("  [empty output - model needs more training]");
    } else {
        println!("  \"{}\"", clean);
    }

    // Also show raw for debugging
    if verbose && !decoded.is_empty() {
        println!("\nRaw decoded ({} chars): {:?}",
            decoded.len(),
            &decoded[..decoded.len().min(100)]);
    }
}

fn interactive_mode(model: &GraphTransformNet, kb: &mut Option<GraphKnowledgeBase>, verbose: bool, top_n: usize, infer_mode: bool) {
    use std::io::{self, BufRead, Write};

    println!("\n GRAPHEME Knowledge Query");
    println!("========================================");
    if infer_mode {
        println!("Mode: TRUE Graph-to-Graph Inference");
    } else if kb.is_some() {
        println!("Mode: Knowledge Base Retrieval");
    } else {
        println!("Mode: Inference (no KB loaded)");
    }
    println!("Commands: 'quit', 'infer <text>', 'kb <text>'");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("query> ");
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).unwrap() == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input.to_lowercase() == "quit" || input.to_lowercase() == "exit" {
            println!("Goodbye!");
            break;
        }

        // Check for mode commands
        if input.starts_with("infer ") {
            let query = input.strip_prefix("infer ").unwrap();
            run_inference_mode(query, model, verbose);
            println!();
            continue;
        }

        if input.starts_with("kb ") {
            if let Some(ref mut kb_ref) = kb {
                let query = input.strip_prefix("kb ").unwrap();
                query_kb(query, model, kb_ref, top_n, verbose);
            } else {
                println!("No knowledge base loaded. Use --kb flag.");
            }
            println!();
            continue;
        }

        // Default mode
        if infer_mode {
            run_inference_mode(input, model, verbose);
        } else if let Some(ref mut kb_ref) = kb {
            query_kb(input, model, kb_ref, top_n, verbose);
        } else {
            run_inference_mode(input, model, verbose);
        }
        println!();
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!(" GRAPHEME Knowledge Query");
    println!("===========================\n");

    // Load the model checkpoint
    println!("Loading model from: {:?}", args.model);

    let model: GraphTransformNet = if let Ok(checkpoint) = UnifiedCheckpoint::load_from_file(&args.model) {
        match checkpoint.load_module::<GraphTransformNet>() {
            Ok(m) => {
                println!("Loaded GraphTransformNet from checkpoint");
                m
            }
            Err(e) => {
                println!("Could not load GraphTransformNet: {}", e);
                println!("Creating default model...");
                GraphTransformNet::new(256, 64, 64, 32)
            }
        }
    } else {
        // Try loading as raw JSON
        let checkpoint_data = std::fs::read_to_string(&args.model)?;
        if let Ok(m) = serde_json::from_str::<GraphTransformNet>(&checkpoint_data) {
            println!("Loaded GraphTransformNet from JSON");
            m
        } else {
            println!("Could not parse checkpoint, creating default model");
            GraphTransformNet::new(256, 64, 64, 32)
        }
    };

    println!("Model: {} hidden dim, {} layers",
        model.hidden_dim,
        model.mp_layers.len());

    // Load knowledge base if provided
    let mut kb: Option<GraphKnowledgeBase> = if let Some(ref kb_path) = args.kb {
        println!("\nLoading knowledge base from: {:?}", kb_path);
        match GraphKnowledgeBase::load(kb_path) {
            Ok(loaded_kb) => {
                let stats = loaded_kb.stats();
                println!("Loaded {} knowledge entries", stats.total_entries);
                for (topic, count) in &stats.entries_by_topic {
                    println!("  - {}: {} entries", topic, count);
                }
                Some(loaded_kb)
            }
            Err(e) => {
                println!("Failed to load KB: {}", e);
                println!("Starting with empty knowledge base.");
                None
            }
        }
    } else {
        println!("\nNo knowledge base specified. Use --kb <path> to load one.");
        println!("Using TRUE inference mode (--infer).\n");
        None
    };

    // Run query
    if let Some(query) = args.query {
        if args.infer {
            run_inference_mode(&query, &model, args.verbose);
        } else if let Some(ref mut kb_ref) = kb {
            query_kb(&query, &model, kb_ref, args.top_n, args.verbose);
        } else {
            run_inference_mode(&query, &model, args.verbose);
        }
    } else {
        interactive_mode(&model, &mut kb, args.verbose, args.top_n, args.infer);
    }

    Ok(())
}
