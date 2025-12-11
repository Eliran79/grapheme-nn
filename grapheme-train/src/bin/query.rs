//! Query GRAPHEME model for knowledge retrieval.
//!
//! This binary allows querying a trained GRAPHEME model to retrieve
//! learned knowledge patterns using the actual trained neural network.

use clap::Parser;
use grapheme_core::{DagNN, GraphTransformNet, UnifiedCheckpoint};
use std::path::PathBuf;
use ndarray::Array1;

#[derive(Parser, Debug)]
#[command(name = "query")]
#[command(about = "Query GRAPHEME model for learned knowledge", long_about = None)]
struct Args {
    /// Path to trained model checkpoint
    #[arg(short, long)]
    model: PathBuf,

    /// Query text (if not provided, enters interactive mode)
    #[arg(short, long)]
    query: Option<String>,

    /// Number of top results to show
    #[arg(short = 'n', long, default_value = "5")]
    top_n: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Pool node embeddings into a single graph embedding
fn pool_embeddings(embeddings: &[Array1<f32>]) -> Array1<f32> {
    if embeddings.is_empty() {
        return Array1::zeros(64);
    }
    let dim = embeddings[0].len();
    let mut pooled = Array1::zeros(dim);
    for emb in embeddings {
        pooled = pooled + emb;
    }
    pooled / embeddings.len() as f32
}

/// Knowledge base with reference embeddings
struct KnowledgeBase {
    entries: Vec<(String, String, Array1<f32>)>, // (concept, description, embedding)
}

impl KnowledgeBase {
    fn new(model: &GraphTransformNet) -> Self {
        // Wikipedia knowledge topics
        let knowledge = vec![
            ("graph theory", "A branch of mathematics studying graphs - structures with vertices/nodes and edges connecting them"),
            ("machine learning", "A field of AI that enables systems to learn and improve from experience without explicit programming"),
            ("neural network", "Computing systems inspired by biological neural networks in the brain"),
            ("artificial intelligence", "The simulation of human intelligence processes by computer systems"),
            ("deep learning", "A subset of machine learning using neural networks with many layers (depth)"),
            ("algorithm", "A step-by-step procedure for solving a problem or accomplishing a task"),
            ("mathematics", "The abstract science of number, quantity, and space"),
            ("calculus", "Mathematical study of continuous change, including derivatives and integrals"),
            ("linear algebra", "Branch of mathematics concerning linear equations and their representations"),
            ("physics", "Natural science studying matter, energy, and fundamental forces of nature"),
            ("vertex", "A fundamental unit of graphs, also called a node"),
            ("edge", "A connection between two vertices in a graph"),
            ("derivative", "Rate of change of a function with respect to a variable"),
            ("integral", "The reverse operation of differentiation, finding area under curves"),
            ("training", "The process of teaching a model by showing it examples"),
            ("inference", "Using a trained model to make predictions on new data"),
            ("gradient descent", "An optimization algorithm that iteratively adjusts parameters to minimize loss"),
            ("backpropagation", "Algorithm for computing gradients in neural networks by chain rule"),
            ("activation function", "Non-linear function applied to neuron outputs (ReLU, sigmoid, tanh)"),
            ("loss function", "Measures difference between predicted and actual outputs"),
        ];

        let mut entries = Vec::new();
        for (concept, description) in knowledge {
            // Encode the concept using the trained model
            let dag = DagNN::from_text(concept).unwrap_or_else(|_| DagNN::new());
            let embeddings = model.encode(&dag);
            let embedding = pool_embeddings(&embeddings);
            entries.push((concept.to_string(), description.to_string(), embedding));
        }

        Self { entries }
    }

    fn search(&self, query_embedding: &Array1<f32>, top_n: usize) -> Vec<(String, String, f32)> {
        let mut results: Vec<_> = self.entries
            .iter()
            .map(|(concept, desc, emb)| {
                let similarity = cosine_similarity(query_embedding, emb);
                (concept.clone(), desc.clone(), similarity)
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().take(top_n).collect()
    }
}

fn query_model(query: &str, model: &GraphTransformNet, kb: &KnowledgeBase, verbose: bool, top_n: usize) -> Vec<(String, f32)> {
    println!("\nQuery: \"{}\"", query);
    println!("{}", "-".repeat(50));

    // Create a DagNN from the query text
    let query_dag = match DagNN::from_text(query) {
        Ok(dag) => dag,
        Err(e) => {
            println!("Error creating query graph: {:?}", e);
            return vec![];
        }
    };

    if verbose {
        println!("Query graph: {} nodes, {} edges",
            query_dag.graph.node_count(),
            query_dag.graph.edge_count());
    }

    // Encode query using the trained model
    let query_embeddings = model.encode(&query_dag);
    let query_embedding = pool_embeddings(&query_embeddings);

    if verbose {
        let norm: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("Query embedding norm: {:.4}", norm);
        println!("Query embedding (first 8): {:?}", &query_embedding.as_slice().unwrap()[..8.min(query_embedding.len())]);
    }

    // Search knowledge base
    let search_results = kb.search(&query_embedding, top_n);

    // Format results
    let mut results = Vec::new();
    for (concept, description, similarity) in search_results {
        let icon = if similarity > 0.8 {
            "📚" // High match
        } else if similarity > 0.5 {
            "📖" // Medium match
        } else {
            "📄" // Low match
        };
        results.push((format!("{} {}: {}", icon, concept.to_uppercase(), description), similarity));
    }

    if results.is_empty() {
        results.push((
            "🤖 No relevant knowledge found. Try asking about: graph theory, machine learning, neural networks, calculus...".to_string(),
            0.0
        ));
    }

    results
}

fn interactive_mode(model: &GraphTransformNet, kb: &KnowledgeBase, verbose: bool, top_n: usize) {
    use std::io::{self, BufRead, Write};

    println!("\n🧠 GRAPHEME Interactive Knowledge Query");
    println!("========================================");
    println!("Ask questions about what the model learned from Wikipedia.");
    println!("The model uses neural embeddings to find semantically similar concepts.");
    println!("Type 'quit' to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("🔍 query> ");
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
            println!("👋 Goodbye!");
            break;
        }

        let results = query_model(input, model, kb, verbose, top_n);

        println!("\n📊 Results:");
        for (i, (text, score)) in results.iter().enumerate() {
            println!("  {}. [sim: {:.4}] {}", i + 1, score, text);
        }
        println!();
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("🧠 GRAPHEME Knowledge Query");
    println!("===========================\n");

    // Load the model checkpoint
    println!("📂 Loading model from: {:?}", args.model);

    // Try to load as UnifiedCheckpoint first
    let model: GraphTransformNet = if let Ok(checkpoint) = UnifiedCheckpoint::load_from_file(&args.model) {
        match checkpoint.load_module::<GraphTransformNet>() {
            Ok(m) => {
                println!("✅ Loaded GraphTransformNet from checkpoint");
                m
            }
            Err(e) => {
                println!("⚠️ Could not load GraphTransformNet: {}", e);
                println!("📝 Creating default model for inference...");
                GraphTransformNet::new(256, 64, 64, 32)
            }
        }
    } else {
        // Try loading as raw JSON
        let checkpoint_data = std::fs::read_to_string(&args.model)?;
        if let Ok(m) = serde_json::from_str::<GraphTransformNet>(&checkpoint_data) {
            println!("✅ Loaded GraphTransformNet from JSON");
            m
        } else {
            println!("⚠️ Could not parse checkpoint, creating default model");
            GraphTransformNet::new(256, 64, 64, 32)
        }
    };

    println!("📊 Model: {} hidden dim, {} layers\n",
        model.hidden_dim,
        model.num_layers);

    // Build knowledge base with model embeddings
    println!("🔨 Building knowledge base embeddings...");
    let kb = KnowledgeBase::new(&model);
    println!("✅ Knowledge base ready ({} entries)\n", kb.entries.len());

    // Run query
    if let Some(query) = args.query {
        let results = query_model(&query, &model, &kb, args.verbose, args.top_n);

        println!("\n📊 Top {} results:", results.len().min(args.top_n));
        for (i, (text, score)) in results.iter().take(args.top_n).enumerate() {
            println!("  {}. [sim: {:.4}] {}", i + 1, score, text);
        }
    } else {
        interactive_mode(&model, &kb, args.verbose, args.top_n);
    }

    Ok(())
}
