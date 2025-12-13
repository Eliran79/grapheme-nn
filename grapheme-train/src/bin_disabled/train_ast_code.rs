//! Train AST-level Code Generation with CodeBrain
//!
//! TRUE GRAPHEME approach for code:
//! - Input: Prompt text → GraphemeGraph (character-level)
//! - Transform: Learn to predict AST structure
//! - Output: CodeGraph with AST nodes (Function, Variable, Call, etc.)
//! - Decode: Render CodeGraph → source code
//!
//! Key insight: CodeBrain has rich AST entities:
//! - Function { name, params, return_type }
//! - Variable { name, var_type }
//! - Call { function, arg_count }
//! - If, Loop, Return, Block
//! - BinaryOp, UnaryOp, Literal
//!
//! The output graph is NOT characters - it's an AST!
//!
//! Architecture:
//! ```text
//!   Prompt (text) → GraphemeGraph → [Encoder] → Latent
//!                                       ↓
//!                               [AST Predictor]
//!                                       ↓
//!               CodeGraph (AST nodes) ← [Decoder]
//!                       ↓
//!               render_to_source()
//!                       ↓
//!               Output Code (text)
//! ```
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin train_ast_code -- \
//!     --data data/code_training --output checkpoints/ast_code.json

use clap::Parser;
use grapheme_code::{
    new_code_node, CodeEdge, CodeGraph, CodeNodeType, Language,
    LiteralValue, BinaryOperator, TreeSitterParser,
};
use grapheme_core::{DomainBrain, GraphemeGraph, GraphTransformNet, Learnable, UnifiedCheckpoint};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_ast_code")]
#[command(about = "Train AST-level code generation using CodeGraph")]
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
    #[arg(short, long, default_value = "16")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.0005")]
    lr: f32,

    /// Resume from checkpoint
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Training example
#[derive(Debug, Clone, Deserialize, Serialize)]
struct CodeExample {
    id: String,
    input: String,   // Prompt with docstring
    target: String,  // Solution code
    #[serde(default)]
    level: u32,
}

/// Load examples from JSONL
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

// =============================================================================
// AST Node Type Vocabulary
// =============================================================================

/// AST node type vocabulary for prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AstNodeTypeId {
    // Structural
    Module = 0,
    Function = 1,
    Variable = 2,
    Block = 3,

    // Expressions
    Literal = 4,
    Identifier = 5,
    BinaryOp = 6,
    UnaryOp = 7,
    Call = 8,

    // Control flow
    If = 9,
    Loop = 10,
    Return = 11,

    // Other
    Assignment = 12,
    Type = 13,
    Comment = 14,
    ExprStmt = 15,

    // Special
    EndOfSequence = 16,
    Padding = 17,
}

impl AstNodeTypeId {
    pub const VOCAB_SIZE: usize = 18;

    pub fn from_code_node_type(node_type: &CodeNodeType) -> Self {
        match node_type {
            CodeNodeType::Module { .. } => AstNodeTypeId::Module,
            CodeNodeType::Function { .. } => AstNodeTypeId::Function,
            CodeNodeType::Variable { .. } => AstNodeTypeId::Variable,
            CodeNodeType::Block => AstNodeTypeId::Block,
            CodeNodeType::Literal(_) => AstNodeTypeId::Literal,
            CodeNodeType::Identifier(_) => AstNodeTypeId::Identifier,
            CodeNodeType::BinaryOp(_) => AstNodeTypeId::BinaryOp,
            CodeNodeType::UnaryOp(_) => AstNodeTypeId::UnaryOp,
            CodeNodeType::Call { .. } => AstNodeTypeId::Call,
            CodeNodeType::If => AstNodeTypeId::If,
            CodeNodeType::Loop { .. } => AstNodeTypeId::Loop,
            CodeNodeType::Return => AstNodeTypeId::Return,
            CodeNodeType::Assignment => AstNodeTypeId::Assignment,
            CodeNodeType::Type(_) => AstNodeTypeId::Type,
            CodeNodeType::Comment(_) => AstNodeTypeId::Comment,
            CodeNodeType::ExprStmt => AstNodeTypeId::ExprStmt,
        }
    }
}

// =============================================================================
// CodeGraph Rendering (AST → Source Code)
// =============================================================================

/// Render a CodeGraph back to source code
fn render_code_graph(graph: &CodeGraph) -> String {
    let Some(root) = graph.root else {
        return String::new();
    };

    render_node(graph, root, 0)
}

fn render_node(graph: &CodeGraph, node_idx: NodeIndex, indent: usize) -> String {
    let node = &graph.graph[node_idx];
    let indent_str = "    ".repeat(indent);

    match &node.node_type {
        CodeNodeType::Module { name, .. } => {
            let mut result = format!("# module: {}\n", name);
            // Render children
            for edge in graph.graph.edges(node_idx) {
                if let CodeEdge::Child(_) = edge.weight() {
                    result.push_str(&render_node(graph, edge.target(), indent));
                }
            }
            result
        }

        CodeNodeType::Function { name, params, return_type } => {
            let params_str = params.join(", ");
            let ret_str = return_type.as_ref().map(|t| format!(" -> {}", t)).unwrap_or_default();
            let mut result = format!("{}def {}({}){}:\n", indent_str, name, params_str, ret_str);

            // Render body (children)
            let mut has_body = false;
            for edge in graph.graph.edges(node_idx) {
                if let CodeEdge::Child(_) = edge.weight() {
                    result.push_str(&render_node(graph, edge.target(), indent + 1));
                    has_body = true;
                }
            }

            if !has_body {
                result.push_str(&format!("{}    pass\n", indent_str));
            }

            result
        }

        CodeNodeType::Variable { name, var_type } => {
            match var_type {
                Some(t) => format!("{}{}: {}\n", indent_str, name, t),
                None => format!("{}{}\n", indent_str, name),
            }
        }

        CodeNodeType::Return => {
            let mut result = format!("{}return ", indent_str);
            // Get return value from children
            for edge in graph.graph.edges(node_idx) {
                if let CodeEdge::Child(0) = edge.weight() {
                    result.push_str(&render_node(graph, edge.target(), 0).trim());
                    break;
                }
            }
            result.push('\n');
            result
        }

        CodeNodeType::Literal(lit) => {
            match lit {
                LiteralValue::Integer(n) => n.to_string(),
                LiteralValue::Float(f) => f.to_string(),
                LiteralValue::String(s) => format!("\"{}\"", s),
                LiteralValue::Boolean(b) => if *b { "True" } else { "False" }.to_string(),
                LiteralValue::Null => "None".to_string(),
            }
        }

        CodeNodeType::Identifier(name) => name.clone(),

        CodeNodeType::BinaryOp(op) => {
            let op_str = match op {
                BinaryOperator::Add => "+",
                BinaryOperator::Sub => "-",
                BinaryOperator::Mul => "*",
                BinaryOperator::Div => "/",
                BinaryOperator::Mod => "%",
                BinaryOperator::Eq => "==",
                BinaryOperator::Ne => "!=",
                BinaryOperator::Lt => "<",
                BinaryOperator::Gt => ">",
                BinaryOperator::Le => "<=",
                BinaryOperator::Ge => ">=",
                BinaryOperator::And => "and",
                BinaryOperator::Or => "or",
                BinaryOperator::BitAnd => "&",
                BinaryOperator::BitOr => "|",
                BinaryOperator::BitXor => "^",
                BinaryOperator::Shl => "<<",
                BinaryOperator::Shr => ">>",
            };

            let mut left = String::new();
            let mut right = String::new();

            for edge in graph.graph.edges(node_idx) {
                match edge.weight() {
                    CodeEdge::Child(0) => left = render_node(graph, edge.target(), 0).trim().to_string(),
                    CodeEdge::Child(1) => right = render_node(graph, edge.target(), 0).trim().to_string(),
                    _ => {}
                }
            }

            format!("{} {} {}", left, op_str, right)
        }

        CodeNodeType::Call { function, arg_count } => {
            let mut args = vec![String::new(); *arg_count];

            for edge in graph.graph.edges(node_idx) {
                if let CodeEdge::Child(i) = edge.weight() {
                    if (*i as usize) < args.len() {
                        args[*i as usize] = render_node(graph, edge.target(), 0).trim().to_string();
                    }
                }
            }

            format!("{}({})", function, args.join(", "))
        }

        CodeNodeType::If => {
            let mut condition = String::new();
            let mut then_block = String::new();
            let mut else_block = String::new();

            for edge in graph.graph.edges(node_idx) {
                match edge.weight() {
                    CodeEdge::Child(0) => condition = render_node(graph, edge.target(), 0).trim().to_string(),
                    CodeEdge::Child(1) => then_block = render_node(graph, edge.target(), indent + 1),
                    CodeEdge::Child(2) => else_block = render_node(graph, edge.target(), indent + 1),
                    _ => {}
                }
            }

            let mut result = format!("{}if {}:\n{}", indent_str, condition, then_block);
            if !else_block.is_empty() {
                result.push_str(&format!("{}else:\n{}", indent_str, else_block));
            }
            result
        }

        CodeNodeType::Block => {
            let mut result = String::new();
            for edge in graph.graph.edges(node_idx) {
                if let CodeEdge::Child(_) = edge.weight() {
                    result.push_str(&render_node(graph, edge.target(), indent));
                }
            }
            result
        }

        CodeNodeType::Assignment => {
            let mut target = String::new();
            let mut value = String::new();

            for edge in graph.graph.edges(node_idx) {
                match edge.weight() {
                    CodeEdge::Child(0) => target = render_node(graph, edge.target(), 0).trim().to_string(),
                    CodeEdge::Child(1) => value = render_node(graph, edge.target(), 0).trim().to_string(),
                    _ => {}
                }
            }

            format!("{}{} = {}\n", indent_str, target, value)
        }

        _ => format!("{}# {}\n", indent_str, format!("{:?}", node.node_type)),
    }
}

// =============================================================================
// Simple AST Parser for Python Code
// =============================================================================

/// Parse Python code into a CodeGraph using TreeSitter
fn parse_python_to_code_graph(code: &str) -> CodeGraph {
    // Use TreeSitter for proper AST parsing
    match TreeSitterParser::parse_python(code) {
        Ok(graph) => graph,
        Err(_) => {
            // Fallback to empty graph on parse error
            let mut graph = CodeGraph::with_language(Language::Python);
            let module = graph.add_node(new_code_node(CodeNodeType::Module {
                name: "main".to_string(),
                language: Language::Python,
            }));
            graph.root = Some(module);
            graph
        }
    }
}

/// Parse a simple expression into CodeGraph nodes
fn parse_expression(graph: &mut CodeGraph, expr: &str) -> NodeIndex {
    let expr = expr.trim();

    // Try to parse as integer
    if let Ok(n) = expr.parse::<i64>() {
        return graph.add_node(new_code_node(CodeNodeType::Literal(LiteralValue::Integer(n))));
    }

    // Try to parse as float
    if let Ok(f) = expr.parse::<f64>() {
        return graph.add_node(new_code_node(CodeNodeType::Literal(LiteralValue::Float(f))));
    }

    // Check for binary operators (simple left-to-right)
    for (op_str, op) in [
        (" + ", BinaryOperator::Add),
        (" - ", BinaryOperator::Sub),
        (" * ", BinaryOperator::Mul),
        (" / ", BinaryOperator::Div),
        (" == ", BinaryOperator::Eq),
        (" != ", BinaryOperator::Ne),
        (" <= ", BinaryOperator::Le),
        (" >= ", BinaryOperator::Ge),
        (" < ", BinaryOperator::Lt),
        (" > ", BinaryOperator::Gt),
    ] {
        if let Some(idx) = expr.find(op_str) {
            let left_str = &expr[..idx];
            let right_str = &expr[idx + op_str.len()..];

            let op_node = graph.add_node(new_code_node(CodeNodeType::BinaryOp(op)));
            let left_node = parse_expression(graph, left_str);
            let right_node = parse_expression(graph, right_str);

            graph.add_edge(op_node, left_node, CodeEdge::Child(0));
            graph.add_edge(op_node, right_node, CodeEdge::Child(1));

            return op_node;
        }
    }

    // Check for function call
    if let Some(paren_idx) = expr.find('(') {
        if expr.ends_with(')') {
            let func_name = expr[..paren_idx].to_string();
            let args_str = &expr[paren_idx + 1..expr.len() - 1];
            let args: Vec<&str> = args_str.split(',').collect();

            let call_node = graph.add_node(new_code_node(CodeNodeType::Call {
                function: func_name,
                arg_count: args.len(),
            }));

            for (i, arg) in args.iter().enumerate() {
                let arg_node = parse_expression(graph, arg);
                graph.add_edge(call_node, arg_node, CodeEdge::Child(i));
            }

            return call_node;
        }
    }

    // Default to identifier
    graph.add_node(new_code_node(CodeNodeType::Identifier(expr.to_string())))
}

// =============================================================================
// AST Graph Similarity (Structural Loss)
// =============================================================================

/// Compute AST graph similarity
fn ast_similarity(pred: &CodeGraph, target: &CodeGraph) -> f32 {
    let pred_types: Vec<AstNodeTypeId> = pred.graph.node_indices()
        .map(|idx| AstNodeTypeId::from_code_node_type(&pred.graph[idx].node_type))
        .collect();
    let target_types: Vec<AstNodeTypeId> = target.graph.node_indices()
        .map(|idx| AstNodeTypeId::from_code_node_type(&target.graph[idx].node_type))
        .collect();

    // Node type overlap
    let mut type_matches = 0;
    let max_len = pred_types.len().max(target_types.len());

    for (p, t) in pred_types.iter().zip(target_types.iter()) {
        if p == t {
            type_matches += 1;
        }
    }

    if max_len == 0 {
        return 1.0;
    }

    type_matches as f32 / max_len as f32
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("AST-Level Code Generation Trainer");
    println!("==================================");
    println!("TRUE GRAPHEME: Graph In (prompt) → Graph Out (AST)\n");

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

    // Parse all training targets to CodeGraphs and analyze
    println!("\nParsing target code to AST graphs...");
    let target_graphs: Vec<CodeGraph> = train_examples
        .par_iter()
        .map(|ex| parse_python_to_code_graph(&ex.target))
        .collect();

    let avg_nodes: f32 = target_graphs.iter().map(|g| g.node_count() as f32).sum::<f32>()
        / target_graphs.len() as f32;
    let avg_edges: f32 = target_graphs.iter().map(|g| g.edge_count() as f32).sum::<f32>()
        / target_graphs.len() as f32;

    println!("Average AST: {:.1} nodes, {:.1} edges", avg_nodes, avg_edges);

    // Demo: Show a parsed example
    if args.verbose && !train_examples.is_empty() {
        println!("\n--- Demo: Parse → Render roundtrip ---");
        let sample = &train_examples[0];
        println!("Original target:\n{}", sample.target);

        let parsed = parse_python_to_code_graph(&sample.target);
        println!("\nParsed AST: {} nodes, {} edges", parsed.node_count(), parsed.edge_count());

        let rendered = render_code_graph(&parsed);
        println!("\nRendered back:\n{}", rendered);
        println!("---");
    }

    // Create output directory
    let output_dir = args.output.parent().unwrap_or(&args.output);
    std::fs::create_dir_all(output_dir)?;

    // Model architecture - we need to predict AST node types
    const CHAR_VOCAB_SIZE: usize = 256;
    const EMBED_DIM: usize = 128;
    const HIDDEN_DIM: usize = 256;
    const MAX_AST_NODES: usize = 64;  // Max AST nodes to predict

    // Initialize GraphTransformNet for encoding prompts
    let mut encoder = if let Some(resume_path) = &args.resume {
        println!("\nResuming from {:?}", resume_path);
        let checkpoint = UnifiedCheckpoint::load_from_file(resume_path)?;
        checkpoint.load_module()?
    } else {
        println!("\nInitializing encoder model...");
        println!("  Char vocab: {}", CHAR_VOCAB_SIZE);
        println!("  Embed dim: {}", EMBED_DIM);
        println!("  Hidden dim: {}", HIDDEN_DIM);
        println!("  Max AST nodes: {}", MAX_AST_NODES);

        let mut model = GraphTransformNet::new(CHAR_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, MAX_AST_NODES);
        // Configure for AST output
        model.sabag_pooling = Some(grapheme_core::SabagPooling::new(MAX_AST_NODES, EMBED_DIM));
        model
    };

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.lr);

    // Training loop
    let start = Instant::now();
    let mut best_val_sim = 0.0f32;

    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut epoch_sim = 0.0f32;
        let mut batch_count = 0;

        // Shuffle
        let mut indices: Vec<usize> = (0..train_examples.len()).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        for batch_start in (0..indices.len()).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];

            encoder.zero_grad();

            let mut batch_loss = 0.0f32;
            let mut batch_sim = 0.0f32;

            for &idx in batch_indices {
                let example = &train_examples[idx];
                let target_graph = &target_graphs[idx];

                // Encode input prompt
                let input_graph = GraphemeGraph::from_text(&example.input);
                let (output_graph, pooling_result) = encoder.forward(&input_graph);

                // For now: compute similarity between output graph structure and target AST
                // This is a placeholder - real implementation would predict AST node types
                let sim = ast_similarity(&parse_python_to_code_graph(&encoder.decode(&pooling_result)), target_graph);
                batch_sim += sim;

                // Simple loss: 1 - similarity
                batch_loss += 1.0 - sim;
            }

            let n = batch_indices.len() as f32;
            batch_loss /= n;
            batch_sim /= n;

            encoder.step(args.lr);

            epoch_loss += batch_loss;
            epoch_sim += batch_sim;
            batch_count += 1;

            if args.verbose && batch_count % 20 == 0 {
                println!("    Batch {}: loss={:.4}, sim={:.1}%", batch_count, batch_loss, batch_sim * 100.0);
            }
        }

        let avg_loss = epoch_loss / batch_count as f32;
        let avg_sim = epoch_sim / batch_count as f32;

        // Validation
        let val_result = if let Some(ref val) = val_examples {
            let val_graphs: Vec<CodeGraph> = val.par_iter()
                .map(|ex| parse_python_to_code_graph(&ex.target))
                .collect();

            let mut val_sim = 0.0f32;
            for (i, example) in val.iter().enumerate() {
                let input_graph = GraphemeGraph::from_text(&example.input);
                let (_, pooling_result) = encoder.forward(&input_graph);
                let decoded = encoder.decode(&pooling_result);
                let pred_graph = parse_python_to_code_graph(&decoded);
                val_sim += ast_similarity(&pred_graph, &val_graphs[i]);
            }
            val_sim /= val.len() as f32;

            if val_sim > best_val_sim {
                best_val_sim = val_sim;
                let best_path = args.output.with_file_name("ast_code_best.json");
                encoder.save_to_file(&best_path)?;
                println!("  [NEW BEST] val_sim={:.1}%", val_sim * 100.0);
            }

            Some(val_sim)
        } else {
            None
        };

        let epoch_time = epoch_start.elapsed();

        print!("Epoch {}/{}: loss={:.4}, sim={:.1}%", epoch + 1, args.epochs, avg_loss, avg_sim * 100.0);
        if let Some(vs) = val_result {
            print!(", val_sim={:.1}%", vs * 100.0);
        }
        println!(", time={:.1}s", epoch_time.as_secs_f64());

        // Checkpoint
        if (epoch + 1) % 20 == 0 {
            let ckpt_path = args.output.with_file_name(format!("ast_code_epoch{}.json", epoch + 1));
            encoder.save_to_file(&ckpt_path)?;
            println!("  Checkpoint: {:?}", ckpt_path);
        }
    }

    let total_time = start.elapsed();

    // Save final
    encoder.save_to_file(&args.output)?;

    println!("\nTraining complete in {:.1}s", total_time.as_secs_f64());
    println!("Final model: {:?}", args.output);
    println!("Best validation similarity: {:.1}%", best_val_sim * 100.0);

    // Final demo
    println!("\n--- Demo: AST Code Generation ---");
    let demo_prompt = "def add(a, b):\n    \"\"\" Add two numbers. \"\"\"\n";
    println!("Input prompt:\n{}", demo_prompt);

    let input_graph = GraphemeGraph::from_text(demo_prompt);
    let (_, pooling_result) = encoder.forward(&input_graph);
    let decoded = encoder.decode(&pooling_result);

    println!("Decoded text ({} chars):", decoded.len());
    println!("{}", &decoded[..decoded.len().min(200)]);

    let pred_graph = parse_python_to_code_graph(&decoded);
    println!("\nPredicted AST: {} nodes, {} edges", pred_graph.node_count(), pred_graph.edge_count());

    let rendered = render_code_graph(&pred_graph);
    println!("\nRendered code:\n{}", rendered);

    Ok(())
}
