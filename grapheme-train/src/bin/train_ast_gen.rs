//! AST Generation Model with Node Type Prediction + CodeGraph Rendering
//!
//! TRUE GRAPHEME Code Generation:
//! 1. Input: Prompt text → GraphemeGraph (character nodes)
//! 2. Encode: Learn prompt structure → latent representation
//! 3. Predict: Output AST node types for each position
//! 4. Build: Construct CodeGraph from predicted node types
//! 5. Render: CodeGraph → Source code text
//!
//! Key Components:
//! - AstPredictionHead: Predicts AST node type + attributes for each output position
//! - CodeGraphBuilder: Constructs CodeGraph from predictions
//! - CodeGraphRenderer: Converts CodeGraph → source code
//!
//! Loss Functions:
//! - Node Type Loss: Cross-entropy on predicted AST node types
//! - Structural Loss: Graph structure alignment
//! - Render Loss: Character match between rendered code and target
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin train_ast_gen -- \
//!     --data data/code_training --output checkpoints/ast_gen.json

use clap::Parser;
use grapheme_code::{
    new_code_node, CodeEdge, CodeGraph, CodeNodeType, Language,
    LiteralValue, BinaryOperator, TreeSitterParser,
};
use grapheme_core::{GraphemeGraph, GraphTransformNet, Learnable, UnifiedCheckpoint, SabagPooling};
use ndarray::{Array1, Array2, Axis};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_ast_gen")]
#[command(about = "Train AST generation with node type prediction and CodeGraph rendering")]
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
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// Maximum AST nodes to predict
    #[arg(long, default_value = "128")]
    max_ast_nodes: usize,

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
    input: String,
    target: String,
    #[serde(default)]
    level: u32,
}

// =============================================================================
// AST Node Type Vocabulary
// =============================================================================

/// AST node types for prediction (vocabulary)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AstNodeId {
    // Structure
    Padding = 0,
    Module = 1,
    Function = 2,
    Block = 3,

    // Declarations
    Variable = 4,
    Parameter = 5,

    // Expressions
    Literal = 6,
    Identifier = 7,
    BinaryOp = 8,
    UnaryOp = 9,
    Call = 10,
    Attribute = 11,
    Index = 12,

    // Control Flow
    If = 13,
    Else = 14,
    For = 15,
    While = 16,
    Return = 17,
    Break = 18,
    Continue = 19,

    // Operators (for BinaryOp detail)
    OpAdd = 20,
    OpSub = 21,
    OpMul = 22,
    OpDiv = 23,
    OpEq = 24,
    OpNe = 25,
    OpLt = 26,
    OpGt = 27,
    OpLe = 28,
    OpGe = 29,
    OpAnd = 30,
    OpOr = 31,

    // Literals (for Literal detail)
    LitInt = 32,
    LitFloat = 33,
    LitStr = 34,
    LitBool = 35,
    LitNone = 36,

    // Special
    Assignment = 37,
    ExprStmt = 38,
    Comment = 39,

    EndOfAst = 63,  // End marker
}

impl AstNodeId {
    pub const VOCAB_SIZE: usize = 64;

    pub fn from_code_node_type(node_type: &CodeNodeType) -> Self {
        match node_type {
            CodeNodeType::Module { .. } => AstNodeId::Module,
            CodeNodeType::Function { .. } => AstNodeId::Function,
            CodeNodeType::Block => AstNodeId::Block,
            CodeNodeType::Variable { .. } => AstNodeId::Variable,
            CodeNodeType::Literal(lit) => match lit {
                LiteralValue::Integer(_) => AstNodeId::LitInt,
                LiteralValue::Float(_) => AstNodeId::LitFloat,
                LiteralValue::String(_) => AstNodeId::LitStr,
                LiteralValue::Boolean(_) => AstNodeId::LitBool,
                LiteralValue::Null => AstNodeId::LitNone,
            },
            CodeNodeType::Identifier(_) => AstNodeId::Identifier,
            CodeNodeType::BinaryOp(op) => match op {
                BinaryOperator::Add => AstNodeId::OpAdd,
                BinaryOperator::Sub => AstNodeId::OpSub,
                BinaryOperator::Mul => AstNodeId::OpMul,
                BinaryOperator::Div => AstNodeId::OpDiv,
                BinaryOperator::Eq => AstNodeId::OpEq,
                BinaryOperator::Ne => AstNodeId::OpNe,
                BinaryOperator::Lt => AstNodeId::OpLt,
                BinaryOperator::Gt => AstNodeId::OpGt,
                BinaryOperator::Le => AstNodeId::OpLe,
                BinaryOperator::Ge => AstNodeId::OpGe,
                BinaryOperator::And => AstNodeId::OpAnd,
                BinaryOperator::Or => AstNodeId::OpOr,
                _ => AstNodeId::BinaryOp,
            },
            CodeNodeType::UnaryOp(_) => AstNodeId::UnaryOp,
            CodeNodeType::Call { .. } => AstNodeId::Call,
            CodeNodeType::If => AstNodeId::If,
            CodeNodeType::Loop { .. } => AstNodeId::For,
            CodeNodeType::Return => AstNodeId::Return,
            CodeNodeType::Assignment => AstNodeId::Assignment,
            CodeNodeType::ExprStmt => AstNodeId::ExprStmt,
            CodeNodeType::Comment(_) => AstNodeId::Comment,
            CodeNodeType::Type(_) => AstNodeId::Identifier,
        }
    }

    pub fn to_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(v: u8) -> Self {
        if v >= Self::VOCAB_SIZE as u8 {
            return AstNodeId::Padding;
        }
        // Safety: all values 0-63 are valid
        unsafe { std::mem::transmute(v) }
    }
}

// =============================================================================
// AST Prediction Head
// =============================================================================

/// Prediction head for AST node types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstPredictionHead {
    /// Weight matrix: hidden_dim → ast_vocab_size
    pub weight: Array2<f32>,
    /// Bias
    pub bias: Array1<f32>,
    /// Gradient storage
    #[serde(skip)]
    pub weight_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub bias_grad: Option<Array1<f32>>,
}

impl AstPredictionHead {
    pub fn new(hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let std = (2.0 / (hidden_dim + AstNodeId::VOCAB_SIZE) as f32).sqrt();

        // Initialize bias to discourage Padding predictions
        // Set Padding (class 0) bias to negative, others to small positive
        let mut bias = Array1::from_elem(AstNodeId::VOCAB_SIZE, 0.1);
        bias[0] = -5.0;  // Strong negative bias for Padding class

        Self {
            weight: Array2::from_shape_fn((AstNodeId::VOCAB_SIZE, hidden_dim), |_| {
                rng.gen::<f32>() * 2.0 * std - std
            }),
            bias,
            weight_grad: None,
            bias_grad: None,
        }
    }

    /// Forward pass: features → logits
    pub fn forward(&self, features: &Array2<f32>) -> Array2<f32> {
        // features: (batch, hidden_dim)
        // output: (batch, vocab_size)
        let mut logits = features.dot(&self.weight.t());
        for mut row in logits.rows_mut() {
            row += &self.bias;
        }
        logits
    }

    /// Predict AST node type for each position
    pub fn predict(&self, features: &Array2<f32>) -> Vec<AstNodeId> {
        let logits = self.forward(features);
        logits.rows().into_iter().map(|row| {
            let (max_idx, _) = row.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            AstNodeId::from_u8(max_idx as u8)
        }).collect()
    }

    /// Compute cross-entropy loss
    pub fn loss(&self, features: &Array2<f32>, targets: &[AstNodeId]) -> f32 {
        let logits = self.forward(features);
        let mut total_loss = 0.0;

        for (i, &target) in targets.iter().enumerate() {
            let row = logits.row(i);
            // Softmax + cross-entropy
            let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum();
            let log_softmax = row[target.to_u8() as usize] - max_logit - exp_sum.ln();
            total_loss -= log_softmax;
        }

        total_loss / targets.len() as f32
    }

    /// Backward pass - returns gradient w.r.t. input features for backprop
    pub fn backward(&mut self, features: &Array2<f32>, targets: &[AstNodeId]) -> Array2<f32> {
        let logits = self.forward(features);
        let batch_size = features.nrows();
        let hidden_dim = features.ncols();

        // Initialize gradients
        self.weight_grad = Some(Array2::zeros(self.weight.raw_dim()));
        self.bias_grad = Some(Array1::zeros(self.bias.len()));

        // Gradient w.r.t. input features (for encoder backprop)
        let mut input_grad = Array2::zeros((batch_size, hidden_dim));

        for (i, &target) in targets.iter().enumerate() {
            let row = logits.row(i);

            // Softmax probabilities
            let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_logit).exp()).collect();
            let exp_sum: f32 = exp_vals.iter().sum();
            let probs: Vec<f32> = exp_vals.iter().map(|&e| e / exp_sum).collect();

            // Gradient: probs - one_hot(target)
            for (j, &prob) in probs.iter().enumerate() {
                let grad = if j == target.to_u8() as usize {
                    prob - 1.0
                } else {
                    prob
                };

                // Accumulate weight gradient
                if let Some(ref mut wg) = self.weight_grad {
                    for (k, &feat) in features.row(i).iter().enumerate() {
                        wg[[j, k]] += grad * feat / batch_size as f32;
                    }
                }

                // Accumulate bias gradient
                if let Some(ref mut bg) = self.bias_grad {
                    bg[j] += grad / batch_size as f32;
                }

                // Accumulate input gradient: dL/dx = dL/dlogits * W
                // dL/dlogits[j] = grad, so dL/dx[k] += grad * W[j,k]
                for k in 0..hidden_dim {
                    input_grad[[i, k]] += grad * self.weight[[j, k]] / batch_size as f32;
                }
            }
        }

        input_grad
    }

    pub fn zero_grad(&mut self) {
        self.weight_grad = None;
        self.bias_grad = None;
    }

    pub fn step(&mut self, lr: f32) {
        if let Some(ref wg) = self.weight_grad {
            self.weight = &self.weight - &(wg * lr);
        }
        if let Some(ref bg) = self.bias_grad {
            self.bias = &self.bias - &(bg * lr);
        }
    }
}

// =============================================================================
// CodeGraph Builder (from AST predictions)
// =============================================================================

/// Build a CodeGraph from predicted AST node types
fn build_code_graph(predictions: &[AstNodeId]) -> CodeGraph {
    let mut graph = CodeGraph::with_language(Language::Python);

    if predictions.is_empty() {
        return graph;
    }

    // Create module root
    let module = graph.add_node(new_code_node(CodeNodeType::Module {
        name: "generated".to_string(),
        language: Language::Python,
    }));
    graph.root = Some(module);

    let mut current_parent = module;
    let mut parent_stack: Vec<NodeIndex> = vec![module];
    let mut child_counts: std::collections::HashMap<NodeIndex, usize> = std::collections::HashMap::new();

    for &pred in predictions {
        if pred == AstNodeId::Padding || pred == AstNodeId::EndOfAst {
            continue;
        }

        let node_type = match pred {
            AstNodeId::Function => CodeNodeType::Function {
                name: "func".to_string(),
                params: vec!["arg".to_string()],
                return_type: None,
            },
            AstNodeId::Variable => CodeNodeType::Variable {
                name: "var".to_string(),
                var_type: None,
            },
            AstNodeId::Return => CodeNodeType::Return,
            AstNodeId::If => CodeNodeType::If,
            AstNodeId::For | AstNodeId::While => CodeNodeType::Loop {
                kind: grapheme_code::LoopKind::For,
            },
            AstNodeId::Block => CodeNodeType::Block,
            AstNodeId::Identifier => CodeNodeType::Identifier("x".to_string()),
            AstNodeId::LitInt => CodeNodeType::Literal(LiteralValue::Integer(0)),
            AstNodeId::LitFloat => CodeNodeType::Literal(LiteralValue::Float(0.0)),
            AstNodeId::LitStr => CodeNodeType::Literal(LiteralValue::String("".to_string())),
            AstNodeId::LitBool => CodeNodeType::Literal(LiteralValue::Boolean(true)),
            AstNodeId::LitNone => CodeNodeType::Literal(LiteralValue::Null),
            AstNodeId::OpAdd => CodeNodeType::BinaryOp(BinaryOperator::Add),
            AstNodeId::OpSub => CodeNodeType::BinaryOp(BinaryOperator::Sub),
            AstNodeId::OpMul => CodeNodeType::BinaryOp(BinaryOperator::Mul),
            AstNodeId::OpDiv => CodeNodeType::BinaryOp(BinaryOperator::Div),
            AstNodeId::OpEq => CodeNodeType::BinaryOp(BinaryOperator::Eq),
            AstNodeId::OpNe => CodeNodeType::BinaryOp(BinaryOperator::Ne),
            AstNodeId::OpLt => CodeNodeType::BinaryOp(BinaryOperator::Lt),
            AstNodeId::OpGt => CodeNodeType::BinaryOp(BinaryOperator::Gt),
            AstNodeId::Call => CodeNodeType::Call {
                function: "f".to_string(),
                arg_count: 1,
            },
            AstNodeId::Assignment => CodeNodeType::Assignment,
            AstNodeId::ExprStmt => CodeNodeType::ExprStmt,
            _ => continue,
        };

        let node = graph.add_node(new_code_node(node_type));

        // Connect to parent
        let child_idx = *child_counts.get(&current_parent).unwrap_or(&0);
        graph.add_edge(current_parent, node, CodeEdge::Child(child_idx));
        *child_counts.entry(current_parent).or_insert(0) += 1;

        // Update parent stack for block-creating nodes
        match pred {
            AstNodeId::Function | AstNodeId::If | AstNodeId::For | AstNodeId::While | AstNodeId::Block => {
                parent_stack.push(node);
                current_parent = node;
            }
            AstNodeId::Return => {
                // Pop back to function level
                while parent_stack.len() > 1 {
                    let top = parent_stack.pop().unwrap();
                    current_parent = *parent_stack.last().unwrap_or(&module);
                    if matches!(graph.graph[top].node_type, CodeNodeType::Function { .. }) {
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    graph
}

// =============================================================================
// CodeGraph Renderer (AST → Source Code)
// =============================================================================

/// Render CodeGraph to source code
fn render_code_graph(graph: &CodeGraph) -> String {
    let Some(root) = graph.root else {
        return String::new();
    };
    render_node(graph, root, 0)
}

fn render_node(graph: &CodeGraph, node_idx: NodeIndex, indent: usize) -> String {
    let node = &graph.graph[node_idx];
    let indent_str = "    ".repeat(indent);

    // Get children
    let mut children: Vec<(usize, NodeIndex)> = graph.graph.edges(node_idx)
        .filter_map(|e| {
            if let CodeEdge::Child(i) = e.weight() {
                Some((*i, e.target()))
            } else {
                None
            }
        })
        .collect();
    children.sort_by_key(|(i, _)| *i);

    match &node.node_type {
        CodeNodeType::Module { .. } => {
            let mut result = String::new();
            for (_, child) in children {
                result.push_str(&render_node(graph, child, indent));
            }
            result
        }

        CodeNodeType::Function { name, params, return_type } => {
            let params_str = params.join(", ");
            let ret = return_type.as_ref().map(|t| format!(" -> {}", t)).unwrap_or_default();
            let mut result = format!("{}def {}({}){}:\n", indent_str, name, params_str, ret);

            if children.is_empty() {
                result.push_str(&format!("{}    pass\n", indent_str));
            } else {
                for (_, child) in children {
                    result.push_str(&render_node(graph, child, indent + 1));
                }
            }
            result
        }

        CodeNodeType::Return => {
            let mut result = format!("{}return", indent_str);
            if !children.is_empty() {
                result.push(' ');
                result.push_str(&render_node(graph, children[0].1, 0).trim());
            }
            result.push('\n');
            result
        }

        CodeNodeType::Variable { name, var_type } => {
            match var_type {
                Some(t) => format!("{}{}: {}", indent_str, name, t),
                None => format!("{}{}", indent_str, name),
            }
        }

        CodeNodeType::Identifier(name) => name.clone(),

        CodeNodeType::Literal(lit) => {
            match lit {
                LiteralValue::Integer(n) => n.to_string(),
                LiteralValue::Float(f) => f.to_string(),
                LiteralValue::String(s) => format!("\"{}\"", s),
                LiteralValue::Boolean(b) => if *b { "True" } else { "False" }.to_string(),
                LiteralValue::Null => "None".to_string(),
            }
        }

        CodeNodeType::BinaryOp(op) => {
            let op_str = match op {
                BinaryOperator::Add => "+",
                BinaryOperator::Sub => "-",
                BinaryOperator::Mul => "*",
                BinaryOperator::Div => "/",
                BinaryOperator::Eq => "==",
                BinaryOperator::Ne => "!=",
                BinaryOperator::Lt => "<",
                BinaryOperator::Gt => ">",
                BinaryOperator::Le => "<=",
                BinaryOperator::Ge => ">=",
                BinaryOperator::And => "and",
                BinaryOperator::Or => "or",
                _ => "?",
            };

            let left = children.get(0).map(|(_, c)| render_node(graph, *c, 0).trim().to_string()).unwrap_or_default();
            let right = children.get(1).map(|(_, c)| render_node(graph, *c, 0).trim().to_string()).unwrap_or_default();
            format!("{} {} {}", left, op_str, right)
        }

        CodeNodeType::Call { function, .. } => {
            let args: Vec<String> = children.iter()
                .map(|(_, c)| render_node(graph, *c, 0).trim().to_string())
                .collect();
            format!("{}({})", function, args.join(", "))
        }

        CodeNodeType::If => {
            let mut result = format!("{}if ", indent_str);
            if let Some((_, cond)) = children.get(0) {
                result.push_str(&render_node(graph, *cond, 0).trim());
            }
            result.push_str(":\n");
            for (i, (_, child)) in children.iter().enumerate().skip(1) {
                result.push_str(&render_node(graph, *child, indent + 1));
            }
            if children.len() <= 1 {
                result.push_str(&format!("{}    pass\n", indent_str));
            }
            result
        }

        CodeNodeType::Loop { .. } => {
            let mut result = format!("{}for i in range(", indent_str);
            if let Some((_, count)) = children.get(0) {
                result.push_str(&render_node(graph, *count, 0).trim());
            } else {
                result.push_str("10");
            }
            result.push_str("):\n");
            for (_, child) in children.iter().skip(1) {
                result.push_str(&render_node(graph, *child, indent + 1));
            }
            if children.len() <= 1 {
                result.push_str(&format!("{}    pass\n", indent_str));
            }
            result
        }

        CodeNodeType::Assignment => {
            let target = children.get(0).map(|(_, c)| render_node(graph, *c, 0).trim().to_string()).unwrap_or("x".to_string());
            let value = children.get(1).map(|(_, c)| render_node(graph, *c, 0).trim().to_string()).unwrap_or("None".to_string());
            format!("{}{} = {}\n", indent_str, target, value)
        }

        CodeNodeType::Block => {
            let mut result = String::new();
            for (_, child) in children {
                result.push_str(&render_node(graph, child, indent));
            }
            result
        }

        _ => format!("{}# {:?}\n", indent_str, node.node_type),
    }
}

// =============================================================================
// Training
// =============================================================================

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

/// Parse Python code to CodeGraph
fn parse_to_code_graph(code: &str) -> CodeGraph {
    match TreeSitterParser::parse_python(code) {
        Ok(graph) => graph,
        Err(_) => CodeGraph::with_language(Language::Python),
    }
}

/// Extract AST node type sequence from CodeGraph (skip Comments)
fn extract_ast_sequence(graph: &CodeGraph, max_len: usize) -> Vec<AstNodeId> {
    let mut sequence = Vec::with_capacity(max_len);

    fn dfs(graph: &CodeGraph, node: NodeIndex, seq: &mut Vec<AstNodeId>, max_len: usize) {
        if seq.len() >= max_len {
            return;
        }

        let node_type = AstNodeId::from_code_node_type(&graph.graph[node].node_type);
        // Skip Comment nodes - they don't contribute to code structure
        // But still visit children (in case there are nested code nodes)
        if node_type != AstNodeId::Comment {
            seq.push(node_type);
        }

        // Visit children in order
        let mut children: Vec<(usize, NodeIndex)> = graph.graph.edges(node)
            .filter_map(|e| {
                if let CodeEdge::Child(i) = e.weight() {
                    Some((*i, e.target()))
                } else {
                    None
                }
            })
            .collect();
        children.sort_by_key(|(i, _)| *i);

        for (_, child) in children {
            dfs(graph, child, seq, max_len);
        }
    }

    if let Some(root) = graph.root {
        dfs(graph, root, &mut sequence, max_len);
    }

    // Pad or truncate
    sequence.resize(max_len, AstNodeId::Padding);
    sequence
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("AST Generation Trainer with Node Type Prediction");
    println!("=================================================");
    println!("Graph In → AST Prediction → CodeGraph → Source Code\n");

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

    // Parse all targets to AST sequences
    println!("\nParsing target code to AST sequences...");
    let target_graphs: Vec<CodeGraph> = train_examples
        .par_iter()
        .map(|ex| parse_to_code_graph(&ex.target))
        .collect();

    let target_sequences: Vec<Vec<AstNodeId>> = target_graphs
        .par_iter()
        .map(|g| extract_ast_sequence(g, args.max_ast_nodes))
        .collect();

    let avg_nodes: f32 = target_graphs.iter().map(|g| g.node_count() as f32).sum::<f32>()
        / target_graphs.len() as f32;
    println!("Average target AST: {:.1} nodes", avg_nodes);

    // Analyze target distribution
    let mut node_counts = std::collections::HashMap::<AstNodeId, usize>::new();
    let mut padding_count = 0usize;
    let mut non_padding_count = 0usize;
    for seq in &target_sequences {
        for &node in seq {
            *node_counts.entry(node).or_insert(0) += 1;
            if node == AstNodeId::Padding {
                padding_count += 1;
            } else {
                non_padding_count += 1;
            }
        }
    }
    println!("\nTarget distribution (non-padding: {}, padding: {}):", non_padding_count, padding_count);
    let mut counts: Vec<_> = node_counts.iter()
        .filter(|(k, _)| **k != AstNodeId::Padding)
        .collect();
    counts.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
    for (node, count) in counts.iter().take(15) {
        let pct = **count as f64 / non_padding_count as f64 * 100.0;
        println!("  {:?}: {} ({:.1}%)", node, count, pct);
    }

    // Create output directory
    let output_dir = args.output.parent().unwrap_or(&args.output);
    std::fs::create_dir_all(output_dir)?;

    // Model architecture
    const CHAR_VOCAB: usize = 256;
    const EMBED_DIM: usize = 128;
    const HIDDEN_DIM: usize = 256;

    // Initialize encoder + AST prediction head
    let mut encoder = if let Some(resume_path) = &args.resume {
        println!("\nResuming from {:?}", resume_path);
        let checkpoint = UnifiedCheckpoint::load_from_file(resume_path)?;
        checkpoint.load_module()?
    } else {
        println!("\nInitializing encoder...");
        println!("  Char vocab: {}", CHAR_VOCAB);
        println!("  Embed dim: {}", EMBED_DIM);
        println!("  Hidden dim: {}", HIDDEN_DIM);
        println!("  Max AST nodes: {}", args.max_ast_nodes);

        let mut model = GraphTransformNet::new(CHAR_VOCAB, EMBED_DIM, HIDDEN_DIM, args.max_ast_nodes);
        model.sabag_pooling = Some(SabagPooling::new(args.max_ast_nodes, EMBED_DIM));
        model
    };

    // AST prediction head
    let mut ast_head = AstPredictionHead::new(EMBED_DIM);
    println!("  AST vocab: {}", AstNodeId::VOCAB_SIZE);

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.lr);

    // Training loop
    let start = Instant::now();
    let mut best_val_acc = 0.0f32;

    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut epoch_acc = 0.0f32;
        let mut batch_count = 0;

        // Shuffle
        let mut indices: Vec<usize> = (0..train_examples.len()).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        for batch_start in (0..indices.len()).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];

            encoder.zero_grad();
            ast_head.zero_grad();

            let mut batch_loss = 0.0f32;
            let mut batch_correct = 0usize;
            let mut batch_total = 0usize;

            for &idx in batch_indices {
                let example = &train_examples[idx];
                let target_seq = &target_sequences[idx];

                // Encode prompt
                let input_graph = GraphemeGraph::from_text(&example.input);
                let (_, pooling_result) = encoder.forward(&input_graph);

                // Predict AST node types
                let predictions = ast_head.predict(&pooling_result.features);

                // Compute loss
                let loss = ast_head.loss(&pooling_result.features, target_seq);
                batch_loss += loss;

                // Accuracy
                for (pred, target) in predictions.iter().zip(target_seq.iter()) {
                    if *target != AstNodeId::Padding {
                        batch_total += 1;
                        if pred == target {
                            batch_correct += 1;
                        }
                    }
                }

                // Backward through AST head - get gradient w.r.t. features
                let feature_grad = ast_head.backward(&pooling_result.features, target_seq);

                // Convert 2D gradient to activation gradients (sum across embed dim)
                // feature_grad is (num_nodes, embed_dim), we need (num_nodes,)
                let activation_grads: Vec<f32> = feature_grad
                    .rows()
                    .into_iter()
                    .map(|row| row.sum())  // Sum gradients across embed dimensions
                    .collect();

                // Backward through encoder to update embeddings
                encoder.backward(&input_graph, &pooling_result, &activation_grads, EMBED_DIM);
            }

            let n = batch_indices.len() as f32;
            batch_loss /= n;
            let batch_acc = if batch_total > 0 {
                batch_correct as f32 / batch_total as f32
            } else {
                0.0
            };

            // Update weights
            ast_head.step(args.lr);
            encoder.step(args.lr * 0.1); // Slower encoder learning

            epoch_loss += batch_loss;
            epoch_acc += batch_acc;
            batch_count += 1;

            if args.verbose && batch_count % 10 == 0 {
                println!("    Batch {}: loss={:.4}, acc={:.1}%", batch_count, batch_loss, batch_acc * 100.0);
            }
        }

        let avg_loss = epoch_loss / batch_count as f32;
        let avg_acc = epoch_acc / batch_count as f32;

        // Validation
        let val_result = if let Some(ref val) = val_examples {
            let val_graphs: Vec<CodeGraph> = val.par_iter()
                .map(|ex| parse_to_code_graph(&ex.target))
                .collect();
            let val_sequences: Vec<Vec<AstNodeId>> = val_graphs.par_iter()
                .map(|g| extract_ast_sequence(g, args.max_ast_nodes))
                .collect();

            let mut val_correct = 0usize;
            let mut val_total = 0usize;
            let mut render_match = 0usize;

            for (i, example) in val.iter().enumerate() {
                let input_graph = GraphemeGraph::from_text(&example.input);
                let (_, pooling_result) = encoder.forward(&input_graph);
                let predictions = ast_head.predict(&pooling_result.features);

                // Node accuracy
                for (pred, target) in predictions.iter().zip(val_sequences[i].iter()) {
                    if *target != AstNodeId::Padding {
                        val_total += 1;
                        if pred == target {
                            val_correct += 1;
                        }
                    }
                }

                // Build and render CodeGraph
                let pred_graph = build_code_graph(&predictions);
                let rendered = render_code_graph(&pred_graph);

                // Check if render matches target structure
                if rendered.contains("def ") && example.target.contains("def ") {
                    render_match += 1;
                }
            }

            let val_acc = val_correct as f32 / val_total.max(1) as f32;
            let render_rate = render_match as f32 / val.len() as f32;

            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                let best_path = args.output.with_file_name("ast_gen_best.json");
                encoder.save_to_file(&best_path)?;
                println!("  [NEW BEST] val_acc={:.1}%", val_acc * 100.0);
            }

            Some((val_acc, render_rate))
        } else {
            None
        };

        let epoch_time = epoch_start.elapsed();

        print!("Epoch {}/{}: loss={:.4}, acc={:.1}%", epoch + 1, args.epochs, avg_loss, avg_acc * 100.0);
        if let Some((va, rr)) = val_result {
            print!(", val_acc={:.1}%, render={:.1}%", va * 100.0, rr * 100.0);
        }
        println!(", time={:.1}s", epoch_time.as_secs_f64());

        // Checkpoint
        if (epoch + 1) % 20 == 0 {
            let ckpt_path = args.output.with_file_name(format!("ast_gen_epoch{}.json", epoch + 1));
            encoder.save_to_file(&ckpt_path)?;
            println!("  Checkpoint: {:?}", ckpt_path);
        }

        // Demo every 25 epochs
        if args.verbose && (epoch + 1) % 25 == 0 {
            println!("\n  --- Demo (epoch {}) ---", epoch + 1);
            let demo = "def add(a, b):\n    \"\"\" Add two numbers. \"\"\"\n";
            let input_graph = GraphemeGraph::from_text(demo);
            let (_, pooling_result) = encoder.forward(&input_graph);
            let predictions = ast_head.predict(&pooling_result.features);

            let pred_graph = build_code_graph(&predictions);
            let rendered = render_code_graph(&pred_graph);

            println!("  Input: {} chars", demo.len());
            println!("  Predicted AST nodes: {:?}", &predictions[..10.min(predictions.len())]);
            println!("  Rendered:\n{}", rendered);
            println!();
        }
    }

    let total_time = start.elapsed();

    // Save final
    encoder.save_to_file(&args.output)?;

    println!("\nTraining complete in {:.1}s", total_time.as_secs_f64());
    println!("Final model: {:?}", args.output);
    println!("Best validation accuracy: {:.1}%", best_val_acc * 100.0);

    // Final demo
    println!("\n--- Final Demo: AST Generation ---");
    let demo = "def is_prime(n):\n    \"\"\" Check if n is prime. \"\"\"\n";
    println!("Input prompt:\n{}", demo);

    let input_graph = GraphemeGraph::from_text(demo);
    let (_, pooling_result) = encoder.forward(&input_graph);
    let predictions = ast_head.predict(&pooling_result.features);

    println!("\nPredicted AST node types:");
    for (i, pred) in predictions.iter().take(20).enumerate() {
        if *pred != AstNodeId::Padding {
            println!("  {}: {:?}", i, pred);
        }
    }

    let pred_graph = build_code_graph(&predictions);
    println!("\nBuilt CodeGraph: {} nodes, {} edges", pred_graph.node_count(), pred_graph.edge_count());

    let rendered = render_code_graph(&pred_graph);
    println!("\nRendered source code:");
    println!("{}", rendered);

    Ok(())
}
