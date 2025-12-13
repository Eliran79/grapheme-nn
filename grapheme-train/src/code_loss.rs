//! Structural Loss Improvements for Code Graphs (backend-230)
//!
//! Provides code-aware structural loss functions for comparing code graphs.
//! Uses polynomial-time algorithms (no NP-hard implementations).
//!
//! **GRAPHEME Protocol**: Uses LeakyReLU (α=0.01), DynamicXavier, Adam (lr=0.001)

use grapheme_code::{CodeEdge, CodeGraph, CodeNode, BinaryOperator, UnaryOperator, LoopKind};
use grapheme_core::DagNN;
use std::collections::HashMap;

/// LeakyReLU constant (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

// ============================================================================
// Code Node Type Categories
// ============================================================================

/// Categories of code node types for comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeCategory {
    /// Module/file structure
    Structure,
    /// Function definitions
    Function,
    /// Variables and assignments
    Variable,
    /// Literal values
    Literal,
    /// Operations (binary, unary)
    Operation,
    /// Control flow (if, loop, return)
    ControlFlow,
    /// Function calls
    Call,
    /// Type annotations
    Type,
    /// Comments
    Comment,
    /// Other/unknown
    Other,
}

impl NodeCategory {
    /// Get category for a code node
    pub fn from_node(node: &CodeNode) -> Self {
        match node {
            CodeNode::Module { .. } | CodeNode::Block => NodeCategory::Structure,
            CodeNode::Function { .. } => NodeCategory::Function,
            CodeNode::Variable { .. } | CodeNode::Identifier(_) => NodeCategory::Variable,
            CodeNode::Literal(_) => NodeCategory::Literal,
            CodeNode::BinaryOp(_) | CodeNode::UnaryOp(_) | CodeNode::Assignment | CodeNode::ExprStmt => NodeCategory::Operation,
            CodeNode::If | CodeNode::Loop { .. } | CodeNode::Return => NodeCategory::ControlFlow,
            CodeNode::Call { .. } => NodeCategory::Call,
            CodeNode::Type(_) => NodeCategory::Type,
            CodeNode::Comment(_) => NodeCategory::Comment,
        }
    }

    /// Get all categories
    pub fn all() -> &'static [NodeCategory] {
        &[
            NodeCategory::Structure,
            NodeCategory::Function,
            NodeCategory::Variable,
            NodeCategory::Literal,
            NodeCategory::Operation,
            NodeCategory::ControlFlow,
            NodeCategory::Call,
            NodeCategory::Type,
            NodeCategory::Comment,
            NodeCategory::Other,
        ]
    }
}

// ============================================================================
// Edge Type Categories
// ============================================================================

/// Categories of edge types for comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeCategory {
    /// Structural edges (parent-child, sibling)
    Structural,
    /// Control flow edges
    ControlFlow,
    /// Data flow edges
    DataFlow,
    /// Type edges
    Type,
}

impl EdgeCategory {
    /// Get category for a code edge
    pub fn from_edge(edge: &CodeEdge) -> Self {
        match edge {
            CodeEdge::Child(_) | CodeEdge::Next => EdgeCategory::Structural,
            CodeEdge::ControlFlow => EdgeCategory::ControlFlow,
            CodeEdge::DataFlow | CodeEdge::DefUse => EdgeCategory::DataFlow,
            CodeEdge::HasType => EdgeCategory::Type,
        }
    }

    /// Get all categories
    pub fn all() -> &'static [EdgeCategory] {
        &[
            EdgeCategory::Structural,
            EdgeCategory::ControlFlow,
            EdgeCategory::DataFlow,
            EdgeCategory::Type,
        ]
    }
}

// ============================================================================
// Code Structural Loss Configuration
// ============================================================================

/// Configuration for code-aware structural loss
#[derive(Debug, Clone)]
pub struct CodeLossConfig {
    /// Weight for node type distribution loss
    pub node_type_weight: f32,
    /// Weight for edge type distribution loss
    pub edge_type_weight: f32,
    /// Weight for control flow structure loss
    pub control_flow_weight: f32,
    /// Weight for function signature loss
    pub function_weight: f32,
    /// Weight for depth distribution loss
    pub depth_weight: f32,
    /// Maximum depth to consider
    pub max_depth: usize,
}

impl Default for CodeLossConfig {
    fn default() -> Self {
        Self {
            node_type_weight: 1.0,
            edge_type_weight: 0.8,
            control_flow_weight: 1.2,
            function_weight: 1.5,
            depth_weight: 0.5,
            max_depth: 10,
        }
    }
}

// ============================================================================
// Code Structural Loss
// ============================================================================

/// Detailed code structural loss breakdown
#[derive(Debug, Clone, Default)]
pub struct CodeStructuralLoss {
    /// Total weighted loss
    pub total: f32,
    /// Node type distribution loss
    pub node_type_loss: f32,
    /// Edge type distribution loss
    pub edge_type_loss: f32,
    /// Control flow structure loss
    pub control_flow_loss: f32,
    /// Function signature loss
    pub function_loss: f32,
    /// Depth distribution loss
    pub depth_loss: f32,
}

/// Compute code-aware structural loss between predicted and target code graphs
///
/// Uses polynomial-time algorithms:
/// - Node type histograms (O(n))
/// - Edge type histograms (O(e))
/// - Control flow pattern matching (O(n))
/// - Function signature comparison (O(f) where f = number of functions)
pub fn code_structural_loss(
    predicted: &CodeGraph,
    target: &CodeGraph,
    config: &CodeLossConfig,
) -> CodeStructuralLoss {
    // Node type distribution loss
    let pred_node_hist = node_type_histogram(predicted);
    let target_node_hist = node_type_histogram(target);
    let node_type_loss = histogram_distance(&pred_node_hist, &target_node_hist);

    // Edge type distribution loss
    let pred_edge_hist = edge_type_histogram(predicted);
    let target_edge_hist = edge_type_histogram(target);
    let edge_type_loss = histogram_distance(&pred_edge_hist, &target_edge_hist);

    // Control flow structure loss
    let control_flow_loss = control_flow_distance(predicted, target);

    // Function signature loss
    let function_loss = function_signature_distance(predicted, target);

    // Depth distribution loss
    let pred_depth_hist = depth_histogram(predicted, config.max_depth);
    let target_depth_hist = depth_histogram(target, config.max_depth);
    let depth_loss = histogram_distance(&pred_depth_hist, &target_depth_hist);

    // Weighted total
    let total = config.node_type_weight * node_type_loss
        + config.edge_type_weight * edge_type_loss
        + config.control_flow_weight * control_flow_loss
        + config.function_weight * function_loss
        + config.depth_weight * depth_loss;

    // Normalize by total weight
    let total_weight = config.node_type_weight
        + config.edge_type_weight
        + config.control_flow_weight
        + config.function_weight
        + config.depth_weight;

    CodeStructuralLoss {
        total: total / total_weight,
        node_type_loss,
        edge_type_loss,
        control_flow_loss,
        function_loss,
        depth_loss,
    }
}

/// Compute node type histogram for a code graph
fn node_type_histogram(graph: &CodeGraph) -> Vec<f32> {
    let categories = NodeCategory::all();
    let mut counts = vec![0.0; categories.len()];
    let mut total = 0.0;

    for node_idx in graph.graph.node_indices() {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let cat = NodeCategory::from_node(node);
            if let Some(idx) = categories.iter().position(|c| *c == cat) {
                counts[idx] += 1.0;
                total += 1.0;
            }
        }
    }

    // Normalize
    if total > 0.0 {
        for c in &mut counts {
            *c /= total;
        }
    }

    counts
}

/// Compute edge type histogram for a code graph
fn edge_type_histogram(graph: &CodeGraph) -> Vec<f32> {
    let categories = EdgeCategory::all();
    let mut counts = vec![0.0; categories.len()];
    let mut total = 0.0;

    for edge_idx in graph.graph.edge_indices() {
        if let Some(edge) = graph.graph.edge_weight(edge_idx) {
            let cat = EdgeCategory::from_edge(edge);
            if let Some(idx) = categories.iter().position(|c| *c == cat) {
                counts[idx] += 1.0;
                total += 1.0;
            }
        }
    }

    // Normalize
    if total > 0.0 {
        for c in &mut counts {
            *c /= total;
        }
    }

    counts
}

/// Compute depth histogram for a code graph
fn depth_histogram(graph: &CodeGraph, max_depth: usize) -> Vec<f32> {
    let mut histogram = vec![0.0; max_depth];
    let mut total = 0.0;

    // Use BFS from root to compute depths
    if let Some(root) = graph.root {
        let mut queue = vec![(root, 0usize)];
        let mut visited = std::collections::HashSet::new();
        visited.insert(root);

        while let Some((node, depth)) = queue.pop() {
            let bin = depth.min(max_depth - 1);
            histogram[bin] += 1.0;
            total += 1.0;

            // Add children
            for neighbor in graph.graph.neighbors(node) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push((neighbor, depth + 1));
                }
            }
        }
    }

    // Normalize
    if total > 0.0 {
        for h in &mut histogram {
            *h /= total;
        }
    }

    histogram
}

/// Compute L1 distance between two normalized histograms
fn histogram_distance(h1: &[f32], h2: &[f32]) -> f32 {
    h1.iter()
        .zip(h2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / 2.0 // Normalize to [0, 1]
}

/// Compute control flow structure distance
fn control_flow_distance(predicted: &CodeGraph, target: &CodeGraph) -> f32 {
    let pred_cf = extract_control_flow_pattern(predicted);
    let target_cf = extract_control_flow_pattern(target);

    // Compare control flow patterns
    let mut distance = 0.0;
    let mut total = 0.0;

    // Compare counts of each control flow type
    for cf_type in ControlFlowType::all() {
        let pred_count = pred_cf.get(cf_type).copied().unwrap_or(0) as f32;
        let target_count = target_cf.get(cf_type).copied().unwrap_or(0) as f32;
        let max_count = pred_count.max(target_count).max(1.0);
        distance += (pred_count - target_count).abs() / max_count;
        total += 1.0;
    }

    if total > 0.0 {
        distance / total
    } else {
        0.0
    }
}

/// Control flow types for pattern matching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ControlFlowType {
    If,
    LoopFor,
    LoopWhile,
    LoopLoop,
    Return,
}

impl ControlFlowType {
    fn all() -> &'static [ControlFlowType] {
        &[
            ControlFlowType::If,
            ControlFlowType::LoopFor,
            ControlFlowType::LoopWhile,
            ControlFlowType::LoopLoop,
            ControlFlowType::Return,
        ]
    }
}

/// Extract control flow pattern from a code graph
fn extract_control_flow_pattern(graph: &CodeGraph) -> HashMap<ControlFlowType, usize> {
    let mut pattern = HashMap::new();

    for node_idx in graph.graph.node_indices() {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let cf_type = match node {
                CodeNode::If => Some(ControlFlowType::If),
                CodeNode::Loop { kind } => Some(match kind {
                    LoopKind::For => ControlFlowType::LoopFor,
                    LoopKind::While | LoopKind::DoWhile => ControlFlowType::LoopWhile,
                    LoopKind::Loop => ControlFlowType::LoopLoop,
                }),
                CodeNode::Return => Some(ControlFlowType::Return),
                _ => None,
            };

            if let Some(cf) = cf_type {
                *pattern.entry(cf).or_insert(0) += 1;
            }
        }
    }

    pattern
}

/// Compute function signature distance
fn function_signature_distance(predicted: &CodeGraph, target: &CodeGraph) -> f32 {
    let pred_funcs = extract_function_signatures(predicted);
    let target_funcs = extract_function_signatures(target);

    // Compare number of functions
    let pred_count = pred_funcs.len() as f32;
    let target_count = target_funcs.len() as f32;
    let max_count = pred_count.max(target_count).max(1.0);
    let count_diff = (pred_count - target_count).abs() / max_count;

    // Compare parameter counts
    let pred_avg_params = if pred_funcs.is_empty() {
        0.0
    } else {
        pred_funcs.iter().map(|f| f.param_count as f32).sum::<f32>() / pred_funcs.len() as f32
    };

    let target_avg_params = if target_funcs.is_empty() {
        0.0
    } else {
        target_funcs.iter().map(|f| f.param_count as f32).sum::<f32>() / target_funcs.len() as f32
    };

    let max_params = pred_avg_params.max(target_avg_params).max(1.0);
    let param_diff = (pred_avg_params - target_avg_params).abs() / max_params;

    // Compare return type presence ratio
    let pred_has_return = pred_funcs.iter().filter(|f| f.has_return_type).count() as f32;
    let target_has_return = target_funcs.iter().filter(|f| f.has_return_type).count() as f32;

    let pred_return_ratio = if pred_funcs.is_empty() {
        0.0
    } else {
        pred_has_return / pred_funcs.len() as f32
    };

    let target_return_ratio = if target_funcs.is_empty() {
        0.0
    } else {
        target_has_return / target_funcs.len() as f32
    };

    let return_diff = (pred_return_ratio - target_return_ratio).abs();

    // Average the differences
    (count_diff + param_diff + return_diff) / 3.0
}

/// Function signature info
struct FunctionSignature {
    param_count: usize,
    has_return_type: bool,
}

/// Extract function signatures from a code graph
fn extract_function_signatures(graph: &CodeGraph) -> Vec<FunctionSignature> {
    let mut signatures = Vec::new();

    for node_idx in graph.graph.node_indices() {
        if let Some(CodeNode::Function { params, return_type, .. }) = graph.graph.node_weight(node_idx) {
            signatures.push(FunctionSignature {
                param_count: params.len(),
                has_return_type: return_type.is_some(),
            });
        }
    }

    signatures
}

// ============================================================================
// DagNN Structural Loss (Code-Aware)
// ============================================================================

/// Code-aware structural loss for DagNN (uses stored metadata if available)
pub fn dagnn_code_loss(predicted: &DagNN, target: &DagNN) -> CodeStructuralLoss {
    // For DagNN, we compute a simpler structural loss based on graph topology
    // since we don't have the CodeNode types directly

    let pred_nodes = predicted.node_count();
    let target_nodes = target.node_count();
    let pred_edges = predicted.edge_count();
    let target_edges = target.edge_count();

    // Node count loss
    let max_nodes = pred_nodes.max(target_nodes).max(1);
    let node_loss = (pred_nodes as f32 - target_nodes as f32).abs() / max_nodes as f32;

    // Edge count loss
    let max_edges = pred_edges.max(target_edges).max(1);
    let edge_loss = (pred_edges as f32 - target_edges as f32).abs() / max_edges as f32;

    // Degree distribution loss
    let pred_degrees = dagnn_degree_histogram(predicted, 10);
    let target_degrees = dagnn_degree_histogram(target, 10);
    let degree_loss = histogram_distance(&pred_degrees, &target_degrees);

    // Depth distribution loss (estimate from node count)
    let pred_depth = (pred_nodes as f32).sqrt();
    let target_depth = (target_nodes as f32).sqrt();
    let max_depth = pred_depth.max(target_depth).max(1.0);
    let depth_loss = (pred_depth - target_depth).abs() / max_depth;

    // Total (using default weights)
    let total = (node_loss + 0.8 * edge_loss + 1.2 * degree_loss + 0.5 * depth_loss) / 3.5;

    CodeStructuralLoss {
        total,
        node_type_loss: node_loss,
        edge_type_loss: edge_loss,
        control_flow_loss: degree_loss,
        function_loss: 0.0, // Not available for DagNN
        depth_loss,
    }
}

/// Compute degree histogram for DagNN
fn dagnn_degree_histogram(graph: &DagNN, num_bins: usize) -> Vec<f32> {
    let mut histogram = vec![0.0; num_bins];
    let node_count = graph.node_count();

    if node_count == 0 {
        return histogram;
    }

    for node in graph.graph.node_indices() {
        let degree = graph.graph.edges(node).count();
        let bin = degree.min(num_bins - 1);
        histogram[bin] += 1.0;
    }

    // Normalize
    for h in &mut histogram {
        *h /= node_count as f32;
    }

    histogram
}

// ============================================================================
// Operation-Aware Loss
// ============================================================================

/// Operation type categories for finer-grained comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    Arithmetic,
    Comparison,
    Logical,
    Bitwise,
    Assignment,
    Other,
}

impl OperationType {
    /// Categorize a binary operator
    pub fn from_binary_op(op: &BinaryOperator) -> Self {
        match op {
            BinaryOperator::Add | BinaryOperator::Sub | BinaryOperator::Mul
            | BinaryOperator::Div | BinaryOperator::Mod => OperationType::Arithmetic,
            BinaryOperator::Eq | BinaryOperator::Ne | BinaryOperator::Lt
            | BinaryOperator::Le | BinaryOperator::Gt | BinaryOperator::Ge => OperationType::Comparison,
            BinaryOperator::And | BinaryOperator::Or => OperationType::Logical,
            BinaryOperator::BitAnd | BinaryOperator::BitOr | BinaryOperator::BitXor
            | BinaryOperator::Shl | BinaryOperator::Shr => OperationType::Bitwise,
        }
    }

    /// Categorize a unary operator
    pub fn from_unary_op(op: &UnaryOperator) -> Self {
        match op {
            UnaryOperator::Neg => OperationType::Arithmetic,
            UnaryOperator::Not => OperationType::Logical,
            UnaryOperator::BitNot => OperationType::Bitwise,
            UnaryOperator::Deref | UnaryOperator::Ref => OperationType::Other,
        }
    }
}

/// Compute operation type histogram
pub fn operation_histogram(graph: &CodeGraph) -> HashMap<OperationType, f32> {
    let mut histogram: HashMap<OperationType, f32> = HashMap::new();
    let mut total = 0.0;

    for node_idx in graph.graph.node_indices() {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let op_type = match node {
                CodeNode::BinaryOp(op) => Some(OperationType::from_binary_op(op)),
                CodeNode::UnaryOp(op) => Some(OperationType::from_unary_op(op)),
                _ => None,
            };

            if let Some(op) = op_type {
                *histogram.entry(op).or_insert(0.0) += 1.0;
                total += 1.0;
            }
        }
    }

    // Normalize
    if total > 0.0 {
        for v in histogram.values_mut() {
            *v /= total;
        }
    }

    histogram
}

/// Compute operation type distance
pub fn operation_distance(predicted: &CodeGraph, target: &CodeGraph) -> f32 {
    let pred_ops = operation_histogram(predicted);
    let target_ops = operation_histogram(target);

    let all_types = [
        OperationType::Arithmetic,
        OperationType::Comparison,
        OperationType::Logical,
        OperationType::Bitwise,
        OperationType::Assignment,
        OperationType::Other,
    ];

    let mut distance = 0.0;
    for op_type in &all_types {
        let pred_val = pred_ops.get(op_type).copied().unwrap_or(0.0);
        let target_val = target_ops.get(op_type).copied().unwrap_or(0.0);
        distance += (pred_val - target_val).abs();
    }

    distance / 2.0 // Normalize to [0, 1]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use grapheme_code::Language;

    fn create_simple_code_graph() -> CodeGraph {
        let mut graph = CodeGraph::new();
        let module = graph.graph.add_node(CodeNode::Module {
            name: "test".to_string(),
            language: Language::Python,
        });
        graph.root = Some(module);

        let func = graph.graph.add_node(CodeNode::Function {
            name: "foo".to_string(),
            params: vec!["x".to_string(), "y".to_string()],
            return_type: Some("int".to_string()),
        });
        graph.graph.add_edge(module, func, CodeEdge::Child(0));

        let var = graph.graph.add_node(CodeNode::Variable {
            name: "z".to_string(),
            var_type: Some("int".to_string()),
        });
        graph.graph.add_edge(func, var, CodeEdge::Child(0));

        graph
    }

    fn create_complex_code_graph() -> CodeGraph {
        let mut graph = create_simple_code_graph();

        // Add control flow
        let if_node = graph.graph.add_node(CodeNode::If);
        let loop_node = graph.graph.add_node(CodeNode::Loop { kind: LoopKind::For });
        let ret = graph.graph.add_node(CodeNode::Return);

        // Add operations
        let add_op = graph.graph.add_node(CodeNode::BinaryOp(BinaryOperator::Add));
        let lt_op = graph.graph.add_node(CodeNode::BinaryOp(BinaryOperator::Lt));

        // Connect
        if let Some(root) = graph.root {
            let func = graph.graph.neighbors(root).next().unwrap();
            graph.graph.add_edge(func, if_node, CodeEdge::Child(1));
            graph.graph.add_edge(func, loop_node, CodeEdge::Child(2));
            graph.graph.add_edge(func, ret, CodeEdge::Child(3));
            graph.graph.add_edge(if_node, add_op, CodeEdge::Child(0));
            graph.graph.add_edge(loop_node, lt_op, CodeEdge::ControlFlow);
        }

        graph
    }

    #[test]
    fn test_node_category() {
        let module = CodeNode::Module {
            name: "test".to_string(),
            language: Language::Python,
        };
        assert_eq!(NodeCategory::from_node(&module), NodeCategory::Structure);

        let func = CodeNode::Function {
            name: "foo".to_string(),
            params: vec![],
            return_type: None,
        };
        assert_eq!(NodeCategory::from_node(&func), NodeCategory::Function);

        let if_node = CodeNode::If;
        assert_eq!(NodeCategory::from_node(&if_node), NodeCategory::ControlFlow);
    }

    #[test]
    fn test_edge_category() {
        assert_eq!(EdgeCategory::from_edge(&CodeEdge::Child(0)), EdgeCategory::Structural);
        assert_eq!(EdgeCategory::from_edge(&CodeEdge::ControlFlow), EdgeCategory::ControlFlow);
        assert_eq!(EdgeCategory::from_edge(&CodeEdge::DataFlow), EdgeCategory::DataFlow);
    }

    #[test]
    fn test_node_type_histogram() {
        let graph = create_simple_code_graph();
        let hist = node_type_histogram(&graph);

        assert_eq!(hist.len(), NodeCategory::all().len());
        // Sum should be ~1.0 (normalized)
        let sum: f32 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_edge_type_histogram() {
        let graph = create_simple_code_graph();
        let hist = edge_type_histogram(&graph);

        assert_eq!(hist.len(), EdgeCategory::all().len());
    }

    #[test]
    fn test_histogram_distance_identical() {
        let h1 = vec![0.5, 0.3, 0.2];
        let h2 = vec![0.5, 0.3, 0.2];
        let dist = histogram_distance(&h1, &h2);
        assert!(dist < 0.01);
    }

    #[test]
    fn test_histogram_distance_different() {
        let h1 = vec![1.0, 0.0, 0.0];
        let h2 = vec![0.0, 0.0, 1.0];
        let dist = histogram_distance(&h1, &h2);
        assert!(dist > 0.5);
    }

    #[test]
    fn test_code_structural_loss_identical() {
        let graph = create_simple_code_graph();
        let config = CodeLossConfig::default();
        let loss = code_structural_loss(&graph, &graph, &config);

        assert!(loss.total < 0.01, "Identical graphs should have near-zero loss");
    }

    #[test]
    fn test_code_structural_loss_different() {
        let simple = create_simple_code_graph();
        let complex = create_complex_code_graph();
        let config = CodeLossConfig::default();
        let loss = code_structural_loss(&simple, &complex, &config);

        assert!(loss.total > 0.1, "Different graphs should have significant loss");
    }

    #[test]
    fn test_control_flow_distance() {
        let simple = create_simple_code_graph();
        let complex = create_complex_code_graph();

        let dist = control_flow_distance(&simple, &complex);
        assert!(dist > 0.0, "Graphs with different control flow should have positive distance");
    }

    #[test]
    fn test_function_signature_distance_identical() {
        let graph = create_simple_code_graph();
        let dist = function_signature_distance(&graph, &graph);
        assert!(dist < 0.01);
    }

    #[test]
    fn test_depth_histogram() {
        let graph = create_simple_code_graph();
        let hist = depth_histogram(&graph, 5);

        assert_eq!(hist.len(), 5);
        // Should have nodes at depth 0, 1, 2
        assert!(hist[0] > 0.0 || hist[1] > 0.0);
    }

    #[test]
    fn test_operation_histogram() {
        let graph = create_complex_code_graph();
        let hist = operation_histogram(&graph);

        // Should have arithmetic and comparison operations
        assert!(hist.len() > 0);
    }

    #[test]
    fn test_operation_distance() {
        let simple = create_simple_code_graph();
        let complex = create_complex_code_graph();

        let dist = operation_distance(&simple, &complex);
        // Complex has operations, simple doesn't
        assert!(dist > 0.0);
    }

    #[test]
    fn test_dagnn_code_loss() {
        use grapheme_core::{Node, Edge};

        let mut dag1 = DagNN::new();
        for _ in 0..5 {
            dag1.graph.add_node(Node::hidden());
        }
        dag1.graph.add_edge(
            petgraph::graph::NodeIndex::new(0),
            petgraph::graph::NodeIndex::new(1),
            Edge::sequential(),
        );

        let dag2 = dag1.clone();
        let loss = dagnn_code_loss(&dag1, &dag2);

        assert!(loss.total < 0.1, "Identical DagNNs should have low loss");
    }

    #[test]
    fn test_code_loss_config_default() {
        let config = CodeLossConfig::default();
        assert!(config.node_type_weight > 0.0);
        assert!(config.function_weight > 0.0);
    }
}
