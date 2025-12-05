//! # grapheme-core
//!
//! Layer 4: The universal interface - character-level natural language processing.
//!
//! This crate provides:
//! - Pure GRAPHEME processing (character-level, vocabulary-free)
//! - Natural language understanding
//! - Support for any human language input
//! - Intent and parameter extraction
//!
//! Key innovations:
//! - No tokenization or vocabulary
//! - Direct character-to-node mapping
//! - Dynamic graph growth with input
//! - Universal Unicode support

use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Type Aliases (matching GRAPHEME_Vision.md)
// ============================================================================

/// Node identifier type
pub type NodeId = NodeIndex;

// ============================================================================
// Error Types
// ============================================================================

/// Errors in core GRAPHEME processing
#[derive(Error, Debug)]
pub enum GraphemeError {
    #[error("Invalid character: {0}")]
    InvalidCharacter(char),
    #[error("Graph construction error: {0}")]
    GraphError(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
    #[error("Cycle detected in graph")]
    CycleDetected,
}

/// Result type for GRAPHEME operations
pub type GraphemeResult<T> = Result<T, GraphemeError>;

// ============================================================================
// Core Data Structures (aligned with GRAPHEME_Vision.md)
// ============================================================================

/// Compression type for compressed nodes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    /// Run-length encoding for repeated characters
    RunLength,
    /// Pattern-based compression
    PatternBased,
    /// Hierarchical compression (nested structures)
    Hierarchical,
    /// Semantic compression (meaning-preserving)
    Semantic,
}

/// Type of node in the GRAPHEME graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    /// Input character node
    Input(char),
    /// Hidden processing node
    Hidden,
    /// Output node
    Output,
    /// Clique (compressed concept) - contains member node IDs
    Clique(Vec<usize>),
    /// Pattern node (compressed repeated sequence)
    Pattern(Vec<u8>),
    /// Compressed region with compression type
    Compressed(CompressionType),
}

/// A node in the GRAPHEME graph (matching GRAPHEME_Vision.md)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// The raw character value (or compressed pattern)
    pub value: Option<u8>,
    /// Current activation level
    pub activation: f32,
    /// Type of this node
    pub node_type: NodeType,
    /// Position in original text (if input node)
    pub position: Option<usize>,
}

impl Node {
    /// Create a new input node from a character
    pub fn input(ch: char, position: usize) -> Self {
        Self {
            value: if ch.is_ascii() {
                Some(ch as u8)
            } else {
                None
            },
            activation: 1.0,
            node_type: NodeType::Input(ch),
            position: Some(position),
        }
    }

    /// Create a new hidden node
    pub fn hidden() -> Self {
        Self {
            value: None,
            activation: 0.0,
            node_type: NodeType::Hidden,
            position: None,
        }
    }

    /// Create a new output node
    pub fn output() -> Self {
        Self {
            value: None,
            activation: 0.0,
            node_type: NodeType::Output,
            position: None,
        }
    }

    /// Create a clique node
    pub fn clique(members: Vec<usize>) -> Self {
        Self {
            value: None,
            activation: 0.0,
            node_type: NodeType::Clique(members),
            position: None,
        }
    }

    /// Create a pattern node
    pub fn pattern(pattern: Vec<u8>) -> Self {
        Self {
            value: None,
            activation: 0.0,
            node_type: NodeType::Pattern(pattern),
            position: None,
        }
    }

    /// Create a compressed node
    pub fn compressed(compression: CompressionType) -> Self {
        Self {
            value: None,
            activation: 0.0,
            node_type: NodeType::Compressed(compression),
            position: None,
        }
    }
}

/// Type of edge in the GRAPHEME graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Sequential character connection
    Sequential,
    /// Semantic relationship
    Semantic,
    /// Structural/syntactic connection
    Structural,
    /// Within-clique connection
    Clique,
    /// Long-range dependency (skip connection)
    Skip,
}

/// An edge in the GRAPHEME graph (matching GRAPHEME_Vision.md)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Edge weight (learnable)
    pub weight: f32,
    /// Type of edge
    pub edge_type: EdgeType,
}

impl Edge {
    /// Create a new edge
    pub fn new(weight: f32, edge_type: EdgeType) -> Self {
        Self { weight, edge_type }
    }

    /// Create a sequential edge (default weight 1.0)
    pub fn sequential() -> Self {
        Self::new(1.0, EdgeType::Sequential)
    }

    /// Create a semantic edge
    pub fn semantic(weight: f32) -> Self {
        Self::new(weight, EdgeType::Semantic)
    }

    /// Create a skip connection edge
    pub fn skip(weight: f32) -> Self {
        Self::new(weight, EdgeType::Skip)
    }

    /// Create a clique edge
    pub fn clique(weight: f32) -> Self {
        Self::new(weight, EdgeType::Clique)
    }
}

// ============================================================================
// Clique Structure
// ============================================================================

/// A clique (densely connected subgraph) representing a learned concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clique {
    /// Unique identifier for this clique
    pub id: usize,
    /// Member node indices
    pub members: Vec<NodeId>,
    /// Activation strength of the clique
    pub strength: f32,
    /// Optional label/name for the concept
    pub label: Option<String>,
}

impl Clique {
    /// Create a new clique
    pub fn new(id: usize, members: Vec<NodeId>) -> Self {
        Self {
            id,
            members,
            strength: 1.0,
            label: None,
        }
    }

    /// Create a labeled clique
    pub fn with_label(id: usize, members: Vec<NodeId>, label: impl Into<String>) -> Self {
        Self {
            id,
            members,
            strength: 1.0,
            label: Some(label.into()),
        }
    }

    /// Get the size of the clique
    pub fn size(&self) -> usize {
        self.members.len()
    }
}

// ============================================================================
// Topological Order
// ============================================================================

/// Topological ordering of nodes for efficient forward propagation
#[derive(Debug, Clone, Default)]
pub struct TopologicalOrder {
    /// Nodes in topological order (from inputs to outputs)
    pub order: Vec<NodeId>,
    /// Mapping from node to its position in the order
    pub position: HashMap<NodeId, usize>,
}

impl TopologicalOrder {
    /// Create a new empty topological order
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute topological order from a graph
    pub fn from_graph(graph: &DiGraph<Node, Edge>) -> GraphemeResult<Self> {
        match toposort(graph, None) {
            Ok(order) => {
                let position: HashMap<NodeId, usize> =
                    order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
                Ok(Self { order, position })
            }
            Err(_) => Err(GraphemeError::CycleDetected),
        }
    }

    /// Get the position of a node in the topological order
    pub fn get_position(&self, node: NodeId) -> Option<usize> {
        self.position.get(&node).copied()
    }

    /// Check if node A comes before node B in topological order
    pub fn comes_before(&self, a: NodeId, b: NodeId) -> Option<bool> {
        match (self.position.get(&a), self.position.get(&b)) {
            (Some(&pos_a), Some(&pos_b)) => Some(pos_a < pos_b),
            _ => None,
        }
    }
}

// ============================================================================
// Graph Memory (for storing learned transformations)
// ============================================================================

/// A stored graph transformation pattern
#[derive(Debug, Clone)]
pub struct TransformationPattern {
    /// Input graph pattern
    pub input_pattern: Vec<NodeType>,
    /// Output graph pattern
    pub output_pattern: Vec<NodeType>,
    /// How often this pattern has been seen
    pub frequency: usize,
    /// Confidence score
    pub confidence: f32,
}

/// Memory for storing graph transformation patterns
#[derive(Debug, Clone, Default)]
pub struct GraphMemory {
    /// Stored transformation patterns
    pub patterns: Vec<TransformationPattern>,
    /// Maximum number of patterns to store
    pub capacity: usize,
}

impl GraphMemory {
    /// Create a new graph memory with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            patterns: Vec::new(),
            capacity,
        }
    }

    /// Store a new transformation pattern
    pub fn store(&mut self, pattern: TransformationPattern) {
        if self.patterns.len() < self.capacity {
            self.patterns.push(pattern);
        } else {
            // Replace lowest frequency pattern
            if let Some(idx) = self
                .patterns
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| p.frequency)
                .map(|(i, _)| i)
            {
                if self.patterns[idx].frequency < pattern.frequency {
                    self.patterns[idx] = pattern;
                }
            }
        }
    }

    /// Get the number of stored patterns
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

// ============================================================================
// DagNN - Main Graph Structure (matching GRAPHEME_Vision.md)
// ============================================================================

/// The main GRAPHEME DAG Neural Network structure
///
/// This is the core data structure matching the GRAPHEME_Vision.md specification:
/// ```text
/// pub struct DagNN {
///     pub nodes: Vec<Node>,
///     pub edges: Vec<Edge>,
///     pub topology: TopologicalOrder,
///     pub cliques: Vec<Clique>,
///     pub memory: GraphMemory,
/// }
/// ```
#[derive(Debug)]
pub struct DagNN {
    /// The underlying directed acyclic graph
    pub graph: DiGraph<Node, Edge>,
    /// Topological ordering for efficient traversal
    pub topology: TopologicalOrder,
    /// Detected cliques (learned concepts)
    pub cliques: Vec<Clique>,
    /// Memory for learned transformations
    pub memory: GraphMemory,
    /// Input nodes in order
    input_nodes: Vec<NodeId>,
    /// Output nodes
    output_nodes: Vec<NodeId>,
}

impl Default for DagNN {
    fn default() -> Self {
        Self::new()
    }
}

impl DagNN {
    /// Create a new empty DagNN
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            topology: TopologicalOrder::new(),
            cliques: Vec::new(),
            memory: GraphMemory::new(1000),
            input_nodes: Vec::new(),
            output_nodes: Vec::new(),
        }
    }

    /// Build a DagNN from text (character by character, NO tokenization)
    pub fn from_text(text: &str) -> GraphemeResult<Self> {
        let mut dag = Self::new();

        let mut prev_node: Option<NodeId> = None;

        for (position, ch) in text.chars().enumerate() {
            let node = dag.graph.add_node(Node::input(ch, position));
            dag.input_nodes.push(node);

            // Connect to previous character
            if let Some(prev) = prev_node {
                dag.graph.add_edge(prev, node, Edge::sequential());
            }

            prev_node = Some(node);
        }

        // Compute topological order
        dag.topology = TopologicalOrder::from_graph(&dag.graph)?;

        Ok(dag)
    }

    /// Add a character to the graph
    pub fn add_character(&mut self, ch: char, position: usize) -> NodeId {
        let node = self.graph.add_node(Node::input(ch, position));
        self.input_nodes.push(node);

        // Connect to previous if exists
        if self.input_nodes.len() > 1 {
            let prev = self.input_nodes[self.input_nodes.len() - 2];
            self.graph.add_edge(prev, node, Edge::sequential());
        }

        node
    }

    /// Add a hidden node
    pub fn add_hidden(&mut self) -> NodeId {
        self.graph.add_node(Node::hidden())
    }

    /// Add an output node
    pub fn add_output(&mut self) -> NodeId {
        let node = self.graph.add_node(Node::output());
        self.output_nodes.push(node);
        node
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, source: NodeId, target: NodeId, edge: Edge) {
        self.graph.add_edge(source, target, edge);
    }

    /// Update topological order after graph modifications
    pub fn update_topology(&mut self) -> GraphemeResult<()> {
        self.topology = TopologicalOrder::from_graph(&self.graph)?;
        Ok(())
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get input nodes
    pub fn input_nodes(&self) -> &[NodeId] {
        &self.input_nodes
    }

    /// Get output nodes
    pub fn output_nodes(&self) -> &[NodeId] {
        &self.output_nodes
    }

    /// Convert graph back to text
    pub fn to_text(&self) -> String {
        self.input_nodes
            .iter()
            .filter_map(|&idx| {
                if let NodeType::Input(ch) = self.graph[idx].node_type {
                    Some(ch)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Form a clique from a set of nodes
    pub fn form_clique(&mut self, members: Vec<NodeId>, label: Option<String>) -> usize {
        let id = self.cliques.len();
        let clique = if let Some(l) = label {
            Clique::with_label(id, members, l)
        } else {
            Clique::new(id, members)
        };
        self.cliques.push(clique);
        id
    }

    /// Compute processing depth based on character complexity
    pub fn compute_processing_depth(ch: char, _context: &[char]) -> usize {
        match ch {
            // Common ASCII letters - shallow
            'a'..='z' | 'A'..='Z' => 2,
            // Digits
            '0'..='9' => 2,
            // Common punctuation
            ' ' | '.' | ',' | '!' | '?' => 1,
            // Mathematical symbols
            '+' | '-' | '*' | '/' | '=' | '<' | '>' => 3,
            // Complex Unicode (non-ASCII)
            _ if !ch.is_ascii() => {
                if ch.len_utf8() > 2 {
                    5
                } else {
                    4
                }
            }
            // Default
            _ => 2,
        }
    }
}

// ============================================================================
// Legacy GraphemeGraph (for backwards compatibility)
// ============================================================================

/// The main GRAPHEME graph structure (legacy - use DagNN for new code)
#[derive(Debug)]
pub struct GraphemeGraph {
    /// The underlying directed acyclic graph
    pub graph: DiGraph<Node, Edge>,
    /// Input nodes in order
    pub input_nodes: Vec<NodeIndex>,
    /// Detected cliques
    pub cliques: Vec<Vec<NodeIndex>>,
}

impl Default for GraphemeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphemeGraph {
    /// Create a new empty GRAPHEME graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            input_nodes: Vec::new(),
            cliques: Vec::new(),
        }
    }

    /// Build a graph from text (character by character, NO tokenization)
    pub fn from_text(text: &str) -> Self {
        let mut graph = Self::new();

        let mut prev_node: Option<NodeIndex> = None;

        for (position, ch) in text.chars().enumerate() {
            let node = graph.graph.add_node(Node::input(ch, position));
            graph.input_nodes.push(node);

            // Connect to previous character
            if let Some(prev) = prev_node {
                graph.graph.add_edge(prev, node, Edge::sequential());
            }

            prev_node = Some(node);
        }

        graph
    }

    /// Add a character to the graph
    pub fn add_character(&mut self, ch: char, position: usize) -> NodeIndex {
        let node = self.graph.add_node(Node::input(ch, position));
        self.input_nodes.push(node);

        // Connect to previous if exists
        if self.input_nodes.len() > 1 {
            let prev = self.input_nodes[self.input_nodes.len() - 2];
            self.graph.add_edge(prev, node, Edge::sequential());
        }

        node
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Convert graph back to text
    pub fn to_text(&self) -> String {
        self.input_nodes
            .iter()
            .filter_map(|&idx| {
                if let NodeType::Input(ch) = self.graph[idx].node_type {
                    Some(ch)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute processing depth based on character complexity
    pub fn compute_processing_depth(ch: char, _context: &[char]) -> usize {
        DagNN::compute_processing_depth(ch, _context)
    }
}

// ============================================================================
// Text Processor Trait
// ============================================================================

/// Text processor trait for GRAPHEME
pub trait TextProcessor {
    /// Convert text to graph (CHARACTER BY CHARACTER, NO TOKENIZATION)
    fn text_to_graph(&mut self, text: &str) -> GraphemeGraph;

    /// Convert graph back to text
    fn graph_to_text(&self, graph: &GraphemeGraph) -> String;

    /// Process any Unicode text without configuration
    fn process_universal(&mut self, text: &str) -> GraphemeResult<GraphemeGraph>;
}

/// Basic implementation of TextProcessor
#[derive(Debug, Default)]
pub struct BasicTextProcessor;

impl BasicTextProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl TextProcessor for BasicTextProcessor {
    fn text_to_graph(&mut self, text: &str) -> GraphemeGraph {
        GraphemeGraph::from_text(text)
    }

    fn graph_to_text(&self, graph: &GraphemeGraph) -> String {
        graph.to_text()
    }

    fn process_universal(&mut self, text: &str) -> GraphemeResult<GraphemeGraph> {
        // Works with any Unicode: "Hello", "你好", "مرحبا", "🚀", "∫dx"
        Ok(self.text_to_graph(text))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_graph() {
        let graph = GraphemeGraph::from_text("Hello");
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4); // 4 sequential edges
    }

    #[test]
    fn test_roundtrip() {
        let original = "Hello, World!";
        let graph = GraphemeGraph::from_text(original);
        let roundtrip = graph.to_text();
        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_unicode() {
        // Test universal language support
        let texts = vec![
            "Hello",
            "你好",
            "مرحبا",
            "🚀🎉",
            "∫x²dx",
            "Hello你好🚀",
        ];

        let mut processor = BasicTextProcessor::new();

        for text in texts {
            let graph = processor.process_universal(text).unwrap();
            let roundtrip = graph.to_text();
            assert_eq!(text, roundtrip, "Failed for: {}", text);
        }
    }

    #[test]
    fn test_processing_depth() {
        assert_eq!(GraphemeGraph::compute_processing_depth('a', &[]), 2);
        assert_eq!(GraphemeGraph::compute_processing_depth('+', &[]), 3);
        // '你' has 3-byte UTF-8 encoding, so depth = 5
        assert_eq!(GraphemeGraph::compute_processing_depth('你', &[]), 5);
    }

    // New tests for DagNN and related structures

    #[test]
    fn test_dagnn_from_text() {
        let dag = DagNN::from_text("Hello").unwrap();
        assert_eq!(dag.node_count(), 5);
        assert_eq!(dag.edge_count(), 4);
        assert_eq!(dag.topology.order.len(), 5);
    }

    #[test]
    fn test_dagnn_roundtrip() {
        let original = "Hello, World!";
        let dag = DagNN::from_text(original).unwrap();
        let roundtrip = dag.to_text();
        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_topological_order() {
        let dag = DagNN::from_text("ABC").unwrap();

        // First character should come before last
        let first = dag.input_nodes()[0];
        let last = dag.input_nodes()[2];
        assert!(dag.topology.comes_before(first, last).unwrap());
    }

    #[test]
    fn test_clique_creation() {
        let mut dag = DagNN::from_text("test").unwrap();
        let members = dag.input_nodes().to_vec();
        let clique_id = dag.form_clique(members.clone(), Some("test_word".into()));

        assert_eq!(dag.cliques.len(), 1);
        assert_eq!(dag.cliques[clique_id].size(), 4);
        assert_eq!(dag.cliques[clique_id].label, Some("test_word".to_string()));
    }

    #[test]
    fn test_node_types() {
        let input = Node::input('a', 0);
        assert!(matches!(input.node_type, NodeType::Input('a')));

        let hidden = Node::hidden();
        assert!(matches!(hidden.node_type, NodeType::Hidden));

        let output = Node::output();
        assert!(matches!(output.node_type, NodeType::Output));

        let clique = Node::clique(vec![1, 2, 3]);
        assert!(matches!(clique.node_type, NodeType::Clique(_)));

        let pattern = Node::pattern(vec![b'a', b'b']);
        assert!(matches!(pattern.node_type, NodeType::Pattern(_)));

        let compressed = Node::compressed(CompressionType::RunLength);
        assert!(matches!(
            compressed.node_type,
            NodeType::Compressed(CompressionType::RunLength)
        ));
    }

    #[test]
    fn test_edge_types() {
        let seq = Edge::sequential();
        assert_eq!(seq.edge_type, EdgeType::Sequential);
        assert_eq!(seq.weight, 1.0);

        let semantic = Edge::semantic(0.8);
        assert_eq!(semantic.edge_type, EdgeType::Semantic);
        assert_eq!(semantic.weight, 0.8);

        let skip = Edge::skip(0.5);
        assert_eq!(skip.edge_type, EdgeType::Skip);
    }

    #[test]
    fn test_graph_memory() {
        let mut memory = GraphMemory::new(2);
        assert!(memory.is_empty());

        memory.store(TransformationPattern {
            input_pattern: vec![NodeType::Input('a')],
            output_pattern: vec![NodeType::Output],
            frequency: 5,
            confidence: 0.9,
        });
        assert_eq!(memory.len(), 1);

        memory.store(TransformationPattern {
            input_pattern: vec![NodeType::Input('b')],
            output_pattern: vec![NodeType::Output],
            frequency: 3,
            confidence: 0.7,
        });
        assert_eq!(memory.len(), 2);

        // Adding a third should replace lowest frequency
        memory.store(TransformationPattern {
            input_pattern: vec![NodeType::Input('c')],
            output_pattern: vec![NodeType::Output],
            frequency: 10,
            confidence: 0.95,
        });
        assert_eq!(memory.len(), 2);
    }
}
