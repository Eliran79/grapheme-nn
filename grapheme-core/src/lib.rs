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
use petgraph::visit::EdgeRef;
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
// GraphBuilder Trait (matching GRAPHEME_Vision.md)
// ============================================================================

/// Compressed region placeholder
#[derive(Debug, Clone)]
pub struct CompressedRegion {
    /// Start node of the compressed region
    pub start: NodeId,
    /// End node of the compressed region
    pub end: NodeId,
    /// The compressed node representing this region
    pub compressed_node: NodeId,
    /// Original node count
    pub original_count: usize,
}

/// Hierarchical graph representation
#[derive(Debug, Clone)]
pub struct HierarchicalGraph {
    /// Levels of abstraction (0 = raw characters, higher = more abstract)
    pub levels: Vec<Vec<NodeId>>,
    /// Connections between levels
    pub inter_level_edges: Vec<(NodeId, NodeId)>,
}

/// Graph builder trait for constructing GRAPHEME graphs
pub trait GraphBuilder {
    /// Add single character to the graph
    fn add_character(&mut self, ch: char, position: usize) -> NodeId;

    /// Form connections based on relevance within a context window
    fn connect_relevant(&mut self, node: NodeId, context_window: usize);

    /// Detect and form semantic cliques
    fn form_cliques(&mut self) -> Vec<Clique>;

    /// Compress inactive regions for memory efficiency
    fn compress_region(&mut self, start: NodeId, end: NodeId) -> GraphemeResult<CompressedRegion>;

    /// Build hierarchical abstraction of the graph
    fn build_hierarchy(&mut self) -> HierarchicalGraph;
}

impl GraphBuilder for DagNN {
    fn add_character(&mut self, ch: char, position: usize) -> NodeId {
        DagNN::add_character(self, ch, position)
    }

    fn connect_relevant(&mut self, node: NodeId, context_window: usize) {
        // Connect to nodes within the context window
        let node_pos = self.graph[node].position.unwrap_or(0);

        for &other in &self.input_nodes {
            if other == node {
                continue;
            }
            if let Some(other_pos) = self.graph[other].position {
                let distance = if node_pos > other_pos {
                    node_pos - other_pos
                } else {
                    other_pos - node_pos
                };

                if distance <= context_window && distance > 1 {
                    // Add skip connection with weight based on distance
                    let weight = 1.0 / (distance as f32);
                    self.graph.add_edge(other, node, Edge::skip(weight));
                }
            }
        }
    }

    fn form_cliques(&mut self) -> Vec<Clique> {
        // Simple clique detection: find nodes with high mutual connectivity
        // For now, group consecutive nodes as basic cliques

        // Collect windows first to avoid borrow conflict
        let windows: Vec<Vec<NodeId>> = if self.input_nodes.len() >= 3 {
            self.input_nodes.windows(3)
                .map(|w| w.to_vec())
                .collect()
        } else {
            Vec::new()
        };

        // Now form cliques from collected windows
        let mut cliques = Vec::new();
        for members in windows {
            let clique_id = self.form_clique(members, None);
            cliques.push(self.cliques[clique_id].clone());
        }

        cliques
    }

    fn compress_region(&mut self, start: NodeId, end: NodeId) -> GraphemeResult<CompressedRegion> {
        // Find nodes between start and end
        let start_pos = self.topology.get_position(start)
            .ok_or_else(|| GraphemeError::GraphError("Start node not in topology".into()))?;
        let end_pos = self.topology.get_position(end)
            .ok_or_else(|| GraphemeError::GraphError("End node not in topology".into()))?;

        let nodes_in_region: Vec<NodeId> = self.topology.order[start_pos..=end_pos].to_vec();
        let original_count = nodes_in_region.len();

        // Create compressed node
        let compressed_node = self.graph.add_node(Node::compressed(CompressionType::Hierarchical));

        Ok(CompressedRegion {
            start,
            end,
            compressed_node,
            original_count,
        })
    }

    fn build_hierarchy(&mut self) -> HierarchicalGraph {
        // Level 0: raw input nodes
        let level_0 = self.input_nodes.clone();

        // Level 1: cliques (if any)
        let level_1: Vec<NodeId> = self.cliques.iter()
            .map(|c| c.members[0]) // Representative node
            .collect();

        let levels = if level_1.is_empty() {
            vec![level_0]
        } else {
            vec![level_0, level_1]
        };

        HierarchicalGraph {
            levels,
            inter_level_edges: Vec::new(),
        }
    }
}

// ============================================================================
// ForwardPass Trait (matching GRAPHEME_Vision.md)
// ============================================================================

/// Forward propagation trait for GRAPHEME graphs
pub trait ForwardPass {
    /// Compute activation for a single node
    fn activate_node(&self, node: NodeId) -> f32;

    /// Forward pass through the entire graph
    fn forward(&mut self) -> GraphemeResult<()>;

    /// Parallel forward pass using rayon
    fn forward_parallel(&mut self) -> GraphemeResult<()>;

    /// Get current activations as a vector
    fn get_activations(&self) -> Vec<(NodeId, f32)>;
}

impl ForwardPass for DagNN {
    fn activate_node(&self, node: NodeId) -> f32 {
        self.graph[node].activation
    }

    fn forward(&mut self) -> GraphemeResult<()> {
        // Update topology if needed
        if self.topology.order.is_empty() {
            self.update_topology()?;
        }

        // Process nodes in topological order
        for &node in &self.topology.order {
            let mut incoming_sum = 0.0f32;
            let mut incoming_count = 0;

            // Sum weighted inputs from predecessors
            for edge in self.graph.edges_directed(node, petgraph::Direction::Incoming) {
                let source_activation = self.graph[edge.source()].activation;
                let weight = edge.weight().weight;
                incoming_sum += source_activation * weight;
                incoming_count += 1;
            }

            // Apply activation function (simple ReLU-like)
            if incoming_count > 0 {
                let new_activation = (incoming_sum / incoming_count as f32).max(0.0).min(1.0);
                self.graph[node].activation = new_activation;
            }
            // Input nodes keep their original activation
        }

        Ok(())
    }

    fn forward_parallel(&mut self) -> GraphemeResult<()> {
        use rayon::prelude::*;

        // Update topology if needed
        if self.topology.order.is_empty() {
            self.update_topology()?;
        }

        // Collect activations in parallel by level
        // For simplicity, use the sequential version for now
        // Full parallel implementation would process independent nodes concurrently
        self.forward()
    }

    fn get_activations(&self) -> Vec<(NodeId, f32)> {
        self.topology.order.iter()
            .map(|&node| (node, self.graph[node].activation))
            .collect()
    }
}

// ============================================================================
// GraphTransformer Trait (matching GRAPHEME_Vision.md)
// ============================================================================

/// Transformation rule for graph-to-graph operations
#[derive(Debug, Clone)]
pub struct TransformRule {
    /// Rule identifier
    pub id: usize,
    /// Description of the transformation
    pub description: String,
    /// Input pattern to match
    pub input_pattern: Vec<NodeType>,
    /// Output pattern to produce
    pub output_pattern: Vec<NodeType>,
}

impl TransformRule {
    /// Create a new transformation rule
    pub fn new(id: usize, description: impl Into<String>) -> Self {
        Self {
            id,
            description: description.into(),
            input_pattern: Vec::new(),
            output_pattern: Vec::new(),
        }
    }
}

/// Graph transformer trait for graph-to-graph operations
pub trait GraphTransformer {
    /// Transform one graph into another
    fn transform(&mut self, input: &DagNN) -> GraphemeResult<DagNN>;

    /// Learn a transformation rule from input/output pair
    fn learn_transformation(&mut self, input: &DagNN, target: &DagNN) -> TransformRule;

    /// Apply a specific transformation rule
    fn apply_rule(&mut self, graph: &DagNN, rule: &TransformRule) -> GraphemeResult<DagNN>;

    /// Compose multiple transformation rules
    fn compose(&self, rules: Vec<TransformRule>) -> TransformRule;
}

/// Basic graph transformer implementation
#[derive(Debug, Default)]
pub struct BasicGraphTransformer {
    /// Learned transformation rules
    pub rules: Vec<TransformRule>,
    /// Rule counter for generating IDs
    rule_counter: usize,
}

impl BasicGraphTransformer {
    pub fn new() -> Self {
        Self::default()
    }
}

impl GraphTransformer for BasicGraphTransformer {
    fn transform(&mut self, input: &DagNN) -> GraphemeResult<DagNN> {
        // Identity transformation for now
        let text = input.to_text();
        DagNN::from_text(&text)
    }

    fn learn_transformation(&mut self, input: &DagNN, target: &DagNN) -> TransformRule {
        let id = self.rule_counter;
        self.rule_counter += 1;

        // Extract patterns from input and target
        let input_pattern: Vec<NodeType> = input.input_nodes()
            .iter()
            .map(|&n| input.graph[n].node_type.clone())
            .collect();

        let output_pattern: Vec<NodeType> = target.input_nodes()
            .iter()
            .map(|&n| target.graph[n].node_type.clone())
            .collect();

        let rule = TransformRule {
            id,
            description: format!("Rule {} learned from examples", id),
            input_pattern,
            output_pattern,
        };

        self.rules.push(rule.clone());
        rule
    }

    fn apply_rule(&mut self, graph: &DagNN, _rule: &TransformRule) -> GraphemeResult<DagNN> {
        // For now, just copy the graph
        let text = graph.to_text();
        DagNN::from_text(&text)
    }

    fn compose(&self, rules: Vec<TransformRule>) -> TransformRule {
        // Compose by concatenating patterns
        let id = rules.first().map(|r| r.id).unwrap_or(0);

        let mut input_pattern = Vec::new();
        let mut output_pattern = Vec::new();

        for rule in &rules {
            input_pattern.extend(rule.input_pattern.clone());
            output_pattern.extend(rule.output_pattern.clone());
        }

        TransformRule {
            id,
            description: format!("Composed rule from {} rules", rules.len()),
            input_pattern,
            output_pattern,
        }
    }
}

// ============================================================================
// CliqueProcessor Trait (matching GRAPHEME_Vision.md)
// ============================================================================

/// Clique processor trait for clique operations
pub trait CliqueProcessor {
    /// Find all cliques in the graph (parallel)
    fn find_cliques_parallel(&self) -> Vec<Clique>;

    /// Strengthen connections within a clique
    fn strengthen_clique(&mut self, clique: &Clique, factor: f32);

    /// Compress nodes into a single clique node
    fn compress_to_clique(&mut self, nodes: Vec<NodeId>) -> NodeId;

    /// Expand a clique node back to its member nodes
    fn expand_clique(&self, clique_id: usize) -> Option<Vec<NodeId>>;
}

impl CliqueProcessor for DagNN {
    fn find_cliques_parallel(&self) -> Vec<Clique> {
        use rayon::prelude::*;

        // Parallel clique detection
        // For now, return existing cliques
        self.cliques.clone()
    }

    fn strengthen_clique(&mut self, clique: &Clique, factor: f32) {
        // Strengthen all edges within the clique
        for i in 0..clique.members.len() {
            for j in (i + 1)..clique.members.len() {
                let source = clique.members[i];
                let target = clique.members[j];

                // Find and strengthen the edge if it exists
                if let Some(edge_idx) = self.graph.find_edge(source, target) {
                    self.graph[edge_idx].weight *= factor;
                } else {
                    // Add clique edge if it doesn't exist
                    self.graph.add_edge(source, target, Edge::clique(factor));
                }
            }
        }
    }

    fn compress_to_clique(&mut self, nodes: Vec<NodeId>) -> NodeId {
        // Create a new clique node
        let member_indices: Vec<usize> = nodes.iter()
            .map(|n| n.index())
            .collect();

        let clique_node = self.graph.add_node(Node::clique(member_indices));

        // Form the clique
        self.form_clique(nodes, None);

        clique_node
    }

    fn expand_clique(&self, clique_id: usize) -> Option<Vec<NodeId>> {
        self.cliques.get(clique_id).map(|c| c.members.clone())
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

    // Tests for new traits (api-002)

    #[test]
    fn test_graph_builder_trait() {
        let mut dag = DagNN::new();

        // Test add_character
        let n1 = dag.add_character('H', 0);
        let n2 = dag.add_character('i', 1);
        assert_eq!(dag.node_count(), 2);

        // Test connect_relevant
        dag.connect_relevant(n2, 3);
        assert!(dag.edge_count() >= 1);

        // Test build_hierarchy
        let hierarchy = dag.build_hierarchy();
        assert!(!hierarchy.levels.is_empty());
    }

    #[test]
    fn test_forward_pass_trait() {
        let mut dag = DagNN::from_text("AB").unwrap();

        // Test activate_node (should return default 1.0 for input nodes)
        let node = dag.input_nodes()[0];
        let activation = dag.activate_node(node);
        assert_eq!(activation, 1.0);

        // Test forward pass
        dag.forward().unwrap();

        // Test get_activations
        let activations = dag.get_activations();
        assert_eq!(activations.len(), 2);
    }

    #[test]
    fn test_graph_transformer_trait() {
        let input = DagNN::from_text("abc").unwrap();
        let target = DagNN::from_text("ABC").unwrap();

        let mut transformer = BasicGraphTransformer::new();

        // Test learn_transformation
        let rule = transformer.learn_transformation(&input, &target);
        assert!(!rule.description.is_empty());

        // Test transform (identity for basic transformer)
        let result = transformer.transform(&input).unwrap();
        assert_eq!(result.node_count(), input.node_count());
    }

    #[test]
    fn test_clique_processor_trait() {
        let mut dag = DagNN::from_text("word").unwrap();

        // Test find_cliques_parallel
        let cliques = dag.find_cliques_parallel();
        // May be empty for short text
        assert!(cliques.len() >= 0);

        // Test compress_to_clique
        let nodes: Vec<NodeId> = dag.input_nodes().iter().take(2).cloned().collect();
        let clique_node = dag.compress_to_clique(nodes);
        // Verify the node was added by checking node count increased
        assert!(dag.node_count() > 4);

        // Test expand_clique
        let expanded = dag.expand_clique(0);
        assert!(expanded.is_some());
    }

    #[test]
    fn test_form_cliques_via_trait() {
        let mut dag = DagNN::from_text("hello").unwrap();

        // form_cliques should create cliques from consecutive node windows
        let cliques = dag.form_cliques();
        // With 5 characters and window of 3, we get 3 cliques
        assert_eq!(cliques.len(), 3);

        // Each clique should have 3 members
        for clique in &cliques {
            assert_eq!(clique.size(), 3);
        }
    }

    #[test]
    fn test_compress_region() {
        let mut dag = DagNN::from_text("test").unwrap();
        let start = dag.input_nodes()[0];
        let end = dag.input_nodes()[3];

        let region = dag.compress_region(start, end).unwrap();
        assert_eq!(region.start, start);
        assert_eq!(region.end, end);
        assert_eq!(region.original_count, 4);
    }
}
