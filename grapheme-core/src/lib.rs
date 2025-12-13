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
use std::collections::{HashMap, HashSet};
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// Feature node for domain-specific processing
    Feature(usize),
    /// Temporal node for time-related processing
    Temporal(String),
    /// Keyword node (e.g., legal terms, operators)
    Keyword(String),
    /// Semantic unit node (e.g., for code or legal concepts)
    SemanticUnit(String),
    /// Pixel node for vision processing (row, col position)
    Pixel { row: usize, col: usize },
    /// Classification output node (class index)
    ClassOutput(usize),
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNN {
    /// The underlying directed acyclic graph
    pub graph: DiGraph<Node, Edge>,
    /// Topological ordering for efficient traversal
    pub topology: TopologicalOrder,
    /// Detected cliques (learned concepts)
    pub cliques: Vec<Clique>,
    /// Memory for learned transformations
    pub memory: GraphMemory,
    /// Input nodes set for O(1) lookup
    input_nodes: HashSet<NodeId>,
    /// Input nodes in insertion order
    input_nodes_order: Vec<NodeId>,
    /// Output nodes set for O(1) lookup
    output_nodes: HashSet<NodeId>,
    /// Output nodes in insertion order
    output_nodes_order: Vec<NodeId>,
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
            input_nodes: HashSet::new(),
            input_nodes_order: Vec::new(),
            output_nodes: HashSet::new(),
            output_nodes_order: Vec::new(),
        }
    }

    /// Build a DagNN from text (character by character, NO tokenization)
    pub fn from_text(text: &str) -> GraphemeResult<Self> {
        let mut dag = Self::new();

        let mut prev_node: Option<NodeId> = None;

        for (position, ch) in text.chars().enumerate() {
            let node = dag.graph.add_node(Node::input(ch, position));
            dag.input_nodes.insert(node);
            dag.input_nodes_order.push(node);

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
        self.input_nodes.insert(node);
        self.input_nodes_order.push(node);

        // Connect to previous if exists
        if self.input_nodes_order.len() > 1 {
            let prev = self.input_nodes_order[self.input_nodes_order.len() - 2];
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
        self.output_nodes.insert(node);
        self.output_nodes_order.push(node);
        node
    }

    /// Mark an existing node as an output node
    pub fn add_output_node(&mut self, node_id: NodeId) {
        if self.output_nodes.insert(node_id) {
            // insert() returns true if the value was newly inserted
            self.output_nodes_order.push(node_id);
        }
    }

    /// Perform a neuromorphic forward pass through the graph
    /// This propagates activations from inputs to outputs following the topology.
    pub fn neuromorphic_forward(&mut self) -> GraphemeResult<()> {
        // Update topology first
        self.update_topology()?;

        // Process nodes in topological order
        for &node_id in &self.topology.order {
            // Skip input nodes (they keep their activation)
            if self.input_nodes.contains(&node_id) {
                continue;
            }

            // Compute activation from incoming edges
            let incoming: Vec<(NodeId, f32)> = self.graph
                .neighbors_directed(node_id, petgraph::Direction::Incoming)
                .map(|src| {
                    let edge_weight = self.graph
                        .edges_connecting(src, node_id)
                        .next()
                        .map(|e| e.weight().weight)
                        .unwrap_or(1.0);
                    let src_activation = self.graph[src].activation;
                    (src, src_activation * edge_weight)
                })
                .collect();

            // Sum weighted activations
            let total: f32 = incoming.iter().map(|(_, w)| w).sum();

            // Apply LeakyReLU activation (GRAPHEME protocol)
            // LeakyReLU: x if x > 0 else 0.01 * x
            let alpha = 0.01;
            self.graph[node_id].activation = if total > 0.0 { total } else { alpha * total };
        }

        Ok(())
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

    /// Get input nodes (in insertion order)
    pub fn input_nodes(&self) -> &[NodeId] {
        &self.input_nodes_order
    }

    /// Get output nodes (in insertion order)
    pub fn output_nodes(&self) -> &[NodeId] {
        &self.output_nodes_order
    }

    /// Check if a node is an input node (O(1) lookup)
    pub fn is_input_node(&self, node_id: NodeId) -> bool {
        self.input_nodes.contains(&node_id)
    }

    /// Check if a node is an output node (O(1) lookup)
    pub fn is_output_node(&self, node_id: NodeId) -> bool {
        self.output_nodes.contains(&node_id)
    }

    /// Convert graph back to text
    pub fn to_text(&self) -> String {
        self.input_nodes_order
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
                let distance = node_pos.abs_diff(other_pos);

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
        let windows: Vec<Vec<NodeId>> = if self.input_nodes_order.len() >= 3 {
            self.input_nodes_order.windows(3)
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
        let level_0 = self.input_nodes_order.clone();

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

            // Apply LeakyReLU with dynamic √n normalization (GRAPHEME protocol)
            if incoming_count > 0 {
                let scale = 1.0 / (incoming_count as f32).sqrt();
                let normalized = scale * incoming_sum;
                let alpha = 0.01;
                let new_activation = if normalized > 0.0 { normalized } else { alpha * normalized };
                self.graph[node].activation = new_activation;
            }
            // Input nodes keep their original activation
        }

        Ok(())
    }

    fn forward_parallel(&mut self) -> GraphemeResult<()> {
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
// MemoryManager Trait (matching GRAPHEME_Vision.md)
// ============================================================================

/// Memory management trait for efficient graph operations
pub trait MemoryManager {
    /// Allocate nodes efficiently
    fn allocate_nodes(&mut self, count: usize) -> Vec<NodeId>;

    /// Garbage collection for disconnected nodes
    fn gc_disconnected(&mut self) -> usize;

    /// Incremental compression based on activation threshold
    fn compress_incremental(&mut self, threshold: f32) -> usize;
}

impl MemoryManager for DagNN {
    fn allocate_nodes(&mut self, count: usize) -> Vec<NodeId> {
        let mut nodes = Vec::with_capacity(count);
        for _ in 0..count {
            nodes.push(self.graph.add_node(Node::hidden()));
        }
        nodes
    }

    fn gc_disconnected(&mut self) -> usize {
        // Find nodes with no edges
        let mut disconnected = Vec::new();

        for node_idx in self.graph.node_indices() {
            let has_incoming = self.graph.edges_directed(node_idx, petgraph::Direction::Incoming).next().is_some();
            let has_outgoing = self.graph.edges_directed(node_idx, petgraph::Direction::Outgoing).next().is_some();

            // Skip input nodes (they're allowed to have no incoming edges)
            if !has_incoming && !has_outgoing
                && !self.input_nodes.contains(&node_idx) {
                    disconnected.push(node_idx);
                }
        }

        let count = disconnected.len();

        // Remove disconnected nodes (in reverse order to maintain indices)
        for node in disconnected.into_iter().rev() {
            self.graph.remove_node(node);
        }

        count
    }

    fn compress_incremental(&mut self, threshold: f32) -> usize {
        // Find regions with low activation variance
        let mut compressed_count = 0;

        // Find consecutive nodes with activation below threshold
        let mut low_activation_run: Vec<NodeId> = Vec::new();

        for &node in &self.input_nodes {
            let activation = self.graph[node].activation;

            if activation < threshold {
                low_activation_run.push(node);
            } else {
                // If we have a run of 3+ low activation nodes, compress them
                if low_activation_run.len() >= 3 {
                    let _compressed = self.graph.add_node(Node::compressed(CompressionType::Semantic));
                    compressed_count += low_activation_run.len();
                }
                low_activation_run.clear();
            }
        }

        // Handle trailing run
        if low_activation_run.len() >= 3 {
            let _compressed = self.graph.add_node(Node::compressed(CompressionType::Semantic));
            compressed_count += low_activation_run.len();
        }

        compressed_count
    }
}

// ============================================================================
// PatternMatcher Trait (matching GRAPHEME_Vision.md)
// ============================================================================

/// A learned pattern (graph motif)
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Pattern identifier
    pub id: usize,
    /// The sequence of node types in this pattern
    pub sequence: Vec<NodeType>,
    /// How often this pattern appears
    pub frequency: usize,
    /// The compressed node representing this pattern (if compressed)
    pub compressed_node: Option<NodeId>,
}

impl Pattern {
    /// Create a new pattern
    pub fn new(id: usize, sequence: Vec<NodeType>) -> Self {
        Self {
            id,
            sequence,
            frequency: 1,
            compressed_node: None,
        }
    }

    /// Get pattern length
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if pattern is empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

/// Hierarchical pattern structure
#[derive(Debug, Clone)]
pub struct PatternHierarchy {
    /// Patterns at each level (0 = character level, higher = more abstract)
    pub levels: Vec<Vec<Pattern>>,
}

/// Pattern matcher trait for learning and compressing patterns
pub trait PatternMatcher {
    /// Learn repeated patterns (graph motifs)
    fn learn_patterns(&self, min_frequency: usize) -> Vec<Pattern>;

    /// Compress learned patterns into single nodes
    fn compress_patterns(&mut self, patterns: &[Pattern]) -> usize;

    /// Extract hierarchical pattern structure
    fn extract_hierarchy(&self) -> PatternHierarchy;
}

impl PatternMatcher for DagNN {
    fn learn_patterns(&self, min_frequency: usize) -> Vec<Pattern> {
        use std::collections::HashMap;

        let mut pattern_counts: HashMap<Vec<NodeType>, usize> = HashMap::new();

        // Look for n-grams of size 2-5
        for window_size in 2..=5 {
            if self.input_nodes_order.len() < window_size {
                continue;
            }

            for window in self.input_nodes_order.windows(window_size) {
                let pattern: Vec<NodeType> = window.iter()
                    .map(|&n| self.graph[n].node_type.clone())
                    .collect();

                *pattern_counts.entry(pattern).or_insert(0) += 1;
            }
        }

        // Filter by minimum frequency and create Pattern structs
        let mut patterns: Vec<Pattern> = pattern_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_frequency)
            .enumerate()
            .map(|(id, (seq, freq))| {
                let mut p = Pattern::new(id, seq);
                p.frequency = freq;
                p
            })
            .collect();

        // Sort by frequency (descending)
        patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));

        patterns
    }

    fn compress_patterns(&mut self, patterns: &[Pattern]) -> usize {
        let mut compressed_count = 0;

        for pattern in patterns {
            if pattern.len() < 2 {
                continue;
            }

            // Create a pattern node for this pattern
            let pattern_bytes: Vec<u8> = pattern.sequence.iter()
                .filter_map(|nt| {
                    if let NodeType::Input(ch) = nt {
                        if ch.is_ascii() {
                            Some(*ch as u8)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            if !pattern_bytes.is_empty() {
                self.graph.add_node(Node::pattern(pattern_bytes));
                compressed_count += 1;
            }
        }

        compressed_count
    }

    fn extract_hierarchy(&self) -> PatternHierarchy {
        // Level 0: individual characters
        let level_0: Vec<Pattern> = self.input_nodes_order.iter()
            .enumerate()
            .map(|(id, &node)| {
                Pattern::new(id, vec![self.graph[node].node_type.clone()])
            })
            .collect();

        // Level 1: bigrams with frequency >= 2
        let level_1 = self.learn_patterns(2);

        // Level 2: trigrams and above with frequency >= 3
        let level_2: Vec<Pattern> = self.learn_patterns(3)
            .into_iter()
            .filter(|p| p.len() >= 3)
            .collect();

        let mut levels = vec![level_0];
        if !level_1.is_empty() {
            levels.push(level_1);
        }
        if !level_2.is_empty() {
            levels.push(level_2);
        }

        PatternHierarchy { levels }
    }
}

// ============================================================================
// DagNN Additional Methods (matching GRAPHEME_Vision.md)
// ============================================================================

impl DagNN {
    /// Spawn a processing chain for a character based on its complexity
    ///
    /// Example: "the" → 2-3 nodes, "quantum" → 5-6 nodes
    pub fn spawn_processing_chain(&mut self, ch: char, context: &[char]) -> Vec<NodeId> {
        let depth = Self::compute_processing_depth(ch, context);
        let mut chain = Vec::with_capacity(depth);

        // Add the input node
        let position = self.input_nodes_order.len();
        let input_node = self.add_character(ch, position);
        chain.push(input_node);

        // Add hidden nodes based on depth
        let mut prev_node = input_node;
        for _ in 1..depth {
            let hidden = self.add_hidden();
            self.add_edge(prev_node, hidden, Edge::sequential());
            chain.push(hidden);
            prev_node = hidden;
        }

        chain
    }

    /// Get nodes by activation level
    pub fn get_nodes_by_activation(&self, min_activation: f32) -> Vec<NodeId> {
        self.graph.node_indices()
            .filter(|&node| self.graph[node].activation >= min_activation)
            .collect()
    }

    /// Prune edges below a weight threshold
    pub fn prune_weak_edges(&mut self, threshold: f32) -> usize {
        let weak_edges: Vec<_> = self.graph.edge_indices()
            .filter(|&e| self.graph[e].weight < threshold)
            .collect();

        let count = weak_edges.len();

        for edge in weak_edges.into_iter().rev() {
            self.graph.remove_edge(edge);
        }

        count
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let total_activation: f32 = self.graph.node_indices()
            .map(|n| self.graph[n].activation)
            .sum();

        let node_count = self.node_count();
        let avg_activation = if node_count > 0 {
            total_activation / node_count as f32
        } else {
            0.0
        };

        GraphStats {
            node_count,
            edge_count: self.edge_count(),
            clique_count: self.cliques.len(),
            input_node_count: self.input_nodes.len(),
            output_node_count: self.output_nodes.len(),
            avg_activation,
        }
    }
}

/// Statistics about a DagNN graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Number of detected cliques
    pub clique_count: usize,
    /// Number of input nodes
    pub input_node_count: usize,
    /// Number of output nodes
    pub output_node_count: usize,
    /// Average node activation
    pub avg_activation: f32,
}

// ============================================================================
// Domain Brain Abstraction (for domain-specific brains)
// ============================================================================

/// Errors that can occur in domain-specific processing
#[derive(Error, Debug)]
pub enum DomainError {
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("Transformation error: {0}")]
    TransformError(String),
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    #[error("Internal error: {0}")]
    InternalError(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Result type for domain-specific operations
pub type DomainResult<T> = Result<T, DomainError>;

impl From<GraphemeError> for DomainError {
    fn from(err: GraphemeError) -> Self {
        DomainError::InternalError(err.to_string())
    }
}

/// Severity of validation issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Informational, not an error
    Info,
    /// Warning, may cause issues
    Warning,
    /// Error, must be fixed
    Error,
}

/// A validation issue found in domain-specific processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Severity of the issue
    pub severity: ValidationSeverity,
    /// Description of the issue
    pub message: String,
    /// Location in the graph (node index if applicable)
    pub location: Option<usize>,
}

impl ValidationIssue {
    /// Create a new validation issue
    pub fn new(severity: ValidationSeverity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            location: None,
        }
    }

    /// Create an info-level issue
    pub fn info(message: impl Into<String>) -> Self {
        Self::new(ValidationSeverity::Info, message)
    }

    /// Create a warning-level issue
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(ValidationSeverity::Warning, message)
    }

    /// Create an error-level issue
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(ValidationSeverity::Error, message)
    }

    /// Set the location (node index) for this issue
    pub fn with_location(mut self, location: usize) -> Self {
        self.location = Some(location);
        self
    }
}

/// A transformation rule in a domain brain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRule {
    /// Unique rule ID
    pub id: usize,
    /// Human-readable name
    pub name: String,
    /// Description of what this rule does
    pub description: String,
    /// Priority (higher = applied first)
    pub priority: i32,
}

impl DomainRule {
    /// Create a new domain rule
    pub fn new(id: usize, name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: description.into(),
            priority: 0,
        }
    }
}

/// Result of executing a domain-specific operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionResult {
    /// Numeric result (e.g., math calculation)
    Numeric(f64),
    /// Text result (e.g., code output)
    Text(String),
    /// Boolean result
    Bool(bool),
    /// Graph result (e.g., transformed graph)
    Graph(Box<DagNN>),
    /// List of results
    List(Vec<ExecutionResult>),
    /// No result (unit type)
    Unit,
    /// Error during execution
    Error(String),
}

impl ExecutionResult {
    /// Create a text result
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Create an error result
    pub fn error(s: impl Into<String>) -> Self {
        Self::Error(s.into())
    }
}

/// A training example for a domain brain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainExample {
    /// Input text/representation
    pub input: String,
    /// Expected output
    pub output: String,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl DomainExample {
    /// Create a new domain example
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to this example
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Trait for domain-specific brain implementations
///
/// A DomainBrain is responsible for:
/// - Parsing domain-specific input into graphs
/// - Validating graphs for correctness
/// - Executing/evaluating graphs
/// - Transforming graphs according to domain rules
/// - Generating training examples
pub trait DomainBrain: Send + Sync + std::fmt::Debug {
    /// Get the unique identifier for this domain
    fn domain_id(&self) -> &str;

    /// Get human-readable name
    fn domain_name(&self) -> &str;

    /// Get version string
    fn version(&self) -> &str;

    /// Check if this brain can process the given input
    fn can_process(&self, input: &str) -> bool;

    /// Parse input text into a domain-specific graph
    fn parse(&self, input: &str) -> DomainResult<DagNN>;

    /// Convert from core graph representation to domain-specific
    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN>;

    /// Convert from domain-specific to core graph representation
    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN>;

    /// Validate a graph for domain-specific correctness
    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>>;

    /// Execute/evaluate a graph
    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult>;

    /// Get all available transformation rules
    fn get_rules(&self) -> Vec<DomainRule>;

    /// Apply a transformation rule to a graph
    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN>;

    /// Generate training examples for this domain
    fn generate_examples(&self, count: usize) -> Vec<DomainExample>;

    /// Get node types this brain can produce (for vocabulary building)
    fn node_types(&self) -> Vec<NodeType> {
        vec![NodeType::Hidden, NodeType::Output]
    }

    /// Number of input nodes this brain expects (0 = variable)
    fn input_node_count(&self) -> usize {
        0
    }

    /// Number of output nodes this brain produces (0 = variable)
    fn output_node_count(&self) -> usize {
        1
    }
}

/// Registry for domain brains
#[derive(Debug, Default)]
pub struct BrainRegistry {
    brains: HashMap<String, Box<dyn DomainBrain>>,
}

impl BrainRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            brains: HashMap::new(),
        }
    }

    /// Register a brain
    pub fn register(&mut self, brain: Box<dyn DomainBrain>) {
        self.brains.insert(brain.domain_id().to_string(), brain);
    }

    /// Get a brain by domain ID
    pub fn get(&self, domain_id: &str) -> Option<&dyn DomainBrain> {
        self.brains.get(domain_id).map(|b| b.as_ref())
    }

    /// Get mutable reference to a brain by domain ID
    pub fn get_mut(&mut self, domain_id: &str) -> Option<&mut Box<dyn DomainBrain>> {
        self.brains.get_mut(domain_id)
    }

    /// List all registered domain IDs
    pub fn domains(&self) -> Vec<String> {
        self.brains.keys().cloned().collect()
    }

    /// Number of registered brains
    pub fn len(&self) -> usize {
        self.brains.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.brains.is_empty()
    }

    /// Find brains that can process the given input
    pub fn find_capable(&self, input: &str) -> Vec<&str> {
        self.brains
            .iter()
            .filter(|(_, brain)| brain.can_process(input))
            .map(|(id, _)| id.as_str())
            .collect()
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
        let _n1 = dag.add_character('H', 0);
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
        // May be empty for short text - just verify it doesn't panic
        let _clique_count = cliques.len();

        // Test compress_to_clique
        let nodes: Vec<NodeId> = dag.input_nodes().iter().take(2).cloned().collect();
        let _clique_node = dag.compress_to_clique(nodes);
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

    // Tests for backend-001: MemoryManager, PatternMatcher, additional methods

    #[test]
    fn test_memory_manager_allocate() {
        let mut dag = DagNN::new();

        // Test allocate_nodes
        let nodes = dag.allocate_nodes(5);
        assert_eq!(nodes.len(), 5);
        assert_eq!(dag.node_count(), 5);

        // All should be hidden nodes
        for node in nodes {
            assert!(matches!(dag.graph[node].node_type, NodeType::Hidden));
        }
    }

    #[test]
    fn test_memory_manager_gc() {
        let mut dag = DagNN::from_text("hi").unwrap();

        // Add some disconnected hidden nodes
        dag.graph.add_node(Node::hidden());
        dag.graph.add_node(Node::hidden());

        assert_eq!(dag.node_count(), 4); // 2 input + 2 disconnected

        // GC should remove the disconnected nodes
        let removed = dag.gc_disconnected();
        assert_eq!(removed, 2);
        assert_eq!(dag.node_count(), 2);
    }

    #[test]
    fn test_pattern_matcher_learn() {
        // Text with repeated patterns
        let dag = DagNN::from_text("abcabcabc").unwrap();

        // Learn patterns with min frequency 2
        let patterns = dag.learn_patterns(2);

        // Should find "abc" pattern repeated
        assert!(!patterns.is_empty());

        // "ab" should appear 3 times
        let ab_pattern: Vec<NodeType> = vec![NodeType::Input('a'), NodeType::Input('b')];
        let has_ab = patterns.iter().any(|p| p.sequence == ab_pattern);
        assert!(has_ab);
    }

    #[test]
    fn test_pattern_matcher_hierarchy() {
        let dag = DagNN::from_text("hello").unwrap();

        let hierarchy = dag.extract_hierarchy();

        // Level 0 should have all characters
        assert_eq!(hierarchy.levels[0].len(), 5);
    }

    #[test]
    fn test_spawn_processing_chain() {
        let mut dag = DagNN::new();

        // Simple ASCII letter should have depth 2
        let chain_a = dag.spawn_processing_chain('a', &[]);
        assert_eq!(chain_a.len(), 2);

        // Math symbol should have depth 3
        let chain_plus = dag.spawn_processing_chain('+', &[]);
        assert_eq!(chain_plus.len(), 3);

        // Complex Unicode should have deeper chain
        let chain_han = dag.spawn_processing_chain('你', &[]);
        assert!(chain_han.len() >= 4);
    }

    #[test]
    fn test_prune_weak_edges() {
        let mut dag = DagNN::from_text("abc").unwrap();

        // Add some weak edges
        let nodes = dag.input_nodes().to_vec();
        dag.add_edge(nodes[0], nodes[2], Edge::skip(0.1));
        dag.add_edge(nodes[1], nodes[2], Edge::semantic(0.05));

        let initial_edges = dag.edge_count();
        assert_eq!(initial_edges, 4); // 2 sequential + 2 weak

        // Prune edges below 0.5
        let pruned = dag.prune_weak_edges(0.5);
        assert_eq!(pruned, 2);
        assert_eq!(dag.edge_count(), 2); // Only sequential edges remain
    }

    #[test]
    fn test_graph_stats() {
        let dag = DagNN::from_text("test").unwrap();

        let stats = dag.stats();

        assert_eq!(stats.node_count, 4);
        assert_eq!(stats.edge_count, 3);
        assert_eq!(stats.input_node_count, 4);
        assert_eq!(stats.clique_count, 0);
        assert_eq!(stats.avg_activation, 1.0); // All input nodes have activation 1.0
    }

    #[test]
    fn test_get_nodes_by_activation() {
        let mut dag = DagNN::from_text("ab").unwrap();

        // Set different activations
        let nodes = dag.input_nodes().to_vec();
        dag.graph[nodes[0]].activation = 0.8;
        dag.graph[nodes[1]].activation = 0.3;

        // Get nodes with activation >= 0.5
        let high_activation = dag.get_nodes_by_activation(0.5);
        assert_eq!(high_activation.len(), 1);
        assert_eq!(high_activation[0], nodes[0]);
    }
}
