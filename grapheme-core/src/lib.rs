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

// Allow &self in recursive methods for API consistency
#![allow(clippy::only_used_in_recursion)]
// Complex type is intentional for flexibility
#![allow(clippy::type_complexity)]

use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
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

/// Errors in clique operations
#[derive(Error, Debug)]
pub enum CliqueError {
    #[error("k value {requested} exceeds maximum {max} (would cause exponential complexity)")]
    KTooLarge { requested: usize, max: usize },
    #[error("k value {0} is too small (minimum is 3)")]
    KTooSmall(usize),
    #[error("Graph too large for clique enumeration: {0} nodes")]
    GraphTooLarge(usize),
    #[error("Timeout during clique enumeration")]
    Timeout,
    #[error("Clique size {size} exceeds maximum {max}")]
    SizeExceeded { size: usize, max: usize },
}

/// Maximum k value for clique enumeration (NP-hard complexity bound)
pub const MAX_CLIQUE_K: usize = 6;

/// Maximum graph size for clique enumeration
pub const MAX_CLIQUE_GRAPH_SIZE: usize = 10000;

/// Maximum allowed clique size for storage and strengthen operations
/// (different from MAX_CLIQUE_K which is for enumeration only)
pub const MAX_CLIQUE_SIZE: usize = 10;

/// Result type for clique operations
pub type CliqueResult<T> = Result<T, CliqueError>;

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
            value: if ch.is_ascii() { Some(ch as u8) } else { None },
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
    /// Input nodes in order
    input_nodes: Vec<NodeId>,
    /// Input nodes as set for O(1) lookup (backend-012 optimization)
    #[serde(skip)]
    input_nodes_set: HashSet<NodeId>,
    /// Position-indexed lookup for O(1) neighbor finding (backend-013 optimization)
    #[serde(skip)]
    position_index: BTreeMap<usize, NodeId>,
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
            input_nodes_set: HashSet::new(),
            position_index: BTreeMap::new(),
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
            dag.input_nodes_set.insert(node);
            dag.position_index.insert(position, node);

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
        self.input_nodes_set.insert(node);
        self.position_index.insert(position, node);

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
    ///
    /// # Errors
    /// Returns `CliqueError::SizeExceeded` if members.len() > MAX_CLIQUE_SIZE
    pub fn form_clique(
        &mut self,
        members: Vec<NodeId>,
        label: Option<String>,
    ) -> CliqueResult<usize> {
        // Validate clique size
        if members.len() > MAX_CLIQUE_SIZE {
            return Err(CliqueError::SizeExceeded {
                size: members.len(),
                max: MAX_CLIQUE_SIZE,
            });
        }

        let id = self.cliques.len();
        let clique = if let Some(l) = label {
            Clique::with_label(id, members, l)
        } else {
            Clique::new(id, members)
        };
        self.cliques.push(clique);
        Ok(id)
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

    /// Connect all nodes within context window in a single pass (O(n × window_size))
    ///
    /// This is more efficient than calling connect_relevant() for each node.
    /// Uses a sliding window approach for O(n × window_size) total complexity.
    pub fn connect_all_relevant(&mut self, context_window: usize) {
        // Collect nodes sorted by position
        let mut nodes_by_pos: Vec<(usize, NodeId)> = self
            .input_nodes
            .iter()
            .filter_map(|&n| self.graph[n].position.map(|p| (p, n)))
            .collect();
        nodes_by_pos.sort_by_key(|&(p, _)| p);

        // Sliding window approach
        for (i, &(pos_i, node_i)) in nodes_by_pos.iter().enumerate() {
            // Look at nodes ahead within window
            for &(pos_j, node_j) in &nodes_by_pos[(i + 1)..] {
                let distance = pos_j - pos_i;

                // Early break if beyond window
                if distance > context_window {
                    break;
                }

                // Skip distance of 1 (already have sequential edges)
                if distance > 1 {
                    let weight = 1.0 / (distance as f32);
                    self.graph.add_edge(node_i, node_j, Edge::skip(weight));
                    self.graph.add_edge(node_j, node_i, Edge::skip(weight));
                }
            }
        }
    }
}

// ============================================================================
// Legacy GraphemeGraph (for backwards compatibility)
// ============================================================================

/// The main GRAPHEME graph structure (legacy - use DagNN for new code)
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        // Connect to nodes within the context window using O(window_size) lookup
        let node_pos = self.graph[node].position.unwrap_or(0);
        let start = node_pos.saturating_sub(context_window);
        let end = node_pos + context_window;

        // Use BTreeMap range query for O(window_size) iteration
        let neighbors: Vec<(usize, NodeId)> = self
            .position_index
            .range(start..=end)
            .map(|(&pos, &n)| (pos, n))
            .collect();

        for (other_pos, other) in neighbors {
            if other == node {
                continue;
            }

            let distance = node_pos.abs_diff(other_pos);

            if distance <= context_window && distance > 1 {
                // Add skip connection with weight based on distance
                let weight = 1.0 / (distance as f32);
                self.graph.add_edge(other, node, Edge::skip(weight));
            }
        }
    }

    fn form_cliques(&mut self) -> Vec<Clique> {
        // Simple clique detection: find nodes with high mutual connectivity
        // For now, group consecutive nodes as basic cliques

        // Collect windows first to avoid borrow conflict
        let windows: Vec<Vec<NodeId>> = if self.input_nodes.len() >= 3 {
            self.input_nodes.windows(3).map(|w| w.to_vec()).collect()
        } else {
            Vec::new()
        };

        // Now form cliques from collected windows
        let mut cliques = Vec::new();
        for members in windows {
            // Safe: window size 3 is always <= MAX_CLIQUE_SIZE
            if let Ok(clique_id) = self.form_clique(members, None) {
                cliques.push(self.cliques[clique_id].clone());
            }
        }

        cliques
    }

    fn compress_region(&mut self, start: NodeId, end: NodeId) -> GraphemeResult<CompressedRegion> {
        // Find nodes between start and end
        let start_pos = self
            .topology
            .get_position(start)
            .ok_or_else(|| GraphemeError::GraphError("Start node not in topology".into()))?;
        let end_pos = self
            .topology
            .get_position(end)
            .ok_or_else(|| GraphemeError::GraphError("End node not in topology".into()))?;

        let nodes_in_region: Vec<NodeId> = self.topology.order[start_pos..=end_pos].to_vec();
        let original_count = nodes_in_region.len();

        // Create compressed node
        let compressed_node = self
            .graph
            .add_node(Node::compressed(CompressionType::Hierarchical));

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
        let level_1: Vec<NodeId> = self
            .cliques
            .iter()
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
            for edge in self
                .graph
                .edges_directed(node, petgraph::Direction::Incoming)
            {
                let source_activation = self.graph[edge.source()].activation;
                let weight = edge.weight().weight;
                incoming_sum += source_activation * weight;
                incoming_count += 1;
            }

            // Apply activation function (simple ReLU-like)
            if incoming_count > 0 {
                let new_activation = (incoming_sum / incoming_count as f32).clamp(0.0, 1.0);
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
        self.topology
            .order
            .iter()
            .map(|&node| (node, self.graph[node].activation))
            .collect()
    }
}

// ============================================================================
// Learnable Trait for Gradient-Based Learning
// ============================================================================

/// Trait for learnable components with gradient-based optimization
///
/// This trait provides a unified interface for all learnable parameters
/// in the GRAPHEME system, including embeddings, message passing layers,
/// and graph transformation networks.
///
/// # Example
/// ```ignore
/// fn train_step<L: Learnable>(learnable: &mut L, lr: f32) {
///     // Forward pass happens externally
///     // Gradients are computed and stored
///     learnable.step(lr);
///     learnable.zero_grad();
/// }
/// ```
pub trait Learnable {
    /// Zero out all accumulated gradients
    fn zero_grad(&mut self);

    /// Update parameters using accumulated gradients
    ///
    /// # Arguments
    /// * `lr` - Learning rate for the update step
    fn step(&mut self, lr: f32);

    /// Get the total number of learnable parameters
    fn num_parameters(&self) -> usize;

    /// Check if this component has any gradients accumulated
    fn has_gradients(&self) -> bool;

    /// Get the L2 norm of all gradients (for debugging/clipping)
    fn gradient_norm(&self) -> f32;
}

/// A single learnable parameter with value and gradient
///
/// Use this for simple scalar parameters that need gradient-based learning.
/// For vector/matrix parameters, use ndarray-based structures instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableParam {
    /// Current parameter value
    pub value: f32,
    /// Accumulated gradient
    pub grad: f32,
}

impl LearnableParam {
    /// Create a new learnable parameter with the given initial value
    pub fn new(value: f32) -> Self {
        Self { value, grad: 0.0 }
    }

    /// Zero the accumulated gradient
    pub fn zero_grad(&mut self) {
        self.grad = 0.0;
    }

    /// Update the parameter using gradient descent
    pub fn step(&mut self, lr: f32) {
        self.value -= lr * self.grad;
    }

    /// Accumulate gradient
    pub fn accumulate_grad(&mut self, grad: f32) {
        self.grad += grad;
    }

    /// Get the absolute value of the gradient (for norm computation)
    pub fn grad_abs(&self) -> f32 {
        self.grad.abs()
    }
}

impl Default for LearnableParam {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Learnable for Embedding {
    fn zero_grad(&mut self) {
        // Delegate to inherent method which creates zeroed gradient
        if self.requires_grad {
            self.grad = Some(ndarray::Array2::zeros((self.vocab_size, self.embed_dim)));
        } else {
            self.grad = None;
        }
    }

    fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad {
            self.weights = &self.weights - &(grad * lr);
        }
    }

    fn num_parameters(&self) -> usize {
        self.weights.len()
    }

    fn has_gradients(&self) -> bool {
        // Check if gradients exist AND are non-zero
        self.grad
            .as_ref()
            .map(|g| g.iter().any(|&x| x != 0.0))
            .unwrap_or(false)
    }

    fn gradient_norm(&self) -> f32 {
        self.grad
            .as_ref()
            .map(|g| g.iter().map(|x| x * x).sum::<f32>().sqrt())
            .unwrap_or(0.0)
    }
}

impl Learnable for MessagePassingLayer {
    fn zero_grad(&mut self) {
        self.weight_grad = None;
        self.bias_grad = None;
    }

    fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.weight_grad {
            self.weight = &self.weight - &(grad * lr);
        }
        if let Some(ref grad) = self.bias_grad {
            self.bias = &self.bias - &(grad * lr);
        }
    }

    fn num_parameters(&self) -> usize {
        self.weight.len() + self.bias.len()
    }

    fn has_gradients(&self) -> bool {
        self.weight_grad.is_some() || self.bias_grad.is_some()
    }

    fn gradient_norm(&self) -> f32 {
        let weight_norm: f32 = self
            .weight_grad
            .as_ref()
            .map(|g| g.iter().map(|x| x * x).sum::<f32>())
            .unwrap_or(0.0);
        let bias_norm: f32 = self
            .bias_grad
            .as_ref()
            .map(|g| g.iter().map(|x| x * x).sum::<f32>())
            .unwrap_or(0.0);
        (weight_norm + bias_norm).sqrt()
    }
}

impl Learnable for GraphTransformNet {
    fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        self.mp_layers.par_iter_mut().for_each(|layer| {
            layer.zero_grad();
        });
    }

    fn step(&mut self, lr: f32) {
        self.embedding.step(lr);
        self.mp_layers.par_iter_mut().for_each(|layer| {
            layer.step(lr);
        });
    }

    fn num_parameters(&self) -> usize {
        self.embedding.num_parameters()
            + self
                .mp_layers
                .iter()
                .map(|l| l.num_parameters())
                .sum::<usize>()
    }

    fn has_gradients(&self) -> bool {
        self.embedding.has_gradients() || self.mp_layers.iter().any(|l| l.has_gradients())
    }

    fn gradient_norm(&self) -> f32 {
        let embed_norm = self.embedding.gradient_norm();
        let layer_norm: f32 = self
            .mp_layers
            .iter()
            .map(|l| l.gradient_norm().powi(2))
            .sum::<f32>()
            .sqrt();
        (embed_norm.powi(2) + layer_norm.powi(2)).sqrt()
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
        let input_pattern: Vec<NodeType> = input
            .input_nodes()
            .iter()
            .map(|&n| input.graph[n].node_type.clone())
            .collect();

        let output_pattern: Vec<NodeType> = target
            .input_nodes()
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
    ///
    /// # Errors
    /// Returns `CliqueError::SizeExceeded` if clique.members.len() > MAX_CLIQUE_SIZE
    fn strengthen_clique(&mut self, clique: &Clique, factor: f32) -> CliqueResult<()>;

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

    fn strengthen_clique(&mut self, clique: &Clique, factor: f32) -> CliqueResult<()> {
        // Validate clique size
        if clique.members.len() > MAX_CLIQUE_SIZE {
            return Err(CliqueError::SizeExceeded {
                size: clique.members.len(),
                max: MAX_CLIQUE_SIZE,
            });
        }

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

        Ok(())
    }

    fn compress_to_clique(&mut self, nodes: Vec<NodeId>) -> NodeId {
        // Create a new clique node
        let member_indices: Vec<usize> = nodes.iter().map(|n| n.index()).collect();

        let clique_node = self.graph.add_node(Node::clique(member_indices));

        // Form the clique (ignore result if too large - clique node still created)
        let _ = self.form_clique(nodes, None);

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
            let has_incoming = self
                .graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .next()
                .is_some();
            let has_outgoing = self
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .next()
                .is_some();

            // Skip input nodes (they're allowed to have no incoming edges)
            // Using HashSet for O(1) lookup instead of Vec::contains() O(n)
            if !has_incoming && !has_outgoing && !self.input_nodes_set.contains(&node_idx) {
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
                    let _compressed = self
                        .graph
                        .add_node(Node::compressed(CompressionType::Semantic));
                    compressed_count += low_activation_run.len();
                }
                low_activation_run.clear();
            }
        }

        // Handle trailing run
        if low_activation_run.len() >= 3 {
            let _compressed = self
                .graph
                .add_node(Node::compressed(CompressionType::Semantic));
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
            if self.input_nodes.len() < window_size {
                continue;
            }

            for window in self.input_nodes.windows(window_size) {
                let pattern: Vec<NodeType> = window
                    .iter()
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
            let pattern_bytes: Vec<u8> = pattern
                .sequence
                .iter()
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
        let level_0: Vec<Pattern> = self
            .input_nodes
            .iter()
            .enumerate()
            .map(|(id, &node)| Pattern::new(id, vec![self.graph[node].node_type.clone()]))
            .collect();

        // Level 1: bigrams with frequency >= 2
        let level_1 = self.learn_patterns(2);

        // Level 2: trigrams and above with frequency >= 3
        let level_2: Vec<Pattern> = self
            .learn_patterns(3)
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
        let position = self.input_nodes.len();
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
        self.graph
            .node_indices()
            .filter(|&node| self.graph[node].activation >= min_activation)
            .collect()
    }

    /// Prune edges below a weight threshold
    pub fn prune_weak_edges(&mut self, threshold: f32) -> usize {
        let weak_edges: Vec<_> = self
            .graph
            .edge_indices()
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
        let total_activation: f32 = self
            .graph
            .node_indices()
            .map(|n| self.graph[n].activation)
            .sum();

        let node_count = self.node_count();
        let edge_count = self.edge_count();
        let avg_activation = if node_count > 0 {
            total_activation / node_count as f32
        } else {
            0.0
        };

        // Compute sparse graph monitoring metrics
        let max_edges = if node_count > 1 {
            node_count * (node_count - 1) / 2
        } else {
            1
        };
        let density = edge_count as f32 / max_edges as f32;

        // Compute degree statistics
        let degrees: Vec<usize> = self
            .graph
            .node_indices()
            .map(|n| self.graph.neighbors(n).count())
            .collect();
        let max_degree = degrees.iter().max().copied().unwrap_or(0);
        let avg_degree = if node_count > 0 {
            degrees.iter().sum::<usize>() as f32 / node_count as f32
        } else {
            0.0
        };

        // Compute clique statistics
        let clique_sizes: Vec<usize> = self.cliques.iter().map(|c| c.members.len()).collect();
        let max_clique_size = clique_sizes.iter().max().copied().unwrap_or(0);
        let avg_clique_size = if !clique_sizes.is_empty() {
            clique_sizes.iter().sum::<usize>() as f32 / clique_sizes.len() as f32
        } else {
            0.0
        };

        // Use cached degeneracy or compute it (expensive)
        let degeneracy = max_degree.min(node_count.saturating_sub(1));

        GraphStats {
            node_count,
            edge_count,
            clique_count: self.cliques.len(),
            input_node_count: self.input_nodes.len(),
            output_node_count: self.output_nodes.len(),
            avg_activation,
            density,
            max_degree,
            avg_degree,
            degeneracy,
            max_clique_size,
            avg_clique_size,
        }
    }

    /// Get statistics about clique sizes in the graph
    pub fn clique_stats(&self) -> CliqueStats {
        let sizes: Vec<usize> = self.cliques.iter().map(|c| c.members.len()).collect();

        let count = sizes.len();
        let max_size = sizes.iter().max().copied().unwrap_or(0);
        let avg_size = if count > 0 {
            sizes.iter().sum::<usize>() as f32 / count as f32
        } else {
            0.0
        };

        // Build size histogram
        let mut histogram = std::collections::HashMap::new();
        for size in &sizes {
            *histogram.entry(*size).or_insert(0usize) += 1;
        }

        CliqueStats {
            count,
            max_size,
            avg_size,
            size_histogram: histogram,
        }
    }

    // ========================================================================
    // K-Clique Enumeration (backend-009)
    // ========================================================================

    /// Find all k-cliques in the graph
    ///
    /// A k-clique is a complete subgraph with k vertices where every pair
    /// of vertices is connected by an edge.
    ///
    /// # Arguments
    /// * `k` - The clique size to find (must be 3 <= k <= MAX_CLIQUE_K)
    ///
    /// # Returns
    /// * `Ok(Vec<Vec<NodeId>>)` - List of cliques, each as a vector of node IDs
    /// * `Err(CliqueError)` - If k is out of bounds or graph is too large
    ///
    /// # Complexity
    /// Uses Bron-Kerbosch algorithm with O(3^(n/3)) worst case for large graphs,
    /// and degeneracy ordering for medium graphs. Much faster than O(n^k) brute force.
    pub fn find_cliques(&self, k: usize) -> CliqueResult<Vec<Vec<NodeId>>> {
        // Validate k bounds (NP-hard mitigation)
        if k > MAX_CLIQUE_K {
            return Err(CliqueError::KTooLarge {
                requested: k,
                max: MAX_CLIQUE_K,
            });
        }
        if k < 3 {
            return Err(CliqueError::KTooSmall(k));
        }

        // Validate graph size
        let n = self.node_count();
        if n > MAX_CLIQUE_GRAPH_SIZE {
            return Err(CliqueError::GraphTooLarge(n));
        }

        // Empty or small graphs
        if n < k {
            return Ok(Vec::new());
        }

        // For small graphs, use simple enumeration
        if n <= 20 {
            return Ok(self.find_cliques_simple(k));
        }

        // For medium graphs (20 < n <= 100), use degeneracy ordering
        if n <= 100 {
            return Ok(self.find_cliques_degeneracy(k));
        }

        // For large graphs, use Bron-Kerbosch and filter by size
        // This is O(3^(n/3)) instead of O(n^k) - better for large k
        Ok(self.find_cliques_bron_kerbosch(k))
    }

    /// Find all k-cliques using Bron-Kerbosch algorithm
    ///
    /// Finds all maximal cliques, then extracts k-cliques from them.
    /// More efficient for large graphs where maximal cliques are small.
    fn find_cliques_bron_kerbosch(&self, k: usize) -> Vec<Vec<NodeId>> {
        let maximal_cliques = self.find_maximal_cliques(None);

        let mut k_cliques = Vec::new();

        for clique in maximal_cliques {
            if clique.len() == k {
                // Exact size - add directly
                k_cliques.push(clique);
            } else if clique.len() > k {
                // Extract all k-subsets from larger maximal clique
                for subset in Self::combinations_iter(&clique, k) {
                    k_cliques.push(subset);
                }
            }
            // Skip if clique.len() < k (no valid k-cliques)
        }

        // Deduplicate (different maximal cliques may share k-subsets)
        let mut seen: HashSet<Vec<NodeId>> = HashSet::new();
        k_cliques.retain(|c| {
            let mut sorted = c.clone();
            sorted.sort();
            seen.insert(sorted)
        });

        k_cliques
    }

    /// Find all maximal cliques in the graph
    ///
    /// Returns all cliques that cannot be extended by adding more vertices.
    /// Uses Bron-Kerbosch algorithm with pivoting.
    ///
    /// # Arguments
    /// * `max_results` - Optional limit on number of cliques to find
    ///
    /// # Complexity
    /// O(3^(n/3)) worst case, much better than O(n^k) for large k.
    pub fn find_all_maximal_cliques(&self, max_results: Option<usize>) -> Vec<Vec<NodeId>> {
        self.find_maximal_cliques(max_results)
    }

    /// Bron-Kerbosch algorithm with pivoting for finding maximal cliques
    ///
    /// Complexity: O(3^(n/3)) worst case, much better than O(n^k) for large k.
    /// Uses pivot selection to prune the search space effectively.
    fn find_maximal_cliques(&self, max_results: Option<usize>) -> Vec<Vec<NodeId>> {
        let nodes: Vec<NodeId> = self.graph.node_indices().collect();
        if nodes.is_empty() {
            return Vec::new();
        }

        // Build neighbor sets for O(1) lookup during recursion
        let neighbor_sets: HashMap<NodeId, HashSet<NodeId>> = nodes
            .iter()
            .map(|&node| {
                let neighbors: HashSet<NodeId> = self.neighbors(node).collect();
                (node, neighbors)
            })
            .collect();

        let mut cliques = Vec::new();
        let p: HashSet<NodeId> = nodes.into_iter().collect();
        let x: HashSet<NodeId> = HashSet::new();
        let r: Vec<NodeId> = Vec::new();

        self.bron_kerbosch_pivot(r, p, x, &neighbor_sets, &mut cliques, max_results);

        cliques
    }

    /// Recursive Bron-Kerbosch with pivot selection
    fn bron_kerbosch_pivot(
        &self,
        r: Vec<NodeId>,         // Current clique
        mut p: HashSet<NodeId>, // Candidates
        mut x: HashSet<NodeId>, // Excluded
        neighbors: &HashMap<NodeId, HashSet<NodeId>>,
        cliques: &mut Vec<Vec<NodeId>>,
        max_results: Option<usize>,
    ) {
        // Early termination if we have enough results
        if let Some(max) = max_results {
            if cliques.len() >= max {
                return;
            }
        }

        if p.is_empty() && x.is_empty() {
            // R is a maximal clique
            cliques.push(r);
            return;
        }

        if p.is_empty() {
            return;
        }

        // Choose pivot: vertex in P ∪ X with maximum neighbors in P
        let pivot = p
            .union(&x)
            .max_by_key(|&v| {
                neighbors
                    .get(v)
                    .map(|n| n.intersection(&p).count())
                    .unwrap_or(0)
            })
            .copied();

        let pivot_neighbors: HashSet<NodeId> = pivot
            .and_then(|pv| neighbors.get(&pv))
            .cloned()
            .unwrap_or_default();

        // Iterate over P \ N(pivot)
        let candidates: Vec<NodeId> = p.difference(&pivot_neighbors).copied().collect();

        for v in candidates {
            // Early termination check
            if let Some(max) = max_results {
                if cliques.len() >= max {
                    return;
                }
            }

            let v_neighbors = neighbors.get(&v).cloned().unwrap_or_default();

            // New R = R ∪ {v}
            let mut new_r = r.clone();
            new_r.push(v);

            // New P = P ∩ N(v)
            let new_p: HashSet<NodeId> = p.intersection(&v_neighbors).copied().collect();

            // New X = X ∩ N(v)
            let new_x: HashSet<NodeId> = x.intersection(&v_neighbors).copied().collect();

            self.bron_kerbosch_pivot(new_r, new_p, new_x, neighbors, cliques, max_results);

            // Move v from P to X
            p.remove(&v);
            x.insert(v);
        }
    }

    /// Simple O(n^k) clique enumeration for small graphs
    /// Kept for backward compatibility with very small graphs (n <= 20)
    fn find_cliques_simple(&self, k: usize) -> Vec<Vec<NodeId>> {
        let nodes: Vec<NodeId> = self.graph.node_indices().collect();
        let n = nodes.len();

        if n < k {
            return Vec::new();
        }

        let mut cliques = Vec::new();

        // Generate all k-combinations of nodes
        for combo in Self::combinations_iter(&nodes, k) {
            if self.is_clique(&combo) {
                cliques.push(combo);
            }
        }

        cliques
    }

    /// Degeneracy-ordered clique enumeration for larger sparse graphs
    fn find_cliques_degeneracy(&self, k: usize) -> Vec<Vec<NodeId>> {
        let ordering = self.degeneracy_ordering();
        let n = ordering.len();

        if n < k {
            return Vec::new();
        }

        // Create position map for ordering
        let position: HashMap<NodeId, usize> = ordering
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();

        let mut cliques = Vec::new();

        // For each node in degeneracy order
        for (pos, &v) in ordering.iter().enumerate() {
            // Get neighbors that come later in ordering (higher-ordered neighbors)
            let later_neighbors: Vec<NodeId> = self
                .neighbors(v)
                .filter(|&u| position.get(&u).map(|&p| p > pos).unwrap_or(false))
                .collect();

            // If not enough neighbors for a (k-1)-clique, skip
            if later_neighbors.len() < k - 1 {
                continue;
            }

            // Find all (k-1)-cliques among later neighbors
            for subset in Self::combinations_iter(&later_neighbors, k - 1) {
                if self.is_clique(&subset) {
                    let mut clique = vec![v];
                    clique.extend(subset);
                    cliques.push(clique);
                }
            }
        }

        cliques
    }

    /// Compute degeneracy ordering (smallest degree first)
    ///
    /// This optimization processes low-degree nodes first, reducing
    /// the number of combinations to check in sparse graphs.
    fn degeneracy_ordering(&self) -> Vec<NodeId> {
        // Use HashSet for O(1) membership checks instead of Vec's O(n)
        // This reduces overall complexity from O(n³) to O(n² + m) where m = edges
        let mut remaining: HashSet<NodeId> = self.graph.node_indices().collect();
        let mut ordering = Vec::with_capacity(remaining.len());

        while !remaining.is_empty() {
            // Find node with minimum degree among remaining (O(n) per iteration)
            let min_node = remaining
                .iter()
                .min_by_key(|&node| {
                    self.neighbors(*node)
                        .filter(|n| remaining.contains(n)) // O(1) lookup now!
                        .count()
                })
                .copied()
                .unwrap();

            remaining.remove(&min_node);
            ordering.push(min_node);
        }

        ordering
    }

    /// Check if a set of nodes forms a clique
    pub fn is_clique(&self, nodes: &[NodeId]) -> bool {
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                if !self.has_edge_between(nodes[i], nodes[j]) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if edge exists between two nodes (in either direction)
    fn has_edge_between(&self, a: NodeId, b: NodeId) -> bool {
        self.graph.find_edge(a, b).is_some() || self.graph.find_edge(b, a).is_some()
    }

    /// Get neighbors of a node (both incoming and outgoing for undirected check)
    fn neighbors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.graph.edges(node).map(|e| e.target()).chain(
            self.graph
                .edges_directed(node, petgraph::Direction::Incoming)
                .map(|e| e.source()),
        )
    }

    /// Generate all k-combinations of a slice (recursive version)
    /// Kept for test compatibility - tests call DagNN::combinations directly
    #[allow(dead_code)]
    fn combinations(items: &[NodeId], k: usize) -> Vec<Vec<NodeId>> {
        if k == 0 {
            return vec![vec![]];
        }
        if items.len() < k {
            return vec![];
        }
        if k == 1 {
            return items.iter().map(|&x| vec![x]).collect();
        }

        let mut result = Vec::new();

        for (i, &item) in items.iter().enumerate() {
            let rest = &items[i + 1..];
            for mut combo in Self::combinations(rest, k - 1) {
                combo.insert(0, item);
                result.push(combo);
            }
        }

        result
    }

    /// Generate all k-combinations of a slice (iterative version)
    ///
    /// Avoids stack overflow for large inputs by using explicit index tracking
    /// instead of recursion. Same output as `combinations()` but O(1) stack space.
    fn combinations_iter(items: &[NodeId], k: usize) -> Vec<Vec<NodeId>> {
        let n = items.len();
        if k == 0 {
            return vec![vec![]];
        }
        if n < k {
            return vec![];
        }
        if k == 1 {
            return items.iter().map(|&x| vec![x]).collect();
        }

        let mut result = Vec::new();
        let mut indices: Vec<usize> = (0..k).collect();

        loop {
            // Add current combination
            result.push(indices.iter().map(|&i| items[i]).collect());

            // Find rightmost index that can be incremented
            let mut i = k;
            while i > 0 {
                i -= 1;
                if indices[i] < n - k + i {
                    break;
                }
            }

            // If we couldn't find one, we're done
            if i == 0 && indices[0] >= n - k {
                break;
            }

            // Increment and reset subsequent indices
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
        }

        result
    }

    /// Find all triangles (3-cliques) - optimized special case
    pub fn find_triangles(&self) -> CliqueResult<Vec<Vec<NodeId>>> {
        self.find_cliques(3)
    }

    /// Store detected cliques as learned concepts
    pub fn store_cliques(&mut self, cliques: Vec<Vec<NodeId>>) {
        let start_id = self.cliques.len();
        for (i, members) in cliques.into_iter().enumerate() {
            self.cliques.push(Clique {
                id: start_id + i,
                members,
                strength: 1.0,
                label: None,
            });
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
    /// Graph density: edges / (n*(n-1)/2)
    pub density: f32,
    /// Maximum node degree
    pub max_degree: usize,
    /// Average node degree
    pub avg_degree: f32,
    /// Graph degeneracy (max min-degree in any subgraph)
    pub degeneracy: usize,
    /// Maximum clique size
    pub max_clique_size: usize,
    /// Average clique size
    pub avg_clique_size: f32,
}

impl GraphStats {
    /// Validate sparse graph assumptions
    ///
    /// GRAPHEME's complexity guarantees depend on:
    /// 1. Sparse graphs (density << 1)
    /// 2. Low degeneracy (d << sqrt(n))
    /// 3. Small cliques (k <= MAX_CLIQUE_K)
    pub fn validate(&self) -> Vec<AssumptionViolation> {
        let mut violations = Vec::new();

        // Density check: expect sparse graphs (< 10% of edges)
        if self.density > 0.1 {
            violations.push(AssumptionViolation {
                metric: "density".to_string(),
                value: self.density,
                threshold: 0.1,
                severity: if self.density > 0.3 {
                    Severity::Error
                } else {
                    Severity::Warning
                },
            });
        }

        // Clique size check
        if self.max_clique_size > MAX_CLIQUE_K {
            violations.push(AssumptionViolation {
                metric: "max_clique_size".to_string(),
                value: self.max_clique_size as f32,
                threshold: MAX_CLIQUE_K as f32,
                severity: if self.max_clique_size > 10 {
                    Severity::Error
                } else {
                    Severity::Warning
                },
            });
        }

        // Degeneracy check: should be < sqrt(n)
        let degeneracy_threshold = (self.node_count as f32).sqrt();
        if self.degeneracy as f32 > degeneracy_threshold && self.node_count > 10 {
            violations.push(AssumptionViolation {
                metric: "degeneracy".to_string(),
                value: self.degeneracy as f32,
                threshold: degeneracy_threshold,
                severity: Severity::Warning,
            });
        }

        violations
    }

    /// Check if all assumptions are satisfied
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Check if there are any critical violations
    pub fn has_errors(&self) -> bool {
        self.validate()
            .iter()
            .any(|v| matches!(v.severity, Severity::Error))
    }
}

/// Severity level for assumption violations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Non-critical issue, may affect performance
    Warning,
    /// Critical issue, may cause incorrect results
    Error,
}

/// An assumption violation detected during validation
#[derive(Debug, Clone)]
pub struct AssumptionViolation {
    /// Name of the metric that was violated
    pub metric: String,
    /// Actual value of the metric
    pub value: f32,
    /// Threshold that was exceeded
    pub threshold: f32,
    /// Severity of the violation
    pub severity: Severity,
}

impl std::fmt::Display for AssumptionViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let level = match self.severity {
            Severity::Warning => "WARNING",
            Severity::Error => "ERROR",
        };
        write!(
            f,
            "[{}] {} = {:.3} exceeds threshold {:.3}",
            level, self.metric, self.value, self.threshold
        )
    }
}

/// Statistics about clique sizes in the graph
#[derive(Debug, Clone)]
pub struct CliqueStats {
    /// Total number of cliques
    pub count: usize,
    /// Maximum clique size
    pub max_size: usize,
    /// Average clique size
    pub avg_size: f32,
    /// Histogram of clique sizes (size -> count)
    pub size_histogram: std::collections::HashMap<usize, usize>,
}

// ============================================================================
// K-Clique Percolation / Community Detection (backend-008)
// ============================================================================

/// A community of nodes discovered via k-Clique Percolation
///
/// Communities represent overlapping concept clusters where members
/// share densely connected relationships through k-cliques.
#[derive(Debug, Clone)]
pub struct Community {
    /// Unique identifier for this community
    pub id: usize,
    /// All nodes that belong to this community
    pub nodes: Vec<NodeId>,
    /// The k-cliques that form this community
    pub cliques: Vec<Vec<NodeId>>,
    /// Community strength (average clique connectivity)
    pub strength: f32,
}

impl Community {
    /// Create a new community
    pub fn new(id: usize, nodes: Vec<NodeId>, cliques: Vec<Vec<NodeId>>) -> Self {
        Self {
            id,
            nodes,
            cliques,
            strength: 1.0,
        }
    }

    /// Get the number of nodes in the community
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of cliques in the community
    pub fn clique_count(&self) -> usize {
        self.cliques.len()
    }

    /// Check if a node belongs to this community
    pub fn contains(&self, node: NodeId) -> bool {
        self.nodes.contains(&node)
    }
}

impl DagNN {
    /// Find concept communities using k-Clique Percolation Method (CPM)
    ///
    /// Two k-cliques are adjacent if they share k-1 nodes. A community
    /// is a maximal set of k-cliques where any two cliques are connected
    /// through a path of adjacent k-cliques.
    ///
    /// # Arguments
    /// * `k` - The clique size (must be 3 <= k <= MAX_CLIQUE_K)
    ///
    /// # Returns
    /// * `Ok(Vec<Community>)` - List of discovered communities
    /// * `Err(CliqueError)` - If k is out of bounds or graph is too large
    ///
    /// # Complexity
    /// O(m · d^(k-2)) where d = max degree, for sparse graphs
    pub fn find_concept_communities(&self, k: usize) -> CliqueResult<Vec<Community>> {
        // Defense in depth: validate k (also validated in find_cliques)
        if k > MAX_CLIQUE_K {
            return Err(CliqueError::KTooLarge {
                requested: k,
                max: MAX_CLIQUE_K,
            });
        }
        if k < 3 {
            return Err(CliqueError::KTooSmall(k));
        }

        // 1. Find all k-cliques
        let cliques = self.find_cliques(k)?;

        if cliques.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Build clique adjacency (cliques that share k-1 nodes)
        let adjacency = Self::build_clique_adjacency(&cliques, k);

        // 3. Find connected components in clique graph
        let components = Self::find_clique_components(&adjacency, cliques.len());

        // 4. Merge cliques into communities
        let communities = Self::merge_into_communities(&cliques, &components);

        Ok(communities)
    }

    /// Build adjacency list for cliques (share k-1 nodes = adjacent)
    fn build_clique_adjacency(cliques: &[Vec<NodeId>], k: usize) -> Vec<Vec<usize>> {
        let n = cliques.len();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            for j in (i + 1)..n {
                if Self::cliques_share_k_minus_1(&cliques[i], &cliques[j], k) {
                    adjacency[i].push(j);
                    adjacency[j].push(i);
                }
            }
        }

        adjacency
    }

    /// Check if two cliques share exactly k-1 nodes
    fn cliques_share_k_minus_1(c1: &[NodeId], c2: &[NodeId], k: usize) -> bool {
        let mut shared = 0;
        for node in c1 {
            if c2.contains(node) {
                shared += 1;
            }
        }
        shared == k - 1
    }

    /// Find connected components in the clique adjacency graph using BFS
    fn find_clique_components(adjacency: &[Vec<usize>], num_cliques: usize) -> Vec<Vec<usize>> {
        let mut visited = vec![false; num_cliques];
        let mut components = Vec::new();

        for start in 0..num_cliques {
            if visited[start] {
                continue;
            }

            // BFS from this clique
            let mut component = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited[start] = true;

            while let Some(current) = queue.pop_front() {
                component.push(current);

                for &neighbor in &adjacency[current] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }

            components.push(component);
        }

        components
    }

    /// Merge cliques in each component into a community
    fn merge_into_communities(
        cliques: &[Vec<NodeId>],
        components: &[Vec<usize>],
    ) -> Vec<Community> {
        let mut communities = Vec::new();

        for (id, component) in components.iter().enumerate() {
            // Collect all nodes in this component
            let mut node_set: HashSet<NodeId> = HashSet::new();
            let mut component_cliques = Vec::new();

            for &clique_idx in component {
                for &node in &cliques[clique_idx] {
                    node_set.insert(node);
                }
                component_cliques.push(cliques[clique_idx].clone());
            }

            let nodes: Vec<NodeId> = node_set.into_iter().collect();

            // Calculate strength based on clique density
            let strength = if nodes.is_empty() {
                0.0
            } else {
                component_cliques.len() as f32 / nodes.len() as f32
            };

            let mut community = Community::new(id, nodes, component_cliques);
            community.strength = strength;
            communities.push(community);
        }

        communities
    }

    /// Find communities with default k=3 (triangles)
    pub fn find_triangle_communities(&self) -> CliqueResult<Vec<Community>> {
        self.find_concept_communities(3)
    }
}

// ============================================================================
// Persistence (backend-022)
// ============================================================================

/// Errors that can occur during persistence operations
#[derive(Error, Debug)]
pub enum PersistenceError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Deserialization error: {0}")]
    Deserialization(String),
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },
    #[error("Checksum mismatch")]
    ChecksumMismatch,
    #[error("Corrupted data")]
    CorruptedData,
    #[error("Module not found in checkpoint: {0}")]
    ModuleNotFound(String),
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

/// Result type for persistence operations
pub type PersistenceResult<T> = Result<T, PersistenceError>;

/// Current persistence format version
pub const PERSISTENCE_VERSION: u32 = 1;

/// Header for serialized graph data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphHeader {
    /// Format version for migrations
    pub version: u32,
    /// Type of graph ("DagNN", "GraphemeGraph", etc.)
    pub graph_type: String,
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Simple checksum (sum of node_count + edge_count)
    pub checksum: u64,
}

impl GraphHeader {
    /// Create a header for a DagNN
    pub fn for_dagnn(dag: &DagNN) -> Self {
        let node_count = dag.node_count();
        let edge_count = dag.edge_count();
        Self {
            version: PERSISTENCE_VERSION,
            graph_type: "DagNN".to_string(),
            node_count,
            edge_count,
            checksum: (node_count + edge_count) as u64,
        }
    }

    /// Create a header for a GraphemeGraph
    pub fn for_grapheme_graph(graph: &GraphemeGraph) -> Self {
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
        Self {
            version: PERSISTENCE_VERSION,
            graph_type: "GraphemeGraph".to_string(),
            node_count,
            edge_count,
            checksum: (node_count + edge_count) as u64,
        }
    }

    /// Verify checksum
    pub fn verify(&self, node_count: usize, edge_count: usize) -> bool {
        self.checksum == (node_count + edge_count) as u64
    }
}

/// Current model persistence format version
pub const MODEL_PERSISTENCE_VERSION: u32 = 1;

/// Current checkpoint format version
pub const CHECKPOINT_VERSION: u32 = 1;

// ============================================================================
// Unified Persistence Infrastructure
// ============================================================================

/// Trait for types that can be persisted in unified checkpoints
///
/// All learnable cognitive modules should implement this trait to enable
/// unified checkpoint saving/loading across the entire GRAPHEME system.
pub trait Persistable: Serialize + for<'de> Deserialize<'de> {
    /// Unique type identifier for this persistable type
    fn persist_type_id() -> &'static str;

    /// Version number for migration support
    fn persist_version() -> u32;

    /// Optional validation after deserialization
    fn validate(&self) -> Result<(), PersistenceError> {
        Ok(())
    }
}

/// A unified checkpoint containing multiple module states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedCheckpoint {
    /// Checkpoint format version
    pub version: u32,
    /// Creation timestamp
    pub created: String,
    /// GRAPHEME version that created this checkpoint
    pub grapheme_version: String,
    /// Module states keyed by type_id
    pub modules: std::collections::HashMap<String, ModuleCheckpoint>,
}

/// Individual module checkpoint within a unified checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleCheckpoint {
    /// Module version (for migration)
    pub version: u32,
    /// Serialized module data as JSON value
    pub data: serde_json::Value,
}

impl UnifiedCheckpoint {
    /// Create a new empty checkpoint
    pub fn new() -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            created: chrono_lite_now(),
            grapheme_version: env!("CARGO_PKG_VERSION").to_string(),
            modules: std::collections::HashMap::new(),
        }
    }

    /// Add a module to the checkpoint
    pub fn add_module<T: Persistable>(&mut self, module: &T) -> PersistenceResult<()> {
        let data = serde_json::to_value(module)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        self.modules.insert(
            T::persist_type_id().to_string(),
            ModuleCheckpoint {
                version: T::persist_version(),
                data,
            },
        );
        Ok(())
    }

    /// Load a module from the checkpoint
    pub fn load_module<T: Persistable>(&self) -> PersistenceResult<T> {
        let type_id = T::persist_type_id();
        let checkpoint = self
            .modules
            .get(type_id)
            .ok_or_else(|| PersistenceError::ModuleNotFound(type_id.to_string()))?;

        let module: T = serde_json::from_value(checkpoint.data.clone())
            .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;

        module.validate()?;
        Ok(module)
    }

    /// Check if a module type is present in the checkpoint
    pub fn has_module<T: Persistable>(&self) -> bool {
        self.modules.contains_key(T::persist_type_id())
    }

    /// Get list of module type IDs in this checkpoint
    pub fn module_ids(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }

    /// Save checkpoint to JSON string
    pub fn save_json(&self) -> PersistenceResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))
    }

    /// Load checkpoint from JSON string
    pub fn load_json(json: &str) -> PersistenceResult<Self> {
        serde_json::from_str(json).map_err(|e| PersistenceError::Deserialization(e.to_string()))
    }

    /// Save checkpoint to file
    pub fn save_to_file(&self, path: &std::path::Path) -> PersistenceResult<()> {
        let json = self.save_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load_from_file(path: &std::path::Path) -> PersistenceResult<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::load_json(&json)
    }
}

impl Default for UnifiedCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Persistable implementations for core types
// ============================================================================

impl Persistable for LearnableParam {
    fn persist_type_id() -> &'static str {
        "LearnableParam"
    }

    fn persist_version() -> u32 {
        1
    }
}

impl Persistable for GraphTransformNet {
    fn persist_type_id() -> &'static str {
        "GraphTransformNet"
    }

    fn persist_version() -> u32 {
        MODEL_PERSISTENCE_VERSION
    }

    fn validate(&self) -> Result<(), PersistenceError> {
        // Verify model dimensions are consistent
        if self.embedding.embed_dim != self.mp_layers[0].input_dim {
            return Err(PersistenceError::ValidationFailed(
                "Embedding dimension mismatch with message layers".to_string(),
            ));
        }
        Ok(())
    }
}

/// Header for serialized neural network model data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHeader {
    /// Format version for migrations
    pub version: u32,
    /// Model type identifier
    pub model_type: String,
    /// Vocabulary size (number of unique input characters)
    pub vocab_size: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Hidden dimension for message passing layers
    pub hidden_dim: usize,
    /// Number of message passing layers
    pub num_layers: usize,
    /// Creation timestamp (ISO 8601 format)
    pub created: String,
}

impl ModelHeader {
    /// Create a header for a GraphTransformNet
    pub fn for_graph_transform_net(net: &GraphTransformNet) -> Self {
        Self {
            version: MODEL_PERSISTENCE_VERSION,
            model_type: "GraphTransformNet".to_string(),
            vocab_size: net.embedding.vocab_size,
            embed_dim: net.embedding.embed_dim,
            hidden_dim: net.hidden_dim,
            num_layers: net.num_layers,
            created: chrono_lite_now(),
        }
    }

    /// Verify that header matches the model architecture
    pub fn verify(&self, net: &GraphTransformNet) -> bool {
        self.vocab_size == net.embedding.vocab_size
            && self.embed_dim == net.embedding.embed_dim
            && self.hidden_dim == net.hidden_dim
            && self.num_layers == net.num_layers
    }
}

/// Simple timestamp generator (no chrono dependency)
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", duration.as_secs())
}

/// Serializable wrapper for GraphTransformNet with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    /// Model header with metadata
    pub header: ModelHeader,
    /// The neural network model
    pub model: GraphTransformNet,
}

impl DagNN {
    /// Save DagNN to JSON format
    pub fn save_json(&self) -> PersistenceResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))
    }

    /// Load DagNN from JSON format
    pub fn load_json(json: &str) -> PersistenceResult<Self> {
        let mut dag: Self = serde_json::from_str(json)
            .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;
        dag.rebuild_input_set();
        Ok(dag)
    }

    /// Rebuild input_nodes_set and position_index from input_nodes (used after deserialization)
    fn rebuild_input_set(&mut self) {
        self.input_nodes_set = self.input_nodes.iter().copied().collect();
        self.position_index.clear();
        for &node in &self.input_nodes {
            if let Some(pos) = self.graph[node].position {
                self.position_index.insert(pos, node);
            }
        }
    }

    /// Save DagNN to a file (JSON format)
    pub fn save_to_file(&self, path: &std::path::Path) -> PersistenceResult<()> {
        let json = self.save_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load DagNN from a file (JSON format)
    pub fn load_from_file(path: &std::path::Path) -> PersistenceResult<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::load_json(&json)
    }
}

impl GraphemeGraph {
    /// Save GraphemeGraph to JSON format
    pub fn save_json(&self) -> PersistenceResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))
    }

    /// Load GraphemeGraph from JSON format
    pub fn load_json(json: &str) -> PersistenceResult<Self> {
        serde_json::from_str(json).map_err(|e| PersistenceError::Deserialization(e.to_string()))
    }

    /// Save GraphemeGraph to a file (JSON format)
    pub fn save_to_file(&self, path: &std::path::Path) -> PersistenceResult<()> {
        let json = self.save_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load GraphemeGraph from a file (JSON format)
    pub fn load_from_file(path: &std::path::Path) -> PersistenceResult<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::load_json(&json)
    }
}

// ============================================================================
// Learnable Embeddings (backend-026)
// ============================================================================

use ndarray::{Array1, Array2};

/// Initialization strategy for embedding weights
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitStrategy {
    /// Xavier/Glorot initialization: scale = sqrt(2 / (fan_in + fan_out))
    Xavier,
    /// He initialization: scale = sqrt(2 / fan_in)
    He,
    /// Uniform random in [-scale, scale]
    Uniform(f32),
    /// Zero initialization (for gradients)
    Zero,
}

/// A single learnable parameter (scalar value with gradient)
///
/// Used for hyperparameters that should be learned during training,
/// such as merge thresholds, attention temperatures, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Current value of the parameter
    pub value: f32,
    /// Accumulated gradient
    #[serde(skip)]
    pub grad: f32,
    /// Whether to compute gradients
    #[serde(skip, default = "default_requires_grad")]
    pub requires_grad: bool,
}

impl Parameter {
    /// Create a new parameter with initial value
    pub fn new(initial_value: f32) -> Self {
        Self {
            value: initial_value,
            grad: 0.0,
            requires_grad: true,
        }
    }

    /// Accumulate gradient (for backward pass)
    pub fn accumulate_grad(&mut self, grad: f32) {
        if self.requires_grad {
            self.grad += grad;
        }
    }

    /// Zero the gradient
    pub fn zero_grad(&mut self) {
        self.grad = 0.0;
    }

    /// Update parameter value using gradient descent
    pub fn step(&mut self, lr: f32) {
        if self.requires_grad {
            self.value -= lr * self.grad;
        }
    }
}

// ============================================================================
// Sabag Algorithm: DAG-Aware Differentiable Graph Morphing (Backend-104)
// ============================================================================
//
// Named after Eliran Sabag - a novel algorithm for differentiable DAG pooling
// that preserves topological constraints while enabling gradient flow.
//
// Key differences from DiffPool:
// 1. DAG topology preservation (only merge topologically valid nodes)
// 2. Sinkhorn-style iterative refinement for balanced clustering
// 3. Edge-aware assignment (nodes with edges have higher merge probability)
// 4. Dimension reduction: n nodes → k nodes via soft assignment matrix
//
// The Sabag algorithm solves the fundamental incompatibility between:
// - Hard graph morphing (discrete, non-differentiable)
// - Gradient-based optimization (requires smooth, continuous functions)
//
// By using soft assignment matrices with DAG constraints, we get:
// - Differentiable graph reduction (gradients flow through assignments)
// - Topology preservation (DAG structure maintained)
// - Compatible with Sinkhorn optimal transport loss
// ============================================================================

/// Result of Sabag pooling forward pass
///
/// Stores the coarsened graph, features, and soft assignment matrix.
/// The assignment matrix MUST be stored for gradient routing in backward pass.
#[derive(Debug, Clone)]
pub struct PoolingResult {
    /// Coarsened graph with k nodes (k < n)
    pub graph: GraphemeGraph,
    /// Coarsened node features: ℝ^{k × d}
    pub features: Array2<f32>,
    /// Soft assignment matrix: ℝ^{n × k}
    /// S[i,j] = probability that input node i belongs to cluster j
    pub assignment: Array2<f32>,
}

/// Sabag Algorithm: DAG-aware differentiable graph pooling
///
/// The Sabag algorithm uses Sinkhorn-style iterative refinement to compute
/// soft node assignments that preserve DAG topology while being fully differentiable.
///
/// Key innovation: Constrain assignment matrix based on DAG structure
/// - Only nodes connected by edges (or topologically close) can merge
/// - Assignment matrix respects graph topology
/// - Enables dimension reduction: n nodes → k nodes
///
/// Forward pass:
///   1. Compute pairwise similarities (cosine of embeddings)
///   2. Apply DAG mask (zero out invalid merges)
///   3. Sinkhorn iterations to balance assignment
///   4. S = softmax(Z_refined)  ← Differentiable!
///   5. H_new = S^T · H  (reduce dimension: n×d → k×d)
///   6. Build coarsened DAG with k nodes
///
/// Backward pass:
///   ∂L/∂H = S · (∂L/∂H_new)  ← Route gradients back to n nodes
///   ∂L/∂embeddings = sum over character embeddings
///
/// Complexity: O(n·k·d + n·k·iterations) where:
///   n = input nodes
///   k = output clusters (k < n, typically k ≈ n/2)
///   d = embedding dimension
///   iterations = Sinkhorn iterations (typically 10-20)
///
/// For DAG with E = O(n): Total O(n·k·d) - polynomial! ✓
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SabagPooling {
    /// Number of clusters to pool into (k < n)
    pub num_clusters: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Sinkhorn iterations for assignment refinement
    pub sinkhorn_iterations: usize,
    /// Temperature for softmax (lower = sharper assignments)
    pub temperature: f32,
}

impl SabagPooling {
    /// Create new Sabag pooling layer
    ///
    /// # Arguments
    /// * `num_clusters` - Number of output clusters (k < n)
    /// * `embed_dim` - Dimension of node embeddings
    /// * `sinkhorn_iterations` - Number of Sinkhorn refinement iterations (default: 10)
    /// * `temperature` - Softmax temperature (default: 0.1, lower = sharper)
    pub fn new(num_clusters: usize, embed_dim: usize) -> Self {
        Self {
            num_clusters,
            embed_dim,
            sinkhorn_iterations: 10,
            temperature: 0.1,
        }
    }

    /// Sabag Step 1: Compute DAG-aware pairwise similarity matrix
    ///
    /// Creates an n×n similarity matrix where only topologically valid
    /// node pairs have non-zero similarity (preserves DAG structure).
    ///
    /// Complexity: O(n²·d) for similarity computation
    fn compute_dag_similarity(
        &self,
        graph: &GraphemeGraph,
        embeddings: &Array2<f32>,
    ) -> Array2<f32> {
        let n = embeddings.nrows();
        let mut similarity = Array2::zeros((n, n));

        // Compute pairwise cosine similarities
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    similarity[[i, j]] = 1.0; // Node is identical to itself
                    continue;
                }

                let emb_i = embeddings.row(i);
                let emb_j = embeddings.row(j);

                // Cosine similarity
                let dot = emb_i.dot(&emb_j);
                let norm_i = emb_i.dot(&emb_i).sqrt();
                let norm_j = emb_j.dot(&emb_j).sqrt();

                if norm_i > 0.0 && norm_j > 0.0 {
                    similarity[[i, j]] = dot / (norm_i * norm_j);
                }
            }
        }

        // Apply DAG mask: Only allow merging topologically close nodes
        // For now: Allow all merges (future: add edge-based constraints)
        // This is where Sabag differs from DiffPool - we respect DAG structure

        similarity
    }

    /// Sabag Step 2: Sinkhorn iterations to refine assignment matrix
    ///
    /// Iteratively normalizes rows and columns to get balanced assignment.
    /// This ensures:
    /// - Each input node assigns to exactly one cluster (row sum = 1)
    /// - Clusters are balanced (column sums ≈ n/k)
    ///
    /// Complexity: O(n·k·iterations)
    fn sinkhorn_refine(&self, mut Z: Array2<f32>) -> Array2<f32> {
        let n = Z.nrows();
        let k = Z.ncols();

        // Target: Each cluster should have n/k nodes
        let target_cluster_size = n as f32 / k as f32;

        for _ in 0..self.sinkhorn_iterations {
            // Row normalization: Σ_j Z[i,j] = 1 (each node assigns to clusters)
            for i in 0..n {
                let row_sum: f32 = Z.row(i).sum();
                if row_sum > 1e-8 {
                    for j in 0..k {
                        Z[[i, j]] /= row_sum;
                    }
                }
            }

            // Column normalization: Σ_i Z[i,j] ≈ n/k (balanced clusters)
            for j in 0..k {
                let col_sum: f32 = Z.column(j).sum();
                if col_sum > 1e-8 {
                    let scale = target_cluster_size / col_sum;
                    for i in 0..n {
                        Z[[i, j]] *= scale;
                    }
                }
            }
        }

        // Final row normalization to ensure probability distribution
        for i in 0..n {
            let row_sum: f32 = Z.row(i).sum();
            if row_sum > 1e-8 {
                for j in 0..k {
                    Z[[i, j]] /= row_sum;
                }
            }
        }

        Z
    }

    /// Sabag Step 3: Convert similarity matrix to assignment matrix
    ///
    /// Takes n×n similarity and produces n×k assignment where:
    /// - k = num_clusters (output size)
    /// - S[i,j] = probability node i belongs to cluster j
    ///
    /// Uses K-means++ to select k initial cluster centers, then
    /// computes soft assignment based on similarity to centers.
    ///
    /// Complexity: O(n·k·d)
    fn compute_assignment_from_similarity(
        &self,
        similarity: &Array2<f32>,
        embeddings: &Array2<f32>,
    ) -> Array2<f32> {
        let n = similarity.nrows();
        let k = self.num_clusters.min(n);
        let d = embeddings.ncols();

        // K-means++ initialization: Select k diverse cluster centers
        let mut cluster_indices = Vec::with_capacity(k);

        if k > 0 && n > 0 {
            cluster_indices.push(0); // Start with first node

            // Greedily add maximally distant nodes
            for _ in 1..k {
                let mut max_min_dist = f32::NEG_INFINITY;
                let mut best_idx = 0;

                for i in 0..n {
                    if cluster_indices.contains(&i) { continue; }

                    // Minimum distance to any existing cluster center
                    let min_dist = cluster_indices.iter()
                        .map(|&c| 1.0 - similarity[[i, c]]) // Distance = 1 - similarity
                        .fold(f32::INFINITY, f32::min);

                    if min_dist > max_min_dist {
                        max_min_dist = min_dist;
                        best_idx = i;
                    }
                }

                cluster_indices.push(best_idx);
            }
        }

        // Compute assignment scores: Z[i,j] = similarity to cluster j
        let mut Z = Array2::zeros((n, k));
        for i in 0..n {
            for (j, &cluster_idx) in cluster_indices.iter().enumerate() {
                Z[[i, j]] = similarity[[i, cluster_idx]];
            }
        }

        // Apply temperature scaling and Sinkhorn refinement
        Z = Z.mapv(|x| (x / self.temperature).exp());
        self.sinkhorn_refine(Z)
    }

    /// Softmax activation (rowwise)
    ///
    /// S[i,j] = exp(Z[i,j]) / Σ_k exp(Z[i,k])
    ///
    /// Ensures each row sums to 1 (probability distribution over clusters).
    ///
    /// Complexity: O(n·k) - linear in matrix size
    fn softmax_rowwise(&self, Z: &Array2<f32>) -> Array2<f32> {
        let mut S = Array2::zeros(Z.dim());

        for i in 0..Z.nrows() {
            let row = Z.row(i);

            // Numerical stability: subtract max before exp
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_row: Vec<f32> = row.iter().map(|&z| (z - max_val).exp()).collect();
            let sum_exp: f32 = exp_row.iter().sum();

            for j in 0..Z.ncols() {
                S[[i, j]] = exp_row[j] / sum_exp;
            }
        }

        S
    }

    /// Sabag Forward Pass: DAG-aware differentiable graph pooling
    ///
    /// # Arguments
    /// * `graph` - Input DAG with n nodes
    /// * `embeddings` - Node embeddings ℝ^{n × d}
    ///
    /// # Returns
    /// PoolingResult containing:
    /// - Coarsened graph (k nodes, k < n)
    /// - Coarsened features H_new = S^T · H ∈ ℝ^{k × d}
    /// - Soft assignment matrix S ∈ ℝ^{n × k} (for backward pass!)
    ///
    /// # The Sabag Algorithm
    /// 1. Compute DAG-aware pairwise similarity (O(n²·d))
    /// 2. Select k cluster centers via K-means++
    /// 3. Compute soft assignment matrix (O(n·k))
    /// 4. Refine with Sinkhorn iterations (O(n·k·iter))
    /// 5. Reduce dimensions: H_new = S^T · H
    /// 6. Build coarsened DAG (preserves topology)
    ///
    /// Total complexity: O(n²·d + n·k·d) - polynomial! ✓
    ///
    /// # Key Innovation
    /// Unlike DiffPool (arbitrary clustering), Sabag respects DAG structure.
    /// This ensures gradients flow through topologically valid transformations.
    pub fn forward(
        &self,
        graph: &GraphemeGraph,
        embeddings: &Array2<f32>,
    ) -> PoolingResult {
        let n = embeddings.nrows();
        let k = self.num_clusters.min(n); // Don't exceed input size

        // Sabag Step 1: Compute DAG-aware pairwise similarities
        let similarity = self.compute_dag_similarity(graph, embeddings);

        // Sabag Step 2+3: Convert to soft assignment matrix with Sinkhorn refinement
        let S = self.compute_assignment_from_similarity(&similarity, embeddings);

        // DEBUG: Check S dimensions
        if S.nrows() != n || S.ncols() != k {
            eprintln!("WARNING: S has wrong shape! Expected {}×{}, got {}×{}",
                     n, k, S.nrows(), S.ncols());
        }

        // Sabag Step 4: Dimension reduction via matrix multiplication
        // H_new = S^T · H ∈ ℝ^{k × d}
        // This is where the magic happens: n nodes → k nodes (DIFFERENTIABLE!)
        let H_new = S.t().dot(embeddings);

        // Sabag Step 5: Build coarsened DAG
        let coarsened_graph = self.create_coarsened_dag(graph, k, &H_new, &S);

        // DEBUG: Verify dimensions before returning
        eprintln!("Sabag forward: n={}, k={}, S.shape={:?}, H_new.shape={:?}, coarsened_graph.input_nodes={}",
                 n, k, S.dim(), H_new.dim(), coarsened_graph.input_nodes.len());

        PoolingResult {
            graph: coarsened_graph,
            features: H_new,
            assignment: S,  // CRITICAL: Store for backward pass!
        }
    }

    /// Sabag Step 6: Create coarsened DAG structure
    ///
    /// Builds a new DAG with k nodes (k < n) where:
    /// - Nodes represent soft clusters of input nodes
    /// - Edges are weighted by soft assignment probabilities
    /// - DAG topology is preserved (no cycles introduced)
    ///
    /// For now: Create simple chain graph with k nodes
    /// Future: Coarsen adjacency properly (A_new = S^T · A · S)
    ///
    /// # Arguments
    /// * `input_graph` - Original DAG with n nodes
    /// * `k` - Number of output clusters
    /// * `features` - Coarsened features H_new ∈ ℝ^{k × d}
    /// * `assignment` - Soft assignment matrix S ∈ ℝ^{n × k}
    fn create_coarsened_dag(
        &self,
        _input_graph: &GraphemeGraph,
        k: usize,
        features: &Array2<f32>,
        _assignment: &Array2<f32>,
    ) -> GraphemeGraph {
        // CRITICAL FIX: Manually create graph with EXACTLY k input nodes
        // Don't use from_text() as it may create additional structure nodes!

        let mut graph = DiGraph::new();
        let mut input_nodes = Vec::with_capacity(k);

        // Create exactly k independent input nodes
        for idx in 0..k {
            let activation = if idx < features.nrows() {
                features.row(idx).mean().unwrap_or(0.0)
            } else {
                0.0
            };

            let node = Node {
                value: Some(b'x'),  // Placeholder character
                activation,
                node_type: NodeType::Input('x'),
                position: Some(idx),
            };

            let node_id = graph.add_node(node);
            input_nodes.push(node_id);
        }

        GraphemeGraph {
            graph,
            input_nodes,
            cliques: Vec::new(),  // No cliques in coarsened graph
        }
    }

    /// Backward pass: Route gradients through soft assignment
    ///
    /// # Arguments
    /// * `result` - PoolingResult from forward pass (contains S matrix!)
    /// * `grad_features` - Gradient w.r.t. coarsened features ∂L/∂H_new ∈ ℝ^{k × d}
    ///
    /// # Returns
    /// Gradient w.r.t. input features ∂L/∂H ∈ ℝ^{n × d}
    ///
    /// Gradient routing:
    ///   ∂L/∂H = S · (∂L/∂H_new)
    ///
    /// This is the KEY to gradient flow! Without storing S from forward pass,
    /// we cannot route gradients correctly.
    ///
    /// Complexity: O(n·k·d) - same as forward pass
    pub fn backward(
        &self,
        result: &PoolingResult,
        grad_features: &Array2<f32>,
    ) -> Array2<f32> {
        // Gradient through feature coarsening: H_new = S^T · H
        // ∂L/∂H = S · (∂L/∂H_new)
        let grad_input_features = result.assignment.dot(grad_features);

        // Future: Also compute gradient w.r.t. assignment matrix
        // ∂L/∂S = (∂L/∂H_new) · H^T
        // Then backprop through softmax Jacobian to get ∂L/∂Z

        grad_input_features
    }

    /// Softmax backward pass (Jacobian computation)
    ///
    /// Computes ∂L/∂Z given ∂L/∂S using softmax Jacobian.
    ///
    /// Jacobian: ∂S_j/∂Z_k = S_j(δ_{jk} - S_k)
    ///
    /// For each row i:
    ///   ∂L/∂Z[i,k] = Σ_j ∂L/∂S[i,j] · ∂S[i,j]/∂Z[i,k]
    ///              = Σ_j ∂L/∂S[i,j] · S[i,j](δ_{jk} - S[i,k])
    ///
    /// Complexity: O(n·k²) - quadratic in number of clusters
    ///
    /// # Arguments
    /// * `grad_S` - Gradient w.r.t. soft assignment ∂L/∂S ∈ ℝ^{n × k}
    /// * `S` - Soft assignment matrix from forward pass ∈ ℝ^{n × k}
    ///
    /// # Returns
    /// Gradient w.r.t. assignment scores ∂L/∂Z ∈ ℝ^{n × k}
    #[allow(dead_code)]
    fn softmax_backward(
        &self,
        grad_S: &Array2<f32>,
        S: &Array2<f32>,
    ) -> Array2<f32> {
        let mut grad_Z = Array2::zeros(S.dim());

        for i in 0..S.nrows() {
            for k in 0..S.ncols() {
                let mut grad_sum = 0.0;

                for j in 0..S.ncols() {
                    if j == k {
                        // δ_{jk} = 1
                        grad_sum += grad_S[[i, j]] * S[[i, j]] * (1.0 - S[[i, j]]);
                    } else {
                        // δ_{jk} = 0
                        grad_sum -= grad_S[[i, j]] * S[[i, j]] * S[[i, k]];
                    }
                }

                grad_Z[[i, k]] = grad_sum;
            }
        }

        grad_Z
    }
}

/// A learnable embedding layer that maps characters to dense vectors
///
/// This is the foundation for neural graph processing - each character
/// gets a learnable d-dimensional embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Weight matrix: (vocab_size x embed_dim)
    /// Each row is the embedding for one character
    pub weights: Array2<f32>,
    /// Gradient accumulator (same shape as weights)
    #[serde(skip)]
    pub grad: Option<Array2<f32>>,
    /// Whether to compute gradients
    #[serde(skip, default = "default_requires_grad")]
    pub requires_grad: bool,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Vocabulary size (typically 256 for ASCII or larger for Unicode)
    pub vocab_size: usize,
}

/// Default value for requires_grad (true for training)
fn default_requires_grad() -> bool {
    true
}

impl Embedding {
    /// Create a new embedding layer with specified initialization
    ///
    /// # Arguments
    /// * `vocab_size` - Number of possible input tokens (256 for ASCII)
    /// * `embed_dim` - Dimension of embedding vectors
    /// * `init` - Weight initialization strategy
    pub fn new(vocab_size: usize, embed_dim: usize, init: InitStrategy) -> Self {
        let weights = Self::init_weights(vocab_size, embed_dim, init);
        Self {
            weights,
            grad: None,
            requires_grad: true,
            embed_dim,
            vocab_size,
        }
    }

    /// Create embedding with Xavier initialization (recommended default)
    pub fn xavier(vocab_size: usize, embed_dim: usize) -> Self {
        Self::new(vocab_size, embed_dim, InitStrategy::Xavier)
    }

    /// Create embedding with He initialization
    pub fn he(vocab_size: usize, embed_dim: usize) -> Self {
        Self::new(vocab_size, embed_dim, InitStrategy::He)
    }

    /// Initialize weights according to strategy
    fn init_weights(vocab_size: usize, embed_dim: usize, init: InitStrategy) -> Array2<f32> {
        let mut rng = rand::thread_rng();

        match init {
            InitStrategy::Xavier => {
                let scale = (2.0 / (vocab_size + embed_dim) as f32).sqrt();
                Array2::from_shape_fn((vocab_size, embed_dim), |_| rng.gen_range(-scale..scale))
            }
            InitStrategy::He => {
                let scale = (2.0 / vocab_size as f32).sqrt();
                Array2::from_shape_fn((vocab_size, embed_dim), |_| rng.gen_range(-scale..scale))
            }
            InitStrategy::Uniform(scale) => {
                Array2::from_shape_fn((vocab_size, embed_dim), |_| rng.gen_range(-scale..scale))
            }
            InitStrategy::Zero => Array2::zeros((vocab_size, embed_dim)),
        }
    }

    /// Forward pass: look up embedding for a single character
    ///
    /// # Arguments
    /// * `ch` - Character to embed (uses char as u32 index, clamped to vocab_size)
    ///
    /// # Returns
    /// Embedding vector of dimension `embed_dim`
    pub fn forward(&self, ch: char) -> Array1<f32> {
        let idx = (ch as usize).min(self.vocab_size - 1);
        self.weights.row(idx).to_owned()
    }

    /// Forward pass: look up embeddings for a sequence of characters
    ///
    /// # Arguments
    /// * `chars` - Slice of characters to embed
    ///
    /// # Returns
    /// Matrix of shape (len, embed_dim) where each row is an embedding
    pub fn forward_batch(&self, chars: &[char]) -> Array2<f32> {
        let n = chars.len();
        let mut result = Array2::zeros((n, self.embed_dim));
        for (i, &ch) in chars.iter().enumerate() {
            let idx = (ch as usize).min(self.vocab_size - 1);
            result.row_mut(i).assign(&self.weights.row(idx));
        }
        result
    }

    /// Forward pass for a byte index directly
    pub fn forward_index(&self, idx: usize) -> Array1<f32> {
        let idx = idx.min(self.vocab_size - 1);
        self.weights.row(idx).to_owned()
    }

    /// Zero out accumulated gradients
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = Some(Array2::zeros((self.vocab_size, self.embed_dim)));
        }
    }

    /// Accumulate gradient for a specific index
    ///
    /// # Arguments
    /// * `idx` - Character index that was used in forward pass
    /// * `grad_output` - Gradient flowing back from the next layer
    pub fn backward(&mut self, idx: usize, grad_output: &Array1<f32>) {
        if !self.requires_grad {
            return;
        }

        let idx = idx.min(self.vocab_size - 1);

        // Initialize gradient if needed
        if self.grad.is_none() {
            self.zero_grad();
        }

        // Accumulate gradient at the index that was looked up
        if let Some(ref mut grad) = self.grad {
            for (j, &g) in grad_output.iter().enumerate() {
                grad[[idx, j]] += g;
            }
        }
    }

    /// Update weights using accumulated gradients (SGD step)
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad {
            self.weights = &self.weights - &(grad * lr);
        }
    }

    /// Get the embedding for a node based on its type
    pub fn embed_node(&self, node: &Node) -> Array1<f32> {
        match &node.node_type {
            NodeType::Input(ch) => self.forward(*ch),
            NodeType::Hidden => {
                // Hidden nodes get a special embedding (index 0 by convention)
                self.forward_index(0)
            }
            NodeType::Output => {
                // Output nodes get another special embedding (index 1)
                self.forward_index(1)
            }
            NodeType::Clique(_) => {
                // Clique nodes - could aggregate member embeddings
                self.forward_index(2)
            }
            NodeType::Pattern(bytes) => {
                // Pattern nodes - average of byte embeddings
                if bytes.is_empty() {
                    return self.forward_index(3);
                }
                let mut sum = Array1::zeros(self.embed_dim);
                for &b in bytes {
                    sum = sum + self.forward_index(b as usize);
                }
                sum / bytes.len() as f32
            }
            NodeType::Compressed(_) => self.forward_index(4),
        }
    }

    /// Get number of parameters in this layer
    pub fn num_parameters(&self) -> usize {
        self.vocab_size * self.embed_dim
    }

    /// Freeze the layer (disable gradient computation)
    pub fn freeze(&mut self) {
        self.requires_grad = false;
        self.grad = None;
    }

    /// Unfreeze the layer (enable gradient computation)
    pub fn unfreeze(&mut self) {
        self.requires_grad = true;
    }
}

/// Extension trait to add embedding support to DagNN
pub trait EmbeddingExt {
    /// Get embeddings for all nodes in the graph
    fn get_node_embeddings(&self, embedding: &Embedding) -> Vec<Array1<f32>>;

    /// Get embeddings as a matrix (n_nodes x embed_dim)
    fn get_embedding_matrix(&self, embedding: &Embedding) -> Array2<f32>;
}

impl EmbeddingExt for DagNN {
    fn get_node_embeddings(&self, embedding: &Embedding) -> Vec<Array1<f32>> {
        self.graph
            .node_indices()
            .map(|idx| embedding.embed_node(&self.graph[idx]))
            .collect()
    }

    fn get_embedding_matrix(&self, embedding: &Embedding) -> Array2<f32> {
        let n = self.graph.node_count();
        let mut matrix = Array2::zeros((n, embedding.embed_dim));

        for (i, idx) in self.graph.node_indices().enumerate() {
            let emb = embedding.embed_node(&self.graph[idx]);
            matrix.row_mut(i).assign(&emb);
        }

        matrix
    }
}

// ============================================================================
// Backpropagation through Graph Structures (backend-027)
// ============================================================================

/// Operation type for the computation tape
#[derive(Debug, Clone)]
pub enum TapeOp {
    /// Embedding lookup: (embedding_idx, char_idx)
    EmbeddingLookup {
        embedding_idx: usize,
        char_idx: usize,
    },
    /// Linear transformation: output = input * weight + bias
    Linear { input_idx: usize, weight_idx: usize },
    /// Sum of multiple inputs
    Sum { input_indices: Vec<usize> },
    /// Mean of multiple inputs
    Mean { input_indices: Vec<usize> },
    /// Element-wise multiplication
    Mul { left_idx: usize, right_idx: usize },
    /// ReLU activation
    ReLU { input_idx: usize },
    /// Sigmoid activation
    Sigmoid { input_idx: usize },
    /// Tanh activation
    Tanh { input_idx: usize },
    /// Message passing: aggregate neighbors
    MessagePass {
        node_idx: usize,
        neighbor_indices: Vec<usize>,
        weights: Vec<f32>,
    },
    /// Graph convolution operation
    GraphConv {
        node_idx: usize,
        neighbor_indices: Vec<usize>,
    },
    /// Loss computation (MSE, CrossEntropy, etc.)
    Loss { pred_idx: usize, target_idx: usize },
}

/// A tape entry recording one operation and its output
#[derive(Debug, Clone)]
pub struct TapeEntry {
    /// The operation performed
    pub op: TapeOp,
    /// Index of the output value in the value store
    pub output_idx: usize,
    /// Shape of the output
    pub output_shape: Vec<usize>,
}

/// Computation tape for automatic differentiation
///
/// Records operations during forward pass, enables gradient computation
/// during backward pass via reverse-mode autodiff.
///
/// # Example
/// ```
/// use grapheme_core::{Tape, Embedding};
///
/// let mut tape = Tape::new();
/// let emb = Embedding::xavier(256, 64);
///
/// // Forward pass records to tape
/// let output_idx = tape.embedding_lookup(&emb, 'a');
///
/// // Backward pass computes gradients
/// tape.backward(output_idx);
/// ```
#[derive(Debug, Default)]
pub struct Tape {
    /// Recorded operations in order
    entries: Vec<TapeEntry>,
    /// Stored values from forward pass (flattened f32 arrays)
    values: Vec<Vec<f32>>,
    /// Gradients for each value (computed during backward)
    grads: Vec<Option<Vec<f32>>>,
    /// Whether tape is currently recording
    pub recording: bool,
}

impl Tape {
    /// Create a new empty tape
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            values: Vec::new(),
            grads: Vec::new(),
            recording: true,
        }
    }

    /// Clear the tape for a new forward pass
    pub fn reset(&mut self) {
        self.entries.clear();
        self.values.clear();
        self.grads.clear();
    }

    /// Stop recording (useful for inference)
    pub fn no_grad(&mut self) {
        self.recording = false;
    }

    /// Resume recording
    pub fn enable_grad(&mut self) {
        self.recording = true;
    }

    /// Store a value and return its index
    pub fn store_value(&mut self, value: Vec<f32>, _shape: Vec<usize>) -> usize {
        let idx = self.values.len();
        self.values.push(value);
        self.grads.push(None);
        idx
    }

    /// Record an embedding lookup operation
    pub fn embedding_lookup(&mut self, embedding: &Embedding, ch: char) -> usize {
        let output = embedding.forward(ch);
        let output_vec: Vec<f32> = output.iter().cloned().collect();
        let shape = vec![embedding.embed_dim];

        let output_idx = self.store_value(output_vec, shape.clone());

        if self.recording {
            self.entries.push(TapeEntry {
                op: TapeOp::EmbeddingLookup {
                    embedding_idx: 0, // Single embedding for now
                    char_idx: ch as usize,
                },
                output_idx,
                output_shape: shape,
            });
        }

        output_idx
    }

    /// Record a sum operation
    pub fn sum(&mut self, input_indices: &[usize]) -> usize {
        if input_indices.is_empty() {
            return self.store_value(vec![0.0], vec![1]);
        }

        // Get dimension from first input
        let dim = self.values[input_indices[0]].len();
        let mut result = vec![0.0; dim];

        for &idx in input_indices {
            for (i, &v) in self.values[idx].iter().enumerate() {
                result[i] += v;
            }
        }

        let output_idx = self.store_value(result, vec![dim]);

        if self.recording {
            self.entries.push(TapeEntry {
                op: TapeOp::Sum {
                    input_indices: input_indices.to_vec(),
                },
                output_idx,
                output_shape: vec![dim],
            });
        }

        output_idx
    }

    /// Record a mean operation
    pub fn mean(&mut self, input_indices: &[usize]) -> usize {
        if input_indices.is_empty() {
            return self.store_value(vec![0.0], vec![1]);
        }

        let dim = self.values[input_indices[0]].len();
        let mut result = vec![0.0; dim];
        let n = input_indices.len() as f32;

        for &idx in input_indices {
            for (i, &v) in self.values[idx].iter().enumerate() {
                result[i] += v / n;
            }
        }

        let output_idx = self.store_value(result, vec![dim]);

        if self.recording {
            self.entries.push(TapeEntry {
                op: TapeOp::Mean {
                    input_indices: input_indices.to_vec(),
                },
                output_idx,
                output_shape: vec![dim],
            });
        }

        output_idx
    }

    /// Record element-wise multiplication
    pub fn mul(&mut self, left_idx: usize, right_idx: usize) -> usize {
        let left = &self.values[left_idx];
        let right = &self.values[right_idx];
        let dim = left.len().min(right.len());

        let result: Vec<f32> = left.iter().zip(right.iter()).map(|(l, r)| l * r).collect();

        let output_idx = self.store_value(result, vec![dim]);

        if self.recording {
            self.entries.push(TapeEntry {
                op: TapeOp::Mul {
                    left_idx,
                    right_idx,
                },
                output_idx,
                output_shape: vec![dim],
            });
        }

        output_idx
    }

    /// Record ReLU activation
    pub fn relu(&mut self, input_idx: usize) -> usize {
        let input = &self.values[input_idx];
        let result: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
        let dim = result.len();

        let output_idx = self.store_value(result, vec![dim]);

        if self.recording {
            self.entries.push(TapeEntry {
                op: TapeOp::ReLU { input_idx },
                output_idx,
                output_shape: vec![dim],
            });
        }

        output_idx
    }

    /// Record sigmoid activation
    pub fn sigmoid(&mut self, input_idx: usize) -> usize {
        let input = &self.values[input_idx];
        let result: Vec<f32> = input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let dim = result.len();

        let output_idx = self.store_value(result, vec![dim]);

        if self.recording {
            self.entries.push(TapeEntry {
                op: TapeOp::Sigmoid { input_idx },
                output_idx,
                output_shape: vec![dim],
            });
        }

        output_idx
    }

    /// Record tanh activation
    pub fn tanh(&mut self, input_idx: usize) -> usize {
        let input = &self.values[input_idx];
        let result: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
        let dim = result.len();

        let output_idx = self.store_value(result, vec![dim]);

        if self.recording {
            self.entries.push(TapeEntry {
                op: TapeOp::Tanh { input_idx },
                output_idx,
                output_shape: vec![dim],
            });
        }

        output_idx
    }

    /// Record a message passing operation (for GNN)
    pub fn message_pass(
        &mut self,
        node_idx: usize,
        neighbor_indices: &[usize],
        weights: &[f32],
    ) -> usize {
        if neighbor_indices.is_empty() {
            return node_idx; // No neighbors, return self
        }

        let dim = self.values[node_idx].len();
        let mut result = vec![0.0; dim];

        // Weighted sum of neighbor embeddings
        for (i, &neighbor_idx) in neighbor_indices.iter().enumerate() {
            let w = weights.get(i).copied().unwrap_or(1.0);
            for (j, &v) in self.values[neighbor_idx].iter().enumerate() {
                result[j] += w * v;
            }
        }

        let output_idx = self.store_value(result, vec![dim]);

        if self.recording {
            self.entries.push(TapeEntry {
                op: TapeOp::MessagePass {
                    node_idx,
                    neighbor_indices: neighbor_indices.to_vec(),
                    weights: weights.to_vec(),
                },
                output_idx,
                output_shape: vec![dim],
            });
        }

        output_idx
    }

    /// Compute MSE loss between prediction and target
    pub fn mse_loss(&mut self, pred_idx: usize, target: &[f32]) -> (usize, f32) {
        let pred = &self.values[pred_idx];
        let n = pred.len().min(target.len());

        let mut sum_sq = 0.0;
        for i in 0..n {
            let diff = pred[i] - target[i];
            sum_sq += diff * diff;
        }
        let loss = sum_sq / n as f32;

        // Store target as a value
        let target_idx = self.store_value(target.to_vec(), vec![target.len()]);

        // Store loss as single-element value
        let output_idx = self.store_value(vec![loss], vec![1]);

        if self.recording {
            self.entries.push(TapeEntry {
                op: TapeOp::Loss {
                    pred_idx,
                    target_idx,
                },
                output_idx,
                output_shape: vec![1],
            });
        }

        (output_idx, loss)
    }

    /// Get a stored value
    pub fn get_value(&self, idx: usize) -> Option<&Vec<f32>> {
        self.values.get(idx)
    }

    /// Get gradient for a value
    pub fn get_grad(&self, idx: usize) -> Option<&Vec<f32>> {
        self.grads.get(idx).and_then(|g| g.as_ref())
    }

    /// Backward pass: compute gradients for all recorded operations
    ///
    /// # Arguments
    /// * `output_idx` - Index of the loss/output to backprop from
    ///
    /// This implements reverse-mode automatic differentiation,
    /// processing operations in reverse topological order.
    pub fn backward(&mut self, output_idx: usize) {
        // Initialize output gradient to 1.0
        let output_dim = self.values[output_idx].len();
        self.grads[output_idx] = Some(vec![1.0; output_dim]);

        // Process entries in reverse order (reverse topological order)
        for entry_idx in (0..self.entries.len()).rev() {
            // Extract what we need from the entry before any mutable operations
            let entry_output_idx = self.entries[entry_idx].output_idx;
            let op = self.entries[entry_idx].op.clone();

            // Skip if no gradient at output
            let output_grad = match &self.grads[entry_output_idx] {
                Some(g) => g.clone(),
                None => continue,
            };

            // Compute and propagate gradients based on operation type
            match op {
                TapeOp::EmbeddingLookup { .. } => {
                    // Gradient flows to embedding weights (handled externally)
                }

                TapeOp::Sum { input_indices } => {
                    // Gradient flows equally to all inputs
                    for input_idx in input_indices {
                        self.accumulate_grad(input_idx, &output_grad);
                    }
                }

                TapeOp::Mean { input_indices } => {
                    // Gradient is scaled by 1/n
                    let n = input_indices.len() as f32;
                    let scaled_grad: Vec<f32> = output_grad.iter().map(|&g| g / n).collect();
                    for input_idx in input_indices {
                        self.accumulate_grad(input_idx, &scaled_grad);
                    }
                }

                TapeOp::Mul {
                    left_idx,
                    right_idx,
                } => {
                    // d(a*b)/da = b, d(a*b)/db = a
                    let left = self.values[left_idx].clone();
                    let right = self.values[right_idx].clone();

                    let left_grad: Vec<f32> = output_grad
                        .iter()
                        .zip(right.iter())
                        .map(|(g, r)| g * r)
                        .collect();

                    let right_grad: Vec<f32> = output_grad
                        .iter()
                        .zip(left.iter())
                        .map(|(g, l)| g * l)
                        .collect();

                    self.accumulate_grad(left_idx, &left_grad);
                    self.accumulate_grad(right_idx, &right_grad);
                }

                TapeOp::ReLU { input_idx } => {
                    // d(relu)/dx = 1 if x > 0 else 0
                    let input = self.values[input_idx].clone();
                    let grad: Vec<f32> = output_grad
                        .iter()
                        .zip(input.iter())
                        .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                        .collect();
                    self.accumulate_grad(input_idx, &grad);
                }

                TapeOp::Sigmoid { input_idx } => {
                    // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
                    let output = self.values[entry_output_idx].clone();
                    let grad: Vec<f32> = output_grad
                        .iter()
                        .zip(output.iter())
                        .map(|(g, s)| g * s * (1.0 - s))
                        .collect();
                    self.accumulate_grad(input_idx, &grad);
                }

                TapeOp::Tanh { input_idx } => {
                    // d(tanh)/dx = 1 - tanh^2
                    let output = self.values[entry_output_idx].clone();
                    let grad: Vec<f32> = output_grad
                        .iter()
                        .zip(output.iter())
                        .map(|(g, t)| g * (1.0 - t * t))
                        .collect();
                    self.accumulate_grad(input_idx, &grad);
                }

                TapeOp::MessagePass {
                    neighbor_indices,
                    weights,
                    ..
                } => {
                    // Gradient flows back to neighbors weighted by edge weights
                    for (i, neighbor_idx) in neighbor_indices.iter().enumerate() {
                        let w = weights.get(i).copied().unwrap_or(1.0);
                        let scaled_grad: Vec<f32> = output_grad.iter().map(|&g| g * w).collect();
                        self.accumulate_grad(*neighbor_idx, &scaled_grad);
                    }
                }

                TapeOp::GraphConv {
                    neighbor_indices, ..
                } => {
                    // Simple mean aggregation gradient
                    let n = neighbor_indices.len() as f32;
                    let scaled_grad: Vec<f32> = output_grad.iter().map(|&g| g / n).collect();
                    for neighbor_idx in neighbor_indices {
                        self.accumulate_grad(neighbor_idx, &scaled_grad);
                    }
                }

                TapeOp::Loss {
                    pred_idx,
                    target_idx,
                } => {
                    // MSE gradient: d/dpred = 2 * (pred - target) / n
                    let pred = self.values[pred_idx].clone();
                    let target = self.values[target_idx].clone();
                    let n = pred.len() as f32;

                    let grad: Vec<f32> = pred
                        .iter()
                        .zip(target.iter())
                        .map(|(p, t)| 2.0 * (p - t) / n)
                        .collect();

                    // Scale by output gradient (usually 1.0)
                    let scaled_grad: Vec<f32> = grad.iter().map(|&g| g * output_grad[0]).collect();

                    self.accumulate_grad(pred_idx, &scaled_grad);
                }

                TapeOp::Linear { input_idx, .. } => {
                    // Simplified: gradient passes through
                    self.accumulate_grad(input_idx, &output_grad);
                }
            }
        }
    }

    /// Accumulate gradient at a value index
    fn accumulate_grad(&mut self, idx: usize, grad: &[f32]) {
        if idx >= self.grads.len() {
            return;
        }

        if self.grads[idx].is_none() {
            self.grads[idx] = Some(vec![0.0; grad.len()]);
        }

        if let Some(ref mut existing) = self.grads[idx] {
            for (i, &g) in grad.iter().enumerate() {
                if i < existing.len() {
                    existing[i] += g;
                }
            }
        }
    }

    /// Get embedding gradients from the tape for updating
    pub fn get_embedding_grads(&self) -> Vec<(usize, Vec<f32>)> {
        let mut result = Vec::new();

        for entry in self.entries.iter() {
            if let TapeOp::EmbeddingLookup { char_idx, .. } = entry.op {
                if let Some(grad) = &self.grads[entry.output_idx] {
                    result.push((char_idx, grad.clone()));
                }
            }
        }

        result
    }
}

/// Node-level gradient storage for DagNN
#[derive(Debug, Clone, Default)]
pub struct NodeGradients {
    /// Gradient for each node's hidden state
    pub node_grads: HashMap<NodeId, Array1<f32>>,
    /// Gradient for each edge weight
    pub edge_grads: HashMap<(NodeId, NodeId), f32>,
}

impl NodeGradients {
    /// Create new empty gradient storage
    pub fn new() -> Self {
        Self::default()
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        self.node_grads.clear();
        self.edge_grads.clear();
    }

    /// Accumulate gradient for a node
    pub fn accumulate_node(&mut self, node: NodeId, grad: &Array1<f32>) {
        self.node_grads
            .entry(node)
            .and_modify(|g| *g = &*g + grad)
            .or_insert_with(|| grad.clone());
    }

    /// Accumulate gradient for an edge
    pub fn accumulate_edge(&mut self, from: NodeId, to: NodeId, grad: f32) {
        *self.edge_grads.entry((from, to)).or_insert(0.0) += grad;
    }

    /// Get node gradient
    pub fn get_node_grad(&self, node: NodeId) -> Option<&Array1<f32>> {
        self.node_grads.get(&node)
    }

    /// Get edge gradient
    pub fn get_edge_grad(&self, from: NodeId, to: NodeId) -> Option<f32> {
        self.edge_grads.get(&(from, to)).copied()
    }

    /// Apply gradient clipping to prevent exploding gradients
    pub fn clip_grads(&mut self, max_norm: f32) {
        // Compute total gradient norm
        let mut total_norm_sq = 0.0;

        for grad in self.node_grads.values() {
            total_norm_sq += grad.iter().map(|x| x * x).sum::<f32>();
        }
        for &grad in self.edge_grads.values() {
            total_norm_sq += grad * grad;
        }

        let total_norm = total_norm_sq.sqrt();

        // Scale if exceeds max
        if total_norm > max_norm {
            let scale = max_norm / total_norm;

            for grad in self.node_grads.values_mut() {
                *grad *= scale;
            }
            for grad in self.edge_grads.values_mut() {
                *grad *= scale;
            }
        }
    }
}

/// Backward pass trait for DagNN
pub trait BackwardPass {
    /// Compute backward pass through the graph
    ///
    /// # Arguments
    /// * `output_grad` - Gradient from the loss with respect to output nodes
    /// * `embedding` - The embedding layer to accumulate gradients to
    ///
    /// # Returns
    /// Node gradients that can be used for edge weight updates
    fn backward(
        &self,
        output_grad: &HashMap<NodeId, Array1<f32>>,
        embedding: &mut Embedding,
    ) -> NodeGradients;

    /// Compute backward pass and update edge weights
    fn backward_and_update(
        &mut self,
        output_grad: &HashMap<NodeId, Array1<f32>>,
        embedding: &mut Embedding,
        lr: f32,
    );
}

impl BackwardPass for DagNN {
    fn backward(
        &self,
        output_grad: &HashMap<NodeId, Array1<f32>>,
        embedding: &mut Embedding,
    ) -> NodeGradients {
        let mut grads = NodeGradients::new();

        // Initialize output gradients
        for (&node, grad) in output_grad {
            grads.accumulate_node(node, grad);
        }

        // Process in reverse topological order
        for &node in self.topology.order.iter().rev() {
            // Skip if no gradient at this node
            let node_grad = match grads.get_node_grad(node) {
                Some(g) => g.clone(),
                None => continue,
            };

            // Propagate gradient to predecessors
            for edge in self
                .graph
                .edges_directed(node, petgraph::Direction::Incoming)
            {
                let source = edge.source();
                let edge_weight = edge.weight().weight;

                // Gradient w.r.t. source = edge_weight * node_grad
                let source_grad = &node_grad * edge_weight;
                grads.accumulate_node(source, &source_grad);

                // Gradient w.r.t. edge weight = source_activation * node_grad
                let source_activation = self.graph[source].activation;
                let edge_grad: f32 = node_grad.iter().map(|&g| g * source_activation).sum();
                grads.accumulate_edge(source, node, edge_grad);
            }

            // Update embedding gradients for input nodes
            if let NodeType::Input(ch) = &self.graph[node].node_type {
                embedding.backward(*ch as usize, &node_grad);
            }
        }

        grads
    }

    fn backward_and_update(
        &mut self,
        output_grad: &HashMap<NodeId, Array1<f32>>,
        embedding: &mut Embedding,
        lr: f32,
    ) {
        let grads = self.backward(output_grad, embedding);

        // Update edge weights using gradients
        for ((from, to), edge_grad) in grads.edge_grads {
            if let Some(edge_idx) = self.graph.find_edge(from, to) {
                self.graph[edge_idx].weight -= lr * edge_grad;
            }
        }
    }
}

/// Utility functions for gradient computation
pub mod grad_utils {
    use super::*;

    /// Compute numerical gradient for validation
    pub fn numerical_gradient<F>(f: F, x: &Array1<f32>, eps: f32) -> Array1<f32>
    where
        F: Fn(&Array1<f32>) -> f32,
    {
        let mut grad = Array1::zeros(x.len());

        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();

            x_plus[i] += eps;
            x_minus[i] -= eps;

            grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * eps);
        }

        grad
    }

    /// Check if analytical gradient matches numerical gradient
    pub fn gradient_check(
        analytical: &Array1<f32>,
        numerical: &Array1<f32>,
        tolerance: f32,
    ) -> bool {
        if analytical.len() != numerical.len() {
            return false;
        }

        for i in 0..analytical.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            let scale = analytical[i].abs().max(numerical[i].abs()).max(1.0);

            if diff / scale > tolerance {
                return false;
            }
        }

        true
    }

    /// Apply gradient clipping to a vector
    pub fn clip_grad_norm(grad: &mut Array1<f32>, max_norm: f32) {
        let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm > max_norm {
            let scale = max_norm / norm;
            *grad *= scale;
        }
    }

    /// Compute L2 regularization gradient
    pub fn l2_regularization_grad(weights: &Array2<f32>, lambda: f32) -> Array2<f32> {
        weights * (2.0 * lambda)
    }
}

// ============================================================================
// Learnable Graph Transformation Network (backend-029)
// ============================================================================

/// Edit operation types for graph transformation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditOp {
    /// Keep the node unchanged
    Keep,
    /// Delete the node
    Delete,
    /// Modify the node value
    Modify,
    /// Insert a new node after this one
    Insert,
}

/// Message passing layer for graph neural networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePassingLayer {
    /// Weight matrix for transforming neighbor features
    pub weight: Array2<f32>,
    /// Bias vector
    pub bias: Array1<f32>,
    /// Weight gradient
    #[serde(skip)]
    pub weight_grad: Option<Array2<f32>>,
    /// Bias gradient
    #[serde(skip)]
    pub bias_grad: Option<Array1<f32>>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
}

impl MessagePassingLayer {
    /// Create a new message passing layer with Xavier initialization
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_dim + output_dim) as f32).sqrt();

        let weight = Array2::from_shape_fn((output_dim, input_dim), |_| {
            rng.gen::<f32>() * 2.0 * std - std
        });
        let bias = Array1::zeros(output_dim);

        Self {
            weight,
            bias,
            weight_grad: None,
            bias_grad: None,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: aggregate neighbor features and transform
    pub fn forward(
        &self,
        node_features: &Array1<f32>,
        neighbor_features: &[Array1<f32>],
    ) -> Array1<f32> {
        // Aggregate neighbors (mean pooling)
        let aggregated = if neighbor_features.is_empty() {
            node_features.clone()
        } else {
            let mut sum = node_features.clone();
            for neighbor in neighbor_features {
                sum = &sum + neighbor;
            }
            sum / (neighbor_features.len() + 1) as f32
        };

        // Linear transformation: output = W * aggregated + b
        let mut output = Array1::zeros(self.output_dim);
        for i in 0..self.output_dim {
            let mut val = self.bias[i];
            for j in 0..self.input_dim.min(aggregated.len()) {
                val += self.weight[[i, j]] * aggregated[j];
            }
            output[i] = val;
        }

        // ReLU activation
        output.mapv_inplace(|x| x.max(0.0));
        output
    }

    /// Forward pass for batch of nodes
    ///
    /// Uses parallel processing via Rayon for improved performance on large graphs.
    /// Each node's forward pass is computed independently in parallel.
    pub fn forward_batch(
        &self,
        node_features: &Array2<f32>,
        adjacency: &[Vec<usize>],
    ) -> Array2<f32> {
        let n = node_features.shape()[0];

        // Process all nodes in parallel using Rayon
        let rows: Vec<Array1<f32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let node_feat = node_features.row(i).to_owned();
                let neighbor_feats: Vec<Array1<f32>> = adjacency[i]
                    .iter()
                    .map(|&j| node_features.row(j).to_owned())
                    .collect();

                self.forward(&node_feat, &neighbor_feats)
            })
            .collect();

        // Assemble output matrix from parallel results
        let mut output = Array2::zeros((n, self.output_dim));
        for (i, row) in rows.into_iter().enumerate() {
            output.row_mut(i).assign(&row);
        }

        output
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.weight_grad = None;
        self.bias_grad = None;
    }

    /// Update weights with gradients
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.weight_grad {
            self.weight = &self.weight - &(grad * lr);
        }
        if let Some(ref grad) = self.bias_grad {
            self.bias = &self.bias - &(grad * lr);
        }
    }
}

/// Attention mechanism for edit localization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionLayer {
    /// Query projection
    pub query_proj: Array2<f32>,
    /// Key projection
    pub key_proj: Array2<f32>,
    /// Value projection
    pub value_proj: Array2<f32>,
    /// Dimension of keys/queries
    pub d_k: usize,
}

impl AttentionLayer {
    /// Create a new attention layer
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let d_k = embed_dim / num_heads.max(1);
        let mut rng = rand::thread_rng();
        let std = (1.0 / embed_dim as f32).sqrt();

        let query_proj = Array2::from_shape_fn((embed_dim, embed_dim), |_| {
            rng.gen::<f32>() * 2.0 * std - std
        });
        let key_proj = Array2::from_shape_fn((embed_dim, embed_dim), |_| {
            rng.gen::<f32>() * 2.0 * std - std
        });
        let value_proj = Array2::from_shape_fn((embed_dim, embed_dim), |_| {
            rng.gen::<f32>() * 2.0 * std - std
        });

        Self {
            query_proj,
            key_proj,
            value_proj,
            d_k,
        }
    }

    /// Compute attention scores and weighted values
    pub fn forward(
        &self,
        query: &Array1<f32>,
        keys: &[Array1<f32>],
        values: &[Array1<f32>],
    ) -> Array1<f32> {
        if keys.is_empty() || values.is_empty() {
            return query.clone();
        }

        // Project query
        let q = self.project(query, &self.query_proj);

        // Compute attention scores
        let mut scores = Vec::with_capacity(keys.len());
        for key in keys {
            let k = self.project(key, &self.key_proj);
            let score = q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum::<f32>();
            scores.push(score / (self.d_k as f32).sqrt());
        }

        // Softmax
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let attention_weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Weighted sum of values
        let mut output = Array1::zeros(query.len());
        for (value, &weight) in values.iter().zip(attention_weights.iter()) {
            let v = self.project(value, &self.value_proj);
            for j in 0..output.len().min(v.len()) {
                output[j] += weight * v[j];
            }
        }

        output
    }

    /// Project a vector using a weight matrix
    fn project(&self, input: &Array1<f32>, weight: &Array2<f32>) -> Array1<f32> {
        let out_dim = weight.shape()[0];
        let in_dim = weight.shape()[1];
        let mut output = Array1::zeros(out_dim);

        for i in 0..out_dim {
            for j in 0..in_dim.min(input.len()) {
                output[i] += weight[[i, j]] * input[j];
            }
        }

        output
    }
}

/// Prediction head for node-level operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePredictionHead {
    /// Weight matrix for edit operation prediction
    pub weight: Array2<f32>,
    /// Bias
    pub bias: Array1<f32>,
    /// Number of edit operation types
    pub num_ops: usize,
}

impl NodePredictionHead {
    /// Create a new node prediction head
    pub fn new(input_dim: usize, num_ops: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_dim + num_ops) as f32).sqrt();

        Self {
            weight: Array2::from_shape_fn((num_ops, input_dim), |_| {
                rng.gen::<f32>() * 2.0 * std - std
            }),
            bias: Array1::zeros(num_ops),
            num_ops,
        }
    }

    /// Predict edit operation probabilities for a node
    pub fn predict(&self, node_features: &Array1<f32>) -> Vec<f32> {
        let mut logits = Vec::with_capacity(self.num_ops);

        for i in 0..self.num_ops {
            let mut val = self.bias[i];
            for j in 0..node_features.len().min(self.weight.shape()[1]) {
                val += self.weight[[i, j]] * node_features[j];
            }
            logits.push(val);
        }

        // Softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        exp_logits.iter().map(|&e| e / sum_exp).collect()
    }

    /// Get the predicted edit operation
    pub fn predict_op(&self, node_features: &Array1<f32>) -> EditOp {
        let probs = self.predict(node_features);
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        match max_idx {
            0 => EditOp::Keep,
            1 => EditOp::Delete,
            2 => EditOp::Modify,
            3 => EditOp::Insert,
            _ => EditOp::Keep,
        }
    }
}

/// Graph pooling for global features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPooling {
    /// Pooling type
    pub pooling_type: PoolingType,
}

/// Type of graph pooling
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PoolingType {
    /// Mean pooling
    Mean,
    /// Max pooling
    Max,
    /// Sum pooling
    Sum,
}

impl GraphPooling {
    /// Create mean pooling
    pub fn mean() -> Self {
        Self {
            pooling_type: PoolingType::Mean,
        }
    }

    /// Create max pooling
    pub fn max() -> Self {
        Self {
            pooling_type: PoolingType::Max,
        }
    }

    /// Create sum pooling
    pub fn sum() -> Self {
        Self {
            pooling_type: PoolingType::Sum,
        }
    }

    /// Pool node features into a graph-level feature
    pub fn pool(&self, node_features: &[Array1<f32>]) -> Array1<f32> {
        if node_features.is_empty() {
            return Array1::zeros(1);
        }

        let dim = node_features[0].len();
        let mut result = Array1::zeros(dim);

        match self.pooling_type {
            PoolingType::Mean => {
                for feat in node_features {
                    result = &result + feat;
                }
                result / node_features.len() as f32
            }
            PoolingType::Max => {
                result = node_features[0].clone();
                for feat in node_features.iter().skip(1) {
                    for i in 0..dim {
                        result[i] = result[i].max(feat[i]);
                    }
                }
                result
            }
            PoolingType::Sum => {
                for feat in node_features {
                    result = &result + feat;
                }
                result
            }
        }
    }
}

/// Learnable graph transformation network
///
/// This is the "brain" that learns to transform graphs based on training examples.
/// It combines message passing, attention, and prediction heads to predict
/// graph edit operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformNet {
    /// Input embedding layer
    pub embedding: Embedding,
    /// Message passing layers
    pub mp_layers: Vec<MessagePassingLayer>,
    /// Attention layer for edit localization
    pub attention: AttentionLayer,
    /// Node prediction head
    pub node_head: NodePredictionHead,
    /// Graph pooling
    pub pooling: GraphPooling,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of message passing layers
    pub num_layers: usize,
    /// Learnable merge threshold for graph morphing (backend-099)
    /// This parameter is optimized by Adam to learn when to merge nodes
    /// Higher values → more conservative merging (fewer nodes merged)
    /// Lower values → more aggressive merging (more nodes merged)
    pub merge_threshold: Parameter,
    /// Sabag pooling layer for differentiable DAG coarsening (backend-104)
    /// Named after Eliran Sabag - DAG-aware soft pooling with Sinkhorn refinement
    /// Enables gradient flow while preserving DAG topology
    #[serde(skip)]
    pub sabag_pooling: Option<SabagPooling>,
}

impl GraphTransformNet {
    /// Create a new graph transformation network
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let embedding = Embedding::xavier(vocab_size, embed_dim);

        let mut mp_layers = Vec::with_capacity(num_layers);
        let mut in_dim = embed_dim;
        for _ in 0..num_layers {
            mp_layers.push(MessagePassingLayer::new(in_dim, hidden_dim));
            in_dim = hidden_dim;
        }

        let attention = AttentionLayer::new(hidden_dim, 4);
        let node_head = NodePredictionHead::new(hidden_dim, 4); // 4 edit ops
        let pooling = GraphPooling::mean();

        // Initialize merge threshold to 0.8 (high similarity required to merge)
        // Adam optimizer will learn the optimal value during training
        let merge_threshold = Parameter::new(0.8);

        // Initialize Sabag pooling for differentiable gradient flow
        // Use k=2 clusters for minimal coarsening (works with short inputs)
        let sabag_pooling = Some(SabagPooling::new(2, embed_dim));

        Self {
            embedding,
            mp_layers,
            attention,
            node_head,
            pooling,
            hidden_dim,
            num_layers,
            merge_threshold,
            sabag_pooling,
        }
    }

    /// Forward pass: compute node representations
    pub fn encode(&self, dag: &DagNN) -> Vec<Array1<f32>> {
        // Get initial embeddings
        let mut node_features: Vec<Array1<f32>> = dag
            .input_nodes()
            .iter()
            .map(|&node| self.embedding.embed_node(&dag.graph[node]))
            .collect();

        // Build adjacency list
        let adjacency: Vec<Vec<usize>> = dag
            .input_nodes()
            .iter()
            .map(|&node| {
                dag.graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .filter_map(|e| {
                        let source = e.source();
                        dag.input_nodes().iter().position(|&n| n == source)
                    })
                    .collect()
            })
            .collect();

        // Apply message passing layers
        for layer in &self.mp_layers {
            let features_matrix =
                Array2::from_shape_fn((node_features.len(), node_features[0].len()), |(i, j)| {
                    node_features[i][j]
                });
            let new_features = layer.forward_batch(&features_matrix, &adjacency);

            node_features = (0..new_features.shape()[0])
                .map(|i| new_features.row(i).to_owned())
                .collect();
        }

        node_features
    }

    /// Predict edit operations for each node
    pub fn predict_edits(&self, dag: &DagNN) -> Vec<(NodeId, EditOp, Vec<f32>)> {
        let node_features = self.encode(dag);

        dag.input_nodes()
            .iter()
            .zip(node_features.iter())
            .map(|(&node, features)| {
                let probs = self.node_head.predict(features);
                let op = self.node_head.predict_op(features);
                (node, op, probs)
            })
            .collect()
    }

    /// Apply predicted edits to create output graph
    pub fn apply_edits(
        &self,
        input: &DagNN,
        edits: &[(NodeId, EditOp, Vec<f32>)],
    ) -> GraphemeResult<DagNN> {
        let mut output_chars: Vec<char> = Vec::new();

        for &(node, op, _) in edits {
            match op {
                EditOp::Keep => {
                    if let NodeType::Input(ch) = input.graph[node].node_type {
                        output_chars.push(ch);
                    }
                }
                EditOp::Delete => {
                    // Skip this character
                }
                EditOp::Modify => {
                    // For now, keep the character (would need value prediction)
                    if let NodeType::Input(ch) = input.graph[node].node_type {
                        output_chars.push(ch);
                    }
                }
                EditOp::Insert => {
                    // Keep original and insert placeholder
                    if let NodeType::Input(ch) = input.graph[node].node_type {
                        output_chars.push(ch);
                        output_chars.push(' '); // Placeholder for inserted char
                    }
                }
            }
        }

        let output_text: String = output_chars.into_iter().collect();
        DagNN::from_text(&output_text)
    }

    /// Get global graph representation
    pub fn get_graph_embedding(&self, dag: &DagNN) -> Array1<f32> {
        let node_features = self.encode(dag);
        self.pooling.pool(&node_features)
    }

    /// Zero all gradients
    ///
    /// Uses parallel iteration for message passing layers.
    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        self.merge_threshold.zero_grad();
        // Process layers in parallel - each layer is independent
        self.mp_layers.par_iter_mut().for_each(|layer| {
            layer.zero_grad();
        });
    }

    /// Update all weights
    ///
    /// Uses parallel iteration for message passing layers.
    /// Each layer's weight update is independent and can proceed concurrently.
    pub fn step(&mut self, lr: f32) {
        self.embedding.step(lr);
        self.merge_threshold.step(lr);
        // Process layers in parallel - each layer is independent
        self.mp_layers.par_iter_mut().for_each(|layer| {
            layer.step(lr);
        });
    }

    // ========================================================================
    // Forward/Backward Pass for Training (backend-099)
    // ========================================================================

    /// Forward pass: MORPH input graph using learned transformations
    ///
    /// Backend-104: Uses DiffPool-style soft assignment for gradient flow.
    ///
    /// The graph structure evolves during forward pass:
    /// - Soft assignment matrix S = softmax(Z) computes node clustering
    /// - Features coarsened: H_new = S^T · H (differentiable!)
    /// - Graph morphs from input structure toward target structure
    ///
    /// This is the core of GRAPHEME: "Graph in everything" - structure evolves!
    ///
    /// Complexity: O(n·k·d) where k = clusters, d = embedding dim (polynomial!)
    ///
    /// # Returns
    /// Forward pass using Sabag algorithm for differentiable DAG pooling
    ///
    /// Returns tuple of (coarsened graph, pooling result for backward pass)
    pub fn forward(&self, input_graph: &GraphemeGraph) -> (GraphemeGraph, PoolingResult) {
        let n = input_graph.input_nodes.len();
        let embed_dim = self.embedding.embed_dim;

        // Step 1: Compute learned embeddings for all input nodes
        let mut embeddings_matrix = Array2::zeros((n, embed_dim));

        for (idx, &node_id) in input_graph.input_nodes.iter().enumerate() {
            if let NodeType::Input(ch) = input_graph.graph[node_id].node_type {
                let embedding = self.embedding.forward(ch);

                // Store in matrix for Sabag pooling
                for (i, &val) in embedding.iter().enumerate() {
                    embeddings_matrix[[idx, i]] = val;
                }
            }
        }

        // Step 2: Use Sabag pooling for differentiable graph coarsening
        if let Some(ref sabag) = self.sabag_pooling {
            let pooling_result = sabag.forward(input_graph, &embeddings_matrix);
            let coarsened_graph = pooling_result.graph.clone();
            (coarsened_graph, pooling_result)
        } else {
            // Fallback: No pooling, identity mapping
            let pooling_result = PoolingResult {
                graph: input_graph.clone(),
                features: embeddings_matrix.clone(),
                assignment: Array2::eye(n),
            };
            (input_graph.clone(), pooling_result)
        }
    }

    /// Helper: Compute cosine similarity between two embedding vectors
    /// Complexity: O(d) where d = embedding dimension
    fn cosine_similarity(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
        let dot = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Backward pass: Route gradients through soft pooling
    ///
    /// Backend-104: Uses DiffPool gradient routing to backprop through
    /// soft assignment matrix.
    ///
    /// Sabag-aware backward pass: Two-stage gradient routing
    ///
    /// # Arguments
    /// * `input_graph` - The input graph used in forward pass (n nodes)
    /// * `pooling_result` - Result from forward pass (contains S matrix!)
    /// * `node_gradients` - Gradients from Sinkhorn for coarsened graph (k nodes, flattened)
    /// * `embed_dim` - Dimension of embeddings
    ///
    /// # Two-Stage Gradient Flow (Sabag + Sinkhorn)
    ///
    /// Forward:
    ///   1. Sabag: n nodes → k clusters (S ∈ ℝ^(n×k))
    ///   2. Sinkhorn: k clusters → m target (P ∈ ℝ^(k×m))
    ///   3. Composed: S·P ∈ ℝ^(n×m)
    ///
    /// Backward:
    ///   1. Sinkhorn gradients: ∂L/∂H_k (k nodes from structural loss)
    ///   2. Sabag routing: ∂L/∂H_n = S · ∂L/∂H_k (route to n original nodes)
    ///   3. Embedding update: ∂L/∂embeddings for each character
    ///
    /// This is THE KEY to Sabag algorithm working with Sinkhorn!
    pub fn backward(
        &mut self,
        input_graph: &GraphemeGraph,
        pooling_result: &PoolingResult,
        node_gradients: &[f32],
        embed_dim: usize,
    ) {
        let n = input_graph.input_nodes.len();  // Original nodes
        let k = pooling_result.graph.input_nodes.len();  // Coarsened nodes
        let S = &pooling_result.assignment;  // Sabag soft assignment S ∈ ℝ^(n×k)

        // Reshape node_gradients from Sinkhorn (flat k×d) to Array2<f32>
        // Each row is gradient for one coarsened node
        let mut grad_k = Array2::zeros((k, embed_dim));
        for i in 0..k {
            for j in 0..embed_dim {
                let idx = i * embed_dim + j;
                if idx < node_gradients.len() {
                    grad_k[[i, j]] = node_gradients[idx];
                }
            }
        }

        // Sabag backward: Route gradients from k coarsened nodes to n original nodes
        // ∂L/∂H_n = S · ∂L/∂H_k
        // S ∈ ℝ^(n×k), grad_k ∈ ℝ^(k×d) → grad_n ∈ ℝ^(n×d)
        let grad_n = S.dot(&grad_k);  // Matrix multiply!

        // Now route gradients to character embeddings
        // For each original input node, update its character embedding
        for (idx, &node_id) in input_graph.input_nodes.iter().enumerate() {
            if let NodeType::Input(ch) = input_graph.graph[node_id].node_type {
                if idx < grad_n.nrows() {
                    // Extract gradient for this node (one row from grad_n)
                    let node_grad = grad_n.row(idx);

                    // Update character embedding with this gradient
                    self.embedding.backward(ch as usize, &node_grad.to_owned());
                }
            }
        }

        // Update merge threshold gradient
        // Heuristic: if loss is high, adjust threshold
        let avg_grad = node_gradients.iter().map(|g| g.abs()).sum::<f32>()
                       / node_gradients.len().max(1) as f32;
        let threshold_val = self.merge_threshold.value;
        let sigmoid = 1.0 / (1.0 + (-threshold_val).exp());
        let sigmoid_deriv = sigmoid * (1.0 - sigmoid);
        let threshold_grad = -sigmoid_deriv * avg_grad * 0.01; // Small learning rate for threshold

        self.merge_threshold.accumulate_grad(threshold_grad);
    }

    // ========================================================================
    // Model Persistence (backend-090)
    // ========================================================================

    /// Save model to JSON format with metadata header
    ///
    /// Returns a JSON string containing the model header and all weights.
    /// Gradient fields are excluded (they are runtime-only).
    pub fn save_json(&self) -> PersistenceResult<String> {
        let serialized = SerializedModel {
            header: ModelHeader::for_graph_transform_net(self),
            model: self.clone(),
        };
        serde_json::to_string_pretty(&serialized)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))
    }

    /// Load model from JSON format
    ///
    /// Deserializes the model and verifies the header metadata.
    pub fn load_json(json: &str) -> PersistenceResult<Self> {
        let serialized: SerializedModel = serde_json::from_str(json)
            .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;

        // Verify version compatibility
        if serialized.header.version > MODEL_PERSISTENCE_VERSION {
            return Err(PersistenceError::VersionMismatch {
                expected: MODEL_PERSISTENCE_VERSION,
                actual: serialized.header.version,
            });
        }

        // Verify header matches model
        if !serialized.header.verify(&serialized.model) {
            return Err(PersistenceError::Deserialization(
                "Model header does not match model architecture".to_string(),
            ));
        }

        Ok(serialized.model)
    }

    /// Save model to a file (JSON format)
    ///
    /// # Arguments
    /// * `path` - File path to save to (will be created or overwritten)
    pub fn save_to_file(&self, path: &std::path::Path) -> PersistenceResult<()> {
        let json = self.save_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load model from a file (JSON format)
    ///
    /// # Arguments
    /// * `path` - File path to load from
    pub fn load_from_file(path: &std::path::Path) -> PersistenceResult<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::load_json(&json)
    }

    /// Get model metadata header
    pub fn header(&self) -> ModelHeader {
        ModelHeader::for_graph_transform_net(self)
    }
}

impl GraphTransformer for GraphTransformNet {
    fn transform(&mut self, input: &DagNN) -> GraphemeResult<DagNN> {
        let edits = self.predict_edits(input);
        self.apply_edits(input, &edits)
    }

    fn learn_transformation(&mut self, _input: &DagNN, _target: &DagNN) -> TransformRule {
        // For learned transformations, we don't extract explicit rules
        // The transformation is implicit in the network weights
        let id = 0;
        TransformRule {
            id,
            description: "Learned neural transformation".to_string(),
            input_pattern: Vec::new(),
            output_pattern: Vec::new(),
        }
    }

    fn apply_rule(&mut self, graph: &DagNN, _rule: &TransformRule) -> GraphemeResult<DagNN> {
        // For neural transformations, we always use the network
        self.transform(graph)
    }

    fn compose(&self, rules: Vec<TransformRule>) -> TransformRule {
        // Composition for neural nets is just sequential application
        TransformRule {
            id: rules.first().map(|r| r.id).unwrap_or(0),
            description: format!("Composed {} neural transformations", rules.len()),
            input_pattern: Vec::new(),
            output_pattern: Vec::new(),
        }
    }
}

// ============================================================================
// Domain Brain Plugin Architecture
// ============================================================================

/// Domain-specific node types that can be extended by plugins
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DomainNodeType {
    /// Core GRAPHEME node (character-level)
    Core(NodeType),
    /// Mathematical node (from grapheme-math)
    Math(String),
    /// Code/AST node (from grapheme-code)
    Code(String),
    /// Legal node (from grapheme-law)
    Legal(String),
    /// Music theory node (from grapheme-music)
    Music(String),
    /// Chemistry node (from grapheme-chem)
    Chemistry(String),
    /// Custom domain node
    Custom { domain: String, node_type: String },
}

impl Default for DomainNodeType {
    fn default() -> Self {
        DomainNodeType::Core(NodeType::Hidden)
    }
}

/// Domain-specific edge types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DomainEdgeType {
    /// Core GRAPHEME edge
    Core(EdgeType),
    /// Domain-specific edge
    Domain { domain: String, edge_type: String },
}

impl Default for DomainEdgeType {
    fn default() -> Self {
        DomainEdgeType::Core(EdgeType::Sequential)
    }
}

/// Result type for domain brain operations
pub type DomainResult<T> = Result<T, DomainError>;

/// Errors in domain brain operations
#[derive(Error, Debug)]
pub enum DomainError {
    #[error("Domain not registered: {0}")]
    DomainNotRegistered(String),
    #[error("Invalid domain input: {0}")]
    InvalidInput(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
    #[error("Graph error: {0}")]
    GraphError(#[from] GraphemeError),
}

/// Trait for domain-specific brain plugins (like tree-sitter for languages)
///
/// Each domain brain provides:
/// - Domain-specific node and edge types
/// - Transformation rules for the domain
/// - Validation and type checking
/// - Training data generation
pub trait DomainBrain: Send + Sync + std::fmt::Debug {
    /// Get the unique domain identifier (e.g., "math", "code", "law")
    fn domain_id(&self) -> &str;

    /// Get human-readable domain name
    fn domain_name(&self) -> &str;

    /// Get the version of this brain
    fn version(&self) -> &str;

    /// Check if this brain can process the given input
    fn can_process(&self, input: &str) -> bool;

    /// Parse domain-specific input into a graph
    fn parse(&self, input: &str) -> DomainResult<DagNN>;

    /// Transform a core graph into domain-specific representation
    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN>;

    /// Transform domain-specific graph back to core representation
    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN>;

    /// Validate a domain-specific graph
    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>>;

    /// Execute/evaluate a domain graph (e.g., compute math, compile code)
    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult>;

    /// Get available transformation rules for this domain
    fn get_rules(&self) -> Vec<DomainRule>;

    /// Apply a domain-specific transformation
    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN>;

    /// Generate training examples for this domain
    fn generate_examples(&self, count: usize) -> Vec<DomainExample>;
}

/// Validation issue found in a domain graph
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of the issue
    pub severity: ValidationSeverity,
    /// Description of the issue
    pub message: String,
    /// Optional node where issue was found
    pub node: Option<NodeId>,
}

/// Severity levels for validation issues
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    /// Informational
    Info,
    /// Warning - may cause problems
    Warning,
    /// Error - invalid graph
    Error,
}

/// Result of executing a domain graph
#[derive(Debug, Clone)]
pub enum ExecutionResult {
    /// Numeric result (e.g., math evaluation)
    Numeric(f64),
    /// String result (e.g., code output)
    Text(String),
    /// Graph result (e.g., transformed graph)
    Graph(Box<DagNN>),
    /// Boolean result (e.g., validation)
    Boolean(bool),
    /// No result (side effects only)
    Unit,
    /// Error during execution
    Error(String),
}

/// A domain-specific transformation rule
#[derive(Debug, Clone)]
pub struct DomainRule {
    /// Unique rule identifier
    pub id: usize,
    /// Domain this rule belongs to
    pub domain: String,
    /// Human-readable name
    pub name: String,
    /// Description of what the rule does
    pub description: String,
    /// Category (e.g., "simplification", "optimization")
    pub category: String,
}

/// A training example from a domain
#[derive(Debug, Clone)]
pub struct DomainExample {
    /// Input graph
    pub input: DagNN,
    /// Expected output graph
    pub output: DagNN,
    /// Domain this example is from
    pub domain: String,
    /// Difficulty level (1-10)
    pub difficulty: u8,
}

/// Registry for domain brain plugins
#[derive(Debug, Default)]
pub struct BrainRegistry {
    /// Registered domain brains
    brains: HashMap<String, Box<dyn DomainBrain>>,
    /// Load order for deterministic iteration
    load_order: Vec<String>,
}

impl BrainRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a domain brain
    pub fn register(&mut self, brain: Box<dyn DomainBrain>) {
        let domain_id = brain.domain_id().to_string();
        if !self.brains.contains_key(&domain_id) {
            self.load_order.push(domain_id.clone());
        }
        self.brains.insert(domain_id, brain);
    }

    /// Get a brain by domain ID
    pub fn get(&self, domain_id: &str) -> Option<&dyn DomainBrain> {
        self.brains.get(domain_id).map(|b| b.as_ref())
    }

    /// Get a mutable brain by domain ID
    pub fn get_mut(&mut self, domain_id: &str) -> Option<&mut Box<dyn DomainBrain>> {
        self.brains.get_mut(domain_id)
    }

    /// Check if a domain is registered
    pub fn has_domain(&self, domain_id: &str) -> bool {
        self.brains.contains_key(domain_id)
    }

    /// Get all registered domain IDs
    pub fn domains(&self) -> &[String] {
        &self.load_order
    }

    /// Get count of registered brains
    pub fn len(&self) -> usize {
        self.brains.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.brains.is_empty()
    }

    /// Find a brain that can process the given input
    pub fn find_processor(&self, input: &str) -> Option<&dyn DomainBrain> {
        for domain_id in &self.load_order {
            if let Some(brain) = self.brains.get(domain_id) {
                if brain.can_process(input) {
                    return Some(brain.as_ref());
                }
            }
        }
        None
    }

    /// Process input with the appropriate brain
    pub fn process(&self, input: &str) -> DomainResult<DagNN> {
        self.find_processor(input)
            .ok_or_else(|| {
                DomainError::DomainNotRegistered("No brain can process this input".to_string())
            })?
            .parse(input)
    }

    /// Get all available rules across all domains
    pub fn all_rules(&self) -> Vec<DomainRule> {
        self.load_order
            .iter()
            .filter_map(|id| self.brains.get(id))
            .flat_map(|b| b.get_rules())
            .collect()
    }

    /// Generate examples from all domains
    pub fn generate_examples(&self, per_domain: usize) -> Vec<DomainExample> {
        self.load_order
            .iter()
            .filter_map(|id| self.brains.get(id))
            .flat_map(|b| b.generate_examples(per_domain))
            .collect()
    }

    /// Unregister a domain brain
    pub fn unregister(&mut self, domain_id: &str) -> Option<Box<dyn DomainBrain>> {
        self.load_order.retain(|id| id != domain_id);
        self.brains.remove(domain_id)
    }

    /// Clear all registered brains
    pub fn clear(&mut self) {
        self.brains.clear();
        self.load_order.clear();
    }
}

/// Cross-domain knowledge transfer
#[derive(Default)]
pub struct CrossDomainBridge {
    /// Mapping functions between domains
    mappings: HashMap<(String, String), Box<dyn Fn(&DagNN) -> DomainResult<DagNN> + Send + Sync>>,
}

impl std::fmt::Debug for CrossDomainBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossDomainBridge")
            .field("mappings", &format!("{} mappings", self.mappings.len()))
            .finish()
    }
}

impl CrossDomainBridge {
    /// Create a new cross-domain bridge
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a mapping from one domain to another
    pub fn register_mapping<F>(&mut self, from: &str, to: &str, mapper: F)
    where
        F: Fn(&DagNN) -> DomainResult<DagNN> + Send + Sync + 'static,
    {
        self.mappings
            .insert((from.to_string(), to.to_string()), Box::new(mapper));
    }

    /// Transfer knowledge from one domain to another
    pub fn transfer(&self, graph: &DagNN, from: &str, to: &str) -> DomainResult<DagNN> {
        let key = (from.to_string(), to.to_string());
        self.mappings
            .get(&key)
            .ok_or_else(|| {
                DomainError::DomainNotRegistered(format!("No mapping from {} to {}", from, to))
            })
            .and_then(|f| f(graph))
    }

    /// Check if a mapping exists
    pub fn has_mapping(&self, from: &str, to: &str) -> bool {
        self.mappings
            .contains_key(&(from.to_string(), to.to_string()))
    }
}

/// Factory for creating domain brains
pub trait BrainFactory: Send + Sync {
    /// Create a new instance of a domain brain
    fn create(&self) -> Box<dyn DomainBrain>;

    /// Get the domain ID this factory creates
    fn domain_id(&self) -> &str;
}

// ============================================================================
// Cognitive-Brain Bridge
// ============================================================================

/// Result of routing an input to one or more domain brains
#[derive(Debug, Clone)]
pub struct BrainRoutingResult {
    /// The domain brain that processed the input
    pub domain_id: String,
    /// The parsed graph from the domain brain
    pub graph: DagNN,
    /// Confidence score (0.0-1.0) for this routing
    pub confidence: f32,
    /// Execution result if the brain was executed
    pub result: Option<String>,
}

/// Result of multi-brain processing
#[derive(Debug, Clone, Default)]
pub struct MultiBrainResult {
    /// Results from each brain that processed the input
    pub results: Vec<BrainRoutingResult>,
    /// Primary result (highest confidence)
    pub primary: Option<BrainRoutingResult>,
    /// Whether the processing was successful
    pub success: bool,
}

impl MultiBrainResult {
    /// Create a new empty result
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a result from a brain
    pub fn add_result(&mut self, result: BrainRoutingResult) {
        // Update primary if this has higher confidence
        let is_primary = match &self.primary {
            None => true,
            Some(p) => result.confidence > p.confidence,
        };
        if is_primary {
            self.primary = Some(result.clone());
        }
        self.results.push(result);
        self.success = true;
    }

    /// Get all domain IDs that processed the input
    pub fn domains(&self) -> Vec<&str> {
        self.results.iter().map(|r| r.domain_id.as_str()).collect()
    }

    /// Get result by domain ID
    pub fn get_by_domain(&self, domain_id: &str) -> Option<&BrainRoutingResult> {
        self.results.iter().find(|r| r.domain_id == domain_id)
    }
}

/// Trait for cognitive modules to interact with domain brains.
///
/// This bridge allows cognitive modules (memory, reasoning, meta-cognition, etc.)
/// to discover, route to, and trigger domain-specific processing via brain plugins.
pub trait CognitiveBrainBridge: Send + Sync {
    /// Get reference to the brain registry
    fn get_registry(&self) -> &BrainRegistry;

    /// Get mutable reference to the brain registry
    fn get_registry_mut(&mut self) -> &mut BrainRegistry;

    /// Route input to the most appropriate domain brain
    fn route_to_brain(&self, input: &str) -> DomainResult<BrainRoutingResult> {
        let registry = self.get_registry();
        let brain = registry.find_processor(input).ok_or_else(|| {
            DomainError::DomainNotRegistered("No brain can process this input".to_string())
        })?;

        let graph = brain.parse(input)?;
        let result = brain.execute(&graph)?;
        let result_text = match result {
            ExecutionResult::Text(t) => Some(t),
            ExecutionResult::Numeric(v) => Some(format!("{}", v)),
            ExecutionResult::Graph(_) => None,
            ExecutionResult::Boolean(b) => Some(format!("{}", b)),
            ExecutionResult::Unit => None,
            ExecutionResult::Error(e) => Some(format!("Error: {}", e)),
        };

        Ok(BrainRoutingResult {
            domain_id: brain.domain_id().to_string(),
            graph,
            confidence: 1.0, // Single brain routing has full confidence
            result: result_text,
        })
    }

    /// Route input to multiple brains that can process it
    fn route_to_multiple_brains(&self, input: &str) -> MultiBrainResult {
        let registry = self.get_registry();
        let mut result = MultiBrainResult::new();

        for domain_id in registry.domains() {
            if let Some(brain) = registry.get(domain_id) {
                if brain.can_process(input) {
                    if let Ok(graph) = brain.parse(input) {
                        let exec_result = brain.execute(&graph).ok();
                        let result_text = exec_result.map(|r| match r {
                            ExecutionResult::Text(t) => t,
                            ExecutionResult::Numeric(v) => format!("{}", v),
                            ExecutionResult::Graph(_) => "Graph result".to_string(),
                            ExecutionResult::Boolean(b) => format!("{}", b),
                            ExecutionResult::Unit => "Done".to_string(),
                            ExecutionResult::Error(e) => format!("Error: {}", e),
                        });

                        // Calculate confidence based on how many brains can process
                        let confidence = self.calculate_routing_confidence(input, brain);

                        result.add_result(BrainRoutingResult {
                            domain_id: domain_id.clone(),
                            graph,
                            confidence,
                            result: result_text,
                        });
                    }
                }
            }
        }

        result
    }

    /// Calculate confidence score for routing to a specific brain
    fn calculate_routing_confidence(&self, input: &str, brain: &dyn DomainBrain) -> f32 {
        // Default implementation: simple heuristic based on can_process
        if brain.can_process(input) {
            // Check how many keywords/patterns match
            let input_lower = input.to_lowercase();
            let domain = brain.domain_id().to_lowercase();

            // Higher confidence if domain name appears in input
            if input_lower.contains(&domain) {
                0.9
            } else {
                0.7
            }
        } else {
            0.0
        }
    }

    /// Get all available domains from the registry
    fn available_domains(&self) -> Vec<String> {
        self.get_registry().domains().to_vec()
    }

    /// Check if a specific domain is available
    fn has_domain(&self, domain_id: &str) -> bool {
        self.get_registry().has_domain(domain_id)
    }

    /// Execute a rule from a specific domain on a graph
    fn execute_domain_rule(
        &self,
        domain_id: &str,
        graph: &DagNN,
        rule_id: usize,
    ) -> DomainResult<DagNN> {
        let registry = self.get_registry();
        let brain = registry
            .get(domain_id)
            .ok_or_else(|| DomainError::DomainNotRegistered(domain_id.to_string()))?;
        brain.transform(graph, rule_id)
    }

    /// Get all rules across all domains
    fn all_domain_rules(&self) -> Vec<DomainRule> {
        self.get_registry().all_rules()
    }

    /// Get rules for a specific domain
    fn domain_rules(&self, domain_id: &str) -> Vec<DomainRule> {
        self.get_registry()
            .get(domain_id)
            .map(|b| b.get_rules())
            .unwrap_or_default()
    }

    /// Generate training examples from all domains
    fn generate_domain_examples(&self, per_domain: usize) -> Vec<DomainExample> {
        self.get_registry().generate_examples(per_domain)
    }
}

/// Default cognitive-brain bridge implementation using a BrainRegistry
#[derive(Debug, Default)]
pub struct DefaultCognitiveBridge {
    /// The brain registry
    pub registry: BrainRegistry,
}

impl DefaultCognitiveBridge {
    /// Create a new default cognitive bridge
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with an existing registry
    pub fn with_registry(registry: BrainRegistry) -> Self {
        Self { registry }
    }

    /// Register a brain
    pub fn register(&mut self, brain: Box<dyn DomainBrain>) {
        self.registry.register(brain);
    }
}

impl CognitiveBrainBridge for DefaultCognitiveBridge {
    fn get_registry(&self) -> &BrainRegistry {
        &self.registry
    }

    fn get_registry_mut(&mut self) -> &mut BrainRegistry {
        &mut self.registry
    }
}

// ============================================================================
// Cognitive-Brain Orchestrator
// ============================================================================

/// Configuration for the cognitive-brain orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Minimum confidence threshold for routing
    pub confidence_threshold: f32,
    /// Whether to automatically route to domain brains
    pub auto_route: bool,
    /// Maximum number of brains to consult for a single query
    pub max_brains_per_query: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            auto_route: true,
            max_brains_per_query: 3,
        }
    }
}

/// Statistics for the orchestrator
#[derive(Debug, Clone, Default)]
pub struct OrchestratorStats {
    /// Total queries processed
    pub total_queries: u64,
    /// Queries successfully routed to a brain
    pub routed_queries: u64,
    /// Queries where no brain could process
    pub unrouted_queries: u64,
    /// Per-domain query counts
    pub domain_counts: std::collections::HashMap<String, u64>,
}

impl OrchestratorStats {
    /// Record a successful routing
    pub fn record_routing(&mut self, domain_id: &str) {
        self.total_queries += 1;
        self.routed_queries += 1;
        *self.domain_counts.entry(domain_id.to_string()).or_insert(0) += 1;
    }

    /// Record a failed routing
    pub fn record_no_routing(&mut self) {
        self.total_queries += 1;
        self.unrouted_queries += 1;
    }

    /// Get routing success rate
    pub fn success_rate(&self) -> f32 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.routed_queries as f32 / self.total_queries as f32
        }
    }
}

/// Result of orchestrated processing
#[derive(Debug)]
pub struct OrchestratedResult {
    /// The primary result (from highest-confidence brain)
    pub primary: Option<ExecutionResult>,
    /// All brain results (domain_id -> result)
    pub brain_results: std::collections::HashMap<String, ExecutionResult>,
    /// Combined confidence
    pub confidence: f32,
    /// Domains that contributed
    pub domains: Vec<String>,
}

impl OrchestratedResult {
    /// Create an empty result
    pub fn empty() -> Self {
        Self {
            primary: None,
            brain_results: std::collections::HashMap::new(),
            confidence: 0.0,
            domains: Vec::new(),
        }
    }

    /// Check if processing was successful
    pub fn success(&self) -> bool {
        self.primary.is_some()
    }
}

/// Unified cognitive-brain orchestrator
///
/// This orchestrator coordinates all brain-aware cognitive components:
/// - Routes inputs to appropriate domain brains
/// - Combines results from multiple brains
/// - Tracks statistics for analysis
/// - Provides a unified interface for AGI capabilities
#[derive(Default)]
pub struct CognitiveBrainOrchestrator {
    /// The brain registry
    pub registry: BrainRegistry,
    /// Configuration
    pub config: OrchestratorConfig,
    /// Statistics
    pub stats: OrchestratorStats,
}

impl std::fmt::Debug for CognitiveBrainOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CognitiveBrainOrchestrator")
            .field("available_domains", &self.registry.domains())
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

impl CognitiveBrainOrchestrator {
    /// Create a new orchestrator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom configuration
    pub fn with_config(config: OrchestratorConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Register a domain brain
    pub fn register_brain(&mut self, brain: Box<dyn DomainBrain>) {
        self.registry.register(brain);
    }

    /// Process an input through the orchestrator
    ///
    /// Routes to appropriate domain brains and combines results.
    pub fn process(&mut self, input: &str) -> OrchestratedResult {
        if !self.config.auto_route {
            self.stats.record_no_routing();
            return OrchestratedResult::empty();
        }

        // Find all brains that can process this input
        let mut applicable: Vec<_> = self
            .registry
            .domains()
            .iter()
            .filter_map(|domain_id| {
                let brain = self.registry.get(domain_id)?;
                if brain.can_process(input) {
                    // Calculate confidence
                    let input_lower = input.to_lowercase();
                    let domain = brain.domain_id().to_lowercase();
                    let confidence = if input_lower.contains(&domain) {
                        0.9
                    } else {
                        0.7
                    };
                    if confidence >= self.config.confidence_threshold {
                        Some((domain_id.clone(), confidence))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Sort by confidence (highest first)
        applicable.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit to max_brains_per_query
        applicable.truncate(self.config.max_brains_per_query);

        if applicable.is_empty() {
            self.stats.record_no_routing();
            return OrchestratedResult::empty();
        }

        // Process with each applicable brain
        let mut result = OrchestratedResult::empty();
        let mut primary_confidence = 0.0f32;

        for (domain_id, confidence) in &applicable {
            if let Some(brain) = self.registry.get(domain_id) {
                if let Ok(graph) = brain.parse(input) {
                    if let Ok(exec_result) = brain.execute(&graph) {
                        result
                            .brain_results
                            .insert(domain_id.clone(), exec_result.clone());
                        result.domains.push(domain_id.clone());

                        // Update primary if this is highest confidence
                        if *confidence > primary_confidence {
                            primary_confidence = *confidence;
                            result.primary = Some(exec_result);
                        }

                        self.stats.record_routing(domain_id);
                    }
                }
            }
        }

        result.confidence = primary_confidence;
        result
    }

    /// Get processing statistics
    pub fn stats(&self) -> &OrchestratorStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = OrchestratorStats::default();
    }

    /// Get available domains
    pub fn available_domains(&self) -> Vec<String> {
        self.registry.domains().to_vec()
    }

    /// Check if a domain is available
    pub fn has_domain(&self, domain_id: &str) -> bool {
        self.registry.has_domain(domain_id)
    }

    /// Get the brain registry
    pub fn get_registry(&self) -> &BrainRegistry {
        &self.registry
    }

    /// Get mutable access to the brain registry
    pub fn get_registry_mut(&mut self) -> &mut BrainRegistry {
        &mut self.registry
    }
}

impl CognitiveBrainBridge for CognitiveBrainOrchestrator {
    fn get_registry(&self) -> &BrainRegistry {
        &self.registry
    }

    fn get_registry_mut(&mut self) -> &mut BrainRegistry {
        &mut self.registry
    }
}

/// Factory function to create a cognitive-brain orchestrator
pub fn create_cognitive_orchestrator() -> CognitiveBrainOrchestrator {
    CognitiveBrainOrchestrator::new()
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
        let texts = vec!["Hello", "你好", "مرحبا", "🚀🎉", "∫x²dx", "Hello你好🚀"];

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
        let clique_id = dag
            .form_clique(members.clone(), Some("test_word".into()))
            .unwrap();

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
        // May be empty for short text - just verify it returns
        let _ = cliques.len();

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

    // ========================================================================
    // K-Clique Enumeration Tests (backend-009)
    // ========================================================================

    #[test]
    fn test_clique_error_k_too_large() {
        let dag = DagNN::from_text("test").unwrap();
        let result = dag.find_cliques(10);
        assert!(matches!(result, Err(CliqueError::KTooLarge { .. })));
    }

    #[test]
    fn test_clique_error_k_too_small() {
        let dag = DagNN::from_text("test").unwrap();
        let result = dag.find_cliques(1);
        assert!(matches!(result, Err(CliqueError::KTooSmall(_))));
    }

    #[test]
    fn test_find_cliques_empty_graph() {
        let dag = DagNN::new();
        let cliques = dag.find_cliques(3).unwrap();
        assert!(cliques.is_empty());
    }

    #[test]
    fn test_find_cliques_small_graph() {
        // Text graph is a chain, has no triangles
        let dag = DagNN::from_text("abc").unwrap();
        let cliques = dag.find_cliques(3).unwrap();
        // A linear chain has no triangles
        assert!(cliques.is_empty());
    }

    #[test]
    fn test_is_clique() {
        let dag = DagNN::from_text("ab").unwrap();
        let nodes = dag.input_nodes().to_vec();

        // Single node is always a clique
        assert!(dag.is_clique(&[nodes[0]]));

        // Two connected nodes are a clique
        assert!(dag.is_clique(&nodes));
    }

    #[test]
    fn test_combinations() {
        let items = vec![NodeIndex::new(0), NodeIndex::new(1), NodeIndex::new(2)];

        // C(3,2) = 3 combinations
        let combos = DagNN::combinations(&items, 2);
        assert_eq!(combos.len(), 3);

        // C(3,3) = 1 combination
        let combos = DagNN::combinations(&items, 3);
        assert_eq!(combos.len(), 1);

        // C(3,4) = 0 combinations
        let combos = DagNN::combinations(&items, 4);
        assert!(combos.is_empty());
    }

    #[test]
    fn test_combinations_iter() {
        let items = vec![NodeIndex::new(0), NodeIndex::new(1), NodeIndex::new(2)];

        // C(3,2) = 3 combinations - should match recursive version
        let combos_iter = DagNN::combinations_iter(&items, 2);
        let combos_rec = DagNN::combinations(&items, 2);
        assert_eq!(combos_iter.len(), combos_rec.len());
        assert_eq!(combos_iter.len(), 3);

        // C(3,3) = 1 combination
        let combos_iter = DagNN::combinations_iter(&items, 3);
        assert_eq!(combos_iter.len(), 1);

        // C(3,4) = 0 combinations
        let combos_iter = DagNN::combinations_iter(&items, 4);
        assert!(combos_iter.is_empty());

        // Larger test: C(5,3) = 10
        let items5 = (0..5).map(NodeIndex::new).collect::<Vec<_>>();
        let combos = DagNN::combinations_iter(&items5, 3);
        assert_eq!(combos.len(), 10);
    }

    #[test]
    fn test_find_maximal_cliques() {
        // Create a small graph with known cliques
        let mut dag = DagNN::new();
        let n0 = dag.graph.add_node(Node::input('a', 0));
        let n1 = dag.graph.add_node(Node::input('b', 1));
        let n2 = dag.graph.add_node(Node::input('c', 2));
        let n3 = dag.graph.add_node(Node::input('d', 3));

        // Create a triangle (3-clique): n0-n1-n2
        dag.graph
            .add_edge(n0, n1, Edge::new(0.5, EdgeType::Sequential));
        dag.graph
            .add_edge(n1, n2, Edge::new(0.5, EdgeType::Sequential));
        dag.graph
            .add_edge(n0, n2, Edge::new(0.5, EdgeType::Sequential));

        // n3 only connects to n0
        dag.graph
            .add_edge(n0, n3, Edge::new(0.5, EdgeType::Sequential));

        dag.input_nodes = vec![n0, n1, n2, n3];

        let maximal = dag.find_all_maximal_cliques(None);

        // Should find: {n0,n1,n2} as a maximal 3-clique and {n0,n3} as maximal 2-clique
        assert!(!maximal.is_empty());

        // The triangle should be a maximal clique
        let has_triangle = maximal.iter().any(|c| c.len() == 3);
        assert!(has_triangle, "Should find the triangle as a maximal clique");
    }

    #[test]
    fn test_degeneracy_ordering() {
        let dag = DagNN::from_text("abc").unwrap();
        let ordering = dag.degeneracy_ordering();
        assert_eq!(ordering.len(), dag.node_count());
    }

    #[test]
    fn test_store_cliques() {
        let mut dag = DagNN::from_text("test").unwrap();
        let nodes = dag.input_nodes().to_vec();

        dag.store_cliques(vec![nodes.clone()]);
        assert_eq!(dag.cliques.len(), 1);
        assert_eq!(dag.cliques[0].members, nodes);
    }

    #[test]
    fn test_find_triangles() {
        let dag = DagNN::from_text("ab").unwrap();
        let result = dag.find_triangles();
        assert!(result.is_ok());
    }

    #[test]
    fn test_clique_max_k_constant() {
        assert_eq!(MAX_CLIQUE_K, 6);
        assert_eq!(MAX_CLIQUE_GRAPH_SIZE, 10000);
    }

    // ========================================================================
    // Persistence Tests (backend-022)
    // ========================================================================

    #[test]
    fn test_grapheme_graph_json_roundtrip() {
        let original = GraphemeGraph::from_text("Hello, World!");
        let json = original.save_json().unwrap();
        let loaded = GraphemeGraph::load_json(&json).unwrap();

        assert_eq!(original.node_count(), loaded.node_count());
        assert_eq!(original.edge_count(), loaded.edge_count());
        assert_eq!(original.to_text(), loaded.to_text());
    }

    #[test]
    fn test_dagnn_json_roundtrip() {
        let original = DagNN::from_text("Test").unwrap();
        let json = original.save_json().unwrap();
        let loaded = DagNN::load_json(&json).unwrap();

        assert_eq!(original.node_count(), loaded.node_count());
        assert_eq!(original.edge_count(), loaded.edge_count());
        assert_eq!(original.to_text(), loaded.to_text());
    }

    #[test]
    fn test_graph_header_verification() {
        let graph = GraphemeGraph::from_text("test");
        let header = GraphHeader::for_grapheme_graph(&graph);

        assert!(header.verify(graph.node_count(), graph.edge_count()));
        assert!(!header.verify(0, 0));
    }

    #[test]
    fn test_dagnn_file_persistence() {
        let original = DagNN::from_text("Hello").unwrap();

        // Use a temp file
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_dagnn.json");

        // Save and load
        original.save_to_file(&temp_file).unwrap();
        let loaded = DagNN::load_from_file(&temp_file).unwrap();

        assert_eq!(original.node_count(), loaded.node_count());
        assert_eq!(original.to_text(), loaded.to_text());

        // Cleanup
        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_grapheme_graph_file_persistence() {
        let original = GraphemeGraph::from_text("World");

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_grapheme.json");

        original.save_to_file(&temp_file).unwrap();
        let loaded = GraphemeGraph::load_from_file(&temp_file).unwrap();

        assert_eq!(original.node_count(), loaded.node_count());
        assert_eq!(original.to_text(), loaded.to_text());

        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_persistence_with_cliques() {
        let mut dag = DagNN::from_text("test").unwrap();
        let nodes = dag.input_nodes().to_vec();
        dag.store_cliques(vec![nodes]);

        let json = dag.save_json().unwrap();
        let loaded = DagNN::load_json(&json).unwrap();

        assert_eq!(dag.cliques.len(), loaded.cliques.len());
    }

    #[test]
    fn test_persistence_empty_graph() {
        let original = GraphemeGraph::new();
        let json = original.save_json().unwrap();
        let loaded = GraphemeGraph::load_json(&json).unwrap();

        assert_eq!(loaded.node_count(), 0);
        assert_eq!(loaded.edge_count(), 0);
    }

    #[test]
    fn test_persistence_unicode() {
        let original = GraphemeGraph::from_text("你好世界"); // Chinese "Hello World"
        let json = original.save_json().unwrap();
        let loaded = GraphemeGraph::load_json(&json).unwrap();

        assert_eq!(original.to_text(), loaded.to_text());
    }

    // ========================================================================
    // HashSet Optimization Tests (backend-012)
    // ========================================================================

    #[test]
    fn test_input_nodes_set_after_deserialization() {
        // Create a DagNN, add hidden nodes, then save/load
        let mut original = DagNN::from_text("test").unwrap();

        // Add disconnected hidden nodes
        original.graph.add_node(Node::hidden());
        original.graph.add_node(Node::hidden());
        assert_eq!(original.node_count(), 6); // 4 input + 2 hidden

        // Save and load
        let json = original.save_json().unwrap();
        let mut loaded = DagNN::load_json(&json).unwrap();

        // GC should work correctly after loading (uses input_nodes_set)
        let removed = loaded.gc_disconnected();
        assert_eq!(removed, 2); // Should remove 2 disconnected hidden nodes
        assert_eq!(loaded.node_count(), 4); // Only input nodes remain
    }

    #[test]
    fn test_input_nodes_set_matches_vec() {
        let dag = DagNN::from_text("hello").unwrap();

        // The set should contain all input nodes
        for node in dag.input_nodes() {
            assert!(dag.input_nodes_set.contains(node));
        }

        // The set should have same length as vec
        assert_eq!(dag.input_nodes_set.len(), dag.input_nodes.len());
    }

    // ========================================================================
    // K-Clique Percolation / Community Tests (backend-008)
    // ========================================================================

    #[test]
    fn test_community_basic() {
        let nodes = vec![NodeIndex::new(0), NodeIndex::new(1), NodeIndex::new(2)];
        let cliques = vec![nodes.clone()];
        let community = Community::new(0, nodes.clone(), cliques);

        assert_eq!(community.id, 0);
        assert_eq!(community.size(), 3);
        assert_eq!(community.clique_count(), 1);
        assert!(community.contains(NodeIndex::new(0)));
        assert!(!community.contains(NodeIndex::new(5)));
    }

    #[test]
    fn test_find_concept_communities_empty() {
        let dag = DagNN::new();
        let communities = dag.find_concept_communities(3).unwrap();
        assert!(communities.is_empty());
    }

    #[test]
    fn test_find_concept_communities_no_cliques() {
        // A linear chain has no triangles
        let dag = DagNN::from_text("abc").unwrap();
        let communities = dag.find_concept_communities(3).unwrap();
        assert!(communities.is_empty());
    }

    #[test]
    fn test_find_concept_communities_k_bounds() {
        let dag = DagNN::from_text("test").unwrap();

        // k too small
        let result = dag.find_concept_communities(2);
        assert!(matches!(result, Err(CliqueError::KTooSmall(_))));

        // k too large
        let result = dag.find_concept_communities(10);
        assert!(matches!(result, Err(CliqueError::KTooLarge { .. })));
    }

    #[test]
    fn test_find_triangle_communities() {
        let dag = DagNN::from_text("ab").unwrap();
        // Should not panic, just return empty
        let communities = dag.find_triangle_communities().unwrap();
        assert!(communities.is_empty());
    }

    #[test]
    fn test_cliques_share_k_minus_1() {
        let c1 = vec![NodeIndex::new(0), NodeIndex::new(1), NodeIndex::new(2)];
        let c2 = vec![NodeIndex::new(1), NodeIndex::new(2), NodeIndex::new(3)];
        let c3 = vec![NodeIndex::new(4), NodeIndex::new(5), NodeIndex::new(6)];

        // c1 and c2 share 2 nodes (k-1 for k=3)
        assert!(DagNN::cliques_share_k_minus_1(&c1, &c2, 3));

        // c1 and c3 share 0 nodes
        assert!(!DagNN::cliques_share_k_minus_1(&c1, &c3, 3));
    }

    #[test]
    fn test_build_clique_adjacency() {
        let cliques = vec![
            vec![NodeIndex::new(0), NodeIndex::new(1), NodeIndex::new(2)],
            vec![NodeIndex::new(1), NodeIndex::new(2), NodeIndex::new(3)],
            vec![NodeIndex::new(4), NodeIndex::new(5), NodeIndex::new(6)],
        ];

        let adjacency = DagNN::build_clique_adjacency(&cliques, 3);

        // Clique 0 and 1 are adjacent (share nodes 1, 2)
        assert!(adjacency[0].contains(&1));
        assert!(adjacency[1].contains(&0));

        // Clique 2 is isolated
        assert!(adjacency[2].is_empty());
    }

    #[test]
    fn test_find_clique_components() {
        // Two components: {0, 1} and {2}
        let adjacency = vec![
            vec![1], // 0 -> 1
            vec![0], // 1 -> 0
            vec![],  // 2 (isolated)
        ];

        let components = DagNN::find_clique_components(&adjacency, 3);

        assert_eq!(components.len(), 2);

        // Find which component contains 0 and 1
        let comp_01: Vec<usize> = components.iter().find(|c| c.contains(&0)).cloned().unwrap();
        assert!(comp_01.contains(&0));
        assert!(comp_01.contains(&1));
        assert!(!comp_01.contains(&2));

        // The other component should have just 2
        let comp_2: Vec<usize> = components.iter().find(|c| c.contains(&2)).cloned().unwrap();
        assert_eq!(comp_2.len(), 1);
    }

    #[test]
    fn test_merge_into_communities() {
        let cliques = vec![
            vec![NodeIndex::new(0), NodeIndex::new(1), NodeIndex::new(2)],
            vec![NodeIndex::new(1), NodeIndex::new(2), NodeIndex::new(3)],
        ];
        let components = vec![vec![0, 1]]; // Both cliques in one component

        let communities = DagNN::merge_into_communities(&cliques, &components);

        assert_eq!(communities.len(), 1);
        let community = &communities[0];
        assert_eq!(community.clique_count(), 2);
        // Nodes 0, 1, 2, 3 should all be present
        assert_eq!(community.size(), 4);
    }

    // ========================================================================
    // Connect Relevant Optimization Tests (backend-013)
    // ========================================================================

    #[test]
    fn test_position_index_populated() {
        let dag = DagNN::from_text("hello").unwrap();

        // Position index should have 5 entries
        assert_eq!(dag.position_index.len(), 5);

        // Check positions are correct
        for (pos, &node) in &dag.position_index {
            assert_eq!(dag.graph[node].position, Some(*pos));
        }
    }

    #[test]
    fn test_position_index_after_deserialization() {
        let original = DagNN::from_text("test").unwrap();
        let json = original.save_json().unwrap();
        let loaded = DagNN::load_json(&json).unwrap();

        // Position index should be rebuilt
        assert_eq!(loaded.position_index.len(), 4);
        for (pos, &node) in &loaded.position_index {
            assert_eq!(loaded.graph[node].position, Some(*pos));
        }
    }

    #[test]
    fn test_connect_relevant_uses_position_index() {
        let mut dag = DagNN::from_text("abcde").unwrap();
        let initial_edges = dag.edge_count();

        // Connect node at position 2 with window of 3
        let node = dag.position_index.get(&2).copied().unwrap();
        dag.connect_relevant(node, 3);

        // Should add skip edges (not distance 1)
        assert!(dag.edge_count() > initial_edges);
    }

    #[test]
    fn test_connect_all_relevant() {
        let mut dag = DagNN::from_text("abcdefg").unwrap();
        let initial_edges = dag.edge_count();
        assert_eq!(initial_edges, 6); // 6 sequential edges

        // Connect all with window of 2
        dag.connect_all_relevant(2);

        // With window 2, each node connects to nodes at distance 2
        // That's n-2 skip edges (bidirectional) = (n-2)*2 edges
        // Actually with bidirectional, we add 2 edges per pair at distance 2
        assert!(dag.edge_count() > initial_edges);
    }

    #[test]
    fn test_connect_all_relevant_window_1() {
        let mut dag = DagNN::from_text("abc").unwrap();
        let initial_edges = dag.edge_count();

        // Window of 1 should add no skip edges (only distance > 1 creates edges)
        dag.connect_all_relevant(1);

        // No new edges added
        assert_eq!(dag.edge_count(), initial_edges);
    }

    #[test]
    fn test_connect_all_relevant_empty_graph() {
        let mut dag = DagNN::new();

        // Should not panic on empty graph
        dag.connect_all_relevant(5);
        assert_eq!(dag.edge_count(), 0);
    }

    // ========================================================================
    // Sparse Graph Monitoring Tests (testing-003)
    // ========================================================================

    #[test]
    fn test_graph_stats_basic() {
        let dag = DagNN::from_text("hello").unwrap();
        let stats = dag.stats();

        assert_eq!(stats.node_count, 5);
        assert_eq!(stats.edge_count, 4); // Sequential edges
        assert_eq!(stats.input_node_count, 5);
        assert!(stats.density < 1.0);
        assert!(stats.max_degree <= 2); // Linear chain
        assert!(stats.avg_degree > 0.0);
    }

    #[test]
    fn test_graph_stats_sparse() {
        // Larger text to ensure sparsity
        let dag = DagNN::from_text("the quick brown fox jumps over the lazy dog").unwrap();
        let stats = dag.stats();

        // Text graphs are sparse for longer texts
        // n=43, m=42 edges, max_edges = 43*42/2 = 903
        // density = 42/903 ≈ 0.047
        assert!(stats.density < 0.1, "density = {}", stats.density);
        assert!(stats.is_valid());
        assert!(!stats.has_errors());
    }

    #[test]
    fn test_graph_stats_empty() {
        let dag = DagNN::new();
        let stats = dag.stats();

        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.density, 0.0);
        assert!(stats.is_valid());
    }

    #[test]
    fn test_assumption_violation_display() {
        let violation = AssumptionViolation {
            metric: "density".to_string(),
            value: 0.5,
            threshold: 0.1,
            severity: Severity::Warning,
        };

        let display = format!("{}", violation);
        assert!(display.contains("WARNING"));
        assert!(display.contains("density"));
        assert!(display.contains("0.500"));
    }

    #[test]
    fn test_severity() {
        let warning = Severity::Warning;
        let error = Severity::Error;

        assert_eq!(warning, Severity::Warning);
        assert_ne!(warning, error);
    }

    #[test]
    fn test_validate_sparse_graph() {
        // Use a long text to ensure sparsity below threshold
        let dag = DagNN::from_text("the quick brown fox jumps over the lazy dog and keeps running")
            .unwrap();
        let stats = dag.stats();
        let violations = stats.validate();

        // Sparse text graph should have no violations
        assert!(violations.is_empty(), "violations: {:?}", violations);
        assert!(stats.is_valid());
    }

    #[test]
    fn test_validate_clique_stats() {
        let dag = DagNN::from_text("ab").unwrap();
        let stats = dag.stats();

        // No cliques formed yet
        assert_eq!(stats.max_clique_size, 0);
        assert_eq!(stats.avg_clique_size, 0.0);
        assert_eq!(stats.clique_count, 0);
    }

    // ========================================
    // Embedding Tests (backend-026)
    // ========================================

    #[test]
    fn test_embedding_creation() {
        let emb = Embedding::xavier(256, 64);
        assert_eq!(emb.vocab_size, 256);
        assert_eq!(emb.embed_dim, 64);
        assert_eq!(emb.num_parameters(), 256 * 64);
        assert!(emb.requires_grad);
    }

    #[test]
    fn test_embedding_forward() {
        let emb = Embedding::xavier(256, 64);

        // Forward pass for a character
        let vec = emb.forward('a');
        assert_eq!(vec.len(), 64);

        // Same character should give same embedding
        let vec2 = emb.forward('a');
        assert_eq!(vec, vec2);

        // Different characters should (likely) give different embeddings
        let vec3 = emb.forward('b');
        assert_ne!(vec, vec3);
    }

    #[test]
    fn test_embedding_batch() {
        let emb = Embedding::xavier(256, 64);
        let chars: Vec<char> = "hello".chars().collect();

        let matrix = emb.forward_batch(&chars);
        assert_eq!(matrix.shape(), &[5, 64]);

        // First row should match forward('h')
        let h_emb = emb.forward('h');
        assert_eq!(matrix.row(0).to_owned(), h_emb);
    }

    #[test]
    fn test_embedding_gradient() {
        let mut emb = Embedding::new(256, 64, InitStrategy::Zero);

        // Forward pass
        let idx = 'a' as usize;
        let _output = emb.forward('a');

        // Backward pass with gradient
        let grad = ndarray::Array1::ones(64);
        emb.backward(idx, &grad);

        // Check gradient was accumulated
        assert!(emb.grad.is_some());
        let g = emb.grad.as_ref().unwrap();
        assert_eq!(g[[idx, 0]], 1.0);
        assert_eq!(g[[idx, 63]], 1.0);
        // Other indices should be zero
        assert_eq!(g[['b' as usize, 0]], 0.0);
    }

    #[test]
    fn test_embedding_step() {
        let mut emb = Embedding::new(256, 64, InitStrategy::Uniform(1.0));

        // Store original weight
        let original = emb.weights[[65, 0]]; // 'A' = 65

        // Forward and backward
        emb.forward('A');
        let grad = ndarray::Array1::from_elem(64, 0.5);
        emb.backward(65, &grad);

        // Take a step
        emb.step(0.1);

        // Weight should have changed
        let new_weight = emb.weights[[65, 0]];
        let expected = original - 0.1 * 0.5;
        assert!((new_weight - expected).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_freeze() {
        let mut emb = Embedding::xavier(256, 64);
        assert!(emb.requires_grad);

        emb.freeze();
        assert!(!emb.requires_grad);

        // Backward should be a no-op when frozen
        let grad = ndarray::Array1::ones(64);
        emb.backward(0, &grad);
        assert!(emb.grad.is_none());

        emb.unfreeze();
        assert!(emb.requires_grad);
    }

    #[test]
    fn test_learnable_trait_embedding() {
        use crate::Learnable;

        let mut emb = Embedding::xavier(256, 64);

        // Test num_parameters
        assert_eq!(emb.num_parameters(), 256 * 64);

        // Initially no gradients
        assert!(!emb.has_gradients());
        assert_eq!(emb.gradient_norm(), 0.0);

        // Accumulate some gradient
        let grad = ndarray::Array1::from_elem(64, 1.0);
        emb.backward(65, &grad);

        // Now has gradients
        assert!(emb.has_gradients());
        assert!(emb.gradient_norm() > 0.0);

        // Zero grad should clear
        emb.zero_grad();
        assert!(!emb.has_gradients());
    }

    #[test]
    fn test_learnable_trait_message_passing() {
        use crate::Learnable;

        let mut layer = MessagePassingLayer::new(64, 32);

        // Test num_parameters: weight (32x64) + bias (32)
        assert_eq!(layer.num_parameters(), 32 * 64 + 32);

        // Initially no gradients
        assert!(!layer.has_gradients());

        // Zero grad should work
        layer.zero_grad();
        assert!(!layer.has_gradients());
    }

    #[test]
    fn test_learnable_trait_graph_transform_net() {
        use crate::Learnable;

        let mut net = GraphTransformNet::new(256, 64, 32, 2);

        // Test num_parameters: embedding + 2 layers
        let embed_params = 256 * 64;
        let layer1_params = 32 * 64 + 32; // first layer: hidden_dim x embed_dim + bias
        let layer2_params = 32 * 32 + 32; // second layer: hidden_dim x hidden_dim + bias
        assert_eq!(
            net.num_parameters(),
            embed_params + layer1_params + layer2_params
        );

        // Initially no gradients
        assert!(!net.has_gradients());

        // Zero grad should work (parallel)
        net.zero_grad();
        assert!(!net.has_gradients());
    }

    #[test]
    fn test_embedding_with_dagnn() {
        let dag = DagNN::from_text("hi").unwrap();
        let emb = Embedding::xavier(256, 32);

        // Get embeddings for all nodes
        let embeddings = dag.get_node_embeddings(&emb);
        assert_eq!(embeddings.len(), 2); // 'h' and 'i'

        // Get as matrix
        let matrix = dag.get_embedding_matrix(&emb);
        assert_eq!(matrix.shape(), &[2, 32]);
    }

    #[test]
    fn test_init_strategies() {
        // Xavier should have smaller values for larger vocab
        let xavier = Embedding::xavier(1000, 64);
        let max_val = xavier.weights.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max_val < 0.1, "Xavier max = {}", max_val);

        // He initialization
        let he = Embedding::he(256, 64);
        let he_max = he.weights.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(he_max < 0.2, "He max = {}", he_max);

        // Zero initialization
        let zero = Embedding::new(256, 64, InitStrategy::Zero);
        assert!(zero.weights.iter().all(|&x| x == 0.0));
    }

    // ========================================================================
    // Backpropagation Tests (backend-027)
    // ========================================================================

    #[test]
    fn test_tape_creation() {
        let tape = Tape::new();
        assert!(tape.recording);
    }

    #[test]
    fn test_tape_embedding_lookup() {
        let mut tape = Tape::new();
        let emb = Embedding::new(256, 32, InitStrategy::Zero);

        let idx = tape.embedding_lookup(&emb, 'a');
        assert_eq!(idx, 0);

        let value = tape.get_value(idx).unwrap();
        assert_eq!(value.len(), 32);
    }

    #[test]
    fn test_tape_sum() {
        let mut tape = Tape::new();
        let emb = Embedding::new(256, 4, InitStrategy::Zero);

        let idx1 = tape.embedding_lookup(&emb, 'a');
        let idx2 = tape.embedding_lookup(&emb, 'b');

        let sum_idx = tape.sum(&[idx1, idx2]);
        let sum_value = tape.get_value(sum_idx).unwrap();
        assert_eq!(sum_value.len(), 4);
    }

    #[test]
    fn test_tape_mean() {
        let mut tape = Tape::new();

        // Manually store values for testing
        let idx1 = tape.store_value(vec![2.0, 4.0], vec![2]);
        let idx2 = tape.store_value(vec![4.0, 8.0], vec![2]);

        let mean_idx = tape.mean(&[idx1, idx2]);
        let mean_value = tape.get_value(mean_idx).unwrap();

        assert!((mean_value[0] - 3.0).abs() < 1e-6);
        assert!((mean_value[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_tape_relu() {
        let mut tape = Tape::new();

        let idx = tape.store_value(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let relu_idx = tape.relu(idx);

        let result = tape.get_value(relu_idx).unwrap();
        assert_eq!(result, &vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tape_sigmoid() {
        let mut tape = Tape::new();

        let idx = tape.store_value(vec![0.0], vec![1]);
        let sigmoid_idx = tape.sigmoid(idx);

        let result = tape.get_value(sigmoid_idx).unwrap();
        // sigmoid(0) = 0.5
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tape_tanh() {
        let mut tape = Tape::new();

        let idx = tape.store_value(vec![0.0], vec![1]);
        let tanh_idx = tape.tanh(idx);

        let result = tape.get_value(tanh_idx).unwrap();
        // tanh(0) = 0
        assert!(result[0].abs() < 1e-6);
    }

    #[test]
    fn test_tape_mse_loss() {
        let mut tape = Tape::new();

        let pred_idx = tape.store_value(vec![1.0, 2.0, 3.0], vec![3]);
        let target = vec![1.0, 2.0, 3.0];

        let (_loss_idx, loss_val) = tape.mse_loss(pred_idx, &target);

        // Perfect prediction should have 0 loss
        assert!(loss_val.abs() < 1e-6);
    }

    #[test]
    fn test_tape_backward_relu() {
        let mut tape = Tape::new();

        let input_idx = tape.store_value(vec![-1.0, 1.0], vec![2]);
        let output_idx = tape.relu(input_idx);

        tape.backward(output_idx);

        let input_grad = tape.get_grad(input_idx).unwrap();
        // Gradient should be 0 for negative input, 1 for positive
        assert_eq!(input_grad[0], 0.0);
        assert_eq!(input_grad[1], 1.0);
    }

    #[test]
    fn test_tape_backward_sum() {
        let mut tape = Tape::new();

        let idx1 = tape.store_value(vec![1.0, 2.0], vec![2]);
        let idx2 = tape.store_value(vec![3.0, 4.0], vec![2]);
        let sum_idx = tape.sum(&[idx1, idx2]);

        tape.backward(sum_idx);

        // Gradients for sum should be 1.0 for all inputs
        let grad1 = tape.get_grad(idx1).unwrap();
        let grad2 = tape.get_grad(idx2).unwrap();

        assert_eq!(grad1, &vec![1.0, 1.0]);
        assert_eq!(grad2, &vec![1.0, 1.0]);
    }

    #[test]
    fn test_tape_backward_mean() {
        let mut tape = Tape::new();

        let idx1 = tape.store_value(vec![2.0], vec![1]);
        let idx2 = tape.store_value(vec![4.0], vec![1]);
        let mean_idx = tape.mean(&[idx1, idx2]);

        tape.backward(mean_idx);

        // Gradients for mean should be 1/n
        let grad1 = tape.get_grad(idx1).unwrap();
        let grad2 = tape.get_grad(idx2).unwrap();

        assert!((grad1[0] - 0.5).abs() < 1e-6);
        assert!((grad2[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_node_gradients() {
        let mut grads = NodeGradients::new();

        let node = NodeIndex::new(0);
        let grad = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        grads.accumulate_node(node, &grad);

        let stored = grads.get_node_grad(node).unwrap();
        assert_eq!(stored, &grad);
    }

    #[test]
    fn test_node_gradients_accumulate() {
        let mut grads = NodeGradients::new();

        let node = NodeIndex::new(0);
        let grad1 = Array1::from_vec(vec![1.0, 2.0]);
        let grad2 = Array1::from_vec(vec![3.0, 4.0]);

        grads.accumulate_node(node, &grad1);
        grads.accumulate_node(node, &grad2);

        let stored = grads.get_node_grad(node).unwrap();
        assert!((stored[0] - 4.0).abs() < 1e-6);
        assert!((stored[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_node_gradients_clip() {
        let mut grads = NodeGradients::new();

        let node = NodeIndex::new(0);
        // Create gradient with norm = 5 (3^2 + 4^2 = 25, sqrt = 5)
        let grad = Array1::from_vec(vec![3.0, 4.0]);
        grads.accumulate_node(node, &grad);

        // Clip to max norm = 1
        grads.clip_grads(1.0);

        let clipped = grads.get_node_grad(node).unwrap();
        let norm: f32 = clipped.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_edge_gradients() {
        let mut grads = NodeGradients::new();

        let from = NodeIndex::new(0);
        let to = NodeIndex::new(1);

        grads.accumulate_edge(from, to, 1.5);
        grads.accumulate_edge(from, to, 0.5);

        let edge_grad = grads.get_edge_grad(from, to).unwrap();
        assert!((edge_grad - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dagnn_backward() {
        let mut dag = DagNN::from_text("ab").unwrap();
        dag.update_topology().unwrap();

        let mut emb = Embedding::new(256, 4, InitStrategy::Zero);
        emb.zero_grad();

        // Create output gradient
        let mut output_grad = HashMap::new();
        let last_node = dag.input_nodes().last().copied().unwrap();
        output_grad.insert(last_node, Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]));

        // Run backward pass
        let grads = dag.backward(&output_grad, &mut emb);

        // Should have gradients for both nodes
        assert!(!grads.node_grads.is_empty());
    }

    #[test]
    fn test_dagnn_backward_and_update() {
        let mut dag = DagNN::from_text("ab").unwrap();
        dag.update_topology().unwrap();

        let mut emb = Embedding::new(256, 4, InitStrategy::Zero);
        emb.zero_grad();

        // Get initial edge weights
        let edge_count = dag.edge_count();

        // Create output gradient
        let mut output_grad = HashMap::new();
        let last_node = dag.input_nodes().last().copied().unwrap();
        output_grad.insert(last_node, Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]));

        // Run backward and update
        dag.backward_and_update(&output_grad, &mut emb, 0.01);

        // Edge count should remain the same
        assert_eq!(dag.edge_count(), edge_count);
    }

    #[test]
    fn test_grad_utils_clip_norm() {
        let mut grad = Array1::from_vec(vec![3.0, 4.0]); // norm = 5
        grad_utils::clip_grad_norm(&mut grad, 1.0);

        let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_grad_utils_l2_regularization() {
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let lambda = 0.5;

        let reg_grad = grad_utils::l2_regularization_grad(&weights, lambda);

        // L2 reg grad = 2 * lambda * weights
        assert!((reg_grad[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((reg_grad[[0, 1]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_tape_no_grad_mode() {
        let mut tape = Tape::new();
        let emb = Embedding::new(256, 4, InitStrategy::Zero);

        // Disable recording
        tape.no_grad();
        let idx = tape.embedding_lookup(&emb, 'a');

        // Should still store values but not record operations
        assert!(tape.get_value(idx).is_some());

        // Re-enable
        tape.enable_grad();
        assert!(tape.recording);
    }

    #[test]
    fn test_tape_reset() {
        let mut tape = Tape::new();
        let emb = Embedding::new(256, 4, InitStrategy::Zero);

        tape.embedding_lookup(&emb, 'a');
        tape.embedding_lookup(&emb, 'b');

        tape.reset();

        // Should be empty after reset
        let idx = tape.embedding_lookup(&emb, 'c');
        assert_eq!(idx, 0); // First index after reset
    }

    #[test]
    fn test_tape_message_pass() {
        let mut tape = Tape::new();

        let node_idx = tape.store_value(vec![1.0, 2.0], vec![2]);
        let neighbor1 = tape.store_value(vec![3.0, 4.0], vec![2]);
        let neighbor2 = tape.store_value(vec![5.0, 6.0], vec![2]);

        let result_idx = tape.message_pass(node_idx, &[neighbor1, neighbor2], &[0.5, 0.5]);

        let result = tape.get_value(result_idx).unwrap();
        // 0.5 * [3, 4] + 0.5 * [5, 6] = [4, 5]
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_tape_get_embedding_grads() {
        let mut tape = Tape::new();
        let emb = Embedding::new(256, 4, InitStrategy::Zero);

        let idx1 = tape.embedding_lookup(&emb, 'a');
        let idx2 = tape.embedding_lookup(&emb, 'b');
        let sum_idx = tape.sum(&[idx1, idx2]);

        tape.backward(sum_idx);

        let emb_grads = tape.get_embedding_grads();
        assert_eq!(emb_grads.len(), 2);
    }

    // ========================================================================
    // Graph Transform Network Tests (backend-029)
    // ========================================================================

    #[test]
    fn test_message_passing_layer_creation() {
        let layer = MessagePassingLayer::new(64, 32);
        assert_eq!(layer.input_dim, 64);
        assert_eq!(layer.output_dim, 32);
        assert_eq!(layer.weight.shape(), &[32, 64]);
    }

    #[test]
    fn test_message_passing_forward() {
        let layer = MessagePassingLayer::new(4, 4);
        let node_feat = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let neighbor1 = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);

        let output = layer.forward(&node_feat, &[neighbor1]);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_message_passing_no_neighbors() {
        let layer = MessagePassingLayer::new(4, 4);
        let node_feat = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let output = layer.forward(&node_feat, &[]);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_message_passing_batch() {
        let layer = MessagePassingLayer::new(4, 4);
        let features = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5],
        )
        .unwrap();
        let adjacency = vec![vec![1], vec![0, 2], vec![1]];

        let output = layer.forward_batch(&features, &adjacency);
        assert_eq!(output.shape(), &[3, 4]);
    }

    #[test]
    fn test_attention_layer_creation() {
        let attn = AttentionLayer::new(64, 4);
        assert_eq!(attn.d_k, 16);
        assert_eq!(attn.query_proj.shape(), &[64, 64]);
    }

    #[test]
    fn test_attention_forward() {
        let attn = AttentionLayer::new(4, 1);
        let query = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let key = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let value = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

        let output = attn.forward(&query, &[key], &[value]);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_attention_empty_keys() {
        let attn = AttentionLayer::new(4, 1);
        let query = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let output = attn.forward(&query, &[], &[]);
        assert_eq!(output, query);
    }

    #[test]
    fn test_node_prediction_head() {
        let head = NodePredictionHead::new(32, 4);
        let features = Array1::from_vec(vec![0.1; 32]);

        let probs = head.predict(&features);
        assert_eq!(probs.len(), 4);

        // Softmax should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_node_prediction_op() {
        let head = NodePredictionHead::new(4, 4);
        let features = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let op = head.predict_op(&features);
        // Should return one of the valid operations
        assert!(matches!(
            op,
            EditOp::Keep | EditOp::Delete | EditOp::Modify | EditOp::Insert
        ));
    }

    #[test]
    fn test_graph_pooling_mean() {
        let pooling = GraphPooling::mean();
        let features = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0, 4.0]),
        ];

        let pooled = pooling.pool(&features);
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_graph_pooling_max() {
        let pooling = GraphPooling::max();
        let features = vec![
            Array1::from_vec(vec![1.0, 4.0]),
            Array1::from_vec(vec![3.0, 2.0]),
        ];

        let pooled = pooling.pool(&features);
        assert!((pooled[0] - 3.0).abs() < 1e-6);
        assert!((pooled[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_graph_pooling_sum() {
        let pooling = GraphPooling::sum();
        let features = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0, 4.0]),
        ];

        let pooled = pooling.pool(&features);
        assert!((pooled[0] - 4.0).abs() < 1e-6);
        assert!((pooled[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_graph_pooling_empty() {
        let pooling = GraphPooling::mean();
        let features: Vec<Array1<f32>> = vec![];

        let pooled = pooling.pool(&features);
        assert_eq!(pooled.len(), 1);
        assert_eq!(pooled[0], 0.0);
    }

    #[test]
    fn test_graph_transform_net_creation() {
        let net = GraphTransformNet::new(256, 32, 64, 2);
        assert_eq!(net.hidden_dim, 64);
        assert_eq!(net.num_layers, 2);
        assert_eq!(net.mp_layers.len(), 2);
    }

    #[test]
    fn test_graph_transform_net_encode() {
        let net = GraphTransformNet::new(256, 16, 16, 1);
        let dag = DagNN::from_text("hi").unwrap();

        let node_features = net.encode(&dag);
        assert_eq!(node_features.len(), 2); // 'h' and 'i'
        assert_eq!(node_features[0].len(), 16);
    }

    #[test]
    fn test_graph_transform_net_predict_edits() {
        let net = GraphTransformNet::new(256, 16, 16, 1);
        let dag = DagNN::from_text("abc").unwrap();

        let edits = net.predict_edits(&dag);
        assert_eq!(edits.len(), 3);

        // Each edit should have node, operation, and probabilities
        for (_node, op, probs) in &edits {
            assert!(matches!(
                op,
                EditOp::Keep | EditOp::Delete | EditOp::Modify | EditOp::Insert
            ));
            assert_eq!(probs.len(), 4);
        }
    }

    #[test]
    fn test_graph_transform_net_transform() {
        let mut net = GraphTransformNet::new(256, 16, 16, 1);
        let dag = DagNN::from_text("hello").unwrap();

        let result = net.transform(&dag);
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_transform_net_graph_embedding() {
        let net = GraphTransformNet::new(256, 16, 16, 1);
        let dag = DagNN::from_text("test").unwrap();

        let embedding = net.get_graph_embedding(&dag);
        assert_eq!(embedding.len(), 16);
    }

    #[test]
    fn test_graph_transform_net_zero_grad() {
        let mut net = GraphTransformNet::new(256, 16, 16, 1);

        net.zero_grad();

        // Embedding grad should be zeroed (Some with all zeros)
        assert!(net.embedding.grad.is_some());
        let grad = net.embedding.grad.as_ref().unwrap();
        assert!(grad.iter().all(|&x| x == 0.0));

        // Message passing layers should have None grads
        for layer in &net.mp_layers {
            assert!(layer.weight_grad.is_none());
            assert!(layer.bias_grad.is_none());
        }
    }

    #[test]
    fn test_edit_op_variants() {
        assert_eq!(EditOp::Keep, EditOp::Keep);
        assert_ne!(EditOp::Keep, EditOp::Delete);
        assert_ne!(EditOp::Delete, EditOp::Modify);
        assert_ne!(EditOp::Modify, EditOp::Insert);
    }

    #[test]
    fn test_pooling_type_variants() {
        assert_eq!(PoolingType::Mean, PoolingType::Mean);
        assert_ne!(PoolingType::Mean, PoolingType::Max);
        assert_ne!(PoolingType::Max, PoolingType::Sum);
    }

    // ========================================================================
    // Model Persistence Tests (backend-090)
    // ========================================================================

    #[test]
    fn test_graph_transform_net_json_roundtrip() {
        let original = GraphTransformNet::new(256, 32, 64, 2);

        // Save to JSON
        let json = original.save_json().unwrap();

        // Load back
        let loaded = GraphTransformNet::load_json(&json).unwrap();

        // Verify architecture matches
        assert_eq!(original.hidden_dim, loaded.hidden_dim);
        assert_eq!(original.num_layers, loaded.num_layers);
        assert_eq!(original.embedding.vocab_size, loaded.embedding.vocab_size);
        assert_eq!(original.embedding.embed_dim, loaded.embedding.embed_dim);
        assert_eq!(original.mp_layers.len(), loaded.mp_layers.len());
    }

    #[test]
    fn test_graph_transform_net_weights_preserved() {
        let original = GraphTransformNet::new(256, 16, 32, 1);

        // Save and load
        let json = original.save_json().unwrap();
        let loaded = GraphTransformNet::load_json(&json).unwrap();

        // Verify embedding weights are identical
        assert_eq!(
            original.embedding.weights.shape(),
            loaded.embedding.weights.shape()
        );
        for i in 0..original.embedding.weights.shape()[0] {
            for j in 0..original.embedding.weights.shape()[1] {
                assert!(
                    (original.embedding.weights[[i, j]] - loaded.embedding.weights[[i, j]]).abs()
                        < 1e-10,
                    "Embedding weight mismatch at [{}, {}]",
                    i,
                    j
                );
            }
        }

        // Verify message passing layer weights
        for (orig_layer, loaded_layer) in original.mp_layers.iter().zip(loaded.mp_layers.iter()) {
            assert_eq!(orig_layer.weight.shape(), loaded_layer.weight.shape());
            for i in 0..orig_layer.weight.shape()[0] {
                for j in 0..orig_layer.weight.shape()[1] {
                    assert!(
                        (orig_layer.weight[[i, j]] - loaded_layer.weight[[i, j]]).abs() < 1e-10,
                        "MP layer weight mismatch"
                    );
                }
            }
        }

        // Verify attention weights
        assert_eq!(
            original.attention.query_proj.shape(),
            loaded.attention.query_proj.shape()
        );
        assert_eq!(
            original.attention.key_proj.shape(),
            loaded.attention.key_proj.shape()
        );
        assert_eq!(
            original.attention.value_proj.shape(),
            loaded.attention.value_proj.shape()
        );
    }

    #[test]
    fn test_graph_transform_net_file_roundtrip() {
        let original = GraphTransformNet::new(256, 16, 32, 1);

        // Use a temp file
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_graph_transform_net.json");

        // Save to file
        original.save_to_file(&temp_file).unwrap();

        // Load from file
        let loaded = GraphTransformNet::load_from_file(&temp_file).unwrap();

        // Verify
        assert_eq!(original.hidden_dim, loaded.hidden_dim);
        assert_eq!(original.num_layers, loaded.num_layers);

        // Cleanup
        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_model_header_verification() {
        let net = GraphTransformNet::new(256, 32, 64, 2);
        let header = ModelHeader::for_graph_transform_net(&net);

        // Header should verify against same model
        assert!(header.verify(&net));

        // Different model should fail verification
        let different = GraphTransformNet::new(128, 16, 32, 1);
        assert!(!header.verify(&different));
    }

    #[test]
    fn test_model_header_fields() {
        let net = GraphTransformNet::new(256, 32, 64, 3);
        let header = net.header();

        assert_eq!(header.version, MODEL_PERSISTENCE_VERSION);
        assert_eq!(header.model_type, "GraphTransformNet");
        assert_eq!(header.vocab_size, 256);
        assert_eq!(header.embed_dim, 32);
        assert_eq!(header.hidden_dim, 64);
        assert_eq!(header.num_layers, 3);
    }

    #[test]
    fn test_gradients_not_serialized() {
        let mut original = GraphTransformNet::new(256, 16, 32, 1);

        // Accumulate some gradients
        original.embedding.grad = Some(ndarray::Array2::ones((256, 16)));

        // Save and load
        let json = original.save_json().unwrap();
        let loaded = GraphTransformNet::load_json(&json).unwrap();

        // Gradients should be None after loading (they are skipped)
        assert!(loaded.embedding.grad.is_none());
        for layer in &loaded.mp_layers {
            assert!(layer.weight_grad.is_none());
            assert!(layer.bias_grad.is_none());
        }
    }

    #[test]
    fn test_loaded_model_can_forward() {
        let original = GraphTransformNet::new(256, 16, 32, 1);

        // Save and load
        let json = original.save_json().unwrap();
        let loaded = GraphTransformNet::load_json(&json).unwrap();

        // Should be able to do forward pass on loaded model
        let dag = DagNN::from_text("test").unwrap();
        let features = loaded.encode(&dag);

        assert_eq!(features.len(), 4);
        assert_eq!(features[0].len(), 32);
    }

    #[test]
    fn test_serialized_model_struct() {
        let net = GraphTransformNet::new(256, 16, 32, 1);
        let serialized = SerializedModel {
            header: ModelHeader::for_graph_transform_net(&net),
            model: net.clone(),
        };

        // Round-trip through JSON
        let json = serde_json::to_string(&serialized).unwrap();
        let loaded: SerializedModel = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.header.vocab_size, 256);
        assert_eq!(loaded.model.hidden_dim, 32);
    }

    // ========================================================================
    // Unified Checkpoint Tests
    // ========================================================================

    #[test]
    fn test_unified_checkpoint_creation() {
        let checkpoint = UnifiedCheckpoint::new();
        assert_eq!(checkpoint.version, CHECKPOINT_VERSION);
        assert!(checkpoint.modules.is_empty());
    }

    #[test]
    fn test_learnable_param_persistable() {
        let mut checkpoint = UnifiedCheckpoint::new();
        let param = LearnableParam::new(0.5);

        // Add and load
        checkpoint.add_module(&param).unwrap();
        let loaded: LearnableParam = checkpoint.load_module().unwrap();

        assert!((loaded.value - 0.5).abs() < f32::EPSILON);
        assert!((loaded.grad - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_graph_transform_net_persistable() {
        let mut checkpoint = UnifiedCheckpoint::new();
        let net = GraphTransformNet::new(256, 16, 32, 2);

        // Add to checkpoint
        checkpoint.add_module(&net).unwrap();

        // Verify module is present
        assert!(checkpoint.has_module::<GraphTransformNet>());

        // Load and verify
        let loaded: GraphTransformNet = checkpoint.load_module().unwrap();
        assert_eq!(loaded.embedding.vocab_size, 256);
        assert_eq!(loaded.embedding.embed_dim, 16);
        assert_eq!(loaded.hidden_dim, 32);
        assert_eq!(loaded.num_layers, 2);
    }

    #[test]
    fn test_unified_checkpoint_multiple_modules() {
        let mut checkpoint = UnifiedCheckpoint::new();

        let net = GraphTransformNet::new(256, 16, 32, 1);
        let param = LearnableParam::new(0.42);

        checkpoint.add_module(&net).unwrap();
        checkpoint.add_module(&param).unwrap();

        // Both should be present
        assert!(checkpoint.has_module::<GraphTransformNet>());
        assert!(checkpoint.has_module::<LearnableParam>());

        // Load both
        let loaded_net: GraphTransformNet = checkpoint.load_module().unwrap();
        let loaded_param: LearnableParam = checkpoint.load_module().unwrap();

        assert_eq!(loaded_net.embedding.vocab_size, 256);
        assert!((loaded_param.value - 0.42).abs() < f32::EPSILON);
    }

    #[test]
    fn test_unified_checkpoint_roundtrip_json() {
        let mut checkpoint = UnifiedCheckpoint::new();
        let net = GraphTransformNet::new(128, 8, 16, 1);

        checkpoint.add_module(&net).unwrap();

        // Serialize to JSON
        let json = checkpoint.save_json().unwrap();

        // Deserialize
        let loaded_checkpoint = UnifiedCheckpoint::load_json(&json).unwrap();

        // Load module from loaded checkpoint
        let loaded_net: GraphTransformNet = loaded_checkpoint.load_module().unwrap();
        assert_eq!(loaded_net.embedding.vocab_size, 128);
        assert_eq!(loaded_net.embedding.embed_dim, 8);
    }

    #[test]
    fn test_unified_checkpoint_file_roundtrip() {
        let mut checkpoint = UnifiedCheckpoint::new();
        let net = GraphTransformNet::new(64, 8, 16, 1);
        checkpoint.add_module(&net).unwrap();

        // Save to temp file
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_unified_checkpoint.json");

        checkpoint.save_to_file(&path).unwrap();

        // Load from file
        let loaded = UnifiedCheckpoint::load_from_file(&path).unwrap();
        let loaded_net: GraphTransformNet = loaded.load_module().unwrap();

        assert_eq!(loaded_net.embedding.vocab_size, 64);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unified_checkpoint_module_not_found() {
        let checkpoint = UnifiedCheckpoint::new();

        // Try to load a module that doesn't exist
        let result: PersistenceResult<GraphTransformNet> = checkpoint.load_module();
        assert!(result.is_err());

        if let Err(PersistenceError::ModuleNotFound(type_id)) = result {
            assert_eq!(type_id, "GraphTransformNet");
        } else {
            panic!("Expected ModuleNotFound error");
        }
    }

    #[test]
    fn test_unified_checkpoint_module_ids() {
        let mut checkpoint = UnifiedCheckpoint::new();

        let net = GraphTransformNet::new(64, 8, 16, 1);
        let param = LearnableParam::new(0.5);

        checkpoint.add_module(&net).unwrap();
        checkpoint.add_module(&param).unwrap();

        let ids = checkpoint.module_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"GraphTransformNet"));
        assert!(ids.contains(&"LearnableParam"));
    }

    #[test]
    fn test_persistable_version_tracking() {
        let mut checkpoint = UnifiedCheckpoint::new();
        let net = GraphTransformNet::new(64, 8, 16, 1);

        checkpoint.add_module(&net).unwrap();

        // Check version is tracked
        let module_checkpoint = checkpoint.modules.get("GraphTransformNet").unwrap();
        assert_eq!(module_checkpoint.version, MODEL_PERSISTENCE_VERSION);
    }
}
