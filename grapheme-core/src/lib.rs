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
#[derive(Debug, Serialize, Deserialize)]
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
    pub fn form_clique(&mut self, members: Vec<NodeId>, label: Option<String>) -> CliqueResult<usize> {
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
        let mut nodes_by_pos: Vec<(usize, NodeId)> = self.input_nodes
            .iter()
            .filter_map(|&n| self.graph[n].position.map(|p| (p, n)))
            .collect();
        nodes_by_pos.sort_by_key(|&(p, _)| p);

        // Sliding window approach
        for i in 0..nodes_by_pos.len() {
            let (pos_i, node_i) = nodes_by_pos[i];

            // Look at nodes ahead within window
            for j in (i + 1)..nodes_by_pos.len() {
                let (pos_j, node_j) = nodes_by_pos[j];
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
#[derive(Debug, Serialize, Deserialize)]
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
        let neighbors: Vec<(usize, NodeId)> = self.position_index
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
            self.input_nodes.windows(3)
                .map(|w| w.to_vec())
                .collect()
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
        let member_indices: Vec<usize> = nodes.iter()
            .map(|n| n.index())
            .collect();

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
            let has_incoming = self.graph.edges_directed(node_idx, petgraph::Direction::Incoming).next().is_some();
            let has_outgoing = self.graph.edges_directed(node_idx, petgraph::Direction::Outgoing).next().is_some();

            // Skip input nodes (they're allowed to have no incoming edges)
            // Using HashSet for O(1) lookup instead of Vec::contains() O(n)
            if !has_incoming && !has_outgoing
                && !self.input_nodes_set.contains(&node_idx) {
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
            if self.input_nodes.len() < window_size {
                continue;
            }

            for window in self.input_nodes.windows(window_size) {
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
        let level_0: Vec<Pattern> = self.input_nodes.iter()
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
        let degrees: Vec<usize> = self.graph.node_indices()
            .map(|n| self.graph.neighbors(n).count())
            .collect();
        let max_degree = degrees.iter().max().copied().unwrap_or(0);
        let avg_degree = if node_count > 0 {
            degrees.iter().sum::<usize>() as f32 / node_count as f32
        } else {
            0.0
        };

        // Compute clique statistics
        let clique_sizes: Vec<usize> = self.cliques.iter()
            .map(|c| c.members.len())
            .collect();
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
        let sizes: Vec<usize> = self.cliques.iter()
            .map(|c| c.members.len())
            .collect();

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
    /// O(n^k) in the worst case, but much faster for sparse graphs using
    /// degeneracy ordering optimization.
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

        // For larger graphs, use degeneracy ordering
        Ok(self.find_cliques_degeneracy(k))
    }

    /// Simple O(n^k) clique enumeration for small graphs
    fn find_cliques_simple(&self, k: usize) -> Vec<Vec<NodeId>> {
        let nodes: Vec<NodeId> = self.graph.node_indices().collect();
        let n = nodes.len();

        if n < k {
            return Vec::new();
        }

        let mut cliques = Vec::new();

        // Generate all k-combinations of nodes
        for combo in Self::combinations(&nodes, k) {
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
            let later_neighbors: Vec<NodeId> = self.neighbors(v)
                .filter(|&u| {
                    position.get(&u).map(|&p| p > pos).unwrap_or(false)
                })
                .collect();

            // If not enough neighbors for a (k-1)-clique, skip
            if later_neighbors.len() < k - 1 {
                continue;
            }

            // Find all (k-1)-cliques among later neighbors
            for subset in Self::combinations(&later_neighbors, k - 1) {
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
        let mut remaining: Vec<NodeId> = self.graph.node_indices().collect();
        let mut ordering = Vec::with_capacity(remaining.len());

        while !remaining.is_empty() {
            // Find node with minimum degree among remaining
            let min_idx = remaining
                .iter()
                .enumerate()
                .min_by_key(|(_, &node)| {
                    self.neighbors(node)
                        .filter(|n| remaining.contains(n))
                        .count()
                })
                .map(|(i, _)| i)
                .unwrap();

            let node = remaining.swap_remove(min_idx);
            ordering.push(node);
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
        self.graph
            .edges(node)
            .map(|e| e.target())
            .chain(
                self.graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .map(|e| e.source())
            )
    }

    /// Generate all k-combinations of a slice
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
        serde_json::from_str(json)
            .map_err(|e| PersistenceError::Deserialization(e.to_string()))
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
use rand::Rng;

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

/// A learnable embedding layer that maps characters to dense vectors
///
/// This is the foundation for neural graph processing - each character
/// gets a learnable d-dimensional embedding vector.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Weight matrix: (vocab_size x embed_dim)
    /// Each row is the embedding for one character
    pub weights: Array2<f32>,
    /// Gradient accumulator (same shape as weights)
    pub grad: Option<Array2<f32>>,
    /// Whether to compute gradients
    pub requires_grad: bool,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Vocabulary size (typically 256 for ASCII or larger for Unicode)
    pub vocab_size: usize,
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
                Array2::from_shape_fn((vocab_size, embed_dim), |_| {
                    rng.gen_range(-scale..scale)
                })
            }
            InitStrategy::He => {
                let scale = (2.0 / vocab_size as f32).sqrt();
                Array2::from_shape_fn((vocab_size, embed_dim), |_| {
                    rng.gen_range(-scale..scale)
                })
            }
            InitStrategy::Uniform(scale) => {
                Array2::from_shape_fn((vocab_size, embed_dim), |_| {
                    rng.gen_range(-scale..scale)
                })
            }
            InitStrategy::Zero => {
                Array2::zeros((vocab_size, embed_dim))
            }
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
            NodeType::Compressed(_) => {
                self.forward_index(4)
            }
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
        let clique_id = dag.form_clique(members.clone(), Some("test_word".into())).unwrap();

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
        assert!(MAX_CLIQUE_GRAPH_SIZE > 0);
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
            vec![1],    // 0 -> 1
            vec![0],    // 1 -> 0
            vec![],     // 2 (isolated)
        ];

        let components = DagNN::find_clique_components(&adjacency, 3);

        assert_eq!(components.len(), 2);

        // Find which component contains 0 and 1
        let comp_01: Vec<usize> = components.iter()
            .find(|c| c.contains(&0))
            .cloned()
            .unwrap();
        assert!(comp_01.contains(&0));
        assert!(comp_01.contains(&1));
        assert!(!comp_01.contains(&2));

        // The other component should have just 2
        let comp_2: Vec<usize> = components.iter()
            .find(|c| c.contains(&2))
            .cloned()
            .unwrap();
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
        let dag = DagNN::from_text("the quick brown fox jumps over the lazy dog and keeps running").unwrap();
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
}
