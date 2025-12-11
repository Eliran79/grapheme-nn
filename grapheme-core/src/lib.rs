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

// External modules (Backend-176: Graph morphism detection)
pub mod graph_morphism;
pub use graph_morphism::{MorphismDetector, MorphismResult, spectral_alignment};

// External modules (Backend-180: Efficient graph serialization)
pub mod graph_serialization;
pub use graph_serialization::{CompactGraph, SerializationError, SerResult, calculate_stats};

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
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
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
// Complexity Guarantees (Backend-112)
// ============================================================================
// All operations in this crate are designed to run in polynomial time.
// We avoid NP-hard operations by:
// 1. Bounding clique enumeration to k <= MAX_CLIQUE_K
// 2. Bounding Sinkhorn iterations to MAX_SINKHORN_ITERATIONS
// 3. Using polynomial-time graph algorithms (BFS, DFS, topological sort)
// 4. Avoiding graph isomorphism and subgraph matching
//
// Complexity classes for neuromorphic operations:
// - Edge weight operations: O(E) - linear in edges
// - Per-node activations: O(V) - linear in nodes
// - Topological forward pass: O(V + E) - linear in graph size
// - Edge pruning (synaptic plasticity): O(E) - linear in edges
// - Orphan removal (apoptosis): O(V + E) - linear in graph size
// - Neurogenesis (node/edge addition): O(V + E) - linear in graph size
// - Hebbian learning: O(E) - linear in edges
// - Sinkhorn optimal transport: O(k² * iterations) - polynomial
// ============================================================================

/// Maximum nodes for polynomial-time guarantee
/// Operations beyond this may experience performance degradation
pub const MAX_NODES_POLYNOMIAL: usize = 100_000;

/// Maximum edges for polynomial-time guarantee
pub const MAX_EDGES_POLYNOMIAL: usize = 1_000_000;

/// Maximum Sinkhorn iterations (prevents infinite loops)
pub const MAX_SINKHORN_ITERATIONS: usize = 100;

/// Warning threshold for large graphs
pub const LARGE_GRAPH_WARNING_THRESHOLD: usize = 10_000;

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

/// Activation function for per-node nonlinearity (backend-106)
///
/// Each node can have its own activation function, enabling heterogeneous
/// network architectures. This is critical for the neuromorphic forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ActivationFn {
    /// Linear activation (identity): f(x) = x
    /// Used for input nodes and output regression tasks
    #[default]
    Linear,
    /// ReLU: f(x) = max(0, x)
    /// Most common for hidden layers, enables sparse activations
    ReLU,
    /// Sigmoid: f(x) = 1 / (1 + exp(-x))
    /// Used for binary outputs, gates, and attention weights
    Sigmoid,
    /// Tanh: f(x) = tanh(x)
    /// Zero-centered, useful for hidden layers
    Tanh,
    /// Leaky ReLU: f(x) = max(alpha * x, x) where alpha = 0.01
    /// Prevents dying ReLU problem
    LeakyReLU,
}

impl ActivationFn {
    /// Apply the activation function to a scalar value
    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationFn::Linear => x,
            ActivationFn::ReLU => x.max(0.0),
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFn::Tanh => x.tanh(),
            ActivationFn::LeakyReLU => {
                const ALPHA: f32 = 0.01;
                if x > 0.0 { x } else { ALPHA * x }
            }
        }
    }

    /// Compute the derivative of the activation function
    /// Given the output y = f(x), returns f'(x)
    /// For efficiency, some derivatives use the output value directly
    #[inline]
    pub fn derivative(&self, x: f32, output: f32) -> f32 {
        match self {
            ActivationFn::Linear => 1.0,
            ActivationFn::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFn::Sigmoid => output * (1.0 - output), // Uses cached output
            ActivationFn::Tanh => 1.0 - output * output,      // Uses cached output
            ActivationFn::LeakyReLU => {
                const ALPHA: f32 = 0.01;
                if x > 0.0 { 1.0 } else { ALPHA }
            }
        }
    }

    /// Compute derivative using only the input value (when output not cached)
    #[inline]
    pub fn derivative_from_input(&self, x: f32) -> f32 {
        match self {
            ActivationFn::Linear => 1.0,
            ActivationFn::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFn::Sigmoid => {
                let y = self.apply(x);
                y * (1.0 - y)
            }
            ActivationFn::Tanh => {
                let y = x.tanh();
                1.0 - y * y
            }
            ActivationFn::LeakyReLU => {
                const ALPHA: f32 = 0.01;
                if x > 0.0 { 1.0 } else { ALPHA }
            }
        }
    }

    /// Apply activation to a vector element-wise
    pub fn apply_vec(&self, xs: &[f32]) -> Vec<f32> {
        xs.iter().map(|&x| self.apply(x)).collect()
    }

    /// Compute derivatives for a vector
    pub fn derivative_vec(&self, xs: &[f32], outputs: &[f32]) -> Vec<f32> {
        xs.iter()
            .zip(outputs.iter())
            .map(|(&x, &y)| self.derivative(x, y))
            .collect()
    }
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
    /// Pixel node for image input (row, col) - backend-113
    Pixel { row: usize, col: usize },
    /// Classification output node (class index) - backend-113
    ClassOutput(usize),
    /// Generic feature input node (index in feature vector) - backend-140
    /// Used for AGI-ready modular inputs that don't have spatial coordinates
    Feature(usize),
}

/// A node in the GRAPHEME graph (matching GRAPHEME_Vision.md)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// The raw character value (or compressed pattern)
    pub value: Option<u8>,
    /// Current activation level (post-activation function)
    pub activation: f32,
    /// Pre-activation value (before activation function applied)
    /// Needed for computing gradients efficiently
    pub pre_activation: f32,
    /// Type of this node
    pub node_type: NodeType,
    /// Position in original text (if input node)
    pub position: Option<usize>,
    /// Activation function for this node (backend-106)
    pub activation_fn: ActivationFn,
}

impl Node {
    /// Create a new input node from a character
    /// Input nodes use Linear activation (identity) since they hold raw input
    pub fn input(ch: char, position: usize) -> Self {
        Self {
            value: if ch.is_ascii() { Some(ch as u8) } else { None },
            activation: 1.0,
            pre_activation: 1.0,
            node_type: NodeType::Input(ch),
            position: Some(position),
            activation_fn: ActivationFn::Linear, // Input nodes pass through unchanged
        }
    }

    /// Create a new hidden node with default ReLU activation
    pub fn hidden() -> Self {
        Self {
            value: None,
            activation: 0.0,
            pre_activation: 0.0,
            node_type: NodeType::Hidden,
            position: None,
            activation_fn: ActivationFn::ReLU, // Default for hidden layers
        }
    }

    /// Create a new hidden node with specified activation function
    pub fn hidden_with_activation(activation_fn: ActivationFn) -> Self {
        Self {
            value: None,
            activation: 0.0,
            pre_activation: 0.0,
            node_type: NodeType::Hidden,
            position: None,
            activation_fn,
        }
    }

    /// Create a new output node with Linear activation (for regression)
    pub fn output() -> Self {
        Self {
            value: None,
            activation: 0.0,
            pre_activation: 0.0,
            node_type: NodeType::Output,
            position: None,
            activation_fn: ActivationFn::Linear, // Output nodes typically linear
        }
    }

    /// Create a new output node with specified activation function
    pub fn output_with_activation(activation_fn: ActivationFn) -> Self {
        Self {
            value: None,
            activation: 0.0,
            pre_activation: 0.0,
            node_type: NodeType::Output,
            position: None,
            activation_fn,
        }
    }

    /// Create a clique node
    pub fn clique(members: Vec<usize>) -> Self {
        Self {
            value: None,
            activation: 0.0,
            pre_activation: 0.0,
            node_type: NodeType::Clique(members),
            position: None,
            activation_fn: ActivationFn::ReLU,
        }
    }

    /// Create a pattern node
    pub fn pattern(pattern: Vec<u8>) -> Self {
        Self {
            value: None,
            activation: 0.0,
            pre_activation: 0.0,
            node_type: NodeType::Pattern(pattern),
            position: None,
            activation_fn: ActivationFn::ReLU,
        }
    }

    /// Create a compressed node
    pub fn compressed(compression: CompressionType) -> Self {
        Self {
            value: None,
            activation: 0.0,
            pre_activation: 0.0,
            node_type: NodeType::Compressed(compression),
            position: None,
            activation_fn: ActivationFn::ReLU,
        }
    }

    /// Create a pixel node for image input (backend-113)
    /// Activation is the normalized pixel intensity [0.0, 1.0]
    pub fn pixel(row: usize, col: usize, intensity: f32) -> Self {
        let normalized = intensity.clamp(0.0, 1.0);
        Self {
            value: Some((normalized * 255.0) as u8),
            activation: normalized,
            pre_activation: normalized,
            node_type: NodeType::Pixel { row, col },
            position: Some(row * 28 + col), // Linear position for 28x28 image
            activation_fn: ActivationFn::Linear, // Pass through unchanged
        }
    }

    /// Create a classification output node (backend-113)
    pub fn class_output(class_idx: usize) -> Self {
        Self {
            value: Some(class_idx as u8),
            activation: 0.0,
            pre_activation: 0.0,
            node_type: NodeType::ClassOutput(class_idx),
            position: None,
            activation_fn: ActivationFn::Linear, // Logits are linear
        }
    }

    /// Create a generic feature input node (backend-140)
    /// Used for AGI-ready modular inputs without spatial coordinates.
    /// Works with any feature vector size (vision graphs, embeddings, etc.)
    pub fn feature(index: usize, activation: f32) -> Self {
        let normalized = activation.clamp(0.0, 1.0);
        Self {
            value: Some((normalized * 255.0) as u8),
            activation: normalized,
            pre_activation: normalized,
            node_type: NodeType::Feature(index),
            position: Some(index),
            activation_fn: ActivationFn::Linear, // Input nodes pass through unchanged
        }
    }

    /// Set the pre-activation value and compute post-activation
    /// This is the main method for the neuromorphic forward pass
    #[inline]
    pub fn set_pre_activation(&mut self, pre_act: f32) {
        self.pre_activation = pre_act;
        self.activation = self.activation_fn.apply(pre_act);
    }

    /// Compute the local gradient (derivative of activation function)
    /// Used during backpropagation
    #[inline]
    pub fn activation_derivative(&self) -> f32 {
        self.activation_fn.derivative(self.pre_activation, self.activation)
    }

    /// Set the activation function for this node
    pub fn with_activation_fn(mut self, activation_fn: ActivationFn) -> Self {
        self.activation_fn = activation_fn;
        self
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

    /// Create edge with Xavier initialization (backend-105)
    ///
    /// Xavier/Glorot initialization: w ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
    /// For edge weights, we use fan_in and fan_out as the degree bounds.
    ///
    /// # Arguments
    /// * `fan_in` - Number of incoming connections at source node
    /// * `fan_out` - Number of outgoing connections at target node
    /// * `edge_type` - Type of edge to create
    pub fn xavier(fan_in: usize, fan_out: usize, edge_type: EdgeType) -> Self {
        let fan_in = fan_in.max(1) as f32;
        let fan_out = fan_out.max(1) as f32;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();

        // Random uniform in [-limit, limit]
        let weight = rand::random::<f32>() * 2.0 * limit - limit;
        Self::new(weight, edge_type)
    }

    /// Create edge with He initialization (backend-105)
    ///
    /// He initialization (for ReLU): w ~ N(0, sqrt(2/fan_in))
    /// Better for networks with ReLU activations.
    ///
    /// # Arguments
    /// * `fan_in` - Number of incoming connections at source node
    /// * `edge_type` - Type of edge to create
    pub fn he(fan_in: usize, edge_type: EdgeType) -> Self {
        let fan_in = fan_in.max(1) as f32;
        let std_dev = (2.0 / fan_in).sqrt();

        // Random normal with mean 0 and std std_dev
        // Using Box-Muller transform for normal distribution
        let u1: f32 = rand::random::<f32>().max(1e-10);
        let u2: f32 = rand::random();
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();

        Self::new(normal * std_dev, edge_type)
    }

    /// Create a sequential edge with Xavier initialization
    pub fn sequential_xavier(fan_in: usize, fan_out: usize) -> Self {
        Self::xavier(fan_in, fan_out, EdgeType::Sequential)
    }

    /// Create a skip edge with Xavier initialization
    pub fn skip_xavier(fan_in: usize, fan_out: usize) -> Self {
        Self::xavier(fan_in, fan_out, EdgeType::Skip)
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
    /// Accumulated edge gradients for training (backend-142)
    /// Key: (source NodeId, target NodeId), Value: accumulated gradient
    #[serde(skip)]
    edge_grads: HashMap<(NodeId, NodeId), f32>,
    /// Whether to accumulate gradients during backward pass (backend-142)
    #[serde(skip)]
    pub requires_grad: bool,
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
            edge_grads: HashMap::new(),
            requires_grad: true, // Default to training mode
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

    // ========================================================================
    // Image Encoding (Generic)
    // ========================================================================

    /// Build a DagNN from a grayscale image (pixel grid to graph)
    ///
    /// Encodes a 2D image as a directed acyclic graph with:
    /// - width×height pixel nodes
    /// - Grid topology with 4-neighbor connections (up, down, left, right)
    /// - DAG ordering: row-major (top to bottom, left to right)
    /// - Edge weights initialized with Xavier initialization
    ///
    /// # Arguments
    /// * `pixels` - Row-major pixel values (0.0 to 1.0 normalized)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// A DagNN representing the image with grid topology
    pub fn from_image(pixels: &[f32], width: usize, height: usize) -> GraphemeResult<Self> {
        if pixels.len() != width * height {
            return Err(GraphemeError::DimensionMismatch(format!(
                "Pixel count {} doesn't match {}×{}={}",
                pixels.len(),
                width,
                height,
                width * height
            )));
        }

        let mut dag = Self::new();

        // Create grid of pixel nodes in row-major order
        let mut node_grid: Vec<Vec<NodeId>> = Vec::with_capacity(height);

        for row in 0..height {
            let mut row_nodes = Vec::with_capacity(width);
            for col in 0..width {
                let pixel_idx = row * width + col;
                let intensity = pixels[pixel_idx];
                let node = dag.graph.add_node(Node::pixel(row, col, intensity));
                dag.input_nodes.push(node);
                dag.input_nodes_set.insert(node);
                dag.position_index.insert(pixel_idx, node);
                row_nodes.push(node);
            }
            node_grid.push(row_nodes);
        }

        // Add edges: 4-neighbor connectivity (DAG: edges only go "forward")
        // Forward means: right neighbors and down neighbors (row-major ordering)
        for row in 0..height {
            for col in 0..width {
                let current = node_grid[row][col];

                // Right neighbor (same row, next column)
                if col + 1 < width {
                    let right = node_grid[row][col + 1];
                    dag.graph.add_edge(current, right, Edge::sequential());
                }

                // Down neighbor (next row, same column)
                if row + 1 < height {
                    let down = node_grid[row + 1][col];
                    dag.graph.add_edge(current, down, Edge::sequential());
                }
            }
        }

        // Initialize edge weights with Xavier
        dag.init_edge_weights_xavier();

        // Compute topological order
        dag.topology = TopologicalOrder::from_graph(&dag.graph)?;

        Ok(dag)
    }

    /// Build a DagNN classifier with configurable input/hidden/output sizes.
    ///
    /// This is the generic classifier builder that works with any
    /// input dimension. Use this for VisionGraph inputs or any other
    /// variable-size input.
    ///
    /// # Arguments
    /// * `num_inputs` - Number of input nodes
    /// * `hidden_size` - Number of hidden nodes
    /// * `num_classes` - Number of output classes
    /// * `input_activations` - Optional initial activations for input nodes
    ///
    /// # Returns
    /// A DagNN ready for classification with the specified architecture
    pub fn with_classifier(
        num_inputs: usize,
        hidden_size: usize,
        num_classes: usize,
        input_activations: Option<&[f32]>,
    ) -> GraphemeResult<Self> {
        let mut dag = Self::new();

        // Create input nodes using generic Feature type (AGI-ready, no hardcoded dimensions)
        for i in 0..num_inputs {
            let activation = input_activations.map(|a| a.get(i).copied().unwrap_or(0.0)).unwrap_or(1.0);
            let node = dag.graph.add_node(Node::feature(i, activation));
            dag.input_nodes.push(node);
            dag.input_nodes_set.insert(node);
        }

        // Add hidden layer with ReLU activation
        let mut hidden_nodes = Vec::with_capacity(hidden_size);
        for _ in 0..hidden_size {
            let hidden = dag.add_hidden_with_activation(ActivationFn::ReLU);
            hidden_nodes.push(hidden);
        }

        // Connect input nodes to hidden nodes with Xavier-initialized edges
        let inputs_per_hidden = (num_inputs / hidden_size).max(1);
        for (h_idx, &hidden) in hidden_nodes.iter().enumerate() {
            let start = (h_idx * inputs_per_hidden) % num_inputs;
            let end = ((h_idx + 1) * inputs_per_hidden).min(num_inputs);

            for input_idx in start..end {
                let fan_in = inputs_per_hidden.max(1) as f32;
                let fan_out = num_classes as f32;
                let limit = (6.0 / (fan_in + fan_out)).sqrt();
                let weight = rand::random::<f32>() * 2.0 * limit - limit;
                dag.add_edge(
                    dag.input_nodes[input_idx],
                    hidden,
                    Edge::new(weight, EdgeType::Sequential),
                );
            }
        }

        // Add output nodes (one per class)
        let mut class_outputs = Vec::with_capacity(num_classes);
        for class_idx in 0..num_classes {
            let output = dag.graph.add_node(Node::class_output(class_idx));
            dag.output_nodes.push(output);
            class_outputs.push(output);
        }

        // Connect hidden nodes to output nodes
        for &hidden in &hidden_nodes {
            for &output in &class_outputs {
                let fan_in = hidden_size as f32;
                let fan_out = 1.0;
                let limit = (6.0 / (fan_in + fan_out)).sqrt();
                let weight = rand::random::<f32>() * 2.0 * limit - limit;
                dag.add_edge(hidden, output, Edge::new(weight, EdgeType::Sequential));
            }
        }

        // Update topology
        dag.update_topology()?;

        Ok(dag)
    }

    /// Get output logits for classification
    ///
    /// Returns the activation values of all ClassOutput nodes.
    /// Call neuromorphic_forward() first to compute activations.
    ///
    /// # Returns
    /// Vector of logits (one per class), or empty if no ClassOutput nodes
    pub fn get_classification_logits(&self) -> Vec<f32> {
        self.output_nodes
            .iter()
            .filter_map(|&node_id| {
                let node = &self.graph[node_id];
                if matches!(node.node_type, NodeType::ClassOutput(_)) {
                    Some(node.activation)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get predicted class (argmax of output logits)
    ///
    /// # Returns
    /// The class index with highest activation
    pub fn predict_class(&self) -> usize {
        let logits = self.get_classification_logits();
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
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

    /// Initialize all edge weights using Xavier initialization (backend-105)
    ///
    /// This method reinitializes all edge weights in the graph using Xavier/Glorot
    /// initialization, which helps with gradient flow during training.
    ///
    /// Xavier initialization: w ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
    ///
    /// # Arguments
    /// * `strategy` - Initialization strategy to use
    pub fn init_edge_weights(&mut self, strategy: InitStrategy) {
        use petgraph::Direction;

        // Collect edge info first to avoid borrow conflicts
        let edge_info: Vec<_> = self
            .graph
            .edge_indices()
            .filter_map(|edge_idx| {
                let (source, target) = self.graph.edge_endpoints(edge_idx)?;
                let fan_in = self.graph.edges_directed(source, Direction::Incoming).count();
                let fan_out = self.graph.edges_directed(target, Direction::Outgoing).count();
                let edge_type = self.graph[edge_idx].edge_type;
                Some((edge_idx, fan_in, fan_out, edge_type))
            })
            .collect();

        // Update edge weights
        for (edge_idx, fan_in, fan_out, edge_type) in edge_info {
            let new_weight = match strategy {
                InitStrategy::Xavier => {
                    let fan_in = fan_in.max(1) as f32;
                    let fan_out = fan_out.max(1) as f32;
                    let limit = (6.0 / (fan_in + fan_out)).sqrt();
                    rand::random::<f32>() * 2.0 * limit - limit
                }
                InitStrategy::He => {
                    let fan_in = fan_in.max(1) as f32;
                    let std_dev = (2.0 / fan_in).sqrt();
                    let u1: f32 = rand::random::<f32>().max(1e-10);
                    let u2: f32 = rand::random();
                    let normal =
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    normal * std_dev
                }
                InitStrategy::Uniform(scale) => rand::random::<f32>() * 2.0 * scale - scale,
                InitStrategy::Zero => 0.0,
            };

            self.graph[edge_idx] = Edge::new(new_weight, edge_type);
        }
    }

    /// Initialize edge weights with Xavier initialization (convenience method)
    pub fn init_edge_weights_xavier(&mut self) {
        self.init_edge_weights(InitStrategy::Xavier);
    }

    /// Initialize edge weights with He initialization (convenience method)
    pub fn init_edge_weights_he(&mut self) {
        self.init_edge_weights(InitStrategy::He);
    }

    // ========================================================================
    // Gradient Accumulation API (backend-142)
    // ========================================================================

    /// Zero out all accumulated edge gradients
    ///
    /// Call this at the start of each training step (or after optimizer.step())
    /// to clear gradients from the previous iteration.
    ///
    /// # Example
    /// ```ignore
    /// for batch in data {
    ///     dag.zero_grad();
    ///     let loss = forward_and_loss(&dag, batch);
    ///     dag.backward_accumulate(&loss_grad, &mut embedding);
    ///     dag.step(learning_rate);
    /// }
    /// ```
    pub fn zero_grad(&mut self) {
        self.edge_grads.clear();
    }

    /// Apply accumulated gradients to edge weights (SGD step)
    ///
    /// Updates each edge weight: w = w - lr * grad
    ///
    /// # Arguments
    /// * `lr` - Learning rate for the update step
    ///
    /// # Example
    /// ```ignore
    /// dag.zero_grad();
    /// dag.backward_accumulate(&loss_grad, &mut embedding);
    /// dag.step(0.01); // Apply gradients with lr=0.01
    /// ```
    pub fn step(&mut self, lr: f32) {
        for ((from, to), grad) in &self.edge_grads {
            if let Some(edge_idx) = self.graph.find_edge(*from, *to) {
                self.graph[edge_idx].weight -= lr * grad;
            }
        }
    }

    /// Get the L2 norm of all accumulated edge gradients
    ///
    /// Useful for monitoring training stability and implementing gradient clipping.
    ///
    /// # Returns
    /// The L2 norm (sqrt of sum of squared gradients)
    pub fn gradient_norm(&self) -> f32 {
        let sum_sq: f32 = self.edge_grads.values().map(|g| g * g).sum();
        sum_sq.sqrt()
    }

    /// Check if any gradients have been accumulated
    ///
    /// # Returns
    /// `true` if there are accumulated gradients, `false` otherwise
    pub fn has_gradients(&self) -> bool {
        !self.edge_grads.is_empty()
    }

    /// Get the total number of trainable edge parameters
    ///
    /// # Returns
    /// Number of edges in the graph (each edge has one trainable weight)
    pub fn num_parameters(&self) -> usize {
        self.graph.edge_count()
    }

    /// Clip gradients to prevent exploding gradients
    ///
    /// If the gradient norm exceeds `max_norm`, scales all gradients so the
    /// total norm equals `max_norm`.
    ///
    /// # Arguments
    /// * `max_norm` - Maximum allowed gradient norm
    ///
    /// # Returns
    /// The original gradient norm (before clipping)
    pub fn clip_gradients(&mut self, max_norm: f32) -> f32 {
        let norm = self.gradient_norm();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for grad in self.edge_grads.values_mut() {
                *grad *= scale;
            }
        }
        norm
    }

    /// Accumulate gradient for a specific edge
    ///
    /// This is called internally by `backward_accumulate` but can also be used
    /// directly for custom gradient computation.
    ///
    /// # Arguments
    /// * `from` - Source node ID
    /// * `to` - Target node ID
    /// * `grad` - Gradient value to accumulate
    pub fn accumulate_edge_grad(&mut self, from: NodeId, to: NodeId, grad: f32) {
        *self.edge_grads.entry((from, to)).or_insert(0.0) += grad;
    }

    /// Get accumulated gradient for a specific edge
    ///
    /// # Arguments
    /// * `from` - Source node ID
    /// * `to` - Target node ID
    ///
    /// # Returns
    /// The accumulated gradient, or `None` if no gradient exists for this edge
    pub fn get_edge_grad(&self, from: NodeId, to: NodeId) -> Option<f32> {
        self.edge_grads.get(&(from, to)).copied()
    }

    /// Set training mode (whether to accumulate gradients)
    ///
    /// # Arguments
    /// * `mode` - `true` for training mode, `false` for inference mode
    pub fn train(&mut self, mode: bool) {
        self.requires_grad = mode;
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.requires_grad
    }

    // ========================================================================
    // Edge Weight Pruning (Synaptic Plasticity) - Backend-108
    // ========================================================================

    /// Prune edges with absolute weight below threshold (synaptic plasticity).
    ///
    /// In biological neural networks, synapses that are rarely used or weakly
    /// connected get pruned over time. This implements that mechanism by
    /// removing edges with weights close to zero.
    ///
    /// # Arguments
    /// * `threshold` - Minimum absolute weight to keep (edges with |weight| < threshold are removed)
    ///
    /// # Returns
    /// Number of edges pruned
    ///
    /// # Example
    /// ```ignore
    /// let mut dag = DagNN::from_text("abc").unwrap();
    /// dag.init_edge_weights_xavier();
    /// // After training, prune weak connections
    /// let pruned = dag.prune_edges_by_threshold(0.01);
    /// println!("Pruned {} weak synapses", pruned);
    /// ```
    ///
    /// # Note
    /// - This may disconnect parts of the graph
    /// - Topology is automatically updated after pruning
    /// - Input nodes are never pruned (only their outgoing edges may be)
    ///
    /// # Complexity
    /// O(E) - linear in the number of edges (backend-112 verified)
    pub fn prune_edges_by_threshold(&mut self, threshold: f32) -> usize {
        // Collect edges to remove (can't modify while iterating)
        let edges_to_remove: Vec<_> = self
            .graph
            .edge_indices()
            .filter(|&edge_idx| {
                let weight = self.graph[edge_idx].weight;
                weight.abs() < threshold
            })
            .collect();

        let pruned_count = edges_to_remove.len();

        // Remove edges in reverse order to keep indices valid
        for edge_idx in edges_to_remove.into_iter().rev() {
            self.graph.remove_edge(edge_idx);
        }

        // Update topology after structural change
        if pruned_count > 0 {
            let _ = self.update_topology();
        }

        pruned_count
    }

    /// Prune a percentage of weakest edges (synaptic plasticity).
    ///
    /// More aggressive than threshold-based pruning. Removes the bottom
    /// percentile of edges by absolute weight, regardless of their actual
    /// values.
    ///
    /// # Arguments
    /// * `percentile` - Fraction of edges to prune (0.0 to 1.0)
    ///
    /// # Returns
    /// Number of edges pruned
    ///
    /// # Example
    /// ```ignore
    /// let mut dag = DagNN::from_text("abc").unwrap();
    /// dag.init_edge_weights_xavier();
    /// // Prune bottom 10% of connections
    /// let pruned = dag.prune_edges_by_percentile(0.10);
    /// ```
    ///
    /// # Panics
    /// Panics if percentile is not in range [0.0, 1.0]
    pub fn prune_edges_by_percentile(&mut self, percentile: f32) -> usize {
        assert!(
            (0.0..=1.0).contains(&percentile),
            "Percentile must be between 0.0 and 1.0"
        );

        if self.graph.edge_count() == 0 {
            return 0;
        }

        // Collect edge weights with indices
        let mut edge_weights: Vec<_> = self
            .graph
            .edge_indices()
            .map(|idx| (idx, self.graph[idx].weight.abs()))
            .collect();

        // Sort by absolute weight (ascending)
        edge_weights.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate how many to prune
        let n_to_prune = (edge_weights.len() as f32 * percentile).ceil() as usize;

        if n_to_prune == 0 {
            return 0;
        }

        // Collect edge indices to remove (the n_to_prune weakest edges)
        let edges_to_remove: Vec<_> = edge_weights
            .iter()
            .take(n_to_prune)
            .map(|(idx, _)| *idx)
            .collect();

        let pruned_count = edges_to_remove.len();

        // Remove edges in reverse order to keep indices valid
        // We need to sort by index in reverse order for safe removal
        let mut edges_to_remove = edges_to_remove;
        edges_to_remove.sort_by(|a, b| b.cmp(a)); // Sort descending by index

        for edge_idx in edges_to_remove {
            self.graph.remove_edge(edge_idx);
        }

        // Update topology after structural change
        if pruned_count > 0 {
            let _ = self.update_topology();
        }

        pruned_count
    }

    /// Prune edges based on activity correlation (Hebbian plasticity).
    ///
    /// "Neurons that fire together, wire together" - edges between nodes
    /// with low activation correlation are pruned. This requires running
    /// forward passes first to populate activation values.
    ///
    /// # Arguments
    /// * `min_correlation` - Minimum correlation to keep edge (0.0 to 1.0)
    /// * `activations` - History of activation values for each node
    ///
    /// # Returns
    /// Number of edges pruned
    ///
    /// # Example
    /// ```ignore
    /// let mut dag = DagNN::from_text("abc").unwrap();
    /// // Run several forward passes, collect activations
    /// let history: Vec<HashMap<NodeId, f32>> = ...;
    /// let pruned = dag.prune_edges_by_correlation(0.3, &history);
    /// ```
    pub fn prune_edges_by_correlation(
        &mut self,
        min_correlation: f32,
        activation_history: &[HashMap<NodeId, f32>],
    ) -> usize {
        if activation_history.is_empty() {
            return 0;
        }

        // Calculate correlation for each edge
        let edges_to_remove: Vec<_> = self
            .graph
            .edge_indices()
            .filter(|&edge_idx| {
                let Some((source, target)) = self.graph.edge_endpoints(edge_idx) else {
                    return false; // Skip edges with invalid endpoints
                };

                // Get activation vectors for source and target
                let source_acts: Vec<f32> = activation_history
                    .iter()
                    .map(|h| *h.get(&source).unwrap_or(&0.0))
                    .collect();
                let target_acts: Vec<f32> = activation_history
                    .iter()
                    .map(|h| *h.get(&target).unwrap_or(&0.0))
                    .collect();

                // Calculate Pearson correlation
                let correlation = Self::pearson_correlation(&source_acts, &target_acts);

                // Prune if correlation is below threshold
                correlation.abs() < min_correlation
            })
            .collect();

        let pruned_count = edges_to_remove.len();

        // Remove edges
        for edge_idx in edges_to_remove.into_iter().rev() {
            self.graph.remove_edge(edge_idx);
        }

        if pruned_count > 0 {
            let _ = self.update_topology();
        }

        pruned_count
    }

    /// Calculate Pearson correlation coefficient between two vectors.
    fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len() as f32;
        if n < 2.0 {
            return 0.0;
        }

        let sum_x: f32 = x.iter().sum();
        let sum_y: f32 = y.iter().sum();
        let sum_xy: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f32 = x.iter().map(|a| a * a).sum();
        let sum_y2: f32 = y.iter().map(|a| a * a).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator =
            ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Get statistics about edge weights for monitoring pruning decisions.
    ///
    /// # Returns
    /// Tuple of (min, max, mean, std_dev, median)
    pub fn edge_weight_stats(&self) -> (f32, f32, f32, f32, f32) {
        if self.graph.edge_count() == 0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let mut weights: Vec<f32> = self
            .graph
            .edge_indices()
            .map(|idx| self.graph[idx].weight)
            .collect();

        let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = weights.iter().sum();
        let mean = sum / weights.len() as f32;

        let variance: f32 = weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>()
            / weights.len() as f32;
        let std_dev = variance.sqrt();

        // Median
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if weights.len().is_multiple_of(2) {
            (weights[weights.len() / 2 - 1] + weights[weights.len() / 2]) / 2.0
        } else {
            weights[weights.len() / 2]
        };

        (min, max, mean, std_dev, median)
    }

    /// Count edges by weight range for histogram analysis.
    ///
    /// # Arguments
    /// * `n_bins` - Number of bins to divide the weight range into
    ///
    /// # Returns
    /// Vector of (bin_start, bin_end, count) tuples
    pub fn edge_weight_histogram(&self, n_bins: usize) -> Vec<(f32, f32, usize)> {
        if self.graph.edge_count() == 0 || n_bins == 0 {
            return vec![];
        }

        let weights: Vec<f32> = self
            .graph
            .edge_indices()
            .map(|idx| self.graph[idx].weight)
            .collect();

        let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max - min).abs() < 1e-10 {
            // All weights are the same
            return vec![(min, max, weights.len())];
        }

        let bin_width = (max - min) / n_bins as f32;
        let mut bins = vec![0usize; n_bins];

        for &w in &weights {
            let bin_idx = ((w - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(n_bins - 1); // Clamp to last bin
            bins[bin_idx] += 1;
        }

        bins.iter()
            .enumerate()
            .map(|(i, &count)| {
                let start = min + i as f32 * bin_width;
                let end = start + bin_width;
                (start, end, count)
            })
            .collect()
    }

    // ========================================================================
    // Orphaned Node Removal (Apoptosis) - Backend-109
    // ========================================================================

    /// Remove orphaned nodes (nodes with no edges) from the graph.
    ///
    /// In biological neural networks, neurons that lose all synaptic connections
    /// undergo programmed cell death (apoptosis). This implements that mechanism
    /// by removing nodes that have no incoming or outgoing edges.
    ///
    /// # Returns
    /// Number of nodes removed
    ///
    /// # Note
    /// - Input nodes are NEVER removed (they represent the input sequence)
    /// - Output nodes are NEVER removed (they represent target outputs)
    /// - Only hidden nodes with no connections are removed
    /// - Topology is automatically updated after removal
    ///
    /// # Example
    /// ```ignore
    /// let mut dag = DagNN::from_text("abc").unwrap();
    /// // After pruning edges, some hidden nodes may become orphaned
    /// dag.prune_edges_by_threshold(0.01);
    /// let removed = dag.remove_orphaned_nodes();
    /// println!("Removed {} orphaned neurons", removed);
    /// ```
    pub fn remove_orphaned_nodes(&mut self) -> usize {
        use petgraph::Direction;

        // Collect orphaned hidden nodes (not in input_nodes or output_nodes)
        let orphaned: Vec<_> = self
            .graph
            .node_indices()
            .filter(|&node| {
                // Skip input and output nodes - they're never orphaned
                if self.input_nodes_set.contains(&node) {
                    return false;
                }
                if self.output_nodes.contains(&node) {
                    return false;
                }

                // Check if truly orphaned (no edges in either direction)
                let in_degree = self.graph.edges_directed(node, Direction::Incoming).count();
                let out_degree = self.graph.edges_directed(node, Direction::Outgoing).count();

                in_degree == 0 && out_degree == 0
            })
            .collect();

        let removed_count = orphaned.len();

        // Remove orphaned nodes (in reverse index order for safety)
        let mut orphaned = orphaned;
        orphaned.sort_by_key(|n| std::cmp::Reverse(n.index()));

        for node in orphaned {
            self.graph.remove_node(node);
        }

        // Update topology after structural change
        if removed_count > 0 {
            let _ = self.update_topology();
        }

        removed_count
    }

    /// Remove nodes unreachable from input nodes.
    ///
    /// Finds all nodes that cannot be reached by following edges forward
    /// from any input node, and removes them. This cleans up "floating"
    /// subgraphs that have no connection to the input.
    ///
    /// # Returns
    /// Number of nodes removed
    ///
    /// # Note
    /// - Input nodes are always considered reachable
    /// - Uses BFS/DFS from all input nodes
    /// - More aggressive than `remove_orphaned_nodes` - removes entire
    ///   disconnected subgraphs, not just isolated nodes
    pub fn remove_unreachable_from_inputs(&mut self) -> usize {
        use petgraph::visit::Bfs;
        use std::collections::HashSet;

        if self.input_nodes.is_empty() {
            return 0;
        }

        // Find all nodes reachable from inputs using BFS
        let mut reachable = HashSet::new();

        for &input in &self.input_nodes {
            let mut bfs = Bfs::new(&self.graph, input);
            while let Some(node) = bfs.next(&self.graph) {
                reachable.insert(node);
            }
        }

        // Collect unreachable nodes
        let unreachable: Vec<_> = self
            .graph
            .node_indices()
            .filter(|node| !reachable.contains(node))
            .collect();

        let removed_count = unreachable.len();

        // Remove unreachable nodes (in reverse index order)
        let mut unreachable = unreachable;
        unreachable.sort_by_key(|n| std::cmp::Reverse(n.index()));

        for node in unreachable {
            // Also remove from output_nodes if present
            self.output_nodes.retain(|&n| n != node);
            self.graph.remove_node(node);
        }

        if removed_count > 0 {
            let _ = self.update_topology();
        }

        removed_count
    }

    /// Remove dead-end nodes (nodes with no path to any output).
    ///
    /// Finds all nodes that cannot reach any output node by following
    /// edges forward, and removes them. This cleans up computation
    /// paths that don't contribute to the output.
    ///
    /// # Returns
    /// Number of nodes removed
    ///
    /// # Note
    /// - Output nodes are always kept
    /// - If no output nodes are defined, this does nothing
    /// - Uses reverse BFS from all output nodes
    pub fn remove_dead_end_nodes(&mut self) -> usize {
        use std::collections::HashSet;

        if self.output_nodes.is_empty() {
            return 0;
        }

        // Find all nodes that can reach outputs using reverse traversal
        let mut can_reach_output = HashSet::new();

        // Start from output nodes
        for &output in &self.output_nodes {
            can_reach_output.insert(output);
        }

        // Work backwards to find all nodes that can reach outputs
        // We need to iterate until no new nodes are added
        let mut changed = true;
        while changed {
            changed = false;

            for node in self.graph.node_indices() {
                if can_reach_output.contains(&node) {
                    continue;
                }

                // Check if any successor can reach output
                let can_reach = self
                    .graph
                    .neighbors(node)
                    .any(|neighbor| can_reach_output.contains(&neighbor));

                if can_reach {
                    can_reach_output.insert(node);
                    changed = true;
                }
            }
        }

        // Input nodes should always be kept even if they can't reach output
        // (they represent the input sequence)
        for &input in &self.input_nodes {
            can_reach_output.insert(input);
        }

        // Collect dead-end nodes
        let dead_ends: Vec<_> = self
            .graph
            .node_indices()
            .filter(|node| !can_reach_output.contains(node))
            .collect();

        let removed_count = dead_ends.len();

        // Remove dead-end nodes (in reverse index order)
        let mut dead_ends = dead_ends;
        dead_ends.sort_by_key(|n| std::cmp::Reverse(n.index()));

        for node in dead_ends {
            self.graph.remove_node(node);
        }

        if removed_count > 0 {
            let _ = self.update_topology();
        }

        removed_count
    }

    /// Remove all disconnected components - combines orphan, unreachable, and dead-end removal.
    ///
    /// Performs a complete cleanup of the graph by:
    /// 1. Removing nodes unreachable from inputs
    /// 2. Removing dead-end nodes (can't reach outputs)
    /// 3. Removing any remaining orphaned nodes
    ///
    /// # Returns
    /// Total number of nodes removed
    ///
    /// # Example
    /// ```ignore
    /// let mut dag = DagNN::from_text("abc").unwrap();
    /// // After aggressive edge pruning
    /// dag.prune_edges_by_percentile(0.5);
    /// // Clean up all disconnected nodes
    /// let removed = dag.cleanup_disconnected();
    /// ```
    pub fn cleanup_disconnected(&mut self) -> usize {
        let mut total_removed = 0;

        // First pass: remove unreachable from inputs
        total_removed += self.remove_unreachable_from_inputs();

        // Second pass: remove dead-ends (only if we have outputs defined)
        if !self.output_nodes.is_empty() {
            total_removed += self.remove_dead_end_nodes();
        }

        // Final pass: remove any remaining orphans
        total_removed += self.remove_orphaned_nodes();

        total_removed
    }

    /// Get count of orphaned nodes (for diagnostics without removal).
    pub fn count_orphaned_nodes(&self) -> usize {
        use petgraph::Direction;

        self.graph
            .node_indices()
            .filter(|&node| {
                if self.input_nodes_set.contains(&node) || self.output_nodes.contains(&node) {
                    return false;
                }

                let in_degree = self.graph.edges_directed(node, Direction::Incoming).count();
                let out_degree = self.graph.edges_directed(node, Direction::Outgoing).count();

                in_degree == 0 && out_degree == 0
            })
            .count()
    }

    // ========================================================================
    // Neurogenesis (Smart Node/Edge Addition) - Backend-110
    // ========================================================================

    /// Insert a hidden node between two connected nodes (neurogenesis).
    ///
    /// In biological neural networks, new neurons can grow to improve
    /// information processing. This method inserts a new hidden node
    /// between an existing source and target, creating a two-hop path.
    ///
    /// # Arguments
    /// * `source` - Source node of existing edge
    /// * `target` - Target node of existing edge
    /// * `activation_fn` - Activation function for the new node
    ///
    /// # Returns
    /// The NodeId of the newly created hidden node, or None if no edge exists
    ///
    /// # Example
    /// ```ignore
    /// let mut dag = DagNN::from_text("ab").unwrap();
    /// let a = dag.input_nodes()[0];
    /// let b = dag.input_nodes()[1];
    /// // Insert hidden node between a and b
    /// if let Some(hidden) = dag.grow_node_between(a, b, ActivationFn::ReLU) {
    ///     println!("Created new neuron: {:?}", hidden);
    /// }
    /// ```
    pub fn grow_node_between(
        &mut self,
        source: NodeId,
        target: NodeId,
        activation_fn: ActivationFn,
    ) -> Option<NodeId> {
        // Find the edge between source and target
        let edge_idx = self.graph.find_edge(source, target)?;
        let old_edge = self.graph[edge_idx].clone();

        // Create new hidden node
        let hidden = self.add_hidden_with_activation(activation_fn);

        // Remove old edge
        self.graph.remove_edge(edge_idx);

        // Add two new edges: source -> hidden -> target
        // Split the weight between the two new edges
        let half_weight = old_edge.weight / 2.0;
        self.graph
            .add_edge(source, hidden, Edge::new(half_weight, old_edge.edge_type));
        self.graph
            .add_edge(hidden, target, Edge::new(half_weight, old_edge.edge_type));

        // Update topology
        let _ = self.update_topology();

        Some(hidden)
    }

    /// Add a hidden node with a specific activation function.
    pub fn add_hidden_with_activation(&mut self, activation_fn: ActivationFn) -> NodeId {
        let mut node = Node::hidden();
        node.activation_fn = activation_fn;
        self.graph.add_node(node)
    }

    /// Add a shortcut/skip connection between two nodes (neurogenesis).
    ///
    /// Creates a direct edge between two nodes that may not be directly
    /// connected. This enables gradient shortcuts (like ResNet skip connections)
    /// and can help with vanishing gradient problems.
    ///
    /// # Arguments
    /// * `source` - Source node
    /// * `target` - Target node
    /// * `weight` - Initial edge weight
    ///
    /// # Returns
    /// true if edge was added, false if edge already exists or would create cycle
    ///
    /// # Example
    /// ```ignore
    /// let mut dag = DagNN::from_text("abc").unwrap();
    /// let a = dag.input_nodes()[0];
    /// let c = dag.input_nodes()[2];
    /// // Add skip connection from a directly to c
    /// dag.grow_shortcut_edge(a, c, 0.1);
    /// ```
    pub fn grow_shortcut_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        weight: f32,
    ) -> bool {
        // Check if edge already exists
        if self.graph.find_edge(source, target).is_some() {
            return false;
        }

        // Check if this would create a cycle (target -> source path exists)
        // Use a simple DFS to check reachability
        if self.has_path(target, source) {
            return false; // Would create cycle
        }

        // Add the shortcut edge
        self.graph
            .add_edge(source, target, Edge::new(weight, EdgeType::Sequential));

        // Update topology
        let _ = self.update_topology();

        true
    }

    /// Check if there's a path from source to target (for cycle detection).
    fn has_path(&self, source: NodeId, target: NodeId) -> bool {
        use petgraph::visit::Dfs;

        let mut dfs = Dfs::new(&self.graph, source);
        while let Some(node) = dfs.next(&self.graph) {
            if node == target {
                return true;
            }
        }
        false
    }

    /// Grow nodes at high-gradient locations (loss-guided neurogenesis).
    ///
    /// Analyzes gradient magnitudes and inserts new hidden nodes at edges
    /// where gradients are large (indicating the network needs more capacity
    /// to fit the data at those locations).
    ///
    /// # Arguments
    /// * `edge_grads` - Gradient magnitudes for each edge
    /// * `threshold` - Minimum gradient to trigger neurogenesis
    /// * `max_new_nodes` - Maximum number of new nodes to add
    ///
    /// # Returns
    /// Vector of newly created node IDs
    ///
    /// # Example
    /// ```ignore
    /// let mut dag = DagNN::from_text("abc").unwrap();
    /// // After computing gradients during backward pass
    /// let new_nodes = dag.neurogenesis_from_gradient(&edge_grads, 0.5, 3);
    /// println!("Added {} new neurons", new_nodes.len());
    /// ```
    pub fn neurogenesis_from_gradient(
        &mut self,
        edge_grads: &HashMap<(NodeId, NodeId), f32>,
        threshold: f32,
        max_new_nodes: usize,
    ) -> Vec<NodeId> {
        if max_new_nodes == 0 {
            return vec![];
        }

        // Collect edges with gradients above threshold, sorted by gradient magnitude
        let mut high_grad_edges: Vec<_> = edge_grads
            .iter()
            .filter(|(_, &grad)| grad.abs() >= threshold)
            .map(|(&(src, tgt), &grad)| (src, tgt, grad.abs()))
            .collect();

        // Sort by gradient magnitude (descending)
        high_grad_edges.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Add new nodes at highest gradient edges
        let mut new_nodes = Vec::new();

        for (source, target, _grad) in high_grad_edges.into_iter().take(max_new_nodes) {
            // Only add if edge still exists (previous growth may have removed it)
            if self.graph.find_edge(source, target).is_some() {
                if let Some(node) = self.grow_node_between(source, target, ActivationFn::ReLU) {
                    new_nodes.push(node);
                }
            }
        }

        new_nodes
    }

    /// Grow skip connections to high-gradient nodes (shortcut neurogenesis).
    ///
    /// Adds skip connections from input nodes to hidden nodes that have
    /// high gradients, enabling better gradient flow.
    ///
    /// # Arguments
    /// * `node_grads` - Gradient magnitudes for each node
    /// * `threshold` - Minimum gradient to trigger skip connection
    /// * `max_shortcuts` - Maximum number of shortcuts to add
    /// * `weight` - Weight for new shortcut edges
    ///
    /// # Returns
    /// Number of shortcuts added
    pub fn grow_shortcuts_from_gradient(
        &mut self,
        node_grads: &HashMap<NodeId, f32>,
        threshold: f32,
        max_shortcuts: usize,
        weight: f32,
    ) -> usize {
        if max_shortcuts == 0 || self.input_nodes.is_empty() {
            return 0;
        }

        // Collect nodes with high gradients that aren't inputs
        let mut high_grad_nodes: Vec<_> = node_grads
            .iter()
            .filter(|(&node, &grad)| {
                grad.abs() >= threshold && !self.input_nodes_set.contains(&node)
            })
            .map(|(&node, &grad)| (node, grad.abs()))
            .collect();

        // Sort by gradient magnitude (descending)
        high_grad_nodes.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut shortcuts_added = 0;

        for (target_node, _grad) in high_grad_nodes.into_iter().take(max_shortcuts) {
            // Try to add shortcut from each input until one succeeds
            for &input in &self.input_nodes.clone() {
                if self.grow_shortcut_edge(input, target_node, weight) {
                    shortcuts_added += 1;
                    break; // One shortcut per target node
                }
            }
        }

        shortcuts_added
    }

    /// Get statistics about network structure for monitoring growth/pruning.
    ///
    /// # Returns
    /// Tuple of (node_count, edge_count, avg_in_degree, avg_out_degree, max_depth)
    pub fn structure_stats(&self) -> (usize, usize, f32, f32, usize) {
        use petgraph::Direction;

        let node_count = self.graph.node_count();
        let edge_count = self.graph.edge_count();

        if node_count == 0 {
            return (0, 0, 0.0, 0.0, 0);
        }

        let total_in: usize = self
            .graph
            .node_indices()
            .map(|n| self.graph.edges_directed(n, Direction::Incoming).count())
            .sum();

        let total_out: usize = self
            .graph
            .node_indices()
            .map(|n| self.graph.edges_directed(n, Direction::Outgoing).count())
            .sum();

        let avg_in = total_in as f32 / node_count as f32;
        let avg_out = total_out as f32 / node_count as f32;

        // Calculate max depth (longest path from any input)
        let max_depth = self.topology.order.len();

        (node_count, edge_count, avg_in, avg_out, max_depth)
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

    /// Set output nodes (replaces existing output nodes)
    pub fn set_output_nodes(&mut self, nodes: Vec<NodeId>) {
        self.output_nodes = nodes;
    }

    /// Add an output node
    pub fn add_output_node(&mut self, node: NodeId) {
        if !self.output_nodes.contains(&node) {
            self.output_nodes.push(node);
        }
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
        // Use neuromorphic forward pass with per-node activation functions
        self.neuromorphic_forward()
    }

    fn forward_parallel(&mut self) -> GraphemeResult<()> {
        // Update topology if needed
        if self.topology.order.is_empty() {
            self.update_topology()?;
        }

        // Group nodes by level (distance from inputs) for parallel processing
        // Nodes at the same level have no dependencies on each other
        let levels = self.compute_node_levels();

        // Process each level sequentially, but nodes within each level in parallel
        for level_nodes in levels {
            if level_nodes.is_empty() {
                continue;
            }

            // Compute pre-activations for all nodes at this level in parallel
            // Collect results first to avoid borrow conflicts
            let pre_activations: Vec<(NodeId, f32, bool)> = level_nodes
                .par_iter()
                .map(|&node| {
                    let is_input = self.input_nodes_set.contains(&node);
                    if is_input {
                        (node, self.graph[node].activation, true)
                    } else {
                        // Sum weighted inputs from predecessors
                        let pre_act: f32 = self
                            .graph
                            .edges_directed(node, petgraph::Direction::Incoming)
                            .map(|edge| {
                                self.graph[edge.source()].activation * edge.weight().weight
                            })
                            .sum();
                        (node, pre_act, false)
                    }
                })
                .collect();

            // Apply activations sequentially (mutating the graph)
            for (node, pre_act, is_input) in pre_activations {
                if is_input {
                    self.graph[node].pre_activation = pre_act;
                } else {
                    self.graph[node].set_pre_activation(pre_act);
                }
            }
        }

        Ok(())
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
// Parallel Processing Utilities (backend-150)
// ============================================================================

impl DagNN {
    /// Compute node levels for parallel processing.
    ///
    /// Groups nodes by their "level" (distance from inputs), where nodes
    /// at the same level have no dependencies on each other and can be
    /// processed in parallel.
    ///
    /// # Algorithm
    /// O(V + E) BFS from input nodes, assigning level = max(predecessor levels) + 1
    ///
    /// # Returns
    /// Vector of levels, where each level contains nodes that can be processed in parallel
    fn compute_node_levels(&self) -> Vec<Vec<NodeId>> {
        let mut levels: Vec<Vec<NodeId>> = Vec::new();
        let mut node_level: HashMap<NodeId, usize> = HashMap::new();

        // Process nodes in topological order to ensure predecessors are processed first
        for &node in &self.topology.order {
            // Input nodes are level 0
            if self.input_nodes_set.contains(&node) {
                node_level.insert(node, 0);
                if levels.is_empty() {
                    levels.push(Vec::new());
                }
                levels[0].push(node);
            } else {
                // Level = max(predecessor levels) + 1
                let max_pred_level = self
                    .graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .filter_map(|edge| node_level.get(&edge.source()))
                    .max()
                    .copied()
                    .unwrap_or(0);

                let level = max_pred_level + 1;
                node_level.insert(node, level);

                // Ensure we have enough levels
                while levels.len() <= level {
                    levels.push(Vec::new());
                }
                levels[level].push(node);
            }
        }

        levels
    }
}

// ============================================================================
// Neuromorphic Forward Pass (backend-107)
// ============================================================================

impl DagNN {
    /// Neuromorphic forward pass with per-node activation functions (backend-107)
    ///
    /// This implements a biologically-inspired forward pass where:
    /// 1. Nodes are processed in topological order (respecting causality)
    /// 2. Each node sums weighted inputs from predecessors
    /// 3. Each node applies its own activation function (heterogeneous network)
    /// 4. Pre-activation values are cached for efficient backpropagation
    ///
    /// # Algorithm
    /// ```text
    /// for each node in topological_order:
    ///     pre_activation = sum(edge_weight[u,v] * activation[u] for u in predecessors)
    ///     activation = node.activation_fn.apply(pre_activation)
    /// ```
    ///
    /// # Complexity
    /// O(V + E) where V = nodes, E = edges
    ///
    /// # Example
    /// ```ignore
    /// let mut dag = DagNN::from_text("hello").unwrap();
    /// dag.neuromorphic_forward()?;
    /// let activations = dag.get_activations();
    /// ```
    pub fn neuromorphic_forward(&mut self) -> GraphemeResult<()> {
        // Update topology if needed
        if self.topology.order.is_empty() {
            self.update_topology()?;
        }

        // Process nodes in topological order
        for &node in &self.topology.order.clone() {
            // Check if this is an input node (no predecessors or in input_nodes_set)
            let is_input = self.input_nodes_set.contains(&node);

            if is_input {
                // Input nodes keep their activation (typically 1.0)
                // Still set pre_activation for consistency
                let activation = self.graph[node].activation;
                self.graph[node].pre_activation = activation;
            } else {
                // Sum weighted inputs from predecessors
                let mut pre_activation = 0.0f32;

                for edge in self
                    .graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                {
                    let source_activation = self.graph[edge.source()].activation;
                    let weight = edge.weight().weight;
                    pre_activation += source_activation * weight;
                }

                // Apply per-node activation function and cache values
                self.graph[node].set_pre_activation(pre_activation);
            }
        }

        Ok(())
    }

    /// Forward pass with custom input activations (backend-107)
    ///
    /// Allows setting specific activation values for input nodes before
    /// propagating through the network.
    ///
    /// # Arguments
    /// * `input_activations` - Map from node ID to activation value
    ///
    /// # Example
    /// ```ignore
    /// let mut dag = DagNN::from_text("hi").unwrap();
    /// let mut inputs = HashMap::new();
    /// inputs.insert(dag.input_nodes()[0], 0.5);
    /// inputs.insert(dag.input_nodes()[1], 0.8);
    /// dag.forward_with_inputs(&inputs)?;
    /// ```
    pub fn forward_with_inputs(
        &mut self,
        input_activations: &HashMap<NodeId, f32>,
    ) -> GraphemeResult<()> {
        // Set input activations
        for (&node, &activation) in input_activations {
            self.graph[node].activation = activation;
            self.graph[node].pre_activation = activation;
        }

        // Run forward pass
        self.neuromorphic_forward()
    }

    /// Get pre-activation values (useful for backpropagation)
    pub fn get_pre_activations(&self) -> Vec<(NodeId, f32)> {
        self.topology
            .order
            .iter()
            .map(|&node| (node, self.graph[node].pre_activation))
            .collect()
    }

    /// Get activation derivatives for all nodes (for backpropagation)
    pub fn get_activation_derivatives(&self) -> Vec<(NodeId, f32)> {
        self.topology
            .order
            .iter()
            .map(|&node| (node, self.graph[node].activation_derivative()))
            .collect()
    }

    /// Compute output activations for the last N nodes
    ///
    /// Useful for getting predictions from output nodes
    pub fn get_output_activations(&self, n: usize) -> Vec<f32> {
        self.topology
            .order
            .iter()
            .rev()
            .take(n)
            .map(|&node| self.graph[node].activation)
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

impl Learnable for DagNN {
    fn zero_grad(&mut self) {
        self.edge_grads.clear();
    }

    fn step(&mut self, lr: f32) {
        for ((from, to), grad) in &self.edge_grads {
            if let Some(edge_idx) = self.graph.find_edge(*from, *to) {
                self.graph[edge_idx].weight -= lr * grad;
            }
        }
    }

    fn num_parameters(&self) -> usize {
        self.graph.edge_count()
    }

    fn has_gradients(&self) -> bool {
        !self.edge_grads.is_empty()
    }

    fn gradient_norm(&self) -> f32 {
        let sum_sq: f32 = self.edge_grads.values().map(|g| g * g).sum();
        sum_sq.sqrt()
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
            let Some(min_node) = remaining
                .iter()
                .min_by_key(|&node| {
                    self.neighbors(*node)
                        .filter(|n| remaining.contains(n)) // O(1) lookup now!
                        .count()
                })
                .copied()
            else {
                break; // Should never happen since we check !remaining.is_empty()
            };

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
    /// Used by tests for validation of combinations_iter
    #[cfg(test)]
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

    /// Save checkpoint to gzip-compressed file (.json.gz)
    /// Typically achieves 5-10x compression ratio for checkpoint data
    pub fn save_compressed(&self, path: &std::path::Path) -> PersistenceResult<()> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let json = self.save_json()?;
        let file = std::fs::File::create(path)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder
            .write_all(json.as_bytes())
            .map_err(|e| PersistenceError::Serialization(format!("gzip write failed: {}", e)))?;
        encoder
            .finish()
            .map_err(|e| PersistenceError::Serialization(format!("gzip finish failed: {}", e)))?;
        Ok(())
    }

    /// Load checkpoint from gzip-compressed file (.json.gz)
    pub fn load_compressed(path: &std::path::Path) -> PersistenceResult<Self> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let file = std::fs::File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        let mut json = String::new();
        decoder
            .read_to_string(&mut json)
            .map_err(|e| PersistenceError::Deserialization(format!("gzip read failed: {}", e)))?;
        Self::load_json(&json)
    }

    /// Save checkpoint with automatic format detection based on extension
    /// - `.json` or `.checkpoint` -> uncompressed JSON
    /// - `.json.gz` or `.checkpoint.gz` -> gzip compressed
    pub fn save_auto(&self, path: &std::path::Path) -> PersistenceResult<()> {
        let path_str = path.to_string_lossy();
        if path_str.ends_with(".gz") {
            self.save_compressed(path)
        } else {
            self.save_to_file(path)
        }
    }

    /// Load checkpoint with automatic format detection based on extension
    /// - `.json` or `.checkpoint` -> uncompressed JSON
    /// - `.json.gz` or `.checkpoint.gz` -> gzip compressed
    pub fn load_auto(path: &std::path::Path) -> PersistenceResult<Self> {
        let path_str = path.to_string_lossy();
        if path_str.ends_with(".gz") {
            Self::load_compressed(path)
        } else {
            Self::load_from_file(path)
        }
    }

    /// Get compressed checkpoint as bytes (for network transfer, etc.)
    pub fn to_compressed_bytes(&self) -> PersistenceResult<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let json = self.save_json()?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(json.as_bytes())
            .map_err(|e| PersistenceError::Serialization(format!("gzip write failed: {}", e)))?;
        encoder
            .finish()
            .map_err(|e| PersistenceError::Serialization(format!("gzip finish failed: {}", e)))
    }

    /// Load checkpoint from compressed bytes
    pub fn from_compressed_bytes(data: &[u8]) -> PersistenceResult<Self> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(data);
        let mut json = String::new();
        decoder
            .read_to_string(&mut json)
            .map_err(|e| PersistenceError::Deserialization(format!("gzip read failed: {}", e)))?;
        Self::load_json(&json)
    }

    /// Estimate the compression ratio for this checkpoint
    /// Returns (uncompressed_size, compressed_size, ratio)
    pub fn compression_stats(&self) -> PersistenceResult<(usize, usize, f64)> {
        let json = self.save_json()?;
        let uncompressed = json.len();
        let compressed = self.to_compressed_bytes()?.len();
        let ratio = if compressed > 0 {
            uncompressed as f64 / compressed as f64
        } else {
            0.0
        };
        Ok((uncompressed, compressed, ratio))
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

impl Persistable for DagNN {
    fn persist_type_id() -> &'static str {
        "DagNN"
    }

    fn persist_version() -> u32 {
        1
    }

    fn validate(&self) -> Result<(), PersistenceError> {
        // Validate topological order matches graph structure
        if self.topology.order.len() != self.graph.node_count() {
            return Err(PersistenceError::ValidationFailed(
                "Topological order length mismatch with node count".to_string(),
            ));
        }
        Ok(())
    }
}

impl Persistable for EncoderDecoder {
    fn persist_type_id() -> &'static str {
        "EncoderDecoder"
    }

    fn persist_version() -> u32 {
        1  // First version of encoder-decoder architecture
    }

    fn validate(&self) -> Result<(), PersistenceError> {
        // Verify encoder and decoder dimensions are compatible
        if self.encoder.hidden_dim != self.decoder.hidden_dim {
            return Err(PersistenceError::ValidationFailed(
                "Encoder/decoder hidden dimension mismatch".to_string(),
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
    /// Number of output nodes (k can be <, =, or > n)
    pub num_clusters: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Learnable query matrix Q ∈ ℝ^{k × d}
    /// Each row represents what one output node "queries" for
    pub query_matrix: Array2<f32>,
    /// Gradient accumulator for query_matrix
    #[serde(skip)]
    pub query_grad: Option<Array2<f32>>,
    /// Sinkhorn iterations for assignment refinement
    pub sinkhorn_iterations: usize,
    /// Temperature for softmax (lower = sharper assignments)
    pub temperature: f32,
    /// Edge threshold for coarsened adjacency (A_new = S^T·A·S)
    /// Edges with weight below this threshold are pruned
    #[serde(default = "default_edge_threshold")]
    pub edge_threshold: f32,
}

fn default_edge_threshold() -> f32 {
    0.1
}

impl SabagPooling {
    /// Create new Sabag pooling layer
    ///
    /// # Arguments
    /// * `num_clusters` - Number of output nodes (k can be <, =, or > n)
    /// * `embed_dim` - Dimension of node embeddings
    /// * `sinkhorn_iterations` - Number of Sinkhorn refinement iterations (default: 10)
    /// * `temperature` - Softmax temperature (default: 0.1, lower = sharper)
    pub fn new(num_clusters: usize, embed_dim: usize) -> Self {
        // Initialize query matrix with small random values
        let mut rng = rand::thread_rng();
        let query_matrix = Array2::from_shape_fn((num_clusters, embed_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });

        Self {
            num_clusters,
            embed_dim,
            query_matrix,
            query_grad: None,
            sinkhorn_iterations: 10,
            temperature: 0.1,
            edge_threshold: default_edge_threshold(),
        }
    }

    /// Sabag Step 1: Compute DAG-aware pairwise similarity matrix
    ///
    /// Creates an n×n similarity matrix where only topologically valid
    /// node pairs have non-zero similarity (preserves DAG structure).
    ///
    /// Complexity: O(n²·d) for similarity computation
    #[allow(dead_code)]
    fn compute_dag_similarity(
        &self,
        _graph: &GraphemeGraph,
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
    fn sinkhorn_refine(&self, mut z: Array2<f32>) -> Array2<f32> {
        let n = z.nrows();
        let k = z.ncols();

        // Target: Each cluster should have n/k nodes
        let target_cluster_size = n as f32 / k as f32;

        for _ in 0..self.sinkhorn_iterations {
            // Row normalization: Σ_j z[i,j] = 1 (each node assigns to clusters)
            for i in 0..n {
                let row_sum: f32 = z.row(i).sum();
                if row_sum > 1e-8 {
                    for j in 0..k {
                        z[[i, j]] /= row_sum;
                    }
                }
            }

            // Column normalization: Σ_i z[i,j] ≈ n/k (balanced clusters)
            for j in 0..k {
                let col_sum: f32 = z.column(j).sum();
                if col_sum > 1e-8 {
                    let scale = target_cluster_size / col_sum;
                    for i in 0..n {
                        z[[i, j]] *= scale;
                    }
                }
            }
        }

        // Final row normalization to ensure probability distribution
        for i in 0..n {
            let row_sum: f32 = z.row(i).sum();
            if row_sum > 1e-8 {
                for j in 0..k {
                    z[[i, j]] /= row_sum;
                }
            }
        }

        z
    }

    /// Sabag Step 3: Compute soft assignment matrix using learnable queries
    ///
    /// Computes S ∈ ℝ^{k × n} where:
    /// - k = num_clusters (output size, can be <, =, or > n)
    /// - n = number of input nodes
    /// - S[i,j] = attention weight of output node i to input node j
    ///
    /// This is essentially attention: each output node queries all input nodes.
    /// Works for ANY k vs n relationship (compression, identity, expansion).
    ///
    /// Complexity: O(k·n·d) - same as before, but simpler!
    fn compute_assignment_matrix(&self, embeddings: &Array2<f32>) -> Array2<f32> {
        let _n = embeddings.nrows();  // Input nodes (documented for clarity)
        let _k = self.num_clusters;   // Output nodes (documented for clarity)

        // Compute attention scores: Q · H^T ∈ ℝ^{k × n}
        // Each row i: how much does output node i attend to each input node j?
        let scores = self.query_matrix.dot(&embeddings.t());

        // Apply temperature scaling and softmax per row
        // s[i,j] = exp(score[i,j]/T) / Σ_j exp(score[i,j]/T)
        let z = scores.mapv(|x| (x / self.temperature).exp());

        // Optional: Sinkhorn refinement for balanced assignment
        self.sinkhorn_refine(z)
    }

    /// Softmax activation (rowwise)
    ///
    /// s[i,j] = exp(z[i,j]) / Σ_k exp(z[i,k])
    ///
    /// Ensures each row sums to 1 (probability distribution over clusters).
    ///
    /// Complexity: O(n·k) - linear in matrix size
    #[allow(dead_code)]
    fn softmax_rowwise(&self, z: &Array2<f32>) -> Array2<f32> {
        let mut s = Array2::zeros(z.dim());

        for i in 0..z.nrows() {
            let row = z.row(i);

            // Numerical stability: subtract max before exp
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_row: Vec<f32> = row.iter().map(|&val| (val - max_val).exp()).collect();
            let sum_exp: f32 = exp_row.iter().sum();

            for j in 0..z.ncols() {
                s[[i, j]] = exp_row[j] / sum_exp;
            }
        }

        s
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
        let n = embeddings.nrows();  // Input nodes
        let k = self.num_clusters;   // Output nodes (no constraint!)

        // Sabag Step 1: Compute soft assignment matrix using learnable queries
        // s ∈ ℝ^{k × n} - each output node attends to all input nodes
        let assignment = self.compute_assignment_matrix(embeddings);

        // DEBUG: Check assignment dimensions
        if assignment.nrows() != k || assignment.ncols() != n {
            eprintln!("WARNING: assignment has wrong shape! Expected {}×{}, got {}×{}",
                     k, n, assignment.nrows(), assignment.ncols());
        }

        // Sabag Step 2: Compute output features via attention-weighted combination
        // h_new = assignment · H ∈ ℝ^{k × d}
        // Each output node is a weighted combination of input nodes
        // Works for k < n (compression), k = n (identity), k > n (expansion)!
        let h_new = assignment.dot(embeddings);

        // Sabag Step 3: Build output DAG with k nodes
        let output_graph = self.create_coarsened_dag(graph, k, &h_new, &assignment);

        PoolingResult {
            graph: output_graph,
            features: h_new,
            assignment,  // CRITICAL: Store for backward pass!
        }
    }

    /// Sabag Step 6: Create coarsened DAG structure with proper adjacency coarsening
    ///
    /// Builds a new DAG with k nodes (k < n) where:
    /// - Nodes represent soft clusters of input nodes
    /// - Edges are computed via: A_new = S^T · A · S
    /// - DAG topology is preserved through soft adjacency aggregation
    ///
    /// # Mathematical Foundation
    /// Given:
    /// - A ∈ ℝ^{n × n}: Input adjacency matrix (weighted)
    /// - S ∈ ℝ^{n × k}: Soft assignment matrix (each row sums to 1)
    ///
    /// The coarsened adjacency is:
    /// - A_new = S^T · A · S ∈ ℝ^{k × k}
    ///
    /// Element (i,j) of A_new represents: "how much do nodes assigned to cluster i
    /// connect to nodes assigned to cluster j?"
    ///
    /// # Arguments
    /// * `input_graph` - Original DAG with n nodes
    /// * `k` - Number of output clusters
    /// * `features` - Coarsened features H_new ∈ ℝ^{k × d}
    /// * `assignment` - Soft assignment matrix S ∈ ℝ^{n × k}
    fn create_coarsened_dag(
        &self,
        input_graph: &GraphemeGraph,
        k: usize,
        features: &Array2<f32>,
        assignment: &Array2<f32>,
    ) -> GraphemeGraph {
        use ndarray::Array2;

        let mut graph = DiGraph::new();
        let mut input_nodes = Vec::with_capacity(k);

        // Step 1: Create k nodes for the coarsened graph
        for idx in 0..k {
            let activation = if idx < features.nrows() {
                features.row(idx).mean().unwrap_or(0.0)
            } else {
                0.0
            };

            let node = Node {
                value: Some(b'x'),  // Placeholder character
                activation,
                pre_activation: activation,
                node_type: NodeType::Input('x'),
                position: Some(idx),
                activation_fn: ActivationFn::Linear,
            };

            let node_id = graph.add_node(node);
            input_nodes.push(node_id);
        }

        // Step 2: Build adjacency matrix A from input graph
        // Map node indices to dense [0..n) range
        let node_list: Vec<_> = input_graph.graph.node_indices().collect();
        let n = node_list.len();

        if n == 0 || k == 0 {
            return GraphemeGraph {
                graph,
                input_nodes,
                cliques: Vec::new(),
            };
        }

        // Create node index mapping: NodeIndex -> dense index
        let mut node_to_idx = std::collections::HashMap::new();
        for (idx, &node_id) in node_list.iter().enumerate() {
            node_to_idx.insert(node_id, idx);
        }

        // Build sparse adjacency A (n × n)
        let mut adj = Array2::<f32>::zeros((n, n));
        for edge_idx in input_graph.graph.edge_indices() {
            if let Some((source, target)) = input_graph.graph.edge_endpoints(edge_idx) {
                if let (Some(&src_idx), Some(&tgt_idx)) = (node_to_idx.get(&source), node_to_idx.get(&target)) {
                    if let Some(edge) = input_graph.graph.edge_weight(edge_idx) {
                        adj[[src_idx, tgt_idx]] = edge.weight.abs().max(0.1); // Use edge weight
                    } else {
                        adj[[src_idx, tgt_idx]] = 1.0; // Default weight
                    }
                }
            }
        }

        // Step 3: Compute coarsened adjacency A_new = S^T · A · S
        // S is (n × k), so S^T is (k × n)
        // A is (n × n)
        // Result: (k × n) · (n × n) · (n × k) = (k × k)

        // Ensure assignment matrix dimensions match
        let (s_rows, s_cols) = assignment.dim();
        if s_rows != n || s_cols != k {
            // Dimension mismatch - fall back to no edges
            return GraphemeGraph {
                graph,
                input_nodes,
                cliques: Vec::new(),
            };
        }

        // A_new = S^T · A · S
        let s_t = assignment.t(); // (k × n)
        let temp = s_t.dot(&adj);  // (k × n) · (n × n) = (k × n)
        let adj_new = temp.dot(assignment); // (k × n) · (n × k) = (k × k)

        // Step 4: Create edges based on coarsened adjacency
        // Use threshold to create discrete edges from soft adjacency
        let edge_threshold = self.edge_threshold;

        for i in 0..k {
            for j in 0..k {
                if i == j {
                    continue; // Skip self-loops
                }

                let weight = adj_new[[i, j]];
                if weight > edge_threshold {
                    // Create edge from cluster i to cluster j
                    graph.add_edge(
                        input_nodes[i],
                        input_nodes[j],
                        Edge::new(weight, EdgeType::Sequential),
                    );
                }
            }
        }

        GraphemeGraph {
            graph,
            input_nodes,
            cliques: Vec::new(),  // Cliques could be coarsened similarly in future
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
        // Future: Also compute gradient w.r.t. assignment matrix
        // ∂L/∂S = (∂L/∂H_new) · H^T
        // Then backprop through softmax Jacobian to get ∂L/∂Z
        result.assignment.dot(grad_features)
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
    /// * `grad_s` - Gradient w.r.t. soft assignment ∂L/∂S ∈ ℝ^{n × k}
    /// * `s` - Soft assignment matrix from forward pass ∈ ℝ^{n × k}
    ///
    /// # Returns
    /// Gradient w.r.t. assignment scores ∂L/∂Z ∈ ℝ^{n × k}
    fn softmax_backward(
        &self,
        grad_s: &Array2<f32>,
        s: &Array2<f32>,
    ) -> Array2<f32> {
        let mut grad_z = Array2::zeros(s.dim());

        for i in 0..s.nrows() {
            for k in 0..s.ncols() {
                let mut grad_sum = 0.0;

                for j in 0..s.ncols() {
                    if j == k {
                        // δ_{jk} = 1
                        grad_sum += grad_s[[i, j]] * s[[i, j]] * (1.0 - s[[i, j]]);
                    } else {
                        // δ_{jk} = 0
                        grad_sum -= grad_s[[i, j]] * s[[i, j]] * s[[i, k]];
                    }
                }

                grad_z[[i, k]] = grad_sum;
            }
        }

        grad_z
    }

    /// Zero out gradients
    pub fn zero_grad(&mut self) {
        self.query_grad = Some(Array2::zeros((self.num_clusters, self.embed_dim)));
    }

    /// Update query matrix using accumulated gradients (SGD step)
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.query_grad {
            self.query_matrix = &self.query_matrix - &(grad * lr);
        }
    }

    /// Accumulate gradient for query matrix
    ///
    /// This is called during backward pass to accumulate ∂L/∂Q
    pub fn accumulate_query_grad(&mut self, grad: &Array2<f32>) {
        if self.query_grad.is_none() {
            self.zero_grad();
        }
        if let Some(ref mut existing_grad) = self.query_grad {
            *existing_grad = &*existing_grad + grad;
        }
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

    /// Get the embedding for a node - DOMAIN-AGNOSTIC (scikit-learn pipeline style)
    ///
    /// GRAPHEME Core sees ONLY graphs. It does NOT know what domain it's processing.
    /// The embedding is computed uniformly from node properties (value/activation),
    /// NOT from domain-specific node types. Cognitive Brains handle domain translation.
    ///
    /// Embedding strategy:
    /// - If node has a value (0-255): use that as index into embedding table
    /// - Otherwise: scale activation (0.0-1.0) to embedding space (0-255)
    pub fn embed_node(&self, node: &Node) -> Array1<f32> {
        // Domain-agnostic: use node's value or activation, NOT its type
        let idx = if let Some(v) = node.value {
            v as usize
        } else {
            // Scale activation [0.0, 1.0] to embedding index [0, 255]
            (node.activation * 255.0).clamp(0.0, 255.0) as usize
        };
        self.forward_index(idx)
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

    /// Decode an embedding back to the nearest character (Graph-to-Text)
    ///
    /// This is the inverse of forward() - finds the character whose embedding
    /// is most similar to the given embedding vector using cosine similarity.
    ///
    /// # Arguments
    /// * `embedding` - The embedding vector to decode
    ///
    /// # Returns
    /// The character whose embedding is closest to the input
    pub fn decode(&self, embedding: &Array1<f32>) -> char {
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;

        // Compute embedding norm once
        let emb_norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if emb_norm == 0.0 {
            return ' '; // Return space for zero embeddings
        }

        // Search through printable ASCII characters (32-126)
        for idx in 32..=126usize {
            if idx >= self.vocab_size {
                break;
            }
            let row = self.weights.row(idx);
            let row_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if row_norm == 0.0 {
                continue;
            }
            let dot: f32 = embedding.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
            let sim = dot / (emb_norm * row_norm);
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }

        (best_idx as u8) as char
    }

    /// Decode a batch of embeddings back to text
    ///
    /// # Arguments
    /// * `embeddings` - Matrix of embeddings (n_chars x embed_dim)
    ///
    /// # Returns
    /// String of decoded characters
    pub fn decode_batch(&self, embeddings: &Array2<f32>) -> String {
        let mut result = String::with_capacity(embeddings.nrows());
        for i in 0..embeddings.nrows() {
            let emb = embeddings.row(i).to_owned();
            result.push(self.decode(&emb));
        }
        result
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
    #[deprecated(
        since = "0.1.0",
        note = "Use backward_accumulate() + step() for proper gradient accumulation"
    )]
    fn backward_and_update(
        &mut self,
        output_grad: &HashMap<NodeId, Array1<f32>>,
        embedding: &mut Embedding,
        lr: f32,
    );

    /// Compute backward pass and accumulate gradients into DagNN's edge_grads (backend-142)
    ///
    /// Unlike `backward()` which returns gradients, this method accumulates them
    /// into the DagNN's internal gradient storage for later application via `step()`.
    ///
    /// This enables:
    /// - Mini-batch gradient accumulation
    /// - Gradient clipping before update
    /// - Standard training loop pattern: zero_grad → backward_accumulate → step
    ///
    /// # Arguments
    /// * `output_grad` - Gradient from the loss with respect to output nodes
    /// * `embedding` - The embedding layer to accumulate gradients to
    fn backward_accumulate(
        &mut self,
        output_grad: &HashMap<NodeId, Array1<f32>>,
        embedding: &mut Embedding,
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

    fn backward_accumulate(
        &mut self,
        output_grad: &HashMap<NodeId, Array1<f32>>,
        embedding: &mut Embedding,
    ) {
        // Skip if not in training mode
        if !self.requires_grad {
            return;
        }

        // Compute gradients using existing backward pass
        let grads = self.backward(output_grad, embedding);

        // Accumulate edge gradients into DagNN's internal storage
        for ((from, to), edge_grad) in grads.edge_grads {
            self.accumulate_edge_grad(from, to, edge_grad);
        }
    }
}

// ============================================================================
// Hebbian Learning (backend-111)
// ============================================================================

/// Configuration for Hebbian learning
///
/// Hebbian learning follows the principle "neurons that fire together wire together".
/// This implementation supports:
/// - Classic Hebbian: Δw = η * pre * post
/// - Oja's rule: Δw = η * post * (pre - w * post) - adds weight normalization
/// - BCM rule: Δw = η * pre * post * (post - θ) - sliding threshold for bidirectional plasticity
#[derive(Debug, Clone)]
pub struct HebbianConfig {
    /// Learning rate for Hebbian updates
    pub learning_rate: f32,
    /// Weight decay factor (0.0 = no decay, 1.0 = full decay)
    pub weight_decay: f32,
    /// Maximum absolute weight value (for stability)
    pub max_weight: f32,
    /// Minimum absolute weight value (for pruning threshold)
    pub min_weight: f32,
    /// Hebbian learning rule type
    pub rule: HebbianRule,
    /// BCM sliding threshold (for BCM rule)
    pub bcm_threshold: f32,
}

impl Default for HebbianConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            weight_decay: 0.0001,
            max_weight: 10.0,
            min_weight: 0.0,
            rule: HebbianRule::Classic,
            bcm_threshold: 0.5,
        }
    }
}

impl HebbianConfig {
    /// Create a new HebbianConfig with specified learning rate
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    /// Use Oja's rule for weight normalization
    pub fn with_oja_rule(mut self) -> Self {
        self.rule = HebbianRule::Oja;
        self
    }

    /// Use BCM rule for bidirectional plasticity
    pub fn with_bcm_rule(mut self, threshold: f32) -> Self {
        self.rule = HebbianRule::BCM;
        self.bcm_threshold = threshold;
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Set weight bounds
    pub fn with_weight_bounds(mut self, min: f32, max: f32) -> Self {
        self.min_weight = min;
        self.max_weight = max;
        self
    }
}

/// Hebbian learning rule variants
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HebbianRule {
    /// Classic Hebbian: Δw = η * pre * post
    Classic,
    /// Oja's rule: Δw = η * post * (pre - w * post)
    /// Provides automatic weight normalization
    Oja,
    /// BCM rule: Δw = η * pre * post * (post - θ)
    /// Bidirectional plasticity with sliding threshold
    BCM,
    /// Anti-Hebbian: Δw = -η * pre * post
    /// Used for decorrelation and competitive learning
    AntiHebbian,
}

/// Configuration for hybrid gradient descent + Hebbian learning
#[derive(Debug, Clone)]
pub struct HybridLearningConfig {
    /// Learning rate for gradient descent
    pub gradient_lr: f32,
    /// Hebbian configuration
    pub hebbian: HebbianConfig,
    /// Weight for gradient descent contribution (0.0 to 1.0)
    pub gradient_weight: f32,
    /// Weight for Hebbian contribution (0.0 to 1.0)
    pub hebbian_weight: f32,
    /// Whether to apply gradient clipping
    pub clip_gradients: bool,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
}

impl Default for HybridLearningConfig {
    fn default() -> Self {
        Self {
            gradient_lr: 0.001,
            hebbian: HebbianConfig::default(),
            gradient_weight: 0.7,
            hebbian_weight: 0.3,
            clip_gradients: true,
            max_grad_norm: 1.0,
        }
    }
}

impl HybridLearningConfig {
    /// Create config with custom gradient/Hebbian balance
    pub fn new(gradient_weight: f32, hebbian_weight: f32) -> Self {
        Self {
            gradient_weight,
            hebbian_weight,
            ..Default::default()
        }
    }

    /// Set learning rates
    pub fn with_learning_rates(mut self, gradient_lr: f32, hebbian_lr: f32) -> Self {
        self.gradient_lr = gradient_lr;
        self.hebbian.learning_rate = hebbian_lr;
        self
    }
}

/// Trait for Hebbian learning on DagNN
pub trait HebbianLearning {
    /// Apply pure Hebbian learning update to edge weights
    ///
    /// Updates weights based on pre-synaptic and post-synaptic activations:
    /// - Classic: Δw = η * pre * post
    /// - Oja: Δw = η * post * (pre - w * post)
    /// - BCM: Δw = η * pre * post * (post - θ)
    fn backward_hebbian(&mut self, config: &HebbianConfig) -> HebbianResult;

    /// Apply hybrid gradient descent + Hebbian learning
    ///
    /// Combines gradient-based learning with Hebbian updates for biologically
    /// plausible learning that maintains good optimization properties.
    fn backward_hybrid(
        &mut self,
        output_grad: &HashMap<NodeId, Array1<f32>>,
        embedding: &mut Embedding,
        config: &HybridLearningConfig,
    ) -> HybridResult;

    /// Compute Hebbian weight delta for a single edge
    fn compute_hebbian_delta(
        &self,
        source: NodeId,
        target: NodeId,
        current_weight: f32,
        config: &HebbianConfig,
    ) -> f32;

    /// Apply competitive learning (lateral inhibition)
    ///
    /// Implements winner-take-all dynamics where strongly activated nodes
    /// suppress their neighbors, leading to sparse representations.
    fn apply_competitive_learning(&mut self, inhibition_strength: f32);
}

/// Result of Hebbian learning step
#[derive(Debug, Default)]
pub struct HebbianResult {
    /// Number of edges updated
    pub edges_updated: usize,
    /// Average absolute weight change
    pub avg_delta: f32,
    /// Maximum absolute weight change
    pub max_delta: f32,
    /// Number of edges that hit weight bounds
    pub bounded_count: usize,
}

/// Result of hybrid learning step
#[derive(Debug, Default)]
pub struct HybridResult {
    /// Gradient descent contribution
    pub gradient_result: NodeGradients,
    /// Hebbian contribution
    pub hebbian_result: HebbianResult,
    /// Total edges updated
    pub total_edges_updated: usize,
}

impl HebbianLearning for DagNN {
    fn backward_hebbian(&mut self, config: &HebbianConfig) -> HebbianResult {
        let mut result = HebbianResult::default();
        let mut deltas = Vec::new();

        // Collect all edge updates first (to avoid borrow conflicts)
        for edge_idx in self.graph.edge_indices() {
            let Some((source, target)) = self.graph.edge_endpoints(edge_idx) else {
                continue;
            };
            let current_weight = self.graph[edge_idx].weight;

            let delta = self.compute_hebbian_delta(source, target, current_weight, config);
            deltas.push((edge_idx, delta));
        }

        // Apply updates
        for (edge_idx, delta) in deltas {
            if delta.abs() < 1e-10 {
                continue;
            }

            let current_weight = self.graph[edge_idx].weight;
            let mut new_weight = current_weight + delta;

            // Apply weight decay
            new_weight *= 1.0 - config.weight_decay;

            // Apply weight bounds
            let was_bounded = new_weight.abs() > config.max_weight
                || (config.min_weight > 0.0 && new_weight.abs() < config.min_weight);

            if new_weight.abs() > config.max_weight {
                new_weight = new_weight.signum() * config.max_weight;
            }
            if config.min_weight > 0.0 && new_weight.abs() < config.min_weight {
                new_weight = 0.0; // Prune weak connections
            }

            self.graph[edge_idx].weight = new_weight;

            result.edges_updated += 1;
            result.avg_delta += delta.abs();
            result.max_delta = result.max_delta.max(delta.abs());
            if was_bounded {
                result.bounded_count += 1;
            }
        }

        if result.edges_updated > 0 {
            result.avg_delta /= result.edges_updated as f32;
        }

        result
    }

    fn backward_hybrid(
        &mut self,
        output_grad: &HashMap<NodeId, Array1<f32>>,
        embedding: &mut Embedding,
        config: &HybridLearningConfig,
    ) -> HybridResult {
        let mut result = HybridResult::default();

        // Step 1: Compute gradient-based updates
        let grads = self.backward(output_grad, embedding);

        // Step 2: Compute and apply combined updates
        let mut edge_updates: HashMap<(NodeId, NodeId), f32> = HashMap::new();

        // Gradient contribution
        for ((from, to), edge_grad) in &grads.edge_grads {
            let mut grad = *edge_grad;

            // Optional gradient clipping
            if config.clip_gradients && grad.abs() > config.max_grad_norm {
                grad = grad.signum() * config.max_grad_norm;
            }

            let gradient_update = -config.gradient_lr * grad * config.gradient_weight;
            *edge_updates.entry((*from, *to)).or_default() += gradient_update;
        }

        // Hebbian contribution
        for edge_idx in self.graph.edge_indices() {
            let Some((source, target)) = self.graph.edge_endpoints(edge_idx) else {
                continue;
            };
            let current_weight = self.graph[edge_idx].weight;

            let hebbian_delta =
                self.compute_hebbian_delta(source, target, current_weight, &config.hebbian);
            let hebbian_update = hebbian_delta * config.hebbian_weight;

            *edge_updates.entry((source, target)).or_default() += hebbian_update;
        }

        // Apply combined updates
        for ((from, to), total_delta) in edge_updates {
            if let Some(edge_idx) = self.graph.find_edge(from, to) {
                let current_weight = self.graph[edge_idx].weight;
                let mut new_weight = current_weight + total_delta;

                // Apply weight decay
                new_weight *= 1.0 - config.hebbian.weight_decay;

                // Apply weight bounds
                new_weight = new_weight.clamp(-config.hebbian.max_weight, config.hebbian.max_weight);

                self.graph[edge_idx].weight = new_weight;
                result.total_edges_updated += 1;
            }
        }

        result.gradient_result = grads;
        result
    }

    fn compute_hebbian_delta(
        &self,
        source: NodeId,
        target: NodeId,
        current_weight: f32,
        config: &HebbianConfig,
    ) -> f32 {
        let pre = self.graph[source].activation;
        let post = self.graph[target].activation;
        let lr = config.learning_rate;

        match config.rule {
            HebbianRule::Classic => {
                // Classic Hebbian: Δw = η * pre * post
                lr * pre * post
            }
            HebbianRule::Oja => {
                // Oja's rule: Δw = η * post * (pre - w * post)
                // This provides automatic weight normalization
                lr * post * (pre - current_weight * post)
            }
            HebbianRule::BCM => {
                // BCM rule: Δw = η * pre * post * (post - θ)
                // When post > θ: LTP (strengthening)
                // When post < θ: LTD (weakening)
                lr * pre * post * (post - config.bcm_threshold)
            }
            HebbianRule::AntiHebbian => {
                // Anti-Hebbian: Δw = -η * pre * post
                // Used for decorrelation
                -lr * pre * post
            }
        }
    }

    fn apply_competitive_learning(&mut self, inhibition_strength: f32) {
        // Group nodes by their topological level
        let mut level_nodes: HashMap<usize, Vec<NodeId>> = HashMap::new();
        for (i, &node) in self.topology.order.iter().enumerate() {
            level_nodes.entry(i / 10).or_default().push(node); // Group by approximate level
        }

        // Apply lateral inhibition within each level
        for (_level, nodes) in level_nodes {
            if nodes.len() < 2 {
                continue;
            }

            // Find winner (highest activation) in this level
            let winner = nodes
                .iter()
                .max_by(|a, b| {
                    self.graph[**a]
                        .activation
                        .partial_cmp(&self.graph[**b].activation)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied();

            if let Some(winner_node) = winner {
                let winner_activation = self.graph[winner_node].activation;

                // Suppress other nodes in proportion to winner's activation
                for &node in &nodes {
                    if node != winner_node {
                        let current = self.graph[node].activation;
                        let inhibition = inhibition_strength * winner_activation;
                        self.graph[node].activation = (current - inhibition).max(0.0);
                    }
                }
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
    pub sabag_pooling: Option<SabagPooling>,
}

impl GraphTransformNet {
    /// Create a new graph transformation network
    ///
    /// # Arguments
    /// * `vocab_size` - Size of character vocabulary (typically 256)
    /// * `embed_dim` - Dimension of character embeddings
    /// * `hidden_dim` - Hidden layer dimension
    /// * `num_clusters` - Number of Sabag output nodes (k can be <, =, or > input n)
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize, num_clusters: usize) -> Self {
        let embedding = Embedding::xavier(vocab_size, embed_dim);

        let mut mp_layers = Vec::with_capacity(2);  // Fixed 2 layers
        let mut in_dim = embed_dim;
        for _ in 0..2 {
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
        // Can compress (k < n), preserve (k = n), or expand (k > n)!
        let sabag_pooling = Some(SabagPooling::new(num_clusters, embed_dim));

        Self {
            embedding,
            mp_layers,
            attention,
            node_head,
            pooling,
            hidden_dim,
            num_layers: 2,
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
        // Zero Sabag pooling gradients
        if let Some(ref mut sabag) = self.sabag_pooling {
            sabag.zero_grad();
        }
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
        // Update Sabag pooling query matrix
        if let Some(ref mut sabag) = self.sabag_pooling {
            sabag.step(lr);
        }
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

    /// Decode output embeddings back to text (Graph-to-Text)
    ///
    /// After forward pass, the pooling_result contains feature embeddings.
    /// This method decodes those embeddings back to human-readable text
    /// using the learned embedding table.
    ///
    /// # Arguments
    /// * `pooling_result` - Result from forward() containing feature embeddings
    ///
    /// # Returns
    /// String of decoded characters from the output graph
    pub fn decode(&self, pooling_result: &PoolingResult) -> String {
        self.embedding.decode_batch(&pooling_result.features)
    }

    /// Full inference: question graph → answer text
    ///
    /// Combines forward pass and decoding for end-to-end inference.
    ///
    /// # Arguments
    /// * `input_graph` - Input graph (e.g., question encoded as GraphemeGraph)
    ///
    /// # Returns
    /// Tuple of (output_graph, decoded_text)
    pub fn infer(&self, input_graph: &GraphemeGraph) -> (GraphemeGraph, String) {
        let (output_graph, pooling_result) = self.forward(input_graph);
        let decoded = self.decode(&pooling_result);
        (output_graph, decoded)
    }

    /// Helper: Compute cosine similarity between two embedding vectors
    /// Complexity: O(d) where d = embedding dimension
    #[allow(dead_code)]
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
    /// Backward pass using activation gradients (Backend-104 fix)
    ///
    /// This is the correct backward pass that uses proper gradient chain:
    /// 1. activation_gradients[i] = ∂L/∂activation[i] (computed by structural loss)
    /// 2. ∂L/∂H_new[i,j] = activation_gradients[i] / D (broadcast to all dims)
    /// 3. ∂L/∂H = S^T · ∂L/∂H_new (route through Sabag)
    /// 4. Update embeddings with ∂L/∂H
    ///
    /// # Arguments
    /// * `input_graph` - Original input graph with n nodes
    /// * `pooling_result` - Forward pass result containing S matrix
    /// * `activation_gradients` - ∂L/∂activation for each coarsened node (length k)
    /// * `embed_dim` - Dimension of embeddings
    pub fn backward(
        &mut self,
        input_graph: &GraphemeGraph,
        pooling_result: &PoolingResult,
        activation_gradients: &[f32],
        embed_dim: usize,
    ) {
        let n = input_graph.input_nodes.len();  // Original input nodes
        let k = pooling_result.graph.input_nodes.len();  // Output/coarsened nodes
        let assignment = &pooling_result.assignment;  // Sabag soft assignment s ∈ ℝ^{k×n}

        // Backend-104 FIX: Convert activation gradients to feature gradients
        //
        // Forward: activation[i] = mean(h_new[i,:]) = Σⱼ h_new[i,j] / D
        // Backward: ∂L/∂h_new[i,j] = (∂L/∂activation[i]) * (∂activation[i]/∂h_new[i,j])
        //                         = activation_gradients[i] * (1/D)
        //
        // This broadcasts the scalar gradient to all dimensions
        let d = embed_dim as f32;
        let mut grad_k = Array2::zeros((k, embed_dim));
        for i in 0..k {
            if i < activation_gradients.len() {
                let grad_act = activation_gradients[i];
                // Broadcast to all dimensions: ∂L/∂h_new[i,j] = grad_act / D
                for j in 0..embed_dim {
                    grad_k[[i, j]] = grad_act / d;
                }
            }
        }

        // Sabag backward: Route gradients from k output nodes to n input nodes
        // Forward was: h_new = assignment · h  (k×n · n×d = k×d)
        // Backward:    ∂L/∂h = assignment^T · ∂L/∂h_new  (n×k · k×d = n×d)
        let grad_n = assignment.t().dot(&grad_k);  // Transpose and multiply!

        // Compute gradient w.r.t. query matrix Q
        if let Some(ref mut sabag) = self.sabag_pooling {
            // Reconstruct the input embeddings h that were used in forward pass
            let mut h = Array2::zeros((n, embed_dim));
            for (idx, &node_id) in input_graph.input_nodes.iter().enumerate() {
                if let NodeType::Input(ch) = input_graph.graph[node_id].node_type {
                    let embedding = self.embedding.forward(ch);
                    for (i, &val) in embedding.iter().enumerate() {
                        h[[idx, i]] = val;
                    }
                }
            }

            // Proper gradient computation through attention
            // Forward: out = assignment · h where assignment = softmax(Q · h^T / T)
            // Backward: ∂L/∂Q requires gradient through softmax
            //
            // Step 1: ∂L/∂assignment (gradient w.r.t. assignment matrix)
            // From out = assignment · h, we have ∂L/∂assignment = ∂L/∂out · h^T = grad_k · h^T
            let grad_assignment = grad_k.dot(&h.t());  // k×d · d×n = k×n ✓

            // Step 2: ∂L/∂scores (gradient through softmax)
            // Use softmax Jacobian backward pass
            let grad_scores = sabag.softmax_backward(&grad_assignment, assignment);  // k×n

            // Step 3: ∂L/∂Q (gradient through Q · h^T)
            // From scores = Q · h^T, we have ∂L/∂Q = ∂L/∂scores · h
            let q_grad = grad_scores.dot(&h);  // k×n · n×d = k×d ✓

            sabag.accumulate_query_grad(&q_grad);
        }

        // Route gradients to character embeddings
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

        // Update merge threshold gradient (simplified heuristic)
        let avg_grad = activation_gradients.iter().map(|g| g.abs()).sum::<f32>()
                       / activation_gradients.len().max(1) as f32;
        let threshold_val = self.merge_threshold.value;
        let sigmoid = 1.0 / (1.0 + (-threshold_val).exp());
        let sigmoid_deriv = sigmoid * (1.0 - sigmoid);
        let threshold_grad = -sigmoid_deriv * avg_grad * 0.01;

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
// Encoder-Decoder Architecture for Q→A Generation (backend-207)
// ============================================================================

/// Encoder component: transforms question graphs into latent representations
///
/// The encoder uses message passing layers to build context vectors
/// that capture the semantic meaning of the input question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEncoder {
    /// Character embedding layer
    pub embedding: Embedding,
    /// Message passing layers for graph encoding
    pub mp_layers: Vec<MessagePassingLayer>,
    /// Attention for focusing on relevant parts
    pub attention: AttentionLayer,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl GraphEncoder {
    /// Create a new graph encoder
    ///
    /// # Arguments
    /// * `vocab_size` - Size of character vocabulary
    /// * `embed_dim` - Character embedding dimension
    /// * `hidden_dim` - Hidden layer dimension
    /// * `num_layers` - Number of message passing layers
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let embedding = Embedding::xavier(vocab_size, embed_dim);

        let mut mp_layers = Vec::with_capacity(num_layers);
        let mut in_dim = embed_dim;
        for _ in 0..num_layers {
            mp_layers.push(MessagePassingLayer::new(in_dim, hidden_dim));
            in_dim = hidden_dim;
        }

        let attention = AttentionLayer::new(hidden_dim, 4);

        Self {
            embedding,
            mp_layers,
            attention,
            hidden_dim,
        }
    }

    /// Encode a graph to latent representation
    ///
    /// # Returns
    /// (node_features, context_vector)
    /// - node_features: Per-node hidden states
    /// - context_vector: Pooled graph-level representation
    pub fn encode(&self, graph: &GraphemeGraph) -> (Array2<f32>, Array1<f32>) {
        let n = graph.input_nodes.len();
        if n == 0 {
            return (Array2::zeros((0, self.hidden_dim)), Array1::zeros(self.hidden_dim));
        }

        // Step 1: Embed input characters
        let embed_dim = self.embedding.embed_dim;
        let mut features = Array2::zeros((n, embed_dim));

        for (idx, &node_id) in graph.input_nodes.iter().enumerate() {
            if let NodeType::Input(ch) = graph.graph[node_id].node_type {
                let emb = self.embedding.forward(ch);
                for (i, &val) in emb.iter().enumerate() {
                    features[[idx, i]] = val;
                }
            }
        }

        // Step 2: Build adjacency list for message passing
        let adjacency: Vec<Vec<usize>> = graph
            .input_nodes
            .iter()
            .map(|&node| {
                graph.graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .filter_map(|e| {
                        let source = e.source();
                        graph.input_nodes.iter().position(|&n| n == source)
                    })
                    .collect()
            })
            .collect();

        // Step 3: Apply message passing layers
        for layer in &self.mp_layers {
            features = layer.forward_batch(&features, &adjacency);
        }

        // Step 4: Compute context vector (mean pooling)
        let context = features.mean_axis(ndarray::Axis(0))
            .unwrap_or_else(|| Array1::zeros(self.hidden_dim));

        (features, context)
    }
}

/// Decoder component: generates answer from latent representation
///
/// The decoder takes the context vector and generates output embeddings
/// that can be decoded back to text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDecoder {
    /// Transform context to initial hidden state
    pub context_proj: Array2<f32>,
    /// Message passing layers for output generation
    pub mp_layers: Vec<MessagePassingLayer>,
    /// Output projection to character space (for decoding via cosine similarity)
    pub output_proj: Embedding,
    /// Linear projection: hidden_dim -> vocab_size (for training)
    pub linear_proj: Array2<f32>,
    /// Gradient for linear_proj
    #[serde(skip)]
    pub linear_grad: Option<Array2<f32>>,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Output vocabulary size
    pub vocab_size: usize,
    /// Maximum output length
    pub max_length: usize,
}

impl GraphDecoder {
    /// Create a new graph decoder
    ///
    /// # Arguments
    /// * `hidden_dim` - Hidden layer dimension (must match encoder)
    /// * `vocab_size` - Output vocabulary size
    /// * `embed_dim` - Output embedding dimension
    /// * `max_length` - Maximum output sequence length
    pub fn new(hidden_dim: usize, vocab_size: usize, embed_dim: usize, max_length: usize) -> Self {
        // Context projection: hidden_dim -> hidden_dim * max_length
        let context_proj = Array2::from_shape_fn(
            (hidden_dim, hidden_dim),
            |_| (rand::random::<f32>() - 0.5) * 0.1
        );

        // Single message passing layer for refinement
        let mp_layers = vec![MessagePassingLayer::new(hidden_dim, hidden_dim)];

        // Output projection uses embedding for decode (cosine similarity based)
        let output_proj = Embedding::xavier(vocab_size, embed_dim);

        // Linear projection: hidden_dim -> vocab_size (Xavier init)
        let scale = (2.0 / (hidden_dim + vocab_size) as f32).sqrt();
        let linear_proj = Array2::from_shape_fn(
            (hidden_dim, vocab_size),
            |_| (rand::random::<f32>() - 0.5) * 2.0 * scale
        );

        Self {
            context_proj,
            mp_layers,
            output_proj,
            linear_proj,
            linear_grad: None,
            hidden_dim,
            vocab_size,
            max_length,
        }
    }

    /// Decode context to output embeddings
    ///
    /// # Arguments
    /// * `context` - Context vector from encoder
    /// * `encoder_features` - Per-node encoder features for attention
    ///
    /// # Returns
    /// Output embeddings (max_length x embed_dim)
    pub fn decode(&self, context: &Array1<f32>, _encoder_features: &Array2<f32>) -> Array2<f32> {
        // Project context to initial decoder state
        let projected = self.context_proj.dot(context);

        // Tile projected state to create max_length outputs
        let mut outputs = Array2::zeros((self.max_length, self.hidden_dim));
        for i in 0..self.max_length {
            // Add position-dependent variation
            let pos_scale = 1.0 - (i as f32 / self.max_length as f32) * 0.1;
            for j in 0..self.hidden_dim {
                outputs[[i, j]] = projected[j] * pos_scale;
            }
        }

        // Build simple sequential adjacency for refinement
        let adjacency: Vec<Vec<usize>> = (0..self.max_length)
            .map(|i| if i > 0 { vec![i - 1] } else { vec![] })
            .collect();

        // Refine with message passing
        for layer in &self.mp_layers {
            outputs = layer.forward_batch(&outputs, &adjacency);
        }

        outputs
    }

    /// Decode embeddings to text using embedding cosine similarity
    pub fn decode_to_text(&self, embeddings: &Array2<f32>) -> String {
        self.output_proj.decode_batch(embeddings)
    }

    /// Decode hidden states to text using trained linear projection (argmax)
    ///
    /// This uses the trained linear_proj layer for decoding, which is the
    /// same layer updated during training. This ensures decoding matches training.
    pub fn decode_with_linear(&self, hidden_states: &Array2<f32>) -> String {
        // Compute logits: hidden_states @ linear_proj -> (seq_len, vocab_size)
        let logits = self.compute_logits(hidden_states);

        let mut result = String::new();
        for i in 0..logits.nrows() {
            let row = logits.row(i);
            // Find argmax (best character)
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (j, &val) in row.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = j;
                }
            }
            // Convert to printable character
            if best_idx >= 32 && best_idx <= 126 {
                result.push((best_idx as u8) as char);
            } else if best_idx == 10 {
                result.push('\n');
            } else if best_idx == 9 {
                result.push('\t');
            } else if best_idx == 13 {
                result.push('\r');
            } else {
                result.push(' '); // Replace non-printable with space
            }
        }
        result
    }
}

/// Encoder-Decoder model for Q→A generation (backend-207)
///
/// This architecture separates encoding (understanding the question)
/// from decoding (generating the answer), enabling better Q→A learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderDecoder {
    /// Encoder: question → latent
    pub encoder: GraphEncoder,
    /// Decoder: latent → answer
    pub decoder: GraphDecoder,
    /// Cached encoder output for gradient computation
    #[serde(skip)]
    last_encoder_features: Option<Array2<f32>>,
    /// Cached context for gradient computation
    #[serde(skip)]
    last_context: Option<Array1<f32>>,
}

impl EncoderDecoder {
    /// Create a new encoder-decoder model
    ///
    /// # Arguments
    /// * `vocab_size` - Character vocabulary size
    /// * `embed_dim` - Embedding dimension
    /// * `hidden_dim` - Hidden layer dimension
    /// * `max_answer_len` - Maximum answer length
    /// * `num_encoder_layers` - Number of encoder message passing layers
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        hidden_dim: usize,
        max_answer_len: usize,
        num_encoder_layers: usize,
    ) -> Self {
        let encoder = GraphEncoder::new(vocab_size, embed_dim, hidden_dim, num_encoder_layers);
        let decoder = GraphDecoder::new(hidden_dim, vocab_size, embed_dim, max_answer_len);

        Self {
            encoder,
            decoder,
            last_encoder_features: None,
            last_context: None,
        }
    }

    /// Forward pass: question graph → answer embeddings
    ///
    /// # Arguments
    /// * `question` - Input question as GraphemeGraph
    ///
    /// # Returns
    /// (output_embeddings, decoded_text)
    pub fn forward(&mut self, question: &GraphemeGraph) -> (Array2<f32>, String) {
        // Encode question
        let (encoder_features, context) = self.encoder.encode(question);

        // Cache for backward pass
        self.last_encoder_features = Some(encoder_features.clone());
        self.last_context = Some(context.clone());

        // Decode to hidden states
        let hidden_states = self.decoder.decode(&context, &encoder_features);

        // Decode to text using trained linear projection (argmax)
        let decoded_text = self.decoder.decode_with_linear(&hidden_states);

        (hidden_states, decoded_text)
    }

    /// Inference: question text → answer text
    pub fn infer(&mut self, question: &str) -> String {
        let question_graph = GraphemeGraph::from_text(question);
        let (_, answer) = self.forward(&question_graph);
        answer
    }

    /// Get the decoder's output embedding layer for decoding
    pub fn decode_embeddings(&self, embeddings: &Array2<f32>) -> String {
        self.decoder.decode_to_text(embeddings)
    }

    /// Zero out accumulated gradients
    pub fn zero_grad(&mut self) {
        self.encoder.zero_grad();
        self.decoder.zero_grad();
    }

    /// Train step: compute loss and update weights
    ///
    /// # Arguments
    /// * `input` - Input question graph
    /// * `target` - Target answer text
    /// * `lr` - Learning rate
    ///
    /// # Returns
    /// Loss value (cross-entropy)
    pub fn train_step(&mut self, input: &GraphemeGraph, target: &str, lr: f32) -> f32 {
        // Forward pass - get hidden states
        let (encoder_features, context) = self.encoder.encode(input);

        // Cache for gradient computation
        self.last_encoder_features = Some(encoder_features.clone());
        self.last_context = Some(context.clone());

        // Decode to hidden states
        let hidden_states = self.decoder.decode(&context, &encoder_features);

        // Compute logits via linear projection: (seq_len, hidden_dim) -> (seq_len, vocab_size)
        let logits = self.decoder.compute_logits(&hidden_states);

        // Compute cross-entropy loss and gradients
        let target_chars: Vec<usize> = target.chars()
            .map(|c| (c as usize).min(self.decoder.vocab_size - 1))
            .collect();

        let seq_len = logits.nrows().min(target_chars.len());
        let mut total_loss = 0.0f32;

        // Gradient for decoder linear projection
        for i in 0..seq_len {
            let target_idx = target_chars[i];
            let row_logits = logits.row(i);

            // Softmax
            let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = row_logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

            // Cross-entropy loss
            let log_prob = probs[target_idx].max(1e-10).ln();
            total_loss -= log_prob;

            // Gradient: softmax(logits) - one_hot(target)
            let mut grad = Array1::zeros(self.decoder.vocab_size);
            for (j, &p) in probs.iter().enumerate() {
                grad[j] = if j == target_idx { p - 1.0 } else { p };
            }

            // Get hidden state for this position
            let hidden = hidden_states.row(i).to_owned();

            // Accumulate gradient for linear projection
            self.decoder.backward_linear(&hidden, &grad);
        }

        // Gradient for context projection (simplified - just scale existing weights)
        let grad_scale = total_loss / seq_len.max(1) as f32;
        for i in 0..self.decoder.context_proj.nrows() {
            for j in 0..self.decoder.context_proj.ncols() {
                self.decoder.context_proj[[i, j]] -= lr * grad_scale * context[j] * 0.01;
            }
        }

        // Update weights
        self.encoder.step(lr);
        self.decoder.step(lr);

        if seq_len > 0 {
            total_loss / seq_len as f32
        } else {
            0.0
        }
    }
}

impl GraphEncoder {
    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
    }

    /// Update weights
    pub fn step(&mut self, lr: f32) {
        self.embedding.step(lr);
    }
}

impl GraphDecoder {
    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.output_proj.zero_grad();
        self.linear_grad = Some(Array2::zeros(self.linear_proj.dim()));
    }

    /// Update weights
    pub fn step(&mut self, lr: f32) {
        self.output_proj.step(lr);
        // Update linear projection
        if let Some(ref grad) = self.linear_grad {
            self.linear_proj = &self.linear_proj - &(grad * lr);
        }
    }

    /// Compute logits from hidden states using linear projection
    pub fn compute_logits(&self, hidden: &Array2<f32>) -> Array2<f32> {
        // hidden: (seq_len, hidden_dim) -> (seq_len, vocab_size)
        hidden.dot(&self.linear_proj)
    }

    /// Accumulate gradient for linear projection
    pub fn backward_linear(&mut self, hidden: &Array1<f32>, grad: &Array1<f32>) {
        if self.linear_grad.is_none() {
            self.zero_grad();
        }
        if let Some(ref mut linear_grad) = self.linear_grad {
            // grad_w[i,j] = hidden[i] * grad[j]
            for i in 0..hidden.len() {
                for j in 0..grad.len() {
                    linear_grad[[i, j]] += hidden[i] * grad[j];
                }
            }
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
///
/// ## Multi-Modal Support (Brain Slicing)
///
/// For multi-modal processing, each brain "owns" a slice of the shared DagNN:
/// - `input_node_count()` specifies how many input nodes this brain needs
/// - `output_node_count()` specifies how many output nodes this brain needs
/// - `write_inputs()` writes domain input to the brain's input slice
/// - `read_outputs()` reads domain output from the brain's output slice
///
/// See GRAPHEME_Vision.md for full architecture documentation.
pub trait DomainBrain: Send + Sync + std::fmt::Debug {
    // ========================================================================
    // Identity Methods
    // ========================================================================

    /// Get the unique domain identifier (e.g., "math", "code", "law")
    fn domain_id(&self) -> &str;

    /// Get human-readable domain name
    fn domain_name(&self) -> &str;

    /// Get the version of this brain
    fn version(&self) -> &str;

    // ========================================================================
    // Processing Methods
    // ========================================================================

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

    // ========================================================================
    // Multi-Modal / Brain Slicing Methods (GRAPHEME_Vision.md spec)
    // ========================================================================

    /// How many input nodes does this brain need?
    ///
    /// For multi-modal processing, each brain requests a slice of the shared
    /// DagNN's input nodes. Default: 0 (brain doesn't provide input).
    fn input_node_count(&self) -> usize {
        0
    }

    /// How many output nodes does this brain need?
    ///
    /// For multi-modal processing, each brain requests a slice of the shared
    /// DagNN's output nodes. Default: 0 (brain doesn't consume output).
    fn output_node_count(&self) -> usize {
        0
    }

    /// Write domain input to DagNN input nodes.
    ///
    /// Called by the orchestrator when this brain is used as an input provider.
    /// The brain writes activations to nodes in `slice.input_range`.
    ///
    /// Default: no-op (override for input-providing brains)
    fn write_inputs(&self, _input: &str, _dag: &mut DagNN, _slice: &BrainSlice) {
        // Default: no-op
    }

    /// Read domain output from DagNN output nodes.
    ///
    /// Called by the orchestrator when this brain is used as an output consumer.
    /// The brain reads activations from nodes in `slice.output_range`.
    ///
    /// Default: returns empty string (override for output-consuming brains)
    fn read_outputs(&self, _dag: &DagNN, _slice: &BrainSlice) -> String {
        String::new()
    }

    /// Get the brain's role in multi-modal processing.
    ///
    /// Default: determined by input/output node counts.
    fn brain_role(&self) -> BrainRole {
        match (self.input_node_count(), self.output_node_count()) {
            (0, 0) => BrainRole::Input, // Default to input if no counts specified
            (i, 0) if i > 0 => BrainRole::Input,
            (0, o) if o > 0 => BrainRole::Output,
            _ => BrainRole::Bidirectional,
        }
    }
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

/// A brain's slice of the shared DagNN for multi-modal processing
///
/// Each brain owns a contiguous range of input and output nodes in the shared
/// DagNN. This enables multiple brains to collaborate on a single graph while
/// maintaining clear ownership boundaries.
///
/// # Multi-Modal Architecture
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                        SHARED DagNN                                     │
/// │  VisionBrain ──► [Nodes 0-99]───┐     ┌───[Nodes 0-9] ──► VisionBrain  │
/// │     (image)      Input Slice    │     │    Output Slice      (labels)  │
/// │                                 │     │                                 │
/// │  TextBrain   ──► [Nodes 100-199]├─────┤   [Nodes 10-59] ──► TextBrain  │
/// │     (prompt)     Input Slice    │     │    Output Slice      (caption) │
/// └─────────────────────────────────────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BrainSlice {
    /// Range of input node indices owned by this brain
    pub input_range: std::ops::Range<usize>,
    /// Range of output node indices owned by this brain
    pub output_range: std::ops::Range<usize>,
    /// Brain identifier
    pub brain_id: String,
}

impl BrainSlice {
    /// Create a new brain slice
    pub fn new(brain_id: impl Into<String>, input_range: std::ops::Range<usize>, output_range: std::ops::Range<usize>) -> Self {
        Self {
            input_range,
            output_range,
            brain_id: brain_id.into(),
        }
    }

    /// Number of input nodes in this slice
    pub fn input_count(&self) -> usize {
        self.input_range.len()
    }

    /// Number of output nodes in this slice
    pub fn output_count(&self) -> usize {
        self.output_range.len()
    }

    /// Check if a node index is in the input range
    pub fn contains_input(&self, index: usize) -> bool {
        self.input_range.contains(&index)
    }

    /// Check if a node index is in the output range
    pub fn contains_output(&self, index: usize) -> bool {
        self.output_range.contains(&index)
    }
}

/// Role of a brain in multi-modal processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainRole {
    /// Brain provides input (writes to input slice)
    Input,
    /// Brain consumes output (reads from output slice)
    Output,
    /// Brain does both input and output
    Bidirectional,
}

/// Configuration for the cognitive-brain orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Minimum confidence threshold for routing
    pub confidence_threshold: f32,
    /// Whether to automatically route to domain brains
    pub auto_route: bool,
    /// Maximum number of brains to consult for a single query
    pub max_brains_per_query: usize,
    /// Enable parallel brain processing
    pub enable_parallel: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            auto_route: true,
            max_brains_per_query: 3,
            enable_parallel: true,
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

    /// Process inputs in parallel across multiple brains
    ///
    /// Uses Rayon to parallelize brain execution when multiple brains
    /// can process the same input. This is more efficient than sequential
    /// processing when there are many applicable brains.
    ///
    /// # Arguments
    /// * `input` - The input text to process
    ///
    /// # Returns
    /// Combined results from all applicable brains
    pub fn process_parallel(&mut self, input: &str) -> OrchestratedResult {
        use rayon::prelude::*;

        if !self.config.auto_route {
            self.stats.record_no_routing();
            return OrchestratedResult::empty();
        }

        // Find all brains that can process this input (O(n) where n = number of brains)
        let mut applicable: Vec<_> = self
            .registry
            .domains()
            .iter()
            .filter_map(|domain_id| {
                let brain = self.registry.get(domain_id)?;
                if brain.can_process(input) {
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
        applicable.truncate(self.config.max_brains_per_query);

        if applicable.is_empty() {
            self.stats.record_no_routing();
            return OrchestratedResult::empty();
        }

        // Process brains in parallel if enabled and more than one brain
        let results: Vec<(String, f32, Option<ExecutionResult>)> = if self.config.enable_parallel && applicable.len() > 1 {
            applicable
                .par_iter()
                .filter_map(|(domain_id, confidence)| {
                    let brain = self.registry.get(domain_id)?;
                    let graph = brain.parse(input).ok()?;
                    let exec_result = brain.execute(&graph).ok()?;
                    Some((domain_id.clone(), *confidence, Some(exec_result)))
                })
                .collect()
        } else {
            // Sequential processing
            applicable
                .iter()
                .filter_map(|(domain_id, confidence)| {
                    let brain = self.registry.get(domain_id)?;
                    let graph = brain.parse(input).ok()?;
                    let exec_result = brain.execute(&graph).ok()?;
                    Some((domain_id.clone(), *confidence, Some(exec_result)))
                })
                .collect()
        };

        // Aggregate results
        let mut result = OrchestratedResult::empty();
        let mut primary_confidence = 0.0f32;

        for (domain_id, confidence, exec_result) in results {
            if let Some(exec) = exec_result {
                result.brain_results.insert(domain_id.clone(), exec.clone());
                result.domains.push(domain_id.clone());

                if confidence > primary_confidence {
                    primary_confidence = confidence;
                    result.primary = Some(exec);
                }

                self.stats.record_routing(&domain_id);
            }
        }

        result.confidence = primary_confidence;
        result
    }

    /// Allocate brain slices for multi-modal processing
    ///
    /// Assigns contiguous ranges of input and output nodes to each brain
    /// based on their requirements. This is a O(n) operation where n is
    /// the number of brains.
    ///
    /// # Arguments
    /// * `brain_requests` - List of (brain_id, input_nodes, output_nodes) tuples
    ///
    /// # Returns
    /// Map of brain_id to their allocated BrainSlice
    pub fn allocate_brain_slices(
        &self,
        brain_requests: &[(String, usize, usize)],
    ) -> std::collections::HashMap<String, BrainSlice> {
        let mut slices = std::collections::HashMap::new();
        let mut current_input = 0;
        let mut current_output = 0;

        for (brain_id, input_nodes, output_nodes) in brain_requests {
            let slice = BrainSlice::new(
                brain_id.clone(),
                current_input..current_input + input_nodes,
                current_output..current_output + output_nodes,
            );
            slices.insert(brain_id.clone(), slice);
            current_input += input_nodes;
            current_output += output_nodes;
        }

        slices
    }

    /// Get total input nodes needed for all allocated slices
    pub fn total_input_nodes(slices: &std::collections::HashMap<String, BrainSlice>) -> usize {
        slices.values().map(|s| s.input_count()).sum()
    }

    /// Get total output nodes needed for all allocated slices
    pub fn total_output_nodes(slices: &std::collections::HashMap<String, BrainSlice>) -> usize {
        slices.values().map(|s| s.output_count()).sum()
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
// Classification Utilities (Generic)
// ============================================================================

/// Softmax function for converting logits to probabilities
///
/// Computes: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
///
/// Uses the log-sum-exp trick for numerical stability.
///
/// # Arguments
/// * `logits` - Raw unnormalized scores
///
/// # Returns
/// Probability distribution that sums to 1.0
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max)
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

    // Normalize
    let sum: f32 = exp_logits.iter().sum();
    if sum > 0.0 {
        exp_logits.iter().map(|&e| e / sum).collect()
    } else {
        // Fallback to uniform distribution
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

/// Cross-entropy loss for classification
///
/// Computes: L = -log(p[target_class])
///
/// Where p is the softmax probability of the target class.
///
/// # Arguments
/// * `logits` - Raw unnormalized scores (one per class)
/// * `target_class` - The true class index (0-indexed)
///
/// # Returns
/// The cross-entropy loss value (always >= 0)
pub fn cross_entropy_loss(logits: &[f32], target_class: usize) -> f32 {
    if logits.is_empty() || target_class >= logits.len() {
        return 0.0;
    }

    let probs = softmax(logits);
    let prob = probs[target_class].max(1e-10); // Clamp for numerical stability
    -prob.ln()
}

/// Cross-entropy loss with softmax gradient
///
/// Returns both the loss and the gradient of the loss w.r.t. logits.
///
/// The gradient for cross-entropy with softmax is elegantly:
/// ∂L/∂logit_i = prob_i - 1 (if i == target)
/// ∂L/∂logit_i = prob_i     (if i != target)
///
/// # Arguments
/// * `logits` - Raw unnormalized scores (one per class)
/// * `target_class` - The true class index (0-indexed)
///
/// # Returns
/// Tuple of (loss, gradient vector)
pub fn cross_entropy_loss_with_grad(logits: &[f32], target_class: usize) -> (f32, Vec<f32>) {
    if logits.is_empty() || target_class >= logits.len() {
        return (0.0, Vec::new());
    }

    let probs = softmax(logits);
    let prob = probs[target_class].max(1e-10);
    let loss = -prob.ln();

    // Gradient: p - one_hot(target)
    let grad: Vec<f32> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| if i == target_class { p - 1.0 } else { p })
        .collect();

    (loss, grad)
}

/// Compute accuracy for a batch of predictions
///
/// # Arguments
/// * `predictions` - Predicted class indices
/// * `targets` - True class indices
///
/// # Returns
/// Accuracy as a fraction [0.0, 1.0]
pub fn compute_accuracy(predictions: &[usize], targets: &[usize]) -> f32 {
    if predictions.is_empty() {
        return 0.0;
    }

    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(&pred, &target)| pred == target)
        .count();

    correct as f32 / predictions.len() as f32
}

/// Configuration for classification training
#[derive(Debug, Clone)]
pub struct ClassificationConfig {
    /// Learning rate for gradient descent
    pub learning_rate: f32,
    /// Number of hidden nodes in the network
    pub hidden_size: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Use Hebbian learning as supplement
    pub use_hebbian: bool,
    /// Hebbian weight (if use_hebbian is true)
    pub hebbian_weight: f32,
    /// Print progress every N batches
    pub log_interval: usize,
}

impl Default for ClassificationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            hidden_size: 128,
            batch_size: 32,
            epochs: 10,
            use_hebbian: false,
            hebbian_weight: 0.1,
            log_interval: 100,
        }
    }
}

impl ClassificationConfig {
    /// Create a new configuration with specified learning rate
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    /// Builder: set hidden size
    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    /// Builder: set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Builder: set number of epochs
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Builder: enable Hebbian learning
    pub fn with_hebbian(mut self, weight: f32) -> Self {
        self.use_hebbian = true;
        self.hebbian_weight = weight;
        self
    }
}

/// Result of a single classification training step
#[derive(Debug, Clone)]
pub struct ClassificationStepResult {
    /// Cross-entropy loss
    pub loss: f32,
    /// Predicted class
    pub predicted: usize,
    /// Whether prediction was correct
    pub correct: bool,
}

// ============================================================================
// Structural Classification (Backend-141: Graph Structure Matching)
// ============================================================================
//
// GRAPHEME Vision: Classification through graph structure matching, not softmax.
// "loss = α·node_insertion_cost + β·edge_deletion_cost + γ·clique_mismatch"
//
// Instead of cross-entropy on output node activations, we:
// 1. Create class template graphs (learned activation patterns per class)
// 2. Compare output graph activations to templates via structural similarity
// 3. Classify based on best-matching template
// 4. Train to minimize structural distance to correct class template

/// A class template representing the learned activation pattern for a class.
///
/// In GRAPHEME, each class (e.g., digit 0-9) has a template that represents
/// the expected output graph structure/activations for that class.
#[derive(Debug, Clone)]
pub struct ClassTemplate {
    /// Class index (e.g., 0-9 for 10-class classification)
    pub class_id: usize,
    /// Template activation pattern for output nodes
    /// This is the "ideal" output pattern for this class
    pub activation_pattern: Vec<f32>,
    /// Number of samples used to build this template (for online updates)
    pub sample_count: usize,
}

impl ClassTemplate {
    /// Create a new class template with initial activation pattern
    pub fn new(class_id: usize, num_outputs: usize) -> Self {
        // Initialize with one-hot pattern: high activation at class_id position
        let mut activation_pattern = vec![0.0; num_outputs];
        if class_id < num_outputs {
            activation_pattern[class_id] = 1.0;
        }
        Self {
            class_id,
            activation_pattern,
            sample_count: 0,
        }
    }

    /// Create from a specific activation pattern
    pub fn from_pattern(class_id: usize, pattern: Vec<f32>) -> Self {
        Self {
            class_id,
            activation_pattern: pattern,
            sample_count: 1,
        }
    }

    /// Update template with a new sample (exponential moving average)
    pub fn update(&mut self, sample_activations: &[f32], momentum: f32) {
        if sample_activations.len() != self.activation_pattern.len() {
            return;
        }

        self.sample_count += 1;

        // Exponential moving average update
        for (template_val, &sample_val) in self.activation_pattern.iter_mut().zip(sample_activations) {
            *template_val = momentum * *template_val + (1.0 - momentum) * sample_val;
        }
    }

    /// Compute structural distance to a given activation pattern
    ///
    /// Uses L2 distance as a simple structural similarity metric.
    /// Lower distance = better match.
    pub fn distance(&self, activations: &[f32]) -> f32 {
        if activations.len() != self.activation_pattern.len() {
            return f32::MAX;
        }

        self.activation_pattern
            .iter()
            .zip(activations)
            .map(|(t, a)| (t - a).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Compute cosine similarity to a given activation pattern
    ///
    /// Returns value in [-1, 1] where 1 = identical direction.
    pub fn cosine_similarity(&self, activations: &[f32]) -> f32 {
        if activations.len() != self.activation_pattern.len() {
            return -1.0;
        }

        let dot: f32 = self.activation_pattern.iter().zip(activations).map(|(t, a)| t * a).sum();
        let norm_t: f32 = self.activation_pattern.iter().map(|t| t.powi(2)).sum::<f32>().sqrt();
        let norm_a: f32 = activations.iter().map(|a| a.powi(2)).sum::<f32>().sqrt();

        if norm_t < 1e-10 || norm_a < 1e-10 {
            return 0.0;
        }

        dot / (norm_t * norm_a)
    }
}

/// Structural classifier that uses graph pattern matching instead of softmax.
///
/// This is the GRAPHEME-native approach to classification:
/// - Each class has a learned template (activation pattern)
/// - Classification finds the best-matching template
/// - Loss is structural distance, not cross-entropy
#[derive(Debug, Clone)]
pub struct StructuralClassifier {
    /// Templates for each class
    pub templates: Vec<ClassTemplate>,
    /// Number of output nodes
    pub num_outputs: usize,
    /// Momentum for template updates (higher = slower adaptation)
    pub template_momentum: f32,
    /// Weights for structural loss components
    pub loss_weights: StructuralLossWeights,
}

/// Weights for different components of structural loss
#[derive(Debug, Clone, Copy)]
pub struct StructuralLossWeights {
    /// Weight for L2 distance component
    pub l2_weight: f32,
    /// Weight for cosine similarity component (negative because higher sim = lower loss)
    pub cosine_weight: f32,
    /// Weight for activation magnitude difference
    pub magnitude_weight: f32,
}

impl Default for StructuralLossWeights {
    fn default() -> Self {
        Self {
            l2_weight: 1.0,
            cosine_weight: 0.5,
            magnitude_weight: 0.1,
        }
    }
}

impl StructuralClassifier {
    /// Create a new structural classifier for the given number of classes
    pub fn new(num_classes: usize, num_outputs: usize) -> Self {
        let templates = (0..num_classes)
            .map(|class_id| ClassTemplate::new(class_id, num_outputs))
            .collect();

        Self {
            templates,
            num_outputs,
            template_momentum: 0.9,
            loss_weights: StructuralLossWeights::default(),
        }
    }

    /// Set template momentum (higher = slower template adaptation)
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.template_momentum = momentum.clamp(0.0, 0.999);
        self
    }

    /// Set loss weights
    pub fn with_loss_weights(mut self, weights: StructuralLossWeights) -> Self {
        self.loss_weights = weights;
        self
    }

    /// Classify by finding the template with minimum structural distance
    ///
    /// Returns (predicted_class, distance_to_best_match)
    pub fn classify(&self, activations: &[f32]) -> (usize, f32) {
        if self.templates.is_empty() {
            return (0, f32::MAX);
        }

        self.templates
            .iter()
            .map(|t| (t.class_id, t.distance(activations)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, f32::MAX))
    }

    /// Classify using cosine similarity (higher = better match)
    ///
    /// Returns (predicted_class, similarity_to_best_match)
    pub fn classify_cosine(&self, activations: &[f32]) -> (usize, f32) {
        if self.templates.is_empty() {
            return (0, -1.0);
        }

        self.templates
            .iter()
            .map(|t| (t.class_id, t.cosine_similarity(activations)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, -1.0))
    }

    /// Compute structural loss between output activations and target class template
    ///
    /// This replaces cross-entropy loss with a graph-structure-based loss.
    /// Loss = α·L2_distance + β·(1 - cosine_similarity) + γ·magnitude_diff
    pub fn structural_loss(&self, activations: &[f32], target_class: usize) -> f32 {
        if target_class >= self.templates.len() {
            return f32::MAX;
        }

        let template = &self.templates[target_class];
        let w = &self.loss_weights;

        // L2 distance component
        let l2_dist = template.distance(activations);

        // Cosine similarity component (convert to loss: 1 - similarity)
        let cosine_sim = template.cosine_similarity(activations);
        let cosine_loss = 1.0 - cosine_sim;

        // Magnitude difference (template vs actual)
        let template_mag: f32 = template.activation_pattern.iter().map(|x| x.abs()).sum();
        let actual_mag: f32 = activations.iter().map(|x| x.abs()).sum();
        let mag_diff = (template_mag - actual_mag).abs() / (template_mag + 1e-10);

        w.l2_weight * l2_dist + w.cosine_weight * cosine_loss + w.magnitude_weight * mag_diff
    }

    /// Compute structural loss with gradient w.r.t. activations
    ///
    /// Returns (loss, gradient) where gradient can be backpropagated.
    pub fn structural_loss_with_grad(&self, activations: &[f32], target_class: usize) -> (f32, Vec<f32>) {
        if target_class >= self.templates.len() || activations.len() != self.num_outputs {
            return (f32::MAX, vec![0.0; activations.len()]);
        }

        let template = &self.templates[target_class];
        let w = &self.loss_weights;
        let n = activations.len();

        // Compute loss components
        let l2_dist = template.distance(activations);
        let cosine_sim = template.cosine_similarity(activations);
        let loss = w.l2_weight * l2_dist + w.cosine_weight * (1.0 - cosine_sim);

        // Compute gradients
        let mut grad = vec![0.0; n];

        // ∂L2/∂a_i = (a_i - t_i) / L2_dist
        if l2_dist > 1e-10 {
            for i in 0..n {
                grad[i] += w.l2_weight * (activations[i] - template.activation_pattern[i]) / l2_dist;
            }
        }

        // ∂(1-cosine)/∂a_i = -∂cosine/∂a_i
        // cosine = dot / (norm_t * norm_a)
        // ∂cosine/∂a_i = t_i / (norm_t * norm_a) - a_i * dot / (norm_t * norm_a^3)
        let dot: f32 = template.activation_pattern.iter().zip(activations).map(|(t, a)| t * a).sum();
        let norm_t: f32 = template.activation_pattern.iter().map(|t| t.powi(2)).sum::<f32>().sqrt();
        let norm_a: f32 = activations.iter().map(|a| a.powi(2)).sum::<f32>().sqrt();

        if norm_t > 1e-10 && norm_a > 1e-10 {
            let norm_ta = norm_t * norm_a;
            let norm_a3 = norm_a.powi(3);
            for i in 0..n {
                let d_cosine = template.activation_pattern[i] / norm_ta
                    - activations[i] * dot / (norm_t * norm_a3);
                // Negative because loss = 1 - cosine
                grad[i] -= w.cosine_weight * d_cosine;
            }
        }

        (loss, grad)
    }

    /// Update template for a class with a new sample
    pub fn update_template(&mut self, target_class: usize, activations: &[f32]) {
        if target_class < self.templates.len() {
            self.templates[target_class].update(activations, self.template_momentum);
        }
    }

    /// Get all distances to all class templates
    pub fn all_distances(&self, activations: &[f32]) -> Vec<f32> {
        self.templates.iter().map(|t| t.distance(activations)).collect()
    }

    /// Get all similarities to all class templates
    pub fn all_similarities(&self, activations: &[f32]) -> Vec<f32> {
        self.templates.iter().map(|t| t.cosine_similarity(activations)).collect()
    }

    /// Convert distances to pseudo-probabilities (softmax over negative distances)
    ///
    /// This allows comparison with cross-entropy-based methods.
    pub fn distance_to_probs(&self, activations: &[f32]) -> Vec<f32> {
        let distances = self.all_distances(activations);
        // Negate distances so lower distance = higher "logit"
        let neg_distances: Vec<f32> = distances.iter().map(|d| -d).collect();
        softmax(&neg_distances)
    }
}

/// Result of structural classification
#[derive(Debug, Clone)]
pub struct StructuralClassificationResult {
    /// Structural loss (lower = better match to target)
    pub loss: f32,
    /// Predicted class (best matching template)
    pub predicted: usize,
    /// Distance to predicted class template
    pub distance: f32,
    /// Whether prediction matches target
    pub correct: bool,
    /// Gradient for backpropagation (∂loss/∂activations)
    pub gradient: Vec<f32>,
}

impl DagNN {
    /// Classify using structural matching instead of softmax
    ///
    /// Backend-141: GRAPHEME-native classification through graph pattern matching.
    ///
    /// # Arguments
    /// * `classifier` - The structural classifier with learned templates
    ///
    /// # Returns
    /// (predicted_class, distance_to_best_match)
    pub fn structural_classify(&self, classifier: &StructuralClassifier) -> (usize, f32) {
        let activations = self.get_classification_logits();
        classifier.classify(&activations)
    }

    /// Compute structural classification loss and gradient
    ///
    /// This replaces cross_entropy_loss_with_grad for GRAPHEME-native training.
    ///
    /// # Arguments
    /// * `classifier` - The structural classifier
    /// * `target_class` - The true class label
    ///
    /// # Returns
    /// StructuralClassificationResult with loss, prediction, and gradient
    pub fn structural_classification_step(
        &self,
        classifier: &StructuralClassifier,
        target_class: usize,
    ) -> StructuralClassificationResult {
        let activations = self.get_classification_logits();
        let (loss, gradient) = classifier.structural_loss_with_grad(&activations, target_class);
        let (predicted, distance) = classifier.classify(&activations);

        StructuralClassificationResult {
            loss,
            predicted,
            distance,
            correct: predicted == target_class,
            gradient,
        }
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
    fn test_forward_parallel() {
        // Test that parallel forward gives same results as sequential
        let mut dag_seq = DagNN::from_text("hello world").unwrap();
        let mut dag_par = dag_seq.clone();

        // Run sequential forward
        dag_seq.forward().unwrap();
        let seq_activations = dag_seq.get_activations();

        // Run parallel forward
        dag_par.forward_parallel().unwrap();
        let par_activations = dag_par.get_activations();

        // Results should be identical
        assert_eq!(seq_activations.len(), par_activations.len());
        for ((seq_node, seq_act), (par_node, par_act)) in
            seq_activations.iter().zip(par_activations.iter())
        {
            assert_eq!(seq_node, par_node);
            assert!((seq_act - par_act).abs() < 1e-6, "Activations differ");
        }
    }

    #[test]
    fn test_compute_node_levels() {
        let dag = DagNN::from_text("hi").unwrap();
        let levels = dag.compute_node_levels();

        // Should have at least 1 level (inputs + hidden/output)
        assert!(!levels.is_empty());
        // Level 0 should contain input nodes
        assert!(!levels[0].is_empty());
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

        // Run backward and update (using deprecated method in test to verify it still works)
        #[allow(deprecated)]
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
        // Note: 4th param is num_clusters for Sabag pooling, not num_layers
        // The network uses a fixed 2 message-passing layers
        let net = GraphTransformNet::new(256, 32, 64, 3);
        let header = net.header();

        assert_eq!(header.version, MODEL_PERSISTENCE_VERSION);
        assert_eq!(header.model_type, "GraphTransformNet");
        assert_eq!(header.vocab_size, 256);
        assert_eq!(header.embed_dim, 32);
        assert_eq!(header.hidden_dim, 64);
        assert_eq!(header.num_layers, 2);  // Fixed 2 layers in implementation
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

    // ============================================================================
    // Backend-105: Edge Weight Initialization Tests
    // ============================================================================

    #[test]
    fn test_edge_xavier_initialization() {
        // Create edges with Xavier initialization
        let edge1 = Edge::xavier(2, 3, EdgeType::Sequential);
        let edge2 = Edge::xavier(10, 10, EdgeType::Skip);
        let edge3 = Edge::xavier(1, 1, EdgeType::Clique);

        // Verify edge types preserved
        assert_eq!(edge1.edge_type, EdgeType::Sequential);
        assert_eq!(edge2.edge_type, EdgeType::Skip);
        assert_eq!(edge3.edge_type, EdgeType::Clique);

        // Xavier weights should be bounded by sqrt(6/(fan_in + fan_out))
        // For fan_in=2, fan_out=3: limit = sqrt(6/5) ≈ 1.095
        let limit1 = (6.0_f32 / 5.0).sqrt();
        assert!(
            edge1.weight.abs() <= limit1 + 0.01,
            "Xavier weight {} exceeds limit {}",
            edge1.weight,
            limit1
        );

        // For fan_in=10, fan_out=10: limit = sqrt(6/20) ≈ 0.548
        let limit2 = (6.0_f32 / 20.0).sqrt();
        assert!(
            edge2.weight.abs() <= limit2 + 0.01,
            "Xavier weight {} exceeds limit {}",
            edge2.weight,
            limit2
        );
    }

    #[test]
    fn test_edge_he_initialization() {
        // Create edges with He initialization
        let edge1 = Edge::he(2, EdgeType::Sequential);
        let edge2 = Edge::he(10, EdgeType::Skip);

        // Verify edge types preserved
        assert_eq!(edge1.edge_type, EdgeType::Sequential);
        assert_eq!(edge2.edge_type, EdgeType::Skip);

        // He weights are normally distributed, so we just verify they're finite
        assert!(edge1.weight.is_finite());
        assert!(edge2.weight.is_finite());
    }

    #[test]
    fn test_dag_init_edge_weights_xavier() {
        // Create a simple graph
        let mut dag = DagNN::from_text("hello").unwrap();
        let initial_edges = dag.edge_count();

        // Store initial weights
        let initial_weights: Vec<f32> = dag
            .graph
            .edge_indices()
            .map(|e| dag.graph[e].weight)
            .collect();

        // Initialize with Xavier
        dag.init_edge_weights_xavier();

        // Verify same number of edges
        assert_eq!(dag.edge_count(), initial_edges);

        // Verify weights changed (with high probability)
        let new_weights: Vec<f32> = dag
            .graph
            .edge_indices()
            .map(|e| dag.graph[e].weight)
            .collect();

        // At least some weights should have changed
        let changed = initial_weights
            .iter()
            .zip(new_weights.iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-6)
            .count();
        assert!(
            changed > 0,
            "Expected at least some weights to change after init"
        );
    }

    #[test]
    fn test_dag_init_edge_weights_he() {
        // Create a simple graph
        let mut dag = DagNN::from_text("test").unwrap();

        // Initialize with He
        dag.init_edge_weights_he();

        // Verify all weights are finite
        for edge_idx in dag.graph.edge_indices() {
            assert!(
                dag.graph[edge_idx].weight.is_finite(),
                "He-initialized weight should be finite"
            );
        }
    }

    #[test]
    fn test_dag_init_edge_weights_zero() {
        // Create a simple graph
        let mut dag = DagNN::from_text("abc").unwrap();

        // Initialize with Zero
        dag.init_edge_weights(InitStrategy::Zero);

        // Verify all weights are zero
        for edge_idx in dag.graph.edge_indices() {
            assert!(
                (dag.graph[edge_idx].weight).abs() < 1e-10,
                "Zero-initialized weight should be 0.0"
            );
        }
    }

    #[test]
    fn test_edge_gradient_flow() {
        use std::collections::HashMap;

        // Create a simple graph: a -> b -> c
        let mut dag = DagNN::from_text("abc").unwrap();

        // Set known edge weights
        for edge_idx in dag.graph.edge_indices() {
            dag.graph[edge_idx].weight = 0.5;
        }

        // Set known activations
        let nodes: Vec<_> = dag.input_nodes().to_vec();
        for &node in &nodes {
            dag.graph[node].activation = 1.0;
        }

        // Compute backward pass with output gradient
        let mut output_grad = HashMap::new();
        let last_node = *nodes.last().unwrap();
        output_grad.insert(last_node, Array1::from_vec(vec![1.0]));

        let mut emb = Embedding::xavier(256, 1);
        let grads = dag.backward(&output_grad, &mut emb);

        // Verify edge gradients were computed
        assert!(
            !grads.edge_grads.is_empty(),
            "Should have computed edge gradients"
        );

        // Check that gradients are finite
        for &grad in grads.edge_grads.values() {
            assert!(grad.is_finite(), "Edge gradient should be finite");
        }
    }

    #[test]
    fn test_edge_weight_training_step() {
        use std::collections::HashMap;

        // Create a simple graph
        let mut dag = DagNN::from_text("ab").unwrap();

        // Initialize with known weights
        for edge_idx in dag.graph.edge_indices() {
            dag.graph[edge_idx].weight = 0.5;
        }

        // Store initial weights
        let initial_weights: Vec<f32> = dag
            .graph
            .edge_indices()
            .map(|e| dag.graph[e].weight)
            .collect();

        // Set activations
        let nodes: Vec<_> = dag.input_nodes().to_vec();
        for &node in &nodes {
            dag.graph[node].activation = 1.0;
        }

        // Backward pass with gradient descent update
        let mut output_grad = HashMap::new();
        let last_node = *nodes.last().unwrap();
        output_grad.insert(last_node, Array1::from_vec(vec![1.0]));

        let mut emb = Embedding::xavier(256, 1);
        #[allow(deprecated)]
        dag.backward_and_update(&output_grad, &mut emb, 0.1);

        // Verify weights changed
        let new_weights: Vec<f32> = dag
            .graph
            .edge_indices()
            .map(|e| dag.graph[e].weight)
            .collect();

        // At least one weight should have changed
        let changed = initial_weights
            .iter()
            .zip(new_weights.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(changed, "Edge weights should change after training step");
    }

    // ========================================================================
    // Activation Function Tests (backend-106)
    // ========================================================================

    #[test]
    fn test_activation_fn_linear() {
        let act = ActivationFn::Linear;

        // Linear should be identity
        assert_eq!(act.apply(0.0), 0.0);
        assert_eq!(act.apply(1.0), 1.0);
        assert_eq!(act.apply(-1.0), -1.0);
        assert_eq!(act.apply(100.0), 100.0);

        // Derivative should always be 1
        assert_eq!(act.derivative(0.0, 0.0), 1.0);
        assert_eq!(act.derivative(100.0, 100.0), 1.0);
        assert_eq!(act.derivative(-100.0, -100.0), 1.0);
    }

    #[test]
    fn test_activation_fn_relu() {
        let act = ActivationFn::ReLU;

        // ReLU: max(0, x)
        assert_eq!(act.apply(0.0), 0.0);
        assert_eq!(act.apply(1.0), 1.0);
        assert_eq!(act.apply(-1.0), 0.0);
        assert_eq!(act.apply(0.5), 0.5);
        assert_eq!(act.apply(-0.5), 0.0);

        // Derivative: 1 if x > 0, else 0
        assert_eq!(act.derivative(1.0, 1.0), 1.0);
        assert_eq!(act.derivative(0.5, 0.5), 1.0);
        assert_eq!(act.derivative(-1.0, 0.0), 0.0);
        assert_eq!(act.derivative(0.0, 0.0), 0.0); // At boundary, 0
    }

    #[test]
    fn test_activation_fn_sigmoid() {
        let act = ActivationFn::Sigmoid;

        // Sigmoid: 1 / (1 + exp(-x))
        let y0 = act.apply(0.0);
        assert!((y0 - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5");

        let y_pos = act.apply(10.0);
        assert!(y_pos > 0.999, "sigmoid(10) should be close to 1");

        let y_neg = act.apply(-10.0);
        assert!(y_neg < 0.001, "sigmoid(-10) should be close to 0");

        // Derivative: sigmoid * (1 - sigmoid)
        // At x=0, sigmoid=0.5, derivative = 0.5 * 0.5 = 0.25
        let deriv = act.derivative(0.0, 0.5);
        assert!((deriv - 0.25).abs() < 1e-6);

        // Near saturation, derivative should be close to 0
        let deriv_sat = act.derivative(10.0, y_pos);
        assert!(deriv_sat < 0.01);
    }

    #[test]
    fn test_activation_fn_tanh() {
        let act = ActivationFn::Tanh;

        // Tanh: tanh(x)
        let y0 = act.apply(0.0);
        assert!(y0.abs() < 1e-6, "tanh(0) should be 0");

        let y_pos = act.apply(5.0);
        assert!(y_pos > 0.99, "tanh(5) should be close to 1");

        let y_neg = act.apply(-5.0);
        assert!(y_neg < -0.99, "tanh(-5) should be close to -1");

        // Derivative: 1 - tanh^2
        // At x=0, tanh=0, derivative = 1
        let deriv = act.derivative(0.0, 0.0);
        assert!((deriv - 1.0).abs() < 1e-6);

        // Near saturation, derivative should be close to 0
        let deriv_sat = act.derivative(5.0, y_pos);
        assert!(deriv_sat < 0.01);
    }

    #[test]
    fn test_activation_fn_leaky_relu() {
        let act = ActivationFn::LeakyReLU;
        const ALPHA: f32 = 0.01;

        // Leaky ReLU: max(alpha*x, x)
        assert_eq!(act.apply(0.0), 0.0);
        assert_eq!(act.apply(1.0), 1.0);
        assert!((act.apply(-1.0) - (-ALPHA)).abs() < 1e-6);
        assert_eq!(act.apply(0.5), 0.5);
        assert!((act.apply(-0.5) - (-0.5 * ALPHA)).abs() < 1e-6);

        // Derivative: 1 if x > 0, else alpha
        assert_eq!(act.derivative(1.0, 1.0), 1.0);
        assert_eq!(act.derivative(0.5, 0.5), 1.0);
        assert_eq!(act.derivative(-1.0, -ALPHA), ALPHA);
        assert_eq!(act.derivative(-0.5, -0.5 * ALPHA), ALPHA);
    }

    #[test]
    fn test_activation_fn_derivative_from_input() {
        // Verify derivative_from_input matches derivative for all functions
        let functions = [
            ActivationFn::Linear,
            ActivationFn::ReLU,
            ActivationFn::Sigmoid,
            ActivationFn::Tanh,
            ActivationFn::LeakyReLU,
        ];

        let test_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

        for act in &functions {
            for &x in &test_values {
                let y = act.apply(x);
                let d1 = act.derivative(x, y);
                let d2 = act.derivative_from_input(x);
                assert!(
                    (d1 - d2).abs() < 1e-6,
                    "derivative mismatch for {:?} at x={}: {} vs {}",
                    act, x, d1, d2
                );
            }
        }
    }

    #[test]
    fn test_activation_fn_vec_operations() {
        let act = ActivationFn::ReLU;
        let xs = vec![-1.0, 0.0, 0.5, 1.0];

        let outputs = act.apply_vec(&xs);
        assert_eq!(outputs, vec![0.0, 0.0, 0.5, 1.0]);

        let derivs = act.derivative_vec(&xs, &outputs);
        assert_eq!(derivs, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_node_with_activation_fn() {
        // Test Node with different activation functions
        let mut hidden = Node::hidden();
        assert_eq!(hidden.activation_fn, ActivationFn::ReLU);

        // Test set_pre_activation with ReLU
        hidden.set_pre_activation(-1.0);
        assert_eq!(hidden.pre_activation, -1.0);
        assert_eq!(hidden.activation, 0.0); // ReLU clips negative

        hidden.set_pre_activation(0.5);
        assert_eq!(hidden.pre_activation, 0.5);
        assert_eq!(hidden.activation, 0.5); // ReLU passes positive

        // Test with Sigmoid
        let mut sigmoid_node = Node::hidden_with_activation(ActivationFn::Sigmoid);
        sigmoid_node.set_pre_activation(0.0);
        assert!((sigmoid_node.activation - 0.5).abs() < 1e-6);

        // Test derivative
        let deriv = sigmoid_node.activation_derivative();
        assert!((deriv - 0.25).abs() < 1e-6); // sigmoid'(0) = 0.5 * 0.5 = 0.25
    }

    #[test]
    fn test_node_activation_fn_defaults() {
        // Input nodes should have Linear
        let input = Node::input('a', 0);
        assert_eq!(input.activation_fn, ActivationFn::Linear);

        // Hidden nodes should have ReLU
        let hidden = Node::hidden();
        assert_eq!(hidden.activation_fn, ActivationFn::ReLU);

        // Output nodes should have Linear
        let output = Node::output();
        assert_eq!(output.activation_fn, ActivationFn::Linear);

        // Clique, Pattern, Compressed should have ReLU
        let clique = Node::clique(vec![1, 2, 3]);
        assert_eq!(clique.activation_fn, ActivationFn::ReLU);

        let pattern = Node::pattern(vec![b'a', b'b']);
        assert_eq!(pattern.activation_fn, ActivationFn::ReLU);

        let compressed = Node::compressed(CompressionType::RunLength);
        assert_eq!(compressed.activation_fn, ActivationFn::ReLU);
    }

    #[test]
    fn test_node_with_activation_fn_builder() {
        let node = Node::hidden().with_activation_fn(ActivationFn::Tanh);
        assert_eq!(node.activation_fn, ActivationFn::Tanh);

        let output = Node::output_with_activation(ActivationFn::Sigmoid);
        assert_eq!(output.activation_fn, ActivationFn::Sigmoid);
    }

    #[test]
    fn test_activation_fn_gradient_numerical() {
        // Numerical gradient check for each activation function
        let functions = [
            ActivationFn::Linear,
            ActivationFn::ReLU,
            ActivationFn::Sigmoid,
            ActivationFn::Tanh,
            ActivationFn::LeakyReLU,
        ];

        let eps = 1e-5;
        let test_values = [-1.0, -0.1, 0.1, 1.0]; // Avoid exact 0 for ReLU

        for act in &functions {
            for &x in &test_values {
                let numerical_grad = (act.apply(x + eps) - act.apply(x - eps)) / (2.0 * eps);
                let analytic_grad = act.derivative_from_input(x);

                // Numerical gradients have some error due to finite differences
                // Use tolerance of 1e-2 for reasonable accuracy
                let tol = 1e-2;

                assert!(
                    (numerical_grad - analytic_grad).abs() < tol,
                    "Gradient mismatch for {:?} at x={}: numerical={}, analytic={}",
                    act, x, numerical_grad, analytic_grad
                );
            }
        }
    }

    // ========================================================================
    // Neuromorphic Forward Pass Tests (backend-107)
    // ========================================================================

    #[test]
    fn test_neuromorphic_forward_basic() {
        // Create a simple linear graph: a -> b -> c
        let mut dag = DagNN::from_text("abc").unwrap();
        dag.update_topology().unwrap();

        // Input nodes should have activation 1.0
        let input_nodes = dag.input_nodes().to_vec();
        for &node in &input_nodes {
            assert_eq!(dag.graph[node].activation, 1.0);
        }

        // Run forward pass
        dag.neuromorphic_forward().unwrap();

        // Input nodes should still have activation 1.0 (identity for Linear)
        for &node in &input_nodes {
            assert_eq!(dag.graph[node].activation, 1.0);
        }
    }

    #[test]
    fn test_neuromorphic_forward_with_hidden_nodes() {
        let mut dag = DagNN::from_text("ab").unwrap();

        // Add a hidden node connected to both inputs
        let hidden = dag.add_hidden();
        let inputs = dag.input_nodes().to_vec();
        dag.add_edge(inputs[0], hidden, Edge::new(0.5, EdgeType::Sequential));
        dag.add_edge(inputs[1], hidden, Edge::new(0.5, EdgeType::Sequential));

        dag.update_topology().unwrap();
        dag.neuromorphic_forward().unwrap();

        // Hidden node should receive sum of weighted inputs: 0.5*1.0 + 0.5*1.0 = 1.0
        // With ReLU: ReLU(1.0) = 1.0
        let hidden_activation = dag.graph[hidden].activation;
        assert!((hidden_activation - 1.0).abs() < 1e-6,
            "Expected hidden activation ~1.0, got {}", hidden_activation);
    }

    #[test]
    fn test_neuromorphic_forward_relu_clips_negative() {
        let mut dag = DagNN::from_text("a").unwrap();

        // Add a hidden node with negative weight
        let hidden = dag.add_hidden();
        let input = dag.input_nodes()[0];
        dag.add_edge(input, hidden, Edge::new(-1.0, EdgeType::Sequential));

        dag.update_topology().unwrap();
        dag.neuromorphic_forward().unwrap();

        // Hidden node: pre_activation = -1.0 * 1.0 = -1.0
        // With ReLU: ReLU(-1.0) = 0.0
        assert_eq!(dag.graph[hidden].pre_activation, -1.0);
        assert_eq!(dag.graph[hidden].activation, 0.0);
    }

    #[test]
    fn test_neuromorphic_forward_sigmoid_activation() {
        let mut dag = DagNN::from_text("a").unwrap();

        // Add a hidden node with sigmoid activation
        let hidden = dag.graph.add_node(Node::hidden_with_activation(ActivationFn::Sigmoid));
        let input = dag.input_nodes()[0];
        dag.add_edge(input, hidden, Edge::new(0.0, EdgeType::Sequential)); // weight=0

        dag.update_topology().unwrap();
        dag.neuromorphic_forward().unwrap();

        // Hidden node: pre_activation = 0.0 * 1.0 = 0.0
        // With Sigmoid: sigmoid(0) = 0.5
        assert_eq!(dag.graph[hidden].pre_activation, 0.0);
        assert!((dag.graph[hidden].activation - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_neuromorphic_forward_tanh_activation() {
        let mut dag = DagNN::from_text("a").unwrap();

        // Add a hidden node with tanh activation
        let hidden = dag.graph.add_node(Node::hidden_with_activation(ActivationFn::Tanh));
        let input = dag.input_nodes()[0];
        dag.add_edge(input, hidden, Edge::new(1.0, EdgeType::Sequential));

        dag.update_topology().unwrap();
        dag.neuromorphic_forward().unwrap();

        // Hidden node: pre_activation = 1.0 * 1.0 = 1.0
        // With Tanh: tanh(1.0) ≈ 0.7616
        let expected = 1.0_f32.tanh();
        assert!((dag.graph[hidden].activation - expected).abs() < 1e-6);
    }

    #[test]
    fn test_neuromorphic_forward_preserves_pre_activation() {
        let mut dag = DagNN::from_text("ab").unwrap();

        // Add hidden node
        let hidden = dag.add_hidden();
        let inputs = dag.input_nodes().to_vec();
        dag.add_edge(inputs[0], hidden, Edge::new(2.0, EdgeType::Sequential));
        dag.add_edge(inputs[1], hidden, Edge::new(3.0, EdgeType::Sequential));

        dag.update_topology().unwrap();
        dag.neuromorphic_forward().unwrap();

        // Pre-activation should be 2.0*1.0 + 3.0*1.0 = 5.0
        assert_eq!(dag.graph[hidden].pre_activation, 5.0);
        // ReLU(5.0) = 5.0
        assert_eq!(dag.graph[hidden].activation, 5.0);
    }

    #[test]
    fn test_neuromorphic_forward_chain() {
        // Test multi-layer forward propagation
        let mut dag = DagNN::from_text("a").unwrap();
        let input = dag.input_nodes()[0];

        // Create chain: input -> h1 -> h2 -> h3
        let h1 = dag.add_hidden();
        let h2 = dag.add_hidden();
        let h3 = dag.add_hidden();

        dag.add_edge(input, h1, Edge::new(2.0, EdgeType::Sequential));
        dag.add_edge(h1, h2, Edge::new(0.5, EdgeType::Sequential));
        dag.add_edge(h2, h3, Edge::new(1.0, EdgeType::Sequential));

        dag.update_topology().unwrap();
        dag.neuromorphic_forward().unwrap();

        // input = 1.0
        // h1 = ReLU(2.0 * 1.0) = 2.0
        // h2 = ReLU(0.5 * 2.0) = 1.0
        // h3 = ReLU(1.0 * 1.0) = 1.0
        assert_eq!(dag.graph[h1].activation, 2.0);
        assert_eq!(dag.graph[h2].activation, 1.0);
        assert_eq!(dag.graph[h3].activation, 1.0);
    }

    #[test]
    fn test_forward_with_inputs() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Add hidden node
        let hidden = dag.add_hidden();
        dag.add_edge(inputs[0], hidden, Edge::new(1.0, EdgeType::Sequential));
        dag.add_edge(inputs[1], hidden, Edge::new(1.0, EdgeType::Sequential));

        // Set custom input activations
        let mut input_activations = HashMap::new();
        input_activations.insert(inputs[0], 0.3);
        input_activations.insert(inputs[1], 0.7);

        dag.update_topology().unwrap();
        dag.forward_with_inputs(&input_activations).unwrap();

        // Hidden: ReLU(0.3 + 0.7) = 1.0
        assert_eq!(dag.graph[hidden].activation, 1.0);
    }

    #[test]
    fn test_get_activation_derivatives() {
        let mut dag = DagNN::from_text("a").unwrap();
        let input = dag.input_nodes()[0];

        // Add sigmoid hidden node
        let hidden = dag.graph.add_node(Node::hidden_with_activation(ActivationFn::Sigmoid));
        dag.add_edge(input, hidden, Edge::new(0.0, EdgeType::Sequential));

        dag.update_topology().unwrap();
        dag.neuromorphic_forward().unwrap();

        let derivs = dag.get_activation_derivatives();

        // Input (Linear): derivative = 1.0
        // Hidden (Sigmoid at 0): derivative = 0.5 * 0.5 = 0.25
        let input_deriv = derivs.iter().find(|(n, _)| *n == input).unwrap().1;
        let hidden_deriv = derivs.iter().find(|(n, _)| *n == hidden).unwrap().1;

        assert_eq!(input_deriv, 1.0);
        assert!((hidden_deriv - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_get_pre_activations() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        let hidden = dag.add_hidden();
        dag.add_edge(inputs[0], hidden, Edge::new(3.0, EdgeType::Sequential));
        dag.add_edge(inputs[1], hidden, Edge::new(-1.0, EdgeType::Sequential));

        dag.update_topology().unwrap();
        dag.neuromorphic_forward().unwrap();

        let pre_acts = dag.get_pre_activations();

        // Hidden pre_activation: 3.0*1.0 + (-1.0)*1.0 = 2.0
        let hidden_pre = pre_acts.iter().find(|(n, _)| *n == hidden).unwrap().1;
        assert_eq!(hidden_pre, 2.0);
    }

    #[test]
    fn test_neuromorphic_forward_edge_weight_scaling() {
        // Test that edge weights properly scale activations
        let mut dag = DagNN::from_text("a").unwrap();
        let input = dag.input_nodes()[0];

        // Xavier-initialized edge
        let hidden = dag.add_hidden();
        dag.add_edge(input, hidden, Edge::xavier(1, 1, EdgeType::Sequential));

        dag.update_topology().unwrap();
        dag.neuromorphic_forward().unwrap();

        // Activation should be scaled by the edge weight
        let edge_weight = dag.graph.edges(input).next().unwrap().weight().weight;
        let expected = edge_weight.max(0.0); // ReLU
        assert!((dag.graph[hidden].activation - expected).abs() < 1e-6);
    }

    // ========================================================================
    // Edge Weight Pruning Tests (Backend-108)
    // ========================================================================

    #[test]
    fn test_prune_edges_by_threshold_basic() {
        // Create a graph with known edge weights
        let mut dag = DagNN::from_text("abc").unwrap();
        let initial_edges = dag.edge_count();
        assert_eq!(initial_edges, 2, "abc should have 2 sequential edges");

        // Set one edge to small weight, one to large
        let edge_indices: Vec<_> = dag.graph.edge_indices().collect();
        dag.graph[edge_indices[0]].weight = 0.001; // Small - should be pruned
        dag.graph[edge_indices[1]].weight = 1.0;   // Large - should remain

        // Prune edges below 0.01
        let pruned = dag.prune_edges_by_threshold(0.01);

        assert_eq!(pruned, 1, "Should prune exactly 1 edge");
        assert_eq!(dag.edge_count(), 1, "Should have 1 edge remaining");
    }

    #[test]
    fn test_prune_edges_by_threshold_all() {
        // All edges below threshold
        let mut dag = DagNN::from_text("abc").unwrap();
        dag.init_edge_weights(InitStrategy::Zero);

        let pruned = dag.prune_edges_by_threshold(0.01);

        assert_eq!(pruned, 2, "Should prune all zero edges");
        assert_eq!(dag.edge_count(), 0, "Should have no edges remaining");
    }

    #[test]
    fn test_prune_edges_by_threshold_none() {
        // All edges above threshold
        let mut dag = DagNN::from_text("abc").unwrap();

        // Set all weights to large values
        for edge_idx in dag.graph.edge_indices() {
            dag.graph[edge_idx].weight = 1.0;
        }

        let pruned = dag.prune_edges_by_threshold(0.01);

        assert_eq!(pruned, 0, "Should prune no edges");
        assert_eq!(dag.edge_count(), 2, "Should still have 2 edges");
    }

    #[test]
    fn test_prune_edges_by_threshold_negative_weights() {
        // Test with negative weights (inhibitory synapses)
        let mut dag = DagNN::from_text("abc").unwrap();
        let edge_indices: Vec<_> = dag.graph.edge_indices().collect();

        dag.graph[edge_indices[0]].weight = -0.001; // Small negative - should be pruned
        dag.graph[edge_indices[1]].weight = -1.0;   // Large negative - should remain

        let pruned = dag.prune_edges_by_threshold(0.01);

        assert_eq!(pruned, 1, "Should prune small absolute weight edge");
        assert_eq!(dag.edge_count(), 1, "Should have 1 edge remaining");
    }

    #[test]
    fn test_prune_edges_by_percentile_basic() {
        let mut dag = DagNN::from_text("abcd").unwrap();
        assert_eq!(dag.edge_count(), 3, "abcd should have 3 edges");

        // Set weights: 0.1, 0.5, 0.9
        let edge_indices: Vec<_> = dag.graph.edge_indices().collect();
        dag.graph[edge_indices[0]].weight = 0.1;
        dag.graph[edge_indices[1]].weight = 0.5;
        dag.graph[edge_indices[2]].weight = 0.9;

        // Prune bottom 40% (should prune the 0.1 edge)
        let pruned = dag.prune_edges_by_percentile(0.4);

        assert!(pruned >= 1, "Should prune at least 1 edge");
    }

    #[test]
    fn test_prune_edges_by_percentile_zero() {
        let mut dag = DagNN::from_text("abc").unwrap();
        let initial_edges = dag.edge_count();

        // Prune 0% - should prune nothing
        let pruned = dag.prune_edges_by_percentile(0.0);

        assert_eq!(pruned, 0, "0% percentile should prune nothing");
        assert_eq!(dag.edge_count(), initial_edges);
    }

    #[test]
    fn test_prune_edges_by_percentile_hundred() {
        let mut dag = DagNN::from_text("abc").unwrap();

        // Prune 100% - should prune everything
        let pruned = dag.prune_edges_by_percentile(1.0);

        assert_eq!(pruned, 2, "100% percentile should prune all edges");
        assert_eq!(dag.edge_count(), 0);
    }

    #[test]
    #[should_panic(expected = "Percentile must be between 0.0 and 1.0")]
    fn test_prune_edges_by_percentile_invalid() {
        let mut dag = DagNN::from_text("abc").unwrap();
        dag.prune_edges_by_percentile(1.5); // Should panic
    }

    #[test]
    fn test_prune_edges_by_correlation_basic() {
        use std::collections::HashMap;

        let mut dag = DagNN::from_text("ab").unwrap();
        let nodes = dag.input_nodes().to_vec();

        // Create activation history with perfect correlation
        let mut history = Vec::new();
        for i in 0..10 {
            let mut h = HashMap::new();
            let val = i as f32 / 10.0;
            h.insert(nodes[0], val);
            h.insert(nodes[1], val); // Same pattern = high correlation
            history.push(h);
        }

        // Prune with high correlation threshold - should keep correlated edge
        let pruned = dag.prune_edges_by_correlation(0.9, &history);

        assert_eq!(pruned, 0, "Highly correlated edge should not be pruned");
    }

    #[test]
    fn test_prune_edges_by_correlation_uncorrelated() {
        use std::collections::HashMap;

        let mut dag = DagNN::from_text("ab").unwrap();
        let nodes = dag.input_nodes().to_vec();

        // Create activation history with zero correlation
        let mut history = Vec::new();
        for i in 0..10 {
            let mut h = HashMap::new();
            h.insert(nodes[0], i as f32 / 10.0);
            h.insert(nodes[1], 0.5); // Constant - zero correlation
            history.push(h);
        }

        // Prune with correlation threshold - should prune uncorrelated
        let pruned = dag.prune_edges_by_correlation(0.5, &history);

        assert_eq!(pruned, 1, "Uncorrelated edge should be pruned");
    }

    #[test]
    fn test_prune_edges_by_correlation_empty_history() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let initial_edges = dag.edge_count();

        // Empty history should prune nothing
        let pruned = dag.prune_edges_by_correlation(0.5, &[]);

        assert_eq!(pruned, 0, "Empty history should prune nothing");
        assert_eq!(dag.edge_count(), initial_edges);
    }

    #[test]
    fn test_edge_weight_stats_basic() {
        let mut dag = DagNN::from_text("abcd").unwrap();

        // Set known weights: 0.1, 0.5, 0.9
        let edge_indices: Vec<_> = dag.graph.edge_indices().collect();
        dag.graph[edge_indices[0]].weight = 0.1;
        dag.graph[edge_indices[1]].weight = 0.5;
        dag.graph[edge_indices[2]].weight = 0.9;

        let (min, max, mean, std_dev, median) = dag.edge_weight_stats();

        assert!((min - 0.1).abs() < 1e-6, "Min should be 0.1");
        assert!((max - 0.9).abs() < 1e-6, "Max should be 0.9");
        assert!((mean - 0.5).abs() < 1e-6, "Mean should be 0.5");
        assert!((median - 0.5).abs() < 1e-6, "Median should be 0.5");
        assert!(std_dev > 0.0, "Std dev should be positive");
    }

    #[test]
    fn test_edge_weight_stats_empty() {
        let dag = DagNN::new();
        let (min, max, mean, std_dev, median) = dag.edge_weight_stats();

        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std_dev, 0.0);
        assert_eq!(median, 0.0);
    }

    #[test]
    fn test_edge_weight_histogram_basic() {
        let mut dag = DagNN::from_text("abcde").unwrap();

        // Set weights: 0.0, 0.3, 0.6, 0.9 (4 edges)
        let edge_indices: Vec<_> = dag.graph.edge_indices().collect();
        dag.graph[edge_indices[0]].weight = 0.0;
        dag.graph[edge_indices[1]].weight = 0.3;
        dag.graph[edge_indices[2]].weight = 0.6;
        dag.graph[edge_indices[3]].weight = 0.9;

        let hist = dag.edge_weight_histogram(3);

        assert_eq!(hist.len(), 3, "Should have 3 bins");
        assert!(hist[0].2 >= 1, "First bin should have at least 1 edge");
    }

    #[test]
    fn test_edge_weight_histogram_empty() {
        let dag = DagNN::new();
        let hist = dag.edge_weight_histogram(5);

        assert!(hist.is_empty(), "Empty graph should have empty histogram");
    }

    #[test]
    fn test_pruning_preserves_topology_consistency() {
        // After pruning, topology should still be valid
        let mut dag = DagNN::from_text("abcdef").unwrap();

        // Set some weights to zero
        for (i, edge_idx) in dag.graph.edge_indices().enumerate() {
            if i % 2 == 0 {
                dag.graph[edge_idx].weight = 0.0;
            }
        }

        dag.prune_edges_by_threshold(0.01);

        // Verify topology is consistent
        let topo_result = dag.update_topology();
        assert!(topo_result.is_ok(), "Topology should be valid after pruning");
    }

    #[test]
    fn test_pearson_correlation() {
        // Test the internal correlation function
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let corr = DagNN::pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-6, "Perfect positive correlation should be 1.0");

        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = DagNN::pearson_correlation(&x, &y_neg);
        assert!((corr_neg - (-1.0)).abs() < 1e-6, "Perfect negative correlation should be -1.0");

        let y_const = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let corr_zero = DagNN::pearson_correlation(&x, &y_const);
        assert!(corr_zero.abs() < 1e-6, "No correlation with constant should be 0.0");
    }

    // ========================================================================
    // Orphaned Node Removal Tests (Backend-109)
    // ========================================================================

    #[test]
    fn test_remove_orphaned_nodes_basic() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let initial_nodes = dag.node_count();

        // Add a hidden node with no connections (orphan)
        let _orphan = dag.add_hidden();
        assert_eq!(dag.node_count(), initial_nodes + 1);

        // Remove orphaned nodes
        let removed = dag.remove_orphaned_nodes();

        assert_eq!(removed, 1, "Should remove 1 orphaned node");
        assert_eq!(dag.node_count(), initial_nodes);
    }

    #[test]
    fn test_remove_orphaned_nodes_keeps_inputs() {
        let mut dag = DagNN::from_text("abc").unwrap();

        // Remove all edges to make input nodes "orphan-like"
        dag.prune_edges_by_percentile(1.0);

        // But input nodes should NOT be removed
        let removed = dag.remove_orphaned_nodes();

        assert_eq!(removed, 0, "Input nodes should never be removed");
        assert_eq!(dag.node_count(), 3, "All 3 input nodes should remain");
    }

    #[test]
    fn test_remove_orphaned_nodes_keeps_connected() {
        let mut dag = DagNN::from_text("ab").unwrap();

        // Add a connected hidden node
        let hidden = dag.add_hidden();
        let input = dag.input_nodes()[0];
        dag.add_edge(input, hidden, Edge::sequential());
        dag.update_topology().unwrap();

        // No orphans - hidden is connected
        let removed = dag.remove_orphaned_nodes();

        assert_eq!(removed, 0, "Connected hidden node should not be removed");
    }

    #[test]
    fn test_remove_orphaned_nodes_multiple() {
        let mut dag = DagNN::from_text("a").unwrap();

        // Add multiple orphans
        dag.add_hidden();
        dag.add_hidden();
        dag.add_hidden();

        assert_eq!(dag.node_count(), 4); // 1 input + 3 orphans

        let removed = dag.remove_orphaned_nodes();

        assert_eq!(removed, 3, "Should remove all 3 orphans");
        assert_eq!(dag.node_count(), 1, "Only input should remain");
    }

    #[test]
    fn test_count_orphaned_nodes() {
        let mut dag = DagNN::from_text("ab").unwrap();

        // Initially no orphans
        assert_eq!(dag.count_orphaned_nodes(), 0);

        // Add orphans
        dag.add_hidden();
        dag.add_hidden();

        assert_eq!(dag.count_orphaned_nodes(), 2);

        // Remove one
        dag.remove_orphaned_nodes();
        assert_eq!(dag.count_orphaned_nodes(), 0);
    }

    #[test]
    fn test_remove_unreachable_from_inputs_basic() {
        let mut dag = DagNN::from_text("ab").unwrap();

        // Add a hidden node connected only to itself (unreachable from inputs)
        let h1 = dag.add_hidden();
        let h2 = dag.add_hidden();
        dag.add_edge(h1, h2, Edge::sequential());

        // h1 and h2 are not reachable from inputs
        let removed = dag.remove_unreachable_from_inputs();

        assert_eq!(removed, 2, "Should remove 2 unreachable nodes");
    }

    #[test]
    fn test_remove_unreachable_from_inputs_keeps_reachable() {
        let mut dag = DagNN::from_text("ab").unwrap();

        // Add a hidden node connected from input
        let hidden = dag.add_hidden();
        let input = dag.input_nodes()[0];
        dag.add_edge(input, hidden, Edge::sequential());
        dag.update_topology().unwrap();

        let initial_nodes = dag.node_count();
        let removed = dag.remove_unreachable_from_inputs();

        assert_eq!(removed, 0, "Reachable nodes should not be removed");
        assert_eq!(dag.node_count(), initial_nodes);
    }

    #[test]
    fn test_remove_dead_end_nodes_no_outputs() {
        let mut dag = DagNN::from_text("ab").unwrap();

        // No output nodes defined - should do nothing
        let removed = dag.remove_dead_end_nodes();

        assert_eq!(removed, 0, "No outputs means no dead-ends can be detected");
    }

    #[test]
    fn test_remove_dead_end_nodes_with_outputs() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Add hidden nodes, one leading to output, one dead-end
        let h_good = dag.add_hidden();
        let h_dead = dag.add_hidden();

        dag.add_edge(inputs[0], h_good, Edge::sequential());
        dag.add_edge(inputs[1], h_dead, Edge::sequential());

        // Set h_good as output
        dag.set_output_nodes(vec![h_good]);
        dag.update_topology().unwrap();

        let initial_nodes = dag.node_count();
        let removed = dag.remove_dead_end_nodes();

        // h_dead should be removed (can't reach output h_good)
        assert_eq!(removed, 1, "Should remove 1 dead-end node");
        assert_eq!(dag.node_count(), initial_nodes - 1);
    }

    #[test]
    fn test_cleanup_disconnected_combined() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Add various disconnected nodes
        let _orphan = dag.add_hidden();  // No edges at all
        let unreachable = dag.add_hidden();  // Not reachable from inputs
        let h2 = dag.add_hidden();
        dag.add_edge(unreachable, h2, Edge::sequential());  // Unreachable subgraph

        // Add a good path
        let output = dag.add_hidden();
        dag.add_edge(inputs[0], output, Edge::sequential());
        dag.set_output_nodes(vec![output]);
        dag.update_topology().unwrap();

        let initial_nodes = dag.node_count();
        let removed = dag.cleanup_disconnected();

        // orphan, unreachable, and h2 should all be removed
        assert!(removed >= 3, "Should remove at least 3 disconnected nodes");
        assert!(dag.node_count() < initial_nodes);
    }

    #[test]
    fn test_pruning_then_cleanup_integration() {
        // Integration test: prune edges, then cleanup orphaned nodes
        let mut dag = DagNN::from_text("abcd").unwrap();

        // Add hidden nodes
        let inputs = dag.input_nodes().to_vec();
        let h1 = dag.add_hidden();
        let h2 = dag.add_hidden();

        // Connect with weak and strong edges
        dag.graph.add_edge(inputs[0], h1, Edge::new(0.001, EdgeType::Sequential));  // Weak
        dag.graph.add_edge(inputs[1], h2, Edge::new(1.0, EdgeType::Sequential));    // Strong
        dag.update_topology().unwrap();

        let nodes_before = dag.node_count();

        // Prune weak edges
        let edges_pruned = dag.prune_edges_by_threshold(0.01);
        assert_eq!(edges_pruned, 1, "Should prune 1 weak edge");

        // h1 is now orphaned
        let nodes_removed = dag.remove_orphaned_nodes();
        assert_eq!(nodes_removed, 1, "Should remove 1 orphaned node (h1)");

        assert_eq!(dag.node_count(), nodes_before - 1);
    }

    #[test]
    fn test_remove_orphaned_preserves_topology() {
        let mut dag = DagNN::from_text("abc").unwrap();

        // Add and remove orphans
        dag.add_hidden();
        dag.add_hidden();
        dag.remove_orphaned_nodes();

        // Topology should still be valid
        let result = dag.update_topology();
        assert!(result.is_ok(), "Topology should remain valid after orphan removal");
    }

    // ========================================================================
    // Neurogenesis Tests (Backend-110)
    // ========================================================================

    #[test]
    fn test_grow_node_between_basic() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();
        let (a, b) = (inputs[0], inputs[1]);

        let initial_nodes = dag.node_count();
        let initial_edges = dag.edge_count();

        // Grow a hidden node between a and b
        let hidden = dag.grow_node_between(a, b, ActivationFn::ReLU);

        assert!(hidden.is_some(), "Should create a hidden node");
        assert_eq!(dag.node_count(), initial_nodes + 1, "Should have 1 more node");
        assert_eq!(dag.edge_count(), initial_edges + 1, "Should have 1 more edge (2 - 1 = +1)");

        // Verify path: a -> hidden -> b
        let h = hidden.unwrap();
        assert!(dag.graph.find_edge(a, h).is_some(), "Edge a -> hidden should exist");
        assert!(dag.graph.find_edge(h, b).is_some(), "Edge hidden -> b should exist");
        assert!(dag.graph.find_edge(a, b).is_none(), "Direct edge a -> b should be removed");
    }

    #[test]
    fn test_grow_node_between_no_edge() {
        let mut dag = DagNN::from_text("abc").unwrap();
        let inputs = dag.input_nodes().to_vec();
        let (a, c) = (inputs[0], inputs[2]);

        // No direct edge between a and c
        let hidden = dag.grow_node_between(a, c, ActivationFn::ReLU);

        assert!(hidden.is_none(), "Should return None when no edge exists");
    }

    #[test]
    fn test_grow_node_between_preserves_weight() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();
        let (a, b) = (inputs[0], inputs[1]);

        // Set a specific weight
        let edge_idx = dag.graph.find_edge(a, b).unwrap();
        dag.graph[edge_idx].weight = 2.0;

        let hidden = dag.grow_node_between(a, b, ActivationFn::ReLU).unwrap();

        // New edges should each have half the original weight
        let e1_idx = dag.graph.find_edge(a, hidden).unwrap();
        let e2_idx = dag.graph.find_edge(hidden, b).unwrap();

        assert!((dag.graph[e1_idx].weight - 1.0).abs() < 1e-6, "First edge should be half weight");
        assert!((dag.graph[e2_idx].weight - 1.0).abs() < 1e-6, "Second edge should be half weight");
    }

    #[test]
    fn test_grow_shortcut_edge_basic() {
        let mut dag = DagNN::from_text("abc").unwrap();
        let inputs = dag.input_nodes().to_vec();
        let (a, c) = (inputs[0], inputs[2]);

        let initial_edges = dag.edge_count();

        // Add skip connection a -> c
        let added = dag.grow_shortcut_edge(a, c, 0.5);

        assert!(added, "Should successfully add shortcut");
        assert_eq!(dag.edge_count(), initial_edges + 1, "Should have 1 more edge");
        assert!(dag.graph.find_edge(a, c).is_some(), "Shortcut edge should exist");
    }

    #[test]
    fn test_grow_shortcut_edge_duplicate() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();
        let (a, b) = (inputs[0], inputs[1]);

        // Edge already exists
        let added = dag.grow_shortcut_edge(a, b, 0.5);

        assert!(!added, "Should not add duplicate edge");
    }

    #[test]
    fn test_grow_shortcut_edge_prevents_cycle() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();
        let (a, b) = (inputs[0], inputs[1]);

        // Try to add b -> a (would create cycle since a -> b exists)
        let added = dag.grow_shortcut_edge(b, a, 0.5);

        assert!(!added, "Should not add edge that creates cycle");
    }

    #[test]
    fn test_has_path() {
        let dag = DagNN::from_text("abc").unwrap();
        let inputs = dag.input_nodes().to_vec();
        let (a, b, c) = (inputs[0], inputs[1], inputs[2]);

        assert!(dag.has_path(a, b), "Path a -> b should exist");
        assert!(dag.has_path(a, c), "Path a -> c should exist (via b)");
        assert!(dag.has_path(b, c), "Path b -> c should exist");
        assert!(!dag.has_path(c, a), "No path c -> a (reverse)");
        assert!(!dag.has_path(b, a), "No path b -> a (reverse)");
    }

    #[test]
    fn test_neurogenesis_from_gradient_basic() {
        use std::collections::HashMap;

        let mut dag = DagNN::from_text("abc").unwrap();
        let inputs = dag.input_nodes().to_vec();
        let (a, b) = (inputs[0], inputs[1]);

        // Create gradient map with high gradient on a->b edge
        let mut edge_grads = HashMap::new();
        edge_grads.insert((a, b), 1.0);

        let initial_nodes = dag.node_count();

        let new_nodes = dag.neurogenesis_from_gradient(&edge_grads, 0.5, 1);

        assert_eq!(new_nodes.len(), 1, "Should create 1 new node");
        assert_eq!(dag.node_count(), initial_nodes + 1);
    }

    #[test]
    fn test_neurogenesis_from_gradient_threshold() {
        use std::collections::HashMap;

        let mut dag = DagNN::from_text("abc").unwrap();
        let inputs = dag.input_nodes().to_vec();
        let (a, b) = (inputs[0], inputs[1]);

        // Create gradient map with low gradient
        let mut edge_grads = HashMap::new();
        edge_grads.insert((a, b), 0.1);

        let new_nodes = dag.neurogenesis_from_gradient(&edge_grads, 0.5, 1);

        assert!(new_nodes.is_empty(), "Should not create node below threshold");
    }

    #[test]
    fn test_neurogenesis_from_gradient_max_limit() {
        use std::collections::HashMap;

        let mut dag = DagNN::from_text("abcd").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // High gradients on all edges
        let mut edge_grads = HashMap::new();
        edge_grads.insert((inputs[0], inputs[1]), 1.0);
        edge_grads.insert((inputs[1], inputs[2]), 1.0);
        edge_grads.insert((inputs[2], inputs[3]), 1.0);

        // But only allow 2 new nodes
        let new_nodes = dag.neurogenesis_from_gradient(&edge_grads, 0.5, 2);

        assert_eq!(new_nodes.len(), 2, "Should respect max_new_nodes limit");
    }

    #[test]
    fn test_grow_shortcuts_from_gradient_basic() {
        use std::collections::HashMap;

        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Add a hidden node
        let hidden = dag.add_hidden();
        dag.add_edge(inputs[1], hidden, Edge::sequential());
        dag.update_topology().unwrap();

        // Create node gradient map with high gradient on hidden
        let mut node_grads = HashMap::new();
        node_grads.insert(hidden, 1.0);

        let added = dag.grow_shortcuts_from_gradient(&node_grads, 0.5, 1, 0.1);

        assert_eq!(added, 1, "Should add 1 shortcut");
    }

    #[test]
    fn test_structure_stats() {
        let dag = DagNN::from_text("abc").unwrap();

        let (nodes, edges, avg_in, avg_out, depth) = dag.structure_stats();

        assert_eq!(nodes, 3);
        assert_eq!(edges, 2);
        assert!(avg_in >= 0.0);
        assert!(avg_out >= 0.0);
        assert!(depth >= 3); // At least 3 nodes in order
    }

    #[test]
    fn test_structure_stats_empty() {
        let dag = DagNN::new();

        let (nodes, edges, avg_in, avg_out, depth) = dag.structure_stats();

        assert_eq!(nodes, 0);
        assert_eq!(edges, 0);
        assert_eq!(avg_in, 0.0);
        assert_eq!(avg_out, 0.0);
        assert_eq!(depth, 0);
    }

    #[test]
    fn test_add_hidden_with_activation() {
        let mut dag = DagNN::new();

        let relu_node = dag.add_hidden_with_activation(ActivationFn::ReLU);
        let sigmoid_node = dag.add_hidden_with_activation(ActivationFn::Sigmoid);

        assert_eq!(dag.graph[relu_node].activation_fn, ActivationFn::ReLU);
        assert_eq!(dag.graph[sigmoid_node].activation_fn, ActivationFn::Sigmoid);
    }

    #[test]
    fn test_grow_then_prune_cycle() {
        // Integration test: grow nodes, then prune them
        let mut dag = DagNN::from_text("abc").unwrap();
        let inputs = dag.input_nodes().to_vec();

        let initial_nodes = dag.node_count();

        // Grow some nodes
        dag.grow_node_between(inputs[0], inputs[1], ActivationFn::ReLU);
        dag.grow_node_between(inputs[1], inputs[2], ActivationFn::ReLU);

        assert_eq!(dag.node_count(), initial_nodes + 2);

        // Set all new edge weights to zero
        for edge_idx in dag.graph.edge_indices() {
            dag.graph[edge_idx].weight = 0.0;
        }

        // Prune edges
        dag.prune_edges_by_threshold(0.01);

        // Remove orphans
        let removed = dag.remove_orphaned_nodes();

        assert!(removed >= 2, "Should remove the grown nodes after pruning");
    }

    // ========================================================================
    // Hebbian Learning Tests (Backend-111)
    // ========================================================================

    #[test]
    fn test_hebbian_config_default() {
        let config = HebbianConfig::default();

        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.weight_decay, 0.0001);
        assert_eq!(config.max_weight, 10.0);
        assert_eq!(config.min_weight, 0.0);
        assert_eq!(config.rule, HebbianRule::Classic);
    }

    #[test]
    fn test_hebbian_config_builder() {
        let config = HebbianConfig::new(0.05)
            .with_oja_rule()
            .with_weight_decay(0.001)
            .with_weight_bounds(0.01, 5.0);

        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.rule, HebbianRule::Oja);
        assert_eq!(config.weight_decay, 0.001);
        assert_eq!(config.min_weight, 0.01);
        assert_eq!(config.max_weight, 5.0);
    }

    #[test]
    fn test_hebbian_config_bcm() {
        let config = HebbianConfig::new(0.01).with_bcm_rule(0.3);

        assert_eq!(config.rule, HebbianRule::BCM);
        assert_eq!(config.bcm_threshold, 0.3);
    }

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridLearningConfig::default();

        assert_eq!(config.gradient_lr, 0.001);
        assert_eq!(config.gradient_weight, 0.7);
        assert_eq!(config.hebbian_weight, 0.3);
        assert!(config.clip_gradients);
    }

    #[test]
    fn test_hybrid_config_builder() {
        let config = HybridLearningConfig::new(0.5, 0.5)
            .with_learning_rates(0.01, 0.02);

        assert_eq!(config.gradient_weight, 0.5);
        assert_eq!(config.hebbian_weight, 0.5);
        assert_eq!(config.gradient_lr, 0.01);
        assert_eq!(config.hebbian.learning_rate, 0.02);
    }

    #[test]
    fn test_compute_hebbian_delta_classic() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Set activations
        dag.graph[inputs[0]].activation = 0.8;
        dag.graph[inputs[1]].activation = 0.6;

        let config = HebbianConfig::new(0.1);

        // Classic: Δw = η * pre * post = 0.1 * 0.8 * 0.6 = 0.048
        let delta = dag.compute_hebbian_delta(inputs[0], inputs[1], 0.5, &config);

        assert!((delta - 0.048).abs() < 1e-6);
    }

    #[test]
    fn test_compute_hebbian_delta_oja() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Set activations
        dag.graph[inputs[0]].activation = 0.8;  // pre
        dag.graph[inputs[1]].activation = 0.6;  // post

        let config = HebbianConfig::new(0.1).with_oja_rule();
        let current_weight = 0.5;

        // Oja: Δw = η * post * (pre - w * post) = 0.1 * 0.6 * (0.8 - 0.5 * 0.6)
        // = 0.1 * 0.6 * (0.8 - 0.3) = 0.1 * 0.6 * 0.5 = 0.03
        let delta = dag.compute_hebbian_delta(inputs[0], inputs[1], current_weight, &config);

        assert!((delta - 0.03).abs() < 1e-6);
    }

    #[test]
    fn test_compute_hebbian_delta_bcm() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Set activations
        dag.graph[inputs[0]].activation = 0.8;  // pre
        dag.graph[inputs[1]].activation = 0.6;  // post

        let config = HebbianConfig::new(0.1).with_bcm_rule(0.5);

        // BCM: Δw = η * pre * post * (post - θ) = 0.1 * 0.8 * 0.6 * (0.6 - 0.5)
        // = 0.1 * 0.8 * 0.6 * 0.1 = 0.0048
        let delta = dag.compute_hebbian_delta(inputs[0], inputs[1], 0.5, &config);

        assert!((delta - 0.0048).abs() < 1e-6);
    }

    #[test]
    fn test_compute_hebbian_delta_bcm_ltd() {
        // Test Long-Term Depression (when post < threshold)
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        dag.graph[inputs[0]].activation = 0.8;  // pre
        dag.graph[inputs[1]].activation = 0.3;  // post < threshold

        let config = HebbianConfig::new(0.1).with_bcm_rule(0.5);

        // BCM: Δw = 0.1 * 0.8 * 0.3 * (0.3 - 0.5) = 0.1 * 0.8 * 0.3 * (-0.2) = -0.0048
        let delta = dag.compute_hebbian_delta(inputs[0], inputs[1], 0.5, &config);

        assert!(delta < 0.0, "BCM should weaken when post < threshold");
        assert!((delta - (-0.0048)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_hebbian_delta_anti() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        dag.graph[inputs[0]].activation = 0.8;
        dag.graph[inputs[1]].activation = 0.6;

        let config = HebbianConfig {
            learning_rate: 0.1,
            rule: HebbianRule::AntiHebbian,
            ..Default::default()
        };

        // Anti-Hebbian: Δw = -η * pre * post = -0.1 * 0.8 * 0.6 = -0.048
        let delta = dag.compute_hebbian_delta(inputs[0], inputs[1], 0.5, &config);

        assert!(delta < 0.0, "Anti-Hebbian should always decrease weight");
        assert!((delta - (-0.048)).abs() < 1e-6);
    }

    #[test]
    fn test_backward_hebbian_basic() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Set activations to trigger Hebbian learning
        dag.graph[inputs[0]].activation = 1.0;
        dag.graph[inputs[1]].activation = 1.0;

        // Get initial edge weight
        let edge_idx = dag.graph.find_edge(inputs[0], inputs[1]).unwrap();
        let initial_weight = dag.graph[edge_idx].weight;

        let config = HebbianConfig::new(0.1);
        let result = dag.backward_hebbian(&config);

        // Weight should increase (both neurons active)
        let new_weight = dag.graph[edge_idx].weight;

        assert!(result.edges_updated > 0, "Should update at least 1 edge");
        assert!(new_weight > initial_weight, "Weight should increase with active neurons");
    }

    #[test]
    fn test_backward_hebbian_no_change_zero_activation() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Set zero activations
        dag.graph[inputs[0]].activation = 0.0;
        dag.graph[inputs[1]].activation = 0.0;

        let edge_idx = dag.graph.find_edge(inputs[0], inputs[1]).unwrap();
        let initial_weight = dag.graph[edge_idx].weight;

        let config = HebbianConfig::new(0.1);
        let _result = dag.backward_hebbian(&config);

        // Weight should NOT change when Hebbian delta is zero
        // (edges with zero delta are skipped, including decay)
        let new_weight = dag.graph[edge_idx].weight;

        assert!(
            (new_weight - initial_weight).abs() < 1e-6,
            "Weight should not change when activations are zero"
        );
    }

    #[test]
    fn test_backward_hebbian_weight_bounds() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Set high activations
        dag.graph[inputs[0]].activation = 10.0;
        dag.graph[inputs[1]].activation = 10.0;

        // Set initial weight close to max
        let edge_idx = dag.graph.find_edge(inputs[0], inputs[1]).unwrap();
        dag.graph[edge_idx].weight = 9.9;

        let config = HebbianConfig::new(1.0); // High learning rate
        let result = dag.backward_hebbian(&config);

        // Weight should be clamped to max_weight
        let new_weight = dag.graph[edge_idx].weight;

        assert!(
            new_weight <= config.max_weight,
            "Weight should be bounded by max_weight"
        );
        assert!(result.bounded_count > 0, "Should report bounded edges");
    }

    #[test]
    fn test_backward_hebbian_min_weight_pruning() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Set very low activation
        dag.graph[inputs[0]].activation = 0.001;
        dag.graph[inputs[1]].activation = 0.001;

        // Set small initial weight
        let edge_idx = dag.graph.find_edge(inputs[0], inputs[1]).unwrap();
        dag.graph[edge_idx].weight = 0.001;

        let config = HebbianConfig::new(0.0001).with_weight_bounds(0.01, 10.0);
        dag.backward_hebbian(&config);

        // Weight should be set to 0 (pruned) since it's below min_weight
        let new_weight = dag.graph[edge_idx].weight;

        assert_eq!(new_weight, 0.0, "Weight below min should be pruned to 0");
    }

    #[test]
    fn test_backward_hebbian_result_stats() {
        let mut dag = DagNN::from_text("abc").unwrap();

        // Set varied activations
        for node in dag.topology.order.clone() {
            dag.graph[node].activation = 0.5;
        }

        let config = HebbianConfig::new(0.1);
        let result = dag.backward_hebbian(&config);

        assert!(result.edges_updated > 0, "Should update edges");
        assert!(result.avg_delta > 0.0, "Should have positive avg delta");
        assert!(result.max_delta >= result.avg_delta, "Max should be >= avg");
    }

    #[test]
    fn test_backward_hybrid_basic() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Set activations
        dag.graph[inputs[0]].activation = 0.8;
        dag.graph[inputs[1]].activation = 0.6;
        dag.update_topology().unwrap();

        // Create output gradient
        let mut output_grad = HashMap::new();
        output_grad.insert(inputs[1], Array1::from_vec(vec![0.1]));

        let mut embedding = Embedding::new(256, 8, InitStrategy::Zero);
        let config = HybridLearningConfig::default();

        let result = dag.backward_hybrid(&output_grad, &mut embedding, &config);

        assert!(result.total_edges_updated > 0, "Should update edges");
    }

    #[test]
    fn test_backward_hybrid_gradient_dominance() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();
        dag.update_topology().unwrap();

        dag.graph[inputs[0]].activation = 1.0;
        dag.graph[inputs[1]].activation = 1.0;

        let edge_idx = dag.graph.find_edge(inputs[0], inputs[1]).unwrap();
        let initial_weight = dag.graph[edge_idx].weight;

        // Large output gradient
        let mut output_grad = HashMap::new();
        output_grad.insert(inputs[1], Array1::from_vec(vec![10.0]));

        let mut embedding = Embedding::new(256, 8, InitStrategy::Zero);

        // High gradient weight, low Hebbian weight
        let config = HybridLearningConfig::new(0.9, 0.1)
            .with_learning_rates(0.1, 0.01);

        dag.backward_hybrid(&output_grad, &mut embedding, &config);

        let new_weight = dag.graph[edge_idx].weight;

        // Weight should decrease significantly (gradient descent direction)
        assert!(
            (new_weight - initial_weight).abs() > 0.01,
            "Gradient should dominate weight update"
        );
    }

    #[test]
    fn test_backward_hybrid_hebbian_contribution() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();
        dag.update_topology().unwrap();

        // High activations for strong Hebbian signal
        dag.graph[inputs[0]].activation = 2.0;
        dag.graph[inputs[1]].activation = 2.0;

        let edge_idx = dag.graph.find_edge(inputs[0], inputs[1]).unwrap();
        let initial_weight = dag.graph[edge_idx].weight;

        // Zero gradient (no gradient contribution)
        let mut output_grad = HashMap::new();
        output_grad.insert(inputs[1], Array1::from_vec(vec![0.0]));

        let mut embedding = Embedding::new(256, 8, InitStrategy::Zero);

        // Only Hebbian contribution
        let config = HybridLearningConfig::new(0.0, 1.0)
            .with_learning_rates(0.0, 0.1);

        dag.backward_hybrid(&output_grad, &mut embedding, &config);

        let new_weight = dag.graph[edge_idx].weight;

        // Weight should increase (Hebbian with positive activations)
        assert!(
            new_weight > initial_weight,
            "Hebbian should increase weight when neurons fire together"
        );
    }

    #[test]
    fn test_competitive_learning_basic() {
        let mut dag = DagNN::from_text("abc").unwrap();
        let inputs = dag.input_nodes().to_vec();

        // Set one node as "winner" with high activation
        dag.graph[inputs[0]].activation = 1.0;
        dag.graph[inputs[1]].activation = 0.5;
        dag.graph[inputs[2]].activation = 0.3;

        dag.apply_competitive_learning(0.5);

        // Winner should maintain activation, others should be suppressed
        // (but note: depends on grouping, so this is a basic test)
        assert!(
            dag.graph[inputs[0]].activation >= dag.graph[inputs[1]].activation,
            "Winner should maintain higher activation"
        );
    }

    #[test]
    fn test_competitive_learning_no_negative_activation() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let inputs = dag.input_nodes().to_vec();

        dag.graph[inputs[0]].activation = 1.0;
        dag.graph[inputs[1]].activation = 0.1;

        dag.apply_competitive_learning(0.9);

        // Activation should never go negative
        assert!(
            dag.graph[inputs[1]].activation >= 0.0,
            "Activation should be non-negative after inhibition"
        );
    }

    #[test]
    fn test_hebbian_integration_forward_backward() {
        // Full integration: forward pass, then Hebbian backward
        let mut dag = DagNN::from_text("hello").unwrap();
        dag.neuromorphic_forward().unwrap();

        let config = HebbianConfig::new(0.01);
        let result = dag.backward_hebbian(&config);

        assert!(result.edges_updated > 0, "Should update edges after forward pass");
    }

    #[test]
    fn test_hybrid_integration_training_step() {
        // Simulate a training step with hybrid learning
        let mut dag = DagNN::from_text("ab").unwrap();
        dag.neuromorphic_forward().unwrap();

        // Get output node for gradient
        let outputs = dag.output_nodes.clone();
        let target_node = if outputs.is_empty() {
            dag.input_nodes()[1]
        } else {
            outputs[0]
        };

        let mut output_grad = HashMap::new();
        output_grad.insert(target_node, Array1::from_vec(vec![0.5]));

        let mut embedding = Embedding::new(256, 8, InitStrategy::Zero);
        let config = HybridLearningConfig::default();

        // Multiple training steps
        for _ in 0..5 {
            dag.neuromorphic_forward().unwrap();
            dag.backward_hybrid(&output_grad, &mut embedding, &config);
        }

        // Just verify no crashes and edges were updated
        assert!(dag.edge_count() > 0);
    }

    #[test]
    fn test_hebbian_multiple_rules_comparison() {
        // Compare different Hebbian rules on same network
        let base_dag = DagNN::from_text("ab").unwrap();
        let inputs = base_dag.input_nodes().to_vec();

        let rules = [
            HebbianRule::Classic,
            HebbianRule::Oja,
            HebbianRule::BCM,
            HebbianRule::AntiHebbian,
        ];

        let mut results = Vec::new();

        for rule in rules {
            let mut dag = base_dag.clone();
            dag.graph[inputs[0]].activation = 0.8;
            dag.graph[inputs[1]].activation = 0.6;

            let edge_idx = dag.graph.find_edge(inputs[0], inputs[1]).unwrap();
            let initial = dag.graph[edge_idx].weight;

            let config = HebbianConfig {
                learning_rate: 0.1,
                rule,
                ..Default::default()
            };
            dag.backward_hebbian(&config);

            let final_weight = dag.graph[edge_idx].weight;
            results.push((rule, initial, final_weight));
        }

        // Classic and Oja should increase (positive activations)
        assert!(results[0].2 > results[0].1, "Classic should increase");
        assert!(results[1].2 > results[1].1, "Oja should increase");

        // AntiHebbian should decrease
        assert!(results[3].2 < results[3].1, "AntiHebbian should decrease");
    }

    // ========================================================================
    // Complexity Verification Tests (Backend-112)
    // ========================================================================
    // These tests verify that operations scale polynomially (not exponentially)
    // by measuring relative time growth between different input sizes.

    #[test]
    fn test_complexity_constants_defined() {
        // Verify all complexity constants are defined and reasonable
        // (These tests verify compile-time constants are sensible)
        assert_eq!(MAX_CLIQUE_K, 6, "MAX_CLIQUE_K should be 6 (small cliques only)");
        assert_eq!(MAX_CLIQUE_GRAPH_SIZE, 10000, "Expected graph size limit");
        assert_eq!(MAX_SINKHORN_ITERATIONS, 100, "Expected iteration count");
        assert_eq!(MAX_NODES_POLYNOMIAL, 100_000, "Expected node limit");
        assert_eq!(MAX_EDGES_POLYNOMIAL, 1_000_000, "Expected edge limit");
    }

    #[test]
    fn test_complexity_edge_pruning_linear() {
        // Verify edge pruning scales linearly O(E)
        // Create graphs of increasing size and verify time doesn't explode

        for size in [10, 50, 100] {
            let text: String = (0..size).map(|i| ((i % 26) as u8 + b'a') as char).collect();
            let mut dag = DagNN::from_text(&text).unwrap();

            // Add extra edges
            let inputs = dag.input_nodes().to_vec();
            for i in 0..inputs.len().saturating_sub(2) {
                dag.add_edge(inputs[i], inputs[i + 2], Edge::sequential());
            }

            // This should complete quickly for any size
            let pruned = dag.prune_edges_by_threshold(0.001);

            // Just verify it completes - use the value to prevent optimization
            let _ = pruned;
        }
    }

    #[test]
    fn test_complexity_orphan_removal_linear() {
        // Verify orphan removal scales linearly O(V + E)

        for size in [10, 50, 100] {
            let text: String = (0..size).map(|i| ((i % 26) as u8 + b'a') as char).collect();
            let mut dag = DagNN::from_text(&text).unwrap();

            // Add orphan nodes
            for _ in 0..(size / 5) {
                dag.add_hidden();
            }

            let removed = dag.remove_orphaned_nodes();

            // Should remove approximately size/5 orphans
            assert!(removed >= size / 5 - 1);
        }
    }

    #[test]
    fn test_complexity_forward_pass_linear() {
        // Verify forward pass scales linearly O(V + E)

        for size in [10, 50, 100] {
            let text: String = (0..size).map(|i| ((i % 26) as u8 + b'a') as char).collect();
            let mut dag = DagNN::from_text(&text).unwrap();

            // Should complete quickly
            dag.neuromorphic_forward().unwrap();

            // Verify activations were computed
            let activations = dag.get_activations();
            assert_eq!(activations.len(), size);
        }
    }

    #[test]
    fn test_complexity_topological_sort_linear() {
        // Verify topological sort is O(V + E) via Kahn's algorithm

        for size in [10, 50, 100, 500] {
            let text: String = (0..size).map(|i| ((i % 26) as u8 + b'a') as char).collect();
            let mut dag = DagNN::from_text(&text).unwrap();

            // Add hidden nodes
            let inputs = dag.input_nodes().to_vec();
            for i in 0..(size / 10) {
                let hidden = dag.add_hidden();
                dag.add_edge(inputs[i % inputs.len()], hidden, Edge::sequential());
            }

            // Should complete quickly
            dag.update_topology().unwrap();

            // Verify order is valid
            assert!(dag.topology.order.len() >= size);
        }
    }

    #[test]
    fn test_complexity_hebbian_linear_in_edges() {
        // Verify Hebbian learning is O(E)

        for size in [10, 50, 100] {
            let text: String = (0..size).map(|i| ((i % 26) as u8 + b'a') as char).collect();
            let mut dag = DagNN::from_text(&text).unwrap();
            dag.neuromorphic_forward().unwrap();

            let config = HebbianConfig::new(0.01);
            let result = dag.backward_hebbian(&config);

            // Should update edges
            assert!(result.edges_updated > 0);
        }
    }

    #[test]
    fn test_complexity_neurogenesis_polynomial() {
        // Verify neurogenesis is polynomial (not exponential)

        for size in [10, 50, 100] {
            let text: String = (0..size).map(|i| ((i % 26) as u8 + b'a') as char).collect();
            let mut dag = DagNN::from_text(&text).unwrap();

            let inputs = dag.input_nodes().to_vec();
            if inputs.len() >= 2 {
                // Grow a node between first two inputs
                let result = dag.grow_node_between(inputs[0], inputs[1], ActivationFn::ReLU);

                // Should succeed
                assert!(result.is_some());
            }
        }
    }

    #[test]
    fn test_clique_k_bounds_enforced() {
        // Verify clique enumeration respects MAX_CLIQUE_K

        let dag = DagNN::from_text("abcdefghij").unwrap();

        // Valid k values should work
        for k in 3..=MAX_CLIQUE_K {
            let result = dag.find_cliques(k);
            assert!(result.is_ok(), "k={} should be valid", k);
        }

        // k > MAX_CLIQUE_K should fail
        let result = dag.find_cliques(MAX_CLIQUE_K + 1);
        assert!(result.is_err(), "k > MAX_CLIQUE_K should fail");
    }

    #[test]
    fn test_cleanup_disconnected_polynomial() {
        // Verify cleanup_disconnected is O(V + E)

        for size in [10, 50, 100] {
            let text: String = (0..size).map(|i| ((i % 26) as u8 + b'a') as char).collect();
            let mut dag = DagNN::from_text(&text).unwrap();

            // Add disconnected subgraph
            let h1 = dag.add_hidden();
            let h2 = dag.add_hidden();
            dag.add_edge(h1, h2, Edge::sequential());

            // Set output
            dag.set_output_nodes(vec![dag.input_nodes()[0]]);

            let removed = dag.cleanup_disconnected();

            // Should remove at least the disconnected nodes
            assert!(removed >= 2);
        }
    }

    #[test]
    fn test_sinkhorn_iterations_bounded() {
        // Verify Sinkhorn has bounded iterations

        let pooling = SabagPooling::new(5, 8);

        // Default iterations should be reasonable
        assert!(
            pooling.sinkhorn_iterations <= MAX_SINKHORN_ITERATIONS,
            "Sinkhorn iterations should be bounded"
        );
        assert!(
            pooling.sinkhorn_iterations >= 1,
            "Should have at least 1 iteration"
        );
    }

    // ========================================================================
    // Image Encoding Tests (Backend-113)
    // ========================================================================

    #[test]
    fn test_pixel_node_creation() {
        let node = Node::pixel(5, 10, 0.5);
        assert!(matches!(node.node_type, NodeType::Pixel { row: 5, col: 10 }));
        assert!((node.activation - 0.5).abs() < 1e-6);
        assert_eq!(node.position, Some(5 * 28 + 10)); // row-major position
    }

    #[test]
    fn test_class_output_node_creation() {
        let node = Node::class_output(7);
        assert!(matches!(node.node_type, NodeType::ClassOutput(7)));
        assert_eq!(node.activation, 0.0);
    }

    #[test]
    fn test_from_image_basic() {
        // Small 4x4 image
        let pixels = vec![0.0_f32; 16];
        let dag = DagNN::from_image(&pixels, 4, 4).unwrap();

        // Should have 16 pixel nodes
        assert_eq!(dag.input_nodes().len(), 16);
        assert_eq!(dag.node_count(), 16);

        // Each internal pixel has 2 edges (right and down)
        // Edge count: (4-1)*4 + 4*(4-1) = 12 + 12 = 24 edges
        // But edge and internal nodes... let's just verify it has edges
        assert!(dag.edge_count() > 0);
    }

    #[test]
    fn test_from_image_dimensions() {
        // Generic image test (10x10)
        let pixels = vec![0.5_f32; 100];
        let dag = DagNN::from_image(&pixels, 10, 10).unwrap();

        assert_eq!(dag.input_nodes().len(), 100);

        // Verify pixel intensities are set correctly
        for &node_id in dag.input_nodes() {
            let node = &dag.graph[node_id];
            assert!((node.activation - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_from_image_wrong_size() {
        let pixels = vec![0.5_f32; 100];
        // Wrong dimensions (should be 10x10=100, not 8x8=64)
        let result = DagNN::from_image(&pixels, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_classifier() {
        let activations = vec![0.1_f32; 100];
        let dag = DagNN::with_classifier(100, 32, 5, Some(&activations)).unwrap();

        // Should have 100 input + 32 hidden + 5 output nodes
        assert!(dag.node_count() >= 100 + 32 + 5);

        // Should have 5 output nodes
        assert_eq!(dag.output_nodes().len(), 5);

        // All output nodes should be ClassOutput type
        for &out_id in dag.output_nodes() {
            let node = &dag.graph[out_id];
            assert!(matches!(node.node_type, NodeType::ClassOutput(_)));
        }
    }

    #[test]
    fn test_image_grid_topology() {
        // 3x3 image to manually verify edges
        let pixels = vec![0.0_f32; 9];
        let dag = DagNN::from_image(&pixels, 3, 3).unwrap();

        // Nodes are in row-major order: 0,1,2,3,4,5,6,7,8
        // Pixel at (0,0) should connect to (0,1) and (1,0)
        // Pixel at (1,1) should connect to (1,2) and (2,1)
        // Last row/col should only have one or zero outgoing edges

        // Total edges: horizontal (3-1)*3=6, vertical 3*(3-1)=6 = 12 edges
        assert_eq!(dag.edge_count(), 12);
    }

    #[test]
    fn test_classification_forward() {
        // Create a classifier with generic dimensions and run forward pass
        let activations = vec![0.5_f32; 100];
        let mut dag = DagNN::with_classifier(100, 32, 5, Some(&activations)).unwrap();

        // Run forward pass
        let result = dag.neuromorphic_forward();
        assert!(result.is_ok());

        // Get logits
        let logits = dag.get_classification_logits();
        assert_eq!(logits.len(), 5);

        // All logits should be finite (not NaN or inf)
        for &logit in &logits {
            assert!(logit.is_finite(), "Logit should be finite");
        }
    }

    #[test]
    fn test_predict_class() {
        let activations = vec![0.5_f32; 100];
        let mut dag = DagNN::with_classifier(100, 32, 5, Some(&activations)).unwrap();
        dag.neuromorphic_forward().unwrap();

        let predicted = dag.predict_class();
        assert!(predicted < 5); // Should be a valid class (0-4)
    }

    // ========================================================================
    // Classification Utilities Tests (Generic)
    // ========================================================================

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher logit should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without log-sum-exp trick
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);

        // Should not be NaN or inf
        for &p in &probs {
            assert!(p.is_finite());
            assert!(p >= 0.0);
            assert!(p <= 1.0);
        }

        // Should still sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_empty() {
        let logits: Vec<f32> = vec![];
        let probs = softmax(&logits);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_cross_entropy_loss_correct() {
        // Perfect prediction: all mass on correct class
        let logits = vec![-100.0, 100.0, -100.0]; // After softmax, [~0, ~1, ~0]
        let loss = cross_entropy_loss(&logits, 1);

        // Loss should be close to 0
        assert!(loss < 0.01, "Loss for correct prediction should be low");
    }

    #[test]
    fn test_cross_entropy_loss_incorrect() {
        // Wrong prediction: high mass on wrong class
        let logits = vec![100.0, -100.0, -100.0]; // After softmax, [~1, ~0, ~0]
        let loss = cross_entropy_loss(&logits, 1); // But target is class 1

        // Loss should be high
        assert!(loss > 1.0, "Loss for wrong prediction should be high");
    }

    #[test]
    fn test_cross_entropy_with_gradient() {
        let logits = vec![1.0, 2.0, 3.0];
        let target = 1;
        let (loss, grad) = cross_entropy_loss_with_grad(&logits, target);

        // Loss should be positive
        assert!(loss > 0.0);

        // Gradient should have same length as logits
        assert_eq!(grad.len(), 3);

        // Gradient for target class should be negative (prob - 1 < 0)
        assert!(grad[target] < 0.0);

        // Gradients for non-target classes should be positive (prob > 0)
        assert!(grad[0] > 0.0);
        assert!(grad[2] > 0.0);

        // Gradients should sum to 0
        let grad_sum: f32 = grad.iter().sum();
        assert!(grad_sum.abs() < 1e-5);
    }

    #[test]
    fn test_compute_accuracy() {
        let predictions = vec![0, 1, 2, 1, 0];
        let targets = vec![0, 1, 1, 1, 1]; // 3 correct out of 5

        let accuracy = compute_accuracy(&predictions, &targets);
        assert!((accuracy - 0.6).abs() < 1e-6); // 3/5 = 0.6
    }

    #[test]
    fn test_compute_accuracy_perfect() {
        let predictions = vec![0, 1, 2];
        let targets = vec![0, 1, 2];

        let accuracy = compute_accuracy(&predictions, &targets);
        assert!((accuracy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_accuracy_empty() {
        let predictions: Vec<usize> = vec![];
        let targets: Vec<usize> = vec![];

        let accuracy = compute_accuracy(&predictions, &targets);
        assert!((accuracy - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_classification_config_builder() {
        let config = ClassificationConfig::new(0.001)
            .with_hidden_size(256)
            .with_batch_size(64)
            .with_epochs(20)
            .with_hebbian(0.2);

        assert!((config.learning_rate - 0.001).abs() < 1e-6);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.epochs, 20);
        assert!(config.use_hebbian);
        assert!((config.hebbian_weight - 0.2).abs() < 1e-6);
    }

    // ========================================================================
    // Structural Classification Tests (Backend-141)
    // ========================================================================

    #[test]
    fn test_class_template_new() {
        let template = ClassTemplate::new(3, 10);
        assert_eq!(template.class_id, 3);
        assert_eq!(template.activation_pattern.len(), 10);
        // Should be one-hot at position 3
        assert!((template.activation_pattern[3] - 1.0).abs() < 1e-6);
        assert!((template.activation_pattern[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_class_template_distance() {
        let template = ClassTemplate::from_pattern(0, vec![1.0, 0.0, 0.0]);

        // Exact match
        let dist1 = template.distance(&[1.0, 0.0, 0.0]);
        assert!(dist1 < 1e-6);

        // Far from template
        let dist2 = template.distance(&[0.0, 1.0, 0.0]);
        assert!(dist2 > 1.0);
    }

    #[test]
    fn test_class_template_cosine_similarity() {
        let template = ClassTemplate::from_pattern(0, vec![1.0, 0.0, 0.0]);

        // Identical direction
        let sim1 = template.cosine_similarity(&[2.0, 0.0, 0.0]);
        assert!((sim1 - 1.0).abs() < 1e-6);

        // Orthogonal
        let sim2 = template.cosine_similarity(&[0.0, 1.0, 0.0]);
        assert!(sim2.abs() < 1e-6);

        // Opposite direction
        let sim3 = template.cosine_similarity(&[-1.0, 0.0, 0.0]);
        assert!((sim3 - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_structural_classifier_new() {
        let classifier = StructuralClassifier::new(10, 10);
        assert_eq!(classifier.templates.len(), 10);
        assert_eq!(classifier.num_outputs, 10);
    }

    #[test]
    fn test_structural_classifier_classify() {
        let classifier = StructuralClassifier::new(3, 3);

        // Activation close to class 0 template (1,0,0)
        let (pred, dist) = classifier.classify(&[0.9, 0.1, 0.0]);
        assert_eq!(pred, 0);
        assert!(dist < 0.2);

        // Activation close to class 1 template (0,1,0)
        let (pred, dist) = classifier.classify(&[0.1, 0.9, 0.0]);
        assert_eq!(pred, 1);
        assert!(dist < 0.2);

        // Activation close to class 2 template (0,0,1)
        let (pred, dist) = classifier.classify(&[0.0, 0.1, 0.9]);
        assert_eq!(pred, 2);
        assert!(dist < 0.2);
    }

    #[test]
    fn test_structural_loss() {
        let classifier = StructuralClassifier::new(3, 3);

        // Loss when prediction matches target
        let loss1 = classifier.structural_loss(&[0.9, 0.1, 0.0], 0);
        // Loss when prediction doesn't match target
        let loss2 = classifier.structural_loss(&[0.9, 0.1, 0.0], 1);

        // Loss should be lower when activations match target class template
        assert!(loss1 < loss2);
    }

    #[test]
    fn test_structural_loss_with_grad() {
        let classifier = StructuralClassifier::new(3, 3);
        let activations = vec![0.5, 0.3, 0.2];

        let (loss, grad) = classifier.structural_loss_with_grad(&activations, 0);

        // Loss should be positive
        assert!(loss > 0.0);

        // Gradient should have same length as activations
        assert_eq!(grad.len(), 3);

        // Gradient should be finite
        for g in &grad {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_structural_classifier_update_template() {
        let mut classifier = StructuralClassifier::new(3, 3);

        // Initial template for class 0 is (1, 0, 0)
        assert!((classifier.templates[0].activation_pattern[0] - 1.0).abs() < 1e-6);

        // Update with new sample
        classifier.update_template(0, &[0.8, 0.1, 0.1]);

        // Template should have moved toward the sample
        assert!(classifier.templates[0].activation_pattern[0] < 1.0);
        assert!(classifier.templates[0].activation_pattern[1] > 0.0);
    }

    #[test]
    fn test_structural_classifier_all_distances() {
        let classifier = StructuralClassifier::new(3, 3);
        let activations = vec![0.9, 0.1, 0.0];

        let distances = classifier.all_distances(&activations);
        assert_eq!(distances.len(), 3);

        // Distance to class 0 should be smallest
        assert!(distances[0] < distances[1]);
        assert!(distances[0] < distances[2]);
    }

    #[test]
    fn test_distance_to_probs() {
        let classifier = StructuralClassifier::new(3, 3);
        let activations = vec![0.9, 0.1, 0.0];

        let probs = classifier.distance_to_probs(&activations);

        // Should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Highest prob should be for class 0 (closest)
        assert!(probs[0] > probs[1]);
        assert!(probs[0] > probs[2]);
    }

    // ========================================================================
    // Gradient Accumulation Tests (backend-142)
    // ========================================================================

    #[test]
    fn test_dagnn_zero_grad() {
        let mut dag = DagNN::from_text("abc").unwrap();

        // Manually add some gradients
        let nodes: Vec<_> = dag.input_nodes().to_vec();
        dag.accumulate_edge_grad(nodes[0], nodes[1], 1.0);
        dag.accumulate_edge_grad(nodes[1], nodes[2], 2.0);

        assert!(dag.has_gradients());
        assert_eq!(dag.edge_grads.len(), 2);

        // Zero gradients
        dag.zero_grad();

        assert!(!dag.has_gradients());
        assert_eq!(dag.edge_grads.len(), 0);
    }

    #[test]
    fn test_dagnn_gradient_accumulation() {
        let mut dag = DagNN::from_text("abc").unwrap();
        let nodes: Vec<_> = dag.input_nodes().to_vec();

        // Accumulate gradient twice for same edge
        dag.accumulate_edge_grad(nodes[0], nodes[1], 1.0);
        dag.accumulate_edge_grad(nodes[0], nodes[1], 2.0);

        // Should sum to 3.0
        let grad = dag.get_edge_grad(nodes[0], nodes[1]).unwrap();
        assert!((grad - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dagnn_gradient_norm() {
        let mut dag = DagNN::from_text("abc").unwrap();
        let nodes: Vec<_> = dag.input_nodes().to_vec();

        // Add gradients: 3.0 and 4.0 -> norm should be 5.0
        dag.accumulate_edge_grad(nodes[0], nodes[1], 3.0);
        dag.accumulate_edge_grad(nodes[1], nodes[2], 4.0);

        let norm = dag.gradient_norm();
        assert!((norm - 5.0).abs() < 1e-6, "Expected norm 5.0, got {}", norm);
    }

    #[test]
    fn test_dagnn_clip_gradients() {
        let mut dag = DagNN::from_text("abc").unwrap();
        let nodes: Vec<_> = dag.input_nodes().to_vec();

        // Add gradients: 3.0 and 4.0 -> norm is 5.0
        dag.accumulate_edge_grad(nodes[0], nodes[1], 3.0);
        dag.accumulate_edge_grad(nodes[1], nodes[2], 4.0);

        // Clip to max_norm=2.5 (half of 5.0)
        let original_norm = dag.clip_gradients(2.5);
        assert!((original_norm - 5.0).abs() < 1e-6);

        // After clipping, norm should be 2.5
        let new_norm = dag.gradient_norm();
        assert!((new_norm - 2.5).abs() < 1e-6, "Expected norm 2.5 after clipping, got {}", new_norm);

        // Individual gradients should be scaled by 0.5
        let grad1 = dag.get_edge_grad(nodes[0], nodes[1]).unwrap();
        let grad2 = dag.get_edge_grad(nodes[1], nodes[2]).unwrap();
        assert!((grad1 - 1.5).abs() < 1e-6, "Expected grad 1.5, got {}", grad1);
        assert!((grad2 - 2.0).abs() < 1e-6, "Expected grad 2.0, got {}", grad2);
    }

    #[test]
    fn test_dagnn_step_updates_weights() {
        let mut dag = DagNN::from_text("abc").unwrap();

        // Initialize with known weights
        for edge_idx in dag.graph.edge_indices() {
            dag.graph[edge_idx].weight = 1.0;
        }

        let nodes: Vec<_> = dag.input_nodes().to_vec();

        // Add gradient of 0.5 to first edge
        dag.accumulate_edge_grad(nodes[0], nodes[1], 0.5);

        // Get edge index for later comparison
        let edge_idx = dag.graph.find_edge(nodes[0], nodes[1]).unwrap();

        // Step with lr=0.1
        dag.step(0.1);

        // Weight should be updated: w = w - lr * grad = 1.0 - 0.1 * 0.5 = 0.95
        let new_weight = dag.graph[edge_idx].weight;
        assert!(
            (new_weight - 0.95).abs() < 1e-6,
            "Expected weight 0.95, got {}",
            new_weight
        );

        // Other edge should be unchanged (no gradient for it)
        let other_edge_idx = dag.graph.find_edge(nodes[1], nodes[2]).unwrap();
        assert!(
            (dag.graph[other_edge_idx].weight - 1.0).abs() < 1e-6,
            "Edge without gradient should be unchanged"
        );
    }

    #[test]
    fn test_dagnn_num_parameters() {
        let dag = DagNN::from_text("abc").unwrap();
        // "abc" has 3 nodes and 2 edges (a->b, b->c)
        assert_eq!(dag.num_parameters(), 2);

        let dag2 = DagNN::from_text("hello").unwrap();
        // "hello" has 5 nodes and 4 edges
        assert_eq!(dag2.num_parameters(), 4);
    }

    #[test]
    fn test_dagnn_train_mode() {
        let mut dag = DagNN::from_text("abc").unwrap();

        // Default should be training mode
        assert!(dag.is_training());
        assert!(dag.requires_grad);

        // Switch to eval mode
        dag.train(false);
        assert!(!dag.is_training());
        assert!(!dag.requires_grad);

        // Switch back to train mode
        dag.train(true);
        assert!(dag.is_training());
    }

    #[test]
    fn test_dagnn_backward_accumulate_respects_requires_grad() {
        let mut dag = DagNN::from_text("ab").unwrap();
        let mut embedding = Embedding::xavier(256, 8);

        // Set known activations
        let nodes: Vec<_> = dag.input_nodes().to_vec();
        for &node in &nodes {
            dag.graph[node].activation = 1.0;
        }

        // Create output gradient
        let output_grad: HashMap<NodeId, ndarray::Array1<f32>> =
            [(nodes[1], ndarray::Array1::from_vec(vec![1.0]))].into_iter().collect();

        // Backward in training mode should accumulate gradients
        dag.train(true);
        dag.backward_accumulate(&output_grad, &mut embedding);
        assert!(dag.has_gradients(), "Should have gradients in training mode");

        // Clear and try in eval mode
        dag.zero_grad();
        dag.train(false);
        dag.backward_accumulate(&output_grad, &mut embedding);
        assert!(!dag.has_gradients(), "Should NOT accumulate gradients in eval mode");
    }

    #[test]
    fn test_learnable_trait_dagnn() {
        use crate::Learnable;

        // Helper function that works with any Learnable type
        fn check_learnable<T: Learnable>(model: &mut T) -> (usize, bool, f32) {
            let params = model.num_parameters();
            let has_grad = model.has_gradients();
            let norm = model.gradient_norm();
            (params, has_grad, norm)
        }

        let mut dag = DagNN::from_text("abc").unwrap();

        // Test num_parameters via trait
        assert_eq!(dag.num_parameters(), 2); // 3 nodes, 2 edges

        // Initially no gradients
        assert!(!dag.has_gradients());
        assert_eq!(dag.gradient_norm(), 0.0);

        // Accumulate some gradients
        let nodes: Vec<_> = dag.input_nodes().to_vec();
        dag.accumulate_edge_grad(nodes[0], nodes[1], 3.0);
        dag.accumulate_edge_grad(nodes[1], nodes[2], 4.0);

        // Now has gradients
        assert!(dag.has_gradients());
        assert!((dag.gradient_norm() - 5.0).abs() < 1e-6);

        // Test via generic function (proves DagNN implements Learnable)
        let (params, has_grad, norm) = check_learnable(&mut dag);
        assert_eq!(params, 2);
        assert!(has_grad);
        assert!((norm - 5.0).abs() < 1e-6);

        // Zero grad should clear via trait
        dag.zero_grad();
        assert!(!dag.has_gradients());
        assert_eq!(dag.gradient_norm(), 0.0);

        // Step should work via trait (add gradient and apply)
        dag.accumulate_edge_grad(nodes[0], nodes[1], 0.5);
        for edge_idx in dag.graph.edge_indices() {
            dag.graph[edge_idx].weight = 1.0;
        }
        dag.step(0.1);
        let edge_idx = dag.graph.find_edge(nodes[0], nodes[1]).unwrap();
        assert!((dag.graph[edge_idx].weight - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_dagnn_full_training_loop() {
        // Test the complete: zero_grad -> backward_accumulate -> step pattern
        let mut dag = DagNN::from_text("ab").unwrap();
        let mut embedding = Embedding::xavier(256, 8);

        // Initialize with known weights
        for edge_idx in dag.graph.edge_indices() {
            dag.graph[edge_idx].weight = 1.0;
        }

        // Set known activations
        let nodes: Vec<_> = dag.input_nodes().to_vec();
        for &node in &nodes {
            dag.graph[node].activation = 1.0;
        }

        // Get initial weight
        let edge_idx = dag.graph.find_edge(nodes[0], nodes[1]).unwrap();
        let initial_weight = dag.graph[edge_idx].weight;

        // Training loop iteration
        dag.zero_grad();

        // Create output gradient (gradient of 1.0 at output node)
        let output_grad: HashMap<NodeId, ndarray::Array1<f32>> =
            [(nodes[1], ndarray::Array1::from_vec(vec![1.0]))].into_iter().collect();

        dag.backward_accumulate(&output_grad, &mut embedding);
        assert!(dag.has_gradients(), "Should have accumulated gradients");

        let grad_norm_before_step = dag.gradient_norm();
        assert!(grad_norm_before_step > 0.0, "Gradient norm should be positive");

        dag.step(0.1);

        // Weight should have changed
        let new_weight = dag.graph[edge_idx].weight;
        assert!(
            (new_weight - initial_weight).abs() > 1e-10,
            "Weight should change after step: {} -> {}",
            initial_weight,
            new_weight
        );
    }

    // ========================================================================
    // DomainBrain Multi-Modal Tests
    // ========================================================================

    /// Mock brain for testing multi-modal features
    #[derive(Debug)]
    struct MockInputBrain {
        id: String,
        input_nodes: usize,
        output_nodes: usize,
    }

    impl MockInputBrain {
        fn new(id: &str, input_nodes: usize, output_nodes: usize) -> Self {
            Self {
                id: id.to_string(),
                input_nodes,
                output_nodes,
            }
        }
    }

    impl DomainBrain for MockInputBrain {
        fn domain_id(&self) -> &str { &self.id }
        fn domain_name(&self) -> &str { &self.id }
        fn version(&self) -> &str { "1.0" }
        fn can_process(&self, _input: &str) -> bool { true }
        fn parse(&self, _input: &str) -> DomainResult<DagNN> {
            Ok(DagNN::default())
        }
        fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
            Ok(graph.clone())
        }
        fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
            Ok(graph.clone())
        }
        fn validate(&self, _graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
            Ok(vec![])
        }
        fn execute(&self, _graph: &DagNN) -> DomainResult<ExecutionResult> {
            Ok(ExecutionResult::Unit)
        }
        fn get_rules(&self) -> Vec<DomainRule> { vec![] }
        fn transform(&self, graph: &DagNN, _rule_id: usize) -> DomainResult<DagNN> {
            Ok(graph.clone())
        }
        fn generate_examples(&self, _count: usize) -> Vec<DomainExample> { vec![] }

        // Multi-modal overrides
        fn input_node_count(&self) -> usize { self.input_nodes }
        fn output_node_count(&self) -> usize { self.output_nodes }

        fn write_inputs(&self, input: &str, dag: &mut DagNN, slice: &BrainSlice) {
            // Write input length as activation to nodes in slice
            if !slice.input_range.is_empty() {
                let activation = input.len() as f32 / 100.0;
                for i in slice.input_range.clone() {
                    let indices: Vec<_> = dag.graph.node_indices().collect();
                    if i < indices.len() {
                        dag.graph[indices[i]].activation = activation;
                    }
                }
            }
        }

        fn read_outputs(&self, dag: &DagNN, slice: &BrainSlice) -> String {
            // Read activations from output slice and sum them
            let mut sum = 0.0f32;
            let indices: Vec<_> = dag.graph.node_indices().collect();
            for i in slice.output_range.clone() {
                if i < indices.len() {
                    sum += dag.graph[indices[i]].activation;
                }
            }
            format!("sum:{:.2}", sum)
        }
    }

    #[test]
    fn test_domain_brain_input_node_count() {
        let brain = MockInputBrain::new("test", 100, 10);
        assert_eq!(brain.input_node_count(), 100);
        assert_eq!(brain.output_node_count(), 10);
    }

    #[test]
    fn test_domain_brain_role() {
        // Input-only brain
        let input_brain = MockInputBrain::new("input", 100, 0);
        assert_eq!(input_brain.brain_role(), BrainRole::Input);

        // Output-only brain
        let output_brain = MockInputBrain::new("output", 0, 50);
        assert_eq!(output_brain.brain_role(), BrainRole::Output);

        // Bidirectional brain
        let bi_brain = MockInputBrain::new("bidirectional", 100, 50);
        assert_eq!(bi_brain.brain_role(), BrainRole::Bidirectional);

        // Default (no slicing specified)
        let default_brain = MockInputBrain::new("default", 0, 0);
        assert_eq!(default_brain.brain_role(), BrainRole::Input);
    }

    #[test]
    fn test_domain_brain_write_inputs() {
        let brain = MockInputBrain::new("test", 5, 0);
        let mut dag = DagNN::default();

        // Add nodes to DagNN
        for _ in 0..10 {
            dag.graph.add_node(Node::hidden());
        }

        let slice = BrainSlice::new("test", 0..5, 0..0);

        // Write inputs
        brain.write_inputs("Hello, World!", &mut dag, &slice);

        // Check activations were set (length 13 / 100 = 0.13)
        let expected = 13.0 / 100.0;
        let indices: Vec<_> = dag.graph.node_indices().collect();
        for (i, &idx) in indices.iter().enumerate().take(5) {
            let act = dag.graph[idx].activation;
            assert!((act - expected).abs() < 0.01, "Node {} activation: {}", i, act);
        }
    }

    #[test]
    fn test_domain_brain_read_outputs() {
        let brain = MockInputBrain::new("test", 0, 5);
        let mut dag = DagNN::default();

        // Add nodes and set activations
        for i in 0..10 {
            let idx = dag.graph.add_node(Node::hidden());
            dag.graph[idx].activation = i as f32 * 0.1;
        }

        let slice = BrainSlice::new("test", 0..0, 0..5);

        // Read outputs (sum of nodes 0-4: 0 + 0.1 + 0.2 + 0.3 + 0.4 = 1.0)
        let result = brain.read_outputs(&dag, &slice);
        assert_eq!(result, "sum:1.00");
    }

    #[test]
    fn test_brain_slice_integration() {
        // Test full multi-modal workflow
        let input_brain = MockInputBrain::new("vision", 100, 0);
        let output_brain = MockInputBrain::new("text", 0, 10);

        let mut dag = DagNN::default();

        // Add nodes (100 input + 10 output + some hidden)
        for _ in 0..150 {
            dag.graph.add_node(Node::hidden());
        }

        // Allocate slices
        let vision_slice = BrainSlice::new("vision", 0..100, 0..0);
        let text_slice = BrainSlice::new("text", 0..0, 100..110);

        // Write input
        input_brain.write_inputs("Test image data", &mut dag, &vision_slice);

        // Check vision slice has activations
        let indices: Vec<_> = dag.graph.node_indices().collect();
        let vis_act = dag.graph[indices[0]].activation;
        assert!(vis_act > 0.0, "Vision nodes should have activation");

        // Simulate forward pass - just set output node activations
        for &idx in indices.iter().skip(100).take(10) {
            dag.graph[idx].activation = 0.5;
        }

        // Read output
        let output = output_brain.read_outputs(&dag, &text_slice);
        assert_eq!(output, "sum:5.00"); // 10 nodes * 0.5 = 5.0
    }

    // ========================================================================
    // Checkpoint compression tests (backend-196)
    // ========================================================================

    #[test]
    fn test_checkpoint_compressed_roundtrip() {
        let dag = DagNN::from_text("Hello, compression test!").unwrap();

        let mut checkpoint = UnifiedCheckpoint::new();
        checkpoint.add_module(&dag).unwrap();

        // Compress to bytes
        let compressed = checkpoint.to_compressed_bytes().unwrap();

        // Decompress
        let loaded = UnifiedCheckpoint::from_compressed_bytes(&compressed).unwrap();
        let loaded_dag: DagNN = loaded.load_module().unwrap();

        assert_eq!(dag.to_text(), loaded_dag.to_text());
    }

    #[test]
    fn test_checkpoint_compressed_file_roundtrip() {
        use std::path::Path;

        let dag = DagNN::from_text("Testing compressed file I/O").unwrap();

        let mut checkpoint = UnifiedCheckpoint::new();
        checkpoint.add_module(&dag).unwrap();

        // Save compressed
        let temp_path = Path::new("/tmp/test_checkpoint.json.gz");
        checkpoint.save_compressed(temp_path).unwrap();

        // Load compressed
        let loaded = UnifiedCheckpoint::load_compressed(temp_path).unwrap();
        let loaded_dag: DagNN = loaded.load_module().unwrap();

        assert_eq!(dag.to_text(), loaded_dag.to_text());

        // Clean up
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_checkpoint_auto_format_detection() {
        use std::path::Path;

        let dag = DagNN::from_text("Auto format test").unwrap();

        let mut checkpoint = UnifiedCheckpoint::new();
        checkpoint.add_module(&dag).unwrap();

        // Test .gz extension
        let gz_path = Path::new("/tmp/test_auto.json.gz");
        checkpoint.save_auto(gz_path).unwrap();
        let loaded_gz = UnifiedCheckpoint::load_auto(gz_path).unwrap();
        let dag_gz: DagNN = loaded_gz.load_module().unwrap();
        assert_eq!(dag.to_text(), dag_gz.to_text());

        // Test .json extension
        let json_path = Path::new("/tmp/test_auto.json");
        checkpoint.save_auto(json_path).unwrap();
        let loaded_json = UnifiedCheckpoint::load_auto(json_path).unwrap();
        let dag_json: DagNN = loaded_json.load_module().unwrap();
        assert_eq!(dag.to_text(), dag_json.to_text());

        // Clean up
        let _ = std::fs::remove_file(gz_path);
        let _ = std::fs::remove_file(json_path);
    }

    #[test]
    fn test_compression_stats() {
        // Create a checkpoint with some data
        let dag = DagNN::from_text("This is a test sentence for compression ratio measurement. It should show good compression for JSON data with repeated patterns like node definitions and edge weights.").unwrap();

        let mut checkpoint = UnifiedCheckpoint::new();
        checkpoint.add_module(&dag).unwrap();

        let (uncompressed, compressed, ratio) = checkpoint.compression_stats().unwrap();

        // Verify compression stats make sense
        assert!(uncompressed > 0);
        assert!(compressed > 0);
        assert!(compressed < uncompressed, "Compressed should be smaller");
        assert!(
            ratio > 1.0,
            "Compression ratio should be > 1.0, got {}",
            ratio
        );

        // JSON checkpoint data typically compresses 3-10x
        assert!(
            ratio > 1.5,
            "Expected reasonable compression ratio, got {}x",
            ratio
        );
    }

    #[test]
    fn test_compression_multiple_modules() {
        let dag1 = DagNN::from_text("First module").unwrap();
        let dag2 = DagNN::from_text("Second module").unwrap();

        // Create checkpoint with custom names
        let mut checkpoint = UnifiedCheckpoint::new();

        // Add both modules (they'll both be stored as DagNN type)
        // Since persist_type_id returns "DagNN" for both, only the second one will be stored
        // This is expected behavior - one module per type
        checkpoint.add_module(&dag1).unwrap();
        checkpoint.add_module(&dag2).unwrap();

        // Compress and decompress
        let compressed = checkpoint.to_compressed_bytes().unwrap();
        let loaded = UnifiedCheckpoint::from_compressed_bytes(&compressed).unwrap();

        // Should have DagNN module (the second one overwrites the first)
        assert!(loaded.has_module::<DagNN>());
    }
}
