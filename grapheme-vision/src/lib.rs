//! # grapheme-vision
//!
//! Vision Brain: Image-to-graph embedding for GRAPHEME (no CNN).
//!
//! This crate provides:
//! - `RawImage` - Raw image representation (any size, grayscale or RGB)
//! - `VisionBrain` - Hierarchical feature extraction (blob detection, edge detection)
//! - `ClassificationBrain` - Output graph to class label conversion
//! - `ImageClassificationModel` - Generic end-to-end image classification pipeline
//! - Deterministic: same image = same graph
//!
//! ## GRAPHEME Vision Principle
//!
//! ```text
//! Image → VisionBrain → Input Graph → GRAPHEME Core → Output Graph → ClassificationBrain → Class
//!       (deterministic)                  (learns)                    (structural matching)
//! ```
//!
//! No CNN. No learned feature extraction. Pure signal processing to graph.
//!
//! ## Generic Architecture
//!
//! VisionBrain handles ANY image size. All components are fully generic:
//! - `RawImage` - Any dimensions, grayscale or RGB
//! - `FeatureConfig` - Configurable feature extraction parameters
//! - `ClassificationConfig` - Any number of classes
//!
//! Dataset-specific configurations belong in training crates, not here.

use grapheme_brain_common::{ActivatedNode, BaseDomainBrain, DomainConfig, TextNormalizer,
                             GraphAutoencoder, LatentGraph, AutoencoderError};
use grapheme_core::{
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule, ExecutionResult, NodeType,
    ValidationIssue, NodeId, Edge, EdgeType,
};
use std::collections::HashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// Neural API Stubs (for vision-specific neural components)
// ============================================================================

/// Initialization strategy for neural network weights (GRAPHEME protocol).
///
/// GRAPHEME uses **Dynamic Xavier** - weights are reinitialized when topology changes.
/// LeakyReLU (α=0.01) is used everywhere instead of standard ReLU.
/// Adam optimizer is the default for training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitStrategy {
    /// Dynamic Xavier/Glorot initialization - recomputes when topology changes (GRAPHEME protocol)
    DynamicXavier,
    #[deprecated(since = "0.1.0", note = "Use DynamicXavier per GRAPHEME protocol")]
    /// Static Xavier - deprecated, use DynamicXavier instead
    Xavier,
    /// He initialization (for LeakyReLU networks)
    He,
    /// Random uniform initialization
    Uniform,
}

impl Default for InitStrategy {
    /// Returns `DynamicXavier` per GRAPHEME protocol (optimized with Adam)
    fn default() -> Self {
        Self::DynamicXavier
    }
}

/// Simple embedding layer for mapping discrete inputs to vectors.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Embedding weights
    weights: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding layer
    pub fn new(vocab_size: usize, embed_dim: usize, _init: InitStrategy) -> Self {
        let weights = vec![0.0; vocab_size * embed_dim];
        Self { vocab_size, embed_dim, weights }
    }

    /// Get embedding for an index
    pub fn forward(&self, index: usize) -> Vec<f32> {
        if index >= self.vocab_size {
            return vec![0.0; self.embed_dim];
        }
        let start = index * self.embed_dim;
        let end = start + self.embed_dim;
        self.weights[start..end].to_vec()
    }

    /// Embed the entire input graph
    pub fn embed(&self, graph: &DagNN) -> Vec<f32> {
        let mut result = vec![0.0; self.embed_dim];
        for &node in graph.input_nodes() {
            if let NodeType::Input(ch) = graph.graph[node].node_type {
                let idx = ch as usize % self.vocab_size;
                let emb = self.forward(idx);
                for (i, v) in emb.iter().enumerate() {
                    result[i] += v;
                }
            }
        }
        result
    }

    /// Get the number of parameters
    pub fn num_params(&self) -> usize {
        self.weights.len()
    }
}

/// Backward pass information for gradient computation.
#[derive(Debug, Clone, Default)]
pub struct BackwardPass {
    /// Gradient w.r.t. embeddings
    pub embedding_grads: Vec<f32>,
    /// Gradient w.r.t. graph edges
    pub edge_grads: HashMap<(NodeId, NodeId), f32>,
}

impl BackwardPass {
    /// Create a new backward pass
    pub fn new() -> Self {
        Self::default()
    }
}

/// Template for a class in structural classification
#[derive(Debug, Clone)]
pub struct ClassTemplate {
    /// Graph representation
    pub graph: DagNN,
    /// Activation pattern for this class
    pub activation_pattern: Vec<f32>,
    /// Number of samples seen
    pub sample_count: usize,
}

impl ClassTemplate {
    /// Create a new class template with given size
    pub fn new(size: usize) -> Self {
        Self {
            graph: DagNN::new(),
            activation_pattern: vec![0.0; size],
            sample_count: 0,
        }
    }
}

/// Structural classifier for GRAPHEME graphs.
#[derive(Debug, Clone)]
pub struct StructuralClassifier {
    /// Number of output classes
    num_classes: usize,
    /// Template graphs for each class
    pub templates: Vec<ClassTemplate>,
    /// Template update momentum
    momentum: f32,
}

/// Classification result from structural classifier.
#[derive(Debug, Clone)]
pub struct StructuralClassificationResult {
    /// Predicted class
    pub predicted_class: usize,
    /// Predicted class (alias for compatibility)
    pub predicted: usize,
    /// Class probabilities
    pub probabilities: Vec<f32>,
    /// Confidence score
    pub confidence: f32,
    /// Loss value
    pub loss: f32,
    /// Whether prediction was correct (set during training)
    pub correct: bool,
    /// Gradient for backpropagation
    pub gradient: Vec<f32>,
}

impl StructuralClassifier {
    /// Create a new structural classifier with initialized templates
    pub fn new(num_classes: usize, _embed_dim: usize) -> Self {
        // Initialize templates for each class (AGI Mesh Ready)
        let templates: Vec<ClassTemplate> = (0..num_classes)
            .map(|_| ClassTemplate {
                graph: DagNN::new(),
                activation_pattern: Vec::new(),
                sample_count: 0,
            })
            .collect();

        Self {
            num_classes,
            templates,
            momentum: 0.9,
        }
    }

    /// Set the momentum for template updates
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Get number of classes
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Classify a graph
    pub fn classify(&self, graph: &DagNN) -> StructuralClassificationResult {
        let mut probabilities = vec![1.0 / self.num_classes as f32; self.num_classes];

        // Simple heuristic based on node count
        let node_count = graph.node_count();
        let predicted = node_count % self.num_classes;
        probabilities[predicted] = 0.5;

        StructuralClassificationResult {
            predicted_class: predicted,
            predicted,
            probabilities: probabilities.clone(),
            confidence: 0.5,
            loss: 0.0,
            correct: false,
            gradient: vec![0.0; self.num_classes],
        }
    }

    /// Convert distances to probabilities (softmax-like)
    pub fn distance_to_probs(&self, logits: &[f32]) -> Vec<f32> {
        // Apply softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        logits.iter().map(|x| (x - max_logit).exp() / exp_sum).collect()
    }

    /// Train on a batch
    pub fn train_batch(&mut self, _graphs: &[&DagNN], _labels: &[usize]) -> f32 {
        // Stub - returns dummy loss
        0.5
    }

    /// Update templates with a new sample
    pub fn update_template(&mut self, _class: usize, _activations: &[f32]) {
        // Stub - would update class template based on activations
    }
}

// ============================================================================
// DagNN Vision Extension (for vision brain - AGI Mesh Ready)
// ============================================================================

/// Extension trait for DagNN to support vision operations
pub trait DagNNVisionExt {
    /// Create a DagNN from image pixels
    fn from_image(pixels: &[f32], width: usize, height: usize) -> DomainResult<DagNN>;
    /// Create with classifier
    fn with_classifier(classifier: &StructuralClassifier) -> DagNN;
    /// Structural classification
    fn structural_classify(&self, classifier: &StructuralClassifier) -> (usize, f32);
    /// Structural classification step (for training)
    fn structural_classification_step(&self, classifier: &StructuralClassifier, target: usize) -> StructuralClassificationResult;
    /// Get classification logits
    fn get_classification_logits(&self) -> Vec<f32>;
    /// Forward pass with custom inputs
    fn forward_with_inputs(&mut self, inputs: &[f32]) -> Vec<f32>;
    /// Backward pass (gradient computation) using LeakyReLU
    fn backward(&mut self, output_grad: &[f32], embedding: &mut Embedding) -> BackwardPass;
    /// Get activations
    fn get_activations(&self) -> Vec<f32>;
}

impl DagNNVisionExt for DagNN {
    /// Convert an image (flattened pixel array) to a DagNN graph.
    fn from_image(pixels: &[f32], width: usize, height: usize) -> DomainResult<DagNN> {
        let mut dagnn = DagNN::new();
        let mut node_map: HashMap<(usize, usize), NodeId> = HashMap::new();

        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                let intensity = pixels.get(idx).copied().unwrap_or(0.0);
                let ch = ((row * width + col) % 256) as u8 as char;
                let node_id = dagnn.add_character(ch, idx);
                dagnn.graph[node_id].activation = intensity;
                node_map.insert((row, col), node_id);
            }
        }

        for row in 0..height {
            for col in 0..width {
                let current = node_map[&(row, col)];
                if col + 1 < width {
                    let right = node_map[&(row, col + 1)];
                    dagnn.add_edge(current, right, Edge::new(1.0, EdgeType::Sequential));
                }
                if row + 1 < height {
                    let bottom = node_map[&(row + 1, col)];
                    dagnn.add_edge(current, bottom, Edge::new(1.0, EdgeType::Structural));
                }
            }
        }

        let _ = dagnn.update_topology();
        Ok(dagnn)
    }

    /// Create a DagNN initialized for classification
    fn with_classifier(classifier: &StructuralClassifier) -> DagNN {
        let mut dagnn = DagNN::new();
        // Add output nodes for each class
        for _ in 0..classifier.num_classes() {
            dagnn.add_output();
        }
        let _ = dagnn.update_topology();
        dagnn
    }

    /// Perform structural classification
    fn structural_classify(&self, classifier: &StructuralClassifier) -> (usize, f32) {
        let result = classifier.classify(self);
        (result.predicted_class, result.confidence)
    }

    /// Perform classification step with gradient for training
    fn structural_classification_step(&self, classifier: &StructuralClassifier, target: usize) -> StructuralClassificationResult {
        let mut result = classifier.classify(self);
        result.correct = result.predicted_class == target;

        // Cross-entropy loss
        let eps = 1e-7;
        result.loss = -(result.probabilities[target] + eps).ln();

        // Gradient for cross-entropy with softmax
        result.gradient = result.probabilities.clone();
        result.gradient[target] -= 1.0;

        result
    }

    /// Get activations as logits for classification
    fn get_classification_logits(&self) -> Vec<f32> {
        self.output_nodes()
            .iter()
            .map(|&node| self.graph[node].activation)
            .collect()
    }

    /// Forward pass with custom input activations
    fn forward_with_inputs(&mut self, inputs: &[f32]) -> Vec<f32> {
        // Set input activations
        for (i, &input) in inputs.iter().enumerate() {
            if let Some(&node) = self.input_nodes().get(i) {
                self.graph[node].activation = input;
            }
        }

        // Propagate through the graph with dynamic √n normalization + LeakyReLU
        let alpha = 0.01; // LeakyReLU slope for negative values
        for node in self.graph.node_indices() {
            let edges: Vec<_> = self.graph.edges_directed(node, petgraph::Direction::Incoming).collect();
            let fan_in = edges.len();

            if fan_in > 0 {
                let weighted_sum: f32 = edges.iter()
                    .map(|e| {
                        let source = e.source();
                        self.graph[source].activation * e.weight().weight
                    })
                    .sum();

                // Dynamic √n normalization (GRAPHEME protocol)
                let scale = 1.0 / (fan_in as f32).sqrt();
                let normalized = scale * weighted_sum;

                // LeakyReLU activation
                self.graph[node].activation = if normalized > 0.0 { normalized } else { alpha * normalized };
            }
        }

        self.get_classification_logits()
    }

    /// Backward pass for gradient computation (LeakyReLU + dynamic √n)
    fn backward(&mut self, output_grad: &[f32], _embedding: &mut Embedding) -> BackwardPass {
        let mut pass = BackwardPass::new();
        let alpha = 0.01; // LeakyReLU slope

        // Node gradients: accumulate gradient flowing into each node
        let mut node_grads: HashMap<NodeId, f32> = HashMap::new();

        // Initialize output node gradients from the loss gradient
        for (i, &node) in self.output_nodes().iter().enumerate() {
            if let Some(&grad) = output_grad.get(i) {
                node_grads.insert(node, grad);
            }
        }

        // Backpropagate through graph (reverse topological order)
        // Collect all nodes and sort by reverse order
        let nodes: Vec<_> = self.graph.node_indices().collect();
        for &node in nodes.iter().rev() {
            let node_grad = *node_grads.get(&node).unwrap_or(&0.0);
            if node_grad.abs() < 1e-10 {
                continue; // Skip nodes with negligible gradient
            }

            let activation = self.graph[node].activation;
            // LeakyReLU derivative
            let deriv = if activation > 0.0 { 1.0 } else { alpha };

            // Get incoming edges for fan_in normalization
            let incoming_edges: Vec<_> = self.graph.edges_directed(node, petgraph::Direction::Incoming).collect();
            let fan_in = incoming_edges.len();
            let scale = if fan_in > 0 { 1.0 / (fan_in as f32).sqrt() } else { 1.0 };

            // Backprop gradient to each incoming edge
            for edge in incoming_edges {
                let source = edge.source();
                let weight = edge.weight().weight;

                // Edge gradient: ∂L/∂w = ∂L/∂output * ∂output/∂weighted_sum * ∂weighted_sum/∂w
                // = node_grad * deriv * scale * source_activation
                let source_activation = self.graph[source].activation;
                let edge_grad = node_grad * deriv * scale * source_activation;
                pass.edge_grads.insert((source, node), edge_grad);

                // Accumulate gradient to source node (for further backprop)
                let source_grad = node_grad * deriv * scale * weight;
                *node_grads.entry(source).or_insert(0.0) += source_grad;
            }
        }

        pass
    }

    /// Get all node activations
    fn get_activations(&self) -> Vec<f32> {
        self.graph.node_indices()
            .map(|n| self.graph[n].activation)
            .collect()
    }
}

// ============================================================================
// Adam Optimizer State (Unified across all learnable parameters)
// ============================================================================

/// Adam optimizer state for edge weights and template parameters.
///
/// Implements Adam (Kingma & Ba, 2014) with decoupled weight decay (AdamW).
/// Maintains per-parameter first moment (m) and second moment (v) estimates.
#[derive(Debug, Clone, Default)]
pub struct AdamState {
    /// First moment estimates for edge weights: (from_node, to_node) -> m
    edge_m: HashMap<(NodeId, NodeId), f32>,
    /// Second moment estimates for edge weights: (from_node, to_node) -> v
    edge_v: HashMap<(NodeId, NodeId), f32>,
    /// First moment estimates for template parameters: (class_id, param_idx) -> m
    template_m: HashMap<(usize, usize), f32>,
    /// Second moment estimates for template parameters: (class_id, param_idx) -> v
    template_v: HashMap<(usize, usize), f32>,
    /// Timestep counter (for bias correction)
    t: usize,
}

impl AdamState {
    /// Create a new Adam state
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the current timestep
    pub fn timestep(&self) -> usize {
        self.t
    }

    /// Increment timestep (call once per training step)
    pub fn step(&mut self) {
        self.t += 1;
    }

    /// Compute Adam update for an edge weight.
    ///
    /// Returns the weight delta to apply: w_new = w_old + delta
    #[allow(clippy::too_many_arguments)]
    pub fn compute_edge_update(
        &mut self,
        from: NodeId,
        to: NodeId,
        gradient: f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        current_weight: f32,
    ) -> f32 {
        let key = (from, to);

        // Get or initialize moment estimates
        let m = self.edge_m.entry(key).or_insert(0.0);
        let v = self.edge_v.entry(key).or_insert(0.0);

        // Update biased first moment estimate
        *m = beta1 * *m + (1.0 - beta1) * gradient;

        // Update biased second moment estimate
        *v = beta2 * *v + (1.0 - beta2) * gradient * gradient;

        // Compute bias-corrected estimates
        let t = self.t.max(1) as f32;
        let m_hat = *m / (1.0 - beta1.powf(t));
        let v_hat = *v / (1.0 - beta2.powf(t));

        // Compute update (AdamW: weight decay applied separately)
        let adam_update = -lr * m_hat / (v_hat.sqrt() + epsilon);
        let decay_update = -lr * weight_decay * current_weight;

        adam_update + decay_update
    }

    /// Compute Adam update for a template parameter.
    ///
    /// Returns the parameter delta to apply.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_template_update(
        &mut self,
        class_id: usize,
        param_idx: usize,
        gradient: f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> f32 {
        let key = (class_id, param_idx);

        // Get or initialize moment estimates
        let m = self.template_m.entry(key).or_insert(0.0);
        let v = self.template_v.entry(key).or_insert(0.0);

        // Update biased first moment estimate
        *m = beta1 * *m + (1.0 - beta1) * gradient;

        // Update biased second moment estimate
        *v = beta2 * *v + (1.0 - beta2) * gradient * gradient;

        // Compute bias-corrected estimates
        let t = self.t.max(1) as f32;
        let m_hat = *m / (1.0 - beta1.powf(t));
        let v_hat = *v / (1.0 - beta2.powf(t));

        // Compute update (no weight decay for templates)
        -lr * m_hat / (v_hat.sqrt() + epsilon)
    }

    /// Reset all optimizer state (useful when restarting training)
    pub fn reset(&mut self) {
        self.edge_m.clear();
        self.edge_v.clear();
        self.template_m.clear();
        self.template_v.clear();
        self.t = 0;
    }

    /// Get number of tracked edge parameters
    pub fn num_edge_params(&self) -> usize {
        self.edge_m.len()
    }

    /// Get number of tracked template parameters
    pub fn num_template_params(&self) -> usize {
        self.template_m.len()
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors in vision graph processing
#[derive(Error, Debug)]
pub enum VisionError {
    #[error("Invalid image dimensions: {0}x{1}")]
    InvalidDimensions(usize, usize),
    #[error("Pixel count mismatch: expected {expected}, got {actual}")]
    PixelCountMismatch { expected: usize, actual: usize },
    #[error("Invalid channel count: {0} (expected 1 or 3)")]
    InvalidChannels(usize),
    #[error("Empty image")]
    EmptyImage,
    #[error("Feature extraction error: {0}")]
    FeatureError(String),
}

/// Result type for vision operations
pub type VisionResult<T> = Result<T, VisionError>;

// ============================================================================
// Raw Image (Simple BMP-like format)
// ============================================================================

/// Raw image - just pixels and dimensions.
///
/// This is the universal input format for VisionBrain.
/// No compression, no color profiles, just raw pixel values.
///
/// # Example
/// ```
/// use grapheme_vision::RawImage;
///
/// // Create images of any size and color format
/// let (width, height) = (100, 80);
///
/// // Grayscale (1 channel)
/// let image = RawImage::grayscale(width, height, vec![0.5f32; width * height]).unwrap();
///
/// // RGB (3 channels)
/// let image = RawImage::rgb(width, height, vec![0.5f32; width * height * 3]).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawImage {
    /// Pixel values normalized to 0.0-1.0
    pub pixels: Vec<f32>,
    /// Image width in pixels
    pub width: usize,
    /// Image height in pixels
    pub height: usize,
    /// Number of channels (1=grayscale, 3=RGB)
    pub channels: usize,
}

impl RawImage {
    /// Create a grayscale image (1 channel)
    pub fn grayscale(width: usize, height: usize, pixels: Vec<f32>) -> VisionResult<Self> {
        let expected = width * height;
        if pixels.len() != expected {
            return Err(VisionError::PixelCountMismatch {
                expected,
                actual: pixels.len(),
            });
        }
        if width == 0 || height == 0 {
            return Err(VisionError::InvalidDimensions(width, height));
        }
        Ok(Self {
            pixels,
            width,
            height,
            channels: 1,
        })
    }

    /// Create an RGB image (3 channels)
    pub fn rgb(width: usize, height: usize, pixels: Vec<f32>) -> VisionResult<Self> {
        let expected = width * height * 3;
        if pixels.len() != expected {
            return Err(VisionError::PixelCountMismatch {
                expected,
                actual: pixels.len(),
            });
        }
        if width == 0 || height == 0 {
            return Err(VisionError::InvalidDimensions(width, height));
        }
        Ok(Self {
            pixels,
            width,
            height,
            channels: 3,
        })
    }

    /// Get pixel value at (x, y) for grayscale
    /// For RGB images, use `get_pixel_rgb` instead
    pub fn get_pixel(&self, x: usize, y: usize) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = y * self.width + x;
        if self.channels == 1 {
            self.pixels.get(idx).copied().unwrap_or(0.0)
        } else {
            // For RGB, return luminance (perceptual grayscale)
            let base = idx * 3;
            let r = self.pixels.get(base).copied().unwrap_or(0.0);
            let g = self.pixels.get(base + 1).copied().unwrap_or(0.0);
            let b = self.pixels.get(base + 2).copied().unwrap_or(0.0);
            0.299 * r + 0.587 * g + 0.114 * b  // Standard luminance formula
        }
    }

    /// Get RGB pixel values at (x, y) - returns (r, g, b)
    /// For grayscale, returns (v, v, v)
    pub fn get_pixel_rgb(&self, x: usize, y: usize) -> (f32, f32, f32) {
        if x >= self.width || y >= self.height {
            return (0.0, 0.0, 0.0);
        }
        let idx = y * self.width + x;
        if self.channels == 1 {
            let v = self.pixels.get(idx).copied().unwrap_or(0.0);
            (v, v, v)
        } else {
            let base = idx * 3;
            let r = self.pixels.get(base).copied().unwrap_or(0.0);
            let g = self.pixels.get(base + 1).copied().unwrap_or(0.0);
            let b = self.pixels.get(base + 2).copied().unwrap_or(0.0);
            (r, g, b)
        }
    }

    /// Get pixel at specific channel (0=R/gray, 1=G, 2=B)
    pub fn get_pixel_channel(&self, x: usize, y: usize, channel: usize) -> f32 {
        if x >= self.width || y >= self.height || channel >= self.channels {
            return 0.0;
        }
        let idx = y * self.width + x;
        if self.channels == 1 {
            self.pixels.get(idx).copied().unwrap_or(0.0)
        } else {
            self.pixels.get(idx * 3 + channel).copied().unwrap_or(0.0)
        }
    }

    /// Convert to grayscale (luminance-based)
    pub fn to_grayscale(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }

        let mut gray_pixels = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                gray_pixels.push(self.get_pixel(x, y));  // Uses luminance formula
            }
        }

        Self {
            pixels: gray_pixels,
            width: self.width,
            height: self.height,
            channels: 1,
        }
    }

    /// Get total pixel count
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

// ============================================================================
// Vision Node Types
// ============================================================================

/// Vision-specific node types for the image graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VisionNodeType {
    /// A detected blob/region
    Blob {
        /// Center x coordinate (normalized 0-1)
        cx: f32,
        /// Center y coordinate (normalized 0-1)
        cy: f32,
        /// Blob size (pixel count)
        size: usize,
        /// Average intensity
        intensity: f32,
    },
    /// An edge/contour point
    Edge {
        /// X coordinate (normalized 0-1)
        x: f32,
        /// Y coordinate (normalized 0-1)
        y: f32,
        /// Edge strength
        strength: f32,
    },
    /// A corner/keypoint
    Corner {
        /// X coordinate (normalized 0-1)
        x: f32,
        /// Y coordinate (normalized 0-1)
        y: f32,
        /// Corner response
        response: f32,
    },
    /// Hierarchical region (groups of blobs)
    Region {
        /// Child blob indices
        children: Vec<usize>,
        /// Region level in hierarchy
        level: usize,
    },
    /// Root node for the image
    ImageRoot {
        width: usize,
        height: usize,
    },
}

/// Get default activation based on vision node type
pub fn vision_type_activation(node_type: &VisionNodeType) -> f32 {
    match node_type {
        VisionNodeType::Blob { intensity, .. } => *intensity,
        VisionNodeType::Edge { strength, .. } => *strength,
        VisionNodeType::Corner { response, .. } => *response,
        VisionNodeType::Region { .. } => 0.8,
        VisionNodeType::ImageRoot { .. } => 1.0,
    }
}

/// A vision node with activation
pub type VisionNode = ActivatedNode<VisionNodeType>;

/// Create a new vision node with activation based on type
pub fn new_vision_node(node_type: VisionNodeType) -> VisionNode {
    ActivatedNode::with_type_activation(node_type, vision_type_activation)
}

// ============================================================================
// Vision Edge Types
// ============================================================================

/// Edge types in vision graphs
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VisionEdge {
    /// Spatial adjacency (blobs touching/near each other)
    Adjacent,
    /// Containment (parent region contains child)
    Contains,
    /// Same contour (edge points on same boundary)
    SameContour,
    /// Hierarchical (lower level to higher level)
    Hierarchy,
    /// Directional: source is above target
    Above,
    /// Directional: source is below target
    Below,
    /// Directional: source is left of target
    LeftOf,
    /// Directional: source is right of target
    RightOf,
    /// Proximity with distance (normalized 0.0-1.0)
    Proximity(f32),
}

// Note: Eq is implemented manually for VisionEdge because Proximity contains f32
// Two Proximity edges are considered equal if their distances are within epsilon
impl Eq for VisionEdge {}

impl std::hash::Hash for VisionEdge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        if let VisionEdge::Proximity(d) = self {
            // Quantize to 3 decimal places for consistent hashing
            ((d * 1000.0).round() as i32).hash(state);
        }
    }
}

/// Spatial relationship between two blobs
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialRelationship {
    /// Horizontal direction (-1.0 = left, 1.0 = right)
    pub dx: f32,
    /// Vertical direction (-1.0 = up, 1.0 = down)
    pub dy: f32,
    /// Distance (normalized 0.0-1.0 relative to image diagonal)
    pub distance: f32,
    /// Angle in radians (0 = right, π/2 = down)
    pub angle: f32,
}

impl SpatialRelationship {
    /// Compute spatial relationship between two blobs
    pub fn from_blobs(a: &Blob, b: &Blob, image_width: usize, image_height: usize) -> Self {
        let (ax, ay) = a.center;
        let (bx, by) = b.center;

        let dx = bx - ax;
        let dy = by - ay;

        // Normalize distance to image diagonal
        let diag = ((image_width * image_width + image_height * image_height) as f32).sqrt();
        let dist = (dx * dx + dy * dy).sqrt();
        let distance = (dist / diag).min(1.0);

        // Compute angle (0 = right, π/2 = down, π = left, -π/2 = up)
        let angle = dy.atan2(dx);

        Self {
            dx: dx / image_width as f32,
            dy: dy / image_height as f32,
            distance,
            angle,
        }
    }

    /// Check if source is predominantly above target
    pub fn is_above(&self) -> bool {
        self.dy < -0.1 && self.dy.abs() > self.dx.abs()
    }

    /// Check if source is predominantly below target
    pub fn is_below(&self) -> bool {
        self.dy > 0.1 && self.dy.abs() > self.dx.abs()
    }

    /// Check if source is predominantly left of target
    pub fn is_left_of(&self) -> bool {
        self.dx < -0.1 && self.dx.abs() > self.dy.abs()
    }

    /// Check if source is predominantly right of target
    pub fn is_right_of(&self) -> bool {
        self.dx > 0.1 && self.dx.abs() > self.dy.abs()
    }

    /// Get the primary directional edge type
    pub fn primary_direction(&self) -> Option<VisionEdge> {
        if self.is_above() {
            Some(VisionEdge::Above)
        } else if self.is_below() {
            Some(VisionEdge::Below)
        } else if self.is_left_of() {
            Some(VisionEdge::LeftOf)
        } else if self.is_right_of() {
            Some(VisionEdge::RightOf)
        } else {
            None // Too close to determine direction
        }
    }
}

// ============================================================================
// Vision Graph
// ============================================================================

/// An image represented as a graph
#[derive(Debug)]
pub struct VisionGraph {
    /// The underlying directed graph
    pub graph: DiGraph<VisionNode, VisionEdge>,
    /// Root node index
    pub root: Option<NodeIndex>,
    /// Source image dimensions
    pub width: usize,
    pub height: usize,
}

impl Default for VisionGraph {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl VisionGraph {
    /// Create a new empty vision graph
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            graph: DiGraph::new(),
            root: None,
            width,
            height,
        }
    }

    /// Add a node
    pub fn add_node(&mut self, node: VisionNode) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Add an edge
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: VisionEdge) {
        self.graph.add_edge(from, to, edge);
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

// ============================================================================
// Feature Extraction (No CNN - Pure Signal Processing)
// ============================================================================

/// Feature extraction mode for VisionBrain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FeatureMode {
    /// Blob detection - extracts connected components (variable node count)
    BlobDetection,
    /// Grid sampling - samples pixels at regular grid points (fixed node count)
    /// This gives dense, consistent features for DagNN input
    #[default]
    GridSampling,
    /// Hybrid mode - grid sampling + blob detection (grid as base, blobs as features)
    Hybrid,
}

/// Configuration for feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Feature extraction mode
    pub mode: FeatureMode,
    /// Grid dimensions for GridSampling mode (e.g., 7x7 = 49 nodes)
    pub grid_size: usize,
    /// Threshold for blob detection (pixels above this are foreground)
    pub blob_threshold: f32,
    /// Minimum blob size in pixels
    pub min_blob_size: usize,
    /// Maximum number of blobs to extract
    pub max_blobs: usize,
    /// Enable edge detection
    pub detect_edges: bool,
    /// Edge detection threshold
    pub edge_threshold: f32,
    /// Enable corner detection
    pub detect_corners: bool,
    /// Build hierarchical regions
    pub build_hierarchy: bool,
    /// Maximum hierarchy levels
    pub max_hierarchy_levels: usize,
    /// Threshold for spatial adjacency (normalized distance 0.0-1.0)
    pub adjacency_threshold: f32,
    /// Enable rich spatial relationships (directional edges, proximity)
    pub build_spatial_graph: bool,
    /// Enable parallel processing for grid sampling (uses Rayon)
    pub use_parallel: bool,
    /// Minimum grid size to enable parallel processing (smaller grids benefit less)
    pub parallel_threshold: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            mode: FeatureMode::GridSampling,
            grid_size: 10, // 10x10 = 100 nodes, matches max_vision_nodes default
            blob_threshold: 0.3,
            min_blob_size: 4,
            max_blobs: 100,
            detect_edges: false,
            edge_threshold: 0.2,
            detect_corners: false,
            build_hierarchy: true,
            max_hierarchy_levels: 3,
            adjacency_threshold: 0.15,
            build_spatial_graph: true,
            use_parallel: true, // Enable parallel by default for performance
            parallel_threshold: 16, // 16x16 = 256 grid cells, below this sequential is faster
        }
    }
}

impl FeatureConfig {
    /// Builder: set feature extraction mode
    pub fn with_mode(mut self, mode: FeatureMode) -> Self {
        self.mode = mode;
        self
    }

    /// Builder: set grid size for GridSampling mode
    pub fn with_grid_size(mut self, size: usize) -> Self {
        self.grid_size = size;
        self
    }

    /// Builder: use blob detection mode
    pub fn blob_detection(mut self) -> Self {
        self.mode = FeatureMode::BlobDetection;
        self
    }

    /// Builder: use grid sampling mode with specified grid size
    pub fn grid_sampling(mut self, grid_size: usize) -> Self {
        self.mode = FeatureMode::GridSampling;
        self.grid_size = grid_size;
        self
    }

    /// Builder: set blob detection threshold
    pub fn with_blob_threshold(mut self, threshold: f32) -> Self {
        self.blob_threshold = threshold;
        self
    }

    /// Builder: set maximum number of blobs
    pub fn with_max_blobs(mut self, max: usize) -> Self {
        self.max_blobs = max;
        self
    }

    /// Builder: set minimum blob size
    pub fn with_min_blob_size(mut self, size: usize) -> Self {
        self.min_blob_size = size;
        self
    }

    /// Builder: enable/disable hierarchy building
    pub fn with_hierarchy(mut self, enabled: bool, max_levels: usize) -> Self {
        self.build_hierarchy = enabled;
        self.max_hierarchy_levels = max_levels;
        self
    }

    /// Builder: enable/disable parallel processing
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.use_parallel = enabled;
        self
    }

    /// Builder: set parallel threshold (min grid size to enable parallel)
    pub fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Check if parallel processing should be used for current config
    pub fn should_use_parallel(&self) -> bool {
        self.use_parallel && self.grid_size >= self.parallel_threshold
    }
}

/// A detected blob (connected component)
#[derive(Debug, Clone)]
pub struct Blob {
    /// Pixel coordinates (x, y)
    pub pixels: Vec<(usize, usize)>,
    /// Center of mass
    pub center: (f32, f32),
    /// Average intensity
    pub intensity: f32,
    /// Bounding box (x, y, w, h)
    pub bbox: (usize, usize, usize, usize),
}

/// A hierarchical blob with parent-child relationships
#[derive(Debug, Clone)]
pub struct HierarchicalBlob {
    /// The blob data
    pub blob: Blob,
    /// Hierarchy level (0 = finest, higher = coarser)
    pub level: usize,
    /// Indices of child blobs (finer scale)
    pub children: Vec<usize>,
    /// Index of parent blob (coarser scale), if any
    pub parent: Option<usize>,
    /// Scale at which this blob was detected
    pub scale: f32,
}

impl HierarchicalBlob {
    /// Create a new hierarchical blob at a given level
    pub fn new(blob: Blob, level: usize, scale: f32) -> Self {
        Self {
            blob,
            level,
            children: Vec::new(),
            parent: None,
            scale,
        }
    }

    /// Check if this blob contains another blob
    pub fn contains(&self, other: &Blob) -> bool {
        // Check if other's center is within this blob's bounding box
        let (bx, by, bw, bh) = self.blob.bbox;
        let (ox, oy) = other.center;
        ox >= bx as f32 && ox < (bx + bw) as f32 &&
        oy >= by as f32 && oy < (by + bh) as f32
    }
}

/// Result of hierarchical blob detection
#[derive(Debug, Clone)]
pub struct BlobHierarchy {
    /// All blobs across all levels
    pub blobs: Vec<HierarchicalBlob>,
    /// Number of hierarchy levels
    pub num_levels: usize,
    /// Indices of root blobs (no parent)
    pub roots: Vec<usize>,
}

/// Extract blobs (connected components) from image
pub fn extract_blobs(image: &RawImage, config: &FeatureConfig) -> Vec<Blob> {
    let gray = image.to_grayscale();
    let mut visited = vec![false; gray.pixel_count()];
    let mut blobs = Vec::new();

    for y in 0..gray.height {
        for x in 0..gray.width {
            let idx = y * gray.width + x;
            if visited[idx] || gray.pixels[idx] < config.blob_threshold {
                continue;
            }

            // Flood fill to find connected component
            let mut pixels = Vec::new();
            let mut stack = vec![(x, y)];
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sum_intensity = 0.0f32;
            let mut min_x = x;
            let mut max_x = x;
            let mut min_y = y;
            let mut max_y = y;

            while let Some((px, py)) = stack.pop() {
                let pidx = py * gray.width + px;
                if visited[pidx] || gray.pixels[pidx] < config.blob_threshold {
                    continue;
                }
                visited[pidx] = true;

                let intensity = gray.pixels[pidx];
                pixels.push((px, py));
                sum_x += px as f32 * intensity;
                sum_y += py as f32 * intensity;
                sum_intensity += intensity;

                min_x = min_x.min(px);
                max_x = max_x.max(px);
                min_y = min_y.min(py);
                max_y = max_y.max(py);

                // 4-connectivity neighbors
                if px > 0 {
                    stack.push((px - 1, py));
                }
                if px + 1 < gray.width {
                    stack.push((px + 1, py));
                }
                if py > 0 {
                    stack.push((px, py - 1));
                }
                if py + 1 < gray.height {
                    stack.push((px, py + 1));
                }
            }

            if pixels.len() >= config.min_blob_size {
                let center = if sum_intensity > 0.0 {
                    (sum_x / sum_intensity, sum_y / sum_intensity)
                } else {
                    (
                        pixels.iter().map(|p| p.0).sum::<usize>() as f32 / pixels.len() as f32,
                        pixels.iter().map(|p| p.1).sum::<usize>() as f32 / pixels.len() as f32,
                    )
                };

                blobs.push(Blob {
                    intensity: sum_intensity / pixels.len() as f32,
                    center,
                    bbox: (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                    pixels,
                });

                if blobs.len() >= config.max_blobs {
                    break;
                }
            }
        }
        if blobs.len() >= config.max_blobs {
            break;
        }
    }

    // Sort by size (largest first)
    blobs.sort_by(|a, b| b.pixels.len().cmp(&a.pixels.len()));
    blobs
}

/// Extract blobs at a specific threshold
fn extract_blobs_at_threshold(image: &RawImage, threshold: f32, min_size: usize, max_blobs: usize) -> Vec<Blob> {
    let gray = image.to_grayscale();
    let mut visited = vec![false; gray.pixel_count()];
    let mut blobs = Vec::new();

    for y in 0..gray.height {
        for x in 0..gray.width {
            let idx = y * gray.width + x;
            if visited[idx] || gray.pixels[idx] < threshold {
                continue;
            }

            // Flood fill to find connected component
            let mut pixels = Vec::new();
            let mut stack = vec![(x, y)];
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sum_intensity = 0.0f32;
            let mut min_x = x;
            let mut max_x = x;
            let mut min_y = y;
            let mut max_y = y;

            while let Some((px, py)) = stack.pop() {
                let pidx = py * gray.width + px;
                if visited[pidx] || gray.pixels[pidx] < threshold {
                    continue;
                }
                visited[pidx] = true;

                let intensity = gray.pixels[pidx];
                pixels.push((px, py));
                sum_x += px as f32 * intensity;
                sum_y += py as f32 * intensity;
                sum_intensity += intensity;

                min_x = min_x.min(px);
                max_x = max_x.max(px);
                min_y = min_y.min(py);
                max_y = max_y.max(py);

                // 4-connectivity neighbors
                if px > 0 {
                    stack.push((px - 1, py));
                }
                if px + 1 < gray.width {
                    stack.push((px + 1, py));
                }
                if py > 0 {
                    stack.push((px, py - 1));
                }
                if py + 1 < gray.height {
                    stack.push((px, py + 1));
                }
            }

            if pixels.len() >= min_size {
                let center = if sum_intensity > 0.0 {
                    (sum_x / sum_intensity, sum_y / sum_intensity)
                } else {
                    (
                        pixels.iter().map(|p| p.0).sum::<usize>() as f32 / pixels.len() as f32,
                        pixels.iter().map(|p| p.1).sum::<usize>() as f32 / pixels.len() as f32,
                    )
                };

                blobs.push(Blob {
                    intensity: sum_intensity / pixels.len() as f32,
                    center,
                    bbox: (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                    pixels,
                });

                if blobs.len() >= max_blobs {
                    break;
                }
            }
        }
        if blobs.len() >= max_blobs {
            break;
        }
    }

    blobs.sort_by(|a, b| b.pixels.len().cmp(&a.pixels.len()));
    blobs
}

/// Extract hierarchical blobs at multiple scales.
///
/// This function detects blobs at multiple intensity thresholds to create
/// a hierarchical structure. Coarser scales (lower thresholds) capture
/// larger structures, while finer scales (higher thresholds) capture details.
///
/// # Algorithm
/// 1. Detect blobs at multiple thresholds (coarse → fine)
/// 2. Link child blobs to parent blobs based on spatial containment
/// 3. Return hierarchy with parent-child relationships
///
/// # Arguments
/// * `image` - The input image
/// * `config` - Feature configuration (uses max_hierarchy_levels)
///
/// # Returns
/// A BlobHierarchy with all detected blobs and their relationships
pub fn extract_hierarchical_blobs(image: &RawImage, config: &FeatureConfig) -> BlobHierarchy {
    let num_levels = config.max_hierarchy_levels.max(1);
    let mut all_blobs: Vec<HierarchicalBlob> = Vec::new();

    // Generate threshold levels from coarse (low) to fine (high)
    // Coarse = low threshold (more permissive, larger blobs)
    // Fine = high threshold (more selective, smaller blobs)
    let base_threshold = config.blob_threshold;
    let threshold_step = (1.0 - base_threshold) / (num_levels as f32 + 1.0);

    for level in 0..num_levels {
        // Level 0 = finest (highest threshold), Level N-1 = coarsest (lowest)
        let threshold = base_threshold + threshold_step * ((num_levels - level - 1) as f32);
        let scale = 1.0 / (level as f32 + 1.0);

        // Adjust min_blob_size for scale (larger at coarser scales)
        let min_size = (config.min_blob_size as f32 * (level as f32 + 1.0).sqrt()) as usize;
        let max_blobs = config.max_blobs / num_levels.max(1);

        let blobs = extract_blobs_at_threshold(image, threshold, min_size, max_blobs.max(10));

        let level_start_idx = all_blobs.len();
        for blob in blobs {
            all_blobs.push(HierarchicalBlob::new(blob, level, scale));
        }

        // Link to parent level (coarser, higher level number)
        if level > 0 {
            let parent_level = level - 1;
            // Find parent blobs from the coarser level
            for child_idx in level_start_idx..all_blobs.len() {
                let child_center = all_blobs[child_idx].blob.center;

                // Find best parent (smallest containing blob at parent level)
                let mut best_parent: Option<(usize, usize)> = None; // (index, size)

                for (parent_idx, parent_blob) in all_blobs.iter().enumerate() {
                    if parent_blob.level != parent_level {
                        continue;
                    }

                    // Check if parent contains child's center
                    let (px, py, pw, ph) = parent_blob.blob.bbox;
                    let (cx, cy) = child_center;

                    if cx >= px as f32 && cx < (px + pw) as f32 &&
                       cy >= py as f32 && cy < (py + ph) as f32 {
                        let parent_size = parent_blob.blob.pixels.len();
                        if best_parent.is_none_or(|(_, best_size)| parent_size < best_size) {
                            best_parent = Some((parent_idx, parent_size));
                        }
                    }
                }

                if let Some((parent_idx, _)) = best_parent {
                    all_blobs[child_idx].parent = Some(parent_idx);
                    all_blobs[parent_idx].children.push(child_idx);
                }
            }
        }
    }

    // Find root blobs (no parent)
    let roots: Vec<usize> = all_blobs
        .iter()
        .enumerate()
        .filter(|(_, b)| b.parent.is_none())
        .map(|(i, _)| i)
        .collect();

    BlobHierarchy {
        blobs: all_blobs,
        num_levels,
        roots,
    }
}

/// Compute all spatial relationships between blobs.
///
/// Returns a vector of (source_idx, target_idx, relationship) tuples
/// for all pairs of blobs within the proximity threshold.
pub fn compute_spatial_relationships(
    blobs: &[Blob],
    image_width: usize,
    image_height: usize,
    max_distance: f32,
) -> Vec<(usize, usize, SpatialRelationship)> {
    let mut relationships = Vec::new();

    for i in 0..blobs.len() {
        for j in (i + 1)..blobs.len() {
            let rel = SpatialRelationship::from_blobs(&blobs[i], &blobs[j], image_width, image_height);

            // Only include relationships within threshold
            if rel.distance <= max_distance {
                relationships.push((i, j, rel));
            }
        }
    }

    relationships
}

/// Build a complete spatial relationship graph from blobs.
///
/// This creates edges for:
/// - Adjacency (blobs touching)
/// - Directional relationships (above, below, left, right)
/// - Proximity with distance
pub fn build_spatial_graph(
    graph: &mut VisionGraph,
    blobs: &[Blob],
    blob_nodes: &[NodeIndex],
    config: &FeatureConfig,
) {
    let max_distance = config.adjacency_threshold.max(0.5); // At least 50% of diagonal

    let relationships = compute_spatial_relationships(
        blobs,
        graph.width,
        graph.height,
        max_distance,
    );

    for (i, j, rel) in relationships {
        if i >= blob_nodes.len() || j >= blob_nodes.len() {
            continue;
        }

        let node_i = blob_nodes[i];
        let node_j = blob_nodes[j];

        // Add adjacency edge if blobs are touching
        if blobs_adjacent(&blobs[i], &blobs[j]) {
            graph.add_edge(node_i, node_j, VisionEdge::Adjacent);
            graph.add_edge(node_j, node_i, VisionEdge::Adjacent);
        }

        // Add directional edges
        if let Some(dir) = rel.primary_direction() {
            graph.add_edge(node_i, node_j, dir);
            // Add inverse direction
            let inverse = match dir {
                VisionEdge::Above => VisionEdge::Below,
                VisionEdge::Below => VisionEdge::Above,
                VisionEdge::LeftOf => VisionEdge::RightOf,
                VisionEdge::RightOf => VisionEdge::LeftOf,
                _ => continue,
            };
            graph.add_edge(node_j, node_i, inverse);
        }

        // Add proximity edge if within threshold
        if rel.distance <= config.adjacency_threshold {
            graph.add_edge(node_i, node_j, VisionEdge::Proximity(rel.distance));
            graph.add_edge(node_j, node_i, VisionEdge::Proximity(rel.distance));
        }
    }
}

/// Check if two blobs are adjacent (bounding boxes touch or overlap)
pub fn blobs_adjacent(a: &Blob, b: &Blob) -> bool {
    let (ax, ay, aw, ah) = a.bbox;
    let (bx, by, bw, bh) = b.bbox;

    // Expand bounding boxes by 1 pixel and check overlap
    let a_left = ax.saturating_sub(1);
    let a_right = ax + aw + 1;
    let a_top = ay.saturating_sub(1);
    let a_bottom = ay + ah + 1;

    let b_left = bx;
    let b_right = bx + bw;
    let b_top = by;
    let b_bottom = by + bh;

    !(a_right < b_left || b_right < a_left || a_bottom < b_top || b_bottom < a_top)
}

// ============================================================================
// Image to Graph Conversion
// ============================================================================

/// Convert an image to a GRAPHEME graph.
///
/// This is the core VisionBrain functionality:
/// - Deterministic: same image always produces same graph
/// - Multiple modes: GridSampling (dense), BlobDetection (sparse), Hybrid
/// - No CNN: pure signal processing
pub fn image_to_graph(image: &RawImage, config: &FeatureConfig) -> VisionResult<VisionGraph> {
    if image.pixels.is_empty() {
        return Err(VisionError::EmptyImage);
    }

    match config.mode {
        FeatureMode::GridSampling => image_to_graph_grid(image, config),
        FeatureMode::BlobDetection => image_to_graph_blobs(image, config),
        FeatureMode::Hybrid => image_to_graph_hybrid(image, config),
    }
}

/// Grid sampling mode: samples pixels at regular grid points.
/// Produces grid_size^2 nodes with consistent, dense activations.
///
/// Uses parallel processing for large grids when `use_parallel` is enabled.
/// The pixel sampling phase is parallelized; graph construction is sequential.
fn image_to_graph_grid(image: &RawImage, config: &FeatureConfig) -> VisionResult<VisionGraph> {
    let grid_size = config.grid_size;

    // Parallel sampling: extract intensities for all grid cells in parallel
    // This is the CPU-intensive part that benefits from parallelization
    let grid_samples: Vec<GridSample> = if config.should_use_parallel() {
        sample_grid_parallel(image, grid_size)
    } else {
        sample_grid_sequential(image, grid_size)
    };

    // Graph construction: must be sequential (graph mutations not thread-safe)
    let mut graph = VisionGraph::new(image.width, image.height);

    // Create root node
    let root = graph.add_node(new_vision_node(VisionNodeType::ImageRoot {
        width: image.width,
        height: image.height,
    }));
    graph.root = Some(root);

    // Add all grid nodes
    let mut grid_nodes = Vec::with_capacity(grid_size * grid_size);
    for sample in &grid_samples {
        let node = graph.add_node(new_vision_node(VisionNodeType::Blob {
            cx: sample.cx,
            cy: sample.cy,
            size: 1, // Grid cell represents 1 logical unit
            intensity: sample.intensity,
        }));
        grid_nodes.push(node);
        graph.add_edge(root, node, VisionEdge::Contains);
    }

    // Build spatial edges between adjacent grid cells
    if config.build_spatial_graph {
        build_spatial_edges(&mut graph, &grid_nodes, grid_size);
    }

    Ok(graph)
}

/// A sampled grid cell with computed values
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // gx, gy used in tests for verification
struct GridSample {
    /// Grid x position
    gx: usize,
    /// Grid y position
    gy: usize,
    /// Normalized center x (0.0 to 1.0)
    cx: f32,
    /// Normalized center y (0.0 to 1.0)
    cy: f32,
    /// Sampled intensity
    intensity: f32,
}

/// Sample grid cells in parallel using Rayon.
/// O(grid_size^2) work distributed across threads.
fn sample_grid_parallel(image: &RawImage, grid_size: usize) -> Vec<GridSample> {
    let width = image.width;
    let height = image.height;
    let grid_size_f32 = grid_size as f32;

    // Create indices for all grid cells
    let indices: Vec<(usize, usize)> = (0..grid_size)
        .flat_map(|gy| (0..grid_size).map(move |gx| (gx, gy)))
        .collect();

    // Sample in parallel - each cell computation is independent
    indices
        .par_iter()
        .map(|&(gx, gy)| {
            // Map grid position to image coordinates
            let x = (gx * width) / grid_size;
            let y = (gy * height) / grid_size;

            // Sample pixel intensity (with 3x3 averaging for robustness)
            let intensity = sample_pixel_region(image, x, y, 1);

            // Normalized center coordinates
            let cx = (gx as f32 + 0.5) / grid_size_f32;
            let cy = (gy as f32 + 0.5) / grid_size_f32;

            GridSample { gx, gy, cx, cy, intensity }
        })
        .collect()
}

/// Sample grid cells sequentially (for small grids or when parallel disabled).
fn sample_grid_sequential(image: &RawImage, grid_size: usize) -> Vec<GridSample> {
    let width = image.width;
    let height = image.height;
    let grid_size_f32 = grid_size as f32;

    let mut samples = Vec::with_capacity(grid_size * grid_size);

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            // Map grid position to image coordinates
            let x = (gx * width) / grid_size;
            let y = (gy * height) / grid_size;

            // Sample pixel intensity (with 3x3 averaging for robustness)
            let intensity = sample_pixel_region(image, x, y, 1);

            // Normalized center coordinates
            let cx = (gx as f32 + 0.5) / grid_size_f32;
            let cy = (gy as f32 + 0.5) / grid_size_f32;

            samples.push(GridSample { gx, gy, cx, cy, intensity });
        }
    }

    samples
}

/// Build spatial edges between adjacent grid cells.
fn build_spatial_edges(graph: &mut VisionGraph, grid_nodes: &[NodeIndex], grid_size: usize) {
    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let idx = gy * grid_size + gx;

            // Connect to right neighbor
            if gx + 1 < grid_size {
                let right_idx = gy * grid_size + (gx + 1);
                graph.add_edge(grid_nodes[idx], grid_nodes[right_idx], VisionEdge::LeftOf);
                graph.add_edge(grid_nodes[right_idx], grid_nodes[idx], VisionEdge::RightOf);
            }

            // Connect to bottom neighbor
            if gy + 1 < grid_size {
                let bottom_idx = (gy + 1) * grid_size + gx;
                graph.add_edge(grid_nodes[idx], grid_nodes[bottom_idx], VisionEdge::Above);
                graph.add_edge(grid_nodes[bottom_idx], grid_nodes[idx], VisionEdge::Below);
            }
        }
    }
}

/// Sample a region around a pixel for robust intensity estimation.
/// Returns average intensity in a (2*radius+1)x(2*radius+1) region.
fn sample_pixel_region(image: &RawImage, x: usize, y: usize, radius: usize) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;

    let x_start = x.saturating_sub(radius);
    let y_start = y.saturating_sub(radius);
    let x_end = (x + radius + 1).min(image.width);
    let y_end = (y + radius + 1).min(image.height);

    for py in y_start..y_end {
        for px in x_start..x_end {
            sum += image.get_pixel(px, py);
            count += 1;
        }
    }

    if count > 0 { sum / count as f32 } else { 0.0 }
}

/// Hybrid mode: combines grid sampling with blob detection.
/// Grid provides consistent base features, blobs provide semantic features.
fn image_to_graph_hybrid(image: &RawImage, config: &FeatureConfig) -> VisionResult<VisionGraph> {
    // Start with grid sampling
    let mut graph = image_to_graph_grid(image, config)?;

    // Add blob features on top
    let blobs = extract_blobs(image, config);
    let w = image.width as f32;
    let h = image.height as f32;

    // Create blob nodes (separate from grid nodes)
    for blob in &blobs {
        let node = graph.add_node(new_vision_node(VisionNodeType::Blob {
            cx: blob.center.0 / w,
            cy: blob.center.1 / h,
            size: blob.pixels.len(),
            intensity: blob.intensity,
        }));

        // Connect blob to root
        if let Some(root) = graph.root {
            graph.add_edge(root, node, VisionEdge::Contains);
        }
    }

    Ok(graph)
}

/// Blob detection mode: extracts connected components (original behavior).
fn image_to_graph_blobs(image: &RawImage, config: &FeatureConfig) -> VisionResult<VisionGraph> {
    let mut graph = VisionGraph::new(image.width, image.height);
    let w = image.width as f32;
    let h = image.height as f32;

    // Create root node
    let root = graph.add_node(new_vision_node(VisionNodeType::ImageRoot {
        width: image.width,
        height: image.height,
    }));
    graph.root = Some(root);

    // Use hierarchical blob detection if enabled
    if config.build_hierarchy && config.max_hierarchy_levels > 1 {
        let hierarchy = extract_hierarchical_blobs(image, config);

        if hierarchy.blobs.is_empty() {
            return Ok(graph);
        }

        // Create nodes for all hierarchical blobs
        let mut blob_nodes: Vec<NodeIndex> = Vec::with_capacity(hierarchy.blobs.len());

        for hblob in &hierarchy.blobs {
            let blob = &hblob.blob;
            let node = graph.add_node(new_vision_node(VisionNodeType::Blob {
                cx: blob.center.0 / w,
                cy: blob.center.1 / h,
                size: blob.pixels.len(),
                intensity: blob.intensity,
            }));
            blob_nodes.push(node);
        }

        // Connect root to top-level blobs (roots in hierarchy)
        for &root_idx in &hierarchy.roots {
            if root_idx < blob_nodes.len() {
                graph.add_edge(root, blob_nodes[root_idx], VisionEdge::Contains);
            }
        }

        // Connect parent-child relationships within hierarchy
        for (idx, hblob) in hierarchy.blobs.iter().enumerate() {
            // Connect to children
            for &child_idx in &hblob.children {
                if child_idx < blob_nodes.len() {
                    graph.add_edge(blob_nodes[idx], blob_nodes[child_idx], VisionEdge::Hierarchy);
                }
            }
        }

        // Build spatial relationships between same-level blobs
        if config.build_spatial_graph {
            // Group blobs by level and build spatial graphs for each level
            for level in 0..hierarchy.num_levels {
                let level_blobs: Vec<&Blob> = hierarchy.blobs
                    .iter()
                    .filter(|h| h.level == level)
                    .map(|h| &h.blob)
                    .collect();
                let level_indices: Vec<usize> = hierarchy.blobs
                    .iter()
                    .enumerate()
                    .filter(|(_, h)| h.level == level)
                    .map(|(i, _)| i)
                    .collect();

                // Build spatial relationships within this level
                if level_blobs.len() > 1 {
                    let owned_blobs: Vec<Blob> = level_blobs.iter().map(|b| (*b).clone()).collect();
                    let level_nodes: Vec<NodeIndex> = level_indices.iter()
                        .filter_map(|&i| blob_nodes.get(i).copied())
                        .collect();
                    build_spatial_graph(&mut graph, &owned_blobs, &level_nodes, config);
                }
            }
        } else {
            // Simple adjacency (old behavior)
            for i in 0..hierarchy.blobs.len() {
                for j in (i + 1)..hierarchy.blobs.len() {
                    if hierarchy.blobs[i].level == hierarchy.blobs[j].level
                        && blobs_adjacent(&hierarchy.blobs[i].blob, &hierarchy.blobs[j].blob)
                    {
                        graph.add_edge(blob_nodes[i], blob_nodes[j], VisionEdge::Adjacent);
                        graph.add_edge(blob_nodes[j], blob_nodes[i], VisionEdge::Adjacent);
                    }
                }
            }
        }
    } else {
        // Single-level blob detection (original behavior)
        let blobs = extract_blobs(image, config);

        if blobs.is_empty() {
            return Ok(graph);
        }

        // Create blob nodes
        let mut blob_nodes = Vec::with_capacity(blobs.len());

        for blob in &blobs {
            let node = graph.add_node(new_vision_node(VisionNodeType::Blob {
                cx: blob.center.0 / w,
                cy: blob.center.1 / h,
                size: blob.pixels.len(),
                intensity: blob.intensity,
            }));
            blob_nodes.push(node);

            // Connect blob to root
            graph.add_edge(root, node, VisionEdge::Contains);
        }

        // Build spatial relationships
        if config.build_spatial_graph && blobs.len() > 1 {
            build_spatial_graph(&mut graph, &blobs, &blob_nodes, config);
        } else {
            // Simple adjacency (old behavior)
            for i in 0..blobs.len() {
                for j in (i + 1)..blobs.len() {
                    if blobs_adjacent(&blobs[i], &blobs[j]) {
                        graph.add_edge(blob_nodes[i], blob_nodes[j], VisionEdge::Adjacent);
                        graph.add_edge(blob_nodes[j], blob_nodes[i], VisionEdge::Adjacent);
                    }
                }
            }
        }

        // Build legacy hierarchy if enabled (spatial clustering)
        if config.build_hierarchy && blobs.len() > 1 {
            build_blob_hierarchy(&mut graph, &blobs, &blob_nodes, root, config);
        }
    }

    Ok(graph)
}

/// Build hierarchical regions from blobs using spatial clustering
fn build_blob_hierarchy(
    graph: &mut VisionGraph,
    blobs: &[Blob],
    blob_nodes: &[NodeIndex],
    root: NodeIndex,
    config: &FeatureConfig,
) {
    if blobs.len() < 2 || config.max_hierarchy_levels == 0 {
        return;
    }

    // Simple spatial clustering: group blobs that are close together
    let mut used = vec![false; blobs.len()];
    let mut regions: Vec<Vec<usize>> = Vec::new();

    // Distance threshold for grouping (relative to image size)
    let threshold = 0.2; // 20% of image diagonal

    for i in 0..blobs.len() {
        if used[i] {
            continue;
        }

        let mut region = vec![i];
        used[i] = true;

        for j in (i + 1)..blobs.len() {
            if used[j] {
                continue;
            }

            // Check distance between blob centers
            let dx = blobs[i].center.0 - blobs[j].center.0;
            let dy = blobs[i].center.1 - blobs[j].center.1;
            let dist = (dx * dx + dy * dy).sqrt()
                / (graph.width as f32 * graph.width as f32
                    + graph.height as f32 * graph.height as f32)
                    .sqrt();

            if dist < threshold {
                region.push(j);
                used[j] = true;
            }
        }

        if region.len() > 1 {
            regions.push(region);
        }
    }

    // Create region nodes
    for (level, region) in regions.iter().enumerate() {
        if level >= config.max_hierarchy_levels {
            break;
        }

        let region_node = graph.add_node(new_vision_node(VisionNodeType::Region {
            children: region.clone(),
            level: level + 1,
        }));

        // Connect root to region
        graph.add_edge(root, region_node, VisionEdge::Hierarchy);

        // Connect region to its blobs
        for &blob_idx in region {
            if blob_idx < blob_nodes.len() {
                graph.add_edge(region_node, blob_nodes[blob_idx], VisionEdge::Contains);
            }
        }
    }
}

// ============================================================================
// Vision Brain
// ============================================================================

/// Create the vision domain configuration
fn create_vision_config() -> DomainConfig {
    let keywords = vec![
        "image", "pixel", "blob", "region", "edge", "corner", "visual", "picture", "photo",
    ];

    let normalizer = TextNormalizer::new().trim_whitespace(true);

    DomainConfig::new("vision", "Computer Vision", keywords)
        .with_version("0.1.0")
        .with_normalizer(normalizer)
        .with_annotation_prefix("@vision:")
}

/// The Vision Brain for image-to-graph embedding.
///
/// Implements GRAPHEME's universal principle:
/// - Same image → Same graph (deterministic)
/// - No CNN, no learned features
/// - Hierarchical structure from signal processing
#[derive(Clone)]
pub struct VisionBrain {
    /// Domain configuration
    config: DomainConfig,
    /// Feature extraction configuration
    feature_config: FeatureConfig,
}

impl Default for VisionBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for VisionBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionBrain")
            .field("domain", &"vision")
            .finish()
    }
}

impl VisionBrain {
    /// Create a new vision brain with default config
    pub fn new() -> Self {
        Self {
            config: create_vision_config(),
            feature_config: FeatureConfig::default(),
        }
    }

    /// Set feature extraction configuration
    pub fn with_feature_config(mut self, config: FeatureConfig) -> Self {
        self.feature_config = config;
        self
    }

    /// Convert image to graph (core VisionBrain operation)
    ///
    /// This is deterministic: same image always produces same graph.
    /// Works with any image size.
    pub fn to_graph(&self, image: &RawImage) -> VisionResult<VisionGraph> {
        image_to_graph(image, &self.feature_config)
    }

    /// Convert VisionGraph to DagNN for GRAPHEME core processing
    ///
    /// This converts the blob-based vision graph into a DagNN using from_image
    /// with blob-weighted pixel values. Blobs that are detected become
    /// high-activation regions.
    ///
    /// Returns error if VisionGraph has no ImageRoot node (dimensions required).
    pub fn to_dagnn(&self, vision_graph: &VisionGraph) -> DomainResult<DagNN> {
        // Get image dimensions from the root node - REQUIRED, no hardcoded fallback
        let (width, height) = vision_graph.graph.node_weights()
            .find_map(|n| match &n.node_type {
                VisionNodeType::ImageRoot { width, height } => Some((*width, *height)),
                _ => None,
            })
            .ok_or_else(|| grapheme_core::DomainError::InvalidInput(
                "VisionGraph missing ImageRoot node - cannot determine dimensions".to_string()
            ))?;

        // Create a pixel array with blob activations
        // Blobs contribute their intensity to their center position
        let mut pixels = vec![0.0f32; width * height];

        for node in vision_graph.graph.node_weights() {
            if let VisionNodeType::Blob { cx, cy, size, intensity } = &node.node_type {
                // Convert normalized coords back to pixel coords
                let px = (*cx * width as f32).round() as usize;
                let py = (*cy * height as f32).round() as usize;

                if px < width && py < height {
                    let idx = py * width + px;
                    // Combine intensity with size for richer activation
                    let activation = intensity * (1.0 + (*size as f32).log2().max(0.0) / 10.0);
                    pixels[idx] = pixels[idx].max(activation.clamp(0.0, 1.0));
                }
            }
        }

        // Use DagNN's from_image to create proper structure with input nodes
        DagNN::from_image(&pixels, width, height)
            .map_err(|e| grapheme_core::DomainError::InvalidInput(e.to_string()))
    }
}

// ============================================================================
// BaseDomainBrain Implementation
// ============================================================================

impl BaseDomainBrain for VisionBrain {
    fn config(&self) -> &DomainConfig {
        &self.config
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

impl DomainBrain for VisionBrain {
    fn domain_id(&self) -> &str {
        &self.config.domain_id
    }

    fn domain_name(&self) -> &str {
        &self.config.domain_name
    }

    fn version(&self) -> &str {
        &self.config.version
    }

    fn can_process(&self, input: &str) -> bool {
        self.default_can_process(input)
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        self.default_parse(input)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        self.default_from_core(graph)
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        self.default_to_core(graph)
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        self.default_validate(graph)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        self.default_execute(graph)
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        vec![
            DomainRule::new(0, "Blob Detection", "Extract connected components from image"),
            DomainRule::new(1, "Spatial Grouping", "Group nearby blobs into regions"),
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        // Vision transforms are applied during to_graph, not as separate rules
        match rule_id {
            0 | 1 => Ok(graph.clone()),
            _ => Err(grapheme_core::DomainError::InvalidInput(format!(
                "Unknown rule ID: {}",
                rule_id
            ))),
        }
    }

    fn generate_examples(&self, _count: usize) -> Vec<DomainExample> {
        // Vision examples would be image-graph pairs
        // For now, return empty (images are loaded externally)
        Vec::new()
    }

    /// Returns all semantic node types that VisionBrain can produce.
    ///
    /// Vision brain uses Pixel nodes for spatial positions in images.
    /// The grid dimensions are determined by the image size, so we return
    /// a representative set of pixel positions.
    fn node_types(&self) -> Vec<NodeType> {
        let mut types = Vec::new();

        // Pixel nodes for a typical grid (parameterized by max dimensions)
        // Vision typically operates on downsampled images, so we use moderate dimensions
        let max_dim = 64; // Support images up to 64x64
        for row in 0..max_dim {
            for col in 0..max_dim {
                types.push(NodeType::Pixel { row, col });
            }
        }

        // Hidden processing nodes
        types.push(NodeType::Hidden);

        // Output nodes for feature extraction
        types.push(NodeType::Output);

        types
    }
}

// ============================================================================
// GraphAutoencoder Implementation for VisionBrain
// ============================================================================

impl GraphAutoencoder for VisionBrain {
    fn encode(&self, input: &str) -> Result<LatentGraph, AutoencoderError> {
        // VisionBrain encodes image data (serialized as base64 or path)
        // For now, we support text-based image references
        let graph = self.parse(input)
            .map_err(|e| AutoencoderError::EncodingError(e.to_string()))?;
        Ok(LatentGraph::new("vision", graph))
    }

    fn decode(&self, latent: &LatentGraph) -> Result<String, AutoencoderError> {
        self.validate_latent(latent)?;

        // Vision decoding returns the text representation of the graph
        // For actual image reconstruction, training-specific decoders handle this
        Ok(latent.graph.to_text())
    }

    fn reconstruction_loss(&self, original: &str, reconstructed: &str) -> f32 {
        // For vision, exact text match is not the goal
        // Instead, we compare structural similarity
        if original == reconstructed {
            return 0.0;
        }

        // Use character-level comparison as a basic metric
        // Real vision loss would compare pixel values or features
        let max_len = original.len().max(reconstructed.len()).max(1);
        let matching: usize = original
            .chars()
            .zip(reconstructed.chars())
            .filter(|(a, b)| a == b)
            .count();

        let len_diff = (original.len() as isize - reconstructed.len() as isize).unsigned_abs();
        let accuracy = matching as f32 / max_len as f32;
        let length_penalty = len_diff as f32 / max_len as f32;

        (1.0 - accuracy + length_penalty * 0.5).clamp(0.0, 1.0)
    }
}

// ============================================================================
// ClassificationBrain - Output graph to class label conversion
// ============================================================================

/// Configuration for classification brain.
///
/// Generic configuration for any number of classes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationConfig {
    /// Number of classes
    pub num_classes: usize,
    /// Number of output nodes in the GRAPHEME graph (usually equals num_classes)
    pub num_outputs: usize,
    /// Template update momentum (higher = slower adaptation)
    pub template_momentum: f32,
    /// Whether to use structural loss (vs cross-entropy)
    pub use_structural: bool,
}

impl ClassificationConfig {
    /// Create configuration with specified number of classes
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            num_outputs: num_classes,
            template_momentum: 0.9,
            use_structural: true,
        }
    }

    /// Create with custom num_outputs (if different from num_classes)
    pub fn with_outputs(num_classes: usize, num_outputs: usize) -> Self {
        Self {
            num_classes,
            num_outputs,
            template_momentum: 0.9,
            use_structural: true,
        }
    }

    /// Set template momentum
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.template_momentum = momentum;
        self
    }

    /// Enable/disable structural loss
    pub fn with_structural(mut self, use_structural: bool) -> Self {
        self.use_structural = use_structural;
        self
    }
}

/// Classification result from ClassificationBrain.
#[derive(Debug, Clone)]
pub struct ClassificationOutput {
    /// Predicted class index
    pub predicted_class: usize,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Probabilities for each class
    pub probabilities: Vec<f32>,
    /// Optional: class label string
    pub label: Option<String>,
}

impl ClassificationOutput {
    /// Create a new classification output
    pub fn new(predicted_class: usize, confidence: f32, probabilities: Vec<f32>) -> Self {
        Self {
            predicted_class,
            confidence,
            probabilities,
            label: None,
        }
    }

    /// Add label for the predicted class
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// ClassificationBrain converts GRAPHEME output graphs to class labels.
///
/// This brain implements the output side of the image classification pipeline:
/// ```text
/// VisionBrain → Input Graph → GRAPHEME Core → Output Graph → ClassificationBrain → Class
/// ```
///
/// Uses StructuralClassifier from grapheme-core for template-based classification
/// instead of softmax/cross-entropy (GRAPHEME-native approach).
#[derive(Debug, Clone)]
pub struct ClassificationBrain {
    config: DomainConfig,
    classification_config: ClassificationConfig,
    classifier: StructuralClassifier,
    /// Class labels (e.g., ["cat", "dog", ...] or ["0", "1", ...])
    labels: Vec<String>,
}

impl ClassificationBrain {
    /// Create a new ClassificationBrain with the given configuration.
    pub fn new(classification_config: ClassificationConfig) -> Self {
        let classifier = StructuralClassifier::new(
            classification_config.num_classes,
            classification_config.num_outputs,
        ).with_momentum(classification_config.template_momentum);

        // Generate default numeric labels
        let labels: Vec<String> = (0..classification_config.num_classes)
            .map(|i| i.to_string())
            .collect();

        let config = DomainConfig::new(
            "classification",
            "Classification Brain",
            vec!["classify", "predict", "label", "class"],
        );

        Self {
            config,
            classification_config,
            classifier,
            labels,
        }
    }

    /// Create a ClassificationBrain with custom class labels.
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        if labels.len() == self.classification_config.num_classes {
            self.labels = labels;
        }
        self
    }

    /// Get the structural classifier (for training).
    pub fn classifier(&self) -> &StructuralClassifier {
        &self.classifier
    }

    /// Get mutable access to the classifier (for template updates during training).
    pub fn classifier_mut(&mut self) -> &mut StructuralClassifier {
        &mut self.classifier
    }

    /// Classify a GRAPHEME output graph.
    ///
    /// Extracts output node activations and uses structural matching
    /// to find the closest class template.
    pub fn classify(&self, graph: &DagNN) -> ClassificationOutput {
        let (predicted_class, distance) = graph.structural_classify(&self.classifier);
        let probabilities = self.classifier.distance_to_probs(&graph.get_classification_logits());

        // Convert distance to confidence (smaller distance = higher confidence)
        let confidence = (-distance).exp().min(1.0);

        let mut output = ClassificationOutput::new(predicted_class, confidence, probabilities);
        if let Some(label) = self.labels.get(predicted_class) {
            output = output.with_label(label.clone());
        }
        output
    }

    /// Get loss and gradient for training.
    ///
    /// Returns the structural loss and gradient with respect to output activations.
    pub fn loss_and_gradient(
        &self,
        graph: &DagNN,
        target_class: usize,
    ) -> StructuralClassificationResult {
        graph.structural_classification_step(&self.classifier, target_class)
    }

    /// Update templates based on observed activations (call during training).
    pub fn update_templates(&mut self, activations: &[f32], true_class: usize) {
        self.classifier.update_template(true_class, activations);
    }

    /// Get the class label for an index.
    pub fn get_label(&self, class_idx: usize) -> Option<&str> {
        self.labels.get(class_idx).map(|s| s.as_str())
    }

    /// Get all class labels.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Get number of classes.
    pub fn num_classes(&self) -> usize {
        self.classification_config.num_classes
    }
}

// ============================================================================
// BaseDomainBrain Implementation for ClassificationBrain
// ============================================================================

impl BaseDomainBrain for ClassificationBrain {
    fn config(&self) -> &DomainConfig {
        &self.config
    }
}

// ============================================================================
// DomainBrain Implementation for ClassificationBrain
// ============================================================================

impl DomainBrain for ClassificationBrain {
    fn domain_id(&self) -> &str {
        &self.config.domain_id
    }

    fn domain_name(&self) -> &str {
        &self.config.domain_name
    }

    fn version(&self) -> &str {
        &self.config.version
    }

    fn can_process(&self, input: &str) -> bool {
        self.default_can_process(input)
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        self.default_parse(input)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        self.default_from_core(graph)
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        self.default_to_core(graph)
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        self.default_validate(graph)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        // Classification-specific execution: return predicted class
        let result = self.classify(graph);
        Ok(ExecutionResult::Text(format!(
            "Predicted class: {} ({}), confidence: {:.2}%",
            result.predicted_class,
            result.label.unwrap_or_default(),
            result.confidence * 100.0
        )))
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        vec![
            DomainRule::new(0, "Structural Matching", "Match output to class templates"),
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        match rule_id {
            0 => Ok(graph.clone()),
            _ => Err(grapheme_core::DomainError::InvalidInput(format!(
                "Unknown rule ID: {}",
                rule_id
            ))),
        }
    }

    fn generate_examples(&self, _count: usize) -> Vec<DomainExample> {
        Vec::new()
    }

    /// Returns all semantic node types that ClassificationBrain can produce.
    ///
    /// Classification uses ClassOutput nodes for class predictions.
    /// The number of classes is configured at brain creation time.
    fn node_types(&self) -> Vec<NodeType> {
        let mut types = Vec::new();

        // ClassOutput nodes for each class
        for class_idx in 0..self.classification_config.num_classes {
            types.push(NodeType::ClassOutput(class_idx));
        }

        // Output node
        types.push(NodeType::Output);

        types
    }
}

// ============================================================================
// ImageClassificationModel - Complete End-to-End Pipeline
// ============================================================================

/// Configuration for ImageClassificationModel
///
/// Generic configuration for any image classification task.
/// Dataset-specific defaults (like MNIST's 10 classes) are set in Default impl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageClassificationConfig {
    /// Vision brain configuration
    pub vision: FeatureConfig,
    /// Classification brain configuration
    pub classification: ClassificationConfig,
    /// Hidden layer size for DAG construction
    pub hidden_size: usize,
    /// Learning rate (Adam default: 0.001)
    pub learning_rate: f32,
    /// Adam beta1: exponential decay rate for first moment (default: 0.9)
    pub beta1: f32,
    /// Adam beta2: exponential decay rate for second moment (default: 0.999)
    pub beta2: f32,
    /// Adam epsilon: numerical stability constant (default: 1e-8)
    pub epsilon: f32,
    /// Weight decay / L2 regularization (default: 0.0)
    pub weight_decay: f32,
    /// Weight for gradient descent contribution (0.0 to 1.0)
    pub gradient_weight: f32,
    /// Weight for Hebbian contribution (0.0 to 1.0)
    pub hebbian_weight: f32,
    /// Whether to use hybrid learning (gradient + Hebbian) vs pure Hebbian
    pub use_hybrid_learning: bool,
    /// Expected number of VisionGraph input nodes (for DagNN sizing)
    pub max_vision_nodes: usize,
}

impl Default for ImageClassificationConfig {
    fn default() -> Self {
        // MNIST-specific defaults: 10 digit classes, grid-based features
        // Grid 10x10 = 100 nodes + 1 root = 101 nodes
        // max_vision_nodes = 101 to include root node
        // Adam optimizer defaults from "Adam: A Method for Stochastic Optimization"
        Self {
            vision: FeatureConfig::default()
                .grid_sampling(10) // 10x10 grid = 100 feature nodes + 1 root
                .with_blob_threshold(0.2)
                .with_min_blob_size(3)
                .with_max_blobs(50),
            classification: ClassificationConfig::new(10), // 10 digit classes
            hidden_size: 64,
            // Adam hyperparameters (Kingma & Ba, 2014)
            learning_rate: 0.001, // Adam default (was 0.01)
            beta1: 0.9,           // First moment decay
            beta2: 0.999,         // Second moment decay
            epsilon: 1e-8,        // Numerical stability
            weight_decay: 0.0,    // L2 regularization (AdamW style)
            gradient_weight: 0.7,
            hebbian_weight: 0.3,
            use_hybrid_learning: true,
            max_vision_nodes: 101, // 10x10 grid + 1 root node
        }
    }
}

impl ImageClassificationConfig {
    /// Create with custom hidden size
    pub fn with_hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }

    /// Create with custom template momentum
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.classification.template_momentum = momentum;
        self
    }

    /// Create with custom learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Create with hybrid learning weights
    pub fn with_hybrid_weights(mut self, gradient: f32, hebbian: f32) -> Self {
        self.gradient_weight = gradient;
        self.hebbian_weight = hebbian;
        self.use_hybrid_learning = true;
        self
    }

    /// Use pure Hebbian learning (no gradients)
    pub fn with_pure_hebbian(mut self) -> Self {
        self.use_hybrid_learning = false;
        self
    }

    /// Set Adam beta1 (first moment decay rate, default: 0.9)
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set Adam beta2 (second moment decay rate, default: 0.999)
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set Adam epsilon (numerical stability, default: 1e-8)
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set weight decay / L2 regularization (default: 0.0)
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Configure all Adam optimizer hyperparameters at once
    pub fn with_adam(mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) -> Self {
        self.learning_rate = lr;
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.epsilon = epsilon;
        self.weight_decay = weight_decay;
        self
    }
}

/// Forward pass result from ImageClassificationModel
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// Predicted class (0-9)
    pub predicted_class: usize,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Class label string
    pub label: String,
    /// Vision graph node count
    pub vision_nodes: usize,
    /// Vision graph edge count
    pub vision_edges: usize,
    /// Whether prediction was correct (if target provided)
    pub correct: Option<bool>,
}

/// Training step result from ImageClassificationModel
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Structural loss
    pub loss: f32,
    /// Predicted class
    pub predicted_class: usize,
    /// Whether prediction was correct
    pub correct: bool,
    /// Gradient for output nodes
    pub gradient: Vec<f32>,
}

/// ImageClassificationModel combines VisionBrain + GRAPHEME Core + ClassificationBrain
/// into a complete end-to-end pipeline for image classification.
///
/// Pipeline:
/// ```text
/// Image → VisionBrain → VisionGraph → DagNN → Forward Pass → ClassificationBrain → Class
///       (deterministic)             (convert)    (learn)      (structural match)
/// ```
///
/// Key properties:
/// - **Deterministic input**: Same image always produces the same input graph
/// - **Persistent DagNN**: Graph structure and weights persist across training samples
/// - **Live learning**: Hebbian/hybrid learning updates both weights AND structure
/// - **Structural classification**: Template matching, no softmax
/// - **Generic**: Works with any image size and number of classes
#[derive(Clone)]
pub struct ImageClassificationModel {
    /// Vision brain for image-to-graph conversion
    vision: VisionBrain,
    /// Classification brain for output-to-class conversion
    classification: ClassificationBrain,
    /// Persistent DagNN that learns across samples (the core innovation)
    dag: DagNN,
    /// Model configuration
    config: ImageClassificationConfig,
    /// Number of training samples seen (for statistics)
    samples_seen: usize,
    /// Adam optimizer state for unified learning across all parameters
    adam: AdamState,
}

impl ImageClassificationModel {
    /// Create a new model with default configuration
    ///
    /// Uses ImageClassificationConfig::default() which provides sensible defaults.
    /// Override with with_config() for custom configurations.
    pub fn new() -> Self {
        Self::with_config(ImageClassificationConfig::default())
    }

    /// Create with custom configuration
    ///
    /// Initializes a persistent DagNN sized for VisionGraph output:
    /// - max_vision_nodes input nodes (configurable for VisionGraph size)
    /// - hidden_size hidden nodes
    /// - num_classes output nodes (from ClassificationConfig)
    /// - Dynamic Xavier-initialized edge weights (GRAPHEME protocol)
    pub fn with_config(config: ImageClassificationConfig) -> Self {
        let classification = ClassificationBrain::new(config.classification.clone());

        // Create DagNN with full network architecture (AGI Mesh Ready)
        let dag = Self::build_network(
            config.max_vision_nodes,
            config.hidden_size,
            classification.classifier().num_classes(),
        );

        Self {
            vision: VisionBrain::new().with_feature_config(config.vision.clone()),
            classification,
            dag,
            config,
            samples_seen: 0,
            adam: AdamState::new(),
        }
    }

    /// Build a fully-connected feedforward network: input -> hidden -> output
    /// Uses simple 1.0 weight init (dynamic √n normalization at activation time)
    fn build_network(num_inputs: usize, hidden_size: usize, num_classes: usize) -> DagNN {
        use grapheme_core::Edge;

        let mut dag = DagNN::new();

        // Layer 1: Input nodes (placeholder characters for vision grid)
        let mut input_nodes = Vec::with_capacity(num_inputs);
        for i in 0..num_inputs {
            let ch = (i % 256) as u8 as char;
            let node = dag.add_character(ch, i);
            input_nodes.push(node);
        }

        // Layer 2: Hidden nodes
        let mut hidden_nodes = Vec::with_capacity(hidden_size);
        for _ in 0..hidden_size {
            hidden_nodes.push(dag.add_hidden());
        }

        // Layer 3: Output nodes (one per class)
        let mut output_nodes = Vec::with_capacity(num_classes);
        for _ in 0..num_classes {
            output_nodes.push(dag.add_output());
        }

        // Simple 1.0 weight initialization (GRAPHEME protocol: NO Xavier)
        // Dynamic √n normalization applied at activation time in forward pass

        // Connect input -> hidden with weight 1.0
        for &input_node in &input_nodes {
            for &hidden_node in &hidden_nodes {
                dag.add_edge(input_node, hidden_node, Edge::semantic(1.0));
            }
        }

        // Connect hidden -> output with weight 1.0
        for &hidden_node in &hidden_nodes {
            for &output_node in &output_nodes {
                dag.add_edge(hidden_node, output_node, Edge::semantic(1.0));
            }
        }

        // Update topology for correct forward pass order
        let _ = dag.update_topology();

        dag
    }

    /// Get model configuration
    pub fn config(&self) -> &ImageClassificationConfig {
        &self.config
    }

    /// Get vision brain reference
    pub fn vision(&self) -> &VisionBrain {
        &self.vision
    }

    /// Get classification brain reference
    pub fn classification(&self) -> &ClassificationBrain {
        &self.classification
    }

    /// Get mutable classification brain for template updates
    pub fn classification_mut(&mut self) -> &mut ClassificationBrain {
        &mut self.classification
    }

    /// Get the persistent DagNN reference
    pub fn dag(&self) -> &DagNN {
        &self.dag
    }

    /// Get mutable DagNN reference for direct manipulation
    pub fn dag_mut(&mut self) -> &mut DagNN {
        &mut self.dag
    }

    /// Get number of training samples seen
    pub fn samples_seen(&self) -> usize {
        self.samples_seen
    }

    /// Get Adam optimizer state reference
    pub fn adam(&self) -> &AdamState {
        &self.adam
    }

    /// Get mutable Adam optimizer state for direct manipulation
    pub fn adam_mut(&mut self) -> &mut AdamState {
        &mut self.adam
    }

    /// Reset Adam optimizer state (useful for learning rate scheduling)
    pub fn reset_optimizer(&mut self) {
        self.adam.reset();
    }

    /// Convert a RawImage to VisionGraph.
    ///
    /// This is the deterministic first stage: same image = same graph.
    /// Works with any image size and format (grayscale or RGB).
    pub fn image_to_vision_graph(&self, image: &RawImage) -> VisionResult<VisionGraph> {
        self.vision.to_graph(image)
    }

    /// Extract input activations from VisionGraph for feeding into DagNN.
    ///
    /// Maps VisionGraph node activations to DagNN input nodes.
    /// The DagNN structure should already match the VisionGraph (via build_dag_from_vision).
    fn vision_to_input_activations(&self, vision_graph: &VisionGraph) -> HashMap<grapheme_core::NodeId, f32> {
        let input_nodes = self.dag.input_nodes();
        let mut input_map = HashMap::new();

        // Map VisionGraph nodes to DagNN input nodes by index
        for (idx, node_idx) in vision_graph.graph.node_indices().enumerate() {
            if idx < input_nodes.len() {
                if let Some(vision_node) = vision_graph.graph.node_weight(node_idx) {
                    input_map.insert(input_nodes[idx], vision_node.activation);
                }
            }
        }

        // If DagNN has more inputs than VisionGraph nodes, set extras to 0
        for node_id in input_nodes.iter().skip(vision_graph.node_count()) {
            input_map.insert(*node_id, 0.0);
        }

        input_map
    }

    /// Run forward pass on a RawImage using the persistent DagNN.
    ///
    /// Pipeline:
    /// 1. VisionBrain: image → VisionGraph (deterministic)
    /// 2. DagNN: VisionGraph activations → forward pass (learnable weights)
    /// 3. ClassificationBrain: output logits → class prediction
    ///
    /// Works with any image size and format (grayscale or RGB).
    pub fn forward(&mut self, image: &RawImage) -> VisionResult<ForwardResult> {
        // Stage 1: VisionBrain - Image to Vision Graph (deterministic)
        let vision_graph = self.image_to_vision_graph(image)?;
        let vision_nodes = vision_graph.node_count();
        let vision_edges = vision_graph.edge_count();

        // Stage 2: DagNN - Extract activations and forward pass
        let input_map = self.vision_to_input_activations(&vision_graph);
        // Convert HashMap to ordered Vec based on input node order
        let input_activations: Vec<f32> = self.dag.input_nodes()
            .iter()
            .map(|&node| *input_map.get(&node).unwrap_or(&0.0))
            .collect();
        let _ = self.dag.forward_with_inputs(&input_activations);

        // Stage 3: ClassificationBrain - output to class
        let result = self.classification.classify(&self.dag);

        Ok(ForwardResult {
            predicted_class: result.predicted_class,
            confidence: result.confidence,
            label: result.label.unwrap_or_else(|| format!("{}", result.predicted_class)),
            vision_nodes,
            vision_edges,
            correct: None,
        })
    }

    /// Run forward pass with target label for accuracy tracking.
    pub fn forward_with_target(&mut self, image: &RawImage, target: usize) -> VisionResult<ForwardResult> {
        let mut result = self.forward(image)?;
        result.correct = Some(result.predicted_class == target);
        Ok(result)
    }

    /// Run training step with Adam optimizer across all learnable parameters.
    ///
    /// This is the core GRAPHEME learning loop with unified Adam optimization:
    /// 1. Feed input activations into persistent DagNN
    /// 2. Forward pass through existing structure
    /// 3. Compute loss via ClassificationBrain
    /// 4. Adam update for DagNN edge weights (gradient + optional Hebbian)
    /// 5. Adam update for ClassificationBrain templates
    ///
    /// Both the DagNN and ClassificationBrain learn with Adam's adaptive learning rates.
    /// Works with any image size and format (grayscale or RGB).
    pub fn train_step(&mut self, image: &RawImage, target: usize) -> VisionResult<TrainResult> {
        // Stage 1: Image to Vision Graph (deterministic)
        let vision_graph = self.image_to_vision_graph(image)?;

        // Stage 2: Extract input activations
        let input_map = self.vision_to_input_activations(&vision_graph);
        // Convert HashMap to ordered Vec based on input node order
        let input_activations: Vec<f32> = self.dag.input_nodes()
            .iter()
            .map(|&node| *input_map.get(&node).unwrap_or(&0.0))
            .collect();

        // Stage 3: Forward pass through persistent DagNN
        let _ = self.dag.forward_with_inputs(&input_activations);

        // Stage 4: Compute loss and gradient via ClassificationBrain
        let struct_result = self.classification.loss_and_gradient(&self.dag, target);

        // Increment Adam timestep (once per training step)
        self.adam.step();

        // Stage 5: Adam update for DagNN edge weights
        self.apply_adam_edge_updates(&struct_result.gradient)?;

        // Stage 6: Adam update for ClassificationBrain templates (only when correct)
        // Only update templates in the right direction - when we correctly classify
        if struct_result.correct {
            let activations = self.dag.get_classification_logits();
            self.apply_adam_template_update(target, &activations);
        }

        self.samples_seen += 1;

        Ok(TrainResult {
            loss: struct_result.loss,
            predicted_class: struct_result.predicted,
            correct: struct_result.correct,
            gradient: struct_result.gradient,
        })
    }

    /// Apply Adam optimizer updates to DagNN edge weights.
    ///
    /// Combines gradient descent with optional Hebbian learning, both using Adam.
    fn apply_adam_edge_updates(&mut self, output_gradient: &[f32]) -> VisionResult<()> {
        // Compute gradients via backpropagation (LeakyReLU)
        #[allow(deprecated)]
        let mut dummy_embedding = Embedding::new(256, 16, InitStrategy::DynamicXavier);
        let grads = self.dag.backward(output_gradient, &mut dummy_embedding);

        // Collect edge updates with Adam
        let mut edge_updates: Vec<(NodeId, NodeId, f32)> = Vec::new();

        // Gradient contribution (scaled by gradient_weight)
        for ((from, to), edge_grad) in &grads.edge_grads {
            // Clip gradient if needed
            let mut grad = *edge_grad * self.config.gradient_weight;
            if grad.abs() > 1.0 {
                grad = grad.signum();
            }

            // Get current weight for AdamW decay
            let current_weight = self.dag.graph
                .find_edge(*from, *to)
                .map(|e| self.dag.graph[e].weight)
                .unwrap_or(0.0);

            // Compute Adam update
            let delta = self.adam.compute_edge_update(
                *from,
                *to,
                grad,
                self.config.learning_rate,
                self.config.beta1,
                self.config.beta2,
                self.config.epsilon,
                self.config.weight_decay,
                current_weight,
            );

            edge_updates.push((*from, *to, delta));
        }

        // Optional Hebbian contribution (scaled by hebbian_weight)
        if self.config.use_hybrid_learning && self.config.hebbian_weight > 0.0 {
            let hebbian_lr = self.config.learning_rate * self.config.hebbian_weight;
            for edge_idx in self.dag.graph.edge_indices() {
                let Some((source, target)) = self.dag.graph.edge_endpoints(edge_idx) else {
                    continue;
                };
                let pre = self.dag.graph[source].activation;
                let post = self.dag.graph[target].activation;

                // Oja's rule for Hebbian: Δw = η * post * (pre - w * post)
                let current_weight = self.dag.graph[edge_idx].weight;
                let hebbian_grad = -post * (pre - current_weight * post); // Negative because we subtract

                // Add Hebbian contribution (use same Adam state for momentum)
                let delta = self.adam.compute_edge_update(
                    source,
                    target,
                    hebbian_grad * self.config.hebbian_weight,
                    hebbian_lr,
                    self.config.beta1,
                    self.config.beta2,
                    self.config.epsilon,
                    0.0, // No weight decay for Hebbian
                    current_weight,
                );

                // Find and update existing entry or add new
                if let Some(entry) = edge_updates.iter_mut().find(|(f, t, _)| *f == source && *t == target) {
                    entry.2 += delta;
                } else {
                    edge_updates.push((source, target, delta));
                }
            }
        }

        // Apply all updates
        for (from, to, delta) in edge_updates {
            if let Some(edge_idx) = self.dag.graph.find_edge(from, to) {
                let current = self.dag.graph[edge_idx].weight;
                let new_weight = (current + delta).clamp(-10.0, 10.0); // Weight bounds
                self.dag.graph[edge_idx].weight = new_weight;
            }
        }

        Ok(())
    }

    /// Apply Adam optimizer update to classification templates.
    ///
    /// Instead of simple EMA, uses Adam to adaptively update template parameters.
    fn apply_adam_template_update(&mut self, target_class: usize, activations: &[f32]) {
        let classifier = self.classification.classifier_mut();

        if target_class >= classifier.templates.len() {
            return;
        }

        let template = &mut classifier.templates[target_class];

        // Compute gradient: template should move toward activations
        // Loss = ||template - activations||^2, so gradient = 2*(template - activations)
        // We want to minimize distance, so update: template -= lr * gradient
        // This simplifies to: gradient = template - activations (direction away from target)

        for (i, &activation) in activations.iter().enumerate() {
            if i >= template.activation_pattern.len() {
                break;
            }

            let current = template.activation_pattern[i];
            let gradient = current - activation; // Gradient pointing away from target

            // Compute Adam update
            let delta = self.adam.compute_template_update(
                target_class,
                i,
                gradient,
                self.config.learning_rate * 0.1, // Templates learn slower
                self.config.beta1,
                self.config.beta2,
                self.config.epsilon,
            );

            template.activation_pattern[i] = (current + delta).clamp(0.0, 1.0);
        }

        template.sample_count += 1;
    }

    /// Batch forward pass for evaluation.
    ///
    /// Returns (accuracy, average_loss, predictions).
    /// Works with any image size and format (grayscale or RGB).
    pub fn evaluate_batch(&mut self, images: &[&RawImage], labels: &[usize]) -> VisionResult<(f32, f32, Vec<usize>)> {
        let mut correct = 0;
        let mut total_loss = 0.0;
        let mut predictions = Vec::with_capacity(images.len());

        for (image, &label) in images.iter().zip(labels.iter()) {
            let result = self.forward_with_target(image, label)?;
            predictions.push(result.predicted_class);
            if result.correct.unwrap_or(false) {
                correct += 1;
            }
            // Use distance-based loss approximation
            total_loss += (1.0 - result.confidence).max(0.0);
        }

        let accuracy = correct as f32 / images.len() as f32;
        let avg_loss = total_loss / images.len() as f32;

        Ok((accuracy, avg_loss, predictions))
    }

    /// Parallel batch evaluation using Rayon.
    ///
    /// Processes multiple images in parallel for faster inference.
    /// Returns (accuracy, average_loss, predictions).
    ///
    /// Note: This creates clones of the DagNN for parallel processing,
    /// so it's best suited for evaluation rather than training.
    pub fn evaluate_batch_parallel(&self, images: &[&RawImage], labels: &[usize]) -> VisionResult<(f32, f32, Vec<usize>)> {
        use rayon::prelude::*;

        // Clone model once upfront for thread-safe parallel access
        let base_model = self.clone();

        // Process all images in parallel
        let results: Vec<_> = images
            .par_iter()
            .zip(labels.par_iter())
            .map(|(image, &label)| {
                // Each thread gets its own copy of the model for inference
                let mut model_clone = base_model.clone();
                match model_clone.forward_with_target(image, label) {
                    Ok(result) => Some((result.predicted_class, result.correct.unwrap_or(false), 1.0 - result.confidence)),
                    Err(_) => None,
                }
            })
            .collect();

        // Aggregate results
        let mut correct = 0;
        let mut total_loss = 0.0;
        let mut predictions = Vec::with_capacity(images.len());
        let mut valid_count = 0;

        for (pred, is_correct, loss) in results.into_iter().flatten() {
            predictions.push(pred);
            if is_correct {
                correct += 1;
            }
            total_loss += loss.max(0.0);
            valid_count += 1;
        }

        if valid_count == 0 {
            return Err(VisionError::FeatureError("No valid results".into()));
        }

        let accuracy = correct as f32 / valid_count as f32;
        let avg_loss = total_loss / valid_count as f32;

        Ok((accuracy, avg_loss, predictions))
    }

    /// Get statistics about the model state
    pub fn stats(&self) -> ModelStats {
        ModelStats {
            samples_seen: self.samples_seen,
            dag_nodes: self.dag.node_count(),
            dag_edges: self.dag.edge_count(),
            num_cliques: self.dag.cliques.len(),
        }
    }
}

/// Statistics about ImageClassificationModel state
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Number of training samples seen
    pub samples_seen: usize,
    /// Number of nodes in the persistent DagNN
    pub dag_nodes: usize,
    /// Number of edges in the persistent DagNN
    pub dag_edges: usize,
    /// Number of cliques detected
    pub num_cliques: usize,
}

impl Default for ImageClassificationModel {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ImageClassificationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageClassificationModel")
            .field("hidden_size", &self.config.hidden_size)
            .field("num_classes", &self.config.classification.num_classes)
            .finish()
    }
}

// ============================================================================
// Parallel Batch Processing
// ============================================================================

/// Process multiple images to graphs in parallel.
///
/// Takes a batch of images and converts them to VisionGraphs using Rayon.
/// This is useful for batch inference or training data preparation.
///
/// # Arguments
/// * `images` - Slice of images to process
/// * `config` - Feature extraction configuration
///
/// # Returns
/// Vec of VisionResults (one per image), preserving order
pub fn batch_images_to_graphs(
    images: &[RawImage],
    config: &FeatureConfig,
) -> Vec<VisionResult<VisionGraph>> {
    images
        .par_iter()
        .map(|image| image_to_graph(image, config))
        .collect()
}

/// Process multiple images to graphs in parallel, filtering errors.
///
/// Like `batch_images_to_graphs` but returns only successful conversions.
/// Use when you want to skip failed images without stopping the batch.
///
/// # Arguments
/// * `images` - Slice of images to process
/// * `config` - Feature extraction configuration
///
/// # Returns
/// Vec of (index, VisionGraph) tuples for successful conversions
pub fn batch_images_to_graphs_ok(
    images: &[RawImage],
    config: &FeatureConfig,
) -> Vec<(usize, VisionGraph)> {
    images
        .par_iter()
        .enumerate()
        .filter_map(|(idx, image)| {
            image_to_graph(image, config).ok().map(|g| (idx, g))
        })
        .collect()
}

/// Statistics from parallel batch processing
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total images processed
    pub total: usize,
    /// Successful conversions
    pub success: usize,
    /// Failed conversions
    pub failed: usize,
    /// Total nodes created (across all graphs)
    pub total_nodes: usize,
    /// Total edges created (across all graphs)
    pub total_edges: usize,
}

/// Process multiple images with statistics tracking.
///
/// Returns both the graphs and statistics about the batch processing.
pub fn batch_images_to_graphs_with_stats(
    images: &[RawImage],
    config: &FeatureConfig,
) -> (Vec<VisionResult<VisionGraph>>, BatchStats) {
    let results: Vec<VisionResult<VisionGraph>> = batch_images_to_graphs(images, config);

    let mut stats = BatchStats {
        total: results.len(),
        ..Default::default()
    };

    for result in &results {
        match result {
            Ok(graph) => {
                stats.success += 1;
                stats.total_nodes += graph.graph.node_count();
                stats.total_edges += graph.graph.edge_count();
            }
            Err(_) => {
                stats.failed += 1;
            }
        }
    }

    (results, stats)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_image_grayscale() {
        let pixels = vec![0.5f32; 100];
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        assert_eq!(image.width, 10);
        assert_eq!(image.height, 10);
        assert_eq!(image.channels, 1);
        assert_eq!(image.pixel_count(), 100);
    }

    #[test]
    fn test_raw_image_grayscale_28x28() {
        // Test creating a 28x28 grayscale image (like MNIST)
        let pixels = vec![0.0f32; 784];
        let image = RawImage::grayscale(28, 28, pixels).unwrap();
        assert_eq!(image.width, 28);
        assert_eq!(image.height, 28);
        assert_eq!(image.channels, 1);
    }

    #[test]
    fn test_raw_image_get_pixel() {
        let mut pixels = vec![0.0f32; 9];
        pixels[4] = 1.0; // Center pixel
        let image = RawImage::grayscale(3, 3, pixels).unwrap();
        assert!((image.get_pixel(1, 1) - 1.0).abs() < 1e-6);
        assert!((image.get_pixel(0, 0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_extract_blobs_simple() {
        // Create a simple image with one blob in the center
        let mut pixels = vec![0.0f32; 100];
        // 3x3 blob in center
        for y in 4..7 {
            for x in 4..7 {
                pixels[y * 10 + x] = 1.0;
            }
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let config = FeatureConfig::default();
        let blobs = extract_blobs(&image, &config);

        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].pixels.len(), 9);
    }

    #[test]
    fn test_extract_blobs_multiple() {
        // Create image with two separate blobs
        let mut pixels = vec![0.0f32; 100];
        // Blob 1: top-left
        for y in 0..3 {
            for x in 0..3 {
                pixels[y * 10 + x] = 1.0;
            }
        }
        // Blob 2: bottom-right
        for y in 7..10 {
            for x in 7..10 {
                pixels[y * 10 + x] = 1.0;
            }
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let config = FeatureConfig::default();
        let blobs = extract_blobs(&image, &config);

        assert_eq!(blobs.len(), 2);
    }

    #[test]
    fn test_image_to_graph() {
        let mut pixels = vec![0.0f32; 100];
        // Create a blob
        for y in 4..7 {
            for x in 4..7 {
                pixels[y * 10 + x] = 0.8;
            }
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let config = FeatureConfig::default();
        let graph = image_to_graph(&image, &config).unwrap();

        assert!(graph.root.is_some());
        assert!(graph.node_count() >= 2); // Root + at least one blob
    }

    #[test]
    fn test_vision_brain_to_graph() {
        let brain = VisionBrain::new();
        assert_eq!(brain.domain_id(), "vision");

        // Test with a small blank image (generic, not MNIST-specific)
        let pixels = vec![0.0f32; 100]; // 10x10
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let graph = brain.to_graph(&image).unwrap();
        assert!(graph.root.is_some());
    }

    #[test]
    fn test_vision_brain_deterministic() {
        let brain = VisionBrain::new();

        // Create a test image with some structure
        let mut pixels = vec![0.0f32; 100]; // 10x10
        for pixel in pixels.iter_mut().take(70).skip(30) {
            *pixel = 0.8;
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        // Convert twice - should produce identical graphs
        let graph1 = brain.to_graph(&image).unwrap();
        let graph2 = brain.to_graph(&image).unwrap();

        assert_eq!(graph1.node_count(), graph2.node_count());
        assert_eq!(graph1.edge_count(), graph2.edge_count());
    }

    #[test]
    fn test_vision_node_activation() {
        let blob_node = new_vision_node(VisionNodeType::Blob {
            cx: 0.5,
            cy: 0.5,
            size: 10,
            intensity: 0.7,
        });
        assert!((blob_node.activation - 0.7).abs() < 1e-6);

        let root_node = new_vision_node(VisionNodeType::ImageRoot {
            width: 28,
            height: 28,
        });
        assert!((root_node.activation - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blobs_adjacent() {
        let blob1 = Blob {
            pixels: vec![(0, 0), (1, 0), (0, 1), (1, 1)],
            center: (0.5, 0.5),
            intensity: 1.0,
            bbox: (0, 0, 2, 2),
        };
        let blob2 = Blob {
            pixels: vec![(3, 0), (4, 0)],
            center: (3.5, 0.0),
            intensity: 1.0,
            bbox: (3, 0, 2, 1),
        };
        let blob3 = Blob {
            pixels: vec![(10, 10)],
            center: (10.0, 10.0),
            intensity: 1.0,
            bbox: (10, 10, 1, 1),
        };

        assert!(blobs_adjacent(&blob1, &blob2)); // Close enough
        assert!(!blobs_adjacent(&blob1, &blob3)); // Far apart
    }

    // ========================================================================
    // Spatial Relationship Tests
    // ========================================================================

    #[test]
    fn test_spatial_relationship_from_blobs() {
        let blob_a = Blob {
            pixels: vec![(5, 5)],
            center: (5.0, 5.0),
            intensity: 1.0,
            bbox: (5, 5, 1, 1),
        };
        let blob_b = Blob {
            pixels: vec![(15, 5)],
            center: (15.0, 5.0),
            intensity: 1.0,
            bbox: (15, 5, 1, 1),
        };

        let rel = SpatialRelationship::from_blobs(&blob_a, &blob_b, 20, 20);

        // B is to the right of A
        assert!(rel.dx > 0.0);
        assert!(rel.dy.abs() < 0.01); // Same row
        assert!(rel.is_right_of());
        assert_eq!(rel.primary_direction(), Some(VisionEdge::RightOf));
    }

    #[test]
    fn test_spatial_relationship_above_below() {
        let blob_top = Blob {
            pixels: vec![(10, 2)],
            center: (10.0, 2.0),
            intensity: 1.0,
            bbox: (10, 2, 1, 1),
        };
        let blob_bottom = Blob {
            pixels: vec![(10, 18)],
            center: (10.0, 18.0),
            intensity: 1.0,
            bbox: (10, 18, 1, 1),
        };

        let rel = SpatialRelationship::from_blobs(&blob_top, &blob_bottom, 20, 20);

        // Bottom is below Top
        assert!(rel.dy > 0.0);
        assert!(rel.is_below());
        assert_eq!(rel.primary_direction(), Some(VisionEdge::Below));

        // Reverse: Top is above Bottom
        let rel_reverse = SpatialRelationship::from_blobs(&blob_bottom, &blob_top, 20, 20);
        assert!(rel_reverse.dy < 0.0);
        assert!(rel_reverse.is_above());
        assert_eq!(rel_reverse.primary_direction(), Some(VisionEdge::Above));
    }

    #[test]
    fn test_spatial_relationship_distance() {
        let blob_a = Blob {
            pixels: vec![(0, 0)],
            center: (0.0, 0.0),
            intensity: 1.0,
            bbox: (0, 0, 1, 1),
        };
        let blob_b = Blob {
            pixels: vec![(10, 0)],
            center: (10.0, 0.0),
            intensity: 1.0,
            bbox: (10, 0, 1, 1),
        };

        let rel = SpatialRelationship::from_blobs(&blob_a, &blob_b, 10, 10);

        // Distance should be normalized to diagonal (sqrt(100+100) = 14.14)
        // Distance 10 / 14.14 ≈ 0.707
        assert!(rel.distance > 0.5 && rel.distance < 0.8);
    }

    #[test]
    fn test_compute_spatial_relationships() {
        let blobs = vec![
            Blob {
                pixels: vec![(2, 2)],
                center: (2.0, 2.0),
                intensity: 1.0,
                bbox: (2, 2, 1, 1),
            },
            Blob {
                pixels: vec![(8, 2)],
                center: (8.0, 2.0),
                intensity: 1.0,
                bbox: (8, 2, 1, 1),
            },
            Blob {
                pixels: vec![(2, 8)],
                center: (2.0, 8.0),
                intensity: 1.0,
                bbox: (2, 8, 1, 1),
            },
        ];

        let relationships = compute_spatial_relationships(&blobs, 10, 10, 1.0);

        // Should have 3 pairs: (0,1), (0,2), (1,2)
        assert_eq!(relationships.len(), 3);

        // Check that relationships are computed
        for (i, j, rel) in &relationships {
            assert!(*i < *j);
            assert!(rel.distance > 0.0);
        }
    }

    #[test]
    fn test_vision_edge_directional() {
        assert_eq!(VisionEdge::Above, VisionEdge::Above);
        assert_ne!(VisionEdge::Above, VisionEdge::Below);
        assert_eq!(VisionEdge::Proximity(0.5), VisionEdge::Proximity(0.5));
    }

    #[test]
    fn test_image_to_graph_with_spatial() {
        // Create two blobs in different positions (closer together for spatial edges)
        let mut pixels = vec![0.0f32; 400]; // 20x20
        // Blob 1: left side
        for y in 8..12 {
            for x in 3..7 {
                pixels[y * 20 + x] = 0.8;
            }
        }
        // Blob 2: right side (close enough for directional relationship)
        for y in 8..12 {
            for x in 13..17 {
                pixels[y * 20 + x] = 0.8;
            }
        }

        let image = RawImage::grayscale(20, 20, pixels).unwrap();
        let mut config = FeatureConfig::default()
            .blob_detection(); // Use blob detection mode for this test
        config.build_spatial_graph = true;
        config.build_hierarchy = false;
        config.max_hierarchy_levels = 1;
        config.adjacency_threshold = 0.5; // Allow more distant relationships

        let graph = image_to_graph(&image, &config).unwrap();

        // Should have root + 2 blobs = 3 nodes (blob detection mode)
        assert_eq!(graph.node_count(), 3);
        // Should have:
        // - 2 Contains edges (root → blob1, root → blob2)
        // - Directional edges (blob1 → blob2 RightOf, blob2 → blob1 LeftOf)
        // - Proximity edges (if within threshold)
        assert!(graph.edge_count() >= 4, "Expected at least 4 edges (2 contains + 2 directional), got {}", graph.edge_count());
    }

    // ========================================================================
    // Hierarchical Blob Detection Tests
    // ========================================================================

    #[test]
    fn test_hierarchical_blob_new() {
        let blob = Blob {
            pixels: vec![(0, 0), (1, 0)],
            center: (0.5, 0.0),
            intensity: 0.8,
            bbox: (0, 0, 2, 1),
        };
        let hblob = HierarchicalBlob::new(blob.clone(), 1, 0.5);
        assert_eq!(hblob.level, 1);
        assert!((hblob.scale - 0.5).abs() < 1e-6);
        assert!(hblob.parent.is_none());
        assert!(hblob.children.is_empty());
    }

    #[test]
    fn test_hierarchical_blob_contains() {
        let parent = HierarchicalBlob::new(
            Blob {
                pixels: vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
                center: (1.0, 0.5),
                intensity: 0.8,
                bbox: (0, 0, 3, 2),
            },
            0,
            1.0,
        );
        let child_inside = Blob {
            pixels: vec![(1, 0)],
            center: (1.0, 0.0),
            intensity: 0.9,
            bbox: (1, 0, 1, 1),
        };
        let child_outside = Blob {
            pixels: vec![(5, 5)],
            center: (5.0, 5.0),
            intensity: 0.7,
            bbox: (5, 5, 1, 1),
        };

        assert!(parent.contains(&child_inside));
        assert!(!parent.contains(&child_outside));
    }

    #[test]
    fn test_extract_hierarchical_blobs_single_blob() {
        // Single blob should result in single-element hierarchy
        let mut pixels = vec![0.0f32; 100];
        for y in 4..7 {
            for x in 4..7 {
                pixels[y * 10 + x] = 0.8;
            }
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let config = FeatureConfig {
            max_hierarchy_levels: 2,
            ..Default::default()
        };

        let hierarchy = extract_hierarchical_blobs(&image, &config);
        // Should find blobs at multiple scales
        assert!(hierarchy.num_levels == 2);
    }

    #[test]
    fn test_extract_hierarchical_blobs_multi_scale() {
        // Create image with varying intensities (should produce hierarchy)
        let mut pixels = vec![0.0f32; 400]; // 20x20
        // Large low-intensity region
        for y in 2..18 {
            for x in 2..18 {
                pixels[y * 20 + x] = 0.4;
            }
        }
        // Smaller high-intensity region inside
        for y in 6..14 {
            for x in 6..14 {
                pixels[y * 20 + x] = 0.7;
            }
        }
        // Even smaller very high intensity core
        for y in 8..12 {
            for x in 8..12 {
                pixels[y * 20 + x] = 0.95;
            }
        }

        let image = RawImage::grayscale(20, 20, pixels).unwrap();
        let config = FeatureConfig {
            max_hierarchy_levels: 3,
            blob_threshold: 0.3,
            ..Default::default()
        };

        let hierarchy = extract_hierarchical_blobs(&image, &config);

        // Should have multiple levels
        assert_eq!(hierarchy.num_levels, 3);
        // Should have some blobs
        assert!(!hierarchy.blobs.is_empty());
    }

    #[test]
    fn test_blob_hierarchy_parent_child_links() {
        // Create nested blobs
        let mut pixels = vec![0.0f32; 400]; // 20x20
        // Outer region (lower intensity)
        for y in 2..18 {
            for x in 2..18 {
                pixels[y * 20 + x] = 0.4;
            }
        }
        // Inner region (higher intensity)
        for y in 6..14 {
            for x in 6..14 {
                pixels[y * 20 + x] = 0.8;
            }
        }

        let image = RawImage::grayscale(20, 20, pixels).unwrap();
        let config = FeatureConfig {
            max_hierarchy_levels: 2,
            blob_threshold: 0.3,
            ..Default::default()
        };

        let hierarchy = extract_hierarchical_blobs(&image, &config);

        // Check that parent-child relationships are consistent
        for (idx, hblob) in hierarchy.blobs.iter().enumerate() {
            // If has parent, parent should have this as child
            if let Some(parent_idx) = hblob.parent {
                assert!(hierarchy.blobs[parent_idx].children.contains(&idx));
            }
            // If has children, children should have this as parent
            for &child_idx in &hblob.children {
                assert_eq!(hierarchy.blobs[child_idx].parent, Some(idx));
            }
        }
    }

    #[test]
    fn test_image_to_graph_hierarchical() {
        // Create nested structure
        let mut pixels = vec![0.0f32; 400]; // 20x20
        for y in 2..18 {
            for x in 2..18 {
                pixels[y * 20 + x] = 0.5;
            }
        }
        for y in 6..14 {
            for x in 6..14 {
                pixels[y * 20 + x] = 0.9;
            }
        }

        let image = RawImage::grayscale(20, 20, pixels).unwrap();
        let config = FeatureConfig {
            build_hierarchy: true,
            max_hierarchy_levels: 2,
            blob_threshold: 0.3,
            ..Default::default()
        };

        let graph = image_to_graph(&image, &config).unwrap();

        // Should have root node
        assert!(graph.root.is_some());
        // Should have blob nodes (at least one from each scale)
        assert!(graph.node_count() > 1);
        // Should have hierarchy edges
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_image_to_graph_single_level_fallback() {
        // With max_hierarchy_levels = 1, should use single-level detection
        let mut pixels = vec![0.0f32; 100];
        for y in 4..7 {
            for x in 4..7 {
                pixels[y * 10 + x] = 0.8;
            }
        }

        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let config = FeatureConfig {
            build_hierarchy: true,
            max_hierarchy_levels: 1, // Single level
            ..Default::default()
        };

        let graph = image_to_graph(&image, &config).unwrap();

        assert!(graph.root.is_some());
        assert!(graph.node_count() >= 2); // Root + at least one blob
    }

    // ========================================================================
    // ClassificationBrain Tests (Generic - no MNIST-specific code)
    // ========================================================================

    #[test]
    fn test_classification_config_new() {
        let config = ClassificationConfig::new(5);
        assert_eq!(config.num_classes, 5);
        assert_eq!(config.num_outputs, 5);
        assert!(config.template_momentum > 0.0);
        assert!(config.use_structural);
    }

    #[test]
    fn test_classification_config_with_outputs() {
        let config = ClassificationConfig::with_outputs(5, 8)
            .with_momentum(0.8)
            .with_structural(false);
        assert_eq!(config.num_classes, 5);
        assert_eq!(config.num_outputs, 8);
        assert!((config.template_momentum - 0.8).abs() < 1e-6);
        assert!(!config.use_structural);
    }

    #[test]
    fn test_classification_brain_new() {
        let brain = ClassificationBrain::new(ClassificationConfig::new(5));
        assert_eq!(brain.domain_id(), "classification");
        assert_eq!(brain.num_classes(), 5);
        assert_eq!(brain.labels().len(), 5);
    }

    #[test]
    fn test_classification_brain_with_labels() {
        let labels = vec![
            "cat".to_string(),
            "dog".to_string(),
            "bird".to_string(),
        ];
        let brain = ClassificationBrain::new(ClassificationConfig::new(3))
            .with_labels(labels);
        assert_eq!(brain.get_label(0), Some("cat"));
        assert_eq!(brain.get_label(1), Some("dog"));
        assert_eq!(brain.get_label(2), Some("bird"));
    }

    #[test]
    fn test_classification_output_new() {
        let probs = vec![0.8, 0.1, 0.1];
        let output = ClassificationOutput::new(0, 0.8, probs.clone());
        assert_eq!(output.predicted_class, 0);
        assert!((output.confidence - 0.8).abs() < 1e-6);
        assert_eq!(output.probabilities, probs);
        assert!(output.label.is_none());
    }

    #[test]
    fn test_classification_output_with_label() {
        let output = ClassificationOutput::new(0, 0.8, vec![0.8, 0.2])
            .with_label("cat");
        assert_eq!(output.label, Some("cat".to_string()));
    }

    #[test]
    fn test_classification_brain_classify() {
        let brain = ClassificationBrain::new(ClassificationConfig::new(5));

        // Create a DagNN with classifier structure
        let classifier = brain.classifier();
        let dag = DagNN::with_classifier(classifier);

        let result = brain.classify(&dag);
        assert!(result.predicted_class < 5);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.probabilities.len(), 5);
    }

    #[test]
    fn test_classification_brain_domain_brain_trait() {
        let brain = ClassificationBrain::new(ClassificationConfig::new(5));

        // Test DomainBrain trait methods
        assert_eq!(brain.domain_id(), "classification");
        assert_eq!(brain.domain_name(), "Classification Brain");
        assert!(brain.can_process("classify this"));
        assert!(!brain.can_process("hello world"));

        let rules = brain.get_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].name, "Structural Matching");
    }

    #[test]
    fn test_classification_brain_classifier_access() {
        let mut brain = ClassificationBrain::new(ClassificationConfig::new(5));

        // Test read access
        let classifier = brain.classifier();
        assert_eq!(classifier.templates.len(), 5);

        // Test mutable access
        let classifier_mut = brain.classifier_mut();
        assert_eq!(classifier_mut.templates.len(), 5);
    }

    #[test]
    fn test_classification_brain_execute() {
        let brain = ClassificationBrain::new(ClassificationConfig::new(5));

        // Create a DagNN with classifier structure
        let classifier = brain.classifier();
        let dag = DagNN::with_classifier(classifier);

        let result = brain.execute(&dag);
        assert!(result.is_ok());
        if let Ok(grapheme_core::ExecutionResult::Text(text)) = result {
            assert!(text.contains("Predicted class:"));
            assert!(text.contains("confidence:"));
        }
    }

    // ========================================================================
    // ImageClassificationModel Tests (Generic - no MNIST-specific code)
    // ========================================================================

    #[test]
    fn test_image_classification_config_default() {
        let config = ImageClassificationConfig::default();
        assert_eq!(config.hidden_size, 64);
        assert!(config.classification.num_classes > 0);
        // FeatureConfig::default() uses max_hierarchy_levels = 3
        assert!(config.vision.max_hierarchy_levels > 0);
    }

    #[test]
    fn test_image_classification_config_builder() {
        let config = ImageClassificationConfig::default()
            .with_hidden_size(128)
            .with_momentum(0.95);

        assert_eq!(config.hidden_size, 128);
        assert!((config.classification.template_momentum - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_image_classification_model_new() {
        let model = ImageClassificationModel::new();
        assert_eq!(model.config().hidden_size, 64);
        assert!(model.config().classification.num_classes > 0);
    }

    #[test]
    fn test_image_classification_model_forward_small() {
        // Test with a small 10x10 grayscale image (generic, not MNIST-specific)
        let mut model = ImageClassificationModel::new();
        let pixels = vec![0.0f32; 100]; // 10x10
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        let result = model.forward(&image);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.predicted_class < model.config().classification.num_classes);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_image_classification_model_forward_with_blob() {
        // Test with an image that has a visible blob
        let mut model = ImageClassificationModel::new();
        let mut pixels = vec![0.0f32; 100]; // 10x10
        // Add a bright blob in the center
        for y in 3..7 {
            for x in 3..7 {
                pixels[y * 10 + x] = 0.9;
            }
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        let result = model.forward(&image);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.vision_nodes >= 1); // At least one node detected
    }

    #[test]
    fn test_image_classification_model_forward_with_target() {
        let mut model = ImageClassificationModel::new();
        let pixels = vec![0.0f32; 100]; // 10x10
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        let result = model.forward_with_target(&image, 0);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.correct.is_some());
    }

    #[test]
    fn test_image_classification_model_train_step() {
        let mut model = ImageClassificationModel::new();
        let pixels = vec![0.0f32; 100]; // 10x10
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        let result = model.train_step(&image, 0);
        assert!(result.is_ok());

        let train_result = result.unwrap();
        assert!(train_result.loss >= 0.0);
        assert!(train_result.predicted_class < model.config().classification.num_classes);
        assert!(!train_result.gradient.is_empty());

        // Verify the persistent DAG has nodes
        assert!(model.dag().node_count() > 0);
        // Verify samples_seen is updated
        assert_eq!(model.samples_seen(), 1);
    }

    #[test]
    fn test_image_classification_model_deterministic() {
        let mut model = ImageClassificationModel::new();

        // Create a test image with structure
        let mut pixels = vec![0.0f32; 100]; // 10x10
        for pixel in pixels.iter_mut().take(70).skip(30) {
            *pixel = 0.8;
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        // Run forward twice - vision graph should be identical
        let result1 = model.forward(&image).unwrap();
        let result2 = model.forward(&image).unwrap();

        // Vision graph structure is deterministic
        assert_eq!(result1.vision_nodes, result2.vision_nodes,
            "Same image should produce same number of vision nodes");
        assert_eq!(result1.vision_edges, result2.vision_edges,
            "Same image should produce same number of vision edges");
    }

    #[test]
    fn test_image_classification_model_persistent_learning() {
        let mut model = ImageClassificationModel::new();

        // Train on a few samples
        let pixels = vec![0.5f32; 100]; // 10x10
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        let num_classes = model.config().classification.num_classes;
        for target in 0..std::cmp::min(3, num_classes) {
            let result = model.train_step(&image, target);
            assert!(result.is_ok());
        }

        // Verify learning happened
        assert!(model.samples_seen() >= 1);

        // DAG structure should still be intact
        assert!(model.dag().node_count() > 0);
        assert!(model.dag().edge_count() > 0);
    }

    #[test]
    fn test_image_classification_model_stats() {
        let mut model = ImageClassificationModel::new();
        let pixels = vec![0.0f32; 100]; // 10x10
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        model.train_step(&image, 0).unwrap();

        let stats = model.stats();
        assert_eq!(stats.samples_seen, 1);
        assert!(stats.dag_nodes > 0);
        assert!(stats.dag_edges > 0);
    }

    #[test]
    fn test_adam_state_accumulation() {
        let mut model = ImageClassificationModel::new();
        let pixels = vec![0.5f32; 100]; // 10x10
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        // Adam state should start empty
        assert_eq!(model.adam().timestep(), 0);
        assert_eq!(model.adam().num_edge_params(), 0);
        assert_eq!(model.adam().num_template_params(), 0);

        // Train for a few steps
        for i in 0..5 {
            model.train_step(&image, i % 10).unwrap();
        }

        // Adam state should have accumulated
        assert_eq!(model.adam().timestep(), 5);
        assert!(model.adam().num_edge_params() > 0, "Adam should track edge momentum");
        // Template momentum only accumulates when classification is correct,
        // so we can't guarantee it will be non-zero after just 5 random samples
    }

    #[test]
    fn test_adam_state_reset() {
        let mut model = ImageClassificationModel::new();
        let pixels = vec![0.5f32; 100];
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        // Train to accumulate state
        model.train_step(&image, 0).unwrap();
        model.train_step(&image, 1).unwrap();

        assert!(model.adam().timestep() > 0);
        assert!(model.adam().num_edge_params() > 0);

        // Reset optimizer
        model.reset_optimizer();

        // State should be cleared
        assert_eq!(model.adam().timestep(), 0);
        assert_eq!(model.adam().num_edge_params(), 0);
        assert_eq!(model.adam().num_template_params(), 0);
    }

    #[test]
    fn test_image_classification_model_vision_access() {
        let model = ImageClassificationModel::new();
        assert_eq!(model.vision().domain_id(), "vision");
    }

    #[test]
    fn test_image_classification_model_classification_access() {
        let mut model = ImageClassificationModel::new();
        assert!(model.config().classification.num_classes > 0);
        assert_eq!(model.classification().domain_id(), "classification");
        let _ = model.classification_mut();
    }

    #[test]
    fn test_image_classification_model_debug() {
        let model = ImageClassificationModel::new();
        let debug = format!("{:?}", model);
        assert!(debug.contains("ImageClassificationModel"));
        assert!(debug.contains("hidden_size"));
    }

    #[test]
    fn test_image_classification_model_rgb() {
        // Test with RGB image (generic support for color images)
        let mut model = ImageClassificationModel::new();
        let pixels = vec![0.5f32; 300]; // 10x10 RGB (10*10*3)
        let image = RawImage::rgb(10, 10, pixels).unwrap();

        let result = model.forward(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_weight_persistence_across_samples() {
        // Test that DagNN edge weights persist and change across training samples
        // This is the critical test for backend-140
        use petgraph::visit::EdgeRef;

        let mut model = ImageClassificationModel::new();

        // Create distinct images for different classes
        let mut image1_pixels = vec![0.0f32; 100]; // 10x10
        for pixel in image1_pixels.iter_mut().take(40).skip(20) { *pixel = 0.9; } // Horizontal bar at top
        let image1 = RawImage::grayscale(10, 10, image1_pixels).unwrap();

        let mut image2_pixels = vec![0.0f32; 100]; // 10x10
        for pixel in image2_pixels.iter_mut().take(80).skip(60) { *pixel = 0.9; } // Horizontal bar at bottom
        let image2 = RawImage::grayscale(10, 10, image2_pixels).unwrap();

        // Get initial edge weights (snapshot)
        let initial_weights: Vec<(_, f32)> = model.dag().graph.edge_references()
            .take(10)
            .map(|e| (e.id(), e.weight().weight))
            .collect();

        assert!(!initial_weights.is_empty(), "DagNN should have edges");

        // Train on first sample
        model.train_step(&image1, 0).unwrap();

        // Get weights after first training step (for potential intermediate checks)
        let _after_first: Vec<f32> = initial_weights.iter()
            .map(|(id, _)| model.dag().graph.edge_weight(*id).unwrap().weight)
            .collect();

        // Train on more samples to accumulate changes
        for _ in 0..5 {
            model.train_step(&image1, 0).unwrap();
            model.train_step(&image2, 1).unwrap();
        }

        // Get weights after multiple training steps
        let after_multiple: Vec<f32> = initial_weights.iter()
            .map(|(id, _)| model.dag().graph.edge_weight(*id).unwrap().weight)
            .collect();

        // Verify weights have changed (learning happened)
        let initial_sum: f32 = initial_weights.iter().map(|(_, w)| w.abs()).sum();
        let after_sum: f32 = after_multiple.iter().map(|w| w.abs()).sum();

        // Calculate total weight change
        let total_change: f32 = initial_weights.iter()
            .zip(after_multiple.iter())
            .map(|((_, initial), final_w)| (final_w - initial).abs())
            .sum();

        println!("Initial weight sum: {:.6}", initial_sum);
        println!("After training weight sum: {:.6}", after_sum);
        println!("Total weight change: {:.6}", total_change);
        println!("Samples trained: {}", model.samples_seen());

        // The critical assertion: weights must have changed
        assert!(
            total_change > 1e-6,
            "Weights should change after training! Total change: {:.10}. \
             This means the persistent DagNN is not learning.",
            total_change
        );

        // Also verify samples_seen counter
        assert_eq!(model.samples_seen(), 11, "Should have seen 11 samples");
    }

    #[test]
    fn test_learnable_trait_for_dagnn_integration() {
        // Test that DagNN's Learnable trait works within ImageClassificationModel

        let mut model = ImageClassificationModel::new();

        // Create test image
        let pixels = vec![0.5f32; 100]; // 10x10
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        // Train (which should accumulate gradients internally via Hebbian/hybrid)
        model.train_step(&image, 0).unwrap();

        // Get edge count (num_parameters = edges)
        let num_params = model.dag().graph.edge_count();
        assert!(num_params > 0, "DagNN should have learnable parameters (edges)");
        println!("DagNN has {} learnable parameters (edges)", num_params);
    }

    #[test]
    fn test_gradient_magnitude_and_direction() {
        // Test that structural gradients have reasonable magnitude and direction

        let mut model = ImageClassificationModel::new();

        // Create an image that should look like class 0
        let mut pixels = vec![0.0f32; 100]; // 10x10
        for pixel in pixels.iter_mut().take(40).skip(20) { *pixel = 0.9; }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        // Check vision graph size
        let vision_graph = model.image_to_vision_graph(&image).unwrap();
        println!("VisionGraph: {} nodes, {} edges",
            vision_graph.node_count(), vision_graph.edge_count());

        // Run forward to set up activations
        let forward_result = model.forward(&image).unwrap();
        println!("Forward result: predicted={}, confidence={:.4}",
            forward_result.predicted_class, forward_result.confidence);

        // Check node activations - this is crucial for backprop!
        let activations: Vec<f32> = model.dag().graph.node_weights()
            .map(|n| n.activation)
            .collect();
        let non_zero_activations = activations.iter().filter(|&&a| a.abs() > 0.01).count();
        let avg_activation: f32 = activations.iter().sum::<f32>() / activations.len() as f32;
        let max_activation = activations.iter().cloned().fold(0.0f32, f32::max);

        println!("Node activations: {} total, {} non-zero (>0.01), avg={:.4}, max={:.4}",
            activations.len(), non_zero_activations, avg_activation, max_activation);

        // Now get the loss and gradient for target class 0
        let struct_result = model.classification().loss_and_gradient(model.dag(), 0);

        println!("Loss for target=0: {:.4}", struct_result.loss);
        println!("Gradient magnitude: {:.6}", struct_result.gradient.iter().map(|g| g.abs()).sum::<f32>());
        println!("Gradient (first 5): {:?}", &struct_result.gradient[..struct_result.gradient.len().min(5)]);

        // Gradient should be non-zero when prediction is wrong
        let grad_magnitude: f32 = struct_result.gradient.iter().map(|g| g.abs()).sum();
        assert!(
            grad_magnitude > 1e-6,
            "Gradient should be non-zero. Magnitude: {:.10}",
            grad_magnitude
        );

        // Now train and verify gradients propagate
        let train_result = model.train_step(&image, 0).unwrap();
        println!("Train result: loss={:.4}, correct={}", train_result.loss, train_result.correct);

        // The training gradient should match what we computed
        let train_grad_magnitude: f32 = train_result.gradient.iter().map(|g| g.abs()).sum();
        println!("Train gradient magnitude: {:.6}", train_grad_magnitude);

        // Check edge weights changed after training
        let edge_weights_after: Vec<f32> = model.dag().graph.edge_references()
            .take(20)
            .map(|e| e.weight().weight)
            .collect();
        println!("Sample edge weights after training: {:?}", &edge_weights_after[..edge_weights_after.len().min(5)]);
    }

    #[test]
    fn test_grid_sampling_mode() {
        // Test that grid sampling produces consistent, dense node counts
        let brain = VisionBrain::new().with_feature_config(
            FeatureConfig::default().grid_sampling(5) // 5x5 = 25 nodes + 1 root
        );

        // Create a test image
        let pixels = vec![0.5f32; 100]; // 10x10
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        let graph = brain.to_graph(&image).unwrap();

        // Should have exactly 26 nodes: 1 root + 25 grid cells
        assert_eq!(graph.node_count(), 26, "Grid 5x5 should produce 26 nodes (1 root + 25 grid)");

        // Should have spatial edges (right and down for each cell)
        // Each row has 4 right edges, each column has 4 down edges = 4*5 + 5*4 = 40 edges
        // Plus 25 Contains edges from root to each grid cell = 65 total
        assert!(graph.edge_count() >= 25, "Should have at least 25 Contains edges");
    }

    #[test]
    fn test_grid_sampling_deterministic() {
        // Grid sampling should be deterministic
        let config = FeatureConfig::default().grid_sampling(7);
        let brain = VisionBrain::new().with_feature_config(config);

        let pixels = vec![0.3f32; 100];
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        let graph1 = brain.to_graph(&image).unwrap();
        let graph2 = brain.to_graph(&image).unwrap();

        assert_eq!(graph1.node_count(), graph2.node_count());
        assert_eq!(graph1.edge_count(), graph2.edge_count());

        // Check that node activations are identical
        let activations1: Vec<f32> = graph1.graph.node_weights().map(|n| n.activation).collect();
        let activations2: Vec<f32> = graph2.graph.node_weights().map(|n| n.activation).collect();
        assert_eq!(activations1, activations2);
    }

    #[test]
    fn test_feature_mode_comparison() {
        // Compare node counts across different modes
        let mut pixels = vec![0.0f32; 100];
        for pixel in pixels.iter_mut().take(40).skip(20) { *pixel = 0.9; } // Horizontal bar
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        // Grid mode: fixed node count
        let grid_config = FeatureConfig::default().grid_sampling(10);
        let grid_graph = image_to_graph(&image, &grid_config).unwrap();
        assert_eq!(grid_graph.node_count(), 101, "Grid 10x10 + root = 101 nodes");

        // Blob mode: variable node count based on image content
        let blob_config = FeatureConfig::default().blob_detection();
        let blob_graph = image_to_graph(&image, &blob_config).unwrap();
        assert!(blob_graph.node_count() < 10, "Blob detection should produce fewer nodes for simple image");

        // Hybrid mode: grid + blobs
        let hybrid_config = FeatureConfig::default()
            .with_mode(FeatureMode::Hybrid)
            .with_grid_size(5);
        let hybrid_graph = image_to_graph(&image, &hybrid_config).unwrap();
        assert!(hybrid_graph.node_count() >= 26, "Hybrid should have at least grid nodes");
    }

    // ========================================================================
    // Parallel Processing Tests
    // ========================================================================

    #[test]
    fn test_parallel_grid_sampling_deterministic() {
        // Parallel and sequential should produce identical results
        let pixels = vec![0.5f32; 784];
        let image = RawImage::grayscale(28, 28, pixels).unwrap();

        // Sequential config
        let seq_config = FeatureConfig::default()
            .grid_sampling(10)
            .with_parallel(false);

        // Parallel config (force parallel even for small grid)
        let par_config = FeatureConfig::default()
            .grid_sampling(10)
            .with_parallel(true)
            .with_parallel_threshold(1); // Force parallel

        let seq_graph = image_to_graph(&image, &seq_config).unwrap();
        let par_graph = image_to_graph(&image, &par_config).unwrap();

        assert_eq!(seq_graph.node_count(), par_graph.node_count());
        assert_eq!(seq_graph.edge_count(), par_graph.edge_count());

        // Check activations match
        let seq_activations: Vec<f32> = seq_graph.graph.node_weights()
            .map(|n| n.activation)
            .collect();
        let par_activations: Vec<f32> = par_graph.graph.node_weights()
            .map(|n| n.activation)
            .collect();
        assert_eq!(seq_activations, par_activations);
    }

    #[test]
    fn test_parallel_grid_sampling_large() {
        // Test with larger grid to ensure parallel works correctly
        let pixels = vec![0.5f32; 10000];
        let image = RawImage::grayscale(100, 100, pixels).unwrap();

        let config = FeatureConfig::default()
            .grid_sampling(32) // 32x32 = 1024 grid cells
            .with_parallel(true);

        let graph = image_to_graph(&image, &config).unwrap();

        // 32x32 grid + 1 root = 1025 nodes
        assert_eq!(graph.node_count(), 1025);
    }

    #[test]
    fn test_should_use_parallel() {
        // Test threshold logic
        let config_small = FeatureConfig::default()
            .grid_sampling(8)
            .with_parallel(true)
            .with_parallel_threshold(16);
        assert!(!config_small.should_use_parallel()); // 8 < 16

        let config_large = FeatureConfig::default()
            .grid_sampling(20)
            .with_parallel(true)
            .with_parallel_threshold(16);
        assert!(config_large.should_use_parallel()); // 20 >= 16

        let config_disabled = FeatureConfig::default()
            .grid_sampling(32)
            .with_parallel(false);
        assert!(!config_disabled.should_use_parallel()); // disabled
    }

    #[test]
    fn test_batch_images_to_graphs() {
        // Create batch of images
        let images: Vec<RawImage> = (0..5)
            .map(|i| {
                let pixels = vec![i as f32 / 10.0; 100];
                RawImage::grayscale(10, 10, pixels).unwrap()
            })
            .collect();

        let config = FeatureConfig::default().grid_sampling(5);

        let results = batch_images_to_graphs(&images, &config);

        assert_eq!(results.len(), 5);
        for result in &results {
            assert!(result.is_ok());
            let graph = result.as_ref().unwrap();
            // 5x5 grid + 1 root = 26 nodes
            assert_eq!(graph.node_count(), 26);
        }
    }

    #[test]
    fn test_batch_images_to_graphs_ok() {
        // Create batch with one empty (invalid) image
        let mut images: Vec<RawImage> = (0..4)
            .map(|i| {
                let pixels = vec![i as f32 / 10.0; 100];
                RawImage::grayscale(10, 10, pixels).unwrap()
            })
            .collect();

        // Add empty image (will fail)
        images.push(RawImage {
            width: 0,
            height: 0,
            channels: 1,
            pixels: vec![],
        });

        let config = FeatureConfig::default().grid_sampling(5);

        let results = batch_images_to_graphs_ok(&images, &config);

        // Should have 4 successful results (indices 0-3)
        assert_eq!(results.len(), 4);
        for (idx, _graph) in &results {
            assert!(*idx < 4); // Empty image at index 4 should fail
        }
    }

    #[test]
    fn test_batch_images_to_graphs_with_stats() {
        let images: Vec<RawImage> = (0..10)
            .map(|_| {
                let pixels = vec![0.5f32; 100];
                RawImage::grayscale(10, 10, pixels).unwrap()
            })
            .collect();

        let config = FeatureConfig::default().grid_sampling(5);

        let (results, stats) = batch_images_to_graphs_with_stats(&images, &config);

        assert_eq!(results.len(), 10);
        assert_eq!(stats.total, 10);
        assert_eq!(stats.success, 10);
        assert_eq!(stats.failed, 0);
        // Each graph has 26 nodes (5x5 + root)
        assert_eq!(stats.total_nodes, 260);
    }

    #[test]
    fn test_grid_sample_struct() {
        // Test GridSample creation in parallel sampling
        let pixels = vec![0.8f32; 100];
        let image = RawImage::grayscale(10, 10, pixels).unwrap();

        let samples = sample_grid_parallel(&image, 5);
        assert_eq!(samples.len(), 25); // 5x5

        // Check first sample (top-left)
        let first = &samples[0];
        assert_eq!(first.gx, 0);
        assert_eq!(first.gy, 0);
        assert!((first.cx - 0.1).abs() < 0.01); // (0 + 0.5) / 5 = 0.1
        assert!((first.cy - 0.1).abs() < 0.01);

        // Check intensity is correct
        assert!((first.intensity - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_parallel_preserves_order() {
        // Verify that parallel processing preserves grid cell order
        let pixels = vec![0.5f32; 400];
        let image = RawImage::grayscale(20, 20, pixels).unwrap();

        let seq = sample_grid_sequential(&image, 10);
        let par = sample_grid_parallel(&image, 10);

        assert_eq!(seq.len(), par.len());

        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.gx, p.gx);
            assert_eq!(s.gy, p.gy);
            assert!((s.cx - p.cx).abs() < 1e-6);
            assert!((s.cy - p.cy).abs() < 1e-6);
            assert!((s.intensity - p.intensity).abs() < 1e-6);
        }
    }
}
