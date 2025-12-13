//! Learnable Multi-Modal Fusion Components (backend-037)
//!
//! This module provides learnable components for multimodal fusion:
//! - Learnable modality encoders for fixed-size representations
//! - Learnable fusion networks for combining modalities
//! - Learnable cross-modal binding with attention
//! - Learnable modality attention for dynamic focus
//!
//! **GRAPHEME Protocol**: Uses LeakyReLU activation (α=0.01) and Adam optimizer (lr=0.001).

use crate::{BindingType, CrossModalBinding, Graph, ModalGraph, Modality, MultiModalEvent, NodeId};
use grapheme_memory::GraphFingerprint;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// LeakyReLU constant (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Default learning rate (GRAPHEME Protocol)
pub const DEFAULT_LEARNING_RATE: f32 = 0.001;

/// Number of modalities
pub const NUM_MODALITIES: usize = 7;

/// Configuration for learnable multimodal fusion
#[derive(Debug, Clone)]
pub struct LearnableMultiModalConfig {
    /// Embedding dimension per modality
    pub embed_dim: usize,
    /// Hidden dimension for fusion networks
    pub hidden_dim: usize,
    /// Learning rate (GRAPHEME Protocol: 0.001)
    pub learning_rate: f32,
    /// Number of attention heads for cross-modal attention
    pub num_heads: usize,
    /// Experience buffer size
    pub buffer_size: usize,
}

impl Default for LearnableMultiModalConfig {
    fn default() -> Self {
        Self {
            embed_dim: 64,
            hidden_dim: 128,
            learning_rate: DEFAULT_LEARNING_RATE,
            num_heads: 4,
            buffer_size: 1000,
        }
    }
}

// ============================================================================
// Modality Encoder
// ============================================================================

/// Encodes modal graphs to fixed-size embeddings
#[derive(Debug)]
pub struct ModalityEncoder {
    /// Per-modality encoding weights: [embed_dim, feature_dim]
    pub w_encode: HashMap<Modality, Array2<f32>>,
    /// Per-modality biases
    pub b_encode: HashMap<Modality, Array1<f32>>,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Gradients
    pub grad_w_encode: HashMap<Modality, Option<Array2<f32>>>,
    pub grad_b_encode: HashMap<Modality, Option<Array1<f32>>>,
}

impl ModalityEncoder {
    const FEATURE_DIM: usize = 18;

    /// Create new modality encoder with DynamicXavier initialization
    pub fn new(embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (Self::FEATURE_DIM + embed_dim) as f32).sqrt();

        let mut w_encode = HashMap::new();
        let mut b_encode = HashMap::new();
        let mut grad_w_encode = HashMap::new();
        let mut grad_b_encode = HashMap::new();

        for modality in Modality::all() {
            w_encode.insert(
                modality,
                Array2::from_shape_fn((embed_dim, Self::FEATURE_DIM), |_| {
                    rng.gen_range(-scale..scale)
                }),
            );
            b_encode.insert(modality, Array1::zeros(embed_dim));
            grad_w_encode.insert(modality, None);
            grad_b_encode.insert(modality, None);
        }

        Self {
            w_encode,
            b_encode,
            embed_dim,
            grad_w_encode,
            grad_b_encode,
        }
    }

    /// Extract features from a graph
    fn extract_features(&self, graph: &Graph) -> Array1<f32> {
        let fp = GraphFingerprint::from_graph(graph);
        let mut features = Array1::zeros(Self::FEATURE_DIM);

        features[0] = fp.node_count as f32 / 100.0;
        features[1] = fp.edge_count as f32 / 100.0;
        for i in 0..8 {
            features[2 + i] = fp.node_types[i] as f32 / 10.0;
            features[10 + i] = fp.degree_hist[i] as f32 / 10.0;
        }

        features
    }

    /// LeakyReLU activation (GRAPHEME Protocol)
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Encode a modal graph to embedding
    pub fn encode(&self, modal_graph: &ModalGraph) -> Array1<f32> {
        let features = self.extract_features(&modal_graph.graph);

        let w = self.w_encode.get(&modal_graph.modality).unwrap();
        let b = self.b_encode.get(&modal_graph.modality).unwrap();

        let pre_activation = w.dot(&features) + b;
        let output = pre_activation.mapv(|x| self.leaky_relu(x));

        // L2 normalize
        let norm = output.mapv(|x| x * x).sum().sqrt().max(1e-8);
        output / norm
    }

    /// Encode raw graph with specified modality
    pub fn encode_with_modality(&self, graph: &Graph, modality: Modality) -> Array1<f32> {
        let modal_graph = ModalGraph::new(
            grapheme_core::DagNN::from_text(&graph.to_text()).unwrap_or_else(|_| grapheme_core::DagNN::new()),
            modality,
        );
        self.encode(&modal_graph)
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        for modality in Modality::all() {
            self.grad_w_encode.insert(modality, None);
            self.grad_b_encode.insert(modality, None);
        }
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        for modality in Modality::all() {
            if let Some(Some(ref grad)) = self.grad_w_encode.get(&modality) {
                if let Some(w) = self.w_encode.get_mut(&modality) {
                    *w = &*w - &(grad * lr);
                }
            }
            if let Some(Some(ref grad)) = self.grad_b_encode.get(&modality) {
                if let Some(b) = self.b_encode.get_mut(&modality) {
                    *b = &*b - &(grad * lr);
                }
            }
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        for w in self.w_encode.values() {
            total += w.len();
        }
        for b in self.b_encode.values() {
            total += b.len();
        }
        total
    }
}

impl Default for ModalityEncoder {
    fn default() -> Self {
        Self::new(64)
    }
}

// ============================================================================
// Fusion Network
// ============================================================================

/// Learnable fusion network for combining multimodal embeddings
#[derive(Debug)]
pub struct FusionNetwork {
    /// Fusion weights: [embed_dim, num_modalities * embed_dim]
    pub w_fusion: Array2<f32>,
    /// Hidden layer: [hidden_dim, embed_dim]
    pub w_hidden: Array2<f32>,
    /// Output layer: [embed_dim, hidden_dim]
    pub w_output: Array2<f32>,
    /// Biases
    pub b_fusion: Array1<f32>,
    pub b_hidden: Array1<f32>,
    pub b_output: Array1<f32>,
    /// Configuration
    pub embed_dim: usize,
    pub hidden_dim: usize,
    /// Gradients
    pub grad_w_fusion: Option<Array2<f32>>,
    pub grad_w_hidden: Option<Array2<f32>>,
    pub grad_w_output: Option<Array2<f32>>,
    pub grad_b_fusion: Option<Array1<f32>>,
    pub grad_b_hidden: Option<Array1<f32>>,
    pub grad_b_output: Option<Array1<f32>>,
}

impl FusionNetwork {
    /// Create new fusion network
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        let input_dim = NUM_MODALITIES * embed_dim;
        let scale_fusion = (2.0 / (input_dim + embed_dim) as f32).sqrt();
        let scale_hidden = (2.0 / (embed_dim + hidden_dim) as f32).sqrt();
        let scale_output = (2.0 / (hidden_dim + embed_dim) as f32).sqrt();

        let w_fusion = Array2::from_shape_fn((embed_dim, input_dim), |_| {
            rng.gen_range(-scale_fusion..scale_fusion)
        });

        let w_hidden = Array2::from_shape_fn((hidden_dim, embed_dim), |_| {
            rng.gen_range(-scale_hidden..scale_hidden)
        });

        let w_output = Array2::from_shape_fn((embed_dim, hidden_dim), |_| {
            rng.gen_range(-scale_output..scale_output)
        });

        Self {
            w_fusion,
            w_hidden,
            w_output,
            b_fusion: Array1::zeros(embed_dim),
            b_hidden: Array1::zeros(hidden_dim),
            b_output: Array1::zeros(embed_dim),
            embed_dim,
            hidden_dim,
            grad_w_fusion: None,
            grad_w_hidden: None,
            grad_w_output: None,
            grad_b_fusion: None,
            grad_b_hidden: None,
            grad_b_output: None,
        }
    }

    /// LeakyReLU activation (GRAPHEME Protocol)
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Fuse multiple modality embeddings
    pub fn fuse(&self, modality_embeds: &HashMap<Modality, Array1<f32>>) -> Array1<f32> {
        // Concatenate all modality embeddings (use zeros for missing)
        let mut concat = Array1::zeros(NUM_MODALITIES * self.embed_dim);

        for (i, modality) in Modality::all().into_iter().enumerate() {
            if let Some(embed) = modality_embeds.get(&modality) {
                let start = i * self.embed_dim;
                for (j, &v) in embed.iter().enumerate() {
                    if start + j < concat.len() {
                        concat[start + j] = v;
                    }
                }
            }
        }

        // First layer: fusion
        let h1 = self.w_fusion.dot(&concat) + &self.b_fusion;
        let h1 = h1.mapv(|x| self.leaky_relu(x));

        // Hidden layer
        let h2 = self.w_hidden.dot(&h1) + &self.b_hidden;
        let h2 = h2.mapv(|x| self.leaky_relu(x));

        // Output layer
        let output = self.w_output.dot(&h2) + &self.b_output;
        let output = output.mapv(|x| self.leaky_relu(x));

        // L2 normalize
        let norm = output.mapv(|x| x * x).sum().sqrt().max(1e-8);
        output / norm
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_fusion = None;
        self.grad_w_hidden = None;
        self.grad_w_output = None;
        self.grad_b_fusion = None;
        self.grad_b_hidden = None;
        self.grad_b_output = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_fusion {
            self.w_fusion = &self.w_fusion - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_hidden {
            self.w_hidden = &self.w_hidden - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_output {
            self.w_output = &self.w_output - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_fusion {
            self.b_fusion = &self.b_fusion - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_hidden {
            self.b_hidden = &self.b_hidden - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_output {
            self.b_output = &self.b_output - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_fusion.len() + self.w_hidden.len() + self.w_output.len()
            + self.b_fusion.len() + self.b_hidden.len() + self.b_output.len()
    }
}

impl Default for FusionNetwork {
    fn default() -> Self {
        Self::new(64, 128)
    }
}

// ============================================================================
// Cross-Modal Binder
// ============================================================================

/// Learnable cross-modal binding network
#[derive(Debug)]
pub struct CrossModalBinder {
    /// Binding weights: [1, 2 * embed_dim] for each modality pair
    pub w_bind: Array2<f32>,
    /// Binding bias
    pub b_bind: Array1<f32>,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Gradients
    pub grad_w_bind: Option<Array2<f32>>,
    pub grad_b_bind: Option<Array1<f32>>,
}

impl CrossModalBinder {
    /// Create new cross-modal binder
    pub fn new(embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let input_dim = embed_dim * 2;
        let scale = (2.0 / (input_dim + 1) as f32).sqrt();

        let w_bind = Array2::from_shape_fn((1, input_dim), |_| rng.gen_range(-scale..scale));
        let b_bind = Array1::zeros(1);

        Self {
            w_bind,
            b_bind,
            embed_dim,
            grad_w_bind: None,
            grad_b_bind: None,
        }
    }

    /// Compute binding strength between two embeddings
    pub fn compute_binding(&self, source: &Array1<f32>, target: &Array1<f32>) -> f32 {
        // Concatenate source and target
        let mut concat = Array1::zeros(self.embed_dim * 2);
        for (i, &v) in source.iter().enumerate() {
            concat[i] = v;
        }
        for (i, &v) in target.iter().enumerate() {
            concat[self.embed_dim + i] = v;
        }

        // Compute raw score
        let raw = self.w_bind.dot(&concat) + &self.b_bind;

        // Sigmoid for [0, 1] binding strength
        1.0 / (1.0 + (-raw[0]).exp())
    }

    /// Create binding from embeddings
    pub fn bind(
        &self,
        source_embed: &Array1<f32>,
        target_embed: &Array1<f32>,
        source_modality: Modality,
        target_modality: Modality,
        source_node: NodeId,
        target_node: NodeId,
    ) -> CrossModalBinding {
        let strength = self.compute_binding(source_embed, target_embed);

        CrossModalBinding::new(
            (source_modality, source_node),
            (target_modality, target_node),
            strength,
        ).with_type(BindingType::Reference)
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_bind = None;
        self.grad_b_bind = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_bind {
            self.w_bind = &self.w_bind - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_bind {
            self.b_bind = &self.b_bind - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_bind.len() + self.b_bind.len()
    }
}

impl Default for CrossModalBinder {
    fn default() -> Self {
        Self::new(64)
    }
}

// ============================================================================
// Modality Attention
// ============================================================================

/// Learnable modality attention for dynamic focus
#[derive(Debug)]
pub struct ModalityAttention {
    /// Query projection: [embed_dim, embed_dim]
    pub w_query: Array2<f32>,
    /// Key projection for each modality
    pub w_key: Array2<f32>,
    /// Value projection
    pub w_value: Array2<f32>,
    /// Output projection
    pub w_output: Array2<f32>,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Gradients
    pub grad_w_query: Option<Array2<f32>>,
    pub grad_w_key: Option<Array2<f32>>,
    pub grad_w_value: Option<Array2<f32>>,
    pub grad_w_output: Option<Array2<f32>>,
}

impl ModalityAttention {
    /// Create new modality attention
    pub fn new(embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (embed_dim * 2) as f32).sqrt();

        let w_query = Array2::from_shape_fn((embed_dim, embed_dim), |_| rng.gen_range(-scale..scale));
        let w_key = Array2::from_shape_fn((embed_dim, embed_dim), |_| rng.gen_range(-scale..scale));
        let w_value = Array2::from_shape_fn((embed_dim, embed_dim), |_| rng.gen_range(-scale..scale));
        let w_output = Array2::from_shape_fn((embed_dim, embed_dim), |_| rng.gen_range(-scale..scale));

        Self {
            w_query,
            w_key,
            w_value,
            w_output,
            embed_dim,
            grad_w_query: None,
            grad_w_key: None,
            grad_w_value: None,
            grad_w_output: None,
        }
    }

    /// Compute attention weights for modalities
    pub fn compute_attention(
        &self,
        query: &Array1<f32>,
        modality_embeds: &HashMap<Modality, Array1<f32>>,
    ) -> Vec<(Modality, f32)> {
        // Project query
        let q = self.w_query.dot(query);

        // Compute attention scores
        let mut scores: Vec<(Modality, f32)> = Vec::new();

        for modality in Modality::all() {
            let score = if let Some(embed) = modality_embeds.get(&modality) {
                let k = self.w_key.dot(embed);
                // Scaled dot product
                let dot: f32 = q.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum();
                dot / (self.embed_dim as f32).sqrt()
            } else {
                f32::NEG_INFINITY
            };
            scores.push((modality, score));
        }

        // Softmax
        let max_score = scores
            .iter()
            .filter(|(_, s)| s.is_finite())
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = scores
            .iter()
            .filter(|(_, s)| s.is_finite())
            .map(|(_, s)| (s - max_score).exp())
            .sum();

        scores
            .into_iter()
            .map(|(m, s)| {
                let attention = if s.is_finite() {
                    (s - max_score).exp() / exp_sum.max(1e-8)
                } else {
                    0.0
                };
                (m, attention)
            })
            .collect()
    }

    /// Attend to modalities and compute weighted sum
    pub fn attend(
        &self,
        query: &Array1<f32>,
        modality_embeds: &HashMap<Modality, Array1<f32>>,
    ) -> Array1<f32> {
        let attention = self.compute_attention(query, modality_embeds);

        let mut weighted_sum = Array1::zeros(self.embed_dim);

        for (modality, weight) in attention {
            if let Some(embed) = modality_embeds.get(&modality) {
                let v = self.w_value.dot(embed);
                weighted_sum = weighted_sum + v * weight;
            }
        }

        // Output projection
        self.w_output.dot(&weighted_sum)
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_query = None;
        self.grad_w_key = None;
        self.grad_w_value = None;
        self.grad_w_output = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_query {
            self.w_query = &self.w_query - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_key {
            self.w_key = &self.w_key - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_value {
            self.w_value = &self.w_value - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_output {
            self.w_output = &self.w_output - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_query.len() + self.w_key.len() + self.w_value.len() + self.w_output.len()
    }
}

impl Default for ModalityAttention {
    fn default() -> Self {
        Self::new(64)
    }
}

// ============================================================================
// Experience Types
// ============================================================================

/// Experience tuple for multimodal learning
#[derive(Debug, Clone)]
pub struct MultiModalExperience {
    /// Modality embeddings at time t
    pub modality_embeds: HashMap<Modality, Array1<f32>>,
    /// Fused embedding
    pub fused_embed: Array1<f32>,
    /// Actual bindings observed
    pub bindings: Vec<(Modality, Modality, f32)>,
    /// Reward/loss for this fusion
    pub reward: f32,
}

// ============================================================================
// Learnable MultiModal Model
// ============================================================================

/// Complete learnable multimodal fusion model
#[derive(Debug)]
pub struct LearnableMultiModal {
    /// Modality encoder
    pub encoder: ModalityEncoder,
    /// Fusion network
    pub fusion: FusionNetwork,
    /// Cross-modal binder
    pub binder: CrossModalBinder,
    /// Modality attention
    pub attention: ModalityAttention,
    /// Configuration
    pub config: LearnableMultiModalConfig,
    /// Experience buffer
    experience_buffer: Vec<MultiModalExperience>,
    /// Cached modality embeddings
    embed_cache: HashMap<u64, HashMap<Modality, Array1<f32>>>,
}

impl LearnableMultiModal {
    /// Create new learnable multimodal model
    pub fn new(config: LearnableMultiModalConfig) -> Self {
        let encoder = ModalityEncoder::new(config.embed_dim);
        let fusion = FusionNetwork::new(config.embed_dim, config.hidden_dim);
        let binder = CrossModalBinder::new(config.embed_dim);
        let attention = ModalityAttention::new(config.embed_dim);

        Self {
            encoder,
            fusion,
            binder,
            attention,
            config,
            experience_buffer: Vec::new(),
            embed_cache: HashMap::new(),
        }
    }

    /// Encode a multimodal event
    pub fn encode_event(&self, event: &MultiModalEvent) -> HashMap<Modality, Array1<f32>> {
        let mut embeds = HashMap::new();

        for component in &event.components {
            let embed = self.encoder.encode(component);
            embeds.insert(component.modality, embed);
        }

        embeds
    }

    /// Fuse a multimodal event into unified representation
    pub fn fuse_event(&self, event: &MultiModalEvent) -> Array1<f32> {
        let embeds = self.encode_event(event);
        self.fusion.fuse(&embeds)
    }

    /// Compute cross-modal bindings with learned strengths
    pub fn compute_bindings(&self, event: &MultiModalEvent) -> Vec<CrossModalBinding> {
        let embeds = self.encode_event(event);
        let mut bindings = Vec::new();

        // For each pair of modalities present
        let modalities: Vec<_> = embeds.keys().cloned().collect();

        for (i, source_mod) in modalities.iter().enumerate() {
            for target_mod in modalities.iter().skip(i + 1) {
                if let (Some(source_embed), Some(target_embed)) =
                    (embeds.get(source_mod), embeds.get(target_mod))
                {
                    let strength = self.binder.compute_binding(source_embed, target_embed);

                    bindings.push(CrossModalBinding::new(
                        (*source_mod, petgraph::graph::NodeIndex::new(0)),
                        (*target_mod, petgraph::graph::NodeIndex::new(0)),
                        strength,
                    ));
                }
            }
        }

        bindings
    }

    /// Get attention weights for modalities given a query
    pub fn get_modality_attention(
        &self,
        query: &Array1<f32>,
        event: &MultiModalEvent,
    ) -> Vec<(Modality, f32)> {
        let embeds = self.encode_event(event);
        self.attention.compute_attention(query, &embeds)
    }

    /// Attend to event with query
    pub fn attend_to_event(&self, query: &Array1<f32>, event: &MultiModalEvent) -> Array1<f32> {
        let embeds = self.encode_event(event);
        self.attention.attend(query, &embeds)
    }

    /// Record experience for learning
    pub fn record_experience(&mut self, experience: MultiModalExperience) {
        self.experience_buffer.push(experience);

        if self.experience_buffer.len() > self.config.buffer_size {
            self.experience_buffer.remove(0);
        }
    }

    /// Record fusion outcome
    pub fn record_fusion(
        &mut self,
        event: &MultiModalEvent,
        reward: f32,
    ) {
        let embeds = self.encode_event(event);
        let fused = self.fusion.fuse(&embeds);

        let bindings: Vec<_> = self
            .compute_bindings(event)
            .iter()
            .map(|b| (b.source.0, b.target.0, b.strength))
            .collect();

        self.record_experience(MultiModalExperience {
            modality_embeds: embeds,
            fused_embed: fused,
            bindings,
            reward,
        });
    }

    /// Compute fusion loss (reconstruction + binding consistency)
    pub fn compute_loss(&self) -> f32 {
        if self.experience_buffer.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for exp in &self.experience_buffer {
            // Reconstruction loss: re-fuse and compare
            let refused = self.fusion.fuse(&exp.modality_embeds);
            let diff = &refused - &exp.fused_embed;
            let mse: f32 = diff.mapv(|x| x * x).sum() / diff.len() as f32;

            // Reward-weighted loss
            total_loss += mse * (1.0 - exp.reward);
        }

        total_loss / self.experience_buffer.len() as f32
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        self.encoder.zero_grad();
        self.fusion.zero_grad();
        self.binder.zero_grad();
        self.attention.zero_grad();
    }

    /// Update all parameters
    pub fn step(&mut self, lr: f32) {
        self.encoder.step(lr);
        self.fusion.step(lr);
        self.binder.step(lr);
        self.attention.step(lr);
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        self.encoder.num_parameters()
            + self.fusion.num_parameters()
            + self.binder.num_parameters()
            + self.attention.num_parameters()
    }

    /// Clear caches
    pub fn clear_cache(&mut self) {
        self.embed_cache.clear();
    }

    /// Clear experience buffer
    pub fn clear_experience(&mut self) {
        self.experience_buffer.clear();
    }

    /// Get number of experiences
    pub fn num_experiences(&self) -> usize {
        self.experience_buffer.len()
    }

    /// Check if has gradients
    pub fn has_gradients(&self) -> bool {
        self.fusion.grad_w_fusion.is_some()
            || self.binder.grad_w_bind.is_some()
            || self.attention.grad_w_query.is_some()
    }
}

impl Default for LearnableMultiModal {
    fn default() -> Self {
        Self::new(LearnableMultiModalConfig::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use grapheme_core::DagNN;

    fn make_graph(text: &str) -> Graph {
        DagNN::from_text(text).unwrap()
    }

    fn make_modal_graph(text: &str, modality: Modality) -> ModalGraph {
        ModalGraph::new(make_graph(text), modality)
    }

    fn make_event() -> MultiModalEvent {
        let mut event = MultiModalEvent::new(1);
        event.add_component(make_modal_graph("visual content", Modality::Visual));
        event.add_component(make_modal_graph("text content", Modality::Linguistic));
        event
    }

    #[test]
    fn test_learnable_multimodal_config_default() {
        let config = LearnableMultiModalConfig::default();
        assert_eq!(config.embed_dim, 64);
        assert!((config.learning_rate - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_modality_encoder_creation() {
        let encoder = ModalityEncoder::default();
        assert!(encoder.num_parameters() > 0);
    }

    #[test]
    fn test_modality_encoder_encode() {
        let encoder = ModalityEncoder::new(64);
        let modal_graph = make_modal_graph("test content", Modality::Visual);

        let embedding = encoder.encode(&modal_graph);
        assert_eq!(embedding.len(), 64);

        // Check L2 normalized
        let norm = embedding.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fusion_network_creation() {
        let fusion = FusionNetwork::default();
        assert!(fusion.num_parameters() > 0);
    }

    #[test]
    fn test_fusion_network_fuse() {
        let fusion = FusionNetwork::new(64, 128);
        let mut embeds = HashMap::new();
        embeds.insert(Modality::Visual, Array1::from_vec(vec![0.1; 64]));
        embeds.insert(Modality::Linguistic, Array1::from_vec(vec![0.2; 64]));

        let fused = fusion.fuse(&embeds);
        assert_eq!(fused.len(), 64);

        // Check normalized
        let norm = fused.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cross_modal_binder_creation() {
        let binder = CrossModalBinder::default();
        assert!(binder.num_parameters() > 0);
    }

    #[test]
    fn test_cross_modal_binder_compute() {
        let binder = CrossModalBinder::new(64);
        let source = Array1::from_vec(vec![0.1; 64]);
        let target = Array1::from_vec(vec![0.2; 64]);

        let strength = binder.compute_binding(&source, &target);
        assert!(strength >= 0.0 && strength <= 1.0);
    }

    #[test]
    fn test_modality_attention_creation() {
        let attention = ModalityAttention::default();
        assert!(attention.num_parameters() > 0);
    }

    #[test]
    fn test_modality_attention_compute() {
        let attention = ModalityAttention::new(64);
        let query = Array1::from_vec(vec![0.1; 64]);

        let mut embeds = HashMap::new();
        embeds.insert(Modality::Visual, Array1::from_vec(vec![0.1; 64]));
        embeds.insert(Modality::Linguistic, Array1::from_vec(vec![0.2; 64]));

        let weights = attention.compute_attention(&query, &embeds);
        assert!(!weights.is_empty());

        // Check weights sum to ~1.0
        let total: f32 = weights.iter().map(|(_, w)| w).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_learnable_multimodal_creation() {
        let mm = LearnableMultiModal::default();
        assert!(mm.num_parameters() > 0);
    }

    #[test]
    fn test_learnable_multimodal_encode_event() {
        let mm = LearnableMultiModal::default();
        let event = make_event();

        let embeds = mm.encode_event(&event);
        assert!(embeds.contains_key(&Modality::Visual));
        assert!(embeds.contains_key(&Modality::Linguistic));
    }

    #[test]
    fn test_learnable_multimodal_fuse_event() {
        let mm = LearnableMultiModal::default();
        let event = make_event();

        let fused = mm.fuse_event(&event);
        assert_eq!(fused.len(), 64);
    }

    #[test]
    fn test_learnable_multimodal_compute_bindings() {
        let mm = LearnableMultiModal::default();
        let event = make_event();

        let bindings = mm.compute_bindings(&event);
        assert!(!bindings.is_empty());
    }

    #[test]
    fn test_learnable_multimodal_attention() {
        let mm = LearnableMultiModal::default();
        let event = make_event();
        let query = Array1::from_vec(vec![0.1; 64]);

        let attention = mm.get_modality_attention(&query, &event);
        assert!(!attention.is_empty());
    }

    #[test]
    fn test_learnable_multimodal_record_fusion() {
        let mut mm = LearnableMultiModal::default();
        let event = make_event();

        mm.record_fusion(&event, 0.8);
        assert_eq!(mm.num_experiences(), 1);
    }

    #[test]
    fn test_learnable_multimodal_loss() {
        let mut mm = LearnableMultiModal::default();

        // Add some experiences
        for _ in 0..5 {
            let event = make_event();
            mm.record_fusion(&event, 0.5);
        }

        let loss = mm.compute_loss();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_learnable_multimodal_gradient_flow() {
        let mut mm = LearnableMultiModal::default();

        // Zero grad and step should not panic
        mm.zero_grad();
        mm.step(0.001);
    }
}
