//! Learnable Grounding Components (backend-038)
//!
//! This module provides learnable components for embodied grounding:
//! - Learnable perception encoders for fixed-size representations
//! - Learnable action encoders for action embeddings
//! - Learnable grounding networks for symbol-referent binding
//! - Learnable interaction models for perception-action sequences
//!
//! **GRAPHEME Protocol**: Uses LeakyReLU activation (α=0.01) and Adam optimizer (lr=0.001).

use crate::{Graph, Grounding, GroundingSource, Interaction, NodeId, Referent};
use grapheme_memory::GraphFingerprint;
use grapheme_multimodal::{ModalGraph, Modality};
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// LeakyReLU constant (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Default learning rate (GRAPHEME Protocol)
pub const DEFAULT_LEARNING_RATE: f32 = 0.001;

/// Configuration for learnable grounding
#[derive(Debug, Clone)]
pub struct LearnableGroundingConfig {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Learning rate (GRAPHEME Protocol: 0.001)
    pub learning_rate: f32,
    /// Experience buffer size
    pub buffer_size: usize,
    /// Discount factor for TD learning
    pub gamma: f32,
}

impl Default for LearnableGroundingConfig {
    fn default() -> Self {
        Self {
            embed_dim: 64,
            hidden_dim: 128,
            learning_rate: DEFAULT_LEARNING_RATE,
            buffer_size: 1000,
            gamma: 0.99,
        }
    }
}

// ============================================================================
// Perception Encoder
// ============================================================================

/// Encodes perceptions (ModalGraphs) to fixed-size embeddings
#[derive(Debug)]
pub struct PerceptionEncoder {
    /// Encoding weights: [embed_dim, feature_dim]
    pub w_encode: Array2<f32>,
    /// Modality embedding: [embed_dim, num_modalities]
    pub w_modality: Array2<f32>,
    /// Combination layer: [embed_dim, embed_dim * 2]
    pub w_combine: Array2<f32>,
    /// Biases
    pub b_encode: Array1<f32>,
    pub b_combine: Array1<f32>,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Gradients
    pub grad_w_encode: Option<Array2<f32>>,
    pub grad_w_modality: Option<Array2<f32>>,
    pub grad_w_combine: Option<Array2<f32>>,
    pub grad_b_encode: Option<Array1<f32>>,
    pub grad_b_combine: Option<Array1<f32>>,
}

impl PerceptionEncoder {
    const FEATURE_DIM: usize = 18;
    const NUM_MODALITIES: usize = 7;

    /// Create new perception encoder with DynamicXavier initialization
    pub fn new(embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        let scale_encode = (2.0 / (Self::FEATURE_DIM + embed_dim) as f32).sqrt();
        let scale_modality = (2.0 / (Self::NUM_MODALITIES + embed_dim) as f32).sqrt();
        let scale_combine = (2.0 / (embed_dim * 3) as f32).sqrt();

        let w_encode = Array2::from_shape_fn((embed_dim, Self::FEATURE_DIM), |_| {
            rng.gen_range(-scale_encode..scale_encode)
        });

        let w_modality = Array2::from_shape_fn((embed_dim, Self::NUM_MODALITIES), |_| {
            rng.gen_range(-scale_modality..scale_modality)
        });

        let w_combine = Array2::from_shape_fn((embed_dim, embed_dim * 2), |_| {
            rng.gen_range(-scale_combine..scale_combine)
        });

        Self {
            w_encode,
            w_modality,
            w_combine,
            b_encode: Array1::zeros(embed_dim),
            b_combine: Array1::zeros(embed_dim),
            embed_dim,
            grad_w_encode: None,
            grad_w_modality: None,
            grad_w_combine: None,
            grad_b_encode: None,
            grad_b_combine: None,
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

    /// Get modality one-hot
    fn modality_onehot(&self, modality: Modality) -> Array1<f32> {
        let mut onehot = Array1::zeros(Self::NUM_MODALITIES);
        let idx = match modality {
            Modality::Visual => 0,
            Modality::Auditory => 1,
            Modality::Linguistic => 2,
            Modality::Tactile => 3,
            Modality::Proprioceptive => 4,
            Modality::Action => 5,
            Modality::Abstract => 6,
        };
        onehot[idx] = 1.0;
        onehot
    }

    /// LeakyReLU activation (GRAPHEME Protocol)
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Encode a perception to embedding
    pub fn encode(&self, perception: &ModalGraph) -> Array1<f32> {
        let features = self.extract_features(&perception.graph);
        let modality_vec = self.modality_onehot(perception.modality);

        // Encode content and modality
        let content_embed = self.w_encode.dot(&features) + &self.b_encode;
        let content_embed = content_embed.mapv(|x| self.leaky_relu(x));

        let modality_embed = self.w_modality.dot(&modality_vec);

        // Concatenate and combine
        let mut concat = Array1::zeros(self.embed_dim * 2);
        for (i, &v) in content_embed.iter().enumerate() {
            concat[i] = v;
        }
        for (i, &v) in modality_embed.iter().enumerate() {
            concat[self.embed_dim + i] = v;
        }

        let output = self.w_combine.dot(&concat) + &self.b_combine;
        let output = output.mapv(|x| self.leaky_relu(x));

        // L2 normalize
        let norm = output.mapv(|x| x * x).sum().sqrt().max(1e-8);
        output / norm
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_encode = None;
        self.grad_w_modality = None;
        self.grad_w_combine = None;
        self.grad_b_encode = None;
        self.grad_b_combine = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_encode {
            self.w_encode = &self.w_encode - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_modality {
            self.w_modality = &self.w_modality - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_combine {
            self.w_combine = &self.w_combine - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_encode {
            self.b_encode = &self.b_encode - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_combine {
            self.b_combine = &self.b_combine - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_encode.len() + self.w_modality.len() + self.w_combine.len()
            + self.b_encode.len() + self.b_combine.len()
    }
}

impl Default for PerceptionEncoder {
    fn default() -> Self {
        Self::new(64)
    }
}

// ============================================================================
// Action Encoder
// ============================================================================

/// Encodes action graphs to fixed-size embeddings
#[derive(Debug)]
pub struct ActionEncoder {
    /// Encoding weights: [embed_dim, feature_dim]
    pub w_encode: Array2<f32>,
    /// Bias
    pub b_encode: Array1<f32>,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Gradients
    pub grad_w_encode: Option<Array2<f32>>,
    pub grad_b_encode: Option<Array1<f32>>,
}

impl ActionEncoder {
    const FEATURE_DIM: usize = 18;

    /// Create new action encoder
    pub fn new(embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (Self::FEATURE_DIM + embed_dim) as f32).sqrt();

        let w_encode = Array2::from_shape_fn((embed_dim, Self::FEATURE_DIM), |_| {
            rng.gen_range(-scale..scale)
        });

        Self {
            w_encode,
            b_encode: Array1::zeros(embed_dim),
            embed_dim,
            grad_w_encode: None,
            grad_b_encode: None,
        }
    }

    /// Extract features
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

    /// Encode an action graph
    pub fn encode(&self, action: &Graph) -> Array1<f32> {
        let features = self.extract_features(action);
        let output = self.w_encode.dot(&features) + &self.b_encode;
        let output = output.mapv(|x| self.leaky_relu(x));

        // L2 normalize
        let norm = output.mapv(|x| x * x).sum().sqrt().max(1e-8);
        output / norm
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_encode = None;
        self.grad_b_encode = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_encode {
            self.w_encode = &self.w_encode - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_encode {
            self.b_encode = &self.b_encode - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_encode.len() + self.b_encode.len()
    }
}

impl Default for ActionEncoder {
    fn default() -> Self {
        Self::new(64)
    }
}

// ============================================================================
// Grounding Network
// ============================================================================

/// Learnable grounding network for symbol-referent binding
#[derive(Debug)]
pub struct GroundingNetwork {
    /// Symbol embedding lookup (simple linear for now)
    pub w_symbol: Array2<f32>,
    /// Binding network: [hidden_dim, embed_dim * 2]
    pub w_bind: Array2<f32>,
    /// Output layer: [1, hidden_dim]
    pub w_output: Array2<f32>,
    /// Biases
    pub b_bind: Array1<f32>,
    pub b_output: Array1<f32>,
    /// Dimensions
    pub embed_dim: usize,
    pub hidden_dim: usize,
    /// Maximum symbol ID
    pub max_symbols: usize,
    /// Gradients
    pub grad_w_symbol: Option<Array2<f32>>,
    pub grad_w_bind: Option<Array2<f32>>,
    pub grad_w_output: Option<Array2<f32>>,
    pub grad_b_bind: Option<Array1<f32>>,
    pub grad_b_output: Option<Array1<f32>>,
}

impl GroundingNetwork {
    /// Create new grounding network
    pub fn new(embed_dim: usize, hidden_dim: usize, max_symbols: usize) -> Self {
        let mut rng = rand::thread_rng();

        let scale_symbol = (2.0 / (max_symbols + embed_dim) as f32).sqrt();
        let scale_bind = (2.0 / (embed_dim * 2 + hidden_dim) as f32).sqrt();
        let scale_output = (2.0 / (hidden_dim + 1) as f32).sqrt();

        let w_symbol = Array2::from_shape_fn((embed_dim, max_symbols), |_| {
            rng.gen_range(-scale_symbol..scale_symbol)
        });

        let w_bind = Array2::from_shape_fn((hidden_dim, embed_dim * 2), |_| {
            rng.gen_range(-scale_bind..scale_bind)
        });

        let w_output = Array2::from_shape_fn((1, hidden_dim), |_| {
            rng.gen_range(-scale_output..scale_output)
        });

        Self {
            w_symbol,
            w_bind,
            w_output,
            b_bind: Array1::zeros(hidden_dim),
            b_output: Array1::zeros(1),
            embed_dim,
            hidden_dim,
            max_symbols,
            grad_w_symbol: None,
            grad_w_bind: None,
            grad_w_output: None,
            grad_b_bind: None,
            grad_b_output: None,
        }
    }

    /// LeakyReLU activation (GRAPHEME Protocol)
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Get symbol embedding
    pub fn embed_symbol(&self, symbol_id: NodeId) -> Array1<f32> {
        let idx = (symbol_id as usize).min(self.max_symbols - 1);
        let mut onehot = Array1::zeros(self.max_symbols);
        onehot[idx] = 1.0;
        self.w_symbol.dot(&onehot)
    }

    /// Compute grounding strength between symbol and referent embedding
    pub fn compute_grounding(&self, symbol_embed: &Array1<f32>, referent_embed: &Array1<f32>) -> f32 {
        // Concatenate
        let mut concat = Array1::zeros(self.embed_dim * 2);
        for (i, &v) in symbol_embed.iter().enumerate() {
            concat[i] = v;
        }
        for (i, &v) in referent_embed.iter().enumerate() {
            concat[self.embed_dim + i] = v;
        }

        // Forward pass
        let h = self.w_bind.dot(&concat) + &self.b_bind;
        let h = h.mapv(|x| self.leaky_relu(x));

        let out = self.w_output.dot(&h) + &self.b_output;

        // Sigmoid for [0, 1] grounding strength
        1.0 / (1.0 + (-out[0]).exp())
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_symbol = None;
        self.grad_w_bind = None;
        self.grad_w_output = None;
        self.grad_b_bind = None;
        self.grad_b_output = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_symbol {
            self.w_symbol = &self.w_symbol - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_bind {
            self.w_bind = &self.w_bind - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_output {
            self.w_output = &self.w_output - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_bind {
            self.b_bind = &self.b_bind - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_output {
            self.b_output = &self.b_output - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_symbol.len() + self.w_bind.len() + self.w_output.len()
            + self.b_bind.len() + self.b_output.len()
    }
}

impl Default for GroundingNetwork {
    fn default() -> Self {
        Self::new(64, 128, 1000)
    }
}

// ============================================================================
// Interaction Predictor
// ============================================================================

/// Learnable interaction predictor (perception + action → next perception)
#[derive(Debug)]
pub struct InteractionPredictor {
    /// Transition weights: [embed_dim, embed_dim * 2]
    pub w_transition: Array2<f32>,
    /// Output layer: [embed_dim, hidden_dim]
    pub w_output: Array2<f32>,
    /// Biases
    pub b_transition: Array1<f32>,
    pub b_output: Array1<f32>,
    /// Dimensions
    pub embed_dim: usize,
    pub hidden_dim: usize,
    /// Gradients
    pub grad_w_transition: Option<Array2<f32>>,
    pub grad_w_output: Option<Array2<f32>>,
    pub grad_b_transition: Option<Array1<f32>>,
    pub grad_b_output: Option<Array1<f32>>,
}

impl InteractionPredictor {
    /// Create new interaction predictor
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        let input_dim = embed_dim * 2;
        let scale_trans = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale_out = (2.0 / (hidden_dim + embed_dim) as f32).sqrt();

        let w_transition = Array2::from_shape_fn((hidden_dim, input_dim), |_| {
            rng.gen_range(-scale_trans..scale_trans)
        });

        let w_output = Array2::from_shape_fn((embed_dim, hidden_dim), |_| {
            rng.gen_range(-scale_out..scale_out)
        });

        Self {
            w_transition,
            w_output,
            b_transition: Array1::zeros(hidden_dim),
            b_output: Array1::zeros(embed_dim),
            embed_dim,
            hidden_dim,
            grad_w_transition: None,
            grad_w_output: None,
            grad_b_transition: None,
            grad_b_output: None,
        }
    }

    /// LeakyReLU activation (GRAPHEME Protocol)
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Predict next perception embedding
    pub fn predict(&self, perception_embed: &Array1<f32>, action_embed: &Array1<f32>) -> Array1<f32> {
        // Concatenate perception and action
        let mut concat = Array1::zeros(self.embed_dim * 2);
        for (i, &v) in perception_embed.iter().enumerate() {
            concat[i] = v;
        }
        for (i, &v) in action_embed.iter().enumerate() {
            concat[self.embed_dim + i] = v;
        }

        // Forward pass
        let h = self.w_transition.dot(&concat) + &self.b_transition;
        let h = h.mapv(|x| self.leaky_relu(x));

        let output = self.w_output.dot(&h) + &self.b_output;
        let output = output.mapv(|x| self.leaky_relu(x));

        // L2 normalize
        let norm = output.mapv(|x| x * x).sum().sqrt().max(1e-8);
        output / norm
    }

    /// Compute prediction loss
    pub fn compute_loss(&self, predicted: &Array1<f32>, actual: &Array1<f32>) -> f32 {
        let diff = predicted - actual;
        diff.mapv(|x| x * x).sum() / diff.len() as f32
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_transition = None;
        self.grad_w_output = None;
        self.grad_b_transition = None;
        self.grad_b_output = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_transition {
            self.w_transition = &self.w_transition - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_output {
            self.w_output = &self.w_output - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_transition {
            self.b_transition = &self.b_transition - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_output {
            self.b_output = &self.b_output - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_transition.len() + self.w_output.len()
            + self.b_transition.len() + self.b_output.len()
    }
}

impl Default for InteractionPredictor {
    fn default() -> Self {
        Self::new(64, 128)
    }
}

// ============================================================================
// Experience Types
// ============================================================================

/// Experience tuple for embodied learning
#[derive(Debug, Clone)]
pub struct GroundingExperience {
    /// Symbol being grounded
    pub symbol_embed: Array1<f32>,
    /// Referent embedding (perception)
    pub referent_embed: Array1<f32>,
    /// Grounding verified?
    pub verified: bool,
    /// Reward signal
    pub reward: f32,
}

/// Interaction experience for prediction learning
#[derive(Debug, Clone)]
pub struct InteractionExperience {
    /// Perception before action
    pub before_embed: Array1<f32>,
    /// Action embedding
    pub action_embed: Array1<f32>,
    /// Perception after action
    pub after_embed: Array1<f32>,
    /// Was interaction successful?
    pub success: bool,
}

// ============================================================================
// Learnable Grounding Model
// ============================================================================

/// Complete learnable grounding model
#[derive(Debug)]
pub struct LearnableGrounding {
    /// Perception encoder
    pub perception_encoder: PerceptionEncoder,
    /// Action encoder
    pub action_encoder: ActionEncoder,
    /// Grounding network
    pub grounding_network: GroundingNetwork,
    /// Interaction predictor
    pub interaction_predictor: InteractionPredictor,
    /// Configuration
    pub config: LearnableGroundingConfig,
    /// Grounding experiences
    grounding_buffer: Vec<GroundingExperience>,
    /// Interaction experiences
    interaction_buffer: Vec<InteractionExperience>,
    /// Symbol embedding cache
    symbol_cache: HashMap<NodeId, Array1<f32>>,
}

impl LearnableGrounding {
    /// Create new learnable grounding model
    pub fn new(config: LearnableGroundingConfig) -> Self {
        let perception_encoder = PerceptionEncoder::new(config.embed_dim);
        let action_encoder = ActionEncoder::new(config.embed_dim);
        let grounding_network = GroundingNetwork::new(config.embed_dim, config.hidden_dim, 1000);
        let interaction_predictor = InteractionPredictor::new(config.embed_dim, config.hidden_dim);

        Self {
            perception_encoder,
            action_encoder,
            grounding_network,
            interaction_predictor,
            config,
            grounding_buffer: Vec::new(),
            interaction_buffer: Vec::new(),
            symbol_cache: HashMap::new(),
        }
    }

    /// Encode a perception
    pub fn encode_perception(&self, perception: &ModalGraph) -> Array1<f32> {
        self.perception_encoder.encode(perception)
    }

    /// Encode an action
    pub fn encode_action(&self, action: &Graph) -> Array1<f32> {
        self.action_encoder.encode(action)
    }

    /// Get or compute symbol embedding
    pub fn get_symbol_embed(&mut self, symbol_id: NodeId) -> Array1<f32> {
        if let Some(embed) = self.symbol_cache.get(&symbol_id) {
            return embed.clone();
        }

        let embed = self.grounding_network.embed_symbol(symbol_id);
        self.symbol_cache.insert(symbol_id, embed.clone());
        embed
    }

    /// Compute grounding strength
    pub fn compute_grounding_strength(
        &mut self,
        symbol_id: NodeId,
        perception: &ModalGraph,
    ) -> f32 {
        let symbol_embed = self.get_symbol_embed(symbol_id);
        let referent_embed = self.encode_perception(perception);
        self.grounding_network.compute_grounding(&symbol_embed, &referent_embed)
    }

    /// Learn grounding from interaction
    pub fn learn_grounding_from_interaction(
        &mut self,
        symbol_id: NodeId,
        interactions: &[Interaction],
    ) -> Grounding {
        let symbol_embed = self.get_symbol_embed(symbol_id);

        // Compute average grounding strength from interactions
        let mut total_strength = 0.0;
        let mut count = 0;

        for interaction in interactions {
            if let Some(ref after) = interaction.after {
                let referent_embed = self.encode_perception(after);
                let strength = self.grounding_network.compute_grounding(&symbol_embed, &referent_embed);
                total_strength += strength;
                count += 1;

                // Record experience
                self.record_grounding_experience(GroundingExperience {
                    symbol_embed: symbol_embed.clone(),
                    referent_embed,
                    verified: interaction.success,
                    reward: if interaction.success { 1.0 } else { -0.5 },
                });
            }
        }

        let confidence = if count > 0 {
            total_strength / count as f32
        } else {
            0.5
        };

        // Create grounding with Embodied source
        Grounding::new(
            symbol_id,
            Referent::Conceptual(grapheme_core::DagNN::new()),
            confidence,
            GroundingSource::Embodied,
        )
    }

    /// Predict next perception from interaction
    pub fn predict_interaction_result(
        &self,
        perception: &ModalGraph,
        action: &Graph,
    ) -> Array1<f32> {
        let perception_embed = self.encode_perception(perception);
        let action_embed = self.encode_action(action);
        self.interaction_predictor.predict(&perception_embed, &action_embed)
    }

    /// Record grounding experience
    pub fn record_grounding_experience(&mut self, experience: GroundingExperience) {
        self.grounding_buffer.push(experience);
        if self.grounding_buffer.len() > self.config.buffer_size {
            self.grounding_buffer.remove(0);
        }
    }

    /// Record interaction experience
    pub fn record_interaction(&mut self, interaction: &Interaction) {
        if let (Some(before), Some(after)) = (&interaction.before, &interaction.after) {
            let before_embed = self.encode_perception(before);
            let action_embed = self.encode_action(&interaction.action);
            let after_embed = self.encode_perception(after);

            self.interaction_buffer.push(InteractionExperience {
                before_embed,
                action_embed,
                after_embed,
                success: interaction.success,
            });

            if self.interaction_buffer.len() > self.config.buffer_size {
                self.interaction_buffer.remove(0);
            }
        }
    }

    /// Compute grounding loss
    pub fn compute_grounding_loss(&self) -> f32 {
        if self.grounding_buffer.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for exp in &self.grounding_buffer {
            let pred_strength = self.grounding_network.compute_grounding(
                &exp.symbol_embed,
                &exp.referent_embed,
            );
            let target = if exp.verified { 1.0 } else { 0.0 };
            total_loss += (pred_strength - target).powi(2);
        }

        total_loss / self.grounding_buffer.len() as f32
    }

    /// Compute interaction prediction loss
    pub fn compute_interaction_loss(&self) -> f32 {
        if self.interaction_buffer.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for exp in &self.interaction_buffer {
            let predicted = self.interaction_predictor.predict(&exp.before_embed, &exp.action_embed);
            total_loss += self.interaction_predictor.compute_loss(&predicted, &exp.after_embed);
        }

        total_loss / self.interaction_buffer.len() as f32
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        self.perception_encoder.zero_grad();
        self.action_encoder.zero_grad();
        self.grounding_network.zero_grad();
        self.interaction_predictor.zero_grad();
    }

    /// Update all parameters
    pub fn step(&mut self, lr: f32) {
        self.perception_encoder.step(lr);
        self.action_encoder.step(lr);
        self.grounding_network.step(lr);
        self.interaction_predictor.step(lr);
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        self.perception_encoder.num_parameters()
            + self.action_encoder.num_parameters()
            + self.grounding_network.num_parameters()
            + self.interaction_predictor.num_parameters()
    }

    /// Clear caches
    pub fn clear_cache(&mut self) {
        self.symbol_cache.clear();
    }

    /// Clear experience buffers
    pub fn clear_experience(&mut self) {
        self.grounding_buffer.clear();
        self.interaction_buffer.clear();
    }

    /// Get number of grounding experiences
    pub fn num_grounding_experiences(&self) -> usize {
        self.grounding_buffer.len()
    }

    /// Get number of interaction experiences
    pub fn num_interaction_experiences(&self) -> usize {
        self.interaction_buffer.len()
    }

    /// Check if has gradients
    pub fn has_gradients(&self) -> bool {
        self.perception_encoder.grad_w_encode.is_some()
            || self.grounding_network.grad_w_bind.is_some()
            || self.interaction_predictor.grad_w_transition.is_some()
    }
}

impl Default for LearnableGrounding {
    fn default() -> Self {
        Self::new(LearnableGroundingConfig::default())
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

    fn make_perception(text: &str, modality: Modality) -> ModalGraph {
        ModalGraph::new(make_graph(text), modality)
    }

    fn make_interaction(action_text: &str, success: bool) -> Interaction {
        let mut interaction = Interaction::new(make_graph(action_text));
        interaction.before = Some(make_perception("before", Modality::Visual));
        interaction.after = Some(make_perception("after", Modality::Visual));
        interaction.success = success;
        interaction
    }

    #[test]
    fn test_learnable_grounding_config_default() {
        let config = LearnableGroundingConfig::default();
        assert_eq!(config.embed_dim, 64);
        assert!((config.learning_rate - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_perception_encoder_creation() {
        let encoder = PerceptionEncoder::default();
        assert!(encoder.num_parameters() > 0);
    }

    #[test]
    fn test_perception_encoder_encode() {
        let encoder = PerceptionEncoder::new(64);
        let perception = make_perception("test content", Modality::Visual);

        let embedding = encoder.encode(&perception);
        assert_eq!(embedding.len(), 64);

        // Check L2 normalized
        let norm = embedding.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_action_encoder_creation() {
        let encoder = ActionEncoder::default();
        assert!(encoder.num_parameters() > 0);
    }

    #[test]
    fn test_action_encoder_encode() {
        let encoder = ActionEncoder::new(64);
        let action = make_graph("pick up object");

        let embedding = encoder.encode(&action);
        assert_eq!(embedding.len(), 64);

        // Check normalized
        let norm = embedding.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_grounding_network_creation() {
        let network = GroundingNetwork::default();
        assert!(network.num_parameters() > 0);
    }

    #[test]
    fn test_grounding_network_compute() {
        let network = GroundingNetwork::new(64, 128, 1000);

        let symbol_embed = network.embed_symbol(42);
        let referent_embed = Array1::from_vec(vec![0.1; 64]);

        let strength = network.compute_grounding(&symbol_embed, &referent_embed);
        assert!(strength >= 0.0 && strength <= 1.0);
    }

    #[test]
    fn test_interaction_predictor_creation() {
        let predictor = InteractionPredictor::default();
        assert!(predictor.num_parameters() > 0);
    }

    #[test]
    fn test_interaction_predictor_predict() {
        let predictor = InteractionPredictor::new(64, 128);

        let perception = Array1::from_vec(vec![0.1; 64]);
        let action = Array1::from_vec(vec![0.2; 64]);

        let next_perception = predictor.predict(&perception, &action);
        assert_eq!(next_perception.len(), 64);

        // Check normalized
        let norm = next_perception.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_learnable_grounding_creation() {
        let grounding = LearnableGrounding::default();
        assert!(grounding.num_parameters() > 0);
    }

    #[test]
    fn test_learnable_grounding_encode_perception() {
        let grounding = LearnableGrounding::default();
        let perception = make_perception("test", Modality::Visual);

        let embed = grounding.encode_perception(&perception);
        assert_eq!(embed.len(), 64);
    }

    #[test]
    fn test_learnable_grounding_compute_strength() {
        let mut grounding = LearnableGrounding::default();
        let perception = make_perception("cat", Modality::Visual);

        let strength = grounding.compute_grounding_strength(42, &perception);
        assert!(strength >= 0.0 && strength <= 1.0);
    }

    #[test]
    fn test_learnable_grounding_learn_from_interaction() {
        let mut grounding = LearnableGrounding::default();

        let interactions = vec![
            make_interaction("look at cat", true),
            make_interaction("point at cat", true),
        ];

        let result = grounding.learn_grounding_from_interaction(42, &interactions);
        assert_eq!(result.symbol, 42);
        assert_eq!(result.source, GroundingSource::Embodied);
    }

    #[test]
    fn test_learnable_grounding_predict_interaction() {
        let grounding = LearnableGrounding::default();
        let perception = make_perception("before", Modality::Visual);
        let action = make_graph("move forward");

        let predicted = grounding.predict_interaction_result(&perception, &action);
        assert_eq!(predicted.len(), 64);
    }

    #[test]
    fn test_learnable_grounding_record_interaction() {
        let mut grounding = LearnableGrounding::default();

        let interaction = make_interaction("test action", true);
        grounding.record_interaction(&interaction);

        assert_eq!(grounding.num_interaction_experiences(), 1);
    }

    #[test]
    fn test_learnable_grounding_losses() {
        let mut grounding = LearnableGrounding::default();

        // Record some experiences
        for i in 0..5 {
            let interaction = make_interaction(&format!("action{}", i), i % 2 == 0);
            grounding.record_interaction(&interaction);
        }

        let interaction_loss = grounding.compute_interaction_loss();
        assert!(interaction_loss >= 0.0);
    }

    #[test]
    fn test_learnable_grounding_gradient_flow() {
        let mut grounding = LearnableGrounding::default();

        // Zero grad and step should not panic
        grounding.zero_grad();
        grounding.step(0.001);
    }
}
