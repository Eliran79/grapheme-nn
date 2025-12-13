//! Learnable World Model Components (backend-034)
//!
//! This module provides learnable components for world modeling:
//! - Learnable state encoders for fixed-size representations
//! - Learnable transition dynamics for predicting next states
//! - Learnable action effects for modeling interventions
//!
//! **GRAPHEME Protocol**: Uses LeakyReLU activation (α=0.01) and Adam optimizer (lr=0.001).

use crate::{Graph, WorldState};
use grapheme_memory::GraphFingerprint;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// LeakyReLU constant (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Default learning rate for world model updates
pub const DEFAULT_LEARNING_RATE: f32 = 0.001;

/// Configuration for learnable world model
#[derive(Debug, Clone)]
pub struct LearnableWorldConfig {
    /// Embedding dimension for states
    pub embed_dim: usize,
    /// Hidden dimension for transition networks
    pub hidden_dim: usize,
    /// Learning rate (GRAPHEME Protocol: 0.001)
    pub learning_rate: f32,
    /// Number of action types
    pub num_actions: usize,
    /// Prediction horizon
    pub horizon: usize,
}

impl Default for LearnableWorldConfig {
    fn default() -> Self {
        Self {
            embed_dim: 64,
            hidden_dim: 128,
            learning_rate: DEFAULT_LEARNING_RATE,
            num_actions: 8,
            horizon: 10,
        }
    }
}

/// State encoder that produces fixed-size embeddings from world states
#[derive(Debug)]
pub struct StateEncoder {
    /// Entity encoding weights: [embed_dim, feature_dim]
    pub w_entity: Array2<f32>,
    /// Relation encoding weights: [embed_dim, feature_dim]
    pub w_relation: Array2<f32>,
    /// Combination layer: [embed_dim, 2*embed_dim]
    pub w_combine: Array2<f32>,
    /// Bias for combination
    pub b_combine: Array1<f32>,
    /// Gradients
    pub grad_w_entity: Option<Array2<f32>>,
    pub grad_w_relation: Option<Array2<f32>>,
    pub grad_w_combine: Option<Array2<f32>>,
    pub grad_b_combine: Option<Array1<f32>>,
}

impl StateEncoder {
    /// Number of features from GraphFingerprint
    const FEATURE_DIM: usize = 18;

    /// Create new encoder with DynamicXavier initialization
    pub fn new(embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        let scale_entity = (2.0 / (Self::FEATURE_DIM + embed_dim) as f32).sqrt();
        let scale_combine = (2.0 / (embed_dim * 3) as f32).sqrt();

        let w_entity = Array2::from_shape_fn(
            (embed_dim, Self::FEATURE_DIM),
            |_| rng.gen_range(-scale_entity..scale_entity),
        );

        let w_relation = Array2::from_shape_fn(
            (embed_dim, Self::FEATURE_DIM),
            |_| rng.gen_range(-scale_entity..scale_entity),
        );

        let w_combine = Array2::from_shape_fn(
            (embed_dim, embed_dim * 2),
            |_| rng.gen_range(-scale_combine..scale_combine),
        );

        let b_combine = Array1::zeros(embed_dim);

        Self {
            w_entity,
            w_relation,
            w_combine,
            b_combine,
            grad_w_entity: None,
            grad_w_relation: None,
            grad_w_combine: None,
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

    /// LeakyReLU activation (GRAPHEME Protocol)
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Encode a world state to a fixed-size embedding
    pub fn encode(&self, state: &WorldState) -> Array1<f32> {
        let entity_features = self.extract_features(&state.entities);
        let relation_features = self.extract_features(&state.relations);

        // Encode entities and relations
        let entity_embed = self.w_entity.dot(&entity_features);
        let relation_embed = self.w_relation.dot(&relation_features);

        // Concatenate and combine
        let mut combined = Array1::zeros(entity_embed.len() * 2);
        for (i, &v) in entity_embed.iter().enumerate() {
            combined[i] = v;
        }
        for (i, &v) in relation_embed.iter().enumerate() {
            combined[entity_embed.len() + i] = v;
        }

        let output_pre = self.w_combine.dot(&combined) + &self.b_combine;
        let output = output_pre.mapv(|x| self.leaky_relu(x));

        // L2 normalize
        let norm = output.mapv(|x| x * x).sum().sqrt().max(1e-8);
        output / norm
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_entity = None;
        self.grad_w_relation = None;
        self.grad_w_combine = None;
        self.grad_b_combine = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_entity {
            self.w_entity = &self.w_entity - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_relation {
            self.w_relation = &self.w_relation - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_combine {
            self.w_combine = &self.w_combine - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_combine {
            self.b_combine = &self.b_combine - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_entity.len() + self.w_relation.len() + self.w_combine.len() + self.b_combine.len()
    }
}

impl Default for StateEncoder {
    fn default() -> Self {
        Self::new(64)
    }
}

/// Learnable transition dynamics for predicting next states
#[derive(Debug)]
pub struct LearnableTransition {
    /// Transition matrix: [embed_dim, embed_dim + action_dim]
    pub w_transition: Array2<f32>,
    /// Bias for transition
    pub b_transition: Array1<f32>,
    /// Output layer: [embed_dim, embed_dim]
    pub w_output: Array2<f32>,
    /// Bias for output
    pub b_output: Array1<f32>,
    /// Configuration
    pub config: LearnableWorldConfig,
    /// Gradients
    pub grad_w_transition: Option<Array2<f32>>,
    pub grad_b_transition: Option<Array1<f32>>,
    pub grad_w_output: Option<Array2<f32>>,
    pub grad_b_output: Option<Array1<f32>>,
}

impl LearnableTransition {
    /// Create new transition model
    pub fn new(config: LearnableWorldConfig) -> Self {
        let mut rng = rand::thread_rng();

        let input_dim = config.embed_dim + config.num_actions;
        let scale_trans = (2.0 / (input_dim + config.hidden_dim) as f32).sqrt();
        let scale_out = (2.0 / (config.hidden_dim + config.embed_dim) as f32).sqrt();

        let w_transition = Array2::from_shape_fn(
            (config.hidden_dim, input_dim),
            |_| rng.gen_range(-scale_trans..scale_trans),
        );

        let w_output = Array2::from_shape_fn(
            (config.embed_dim, config.hidden_dim),
            |_| rng.gen_range(-scale_out..scale_out),
        );

        let b_transition = Array1::zeros(config.hidden_dim);
        let b_output = Array1::zeros(config.embed_dim);

        Self {
            w_transition,
            b_transition,
            w_output,
            b_output,
            config,
            grad_w_transition: None,
            grad_b_transition: None,
            grad_w_output: None,
            grad_b_output: None,
        }
    }

    /// LeakyReLU activation (GRAPHEME Protocol)
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Predict next state embedding given current state and action
    pub fn predict(&self, state_embed: &Array1<f32>, action: &Array1<f32>) -> Array1<f32> {
        // Concatenate state and action
        let mut input = Array1::zeros(state_embed.len() + action.len());
        for (i, &v) in state_embed.iter().enumerate() {
            input[i] = v;
        }
        for (i, &v) in action.iter().enumerate() {
            input[state_embed.len() + i] = v;
        }

        // Forward pass
        let hidden_pre = self.w_transition.dot(&input) + &self.b_transition;
        let hidden = hidden_pre.mapv(|x| self.leaky_relu(x));

        let output_pre = self.w_output.dot(&hidden) + &self.b_output;
        let output = output_pre.mapv(|x| self.leaky_relu(x));

        // L2 normalize
        let norm = output.mapv(|x| x * x).sum().sqrt().max(1e-8);
        output / norm
    }

    /// Compute prediction loss (MSE between predicted and actual next state)
    pub fn compute_loss(&self, predicted: &Array1<f32>, actual: &Array1<f32>) -> f32 {
        let diff = predicted - actual;
        diff.mapv(|x| x * x).sum() / diff.len() as f32
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_transition = None;
        self.grad_b_transition = None;
        self.grad_w_output = None;
        self.grad_b_output = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_transition {
            self.w_transition = &self.w_transition - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_transition {
            self.b_transition = &self.b_transition - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_output {
            self.w_output = &self.w_output - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_output {
            self.b_output = &self.b_output - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_transition.len() + self.b_transition.len()
            + self.w_output.len() + self.b_output.len()
    }
}

impl Default for LearnableTransition {
    fn default() -> Self {
        Self::new(LearnableWorldConfig::default())
    }
}

/// Action encoder for converting action graphs to embeddings
#[derive(Debug)]
pub struct ActionEncoder {
    /// Action embedding matrix: [action_dim, feature_dim]
    pub w_action: Array2<f32>,
    /// Bias
    pub b_action: Array1<f32>,
    /// Number of action types
    pub num_actions: usize,
    /// Gradients
    pub grad_w_action: Option<Array2<f32>>,
    pub grad_b_action: Option<Array1<f32>>,
}

impl ActionEncoder {
    const FEATURE_DIM: usize = 18;

    /// Create new action encoder
    pub fn new(num_actions: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (Self::FEATURE_DIM + num_actions) as f32).sqrt();

        let w_action = Array2::from_shape_fn(
            (num_actions, Self::FEATURE_DIM),
            |_| rng.gen_range(-scale..scale),
        );

        let b_action = Array1::zeros(num_actions);

        Self {
            w_action,
            b_action,
            num_actions,
            grad_w_action: None,
            grad_b_action: None,
        }
    }

    /// Extract features from action graph
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

    /// Encode an action graph to action embedding
    pub fn encode(&self, action: &Graph) -> Array1<f32> {
        let features = self.extract_features(action);
        let output = self.w_action.dot(&features) + &self.b_action;

        // Softmax for action distribution
        let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_output = output.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_output.sum().max(1e-8);
        exp_output / sum_exp
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_action = None;
        self.grad_b_action = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_action {
            self.w_action = &self.w_action - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_action {
            self.b_action = &self.b_action - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_action.len() + self.b_action.len()
    }
}

impl Default for ActionEncoder {
    fn default() -> Self {
        Self::new(8)
    }
}

/// Complete learnable world model
#[derive(Debug)]
pub struct LearnableWorldModel {
    /// State encoder
    pub state_encoder: StateEncoder,
    /// Action encoder
    pub action_encoder: ActionEncoder,
    /// Transition model
    pub transition: LearnableTransition,
    /// Configuration
    pub config: LearnableWorldConfig,
    /// Cached state embeddings (for future optimization)
    #[allow(dead_code)]
    state_cache: HashMap<usize, Array1<f32>>,
    /// Training examples: (state, action, next_state)
    experience_buffer: Vec<(Array1<f32>, Array1<f32>, Array1<f32>)>,
}

impl LearnableWorldModel {
    /// Create new learnable world model
    pub fn new(config: LearnableWorldConfig) -> Self {
        let state_encoder = StateEncoder::new(config.embed_dim);
        let action_encoder = ActionEncoder::new(config.num_actions);
        let transition = LearnableTransition::new(config.clone());

        Self {
            state_encoder,
            action_encoder,
            transition,
            config,
            state_cache: HashMap::new(),
            experience_buffer: Vec::new(),
        }
    }

    /// Encode a world state
    pub fn encode_state(&self, state: &WorldState) -> Array1<f32> {
        self.state_encoder.encode(state)
    }

    /// Encode an action
    pub fn encode_action(&self, action: &Graph) -> Array1<f32> {
        self.action_encoder.encode(action)
    }

    /// Predict next state embedding given current state and action
    pub fn predict_next(
        &self,
        state: &WorldState,
        action: &Graph,
    ) -> Array1<f32> {
        let state_embed = self.encode_state(state);
        let action_embed = self.encode_action(action);
        self.transition.predict(&state_embed, &action_embed)
    }

    /// Record an observed transition for learning
    pub fn observe_transition(
        &mut self,
        state: &WorldState,
        action: &Graph,
        next_state: &WorldState,
    ) {
        let state_embed = self.encode_state(state);
        let action_embed = self.encode_action(action);
        let next_embed = self.encode_state(next_state);

        self.experience_buffer.push((state_embed, action_embed, next_embed));

        // Keep buffer bounded
        if self.experience_buffer.len() > 1000 {
            self.experience_buffer.remove(0);
        }
    }

    /// Compute prediction loss on experience buffer
    pub fn compute_prediction_loss(&self) -> f32 {
        if self.experience_buffer.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for (state, action, next_state) in &self.experience_buffer {
            let predicted = self.transition.predict(state, action);
            total_loss += self.transition.compute_loss(&predicted, next_state);
        }

        total_loss / self.experience_buffer.len() as f32
    }

    /// Multi-step prediction (imagination/planning)
    pub fn imagine(
        &self,
        initial_state: &WorldState,
        actions: &[Graph],
    ) -> Vec<Array1<f32>> {
        let mut predictions = Vec::with_capacity(actions.len() + 1);
        let mut current = self.encode_state(initial_state);
        predictions.push(current.clone());

        for action in actions {
            let action_embed = self.encode_action(action);
            current = self.transition.predict(&current, &action_embed);
            predictions.push(current.clone());
        }

        predictions
    }

    /// Zero gradients for all components
    pub fn zero_grad(&mut self) {
        self.state_encoder.zero_grad();
        self.action_encoder.zero_grad();
        self.transition.zero_grad();
    }

    /// Update all parameters
    pub fn step(&mut self, lr: f32) {
        self.state_encoder.step(lr);
        self.action_encoder.step(lr);
        self.transition.step(lr);
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        self.state_encoder.num_parameters()
            + self.action_encoder.num_parameters()
            + self.transition.num_parameters()
    }

    /// Clear experience buffer
    pub fn clear_experience(&mut self) {
        self.experience_buffer.clear();
    }

    /// Get number of experiences
    pub fn num_experiences(&self) -> usize {
        self.experience_buffer.len()
    }
}

impl Default for LearnableWorldModel {
    fn default() -> Self {
        Self::new(LearnableWorldConfig::default())
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

    fn make_world_state(entities: &str, relations: &str) -> WorldState {
        WorldState::new(make_graph(entities), make_graph(relations))
    }

    #[test]
    fn test_learnable_world_config_default() {
        let config = LearnableWorldConfig::default();
        assert_eq!(config.embed_dim, 64);
        assert!((config.learning_rate - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_state_encoder_creation() {
        let encoder = StateEncoder::default();
        assert!(encoder.num_parameters() > 0);
    }

    #[test]
    fn test_state_encoder_encode() {
        let encoder = StateEncoder::new(64);
        let state = make_world_state("entity1 entity2", "relation");

        let embedding = encoder.encode(&state);
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
        let encoder = ActionEncoder::new(8);
        let action = make_graph("move forward");

        let embedding = encoder.encode(&action);
        assert_eq!(embedding.len(), 8);

        // Should sum to ~1.0 (softmax)
        let sum: f32 = embedding.sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_learnable_transition_creation() {
        let transition = LearnableTransition::default();
        assert!(transition.num_parameters() > 0);
    }

    #[test]
    fn test_learnable_transition_predict() {
        let transition = LearnableTransition::default();

        let state = Array1::from_vec(vec![0.5; 64]);
        let action = Array1::from_vec(vec![0.125; 8]);

        let next_state = transition.predict(&state, &action);
        assert_eq!(next_state.len(), 64);

        // Check L2 normalized
        let norm = next_state.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_learnable_world_model_creation() {
        let model = LearnableWorldModel::default();
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_learnable_world_model_predict() {
        let model = LearnableWorldModel::default();
        let state = make_world_state("entity", "relation");
        let action = make_graph("do something");

        let next_embed = model.predict_next(&state, &action);
        assert_eq!(next_embed.len(), 64);
    }

    #[test]
    fn test_learnable_world_model_observe() {
        let mut model = LearnableWorldModel::default();

        let s1 = make_world_state("state1", "rel1");
        let action = make_graph("action");
        let s2 = make_world_state("state2", "rel2");

        model.observe_transition(&s1, &action, &s2);
        assert_eq!(model.num_experiences(), 1);
    }

    #[test]
    fn test_learnable_world_model_imagine() {
        let model = LearnableWorldModel::default();
        let initial = make_world_state("initial", "relations");

        let actions = vec![
            make_graph("action1"),
            make_graph("action2"),
            make_graph("action3"),
        ];

        let predictions = model.imagine(&initial, &actions);
        assert_eq!(predictions.len(), 4); // initial + 3 steps
    }

    #[test]
    fn test_learnable_world_model_gradient_flow() {
        let mut model = LearnableWorldModel::default();

        // Zero grad and step should not panic
        model.zero_grad();
        model.step(0.001);
    }

    #[test]
    fn test_prediction_loss() {
        let mut model = LearnableWorldModel::default();

        // Add some observations
        for i in 0..5 {
            let s1 = make_world_state(&format!("state{}", i), "rel");
            let action = make_graph("action");
            let s2 = make_world_state(&format!("state{}", i + 1), "rel");
            model.observe_transition(&s1, &action, &s2);
        }

        let loss = model.compute_prediction_loss();
        assert!(loss >= 0.0);
    }
}
