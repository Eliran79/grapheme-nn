//! Learnable Agency Components (backend-036)
//!
//! This module provides learnable components for adaptive agency:
//! - Learnable goal encoders for fixed-size goal representations
//! - Learnable value networks for state/goal value estimation
//! - Learnable drive networks for adaptive motivation
//! - Learnable priority networks for goal prioritization
//!
//! **GRAPHEME Protocol**: Uses LeakyReLU activation (α=0.01) and Adam optimizer (lr=0.001).

use crate::{Drive, Goal, GoalId, GoalStatus, Graph};
use grapheme_memory::GraphFingerprint;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// LeakyReLU constant (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Default learning rate for agency updates (GRAPHEME Protocol)
pub const DEFAULT_LEARNING_RATE: f32 = 0.001;

/// Configuration for learnable agency
#[derive(Debug, Clone)]
pub struct LearnableAgencyConfig {
    /// Embedding dimension for goals and states
    pub embed_dim: usize,
    /// Hidden dimension for networks
    pub hidden_dim: usize,
    /// Learning rate (GRAPHEME Protocol: 0.001)
    pub learning_rate: f32,
    /// Number of drive types
    pub num_drives: usize,
    /// Maximum priority levels
    pub max_priority_levels: usize,
    /// Experience buffer size
    pub buffer_size: usize,
}

impl Default for LearnableAgencyConfig {
    fn default() -> Self {
        Self {
            embed_dim: 64,
            hidden_dim: 128,
            learning_rate: DEFAULT_LEARNING_RATE,
            num_drives: 5, // Curiosity, Efficiency, Safety, Helpfulness, Learning
            max_priority_levels: 10,
            buffer_size: 1000,
        }
    }
}

// ============================================================================
// Goal Encoder
// ============================================================================

/// Encodes goal graphs to fixed-size embeddings
#[derive(Debug)]
pub struct GoalEncoder {
    /// Goal content encoding weights
    pub w_content: Array2<f32>,
    /// Priority encoding weights
    pub w_priority: Array2<f32>,
    /// Combination layer
    pub w_combine: Array2<f32>,
    /// Bias
    pub b_combine: Array1<f32>,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Gradients
    pub grad_w_content: Option<Array2<f32>>,
    pub grad_w_priority: Option<Array2<f32>>,
    pub grad_w_combine: Option<Array2<f32>>,
    pub grad_b_combine: Option<Array1<f32>>,
}

impl GoalEncoder {
    const FEATURE_DIM: usize = 18;
    const PRIORITY_DIM: usize = 4; // priority, deadline_urgency, complexity, status

    /// Create new goal encoder with DynamicXavier initialization
    pub fn new(embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        let scale_content = (2.0 / (Self::FEATURE_DIM + embed_dim) as f32).sqrt();
        let scale_priority = (2.0 / (Self::PRIORITY_DIM + embed_dim) as f32).sqrt();
        let scale_combine = (2.0 / (embed_dim * 2 + embed_dim) as f32).sqrt();

        let w_content = Array2::from_shape_fn(
            (embed_dim, Self::FEATURE_DIM),
            |_| rng.gen_range(-scale_content..scale_content),
        );

        let w_priority = Array2::from_shape_fn(
            (embed_dim, Self::PRIORITY_DIM),
            |_| rng.gen_range(-scale_priority..scale_priority),
        );

        let w_combine = Array2::from_shape_fn(
            (embed_dim, embed_dim * 2),
            |_| rng.gen_range(-scale_combine..scale_combine),
        );

        let b_combine = Array1::zeros(embed_dim);

        Self {
            w_content,
            w_priority,
            w_combine,
            b_combine,
            embed_dim,
            grad_w_content: None,
            grad_w_priority: None,
            grad_w_combine: None,
            grad_b_combine: None,
        }
    }

    /// Extract graph features
    fn extract_graph_features(&self, graph: &Graph) -> Array1<f32> {
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

    /// Extract priority features from goal
    fn extract_priority_features(&self, goal: &Goal) -> Array1<f32> {
        let mut features = Array1::zeros(Self::PRIORITY_DIM);

        features[0] = goal.priority;
        features[1] = goal.deadline.map(|d| 1.0 / (d as f32 + 1.0)).unwrap_or(0.0);
        features[2] = (goal.complexity as f32 / 20.0).min(1.0);
        features[3] = match goal.status {
            GoalStatus::Pending => 0.0,
            GoalStatus::Active => 0.5,
            GoalStatus::Achieved => 1.0,
            GoalStatus::Failed(_) => -0.5,
            GoalStatus::Abandoned => -1.0,
        };

        features
    }

    /// LeakyReLU activation (GRAPHEME Protocol)
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Encode a goal to fixed-size embedding
    pub fn encode(&self, goal: &Goal) -> Array1<f32> {
        let content_features = self.extract_graph_features(&goal.description);
        let priority_features = self.extract_priority_features(goal);

        // Encode content and priority separately
        let content_embed = self.w_content.dot(&content_features);
        let priority_embed = self.w_priority.dot(&priority_features);

        // Concatenate and combine
        let mut combined = Array1::zeros(self.embed_dim * 2);
        for (i, &v) in content_embed.iter().enumerate() {
            combined[i] = v;
        }
        for (i, &v) in priority_embed.iter().enumerate() {
            combined[self.embed_dim + i] = v;
        }

        let output_pre = self.w_combine.dot(&combined) + &self.b_combine;
        let output = output_pre.mapv(|x| self.leaky_relu(x));

        // L2 normalize
        let norm = output.mapv(|x| x * x).sum().sqrt().max(1e-8);
        output / norm
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_content = None;
        self.grad_w_priority = None;
        self.grad_w_combine = None;
        self.grad_b_combine = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_content {
            self.w_content = &self.w_content - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_priority {
            self.w_priority = &self.w_priority - &(grad * lr);
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
        self.w_content.len() + self.w_priority.len() + self.w_combine.len() + self.b_combine.len()
    }
}

impl Default for GoalEncoder {
    fn default() -> Self {
        Self::new(64)
    }
}

// ============================================================================
// Value Network
// ============================================================================

/// Learnable value function for estimating goal/state value
#[derive(Debug)]
pub struct ValueNetwork {
    /// Input layer: [hidden_dim, embed_dim]
    pub w_input: Array2<f32>,
    /// Hidden layer: [hidden_dim, hidden_dim]
    pub w_hidden: Array2<f32>,
    /// Output layer: [1, hidden_dim]
    pub w_output: Array2<f32>,
    /// Biases
    pub b_input: Array1<f32>,
    pub b_hidden: Array1<f32>,
    pub b_output: Array1<f32>,
    /// Gradients
    pub grad_w_input: Option<Array2<f32>>,
    pub grad_w_hidden: Option<Array2<f32>>,
    pub grad_w_output: Option<Array2<f32>>,
    pub grad_b_input: Option<Array1<f32>>,
    pub grad_b_hidden: Option<Array1<f32>>,
    pub grad_b_output: Option<Array1<f32>>,
}

impl ValueNetwork {
    /// Create new value network with DynamicXavier initialization
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        let scale_input = (2.0 / (embed_dim + hidden_dim) as f32).sqrt();
        let scale_hidden = (2.0 / (hidden_dim * 2) as f32).sqrt();
        let scale_output = (2.0 / (hidden_dim + 1) as f32).sqrt();

        let w_input = Array2::from_shape_fn(
            (hidden_dim, embed_dim),
            |_| rng.gen_range(-scale_input..scale_input),
        );

        let w_hidden = Array2::from_shape_fn(
            (hidden_dim, hidden_dim),
            |_| rng.gen_range(-scale_hidden..scale_hidden),
        );

        let w_output = Array2::from_shape_fn(
            (1, hidden_dim),
            |_| rng.gen_range(-scale_output..scale_output),
        );

        Self {
            w_input,
            w_hidden,
            w_output,
            b_input: Array1::zeros(hidden_dim),
            b_hidden: Array1::zeros(hidden_dim),
            b_output: Array1::zeros(1),
            grad_w_input: None,
            grad_w_hidden: None,
            grad_w_output: None,
            grad_b_input: None,
            grad_b_hidden: None,
            grad_b_output: None,
        }
    }

    /// LeakyReLU activation (GRAPHEME Protocol)
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Estimate value of an embedding (goal or state)
    pub fn estimate(&self, embedding: &Array1<f32>) -> f32 {
        // Input layer
        let h1_pre = self.w_input.dot(embedding) + &self.b_input;
        let h1 = h1_pre.mapv(|x| self.leaky_relu(x));

        // Hidden layer
        let h2_pre = self.w_hidden.dot(&h1) + &self.b_hidden;
        let h2 = h2_pre.mapv(|x| self.leaky_relu(x));

        // Output layer (sigmoid for [0, 1] range)
        let out = self.w_output.dot(&h2) + &self.b_output;
        1.0 / (1.0 + (-out[0]).exp())
    }

    /// Compute TD error for value learning
    pub fn td_error(&self, current_value: f32, reward: f32, next_value: f32, gamma: f32) -> f32 {
        reward + gamma * next_value - current_value
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_input = None;
        self.grad_w_hidden = None;
        self.grad_w_output = None;
        self.grad_b_input = None;
        self.grad_b_hidden = None;
        self.grad_b_output = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_input {
            self.w_input = &self.w_input - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_hidden {
            self.w_hidden = &self.w_hidden - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_output {
            self.w_output = &self.w_output - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_input {
            self.b_input = &self.b_input - &(grad * lr);
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
        self.w_input.len() + self.w_hidden.len() + self.w_output.len()
            + self.b_input.len() + self.b_hidden.len() + self.b_output.len()
    }
}

impl Default for ValueNetwork {
    fn default() -> Self {
        Self::new(64, 128)
    }
}

// ============================================================================
// Drive Network
// ============================================================================

/// Learnable drive network for adaptive motivation
#[derive(Debug)]
pub struct DriveNetwork {
    /// Context to drive weights: [num_drives, embed_dim]
    pub w_context: Array2<f32>,
    /// Bias
    pub b_context: Array1<f32>,
    /// Drive names for interpretation
    pub drive_names: Vec<String>,
    /// Gradients
    pub grad_w_context: Option<Array2<f32>>,
    pub grad_b_context: Option<Array1<f32>>,
}

impl DriveNetwork {
    /// Create new drive network
    pub fn new(embed_dim: usize, num_drives: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (embed_dim + num_drives) as f32).sqrt();

        let w_context = Array2::from_shape_fn(
            (num_drives, embed_dim),
            |_| rng.gen_range(-scale..scale),
        );

        let b_context = Array1::zeros(num_drives);

        let drive_names = vec![
            "Curiosity".to_string(),
            "Efficiency".to_string(),
            "Safety".to_string(),
            "Helpfulness".to_string(),
            "Learning".to_string(),
        ];

        Self {
            w_context,
            b_context,
            drive_names,
            grad_w_context: None,
            grad_b_context: None,
        }
    }

    /// Compute drive strengths given context embedding
    pub fn compute_drives(&self, context: &Array1<f32>) -> Vec<Drive> {
        let raw = self.w_context.dot(context) + &self.b_context;

        // Softmax to get normalized strengths
        let max_val = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_raw = raw.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_raw.sum().max(1e-8);
        let strengths = exp_raw / sum_exp;

        vec![
            Drive::Curiosity { strength: strengths[0] },
            Drive::Efficiency { strength: strengths[1] },
            Drive::Safety { strength: strengths[2] },
            Drive::Helpfulness { strength: strengths[3] },
            Drive::Learning { strength: strengths[4] },
        ]
    }

    /// Get drive strengths as array
    pub fn compute_strengths(&self, context: &Array1<f32>) -> Array1<f32> {
        let raw = self.w_context.dot(context) + &self.b_context;

        // Softmax
        let max_val = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_raw = raw.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_raw.sum().max(1e-8);
        exp_raw / sum_exp
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_context = None;
        self.grad_b_context = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_context {
            self.w_context = &self.w_context - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_context {
            self.b_context = &self.b_context - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_context.len() + self.b_context.len()
    }
}

impl Default for DriveNetwork {
    fn default() -> Self {
        Self::new(64, 5)
    }
}

// ============================================================================
// Priority Network
// ============================================================================

/// Learnable priority network for goal prioritization
#[derive(Debug)]
pub struct PriorityNetwork {
    /// Input layer: [hidden_dim, embed_dim * 2] (goal + context)
    pub w_input: Array2<f32>,
    /// Output layer: [1, hidden_dim]
    pub w_output: Array2<f32>,
    /// Biases
    pub b_input: Array1<f32>,
    pub b_output: Array1<f32>,
    /// Gradients
    pub grad_w_input: Option<Array2<f32>>,
    pub grad_w_output: Option<Array2<f32>>,
    pub grad_b_input: Option<Array1<f32>>,
    pub grad_b_output: Option<Array1<f32>>,
}

impl PriorityNetwork {
    /// Create new priority network
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        let input_dim = embed_dim * 2;
        let scale_input = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale_output = (2.0 / (hidden_dim + 1) as f32).sqrt();

        let w_input = Array2::from_shape_fn(
            (hidden_dim, input_dim),
            |_| rng.gen_range(-scale_input..scale_input),
        );

        let w_output = Array2::from_shape_fn(
            (1, hidden_dim),
            |_| rng.gen_range(-scale_output..scale_output),
        );

        Self {
            w_input,
            w_output,
            b_input: Array1::zeros(hidden_dim),
            b_output: Array1::zeros(1),
            grad_w_input: None,
            grad_w_output: None,
            grad_b_input: None,
            grad_b_output: None,
        }
    }

    /// LeakyReLU activation (GRAPHEME Protocol)
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Compute priority score for a goal given context
    pub fn compute_priority(&self, goal_embed: &Array1<f32>, context_embed: &Array1<f32>) -> f32 {
        // Concatenate goal and context
        let mut input = Array1::zeros(goal_embed.len() + context_embed.len());
        for (i, &v) in goal_embed.iter().enumerate() {
            input[i] = v;
        }
        for (i, &v) in context_embed.iter().enumerate() {
            input[goal_embed.len() + i] = v;
        }

        // Forward pass
        let h = self.w_input.dot(&input) + &self.b_input;
        let h = h.mapv(|x| self.leaky_relu(x));

        let out = self.w_output.dot(&h) + &self.b_output;

        // Sigmoid for [0, 1] priority
        1.0 / (1.0 + (-out[0]).exp())
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_input = None;
        self.grad_w_output = None;
        self.grad_b_input = None;
        self.grad_b_output = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_input {
            self.w_input = &self.w_input - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_output {
            self.w_output = &self.w_output - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_input {
            self.b_input = &self.b_input - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_output {
            self.b_output = &self.b_output - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_input.len() + self.w_output.len() + self.b_input.len() + self.b_output.len()
    }
}

impl Default for PriorityNetwork {
    fn default() -> Self {
        Self::new(64, 64)
    }
}

// ============================================================================
// Experience Types
// ============================================================================

/// Experience tuple for goal-based learning
#[derive(Debug, Clone)]
pub struct GoalExperience {
    /// Goal embedding at time t
    pub goal_embed: Array1<f32>,
    /// Context embedding at time t
    pub context_embed: Array1<f32>,
    /// Action taken
    pub action_embed: Array1<f32>,
    /// Reward received
    pub reward: f32,
    /// Goal achieved?
    pub achieved: bool,
    /// Next goal embedding (if any)
    pub next_goal_embed: Option<Array1<f32>>,
}

// ============================================================================
// Learnable Agency Model
// ============================================================================

/// Complete learnable agency model with adaptive goals and values
#[derive(Debug)]
pub struct LearnableAgency {
    /// Goal encoder
    pub goal_encoder: GoalEncoder,
    /// Value network
    pub value_network: ValueNetwork,
    /// Drive network
    pub drive_network: DriveNetwork,
    /// Priority network
    pub priority_network: PriorityNetwork,
    /// Configuration
    pub config: LearnableAgencyConfig,
    /// Experience buffer
    experience_buffer: Vec<GoalExperience>,
    /// Goal embedding cache
    goal_cache: HashMap<GoalId, Array1<f32>>,
    /// Current context embedding
    current_context: Option<Array1<f32>>,
}

impl LearnableAgency {
    /// Create new learnable agency
    pub fn new(config: LearnableAgencyConfig) -> Self {
        let goal_encoder = GoalEncoder::new(config.embed_dim);
        let value_network = ValueNetwork::new(config.embed_dim, config.hidden_dim);
        let drive_network = DriveNetwork::new(config.embed_dim, config.num_drives);
        let priority_network = PriorityNetwork::new(config.embed_dim, config.hidden_dim);

        Self {
            goal_encoder,
            value_network,
            drive_network,
            priority_network,
            config,
            experience_buffer: Vec::new(),
            goal_cache: HashMap::new(),
            current_context: None,
        }
    }

    /// Encode a goal (with caching)
    pub fn encode_goal(&mut self, goal: &Goal) -> Array1<f32> {
        if let Some(cached) = self.goal_cache.get(&goal.id) {
            return cached.clone();
        }

        let embed = self.goal_encoder.encode(goal);
        self.goal_cache.insert(goal.id, embed.clone());
        embed
    }

    /// Set current context from a graph
    pub fn set_context(&mut self, context: &Graph) {
        let fp = GraphFingerprint::from_graph(context);
        let mut features = Array1::zeros(self.config.embed_dim);

        // Simple context encoding
        features[0] = fp.node_count as f32 / 100.0;
        features[1] = fp.edge_count as f32 / 100.0;
        for i in 0..8.min(self.config.embed_dim - 2) {
            features[2 + i] = fp.node_types[i] as f32 / 10.0;
        }

        // Normalize
        let norm = features.mapv(|x| x * x).sum().sqrt().max(1e-8);
        self.current_context = Some(features / norm);
    }

    /// Get current context (or default)
    fn get_context(&self) -> Array1<f32> {
        self.current_context.clone().unwrap_or_else(|| {
            Array1::from_elem(self.config.embed_dim, 0.1)
        })
    }

    /// Estimate value of a goal
    pub fn estimate_goal_value(&mut self, goal: &Goal) -> f32 {
        let embed = self.encode_goal(goal);
        self.value_network.estimate(&embed)
    }

    /// Get adaptive drives for current context
    pub fn get_adaptive_drives(&self) -> Vec<Drive> {
        let context = self.get_context();
        self.drive_network.compute_drives(&context)
    }

    /// Compute priority for a goal in current context
    pub fn compute_goal_priority(&mut self, goal: &Goal) -> f32 {
        let goal_embed = self.encode_goal(goal);
        let context = self.get_context();
        self.priority_network.compute_priority(&goal_embed, &context)
    }

    /// Select best goal from candidates based on learned priority
    pub fn select_best_goal<'a>(&mut self, goals: &'a [Goal]) -> Option<&'a Goal> {
        if goals.is_empty() {
            return None;
        }

        let context = self.get_context();
        let mut best_goal = None;
        let mut best_score = f32::NEG_INFINITY;

        for goal in goals {
            if goal.status.is_terminal() {
                continue;
            }

            let goal_embed = self.encode_goal(goal);
            let priority = self.priority_network.compute_priority(&goal_embed, &context);
            let value = self.value_network.estimate(&goal_embed);

            // Combined score: priority * value
            let score = priority * value;

            if score > best_score {
                best_score = score;
                best_goal = Some(goal);
            }
        }

        best_goal
    }

    /// Record experience for learning
    pub fn record_experience(&mut self, experience: GoalExperience) {
        self.experience_buffer.push(experience);

        // Keep buffer bounded
        if self.experience_buffer.len() > self.config.buffer_size {
            self.experience_buffer.remove(0);
        }
    }

    /// Record goal outcome for learning
    pub fn record_goal_outcome(
        &mut self,
        goal: &Goal,
        action: &Graph,
        reward: f32,
        achieved: bool,
        next_goal: Option<&Goal>,
    ) {
        let goal_embed = self.encode_goal(goal);
        let context_embed = self.get_context();

        // Simple action encoding
        let fp = GraphFingerprint::from_graph(action);
        let mut action_embed = Array1::zeros(self.config.embed_dim);
        action_embed[0] = fp.node_count as f32 / 100.0;
        action_embed[1] = fp.edge_count as f32 / 100.0;

        let next_goal_embed = next_goal.map(|g| self.encode_goal(g));

        self.record_experience(GoalExperience {
            goal_embed,
            context_embed,
            action_embed,
            reward,
            achieved,
            next_goal_embed,
        });
    }

    /// Compute value learning loss (TD error)
    pub fn compute_value_loss(&self, gamma: f32) -> f32 {
        if self.experience_buffer.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for exp in &self.experience_buffer {
            let current_value = self.value_network.estimate(&exp.goal_embed);
            let next_value = exp.next_goal_embed.as_ref()
                .map(|e| self.value_network.estimate(e))
                .unwrap_or(0.0);

            let td_error = self.value_network.td_error(current_value, exp.reward, next_value, gamma);
            total_loss += td_error * td_error;
        }

        total_loss / self.experience_buffer.len() as f32
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        self.goal_encoder.zero_grad();
        self.value_network.zero_grad();
        self.drive_network.zero_grad();
        self.priority_network.zero_grad();
    }

    /// Update all parameters
    pub fn step(&mut self, lr: f32) {
        self.goal_encoder.step(lr);
        self.value_network.step(lr);
        self.drive_network.step(lr);
        self.priority_network.step(lr);
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        self.goal_encoder.num_parameters()
            + self.value_network.num_parameters()
            + self.drive_network.num_parameters()
            + self.priority_network.num_parameters()
    }

    /// Clear caches
    pub fn clear_cache(&mut self) {
        self.goal_cache.clear();
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
        self.goal_encoder.grad_w_content.is_some()
            || self.value_network.grad_w_input.is_some()
            || self.drive_network.grad_w_context.is_some()
            || self.priority_network.grad_w_input.is_some()
    }
}

impl Default for LearnableAgency {
    fn default() -> Self {
        Self::new(LearnableAgencyConfig::default())
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

    fn make_goal(id: GoalId, name: &str, priority: f32) -> Goal {
        Goal::new(id, name, make_graph(name)).with_priority(priority)
    }

    #[test]
    fn test_learnable_agency_config_default() {
        let config = LearnableAgencyConfig::default();
        assert_eq!(config.embed_dim, 64);
        assert!((config.learning_rate - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_goal_encoder_creation() {
        let encoder = GoalEncoder::default();
        assert!(encoder.num_parameters() > 0);
    }

    #[test]
    fn test_goal_encoder_encode() {
        let encoder = GoalEncoder::new(64);
        let goal = make_goal(1, "test goal", 0.5);

        let embedding = encoder.encode(&goal);
        assert_eq!(embedding.len(), 64);

        // Check L2 normalized
        let norm = embedding.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_value_network_creation() {
        let network = ValueNetwork::default();
        assert!(network.num_parameters() > 0);
    }

    #[test]
    fn test_value_network_estimate() {
        let network = ValueNetwork::new(64, 128);
        let embedding = Array1::from_vec(vec![0.1; 64]);

        let value = network.estimate(&embedding);
        assert!(value >= 0.0 && value <= 1.0);
    }

    #[test]
    fn test_drive_network_creation() {
        let network = DriveNetwork::default();
        assert!(network.num_parameters() > 0);
    }

    #[test]
    fn test_drive_network_compute() {
        let network = DriveNetwork::new(64, 5);
        let context = Array1::from_vec(vec![0.1; 64]);

        let drives = network.compute_drives(&context);
        assert_eq!(drives.len(), 5);

        // Check drives sum to ~1.0 (softmax)
        let total: f32 = drives.iter().map(|d| d.strength()).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_priority_network_creation() {
        let network = PriorityNetwork::default();
        assert!(network.num_parameters() > 0);
    }

    #[test]
    fn test_priority_network_compute() {
        let network = PriorityNetwork::new(64, 64);
        let goal_embed = Array1::from_vec(vec![0.1; 64]);
        let context_embed = Array1::from_vec(vec![0.1; 64]);

        let priority = network.compute_priority(&goal_embed, &context_embed);
        assert!(priority >= 0.0 && priority <= 1.0);
    }

    #[test]
    fn test_learnable_agency_creation() {
        let agency = LearnableAgency::default();
        assert!(agency.num_parameters() > 0);
    }

    #[test]
    fn test_learnable_agency_encode_goal() {
        let mut agency = LearnableAgency::default();
        let goal = make_goal(1, "test", 0.5);

        let embed = agency.encode_goal(&goal);
        assert_eq!(embed.len(), 64);
    }

    #[test]
    fn test_learnable_agency_estimate_value() {
        let mut agency = LearnableAgency::default();
        let goal = make_goal(1, "test", 0.5);

        let value = agency.estimate_goal_value(&goal);
        assert!(value >= 0.0 && value <= 1.0);
    }

    #[test]
    fn test_learnable_agency_adaptive_drives() {
        let agency = LearnableAgency::default();
        let drives = agency.get_adaptive_drives();

        assert_eq!(drives.len(), 5);
    }

    #[test]
    fn test_learnable_agency_goal_priority() {
        let mut agency = LearnableAgency::default();
        let goal = make_goal(1, "test", 0.5);

        let priority = agency.compute_goal_priority(&goal);
        assert!(priority >= 0.0 && priority <= 1.0);
    }

    #[test]
    fn test_learnable_agency_select_best_goal() {
        let mut agency = LearnableAgency::default();

        let goals = vec![
            make_goal(1, "low priority", 0.2),
            make_goal(2, "high priority", 0.9),
            make_goal(3, "medium priority", 0.5),
        ];

        let best = agency.select_best_goal(&goals);
        assert!(best.is_some());
    }

    #[test]
    fn test_learnable_agency_record_outcome() {
        let mut agency = LearnableAgency::default();
        let goal = make_goal(1, "test", 0.5);
        let action = make_graph("action");

        agency.record_goal_outcome(&goal, &action, 1.0, true, None);
        assert_eq!(agency.num_experiences(), 1);
    }

    #[test]
    fn test_learnable_agency_value_loss() {
        let mut agency = LearnableAgency::default();

        // Add some experiences
        for i in 0..5 {
            let goal = make_goal(i, &format!("goal{}", i), 0.5);
            let action = make_graph("action");
            agency.record_goal_outcome(&goal, &action, 0.5, false, None);
        }

        let loss = agency.compute_value_loss(0.99);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_learnable_agency_gradient_flow() {
        let mut agency = LearnableAgency::default();

        // Zero grad and step should not panic
        agency.zero_grad();
        agency.step(0.001);
    }

    #[test]
    fn test_learnable_agency_context() {
        let mut agency = LearnableAgency::default();
        let context = make_graph("current situation");

        agency.set_context(&context);

        // Should affect drives
        let drives = agency.get_adaptive_drives();
        assert!(!drives.is_empty());
    }
}
