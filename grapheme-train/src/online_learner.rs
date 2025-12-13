//! Online Learning Module
//!
//! Provides continuous online learning for AGI using grapheme-memory integration.
//! Supports streaming data, experience replay, and consolidation.

use grapheme_core::{BackwardPass, DagNN, Embedding, InitStrategy, Node, NodeId, UnifiedCheckpoint};
use grapheme_memory::{
    ContinualLearning, Episode, EpisodicMemory, RetentionPolicy, SimpleContinualLearning,
    SimpleEpisodicMemory, Timestamp,
};
use ndarray::Array1;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};

/// Strategy for sampling from experience replay buffer
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReplayStrategy {
    /// Uniform random sampling
    Uniform,
    /// Prioritized by loss (higher loss = higher priority)
    PrioritizedLoss,
    /// Prioritized by recency (more recent = higher priority)
    PrioritizedRecency,
    /// Mixed strategy: 50% loss-based, 50% uniform
    Mixed,
    /// Domain-balanced sampling
    DomainBalanced,
}

impl Default for ReplayStrategy {
    fn default() -> Self {
        Self::Mixed
    }
}

/// Configuration for online learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearnerConfig {
    /// Learning rate for gradient updates
    pub learning_rate: f32,
    /// Batch size for mini-batch training
    pub batch_size: usize,
    /// Replay buffer capacity (max episodes to store)
    pub replay_capacity: usize,
    /// Ratio of replay samples in each batch (0.0 to 1.0)
    pub replay_ratio: f32,
    /// Consolidation interval (number of examples between consolidations)
    pub consolidation_interval: usize,
    /// Whether to use EWC regularization
    pub use_ewc: bool,
    /// EWC lambda (importance weight)
    pub ewc_lambda: f32,
    /// Replay sampling strategy
    pub replay_strategy: ReplayStrategy,
    /// Priority exponent for prioritized replay (higher = more focus on high-loss)
    pub priority_alpha: f32,
}

impl Default for OnlineLearnerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            replay_capacity: 10_000,
            replay_ratio: 0.5,
            consolidation_interval: 1000,
            use_ewc: false,
            ewc_lambda: 0.4,
            replay_strategy: ReplayStrategy::Mixed,
            priority_alpha: 0.6,
        }
    }
}

impl OnlineLearnerConfig {
    /// Create config for fast learning (less replay, more new data)
    pub fn fast() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 16,
            replay_capacity: 1_000,
            replay_ratio: 0.2,
            consolidation_interval: 500,
            use_ewc: false,
            ewc_lambda: 0.0,
            replay_strategy: ReplayStrategy::Uniform,
            priority_alpha: 0.0,
        }
    }

    /// Create config for stable learning (more replay, EWC enabled)
    pub fn stable() -> Self {
        Self {
            learning_rate: 0.0001,
            batch_size: 64,
            replay_capacity: 50_000,
            replay_ratio: 0.7,
            consolidation_interval: 5000,
            use_ewc: true,
            ewc_lambda: 0.5,
            replay_strategy: ReplayStrategy::PrioritizedLoss,
            priority_alpha: 0.8,
        }
    }

    /// Create config with prioritized replay (focus on hard examples)
    pub fn prioritized() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            replay_capacity: 20_000,
            replay_ratio: 0.6,
            consolidation_interval: 2000,
            use_ewc: false,
            ewc_lambda: 0.0,
            replay_strategy: ReplayStrategy::PrioritizedLoss,
            priority_alpha: 1.0,
        }
    }

    /// Create config for domain-balanced learning
    pub fn balanced() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            replay_capacity: 15_000,
            replay_ratio: 0.5,
            consolidation_interval: 1500,
            use_ewc: false,
            ewc_lambda: 0.0,
            replay_strategy: ReplayStrategy::DomainBalanced,
            priority_alpha: 0.5,
        }
    }
}

/// A training example for online learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineExample {
    /// Unique identifier
    pub id: String,
    /// Input representation (flattened activations)
    pub input: Vec<f32>,
    /// Target output (flattened activations)
    pub target: Vec<f32>,
    /// Domain tag (math, text, vision, timeseries)
    pub domain: String,
    /// Difficulty level (1-7 for curriculum)
    pub level: u8,
    /// Loss when this example was last seen (for prioritized replay)
    pub last_loss: f32,
}

impl OnlineExample {
    /// Create a new online example
    pub fn new(id: impl Into<String>, input: Vec<f32>, target: Vec<f32>, domain: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            input,
            target,
            domain: domain.into(),
            level: 1,
            last_loss: f32::MAX,
        }
    }

    /// Set curriculum level
    pub fn with_level(mut self, level: u8) -> Self {
        self.level = level;
        self
    }
}

/// Statistics for online learning
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OnlineLearnerStats {
    /// Total examples seen
    pub examples_seen: usize,
    /// Total batches trained
    pub batches_trained: usize,
    /// Total consolidations performed
    pub consolidations: usize,
    /// Running average loss
    pub avg_loss: f32,
    /// Best loss seen
    pub best_loss: f32,
    /// Examples per domain
    pub domain_counts: HashMap<String, usize>,
    /// Current curriculum level
    pub current_level: u8,
}

/// Online learner trait - core interface for continuous learning
pub trait OnlineLearner: Send + Sync {
    /// Learn from a single example (may buffer internally)
    fn learn_one(&mut self, example: OnlineExample) -> f32;

    /// Learn from a batch of examples
    fn learn_batch(&mut self, batch: &[OnlineExample]) -> f32;

    /// Trigger consolidation (replay + memory integration)
    fn consolidate(&mut self);

    /// Get current model reference
    fn model(&self) -> &DagNN;

    /// Get mutable model reference
    fn model_mut(&mut self) -> &mut DagNN;

    /// Get learning statistics
    fn stats(&self) -> &OnlineLearnerStats;

    /// Check if consolidation is needed
    fn should_consolidate(&self) -> bool;

    /// Save learner state
    fn save_state<W: Write>(&self, writer: W) -> std::io::Result<()>;

    /// Load learner state
    fn load_state<R: Read>(&mut self, reader: R) -> std::io::Result<()>;
}

/// Replay metadata for prioritized sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayMetadata {
    /// Episode ID in episodic memory
    pub episode_id: u64,
    /// Last recorded loss for this example
    pub loss: f32,
    /// Domain tag
    pub domain: String,
    /// Timestamp when stored
    pub timestamp: Timestamp,
    /// Number of times replayed
    pub replay_count: usize,
}

// ========== EWC (Elastic Weight Consolidation) for forgetting prevention ==========

/// Edge key type for EWC (serializable, hashable representation of edge)
pub type EdgeKey = (usize, usize);

/// Elastic Weight Consolidation state (backend-205)
/// Stores Fisher information diagonal and optimal parameters for edge weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWCState {
    /// Fisher information diagonal (importance weight per edge parameter)
    /// Maps (src_idx, tgt_idx) -> Fisher value
    pub fisher_diag: HashMap<EdgeKey, f32>,
    /// Optimal edge weights from previous tasks
    /// Maps (src_idx, tgt_idx) -> weight at consolidation
    pub optimal_params: HashMap<EdgeKey, f32>,
    /// Number of consolidations (tasks) recorded
    pub task_count: usize,
    /// Total samples used to estimate Fisher information
    pub samples_used: usize,
    /// Whether EWC is active (has been computed at least once)
    pub is_active: bool,
}

impl Default for EWCState {
    fn default() -> Self {
        Self {
            fisher_diag: HashMap::new(),
            optimal_params: HashMap::new(),
            task_count: 0,
            samples_used: 0,
            is_active: false,
        }
    }
}

impl EWCState {
    /// Create new EWC state
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute EWC penalty loss for current edge weights
    /// L_ewc = (lambda/2) * sum_i(F_i * (theta_i - theta_i*)^2)
    pub fn compute_penalty(&self, current_weights: &HashMap<EdgeKey, f32>, lambda: f32) -> f32 {
        if !self.is_active {
            return 0.0;
        }

        let mut penalty = 0.0;
        for (edge_key, &optimal) in &self.optimal_params {
            if let Some(&current) = current_weights.get(edge_key) {
                if let Some(&fisher) = self.fisher_diag.get(edge_key) {
                    let diff = current - optimal;
                    penalty += fisher * diff * diff;
                }
            }
        }
        penalty * lambda / 2.0
    }

    /// Compute EWC gradient penalty for a specific edge
    /// grad_ewc = lambda * F_i * (theta_i - theta_i*)
    pub fn compute_gradient_penalty(&self, edge_key: EdgeKey, current_weight: f32, lambda: f32) -> f32 {
        if !self.is_active {
            return 0.0;
        }

        let optimal = self.optimal_params.get(&edge_key).copied().unwrap_or(current_weight);
        let fisher = self.fisher_diag.get(&edge_key).copied().unwrap_or(0.0);
        lambda * fisher * (current_weight - optimal)
    }

    /// Update Fisher information with new gradient samples
    /// Uses online averaging: F_new = F_old + (grad^2 - F_old) / n
    pub fn update_fisher(&mut self, edge_grads: &HashMap<EdgeKey, f32>) {
        self.samples_used += 1;
        let n = self.samples_used as f32;

        for (&edge_key, &grad) in edge_grads {
            let grad_sq = grad * grad;
            let fisher = self.fisher_diag.entry(edge_key).or_insert(0.0);
            // Online mean update
            *fisher += (grad_sq - *fisher) / n;
        }
    }

    /// Consolidate current parameters as optimal for this task
    pub fn consolidate(&mut self, current_weights: &HashMap<EdgeKey, f32>) {
        self.optimal_params = current_weights.clone();
        self.task_count += 1;
        // Only set is_active if we have parameters to protect
        if !current_weights.is_empty() {
            self.is_active = true;
        }
    }

    /// Reset Fisher information for new task (keep optimal params)
    pub fn reset_fisher(&mut self) {
        self.fisher_diag.clear();
        self.samples_used = 0;
    }

    /// Get Fisher information statistics
    pub fn fisher_stats(&self) -> EWCStats {
        if self.fisher_diag.is_empty() {
            return EWCStats::default();
        }

        let values: Vec<f32> = self.fisher_diag.values().copied().collect();
        let sum: f32 = values.iter().sum();
        let mean = sum / values.len() as f32;
        let max = values.iter().cloned().fold(0.0f32, f32::max);
        let min = values.iter().cloned().fold(f32::MAX, f32::min);

        EWCStats {
            param_count: self.fisher_diag.len(),
            task_count: self.task_count,
            samples_used: self.samples_used,
            fisher_mean: mean,
            fisher_max: max,
            fisher_min: min,
            is_active: self.is_active,
        }
    }
}

/// Statistics about EWC state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EWCStats {
    /// Number of parameters tracked
    pub param_count: usize,
    /// Number of tasks consolidated
    pub task_count: usize,
    /// Samples used for Fisher estimation
    pub samples_used: usize,
    /// Mean Fisher information value
    pub fisher_mean: f32,
    /// Maximum Fisher value
    pub fisher_max: f32,
    /// Minimum Fisher value
    pub fisher_min: f32,
    /// Whether EWC is active
    pub is_active: bool,
}

/// Memory-integrated online learner using grapheme-memory
pub struct MemoryOnlineLearner {
    /// The DagNN model being trained
    model: DagNN,
    /// Embedding layer for gradient tracking
    embedding: Embedding,
    /// Episodic memory for experience replay
    episodic_memory: SimpleEpisodicMemory,
    /// Continual learning system
    continual_learning: SimpleContinualLearning,
    /// Configuration
    config: OnlineLearnerConfig,
    /// Statistics
    stats: OnlineLearnerStats,
    /// Recent examples buffer (before consolidation)
    recent_buffer: Vec<OnlineExample>,
    /// Examples since last consolidation
    examples_since_consolidation: usize,
    /// Episode ID counter
    next_episode_id: u64,
    /// Replay metadata for prioritized sampling
    replay_metadata: Vec<ReplayMetadata>,
    /// Random number generator for sampling
    rng_seed: u64,
    /// EWC state for forgetting prevention (backend-205)
    ewc_state: EWCState,
}

impl MemoryOnlineLearner {
    /// Create a new memory-integrated online learner
    pub fn new(model: DagNN, config: OnlineLearnerConfig) -> Self {
        #[allow(deprecated)]
        let embedding = Embedding::new(256, 64, InitStrategy::DynamicXavier);
        let episodic_memory = SimpleEpisodicMemory::new(Some(config.replay_capacity));
        let continual_learning = SimpleContinualLearning::new(config.replay_capacity);
        let rng_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);

        Self {
            model,
            embedding,
            episodic_memory,
            continual_learning,
            config,
            stats: OnlineLearnerStats {
                best_loss: f32::MAX,
                current_level: 1,
                ..Default::default()
            },
            recent_buffer: Vec::new(),
            examples_since_consolidation: 0,
            next_episode_id: 0,
            replay_metadata: Vec::new(),
            rng_seed,
            ewc_state: EWCState::new(),
        }
    }

    /// Create with default DagNN
    pub fn with_default_model(config: OnlineLearnerConfig) -> Self {
        let model = DagNN::from_text("init").unwrap_or_else(|_| DagNN::new());
        Self::new(model, config)
    }

    /// Get current timestamp in milliseconds
    fn current_timestamp() -> Timestamp {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Convert OnlineExample to Episode for memory storage
    fn example_to_episode(&mut self, example: &OnlineExample) -> Episode {
        let id = self.next_episode_id;
        self.next_episode_id += 1;

        // Create context graph from input
        let mut context = DagNN::new();
        for (i, &val) in example.input.iter().enumerate() {
            let mut node = Node::input(' ', i);  // Use space as placeholder char
            node.activation = val;
            context.graph.add_node(node);
        }

        // Create content graph from target
        let mut content = DagNN::new();
        for &val in &example.target {
            let mut node = Node::output();
            node.activation = val;
            content.graph.add_node(node);
        }

        Episode::new(id, Self::current_timestamp(), context, content)
            .with_importance(1.0 / (1.0 + example.last_loss))
            .with_tags(vec![example.domain.clone(), format!("level_{}", example.level)])
    }

    /// Train on a single example and return loss
    fn train_single(&mut self, example: &OnlineExample) -> f32 {
        // Zero gradients
        self.model.zero_grad();
        self.embedding.zero_grad();

        // Set input activations
        let input_nodes = self.model.input_nodes();
        let mut input_map: HashMap<NodeId, f32> = HashMap::new();
        for (i, &node) in input_nodes.iter().enumerate() {
            if i < example.input.len() {
                input_map.insert(node, example.input[i]);
            }
        }

        // Forward pass
        let _ = self.model.forward_with_inputs(&input_map);

        // Get output activations
        let output = self.model.get_output_activations(example.target.len());

        // Compute MSE loss
        let mut loss = 0.0;
        for i in 0..output.len().min(example.target.len()) {
            let diff = output[i] - example.target[i];
            loss += diff * diff;
        }
        loss /= output.len().max(1) as f32;

        // Compute output gradients
        let output_nodes = self.model.output_nodes();
        let mut output_grad: HashMap<NodeId, Array1<f32>> = HashMap::new();
        for (i, &node) in output_nodes.iter().enumerate() {
            if i < output.len() && i < example.target.len() {
                let grad = 2.0 * (output[i] - example.target[i]) / output.len() as f32;
                output_grad.insert(node, Array1::from_vec(vec![grad]));
            }
        }

        // Backward pass
        self.model.backward_accumulate(&output_grad, &mut self.embedding);

        // EWC: Add gradient penalty to edge weights to prevent forgetting (backend-205)
        if self.config.use_ewc && self.ewc_state.is_active {
            let lambda = self.config.ewc_lambda;
            // Collect edge info first to avoid borrow conflict
            let edge_info: Vec<_> = self.model.graph.edge_references()
                .map(|edge_ref| {
                    let src = edge_ref.source();
                    let tgt = edge_ref.target();
                    let edge_key = (src.index(), tgt.index());
                    let current_weight = edge_ref.weight().weight;
                    (src, tgt, edge_key, current_weight)
                })
                .collect();

            for (src, tgt, edge_key, current_weight) in edge_info {
                let ewc_grad = self.ewc_state.compute_gradient_penalty(edge_key, current_weight, lambda);
                // Accumulate EWC gradient to existing edge gradient
                self.model.accumulate_edge_grad(src, tgt, ewc_grad);
            }
        }

        // Apply gradients
        self.model.step(self.config.learning_rate);

        loss
    }

    /// Simple random number generator (xorshift64)
    fn next_random(&mut self) -> u64 {
        self.rng_seed ^= self.rng_seed << 13;
        self.rng_seed ^= self.rng_seed >> 7;
        self.rng_seed ^= self.rng_seed << 17;
        self.rng_seed
    }

    /// Get random float in [0, 1)
    fn random_f32(&mut self) -> f32 {
        (self.next_random() as f64 / u64::MAX as f64) as f32
    }

    /// Sample indices based on priority weights (higher weight = higher probability)
    fn sample_by_priority(&mut self, weights: &[f32], count: usize) -> Vec<usize> {
        if weights.is_empty() {
            return vec![];
        }

        // Calculate cumulative sum of priorities
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            // Fallback to uniform if weights are zero
            return self.sample_uniform(weights.len(), count);
        }

        let mut selected = Vec::with_capacity(count);
        let mut available: Vec<usize> = (0..weights.len()).collect();

        for _ in 0..count.min(weights.len()) {
            // Calculate remaining total
            let remaining_total: f32 = available.iter().map(|&i| weights[i]).sum();
            if remaining_total <= 0.0 {
                break;
            }

            // Sample based on priorities
            let threshold = self.random_f32() * remaining_total;
            let mut cumsum = 0.0;

            for (pos, &idx) in available.iter().enumerate() {
                cumsum += weights[idx];
                if cumsum >= threshold {
                    selected.push(idx);
                    available.remove(pos);
                    break;
                }
            }
        }

        selected
    }

    /// Sample indices uniformly
    fn sample_uniform(&mut self, total: usize, count: usize) -> Vec<usize> {
        if total == 0 {
            return vec![];
        }

        let mut available: Vec<usize> = (0..total).collect();
        let mut selected = Vec::with_capacity(count);

        for _ in 0..count.min(total) {
            let idx = (self.next_random() as usize) % available.len();
            selected.push(available.remove(idx));
        }

        selected
    }

    /// Sample from episodic memory for replay using configured strategy
    fn sample_replay(&mut self, count: usize) -> Vec<OnlineExample> {
        if self.replay_metadata.is_empty() {
            return vec![];
        }

        let indices = match self.config.replay_strategy {
            ReplayStrategy::Uniform => self.sample_uniform(self.replay_metadata.len(), count),

            ReplayStrategy::PrioritizedLoss => {
                // Higher loss = higher priority (focus on hard examples)
                let alpha = self.config.priority_alpha;
                let weights: Vec<f32> = self.replay_metadata
                    .iter()
                    .map(|m| m.loss.powf(alpha))
                    .collect();
                self.sample_by_priority(&weights, count)
            }

            ReplayStrategy::PrioritizedRecency => {
                // More recent = higher priority
                let alpha = self.config.priority_alpha;
                let max_ts = self.replay_metadata.iter().map(|m| m.timestamp).max().unwrap_or(1);
                let weights: Vec<f32> = self.replay_metadata
                    .iter()
                    .map(|m| ((m.timestamp as f64 / max_ts as f64) as f32).powf(alpha))
                    .collect();
                self.sample_by_priority(&weights, count)
            }

            ReplayStrategy::Mixed => {
                // 50% prioritized by loss, 50% uniform
                let loss_count = count / 2;
                let uniform_count = count - loss_count;

                let alpha = self.config.priority_alpha;
                let weights: Vec<f32> = self.replay_metadata
                    .iter()
                    .map(|m| m.loss.powf(alpha))
                    .collect();

                let mut selected = self.sample_by_priority(&weights, loss_count);
                let uniform_samples = self.sample_uniform(self.replay_metadata.len(), uniform_count * 2);

                // Add uniform samples that aren't already selected
                for idx in uniform_samples {
                    if !selected.contains(&idx) && selected.len() < count {
                        selected.push(idx);
                    }
                }

                selected
            }

            ReplayStrategy::DomainBalanced => {
                // Sample equally from each domain
                let mut domain_indices: HashMap<String, Vec<usize>> = HashMap::new();
                for (i, meta) in self.replay_metadata.iter().enumerate() {
                    domain_indices.entry(meta.domain.clone()).or_default().push(i);
                }

                if domain_indices.is_empty() {
                    return vec![];
                }

                let per_domain = count / domain_indices.len();
                let mut selected = Vec::new();

                for indices in domain_indices.values() {
                    let domain_samples = self.sample_uniform(indices.len(), per_domain);
                    for idx in domain_samples {
                        if idx < indices.len() {
                            selected.push(indices[idx]);
                        }
                    }
                }

                // Fill remaining with uniform sampling
                while selected.len() < count {
                    let idx = (self.next_random() as usize) % self.replay_metadata.len();
                    if !selected.contains(&idx) {
                        selected.push(idx);
                    }
                }

                selected.truncate(count);
                selected
            }
        };

        // Convert selected indices to examples
        let mut examples = Vec::with_capacity(indices.len());
        for idx in indices {
            if let Some(meta) = self.replay_metadata.get(idx) {
                if let Some(episode) = self.episodic_memory.get(meta.episode_id) {
                    let input: Vec<f32> = episode.context.graph.node_indices()
                        .map(|i| episode.context.graph[i].activation)
                        .collect();
                    let target: Vec<f32> = episode.content.graph.node_indices()
                        .map(|i| episode.content.graph[i].activation)
                        .collect();

                    let mut example = OnlineExample::new(
                        format!("replay_{}", meta.episode_id),
                        input,
                        target,
                        meta.domain.clone(),
                    );
                    example.last_loss = meta.loss;
                    examples.push(example);
                }
            }
        }

        examples
    }

    /// Update replay metadata when an example is replayed (for tracking replay count)
    fn update_replay_metadata(&mut self, episode_id: u64, new_loss: f32) {
        if let Some(meta) = self.replay_metadata.iter_mut().find(|m| m.episode_id == episode_id) {
            meta.loss = new_loss;
            meta.replay_count += 1;
        }
    }

    /// Add metadata when storing a new example
    fn add_replay_metadata(&mut self, example: &OnlineExample, episode_id: u64) {
        let meta = ReplayMetadata {
            episode_id,
            loss: example.last_loss,
            domain: example.domain.clone(),
            timestamp: Self::current_timestamp(),
            replay_count: 0,
        };

        // Enforce capacity
        if self.replay_metadata.len() >= self.config.replay_capacity {
            // Remove lowest priority (lowest loss for prioritized, oldest for recency)
            match self.config.replay_strategy {
                ReplayStrategy::PrioritizedLoss | ReplayStrategy::Mixed => {
                    // Remove example with lowest loss (easiest)
                    if let Some((idx, _)) = self.replay_metadata
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal))
                    {
                        self.replay_metadata.remove(idx);
                    }
                }
                ReplayStrategy::PrioritizedRecency => {
                    // Remove oldest
                    if let Some((idx, _)) = self.replay_metadata
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, m)| m.timestamp)
                    {
                        self.replay_metadata.remove(idx);
                    }
                }
                _ => {
                    // Default: remove first
                    self.replay_metadata.remove(0);
                }
            }
        }

        self.replay_metadata.push(meta);
    }

    /// Get replay statistics
    pub fn replay_stats(&self) -> ReplayStats {
        let total_replays: usize = self.replay_metadata.iter().map(|m| m.replay_count).sum();
        let avg_loss = if self.replay_metadata.is_empty() {
            0.0
        } else {
            self.replay_metadata.iter().map(|m| m.loss).sum::<f32>() / self.replay_metadata.len() as f32
        };

        let mut domain_counts: HashMap<String, usize> = HashMap::new();
        for meta in &self.replay_metadata {
            *domain_counts.entry(meta.domain.clone()).or_insert(0) += 1;
        }

        ReplayStats {
            buffer_size: self.replay_metadata.len(),
            total_replays,
            avg_loss,
            domain_distribution: domain_counts,
            strategy: self.config.replay_strategy,
        }
    }

    /// Get EWC statistics
    pub fn ewc_stats(&self) -> EWCStats {
        self.ewc_state.fisher_stats()
    }

    /// Check if EWC is enabled and active
    pub fn ewc_active(&self) -> bool {
        self.config.use_ewc && self.ewc_state.is_active
    }

    /// Get configuration reference
    pub fn config(&self) -> &OnlineLearnerConfig {
        &self.config
    }

    /// Get current edge weights as HashMap for EWC
    fn get_current_edge_weights(&self) -> HashMap<EdgeKey, f32> {
        let mut weights = HashMap::new();
        for edge_ref in self.model.graph.edge_references() {
            let src = edge_ref.source().index();
            let tgt = edge_ref.target().index();
            weights.insert((src, tgt), edge_ref.weight().weight);
        }
        weights
    }

    /// Get edge gradients from model for EWC Fisher estimation
    fn get_edge_gradients(&self) -> HashMap<EdgeKey, f32> {
        let mut grads = HashMap::new();
        // Iterate over edges and get accumulated gradients
        for edge_ref in self.model.graph.edge_references() {
            let src = edge_ref.source();
            let tgt = edge_ref.target();
            if let Some(grad) = self.model.get_edge_grad(src, tgt) {
                grads.insert((src.index(), tgt.index()), grad);
            }
        }
        grads
    }

    /// Update EWC Fisher information from recent buffer (called during consolidation)
    fn update_ewc_fisher_from_buffer(&mut self) {
        if !self.config.use_ewc || self.recent_buffer.is_empty() {
            return;
        }

        // Sample from recent buffer to estimate Fisher information
        let sample_count = self.recent_buffer.len().min(100);
        for example in self.recent_buffer.iter().take(sample_count) {
            // Compute gradient for this example (forward + backward without step)
            self.model.zero_grad();
            self.embedding.zero_grad();

            // Set input activations
            let input_nodes = self.model.input_nodes();
            let mut input_map: HashMap<NodeId, f32> = HashMap::new();
            for (i, &node) in input_nodes.iter().enumerate() {
                if i < example.input.len() {
                    input_map.insert(node, example.input[i]);
                }
            }

            // Forward pass
            let _ = self.model.forward_with_inputs(&input_map);

            // Get output and compute loss gradient
            let output = self.model.get_output_activations(example.target.len());
            let output_nodes = self.model.output_nodes();
            let mut output_grad: HashMap<NodeId, Array1<f32>> = HashMap::new();
            for (i, &node) in output_nodes.iter().enumerate() {
                if i < output.len() && i < example.target.len() {
                    let grad = 2.0 * (output[i] - example.target[i]) / output.len() as f32;
                    output_grad.insert(node, Array1::from_vec(vec![grad]));
                }
            }

            // Backward pass to compute gradients
            self.model.backward_accumulate(&output_grad, &mut self.embedding);

            // Update Fisher with edge gradients
            let edge_grads = self.get_edge_gradients();
            self.ewc_state.update_fisher(&edge_grads);
        }
    }

    /// Consolidate EWC state (record optimal edge weights)
    fn consolidate_ewc(&mut self) {
        if self.config.use_ewc {
            let current_weights = self.get_current_edge_weights();
            self.ewc_state.consolidate(&current_weights);
        }
    }

    /// Compute EWC penalty for current model state
    pub fn compute_ewc_penalty(&self) -> f32 {
        if !self.config.use_ewc {
            return 0.0;
        }
        let current_weights = self.get_current_edge_weights();
        self.ewc_state.compute_penalty(&current_weights, self.config.ewc_lambda)
    }
}

/// Statistics about the replay buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayStats {
    /// Current buffer size
    pub buffer_size: usize,
    /// Total number of replay operations
    pub total_replays: usize,
    /// Average loss in buffer
    pub avg_loss: f32,
    /// Distribution by domain
    pub domain_distribution: HashMap<String, usize>,
    /// Current strategy
    pub strategy: ReplayStrategy,
}

/// Curriculum progression configuration (backend-204)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumConfig {
    /// Starting level (1-7)
    pub start_level: u8,
    /// Maximum level (1-7)
    pub max_level: u8,
    /// Examples required to advance to next level
    pub examples_per_level: usize,
    /// Loss threshold to advance (if achieved, advance early)
    pub advance_loss_threshold: Option<f32>,
    /// Minimum examples before early advancement
    pub min_examples_before_advance: usize,
    /// Whether to allow regression to lower levels
    pub allow_regression: bool,
    /// Loss threshold that triggers regression
    pub regression_threshold: f32,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            start_level: 1,
            max_level: 7,
            examples_per_level: 500,
            advance_loss_threshold: Some(0.1),
            min_examples_before_advance: 100,
            allow_regression: false,
            regression_threshold: 0.5,
        }
    }
}

impl CurriculumConfig {
    /// Create config for fast curriculum (fewer examples per level)
    pub fn fast() -> Self {
        Self {
            examples_per_level: 200,
            advance_loss_threshold: Some(0.2),
            min_examples_before_advance: 50,
            ..Default::default()
        }
    }

    /// Create config for thorough curriculum (more examples per level)
    pub fn thorough() -> Self {
        Self {
            examples_per_level: 1000,
            advance_loss_threshold: Some(0.05),
            min_examples_before_advance: 500,
            ..Default::default()
        }
    }

    /// Create config with specific level range
    pub fn with_levels(start: u8, max: u8) -> Self {
        Self {
            start_level: start.clamp(1, 7),
            max_level: max.clamp(1, 7),
            ..Default::default()
        }
    }
}

/// Curriculum state tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumState {
    /// Current level
    pub current_level: u8,
    /// Examples seen at current level
    pub examples_at_level: usize,
    /// Best loss achieved at current level
    pub best_loss_at_level: f32,
    /// Total level advancements
    pub advancements: usize,
    /// Total level regressions
    pub regressions: usize,
    /// Configuration
    config: CurriculumConfig,
}

impl CurriculumState {
    /// Create new curriculum state with config
    pub fn new(config: CurriculumConfig) -> Self {
        Self {
            current_level: config.start_level,
            examples_at_level: 0,
            best_loss_at_level: f32::MAX,
            advancements: 0,
            regressions: 0,
            config,
        }
    }

    /// Record example and return true if level changed
    pub fn record_example(&mut self, loss: f32) -> bool {
        self.examples_at_level += 1;
        if loss < self.best_loss_at_level {
            self.best_loss_at_level = loss;
        }

        let mut changed = false;

        // Check for advancement
        if self.current_level < self.config.max_level {
            let should_advance = self.examples_at_level >= self.config.examples_per_level
                || (self.config.advance_loss_threshold.is_some()
                    && self.examples_at_level >= self.config.min_examples_before_advance
                    && loss < self.config.advance_loss_threshold.unwrap());

            if should_advance {
                self.current_level += 1;
                self.examples_at_level = 0;
                self.best_loss_at_level = f32::MAX;
                self.advancements += 1;
                changed = true;
            }
        }

        // Check for regression
        if self.config.allow_regression
            && self.current_level > self.config.start_level
            && loss > self.config.regression_threshold
            && self.examples_at_level > self.config.min_examples_before_advance
        {
            self.current_level -= 1;
            self.examples_at_level = 0;
            self.best_loss_at_level = f32::MAX;
            self.regressions += 1;
            changed = true;
        }

        changed
    }

    /// Get current level
    pub fn level(&self) -> u8 {
        self.current_level
    }

    /// Check if at max level
    pub fn at_max_level(&self) -> bool {
        self.current_level >= self.config.max_level
    }

    /// Get progress percentage within current level
    pub fn level_progress(&self) -> f32 {
        (self.examples_at_level as f32 / self.config.examples_per_level as f32).min(1.0)
    }
}

/// Consolidation trigger mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConsolidationTrigger {
    /// Consolidate after fixed number of examples
    ExampleCount(usize),
    /// Consolidate after fixed number of batches
    BatchCount(usize),
    /// Consolidate when buffer reaches percentage of capacity
    BufferThreshold(u8),
    /// Consolidate when average loss drops below threshold
    LossThreshold,
    /// Never auto-consolidate (manual only)
    Manual,
}

impl Default for ConsolidationTrigger {
    fn default() -> Self {
        Self::ExampleCount(1000)
    }
}

/// Scheduler for periodic consolidation (backend-203)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationScheduler {
    /// Primary trigger mode
    pub trigger: ConsolidationTrigger,
    /// Loss threshold for LossThreshold mode
    pub loss_threshold: f32,
    /// Buffer threshold percentage (0-100) for BufferThreshold mode
    pub buffer_threshold_pct: u8,
    /// Minimum time between consolidations (in examples)
    pub min_interval: usize,
    /// Maximum time between consolidations (in examples)
    pub max_interval: usize,
    /// Examples since last consolidation
    examples_since: usize,
    /// Batches since last consolidation
    batches_since: usize,
    /// Last consolidation loss
    last_consolidation_loss: f32,
}

impl Default for ConsolidationScheduler {
    fn default() -> Self {
        Self {
            trigger: ConsolidationTrigger::ExampleCount(1000),
            loss_threshold: 0.1,
            buffer_threshold_pct: 80,
            min_interval: 100,
            max_interval: 10000,
            examples_since: 0,
            batches_since: 0,
            last_consolidation_loss: f32::MAX,
        }
    }
}

impl ConsolidationScheduler {
    /// Create scheduler with example count trigger
    pub fn with_example_count(count: usize) -> Self {
        Self {
            trigger: ConsolidationTrigger::ExampleCount(count),
            ..Default::default()
        }
    }

    /// Create scheduler with batch count trigger
    pub fn with_batch_count(count: usize) -> Self {
        Self {
            trigger: ConsolidationTrigger::BatchCount(count),
            ..Default::default()
        }
    }

    /// Create scheduler with buffer threshold trigger
    pub fn with_buffer_threshold(pct: u8) -> Self {
        Self {
            trigger: ConsolidationTrigger::BufferThreshold(pct.min(100)),
            buffer_threshold_pct: pct.min(100),
            ..Default::default()
        }
    }

    /// Create scheduler with loss threshold trigger
    pub fn with_loss_threshold(threshold: f32) -> Self {
        Self {
            trigger: ConsolidationTrigger::LossThreshold,
            loss_threshold: threshold,
            ..Default::default()
        }
    }

    /// Create manual-only scheduler
    pub fn manual() -> Self {
        Self {
            trigger: ConsolidationTrigger::Manual,
            ..Default::default()
        }
    }

    /// Record that examples were processed
    pub fn record_examples(&mut self, count: usize) {
        self.examples_since += count;
    }

    /// Record that a batch was processed
    pub fn record_batch(&mut self) {
        self.batches_since += 1;
    }

    /// Check if consolidation should trigger
    pub fn should_consolidate(&self, current_loss: f32, buffer_size: usize, buffer_capacity: usize) -> bool {
        // Enforce minimum interval
        if self.examples_since < self.min_interval {
            return false;
        }

        // Force consolidation at max interval
        if self.examples_since >= self.max_interval {
            return true;
        }

        match self.trigger {
            ConsolidationTrigger::ExampleCount(count) => self.examples_since >= count,
            ConsolidationTrigger::BatchCount(count) => self.batches_since >= count,
            ConsolidationTrigger::BufferThreshold(pct) => {
                let threshold = (buffer_capacity * pct as usize) / 100;
                buffer_size >= threshold
            }
            ConsolidationTrigger::LossThreshold => {
                current_loss < self.loss_threshold && current_loss < self.last_consolidation_loss * 0.9
            }
            ConsolidationTrigger::Manual => false,
        }
    }

    /// Reset counters after consolidation
    pub fn reset(&mut self, current_loss: f32) {
        self.examples_since = 0;
        self.batches_since = 0;
        self.last_consolidation_loss = current_loss;
    }

    /// Get examples since last consolidation
    pub fn examples_since_consolidation(&self) -> usize {
        self.examples_since
    }
}

impl OnlineLearner for MemoryOnlineLearner {
    fn learn_one(&mut self, example: OnlineExample) -> f32 {
        // Train on this example
        let loss = self.train_single(&example);

        // Update stats
        self.stats.examples_seen += 1;
        self.stats.avg_loss = 0.99 * self.stats.avg_loss + 0.01 * loss;
        if loss < self.stats.best_loss {
            self.stats.best_loss = loss;
        }
        *self.stats.domain_counts.entry(example.domain.clone()).or_insert(0) += 1;

        // Store in episodic memory
        let mut example_with_loss = example.clone();
        example_with_loss.last_loss = loss;
        let episode_id = self.next_episode_id;
        let episode = self.example_to_episode(&example_with_loss);
        self.episodic_memory.store(episode);

        // Track replay metadata for prioritized sampling
        self.add_replay_metadata(&example_with_loss, episode_id);

        // Add to recent buffer
        self.recent_buffer.push(example_with_loss);
        self.examples_since_consolidation += 1;

        // Check if consolidation needed
        if self.should_consolidate() {
            self.consolidate();
        }

        loss
    }

    fn learn_batch(&mut self, batch: &[OnlineExample]) -> f32 {
        let mut total_loss = 0.0;

        // Calculate how many replay samples to include
        let replay_count = (batch.len() as f32 * self.config.replay_ratio) as usize;
        let new_count = batch.len() - replay_count;

        // Zero gradients once for the batch
        self.model.zero_grad();
        self.embedding.zero_grad();

        // Train on new examples
        for example in batch.iter().take(new_count) {
            let loss = self.train_single(example);
            total_loss += loss;

            // Store in memory with metadata
            let mut example_with_loss = example.clone();
            example_with_loss.last_loss = loss;
            let episode_id = self.next_episode_id;
            let episode = self.example_to_episode(&example_with_loss);
            self.episodic_memory.store(episode);
            self.add_replay_metadata(&example_with_loss, episode_id);

            self.stats.examples_seen += 1;
            *self.stats.domain_counts.entry(example.domain.clone()).or_insert(0) += 1;
        }

        // Train on replay samples using configured strategy
        if replay_count > 0 {
            let replay_examples = self.sample_replay(replay_count);
            for example in &replay_examples {
                let loss = self.train_single(example);
                total_loss += loss;

                // Update loss in replay metadata for adaptive sampling
                if let Some(id_str) = example.id.strip_prefix("replay_") {
                    if let Ok(episode_id) = id_str.parse::<u64>() {
                        self.update_replay_metadata(episode_id, loss);
                    }
                }
            }
        }

        let avg_loss = total_loss / batch.len() as f32;
        self.stats.batches_trained += 1;
        self.stats.avg_loss = 0.9 * self.stats.avg_loss + 0.1 * avg_loss;
        if avg_loss < self.stats.best_loss {
            self.stats.best_loss = avg_loss;
        }

        self.examples_since_consolidation += batch.len();

        if self.should_consolidate() {
            self.consolidate();
        }

        avg_loss
    }

    fn consolidate(&mut self) {
        // Apply retention policy to episodic memory
        let policy = RetentionPolicy {
            max_episodes: Some(self.config.replay_capacity),
            min_importance: 0.01,
            consolidate_similar: true,
            ..Default::default()
        };
        self.episodic_memory.consolidate(&policy);

        // Trigger continual learning replay and integration
        // This does offline processing similar to sleep-based memory consolidation
        self.continual_learning.replay_and_integrate();

        // EWC: Update Fisher information from recent experiences before clearing buffer
        self.update_ewc_fisher_from_buffer();

        // EWC: Consolidate current parameters as optimal
        self.consolidate_ewc();

        // Clear recent buffer
        self.recent_buffer.clear();
        self.examples_since_consolidation = 0;
        self.stats.consolidations += 1;
    }

    fn model(&self) -> &DagNN {
        &self.model
    }

    fn model_mut(&mut self) -> &mut DagNN {
        &mut self.model
    }

    fn stats(&self) -> &OnlineLearnerStats {
        &self.stats
    }

    fn should_consolidate(&self) -> bool {
        self.examples_since_consolidation >= self.config.consolidation_interval
    }

    fn save_state<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        // Create unified checkpoint for model
        let mut checkpoint = UnifiedCheckpoint::new();
        checkpoint.add_module(&self.model)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let model_json = serde_json::to_string(&checkpoint)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Serialize stats
        let stats_json = serde_json::to_string(&self.stats)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Serialize config
        let config_json = serde_json::to_string(&self.config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Write as JSON object
        let state = serde_json::json!({
            "model": model_json,
            "stats": stats_json,
            "config": config_json,
        });
        writeln!(writer, "{}", serde_json::to_string_pretty(&state)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?)?;

        Ok(())
    }

    fn load_state<R: Read>(&mut self, mut reader: R) -> std::io::Result<()> {
        let mut content = String::new();
        reader.read_to_string(&mut content)?;

        let state: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Parse model from checkpoint
        let model_json_str = state["model"].as_str()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing model field"))?;
        let checkpoint: UnifiedCheckpoint = serde_json::from_str(model_json_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.model = checkpoint.load_module()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        // Parse stats
        let stats_json_str = state["stats"].as_str()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing stats field"))?;
        self.stats = serde_json::from_str(stats_json_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Parse config
        let config_json_str = state["config"].as_str()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing config field"))?;
        self.config = serde_json::from_str(config_json_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_learner_config_default() {
        let config = OnlineLearnerConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.replay_capacity, 10_000);
        assert!(!config.use_ewc);
    }

    #[test]
    fn test_online_learner_config_fast() {
        let config = OnlineLearnerConfig::fast();
        assert_eq!(config.batch_size, 16);
        assert!(config.learning_rate > 0.001);
    }

    #[test]
    fn test_online_learner_config_stable() {
        let config = OnlineLearnerConfig::stable();
        assert!(config.use_ewc);
        assert!(config.replay_ratio > 0.5);
    }

    #[test]
    fn test_online_example_creation() {
        let example = OnlineExample::new(
            "test_1",
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
            "math",
        ).with_level(3);

        assert_eq!(example.id, "test_1");
        assert_eq!(example.level, 3);
        assert_eq!(example.domain, "math");
    }

    #[test]
    fn test_memory_online_learner_creation() {
        let config = OnlineLearnerConfig::default();
        let learner = MemoryOnlineLearner::with_default_model(config);

        assert_eq!(learner.stats().examples_seen, 0);
        assert_eq!(learner.stats().current_level, 1);
    }

    #[test]
    fn test_learn_one() {
        let config = OnlineLearnerConfig::fast();
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        let example = OnlineExample::new(
            "test_1",
            vec![0.5; 10],
            vec![0.5; 5],
            "math",
        );

        let loss = learner.learn_one(example);
        assert!(loss >= 0.0);
        assert_eq!(learner.stats().examples_seen, 1);
    }

    #[test]
    fn test_learn_batch() {
        let config = OnlineLearnerConfig::fast();
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        let batch: Vec<OnlineExample> = (0..10)
            .map(|i| OnlineExample::new(
                format!("test_{}", i),
                vec![0.5; 10],
                vec![0.5; 5],
                "math",
            ))
            .collect();

        let loss = learner.learn_batch(&batch);
        assert!(loss >= 0.0);
        assert_eq!(learner.stats().batches_trained, 1);
    }

    #[test]
    fn test_should_consolidate() {
        let mut config = OnlineLearnerConfig::fast();
        config.consolidation_interval = 5;
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        // Learn 4 examples - should not consolidate
        for i in 0..4 {
            let example = OnlineExample::new(format!("test_{}", i), vec![0.5; 10], vec![0.5; 5], "math");
            learner.learn_one(example);
        }
        assert!(!learner.should_consolidate());

        // Learn 1 more - should trigger consolidation
        let example = OnlineExample::new("test_4", vec![0.5; 10], vec![0.5; 5], "math");
        learner.learn_one(example);
        // Consolidation already happened in learn_one
        assert_eq!(learner.stats().consolidations, 1);
    }

    #[test]
    fn test_stats_tracking() {
        let config = OnlineLearnerConfig::fast();
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        // Learn examples from different domains
        for domain in &["math", "text", "vision"] {
            let example = OnlineExample::new(
                format!("test_{}", domain),
                vec![0.5; 10],
                vec![0.5; 5],
                *domain,
            );
            learner.learn_one(example);
        }

        let stats = learner.stats();
        assert_eq!(stats.examples_seen, 3);
        assert_eq!(stats.domain_counts.get("math"), Some(&1));
        assert_eq!(stats.domain_counts.get("text"), Some(&1));
        assert_eq!(stats.domain_counts.get("vision"), Some(&1));
    }

    // ========== New tests for backend-201: Experience Replay ==========

    #[test]
    fn test_replay_strategy_enum() {
        // Test default is Mixed
        assert_eq!(ReplayStrategy::default(), ReplayStrategy::Mixed);

        // Test serialization
        let strategy = ReplayStrategy::PrioritizedLoss;
        let json = serde_json::to_string(&strategy).unwrap();
        assert!(json.contains("PrioritizedLoss"));
    }

    #[test]
    fn test_config_presets_have_strategies() {
        let fast = OnlineLearnerConfig::fast();
        assert_eq!(fast.replay_strategy, ReplayStrategy::Uniform);

        let stable = OnlineLearnerConfig::stable();
        assert_eq!(stable.replay_strategy, ReplayStrategy::PrioritizedLoss);

        let prioritized = OnlineLearnerConfig::prioritized();
        assert_eq!(prioritized.replay_strategy, ReplayStrategy::PrioritizedLoss);
        assert_eq!(prioritized.priority_alpha, 1.0);

        let balanced = OnlineLearnerConfig::balanced();
        assert_eq!(balanced.replay_strategy, ReplayStrategy::DomainBalanced);
    }

    #[test]
    fn test_replay_metadata_tracking() {
        let config = OnlineLearnerConfig::fast();
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        // Learn some examples
        for i in 0..5 {
            let example = OnlineExample::new(
                format!("test_{}", i),
                vec![0.5; 10],
                vec![0.5; 5],
                "math",
            );
            learner.learn_one(example);
        }

        // Check replay stats
        let replay_stats = learner.replay_stats();
        assert_eq!(replay_stats.buffer_size, 5);
        assert_eq!(replay_stats.total_replays, 0); // No replays yet
        assert!(replay_stats.avg_loss >= 0.0);
    }

    #[test]
    fn test_replay_with_prioritized_loss() {
        let mut config = OnlineLearnerConfig::default();
        config.replay_strategy = ReplayStrategy::PrioritizedLoss;
        config.priority_alpha = 1.0;
        config.replay_ratio = 0.5;
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        // Learn examples with varying difficulties (simulated by different inputs)
        for i in 0..10 {
            let example = OnlineExample::new(
                format!("test_{}", i),
                vec![(i as f32) * 0.1; 10],
                vec![0.5; 5],
                "math",
            );
            learner.learn_one(example);
        }

        // Now do batch learning which will trigger replay
        let batch: Vec<OnlineExample> = (10..15)
            .map(|i| OnlineExample::new(
                format!("test_{}", i),
                vec![0.5; 10],
                vec![0.5; 5],
                "math",
            ))
            .collect();

        learner.learn_batch(&batch);

        // Verify replay occurred
        let stats = learner.replay_stats();
        assert!(stats.total_replays > 0, "Expected some replays to have occurred");
    }

    #[test]
    fn test_replay_domain_balanced() {
        let mut config = OnlineLearnerConfig::balanced();
        config.replay_ratio = 0.5;
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        // Add examples from multiple domains
        for domain in &["math", "text", "vision", "code"] {
            for i in 0..5 {
                let example = OnlineExample::new(
                    format!("{}_{}", domain, i),
                    vec![0.5; 10],
                    vec![0.5; 5],
                    *domain,
                );
                learner.learn_one(example);
            }
        }

        let stats = learner.replay_stats();
        assert_eq!(stats.buffer_size, 20); // 4 domains * 5 examples
        assert_eq!(stats.domain_distribution.len(), 4);
        for &count in stats.domain_distribution.values() {
            assert_eq!(count, 5);
        }
    }

    #[test]
    fn test_replay_capacity_enforcement() {
        let mut config = OnlineLearnerConfig::fast();
        config.replay_capacity = 5;
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        // Add more than capacity
        for i in 0..10 {
            let example = OnlineExample::new(
                format!("test_{}", i),
                vec![0.5; 10],
                vec![0.5; 5],
                "math",
            );
            learner.learn_one(example);
        }

        // Buffer should be capped at capacity
        let stats = learner.replay_stats();
        assert_eq!(stats.buffer_size, 5);
    }

    #[test]
    fn test_replay_mixed_strategy() {
        let mut config = OnlineLearnerConfig::default();
        config.replay_strategy = ReplayStrategy::Mixed;
        config.priority_alpha = 0.6;
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        // Add examples
        for i in 0..10 {
            let example = OnlineExample::new(
                format!("test_{}", i),
                vec![0.5; 10],
                vec![0.5; 5],
                "math",
            );
            learner.learn_one(example);
        }

        let stats = learner.replay_stats();
        assert_eq!(stats.strategy, ReplayStrategy::Mixed);
        assert_eq!(stats.buffer_size, 10);
    }

    #[test]
    fn test_replay_recency_strategy() {
        let mut config = OnlineLearnerConfig::default();
        config.replay_strategy = ReplayStrategy::PrioritizedRecency;
        config.priority_alpha = 1.0;
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        // Add examples
        for i in 0..10 {
            let example = OnlineExample::new(
                format!("test_{}", i),
                vec![0.5; 10],
                vec![0.5; 5],
                "math",
            );
            learner.learn_one(example);
        }

        let stats = learner.replay_stats();
        assert_eq!(stats.strategy, ReplayStrategy::PrioritizedRecency);
    }

    #[test]
    fn test_replay_stats_serialization() {
        let mut domain_dist = HashMap::new();
        domain_dist.insert("math".to_string(), 10);
        domain_dist.insert("text".to_string(), 5);

        let stats = ReplayStats {
            buffer_size: 15,
            total_replays: 100,
            avg_loss: 0.25,
            domain_distribution: domain_dist,
            strategy: ReplayStrategy::PrioritizedLoss,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let parsed: ReplayStats = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.buffer_size, 15);
        assert_eq!(parsed.total_replays, 100);
        assert_eq!(parsed.strategy, ReplayStrategy::PrioritizedLoss);
    }

    // ========== Tests for backend-203: ConsolidationScheduler ==========

    #[test]
    fn test_consolidation_trigger_default() {
        let trigger = ConsolidationTrigger::default();
        assert_eq!(trigger, ConsolidationTrigger::ExampleCount(1000));
    }

    #[test]
    fn test_consolidation_scheduler_default() {
        let scheduler = ConsolidationScheduler::default();
        assert_eq!(scheduler.trigger, ConsolidationTrigger::ExampleCount(1000));
        assert_eq!(scheduler.min_interval, 100);
        assert_eq!(scheduler.max_interval, 10000);
    }

    #[test]
    fn test_consolidation_scheduler_constructors() {
        let s1 = ConsolidationScheduler::with_example_count(500);
        assert_eq!(s1.trigger, ConsolidationTrigger::ExampleCount(500));

        let s2 = ConsolidationScheduler::with_batch_count(10);
        assert_eq!(s2.trigger, ConsolidationTrigger::BatchCount(10));

        let s3 = ConsolidationScheduler::with_buffer_threshold(75);
        assert_eq!(s3.trigger, ConsolidationTrigger::BufferThreshold(75));

        let s4 = ConsolidationScheduler::with_loss_threshold(0.05);
        assert_eq!(s4.trigger, ConsolidationTrigger::LossThreshold);
        assert_eq!(s4.loss_threshold, 0.05);

        let s5 = ConsolidationScheduler::manual();
        assert_eq!(s5.trigger, ConsolidationTrigger::Manual);
    }

    #[test]
    fn test_consolidation_scheduler_example_count() {
        let mut scheduler = ConsolidationScheduler::with_example_count(100);
        scheduler.min_interval = 10;

        // Not enough examples yet
        scheduler.record_examples(50);
        assert!(!scheduler.should_consolidate(0.5, 50, 100));

        // Now enough
        scheduler.record_examples(60);
        assert!(scheduler.should_consolidate(0.5, 50, 100));

        // Reset and verify
        scheduler.reset(0.5);
        assert_eq!(scheduler.examples_since_consolidation(), 0);
        assert!(!scheduler.should_consolidate(0.5, 50, 100));
    }

    #[test]
    fn test_consolidation_scheduler_batch_count() {
        let mut scheduler = ConsolidationScheduler::with_batch_count(5);
        scheduler.min_interval = 0;

        for _ in 0..4 {
            scheduler.record_batch();
        }
        assert!(!scheduler.should_consolidate(0.5, 50, 100));

        scheduler.record_batch();
        assert!(scheduler.should_consolidate(0.5, 50, 100));
    }

    #[test]
    fn test_consolidation_scheduler_buffer_threshold() {
        let mut scheduler = ConsolidationScheduler::with_buffer_threshold(80);
        scheduler.min_interval = 0;
        scheduler.record_examples(1); // Need at least 1 to pass min_interval=0

        // 70% full - not triggered
        assert!(!scheduler.should_consolidate(0.5, 70, 100));

        // 80% full - triggered
        assert!(scheduler.should_consolidate(0.5, 80, 100));
    }

    #[test]
    fn test_consolidation_scheduler_min_max_interval() {
        let mut scheduler = ConsolidationScheduler::with_example_count(100);
        scheduler.min_interval = 50;
        scheduler.max_interval = 200;

        // Below min_interval - never triggers
        scheduler.record_examples(30);
        assert!(!scheduler.should_consolidate(0.5, 50, 100));

        // Above threshold but below min - still no
        scheduler.record_examples(80);
        assert!(scheduler.should_consolidate(0.5, 50, 100));

        // At max_interval - always triggers
        scheduler.reset(0.5);
        scheduler.record_examples(200);
        assert!(scheduler.should_consolidate(0.5, 0, 100));
    }

    #[test]
    fn test_consolidation_scheduler_manual() {
        let mut scheduler = ConsolidationScheduler::manual();
        scheduler.min_interval = 0;
        scheduler.record_examples(10000);
        scheduler.record_batch();

        // Manual mode never auto-triggers (unless max_interval reached)
        scheduler.max_interval = 100000;
        assert!(!scheduler.should_consolidate(0.0, 100, 100));
    }

    // ========== Tests for backend-204: Curriculum Progression ==========

    #[test]
    fn test_curriculum_config_default() {
        let config = CurriculumConfig::default();
        assert_eq!(config.start_level, 1);
        assert_eq!(config.max_level, 7);
        assert_eq!(config.examples_per_level, 500);
        assert!(!config.allow_regression);
    }

    #[test]
    fn test_curriculum_config_fast() {
        let config = CurriculumConfig::fast();
        assert_eq!(config.examples_per_level, 200);
        assert_eq!(config.min_examples_before_advance, 50);
        assert!(config.advance_loss_threshold.is_some());
    }

    #[test]
    fn test_curriculum_config_thorough() {
        let config = CurriculumConfig::thorough();
        assert_eq!(config.examples_per_level, 1000);
        assert_eq!(config.min_examples_before_advance, 500);
    }

    #[test]
    fn test_curriculum_config_with_levels() {
        let config = CurriculumConfig::with_levels(3, 5);
        assert_eq!(config.start_level, 3);
        assert_eq!(config.max_level, 5);

        // Test clamping
        let config2 = CurriculumConfig::with_levels(0, 10);
        assert_eq!(config2.start_level, 1);
        assert_eq!(config2.max_level, 7);
    }

    #[test]
    fn test_curriculum_state_new() {
        let config = CurriculumConfig::default();
        let state = CurriculumState::new(config);
        assert_eq!(state.level(), 1);
        assert_eq!(state.examples_at_level, 0);
        assert!(!state.at_max_level());
    }

    #[test]
    fn test_curriculum_state_record_example() {
        let config = CurriculumConfig {
            examples_per_level: 10,
            ..CurriculumConfig::default()
        };
        let mut state = CurriculumState::new(config);

        // Record examples below threshold
        for _ in 0..9 {
            let changed = state.record_example(0.5);
            assert!(!changed);
        }
        assert_eq!(state.level(), 1);
        assert_eq!(state.examples_at_level, 9);

        // This should trigger advancement
        let changed = state.record_example(0.5);
        assert!(changed);
        assert_eq!(state.level(), 2);
        assert_eq!(state.examples_at_level, 0);
        assert_eq!(state.advancements, 1);
    }

    #[test]
    fn test_curriculum_state_early_advance() {
        let config = CurriculumConfig {
            examples_per_level: 100,
            advance_loss_threshold: Some(0.1),
            min_examples_before_advance: 5,
            ..CurriculumConfig::default()
        };
        let mut state = CurriculumState::new(config);

        // Not enough examples yet
        for _ in 0..4 {
            let changed = state.record_example(0.05);
            assert!(!changed);
        }

        // Now meets min_examples and loss threshold
        let changed = state.record_example(0.05);
        assert!(changed);
        assert_eq!(state.level(), 2);
    }

    #[test]
    fn test_curriculum_state_regression() {
        let config = CurriculumConfig {
            examples_per_level: 10,
            allow_regression: true,
            regression_threshold: 0.5,
            min_examples_before_advance: 5,
            ..CurriculumConfig::default()
        };
        let mut state = CurriculumState::new(config);

        // Advance to level 2
        for _ in 0..10 {
            state.record_example(0.2);
        }
        assert_eq!(state.level(), 2);

        // Record examples at level 2, then trigger regression
        for _ in 0..5 {
            state.record_example(0.3);
        }
        // This high loss triggers regression
        let changed = state.record_example(0.8);
        assert!(changed);
        assert_eq!(state.level(), 1);
        assert_eq!(state.regressions, 1);
    }

    #[test]
    fn test_curriculum_state_max_level() {
        let config = CurriculumConfig {
            max_level: 2,
            examples_per_level: 5,
            ..CurriculumConfig::default()
        };
        let mut state = CurriculumState::new(config);

        // Advance to max level
        for _ in 0..5 {
            state.record_example(0.2);
        }
        assert_eq!(state.level(), 2);
        assert!(state.at_max_level());

        // Should not advance past max
        for _ in 0..10 {
            let changed = state.record_example(0.1);
            assert!(!changed);
        }
        assert_eq!(state.level(), 2);
    }

    #[test]
    fn test_curriculum_state_level_progress() {
        let config = CurriculumConfig {
            examples_per_level: 100,
            ..CurriculumConfig::default()
        };
        let mut state = CurriculumState::new(config);

        assert_eq!(state.level_progress(), 0.0);

        for _ in 0..50 {
            state.record_example(0.5);
        }
        assert!((state.level_progress() - 0.5).abs() < 0.01);

        for _ in 0..50 {
            state.record_example(0.5);
        }
        // After advancing, progress resets
        assert_eq!(state.level_progress(), 0.0);
    }

    #[test]
    fn test_curriculum_state_best_loss_tracking() {
        let config = CurriculumConfig::default();
        let mut state = CurriculumState::new(config);

        state.record_example(0.5);
        assert_eq!(state.best_loss_at_level, 0.5);

        state.record_example(0.3);
        assert_eq!(state.best_loss_at_level, 0.3);

        state.record_example(0.4);
        assert_eq!(state.best_loss_at_level, 0.3); // Still 0.3
    }

    #[test]
    fn test_curriculum_state_serialization() {
        let config = CurriculumConfig::fast();
        let mut state = CurriculumState::new(config);
        state.record_example(0.3);
        state.record_example(0.2);

        let json = serde_json::to_string(&state).unwrap();
        let parsed: CurriculumState = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.level(), state.level());
        assert_eq!(parsed.examples_at_level, state.examples_at_level);
        assert_eq!(parsed.best_loss_at_level, state.best_loss_at_level);
    }

    // ========== Tests for backend-205: EWC (Elastic Weight Consolidation) ==========

    #[test]
    fn test_ewc_state_new() {
        let state = EWCState::new();
        assert!(!state.is_active);
        assert_eq!(state.task_count, 0);
        assert_eq!(state.samples_used, 0);
        assert!(state.fisher_diag.is_empty());
        assert!(state.optimal_params.is_empty());
    }

    #[test]
    fn test_ewc_state_update_fisher() {
        let mut state = EWCState::new();

        // Simulate gradient samples (edge keys as (src, tgt) pairs)
        let mut grads = HashMap::new();
        grads.insert((0, 1), 1.0);  // edge from node 0 to node 1
        grads.insert((1, 2), 2.0);  // edge from node 1 to node 2

        state.update_fisher(&grads);

        assert_eq!(state.samples_used, 1);
        assert_eq!(state.fisher_diag.get(&(0, 1)), Some(&1.0)); // grad^2 = 1.0
        assert_eq!(state.fisher_diag.get(&(1, 2)), Some(&4.0)); // grad^2 = 4.0

        // Add another sample - should average
        let mut grads2 = HashMap::new();
        grads2.insert((0, 1), 3.0); // grad^2 = 9.0
        grads2.insert((1, 2), 0.0); // grad^2 = 0.0

        state.update_fisher(&grads2);

        assert_eq!(state.samples_used, 2);
        // Online mean: (1.0 + 9.0) / 2 = 5.0
        assert!((state.fisher_diag.get(&(0, 1)).unwrap() - 5.0).abs() < 0.001);
        // Online mean: (4.0 + 0.0) / 2 = 2.0
        assert!((state.fisher_diag.get(&(1, 2)).unwrap() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_ewc_state_consolidate() {
        let mut state = EWCState::new();
        assert!(!state.is_active);

        let mut params = HashMap::new();
        params.insert((0, 1), 0.5);
        params.insert((1, 2), -0.3);

        state.consolidate(&params);

        assert!(state.is_active);
        assert_eq!(state.task_count, 1);
        assert_eq!(state.optimal_params.get(&(0, 1)), Some(&0.5));
        assert_eq!(state.optimal_params.get(&(1, 2)), Some(&-0.3));
    }

    #[test]
    fn test_ewc_state_compute_penalty_inactive() {
        let state = EWCState::new();
        let params = HashMap::new();
        let penalty = state.compute_penalty(&params, 1.0);
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn test_ewc_state_compute_penalty_active() {
        let mut state = EWCState::new();

        // Set Fisher values
        state.fisher_diag.insert((0, 1), 1.0);
        state.fisher_diag.insert((1, 2), 2.0);

        // Set optimal params
        let mut optimal = HashMap::new();
        optimal.insert((0, 1), 0.5);
        optimal.insert((1, 2), 1.0);
        state.consolidate(&optimal);

        // Current params
        let mut current = HashMap::new();
        current.insert((0, 1), 0.7);  // diff = 0.2
        current.insert((1, 2), 0.5);  // diff = -0.5

        // penalty = (lambda/2) * sum(F * diff^2)
        // = (1.0/2) * (1.0 * 0.04 + 2.0 * 0.25)
        // = 0.5 * (0.04 + 0.5) = 0.5 * 0.54 = 0.27
        let penalty = state.compute_penalty(&current, 1.0);
        assert!((penalty - 0.27).abs() < 0.001);
    }

    #[test]
    fn test_ewc_state_gradient_penalty() {
        let mut state = EWCState::new();

        state.fisher_diag.insert((0, 1), 2.0);
        let mut optimal = HashMap::new();
        optimal.insert((0, 1), 0.5);
        state.consolidate(&optimal);

        // grad_ewc = lambda * F * (current - optimal)
        // = 1.0 * 2.0 * (0.8 - 0.5) = 0.6
        let grad = state.compute_gradient_penalty((0, 1), 0.8, 1.0);
        assert!((grad - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_ewc_state_reset_fisher() {
        let mut state = EWCState::new();
        state.fisher_diag.insert((0, 1), 1.0);
        state.samples_used = 10;

        state.reset_fisher();

        assert!(state.fisher_diag.is_empty());
        assert_eq!(state.samples_used, 0);
    }

    #[test]
    fn test_ewc_stats() {
        let mut state = EWCState::new();
        state.fisher_diag.insert((0, 1), 1.0);
        state.fisher_diag.insert((1, 2), 3.0);
        state.samples_used = 5;
        state.task_count = 2;
        state.is_active = true;

        let stats = state.fisher_stats();
        assert_eq!(stats.param_count, 2);
        assert_eq!(stats.task_count, 2);
        assert_eq!(stats.samples_used, 5);
        assert!((stats.fisher_mean - 2.0).abs() < 0.001);
        assert_eq!(stats.fisher_max, 3.0);
        assert_eq!(stats.fisher_min, 1.0);
        assert!(stats.is_active);
    }

    #[test]
    fn test_ewc_integration_with_learner() {
        let mut config = OnlineLearnerConfig::stable(); // stable() enables EWC
        config.consolidation_interval = 5;
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        // Initially EWC should not be active (no consolidation yet)
        assert!(!learner.ewc_active());

        // Verify config was applied correctly
        assert!(learner.config.use_ewc, "EWC should be enabled in stable config");
        assert!(learner.config.ewc_lambda > 0.0, "EWC lambda should be positive");

        // Train some examples to trigger consolidation
        for i in 0..6 {
            let example = OnlineExample::new(
                format!("test_{}", i),
                vec![0.5; 10],
                vec![0.5; 5],
                "math",
            );
            learner.learn_one(example);
        }

        // Check that consolidation happened (stats.consolidations > 0)
        let train_stats = learner.stats();
        assert!(train_stats.consolidations >= 1, "Consolidation should have triggered");

        // EWC is_active depends on having edge weights to protect
        // With a minimal model, there may be no edges
        let ewc_stats = learner.ewc_stats();
        // If there are params, EWC should be active
        if ewc_stats.param_count > 0 {
            assert!(ewc_stats.is_active);
            assert!(learner.ewc_active());
        }
    }

    #[test]
    fn test_ewc_penalty_computation_on_learner() {
        let mut config = OnlineLearnerConfig::stable();
        config.consolidation_interval = 3;
        let mut learner = MemoryOnlineLearner::with_default_model(config);

        // Before any consolidation
        let penalty_before = learner.compute_ewc_penalty();
        assert_eq!(penalty_before, 0.0);

        // Train and trigger consolidation
        for i in 0..4 {
            let example = OnlineExample::new(
                format!("test_{}", i),
                vec![0.5; 10],
                vec![0.5; 5],
                "math",
            );
            learner.learn_one(example);
        }

        // Now penalty should be computable (though may be small)
        let penalty_after = learner.compute_ewc_penalty();
        assert!(penalty_after >= 0.0);
    }

    #[test]
    fn test_ewc_disabled_by_default() {
        let config = OnlineLearnerConfig::default();
        assert!(!config.use_ewc);

        let learner = MemoryOnlineLearner::with_default_model(config);
        assert!(!learner.ewc_active());
    }

    #[test]
    fn test_ewc_stats_serialization() {
        let stats = EWCStats {
            param_count: 100,
            task_count: 3,
            samples_used: 500,
            fisher_mean: 0.5,
            fisher_max: 2.0,
            fisher_min: 0.01,
            is_active: true,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let parsed: EWCStats = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.param_count, 100);
        assert_eq!(parsed.task_count, 3);
        assert!(parsed.is_active);
    }
}
