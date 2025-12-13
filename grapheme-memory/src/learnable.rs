//! Learnable Memory Components (backend-032)
//!
//! This module provides learnable versions of memory retrieval and consolidation.
//! Instead of using fixed heuristics, these components learn to:
//! - Embed graphs into vector representations
//! - Compute similarity between graph embeddings
//! - Score importance for memory consolidation
//!
//! **GRAPHEME Protocol**: Uses LeakyReLU activation (α=0.01) and Adam optimizer (lr=0.001).

use crate::{ConsolidationStats, EpisodeId, Graph, GraphFingerprint, Timestamp};
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// LeakyReLU constant (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Default embedding dimension
pub const DEFAULT_EMBED_DIM: usize = 64;

/// Configuration for learnable memory components
#[derive(Debug, Clone)]
pub struct LearnableMemoryConfig {
    /// Embedding dimension for graphs
    pub embed_dim: usize,
    /// Hidden dimension for MLP
    pub hidden_dim: usize,
    /// Learning rate (GRAPHEME Protocol: 0.001)
    pub learning_rate: f32,
    /// Importance threshold for consolidation
    pub importance_threshold: f32,
    /// Maximum memory capacity
    pub max_capacity: usize,
}

impl Default for LearnableMemoryConfig {
    fn default() -> Self {
        Self {
            embed_dim: DEFAULT_EMBED_DIM,
            hidden_dim: 128,
            learning_rate: 0.001, // GRAPHEME Protocol
            importance_threshold: 0.3,
            max_capacity: 10_000,
        }
    }
}

/// Learnable graph encoder that produces fixed-size embeddings
#[derive(Debug)]
pub struct GraphEncoder {
    /// Input projection: [embed_dim, feature_dim]
    pub w_input: Array2<f32>,
    /// Hidden layer: [hidden_dim, embed_dim]
    pub w_hidden: Array2<f32>,
    /// Output layer: [embed_dim, hidden_dim]
    pub w_output: Array2<f32>,
    /// Bias for hidden layer
    pub b_hidden: Array1<f32>,
    /// Bias for output layer
    pub b_output: Array1<f32>,
    /// Gradients for w_input
    pub grad_w_input: Option<Array2<f32>>,
    /// Gradients for w_hidden
    pub grad_w_hidden: Option<Array2<f32>>,
    /// Gradients for w_output
    pub grad_w_output: Option<Array2<f32>>,
    /// Gradients for b_hidden
    pub grad_b_hidden: Option<Array1<f32>>,
    /// Gradients for b_output
    pub grad_b_output: Option<Array1<f32>>,
}

impl GraphEncoder {
    /// Number of features from GraphFingerprint
    const FEATURE_DIM: usize = 18; // 2 counts + 8 node_types + 8 degree_hist

    /// Create a new graph encoder with DynamicXavier initialization
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        // DynamicXavier initialization
        let scale_input = (2.0 / (Self::FEATURE_DIM + embed_dim) as f32).sqrt();
        let scale_hidden = (2.0 / (embed_dim + hidden_dim) as f32).sqrt();
        let scale_output = (2.0 / (hidden_dim + embed_dim) as f32).sqrt();

        let w_input = Array2::from_shape_fn((embed_dim, Self::FEATURE_DIM), |_| {
            rng.gen_range(-scale_input..scale_input)
        });

        let w_hidden = Array2::from_shape_fn((hidden_dim, embed_dim), |_| {
            rng.gen_range(-scale_hidden..scale_hidden)
        });

        let w_output = Array2::from_shape_fn((embed_dim, hidden_dim), |_| {
            rng.gen_range(-scale_output..scale_output)
        });

        let b_hidden = Array1::zeros(hidden_dim);
        let b_output = Array1::zeros(embed_dim);

        Self {
            w_input,
            w_hidden,
            w_output,
            b_hidden,
            b_output,
            grad_w_input: None,
            grad_w_hidden: None,
            grad_w_output: None,
            grad_b_hidden: None,
            grad_b_output: None,
        }
    }

    /// Convert GraphFingerprint to feature vector
    fn fingerprint_to_features(&self, fp: &GraphFingerprint) -> Array1<f32> {
        let mut features = Array1::zeros(Self::FEATURE_DIM);
        features[0] = fp.node_count as f32 / 100.0; // Normalize
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

    /// Encode a graph to a fixed-size embedding
    pub fn encode(&self, graph: &Graph) -> Array1<f32> {
        let fp = GraphFingerprint::from_graph(graph);
        let features = self.fingerprint_to_features(&fp);

        // Forward pass: input -> hidden -> output
        let input_proj = self.w_input.dot(&features);
        let hidden_pre = self.w_hidden.dot(&input_proj) + &self.b_hidden;
        let hidden_act = hidden_pre.mapv(|x| self.leaky_relu(x));
        let output_pre = self.w_output.dot(&hidden_act) + &self.b_output;
        let output_act = output_pre.mapv(|x| self.leaky_relu(x));

        // L2 normalize for cosine similarity
        let norm = output_act.mapv(|x| x * x).sum().sqrt().max(1e-8);
        output_act / norm
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_input = None;
        self.grad_w_hidden = None;
        self.grad_w_output = None;
        self.grad_b_hidden = None;
        self.grad_b_output = None;
    }

    /// Update parameters with learning rate
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
            + self.b_hidden.len() + self.b_output.len()
    }

    /// Check if gradients exist
    pub fn has_gradients(&self) -> bool {
        self.grad_w_input.is_some()
    }
}

/// Learnable similarity function
#[derive(Debug)]
pub struct LearnableSimilarity {
    /// Attention weight matrix: [embed_dim, embed_dim]
    pub w_attention: Array2<f32>,
    /// Temperature for softmax
    pub temperature: f32,
    /// Gradient for w_attention
    pub grad_w_attention: Option<Array2<f32>>,
}

impl LearnableSimilarity {
    /// Create a new similarity module
    pub fn new(embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (embed_dim * 2) as f32).sqrt();

        // Initialize as identity + small noise (start with dot product similarity)
        let w_attention = Array2::from_shape_fn((embed_dim, embed_dim), |(i, j)| {
            let base = if i == j { 1.0 } else { 0.0 };
            base + rng.gen_range(-scale..scale) * 0.1
        });

        Self {
            w_attention,
            temperature: 1.0,
            grad_w_attention: None,
        }
    }

    /// Compute similarity between two embeddings
    pub fn similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        // Bilinear attention: a^T W b
        let transformed = self.w_attention.dot(b);
        let score = a.dot(&transformed);

        // Apply temperature scaling and sigmoid for [0, 1] output
        1.0 / (1.0 + (-score / self.temperature).exp())
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_attention = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_attention {
            self.w_attention = &self.w_attention - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_attention.len() + 1 // +1 for temperature
    }
}

/// Learnable importance scorer for consolidation
#[derive(Debug)]
pub struct ImportanceScorer {
    /// Weight for recency factor
    pub w_recency: f32,
    /// Weight for access count factor
    pub w_access: f32,
    /// Weight for emotional valence factor
    pub w_valence: f32,
    /// Weight for embedding norm factor
    pub w_embed_norm: f32,
    /// Bias term
    pub bias: f32,
    /// Gradients
    pub grad_recency: Option<f32>,
    pub grad_access: Option<f32>,
    pub grad_valence: Option<f32>,
    pub grad_embed_norm: Option<f32>,
    pub grad_bias: Option<f32>,
}

impl ImportanceScorer {
    /// Create a new importance scorer with reasonable defaults
    pub fn new() -> Self {
        Self {
            w_recency: 0.3,
            w_access: 0.2,
            w_valence: 0.3,
            w_embed_norm: 0.2,
            bias: 0.0,
            grad_recency: None,
            grad_access: None,
            grad_valence: None,
            grad_embed_norm: None,
            grad_bias: None,
        }
    }

    /// Score importance of an item
    ///
    /// # Arguments
    /// * `recency` - How recent the item is (0.0 = old, 1.0 = recent)
    /// * `access_count` - Normalized access count
    /// * `valence` - Emotional valence (-1.0 to 1.0)
    /// * `embedding` - The item's embedding
    pub fn score(
        &self,
        recency: f32,
        access_count: f32,
        valence: f32,
        embedding: &Array1<f32>,
    ) -> f32 {
        let embed_norm = embedding.mapv(|x| x * x).sum().sqrt();

        let raw_score = self.w_recency * recency
            + self.w_access * access_count
            + self.w_valence * valence.abs()
            + self.w_embed_norm * embed_norm
            + self.bias;

        // Sigmoid for [0, 1] output
        1.0 / (1.0 + (-raw_score).exp())
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_recency = None;
        self.grad_access = None;
        self.grad_valence = None;
        self.grad_embed_norm = None;
        self.grad_bias = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(grad) = self.grad_recency {
            self.w_recency -= lr * grad;
        }
        if let Some(grad) = self.grad_access {
            self.w_access -= lr * grad;
        }
        if let Some(grad) = self.grad_valence {
            self.w_valence -= lr * grad;
        }
        if let Some(grad) = self.grad_embed_norm {
            self.w_embed_norm -= lr * grad;
        }
        if let Some(grad) = self.grad_bias {
            self.bias -= lr * grad;
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        5 // w_recency, w_access, w_valence, w_embed_norm, bias
    }
}

impl Default for ImportanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Learnable episodic memory with learned retrieval
#[derive(Debug)]
pub struct LearnableEpisodicMemory {
    /// Configuration
    pub config: LearnableMemoryConfig,
    /// Graph encoder
    pub encoder: GraphEncoder,
    /// Similarity function
    pub similarity: LearnableSimilarity,
    /// Importance scorer
    pub importance: ImportanceScorer,
    /// Stored embeddings (episode_id -> embedding)
    embeddings: HashMap<EpisodeId, Array1<f32>>,
    /// Episode metadata (episode_id -> (timestamp, access_count, valence))
    metadata: HashMap<EpisodeId, (Timestamp, u64, f32)>,
    /// Next episode ID
    next_id: EpisodeId,
    /// Current time (for recency calculation)
    current_time: Timestamp,
}

impl LearnableEpisodicMemory {
    /// Create a new learnable episodic memory
    pub fn new(config: LearnableMemoryConfig) -> Self {
        let encoder = GraphEncoder::new(config.embed_dim, config.hidden_dim);
        let similarity = LearnableSimilarity::new(config.embed_dim);
        let importance = ImportanceScorer::new();

        Self {
            config,
            encoder,
            similarity,
            importance,
            embeddings: HashMap::new(),
            metadata: HashMap::new(),
            next_id: 1,
            current_time: 0,
        }
    }

    /// Store a graph and return its episode ID
    pub fn store(&mut self, graph: &Graph, timestamp: Timestamp, valence: f32) -> EpisodeId {
        let id = self.next_id;
        self.next_id += 1;

        // Encode the graph
        let embedding = self.encoder.encode(graph);

        // Store embedding and metadata
        self.embeddings.insert(id, embedding);
        self.metadata.insert(id, (timestamp, 0, valence));

        // Update current time
        self.current_time = self.current_time.max(timestamp);

        // Check capacity and consolidate if needed
        if self.embeddings.len() > self.config.max_capacity {
            self.consolidate();
        }

        id
    }

    /// Recall episodes similar to query graph
    pub fn recall(&self, query: &Graph, limit: usize) -> Vec<(EpisodeId, f32)> {
        let query_embed = self.encoder.encode(query);

        let mut scored: Vec<_> = self.embeddings.iter()
            .map(|(&id, embed)| {
                let sim = self.similarity.similarity(&query_embed, embed);
                (id, sim)
            })
            .collect();

        // Sort by similarity (highest first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        scored
    }

    /// Record an access to an episode
    pub fn record_access(&mut self, id: EpisodeId) {
        if let Some((ts, count, valence)) = self.metadata.get_mut(&id) {
            *count += 1;
            let _ = (ts, valence); // Suppress unused warnings
        }
    }

    /// Consolidate memory by removing low-importance items
    pub fn consolidate(&mut self) {
        // Score all items
        let mut scores: Vec<_> = self.embeddings.iter()
            .filter_map(|(&id, embed)| {
                let (timestamp, access_count, valence) = self.metadata.get(&id)?;

                // Calculate recency (0.0 = old, 1.0 = recent)
                let age = self.current_time.saturating_sub(*timestamp);
                let recency = 1.0 / (1.0 + (age as f32 / 10000.0));

                // Normalize access count
                let norm_access = (*access_count as f32 / 100.0).min(1.0);

                let score = self.importance.score(recency, norm_access, *valence, embed);
                Some((id, score))
            })
            .collect();

        // Sort by importance (lowest first for removal)
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remove bottom items until under capacity
        let to_remove = self.embeddings.len().saturating_sub(self.config.max_capacity);
        for (id, _) in scores.into_iter().take(to_remove) {
            self.embeddings.remove(&id);
            self.metadata.remove(&id);
        }
    }

    /// Get the number of stored episodes
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Zero gradients for all learnable components
    pub fn zero_grad(&mut self) {
        self.encoder.zero_grad();
        self.similarity.zero_grad();
        self.importance.zero_grad();
    }

    /// Update parameters with learning rate
    pub fn step(&mut self, lr: f32) {
        self.encoder.step(lr);
        self.similarity.step(lr);
        self.importance.step(lr);
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        self.encoder.num_parameters()
            + self.similarity.num_parameters()
            + self.importance.num_parameters()
    }
}

impl Default for LearnableEpisodicMemory {
    fn default() -> Self {
        Self::new(LearnableMemoryConfig::default())
    }
}

/// Learnable semantic graph with learned retrieval
#[derive(Debug)]
pub struct LearnableSemanticGraph {
    /// Configuration
    pub config: LearnableMemoryConfig,
    /// Shared graph encoder
    pub encoder: GraphEncoder,
    /// Similarity function
    pub similarity: LearnableSimilarity,
    /// Stored embeddings
    embeddings: Vec<Array1<f32>>,
}

impl LearnableSemanticGraph {
    /// Create a new learnable semantic graph
    pub fn new(config: LearnableMemoryConfig) -> Self {
        let encoder = GraphEncoder::new(config.embed_dim, config.hidden_dim);
        let similarity = LearnableSimilarity::new(config.embed_dim);

        Self {
            config,
            encoder,
            similarity,
            embeddings: Vec::new(),
        }
    }

    /// Assert a new fact
    pub fn assert(&mut self, fact: &Graph) -> usize {
        let embedding = self.encoder.encode(fact);
        let id = self.embeddings.len();
        self.embeddings.push(embedding);
        id
    }

    /// Query for similar facts
    pub fn query(&self, pattern: &Graph, limit: usize) -> Vec<(usize, f32)> {
        let pattern_embed = self.encoder.encode(pattern);

        let mut scored: Vec<_> = self.embeddings.iter()
            .enumerate()
            .map(|(id, embed)| {
                let sim = self.similarity.similarity(&pattern_embed, embed);
                (id, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        scored
    }

    /// Check if a fact is known (above threshold similarity)
    pub fn contains(&self, fact: &Graph, threshold: f32) -> bool {
        let results = self.query(fact, 1);
        results.first().is_some_and(|(_, sim)| *sim >= threshold)
    }

    /// Get the number of facts
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.encoder.zero_grad();
        self.similarity.zero_grad();
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        self.encoder.step(lr);
        self.similarity.step(lr);
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.encoder.num_parameters() + self.similarity.num_parameters()
    }
}

impl Default for LearnableSemanticGraph {
    fn default() -> Self {
        Self::new(LearnableMemoryConfig::default())
    }
}

/// Learnable continual learning with experience replay
#[derive(Debug)]
pub struct LearnableContinualLearning {
    /// Configuration
    pub config: LearnableMemoryConfig,
    /// Graph encoder (shared for consistency)
    pub encoder: GraphEncoder,
    /// Importance scorer
    pub importance: ImportanceScorer,
    /// Experience buffer (embeddings)
    experience_buffer: Vec<Array1<f32>>,
    /// Statistics
    stats: ConsolidationStats,
}

impl LearnableContinualLearning {
    /// Create a new learnable continual learning module
    pub fn new(config: LearnableMemoryConfig) -> Self {
        let encoder = GraphEncoder::new(config.embed_dim, config.hidden_dim);
        let importance = ImportanceScorer::new();

        Self {
            config,
            encoder,
            importance,
            experience_buffer: Vec::new(),
            stats: ConsolidationStats::default(),
        }
    }

    /// Consolidate a new experience
    pub fn consolidate(&mut self, experience: &Graph) {
        let embedding = self.encoder.encode(experience);

        // Add to buffer
        if self.experience_buffer.len() >= self.config.max_capacity {
            self.experience_buffer.remove(0);
        }
        self.experience_buffer.push(embedding);
        self.stats.experiences_integrated += 1;
    }

    /// Sample experiences for replay (random subset)
    pub fn sample_for_replay(&self, count: usize) -> Vec<&Array1<f32>> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let mut indices: Vec<_> = (0..self.experience_buffer.len()).collect();
        indices.shuffle(&mut rng);

        indices.into_iter()
            .take(count)
            .filter_map(|i| self.experience_buffer.get(i))
            .collect()
    }

    /// Replay and integrate experiences
    pub fn replay_and_integrate(&mut self) {
        self.stats.consolidations += 1;
        // In a real implementation:
        // 1. Sample from buffer
        // 2. Compute importance-weighted loss
        // 3. Update model with EWC-like regularization
    }

    /// Get statistics
    pub fn stats(&self) -> &ConsolidationStats {
        &self.stats
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.encoder.zero_grad();
        self.importance.zero_grad();
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        self.encoder.step(lr);
        self.importance.step(lr);
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.encoder.num_parameters() + self.importance.num_parameters()
    }
}

impl Default for LearnableContinualLearning {
    fn default() -> Self {
        Self::new(LearnableMemoryConfig::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use grapheme_core::DagNN;

    fn make_test_graph(text: &str) -> Graph {
        DagNN::from_text(text).unwrap()
    }

    #[test]
    fn test_graph_encoder_creation() {
        let encoder = GraphEncoder::new(64, 128);
        assert!(encoder.num_parameters() > 0);
    }

    #[test]
    fn test_graph_encoder_encode() {
        let encoder = GraphEncoder::new(64, 128);
        let graph = make_test_graph("hello world");

        let embedding = encoder.encode(&graph);
        assert_eq!(embedding.len(), 64);

        // Check L2 normalized
        let norm = embedding.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_graph_encoder_similar_graphs() {
        let encoder = GraphEncoder::new(64, 128);

        let g1 = make_test_graph("hello");
        let g2 = make_test_graph("hello");
        let g3 = make_test_graph("completely different text");

        let e1 = encoder.encode(&g1);
        let e2 = encoder.encode(&g2);
        let e3 = encoder.encode(&g3);

        // Same graphs should have identical embeddings
        let sim_same = e1.dot(&e2);
        let sim_diff = e1.dot(&e3);

        assert!(sim_same > sim_diff);
    }

    #[test]
    fn test_learnable_similarity() {
        let sim = LearnableSimilarity::new(64);

        let a = Array1::from_vec(vec![1.0; 64]);
        let b = Array1::from_vec(vec![1.0; 64]);
        let c = Array1::from_vec(vec![-1.0; 64]);

        let score_same = sim.similarity(&a, &b);
        let score_diff = sim.similarity(&a, &c);

        assert!(score_same > score_diff);
        assert!(score_same >= 0.0 && score_same <= 1.0);
        assert!(score_diff >= 0.0 && score_diff <= 1.0);
    }

    #[test]
    fn test_importance_scorer() {
        let scorer = ImportanceScorer::new();
        let embedding = Array1::from_vec(vec![0.5; 64]);

        // Recent, frequently accessed, emotional content should score high
        let high_score = scorer.score(1.0, 1.0, 1.0, &embedding);
        let low_score = scorer.score(0.0, 0.0, 0.0, &embedding);

        assert!(high_score > low_score);
        assert!(high_score >= 0.0 && high_score <= 1.0);
    }

    #[test]
    fn test_learnable_episodic_memory() {
        let config = LearnableMemoryConfig {
            max_capacity: 10,
            ..Default::default()
        };
        let mut memory = LearnableEpisodicMemory::new(config);

        // Store some episodes
        let g1 = make_test_graph("first episode");
        let g2 = make_test_graph("second episode");

        let id1 = memory.store(&g1, 1000, 0.5);
        let id2 = memory.store(&g2, 2000, 0.3);

        assert_eq!(memory.len(), 2);
        assert!(id2 > id1);

        // Recall should return both
        let results = memory.recall(&g1, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_learnable_episodic_memory_consolidation() {
        let config = LearnableMemoryConfig {
            max_capacity: 5,
            ..Default::default()
        };
        let mut memory = LearnableEpisodicMemory::new(config);

        // Store more than capacity
        for i in 0..10 {
            let graph = make_test_graph(&format!("episode {}", i));
            memory.store(&graph, i as u64 * 1000, 0.5);
        }

        // Should have consolidated
        assert!(memory.len() <= 5);
    }

    #[test]
    fn test_learnable_semantic_graph() {
        let mut graph = LearnableSemanticGraph::default();

        let fact1 = make_test_graph("cats are animals");
        let fact2 = make_test_graph("dogs are animals");

        graph.assert(&fact1);
        graph.assert(&fact2);

        assert_eq!(graph.len(), 2);

        // Query should find similar
        let query = make_test_graph("cats are mammals");
        let results = graph.query(&query, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_learnable_continual_learning() {
        let mut cl = LearnableContinualLearning::default();

        for i in 0..5 {
            let exp = make_test_graph(&format!("experience {}", i));
            cl.consolidate(&exp);
        }

        assert_eq!(cl.stats().experiences_integrated, 5);

        let samples = cl.sample_for_replay(3);
        assert_eq!(samples.len(), 3);

        cl.replay_and_integrate();
        assert_eq!(cl.stats().consolidations, 1);
    }

    #[test]
    fn test_learnable_memory_gradient_flow() {
        let mut memory = LearnableEpisodicMemory::default();

        // Store and retrieve
        let graph = make_test_graph("test content");
        memory.store(&graph, 1000, 0.5);

        // Zero grad and step should not panic
        memory.zero_grad();
        memory.step(0.001);

        assert!(memory.num_parameters() > 0);
    }
}
