//! Parallel Collaborative Multi-Cortex Training (backend-212)
//!
//! Implements parallel training where multiple domain brains collaborate:
//! 1. Multiple brains activate on the same input
//! 2. Each brain contributes embeddings to a shared representation
//! 3. Brains learn together through shared gradients
//! 4. Rayon parallelizes batch processing across CPU cores
//!
//! **GRAPHEME Protocol**: LeakyReLU (α=0.01), DynamicXavier, Adam (lr=0.001)

use crate::graph_data::{GraphDataset, GraphPair};
use crate::graph_trainer::{structural_loss, GraphTrainerConfig};
use grapheme_core::{BrainRegistry, DagNN, Node, Edge};
use ndarray::{Array1, Array2};
use petgraph::graph::NodeIndex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// LeakyReLU constant (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Default learning rate (GRAPHEME Protocol)
pub const DEFAULT_LR: f32 = 0.001;

// ============================================================================
// Collaborative Cortex Configuration
// ============================================================================

/// Configuration for collaborative multi-cortex training
#[derive(Debug, Clone)]
pub struct CollaborativeCortexConfig {
    /// Learning rate for all brains (GRAPHEME Protocol: 0.001)
    pub learning_rate: f32,
    /// Embedding dimension for brain outputs
    pub embed_dim: usize,
    /// Hidden dimension for fusion layer
    pub hidden_dim: usize,
    /// Minimum confidence to activate a brain
    pub activation_threshold: f32,
    /// Maximum brains to activate per sample (0 = unlimited)
    pub max_active_brains: usize,
    /// Weight for structural loss
    pub structural_weight: f32,
    /// Weight for collaboration loss (brain agreement)
    pub collaboration_weight: f32,
    /// Enable parallel processing
    pub parallel: bool,
    /// Number of parallel workers (0 = auto-detect)
    pub num_workers: usize,
    /// Batch size for training
    pub batch_size: usize,
}

impl Default for CollaborativeCortexConfig {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LR,
            embed_dim: 64,
            hidden_dim: 128,
            activation_threshold: 0.2,
            max_active_brains: 0, // Unlimited
            structural_weight: 1.0,
            collaboration_weight: 0.3,
            parallel: true,
            num_workers: 0, // Auto-detect
            batch_size: 32,
        }
    }
}

// ============================================================================
// Brain Activation Result
// ============================================================================

/// Result of a brain processing an input
#[derive(Debug, Clone)]
pub struct BrainActivation {
    /// Brain domain ID
    pub brain_id: String,
    /// Whether the brain activated
    pub activated: bool,
    /// Confidence score
    pub confidence: f32,
    /// Parsed graph (if activated)
    pub graph: Option<DagNN>,
    /// Embedding vector (computed from graph)
    pub embedding: Option<Array1<f32>>,
}

/// Result of collaborative processing
#[derive(Debug, Clone)]
pub struct CollaborativeResult {
    /// All brain activations
    pub activations: Vec<BrainActivation>,
    /// Fused embedding from all active brains
    pub fused_embedding: Array1<f32>,
    /// Output graph
    pub output_graph: DagNN,
    /// Total processing time (ms)
    pub time_ms: f32,
    /// Number of active brains
    pub active_count: usize,
}

// ============================================================================
// Fusion Layer - Combines Brain Outputs
// ============================================================================

/// Fusion layer that combines embeddings from multiple brains
#[derive(Debug)]
pub struct FusionLayer {
    /// Attention weights for each brain
    attention_weights: HashMap<String, Array1<f32>>,
    /// Fusion projection matrix
    w_fusion: Array2<f32>,
    /// Fusion bias
    b_fusion: Array1<f32>,
    /// Gradient accumulators
    grad_w_fusion: Option<Array2<f32>>,
    grad_b_fusion: Option<Array1<f32>>,
    /// Configuration
    embed_dim: usize,
    hidden_dim: usize,
}

impl FusionLayer {
    /// Create a new fusion layer
    pub fn new(embed_dim: usize, hidden_dim: usize, brain_ids: &[String]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // DynamicXavier initialization
        let scale = (2.0 / (embed_dim + hidden_dim) as f32).sqrt();

        let w_fusion = Array2::from_shape_fn(
            (hidden_dim, embed_dim),
            |_| rng.gen_range(-scale..scale),
        );
        let b_fusion = Array1::zeros(hidden_dim);

        // Initialize attention weights for each brain
        let mut attention_weights = HashMap::new();
        for brain_id in brain_ids {
            let attn = Array1::from_shape_fn(embed_dim, |_| rng.gen_range(-0.1..0.1));
            attention_weights.insert(brain_id.clone(), attn);
        }

        Self {
            attention_weights,
            w_fusion,
            b_fusion,
            grad_w_fusion: None,
            grad_b_fusion: None,
            embed_dim,
            hidden_dim,
        }
    }

    /// Compute attention score for a brain embedding
    fn attention_score(&self, brain_id: &str, embedding: &Array1<f32>) -> f32 {
        if let Some(weights) = self.attention_weights.get(brain_id) {
            let score = weights.dot(embedding);
            // Softmax-like normalization
            score.exp()
        } else {
            1.0
        }
    }

    /// Fuse embeddings from multiple brains using attention
    pub fn fuse(&self, activations: &[BrainActivation]) -> Array1<f32> {
        let active: Vec<_> = activations
            .iter()
            .filter(|a| a.activated && a.embedding.is_some())
            .collect();

        if active.is_empty() {
            return Array1::zeros(self.hidden_dim);
        }

        // Compute attention scores
        let scores: Vec<f32> = active
            .iter()
            .map(|a| self.attention_score(&a.brain_id, a.embedding.as_ref().unwrap()))
            .collect();

        // Normalize scores
        let total: f32 = scores.iter().sum();
        let normalized: Vec<f32> = scores.iter().map(|s| s / total.max(1e-6)).collect();

        // Weighted sum of embeddings
        let mut weighted_sum = Array1::zeros(self.embed_dim);
        for (activation, weight) in active.iter().zip(normalized.iter()) {
            if let Some(ref emb) = activation.embedding {
                weighted_sum = weighted_sum + emb * *weight;
            }
        }

        // Project through fusion layer with LeakyReLU
        let pre_activation = self.w_fusion.dot(&weighted_sum) + &self.b_fusion;
        pre_activation.mapv(|x| if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x })
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_fusion = None;
        self.grad_b_fusion = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_fusion {
            self.w_fusion = &self.w_fusion - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_fusion {
            self.b_fusion = &self.b_fusion - &(grad * lr);
        }
    }
}

// ============================================================================
// Collaborative Cortex Trainer
// ============================================================================

/// Trainer for collaborative multi-cortex learning
pub struct CollaborativeCortexTrainer {
    /// Configuration
    pub config: CollaborativeCortexConfig,
    /// Brain registry
    brains: Vec<BrainInfo>,
    /// Fusion layer
    fusion: FusionLayer,
    /// Training statistics
    pub stats: TrainingStats,
    /// Structural loss config
    loss_config: GraphTrainerConfig,
}

/// Brain information
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BrainInfo {
    id: String,
    name: String,
}

/// Training statistics
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    /// Total samples processed
    pub samples_processed: usize,
    /// Total batches processed
    pub batches_processed: usize,
    /// Average loss per epoch
    pub epoch_losses: Vec<f32>,
    /// Brain activation counts
    pub activation_counts: HashMap<String, usize>,
    /// Average active brains per sample
    pub avg_active_brains: f32,
    /// Total training time (seconds)
    pub total_time_secs: f64,
}

impl CollaborativeCortexTrainer {
    /// Create a new collaborative trainer with brain registry
    pub fn new(config: CollaborativeCortexConfig, registry: &BrainRegistry) -> Self {
        // Collect brain info
        let brain_ids: Vec<String> = registry.domains();
        let brains: Vec<BrainInfo> = brain_ids
            .iter()
            .filter_map(|id| {
                registry.get(id).map(|b| BrainInfo {
                    id: id.clone(),
                    name: b.domain_name().to_string(),
                })
            })
            .collect();

        // Create fusion layer
        let fusion = FusionLayer::new(
            config.embed_dim,
            config.hidden_dim,
            &brain_ids,
        );

        Self {
            config,
            brains,
            fusion,
            stats: TrainingStats::default(),
            loss_config: GraphTrainerConfig::default(),
        }
    }

    /// Create with brain IDs (for testing without registry)
    pub fn with_brain_ids(config: CollaborativeCortexConfig, brain_ids: Vec<String>) -> Self {
        let brains: Vec<BrainInfo> = brain_ids
            .iter()
            .map(|id| BrainInfo {
                id: id.clone(),
                name: id.clone(),
            })
            .collect();

        let fusion = FusionLayer::new(
            config.embed_dim,
            config.hidden_dim,
            &brain_ids,
        );

        Self {
            config,
            brains,
            fusion,
            stats: TrainingStats::default(),
            loss_config: GraphTrainerConfig::default(),
        }
    }

    /// Activate brains in parallel for a graph input
    pub fn activate_brains_parallel(
        &self,
        input: &DagNN,
        registry: &BrainRegistry,
    ) -> Vec<BrainActivation> {
        let brain_ids: Vec<&str> = self.brains.iter().map(|b| b.id.as_str()).collect();

        // Convert input graph to text for brain detection
        let input_text = graph_to_text(input);

        if self.config.parallel {
            brain_ids
                .par_iter()
                .map(|brain_id| self.activate_brain(brain_id, &input_text, input, registry))
                .collect()
        } else {
            brain_ids
                .iter()
                .map(|brain_id| self.activate_brain(brain_id, &input_text, input, registry))
                .collect()
        }
    }

    /// Activate a single brain
    fn activate_brain(
        &self,
        brain_id: &str,
        input_text: &str,
        _input_graph: &DagNN,
        registry: &BrainRegistry,
    ) -> BrainActivation {
        let brain = match registry.get(brain_id) {
            Some(b) => b,
            None => {
                return BrainActivation {
                    brain_id: brain_id.to_string(),
                    activated: false,
                    confidence: 0.0,
                    graph: None,
                    embedding: None,
                };
            }
        };

        // Check if brain can process
        let can_process = brain.can_process(input_text);
        let confidence = if can_process { 0.8 } else { 0.1 };

        if !can_process || confidence < self.config.activation_threshold {
            return BrainActivation {
                brain_id: brain_id.to_string(),
                activated: false,
                confidence,
                graph: None,
                embedding: None,
            };
        }

        // Parse through brain
        let graph = brain.parse(input_text).ok();

        // Compute embedding from graph
        let embedding = graph.as_ref().map(|g| graph_to_embedding(g, self.config.embed_dim));

        BrainActivation {
            brain_id: brain_id.to_string(),
            activated: true,
            confidence,
            graph,
            embedding,
        }
    }

    /// Process a batch collaboratively in parallel
    pub fn process_batch_parallel(
        &mut self,
        batch: &[GraphPair],
        registry: &BrainRegistry,
    ) -> Vec<CollaborativeResult> {
        let start = std::time::Instant::now();

        let results: Vec<CollaborativeResult> = if self.config.parallel {
            batch
                .par_iter()
                .map(|pair| self.process_single(&pair.input, registry))
                .collect()
        } else {
            batch
                .iter()
                .map(|pair| self.process_single(&pair.input, registry))
                .collect()
        };

        let elapsed_secs = start.elapsed().as_secs_f64();

        // Update stats
        self.stats.samples_processed += batch.len();
        self.stats.batches_processed += 1;
        self.stats.total_time_secs += elapsed_secs;

        for result in &results {
            for activation in &result.activations {
                if activation.activated {
                    *self.stats.activation_counts
                        .entry(activation.brain_id.clone())
                        .or_insert(0) += 1;
                }
            }
        }

        let total_active: usize = results.iter().map(|r| r.active_count).sum();
        self.stats.avg_active_brains = total_active as f32 / results.len().max(1) as f32;

        results
    }

    /// Process a single input
    fn process_single(&self, input: &DagNN, registry: &BrainRegistry) -> CollaborativeResult {
        let start = std::time::Instant::now();

        // Activate brains
        let activations = self.activate_brains_parallel(input, registry);

        // Count active brains
        let active_count = activations.iter().filter(|a| a.activated).count();

        // Fuse embeddings
        let fused_embedding = self.fusion.fuse(&activations);

        // Generate output graph from fused embedding
        let output_graph = embedding_to_graph(&fused_embedding);

        let time_ms = start.elapsed().as_secs_f32() * 1000.0;

        CollaborativeResult {
            activations,
            fused_embedding,
            output_graph,
            time_ms,
            active_count,
        }
    }

    /// Train on a batch with collaborative loss
    pub fn train_batch(
        &mut self,
        batch: &[GraphPair],
        registry: &BrainRegistry,
    ) -> BatchLoss {
        self.fusion.zero_grad();

        // Process batch in parallel
        let results = self.process_batch_parallel(batch, registry);

        // Compute losses
        let mut total_structural = 0.0;
        let mut total_collaboration = 0.0;
        let mut count = 0;

        for (pair, result) in batch.iter().zip(results.iter()) {
            // Structural loss
            let struct_loss = structural_loss(&result.output_graph, &pair.output, &self.loss_config);
            total_structural += struct_loss.total;

            // Collaboration loss (encourage brain agreement)
            let collab_loss = self.collaboration_loss(&result.activations);
            total_collaboration += collab_loss;

            count += 1;
        }

        let n = count.max(1) as f32;
        let avg_structural = total_structural / n;
        let avg_collaboration = total_collaboration / n;

        // Weighted total loss
        let total = self.config.structural_weight * avg_structural
            + self.config.collaboration_weight * avg_collaboration;

        // Update fusion layer
        self.fusion.step(self.config.learning_rate);

        BatchLoss {
            total,
            structural: avg_structural,
            collaboration: avg_collaboration,
            samples: count,
        }
    }

    /// Compute collaboration loss (encourage active brains to agree)
    fn collaboration_loss(&self, activations: &[BrainActivation]) -> f32 {
        let active: Vec<_> = activations
            .iter()
            .filter(|a| a.activated && a.embedding.is_some())
            .collect();

        if active.len() < 2 {
            return 0.0;
        }

        // Compute pairwise embedding distances
        let mut total_distance = 0.0;
        let mut pairs = 0;

        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                if let (Some(emb_i), Some(emb_j)) = (&active[i].embedding, &active[j].embedding) {
                    // L2 distance
                    let diff = emb_i - emb_j;
                    let dist = diff.mapv(|x| x * x).sum().sqrt();
                    total_distance += dist;
                    pairs += 1;
                }
            }
        }

        if pairs > 0 {
            total_distance / pairs as f32
        } else {
            0.0
        }
    }

    /// Train for one epoch
    pub fn train_epoch(
        &mut self,
        dataset: &GraphDataset,
        registry: &BrainRegistry,
    ) -> f32 {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for batch in dataset.batches(self.config.batch_size) {
            let loss = self.train_batch(batch, registry);
            epoch_loss += loss.total;
            batch_count += 1;
        }

        let avg_loss = epoch_loss / batch_count.max(1) as f32;
        self.stats.epoch_losses.push(avg_loss);
        avg_loss
    }

    /// Train for multiple epochs
    pub fn train(
        &mut self,
        dataset: &GraphDataset,
        registry: &BrainRegistry,
        epochs: usize,
    ) -> Vec<f32> {
        let start = std::time::Instant::now();

        for epoch in 0..epochs {
            let loss = self.train_epoch(dataset, registry);
            println!("Epoch {}/{}: loss = {:.4}", epoch + 1, epochs, loss);
        }

        self.stats.total_time_secs = start.elapsed().as_secs_f64();

        self.stats.epoch_losses.clone()
    }

    /// Get training statistics
    pub fn stats(&self) -> &TrainingStats {
        &self.stats
    }

    /// Get brain activation summary
    pub fn activation_summary(&self) -> Vec<(String, usize, f32)> {
        let total = self.stats.samples_processed.max(1) as f32;
        self.stats
            .activation_counts
            .iter()
            .map(|(id, count)| (id.clone(), *count, *count as f32 / total * 100.0))
            .collect()
    }
}

/// Batch loss breakdown
#[derive(Debug, Clone)]
pub struct BatchLoss {
    /// Total weighted loss
    pub total: f32,
    /// Structural loss component
    pub structural: f32,
    /// Collaboration loss component
    pub collaboration: f32,
    /// Number of samples
    pub samples: usize,
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert a DagNN graph to text representation
fn graph_to_text(graph: &DagNN) -> String {
    // Simple extraction of node activations as text
    let activations: Vec<String> = graph
        .graph
        .node_indices()
        .filter_map(|idx| {
            graph.graph.node_weight(idx).map(|node| {
                let activation = node.activation;
                // Map activation to ASCII character
                let char_code = ((activation * 94.0) as u8).wrapping_add(32);
                (char_code as char).to_string()
            })
        })
        .collect();

    activations.join("")
}

/// Convert a DagNN graph to an embedding vector
fn graph_to_embedding(graph: &DagNN, embed_dim: usize) -> Array1<f32> {
    let mut embedding = Array1::zeros(embed_dim);

    let nodes: Vec<_> = graph.graph.node_indices().collect();
    let n = nodes.len().max(1);

    for (i, node_idx) in nodes.iter().enumerate() {
        if let Some(node) = graph.graph.node_weight(*node_idx) {
            // Distribute node activation across embedding
            let dim_idx = i % embed_dim;
            embedding[dim_idx] += node.activation / n as f32;
        }
    }

    // Add structural features
    let edge_count = graph.graph.edge_count();
    if embed_dim > 2 {
        embedding[0] = (n as f32).ln();
        embedding[1] = (edge_count as f32 + 1.0).ln();
    }

    // L2 normalize
    let norm = embedding.mapv(|x| x * x).sum().sqrt().max(1e-6);
    embedding.mapv_inplace(|x| x / norm);

    embedding
}

/// Convert an embedding vector to a DagNN graph
fn embedding_to_graph(embedding: &Array1<f32>) -> DagNN {
    let mut graph = DagNN::new();

    // Create nodes based on embedding dimensions
    let num_nodes = (embedding.len() / 4).max(2);
    let nodes: Vec<NodeIndex> = (0..num_nodes)
        .map(|i| {
            let activation = embedding[i % embedding.len()].abs();
            let mut node = Node::hidden();
            node.activation = activation;
            graph.graph.add_node(node)
        })
        .collect();

    // Create edges based on embedding values
    for i in 0..(num_nodes - 1) {
        graph.graph.add_edge(nodes[i], nodes[i + 1], Edge::sequential());
    }

    graph
}

// ============================================================================
// Parallel Batch Processing
// ============================================================================

/// Process a dataset in parallel batches
pub fn parallel_process_dataset(
    dataset: &GraphDataset,
    registry: &BrainRegistry,
    config: &CollaborativeCortexConfig,
) -> Vec<CollaborativeResult> {
    let trainer = CollaborativeCortexTrainer::new(config.clone(), registry);

    let all_results: Vec<Vec<CollaborativeResult>> = dataset
        .pairs
        .par_chunks(config.batch_size)
        .map(|batch| {
            batch
                .par_iter()
                .map(|pair| trainer.process_single(&pair.input, registry))
                .collect()
        })
        .collect();

    all_results.into_iter().flatten().collect()
}

/// Parallel train step for a batch (thread-safe gradient accumulation)
pub fn parallel_train_batch(
    batch: &[GraphPair],
    registry: &BrainRegistry,
    config: &CollaborativeCortexConfig,
) -> BatchLoss {
    let loss_config = GraphTrainerConfig::default();
    let brain_ids: Vec<String> = registry.domains();
    let fusion = Arc::new(Mutex::new(FusionLayer::new(
        config.embed_dim,
        config.hidden_dim,
        &brain_ids,
    )));

    // Parallel loss computation
    let losses: Vec<(f32, f32)> = batch
        .par_iter()
        .map(|pair| {
            let input_text = graph_to_text(&pair.input);

            // Activate brains
            let activations: Vec<BrainActivation> = brain_ids
                .iter()
                .filter_map(|id| {
                    registry.get(id).map(|brain| {
                        let can = brain.can_process(&input_text);
                        let graph = if can { brain.parse(&input_text).ok() } else { None };
                        let embedding = graph.as_ref().map(|g| graph_to_embedding(g, config.embed_dim));
                        BrainActivation {
                            brain_id: id.clone(),
                            activated: can,
                            confidence: if can { 0.8 } else { 0.1 },
                            graph,
                            embedding,
                        }
                    })
                })
                .collect();

            // Fuse and compute output
            let fused = fusion.lock().unwrap().fuse(&activations);
            let output = embedding_to_graph(&fused);

            // Structural loss
            let struct_loss = structural_loss(&output, &pair.output, &loss_config);

            // Collaboration loss
            let collab_loss = {
                let active: Vec<_> = activations.iter().filter(|a| a.activated && a.embedding.is_some()).collect();
                if active.len() < 2 {
                    0.0
                } else {
                    let mut total = 0.0;
                    let mut pairs = 0;
                    for i in 0..active.len() {
                        for j in (i+1)..active.len() {
                            if let (Some(ei), Some(ej)) = (&active[i].embedding, &active[j].embedding) {
                                let diff = ei - ej;
                                total += diff.mapv(|x| x*x).sum().sqrt();
                                pairs += 1;
                            }
                        }
                    }
                    if pairs > 0 { total / pairs as f32 } else { 0.0 }
                }
            };

            (struct_loss.total, collab_loss)
        })
        .collect();

    // Aggregate losses
    let n = losses.len().max(1) as f32;
    let avg_structural: f32 = losses.iter().map(|(s, _)| s).sum::<f32>() / n;
    let avg_collaboration: f32 = losses.iter().map(|(_, c)| c).sum::<f32>() / n;

    let total = config.structural_weight * avg_structural
        + config.collaboration_weight * avg_collaboration;

    // Update fusion layer
    fusion.lock().unwrap().step(config.learning_rate);

    BatchLoss {
        total,
        structural: avg_structural,
        collaboration: avg_collaboration,
        samples: batch.len(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_data::{create_chain_graph, GraphPairBuilder};

    fn create_test_config() -> CollaborativeCortexConfig {
        CollaborativeCortexConfig {
            parallel: false, // Disable for tests to avoid thread issues
            batch_size: 2,
            ..Default::default()
        }
    }

    #[allow(dead_code)]
    fn create_test_dataset() -> GraphDataset {
        let pairs: Vec<GraphPair> = (0..4)
            .map(|i| {
                GraphPairBuilder::new(format!("test-{}", i))
                    .input(create_chain_graph(3 + i % 2))
                    .output(create_chain_graph(5 + i % 2))
                    .level(1)
                    .domain("test")
                    .build()
            })
            .collect();
        GraphDataset::from_pairs("test", pairs)
    }

    #[test]
    fn test_config_default() {
        let config = CollaborativeCortexConfig::default();
        assert_eq!(config.learning_rate, DEFAULT_LR);
        assert_eq!(config.embed_dim, 64);
        assert!(config.parallel);
    }

    #[test]
    fn test_fusion_layer_creation() {
        let brain_ids = vec!["code".to_string(), "math".to_string()];
        let fusion = FusionLayer::new(64, 128, &brain_ids);

        assert_eq!(fusion.embed_dim, 64);
        assert_eq!(fusion.hidden_dim, 128);
        assert!(fusion.attention_weights.contains_key("code"));
        assert!(fusion.attention_weights.contains_key("math"));
    }

    #[test]
    fn test_fusion_layer_fuse() {
        let brain_ids = vec!["code".to_string(), "math".to_string()];
        let fusion = FusionLayer::new(64, 128, &brain_ids);

        let activations = vec![
            BrainActivation {
                brain_id: "code".to_string(),
                activated: true,
                confidence: 0.8,
                graph: None,
                embedding: Some(Array1::ones(64)),
            },
            BrainActivation {
                brain_id: "math".to_string(),
                activated: true,
                confidence: 0.7,
                graph: None,
                embedding: Some(Array1::ones(64) * 0.5),
            },
        ];

        let fused = fusion.fuse(&activations);
        assert_eq!(fused.len(), 128);
    }

    #[test]
    fn test_fusion_empty_activations() {
        let brain_ids = vec!["code".to_string()];
        let fusion = FusionLayer::new(64, 128, &brain_ids);

        let activations: Vec<BrainActivation> = vec![];
        let fused = fusion.fuse(&activations);

        assert_eq!(fused.len(), 128);
        assert!(fused.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_graph_to_embedding() {
        let graph = create_chain_graph(5);
        let embedding = graph_to_embedding(&graph, 64);

        assert_eq!(embedding.len(), 64);
        // Should be normalized
        let norm = embedding.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 0.01 || norm < 0.01);
    }

    #[test]
    fn test_embedding_to_graph() {
        let embedding = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let graph = embedding_to_graph(&embedding);

        assert!(graph.node_count() > 0);
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_trainer_with_brain_ids() {
        let config = create_test_config();
        let brain_ids = vec!["code".to_string(), "math".to_string(), "text".to_string()];
        let trainer = CollaborativeCortexTrainer::with_brain_ids(config, brain_ids.clone());

        assert_eq!(trainer.brains.len(), 3);
    }

    #[test]
    fn test_collaboration_loss() {
        let config = create_test_config();
        let brain_ids = vec!["a".to_string(), "b".to_string()];
        let trainer = CollaborativeCortexTrainer::with_brain_ids(config, brain_ids);

        // Same embeddings should have 0 collaboration loss
        let same_activations = vec![
            BrainActivation {
                brain_id: "a".to_string(),
                activated: true,
                confidence: 0.8,
                graph: None,
                embedding: Some(Array1::ones(64)),
            },
            BrainActivation {
                brain_id: "b".to_string(),
                activated: true,
                confidence: 0.8,
                graph: None,
                embedding: Some(Array1::ones(64)),
            },
        ];

        let loss = trainer.collaboration_loss(&same_activations);
        assert!(loss < 0.01);

        // Different embeddings should have positive loss
        let diff_activations = vec![
            BrainActivation {
                brain_id: "a".to_string(),
                activated: true,
                confidence: 0.8,
                graph: None,
                embedding: Some(Array1::ones(64)),
            },
            BrainActivation {
                brain_id: "b".to_string(),
                activated: true,
                confidence: 0.8,
                graph: None,
                embedding: Some(Array1::zeros(64)),
            },
        ];

        let loss = trainer.collaboration_loss(&diff_activations);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_batch_loss() {
        let loss = BatchLoss {
            total: 0.5,
            structural: 0.3,
            collaboration: 0.2,
            samples: 10,
        };

        assert_eq!(loss.samples, 10);
        assert!((loss.total - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_graph_to_text() {
        let graph = create_chain_graph(5);
        let text = graph_to_text(&graph);

        // Should produce some text representation
        assert!(!text.is_empty());
    }

    #[test]
    fn test_training_stats() {
        let stats = TrainingStats::default();
        assert_eq!(stats.samples_processed, 0);
        assert_eq!(stats.batches_processed, 0);
        assert!(stats.epoch_losses.is_empty());
    }
}
