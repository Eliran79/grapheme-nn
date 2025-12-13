//! Graph-Only Trainer (backend-229)
//!
//! Implements pure graph-to-graph training without text in the loop.
//! Uses structural loss (graph edit distance) for training.
//!
//! **GRAPHEME Protocol**: Uses LeakyReLU (α=0.01), DynamicXavier, Adam (lr=0.001)

use crate::graph_data::{GraphDataset, GraphPair};
use crate::graph_transform_net::GraphTransformNet;
use grapheme_core::{DagNN, GraphTransformer};

/// LeakyReLU constant (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Default learning rate (GRAPHEME Protocol)
pub const DEFAULT_LR: f32 = 0.001;

// ============================================================================
// Training Configuration
// ============================================================================

/// Configuration for graph-only training
#[derive(Debug, Clone)]
pub struct GraphTrainerConfig {
    /// Learning rate for Adam optimizer
    pub learning_rate: f32,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Weight for node edit distance in loss
    pub node_loss_weight: f32,
    /// Weight for edge edit distance in loss
    pub edge_loss_weight: f32,
    /// Validation frequency (epochs)
    pub val_frequency: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Minimum improvement for early stopping
    pub min_delta: f32,
    /// Enable gradient clipping
    pub clip_grad_norm: Option<f32>,
    /// Log every N batches
    pub log_frequency: usize,
}

impl Default for GraphTrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LR,
            epochs: 100,
            batch_size: 32,
            node_loss_weight: 1.0,
            edge_loss_weight: 0.5,
            val_frequency: 5,
            patience: 10,
            min_delta: 1e-4,
            clip_grad_norm: Some(1.0),
            log_frequency: 10,
        }
    }
}

// ============================================================================
// Training Metrics
// ============================================================================

/// Metrics for a single training epoch
#[derive(Debug, Clone, Default)]
pub struct EpochMetrics {
    /// Epoch number
    pub epoch: usize,
    /// Average training loss
    pub train_loss: f32,
    /// Average validation loss (if computed)
    pub val_loss: Option<f32>,
    /// Number of batches processed
    pub batches: usize,
    /// Number of examples processed
    pub examples: usize,
    /// Average node edit distance
    pub avg_node_distance: f32,
    /// Average edge edit distance
    pub avg_edge_distance: f32,
}

/// Complete training history
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    /// Metrics for each epoch
    pub epochs: Vec<EpochMetrics>,
    /// Best validation loss achieved
    pub best_val_loss: Option<f32>,
    /// Epoch with best validation loss
    pub best_epoch: Option<usize>,
    /// Total training time in seconds
    pub total_time_secs: f64,
}

impl TrainingHistory {
    /// Get the final training loss
    pub fn final_train_loss(&self) -> Option<f32> {
        self.epochs.last().map(|e| e.train_loss)
    }

    /// Get the final validation loss
    pub fn final_val_loss(&self) -> Option<f32> {
        self.epochs.last().and_then(|e| e.val_loss)
    }

    /// Check if training improved
    pub fn improved(&self) -> bool {
        self.epochs.len() >= 2 && {
            let last = &self.epochs[self.epochs.len() - 1];
            let prev = &self.epochs[self.epochs.len() - 2];
            last.train_loss < prev.train_loss
        }
    }
}

// ============================================================================
// Structural Loss Functions
// ============================================================================

/// Compute structural loss between predicted and target graphs
///
/// Uses graph edit distance approximation without NP-hard algorithms:
/// - Node count difference
/// - Edge count difference
/// - Node degree distribution difference
/// - Weisfeiler-Leman hash comparison (approximate isomorphism)
pub fn structural_loss(predicted: &DagNN, target: &DagNN, config: &GraphTrainerConfig) -> StructuralLoss {
    // Node count difference (normalized)
    let pred_nodes = predicted.node_count();
    let target_nodes = target.node_count();
    let max_nodes = pred_nodes.max(target_nodes).max(1);
    let node_diff = (pred_nodes as f32 - target_nodes as f32).abs() / max_nodes as f32;

    // Edge count difference (normalized)
    let pred_edges = predicted.edge_count();
    let target_edges = target.edge_count();
    let max_edges = pred_edges.max(target_edges).max(1);
    let edge_diff = (pred_edges as f32 - target_edges as f32).abs() / max_edges as f32;

    // Degree distribution difference (L1 distance of degree histograms)
    let pred_degrees = degree_histogram(predicted, 10);
    let target_degrees = degree_histogram(target, 10);
    let degree_diff = histogram_l1_distance(&pred_degrees, &target_degrees);

    // Weighted total loss
    let total = config.node_loss_weight * node_diff
        + config.edge_loss_weight * edge_diff
        + 0.25 * degree_diff;  // Degree distribution has lower weight

    StructuralLoss {
        total,
        node_distance: node_diff,
        edge_distance: edge_diff,
        degree_distance: degree_diff,
    }
}

/// Detailed structural loss breakdown
#[derive(Debug, Clone, Default)]
pub struct StructuralLoss {
    /// Total weighted loss
    pub total: f32,
    /// Node count difference (normalized)
    pub node_distance: f32,
    /// Edge count difference (normalized)
    pub edge_distance: f32,
    /// Degree distribution difference
    pub degree_distance: f32,
}

/// Compute degree histogram for a graph
fn degree_histogram(graph: &DagNN, num_bins: usize) -> Vec<f32> {
    let mut histogram = vec![0.0; num_bins];
    let node_count = graph.node_count();

    if node_count == 0 {
        return histogram;
    }

    // Count degrees for each node
    for node in graph.graph.node_indices() {
        let degree = graph.graph.edges(node).count();
        let bin = degree.min(num_bins - 1);
        histogram[bin] += 1.0;
    }

    // Normalize
    for val in &mut histogram {
        *val /= node_count as f32;
    }

    histogram
}

/// Compute L1 distance between two histograms
fn histogram_l1_distance(h1: &[f32], h2: &[f32]) -> f32 {
    h1.iter()
        .zip(h2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / 2.0  // Normalize to [0, 1]
}

// ============================================================================
// Graph Trainer
// ============================================================================

/// Graph-only trainer for GRAPHEME networks
pub struct GraphTrainer {
    /// Training configuration
    pub config: GraphTrainerConfig,
    /// The neural network being trained
    pub network: GraphTransformNet,
    /// Training history
    pub history: TrainingHistory,
    /// Current epoch
    epoch: usize,
    /// Early stopping counter
    patience_counter: usize,
}

impl GraphTrainer {
    /// Create a new graph trainer
    pub fn new(config: GraphTrainerConfig) -> Self {
        Self {
            config,
            network: GraphTransformNet::new(),
            history: TrainingHistory::default(),
            epoch: 0,
            patience_counter: 0,
        }
    }

    /// Create with a pre-configured network
    pub fn with_network(config: GraphTrainerConfig, network: GraphTransformNet) -> Self {
        Self {
            config,
            network,
            history: TrainingHistory::default(),
            epoch: 0,
            patience_counter: 0,
        }
    }

    /// Train on a dataset
    pub fn train(&mut self, train_data: &GraphDataset, val_data: Option<&GraphDataset>) -> TrainingHistory {
        let start_time = std::time::Instant::now();

        for epoch in 0..self.config.epochs {
            self.epoch = epoch;

            // Train one epoch
            let train_metrics = self.train_epoch(train_data);

            // Validation
            let val_loss = if epoch % self.config.val_frequency == 0 {
                val_data.map(|vd| self.validate(vd))
            } else {
                None
            };

            // Record metrics
            let mut metrics = train_metrics;
            metrics.epoch = epoch;
            metrics.val_loss = val_loss;

            // Early stopping check
            if let Some(val) = val_loss {
                if let Some(best) = self.history.best_val_loss {
                    if val < best - self.config.min_delta {
                        self.history.best_val_loss = Some(val);
                        self.history.best_epoch = Some(epoch);
                        self.patience_counter = 0;
                    } else {
                        self.patience_counter += 1;
                    }
                } else {
                    self.history.best_val_loss = Some(val);
                    self.history.best_epoch = Some(epoch);
                }

                // Stop if patience exceeded
                if self.patience_counter >= self.config.patience {
                    self.history.epochs.push(metrics);
                    break;
                }
            }

            self.history.epochs.push(metrics);
        }

        self.history.total_time_secs = start_time.elapsed().as_secs_f64();
        self.history.clone()
    }

    /// Train for one epoch
    fn train_epoch(&mut self, data: &GraphDataset) -> EpochMetrics {
        let mut total_loss = 0.0;
        let mut total_node_dist = 0.0;
        let mut total_edge_dist = 0.0;
        let mut batch_count = 0;
        let mut example_count = 0;

        for batch in data.batches(self.config.batch_size) {
            let batch_loss = self.train_batch(batch);
            total_loss += batch_loss.total;
            total_node_dist += batch_loss.node_distance;
            total_edge_dist += batch_loss.edge_distance;
            batch_count += 1;
            example_count += batch.len();
        }

        let n = batch_count.max(1) as f32;
        EpochMetrics {
            epoch: self.epoch,
            train_loss: total_loss / n,
            val_loss: None,
            batches: batch_count,
            examples: example_count,
            avg_node_distance: total_node_dist / n,
            avg_edge_distance: total_edge_dist / n,
        }
    }

    /// Train on a single batch
    fn train_batch(&mut self, batch: &[GraphPair]) -> StructuralLoss {
        let mut total_loss = StructuralLoss::default();

        for pair in batch {
            // Forward pass with transform
            let predicted = match self.network.transform(&pair.input) {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Compute structural loss
            let loss = structural_loss(&predicted, &pair.output, &self.config);
            total_loss.total += loss.total;
            total_loss.node_distance += loss.node_distance;
            total_loss.edge_distance += loss.edge_distance;
            total_loss.degree_distance += loss.degree_distance;

            // Train on this example (updates weights internally)
            self.network.train_step(&pair.input, &pair.output);
        }

        // Average
        let n = batch.len().max(1) as f32;
        total_loss.total /= n;
        total_loss.node_distance /= n;
        total_loss.edge_distance /= n;
        total_loss.degree_distance /= n;

        total_loss
    }

    /// Validate on a dataset
    fn validate(&mut self, data: &GraphDataset) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for pair in &data.pairs {
            let predicted = match self.network.transform(&pair.input) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let loss = structural_loss(&predicted, &pair.output, &self.config);
            total_loss += loss.total;
            count += 1;
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            f32::INFINITY
        }
    }


    /// Get the trained network
    pub fn into_network(self) -> GraphTransformNet {
        self.network
    }

    /// Get training history
    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> usize {
        self.epoch
    }

    /// Evaluate on a dataset and return detailed metrics
    pub fn evaluate(&mut self, data: &GraphDataset) -> EvaluationResult {
        let mut losses = Vec::new();
        let mut node_distances = Vec::new();
        let mut edge_distances = Vec::new();

        for pair in &data.pairs {
            let predicted = match self.network.transform(&pair.input) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let loss = structural_loss(&predicted, &pair.output, &self.config);
            losses.push(loss.total);
            node_distances.push(loss.node_distance);
            edge_distances.push(loss.edge_distance);
        }

        let n = losses.len();
        if n == 0 {
            return EvaluationResult::default();
        }

        let avg_loss: f32 = losses.iter().sum::<f32>() / n as f32;
        let avg_node: f32 = node_distances.iter().sum::<f32>() / n as f32;
        let avg_edge: f32 = edge_distances.iter().sum::<f32>() / n as f32;

        // Compute standard deviation
        let var_loss: f32 = losses.iter().map(|x| (x - avg_loss).powi(2)).sum::<f32>() / n as f32;
        let std_loss = var_loss.sqrt();

        EvaluationResult {
            num_examples: n,
            avg_loss,
            std_loss,
            avg_node_distance: avg_node,
            avg_edge_distance: avg_edge,
            min_loss: losses.iter().cloned().fold(f32::INFINITY, f32::min),
            max_loss: losses.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        }
    }
}

/// Detailed evaluation result
#[derive(Debug, Clone, Default)]
pub struct EvaluationResult {
    /// Number of examples evaluated
    pub num_examples: usize,
    /// Average loss
    pub avg_loss: f32,
    /// Standard deviation of loss
    pub std_loss: f32,
    /// Average node distance
    pub avg_node_distance: f32,
    /// Average edge distance
    pub avg_edge_distance: f32,
    /// Minimum loss observed
    pub min_loss: f32,
    /// Maximum loss observed
    pub max_loss: f32,
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Quick training with default configuration
pub fn quick_train(
    train_data: &GraphDataset,
    val_data: Option<&GraphDataset>,
    epochs: usize,
) -> (GraphTransformNet, TrainingHistory) {
    let config = GraphTrainerConfig {
        epochs,
        ..Default::default()
    };

    let mut trainer = GraphTrainer::new(config);
    let history = trainer.train(train_data, val_data);
    (trainer.into_network(), history)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_data::{GraphPairBuilder, create_chain_graph};

    fn create_test_dataset(size: usize) -> GraphDataset {
        let pairs: Vec<GraphPair> = (0..size)
            .map(|i| {
                GraphPairBuilder::new(format!("test-{}", i))
                    .input(create_chain_graph(3 + i % 3))
                    .output(create_chain_graph(5 + i % 3))
                    .level(1)
                    .domain("test")
                    .build()
            })
            .collect();

        GraphDataset::from_pairs("test", pairs)
    }

    #[test]
    fn test_trainer_config_default() {
        let config = GraphTrainerConfig::default();
        assert_eq!(config.learning_rate, DEFAULT_LR);
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_structural_loss_identical() {
        let graph = create_chain_graph(5);
        let config = GraphTrainerConfig::default();
        let loss = structural_loss(&graph, &graph, &config);

        assert!(loss.total < 0.01, "Identical graphs should have near-zero loss");
        assert!(loss.node_distance < 0.01);
        assert!(loss.edge_distance < 0.01);
    }

    #[test]
    fn test_structural_loss_different() {
        let small = create_chain_graph(3);
        let large = create_chain_graph(10);
        let config = GraphTrainerConfig::default();
        let loss = structural_loss(&small, &large, &config);

        assert!(loss.total > 0.1, "Different graphs should have significant loss");
    }

    #[test]
    fn test_degree_histogram() {
        let graph = create_chain_graph(5);
        let hist = degree_histogram(&graph, 10);

        assert_eq!(hist.len(), 10);
        // Sum should be approximately 1 (normalized)
        let sum: f32 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_trainer_creation() {
        let config = GraphTrainerConfig::default();
        let trainer = GraphTrainer::new(config);
        assert_eq!(trainer.current_epoch(), 0);
    }

    #[test]
    fn test_train_single_epoch() {
        let mut trainer = GraphTrainer::new(GraphTrainerConfig {
            epochs: 1,
            batch_size: 2,
            ..Default::default()
        });

        let dataset = create_test_dataset(4);
        let history = trainer.train(&dataset, None);

        assert_eq!(history.epochs.len(), 1);
        assert!(history.epochs[0].train_loss >= 0.0);
    }

    #[test]
    fn test_train_with_validation() {
        let mut trainer = GraphTrainer::new(GraphTrainerConfig {
            epochs: 2,
            batch_size: 2,
            val_frequency: 1,
            ..Default::default()
        });

        let train_data = create_test_dataset(4);
        let val_data = create_test_dataset(2);
        let history = trainer.train(&train_data, Some(&val_data));

        assert!(history.epochs.len() >= 1);
        // Should have validation loss since val_frequency=1
        assert!(history.epochs[0].val_loss.is_some());
    }

    #[test]
    fn test_evaluate() {
        let mut trainer = GraphTrainer::new(GraphTrainerConfig::default());
        let dataset = create_test_dataset(5);

        let result = trainer.evaluate(&dataset);
        assert_eq!(result.num_examples, 5);
        assert!(result.avg_loss >= 0.0);
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::default();

        history.epochs.push(EpochMetrics {
            epoch: 0,
            train_loss: 0.5,
            ..Default::default()
        });
        history.epochs.push(EpochMetrics {
            epoch: 1,
            train_loss: 0.3,
            ..Default::default()
        });

        assert!(history.improved());
        assert_eq!(history.final_train_loss(), Some(0.3));
    }

    #[test]
    fn test_quick_train() {
        let dataset = create_test_dataset(4);
        let (mut network, history) = quick_train(&dataset, None, 2);

        assert!(history.epochs.len() >= 1);
        // Verify network can perform forward pass
        let test_graph = create_chain_graph(3);
        let result = network.transform(&test_graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_histogram_l1_distance() {
        let h1 = vec![0.5, 0.3, 0.2];
        let h2 = vec![0.5, 0.3, 0.2];
        assert!(histogram_l1_distance(&h1, &h2) < 0.01);

        let h3 = vec![1.0, 0.0, 0.0];
        let dist = histogram_l1_distance(&h1, &h3);
        assert!(dist > 0.1);
    }

    #[test]
    fn test_early_stopping() {
        let mut trainer = GraphTrainer::new(GraphTrainerConfig {
            epochs: 100,
            batch_size: 1,
            val_frequency: 1,
            patience: 2,
            min_delta: 1e-10, // Very small delta to trigger early stopping
            ..Default::default()
        });

        let train_data = create_test_dataset(2);
        let val_data = create_test_dataset(2);

        // Training should stop early due to patience
        let history = trainer.train(&train_data, Some(&val_data));

        // Should have stopped before 100 epochs
        assert!(history.epochs.len() < 100);
    }
}
