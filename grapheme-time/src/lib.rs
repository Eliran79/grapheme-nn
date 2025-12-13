//! Time Series Brain: Temporal-to-graph embedding for GRAPHEME forecasting
//!
//! This crate implements a cognitive brain for time series data, converting
//! sequential temporal data into graph representations suitable for GRAPHEME's
//! neuromorphic learning.
//!
//! # Architecture
//!
//! Time series data is encoded as a DAG where:
//! - Each timestep becomes a node
//! - Sequential edges connect t → t+1 (temporal causality)
//! - Optional skip connections for long-term dependencies
//! - Output node aggregates for regression predictions
//!
//! # Example
//!
//! ```ignore
//! use grapheme_time::{TimeBrain, TimeSeriesConfig};
//!
//! let config = TimeSeriesConfig::default().with_window_size(10);
//! let brain = TimeBrain::new(config);
//!
//! // Convert time series window to graph
//! let window = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
//! let graph = brain.to_graph(&window)?;
//! ```

use grapheme_core::{
    DagNN, DomainBrain, DomainError, DomainExample, DomainResult, DomainRule, Edge, EdgeType,
    ExecutionResult, Node, NodeId, NodeType, ValidationIssue, ValidationSeverity,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error types for time series operations
#[derive(Error, Debug)]
pub enum TimeSeriesError {
    #[error("Window size must be at least 2, got {0}")]
    WindowTooSmall(usize),

    #[error("Input sequence too short: need at least {needed} values, got {got}")]
    SequenceTooShort { needed: usize, got: usize },

    #[error("Empty time series")]
    EmptySequence,

    #[error("Invalid value at index {index}: {reason}")]
    InvalidValue { index: usize, reason: String },

    #[error("Normalization failed: {0}")]
    NormalizationError(String),
}

/// Result type for time series operations
pub type TimeResult<T> = Result<T, TimeSeriesError>;

/// Configuration for time series encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesConfig {
    /// Window size (number of past timesteps to use for prediction)
    pub window_size: usize,

    /// Whether to add skip connections for long-term dependencies
    pub use_skip_connections: bool,

    /// Skip connection interval (e.g., 3 means connect every 3rd timestep)
    pub skip_interval: usize,

    /// Normalization method
    pub normalization: NormalizationMethod,

    /// Whether to add hidden nodes between input and output
    pub hidden_nodes: usize,
}

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            use_skip_connections: true,
            skip_interval: 3,
            normalization: NormalizationMethod::MinMax,
            hidden_nodes: 4,
        }
    }
}

impl TimeSeriesConfig {
    /// Create config with specified window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Enable/disable skip connections
    pub fn with_skip_connections(mut self, enable: bool) -> Self {
        self.use_skip_connections = enable;
        self
    }

    /// Set skip connection interval
    pub fn with_skip_interval(mut self, interval: usize) -> Self {
        self.skip_interval = interval;
        self
    }

    /// Set normalization method
    pub fn with_normalization(mut self, method: NormalizationMethod) -> Self {
        self.normalization = method;
        self
    }

    /// Set number of hidden nodes
    pub fn with_hidden_nodes(mut self, count: usize) -> Self {
        self.hidden_nodes = count;
        self
    }
}

/// Normalization methods for time series values
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Min-max scaling to [0, 1]
    MinMax,
    /// Z-score standardization
    ZScore,
    /// No normalization
    None,
}

/// Normalization parameters for reversing predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub method: NormalizationMethod,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
}

impl Default for NormalizationParams {
    fn default() -> Self {
        Self {
            method: NormalizationMethod::None,
            min: 0.0,
            max: 1.0,
            mean: 0.0,
            std: 1.0,
        }
    }
}

impl NormalizationParams {
    /// Compute normalization parameters from data
    pub fn from_data(data: &[f32], method: NormalizationMethod) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt().max(1e-8); // Avoid division by zero

        Self {
            method,
            min,
            max,
            mean,
            std,
        }
    }

    /// Normalize a single value
    pub fn normalize(&self, value: f32) -> f32 {
        match self.method {
            NormalizationMethod::MinMax => {
                let range = (self.max - self.min).max(1e-8);
                (value - self.min) / range
            }
            NormalizationMethod::ZScore => (value - self.mean) / self.std,
            NormalizationMethod::None => value,
        }
    }

    /// Denormalize a single value (reverse normalization)
    pub fn denormalize(&self, normalized: f32) -> f32 {
        match self.method {
            NormalizationMethod::MinMax => {
                let range = self.max - self.min;
                normalized * range + self.min
            }
            NormalizationMethod::ZScore => normalized * self.std + self.mean,
            NormalizationMethod::None => normalized,
        }
    }

    /// Normalize a slice of values
    pub fn normalize_slice(&self, values: &[f32]) -> Vec<f32> {
        values.iter().map(|&v| self.normalize(v)).collect()
    }
}

/// Time Series Brain: converts temporal sequences to graphs
#[derive(Clone)]
pub struct TimeBrain {
    config: TimeSeriesConfig,
    norm_params: Option<NormalizationParams>,
}

impl TimeBrain {
    /// Create a new TimeBrain with the given configuration
    pub fn new(config: TimeSeriesConfig) -> Self {
        Self {
            config,
            norm_params: None,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(TimeSeriesConfig::default())
    }

    /// Get configuration
    pub fn config(&self) -> &TimeSeriesConfig {
        &self.config
    }

    /// Set normalization parameters (computed from training data)
    pub fn set_normalization(&mut self, params: NormalizationParams) {
        self.norm_params = Some(params);
    }

    /// Get normalization parameters
    pub fn normalization(&self) -> Option<&NormalizationParams> {
        self.norm_params.as_ref()
    }

    /// Compute and set normalization parameters from training data
    pub fn fit_normalization(&mut self, data: &[f32]) {
        let params = NormalizationParams::from_data(data, self.config.normalization);
        self.norm_params = Some(params);
    }

    /// Convert a time series window to a graph
    ///
    /// The graph structure:
    /// - Window nodes: w_0 → w_1 → ... → w_{n-1} (sequential edges)
    /// - Skip connections: w_i → w_{i+skip} for long-term dependencies
    /// - Hidden nodes: aggregate from window nodes
    /// - Output node: prediction target
    pub fn to_graph(&self, window: &[f32]) -> TimeResult<DagNN> {
        if window.is_empty() {
            return Err(TimeSeriesError::EmptySequence);
        }

        if window.len() < 2 {
            return Err(TimeSeriesError::WindowTooSmall(window.len()));
        }

        // Normalize values
        let normalized = match &self.norm_params {
            Some(params) => params.normalize_slice(window),
            None => {
                // Auto-normalize based on window
                let params = NormalizationParams::from_data(window, self.config.normalization);
                params.normalize_slice(window)
            }
        };

        // Create DAG
        let mut dag = DagNN::new();

        // Add input nodes for each timestep using the DagNN's add_character method
        // which properly registers them in input_nodes_set
        let mut input_nodes: Vec<NodeId> = Vec::with_capacity(normalized.len());
        for (i, &value) in normalized.iter().enumerate() {
            // Use 't' as placeholder character for all timesteps, position encodes the index
            // add_character already creates sequential edges between consecutive nodes
            let node_id = dag.add_character('t', i);
            dag.graph[node_id].activation = value;
            input_nodes.push(node_id);
        }

        // Add skip connections for long-term dependencies
        if self.config.use_skip_connections && self.config.skip_interval > 1 {
            for i in 0..input_nodes.len() {
                let skip_target = i + self.config.skip_interval;
                if skip_target < input_nodes.len() {
                    // Weight skip connections lower initially
                    dag.graph.add_edge(
                        input_nodes[i],
                        input_nodes[skip_target],
                        Edge::new(0.5, EdgeType::Semantic),
                    );
                }
            }
        }

        // Add hidden nodes
        let mut hidden_nodes: Vec<NodeId> = Vec::with_capacity(self.config.hidden_nodes);
        for h in 0..self.config.hidden_nodes {
            let hidden_id = dag.graph.add_node(Node::hidden());
            hidden_nodes.push(hidden_id);

            // Connect last few input nodes to hidden nodes
            let start = input_nodes.len().saturating_sub(4);
            for &input_id in &input_nodes[start..] {
                dag.graph
                    .add_edge(input_id, hidden_id, Edge::new(0.25, EdgeType::Semantic));
            }

            // Also connect from earlier inputs with skip connections
            if h < input_nodes.len() / 2 {
                dag.graph.add_edge(
                    input_nodes[h],
                    hidden_id,
                    Edge::new(0.1, EdgeType::Semantic),
                );
            }
        }

        // Add output node for regression
        let output_id = dag.graph.add_node(Node::output());
        dag.add_output_node(output_id);

        // Connect hidden nodes to output
        for &hidden_id in &hidden_nodes {
            dag.graph
                .add_edge(hidden_id, output_id, Edge::new(0.25, EdgeType::Semantic));
        }

        // Also connect last input directly to output
        if let Some(&last_input) = input_nodes.last() {
            dag.graph
                .add_edge(last_input, output_id, Edge::new(0.5, EdgeType::Semantic));
        }

        // Update topology
        if let Err(e) = dag.update_topology() {
            return Err(TimeSeriesError::NormalizationError(format!(
                "Failed to update topology: {}",
                e
            )));
        }

        Ok(dag)
    }

    /// Forward pass: run neuromorphic forward and get prediction
    pub fn predict(&self, dag: &mut DagNN) -> TimeResult<f32> {
        // Run forward pass
        if let Err(e) = dag.neuromorphic_forward() {
            return Err(TimeSeriesError::NormalizationError(format!(
                "Forward pass failed: {}",
                e
            )));
        }

        // Find output node and get its activation
        let output_nodes = dag.output_nodes();
        if output_nodes.is_empty() {
            return Err(TimeSeriesError::NormalizationError(
                "No output nodes found".to_string(),
            ));
        }

        let output_activation = dag.graph[output_nodes[0]].activation;

        // Denormalize prediction
        let prediction = match &self.norm_params {
            Some(params) => params.denormalize(output_activation),
            None => output_activation,
        };

        Ok(prediction)
    }

    /// Compute MSE loss between prediction and target
    pub fn mse_loss(prediction: f32, target: f32) -> f32 {
        (prediction - target).powi(2)
    }

    /// Compute MSE gradient: d(loss)/d(prediction) = 2 * (prediction - target)
    pub fn mse_gradient(prediction: f32, target: f32) -> f32 {
        2.0 * (prediction - target)
    }
}

impl DomainBrain for TimeBrain {
    fn domain_id(&self) -> &str {
        "time"
    }

    fn domain_name(&self) -> &str {
        "Time Series"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn can_process(&self, input: &str) -> bool {
        // Can process comma-separated float values
        input
            .split(',')
            .all(|s| s.trim().parse::<f32>().is_ok())
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        // Parse comma-separated values
        let values: Result<Vec<f32>, _> = input.split(',').map(|s| s.trim().parse::<f32>()).collect();

        match values {
            Ok(v) if v.len() >= 2 => self.to_graph(&v).map_err(|e| DomainError::InvalidInput(e.to_string())),
            Ok(_) => Err(DomainError::InvalidInput(
                "Need at least 2 values".to_string(),
            )),
            Err(e) => Err(DomainError::InvalidInput(format!(
                "Failed to parse values: {}",
                e
            ))),
        }
    }

    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Time series graphs are already in core format
        Ok(graph.clone())
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Time series graphs are already in core format
        Ok(graph.clone())
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        if graph.node_count() < 3 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Time series graph has very few nodes".to_string(),
                location: None,
            });
        }

        if graph.edge_count() == 0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                message: "Time series graph has no edges".to_string(),
                location: None,
            });
        }

        Ok(issues)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        // Execute forward pass and return prediction
        let mut dag = graph.clone();
        if let Err(e) = dag.neuromorphic_forward() {
            return Ok(ExecutionResult::Error(format!("Forward pass failed: {}", e)));
        }

        let output_nodes = dag.output_nodes();
        if output_nodes.is_empty() {
            return Ok(ExecutionResult::Error("No output nodes".to_string()));
        }

        let prediction = dag.graph[output_nodes[0]].activation;
        Ok(ExecutionResult::Numeric(prediction as f64))
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        // No specific transformation rules for time series
        vec![]
    }

    fn transform(&self, graph: &DagNN, _rule_id: usize) -> DomainResult<DagNN> {
        // No transformations - return as-is
        Ok(graph.clone())
    }

    fn generate_examples(&self, count: usize) -> Vec<DomainExample> {
        // Generate synthetic sine wave examples
        let sine = generate_sine_wave(count + self.config.window_size, 0.1, 1.0, 0.0);
        let mut examples = Vec::with_capacity(count);

        for i in 0..count.min(sine.len().saturating_sub(self.config.window_size)) {
            let window = &sine[i..i + self.config.window_size];
            let target_val = sine.get(i + self.config.window_size).copied().unwrap_or(0.0);

            // Create input graph from window
            let input_dag = match self.to_graph(window) {
                Ok(dag) => dag,
                Err(_) => continue,
            };

            // Store input and output as serialized JSON
            let input_json = serde_json::to_string(&input_dag).unwrap_or_default();
            let output_json = format!("{}", target_val);

            examples.push(DomainExample::new(input_json, output_json)
                .with_metadata("domain", "time")
                .with_metadata("difficulty", "1"));
        }

        examples
    }

    fn input_node_count(&self) -> usize {
        self.config.window_size
    }

    fn output_node_count(&self) -> usize {
        1 // Single prediction output
    }

    /// Returns all semantic node types that TimeBrain can produce.
    ///
    /// Time series data uses Feature nodes for numerical values rather than
    /// character-based Input nodes. Each feature position in the window
    /// gets its own Feature(index) node type.
    fn node_types(&self) -> Vec<NodeType> {
        let mut types = Vec::new();

        // Feature nodes for window positions
        for i in 0..self.config.window_size {
            types.push(NodeType::Feature(i));
        }

        // Output node for prediction
        types.push(NodeType::Output);

        // Also include digits and decimal for text representations
        for c in '0'..='9' {
            types.push(NodeType::Input(c));
        }
        types.push(NodeType::Input('.'));
        types.push(NodeType::Input('-'));
        types.push(NodeType::Input(','));
        types.push(NodeType::Input(' '));

        types
    }
}

// Required for DomainBrain trait
impl std::fmt::Debug for TimeBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimeBrain")
            .field("config", &self.config)
            .field("norm_params", &self.norm_params)
            .finish()
    }
}

/// Generate synthetic sine wave data for testing
pub fn generate_sine_wave(length: usize, frequency: f32, amplitude: f32, offset: f32) -> Vec<f32> {
    (0..length)
        .map(|t| offset + amplitude * (frequency * t as f32).sin())
        .collect()
}

/// Generate windowed training pairs from time series
/// Returns (input_windows, targets) where each input window predicts the next value
pub fn create_training_pairs(
    series: &[f32],
    window_size: usize,
) -> TimeResult<(Vec<Vec<f32>>, Vec<f32>)> {
    if series.len() < window_size + 1 {
        return Err(TimeSeriesError::SequenceTooShort {
            needed: window_size + 1,
            got: series.len(),
        });
    }

    let mut windows = Vec::new();
    let mut targets = Vec::new();

    for i in 0..series.len() - window_size {
        let window = series[i..i + window_size].to_vec();
        let target = series[i + window_size];
        windows.push(window);
        targets.push(target);
    }

    Ok((windows, targets))
}

/// Training result for time series forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub epoch: usize,
    pub train_mse: f32,
    pub test_mse: Option<f32>,
    pub predictions_sample: Vec<(f32, f32)>, // (predicted, actual)
}

/// Simple time series trainer using gradient descent
///
/// This trainer maintains a template graph with learned edge weights that
/// are applied to all new graphs created during training and inference.
pub struct TimeSeriesTrainer {
    brain: TimeBrain,
    learning_rate: f32,
    /// Template graph with learned edge weights
    /// The edge weights from this graph are used to initialize new graphs
    template_weights: Vec<f32>,
}

impl TimeSeriesTrainer {
    pub fn new(brain: TimeBrain, learning_rate: f32) -> Self {
        // Initialize template weights based on graph structure
        // We'll create a dummy graph to get the edge count
        let window: Vec<f32> = (0..brain.config().window_size)
            .map(|i| i as f32 / brain.config().window_size as f32)
            .collect();

        let edge_count = if let Ok(dag) = brain.to_graph(&window) {
            dag.edge_count()
        } else {
            0
        };

        // Initialize weights uniformly
        let template_weights = vec![0.5_f32; edge_count];

        Self {
            brain,
            learning_rate,
            template_weights,
        }
    }

    /// Apply learned weights to a new graph
    fn apply_template_weights(&self, dag: &mut DagNN) {
        for (i, edge_idx) in dag.graph.edge_indices().enumerate() {
            if i < self.template_weights.len() {
                dag.graph[edge_idx].weight = self.template_weights[i];
            }
        }
    }

    /// Train on a single window-target pair
    pub fn train_step(&mut self, window: &[f32], target: f32) -> TimeResult<f32> {
        // Create graph from window
        let mut dag = self.brain.to_graph(window)?;

        // Apply learned weights
        self.apply_template_weights(&mut dag);

        // Forward pass
        let prediction = self.brain.predict(&mut dag)?;

        // Compute loss and gradient
        let loss = TimeBrain::mse_loss(prediction, target);
        let grad = TimeBrain::mse_gradient(prediction, target);

        // Backward pass: update template weights using gradient descent
        let output_nodes = dag.output_nodes();
        if !output_nodes.is_empty() {
            let output_id = output_nodes[0];

            // Update edges pointing to output (and all edges proportionally)
            for (i, edge_idx) in dag.graph.edge_indices().enumerate() {
                let Some((source, target_node)) = dag.graph.edge_endpoints(edge_idx) else {
                    continue;
                };

                if i < self.template_weights.len() {
                    // Compute weight update
                    let source_act = dag.graph[source].activation;

                    // Edges to output get full gradient, others get scaled gradient
                    let scale = if target_node == output_id {
                        1.0
                    } else {
                        // Propagate gradient to earlier layers (simplified backprop)
                        0.1
                    };

                    let delta = self.learning_rate * grad * source_act * scale;
                    self.template_weights[i] -= delta;
                    self.template_weights[i] = self.template_weights[i].clamp(-5.0, 5.0);
                }
            }
        }

        Ok(loss)
    }

    /// Get the brain
    pub fn brain(&self) -> &TimeBrain {
        &self.brain
    }

    /// Get mutable brain
    pub fn brain_mut(&mut self) -> &mut TimeBrain {
        &mut self.brain
    }

    /// Create a graph with learned weights for inference
    pub fn create_inference_graph(&self, window: &[f32]) -> TimeResult<DagNN> {
        let mut dag = self.brain.to_graph(window)?;
        self.apply_template_weights(&mut dag);
        Ok(dag)
    }

    /// Make a prediction using learned weights
    pub fn predict(&self, window: &[f32]) -> TimeResult<f32> {
        let mut dag = self.create_inference_graph(window)?;
        self.brain.predict(&mut dag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_brain_creation() {
        let brain = TimeBrain::default_config();
        assert_eq!(brain.config().window_size, 10);
        assert!(brain.config().use_skip_connections);
    }

    #[test]
    fn test_config_builder() {
        let config = TimeSeriesConfig::default()
            .with_window_size(20)
            .with_skip_connections(false)
            .with_hidden_nodes(8);

        assert_eq!(config.window_size, 20);
        assert!(!config.use_skip_connections);
        assert_eq!(config.hidden_nodes, 8);
    }

    #[test]
    fn test_normalization_minmax() {
        let data = vec![0.0, 5.0, 10.0];
        let params = NormalizationParams::from_data(&data, NormalizationMethod::MinMax);

        assert!((params.normalize(0.0) - 0.0).abs() < 1e-6);
        assert!((params.normalize(5.0) - 0.5).abs() < 1e-6);
        assert!((params.normalize(10.0) - 1.0).abs() < 1e-6);

        // Test denormalization
        assert!((params.denormalize(0.0) - 0.0).abs() < 1e-6);
        assert!((params.denormalize(0.5) - 5.0).abs() < 1e-6);
        assert!((params.denormalize(1.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalization_zscore() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = NormalizationParams::from_data(&data, NormalizationMethod::ZScore);

        // Mean should be 3.0, normalized value of 3.0 should be 0.0
        assert!((params.normalize(3.0) - 0.0).abs() < 1e-6);
        assert!((params.denormalize(0.0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_graph_basic() {
        let brain = TimeBrain::default_config();
        let window = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let graph = brain.to_graph(&window).unwrap();

        // Should have input nodes + hidden nodes + output node
        assert!(graph.node_count() >= window.len());
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_to_graph_with_skip_connections() {
        let config = TimeSeriesConfig::default()
            .with_window_size(10)
            .with_skip_connections(true)
            .with_skip_interval(2);
        let brain = TimeBrain::new(config);

        let window: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let graph = brain.to_graph(&window).unwrap();

        // Should have more edges due to skip connections
        assert!(graph.edge_count() > 10);
    }

    #[test]
    fn test_to_graph_error_empty() {
        let brain = TimeBrain::default_config();
        let result = brain.to_graph(&[]);
        assert!(matches!(result, Err(TimeSeriesError::EmptySequence)));
    }

    #[test]
    fn test_to_graph_error_too_small() {
        let brain = TimeBrain::default_config();
        let result = brain.to_graph(&[0.5]);
        assert!(matches!(result, Err(TimeSeriesError::WindowTooSmall(_))));
    }

    #[test]
    fn test_generate_sine_wave() {
        let wave = generate_sine_wave(100, 0.1, 1.0, 0.0);
        assert_eq!(wave.len(), 100);

        // Check values are in expected range
        for &v in &wave {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_create_training_pairs() {
        let series: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let (windows, targets) = create_training_pairs(&series, 5).unwrap();

        assert_eq!(windows.len(), 15); // 20 - 5 = 15 pairs
        assert_eq!(targets.len(), 15);

        // First window should be [0, 1, 2, 3, 4], target should be 5
        assert_eq!(windows[0], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!((targets[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss() {
        assert!((TimeBrain::mse_loss(2.0, 3.0) - 1.0).abs() < 1e-6);
        assert!((TimeBrain::mse_loss(5.0, 5.0) - 0.0).abs() < 1e-6);
        assert!((TimeBrain::mse_loss(0.0, 2.0) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_gradient() {
        // Gradient should be 2 * (pred - target)
        assert!((TimeBrain::mse_gradient(3.0, 2.0) - 2.0).abs() < 1e-6);
        assert!((TimeBrain::mse_gradient(2.0, 3.0) + 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_predict_forward_pass() {
        let mut brain = TimeBrain::default_config();
        brain.fit_normalization(&[0.0, 0.5, 1.0]);

        let window = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut dag = brain.to_graph(&window).unwrap();

        let prediction = brain.predict(&mut dag);
        // Check what error occurred if any
        match &prediction {
            Ok(v) => assert!(!v.is_nan()),
            Err(e) => panic!("Prediction failed: {:?}", e),
        }
    }

    #[test]
    fn test_trainer_step() {
        let brain = TimeBrain::new(TimeSeriesConfig::default().with_window_size(5));
        let mut trainer = TimeSeriesTrainer::new(brain, 0.01);

        let window = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let target = 0.6;

        let loss = trainer.train_step(&window, target);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_domain_brain_trait() {
        let brain = TimeBrain::default_config();

        assert_eq!(brain.domain_id(), "time");
        assert_eq!(brain.domain_name(), "Time Series");
    }

    #[test]
    fn test_can_process() {
        let brain = TimeBrain::default_config();

        assert!(brain.can_process("0.1, 0.2, 0.3"));
        assert!(brain.can_process("1.0,2.0,3.0"));
        assert!(!brain.can_process("hello, world"));
    }

    #[test]
    fn test_parse() {
        let brain = TimeBrain::default_config();

        let result = brain.parse("0.1, 0.2, 0.3, 0.4, 0.5");
        assert!(result.is_ok());

        let dag = result.unwrap();
        assert!(dag.node_count() >= 5);
    }

    #[test]
    fn test_validation() {
        let brain = TimeBrain::default_config();
        let window = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let dag = brain.to_graph(&window).unwrap();

        let issues = brain.validate(&dag).unwrap();
        // Should have no errors for valid graph
        assert!(issues.iter().all(|i| i.severity != ValidationSeverity::Error));
    }

    #[test]
    fn test_generate_examples() {
        let brain = TimeBrain::new(TimeSeriesConfig::default().with_window_size(5));
        let examples = brain.generate_examples(10);

        assert!(!examples.is_empty());
        for example in examples {
            assert!(!example.input.is_empty());
            assert!(!example.output.is_empty());
            assert_eq!(example.metadata.get("domain").map(|s| s.as_str()), Some("time"));
        }
    }
}
