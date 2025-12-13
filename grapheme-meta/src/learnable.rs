//! Learnable Meta-Cognition Components (backend-035)
//!
//! This module provides learnable components for meta-cognition:
//! - Learnable uncertainty estimation with calibration
//! - Learnable confidence prediction
//! - Learnable introspection for cognitive state monitoring
//!
//! **GRAPHEME Protocol**: Uses LeakyReLU activation (α=0.01) and Adam optimizer (lr=0.001).

use crate::{CognitiveState, Graph, UncertaintyEstimate};
use grapheme_memory::GraphFingerprint;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::VecDeque;

/// LeakyReLU constant (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Default learning rate
pub const DEFAULT_LEARNING_RATE: f32 = 0.001;

/// Configuration for learnable meta-cognition
#[derive(Debug, Clone)]
pub struct LearnableMetaConfig {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Calibration window size
    pub calibration_window: usize,
    /// Target calibration error
    pub target_calibration: f32,
}

impl Default for LearnableMetaConfig {
    fn default() -> Self {
        Self {
            embed_dim: 32,
            hidden_dim: 64,
            learning_rate: DEFAULT_LEARNING_RATE,
            calibration_window: 100,
            target_calibration: 0.1,
        }
    }
}

/// Learnable uncertainty estimator with neural network
#[derive(Debug)]
pub struct LearnableUncertaintyEstimator {
    /// Input layer: [hidden_dim, feature_dim]
    pub w_input: Array2<f32>,
    /// Bias for input
    pub b_input: Array1<f32>,
    /// Epistemic head: [1, hidden_dim]
    pub w_epistemic: Array2<f32>,
    /// Aleatoric head: [1, hidden_dim]
    pub w_aleatoric: Array2<f32>,
    /// Configuration
    pub config: LearnableMetaConfig,
    /// Gradients
    pub grad_w_input: Option<Array2<f32>>,
    pub grad_b_input: Option<Array1<f32>>,
    pub grad_w_epistemic: Option<Array2<f32>>,
    pub grad_w_aleatoric: Option<Array2<f32>>,
}

impl LearnableUncertaintyEstimator {
    /// Number of input features
    const FEATURE_DIM: usize = 18;

    /// Create new uncertainty estimator with DynamicXavier initialization
    pub fn new(config: LearnableMetaConfig) -> Self {
        let mut rng = rand::thread_rng();

        let scale_input = (2.0 / (Self::FEATURE_DIM + config.hidden_dim) as f32).sqrt();
        let scale_head = (2.0 / (config.hidden_dim + 1) as f32).sqrt();

        let w_input = Array2::from_shape_fn(
            (config.hidden_dim, Self::FEATURE_DIM),
            |_| rng.gen_range(-scale_input..scale_input),
        );

        let w_epistemic = Array2::from_shape_fn(
            (1, config.hidden_dim),
            |_| rng.gen_range(-scale_head..scale_head),
        );

        let w_aleatoric = Array2::from_shape_fn(
            (1, config.hidden_dim),
            |_| rng.gen_range(-scale_head..scale_head),
        );

        let b_input = Array1::zeros(config.hidden_dim);

        Self {
            w_input,
            b_input,
            w_epistemic,
            w_aleatoric,
            config,
            grad_w_input: None,
            grad_b_input: None,
            grad_w_epistemic: None,
            grad_w_aleatoric: None,
        }
    }

    /// Extract features from graph
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

    /// LeakyReLU activation
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Sigmoid for [0, 1] output
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Estimate uncertainty for a graph
    pub fn estimate(&self, graph: &Graph) -> UncertaintyEstimate {
        let features = self.extract_features(graph);

        // Hidden layer
        let hidden_pre = self.w_input.dot(&features) + &self.b_input;
        let hidden = hidden_pre.mapv(|x| self.leaky_relu(x));

        // Epistemic and aleatoric heads
        let epistemic_raw = self.w_epistemic.dot(&hidden)[[0]];
        let aleatoric_raw = self.w_aleatoric.dot(&hidden)[[0]];

        let epistemic = self.sigmoid(epistemic_raw);
        let aleatoric = self.sigmoid(aleatoric_raw);

        UncertaintyEstimate::new(epistemic, aleatoric)
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_input = None;
        self.grad_b_input = None;
        self.grad_w_epistemic = None;
        self.grad_w_aleatoric = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_input {
            self.w_input = &self.w_input - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_input {
            self.b_input = &self.b_input - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_epistemic {
            self.w_epistemic = &self.w_epistemic - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_aleatoric {
            self.w_aleatoric = &self.w_aleatoric - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_input.len() + self.b_input.len()
            + self.w_epistemic.len() + self.w_aleatoric.len()
    }
}

impl Default for LearnableUncertaintyEstimator {
    fn default() -> Self {
        Self::new(LearnableMetaConfig::default())
    }
}

/// Confidence calibrator that learns to produce calibrated confidence scores
#[derive(Debug)]
pub struct ConfidenceCalibrator {
    /// Temperature for calibration
    pub temperature: f32,
    /// Bias for calibration
    pub bias: f32,
    /// Calibration history: (predicted_confidence, actual_outcome)
    history: VecDeque<(f32, bool)>,
    /// Maximum history size
    max_history: usize,
    /// Expected calibration error
    ece: f32,
}

impl ConfidenceCalibrator {
    /// Create new calibrator
    pub fn new(max_history: usize) -> Self {
        Self {
            temperature: 1.0,
            bias: 0.0,
            history: VecDeque::with_capacity(max_history),
            max_history,
            ece: 0.5, // Start with high expected error
        }
    }

    /// Calibrate a raw confidence score
    pub fn calibrate(&self, raw_confidence: f32) -> f32 {
        // Apply temperature scaling and bias
        let scaled = (raw_confidence / self.temperature) + self.bias;
        // Sigmoid to keep in [0, 1]
        1.0 / (1.0 + (-scaled).exp())
    }

    /// Record an outcome for calibration
    pub fn record_outcome(&mut self, predicted_confidence: f32, was_correct: bool) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back((predicted_confidence, was_correct));

        // Update calibration
        self.update_calibration();
    }

    /// Update calibration parameters based on history
    fn update_calibration(&mut self) {
        if self.history.len() < 10 {
            return;
        }

        // Compute expected calibration error (ECE)
        // Group predictions into bins
        const NUM_BINS: usize = 10;
        let mut bin_counts = [0usize; NUM_BINS];
        let mut bin_correct = [0usize; NUM_BINS];
        let mut bin_confidence_sum = [0.0f32; NUM_BINS];

        for (conf, correct) in &self.history {
            let bin = ((conf * NUM_BINS as f32) as usize).min(NUM_BINS - 1);
            bin_counts[bin] += 1;
            bin_confidence_sum[bin] += conf;
            if *correct {
                bin_correct[bin] += 1;
            }
        }

        // Compute ECE and adjust temperature
        let mut ece_sum = 0.0;
        let total = self.history.len() as f32;

        for i in 0..NUM_BINS {
            if bin_counts[i] > 0 {
                let avg_conf = bin_confidence_sum[i] / bin_counts[i] as f32;
                let accuracy = bin_correct[i] as f32 / bin_counts[i] as f32;
                let bin_weight = bin_counts[i] as f32 / total;
                ece_sum += bin_weight * (avg_conf - accuracy).abs();
            }
        }

        self.ece = ece_sum;

        // Adjust temperature based on ECE trend
        // If predictions are overconfident, increase temperature
        // If underconfident, decrease temperature
        let avg_conf: f32 = self.history.iter().map(|(c, _)| c).sum::<f32>() / total;
        let avg_correct: f32 = self.history.iter().filter(|(_, c)| *c).count() as f32 / total;

        if avg_conf > avg_correct + 0.05 {
            // Overconfident - increase temperature (dampens confidence)
            self.temperature *= 1.01;
        } else if avg_conf < avg_correct - 0.05 {
            // Underconfident - decrease temperature
            self.temperature *= 0.99;
        }

        // Clamp temperature
        self.temperature = self.temperature.clamp(0.1, 10.0);
    }

    /// Get expected calibration error
    pub fn expected_calibration_error(&self) -> f32 {
        self.ece
    }

    /// Is calibration good?
    pub fn is_calibrated(&self, threshold: f32) -> bool {
        self.ece < threshold
    }

    /// Get number of samples in history
    pub fn history_size(&self) -> usize {
        self.history.len()
    }
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Learnable introspection monitor for cognitive state
#[derive(Debug)]
pub struct IntrospectionMonitor {
    /// State encoder: [hidden_dim, state_dim]
    pub w_state: Array2<f32>,
    /// Bias for state
    pub b_state: Array1<f32>,
    /// Overload predictor: [1, hidden_dim]
    pub w_overload: Array2<f32>,
    /// Stuck predictor: [1, hidden_dim]
    pub w_stuck: Array2<f32>,
    /// Configuration
    pub config: LearnableMetaConfig,
    /// Historical states for pattern detection
    state_history: VecDeque<Array1<f32>>,
    /// Gradients
    pub grad_w_state: Option<Array2<f32>>,
    pub grad_b_state: Option<Array1<f32>>,
    pub grad_w_overload: Option<Array2<f32>>,
    pub grad_w_stuck: Option<Array2<f32>>,
}

impl IntrospectionMonitor {
    /// Cognitive state dimension
    const STATE_DIM: usize = 7; // load, depth, contradictions, confidence, steps, subgoals, time

    /// Create new introspection monitor
    pub fn new(config: LearnableMetaConfig) -> Self {
        let mut rng = rand::thread_rng();

        let scale_state = (2.0 / (Self::STATE_DIM + config.hidden_dim) as f32).sqrt();
        let scale_head = (2.0 / (config.hidden_dim + 1) as f32).sqrt();

        let w_state = Array2::from_shape_fn(
            (config.hidden_dim, Self::STATE_DIM),
            |_| rng.gen_range(-scale_state..scale_state),
        );

        let w_overload = Array2::from_shape_fn(
            (1, config.hidden_dim),
            |_| rng.gen_range(-scale_head..scale_head),
        );

        let w_stuck = Array2::from_shape_fn(
            (1, config.hidden_dim),
            |_| rng.gen_range(-scale_head..scale_head),
        );

        let b_state = Array1::zeros(config.hidden_dim);

        Self {
            w_state,
            b_state,
            w_overload,
            w_stuck,
            config,
            state_history: VecDeque::with_capacity(50),
            grad_w_state: None,
            grad_b_state: None,
            grad_w_overload: None,
            grad_w_stuck: None,
        }
    }

    /// Extract features from cognitive state
    fn extract_state_features(&self, state: &CognitiveState) -> Array1<f32> {
        let mut features = Array1::zeros(Self::STATE_DIM);
        features[0] = state.working_memory_load;
        features[1] = (state.reasoning_depth as f32 / 20.0).min(1.0);
        features[2] = (state.contradiction_count as f32 / 10.0).min(1.0);
        features[3] = state.confidence;
        features[4] = (state.steps_taken as f32 / 1000.0).min(1.0);
        features[5] = (state.subgoals.len() as f32 / 10.0).min(1.0);
        features[6] = (state.elapsed.as_secs_f32() / 60.0).min(1.0);
        features
    }

    /// LeakyReLU activation
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x }
    }

    /// Sigmoid for probability output
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Predict if system is likely to become overloaded
    pub fn predict_overload(&self, state: &CognitiveState) -> f32 {
        let features = self.extract_state_features(state);

        let hidden_pre = self.w_state.dot(&features) + &self.b_state;
        let hidden = hidden_pre.mapv(|x| self.leaky_relu(x));

        let overload_raw = self.w_overload.dot(&hidden)[[0]];
        self.sigmoid(overload_raw)
    }

    /// Predict if reasoning is stuck
    pub fn predict_stuck(&self, state: &CognitiveState) -> f32 {
        let features = self.extract_state_features(state);

        let hidden_pre = self.w_state.dot(&features) + &self.b_state;
        let hidden = hidden_pre.mapv(|x| self.leaky_relu(x));

        let stuck_raw = self.w_stuck.dot(&hidden)[[0]];
        self.sigmoid(stuck_raw)
    }

    /// Record state for pattern detection
    pub fn observe_state(&mut self, state: &CognitiveState) {
        let features = self.extract_state_features(state);

        if self.state_history.len() >= 50 {
            self.state_history.pop_front();
        }
        self.state_history.push_back(features);
    }

    /// Detect if state is deteriorating
    pub fn is_deteriorating(&self) -> bool {
        if self.state_history.len() < 5 {
            return false;
        }

        // Check if working memory load is increasing
        let recent: Vec<_> = self.state_history.iter().rev().take(5).collect();
        let mut increasing = 0;
        for i in 1..recent.len() {
            if recent[i - 1][0] > recent[i][0] {
                increasing += 1;
            }
        }

        increasing >= 3
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_state = None;
        self.grad_b_state = None;
        self.grad_w_overload = None;
        self.grad_w_stuck = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_state {
            self.w_state = &self.w_state - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_state {
            self.b_state = &self.b_state - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_overload {
            self.w_overload = &self.w_overload - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_stuck {
            self.w_stuck = &self.w_stuck - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_state.len() + self.b_state.len()
            + self.w_overload.len() + self.w_stuck.len()
    }
}

impl Default for IntrospectionMonitor {
    fn default() -> Self {
        Self::new(LearnableMetaConfig::default())
    }
}

/// Complete learnable meta-cognition system
#[derive(Debug)]
pub struct LearnableMetaCognition {
    /// Uncertainty estimator
    pub uncertainty: LearnableUncertaintyEstimator,
    /// Confidence calibrator
    pub calibrator: ConfidenceCalibrator,
    /// Introspection monitor
    pub introspection: IntrospectionMonitor,
    /// Configuration
    pub config: LearnableMetaConfig,
}

impl LearnableMetaCognition {
    /// Create new learnable meta-cognition system
    pub fn new(config: LearnableMetaConfig) -> Self {
        let uncertainty = LearnableUncertaintyEstimator::new(config.clone());
        let calibrator = ConfidenceCalibrator::new(config.calibration_window);
        let introspection = IntrospectionMonitor::new(config.clone());

        Self {
            uncertainty,
            calibrator,
            introspection,
            config,
        }
    }

    /// Estimate uncertainty for a graph
    pub fn estimate_uncertainty(&self, graph: &Graph) -> UncertaintyEstimate {
        self.uncertainty.estimate(graph)
    }

    /// Get calibrated confidence
    pub fn calibrated_confidence(&self, raw_confidence: f32) -> f32 {
        self.calibrator.calibrate(raw_confidence)
    }

    /// Record outcome for calibration
    pub fn record_outcome(&mut self, confidence: f32, was_correct: bool) {
        self.calibrator.record_outcome(confidence, was_correct);
    }

    /// Monitor cognitive state
    pub fn monitor_state(&mut self, state: &CognitiveState) {
        self.introspection.observe_state(state);
    }

    /// Predict if system will overload
    pub fn predict_overload(&self, state: &CognitiveState) -> f32 {
        self.introspection.predict_overload(state)
    }

    /// Predict if reasoning is stuck
    pub fn predict_stuck(&self, state: &CognitiveState) -> f32 {
        self.introspection.predict_stuck(state)
    }

    /// Check if state is deteriorating
    pub fn is_deteriorating(&self) -> bool {
        self.introspection.is_deteriorating()
    }

    /// Get calibration error
    pub fn calibration_error(&self) -> f32 {
        self.calibrator.expected_calibration_error()
    }

    /// Zero gradients for all components
    pub fn zero_grad(&mut self) {
        self.uncertainty.zero_grad();
        self.introspection.zero_grad();
    }

    /// Update all parameters
    pub fn step(&mut self, lr: f32) {
        self.uncertainty.step(lr);
        self.introspection.step(lr);
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        self.uncertainty.num_parameters() + self.introspection.num_parameters()
    }
}

impl Default for LearnableMetaCognition {
    fn default() -> Self {
        Self::new(LearnableMetaConfig::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use grapheme_core::DagNN;
    use std::time::Duration;

    fn make_graph(text: &str) -> Graph {
        DagNN::from_text(text).unwrap()
    }

    fn make_cognitive_state() -> CognitiveState {
        CognitiveState {
            working_memory_load: 0.5,
            reasoning_depth: 5,
            contradiction_count: 0,
            confidence: 0.8,
            current_goal: Some("test".to_string()),
            subgoals: vec!["sub1".to_string()],
            steps_taken: 100,
            elapsed: Duration::from_secs(5),
        }
    }

    #[test]
    fn test_learnable_meta_config_default() {
        let config = LearnableMetaConfig::default();
        assert_eq!(config.embed_dim, 32);
        assert!((config.learning_rate - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_uncertainty_estimator_creation() {
        let estimator = LearnableUncertaintyEstimator::default();
        assert!(estimator.num_parameters() > 0);
    }

    #[test]
    fn test_uncertainty_estimator_estimate() {
        let estimator = LearnableUncertaintyEstimator::default();
        let graph = make_graph("test input");

        let estimate = estimator.estimate(&graph);
        assert!(estimate.epistemic >= 0.0 && estimate.epistemic <= 1.0);
        assert!(estimate.aleatoric >= 0.0 && estimate.aleatoric <= 1.0);
        assert!(estimate.total >= 0.0 && estimate.total <= 1.0);
    }

    #[test]
    fn test_confidence_calibrator_creation() {
        let calibrator = ConfidenceCalibrator::default();
        assert_eq!(calibrator.history_size(), 0);
    }

    #[test]
    fn test_confidence_calibrator_calibrate() {
        let calibrator = ConfidenceCalibrator::default();

        let calibrated = calibrator.calibrate(0.8);
        assert!(calibrated >= 0.0 && calibrated <= 1.0);
    }

    #[test]
    fn test_confidence_calibrator_record() {
        let mut calibrator = ConfidenceCalibrator::new(10);

        for i in 0..15 {
            let conf = (i as f32) / 15.0;
            let correct = conf > 0.5;
            calibrator.record_outcome(conf, correct);
        }

        assert_eq!(calibrator.history_size(), 10); // Bounded by max
    }

    #[test]
    fn test_confidence_calibrator_calibration() {
        let mut calibrator = ConfidenceCalibrator::new(100);

        // Add overconfident predictions
        for _ in 0..50 {
            calibrator.record_outcome(0.9, false); // Wrong but confident
        }

        // Temperature should increase (dampening confidence)
        assert!(calibrator.temperature > 1.0);
    }

    #[test]
    fn test_introspection_monitor_creation() {
        let monitor = IntrospectionMonitor::default();
        assert!(monitor.num_parameters() > 0);
    }

    #[test]
    fn test_introspection_predict_overload() {
        let monitor = IntrospectionMonitor::default();
        let state = make_cognitive_state();

        let overload_prob = monitor.predict_overload(&state);
        assert!(overload_prob >= 0.0 && overload_prob <= 1.0);
    }

    #[test]
    fn test_introspection_predict_stuck() {
        let monitor = IntrospectionMonitor::default();
        let state = make_cognitive_state();

        let stuck_prob = monitor.predict_stuck(&state);
        assert!(stuck_prob >= 0.0 && stuck_prob <= 1.0);
    }

    #[test]
    fn test_introspection_observe_state() {
        let mut monitor = IntrospectionMonitor::default();
        let state = make_cognitive_state();

        for _ in 0..10 {
            monitor.observe_state(&state);
        }

        // Should have recorded states
        assert!(!monitor.state_history.is_empty());
    }

    #[test]
    fn test_learnable_meta_cognition_creation() {
        let meta = LearnableMetaCognition::default();
        assert!(meta.num_parameters() > 0);
    }

    #[test]
    fn test_learnable_meta_cognition_estimate_uncertainty() {
        let meta = LearnableMetaCognition::default();
        let graph = make_graph("test");

        let estimate = meta.estimate_uncertainty(&graph);
        assert!(estimate.total >= 0.0);
    }

    #[test]
    fn test_learnable_meta_cognition_calibrated_confidence() {
        let meta = LearnableMetaCognition::default();

        let calibrated = meta.calibrated_confidence(0.7);
        assert!(calibrated >= 0.0 && calibrated <= 1.0);
    }

    #[test]
    fn test_learnable_meta_cognition_gradient_flow() {
        let mut meta = LearnableMetaCognition::default();

        meta.zero_grad();
        meta.step(0.001);
    }
}
