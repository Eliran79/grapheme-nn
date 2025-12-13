//! Learnable Reasoning Components (backend-033)
//!
//! This module provides learnable versions of reasoning that can update
//! rule confidence scores based on outcomes. Key features:
//! - Learnable rule confidence with gradient updates
//! - Bayesian confidence updating based on outcomes
//! - Neural rule selection and ranking
//!
//! **GRAPHEME Protocol**: Uses LeakyReLU activation (α=0.01) and Adam optimizer (lr=0.001).

use crate::{
    ComplexityBounds, Deduction, Graph, Implication, LogicRules,
    ReasoningResult, ReasoningStep, ReasoningTrace, StepType,
};
use grapheme_core::DagNN;
use grapheme_memory::GraphFingerprint;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// LeakyReLU constant (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Default learning rate for confidence updates
pub const DEFAULT_LEARNING_RATE: f32 = 0.001;

/// Configuration for learnable reasoning
#[derive(Debug, Clone)]
pub struct LearnableReasoningConfig {
    /// Learning rate for confidence updates
    pub learning_rate: f32,
    /// Prior confidence for new rules
    pub prior_confidence: f32,
    /// Decay rate for confidence on failures
    pub decay_rate: f32,
    /// Boost rate for confidence on successes
    pub boost_rate: f32,
    /// Hidden dimension for rule embeddings
    pub hidden_dim: usize,
    /// Embedding dimension
    pub embed_dim: usize,
}

impl Default for LearnableReasoningConfig {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            prior_confidence: 0.5,
            decay_rate: 0.1,
            boost_rate: 0.2,
            hidden_dim: 64,
            embed_dim: 32,
        }
    }
}

/// Outcome of applying a rule
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RuleOutcome {
    /// Rule application succeeded
    Success,
    /// Rule application failed
    Failure,
    /// Rule was applicable but result unknown
    Uncertain,
}

/// Learnable rule confidence with Bayesian updates
#[derive(Debug)]
pub struct LearnableRuleConfidence {
    /// Rule ID to confidence mapping
    confidences: HashMap<usize, f32>,
    /// Success counts for Bayesian update
    successes: HashMap<usize, u32>,
    /// Total application counts
    applications: HashMap<usize, u32>,
    /// Configuration
    pub config: LearnableReasoningConfig,
}

impl LearnableRuleConfidence {
    /// Create new learnable confidence tracker
    pub fn new(config: LearnableReasoningConfig) -> Self {
        Self {
            confidences: HashMap::new(),
            successes: HashMap::new(),
            applications: HashMap::new(),
            config,
        }
    }

    /// Get confidence for a rule
    pub fn get_confidence(&self, rule_id: usize) -> f32 {
        self.confidences
            .get(&rule_id)
            .copied()
            .unwrap_or(self.config.prior_confidence)
    }

    /// Update confidence based on outcome (Bayesian update)
    pub fn update(&mut self, rule_id: usize, outcome: RuleOutcome) {
        // Track counts
        let apps = self.applications.entry(rule_id).or_insert(0);
        *apps += 1;

        if outcome == RuleOutcome::Success {
            let succ = self.successes.entry(rule_id).or_insert(0);
            *succ += 1;
        }

        // Bayesian confidence update: (successes + 1) / (applications + 2)
        // This is Laplace smoothing
        let successes = *self.successes.get(&rule_id).unwrap_or(&0) as f32;
        let applications = *self.applications.get(&rule_id).unwrap_or(&0) as f32;
        let new_confidence = (successes + 1.0) / (applications + 2.0);

        self.confidences.insert(rule_id, new_confidence);
    }

    /// Get confidence with gradient (for learning)
    pub fn confidence_with_grad(&self, rule_id: usize) -> (f32, f32) {
        let conf = self.get_confidence(rule_id);
        // Gradient of sigmoid-like function
        let grad = conf * (1.0 - conf);
        (conf, grad)
    }

    /// Apply gradient update to confidence
    pub fn apply_gradient(&mut self, rule_id: usize, grad: f32) {
        let current = self.get_confidence(rule_id);
        let new_conf = (current + self.config.learning_rate * grad).clamp(0.01, 0.99);
        self.confidences.insert(rule_id, new_conf);
    }

    /// Get statistics for a rule
    pub fn get_stats(&self, rule_id: usize) -> (u32, u32, f32) {
        let apps = *self.applications.get(&rule_id).unwrap_or(&0);
        let succ = *self.successes.get(&rule_id).unwrap_or(&0);
        let conf = self.get_confidence(rule_id);
        (apps, succ, conf)
    }

    /// Number of tracked rules
    pub fn len(&self) -> usize {
        self.confidences.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.confidences.is_empty()
    }
}

impl Default for LearnableRuleConfidence {
    fn default() -> Self {
        Self::new(LearnableReasoningConfig::default())
    }
}

/// Neural rule selector for choosing which rules to apply
#[derive(Debug)]
pub struct NeuralRuleSelector {
    /// Weight matrix for rule scoring: [hidden_dim, embed_dim]
    pub w_score: Array2<f32>,
    /// Bias for scoring
    pub b_score: Array1<f32>,
    /// Output layer: [1, hidden_dim]
    pub w_out: Array2<f32>,
    /// Configuration
    pub config: LearnableReasoningConfig,
    /// Gradients for w_score
    pub grad_w_score: Option<Array2<f32>>,
    /// Gradients for b_score
    pub grad_b_score: Option<Array1<f32>>,
    /// Gradients for w_out
    pub grad_w_out: Option<Array2<f32>>,
}

impl NeuralRuleSelector {
    /// Number of features from GraphFingerprint
    const FEATURE_DIM: usize = 18;

    /// Create new rule selector with DynamicXavier initialization
    pub fn new(config: LearnableReasoningConfig) -> Self {
        let mut rng = rand::thread_rng();

        let scale_score = (2.0 / (Self::FEATURE_DIM + config.hidden_dim) as f32).sqrt();
        let scale_out = (2.0 / (config.hidden_dim + 1) as f32).sqrt();

        let w_score = Array2::from_shape_fn(
            (config.hidden_dim, Self::FEATURE_DIM),
            |_| rng.gen_range(-scale_score..scale_score),
        );

        let w_out = Array2::from_shape_fn(
            (1, config.hidden_dim),
            |_| rng.gen_range(-scale_out..scale_out),
        );

        let b_score = Array1::zeros(config.hidden_dim);

        Self {
            w_score,
            b_score,
            w_out,
            config,
            grad_w_score: None,
            grad_b_score: None,
            grad_w_out: None,
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

    /// Score a rule given input graph and rule pattern
    pub fn score_rule(&self, input: &Graph, rule_antecedent: &Graph) -> f32 {
        let input_features = self.extract_features(input);
        let rule_features = self.extract_features(rule_antecedent);

        // Combine features
        let combined = (&input_features + &rule_features) / 2.0;

        // Hidden layer
        let hidden = self.w_score.dot(&combined) + &self.b_score;
        let hidden_act = hidden.mapv(|x| self.leaky_relu(x));

        // Output score (sigmoid for [0, 1])
        let score_raw = self.w_out.dot(&hidden_act)[[0]];
        1.0 / (1.0 + (-score_raw).exp())
    }

    /// Score and rank multiple rules
    pub fn rank_rules(&self, input: &Graph, rules: &[Implication]) -> Vec<(usize, f32)> {
        let mut scored: Vec<_> = rules
            .iter()
            .map(|r| (r.id, self.score_rule(input, &r.antecedent)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_w_score = None;
        self.grad_b_score = None;
        self.grad_w_out = None;
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        if let Some(ref grad) = self.grad_w_score {
            self.w_score = &self.w_score - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_b_score {
            self.b_score = &self.b_score - &(grad * lr);
        }
        if let Some(ref grad) = self.grad_w_out {
            self.w_out = &self.w_out - &(grad * lr);
        }
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.w_score.len() + self.b_score.len() + self.w_out.len()
    }
}

impl Default for NeuralRuleSelector {
    fn default() -> Self {
        Self::new(LearnableReasoningConfig::default())
    }
}

/// Learnable deduction with confidence updates
#[derive(Debug)]
pub struct LearnableDeduction {
    /// Rule confidence tracker
    pub confidences: LearnableRuleConfidence,
    /// Neural rule selector
    pub selector: NeuralRuleSelector,
    /// Similarity threshold for pattern matching
    pub match_threshold: f32,
}

impl LearnableDeduction {
    /// Create new learnable deduction
    pub fn new(config: LearnableReasoningConfig) -> Self {
        Self {
            confidences: LearnableRuleConfidence::new(config.clone()),
            selector: NeuralRuleSelector::new(config),
            match_threshold: 0.7,
        }
    }

    /// Check if pattern matches graph
    fn pattern_matches(&self, pattern: &Graph, graph: &Graph) -> bool {
        let pattern_fp = GraphFingerprint::from_graph(pattern);
        let graph_fp = GraphFingerprint::from_graph(graph);
        pattern_fp.similarity(&graph_fp) >= self.match_threshold
    }

    /// Update confidence based on reasoning outcome
    pub fn update_confidence(&mut self, rule_id: usize, outcome: RuleOutcome) {
        self.confidences.update(rule_id, outcome);
    }

    /// Learn from a successful proof
    pub fn learn_from_proof(&mut self, trace: &ReasoningTrace) {
        for step in &trace.steps {
            if let StepType::ModusPonens { rule_id } = step.step_type {
                let outcome = if trace.success {
                    RuleOutcome::Success
                } else {
                    RuleOutcome::Failure
                };
                self.update_confidence(rule_id, outcome);
            }
        }
    }

    /// Get applicable rules ranked by learned scores
    pub fn get_ranked_rules(&self, input: &Graph, rules: &[Implication]) -> Vec<(usize, f32)> {
        let neural_scores = self.selector.rank_rules(input, rules);

        // Combine neural scores with confidence
        neural_scores
            .into_iter()
            .map(|(id, neural_score)| {
                let conf = self.confidences.get_confidence(id);
                // Combined score: neural * confidence
                (id, neural_score * conf)
            })
            .collect()
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.selector.zero_grad();
    }

    /// Update parameters
    pub fn step(&mut self, lr: f32) {
        self.selector.step(lr);
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.selector.num_parameters()
    }
}

impl Default for LearnableDeduction {
    fn default() -> Self {
        Self::new(LearnableReasoningConfig::default())
    }
}

impl Deduction for LearnableDeduction {
    fn deduce(
        &self,
        premises: Vec<Graph>,
        rules: &LogicRules,
        bounds: &ComplexityBounds,
    ) -> ReasoningResult<Vec<Graph>> {
        let mut derived = Vec::new();

        // Get rules ranked by learned scores
        for premise in &premises {
            let ranked = self.get_ranked_rules(premise, &rules.implications);

            // Apply top-scoring rules
            for (rule_id, score) in ranked {
                if score < 0.3 {
                    break; // Skip low-confidence rules
                }

                let rule = rules.implications.iter().find(|r| r.id == rule_id);
                if let Some(rule) = rule {
                    if self.pattern_matches(&rule.antecedent, premise) {
                        derived.push(clone_graph(&rule.consequent));
                    }
                }
            }
        }

        // Limit by depth bound
        derived.truncate(bounds.max_proof_depth);
        Ok(derived)
    }

    fn prove(
        &self,
        goal: &Graph,
        premises: &[Graph],
        rules: &LogicRules,
        _bounds: &ComplexityBounds,
    ) -> ReasoningResult<ReasoningTrace> {
        let mut steps = Vec::new();

        // Check if goal is already in premises
        for (i, premise) in premises.iter().enumerate() {
            if self.pattern_matches(goal, premise) {
                steps.push(ReasoningStep::new(
                    0,
                    StepType::Assumption,
                    &format!("Goal matches premise {}", i),
                    clone_graph(premise),
                ));
                return Ok(ReasoningTrace::success(steps, clone_graph(goal)));
            }
        }

        // Try rules ranked by learned confidence
        for premise in premises {
            let ranked = self.get_ranked_rules(premise, &rules.implications);

            for (rule_id, score) in ranked {
                let rule = rules.implications.iter().find(|r| r.id == rule_id);
                if let Some(rule) = rule {
                    if self.pattern_matches(&rule.consequent, goal)
                        && self.pattern_matches(&rule.antecedent, premise)
                    {
                        let confidence = self.confidences.get_confidence(rule_id);

                        let mut step = ReasoningStep::new(
                            steps.len(),
                            StepType::ModusPonens { rule_id },
                            &format!("Applied rule {} (conf: {:.2}, score: {:.2})", rule.name, confidence, score),
                            clone_graph(goal),
                        );
                        step.confidence = confidence;

                        steps.push(ReasoningStep::new(
                            0,
                            StepType::Assumption,
                            "Premise established",
                            clone_graph(premise),
                        ));
                        steps.push(step);

                        return Ok(ReasoningTrace::success(steps, clone_graph(goal)));
                    }
                }
            }
        }

        Ok(ReasoningTrace::failure(steps))
    }

    fn entails(
        &self,
        premises: &[Graph],
        conclusion: &Graph,
        rules: &LogicRules,
        bounds: &ComplexityBounds,
    ) -> ReasoningResult<bool> {
        let trace = self.prove(conclusion, premises, rules, bounds)?;
        Ok(trace.success)
    }
}

/// Helper function to clone a DagNN graph
fn clone_graph(graph: &DagNN) -> DagNN {
    let text = graph.to_text();
    DagNN::from_text(&text).unwrap_or_else(|_| DagNN::new())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph(text: &str) -> Graph {
        DagNN::from_text(text).unwrap()
    }

    #[test]
    fn test_learnable_config_default() {
        let config = LearnableReasoningConfig::default();
        assert!((config.learning_rate - 0.001).abs() < 0.0001);
        assert!((config.prior_confidence - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_rule_confidence_creation() {
        let conf = LearnableRuleConfidence::default();
        assert!(conf.is_empty());
        assert!((conf.get_confidence(999) - 0.5).abs() < 0.001); // Prior
    }

    #[test]
    fn test_rule_confidence_update_success() {
        let mut conf = LearnableRuleConfidence::default();

        // Update with successes
        for _ in 0..10 {
            conf.update(1, RuleOutcome::Success);
        }

        let (apps, succ, c) = conf.get_stats(1);
        assert_eq!(apps, 10);
        assert_eq!(succ, 10);
        assert!(c > 0.8); // High confidence after all successes
    }

    #[test]
    fn test_rule_confidence_update_failure() {
        let mut conf = LearnableRuleConfidence::default();

        // Update with failures
        for _ in 0..10 {
            conf.update(1, RuleOutcome::Failure);
        }

        let (apps, succ, c) = conf.get_stats(1);
        assert_eq!(apps, 10);
        assert_eq!(succ, 0);
        assert!(c < 0.2); // Low confidence after all failures
    }

    #[test]
    fn test_rule_confidence_bayesian() {
        let mut conf = LearnableRuleConfidence::default();

        // 7 successes, 3 failures
        for _ in 0..7 {
            conf.update(1, RuleOutcome::Success);
        }
        for _ in 0..3 {
            conf.update(1, RuleOutcome::Failure);
        }

        let c = conf.get_confidence(1);
        // Bayesian: (7+1)/(10+2) = 8/12 = 0.667
        assert!((c - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_neural_rule_selector_creation() {
        let selector = NeuralRuleSelector::default();
        assert!(selector.num_parameters() > 0);
    }

    #[test]
    fn test_neural_rule_selector_score() {
        let selector = NeuralRuleSelector::default();
        let input = make_graph("hello");
        let antecedent = make_graph("world");

        let score = selector.score_rule(&input, &antecedent);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_neural_rule_selector_rank() {
        let selector = NeuralRuleSelector::default();
        let input = make_graph("test input");

        let rules = vec![
            Implication::new(1, "rule1", make_graph("a"), make_graph("b")),
            Implication::new(2, "rule2", make_graph("c"), make_graph("d")),
        ];

        let ranked = selector.rank_rules(&input, &rules);
        assert_eq!(ranked.len(), 2);
        // First should have higher or equal score than second
        assert!(ranked[0].1 >= ranked[1].1);
    }

    #[test]
    fn test_learnable_deduction_creation() {
        let deduction = LearnableDeduction::default();
        assert!(deduction.num_parameters() > 0);
    }

    #[test]
    fn test_learnable_deduction_deduce() {
        let deduction = LearnableDeduction::default();
        let bounds = ComplexityBounds::default();

        let premises = vec![make_graph("premise")];
        let rules = LogicRules::new();

        let derived = deduction.deduce(premises, &rules, &bounds).unwrap();
        // No rules, no derivations
        assert!(derived.is_empty());
    }

    #[test]
    fn test_learnable_deduction_learn_from_proof() {
        let mut deduction = LearnableDeduction::default();

        // Create a successful trace
        let steps = vec![
            ReasoningStep::new(0, StepType::Assumption, "premise", make_graph("a")),
            ReasoningStep::new(1, StepType::ModusPonens { rule_id: 42 }, "applied rule", make_graph("b")),
        ];
        let trace = ReasoningTrace::success(steps, make_graph("conclusion"));

        // Learning should update confidence
        deduction.learn_from_proof(&trace);

        let (apps, succ, _) = deduction.confidences.get_stats(42);
        assert_eq!(apps, 1);
        assert_eq!(succ, 1);
    }

    #[test]
    fn test_learnable_deduction_gradient_flow() {
        let mut deduction = LearnableDeduction::default();

        // Zero grad and step should not panic
        deduction.zero_grad();
        deduction.step(0.001);
    }
}
