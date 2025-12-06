//! # grapheme-meta
//!
//! Meta-cognition for GRAPHEME neural network.
//!
//! This crate provides self-awareness of cognitive states:
//! - **Uncertainty Estimation**: Know what you don't know
//! - **Introspection**: Monitor reasoning state
//! - **Error Detection**: Find contradictions
//! - **Adaptive Computation**: Allocate resources wisely
//! - **Limitation Recognition**: Know when to ask for help
//!
//! ## Design Philosophy
//!
//! Meta-cognition is an open research problem. This implementation provides:
//! - Basic uncertainty quantification (epistemic vs aleatoric)
//! - Simple confidence calibration
//! - Compute budget management
//! - Contradiction detection hooks
//!
//! Real meta-cognition may require learned policies.

use grapheme_core::{
    BrainRegistry, CognitiveBrainBridge, DagNN, DefaultCognitiveBridge,
    DomainBrain, Learnable, LearnableParam,
};
use grapheme_reason::ReasoningStep;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::time::Duration;
use thiserror::Error;

// ============================================================================
// Type Aliases
// ============================================================================

/// Graph type
pub type Graph = DagNN;

// ============================================================================
// Error Types
// ============================================================================

/// Errors in meta-cognition operations
#[derive(Error, Debug)]
pub enum MetaError {
    #[error("Introspection failed: {0}")]
    IntrospectionFailed(String),
    #[error("Calibration failed: insufficient data")]
    CalibrationFailed,
    #[error("Budget exceeded")]
    BudgetExceeded,
}

/// Result type for meta-cognition operations
pub type MetaResult<T> = Result<T, MetaError>;

// ============================================================================
// Uncertainty Types
// ============================================================================

/// Source of uncertainty
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintySource {
    /// Lack of relevant knowledge
    KnowledgeGap,
    /// Ambiguous input
    Ambiguity,
    /// Low sample count
    LimitedExamples,
    /// Out of distribution
    OutOfDistribution,
    /// Internal inconsistency
    Contradiction,
    /// Model limitation
    ModelLimit,
}

/// Uncertainty estimate for a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyEstimate {
    /// Epistemic uncertainty (reducible with more knowledge)
    pub epistemic: f32,
    /// Aleatoric uncertainty (inherent randomness)
    pub aleatoric: f32,
    /// Total combined uncertainty
    pub total: f32,
    /// Sources of uncertainty
    pub sources: Vec<UncertaintySource>,
}

impl UncertaintyEstimate {
    /// Create a new uncertainty estimate
    pub fn new(epistemic: f32, aleatoric: f32) -> Self {
        let epistemic = epistemic.clamp(0.0, 1.0);
        let aleatoric = aleatoric.clamp(0.0, 1.0);
        // Combine using independence assumption
        let total = 1.0 - (1.0 - epistemic) * (1.0 - aleatoric);
        Self {
            epistemic,
            aleatoric,
            total,
            sources: Vec::new(),
        }
    }

    /// Create certain estimate
    pub fn certain() -> Self {
        Self {
            epistemic: 0.0,
            aleatoric: 0.0,
            total: 0.0,
            sources: Vec::new(),
        }
    }

    /// Create uncertain estimate
    pub fn uncertain(reason: UncertaintySource) -> Self {
        Self {
            epistemic: 0.8,
            aleatoric: 0.2,
            total: 0.9,
            sources: vec![reason],
        }
    }

    /// Add uncertainty source
    pub fn with_source(mut self, source: UncertaintySource) -> Self {
        self.sources.push(source);
        self
    }

    /// Is this estimate confident?
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.total < threshold
    }
}

impl Default for UncertaintyEstimate {
    fn default() -> Self {
        Self::new(0.5, 0.1)
    }
}

// ============================================================================
// Cognitive State
// ============================================================================

/// Current cognitive state for introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    /// Working memory load (0.0 to 1.0)
    pub working_memory_load: f32,
    /// Current reasoning depth
    pub reasoning_depth: usize,
    /// Number of detected contradictions
    pub contradiction_count: usize,
    /// Overall confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Current goal (if any)
    pub current_goal: Option<String>,
    /// Active subgoals
    pub subgoals: Vec<String>,
    /// Steps taken so far
    pub steps_taken: usize,
    /// Time elapsed
    pub elapsed: Duration,
}

impl CognitiveState {
    /// Create new cognitive state
    pub fn new() -> Self {
        Self {
            working_memory_load: 0.0,
            reasoning_depth: 0,
            contradiction_count: 0,
            confidence: 1.0,
            current_goal: None,
            subgoals: Vec::new(),
            steps_taken: 0,
            elapsed: Duration::ZERO,
        }
    }

    /// Set goal
    pub fn with_goal(mut self, goal: &str) -> Self {
        self.current_goal = Some(goal.to_string());
        self
    }

    /// Is the system overloaded?
    pub fn is_overloaded(&self) -> bool {
        self.working_memory_load > 0.9
    }

    /// Is reasoning stuck?
    pub fn is_stuck(&self, max_depth: usize) -> bool {
        self.reasoning_depth >= max_depth
    }
}

impl Default for CognitiveState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Compute Budget
// ============================================================================

/// Budget for computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeBudget {
    /// Maximum steps to take
    pub max_steps: usize,
    /// Maximum time to spend
    pub max_time: Duration,
    /// Maximum memory (in nodes)
    pub max_memory: usize,
    /// Priority level (higher = more important)
    pub priority: u8,
}

impl ComputeBudget {
    /// Create a new compute budget
    pub fn new(max_steps: usize, max_time: Duration, max_memory: usize) -> Self {
        Self {
            max_steps,
            max_time,
            max_memory,
            priority: 5,
        }
    }

    /// Quick budget (small)
    pub fn quick() -> Self {
        Self::new(100, Duration::from_millis(100), 1000)
    }

    /// Standard budget
    pub fn standard() -> Self {
        Self::new(1000, Duration::from_secs(10), 100_000)
    }

    /// Extended budget (for hard problems)
    pub fn extended() -> Self {
        Self::new(10_000, Duration::from_secs(60), 1_000_000)
    }

    /// Check if budget is exceeded
    pub fn is_exceeded(&self, state: &CognitiveState) -> bool {
        state.steps_taken >= self.max_steps || state.elapsed >= self.max_time
    }

    /// Remaining steps
    pub fn remaining_steps(&self, state: &CognitiveState) -> usize {
        self.max_steps.saturating_sub(state.steps_taken)
    }
}

impl Default for ComputeBudget {
    fn default() -> Self {
        Self::standard()
    }
}

// ============================================================================
// Limitation Types
// ============================================================================

/// Type of cognitive limitation encountered
#[derive(Debug, Clone)]
pub enum LimitationType {
    /// Don't have relevant knowledge
    KnowledgeGap {
        /// What knowledge is missing
        missing: String,
    },
    /// Can't derive conclusion
    ReasoningLimit {
        /// Why reasoning failed
        reason: String,
    },
    /// Multiple valid interpretations
    Ambiguity {
        /// Possible interpretations
        alternatives: Vec<String>,
    },
    /// Inconsistent beliefs
    Contradiction {
        /// Description of contradiction
        description: String,
    },
    /// Out of compute budget
    ResourceExhausted {
        /// Which resource
        resource: String,
    },
    /// Task is beyond capabilities
    BeyondCapabilities {
        /// What capability is missing
        capability: String,
    },
}

impl LimitationType {
    /// Get a human-readable description
    pub fn description(&self) -> String {
        match self {
            LimitationType::KnowledgeGap { missing } => {
                format!("Missing knowledge: {}", missing)
            }
            LimitationType::ReasoningLimit { reason } => {
                format!("Reasoning limit: {}", reason)
            }
            LimitationType::Ambiguity { alternatives } => {
                format!("Ambiguous ({} interpretations)", alternatives.len())
            }
            LimitationType::Contradiction { description } => {
                format!("Contradiction: {}", description)
            }
            LimitationType::ResourceExhausted { resource } => {
                format!("Resource exhausted: {}", resource)
            }
            LimitationType::BeyondCapabilities { capability } => {
                format!("Beyond capabilities: {}", capability)
            }
        }
    }
}

// ============================================================================
// Computation Strategy
// ============================================================================

/// Strategy for allocating computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeStrategy {
    /// How many steps to allocate
    pub steps: usize,
    /// Depth limit for this phase
    pub depth_limit: usize,
    /// Whether to use approximations
    pub approximate: bool,
    /// Early stopping threshold
    pub early_stop_confidence: f32,
    /// Strategy type
    pub strategy_type: StrategyType,
}

/// Type of computation strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyType {
    /// Quick, shallow search
    Quick,
    /// Standard exhaustive search
    Standard,
    /// Deep, careful analysis
    Deep,
    /// Breadth-first exploration
    Exploratory,
    /// Best-effort with fallback
    BestEffort,
}

impl ComputeStrategy {
    /// Create quick strategy
    pub fn quick() -> Self {
        Self {
            steps: 10,
            depth_limit: 3,
            approximate: true,
            early_stop_confidence: 0.8,
            strategy_type: StrategyType::Quick,
        }
    }

    /// Create standard strategy
    pub fn standard() -> Self {
        Self {
            steps: 100,
            depth_limit: 10,
            approximate: false,
            early_stop_confidence: 0.95,
            strategy_type: StrategyType::Standard,
        }
    }

    /// Create deep strategy
    pub fn deep() -> Self {
        Self {
            steps: 1000,
            depth_limit: 20,
            approximate: false,
            early_stop_confidence: 0.99,
            strategy_type: StrategyType::Deep,
        }
    }
}

impl Default for ComputeStrategy {
    fn default() -> Self {
        Self::standard()
    }
}

// ============================================================================
// Contradiction
// ============================================================================

/// A detected contradiction
#[derive(Debug, Clone)]
pub struct Contradiction {
    /// Description
    pub description: String,
    /// Conflicting step indices
    pub conflicting_steps: Vec<usize>,
    /// Severity (0.0 to 1.0)
    pub severity: f32,
}

impl Contradiction {
    pub fn new(description: &str, steps: Vec<usize>, severity: f32) -> Self {
        Self {
            description: description.to_string(),
            conflicting_steps: steps,
            severity: severity.clamp(0.0, 1.0),
        }
    }
}

// ============================================================================
// Meta-Cognition Trait
// ============================================================================

/// Trait for cognitive self-monitoring
pub trait MetaCognition: Send + Sync + Debug {
    /// Estimate uncertainty for a query
    fn estimate_uncertainty(&self, query: &Graph) -> UncertaintyEstimate;

    /// Get current cognitive state
    fn introspect(&self) -> CognitiveState;

    /// Decide how much computation to allocate
    fn allocate_computation(&self, task: &Graph, budget: &ComputeBudget) -> ComputeStrategy;

    /// Check reasoning trace for consistency
    fn verify_consistency(&self, trace: &[ReasoningStep]) -> Vec<Contradiction>;

    /// Detect and classify cognitive limits
    fn recognize_limits(&self, task: &Graph) -> Option<LimitationType>;

    /// Should we continue reasoning?
    fn should_continue(&self, state: &CognitiveState, budget: &ComputeBudget) -> bool;

    /// Calibrate confidence estimates
    fn calibrate(&mut self, predictions: &[(f32, bool)]);

    /// Get current calibration error
    fn calibration_error(&self) -> f32;
}

// ============================================================================
// Simple Implementation
// ============================================================================

/// Simple meta-cognition implementation
#[derive(Debug, Default)]
pub struct SimpleMetaCognition {
    state: CognitiveState,
    calibration_data: Vec<(f32, bool)>,
}

impl SimpleMetaCognition {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update cognitive state
    pub fn update_state(&mut self, state: CognitiveState) {
        self.state = state;
    }
}

impl MetaCognition for SimpleMetaCognition {
    fn estimate_uncertainty(&self, query: &Graph) -> UncertaintyEstimate {
        // Simple heuristic: larger graphs have more uncertainty
        let complexity = query.node_count() as f32 / 100.0;
        let epistemic = (complexity * 0.5).min(0.9);
        let aleatoric = 0.1;

        UncertaintyEstimate::new(epistemic, aleatoric)
    }

    fn introspect(&self) -> CognitiveState {
        self.state.clone()
    }

    fn allocate_computation(&self, task: &Graph, budget: &ComputeBudget) -> ComputeStrategy {
        // Simple heuristic: larger tasks get more computation
        let task_size = task.node_count();

        if task_size < 10 || budget.max_steps < 100 {
            ComputeStrategy::quick()
        } else if task_size < 100 || budget.max_steps < 1000 {
            ComputeStrategy::standard()
        } else {
            ComputeStrategy::deep()
        }
    }

    fn verify_consistency(&self, _trace: &[ReasoningStep]) -> Vec<Contradiction> {
        // Simplified: no contradictions detected
        // Real implementation would check for conflicting conclusions
        Vec::new()
    }

    fn recognize_limits(&self, task: &Graph) -> Option<LimitationType> {
        // Simple heuristics
        let state = self.introspect();

        if state.is_overloaded() {
            return Some(LimitationType::ResourceExhausted {
                resource: "working_memory".to_string(),
            });
        }

        if state.contradiction_count > 5 {
            return Some(LimitationType::Contradiction {
                description: "Too many contradictions".to_string(),
            });
        }

        // Large tasks might be beyond capabilities
        if task.node_count() > 10000 {
            return Some(LimitationType::BeyondCapabilities {
                capability: "large_graph_processing".to_string(),
            });
        }

        None
    }

    fn should_continue(&self, state: &CognitiveState, budget: &ComputeBudget) -> bool {
        // Continue if:
        // 1. Budget not exceeded
        // 2. Not stuck
        // 3. Still making progress

        if budget.is_exceeded(state) {
            return false;
        }

        if state.is_stuck(20) {
            return false;
        }

        if state.confidence > 0.99 {
            return false; // Already confident enough
        }

        true
    }

    fn calibrate(&mut self, predictions: &[(f32, bool)]) {
        self.calibration_data.extend_from_slice(predictions);

        // Keep only recent data
        if self.calibration_data.len() > 1000 {
            self.calibration_data = self.calibration_data[self.calibration_data.len() - 1000..].to_vec();
        }
    }

    fn calibration_error(&self) -> f32 {
        if self.calibration_data.is_empty() {
            return 0.0;
        }

        // Compute expected calibration error (ECE)
        // Group by confidence bins and compare predicted vs actual
        let mut total_error = 0.0;
        let n = self.calibration_data.len() as f32;

        for (confidence, outcome) in &self.calibration_data {
            let actual = if *outcome { 1.0 } else { 0.0 };
            total_error += (confidence - actual).abs();
        }

        total_error / n
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a default meta-cognition system
pub fn create_default_metacognition() -> SimpleMetaCognition {
    SimpleMetaCognition::new()
}

// ============================================================================
// Learnable Meta-Cognition
// ============================================================================

/// Learnable meta-cognition with trainable calibration and allocation
///
/// This module learns to estimate uncertainty and allocate computation
/// more accurately based on experience.
#[derive(Debug, Clone)]
pub struct LearnableMetaCognition {
    /// Bias for calibrating confidence predictions
    pub calibration_bias: LearnableParam,
    /// Scale for uncertainty estimates
    pub uncertainty_scale: LearnableParam,
    /// Weight for epistemic uncertainty
    pub epistemic_weight: LearnableParam,
    /// Bias for compute allocation
    pub compute_bias: LearnableParam,
    /// Early stopping threshold
    pub early_stop_threshold: LearnableParam,
}

impl LearnableMetaCognition {
    /// Create a new learnable meta-cognition module
    pub fn new() -> Self {
        Self {
            calibration_bias: LearnableParam::new(0.0),
            uncertainty_scale: LearnableParam::new(1.0),
            epistemic_weight: LearnableParam::new(0.5),
            compute_bias: LearnableParam::new(0.0),
            early_stop_threshold: LearnableParam::new(0.95),
        }
    }

    /// Calibrate a raw confidence score
    pub fn calibrate_confidence(&self, raw_confidence: f32) -> f32 {
        (raw_confidence + self.calibration_bias.value).clamp(0.0, 1.0)
    }

    /// Scale uncertainty estimate
    pub fn scale_uncertainty(&self, raw_uncertainty: f32) -> f32 {
        (raw_uncertainty * self.uncertainty_scale.value).clamp(0.0, 1.0)
    }

    /// Weight epistemic vs aleatoric uncertainty
    pub fn weighted_uncertainty(&self, epistemic: f32, aleatoric: f32) -> f32 {
        let w = self.epistemic_weight.value.clamp(0.0, 1.0);
        w * epistemic + (1.0 - w) * aleatoric
    }

    /// Compute adjusted compute budget
    pub fn adjusted_compute(&self, base_compute: usize) -> usize {
        let adjustment = 1.0 + self.compute_bias.value.clamp(-0.5, 0.5);
        ((base_compute as f32) * adjustment).max(1.0) as usize
    }

    /// Check if should early stop based on confidence
    pub fn should_early_stop(&self, confidence: f32) -> bool {
        confidence >= self.early_stop_threshold.value
    }
}

impl Default for LearnableMetaCognition {
    fn default() -> Self {
        Self::new()
    }
}

impl Learnable for LearnableMetaCognition {
    fn zero_grad(&mut self) {
        self.calibration_bias.zero_grad();
        self.uncertainty_scale.zero_grad();
        self.epistemic_weight.zero_grad();
        self.compute_bias.zero_grad();
        self.early_stop_threshold.zero_grad();
    }

    fn step(&mut self, lr: f32) {
        self.calibration_bias.step(lr);
        self.uncertainty_scale.step(lr);
        self.epistemic_weight.step(lr);
        self.compute_bias.step(lr);
        self.early_stop_threshold.step(lr);

        // Ensure valid ranges
        self.uncertainty_scale.value = self.uncertainty_scale.value.max(0.01);
        self.early_stop_threshold.value = self.early_stop_threshold.value.clamp(0.5, 1.0);
    }

    fn num_parameters(&self) -> usize {
        5
    }

    fn has_gradients(&self) -> bool {
        self.calibration_bias.grad != 0.0
            || self.uncertainty_scale.grad != 0.0
            || self.epistemic_weight.grad != 0.0
            || self.compute_bias.grad != 0.0
            || self.early_stop_threshold.grad != 0.0
    }

    fn gradient_norm(&self) -> f32 {
        (self.calibration_bias.grad.powi(2)
            + self.uncertainty_scale.grad.powi(2)
            + self.epistemic_weight.grad.powi(2)
            + self.compute_bias.grad.powi(2)
            + self.early_stop_threshold.grad.powi(2))
        .sqrt()
    }
}

// ============================================================================
// Brain-Aware Meta-Cognition
// ============================================================================

/// Domain-specific uncertainty adjustment based on available domain expertise
#[derive(Debug, Clone)]
pub struct DomainUncertaintyAdjustment {
    /// The domain ID
    pub domain_id: String,
    /// How much to reduce epistemic uncertainty (0.0-1.0)
    pub epistemic_reduction: f32,
    /// Confidence in this adjustment
    pub confidence: f32,
}

/// Result of brain-aware uncertainty estimation
#[derive(Debug)]
pub struct BrainAwareUncertaintyResult {
    /// Base uncertainty estimate
    pub base_uncertainty: UncertaintyEstimate,
    /// Domain-specific adjustments
    pub domain_adjustments: Vec<DomainUncertaintyAdjustment>,
    /// Adjusted uncertainty after domain expertise
    pub adjusted_uncertainty: UncertaintyEstimate,
    /// Which domain brains were consulted
    pub consulted_domains: Vec<String>,
}

/// Brain-aware meta-cognition that uses domain brains for better uncertainty estimation
pub struct BrainAwareMetaCognition {
    /// The underlying meta-cognition component
    pub meta: Box<dyn MetaCognition>,
    /// The cognitive-brain bridge for domain routing
    pub bridge: DefaultCognitiveBridge,
    /// Whether to use domain brains for uncertainty reduction
    pub use_domain_expertise: bool,
}

impl Debug for BrainAwareMetaCognition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrainAwareMetaCognition")
            .field("available_domains", &self.bridge.available_domains())
            .field("use_domain_expertise", &self.use_domain_expertise)
            .finish()
    }
}

impl BrainAwareMetaCognition {
    /// Create a new brain-aware meta-cognition
    pub fn new(meta: Box<dyn MetaCognition>) -> Self {
        Self {
            meta,
            bridge: DefaultCognitiveBridge::new(),
            use_domain_expertise: true,
        }
    }

    /// Create with a pre-configured bridge
    pub fn with_bridge(meta: Box<dyn MetaCognition>, bridge: DefaultCognitiveBridge) -> Self {
        Self {
            meta,
            bridge,
            use_domain_expertise: true,
        }
    }

    /// Register a domain brain
    pub fn register_brain(&mut self, brain: Box<dyn DomainBrain>) {
        self.bridge.register(brain);
    }

    /// Estimate uncertainty with domain brain awareness
    ///
    /// If a domain brain can handle the query, epistemic uncertainty may be reduced
    /// because we have specialized expertise for this type of query.
    pub fn estimate_uncertainty_with_domains(
        &self,
        query: &Graph,
        query_text: Option<&str>,
    ) -> BrainAwareUncertaintyResult {
        // Get base uncertainty
        let base = self.meta.estimate_uncertainty(query);

        // If we have text and should use domain expertise, check domain brains
        if self.use_domain_expertise {
            if let Some(text) = query_text {
                let routing = self.bridge.route_to_multiple_brains(text);
                if routing.success && !routing.results.is_empty() {
                    // Calculate uncertainty adjustments based on domain expertise
                    let mut adjustments = Vec::new();
                    let mut total_reduction = 0.0;

                    for result in &routing.results {
                        // Domain expertise reduces epistemic uncertainty
                        let reduction = result.confidence * 0.3; // Max 30% reduction per domain
                        adjustments.push(DomainUncertaintyAdjustment {
                            domain_id: result.domain_id.clone(),
                            epistemic_reduction: reduction,
                            confidence: result.confidence,
                        });
                        total_reduction += reduction;
                    }

                    // Cap total reduction at 50%
                    let reduction = total_reduction.min(0.5);
                    let adjusted_epistemic = base.epistemic * (1.0 - reduction);
                    let mut adjusted = UncertaintyEstimate::new(adjusted_epistemic, base.aleatoric);
                    adjusted.sources = base.sources.clone();

                    return BrainAwareUncertaintyResult {
                        base_uncertainty: base,
                        domain_adjustments: adjustments,
                        adjusted_uncertainty: adjusted,
                        consulted_domains: routing.domains().iter().map(|s| s.to_string()).collect(),
                    };
                }
            }
        }

        // No domain expertise available
        BrainAwareUncertaintyResult {
            base_uncertainty: base.clone(),
            domain_adjustments: Vec::new(),
            adjusted_uncertainty: base,
            consulted_domains: Vec::new(),
        }
    }

    /// Recognize limits with domain brain awareness
    ///
    /// If no domain brain can handle the query, that's a limitation.
    pub fn recognize_limits_with_domains(
        &self,
        task: &Graph,
        task_text: Option<&str>,
    ) -> Option<LimitationType> {
        // Check base meta-cognition first
        if let Some(limit) = self.meta.recognize_limits(task) {
            return Some(limit);
        }

        // Check if any domain brain can handle this
        if let Some(text) = task_text {
            let routing = self.bridge.route_to_multiple_brains(text);
            if !routing.success || routing.results.is_empty() {
                // No domain expertise for this query
                return Some(LimitationType::KnowledgeGap {
                    missing: format!("No domain expertise for: {}", text),
                });
            }
        }

        None
    }

    /// Get available domains
    pub fn available_domains(&self) -> Vec<String> {
        self.bridge.available_domains()
    }

    /// Check if a domain is available
    pub fn has_domain(&self, domain_id: &str) -> bool {
        self.bridge.has_domain(domain_id)
    }
}

impl CognitiveBrainBridge for BrainAwareMetaCognition {
    fn get_registry(&self) -> &BrainRegistry {
        self.bridge.get_registry()
    }

    fn get_registry_mut(&mut self) -> &mut BrainRegistry {
        self.bridge.get_registry_mut()
    }
}

/// Factory function to create brain-aware meta-cognition
pub fn create_brain_aware_metacognition() -> BrainAwareMetaCognition {
    BrainAwareMetaCognition::new(Box::new(SimpleMetaCognition::new()))
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
    fn test_uncertainty_estimate() {
        let u = UncertaintyEstimate::new(0.5, 0.2);
        assert!((u.epistemic - 0.5).abs() < 0.001);
        assert!((u.aleatoric - 0.2).abs() < 0.001);
        assert!(u.total > 0.5);
    }

    #[test]
    fn test_uncertainty_certain() {
        let u = UncertaintyEstimate::certain();
        assert_eq!(u.total, 0.0);
        assert!(u.is_confident(0.1));
    }

    #[test]
    fn test_uncertainty_sources() {
        let u = UncertaintyEstimate::uncertain(UncertaintySource::KnowledgeGap)
            .with_source(UncertaintySource::Ambiguity);
        assert_eq!(u.sources.len(), 2);
    }

    #[test]
    fn test_cognitive_state() {
        let state = CognitiveState::new().with_goal("test goal");
        assert_eq!(state.current_goal, Some("test goal".to_string()));
        assert!(!state.is_overloaded());
        assert!(!state.is_stuck(10));
    }

    #[test]
    fn test_cognitive_state_overload() {
        let mut state = CognitiveState::new();
        state.working_memory_load = 0.95;
        assert!(state.is_overloaded());
    }

    #[test]
    fn test_compute_budget() {
        let budget = ComputeBudget::standard();
        assert_eq!(budget.max_steps, 1000);

        let state = CognitiveState::new();
        assert!(!budget.is_exceeded(&state));
        assert_eq!(budget.remaining_steps(&state), 1000);
    }

    #[test]
    fn test_compute_budget_exceeded() {
        let budget = ComputeBudget::quick();
        let mut state = CognitiveState::new();
        state.steps_taken = 150;
        assert!(budget.is_exceeded(&state));
    }

    #[test]
    fn test_limitation_types() {
        let gap = LimitationType::KnowledgeGap {
            missing: "calculus".to_string(),
        };
        assert!(gap.description().contains("calculus"));

        let ambig = LimitationType::Ambiguity {
            alternatives: vec!["a".to_string(), "b".to_string()],
        };
        assert!(ambig.description().contains("2"));
    }

    #[test]
    fn test_compute_strategy() {
        let quick = ComputeStrategy::quick();
        assert_eq!(quick.strategy_type, StrategyType::Quick);
        assert!(quick.approximate);

        let deep = ComputeStrategy::deep();
        assert_eq!(deep.strategy_type, StrategyType::Deep);
        assert!(!deep.approximate);
    }

    #[test]
    fn test_contradiction() {
        let c = Contradiction::new("A != B but used interchangeably", vec![3, 7], 0.8);
        assert_eq!(c.conflicting_steps.len(), 2);
        assert!(c.severity > 0.7);
    }

    #[test]
    fn test_simple_metacognition() {
        let meta = SimpleMetaCognition::new();
        let graph = make_graph("test query");

        let uncertainty = meta.estimate_uncertainty(&graph);
        assert!(uncertainty.total >= 0.0 && uncertainty.total <= 1.0);
    }

    #[test]
    fn test_metacognition_introspect() {
        let meta = SimpleMetaCognition::new();
        let state = meta.introspect();
        assert_eq!(state.steps_taken, 0);
    }

    #[test]
    fn test_metacognition_allocate() {
        let meta = SimpleMetaCognition::new();
        let budget = ComputeBudget::standard();

        let small_task = make_graph("a");
        let strategy = meta.allocate_computation(&small_task, &budget);
        assert_eq!(strategy.strategy_type, StrategyType::Quick);
    }

    #[test]
    fn test_metacognition_should_continue() {
        let meta = SimpleMetaCognition::new();
        let budget = ComputeBudget::standard();
        let mut state = CognitiveState::new();
        state.confidence = 0.5; // Not confident yet

        assert!(meta.should_continue(&state, &budget));

        // Test that high confidence stops
        state.confidence = 1.0;
        assert!(!meta.should_continue(&state, &budget));
    }

    #[test]
    fn test_metacognition_calibrate() {
        let mut meta = SimpleMetaCognition::new();

        let predictions = vec![
            (0.8, true),
            (0.2, false),
            (0.6, true),
        ];

        meta.calibrate(&predictions);
        let error = meta.calibration_error();
        assert!((0.0..=1.0).contains(&error));
    }

    #[test]
    fn test_metacognition_limits() {
        let mut meta = SimpleMetaCognition::new();
        let mut state = CognitiveState::new();
        state.working_memory_load = 0.95;
        meta.update_state(state);

        let task = make_graph("task");
        let limit = meta.recognize_limits(&task);
        assert!(limit.is_some());
    }
}
