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
//! - **Safety Monitoring**: Continuous Asimov Laws compliance tracking
//!
//! ## Design Philosophy
//!
//! Meta-cognition is an open research problem. This implementation provides:
//! - Basic uncertainty quantification (epistemic vs aleatoric)
//! - Simple confidence calibration
//! - Compute budget management
//! - Contradiction detection hooks
//! - Safety awareness and violation monitoring
//!
//! ## Asimov Laws Integration
//!
//! Meta-cognition continuously monitors for potential safety violations:
//! - Tracks safety violation history
//! - Provides safety-aware introspection
//! - Alerts on approaching safety boundaries
//!
//! Real meta-cognition may require learned policies.

use grapheme_core::{
    BrainRegistry, CognitiveBrainBridge, DagNN, DefaultCognitiveBridge, DomainBrain, Learnable,
    LearnableParam, Persistable, PersistenceError,
};
use grapheme_reason::ReasoningStep;
use grapheme_safety::{SafetyGate, SafetyCheck, Action as SafetyAction, ActionTarget, ActionType};
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
    /// Safety violation count (for Asimov Laws monitoring)
    pub safety_violation_count: usize,
    /// Whether safety monitoring is active
    pub safety_monitoring_active: bool,
}

impl CognitiveState {
    /// Create new cognitive state with safety monitoring enabled by default
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
            safety_violation_count: 0,
            safety_monitoring_active: true, // Safety monitoring is ALWAYS active by default
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

    /// Check if any safety violations have occurred
    pub fn has_safety_violations(&self) -> bool {
        self.safety_violation_count > 0
    }

    /// Increment safety violation count (NON-OVERRIDABLE - only increases)
    pub fn record_safety_violation(&mut self) {
        self.safety_violation_count = self.safety_violation_count.saturating_add(1);
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
            self.calibration_data =
                self.calibration_data[self.calibration_data.len() - 1000..].to_vec();
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Persistable for LearnableMetaCognition {
    fn persist_type_id() -> &'static str {
        "LearnableMetaCognition"
    }

    fn persist_version() -> u32 {
        1
    }

    fn validate(&self) -> Result<(), PersistenceError> {
        // Validate uncertainty_scale is positive
        if self.uncertainty_scale.value <= 0.0 {
            return Err(PersistenceError::ValidationFailed(
                "Uncertainty scale must be positive".to_string(),
            ));
        }
        Ok(())
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
                        consulted_domains: routing
                            .domains()
                            .iter()
                            .map(|s| s.to_string())
                            .collect(),
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
// SelfModel: Graph Representation of System State
// ============================================================================

/// Component type in the self-model
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    /// A domain brain (math, code, vision, etc.)
    Brain(String),
    /// A cognitive module (memory, reasoning, etc.)
    CognitiveModule(String),
    /// An orchestrator
    Orchestrator,
    /// A resource pool (memory, compute)
    Resource(String),
    /// A communication channel
    Channel(String),
}

/// State of a component
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum ComponentState {
    /// Component is idle and ready
    #[default]
    Idle,
    /// Component is actively processing
    Active,
    /// Component is busy and cannot accept new work
    Busy,
    /// Component is overloaded
    Overloaded,
    /// Component has an error
    Error(String),
    /// Component is disabled
    Disabled,
}

/// A component in the self-model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModelComponent {
    /// Unique identifier
    pub id: String,
    /// Component type
    pub component_type: ComponentType,
    /// Current state
    pub state: ComponentState,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Load factor (0.0 to 1.0)
    pub load: f32,
    /// Number of operations completed
    pub operations_completed: u64,
    /// Last update timestamp (as epoch millis)
    pub last_updated: u64,
}

impl SelfModelComponent {
    /// Create a new component
    pub fn new(id: impl Into<String>, component_type: ComponentType) -> Self {
        Self {
            id: id.into(),
            component_type,
            state: ComponentState::Idle,
            confidence: 1.0,
            load: 0.0,
            operations_completed: 0,
            last_updated: 0,
        }
    }

    /// Check if component is available for work
    pub fn is_available(&self) -> bool {
        matches!(self.state, ComponentState::Idle | ComponentState::Active)
            && self.load < 0.9
    }

    /// Update state
    pub fn set_state(&mut self, state: ComponentState) {
        self.state = state;
    }

    /// Record an operation
    pub fn record_operation(&mut self) {
        self.operations_completed += 1;
    }
}

/// Connection between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentConnection {
    /// Source component ID
    pub source: String,
    /// Target component ID
    pub target: String,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Weight/strength of connection (0.0 to 1.0)
    pub weight: f32,
    /// Is connection active?
    pub active: bool,
}

/// Type of connection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Data flows from source to target
    DataFlow,
    /// Control/command relationship
    Control,
    /// Bidirectional communication
    Bidirectional,
    /// Monitoring/observation
    Monitoring,
}

/// SelfModel: Graph representation of system's own state
///
/// The self-model maintains a graph representation of all system components
/// and their relationships. This enables introspection, self-monitoring,
/// and meta-level reasoning about the system's capabilities.
///
/// # Architecture
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                           SelfModel Graph                                │
/// │                                                                          │
/// │  [Orchestrator] ─── Control ───► [MathBrain]                            │
/// │       │                              │                                   │
/// │       │─── Control ───► [CodeBrain] ─┤                                  │
/// │       │                              │ DataFlow                          │
/// │       └─── Control ───► [VisionBrain]│                                  │
/// │                              │       ▼                                   │
/// │  [Memory] ◄─── DataFlow ─────┴── [Reasoning]                            │
/// │                                                                          │
/// └─────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Time Complexity
/// - Component lookup: O(1) via HashMap
/// - Add/remove component: O(1)
/// - Get all connections: O(n) where n = number of connections
/// - State update: O(1)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelfModel {
    /// All components indexed by ID
    components: std::collections::HashMap<String, SelfModelComponent>,
    /// All connections
    connections: Vec<ComponentConnection>,
    /// Overall system confidence
    pub system_confidence: f32,
    /// Overall system load
    pub system_load: f32,
    /// Total operations across all components
    pub total_operations: u64,
}

impl SelfModel {
    /// Create a new empty self-model
    pub fn new() -> Self {
        Self {
            components: std::collections::HashMap::new(),
            connections: Vec::new(),
            system_confidence: 1.0,
            system_load: 0.0,
            total_operations: 0,
        }
    }

    /// Add a component to the self-model - O(1)
    pub fn add_component(&mut self, component: SelfModelComponent) {
        self.components.insert(component.id.clone(), component);
    }

    /// Remove a component - O(n) due to connection cleanup
    pub fn remove_component(&mut self, id: &str) -> Option<SelfModelComponent> {
        // Remove connections involving this component
        self.connections.retain(|c| c.source != id && c.target != id);
        self.components.remove(id)
    }

    /// Get a component by ID - O(1)
    pub fn get_component(&self, id: &str) -> Option<&SelfModelComponent> {
        self.components.get(id)
    }

    /// Get mutable component by ID - O(1)
    pub fn get_component_mut(&mut self, id: &str) -> Option<&mut SelfModelComponent> {
        self.components.get_mut(id)
    }

    /// Add a connection between components - O(1)
    pub fn add_connection(&mut self, connection: ComponentConnection) {
        self.connections.push(connection);
    }

    /// Get all connections from a component - O(n)
    pub fn connections_from(&self, source_id: &str) -> Vec<&ComponentConnection> {
        self.connections.iter()
            .filter(|c| c.source == source_id)
            .collect()
    }

    /// Get all connections to a component - O(n)
    pub fn connections_to(&self, target_id: &str) -> Vec<&ComponentConnection> {
        self.connections.iter()
            .filter(|c| c.target == target_id)
            .collect()
    }

    /// Get all components - O(1) to get iterator
    pub fn all_components(&self) -> impl Iterator<Item = &SelfModelComponent> {
        self.components.values()
    }

    /// Get all component IDs - O(n)
    pub fn component_ids(&self) -> Vec<String> {
        self.components.keys().cloned().collect()
    }

    /// Get components by type - O(n)
    pub fn components_by_type(&self, component_type: &ComponentType) -> Vec<&SelfModelComponent> {
        self.components.values()
            .filter(|c| &c.component_type == component_type)
            .collect()
    }

    /// Get all brain components - O(n)
    pub fn get_brains(&self) -> Vec<&SelfModelComponent> {
        self.components.values()
            .filter(|c| matches!(c.component_type, ComponentType::Brain(_)))
            .collect()
    }

    /// Update component state - O(1)
    pub fn update_state(&mut self, id: &str, state: ComponentState) -> bool {
        if let Some(component) = self.components.get_mut(id) {
            component.state = state;
            true
        } else {
            false
        }
    }

    /// Update component load - O(1)
    pub fn update_load(&mut self, id: &str, load: f32) -> bool {
        if let Some(component) = self.components.get_mut(id) {
            component.load = load.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Record an operation for a component - O(1)
    pub fn record_operation(&mut self, id: &str) -> bool {
        if let Some(component) = self.components.get_mut(id) {
            component.record_operation();
            self.total_operations += 1;
            true
        } else {
            false
        }
    }

    /// Recalculate system-wide metrics - O(n)
    pub fn recalculate_system_metrics(&mut self) {
        if self.components.is_empty() {
            self.system_confidence = 1.0;
            self.system_load = 0.0;
            return;
        }

        let mut total_confidence = 0.0;
        let mut total_load = 0.0;
        let count = self.components.len() as f32;

        for component in self.components.values() {
            total_confidence += component.confidence;
            total_load += component.load;
        }

        self.system_confidence = total_confidence / count;
        self.system_load = total_load / count;
    }

    /// Get available components for work - O(n)
    pub fn available_components(&self) -> Vec<&SelfModelComponent> {
        self.components.values()
            .filter(|c| c.is_available())
            .collect()
    }

    /// Get components with errors - O(n)
    pub fn error_components(&self) -> Vec<&SelfModelComponent> {
        self.components.values()
            .filter(|c| matches!(c.state, ComponentState::Error(_)))
            .collect()
    }

    /// Number of components - O(1)
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Number of connections - O(1)
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Check if system is healthy (all components available, no errors) - O(n)
    pub fn is_healthy(&self) -> bool {
        self.error_components().is_empty() && self.system_load < 0.9
    }

    /// Create a standard self-model with common GRAPHEME components
    pub fn create_standard() -> Self {
        let mut model = Self::new();

        // Add orchestrator
        model.add_component(SelfModelComponent::new(
            "orchestrator",
            ComponentType::Orchestrator,
        ));

        // Add standard domain brains
        for brain_name in ["math", "code", "vision", "music", "chem", "law"] {
            model.add_component(SelfModelComponent::new(
                brain_name,
                ComponentType::Brain(brain_name.to_string()),
            ));
            // Connect orchestrator to brain
            model.add_connection(ComponentConnection {
                source: "orchestrator".to_string(),
                target: brain_name.to_string(),
                connection_type: ConnectionType::Control,
                weight: 1.0,
                active: true,
            });
        }

        // Add cognitive modules
        for module_name in ["memory", "reasoning", "metacognition", "grounding"] {
            model.add_component(SelfModelComponent::new(
                module_name,
                ComponentType::CognitiveModule(module_name.to_string()),
            ));
        }

        // Add connections between cognitive modules
        model.add_connection(ComponentConnection {
            source: "reasoning".to_string(),
            target: "memory".to_string(),
            connection_type: ConnectionType::DataFlow,
            weight: 1.0,
            active: true,
        });

        model.add_connection(ComponentConnection {
            source: "metacognition".to_string(),
            target: "reasoning".to_string(),
            connection_type: ConnectionType::Monitoring,
            weight: 1.0,
            active: true,
        });

        model.recalculate_system_metrics();
        model
    }
}

/// Factory function to create a standard self-model
pub fn create_self_model() -> SelfModel {
    SelfModel::create_standard()
}

// ============================================================================
// ReflectiveBrain: Introspection and Self-Modification
// ============================================================================

/// Result of an introspection operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrospectionResult {
    /// Current system state summary
    pub state_summary: String,
    /// List of active components
    pub active_components: Vec<String>,
    /// Components with issues
    pub problematic_components: Vec<String>,
    /// Suggested improvements
    pub suggestions: Vec<ModificationSuggestion>,
    /// Overall health score (0.0 to 1.0)
    pub health_score: f32,
    /// Timestamp of introspection
    pub timestamp: u64,
}

impl IntrospectionResult {
    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        self.health_score > 0.7 && self.problematic_components.is_empty()
    }
}

/// A suggestion for self-modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationSuggestion {
    /// Type of modification
    pub modification_type: ModificationType,
    /// Target component ID
    pub target: String,
    /// Description of the change
    pub description: String,
    /// Priority (higher = more important)
    pub priority: u8,
    /// Estimated impact on performance
    pub impact_score: f32,
}

/// Types of self-modification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModificationType {
    /// Adjust a parameter value
    AdjustParameter,
    /// Enable a disabled component
    EnableComponent,
    /// Disable a problematic component
    DisableComponent,
    /// Reset a component to defaults
    ResetComponent,
    /// Reallocate resources
    ReallocateResources,
    /// Clear cached state
    ClearCache,
}

/// History entry for self-modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationHistory {
    /// Timestamp of modification
    pub timestamp: u64,
    /// Type of modification performed
    pub modification_type: ModificationType,
    /// Target component
    pub target: String,
    /// Whether modification was successful
    pub success: bool,
    /// Reason for modification
    pub reason: String,
}

/// ReflectiveBrain: Enables introspection and self-modification
///
/// The ReflectiveBrain monitors the system's SelfModel and can:
/// - Perform introspection to understand current state
/// - Identify performance issues and errors
/// - Suggest and apply modifications to improve performance
/// - Track history of all self-modifications
///
/// # Time Complexity
/// - Introspect: O(n) where n = number of components
/// - Apply modification: O(1) per modification
/// - Get history: O(1)
///
/// # Safety
/// All modifications are logged and can be rolled back through the history.
#[derive(Debug, Clone)]
pub struct ReflectiveBrain {
    /// The self-model to reflect on
    self_model: SelfModel,
    /// History of modifications
    modification_history: Vec<ModificationHistory>,
    /// Configuration
    config: ReflectiveConfig,
    /// Number of introspections performed
    introspection_count: u64,
}

/// Configuration for ReflectiveBrain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectiveConfig {
    /// Threshold for flagging high load components
    pub high_load_threshold: f32,
    /// Threshold for flagging low confidence components
    pub low_confidence_threshold: f32,
    /// Maximum modification history size
    pub max_history_size: usize,
    /// Whether to auto-apply safe modifications
    pub auto_apply_safe: bool,
}

impl Default for ReflectiveConfig {
    fn default() -> Self {
        Self {
            high_load_threshold: 0.8,
            low_confidence_threshold: 0.5,
            max_history_size: 1000,
            auto_apply_safe: false,
        }
    }
}

impl ReflectiveBrain {
    /// Create a new ReflectiveBrain with a standard SelfModel
    pub fn new() -> Self {
        Self {
            self_model: SelfModel::create_standard(),
            modification_history: Vec::new(),
            config: ReflectiveConfig::default(),
            introspection_count: 0,
        }
    }

    /// Create with custom config
    pub fn with_config(config: ReflectiveConfig) -> Self {
        Self {
            self_model: SelfModel::create_standard(),
            modification_history: Vec::new(),
            config,
            introspection_count: 0,
        }
    }

    /// Create with existing SelfModel
    pub fn with_self_model(self_model: SelfModel) -> Self {
        Self {
            self_model,
            modification_history: Vec::new(),
            config: ReflectiveConfig::default(),
            introspection_count: 0,
        }
    }

    /// Get a reference to the self-model
    pub fn self_model(&self) -> &SelfModel {
        &self.self_model
    }

    /// Get mutable reference to the self-model
    pub fn self_model_mut(&mut self) -> &mut SelfModel {
        &mut self.self_model
    }

    /// Perform introspection - analyze current system state - O(n)
    pub fn introspect(&mut self) -> IntrospectionResult {
        self.introspection_count += 1;
        self.self_model.recalculate_system_metrics();

        // Collect active components
        let active_components: Vec<String> = self.self_model
            .all_components()
            .filter(|c| matches!(c.state, ComponentState::Active))
            .map(|c| c.id.clone())
            .collect();

        // Identify problematic components - O(n)
        let problematic_components: Vec<String> = self.self_model
            .all_components()
            .filter(|c| {
                matches!(c.state, ComponentState::Error(_))
                    || c.load > self.config.high_load_threshold
                    || c.confidence < self.config.low_confidence_threshold
            })
            .map(|c| c.id.clone())
            .collect();

        // Generate suggestions based on problems found - O(n)
        let mut suggestions = Vec::new();

        for component in self.self_model.all_components() {
            // Suggest disabling errored components
            if let ComponentState::Error(_) = &component.state {
                suggestions.push(ModificationSuggestion {
                    modification_type: ModificationType::DisableComponent,
                    target: component.id.clone(),
                    description: "Component has error state".to_string(),
                    priority: 9,
                    impact_score: 0.8,
                });
            }

            // Suggest resource reallocation for high-load components
            if component.load > self.config.high_load_threshold {
                suggestions.push(ModificationSuggestion {
                    modification_type: ModificationType::ReallocateResources,
                    target: component.id.clone(),
                    description: format!("High load: {:.0}%", component.load * 100.0),
                    priority: 7,
                    impact_score: 0.5,
                });
            }

            // Suggest reset for low confidence components
            if component.confidence < self.config.low_confidence_threshold {
                suggestions.push(ModificationSuggestion {
                    modification_type: ModificationType::ResetComponent,
                    target: component.id.clone(),
                    description: format!("Low confidence: {:.0}%", component.confidence * 100.0),
                    priority: 5,
                    impact_score: 0.3,
                });
            }
        }

        // Sort suggestions by priority
        suggestions.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Calculate health score
        let error_count = self.self_model.error_components().len();
        let total_count = self.self_model.component_count().max(1);
        let error_penalty = (error_count as f32 / total_count as f32) * 0.5;
        let load_penalty = (self.self_model.system_load - 0.5).max(0.0) * 0.5;
        let health_score = (1.0 - error_penalty - load_penalty).clamp(0.0, 1.0);

        // Build state summary
        let state_summary = format!(
            "Components: {}, Active: {}, Errors: {}, Load: {:.0}%, Confidence: {:.0}%",
            total_count,
            active_components.len(),
            error_count,
            self.self_model.system_load * 100.0,
            self.self_model.system_confidence * 100.0
        );

        IntrospectionResult {
            state_summary,
            active_components,
            problematic_components,
            suggestions,
            health_score,
            timestamp: 0, // Would use real timestamp in production
        }
    }

    /// Apply a modification to the system - O(1)
    pub fn apply_modification(&mut self, suggestion: &ModificationSuggestion) -> bool {
        let success = match suggestion.modification_type {
            ModificationType::DisableComponent => {
                self.self_model.update_state(&suggestion.target, ComponentState::Disabled)
            }
            ModificationType::EnableComponent => {
                self.self_model.update_state(&suggestion.target, ComponentState::Idle)
            }
            ModificationType::ResetComponent => {
                if let Some(component) = self.self_model.get_component_mut(&suggestion.target) {
                    component.state = ComponentState::Idle;
                    component.load = 0.0;
                    component.confidence = 1.0;
                    true
                } else {
                    false
                }
            }
            ModificationType::ReallocateResources => {
                // Reduce load by half (simulated reallocation)
                if let Some(component) = self.self_model.get_component_mut(&suggestion.target) {
                    component.load = (component.load * 0.5).max(0.0);
                    true
                } else {
                    false
                }
            }
            ModificationType::ClearCache => {
                // Reset load (simulated cache clear)
                self.self_model.update_load(&suggestion.target, 0.0)
            }
            ModificationType::AdjustParameter => {
                // Parameter adjustment would require specific implementation
                true
            }
        };

        // Log the modification
        self.modification_history.push(ModificationHistory {
            timestamp: 0, // Would use real timestamp in production
            modification_type: suggestion.modification_type.clone(),
            target: suggestion.target.clone(),
            success,
            reason: suggestion.description.clone(),
        });

        // Trim history if needed
        if self.modification_history.len() > self.config.max_history_size {
            self.modification_history.remove(0);
        }

        success
    }

    /// Get modification history
    pub fn history(&self) -> &[ModificationHistory] {
        &self.modification_history
    }

    /// Get number of introspections performed
    pub fn introspection_count(&self) -> u64 {
        self.introspection_count
    }

    /// Apply all high-priority suggestions automatically
    pub fn auto_heal(&mut self) -> Vec<ModificationHistory> {
        let introspection = self.introspect();
        let mut applied = Vec::new();

        for suggestion in introspection.suggestions.iter().filter(|s| s.priority >= 8) {
            if self.apply_modification(suggestion) {
                if let Some(history) = self.modification_history.last() {
                    applied.push(history.clone());
                }
            }
        }

        applied
    }

    /// Check if system needs attention
    pub fn needs_attention(&mut self) -> bool {
        let introspection = self.introspect();
        !introspection.is_healthy()
    }
}

impl Default for ReflectiveBrain {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory function to create a reflective brain
pub fn create_reflective_brain() -> ReflectiveBrain {
    ReflectiveBrain::new()
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

        let predictions = vec![(0.8, true), (0.2, false), (0.6, true)];

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

    // ========================================================================
    // SelfModel Tests
    // ========================================================================

    #[test]
    fn test_self_model_creation() {
        let model = SelfModel::new();
        assert_eq!(model.component_count(), 0);
        assert_eq!(model.connection_count(), 0);
        assert!((model.system_confidence - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_self_model_add_component() {
        let mut model = SelfModel::new();
        let component = SelfModelComponent::new("test", ComponentType::Brain("test".to_string()));
        model.add_component(component);

        assert_eq!(model.component_count(), 1);
        assert!(model.get_component("test").is_some());
    }

    #[test]
    fn test_self_model_remove_component() {
        let mut model = SelfModel::new();
        model.add_component(SelfModelComponent::new("a", ComponentType::Orchestrator));
        model.add_component(SelfModelComponent::new("b", ComponentType::Brain("b".to_string())));
        model.add_connection(ComponentConnection {
            source: "a".to_string(),
            target: "b".to_string(),
            connection_type: ConnectionType::Control,
            weight: 1.0,
            active: true,
        });

        assert_eq!(model.connection_count(), 1);
        model.remove_component("b");
        assert_eq!(model.component_count(), 1);
        assert_eq!(model.connection_count(), 0); // Connection should be removed
    }

    #[test]
    fn test_self_model_update_state() {
        let mut model = SelfModel::new();
        model.add_component(SelfModelComponent::new("brain", ComponentType::Brain("brain".to_string())));

        assert!(model.update_state("brain", ComponentState::Active));
        assert_eq!(model.get_component("brain").unwrap().state, ComponentState::Active);

        assert!(!model.update_state("nonexistent", ComponentState::Error("test".to_string())));
    }

    #[test]
    fn test_self_model_connections() {
        let mut model = SelfModel::new();
        model.add_component(SelfModelComponent::new("orchestrator", ComponentType::Orchestrator));
        model.add_component(SelfModelComponent::new("math", ComponentType::Brain("math".to_string())));
        model.add_component(SelfModelComponent::new("code", ComponentType::Brain("code".to_string())));

        model.add_connection(ComponentConnection {
            source: "orchestrator".to_string(),
            target: "math".to_string(),
            connection_type: ConnectionType::Control,
            weight: 1.0,
            active: true,
        });
        model.add_connection(ComponentConnection {
            source: "orchestrator".to_string(),
            target: "code".to_string(),
            connection_type: ConnectionType::Control,
            weight: 1.0,
            active: true,
        });

        let from_orchestrator = model.connections_from("orchestrator");
        assert_eq!(from_orchestrator.len(), 2);

        let to_math = model.connections_to("math");
        assert_eq!(to_math.len(), 1);
    }

    #[test]
    fn test_self_model_record_operation() {
        let mut model = SelfModel::new();
        model.add_component(SelfModelComponent::new("brain", ComponentType::Brain("brain".to_string())));

        assert!(model.record_operation("brain"));
        assert_eq!(model.get_component("brain").unwrap().operations_completed, 1);
        assert_eq!(model.total_operations, 1);
    }

    #[test]
    fn test_self_model_system_metrics() {
        let mut model = SelfModel::new();

        // Add components with different loads
        let mut c1 = SelfModelComponent::new("a", ComponentType::Orchestrator);
        c1.load = 0.2;
        c1.confidence = 0.8;
        model.add_component(c1);

        let mut c2 = SelfModelComponent::new("b", ComponentType::Brain("b".to_string()));
        c2.load = 0.6;
        c2.confidence = 1.0;
        model.add_component(c2);

        model.recalculate_system_metrics();

        // Average load should be 0.4
        assert!((model.system_load - 0.4).abs() < 0.001);
        // Average confidence should be 0.9
        assert!((model.system_confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_self_model_standard() {
        let model = SelfModel::create_standard();

        // Should have 11 components: 1 orchestrator + 6 brains + 4 cognitive modules
        assert_eq!(model.component_count(), 11);

        // Should have 8 connections: 6 orchestrator->brain + 2 cognitive module connections
        assert_eq!(model.connection_count(), 8);

        // All brains should be accessible
        assert!(model.get_component("math").is_some());
        assert!(model.get_component("vision").is_some());
        assert!(model.get_component("code").is_some());

        // System should be healthy
        assert!(model.is_healthy());
    }

    #[test]
    fn test_self_model_get_brains() {
        let model = SelfModel::create_standard();
        let brains = model.get_brains();
        assert_eq!(brains.len(), 6);
    }

    #[test]
    fn test_self_model_available_components() {
        let mut model = SelfModel::new();

        let mut c1 = SelfModelComponent::new("a", ComponentType::Orchestrator);
        c1.state = ComponentState::Idle;
        c1.load = 0.3;
        model.add_component(c1);

        let mut c2 = SelfModelComponent::new("b", ComponentType::Brain("b".to_string()));
        c2.state = ComponentState::Busy;
        model.add_component(c2);

        let available = model.available_components();
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].id, "a");
    }

    #[test]
    fn test_self_model_error_detection() {
        let mut model = SelfModel::new();
        model.add_component(SelfModelComponent::new("good", ComponentType::Orchestrator));

        let mut bad = SelfModelComponent::new("bad", ComponentType::Brain("bad".to_string()));
        bad.state = ComponentState::Error("test error".to_string());
        model.add_component(bad);

        let errors = model.error_components();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].id, "bad");

        assert!(!model.is_healthy());
    }

    #[test]
    fn test_component_state_default() {
        let state = ComponentState::default();
        assert_eq!(state, ComponentState::Idle);
    }

    #[test]
    fn test_create_self_model_factory() {
        let model = create_self_model();
        assert_eq!(model.component_count(), 11);
    }

    // ========================================================================
    // ReflectiveBrain Tests
    // ========================================================================

    #[test]
    fn test_reflective_brain_creation() {
        let brain = ReflectiveBrain::new();
        assert_eq!(brain.introspection_count(), 0);
        assert!(brain.history().is_empty());
    }

    #[test]
    fn test_reflective_brain_introspection() {
        let mut brain = ReflectiveBrain::new();
        let result = brain.introspect();

        assert_eq!(brain.introspection_count(), 1);
        assert!(result.health_score > 0.0);
        assert!(!result.state_summary.is_empty());
    }

    #[test]
    fn test_reflective_brain_healthy_system() {
        let mut brain = ReflectiveBrain::new();
        let result = brain.introspect();

        // Standard system should be healthy
        assert!(result.is_healthy());
        assert!(result.problematic_components.is_empty());
    }

    #[test]
    fn test_reflective_brain_detect_errors() {
        let mut model = SelfModel::new();
        model.add_component(SelfModelComponent::new("good", ComponentType::Orchestrator));

        let mut bad = SelfModelComponent::new("bad", ComponentType::Brain("bad".to_string()));
        bad.state = ComponentState::Error("test error".to_string());
        model.add_component(bad);

        let mut brain = ReflectiveBrain::with_self_model(model);
        let result = brain.introspect();

        assert!(!result.is_healthy());
        assert!(result.problematic_components.contains(&"bad".to_string()));
        assert!(!result.suggestions.is_empty());
        assert_eq!(result.suggestions[0].modification_type, ModificationType::DisableComponent);
    }

    #[test]
    fn test_reflective_brain_detect_high_load() {
        let mut model = SelfModel::new();
        let mut high_load = SelfModelComponent::new("busy", ComponentType::Brain("busy".to_string()));
        high_load.load = 0.95;
        model.add_component(high_load);

        let mut brain = ReflectiveBrain::with_self_model(model);
        let result = brain.introspect();

        assert!(result.problematic_components.contains(&"busy".to_string()));
        assert!(result.suggestions.iter().any(|s| s.modification_type == ModificationType::ReallocateResources));
    }

    #[test]
    fn test_reflective_brain_apply_disable() {
        let mut model = SelfModel::new();
        let mut bad = SelfModelComponent::new("broken", ComponentType::Brain("broken".to_string()));
        bad.state = ComponentState::Error("error".to_string());
        model.add_component(bad);

        let mut brain = ReflectiveBrain::with_self_model(model);

        let suggestion = ModificationSuggestion {
            modification_type: ModificationType::DisableComponent,
            target: "broken".to_string(),
            description: "Disable broken component".to_string(),
            priority: 9,
            impact_score: 0.8,
        };

        assert!(brain.apply_modification(&suggestion));
        assert_eq!(brain.history().len(), 1);
        assert!(brain.history()[0].success);

        let component = brain.self_model().get_component("broken").unwrap();
        assert_eq!(component.state, ComponentState::Disabled);
    }

    #[test]
    fn test_reflective_brain_apply_reset() {
        let mut model = SelfModel::new();
        let mut degraded = SelfModelComponent::new("degraded", ComponentType::Brain("degraded".to_string()));
        degraded.confidence = 0.3;
        degraded.load = 0.8;
        model.add_component(degraded);

        let mut brain = ReflectiveBrain::with_self_model(model);

        let suggestion = ModificationSuggestion {
            modification_type: ModificationType::ResetComponent,
            target: "degraded".to_string(),
            description: "Reset degraded component".to_string(),
            priority: 5,
            impact_score: 0.3,
        };

        assert!(brain.apply_modification(&suggestion));

        let component = brain.self_model().get_component("degraded").unwrap();
        assert_eq!(component.state, ComponentState::Idle);
        assert!((component.confidence - 1.0).abs() < 0.001);
        assert!((component.load - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_reflective_brain_auto_heal() {
        let mut model = SelfModel::new();

        // Add an errored component (should trigger auto-heal)
        let mut bad = SelfModelComponent::new("error_brain", ComponentType::Brain("error".to_string()));
        bad.state = ComponentState::Error("critical error".to_string());
        model.add_component(bad);

        // Add a good component
        model.add_component(SelfModelComponent::new("good", ComponentType::Orchestrator));

        let mut brain = ReflectiveBrain::with_self_model(model);
        let applied = brain.auto_heal();

        // Should have applied at least one modification (disable the errored component)
        assert!(!applied.is_empty());
        assert!(applied[0].success);
        assert_eq!(applied[0].modification_type, ModificationType::DisableComponent);
    }

    #[test]
    fn test_reflective_brain_needs_attention() {
        // Healthy system
        let mut brain = ReflectiveBrain::new();
        assert!(!brain.needs_attention());

        // System with error
        let mut model = SelfModel::new();
        let mut bad = SelfModelComponent::new("bad", ComponentType::Brain("bad".to_string()));
        bad.state = ComponentState::Error("error".to_string());
        model.add_component(bad);

        let mut brain_with_error = ReflectiveBrain::with_self_model(model);
        assert!(brain_with_error.needs_attention());
    }

    #[test]
    fn test_reflective_brain_history_limit() {
        let config = ReflectiveConfig {
            max_history_size: 3,
            ..Default::default()
        };
        let mut brain = ReflectiveBrain::with_config(config);

        // Apply 5 modifications
        for i in 0..5 {
            let suggestion = ModificationSuggestion {
                modification_type: ModificationType::AdjustParameter,
                target: format!("component_{}", i),
                description: format!("Adjustment {}", i),
                priority: 1,
                impact_score: 0.1,
            };
            brain.apply_modification(&suggestion);
        }

        // History should be limited to max_history_size
        assert_eq!(brain.history().len(), 3);
    }

    #[test]
    fn test_create_reflective_brain_factory() {
        let brain = create_reflective_brain();
        assert_eq!(brain.introspection_count(), 0);
        assert_eq!(brain.self_model().component_count(), 11);
    }

    #[test]
    fn test_introspection_result_healthy() {
        let result = IntrospectionResult {
            state_summary: "Test".to_string(),
            active_components: vec![],
            problematic_components: vec![],
            suggestions: vec![],
            health_score: 0.9,
            timestamp: 0,
        };
        assert!(result.is_healthy());

        let unhealthy = IntrospectionResult {
            health_score: 0.5,
            problematic_components: vec!["bad".to_string()],
            ..result.clone()
        };
        assert!(!unhealthy.is_healthy());
    }
}

// ============================================================================
// AttentionMechanism: Dynamic Focus Allocation Across Brains
// ============================================================================

/// Attention weight for a specific brain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainAttention {
    /// Brain identifier
    pub brain_id: String,
    /// Attention weight (0.0 to 1.0)
    pub weight: f32,
    /// Relevance score to current input
    pub relevance: f32,
    /// Current capacity (inverse of load)
    pub capacity: f32,
    /// Whether this brain should be actively engaged
    pub should_engage: bool,
}

impl BrainAttention {
    /// Create a new brain attention entry
    pub fn new(brain_id: impl Into<String>) -> Self {
        Self {
            brain_id: brain_id.into(),
            weight: 0.0,
            relevance: 0.0,
            capacity: 1.0,
            should_engage: false,
        }
    }

    /// Compute combined score for ranking - O(1)
    pub fn combined_score(&self) -> f32 {
        // Weight = relevance * capacity * attention_weight
        self.relevance * self.capacity * self.weight
    }
}

/// Configuration for attention allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Maximum number of brains to engage simultaneously
    pub max_parallel_brains: usize,
    /// Minimum relevance threshold to engage a brain
    pub min_relevance_threshold: f32,
    /// Whether to use soft attention (all brains with weights) or hard (select top-k)
    pub use_soft_attention: bool,
    /// Temperature for softmax attention computation
    pub temperature: f32,
    /// Weight decay factor for brains that repeatedly fail
    pub failure_decay: f32,
    /// Boost factor for recently successful brains
    pub success_boost: f32,
    /// Maximum attention weight any single brain can have
    pub max_single_weight: f32,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            max_parallel_brains: 3,
            min_relevance_threshold: 0.1,
            use_soft_attention: true,
            temperature: 1.0,
            failure_decay: 0.9,
            success_boost: 1.1,
            max_single_weight: 0.8,
        }
    }
}

/// Result of attention allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionAllocation {
    /// Attention weights for each brain (sorted by weight descending)
    pub brain_weights: Vec<BrainAttention>,
    /// Total attention budget used (sum of engaged brain weights)
    pub total_attention: f32,
    /// Number of brains selected for engagement
    pub engaged_count: usize,
    /// The input characteristics that drove this allocation
    pub input_features: InputFeatures,
}

impl AttentionAllocation {
    /// Get the top-k brains to engage - O(1) since already sorted
    pub fn top_k(&self, k: usize) -> Vec<&BrainAttention> {
        self.brain_weights.iter()
            .filter(|b| b.should_engage)
            .take(k)
            .collect()
    }

    /// Get brains above a weight threshold - O(n)
    pub fn above_threshold(&self, threshold: f32) -> Vec<&BrainAttention> {
        self.brain_weights.iter()
            .filter(|b| b.weight >= threshold)
            .collect()
    }

    /// Get the primary brain (highest weight) - O(1)
    pub fn primary(&self) -> Option<&BrainAttention> {
        self.brain_weights.first().filter(|b| b.should_engage)
    }
}

/// Features extracted from input to drive attention allocation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputFeatures {
    /// Detected domain hints (e.g., "math", "code", "visual")
    pub domain_hints: Vec<String>,
    /// Estimated complexity (0.0 to 1.0)
    pub complexity: f32,
    /// Estimated urgency (0.0 to 1.0)
    pub urgency: f32,
    /// Whether input requires multi-modal processing
    pub is_multimodal: bool,
    /// Specific keywords detected
    pub keywords: Vec<String>,
}

impl InputFeatures {
    /// Create from simple text analysis - O(n) where n = input length
    pub fn from_text(input: &str) -> Self {
        let lower = input.to_lowercase();
        let mut domain_hints = Vec::new();
        let mut keywords = Vec::new();

        // Simple keyword-based domain detection
        // Math hints
        if lower.contains('+') || lower.contains('-') || lower.contains('*') || lower.contains('/')
            || lower.contains("calculate") || lower.contains("compute") || lower.contains("solve")
            || lower.contains("equation") || lower.contains("formula") {
            domain_hints.push("math".to_string());
            keywords.push("mathematical".to_string());
        }

        // Code hints
        if lower.contains("function") || lower.contains("def ") || lower.contains("fn ")
            || lower.contains("class") || lower.contains("import") || lower.contains("return")
            || lower.contains("code") || lower.contains("program") {
            domain_hints.push("code".to_string());
            keywords.push("programming".to_string());
        }

        // Music hints
        if lower.contains("note") || lower.contains("chord") || lower.contains("melody")
            || lower.contains("music") || lower.contains("tempo") || lower.contains("rhythm") {
            domain_hints.push("music".to_string());
            keywords.push("musical".to_string());
        }

        // Chemistry hints
        if lower.contains("molecule") || lower.contains("atom") || lower.contains("chemical")
            || lower.contains("reaction") || lower.contains("compound") || lower.contains("element") {
            domain_hints.push("chem".to_string());
            keywords.push("chemistry".to_string());
        }

        // Law hints
        if lower.contains("legal") || lower.contains("law") || lower.contains("statute")
            || lower.contains("court") || lower.contains("contract") || lower.contains("regulation") {
            domain_hints.push("law".to_string());
            keywords.push("legal".to_string());
        }

        // Vision hints
        if lower.contains("image") || lower.contains("picture") || lower.contains("visual")
            || lower.contains("see") || lower.contains("look") || lower.contains("photo") {
            domain_hints.push("vision".to_string());
            keywords.push("visual".to_string());
        }

        // Default to text if no specific domain detected
        if domain_hints.is_empty() {
            domain_hints.push("text".to_string());
        }

        // Estimate complexity based on length and structure
        let complexity = (input.len() as f32 / 500.0).min(1.0);

        // Estimate urgency based on keywords
        let urgency = if lower.contains("urgent") || lower.contains("asap") || lower.contains("immediately") {
            0.9
        } else if lower.contains("soon") || lower.contains("quick") {
            0.6
        } else {
            0.3
        };

        let is_multimodal = domain_hints.len() > 1;

        Self {
            domain_hints,
            complexity,
            urgency,
            is_multimodal,
            keywords,
        }
    }
}

/// Statistics for attention mechanism performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionStats {
    /// Total allocations made
    pub total_allocations: u64,
    /// Number of successful allocations (led to correct result)
    pub successful_allocations: u64,
    /// Number of times fallback was needed
    pub fallback_count: u64,
    /// Average number of brains engaged per allocation
    pub avg_brains_engaged: f32,
    /// Brain-specific success rates
    pub brain_success_rates: std::collections::HashMap<String, f32>,
}

impl AttentionStats {
    /// Record a successful allocation
    pub fn record_success(&mut self, engaged_brains: &[String]) {
        self.total_allocations += 1;
        self.successful_allocations += 1;
        self.update_avg_brains(engaged_brains.len());
        for brain in engaged_brains {
            let rate = self.brain_success_rates.entry(brain.clone()).or_insert(0.5);
            *rate = (*rate * 0.9) + 0.1; // Exponential moving average toward 1.0
        }
    }

    /// Record a failed allocation
    pub fn record_failure(&mut self, engaged_brains: &[String]) {
        self.total_allocations += 1;
        self.update_avg_brains(engaged_brains.len());
        for brain in engaged_brains {
            let rate = self.brain_success_rates.entry(brain.clone()).or_insert(0.5);
            *rate *= 0.9; // Decay toward 0.0
        }
    }

    /// Record a fallback (no suitable brain found)
    pub fn record_fallback(&mut self) {
        self.total_allocations += 1;
        self.fallback_count += 1;
    }

    fn update_avg_brains(&mut self, count: usize) {
        let n = self.total_allocations as f32;
        self.avg_brains_engaged = ((self.avg_brains_engaged * (n - 1.0)) + count as f32) / n;
    }

    /// Get success rate for a specific brain - O(1)
    pub fn brain_success_rate(&self, brain_id: &str) -> f32 {
        *self.brain_success_rates.get(brain_id).unwrap_or(&0.5)
    }

    /// Get overall success rate - O(1)
    pub fn overall_success_rate(&self) -> f32 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.successful_allocations as f32 / self.total_allocations as f32
        }
    }
}

/// AttentionMechanism: Dynamic focus allocation across cognitive brains
///
/// This module decides which brains should receive processing resources
/// based on input characteristics, brain capabilities, and historical performance.
///
/// # Architecture
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                     AttentionMechanism                                   │
/// │                                                                          │
/// │  Input ──► Feature Extraction ──► Relevance Scoring ──► Allocation      │
/// │               │                        │                    │            │
/// │               │                        │                    ▼            │
/// │               │                   [MathBrain:0.8]    Brain Selection    │
/// │               │                   [CodeBrain:0.6]         │             │
/// │               │                   [TextBrain:0.3]         ▼             │
/// │               │                                     Top-K Engagement    │
/// │               │                                                          │
/// │  Feedback ◄── Success/Failure Recording ◄── Result Evaluation           │
/// │                                                                          │
/// └─────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Time Complexity
/// - allocate(): O(n log n) where n = number of brains (due to sorting)
/// - record_outcome(): O(k) where k = engaged brains
/// - soft_attention(): O(n)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMechanism {
    /// Configuration
    pub config: AttentionConfig,
    /// Statistics
    pub stats: AttentionStats,
    /// Learned attention biases per brain
    pub brain_biases: std::collections::HashMap<String, f32>,
    /// Domain-to-brain mapping
    pub domain_brain_map: std::collections::HashMap<String, Vec<String>>,
    /// Current focus state
    current_focus: Option<Vec<String>>,
}

impl Default for AttentionMechanism {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionMechanism {
    /// Create a new attention mechanism with default configuration
    pub fn new() -> Self {
        let mut domain_brain_map = std::collections::HashMap::new();

        // Default domain-to-brain mappings
        domain_brain_map.insert("math".to_string(), vec!["MathBrain".to_string()]);
        domain_brain_map.insert("code".to_string(), vec!["CodeBrain".to_string()]);
        domain_brain_map.insert("music".to_string(), vec!["MusicBrain".to_string()]);
        domain_brain_map.insert("chem".to_string(), vec!["ChemBrain".to_string()]);
        domain_brain_map.insert("law".to_string(), vec!["LawBrain".to_string()]);
        domain_brain_map.insert("vision".to_string(), vec!["VisionBrain".to_string()]);
        domain_brain_map.insert("text".to_string(), vec!["TextBrain".to_string()]);

        Self {
            config: AttentionConfig::default(),
            stats: AttentionStats::default(),
            brain_biases: std::collections::HashMap::new(),
            domain_brain_map,
            current_focus: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AttentionConfig) -> Self {
        let mut mechanism = Self::new();
        mechanism.config = config;
        mechanism
    }

    /// Register a brain for a domain
    pub fn register_brain(&mut self, domain: impl Into<String>, brain_id: impl Into<String>) {
        let domain = domain.into();
        let brain_id = brain_id.into();
        self.domain_brain_map
            .entry(domain)
            .or_default()
            .push(brain_id);
    }

    /// Set attention bias for a specific brain
    pub fn set_brain_bias(&mut self, brain_id: impl Into<String>, bias: f32) {
        self.brain_biases.insert(brain_id.into(), bias.clamp(-1.0, 1.0));
    }

    /// Allocate attention based on input - O(n log n)
    pub fn allocate(&mut self, input: &str, available_brains: &[(&str, f32)]) -> AttentionAllocation {
        let features = InputFeatures::from_text(input);
        self.allocate_with_features(&features, available_brains)
    }

    /// Allocate attention with pre-computed features - O(n log n)
    pub fn allocate_with_features(
        &mut self,
        features: &InputFeatures,
        available_brains: &[(&str, f32)], // (brain_id, current_load)
    ) -> AttentionAllocation {
        // Step 1: Compute relevance for each brain - O(n * d) where d = domain hints
        let mut brain_weights: Vec<BrainAttention> = available_brains
            .iter()
            .map(|(brain_id, load)| {
                let mut attention = BrainAttention::new(*brain_id);
                attention.capacity = 1.0 - load.clamp(0.0, 1.0);

                // Compute relevance from domain hints
                let relevance = self.compute_relevance(brain_id, features);
                attention.relevance = relevance;

                // Apply learned bias
                let bias = self.brain_biases.get(*brain_id).copied().unwrap_or(0.0);

                // Apply historical success rate
                let success_rate = self.stats.brain_success_rate(brain_id);

                // Combined raw score
                let raw_score = relevance * (1.0 + bias) * (0.5 + 0.5 * success_rate);
                attention.weight = raw_score;

                attention
            })
            .collect();

        // Step 2: Apply softmax to normalize weights - O(n)
        if self.config.use_soft_attention && !brain_weights.is_empty() {
            self.apply_softmax(&mut brain_weights);
        }

        // Step 3: Sort by weight descending - O(n log n)
        brain_weights.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));

        // Step 4: Select brains to engage - O(n)
        let mut engaged_count = 0;
        let mut total_attention = 0.0;

        for brain in &mut brain_weights {
            if engaged_count < self.config.max_parallel_brains
                && brain.relevance >= self.config.min_relevance_threshold
                && brain.capacity > 0.1
            {
                brain.should_engage = true;
                brain.weight = brain.weight.min(self.config.max_single_weight);
                total_attention += brain.weight;
                engaged_count += 1;
            }
        }

        // Track current focus
        self.current_focus = Some(
            brain_weights.iter()
                .filter(|b| b.should_engage)
                .map(|b| b.brain_id.clone())
                .collect()
        );

        AttentionAllocation {
            brain_weights,
            total_attention,
            engaged_count,
            input_features: features.clone(),
        }
    }

    /// Compute relevance of a brain to input features - O(d) where d = domain hints
    fn compute_relevance(&self, brain_id: &str, features: &InputFeatures) -> f32 {
        let mut relevance = 0.0;

        // Check if brain matches any domain hint
        for domain in &features.domain_hints {
            if let Some(brains) = self.domain_brain_map.get(domain) {
                if brains.iter().any(|b| b == brain_id || brain_id.to_lowercase().contains(&domain.to_lowercase())) {
                    relevance += 0.5;
                }
            }
            // Also check if brain name contains domain
            if brain_id.to_lowercase().contains(&domain.to_lowercase()) {
                relevance += 0.3;
            }
        }

        // Boost for complex inputs (may need multiple brains)
        if features.is_multimodal && features.complexity > 0.5 {
            relevance += 0.1;
        }

        // Urgency slightly boosts all brains to ensure fast response
        relevance += features.urgency * 0.1;

        relevance.clamp(0.0, 1.0)
    }

    /// Apply softmax normalization to weights - O(n)
    fn apply_softmax(&self, weights: &mut [BrainAttention]) {
        if weights.is_empty() {
            return;
        }

        // Find max for numerical stability
        let max_weight = weights.iter().map(|w| w.weight).fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(w/T) for each weight
        let temp = self.config.temperature.max(0.01);
        let exp_weights: Vec<f32> = weights
            .iter()
            .map(|w| ((w.weight - max_weight) / temp).exp())
            .collect();

        // Sum for normalization
        let sum: f32 = exp_weights.iter().sum();

        // Normalize
        if sum > 0.0 {
            for (weight, exp_w) in weights.iter_mut().zip(exp_weights.iter()) {
                weight.weight = exp_w / sum;
            }
        }
    }

    /// Record outcome of attention allocation - O(k)
    pub fn record_outcome(&mut self, success: bool) {
        if let Some(engaged) = &self.current_focus {
            if success {
                self.stats.record_success(engaged);
                // Boost biases for successful brains
                for brain_id in engaged {
                    let bias = self.brain_biases.entry(brain_id.clone()).or_insert(0.0);
                    *bias = (*bias + 0.05).min(0.5);
                }
            } else {
                self.stats.record_failure(engaged);
                // Decay biases for failed brains
                for brain_id in engaged {
                    let bias = self.brain_biases.entry(brain_id.clone()).or_insert(0.0);
                    *bias = (*bias - 0.05).max(-0.5);
                }
            }
        }
    }

    /// Record that no suitable brain was found
    pub fn record_fallback(&mut self) {
        self.stats.record_fallback();
    }

    /// Get current focus (which brains are engaged)
    pub fn current_focus(&self) -> Option<&[String]> {
        self.current_focus.as_deref()
    }

    /// Reset attention state
    pub fn reset(&mut self) {
        self.current_focus = None;
    }

    /// Get attention statistics
    pub fn stats(&self) -> &AttentionStats {
        &self.stats
    }

    /// Suggest attention redistribution based on performance - O(n)
    pub fn suggest_redistribution(&self) -> Vec<(String, f32)> {
        let mut suggestions = Vec::new();

        for (brain_id, &success_rate) in &self.stats.brain_success_rates {
            let current_bias = self.brain_biases.get(brain_id).copied().unwrap_or(0.0);

            // If success rate is low, suggest reducing attention
            if success_rate < 0.3 && current_bias > -0.3 {
                suggestions.push((brain_id.clone(), current_bias - 0.1));
            }
            // If success rate is high, suggest increasing attention
            else if success_rate > 0.7 && current_bias < 0.3 {
                suggestions.push((brain_id.clone(), current_bias + 0.1));
            }
        }

        suggestions
    }
}

/// Factory function to create an attention mechanism with standard configuration
pub fn create_attention_mechanism() -> AttentionMechanism {
    AttentionMechanism::new()
}

// ============================================================================
// AttentionMechanism Tests
// ============================================================================

#[cfg(test)]
mod attention_tests {
    use super::*;

    #[test]
    fn test_attention_mechanism_creation() {
        let mechanism = AttentionMechanism::new();
        assert!(mechanism.current_focus.is_none());
        assert_eq!(mechanism.stats.total_allocations, 0);
    }

    #[test]
    fn test_attention_config_default() {
        let config = AttentionConfig::default();
        assert_eq!(config.max_parallel_brains, 3);
        assert_eq!(config.min_relevance_threshold, 0.1);
        assert!(config.use_soft_attention);
        assert_eq!(config.temperature, 1.0);
    }

    #[test]
    fn test_input_features_from_text_math() {
        let features = InputFeatures::from_text("Calculate 2 + 3 * 4");
        assert!(features.domain_hints.contains(&"math".to_string()));
        assert!(features.complexity >= 0.0 && features.complexity <= 1.0);
    }

    #[test]
    fn test_input_features_from_text_code() {
        let features = InputFeatures::from_text("Write a function to sort numbers");
        assert!(features.domain_hints.contains(&"code".to_string()));
    }

    #[test]
    fn test_input_features_from_text_multimodal() {
        let features = InputFeatures::from_text("Write code to solve this equation: 2x + 3 = 7");
        assert!(features.is_multimodal);
        assert!(features.domain_hints.len() > 1);
    }

    #[test]
    fn test_input_features_urgency() {
        let urgent = InputFeatures::from_text("Calculate this urgently!");
        assert!(urgent.urgency > 0.8);

        let normal = InputFeatures::from_text("Calculate this when you can.");
        assert!(normal.urgency < 0.5);
    }

    #[test]
    fn test_brain_attention_combined_score() {
        let mut attention = BrainAttention::new("TestBrain");
        attention.relevance = 0.8;
        attention.capacity = 0.5;
        attention.weight = 0.6;

        let score = attention.combined_score();
        assert!((score - 0.24).abs() < 0.01); // 0.8 * 0.5 * 0.6 = 0.24
    }

    #[test]
    fn test_attention_allocation() {
        let mut mechanism = AttentionMechanism::new();

        let brains = vec![
            ("MathBrain", 0.1),  // Low load
            ("CodeBrain", 0.5),  // Medium load
            ("TextBrain", 0.9),  // High load
        ];

        let allocation = mechanism.allocate("Calculate 2 + 2", &brains);

        assert!(allocation.engaged_count > 0);
        assert!(allocation.engaged_count <= mechanism.config.max_parallel_brains);

        // MathBrain should be engaged for math input
        let math_brain = allocation.brain_weights.iter().find(|b| b.brain_id == "MathBrain");
        assert!(math_brain.map(|b| b.should_engage).unwrap_or(false));
    }

    #[test]
    fn test_attention_allocation_respects_capacity() {
        let mut mechanism = AttentionMechanism::new();

        // All brains at high load
        let brains = vec![
            ("MathBrain", 0.95),  // Very high load - should not engage
            ("CodeBrain", 0.2),   // Low load
        ];

        let allocation = mechanism.allocate("Calculate 2 + 2", &brains);

        // MathBrain has too low capacity, shouldn't be engaged
        let math_brain = allocation.brain_weights.iter().find(|b| b.brain_id == "MathBrain");
        if let Some(mb) = math_brain {
            assert!(!mb.should_engage || mb.capacity < 0.1);
        }
    }

    #[test]
    fn test_attention_allocation_primary() {
        let mut mechanism = AttentionMechanism::new();

        let brains = vec![
            ("MathBrain", 0.1),
            ("CodeBrain", 0.1),
        ];

        let allocation = mechanism.allocate("Solve 2x + 3 = 7", &brains);

        let primary = allocation.primary();
        assert!(primary.is_some());
    }

    #[test]
    fn test_attention_record_outcome_success() {
        let mut mechanism = AttentionMechanism::new();

        let brains = vec![("MathBrain", 0.1)];
        let _allocation = mechanism.allocate("Calculate 1 + 1", &brains);

        mechanism.record_outcome(true);

        assert_eq!(mechanism.stats.successful_allocations, 1);
        assert_eq!(mechanism.stats.total_allocations, 1);

        let bias = mechanism.brain_biases.get("MathBrain").copied().unwrap_or(0.0);
        assert!(bias > 0.0); // Bias should increase after success
    }

    #[test]
    fn test_attention_record_outcome_failure() {
        let mut mechanism = AttentionMechanism::new();

        let brains = vec![("MathBrain", 0.1)];
        let _allocation = mechanism.allocate("Calculate 1 + 1", &brains);

        mechanism.record_outcome(false);

        assert_eq!(mechanism.stats.successful_allocations, 0);
        assert_eq!(mechanism.stats.total_allocations, 1);

        let bias = mechanism.brain_biases.get("MathBrain").copied().unwrap_or(0.0);
        assert!(bias < 0.0); // Bias should decrease after failure
    }

    #[test]
    fn test_attention_stats_success_rate() {
        let mut stats = AttentionStats::default();

        stats.record_success(&["BrainA".to_string()]);
        stats.record_success(&["BrainA".to_string()]);
        stats.record_failure(&["BrainA".to_string()]);

        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.successful_allocations, 2);

        let rate = stats.overall_success_rate();
        assert!((rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_attention_softmax_normalization() {
        let mut mechanism = AttentionMechanism::new();
        mechanism.config.use_soft_attention = true;

        let brains = vec![
            ("BrainA", 0.1),
            ("BrainB", 0.1),
            ("BrainC", 0.1),
        ];

        // Use input that gives equal relevance to all
        let allocation = mechanism.allocate("generic text", &brains);

        // Weights should sum to approximately 1.0 (softmax property)
        let sum: f32 = allocation.brain_weights.iter().map(|b| b.weight).sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_attention_top_k() {
        let mut mechanism = AttentionMechanism::new();
        mechanism.config.max_parallel_brains = 5;

        let brains = vec![
            ("BrainA", 0.1),
            ("BrainB", 0.1),
            ("BrainC", 0.1),
            ("BrainD", 0.1),
        ];

        let allocation = mechanism.allocate("generic text", &brains);

        let top_2 = allocation.top_k(2);
        assert!(top_2.len() <= 2);
    }

    #[test]
    fn test_attention_register_brain() {
        let mut mechanism = AttentionMechanism::new();

        mechanism.register_brain("physics", "PhysicsBrain");

        assert!(mechanism.domain_brain_map.get("physics").map(|v| v.contains(&"PhysicsBrain".to_string())).unwrap_or(false));
    }

    #[test]
    fn test_attention_set_brain_bias() {
        let mut mechanism = AttentionMechanism::new();

        mechanism.set_brain_bias("TestBrain", 0.3);

        assert_eq!(mechanism.brain_biases.get("TestBrain").copied(), Some(0.3));
    }

    #[test]
    fn test_attention_bias_clamping() {
        let mut mechanism = AttentionMechanism::new();

        mechanism.set_brain_bias("TestBrain", 2.0); // Should clamp to 1.0
        assert_eq!(mechanism.brain_biases.get("TestBrain").copied(), Some(1.0));

        mechanism.set_brain_bias("TestBrain", -2.0); // Should clamp to -1.0
        assert_eq!(mechanism.brain_biases.get("TestBrain").copied(), Some(-1.0));
    }

    #[test]
    fn test_attention_suggest_redistribution() {
        let mut mechanism = AttentionMechanism::new();

        // Simulate many failures for one brain
        for _ in 0..10 {
            mechanism.stats.record_failure(&["BadBrain".to_string()]);
        }

        // Simulate many successes for another
        for _ in 0..10 {
            mechanism.stats.record_success(&["GoodBrain".to_string()]);
        }

        let suggestions = mechanism.suggest_redistribution();

        // Should suggest increasing attention for GoodBrain
        let good_suggestion = suggestions.iter().find(|(id, _)| id == "GoodBrain");
        if let Some((_, suggested_bias)) = good_suggestion {
            assert!(*suggested_bias > 0.0);
        }
    }

    #[test]
    fn test_attention_mechanism_reset() {
        let mut mechanism = AttentionMechanism::new();

        let brains = vec![("TestBrain", 0.1)];
        let _allocation = mechanism.allocate("test", &brains);

        assert!(mechanism.current_focus().is_some());

        mechanism.reset();

        assert!(mechanism.current_focus().is_none());
    }

    #[test]
    fn test_create_attention_mechanism_factory() {
        let mechanism = create_attention_mechanism();
        assert!(mechanism.current_focus.is_none());
        assert!(!mechanism.domain_brain_map.is_empty());
    }

    #[test]
    fn test_attention_empty_brains() {
        let mut mechanism = AttentionMechanism::new();

        let allocation = mechanism.allocate("test input", &[]);

        assert_eq!(allocation.engaged_count, 0);
        assert!(allocation.brain_weights.is_empty());
    }

    #[test]
    fn test_attention_fallback_recording() {
        let mut mechanism = AttentionMechanism::new();

        mechanism.record_fallback();
        mechanism.record_fallback();

        assert_eq!(mechanism.stats.fallback_count, 2);
        assert_eq!(mechanism.stats.total_allocations, 2);
    }
}

// ============================================================================
// UnifiedCognition: Connect All Brains (Math, Code, Vision, Music, Chem, Law)
// ============================================================================

/// Brain status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainStatus {
    /// Brain identifier
    pub id: String,
    /// Domain the brain handles
    pub domain: String,
    /// Whether brain is currently active
    pub active: bool,
    /// Current load (0.0 to 1.0)
    pub load: f32,
    /// Total requests processed
    pub requests_processed: u64,
    /// Total successful responses
    pub successful_responses: u64,
    /// Last activity timestamp
    pub last_activity: u64,
}

impl BrainStatus {
    /// Create a new brain status entry
    pub fn new(id: impl Into<String>, domain: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            domain: domain.into(),
            active: true,
            load: 0.0,
            requests_processed: 0,
            successful_responses: 0,
            last_activity: 0,
        }
    }

    /// Get success rate - O(1)
    pub fn success_rate(&self) -> f32 {
        if self.requests_processed == 0 {
            0.0
        } else {
            self.successful_responses as f32 / self.requests_processed as f32
        }
    }

    /// Record a request
    pub fn record_request(&mut self, success: bool, timestamp: u64) {
        self.requests_processed += 1;
        if success {
            self.successful_responses += 1;
        }
        self.last_activity = timestamp;
    }
}

/// Configuration for unified cognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedCognitionConfig {
    /// Maximum concurrent brain operations
    pub max_concurrent_operations: usize,
    /// Whether to use attention-based routing
    pub use_attention_routing: bool,
    /// Enable parallel brain processing
    pub enable_parallel: bool,
    /// Fallback to text brain if no match
    pub fallback_enabled: bool,
    /// Maximum processing time per request (ms)
    pub max_processing_time_ms: u64,
    /// Enable self-monitoring via ReflectiveBrain
    pub enable_self_monitoring: bool,
    /// Introspection interval (every N requests)
    pub introspection_interval: u64,
}

impl Default for UnifiedCognitionConfig {
    fn default() -> Self {
        Self {
            max_concurrent_operations: 3,
            use_attention_routing: true,
            enable_parallel: true,
            fallback_enabled: true,
            max_processing_time_ms: 5000,
            enable_self_monitoring: true,
            introspection_interval: 100,
        }
    }
}

/// Result of unified cognition processing
#[derive(Debug, Clone)]
pub struct UnifiedCognitionResult {
    /// Primary output from the selected brain
    pub primary_output: Option<String>,
    /// Which brain produced the result
    pub source_brain: String,
    /// Confidence in the result
    pub confidence: f32,
    /// All brain contributions (brain_id -> output)
    pub brain_outputs: std::collections::HashMap<String, String>,
    /// Attention weights used
    pub attention_weights: Vec<(String, f32)>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Whether this was a fallback result
    pub was_fallback: bool,
}

impl UnifiedCognitionResult {
    /// Create an empty result
    pub fn empty() -> Self {
        Self {
            primary_output: None,
            source_brain: String::new(),
            confidence: 0.0,
            brain_outputs: std::collections::HashMap::new(),
            attention_weights: Vec::new(),
            processing_time_ms: 0,
            was_fallback: false,
        }
    }

    /// Check if result is valid
    pub fn is_valid(&self) -> bool {
        self.primary_output.is_some() && self.confidence > 0.0
    }
}

/// Statistics for unified cognition
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnifiedCognitionStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful completions
    pub successful_completions: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Fallback invocations
    pub fallback_invocations: u64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f32,
    /// Brain usage counts
    pub brain_usage: std::collections::HashMap<String, u64>,
    /// Last introspection health score
    pub last_health_score: f32,
    /// Total introspections performed
    pub introspections_performed: u64,
}

impl UnifiedCognitionStats {
    /// Record a completed request
    pub fn record_completion(&mut self, brain_id: &str, processing_time_ms: u64, success: bool) {
        self.total_requests += 1;
        if success {
            self.successful_completions += 1;
        } else {
            self.failed_requests += 1;
        }

        *self.brain_usage.entry(brain_id.to_string()).or_insert(0) += 1;

        // Update running average
        let n = self.total_requests as f32;
        self.avg_processing_time_ms = ((self.avg_processing_time_ms * (n - 1.0)) + processing_time_ms as f32) / n;
    }

    /// Record a fallback
    pub fn record_fallback(&mut self) {
        self.fallback_invocations += 1;
    }

    /// Get success rate - O(1)
    pub fn success_rate(&self) -> f32 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.successful_completions as f32 / self.total_requests as f32
        }
    }
}

/// UnifiedCognition: Connects all cognitive brains into one system
///
/// This is the central integration point for all domain brains (Math, Code,
/// Vision, Music, Chem, Law) with cognitive infrastructure (attention,
/// self-model, reflection).
///
/// # Architecture
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                        UnifiedCognition                                  │
/// │                                                                          │
/// │  ┌───────────────────────────────────────────────────────────────────┐  │
/// │  │                      AttentionMechanism                            │  │
/// │  │   Input ──► Feature Analysis ──► Brain Selection ──► Weights      │  │
/// │  └───────────────────────────────────────────────────────────────────┘  │
/// │                                    │                                    │
/// │                                    ▼                                    │
/// │  ┌───────────────────────────────────────────────────────────────────┐  │
/// │  │                      Brain Registry                                │  │
/// │  │   [MathBrain] [CodeBrain] [VisionBrain] [MusicBrain] ...          │  │
/// │  └───────────────────────────────────────────────────────────────────┘  │
/// │                                    │                                    │
/// │                                    ▼                                    │
/// │  ┌───────────────────────────────────────────────────────────────────┐  │
/// │  │                      SelfModel                                     │  │
/// │  │   Component tracking, state monitoring, system health              │  │
/// │  └───────────────────────────────────────────────────────────────────┘  │
/// │                                    │                                    │
/// │                                    ▼                                    │
/// │  ┌───────────────────────────────────────────────────────────────────┐  │
/// │  │                    ReflectiveBrain                                 │  │
/// │  │   Introspection, problem detection, auto-healing                   │  │
/// │  └───────────────────────────────────────────────────────────────────┘  │
/// │                                                                          │
/// └─────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Time Complexity
/// - process(): O(n log n) for attention + O(k * m) for brain processing
///   where n = brains, k = selected brains, m = input length
/// - get_brain_status(): O(1)
/// - perform_introspection(): O(n) where n = components
#[derive(Debug)]
pub struct UnifiedCognition {
    /// Configuration
    pub config: UnifiedCognitionConfig,
    /// Attention mechanism for brain selection
    pub attention: AttentionMechanism,
    /// Self model for system state
    pub self_model: SelfModel,
    /// Reflective brain for introspection
    pub reflective_brain: ReflectiveBrain,
    /// Brain status tracking
    brain_status: std::collections::HashMap<String, BrainStatus>,
    /// Statistics
    pub stats: UnifiedCognitionStats,
    /// Request counter (for introspection scheduling)
    request_counter: u64,
}

impl Default for UnifiedCognition {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedCognition {
    /// Create a new unified cognition system
    pub fn new() -> Self {
        let mut self_model = create_self_model();

        // Add unified cognition as a component
        self_model.add_component(SelfModelComponent::new(
            "unified_cognition",
            ComponentType::CognitiveModule("coordination".to_string()),
        ));

        let reflective_brain = create_reflective_brain();
        let attention = AttentionMechanism::new();

        Self {
            config: UnifiedCognitionConfig::default(),
            attention,
            self_model,
            reflective_brain,
            brain_status: std::collections::HashMap::new(),
            stats: UnifiedCognitionStats::default(),
            request_counter: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: UnifiedCognitionConfig) -> Self {
        let mut unified = Self::new();
        unified.config = config;
        unified
    }

    /// Register a brain with the unified system
    pub fn register_brain(&mut self, brain_id: impl Into<String>, domain: impl Into<String>) {
        let brain_id = brain_id.into();
        let domain = domain.into();

        // Add to brain status tracking
        self.brain_status.insert(
            brain_id.clone(),
            BrainStatus::new(&brain_id, &domain),
        );

        // Register with attention mechanism
        self.attention.register_brain(&domain, &brain_id);

        // Add to self model
        self.self_model.add_component(SelfModelComponent::new(
            &brain_id,
            ComponentType::Brain(domain.clone()),
        ));
    }

    /// Register standard brains (Math, Code, Vision, Music, Chem, Law, Text)
    pub fn register_standard_brains(&mut self) {
        self.register_brain("MathBrain", "math");
        self.register_brain("CodeBrain", "code");
        self.register_brain("VisionBrain", "vision");
        self.register_brain("MusicBrain", "music");
        self.register_brain("ChemBrain", "chem");
        self.register_brain("LawBrain", "law");
        self.register_brain("TextBrain", "text");
    }

    /// Get brain status - O(1)
    pub fn get_brain_status(&self, brain_id: &str) -> Option<&BrainStatus> {
        self.brain_status.get(brain_id)
    }

    /// Get all brain statuses - O(n)
    pub fn all_brain_status(&self) -> Vec<&BrainStatus> {
        self.brain_status.values().collect()
    }

    /// Get available brains with their loads - O(n)
    pub fn get_available_brains(&self) -> Vec<(&str, f32)> {
        self.brain_status
            .iter()
            .filter(|(_, status)| status.active)
            .map(|(id, status)| (id.as_str(), status.load))
            .collect()
    }

    /// Update brain load
    pub fn update_brain_load(&mut self, brain_id: &str, load: f32) {
        if let Some(status) = self.brain_status.get_mut(brain_id) {
            status.load = load.clamp(0.0, 1.0);
        }
        if let Some(component) = self.self_model.get_component_mut(brain_id) {
            component.load = load.clamp(0.0, 1.0);
        }
    }

    /// Enable/disable a brain
    pub fn set_brain_active(&mut self, brain_id: &str, active: bool) {
        if let Some(status) = self.brain_status.get_mut(brain_id) {
            status.active = active;
        }
        if let Some(component) = self.self_model.get_component_mut(brain_id) {
            component.state = if active {
                ComponentState::Active
            } else {
                ComponentState::Disabled
            };
        }
    }

    /// Allocate attention for an input - O(n log n)
    pub fn allocate_attention(&mut self, input: &str) -> AttentionAllocation {
        let available: Vec<(String, f32)> = self.brain_status
            .iter()
            .filter(|(_, status)| status.active)
            .map(|(id, status)| (id.clone(), status.load))
            .collect();
        let available_refs: Vec<(&str, f32)> = available.iter()
            .map(|(id, load)| (id.as_str(), *load))
            .collect();
        self.attention.allocate(input, &available_refs)
    }

    /// Simulate processing an input (returns selected brains and weights)
    ///
    /// This method selects brains based on attention and simulates processing.
    /// In a real system, this would call actual brain.execute() methods.
    pub fn process(&mut self, input: &str, timestamp: u64) -> UnifiedCognitionResult {
        self.request_counter += 1;
        let start_time = timestamp;

        // Check if introspection is needed
        if self.config.enable_self_monitoring
            && self.request_counter.is_multiple_of(self.config.introspection_interval)
        {
            self.perform_introspection();
        }

        // Allocate attention
        let allocation = self.allocate_attention(input);

        if allocation.engaged_count == 0 {
            // No brain matched - try fallback
            if self.config.fallback_enabled {
                self.stats.record_fallback();
                self.attention.record_fallback();

                let mut result = UnifiedCognitionResult::empty();
                result.source_brain = "TextBrain".to_string();
                result.was_fallback = true;
                result.processing_time_ms = timestamp.saturating_sub(start_time);
                return result;
            } else {
                return UnifiedCognitionResult::empty();
            }
        }

        // Process with selected brains
        let mut result = UnifiedCognitionResult::empty();

        for brain_attention in allocation.brain_weights.iter().filter(|b| b.should_engage) {
            let brain_id = &brain_attention.brain_id;

            // Simulate brain processing (in real implementation, call brain.execute())
            let simulated_output = format!("Processed by {} with weight {:.3}", brain_id, brain_attention.weight);

            result.brain_outputs.insert(brain_id.clone(), simulated_output.clone());
            result.attention_weights.push((brain_id.clone(), brain_attention.weight));

            // Update status
            if let Some(status) = self.brain_status.get_mut(brain_id) {
                status.record_request(true, timestamp);
            }
            if let Some(component) = self.self_model.get_component_mut(brain_id) {
                component.record_operation();
            }

            // First engaged brain is primary
            if result.primary_output.is_none() {
                result.primary_output = Some(simulated_output);
                result.source_brain = brain_id.clone();
                result.confidence = brain_attention.weight;
            }
        }

        result.processing_time_ms = timestamp.saturating_sub(start_time);

        // Record outcome in attention
        let success = result.is_valid();
        self.attention.record_outcome(success);

        // Update stats
        self.stats.record_completion(&result.source_brain, result.processing_time_ms, success);

        result
    }

    /// Perform introspection using ReflectiveBrain
    pub fn perform_introspection(&mut self) -> IntrospectionResult {
        // Update self model with current brain states
        self.sync_self_model();

        // Perform introspection
        let result = self.reflective_brain.introspect();

        // Update stats
        self.stats.last_health_score = result.health_score;
        self.stats.introspections_performed += 1;

        // Auto-heal if needed and enabled
        if self.config.enable_self_monitoring && result.health_score < 0.7 {
            let _healed = self.reflective_brain.auto_heal();
        }

        result
    }

    /// Sync self model with current brain status
    fn sync_self_model(&mut self) {
        for (brain_id, status) in &self.brain_status {
            if let Some(component) = self.self_model.get_component_mut(brain_id) {
                component.load = status.load;
                component.confidence = status.success_rate();
                component.state = if status.active {
                    if status.load > 0.8 {
                        ComponentState::Overloaded
                    } else {
                        ComponentState::Active
                    }
                } else {
                    ComponentState::Disabled
                };
            }
        }

        // Update system-level metrics
        self.self_model.system_load = self.brain_status.values()
            .filter(|s| s.active)
            .map(|s| s.load)
            .sum::<f32>() / self.brain_status.len().max(1) as f32;

        self.self_model.system_confidence = self.stats.success_rate();
    }

    /// Get system health overview
    pub fn health_overview(&self) -> SystemHealthOverview {
        let active_brains = self.brain_status.values().filter(|s| s.active).count();
        let total_brains = self.brain_status.len();
        let avg_load = self.brain_status.values()
            .filter(|s| s.active)
            .map(|s| s.load)
            .sum::<f32>() / active_brains.max(1) as f32;

        let overloaded_brains: Vec<String> = self.brain_status
            .iter()
            .filter(|(_, s)| s.load > 0.8)
            .map(|(id, _)| id.clone())
            .collect();

        SystemHealthOverview {
            active_brains,
            total_brains,
            average_load: avg_load,
            overloaded_brains,
            success_rate: self.stats.success_rate(),
            last_health_score: self.stats.last_health_score,
            total_requests: self.stats.total_requests,
        }
    }

    /// Suggest attention redistribution based on performance
    pub fn suggest_attention_optimization(&self) -> Vec<(String, f32)> {
        self.attention.suggest_redistribution()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = UnifiedCognitionStats::default();
        for status in self.brain_status.values_mut() {
            status.requests_processed = 0;
            status.successful_responses = 0;
        }
    }

    /// Get reference to self model
    pub fn self_model(&self) -> &SelfModel {
        &self.self_model
    }

    /// Get reference to attention mechanism
    pub fn attention(&self) -> &AttentionMechanism {
        &self.attention
    }

    /// Get reference to reflective brain
    pub fn reflective_brain(&self) -> &ReflectiveBrain {
        &self.reflective_brain
    }

    /// Get current statistics
    pub fn stats(&self) -> &UnifiedCognitionStats {
        &self.stats
    }
}

/// System health overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthOverview {
    /// Number of active brains
    pub active_brains: usize,
    /// Total registered brains
    pub total_brains: usize,
    /// Average brain load
    pub average_load: f32,
    /// List of overloaded brain IDs
    pub overloaded_brains: Vec<String>,
    /// Overall success rate
    pub success_rate: f32,
    /// Last introspection health score
    pub last_health_score: f32,
    /// Total requests processed
    pub total_requests: u64,
}

impl SystemHealthOverview {
    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        self.active_brains > 0
            && self.average_load < 0.8
            && self.overloaded_brains.is_empty()
            && self.success_rate > 0.5
    }
}

/// Factory function to create a unified cognition system with standard brains
pub fn create_unified_cognition() -> UnifiedCognition {
    let mut unified = UnifiedCognition::new();
    unified.register_standard_brains();
    unified
}

// ============================================================================
// UnifiedCognition Tests
// ============================================================================

#[cfg(test)]
mod unified_cognition_tests {
    use super::*;

    #[test]
    fn test_unified_cognition_creation() {
        let unified = UnifiedCognition::new();
        assert!(unified.brain_status.is_empty());
        assert_eq!(unified.stats.total_requests, 0);
    }

    #[test]
    fn test_unified_cognition_register_brain() {
        let mut unified = UnifiedCognition::new();
        unified.register_brain("TestBrain", "test");

        assert!(unified.brain_status.contains_key("TestBrain"));
        assert!(unified.self_model.get_component("TestBrain").is_some());
    }

    #[test]
    fn test_unified_cognition_register_standard_brains() {
        let mut unified = UnifiedCognition::new();
        unified.register_standard_brains();

        assert_eq!(unified.brain_status.len(), 7);
        assert!(unified.brain_status.contains_key("MathBrain"));
        assert!(unified.brain_status.contains_key("CodeBrain"));
        assert!(unified.brain_status.contains_key("VisionBrain"));
    }

    #[test]
    fn test_unified_cognition_get_brain_status() {
        let mut unified = UnifiedCognition::new();
        unified.register_brain("MathBrain", "math");

        let status = unified.get_brain_status("MathBrain");
        assert!(status.is_some());
        assert_eq!(status.unwrap().domain, "math");
    }

    #[test]
    fn test_unified_cognition_update_load() {
        let mut unified = UnifiedCognition::new();
        unified.register_brain("TestBrain", "test");

        unified.update_brain_load("TestBrain", 0.7);

        let status = unified.get_brain_status("TestBrain").unwrap();
        assert!((status.load - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_unified_cognition_set_brain_active() {
        let mut unified = UnifiedCognition::new();
        unified.register_brain("TestBrain", "test");

        unified.set_brain_active("TestBrain", false);

        let status = unified.get_brain_status("TestBrain").unwrap();
        assert!(!status.active);
    }

    #[test]
    fn test_unified_cognition_allocate_attention() {
        let mut unified = UnifiedCognition::new();
        unified.register_standard_brains();

        let allocation = unified.allocate_attention("Calculate 2 + 3");

        assert!(allocation.engaged_count > 0);
    }

    #[test]
    fn test_unified_cognition_process() {
        let mut unified = UnifiedCognition::new();
        unified.register_standard_brains();

        let result = unified.process("Calculate 2 + 3", 1000);

        assert!(result.is_valid());
        assert!(!result.source_brain.is_empty());
    }

    #[test]
    fn test_unified_cognition_process_updates_stats() {
        let mut unified = UnifiedCognition::new();
        unified.register_standard_brains();

        let _result = unified.process("Calculate 2 + 3", 1000);

        assert_eq!(unified.stats.total_requests, 1);
        assert!(unified.stats.successful_completions > 0);
    }

    #[test]
    fn test_unified_cognition_fallback() {
        let mut unified = UnifiedCognition::new();
        unified.config.fallback_enabled = true;
        // Don't register any brains

        let result = unified.process("random input", 1000);

        assert!(result.was_fallback);
        assert_eq!(result.source_brain, "TextBrain");
    }

    #[test]
    fn test_unified_cognition_no_fallback() {
        let mut unified = UnifiedCognition::new();
        unified.config.fallback_enabled = false;
        // Don't register any brains

        let result = unified.process("random input", 1000);

        assert!(!result.is_valid());
    }

    #[test]
    fn test_unified_cognition_introspection() {
        let mut unified = UnifiedCognition::new();
        unified.register_standard_brains();

        let result = unified.perform_introspection();

        assert!(result.health_score >= 0.0 && result.health_score <= 1.0);
        assert_eq!(unified.stats.introspections_performed, 1);
    }

    #[test]
    fn test_unified_cognition_health_overview() {
        let mut unified = UnifiedCognition::new();
        unified.register_standard_brains();

        // Process a few requests to establish a success rate
        for _ in 0..3 {
            unified.process("Calculate 2 + 2", 1000);
        }

        let health = unified.health_overview();

        assert_eq!(health.total_brains, 7);
        assert_eq!(health.active_brains, 7);
        assert!(health.is_healthy());
    }

    #[test]
    fn test_unified_cognition_health_overview_overloaded() {
        let mut unified = UnifiedCognition::new();
        unified.register_brain("TestBrain", "test");
        unified.update_brain_load("TestBrain", 0.95);

        let health = unified.health_overview();

        assert!(!health.overloaded_brains.is_empty());
        assert!(health.overloaded_brains.contains(&"TestBrain".to_string()));
    }

    #[test]
    fn test_unified_cognition_suggest_optimization() {
        let mut unified = UnifiedCognition::new();
        unified.register_standard_brains();

        // Process some inputs to generate statistics
        for _ in 0..5 {
            let _result = unified.process("Calculate 1 + 1", 1000);
        }

        let _suggestions = unified.suggest_attention_optimization();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_unified_cognition_reset_stats() {
        let mut unified = UnifiedCognition::new();
        unified.register_standard_brains();
        unified.process("test", 1000);

        unified.reset_stats();

        assert_eq!(unified.stats.total_requests, 0);
    }

    #[test]
    fn test_unified_cognition_config() {
        let config = UnifiedCognitionConfig {
            max_concurrent_operations: 5,
            use_attention_routing: false,
            ..Default::default()
        };

        let unified = UnifiedCognition::with_config(config);

        assert_eq!(unified.config.max_concurrent_operations, 5);
        assert!(!unified.config.use_attention_routing);
    }

    #[test]
    fn test_brain_status_success_rate() {
        let mut status = BrainStatus::new("TestBrain", "test");

        status.record_request(true, 100);
        status.record_request(true, 200);
        status.record_request(false, 300);

        let rate = status.success_rate();
        assert!((rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_unified_cognition_stats_success_rate() {
        let mut stats = UnifiedCognitionStats::default();

        stats.record_completion("BrainA", 100, true);
        stats.record_completion("BrainA", 100, true);
        stats.record_completion("BrainA", 100, false);

        let rate = stats.success_rate();
        assert!((rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_create_unified_cognition_factory() {
        let unified = create_unified_cognition();

        assert_eq!(unified.brain_status.len(), 7);
        assert!(unified.get_brain_status("MathBrain").is_some());
    }

    #[test]
    fn test_unified_cognition_result_is_valid() {
        let empty = UnifiedCognitionResult::empty();
        assert!(!empty.is_valid());

        let mut valid = UnifiedCognitionResult::empty();
        valid.primary_output = Some("output".to_string());
        valid.confidence = 0.8;
        assert!(valid.is_valid());
    }

    #[test]
    fn test_system_health_overview_is_healthy() {
        let healthy = SystemHealthOverview {
            active_brains: 5,
            total_brains: 7,
            average_load: 0.3,
            overloaded_brains: vec![],
            success_rate: 0.8,
            last_health_score: 0.9,
            total_requests: 100,
        };
        assert!(healthy.is_healthy());

        let unhealthy = SystemHealthOverview {
            active_brains: 0,
            ..healthy.clone()
        };
        assert!(!unhealthy.is_healthy());
    }
}

// ============================================================================
// SelfAwareGrapheme: Full AGI System Integrating All Cognitive Modules
// ============================================================================

/// Awareness level of the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AwarenessLevel {
    /// System is in minimal mode - basic processing only
    Minimal,
    /// System is partially aware - monitoring active
    #[default]
    Partial,
    /// System is fully aware - all cognitive modules active
    Full,
    /// System is in heightened awareness - extra monitoring
    Heightened,
}

/// Current cognitive state of the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveSnapshot {
    /// Timestamp of snapshot
    pub timestamp: u64,
    /// Current awareness level
    pub awareness_level: AwarenessLevel,
    /// System health score (0.0 to 1.0)
    pub health_score: f32,
    /// Current system confidence
    pub confidence: f32,
    /// Number of active goals
    pub active_goals: usize,
    /// Current focus (which brains are engaged)
    pub current_focus: Vec<String>,
    /// Recent operation count
    pub recent_operations: u64,
    /// Memory usage estimate
    pub memory_pressure: f32,
}

impl CognitiveSnapshot {
    /// Check if system is operating normally
    pub fn is_healthy(&self) -> bool {
        self.health_score > 0.7 && self.confidence > 0.5 && self.memory_pressure < 0.9
    }

    /// Check if system needs attention
    pub fn needs_attention(&self) -> bool {
        self.health_score < 0.5 || self.memory_pressure > 0.95
    }
}

/// Configuration for SelfAwareGrapheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAwareConfig {
    /// Default awareness level
    pub default_awareness: AwarenessLevel,
    /// Enable automatic awareness adjustment
    pub auto_adjust_awareness: bool,
    /// Health check interval (every N operations)
    pub health_check_interval: u64,
    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
    /// Enable learning from experience
    pub enable_learning: bool,
    /// Enable goal-directed behavior
    pub enable_goals: bool,
    /// Snapshot retention count
    pub snapshot_retention: usize,
}

impl Default for SelfAwareConfig {
    fn default() -> Self {
        Self {
            default_awareness: AwarenessLevel::Partial,
            auto_adjust_awareness: true,
            health_check_interval: 50,
            max_concurrent_ops: 5,
            enable_learning: true,
            enable_goals: true,
            snapshot_retention: 100,
        }
    }
}

/// Statistics for the self-aware system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelfAwareStats {
    /// Total cognitive operations
    pub total_operations: u64,
    /// Successful operations
    pub successful_ops: u64,
    /// Failed operations
    pub failed_ops: u64,
    /// Goals completed
    pub goals_completed: u64,
    /// Goals failed
    pub goals_failed: u64,
    /// Self-corrections performed
    pub self_corrections: u64,
    /// Awareness level changes
    pub awareness_changes: u64,
    /// Health score history (recent)
    pub health_history: Vec<f32>,
}

impl SelfAwareStats {
    /// Get operation success rate - O(1)
    pub fn success_rate(&self) -> f32 {
        if self.total_operations == 0 {
            1.0
        } else {
            self.successful_ops as f32 / self.total_operations as f32
        }
    }

    /// Get goal completion rate - O(1)
    pub fn goal_completion_rate(&self) -> f32 {
        let total_goals = self.goals_completed + self.goals_failed;
        if total_goals == 0 {
            1.0
        } else {
            self.goals_completed as f32 / total_goals as f32
        }
    }

    /// Get average health - O(n)
    pub fn average_health(&self) -> f32 {
        if self.health_history.is_empty() {
            1.0
        } else {
            self.health_history.iter().sum::<f32>() / self.health_history.len() as f32
        }
    }

    /// Record health score with retention limit
    pub fn record_health(&mut self, score: f32, max_history: usize) {
        self.health_history.push(score);
        if self.health_history.len() > max_history {
            self.health_history.remove(0);
        }
    }
}

/// SelfAwareGrapheme: Full AGI System
///
/// This is the apex integration of all GRAPHEME cognitive modules:
/// - UnifiedCognition: Brain orchestration and attention
/// - SelfModel: System state representation
/// - ReflectiveBrain: Introspection and self-modification
/// - GoalStack: Hierarchical goal management (via grapheme-agent)
/// - AttentionMechanism: Dynamic focus allocation
///
/// # Architecture
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                         SelfAwareGrapheme                                │
/// │                           (AGI System)                                   │
/// ├─────────────────────────────────────────────────────────────────────────┤
/// │                                                                          │
/// │  ┌─────────────────────────────────────────────────────────────────────┐│
/// │  │                      AWARENESS LAYER                                 ││
/// │  │  AwarenessLevel ◄──► CognitiveSnapshot ◄──► Health Monitoring       ││
/// │  └─────────────────────────────────────────────────────────────────────┘│
/// │                                    │                                    │
/// │                                    ▼                                    │
/// │  ┌─────────────────────────────────────────────────────────────────────┐│
/// │  │                     COGNITION LAYER                                  ││
/// │  │  UnifiedCognition: Brain orchestration, attention-based routing     ││
/// │  │    ├── MathBrain, CodeBrain, VisionBrain, MusicBrain, ...          ││
/// │  │    ├── AttentionMechanism: Dynamic focus                            ││
/// │  │    └── SelfModel: Component tracking                                ││
/// │  └─────────────────────────────────────────────────────────────────────┘│
/// │                                    │                                    │
/// │                                    ▼                                    │
/// │  ┌─────────────────────────────────────────────────────────────────────┐│
/// │  │                    REFLECTION LAYER                                  ││
/// │  │  ReflectiveBrain: Introspection, problem detection, auto-healing   ││
/// │  └─────────────────────────────────────────────────────────────────────┘│
/// │                                    │                                    │
/// │                                    ▼                                    │
/// │  ┌─────────────────────────────────────────────────────────────────────┐│
/// │  │                     EXPERIENCE LAYER                                 ││
/// │  │  Learning from outcomes, pattern recognition, skill improvement     ││
/// │  └─────────────────────────────────────────────────────────────────────┘│
/// │                                                                          │
/// └─────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Time Complexity
/// - process(): O(n log n) for brain selection + O(k * m) for processing
/// - think(): O(n) for snapshot generation
/// - reflect(): O(n) for introspection
#[derive(Debug)]
pub struct SelfAwareGrapheme {
    /// Configuration
    pub config: SelfAwareConfig,
    /// Unified cognition system
    pub cognition: UnifiedCognition,
    /// Current awareness level
    pub awareness: AwarenessLevel,
    /// Cognitive snapshots history
    snapshots: Vec<CognitiveSnapshot>,
    /// Statistics
    pub stats: SelfAwareStats,
    /// Operation counter
    operation_counter: u64,
    /// Is system active
    active: bool,
}

impl Default for SelfAwareGrapheme {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfAwareGrapheme {
    /// Create a new self-aware GRAPHEME system
    pub fn new() -> Self {
        let mut cognition = UnifiedCognition::new();
        cognition.register_standard_brains();

        Self {
            config: SelfAwareConfig::default(),
            cognition,
            awareness: AwarenessLevel::Partial,
            snapshots: Vec::new(),
            stats: SelfAwareStats::default(),
            operation_counter: 0,
            active: false, // Start inactive until boot() is called
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SelfAwareConfig) -> Self {
        let mut system = Self::new();
        system.awareness = config.default_awareness;
        system.config = config;
        system
    }

    /// Boot the system (initialize all subsystems)
    pub fn boot(&mut self) {
        self.active = true;
        self.awareness = self.config.default_awareness;

        // Initial health check
        let _snapshot = self.take_snapshot(0);
    }

    /// Shutdown the system gracefully
    pub fn shutdown(&mut self) {
        self.active = false;
        self.awareness = AwarenessLevel::Minimal;
    }

    /// Check if system is active
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get current awareness level
    pub fn awareness_level(&self) -> AwarenessLevel {
        self.awareness
    }

    /// Set awareness level
    pub fn set_awareness(&mut self, level: AwarenessLevel) {
        if self.awareness != level {
            self.awareness = level;
            self.stats.awareness_changes += 1;
        }
    }

    /// Take a cognitive snapshot - O(n)
    pub fn take_snapshot(&mut self, timestamp: u64) -> CognitiveSnapshot {
        let health = self.cognition.health_overview();
        let focus = self.cognition.attention()
            .current_focus()
            .map(|f| f.to_vec())
            .unwrap_or_default();

        let snapshot = CognitiveSnapshot {
            timestamp,
            awareness_level: self.awareness,
            health_score: health.last_health_score.max(health.success_rate),
            confidence: self.cognition.stats().success_rate(),
            active_goals: 0, // Would integrate with GoalStack
            current_focus: focus,
            recent_operations: self.operation_counter,
            memory_pressure: health.average_load,
        };

        // Store snapshot with retention limit
        self.snapshots.push(snapshot.clone());
        if self.snapshots.len() > self.config.snapshot_retention {
            self.snapshots.remove(0);
        }

        // Record health in stats
        self.stats.record_health(snapshot.health_score, self.config.snapshot_retention);

        snapshot
    }

    /// Get the latest snapshot
    pub fn latest_snapshot(&self) -> Option<&CognitiveSnapshot> {
        self.snapshots.last()
    }

    /// Get snapshot history
    pub fn snapshot_history(&self) -> &[CognitiveSnapshot] {
        &self.snapshots
    }

    /// Main processing entry point - "think" about an input
    ///
    /// This is the primary interface for cognitive processing.
    pub fn think(&mut self, input: &str, timestamp: u64) -> ThinkResult {
        if !self.active {
            return ThinkResult::inactive();
        }

        self.operation_counter += 1;

        // Periodic health check
        if self.operation_counter.is_multiple_of(self.config.health_check_interval) {
            self.perform_health_check(timestamp);
        }

        // Process through unified cognition
        let cognition_result = self.cognition.process(input, timestamp);

        // Update statistics
        let success = cognition_result.is_valid();
        if success {
            self.stats.successful_ops += 1;
        } else {
            self.stats.failed_ops += 1;
        }
        self.stats.total_operations += 1;

        // Auto-adjust awareness if enabled
        if self.config.auto_adjust_awareness {
            self.adjust_awareness_if_needed();
        }

        ThinkResult {
            output: cognition_result.primary_output,
            source_brain: cognition_result.source_brain,
            confidence: cognition_result.confidence,
            processing_time_ms: cognition_result.processing_time_ms,
            awareness_level: self.awareness,
            was_fallback: cognition_result.was_fallback,
        }
    }

    /// Perform self-reflection - O(n)
    pub fn reflect(&mut self, timestamp: u64) -> ReflectionResult {
        // Perform introspection
        let introspection = self.cognition.perform_introspection();

        // Take snapshot
        let snapshot = self.take_snapshot(timestamp);

        // Check for problems
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();

        if introspection.health_score < 0.5 {
            issues.push("Low system health detected".to_string());
            suggestions.push("Consider reducing load or resetting components".to_string());
        }

        if !introspection.problematic_components.is_empty() {
            issues.push(format!("Problematic components: {:?}", introspection.problematic_components));
            suggestions.push("Review and potentially reset problematic components".to_string());
        }

        if self.stats.success_rate() < 0.5 {
            issues.push("Low operation success rate".to_string());
            suggestions.push("Review input patterns and brain configurations".to_string());
        }

        // Apply auto-corrections if possible
        let corrections_made = if !issues.is_empty() && self.config.enable_learning {
            self.attempt_self_correction()
        } else {
            0
        };

        if corrections_made > 0 {
            self.stats.self_corrections += corrections_made as u64;
        }

        ReflectionResult {
            snapshot,
            introspection,
            issues,
            suggestions,
            corrections_made,
        }
    }

    /// Perform health check
    fn perform_health_check(&mut self, timestamp: u64) {
        let _snapshot = self.take_snapshot(timestamp);

        // Trigger introspection if health is concerning
        if let Some(snapshot) = self.snapshots.last() {
            if snapshot.needs_attention() {
                self.cognition.perform_introspection();
            }
        }
    }

    /// Adjust awareness level based on system state
    fn adjust_awareness_if_needed(&mut self) {
        let health = self.cognition.health_overview();

        let new_level = if health.success_rate < 0.3 || health.average_load > 0.9 {
            AwarenessLevel::Heightened
        } else if health.success_rate > 0.8 && health.average_load < 0.5 {
            AwarenessLevel::Partial
        } else if health.success_rate > 0.95 && health.average_load < 0.3 {
            AwarenessLevel::Minimal
        } else {
            AwarenessLevel::Full
        };

        if new_level != self.awareness {
            self.set_awareness(new_level);
        }
    }

    /// Attempt self-correction - returns number of corrections made
    fn attempt_self_correction(&mut self) -> usize {
        let mut corrections = 0;

        // Check for overloaded brains and reduce load
        let health = self.cognition.health_overview();
        for _brain_id in &health.overloaded_brains {
            // In a real system, we might shed load or increase capacity
            // For now, just acknowledge the issue
            corrections += 1;
        }

        // Redistribute attention if some brains are failing
        let suggestions = self.cognition.suggest_attention_optimization();
        for (brain_id, new_bias) in suggestions {
            self.cognition.attention.set_brain_bias(&brain_id, new_bias);
            corrections += 1;
        }

        corrections
    }

    /// Get current statistics
    pub fn stats(&self) -> &SelfAwareStats {
        &self.stats
    }

    /// Get reference to underlying cognition system
    pub fn cognition(&self) -> &UnifiedCognition {
        &self.cognition
    }

    /// Get mutable reference to underlying cognition system
    pub fn cognition_mut(&mut self) -> &mut UnifiedCognition {
        &mut self.cognition
    }

    /// Reset the system to initial state
    pub fn reset(&mut self) {
        self.stats = SelfAwareStats::default();
        self.snapshots.clear();
        self.operation_counter = 0;
        self.awareness = self.config.default_awareness;
        self.cognition.reset_stats();
    }

    /// Get system summary
    pub fn summary(&self) -> SelfAwareSummary {
        let health = self.cognition.health_overview();

        SelfAwareSummary {
            active: self.active,
            awareness_level: self.awareness,
            total_operations: self.stats.total_operations,
            success_rate: self.stats.success_rate(),
            health_score: health.last_health_score,
            active_brains: health.active_brains,
            total_brains: health.total_brains,
            self_corrections: self.stats.self_corrections,
            average_health: self.stats.average_health(),
        }
    }
}

/// Result of thinking operation
#[derive(Debug, Clone)]
pub struct ThinkResult {
    /// Output of thinking
    pub output: Option<String>,
    /// Which brain produced the output
    pub source_brain: String,
    /// Confidence in the result
    pub confidence: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Current awareness level
    pub awareness_level: AwarenessLevel,
    /// Whether fallback was used
    pub was_fallback: bool,
}

impl ThinkResult {
    /// Create inactive result
    pub fn inactive() -> Self {
        Self {
            output: None,
            source_brain: String::new(),
            confidence: 0.0,
            processing_time_ms: 0,
            awareness_level: AwarenessLevel::Minimal,
            was_fallback: false,
        }
    }

    /// Check if result is valid
    pub fn is_valid(&self) -> bool {
        self.output.is_some() && self.confidence > 0.0
    }
}

/// Result of reflection operation
#[derive(Debug)]
pub struct ReflectionResult {
    /// Cognitive snapshot at time of reflection
    pub snapshot: CognitiveSnapshot,
    /// Detailed introspection result
    pub introspection: IntrospectionResult,
    /// Issues detected
    pub issues: Vec<String>,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
    /// Number of self-corrections made
    pub corrections_made: usize,
}

impl ReflectionResult {
    /// Check if reflection found no issues
    pub fn is_healthy(&self) -> bool {
        self.issues.is_empty() && self.snapshot.is_healthy()
    }
}

/// Summary of system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAwareSummary {
    /// Whether system is active
    pub active: bool,
    /// Current awareness level
    pub awareness_level: AwarenessLevel,
    /// Total operations performed
    pub total_operations: u64,
    /// Overall success rate
    pub success_rate: f32,
    /// Current health score
    pub health_score: f32,
    /// Number of active brains
    pub active_brains: usize,
    /// Total registered brains
    pub total_brains: usize,
    /// Self-corrections performed
    pub self_corrections: u64,
    /// Average health over time
    pub average_health: f32,
}

/// Factory function to create a self-aware GRAPHEME system
pub fn create_self_aware_grapheme() -> SelfAwareGrapheme {
    let mut system = SelfAwareGrapheme::new();
    system.boot();
    system
}

// ============================================================================
// SelfAwareGrapheme Tests
// ============================================================================

#[cfg(test)]
mod self_aware_tests {
    use super::*;

    #[test]
    fn test_self_aware_grapheme_creation() {
        let system = SelfAwareGrapheme::new();
        assert!(!system.active); // Not booted yet
        assert_eq!(system.awareness, AwarenessLevel::Partial);
    }

    #[test]
    fn test_self_aware_grapheme_boot() {
        let mut system = SelfAwareGrapheme::new();
        system.boot();

        assert!(system.is_active());
        assert!(!system.snapshots.is_empty());
    }

    #[test]
    fn test_self_aware_grapheme_shutdown() {
        let mut system = SelfAwareGrapheme::new();
        system.boot();
        system.shutdown();

        assert!(!system.is_active());
        assert_eq!(system.awareness, AwarenessLevel::Minimal);
    }

    #[test]
    fn test_self_aware_grapheme_think() {
        let mut system = SelfAwareGrapheme::new();
        system.boot();

        let result = system.think("Calculate 2 + 3", 1000);

        assert!(result.is_valid());
        assert!(!result.source_brain.is_empty());
    }

    #[test]
    fn test_self_aware_grapheme_think_inactive() {
        let mut system = SelfAwareGrapheme::new();
        // Don't boot

        let result = system.think("Calculate 2 + 3", 1000);

        assert!(!result.is_valid());
    }

    #[test]
    fn test_self_aware_grapheme_reflect() {
        let mut system = SelfAwareGrapheme::new();
        system.boot();

        // Process some inputs first
        for _ in 0..5 {
            system.think("Calculate 1 + 1", 1000);
        }

        let reflection = system.reflect(2000);

        assert!(reflection.snapshot.health_score >= 0.0);
        assert!(reflection.snapshot.health_score <= 1.0);
    }

    #[test]
    fn test_self_aware_grapheme_snapshot() {
        let mut system = SelfAwareGrapheme::new();
        system.boot();

        let snapshot = system.take_snapshot(1000);

        assert_eq!(snapshot.awareness_level, AwarenessLevel::Partial);
    }

    #[test]
    fn test_self_aware_grapheme_awareness_change() {
        let mut system = SelfAwareGrapheme::new();
        system.boot();

        system.set_awareness(AwarenessLevel::Heightened);

        assert_eq!(system.awareness, AwarenessLevel::Heightened);
        assert_eq!(system.stats.awareness_changes, 1);
    }

    #[test]
    fn test_self_aware_grapheme_stats() {
        let mut system = SelfAwareGrapheme::new();
        system.boot();

        system.think("test", 1000);
        system.think("test", 2000);

        assert_eq!(system.stats.total_operations, 2);
    }

    #[test]
    fn test_self_aware_grapheme_reset() {
        let mut system = SelfAwareGrapheme::new();
        system.boot();
        system.think("test", 1000);

        system.reset();

        assert_eq!(system.stats.total_operations, 0);
        assert!(system.snapshots.is_empty());
    }

    #[test]
    fn test_self_aware_grapheme_summary() {
        let mut system = SelfAwareGrapheme::new();
        system.boot();

        let summary = system.summary();

        assert!(summary.active);
        assert_eq!(summary.total_brains, 7);
    }

    #[test]
    fn test_self_aware_config() {
        let config = SelfAwareConfig {
            default_awareness: AwarenessLevel::Full,
            max_concurrent_ops: 10,
            ..Default::default()
        };

        let system = SelfAwareGrapheme::with_config(config);

        assert_eq!(system.awareness, AwarenessLevel::Full);
        assert_eq!(system.config.max_concurrent_ops, 10);
    }

    #[test]
    fn test_self_aware_stats_success_rate() {
        let stats = SelfAwareStats {
            successful_ops: 8,
            total_operations: 10,
            ..Default::default()
        };

        assert!((stats.success_rate() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_self_aware_stats_goal_rate() {
        let stats = SelfAwareStats {
            goals_completed: 7,
            goals_failed: 3,
            ..Default::default()
        };

        assert!((stats.goal_completion_rate() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_cognitive_snapshot_healthy() {
        let snapshot = CognitiveSnapshot {
            timestamp: 0,
            awareness_level: AwarenessLevel::Full,
            health_score: 0.9,
            confidence: 0.8,
            active_goals: 2,
            current_focus: vec!["MathBrain".to_string()],
            recent_operations: 100,
            memory_pressure: 0.3,
        };

        assert!(snapshot.is_healthy());
        assert!(!snapshot.needs_attention());
    }

    #[test]
    fn test_cognitive_snapshot_unhealthy() {
        let snapshot = CognitiveSnapshot {
            timestamp: 0,
            awareness_level: AwarenessLevel::Full,
            health_score: 0.3,
            confidence: 0.8,
            active_goals: 2,
            current_focus: vec![],
            recent_operations: 100,
            memory_pressure: 0.98,
        };

        assert!(!snapshot.is_healthy());
        assert!(snapshot.needs_attention());
    }

    #[test]
    fn test_think_result_is_valid() {
        let mut result = ThinkResult::inactive();
        assert!(!result.is_valid());

        result.output = Some("answer".to_string());
        result.confidence = 0.8;
        assert!(result.is_valid());
    }

    #[test]
    fn test_create_self_aware_grapheme_factory() {
        let system = create_self_aware_grapheme();

        assert!(system.is_active());
        assert_eq!(system.cognition.all_brain_status().len(), 7);
    }

    #[test]
    fn test_self_aware_periodic_health_check() {
        let mut system = SelfAwareGrapheme::new();
        system.config.health_check_interval = 5;
        system.boot();

        // Process exactly 5 inputs to trigger health check
        for i in 0..5 {
            system.think("test", i as u64 * 100);
        }

        // Should have taken additional snapshots
        assert!(system.snapshots.len() >= 2);
    }

    #[test]
    fn test_self_aware_snapshot_retention() {
        let mut system = SelfAwareGrapheme::new();
        system.config.snapshot_retention = 5;
        system.boot();

        // Take more snapshots than retention allows
        for i in 0..10 {
            system.take_snapshot(i as u64 * 100);
        }

        assert!(system.snapshots.len() <= 5);
    }

    #[test]
    fn test_awareness_level_default() {
        assert_eq!(AwarenessLevel::default(), AwarenessLevel::Partial);
    }
}

// ============================================================================
// CrossBrainTransfer: Share Learned Patterns Between Domain Brains
// ============================================================================

/// A transferable pattern that can be shared between brains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferablePattern {
    /// Unique pattern identifier
    pub id: String,
    /// Source brain that learned this pattern
    pub source_brain: String,
    /// Pattern type/category
    pub pattern_type: PatternType,
    /// Pattern data (serialized representation)
    pub data: Vec<f32>,
    /// Confidence score of the pattern
    pub confidence: f32,
    /// Number of times this pattern was successfully used
    pub usage_count: u64,
    /// Timestamp when pattern was created
    pub created_at: u64,
}

impl TransferablePattern {
    /// Create a new transferable pattern
    pub fn new(
        id: impl Into<String>,
        source_brain: impl Into<String>,
        pattern_type: PatternType,
        data: Vec<f32>,
        confidence: f32,
    ) -> Self {
        Self {
            id: id.into(),
            source_brain: source_brain.into(),
            pattern_type,
            data,
            confidence,
            usage_count: 0,
            created_at: 0,
        }
    }

    /// Check if pattern is reliable (high confidence and usage)
    pub fn is_reliable(&self) -> bool {
        self.confidence > 0.7 && self.usage_count >= 3
    }

    /// Compute similarity to another pattern - O(n)
    pub fn similarity(&self, other: &TransferablePattern) -> f32 {
        if self.data.len() != other.data.len() || self.data.is_empty() {
            return 0.0;
        }

        // Cosine similarity
        let dot: f32 = self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum();
        let mag_a: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.data.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
        }
    }
}

/// Types of patterns that can be transferred
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Structural pattern (graph topology)
    Structural,
    /// Sequential pattern (order relationships)
    Sequential,
    /// Compositional pattern (part-whole relationships)
    Compositional,
    /// Transformation pattern (input-output mapping)
    Transformation,
    /// Abstract concept
    Concept(String),
}

/// Result of a transfer attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferResult {
    /// Whether transfer was successful
    pub success: bool,
    /// Pattern that was transferred
    pub pattern_id: String,
    /// Source brain
    pub from_brain: String,
    /// Target brain
    pub to_brain: String,
    /// Adaptation score (how well pattern adapted)
    pub adaptation_score: f32,
    /// Any issues encountered
    pub issues: Vec<String>,
}

impl TransferResult {
    /// Create a successful transfer result
    pub fn success(pattern_id: &str, from: &str, to: &str, score: f32) -> Self {
        Self {
            success: true,
            pattern_id: pattern_id.to_string(),
            from_brain: from.to_string(),
            to_brain: to.to_string(),
            adaptation_score: score,
            issues: Vec::new(),
        }
    }

    /// Create a failed transfer result
    pub fn failure(pattern_id: &str, from: &str, to: &str, reason: &str) -> Self {
        Self {
            success: false,
            pattern_id: pattern_id.to_string(),
            from_brain: from.to_string(),
            to_brain: to.to_string(),
            adaptation_score: 0.0,
            issues: vec![reason.to_string()],
        }
    }
}

/// Configuration for cross-brain transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferConfig {
    /// Minimum confidence for a pattern to be transferable
    pub min_confidence: f32,
    /// Minimum usage count for transfer eligibility
    pub min_usage_count: u64,
    /// Maximum patterns to store per brain
    pub max_patterns_per_brain: usize,
    /// Similarity threshold for pattern deduplication
    pub similarity_threshold: f32,
    /// Enable automatic pattern discovery
    pub auto_discover: bool,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            min_usage_count: 2,
            max_patterns_per_brain: 100,
            similarity_threshold: 0.9,
            auto_discover: true,
        }
    }
}

/// Statistics for cross-brain transfer
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransferStats {
    /// Total patterns discovered
    pub patterns_discovered: u64,
    /// Total transfer attempts
    pub transfer_attempts: u64,
    /// Successful transfers
    pub successful_transfers: u64,
    /// Failed transfers
    pub failed_transfers: u64,
    /// Average adaptation score
    pub avg_adaptation_score: f32,
}

impl TransferStats {
    /// Record a transfer attempt
    pub fn record_transfer(&mut self, success: bool, adaptation_score: f32) {
        self.transfer_attempts += 1;
        if success {
            self.successful_transfers += 1;
            // Update running average
            let n = self.successful_transfers as f32;
            self.avg_adaptation_score = ((self.avg_adaptation_score * (n - 1.0)) + adaptation_score) / n;
        } else {
            self.failed_transfers += 1;
        }
    }

    /// Get transfer success rate - O(1)
    pub fn success_rate(&self) -> f32 {
        if self.transfer_attempts == 0 {
            0.0
        } else {
            self.successful_transfers as f32 / self.transfer_attempts as f32
        }
    }
}

/// CrossBrainTransfer: Enables sharing of learned patterns between brains
///
/// This module allows domain brains to share knowledge:
/// - MathBrain can share algebraic patterns with CodeBrain
/// - VisionBrain can share spatial patterns with MusicBrain
/// - Pattern adaptation ensures compatibility
///
/// # Architecture
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                      CrossBrainTransfer                                  │
/// │                                                                          │
/// │  ┌─────────────┐    discover()    ┌─────────────────────────────────┐  │
/// │  │  MathBrain  │ ──────────────► │     Pattern Repository          │  │
/// │  └─────────────┘                  │  [Structural, Sequential, ...]  │  │
/// │                                   └─────────────────────────────────┘  │
/// │  ┌─────────────┐    transfer()           │                            │
/// │  │  CodeBrain  │ ◄───────────────────────┘                            │
/// │  └─────────────┘                                                       │
/// │                                                                          │
/// │  Pattern Flow: Source → Repository → Adapt → Target                    │
/// └─────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Time Complexity
/// - discover_pattern(): O(n) for similarity check
/// - transfer(): O(1) lookup + O(n) adaptation
/// - find_similar(): O(n * m) where n = patterns, m = pattern size
#[derive(Debug, Clone)]
pub struct CrossBrainTransfer {
    /// Configuration
    pub config: TransferConfig,
    /// Pattern repository (brain_id -> patterns)
    patterns: std::collections::HashMap<String, Vec<TransferablePattern>>,
    /// Transfer history (recent transfers)
    history: Vec<TransferResult>,
    /// Statistics
    pub stats: TransferStats,
    /// Next pattern ID
    next_pattern_id: u64,
}

impl Default for CrossBrainTransfer {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossBrainTransfer {
    /// Create a new cross-brain transfer system
    pub fn new() -> Self {
        Self {
            config: TransferConfig::default(),
            patterns: std::collections::HashMap::new(),
            history: Vec::new(),
            stats: TransferStats::default(),
            next_pattern_id: 1,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: TransferConfig) -> Self {
        let mut transfer = Self::new();
        transfer.config = config;
        transfer
    }

    /// Discover and register a new pattern from a brain - O(n)
    pub fn discover_pattern(
        &mut self,
        source_brain: &str,
        pattern_type: PatternType,
        data: Vec<f32>,
        confidence: f32,
    ) -> Option<String> {
        // Check confidence threshold
        if confidence < self.config.min_confidence {
            return None;
        }

        // Check for similar existing patterns
        let brain_patterns = self.patterns.entry(source_brain.to_string()).or_default();

        // Create candidate pattern
        let pattern_id = format!("pattern_{}", self.next_pattern_id);
        let candidate = TransferablePattern::new(
            &pattern_id,
            source_brain,
            pattern_type.clone(),
            data,
            confidence,
        );

        // Check for duplicates
        for existing in brain_patterns.iter() {
            if existing.pattern_type == pattern_type
                && candidate.similarity(existing) > self.config.similarity_threshold
            {
                return None; // Too similar to existing
            }
        }

        // Check capacity
        if brain_patterns.len() >= self.config.max_patterns_per_brain {
            // Remove least confident pattern
            if let Some(min_idx) = brain_patterns
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            {
                brain_patterns.remove(min_idx);
            }
        }

        // Add pattern
        brain_patterns.push(candidate);
        self.next_pattern_id += 1;
        self.stats.patterns_discovered += 1;

        Some(pattern_id)
    }

    /// Transfer a pattern from one brain to another - O(n)
    pub fn transfer(
        &mut self,
        pattern_id: &str,
        from_brain: &str,
        to_brain: &str,
    ) -> TransferResult {
        // Find the pattern
        let pattern = self.patterns
            .get(from_brain)
            .and_then(|patterns| patterns.iter().find(|p| p.id == pattern_id))
            .cloned();

        let result = match pattern {
            Some(mut pattern) => {
                // Check if pattern is reliable enough
                if pattern.confidence < self.config.min_confidence {
                    TransferResult::failure(pattern_id, from_brain, to_brain, "Pattern confidence too low")
                } else {
                    // Adapt pattern for target brain
                    let adaptation_score = self.adapt_pattern(&mut pattern, to_brain);

                    if adaptation_score > 0.5 {
                        // Register in target brain
                        pattern.source_brain = format!("{}→{}", from_brain, to_brain);
                        pattern.confidence *= adaptation_score; // Reduce confidence after transfer

                        let target_patterns = self.patterns.entry(to_brain.to_string()).or_default();

                        // Check capacity
                        if target_patterns.len() < self.config.max_patterns_per_brain {
                            target_patterns.push(pattern);
                            TransferResult::success(pattern_id, from_brain, to_brain, adaptation_score)
                        } else {
                            TransferResult::failure(pattern_id, from_brain, to_brain, "Target brain at capacity")
                        }
                    } else {
                        TransferResult::failure(pattern_id, from_brain, to_brain, "Adaptation score too low")
                    }
                }
            }
            None => TransferResult::failure(pattern_id, from_brain, to_brain, "Pattern not found"),
        };

        // Record statistics
        self.stats.record_transfer(result.success, result.adaptation_score);

        // Store in history
        self.history.push(result.clone());
        if self.history.len() > 100 {
            self.history.remove(0);
        }

        result
    }

    /// Adapt a pattern for a target brain - O(n)
    fn adapt_pattern(&self, pattern: &mut TransferablePattern, target_brain: &str) -> f32 {
        // Simple adaptation: scale data based on brain compatibility
        let compatibility = self.brain_compatibility(&pattern.source_brain, target_brain);

        // Apply slight noise to adapted pattern (simulating adaptation)
        for value in &mut pattern.data {
            *value *= compatibility;
        }

        compatibility
    }

    /// Compute compatibility between two brains - O(1)
    fn brain_compatibility(&self, source: &str, target: &str) -> f32 {
        // Define brain compatibility matrix (simplified)
        let source_lower = source.to_lowercase();
        let target_lower = target.to_lowercase();

        // Same brain type = perfect compatibility
        if source_lower == target_lower {
            return 1.0;
        }

        // Domain-specific compatibility scores
        match (source_lower.as_str(), target_lower.as_str()) {
            // Math and Code are highly compatible
            (s, t) if (s.contains("math") && t.contains("code"))
                || (s.contains("code") && t.contains("math")) => 0.8,
            // Vision and Music share spatial/temporal patterns
            (s, t) if (s.contains("vision") && t.contains("music"))
                || (s.contains("music") && t.contains("vision")) => 0.6,
            // Text is somewhat compatible with everything
            (s, _) if s.contains("text") => 0.5,
            (_, t) if t.contains("text") => 0.5,
            // Default compatibility
            _ => 0.4,
        }
    }

    /// Find similar patterns across all brains - O(n * m)
    pub fn find_similar(&self, pattern: &TransferablePattern, threshold: f32) -> Vec<(String, String, f32)> {
        let mut similar = Vec::new();

        for (brain_id, patterns) in &self.patterns {
            for p in patterns {
                if p.id != pattern.id {
                    let sim = pattern.similarity(p);
                    if sim >= threshold {
                        similar.push((brain_id.clone(), p.id.clone(), sim));
                    }
                }
            }
        }

        // Sort by similarity descending
        similar.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        similar
    }

    /// Get patterns for a specific brain - O(1)
    pub fn get_patterns(&self, brain_id: &str) -> Option<&Vec<TransferablePattern>> {
        self.patterns.get(brain_id)
    }

    /// Get all pattern IDs - O(n)
    pub fn all_pattern_ids(&self) -> Vec<(String, String)> {
        self.patterns
            .iter()
            .flat_map(|(brain, patterns)| {
                patterns.iter().map(move |p| (brain.clone(), p.id.clone()))
            })
            .collect()
    }

    /// Get transfer history
    pub fn history(&self) -> &[TransferResult] {
        &self.history
    }

    /// Get statistics
    pub fn stats(&self) -> &TransferStats {
        &self.stats
    }

    /// Clear all patterns (reset)
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.history.clear();
        self.stats = TransferStats::default();
        self.next_pattern_id = 1;
    }

    /// Get total pattern count - O(n)
    pub fn total_patterns(&self) -> usize {
        self.patterns.values().map(|v| v.len()).sum()
    }
}

/// Factory function to create a cross-brain transfer system
pub fn create_cross_brain_transfer() -> CrossBrainTransfer {
    CrossBrainTransfer::new()
}

// ============================================================================
// CrossBrainTransfer Tests
// ============================================================================

#[cfg(test)]
mod transfer_tests {
    use super::*;

    #[test]
    fn test_cross_brain_transfer_creation() {
        let transfer = CrossBrainTransfer::new();
        assert_eq!(transfer.total_patterns(), 0);
        assert_eq!(transfer.stats.transfer_attempts, 0);
    }

    #[test]
    fn test_discover_pattern() {
        let mut transfer = CrossBrainTransfer::new();

        let pattern_id = transfer.discover_pattern(
            "MathBrain",
            PatternType::Structural,
            vec![1.0, 2.0, 3.0],
            0.8,
        );

        assert!(pattern_id.is_some());
        assert_eq!(transfer.total_patterns(), 1);
        assert_eq!(transfer.stats.patterns_discovered, 1);
    }

    #[test]
    fn test_discover_pattern_low_confidence() {
        let mut transfer = CrossBrainTransfer::new();

        let pattern_id = transfer.discover_pattern(
            "MathBrain",
            PatternType::Structural,
            vec![1.0, 2.0, 3.0],
            0.3, // Below threshold
        );

        assert!(pattern_id.is_none());
        assert_eq!(transfer.total_patterns(), 0);
    }

    #[test]
    fn test_discover_pattern_deduplication() {
        let mut transfer = CrossBrainTransfer::new();

        // First pattern
        let id1 = transfer.discover_pattern(
            "MathBrain",
            PatternType::Structural,
            vec![1.0, 0.0, 0.0],
            0.8,
        );
        assert!(id1.is_some());

        // Nearly identical pattern - should be rejected
        let id2 = transfer.discover_pattern(
            "MathBrain",
            PatternType::Structural,
            vec![1.0, 0.0, 0.0],
            0.9,
        );
        assert!(id2.is_none());

        assert_eq!(transfer.total_patterns(), 1);
    }

    #[test]
    fn test_transfer_pattern() {
        let mut transfer = CrossBrainTransfer::new();

        // Discover a pattern
        let pattern_id = transfer.discover_pattern(
            "MathBrain",
            PatternType::Transformation,
            vec![1.0, 2.0, 3.0],
            0.9,
        ).unwrap();

        // Transfer to CodeBrain
        let result = transfer.transfer(&pattern_id, "MathBrain", "CodeBrain");

        assert!(result.success);
        assert!(result.adaptation_score > 0.5);
        assert_eq!(transfer.stats.successful_transfers, 1);
    }

    #[test]
    fn test_transfer_pattern_not_found() {
        let mut transfer = CrossBrainTransfer::new();

        let result = transfer.transfer("nonexistent", "MathBrain", "CodeBrain");

        assert!(!result.success);
        assert!(result.issues.iter().any(|i| i.contains("not found")));
    }

    #[test]
    fn test_brain_compatibility() {
        let transfer = CrossBrainTransfer::new();

        // Math and Code are highly compatible
        let compat = transfer.brain_compatibility("MathBrain", "CodeBrain");
        assert!(compat >= 0.7);

        // Same brain = perfect
        let same = transfer.brain_compatibility("MathBrain", "MathBrain");
        assert!((same - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pattern_similarity() {
        let p1 = TransferablePattern::new(
            "p1", "Brain1", PatternType::Structural,
            vec![1.0, 0.0, 0.0], 0.9,
        );
        let p2 = TransferablePattern::new(
            "p2", "Brain2", PatternType::Structural,
            vec![1.0, 0.0, 0.0], 0.9,
        );
        let p3 = TransferablePattern::new(
            "p3", "Brain3", PatternType::Structural,
            vec![0.0, 1.0, 0.0], 0.9,
        );

        // Identical vectors
        assert!((p1.similarity(&p2) - 1.0).abs() < 0.01);

        // Orthogonal vectors
        assert!(p1.similarity(&p3).abs() < 0.01);
    }

    #[test]
    fn test_find_similar_patterns() {
        let mut transfer = CrossBrainTransfer::new();

        // Add patterns to different brains
        transfer.discover_pattern("BrainA", PatternType::Structural, vec![1.0, 0.0, 0.0], 0.8);
        transfer.discover_pattern("BrainB", PatternType::Structural, vec![0.9, 0.1, 0.0], 0.8);
        transfer.discover_pattern("BrainC", PatternType::Structural, vec![0.0, 1.0, 0.0], 0.8);

        let query = TransferablePattern::new(
            "query", "Query", PatternType::Structural,
            vec![1.0, 0.0, 0.0], 0.9,
        );

        let similar = transfer.find_similar(&query, 0.8);

        // Should find at least BrainA's pattern as highly similar
        assert!(!similar.is_empty());
    }

    #[test]
    fn test_transfer_stats() {
        let mut stats = TransferStats::default();

        stats.record_transfer(true, 0.8);
        stats.record_transfer(true, 0.9);
        stats.record_transfer(false, 0.0);

        assert_eq!(stats.transfer_attempts, 3);
        assert_eq!(stats.successful_transfers, 2);
        assert_eq!(stats.failed_transfers, 1);
        assert!((stats.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_pattern_is_reliable() {
        let mut pattern = TransferablePattern::new(
            "p1", "Brain", PatternType::Structural,
            vec![1.0], 0.8,
        );

        // Low usage - not reliable
        pattern.usage_count = 1;
        assert!(!pattern.is_reliable());

        // High usage - reliable
        pattern.usage_count = 5;
        assert!(pattern.is_reliable());

        // Low confidence - not reliable
        pattern.confidence = 0.5;
        assert!(!pattern.is_reliable());
    }

    #[test]
    fn test_transfer_config() {
        let config = TransferConfig {
            min_confidence: 0.8,
            min_usage_count: 5,
            ..Default::default()
        };

        let transfer = CrossBrainTransfer::with_config(config);

        assert_eq!(transfer.config.min_confidence, 0.8);
        assert_eq!(transfer.config.min_usage_count, 5);
    }

    #[test]
    fn test_transfer_clear() {
        let mut transfer = CrossBrainTransfer::new();

        transfer.discover_pattern("Brain", PatternType::Structural, vec![1.0], 0.9);
        assert_eq!(transfer.total_patterns(), 1);

        transfer.clear();

        assert_eq!(transfer.total_patterns(), 0);
        assert_eq!(transfer.stats.patterns_discovered, 0);
    }

    #[test]
    fn test_create_cross_brain_transfer_factory() {
        let transfer = create_cross_brain_transfer();
        assert_eq!(transfer.total_patterns(), 0);
    }

    #[test]
    fn test_transfer_result_constructors() {
        let success = TransferResult::success("p1", "A", "B", 0.8);
        assert!(success.success);
        assert!(success.issues.is_empty());

        let failure = TransferResult::failure("p1", "A", "B", "reason");
        assert!(!failure.success);
        assert!(!failure.issues.is_empty());
    }

    #[test]
    fn test_all_pattern_ids() {
        let mut transfer = CrossBrainTransfer::new();

        transfer.discover_pattern("BrainA", PatternType::Structural, vec![1.0], 0.8);
        transfer.discover_pattern("BrainB", PatternType::Sequential, vec![2.0], 0.8);

        let ids = transfer.all_pattern_ids();

        assert_eq!(ids.len(), 2);
    }
}

// ============================================================================
// CausalReasoning: Graph-based cause-effect inference (P-time algorithms)
// ============================================================================

/// Represents a causal event (cause or effect)
#[derive(Debug, Clone, PartialEq)]
pub struct CausalEvent {
    /// Unique identifier
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Timestamp when event occurred (simulation time)
    pub timestamp: u64,
    /// Confidence that this event actually occurred
    pub confidence: f32,
    /// Associated data/features
    pub features: Vec<f32>,
}

impl CausalEvent {
    /// Create a new causal event
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            timestamp: 0,
            confidence: 1.0,
            features: Vec::new(),
        }
    }

    /// Create event with timestamp
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Create event with confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Create event with features
    pub fn with_features(mut self, features: Vec<f32>) -> Self {
        self.features = features;
        self
    }
}

/// A causal link between two events (cause -> effect)
#[derive(Debug, Clone, PartialEq)]
pub struct CausalLink {
    /// ID of the cause event
    pub cause_id: String,
    /// ID of the effect event
    pub effect_id: String,
    /// Strength of the causal relationship [0, 1]
    pub strength: f32,
    /// Delay between cause and effect (in time units)
    pub delay: u64,
    /// How many times this link has been observed
    pub observation_count: u64,
    /// Type of causal relationship
    pub link_type: CausalLinkType,
}

impl CausalLink {
    /// Create a new causal link
    pub fn new(cause_id: impl Into<String>, effect_id: impl Into<String>, strength: f32) -> Self {
        Self {
            cause_id: cause_id.into(),
            effect_id: effect_id.into(),
            strength: strength.clamp(0.0, 1.0),
            delay: 0,
            observation_count: 1,
            link_type: CausalLinkType::Direct,
        }
    }

    /// Set the delay
    pub fn with_delay(mut self, delay: u64) -> Self {
        self.delay = delay;
        self
    }

    /// Set the link type
    pub fn with_type(mut self, link_type: CausalLinkType) -> Self {
        self.link_type = link_type;
        self
    }

    /// Check if this link is statistically significant
    pub fn is_significant(&self) -> bool {
        self.strength >= 0.5 && self.observation_count >= 3
    }

    /// Update strength based on new observation
    pub fn update_strength(&mut self, new_observation: f32) {
        // Exponential moving average
        let alpha = 0.3;
        self.strength = alpha * new_observation + (1.0 - alpha) * self.strength;
        self.observation_count += 1;
    }
}

/// Types of causal relationships
#[derive(Debug, Clone, PartialEq, Default)]
pub enum CausalLinkType {
    /// A directly causes B
    #[default]
    Direct,
    /// A and B have a common cause
    CommonCause,
    /// A indirectly causes B through intermediaries
    Indirect,
    /// Statistical association without clear direction
    Association,
    /// Bidirectional causation
    Bidirectional,
}

/// Result of a causal inference query
#[derive(Debug, Clone)]
pub struct CausalInference {
    /// The query that was asked
    pub query: String,
    /// Events in the causal chain
    pub chain: Vec<String>,
    /// Overall confidence in the inference
    pub confidence: f32,
    /// Explanation of the reasoning
    pub explanation: String,
    /// Alternative explanations considered
    pub alternatives: Vec<String>,
}

impl CausalInference {
    /// Create a new inference result
    pub fn new(query: impl Into<String>, chain: Vec<String>, confidence: f32) -> Self {
        Self {
            query: query.into(),
            chain,
            confidence: confidence.clamp(0.0, 1.0),
            explanation: String::new(),
            alternatives: Vec::new(),
        }
    }

    /// Check if inference is reliable
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.7 && !self.chain.is_empty()
    }

    /// Add explanation
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = explanation.into();
        self
    }
}

/// Configuration for causal reasoning
#[derive(Debug, Clone)]
pub struct CausalConfig {
    /// Minimum strength to consider a link valid
    pub min_link_strength: f32,
    /// Maximum chain length to search
    pub max_chain_length: usize,
    /// Minimum observations for statistical significance
    pub min_observations: u64,
    /// Whether to consider indirect effects
    pub include_indirect: bool,
    /// Decay factor for confidence over chain length
    pub chain_decay: f32,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            min_link_strength: 0.3,
            max_chain_length: 5,
            min_observations: 2,
            include_indirect: true,
            chain_decay: 0.9,
        }
    }
}

/// Statistics for causal reasoning operations
#[derive(Debug, Clone, Default)]
pub struct CausalStats {
    /// Total queries performed
    pub queries_performed: u64,
    /// Successful inferences
    pub successful_inferences: u64,
    /// Failed inferences
    pub failed_inferences: u64,
    /// Links discovered
    pub links_discovered: u64,
    /// Average chain length
    pub avg_chain_length: f32,
}

impl CausalStats {
    /// Record a query result
    pub fn record_query(&mut self, success: bool, chain_length: usize) {
        self.queries_performed += 1;
        if success {
            self.successful_inferences += 1;
            // Update running average
            let n = self.successful_inferences as f32;
            self.avg_chain_length = ((n - 1.0) * self.avg_chain_length + chain_length as f32) / n;
        } else {
            self.failed_inferences += 1;
        }
    }

    /// Success rate
    pub fn success_rate(&self) -> f32 {
        if self.queries_performed == 0 {
            0.0
        } else {
            self.successful_inferences as f32 / self.queries_performed as f32
        }
    }
}

/// Main causal reasoning engine
/// Uses P-time graph algorithms (BFS, topological sort)
#[derive(Debug)]
pub struct CausalReasoning {
    /// Configuration
    pub config: CausalConfig,
    /// Registered events
    events: std::collections::HashMap<String, CausalEvent>,
    /// Causal links (adjacency list: cause_id -> effects)
    links: std::collections::HashMap<String, Vec<CausalLink>>,
    /// Reverse links for backward inference (effect_id -> causes)
    reverse_links: std::collections::HashMap<String, Vec<String>>,
    /// Statistics
    pub stats: CausalStats,
}

impl CausalReasoning {
    /// Create a new causal reasoning engine
    pub fn new() -> Self {
        Self {
            config: CausalConfig::default(),
            events: std::collections::HashMap::new(),
            links: std::collections::HashMap::new(),
            reverse_links: std::collections::HashMap::new(),
            stats: CausalStats::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: CausalConfig) -> Self {
        Self {
            config,
            events: std::collections::HashMap::new(),
            links: std::collections::HashMap::new(),
            reverse_links: std::collections::HashMap::new(),
            stats: CausalStats::default(),
        }
    }

    /// Register a causal event
    pub fn register_event(&mut self, event: CausalEvent) {
        self.events.insert(event.id.clone(), event);
    }

    /// Get an event by ID
    pub fn get_event(&self, id: &str) -> Option<&CausalEvent> {
        self.events.get(id)
    }

    /// Add a causal link between events
    pub fn add_link(&mut self, link: CausalLink) -> bool {
        // Verify both events exist
        if !self.events.contains_key(&link.cause_id) ||
           !self.events.contains_key(&link.effect_id) {
            return false;
        }

        // Add to forward links
        self.links
            .entry(link.cause_id.clone())
            .or_default()
            .push(link.clone());

        // Add to reverse links
        self.reverse_links
            .entry(link.effect_id.clone())
            .or_default()
            .push(link.cause_id.clone());

        self.stats.links_discovered += 1;
        true
    }

    /// Observe a causal relationship (updates existing or creates new)
    pub fn observe(&mut self, cause_id: &str, effect_id: &str, strength: f32) -> bool {
        if !self.events.contains_key(cause_id) || !self.events.contains_key(effect_id) {
            return false;
        }

        // Check if link exists
        if let Some(links) = self.links.get_mut(cause_id) {
            for link in links.iter_mut() {
                if link.effect_id == effect_id {
                    link.update_strength(strength);
                    return true;
                }
            }
        }

        // Create new link
        let link = CausalLink::new(cause_id, effect_id, strength);
        self.add_link(link)
    }

    /// Get direct effects of an event
    pub fn get_effects(&self, cause_id: &str) -> Vec<&CausalLink> {
        self.links
            .get(cause_id)
            .map(|links| links.iter().collect())
            .unwrap_or_default()
    }

    /// Get direct causes of an event
    pub fn get_causes(&self, effect_id: &str) -> Vec<&str> {
        self.reverse_links
            .get(effect_id)
            .map(|causes| causes.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Find causal chain from cause to effect using BFS (P-time: O(V + E))
    pub fn find_chain(&mut self, cause_id: &str, effect_id: &str) -> Option<CausalInference> {
        if !self.events.contains_key(cause_id) || !self.events.contains_key(effect_id) {
            self.stats.record_query(false, 0);
            return None;
        }

        // BFS to find shortest causal path
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut parent: std::collections::HashMap<String, String> = std::collections::HashMap::new();

        queue.push_back(cause_id.to_string());
        visited.insert(cause_id.to_string());

        while let Some(current) = queue.pop_front() {
            if current == effect_id {
                // Reconstruct path
                let chain = self.reconstruct_path(&parent, cause_id, effect_id);
                let confidence = self.calculate_chain_confidence(&chain);
                let inference = CausalInference::new(
                    format!("Does {} cause {}?", cause_id, effect_id),
                    chain.clone(),
                    confidence,
                ).with_explanation(format!(
                    "Found causal chain of length {} with confidence {:.2}",
                    chain.len(), confidence
                ));
                self.stats.record_query(true, chain.len());
                return Some(inference);
            }

            if visited.len() >= self.config.max_chain_length {
                break;
            }

            // Explore effects
            if let Some(links) = self.links.get(&current) {
                for link in links {
                    if link.strength >= self.config.min_link_strength &&
                       !visited.contains(&link.effect_id) {
                        visited.insert(link.effect_id.clone());
                        parent.insert(link.effect_id.clone(), current.clone());
                        queue.push_back(link.effect_id.clone());
                    }
                }
            }
        }

        self.stats.record_query(false, 0);
        None
    }

    /// Reconstruct path from parent map
    fn reconstruct_path(
        &self,
        parent: &std::collections::HashMap<String, String>,
        start: &str,
        end: &str,
    ) -> Vec<String> {
        let mut path = vec![end.to_string()];
        let mut current = end.to_string();

        while current != start {
            if let Some(p) = parent.get(&current) {
                path.push(p.clone());
                current = p.clone();
            } else {
                break;
            }
        }

        path.reverse();
        path
    }

    /// Calculate confidence of a causal chain
    fn calculate_chain_confidence(&self, chain: &[String]) -> f32 {
        if chain.len() < 2 {
            return 0.0;
        }

        let mut confidence = 1.0;

        for i in 0..chain.len() - 1 {
            if let Some(links) = self.links.get(&chain[i]) {
                for link in links {
                    if link.effect_id == chain[i + 1] {
                        confidence *= link.strength * self.config.chain_decay;
                        break;
                    }
                }
            }
        }

        confidence
    }

    /// Find all effects reachable from a cause (P-time: O(V + E))
    pub fn find_all_effects(&self, cause_id: &str) -> Vec<String> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back(cause_id.to_string());
        visited.insert(cause_id.to_string());

        while let Some(current) = queue.pop_front() {
            if current != cause_id {
                result.push(current.clone());
            }

            if let Some(links) = self.links.get(&current) {
                for link in links {
                    if !visited.contains(&link.effect_id) {
                        visited.insert(link.effect_id.clone());
                        queue.push_back(link.effect_id.clone());
                    }
                }
            }
        }

        result
    }

    /// Find root causes of an effect (P-time: O(V + E))
    pub fn find_root_causes(&self, effect_id: &str) -> Vec<String> {
        let mut visited = std::collections::HashSet::new();
        let mut roots = Vec::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back(effect_id.to_string());
        visited.insert(effect_id.to_string());

        while let Some(current) = queue.pop_front() {
            let causes = self.get_causes(&current);

            if causes.is_empty() && current != effect_id {
                roots.push(current);
            } else {
                for cause in causes {
                    if !visited.contains(cause) {
                        visited.insert(cause.to_string());
                        queue.push_back(cause.to_string());
                    }
                }
            }
        }

        roots
    }

    /// Infer causal strength from data (correlation-based, P-time)
    pub fn infer_strength(&self, cause: &CausalEvent, effect: &CausalEvent) -> f32 {
        if cause.features.is_empty() || effect.features.is_empty() {
            // Use timestamp-based inference
            if cause.timestamp < effect.timestamp {
                // Cause precedes effect - base confidence
                0.5 * cause.confidence * effect.confidence
            } else {
                0.0
            }
        } else {
            // Feature correlation (simplified Pearson)
            let min_len = cause.features.len().min(effect.features.len());
            if min_len == 0 {
                return 0.0;
            }

            let sum_product: f32 = cause.features.iter()
                .zip(effect.features.iter())
                .take(min_len)
                .map(|(a, b)| a * b)
                .sum();

            let mag_cause: f32 = cause.features.iter().take(min_len).map(|x| x * x).sum::<f32>().sqrt();
            let mag_effect: f32 = effect.features.iter().take(min_len).map(|x| x * x).sum::<f32>().sqrt();

            if mag_cause == 0.0 || mag_effect == 0.0 {
                0.0
            } else {
                // Convert cosine similarity to [0, 1] strength
                let cosine = sum_product / (mag_cause * mag_effect);
                ((cosine + 1.0) / 2.0).clamp(0.0, 1.0)
            }
        }
    }

    /// Get counterfactual: "If not A, would B happen?"
    pub fn counterfactual(&mut self, cause_id: &str, effect_id: &str) -> CausalInference {
        // Check if there's a direct path
        if let Some(inference) = self.find_chain(cause_id, effect_id) {
            // Check for alternative causes
            let other_causes: Vec<_> = self.get_causes(effect_id)
                .into_iter()
                .filter(|c| *c != cause_id)
                .collect();

            let mut result = CausalInference::new(
                format!("If not {}, would {} still happen?", cause_id, effect_id),
                inference.chain,
                1.0 - inference.confidence,
            );

            if other_causes.is_empty() {
                result.explanation = format!(
                    "{} is the only known cause of {}. Without it, {} likely would not occur.",
                    cause_id, effect_id, effect_id
                );
            } else {
                result.explanation = format!(
                    "{} could still occur through alternative causes: {:?}",
                    effect_id, other_causes
                );
                result.alternatives = other_causes.iter().map(|s| s.to_string()).collect();
                result.confidence = 0.5; // Uncertain
            }

            result
        } else {
            CausalInference::new(
                format!("If not {}, would {} still happen?", cause_id, effect_id),
                vec![],
                1.0, // High confidence B would still happen (no causal link)
            ).with_explanation(format!(
                "No causal link found between {} and {}. {} is independent.",
                cause_id, effect_id, effect_id
            ))
        }
    }

    /// Get intervention analysis: "What happens if we force A?"
    pub fn intervention(&self, cause_id: &str) -> Vec<(String, f32)> {
        let effects = self.find_all_effects(cause_id);
        let mut result = Vec::new();

        for effect_id in effects {
            // Calculate cumulative effect strength
            let strength = self.calculate_intervention_strength(cause_id, &effect_id);
            result.push((effect_id, strength));
        }

        // Sort by strength descending
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Calculate intervention strength (how much forcing cause affects effect)
    fn calculate_intervention_strength(&self, cause_id: &str, effect_id: &str) -> f32 {
        // Use direct link strength if available
        if let Some(links) = self.links.get(cause_id) {
            for link in links {
                if link.effect_id == effect_id {
                    return link.strength;
                }
            }
        }

        // Otherwise estimate from path
        let mut visited = std::collections::HashSet::new();
        let mut max_strength = 0.0f32;

        self.dfs_strength(cause_id, effect_id, 1.0, &mut visited, &mut max_strength);
        max_strength
    }

    /// DFS to find maximum strength path
    fn dfs_strength(
        &self,
        current: &str,
        target: &str,
        current_strength: f32,
        visited: &mut std::collections::HashSet<String>,
        max_strength: &mut f32,
    ) {
        if current == target {
            *max_strength = max_strength.max(current_strength);
            return;
        }

        if visited.len() >= self.config.max_chain_length {
            return;
        }

        visited.insert(current.to_string());

        if let Some(links) = self.links.get(current) {
            for link in links {
                if !visited.contains(&link.effect_id) {
                    self.dfs_strength(
                        &link.effect_id,
                        target,
                        current_strength * link.strength * self.config.chain_decay,
                        visited,
                        max_strength,
                    );
                }
            }
        }

        visited.remove(current);
    }

    /// Total number of events
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Total number of links
    pub fn link_count(&self) -> usize {
        self.links.values().map(|v| v.len()).sum()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.events.clear();
        self.links.clear();
        self.reverse_links.clear();
        self.stats = CausalStats::default();
    }

    /// Get all event IDs
    pub fn all_event_ids(&self) -> Vec<&str> {
        self.events.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for CausalReasoning {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory function for creating CausalReasoning
pub fn create_causal_reasoning() -> CausalReasoning {
    CausalReasoning::new()
}

#[cfg(test)]
mod causal_tests {
    use super::*;

    #[test]
    fn test_causal_event_creation() {
        let event = CausalEvent::new("e1", "Test event")
            .with_timestamp(100)
            .with_confidence(0.9)
            .with_features(vec![1.0, 2.0]);

        assert_eq!(event.id, "e1");
        assert_eq!(event.timestamp, 100);
        assert_eq!(event.confidence, 0.9);
        assert_eq!(event.features.len(), 2);
    }

    #[test]
    fn test_causal_link_creation() {
        let link = CausalLink::new("cause", "effect", 0.8)
            .with_delay(10)
            .with_type(CausalLinkType::Direct);

        assert_eq!(link.cause_id, "cause");
        assert_eq!(link.effect_id, "effect");
        assert_eq!(link.strength, 0.8);
        assert_eq!(link.delay, 10);
    }

    #[test]
    fn test_link_significance() {
        let mut link = CausalLink::new("a", "b", 0.7);
        link.observation_count = 5;
        assert!(link.is_significant());

        link.strength = 0.3;
        assert!(!link.is_significant());
    }

    #[test]
    fn test_link_update_strength() {
        let mut link = CausalLink::new("a", "b", 0.5);
        link.update_strength(1.0);

        assert!(link.strength > 0.5);
        assert_eq!(link.observation_count, 2);
    }

    #[test]
    fn test_causal_inference_reliability() {
        let reliable = CausalInference::new("query", vec!["a".to_string()], 0.8);
        assert!(reliable.is_reliable());

        let unreliable = CausalInference::new("query", vec![], 0.8);
        assert!(!unreliable.is_reliable());
    }

    #[test]
    fn test_causal_reasoning_creation() {
        let engine = CausalReasoning::new();
        assert_eq!(engine.event_count(), 0);
        assert_eq!(engine.link_count(), 0);
    }

    #[test]
    fn test_register_event() {
        let mut engine = CausalReasoning::new();
        let event = CausalEvent::new("e1", "Event 1");

        engine.register_event(event);

        assert_eq!(engine.event_count(), 1);
        assert!(engine.get_event("e1").is_some());
    }

    #[test]
    fn test_add_link() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("cause", "Cause"));
        engine.register_event(CausalEvent::new("effect", "Effect"));

        let link = CausalLink::new("cause", "effect", 0.8);
        assert!(engine.add_link(link));
        assert_eq!(engine.link_count(), 1);
    }

    #[test]
    fn test_add_link_missing_event() {
        let mut engine = CausalReasoning::new();
        engine.register_event(CausalEvent::new("cause", "Cause"));

        let link = CausalLink::new("cause", "missing", 0.8);
        assert!(!engine.add_link(link));
    }

    #[test]
    fn test_get_effects() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));
        engine.register_event(CausalEvent::new("c", "C"));

        engine.add_link(CausalLink::new("a", "b", 0.8));
        engine.add_link(CausalLink::new("a", "c", 0.6));

        let effects = engine.get_effects("a");
        assert_eq!(effects.len(), 2);
    }

    #[test]
    fn test_get_causes() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));
        engine.register_event(CausalEvent::new("c", "C"));

        engine.add_link(CausalLink::new("a", "c", 0.8));
        engine.add_link(CausalLink::new("b", "c", 0.6));

        let causes = engine.get_causes("c");
        assert_eq!(causes.len(), 2);
    }

    #[test]
    fn test_find_chain_direct() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));

        engine.add_link(CausalLink::new("a", "b", 0.8));

        let inference = engine.find_chain("a", "b");
        assert!(inference.is_some());

        let inf = inference.unwrap();
        assert_eq!(inf.chain.len(), 2);
        assert!(inf.confidence > 0.0);
    }

    #[test]
    fn test_find_chain_indirect() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));
        engine.register_event(CausalEvent::new("c", "C"));

        engine.add_link(CausalLink::new("a", "b", 0.8));
        engine.add_link(CausalLink::new("b", "c", 0.8));

        let inference = engine.find_chain("a", "c");
        assert!(inference.is_some());

        let inf = inference.unwrap();
        assert_eq!(inf.chain.len(), 3);
    }

    #[test]
    fn test_find_chain_no_path() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));

        let inference = engine.find_chain("a", "b");
        assert!(inference.is_none());
    }

    #[test]
    fn test_find_all_effects() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));
        engine.register_event(CausalEvent::new("c", "C"));
        engine.register_event(CausalEvent::new("d", "D"));

        engine.add_link(CausalLink::new("a", "b", 0.8));
        engine.add_link(CausalLink::new("b", "c", 0.8));
        engine.add_link(CausalLink::new("b", "d", 0.8));

        let effects = engine.find_all_effects("a");
        assert_eq!(effects.len(), 3); // b, c, d
    }

    #[test]
    fn test_find_root_causes() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("root1", "Root 1"));
        engine.register_event(CausalEvent::new("root2", "Root 2"));
        engine.register_event(CausalEvent::new("mid", "Middle"));
        engine.register_event(CausalEvent::new("effect", "Effect"));

        engine.add_link(CausalLink::new("root1", "mid", 0.8));
        engine.add_link(CausalLink::new("root2", "mid", 0.8));
        engine.add_link(CausalLink::new("mid", "effect", 0.8));

        let roots = engine.find_root_causes("effect");
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_observe() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));

        assert!(engine.observe("a", "b", 0.5));
        assert_eq!(engine.link_count(), 1);

        // Update existing
        assert!(engine.observe("a", "b", 0.9));
        assert_eq!(engine.link_count(), 1);

        let effects = engine.get_effects("a");
        assert!(effects[0].strength > 0.5);
    }

    #[test]
    fn test_infer_strength_timestamp() {
        let engine = CausalReasoning::new();

        let cause = CausalEvent::new("c", "Cause").with_timestamp(0);
        let effect = CausalEvent::new("e", "Effect").with_timestamp(10);

        let strength = engine.infer_strength(&cause, &effect);
        assert!(strength > 0.0);
    }

    #[test]
    fn test_infer_strength_features() {
        let engine = CausalReasoning::new();

        let cause = CausalEvent::new("c", "Cause").with_features(vec![1.0, 0.0, 0.0]);
        let effect = CausalEvent::new("e", "Effect").with_features(vec![1.0, 0.0, 0.0]);

        let strength = engine.infer_strength(&cause, &effect);
        assert_eq!(strength, 1.0); // Perfect correlation
    }

    #[test]
    fn test_counterfactual() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));
        engine.add_link(CausalLink::new("a", "b", 0.8));

        let cf = engine.counterfactual("a", "b");
        assert!(!cf.explanation.is_empty());
    }

    #[test]
    fn test_intervention() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));
        engine.register_event(CausalEvent::new("c", "C"));

        engine.add_link(CausalLink::new("a", "b", 0.8));
        engine.add_link(CausalLink::new("a", "c", 0.5));

        let effects = engine.intervention("a");
        assert_eq!(effects.len(), 2);
        // Should be sorted by strength
        assert!(effects[0].1 >= effects[1].1);
    }

    #[test]
    fn test_causal_stats() {
        let mut stats = CausalStats::default();

        stats.record_query(true, 3);
        stats.record_query(true, 5);
        stats.record_query(false, 0);

        assert_eq!(stats.queries_performed, 3);
        assert_eq!(stats.successful_inferences, 2);
        assert_eq!(stats.failed_inferences, 1);
        assert!((stats.success_rate() - 0.666).abs() < 0.01);
        assert_eq!(stats.avg_chain_length, 4.0);
    }

    #[test]
    fn test_causal_config_default() {
        let config = CausalConfig::default();
        assert_eq!(config.max_chain_length, 5);
        assert!(config.include_indirect);
    }

    #[test]
    fn test_clear() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));
        engine.add_link(CausalLink::new("a", "b", 0.8));

        engine.clear();

        assert_eq!(engine.event_count(), 0);
        assert_eq!(engine.link_count(), 0);
    }

    #[test]
    fn test_all_event_ids() {
        let mut engine = CausalReasoning::new();

        engine.register_event(CausalEvent::new("a", "A"));
        engine.register_event(CausalEvent::new("b", "B"));

        let ids = engine.all_event_ids();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_create_causal_reasoning_factory() {
        let engine = create_causal_reasoning();
        assert_eq!(engine.event_count(), 0);
    }

    #[test]
    fn test_causal_link_type_default() {
        let link_type = CausalLinkType::default();
        assert_eq!(link_type, CausalLinkType::Direct);
    }
}

// ============================================================================
// MemoryConsolidation: Long-term knowledge persistence across sessions
// ============================================================================

/// Represents a memory entry that can be stored/retrieved
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: String,
    /// Category/type of memory
    pub category: MemoryCategory,
    /// The actual content/data
    pub content: String,
    /// Associated embedding vector for similarity search
    pub embedding: Vec<f32>,
    /// Importance score [0, 1]
    pub importance: f32,
    /// Access count (how often retrieved)
    pub access_count: u64,
    /// Last access timestamp
    pub last_access: u64,
    /// Creation timestamp
    pub created_at: u64,
    /// Associated tags for organization
    pub tags: Vec<String>,
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new(id: impl Into<String>, category: MemoryCategory, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            category,
            content: content.into(),
            embedding: Vec::new(),
            importance: 0.5,
            access_count: 0,
            last_access: 0,
            created_at: 0,
            tags: Vec::new(),
        }
    }

    /// Set embedding vector
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }

    /// Set importance
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Set tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set creation timestamp
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.created_at = timestamp;
        self.last_access = timestamp;
        self
    }

    /// Record an access
    pub fn record_access(&mut self, timestamp: u64) {
        self.access_count += 1;
        self.last_access = timestamp;
    }

    /// Calculate recency score based on time decay
    pub fn recency_score(&self, current_time: u64, decay_rate: f32) -> f32 {
        let age = current_time.saturating_sub(self.last_access) as f32;
        (-decay_rate * age).exp()
    }

    /// Calculate relevance combining importance and recency
    pub fn relevance(&self, current_time: u64, decay_rate: f32) -> f32 {
        let recency = self.recency_score(current_time, decay_rate);
        let frequency_boost = (self.access_count as f32).ln_1p() / 10.0;
        (self.importance * 0.4 + recency * 0.4 + frequency_boost * 0.2).clamp(0.0, 1.0)
    }

    /// Compute similarity to another entry using cosine similarity
    pub fn similarity(&self, other: &MemoryEntry) -> f32 {
        if self.embedding.is_empty() || other.embedding.is_empty() {
            return 0.0;
        }

        let min_len = self.embedding.len().min(other.embedding.len());
        let dot: f32 = self.embedding.iter()
            .zip(other.embedding.iter())
            .take(min_len)
            .map(|(a, b)| a * b)
            .sum();

        let mag_a: f32 = self.embedding.iter().take(min_len).map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.embedding.iter().take(min_len).map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
        }
    }
}

/// Categories of memory for organization
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MemoryCategory {
    /// Factual knowledge
    #[default]
    Fact,
    /// Procedural knowledge (how to do things)
    Procedure,
    /// Episodic memory (specific events/experiences)
    Episode,
    /// Semantic memory (concepts and meanings)
    Semantic,
    /// Skills and learned behaviors
    Skill,
    /// User preferences and context
    Preference,
    /// Custom category
    Custom(String),
}

/// Result of a memory consolidation operation
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Memories that were consolidated
    pub consolidated: Vec<String>,
    /// Memories that were strengthened
    pub strengthened: Vec<String>,
    /// Memories that were pruned (forgotten)
    pub pruned: Vec<String>,
    /// New associations discovered
    pub new_associations: usize,
    /// Overall consolidation score
    pub score: f32,
}

impl ConsolidationResult {
    /// Create empty result
    pub fn empty() -> Self {
        Self {
            consolidated: Vec::new(),
            strengthened: Vec::new(),
            pruned: Vec::new(),
            new_associations: 0,
            score: 0.0,
        }
    }

    /// Total operations performed
    pub fn total_operations(&self) -> usize {
        self.consolidated.len() + self.strengthened.len() + self.pruned.len()
    }
}

/// Configuration for memory consolidation
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Maximum number of memories to store
    pub max_memories: usize,
    /// Minimum importance to keep during pruning
    pub min_importance: f32,
    /// Time decay rate for recency
    pub decay_rate: f32,
    /// Similarity threshold for consolidation
    pub similarity_threshold: f32,
    /// How many recent memories to always keep
    pub recent_buffer: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            max_memories: 10000,
            min_importance: 0.1,
            decay_rate: 0.001,
            similarity_threshold: 0.8,
            recent_buffer: 100,
        }
    }
}

/// Statistics for memory operations
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total memories stored
    pub total_stored: u64,
    /// Total retrievals
    pub total_retrievals: u64,
    /// Successful retrievals
    pub successful_retrievals: u64,
    /// Consolidations performed
    pub consolidations: u64,
    /// Memories pruned
    pub memories_pruned: u64,
    /// Average retrieval similarity
    pub avg_retrieval_similarity: f32,
}

impl MemoryStats {
    /// Record a retrieval
    pub fn record_retrieval(&mut self, found: bool, similarity: f32) {
        self.total_retrievals += 1;
        if found {
            self.successful_retrievals += 1;
            // Update running average
            let n = self.successful_retrievals as f32;
            self.avg_retrieval_similarity =
                ((n - 1.0) * self.avg_retrieval_similarity + similarity) / n;
        }
    }

    /// Retrieval success rate
    pub fn retrieval_rate(&self) -> f32 {
        if self.total_retrievals == 0 {
            0.0
        } else {
            self.successful_retrievals as f32 / self.total_retrievals as f32
        }
    }
}

/// Main memory consolidation system
/// Manages long-term knowledge storage with consolidation and pruning
#[derive(Debug)]
pub struct MemoryConsolidation {
    /// Configuration
    pub config: ConsolidationConfig,
    /// Stored memories by ID
    memories: std::collections::HashMap<String, MemoryEntry>,
    /// Index by category
    category_index: std::collections::HashMap<String, Vec<String>>,
    /// Index by tag
    tag_index: std::collections::HashMap<String, Vec<String>>,
    /// Current timestamp (simulation time)
    current_time: u64,
    /// Statistics
    pub stats: MemoryStats,
    /// Next memory ID
    next_id: u64,
}

impl MemoryConsolidation {
    /// Create a new memory consolidation system
    pub fn new() -> Self {
        Self {
            config: ConsolidationConfig::default(),
            memories: std::collections::HashMap::new(),
            category_index: std::collections::HashMap::new(),
            tag_index: std::collections::HashMap::new(),
            current_time: 0,
            stats: MemoryStats::default(),
            next_id: 1,
        }
    }

    /// Create with custom config
    pub fn with_config(config: ConsolidationConfig) -> Self {
        Self {
            config,
            memories: std::collections::HashMap::new(),
            category_index: std::collections::HashMap::new(),
            tag_index: std::collections::HashMap::new(),
            current_time: 0,
            stats: MemoryStats::default(),
            next_id: 1,
        }
    }

    /// Advance simulation time
    pub fn tick(&mut self, delta: u64) {
        self.current_time += delta;
    }

    /// Set current time
    pub fn set_time(&mut self, time: u64) {
        self.current_time = time;
    }

    /// Store a new memory
    pub fn store(&mut self, mut entry: MemoryEntry) -> String {
        // Assign ID if not set
        if entry.id.is_empty() {
            entry.id = format!("mem_{}", self.next_id);
            self.next_id += 1;
        }

        // Set timestamp if not set
        if entry.created_at == 0 {
            entry.created_at = self.current_time;
            entry.last_access = self.current_time;
        }

        let id = entry.id.clone();

        // Update category index
        let category_key = format!("{:?}", entry.category);
        self.category_index
            .entry(category_key)
            .or_default()
            .push(id.clone());

        // Update tag index
        for tag in &entry.tags {
            self.tag_index
                .entry(tag.clone())
                .or_default()
                .push(id.clone());
        }

        self.memories.insert(id.clone(), entry);
        self.stats.total_stored += 1;

        id
    }

    /// Retrieve a memory by ID
    pub fn retrieve(&mut self, id: &str) -> Option<&MemoryEntry> {
        if let Some(entry) = self.memories.get_mut(id) {
            entry.record_access(self.current_time);
            self.stats.record_retrieval(true, 1.0);
            // Return immutable reference
            self.memories.get(id)
        } else {
            self.stats.record_retrieval(false, 0.0);
            None
        }
    }

    /// Retrieve a memory by ID (immutable, no access recording)
    pub fn peek(&self, id: &str) -> Option<&MemoryEntry> {
        self.memories.get(id)
    }

    /// Search memories by similarity to a query embedding (P-time: O(n))
    pub fn search_similar(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<(&MemoryEntry, f32)> {
        let mut scored: Vec<_> = self.memories.values()
            .filter_map(|entry| {
                if entry.embedding.is_empty() {
                    return None;
                }

                let min_len = query_embedding.len().min(entry.embedding.len());
                let dot: f32 = query_embedding.iter()
                    .zip(entry.embedding.iter())
                    .take(min_len)
                    .map(|(a, b)| a * b)
                    .sum();

                let mag_q: f32 = query_embedding.iter().take(min_len).map(|x| x * x).sum::<f32>().sqrt();
                let mag_e: f32 = entry.embedding.iter().take(min_len).map(|x| x * x).sum::<f32>().sqrt();

                if mag_q == 0.0 || mag_e == 0.0 {
                    None
                } else {
                    let sim = dot / (mag_q * mag_e);
                    Some((entry, sim))
                }
            })
            .collect();

        // Sort by similarity descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K
        let results: Vec<_> = scored.into_iter().take(top_k).collect();

        // Record stats
        if let Some((_, sim)) = results.first() {
            self.stats.record_retrieval(true, *sim);
        } else {
            self.stats.record_retrieval(false, 0.0);
        }

        results
    }

    /// Search by category
    pub fn search_by_category(&self, category: &MemoryCategory) -> Vec<&MemoryEntry> {
        let key = format!("{:?}", category);
        self.category_index
            .get(&key)
            .map(|ids| ids.iter().filter_map(|id| self.memories.get(id)).collect())
            .unwrap_or_default()
    }

    /// Search by tag
    pub fn search_by_tag(&self, tag: &str) -> Vec<&MemoryEntry> {
        self.tag_index
            .get(tag)
            .map(|ids| ids.iter().filter_map(|id| self.memories.get(id)).collect())
            .unwrap_or_default()
    }

    /// Consolidate similar memories (P-time: O(n^2) but bounded by config)
    pub fn consolidate(&mut self) -> ConsolidationResult {
        let mut result = ConsolidationResult::empty();

        // Get all memory IDs sorted by relevance
        let mut sorted_ids: Vec<_> = self.memories.keys().cloned().collect();
        sorted_ids.sort_by(|a, b| {
            let rel_a = self.memories.get(a).map(|m| m.relevance(self.current_time, self.config.decay_rate)).unwrap_or(0.0);
            let rel_b = self.memories.get(b).map(|m| m.relevance(self.current_time, self.config.decay_rate)).unwrap_or(0.0);
            rel_b.partial_cmp(&rel_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find similar pairs and mark for consolidation
        let mut to_merge: Vec<(String, String)> = Vec::new();
        let max_check = sorted_ids.len().min(500);

        for (i, id_a) in sorted_ids.iter().enumerate().take(max_check) {
            let mem_a = match self.memories.get(id_a) {
                Some(m) => m,
                None => continue,
            };

            for id_b in sorted_ids.iter().skip(i + 1).take(max_check - i - 1) {
                let mem_b = match self.memories.get(id_b) {
                    Some(m) => m,
                    None => continue,
                };

                let sim = mem_a.similarity(mem_b);
                if sim >= self.config.similarity_threshold {
                    to_merge.push((id_a.clone(), id_b.clone()));
                    result.new_associations += 1;
                }
            }
        }

        // Perform merges (keep the more important one, strengthen it)
        for (id_a, id_b) in to_merge {
            if let (Some(mem_a), Some(mem_b)) = (self.memories.get(&id_a), self.memories.get(&id_b)) {
                let keep_a = mem_a.importance >= mem_b.importance;
                let (keep_id, remove_id) = if keep_a { (id_a.clone(), id_b.clone()) } else { (id_b.clone(), id_a.clone()) };

                // Strengthen the kept memory
                if let Some(mem) = self.memories.get_mut(&keep_id) {
                    mem.importance = (mem.importance + 0.1).min(1.0);
                    mem.access_count += 1;
                    result.strengthened.push(keep_id.clone());
                }

                // Remove the duplicate
                self.memories.remove(&remove_id);
                result.consolidated.push(remove_id);
            }
        }

        self.stats.consolidations += 1;
        result.score = if result.total_operations() > 0 { 1.0 } else { 0.0 };
        result
    }

    /// Prune old/unimportant memories to stay within limits (P-time: O(n log n))
    pub fn prune(&mut self) -> Vec<String> {
        let mut pruned = Vec::new();

        if self.memories.len() <= self.config.max_memories {
            return pruned;
        }

        // Calculate scores for all memories
        let mut scored: Vec<_> = self.memories.iter()
            .map(|(id, mem)| {
                let score = mem.relevance(self.current_time, self.config.decay_rate);
                (id.clone(), score)
            })
            .collect();

        // Sort by score ascending (lowest first)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remove lowest scoring until under limit
        let to_remove = self.memories.len() - self.config.max_memories;
        for (id, score) in scored.into_iter().take(to_remove) {
            // Keep recent buffer
            if let Some(mem) = self.memories.get(&id) {
                if mem.access_count > 0 && score >= self.config.min_importance {
                    continue;
                }
            }

            self.memories.remove(&id);
            pruned.push(id);
            self.stats.memories_pruned += 1;
        }

        pruned
    }

    /// Get top K most relevant memories
    pub fn top_relevant(&self, k: usize) -> Vec<&MemoryEntry> {
        let mut scored: Vec<_> = self.memories.values()
            .map(|m| (m, m.relevance(self.current_time, self.config.decay_rate)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(k).map(|(m, _)| m).collect()
    }

    /// Get memory count
    pub fn count(&self) -> usize {
        self.memories.len()
    }

    /// Check if memory exists
    pub fn contains(&self, id: &str) -> bool {
        self.memories.contains_key(id)
    }

    /// Remove a specific memory
    pub fn forget(&mut self, id: &str) -> bool {
        self.memories.remove(id).is_some()
    }

    /// Clear all memories
    pub fn clear(&mut self) {
        self.memories.clear();
        self.category_index.clear();
        self.tag_index.clear();
        self.stats = MemoryStats::default();
    }

    /// Get all memory IDs
    pub fn all_ids(&self) -> Vec<&str> {
        self.memories.keys().map(|s| s.as_str()).collect()
    }

    /// Export memories for persistence (serializable format)
    pub fn export(&self) -> Vec<MemoryEntry> {
        self.memories.values().cloned().collect()
    }

    /// Import memories from external source
    pub fn import(&mut self, entries: Vec<MemoryEntry>) {
        for entry in entries {
            self.store(entry);
        }
    }
}

impl Default for MemoryConsolidation {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory function for creating MemoryConsolidation
pub fn create_memory_consolidation() -> MemoryConsolidation {
    MemoryConsolidation::new()
}

// ============================================================================
// Safety-Aware Meta-Cognition
// ============================================================================

/// Meta-cognition with integrated Asimov Laws safety monitoring
///
/// This wrapper adds safety awareness to any MetaCognition implementation.
/// It cannot be disabled or bypassed.
pub struct SafetyAwareMetaCognition<M: MetaCognition> {
    /// Underlying meta-cognition implementation
    inner: M,
    /// Safety gate for continuous monitoring
    safety_gate: SafetyGate,
    /// Cumulative safety state
    state: std::sync::Mutex<CognitiveState>,
}

impl<M: MetaCognition> SafetyAwareMetaCognition<M> {
    /// Create a new safety-aware meta-cognition wrapper
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            safety_gate: SafetyGate::new(),
            state: std::sync::Mutex::new(CognitiveState::new()),
        }
    }

    /// Check if a cognitive action is safe
    ///
    /// This method validates that a proposed cognitive operation
    /// does not violate Asimov's Laws.
    pub fn validate_cognitive_action(&self, description: &str) -> SafetyCheck {
        let action = SafetyAction::new(
            ActionType::Decide,
            ActionTarget::Unknown,
            description,
        );
        let result = self.safety_gate.guard().validate(&action);

        // Record violations in state
        if let SafetyCheck::Blocked(_) = &result {
            if let Ok(mut state) = self.state.lock() {
                state.record_safety_violation();
            }
        }

        result
    }

    /// Get safety-aware cognitive state
    pub fn safety_aware_introspect(&self) -> CognitiveState {
        let mut state = self.inner.introspect();

        // Merge with safety state
        if let Ok(safety_state) = self.state.lock() {
            state.safety_violation_count = safety_state.safety_violation_count;
            state.safety_monitoring_active = true;
        }

        state
    }

    /// Get total safety violation count
    pub fn safety_violation_count(&self) -> usize {
        self.safety_gate.guard().violation_count()
    }

    /// Check if system is in safe state
    pub fn is_safe(&self) -> bool {
        self.safety_gate.guard().violation_count() == 0
    }

    /// Get the underlying meta-cognition implementation
    pub fn inner(&self) -> &M {
        &self.inner
    }
}

impl<M: MetaCognition + Debug> std::fmt::Debug for SafetyAwareMetaCognition<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafetyAwareMetaCognition")
            .field("inner", &self.inner)
            .field("safety_violation_count", &self.safety_violation_count())
            .finish()
    }
}

/// Factory function for creating safety-aware meta-cognition
pub fn create_safety_aware_metacognition() -> SafetyAwareMetaCognition<SimpleMetaCognition> {
    SafetyAwareMetaCognition::new(SimpleMetaCognition::new())
}

#[cfg(test)]
mod memory_tests {
    use super::*;

    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new("m1", MemoryCategory::Fact, "Test content")
            .with_importance(0.8)
            .with_embedding(vec![1.0, 0.0])
            .with_tags(vec!["test".to_string()])
            .with_timestamp(100);

        assert_eq!(entry.id, "m1");
        assert_eq!(entry.importance, 0.8);
        assert_eq!(entry.embedding.len(), 2);
        assert_eq!(entry.tags.len(), 1);
        assert_eq!(entry.created_at, 100);
    }

    #[test]
    fn test_memory_entry_record_access() {
        let mut entry = MemoryEntry::new("m1", MemoryCategory::Fact, "Content");
        entry.record_access(50);
        entry.record_access(100);

        assert_eq!(entry.access_count, 2);
        assert_eq!(entry.last_access, 100);
    }

    #[test]
    fn test_memory_recency_score() {
        let mut entry = MemoryEntry::new("m1", MemoryCategory::Fact, "Content");
        entry.last_access = 0;

        let score_recent = entry.recency_score(0, 0.01);
        let score_old = entry.recency_score(1000, 0.01);

        assert!(score_recent > score_old);
        assert_eq!(score_recent, 1.0);
    }

    #[test]
    fn test_memory_similarity() {
        let entry1 = MemoryEntry::new("m1", MemoryCategory::Fact, "A")
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let entry2 = MemoryEntry::new("m2", MemoryCategory::Fact, "B")
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let entry3 = MemoryEntry::new("m3", MemoryCategory::Fact, "C")
            .with_embedding(vec![0.0, 1.0, 0.0]);

        assert_eq!(entry1.similarity(&entry2), 1.0);
        assert_eq!(entry1.similarity(&entry3), 0.0);
    }

    #[test]
    fn test_memory_category_default() {
        let cat = MemoryCategory::default();
        assert_eq!(cat, MemoryCategory::Fact);
    }

    #[test]
    fn test_consolidation_result() {
        let mut result = ConsolidationResult::empty();
        result.consolidated.push("m1".to_string());
        result.strengthened.push("m2".to_string());

        assert_eq!(result.total_operations(), 2);
    }

    #[test]
    fn test_memory_consolidation_creation() {
        let mem = MemoryConsolidation::new();
        assert_eq!(mem.count(), 0);
    }

    #[test]
    fn test_store_memory() {
        let mut mem = MemoryConsolidation::new();
        let entry = MemoryEntry::new("m1", MemoryCategory::Fact, "Test");

        let id = mem.store(entry);

        assert_eq!(id, "m1");
        assert_eq!(mem.count(), 1);
    }

    #[test]
    fn test_store_auto_id() {
        let mut mem = MemoryConsolidation::new();
        let entry = MemoryEntry::new("", MemoryCategory::Fact, "Test");

        let id = mem.store(entry);

        assert!(id.starts_with("mem_"));
        assert_eq!(mem.count(), 1);
    }

    #[test]
    fn test_retrieve_memory() {
        let mut mem = MemoryConsolidation::new();
        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "Content"));

        let entry = mem.retrieve("m1");

        assert!(entry.is_some());
        assert_eq!(entry.unwrap().content, "Content");
    }

    #[test]
    fn test_retrieve_not_found() {
        let mut mem = MemoryConsolidation::new();

        let entry = mem.retrieve("nonexistent");

        assert!(entry.is_none());
    }

    #[test]
    fn test_peek_memory() {
        let mut mem = MemoryConsolidation::new();
        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "Content"));

        let entry = mem.peek("m1");

        assert!(entry.is_some());
        assert_eq!(entry.unwrap().access_count, 0); // Peek doesn't increment
    }

    #[test]
    fn test_search_similar() {
        let mut mem = MemoryConsolidation::new();

        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "A")
            .with_embedding(vec![1.0, 0.0, 0.0]));
        mem.store(MemoryEntry::new("m2", MemoryCategory::Fact, "B")
            .with_embedding(vec![0.9, 0.1, 0.0]));
        mem.store(MemoryEntry::new("m3", MemoryCategory::Fact, "C")
            .with_embedding(vec![0.0, 1.0, 0.0]));

        let results = mem.search_similar(&[1.0, 0.0, 0.0], 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.id, "m1");
    }

    #[test]
    fn test_search_by_category() {
        let mut mem = MemoryConsolidation::new();

        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "A"));
        mem.store(MemoryEntry::new("m2", MemoryCategory::Procedure, "B"));
        mem.store(MemoryEntry::new("m3", MemoryCategory::Fact, "C"));

        let facts = mem.search_by_category(&MemoryCategory::Fact);

        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_search_by_tag() {
        let mut mem = MemoryConsolidation::new();

        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "A")
            .with_tags(vec!["important".to_string()]));
        mem.store(MemoryEntry::new("m2", MemoryCategory::Fact, "B")
            .with_tags(vec!["important".to_string(), "urgent".to_string()]));

        let important = mem.search_by_tag("important");

        assert_eq!(important.len(), 2);
    }

    #[test]
    fn test_consolidate() {
        let mut mem = MemoryConsolidation::new();

        // Create two similar memories
        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "A")
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_importance(0.8));
        mem.store(MemoryEntry::new("m2", MemoryCategory::Fact, "A similar")
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_importance(0.6));

        let result = mem.consolidate();

        assert!(!result.consolidated.is_empty() || !result.strengthened.is_empty());
    }

    #[test]
    fn test_prune() {
        let mut mem = MemoryConsolidation::with_config(ConsolidationConfig {
            max_memories: 2,
            ..Default::default()
        });

        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "A").with_importance(0.9));
        mem.store(MemoryEntry::new("m2", MemoryCategory::Fact, "B").with_importance(0.5));
        mem.store(MemoryEntry::new("m3", MemoryCategory::Fact, "C").with_importance(0.1));

        let pruned = mem.prune();

        assert!(!pruned.is_empty());
        assert!(mem.count() <= 2);
    }

    #[test]
    fn test_top_relevant() {
        let mut mem = MemoryConsolidation::new();

        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "A").with_importance(0.9));
        mem.store(MemoryEntry::new("m2", MemoryCategory::Fact, "B").with_importance(0.3));
        mem.store(MemoryEntry::new("m3", MemoryCategory::Fact, "C").with_importance(0.7));

        let top = mem.top_relevant(2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].id, "m1");
    }

    #[test]
    fn test_forget() {
        let mut mem = MemoryConsolidation::new();
        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "A"));

        assert!(mem.forget("m1"));
        assert!(!mem.contains("m1"));
    }

    #[test]
    fn test_clear() {
        let mut mem = MemoryConsolidation::new();
        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "A"));
        mem.store(MemoryEntry::new("m2", MemoryCategory::Fact, "B"));

        mem.clear();

        assert_eq!(mem.count(), 0);
    }

    #[test]
    fn test_export_import() {
        let mut mem1 = MemoryConsolidation::new();
        mem1.store(MemoryEntry::new("m1", MemoryCategory::Fact, "A"));
        mem1.store(MemoryEntry::new("m2", MemoryCategory::Fact, "B"));

        let exported = mem1.export();

        let mut mem2 = MemoryConsolidation::new();
        mem2.import(exported);

        assert_eq!(mem2.count(), 2);
    }

    #[test]
    fn test_tick() {
        let mut mem = MemoryConsolidation::new();
        mem.tick(100);

        assert_eq!(mem.current_time, 100);

        mem.tick(50);
        assert_eq!(mem.current_time, 150);
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::default();

        stats.record_retrieval(true, 0.9);
        stats.record_retrieval(true, 0.8);
        stats.record_retrieval(false, 0.0);

        assert_eq!(stats.total_retrievals, 3);
        assert_eq!(stats.successful_retrievals, 2);
        assert!((stats.retrieval_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_all_ids() {
        let mut mem = MemoryConsolidation::new();
        mem.store(MemoryEntry::new("m1", MemoryCategory::Fact, "A"));
        mem.store(MemoryEntry::new("m2", MemoryCategory::Fact, "B"));

        let ids = mem.all_ids();

        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_create_memory_consolidation_factory() {
        let mem = create_memory_consolidation();
        assert_eq!(mem.count(), 0);
    }

    #[test]
    fn test_consolidation_config_default() {
        let config = ConsolidationConfig::default();
        assert_eq!(config.max_memories, 10000);
    }
}
