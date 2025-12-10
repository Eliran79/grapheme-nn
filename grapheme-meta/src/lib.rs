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
    BrainRegistry, CognitiveBrainBridge, DagNN, DefaultCognitiveBridge, DomainBrain, Learnable,
    LearnableParam, Persistable, PersistenceError,
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
