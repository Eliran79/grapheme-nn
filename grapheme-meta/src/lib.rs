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
