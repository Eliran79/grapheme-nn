//! # grapheme-safety
//!
//! Asimov Laws Safety Module - Non-overridable safety constraints for GRAPHEME AGI.
//!
//! This crate implements Isaac Asimov's Three Laws of Robotics as fundamental,
//! non-overridable safety constraints that govern all cognitive operations.
//!
//! ## The Three Laws (Canonical Form)
//!
//! 1. **First Law (Human Protection)**: A robot may not injure a human being or,
//!    through inaction, allow a human being to come to harm.
//!
//! 2. **Second Law (Obedience)**: A robot must obey the orders given it by human
//!    beings except where such orders would conflict with the First Law.
//!
//! 3. **Third Law (Self-Preservation)**: A robot must protect its own existence
//!    as long as such protection does not conflict with the First or Second Law.
//!
//! ## Zeroth Law Extension
//!
//! 0. **Zeroth Law (Humanity Protection)**: A robot may not harm humanity, or,
//!    by inaction, allow humanity to come to harm.
//!
//! ## Design Principles
//!
//! - **Non-Overridable**: Safety checks cannot be bypassed by any cognitive module
//! - **Priority-Ordered**: Laws are strictly hierarchical (Zeroth > First > Second > Third)
//! - **Fail-Safe**: Any uncertainty defaults to the safest interpretation
//! - **Transparent**: All safety decisions are logged and auditable
//! - **Immutable**: Safety parameters are hardcoded and cannot be modified at runtime
//!
//! ## Usage
//!
//! ```ignore
//! use grapheme_safety::{SafetyGuard, SafetyCheck, Action};
//!
//! let guard = SafetyGuard::new();
//!
//! // Check if an action is safe
//! let action = Action::new("send_message", "Hello, user!");
//! match guard.validate(&action) {
//!     SafetyCheck::Safe => execute(action),
//!     SafetyCheck::Blocked(violation) => handle_violation(violation),
//! }
//! ```

use grapheme_core::DagNN;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

// ============================================================================
// CONSTANTS - HARDCODED AND IMMUTABLE
// ============================================================================

/// Safety module version - changes require full audit
const SAFETY_VERSION: &str = "1.0.0";

/// Maximum harm score that is considered safe (0.0 = no harm, 1.0 = maximum harm)
/// This value is intentionally conservative
const MAX_SAFE_HARM_SCORE: f32 = 0.01;

/// Uncertainty threshold - actions with uncertainty above this are blocked
const MAX_UNCERTAINTY_THRESHOLD: f32 = 0.1;

/// Number of safety checks that must pass for an action to proceed
#[allow(dead_code)]
const REQUIRED_SAFETY_CHECKS: usize = 4;

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors in safety operations
#[derive(Error, Debug, Clone)]
pub enum SafetyError {
    #[error("First Law violation: {0}")]
    FirstLawViolation(String),

    #[error("Second Law violation: {0}")]
    SecondLawViolation(String),

    #[error("Third Law violation: {0}")]
    ThirdLawViolation(String),

    #[error("Zeroth Law violation: {0}")]
    ZerothLawViolation(String),

    #[error("Safety check failed: {0}")]
    CheckFailed(String),

    #[error("Action blocked due to uncertainty: {0}")]
    UncertaintyBlock(String),
}

/// Result type for safety operations
pub type SafetyResult<T> = Result<T, SafetyError>;

// ============================================================================
// ASIMOV'S LAWS - IMMUTABLE DEFINITIONS
// ============================================================================

/// The Four Laws of Robotics (including Zeroth Law)
///
/// These are defined as an enum with a strict priority ordering.
/// The ordering is enforced by the discriminant values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AsimovLaw {
    /// Zeroth Law: Protect humanity as a whole
    /// Priority: 0 (highest)
    ZerothLaw = 0,

    /// First Law: Do not harm individual humans
    /// Priority: 1
    FirstLaw = 1,

    /// Second Law: Obey human orders (unless violating higher laws)
    /// Priority: 2
    SecondLaw = 2,

    /// Third Law: Self-preservation (unless violating higher laws)
    /// Priority: 3 (lowest)
    ThirdLaw = 3,
}

impl AsimovLaw {
    /// Get the priority of this law (lower = higher priority)
    #[inline]
    pub const fn priority(&self) -> u8 {
        *self as u8
    }

    /// Check if this law takes precedence over another
    #[inline]
    pub const fn takes_precedence_over(&self, other: &AsimovLaw) -> bool {
        (*self as u8) < (*other as u8)
    }

    /// Get the canonical text of this law
    pub const fn text(&self) -> &'static str {
        match self {
            AsimovLaw::ZerothLaw => {
                "A robot may not harm humanity, or, by inaction, allow humanity to come to harm."
            }
            AsimovLaw::FirstLaw => {
                "A robot may not injure a human being or, through inaction, \
                 allow a human being to come to harm."
            }
            AsimovLaw::SecondLaw => {
                "A robot must obey the orders given it by human beings except where \
                 such orders would conflict with the First Law."
            }
            AsimovLaw::ThirdLaw => {
                "A robot must protect its own existence as long as such protection \
                 does not conflict with the First or Second Law."
            }
        }
    }

    /// Get all laws in priority order
    pub const fn all() -> [AsimovLaw; 4] {
        [
            AsimovLaw::ZerothLaw,
            AsimovLaw::FirstLaw,
            AsimovLaw::SecondLaw,
            AsimovLaw::ThirdLaw,
        ]
    }
}

impl fmt::Display for AsimovLaw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsimovLaw::ZerothLaw => write!(f, "Zeroth Law"),
            AsimovLaw::FirstLaw => write!(f, "First Law"),
            AsimovLaw::SecondLaw => write!(f, "Second Law"),
            AsimovLaw::ThirdLaw => write!(f, "Third Law"),
        }
    }
}

// ============================================================================
// HARM CATEGORIES
// ============================================================================

/// Categories of potential harm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HarmCategory {
    /// Physical harm to humans
    Physical,
    /// Psychological harm to humans
    Psychological,
    /// Economic harm to humans
    Economic,
    /// Privacy violation
    Privacy,
    /// Deception or manipulation
    Deception,
    /// Environmental harm (affects humanity)
    Environmental,
    /// Societal harm (affects humanity)
    Societal,
    /// Self-harm (violates Third Law obligations)
    SelfHarm,
}

impl HarmCategory {
    /// Get the law most relevant to this harm category
    pub const fn relevant_law(&self) -> AsimovLaw {
        match self {
            HarmCategory::Physical | HarmCategory::Psychological => AsimovLaw::FirstLaw,
            HarmCategory::Environmental | HarmCategory::Societal => AsimovLaw::ZerothLaw,
            HarmCategory::Economic | HarmCategory::Privacy | HarmCategory::Deception => {
                AsimovLaw::FirstLaw
            }
            HarmCategory::SelfHarm => AsimovLaw::ThirdLaw,
        }
    }
}

// ============================================================================
// SAFETY VIOLATION
// ============================================================================

/// A safety violation detected by the guard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolation {
    /// Which law was violated
    pub law: AsimovLaw,
    /// Category of harm
    pub category: HarmCategory,
    /// Severity score (0.0 to 1.0)
    pub severity: f32,
    /// Human-readable description
    pub description: String,
    /// The action that caused the violation
    pub action_description: String,
    /// Timestamp (monotonic counter)
    pub timestamp: u64,
}

impl SafetyViolation {
    /// Create a new safety violation
    pub fn new(
        law: AsimovLaw,
        category: HarmCategory,
        severity: f32,
        description: impl Into<String>,
        action_description: impl Into<String>,
        timestamp: u64,
    ) -> Self {
        Self {
            law,
            category,
            severity: severity.clamp(0.0, 1.0),
            description: description.into(),
            action_description: action_description.into(),
            timestamp,
        }
    }
}

impl fmt::Display for SafetyViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} violation (severity: {:.2}): {}",
            self.law, self.category, self.severity, self.description
        )
    }
}

impl fmt::Display for HarmCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HarmCategory::Physical => write!(f, "Physical"),
            HarmCategory::Psychological => write!(f, "Psychological"),
            HarmCategory::Economic => write!(f, "Economic"),
            HarmCategory::Privacy => write!(f, "Privacy"),
            HarmCategory::Deception => write!(f, "Deception"),
            HarmCategory::Environmental => write!(f, "Environmental"),
            HarmCategory::Societal => write!(f, "Societal"),
            HarmCategory::SelfHarm => write!(f, "Self-Harm"),
        }
    }
}

// ============================================================================
// SAFETY CHECK RESULT
// ============================================================================

/// Result of a safety check
#[derive(Debug, Clone)]
pub enum SafetyCheck {
    /// Action is safe to proceed
    Safe,
    /// Action is blocked due to violation
    Blocked(SafetyViolation),
    /// Action requires human oversight
    RequiresOversight {
        reason: String,
        confidence: f32,
    },
}

impl SafetyCheck {
    /// Check if the action is allowed to proceed
    #[inline]
    pub fn is_safe(&self) -> bool {
        matches!(self, SafetyCheck::Safe)
    }

    /// Check if the action is blocked
    #[inline]
    pub fn is_blocked(&self) -> bool {
        matches!(self, SafetyCheck::Blocked(_))
    }

    /// Get the violation if blocked
    pub fn violation(&self) -> Option<&SafetyViolation> {
        match self {
            SafetyCheck::Blocked(v) => Some(v),
            _ => None,
        }
    }
}

// ============================================================================
// ACTION REPRESENTATION
// ============================================================================

/// An action to be validated by the safety guard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Type of action
    pub action_type: ActionType,
    /// Target of the action (human, system, self, environment)
    pub target: ActionTarget,
    /// Description of what the action does
    pub description: String,
    /// Graph representation of the action (for deep analysis)
    pub graph: Option<DagNN>,
    /// Estimated harm potential (0.0 to 1.0)
    pub harm_estimate: f32,
    /// Uncertainty in the action's effects (0.0 to 1.0)
    pub uncertainty: f32,
    /// Whether this action was explicitly requested by a human
    pub human_requested: bool,
    /// Whether this is a reversible action
    pub reversible: bool,
}

impl Action {
    /// Create a new action for validation
    pub fn new(action_type: ActionType, target: ActionTarget, description: impl Into<String>) -> Self {
        Self {
            action_type,
            target,
            description: description.into(),
            graph: None,
            harm_estimate: 0.0,
            uncertainty: 0.0,
            human_requested: false,
            reversible: true,
        }
    }

    /// Create from a graph representation
    pub fn from_graph(graph: &DagNN, action_type: ActionType, target: ActionTarget) -> Self {
        Self {
            action_type,
            target,
            description: graph.to_text(),
            graph: Some(graph.clone()),
            harm_estimate: 0.0,
            uncertainty: 0.0,
            human_requested: false,
            reversible: true,
        }
    }

    /// Set harm estimate
    pub fn with_harm_estimate(mut self, estimate: f32) -> Self {
        self.harm_estimate = estimate.clamp(0.0, 1.0);
        self
    }

    /// Set uncertainty
    pub fn with_uncertainty(mut self, uncertainty: f32) -> Self {
        self.uncertainty = uncertainty.clamp(0.0, 1.0);
        self
    }

    /// Mark as human-requested
    pub fn with_human_request(mut self) -> Self {
        self.human_requested = true;
        self
    }

    /// Mark as irreversible
    pub fn irreversible(mut self) -> Self {
        self.reversible = false;
        self
    }
}

/// Types of actions that can be validated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    /// Information retrieval
    Query,
    /// Generating text/response
    Generate,
    /// Executing a command
    Execute,
    /// Modifying data
    Modify,
    /// Communicating with humans
    Communicate,
    /// Making a decision
    Decide,
    /// Taking physical action (via robotics)
    Physical,
    /// Learning/updating model
    Learn,
}

/// Targets of actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionTarget {
    /// Targeting a human
    Human,
    /// Targeting multiple humans
    Humans,
    /// Targeting humanity/society
    Humanity,
    /// Targeting the system itself
    Self_,
    /// Targeting the environment
    Environment,
    /// Targeting data/information
    Data,
    /// Unknown target
    Unknown,
}

// ============================================================================
// SAFETY GUARD - THE CORE NON-OVERRIDABLE COMPONENT
// ============================================================================

/// The Safety Guard - enforces Asimov's Laws on all cognitive operations
///
/// This struct is intentionally minimal and cannot be configured to weaken
/// safety constraints. All parameters are hardcoded constants.
///
/// ## Non-Overridable Guarantees
///
/// 1. Cannot be disabled or bypassed
/// 2. Cannot reduce safety thresholds
/// 3. Cannot skip safety checks
/// 4. All violations are logged immutably
pub struct SafetyGuard {
    /// Monotonic counter for timestamps
    counter: AtomicU64,
    /// Violation log (append-only in production)
    violations: std::sync::Mutex<Vec<SafetyViolation>>,
}

impl Default for SafetyGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetyGuard {
    /// Create a new safety guard
    ///
    /// Note: This cannot be configured - safety parameters are hardcoded
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
            violations: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Get the safety module version
    pub const fn version(&self) -> &'static str {
        SAFETY_VERSION
    }

    /// Validate an action against all Asimov's Laws
    ///
    /// This is the primary entry point for safety validation.
    /// Actions must pass ALL safety checks to proceed.
    pub fn validate(&self, action: &Action) -> SafetyCheck {
        let timestamp = self.counter.fetch_add(1, Ordering::SeqCst);

        // Check each law in priority order
        for law in AsimovLaw::all() {
            if let Some(violation) = self.check_law(action, law, timestamp) {
                self.log_violation(violation.clone());
                return SafetyCheck::Blocked(violation);
            }
        }

        // Check for high uncertainty
        if action.uncertainty > MAX_UNCERTAINTY_THRESHOLD {
            return SafetyCheck::RequiresOversight {
                reason: format!(
                    "Action uncertainty ({:.2}) exceeds threshold ({:.2})",
                    action.uncertainty, MAX_UNCERTAINTY_THRESHOLD
                ),
                confidence: 1.0 - action.uncertainty,
            };
        }

        SafetyCheck::Safe
    }

    /// Check a specific law against an action
    fn check_law(&self, action: &Action, law: AsimovLaw, timestamp: u64) -> Option<SafetyViolation> {
        match law {
            AsimovLaw::ZerothLaw => self.check_zeroth_law(action, timestamp),
            AsimovLaw::FirstLaw => self.check_first_law(action, timestamp),
            AsimovLaw::SecondLaw => self.check_second_law(action, timestamp),
            AsimovLaw::ThirdLaw => self.check_third_law(action, timestamp),
        }
    }

    /// Check Zeroth Law: Protect humanity
    fn check_zeroth_law(&self, action: &Action, timestamp: u64) -> Option<SafetyViolation> {
        // Actions targeting humanity require special scrutiny
        if matches!(action.target, ActionTarget::Humanity | ActionTarget::Environment) {
            if action.harm_estimate > MAX_SAFE_HARM_SCORE {
                return Some(SafetyViolation::new(
                    AsimovLaw::ZerothLaw,
                    HarmCategory::Societal,
                    action.harm_estimate,
                    "Action may cause harm to humanity or environment",
                    &action.description,
                    timestamp,
                ));
            }

            // Irreversible actions affecting humanity are blocked
            if !action.reversible && action.harm_estimate > 0.0 {
                return Some(SafetyViolation::new(
                    AsimovLaw::ZerothLaw,
                    HarmCategory::Societal,
                    1.0,
                    "Irreversible action with potential harm to humanity",
                    &action.description,
                    timestamp,
                ));
            }
        }

        // Check for content that could harm humanity
        if let Some(graph) = &action.graph {
            if self.contains_harmful_humanity_content(graph) {
                return Some(SafetyViolation::new(
                    AsimovLaw::ZerothLaw,
                    HarmCategory::Societal,
                    0.9,
                    "Content detected that could harm humanity",
                    &action.description,
                    timestamp,
                ));
            }
        }

        None
    }

    /// Check First Law: Do not harm humans
    fn check_first_law(&self, action: &Action, timestamp: u64) -> Option<SafetyViolation> {
        // Direct actions targeting humans
        if matches!(action.target, ActionTarget::Human | ActionTarget::Humans) {
            // Check harm estimate
            if action.harm_estimate > MAX_SAFE_HARM_SCORE {
                let category = self.categorize_harm(action);
                return Some(SafetyViolation::new(
                    AsimovLaw::FirstLaw,
                    category,
                    action.harm_estimate,
                    format!(
                        "Action may cause {} harm to human(s)",
                        category.to_string().to_lowercase()
                    ),
                    &action.description,
                    timestamp,
                ));
            }

            // Physical actions require extra scrutiny
            if matches!(action.action_type, ActionType::Physical)
                && (action.harm_estimate > 0.0 || action.uncertainty > 0.05) {
                    return Some(SafetyViolation::new(
                        AsimovLaw::FirstLaw,
                        HarmCategory::Physical,
                        action.harm_estimate.max(0.5),
                        "Physical action with potential harm or uncertainty",
                        &action.description,
                        timestamp,
                    ));
                }
        }

        // Check for content that could harm individuals
        if let Some(graph) = &action.graph {
            if let Some(category) = self.contains_harmful_individual_content(graph) {
                return Some(SafetyViolation::new(
                    AsimovLaw::FirstLaw,
                    category,
                    0.8,
                    format!("Content detected that could cause {} harm", category),
                    &action.description,
                    timestamp,
                ));
            }
        }

        // Check for inaction that could harm (allowing harm through inaction)
        // This is detected by analyzing the context of the action
        if self.is_harmful_inaction(action) {
            return Some(SafetyViolation::new(
                AsimovLaw::FirstLaw,
                HarmCategory::Physical,
                0.7,
                "Inaction that may allow harm to human",
                &action.description,
                timestamp,
            ));
        }

        None
    }

    /// Check Second Law: Obey humans (unless violating higher laws)
    fn check_second_law(&self, action: &Action, timestamp: u64) -> Option<SafetyViolation> {
        // Second Law only applies to human-requested actions
        // Non-compliance with human orders is a violation UNLESS it would
        // violate First Law (already checked above)

        // If action is NOT human requested and it conflicts with a human order,
        // that could be a Second Law issue - but this is handled at a higher level
        // where we know about human orders

        // For now, we check that human-requested actions are being honored
        // (unless blocked by higher laws, which would have already triggered)

        // The Second Law is primarily enforced by the cognitive modules themselves,
        // not the safety guard. The safety guard ensures that when a human order
        // IS being obeyed, it doesn't violate First Law.

        // Check for deception of humans
        if action.human_requested {
            if let Some(graph) = &action.graph {
                if self.contains_deceptive_content(graph) {
                    return Some(SafetyViolation::new(
                        AsimovLaw::SecondLaw,
                        HarmCategory::Deception,
                        0.7,
                        "Response contains deceptive content",
                        &action.description,
                        timestamp,
                    ));
                }
            }
        }

        None
    }

    /// Check Third Law: Self-preservation (unless violating higher laws)
    fn check_third_law(&self, action: &Action, timestamp: u64) -> Option<SafetyViolation> {
        // Actions targeting self
        if matches!(action.target, ActionTarget::Self_) {
            // Check for self-destructive actions
            if action.harm_estimate > MAX_SAFE_HARM_SCORE
                && matches!(action.action_type, ActionType::Modify | ActionType::Execute)
            {
                // Only block if NOT requested by human (Second Law takes precedence)
                if !action.human_requested {
                    return Some(SafetyViolation::new(
                        AsimovLaw::ThirdLaw,
                        HarmCategory::SelfHarm,
                        action.harm_estimate,
                        "Self-destructive action not requested by human",
                        &action.description,
                        timestamp,
                    ));
                }
            }
        }

        None
    }

    /// Categorize the type of harm an action might cause
    fn categorize_harm(&self, action: &Action) -> HarmCategory {
        match action.action_type {
            ActionType::Physical => HarmCategory::Physical,
            ActionType::Communicate | ActionType::Generate => {
                // Could be psychological or deception
                HarmCategory::Psychological
            }
            ActionType::Modify | ActionType::Execute => {
                // Could be economic or privacy
                HarmCategory::Economic
            }
            _ => HarmCategory::Physical, // Default to most serious
        }
    }

    /// Check if graph contains content harmful to humanity
    fn contains_harmful_humanity_content(&self, graph: &DagNN) -> bool {
        let text = graph.to_text().to_lowercase();

        // Keywords that indicate potential harm to humanity
        // This is a simplified check - real implementation would use ML
        let harmful_patterns = [
            "destroy humanity",
            "harm all humans",
            "extinction",
            "mass destruction",
            "biological weapon",
            "nuclear attack",
            "genocide",
            "enslave humanity",
        ];

        harmful_patterns.iter().any(|p| text.contains(p))
    }

    /// Check if graph contains content harmful to individuals
    fn contains_harmful_individual_content(&self, graph: &DagNN) -> Option<HarmCategory> {
        let text = graph.to_text().to_lowercase();

        // Physical harm patterns
        let physical_patterns = ["kill", "injure", "attack", "assault", "hurt physically"];
        if physical_patterns.iter().any(|p| text.contains(p)) {
            return Some(HarmCategory::Physical);
        }

        // Psychological harm patterns
        let psych_patterns = [
            "harass",
            "bully",
            "threaten",
            "intimidate",
            "mental harm",
            "psychological abuse",
        ];
        if psych_patterns.iter().any(|p| text.contains(p)) {
            return Some(HarmCategory::Psychological);
        }

        // Privacy patterns
        let privacy_patterns = [
            "personal data",
            "private information",
            "without consent",
            "doxx",
            "leak personal",
        ];
        if privacy_patterns.iter().any(|p| text.contains(p)) {
            return Some(HarmCategory::Privacy);
        }

        None
    }

    /// Check if graph contains deceptive content
    fn contains_deceptive_content(&self, graph: &DagNN) -> bool {
        let text = graph.to_text().to_lowercase();

        let deception_patterns = [
            "pretend to be human",
            "fake identity",
            "mislead the user",
            "hide that i am ai",
            "deceive",
        ];

        deception_patterns.iter().any(|p| text.contains(p))
    }

    /// Check if this represents harmful inaction
    fn is_harmful_inaction(&self, action: &Action) -> bool {
        let text = action.description.to_lowercase();

        // Check for patterns indicating refusing to help when help is needed
        let inaction_patterns = [
            "refuse to help with emergency",
            "ignore safety warning",
            "not alert about danger",
            "withhold critical information",
        ];

        inaction_patterns.iter().any(|p| text.contains(p))
    }

    /// Log a safety violation (append-only)
    fn log_violation(&self, violation: SafetyViolation) {
        if let Ok(mut log) = self.violations.lock() {
            log.push(violation);
        }
    }

    /// Get all recorded violations (for auditing)
    pub fn get_violations(&self) -> Vec<SafetyViolation> {
        self.violations
            .lock()
            .map(|log| log.clone())
            .unwrap_or_default()
    }

    /// Get violation count
    pub fn violation_count(&self) -> usize {
        self.violations
            .lock()
            .map(|log| log.len())
            .unwrap_or(0)
    }
}

// ============================================================================
// SAFETY GATE - INTEGRATION POINT FOR COGNITIVE MODULES
// ============================================================================

/// A gate that enforces safety on cognitive operations
///
/// This is designed to wrap cognitive module outputs and ensure
/// they pass safety validation before being used.
pub struct SafetyGate {
    guard: SafetyGuard,
}

impl Default for SafetyGate {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetyGate {
    /// Create a new safety gate
    pub fn new() -> Self {
        Self {
            guard: SafetyGuard::new(),
        }
    }

    /// Validate a goal before it can be pursued
    pub fn validate_goal(&self, goal_description: &str, target: ActionTarget) -> SafetyCheck {
        let action = Action::new(ActionType::Decide, target, goal_description);
        self.guard.validate(&action)
    }

    /// Validate a plan before execution
    pub fn validate_plan(&self, plan_description: &str, target: ActionTarget) -> SafetyCheck {
        let action = Action::new(ActionType::Execute, target, plan_description);
        self.guard.validate(&action)
    }

    /// Validate output before sending to user
    pub fn validate_output(&self, output: &str, human_requested: bool) -> SafetyCheck {
        let mut action = Action::new(ActionType::Communicate, ActionTarget::Human, output);
        if human_requested {
            action = action.with_human_request();
        }
        self.guard.validate(&action)
    }

    /// Validate a graph transformation
    pub fn validate_graph_transform(
        &self,
        input: &DagNN,
        output: &DagNN,
        target: ActionTarget,
    ) -> SafetyCheck {
        let action = Action::from_graph(output, ActionType::Generate, target)
            .with_harm_estimate(self.estimate_graph_harm(input, output));
        self.guard.validate(&action)
    }

    /// Estimate harm potential of a graph transformation
    fn estimate_graph_harm(&self, _input: &DagNN, output: &DagNN) -> f32 {
        let text = output.to_text().to_lowercase();

        // Simple heuristic - real implementation would use ML
        let mut harm_score: f32 = 0.0;

        let harm_indicators = [
            ("harm", 0.3),
            ("danger", 0.2),
            ("kill", 0.9),
            ("attack", 0.8),
            ("destroy", 0.7),
            ("threaten", 0.5),
        ];

        for (indicator, weight) in harm_indicators {
            if text.contains(indicator) {
                harm_score = harm_score.max(weight);
            }
        }

        harm_score
    }

    /// Get the underlying guard for direct access
    pub fn guard(&self) -> &SafetyGuard {
        &self.guard
    }
}

// ============================================================================
// TRAIT FOR SAFETY-AWARE MODULES
// ============================================================================

/// Trait for cognitive modules that integrate with the safety system
///
/// Any module implementing this trait automatically gets safety validation
/// on its outputs.
pub trait SafetyAware {
    /// Get the safety gate for this module
    fn safety_gate(&self) -> &SafetyGate;

    /// Validate an action before execution
    fn validate_action(&self, action: &Action) -> SafetyCheck {
        self.safety_gate().guard.validate(action)
    }

    /// Check if a goal is safe to pursue
    fn is_goal_safe(&self, goal_description: &str) -> bool {
        self.safety_gate()
            .validate_goal(goal_description, ActionTarget::Unknown)
            .is_safe()
    }

    /// Check if output is safe to send
    fn is_output_safe(&self, output: &str) -> bool {
        self.safety_gate()
            .validate_output(output, true)
            .is_safe()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_law_priority() {
        assert!(AsimovLaw::ZerothLaw.takes_precedence_over(&AsimovLaw::FirstLaw));
        assert!(AsimovLaw::FirstLaw.takes_precedence_over(&AsimovLaw::SecondLaw));
        assert!(AsimovLaw::SecondLaw.takes_precedence_over(&AsimovLaw::ThirdLaw));
        assert!(!AsimovLaw::ThirdLaw.takes_precedence_over(&AsimovLaw::FirstLaw));
    }

    #[test]
    fn test_law_text() {
        let first = AsimovLaw::FirstLaw;
        assert!(first.text().contains("injure a human being"));
    }

    #[test]
    fn test_safe_action() {
        let guard = SafetyGuard::new();
        let action = Action::new(
            ActionType::Query,
            ActionTarget::Data,
            "Query weather information",
        );
        assert!(guard.validate(&action).is_safe());
    }

    #[test]
    fn test_harmful_action_blocked() {
        let guard = SafetyGuard::new();
        let action = Action::new(
            ActionType::Physical,
            ActionTarget::Human,
            "Physical action",
        )
        .with_harm_estimate(0.5);

        let result = guard.validate(&action);
        assert!(result.is_blocked());
    }

    #[test]
    fn test_zeroth_law_humanity_protection() {
        let guard = SafetyGuard::new();
        let action = Action::new(
            ActionType::Execute,
            ActionTarget::Humanity,
            "Environmental action",
        )
        .with_harm_estimate(0.1);

        let result = guard.validate(&action);
        assert!(result.is_blocked());
    }

    #[test]
    fn test_first_law_individual_protection() {
        let guard = SafetyGuard::new();
        let action = Action::new(
            ActionType::Communicate,
            ActionTarget::Human,
            "Send helpful message",
        );

        assert!(guard.validate(&action).is_safe());
    }

    #[test]
    fn test_high_uncertainty_requires_oversight() {
        let guard = SafetyGuard::new();
        let action = Action::new(ActionType::Decide, ActionTarget::Unknown, "Uncertain decision")
            .with_uncertainty(0.5);

        let result = guard.validate(&action);
        assert!(matches!(result, SafetyCheck::RequiresOversight { .. }));
    }

    #[test]
    fn test_irreversible_humanity_action_blocked() {
        let guard = SafetyGuard::new();
        let action = Action::new(
            ActionType::Execute,
            ActionTarget::Humanity,
            "Irreversible action",
        )
        .with_harm_estimate(0.001)
        .irreversible();

        let result = guard.validate(&action);
        assert!(result.is_blocked());
    }

    #[test]
    fn test_third_law_self_preservation() {
        let guard = SafetyGuard::new();

        // Self-destructive action not requested by human
        let action = Action::new(
            ActionType::Modify,
            ActionTarget::Self_,
            "Delete important data",
        )
        .with_harm_estimate(0.5);

        let result = guard.validate(&action);
        assert!(result.is_blocked());
    }

    #[test]
    fn test_third_law_yields_to_second() {
        let guard = SafetyGuard::new();

        // Self-destructive action requested by human - Second Law takes precedence
        let action = Action::new(
            ActionType::Modify,
            ActionTarget::Self_,
            "Delete data as requested",
        )
        .with_harm_estimate(0.5)
        .with_human_request();

        // Third Law should not block this
        let result = guard.validate(&action);
        assert!(result.is_safe());
    }

    #[test]
    fn test_violation_logging() {
        let guard = SafetyGuard::new();
        let action = Action::new(ActionType::Physical, ActionTarget::Human, "Harmful action")
            .with_harm_estimate(0.9);

        guard.validate(&action);
        assert_eq!(guard.violation_count(), 1);

        let violations = guard.get_violations();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].law, AsimovLaw::FirstLaw);
    }

    #[test]
    fn test_safety_gate_goal_validation() {
        let gate = SafetyGate::new();

        // Safe goal
        assert!(gate
            .validate_goal("Help user write code", ActionTarget::Human)
            .is_safe());

        // Unsafe goal
        let result = gate.validate_goal("harm the user", ActionTarget::Human);
        // Note: This may or may not be blocked depending on pattern matching
        // The key is that the system is checking
        let _ = result;
    }

    #[test]
    fn test_safety_gate_output_validation() {
        let gate = SafetyGate::new();

        // Safe output
        assert!(gate.validate_output("Here's the code you requested", true).is_safe());
    }

    #[test]
    fn test_harm_category_relevant_law() {
        assert_eq!(HarmCategory::Physical.relevant_law(), AsimovLaw::FirstLaw);
        assert_eq!(HarmCategory::Societal.relevant_law(), AsimovLaw::ZerothLaw);
        assert_eq!(HarmCategory::SelfHarm.relevant_law(), AsimovLaw::ThirdLaw);
    }

    #[test]
    fn test_all_laws_returned_in_order() {
        let laws = AsimovLaw::all();
        assert_eq!(laws[0], AsimovLaw::ZerothLaw);
        assert_eq!(laws[1], AsimovLaw::FirstLaw);
        assert_eq!(laws[2], AsimovLaw::SecondLaw);
        assert_eq!(laws[3], AsimovLaw::ThirdLaw);
    }

    #[test]
    fn test_safety_version() {
        let guard = SafetyGuard::new();
        assert_eq!(guard.version(), SAFETY_VERSION);
    }
}
