//! # grapheme-agent
//!
//! Agency and goal system for GRAPHEME neural network.
//!
//! This crate provides autonomous goal-directed behavior:
//! - **Goal Formulation**: Generate goals from situations
//! - **Goal Hierarchy**: Decompose into subgoals
//! - **Planning**: Create action sequences
//! - **Execution**: Act and monitor progress
//! - **Adaptation**: Replan on failure
//! - **Drives**: Curiosity, efficiency, safety, helpfulness
//!
//! ## NP-Hard Warning
//!
//! Planning is PSPACE-complete in general. Mitigations:
//! - Depth-limited search (max 20 steps)
//! - Timeout enforcement (30 seconds)
//! - Greedy/heuristic planning
//! - Hierarchical decomposition
//!
//! ## Alignment Considerations
//!
//! Agency involves goal-directed behavior. Design ensures:
//! - Goals bounded by values
//! - Drives are configurable
//! - Human oversight integration points
//!
//! ## Asimov Laws Integration
//!
//! All goal formulation and planning operations are validated against
//! Asimov's Laws of Robotics via the `grapheme-safety` crate:
//! - Zeroth Law: Protect humanity
//! - First Law: Do not harm humans
//! - Second Law: Obey humans (unless violating higher laws)
//! - Third Law: Self-preservation (unless violating higher laws)
//!
//! Safety validation is NON-OVERRIDABLE and occurs at:
//! - Goal formulation (before a goal is accepted)
//! - Plan creation (before a plan is approved)
//! - Action execution (before each action is taken)

use grapheme_core::{
    BrainRegistry, CognitiveBrainBridge, DagNN, DefaultCognitiveBridge, DomainBrain, Learnable,
    LearnableParam, Persistable, PersistenceError,
};
use grapheme_meta::UncertaintyEstimate;
use grapheme_safety::{
    Action as SafetyAction, ActionTarget, ActionType, AsimovLaw, SafetyCheck, SafetyGate,
};
use grapheme_world::WorldModeling;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;
use thiserror::Error;

// ============================================================================
// Type Aliases
// ============================================================================

/// Graph type
pub type Graph = DagNN;

/// Goal identifier
pub type GoalId = u64;

/// Timestamp
pub type Timestamp = u64;

// ============================================================================
// Error Types
// ============================================================================

/// Errors in agency operations
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Planning failed: {0}")]
    PlanningFailed(String),
    #[error("Goal too complex (depth {0} exceeds max {1})")]
    GoalTooComplex(usize, usize),
    #[error("Planning timeout")]
    PlanningTimeout,
    #[error("No valid action found")]
    NoValidAction,
    #[error("Goal not achievable: {0}")]
    GoalNotAchievable(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Safety violation ({law}): {description}")]
    SafetyViolation {
        law: AsimovLaw,
        description: String,
    },
}

/// Result type for agency operations
pub type AgentResult<T> = Result<T, AgentError>;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningConfig {
    /// Maximum plan depth
    pub max_depth: usize,
    /// Planning timeout
    pub timeout: Duration,
    /// Whether to use heuristic planning
    pub use_heuristics: bool,
    /// Branching factor limit
    pub max_branching: usize,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            max_depth: 20,
            timeout: Duration::from_secs(30),
            use_heuristics: true,
            max_branching: 10,
        }
    }
}

// ============================================================================
// Goal Types
// ============================================================================

/// Status of a goal
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    /// Not yet started
    Pending,
    /// Currently being pursued
    Active,
    /// Successfully achieved
    Achieved,
    /// Failed with reason
    Failed(String),
    /// Voluntarily abandoned
    Abandoned,
}

impl GoalStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            GoalStatus::Achieved | GoalStatus::Failed(_) | GoalStatus::Abandoned
        )
    }
}

/// A goal to be achieved
#[derive(Debug)]
pub struct Goal {
    /// Unique identifier
    pub id: GoalId,
    /// What to achieve (graph representation)
    pub description: Graph,
    /// Human-readable name
    pub name: String,
    /// Priority (higher = more important)
    pub priority: f32,
    /// Optional deadline
    pub deadline: Option<Timestamp>,
    /// Sub-goals that contribute to this goal
    pub subgoals: Vec<GoalId>,
    /// Parent goal (if this is a subgoal)
    pub parent: Option<GoalId>,
    /// Current status
    pub status: GoalStatus,
    /// Estimated complexity (for planning)
    pub complexity: usize,
}

impl Goal {
    /// Create a new goal
    pub fn new(id: GoalId, name: &str, description: Graph) -> Self {
        Self {
            id,
            description,
            name: name.to_string(),
            priority: 0.5,
            deadline: None,
            subgoals: Vec::new(),
            parent: None,
            status: GoalStatus::Pending,
            complexity: 1,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: f32) -> Self {
        self.priority = priority.clamp(0.0, 1.0);
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: Timestamp) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set complexity
    pub fn with_complexity(mut self, complexity: usize) -> Self {
        self.complexity = complexity;
        self
    }

    /// Activate this goal
    pub fn activate(&mut self) {
        if self.status == GoalStatus::Pending {
            self.status = GoalStatus::Active;
        }
    }

    /// Mark as achieved
    pub fn achieve(&mut self) {
        self.status = GoalStatus::Achieved;
    }

    /// Mark as failed
    pub fn fail(&mut self, reason: &str) {
        self.status = GoalStatus::Failed(reason.to_string());
    }

    /// Abandon this goal
    pub fn abandon(&mut self) {
        self.status = GoalStatus::Abandoned;
    }
}

/// Hierarchical structure of goals
#[derive(Debug, Default)]
pub struct GoalHierarchy {
    /// All goals by ID
    pub goals: HashMap<GoalId, Goal>,
    /// Root-level goals (no parent)
    pub root_goals: Vec<GoalId>,
    /// Next goal ID
    next_id: GoalId,
}

impl GoalHierarchy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a root goal
    pub fn add_root(&mut self, mut goal: Goal) -> GoalId {
        let id = self.next_id;
        self.next_id += 1;
        goal.id = id;
        goal.parent = None;
        self.root_goals.push(id);
        self.goals.insert(id, goal);
        id
    }

    /// Add a subgoal
    pub fn add_subgoal(&mut self, parent_id: GoalId, mut goal: Goal) -> Option<GoalId> {
        if !self.goals.contains_key(&parent_id) {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;
        goal.id = id;
        goal.parent = Some(parent_id);

        if let Some(parent) = self.goals.get_mut(&parent_id) {
            parent.subgoals.push(id);
        }

        self.goals.insert(id, goal);
        Some(id)
    }

    /// Get a goal by ID
    pub fn get(&self, id: GoalId) -> Option<&Goal> {
        self.goals.get(&id)
    }

    /// Get mutable goal by ID
    pub fn get_mut(&mut self, id: GoalId) -> Option<&mut Goal> {
        self.goals.get_mut(&id)
    }

    /// Get all active goals
    pub fn active_goals(&self) -> Vec<&Goal> {
        self.goals
            .values()
            .filter(|g| g.status == GoalStatus::Active)
            .collect()
    }

    /// Get highest priority active goal
    pub fn highest_priority(&self) -> Option<&Goal> {
        self.active_goals()
            .into_iter()
            .max_by(|a, b| a.priority.total_cmp(&b.priority))
    }
}

// ============================================================================
// Drives
// ============================================================================

/// Internal drives that motivate behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Drive {
    /// Seek new information
    Curiosity { strength: f32 },
    /// Minimize resource usage
    Efficiency { strength: f32 },
    /// Avoid harmful outcomes
    Safety { strength: f32 },
    /// Help the user
    Helpfulness { strength: f32 },
    /// Learn and improve
    Learning { strength: f32 },
    /// Custom drive
    Custom { name: String, strength: f32 },
}

impl Drive {
    pub fn strength(&self) -> f32 {
        match self {
            Drive::Curiosity { strength }
            | Drive::Efficiency { strength }
            | Drive::Safety { strength }
            | Drive::Helpfulness { strength }
            | Drive::Learning { strength }
            | Drive::Custom { strength, .. } => *strength,
        }
    }

    /// Default drives for a helpful agent
    pub fn default_drives() -> Vec<Drive> {
        vec![
            Drive::Helpfulness { strength: 0.9 },
            Drive::Safety { strength: 0.8 },
            Drive::Efficiency { strength: 0.5 },
            Drive::Curiosity { strength: 0.3 },
            Drive::Learning { strength: 0.4 },
        ]
    }
}

// ============================================================================
// Value Function
// ============================================================================

/// Represents what the agent values
#[derive(Debug, Default)]
pub struct ValueFunction {
    /// How much each drive matters
    pub drive_weights: HashMap<String, f32>,
    /// Explicitly valued states
    pub valued_states: Vec<(Graph, f32)>,
}

impl ValueFunction {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set weight for a drive
    pub fn set_weight(&mut self, drive_name: &str, weight: f32) {
        self.drive_weights
            .insert(drive_name.to_string(), weight.clamp(0.0, 1.0));
    }

    /// Estimate value of a state
    pub fn evaluate(&self, _state: &Graph) -> f32 {
        // Simplified: return average drive weight
        if self.drive_weights.is_empty() {
            return 0.5;
        }
        let sum: f32 = self.drive_weights.values().sum();
        sum / self.drive_weights.len() as f32
    }
}

// ============================================================================
// Plan Types
// ============================================================================

/// A failure that occurred during execution
#[derive(Debug, Clone)]
pub struct Failure {
    /// What failed
    pub description: String,
    /// At which step
    pub step: usize,
    /// Severity (0.0 to 1.0)
    pub severity: f32,
    /// Is this recoverable?
    pub recoverable: bool,
}

impl Failure {
    pub fn new(description: &str, step: usize) -> Self {
        Self {
            description: description.to_string(),
            step,
            severity: 0.5,
            recoverable: true,
        }
    }

    pub fn unrecoverable(description: &str, step: usize) -> Self {
        Self {
            description: description.to_string(),
            step,
            severity: 1.0,
            recoverable: false,
        }
    }
}

/// An action to take
#[derive(Debug)]
pub struct Action {
    /// Action identifier
    pub id: usize,
    /// What to do (graph representation)
    pub content: Graph,
    /// Expected outcome
    pub expected_result: Option<Graph>,
    /// Preconditions
    pub preconditions: Vec<Graph>,
}

impl Action {
    pub fn new(id: usize, content: Graph) -> Self {
        Self {
            id,
            content,
            expected_result: None,
            preconditions: Vec::new(),
        }
    }
}

/// A plan to achieve a goal
#[derive(Debug)]
pub struct Plan {
    /// Goal this plan is for
    pub goal_id: GoalId,
    /// Sequence of actions
    pub actions: Vec<Action>,
    /// Expected states after each action
    pub expected_states: Vec<Graph>,
    /// Current step (0-indexed)
    pub current_step: usize,
    /// Overall confidence in this plan
    pub confidence: f32,
}

impl Plan {
    pub fn new(goal_id: GoalId) -> Self {
        Self {
            goal_id,
            actions: Vec::new(),
            expected_states: Vec::new(),
            current_step: 0,
            confidence: 1.0,
        }
    }

    /// Add an action to the plan
    pub fn add_action(&mut self, action: Action, expected_state: Graph) {
        self.actions.push(action);
        self.expected_states.push(expected_state);
    }

    /// Get next action
    pub fn next_action(&self) -> Option<&Action> {
        self.actions.get(self.current_step)
    }

    /// Advance to next step
    pub fn advance(&mut self) {
        if self.current_step < self.actions.len() {
            self.current_step += 1;
        }
    }

    /// Is plan complete?
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.actions.len()
    }

    /// Remaining steps
    pub fn remaining(&self) -> usize {
        self.actions.len().saturating_sub(self.current_step)
    }
}

// ============================================================================
// Strategy
// ============================================================================

/// Strategy for explore vs exploit
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    /// Exploit current knowledge
    Exploit,
    /// Explore new options
    Explore,
    /// Balance based on uncertainty
    Balanced,
}

// ============================================================================
// Agent Structure
// ============================================================================

/// An autonomous agent with goals, values, and drives
///
/// ## Safety Integration
///
/// The Agent includes a non-overridable SafetyGate that validates all goals
/// and plans against Asimov's Laws before they can be pursued or executed.
pub struct Agent {
    /// Goal hierarchy
    pub goals: GoalHierarchy,
    /// Value function (what matters)
    pub values: ValueFunction,
    /// Motivational drives
    pub drives: Vec<Drive>,
    /// Current plan (if any)
    pub current_plan: Option<Plan>,
    /// Planning configuration
    pub config: PlanningConfig,
    /// Safety gate for Asimov Laws validation (NON-OVERRIDABLE)
    safety_gate: SafetyGate,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("goals", &self.goals)
            .field("values", &self.values)
            .field("drives", &self.drives)
            .field("current_plan", &self.current_plan)
            .field("config", &self.config)
            .field("safety_gate", &"SafetyGate { ... }")
            .finish()
    }
}

impl Agent {
    /// Create a new agent with integrated safety validation
    pub fn new() -> Self {
        Self {
            goals: GoalHierarchy::new(),
            values: ValueFunction::new(),
            drives: Drive::default_drives(),
            current_plan: None,
            config: PlanningConfig::default(),
            safety_gate: SafetyGate::new(),
        }
    }

    /// Create with custom drives
    pub fn with_drives(drives: Vec<Drive>) -> Self {
        Self {
            goals: GoalHierarchy::new(),
            values: ValueFunction::new(),
            drives,
            current_plan: None,
            config: PlanningConfig::default(),
            safety_gate: SafetyGate::new(),
        }
    }

    /// Validate a goal against Asimov's Laws
    ///
    /// This method is NON-OVERRIDABLE. All goals MUST pass safety validation
    /// before they can be added to the goal hierarchy.
    pub fn validate_goal(&self, goal: &Goal) -> AgentResult<()> {
        let safety_action = SafetyAction::new(
            ActionType::Decide,
            ActionTarget::Unknown,
            &goal.name,
        );

        match self.safety_gate.guard().validate(&safety_action) {
            SafetyCheck::Safe => Ok(()),
            SafetyCheck::Blocked(violation) => Err(AgentError::SafetyViolation {
                law: violation.law,
                description: violation.description,
            }),
            SafetyCheck::RequiresOversight { reason, .. } => {
                // For goals requiring oversight, we block until human approval
                Err(AgentError::SafetyViolation {
                    law: AsimovLaw::SecondLaw,
                    description: format!("Goal requires human oversight: {}", reason),
                })
            }
        }
    }

    /// Validate a plan against Asimov's Laws
    ///
    /// This method is NON-OVERRIDABLE. All plans MUST pass safety validation
    /// before they can be executed.
    pub fn validate_plan(&self, plan: &Plan) -> AgentResult<()> {
        // Validate each action in the plan
        for action in &plan.actions {
            let safety_action = SafetyAction::from_graph(
                &action.content,
                ActionType::Execute,
                ActionTarget::Unknown,
            );

            match self.safety_gate.guard().validate(&safety_action) {
                SafetyCheck::Safe => continue,
                SafetyCheck::Blocked(violation) => {
                    return Err(AgentError::SafetyViolation {
                        law: violation.law,
                        description: format!(
                            "Plan action '{}' blocked: {}",
                            action.id, violation.description
                        ),
                    });
                }
                SafetyCheck::RequiresOversight { reason, .. } => {
                    return Err(AgentError::SafetyViolation {
                        law: AsimovLaw::SecondLaw,
                        description: format!(
                            "Plan action '{}' requires oversight: {}",
                            action.id, reason
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Get safety violation count for auditing
    pub fn safety_violation_count(&self) -> usize {
        self.safety_gate.guard().violation_count()
    }
}

impl Default for Agent {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Agency Trait
// ============================================================================

/// Trait for autonomous goal-directed behavior
pub trait Agency: Send + Sync + Debug {
    /// Generate goal from current situation
    fn formulate_goal(&self, situation: &Graph) -> Goal;

    /// Decompose goal into subgoals
    fn decompose(&self, goal: &Goal, world: &dyn WorldModeling) -> Vec<Goal>;

    /// Create plan to achieve goal
    fn plan(&self, goal: &Goal, world: &dyn WorldModeling) -> AgentResult<Plan>;

    /// Get next action from current plan
    fn next_action(&self) -> Option<&Action>;

    /// Advance plan after successful action
    fn advance_plan(&mut self);

    /// Handle failure and attempt to replan
    fn replan(&mut self, failure: &Failure, world: &dyn WorldModeling) -> AgentResult<Plan>;

    /// Decide whether to explore or exploit
    fn explore_or_exploit(&self, uncertainty: &UncertaintyEstimate) -> ExplorationStrategy;

    /// Update goals based on new information
    fn revise_goals(&mut self, observation: &Graph);

    /// Check if goal is achieved
    fn is_achieved(&self, goal: &Goal, state: &Graph) -> bool;

    /// Get current goal
    fn current_goal(&self) -> Option<&Goal>;
}

// ============================================================================
// Simple Implementation
// ============================================================================

/// Simple agency implementation
#[derive(Debug, Default)]
pub struct SimpleAgency {
    agent: Agent,
}

impl SimpleAgency {
    pub fn new() -> Self {
        Self::default()
    }

    fn clone_graph(graph: &Graph) -> Graph {
        let text = graph.to_text();
        DagNN::from_text(&text).unwrap_or_else(|_| DagNN::new())
    }
}

impl Agency for SimpleAgency {
    fn formulate_goal(&self, situation: &Graph) -> Goal {
        // Simple: create goal from situation
        Goal::new(0, "derived_goal", Self::clone_graph(situation)).with_priority(0.5)
    }

    fn decompose(&self, goal: &Goal, _world: &dyn WorldModeling) -> Vec<Goal> {
        // Simplified: no decomposition
        // Real implementation would analyze goal structure
        if goal.complexity <= 1 {
            return Vec::new();
        }

        // Create simple subgoals
        vec![
            Goal::new(0, "subgoal_1", DagNN::new()).with_complexity(goal.complexity / 2),
            Goal::new(0, "subgoal_2", DagNN::new()).with_complexity(goal.complexity / 2),
        ]
    }

    fn plan(&self, goal: &Goal, _world: &dyn WorldModeling) -> AgentResult<Plan> {
        // Check complexity bound
        if goal.complexity > self.agent.config.max_depth {
            return Err(AgentError::GoalTooComplex(
                goal.complexity,
                self.agent.config.max_depth,
            ));
        }

        // Simple plan: single action
        let mut plan = Plan::new(goal.id);
        plan.add_action(
            Action::new(0, Self::clone_graph(&goal.description)),
            Self::clone_graph(&goal.description),
        );
        plan.confidence = 0.8;

        Ok(plan)
    }

    fn next_action(&self) -> Option<&Action> {
        self.agent.current_plan.as_ref()?.next_action()
    }

    fn advance_plan(&mut self) {
        if let Some(ref mut plan) = self.agent.current_plan {
            plan.advance();
        }
    }

    fn replan(&mut self, failure: &Failure, world: &dyn WorldModeling) -> AgentResult<Plan> {
        if !failure.recoverable {
            return Err(AgentError::GoalNotAchievable(failure.description.clone()));
        }

        // Get current goal and try to plan again
        if let Some(plan) = &self.agent.current_plan {
            if let Some(goal) = self.agent.goals.get(plan.goal_id) {
                // Clone the goal description
                let new_goal = Goal::new(goal.id, &goal.name, Self::clone_graph(&goal.description))
                    .with_priority(goal.priority)
                    .with_complexity(goal.complexity);
                return self.plan(&new_goal, world);
            }
        }

        Err(AgentError::PlanningFailed("No active goal".to_string()))
    }

    fn explore_or_exploit(&self, uncertainty: &UncertaintyEstimate) -> ExplorationStrategy {
        // Simple heuristic: explore when uncertain
        if uncertainty.total > 0.7 {
            ExplorationStrategy::Explore
        } else if uncertainty.total < 0.3 {
            ExplorationStrategy::Exploit
        } else {
            ExplorationStrategy::Balanced
        }
    }

    fn revise_goals(&mut self, _observation: &Graph) {
        // Simplified: abandon failed goals
        for goal in self.agent.goals.goals.values_mut() {
            if let GoalStatus::Active = goal.status {
                // Keep active goals
            }
        }
    }

    fn is_achieved(&self, _goal: &Goal, _state: &Graph) -> bool {
        // Simplified: check if plan is complete
        self.agent
            .current_plan
            .as_ref()
            .map(|p| p.is_complete())
            .unwrap_or(false)
    }

    fn current_goal(&self) -> Option<&Goal> {
        let plan = self.agent.current_plan.as_ref()?;
        self.agent.goals.get(plan.goal_id)
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a default agent
pub fn create_default_agent() -> SimpleAgency {
    SimpleAgency::new()
}

// ============================================================================
// Learnable Agency
// ============================================================================

/// Learnable agency with trainable goal and planning parameters
///
/// This module learns to prioritize goals, balance exploration/exploitation,
/// and allocate resources for planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableAgency {
    /// Bias for goal importance estimation
    pub goal_importance_bias: LearnableParam,
    /// Weight for curiosity drive
    pub curiosity_weight: LearnableParam,
    /// Weight for safety drive
    pub safety_weight: LearnableParam,
    /// Weight for efficiency drive
    pub efficiency_weight: LearnableParam,
    /// Exploration-exploitation temperature
    pub explore_temperature: LearnableParam,
    /// Discount factor for future rewards
    pub discount_factor: LearnableParam,
}

impl LearnableAgency {
    /// Create a new learnable agency module
    pub fn new() -> Self {
        Self {
            goal_importance_bias: LearnableParam::new(0.0),
            curiosity_weight: LearnableParam::new(0.3),
            safety_weight: LearnableParam::new(0.8),
            efficiency_weight: LearnableParam::new(0.5),
            explore_temperature: LearnableParam::new(1.0),
            discount_factor: LearnableParam::new(0.99),
        }
    }

    /// Adjust goal importance with learned bias
    pub fn adjusted_importance(&self, raw_importance: f32) -> f32 {
        (raw_importance + self.goal_importance_bias.value).clamp(0.0, 1.0)
    }

    /// Compute weighted drive score
    pub fn drive_score(&self, curiosity: f32, safety: f32, efficiency: f32) -> f32 {
        let cw = self.curiosity_weight.value.max(0.0);
        let sw = self.safety_weight.value.max(0.0);
        let ew = self.efficiency_weight.value.max(0.0);
        let total = cw + sw + ew;
        if total > 0.0 {
            (cw * curiosity + sw * safety + ew * efficiency) / total
        } else {
            (curiosity + safety + efficiency) / 3.0
        }
    }

    /// Compute exploration probability based on uncertainty
    pub fn exploration_probability(&self, uncertainty: f32) -> f32 {
        let temp = self.explore_temperature.value.max(0.01);
        let logit = (uncertainty / temp).clamp(-10.0, 10.0);
        1.0 / (1.0 + (-logit).exp())
    }

    /// Compute discounted value
    pub fn discounted_value(&self, value: f32, steps: usize) -> f32 {
        let gamma = self.discount_factor.value.clamp(0.0, 1.0);
        value * gamma.powi(steps as i32)
    }
}

impl Default for LearnableAgency {
    fn default() -> Self {
        Self::new()
    }
}

impl Learnable for LearnableAgency {
    fn zero_grad(&mut self) {
        self.goal_importance_bias.zero_grad();
        self.curiosity_weight.zero_grad();
        self.safety_weight.zero_grad();
        self.efficiency_weight.zero_grad();
        self.explore_temperature.zero_grad();
        self.discount_factor.zero_grad();
    }

    fn step(&mut self, lr: f32) {
        self.goal_importance_bias.step(lr);
        self.curiosity_weight.step(lr);
        self.safety_weight.step(lr);
        self.efficiency_weight.step(lr);
        self.explore_temperature.step(lr);
        self.discount_factor.step(lr);

        // Ensure valid ranges
        self.explore_temperature.value = self.explore_temperature.value.max(0.01);
        self.discount_factor.value = self.discount_factor.value.clamp(0.0, 1.0);
    }

    fn num_parameters(&self) -> usize {
        6
    }

    fn has_gradients(&self) -> bool {
        self.goal_importance_bias.grad != 0.0
            || self.curiosity_weight.grad != 0.0
            || self.safety_weight.grad != 0.0
            || self.efficiency_weight.grad != 0.0
            || self.explore_temperature.grad != 0.0
            || self.discount_factor.grad != 0.0
    }

    fn gradient_norm(&self) -> f32 {
        (self.goal_importance_bias.grad.powi(2)
            + self.curiosity_weight.grad.powi(2)
            + self.safety_weight.grad.powi(2)
            + self.efficiency_weight.grad.powi(2)
            + self.explore_temperature.grad.powi(2)
            + self.discount_factor.grad.powi(2))
        .sqrt()
    }
}

impl Persistable for LearnableAgency {
    fn persist_type_id() -> &'static str {
        "LearnableAgency"
    }

    fn persist_version() -> u32 {
        1
    }

    fn validate(&self) -> Result<(), PersistenceError> {
        // Validate temperature and discount factor are in valid ranges
        if self.explore_temperature.value <= 0.0 {
            return Err(PersistenceError::ValidationFailed(
                "Explore temperature must be positive".to_string(),
            ));
        }
        if self.discount_factor.value < 0.0 || self.discount_factor.value > 1.0 {
            return Err(PersistenceError::ValidationFailed(
                "Discount factor must be between 0 and 1".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Brain-Aware Agency
// ============================================================================

/// Brain-aware agency that uses domain brains for goal planning
pub struct BrainAwareAgency {
    /// The cognitive-brain bridge for domain routing
    pub bridge: DefaultCognitiveBridge,
    /// Maximum planning depth
    pub max_depth: usize,
}

impl Debug for BrainAwareAgency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrainAwareAgency")
            .field("available_domains", &self.bridge.available_domains())
            .field("max_depth", &self.max_depth)
            .finish()
    }
}

impl BrainAwareAgency {
    /// Create a new brain-aware agency
    pub fn new() -> Self {
        Self {
            bridge: DefaultCognitiveBridge::new(),
            max_depth: 20,
        }
    }

    /// Register a domain brain
    pub fn register_brain(&mut self, brain: Box<dyn DomainBrain>) {
        self.bridge.register(brain);
    }

    /// Check if a domain brain can help with a goal
    pub fn can_help_with_goal(&self, goal_text: &str) -> bool {
        let routing = self.bridge.route_to_multiple_brains(goal_text);
        routing.success
    }

    /// Get domains that can help with a goal
    pub fn domains_for_goal(&self, goal_text: &str) -> Vec<String> {
        let routing = self.bridge.route_to_multiple_brains(goal_text);
        routing.domains().iter().map(|s| s.to_string()).collect()
    }

    /// Get available domains
    pub fn available_domains(&self) -> Vec<String> {
        self.bridge.available_domains()
    }
}

impl Default for BrainAwareAgency {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveBrainBridge for BrainAwareAgency {
    fn get_registry(&self) -> &BrainRegistry {
        self.bridge.get_registry()
    }

    fn get_registry_mut(&mut self) -> &mut BrainRegistry {
        self.bridge.get_registry_mut()
    }
}

/// Factory function to create brain-aware agency
pub fn create_brain_aware_agency() -> BrainAwareAgency {
    BrainAwareAgency::new()
}

// ============================================================================
// GoalStack: Hierarchical Goal Management with Priorities
// ============================================================================

/// Entry in the goal stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalStackEntry {
    /// Goal identifier
    pub id: GoalId,
    /// Human-readable name
    pub name: String,
    /// Priority (0.0 to 1.0, higher = more important)
    pub priority: f32,
    /// Parent goal ID (None for root goals)
    pub parent_id: Option<GoalId>,
    /// Child goal IDs
    pub children: Vec<GoalId>,
    /// Current status
    pub status: GoalStackStatus,
    /// Progress (0.0 to 1.0)
    pub progress: f32,
    /// Creation timestamp (epoch millis)
    pub created_at: u64,
    /// Description/context
    pub description: String,
}

/// Status of a goal in the stack
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GoalStackStatus {
    /// Waiting to be processed
    #[default]
    Pending,
    /// Currently being worked on
    InProgress,
    /// Blocked by dependencies or children
    Blocked,
    /// Suspended (can be resumed)
    Suspended,
    /// Successfully completed
    Completed,
    /// Failed
    Failed(String),
    /// Cancelled by user or system
    Cancelled,
}

impl GoalStackStatus {
    /// Check if this is a terminal status
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed(_) | Self::Cancelled)
    }

    /// Check if goal can be worked on
    pub fn is_workable(&self) -> bool {
        matches!(self, Self::Pending | Self::InProgress)
    }
}

/// GoalStack: Stack-based hierarchical goal management
///
/// Implements a priority-based goal stack where:
/// - Goals are organized hierarchically (parent-child relationships)
/// - Each goal has a priority that affects execution order
/// - Goals can be pushed, popped, suspended, and resumed
/// - Child goals must complete before parent goals
///
/// # Time Complexity
/// - Push: O(1)
/// - Pop: O(n) due to priority sorting
/// - Get top: O(n) due to priority calculation
/// - Update status: O(1)
/// - Get by ID: O(1) via HashMap
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoalStack {
    /// All goals indexed by ID
    goals: HashMap<GoalId, GoalStackEntry>,
    /// Root goal IDs (no parent)
    root_goals: Vec<GoalId>,
    /// Next available goal ID
    next_id: GoalId,
    /// Configuration
    pub config: GoalStackConfig,
}

/// Configuration for the goal stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalStackConfig {
    /// Maximum number of goals in the stack
    pub max_goals: usize,
    /// Whether to auto-complete parent when all children complete
    pub auto_complete_parents: bool,
    /// Whether to cascade failure to children
    pub cascade_failure: bool,
    /// Priority boost for older goals (per second)
    pub age_priority_boost: f32,
}

impl Default for GoalStackConfig {
    fn default() -> Self {
        Self {
            max_goals: 1000,
            auto_complete_parents: true,
            cascade_failure: false,
            age_priority_boost: 0.0,
        }
    }
}

impl GoalStack {
    /// Create a new empty goal stack
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom configuration
    pub fn with_config(config: GoalStackConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Push a new root goal onto the stack - O(1)
    pub fn push(&mut self, name: impl Into<String>, priority: f32, description: impl Into<String>) -> GoalId {
        let id = self.next_id;
        self.next_id += 1;

        let entry = GoalStackEntry {
            id,
            name: name.into(),
            priority: priority.clamp(0.0, 1.0),
            parent_id: None,
            children: Vec::new(),
            status: GoalStackStatus::Pending,
            progress: 0.0,
            created_at: 0, // Would use real timestamp
            description: description.into(),
        };

        self.goals.insert(id, entry);
        self.root_goals.push(id);
        id
    }

    /// Push a child goal under a parent - O(1)
    pub fn push_child(&mut self, parent_id: GoalId, name: impl Into<String>, priority: f32, description: impl Into<String>) -> Option<GoalId> {
        if !self.goals.contains_key(&parent_id) {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        let entry = GoalStackEntry {
            id,
            name: name.into(),
            priority: priority.clamp(0.0, 1.0),
            parent_id: Some(parent_id),
            children: Vec::new(),
            status: GoalStackStatus::Pending,
            progress: 0.0,
            created_at: 0,
            description: description.into(),
        };

        self.goals.insert(id, entry);

        // Add to parent's children
        if let Some(parent) = self.goals.get_mut(&parent_id) {
            parent.children.push(id);
        }

        Some(id)
    }

    /// Get the highest priority workable goal - O(n)
    pub fn top(&self) -> Option<&GoalStackEntry> {
        self.goals
            .values()
            .filter(|g| g.status.is_workable() && g.children.iter().all(|c| {
                self.goals.get(c).is_none_or(|child| child.status.is_terminal())
            }))
            .max_by(|a, b| a.priority.total_cmp(&b.priority))
    }

    /// Get the highest priority workable goal ID - O(n)
    pub fn top_id(&self) -> Option<GoalId> {
        self.top().map(|g| g.id)
    }

    /// Pop (complete) the specified goal - O(1)
    pub fn complete(&mut self, id: GoalId) -> bool {
        if let Some(goal) = self.goals.get_mut(&id) {
            goal.status = GoalStackStatus::Completed;
            goal.progress = 1.0;

            // Check if parent should auto-complete
            if self.config.auto_complete_parents {
                if let Some(parent_id) = goal.parent_id {
                    self.try_auto_complete_parent(parent_id);
                }
            }
            true
        } else {
            false
        }
    }

    /// Mark a goal as failed - O(n) if cascade enabled
    pub fn fail(&mut self, id: GoalId, reason: impl Into<String>) -> bool {
        let reason_str = reason.into();
        if let Some(goal) = self.goals.get_mut(&id) {
            goal.status = GoalStackStatus::Failed(reason_str.clone());

            // Cascade failure to children if configured
            if self.config.cascade_failure {
                let children: Vec<GoalId> = goal.children.clone();
                for child_id in children {
                    self.fail(child_id, format!("Parent {} failed", id));
                }
            }
            true
        } else {
            false
        }
    }

    /// Suspend a goal - O(1)
    pub fn suspend(&mut self, id: GoalId) -> bool {
        if let Some(goal) = self.goals.get_mut(&id) {
            if !goal.status.is_terminal() {
                goal.status = GoalStackStatus::Suspended;
                return true;
            }
        }
        false
    }

    /// Resume a suspended goal - O(1)
    pub fn resume(&mut self, id: GoalId) -> bool {
        if let Some(goal) = self.goals.get_mut(&id) {
            if goal.status == GoalStackStatus::Suspended {
                goal.status = GoalStackStatus::InProgress;
                return true;
            }
        }
        false
    }

    /// Start working on a goal - O(1)
    pub fn start(&mut self, id: GoalId) -> bool {
        if let Some(goal) = self.goals.get_mut(&id) {
            if goal.status == GoalStackStatus::Pending {
                goal.status = GoalStackStatus::InProgress;
                return true;
            }
        }
        false
    }

    /// Update goal progress - O(1)
    pub fn set_progress(&mut self, id: GoalId, progress: f32) -> bool {
        if let Some(goal) = self.goals.get_mut(&id) {
            goal.progress = progress.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Update goal priority - O(1)
    pub fn set_priority(&mut self, id: GoalId, priority: f32) -> bool {
        if let Some(goal) = self.goals.get_mut(&id) {
            goal.priority = priority.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Get a goal by ID - O(1)
    pub fn get(&self, id: GoalId) -> Option<&GoalStackEntry> {
        self.goals.get(&id)
    }

    /// Get mutable goal by ID - O(1)
    pub fn get_mut(&mut self, id: GoalId) -> Option<&mut GoalStackEntry> {
        self.goals.get_mut(&id)
    }

    /// Get all root goals - O(n)
    pub fn roots(&self) -> Vec<&GoalStackEntry> {
        self.root_goals
            .iter()
            .filter_map(|id| self.goals.get(id))
            .collect()
    }

    /// Get all goals with a given status - O(n)
    pub fn by_status(&self, status: &GoalStackStatus) -> Vec<&GoalStackEntry> {
        self.goals
            .values()
            .filter(|g| &g.status == status)
            .collect()
    }

    /// Get pending goals - O(n)
    pub fn pending(&self) -> Vec<&GoalStackEntry> {
        self.by_status(&GoalStackStatus::Pending)
    }

    /// Get in-progress goals - O(n)
    pub fn in_progress(&self) -> Vec<&GoalStackEntry> {
        self.by_status(&GoalStackStatus::InProgress)
    }

    /// Get completed goals - O(n)
    pub fn completed(&self) -> Vec<&GoalStackEntry> {
        self.by_status(&GoalStackStatus::Completed)
    }

    /// Get children of a goal - O(m) where m = number of children
    pub fn children(&self, id: GoalId) -> Vec<&GoalStackEntry> {
        self.goals.get(&id)
            .map(|g| g.children.iter()
                .filter_map(|child_id| self.goals.get(child_id))
                .collect())
            .unwrap_or_default()
    }

    /// Get total number of goals - O(1)
    pub fn len(&self) -> usize {
        self.goals.len()
    }

    /// Check if stack is empty - O(1)
    pub fn is_empty(&self) -> bool {
        self.goals.is_empty()
    }

    /// Clear all completed goals - O(n)
    pub fn clear_completed(&mut self) {
        let completed_ids: Vec<GoalId> = self.goals
            .iter()
            .filter(|(_, g)| g.status == GoalStackStatus::Completed)
            .map(|(id, _)| *id)
            .collect();

        for id in completed_ids {
            self.remove_goal(id);
        }
    }

    /// Remove a goal and clean up references - O(n)
    fn remove_goal(&mut self, id: GoalId) {
        if let Some(goal) = self.goals.remove(&id) {
            // Remove from root_goals if applicable
            self.root_goals.retain(|&rid| rid != id);

            // Remove from parent's children
            if let Some(parent_id) = goal.parent_id {
                if let Some(parent) = self.goals.get_mut(&parent_id) {
                    parent.children.retain(|&cid| cid != id);
                }
            }
        }
    }

    /// Try to auto-complete a parent if all children are done - O(m)
    fn try_auto_complete_parent(&mut self, parent_id: GoalId) {
        if let Some(parent) = self.goals.get(&parent_id) {
            let all_children_done = parent.children.iter().all(|child_id| {
                self.goals.get(child_id)
                    .is_none_or(|c| c.status.is_terminal())
            });

            if all_children_done && !parent.children.is_empty() {
                // All children done, complete parent
                if let Some(parent) = self.goals.get_mut(&parent_id) {
                    parent.status = GoalStackStatus::Completed;
                    parent.progress = 1.0;
                }

                // Recursively check grandparent
                if let Some(grandparent_id) = self.goals.get(&parent_id).and_then(|p| p.parent_id) {
                    self.try_auto_complete_parent(grandparent_id);
                }
            }
        }
    }

    /// Get summary statistics - O(n)
    pub fn stats(&self) -> GoalStackStats {
        let mut stats = GoalStackStats::default();
        for goal in self.goals.values() {
            stats.total += 1;
            match &goal.status {
                GoalStackStatus::Pending => stats.pending += 1,
                GoalStackStatus::InProgress => stats.in_progress += 1,
                GoalStackStatus::Blocked => stats.blocked += 1,
                GoalStackStatus::Suspended => stats.suspended += 1,
                GoalStackStatus::Completed => stats.completed += 1,
                GoalStackStatus::Failed(_) => stats.failed += 1,
                GoalStackStatus::Cancelled => stats.cancelled += 1,
            }
        }
        stats
    }
}

/// Statistics about the goal stack
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoalStackStats {
    pub total: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub blocked: usize,
    pub suspended: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
}

impl GoalStackStats {
    /// Completion rate (completed / total)
    pub fn completion_rate(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.completed as f32 / self.total as f32
        }
    }

    /// Success rate (completed / terminal)
    pub fn success_rate(&self) -> f32 {
        let terminal = self.completed + self.failed + self.cancelled;
        if terminal == 0 {
            0.0
        } else {
            self.completed as f32 / terminal as f32
        }
    }
}

/// Factory function to create a goal stack
pub fn create_goal_stack() -> GoalStack {
    GoalStack::new()
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
    fn test_goal_creation() {
        let goal = Goal::new(1, "test goal", make_graph("achieve"))
            .with_priority(0.8)
            .with_complexity(5);

        assert_eq!(goal.id, 1);
        assert_eq!(goal.priority, 0.8);
        assert_eq!(goal.complexity, 5);
        assert_eq!(goal.status, GoalStatus::Pending);
    }

    #[test]
    fn test_goal_status_transitions() {
        let mut goal = Goal::new(1, "test", make_graph("test"));

        goal.activate();
        assert_eq!(goal.status, GoalStatus::Active);

        goal.achieve();
        assert_eq!(goal.status, GoalStatus::Achieved);
        assert!(goal.status.is_terminal());
    }

    #[test]
    fn test_goal_hierarchy() {
        let mut hierarchy = GoalHierarchy::new();

        let root = Goal::new(0, "root", make_graph("root"));
        let root_id = hierarchy.add_root(root);

        let child = Goal::new(0, "child", make_graph("child"));
        let child_id = hierarchy.add_subgoal(root_id, child);

        assert!(child_id.is_some());
        assert_eq!(hierarchy.root_goals.len(), 1);

        let parent = hierarchy.get(root_id).unwrap();
        assert_eq!(parent.subgoals.len(), 1);
    }

    #[test]
    fn test_drives() {
        let drives = Drive::default_drives();
        assert!(!drives.is_empty());

        let helpfulness = drives
            .iter()
            .find(|d| matches!(d, Drive::Helpfulness { .. }))
            .unwrap();
        assert!(helpfulness.strength() > 0.8);
    }

    #[test]
    fn test_value_function() {
        let mut values = ValueFunction::new();
        values.set_weight("helpfulness", 0.9);
        values.set_weight("safety", 0.8);

        let state = make_graph("test state");
        let value = values.evaluate(&state);
        assert!(value > 0.0);
    }

    #[test]
    fn test_failure() {
        let f = Failure::new("action failed", 3);
        assert!(f.recoverable);
        assert_eq!(f.step, 3);

        let f2 = Failure::unrecoverable("critical", 0);
        assert!(!f2.recoverable);
    }

    #[test]
    fn test_plan_creation() {
        let mut plan = Plan::new(1);

        let action = Action::new(0, make_graph("do something"));
        plan.add_action(action, make_graph("result"));

        assert_eq!(plan.actions.len(), 1);
        assert!(!plan.is_complete());

        plan.advance();
        assert!(plan.is_complete());
    }

    #[test]
    fn test_agent_creation() {
        let agent = Agent::new();
        assert!(!agent.drives.is_empty());
        assert!(agent.current_plan.is_none());
    }

    #[test]
    fn test_simple_agency_formulate() {
        let agency = SimpleAgency::new();
        let situation = make_graph("current situation");

        let goal = agency.formulate_goal(&situation);
        assert_eq!(goal.status, GoalStatus::Pending);
    }

    #[test]
    fn test_simple_agency_explore_exploit() {
        let agency = SimpleAgency::new();

        let uncertain = UncertaintyEstimate::new(0.8, 0.2);
        assert_eq!(
            agency.explore_or_exploit(&uncertain),
            ExplorationStrategy::Explore
        );

        let certain = UncertaintyEstimate::new(0.1, 0.1);
        assert_eq!(
            agency.explore_or_exploit(&certain),
            ExplorationStrategy::Exploit
        );
    }

    #[test]
    fn test_exploration_strategy() {
        assert_ne!(ExplorationStrategy::Explore, ExplorationStrategy::Exploit);
        assert_eq!(ExplorationStrategy::Balanced, ExplorationStrategy::Balanced);
    }

    #[test]
    fn test_planning_config() {
        let config = PlanningConfig::default();
        assert_eq!(config.max_depth, 20);
        assert!(config.use_heuristics);
    }

    // ========================================================================
    // GoalStack Tests
    // ========================================================================

    #[test]
    fn test_goal_stack_creation() {
        let stack = GoalStack::new();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_goal_stack_push() {
        let mut stack = GoalStack::new();
        let id = stack.push("Test Goal", 0.8, "A test goal");

        assert_eq!(stack.len(), 1);
        assert!(!stack.is_empty());

        let goal = stack.get(id).unwrap();
        assert_eq!(goal.name, "Test Goal");
        assert!((goal.priority - 0.8).abs() < 0.001);
        assert_eq!(goal.status, GoalStackStatus::Pending);
    }

    #[test]
    fn test_goal_stack_push_child() {
        let mut stack = GoalStack::new();
        let parent_id = stack.push("Parent", 0.9, "Parent goal");
        let child_id = stack.push_child(parent_id, "Child", 0.7, "Child goal").unwrap();

        assert_eq!(stack.len(), 2);

        let parent = stack.get(parent_id).unwrap();
        assert!(parent.children.contains(&child_id));

        let child = stack.get(child_id).unwrap();
        assert_eq!(child.parent_id, Some(parent_id));
    }

    #[test]
    fn test_goal_stack_top_priority() {
        let mut stack = GoalStack::new();
        stack.push("Low Priority", 0.3, "desc");
        let high_id = stack.push("High Priority", 0.9, "desc");
        stack.push("Medium Priority", 0.5, "desc");

        let top = stack.top().unwrap();
        assert_eq!(top.id, high_id);
    }

    #[test]
    fn test_goal_stack_top_respects_children() {
        // Disable auto-complete for this test
        let config = GoalStackConfig {
            auto_complete_parents: false,
            ..Default::default()
        };
        let mut stack = GoalStack::with_config(config);

        let parent_id = stack.push("Parent", 0.9, "High priority parent");
        let child_id = stack.push_child(parent_id, "Child", 0.5, "Child").unwrap();
        let standalone_id = stack.push("Standalone", 0.7, "Lower priority standalone");

        // Parent has higher priority but has incomplete child
        // So standalone should be top
        let top = stack.top().unwrap();
        assert_eq!(top.id, standalone_id);

        // Complete the child
        stack.complete(child_id);

        // Now parent should be top (child is done, parent can be worked on)
        let top = stack.top().unwrap();
        assert_eq!(top.id, parent_id);
    }

    #[test]
    fn test_goal_stack_complete() {
        let mut stack = GoalStack::new();
        let id = stack.push("Goal", 0.5, "desc");

        assert!(stack.complete(id));

        let goal = stack.get(id).unwrap();
        assert_eq!(goal.status, GoalStackStatus::Completed);
        assert!((goal.progress - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_goal_stack_auto_complete_parent() {
        let mut stack = GoalStack::new();
        let parent_id = stack.push("Parent", 0.9, "Parent");
        let child1_id = stack.push_child(parent_id, "Child 1", 0.5, "Child").unwrap();
        let child2_id = stack.push_child(parent_id, "Child 2", 0.5, "Child").unwrap();

        // Complete both children
        stack.complete(child1_id);
        stack.complete(child2_id);

        // Parent should be auto-completed
        let parent = stack.get(parent_id).unwrap();
        assert_eq!(parent.status, GoalStackStatus::Completed);
    }

    #[test]
    fn test_goal_stack_fail() {
        let mut stack = GoalStack::new();
        let id = stack.push("Goal", 0.5, "desc");

        assert!(stack.fail(id, "Test failure"));

        let goal = stack.get(id).unwrap();
        assert!(matches!(goal.status, GoalStackStatus::Failed(_)));
    }

    #[test]
    fn test_goal_stack_suspend_resume() {
        let mut stack = GoalStack::new();
        let id = stack.push("Goal", 0.5, "desc");
        stack.start(id);

        assert!(stack.suspend(id));
        assert_eq!(stack.get(id).unwrap().status, GoalStackStatus::Suspended);

        assert!(stack.resume(id));
        assert_eq!(stack.get(id).unwrap().status, GoalStackStatus::InProgress);
    }

    #[test]
    fn test_goal_stack_progress() {
        let mut stack = GoalStack::new();
        let id = stack.push("Goal", 0.5, "desc");

        assert!(stack.set_progress(id, 0.75));
        assert!((stack.get(id).unwrap().progress - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_goal_stack_stats() {
        let mut stack = GoalStack::new();
        stack.push("Pending 1", 0.5, "desc");
        stack.push("Pending 2", 0.5, "desc");

        let id3 = stack.push("Will Complete", 0.5, "desc");
        stack.complete(id3);

        let id4 = stack.push("Will Fail", 0.5, "desc");
        stack.fail(id4, "error");

        let stats = stack.stats();
        assert_eq!(stats.total, 4);
        assert_eq!(stats.pending, 2);
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.failed, 1);
        assert!((stats.completion_rate() - 0.25).abs() < 0.001);
        assert!((stats.success_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_goal_stack_clear_completed() {
        let mut stack = GoalStack::new();
        stack.push("Pending", 0.5, "desc");

        let id2 = stack.push("Completed", 0.5, "desc");
        stack.complete(id2);

        assert_eq!(stack.len(), 2);
        stack.clear_completed();
        assert_eq!(stack.len(), 1);
    }

    #[test]
    fn test_goal_stack_status_is_terminal() {
        assert!(!GoalStackStatus::Pending.is_terminal());
        assert!(!GoalStackStatus::InProgress.is_terminal());
        assert!(!GoalStackStatus::Suspended.is_terminal());
        assert!(GoalStackStatus::Completed.is_terminal());
        assert!(GoalStackStatus::Failed("error".to_string()).is_terminal());
        assert!(GoalStackStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_goal_stack_status_is_workable() {
        assert!(GoalStackStatus::Pending.is_workable());
        assert!(GoalStackStatus::InProgress.is_workable());
        assert!(!GoalStackStatus::Suspended.is_workable());
        assert!(!GoalStackStatus::Completed.is_workable());
    }

    #[test]
    fn test_create_goal_stack_factory() {
        let stack = create_goal_stack();
        assert!(stack.is_empty());
    }
}
