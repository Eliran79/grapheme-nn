//! # grapheme-world
//!
//! World model for GRAPHEME neural network.
//!
//! This crate provides an internal simulation of reality for:
//! - **Prediction**: Forecasting future states given actions
//! - **Explanation**: Understanding why things happen (causal graphs)
//! - **Imagination**: Generating novel situations from constraints
//! - **Simulation**: Running multi-step action sequences
//!
//! The world model maintains:
//! - Entities (objects and agents)
//! - Relations (spatial, causal, social)
//! - Dynamics (transition rules for how things change)
//!
//! This enables planning, counterfactual reasoning, and mental simulation.

use grapheme_core::{DagNN, Learnable, LearnableParam, TransformRule};
use grapheme_reason::CausalGraph;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;

// ============================================================================
// Type Aliases
// ============================================================================

/// Graph type for world states
pub type Graph = DagNN;

// ============================================================================
// Error Types
// ============================================================================

/// Errors in world modeling operations
#[derive(Error, Debug)]
pub enum WorldError {
    #[error("Invalid action: {0}")]
    InvalidAction(String),
    #[error("Prediction failed: {0}")]
    PredictionFailed(String),
    #[error("Simulation limit exceeded: {0} steps")]
    SimulationLimitExceeded(usize),
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    #[error("No causal explanation found")]
    NoCausalExplanation,
}

/// Result type for world modeling operations
pub type WorldResult<T> = Result<T, WorldError>;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for world modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    /// Maximum simulation steps
    pub max_simulation_steps: usize,
    /// Maximum prediction horizon
    pub max_prediction_horizon: usize,
    /// Minimum probability threshold for predictions
    pub min_probability: f32,
    /// Whether to track uncertainty
    pub track_uncertainty: bool,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            max_simulation_steps: 100,
            max_prediction_horizon: 20,
            min_probability: 0.01,
            track_uncertainty: true,
        }
    }
}

// ============================================================================
// Prediction
// ============================================================================

/// A prediction of a future state
#[derive(Debug)]
pub struct Prediction {
    /// The predicted state graph
    pub state: Graph,
    /// Probability of this prediction (0.0 to 1.0)
    pub probability: f32,
    /// Time step this prediction is for
    pub time_step: usize,
    /// Uncertainty estimate (higher = less certain)
    pub uncertainty: f32,
    /// Actions that led to this state
    pub action_trace: Vec<usize>,
}

impl Prediction {
    /// Create a new prediction
    pub fn new(state: Graph, probability: f32, time_step: usize) -> Self {
        Self {
            state,
            probability: probability.clamp(0.0, 1.0),
            time_step,
            uncertainty: 1.0 - probability.clamp(0.0, 1.0),
            action_trace: Vec::new(),
        }
    }

    /// Create initial state (t=0) prediction
    pub fn initial(state: Graph) -> Self {
        Self {
            state,
            probability: 1.0,
            time_step: 0,
            uncertainty: 0.0,
            action_trace: Vec::new(),
        }
    }
}

// ============================================================================
// Dynamics (Transition Rules)
// ============================================================================

/// A transition rule describing how states change
#[derive(Debug)]
pub struct TransitionRule {
    /// Rule identifier
    pub id: usize,
    /// Rule name
    pub name: String,
    /// Precondition pattern
    pub precondition: Graph,
    /// Action pattern (what triggers this transition)
    pub action: Graph,
    /// Effect pattern (how state changes)
    pub effect: Graph,
    /// Probability of this transition occurring
    pub probability: f32,
}

impl TransitionRule {
    pub fn new(id: usize, name: &str, precondition: Graph, action: Graph, effect: Graph) -> Self {
        Self {
            id,
            name: name.to_string(),
            precondition,
            action,
            effect,
            probability: 1.0,
        }
    }

    pub fn with_probability(mut self, probability: f32) -> Self {
        self.probability = probability.clamp(0.0, 1.0);
        self
    }
}

/// Collection of dynamics rules
#[derive(Debug, Default)]
pub struct Dynamics {
    /// Transition rules
    pub rules: Vec<TransitionRule>,
    /// Default rules (always applicable)
    pub default_rules: Vec<TransformRule>,
}

impl Dynamics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a transition rule
    pub fn add_rule(&mut self, rule: TransitionRule) {
        self.rules.push(rule);
    }

    /// Get applicable rules for a state and action
    pub fn applicable(&self, _state: &Graph, _action: &Graph) -> Vec<&TransitionRule> {
        // Simplified: return all rules
        // Real implementation would check pattern matching
        self.rules.iter().collect()
    }

    /// Number of rules
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

// ============================================================================
// World State
// ============================================================================

/// A snapshot of the world state
#[derive(Debug)]
pub struct WorldState {
    /// Current entities in the world
    pub entities: Graph,
    /// Current relations between entities
    pub relations: Graph,
    /// Current time step
    pub time: usize,
    /// State metadata
    pub metadata: StateMetadata,
}

/// Metadata about a world state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateMetadata {
    /// Whether this is a hypothetical state
    pub is_hypothetical: bool,
    /// Whether this state was observed or inferred
    pub is_observed: bool,
    /// Confidence in this state
    pub confidence: f32,
}

impl WorldState {
    /// Create a new world state
    pub fn new(entities: Graph, relations: Graph) -> Self {
        Self {
            entities,
            relations,
            time: 0,
            metadata: StateMetadata {
                is_hypothetical: false,
                is_observed: true,
                confidence: 1.0,
            },
        }
    }

    /// Create a hypothetical state
    pub fn hypothetical(entities: Graph, relations: Graph, confidence: f32) -> Self {
        Self {
            entities,
            relations,
            time: 0,
            metadata: StateMetadata {
                is_hypothetical: true,
                is_observed: false,
                confidence: confidence.clamp(0.0, 1.0),
            },
        }
    }

    /// Advance time
    pub fn advance(&mut self) {
        self.time += 1;
    }
}

// ============================================================================
// World Model
// ============================================================================

/// The complete world model
#[derive(Debug)]
pub struct WorldModel {
    /// Current world state
    pub state: WorldState,
    /// Dynamics rules
    pub dynamics: Dynamics,
    /// Counterfactual states (alternative realities)
    pub counterfactuals: Vec<WorldState>,
    /// Configuration
    pub config: WorldConfig,
    /// History of past states
    #[allow(dead_code)]
    history: Vec<WorldState>,
}

impl WorldModel {
    /// Create a new world model
    pub fn new(initial_state: WorldState) -> Self {
        Self {
            state: initial_state,
            dynamics: Dynamics::new(),
            counterfactuals: Vec::new(),
            config: WorldConfig::default(),
            history: Vec::new(),
        }
    }

    /// Create with configuration
    pub fn with_config(initial_state: WorldState, config: WorldConfig) -> Self {
        Self {
            state: initial_state,
            dynamics: Dynamics::new(),
            counterfactuals: Vec::new(),
            config,
            history: Vec::new(),
        }
    }

    /// Add a transition rule
    pub fn add_rule(&mut self, rule: TransitionRule) {
        self.dynamics.add_rule(rule);
    }

    /// Get current time
    pub fn current_time(&self) -> usize {
        self.state.time
    }
}

// ============================================================================
// World Modeling Trait
// ============================================================================

/// Trait for world simulation and prediction
pub trait WorldModeling: Send + Sync + Debug {
    /// Predict future states given current state and action
    ///
    /// Returns multiple predictions representing possible outcomes.
    fn predict(&self, state: &Graph, action: &Graph, horizon: usize)
        -> WorldResult<Vec<Prediction>>;

    /// Explain an observation via causal graph
    ///
    /// Returns the causal structure that explains the observation.
    fn explain(&self, observation: &Graph) -> WorldResult<CausalGraph>;

    /// Generate novel situation from constraints
    ///
    /// Creates a hypothetical world state satisfying the constraints.
    fn imagine(&self, constraints: &Graph) -> WorldResult<Graph>;

    /// Update world model from observation
    fn update(&mut self, observation: &Graph) -> WorldResult<()>;

    /// Simulate action sequence
    ///
    /// Returns the sequence of states after applying each action.
    fn simulate(&self, initial: &Graph, actions: &[Graph]) -> WorldResult<Vec<Graph>>;

    /// Generate counterfactual: "What if X had been different?"
    fn counterfactual(&self, actual: &Graph, change: &Graph) -> WorldResult<Graph>;

    /// Get the most likely next state
    fn next_state(&self, current: &Graph, action: &Graph) -> WorldResult<Graph>;
}

// ============================================================================
// Entity and Relation Types
// ============================================================================

/// Type of entity in the world
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    /// Physical object
    Object,
    /// Agent that can take actions
    Agent,
    /// Location/place
    Location,
    /// Abstract concept
    Concept,
    /// Event/happening
    Event,
    /// Custom type
    Custom(String),
}

/// Type of relation between entities
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationType {
    /// Spatial: A is at/in/on B
    Spatial,
    /// Causal: A causes B
    Causal,
    /// Temporal: A happens before B
    Temporal,
    /// Ownership: A owns/has B
    Ownership,
    /// Social: A knows/likes/is-related-to B
    Social,
    /// Part-whole: A is part of B
    PartOf,
    /// Custom relation
    Custom(String),
}

// ============================================================================
// Simple Implementation
// ============================================================================

/// Simple world model implementation
#[derive(Debug)]
pub struct SimpleWorldModel {
    state: WorldState,
    dynamics: Dynamics,
    config: WorldConfig,
}

impl SimpleWorldModel {
    /// Create a new simple world model
    pub fn new(initial_state: WorldState) -> Self {
        Self {
            state: initial_state,
            dynamics: Dynamics::new(),
            config: WorldConfig::default(),
        }
    }

    /// Add a dynamics rule
    pub fn add_rule(&mut self, rule: TransitionRule) {
        self.dynamics.add_rule(rule);
    }

    fn clone_graph(graph: &Graph) -> Graph {
        let text = graph.to_text();
        DagNN::from_text(&text).unwrap_or_else(|_| DagNN::new())
    }
}

impl WorldModeling for SimpleWorldModel {
    fn predict(&self, state: &Graph, _action: &Graph, horizon: usize)
        -> WorldResult<Vec<Prediction>> {
        if horizon > self.config.max_prediction_horizon {
            return Err(WorldError::SimulationLimitExceeded(horizon));
        }

        // Simplified: return same state with decreasing probability
        let mut predictions = Vec::new();
        let mut prob = 1.0f32;

        for t in 0..=horizon {
            predictions.push(Prediction::new(
                Self::clone_graph(state),
                prob,
                t,
            ));
            prob *= 0.9; // Probability decreases over time
        }

        Ok(predictions)
    }

    fn explain(&self, observation: &Graph) -> WorldResult<CausalGraph> {
        // Simplified: return observation as causal graph
        Ok(CausalGraph::new(Self::clone_graph(observation)))
    }

    fn imagine(&self, constraints: &Graph) -> WorldResult<Graph> {
        // Simplified: return constraints as imagined state
        Ok(Self::clone_graph(constraints))
    }

    fn update(&mut self, observation: &Graph) -> WorldResult<()> {
        // Update entities with observation
        self.state.entities = Self::clone_graph(observation);
        self.state.advance();
        Ok(())
    }

    fn simulate(&self, initial: &Graph, actions: &[Graph]) -> WorldResult<Vec<Graph>> {
        if actions.len() > self.config.max_simulation_steps {
            return Err(WorldError::SimulationLimitExceeded(actions.len()));
        }

        // Simplified: return initial state for each action
        let states: Vec<Graph> = std::iter::once(Self::clone_graph(initial))
            .chain(actions.iter().map(|_| Self::clone_graph(initial)))
            .collect();

        Ok(states)
    }

    fn counterfactual(&self, actual: &Graph, _change: &Graph) -> WorldResult<Graph> {
        // Simplified: return actual (real impl would apply change)
        Ok(Self::clone_graph(actual))
    }

    fn next_state(&self, current: &Graph, _action: &Graph) -> WorldResult<Graph> {
        // Simplified: return current state
        Ok(Self::clone_graph(current))
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a default world model
pub fn create_default_world_model() -> SimpleWorldModel {
    let entities = DagNN::new();
    let relations = DagNN::new();
    let state = WorldState::new(entities, relations);
    SimpleWorldModel::new(state)
}

// ============================================================================
// Learnable World Model
// ============================================================================

/// Learnable world model with trainable transition dynamics
///
/// This module learns to predict state transitions and adjust
/// uncertainty estimates based on experience.
#[derive(Debug, Clone)]
pub struct LearnableWorldModel {
    /// Bias for transition probability estimation
    pub transition_bias: LearnableParam,
    /// Confidence in predictions
    pub prediction_confidence: LearnableParam,
    /// Temporal discount factor for multi-step predictions
    pub temporal_discount: LearnableParam,
    /// Weight for entity changes in state updates
    pub entity_weight: LearnableParam,
    /// Weight for relation changes in state updates
    pub relation_weight: LearnableParam,
    /// Uncertainty scaling factor
    pub uncertainty_scale: LearnableParam,
}

impl LearnableWorldModel {
    /// Create a new learnable world model with default weights
    pub fn new() -> Self {
        Self {
            transition_bias: LearnableParam::new(0.0),
            prediction_confidence: LearnableParam::new(0.8),
            temporal_discount: LearnableParam::new(0.9),
            entity_weight: LearnableParam::new(0.5),
            relation_weight: LearnableParam::new(0.5),
            uncertainty_scale: LearnableParam::new(1.0),
        }
    }

    /// Compute transition probability with learned parameters
    pub fn transition_probability(&self, base_prob: f32) -> f32 {
        (base_prob + self.transition_bias.value).clamp(0.0, 1.0)
    }

    /// Compute prediction confidence with temporal decay
    pub fn decayed_confidence(&self, time_steps: usize) -> f32 {
        let base = self.prediction_confidence.value;
        let discount = self.temporal_discount.value.clamp(0.0, 1.0);
        base * discount.powi(time_steps as i32)
    }

    /// Compute combined state change score
    pub fn state_change_score(&self, entity_change: f32, relation_change: f32) -> f32 {
        let ew = self.entity_weight.value.clamp(0.0, 1.0);
        let rw = self.relation_weight.value.clamp(0.0, 1.0);
        let total = ew + rw;
        if total > 0.0 {
            (ew * entity_change + rw * relation_change) / total
        } else {
            0.5 * entity_change + 0.5 * relation_change
        }
    }

    /// Compute uncertainty for a prediction
    pub fn prediction_uncertainty(&self, base_uncertainty: f32) -> f32 {
        (base_uncertainty * self.uncertainty_scale.value).clamp(0.0, 1.0)
    }
}

impl Default for LearnableWorldModel {
    fn default() -> Self {
        Self::new()
    }
}

impl Learnable for LearnableWorldModel {
    fn zero_grad(&mut self) {
        self.transition_bias.zero_grad();
        self.prediction_confidence.zero_grad();
        self.temporal_discount.zero_grad();
        self.entity_weight.zero_grad();
        self.relation_weight.zero_grad();
        self.uncertainty_scale.zero_grad();
    }

    fn step(&mut self, lr: f32) {
        self.transition_bias.step(lr);
        self.prediction_confidence.step(lr);
        self.temporal_discount.step(lr);
        self.entity_weight.step(lr);
        self.relation_weight.step(lr);
        self.uncertainty_scale.step(lr);

        // Ensure values stay in valid ranges
        self.prediction_confidence.value = self.prediction_confidence.value.clamp(0.0, 1.0);
        self.temporal_discount.value = self.temporal_discount.value.clamp(0.0, 1.0);
        self.uncertainty_scale.value = self.uncertainty_scale.value.max(0.01);
    }

    fn num_parameters(&self) -> usize {
        6 // transition_bias, prediction_confidence, temporal_discount, entity/relation weights, uncertainty_scale
    }

    fn has_gradients(&self) -> bool {
        self.transition_bias.grad != 0.0
            || self.prediction_confidence.grad != 0.0
            || self.temporal_discount.grad != 0.0
            || self.entity_weight.grad != 0.0
            || self.relation_weight.grad != 0.0
            || self.uncertainty_scale.grad != 0.0
    }

    fn gradient_norm(&self) -> f32 {
        (self.transition_bias.grad.powi(2)
            + self.prediction_confidence.grad.powi(2)
            + self.temporal_discount.grad.powi(2)
            + self.entity_weight.grad.powi(2)
            + self.relation_weight.grad.powi(2)
            + self.uncertainty_scale.grad.powi(2))
        .sqrt()
    }
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
    fn test_prediction_creation() {
        let state = make_graph("state");
        let pred = Prediction::new(state, 0.8, 5);

        assert_eq!(pred.probability, 0.8);
        assert_eq!(pred.time_step, 5);
        assert!((pred.uncertainty - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_prediction_initial() {
        let state = make_graph("initial");
        let pred = Prediction::initial(state);

        assert_eq!(pred.probability, 1.0);
        assert_eq!(pred.time_step, 0);
        assert_eq!(pred.uncertainty, 0.0);
    }

    #[test]
    fn test_transition_rule() {
        let pre = make_graph("precondition");
        let action = make_graph("action");
        let effect = make_graph("effect");

        let rule = TransitionRule::new(1, "test_rule", pre, action, effect)
            .with_probability(0.9);

        assert_eq!(rule.id, 1);
        assert_eq!(rule.probability, 0.9);
    }

    #[test]
    fn test_dynamics() {
        let mut dynamics = Dynamics::new();
        assert!(dynamics.is_empty());

        let rule = TransitionRule::new(
            1, "rule1",
            make_graph("pre"),
            make_graph("act"),
            make_graph("eff"),
        );
        dynamics.add_rule(rule);

        assert_eq!(dynamics.len(), 1);
    }

    #[test]
    fn test_world_state() {
        let entities = make_graph("entities");
        let relations = make_graph("relations");

        let mut state = WorldState::new(entities, relations);
        assert_eq!(state.time, 0);
        assert!(state.metadata.is_observed);

        state.advance();
        assert_eq!(state.time, 1);
    }

    #[test]
    fn test_hypothetical_state() {
        let entities = make_graph("hypothetical");
        let relations = make_graph("relations");

        let state = WorldState::hypothetical(entities, relations, 0.7);
        assert!(state.metadata.is_hypothetical);
        assert!(!state.metadata.is_observed);
        assert!((state.metadata.confidence - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_world_model_creation() {
        let state = WorldState::new(make_graph("e"), make_graph("r"));
        let model = WorldModel::new(state);

        assert_eq!(model.current_time(), 0);
        assert!(model.dynamics.is_empty());
    }

    #[test]
    fn test_simple_world_model_predict() {
        let model = create_default_world_model();
        let state = make_graph("test_state");
        let action = make_graph("action");

        let predictions = model.predict(&state, &action, 5).unwrap();
        assert_eq!(predictions.len(), 6); // 0 to 5 inclusive
    }

    #[test]
    fn test_simple_world_model_simulate() {
        let model = create_default_world_model();
        let initial = make_graph("initial");
        let actions = vec![
            make_graph("action1"),
            make_graph("action2"),
        ];

        let states = model.simulate(&initial, &actions).unwrap();
        assert_eq!(states.len(), 3); // initial + 2 actions
    }

    #[test]
    fn test_simple_world_model_update() {
        let mut model = create_default_world_model();
        let observation = make_graph("new_observation");

        model.update(&observation).unwrap();
        assert_eq!(model.state.time, 1);
    }

    #[test]
    fn test_entity_types() {
        let obj = EntityType::Object;
        let agent = EntityType::Agent;
        let custom = EntityType::Custom("MyType".to_string());

        assert_ne!(obj, agent);
        assert_eq!(custom, EntityType::Custom("MyType".to_string()));
    }

    #[test]
    fn test_relation_types() {
        let spatial = RelationType::Spatial;
        let causal = RelationType::Causal;

        assert_ne!(spatial, causal);
    }

    #[test]
    fn test_world_config() {
        let config = WorldConfig::default();
        assert_eq!(config.max_simulation_steps, 100);
        assert_eq!(config.max_prediction_horizon, 20);
    }
}
