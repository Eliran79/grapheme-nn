//! # grapheme-ground
//!
//! Grounding layer for GRAPHEME neural network.
//!
//! This crate provides symbol-referent binding and embodiment:
//! - **Grounding**: Connect symbols to perceptual/conceptual referents
//! - **Perception**: Sensor interface for multi-modal perception
//! - **Action**: Actuator interface for world interaction
//! - **Embodiment**: Perception-action loop for learning through interaction
//!
//! ## The Symbol Grounding Problem
//!
//! Symbols (like "cat") have no intrinsic meaning - they're just graph structures.
//! Grounding connects these symbols to referents (visual representations, behaviors,
//! etc.) that give them meaning.
//!
//! ## Research Status
//!
//! True grounding may require embodiment. This module provides interfaces for:
//! - Supervised labeling (weak grounding)
//! - Co-occurrence statistics (LLM-style)
//! - Simulated interaction (compromise)
//! - Embodied interaction (ideal but hard)

use grapheme_core::{
    BrainRegistry, CognitiveBrainBridge, DagNN, DefaultCognitiveBridge, DomainBrain, Learnable,
    LearnableParam, Persistable, PersistenceError,
};
use grapheme_multimodal::{ModalGraph, Modality};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;

// ============================================================================
// Type Aliases
// ============================================================================

/// Graph type
pub type Graph = DagNN;

/// Node identifier
pub type NodeId = u64;

// ============================================================================
// Error Types
// ============================================================================

/// Errors in grounding operations
#[derive(Error, Debug)]
pub enum GroundingError {
    #[error("Symbol not found: {0}")]
    SymbolNotFound(NodeId),
    #[error("No grounding available for symbol")]
    NoGrounding,
    #[error("Grounding verification failed: {0}")]
    VerificationFailed(String),
    #[error("Perception error: {0}")]
    PerceptionError(String),
}

/// Errors in action execution
#[derive(Error, Debug)]
pub enum ActionError {
    #[error("Action cannot be executed: {0}")]
    CannotExecute(String),
    #[error("Precondition not met: {0}")]
    PreconditionNotMet(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Unknown action type")]
    UnknownAction,
}

/// Result type for grounding operations
pub type GroundingResult<T> = Result<T, GroundingError>;

// ============================================================================
// Referent Types
// ============================================================================

/// External reference (to APIs, databases, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalRef {
    /// Type of external system
    pub system_type: String,
    /// Identifier within that system
    pub identifier: String,
    /// Optional metadata
    pub metadata: Option<String>,
}

impl ExternalRef {
    pub fn new(system_type: &str, identifier: &str) -> Self {
        Self {
            system_type: system_type.to_string(),
            identifier: identifier.to_string(),
            metadata: None,
        }
    }
}

/// Something in the world that a symbol refers to
#[derive(Debug)]
pub enum Referent {
    /// Perceptual representation (image region, sound, etc.)
    Perceptual(ModalGraph),

    /// Abstract concept (defined by relations to other concepts)
    Conceptual(Graph),

    /// Action/procedure (how to do something)
    Procedural(ProcedureSpec),

    /// External entity (API, database, etc.)
    External(ExternalRef),
}

/// Specification of a procedure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureSpec {
    /// Name of the procedure
    pub name: String,
    /// Steps to execute
    pub steps: Vec<String>,
    /// Preconditions
    pub preconditions: Vec<String>,
}

impl ProcedureSpec {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            steps: Vec::new(),
            preconditions: Vec::new(),
        }
    }

    pub fn with_steps(mut self, steps: Vec<&str>) -> Self {
        self.steps = steps.into_iter().map(|s| s.to_string()).collect();
        self
    }
}

// ============================================================================
// Grounding Types
// ============================================================================

/// How the grounding was established
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroundingSource {
    /// Explicitly labeled by human
    Supervised,
    /// Inferred from context
    Inferred,
    /// Learned through interaction
    Embodied,
    /// Derived from co-occurrence statistics
    Linguistic,
    /// From simulation
    Simulated,
}

/// Grounding binding between symbol and referent
#[derive(Debug)]
pub struct Grounding {
    /// The symbol being grounded
    pub symbol: NodeId,
    /// What it refers to
    pub referent: Referent,
    /// Confidence in this grounding (0.0 to 1.0)
    pub confidence: f32,
    /// How was this grounding established
    pub source: GroundingSource,
    /// Verification count (how many times verified)
    pub verification_count: usize,
}

impl Grounding {
    /// Create a new grounding
    pub fn new(
        symbol: NodeId,
        referent: Referent,
        confidence: f32,
        source: GroundingSource,
    ) -> Self {
        Self {
            symbol,
            referent,
            confidence: confidence.clamp(0.0, 1.0),
            source,
            verification_count: 0,
        }
    }

    /// Create supervised grounding
    pub fn supervised(symbol: NodeId, referent: Referent) -> Self {
        Self::new(symbol, referent, 1.0, GroundingSource::Supervised)
    }

    /// Create inferred grounding
    pub fn inferred(symbol: NodeId, referent: Referent, confidence: f32) -> Self {
        Self::new(symbol, referent, confidence, GroundingSource::Inferred)
    }

    /// Update confidence after verification
    pub fn update_confidence(&mut self, verified: bool) {
        self.verification_count += 1;
        if verified {
            self.confidence = (self.confidence + 1.0) / 2.0; // Move toward 1.0
        } else {
            self.confidence /= 2.0; // Move toward 0.0
        }
    }
}

/// Identifier for a grounding
pub type GroundingId = u64;

// ============================================================================
// Interaction Types
// ============================================================================

/// An interaction with the world
#[derive(Debug)]
pub struct Interaction {
    /// Action taken
    pub action: Graph,
    /// Perception before action
    pub before: Option<ModalGraph>,
    /// Perception after action
    pub after: Option<ModalGraph>,
    /// Was the interaction successful?
    pub success: bool,
}

impl Interaction {
    pub fn new(action: Graph) -> Self {
        Self {
            action,
            before: None,
            after: None,
            success: false,
        }
    }

    pub fn successful(mut self) -> Self {
        self.success = true;
        self
    }
}

// ============================================================================
// Grounded Graph Trait
// ============================================================================

/// A graph with symbol-referent bindings
pub trait GroundedGraph: Send + Sync + Debug {
    /// Connect graph node to perceptual representation
    fn ground_to_perception(&mut self, node: NodeId, modality: Modality) -> Option<GroundingId>;

    /// Bind symbol to external referent
    fn bind_referent(&mut self, node: NodeId, referent: Referent) -> GroundingId;

    /// Get all groundings for a symbol
    fn groundings(&self, node: NodeId) -> Vec<&Grounding>;

    /// Get grounding by ID
    fn get_grounding(&self, id: GroundingId) -> Option<&Grounding>;

    /// Verify grounding against perception
    fn verify_grounding(&self, grounding: &Grounding, perception: &ModalGraph) -> f32;

    /// Simulate consequence of action
    fn simulate_consequence(&self, action: &Graph) -> Graph;

    /// Total number of groundings
    fn grounding_count(&self) -> usize;
}

// ============================================================================
// Sensor Trait
// ============================================================================

/// Interface for perception
pub trait Sensor: Send + Sync + Debug {
    /// What modality does this sensor perceive?
    fn modality(&self) -> Modality;

    /// Get current perception
    fn perceive(&mut self) -> ModalGraph;

    /// Direct attention to a region
    fn attend(&mut self, region: &Graph);

    /// Is the sensor active?
    fn is_active(&self) -> bool;
}

// ============================================================================
// Actuator Trait
// ============================================================================

/// Interface for action
pub trait Actuator: Send + Sync + Debug {
    /// Execute an action
    fn execute(&mut self, action: &Graph) -> Result<(), ActionError>;

    /// Can this action be executed?
    fn can_execute(&self, action: &Graph) -> bool;

    /// Get the action domain
    fn domain(&self) -> &str;
}

// ============================================================================
// World Interface
// ============================================================================

/// Interface to the world (sensors + actuators)
#[derive(Debug, Default)]
pub struct WorldInterface {
    /// Available sensors
    sensors: Vec<Box<dyn Sensor>>,
    /// Available actuators
    actuators: Vec<Box<dyn Actuator>>,
}

impl WorldInterface {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sensor
    pub fn add_sensor(&mut self, sensor: Box<dyn Sensor>) {
        self.sensors.push(sensor);
    }

    /// Add an actuator
    pub fn add_actuator(&mut self, actuator: Box<dyn Actuator>) {
        self.actuators.push(actuator);
    }

    /// Get sensors for a modality
    pub fn sensors_for(&self, modality: Modality) -> Vec<&dyn Sensor> {
        self.sensors
            .iter()
            .filter(|s| s.modality() == modality)
            .map(|s| s.as_ref())
            .collect()
    }

    /// Perceive from all sensors
    pub fn perceive_all(&mut self) -> Vec<ModalGraph> {
        self.sensors.iter_mut().map(|s| s.perceive()).collect()
    }

    /// Execute action on suitable actuator
    pub fn execute(&mut self, action: &Graph) -> Result<(), ActionError> {
        for actuator in &mut self.actuators {
            if actuator.can_execute(action) {
                return actuator.execute(action);
            }
        }
        Err(ActionError::CannotExecute(
            "No suitable actuator".to_string(),
        ))
    }

    /// Number of sensors
    pub fn sensor_count(&self) -> usize {
        self.sensors.len()
    }

    /// Number of actuators
    pub fn actuator_count(&self) -> usize {
        self.actuators.len()
    }
}

// ============================================================================
// Embodied Agent Trait
// ============================================================================

/// An agent with perception-action capabilities
pub trait EmbodiedAgent: Send + Sync + Debug {
    /// Full perception-action cycle
    fn sense_think_act(&mut self, world: &mut WorldInterface) -> Graph;

    /// Update internal model from perception
    fn update_from_perception(&mut self, perception: &ModalGraph);

    /// Ground new symbols through interaction
    fn learn_grounding(&mut self, symbol: NodeId, interactions: &[Interaction]) -> Grounding;

    /// Get the agent's internal representation
    fn internal_model(&self) -> &Graph;

    /// Get attention focus
    fn attention(&self) -> Option<&Graph>;
}

// ============================================================================
// Simple Implementations
// ============================================================================

/// Simple grounded graph implementation
#[derive(Debug, Default)]
pub struct SimpleGroundedGraph {
    groundings: Vec<Grounding>,
    next_id: GroundingId,
}

impl SimpleGroundedGraph {
    pub fn new() -> Self {
        Self::default()
    }

    fn clone_graph(graph: &Graph) -> Graph {
        let text = graph.to_text();
        DagNN::from_text(&text).unwrap_or_else(|_| DagNN::new())
    }
}

impl GroundedGraph for SimpleGroundedGraph {
    fn ground_to_perception(&mut self, node: NodeId, modality: Modality) -> Option<GroundingId> {
        // Create a simple perceptual grounding
        let modal_graph = ModalGraph::new(DagNN::new(), modality);
        let grounding = Grounding::new(
            node,
            Referent::Perceptual(modal_graph),
            0.5,
            GroundingSource::Inferred,
        );
        let id = self.next_id;
        self.next_id += 1;
        self.groundings.push(grounding);
        Some(id)
    }

    fn bind_referent(&mut self, node: NodeId, referent: Referent) -> GroundingId {
        let grounding = Grounding::supervised(node, referent);
        let id = self.next_id;
        self.next_id += 1;
        self.groundings.push(grounding);
        id
    }

    fn groundings(&self, node: NodeId) -> Vec<&Grounding> {
        self.groundings
            .iter()
            .filter(|g| g.symbol == node)
            .collect()
    }

    fn get_grounding(&self, id: GroundingId) -> Option<&Grounding> {
        self.groundings.get(id as usize)
    }

    fn verify_grounding(&self, grounding: &Grounding, _perception: &ModalGraph) -> f32 {
        // Simplified: return current confidence
        grounding.confidence
    }

    fn simulate_consequence(&self, action: &Graph) -> Graph {
        // Simplified: return action unchanged
        Self::clone_graph(action)
    }

    fn grounding_count(&self) -> usize {
        self.groundings.len()
    }
}

/// Simple sensor for testing
#[derive(Debug)]
pub struct SimpleSensor {
    modality: Modality,
    active: bool,
}

impl SimpleSensor {
    pub fn new(modality: Modality) -> Self {
        Self {
            modality,
            active: true,
        }
    }
}

impl Sensor for SimpleSensor {
    fn modality(&self) -> Modality {
        self.modality
    }

    fn perceive(&mut self) -> ModalGraph {
        ModalGraph::new(DagNN::new(), self.modality)
    }

    fn attend(&mut self, _region: &Graph) {
        // Simplified: no attention mechanism
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

/// Simple actuator for testing
#[derive(Debug)]
pub struct SimpleActuator {
    domain: String,
}

impl SimpleActuator {
    pub fn new(domain: &str) -> Self {
        Self {
            domain: domain.to_string(),
        }
    }
}

impl Actuator for SimpleActuator {
    fn execute(&mut self, _action: &Graph) -> Result<(), ActionError> {
        // Simplified: always succeed
        Ok(())
    }

    fn can_execute(&self, _action: &Graph) -> bool {
        true
    }

    fn domain(&self) -> &str {
        &self.domain
    }
}

/// Simple embodied agent
#[derive(Debug)]
pub struct SimpleEmbodiedAgent {
    model: Graph,
    attention: Option<Graph>,
}

impl SimpleEmbodiedAgent {
    pub fn new() -> Self {
        Self {
            model: DagNN::new(),
            attention: None,
        }
    }

    fn clone_graph(graph: &Graph) -> Graph {
        let text = graph.to_text();
        DagNN::from_text(&text).unwrap_or_else(|_| DagNN::new())
    }
}

impl Default for SimpleEmbodiedAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbodiedAgent for SimpleEmbodiedAgent {
    fn sense_think_act(&mut self, world: &mut WorldInterface) -> Graph {
        // 1. Sense
        let perceptions = world.perceive_all();

        // 2. Think (update model)
        for perception in &perceptions {
            self.update_from_perception(perception);
        }

        // 3. Act (return action representation)
        Self::clone_graph(&self.model)
    }

    fn update_from_perception(&mut self, _perception: &ModalGraph) {
        // Simplified: no model update
    }

    fn learn_grounding(&mut self, symbol: NodeId, _interactions: &[Interaction]) -> Grounding {
        // Simplified: create embodied grounding
        Grounding::new(
            symbol,
            Referent::Conceptual(DagNN::new()),
            0.5,
            GroundingSource::Embodied,
        )
    }

    fn internal_model(&self) -> &Graph {
        &self.model
    }

    fn attention(&self) -> Option<&Graph> {
        self.attention.as_ref()
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a default grounded graph
pub fn create_default_grounded_graph() -> SimpleGroundedGraph {
    SimpleGroundedGraph::new()
}

/// Create a visual sensor
pub fn create_visual_sensor() -> SimpleSensor {
    SimpleSensor::new(Modality::Visual)
}

/// Create a linguistic sensor
pub fn create_linguistic_sensor() -> SimpleSensor {
    SimpleSensor::new(Modality::Linguistic)
}

/// Create a simple embodied agent
pub fn create_default_embodied_agent() -> SimpleEmbodiedAgent {
    SimpleEmbodiedAgent::new()
}

// ============================================================================
// Learnable Grounding
// ============================================================================

/// Learnable grounding with trainable binding thresholds
///
/// This module learns to ground symbols to referents by adjusting
/// binding thresholds and exploration behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableGrounding {
    /// Threshold for grounding confidence
    pub grounding_threshold: LearnableParam,
    /// Weight for perception-based grounding
    pub perception_weight: LearnableParam,
    /// Weight for action-based grounding
    pub action_weight: LearnableParam,
    /// Exploration bonus for novel referents
    pub exploration_bonus: LearnableParam,
    /// Co-occurrence learning rate
    pub cooccurrence_rate: LearnableParam,
}

impl LearnableGrounding {
    /// Create a new learnable grounding module
    pub fn new() -> Self {
        Self {
            grounding_threshold: LearnableParam::new(0.5),
            perception_weight: LearnableParam::new(0.6),
            action_weight: LearnableParam::new(0.4),
            exploration_bonus: LearnableParam::new(0.1),
            cooccurrence_rate: LearnableParam::new(0.01),
        }
    }

    /// Check if a grounding is confident enough
    pub fn is_grounded(&self, confidence: f32) -> bool {
        confidence >= self.grounding_threshold.value
    }

    /// Compute weighted grounding score from perception and action
    pub fn grounding_score(&self, perception_confidence: f32, action_confidence: f32) -> f32 {
        let pw = self.perception_weight.value.max(0.0);
        let aw = self.action_weight.value.max(0.0);
        let total = pw + aw;
        if total > 0.0 {
            (pw * perception_confidence + aw * action_confidence) / total
        } else {
            (perception_confidence + action_confidence) / 2.0
        }
    }

    /// Compute exploration bonus for a novel referent
    pub fn novelty_bonus(&self, novelty: f32) -> f32 {
        self.exploration_bonus.value.max(0.0) * novelty.clamp(0.0, 1.0)
    }

    /// Update co-occurrence strength
    pub fn update_cooccurrence(&self, current_strength: f32, observed: bool) -> f32 {
        let rate = self.cooccurrence_rate.value.clamp(0.0, 1.0);
        let target = if observed { 1.0 } else { 0.0 };
        current_strength + rate * (target - current_strength)
    }
}

impl Default for LearnableGrounding {
    fn default() -> Self {
        Self::new()
    }
}

impl Learnable for LearnableGrounding {
    fn zero_grad(&mut self) {
        self.grounding_threshold.zero_grad();
        self.perception_weight.zero_grad();
        self.action_weight.zero_grad();
        self.exploration_bonus.zero_grad();
        self.cooccurrence_rate.zero_grad();
    }

    fn step(&mut self, lr: f32) {
        self.grounding_threshold.step(lr);
        self.perception_weight.step(lr);
        self.action_weight.step(lr);
        self.exploration_bonus.step(lr);
        self.cooccurrence_rate.step(lr);

        // Ensure valid ranges
        self.grounding_threshold.value = self.grounding_threshold.value.clamp(0.0, 1.0);
        self.cooccurrence_rate.value = self.cooccurrence_rate.value.clamp(0.0, 1.0);
    }

    fn num_parameters(&self) -> usize {
        5
    }

    fn has_gradients(&self) -> bool {
        self.grounding_threshold.grad != 0.0
            || self.perception_weight.grad != 0.0
            || self.action_weight.grad != 0.0
            || self.exploration_bonus.grad != 0.0
            || self.cooccurrence_rate.grad != 0.0
    }

    fn gradient_norm(&self) -> f32 {
        (self.grounding_threshold.grad.powi(2)
            + self.perception_weight.grad.powi(2)
            + self.action_weight.grad.powi(2)
            + self.exploration_bonus.grad.powi(2)
            + self.cooccurrence_rate.grad.powi(2))
        .sqrt()
    }
}

impl Persistable for LearnableGrounding {
    fn persist_type_id() -> &'static str {
        "LearnableGrounding"
    }

    fn persist_version() -> u32 {
        1
    }

    fn validate(&self) -> Result<(), PersistenceError> {
        // Validate grounding threshold is in valid range
        if self.grounding_threshold.value < 0.0 || self.grounding_threshold.value > 1.0 {
            return Err(PersistenceError::ValidationFailed(
                "Grounding threshold must be between 0 and 1".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Brain-Aware Grounding
// ============================================================================

/// Brain-aware grounding that uses domain brains for embodied interaction
pub struct BrainAwareGrounding {
    /// The cognitive-brain bridge for domain routing
    pub bridge: DefaultCognitiveBridge,
}

impl Debug for BrainAwareGrounding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrainAwareGrounding")
            .field("available_domains", &self.bridge.available_domains())
            .finish()
    }
}

impl BrainAwareGrounding {
    /// Create a new brain-aware grounding
    pub fn new() -> Self {
        Self {
            bridge: DefaultCognitiveBridge::new(),
        }
    }

    /// Register a domain brain
    pub fn register_brain(&mut self, brain: Box<dyn DomainBrain>) {
        self.bridge.register(brain);
    }

    /// Check if a domain brain can help with grounding
    pub fn can_ground(&self, percept_text: &str) -> bool {
        self.bridge.route_to_multiple_brains(percept_text).success
    }

    /// Get domains relevant to a percept
    pub fn domains_for_percept(&self, percept_text: &str) -> Vec<String> {
        self.bridge
            .route_to_multiple_brains(percept_text)
            .domains()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get available domains
    pub fn available_domains(&self) -> Vec<String> {
        self.bridge.available_domains()
    }
}

impl Default for BrainAwareGrounding {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveBrainBridge for BrainAwareGrounding {
    fn get_registry(&self) -> &BrainRegistry {
        self.bridge.get_registry()
    }

    fn get_registry_mut(&mut self) -> &mut BrainRegistry {
        self.bridge.get_registry_mut()
    }
}

/// Factory function to create brain-aware grounding
pub fn create_brain_aware_grounding() -> BrainAwareGrounding {
    BrainAwareGrounding::new()
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
    fn test_external_ref() {
        let ext = ExternalRef::new("database", "users/123");
        assert_eq!(ext.system_type, "database");
        assert_eq!(ext.identifier, "users/123");
    }

    #[test]
    fn test_procedure_spec() {
        let proc =
            ProcedureSpec::new("make_coffee").with_steps(vec!["grind beans", "add water", "brew"]);
        assert_eq!(proc.name, "make_coffee");
        assert_eq!(proc.steps.len(), 3);
    }

    #[test]
    fn test_grounding_source() {
        assert_ne!(GroundingSource::Supervised, GroundingSource::Inferred);
        assert_eq!(GroundingSource::Embodied, GroundingSource::Embodied);
    }

    #[test]
    fn test_grounding_creation() {
        let referent = Referent::Conceptual(make_graph("cat"));
        let grounding = Grounding::new(1, referent, 0.8, GroundingSource::Supervised);

        assert_eq!(grounding.symbol, 1);
        assert_eq!(grounding.confidence, 0.8);
        assert_eq!(grounding.source, GroundingSource::Supervised);
    }

    #[test]
    fn test_grounding_confidence_update() {
        let referent = Referent::Conceptual(make_graph("test"));
        let mut grounding = Grounding::new(1, referent, 0.5, GroundingSource::Inferred);

        grounding.update_confidence(true);
        assert!(grounding.confidence > 0.5);

        let referent2 = Referent::Conceptual(make_graph("test2"));
        let mut grounding2 = Grounding::new(2, referent2, 0.5, GroundingSource::Inferred);
        grounding2.update_confidence(false);
        assert!(grounding2.confidence < 0.5);
    }

    #[test]
    fn test_interaction() {
        let action = make_graph("pick up cup");
        let interaction = Interaction::new(action).successful();
        assert!(interaction.success);
    }

    #[test]
    fn test_simple_grounded_graph() {
        let mut gg = SimpleGroundedGraph::new();

        let id1 = gg.ground_to_perception(1, Modality::Visual);
        assert!(id1.is_some());

        let referent = Referent::External(ExternalRef::new("api", "weather/today"));
        let id2 = gg.bind_referent(2, referent);

        assert_eq!(gg.grounding_count(), 2);

        let groundings = gg.groundings(1);
        assert_eq!(groundings.len(), 1);

        assert!(gg.get_grounding(id2).is_some());
    }

    #[test]
    fn test_simple_sensor() {
        let mut sensor = SimpleSensor::new(Modality::Visual);
        assert_eq!(sensor.modality(), Modality::Visual);
        assert!(sensor.is_active());

        let perception = sensor.perceive();
        assert_eq!(perception.modality, Modality::Visual);
    }

    #[test]
    fn test_simple_actuator() {
        let mut actuator = SimpleActuator::new("manipulation");
        assert_eq!(actuator.domain(), "manipulation");

        let action = make_graph("move arm");
        assert!(actuator.can_execute(&action));
        assert!(actuator.execute(&action).is_ok());
    }

    #[test]
    fn test_world_interface() {
        let mut world = WorldInterface::new();
        world.add_sensor(Box::new(SimpleSensor::new(Modality::Visual)));
        world.add_sensor(Box::new(SimpleSensor::new(Modality::Linguistic)));
        world.add_actuator(Box::new(SimpleActuator::new("motor")));

        assert_eq!(world.sensor_count(), 2);
        assert_eq!(world.actuator_count(), 1);

        let visual_sensors = world.sensors_for(Modality::Visual);
        assert_eq!(visual_sensors.len(), 1);

        let perceptions = world.perceive_all();
        assert_eq!(perceptions.len(), 2);

        let action = make_graph("action");
        assert!(world.execute(&action).is_ok());
    }

    #[test]
    fn test_simple_embodied_agent() {
        let mut agent = SimpleEmbodiedAgent::new();
        let mut world = WorldInterface::new();
        world.add_sensor(Box::new(SimpleSensor::new(Modality::Visual)));

        let result = agent.sense_think_act(&mut world);
        let _ = result.node_count(); // Just verify it returns
    }

    #[test]
    fn test_embodied_agent_grounding() {
        let mut agent = SimpleEmbodiedAgent::new();
        let interactions = vec![];

        let grounding = agent.learn_grounding(42, &interactions);
        assert_eq!(grounding.symbol, 42);
        assert_eq!(grounding.source, GroundingSource::Embodied);
    }

    #[test]
    fn test_verify_grounding() {
        let gg = SimpleGroundedGraph::new();
        let referent = Referent::Conceptual(make_graph("thing"));
        let grounding = Grounding::new(1, referent, 0.7, GroundingSource::Linguistic);

        let perception = ModalGraph::new(make_graph("perception"), Modality::Visual);
        let score = gg.verify_grounding(&grounding, &perception);
        assert_eq!(score, 0.7);
    }

    #[test]
    fn test_simulate_consequence() {
        let gg = SimpleGroundedGraph::new();
        let action = make_graph("action");
        let result = gg.simulate_consequence(&action);
        let _ = result.node_count(); // Just verify it returns
    }

    #[test]
    fn test_factory_functions() {
        let gg = create_default_grounded_graph();
        assert_eq!(gg.grounding_count(), 0);

        let vs = create_visual_sensor();
        assert_eq!(vs.modality(), Modality::Visual);

        let ls = create_linguistic_sensor();
        assert_eq!(ls.modality(), Modality::Linguistic);

        let agent = create_default_embodied_agent();
        assert!(agent.attention().is_none());
    }
}
