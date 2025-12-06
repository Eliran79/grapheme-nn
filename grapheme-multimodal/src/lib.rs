//! # grapheme-multimodal
//!
//! Multi-modal fusion for GRAPHEME neural network.
//!
//! This crate enables unified representation across modalities:
//! - **Visual**: Images, video frames, spatial scenes
//! - **Auditory**: Speech, sounds, music
//! - **Linguistic**: Text, language, semantics
//! - **Tactile**: Touch, texture, pressure
//! - **Proprioceptive**: Body position, movement
//! - **Action**: Motor commands, behavior
//!
//! Key features:
//! - Modality-tagged graphs (`ModalGraph`)
//! - Cross-modal binding (word ↔ image region)
//! - Unified multi-modal events
//! - Modality translation (cross-modal inference)

use grapheme_core::{DagNN, Learnable, LearnableParam};
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;

// ============================================================================
// Type Aliases
// ============================================================================

/// Graph type for modal data
pub type Graph = DagNN;

/// Node identifier
pub type NodeId = NodeIndex;

/// Timestamp in milliseconds
pub type Timestamp = u64;

// ============================================================================
// Error Types
// ============================================================================

/// Errors in multi-modal operations
#[derive(Error, Debug)]
pub enum MultiModalError {
    #[error("Modality not found: {0:?}")]
    ModalityNotFound(Modality),
    #[error("Binding failed: {0}")]
    BindingFailed(String),
    #[error("Translation not supported: {0:?} -> {1:?}")]
    TranslationNotSupported(Modality, Modality),
    #[error("Fusion failed: {0}")]
    FusionFailed(String),
    #[error("Empty input")]
    EmptyInput,
}

/// Result type for multi-modal operations
pub type MultiModalResult<T> = Result<T, MultiModalError>;

// ============================================================================
// Modality Enum
// ============================================================================

/// Sensory/motor modality type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Visual input (images, video, spatial)
    Visual,
    /// Auditory input (speech, sounds, music)
    Auditory,
    /// Linguistic input (text, language)
    Linguistic,
    /// Tactile input (touch, texture, pressure)
    Tactile,
    /// Proprioceptive input (body position, movement)
    Proprioceptive,
    /// Action output (motor commands)
    Action,
    /// Abstract/symbolic (not tied to any sense)
    Abstract,
}

impl Modality {
    /// Check if this is an input modality
    pub fn is_input(&self) -> bool {
        !matches!(self, Modality::Action)
    }

    /// Check if this is an output modality
    pub fn is_output(&self) -> bool {
        matches!(self, Modality::Action)
    }

    /// Get all modalities
    pub fn all() -> Vec<Modality> {
        vec![
            Modality::Visual,
            Modality::Auditory,
            Modality::Linguistic,
            Modality::Tactile,
            Modality::Proprioceptive,
            Modality::Action,
            Modality::Abstract,
        ]
    }
}

// ============================================================================
// Modal Graph
// ============================================================================

/// A graph tagged with its modality
#[derive(Debug)]
pub struct ModalGraph {
    /// The graph content
    pub graph: Graph,
    /// Which modality this represents
    pub modality: Modality,
    /// Optional timestamp (for temporal alignment)
    pub timestamp: Option<Timestamp>,
    /// Spatial region (for visual modality)
    pub region: Option<SpatialRegion>,
    /// Confidence in this modality data
    pub confidence: f32,
}

impl ModalGraph {
    /// Create a new modal graph
    pub fn new(graph: Graph, modality: Modality) -> Self {
        Self {
            graph,
            modality,
            timestamp: None,
            region: None,
            confidence: 1.0,
        }
    }

    /// Create with timestamp
    pub fn with_timestamp(mut self, timestamp: Timestamp) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Create with spatial region
    pub fn with_region(mut self, region: SpatialRegion) -> Self {
        self.region = Some(region);
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Spatial region for visual modality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialRegion {
    /// X coordinate (0.0 to 1.0)
    pub x: f32,
    /// Y coordinate (0.0 to 1.0)
    pub y: f32,
    /// Width (0.0 to 1.0)
    pub width: f32,
    /// Height (0.0 to 1.0)
    pub height: f32,
}

impl SpatialRegion {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            width: width.clamp(0.0, 1.0),
            height: height.clamp(0.0, 1.0),
        }
    }

    /// Full frame region
    pub fn full() -> Self {
        Self { x: 0.0, y: 0.0, width: 1.0, height: 1.0 }
    }

    /// Check if regions overlap
    pub fn overlaps(&self, other: &SpatialRegion) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
}

// ============================================================================
// Cross-Modal Binding
// ============================================================================

/// A binding between nodes in different modalities
#[derive(Debug, Clone)]
pub struct CrossModalBinding {
    /// Source modality and node
    pub source: (Modality, NodeId),
    /// Target modality and node
    pub target: (Modality, NodeId),
    /// Binding strength (0.0 to 1.0)
    pub strength: f32,
    /// Binding type
    pub binding_type: BindingType,
}

impl CrossModalBinding {
    /// Create a new binding
    pub fn new(source: (Modality, NodeId), target: (Modality, NodeId), strength: f32) -> Self {
        Self {
            source,
            target,
            strength: strength.clamp(0.0, 1.0),
            binding_type: BindingType::Reference,
        }
    }

    /// Set binding type
    pub fn with_type(mut self, binding_type: BindingType) -> Self {
        self.binding_type = binding_type;
        self
    }
}

/// Type of cross-modal binding
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingType {
    /// Reference: one points to the other (e.g., word refers to object)
    Reference,
    /// Synchrony: they occur together in time
    Synchrony,
    /// Causation: one causes the other
    Causation,
    /// Identity: same concept in different modalities
    Identity,
}

// ============================================================================
// Multi-Modal Event
// ============================================================================

/// A multi-modal event combining multiple modalities
#[derive(Debug)]
pub struct MultiModalEvent {
    /// Component graphs from each modality
    pub components: Vec<ModalGraph>,
    /// Bindings between modality nodes
    pub bindings: Vec<CrossModalBinding>,
    /// Event timestamp
    pub timestamp: Option<Timestamp>,
    /// Event ID
    pub id: u64,
}

impl MultiModalEvent {
    /// Create a new multi-modal event
    pub fn new(id: u64) -> Self {
        Self {
            components: Vec::new(),
            bindings: Vec::new(),
            timestamp: None,
            id,
        }
    }

    /// Add a component
    pub fn add_component(&mut self, component: ModalGraph) {
        self.components.push(component);
    }

    /// Add a binding
    pub fn add_binding(&mut self, binding: CrossModalBinding) {
        self.bindings.push(binding);
    }

    /// Get component by modality
    pub fn get_component(&self, modality: Modality) -> Option<&ModalGraph> {
        self.components.iter().find(|c| c.modality == modality)
    }

    /// Check if event has a specific modality
    pub fn has_modality(&self, modality: Modality) -> bool {
        self.components.iter().any(|c| c.modality == modality)
    }

    /// Get all modalities present
    pub fn modalities(&self) -> Vec<Modality> {
        self.components.iter().map(|c| c.modality).collect()
    }
}

// ============================================================================
// Multi-Modal Graph Trait
// ============================================================================

/// Trait for multi-modal graph processing
pub trait MultiModalGraph: Send + Sync + Debug {
    /// Fuse multiple modality graphs into unified representation
    fn fuse(
        &mut self,
        visual: Option<ModalGraph>,
        auditory: Option<ModalGraph>,
        linguistic: Option<ModalGraph>,
        tactile: Option<ModalGraph>,
    ) -> MultiModalResult<Graph>;

    /// Translate content from one modality to another
    fn translate_modality(&self, source: &ModalGraph, target_modality: Modality)
        -> MultiModalResult<ModalGraph>;

    /// Bind representations across modalities
    fn cross_modal_bind(&mut self, event: MultiModalEvent) -> MultiModalResult<Graph>;

    /// Attend to specific modality (extract modality-specific subgraph)
    fn attend(&self, unified: &Graph, modality: Modality) -> MultiModalResult<Graph>;

    /// Extract modality-specific subgraph
    fn extract(&self, unified: &Graph, modality: Modality) -> Option<ModalGraph>;

    /// Get attention weights for each modality
    fn modality_attention(&self, unified: &Graph) -> Vec<(Modality, f32)>;
}

// ============================================================================
// Simple Implementation
// ============================================================================

/// Simple multi-modal graph implementation
#[derive(Debug, Default)]
pub struct SimpleMultiModal {
    /// Current modality focus
    #[allow(dead_code)]
    focus: Option<Modality>,
    /// Fusion buffer
    fused_graphs: Vec<ModalGraph>,
}

impl SimpleMultiModal {
    pub fn new() -> Self {
        Self::default()
    }

    fn clone_graph(graph: &Graph) -> Graph {
        let text = graph.to_text();
        DagNN::from_text(&text).unwrap_or_else(|_| DagNN::new())
    }
}

impl MultiModalGraph for SimpleMultiModal {
    fn fuse(
        &mut self,
        visual: Option<ModalGraph>,
        auditory: Option<ModalGraph>,
        linguistic: Option<ModalGraph>,
        tactile: Option<ModalGraph>,
    ) -> MultiModalResult<Graph> {
        self.fused_graphs.clear();

        let mut has_input = false;

        if let Some(v) = visual {
            has_input = true;
            self.fused_graphs.push(v);
        }
        if let Some(a) = auditory {
            has_input = true;
            self.fused_graphs.push(a);
        }
        if let Some(l) = linguistic {
            has_input = true;
            self.fused_graphs.push(l);
        }
        if let Some(t) = tactile {
            has_input = true;
            self.fused_graphs.push(t);
        }

        if !has_input {
            return Err(MultiModalError::EmptyInput);
        }

        // Simplified: return first graph (real impl would merge)
        Ok(Self::clone_graph(&self.fused_graphs[0].graph))
    }

    fn translate_modality(&self, source: &ModalGraph, target_modality: Modality)
        -> MultiModalResult<ModalGraph> {
        // Simplified: copy graph with new modality tag
        // Real implementation would learn cross-modal mappings
        Ok(ModalGraph::new(
            Self::clone_graph(&source.graph),
            target_modality,
        ))
    }

    fn cross_modal_bind(&mut self, event: MultiModalEvent) -> MultiModalResult<Graph> {
        if event.components.is_empty() {
            return Err(MultiModalError::EmptyInput);
        }

        // Simplified: return first component's graph
        // Real implementation would create cross-modal edges
        Ok(Self::clone_graph(&event.components[0].graph))
    }

    fn attend(&self, unified: &Graph, _modality: Modality) -> MultiModalResult<Graph> {
        // Simplified: return whole graph
        // Real implementation would filter by modality
        Ok(Self::clone_graph(unified))
    }

    fn extract(&self, unified: &Graph, modality: Modality) -> Option<ModalGraph> {
        // Simplified: wrap whole graph
        Some(ModalGraph::new(Self::clone_graph(unified), modality))
    }

    fn modality_attention(&self, _unified: &Graph) -> Vec<(Modality, f32)> {
        // Simplified: equal attention to all modalities
        Modality::all()
            .into_iter()
            .map(|m| (m, 1.0 / 7.0))
            .collect()
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a default multi-modal processor
pub fn create_default_multimodal() -> SimpleMultiModal {
    SimpleMultiModal::new()
}

// ============================================================================
// Learnable Multimodal
// ============================================================================

/// Learnable multimodal fusion with trainable modality weights
///
/// This module learns to weight different modalities and adjust
/// binding strength for cross-modal associations.
#[derive(Debug, Clone)]
pub struct LearnableMultimodal {
    /// Weight for visual modality
    pub visual_weight: LearnableParam,
    /// Weight for auditory modality
    pub auditory_weight: LearnableParam,
    /// Weight for linguistic modality
    pub linguistic_weight: LearnableParam,
    /// Weight for tactile modality
    pub tactile_weight: LearnableParam,
    /// Binding strength for cross-modal associations
    pub binding_strength: LearnableParam,
    /// Temperature for fusion attention
    pub fusion_temperature: LearnableParam,
}

impl LearnableMultimodal {
    /// Create a new learnable multimodal module
    pub fn new() -> Self {
        Self {
            visual_weight: LearnableParam::new(0.3),
            auditory_weight: LearnableParam::new(0.2),
            linguistic_weight: LearnableParam::new(0.4),
            tactile_weight: LearnableParam::new(0.1),
            binding_strength: LearnableParam::new(0.5),
            fusion_temperature: LearnableParam::new(1.0),
        }
    }

    /// Get normalized modality weights
    pub fn normalized_weights(&self) -> [f32; 4] {
        let weights = [
            self.visual_weight.value.max(0.0),
            self.auditory_weight.value.max(0.0),
            self.linguistic_weight.value.max(0.0),
            self.tactile_weight.value.max(0.0),
        ];
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            [weights[0] / sum, weights[1] / sum, weights[2] / sum, weights[3] / sum]
        } else {
            [0.25, 0.25, 0.25, 0.25]
        }
    }

    /// Compute weighted fusion of modality values
    pub fn weighted_fusion(&self, visual: f32, auditory: f32, linguistic: f32, tactile: f32) -> f32 {
        let w = self.normalized_weights();
        w[0] * visual + w[1] * auditory + w[2] * linguistic + w[3] * tactile
    }

    /// Compute binding score between two elements
    pub fn binding_score(&self, similarity: f32) -> f32 {
        let strength = self.binding_strength.value.clamp(0.0, 1.0);
        strength * similarity
    }

    /// Compute attention weights with temperature scaling
    pub fn attention_weights(&self, scores: &[f32]) -> Vec<f32> {
        let temp = self.fusion_temperature.value.max(0.01);
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter()
            .map(|&s| ((s - max_score) / temp).exp())
            .collect();
        let sum: f32 = exp_scores.iter().sum();
        if sum > 0.0 {
            exp_scores.iter().map(|&e| e / sum).collect()
        } else {
            vec![1.0 / scores.len() as f32; scores.len()]
        }
    }
}

impl Default for LearnableMultimodal {
    fn default() -> Self {
        Self::new()
    }
}

impl Learnable for LearnableMultimodal {
    fn zero_grad(&mut self) {
        self.visual_weight.zero_grad();
        self.auditory_weight.zero_grad();
        self.linguistic_weight.zero_grad();
        self.tactile_weight.zero_grad();
        self.binding_strength.zero_grad();
        self.fusion_temperature.zero_grad();
    }

    fn step(&mut self, lr: f32) {
        self.visual_weight.step(lr);
        self.auditory_weight.step(lr);
        self.linguistic_weight.step(lr);
        self.tactile_weight.step(lr);
        self.binding_strength.step(lr);
        self.fusion_temperature.step(lr);

        // Ensure valid ranges
        self.fusion_temperature.value = self.fusion_temperature.value.max(0.01);
    }

    fn num_parameters(&self) -> usize {
        6
    }

    fn has_gradients(&self) -> bool {
        self.visual_weight.grad != 0.0
            || self.auditory_weight.grad != 0.0
            || self.linguistic_weight.grad != 0.0
            || self.tactile_weight.grad != 0.0
            || self.binding_strength.grad != 0.0
            || self.fusion_temperature.grad != 0.0
    }

    fn gradient_norm(&self) -> f32 {
        (self.visual_weight.grad.powi(2)
            + self.auditory_weight.grad.powi(2)
            + self.linguistic_weight.grad.powi(2)
            + self.tactile_weight.grad.powi(2)
            + self.binding_strength.grad.powi(2)
            + self.fusion_temperature.grad.powi(2))
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
    fn test_modality_enum() {
        assert!(Modality::Visual.is_input());
        assert!(!Modality::Action.is_input());
        assert!(Modality::Action.is_output());
        assert_eq!(Modality::all().len(), 7);
    }

    #[test]
    fn test_modal_graph_creation() {
        let graph = make_graph("test content");
        let modal = ModalGraph::new(graph, Modality::Linguistic)
            .with_timestamp(1000)
            .with_confidence(0.9);

        assert_eq!(modal.modality, Modality::Linguistic);
        assert_eq!(modal.timestamp, Some(1000));
        assert!((modal.confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_spatial_region() {
        let r1 = SpatialRegion::new(0.0, 0.0, 0.5, 0.5);
        let r2 = SpatialRegion::new(0.25, 0.25, 0.5, 0.5);
        let r3 = SpatialRegion::new(0.7, 0.7, 0.2, 0.2);

        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3));
    }

    #[test]
    fn test_cross_modal_binding() {
        let binding = CrossModalBinding::new(
            (Modality::Linguistic, NodeIndex::new(0)),
            (Modality::Visual, NodeIndex::new(5)),
            0.95,
        ).with_type(BindingType::Reference);

        assert_eq!(binding.source.0, Modality::Linguistic);
        assert_eq!(binding.target.0, Modality::Visual);
        assert!(binding.strength > 0.9);
        assert_eq!(binding.binding_type, BindingType::Reference);
    }

    #[test]
    fn test_multimodal_event() {
        let mut event = MultiModalEvent::new(1);

        event.add_component(ModalGraph::new(make_graph("visual"), Modality::Visual));
        event.add_component(ModalGraph::new(make_graph("text"), Modality::Linguistic));

        assert!(event.has_modality(Modality::Visual));
        assert!(event.has_modality(Modality::Linguistic));
        assert!(!event.has_modality(Modality::Auditory));

        let modalities = event.modalities();
        assert_eq!(modalities.len(), 2);
    }

    #[test]
    fn test_simple_multimodal_fuse() {
        let mut mm = SimpleMultiModal::new();

        let visual = ModalGraph::new(make_graph("visual content"), Modality::Visual);
        let linguistic = ModalGraph::new(make_graph("text content"), Modality::Linguistic);

        let result = mm.fuse(Some(visual), None, Some(linguistic), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simple_multimodal_fuse_empty() {
        let mut mm = SimpleMultiModal::new();

        let result = mm.fuse(None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_multimodal_translate() {
        let mm = SimpleMultiModal::new();

        let source = ModalGraph::new(make_graph("source"), Modality::Linguistic);
        let result = mm.translate_modality(&source, Modality::Visual);

        assert!(result.is_ok());
        let translated = result.unwrap();
        assert_eq!(translated.modality, Modality::Visual);
    }

    #[test]
    fn test_simple_multimodal_bind() {
        let mut mm = SimpleMultiModal::new();

        let mut event = MultiModalEvent::new(1);
        event.add_component(ModalGraph::new(make_graph("content"), Modality::Visual));

        let result = mm.cross_modal_bind(event);
        assert!(result.is_ok());
    }

    #[test]
    fn test_modality_attention() {
        let mm = SimpleMultiModal::new();
        let graph = make_graph("test");

        let attention = mm.modality_attention(&graph);
        assert_eq!(attention.len(), 7);

        let total: f32 = attention.iter().map(|(_, w)| w).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_binding_types() {
        assert_ne!(BindingType::Reference, BindingType::Identity);
        assert_eq!(BindingType::Causation, BindingType::Causation);
    }
}
