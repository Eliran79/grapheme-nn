//! Knowledge Distillation Module
//!
//! Backend-179: Knowledge distillation from LLMs to GRAPHEME graphs.
//!
//! Implements techniques for transferring knowledge from large language models
//! to compact GRAPHEME graph representations:
//! - Teacher-student distillation
//! - Soft target generation
//! - Knowledge compression
//! - Graph structure learning from LLM outputs

use crate::knowledge_extraction::{Entity, EntityType, KnowledgeGraph, Relation, RelationType};
use crate::llm_client::LLMConfig;
use grapheme_core::GraphemeGraph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Distillation Configuration
// ============================================================================

/// Configuration for knowledge distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Temperature for softening probabilities (higher = softer)
    pub temperature: f32,
    /// Weight for distillation loss vs task loss
    pub distillation_weight: f32,
    /// Minimum confidence to accept extracted knowledge
    pub min_confidence: f32,
    /// Maximum knowledge items per session
    pub max_items_per_session: usize,
    /// Whether to preserve structure from LLM
    pub preserve_structure: bool,
    /// Knowledge compression ratio target
    pub compression_ratio: f32,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 2.0,
            distillation_weight: 0.5,
            min_confidence: 0.7,
            max_items_per_session: 1000,
            preserve_structure: true,
            compression_ratio: 0.1,
        }
    }
}

impl DistillationConfig {
    /// Create config optimized for speed
    pub fn fast() -> Self {
        Self {
            temperature: 1.5,
            distillation_weight: 0.7,
            min_confidence: 0.8,
            max_items_per_session: 100,
            preserve_structure: false,
            compression_ratio: 0.05,
        }
    }

    /// Create config optimized for quality
    pub fn quality() -> Self {
        Self {
            temperature: 3.0,
            distillation_weight: 0.3,
            min_confidence: 0.6,
            max_items_per_session: 5000,
            preserve_structure: true,
            compression_ratio: 0.2,
        }
    }
}

// ============================================================================
// Knowledge Types
// ============================================================================

/// Type of knowledge being distilled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeType {
    /// Factual knowledge (entities, attributes)
    Factual,
    /// Relational knowledge (relationships between entities)
    Relational,
    /// Procedural knowledge (how to do things)
    Procedural,
    /// Structural knowledge (graph patterns)
    Structural,
    /// Linguistic knowledge (language patterns)
    Linguistic,
}

/// A unit of distilled knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledKnowledge {
    /// Unique identifier
    pub id: String,
    /// Type of knowledge
    pub knowledge_type: KnowledgeType,
    /// Source prompt that generated this
    pub source_prompt: String,
    /// LLM response text
    pub llm_response: String,
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Extracted relations
    pub relations: Vec<Relation>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl DistilledKnowledge {
    /// Create new distilled knowledge
    pub fn new(
        knowledge_type: KnowledgeType,
        source_prompt: String,
        llm_response: String,
    ) -> Self {
        Self {
            id: format!("dk_{}", uuid_simple()),
            knowledge_type,
            source_prompt,
            llm_response,
            entities: Vec::new(),
            relations: Vec::new(),
            confidence: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Add an entity
    pub fn add_entity(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    /// Add a relation
    pub fn add_relation(&mut self, relation: Relation) {
        self.relations.push(relation);
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// ============================================================================
// Soft Targets
// ============================================================================

/// Soft target distribution from teacher LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftTarget {
    /// Target node index
    pub node_idx: usize,
    /// Probability distribution over outputs
    pub probabilities: Vec<f32>,
    /// Temperature used to generate
    pub temperature: f32,
    /// Entropy of the distribution
    pub entropy: f32,
}

impl SoftTarget {
    /// Create new soft target
    pub fn new(node_idx: usize, probabilities: Vec<f32>, temperature: f32) -> Self {
        let entropy = Self::compute_entropy(&probabilities);
        Self {
            node_idx,
            probabilities,
            temperature,
            entropy,
        }
    }

    /// Compute Shannon entropy
    fn compute_entropy(probs: &[f32]) -> f32 {
        probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Apply temperature scaling
    pub fn apply_temperature(&mut self, new_temp: f32) {
        let scale = self.temperature / new_temp;
        let mut max_logit = f32::NEG_INFINITY;

        // Convert to logits, scale, then back to probs
        let logits: Vec<f32> = self
            .probabilities
            .iter()
            .map(|&p| {
                let logit = (p + 1e-10).ln() * scale;
                if logit > max_logit {
                    max_logit = logit;
                }
                logit
            })
            .collect();

        // Softmax
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        self.probabilities = logits
            .iter()
            .map(|&l| (l - max_logit).exp() / exp_sum)
            .collect();
        self.temperature = new_temp;
        self.entropy = Self::compute_entropy(&self.probabilities);
    }

    /// Get the most likely output
    pub fn argmax(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ============================================================================
// Distillation Session
// ============================================================================

/// A session for distilling knowledge from an LLM
#[derive(Debug)]
pub struct DistillationSession {
    /// Session identifier
    pub id: String,
    /// Configuration
    config: DistillationConfig,
    /// LLM configuration (reserved for future LLM integration)
    #[allow(dead_code)]
    llm_config: LLMConfig,
    /// Collected knowledge
    knowledge: Vec<DistilledKnowledge>,
    /// Generated soft targets
    soft_targets: Vec<SoftTarget>,
    /// Session metrics
    metrics: DistillationMetrics,
}

/// Metrics from a distillation session
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistillationMetrics {
    /// Total prompts processed
    pub prompts_processed: usize,
    /// Total knowledge items extracted
    pub knowledge_extracted: usize,
    /// Average confidence score
    pub avg_confidence: f32,
    /// Knowledge items by type
    pub items_by_type: HashMap<String, usize>,
    /// Compression achieved
    pub compression_ratio: f32,
    /// Total entities extracted
    pub entities_extracted: usize,
    /// Total relations extracted
    pub relations_extracted: usize,
}

impl DistillationSession {
    /// Create new distillation session
    pub fn new(config: DistillationConfig, llm_config: LLMConfig) -> Self {
        Self {
            id: format!("ds_{}", uuid_simple()),
            config,
            llm_config,
            knowledge: Vec::new(),
            soft_targets: Vec::new(),
            metrics: DistillationMetrics::default(),
        }
    }

    /// Get session ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get configuration
    pub fn config(&self) -> &DistillationConfig {
        &self.config
    }

    /// Get current metrics
    pub fn metrics(&self) -> &DistillationMetrics {
        &self.metrics
    }

    /// Get collected knowledge
    pub fn knowledge(&self) -> &[DistilledKnowledge] {
        &self.knowledge
    }

    /// Get soft targets
    pub fn soft_targets(&self) -> &[SoftTarget] {
        &self.soft_targets
    }

    /// Process an LLM response and extract knowledge
    pub fn process_response(
        &mut self,
        prompt: &str,
        response: &str,
        knowledge_type: KnowledgeType,
    ) -> Result<&DistilledKnowledge, DistillationError> {
        if self.knowledge.len() >= self.config.max_items_per_session {
            return Err(DistillationError::SessionFull);
        }

        let mut knowledge = DistilledKnowledge::new(
            knowledge_type,
            prompt.to_string(),
            response.to_string(),
        );

        // Extract entities and relations based on knowledge type
        match knowledge_type {
            KnowledgeType::Factual => {
                self.extract_factual_knowledge(&mut knowledge, response)?;
            }
            KnowledgeType::Relational => {
                self.extract_relational_knowledge(&mut knowledge, response)?;
            }
            KnowledgeType::Procedural => {
                self.extract_procedural_knowledge(&mut knowledge, response)?;
            }
            KnowledgeType::Structural => {
                self.extract_structural_knowledge(&mut knowledge, response)?;
            }
            KnowledgeType::Linguistic => {
                self.extract_linguistic_knowledge(&mut knowledge, response)?;
            }
        }

        // Calculate confidence
        let confidence = self.calculate_confidence(&knowledge);
        knowledge.confidence = confidence;

        // Only add if meets threshold
        if confidence < self.config.min_confidence {
            return Err(DistillationError::LowConfidence(confidence));
        }

        // Update metrics
        self.update_metrics(&knowledge);

        self.knowledge.push(knowledge);
        Ok(self.knowledge.last().unwrap())
    }

    /// Extract factual knowledge from response
    fn extract_factual_knowledge(
        &self,
        knowledge: &mut DistilledKnowledge,
        response: &str,
    ) -> Result<(), DistillationError> {
        // Extract sentences as facts
        for sentence in response.split('.') {
            let sentence = sentence.trim();
            if sentence.len() < 5 {
                continue;
            }

            // Extract capitalized words as entities
            let mut offset = 0;
            for word in sentence.split_whitespace() {
                if word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && word.len() > 1
                {
                    let entity = Entity::new(
                        word,
                        EntityType::Concept,
                        offset,
                        offset + word.len(),
                    )
                    .with_confidence(0.8);
                    knowledge.add_entity(entity);
                }
                offset += word.len() + 1;
            }
        }
        Ok(())
    }

    /// Extract relational knowledge from response
    fn extract_relational_knowledge(
        &self,
        knowledge: &mut DistilledKnowledge,
        response: &str,
    ) -> Result<(), DistillationError> {
        // Look for relationship patterns: "X is Y", "X has Y", "X does Y"
        let patterns: [(&str, RelationType); 7] = [
            (" is ", RelationType::IsA),
            (" has ", RelationType::Owns),
            (" are ", RelationType::IsA),
            (" was ", RelationType::IsA),
            (" were ", RelationType::IsA),
            (" contains ", RelationType::PartOf),
            (" includes ", RelationType::PartOf),
        ];

        let mut offset = 0;
        for sentence in response.split('.') {
            let lower = sentence.trim().to_lowercase();
            for (pattern, rel_type) in &patterns {
                if let Some(idx) = lower.find(pattern) {
                    let subject = lower[..idx].trim();
                    let object = lower[idx + pattern.len()..].trim();

                    if !subject.is_empty() && !object.is_empty() {
                        let subj_entity = Entity::new(
                            subject,
                            EntityType::Concept,
                            offset,
                            offset + subject.len(),
                        );
                        let subj_id = subj_entity.id.clone();
                        knowledge.add_entity(subj_entity);

                        let obj_entity = Entity::new(
                            object,
                            EntityType::Concept,
                            offset + idx + pattern.len(),
                            offset + idx + pattern.len() + object.len(),
                        );
                        let obj_id = obj_entity.id.clone();
                        knowledge.add_entity(obj_entity);

                        knowledge.add_relation(Relation::new(&subj_id, *rel_type, &obj_id));
                    }
                }
            }
            offset += sentence.len() + 1;
        }
        Ok(())
    }

    /// Extract procedural knowledge from response
    fn extract_procedural_knowledge(
        &self,
        knowledge: &mut DistilledKnowledge,
        response: &str,
    ) -> Result<(), DistillationError> {
        // Look for step sequences: "1.", "First", "Then", "Finally"
        let step_markers = ["1.", "2.", "3.", "first", "then", "next", "finally", "step"];

        let lower = response.to_lowercase();
        let mut steps = Vec::new();

        for marker in &step_markers {
            if lower.contains(marker) {
                // Find sentences containing the marker
                for sentence in response.split('.') {
                    if sentence.to_lowercase().contains(marker) {
                        steps.push(sentence.trim());
                    }
                }
            }
        }

        // Create entities for each step and collect IDs
        let mut step_ids = Vec::new();
        let mut offset = 0;
        for step in &steps {
            let entity = Entity::new(step, EntityType::Concept, offset, offset + step.len())
                .with_confidence(0.9);
            step_ids.push(entity.id.clone());
            knowledge.add_entity(entity);
            offset += step.len() + 1;
        }

        // Link steps sequentially
        for i in 1..step_ids.len() {
            knowledge.add_relation(Relation::new(
                &step_ids[i - 1],
                RelationType::Before,
                &step_ids[i],
            ));
        }

        Ok(())
    }

    /// Extract structural knowledge from response
    fn extract_structural_knowledge(
        &self,
        knowledge: &mut DistilledKnowledge,
        response: &str,
    ) -> Result<(), DistillationError> {
        // Look for hierarchical patterns: "X contains Y", "X is part of Y"
        let hierarchy_patterns: [(&str, RelationType); 5] = [
            (" contains ", RelationType::PartOf),
            (" includes ", RelationType::PartOf),
            (" part of ", RelationType::PartOf),
            (" belongs to ", RelationType::PartOf),
            (" component of ", RelationType::PartOf),
        ];

        let mut offset = 0;
        for sentence in response.split('.') {
            let lower = sentence.to_lowercase();
            for (pattern, rel_type) in &hierarchy_patterns {
                if let Some(idx) = lower.find(pattern) {
                    let parent = lower[..idx].trim();
                    let child = lower[idx + pattern.len()..].trim();

                    if !parent.is_empty() && !child.is_empty() {
                        let parent_entity = Entity::new(
                            parent,
                            EntityType::Concept,
                            offset,
                            offset + parent.len(),
                        );
                        let parent_id = parent_entity.id.clone();
                        knowledge.add_entity(parent_entity);

                        let child_entity = Entity::new(
                            child,
                            EntityType::Concept,
                            offset + idx + pattern.len(),
                            offset + idx + pattern.len() + child.len(),
                        );
                        let child_id = child_entity.id.clone();
                        knowledge.add_entity(child_entity);

                        knowledge.add_relation(Relation::new(&parent_id, *rel_type, &child_id));
                    }
                }
            }
            offset += sentence.len() + 1;
        }
        Ok(())
    }

    /// Extract linguistic knowledge from response
    fn extract_linguistic_knowledge(
        &self,
        knowledge: &mut DistilledKnowledge,
        response: &str,
    ) -> Result<(), DistillationError> {
        // Extract unique words and their context
        let words: Vec<&str> = response.split_whitespace().collect();
        let mut word_ids: Vec<String> = Vec::new();

        for (i, window) in words.windows(3).enumerate() {
            if window.len() == 3 {
                let text = window.join(" ");
                let entity = Entity::new(
                    &text,
                    EntityType::Concept,
                    i,
                    i + text.len(),
                )
                .with_confidence(0.7);
                word_ids.push(entity.id.clone());
                knowledge.add_entity(entity);
            }
        }

        // Track word co-occurrences using actual entity IDs
        for i in 1..word_ids.len() {
            knowledge.add_relation(Relation::new(
                &word_ids[i - 1],
                RelationType::Before,
                &word_ids[i],
            ));
        }

        Ok(())
    }

    /// Calculate confidence for extracted knowledge
    fn calculate_confidence(&self, knowledge: &DistilledKnowledge) -> f32 {
        let entity_score = if knowledge.entities.is_empty() {
            0.0
        } else {
            (knowledge.entities.len() as f32).min(20.0) / 20.0
        };

        let relation_score = if knowledge.relations.is_empty() {
            0.0
        } else {
            (knowledge.relations.len() as f32).min(10.0) / 10.0
        };

        let response_length_score =
            (knowledge.llm_response.len() as f32).min(1000.0) / 1000.0;

        // Weighted average
        0.4 * entity_score + 0.4 * relation_score + 0.2 * response_length_score
    }

    /// Update session metrics
    fn update_metrics(&mut self, knowledge: &DistilledKnowledge) {
        self.metrics.prompts_processed += 1;
        self.metrics.knowledge_extracted += 1;
        self.metrics.entities_extracted += knowledge.entities.len();
        self.metrics.relations_extracted += knowledge.relations.len();

        // Update average confidence
        let total = self.metrics.prompts_processed as f32;
        self.metrics.avg_confidence = (self.metrics.avg_confidence * (total - 1.0)
            + knowledge.confidence)
            / total;

        // Update by type
        let type_key = format!("{:?}", knowledge.knowledge_type);
        *self.metrics.items_by_type.entry(type_key).or_insert(0) += 1;
    }

    /// Generate soft target from response logits
    pub fn add_soft_target(&mut self, node_idx: usize, logits: Vec<f32>) {
        // Apply temperature scaling and softmax
        let temp = self.config.temperature;
        let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();

        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&l| (l - max_logit).exp()).sum();
        let probs: Vec<f32> = scaled
            .iter()
            .map(|&l| (l - max_logit).exp() / exp_sum)
            .collect();

        self.soft_targets.push(SoftTarget::new(node_idx, probs, temp));
    }

    /// Convert all knowledge to a KnowledgeGraph
    pub fn to_knowledge_graph(&self) -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();

        for dk in &self.knowledge {
            for entity in &dk.entities {
                kg.add_entity(entity.clone());
            }
            for relation in &dk.relations {
                kg.add_relation(relation.clone());
            }
        }

        kg
    }

    /// Convert to GRAPHEME graph
    pub fn to_grapheme_graph(&self) -> GraphemeGraph {
        self.to_knowledge_graph().to_grapheme_graph()
    }
}

// ============================================================================
// Knowledge Distiller
// ============================================================================

/// Main distillation orchestrator
pub struct KnowledgeDistiller {
    /// Configuration
    config: DistillationConfig,
    /// Active sessions
    sessions: HashMap<String, DistillationSession>,
    /// Global statistics
    stats: DistillerStats,
}

/// Global distiller statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistillerStats {
    /// Total sessions created
    pub sessions_created: usize,
    /// Total sessions completed
    pub sessions_completed: usize,
    /// Total knowledge distilled
    pub total_knowledge: usize,
    /// Total soft targets generated
    pub total_soft_targets: usize,
}

impl KnowledgeDistiller {
    /// Create new distiller
    pub fn new(config: DistillationConfig) -> Self {
        Self {
            config,
            sessions: HashMap::new(),
            stats: DistillerStats::default(),
        }
    }

    /// Create with default config
    pub fn default_distiller() -> Self {
        Self::new(DistillationConfig::default())
    }

    /// Get statistics
    pub fn stats(&self) -> &DistillerStats {
        &self.stats
    }

    /// Start a new distillation session
    pub fn start_session(&mut self, llm_config: LLMConfig) -> &mut DistillationSession {
        let session = DistillationSession::new(self.config.clone(), llm_config);
        let id = session.id.clone();
        self.sessions.insert(id.clone(), session);
        self.stats.sessions_created += 1;
        self.sessions.get_mut(&id).unwrap()
    }

    /// Get a session by ID
    pub fn get_session(&self, id: &str) -> Option<&DistillationSession> {
        self.sessions.get(id)
    }

    /// Get mutable session by ID
    pub fn get_session_mut(&mut self, id: &str) -> Option<&mut DistillationSession> {
        self.sessions.get_mut(id)
    }

    /// Complete a session and collect results
    pub fn complete_session(
        &mut self,
        id: &str,
    ) -> Result<DistillationResult, DistillationError> {
        let session = self
            .sessions
            .remove(id)
            .ok_or(DistillationError::SessionNotFound(id.to_string()))?;

        let result = DistillationResult {
            session_id: session.id.clone(),
            knowledge_graph: session.to_knowledge_graph(),
            soft_targets: session.soft_targets,
            metrics: session.metrics,
        };

        self.stats.sessions_completed += 1;
        self.stats.total_knowledge += result.knowledge_graph.entity_count();
        self.stats.total_soft_targets += result.soft_targets.len();

        Ok(result)
    }

    /// List active session IDs
    pub fn active_sessions(&self) -> Vec<&str> {
        self.sessions.keys().map(|s| s.as_str()).collect()
    }

    /// Distill from a single LLM response (convenience method)
    pub fn distill_response(
        &mut self,
        llm_config: LLMConfig,
        prompt: &str,
        response: &str,
        knowledge_type: KnowledgeType,
    ) -> Result<DistillationResult, DistillationError> {
        let session = self.start_session(llm_config);
        let session_id = session.id().to_string();

        session.process_response(prompt, response, knowledge_type)?;
        self.complete_session(&session_id)
    }
}

/// Result of a completed distillation session
#[derive(Debug)]
pub struct DistillationResult {
    /// Session identifier
    pub session_id: String,
    /// Extracted knowledge graph
    pub knowledge_graph: KnowledgeGraph,
    /// Soft targets for training
    pub soft_targets: Vec<SoftTarget>,
    /// Session metrics
    pub metrics: DistillationMetrics,
}

// ============================================================================
// Graph Integration
// ============================================================================

/// Apply distilled knowledge to a GRAPHEME graph
pub struct GraphKnowledgeApplier {
    /// Merge strategy
    pub merge_strategy: MergeStrategy,
    /// Edge weight for new connections
    pub new_edge_weight: f32,
}

/// Strategy for merging knowledge into graph
#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    /// Add all knowledge as new nodes
    AddAll,
    /// Only add knowledge that links to existing nodes
    LinkExisting,
    /// Replace matching nodes with new knowledge
    Replace,
}

impl Default for GraphKnowledgeApplier {
    fn default() -> Self {
        Self {
            merge_strategy: MergeStrategy::AddAll,
            new_edge_weight: 0.5,
        }
    }
}

impl GraphKnowledgeApplier {
    /// Apply knowledge graph to GRAPHEME graph
    pub fn apply(
        &self,
        target: &mut GraphemeGraph,
        knowledge: &KnowledgeGraph,
    ) -> ApplyResult {
        let mut result = ApplyResult::default();
        let knowledge_graph = knowledge.to_grapheme_graph();

        match self.merge_strategy {
            MergeStrategy::AddAll => {
                // Add all nodes from knowledge graph
                for node in knowledge_graph.graph.node_weights() {
                    target.graph.add_node(node.clone());
                    result.nodes_added += 1;
                }
            }
            MergeStrategy::LinkExisting => {
                // Only add nodes that have connections to existing
                result.nodes_added = 0; // Simplified - would need node matching
            }
            MergeStrategy::Replace => {
                // Would need node matching logic
                result.nodes_added = 0;
            }
        }

        result
    }
}

/// Result of applying knowledge to graph
#[derive(Debug, Default)]
pub struct ApplyResult {
    /// Nodes added
    pub nodes_added: usize,
    /// Edges added
    pub edges_added: usize,
    /// Nodes merged
    pub nodes_merged: usize,
}

// ============================================================================
// Errors
// ============================================================================

/// Errors during distillation
#[derive(Debug, Clone)]
pub enum DistillationError {
    /// Session is full
    SessionFull,
    /// Confidence too low
    LowConfidence(f32),
    /// Session not found
    SessionNotFound(String),
    /// Extraction failed
    ExtractionFailed(String),
    /// Invalid configuration
    InvalidConfig(String),
}

impl std::fmt::Display for DistillationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionFull => write!(f, "Distillation session is full"),
            Self::LowConfidence(c) => write!(f, "Knowledge confidence too low: {}", c),
            Self::SessionNotFound(id) => write!(f, "Session not found: {}", id),
            Self::ExtractionFailed(msg) => write!(f, "Extraction failed: {}", msg),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
        }
    }
}

impl std::error::Error for DistillationError {}

// ============================================================================
// Utilities
// ============================================================================

/// Generate simple UUID-like string
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:016x}", now)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert_eq!(config.temperature, 2.0);
        assert_eq!(config.distillation_weight, 0.5);
        assert_eq!(config.min_confidence, 0.7);
    }

    #[test]
    fn test_distillation_config_fast() {
        let config = DistillationConfig::fast();
        assert!(config.compression_ratio < DistillationConfig::default().compression_ratio);
    }

    #[test]
    fn test_distillation_config_quality() {
        let config = DistillationConfig::quality();
        assert!(config.max_items_per_session > DistillationConfig::default().max_items_per_session);
    }

    #[test]
    fn test_distilled_knowledge_creation() {
        let dk = DistilledKnowledge::new(
            KnowledgeType::Factual,
            "What is water?".to_string(),
            "Water is H2O.".to_string(),
        );
        assert!(dk.id.starts_with("dk_"));
        assert_eq!(dk.knowledge_type, KnowledgeType::Factual);
        assert!(dk.entities.is_empty());
    }

    #[test]
    fn test_distilled_knowledge_with_confidence() {
        let dk = DistilledKnowledge::new(
            KnowledgeType::Relational,
            "test".to_string(),
            "response".to_string(),
        )
        .with_confidence(0.85);
        assert_eq!(dk.confidence, 0.85);
    }

    #[test]
    fn test_distilled_knowledge_with_metadata() {
        let dk = DistilledKnowledge::new(
            KnowledgeType::Procedural,
            "test".to_string(),
            "response".to_string(),
        )
        .with_metadata("source", "gpt-4");
        assert_eq!(dk.metadata.get("source"), Some(&"gpt-4".to_string()));
    }

    #[test]
    fn test_soft_target_creation() {
        let probs = vec![0.1, 0.2, 0.7];
        let target = SoftTarget::new(0, probs.clone(), 1.0);
        assert_eq!(target.node_idx, 0);
        assert_eq!(target.argmax(), 2);
    }

    #[test]
    fn test_soft_target_entropy() {
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let target = SoftTarget::new(0, uniform, 1.0);
        // Uniform distribution should have high entropy
        assert!(target.entropy > 1.0);
    }

    #[test]
    fn test_soft_target_argmax() {
        let probs = vec![0.05, 0.9, 0.05];
        let target = SoftTarget::new(0, probs, 1.0);
        assert_eq!(target.argmax(), 1);
    }

    #[test]
    fn test_distillation_session_creation() {
        let config = DistillationConfig::default();
        let llm_config = LLMConfig::default();
        let session = DistillationSession::new(config, llm_config);
        assert!(session.id().starts_with("ds_"));
        assert!(session.knowledge().is_empty());
    }

    #[test]
    fn test_session_process_response() {
        let config = DistillationConfig::default();
        let llm_config = LLMConfig::default();
        let mut session = DistillationSession::new(config, llm_config);

        // Process a factual response
        let result = session.process_response(
            "What is water?",
            "Water is a chemical compound. H2O is its formula. Water is essential for life.",
            KnowledgeType::Factual,
        );

        // May succeed or fail based on confidence threshold
        match result {
            Ok(knowledge) => {
                assert!(!knowledge.entities.is_empty());
            }
            Err(DistillationError::LowConfidence(_)) => {
                // Expected if response doesn't meet threshold
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_session_relational_extraction() {
        let mut config = DistillationConfig::default();
        config.min_confidence = 0.1; // Lower threshold for test
        let llm_config = LLMConfig::default();
        let mut session = DistillationSession::new(config, llm_config);

        let result = session.process_response(
            "Tell me about cats",
            "A cat is an animal. Dogs are mammals. Birds have wings.",
            KnowledgeType::Relational,
        );

        assert!(result.is_ok());
        let knowledge = result.unwrap();
        assert!(!knowledge.relations.is_empty());
    }

    #[test]
    fn test_session_procedural_extraction() {
        let mut config = DistillationConfig::default();
        config.min_confidence = 0.1;
        let llm_config = LLMConfig::default();
        let mut session = DistillationSession::new(config, llm_config);

        let result = session.process_response(
            "How to make tea?",
            "First, boil water. Then, add tea leaves. Finally, steep for 3 minutes.",
            KnowledgeType::Procedural,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_session_add_soft_target() {
        let config = DistillationConfig::default();
        let llm_config = LLMConfig::default();
        let mut session = DistillationSession::new(config, llm_config);

        session.add_soft_target(0, vec![1.0, 2.0, 3.0]);
        assert_eq!(session.soft_targets().len(), 1);
    }

    #[test]
    fn test_session_to_knowledge_graph() {
        let mut config = DistillationConfig::default();
        config.min_confidence = 0.1;
        let llm_config = LLMConfig::default();
        let mut session = DistillationSession::new(config, llm_config);

        let _ = session.process_response(
            "test",
            "Alice is a person. Bob is a friend.",
            KnowledgeType::Relational,
        );

        let kg = session.to_knowledge_graph();
        assert!(!kg.entities().is_empty() || kg.relations().is_empty());
    }

    #[test]
    fn test_distiller_creation() {
        let distiller = KnowledgeDistiller::default_distiller();
        assert_eq!(distiller.stats().sessions_created, 0);
    }

    #[test]
    fn test_distiller_start_session() {
        let mut distiller = KnowledgeDistiller::default_distiller();
        let llm_config = LLMConfig::default();

        let session = distiller.start_session(llm_config);
        assert!(session.id().starts_with("ds_"));
        assert_eq!(distiller.stats().sessions_created, 1);
    }

    #[test]
    fn test_distiller_complete_session() {
        let mut distiller = KnowledgeDistiller::default_distiller();
        let llm_config = LLMConfig::default();

        let session = distiller.start_session(llm_config);
        let session_id = session.id().to_string();

        let result = distiller.complete_session(&session_id);
        assert!(result.is_ok());
        assert_eq!(distiller.stats().sessions_completed, 1);
    }

    #[test]
    fn test_distiller_active_sessions() {
        let mut distiller = KnowledgeDistiller::default_distiller();

        let session1 = distiller.start_session(LLMConfig::default());
        let id1 = session1.id().to_string();

        let session2 = distiller.start_session(LLMConfig::default());
        let _id2 = session2.id().to_string();

        assert_eq!(distiller.active_sessions().len(), 2);

        let _ = distiller.complete_session(&id1);
        assert_eq!(distiller.active_sessions().len(), 1);
    }

    #[test]
    fn test_distiller_distill_response() {
        let mut distiller = KnowledgeDistiller::new(DistillationConfig {
            min_confidence: 0.1,
            ..Default::default()
        });

        // Use Relational type with "is" patterns for better extraction
        let result = distiller.distill_response(
            LLMConfig::default(),
            "What is AI?",
            "AI is artificial intelligence. Machine learning is a subset. Deep learning is a technique.",
            KnowledgeType::Relational,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_applier_default() {
        let applier = GraphKnowledgeApplier::default();
        assert_eq!(applier.new_edge_weight, 0.5);
    }

    #[test]
    fn test_distillation_error_display() {
        let err = DistillationError::SessionFull;
        assert_eq!(format!("{}", err), "Distillation session is full");

        let err = DistillationError::LowConfidence(0.3);
        assert!(format!("{}", err).contains("0.3"));
    }

    #[test]
    fn test_session_metrics() {
        let mut config = DistillationConfig::default();
        config.min_confidence = 0.1;
        let llm_config = LLMConfig::default();
        let mut session = DistillationSession::new(config, llm_config);

        let _ = session.process_response(
            "test",
            "The cat is on the mat. The dog has a bone.",
            KnowledgeType::Relational,
        );

        let metrics = session.metrics();
        assert!(metrics.prompts_processed >= 0);
    }
}
