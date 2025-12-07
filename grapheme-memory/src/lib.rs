//! # grapheme-memory
//!
//! Memory architecture for GRAPHEME neural network.
//!
//! This crate provides four memory types aligned with cognitive science:
//! - **Episodic Memory**: Specific experiences with temporal context
//! - **Semantic Memory**: General knowledge graph (facts, concepts)
//! - **Procedural Memory**: Learned skills and procedures
//! - **Working Memory**: Active reasoning context (limited capacity)
//!
//! Plus:
//! - **Continual Learning**: Memory consolidation without catastrophic forgetting
//!
//! ## Design Principles
//!
//! 1. All memories store/retrieve Graphs (native to GRAPHEME)
//! 2. Episodic has temporal ordering; semantic is atemporal
//! 3. Working memory has limited capacity (~7 items)
//! 4. Continual learning handles contradictions explicitly
//! 5. Traits are object-safe for runtime polymorphism
//!
//! ## NP-Hard Complexity Warning
//!
//! Graph retrieval operations use approximate similarity (not exact isomorphism):
//! - Weisfeiler-Leman kernel for O(n*m*k) similarity
//! - Feature-based hashing for O(n) approximate retrieval
//! - Index by structural features (node count, edge count, degree histogram)

use grapheme_core::{
    BrainRegistry, CognitiveBrainBridge, DagNN, DefaultCognitiveBridge,
    DomainBrain, Learnable, LearnableParam, MultiBrainResult, Persistable,
    PersistenceError, TransformRule,
};
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Errors in memory operations
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Memory capacity exceeded")]
    CapacityExceeded,
    #[error("Episode not found: {0}")]
    EpisodeNotFound(EpisodeId),
    #[error("Fact not found: {0}")]
    FactNotFound(FactId),
    #[error("Procedure not found: {0}")]
    ProcedureNotFound(String),
    #[error("Consolidation error: {0}")]
    ConsolidationError(String),
    #[error("Retrieval error: {0}")]
    RetrievalError(String),
    #[error("Contradiction detected: {0}")]
    ContradictionDetected(String),
}

/// Result type for memory operations
pub type MemoryResult<T> = Result<T, MemoryError>;

// ============================================================================
// Core Type Aliases
// ============================================================================

/// Unique identifier for episodes
pub type EpisodeId = u64;

/// Unique identifier for facts
pub type FactId = u64;

/// Timestamp type (milliseconds since epoch)
pub type Timestamp = u64;

/// Node identifier from graphs
pub type NodeId = NodeIndex;

/// A graph type alias for memory storage
pub type Graph = DagNN;

// ============================================================================
// Supporting Types
// ============================================================================

/// Source of knowledge (for provenance tracking)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    /// Directly observed/experienced
    Direct,
    /// Inferred from other facts
    Inferred,
    /// Learned from external input
    External(String),
    /// Unknown provenance
    Unknown,
}

/// Retention policy for memory consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Maximum age in milliseconds (older episodes may be forgotten)
    pub max_age_ms: Option<u64>,
    /// Minimum importance threshold (less important episodes may be forgotten)
    pub min_importance: f32,
    /// Maximum number of episodes to retain
    pub max_episodes: Option<usize>,
    /// Whether to consolidate similar episodes
    pub consolidate_similar: bool,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_age_ms: None,
            min_importance: 0.0,
            max_episodes: Some(10_000),
            consolidate_similar: true,
        }
    }
}

/// Feedback for refining procedures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureFeedback {
    /// Whether the procedure succeeded
    pub success: bool,
    /// Reward signal (-1.0 to 1.0)
    pub reward: f32,
    /// Optional error message
    pub error: Option<String>,
}

impl ProcedureFeedback {
    /// Create positive feedback
    pub fn success(reward: f32) -> Self {
        Self {
            success: true,
            reward: reward.clamp(-1.0, 1.0),
            error: None,
        }
    }

    /// Create negative feedback
    pub fn failure(reward: f32) -> Self {
        Self {
            success: false,
            reward: reward.clamp(-1.0, 1.0),
            error: None,
        }
    }

    /// Create failure with error message
    pub fn failure_with_error(reward: f32, error: String) -> Self {
        Self {
            success: false,
            reward: reward.clamp(-1.0, 1.0),
            error: Some(error),
        }
    }
}

/// Result of reconciling new facts with existing knowledge
#[derive(Debug, Clone)]
pub enum ReconciliationResult {
    /// New fact is consistent with existing knowledge
    Consistent,
    /// New fact contradicts existing - details provided
    Contradiction {
        existing_fact: FactId,
        resolution: ConflictResolution,
    },
    /// New fact is redundant (already known)
    Redundant(FactId),
    /// New fact provides more specific information
    Refinement(FactId),
}

/// How to resolve a knowledge conflict
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictResolution {
    /// Keep the existing fact
    KeepExisting,
    /// Replace with the new fact
    ReplaceWithNew,
    /// Keep both (contradiction remains)
    KeepBoth,
    /// Merge into a combined fact
    Merge,
}

// ============================================================================
// Episode Structure
// ============================================================================

/// An episode: a specific experience with temporal context
///
/// Note: Episodes contain Graph types which don't derive Clone/Serialize.
/// Use the accessor methods to work with episode data.
#[derive(Debug)]
pub struct Episode {
    /// Unique identifier
    pub id: EpisodeId,
    /// When this episode occurred
    pub timestamp: Timestamp,
    /// The situational context (graph representation)
    pub context: Graph,
    /// The content of what happened (graph representation)
    pub content: Graph,
    /// The outcome/consequence (if known)
    pub outcome: Option<Graph>,
    /// Emotional valence for prioritization (-1.0 to 1.0)
    pub emotional_valence: f32,
    /// Importance score (0.0 to 1.0)
    pub importance: f32,
    /// Access count (for memory consolidation)
    pub access_count: u64,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl Episode {
    /// Create a new episode
    pub fn new(id: EpisodeId, timestamp: Timestamp, context: Graph, content: Graph) -> Self {
        Self {
            id,
            timestamp,
            context,
            content,
            outcome: None,
            emotional_valence: 0.0,
            importance: 0.5,
            access_count: 0,
            tags: Vec::new(),
        }
    }

    /// Set the outcome
    pub fn with_outcome(mut self, outcome: Graph) -> Self {
        self.outcome = Some(outcome);
        self
    }

    /// Set emotional valence
    pub fn with_valence(mut self, valence: f32) -> Self {
        self.emotional_valence = valence.clamp(-1.0, 1.0);
        self
    }

    /// Set importance
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Increment access count
    pub fn record_access(&mut self) {
        self.access_count += 1;
    }
}

// ============================================================================
// Episodic Memory Trait
// ============================================================================

/// Episodic memory: stores specific experiences with temporal context
///
/// ## NP-Hard Mitigation
/// - `recall(query)` uses approximate graph similarity (WL kernel)
/// - Retrieval is bounded by `limit` parameter
/// - Index by structural features for O(n) candidate filtering
pub trait EpisodicMemory: Send + Sync + Debug {
    /// Store a new episode
    fn store(&mut self, episode: Episode) -> EpisodeId;

    /// Retrieve episode IDs similar to query (bounded by limit)
    ///
    /// Uses approximate graph similarity, not exact isomorphism.
    /// Returns IDs sorted by similarity (highest first).
    fn recall(&self, query: &Graph, limit: usize) -> Vec<EpisodeId>;

    /// Retrieve episode IDs from a time range
    fn recall_temporal(&self, start: Timestamp, end: Timestamp) -> Vec<EpisodeId>;

    /// Retrieve episode IDs by tags
    fn recall_by_tags(&self, tags: &[String], limit: usize) -> Vec<EpisodeId>;

    /// Get a specific episode by ID
    fn get(&self, id: EpisodeId) -> Option<&Episode>;

    /// Get a mutable reference to an episode
    fn get_mut(&mut self, id: EpisodeId) -> Option<&mut Episode>;

    /// Consolidate memory according to retention policy
    fn consolidate(&mut self, policy: &RetentionPolicy);

    /// Get the number of stored episodes
    fn len(&self) -> usize;

    /// Check if memory is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get total capacity (if bounded)
    fn capacity(&self) -> Option<usize>;
}

// ============================================================================
// Semantic Memory Trait
// ============================================================================

/// Semantic memory: general knowledge graph
///
/// Stores facts, concepts, and relationships as graphs.
/// Unlike episodic memory, semantic memory is atemporal.
///
/// ## NP-Hard Mitigation
/// - `query(pattern)` uses WL kernel similarity
/// - Index facts by structural fingerprint
/// - Bound query complexity with top-k retrieval
pub trait SemanticGraph: Send + Sync + Debug {
    /// Add a fact to the knowledge graph
    fn assert(&mut self, fact: Graph) -> FactId;

    /// Add a fact with source provenance
    fn assert_with_source(&mut self, fact: Graph, source: Source) -> FactId;

    /// Query the knowledge graph for matching patterns
    ///
    /// Uses approximate similarity, not exact subgraph matching.
    /// Returns FactIds sorted by similarity (highest first).
    fn query(&self, pattern: &Graph, limit: usize) -> Vec<FactId>;

    /// Check if a fact is known (approximately)
    fn contains(&self, fact: &Graph) -> bool;

    /// Get all facts about an entity (node)
    fn about(&self, entity: NodeId) -> Vec<FactId>;

    /// Update/revise a fact with provenance tracking
    fn revise(&mut self, old_fact_id: FactId, new_fact: Graph, source: Source) -> MemoryResult<FactId>;

    /// Get a fact by ID
    fn get(&self, id: FactId) -> Option<&Graph>;

    /// Get the source of a fact
    fn get_source(&self, id: FactId) -> Option<&Source>;

    /// Remove a fact (consumes it, can't return since Graph doesn't implement Clone)
    fn retract(&mut self, id: FactId) -> bool;

    /// Get the number of facts
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// Procedural Memory Trait
// ============================================================================

/// Procedural memory: learned skills and procedures
///
/// Stores graph transformation patterns (how to do things).
/// Procedures are named TransformRules that can be applied to graphs.
pub trait ProceduralMemory: Send + Sync + Debug {
    /// Store a procedure (graph transformation pattern)
    fn learn(&mut self, name: &str, procedure: TransformRule);

    /// Retrieve a procedure by name
    fn recall(&self, name: &str) -> Option<&TransformRule>;

    /// Find procedures applicable to a situation
    ///
    /// Returns procedures whose input pattern matches the situation.
    fn applicable(&self, situation: &Graph, limit: usize) -> Vec<(&str, &TransformRule)>;

    /// Improve a procedure based on feedback
    fn refine(&mut self, name: &str, feedback: ProcedureFeedback);

    /// Get performance statistics for a procedure
    fn stats(&self, name: &str) -> Option<ProcedureStats>;

    /// List all procedure names
    fn list(&self) -> Vec<&str>;

    /// Check if a procedure exists
    fn contains(&self, name: &str) -> bool;

    /// Remove a procedure
    fn forget(&mut self, name: &str) -> Option<TransformRule>;

    /// Get the number of procedures
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Statistics for a procedure
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcedureStats {
    /// Number of times applied
    pub applications: u64,
    /// Number of successful applications
    pub successes: u64,
    /// Average reward received
    pub avg_reward: f32,
    /// Last application timestamp
    pub last_used: Option<Timestamp>,
}

impl ProcedureStats {
    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f32 {
        if self.applications == 0 {
            0.0
        } else {
            self.successes as f32 / self.applications as f32
        }
    }
}

// ============================================================================
// Working Memory Trait
// ============================================================================

/// Working memory: active reasoning context
///
/// Limited capacity buffer (~7 items, like human working memory).
/// Items are evicted when capacity is exceeded (LRU or priority-based).
pub trait WorkingMemory: Send + Sync + Debug {
    /// Get current capacity
    fn capacity(&self) -> usize;

    /// Get current number of items
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if at capacity
    fn is_full(&self) -> bool {
        self.len() >= self.capacity()
    }

    /// Add item to working memory
    ///
    /// Returns true if an item was evicted due to capacity.
    fn attend(&mut self, item: Graph) -> bool;

    /// Get an item by index
    fn get(&self, index: usize) -> Option<&Graph>;

    /// Get all item indices (for iteration)
    fn indices(&self) -> std::ops::Range<usize>;

    /// Clear working memory
    fn clear(&mut self);

    /// Focus on a specific item (boost its priority)
    ///
    /// Returns true if the item was found and focused.
    fn focus(&mut self, index: usize) -> bool;

    /// Get the most recently attended item
    fn current(&self) -> Option<&Graph>;

    /// Remove a specific item by index (returns true if removed)
    fn remove(&mut self, index: usize) -> bool;
}

// ============================================================================
// Continual Learning Trait
// ============================================================================

/// Continual learning: consolidation and integration
///
/// Handles learning new information without catastrophic forgetting.
/// Implements strategies like:
/// - Elastic Weight Consolidation (EWC)
/// - Experience Replay
/// - Sleep-like offline consolidation
pub trait ContinualLearning: Send + Sync + Debug {
    /// Consolidate new experience without forgetting old
    fn consolidate(&mut self, new_experience: Graph);

    /// Detect and resolve contradictions with existing knowledge
    fn reconcile(
        &mut self,
        new_fact: Graph,
        existing: &dyn SemanticGraph,
    ) -> ReconciliationResult;

    /// Sleep-like offline processing (replay and integrate)
    ///
    /// This is meant to be called periodically to strengthen important
    /// memories and prune unimportant ones.
    fn replay_and_integrate(&mut self);

    /// Estimate knowledge coverage of a domain
    ///
    /// Returns a score from 0.0 (no coverage) to 1.0 (complete coverage).
    fn coverage(&self, domain: &Graph) -> f32;

    /// Get consolidation statistics
    fn stats(&self) -> ConsolidationStats;
}

/// Statistics for continual learning
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationStats {
    /// Number of consolidation cycles
    pub consolidations: u64,
    /// Number of experiences integrated
    pub experiences_integrated: u64,
    /// Number of contradictions resolved
    pub contradictions_resolved: u64,
    /// Number of facts pruned
    pub facts_pruned: u64,
}

// ============================================================================
// Unified Memory System
// ============================================================================

/// Unified memory system combining all memory types
pub struct MemorySystem {
    /// Episodic memory: specific experiences with temporal context
    pub episodic: Box<dyn EpisodicMemory>,
    /// Semantic memory: general knowledge (facts, concepts)
    pub semantic: Box<dyn SemanticGraph>,
    /// Procedural memory: how to do things
    pub procedural: Box<dyn ProceduralMemory>,
    /// Working memory: active reasoning context
    pub working: Box<dyn WorkingMemory>,
    /// Continual learning: memory consolidation
    pub learning: Box<dyn ContinualLearning>,
}

impl MemorySystem {
    /// Create a new memory system with the given components
    pub fn new(
        episodic: Box<dyn EpisodicMemory>,
        semantic: Box<dyn SemanticGraph>,
        procedural: Box<dyn ProceduralMemory>,
        working: Box<dyn WorkingMemory>,
        learning: Box<dyn ContinualLearning>,
    ) -> Self {
        Self {
            episodic,
            semantic,
            procedural,
            working,
            learning,
        }
    }

    /// Consolidate all memories (should be called periodically)
    pub fn consolidate(&mut self, policy: &RetentionPolicy) {
        self.episodic.consolidate(policy);
        self.learning.replay_and_integrate();
    }
}

impl Debug for MemorySystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemorySystem")
            .field("episodic", &format!("{} episodes", self.episodic.len()))
            .field("semantic", &format!("{} facts", self.semantic.len()))
            .field("procedural", &format!("{} procedures", self.procedural.len()))
            .field("working", &format!("{}/{} items", self.working.len(), self.working.capacity()))
            .finish()
    }
}

// ============================================================================
// Graph Similarity Utilities
// ============================================================================

/// Graph similarity using structural features (not exact isomorphism)
///
/// This provides O(n) approximate similarity rather than NP-complete exact matching.
#[derive(Debug, Clone, Default)]
pub struct GraphFingerprint {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Node type distribution
    pub node_types: [usize; 8],
    /// Degree histogram (0-7+)
    pub degree_hist: [usize; 8],
    /// Hash of structure
    pub structure_hash: u64,
}

impl GraphFingerprint {
    /// Compute fingerprint from a graph
    pub fn from_graph(graph: &Graph) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let node_count = graph.node_count();
        let edge_count = graph.edge_count();

        // Compute degree histogram and node type distribution
        let mut degree_hist = [0usize; 8];
        let mut node_types = [0usize; 8];

        for node in graph.graph.node_indices() {
            let degree = graph.graph.edges(node).count();
            let bucket = degree.min(7);
            degree_hist[bucket] += 1;

            // Categorize nodes by activation level (buckets 0-7)
            let node_data = &graph.graph[node];
            let activation_bucket = ((node_data.activation * 7.0).clamp(0.0, 7.0)) as usize;
            node_types[activation_bucket] += 1;
        }

        // Simple structure hash
        let mut hasher = DefaultHasher::new();
        node_count.hash(&mut hasher);
        edge_count.hash(&mut hasher);
        degree_hist.hash(&mut hasher);
        node_types.hash(&mut hasher);
        let structure_hash = hasher.finish();

        Self {
            node_count,
            edge_count,
            node_types,
            degree_hist,
            structure_hash,
        }
    }

    /// Compute similarity between two fingerprints (0.0 to 1.0)
    pub fn similarity(&self, other: &GraphFingerprint) -> f32 {
        // Jaccard-like similarity on features
        let node_sim = 1.0 - (self.node_count as f32 - other.node_count as f32).abs()
            / (self.node_count.max(other.node_count).max(1) as f32);

        let edge_sim = 1.0 - (self.edge_count as f32 - other.edge_count as f32).abs()
            / (self.edge_count.max(other.edge_count).max(1) as f32);

        // Degree histogram similarity
        let mut degree_diff = 0.0f32;
        let mut degree_total = 0.0f32;
        for i in 0..8 {
            degree_diff += (self.degree_hist[i] as f32 - other.degree_hist[i] as f32).abs();
            degree_total += (self.degree_hist[i] + other.degree_hist[i]) as f32;
        }
        let degree_sim = if degree_total > 0.0 {
            1.0 - degree_diff / degree_total
        } else {
            1.0
        };

        // Node type distribution similarity
        let mut type_diff = 0.0f32;
        let mut type_total = 0.0f32;
        for i in 0..8 {
            type_diff += (self.node_types[i] as f32 - other.node_types[i] as f32).abs();
            type_total += (self.node_types[i] + other.node_types[i]) as f32;
        }
        let type_sim = if type_total > 0.0 {
            1.0 - type_diff / type_total
        } else {
            1.0
        };

        // Combined similarity (weighted average)
        (node_sim + edge_sim + degree_sim + type_sim) / 4.0
    }
}

// ============================================================================
// Simple In-Memory Implementations (for testing/prototyping)
// ============================================================================

/// Simple in-memory episodic memory implementation
#[derive(Debug, Default)]
pub struct SimpleEpisodicMemory {
    episodes: Vec<Episode>,
    next_id: EpisodeId,
    capacity: Option<usize>,
}

impl SimpleEpisodicMemory {
    /// Create with optional capacity limit
    pub fn new(capacity: Option<usize>) -> Self {
        Self {
            episodes: Vec::new(),
            next_id: 1,
            capacity,
        }
    }
}

impl EpisodicMemory for SimpleEpisodicMemory {
    fn store(&mut self, mut episode: Episode) -> EpisodeId {
        let id = self.next_id;
        self.next_id += 1;
        episode.id = id;
        self.episodes.push(episode);
        id
    }

    fn recall(&self, query: &Graph, limit: usize) -> Vec<EpisodeId> {
        let query_fp = GraphFingerprint::from_graph(query);

        let mut scored: Vec<_> = self.episodes.iter()
            .map(|e| {
                let fp = GraphFingerprint::from_graph(&e.content);
                (fp.similarity(&query_fp), e.id)
            })
            .collect();

        scored.sort_by(|a, b| b.0.total_cmp(&a.0));

        scored.into_iter()
            .take(limit)
            .map(|(_, id)| id)
            .collect()
    }

    fn recall_temporal(&self, start: Timestamp, end: Timestamp) -> Vec<EpisodeId> {
        self.episodes.iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .map(|e| e.id)
            .collect()
    }

    fn recall_by_tags(&self, tags: &[String], limit: usize) -> Vec<EpisodeId> {
        self.episodes.iter()
            .filter(|e| tags.iter().any(|t| e.tags.contains(t)))
            .take(limit)
            .map(|e| e.id)
            .collect()
    }

    fn get(&self, id: EpisodeId) -> Option<&Episode> {
        self.episodes.iter().find(|e| e.id == id)
    }

    fn get_mut(&mut self, id: EpisodeId) -> Option<&mut Episode> {
        self.episodes.iter_mut().find(|e| e.id == id)
    }

    fn consolidate(&mut self, policy: &RetentionPolicy) {
        // Remove episodes below importance threshold
        self.episodes.retain(|e| e.importance >= policy.min_importance);

        // Enforce max episodes
        if let Some(max) = policy.max_episodes {
            if self.episodes.len() > max {
                // Sort by importance and keep top N
                self.episodes.sort_by(|a, b| {
                    b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
                });
                self.episodes.truncate(max);
            }
        }
    }

    fn len(&self) -> usize {
        self.episodes.len()
    }

    fn capacity(&self) -> Option<usize> {
        self.capacity
    }
}

/// Simple in-memory semantic graph implementation
#[derive(Debug, Default)]
pub struct SimpleSemanticGraph {
    facts: Vec<(Graph, Source)>,
    next_id: FactId,
}

impl SimpleSemanticGraph {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SemanticGraph for SimpleSemanticGraph {
    fn assert(&mut self, fact: Graph) -> FactId {
        self.assert_with_source(fact, Source::Unknown)
    }

    fn assert_with_source(&mut self, fact: Graph, source: Source) -> FactId {
        let id = self.next_id;
        self.next_id += 1;
        self.facts.push((fact, source));
        id
    }

    fn query(&self, pattern: &Graph, limit: usize) -> Vec<FactId> {
        let pattern_fp = GraphFingerprint::from_graph(pattern);

        let mut scored: Vec<_> = self.facts.iter()
            .enumerate()
            .map(|(idx, (g, _))| {
                let fp = GraphFingerprint::from_graph(g);
                (fp.similarity(&pattern_fp), idx as FactId)
            })
            .collect();

        scored.sort_by(|a, b| b.0.total_cmp(&a.0));

        scored.into_iter()
            .take(limit)
            .map(|(_, id)| id)
            .collect()
    }

    fn contains(&self, fact: &Graph) -> bool {
        let fp = GraphFingerprint::from_graph(fact);
        self.facts.iter().any(|(g, _)| {
            let other_fp = GraphFingerprint::from_graph(g);
            fp.similarity(&other_fp) > 0.95
        })
    }

    fn about(&self, _entity: NodeId) -> Vec<FactId> {
        // Simplified: return all fact IDs (proper implementation would filter)
        (0..self.facts.len() as FactId).collect()
    }

    fn revise(&mut self, old_fact_id: FactId, new_fact: Graph, source: Source) -> MemoryResult<FactId> {
        let idx = old_fact_id as usize;
        if idx < self.facts.len() {
            self.facts[idx] = (new_fact, source);
            Ok(old_fact_id)
        } else {
            Err(MemoryError::FactNotFound(old_fact_id))
        }
    }

    fn get(&self, id: FactId) -> Option<&Graph> {
        self.facts.get(id as usize).map(|(g, _)| g)
    }

    fn get_source(&self, id: FactId) -> Option<&Source> {
        self.facts.get(id as usize).map(|(_, s)| s)
    }

    fn retract(&mut self, id: FactId) -> bool {
        let idx = id as usize;
        if idx < self.facts.len() {
            self.facts.remove(idx);
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.facts.len()
    }
}

/// Simple in-memory procedural memory implementation
#[derive(Debug, Default)]
pub struct SimpleProceduralMemory {
    procedures: std::collections::HashMap<String, (TransformRule, ProcedureStats)>,
}

impl SimpleProceduralMemory {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ProceduralMemory for SimpleProceduralMemory {
    fn learn(&mut self, name: &str, procedure: TransformRule) {
        self.procedures.insert(
            name.to_string(),
            (procedure, ProcedureStats::default()),
        );
    }

    fn recall(&self, name: &str) -> Option<&TransformRule> {
        self.procedures.get(name).map(|(p, _)| p)
    }

    fn applicable(&self, _situation: &Graph, limit: usize) -> Vec<(&str, &TransformRule)> {
        // Simplified: return all procedures (proper implementation would match patterns)
        self.procedures.iter()
            .take(limit)
            .map(|(name, (rule, _))| (name.as_str(), rule))
            .collect()
    }

    fn refine(&mut self, name: &str, feedback: ProcedureFeedback) {
        if let Some((_, stats)) = self.procedures.get_mut(name) {
            stats.applications += 1;
            if feedback.success {
                stats.successes += 1;
            }
            // Update average reward
            let n = stats.applications as f32;
            stats.avg_reward = ((n - 1.0) * stats.avg_reward + feedback.reward) / n;
        }
    }

    fn stats(&self, name: &str) -> Option<ProcedureStats> {
        self.procedures.get(name).map(|(_, s)| s.clone())
    }

    fn list(&self) -> Vec<&str> {
        self.procedures.keys().map(|s| s.as_str()).collect()
    }

    fn contains(&self, name: &str) -> bool {
        self.procedures.contains_key(name)
    }

    fn forget(&mut self, name: &str) -> Option<TransformRule> {
        self.procedures.remove(name).map(|(p, _)| p)
    }

    fn len(&self) -> usize {
        self.procedures.len()
    }
}

/// Simple in-memory working memory implementation
#[derive(Debug)]
pub struct SimpleWorkingMemory {
    items: Vec<Graph>,
    capacity: usize,
}

impl SimpleWorkingMemory {
    /// Create with capacity (default ~7 like human working memory)
    pub fn new(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            capacity,
        }
    }
}

impl Default for SimpleWorkingMemory {
    fn default() -> Self {
        Self::new(7)
    }
}

impl WorkingMemory for SimpleWorkingMemory {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn attend(&mut self, item: Graph) -> bool {
        let evicted = if self.items.len() >= self.capacity {
            self.items.remove(0); // LRU: remove oldest
            true
        } else {
            false
        };
        self.items.push(item);
        evicted
    }

    fn get(&self, index: usize) -> Option<&Graph> {
        self.items.get(index)
    }

    fn indices(&self) -> std::ops::Range<usize> {
        0..self.items.len()
    }

    fn clear(&mut self) {
        self.items.clear();
    }

    fn focus(&mut self, index: usize) -> bool {
        if index < self.items.len() {
            let item = self.items.remove(index);
            self.items.push(item); // Move to end (most recent)
            true
        } else {
            false
        }
    }

    fn current(&self) -> Option<&Graph> {
        self.items.last()
    }

    fn remove(&mut self, index: usize) -> bool {
        if index < self.items.len() {
            self.items.remove(index);
            true
        } else {
            false
        }
    }
}

/// Simple continual learning implementation
#[derive(Debug, Default)]
pub struct SimpleContinualLearning {
    experience_buffer: Vec<Graph>,
    buffer_capacity: usize,
    stats: ConsolidationStats,
}

impl SimpleContinualLearning {
    pub fn new(buffer_capacity: usize) -> Self {
        Self {
            experience_buffer: Vec::new(),
            buffer_capacity,
            stats: ConsolidationStats::default(),
        }
    }
}

impl ContinualLearning for SimpleContinualLearning {
    fn consolidate(&mut self, new_experience: Graph) {
        if self.experience_buffer.len() >= self.buffer_capacity {
            self.experience_buffer.remove(0);
        }
        self.experience_buffer.push(new_experience);
        self.stats.experiences_integrated += 1;
    }

    fn reconcile(
        &mut self,
        _new_fact: Graph,
        _existing: &dyn SemanticGraph,
    ) -> ReconciliationResult {
        // Simplified: always consistent
        ReconciliationResult::Consistent
    }

    fn replay_and_integrate(&mut self) {
        self.stats.consolidations += 1;
        // In a real implementation, this would:
        // 1. Sample from experience buffer
        // 2. Replay experiences through the system
        // 3. Strengthen important connections
    }

    fn coverage(&self, _domain: &Graph) -> f32 {
        // Simplified: return proportion of buffer filled
        self.experience_buffer.len() as f32 / self.buffer_capacity as f32
    }

    fn stats(&self) -> ConsolidationStats {
        self.stats.clone()
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a default memory system with simple implementations
pub fn create_default_memory_system() -> MemorySystem {
    MemorySystem::new(
        Box::new(SimpleEpisodicMemory::new(Some(10_000))),
        Box::new(SimpleSemanticGraph::new()),
        Box::new(SimpleProceduralMemory::new()),
        Box::new(SimpleWorkingMemory::new(7)),
        Box::new(SimpleContinualLearning::new(1000)),
    )
}

// ============================================================================
// Learnable Memory Retrieval
// ============================================================================

/// Learnable memory retrieval with trainable similarity weights
///
/// This module learns to weight different aspects of graph similarity
/// for memory retrieval (node count, edge count, degree distribution, type distribution).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableMemoryRetrieval {
    /// Weight for node count similarity
    pub node_weight: LearnableParam,
    /// Weight for edge count similarity
    pub edge_weight: LearnableParam,
    /// Weight for degree histogram similarity
    pub degree_weight: LearnableParam,
    /// Weight for node type distribution similarity
    pub type_weight: LearnableParam,
    /// Bias term for importance scoring
    pub importance_bias: LearnableParam,
    /// Temperature for softmax-like retrieval
    pub temperature: LearnableParam,
}

impl LearnableMemoryRetrieval {
    /// Create a new learnable memory retrieval with default weights
    pub fn new() -> Self {
        Self {
            node_weight: LearnableParam::new(0.25),
            edge_weight: LearnableParam::new(0.25),
            degree_weight: LearnableParam::new(0.25),
            type_weight: LearnableParam::new(0.25),
            importance_bias: LearnableParam::new(0.0),
            temperature: LearnableParam::new(1.0),
        }
    }

    /// Compute weighted similarity between two fingerprints
    pub fn weighted_similarity(&self, a: &GraphFingerprint, b: &GraphFingerprint) -> f32 {
        // Node count similarity
        let node_sim = 1.0 - (a.node_count as f32 - b.node_count as f32).abs()
            / (a.node_count.max(b.node_count).max(1) as f32);

        // Edge count similarity
        let edge_sim = 1.0 - (a.edge_count as f32 - b.edge_count as f32).abs()
            / (a.edge_count.max(b.edge_count).max(1) as f32);

        // Degree histogram similarity
        let mut degree_diff = 0.0f32;
        let mut degree_total = 0.0f32;
        for i in 0..8 {
            degree_diff += (a.degree_hist[i] as f32 - b.degree_hist[i] as f32).abs();
            degree_total += (a.degree_hist[i] + b.degree_hist[i]) as f32;
        }
        let degree_sim = if degree_total > 0.0 {
            1.0 - degree_diff / degree_total
        } else {
            1.0
        };

        // Node type distribution similarity
        let mut type_diff = 0.0f32;
        let mut type_total = 0.0f32;
        for i in 0..8 {
            type_diff += (a.node_types[i] as f32 - b.node_types[i] as f32).abs();
            type_total += (a.node_types[i] + b.node_types[i]) as f32;
        }
        let type_sim = if type_total > 0.0 {
            1.0 - type_diff / type_total
        } else {
            1.0
        };

        // Weighted sum with softmax-normalized weights
        let weights = [
            self.node_weight.value.exp(),
            self.edge_weight.value.exp(),
            self.degree_weight.value.exp(),
            self.type_weight.value.exp(),
        ];
        let weight_sum: f32 = weights.iter().sum();

        (weights[0] * node_sim + weights[1] * edge_sim +
         weights[2] * degree_sim + weights[3] * type_sim) / weight_sum
         + self.importance_bias.value
    }

    /// Score an episode for retrieval given a query
    pub fn score_episode(&self, query_fp: &GraphFingerprint, episode: &Episode) -> f32 {
        let content_fp = GraphFingerprint::from_graph(&episode.content);
        let sim = self.weighted_similarity(query_fp, &content_fp);

        // Apply temperature scaling
        (sim / self.temperature.value.max(0.01)).clamp(-10.0, 10.0)
    }
}

impl Default for LearnableMemoryRetrieval {
    fn default() -> Self {
        Self::new()
    }
}

impl Learnable for LearnableMemoryRetrieval {
    fn zero_grad(&mut self) {
        self.node_weight.zero_grad();
        self.edge_weight.zero_grad();
        self.degree_weight.zero_grad();
        self.type_weight.zero_grad();
        self.importance_bias.zero_grad();
        self.temperature.zero_grad();
    }

    fn step(&mut self, lr: f32) {
        self.node_weight.step(lr);
        self.edge_weight.step(lr);
        self.degree_weight.step(lr);
        self.type_weight.step(lr);
        self.importance_bias.step(lr);
        self.temperature.step(lr);
        // Ensure temperature stays positive
        self.temperature.value = self.temperature.value.max(0.01);
    }

    fn num_parameters(&self) -> usize {
        6 // node, edge, degree, type weights + importance_bias + temperature
    }

    fn has_gradients(&self) -> bool {
        self.node_weight.grad != 0.0
            || self.edge_weight.grad != 0.0
            || self.degree_weight.grad != 0.0
            || self.type_weight.grad != 0.0
            || self.importance_bias.grad != 0.0
            || self.temperature.grad != 0.0
    }

    fn gradient_norm(&self) -> f32 {
        (self.node_weight.grad.powi(2)
            + self.edge_weight.grad.powi(2)
            + self.degree_weight.grad.powi(2)
            + self.type_weight.grad.powi(2)
            + self.importance_bias.grad.powi(2)
            + self.temperature.grad.powi(2))
        .sqrt()
    }
}

impl Persistable for LearnableMemoryRetrieval {
    fn persist_type_id() -> &'static str {
        "LearnableMemoryRetrieval"
    }

    fn persist_version() -> u32 {
        1
    }

    fn validate(&self) -> Result<(), PersistenceError> {
        // Validate temperature is positive
        if self.temperature.value <= 0.0 {
            return Err(PersistenceError::ValidationFailed(
                "Temperature must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Domain-Aware Memory
// ============================================================================

/// Metadata for a domain-tagged memory item
#[derive(Debug, Clone)]
pub struct DomainMemoryMetadata {
    /// The domain this memory is associated with (e.g., "math", "code", "chemistry")
    pub domain_id: Option<String>,
    /// Confidence in the domain classification
    pub domain_confidence: f32,
    /// Additional domains that might be relevant
    pub related_domains: Vec<String>,
    /// Source input text (for re-routing if needed)
    pub source_text: Option<String>,
}

impl Default for DomainMemoryMetadata {
    fn default() -> Self {
        Self {
            domain_id: None,
            domain_confidence: 0.0,
            related_domains: Vec::new(),
            source_text: None,
        }
    }
}

impl DomainMemoryMetadata {
    /// Create metadata for a specific domain
    pub fn for_domain(domain_id: &str, confidence: f32) -> Self {
        Self {
            domain_id: Some(domain_id.to_string()),
            domain_confidence: confidence.clamp(0.0, 1.0),
            related_domains: Vec::new(),
            source_text: None,
        }
    }

    /// Add related domains
    pub fn with_related(mut self, domains: Vec<String>) -> Self {
        self.related_domains = domains;
        self
    }

    /// Set source text
    pub fn with_source(mut self, text: &str) -> Self {
        self.source_text = Some(text.to_string());
        self
    }
}

/// Result of domain-aware memory storage
#[derive(Debug)]
pub struct DomainMemoryStorageResult {
    /// ID of the stored item (episode or fact)
    pub storage_id: u64,
    /// Domain metadata
    pub metadata: DomainMemoryMetadata,
    /// Whether a brain was used to process the input
    pub brain_processed: bool,
}

/// Result of domain-aware memory retrieval
#[derive(Debug)]
pub struct DomainMemoryRetrievalResult {
    /// Retrieved IDs
    pub ids: Vec<u64>,
    /// Domain routing results (if domain filtering was used)
    pub routing_result: Option<MultiBrainResult>,
    /// Domain filter that was applied
    pub domain_filter: Option<String>,
}

/// Domain-aware memory system that integrates with domain brains
///
/// This system can:
/// - Automatically classify memories by domain
/// - Store memories with domain metadata
/// - Retrieve memories filtered by domain
/// - Route queries to appropriate domain brains
pub struct DomainAwareMemory {
    /// The underlying memory system
    pub memory: MemorySystem,
    /// The cognitive-brain bridge for domain routing
    pub bridge: DefaultCognitiveBridge,
    /// Domain metadata for episodic memories
    episodic_metadata: std::collections::HashMap<EpisodeId, DomainMemoryMetadata>,
    /// Domain metadata for semantic facts
    semantic_metadata: std::collections::HashMap<FactId, DomainMemoryMetadata>,
    /// Whether to auto-classify memories by domain
    pub auto_classify: bool,
    /// Minimum confidence for domain classification
    pub classification_threshold: f32,
}

impl Debug for DomainAwareMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DomainAwareMemory")
            .field("episodic_count", &self.memory.episodic.len())
            .field("semantic_count", &self.memory.semantic.len())
            .field("available_domains", &self.bridge.available_domains())
            .field("auto_classify", &self.auto_classify)
            .finish()
    }
}

impl DomainAwareMemory {
    /// Create a new domain-aware memory system
    pub fn new(memory: MemorySystem) -> Self {
        Self {
            memory,
            bridge: DefaultCognitiveBridge::new(),
            episodic_metadata: std::collections::HashMap::new(),
            semantic_metadata: std::collections::HashMap::new(),
            auto_classify: true,
            classification_threshold: 0.5,
        }
    }

    /// Create with a pre-configured bridge
    pub fn with_bridge(memory: MemorySystem, bridge: DefaultCognitiveBridge) -> Self {
        Self {
            memory,
            bridge,
            episodic_metadata: std::collections::HashMap::new(),
            semantic_metadata: std::collections::HashMap::new(),
            auto_classify: true,
            classification_threshold: 0.5,
        }
    }

    /// Register a domain brain
    pub fn register_brain(&mut self, brain: Box<dyn DomainBrain>) {
        self.bridge.register(brain);
    }

    /// Classify input text to a domain
    pub fn classify_domain(&self, text: &str) -> Option<DomainMemoryMetadata> {
        let result = self.bridge.route_to_multiple_brains(text);
        if result.success {
            if let Some(primary) = &result.primary {
                if primary.confidence >= self.classification_threshold {
                    return Some(DomainMemoryMetadata {
                        domain_id: Some(primary.domain_id.clone()),
                        domain_confidence: primary.confidence,
                        related_domains: result.domains().iter()
                            .filter(|&d| *d != primary.domain_id)
                            .map(|s| s.to_string())
                            .collect(),
                        source_text: Some(text.to_string()),
                    });
                }
            }
        }
        None
    }

    /// Store an episode with automatic domain classification
    pub fn store_episode_with_domain(
        &mut self,
        episode: Episode,
        text_hint: Option<&str>,
    ) -> DomainMemoryStorageResult {
        // Classify domain from text hint or episode content
        let metadata = if self.auto_classify {
            let text = text_hint.unwrap_or("");
            self.classify_domain(text).unwrap_or_default()
        } else {
            DomainMemoryMetadata::default()
        };

        // Store the episode
        let id = self.memory.episodic.store(episode);

        // Store metadata
        let brain_processed = metadata.domain_id.is_some();
        self.episodic_metadata.insert(id, metadata.clone());

        DomainMemoryStorageResult {
            storage_id: id,
            metadata,
            brain_processed,
        }
    }

    /// Store a fact with automatic domain classification
    pub fn assert_with_domain(
        &mut self,
        fact: Graph,
        text_hint: Option<&str>,
    ) -> DomainMemoryStorageResult {
        // Classify domain from text hint
        let metadata = if let Some(hint) = text_hint.filter(|_| self.auto_classify) {
            self.classify_domain(hint).unwrap_or_default()
        } else {
            DomainMemoryMetadata::default()
        };

        // Store the fact
        let id = self.memory.semantic.assert(fact);

        // Store metadata
        let brain_processed = metadata.domain_id.is_some();
        self.semantic_metadata.insert(id, metadata.clone());

        DomainMemoryStorageResult {
            storage_id: id,
            metadata,
            brain_processed,
        }
    }

    /// Store a fact with explicit domain
    pub fn assert_in_domain(
        &mut self,
        fact: Graph,
        domain_id: &str,
        confidence: f32,
    ) -> DomainMemoryStorageResult {
        let metadata = DomainMemoryMetadata::for_domain(domain_id, confidence);
        let id = self.memory.semantic.assert(fact);
        self.semantic_metadata.insert(id, metadata.clone());

        DomainMemoryStorageResult {
            storage_id: id,
            metadata,
            brain_processed: false,
        }
    }

    /// Recall episodes from a specific domain
    pub fn recall_by_domain(
        &self,
        query: &Graph,
        domain_id: &str,
        limit: usize,
    ) -> DomainMemoryRetrievalResult {
        // Get episodes that match the query
        let all_matches = self.memory.episodic.recall(query, limit * 2);

        // Filter by domain
        let filtered: Vec<_> = all_matches.into_iter()
            .filter(|id| {
                self.episodic_metadata.get(id)
                    .and_then(|m| m.domain_id.as_ref())
                    .is_some_and(|d| d == domain_id)
            })
            .take(limit)
            .collect();

        DomainMemoryRetrievalResult {
            ids: filtered,
            routing_result: None,
            domain_filter: Some(domain_id.to_string()),
        }
    }

    /// Query facts from a specific domain
    pub fn query_by_domain(
        &self,
        pattern: &Graph,
        domain_id: &str,
        limit: usize,
    ) -> DomainMemoryRetrievalResult {
        // Get facts that match the pattern
        let all_matches = self.memory.semantic.query(pattern, limit * 2);

        // Filter by domain
        let filtered: Vec<_> = all_matches.into_iter()
            .filter(|id| {
                self.semantic_metadata.get(id)
                    .and_then(|m| m.domain_id.as_ref())
                    .is_some_and(|d| d == domain_id)
            })
            .take(limit)
            .collect();

        DomainMemoryRetrievalResult {
            ids: filtered,
            routing_result: None,
            domain_filter: Some(domain_id.to_string()),
        }
    }

    /// Recall episodes using brain routing
    ///
    /// Routes the query text to domain brains first, then retrieves
    /// memories from the matching domain
    pub fn recall_with_brain_routing(
        &self,
        query: &Graph,
        query_text: &str,
        limit: usize,
    ) -> DomainMemoryRetrievalResult {
        // Route to brains
        let routing = self.bridge.route_to_multiple_brains(query_text);

        // If a domain brain matched, filter by that domain
        if routing.success {
            if let Some(primary) = &routing.primary {
                let domain_id = primary.domain_id.clone();
                let domain_results = self.recall_by_domain(query, &domain_id, limit);
                return DomainMemoryRetrievalResult {
                    ids: domain_results.ids,
                    routing_result: Some(routing),
                    domain_filter: Some(domain_id),
                };
            }
        }

        // No domain match - return all matching episodes
        DomainMemoryRetrievalResult {
            ids: self.memory.episodic.recall(query, limit),
            routing_result: Some(routing),
            domain_filter: None,
        }
    }

    /// Get domain metadata for an episode
    pub fn get_episode_domain(&self, id: EpisodeId) -> Option<&DomainMemoryMetadata> {
        self.episodic_metadata.get(&id)
    }

    /// Get domain metadata for a fact
    pub fn get_fact_domain(&self, id: FactId) -> Option<&DomainMemoryMetadata> {
        self.semantic_metadata.get(&id)
    }

    /// Get all episodes in a domain
    pub fn episodes_in_domain(&self, domain_id: &str) -> Vec<EpisodeId> {
        self.episodic_metadata.iter()
            .filter(|(_, m)| m.domain_id.as_ref().is_some_and(|d| d == domain_id))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get all facts in a domain
    pub fn facts_in_domain(&self, domain_id: &str) -> Vec<FactId> {
        self.semantic_metadata.iter()
            .filter(|(_, m)| m.domain_id.as_ref().is_some_and(|d| d == domain_id))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get domain statistics
    pub fn domain_stats(&self) -> std::collections::HashMap<String, (usize, usize)> {
        let mut stats: std::collections::HashMap<String, (usize, usize)> = std::collections::HashMap::new();

        for meta in self.episodic_metadata.values() {
            if let Some(domain) = &meta.domain_id {
                let entry = stats.entry(domain.clone()).or_insert((0, 0));
                entry.0 += 1;
            }
        }

        for meta in self.semantic_metadata.values() {
            if let Some(domain) = &meta.domain_id {
                let entry = stats.entry(domain.clone()).or_insert((0, 0));
                entry.1 += 1;
            }
        }

        stats
    }

    /// Get all available domains from the registry
    pub fn available_domains(&self) -> Vec<String> {
        self.bridge.available_domains()
    }
}

impl CognitiveBrainBridge for DomainAwareMemory {
    fn get_registry(&self) -> &BrainRegistry {
        self.bridge.get_registry()
    }

    fn get_registry_mut(&mut self) -> &mut BrainRegistry {
        self.bridge.get_registry_mut()
    }
}

/// Factory function to create a domain-aware memory system with simple implementations
pub fn create_domain_aware_memory() -> DomainAwareMemory {
    DomainAwareMemory::new(create_default_memory_system())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph(text: &str) -> Graph {
        DagNN::from_text(text).unwrap()
    }

    #[test]
    fn test_episode_creation() {
        let context = make_test_graph("context");
        let content = make_test_graph("content");

        let episode = Episode::new(1, 1000, context, content)
            .with_valence(0.5)
            .with_importance(0.8)
            .with_tags(vec!["test".to_string()]);

        assert_eq!(episode.id, 1);
        assert_eq!(episode.timestamp, 1000);
        assert_eq!(episode.emotional_valence, 0.5);
        assert_eq!(episode.importance, 0.8);
        assert!(episode.tags.contains(&"test".to_string()));
    }

    #[test]
    fn test_episodic_memory_store_recall() {
        let mut memory = SimpleEpisodicMemory::new(None);

        let context = make_test_graph("context");
        let content = make_test_graph("hello");
        let episode = Episode::new(0, 1000, context, content);

        let id = memory.store(episode);
        assert_eq!(id, 1);
        assert_eq!(memory.len(), 1);

        let query = make_test_graph("hello");
        let results = memory.recall(&query, 5);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_episodic_memory_temporal_recall() {
        let mut memory = SimpleEpisodicMemory::new(None);

        for i in 0..5 {
            let context = make_test_graph("ctx");
            let content = make_test_graph(&format!("content{}", i));
            let episode = Episode::new(0, i * 100, context, content);
            memory.store(episode);
        }

        let results = memory.recall_temporal(100, 300);
        assert_eq!(results.len(), 3); // timestamps 100, 200, 300
    }

    #[test]
    fn test_semantic_graph_assert_query() {
        let mut graph = SimpleSemanticGraph::new();

        let fact = make_test_graph("fact1");
        let id = graph.assert(fact);
        assert_eq!(id, 0);

        let query = make_test_graph("fact1");
        let results = graph.query(&query, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_semantic_graph_contains() {
        let mut graph = SimpleSemanticGraph::new();

        let fact = make_test_graph("test_fact");
        graph.assert(fact);

        let query = make_test_graph("test_fact");
        assert!(graph.contains(&query));

        let other = make_test_graph("different_fact_completely_unrelated_long_text");
        assert!(!graph.contains(&other));
    }

    #[test]
    fn test_procedural_memory() {
        let mut memory = SimpleProceduralMemory::new();

        let rule = TransformRule::new(1, "test_rule");
        memory.learn("add_one", rule);

        assert!(memory.contains("add_one"));
        assert!(!memory.contains("unknown"));

        let recalled = memory.recall("add_one");
        assert!(recalled.is_some());

        // Test refinement
        memory.refine("add_one", ProcedureFeedback::success(0.9));
        let stats = memory.stats("add_one").unwrap();
        assert_eq!(stats.applications, 1);
        assert_eq!(stats.successes, 1);
    }

    #[test]
    fn test_working_memory_capacity() {
        let mut wm = SimpleWorkingMemory::new(3);

        // Fill to capacity
        wm.attend(make_test_graph("a"));
        wm.attend(make_test_graph("b"));
        wm.attend(make_test_graph("c"));

        assert!(wm.is_full());
        assert_eq!(wm.len(), 3);

        // Adding another should evict
        let evicted = wm.attend(make_test_graph("d"));
        assert!(evicted); // true means an item was evicted
        assert_eq!(wm.len(), 3);
    }

    #[test]
    fn test_working_memory_focus() {
        let mut wm = SimpleWorkingMemory::new(5);

        wm.attend(make_test_graph("first"));
        wm.attend(make_test_graph("second"));
        wm.attend(make_test_graph("third"));

        // Focus on first item (index 0)
        assert!(wm.focus(0));

        // Now the formerly-first item should be last (current)
        let current = wm.current().unwrap();
        assert_eq!(current.to_text(), "first");
    }

    #[test]
    fn test_continual_learning() {
        let mut cl = SimpleContinualLearning::new(100);

        for i in 0..10 {
            cl.consolidate(make_test_graph(&format!("exp{}", i)));
        }

        let stats = cl.stats();
        assert_eq!(stats.experiences_integrated, 10);

        cl.replay_and_integrate();
        assert_eq!(cl.stats().consolidations, 1);
    }

    #[test]
    fn test_graph_fingerprint() {
        let g1 = make_test_graph("hello");
        let g2 = make_test_graph("hello");
        let g3 = make_test_graph("completely different text here");

        let fp1 = GraphFingerprint::from_graph(&g1);
        let fp2 = GraphFingerprint::from_graph(&g2);
        let fp3 = GraphFingerprint::from_graph(&g3);

        // Same graphs should have high similarity
        assert!(fp1.similarity(&fp2) > 0.99);

        // Different graphs should have lower similarity
        assert!(fp1.similarity(&fp3) < 0.5);
    }

    #[test]
    fn test_memory_system_creation() {
        let system = create_default_memory_system();

        assert_eq!(system.episodic.len(), 0);
        assert_eq!(system.semantic.len(), 0);
        assert_eq!(system.procedural.len(), 0);
        assert_eq!(system.working.len(), 0);
    }

    #[test]
    fn test_retention_policy() {
        let mut memory = SimpleEpisodicMemory::new(None);

        // Add episodes with varying importance
        for i in 0..10 {
            let episode = Episode::new(0, i * 100, make_test_graph("ctx"), make_test_graph("cnt"))
                .with_importance(i as f32 / 10.0);
            memory.store(episode);
        }

        assert_eq!(memory.len(), 10);

        let policy = RetentionPolicy {
            min_importance: 0.5,
            max_episodes: Some(5),
            ..Default::default()
        };

        memory.consolidate(&policy);

        // Should have kept only high-importance episodes
        assert!(memory.len() <= 5);
    }

    #[test]
    fn test_procedure_stats() {
        let mut stats = ProcedureStats::default();
        assert_eq!(stats.success_rate(), 0.0);

        stats.applications = 10;
        stats.successes = 8;
        assert!((stats.success_rate() - 0.8).abs() < 0.001);
    }
}
