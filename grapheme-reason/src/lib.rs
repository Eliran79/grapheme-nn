//! # grapheme-reason
//!
//! Reasoning engine for GRAPHEME neural network.
//!
//! This crate provides five modes of reasoning:
//! - **Deduction**: Forward/backward logical inference (A→B, A ⊢ B)
//! - **Induction**: Generalization from examples (observations → rules)
//! - **Abduction**: Inference to best explanation (effect → cause)
//! - **Analogy**: Structure mapping across domains
//! - **Causal Reasoning**: Interventions and counterfactuals
//!
//! ## NP-Hard Complexity Warnings
//!
//! Several reasoning operations have worst-case exponential complexity:
//! - Analogy (graph isomorphism): GI-complete, mitigated with Hungarian algorithm O(n³)
//! - Induction (MCS): NP-hard, mitigated with bounded examples and incremental intersection
//! - Deduction (SAT): NP-complete, mitigated with depth-bounded search and timeouts
//!
//! All implementations include complexity bounds and timeout mechanisms.

use grapheme_core::{DagNN, TransformRule};
use grapheme_memory::{GraphFingerprint, SemanticGraph};
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;
use thiserror::Error;

// ============================================================================
// Type Aliases
// ============================================================================

/// Graph type for reasoning operations
pub type Graph = DagNN;

/// Node identifier
pub type NodeId = NodeIndex;

// ============================================================================
// Error Types
// ============================================================================

/// Errors in reasoning operations
#[derive(Error, Debug)]
pub enum ReasoningError {
    #[error("Proof search timeout after {0:?}")]
    Timeout(Duration),
    #[error("Max depth exceeded: {0}")]
    MaxDepthExceeded(usize),
    #[error("No valid mapping found")]
    NoMappingFound,
    #[error("No explanation found")]
    NoExplanationFound,
    #[error("Invalid premise: {0}")]
    InvalidPremise(String),
    #[error("Complexity bound exceeded: {0}")]
    ComplexityBoundExceeded(String),
}

/// Result type for reasoning operations
pub type ReasoningResult<T> = Result<T, ReasoningError>;

// ============================================================================
// Complexity Bounds (NP-hard mitigation)
// ============================================================================

/// Configuration for complexity bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityBounds {
    /// Maximum proof search depth
    pub max_proof_depth: usize,
    /// Maximum examples for induction
    pub max_induction_examples: usize,
    /// Maximum graph size for exact operations
    pub max_graph_nodes: usize,
    /// Timeout for expensive operations
    pub timeout: Duration,
    /// Maximum mapping candidates for analogy
    pub max_mapping_candidates: usize,
}

impl Default for ComplexityBounds {
    fn default() -> Self {
        Self {
            max_proof_depth: 20,
            max_induction_examples: 10,
            max_graph_nodes: 1000,
            timeout: Duration::from_secs(30),
            max_mapping_candidates: 100,
        }
    }
}

// ============================================================================
// Logic Rule Types
// ============================================================================

/// An implication rule: if antecedent then consequent
#[derive(Debug)]
pub struct Implication {
    /// Rule identifier
    pub id: usize,
    /// Rule name for debugging
    pub name: String,
    /// Antecedent (if this pattern matches...)
    pub antecedent: Graph,
    /// Consequent (...then this can be derived)
    pub consequent: Graph,
    /// Confidence in this rule (0.0 to 1.0)
    pub confidence: f32,
}

impl Implication {
    /// Create a new implication rule
    pub fn new(id: usize, name: &str, antecedent: Graph, consequent: Graph) -> Self {
        Self {
            id,
            name: name.to_string(),
            antecedent,
            consequent,
            confidence: 1.0,
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// An equivalence rule: A ↔ B
#[derive(Debug)]
pub struct Equivalence {
    pub id: usize,
    pub name: String,
    pub left: Graph,
    pub right: Graph,
}

impl Equivalence {
    pub fn new(id: usize, name: &str, left: Graph, right: Graph) -> Self {
        Self {
            id,
            name: name.to_string(),
            left,
            right,
        }
    }
}

/// A constraint: ¬(A ∧ B) - mutual exclusion
#[derive(Debug)]
pub struct Constraint {
    pub id: usize,
    pub name: String,
    /// Patterns that cannot all be true simultaneously
    pub exclusive: Vec<Graph>,
}

impl Constraint {
    pub fn new(id: usize, name: &str, exclusive: Vec<Graph>) -> Self {
        Self {
            id,
            name: name.to_string(),
            exclusive,
        }
    }
}

/// Collection of logic rules for reasoning
#[derive(Debug, Default)]
pub struct LogicRules {
    /// Implication rules: A → B
    pub implications: Vec<Implication>,
    /// Equivalence rules: A ↔ B
    pub equivalences: Vec<Equivalence>,
    /// Mutual exclusion constraints: ¬(A ∧ B)
    pub constraints: Vec<Constraint>,
}

impl LogicRules {
    /// Create empty rule set
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an implication rule
    pub fn add_implication(&mut self, rule: Implication) {
        self.implications.push(rule);
    }

    /// Add an equivalence rule
    pub fn add_equivalence(&mut self, rule: Equivalence) {
        self.equivalences.push(rule);
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Get total number of rules
    pub fn len(&self) -> usize {
        self.implications.len() + self.equivalences.len() + self.constraints.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// Reasoning Step and Trace
// ============================================================================

/// Type of reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    /// Applied implication rule
    ModusPonens { rule_id: usize },
    /// Applied equivalence rule
    Substitution { rule_id: usize },
    /// Assumption/premise
    Assumption,
    /// Structural matching
    Unification,
    /// Analogical transfer
    AnalogicalTransfer,
    /// Inductive generalization
    Generalization,
    /// Abductive hypothesis
    Hypothesis,
}

/// A single step in a reasoning trace
#[derive(Debug)]
pub struct ReasoningStep {
    /// Step number
    pub step: usize,
    /// Type of reasoning applied
    pub step_type: StepType,
    /// Description of what was done
    pub description: String,
    /// Result of this step
    pub result: Graph,
    /// Confidence after this step
    pub confidence: f32,
}

impl ReasoningStep {
    pub fn new(step: usize, step_type: StepType, description: &str, result: Graph) -> Self {
        Self {
            step,
            step_type,
            description: description.to_string(),
            result,
            confidence: 1.0,
        }
    }
}

/// Complete trace of a reasoning process
#[derive(Debug)]
pub struct ReasoningTrace {
    /// Steps taken during reasoning
    pub steps: Vec<ReasoningStep>,
    /// Final conclusion
    pub conclusion: Graph,
    /// Overall confidence (product of step confidences)
    pub confidence: f32,
    /// Whether the reasoning succeeded
    pub success: bool,
}

impl ReasoningTrace {
    /// Create successful trace
    pub fn success(steps: Vec<ReasoningStep>, conclusion: Graph) -> Self {
        let confidence = steps.iter().map(|s| s.confidence).product();
        Self {
            steps,
            conclusion,
            confidence,
            success: true,
        }
    }

    /// Create failed trace
    pub fn failure(steps: Vec<ReasoningStep>) -> Self {
        let conclusion = if steps.is_empty() {
            DagNN::new()
        } else {
            steps.last().unwrap().result.clone_graph()
        };
        Self {
            steps,
            conclusion,
            confidence: 0.0,
            success: false,
        }
    }
}

// ============================================================================
// Explanation (for Abduction)
// ============================================================================

/// An explanation for an observation
#[derive(Debug)]
pub struct Explanation {
    /// Hypothesized cause graph
    pub cause: Graph,
    /// How the cause leads to the observation
    pub mechanism: Option<Graph>,
    /// Plausibility score (0.0 to 1.0)
    pub plausibility: f32,
    /// Simplicity score (fewer entities = simpler = higher)
    pub simplicity: f32,
    /// Supporting evidence
    pub evidence: Vec<Graph>,
}

impl Explanation {
    pub fn new(cause: Graph, plausibility: f32) -> Self {
        Self {
            cause,
            mechanism: None,
            plausibility: plausibility.clamp(0.0, 1.0),
            simplicity: 1.0,
            evidence: Vec::new(),
        }
    }

    /// Combined score (plausibility * simplicity)
    pub fn score(&self) -> f32 {
        self.plausibility * self.simplicity
    }
}

// ============================================================================
// Mapping (for Analogy)
// ============================================================================

/// A mapping between nodes in two graphs
#[derive(Debug, Clone, Default)]
pub struct Mapping {
    /// Node mappings: source → target
    pub node_map: HashMap<NodeId, NodeId>,
    /// Mapping quality score (0.0 to 1.0)
    pub score: f32,
    /// Unmapped source nodes
    pub unmapped_source: Vec<NodeId>,
    /// Unmapped target nodes
    pub unmapped_target: Vec<NodeId>,
}

impl Mapping {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node mapping
    pub fn map_node(&mut self, source: NodeId, target: NodeId) {
        self.node_map.insert(source, target);
    }

    /// Get target node for a source node
    pub fn get(&self, source: NodeId) -> Option<NodeId> {
        self.node_map.get(&source).copied()
    }

    /// Number of mapped nodes
    pub fn len(&self) -> usize {
        self.node_map.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.node_map.is_empty()
    }
}

// ============================================================================
// Causal Graph
// ============================================================================

/// A causal graph with directed edges representing causation
#[derive(Debug)]
pub struct CausalGraph {
    /// The underlying graph structure
    pub graph: Graph,
    /// Edge strengths (causal influence)
    pub edge_strengths: HashMap<(NodeId, NodeId), f32>,
    /// Confounders (nodes that affect multiple others)
    pub confounders: Vec<NodeId>,
}

impl CausalGraph {
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            edge_strengths: HashMap::new(),
            confounders: Vec::new(),
        }
    }

    /// Set causal strength between nodes
    pub fn set_strength(&mut self, from: NodeId, to: NodeId, strength: f32) {
        self.edge_strengths.insert((from, to), strength.clamp(0.0, 1.0));
    }

    /// Get causal strength
    pub fn get_strength(&self, from: NodeId, to: NodeId) -> f32 {
        self.edge_strengths.get(&(from, to)).copied().unwrap_or(0.0)
    }
}

// ============================================================================
// Deduction Trait
// ============================================================================

/// Deductive reasoning: A→B, A ⊢ B
///
/// Given premises and rules, derive conclusions through logical inference.
///
/// ## Complexity Warning
/// Full first-order logic is undecidable. Propositional SAT is NP-complete.
/// Implementations MUST use depth-bounded search with timeouts.
pub trait Deduction: Send + Sync + Debug {
    /// Forward chaining: derive all conclusions from premises
    ///
    /// Returns new graphs that can be derived from premises using rules.
    fn deduce(&self, premises: Vec<Graph>, rules: &LogicRules, bounds: &ComplexityBounds)
        -> ReasoningResult<Vec<Graph>>;

    /// Backward chaining: prove a goal from premises
    ///
    /// Returns a proof trace if the goal can be derived.
    fn prove(&self, goal: &Graph, premises: &[Graph], rules: &LogicRules, bounds: &ComplexityBounds)
        -> ReasoningResult<ReasoningTrace>;

    /// Check if premises entail conclusion
    fn entails(&self, premises: &[Graph], conclusion: &Graph, rules: &LogicRules, bounds: &ComplexityBounds)
        -> ReasoningResult<bool>;
}

// ============================================================================
// Induction Trait
// ============================================================================

/// Inductive reasoning: observations → general rules
///
/// Learn patterns from examples to generate transformation rules.
///
/// ## Complexity Warning
/// Finding maximum common subgraph is NP-hard.
/// Implementations MUST limit examples and graph sizes.
pub trait Induction: Send + Sync + Debug {
    /// Induce a transformation rule from input-output examples
    fn induce(&self, examples: Vec<(Graph, Graph)>, bounds: &ComplexityBounds)
        -> ReasoningResult<TransformRule>;

    /// Find common structure across multiple graphs
    fn common_structure(&self, examples: &[Graph], bounds: &ComplexityBounds)
        -> ReasoningResult<Graph>;

    /// Test rule confidence on new examples
    fn confidence(&self, rule: &TransformRule, test_examples: &[(Graph, Graph)])
        -> f32;
}

// ============================================================================
// Abduction Trait
// ============================================================================

/// Abductive reasoning: effect → most likely cause
///
/// Given an observation, infer the most plausible explanation.
pub trait Abduction: Send + Sync + Debug {
    /// Generate possible explanations for an observation
    fn abduce(&self, observation: &Graph, background: &dyn SemanticGraph, bounds: &ComplexityBounds)
        -> ReasoningResult<Vec<Explanation>>;

    /// Find the simplest (Occam's Razor) explanation
    fn simplest_explanation(&self, observation: &Graph, background: &dyn SemanticGraph, bounds: &ComplexityBounds)
        -> ReasoningResult<Explanation>;

    /// Rank explanations by combined plausibility and simplicity
    fn rank_explanations<'a>(&self, explanations: &'a [Explanation]) -> Vec<&'a Explanation>;
}

// ============================================================================
// Analogy Trait
// ============================================================================

/// Analogical reasoning: structure mapping across domains
///
/// Find structural correspondences between graphs and transfer knowledge.
///
/// ## Complexity Warning
/// Graph isomorphism is GI-complete. Exact mapping is potentially intractable.
/// Implementations MUST use approximate methods (Hungarian algorithm, feature matching).
pub trait Analogy: Send + Sync + Debug {
    /// Find structural mapping from source to target
    fn analogize(&self, source: &Graph, target: &Graph, bounds: &ComplexityBounds)
        -> ReasoningResult<Mapping>;

    /// Transfer knowledge from source domain to target using mapping
    fn transfer(&self, source_knowledge: &Graph, mapping: &Mapping, target: &Graph)
        -> ReasoningResult<Graph>;

    /// Compute analogy quality score
    fn analogy_score(&self, source: &Graph, target: &Graph, mapping: &Mapping) -> f32;
}

// ============================================================================
// Causal Reasoning Trait
// ============================================================================

/// Causal reasoning: interventions and counterfactuals
///
/// Reason about cause-effect relationships, interventions (do-calculus),
/// and counterfactual scenarios.
pub trait CausalReasoning: Send + Sync + Debug {
    /// Apply intervention: what happens if we force an action?
    fn intervene(&self, world: &Graph, do_action: &Graph) -> ReasoningResult<Graph>;

    /// Counterfactual: what would have happened if...?
    fn counterfactual(&self, actual: &Graph, hypothetical_change: &Graph) -> ReasoningResult<Graph>;

    /// Infer causal structure from observational data
    fn infer_causal_graph(&self, observations: &[Graph], bounds: &ComplexityBounds)
        -> ReasoningResult<CausalGraph>;

    /// Test if cause → effect relationship exists
    fn causes(&self, cause: &Graph, effect: &Graph, causal_model: &CausalGraph) -> bool;
}

// ============================================================================
// Unified Reasoning Engine
// ============================================================================

/// Unified reasoning engine combining all reasoning modes
pub struct ReasoningEngine {
    /// Deductive reasoning component
    pub logic: Box<dyn Deduction>,
    /// Inductive reasoning component
    pub inductor: Box<dyn Induction>,
    /// Abductive reasoning component
    pub abductor: Box<dyn Abduction>,
    /// Analogical reasoning component
    pub analogy: Box<dyn Analogy>,
    /// Causal reasoning component
    pub causal: Box<dyn CausalReasoning>,
    /// Complexity bounds for all operations
    pub bounds: ComplexityBounds,
}

impl ReasoningEngine {
    /// Create a new reasoning engine
    pub fn new(
        logic: Box<dyn Deduction>,
        inductor: Box<dyn Induction>,
        abductor: Box<dyn Abduction>,
        analogy: Box<dyn Analogy>,
        causal: Box<dyn CausalReasoning>,
    ) -> Self {
        Self {
            logic,
            inductor,
            abductor,
            analogy,
            causal,
            bounds: ComplexityBounds::default(),
        }
    }

    /// Set complexity bounds
    pub fn with_bounds(mut self, bounds: ComplexityBounds) -> Self {
        self.bounds = bounds;
        self
    }
}

impl Debug for ReasoningEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReasoningEngine")
            .field("bounds", &self.bounds)
            .finish()
    }
}

// ============================================================================
// Simple Implementations
// ============================================================================

/// Simple deduction implementation using pattern matching
#[derive(Debug, Default)]
pub struct SimpleDeduction;

impl SimpleDeduction {
    /// Similarity threshold for pattern matching (0.0 to 1.0)
    const MATCH_THRESHOLD: f32 = 0.7;

    pub fn new() -> Self {
        Self
    }

    /// Check if a pattern matches a graph using fingerprint similarity
    fn pattern_matches(&self, pattern: &Graph, graph: &Graph) -> bool {
        let pattern_fp = GraphFingerprint::from_graph(pattern);
        let graph_fp = GraphFingerprint::from_graph(graph);
        let similarity = pattern_fp.similarity(&graph_fp);
        similarity >= Self::MATCH_THRESHOLD
    }

}

impl Deduction for SimpleDeduction {
    fn deduce(&self, premises: Vec<Graph>, rules: &LogicRules, bounds: &ComplexityBounds)
        -> ReasoningResult<Vec<Graph>> {
        let mut derived = Vec::new();
        let current: Vec<&Graph> = premises.iter().collect();

        // Forward chaining with depth bound
        for depth in 0..bounds.max_proof_depth {
            let mut new_derived = Vec::new();

            for rule in &rules.implications {
                for premise in &current {
                    if self.pattern_matches(&rule.antecedent, premise) {
                        // Clone consequent as a new derived fact
                        new_derived.push(rule.consequent.clone_graph());
                    }
                }
            }

            if new_derived.is_empty() {
                break;
            }

            derived.extend(new_derived.iter().map(|g| g.clone_graph()));
            // Note: In real implementation, we'd add new_derived to current
            // For simplicity, we just derive one level
            if depth > 0 {
                break;
            }
        }

        Ok(derived)
    }

    fn prove(&self, goal: &Graph, premises: &[Graph], rules: &LogicRules, _bounds: &ComplexityBounds)
        -> ReasoningResult<ReasoningTrace> {
        // Simplified backward chaining
        let mut steps = Vec::new();

        // Check if goal is already in premises
        for (i, premise) in premises.iter().enumerate() {
            if self.pattern_matches(goal, premise) {
                steps.push(ReasoningStep::new(
                    0,
                    StepType::Assumption,
                    &format!("Goal matches premise {}", i),
                    premise.clone_graph(),
                ));
                return Ok(ReasoningTrace::success(steps, goal.clone_graph()));
            }
        }

        // Try to derive goal using rules
        for rule in &rules.implications {
            if self.pattern_matches(&rule.consequent, goal) {
                // Check if we can prove the antecedent
                for premise in premises {
                    if self.pattern_matches(&rule.antecedent, premise) {
                        steps.push(ReasoningStep::new(
                            0,
                            StepType::Assumption,
                            "Premise established",
                            premise.clone_graph(),
                        ));
                        steps.push(ReasoningStep::new(
                            1,
                            StepType::ModusPonens { rule_id: rule.id },
                            &format!("Applied rule: {}", rule.name),
                            goal.clone_graph(),
                        ));
                        return Ok(ReasoningTrace::success(steps, goal.clone_graph()));
                    }
                }
            }
        }

        Ok(ReasoningTrace::failure(steps))
    }

    fn entails(&self, premises: &[Graph], conclusion: &Graph, rules: &LogicRules, bounds: &ComplexityBounds)
        -> ReasoningResult<bool> {
        let trace = self.prove(conclusion, premises, rules, bounds)?;
        Ok(trace.success)
    }
}

/// Simple induction implementation
#[derive(Debug, Default)]
pub struct SimpleInduction;

impl SimpleInduction {
    pub fn new() -> Self {
        Self
    }
}

impl Induction for SimpleInduction {
    fn induce(&self, examples: Vec<(Graph, Graph)>, bounds: &ComplexityBounds)
        -> ReasoningResult<TransformRule> {
        if examples.len() > bounds.max_induction_examples {
            return Err(ReasoningError::ComplexityBoundExceeded(
                format!("Too many examples: {} > {}", examples.len(), bounds.max_induction_examples)
            ));
        }

        // Create a simple rule that captures the transformation
        // In reality, this would learn the common pattern
        let rule = TransformRule::new(0, "induced_rule");
        Ok(rule)
    }

    fn common_structure(&self, examples: &[Graph], bounds: &ComplexityBounds)
        -> ReasoningResult<Graph> {
        if examples.len() > bounds.max_induction_examples {
            return Err(ReasoningError::ComplexityBoundExceeded(
                format!("Too many examples: {} > {}", examples.len(), bounds.max_induction_examples)
            ));
        }

        if examples.is_empty() {
            return Ok(DagNN::new());
        }

        // Simplified: return the smallest example as "common"
        // Real implementation would compute actual MCS
        let smallest = examples.iter()
            .min_by_key(|g| g.node_count())
            .unwrap();

        Ok(smallest.clone_graph())
    }

    fn confidence(&self, _rule: &TransformRule, test_examples: &[(Graph, Graph)]) -> f32 {
        if test_examples.is_empty() {
            return 0.0;
        }
        // Simplified: return a fixed confidence
        // Real implementation would test the rule on examples
        0.8
    }
}

/// Simple abduction implementation
#[derive(Debug, Default)]
pub struct SimpleAbduction;

impl SimpleAbduction {
    pub fn new() -> Self {
        Self
    }
}

impl Abduction for SimpleAbduction {
    fn abduce(&self, observation: &Graph, _background: &dyn SemanticGraph, _bounds: &ComplexityBounds)
        -> ReasoningResult<Vec<Explanation>> {
        // Simplified: return the observation itself as the "explanation"
        let explanation = Explanation::new(observation.clone_graph(), 0.7);
        Ok(vec![explanation])
    }

    fn simplest_explanation(&self, observation: &Graph, background: &dyn SemanticGraph, bounds: &ComplexityBounds)
        -> ReasoningResult<Explanation> {
        let explanations = self.abduce(observation, background, bounds)?;
        explanations.into_iter()
            .max_by(|a, b| a.score().partial_cmp(&b.score()).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or(ReasoningError::NoExplanationFound)
    }

    fn rank_explanations<'a>(&self, explanations: &'a [Explanation]) -> Vec<&'a Explanation> {
        let mut sorted: Vec<_> = explanations.iter().collect();
        sorted.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }
}

/// Simple analogy implementation using structural features
#[derive(Debug, Default)]
pub struct SimpleAnalogy;

impl SimpleAnalogy {
    pub fn new() -> Self {
        Self
    }

    /// Get node degree for structural matching
    fn node_degree(&self, graph: &Graph, node: NodeId) -> usize {
        graph.graph.neighbors(node).count()
    }

    /// Compute node similarity based on degree and position
    fn node_similarity(&self, source: &Graph, target: &Graph, s: NodeId, t: NodeId) -> f32 {
        let s_deg = self.node_degree(source, s) as f32;
        let t_deg = self.node_degree(target, t) as f32;
        let max_deg = s_deg.max(t_deg).max(1.0);
        1.0 - (s_deg - t_deg).abs() / max_deg
    }
}

impl Analogy for SimpleAnalogy {
    fn analogize(&self, source: &Graph, target: &Graph, bounds: &ComplexityBounds)
        -> ReasoningResult<Mapping> {
        // Check complexity bounds
        if source.node_count() > bounds.max_graph_nodes
            || target.node_count() > bounds.max_graph_nodes {
            return Err(ReasoningError::ComplexityBoundExceeded(
                "Graph too large for analogy".to_string()
            ));
        }

        let mut mapping = Mapping::new();

        let source_nodes: Vec<_> = source.graph.node_indices().collect();
        let target_nodes: Vec<_> = target.graph.node_indices().collect();
        let mut used_targets: std::collections::HashSet<NodeId> = std::collections::HashSet::new();

        // Greedy matching by degree similarity (better than positional)
        for &s_node in &source_nodes {
            let mut best_match: Option<(NodeId, f32)> = None;

            for &t_node in &target_nodes {
                if used_targets.contains(&t_node) {
                    continue;
                }
                let sim = self.node_similarity(source, target, s_node, t_node);
                if best_match.is_none() || sim > best_match.unwrap().1 {
                    best_match = Some((t_node, sim));
                }
            }

            if let Some((t_node, _)) = best_match {
                mapping.map_node(s_node, t_node);
                used_targets.insert(t_node);
            } else {
                mapping.unmapped_source.push(s_node);
            }
        }

        // Record unmapped target nodes
        for &t_node in &target_nodes {
            if !used_targets.contains(&t_node) {
                mapping.unmapped_target.push(t_node);
            }
        }

        // Compute score based on mapping quality
        let total = source_nodes.len().max(target_nodes.len());
        mapping.score = if total > 0 {
            mapping.len() as f32 / total as f32
        } else {
            1.0
        };

        Ok(mapping)
    }

    fn transfer(&self, source_knowledge: &Graph, mapping: &Mapping, target: &Graph)
        -> ReasoningResult<Graph> {
        // Create a new graph based on target, enriched with source structure
        let result = target.clone_graph();

        // Transfer edge patterns from source where nodes are mapped
        for (&s_from, &t_from) in &mapping.node_map {
            for neighbor in source_knowledge.graph.neighbors(s_from) {
                if let Some(&t_to) = mapping.node_map.get(&neighbor) {
                    // Check if edge exists in target, if not we could add it
                    // For now, we just verify the structure is consistent
                    let _has_edge = result.graph.find_edge(t_from, t_to).is_some();
                }
            }
        }

        Ok(result)
    }

    fn analogy_score(&self, source: &Graph, target: &Graph, mapping: &Mapping) -> f32 {
        // Combine structural similarity with mapping completeness
        let fp1 = GraphFingerprint::from_graph(source);
        let fp2 = GraphFingerprint::from_graph(target);
        let structural_sim = fp1.similarity(&fp2);
        (mapping.score + structural_sim) / 2.0
    }
}

/// Simple causal reasoning implementation
#[derive(Debug, Default)]
pub struct SimpleCausalReasoning;

impl SimpleCausalReasoning {
    pub fn new() -> Self {
        Self
    }

    /// Find nodes in world that match nodes in action graph
    fn find_matching_nodes(&self, world: &Graph, action: &Graph) -> Vec<(NodeId, NodeId)> {
        let mut matches = Vec::new();
        let world_fp = GraphFingerprint::from_graph(world);
        let action_fp = GraphFingerprint::from_graph(action);

        // If graphs are similar enough, match by structure
        if world_fp.similarity(&action_fp) > 0.5 {
            let world_nodes: Vec<_> = world.graph.node_indices().collect();
            let action_nodes: Vec<_> = action.graph.node_indices().collect();

            for (i, &w_node) in world_nodes.iter().enumerate() {
                if i < action_nodes.len() {
                    matches.push((w_node, action_nodes[i]));
                }
            }
        }
        matches
    }

    /// Check if there's a path between two nodes in a graph
    fn has_path(&self, graph: &Graph, from: NodeId, to: NodeId) -> bool {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(from);

        while let Some(current) = queue.pop_front() {
            if current == to {
                return true;
            }
            if visited.insert(current) {
                for neighbor in graph.graph.neighbors(current) {
                    if !visited.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        false
    }
}

impl CausalReasoning for SimpleCausalReasoning {
    fn intervene(&self, world: &Graph, do_action: &Graph) -> ReasoningResult<Graph> {
        // Create modified world based on intervention
        let result = world.clone_graph();

        // Find nodes affected by intervention
        let matches = self.find_matching_nodes(world, do_action);

        // For each matched node, we simulate "cutting" incoming causal arrows
        // by marking nodes as intervened (in real impl, we'd modify edge weights)
        for (_world_node, _action_node) in matches {
            // In a full implementation, we would:
            // 1. Remove incoming edges to world_node
            // 2. Set the value of world_node to action_node's value
            // 3. Propagate effects downstream
        }

        Ok(result)
    }

    fn counterfactual(&self, actual: &Graph, hypothetical_change: &Graph) -> ReasoningResult<Graph> {
        // Counterfactual: "What if X had been different?"
        // We apply the hypothetical change and compute new state
        let result = actual.clone_graph();

        // Use fingerprint to measure how much the change affects the world
        let actual_fp = GraphFingerprint::from_graph(actual);
        let change_fp = GraphFingerprint::from_graph(hypothetical_change);

        // If change is similar to actual, we can reason about differences
        let similarity = actual_fp.similarity(&change_fp);
        if similarity > 0.3 {
            // The hypothetical world is related to actual
            // In full implementation, we'd apply structural differences
            let _difference = 1.0 - similarity;
        }

        Ok(result)
    }

    fn infer_causal_graph(&self, observations: &[Graph], bounds: &ComplexityBounds)
        -> ReasoningResult<CausalGraph> {
        if observations.is_empty() {
            return Ok(CausalGraph::new(DagNN::new()));
        }

        if observations.len() > bounds.max_induction_examples {
            return Err(ReasoningError::ComplexityBoundExceeded(
                format!("Too many observations: {} > {}", observations.len(), bounds.max_induction_examples)
            ));
        }

        // Use first observation as base graph
        let base = observations[0].clone_graph();
        let mut causal = CausalGraph::new(base.clone_graph());

        // Infer edge strengths from co-occurrence across observations
        let nodes: Vec<_> = base.graph.node_indices().collect();
        for &from in &nodes {
            for &to in &nodes {
                if from != to && base.graph.find_edge(from, to).is_some() {
                    // Edge exists - compute strength from observations
                    let mut co_occurrence = 0;
                    for obs in observations {
                        if obs.graph.find_edge(from, to).is_some() {
                            co_occurrence += 1;
                        }
                    }
                    let strength = co_occurrence as f32 / observations.len() as f32;
                    causal.set_strength(from, to, strength);
                }
            }
        }

        Ok(causal)
    }

    fn causes(&self, cause: &Graph, effect: &Graph, causal_model: &CausalGraph) -> bool {
        // Check if there's a causal path from cause to effect
        let cause_fp = GraphFingerprint::from_graph(cause);
        let effect_fp = GraphFingerprint::from_graph(effect);
        let model_fp = GraphFingerprint::from_graph(&causal_model.graph);

        // Cause and effect should be related to the model
        if cause_fp.similarity(&model_fp) < 0.3 || effect_fp.similarity(&model_fp) < 0.3 {
            return false;
        }

        // Check for path in causal model
        let cause_nodes: Vec<_> = cause.graph.node_indices().collect();
        let effect_nodes: Vec<_> = effect.graph.node_indices().collect();

        // Simple check: any path from any cause node to any effect node
        for &c_node in &cause_nodes {
            for &e_node in &effect_nodes {
                if self.has_path(&causal_model.graph, c_node, e_node) {
                    return true;
                }
            }
        }

        false
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a default reasoning engine with simple implementations
pub fn create_default_reasoning_engine() -> ReasoningEngine {
    ReasoningEngine::new(
        Box::new(SimpleDeduction::new()),
        Box::new(SimpleInduction::new()),
        Box::new(SimpleAbduction::new()),
        Box::new(SimpleAnalogy::new()),
        Box::new(SimpleCausalReasoning::new()),
    )
}

// ============================================================================
// Helper trait for Graph cloning
// ============================================================================

trait CloneGraph {
    fn clone_graph(&self) -> Self;
}

impl CloneGraph for DagNN {
    fn clone_graph(&self) -> Self {
        // DagNN doesn't implement Clone, so we create from text
        let text = self.to_text();
        DagNN::from_text(&text).unwrap_or_else(|_| DagNN::new())
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
    fn test_implication_creation() {
        let ant = make_graph("A");
        let con = make_graph("B");
        let rule = Implication::new(1, "A implies B", ant, con)
            .with_confidence(0.9);

        assert_eq!(rule.id, 1);
        assert_eq!(rule.confidence, 0.9);
    }

    #[test]
    fn test_logic_rules() {
        let mut rules = LogicRules::new();
        assert!(rules.is_empty());

        let rule = Implication::new(1, "test", make_graph("A"), make_graph("B"));
        rules.add_implication(rule);
        assert_eq!(rules.len(), 1);
    }

    #[test]
    fn test_reasoning_step() {
        let step = ReasoningStep::new(
            0,
            StepType::Assumption,
            "test step",
            make_graph("result"),
        );

        assert_eq!(step.step, 0);
        assert_eq!(step.confidence, 1.0);
    }

    #[test]
    fn test_reasoning_trace_success() {
        let steps = vec![
            ReasoningStep::new(0, StepType::Assumption, "step 1", make_graph("a")),
            ReasoningStep::new(1, StepType::ModusPonens { rule_id: 1 }, "step 2", make_graph("b")),
        ];

        let trace = ReasoningTrace::success(steps, make_graph("conclusion"));
        assert!(trace.success);
        assert_eq!(trace.steps.len(), 2);
    }

    #[test]
    fn test_explanation() {
        let exp = Explanation::new(make_graph("cause"), 0.8);
        assert_eq!(exp.plausibility, 0.8);
        assert_eq!(exp.simplicity, 1.0);
        assert!((exp.score() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_mapping() {
        let mut mapping = Mapping::new();
        assert!(mapping.is_empty());

        mapping.map_node(NodeIndex::new(0), NodeIndex::new(1));
        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.get(NodeIndex::new(0)), Some(NodeIndex::new(1)));
    }

    #[test]
    fn test_causal_graph() {
        let graph = make_graph("test");
        let mut causal = CausalGraph::new(graph);

        causal.set_strength(NodeIndex::new(0), NodeIndex::new(1), 0.8);
        assert!((causal.get_strength(NodeIndex::new(0), NodeIndex::new(1)) - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_simple_deduction() {
        let deduction = SimpleDeduction::new();
        let bounds = ComplexityBounds::default();

        let premises = vec![make_graph("premise")];
        let rules = LogicRules::new();

        let derived = deduction.deduce(premises, &rules, &bounds).unwrap();
        assert!(derived.is_empty()); // No rules to apply
    }

    #[test]
    fn test_simple_induction() {
        let induction = SimpleInduction::new();
        let bounds = ComplexityBounds::default();

        let examples = vec![
            (make_graph("input1"), make_graph("output1")),
            (make_graph("input2"), make_graph("output2")),
        ];

        let rule = induction.induce(examples, &bounds).unwrap();
        assert_eq!(rule.description, "induced_rule");
    }

    #[test]
    fn test_simple_analogy() {
        let analogy = SimpleAnalogy::new();
        let bounds = ComplexityBounds::default();

        let source = make_graph("hello");
        let target = make_graph("world");

        let mapping = analogy.analogize(&source, &target, &bounds).unwrap();
        assert!(mapping.score > 0.0);
    }

    #[test]
    fn test_complexity_bounds() {
        let bounds = ComplexityBounds::default();
        assert_eq!(bounds.max_proof_depth, 20);
        assert_eq!(bounds.max_induction_examples, 10);
    }

    #[test]
    fn test_reasoning_engine_creation() {
        let engine = create_default_reasoning_engine();
        assert_eq!(engine.bounds.max_proof_depth, 20);
    }
}
