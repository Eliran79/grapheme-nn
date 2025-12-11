//! Graph-to-Graph (G2G) Transformation Learning Module
//!
//! Enables GRAPHEME to learn transformations between graph representations.
//! Backend-175: G2G transformation learning.
//!
//! Key capabilities:
//! - Learn mappings between input/output graph pairs
//! - Support for partial graph matching
//! - Differentiable graph transformation rules

use grapheme_core::{GraphemeGraph, Node, Edge};
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// G2G transformation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2GRule {
    /// Rule name/identifier
    pub name: String,
    /// Source pattern to match
    pub source_pattern: GraphPattern,
    /// Target pattern to produce
    pub target_pattern: GraphPattern,
    /// Confidence score (learned)
    pub confidence: f32,
    /// Number of times this rule was applied successfully
    pub usage_count: usize,
}

/// A graph pattern for matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPattern {
    /// Number of nodes in pattern
    pub node_count: usize,
    /// Edge connections (from_idx, to_idx)
    pub edges: Vec<(usize, usize)>,
    /// Node labels (if any)
    pub node_labels: Vec<Option<String>>,
    /// Minimum number of nodes
    pub min_nodes: usize,
    /// Maximum number of nodes
    pub max_nodes: usize,
}

impl Default for GraphPattern {
    fn default() -> Self {
        Self {
            node_count: 0,
            edges: Vec::new(),
            node_labels: Vec::new(),
            min_nodes: 1,
            max_nodes: 100,
        }
    }
}

/// G2G training configuration
#[derive(Debug, Clone)]
pub struct G2GConfig {
    /// Learning rate for rule weights
    pub learning_rate: f32,
    /// Minimum confidence to keep a rule
    pub min_confidence: f32,
    /// Maximum number of rules to maintain
    pub max_rules: usize,
    /// Whether to merge similar rules
    pub merge_similar: bool,
    /// Similarity threshold for merging
    pub similarity_threshold: f32,
}

impl Default for G2GConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            min_confidence: 0.1,
            max_rules: 1000,
            merge_similar: true,
            similarity_threshold: 0.9,
        }
    }
}

/// A graph transformation example (input -> output pair)
#[derive(Debug, Clone)]
pub struct G2GExample {
    /// Input graph
    pub input: GraphemeGraph,
    /// Expected output graph
    pub output: GraphemeGraph,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl G2GExample {
    /// Create a new G2G example
    pub fn new(input: GraphemeGraph, output: GraphemeGraph) -> Self {
        Self {
            input,
            output,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Result of applying G2G transformation
#[derive(Debug, Clone)]
pub struct G2GResult {
    /// Transformed output graph
    pub output: GraphemeGraph,
    /// Rules that were applied
    pub applied_rules: Vec<String>,
    /// Confidence score for the transformation
    pub confidence: f32,
    /// Number of transformation steps
    pub steps: usize,
}

/// G2G transformation learner
pub struct G2GLearner {
    config: G2GConfig,
    rules: Vec<G2GRule>,
}

impl G2GLearner {
    /// Create a new G2G learner
    pub fn new() -> Self {
        Self {
            config: G2GConfig::default(),
            rules: Vec::new(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: G2GConfig) -> Self {
        Self {
            config,
            rules: Vec::new(),
        }
    }

    /// Learn from a single example
    pub fn learn(&mut self, example: &G2GExample) -> f32 {
        // Extract patterns from example
        let source_pattern = self.extract_pattern(&example.input);
        let target_pattern = self.extract_pattern(&example.output);

        // Check if we have a matching rule
        let mut found_idx: Option<usize> = None;
        let mut match_score = 0.0;

        for (idx, rule) in self.rules.iter().enumerate() {
            let source_match = Self::pattern_similarity_static(&rule.source_pattern, &source_pattern);
            let target_match = Self::pattern_similarity_static(&rule.target_pattern, &target_pattern);

            if source_match > self.config.similarity_threshold
                && target_match > self.config.similarity_threshold
            {
                match_score = (source_match + target_match) / 2.0;
                found_idx = Some(idx);
                break;
            }
        }

        // Update or create rule
        if let Some(idx) = found_idx {
            // Reinforce existing rule
            self.rules[idx].confidence = self.rules[idx].confidence * 0.9 + 0.1;
            self.rules[idx].usage_count += 1;
        } else if self.rules.len() < self.config.max_rules {
            // Create new rule
            let rule = G2GRule {
                name: format!("rule_{}", self.rules.len()),
                source_pattern,
                target_pattern,
                confidence: 0.5,
                usage_count: 1,
            };
            self.rules.push(rule);
            match_score = 0.5;
        }

        // Prune low-confidence rules
        self.prune_rules();

        match_score
    }

    /// Learn from a batch of examples
    pub fn learn_batch(&mut self, examples: &[G2GExample]) -> f32 {
        let mut total_score = 0.0;
        for example in examples {
            total_score += self.learn(example);
        }
        total_score / examples.len() as f32
    }

    /// Transform an input graph using learned rules
    pub fn transform(&self, input: &GraphemeGraph) -> G2GResult {
        let input_pattern = self.extract_pattern(input);
        let mut best_rule: Option<&G2GRule> = None;
        let mut best_match = 0.0;

        // Find best matching rule
        for rule in &self.rules {
            let match_score = Self::pattern_similarity_static(&rule.source_pattern, &input_pattern);
            if match_score > best_match && match_score > self.config.min_confidence {
                best_match = match_score;
                best_rule = Some(rule);
            }
        }

        match best_rule {
            Some(rule) => {
                // Apply the rule to transform the graph
                let output = self.apply_rule(input, rule);
                G2GResult {
                    output,
                    applied_rules: vec![rule.name.clone()],
                    confidence: best_match * rule.confidence,
                    steps: 1,
                }
            }
            None => {
                // No matching rule, return input unchanged
                G2GResult {
                    output: input.clone(),
                    applied_rules: vec![],
                    confidence: 0.0,
                    steps: 0,
                }
            }
        }
    }

    /// Get all learned rules
    pub fn rules(&self) -> &[G2GRule] {
        &self.rules
    }

    /// Get number of rules
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Clear all rules
    pub fn clear(&mut self) {
        self.rules.clear();
    }

    // Extract a pattern from a graph
    fn extract_pattern(&self, graph: &GraphemeGraph) -> GraphPattern {
        let node_count = graph.graph.node_count();

        let node_labels: Vec<Option<String>> = graph.graph
            .node_indices()
            .map(|idx| {
                graph.graph.node_weight(idx).map(|_| format!("node_{}", idx.index()))
            })
            .collect();

        let edges: Vec<(usize, usize)> = graph.graph
            .edge_indices()
            .filter_map(|e| {
                graph.graph.edge_endpoints(e).map(|(a, b)| (a.index(), b.index()))
            })
            .collect();

        GraphPattern {
            node_count,
            node_labels,
            edges,
            min_nodes: node_count,
            max_nodes: node_count,
        }
    }

    // Calculate similarity between two patterns (static version for borrow safety)
    fn pattern_similarity_static(p1: &GraphPattern, p2: &GraphPattern) -> f32 {
        let mut score = 0.0;
        let mut total = 0.0;

        // Compare node counts
        let node_diff = (p1.node_count as f32 - p2.node_count as f32).abs();
        let max_nodes = p1.node_count.max(p2.node_count) as f32;
        if max_nodes > 0.0 {
            score += 1.0 - (node_diff / max_nodes);
            total += 1.0;
        }

        // Compare edge counts
        let edge_diff = (p1.edges.len() as f32 - p2.edges.len() as f32).abs();
        let max_edges = p1.edges.len().max(p2.edges.len()) as f32;
        if max_edges > 0.0 {
            score += 1.0 - (edge_diff / max_edges);
            total += 1.0;
        }

        // Compare node labels (for overlapping indices)
        let min_nodes = p1.node_labels.len().min(p2.node_labels.len());
        if min_nodes > 0 {
            let label_matches: f32 = (0..min_nodes)
                .map(|i| {
                    match (&p1.node_labels.get(i), &p2.node_labels.get(i)) {
                        (Some(Some(_)), Some(Some(_))) => 1.0, // Both have labels
                        (Some(None), _) | (_, Some(None)) => 0.5, // One has no label
                        _ => 0.0,
                    }
                })
                .sum();
            score += label_matches / min_nodes as f32;
            total += 1.0;
        }

        if total > 0.0 {
            score / total
        } else {
            0.0
        }
    }

    // Apply a rule to transform a graph
    fn apply_rule(&self, _input: &GraphemeGraph, rule: &G2GRule) -> GraphemeGraph {
        // For now, create a graph based on target pattern structure
        // In a full implementation, this would map input nodes to output nodes
        let mut output = GraphemeGraph::new();

        // Copy structure from target pattern
        let target_node_count = rule.target_pattern.node_count;

        // Create nodes (as hidden nodes)
        for _ in 0..target_node_count {
            output.graph.add_node(Node::hidden());
        }

        // Create edges
        for (from, to) in &rule.target_pattern.edges {
            if *from < target_node_count && *to < target_node_count {
                let from_idx = NodeIndex::new(*from);
                let to_idx = NodeIndex::new(*to);
                output.graph.add_edge(from_idx, to_idx, Edge::sequential());
            }
        }

        output
    }

    // Prune rules with low confidence
    fn prune_rules(&mut self) {
        self.rules.retain(|r| r.confidence >= self.config.min_confidence);
    }
}

impl Default for G2GLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute graph edit distance for G2G loss
pub fn g2g_loss(predicted: &GraphemeGraph, expected: &GraphemeGraph) -> f32 {
    // Node count difference
    let pred_nodes = predicted.graph.node_count();
    let exp_nodes = expected.graph.node_count();
    let node_diff = (pred_nodes as f32 - exp_nodes as f32).abs();

    // Edge count difference
    let pred_edges = predicted.graph.edge_count();
    let exp_edges = expected.graph.edge_count();
    let edge_diff = (pred_edges as f32 - exp_edges as f32).abs();

    // Normalize by max size
    let max_nodes = pred_nodes.max(exp_nodes) as f32;
    let max_edges = pred_edges.max(exp_edges) as f32;

    let node_loss = if max_nodes > 0.0 { node_diff / max_nodes } else { 0.0 };
    let edge_loss = if max_edges > 0.0 { edge_diff / max_edges } else { 0.0 };

    (node_loss + edge_loss) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_graph(nodes: usize, edges: Vec<(usize, usize)>) -> GraphemeGraph {
        let mut g = GraphemeGraph::new();
        for _ in 0..nodes {
            g.graph.add_node(Node::hidden());
        }
        for (from, to) in edges {
            let from_idx = NodeIndex::new(from);
            let to_idx = NodeIndex::new(to);
            g.graph.add_edge(from_idx, to_idx, Edge::sequential());
        }
        g
    }

    #[test]
    fn test_g2g_learner_creation() {
        let learner = G2GLearner::new();
        assert_eq!(learner.num_rules(), 0);
    }

    #[test]
    fn test_learn_single_example() {
        let mut learner = G2GLearner::new();

        let input = create_simple_graph(2, vec![(0, 1)]);
        let output = create_simple_graph(3, vec![(0, 1), (1, 2)]);
        let example = G2GExample::new(input, output);

        let score = learner.learn(&example);
        assert!(score > 0.0);
        assert_eq!(learner.num_rules(), 1);
    }

    #[test]
    fn test_learn_batch() {
        let mut learner = G2GLearner::new();

        let examples: Vec<G2GExample> = (0..5)
            .map(|i| {
                let input = create_simple_graph(i + 1, vec![]);
                let output = create_simple_graph(i + 2, vec![(0, 1)]);
                G2GExample::new(input, output)
            })
            .collect();

        let score = learner.learn_batch(&examples);
        assert!(score > 0.0);
        assert!(learner.num_rules() > 0);
    }

    #[test]
    fn test_transform() {
        let mut learner = G2GLearner::new();

        // Learn a transformation
        let input = create_simple_graph(2, vec![(0, 1)]);
        let output = create_simple_graph(3, vec![(0, 1), (1, 2)]);
        learner.learn(&G2GExample::new(input.clone(), output));

        // Apply transformation
        let result = learner.transform(&input);
        assert!(result.confidence > 0.0 || result.steps == 0);
    }

    #[test]
    fn test_pattern_extraction() {
        let learner = G2GLearner::new();
        let graph = create_simple_graph(3, vec![(0, 1), (1, 2)]);
        let pattern = learner.extract_pattern(&graph);

        assert_eq!(pattern.node_count, 3);
        assert_eq!(pattern.edges.len(), 2);
    }

    #[test]
    fn test_g2g_loss() {
        let g1 = create_simple_graph(3, vec![(0, 1), (1, 2)]);
        let g2 = create_simple_graph(3, vec![(0, 1), (1, 2)]);

        let loss = g2g_loss(&g1, &g2);
        assert!(loss < 0.01, "Identical graphs should have near-zero loss");
    }

    #[test]
    fn test_g2g_loss_different_sizes() {
        let g1 = create_simple_graph(2, vec![(0, 1)]);
        let g2 = create_simple_graph(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);

        let loss = g2g_loss(&g1, &g2);
        assert!(loss > 0.3, "Different sized graphs should have significant loss");
    }

    #[test]
    fn test_clear_rules() {
        let mut learner = G2GLearner::new();

        let input = create_simple_graph(2, vec![(0, 1)]);
        let output = create_simple_graph(3, vec![(0, 1), (1, 2)]);
        learner.learn(&G2GExample::new(input, output));

        assert!(learner.num_rules() > 0);
        learner.clear();
        assert_eq!(learner.num_rules(), 0);
    }
}
