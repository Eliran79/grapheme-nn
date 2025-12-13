//! Graph Morphism Detection and Alignment
//!
//! Backend-176: Implements graph morphism detection for structural comparison.
//!
//! Supports:
//! - Approximate graph matching using Weisfeiler-Leman (WL) hashing
//! - Node alignment based on structural similarity
//! - Subgraph detection for pattern matching

use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

/// Result of graph morphism detection
#[derive(Debug, Clone)]
pub struct MorphismResult {
    /// Node alignment: maps nodes from graph A to graph B
    pub alignment: HashMap<NodeIndex, NodeIndex>,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// Whether graphs are approximately isomorphic
    pub is_isomorphic: bool,
    /// Matched subgraph size
    pub matched_nodes: usize,
}

/// Graph morphism detector using WL-based hashing
pub struct MorphismDetector {
    /// Number of WL iterations
    pub wl_iterations: usize,
    /// Similarity threshold for isomorphism
    pub iso_threshold: f32,
}

impl Default for MorphismDetector {
    fn default() -> Self {
        Self {
            wl_iterations: 3,
            iso_threshold: 0.95,
        }
    }
}

impl MorphismDetector {
    pub fn new(wl_iterations: usize, iso_threshold: f32) -> Self {
        Self {
            wl_iterations,
            iso_threshold,
        }
    }

    /// Detect morphism between two graphs
    pub fn detect<N, E>(
        &self,
        graph_a: &DiGraph<N, E>,
        graph_b: &DiGraph<N, E>,
    ) -> MorphismResult
    where
        N: std::hash::Hash + Clone,
        E: Clone,
    {
        // Compute WL hashes for both graphs
        let hashes_a = self.compute_wl_hashes(graph_a);
        let hashes_b = self.compute_wl_hashes(graph_b);

        // Find best alignment based on hash similarity
        let alignment = self.compute_alignment(&hashes_a, &hashes_b);

        // Calculate similarity score
        let similarity = self.compute_similarity(&hashes_a, &hashes_b, &alignment);

        let matched_nodes = alignment.len();
        let is_isomorphic = similarity >= self.iso_threshold
            && graph_a.node_count() == graph_b.node_count()
            && matched_nodes == graph_a.node_count();

        MorphismResult {
            alignment,
            similarity,
            is_isomorphic,
            matched_nodes,
        }
    }

    /// Compute Weisfeiler-Leman hashes for all nodes
    fn compute_wl_hashes<N, E>(&self, graph: &DiGraph<N, E>) -> HashMap<NodeIndex, u64>
    where
        N: std::hash::Hash,
    {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hashes: HashMap<NodeIndex, u64> = HashMap::new();

        // Initialize with node degree
        for node in graph.node_indices() {
            let in_deg = graph.edges_directed(node, petgraph::Direction::Incoming).count();
            let out_deg = graph.edges_directed(node, petgraph::Direction::Outgoing).count();
            let mut hasher = DefaultHasher::new();
            in_deg.hash(&mut hasher);
            out_deg.hash(&mut hasher);
            hashes.insert(node, hasher.finish());
        }

        // WL iterations
        for _ in 0..self.wl_iterations {
            let mut new_hashes: HashMap<NodeIndex, u64> = HashMap::new();

            for node in graph.node_indices() {
                let mut hasher = DefaultHasher::new();

                // Include current hash
                hashes.get(&node).unwrap_or(&0).hash(&mut hasher);

                // Collect and sort neighbor hashes
                let mut neighbor_hashes: Vec<u64> = graph
                    .neighbors(node)
                    .filter_map(|n| hashes.get(&n).copied())
                    .collect();
                neighbor_hashes.sort();

                for h in neighbor_hashes {
                    h.hash(&mut hasher);
                }

                new_hashes.insert(node, hasher.finish());
            }

            hashes = new_hashes;
        }

        hashes
    }

    /// Compute node alignment based on hash similarity
    fn compute_alignment(
        &self,
        hashes_a: &HashMap<NodeIndex, u64>,
        hashes_b: &HashMap<NodeIndex, u64>,
    ) -> HashMap<NodeIndex, NodeIndex> {
        let mut alignment: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut used_b: std::collections::HashSet<NodeIndex> = std::collections::HashSet::new();

        // Group nodes by hash
        let mut hash_to_nodes_b: HashMap<u64, Vec<NodeIndex>> = HashMap::new();
        for (&node, &hash) in hashes_b {
            hash_to_nodes_b.entry(hash).or_default().push(node);
        }

        // Match nodes with identical hashes first
        for (&node_a, &hash_a) in hashes_a {
            if let Some(candidates) = hash_to_nodes_b.get(&hash_a) {
                for &node_b in candidates {
                    if !used_b.contains(&node_b) {
                        alignment.insert(node_a, node_b);
                        used_b.insert(node_b);
                        break;
                    }
                }
            }
        }

        alignment
    }

    /// Compute similarity score between graphs
    fn compute_similarity(
        &self,
        hashes_a: &HashMap<NodeIndex, u64>,
        hashes_b: &HashMap<NodeIndex, u64>,
        alignment: &HashMap<NodeIndex, NodeIndex>,
    ) -> f32 {
        if hashes_a.is_empty() || hashes_b.is_empty() {
            return 0.0;
        }

        // Jaccard-like similarity of hash multisets
        let set_a: std::collections::HashSet<u64> = hashes_a.values().copied().collect();
        let set_b: std::collections::HashSet<u64> = hashes_b.values().copied().collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        let hash_sim = if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        };

        // Alignment coverage
        let max_nodes = hashes_a.len().max(hashes_b.len());
        let align_sim = if max_nodes > 0 {
            alignment.len() as f32 / max_nodes as f32
        } else {
            0.0
        };

        // Combined score
        0.6 * hash_sim + 0.4 * align_sim
    }

    /// Find common subgraph between two graphs
    pub fn find_common_subgraph<N, E>(
        &self,
        graph_a: &DiGraph<N, E>,
        graph_b: &DiGraph<N, E>,
    ) -> Vec<(NodeIndex, NodeIndex)>
    where
        N: std::hash::Hash + Clone,
        E: Clone,
    {
        let result = self.detect(graph_a, graph_b);
        result.alignment.into_iter().collect()
    }

    /// Check if graph_b contains graph_a as a subgraph (approximate)
    pub fn contains_subgraph<N, E>(
        &self,
        graph_a: &DiGraph<N, E>,
        graph_b: &DiGraph<N, E>,
    ) -> bool
    where
        N: std::hash::Hash + Clone,
        E: Clone,
    {
        if graph_a.node_count() > graph_b.node_count() {
            return false;
        }

        let result = self.detect(graph_a, graph_b);
        // Approximate subgraph: check if most nodes from A match nodes in B
        // and the similarity is above a threshold
        let coverage = result.matched_nodes as f32 / graph_a.node_count() as f32;
        coverage >= 0.5 || result.similarity >= 0.5
    }
}

/// Efficient graph alignment using spectral methods
pub fn spectral_alignment<N, E>(
    graph_a: &DiGraph<N, E>,
    graph_b: &DiGraph<N, E>,
) -> HashMap<NodeIndex, NodeIndex>
where
    N: Clone,
    E: Clone,
{
    // Simplified spectral alignment based on degree sequence
    let mut degrees_a: Vec<(NodeIndex, usize)> = graph_a
        .node_indices()
        .map(|n| (n, graph_a.neighbors(n).count()))
        .collect();
    let mut degrees_b: Vec<(NodeIndex, usize)> = graph_b
        .node_indices()
        .map(|n| (n, graph_b.neighbors(n).count()))
        .collect();

    degrees_a.sort_by_key(|&(_, d)| std::cmp::Reverse(d));
    degrees_b.sort_by_key(|&(_, d)| std::cmp::Reverse(d));

    let mut alignment = HashMap::new();
    for (i, &(node_a, _)) in degrees_a.iter().enumerate() {
        if i < degrees_b.len() {
            alignment.insert(node_a, degrees_b[i].0);
        }
    }

    alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_graphs() {
        let mut g1: DiGraph<i32, ()> = DiGraph::new();
        let a = g1.add_node(1);
        let b = g1.add_node(2);
        let c = g1.add_node(3);
        g1.add_edge(a, b, ());
        g1.add_edge(b, c, ());

        let mut g2: DiGraph<i32, ()> = DiGraph::new();
        let x = g2.add_node(1);
        let y = g2.add_node(2);
        let z = g2.add_node(3);
        g2.add_edge(x, y, ());
        g2.add_edge(y, z, ());

        let detector = MorphismDetector::default();
        let result = detector.detect(&g1, &g2);

        assert!(result.similarity > 0.9);
        assert!(result.is_isomorphic);
    }

    #[test]
    fn test_different_graphs() {
        let mut g1: DiGraph<i32, ()> = DiGraph::new();
        let a = g1.add_node(1);
        let b = g1.add_node(2);
        g1.add_edge(a, b, ());

        let mut g2: DiGraph<i32, ()> = DiGraph::new();
        let x = g2.add_node(1);
        let y = g2.add_node(2);
        let z = g2.add_node(3);
        g2.add_edge(x, y, ());
        g2.add_edge(y, z, ());
        g2.add_edge(z, x, ());

        let detector = MorphismDetector::default();
        let result = detector.detect(&g1, &g2);

        assert!(!result.is_isomorphic);
    }

    #[test]
    fn test_subgraph_detection() {
        let mut small: DiGraph<i32, ()> = DiGraph::new();
        let a = small.add_node(1);
        let b = small.add_node(2);
        small.add_edge(a, b, ());

        let mut large: DiGraph<i32, ()> = DiGraph::new();
        let x = large.add_node(1);
        let y = large.add_node(2);
        let z = large.add_node(3);
        large.add_edge(x, y, ());
        large.add_edge(y, z, ());

        let detector = MorphismDetector::default();
        assert!(detector.contains_subgraph(&small, &large));
    }

    #[test]
    fn test_spectral_alignment() {
        let mut g1: DiGraph<i32, ()> = DiGraph::new();
        let a = g1.add_node(1);
        let b = g1.add_node(2);
        g1.add_edge(a, b, ());

        let mut g2: DiGraph<i32, ()> = DiGraph::new();
        let x = g2.add_node(10);
        let y = g2.add_node(20);
        g2.add_edge(x, y, ());

        let alignment = spectral_alignment(&g1, &g2);
        assert_eq!(alignment.len(), 2);
    }
}
