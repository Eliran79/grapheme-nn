//! Training Utilities for CortexMesh Trainers (backend-220)
//!
//! This module provides shared helper functions for semantic accuracy metrics,
//! decoder batch preparation, and graph feature decoding used across multiple
//! training binaries (train_cortex_mesh.rs, train_mesh_code.rs, etc.).
//!
//! # Example Usage
//! ```ignore
//! use grapheme_train::training_utils::{semantic_accuracy, prepare_decoder_batch};
//!
//! // Compute semantic accuracy between predicted and target graphs
//! let accuracy = semantic_accuracy(&pred_graph, &target_graph);
//!
//! // Prepare decoder batch from features and target graph
//! let batch = prepare_decoder_batch(&features, &target_graph, &decoder);
//! ```

use grapheme_core::{ActivationFn, Edge, GraphemeGraph, Node};
use petgraph::graph::DiGraph;
use rayon::iter::ParallelIterator;

use crate::semantic_decoder::SemanticDecoder;

/// Compute semantic accuracy: percentage of nodes with matching node types
///
/// This compares the node types (semantic categories) between predicted and
/// target graphs, giving a measure of how well the model captures semantic
/// structure beyond just character-level accuracy.
///
/// # Arguments
/// * `pred` - Predicted graph from the model
/// * `target` - Target/ground truth graph
///
/// # Returns
/// Accuracy as a float between 0.0 and 1.0
pub fn semantic_accuracy(pred: &GraphemeGraph, target: &GraphemeGraph) -> f32 {
    let pred_types: Vec<_> = pred.graph.node_indices()
        .map(|idx| &pred.graph[idx].node_type)
        .collect();
    let target_types: Vec<_> = target.graph.node_indices()
        .map(|idx| &target.graph[idx].node_type)
        .collect();

    let min_len = pred_types.len().min(target_types.len());
    if min_len == 0 {
        return 0.0;
    }

    let matches = pred_types.iter().take(min_len)
        .zip(target_types.iter().take(min_len))
        .filter(|(p, t)| p == t)
        .count();

    matches as f32 / min_len as f32
}

/// Decode pooled features to a semantic graph using SemanticDecoder
///
/// This converts a matrix of feature vectors (one per node) into a GraphemeGraph
/// with proper semantic node types decoded from the features.
///
/// # Arguments
/// * `features` - 2D array of shape (n_nodes, hidden_dim)
/// * `decoder` - SemanticDecoder with vocabulary
///
/// # Returns
/// A GraphemeGraph with decoded semantic node types
#[allow(dead_code)]  // Will be used for decoded graph visualization in future
pub fn decode_features_to_graph(
    features: &ndarray::Array2<f32>,
    decoder: &SemanticDecoder,
) -> GraphemeGraph {
    let mut graph: DiGraph<Node, Edge> = DiGraph::new();
    let mut prev_idx = None;
    let mut input_nodes = Vec::new();

    // Decode each feature vector to a semantic node type
    for i in 0..features.nrows() {
        let hidden: Vec<f32> = features.row(i).to_vec();
        let (node_type, confidence) = decoder.decode(&hidden);

        let node = Node {
            value: None,
            activation: confidence,
            pre_activation: confidence,
            node_type,
            position: Some(i),
            activation_fn: ActivationFn::Linear,
        };

        let idx = graph.add_node(node);
        input_nodes.push(idx);

        // Add sequential edge
        if let Some(prev) = prev_idx {
            graph.add_edge(prev, idx, Edge::sequential());
        }
        prev_idx = Some(idx);
    }

    GraphemeGraph {
        graph,
        input_nodes,
        cliques: Vec::new(),
    }
}

/// Prepare training batch for SemanticDecoder
///
/// This creates a batch of (hidden_vector, target_index) pairs for training
/// the SemanticDecoder, where each pair maps a feature vector to its target
/// semantic node type index.
///
/// # Arguments
/// * `features` - 2D array of shape (n_nodes, hidden_dim)
/// * `target_graph` - Target graph with ground truth node types
/// * `decoder` - SemanticDecoder for looking up target indices
///
/// # Returns
/// Vector of (hidden_vector, target_index) pairs
pub fn prepare_decoder_batch(
    features: &ndarray::Array2<f32>,
    target_graph: &GraphemeGraph,
    decoder: &SemanticDecoder,
) -> Vec<(Vec<f32>, usize)> {
    let mut batch = Vec::new();
    let target_nodes: Vec<_> = target_graph.graph.node_indices()
        .map(|idx| &target_graph.graph[idx])
        .collect();

    let n = features.nrows().min(target_nodes.len());

    for i in 0..n {
        let hidden: Vec<f32> = features.row(i).to_vec();
        let target_type = &target_nodes[i].node_type;

        if let Some(target_idx) = decoder.get_index(target_type) {
            batch.push((hidden, target_idx));
        }
    }

    batch
}

/// Generate hash-based embedding features from a graph
///
/// This creates a simple feature matrix for a graph by hashing character values
/// to create embedding-like features. Used when actual model features are not
/// available (e.g., in MeshCodeGen where the EncoderDecoder doesn't expose
/// intermediate representations).
///
/// # Arguments
/// * `graph` - Graph to generate features for
/// * `embed_dim` - Embedding dimension for features
///
/// # Returns
/// 2D array of shape (n_nodes, embed_dim)
pub fn hash_based_features(
    graph: &GraphemeGraph,
    embed_dim: usize,
) -> ndarray::Array2<f32> {
    let n_nodes = graph.graph.node_count();
    let mut features = ndarray::Array2::<f32>::zeros((n_nodes, embed_dim));

    for (i, node_idx) in graph.graph.node_indices().enumerate() {
        let node = &graph.graph[node_idx];
        if let Some(c) = node.value {
            let hash = (c as u32 * 31) as f32 / 256.0;
            for j in 0..embed_dim {
                features[[i, j]] = ((hash + j as f32 * 0.1) % 1.0) - 0.5;
            }
        }
    }

    features
}

/// Compute validation metrics in parallel for a batch of (predicted, target) string pairs
///
/// This is a helper for computing character-level accuracy and semantic accuracy
/// in parallel using rayon. Used when model.generate() returns strings that need
/// to be converted to graphs for semantic comparison.
///
/// # Arguments
/// * `pairs` - Iterator of (predicted_string, target_string) pairs
///
/// # Returns
/// Vector of (char_accuracy, is_exact_match, semantic_accuracy)
pub fn compute_validation_metrics<'a, I>(
    pairs: I,
) -> Vec<(f32, bool, f32)>
where
    I: rayon::iter::ParallelIterator<Item = (&'a str, &'a str)>,
{
    pairs
        .map(|(predicted, target)| {
            let char_acc = char_accuracy(predicted, target);
            let is_exact = exact_match(predicted, target);

            let pred_graph = GraphemeGraph::from_text(predicted);
            let target_graph = GraphemeGraph::from_text(target);
            let sem_acc = semantic_accuracy(&pred_graph, &target_graph);

            (char_acc, is_exact, sem_acc)
        })
        .collect()
}

/// Character-level accuracy between predicted and target strings
pub fn char_accuracy(predicted: &str, target: &str) -> f32 {
    let pred_chars: Vec<char> = predicted.chars().collect();
    let target_chars: Vec<char> = target.chars().collect();

    let min_len = pred_chars.len().min(target_chars.len());
    if min_len == 0 {
        return if target_chars.is_empty() { 1.0 } else { 0.0 };
    }

    let matches = pred_chars.iter().take(min_len)
        .zip(target_chars.iter().take(min_len))
        .filter(|(p, t)| p == t)
        .count();

    matches as f32 / target_chars.len().max(1) as f32
}

/// Check if predicted string exactly matches target
pub fn exact_match(predicted: &str, target: &str) -> bool {
    predicted == target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_accuracy_identical() {
        let graph1 = GraphemeGraph::from_text("hello");
        let graph2 = GraphemeGraph::from_text("hello");
        assert!((semantic_accuracy(&graph1, &graph2) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_semantic_accuracy_different() {
        let graph1 = GraphemeGraph::from_text("hello");
        let graph2 = GraphemeGraph::from_text("world");
        // Different text with different characters = different node types
        // Accuracy depends on how many character positions have the same node type
        let acc = semantic_accuracy(&graph1, &graph2);
        // The accuracy should be between 0 and 1 (valid range)
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_semantic_accuracy_empty() {
        let empty = GraphemeGraph::new();
        let graph = GraphemeGraph::from_text("test");
        assert_eq!(semantic_accuracy(&empty, &graph), 0.0);
    }

    #[test]
    fn test_char_accuracy() {
        assert!((char_accuracy("hello", "hello") - 1.0).abs() < 0.001);
        assert!((char_accuracy("hello", "hella") - 0.8).abs() < 0.001);
        assert!((char_accuracy("", "") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_exact_match() {
        assert!(exact_match("hello", "hello"));
        assert!(!exact_match("hello", "world"));
    }

    #[test]
    fn test_hash_based_features() {
        let graph = GraphemeGraph::from_text("ab");
        let features = hash_based_features(&graph, 64);
        assert_eq!(features.nrows(), graph.graph.node_count());
        assert_eq!(features.ncols(), 64);
    }
}
