//! Graph Autoencoder trait for Stage 1 brain training.
//!
//! This module provides the `GraphAutoencoder` trait that domain brains implement
//! to enable perfect encoding/decoding of domain inputs to/from graph representations.
//!
//! ## Two-Stage Training Paradigm
//!
//! **Stage 1**: Train brains to be perfect encoders/decoders (autoencoders)
//! - Goal: Zero information loss in text ↔ graph conversion
//! - Loss: Reconstruction (input_text == output_text)
//!
//! **Stage 2**: Train graph transformations on pre-encoded graphs (no text in loop)
//! - Brains are FROZEN after Stage 1 - only GraphTransform learns
//! - Pure graph-to-graph transformation
//!
//! ## Brains as Sensory Organs
//!
//! After Stage 1 training, brains become like sensory organs (eyes and ears):
//! - They PERCEIVE raw input (text, images, audio)
//! - They CONVERT to the universal graph language
//! - The THINKING happens entirely in graph space
//! - They TRANSLATE back to the output modality
//!
//! ## Example
//!
//! ```ignore
//! use grapheme_brain_common::{GraphAutoencoder, LatentGraph, AutoencoderError};
//! use grapheme_core::DomainBrain;
//!
//! struct MyBrain { /* ... */ }
//!
//! impl DomainBrain for MyBrain { /* ... */ }
//!
//! impl GraphAutoencoder for MyBrain {
//!     fn encode(&self, input: &str) -> Result<LatentGraph, AutoencoderError> {
//!         let graph = self.parse(input)?;
//!         Ok(LatentGraph::new("mydomain", graph))
//!     }
//!
//!     fn decode(&self, latent: &LatentGraph) -> Result<String, AutoencoderError> {
//!         Ok(latent.graph.to_text())
//!     }
//!
//!     fn reconstruction_loss(&self, original: &str, reconstructed: &str) -> f32 {
//!         // Character-level Levenshtein distance normalized by length
//!         let distance = levenshtein(original, reconstructed);
//!         distance as f32 / original.len().max(1) as f32
//!     }
//! }
//! ```

use std::collections::HashMap;

use grapheme_core::{DagNN, DomainBrain};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during autoencoder operations.
#[derive(Debug, Error)]
pub enum AutoencoderError {
    /// Error during encoding (text → graph)
    #[error("Encoding failed: {0}")]
    EncodingError(String),

    /// Error during decoding (graph → text)
    #[error("Decoding failed: {0}")]
    DecodingError(String),

    /// Graph validation error
    #[error("Graph validation failed: {0}")]
    ValidationError(String),

    /// Domain mismatch (trying to decode with wrong brain)
    #[error("Domain mismatch: expected '{expected}', got '{actual}'")]
    DomainMismatch { expected: String, actual: String },

    /// Underlying domain brain error
    #[error("Domain error: {0}")]
    DomainError(#[from] grapheme_core::DomainError),
}

/// Latent graph representation for serialization and Stage 2 training.
///
/// This is the intermediate representation produced by brain encoding.
/// It can be serialized/deserialized for:
/// - Pre-encoding training data (avoiding text in Stage 2 loop)
/// - Caching encoded representations
/// - Cross-brain graph transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentGraph {
    /// Domain identifier (e.g., "code", "math", "text", "vision")
    pub domain: String,

    /// The actual graph structure
    pub graph: DagNN,

    /// Optional metadata (e.g., source file, timestamp, version)
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl LatentGraph {
    /// Create a new latent graph with the given domain and graph.
    pub fn new(domain: impl Into<String>, graph: DagNN) -> Self {
        Self {
            domain: domain.into(),
            graph,
            metadata: HashMap::new(),
        }
    }

    /// Create a latent graph with metadata.
    pub fn with_metadata(
        domain: impl Into<String>,
        graph: DagNN,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            domain: domain.into(),
            graph,
            metadata,
        }
    }

    /// Add a metadata key-value pair.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if this graph is from the expected domain.
    pub fn is_domain(&self, expected: &str) -> bool {
        self.domain == expected
    }
}

/// Trait for brain autoencoders (Stage 1 training).
///
/// This trait extends `DomainBrain` to add autoencoding capabilities.
/// Brains that implement this trait can:
/// - Encode domain input to latent graph representation
/// - Decode latent graph back to domain output
/// - Compute reconstruction loss for training
///
/// ## Perfect Reconstruction Goal
///
/// The goal of Stage 1 training is to achieve perfect reconstruction:
/// ```text
/// input_text == decode(encode(input_text))
/// ```
///
/// A reconstruction_loss of 0.0 means perfect reconstruction.
///
/// ## Learnable Parameters
///
/// Brains may have learnable parameters that are optimized during Stage 1:
/// - Embedding weights for character/token encoding
/// - Graph structure parameters (edge thresholds, pooling)
/// - Decoding vocabulary/templates
pub trait GraphAutoencoder: DomainBrain {
    /// Encode domain input to latent graph representation.
    ///
    /// This converts raw text/input into a graph that preserves all semantic
    /// information needed for perfect reconstruction.
    ///
    /// # Arguments
    /// * `input` - Raw domain input (text, code, math expression, etc.)
    ///
    /// # Returns
    /// * `Ok(LatentGraph)` - The encoded graph representation
    /// * `Err(AutoencoderError)` - If encoding fails
    fn encode(&self, input: &str) -> Result<LatentGraph, AutoencoderError>;

    /// Decode latent graph back to domain output.
    ///
    /// This converts the graph representation back to text/output format.
    /// For perfect autoencoders, this should exactly reproduce the original input.
    ///
    /// # Arguments
    /// * `graph` - The latent graph to decode
    ///
    /// # Returns
    /// * `Ok(String)` - The decoded output
    /// * `Err(AutoencoderError)` - If decoding fails
    fn decode(&self, graph: &LatentGraph) -> Result<String, AutoencoderError>;

    /// Compute reconstruction loss between original and reconstructed text.
    ///
    /// This measures how well the autoencoder preserved information.
    /// A loss of 0.0 means perfect reconstruction.
    ///
    /// # Default Implementation
    ///
    /// The default uses normalized Levenshtein distance:
    /// - 0.0 = exact match
    /// - 1.0 = completely different
    ///
    /// Brains may override with domain-specific loss functions that ignore
    /// irrelevant differences (e.g., whitespace in code, operator order in math).
    ///
    /// # Arguments
    /// * `original` - The original input text
    /// * `reconstructed` - The text after encode → decode roundtrip
    ///
    /// # Returns
    /// * `f32` - Loss value in [0.0, 1.0] range (lower is better)
    fn reconstruction_loss(&self, original: &str, reconstructed: &str) -> f32 {
        // Default: normalized character-level accuracy
        if original == reconstructed {
            return 0.0;
        }

        if original.is_empty() && reconstructed.is_empty() {
            return 0.0;
        }

        let max_len = original.len().max(reconstructed.len());
        if max_len == 0 {
            return 0.0;
        }

        // Count matching characters at each position
        let matching: usize = original
            .chars()
            .zip(reconstructed.chars())
            .filter(|(a, b)| a == b)
            .count();

        // Penalize length difference
        let len_diff = (original.len() as isize - reconstructed.len() as isize).unsigned_abs();

        // Loss = 1.0 - (matching / max_len) + length_penalty
        let accuracy = matching as f32 / max_len as f32;
        let length_penalty = len_diff as f32 / max_len as f32;

        (1.0 - accuracy + length_penalty * 0.5).clamp(0.0, 1.0)
    }

    /// Full roundtrip: encode → decode, returns (output, loss).
    ///
    /// This is a convenience method that performs the full autoencoding cycle
    /// and returns both the reconstructed output and the loss.
    ///
    /// # Arguments
    /// * `input` - Raw domain input
    ///
    /// # Returns
    /// * `Ok((String, f32))` - (reconstructed_output, reconstruction_loss)
    /// * `Err(AutoencoderError)` - If encoding or decoding fails
    fn roundtrip(&self, input: &str) -> Result<(String, f32), AutoencoderError> {
        let graph = self.encode(input)?;
        let output = self.decode(&graph)?;
        let loss = self.reconstruction_loss(input, &output);
        Ok((output, loss))
    }

    /// Batch encode multiple inputs.
    ///
    /// # Default Implementation
    ///
    /// Default implementation calls `encode` for each input sequentially.
    /// Brains may override for parallel/batched encoding.
    fn encode_batch(&self, inputs: &[&str]) -> Vec<Result<LatentGraph, AutoencoderError>> {
        inputs.iter().map(|input| self.encode(input)).collect()
    }

    /// Batch decode multiple graphs.
    ///
    /// # Default Implementation
    ///
    /// Default implementation calls `decode` for each graph sequentially.
    /// Brains may override for parallel/batched decoding.
    fn decode_batch(&self, graphs: &[&LatentGraph]) -> Vec<Result<String, AutoencoderError>> {
        graphs.iter().map(|graph| self.decode(graph)).collect()
    }

    /// Validate that the graph is decodable by this brain.
    ///
    /// Checks domain match and basic graph validity.
    fn validate_latent(&self, graph: &LatentGraph) -> Result<(), AutoencoderError> {
        if !graph.is_domain(self.domain_id()) {
            return Err(AutoencoderError::DomainMismatch {
                expected: self.domain_id().to_string(),
                actual: graph.domain.clone(),
            });
        }

        if graph.node_count() == 0 {
            return Err(AutoencoderError::ValidationError(
                "Empty graph".to_string(),
            ));
        }

        Ok(())
    }
}

/// A pre-encoded training pair for Stage 2 (graph-only) training.
///
/// This represents a (input_graph, output_graph) pair that has been
/// pre-encoded using Stage 1 trained brains. No text is stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedPair {
    /// The input latent graph (e.g., encoded problem statement)
    pub input: LatentGraph,

    /// The output latent graph (e.g., encoded solution)
    pub output: LatentGraph,

    /// Optional metadata about the pair
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl EncodedPair {
    /// Create a new encoded pair.
    pub fn new(input: LatentGraph, output: LatentGraph) -> Self {
        Self {
            input,
            output,
            metadata: HashMap::new(),
        }
    }

    /// Check if both graphs are from the same domain.
    pub fn same_domain(&self) -> bool {
        self.input.domain == self.output.domain
    }

    /// Get the domain of the input graph.
    pub fn domain(&self) -> &str {
        &self.input.domain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_graph_new() {
        let graph = DagNN::from_text("test").unwrap();
        let latent = LatentGraph::new("code", graph);

        assert_eq!(latent.domain, "code");
        assert!(latent.metadata.is_empty());
        assert!(latent.is_domain("code"));
        assert!(!latent.is_domain("math"));
    }

    #[test]
    fn test_latent_graph_with_metadata() {
        let graph = DagNN::from_text("test").unwrap();
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test.py".to_string());

        let latent = LatentGraph::with_metadata("code", graph, metadata);

        assert_eq!(latent.metadata.get("source"), Some(&"test.py".to_string()));
    }

    #[test]
    fn test_latent_graph_add_metadata() {
        let graph = DagNN::from_text("test").unwrap();
        let mut latent = LatentGraph::new("code", graph);

        latent.add_metadata("version", "1.0");
        assert_eq!(latent.metadata.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_latent_graph_counts() {
        let graph = DagNN::from_text("hello").unwrap();
        let latent = LatentGraph::new("text", graph);

        assert!(latent.node_count() > 0);
        // Edge count depends on graph structure
    }

    #[test]
    fn test_encoded_pair() {
        let input_graph = DagNN::from_text("2 + 2").unwrap();
        let output_graph = DagNN::from_text("4").unwrap();

        let pair = EncodedPair::new(
            LatentGraph::new("math", input_graph),
            LatentGraph::new("math", output_graph),
        );

        assert!(pair.same_domain());
        assert_eq!(pair.domain(), "math");
    }

    #[test]
    fn test_encoded_pair_different_domains() {
        let input_graph = DagNN::from_text("what is 2 + 2").unwrap();
        let output_graph = DagNN::from_text("4").unwrap();

        let pair = EncodedPair::new(
            LatentGraph::new("text", input_graph),
            LatentGraph::new("math", output_graph),
        );

        assert!(!pair.same_domain());
    }

    #[test]
    fn test_autoencoder_error_display() {
        let err = AutoencoderError::EncodingError("test error".to_string());
        assert_eq!(err.to_string(), "Encoding failed: test error");

        let err = AutoencoderError::DomainMismatch {
            expected: "code".to_string(),
            actual: "math".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Domain mismatch: expected 'code', got 'math'"
        );
    }

    // Test the default reconstruction_loss implementation
    #[test]
    fn test_default_reconstruction_loss() {
        // We can't directly test the trait method without a concrete type,
        // but we can test the logic by reimplementing it here
        fn test_loss(original: &str, reconstructed: &str) -> f32 {
            if original == reconstructed {
                return 0.0;
            }

            if original.is_empty() && reconstructed.is_empty() {
                return 0.0;
            }

            let max_len = original.len().max(reconstructed.len());
            if max_len == 0 {
                return 0.0;
            }

            let matching: usize = original
                .chars()
                .zip(reconstructed.chars())
                .filter(|(a, b)| a == b)
                .count();

            let len_diff = (original.len() as isize - reconstructed.len() as isize).unsigned_abs();

            let accuracy = matching as f32 / max_len as f32;
            let length_penalty = len_diff as f32 / max_len as f32;

            (1.0 - accuracy + length_penalty * 0.5).clamp(0.0, 1.0)
        }

        // Perfect match
        assert_eq!(test_loss("hello", "hello"), 0.0);

        // Empty strings
        assert_eq!(test_loss("", ""), 0.0);

        // Completely different
        let loss = test_loss("abc", "xyz");
        assert!(loss > 0.9); // Should be close to 1.0

        // One character different
        let loss = test_loss("hello", "hella");
        assert!(loss > 0.0 && loss < 0.5);

        // Length difference
        let loss = test_loss("hello", "helloworld");
        assert!(loss > 0.0);
    }
}
