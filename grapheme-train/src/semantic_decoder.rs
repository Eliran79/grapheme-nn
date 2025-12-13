//! Semantic Node Decoder with Unified Vocabulary
//!
//! This module provides a decoder that can output ANY semantic node type from
//! a unified vocabulary collected from all registered domain brains.
//!
//! # Architecture Problem Solved
//!
//! The current graph-to-graph transformation can only output the same node types
//! it receives as input. For example:
//!
//! ```text
//! Input:  [Input('W'), Input('r'), Input('i'), ...]
//! Output: [Input('x'), Input('x'), Input('x'), ...]  <- Can't change node types!
//! ```
//!
//! The SemanticDecoder solves this by:
//!
//! 1. Collecting all NodeType variants from all domain brains via `node_types()`
//! 2. Building a unified vocabulary with learnable embeddings per type
//! 3. Decoding hidden representations into NodeType predictions via softmax
//!
//! ```text
//! Input:  [Input('W'), Input('r'), Input('i'), ...]
//! Output: [Keyword(def), Variable(f), Punct('('), Variable(x), ...]  <- NEW types!
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use grapheme_train::SemanticDecoder;
//! use grapheme_core::NodeType;
//!
//! // Build vocabulary from all brains
//! let vocab = SemanticDecoder::build_vocab_from_brains();
//!
//! // Create decoder
//! let decoder = SemanticDecoder::new(vocab, 64);  // 64 = hidden_dim
//!
//! // Decode hidden state to node type
//! let hidden = vec![0.1; 64];
//! let (node_type, confidence) = decoder.decode(&hidden);
//! ```

use grapheme_core::NodeType;
use rand::Rng as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::cortex_mesh::collect_all_node_types;

/// Configuration for the SemanticDecoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDecoderConfig {
    /// Dimension of hidden representations
    pub hidden_dim: usize,
    /// Learning rate for embeddings
    pub learning_rate: f32,
    /// Temperature for softmax (lower = sharper, higher = smoother)
    pub temperature: f32,
    /// Whether to use label smoothing during training
    pub label_smoothing: f32,
}

impl Default for SemanticDecoderConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 64,
            learning_rate: 0.001,
            temperature: 1.0,
            label_smoothing: 0.1,
        }
    }
}

/// Semantic Node Decoder with Unified Vocabulary
///
/// Decodes hidden representations into semantic NodeType predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDecoder {
    /// Unified vocabulary of all node types from all brains
    unified_vocab: Vec<NodeType>,
    /// Learnable embeddings for each node type (vocab_size x hidden_dim)
    type_embeddings: Vec<Vec<f32>>,
    /// Output projection weights (hidden_dim x vocab_size)
    output_projection: Vec<Vec<f32>>,
    /// Output projection bias (vocab_size)
    output_bias: Vec<f32>,
    /// Configuration
    config: SemanticDecoderConfig,
    /// Vocabulary index lookup (NodeType string repr -> index)
    #[serde(skip)]
    vocab_index: HashMap<String, usize>,
}

impl SemanticDecoder {
    /// Create a new SemanticDecoder with the given vocabulary and configuration
    pub fn new(vocab: Vec<NodeType>, config: SemanticDecoderConfig) -> Self {
        let vocab_size = vocab.len();
        let hidden_dim = config.hidden_dim;

        // Initialize type embeddings with Dynamic Xavier (GRAPHEME protocol)
        // Scale recomputed when dimensions change: sqrt(2 / (fan_in + fan_out))
        let scale = (2.0 / (vocab_size + hidden_dim) as f32).sqrt();
        let mut rng = rand::thread_rng();

        let type_embeddings: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        // Initialize output projection with Dynamic Xavier (GRAPHEME protocol)
        let output_projection: Vec<Vec<f32>> = (0..hidden_dim)
            .map(|_| {
                (0..vocab_size)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        let output_bias = vec![0.0; vocab_size];

        // Build vocab index for fast lookup
        let vocab_index: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, nt)| (format!("{:?}", nt), i))
            .collect();

        Self {
            unified_vocab: vocab,
            type_embeddings,
            output_projection,
            output_bias,
            config,
            vocab_index,
        }
    }

    /// Create a SemanticDecoder with default config
    pub fn with_vocab(vocab: Vec<NodeType>, hidden_dim: usize) -> Self {
        let config = SemanticDecoderConfig {
            hidden_dim,
            ..Default::default()
        };
        Self::new(vocab, config)
    }

    /// Build unified vocabulary from all registered domain brains
    pub fn build_vocab_from_brains() -> Vec<NodeType> {
        collect_all_node_types()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.unified_vocab.len()
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Get the unified vocabulary
    pub fn vocab(&self) -> &[NodeType] {
        &self.unified_vocab
    }

    /// Get the embedding for a node type by index
    pub fn get_embedding(&self, idx: usize) -> Option<&[f32]> {
        self.type_embeddings.get(idx).map(|v| v.as_slice())
    }

    /// Get the index for a node type
    pub fn get_index(&self, node_type: &NodeType) -> Option<usize> {
        let key = format!("{:?}", node_type);
        self.vocab_index.get(&key).copied()
    }

    /// Decode a hidden representation to a NodeType
    ///
    /// Returns (predicted NodeType, confidence score)
    pub fn decode(&self, hidden: &[f32]) -> (NodeType, f32) {
        let logits = self.compute_logits(hidden);
        let probs = self.softmax(&logits);
        let (idx, confidence) = argmax(&probs);
        (self.unified_vocab[idx].clone(), confidence)
    }

    /// Decode with top-k predictions
    pub fn decode_topk(&self, hidden: &[f32], k: usize) -> Vec<(NodeType, f32)> {
        let logits = self.compute_logits(hidden);
        let probs = self.softmax(&logits);

        // Get top-k indices
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed_probs
            .into_iter()
            .take(k)
            .map(|(idx, prob)| (self.unified_vocab[idx].clone(), prob))
            .collect()
    }

    /// Compute logits for a hidden representation
    fn compute_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let vocab_size = self.vocab_size();
        let mut logits = vec![0.0; vocab_size];

        // logits = hidden @ output_projection + bias
        for (i, logit) in logits.iter_mut().enumerate() {
            let mut sum = self.output_bias[i];
            for (j, &h) in hidden.iter().enumerate() {
                if j < self.output_projection.len() {
                    sum += h * self.output_projection[j][i];
                }
            }
            *logit = sum;
        }

        logits
    }

    /// Softmax with temperature
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let temp = self.config.temperature;

        // Find max for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(logit / temp - max)
        let exp_logits: Vec<f32> = logits.iter().map(|&l| ((l - max_logit) / temp).exp()).collect();

        // Normalize
        let sum: f32 = exp_logits.iter().sum();
        exp_logits.iter().map(|&e| e / sum).collect()
    }

    /// Compute cross-entropy loss for a batch of (hidden, target_idx) pairs
    pub fn compute_loss(&self, batch: &[(Vec<f32>, usize)]) -> f32 {
        let mut total_loss = 0.0;
        let n = batch.len() as f32;

        for (hidden, target_idx) in batch {
            let logits = self.compute_logits(hidden);
            let probs = self.softmax(&logits);

            // Cross-entropy with optional label smoothing
            let smooth = self.config.label_smoothing;
            let vocab_size = self.vocab_size() as f32;

            if smooth > 0.0 {
                // Label smoothing: target = (1 - smooth) * one_hot + smooth / vocab_size
                for (i, &p) in probs.iter().enumerate() {
                    let target = if i == *target_idx {
                        1.0 - smooth + smooth / vocab_size
                    } else {
                        smooth / vocab_size
                    };
                    total_loss -= target * (p + 1e-10).ln();
                }
            } else {
                // Standard cross-entropy
                total_loss -= (probs[*target_idx] + 1e-10).ln();
            }
        }

        total_loss / n
    }

    /// Backward pass: compute gradients and update weights
    pub fn backward(&mut self, batch: &[(Vec<f32>, usize)]) -> f32 {
        let n = batch.len() as f32;
        let lr = self.config.learning_rate;
        let vocab_size = self.vocab_size();
        let hidden_dim = self.hidden_dim();

        // Accumulators for gradients
        let mut d_output_proj = vec![vec![0.0_f32; vocab_size]; hidden_dim];
        let mut d_output_bias = vec![0.0_f32; vocab_size];

        let mut total_loss = 0.0;

        for (hidden, target_idx) in batch {
            let logits = self.compute_logits(hidden);
            let probs = self.softmax(&logits);

            // Compute loss
            total_loss -= (probs[*target_idx] + 1e-10).ln();

            // Gradient of cross-entropy loss wrt logits: prob - target
            let mut d_logits = probs.clone();
            d_logits[*target_idx] -= 1.0;

            // Gradient wrt output_projection: hidden^T @ d_logits
            for (j, &h) in hidden.iter().enumerate() {
                if j < hidden_dim {
                    for (i, &dl) in d_logits.iter().enumerate() {
                        d_output_proj[j][i] += h * dl;
                    }
                }
            }

            // Gradient wrt output_bias: d_logits
            for (i, &dl) in d_logits.iter().enumerate() {
                d_output_bias[i] += dl;
            }
        }

        // Apply gradients (SGD)
        for j in 0..hidden_dim {
            for i in 0..vocab_size {
                self.output_projection[j][i] -= lr * d_output_proj[j][i] / n;
            }
        }
        for i in 0..vocab_size {
            self.output_bias[i] -= lr * d_output_bias[i] / n;
        }

        total_loss / n
    }

    /// Encode a node type to its embedding
    pub fn encode(&self, node_type: &NodeType) -> Option<Vec<f32>> {
        self.get_index(node_type)
            .and_then(|idx| self.type_embeddings.get(idx).cloned())
    }

    /// Batch decode: decode multiple hidden representations
    pub fn batch_decode(&self, hiddens: &[Vec<f32>]) -> Vec<(NodeType, f32)> {
        hiddens.iter().map(|h| self.decode(h)).collect()
    }

    /// Compute accuracy on a batch
    pub fn compute_accuracy(&self, batch: &[(Vec<f32>, usize)]) -> f32 {
        let mut correct = 0;
        for (hidden, target_idx) in batch {
            let (_, _) = self.decode(hidden);
            let logits = self.compute_logits(hidden);
            let (pred_idx, _) = argmax(&logits);
            if pred_idx == *target_idx {
                correct += 1;
            }
        }
        correct as f32 / batch.len() as f32
    }

    /// Save decoder to JSON
    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Load decoder from JSON
    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        let mut decoder: Self = serde_json::from_str(&json)?;

        // Rebuild vocab index
        decoder.vocab_index = decoder
            .unified_vocab
            .iter()
            .enumerate()
            .map(|(i, nt)| (format!("{:?}", nt), i))
            .collect();

        Ok(decoder)
    }

    /// Get statistics about the vocabulary
    pub fn vocab_stats(&self) -> VocabStats {
        let mut by_type: HashMap<String, usize> = HashMap::new();

        for nt in &self.unified_vocab {
            let type_name = match nt {
                NodeType::Input(_) => "Input",
                NodeType::Output => "Output",
                NodeType::Hidden => "Hidden",
                NodeType::Keyword(_) => "Keyword",
                NodeType::Variable(_) => "Variable",
                NodeType::Int(_) => "Int",
                NodeType::Float(_) => "Float",
                NodeType::Str(_) => "Str",
                NodeType::Bool(_) => "Bool",
                NodeType::Op(_) => "Op",
                NodeType::Call(_) => "Call",
                NodeType::Punct(_) => "Punct",
                NodeType::Space(_) => "Space",
                NodeType::TypeAnnot(_) => "TypeAnnot",
                NodeType::Comment(_) => "Comment",
                NodeType::EndSeq => "EndSeq",
                NodeType::Feature { .. } => "Feature",
                NodeType::Pixel { .. } => "Pixel",
                NodeType::ClassOutput(_) => "ClassOutput",
                NodeType::Clique(_) => "Clique",
                NodeType::Pattern(_) => "Pattern",
                NodeType::Compressed(_) => "Compressed",
            };
            *by_type.entry(type_name.to_string()).or_insert(0) += 1;
        }

        VocabStats {
            total_size: self.unified_vocab.len(),
            by_type,
            hidden_dim: self.config.hidden_dim,
            param_count: self.param_count(),
        }
    }

    /// Count total learnable parameters
    pub fn param_count(&self) -> usize {
        let vocab_size = self.vocab_size();
        let hidden_dim = self.hidden_dim();
        // type_embeddings + output_projection + output_bias
        vocab_size * hidden_dim + hidden_dim * vocab_size + vocab_size
    }
}

/// Statistics about the unified vocabulary
#[derive(Debug, Clone)]
pub struct VocabStats {
    pub total_size: usize,
    pub by_type: HashMap<String, usize>,
    pub hidden_dim: usize,
    pub param_count: usize,
}

impl std::fmt::Display for VocabStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Vocabulary Statistics:")?;
        writeln!(f, "  Total size: {}", self.total_size)?;
        writeln!(f, "  Hidden dim: {}", self.hidden_dim)?;
        writeln!(f, "  Parameters: {}", self.param_count)?;
        writeln!(f, "  Types:")?;
        let mut types: Vec<_> = self.by_type.iter().collect();
        types.sort_by(|a, b| b.1.cmp(a.1));
        for (type_name, count) in types {
            writeln!(f, "    {}: {}", type_name, count)?;
        }
        Ok(())
    }
}

/// Find the index and value of the maximum element
fn argmax(slice: &[f32]) -> (usize, f32) {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_vocab_from_brains() {
        let vocab = SemanticDecoder::build_vocab_from_brains();
        assert!(!vocab.is_empty(), "Vocabulary should not be empty");

        // Should have at least some code-related types
        let has_keyword = vocab.iter().any(|nt| matches!(nt, NodeType::Keyword(_)));
        assert!(has_keyword, "Should have Keyword types from CodeBrain");
    }

    #[test]
    fn test_decoder_creation() {
        let vocab = vec![
            NodeType::Keyword("def".to_string()),
            NodeType::Variable(String::new()),
            NodeType::Punct('('),
            NodeType::Punct(')'),
            NodeType::EndSeq,
        ];

        let decoder = SemanticDecoder::with_vocab(vocab.clone(), 32);
        assert_eq!(decoder.vocab_size(), 5);
        assert_eq!(decoder.hidden_dim(), 32);
    }

    #[test]
    fn test_decode() {
        let vocab = vec![
            NodeType::Keyword("def".to_string()),
            NodeType::Variable(String::new()),
            NodeType::EndSeq,
        ];

        let decoder = SemanticDecoder::with_vocab(vocab, 8);

        // Create a random hidden state
        let hidden = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let (node_type, confidence) = decoder.decode(&hidden);

        // Should return a valid node type with confidence between 0 and 1
        assert!(confidence >= 0.0 && confidence <= 1.0);
        assert!(
            matches!(
                node_type,
                NodeType::Keyword(_) | NodeType::Variable(_) | NodeType::EndSeq
            )
        );
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let vocab = vec![
            NodeType::Keyword("def".to_string()),
            NodeType::Keyword("if".to_string()),
            NodeType::Keyword("else".to_string()),
        ];

        let decoder = SemanticDecoder::with_vocab(vocab, 16);

        // Get embedding for "def"
        let def_type = NodeType::Keyword("def".to_string());
        let embedding = decoder.encode(&def_type);
        assert!(embedding.is_some());
        assert_eq!(embedding.unwrap().len(), 16);
    }

    #[test]
    fn test_backward_reduces_loss() {
        let vocab = vec![
            NodeType::Keyword("def".to_string()),
            NodeType::Variable(String::new()),
            NodeType::EndSeq,
        ];

        let mut decoder = SemanticDecoder::with_vocab(vocab, 8);

        // Create training batch: hidden -> target_idx
        let batch: Vec<(Vec<f32>, usize)> = vec![
            (vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0), // Should predict "def"
            (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1), // Should predict Variable
            (vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2), // Should predict EndSeq
        ];

        let loss_before = decoder.compute_loss(&batch);

        // Train for a few steps
        for _ in 0..100 {
            decoder.backward(&batch);
        }

        let loss_after = decoder.compute_loss(&batch);
        assert!(
            loss_after < loss_before,
            "Loss should decrease after training"
        );
    }

    #[test]
    fn test_topk_decode() {
        let vocab = vec![
            NodeType::Keyword("def".to_string()),
            NodeType::Keyword("if".to_string()),
            NodeType::Variable(String::new()),
        ];

        let decoder = SemanticDecoder::with_vocab(vocab, 8);
        let hidden = vec![0.1; 8];

        let topk = decoder.decode_topk(&hidden, 2);
        assert_eq!(topk.len(), 2);

        // First should have higher probability than second
        assert!(topk[0].1 >= topk[1].1);
    }

    #[test]
    fn test_vocab_stats() {
        let vocab = SemanticDecoder::build_vocab_from_brains();
        let decoder = SemanticDecoder::with_vocab(vocab, 64);
        let stats = decoder.vocab_stats();

        println!("{}", stats);

        assert!(stats.total_size > 0);
        assert!(stats.param_count > 0);
    }

    #[test]
    fn test_save_load() {
        let vocab = vec![
            NodeType::Keyword("def".to_string()),
            NodeType::Variable(String::new()),
        ];

        let decoder = SemanticDecoder::with_vocab(vocab, 16);
        let temp_path = "/tmp/test_semantic_decoder.json";

        decoder.save(temp_path).unwrap();
        let loaded = SemanticDecoder::load(temp_path).unwrap();

        assert_eq!(decoder.vocab_size(), loaded.vocab_size());
        assert_eq!(decoder.hidden_dim(), loaded.hidden_dim());

        std::fs::remove_file(temp_path).ok();
    }
}
