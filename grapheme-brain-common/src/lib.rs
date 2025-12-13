//! # grapheme-brain-common
//!
//! Common abstractions for GRAPHEME cognitive brain implementations.
//!
//! This crate provides reusable generic types and utilities to reduce code
//! duplication across domain-specific brain crates (math, code, law, music, chem).
//!
//! ## Key Components
//!
//! - [`ActivatedNode<T>`] - Generic node wrapper with activation field
//! - [`TypedGraph<N, E>`] - Generic graph wrapper for domain-specific graphs
//! - [`TextTransformRule`] - Reusable text-based transformation rules
//! - [`KeywordCapabilityDetector`] - Domain detection via keyword matching
//! - [`TextNormalizer`] - Input text normalization
//! - [`GraphAutoencoder`] - Trait for Stage 1 brain autoencoder training
//! - [`LatentGraph`] - Serializable latent graph representation

mod node;
mod graph;
mod transform;
mod traits;
mod utils;
mod autoencoder;
mod text_brain;

pub use node::ActivatedNode;
pub use graph::TypedGraph;
pub use transform::{TextTransformRule, TransformRuleSet};
pub use traits::{BaseDomainBrain, DomainConfig};
pub use utils::{KeywordCapabilityDetector, TextNormalizer, math_normalizer, code_normalizer, legal_normalizer};
pub use autoencoder::{GraphAutoencoder, LatentGraph, EncodedPair, AutoencoderError};
pub use text_brain::TextBrain;

// Re-export commonly used types from dependencies
pub use petgraph::graph::{DiGraph, NodeIndex, EdgeIndex};
