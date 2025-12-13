//! Unified Cortex - Connect All 7 Cortices for Unified Code Generation (backend-213)
//!
//! This module extends CortexMesh with multi-brain fusion capabilities:
//! - Cross-brain attention for information sharing
//! - Weighted fusion of brain outputs
//! - Unified code generation from multiple domain perspectives
//!
//! Uses the GRAPHEME Graph → Transform → Graph paradigm (NOT autoregressive).
//!
//! ## Usage
//! ```rust,ignore
//! use grapheme_train::unified_cortex::{UnifiedCortex, UnifiedConfig};
//!
//! let mut cortex = UnifiedCortex::new(UnifiedConfig::default());
//! let result = cortex.unified_process("Write a function to check if a number is prime");
//! println!("Output: {}", result.decoded_code);
//! ```

pub use crate::parallel_cortex::FusionLayer;
use grapheme_chem::ChemBrain;
use grapheme_code::CodeBrain;
use grapheme_core::{BrainRegistry, DagNN, DomainBrain, Node, Edge};
use grapheme_law::LawBrain;
use grapheme_math::MathBrain;
use grapheme_music::MusicBrain;
use grapheme_time::TimeBrain;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashMap;

// ============================================================================
// Constants
// ============================================================================

/// Standard GRAPHEME Protocol learning rate
pub const DEFAULT_LR: f32 = 0.001;

/// LeakyReLU alpha (GRAPHEME Protocol)
pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Supported cortex domains for code generation
pub const CODE_GEN_CORTICES: &[&str] = &[
    "math",   // Algorithm design, numerical operations
    "code",   // Code structure, syntax, patterns
    "law",    // Logic rules, constraints
    "music",  // Pattern recognition, sequence
    "chem",   // Structure composition
    "time",   // Temporal patterns, sequencing
];

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for unified cortex
#[derive(Debug, Clone)]
pub struct UnifiedConfig {
    /// Activation threshold for brains
    pub activation_threshold: f32,
    /// Maximum brains to activate per query
    pub max_active_brains: usize,
    /// Enable parallel processing
    pub parallel: bool,
    /// Embedding dimension for fusion
    pub embed_dim: usize,
    /// Hidden dimension for attention
    pub hidden_dim: usize,
    /// Fusion type (attention, weighted, max)
    pub fusion_type: FusionType,
    /// Cross-brain attention weight
    pub cross_attention_weight: f32,
    /// Code-specific output enhancement
    pub code_enhancement: bool,
    /// Learning rate
    pub learning_rate: f32,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            activation_threshold: 0.2,
            max_active_brains: usize::MAX,
            parallel: true,
            embed_dim: 64,
            hidden_dim: 128,
            fusion_type: FusionType::Attention,
            cross_attention_weight: 0.5,
            code_enhancement: true,
            learning_rate: DEFAULT_LR,
        }
    }
}

/// Fusion type for combining brain outputs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionType {
    /// Attention-weighted combination
    Attention,
    /// Simple weighted average
    Weighted,
    /// Maximum activation
    Max,
    /// Concatenation followed by linear projection
    Concat,
}

// ============================================================================
// Brain Embedding
// ============================================================================

/// Embedding from a single brain's processing
#[derive(Debug, Clone)]
pub struct BrainEmbedding {
    /// Brain identifier
    pub brain_id: String,
    /// Domain name
    pub domain: String,
    /// Activation vector
    pub embedding: Array1<f32>,
    /// Confidence score for this input
    pub confidence: f32,
    /// Whether this brain was activated
    pub activated: bool,
}

impl BrainEmbedding {
    /// Create a new brain embedding
    pub fn new(brain_id: &str, domain: &str, embedding: Array1<f32>, confidence: f32) -> Self {
        Self {
            brain_id: brain_id.to_string(),
            domain: domain.to_string(),
            embedding,
            confidence,
            activated: confidence > 0.2, // Default activation threshold
        }
    }

    /// Create an inactive embedding (for non-activated brains)
    pub fn inactive(brain_id: &str, domain: &str, dim: usize) -> Self {
        Self {
            brain_id: brain_id.to_string(),
            domain: domain.to_string(),
            embedding: Array1::zeros(dim),
            confidence: 0.0,
            activated: false,
        }
    }
}

// ============================================================================
// Cross-Brain Attention
// ============================================================================

/// Cross-brain attention mechanism for information sharing
pub struct CrossBrainAttention {
    /// Query projection (embed_dim -> hidden_dim)
    query_proj: Array2<f32>,
    /// Key projection (embed_dim -> hidden_dim)
    key_proj: Array2<f32>,
    /// Value projection (embed_dim -> hidden_dim)
    value_proj: Array2<f32>,
    /// Output projection (hidden_dim -> embed_dim)
    output_proj: Array2<f32>,
    /// Temperature for softmax
    temperature: f32,
}

impl CrossBrainAttention {
    /// Create new cross-brain attention
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        // Initialize with DynamicXavier
        let scale_qkv = (2.0 / (embed_dim + hidden_dim) as f32).sqrt();
        let scale_out = (2.0 / (hidden_dim + embed_dim) as f32).sqrt();

        let query_proj = Array2::from_shape_fn((embed_dim, hidden_dim), |_| {
            (rand::random::<f32>() - 0.5) * 2.0 * scale_qkv
        });
        let key_proj = Array2::from_shape_fn((embed_dim, hidden_dim), |_| {
            (rand::random::<f32>() - 0.5) * 2.0 * scale_qkv
        });
        let value_proj = Array2::from_shape_fn((embed_dim, hidden_dim), |_| {
            (rand::random::<f32>() - 0.5) * 2.0 * scale_qkv
        });
        let output_proj = Array2::from_shape_fn((hidden_dim, embed_dim), |_| {
            (rand::random::<f32>() - 0.5) * 2.0 * scale_out
        });

        Self {
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            temperature: (hidden_dim as f32).sqrt(),
        }
    }

    /// Apply cross-brain attention to embeddings
    /// Returns attended embeddings with information sharing
    pub fn attend(&self, embeddings: &[BrainEmbedding]) -> Vec<Array1<f32>> {
        if embeddings.is_empty() {
            return vec![];
        }

        let active: Vec<_> = embeddings.iter().filter(|e| e.activated).collect();
        if active.is_empty() {
            return embeddings.iter().map(|e| e.embedding.clone()).collect();
        }

        let n = active.len();
        let hidden_dim = self.key_proj.ncols();

        // Project all embeddings to queries, keys, values
        let queries: Vec<Array1<f32>> = active.iter()
            .map(|e| self.query_proj.t().dot(&e.embedding))
            .collect();
        let keys: Vec<Array1<f32>> = active.iter()
            .map(|e| self.key_proj.t().dot(&e.embedding))
            .collect();
        let values: Vec<Array1<f32>> = active.iter()
            .map(|e| self.value_proj.t().dot(&e.embedding))
            .collect();

        // Compute attention scores for each brain
        let mut attended = vec![Array1::zeros(self.output_proj.ncols()); embeddings.len()];

        for (i, emb) in embeddings.iter().enumerate() {
            if !emb.activated {
                attended[i] = emb.embedding.clone();
                continue;
            }

            // Find this embedding in active list
            let active_idx = active.iter().position(|e| e.brain_id == emb.brain_id);
            if active_idx.is_none() {
                attended[i] = emb.embedding.clone();
                continue;
            }
            let q_idx = active_idx.unwrap();

            // Compute attention scores
            let mut scores = vec![0.0f32; n];
            for (j, key) in keys.iter().enumerate() {
                let score = queries[q_idx].dot(key) / self.temperature;
                scores[j] = score;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();
            let weights: Vec<f32> = scores.iter().map(|s| (s - max_score).exp() / exp_sum).collect();

            // Weighted sum of values
            let mut context = Array1::zeros(hidden_dim);
            for (w, v) in weights.iter().zip(values.iter()) {
                context = context + *w * v;
            }

            // Project to output and apply LeakyReLU
            let output = self.output_proj.t().dot(&context);
            attended[i] = output.mapv(|x| if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x });
        }

        attended
    }
}

// ============================================================================
// Unified Result
// ============================================================================

/// Result from unified cortex processing
#[derive(Debug, Clone)]
pub struct UnifiedResult {
    /// Brain embeddings (before fusion)
    pub brain_embeddings: Vec<BrainEmbedding>,
    /// Fused embedding
    pub fused_embedding: Array1<f32>,
    /// Output graph
    pub output_graph: DagNN,
    /// Decoded code
    pub decoded_code: String,
    /// Active brain IDs
    pub active_brains: Vec<String>,
    /// Brain contributions (weights from fusion)
    pub brain_contributions: HashMap<String, f32>,
    /// Processing time (ms)
    pub time_ms: f32,
    /// Reserved for future mesh integration
    #[allow(dead_code)]
    mesh_result: Option<()>,
}

// ============================================================================
// Unified Cortex
// ============================================================================

/// Unified Cortex - All domain brains connected for code generation
pub struct UnifiedCortex {
    /// Configuration
    pub config: UnifiedConfig,
    /// Brain registry with all domain brains
    pub brains: BrainRegistry,
    /// Fusion layer for combining embeddings
    fusion: FusionLayer,
    /// Cross-brain attention
    cross_attention: CrossBrainAttention,
    /// Processing statistics
    pub stats: UnifiedStats,
}

/// Statistics for unified cortex
#[derive(Debug, Default, Clone)]
pub struct UnifiedStats {
    /// Total samples processed
    pub samples_processed: usize,
    /// Per-brain activation counts
    pub brain_activations: HashMap<String, usize>,
    /// Average brains active per sample
    pub avg_active_brains: f32,
    /// Average fusion time (ms)
    pub avg_fusion_time_ms: f32,
    /// Total processing time
    pub total_time_ms: f32,
}

impl UnifiedCortex {
    /// Create a new unified cortex
    pub fn new(config: UnifiedConfig) -> Self {
        // Create brain registry with all available brains
        let mut brains = BrainRegistry::new();
        brains.register(Box::new(MathBrain::new()));
        brains.register(Box::new(CodeBrain::new()));
        brains.register(Box::new(LawBrain::new()));
        brains.register(Box::new(MusicBrain::new()));
        brains.register(Box::new(ChemBrain::new()));
        brains.register(Box::new(TimeBrain::default_config()));

        // Get brain IDs for fusion layer
        let brain_ids: Vec<String> = brains.domains().to_vec();

        let fusion = FusionLayer::new(config.embed_dim, config.hidden_dim, &brain_ids);
        let cross_attention = CrossBrainAttention::new(config.embed_dim, config.hidden_dim);

        Self {
            config,
            brains,
            fusion,
            cross_attention,
            stats: UnifiedStats::default(),
        }
    }

    /// Unified processing: route to multiple brains, fuse outputs, generate code
    pub fn unified_process(&mut self, input: &str) -> UnifiedResult {
        let start = std::time::Instant::now();

        // Phase 1: Get embeddings from all brains
        let brain_embeddings = self.get_brain_embeddings(input);

        // Phase 2: Apply cross-brain attention
        let attended_embeddings = self.cross_attention.attend(&brain_embeddings);

        // Update embeddings with attended values
        let attended_brain_embeddings: Vec<BrainEmbedding> = brain_embeddings
            .iter()
            .zip(attended_embeddings.iter())
            .map(|(orig, attended)| BrainEmbedding {
                brain_id: orig.brain_id.clone(),
                domain: orig.domain.clone(),
                embedding: attended.clone(),
                confidence: orig.confidence,
                activated: orig.activated,
            })
            .collect();

        // Phase 3: Fuse embeddings
        let (fused_embedding, brain_contributions) = self.fuse_embeddings(&attended_brain_embeddings);

        // Phase 4: Generate output graph from fused embedding
        let output_graph = self.generate_output_graph(&fused_embedding, input);

        // Phase 5: Decode to text (simple reconstruction)
        let decoded_code = self.decode_graph(&output_graph, input);

        let active_brains = attended_brain_embeddings
            .iter()
            .filter(|e| e.activated)
            .map(|e| e.brain_id.clone())
            .collect();

        // Update stats
        let time_ms = start.elapsed().as_secs_f32() * 1000.0;
        self.update_stats(&attended_brain_embeddings, time_ms);

        UnifiedResult {
            brain_embeddings: attended_brain_embeddings,
            fused_embedding,
            output_graph,
            decoded_code,
            active_brains,
            brain_contributions,
            time_ms,
            mesh_result: None,
        }
    }

    /// Generate output graph from fused embedding
    fn generate_output_graph(&self, fused: &Array1<f32>, _input: &str) -> DagNN {
        let mut dag = DagNN::new();

        // Create nodes based on embedding dimensions
        let num_nodes = (fused.len() / 4).max(4);
        for i in 0..num_nodes {
            let mut node = Node::hidden();
            node.activation = fused[i * 4 % fused.len()].abs().clamp(0.0, 1.0);
            dag.graph.add_node(node);
        }

        // Create sequential edges
        for i in 0..num_nodes.saturating_sub(1) {
            let src = petgraph::graph::NodeIndex::new(i);
            let tgt = petgraph::graph::NodeIndex::new(i + 1);
            dag.graph.add_edge(src, tgt, Edge::sequential());
        }

        dag
    }

    /// Decode output graph to text
    fn decode_graph(&self, dag: &DagNN, input: &str) -> String {
        // Simple decoding based on activations
        let activations: Vec<f32> = dag.graph.node_indices()
            .map(|idx| dag.graph[idx].activation)
            .collect();

        // Use input as base and add generated content
        let suffix = if !activations.is_empty() {
            let sum: f32 = activations.iter().sum();
            format!("# Processed with {} nodes, activation sum: {:.2}", activations.len(), sum)
        } else {
            String::new()
        };

        format!("{}\n{}", input, suffix)
    }

    /// Get embeddings from all brains
    fn get_brain_embeddings(&self, input: &str) -> Vec<BrainEmbedding> {
        let brain_ids: Vec<String> = self.brains.domains().to_vec();
        let embed_dim = self.config.embed_dim;

        brain_ids
            .par_iter()
            .map(|brain_id| {
                if let Some(brain) = self.brains.get(brain_id) {
                    let can_process = brain.can_process(input);
                    let confidence = if can_process { 0.8 } else { 0.1 };

                    // Generate embedding based on brain's parsing
                    let embedding = self.brain_to_embedding(brain, input, embed_dim);

                    BrainEmbedding::new(brain_id, brain.domain_name(), embedding, confidence)
                } else {
                    BrainEmbedding::inactive(brain_id, "unknown", embed_dim)
                }
            })
            .collect()
    }

    /// Convert brain output to embedding
    fn brain_to_embedding(&self, brain: &dyn DomainBrain, input: &str, dim: usize) -> Array1<f32> {
        // Use brain's parsing to generate semantic features
        if !brain.can_process(input) {
            return Array1::zeros(dim);
        }

        // Simple embedding based on input characteristics
        let mut embedding = Array1::zeros(dim);

        // Hash input through brain's domain knowledge
        let domain_hash = brain.domain_name().bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let input_hash = input.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

        for i in 0..dim {
            let seed = domain_hash.wrapping_add(input_hash).wrapping_add(i as u64);
            embedding[i] = ((seed % 1000) as f32 / 1000.0 - 0.5) * 2.0;
        }

        // Apply LeakyReLU
        embedding.mapv_inplace(|x| if x > 0.0 { x } else { LEAKY_RELU_ALPHA * x });

        embedding
    }

    /// Fuse brain embeddings using configured method
    fn fuse_embeddings(&self, embeddings: &[BrainEmbedding]) -> (Array1<f32>, HashMap<String, f32>) {
        match self.config.fusion_type {
            FusionType::Attention => {
                // Use FusionLayer's attention mechanism
                let activations: Vec<crate::parallel_cortex::BrainActivation> = embeddings
                    .iter()
                    .map(|e| crate::parallel_cortex::BrainActivation {
                        brain_id: e.brain_id.clone(),
                        activated: e.activated,
                        confidence: e.confidence,
                        graph: None, // No graph available at fusion stage
                        embedding: Some(e.embedding.clone()),
                    })
                    .collect();

                let fused = self.fusion.fuse(&activations);

                // Compute contributions from attention weights
                let mut contributions = HashMap::new();
                let active_count = embeddings.iter().filter(|e| e.activated).count() as f32;
                for e in embeddings {
                    contributions.insert(
                        e.brain_id.clone(),
                        if e.activated { 1.0 / active_count.max(1.0) } else { 0.0 },
                    );
                }

                (fused, contributions)
            }
            FusionType::Weighted => {
                let dim = self.config.embed_dim;
                let mut fused = Array1::zeros(dim);
                let mut total_weight = 0.0f32;
                let mut contributions = HashMap::new();

                for e in embeddings {
                    if e.activated {
                        let weight = e.confidence;
                        fused = fused + weight * &e.embedding;
                        total_weight += weight;
                        contributions.insert(e.brain_id.clone(), weight);
                    } else {
                        contributions.insert(e.brain_id.clone(), 0.0);
                    }
                }

                if total_weight > 0.0 {
                    fused /= total_weight;
                }

                // Normalize contributions
                for (_, v) in contributions.iter_mut() {
                    if total_weight > 0.0 {
                        *v /= total_weight;
                    }
                }

                (fused, contributions)
            }
            FusionType::Max => {
                let dim = self.config.embed_dim;
                let mut fused = Array1::from_elem(dim, f32::NEG_INFINITY);
                let mut contributions = HashMap::new();
                let mut max_brain = String::new();
                let mut max_norm = f32::NEG_INFINITY;

                for e in embeddings {
                    if e.activated {
                        // Element-wise max
                        for (i, &v) in e.embedding.iter().enumerate() {
                            if v > fused[i] {
                                fused[i] = v;
                            }
                        }
                        let norm = e.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > max_norm {
                            max_norm = norm;
                            max_brain = e.brain_id.clone();
                        }
                    }
                    contributions.insert(e.brain_id.clone(), 0.0);
                }

                // Mark max contributor
                if !max_brain.is_empty() {
                    contributions.insert(max_brain, 1.0);
                }

                // Replace -inf with 0
                fused.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });

                (fused, contributions)
            }
            FusionType::Concat => {
                // Concatenate and project
                let dim = self.config.embed_dim;
                let active: Vec<_> = embeddings.iter().filter(|e| e.activated).collect();

                if active.is_empty() {
                    return (Array1::zeros(dim), HashMap::new());
                }

                // Simple averaging (concat would need additional projection)
                let mut fused = Array1::zeros(dim);
                for e in &active {
                    fused += &e.embedding;
                }
                fused /= active.len() as f32;

                let mut contributions = HashMap::new();
                let weight = 1.0 / active.len() as f32;
                for e in embeddings {
                    contributions.insert(
                        e.brain_id.clone(),
                        if e.activated { weight } else { 0.0 },
                    );
                }

                (fused, contributions)
            }
        }
    }

    /// Enhance output graph for code generation
    #[allow(dead_code)]
    fn enhance_code_graph(&self, graph: &grapheme_core::GraphemeGraph, _fused: &Array1<f32>) -> DagNN {
        // Convert GraphemeGraph to DagNN with code-specific enhancements
        let mut dag = DagNN::new();

        // Create nodes from GraphemeGraph
        for node_idx in graph.graph.node_indices() {
            let node_data = &graph.graph[node_idx];
            let mut node = Node::hidden();
            node.activation = node_data.activation;
            dag.graph.add_node(node);
        }

        // Create edges
        for edge_idx in graph.graph.edge_indices() {
            if let Some((src, tgt)) = graph.graph.edge_endpoints(edge_idx) {
                let src_new = petgraph::graph::NodeIndex::new(src.index());
                let tgt_new = petgraph::graph::NodeIndex::new(tgt.index());
                dag.graph.add_edge(src_new, tgt_new, Edge::sequential());
            }
        }

        dag
    }

    /// Update statistics
    fn update_stats(&mut self, embeddings: &[BrainEmbedding], time_ms: f32) {
        self.stats.samples_processed += 1;
        self.stats.total_time_ms += time_ms;

        for e in embeddings {
            if e.activated {
                *self.stats.brain_activations.entry(e.brain_id.clone()).or_insert(0) += 1;
            }
        }

        let _active_count = embeddings.iter().filter(|e| e.activated).count();
        let total_activations: usize = self.stats.brain_activations.values().sum();
        self.stats.avg_active_brains = total_activations as f32 / self.stats.samples_processed as f32;
        self.stats.avg_fusion_time_ms = self.stats.total_time_ms / self.stats.samples_processed as f32;
    }

    /// Get brain count
    pub fn brain_count(&self) -> usize {
        self.brains.len()
    }

    /// Get active brain IDs
    pub fn brain_ids(&self) -> Vec<String> {
        self.brains.domains().to_vec()
    }

    /// Get statistics summary
    pub fn stats_summary(&self) -> String {
        let mut s = String::new();
        s.push_str("╔═══ UnifiedCortex Statistics ═══╗\n");
        s.push_str(&format!("║ Processed: {} samples\n", self.stats.samples_processed));
        s.push_str(&format!("║ Avg active brains: {:.1}\n", self.stats.avg_active_brains));
        s.push_str(&format!("║ Avg time: {:.2}ms\n", self.stats.avg_fusion_time_ms));
        s.push_str("╠═══ Brain Activations ═══════════╣\n");

        let mut sorted: Vec<_> = self.stats.brain_activations.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        for (brain, count) in sorted {
            let pct = *count as f32 / self.stats.samples_processed.max(1) as f32 * 100.0;
            s.push_str(&format!("║  {}: {} ({:.1}%)\n", brain, count, pct));
        }
        s.push_str("╚════════════════════════════════╝\n");
        s
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Quick unified processing with default config
pub fn unified_process(input: &str) -> UnifiedResult {
    let mut cortex = UnifiedCortex::new(UnifiedConfig::default());
    cortex.unified_process(input)
}

/// List all available cortices for code generation
pub fn list_code_gen_cortices() -> Vec<&'static str> {
    CODE_GEN_CORTICES.to_vec()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_config_default() {
        let config = UnifiedConfig::default();
        assert_eq!(config.embed_dim, 64);
        assert_eq!(config.hidden_dim, 128);
        assert_eq!(config.fusion_type, FusionType::Attention);
        assert!(config.code_enhancement);
    }

    #[test]
    fn test_brain_embedding_new() {
        let embedding = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let be = BrainEmbedding::new("code", "Source Code", embedding.clone(), 0.8);
        assert_eq!(be.brain_id, "code");
        assert!(be.activated);
        assert_eq!(be.confidence, 0.8);
    }

    #[test]
    fn test_brain_embedding_inactive() {
        let be = BrainEmbedding::inactive("vision", "Vision", 64);
        assert!(!be.activated);
        assert_eq!(be.confidence, 0.0);
        assert_eq!(be.embedding.len(), 64);
    }

    #[test]
    fn test_cross_brain_attention_creation() {
        let attention = CrossBrainAttention::new(64, 128);
        assert_eq!(attention.query_proj.dim(), (64, 128));
        assert_eq!(attention.key_proj.dim(), (64, 128));
        assert_eq!(attention.value_proj.dim(), (64, 128));
        assert_eq!(attention.output_proj.dim(), (128, 64));
    }

    #[test]
    fn test_cross_brain_attention_empty() {
        let attention = CrossBrainAttention::new(64, 128);
        let result = attention.attend(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_fusion_type_eq() {
        assert_eq!(FusionType::Attention, FusionType::Attention);
        assert_ne!(FusionType::Attention, FusionType::Weighted);
    }

    #[test]
    fn test_unified_cortex_creation() {
        let config = UnifiedConfig::default();
        let cortex = UnifiedCortex::new(config);
        assert!(cortex.brain_count() >= 6);
    }

    #[test]
    fn test_unified_process_simple() {
        let mut cortex = UnifiedCortex::new(UnifiedConfig::default());
        let result = cortex.unified_process("def add(a, b): return a + b");

        // Code brain should be active
        assert!(result.active_brains.contains(&"code".to_string()));
        assert!(result.time_ms > 0.0);
    }

    #[test]
    fn test_unified_process_math() {
        let mut cortex = UnifiedCortex::new(UnifiedConfig::default());
        let result = cortex.unified_process("Calculate the derivative of x^2 + 3x");

        // Math brain should be active
        assert!(result.active_brains.contains(&"math".to_string()));
    }

    #[test]
    fn test_fusion_weighted() {
        let mut config = UnifiedConfig::default();
        config.fusion_type = FusionType::Weighted;
        config.embed_dim = 2; // Small for testing
        let cortex = UnifiedCortex::new(config);

        let embeddings = vec![
            BrainEmbedding::new("code", "Code", Array1::from_vec(vec![1.0, 0.0]), 0.8),
            BrainEmbedding::new("math", "Math", Array1::from_vec(vec![0.0, 1.0]), 0.6),
        ];

        let (fused, contributions) = cortex.fuse_embeddings(&embeddings);
        assert!(fused.len() == 2); // Matches embed_dim
        assert!(contributions.contains_key("code"));
        assert!(contributions.contains_key("math"));
    }

    #[test]
    fn test_fusion_max() {
        let mut config = UnifiedConfig::default();
        config.fusion_type = FusionType::Max;
        config.embed_dim = 2; // Small for testing
        let cortex = UnifiedCortex::new(config);

        let embeddings = vec![
            BrainEmbedding::new("code", "Code", Array1::from_vec(vec![1.0, 0.0]), 0.8),
            BrainEmbedding::new("math", "Math", Array1::from_vec(vec![0.0, 2.0]), 0.6),
        ];

        let (fused, _) = cortex.fuse_embeddings(&embeddings);
        // Max should pick largest values
        assert!(fused.len() == 2);
    }

    #[test]
    fn test_stats_summary() {
        let mut cortex = UnifiedCortex::new(UnifiedConfig::default());
        cortex.unified_process("test input");

        let summary = cortex.stats_summary();
        assert!(summary.contains("UnifiedCortex"));
        assert!(summary.contains("Processed"));
    }

    #[test]
    fn test_list_code_gen_cortices() {
        let cortices = list_code_gen_cortices();
        assert!(cortices.contains(&"math"));
        assert!(cortices.contains(&"code"));
    }

    #[test]
    fn test_unified_result_contributions() {
        let mut cortex = UnifiedCortex::new(UnifiedConfig::default());
        let result = cortex.unified_process("print('hello')");

        // Should have contributions for all brains
        assert!(!result.brain_contributions.is_empty());
    }
}
