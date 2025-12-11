//! CortexMesh - Auto-Discovery and Parallel Processing for ALL GRAPHEME Components
//!
//! This module provides compile-time auto-discovery of all brains, modules, and
//! cognitive systems. Just call `CortexMesh::discover()` to get everything
//! meshed and ready for parallel processing.
//!
//! ## PARALLEL BY DEFAULT
//! All operations use Rayon for automatic CPU parallelization.
//! Call `init_parallel()` at program start to configure thread pool.
//!
//! ## Auto-Discovered Components
//! - **Domain Brains** (8): math, code, vision, classification, law, music, chem, time
//! - **Router Modules** (4): text, math, timeseries, vision
//! - **Cognitive Systems** (8): reasoning, metacognition, memory, safety, world, grounding, agency, multimodal
//!
//! ## Usage
//! ```rust,ignore
//! use grapheme_train::cortex_mesh::{CortexMesh, init_parallel};
//!
//! // Initialize parallel processing (call once at start)
//! init_parallel(0); // 0 = auto-detect CPU cores
//!
//! // Auto-discover and mesh ALL components
//! let mut mesh = CortexMesh::discover();
//! println!("Discovered {} brains, {} cognitive systems", mesh.brain_count(), mesh.cognitive_system_count());
//!
//! // Process with ALL brains in parallel (default)
//! let result = mesh.process_parallel("def fibonacci(n): ...");
//! println!("Active brains: {:?}", result.active_brains);
//! ```
//!
//! ## Adding New Components
//! - Add new brains to `BRAIN_FACTORIES`
//! - Add new modules to `MODULE_FACTORIES`
//! - Add new cognitive systems to `COGNITIVE_SYSTEM_FACTORIES`

use grapheme_chem::ChemBrain;
use grapheme_code::CodeBrain;
use grapheme_core::{BrainRegistry, DomainBrain, GraphemeGraph, GraphTransformNet};
use grapheme_law::LawBrain;
use grapheme_math::MathBrain;
use grapheme_music::MusicBrain;
use grapheme_router::{CognitiveModule, CognitiveRouter, MathModule, TextModule, TimeSeriesModule, VisionModule as RouterVisionModule};
use grapheme_time::TimeBrain;
use grapheme_vision::{ClassificationBrain, ClassificationConfig, VisionBrain};

// Cognitive Systems imports
use grapheme_reason::{ReasoningEngine, create_default_reasoning_engine};
use grapheme_meta::{SimpleMetaCognition, create_default_metacognition, SelfModel, create_self_model, AttentionMechanism, create_attention_mechanism};
use grapheme_memory::{MemorySystem, create_default_memory_system};
use grapheme_safety::{SafetyGuard, SafetyGate};
use grapheme_world::{SimpleWorldModel, create_default_world_model};
use grapheme_ground::{SimpleGroundedGraph, SimpleEmbodiedAgent, create_default_grounded_graph, create_default_embodied_agent};
use grapheme_agent::{SimpleAgency, GoalStack, create_default_agent, create_goal_stack};
use grapheme_multimodal::{SimpleMultiModal, create_default_multimodal};

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Once;
use std::time::Instant;

use crate::{compute_structural_loss, StructuralLossConfig};

// ============================================================================
// Parallel Initialization - CALL ONCE AT PROGRAM START
// ============================================================================

static PARALLEL_INIT: Once = Once::new();

/// Initialize parallel processing with Rayon thread pool.
///
/// Call this once at the start of your program. If not called explicitly,
/// `CortexMesh::discover()` will auto-initialize with all available cores.
///
/// # Arguments
/// * `num_threads` - Number of threads (0 = auto-detect all CPU cores)
///
/// # Example
/// ```rust,ignore
/// use grapheme_train::cortex_mesh::init_parallel;
/// init_parallel(0); // Use all cores
/// init_parallel(8); // Use exactly 8 threads
/// ```
pub fn init_parallel(num_threads: usize) -> usize {
    let mut actual_threads = num_threads;

    PARALLEL_INIT.call_once(|| {
        if num_threads == 0 {
            // Auto-detect: use all available cores
            actual_threads = rayon::current_num_threads();
            println!("Parallel CPU: {} threads (auto-detected)", actual_threads);
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .unwrap_or_else(|_| {
                    // Already initialized, that's OK
                });
            actual_threads = num_threads;
            println!("Parallel CPU: {} threads (configured)", actual_threads);
        }
    });

    // Return actual thread count
    rayon::current_num_threads()
}

/// Get current number of parallel threads
pub fn parallel_threads() -> usize {
    rayon::current_num_threads()
}

// ============================================================================
// Brain Factory - Add new brains here for auto-discovery
// ============================================================================

/// All known brain factories - ADD NEW BRAINS HERE
const BRAIN_FACTORIES: &[(&str, fn() -> Box<dyn DomainBrain>)] = &[
    ("math", || Box::new(MathBrain::new())),
    ("code", || Box::new(CodeBrain::new())),
    ("vision", || Box::new(VisionBrain::new())),
    ("classification", || Box::new(ClassificationBrain::new(ClassificationConfig::new(10)))),
    ("law", || Box::new(LawBrain::new())),
    ("music", || Box::new(MusicBrain::new())),
    ("chem", || Box::new(ChemBrain::new())),
    ("time", || Box::new(TimeBrain::default_config())),
    // ADD NEW BRAINS HERE:
    // ("new_brain", || Box::new(NewBrain::new())),
];

/// All known router module factories - ADD NEW MODULES HERE
const MODULE_FACTORIES: &[(&str, fn() -> Box<dyn CognitiveModule>)] = &[
    ("text", || Box::new(TextModule::new())),
    ("math", || Box::new(MathModule::new())),
    ("timeseries", || Box::new(TimeSeriesModule::new())),
    ("vision", || Box::new(RouterVisionModule::new())),
    // ADD NEW MODULES HERE:
    // ("new_module", || Box::new(NewModule::new())),
];

// ============================================================================
// Cognitive Systems Factory - Add new cognitive systems here for auto-discovery
// ============================================================================

/// Cognitive system names for auto-discovery
pub const COGNITIVE_SYSTEM_NAMES: &[&str] = &[
    "reasoning",
    "metacognition",
    "memory",
    "safety",
    "world",
    "grounding",
    "agency",
    "multimodal",
];

/// All cognitive systems auto-discovered - holds the actual instances
pub struct CognitiveSystems {
    /// Reasoning engine for deduction, induction, abduction, analogy
    pub reasoning: ReasoningEngine,
    /// Metacognition for self-monitoring and uncertainty estimation
    pub metacognition: SimpleMetaCognition,
    /// Self-model for introspection
    pub self_model: SelfModel,
    /// Attention mechanism for brain allocation
    pub attention: AttentionMechanism,
    /// Memory system (episodic, semantic, procedural, working)
    pub memory: MemorySystem,
    /// Safety guard for action filtering
    pub safety_guard: SafetyGuard,
    /// Safety gate for input/output filtering
    pub safety_gate: SafetyGate,
    /// World model for prediction and planning
    pub world: SimpleWorldModel,
    /// Grounded graph for symbol grounding
    pub grounding: SimpleGroundedGraph,
    /// Embodied agent for sensorimotor integration
    pub embodied: SimpleEmbodiedAgent,
    /// Agency for goal management and planning
    pub agency: SimpleAgency,
    /// Goal stack for hierarchical goal tracking
    pub goal_stack: GoalStack,
    /// Multimodal integration
    pub multimodal: SimpleMultiModal,
}

impl CognitiveSystems {
    /// Auto-discover and create all cognitive systems
    pub fn discover() -> Self {
        Self {
            reasoning: create_default_reasoning_engine(),
            metacognition: create_default_metacognition(),
            self_model: create_self_model(),
            attention: create_attention_mechanism(),
            memory: create_default_memory_system(),
            safety_guard: SafetyGuard::default(),
            safety_gate: SafetyGate::default(),
            world: create_default_world_model(),
            grounding: create_default_grounded_graph(),
            embodied: create_default_embodied_agent(),
            agency: create_default_agent(),
            goal_stack: create_goal_stack(),
            multimodal: create_default_multimodal(),
        }
    }

    /// Get count of cognitive system categories
    pub fn count() -> usize {
        COGNITIVE_SYSTEM_NAMES.len()
    }

    /// List all cognitive system names
    pub fn list() -> Vec<&'static str> {
        COGNITIVE_SYSTEM_NAMES.to_vec()
    }
}

impl Default for CognitiveSystems {
    fn default() -> Self {
        Self::discover()
    }
}

// ============================================================================
// CortexMesh - The Full AGI Mesh
// ============================================================================

/// The CortexMesh - ALL cognitive components meshed with parallel processing
pub struct CortexMesh {
    /// All domain brains (auto-discovered)
    pub brains: BrainRegistry,

    /// Cognitive router with all modules
    pub router: CognitiveRouter,

    /// All cognitive systems (auto-discovered)
    pub cognitive_systems: CognitiveSystems,

    /// Core neural network model
    pub model: GraphTransformNet,

    /// Processing statistics
    pub stats: MeshStats,

    /// Configuration
    pub config: MeshConfig,

    /// Loss configuration
    pub loss_config: StructuralLossConfig,
}

/// Mesh configuration
#[derive(Debug, Clone)]
pub struct MeshConfig {
    /// Minimum confidence to activate a brain (default: 0.2 = activate most)
    pub activation_threshold: f32,
    /// Maximum brains to activate per query (default: all)
    pub max_active_brains: usize,
    /// Enable parallel processing (default: true)
    pub parallel: bool,
    /// Hidden dimension for model
    pub hidden_dim: usize,
    /// Number of layers for model
    pub num_layers: usize,
    /// Vocabulary size (256 for ASCII)
    pub vocab_size: usize,
    /// Embedding dimension
    pub embed_dim: usize,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            activation_threshold: 0.2,
            max_active_brains: usize::MAX, // All brains
            parallel: true,
            hidden_dim: 256,
            num_layers: 6,
            vocab_size: 256,
            embed_dim: 64,
        }
    }
}

/// Processing statistics
#[derive(Debug, Default, Clone)]
pub struct MeshStats {
    pub total_processed: usize,
    pub brain_activations: HashMap<String, usize>,
    pub module_activations: HashMap<String, usize>,
    pub avg_active_brains: f32,
    pub avg_processing_time_ms: f32,
    pub total_time_ms: f32,
}

/// Result from mesh processing
#[derive(Debug, Clone)]
pub struct MeshResult {
    /// Which brains were activated
    pub active_brains: Vec<String>,
    /// Confidence per brain
    pub brain_confidences: HashMap<String, f32>,
    /// Which router modules were activated
    pub active_modules: Vec<String>,
    /// Output graph from model
    pub output_graph: GraphemeGraph,
    /// Decoded text output
    pub decoded: String,
    /// Processing time in ms
    pub time_ms: f32,
    /// Pooling result for backward pass
    pub pooling: grapheme_core::PoolingResult,
}

impl CortexMesh {
    /// Discover and create mesh with ALL components (default config)
    pub fn discover() -> Self {
        Self::discover_with_config(MeshConfig::default())
    }

    /// Discover with custom configuration
    pub fn discover_with_config(config: MeshConfig) -> Self {
        // AUTO-INITIALIZE parallel processing if not already done
        let num_threads = init_parallel(0);

        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║          GRAPHEME CortexMesh Auto-Discovery                  ║");
        println!("║      ALL Components Meshed with Parallel Processing          ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  PARALLEL CPU: {} threads                                    ║", num_threads);
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Discover brains
        println!("Discovering domain brains...");
        let mut brains = BrainRegistry::new();
        for (id, factory) in BRAIN_FACTORIES {
            let brain = factory();
            println!("  [+] {} - {}", id, brain.domain_name());
            brains.register(brain);
        }
        println!("  Total: {} brains\n", brains.len());

        // Discover router modules
        println!("Discovering router modules...");
        let mut router = CognitiveRouter::new(config.activation_threshold);
        for (id, factory) in MODULE_FACTORIES {
            let module = factory();
            println!("  [+] {}", id);
            router.register_module(module);
        }
        println!("  Total: {} modules\n", MODULE_FACTORIES.len());

        // Discover cognitive systems
        println!("Discovering cognitive systems...");
        let cognitive_systems = CognitiveSystems::discover();
        for name in COGNITIVE_SYSTEM_NAMES {
            println!("  [+] {}", name);
        }
        println!("  Total: {} cognitive systems\n", CognitiveSystems::count());

        // Create model with proper 4-argument constructor
        println!("Initializing neural network...");
        let mut model = GraphTransformNet::new(
            config.vocab_size,
            config.embed_dim,
            config.hidden_dim,
            config.num_layers,
        );

        // Add Sabag pooling for code generation (expand to 512 output clusters)
        // This is CRITICAL for learning - without it, the model can't produce enough output nodes
        let output_clusters = 512;
        model.sabag_pooling = Some(grapheme_core::SabagPooling::new(output_clusters, config.embed_dim));
        println!("  Vocab: {}, Embed: {}, Hidden: {}, Layers: {}",
            config.vocab_size, config.embed_dim, config.hidden_dim, config.num_layers);
        println!("  Sabag pooling: {} output clusters\n", output_clusters);

        println!("CortexMesh ready!");
        println!("  Brains: {}", brains.len());
        println!("  Modules: {}", MODULE_FACTORIES.len());
        println!("  Cognitive Systems: {}", CognitiveSystems::count());
        println!("  Parallel: {}\n", config.parallel);

        Self {
            brains,
            router,
            cognitive_systems,
            model,
            stats: MeshStats::default(),
            config,
            loss_config: StructuralLossConfig::default(),
        }
    }

    /// Get number of registered brains
    pub fn brain_count(&self) -> usize {
        self.brains.len()
    }

    /// Get number of registered modules
    pub fn module_count(&self) -> usize {
        MODULE_FACTORIES.len()
    }

    /// List all brain IDs
    pub fn brain_ids(&self) -> Vec<String> {
        self.brains.domains().to_vec()
    }

    /// Get number of cognitive systems
    pub fn cognitive_system_count(&self) -> usize {
        CognitiveSystems::count()
    }

    /// List all cognitive system names
    pub fn cognitive_system_names(&self) -> Vec<&'static str> {
        CognitiveSystems::list()
    }

    /// Process input through ALL brains in PARALLEL (default)
    pub fn process_parallel(&mut self, input: &str) -> MeshResult {
        let start = Instant::now();

        // ========== PHASE 1: Parallel Brain Detection ==========
        let brain_ids: Vec<String> = self.brains.domains().to_vec();

        let activations: Vec<(String, bool, f32)> = if self.config.parallel {
            brain_ids
                .par_iter()
                .map(|id| {
                    if let Some(brain) = self.brains.get(id) {
                        let can = brain.can_process(input);
                        let conf = if can { 0.8 } else { 0.1 };
                        (id.clone(), can, conf)
                    } else {
                        (id.clone(), false, 0.0)
                    }
                })
                .collect()
        } else {
            brain_ids
                .iter()
                .map(|id| {
                    if let Some(brain) = self.brains.get(id) {
                        let can = brain.can_process(input);
                        let conf = if can { 0.8 } else { 0.1 };
                        (id.clone(), can, conf)
                    } else {
                        (id.clone(), false, 0.0)
                    }
                })
                .collect()
        };

        let mut active_brains = Vec::new();
        let mut brain_confidences = HashMap::new();

        for (id, can, conf) in activations {
            if can && conf >= self.config.activation_threshold {
                active_brains.push(id.clone());
                *self.stats.brain_activations.entry(id.clone()).or_insert(0) += 1;
            }
            brain_confidences.insert(id, conf);
        }

        // Limit active brains if configured
        if active_brains.len() > self.config.max_active_brains {
            active_brains.truncate(self.config.max_active_brains);
        }

        // ========== PHASE 2: Forward Through Model ==========
        let input_graph = GraphemeGraph::from_text(input);
        let (output_graph, pooling) = self.model.forward(&input_graph);
        let decoded = self.model.decode(&pooling);

        let time_ms = start.elapsed().as_secs_f32() * 1000.0;

        // ========== Update Stats ==========
        self.stats.total_processed += 1;
        self.stats.total_time_ms += time_ms;
        self.stats.avg_processing_time_ms =
            self.stats.total_time_ms / self.stats.total_processed as f32;

        let total_activations: usize = self.stats.brain_activations.values().sum();
        self.stats.avg_active_brains =
            total_activations as f32 / self.stats.total_processed as f32;

        MeshResult {
            active_brains,
            brain_confidences,
            active_modules: vec![], // TODO: track module activations
            output_graph,
            decoded,
            time_ms,
            pooling,
        }
    }

    /// Train step with structural loss only (legacy)
    pub fn train_step(&mut self, input: &str, target: &str, lr: f32) -> f32 {
        // Use unified training with structural loss weight = 1.0, char loss weight = 0.0
        let (total_loss, _, _) = self.train_step_unified(input, target, lr, 1.0, 0.0);
        total_loss
    }

    /// Unified training step: structural loss + character-level cross-entropy
    ///
    /// This trains BOTH:
    /// 1. Graph structure (node/edge topology matching target)
    /// 2. Character prediction (cross-entropy on decoded text)
    ///
    /// NOTE: This method calls step() internally, making it suitable for single-sample
    /// training. For batch training, use backward_unified() + step() separately.
    ///
    /// # Arguments
    /// * `input` - Input text (e.g., docstring/prompt)
    /// * `target` - Target text (e.g., code solution)
    /// * `lr` - Learning rate
    /// * `struct_weight` - Weight for structural loss (α)
    /// * `char_weight` - Weight for character cross-entropy loss (β)
    ///
    /// # Returns
    /// (total_loss, structural_loss, char_loss)
    pub fn train_step_unified(
        &mut self,
        input: &str,
        target: &str,
        lr: f32,
        struct_weight: f32,
        char_weight: f32,
    ) -> (f32, f32, f32) {
        self.model.zero_grad();
        let result = self.backward_unified(input, target, char_weight);
        self.step(lr, struct_weight, char_weight);
        result
    }

    /// Zero all gradients in the model (call at start of batch)
    pub fn zero_grad(&mut self) {
        self.model.zero_grad();
    }

    /// Forward + backward pass, accumulating gradients WITHOUT applying them
    ///
    /// Use this for proper batch training:
    /// ```ignore
    /// mesh.zero_grad();
    /// for sample in batch {
    ///     mesh.backward_unified(&sample.input, &sample.output, char_weight);
    /// }
    /// mesh.step(lr / batch_size, struct_weight, char_weight);
    /// ```
    pub fn backward_unified(
        &mut self,
        input: &str,
        target: &str,
        char_weight: f32,
    ) -> (f32, f32, f32) {
        // Forward pass through model
        let input_graph = GraphemeGraph::from_text(input);
        let target_graph = GraphemeGraph::from_text(target);
        let (output_graph, pooling) = self.model.forward(&input_graph);

        // 1. Structural loss (graph topology)
        let loss_result = compute_structural_loss(&output_graph, &target_graph, &self.loss_config);
        let structural_loss = loss_result.total_loss;

        // 2. Character-level cross-entropy loss
        let char_loss = if char_weight > 0.0 {
            self.compute_char_loss(&pooling, target)
        } else {
            0.0
        };

        // Combined loss (for reporting only - weights applied in step())
        let total_loss = structural_loss + char_loss;

        // Backward pass - accumulate gradients
        self.model.backward(&input_graph, &pooling, &loss_result.activation_gradients, self.config.embed_dim);

        // Also accumulate decoder gradients (but don't apply)
        if char_weight > 0.0 {
            self.accumulate_decoder_gradients(&pooling, target);
        }

        // Update stats
        self.stats.total_processed += 1;

        (total_loss, structural_loss, char_loss)
    }

    /// Apply accumulated gradients (call once per batch)
    pub fn step(&mut self, lr: f32, struct_weight: f32, char_weight: f32) {
        // Apply model gradients with structural weight
        self.model.step(lr * struct_weight);

        // Apply decoder gradients with char weight
        if char_weight > 0.0 {
            self.apply_decoder_gradients(lr * char_weight);
        }
    }

    /// Accumulate decoder gradients without applying them
    fn accumulate_decoder_gradients(&mut self, pooling: &grapheme_core::PoolingResult, target: &str) {
        let features = &pooling.features;
        let target_chars: Vec<u8> = target.bytes().collect();
        let vocab_size = self.config.vocab_size;

        if let Some(ref mut sabag) = self.model.sabag_pooling {
            let (num_clusters, embed_dim) = features.dim();
            let seq_len = num_clusters.min(target_chars.len());
            let query_ncols = sabag.query_matrix.ncols();

            let (grad_nrows, grad_ncols) = if let Some(ref qg) = sabag.query_grad {
                (qg.nrows(), qg.ncols())
            } else {
                (0, 0)
            };

            for i in 0..seq_len {
                let target_idx = target_chars[i] as usize;
                if target_idx >= vocab_size {
                    continue;
                }

                let feat = features.row(i);

                // Accumulate gradients (no learning rate - that's applied in step())
                for j in 0..embed_dim.min(query_ncols) {
                    let char_signal = if j == target_idx % embed_dim { 1.0 } else { -0.01 };
                    let grad = feat[j % feat.len()] * char_signal / (seq_len as f32);

                    if let Some(ref mut query_grad) = sabag.query_grad {
                        let row_idx = i % grad_nrows.max(1);
                        if row_idx < grad_nrows && j < grad_ncols {
                            query_grad[[row_idx, j]] += grad;
                        }
                    }
                }
            }
        }
    }

    /// Apply accumulated decoder gradients
    fn apply_decoder_gradients(&mut self, lr: f32) {
        if let Some(ref mut sabag) = self.model.sabag_pooling {
            // Apply gradients: w = w - lr * grad
            if let Some(ref grad) = sabag.query_grad {
                let grad_clone = grad.clone();
                ndarray::Zip::from(&mut sabag.query_matrix).and(&grad_clone).for_each(|w, &g| {
                    *w -= lr * g;
                });
            }
            // Zero decoder gradients after applying
            if let Some(ref mut qg) = sabag.query_grad {
                qg.fill(0.0);
            }
        }
    }

    /// Compute character-level cross-entropy loss
    fn compute_char_loss(&self, pooling: &grapheme_core::PoolingResult, target: &str) -> f32 {
        let decoded = self.model.decode(pooling);
        let target_chars: Vec<u8> = target.bytes().collect();
        let decoded_chars: Vec<u8> = decoded.bytes().collect();

        let vocab_size = self.config.vocab_size;
        let seq_len = target_chars.len().min(decoded_chars.len()).max(1);

        let mut total_loss = 0.0f32;

        // Cross-entropy: -log(p[target_char])
        // Using softmax approximation from activation values
        for i in 0..seq_len {
            let target_idx = target_chars.get(i).copied().unwrap_or(0) as usize;
            let pred_idx = decoded_chars.get(i).copied().unwrap_or(0) as usize;

            // Simple cross-entropy approximation
            // If prediction matches, low loss; otherwise high loss
            if target_idx == pred_idx {
                total_loss += 0.1; // Small loss for correct
            } else {
                // Distance-based loss
                let dist = (target_idx as i32 - pred_idx as i32).abs() as f32;
                total_loss += (1.0 + dist / vocab_size as f32).ln();
            }
        }

        // Add length penalty if decoded is shorter
        if decoded_chars.len() < target_chars.len() {
            let missing = target_chars.len() - decoded_chars.len();
            total_loss += missing as f32 * 2.0; // Penalty per missing char
        }

        total_loss / seq_len as f32
    }

    /// Get statistics summary
    pub fn stats_summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("╔═══ CortexMesh Statistics ═══╗\n"));
        s.push_str(&format!("║ Processed: {} samples\n", self.stats.total_processed));
        s.push_str(&format!("║ Avg active brains: {:.1}\n", self.stats.avg_active_brains));
        s.push_str(&format!("║ Avg time: {:.2}ms\n", self.stats.avg_processing_time_ms));
        s.push_str(&format!("╠═══ Brain Activations ═══════╣\n"));

        let mut sorted: Vec<_> = self.stats.brain_activations.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        for (brain, count) in sorted {
            let pct = *count as f32 / self.stats.total_processed.max(1) as f32 * 100.0;
            s.push_str(&format!("║  {}: {} ({:.1}%)\n", brain, count, pct));
        }
        s.push_str(&format!("╚═════════════════════════════╝\n"));
        s
    }

    /// Save model checkpoint
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        self.model.save_to_file(path)?;
        Ok(())
    }

    /// Load model from checkpoint
    pub fn load(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        self.model = GraphTransformNet::load_from_file(path)?;
        Ok(())
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// List all discoverable brain IDs (compile-time)
pub fn list_all_brains() -> Vec<&'static str> {
    BRAIN_FACTORIES.iter().map(|(id, _)| *id).collect()
}

/// List all discoverable module IDs (compile-time)
pub fn list_all_modules() -> Vec<&'static str> {
    MODULE_FACTORIES.iter().map(|(id, _)| *id).collect()
}

/// Count of all discoverable brains
pub fn brain_count() -> usize {
    BRAIN_FACTORIES.len()
}

/// Count of all discoverable modules
pub fn module_count() -> usize {
    MODULE_FACTORIES.len()
}

/// List all discoverable cognitive systems (compile-time)
pub fn list_all_cognitive_systems() -> Vec<&'static str> {
    CognitiveSystems::list()
}

/// Count of all discoverable cognitive systems
pub fn cognitive_system_count() -> usize {
    CognitiveSystems::count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_brains() {
        let brains = list_all_brains();
        assert!(brains.len() >= 8);
        assert!(brains.contains(&"math"));
        assert!(brains.contains(&"code"));
        assert!(brains.contains(&"vision"));
    }

    #[test]
    fn test_list_modules() {
        let modules = list_all_modules();
        assert!(modules.len() >= 4);
        assert!(modules.contains(&"text"));
        assert!(modules.contains(&"math"));
    }

    #[test]
    fn test_list_cognitive_systems() {
        let systems = list_all_cognitive_systems();
        assert_eq!(systems.len(), 8);
        assert!(systems.contains(&"reasoning"));
        assert!(systems.contains(&"metacognition"));
        assert!(systems.contains(&"memory"));
        assert!(systems.contains(&"safety"));
        assert!(systems.contains(&"world"));
        assert!(systems.contains(&"grounding"));
        assert!(systems.contains(&"agency"));
        assert!(systems.contains(&"multimodal"));
    }

    #[test]
    fn test_mesh_discovery() {
        let mesh = CortexMesh::discover();
        assert_eq!(mesh.brain_count(), BRAIN_FACTORIES.len());
        assert_eq!(mesh.module_count(), MODULE_FACTORIES.len());
        assert_eq!(mesh.cognitive_system_count(), 8);
    }

    #[test]
    fn test_parallel_processing() {
        let mut mesh = CortexMesh::discover();
        let result = mesh.process_parallel("def add(a, b): return a + b");

        // Code brain should activate
        assert!(result.active_brains.contains(&"code".to_string()));
        assert!(result.time_ms > 0.0);
    }
}
