//! CortexMesh - Auto-Discovery and Parallel Processing for ALL GRAPHEME Components
//!
//! This module provides compile-time auto-discovery of all brains, modules, and
//! cognitive components. Just call `CortexMesh::discover()` to get everything
//! meshed and ready for parallel processing.
//!
//! ## PARALLEL BY DEFAULT
//! All operations use Rayon for automatic CPU parallelization.
//! Call `init_parallel()` at program start to configure thread pool.
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
//! println!("Discovered {} brains", mesh.brain_count());
//!
//! // Process with ALL brains in parallel (default)
//! let result = mesh.process_parallel("def fibonacci(n): ...");
//! println!("Active brains: {:?}", result.active_brains);
//! ```
//!
//! ## Adding New Brains
//! To add a new brain, just add it to the `BRAIN_FACTORIES` array below.
//! The mesh will automatically discover and use it.

use grapheme_chem::ChemBrain;
use grapheme_code::CodeBrain;
use grapheme_core::{BrainRegistry, DomainBrain, GraphemeGraph, GraphTransformNet};
use grapheme_law::LawBrain;
use grapheme_math::MathBrain;
use grapheme_music::MusicBrain;
use grapheme_router::{CognitiveModule, CognitiveRouter, MathModule, TextModule, TimeSeriesModule, VisionModule as RouterVisionModule};
use grapheme_time::TimeBrain;
use grapheme_vision::{ClassificationBrain, ClassificationConfig, VisionBrain};
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
// CortexMesh - The Full AGI Mesh
// ============================================================================

/// The CortexMesh - ALL cognitive components meshed with parallel processing
pub struct CortexMesh {
    /// All domain brains (auto-discovered)
    pub brains: BrainRegistry,

    /// Cognitive router with all modules
    pub router: CognitiveRouter,

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

        // Create model with proper 4-argument constructor
        println!("Initializing neural network...");
        let model = GraphTransformNet::new(
            config.vocab_size,
            config.embed_dim,
            config.hidden_dim,
            config.num_layers,
        );
        println!("  Vocab: {}, Embed: {}, Hidden: {}, Layers: {}\n",
            config.vocab_size, config.embed_dim, config.hidden_dim, config.num_layers);

        println!("CortexMesh ready!");
        println!("  Brains: {}", brains.len());
        println!("  Modules: {}", MODULE_FACTORIES.len());
        println!("  Parallel: {}\n", config.parallel);

        Self {
            brains,
            router,
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

    /// Train step with structural loss
    pub fn train_step(&mut self, input: &str, target: &str, lr: f32) -> f32 {
        // Forward pass
        let input_graph = GraphemeGraph::from_text(input);
        let target_graph = GraphemeGraph::from_text(target);
        let (output_graph, pooling) = self.model.forward(&input_graph);

        // Compute structural loss
        let loss_result = compute_structural_loss(&output_graph, &target_graph, &self.loss_config);

        // Backward pass
        self.model.backward(&input_graph, &pooling, &loss_result.activation_gradients, self.config.embed_dim);

        // Update weights
        self.model.step(lr);

        // Update stats
        self.stats.total_processed += 1;

        // Return loss
        loss_result.total_loss
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
    fn test_mesh_discovery() {
        let mesh = CortexMesh::discover();
        assert_eq!(mesh.brain_count(), BRAIN_FACTORIES.len());
        assert_eq!(mesh.module_count(), MODULE_FACTORIES.len());
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
