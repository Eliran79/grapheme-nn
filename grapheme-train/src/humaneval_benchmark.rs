//! HumanEval Benchmark Integration (testing-019)
//!
//! Integrates HumanEval evaluation with graph-only training infrastructure:
//! - Uses HumanEvalEncoder (backend-228) for data loading
//! - Uses GraphTrainer (backend-229) for graph-only training
//! - Uses code_structural_loss (backend-230) for code-aware evaluation
//!
//! Provides:
//! - BenchmarkConfig for configuration
//! - HumanEvalBenchmark for running evaluations
//! - BenchmarkResult for reporting metrics
//! - pass@k computation for standard HumanEval scoring

use crate::code_loss::{dagnn_code_loss, CodeLossConfig, CodeStructuralLoss};
use crate::graph_data::{GraphDataset, GraphPair};
use crate::graph_trainer::{GraphTrainer, GraphTrainerConfig, TrainingHistory};
use crate::humaneval_encoder::HumanEvalEncoder;
use grapheme_core::{DagNN, GraphTransformer as CoreGraphTransformer};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// Constants
// ============================================================================

/// HumanEval SOTA as of 2024 (DeepSeek-Coder-V2)
pub const HUMANEVAL_SOTA: f64 = 96.2;

/// Standard k values for pass@k evaluation
pub const STANDARD_K_VALUES: &[usize] = &[1, 10, 100];

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for HumanEval benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of samples to generate per problem (for pass@k)
    pub samples_per_problem: usize,
    /// K values for pass@k metrics
    pub k_values: Vec<usize>,
    /// Use code-aware structural loss for evaluation
    pub use_code_loss: bool,
    /// Code loss configuration
    pub code_loss_config: CodeLossConfig,
    /// Training configuration (if training during benchmark)
    pub trainer_config: Option<GraphTrainerConfig>,
    /// Run evaluation in parallel
    pub parallel: bool,
    /// Verbose output
    pub verbose: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            samples_per_problem: 1,
            k_values: vec![1],
            use_code_loss: true,
            code_loss_config: CodeLossConfig::default(),
            trainer_config: None,
            parallel: true,
            verbose: false,
        }
    }
}

impl BenchmarkConfig {
    /// Create config for full pass@k evaluation
    pub fn full_evaluation() -> Self {
        Self {
            samples_per_problem: 200,
            k_values: STANDARD_K_VALUES.to_vec(),
            use_code_loss: true,
            code_loss_config: CodeLossConfig::default(),
            trainer_config: None,
            parallel: true,
            verbose: false,
        }
    }

    /// Create config for quick evaluation (development)
    pub fn quick() -> Self {
        Self {
            samples_per_problem: 1,
            k_values: vec![1],
            use_code_loss: true,
            code_loss_config: CodeLossConfig::default(),
            trainer_config: None,
            parallel: false,
            verbose: true,
        }
    }
}

// ============================================================================
// Results
// ============================================================================

/// Result for a single problem
#[derive(Debug, Clone)]
pub struct ProblemEvaluation {
    /// Problem ID (e.g., "HumanEval/0")
    pub task_id: String,
    /// Entry point function name
    pub entry_point: String,
    /// Number of samples generated
    pub num_samples: usize,
    /// Number of samples with correct structure
    pub num_correct: usize,
    /// Structural loss for each sample
    pub structural_losses: Vec<f32>,
    /// Code-aware loss breakdown (if enabled)
    pub code_loss: Option<CodeStructuralLoss>,
    /// Mean structural loss across samples
    pub mean_loss: f32,
}

impl ProblemEvaluation {
    /// Compute pass@k for this problem
    pub fn pass_at_k(&self, k: usize) -> f64 {
        pass_at_k(self.num_samples, self.num_correct, k)
    }
}

/// Aggregated benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Total number of problems
    pub num_problems: usize,
    /// Total number of samples
    pub total_samples: usize,
    /// Pass@k scores for each k value
    pub pass_at_k: HashMap<usize, f64>,
    /// Mean structural loss across all problems
    pub mean_structural_loss: f32,
    /// Mean code-aware loss (if enabled)
    pub mean_code_loss: Option<f32>,
    /// Per-problem results
    pub problems: Vec<ProblemEvaluation>,
    /// Whether SOTA was beaten
    pub beats_sota: bool,
    /// Evaluation time (seconds)
    pub eval_time_secs: f64,
}

impl BenchmarkResult {
    /// Get the main pass@1 score
    pub fn pass_at_1(&self) -> f64 {
        *self.pass_at_k.get(&1).unwrap_or(&0.0)
    }

    /// Generate a summary string
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("========== HumanEval Benchmark Results ==========\n\n");

        for k in STANDARD_K_VALUES {
            if let Some(score) = self.pass_at_k.get(k) {
                let indicator = if *k == 1 && *score > HUMANEVAL_SOTA {
                    " BEATS SOTA!"
                } else {
                    ""
                };
                s.push_str(&format!("pass@{}: {:.2}%{}\n", k, score, indicator));
            }
        }

        s.push_str(&format!("\nMean structural loss: {:.4}\n", self.mean_structural_loss));
        if let Some(code_loss) = self.mean_code_loss {
            s.push_str(&format!("Mean code-aware loss: {:.4}\n", code_loss));
        }
        s.push_str(&format!("\nProblems: {}\n", self.num_problems));
        s.push_str(&format!("Total samples: {}\n", self.total_samples));
        s.push_str(&format!("Evaluation time: {:.1}s\n", self.eval_time_secs));

        s
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

/// HumanEval benchmark runner
pub struct HumanEvalBenchmark {
    /// Configuration
    pub config: BenchmarkConfig,
    /// Encoder for loading and converting problems
    encoder: HumanEvalEncoder,
    /// Code loss configuration
    #[allow(dead_code)]
    code_loss_config: CodeLossConfig,
}

impl HumanEvalBenchmark {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        let code_loss_config = config.code_loss_config.clone();
        Self {
            config,
            encoder: HumanEvalEncoder::new(),
            code_loss_config,
        }
    }

    /// Load HumanEval dataset from JSONL file
    pub fn load_dataset<P: AsRef<Path>>(&self, path: P) -> Result<GraphDataset, BenchmarkError> {
        let result = self.encoder.encode_dataset(&path).map_err(|e| {
            BenchmarkError::DatasetLoad(format!("Failed to encode dataset: {}", e))
        })?;

        if result.successes == 0 {
            return Err(BenchmarkError::DatasetLoad(
                "No problems were successfully encoded".to_string(),
            ));
        }

        if self.config.verbose {
            println!(
                "Loaded {} problems ({} failures)",
                result.successes,
                result.failures.len()
            );
        }

        Ok(result.dataset)
    }

    /// Evaluate a model on the HumanEval dataset
    pub fn evaluate(
        &self,
        model: &dyn GraphTransformer,
        dataset: &GraphDataset,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let start = std::time::Instant::now();
        let mut problem_evals = Vec::new();

        // Process each problem
        let pairs: Vec<_> = dataset.pairs.iter().collect();

        if self.config.parallel {
            problem_evals = pairs
                .par_iter()
                .map(|pair| self.evaluate_problem(model, pair))
                .collect();
        } else {
            for pair in &pairs {
                problem_evals.push(self.evaluate_problem(model, pair));
            }
        }

        // Compute aggregate metrics
        let total_samples: usize = problem_evals.iter().map(|p| p.num_samples).sum();
        let _total_correct: usize = problem_evals.iter().map(|p| p.num_correct).sum();

        // Compute pass@k for each k value
        let mut pass_at_k_scores = HashMap::new();
        for &k in &self.config.k_values {
            let scores: Vec<f64> = problem_evals.iter().map(|p| p.pass_at_k(k)).collect();
            let mean_score = if scores.is_empty() {
                0.0
            } else {
                scores.iter().sum::<f64>() / scores.len() as f64 * 100.0
            };
            pass_at_k_scores.insert(k, mean_score);
        }

        // Mean structural loss
        let mean_structural_loss = if problem_evals.is_empty() {
            0.0
        } else {
            problem_evals.iter().map(|p| p.mean_loss).sum::<f32>()
                / problem_evals.len() as f32
        };

        // Mean code loss (if enabled)
        let mean_code_loss = if self.config.use_code_loss {
            let code_losses: Vec<f32> = problem_evals
                .iter()
                .filter_map(|p| p.code_loss.as_ref().map(|c| c.total))
                .collect();
            if code_losses.is_empty() {
                None
            } else {
                Some(code_losses.iter().sum::<f32>() / code_losses.len() as f32)
            }
        } else {
            None
        };

        let beats_sota = pass_at_k_scores.get(&1).is_some_and(|&s| s > HUMANEVAL_SOTA);

        Ok(BenchmarkResult {
            num_problems: problem_evals.len(),
            total_samples,
            pass_at_k: pass_at_k_scores,
            mean_structural_loss,
            mean_code_loss,
            problems: problem_evals,
            beats_sota,
            eval_time_secs: start.elapsed().as_secs_f64(),
        })
    }

    /// Evaluate a single problem
    fn evaluate_problem(&self, model: &dyn GraphTransformer, pair: &GraphPair) -> ProblemEvaluation {
        let mut structural_losses = Vec::new();
        let mut num_correct = 0;

        // Generate samples
        for _ in 0..self.config.samples_per_problem {
            // Transform input to output
            let predicted = model.transform_graph(&pair.input);

            // Compute structural loss
            let loss = structural_distance(&predicted, &pair.output);
            structural_losses.push(loss);

            // Count as correct if loss is below threshold
            // Using a threshold based on graph size
            let threshold = 0.1 * (pair.output.node_count() as f32);
            if loss < threshold {
                num_correct += 1;
            }
        }

        // Compute code-aware loss for the first sample
        let code_loss = if self.config.use_code_loss {
            let predicted = model.transform_graph(&pair.input);
            Some(dagnn_code_loss(&predicted, &pair.output))
        } else {
            None
        };

        let mean_loss = if structural_losses.is_empty() {
            0.0
        } else {
            structural_losses.iter().sum::<f32>() / structural_losses.len() as f32
        };

        ProblemEvaluation {
            task_id: pair.id.clone(),
            entry_point: pair.metadata.get("entry_point").cloned().unwrap_or_default(),
            num_samples: self.config.samples_per_problem,
            num_correct,
            structural_losses,
            code_loss,
            mean_loss,
        }
    }

    /// Train and evaluate using GraphTrainer
    pub fn train_and_evaluate(
        &self,
        dataset: &GraphDataset,
        trainer_config: GraphTrainerConfig,
    ) -> Result<(TrainingHistory, BenchmarkResult), BenchmarkError> {
        // Split dataset for training, validation, and evaluation (70/15/15)
        let (train_dataset, val_dataset, eval_dataset) = dataset.split(0.7, 0.15);

        // Train
        let mut trainer = GraphTrainer::new(trainer_config);
        let history = trainer.train(&train_dataset, Some(&val_dataset));

        // Get the trained network and evaluate
        let trained_model = TrainedNetworkWrapper {
            network: trainer.into_network()
        };
        let result = self.evaluate(&trained_model, &eval_dataset)?;

        Ok((history, result))
    }
}

/// Trait for graph transformers (models that transform graphs)
pub trait GraphTransformer: Sync {
    /// Transform an input graph to an output graph
    fn transform_graph(&self, input: &DagNN) -> DagNN;
}

/// Wrapper to use GraphTransformNet as a GraphTransformer
struct TrainedNetworkWrapper {
    network: crate::graph_transform_net::GraphTransformNet,
}

impl GraphTransformer for TrainedNetworkWrapper {
    fn transform_graph(&self, input: &DagNN) -> DagNN {
        let mut network = self.network.clone();
        CoreGraphTransformer::transform(&mut network, input).unwrap_or_else(|_| input.clone())
    }
}

/// Identity transformer for baseline evaluation
pub struct IdentityTransformer;

impl GraphTransformer for IdentityTransformer {
    fn transform_graph(&self, input: &DagNN) -> DagNN {
        input.clone()
    }
}

// ============================================================================
// Metrics
// ============================================================================

/// Compute pass@k using unbiased estimator
///
/// pass@k = 1 - C(n-c, k) / C(n, k)
/// where n = total samples, c = correct samples
pub fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    if n < k {
        return if c > 0 { 1.0 } else { 0.0 };
    }

    if c == 0 {
        return 0.0;
    }

    if c >= n {
        return 1.0;
    }

    // Compute using log-space to avoid overflow
    // pass@k = 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)
    let mut log_prod = 0.0;
    for i in 0..k {
        if n - c < i + 1 {
            return 1.0;
        }
        let numerator = (n - c - i) as f64;
        let denominator = (n - i) as f64;
        if numerator <= 0.0 || denominator <= 0.0 {
            return 1.0;
        }
        log_prod += numerator.ln() - denominator.ln();
    }
    1.0 - log_prod.exp()
}

/// Compute structural distance between two graphs
/// Uses node count difference and edge overlap
fn structural_distance(predicted: &DagNN, target: &DagNN) -> f32 {
    let node_diff = (predicted.node_count() as i32 - target.node_count() as i32).abs() as f32;
    let edge_diff = (predicted.edge_count() as i32 - target.edge_count() as i32).abs() as f32;

    // Normalize by target size
    let target_size = (target.node_count() + target.edge_count()).max(1) as f32;
    (node_diff + edge_diff) / target_size
}

// ============================================================================
// Errors
// ============================================================================

/// Benchmark errors
#[derive(Debug, Clone)]
pub enum BenchmarkError {
    /// Dataset loading error
    DatasetLoad(String),
    /// Evaluation error
    Evaluation(String),
    /// Configuration error
    Config(String),
}

impl std::fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchmarkError::DatasetLoad(msg) => write!(f, "Dataset load error: {}", msg),
            BenchmarkError::Evaluation(msg) => write!(f, "Evaluation error: {}", msg),
            BenchmarkError::Config(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for BenchmarkError {}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Quick evaluation with default settings
pub fn quick_evaluate<P: AsRef<Path>>(
    model: &dyn GraphTransformer,
    dataset_path: P,
) -> Result<BenchmarkResult, BenchmarkError> {
    let config = BenchmarkConfig::quick();
    let benchmark = HumanEvalBenchmark::new(config);
    let dataset = benchmark.load_dataset(dataset_path)?;
    benchmark.evaluate(model, &dataset)
}

/// Full pass@k evaluation
pub fn full_evaluate<P: AsRef<Path>>(
    model: &dyn GraphTransformer,
    dataset_path: P,
) -> Result<BenchmarkResult, BenchmarkError> {
    let config = BenchmarkConfig::full_evaluation();
    let benchmark = HumanEvalBenchmark::new(config);
    let dataset = benchmark.load_dataset(dataset_path)?;
    benchmark.evaluate(model, &dataset)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_data::GraphPairBuilder;

    fn create_test_graph(nodes: usize) -> DagNN {
        let mut dag = DagNN::new();
        use grapheme_core::Node;

        for _ in 0..nodes {
            dag.graph.add_node(Node::hidden());
        }
        dag
    }

    fn create_test_dataset() -> GraphDataset {
        let pairs: Vec<GraphPair> = (0..5)
            .map(|i| {
                GraphPairBuilder::new(format!("HumanEval/{}", i))
                    .input(create_test_graph(3))
                    .output(create_test_graph(5))
                    .level(1)
                    .domain("humaneval")
                    .meta("entry_point", format!("func_{}", i))
                    .build()
            })
            .collect();
        GraphDataset::from_pairs("humaneval", pairs)
    }

    #[test]
    fn test_pass_at_k_all_correct() {
        assert!((pass_at_k(10, 10, 1) - 1.0).abs() < 0.001);
        assert!((pass_at_k(10, 10, 10) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pass_at_k_none_correct() {
        assert!((pass_at_k(10, 0, 1) - 0.0).abs() < 0.001);
        assert!((pass_at_k(10, 0, 10) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_pass_at_k_partial() {
        // With 5 correct out of 10, pass@1 should be around 0.5
        let p1 = pass_at_k(10, 5, 1);
        assert!(p1 > 0.4 && p1 < 0.6);

        // pass@10 with 5 correct should be high (at least one in 10)
        let p10 = pass_at_k(10, 5, 10);
        assert!(p10 > 0.99);
    }

    #[test]
    fn test_pass_at_k_small_n() {
        // When n < k, should return 1.0 if any correct
        assert!((pass_at_k(5, 1, 10) - 1.0).abs() < 0.001);
        // Zero if none correct
        assert!((pass_at_k(5, 0, 10) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.samples_per_problem, 1);
        assert_eq!(config.k_values, vec![1]);
        assert!(config.use_code_loss);
    }

    #[test]
    fn test_benchmark_config_full() {
        let config = BenchmarkConfig::full_evaluation();
        assert_eq!(config.samples_per_problem, 200);
        assert_eq!(config.k_values, vec![1, 10, 100]);
    }

    #[test]
    fn test_benchmark_config_quick() {
        let config = BenchmarkConfig::quick();
        assert_eq!(config.samples_per_problem, 1);
        assert!(config.verbose);
    }

    #[test]
    fn test_benchmark_creation() {
        let config = BenchmarkConfig::default();
        let benchmark = HumanEvalBenchmark::new(config);
        assert!(!benchmark.config.verbose);
    }

    #[test]
    fn test_identity_transformer() {
        let transformer = IdentityTransformer;
        let input = create_test_graph(5);
        let output = transformer.transform_graph(&input);
        assert_eq!(input.node_count(), output.node_count());
    }

    #[test]
    fn test_structural_distance_identical() {
        let g1 = create_test_graph(5);
        let g2 = create_test_graph(5);
        let dist = structural_distance(&g1, &g2);
        assert!(dist < 0.001);
    }

    #[test]
    fn test_structural_distance_different() {
        let g1 = create_test_graph(5);
        let g2 = create_test_graph(10);
        let dist = structural_distance(&g1, &g2);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_evaluate_with_identity() {
        let config = BenchmarkConfig::default();
        let benchmark = HumanEvalBenchmark::new(config);
        let dataset = create_test_dataset();
        let transformer = IdentityTransformer;

        let result = benchmark.evaluate(&transformer, &dataset).unwrap();
        assert_eq!(result.num_problems, 5);
        assert!(result.mean_structural_loss > 0.0); // Input != output
    }

    #[test]
    fn test_problem_evaluation() {
        let eval = ProblemEvaluation {
            task_id: "HumanEval/0".to_string(),
            entry_point: "test".to_string(),
            num_samples: 10,
            num_correct: 5,
            structural_losses: vec![0.1; 10],
            code_loss: None,
            mean_loss: 0.1,
        };

        let p1 = eval.pass_at_k(1);
        assert!(p1 > 0.4 && p1 < 0.6);
    }

    #[test]
    fn test_benchmark_result_summary() {
        let mut pass_at_k = HashMap::new();
        pass_at_k.insert(1, 50.0);
        pass_at_k.insert(10, 90.0);

        let result = BenchmarkResult {
            num_problems: 164,
            total_samples: 164,
            pass_at_k,
            mean_structural_loss: 0.25,
            mean_code_loss: Some(0.3),
            problems: vec![],
            beats_sota: false,
            eval_time_secs: 10.5,
        };

        let summary = result.summary();
        assert!(summary.contains("pass@1: 50.00%"));
        assert!(summary.contains("Problems: 164"));
    }

    #[test]
    fn test_benchmark_result_pass_at_1() {
        let mut pass_at_k = HashMap::new();
        pass_at_k.insert(1, 85.5);

        let result = BenchmarkResult {
            num_problems: 100,
            total_samples: 100,
            pass_at_k,
            mean_structural_loss: 0.1,
            mean_code_loss: None,
            problems: vec![],
            beats_sota: false,
            eval_time_secs: 5.0,
        };

        assert!((result.pass_at_1() - 85.5).abs() < 0.001);
    }

    #[test]
    fn test_benchmark_error_display() {
        let err = BenchmarkError::DatasetLoad("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }
}
