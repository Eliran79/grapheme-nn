//! # grapheme-train
//!
//! Training infrastructure for GRAPHEME neural network.
//!
//! This crate provides:
//! - Training data generation from the engine
//! - Graph edit distance loss computation
//! - Curriculum learning support
//! - Dataset management
//!
//! Key training concepts:
//! - Engine generates infinite verified training pairs
//! - Loss is graph edit distance, not cross-entropy
//! - Brain learns to approximate transformations
//! - All outputs validated against engine

use grapheme_core::GraphemeGraph;
use grapheme_engine::{Expr, MathEngine, MathOp, Value};
use grapheme_math::MathGraph;
use grapheme_polish::expr_to_polish;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Training errors
#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Data generation error: {0}")]
    DataGenerationError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Engine error: {0}")]
    EngineError(#[from] grapheme_engine::EngineError),
}

/// Result type for training operations
pub type TrainingResult<T> = Result<T, TrainingError>;

/// A training example: input expression -> expected result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// The input expression in Polish notation
    pub input_polish: String,
    /// The input expression
    pub input_expr: Expr,
    /// The expected result
    pub expected_result: f64,
    /// Difficulty level (1-7)
    pub level: u8,
}

/// Curriculum levels for training
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurriculumLevel {
    /// Level 1: Single binary operations (2 + 3)
    BasicArithmetic = 1,
    /// Level 2: Nested operations ((2 + 3) * 4)
    NestedOperations = 2,
    /// Level 3: Symbol substitution (x + 3 where x = 2)
    SymbolSubstitution = 3,
    /// Level 4: Basic functions (sqrt(16))
    BasicFunctions = 4,
    /// Level 5: Symbolic differentiation (future)
    Differentiation = 5,
    /// Level 6: Integration (future)
    Integration = 6,
    /// Level 7: Equation solving (future)
    EquationSolving = 7,
}

/// Training data generator
#[derive(Debug)]
pub struct DataGenerator {
    engine: MathEngine,
    rng_seed: u64,
}

impl DataGenerator {
    /// Create a new data generator
    pub fn new(seed: u64) -> Self {
        Self {
            engine: MathEngine::new(),
            rng_seed: seed,
        }
    }

    /// Generate examples for a specific curriculum level
    pub fn generate_level(&self, level: CurriculumLevel, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::with_capacity(count);

        match level {
            CurriculumLevel::BasicArithmetic => {
                // Simple a op b expressions
                let ops = [MathOp::Add, MathOp::Sub, MathOp::Mul];
                for i in 0..count {
                    let a = ((i * 7 + self.rng_seed as usize) % 20) as i64;
                    let b = ((i * 13 + self.rng_seed as usize) % 20 + 1) as i64;
                    let op = ops[i % ops.len()];

                    let expr = Expr::BinOp {
                        op,
                        left: Box::new(Expr::Value(Value::Integer(a))),
                        right: Box::new(Expr::Value(Value::Integer(b))),
                    };

                    if let Ok(result) = self.engine.evaluate(&expr) {
                        examples.push(TrainingExample {
                            input_polish: expr_to_polish(&expr),
                            input_expr: expr,
                            expected_result: result,
                            level: 1,
                        });
                    }
                }
            }
            CurriculumLevel::NestedOperations => {
                // Nested expressions: (a op b) op2 c
                let ops = [MathOp::Add, MathOp::Mul];
                for i in 0..count {
                    let a = ((i * 3 + self.rng_seed as usize) % 10 + 1) as i64;
                    let b = ((i * 7 + self.rng_seed as usize) % 10 + 1) as i64;
                    let c = ((i * 11 + self.rng_seed as usize) % 10 + 1) as i64;
                    let op1 = ops[i % ops.len()];
                    let op2 = ops[(i + 1) % ops.len()];

                    let expr = Expr::BinOp {
                        op: op2,
                        left: Box::new(Expr::BinOp {
                            op: op1,
                            left: Box::new(Expr::Value(Value::Integer(a))),
                            right: Box::new(Expr::Value(Value::Integer(b))),
                        }),
                        right: Box::new(Expr::Value(Value::Integer(c))),
                    };

                    if let Ok(result) = self.engine.evaluate(&expr) {
                        examples.push(TrainingExample {
                            input_polish: expr_to_polish(&expr),
                            input_expr: expr,
                            expected_result: result,
                            level: 2,
                        });
                    }
                }
            }
            CurriculumLevel::BasicFunctions => {
                // Function applications: sqrt(n), abs(n)
                use grapheme_engine::MathFn;
                let perfect_squares = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100];
                for i in 0..count.min(perfect_squares.len()) {
                    let n = perfect_squares[i];

                    let expr = Expr::Function {
                        func: MathFn::Sqrt,
                        args: vec![Expr::Value(Value::Integer(n))],
                    };

                    if let Ok(result) = self.engine.evaluate(&expr) {
                        examples.push(TrainingExample {
                            input_polish: expr_to_polish(&expr),
                            input_expr: expr,
                            expected_result: result,
                            level: 4,
                        });
                    }
                }
            }
            _ => {
                // Placeholder for future levels
            }
        }

        examples
    }

    /// Generate a full curriculum dataset
    pub fn generate_curriculum(&self, examples_per_level: usize) -> Vec<TrainingExample> {
        let mut all_examples = Vec::new();

        for level in [
            CurriculumLevel::BasicArithmetic,
            CurriculumLevel::NestedOperations,
            CurriculumLevel::BasicFunctions,
        ] {
            all_examples.extend(self.generate_level(level, examples_per_level));
        }

        all_examples
    }
}

/// Graph edit distance computation (the core loss function)
#[derive(Debug, Clone, Default)]
pub struct GraphEditDistance {
    /// Cost of inserting a node
    pub node_insertion_cost: f32,
    /// Cost of deleting an edge
    pub edge_deletion_cost: f32,
    /// Clique mismatch penalty
    pub clique_mismatch: f32,
}

impl GraphEditDistance {
    /// Compute the total loss
    pub fn total(&self) -> f32 {
        self.node_insertion_cost + self.edge_deletion_cost + self.clique_mismatch
    }

    /// Compute graph edit distance between two GRAPHEME graphs
    pub fn compute(predicted: &GraphemeGraph, target: &GraphemeGraph) -> Self {
        // Simplified implementation - real version would use optimal graph matching
        let node_diff = (predicted.node_count() as i32 - target.node_count() as i32).abs() as f32;
        let edge_diff = (predicted.edge_count() as i32 - target.edge_count() as i32).abs() as f32;

        Self {
            node_insertion_cost: node_diff * 0.5,
            edge_deletion_cost: edge_diff * 0.3,
            clique_mismatch: 0.0, // TODO: implement clique comparison
        }
    }

    /// Compute graph edit distance between two math graphs
    pub fn compute_math(predicted: &MathGraph, target: &MathGraph) -> Self {
        let node_diff = (predicted.node_count() as i32 - target.node_count() as i32).abs() as f32;
        let edge_diff = (predicted.edge_count() as i32 - target.edge_count() as i32).abs() as f32;

        Self {
            node_insertion_cost: node_diff * 0.5,
            edge_deletion_cost: edge_diff * 0.3,
            clique_mismatch: 0.0,
        }
    }
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Weight for node insertion cost in loss
    pub alpha: f32,
    /// Weight for edge deletion cost in loss
    pub beta: f32,
    /// Weight for clique mismatch in loss
    pub gamma: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            alpha: 1.0,
            beta: 1.0,
            gamma: 0.5,
        }
    }
}

/// Trainer for GRAPHEME models
#[derive(Debug)]
pub struct Trainer {
    config: TrainingConfig,
    engine: MathEngine,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            engine: MathEngine::new(),
        }
    }

    /// Validate a prediction against the engine
    pub fn validate(&self, expr: &Expr, predicted: f64) -> bool {
        match self.engine.evaluate(expr) {
            Ok(expected) => (predicted - expected).abs() < 1e-10,
            Err(_) => false,
        }
    }

    /// Get the training configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation_level1() {
        let generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::BasicArithmetic, 10);

        assert_eq!(examples.len(), 10);
        for example in &examples {
            assert_eq!(example.level, 1);
            // Verify the result is correct
            let engine = MathEngine::new();
            let computed = engine.evaluate(&example.input_expr).unwrap();
            assert!((computed - example.expected_result).abs() < 1e-10);
        }
    }

    #[test]
    fn test_data_generation_level2() {
        let generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::NestedOperations, 5);

        assert!(!examples.is_empty());
        for example in &examples {
            assert_eq!(example.level, 2);
            // Verify Polish notation is nested
            assert!(example.input_polish.matches('(').count() >= 2);
        }
    }

    #[test]
    fn test_curriculum_generation() {
        let generator = DataGenerator::new(42);
        let examples = generator.generate_curriculum(5);

        // Should have examples from multiple levels
        let levels: std::collections::HashSet<u8> =
            examples.iter().map(|e| e.level).collect();
        assert!(levels.len() >= 2);
    }

    #[test]
    fn test_graph_edit_distance() {
        use grapheme_core::GraphemeGraph;

        let g1 = GraphemeGraph::from_text("Hello");
        let g2 = GraphemeGraph::from_text("Hello!");

        let distance = GraphEditDistance::compute(&g1, &g2);
        assert!(distance.total() > 0.0);

        // Same graph should have zero distance
        let g3 = GraphemeGraph::from_text("Hello");
        let same_distance = GraphEditDistance::compute(&g1, &g3);
        assert_eq!(same_distance.total(), 0.0);
    }
}
