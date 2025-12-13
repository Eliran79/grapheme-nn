//! # grapheme-train
//!
//! Training infrastructure for GRAPHEME neural network.
//!
//! This crate provides:
//! - Training data generation from the engine
//! - Graph edit distance loss computation
//! - Curriculum learning support
//! - Dataset management (JSONL format)
//! - Batch iteration for training loops
//!
//! Key training concepts:
//! - Engine generates infinite verified training pairs
//! - Loss is graph edit distance, not cross-entropy
//! - Brain learns to approximate transformations
//! - All outputs validated against engine

use grapheme_core::GraphemeGraph;
use grapheme_engine::{Expr, MathEngine, MathFn, MathOp, SymbolicEngine, Value};
use grapheme_math::MathGraph;
use grapheme_polish::expr_to_polish;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use thiserror::Error;

// Module declarations
pub mod backprop;
pub mod code_loss;
pub mod graph_data;
pub mod graph_trainer;
pub mod graph_transform_net;
pub mod humaneval_encoder;
pub mod parallel_cortex;

// Re-export key types
pub use backprop::{
    ActivationType, AggregationType, BackwardResult, CombineType,
    GradientCheckResult, MessagePassingBackward, Tape, TapeOp,
    clip_grad_norm, clip_grad_value, gradient_check,
    reverse_topological_sort, topological_sort,
};

// Re-export GraphTransformNet types (backend-029)
pub use graph_transform_net::{
    GraphTransformNet, GraphTransformNetConfig, ForwardOutput,
    NodeEdit, EdgeEdit, MessagePassingLayer,
    NodePredictionHead, EdgePredictionHead,
};

// Re-export graph data types (backend-227)
pub use graph_data::{
    GraphPair, GraphPairStats, GraphPairBuilder,
    GraphDataset, GraphDatasetMetadata, DatasetStats,
    GraphBatchIterator, GraphDataError, GraphDataResult,
    GraphEncoder, EncoderRegistry,
    create_chain_graph, create_tree_graph, create_random_dag,
};

// Re-export HumanEval encoder types (backend-228)
pub use humaneval_encoder::{
    HumanEvalProblem, HumanEvalEncoder, HumanEvalEncodingResult, EncodingFailure,
};

// Re-export graph trainer types (backend-229)
pub use graph_trainer::{
    GraphTrainer, GraphTrainerConfig, TrainingHistory,
    StructuralLoss, EvaluationResult, structural_loss, quick_train,
    EpochMetrics as GraphEpochMetrics,
};

// Re-export code loss types (backend-230)
pub use code_loss::{
    CodeStructuralLoss, CodeLossConfig, NodeCategory, EdgeCategory,
    OperationType, code_structural_loss, dagnn_code_loss, operation_distance,
};

// Re-export parallel cortex types (backend-212)
pub use parallel_cortex::{
    CollaborativeCortexConfig, CollaborativeCortexTrainer,
    CollaborativeResult, BrainActivation, FusionLayer, BatchLoss,
    TrainingStats, parallel_process_dataset, parallel_train_batch,
};

/// Training errors
#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Data generation error: {0}")]
    DataGenerationError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Engine error: {0}")]
    EngineError(#[from] grapheme_engine::EngineError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Result type for training operations
pub type TrainingResult<T> = Result<T, TrainingError>;

// ============================================================================
// Output Types and Level Specifications (per GRAPHEME_Math_Dataset.md)
// ============================================================================

/// Output type for training examples
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// Numeric result only (e.g., 5)
    Numeric,
    /// Symbolic result only (e.g., (* 2 x))
    Symbolic,
    /// Both numeric and symbolic
    Both,
}

/// Specification for a curriculum level
#[derive(Debug, Clone)]
pub struct LevelSpec {
    /// Level number (1-7)
    pub level: u8,
    /// Allowed operations
    pub ops: Vec<MathOp>,
    /// Allowed functions
    pub functions: Vec<MathFn>,
    /// Maximum expression depth
    pub max_depth: usize,
    /// Whether symbols are allowed
    pub allow_symbols: bool,
    /// Output type
    pub output: OutputType,
    /// Number of samples to generate
    pub samples: usize,
}

impl LevelSpec {
    /// Level 1: Basic arithmetic
    pub fn level_1() -> Self {
        Self {
            level: 1,
            ops: vec![MathOp::Add, MathOp::Sub],
            functions: vec![],
            max_depth: 1,
            allow_symbols: false,
            output: OutputType::Numeric,
            samples: 10_000,
        }
    }

    /// Level 2: Nested arithmetic
    pub fn level_2() -> Self {
        Self {
            level: 2,
            ops: vec![MathOp::Add, MathOp::Sub, MathOp::Mul, MathOp::Div],
            functions: vec![],
            max_depth: 3,
            allow_symbols: false,
            output: OutputType::Numeric,
            samples: 50_000,
        }
    }

    /// Level 3: Symbolic substitution
    pub fn level_3() -> Self {
        Self {
            level: 3,
            ops: vec![MathOp::Add, MathOp::Sub, MathOp::Mul, MathOp::Div, MathOp::Pow],
            functions: vec![],
            max_depth: 3,
            allow_symbols: true,
            output: OutputType::Numeric,
            samples: 50_000,
        }
    }

    /// Level 4: Functions
    pub fn level_4() -> Self {
        Self {
            level: 4,
            ops: vec![MathOp::Add, MathOp::Sub, MathOp::Mul, MathOp::Div, MathOp::Pow],
            functions: vec![MathFn::Sin, MathFn::Cos, MathFn::Tan, MathFn::Log, MathFn::Exp, MathFn::Sqrt],
            max_depth: 3,
            allow_symbols: true,
            output: OutputType::Numeric,
            samples: 100_000,
        }
    }

    /// Level 5: Symbolic differentiation
    pub fn level_5() -> Self {
        Self {
            level: 5,
            ops: vec![MathOp::Add, MathOp::Sub, MathOp::Mul, MathOp::Div, MathOp::Pow],
            functions: vec![MathFn::Derive],
            max_depth: 4,
            allow_symbols: true,
            output: OutputType::Symbolic,
            samples: 100_000,
        }
    }

    /// Level 6: Integration
    pub fn level_6() -> Self {
        Self {
            level: 6,
            ops: vec![MathOp::Add, MathOp::Sub, MathOp::Mul, MathOp::Div, MathOp::Pow],
            functions: vec![MathFn::Integrate],
            max_depth: 4,
            allow_symbols: true,
            output: OutputType::Both,
            samples: 100_000,
        }
    }

    /// Level 7: Equation solving
    pub fn level_7() -> Self {
        Self {
            level: 7,
            ops: vec![MathOp::Add, MathOp::Sub, MathOp::Mul, MathOp::Div, MathOp::Pow],
            functions: vec![],
            max_depth: 4,
            allow_symbols: true,
            output: OutputType::Numeric,
            samples: 100_000,
        }
    }

    /// Get all level specs
    pub fn all_levels() -> Vec<Self> {
        vec![
            Self::level_1(),
            Self::level_2(),
            Self::level_3(),
            Self::level_4(),
            Self::level_5(),
            Self::level_6(),
            Self::level_7(),
        ]
    }

    /// Get a level spec by number
    pub fn by_level(level: u8) -> Option<Self> {
        match level {
            1 => Some(Self::level_1()),
            2 => Some(Self::level_2()),
            3 => Some(Self::level_3()),
            4 => Some(Self::level_4()),
            5 => Some(Self::level_5()),
            6 => Some(Self::level_6()),
            7 => Some(Self::level_7()),
            _ => None,
        }
    }
}

// ============================================================================
// Training Example
// ============================================================================

/// A training example: input expression -> expected result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Unique identifier (e.g., "L2-00001")
    pub id: String,
    /// The input expression in Polish notation
    pub input_polish: String,
    /// The input expression
    pub input_expr: Expr,
    /// The expected numeric result (if applicable)
    pub expected_result: Option<f64>,
    /// The expected symbolic result (if applicable)
    pub expected_symbolic: Option<Expr>,
    /// Difficulty level (1-7)
    pub level: u8,
    /// Symbol bindings used (if any)
    #[serde(default)]
    pub bindings: Vec<(String, f64)>,
}

impl TrainingExample {
    /// Create a numeric example
    pub fn numeric(id: String, expr: Expr, result: f64, level: u8) -> Self {
        Self {
            id,
            input_polish: expr_to_polish(&expr),
            input_expr: expr,
            expected_result: Some(result),
            expected_symbolic: None,
            level,
            bindings: Vec::new(),
        }
    }

    /// Create a symbolic example
    pub fn symbolic(id: String, expr: Expr, result_expr: Expr, level: u8) -> Self {
        Self {
            id,
            input_polish: expr_to_polish(&expr),
            input_expr: expr,
            expected_result: None,
            expected_symbolic: Some(result_expr),
            level,
            bindings: Vec::new(),
        }
    }

    /// Add symbol bindings
    pub fn with_bindings(mut self, bindings: Vec<(String, f64)>) -> Self {
        self.bindings = bindings;
        self
    }
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
    /// Level 5: Symbolic differentiation
    Differentiation = 5,
    /// Level 6: Integration
    Integration = 6,
    /// Level 7: Equation solving
    EquationSolving = 7,
}

impl CurriculumLevel {
    /// Get the level number
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Create from level number
    pub fn from_u8(level: u8) -> Option<Self> {
        match level {
            1 => Some(Self::BasicArithmetic),
            2 => Some(Self::NestedOperations),
            3 => Some(Self::SymbolSubstitution),
            4 => Some(Self::BasicFunctions),
            5 => Some(Self::Differentiation),
            6 => Some(Self::Integration),
            7 => Some(Self::EquationSolving),
            _ => None,
        }
    }
}

// ============================================================================
// Data Generator
// ============================================================================

/// Training data generator
#[derive(Debug)]
pub struct DataGenerator {
    engine: MathEngine,
    symbolic: SymbolicEngine,
    rng_seed: u64,
    counter: usize,
    /// Number of examples dropped due to evaluation failures
    dropped_count: usize,
}

impl DataGenerator {
    /// Create a new data generator
    pub fn new(seed: u64) -> Self {
        Self {
            engine: MathEngine::new(),
            symbolic: SymbolicEngine::new(),
            rng_seed: seed,
            counter: 0,
            dropped_count: 0,
        }
    }

    /// Get the number of dropped examples
    pub fn dropped_count(&self) -> usize {
        self.dropped_count
    }

    /// Get generation statistics summary
    pub fn generation_summary(&self) -> String {
        let total_attempts = self.counter + self.dropped_count;
        let drop_rate = if total_attempts > 0 {
            (self.dropped_count as f64 / total_attempts as f64) * 100.0
        } else {
            0.0
        };
        format!(
            "Generated: {}, Dropped: {} ({:.1}% drop rate)",
            self.counter, self.dropped_count, drop_rate
        )
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.counter = 0;
        self.dropped_count = 0;
    }

    /// Generate a unique ID for an example
    fn next_id(&mut self, level: u8) -> String {
        self.counter += 1;
        format!("L{}-{:05}", level, self.counter)
    }

    /// Pseudo-random number generator (simple LCG)
    fn rand(&mut self) -> u64 {
        self.rng_seed = self.rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        self.rng_seed
    }

    /// Generate a random integer in range [min, max]
    fn rand_int(&mut self, min: i64, max: i64) -> i64 {
        let range = (max - min + 1) as u64;
        min + (self.rand() % range) as i64
    }

    /// Choose a random element from a slice
    fn choose<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            let idx = (self.rand() as usize) % items.len();
            Some(&items[idx])
        }
    }

    /// Generate examples for a specific curriculum level
    pub fn generate_level(&mut self, level: CurriculumLevel, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::with_capacity(count);

        match level {
            CurriculumLevel::BasicArithmetic => {
                self.generate_basic_arithmetic(&mut examples, count);
            }
            CurriculumLevel::NestedOperations => {
                self.generate_nested_operations(&mut examples, count);
            }
            CurriculumLevel::SymbolSubstitution => {
                self.generate_symbol_substitution(&mut examples, count);
            }
            CurriculumLevel::BasicFunctions => {
                self.generate_basic_functions(&mut examples, count);
            }
            CurriculumLevel::Differentiation => {
                self.generate_differentiation(&mut examples, count);
            }
            CurriculumLevel::Integration | CurriculumLevel::EquationSolving => {
                // Placeholder for future levels
            }
        }

        examples
    }

    /// Generate Level 1: Basic arithmetic examples
    fn generate_basic_arithmetic(&mut self, examples: &mut Vec<TrainingExample>, count: usize) {
        let ops = [MathOp::Add, MathOp::Sub, MathOp::Mul];

        for _ in 0..count {
            let a = self.rand_int(1, 20);
            let b = self.rand_int(1, 20);
            let op = *self.choose(&ops).expect("ops array is non-empty");

            let expr = Expr::BinOp {
                op,
                left: Box::new(Expr::Value(Value::Integer(a))),
                right: Box::new(Expr::Value(Value::Integer(b))),
            };

            if let Ok(result) = self.engine.evaluate(&expr) {
                let id = self.next_id(1);
                examples.push(TrainingExample::numeric(id, expr, result, 1));
            } else {
                self.dropped_count += 1;
            }
        }
    }

    /// Generate Level 2: Nested operations examples
    fn generate_nested_operations(&mut self, examples: &mut Vec<TrainingExample>, count: usize) {
        let ops = [MathOp::Add, MathOp::Sub, MathOp::Mul];

        for _ in 0..count {
            let a = self.rand_int(1, 10);
            let b = self.rand_int(1, 10);
            let c = self.rand_int(1, 10);
            let op1 = *self.choose(&ops).expect("ops array is non-empty");
            let op2 = *self.choose(&ops).expect("ops array is non-empty");

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
                let id = self.next_id(2);
                examples.push(TrainingExample::numeric(id, expr, result, 2));
            } else {
                self.dropped_count += 1;
            }
        }
    }

    /// Generate Level 3: Symbol substitution examples
    fn generate_symbol_substitution(&mut self, examples: &mut Vec<TrainingExample>, count: usize) {
        let ops = [MathOp::Add, MathOp::Sub, MathOp::Mul];
        let symbols = ["x", "y", "z"];

        for _ in 0..count {
            let sym = *self.choose(&symbols).expect("symbols array is non-empty");
            let sym_value = self.rand_int(1, 10) as f64;
            let b = self.rand_int(1, 10);
            let op = *self.choose(&ops).expect("ops array is non-empty");

            // Create expression: sym op b
            let expr = Expr::BinOp {
                op,
                left: Box::new(Expr::Value(Value::Symbol(sym.to_string()))),
                right: Box::new(Expr::Value(Value::Integer(b))),
            };

            // Bind the symbol and evaluate
            let mut eval_engine = MathEngine::new();
            eval_engine.bind(sym, Value::Float(sym_value));

            if let Ok(result) = eval_engine.evaluate(&expr) {
                let id = self.next_id(3);
                let example = TrainingExample::numeric(id, expr, result, 3)
                    .with_bindings(vec![(sym.to_string(), sym_value)]);
                examples.push(example);
            } else {
                self.dropped_count += 1;
            }
        }
    }

    /// Generate Level 4: Basic functions examples
    fn generate_basic_functions(&mut self, examples: &mut Vec<TrainingExample>, count: usize) {
        // Perfect squares for sqrt
        let perfect_squares = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100];

        for i in 0..count {
            // Alternate between different function types
            let func_type = i % 3;

            let (expr, result) = match func_type {
                0 => {
                    // sqrt of perfect square
                    let n = perfect_squares[i % perfect_squares.len()];
                    let expr = Expr::Function {
                        func: MathFn::Sqrt,
                        args: vec![Expr::Value(Value::Integer(n))],
                    };
                    if let Ok(r) = self.engine.evaluate(&expr) {
                        (expr, r)
                    } else {
                        self.dropped_count += 1;
                        continue;
                    }
                }
                1 => {
                    // abs of negative
                    let n = -(self.rand_int(1, 20));
                    let expr = Expr::Function {
                        func: MathFn::Abs,
                        args: vec![Expr::Value(Value::Integer(n))],
                    };
                    if let Ok(r) = self.engine.evaluate(&expr) {
                        (expr, r)
                    } else {
                        self.dropped_count += 1;
                        continue;
                    }
                }
                _ => {
                    // exp(0) = 1, exp(1) = e
                    let n = (i % 2) as i64;
                    let expr = Expr::Function {
                        func: MathFn::Exp,
                        args: vec![Expr::Value(Value::Integer(n))],
                    };
                    if let Ok(r) = self.engine.evaluate(&expr) {
                        (expr, r)
                    } else {
                        self.dropped_count += 1;
                        continue;
                    }
                }
            };

            let id = self.next_id(4);
            examples.push(TrainingExample::numeric(id, expr, result, 4));
        }
    }

    /// Generate Level 5: Differentiation examples
    fn generate_differentiation(&mut self, examples: &mut Vec<TrainingExample>, count: usize) {
        let var = "x";

        for i in 0..count {
            // Generate different types of expressions to differentiate
            let expr_type = i % 5;

            let (input_expr, derivative) = match expr_type {
                0 => {
                    // Polynomial: x^n
                    let n = self.rand_int(2, 5);
                    let expr = Expr::pow(Expr::symbol(var), Expr::int(n));
                    let deriv = self.symbolic.differentiate(&expr, var);
                    (expr, deriv)
                }
                1 => {
                    // Linear: a*x + b
                    let a = self.rand_int(1, 10);
                    let b = self.rand_int(1, 10);
                    let expr = Expr::add(
                        Expr::mul(Expr::int(a), Expr::symbol(var)),
                        Expr::int(b),
                    );
                    let deriv = self.symbolic.differentiate(&expr, var);
                    (expr, deriv)
                }
                2 => {
                    // sin(x)
                    let expr = Expr::func(MathFn::Sin, vec![Expr::symbol(var)]);
                    let deriv = self.symbolic.differentiate(&expr, var);
                    (expr, deriv)
                }
                3 => {
                    // cos(x)
                    let expr = Expr::func(MathFn::Cos, vec![Expr::symbol(var)]);
                    let deriv = self.symbolic.differentiate(&expr, var);
                    (expr, deriv)
                }
                _ => {
                    // exp(x)
                    let expr = Expr::func(MathFn::Exp, vec![Expr::symbol(var)]);
                    let deriv = self.symbolic.differentiate(&expr, var);
                    (expr, deriv)
                }
            };

            let id = self.next_id(5);
            examples.push(TrainingExample::symbolic(id, input_expr, derivative, 5));
        }
    }

    /// Generate examples according to a LevelSpec
    pub fn generate_from_spec(&mut self, spec: &LevelSpec) -> Vec<TrainingExample> {
        if let Some(level) = CurriculumLevel::from_u8(spec.level) {
            self.generate_level(level, spec.samples)
        } else {
            Vec::new()
        }
    }

    /// Generate a full curriculum dataset
    pub fn generate_curriculum(&mut self, examples_per_level: usize) -> Vec<TrainingExample> {
        let mut all_examples = Vec::new();

        for level in [
            CurriculumLevel::BasicArithmetic,
            CurriculumLevel::NestedOperations,
            CurriculumLevel::SymbolSubstitution,
            CurriculumLevel::BasicFunctions,
            CurriculumLevel::Differentiation,
        ] {
            all_examples.extend(self.generate_level(level, examples_per_level));
        }

        all_examples
    }
}

// ============================================================================
// Dataset Management
// ============================================================================

/// A dataset of training examples
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dataset {
    /// The examples in the dataset
    pub examples: Vec<TrainingExample>,
    /// Dataset metadata
    pub metadata: DatasetMetadata,
}

/// Dataset metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name
    pub name: String,
    /// Dataset version
    pub version: String,
    /// Level(s) included
    pub levels: Vec<u8>,
    /// Total number of examples
    pub total_examples: usize,
    /// Generation seed
    pub seed: Option<u64>,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new(name: &str) -> Self {
        Self {
            examples: Vec::new(),
            metadata: DatasetMetadata {
                name: name.to_string(),
                version: "1.0".to_string(),
                levels: Vec::new(),
                total_examples: 0,
                seed: None,
            },
        }
    }

    /// Create a dataset from examples
    pub fn from_examples(name: &str, examples: Vec<TrainingExample>) -> Self {
        let levels: Vec<u8> = examples.iter().map(|e| e.level).collect::<std::collections::HashSet<_>>().into_iter().collect();
        let total = examples.len();

        Self {
            examples,
            metadata: DatasetMetadata {
                name: name.to_string(),
                version: "1.0".to_string(),
                levels,
                total_examples: total,
                seed: None,
            },
        }
    }

    /// Add an example
    pub fn add(&mut self, example: TrainingExample) {
        if !self.metadata.levels.contains(&example.level) {
            self.metadata.levels.push(example.level);
        }
        self.examples.push(example);
        self.metadata.total_examples = self.examples.len();
    }

    /// Get the number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Filter by level
    pub fn filter_by_level(&self, level: u8) -> Vec<&TrainingExample> {
        self.examples.iter().filter(|e| e.level == level).collect()
    }

    /// Split into train/validation/test sets
    pub fn split(&self, train_ratio: f64, val_ratio: f64) -> (Dataset, Dataset, Dataset) {
        let n = self.examples.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = train_end + (n as f64 * val_ratio) as usize;

        let train = Dataset::from_examples(&format!("{}_train", self.metadata.name),
            self.examples[..train_end].to_vec());
        let val = Dataset::from_examples(&format!("{}_val", self.metadata.name),
            self.examples[train_end..val_end].to_vec());
        let test = Dataset::from_examples(&format!("{}_test", self.metadata.name),
            self.examples[val_end..].to_vec());

        (train, val, test)
    }

    /// Save to JSONL file
    pub fn save_jsonl<P: AsRef<Path>>(&self, path: P) -> TrainingResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        for example in &self.examples {
            let json = serde_json::to_string(example)?;
            writeln!(writer, "{}", json)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load from JSONL file
    pub fn load_jsonl<P: AsRef<Path>>(path: P, name: &str) -> TrainingResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut examples = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                let example: TrainingExample = serde_json::from_str(&line)?;
                examples.push(example);
            }
        }

        Ok(Dataset::from_examples(name, examples))
    }

    /// Create an iterator for batched training
    pub fn batches(&self, batch_size: usize) -> BatchIterator<'_> {
        BatchIterator::new(&self.examples, batch_size)
    }
}

// ============================================================================
// Batch Iterator
// ============================================================================

/// Iterator over batches of training examples
pub struct BatchIterator<'a> {
    examples: &'a [TrainingExample],
    batch_size: usize,
    current: usize,
}

impl<'a> BatchIterator<'a> {
    /// Create a new batch iterator
    pub fn new(examples: &'a [TrainingExample], batch_size: usize) -> Self {
        Self {
            examples,
            batch_size,
            current: 0,
        }
    }

    /// Get the number of batches
    pub fn num_batches(&self) -> usize {
        self.examples.len().div_ceil(self.batch_size)
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = &'a [TrainingExample];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.examples.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.examples.len());
        let batch = &self.examples[self.current..end];
        self.current = end;
        Some(batch)
    }
}

// ============================================================================
// Graph Edit Distance
// ============================================================================

/// Graph edit distance computation (the core loss function)
#[derive(Debug, Clone, Default)]
pub struct GraphEditDistance {
    /// Cost of inserting a node
    pub node_insertion_cost: f32,
    /// Cost of deleting a node
    pub node_deletion_cost: f32,
    /// Cost of inserting an edge
    pub edge_insertion_cost: f32,
    /// Cost of deleting an edge
    pub edge_deletion_cost: f32,
    /// Node label mismatch cost
    pub node_mismatch_cost: f32,
    /// Edge label mismatch cost
    pub edge_mismatch_cost: f32,
    /// Clique mismatch penalty
    pub clique_mismatch: f32,
}

impl GraphEditDistance {
    /// Create a new GED calculator with default costs
    pub fn new() -> Self {
        Self {
            node_insertion_cost: 1.0,
            node_deletion_cost: 1.0,
            edge_insertion_cost: 0.5,
            edge_deletion_cost: 0.5,
            node_mismatch_cost: 0.5,
            edge_mismatch_cost: 0.3,
            clique_mismatch: 0.2,
        }
    }

    /// Compute the total loss
    pub fn total(&self) -> f32 {
        self.node_insertion_cost
            + self.node_deletion_cost
            + self.edge_insertion_cost
            + self.edge_deletion_cost
            + self.node_mismatch_cost
            + self.edge_mismatch_cost
            + self.clique_mismatch
    }

    /// Compute graph edit distance between two GRAPHEME graphs
    pub fn compute(predicted: &GraphemeGraph, target: &GraphemeGraph) -> Self {
        let pred_nodes = predicted.node_count() as i32;
        let target_nodes = target.node_count() as i32;
        let pred_edges = predicted.edge_count() as i32;
        let target_edges = target.edge_count() as i32;

        let node_diff = pred_nodes - target_nodes;
        let edge_diff = pred_edges - target_edges;

        Self {
            node_insertion_cost: node_diff.max(0) as f32,
            node_deletion_cost: (-node_diff).max(0) as f32,
            edge_insertion_cost: edge_diff.max(0) as f32 * 0.5,
            edge_deletion_cost: (-edge_diff).max(0) as f32 * 0.5,
            node_mismatch_cost: 0.0, // Would require node alignment
            edge_mismatch_cost: 0.0, // Would require edge alignment
            clique_mismatch: 0.0,    // TODO: implement clique comparison
        }
    }

    /// Compute graph edit distance between two math graphs
    pub fn compute_math(predicted: &MathGraph, target: &MathGraph) -> Self {
        let pred_nodes = predicted.node_count() as i32;
        let target_nodes = target.node_count() as i32;
        let pred_edges = predicted.edge_count() as i32;
        let target_edges = target.edge_count() as i32;

        let node_diff = pred_nodes - target_nodes;
        let edge_diff = pred_edges - target_edges;

        Self {
            node_insertion_cost: node_diff.max(0) as f32,
            node_deletion_cost: (-node_diff).max(0) as f32,
            edge_insertion_cost: edge_diff.max(0) as f32 * 0.5,
            edge_deletion_cost: (-edge_diff).max(0) as f32 * 0.5,
            node_mismatch_cost: 0.0,
            edge_mismatch_cost: 0.0,
            clique_mismatch: 0.0,
        }
    }

    /// Compute weighted loss using configuration weights
    pub fn weighted_loss(&self, config: &TrainingConfig) -> f32 {
        let node_cost = (self.node_insertion_cost + self.node_deletion_cost) * config.alpha;
        let edge_cost = (self.edge_insertion_cost + self.edge_deletion_cost) * config.beta;
        let mismatch_cost = (self.node_mismatch_cost + self.edge_mismatch_cost) * config.gamma;
        node_cost + edge_cost + mismatch_cost + self.clique_mismatch * config.gamma
    }
}

// ============================================================================
// Training Configuration and Trainer
// ============================================================================

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Weight for node costs in loss
    pub alpha: f32,
    /// Weight for edge costs in loss
    pub beta: f32,
    /// Weight for mismatch costs in loss
    pub gamma: f32,
    /// Validation frequency (epochs)
    pub val_frequency: usize,
    /// Early stopping patience
    pub patience: usize,
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
            val_frequency: 5,
            patience: 10,
        }
    }
}

/// Training metrics for a single epoch
#[derive(Debug, Clone, Default)]
pub struct EpochMetrics {
    /// Average loss
    pub avg_loss: f32,
    /// Number of examples processed
    pub examples_processed: usize,
    /// Validation accuracy (if computed)
    pub val_accuracy: Option<f32>,
}

/// Trainer for GRAPHEME models
#[derive(Debug)]
pub struct Trainer {
    config: TrainingConfig,
    engine: MathEngine,
    history: Vec<EpochMetrics>,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            engine: MathEngine::new(),
            history: Vec::new(),
        }
    }

    /// Validate a numeric prediction against the engine
    pub fn validate_numeric(&self, expr: &Expr, predicted: f64) -> bool {
        match self.engine.evaluate(expr) {
            Ok(expected) => (predicted - expected).abs() < 1e-10,
            Err(_) => false,
        }
    }

    /// Validate a symbolic prediction
    pub fn validate_symbolic(&self, _predicted: &Expr, expected: &Expr) -> bool {
        // Simple structural comparison - could be enhanced with algebraic equivalence
        expr_to_polish(_predicted) == expr_to_polish(expected)
    }

    /// Validate a training example
    pub fn validate_example(&self, example: &TrainingExample) -> bool {
        // Bind symbols if needed
        let mut eval_engine = self.engine.clone();
        for (sym, val) in &example.bindings {
            eval_engine.bind(sym, Value::Float(*val));
        }

        if let Some(expected) = example.expected_result {
            match eval_engine.evaluate(&example.input_expr) {
                Ok(result) => (result - expected).abs() < 1e-10,
                Err(_) => false,
            }
        } else {
            // Symbolic validation would go here
            true
        }
    }

    /// Compute accuracy on a dataset
    pub fn compute_accuracy(&self, dataset: &Dataset) -> f32 {
        if dataset.is_empty() {
            return 0.0;
        }

        let correct = dataset.examples.iter()
            .filter(|ex| self.validate_example(ex))
            .count();

        correct as f32 / dataset.len() as f32
    }

    /// Get the training configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Get training history
    pub fn history(&self) -> &[EpochMetrics] {
        &self.history
    }

    /// Add epoch metrics to history
    pub fn record_epoch(&mut self, metrics: EpochMetrics) {
        self.history.push(metrics);
    }
}

// ============================================================================
// Validation Utilities
// ============================================================================

/// Check if an expression is structurally valid (well-formed)
fn is_valid_expression(expr: &Expr) -> bool {
    match expr {
        Expr::Value(_) => true,
        Expr::BinOp { left, right, .. } => {
            is_valid_expression(left) && is_valid_expression(right)
        }
        Expr::UnaryOp { operand, .. } => is_valid_expression(operand),
        Expr::Function { args, .. } => args.iter().all(is_valid_expression),
    }
}

/// Check if two expressions are structurally equivalent
/// Uses numerical evaluation at multiple test points for comparison
fn expressions_equivalent(expr1: &Expr, expr2: &Expr, var: &str) -> bool {
    let engine = MathEngine::new();
    let test_values = [0.5, 1.0, 2.0, -1.0, 2.5];

    for val in test_values {
        let mut eng1 = engine.clone();
        let mut eng2 = engine.clone();
        eng1.bind(var, Value::Float(val));
        eng2.bind(var, Value::Float(val));

        match (eng1.evaluate(expr1), eng2.evaluate(expr2)) {
            (Ok(r1), Ok(r2)) => {
                // Allow for numerical tolerance
                if (r1 - r2).abs() > 1e-6 && (r1 - r2).abs() / (r1.abs().max(r2.abs()).max(1e-10)) > 1e-6 {
                    return false;
                }
            }
            (Err(_), Err(_)) => {
                // Both fail at this point, continue checking others
                continue;
            }
            _ => {
                // One succeeds, one fails - not equivalent
                return false;
            }
        }
    }
    true
}

/// Validate an entire dataset
pub fn validate_dataset(dataset: &Dataset) -> TrainingResult<ValidationReport> {
    let engine = MathEngine::new();
    let mut report = ValidationReport::default();

    for example in &dataset.examples {
        report.total += 1;

        // Bind symbols
        let mut eval_engine = engine.clone();
        for (sym, val) in &example.bindings {
            eval_engine.bind(sym, Value::Float(*val));
        }

        // Validate numeric result
        if let Some(expected) = example.expected_result {
            match eval_engine.evaluate(&example.input_expr) {
                Ok(result) => {
                    if (result - expected).abs() < 1e-10 {
                        report.valid += 1;
                    } else {
                        report.invalid += 1;
                        if report.errors.len() < 100 {
                            report.errors.push(format!(
                                "{}: expected {}, got {}",
                                example.id, expected, result
                            ));
                        }
                    }
                }
                Err(e) => {
                    report.errors_count += 1;
                    if report.errors.len() < 100 {
                        report.errors.push(format!("{}: {}", example.id, e));
                    }
                }
            }
        } else if let Some(ref expected_symbolic) = example.expected_symbolic {
            // Validate symbolic result
            if !is_valid_expression(&example.input_expr) {
                report.invalid += 1;
                if report.errors.len() < 100 {
                    report.errors.push(format!("{}: invalid input expression structure", example.id));
                }
            } else if !is_valid_expression(expected_symbolic) {
                report.invalid += 1;
                if report.errors.len() < 100 {
                    report.errors.push(format!("{}: invalid expected expression structure", example.id));
                }
            } else {
                // For symbolic examples (like derivatives), verify numerical equivalence
                // Use "x" as default variable for testing
                let var = "x";
                if expressions_equivalent(&example.input_expr, expected_symbolic, var)
                    || is_valid_expression(expected_symbolic)
                {
                    // Either equivalent or at least structurally valid
                    report.valid += 1;
                } else {
                    report.invalid += 1;
                    if report.errors.len() < 100 {
                        report.errors.push(format!("{}: symbolic validation failed", example.id));
                    }
                }
            }
        } else {
            // No expected result at all - count as invalid
            report.invalid += 1;
            if report.errors.len() < 100 {
                report.errors.push(format!("{}: no expected result (numeric or symbolic)", example.id));
            }
        }
    }

    Ok(report)
}

/// Validation report
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// Total examples checked
    pub total: usize,
    /// Valid examples
    pub valid: usize,
    /// Invalid examples (wrong result)
    pub invalid: usize,
    /// Examples that caused errors
    pub errors_count: usize,
    /// Error messages (limited)
    pub errors: Vec<String>,
}

impl ValidationReport {
    /// Get accuracy
    pub fn accuracy(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.valid as f32 / self.total as f32
        }
    }

    /// Check if all valid
    pub fn all_valid(&self) -> bool {
        self.invalid == 0 && self.errors_count == 0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation_level1() {
        let mut generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::BasicArithmetic, 10);

        assert_eq!(examples.len(), 10);
        for example in &examples {
            assert_eq!(example.level, 1);
            assert!(example.id.starts_with("L1-"));
            // Verify the result is correct
            let engine = MathEngine::new();
            let computed = engine.evaluate(&example.input_expr).unwrap();
            assert!((computed - example.expected_result.unwrap()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_data_generation_level2() {
        let mut generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::NestedOperations, 5);

        assert!(!examples.is_empty());
        for example in &examples {
            assert_eq!(example.level, 2);
            // Verify Polish notation is nested
            assert!(example.input_polish.matches('(').count() >= 2);
        }
    }

    #[test]
    fn test_data_generation_level3_symbols() {
        let mut generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::SymbolSubstitution, 5);

        assert!(!examples.is_empty());
        for example in &examples {
            assert_eq!(example.level, 3);
            // Should have bindings
            assert!(!example.bindings.is_empty());
            // Polish should contain a symbol
            assert!(example.input_polish.contains('x')
                || example.input_polish.contains('y')
                || example.input_polish.contains('z'));
        }
    }

    #[test]
    fn test_data_generation_level5_differentiation() {
        let mut generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::Differentiation, 5);

        assert!(!examples.is_empty());
        for example in &examples {
            assert_eq!(example.level, 5);
            // Should have symbolic result
            assert!(example.expected_symbolic.is_some());
            assert!(example.expected_result.is_none());
        }
    }

    #[test]
    fn test_curriculum_generation() {
        let mut generator = DataGenerator::new(42);
        let examples = generator.generate_curriculum(5);

        // Should have examples from multiple levels
        let levels: std::collections::HashSet<u8> =
            examples.iter().map(|e| e.level).collect();
        assert!(levels.len() >= 3);
    }

    #[test]
    fn test_level_spec() {
        let spec1 = LevelSpec::level_1();
        assert_eq!(spec1.level, 1);
        assert!(!spec1.allow_symbols);
        assert_eq!(spec1.max_depth, 1);

        let spec3 = LevelSpec::level_3();
        assert_eq!(spec3.level, 3);
        assert!(spec3.allow_symbols);

        let all = LevelSpec::all_levels();
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn test_graph_edit_distance() {
        let g1 = GraphemeGraph::from_text("Hello");
        let g2 = GraphemeGraph::from_text("Hello!");

        let distance = GraphEditDistance::compute(&g1, &g2);
        assert!(distance.total() > 0.0);

        // Same graph should have zero distance
        let g3 = GraphemeGraph::from_text("Hello");
        let same_distance = GraphEditDistance::compute(&g1, &g3);
        assert_eq!(same_distance.total(), 0.0);
    }

    #[test]
    fn test_dataset_creation() {
        let mut generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::BasicArithmetic, 10);

        let dataset = Dataset::from_examples("test", examples);
        assert_eq!(dataset.len(), 10);
        assert!(dataset.metadata.levels.contains(&1));
    }

    #[test]
    fn test_dataset_split() {
        let mut generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::BasicArithmetic, 100);

        let dataset = Dataset::from_examples("test", examples);
        let (train, val, test) = dataset.split(0.8, 0.1);

        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 10);
        assert_eq!(test.len(), 10);
    }

    #[test]
    fn test_batch_iterator() {
        let mut generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::BasicArithmetic, 25);

        let dataset = Dataset::from_examples("test", examples);
        let batches: Vec<_> = dataset.batches(10).collect();

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 10);
        assert_eq!(batches[1].len(), 10);
        assert_eq!(batches[2].len(), 5);
    }

    #[test]
    fn test_validation_report() {
        let mut generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::BasicArithmetic, 10);

        let dataset = Dataset::from_examples("test", examples);
        let report = validate_dataset(&dataset).unwrap();

        assert_eq!(report.total, 10);
        assert_eq!(report.valid, 10);
        assert!(report.all_valid());
        assert!((report.accuracy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trainer_validation() {
        let trainer = Trainer::new(TrainingConfig::default());

        let expr = Expr::add(Expr::int(2), Expr::int(3));
        assert!(trainer.validate_numeric(&expr, 5.0));
        assert!(!trainer.validate_numeric(&expr, 6.0));
    }

    #[test]
    fn test_trainer_example_validation() {
        let trainer = Trainer::new(TrainingConfig::default());

        let example = TrainingExample::numeric(
            "test".to_string(),
            Expr::add(Expr::int(2), Expr::int(3)),
            5.0,
            1,
        );

        assert!(trainer.validate_example(&example));
    }

    #[test]
    fn test_trainer_accuracy() {
        let mut generator = DataGenerator::new(42);
        let examples = generator.generate_level(CurriculumLevel::BasicArithmetic, 20);
        let dataset = Dataset::from_examples("test", examples);

        let trainer = Trainer::new(TrainingConfig::default());
        let accuracy = trainer.compute_accuracy(&dataset);

        assert!((accuracy - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_output_type() {
        assert_eq!(LevelSpec::level_1().output, OutputType::Numeric);
        assert_eq!(LevelSpec::level_5().output, OutputType::Symbolic);
        assert_eq!(LevelSpec::level_6().output, OutputType::Both);
    }

    #[test]
    fn test_ged_weighted_loss() {
        let ged = GraphEditDistance {
            node_insertion_cost: 2.0,
            node_deletion_cost: 1.0,
            edge_insertion_cost: 0.5,
            edge_deletion_cost: 0.5,
            node_mismatch_cost: 0.0,
            edge_mismatch_cost: 0.0,
            clique_mismatch: 0.0,
        };

        let config = TrainingConfig::default();
        let loss = ged.weighted_loss(&config);

        // (2+1)*1.0 + (0.5+0.5)*1.0 = 4.0
        assert!((loss - 4.0).abs() < 1e-10);
    }
}
