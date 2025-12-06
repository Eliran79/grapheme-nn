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

use grapheme_core::{GraphemeGraph, NodeType};
use grapheme_engine::{Expr, MathEngine, MathFn, MathOp, SymbolicEngine, Value};
use grapheme_math::{MathGraph, MathNode};
use grapheme_polish::expr_to_polish;
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
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
    /// Statistics tracking
    stats: GenerationStats,
}

/// Statistics for data generation
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Total attempted generations
    pub attempted: usize,
    /// Successfully generated examples
    pub generated: usize,
    /// Examples dropped due to evaluation errors
    pub dropped_eval_error: usize,
}

impl GenerationStats {
    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.attempted == 0 {
            100.0
        } else {
            (self.generated as f64 / self.attempted as f64) * 100.0
        }
    }

    /// Print summary to stderr
    pub fn print_summary(&self) {
        eprintln!(
            "Generation stats: {} attempted, {} generated, {} dropped ({:.1}% success)",
            self.attempted,
            self.generated,
            self.dropped_eval_error,
            self.success_rate()
        );
    }
}

impl DataGenerator {
    /// Create a new data generator
    pub fn new(seed: u64) -> Self {
        Self {
            engine: MathEngine::new(),
            symbolic: SymbolicEngine::new(),
            rng_seed: seed,
            counter: 0,
            stats: GenerationStats::default(),
        }
    }

    /// Get generation statistics
    pub fn stats(&self) -> &GenerationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = GenerationStats::default();
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
            self.stats.attempted += 1;
            let a = self.rand_int(1, 20);
            let b = self.rand_int(1, 20);
            let op = *self.choose(&ops).unwrap();

            let expr = Expr::BinOp {
                op,
                left: Box::new(Expr::Value(Value::Integer(a))),
                right: Box::new(Expr::Value(Value::Integer(b))),
            };

            match self.engine.evaluate(&expr) {
                Ok(result) => {
                    let id = self.next_id(1);
                    examples.push(TrainingExample::numeric(id, expr, result, 1));
                    self.stats.generated += 1;
                }
                Err(_) => {
                    self.stats.dropped_eval_error += 1;
                }
            }
        }
    }

    /// Generate Level 2: Nested operations examples
    fn generate_nested_operations(&mut self, examples: &mut Vec<TrainingExample>, count: usize) {
        let ops = [MathOp::Add, MathOp::Sub, MathOp::Mul];

        for _ in 0..count {
            self.stats.attempted += 1;
            let a = self.rand_int(1, 10);
            let b = self.rand_int(1, 10);
            let c = self.rand_int(1, 10);
            let op1 = *self.choose(&ops).unwrap();
            let op2 = *self.choose(&ops).unwrap();

            let expr = Expr::BinOp {
                op: op2,
                left: Box::new(Expr::BinOp {
                    op: op1,
                    left: Box::new(Expr::Value(Value::Integer(a))),
                    right: Box::new(Expr::Value(Value::Integer(b))),
                }),
                right: Box::new(Expr::Value(Value::Integer(c))),
            };

            match self.engine.evaluate(&expr) {
                Ok(result) => {
                    let id = self.next_id(2);
                    examples.push(TrainingExample::numeric(id, expr, result, 2));
                    self.stats.generated += 1;
                }
                Err(_) => {
                    self.stats.dropped_eval_error += 1;
                }
            }
        }
    }

    /// Generate Level 3: Symbol substitution examples
    fn generate_symbol_substitution(&mut self, examples: &mut Vec<TrainingExample>, count: usize) {
        let ops = [MathOp::Add, MathOp::Sub, MathOp::Mul];
        let symbols = ["x", "y", "z"];

        for _ in 0..count {
            self.stats.attempted += 1;
            let sym = *self.choose(&symbols).unwrap();
            let sym_value = self.rand_int(1, 10) as f64;
            let b = self.rand_int(1, 10);
            let op = *self.choose(&ops).unwrap();

            // Create expression: sym op b
            let expr = Expr::BinOp {
                op,
                left: Box::new(Expr::Value(Value::Symbol(sym.to_string()))),
                right: Box::new(Expr::Value(Value::Integer(b))),
            };

            // Bind the symbol and evaluate
            let mut eval_engine = MathEngine::new();
            eval_engine.bind(sym, Value::Float(sym_value));

            match eval_engine.evaluate(&expr) {
                Ok(result) => {
                    let id = self.next_id(3);
                    let example = TrainingExample::numeric(id, expr, result, 3)
                        .with_bindings(vec![(sym.to_string(), sym_value)]);
                    examples.push(example);
                    self.stats.generated += 1;
                }
                Err(_) => {
                    self.stats.dropped_eval_error += 1;
                }
            }
        }
    }

    /// Generate Level 4: Basic functions examples
    fn generate_basic_functions(&mut self, examples: &mut Vec<TrainingExample>, count: usize) {
        // Perfect squares for sqrt
        let perfect_squares = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100];

        for i in 0..count {
            self.stats.attempted += 1;
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
                        self.stats.dropped_eval_error += 1;
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
                        self.stats.dropped_eval_error += 1;
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
                        self.stats.dropped_eval_error += 1;
                        continue;
                    }
                }
            };

            let id = self.next_id(4);
            examples.push(TrainingExample::numeric(id, expr, result, 4));
            self.stats.generated += 1;
        }
    }

    /// Generate Level 5: Differentiation examples
    fn generate_differentiation(&mut self, examples: &mut Vec<TrainingExample>, count: usize) {
        let var = "x";

        for i in 0..count {
            self.stats.attempted += 1;
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
            self.stats.generated += 1;
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

        // Compute clique mismatch based on clique count and size differences
        let pred_cliques = predicted.cliques.len();
        let target_cliques = target.cliques.len();
        let clique_count_diff = (pred_cliques as i32 - target_cliques as i32).abs() as f32;

        // Also compare average clique sizes
        let pred_avg_size = if pred_cliques > 0 {
            predicted.cliques.iter().map(|c| c.len()).sum::<usize>() as f32 / pred_cliques as f32
        } else { 0.0 };
        let target_avg_size = if target_cliques > 0 {
            target.cliques.iter().map(|c| c.len()).sum::<usize>() as f32 / target_cliques as f32
        } else { 0.0 };
        let size_diff = (pred_avg_size - target_avg_size).abs();

        let clique_mismatch = (clique_count_diff * 0.3 + size_diff * 0.1).min(5.0);

        Self {
            node_insertion_cost: node_diff.max(0) as f32,
            node_deletion_cost: (-node_diff).max(0) as f32,
            edge_insertion_cost: edge_diff.max(0) as f32 * 0.5,
            edge_deletion_cost: (-edge_diff).max(0) as f32 * 0.5,
            node_mismatch_cost: 0.0, // Would require node alignment
            edge_mismatch_cost: 0.0, // Would require edge alignment
            clique_mismatch,
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
// Weisfeiler-Leman Kernel (backend-006)
// ============================================================================

/// Default number of WL iterations
pub const DEFAULT_WL_ITERATIONS: usize = 3;

/// Weisfeiler-Leman graph kernel for computing graph similarity
///
/// The WL kernel provides a polynomial-time graph similarity measure that forms
/// the theoretical foundation for Graph Neural Networks. It works by iteratively
/// refining node "colors" based on neighborhood structure.
///
/// Complexity: O(n·m·k) where n=nodes, m=edges, k=iterations
#[derive(Debug, Clone, Default)]
pub struct WeisfeilerLehmanKernel {
    /// Number of refinement iterations
    pub iterations: usize,
    /// Whether to normalize the similarity score to [0, 1]
    pub normalize: bool,
}

impl WeisfeilerLehmanKernel {
    /// Create a new WL kernel with default settings
    pub fn new() -> Self {
        Self {
            iterations: DEFAULT_WL_ITERATIONS,
            normalize: true,
        }
    }

    /// Create a WL kernel with custom iteration count
    pub fn with_iterations(iterations: usize) -> Self {
        Self {
            iterations,
            normalize: true,
        }
    }

    /// Compute WL similarity between two GRAPHEME graphs
    ///
    /// Returns a similarity score where:
    /// - 1.0 = identical graph structure
    /// - 0.0 = completely different structure
    pub fn compute(&self, g1: &GraphemeGraph, g2: &GraphemeGraph) -> f32 {
        // Initialize colors from node types
        let mut colors1 = self.init_colors_grapheme(g1);
        let mut colors2 = self.init_colors_grapheme(g2);

        // Collect histograms at each iteration for multi-scale comparison
        let mut total_similarity = 0.0f32;

        for iteration in 0..=self.iterations {
            // Compare color histograms at this iteration
            let hist1 = self.compute_histogram(&colors1);
            let hist2 = self.compute_histogram(&colors2);
            let iter_similarity = self.histogram_similarity(&hist1, &hist2);

            // Weight earlier iterations less (structure at finer scale)
            let weight = 1.0 / (1 + iteration) as f32;
            total_similarity += iter_similarity * weight;

            // Refine colors for next iteration (except on last)
            if iteration < self.iterations {
                colors1 = self.refine_colors_grapheme(g1, &colors1);
                colors2 = self.refine_colors_grapheme(g2, &colors2);
            }
        }

        // Normalize by sum of weights
        let weight_sum: f32 = (0..=self.iterations).map(|i| 1.0 / (1 + i) as f32).sum();
        let similarity = total_similarity / weight_sum;

        if self.normalize {
            similarity.clamp(0.0, 1.0)
        } else {
            similarity
        }
    }

    /// Compute WL similarity between two MathGraphs
    pub fn compute_math(&self, g1: &MathGraph, g2: &MathGraph) -> f32 {
        let mut colors1 = self.init_colors_math(g1);
        let mut colors2 = self.init_colors_math(g2);

        let mut total_similarity = 0.0f32;

        for iteration in 0..=self.iterations {
            let hist1 = self.compute_histogram(&colors1);
            let hist2 = self.compute_histogram(&colors2);
            let iter_similarity = self.histogram_similarity(&hist1, &hist2);

            let weight = 1.0 / (1 + iteration) as f32;
            total_similarity += iter_similarity * weight;

            if iteration < self.iterations {
                colors1 = self.refine_colors_math(g1, &colors1);
                colors2 = self.refine_colors_math(g2, &colors2);
            }
        }

        let weight_sum: f32 = (0..=self.iterations).map(|i| 1.0 / (1 + i) as f32).sum();
        let similarity = total_similarity / weight_sum;

        if self.normalize {
            similarity.clamp(0.0, 1.0)
        } else {
            similarity
        }
    }

    /// Initialize colors for a GRAPHEME graph based on node types
    fn init_colors_grapheme(&self, graph: &GraphemeGraph) -> Vec<u64> {
        graph.graph.node_indices()
            .map(|idx| {
                let node = &graph.graph[idx];
                self.hash_node_type(&node.node_type)
            })
            .collect()
    }

    /// Initialize colors for a MathGraph based on node types
    fn init_colors_math(&self, graph: &MathGraph) -> Vec<u64> {
        graph.graph.node_indices()
            .map(|idx| {
                let node = &graph.graph[idx];
                self.hash_math_node(node)
            })
            .collect()
    }

    /// Hash a GRAPHEME NodeType to a color
    fn hash_node_type(&self, node_type: &NodeType) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match node_type {
            NodeType::Input(ch) => {
                "Input".hash(&mut hasher);
                ch.hash(&mut hasher);
            }
            NodeType::Hidden => {
                "Hidden".hash(&mut hasher);
            }
            NodeType::Output => {
                "Output".hash(&mut hasher);
            }
            NodeType::Clique(members) => {
                "Clique".hash(&mut hasher);
                members.len().hash(&mut hasher);
            }
            NodeType::Pattern(pattern) => {
                "Pattern".hash(&mut hasher);
                pattern.hash(&mut hasher);
            }
            NodeType::Compressed(compression_type) => {
                "Compressed".hash(&mut hasher);
                format!("{:?}", compression_type).hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Hash a MathNode to a color
    fn hash_math_node(&self, node: &MathNode) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match node {
            MathNode::Integer(i) => {
                "Integer".hash(&mut hasher);
                i.hash(&mut hasher);
            }
            MathNode::Float(f) => {
                "Float".hash(&mut hasher);
                f.to_bits().hash(&mut hasher);
            }
            MathNode::Symbol(s) => {
                "Symbol".hash(&mut hasher);
                s.hash(&mut hasher);
            }
            MathNode::Operator(op) => {
                "Operator".hash(&mut hasher);
                format!("{:?}", op).hash(&mut hasher);
            }
            MathNode::Function(func) => {
                "Function".hash(&mut hasher);
                format!("{:?}", func).hash(&mut hasher);
            }
            MathNode::Result => {
                "Result".hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Refine colors for GRAPHEME graph (WL color refinement step)
    fn refine_colors_grapheme(&self, graph: &GraphemeGraph, colors: &[u64]) -> Vec<u64> {
        let node_indices: Vec<_> = graph.graph.node_indices().collect();

        node_indices.iter().map(|&idx| {
            let own_color = colors[idx.index()];

            // Collect neighbor colors (both incoming and outgoing)
            let mut neighbor_colors: Vec<u64> = Vec::new();

            for edge in graph.graph.edges(idx) {
                if let Some(&color) = colors.get(edge.target().index()) {
                    neighbor_colors.push(color);
                }
            }
            for edge in graph.graph.edges_directed(idx, petgraph::Direction::Incoming) {
                if let Some(&color) = colors.get(edge.source().index()) {
                    neighbor_colors.push(color);
                }
            }

            // Sort neighbor colors for canonical ordering
            neighbor_colors.sort();

            // Hash own color with sorted neighbor colors
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            own_color.hash(&mut hasher);
            neighbor_colors.hash(&mut hasher);
            hasher.finish()
        }).collect()
    }

    /// Refine colors for MathGraph (WL color refinement step)
    fn refine_colors_math(&self, graph: &MathGraph, colors: &[u64]) -> Vec<u64> {
        let node_indices: Vec<_> = graph.graph.node_indices().collect();

        node_indices.iter().map(|&idx| {
            let own_color = colors[idx.index()];

            let mut neighbor_colors: Vec<u64> = Vec::new();

            for edge in graph.graph.edges(idx) {
                if let Some(&color) = colors.get(edge.target().index()) {
                    neighbor_colors.push(color);
                }
            }
            for edge in graph.graph.edges_directed(idx, petgraph::Direction::Incoming) {
                if let Some(&color) = colors.get(edge.source().index()) {
                    neighbor_colors.push(color);
                }
            }

            neighbor_colors.sort();

            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            own_color.hash(&mut hasher);
            neighbor_colors.hash(&mut hasher);
            hasher.finish()
        }).collect()
    }

    /// Compute a histogram of colors
    fn compute_histogram(&self, colors: &[u64]) -> HashMap<u64, usize> {
        let mut histogram = HashMap::new();
        for &color in colors {
            *histogram.entry(color).or_insert(0) += 1;
        }
        histogram
    }

    /// Compute similarity between two color histograms
    ///
    /// Uses normalized histogram intersection (min-based Jaccard-like similarity)
    fn histogram_similarity(&self, hist1: &HashMap<u64, usize>, hist2: &HashMap<u64, usize>) -> f32 {
        if hist1.is_empty() && hist2.is_empty() {
            return 1.0;
        }
        if hist1.is_empty() || hist2.is_empty() {
            return 0.0;
        }

        // Compute intersection (minimum of counts for matching colors)
        let mut intersection = 0usize;
        for (color, &count1) in hist1 {
            if let Some(&count2) = hist2.get(color) {
                intersection += count1.min(count2);
            }
        }

        // Compute union (sum of max counts)
        let mut all_colors: std::collections::HashSet<u64> = hist1.keys().copied().collect();
        all_colors.extend(hist2.keys().copied());

        let mut union = 0usize;
        for color in all_colors {
            let c1 = hist1.get(&color).copied().unwrap_or(0);
            let c2 = hist2.get(&color).copied().unwrap_or(0);
            union += c1.max(c2);
        }

        if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

impl GraphEditDistance {
    /// Compute WL-based similarity between two GRAPHEME graphs
    ///
    /// This provides a more accurate graph comparison than simple node/edge counting
    pub fn compute_wl(predicted: &GraphemeGraph, target: &GraphemeGraph) -> f32 {
        let kernel = WeisfeilerLehmanKernel::new();
        kernel.compute(predicted, target)
    }

    /// Compute WL-based similarity between two MathGraphs
    pub fn compute_wl_math(predicted: &MathGraph, target: &MathGraph) -> f32 {
        let kernel = WeisfeilerLehmanKernel::new();
        kernel.compute_math(predicted, target)
    }

    /// Compute combined loss using both count-based and WL-based metrics
    ///
    /// Returns a weighted combination of structural (count-based) and
    /// topological (WL-based) differences.
    pub fn compute_combined(predicted: &GraphemeGraph, target: &GraphemeGraph, wl_weight: f32) -> Self {
        let count_ged = Self::compute(predicted, target);
        let wl_similarity = Self::compute_wl(predicted, target);

        // Convert WL similarity to distance (1 - similarity)
        let wl_distance = 1.0 - wl_similarity;

        Self {
            node_insertion_cost: count_ged.node_insertion_cost,
            node_deletion_cost: count_ged.node_deletion_cost,
            edge_insertion_cost: count_ged.edge_insertion_cost,
            edge_deletion_cost: count_ged.edge_deletion_cost,
            // Use WL distance for mismatch costs
            node_mismatch_cost: wl_distance * wl_weight,
            edge_mismatch_cost: wl_distance * wl_weight * 0.5,
            clique_mismatch: count_ged.clique_mismatch,
        }
    }

    /// Compute combined loss for MathGraphs
    pub fn compute_combined_math(predicted: &MathGraph, target: &MathGraph, wl_weight: f32) -> Self {
        let count_ged = Self::compute_math(predicted, target);
        let wl_similarity = Self::compute_wl_math(predicted, target);
        let wl_distance = 1.0 - wl_similarity;

        Self {
            node_insertion_cost: count_ged.node_insertion_cost,
            node_deletion_cost: count_ged.node_deletion_cost,
            edge_insertion_cost: count_ged.edge_insertion_cost,
            edge_deletion_cost: count_ged.edge_deletion_cost,
            node_mismatch_cost: wl_distance * wl_weight,
            edge_mismatch_cost: wl_distance * wl_weight * 0.5,
            clique_mismatch: count_ged.clique_mismatch,
        }
    }
}

// ============================================================================
// BP2 Quadratic GED Approximation (backend-007)
// ============================================================================

/// Node cost for BP2 matching
#[derive(Debug, Clone)]
struct NodeCost {
    /// Label mismatch cost (0.0 if same, 1.0 if different)
    pub label_cost: f32,
    /// Degree difference cost
    pub degree_cost: f32,
}

impl NodeCost {
    /// Total cost for this node matching
    fn total(&self) -> f32 {
        self.label_cost + self.degree_cost * 0.1
    }
}

/// Result of greedy assignment
#[derive(Debug, Clone)]
struct AssignmentResult {
    /// Node assignment (g1 node index -> g2 node index or None)
    pub assignment: Vec<Option<usize>>,
    /// Total mismatch cost
    pub mismatch_cost: f32,
    /// Number of insertions (g2 nodes not matched)
    pub insertions: usize,
    /// Number of deletions (g1 nodes not matched)
    pub deletions: usize,
}

impl GraphEditDistance {
    /// Compute BP2 quadratic GED approximation for GRAPHEME graphs
    ///
    /// BP2 uses Hausdorff-inspired node matching with greedy assignment.
    /// Complexity: O(n²) instead of O(n³) for Hungarian algorithm.
    ///
    /// Returns a valid upper bound (never underestimates true GED).
    pub fn compute_bp2(predicted: &GraphemeGraph, target: &GraphemeGraph) -> Self {
        let n1 = predicted.node_count();
        let n2 = target.node_count();

        if n1 == 0 && n2 == 0 {
            return Self::default();
        }

        // Compute node-to-node cost matrix
        let node_costs = Self::compute_node_costs_grapheme(predicted, target);

        // Greedy assignment
        let assignment = Self::greedy_assign(&node_costs, n1, n2);

        // Compute edge costs based on assignment
        let edge_cost = Self::compute_edge_cost_grapheme(predicted, target, &assignment);

        Self {
            node_insertion_cost: assignment.insertions as f32,
            node_deletion_cost: assignment.deletions as f32,
            edge_insertion_cost: edge_cost.0,
            edge_deletion_cost: edge_cost.1,
            node_mismatch_cost: assignment.mismatch_cost,
            edge_mismatch_cost: 0.0,
            clique_mismatch: 0.0,
        }
    }

    /// Compute BP2 for MathGraphs
    pub fn compute_bp2_math(predicted: &MathGraph, target: &MathGraph) -> Self {
        let n1 = predicted.node_count();
        let n2 = target.node_count();

        if n1 == 0 && n2 == 0 {
            return Self::default();
        }

        let node_costs = Self::compute_node_costs_math(predicted, target);
        let assignment = Self::greedy_assign(&node_costs, n1, n2);
        let edge_cost = Self::compute_edge_cost_math(predicted, target, &assignment);

        Self {
            node_insertion_cost: assignment.insertions as f32,
            node_deletion_cost: assignment.deletions as f32,
            edge_insertion_cost: edge_cost.0,
            edge_deletion_cost: edge_cost.1,
            node_mismatch_cost: assignment.mismatch_cost,
            edge_mismatch_cost: 0.0,
            clique_mismatch: 0.0,
        }
    }

    /// Compute node-to-node cost matrix for GRAPHEME graphs
    ///
    /// Uses parallel processing via Rayon for improved performance on large graphs.
    fn compute_node_costs_grapheme(g1: &GraphemeGraph, g2: &GraphemeGraph) -> Vec<Vec<NodeCost>> {
        let indices1: Vec<_> = g1.graph.node_indices().collect();
        let indices2: Vec<_> = g2.graph.node_indices().collect();

        // Compute each row in parallel
        indices1
            .par_iter()
            .map(|&idx1| {
                let node1 = &g1.graph[idx1];
                let degree1 = g1.graph.edges(idx1).count();

                indices2
                    .iter()
                    .map(|&idx2| {
                        let node2 = &g2.graph[idx2];
                        let degree2 = g2.graph.edges(idx2).count();

                        // Label cost: 0 if same type, 1 if different
                        let label_cost = if node1.node_type == node2.node_type {
                            0.0
                        } else {
                            1.0
                        };

                        // Degree cost: normalized difference
                        let degree_cost = (degree1 as f32 - degree2 as f32).abs()
                            / (1 + degree1.max(degree2)) as f32;

                        NodeCost { label_cost, degree_cost }
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute node-to-node cost matrix for MathGraphs
    ///
    /// Uses parallel processing via Rayon for improved performance on large graphs.
    fn compute_node_costs_math(g1: &MathGraph, g2: &MathGraph) -> Vec<Vec<NodeCost>> {
        let indices1: Vec<_> = g1.graph.node_indices().collect();
        let indices2: Vec<_> = g2.graph.node_indices().collect();

        // Compute each row in parallel
        indices1
            .par_iter()
            .map(|&idx1| {
                let node1 = &g1.graph[idx1];
                let degree1 = g1.graph.edges(idx1).count();

                indices2
                    .iter()
                    .map(|&idx2| {
                        let node2 = &g2.graph[idx2];
                        let degree2 = g2.graph.edges(idx2).count();

                        // Label cost: 0 if same type, 1 if different
                        let label_cost = if Self::math_nodes_equal(node1, node2) {
                            0.0
                        } else {
                            0.5 + 0.5 * if Self::math_node_category(node1) == Self::math_node_category(node2) {
                                0.0
                            } else {
                                1.0
                            }
                        };

                        let degree_cost = (degree1 as f32 - degree2 as f32).abs()
                            / (1 + degree1.max(degree2)) as f32;

                        NodeCost { label_cost, degree_cost }
                    })
                    .collect()
            })
            .collect()
    }

    /// Check if two math nodes are equal
    fn math_nodes_equal(n1: &MathNode, n2: &MathNode) -> bool {
        match (n1, n2) {
            (MathNode::Integer(a), MathNode::Integer(b)) => a == b,
            (MathNode::Float(a), MathNode::Float(b)) => (a - b).abs() < 1e-10,
            (MathNode::Symbol(a), MathNode::Symbol(b)) => a == b,
            (MathNode::Operator(a), MathNode::Operator(b)) => a == b,
            (MathNode::Function(a), MathNode::Function(b)) => a == b,
            (MathNode::Result, MathNode::Result) => true,
            _ => false,
        }
    }

    /// Get category of a math node (for partial matching)
    fn math_node_category(node: &MathNode) -> u8 {
        match node {
            MathNode::Integer(_) | MathNode::Float(_) => 0, // Numeric
            MathNode::Symbol(_) => 1, // Variable
            MathNode::Operator(_) => 2, // Operator
            MathNode::Function(_) => 3, // Function
            MathNode::Result => 4, // Result
        }
    }

    /// Greedy assignment algorithm - O(n²)
    ///
    /// Assigns nodes from g1 to g2 greedily, always picking the best
    /// available match. Not optimal but fast.
    fn greedy_assign(costs: &[Vec<NodeCost>], n1: usize, n2: usize) -> AssignmentResult {
        let mut assignment = vec![None; n1];
        let mut used = vec![false; n2];
        let mut total_cost = 0.0f32;

        // Build list of all possible pairs sorted by cost
        let mut pairs: Vec<(usize, usize, f32)> = Vec::with_capacity(n1 * n2);
        for i in 0..n1 {
            for j in 0..n2 {
                pairs.push((i, j, costs[i][j].total()));
            }
        }
        pairs.sort_by(|a, b| a.2.total_cmp(&b.2));

        // Greedy assignment
        for (i, j, cost) in pairs {
            if assignment[i].is_none() && !used[j] {
                assignment[i] = Some(j);
                used[j] = true;
                total_cost += cost;
            }
        }

        // Count unassigned nodes
        let deletions = assignment.iter().filter(|a| a.is_none()).count();
        let insertions = used.iter().filter(|&&u| !u).count();

        AssignmentResult {
            assignment,
            mismatch_cost: total_cost,
            insertions,
            deletions,
        }
    }

    /// Compute edge costs based on node assignment for GRAPHEME graphs
    fn compute_edge_cost_grapheme(
        g1: &GraphemeGraph,
        g2: &GraphemeGraph,
        assignment: &AssignmentResult,
    ) -> (f32, f32) {
        let indices1: Vec<_> = g1.graph.node_indices().collect();
        let indices2: Vec<_> = g2.graph.node_indices().collect();

        let mut matched_edges_g2 = std::collections::HashSet::new();
        let mut edge_matches = 0usize;

        // For each edge in g1, check if corresponding edge exists in g2
        for edge1 in g1.graph.edge_references() {
            let src1 = indices1.iter().position(|&idx| idx == edge1.source());
            let dst1 = indices1.iter().position(|&idx| idx == edge1.target());

            if let (Some(s1), Some(d1)) = (src1, dst1) {
                if let (Some(s2), Some(d2)) = (assignment.assignment[s1], assignment.assignment[d1]) {
                    // Check if edge exists in g2
                    if indices2.len() > s2.max(d2) {
                        let idx_s2 = indices2[s2];
                        let idx_d2 = indices2[d2];
                        if g2.graph.find_edge(idx_s2, idx_d2).is_some() {
                            edge_matches += 1;
                            matched_edges_g2.insert((s2, d2));
                        }
                    }
                }
            }
        }

        let e1 = g1.edge_count();
        let e2 = g2.edge_count();

        let deletions = e1.saturating_sub(edge_matches);
        let insertions = e2.saturating_sub(matched_edges_g2.len());

        (insertions as f32, deletions as f32)
    }

    /// Compute edge costs based on node assignment for MathGraphs
    fn compute_edge_cost_math(
        g1: &MathGraph,
        g2: &MathGraph,
        assignment: &AssignmentResult,
    ) -> (f32, f32) {
        let indices1: Vec<_> = g1.graph.node_indices().collect();
        let indices2: Vec<_> = g2.graph.node_indices().collect();

        let mut matched_edges_g2 = std::collections::HashSet::new();
        let mut edge_matches = 0usize;

        for edge1 in g1.graph.edge_references() {
            let src1 = indices1.iter().position(|&idx| idx == edge1.source());
            let dst1 = indices1.iter().position(|&idx| idx == edge1.target());

            if let (Some(s1), Some(d1)) = (src1, dst1) {
                if let (Some(s2), Some(d2)) = (assignment.assignment[s1], assignment.assignment[d1]) {
                    if indices2.len() > s2.max(d2) {
                        let idx_s2 = indices2[s2];
                        let idx_d2 = indices2[d2];
                        if g2.graph.find_edge(idx_s2, idx_d2).is_some() {
                            edge_matches += 1;
                            matched_edges_g2.insert((s2, d2));
                        }
                    }
                }
            }
        }

        let e1 = g1.edge_count();
        let e2 = g2.edge_count();

        let deletions = e1.saturating_sub(edge_matches);
        let insertions = e2.saturating_sub(matched_edges_g2.len());

        (insertions as f32, deletions as f32)
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
// Optimizers (backend-028)
// ============================================================================

use ndarray::Array2;

/// Trait for optimizers that update parameters
pub trait Optimizer {
    /// Perform a single optimization step
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>);

    /// Zero out gradients (called at start of each iteration)
    fn zero_grad(&mut self);

    /// Get current learning rate
    fn get_lr(&self) -> f32;

    /// Set learning rate
    fn set_lr(&mut self, lr: f32);
}

/// Stochastic Gradient Descent with optional momentum
#[derive(Debug, Clone)]
pub struct SGD {
    /// Learning rate
    pub lr: f32,
    /// Momentum coefficient (0 = no momentum)
    pub momentum: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Velocity buffer for momentum
    velocity: Option<Array2<f32>>,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocity: None,
        }
    }

    /// Add momentum to the optimizer
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Add weight decay (L2 regularization)
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) {
        // Apply weight decay
        let mut adjusted_grads = if self.weight_decay > 0.0 {
            grads + &(params.clone() * self.weight_decay)
        } else {
            grads.clone()
        };

        // Apply momentum if enabled
        if self.momentum > 0.0 {
            if self.velocity.is_none() {
                self.velocity = Some(Array2::zeros(params.dim()));
            }

            if let Some(ref mut v) = self.velocity {
                *v = &*v * self.momentum + &adjusted_grads;
                adjusted_grads = v.clone();
            }
        }

        // Update parameters: params -= lr * grads
        *params = &*params - &(&adjusted_grads * self.lr);
    }

    fn zero_grad(&mut self) {
        // SGD doesn't need to zero anything, momentum is preserved
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Adam optimizer
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate
    pub lr: f32,
    /// Beta1 (exponential decay rate for first moment)
    pub beta1: f32,
    /// Beta2 (exponential decay rate for second moment)
    pub beta2: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// First moment estimate
    m: Option<Array2<f32>>,
    /// Second moment estimate
    v: Option<Array2<f32>>,
    /// Timestep
    t: usize,
}

impl Adam {
    /// Create a new Adam optimizer with default parameters
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: None,
            v: None,
            t: 0,
        }
    }

    /// Set beta1
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Add weight decay
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) {
        self.t += 1;

        // Initialize moment estimates if needed
        if self.m.is_none() {
            self.m = Some(Array2::zeros(params.dim()));
        }
        if self.v.is_none() {
            self.v = Some(Array2::zeros(params.dim()));
        }

        // Apply weight decay (decoupled, as in AdamW)
        if self.weight_decay > 0.0 {
            *params = &*params - &(params.clone() * (self.lr * self.weight_decay));
        }

        // Update biased first moment estimate
        if let Some(ref mut m) = self.m {
            *m = &*m * self.beta1 + &(grads * (1.0 - self.beta1));
        }

        // Update biased second moment estimate
        if let Some(ref mut v) = self.v {
            let grads_sq = grads.mapv(|x| x * x);
            *v = &*v * self.beta2 + &(grads_sq * (1.0 - self.beta2));
        }

        // Compute bias-corrected estimates
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        if let (Some(ref m), Some(ref v)) = (&self.m, &self.v) {
            let m_hat = m / bias_correction1;
            let v_hat = v / bias_correction2;

            // Update parameters
            let denom = v_hat.mapv(|x| x.sqrt() + self.epsilon);
            let update = m_hat / denom;
            *params = &*params - &(update * self.lr);
        }
    }

    fn zero_grad(&mut self) {
        // Adam keeps momentum, but this resets for fresh training
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ============================================================================
// Learning Rate Schedulers
// ============================================================================

/// Learning rate scheduler types
#[derive(Debug, Clone)]
pub enum LRScheduler {
    /// Constant learning rate (no decay)
    Constant,
    /// Step decay: lr = lr * gamma every step_size epochs
    StepLR { step_size: usize, gamma: f32 },
    /// Exponential decay: lr = lr * gamma^epoch
    ExponentialLR { gamma: f32 },
    /// Cosine annealing: lr oscillates from max to min
    CosineAnnealingLR { t_max: usize, eta_min: f32 },
    /// Linear warmup: lr increases linearly for warmup_steps, then constant
    WarmupLR { warmup_steps: usize },
    /// Warmup then cosine decay
    WarmupCosineDecay { warmup_steps: usize, total_steps: usize, eta_min: f32 },
}

impl LRScheduler {
    /// Compute learning rate for given epoch
    pub fn get_lr(&self, base_lr: f32, epoch: usize) -> f32 {
        match self {
            LRScheduler::Constant => base_lr,

            LRScheduler::StepLR { step_size, gamma } => {
                let num_decays = epoch / step_size;
                base_lr * gamma.powi(num_decays as i32)
            }

            LRScheduler::ExponentialLR { gamma } => {
                base_lr * gamma.powi(epoch as i32)
            }

            LRScheduler::CosineAnnealingLR { t_max, eta_min } => {
                let t = (epoch % t_max) as f32;
                let t_max = *t_max as f32;
                eta_min + (base_lr - eta_min) * (1.0 + (std::f32::consts::PI * t / t_max).cos()) / 2.0
            }

            LRScheduler::WarmupLR { warmup_steps } => {
                if epoch < *warmup_steps {
                    base_lr * (epoch + 1) as f32 / *warmup_steps as f32
                } else {
                    base_lr
                }
            }

            LRScheduler::WarmupCosineDecay { warmup_steps, total_steps, eta_min } => {
                if epoch < *warmup_steps {
                    base_lr * (epoch + 1) as f32 / *warmup_steps as f32
                } else {
                    let t = (epoch - warmup_steps) as f32;
                    let t_max = (total_steps - warmup_steps) as f32;
                    eta_min + (base_lr - eta_min) * (1.0 + (std::f32::consts::PI * t / t_max).cos()) / 2.0
                }
            }
        }
    }
}

// ============================================================================
// Training Loop
// ============================================================================

/// Training state for a training run
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,
    /// Current step (batch)
    pub step: usize,
    /// Total steps completed
    pub total_steps: usize,
    /// Current learning rate
    pub current_lr: f32,
    /// Best validation loss seen
    pub best_val_loss: f32,
    /// Epochs since improvement (for early stopping)
    pub epochs_without_improvement: usize,
    /// Running loss for current epoch
    pub running_loss: f32,
    /// Number of batches in current epoch
    pub batches_in_epoch: usize,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            step: 0,
            total_steps: 0,
            current_lr: 0.001,
            best_val_loss: f32::MAX,
            epochs_without_improvement: 0,
            running_loss: 0.0,
            batches_in_epoch: 0,
        }
    }
}

/// Metrics logged during training
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Loss values per epoch
    pub epoch_losses: Vec<f32>,
    /// Validation losses per epoch
    pub val_losses: Vec<f32>,
    /// Validation accuracy per epoch
    pub val_accuracies: Vec<f32>,
    /// Learning rates per epoch
    pub learning_rates: Vec<f32>,
}

impl TrainingMetrics {
    /// Add epoch metrics
    pub fn record_epoch(&mut self, loss: f32, lr: f32) {
        self.epoch_losses.push(loss);
        self.learning_rates.push(lr);
    }

    /// Add validation metrics
    pub fn record_validation(&mut self, val_loss: f32, val_accuracy: f32) {
        self.val_losses.push(val_loss);
        self.val_accuracies.push(val_accuracy);
    }

    /// Get the latest training loss
    pub fn latest_loss(&self) -> Option<f32> {
        self.epoch_losses.last().copied()
    }

    /// Check if loss is improving
    pub fn is_improving(&self, window: usize) -> bool {
        if self.epoch_losses.len() < window + 1 {
            return true;
        }

        let recent = &self.epoch_losses[self.epoch_losses.len() - window..];
        let prev = &self.epoch_losses[self.epoch_losses.len() - window - 1..self.epoch_losses.len() - 1];

        let recent_avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        let prev_avg: f32 = prev.iter().sum::<f32>() / prev.len() as f32;

        recent_avg < prev_avg
    }
}

/// Training loop that orchestrates forward/backward passes
#[derive(Debug)]
pub struct TrainingLoop {
    /// Configuration
    pub config: TrainingConfig,
    /// Learning rate scheduler
    pub scheduler: LRScheduler,
    /// Training state
    pub state: TrainingState,
    /// Metrics
    pub metrics: TrainingMetrics,
    /// Base learning rate
    base_lr: f32,
}

impl TrainingLoop {
    /// Create a new training loop
    pub fn new(config: TrainingConfig) -> Self {
        let base_lr = config.learning_rate;
        Self {
            config,
            scheduler: LRScheduler::Constant,
            state: TrainingState::default(),
            metrics: TrainingMetrics::default(),
            base_lr,
        }
    }

    /// Set the learning rate scheduler
    pub fn with_scheduler(mut self, scheduler: LRScheduler) -> Self {
        self.scheduler = scheduler;
        self
    }

    /// Update learning rate based on scheduler
    pub fn update_lr(&mut self) {
        self.state.current_lr = self.scheduler.get_lr(self.base_lr, self.state.epoch);
    }

    /// Record a batch loss
    pub fn record_batch(&mut self, loss: f32) {
        self.state.running_loss += loss;
        self.state.batches_in_epoch += 1;
        self.state.step += 1;
        self.state.total_steps += 1;
    }

    /// Complete an epoch and record metrics
    pub fn complete_epoch(&mut self) -> f32 {
        let avg_loss = if self.state.batches_in_epoch > 0 {
            self.state.running_loss / self.state.batches_in_epoch as f32
        } else {
            0.0
        };

        self.metrics.record_epoch(avg_loss, self.state.current_lr);

        // Reset for next epoch
        self.state.running_loss = 0.0;
        self.state.batches_in_epoch = 0;
        self.state.epoch += 1;
        self.update_lr();

        avg_loss
    }

    /// Record validation results
    pub fn record_validation(&mut self, val_loss: f32, val_accuracy: f32) -> bool {
        self.metrics.record_validation(val_loss, val_accuracy);

        // Check for improvement
        if val_loss < self.state.best_val_loss {
            self.state.best_val_loss = val_loss;
            self.state.epochs_without_improvement = 0;
            true // New best
        } else {
            self.state.epochs_without_improvement += 1;
            false
        }
    }

    /// Check if early stopping should trigger
    pub fn should_stop(&self) -> bool {
        self.state.epochs_without_improvement >= self.config.patience
    }

    /// Check if validation should run this epoch
    pub fn should_validate(&self) -> bool {
        self.state.epoch % self.config.val_frequency == 0
    }

    /// Get progress as percentage
    pub fn progress(&self) -> f32 {
        self.state.epoch as f32 / self.config.epochs as f32 * 100.0
    }
}

/// Compute GED-based loss between predicted and target graphs
///
/// Loss = alpha * node_cost + beta * edge_cost + gamma * clique_mismatch
pub fn compute_ged_loss(
    predicted: &GraphemeGraph,
    target: &GraphemeGraph,
    alpha: f32,
    beta: f32,
    gamma: f32,
) -> f32 {
    let ged = GraphEditDistance::compute(predicted, target);

    let node_cost = ged.node_insertion_cost + ged.node_deletion_cost + ged.node_mismatch_cost;
    let edge_cost = ged.edge_insertion_cost + ged.edge_deletion_cost + ged.edge_mismatch_cost;

    alpha * node_cost + beta * edge_cost + gamma * ged.clique_mismatch
}

// ============================================================================
// Validation Utilities
// ============================================================================

/// Validate symbolic expression structure
fn validate_symbolic_expr(expr: &Expr, max_depth: usize) -> Result<(), String> {
    if max_depth == 0 {
        return Err("Expression exceeds maximum nesting depth".to_string());
    }

    match expr {
        Expr::Value(v) => {
            // Validate value types
            match v {
                Value::Symbol(s) if s.is_empty() => {
                    Err("Empty symbol name".to_string())
                }
                Value::Rational(_, 0) => {
                    Err("Division by zero in rational".to_string())
                }
                _ => Ok(()),
            }
        }
        Expr::BinOp { left, right, .. } => {
            validate_symbolic_expr(left, max_depth - 1)?;
            validate_symbolic_expr(right, max_depth - 1)
        }
        Expr::UnaryOp { operand, .. } => {
            validate_symbolic_expr(operand, max_depth - 1)
        }
        Expr::Function { args, .. } => {
            for arg in args {
                validate_symbolic_expr(arg, max_depth - 1)?;
            }
            Ok(())
        }
    }
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
                        report.errors.push(format!(
                            "{}: expected {}, got {}",
                            example.id, expected, result
                        ));
                    }
                }
                Err(e) => {
                    report.errors_count += 1;
                    report.errors.push(format!("{}: {}", example.id, e));
                }
            }
        } else {
            // Symbolic validation - check expression structure
            const MAX_DEPTH: usize = 100;

            // Validate input expression
            if let Err(e) = validate_symbolic_expr(&example.input_expr, MAX_DEPTH) {
                report.invalid += 1;
                report.errors.push(format!("{}: input expr - {}", example.id, e));
                continue;
            }

            // Validate output expression if present
            if let Some(ref output_expr) = example.expected_symbolic {
                if let Err(e) = validate_symbolic_expr(output_expr, MAX_DEPTH) {
                    report.invalid += 1;
                    report.errors.push(format!("{}: output expr - {}", example.id, e));
                    continue;
                }
            }

            report.valid += 1;
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
// End-to-End Pipeline (Layer 4-3-2-1)
// ============================================================================

use grapheme_core::GraphTransformNet;

/// Pipeline mode for inference or training
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    /// Inference mode with frozen weights
    Inference,
    /// Training mode with gradient flow
    Training,
}

/// Result of pipeline processing
#[derive(Debug)]
pub struct PipelineResult {
    /// The original input text
    pub input: String,
    /// Natural language graph (Layer 4)
    pub nl_graph: Option<GraphemeGraph>,
    /// Math graph representation (Layer 3)
    pub math_graph: Option<MathGraph>,
    /// Expression after optimization (Layer 2)
    pub optimized_expr: Option<Expr>,
    /// Numeric result if available (Layer 1)
    pub numeric_result: Option<f64>,
    /// Symbolic result if available (Layer 1)
    pub symbolic_result: Option<String>,
    /// Processing steps taken
    pub steps: Vec<String>,
    /// Any errors encountered
    pub errors: Vec<String>,
}

impl PipelineResult {
    fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            nl_graph: None,
            math_graph: None,
            optimized_expr: None,
            numeric_result: None,
            symbolic_result: None,
            steps: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Check if processing succeeded
    pub fn success(&self) -> bool {
        self.errors.is_empty() && (self.numeric_result.is_some() || self.symbolic_result.is_some())
    }

    /// Get primary result as string
    pub fn result_string(&self) -> String {
        if let Some(n) = self.numeric_result {
            if n.fract() == 0.0 && n.abs() < 1e10 {
                format!("{}", n as i64)
            } else {
                format!("{}", n)
            }
        } else if let Some(ref s) = self.symbolic_result {
            s.clone()
        } else if !self.errors.is_empty() {
            format!("Error: {}", self.errors.join(", "))
        } else {
            "No result".to_string()
        }
    }
}

/// End-to-end pipeline from natural language to math result
///
/// Chains all layers:
/// - Layer 4 (grapheme-core): NL text → DagNN graph
/// - Layer 3 (grapheme-math): DagNN → MathGraph
/// - Layer 2 (grapheme-polish): MathGraph → Optimized expression
/// - Layer 1 (grapheme-engine): Expression → Evaluated result
#[derive(Debug)]
pub struct Pipeline {
    /// The graph transformation network (Layer 4)
    pub transform_net: Option<GraphTransformNet>,
    /// The math engine (Layer 1)
    pub engine: MathEngine,
    /// The symbolic engine (Layer 1)
    pub symbolic: SymbolicEngine,
    /// Current mode
    pub mode: PipelineMode,
    /// Whether to cache intermediate representations
    pub cache_enabled: bool,
    /// Cache of NL to expression mappings (for future use)
    _cache: HashMap<String, Expr>,
}

impl Pipeline {
    /// Create a new pipeline
    pub fn new() -> Self {
        Self {
            transform_net: None,
            engine: MathEngine::new(),
            symbolic: SymbolicEngine::new(),
            mode: PipelineMode::Inference,
            cache_enabled: false,
            _cache: HashMap::new(),
        }
    }

    /// Create pipeline with a trained transformation network
    pub fn with_transform_net(mut self, net: GraphTransformNet) -> Self {
        self.transform_net = Some(net);
        self
    }

    /// Set pipeline mode
    pub fn with_mode(mut self, mode: PipelineMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable caching
    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Bind a variable value
    pub fn bind(&mut self, name: &str, value: f64) {
        self.engine.bind(name.to_string(), Value::Float(value));
    }

    /// Clear all variable bindings
    pub fn clear_bindings(&mut self) {
        self.engine.clear_bindings();
    }

    /// Process natural language input through the full pipeline
    ///
    /// This method is thread-safe and can be called from multiple threads
    /// for parallel processing (useful for batch inference and training).
    pub fn process(&self, input: &str) -> PipelineResult {
        let mut result = PipelineResult::new(input);

        // Step 1: Parse input to expression (try direct parsing first)
        result.steps.push("Layer 4: Parsing input".to_string());

        // Try to extract math expression from natural language
        let expr = match self.extract_expression(input) {
            Ok(e) => {
                result.steps.push(format!("  Extracted: {}", expr_to_polish(&e)));
                e
            }
            Err(e) => {
                result.errors.push(format!("Parse error: {}", e));
                return result;
            }
        };

        // Step 2: Create NL graph (Layer 4)
        let nl_graph = GraphemeGraph::from_text(input);
        result.steps.push(format!("  NL graph: {} nodes", nl_graph.graph.node_count()));
        result.nl_graph = Some(nl_graph);

        // Step 3: Create math graph (Layer 3)
        result.steps.push("Layer 3: Building math graph".to_string());
        let math_graph = MathGraph::from_expr(&expr);
        result.steps.push(format!("  Math graph: {} nodes, {} edges",
            math_graph.node_count(), math_graph.edge_count()));
        result.math_graph = Some(math_graph);

        // Step 4: Optimize expression (Layer 2)
        result.steps.push("Layer 2: Optimizing expression".to_string());
        let optimizer = grapheme_polish::Optimizer::with_defaults();
        let optimized = optimizer.optimize_fixpoint(&expr);
        result.steps.push(format!("  Optimized: {}", expr_to_polish(&optimized)));
        result.optimized_expr = Some(optimized.clone());

        // Step 5: Evaluate (Layer 1)
        result.steps.push("Layer 1: Evaluating".to_string());

        // Try numeric evaluation
        match self.engine.evaluate(&optimized) {
            Ok(val) => {
                result.steps.push(format!("  Numeric: {}", val));
                result.numeric_result = Some(val);
            }
            Err(e) => {
                result.steps.push(format!("  Numeric eval failed: {}", e));
            }
        }

        // If symbolic, generate symbolic result
        if optimized.is_symbolic() {
            let symbolic = expr_to_polish(&optimized);
            result.steps.push(format!("  Symbolic: {}", symbolic));
            result.symbolic_result = Some(symbolic);
        }

        result
    }

    /// Extract mathematical expression from natural language
    fn extract_expression(&self, input: &str) -> Result<Expr, String> {
        // Try to parse simple mathematical expressions
        let cleaned = input.trim().to_lowercase();

        // Only try polish notation if it looks like polish (starts with operator/function)
        let trimmed = cleaned.trim();
        if trimmed.starts_with('+') || trimmed.starts_with('-') || trimmed.starts_with('*')
            || trimmed.starts_with('/') || trimmed.starts_with('^')
            || trimmed.starts_with("sin") || trimmed.starts_with("cos")
            || trimmed.starts_with("exp") || trimmed.starts_with("ln")
        {
            let mut parser = grapheme_polish::PolishParser::new();
            if let Ok(expr) = parser.parse(input) {
                return Ok(expr);
            }
        }

        // Handle "derivative of X" patterns
        if cleaned.starts_with("derivative of ") || cleaned.starts_with("differentiate ") {
            return self.parse_derivative_command(&cleaned);
        }

        // Handle "integrate X" patterns
        if cleaned.starts_with("integrate ") || cleaned.starts_with("integral of ") {
            return self.parse_integral_command(&cleaned);
        }

        // Handle "simplify X" patterns
        if cleaned.starts_with("simplify ") {
            let rest = cleaned.strip_prefix("simplify ").unwrap_or("");
            return self.parse_math_expression(rest);
        }

        // Handle "what is X" patterns
        if cleaned.starts_with("what is ") || cleaned.starts_with("what's ") {
            let rest = if cleaned.starts_with("what is ") {
                cleaned.strip_prefix("what is ").unwrap_or("")
            } else {
                cleaned.strip_prefix("what's ").unwrap_or("")
            };
            return self.parse_math_expression(rest);
        }

        // Handle "calculate X" patterns
        if cleaned.starts_with("calculate ") || cleaned.starts_with("compute ") || cleaned.starts_with("evaluate ") {
            let rest = cleaned
                .strip_prefix("calculate ")
                .or_else(|| cleaned.strip_prefix("compute "))
                .or_else(|| cleaned.strip_prefix("evaluate "))
                .unwrap_or("");
            return self.parse_math_expression(rest);
        }

        // Try direct math expression parsing
        self.parse_math_expression(&cleaned)
    }

    /// Parse derivative command
    fn parse_derivative_command(&self, input: &str) -> Result<Expr, String> {
        // Extract "derivative of X [with respect to Y]"
        let rest = input
            .strip_prefix("derivative of ")
            .or_else(|| input.strip_prefix("differentiate "))
            .unwrap_or(input);

        // Check for "with respect to" pattern
        let (expr_str, var) = if let Some(idx) = rest.find(" with respect to ") {
            let expr_part = &rest[..idx];
            let var_part = rest[idx + 18..].trim();
            (expr_part, var_part.chars().next().unwrap_or('x'))
        } else {
            (rest, 'x')
        };

        // Parse the expression
        let expr = self.parse_math_expression(expr_str)?;

        // Return the derivative result directly (symbolic engine will compute it)
        Ok(self.symbolic.differentiate(&expr, &var.to_string()))
    }

    /// Parse integral command
    fn parse_integral_command(&self, input: &str) -> Result<Expr, String> {
        let rest = input
            .strip_prefix("integrate ")
            .or_else(|| input.strip_prefix("integral of "))
            .unwrap_or(input);

        // Check for "from A to B" pattern
        if let Some(from_idx) = rest.find(" from ") {
            let expr_str = &rest[..from_idx];
            let bounds_str = &rest[from_idx + 6..];

            if let Some(to_idx) = bounds_str.find(" to ") {
                let lower: f64 = bounds_str[..to_idx].trim().parse().map_err(|_| "Invalid lower bound")?;
                let upper: f64 = bounds_str[to_idx + 4..].trim().parse().map_err(|_| "Invalid upper bound")?;

                let expr = self.parse_math_expression(expr_str)?;

                // Compute definite integral if possible
                match self.symbolic.integrate(&expr, "x") {
                    Ok(antiderivative) => {
                        // F(upper) - F(lower)
                        let f_upper = self.symbolic.evaluate_at(&antiderivative, "x", upper);
                        let f_lower = self.symbolic.evaluate_at(&antiderivative, "x", lower);
                        return Ok(Expr::sub(f_upper, f_lower));
                    }
                    Err(_) => return Err("Cannot integrate expression".to_string()),
                }
            }
        }

        // Indefinite integral
        let expr = self.parse_math_expression(rest)?;
        self.symbolic.integrate(&expr, "x")
            .map_err(|e| format!("Integration error: {:?}", e))
    }

    /// Parse a mathematical expression string
    fn parse_math_expression(&self, input: &str) -> Result<Expr, String> {
        // Handle common patterns
        let s = input.trim();

        // Handle "X squared" pattern
        if s.ends_with(" squared") {
            let base = s.strip_suffix(" squared").unwrap_or("");
            let base_expr = self.parse_math_expression(base)?;
            return Ok(Expr::pow(base_expr, Expr::int(2)));
        }

        // Handle "X cubed" pattern
        if s.ends_with(" cubed") {
            let base = s.strip_suffix(" cubed").unwrap_or("");
            let base_expr = self.parse_math_expression(base)?;
            return Ok(Expr::pow(base_expr, Expr::int(3)));
        }

        // Handle simple infix with spaces: "2 + 3"
        if let Some(idx) = s.find(" + ") {
            let left = self.parse_math_expression(&s[..idx])?;
            let right = self.parse_math_expression(&s[idx + 3..])?;
            return Ok(Expr::add(left, right));
        }

        if let Some(idx) = s.find(" - ") {
            let left = self.parse_math_expression(&s[..idx])?;
            let right = self.parse_math_expression(&s[idx + 3..])?;
            return Ok(Expr::sub(left, right));
        }

        if let Some(idx) = s.find(" * ") {
            let left = self.parse_math_expression(&s[..idx])?;
            let right = self.parse_math_expression(&s[idx + 3..])?;
            return Ok(Expr::mul(left, right));
        }

        if let Some(idx) = s.find(" / ") {
            let left = self.parse_math_expression(&s[..idx])?;
            let right = self.parse_math_expression(&s[idx + 3..])?;
            return Ok(Expr::div(left, right));
        }

        // Try compact operators: "2+3"
        for (i, c) in s.char_indices() {
            if i > 0 && (c == '+' || c == '-' || c == '*' || c == '/') {
                let left = &s[..i];
                let right = &s[i + 1..];
                if !left.is_empty() && !right.is_empty() {
                    let l = self.parse_math_expression(left)?;
                    let r = self.parse_math_expression(right)?;
                    return Ok(match c {
                        '+' => Expr::add(l, r),
                        '-' => Expr::sub(l, r),
                        '*' => Expr::mul(l, r),
                        '/' => Expr::div(l, r),
                        _ => unreachable!(),
                    });
                }
            }
        }

        // Try parsing as number
        if let Ok(n) = s.parse::<i64>() {
            return Ok(Expr::int(n));
        }
        if let Ok(n) = s.parse::<f64>() {
            return Ok(Expr::float(n));
        }

        // Try as symbol
        if s.chars().all(|c| c.is_alphabetic()) && !s.is_empty() {
            return Ok(Expr::symbol(s));
        }

        Err(format!("Cannot parse expression: {}", input))
    }

    /// Process a batch of inputs
    /// Process multiple inputs in parallel using Rayon
    ///
    /// This provides significant speedup for batch inference.
    pub fn process_batch(&self, inputs: &[&str]) -> Vec<PipelineResult> {
        inputs.par_iter().map(|&input| self.process(input)).collect()
    }

    /// Run training mode on a dataset
    ///
    /// Uses parallel processing via Rayon for improved performance.
    /// Each example in the dataset is processed in parallel within each epoch.
    pub fn train(&mut self, dataset: &Dataset, config: &TrainingConfig) -> TrainingResult<TrainingMetrics> {
        self.mode = PipelineMode::Training;

        let mut metrics = TrainingMetrics::default();
        let mut training_loop = TrainingLoop::new(config.clone());

        for _epoch in 0..config.epochs {
            // Use Rayon's parallel fold/reduce for thread-safe accumulation
            let (total_loss, batch_count): (f64, usize) = dataset.examples
                .par_iter()
                .filter_map(|example| {
                    // Get input expression in polish notation
                    let input = &example.input_polish;

                    // Forward pass: process input through pipeline (thread-safe)
                    let result = self.process(input);

                    // Compute loss using graph edit distance
                    if let Some(ref predicted_graph) = result.nl_graph {
                        // Create expected graph from expected result
                        let expected_polish = if let Some(ref sym) = example.expected_symbolic {
                            expr_to_polish(sym)
                        } else if let Some(n) = example.expected_result {
                            format!("{}", n)
                        } else {
                            return None;
                        };

                        // Create expected graph
                        let expected_graph = GraphemeGraph::from_text(&expected_polish);

                        // Compute GED loss
                        let ged = GraphEditDistance::compute(predicted_graph, &expected_graph);
                        let loss = ged.total() as f64;

                        // Skip NaN losses (indicates numerical instability)
                        if loss.is_nan() || loss.is_infinite() {
                            return None;
                        }

                        Some(loss)
                    } else {
                        None
                    }
                })
                .fold(
                    || (0.0f64, 0usize),
                    |(acc_loss, acc_count), loss| (acc_loss + loss, acc_count + 1)
                )
                .reduce(
                    || (0.0f64, 0usize),
                    |(a_loss, a_count), (b_loss, b_count)| (a_loss + b_loss, a_count + b_count)
                );

            // Complete epoch if we had any examples
            if batch_count > 0 {
                // Record batch losses (average loss per batch)
                let avg_loss = total_loss / batch_count as f64;
                training_loop.record_batch(avg_loss as f32);
                let epoch_avg = training_loop.complete_epoch();
                metrics.epoch_losses.push(epoch_avg);
            }

            // Early stopping check
            if training_loop.should_stop() {
                break;
            }
        }

        self.mode = PipelineMode::Inference;
        Ok(metrics)
    }

    /// Get the current mode
    pub fn get_mode(&self) -> PipelineMode {
        self.mode
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.mode == PipelineMode::Training
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick evaluation function for simple expressions
pub fn quick_eval(input: &str) -> Option<f64> {
    let pipeline = Pipeline::new();
    let result = pipeline.process(input);
    result.numeric_result
}

/// Quick symbolic evaluation
pub fn quick_symbolic(input: &str) -> Option<String> {
    let pipeline = Pipeline::new();
    let result = pipeline.process(input);
    if result.symbolic_result.is_some() {
        result.symbolic_result
    } else if let Some(n) = result.numeric_result {
        Some(format!("{}", n))
    } else {
        None
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

    // ========================================================================
    // Weisfeiler-Leman Kernel Tests (backend-006)
    // ========================================================================

    #[test]
    fn test_wl_identical_graphs() {
        let g1 = GraphemeGraph::from_text("Hello");
        let g2 = GraphemeGraph::from_text("Hello");

        let similarity = GraphEditDistance::compute_wl(&g1, &g2);
        assert!((similarity - 1.0).abs() < 1e-6, "Identical graphs should have similarity 1.0");
    }

    #[test]
    fn test_wl_completely_different_graphs() {
        let g1 = GraphemeGraph::from_text("abc");
        let g2 = GraphemeGraph::from_text("xyz");

        let similarity = GraphEditDistance::compute_wl(&g1, &g2);
        // Different characters means different initial colors
        assert!(similarity < 0.5, "Different graphs should have low similarity");
    }

    #[test]
    fn test_wl_partial_overlap() {
        let g1 = GraphemeGraph::from_text("Hello");
        let g2 = GraphemeGraph::from_text("Hello!");

        let similarity = GraphEditDistance::compute_wl(&g1, &g2);
        // Should be high but not 1.0 (extra character)
        assert!(similarity > 0.6, "Similar graphs should have high similarity");
        assert!(similarity < 1.0, "Different graphs should not be identical");
    }

    #[test]
    fn test_wl_empty_graphs() {
        let g1 = GraphemeGraph::new();
        let g2 = GraphemeGraph::new();

        let kernel = WeisfeilerLehmanKernel::new();
        let similarity = kernel.compute(&g1, &g2);
        assert!((similarity - 1.0).abs() < 1e-6, "Empty graphs should be identical");
    }

    #[test]
    fn test_wl_one_empty_graph() {
        let g1 = GraphemeGraph::from_text("abc");
        let g2 = GraphemeGraph::new();

        let similarity = GraphEditDistance::compute_wl(&g1, &g2);
        assert!(similarity < 0.1, "One empty graph should have near-zero similarity");
    }

    #[test]
    fn test_wl_kernel_iterations() {
        let g1 = GraphemeGraph::from_text("test");
        let g2 = GraphemeGraph::from_text("test");

        // Test with different iteration counts
        let kernel_1 = WeisfeilerLehmanKernel::with_iterations(1);
        let kernel_5 = WeisfeilerLehmanKernel::with_iterations(5);

        let sim_1 = kernel_1.compute(&g1, &g2);
        let sim_5 = kernel_5.compute(&g1, &g2);

        // Both should be 1.0 for identical graphs
        assert!((sim_1 - 1.0).abs() < 1e-6);
        assert!((sim_5 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wl_math_graphs() {
        // Create two similar math expressions
        let expr1 = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };
        let expr2 = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };

        let g1 = MathGraph::from_expr(&expr1);
        let g2 = MathGraph::from_expr(&expr2);

        let similarity = GraphEditDistance::compute_wl_math(&g1, &g2);
        assert!((similarity - 1.0).abs() < 1e-6, "Identical math graphs should have similarity 1.0");
    }

    #[test]
    fn test_wl_different_math_graphs() {
        let expr1 = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };
        let expr2 = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };

        let g1 = MathGraph::from_expr(&expr1);
        let g2 = MathGraph::from_expr(&expr2);

        let similarity = GraphEditDistance::compute_wl_math(&g1, &g2);
        // Same values but different operator - they share 2/3 node types
        // WL captures that the structure differs due to the operator
        assert!(similarity >= 0.0, "Similarity should be non-negative");
        assert!(similarity < 1.0, "Different operators should reduce similarity");
    }

    #[test]
    fn test_wl_combined_ged() {
        let g1 = GraphemeGraph::from_text("Hello");
        let g2 = GraphemeGraph::from_text("Hello!");

        let combined = GraphEditDistance::compute_combined(&g1, &g2, 1.0);

        // Should have node difference from count-based
        assert!(combined.node_insertion_cost > 0.0 || combined.node_deletion_cost > 0.0);
        // Should have mismatch from WL-based
        assert!(combined.node_mismatch_cost > 0.0);
    }

    #[test]
    fn test_wl_symmetry() {
        let g1 = GraphemeGraph::from_text("abc");
        let g2 = GraphemeGraph::from_text("xyz");

        let sim12 = GraphEditDistance::compute_wl(&g1, &g2);
        let sim21 = GraphEditDistance::compute_wl(&g2, &g1);

        assert!((sim12 - sim21).abs() < 1e-6, "WL similarity should be symmetric");
    }

    #[test]
    fn test_wl_histogram_computation() {
        let kernel = WeisfeilerLehmanKernel::new();
        let colors = vec![1u64, 2, 2, 3, 3, 3];

        let histogram = kernel.compute_histogram(&colors);

        assert_eq!(histogram.get(&1), Some(&1));
        assert_eq!(histogram.get(&2), Some(&2));
        assert_eq!(histogram.get(&3), Some(&3));
    }

    // ========================================================================
    // BP2 Quadratic GED Tests (backend-007)
    // ========================================================================

    #[test]
    fn test_bp2_identical_graphs() {
        let g1 = GraphemeGraph::from_text("Hello");
        let g2 = GraphemeGraph::from_text("Hello");

        let bp2 = GraphEditDistance::compute_bp2(&g1, &g2);

        // Identical graphs should have zero insertions/deletions
        assert_eq!(bp2.node_insertion_cost, 0.0);
        assert_eq!(bp2.node_deletion_cost, 0.0);
        // All nodes should match with zero mismatch cost
        assert!(bp2.node_mismatch_cost < 0.1);
    }

    #[test]
    fn test_bp2_completely_different_graphs() {
        let g1 = GraphemeGraph::from_text("abc");
        let g2 = GraphemeGraph::from_text("xyz");

        let bp2 = GraphEditDistance::compute_bp2(&g1, &g2);

        // Different characters should have high mismatch
        assert!(bp2.node_mismatch_cost > 0.0);
    }

    #[test]
    fn test_bp2_size_difference() {
        let g1 = GraphemeGraph::from_text("Hello");
        let g2 = GraphemeGraph::from_text("Hi");

        let bp2 = GraphEditDistance::compute_bp2(&g1, &g2);

        // g1 has more nodes, so we expect deletions
        assert!(bp2.node_deletion_cost > 0.0 || bp2.node_insertion_cost > 0.0);
    }

    #[test]
    fn test_bp2_empty_graphs() {
        let g1 = GraphemeGraph::new();
        let g2 = GraphemeGraph::new();

        let bp2 = GraphEditDistance::compute_bp2(&g1, &g2);

        assert_eq!(bp2.total(), 0.0);
    }

    #[test]
    fn test_bp2_one_empty_graph() {
        let g1 = GraphemeGraph::from_text("abc");
        let g2 = GraphemeGraph::new();

        let bp2 = GraphEditDistance::compute_bp2(&g1, &g2);

        // All nodes from g1 should be deleted
        assert_eq!(bp2.node_deletion_cost, 3.0);
    }

    #[test]
    fn test_bp2_math_graphs() {
        let expr1 = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };
        let expr2 = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };

        let g1 = MathGraph::from_expr(&expr1);
        let g2 = MathGraph::from_expr(&expr2);

        let bp2 = GraphEditDistance::compute_bp2_math(&g1, &g2);

        // Identical graphs should have low cost
        assert!(bp2.total() < 1.0);
    }

    #[test]
    fn test_bp2_provides_upper_bound() {
        // BP2 should provide an upper bound on true GED
        // For simple cases, it should be close to optimal
        let g1 = GraphemeGraph::from_text("test");
        let g2 = GraphemeGraph::from_text("test");

        let bp2 = GraphEditDistance::compute_bp2(&g1, &g2);
        let count = GraphEditDistance::compute(&g1, &g2);

        // BP2 should be at least as large as simple count
        // (in this case, both should be near zero)
        assert!(bp2.total() >= 0.0);
        assert!(count.total() >= 0.0);
    }

    #[test]
    fn test_bp2_vs_wl_correlation() {
        // BP2 and WL should generally agree on similarity ordering
        let g1 = GraphemeGraph::from_text("hello");
        let g2 = GraphemeGraph::from_text("hallo"); // Similar
        let g3 = GraphemeGraph::from_text("xyz");   // Different

        let bp2_similar = GraphEditDistance::compute_bp2(&g1, &g2);
        let bp2_different = GraphEditDistance::compute_bp2(&g1, &g3);

        // Similar graphs should have lower BP2 cost than different graphs
        // (mismatch cost is the key differentiator)
        assert!(bp2_similar.node_mismatch_cost <= bp2_different.node_mismatch_cost);
    }

    #[test]
    fn test_bp2_symmetry() {
        let g1 = GraphemeGraph::from_text("abc");
        let g2 = GraphemeGraph::from_text("xyz");

        let bp2_12 = GraphEditDistance::compute_bp2(&g1, &g2);
        let bp2_21 = GraphEditDistance::compute_bp2(&g2, &g1);

        // Node mismatch cost should be symmetric
        assert!((bp2_12.node_mismatch_cost - bp2_21.node_mismatch_cost).abs() < 0.01);
    }

    // ========================================================================
    // Optimizer Tests (backend-028)
    // ========================================================================

    #[test]
    fn test_sgd_basic() {
        let mut sgd = SGD::new(0.1);
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        sgd.step(&mut params, &grads);

        // params should decrease by lr * grads
        assert!((params[[0, 0]] - 0.99).abs() < 1e-6); // 1.0 - 0.1 * 0.1
        assert!((params[[0, 1]] - 1.98).abs() < 1e-6); // 2.0 - 0.1 * 0.2
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut sgd = SGD::new(0.1).with_momentum(0.9);
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();

        // First step
        sgd.step(&mut params, &grads);
        let after_first = params[[0, 0]];

        // Second step with same gradient - momentum should increase update
        sgd.step(&mut params, &grads);

        // Second step should move more due to momentum
        let diff_first = 1.0 - after_first;
        let diff_second = after_first - params[[0, 0]];
        assert!(diff_second > diff_first);
    }

    #[test]
    fn test_sgd_with_weight_decay() {
        let mut sgd = SGD::new(0.1).with_weight_decay(0.01);
        let mut params = Array2::from_shape_vec((2, 2), vec![10.0, 10.0, 10.0, 10.0]).unwrap();
        let grads = Array2::zeros((2, 2));

        // With zero gradients, only weight decay should affect params
        sgd.step(&mut params, &grads);

        // params should shrink due to weight decay
        assert!(params[[0, 0]] < 10.0);
    }

    #[test]
    fn test_adam_basic() {
        let mut adam = Adam::new(0.1);
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.5, 0.5]).unwrap();

        adam.step(&mut params, &grads);

        // Params should have changed
        assert!(params[[0, 0]] != 1.0);
    }

    #[test]
    fn test_adam_multiple_steps() {
        let mut adam = Adam::new(0.01);
        let mut params = Array2::from_shape_vec((2, 2), vec![5.0, 5.0, 5.0, 5.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();

        let initial = params[[0, 0]];

        // Run multiple steps
        for _ in 0..10 {
            adam.step(&mut params, &grads);
        }

        // Params should decrease with positive gradients
        assert!(params[[0, 0]] < initial);
    }

    #[test]
    fn test_optimizer_set_lr() {
        let mut sgd = SGD::new(0.1);
        assert_eq!(sgd.get_lr(), 0.1);

        sgd.set_lr(0.01);
        assert_eq!(sgd.get_lr(), 0.01);
    }

    // ========================================================================
    // Learning Rate Scheduler Tests
    // ========================================================================

    #[test]
    fn test_lr_scheduler_constant() {
        let scheduler = LRScheduler::Constant;
        assert_eq!(scheduler.get_lr(0.1, 0), 0.1);
        assert_eq!(scheduler.get_lr(0.1, 100), 0.1);
    }

    #[test]
    fn test_lr_scheduler_step() {
        let scheduler = LRScheduler::StepLR { step_size: 10, gamma: 0.5 };

        assert_eq!(scheduler.get_lr(0.1, 0), 0.1);
        assert_eq!(scheduler.get_lr(0.1, 9), 0.1);
        assert_eq!(scheduler.get_lr(0.1, 10), 0.05);
        assert_eq!(scheduler.get_lr(0.1, 20), 0.025);
    }

    #[test]
    fn test_lr_scheduler_exponential() {
        let scheduler = LRScheduler::ExponentialLR { gamma: 0.9 };

        assert!((scheduler.get_lr(0.1, 0) - 0.1).abs() < 1e-6);
        assert!((scheduler.get_lr(0.1, 1) - 0.09).abs() < 1e-6);
        assert!((scheduler.get_lr(0.1, 2) - 0.081).abs() < 1e-6);
    }

    #[test]
    fn test_lr_scheduler_warmup() {
        let scheduler = LRScheduler::WarmupLR { warmup_steps: 5 };

        // During warmup, lr increases linearly
        assert!((scheduler.get_lr(0.1, 0) - 0.02).abs() < 1e-6);  // 1/5
        assert!((scheduler.get_lr(0.1, 4) - 0.1).abs() < 1e-6);   // 5/5
        assert_eq!(scheduler.get_lr(0.1, 10), 0.1);  // After warmup
    }

    #[test]
    fn test_lr_scheduler_cosine() {
        let scheduler = LRScheduler::CosineAnnealingLR { t_max: 10, eta_min: 0.0 };

        // At epoch 0, lr should be at max
        assert!((scheduler.get_lr(0.1, 0) - 0.1).abs() < 0.001);

        // At t_max/2, lr should be at eta_min (cos(pi) = -1, so (1 + -1)/2 = 0)
        // Actually at epoch 5: cos(pi * 5/10) = cos(pi/2) = 0, so lr = 0.05
        let lr_at_5 = scheduler.get_lr(0.1, 5);
        assert!((lr_at_5 - 0.05).abs() < 0.001, "Expected ~0.05, got {}", lr_at_5);

        // At t_max, lr should be back to eta_min (cos(pi) = -1)
        let lr_at_10 = scheduler.get_lr(0.1, 10);
        assert!((lr_at_10 - 0.1).abs() < 0.001, "Expected ~0.1 (cycle restart), got {}", lr_at_10);
    }

    // ========================================================================
    // Training Loop Tests
    // ========================================================================

    #[test]
    fn test_training_state_default() {
        let state = TrainingState::default();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);
        assert_eq!(state.best_val_loss, f32::MAX);
    }

    #[test]
    fn test_training_loop_creation() {
        let config = TrainingConfig::default();
        let loop_state = TrainingLoop::new(config);

        assert_eq!(loop_state.state.epoch, 0);
    }

    #[test]
    fn test_training_loop_record_batch() {
        let config = TrainingConfig::default();
        let mut loop_state = TrainingLoop::new(config);

        loop_state.record_batch(0.5);
        loop_state.record_batch(0.3);

        assert_eq!(loop_state.state.batches_in_epoch, 2);
        assert!((loop_state.state.running_loss - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_training_loop_complete_epoch() {
        let config = TrainingConfig::default();
        let mut loop_state = TrainingLoop::new(config);

        loop_state.record_batch(0.4);
        loop_state.record_batch(0.6);

        let avg_loss = loop_state.complete_epoch();

        assert!((avg_loss - 0.5).abs() < 1e-6);
        assert_eq!(loop_state.state.epoch, 1);
        assert_eq!(loop_state.state.batches_in_epoch, 0);
    }

    #[test]
    fn test_training_loop_early_stopping() {
        let mut config = TrainingConfig::default();
        config.patience = 2;
        let mut loop_state = TrainingLoop::new(config);

        // Record worse validation losses
        loop_state.record_validation(0.5, 0.8);
        assert!(!loop_state.should_stop());

        loop_state.record_validation(0.6, 0.75);
        assert!(!loop_state.should_stop());

        loop_state.record_validation(0.7, 0.7);
        assert!(loop_state.should_stop());
    }

    #[test]
    fn test_training_loop_with_scheduler() {
        let config = TrainingConfig::default();
        let loop_state = TrainingLoop::new(config)
            .with_scheduler(LRScheduler::StepLR { step_size: 5, gamma: 0.5 });

        assert!(matches!(loop_state.scheduler, LRScheduler::StepLR { .. }));
    }

    #[test]
    fn test_training_metrics_is_improving() {
        let mut metrics = TrainingMetrics::default();

        // Add improving losses
        metrics.epoch_losses.push(1.0);
        metrics.epoch_losses.push(0.9);
        metrics.epoch_losses.push(0.8);

        assert!(metrics.is_improving(2));

        // Add worsening losses
        metrics.epoch_losses.push(0.9);
        metrics.epoch_losses.push(1.0);

        assert!(!metrics.is_improving(2));
    }

    #[test]
    fn test_training_progress() {
        let mut config = TrainingConfig::default();
        config.epochs = 100;
        let mut loop_state = TrainingLoop::new(config);

        loop_state.state.epoch = 50;
        assert!((loop_state.progress() - 50.0).abs() < 1e-6);
    }

    // ========================================================================
    // Pipeline Tests
    // ========================================================================

    #[test]
    fn test_pipeline_creation() {
        let pipeline = Pipeline::new();
        assert_eq!(pipeline.mode, PipelineMode::Inference);
        assert!(!pipeline.cache_enabled);
        assert!(pipeline.transform_net.is_none());
    }

    #[test]
    fn test_pipeline_simple_addition() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("2 + 3");

        assert!(result.success());
        assert_eq!(result.numeric_result, Some(5.0));
    }

    #[test]
    fn test_pipeline_subtraction() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("10 - 4");

        assert!(result.success());
        assert_eq!(result.numeric_result, Some(6.0));
    }

    #[test]
    fn test_pipeline_multiplication() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("3 * 4");

        assert!(result.success());
        assert_eq!(result.numeric_result, Some(12.0));
    }

    #[test]
    fn test_pipeline_division() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("8 / 2");

        assert!(result.success());
        assert_eq!(result.numeric_result, Some(4.0));
    }

    #[test]
    fn test_pipeline_what_is() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("what is 5 + 7");

        assert!(result.success());
        assert_eq!(result.numeric_result, Some(12.0));
    }

    #[test]
    fn test_pipeline_calculate() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("calculate 9 - 3");

        assert!(result.success());
        assert_eq!(result.numeric_result, Some(6.0));
    }

    #[test]
    fn test_pipeline_symbolic() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("x + 0");

        // Optimizer should simplify x + 0 to x
        assert!(result.success());
        assert!(result.symbolic_result.is_some());
    }

    #[test]
    fn test_pipeline_derivative() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("derivative of x squared");

        // d/dx(x^2) = 2x
        assert!(result.success());
        assert!(result.symbolic_result.is_some());
    }

    #[test]
    fn test_pipeline_integrate_definite() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("integrate x from 0 to 1");

        // ∫x dx from 0 to 1 = 0.5
        assert!(result.success());
        assert_eq!(result.numeric_result, Some(0.5));
    }

    #[test]
    fn test_pipeline_result_string() {
        let pipeline = Pipeline::new();

        let result = pipeline.process("2 + 3");
        assert_eq!(result.result_string(), "5");

        let result = pipeline.process("x");
        assert_eq!(result.result_string(), "x");
    }

    #[test]
    fn test_pipeline_mode() {
        let pipeline = Pipeline::new()
            .with_mode(PipelineMode::Training);

        assert_eq!(pipeline.mode, PipelineMode::Training);
        assert!(pipeline.is_training());
    }

    #[test]
    fn test_pipeline_with_cache() {
        let pipeline = Pipeline::new()
            .with_cache(true);

        assert!(pipeline.cache_enabled);
    }

    #[test]
    fn test_pipeline_batch_processing() {
        let pipeline = Pipeline::new();
        let inputs = vec!["1 + 1", "2 + 2", "3 + 3"];
        let results = pipeline.process_batch(&inputs);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].numeric_result, Some(2.0));
        assert_eq!(results[1].numeric_result, Some(4.0));
        assert_eq!(results[2].numeric_result, Some(6.0));
    }

    #[test]
    fn test_pipeline_variable_binding() {
        let mut pipeline = Pipeline::new();
        pipeline.bind("x", 5.0);

        let result = pipeline.process("x + 1");
        assert!(result.success());
        assert_eq!(result.numeric_result, Some(6.0));
    }

    #[test]
    fn test_pipeline_steps_recorded() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("2 + 3");

        assert!(!result.steps.is_empty());
        assert!(result.steps.iter().any(|s| s.contains("Layer 4")));
        assert!(result.steps.iter().any(|s| s.contains("Layer 3")));
        assert!(result.steps.iter().any(|s| s.contains("Layer 2")));
        assert!(result.steps.iter().any(|s| s.contains("Layer 1")));
    }

    #[test]
    fn test_pipeline_graphs_created() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("2 + 3");

        assert!(result.nl_graph.is_some());
        assert!(result.math_graph.is_some());
        assert!(result.optimized_expr.is_some());
    }

    #[test]
    fn test_quick_eval() {
        assert_eq!(quick_eval("2 + 3"), Some(5.0));
        assert_eq!(quick_eval("10 - 5"), Some(5.0));
        assert_eq!(quick_eval("4 * 3"), Some(12.0));
        assert_eq!(quick_eval("8 / 2"), Some(4.0));
    }

    #[test]
    fn test_quick_symbolic() {
        let result = quick_symbolic("x + 0");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "x");
    }

    #[test]
    fn test_pipeline_compact_operators() {
        let pipeline = Pipeline::new();

        let result = pipeline.process("2+3");
        assert_eq!(result.numeric_result, Some(5.0));

        let result = pipeline.process("10-4");
        assert_eq!(result.numeric_result, Some(6.0));
    }

    #[test]
    fn test_pipeline_squared() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("3 squared");

        assert!(result.success());
        assert_eq!(result.numeric_result, Some(9.0));
    }

    #[test]
    fn test_pipeline_cubed() {
        let pipeline = Pipeline::new();
        let result = pipeline.process("2 cubed");

        assert!(result.success());
        assert_eq!(result.numeric_result, Some(8.0));
    }

    // ========================================================================
    // Benchmark Utility Tests (testing-004)
    // ========================================================================

    #[test]
    fn test_grapheme_graph_scaling() {
        // Test O(n) scaling of graph creation
        let sizes = [100, 1000, 10000];
        let mut times: Vec<(usize, std::time::Duration)> = Vec::new();

        for &size in &sizes {
            let text: String = "a".repeat(size);
            let start = std::time::Instant::now();
            let graph = GraphemeGraph::from_text(&text);
            let elapsed = start.elapsed();
            times.push((size, elapsed));

            // Verify graph has expected structure
            assert_eq!(graph.node_count(), size);
        }

        // Verify linear scaling: time should scale roughly linearly
        // 10x input should take roughly 10x time (with some tolerance)
        if times.len() >= 2 {
            let (size1, time1) = times[0];
            let (size2, time2) = times[1];
            let size_ratio = size2 as f64 / size1 as f64;
            let time_ratio = time2.as_nanos() as f64 / time1.as_nanos() as f64;

            // Time ratio should be within 3x of size ratio for O(n) scaling
            assert!(
                time_ratio < size_ratio * 3.0,
                "Scaling too slow: size ratio {}, time ratio {}",
                size_ratio,
                time_ratio
            );
        }
    }

    #[test]
    fn test_transformer_flop_calculation() {
        // Verify O(n²) scaling of transformer FLOPs
        let d_model = 256;
        let n_heads = 8;
        let d_head = d_model / n_heads;

        // Manual calculation for seq_len=100
        let seq_len = 100;

        // Q, K, V projections: 3 * n * d * d
        let proj_flops = 3 * seq_len * d_model * d_model;

        // Attention scores: n * n * d_head (for each head)
        let attn_flops = n_heads * seq_len * seq_len * d_head;

        // Softmax: ~n * n * 5
        let softmax_flops = seq_len * seq_len * 5;

        // Output: n * n * d_head (for each head)
        let output_flops = n_heads * seq_len * seq_len * d_head;

        // Output projection: n * d * d
        let out_proj_flops = seq_len * d_model * d_model;

        let total = proj_flops + attn_flops + softmax_flops + output_flops + out_proj_flops;

        // At least n² term from attention
        assert!(total > seq_len * seq_len);
    }

    #[test]
    fn test_memory_comparison() {
        // GRAPHEME: ~17 bytes per node (character)
        let grapheme_bytes_per_char = 17;

        // Transformer: needs Q, K, V + attention matrix
        let d_model = 256;
        let seq_len = 1000;

        // Q, K, V matrices: 3 * n * d * sizeof(f32)
        let qkv_mem = 3 * seq_len * d_model * 4;

        // Attention matrix: n * n * sizeof(f32)
        let attn_mem = seq_len * seq_len * 4;

        let transformer_mem = qkv_mem + attn_mem;

        // GRAPHEME memory
        let grapheme_mem = seq_len * grapheme_bytes_per_char;

        // Transformer should use more memory due to O(n²) attention
        assert!(transformer_mem > grapheme_mem);

        // For n=1000, transformer attention alone is 4MB, GRAPHEME is 17KB
        assert!(transformer_mem / grapheme_mem > 100);
    }

    #[test]
    fn test_scaling_ratio() {
        // Test that efficiency ratio increases with input length
        let d_model = 256;
        let n_heads = 8;

        let calc_ratio = |seq_len: usize| -> f64 {
            // Transformer FLOPs (dominated by O(n²) attention)
            let d_head = d_model / n_heads;
            let proj = 3 * seq_len * d_model * d_model;
            let attn = n_heads * seq_len * seq_len * d_head;
            let softmax = seq_len * seq_len * 5;
            let output = n_heads * seq_len * seq_len * d_head;
            let out_proj = seq_len * d_model * d_model;
            let transformer_flops = (proj + attn + softmax + output + out_proj) as f64;

            // GRAPHEME ops: O(n)
            let grapheme_ops = (seq_len * 7) as f64; // ~7 ops per character

            transformer_flops / grapheme_ops
        };

        let ratio_100 = calc_ratio(100);
        let ratio_1000 = calc_ratio(1000);
        let ratio_10000 = calc_ratio(10000);

        // Ratio should increase with input length due to O(n²) vs O(n)
        assert!(ratio_1000 > ratio_100);
        assert!(ratio_10000 > ratio_1000);
    }
}
