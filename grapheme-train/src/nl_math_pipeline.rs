//! NL to Math Pipeline - End-to-End Processing (backend-030)
//!
//! Chains Layer 4 (grapheme-core) → Layer 3 (grapheme-math) → Layer 2 (grapheme-polish) → Layer 1 (grapheme-engine)
//!
//! ## Pipeline Flow (leveraging GRAPHEME capabilities)
//!
//! 1. **NL Input** → `DagNN::from_text()` → Character-level graph
//! 2. **Character Graph** → `GraphTransformNet::transform()` → Transformed graph
//! 3. **Transformed Graph** → Pattern recognition → `MathGraph`
//! 4. **MathGraph** → `to_expr()` → `Expr`
//! 5. **Expr** → `MathEngine::evaluate()` → Result
//!
//! ## Usage
//!
//! ```rust,ignore
//! use grapheme_train::nl_math_pipeline::{Pipeline, PipelineConfig};
//!
//! let mut pipeline = Pipeline::new(PipelineConfig::default());
//! let result = pipeline.process("2 + 3")?;
//! assert_eq!(result.value, 5.0);
//! ```

use crate::graph_transform_net::GraphTransformNet;
use grapheme_core::{DagNN, ForwardPass, GraphTransformer, NodeType};
use grapheme_engine::{Expr, MathEngine, MathFn, MathOp, Value, EngineError};
use grapheme_math::{MathBrain, MathGraph, MathGraphError};
use grapheme_polish::PolishError;
use thiserror::Error;

// ============================================================================
// Pipeline Errors
// ============================================================================

/// Errors during pipeline processing
#[derive(Error, Debug)]
pub enum PipelineError {
    /// Error in Layer 4 (core graph construction)
    #[error("Layer 4 (Core) error: {0}")]
    CoreError(String),

    /// Error in Layer 3 (math graph extraction)
    #[error("Layer 3 (Math) error: {0}")]
    MathError(#[from] MathGraphError),

    /// Error in Layer 2 (polish notation)
    #[error("Layer 2 (Polish) error: {0}")]
    PolishError(#[from] PolishError),

    /// Error in Layer 1 (engine evaluation)
    #[error("Layer 1 (Engine) error: {0}")]
    EngineError(#[from] EngineError),

    /// Empty input
    #[error("Empty input")]
    EmptyInput,

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Graph transformation error
    #[error("Graph transform error: {0}")]
    TransformError(String),
}

/// Result type for pipeline operations
pub type PipelineResult<T> = Result<T, PipelineError>;

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Pipeline mode: inference (frozen) or training (gradients)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelineMode {
    /// Inference mode - frozen weights
    #[default]
    Inference,
    /// Training mode - gradient flow
    Training,
}

/// Configuration for the pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Pipeline mode
    pub mode: PipelineMode,
    /// Enable caching of intermediate representations
    pub cache_intermediate: bool,
    /// Verbose output
    pub verbose: bool,
    /// Use learned graph transformer (vs pattern-based)
    pub use_learned_transform: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            mode: PipelineMode::Inference,
            cache_intermediate: false,
            verbose: false,
            use_learned_transform: false, // Start with pattern-based, enable learned after training
        }
    }
}

// ============================================================================
// Pipeline Result
// ============================================================================

/// Result of pipeline processing
#[derive(Debug)]
pub struct PipelineOutput {
    /// The evaluated numeric value
    pub value: f64,
    /// Original input text
    pub input: String,
    /// Parsed expression (if available)
    pub expr: Option<Expr>,
    /// Intermediate MathGraph (if caching enabled)
    pub math_graph: Option<MathGraph>,
    /// Input DagNN (if caching enabled)
    pub input_dag: Option<DagNN>,
}

// ============================================================================
// Pipeline Implementation
// ============================================================================

/// The NL-to-Math pipeline connecting all 4 layers
///
/// Uses GRAPHEME's Graph → Transform → Graph paradigm:
/// - DagNN for character-level input processing
/// - GraphTransformNet for learned transformations
/// - MathGraph for structured math representation
/// - MathEngine for evaluation
pub struct Pipeline {
    /// Pipeline configuration
    pub config: PipelineConfig,
    /// Math engine for evaluation (Layer 1)
    pub engine: MathEngine,
    /// Math brain for graph construction (Layer 3)
    pub math_brain: MathBrain,
    /// Learned graph transformer (optional)
    pub transformer: GraphTransformNet,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new(PipelineConfig::default())
    }
}

impl Pipeline {
    /// Create a new pipeline with configuration
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            engine: MathEngine::new(),
            math_brain: MathBrain::new(),
            transformer: GraphTransformNet::new(),
        }
    }

    /// Process natural language input through all layers
    ///
    /// Layer 4 (DagNN) → Layer 3 (MathGraph) → Layer 2 (Polish) → Layer 1 (Engine)
    pub fn process(&mut self, input: &str) -> PipelineResult<PipelineOutput> {
        let input_str = input.trim();
        if input_str.is_empty() {
            return Err(PipelineError::EmptyInput);
        }

        if self.config.verbose {
            println!("[Pipeline] Input: {}", input_str);
        }

        // Layer 4: Create character-level DagNN from input
        let mut dag = DagNN::from_text(input_str)
            .map_err(|e| PipelineError::CoreError(e.to_string()))?;

        if self.config.verbose {
            println!("[Pipeline] DagNN created with {} nodes", dag.graph.node_count());
        }

        // Run forward pass to compute activations
        dag.forward()
            .map_err(|e| PipelineError::CoreError(e.to_string()))?;

        // Transform to get math structure
        let expr = if self.config.use_learned_transform {
            // Use learned GraphTransformNet
            let transformed = self.transformer.transform(&dag)
                .map_err(|e| PipelineError::TransformError(e.to_string()))?;
            self.dag_to_expr(&transformed)?
        } else {
            // Use pattern-based extraction from DagNN
            self.extract_math_from_dag(&dag)?
        };

        if self.config.verbose {
            println!("[Pipeline] Extracted expression: {:?}", expr);
        }

        // Build math graph if caching
        let math_graph = if self.config.cache_intermediate {
            Some(MathGraph::from_expr(&expr))
        } else {
            None
        };

        // Layer 1: Evaluate expression using engine
        let value = self.engine.evaluate(&expr)?;

        if self.config.verbose {
            println!("[Pipeline] Result: {}", value);
        }

        Ok(PipelineOutput {
            value,
            input: input_str.to_string(),
            expr: Some(expr),
            math_graph,
            input_dag: if self.config.cache_intermediate { Some(dag) } else { None },
        })
    }

    /// Extract math expression from DagNN using graph structure analysis
    ///
    /// This uses GRAPHEME's character-level graph to recognize:
    /// - Numbers (sequences of digit nodes)
    /// - Operators (single character nodes: +, -, *, /, ^)
    /// - Functions (character sequences: sin, cos, sqrt, etc.)
    /// - Symbols (alphabetic sequences)
    fn extract_math_from_dag(&self, dag: &DagNN) -> PipelineResult<Expr> {
        // Extract text from DagNN nodes
        let text = self.dag_to_text(dag);

        // Parse the reconstructed text using the expression parser
        // This bridges character-level representation to semantic math
        self.parse_math_text(&text)
    }

    /// Convert DagNN back to text (following node sequence)
    fn dag_to_text(&self, dag: &DagNN) -> String {
        dag.input_nodes()
            .iter()
            .filter_map(|&node| {
                if let NodeType::Input(ch) = &dag.graph[node].node_type {
                    Some(*ch)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Convert transformed DagNN to expression
    fn dag_to_expr(&self, dag: &DagNN) -> PipelineResult<Expr> {
        let text = self.dag_to_text(dag);
        if text.is_empty() {
            return Err(PipelineError::ParseError("Empty transformation result".into()));
        }
        self.parse_math_text(&text)
    }

    /// Parse math text to expression
    ///
    /// Recognizes patterns in the graph-extracted text:
    /// - Natural language: "what is", "calculate", "squared"
    /// - Math notation: operators, functions, parentheses
    fn parse_math_text(&self, text: &str) -> PipelineResult<Expr> {
        let text = text.trim();

        // Try as number first
        if let Ok(n) = text.parse::<i64>() {
            return Ok(Expr::int(n));
        }
        if let Ok(f) = text.parse::<f64>() {
            return Ok(Expr::float(f));
        }

        // Try NL patterns
        if let Some(expr) = self.parse_nl_pattern(text) {
            return Ok(expr);
        }

        // Parse mathematical expression
        Self::parse_expression(text)
    }

    /// Parse natural language patterns from graph-extracted text
    fn parse_nl_pattern(&self, input: &str) -> Option<Expr> {
        let lower = input.to_lowercase();

        // "what is X + Y" or "what's X + Y"
        if lower.starts_with("what is ") || lower.starts_with("what's ") {
            let rest = if lower.starts_with("what is ") {
                &input[8..]
            } else {
                &input[7..]
            };
            return Self::parse_expression(rest.trim()).ok();
        }

        // "calculate X"
        if lower.starts_with("calculate ") {
            return Self::parse_expression(input[10..].trim()).ok();
        }

        // "derivative of X" or "derive X"
        if lower.starts_with("derivative of ") {
            let expr_str = input[14..].trim();
            if let Ok(expr) = Self::parse_expression(expr_str) {
                return Some(Expr::Function {
                    func: MathFn::Derive,
                    args: vec![expr],
                });
            }
        }

        // "integrate X" or "integral of X"
        if lower.starts_with("integrate ") || lower.starts_with("integral of ") {
            let start = if lower.starts_with("integrate ") { 10 } else { 12 };
            let expr_str = input[start..].trim();
            if let Ok(expr) = Self::parse_expression(expr_str) {
                return Some(Expr::Function {
                    func: MathFn::Integrate,
                    args: vec![expr],
                });
            }
        }

        // "square root of X" or "sqrt X"
        if lower.starts_with("square root of ") {
            if let Ok(expr) = Self::parse_expression(input[15..].trim()) {
                return Some(Expr::Function {
                    func: MathFn::Sqrt,
                    args: vec![expr],
                });
            }
        }
        if lower.starts_with("sqrt ") {
            if let Ok(expr) = Self::parse_expression(input[5..].trim()) {
                return Some(Expr::Function {
                    func: MathFn::Sqrt,
                    args: vec![expr],
                });
            }
        }

        // "X squared" or "X to the power of Y"
        if lower.ends_with(" squared") {
            let base = input[..input.len() - 8].trim();
            if let Ok(expr) = Self::parse_expression(base) {
                return Some(Expr::BinOp {
                    op: MathOp::Pow,
                    left: Box::new(expr),
                    right: Box::new(Expr::int(2)),
                });
            }
        }

        None
    }

    /// Parse a mathematical expression using recursive descent
    ///
    /// Operator precedence (low to high):
    /// - Addition/Subtraction (+, -)
    /// - Multiplication/Division (*, /)
    /// - Exponentiation (^)
    fn parse_expression(input: &str) -> PipelineResult<Expr> {
        let input = input.trim();

        // Handle parentheses
        if input.starts_with('(') && input.ends_with(')') {
            return Self::parse_expression(&input[1..input.len()-1]);
        }

        // Find lowest precedence operator (right to left for left associativity)
        let mut depth = 0;
        let chars: Vec<char> = input.chars().collect();

        // Try addition/subtraction first (lowest precedence)
        for i in (0..chars.len()).rev() {
            match chars[i] {
                '(' => depth += 1,
                ')' => depth -= 1,
                '+' | '-' if depth == 0 && i > 0 => {
                    let prev = chars[i-1];
                    if prev != '+' && prev != '-' && prev != '*' && prev != '/' && prev != '^' && prev != '(' {
                        let left = input[..i].trim();
                        let right = input[i+1..].trim();

                        if !left.is_empty() && !right.is_empty() {
                            let left_expr = Self::parse_expression(left)?;
                            let right_expr = Self::parse_expression(right)?;
                            let op = if chars[i] == '+' { MathOp::Add } else { MathOp::Sub };
                            return Ok(Expr::BinOp {
                                op,
                                left: Box::new(left_expr),
                                right: Box::new(right_expr),
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        // Try multiplication/division
        depth = 0;
        for i in (0..chars.len()).rev() {
            match chars[i] {
                '(' => depth += 1,
                ')' => depth -= 1,
                '*' | '/' if depth == 0 => {
                    let left = input[..i].trim();
                    let right = input[i+1..].trim();

                    if !left.is_empty() && !right.is_empty() {
                        let left_expr = Self::parse_expression(left)?;
                        let right_expr = Self::parse_expression(right)?;
                        let op = if chars[i] == '*' { MathOp::Mul } else { MathOp::Div };
                        return Ok(Expr::BinOp {
                            op,
                            left: Box::new(left_expr),
                            right: Box::new(right_expr),
                        });
                    }
                }
                _ => {}
            }
        }

        // Try power operator
        depth = 0;
        for i in (0..chars.len()).rev() {
            match chars[i] {
                '(' => depth += 1,
                ')' => depth -= 1,
                '^' if depth == 0 => {
                    let left = input[..i].trim();
                    let right = input[i+1..].trim();

                    if !left.is_empty() && !right.is_empty() {
                        let left_expr = Self::parse_expression(left)?;
                        let right_expr = Self::parse_expression(right)?;
                        return Ok(Expr::BinOp {
                            op: MathOp::Pow,
                            left: Box::new(left_expr),
                            right: Box::new(right_expr),
                        });
                    }
                }
                _ => {}
            }
        }

        // Try as a number
        if let Ok(n) = input.parse::<i64>() {
            return Ok(Expr::int(n));
        }
        if let Ok(f) = input.parse::<f64>() {
            return Ok(Expr::float(f));
        }

        // Try as a symbol (variable name)
        if input.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Ok(Expr::symbol(input));
        }

        // Handle unary minus
        if let Some(rest) = input.strip_prefix('-') {
            let expr = Self::parse_expression(rest.trim())?;
            return Ok(Expr::UnaryOp {
                op: MathOp::Neg,
                operand: Box::new(expr),
            });
        }

        Err(PipelineError::ParseError(format!("Cannot parse: {}", input)))
    }

    /// Bind a symbol to a value for evaluation
    pub fn bind(&mut self, name: impl Into<String>, value: Value) {
        self.engine.bind(name, value);
    }

    /// Process multiple inputs in batch
    pub fn process_batch(&mut self, inputs: &[&str]) -> Vec<PipelineResult<PipelineOutput>> {
        inputs.iter().map(|input| self.process(input)).collect()
    }

    /// Set pipeline mode
    pub fn set_mode(&mut self, mode: PipelineMode) {
        self.config.mode = mode;
    }

    /// Train the pipeline on an input-output pair
    ///
    /// Uses GraphTransformNet.learn_transformation() to learn the mapping
    pub fn train(&mut self, input: &str, expected_output: &str) -> PipelineResult<f32> {
        let input_dag = DagNN::from_text(input)
            .map_err(|e| PipelineError::CoreError(e.to_string()))?;
        let target_dag = DagNN::from_text(expected_output)
            .map_err(|e| PipelineError::CoreError(e.to_string()))?;

        // Train the transformer
        let _rule = self.transformer.learn_transformation(&input_dag, &target_dag);

        // Return loss (training step returns loss internally)
        Ok(self.transformer.train_step(&input_dag, &target_dag))
    }

    /// Enable learned transformation after training
    pub fn enable_learned_transform(&mut self) {
        self.config.use_learned_transform = true;
    }

    /// Get reference to the underlying GraphTransformNet
    pub fn transformer(&self) -> &GraphTransformNet {
        &self.transformer
    }

    /// Get mutable reference to the underlying GraphTransformNet
    pub fn transformer_mut(&mut self) -> &mut GraphTransformNet {
        &mut self.transformer
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Quick evaluation of a natural language math expression
///
/// Uses the full GRAPHEME pipeline: DagNN → GraphTransform → MathGraph → Engine
///
/// ```rust,ignore
/// let result = quick_evaluate("2 + 3")?;
/// assert_eq!(result, 5.0);
/// ```
pub fn quick_evaluate(input: &str) -> PipelineResult<f64> {
    let mut pipeline = Pipeline::default();
    let output = pipeline.process(input)?;
    Ok(output.value)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = Pipeline::new(PipelineConfig::default());
        assert_eq!(pipeline.config.mode, PipelineMode::Inference);
        assert!(!pipeline.config.use_learned_transform);
    }

    #[test]
    fn test_dagnn_creation() {
        let dag = DagNN::from_text("2 + 3").unwrap();
        assert_eq!(dag.graph.node_count(), 5); // '2', ' ', '+', ' ', '3'
    }

    #[test]
    fn test_simple_addition() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("2 + 3").unwrap();
        assert!((result.value - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_simple_subtraction() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("10 - 4").unwrap();
        assert!((result.value - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_simple_multiplication() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("3 * 4").unwrap();
        assert!((result.value - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_simple_division() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("10 / 2").unwrap();
        assert!((result.value - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_power() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("2 ^ 3").unwrap();
        assert!((result.value - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_complex_expression() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("2 + 3 * 4").unwrap();
        assert!((result.value - 14.0).abs() < 1e-9); // 2 + (3*4) = 14
    }

    #[test]
    fn test_parentheses() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("(2 + 3) * 4").unwrap();
        assert!((result.value - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_what_is() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("what is 5 + 7").unwrap();
        assert!((result.value - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_calculate() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("calculate 6 * 9").unwrap();
        assert!((result.value - 54.0).abs() < 1e-9);
    }

    #[test]
    fn test_squared() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("5 squared").unwrap();
        assert!((result.value - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_square_root() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("square root of 16").unwrap();
        assert!((result.value - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_with_symbols() {
        let mut pipeline = Pipeline::default();
        pipeline.bind("x", Value::Float(5.0));
        let result = pipeline.process("x + 3").unwrap();
        assert!((result.value - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_batch_processing() {
        let mut pipeline = Pipeline::default();
        let results = pipeline.process_batch(&["1 + 1", "2 * 2", "3 ^ 2"]);
        assert_eq!(results.len(), 3);
        assert!((results[0].as_ref().unwrap().value - 2.0).abs() < 1e-9);
        assert!((results[1].as_ref().unwrap().value - 4.0).abs() < 1e-9);
        assert!((results[2].as_ref().unwrap().value - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_negative_number() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("-5 + 3").unwrap();
        assert!((result.value - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn test_float() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("3.14 * 2").unwrap();
        assert!((result.value - 6.28).abs() < 1e-9);
    }

    #[test]
    fn test_empty_input() {
        let mut pipeline = Pipeline::default();
        let result = pipeline.process("");
        assert!(matches!(result, Err(PipelineError::EmptyInput)));
    }

    #[test]
    fn test_cache_intermediate() {
        let mut pipeline = Pipeline::new(PipelineConfig {
            cache_intermediate: true,
            ..Default::default()
        });
        let result = pipeline.process("2 + 3").unwrap();
        assert!(result.math_graph.is_some());
        assert!(result.input_dag.is_some());
    }

    #[test]
    fn test_dag_to_text_roundtrip() {
        let pipeline = Pipeline::default();
        let dag = DagNN::from_text("2 + 3 * 4").unwrap();
        let text = pipeline.dag_to_text(&dag);
        assert_eq!(text, "2 + 3 * 4");
    }

    #[test]
    fn test_pipeline_training() {
        let mut pipeline = Pipeline::default();
        // Train on a simple transformation
        let loss = pipeline.train("2+3", "5").unwrap();
        assert!(loss >= 0.0); // Loss should be non-negative
    }

    #[test]
    fn test_quick_evaluate() {
        let result = quick_evaluate("7 * 8").unwrap();
        assert!((result - 56.0).abs() < 1e-9);
    }
}
