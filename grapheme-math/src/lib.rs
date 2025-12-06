//! # grapheme-math
//!
//! Layer 3: The learned math brain for GRAPHEME.
//!
//! This crate provides:
//! - Typed math nodes (Int, Float, Operator, Function, Symbol)
//! - Graph-based math intent extraction
//! - Learned graph transformations
//! - Training by Layer 1, validation by Layer 1
//!
//! Unlike traditional NLP, we use typed nodes rather than embeddings,
//! but remain vocabulary-free through the type system.

// Allow &self in recursive methods for API consistency
#![allow(clippy::only_used_in_recursion)]

use grapheme_core::{DagNN, DomainBrain, DomainExample, DomainResult, DomainRule, ExecutionResult, ValidationIssue, ValidationSeverity};
use grapheme_engine::{Expr, MathEngine, MathFn, MathOp, Value};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors in math graph processing
#[derive(Error, Debug)]
pub enum MathGraphError {
    #[error("Invalid graph structure: {0}")]
    InvalidStructure(String),
    #[error("Type mismatch in graph: {0}")]
    TypeMismatch(String),
    #[error("Engine error: {0}")]
    EngineError(#[from] grapheme_engine::EngineError),
}

/// Result type for math graph operations
pub type MathGraphResult<T> = Result<T, MathGraphError>;

/// Typed node for mathematical expressions
/// These are semantic types, not embeddings - still vocabulary-free
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MathNode {
    /// Integer literal
    Integer(i64),
    /// Floating-point literal
    Float(f64),
    /// Symbol/variable name
    Symbol(String),
    /// Binary operator
    Operator(MathOp),
    /// Mathematical function
    Function(MathFn),
    /// Result placeholder
    Result,
}

/// Edge types in the math graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathEdge {
    /// Left operand of binary operation
    LeftOperand,
    /// Right operand of binary operation
    RightOperand,
    /// Argument to function
    Argument(usize),
    /// Flow from expression to result
    Produces,
}

/// A mathematical expression represented as a graph
#[derive(Debug)]
pub struct MathGraph {
    /// The underlying directed graph
    pub graph: DiGraph<MathNode, MathEdge>,
    /// Root node of the expression
    pub root: Option<NodeIndex>,
}

impl Default for MathGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MathGraph {
    /// Create a new empty math graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            root: None,
        }
    }

    /// Build a graph from an expression
    pub fn from_expr(expr: &Expr) -> Self {
        let mut graph = Self::new();
        graph.root = Some(graph.add_expr(expr));
        graph
    }

    /// Add an expression to the graph, returning the root node
    fn add_expr(&mut self, expr: &Expr) -> NodeIndex {
        match expr {
            Expr::Value(v) => self.add_value(v),
            Expr::BinOp { op, left, right } => {
                let op_node = self.graph.add_node(MathNode::Operator(*op));
                let left_node = self.add_expr(left);
                let right_node = self.add_expr(right);
                self.graph.add_edge(op_node, left_node, MathEdge::LeftOperand);
                self.graph
                    .add_edge(op_node, right_node, MathEdge::RightOperand);
                op_node
            }
            Expr::UnaryOp { op, operand } => {
                let op_node = self.graph.add_node(MathNode::Operator(*op));
                let operand_node = self.add_expr(operand);
                self.graph
                    .add_edge(op_node, operand_node, MathEdge::LeftOperand);
                op_node
            }
            Expr::Function { func, args } => {
                let func_node = self.graph.add_node(MathNode::Function(*func));
                for (i, arg) in args.iter().enumerate() {
                    let arg_node = self.add_expr(arg);
                    self.graph
                        .add_edge(func_node, arg_node, MathEdge::Argument(i));
                }
                func_node
            }
        }
    }

    fn add_value(&mut self, value: &Value) -> NodeIndex {
        let node = match value {
            Value::Integer(i) => MathNode::Integer(*i),
            Value::Float(f) => MathNode::Float(*f),
            Value::Symbol(s) => MathNode::Symbol(s.clone()),
            Value::Rational(n, d) => MathNode::Float(*n as f64 / *d as f64),
        };
        self.graph.add_node(node)
    }

    /// Convert the graph back to an expression
    pub fn to_expr(&self) -> MathGraphResult<Expr> {
        let root = self
            .root
            .ok_or_else(|| MathGraphError::InvalidStructure("No root node".into()))?;
        self.node_to_expr(root)
    }

    fn node_to_expr(&self, node: NodeIndex) -> MathGraphResult<Expr> {
        let math_node = &self.graph[node];

        match math_node {
            MathNode::Integer(i) => Ok(Expr::Value(Value::Integer(*i))),
            MathNode::Float(f) => Ok(Expr::Value(Value::Float(*f))),
            MathNode::Symbol(s) => Ok(Expr::Value(Value::Symbol(s.clone()))),
            MathNode::Operator(op) => {
                let mut left = None;
                let mut right = None;

                for edge in self.graph.edges(node) {
                    match edge.weight() {
                        MathEdge::LeftOperand => {
                            left = Some(self.node_to_expr(edge.target())?);
                        }
                        MathEdge::RightOperand => {
                            right = Some(self.node_to_expr(edge.target())?);
                        }
                        _ => {}
                    }
                }

                let left =
                    left.ok_or_else(|| MathGraphError::InvalidStructure("Missing left operand".into()))?;

                if let Some(right) = right {
                    Ok(Expr::BinOp {
                        op: *op,
                        left: Box::new(left),
                        right: Box::new(right),
                    })
                } else {
                    Ok(Expr::UnaryOp {
                        op: *op,
                        operand: Box::new(left),
                    })
                }
            }
            MathNode::Function(func) => {
                let mut args: Vec<(usize, Expr)> = Vec::new();

                for edge in self.graph.edges(node) {
                    if let MathEdge::Argument(i) = edge.weight() {
                        args.push((*i, self.node_to_expr(edge.target())?));
                    }
                }

                args.sort_by_key(|(i, _)| *i);
                let args: Vec<Expr> = args.into_iter().map(|(_, e)| e).collect();

                Ok(Expr::Function { func: *func, args })
            }
            MathNode::Result => Err(MathGraphError::InvalidStructure(
                "Result node cannot be converted to expression".into(),
            )),
        }
    }

    /// Evaluate the graph using the engine
    pub fn evaluate(&self, engine: &MathEngine) -> MathGraphResult<f64> {
        let expr = self.to_expr()?;
        Ok(engine.evaluate(&expr)?)
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

// ============================================================================
// Math Intent (extracted from expressions)
// ============================================================================

/// Mathematical intent extracted from an expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MathIntent {
    /// Arithmetic computation (2 + 3)
    Compute,
    /// Simplification (x + 0 → x)
    Simplify,
    /// Solve equation (x + 2 = 5 → x = 3)
    Solve,
    /// Evaluate at point (f(x) at x=2)
    Evaluate,
    /// Differentiate (d/dx of x²)
    Differentiate,
    /// Integrate (∫x dx)
    Integrate,
    /// Factor expression (x² - 1 → (x-1)(x+1))
    Factor,
    /// Expand expression ((x+1)² → x² + 2x + 1)
    Expand,
}

/// Extracted math problem with intent and components
#[derive(Debug, Clone)]
pub struct MathProblem {
    /// The identified intent
    pub intent: MathIntent,
    /// The main expression
    pub expression: Expr,
    /// Variable of interest (for calculus operations)
    pub variable: Option<String>,
    /// Bounds (for definite integrals)
    pub bounds: Option<(f64, f64)>,
    /// Expected result (if known)
    pub expected: Option<f64>,
}

// ============================================================================
// Simplification Rules
// ============================================================================

/// A simplification rule for algebraic expressions
#[derive(Debug, Clone)]
pub struct SimplificationRule {
    /// Rule name
    pub name: &'static str,
    /// Description
    pub description: &'static str,
}

impl SimplificationRule {
    /// Identity rules (x + 0 = x, x * 1 = x, etc.)
    pub const ADDITIVE_IDENTITY: Self = SimplificationRule {
        name: "additive_identity",
        description: "x + 0 = x",
    };

    pub const MULTIPLICATIVE_IDENTITY: Self = SimplificationRule {
        name: "multiplicative_identity",
        description: "x * 1 = x",
    };

    pub const ZERO_PRODUCT: Self = SimplificationRule {
        name: "zero_product",
        description: "x * 0 = 0",
    };

    pub const ADDITIVE_INVERSE: Self = SimplificationRule {
        name: "additive_inverse",
        description: "x - x = 0",
    };

    pub const DIVISION_IDENTITY: Self = SimplificationRule {
        name: "division_identity",
        description: "x / 1 = x",
    };

    pub const POWER_ONE: Self = SimplificationRule {
        name: "power_one",
        description: "x ^ 1 = x",
    };

    pub const POWER_ZERO: Self = SimplificationRule {
        name: "power_zero",
        description: "x ^ 0 = 1",
    };
}

// ============================================================================
// Math Transformer
// ============================================================================

/// Transformer for applying algebraic transformations to expressions
#[derive(Debug, Default)]
pub struct MathTransformer {
    /// Applied rules during transformation
    applied_rules: Vec<SimplificationRule>,
}

impl MathTransformer {
    /// Create a new transformer
    pub fn new() -> Self {
        Self::default()
    }

    /// Simplify an expression by applying algebraic rules
    pub fn simplify(&mut self, expr: &Expr) -> Expr {
        self.applied_rules.clear();
        self.simplify_recursive(expr)
    }

    fn simplify_recursive(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Value(_) => expr.clone(),

            Expr::BinOp { op, left, right } => {
                let left_simplified = self.simplify_recursive(left);
                let right_simplified = self.simplify_recursive(right);

                // Apply simplification rules
                match op {
                    // Addition rules
                    MathOp::Add => {
                        // x + 0 = x
                        if self.is_zero(&right_simplified) {
                            self.applied_rules.push(SimplificationRule::ADDITIVE_IDENTITY);
                            return left_simplified;
                        }
                        // 0 + x = x
                        if self.is_zero(&left_simplified) {
                            self.applied_rules.push(SimplificationRule::ADDITIVE_IDENTITY);
                            return right_simplified;
                        }
                    }

                    // Subtraction rules
                    MathOp::Sub => {
                        // x - 0 = x
                        if self.is_zero(&right_simplified) {
                            self.applied_rules.push(SimplificationRule::ADDITIVE_IDENTITY);
                            return left_simplified;
                        }
                        // x - x = 0
                        if self.exprs_equal(&left_simplified, &right_simplified) {
                            self.applied_rules.push(SimplificationRule::ADDITIVE_INVERSE);
                            return Expr::Value(Value::Integer(0));
                        }
                    }

                    // Multiplication rules
                    MathOp::Mul => {
                        // x * 0 = 0
                        if self.is_zero(&left_simplified) || self.is_zero(&right_simplified) {
                            self.applied_rules.push(SimplificationRule::ZERO_PRODUCT);
                            return Expr::Value(Value::Integer(0));
                        }
                        // x * 1 = x
                        if self.is_one(&right_simplified) {
                            self.applied_rules.push(SimplificationRule::MULTIPLICATIVE_IDENTITY);
                            return left_simplified;
                        }
                        // 1 * x = x
                        if self.is_one(&left_simplified) {
                            self.applied_rules.push(SimplificationRule::MULTIPLICATIVE_IDENTITY);
                            return right_simplified;
                        }
                    }

                    // Division rules
                    MathOp::Div => {
                        // x / 1 = x
                        if self.is_one(&right_simplified) {
                            self.applied_rules.push(SimplificationRule::DIVISION_IDENTITY);
                            return left_simplified;
                        }
                        // 0 / x = 0 (when x != 0)
                        if self.is_zero(&left_simplified) && !self.is_zero(&right_simplified) {
                            self.applied_rules.push(SimplificationRule::ZERO_PRODUCT);
                            return Expr::Value(Value::Integer(0));
                        }
                    }

                    // Power rules
                    MathOp::Pow => {
                        // x ^ 0 = 1
                        if self.is_zero(&right_simplified) {
                            self.applied_rules.push(SimplificationRule::POWER_ZERO);
                            return Expr::Value(Value::Integer(1));
                        }
                        // x ^ 1 = x
                        if self.is_one(&right_simplified) {
                            self.applied_rules.push(SimplificationRule::POWER_ONE);
                            return left_simplified;
                        }
                    }

                    _ => {}
                }

                // No simplification applied, return with simplified children
                Expr::BinOp {
                    op: *op,
                    left: Box::new(left_simplified),
                    right: Box::new(right_simplified),
                }
            }

            Expr::UnaryOp { op, operand } => {
                let operand_simplified = self.simplify_recursive(operand);
                Expr::UnaryOp {
                    op: *op,
                    operand: Box::new(operand_simplified),
                }
            }

            Expr::Function { func, args } => {
                let args_simplified: Vec<Expr> = args
                    .iter()
                    .map(|a| self.simplify_recursive(a))
                    .collect();
                Expr::Function {
                    func: *func,
                    args: args_simplified,
                }
            }
        }
    }

    fn is_zero(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Value(Value::Integer(0)) => true,
            Expr::Value(Value::Float(f)) if *f == 0.0 => true,
            _ => false,
        }
    }

    fn is_one(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Value(Value::Integer(1)) => true,
            Expr::Value(Value::Float(f)) if (*f - 1.0).abs() < 1e-10 => true,
            _ => false,
        }
    }

    fn exprs_equal(&self, a: &Expr, b: &Expr) -> bool {
        // Simple structural equality check
        match (a, b) {
            (Expr::Value(va), Expr::Value(vb)) => {
                match (va, vb) {
                    (Value::Integer(ia), Value::Integer(ib)) => ia == ib,
                    (Value::Float(fa), Value::Float(fb)) => (fa - fb).abs() < 1e-10,
                    (Value::Symbol(sa), Value::Symbol(sb)) => sa == sb,
                    _ => false,
                }
            }
            _ => false, // More complex equality would require deeper comparison
        }
    }

    /// Get the rules applied during the last simplification
    pub fn applied_rules(&self) -> &[SimplificationRule] {
        &self.applied_rules
    }

    /// Constant folding - evaluate constant subexpressions
    pub fn fold_constants(&self, expr: &Expr, engine: &MathEngine) -> Expr {
        match expr {
            Expr::Value(_) => expr.clone(),

            Expr::BinOp { op, left, right } => {
                let left_folded = self.fold_constants(left, engine);
                let right_folded = self.fold_constants(right, engine);

                // If both operands are constants, evaluate
                if let (Expr::Value(Value::Integer(_)), Expr::Value(Value::Integer(_))) =
                    (&left_folded, &right_folded)
                {
                    let result_expr = Expr::BinOp {
                        op: *op,
                        left: Box::new(left_folded.clone()),
                        right: Box::new(right_folded.clone()),
                    };
                    if let Ok(result) = engine.evaluate(&result_expr) {
                        if result.fract() == 0.0 {
                            return Expr::Value(Value::Integer(result as i64));
                        } else {
                            return Expr::Value(Value::Float(result));
                        }
                    }
                }

                Expr::BinOp {
                    op: *op,
                    left: Box::new(left_folded),
                    right: Box::new(right_folded),
                }
            }

            Expr::UnaryOp { op, operand } => {
                let operand_folded = self.fold_constants(operand, engine);
                Expr::UnaryOp {
                    op: *op,
                    operand: Box::new(operand_folded),
                }
            }

            Expr::Function { func, args } => {
                let args_folded: Vec<Expr> = args
                    .iter()
                    .map(|a| self.fold_constants(a, engine))
                    .collect();
                Expr::Function {
                    func: *func,
                    args: args_folded,
                }
            }
        }
    }
}

// ============================================================================
// Math Brain
// ============================================================================

/// The math brain that learns graph transformations
#[derive(Default)]
pub struct MathBrain {
    /// The underlying engine for validation
    engine: MathEngine,
    /// The transformer for algebraic simplifications
    transformer: MathTransformer,
}

impl MathBrain {
    /// Create a new math brain
    pub fn new() -> Self {
        Self {
            engine: MathEngine::new(),
            transformer: MathTransformer::new(),
        }
    }

    /// Process an expression through the brain
    pub fn process(&self, expr: &Expr) -> MathGraphResult<MathGraph> {
        Ok(MathGraph::from_expr(expr))
    }

    /// Validate a graph against the engine
    pub fn validate(&self, graph: &MathGraph, expected: f64) -> MathGraphResult<bool> {
        let result = graph.evaluate(&self.engine)?;
        Ok((result - expected).abs() < 1e-10)
    }

    /// Get mutable access to the engine for symbol binding
    pub fn engine_mut(&mut self) -> &mut MathEngine {
        &mut self.engine
    }

    /// Extract intent from an expression
    pub fn extract_intent(&self, expr: &Expr) -> MathIntent {
        // Analyze the expression structure to determine intent
        match expr {
            Expr::Function { func, .. } => {
                match func {
                    MathFn::Sin | MathFn::Cos | MathFn::Tan |
                    MathFn::Sqrt | MathFn::Abs | MathFn::Log | MathFn::Ln |
                    MathFn::Exp | MathFn::Floor | MathFn::Ceil => MathIntent::Compute,
                    MathFn::Derive => MathIntent::Differentiate,
                    MathFn::Integrate => MathIntent::Integrate,
                }
            }
            Expr::BinOp { .. } | Expr::UnaryOp { .. } => {
                // Check if it contains symbols (needs evaluation/simplification)
                if self.contains_symbol(expr) {
                    MathIntent::Simplify
                } else {
                    MathIntent::Compute
                }
            }
            Expr::Value(Value::Symbol(_)) => MathIntent::Simplify,
            Expr::Value(_) => MathIntent::Compute,
        }
    }

    fn contains_symbol(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Value(Value::Symbol(_)) => true,
            Expr::Value(_) => false,
            Expr::BinOp { left, right, .. } => {
                self.contains_symbol(left) || self.contains_symbol(right)
            }
            Expr::UnaryOp { operand, .. } => self.contains_symbol(operand),
            Expr::Function { args, .. } => args.iter().any(|a| self.contains_symbol(a)),
        }
    }

    /// Simplify an expression
    pub fn simplify(&mut self, expr: &Expr) -> Expr {
        self.transformer.simplify(expr)
    }

    /// Fold constants in an expression
    pub fn fold_constants(&self, expr: &Expr) -> Expr {
        self.transformer.fold_constants(expr, &self.engine)
    }

    /// Create a math problem from components
    pub fn create_problem(&self, intent: MathIntent, expr: Expr) -> MathProblem {
        MathProblem {
            intent,
            expression: expr,
            variable: None,
            bounds: None,
            expected: None,
        }
    }

    /// Process a problem and return the result
    pub fn solve(&self, problem: &MathProblem) -> MathGraphResult<f64> {
        let graph = self.process(&problem.expression)?;
        graph.evaluate(&self.engine)
    }

    /// Normalize math text for domain processing
    /// Standardizes mathematical notation and operators
    fn normalize_math_text(&self, text: &str) -> String {
        // Normalize operator spacing
        let normalized = text
            .replace(" + ", "+")
            .replace(" - ", "-")
            .replace(" * ", "*")
            .replace(" / ", "/");

        // Normalize power notation
        let normalized = normalized
            .replace("**", "^")
            .replace(" ^ ", "^");

        // Normalize common symbols
        let normalized = normalized
            .replace("pi", "π")
            .replace("PI", "π")
            .replace("inf", "∞")
            .replace("infinity", "∞");

        // Trim whitespace
        normalized.trim().to_string()
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

impl std::fmt::Debug for MathBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MathBrain")
            .field("domain", &"mathematics")
            .finish()
    }
}

impl DomainBrain for MathBrain {
    fn domain_id(&self) -> &str {
        "math"
    }

    fn domain_name(&self) -> &str {
        "Mathematics"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn can_process(&self, input: &str) -> bool {
        // Check for mathematical expressions
        let math_keywords = [
            "calculate", "compute", "solve", "simplify", "evaluate",
            "derive", "differentiate", "integrate", "factor",
            "+", "-", "*", "/", "^", "=",
            "sin", "cos", "tan", "sqrt", "log", "exp",
        ];
        let lower = input.to_lowercase();
        math_keywords.iter().any(|kw| lower.contains(kw))
            || input.chars().any(|c| c.is_ascii_digit())
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        // Parse mathematical input into a graph
        // For now, create a simple graph from the input
        DagNN::from_text(input).map_err(|e| e.into())
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Convert core DagNN to math domain representation
        // Normalize mathematical notation
        let text = graph.to_text();

        // Apply math-specific normalization
        let normalized = self.normalize_math_text(&text);

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Convert math domain representation back to generic core format
        let text = graph.to_text();

        // Remove any math-specific annotations
        let cleaned = text
            .lines()
            .filter(|line| !line.trim().starts_with("@math:"))
            .collect::<Vec<_>>()
            .join("\n");

        if cleaned != text {
            DagNN::from_text(&cleaned).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        // Check for empty graph
        if graph.input_nodes().is_empty() {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Empty math expression graph".to_string(),
                node: None,
            });
            return Ok(issues);
        }

        // Get text representation for parsing-based validation
        let text = graph.to_text();

        // Validate balanced parentheses
        issues.extend(self.validate_parentheses(&text));

        // Validate operator usage
        issues.extend(self.validate_operators(&text));

        // Check for division by zero
        issues.extend(self.validate_division(&text));

        // Validate function arity
        issues.extend(self.validate_functions(&text));

        // Validate numeric literals
        issues.extend(self.validate_literals(&text));

        Ok(issues)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        // Try to evaluate the graph as a mathematical expression
        // For now, return the graph text representation
        let text = graph.to_text();

        // Try to parse and evaluate if possible
        if let Ok(result) = self.parse_and_evaluate(&text) {
            Ok(ExecutionResult::Numeric(result))
        } else {
            Ok(ExecutionResult::Text(text))
        }
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        // Return simplification rules
        vec![
            DomainRule {
                id: 0,
                domain: "math".to_string(),
                name: "Zero Addition".to_string(),
                description: "x + 0 = x".to_string(),
                category: "simplification".to_string(),
            },
            DomainRule {
                id: 1,
                domain: "math".to_string(),
                name: "Zero Multiplication".to_string(),
                description: "x * 0 = 0".to_string(),
                category: "simplification".to_string(),
            },
            DomainRule {
                id: 2,
                domain: "math".to_string(),
                name: "One Multiplication".to_string(),
                description: "x * 1 = x".to_string(),
                category: "simplification".to_string(),
            },
            DomainRule {
                id: 3,
                domain: "math".to_string(),
                name: "Power Zero".to_string(),
                description: "x^0 = 1".to_string(),
                category: "simplification".to_string(),
            },
            DomainRule {
                id: 4,
                domain: "math".to_string(),
                name: "Power One".to_string(),
                description: "x^1 = x".to_string(),
                category: "simplification".to_string(),
            },
            DomainRule {
                id: 5,
                domain: "math".to_string(),
                name: "Constant Folding".to_string(),
                description: "Evaluate constant subexpressions".to_string(),
                category: "optimization".to_string(),
            },
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        match rule_id {
            0 => self.apply_zero_addition(graph),
            1 => self.apply_zero_multiplication(graph),
            2 => self.apply_one_multiplication(graph),
            3 => self.apply_power_zero(graph),
            4 => self.apply_power_one(graph),
            5 => self.apply_constant_folding(graph),
            _ => Err(grapheme_core::DomainError::InvalidInput(
                format!("Unknown rule ID: {}", rule_id)
            )),
        }
    }

    fn generate_examples(&self, count: usize) -> Vec<DomainExample> {
        let mut examples = Vec::with_capacity(count);

        for i in 0..count {
            // Generate simple arithmetic examples
            let a = (i % 10) as i32 + 1;
            let b = ((i / 10) % 10) as i32 + 1;
            let ops = ["+", "-", "*"];
            let op = ops[i % 3];

            let input = format!("{} {} {}", a, op, b);
            let output = match op {
                "+" => format!("{}", a + b),
                "-" => format!("{}", a - b),
                "*" => format!("{}", a * b),
                _ => unreachable!(),
            };

            if let (Ok(input_graph), Ok(output_graph)) = (
                DagNN::from_text(&input),
                DagNN::from_text(&output),
            ) {
                examples.push(DomainExample {
                    input: input_graph,
                    output: output_graph,
                    domain: "math".to_string(),
                    difficulty: ((i % 5) + 1) as u8,
                });
            }
        }

        examples
    }
}

// ============================================================================
// Transform Helper Methods
// ============================================================================

impl MathBrain {
    /// Rule 0: Zero Addition - x + 0 = x
    fn apply_zero_addition(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Apply x + 0 = x and 0 + x = x
        let normalized = text
            .replace(" + 0", "")
            .replace("+ 0", "")
            .replace("0 + ", "")
            .replace("0 +", "");

        if normalized != text && !normalized.is_empty() {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 1: Zero Multiplication - x * 0 = 0
    fn apply_zero_multiplication(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Check for multiplication by zero patterns
        if text.contains(" * 0") || text.contains("* 0") ||
           text.contains("0 * ") || text.contains("0 *") {
            DagNN::from_text("0").map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 2: One Multiplication - x * 1 = x
    fn apply_one_multiplication(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Apply x * 1 = x and 1 * x = x
        let normalized = text
            .replace(" * 1", "")
            .replace("* 1", "")
            .replace("1 * ", "")
            .replace("1 *", "");

        if normalized != text && !normalized.is_empty() {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 3: Power Zero - x^0 = 1
    fn apply_power_zero(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Check for power of zero
        if text.contains("^0") || text.contains("^ 0") {
            DagNN::from_text("1").map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 4: Power One - x^1 = x
    fn apply_power_one(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Remove power of one
        let normalized = text
            .replace("^1", "")
            .replace("^ 1", "");

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 5: Constant Folding - evaluate constant subexpressions
    fn apply_constant_folding(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Try to evaluate as a simple expression
        if let Ok(result) = self.parse_and_evaluate(&text) {
            if result.fract() == 0.0 {
                DagNN::from_text(&format!("{}", result as i64)).map_err(|e| e.into())
            } else {
                DagNN::from_text(&format!("{}", result)).map_err(|e| e.into())
            }
        } else {
            Ok(graph.clone())
        }
    }
}

impl MathBrain {
    /// Parse and evaluate a simple expression string
    fn parse_and_evaluate(&self, text: &str) -> Result<f64, MathGraphError> {
        // Simple parser for basic arithmetic
        let trimmed = text.trim();

        // Try to parse as a number first
        if let Ok(n) = trimmed.parse::<f64>() {
            return Ok(n);
        }

        // Try simple binary operations
        for (op_str, op) in [("+", MathOp::Add), ("-", MathOp::Sub), ("*", MathOp::Mul), ("/", MathOp::Div)] {
            if let Some(idx) = trimmed.rfind(op_str) {
                if idx > 0 && idx < trimmed.len() - 1 {
                    let left = trimmed[..idx].trim().parse::<f64>();
                    let right = trimmed[idx + 1..].trim().parse::<f64>();

                    if let (Ok(l), Ok(r)) = (left, right) {
                        let result = match op {
                            MathOp::Add => l + r,
                            MathOp::Sub => l - r,
                            MathOp::Mul => l * r,
                            MathOp::Div => {
                                if r == 0.0 {
                                    return Err(MathGraphError::InvalidStructure("Division by zero".to_string()));
                                }
                                l / r
                            }
                            _ => return Err(MathGraphError::InvalidStructure("Unsupported operation".to_string())),
                        };
                        return Ok(result);
                    }
                }
            }
        }

        Err(MathGraphError::InvalidStructure(format!("Cannot parse: {}", text)))
    }
}

// ============================================================================
// Validation Helper Methods
// ============================================================================

impl MathBrain {
    /// Validate balanced parentheses
    fn validate_parentheses(&self, text: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let mut depth = 0i32;
        let mut position = 0usize;

        for (i, ch) in text.chars().enumerate() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        position = i;
                    }
                }
                _ => {}
            }
        }

        if depth > 0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                message: format!("Unbalanced parentheses: {} unclosed '('", depth),
                node: None,
            });
        } else if depth < 0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                message: format!("Unbalanced parentheses: extra ')' at position {}", position),
                node: None,
            });
        }

        issues
    }

    /// Validate operator usage (no consecutive operators, proper placement)
    fn validate_operators(&self, text: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let binary_operators = ['+', '*', '/', '^', '%']; // Exclude unary minus

        let chars: Vec<char> = text.chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            if binary_operators.contains(&ch) {
                // Check for consecutive binary operators (excluding unary minus after operator)
                if i > 0 {
                    let prev = chars[i - 1];
                    if binary_operators.contains(&prev) {
                        issues.push(ValidationIssue {
                            severity: ValidationSeverity::Error,
                            message: format!("Invalid consecutive operators '{}{}' at position {}", prev, ch, i),
                            node: None,
                        });
                    }
                }

                // Check for operator at start (only * / ^ % are errors)
                if i == 0 && ch != '+' && ch != '-' {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: format!("Expression cannot start with '{}'", ch),
                        node: None,
                    });
                }

                // Check for operator at end
                if i == chars.len() - 1 {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: format!("Expression cannot end with '{}'", ch),
                        node: None,
                    });
                }

                // Check for operator after open paren (except unary minus)
                if i > 0 && chars[i - 1] == '(' && ch != '-' && ch != '+' {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: format!("Invalid operator '{}' after '('", ch),
                        node: None,
                    });
                }

                // Check for operator before close paren
                if i + 1 < chars.len() && chars[i + 1] == ')' {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: format!("Invalid operator '{}' before ')'", ch),
                        node: None,
                    });
                }
            }
        }

        issues
    }

    /// Check for potential division by zero
    fn validate_division(&self, text: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Check for literal division by zero patterns
        let patterns = [
            "/0", "/ 0", "/0.0", "/ 0.0",
            "/ (0)", "/(0)", "/ ( 0 )", "/( 0 )",
        ];

        for pattern in patterns {
            if text.contains(pattern) {
                // Check if followed by more digits (would not be zero)
                let idx = text.find(pattern).unwrap();
                let after_pattern = idx + pattern.len();
                if after_pattern >= text.len() || !text.chars().nth(after_pattern).map(|c| c.is_ascii_digit()).unwrap_or(false) {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: "Division by zero detected".to_string(),
                        node: None,
                    });
                    break;
                }
            }
        }

        issues
    }

    /// Validate function arity
    fn validate_functions(&self, text: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Single-argument functions
        let single_arg_fns = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                              "sqrt", "cbrt", "exp", "log", "ln", "abs", "floor", "ceil", "round"];

        // Two-argument functions
        let two_arg_fns = ["pow", "log_base", "max", "min", "atan2"];

        for func in single_arg_fns {
            // Find function calls like "sin(...)"
            let pattern = format!("{}(", func);
            if let Some(start) = text.find(&pattern) {
                let after = start + pattern.len();
                let rest = &text[after..];
                // Count commas until matching paren
                let mut depth = 1;
                let mut commas = 0;
                for ch in rest.chars() {
                    match ch {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 { break; }
                        }
                        ',' if depth == 1 => commas += 1,
                        _ => {}
                    }
                }
                if commas > 0 {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: format!("Function '{}' expects 1 argument, got {}", func, commas + 1),
                        node: None,
                    });
                }
            }
        }

        for func in two_arg_fns {
            let pattern = format!("{}(", func);
            if let Some(start) = text.find(&pattern) {
                let after = start + pattern.len();
                let rest = &text[after..];
                let mut depth = 1;
                let mut commas = 0;
                for ch in rest.chars() {
                    match ch {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 { break; }
                        }
                        ',' if depth == 1 => commas += 1,
                        _ => {}
                    }
                }
                if commas != 1 {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: format!("Function '{}' expects 2 arguments, got {}", func, commas + 1),
                        node: None,
                    });
                }
            }
        }

        issues
    }

    /// Validate numeric literals are well-formed
    fn validate_literals(&self, text: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Check for multiple decimal points in a number
        let mut in_number = false;
        let mut decimal_count = 0;
        let mut number_start = 0;

        for (i, ch) in text.chars().enumerate() {
            if ch.is_ascii_digit() {
                if !in_number {
                    in_number = true;
                    number_start = i;
                    decimal_count = 0;
                }
            } else if ch == '.' && in_number {
                decimal_count += 1;
                if decimal_count > 1 {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: format!("Invalid numeric literal with multiple decimal points starting at position {}", number_start),
                        node: None,
                    });
                }
            } else if in_number {
                in_number = false;
                decimal_count = 0;
            }
        }

        // Check for leading zeros followed by digits (not valid in many contexts)
        for (i, _) in text.match_indices("00") {
            if i > 0 {
                let prev = text.chars().nth(i - 1);
                if prev.is_some_and(|c| !c.is_ascii_digit() && c != '.') {
                    // This is a standalone 00 - warn about it
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Info,
                        message: format!("Leading zeros in number at position {}", i),
                        node: None,
                    });
                }
            }
        }

        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_from_expr() {
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };

        let graph = MathGraph::from_expr(&expr);
        assert_eq!(graph.node_count(), 3); // op + 2 values
        assert_eq!(graph.edge_count(), 2); // 2 operand edges
    }

    #[test]
    fn test_graph_roundtrip() {
        let original = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Integer(2))),
                right: Box::new(Expr::Value(Value::Integer(3))),
            }),
            right: Box::new(Expr::Value(Value::Integer(4))),
        };

        let graph = MathGraph::from_expr(&original);
        let roundtrip = graph.to_expr().unwrap();

        let engine = MathEngine::new();
        assert_eq!(
            engine.evaluate(&original).unwrap(),
            engine.evaluate(&roundtrip).unwrap()
        );
    }

    #[test]
    fn test_brain_validation() {
        let brain = MathBrain::new();

        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };

        let graph = brain.process(&expr).unwrap();
        assert!(brain.validate(&graph, 5.0).unwrap());
        assert!(!brain.validate(&graph, 6.0).unwrap());
    }

    // Tests for backend-002: MathIntent, Simplification, Transformer

    #[test]
    fn test_math_intent_compute() {
        let brain = MathBrain::new();

        // Pure arithmetic should be Compute
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };
        assert_eq!(brain.extract_intent(&expr), MathIntent::Compute);
    }

    #[test]
    fn test_math_intent_simplify() {
        let brain = MathBrain::new();

        // Expression with symbol should be Simplify
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Symbol("x".into()))),
            right: Box::new(Expr::Value(Value::Integer(0))),
        };
        assert_eq!(brain.extract_intent(&expr), MathIntent::Simplify);
    }

    #[test]
    fn test_simplify_additive_identity() {
        let mut brain = MathBrain::new();

        // x + 0 should simplify to x
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Symbol("x".into()))),
            right: Box::new(Expr::Value(Value::Integer(0))),
        };

        let simplified = brain.simplify(&expr);
        assert_eq!(simplified, Expr::Value(Value::Symbol("x".into())));
    }

    #[test]
    fn test_simplify_multiplicative_identity() {
        let mut brain = MathBrain::new();

        // x * 1 should simplify to x
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::Value(Value::Symbol("y".into()))),
            right: Box::new(Expr::Value(Value::Integer(1))),
        };

        let simplified = brain.simplify(&expr);
        assert_eq!(simplified, Expr::Value(Value::Symbol("y".into())));
    }

    #[test]
    fn test_simplify_zero_product() {
        let mut brain = MathBrain::new();

        // x * 0 should simplify to 0
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::Value(Value::Symbol("x".into()))),
            right: Box::new(Expr::Value(Value::Integer(0))),
        };

        let simplified = brain.simplify(&expr);
        assert_eq!(simplified, Expr::Value(Value::Integer(0)));
    }

    #[test]
    fn test_simplify_power_zero() {
        let mut brain = MathBrain::new();

        // x ^ 0 should simplify to 1
        let expr = Expr::BinOp {
            op: MathOp::Pow,
            left: Box::new(Expr::Value(Value::Symbol("x".into()))),
            right: Box::new(Expr::Value(Value::Integer(0))),
        };

        let simplified = brain.simplify(&expr);
        assert_eq!(simplified, Expr::Value(Value::Integer(1)));
    }

    #[test]
    fn test_simplify_power_one() {
        let mut brain = MathBrain::new();

        // x ^ 1 should simplify to x
        let expr = Expr::BinOp {
            op: MathOp::Pow,
            left: Box::new(Expr::Value(Value::Symbol("x".into()))),
            right: Box::new(Expr::Value(Value::Integer(1))),
        };

        let simplified = brain.simplify(&expr);
        assert_eq!(simplified, Expr::Value(Value::Symbol("x".into())));
    }

    #[test]
    fn test_fold_constants() {
        let brain = MathBrain::new();

        // (2 + 3) * x should fold to 5 * x
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Integer(2))),
                right: Box::new(Expr::Value(Value::Integer(3))),
            }),
            right: Box::new(Expr::Value(Value::Symbol("x".into()))),
        };

        let folded = brain.fold_constants(&expr);

        // The left side should be folded to 5
        if let Expr::BinOp { left, .. } = folded {
            assert_eq!(*left, Expr::Value(Value::Integer(5)));
        } else {
            panic!("Expected BinOp");
        }
    }

    #[test]
    fn test_math_problem() {
        let brain = MathBrain::new();

        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(10))),
            right: Box::new(Expr::Value(Value::Integer(5))),
        };

        let problem = brain.create_problem(MathIntent::Compute, expr);
        let result = brain.solve(&problem).unwrap();

        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_nested_simplification() {
        let mut brain = MathBrain::new();

        // (x + 0) * 1 should simplify to x
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Symbol("x".into()))),
                right: Box::new(Expr::Value(Value::Integer(0))),
            }),
            right: Box::new(Expr::Value(Value::Integer(1))),
        };

        let simplified = brain.simplify(&expr);
        assert_eq!(simplified, Expr::Value(Value::Symbol("x".into())));
    }

    // DomainBrain implementation tests
    #[test]
    fn test_domain_brain_id() {
        let brain = MathBrain::new();
        assert_eq!(brain.domain_id(), "math");
        assert_eq!(brain.domain_name(), "Mathematics");
        assert_eq!(brain.version(), "0.1.0");
    }

    #[test]
    fn test_domain_brain_can_process() {
        let brain = MathBrain::new();
        assert!(brain.can_process("calculate 2 + 2"));
        assert!(brain.can_process("sin(x)"));
        assert!(brain.can_process("5 * 3"));
        assert!(!brain.can_process("hello world")); // No math keywords or digits
    }

    #[test]
    fn test_domain_brain_parse() {
        let brain = MathBrain::new();
        let result = brain.parse("2 + 3");
        assert!(result.is_ok());
    }

    #[test]
    fn test_domain_brain_get_rules() {
        let brain = MathBrain::new();
        let rules = brain.get_rules();
        assert_eq!(rules.len(), 6);
        assert_eq!(rules[0].name, "Zero Addition");
        assert_eq!(rules[0].domain, "math");
    }

    #[test]
    fn test_domain_brain_generate_examples() {
        let brain = MathBrain::new();
        let examples = brain.generate_examples(10);
        assert_eq!(examples.len(), 10);
        for example in &examples {
            assert_eq!(example.domain, "math");
            assert!(example.difficulty >= 1 && example.difficulty <= 5);
        }
    }

    #[test]
    fn test_domain_brain_execute() {
        let brain = MathBrain::new();
        let graph = DagNN::from_text("42").unwrap();
        let result = brain.execute(&graph);
        assert!(result.is_ok());
    }

    // Math validation tests
    #[test]
    fn test_validate_balanced_parentheses() {
        let brain = MathBrain::new();

        // Valid parentheses
        let graph = DagNN::from_text("(2 + 3) * 4").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        let paren_errors: Vec<_> = issues.iter().filter(|i| i.message.contains("parentheses")).collect();
        assert!(paren_errors.is_empty());

        // Unbalanced - unclosed paren
        let graph = DagNN::from_text("(2 + 3").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        assert!(issues.iter().any(|i| i.message.contains("Unbalanced")));

        // Unbalanced - extra close paren
        let graph = DagNN::from_text("2 + 3)").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        assert!(issues.iter().any(|i| i.message.contains("Unbalanced")));
    }

    #[test]
    fn test_validate_operators() {
        let brain = MathBrain::new();

        // Valid operators
        let graph = DagNN::from_text("2 + 3 * 4").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        let op_errors: Vec<_> = issues.iter().filter(|i| i.message.contains("operator")).collect();
        assert!(op_errors.is_empty());

        // Expression ending with operator
        let graph = DagNN::from_text("2 +").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        assert!(issues.iter().any(|i| i.message.contains("end with")));

        // Expression starting with * (invalid)
        let graph = DagNN::from_text("* 2").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        assert!(issues.iter().any(|i| i.message.contains("start with")));
    }

    #[test]
    fn test_validate_division_by_zero() {
        let brain = MathBrain::new();

        // Valid division
        let graph = DagNN::from_text("10 / 2").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        let div_errors: Vec<_> = issues.iter().filter(|i| i.message.contains("Division by zero")).collect();
        assert!(div_errors.is_empty());

        // Division by zero
        let graph = DagNN::from_text("10 / 0").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        assert!(issues.iter().any(|i| i.message.contains("Division by zero")));
    }

    #[test]
    fn test_validate_function_arity() {
        let brain = MathBrain::new();

        // Valid single-arg function
        let graph = DagNN::from_text("sin(x)").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        let arity_errors: Vec<_> = issues.iter().filter(|i| i.message.contains("expects")).collect();
        assert!(arity_errors.is_empty());

        // Invalid: sin with 2 args
        let graph = DagNN::from_text("sin(x, y)").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        assert!(issues.iter().any(|i| i.message.contains("sin") && i.message.contains("expects 1")));

        // Valid two-arg function
        let graph = DagNN::from_text("pow(x, 2)").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        let arity_errors: Vec<_> = issues.iter().filter(|i| i.message.contains("pow")).collect();
        assert!(arity_errors.is_empty());

        // Invalid: pow with 1 arg
        let graph = DagNN::from_text("pow(x)").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        assert!(issues.iter().any(|i| i.message.contains("pow") && i.message.contains("expects 2")));
    }

    #[test]
    fn test_validate_literals() {
        let brain = MathBrain::new();

        // Valid literals
        let graph = DagNN::from_text("3.14").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        let literal_errors: Vec<_> = issues.iter().filter(|i| i.message.contains("decimal")).collect();
        assert!(literal_errors.is_empty());

        // Invalid: multiple decimal points
        let graph = DagNN::from_text("3.14.15").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        assert!(issues.iter().any(|i| i.message.contains("multiple decimal")));
    }

    #[test]
    fn test_validate_empty_graph() {
        let brain = MathBrain::new();
        let graph = DagNN::new();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        assert!(issues.iter().any(|i| i.message.contains("Empty")));
    }

    #[test]
    fn test_validate_valid_complex_expression() {
        let brain = MathBrain::new();

        // Complex but valid expression
        let graph = DagNN::from_text("(2 + 3) * sin(x) / (y - 1)").unwrap();
        let issues = DomainBrain::validate(&brain, &graph).unwrap();
        // Should have no errors (only possible infos about leading zeros, which shouldn't be here)
        let errors: Vec<_> = issues.iter().filter(|i| matches!(i.severity, ValidationSeverity::Error)).collect();
        assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);
    }
}
