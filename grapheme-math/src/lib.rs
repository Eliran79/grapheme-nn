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

use grapheme_core::ValidationIssue;
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

    /// Validate the mathematical graph structure
    ///
    /// Checks for:
    /// - Operator arity (binary ops need 2 inputs, unary need 1)
    /// - Function arity (correct number of arguments)
    /// - Division by zero possibilities
    /// - Disconnected nodes
    /// - Missing root
    pub fn validate_structure(&self) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Check for empty graph
        if self.graph.node_count() == 0 {
            issues.push(ValidationIssue::warning("Math graph is empty"));
            return issues;
        }

        // Check for missing root
        if self.root.is_none() {
            issues.push(ValidationIssue::error("Math graph has no root node"));
            return issues;
        }

        let root = self.root.unwrap();

        // Validate each node
        for node_idx in self.graph.node_indices() {
            let node = &self.graph[node_idx];
            let incoming_edges: Vec<_> = self.graph.edges_directed(node_idx, petgraph::Direction::Incoming).collect();
            let outgoing_edges: Vec<_> = self.graph.edges(node_idx).collect();

            match node {
                MathNode::Operator(op) => {
                    // Binary operators need exactly 2 children (via LeftOperand and RightOperand)
                    let left_count = outgoing_edges.iter()
                        .filter(|e| matches!(e.weight(), MathEdge::LeftOperand))
                        .count();
                    let right_count = outgoing_edges.iter()
                        .filter(|e| matches!(e.weight(), MathEdge::RightOperand))
                        .count();

                    if left_count != 1 {
                        issues.push(ValidationIssue::error(format!(
                            "Operator {:?} at node {} has {} left operands, expected 1",
                            op, node_idx.index(), left_count
                        )).with_location(node_idx.index()));
                    }
                    if right_count != 1 {
                        issues.push(ValidationIssue::error(format!(
                            "Operator {:?} at node {} has {} right operands, expected 1",
                            op, node_idx.index(), right_count
                        )).with_location(node_idx.index()));
                    }

                    // Check for division by zero
                    if matches!(op, MathOp::Div) {
                        for edge in &outgoing_edges {
                            if matches!(edge.weight(), MathEdge::RightOperand) {
                                let target = edge.target();
                                let is_zero = match &self.graph[target] {
                                    MathNode::Integer(0) => true,
                                    MathNode::Float(f) if *f == 0.0 => true,
                                    _ => false,
                                };
                                if is_zero {
                                    issues.push(ValidationIssue::warning(
                                        "Division by zero detected"
                                    ).with_location(node_idx.index()));
                                }
                            }
                        }
                    }
                }
                MathNode::Function(func) => {
                    // Count function arguments
                    let arg_count = outgoing_edges.iter()
                        .filter(|e| matches!(e.weight(), MathEdge::Argument(_)))
                        .count();

                    let expected_args = match func {
                        MathFn::Sin | MathFn::Cos | MathFn::Tan |
                        MathFn::Sqrt | MathFn::Abs | MathFn::Log | MathFn::Ln |
                        MathFn::Exp | MathFn::Floor | MathFn::Ceil => 1,
                        MathFn::Derive | MathFn::Integrate => 2, // expression and variable
                    };

                    if arg_count != expected_args {
                        issues.push(ValidationIssue::error(format!(
                            "Function {:?} at node {} has {} arguments, expected {}",
                            func, node_idx.index(), arg_count, expected_args
                        )).with_location(node_idx.index()));
                    }

                    // Check for sqrt of negative
                    if matches!(func, MathFn::Sqrt) {
                        for edge in &outgoing_edges {
                            if matches!(edge.weight(), MathEdge::Argument(0)) {
                                if let MathNode::Integer(n) = &self.graph[edge.target()] {
                                    if *n < 0 {
                                        issues.push(ValidationIssue::warning(
                                            "Square root of negative number"
                                        ).with_location(node_idx.index()));
                                    }
                                }
                                if let MathNode::Float(f) = &self.graph[edge.target()] {
                                    if *f < 0.0 {
                                        issues.push(ValidationIssue::warning(
                                            "Square root of negative number"
                                        ).with_location(node_idx.index()));
                                    }
                                }
                            }
                        }
                    }

                    // Check for log of non-positive
                    if matches!(func, MathFn::Log | MathFn::Ln) {
                        for edge in &outgoing_edges {
                            if matches!(edge.weight(), MathEdge::Argument(0)) {
                                if let MathNode::Integer(n) = &self.graph[edge.target()] {
                                    if *n <= 0 {
                                        issues.push(ValidationIssue::warning(
                                            "Logarithm of non-positive number"
                                        ).with_location(node_idx.index()));
                                    }
                                }
                                if let MathNode::Float(f) = &self.graph[edge.target()] {
                                    if *f <= 0.0 {
                                        issues.push(ValidationIssue::warning(
                                            "Logarithm of non-positive number"
                                        ).with_location(node_idx.index()));
                                    }
                                }
                            }
                        }
                    }
                }
                MathNode::Integer(_) | MathNode::Float(_) | MathNode::Symbol(_) => {
                    // Leaf nodes should have no outgoing edges to operands
                    let operand_edges = outgoing_edges.iter()
                        .filter(|e| matches!(e.weight(), MathEdge::LeftOperand | MathEdge::RightOperand | MathEdge::Argument(_)))
                        .count();
                    if operand_edges > 0 {
                        issues.push(ValidationIssue::error(format!(
                            "Value node at {} has {} unexpected operand edges",
                            node_idx.index(), operand_edges
                        )).with_location(node_idx.index()));
                    }
                }
                MathNode::Result => {
                    // Result nodes should have no outgoing edges
                    if !outgoing_edges.is_empty() {
                        issues.push(ValidationIssue::warning(format!(
                            "Result node at {} has {} outgoing edges",
                            node_idx.index(), outgoing_edges.len()
                        )).with_location(node_idx.index()));
                    }
                }
            }

            // Check for disconnected nodes (except root)
            if node_idx != root && incoming_edges.is_empty() && outgoing_edges.is_empty() {
                issues.push(ValidationIssue::warning(format!(
                    "Node at {} is disconnected from the graph",
                    node_idx.index()
                )).with_location(node_idx.index()));
            }
        }

        issues
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
    #[allow(clippy::only_used_in_recursion)]
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
#[derive(Debug, Default)]
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

    #[allow(clippy::only_used_in_recursion)]
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

    /// Parse a math expression string into an Expr
    /// Supports: integers, floats, basic operators (+, -, *, /, ^), parentheses
    pub fn parse_expr(&self, input: &str) -> MathGraphResult<Expr> {
        parse_math_expr(input)
    }

    /// Format an expression as a string
    pub fn format_expr(&self, expr: &Expr) -> String {
        format_expr(expr)
    }
}

// ============================================================================
// Expression Parser (simple recursive descent)
// ============================================================================

/// Parse a math expression string into an Expr
fn parse_math_expr(input: &str) -> MathGraphResult<Expr> {
    let input = input.trim();
    if input.is_empty() {
        return Err(MathGraphError::InvalidStructure("Empty expression".into()));
    }

    let tokens = tokenize(input)?;
    let mut pos = 0;
    parse_additive(&tokens, &mut pos)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Symbol(String),
    Op(char),
    LParen,
    RParen,
    Func(String),
}

fn tokenize(input: &str) -> MathGraphResult<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c.is_whitespace() {
            i += 1;
            continue;
        }

        if c.is_ascii_digit() || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let num_str: String = chars[start..i].iter().collect();
            let num: f64 = num_str.parse().map_err(|_| {
                MathGraphError::InvalidStructure(format!("Invalid number: {}", num_str))
            })?;
            tokens.push(Token::Number(num));
            continue;
        }

        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let name: String = chars[start..i].iter().collect();
            // Check if it's a function
            let is_func = matches!(name.to_lowercase().as_str(),
                "sin" | "cos" | "tan" | "sqrt" | "abs" | "log" | "ln" | "exp" | "floor" | "ceil");
            if is_func {
                tokens.push(Token::Func(name));
            } else {
                tokens.push(Token::Symbol(name));
            }
            continue;
        }

        match c {
            '+' | '-' | '*' | '/' | '^' | '%' => {
                tokens.push(Token::Op(c));
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            _ => {
                return Err(MathGraphError::InvalidStructure(format!(
                    "Unexpected character: '{}'", c
                )));
            }
        }
    }

    Ok(tokens)
}

fn parse_additive(tokens: &[Token], pos: &mut usize) -> MathGraphResult<Expr> {
    let mut left = parse_multiplicative(tokens, pos)?;

    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Op('+') => {
                *pos += 1;
                let right = parse_multiplicative(tokens, pos)?;
                left = Expr::add(left, right);
            }
            Token::Op('-') => {
                *pos += 1;
                let right = parse_multiplicative(tokens, pos)?;
                left = Expr::sub(left, right);
            }
            _ => break,
        }
    }

    Ok(left)
}

fn parse_multiplicative(tokens: &[Token], pos: &mut usize) -> MathGraphResult<Expr> {
    let mut left = parse_power(tokens, pos)?;

    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Op('*') => {
                *pos += 1;
                let right = parse_power(tokens, pos)?;
                left = Expr::mul(left, right);
            }
            Token::Op('/') => {
                *pos += 1;
                let right = parse_power(tokens, pos)?;
                left = Expr::div(left, right);
            }
            Token::Op('%') => {
                *pos += 1;
                let right = parse_power(tokens, pos)?;
                left = Expr::BinOp {
                    op: MathOp::Mod,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            }
            _ => break,
        }
    }

    Ok(left)
}

fn parse_power(tokens: &[Token], pos: &mut usize) -> MathGraphResult<Expr> {
    let base = parse_unary(tokens, pos)?;

    if *pos < tokens.len() {
        if let Token::Op('^') = &tokens[*pos] {
            *pos += 1;
            let exp = parse_power(tokens, pos)?; // Right associative
            return Ok(Expr::pow(base, exp));
        }
    }

    Ok(base)
}

fn parse_unary(tokens: &[Token], pos: &mut usize) -> MathGraphResult<Expr> {
    if *pos < tokens.len() {
        if let Token::Op('-') = &tokens[*pos] {
            *pos += 1;
            let operand = parse_unary(tokens, pos)?;
            return Ok(Expr::neg(operand));
        }
    }
    parse_primary(tokens, pos)
}

fn parse_primary(tokens: &[Token], pos: &mut usize) -> MathGraphResult<Expr> {
    if *pos >= tokens.len() {
        return Err(MathGraphError::InvalidStructure("Unexpected end of expression".into()));
    }

    match &tokens[*pos] {
        Token::Number(n) => {
            let n = *n;
            *pos += 1;
            if n.fract() == 0.0 && n.abs() < i64::MAX as f64 {
                Ok(Expr::int(n as i64))
            } else {
                Ok(Expr::float(n))
            }
        }
        Token::Symbol(s) => {
            let s = s.clone();
            *pos += 1;
            Ok(Expr::symbol(s))
        }
        Token::Func(name) => {
            let func = match name.to_lowercase().as_str() {
                "sin" => MathFn::Sin,
                "cos" => MathFn::Cos,
                "tan" => MathFn::Tan,
                "sqrt" => MathFn::Sqrt,
                "abs" => MathFn::Abs,
                "log" => MathFn::Log,
                "ln" => MathFn::Ln,
                "exp" => MathFn::Exp,
                "floor" => MathFn::Floor,
                "ceil" => MathFn::Ceil,
                _ => return Err(MathGraphError::InvalidStructure(format!("Unknown function: {}", name))),
            };
            *pos += 1;

            // Expect '('
            if *pos >= tokens.len() || tokens[*pos] != Token::LParen {
                return Err(MathGraphError::InvalidStructure("Expected '(' after function name".into()));
            }
            *pos += 1;

            let arg = parse_additive(tokens, pos)?;

            // Expect ')'
            if *pos >= tokens.len() || tokens[*pos] != Token::RParen {
                return Err(MathGraphError::InvalidStructure("Expected ')' after function argument".into()));
            }
            *pos += 1;

            Ok(Expr::func(func, vec![arg]))
        }
        Token::LParen => {
            *pos += 1;
            let expr = parse_additive(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != Token::RParen {
                return Err(MathGraphError::InvalidStructure("Mismatched parentheses".into()));
            }
            *pos += 1;
            Ok(expr)
        }
        Token::RParen => {
            Err(MathGraphError::InvalidStructure("Unexpected ')'".into()))
        }
        Token::Op(c) => {
            Err(MathGraphError::InvalidStructure(format!("Unexpected operator: '{}'", c)))
        }
    }
}

/// Format an expression as a string
fn format_expr(expr: &Expr) -> String {
    match expr {
        Expr::Value(v) => match v {
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => format!("{}", f),
            Value::Symbol(s) => s.clone(),
            Value::Rational(n, d) => format!("{}/{}", n, d),
        },
        Expr::BinOp { op, left, right } => {
            let op_str = match op {
                MathOp::Add => "+",
                MathOp::Sub => "-",
                MathOp::Mul => "*",
                MathOp::Div => "/",
                MathOp::Pow => "^",
                MathOp::Mod => "%",
                MathOp::Neg => "-",
            };
            format!("({} {} {})", format_expr(left), op_str, format_expr(right))
        }
        Expr::UnaryOp { op, operand } => {
            match op {
                MathOp::Neg => format!("(-{})", format_expr(operand)),
                _ => format!("({}{})", format_math_op(op), format_expr(operand)),
            }
        }
        Expr::Function { func, args } => {
            let func_name = match func {
                MathFn::Sin => "sin",
                MathFn::Cos => "cos",
                MathFn::Tan => "tan",
                MathFn::Sqrt => "sqrt",
                MathFn::Abs => "abs",
                MathFn::Log => "log",
                MathFn::Ln => "ln",
                MathFn::Exp => "exp",
                MathFn::Floor => "floor",
                MathFn::Ceil => "ceil",
                MathFn::Derive => "derive",
                MathFn::Integrate => "integrate",
            };
            let args_str: Vec<String> = args.iter().map(format_expr).collect();
            format!("{}({})", func_name, args_str.join(", "))
        }
    }
}

fn format_math_op(op: &MathOp) -> &'static str {
    match op {
        MathOp::Add => "+",
        MathOp::Sub => "-",
        MathOp::Mul => "*",
        MathOp::Div => "/",
        MathOp::Pow => "^",
        MathOp::Mod => "%",
        MathOp::Neg => "-",
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

use grapheme_core::{DagNN, DomainBrain, DomainError, DomainResult, DomainRule, DomainExample,
                    ExecutionResult, ValidationSeverity, EdgeType, Edge, NodeType};
use grapheme_brain_common::{GraphAutoencoder, LatentGraph, AutoencoderError};

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
        // Can process if it looks like a math expression
        let input = input.trim();
        if input.is_empty() {
            return false;
        }

        // Check for math-like content: numbers, operators, common math symbols
        let has_numbers = input.chars().any(|c| c.is_ascii_digit());
        let has_ops = input.chars().any(|c| matches!(c, '+' | '-' | '*' | '/' | '^' | '=' | '(' | ')'));
        let has_math_funcs = ["sin", "cos", "tan", "sqrt", "log", "ln", "exp"]
            .iter()
            .any(|f| input.to_lowercase().contains(f));

        has_numbers || has_ops || has_math_funcs
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        // Parse the expression and convert to DagNN
        let expr = self.parse_expr(input).map_err(|e| DomainError::ParseError(e.to_string()))?;
        let math_graph = MathGraph::from_expr(&expr);

        // Convert MathGraph to DagNN
        math_graph_to_dagnn(&math_graph)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Strengthen semantic edges (math relationships) and skip edges (distant relations)
        let mut result = graph.clone();

        let edges_to_update: Vec<_> = result.graph.edge_references()
            .filter_map(|edge| {
                match edge.weight().edge_type {
                    EdgeType::Semantic => {
                        // Semantic relationships (e.g., operand relationships)
                        Some((edge.source(), edge.target(), edge.weight().weight * 1.2))
                    }
                    EdgeType::Skip => {
                        // Skip connections for distant operands
                        Some((edge.source(), edge.target(), edge.weight().weight * 0.9))
                    }
                    _ => None
                }
            })
            .collect();

        for (src, tgt, new_weight) in edges_to_update {
            if let Some(edge) = result.graph.find_edge(src, tgt) {
                result.graph[edge].weight = new_weight.clamp(0.1, 2.0);
            }
        }

        let _ = result.update_topology();
        Ok(result)
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let mut result = graph.clone();
        result.prune_weak_edges(0.05);
        let _ = result.update_topology();
        Ok(result)
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        if graph.node_count() == 0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                message: "Empty math graph".to_string(),
                location: None,
            });
        }

        // Check for very deep nesting (potential stack overflow in evaluation)
        if graph.node_count() > 1000 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Very large expression graph".to_string(),
                location: None,
            });
        }

        Ok(issues)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        // Try to evaluate the graph as a math expression
        let text = graph.to_text();

        // Try parsing and evaluating
        match self.parse_expr(&text) {
            Ok(expr) => {
                match self.engine.evaluate(&expr) {
                    Ok(result) => Ok(ExecutionResult::Text(format!("{}", result))),
                    Err(_) => Ok(ExecutionResult::Text(self.format_expr(&expr))),
                }
            }
            Err(_) => Ok(ExecutionResult::Text(text)),
        }
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        vec![
            DomainRule::new(0, "Simplify", "Apply algebraic simplification rules"),
            DomainRule::new(1, "Fold Constants", "Evaluate constant subexpressions"),
            DomainRule::new(2, "Differentiate", "Take derivative with respect to x"),
            DomainRule::new(3, "Factor", "Factor expression"),
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        let text = graph.to_text();
        let expr = self.parse_expr(&text).map_err(|e| DomainError::TransformError(e.to_string()))?;

        let mut brain = MathBrain::new();
        let transformed = match rule_id {
            0 => brain.simplify(&expr),
            1 => brain.fold_constants(&expr),
            2 => {
                let symbolic = grapheme_engine::SymbolicEngine::new();
                symbolic.differentiate(&expr, "x")
            }
            _ => return Err(DomainError::InvalidInput(format!("Unknown rule ID: {}", rule_id))),
        };

        let math_graph = MathGraph::from_expr(&transformed);
        math_graph_to_dagnn(&math_graph)
    }

    fn generate_examples(&self, count: usize) -> Vec<DomainExample> {
        let generator = grapheme_engine::TrainingDataGenerator::new();
        let examples = generator.generate_arithmetic(count, 2);

        examples.iter().map(|(expr, result)| {
            let input = self.format_expr(expr);
            let output = format!("{}", result);
            DomainExample::new(input, output)
                .with_metadata("domain", "math")
        }).collect()
    }

    fn node_types(&self) -> Vec<NodeType> {
        vec![
            NodeType::Input('0'),  // For operands (digit placeholder)
            NodeType::Hidden, // For operators
            NodeType::Output, // For result
        ]
    }
}

// ============================================================================
// GraphAutoencoder Implementation
// ============================================================================

impl GraphAutoencoder for MathBrain {
    fn encode(&self, input: &str) -> Result<LatentGraph, AutoencoderError> {
        let graph = self.parse(input).map_err(|e| AutoencoderError::EncodingError(e.to_string()))?;
        Ok(LatentGraph::new("math", graph))
    }

    fn decode(&self, latent: &LatentGraph) -> Result<String, AutoencoderError> {
        self.validate_latent(latent)?;

        // Convert DagNN back to text representation
        Ok(latent.graph.to_text())
    }

    fn reconstruction_loss(&self, original: &str, reconstructed: &str) -> f32 {
        // For math, we can use semantic comparison
        // First try exact match
        if original.trim() == reconstructed.trim() {
            return 0.0;
        }

        // Try parsing both and comparing evaluation
        let orig_expr = self.parse_expr(original);
        let recon_expr = self.parse_expr(reconstructed);

        match (orig_expr, recon_expr) {
            (Ok(orig), Ok(recon)) => {
                // If both are constant expressions, compare values
                if orig.is_constant() && recon.is_constant() {
                    match (self.engine.evaluate(&orig), self.engine.evaluate(&recon)) {
                        (Ok(v1), Ok(v2)) => {
                            if (v1 - v2).abs() < 1e-10 {
                                return 0.0; // Semantically equivalent
                            }
                            // Relative difference
                            let max_val = v1.abs().max(v2.abs()).max(1.0);
                            ((v1 - v2).abs() / max_val).min(1.0) as f32
                        }
                        _ => 0.5, // Evaluation failed
                    }
                } else {
                    // Symbolic expressions - use default string comparison
                    let max_len = original.len().max(reconstructed.len()).max(1);
                    let matching: usize = original.chars()
                        .zip(reconstructed.chars())
                        .filter(|(a, b)| a == b)
                        .count();
                    1.0 - (matching as f32 / max_len as f32)
                }
            }
            _ => {
                // Parsing failed - use default string comparison
                let max_len = original.len().max(reconstructed.len()).max(1);
                let matching: usize = original.chars()
                    .zip(reconstructed.chars())
                    .filter(|(a, b)| a == b)
                    .count();
                1.0 - (matching as f32 / max_len as f32)
            }
        }
    }
}

// ============================================================================
// Helper: Convert MathGraph to DagNN
// ============================================================================

fn math_graph_to_dagnn(math_graph: &MathGraph) -> DomainResult<DagNN> {
    use std::collections::HashMap;

    let mut dagnn = DagNN::new();
    let mut node_map: HashMap<NodeIndex, petgraph::graph::NodeIndex> = HashMap::new();

    // Add all nodes - we'll use a simple strategy:
    // - Leaf nodes (values) -> use first char of their string representation as input char
    // - Root node -> output node
    // - Others -> hidden nodes
    for node_idx in math_graph.graph.node_indices() {
        let math_node = &math_graph.graph[node_idx];
        let is_root = math_graph.root == Some(node_idx);
        let is_leaf = math_graph.graph.neighbors(node_idx).count() == 0;

        let dagnn_idx = if is_root {
            dagnn.add_output()
        } else if is_leaf {
            // Leaf nodes - use the first char of their representation
            let ch = match math_node {
                MathNode::Integer(i) => {
                    let s = i.to_string();
                    s.chars().next().unwrap_or('0')
                }
                MathNode::Float(f) => {
                    let s = format!("{}", f);
                    s.chars().next().unwrap_or('0')
                }
                MathNode::Symbol(s) => s.chars().next().unwrap_or('x'),
                _ => '?',
            };
            dagnn.add_character(ch, node_idx.index())
        } else {
            dagnn.add_hidden()
        };

        node_map.insert(node_idx, dagnn_idx);
    }

    // Add all edges
    for edge in math_graph.graph.edge_references() {
        let src = node_map[&edge.source()];
        let tgt = node_map[&edge.target()];
        let weight = match edge.weight() {
            MathEdge::LeftOperand => 1.0,
            MathEdge::RightOperand => 0.9,
            MathEdge::Argument(i) => 1.0 - (*i as f32 * 0.1).min(0.5),
            MathEdge::Produces => 1.0,
        };
        dagnn.add_edge(src, tgt, Edge::new(weight, EdgeType::Semantic));
    }

    let _ = dagnn.update_topology();
    Ok(dagnn)
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

    // Tests for backend-083: MathGraph validation

    #[test]
    fn test_validate_valid_graph() {
        // Valid expression: 2 + 3
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };

        let graph = MathGraph::from_expr(&expr);
        let issues = graph.validate_structure();
        assert!(issues.is_empty(), "Valid graph should have no issues: {:?}", issues);
    }

    #[test]
    fn test_validate_empty_graph() {
        let graph = MathGraph::new();
        let issues = graph.validate_structure();
        assert!(!issues.is_empty(), "Empty graph should have issues");
        assert!(issues.iter().any(|i| i.message.contains("empty")));
    }

    #[test]
    fn test_validate_division_by_zero() {
        // Expression: 10 / 0
        let expr = Expr::BinOp {
            op: MathOp::Div,
            left: Box::new(Expr::Value(Value::Integer(10))),
            right: Box::new(Expr::Value(Value::Integer(0))),
        };

        let graph = MathGraph::from_expr(&expr);
        let issues = graph.validate_structure();
        assert!(issues.iter().any(|i| i.message.contains("Division by zero")),
            "Should detect division by zero: {:?}", issues);
    }

    #[test]
    fn test_validate_sqrt_negative() {
        // Expression: sqrt(-4)
        let expr = Expr::Function {
            func: MathFn::Sqrt,
            args: vec![Expr::Value(Value::Integer(-4))],
        };

        let graph = MathGraph::from_expr(&expr);
        let issues = graph.validate_structure();
        assert!(issues.iter().any(|i| i.message.contains("Square root of negative")),
            "Should detect sqrt of negative: {:?}", issues);
    }

    #[test]
    fn test_validate_log_non_positive() {
        // Expression: ln(0)
        let expr = Expr::Function {
            func: MathFn::Ln,
            args: vec![Expr::Value(Value::Integer(0))],
        };

        let graph = MathGraph::from_expr(&expr);
        let issues = graph.validate_structure();
        assert!(issues.iter().any(|i| i.message.contains("Logarithm of non-positive")),
            "Should detect log of non-positive: {:?}", issues);
    }

    #[test]
    fn test_validate_complex_expression() {
        // Valid complex expression: (x + 2) * 3
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Symbol("x".into()))),
                right: Box::new(Expr::Value(Value::Integer(2))),
            }),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };

        let graph = MathGraph::from_expr(&expr);
        let issues = graph.validate_structure();
        assert!(issues.is_empty(), "Valid complex expression should have no issues: {:?}", issues);
    }

    #[test]
    fn test_validate_sin_function() {
        // Expression: sin(x)
        let expr = Expr::Function {
            func: MathFn::Sin,
            args: vec![Expr::Value(Value::Symbol("x".into()))],
        };

        let graph = MathGraph::from_expr(&expr);
        let issues = graph.validate_structure();
        assert!(issues.is_empty(), "Valid sin function should have no issues: {:?}", issues);
    }
}
