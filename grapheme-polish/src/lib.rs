//! # grapheme-polish
//!
//! Layer 2: Polish notation intermediate representation.
//!
//! This crate provides:
//! - Unambiguous Polish (prefix) notation
//! - Bidirectional conversion: text <-> graph
//! - Direct graph mapping for expressions
//! - Optimization passes
//!
//! Polish notation is ideal for graph representation because:
//! - No parentheses needed (unambiguous)
//! - Natural tree/graph structure
//! - Easy to parse and generate

// Allow &self in recursive methods for API consistency
#![allow(clippy::only_used_in_recursion)]

use grapheme_engine::{Expr, MathEngine, MathFn, MathOp, Value};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors during Polish notation processing
#[derive(Error, Debug)]
pub enum PolishError {
    #[error("Parse error at position {position}: {message}")]
    ParseError { position: usize, message: String },
    #[error("Unexpected end of input")]
    UnexpectedEof,
    #[error("Unknown operator: {0}")]
    UnknownOperator(String),
    #[error("Unknown function: {0}")]
    UnknownFunction(String),
    #[error("Invalid token: {0}")]
    InvalidToken(String),
}

/// Result type for Polish operations
pub type PolishResult<T> = Result<T, PolishError>;

/// A token in Polish notation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Token {
    Number(f64),
    Symbol(String),
    Operator(MathOp),
    Function(MathFn),
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
}

// ============================================================================
// Graph Mapping Types
// ============================================================================

/// A node in the Polish expression graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GraphNode {
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Symbol (variable)
    Symbol(String),
    /// Rational number
    Rational(i64, i64),
    /// Binary operator
    Operator(MathOp),
    /// Function application
    Function(MathFn),
}

impl GraphNode {
    /// Check if this is a value node (leaf)
    pub fn is_value(&self) -> bool {
        matches!(
            self,
            GraphNode::Integer(_)
                | GraphNode::Float(_)
                | GraphNode::Symbol(_)
                | GraphNode::Rational(_, _)
        )
    }

    /// Check if this is an operator node
    pub fn is_operator(&self) -> bool {
        matches!(self, GraphNode::Operator(_))
    }

    /// Check if this is a function node
    pub fn is_function(&self) -> bool {
        matches!(self, GraphNode::Function(_))
    }
}

/// Edge type in the expression graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphEdge {
    /// Left operand (for binary ops)
    Left,
    /// Right operand (for binary ops)
    Right,
    /// Operand (for unary ops)
    Operand,
    /// Argument with position (for functions)
    Arg(usize),
}

/// A graph representation of a Polish expression
#[derive(Debug, Clone)]
pub struct PolishGraph {
    /// The underlying directed graph
    pub graph: DiGraph<GraphNode, GraphEdge>,
    /// The root node of the expression
    pub root: Option<NodeIndex>,
}

impl Default for PolishGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl PolishGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            root: None,
        }
    }

    /// Create a graph from an expression
    pub fn from_expr(expr: &Expr) -> Self {
        let mut pg = Self::new();
        pg.root = Some(pg.add_expr(expr));
        pg
    }

    /// Add an expression to the graph, returning the root node index
    fn add_expr(&mut self, expr: &Expr) -> NodeIndex {
        match expr {
            Expr::Value(v) => {
                let node = match v {
                    Value::Integer(i) => GraphNode::Integer(*i),
                    Value::Float(f) => GraphNode::Float(*f),
                    Value::Symbol(s) => GraphNode::Symbol(s.clone()),
                    Value::Rational(n, d) => GraphNode::Rational(*n, *d),
                };
                self.graph.add_node(node)
            }
            Expr::BinOp { op, left, right } => {
                let op_node = self.graph.add_node(GraphNode::Operator(*op));
                let left_node = self.add_expr(left);
                let right_node = self.add_expr(right);
                self.graph.add_edge(op_node, left_node, GraphEdge::Left);
                self.graph.add_edge(op_node, right_node, GraphEdge::Right);
                op_node
            }
            Expr::UnaryOp { op, operand } => {
                let op_node = self.graph.add_node(GraphNode::Operator(*op));
                let operand_node = self.add_expr(operand);
                self.graph.add_edge(op_node, operand_node, GraphEdge::Operand);
                op_node
            }
            Expr::Function { func, args } => {
                let func_node = self.graph.add_node(GraphNode::Function(*func));
                for (i, arg) in args.iter().enumerate() {
                    let arg_node = self.add_expr(arg);
                    self.graph.add_edge(func_node, arg_node, GraphEdge::Arg(i));
                }
                func_node
            }
        }
    }

    /// Convert the graph back to an expression
    pub fn to_expr(&self) -> Option<Expr> {
        self.root.map(|r| self.node_to_expr(r))
    }

    /// Convert a node and its children to an expression
    fn node_to_expr(&self, node: NodeIndex) -> Expr {
        let graph_node = &self.graph[node];

        match graph_node {
            GraphNode::Integer(i) => Expr::Value(Value::Integer(*i)),
            GraphNode::Float(f) => Expr::Value(Value::Float(*f)),
            GraphNode::Symbol(s) => Expr::Value(Value::Symbol(s.clone())),
            GraphNode::Rational(n, d) => Expr::Value(Value::Rational(*n, *d)),
            GraphNode::Operator(op) => {
                let edges: Vec<_> = self.graph.edges(node).collect();

                // Check if unary (has Operand edge) or binary (has Left/Right edges)
                let has_operand = edges.iter().any(|e| *e.weight() == GraphEdge::Operand);

                if has_operand {
                    let operand_idx = edges
                        .iter()
                        .find(|e| *e.weight() == GraphEdge::Operand)
                        .map(|e| e.target())
                        .expect("Operand edge missing for unary operator (malformed graph)");
                    Expr::UnaryOp {
                        op: *op,
                        operand: Box::new(self.node_to_expr(operand_idx)),
                    }
                } else {
                    // Binary operator - both Left and Right edges must exist
                    let left_idx = edges
                        .iter()
                        .find(|e| *e.weight() == GraphEdge::Left)
                        .map(|e| e.target());
                    let right_idx = edges
                        .iter()
                        .find(|e| *e.weight() == GraphEdge::Right)
                        .map(|e| e.target());

                    match (left_idx, right_idx) {
                        (Some(left), Some(right)) => Expr::BinOp {
                            op: *op,
                            left: Box::new(self.node_to_expr(left)),
                            right: Box::new(self.node_to_expr(right)),
                        },
                        (Some(left), None) => {
                            // Treat as unary if only left operand exists
                            Expr::UnaryOp {
                                op: *op,
                                operand: Box::new(self.node_to_expr(left)),
                            }
                        }
                        _ => {
                            // Fallback: treat operator as a symbol value if no edges
                            Expr::Value(Value::Symbol(format!("{:?}", op)))
                        }
                    }
                }
            }
            GraphNode::Function(func) => {
                let mut args: Vec<(usize, Expr)> = self
                    .graph
                    .edges(node)
                    .filter_map(|e| {
                        if let GraphEdge::Arg(i) = e.weight() {
                            Some((*i, self.node_to_expr(e.target())))
                        } else {
                            None
                        }
                    })
                    .collect();
                args.sort_by_key(|(i, _)| *i);
                Expr::Function {
                    func: *func,
                    args: args.into_iter().map(|(_, e)| e).collect(),
                }
            }
        }
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get all nodes of a specific type
    pub fn nodes_of_type(&self, predicate: impl Fn(&GraphNode) -> bool) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&idx| predicate(&self.graph[idx]))
            .collect()
    }

    /// Get all value (leaf) nodes
    pub fn leaf_nodes(&self) -> Vec<NodeIndex> {
        self.nodes_of_type(|n| n.is_value())
    }

    /// Get all operator nodes
    pub fn operator_nodes(&self) -> Vec<NodeIndex> {
        self.nodes_of_type(|n| n.is_operator())
    }
}

// ============================================================================
// Optimization Passes
// ============================================================================

/// Trait for optimization passes on Polish expressions
pub trait OptimizationPass {
    /// Apply the optimization pass to an expression
    fn optimize(&self, expr: &Expr) -> Expr;

    /// Get the name of this optimization pass
    fn name(&self) -> &'static str;
}

/// Constant folding optimization - evaluates constant subexpressions
#[derive(Debug, Default)]
pub struct ConstantFolding {
    engine: MathEngine,
}

impl ConstantFolding {
    /// Create a new constant folding pass
    pub fn new() -> Self {
        Self {
            engine: MathEngine::new(),
        }
    }

    /// Check if an expression is purely constant (no symbols)
    fn is_constant(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Value(Value::Symbol(_)) => false,
            Expr::Value(_) => true,
            Expr::BinOp { left, right, .. } => self.is_constant(left) && self.is_constant(right),
            Expr::UnaryOp { operand, .. } => self.is_constant(operand),
            Expr::Function { args, .. } => args.iter().all(|a| self.is_constant(a)),
        }
    }
}

impl OptimizationPass for ConstantFolding {
    fn optimize(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::Value(_) => expr.clone(),
            Expr::BinOp { op, left, right } => {
                let opt_left = self.optimize(left);
                let opt_right = self.optimize(right);
                let new_expr = Expr::BinOp {
                    op: *op,
                    left: Box::new(opt_left),
                    right: Box::new(opt_right),
                };
                // Try to fold if both operands are constant
                if self.is_constant(&new_expr) {
                    if let Ok(result) = self.engine.evaluate(&new_expr) {
                        return Expr::Value(Value::Float(result));
                    }
                }
                new_expr
            }
            Expr::UnaryOp { op, operand } => {
                let opt_operand = self.optimize(operand);
                let new_expr = Expr::UnaryOp {
                    op: *op,
                    operand: Box::new(opt_operand),
                };
                if self.is_constant(&new_expr) {
                    if let Ok(result) = self.engine.evaluate(&new_expr) {
                        return Expr::Value(Value::Float(result));
                    }
                }
                new_expr
            }
            Expr::Function { func, args } => {
                let opt_args: Vec<_> = args.iter().map(|a| self.optimize(a)).collect();
                let new_expr = Expr::Function {
                    func: *func,
                    args: opt_args,
                };
                if self.is_constant(&new_expr) {
                    if let Ok(result) = self.engine.evaluate(&new_expr) {
                        return Expr::Value(Value::Float(result));
                    }
                }
                new_expr
            }
        }
    }

    fn name(&self) -> &'static str {
        "constant_folding"
    }
}

/// Identity elimination - removes identity operations
/// x + 0 = x, x * 1 = x, x - 0 = x, x / 1 = x, x ^ 1 = x, x ^ 0 = 1
#[derive(Debug, Default)]
pub struct IdentityElimination;

impl IdentityElimination {
    /// Create a new identity elimination pass
    pub fn new() -> Self {
        Self
    }

    fn is_zero(expr: &Expr) -> bool {
        match expr {
            Expr::Value(Value::Integer(0)) => true,
            Expr::Value(Value::Float(f)) => *f == 0.0,
            _ => false,
        }
    }

    fn is_one(expr: &Expr) -> bool {
        match expr {
            Expr::Value(Value::Integer(1)) => true,
            Expr::Value(Value::Float(f)) => *f == 1.0,
            _ => false,
        }
    }
}

impl OptimizationPass for IdentityElimination {
    fn optimize(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::Value(_) => expr.clone(),
            Expr::BinOp { op, left, right } => {
                let opt_left = self.optimize(left);
                let opt_right = self.optimize(right);

                match op {
                    // x + 0 = x, 0 + x = x
                    MathOp::Add => {
                        if Self::is_zero(&opt_right) {
                            return opt_left;
                        }
                        if Self::is_zero(&opt_left) {
                            return opt_right;
                        }
                    }
                    // x - 0 = x
                    MathOp::Sub => {
                        if Self::is_zero(&opt_right) {
                            return opt_left;
                        }
                    }
                    // x * 1 = x, 1 * x = x, x * 0 = 0, 0 * x = 0
                    MathOp::Mul => {
                        if Self::is_one(&opt_right) {
                            return opt_left;
                        }
                        if Self::is_one(&opt_left) {
                            return opt_right;
                        }
                        if Self::is_zero(&opt_left) || Self::is_zero(&opt_right) {
                            return Expr::Value(Value::Integer(0));
                        }
                    }
                    // x / 1 = x
                    MathOp::Div => {
                        if Self::is_one(&opt_right) {
                            return opt_left;
                        }
                    }
                    // x ^ 1 = x, x ^ 0 = 1
                    MathOp::Pow => {
                        if Self::is_one(&opt_right) {
                            return opt_left;
                        }
                        if Self::is_zero(&opt_right) {
                            return Expr::Value(Value::Integer(1));
                        }
                    }
                    _ => {}
                }

                Expr::BinOp {
                    op: *op,
                    left: Box::new(opt_left),
                    right: Box::new(opt_right),
                }
            }
            Expr::UnaryOp { op, operand } => Expr::UnaryOp {
                op: *op,
                operand: Box::new(self.optimize(operand)),
            },
            Expr::Function { func, args } => Expr::Function {
                func: *func,
                args: args.iter().map(|a| self.optimize(a)).collect(),
            },
        }
    }

    fn name(&self) -> &'static str {
        "identity_elimination"
    }
}

/// Common subexpression elimination - identifies and reuses repeated subexpressions
///
/// This pass identifies subexpressions that appear multiple times in an expression tree
/// and replaces them with variable references to avoid redundant computation.
///
/// # Example
/// Input: `(* (+ a b) (+ a b))`
/// Output: `(* _cse0 _cse0)` with bindings `{_cse0: (+ a b)}`
///
/// The bindings are stored in `CommonSubexpressionElimination::bindings()` after optimization.
#[derive(Debug, Default)]
pub struct CommonSubexpressionElimination {
    /// Minimum number of occurrences before CSE is applied
    min_occurrences: usize,
    /// Bindings from CSE variables to their expressions (populated after optimize())
    bindings: std::cell::RefCell<Vec<(String, Expr)>>,
}

impl CommonSubexpressionElimination {
    /// Create a new CSE pass with default settings (min 2 occurrences)
    pub fn new() -> Self {
        Self {
            min_occurrences: 2,
            bindings: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Create a CSE pass with custom minimum occurrence threshold
    pub fn with_min_occurrences(min: usize) -> Self {
        Self {
            min_occurrences: min.max(2), // At least 2 to make CSE worthwhile
            bindings: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Get the bindings produced by the last optimize() call
    /// Returns pairs of (variable_name, expression)
    pub fn bindings(&self) -> Vec<(String, Expr)> {
        self.bindings.borrow().clone()
    }

    /// Clear bindings from previous optimization
    pub fn clear_bindings(&self) {
        self.bindings.borrow_mut().clear();
    }

    /// Count occurrences of all subexpressions
    /// Uses Polish notation as canonical form for hashing (Expr contains f64 which can't Hash)
    fn count_subexpressions(expr: &Expr, counts: &mut std::collections::HashMap<String, usize>) {
        let key = expr_to_polish(expr);

        // Only count non-trivial expressions (not simple values)
        if !matches!(expr, Expr::Value(_)) {
            *counts.entry(key).or_insert(0) += 1;
        }

        // Recursively count children
        match expr {
            Expr::Value(_) => {}
            Expr::BinOp { left, right, .. } => {
                Self::count_subexpressions(left, counts);
                Self::count_subexpressions(right, counts);
            }
            Expr::UnaryOp { operand, .. } => {
                Self::count_subexpressions(operand, counts);
            }
            Expr::Function { args, .. } => {
                for arg in args {
                    Self::count_subexpressions(arg, counts);
                }
            }
        }
    }

    /// Replace repeated subexpressions with variable references
    fn replace_common(
        &self,
        expr: &Expr,
        candidates: &std::collections::HashMap<String, String>,
    ) -> Expr {
        let key = expr_to_polish(expr);

        // If this expression should be replaced with a CSE variable, do it
        if let Some(var_name) = candidates.get(&key) {
            return Expr::Value(Value::Symbol(var_name.clone()));
        }

        // Otherwise recurse
        match expr {
            Expr::Value(_) => expr.clone(),
            Expr::BinOp { op, left, right } => Expr::BinOp {
                op: *op,
                left: Box::new(self.replace_common(left, candidates)),
                right: Box::new(self.replace_common(right, candidates)),
            },
            Expr::UnaryOp { op, operand } => Expr::UnaryOp {
                op: *op,
                operand: Box::new(self.replace_common(operand, candidates)),
            },
            Expr::Function { func, args } => Expr::Function {
                func: *func,
                args: args
                    .iter()
                    .map(|a| self.replace_common(a, candidates))
                    .collect(),
            },
        }
    }
}

impl OptimizationPass for CommonSubexpressionElimination {
    fn optimize(&self, expr: &Expr) -> Expr {
        // Clear previous bindings
        self.clear_bindings();

        // Phase 1: Count all subexpression occurrences
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        Self::count_subexpressions(expr, &mut counts);

        // Phase 2: Identify candidates for CSE (appear >= min_occurrences times)
        // Sort by Polish notation length (longer = more complex = higher priority for CSE)
        let mut candidates: Vec<(String, usize)> = counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_occurrences)
            .collect();
        candidates.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // Longer (more complex) first

        // If no candidates, return unchanged
        if candidates.is_empty() {
            return expr.clone();
        }

        // Phase 3: Assign CSE variable names and build replacement map
        let mut parser = PolishParser::new();
        let mut replacement_map: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        let mut bindings = Vec::new();

        for (i, (polish, _)) in candidates.iter().enumerate() {
            let var_name = format!("_cse{}", i);

            // Parse the Polish notation back to an Expr for storage
            if let Ok(original_expr) = parser.parse(polish) {
                bindings.push((var_name.clone(), original_expr));
                replacement_map.insert(polish.clone(), var_name);
            }
        }

        // Store bindings
        *self.bindings.borrow_mut() = bindings;

        // Phase 4: Replace common subexpressions with variables
        self.replace_common(expr, &replacement_map)
    }

    fn name(&self) -> &'static str {
        "common_subexpression_elimination"
    }
}

/// An optimizer that chains multiple passes
#[derive(Debug, Default)]
pub struct Optimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl Optimizer {
    /// Create a new optimizer
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Create an optimizer with default passes
    pub fn with_defaults() -> Self {
        let mut opt = Self::new();
        opt.add_pass(Box::new(IdentityElimination::new()));
        opt.add_pass(Box::new(ConstantFolding::new()));
        opt
    }

    /// Add an optimization pass
    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }

    /// Run all optimization passes
    pub fn optimize(&self, expr: &Expr) -> Expr {
        let mut result = expr.clone();
        for pass in &self.passes {
            result = pass.optimize(&result);
        }
        result
    }

    /// Run optimization passes until no changes occur (fixed point)
    pub fn optimize_fixpoint(&self, expr: &Expr) -> Expr {
        let mut result = expr.clone();
        loop {
            let optimized = self.optimize(&result);
            if expr_to_polish(&optimized) == expr_to_polish(&result) {
                break;
            }
            result = optimized;
        }
        result
    }
}

impl std::fmt::Debug for dyn OptimizationPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OptimizationPass({})", self.name())
    }
}

/// Parser for Polish notation expressions
#[derive(Debug, Default)]
pub struct PolishParser {
    tokens: Vec<Token>,
    position: usize,
}

impl PolishParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokenize a Polish notation string
    pub fn tokenize(&mut self, input: &str) -> PolishResult<Vec<Token>> {
        let mut tokens = Vec::new();
        let mut chars = input.chars().peekable();

        while let Some(&ch) = chars.peek() {
            match ch {
                ' ' | '\t' | '\n' => {
                    chars.next();
                }
                '(' => {
                    tokens.push(Token::OpenParen);
                    chars.next();
                }
                ')' => {
                    tokens.push(Token::CloseParen);
                    chars.next();
                }
                '[' => {
                    tokens.push(Token::OpenBracket);
                    chars.next();
                }
                ']' => {
                    tokens.push(Token::CloseBracket);
                    chars.next();
                }
                '+' => {
                    tokens.push(Token::Operator(MathOp::Add));
                    chars.next();
                }
                '-' => {
                    chars.next();
                    // Check if it's a negative number
                    if chars.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        let num = self.read_number(&mut chars, true)?;
                        tokens.push(Token::Number(num));
                    } else {
                        tokens.push(Token::Operator(MathOp::Sub));
                    }
                }
                '*' => {
                    tokens.push(Token::Operator(MathOp::Mul));
                    chars.next();
                }
                '/' => {
                    tokens.push(Token::Operator(MathOp::Div));
                    chars.next();
                }
                '^' => {
                    tokens.push(Token::Operator(MathOp::Pow));
                    chars.next();
                }
                '%' => {
                    tokens.push(Token::Operator(MathOp::Mod));
                    chars.next();
                }
                '0'..='9' | '.' => {
                    let num = self.read_number(&mut chars, false)?;
                    tokens.push(Token::Number(num));
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    let ident = self.read_identifier(&mut chars);
                    tokens.push(self.classify_identifier(&ident)?);
                }
                _ => {
                    return Err(PolishError::InvalidToken(ch.to_string()));
                }
            }
        }

        self.tokens = tokens.clone();
        Ok(tokens)
    }

    fn read_number(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
        negative: bool,
    ) -> PolishResult<f64> {
        let mut num_str = if negative {
            "-".to_string()
        } else {
            String::new()
        };

        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                num_str.push(ch);
                chars.next();
            } else {
                break;
            }
        }

        num_str
            .parse()
            .map_err(|_| PolishError::InvalidToken(num_str))
    }

    fn read_identifier(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
        let mut ident = String::new();

        while let Some(&ch) = chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                chars.next();
            } else {
                break;
            }
        }

        ident
    }

    fn classify_identifier(&self, ident: &str) -> PolishResult<Token> {
        match ident.to_lowercase().as_str() {
            "sin" => Ok(Token::Function(MathFn::Sin)),
            "cos" => Ok(Token::Function(MathFn::Cos)),
            "tan" => Ok(Token::Function(MathFn::Tan)),
            "log" => Ok(Token::Function(MathFn::Log)),
            "ln" => Ok(Token::Function(MathFn::Ln)),
            "exp" => Ok(Token::Function(MathFn::Exp)),
            "sqrt" => Ok(Token::Function(MathFn::Sqrt)),
            "abs" => Ok(Token::Function(MathFn::Abs)),
            "floor" => Ok(Token::Function(MathFn::Floor)),
            "ceil" => Ok(Token::Function(MathFn::Ceil)),
            "derive" => Ok(Token::Function(MathFn::Derive)),
            "integrate" => Ok(Token::Function(MathFn::Integrate)),
            _ => Ok(Token::Symbol(ident.to_string())),
        }
    }

    /// Parse tokens into an expression tree
    pub fn parse(&mut self, input: &str) -> PolishResult<Expr> {
        self.tokenize(input)?;
        self.position = 0;
        self.parse_expr()
    }

    fn parse_expr(&mut self) -> PolishResult<Expr> {
        let token = self.next_token()?;

        match token {
            Token::Number(n) => Ok(Expr::Value(Value::Float(n))),
            Token::Symbol(s) => Ok(Expr::Value(Value::Symbol(s))),
            Token::OpenParen => {
                let expr = self.parse_sexp()?;
                self.expect_token(&Token::CloseParen)?;
                Ok(expr)
            }
            _ => Err(PolishError::ParseError {
                position: self.position,
                message: format!("Unexpected token: {:?}", token),
            }),
        }
    }

    fn parse_sexp(&mut self) -> PolishResult<Expr> {
        let token = self.next_token()?;

        match token {
            Token::Operator(op) => {
                let left = self.parse_expr()?;
                let right = self.parse_expr()?;
                Ok(Expr::BinOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }
            Token::Function(func) => {
                let mut args = Vec::new();
                while self.peek_token() != Some(&Token::CloseParen) {
                    args.push(self.parse_expr()?);
                }
                Ok(Expr::Function { func, args })
            }
            _ => Err(PolishError::ParseError {
                position: self.position,
                message: format!("Expected operator or function, got {:?}", token),
            }),
        }
    }

    fn next_token(&mut self) -> PolishResult<Token> {
        if self.position >= self.tokens.len() {
            return Err(PolishError::UnexpectedEof);
        }
        let token = self.tokens[self.position].clone();
        self.position += 1;
        Ok(token)
    }

    fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn expect_token(&mut self, expected: &Token) -> PolishResult<()> {
        let token = self.next_token()?;
        if &token == expected {
            Ok(())
        } else {
            Err(PolishError::ParseError {
                position: self.position,
                message: format!("Expected {:?}, got {:?}", expected, token),
            })
        }
    }
}

/// Convert an expression to Polish notation string
pub fn expr_to_polish(expr: &Expr) -> String {
    match expr {
        Expr::Value(Value::Integer(i)) => i.to_string(),
        Expr::Value(Value::Float(f)) => f.to_string(),
        Expr::Value(Value::Symbol(s)) => s.clone(),
        Expr::Value(Value::Rational(n, d)) => format!("(/ {} {})", n, d),
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
            format!(
                "({} {} {})",
                op_str,
                expr_to_polish(left),
                expr_to_polish(right)
            )
        }
        Expr::UnaryOp { op, operand } => {
            let op_str = match op {
                MathOp::Neg => "-",
                _ => "?",
            };
            format!("({} {})", op_str, expr_to_polish(operand))
        }
        Expr::Function { func, args } => {
            let func_str = match func {
                MathFn::Sin => "sin",
                MathFn::Cos => "cos",
                MathFn::Tan => "tan",
                MathFn::Log => "log",
                MathFn::Ln => "ln",
                MathFn::Exp => "exp",
                MathFn::Sqrt => "sqrt",
                MathFn::Abs => "abs",
                MathFn::Floor => "floor",
                MathFn::Ceil => "ceil",
                MathFn::Derive => "derive",
                MathFn::Integrate => "integrate",
            };
            let args_str: Vec<String> = args.iter().map(expr_to_polish).collect();
            format!("({} {})", func_str, args_str.join(" "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =====================================================
    // Parser Tests (original)
    // =====================================================

    #[test]
    fn test_tokenize() {
        let mut parser = PolishParser::new();
        let tokens = parser.tokenize("(+ 1 2)").unwrap();

        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], Token::OpenParen);
        assert_eq!(tokens[1], Token::Operator(MathOp::Add));
        assert_eq!(tokens[2], Token::Number(1.0));
        assert_eq!(tokens[3], Token::Number(2.0));
        assert_eq!(tokens[4], Token::CloseParen);
    }

    #[test]
    fn test_parse_simple() {
        let mut parser = PolishParser::new();
        let expr = parser.parse("(+ 2 3)").unwrap();

        match expr {
            Expr::BinOp { op, left, right } => {
                assert_eq!(op, MathOp::Add);
                assert_eq!(*left, Expr::Value(Value::Float(2.0)));
                assert_eq!(*right, Expr::Value(Value::Float(3.0)));
            }
            _ => panic!("Expected BinOp"),
        }
    }

    #[test]
    fn test_roundtrip() {
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Integer(2))),
                right: Box::new(Expr::Value(Value::Integer(3))),
            }),
            right: Box::new(Expr::Value(Value::Integer(4))),
        };

        let polish = expr_to_polish(&expr);
        assert_eq!(polish, "(* (+ 2 3) 4)");
    }

    // =====================================================
    // Graph Mapping Tests
    // =====================================================

    #[test]
    fn test_graph_from_value() {
        let expr = Expr::Value(Value::Integer(42));
        let graph = PolishGraph::from_expr(&expr);

        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.root.is_some());

        let recovered = graph.to_expr().unwrap();
        assert_eq!(recovered, Expr::Value(Value::Integer(42)));
    }

    #[test]
    fn test_graph_from_binop() {
        // (+ 2 3)
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };
        let graph = PolishGraph::from_expr(&expr);

        // 3 nodes: op, left, right
        assert_eq!(graph.node_count(), 3);
        // 2 edges: op->left, op->right
        assert_eq!(graph.edge_count(), 2);

        // Should have 2 leaf nodes
        assert_eq!(graph.leaf_nodes().len(), 2);
        // Should have 1 operator node
        assert_eq!(graph.operator_nodes().len(), 1);
    }

    #[test]
    fn test_graph_roundtrip_complex() {
        // (* (+ 2 3) (- 5 1))
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Integer(2))),
                right: Box::new(Expr::Value(Value::Integer(3))),
            }),
            right: Box::new(Expr::BinOp {
                op: MathOp::Sub,
                left: Box::new(Expr::Value(Value::Integer(5))),
                right: Box::new(Expr::Value(Value::Integer(1))),
            }),
        };

        let graph = PolishGraph::from_expr(&expr);
        let recovered = graph.to_expr().unwrap();

        // Compare Polish notation strings for equality
        assert_eq!(expr_to_polish(&expr), expr_to_polish(&recovered));
    }

    #[test]
    fn test_graph_from_function() {
        // (sin x)
        let expr = Expr::Function {
            func: MathFn::Sin,
            args: vec![Expr::Value(Value::Symbol("x".to_string()))],
        };

        let graph = PolishGraph::from_expr(&expr);

        // 2 nodes: function, argument
        assert_eq!(graph.node_count(), 2);
        // 1 edge: func->arg
        assert_eq!(graph.edge_count(), 1);

        let recovered = graph.to_expr().unwrap();
        assert_eq!(expr_to_polish(&expr), expr_to_polish(&recovered));
    }

    #[test]
    fn test_graph_node_types() {
        // Test different node types
        assert!(GraphNode::Integer(1).is_value());
        assert!(GraphNode::Float(1.0).is_value());
        assert!(GraphNode::Symbol("x".to_string()).is_value());
        assert!(GraphNode::Rational(1, 2).is_value());

        assert!(GraphNode::Operator(MathOp::Add).is_operator());
        assert!(!GraphNode::Operator(MathOp::Add).is_value());

        assert!(GraphNode::Function(MathFn::Sin).is_function());
        assert!(!GraphNode::Function(MathFn::Sin).is_value());
    }

    // =====================================================
    // Optimization Pass Tests
    // =====================================================

    #[test]
    fn test_constant_folding_simple() {
        // (+ 2 3) -> 5
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Integer(3))),
        };

        let folder = ConstantFolding::new();
        let result = folder.optimize(&expr);

        assert_eq!(result, Expr::Value(Value::Float(5.0)));
    }

    #[test]
    fn test_constant_folding_nested() {
        // (* (+ 2 3) 4) -> 20
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Integer(2))),
                right: Box::new(Expr::Value(Value::Integer(3))),
            }),
            right: Box::new(Expr::Value(Value::Integer(4))),
        };

        let folder = ConstantFolding::new();
        let result = folder.optimize(&expr);

        assert_eq!(result, Expr::Value(Value::Float(20.0)));
    }

    #[test]
    fn test_constant_folding_with_symbol() {
        // (+ 2 x) should stay as-is (has symbol)
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Integer(2))),
            right: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
        };

        let folder = ConstantFolding::new();
        let result = folder.optimize(&expr);

        // Should be unchanged
        assert_eq!(expr_to_polish(&result), "(+ 2 x)");
    }

    #[test]
    fn test_identity_elimination_add_zero() {
        // (+ x 0) -> x
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
            right: Box::new(Expr::Value(Value::Integer(0))),
        };

        let pass = IdentityElimination::new();
        let result = pass.optimize(&expr);

        assert_eq!(result, Expr::Value(Value::Symbol("x".to_string())));
    }

    #[test]
    fn test_identity_elimination_mul_one() {
        // (* x 1) -> x
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
            right: Box::new(Expr::Value(Value::Integer(1))),
        };

        let pass = IdentityElimination::new();
        let result = pass.optimize(&expr);

        assert_eq!(result, Expr::Value(Value::Symbol("x".to_string())));
    }

    #[test]
    fn test_identity_elimination_mul_zero() {
        // (* x 0) -> 0
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
            right: Box::new(Expr::Value(Value::Integer(0))),
        };

        let pass = IdentityElimination::new();
        let result = pass.optimize(&expr);

        assert_eq!(result, Expr::Value(Value::Integer(0)));
    }

    #[test]
    fn test_identity_elimination_pow_zero() {
        // (^ x 0) -> 1
        let expr = Expr::BinOp {
            op: MathOp::Pow,
            left: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
            right: Box::new(Expr::Value(Value::Integer(0))),
        };

        let pass = IdentityElimination::new();
        let result = pass.optimize(&expr);

        assert_eq!(result, Expr::Value(Value::Integer(1)));
    }

    #[test]
    fn test_identity_elimination_pow_one() {
        // (^ x 1) -> x
        let expr = Expr::BinOp {
            op: MathOp::Pow,
            left: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
            right: Box::new(Expr::Value(Value::Integer(1))),
        };

        let pass = IdentityElimination::new();
        let result = pass.optimize(&expr);

        assert_eq!(result, Expr::Value(Value::Symbol("x".to_string())));
    }

    #[test]
    fn test_optimizer_chain() {
        // (* (+ 2 3) 1) -> identity first: (* (+ 2 3) 1) -> (+ 2 3), then fold: 5
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Integer(2))),
                right: Box::new(Expr::Value(Value::Integer(3))),
            }),
            right: Box::new(Expr::Value(Value::Integer(1))),
        };

        let optimizer = Optimizer::with_defaults();
        let result = optimizer.optimize(&expr);

        assert_eq!(result, Expr::Value(Value::Float(5.0)));
    }

    #[test]
    fn test_optimizer_fixpoint() {
        // (+ (+ x 0) 0) -> x (requires multiple passes)
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
                right: Box::new(Expr::Value(Value::Integer(0))),
            }),
            right: Box::new(Expr::Value(Value::Integer(0))),
        };

        let optimizer = Optimizer::with_defaults();
        let result = optimizer.optimize_fixpoint(&expr);

        assert_eq!(result, Expr::Value(Value::Symbol("x".to_string())));
    }

    #[test]
    fn test_optimizer_with_partial_constants() {
        // (+ (* 2 3) x) -> (+ 6 x)
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::BinOp {
                op: MathOp::Mul,
                left: Box::new(Expr::Value(Value::Integer(2))),
                right: Box::new(Expr::Value(Value::Integer(3))),
            }),
            right: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
        };

        let optimizer = Optimizer::with_defaults();
        let result = optimizer.optimize(&expr);

        // The constant part should be folded
        assert_eq!(expr_to_polish(&result), "(+ 6 x)");
    }

    #[test]
    fn test_graph_with_optimization() {
        // Full pipeline: Expr -> Graph -> Expr -> Optimize -> Polish
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
            right: Box::new(Expr::Value(Value::Integer(0))),
        };

        let graph = PolishGraph::from_expr(&expr);
        let recovered = graph.to_expr().unwrap();

        let optimizer = Optimizer::with_defaults();
        let optimized = optimizer.optimize(&recovered);

        assert_eq!(optimized, Expr::Value(Value::Symbol("x".to_string())));
    }

    // =====================================================
    // Common Subexpression Elimination Tests
    // =====================================================

    #[test]
    fn test_cse_simple_duplicate() {
        // (* (+ a b) (+ a b)) -> (* _cse0 _cse0) with _cse0 = (+ a b)
        let add_ab = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Symbol("a".to_string()))),
            right: Box::new(Expr::Value(Value::Symbol("b".to_string()))),
        };
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(add_ab.clone()),
            right: Box::new(add_ab),
        };

        let cse = CommonSubexpressionElimination::new();
        let result = cse.optimize(&expr);

        // The result should have both operands be the same CSE variable
        assert_eq!(expr_to_polish(&result), "(* _cse0 _cse0)");

        // Check bindings
        let bindings = cse.bindings();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, "_cse0");
        assert_eq!(expr_to_polish(&bindings[0].1), "(+ a b)");
    }

    #[test]
    fn test_cse_no_duplicates() {
        // (+ a b) has no duplicates, should be unchanged
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Symbol("a".to_string()))),
            right: Box::new(Expr::Value(Value::Symbol("b".to_string()))),
        };

        let cse = CommonSubexpressionElimination::new();
        let result = cse.optimize(&expr);

        // Should be unchanged
        assert_eq!(expr_to_polish(&result), "(+ a b)");
        assert!(cse.bindings().is_empty());
    }

    #[test]
    fn test_cse_triple_occurrence() {
        // (+ (+ a b) (+ (+ a b) (+ a b))) - (+ a b) appears 3 times
        let add_ab = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Symbol("a".to_string()))),
            right: Box::new(Expr::Value(Value::Symbol("b".to_string()))),
        };
        let inner = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(add_ab.clone()),
            right: Box::new(add_ab.clone()),
        };
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(add_ab),
            right: Box::new(inner),
        };

        let cse = CommonSubexpressionElimination::new();
        let result = cse.optimize(&expr);

        // All occurrences of (+ a b) should be replaced
        let result_polish = expr_to_polish(&result);
        assert!(result_polish.contains("_cse"));
        assert!(!result_polish.contains("(+ a b)")); // Original subexpr should be replaced
    }

    #[test]
    fn test_cse_nested_duplicates() {
        // (+ (* x y) (+ (* x y) (* x y)))
        // Both (* x y) and (+ (* x y) (* x y)) might be candidates
        let mul_xy = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::Value(Value::Symbol("x".to_string()))),
            right: Box::new(Expr::Value(Value::Symbol("y".to_string()))),
        };
        let add_inner = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(mul_xy.clone()),
            right: Box::new(mul_xy.clone()),
        };
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(mul_xy),
            right: Box::new(add_inner),
        };

        let cse = CommonSubexpressionElimination::new();
        let result = cse.optimize(&expr);

        // Should have at least one CSE binding for (* x y) which appears 3 times
        let bindings = cse.bindings();
        assert!(!bindings.is_empty());

        // Result should be simplified
        let result_polish = expr_to_polish(&result);
        assert!(result_polish.contains("_cse"));
    }

    #[test]
    fn test_cse_with_functions() {
        // (+ (sin x) (sin x)) - function calls can also be CSE'd
        let sin_x = Expr::Function {
            func: MathFn::Sin,
            args: vec![Expr::Value(Value::Symbol("x".to_string()))],
        };
        let expr = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(sin_x.clone()),
            right: Box::new(sin_x),
        };

        let cse = CommonSubexpressionElimination::new();
        let result = cse.optimize(&expr);

        assert_eq!(expr_to_polish(&result), "(+ _cse0 _cse0)");
        let bindings = cse.bindings();
        assert_eq!(bindings.len(), 1);
        assert_eq!(expr_to_polish(&bindings[0].1), "(sin x)");
    }

    #[test]
    fn test_cse_min_occurrences() {
        // With min_occurrences=3, a subexpr appearing twice shouldn't be eliminated
        let add_ab = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Symbol("a".to_string()))),
            right: Box::new(Expr::Value(Value::Symbol("b".to_string()))),
        };
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(add_ab.clone()),
            right: Box::new(add_ab),
        };

        let cse = CommonSubexpressionElimination::with_min_occurrences(3);
        let result = cse.optimize(&expr);

        // Should be unchanged since (+ a b) only appears twice
        assert_eq!(expr_to_polish(&result), "(* (+ a b) (+ a b))");
        assert!(cse.bindings().is_empty());
    }

    #[test]
    fn test_cse_clears_bindings() {
        // Ensure bindings are cleared between optimize() calls
        let add_ab = Expr::BinOp {
            op: MathOp::Add,
            left: Box::new(Expr::Value(Value::Symbol("a".to_string()))),
            right: Box::new(Expr::Value(Value::Symbol("b".to_string()))),
        };
        let expr1 = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(add_ab.clone()),
            right: Box::new(add_ab),
        };

        let cse = CommonSubexpressionElimination::new();

        // First optimization
        let _ = cse.optimize(&expr1);
        assert_eq!(cse.bindings().len(), 1);

        // Second optimization with no duplicates
        let expr2 = Expr::Value(Value::Symbol("x".to_string()));
        let _ = cse.optimize(&expr2);
        assert!(cse.bindings().is_empty());
    }
}
