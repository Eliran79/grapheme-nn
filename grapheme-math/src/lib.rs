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

/// The math brain that learns graph transformations
#[derive(Debug, Default)]
pub struct MathBrain {
    /// The underlying engine for validation
    engine: MathEngine,
}

impl MathBrain {
    /// Create a new math brain
    pub fn new() -> Self {
        Self {
            engine: MathEngine::new(),
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
}
