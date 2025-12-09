//! # grapheme-code
//!
//! Code Brain: AST-based code analysis and compilation for GRAPHEME.
//!
//! This crate provides:
//! - AST node types for representing source code structure
//! - Code graph construction from source text
//! - Type checking and validation infrastructure
//! - Code transformation rules
//!
//! Future enhancements:
//! - Tree-sitter integration for multi-language parsing
//! - Full compilation and execution support

use grapheme_core::{
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule, ExecutionResult, ValidationIssue,
    ValidationSeverity,
};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors in code graph processing
#[derive(Error, Debug)]
pub enum CodeGraphError {
    #[error("Invalid syntax: {0}")]
    InvalidSyntax(String),
    #[error("Type error: {0}")]
    TypeError(String),
    #[error("Undefined identifier: {0}")]
    UndefinedIdentifier(String),
    #[error("Parse error at line {line}, column {column}: {message}")]
    ParseError {
        line: usize,
        column: usize,
        message: String,
    },
}

/// Result type for code graph operations
pub type CodeGraphResult<T> = Result<T, CodeGraphError>;

/// Programming language identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Language {
    /// Rust programming language
    Rust,
    /// Python programming language
    Python,
    /// JavaScript/TypeScript
    JavaScript,
    /// C/C++
    C,
    /// Generic/unknown language
    #[default]
    Generic,
}

/// AST node type variants for code representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CodeNodeType {
    /// Module/file root
    Module { name: String, language: Language },
    /// Function definition
    Function {
        name: String,
        params: Vec<String>,
        return_type: Option<String>,
    },
    /// Variable declaration
    Variable {
        name: String,
        var_type: Option<String>,
    },
    /// Literal value
    Literal(LiteralValue),
    /// Binary operation
    BinaryOp(BinaryOperator),
    /// Unary operation
    UnaryOp(UnaryOperator),
    /// Function call
    Call { function: String, arg_count: usize },
    /// Control flow: if/else
    If,
    /// Control flow: loop
    Loop { kind: LoopKind },
    /// Return statement
    Return,
    /// Block of statements
    Block,
    /// Type annotation
    Type(String),
    /// Comment
    Comment(String),
    /// Identifier reference
    Identifier(String),
    /// Assignment
    Assignment,
    /// Expression statement
    ExprStmt,
}

/// Code node with activation for gradient flow (Backend-117)
///
/// Wraps CodeNodeType with a learned activation value that enables
/// gradients to flow through structural loss during training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeNode {
    /// The underlying node type
    pub node_type: CodeNodeType,
    /// Learned activation value for gradient flow
    pub activation: f32,
}

impl CodeNode {
    /// Create a new code node with default activation based on type
    pub fn new(node_type: CodeNodeType) -> Self {
        let activation = Self::type_activation(&node_type);
        Self { node_type, activation }
    }

    /// Create node with explicit activation value
    pub fn with_activation(node_type: CodeNodeType, activation: f32) -> Self {
        Self { node_type, activation }
    }

    /// Get default activation based on node type
    /// Higher values for more semantically important constructs
    fn type_activation(node_type: &CodeNodeType) -> f32 {
        match node_type {
            CodeNodeType::Module { .. } => 0.9,      // Root structure
            CodeNodeType::Function { .. } => 0.8,   // Key construct
            CodeNodeType::Variable { .. } => 0.4,   // Data storage
            CodeNodeType::Literal(_) => 0.2,        // Constants
            CodeNodeType::BinaryOp(_) => 0.5,       // Operations
            CodeNodeType::UnaryOp(_) => 0.4,        // Operations
            CodeNodeType::Call { .. } => 0.7,       // Function invocation
            CodeNodeType::If => 0.6,                // Control flow
            CodeNodeType::Loop { .. } => 0.6,       // Control flow
            CodeNodeType::Return => 0.5,            // Control flow
            CodeNodeType::Block => 0.3,             // Structural
            CodeNodeType::Type(_) => 0.4,           // Type annotation
            CodeNodeType::Comment(_) => 0.1,        // Documentation
            CodeNodeType::Identifier(_) => 0.3,     // Reference
            CodeNodeType::Assignment => 0.5,        // State change
            CodeNodeType::ExprStmt => 0.3,          // Expression wrapper
        }
    }
}

/// Literal value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiteralValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Neg,
    Not,
    BitNot,
    Deref,
    Ref,
}

/// Loop kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopKind {
    For,
    While,
    Loop,
    DoWhile,
}

// ============================================================================
// Type System for Type Inference
// ============================================================================

/// Type variable identifier for unification
pub type TypeVar = usize;

/// Types in the inference system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferredType {
    /// Unknown type (type variable for unification)
    Unknown(TypeVar),
    /// Integer type
    Int,
    /// Floating point type
    Float,
    /// Boolean type
    Bool,
    /// String type
    String,
    /// Unit/void type
    Unit,
    /// Null/None type
    Null,
    /// Function type: (param_types) -> return_type
    Function {
        params: Vec<InferredType>,
        ret: Box<InferredType>,
    },
    /// Array/list type
    Array(Box<InferredType>),
    /// Tuple type
    Tuple(Vec<InferredType>),
    /// Reference type
    Ref(Box<InferredType>),
    /// Named/custom type
    Named(String),
    /// Error type (type inference failed)
    Error,
}

impl Default for InferredType {
    fn default() -> Self {
        InferredType::Unknown(0)
    }
}

impl std::fmt::Display for InferredType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferredType::Unknown(v) => write!(f, "?T{}", v),
            InferredType::Int => write!(f, "Int"),
            InferredType::Float => write!(f, "Float"),
            InferredType::Bool => write!(f, "Bool"),
            InferredType::String => write!(f, "String"),
            InferredType::Unit => write!(f, "()"),
            InferredType::Null => write!(f, "Null"),
            InferredType::Function { params, ret } => {
                write!(f, "(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            InferredType::Array(inner) => write!(f, "[{}]", inner),
            InferredType::Tuple(types) => {
                write!(f, "(")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            InferredType::Ref(inner) => write!(f, "&{}", inner),
            InferredType::Named(name) => write!(f, "{}", name),
            InferredType::Error => write!(f, "Error"),
        }
    }
}

impl InferredType {
    /// Check if this type contains a type variable
    pub fn contains_var(&self, var: TypeVar) -> bool {
        match self {
            InferredType::Unknown(v) => *v == var,
            InferredType::Function { params, ret } => {
                params.iter().any(|p| p.contains_var(var)) || ret.contains_var(var)
            }
            InferredType::Array(inner) | InferredType::Ref(inner) => inner.contains_var(var),
            InferredType::Tuple(types) => types.iter().any(|t| t.contains_var(var)),
            _ => false,
        }
    }

    /// Substitute a type variable with a concrete type
    pub fn substitute(&self, var: TypeVar, replacement: &InferredType) -> InferredType {
        match self {
            InferredType::Unknown(v) if *v == var => replacement.clone(),
            InferredType::Function { params, ret } => InferredType::Function {
                params: params
                    .iter()
                    .map(|p| p.substitute(var, replacement))
                    .collect(),
                ret: Box::new(ret.substitute(var, replacement)),
            },
            InferredType::Array(inner) => {
                InferredType::Array(Box::new(inner.substitute(var, replacement)))
            }
            InferredType::Ref(inner) => {
                InferredType::Ref(Box::new(inner.substitute(var, replacement)))
            }
            InferredType::Tuple(types) => InferredType::Tuple(
                types
                    .iter()
                    .map(|t| t.substitute(var, replacement))
                    .collect(),
            ),
            _ => self.clone(),
        }
    }

    /// Parse a type string into an InferredType
    pub fn from_type_string(s: &str) -> Self {
        let s = s.trim();
        match s {
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "int" | "Int" | "integer" => {
                InferredType::Int
            }
            "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => InferredType::Int,
            "f32" | "f64" | "float" | "Float" | "double" => InferredType::Float,
            "bool" | "Bool" | "boolean" => InferredType::Bool,
            "str" | "String" | "string" | "&str" => InferredType::String,
            "()" | "void" | "None" | "Unit" => InferredType::Unit,
            "null" | "Null" | "nil" => InferredType::Null,
            _ if s.starts_with("Vec<") || s.starts_with('[') => {
                let inner = if s.starts_with("Vec<") {
                    s.strip_prefix("Vec<")
                        .and_then(|s| s.strip_suffix('>'))
                        .unwrap_or("Unknown")
                } else {
                    s.strip_prefix('[')
                        .and_then(|s| s.strip_suffix(']'))
                        .unwrap_or("Unknown")
                };
                InferredType::Array(Box::new(Self::from_type_string(inner)))
            }
            _ if s.starts_with('&') => InferredType::Ref(Box::new(Self::from_type_string(&s[1..]))),
            _ => InferredType::Named(s.to_string()),
        }
    }
}

// ============================================================================
// Type Constraint for Unification
// ============================================================================

/// A type constraint representing that two types must be equal
#[derive(Debug, Clone)]
pub struct TypeConstraint {
    pub left: InferredType,
    pub right: InferredType,
    pub source: Option<NodeIndex>,
}

// ============================================================================
// Type Inference Engine
// ============================================================================

use std::collections::HashMap;

/// Type inference engine using Hindley-Milner style constraint solving
#[derive(Debug)]
pub struct TypeInferenceEngine {
    /// Next type variable ID
    next_var: TypeVar,
    /// Substitution map from type variables to types
    substitutions: HashMap<TypeVar, InferredType>,
    /// Type environment: variable name -> type
    environment: HashMap<String, InferredType>,
    /// Node types: node index -> inferred type
    node_types: HashMap<NodeIndex, InferredType>,
    /// Constraints to solve
    constraints: Vec<TypeConstraint>,
    /// Type errors encountered
    errors: Vec<String>,
}

impl Default for TypeInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeInferenceEngine {
    /// Create a new type inference engine
    pub fn new() -> Self {
        Self {
            next_var: 0,
            substitutions: HashMap::new(),
            environment: HashMap::new(),
            node_types: HashMap::new(),
            constraints: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Generate a fresh type variable
    pub fn fresh_var(&mut self) -> InferredType {
        let var = self.next_var;
        self.next_var += 1;
        InferredType::Unknown(var)
    }

    /// Add a constraint that two types must be equal
    pub fn add_constraint(
        &mut self,
        left: InferredType,
        right: InferredType,
        source: Option<NodeIndex>,
    ) {
        self.constraints.push(TypeConstraint {
            left,
            right,
            source,
        });
    }

    /// Bind a variable name to a type in the environment
    pub fn bind_var(&mut self, name: &str, ty: InferredType) {
        self.environment.insert(name.to_string(), ty);
    }

    /// Look up a variable's type in the environment
    pub fn lookup_var(&self, name: &str) -> Option<&InferredType> {
        self.environment.get(name)
    }

    /// Set the inferred type for a node
    pub fn set_node_type(&mut self, node: NodeIndex, ty: InferredType) {
        self.node_types.insert(node, ty);
    }

    /// Get the inferred type for a node
    pub fn get_node_type(&self, node: NodeIndex) -> Option<&InferredType> {
        self.node_types.get(&node)
    }

    /// Get all errors encountered during inference
    pub fn get_errors(&self) -> &[String] {
        &self.errors
    }

    /// Apply current substitutions to a type
    pub fn apply_substitutions(&self, ty: &InferredType) -> InferredType {
        let mut result = ty.clone();
        for (&var, replacement) in &self.substitutions {
            result = result.substitute(var, replacement);
        }
        result
    }

    /// Unify two types, updating substitutions
    pub fn unify(&mut self, t1: &InferredType, t2: &InferredType) -> Result<(), String> {
        let t1 = self.apply_substitutions(t1);
        let t2 = self.apply_substitutions(t2);

        match (&t1, &t2) {
            // Same types unify trivially
            (a, b) if a == b => Ok(()),

            // Type variable unification
            (InferredType::Unknown(v), t) | (t, InferredType::Unknown(v)) => {
                // Occurs check: prevent infinite types
                if t.contains_var(*v) {
                    return Err(format!("Infinite type: ?T{} occurs in {}", v, t));
                }
                self.substitutions.insert(*v, t.clone());
                Ok(())
            }

            // Function types
            (
                InferredType::Function {
                    params: p1,
                    ret: r1,
                },
                InferredType::Function {
                    params: p2,
                    ret: r2,
                },
            ) => {
                if p1.len() != p2.len() {
                    return Err(format!(
                        "Function arity mismatch: {} vs {} parameters",
                        p1.len(),
                        p2.len()
                    ));
                }
                for (a, b) in p1.iter().zip(p2.iter()) {
                    self.unify(a, b)?;
                }
                self.unify(r1, r2)
            }

            // Array types
            (InferredType::Array(a), InferredType::Array(b)) => self.unify(a, b),

            // Reference types
            (InferredType::Ref(a), InferredType::Ref(b)) => self.unify(a, b),

            // Tuple types
            (InferredType::Tuple(a), InferredType::Tuple(b)) => {
                if a.len() != b.len() {
                    return Err(format!("Tuple length mismatch: {} vs {}", a.len(), b.len()));
                }
                for (x, y) in a.iter().zip(b.iter()) {
                    self.unify(x, y)?;
                }
                Ok(())
            }

            // Numeric coercion: Int can be promoted to Float
            (InferredType::Int, InferredType::Float) | (InferredType::Float, InferredType::Int) => {
                // Allow numeric coercion - result is Float
                Ok(())
            }

            // Error type unifies with anything (to allow recovery)
            (InferredType::Error, _) | (_, InferredType::Error) => Ok(()),

            // Named types with same name
            (InferredType::Named(a), InferredType::Named(b)) if a == b => Ok(()),

            // Otherwise, types don't unify
            _ => Err(format!("Cannot unify {} with {}", t1, t2)),
        }
    }

    /// Solve all accumulated constraints
    pub fn solve_constraints(&mut self) -> bool {
        let constraints = std::mem::take(&mut self.constraints);
        let mut all_ok = true;

        for constraint in constraints {
            if let Err(e) = self.unify(&constraint.left, &constraint.right) {
                self.errors.push(format!(
                    "Type error{}: {}",
                    constraint
                        .source
                        .map(|n| format!(" at node {:?}", n))
                        .unwrap_or_default(),
                    e
                ));
                all_ok = false;
            }
        }

        all_ok
    }

    /// Run type inference on a CodeGraph
    pub fn infer_types(&mut self, graph: &CodeGraph) -> HashMap<NodeIndex, InferredType> {
        // Phase 1: Generate initial types and constraints
        for node_idx in graph.graph.node_indices() {
            let initial_type = self.infer_node_type(graph, node_idx);
            self.set_node_type(node_idx, initial_type);
        }

        // Phase 2: Propagate types through edges
        self.propagate_types(graph);

        // Phase 3: Solve constraints
        self.solve_constraints();

        // Phase 4: Apply substitutions to get final types
        let mut final_types = HashMap::new();
        for (&node_idx, ty) in &self.node_types {
            final_types.insert(node_idx, self.apply_substitutions(ty));
        }

        final_types
    }

    /// Infer the initial type for a single node
    fn infer_node_type(&mut self, graph: &CodeGraph, node_idx: NodeIndex) -> InferredType {
        let node = &graph.graph[node_idx];

        match node {
            CodeNode { node_type: CodeNodeType::Literal(lit), .. } => match lit {
                LiteralValue::Integer(_) => InferredType::Int,
                LiteralValue::Float(_) => InferredType::Float,
                LiteralValue::String(_) => InferredType::String,
                LiteralValue::Boolean(_) => InferredType::Bool,
                LiteralValue::Null => InferredType::Null,
            },

            CodeNode { node_type: CodeNodeType::Variable { name, var_type }, .. } => {
                if let Some(ty_str) = var_type {
                    let ty = InferredType::from_type_string(ty_str);
                    self.bind_var(name, ty.clone());
                    ty
                } else if let Some(ty) = self.lookup_var(name).cloned() {
                    ty
                } else {
                    let ty = self.fresh_var();
                    self.bind_var(name, ty.clone());
                    ty
                }
            }

            CodeNode { node_type: CodeNodeType::Identifier(name), .. } => {
                if let Some(ty) = self.lookup_var(name).cloned() {
                    ty
                } else {
                    // Unknown identifier, create fresh type variable
                    let ty = self.fresh_var();
                    self.bind_var(name, ty.clone());
                    ty
                }
            }

            CodeNode { node_type: CodeNodeType::Function {
                return_type,
                params,
                ..
            }, .. } => {
                let ret_ty = if let Some(ret_str) = return_type {
                    InferredType::from_type_string(ret_str)
                } else {
                    self.fresh_var()
                };

                let param_types: Vec<_> = params.iter().map(|_| self.fresh_var()).collect();

                InferredType::Function {
                    params: param_types,
                    ret: Box::new(ret_ty),
                }
            }

            CodeNode { node_type: CodeNodeType::Call { function, .. }, .. } => {
                // Look up function type and return its return type
                if let Some(InferredType::Function { ret, .. }) = self.lookup_var(function) {
                    (**ret).clone()
                } else {
                    self.fresh_var()
                }
            }

            CodeNode { node_type: CodeNodeType::BinaryOp(op), .. } => {
                // Binary ops: determine result type based on operator
                match op {
                    BinaryOperator::Add
                    | BinaryOperator::Sub
                    | BinaryOperator::Mul
                    | BinaryOperator::Div
                    | BinaryOperator::Mod => {
                        // Arithmetic operators: result is numeric (exact type depends on operands)
                        self.fresh_var()
                    }
                    BinaryOperator::Eq
                    | BinaryOperator::Ne
                    | BinaryOperator::Lt
                    | BinaryOperator::Le
                    | BinaryOperator::Gt
                    | BinaryOperator::Ge => {
                        // Comparison operators always return Bool
                        InferredType::Bool
                    }
                    BinaryOperator::And | BinaryOperator::Or => {
                        // Logical operators return Bool
                        InferredType::Bool
                    }
                    BinaryOperator::BitAnd
                    | BinaryOperator::BitOr
                    | BinaryOperator::BitXor
                    | BinaryOperator::Shl
                    | BinaryOperator::Shr => {
                        // Bitwise operators return Int
                        InferredType::Int
                    }
                }
            }

            CodeNode { node_type: CodeNodeType::UnaryOp(op), .. } => match op {
                UnaryOperator::Neg | UnaryOperator::BitNot => self.fresh_var(),
                UnaryOperator::Not => InferredType::Bool,
                UnaryOperator::Deref => self.fresh_var(),
                UnaryOperator::Ref => self.fresh_var(),
            },

            CodeNode { node_type: CodeNodeType::If, .. } => {
                // If expressions can have a result type
                self.fresh_var()
            }

            CodeNode { node_type: CodeNodeType::Loop { .. }, .. } => {
                // Loops typically return unit unless they have break with value
                InferredType::Unit
            }

            CodeNode { node_type: CodeNodeType::Return, .. } => {
                // Return type depends on the expression being returned
                self.fresh_var()
            }

            CodeNode { node_type: CodeNodeType::Block, .. } => {
                // Block type is the type of the last expression
                self.fresh_var()
            }

            CodeNode { node_type: CodeNodeType::Type(ty_str), .. } => InferredType::from_type_string(ty_str),

            CodeNode { node_type: CodeNodeType::Module { .. }, .. }
            | CodeNode { node_type: CodeNodeType::Comment(_), .. }
            | CodeNode { node_type: CodeNodeType::Assignment, .. }
            | CodeNode { node_type: CodeNodeType::ExprStmt, .. } => InferredType::Unit,
        }
    }

    /// Propagate types through graph edges
    fn propagate_types(&mut self, graph: &CodeGraph) {
        // Process edges to add constraints
        for edge_idx in graph.graph.edge_indices() {
            let (source, target) = graph.graph.edge_endpoints(edge_idx).unwrap();
            let edge = &graph.graph[edge_idx];

            match edge {
                CodeEdge::HasType => {
                    // Target node is the type of source node
                    if let Some(source_ty) = self.node_types.get(&source).cloned() {
                        if let Some(target_ty) = self.node_types.get(&target).cloned() {
                            self.add_constraint(source_ty, target_ty, Some(source));
                        }
                    }
                }

                CodeEdge::Child(_) => {
                    // For binary operators, constrain operands
                    if let CodeNode { node_type: CodeNodeType::BinaryOp(op), .. } = &graph.graph[source] {
                        self.propagate_binary_op_constraints(source, target, op);
                    }
                }

                CodeEdge::DataFlow => {
                    // Data flows should have same type
                    if let (Some(src_ty), Some(tgt_ty)) = (
                        self.node_types.get(&source).cloned(),
                        self.node_types.get(&target).cloned(),
                    ) {
                        self.add_constraint(src_ty, tgt_ty, Some(source));
                    }
                }

                CodeEdge::DefUse => {
                    // Definition and use should have same type
                    if let (Some(def_ty), Some(use_ty)) = (
                        self.node_types.get(&source).cloned(),
                        self.node_types.get(&target).cloned(),
                    ) {
                        self.add_constraint(def_ty, use_ty, Some(target));
                    }
                }

                _ => {}
            }
        }
    }

    /// Add constraints for binary operator operands
    fn propagate_binary_op_constraints(
        &mut self,
        op_node: NodeIndex,
        operand_node: NodeIndex,
        op: &BinaryOperator,
    ) {
        let operand_ty = self
            .node_types
            .get(&operand_node)
            .cloned()
            .unwrap_or_else(|| self.fresh_var());
        let result_ty = self
            .node_types
            .get(&op_node)
            .cloned()
            .unwrap_or_else(|| self.fresh_var());

        match op {
            BinaryOperator::Add
            | BinaryOperator::Sub
            | BinaryOperator::Mul
            | BinaryOperator::Div
            | BinaryOperator::Mod => {
                // Arithmetic: operand type should match result type (for numeric operations)
                // Both operands should be numeric
                self.add_constraint(operand_ty, result_ty, Some(operand_node));
            }

            BinaryOperator::Eq
            | BinaryOperator::Ne
            | BinaryOperator::Lt
            | BinaryOperator::Le
            | BinaryOperator::Gt
            | BinaryOperator::Ge => {
                // Comparison: operands should be comparable (same type or numeric)
                // Result is always Bool (already set)
            }

            BinaryOperator::And | BinaryOperator::Or => {
                // Logical: operands should be Bool
                self.add_constraint(operand_ty, InferredType::Bool, Some(operand_node));
            }

            BinaryOperator::BitAnd
            | BinaryOperator::BitOr
            | BinaryOperator::BitXor
            | BinaryOperator::Shl
            | BinaryOperator::Shr => {
                // Bitwise: operands should be Int
                self.add_constraint(operand_ty, InferredType::Int, Some(operand_node));
            }
        }
    }

    /// Get all inferred types, applying final substitutions
    pub fn get_all_types(&self) -> HashMap<NodeIndex, InferredType> {
        self.node_types
            .iter()
            .map(|(&k, v)| (k, self.apply_substitutions(v)))
            .collect()
    }
}

/// Edge types in the code graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeEdge {
    /// Child node relationship
    Child(usize),
    /// Next sibling in sequence
    Next,
    /// Control flow edge
    ControlFlow,
    /// Data flow edge
    DataFlow,
    /// Type relationship
    HasType,
    /// Definition to use
    DefUse,
}

/// A code snippet represented as a graph
#[derive(Debug)]
pub struct CodeGraph {
    /// The underlying directed graph
    pub graph: DiGraph<CodeNode, CodeEdge>,
    /// Root node of the AST
    pub root: Option<NodeIndex>,
    /// Source language
    pub language: Language,
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeGraph {
    /// Create a new empty code graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            root: None,
            language: Language::Generic,
        }
    }

    /// Create a code graph with a specific language
    pub fn with_language(language: Language) -> Self {
        Self {
            graph: DiGraph::new(),
            root: None,
            language,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: CodeNode) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: CodeEdge) {
        self.graph.add_edge(from, to, edge);
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Parse a simple expression into a code graph
    pub fn from_simple_expr(expr: &str) -> CodeGraphResult<Self> {
        let mut graph = Self::new();
        let trimmed = expr.trim();

        // Try to parse as a number
        if let Ok(n) = trimmed.parse::<i64>() {
            let node = graph.add_node(CodeNode::new(CodeNodeType::Literal(LiteralValue::Integer(n))));
            graph.root = Some(node);
            return Ok(graph);
        }

        if let Ok(n) = trimmed.parse::<f64>() {
            let node = graph.add_node(CodeNode::new(CodeNodeType::Literal(LiteralValue::Float(n))));
            graph.root = Some(node);
            return Ok(graph);
        }

        // Try to parse as a boolean
        if trimmed == "true" {
            let node = graph.add_node(CodeNode::new(CodeNodeType::Literal(LiteralValue::Boolean(true))));
            graph.root = Some(node);
            return Ok(graph);
        }
        if trimmed == "false" {
            let node = graph.add_node(CodeNode::new(CodeNodeType::Literal(LiteralValue::Boolean(false))));
            graph.root = Some(node);
            return Ok(graph);
        }

        // Try to parse as a simple binary expression
        for (op_str, op) in [
            ("+", BinaryOperator::Add),
            ("-", BinaryOperator::Sub),
            ("*", BinaryOperator::Mul),
            ("/", BinaryOperator::Div),
            ("==", BinaryOperator::Eq),
            ("!=", BinaryOperator::Ne),
            ("<=", BinaryOperator::Le),
            (">=", BinaryOperator::Ge),
            ("<", BinaryOperator::Lt),
            (">", BinaryOperator::Gt),
        ] {
            if let Some(idx) = trimmed.find(op_str) {
                if idx > 0 && idx < trimmed.len() - op_str.len() {
                    let left_str = trimmed[..idx].trim();
                    let right_str = trimmed[idx + op_str.len()..].trim();

                    // Recursively parse left and right
                    let left_graph = Self::from_simple_expr(left_str)?;
                    let right_graph = Self::from_simple_expr(right_str)?;

                    // Create the binary op node
                    let op_node = graph.add_node(CodeNode::new(CodeNodeType::BinaryOp(op)));

                    // Add left and right as children (simplified - just copy root values)
                    if let Some(left_root) = left_graph.root {
                        let left_node = graph.add_node(left_graph.graph[left_root].clone());
                        graph.add_edge(op_node, left_node, CodeEdge::Child(0));
                    }
                    if let Some(right_root) = right_graph.root {
                        let right_node = graph.add_node(right_graph.graph[right_root].clone());
                        graph.add_edge(op_node, right_node, CodeEdge::Child(1));
                    }

                    graph.root = Some(op_node);
                    return Ok(graph);
                }
            }
        }

        // Treat as identifier
        let node = graph.add_node(CodeNode::new(CodeNodeType::Identifier(trimmed.to_string())));
        graph.root = Some(node);
        Ok(graph)
    }

    /// Run type inference on this code graph
    ///
    /// Returns a map from node indices to their inferred types.
    /// This uses Hindley-Milner style constraint-based type inference.
    pub fn infer_types(&self) -> HashMap<NodeIndex, InferredType> {
        let mut engine = TypeInferenceEngine::new();
        engine.infer_types(self)
    }

    /// Run type inference and return both types and any errors
    pub fn infer_types_with_errors(&self) -> (HashMap<NodeIndex, InferredType>, Vec<String>) {
        let mut engine = TypeInferenceEngine::new();
        let types = engine.infer_types(self);
        (types, engine.get_errors().to_vec())
    }

    /// Get the inferred type of a specific node
    pub fn get_node_type(&self, node_idx: NodeIndex) -> Option<InferredType> {
        self.infer_types().get(&node_idx).cloned()
    }

    /// Check if the graph is well-typed (no type errors)
    pub fn is_well_typed(&self) -> bool {
        let (_, errors) = self.infer_types_with_errors();
        errors.is_empty()
    }

    /// Add a type annotation edge from a node to its type
    pub fn annotate_type(&mut self, node: NodeIndex, type_str: &str) -> NodeIndex {
        let type_node = self.add_node(CodeNode::new(CodeNodeType::Type(type_str.to_string())));
        self.add_edge(node, type_node, CodeEdge::HasType);
        type_node
    }
}

/// Result of type inference on a code graph
#[derive(Debug, Clone)]
pub struct TypeInferenceResult {
    /// Inferred types for each node
    pub node_types: HashMap<NodeIndex, InferredType>,
    /// Type errors encountered
    pub errors: Vec<String>,
    /// Whether inference was successful (no errors)
    pub success: bool,
}

impl TypeInferenceResult {
    /// Get the type of a specific node
    pub fn get_type(&self, node: NodeIndex) -> Option<&InferredType> {
        self.node_types.get(&node)
    }

    /// Check if a node has a concrete (non-unknown) type
    pub fn has_concrete_type(&self, node: NodeIndex) -> bool {
        self.node_types
            .get(&node)
            .is_some_and(|ty| !matches!(ty, InferredType::Unknown(_)))
    }
}

// ============================================================================
// Code Brain
// ============================================================================

/// The Code Brain that processes source code as graphs
pub struct CodeBrain {
    /// Supported languages
    languages: Vec<Language>,
}

impl Default for CodeBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CodeBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeBrain")
            .field("domain", &"code")
            .field("languages", &self.languages)
            .finish()
    }
}

impl CodeBrain {
    /// Create a new code brain
    pub fn new() -> Self {
        Self {
            languages: vec![
                Language::Rust,
                Language::Python,
                Language::JavaScript,
                Language::C,
                Language::Generic,
            ],
        }
    }

    /// Check if code contains function-like syntax
    fn looks_like_code(&self, input: &str) -> bool {
        let code_patterns = [
            "fn ",
            "def ",
            "function ",
            "func ",
            "let ",
            "var ",
            "const ",
            "if ",
            "else ",
            "while ",
            "for ",
            "return ",
            "class ",
            "struct ",
            "import ",
            "from ",
            "use ",
            "->",
            "=>",
            "::",
            "{",
            "}",
            "(",
            ")",
        ];
        let lower = input.to_lowercase();
        code_patterns.iter().any(|p| lower.contains(p))
    }

    /// Detect language from code snippet
    pub fn detect_language(&self, code: &str) -> Language {
        if code.contains("fn ") && code.contains("->") {
            Language::Rust
        } else if code.contains("def ") && code.contains(":") {
            Language::Python
        } else if code.contains("function ") || code.contains("=>") {
            Language::JavaScript
        } else if code.contains("#include") || code.contains("int main") {
            Language::C
        } else {
            Language::Generic
        }
    }

    /// Normalize code text for domain processing
    /// Handles whitespace, removes empty lines, normalizes line endings
    fn normalize_code_text(&self, text: &str) -> String {
        // Normalize line endings to LF
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");

        // Remove trailing whitespace from each line
        let lines: Vec<&str> = normalized.lines().map(|line| line.trim_end()).collect();

        // Join and trim overall
        lines.join("\n").trim().to_string()
    }

    /// Parse code into a CodeGraph
    pub fn parse_code(&self, code: &str) -> CodeGraphResult<CodeGraph> {
        CodeGraph::from_simple_expr(code)
    }

    /// Validate a code graph for common issues
    pub fn validate_code(&self, graph: &CodeGraph) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        if graph.node_count() == 0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Empty code graph".to_string(),
                node: None,
            });
        }

        // Check for undefined identifiers (simplified)
        for node_idx in graph.graph.node_indices() {
            if let CodeNode { node_type: CodeNodeType::Identifier(name), .. } = &graph.graph[node_idx] {
                // In a real implementation, we'd check against defined symbols
                if name.starts_with("undefined_") {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: format!("Potentially undefined identifier: {}", name),
                        node: Some(node_idx),
                    });
                }
            }
        }

        issues
    }

    // Transform helper methods for DomainBrain::transform

    /// Rule 0: Dead Code Elimination - remove unreachable/unused nodes
    fn apply_dead_code_elimination(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // For DagNN, we remove isolated nodes (no connections)
        // In practice, a proper DCE would need data flow analysis
        let text = graph.to_text();

        // Simple heuristic: remove trailing whitespace and empty blocks
        let cleaned = text.trim().to_string();

        // If text changed, create new graph; otherwise return clone
        if cleaned != text {
            DagNN::from_text(&cleaned).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 1: Constant Folding - evaluate constant expressions
    fn apply_constant_folding(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Simple constant folding for basic arithmetic
        let folded = self.fold_constants_in_text(&text);

        if folded != text {
            DagNN::from_text(&folded).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Helper: fold simple constant expressions in text
    #[allow(clippy::type_complexity)]
    fn fold_constants_in_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Pattern: number op number (very simple)
        // Look for patterns like "2 + 3" or "4 * 5"
        // Use function pointers instead of closures to avoid type mismatch
        let ops: [(&str, fn(i64, i64) -> i64); 3] = [
            ("+", |a, b| a + b),
            ("-", |a, b| a - b),
            ("*", |a, b| a * b),
        ];

        for (op, func) in &ops {
            // Simple regex-free pattern matching
            let parts: Vec<&str> = result.split(op).collect();
            if parts.len() == 2 {
                if let (Ok(a), Ok(b)) = (
                    parts[0].trim().parse::<i64>(),
                    parts[1].trim().parse::<i64>(),
                ) {
                    result = func(a, b).to_string();
                }
            }
        }

        result
    }

    /// Rule 2: Inline Expansion - expand function calls inline
    fn apply_inline_expansion(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // For DagNN character graphs, inline expansion is not directly applicable
        // Return unchanged
        Ok(graph.clone())
    }

    /// Rule 3: Loop Unrolling - expand loop iterations
    fn apply_loop_unrolling(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // For DagNN character graphs, loop unrolling is not directly applicable
        // Return unchanged
        Ok(graph.clone())
    }

    /// Rule 4: Type Inference - infer types for untyped variables
    ///
    /// This performs Hindley-Milner style type inference on the code graph:
    /// 1. Parse the DagNN text representation into a CodeGraph
    /// 2. Run constraint-based type inference
    /// 3. Annotate the result with inferred types as comments
    fn apply_type_inference(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Try to parse as a code expression
        match CodeGraph::from_simple_expr(&text) {
            Ok(code_graph) => {
                // Run type inference
                let (types, errors) = code_graph.infer_types_with_errors();

                // Build annotated output
                let mut result = text.clone();

                // Find the root node's type if available
                if let Some(root) = code_graph.root {
                    if let Some(root_type) = types.get(&root) {
                        // Annotate with inferred type
                        result = format!("{} /* : {} */", text.trim(), root_type);
                    }
                }

                // Add error comments if any
                if !errors.is_empty() {
                    result = format!("{}\n/* Type errors:\n{}\n*/", result, errors.join("\n"));
                }

                DagNN::from_text(&result).map_err(|e| e.into())
            }
            Err(_) => {
                // If we can't parse, just return as-is with basic annotation
                self.infer_types_from_patterns(&text, graph)
            }
        }
    }

    /// Fallback type inference based on pattern matching
    fn infer_types_from_patterns(&self, text: &str, graph: &DagNN) -> DomainResult<DagNN> {
        let trimmed = text.trim();

        // Pattern-based type inference for simple cases
        let inferred_type = if trimmed.parse::<i64>().is_ok() {
            Some("Int")
        } else if trimmed.parse::<f64>().is_ok() {
            Some("Float")
        } else if trimmed == "true" || trimmed == "false" {
            Some("Bool")
        } else if (trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
        {
            Some("String")
        } else if trimmed == "null" || trimmed == "None" || trimmed == "nil" {
            Some("Null")
        } else {
            None
        };

        if let Some(ty) = inferred_type {
            let annotated = format!("{} /* : {} */", trimmed, ty);
            DagNN::from_text(&annotated).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

impl DomainBrain for CodeBrain {
    fn domain_id(&self) -> &str {
        "code"
    }

    fn domain_name(&self) -> &str {
        "Source Code"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn can_process(&self, input: &str) -> bool {
        self.looks_like_code(input)
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        DagNN::from_text(input).map_err(|e| e.into())
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Convert core DagNN to code domain representation
        // Normalize code formatting for processing
        let text = graph.to_text();

        // Apply code-specific normalization
        let normalized = self.normalize_code_text(&text);

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Convert code domain representation back to generic core format
        // Clean up any domain-specific artifacts
        let text = graph.to_text();

        // Remove any domain-specific metadata comments
        let cleaned = text
            .lines()
            .filter(|line| !line.trim().starts_with("// @code:"))
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

        if graph.input_nodes().is_empty() {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Empty code graph".to_string(),
                node: None,
            });
        }

        Ok(issues)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        // For now, just return the text representation
        // Future: actual code execution or compilation
        let text = graph.to_text();
        Ok(ExecutionResult::Text(format!("Code: {}", text)))
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        vec![
            DomainRule {
                id: 0,
                domain: "code".to_string(),
                name: "Dead Code Elimination".to_string(),
                description: "Remove unreachable code".to_string(),
                category: "optimization".to_string(),
            },
            DomainRule {
                id: 1,
                domain: "code".to_string(),
                name: "Constant Folding".to_string(),
                description: "Evaluate constant expressions at compile time".to_string(),
                category: "optimization".to_string(),
            },
            DomainRule {
                id: 2,
                domain: "code".to_string(),
                name: "Inline Expansion".to_string(),
                description: "Replace function calls with function body".to_string(),
                category: "optimization".to_string(),
            },
            DomainRule {
                id: 3,
                domain: "code".to_string(),
                name: "Loop Unrolling".to_string(),
                description: "Expand loop iterations".to_string(),
                category: "optimization".to_string(),
            },
            DomainRule {
                id: 4,
                domain: "code".to_string(),
                name: "Type Inference".to_string(),
                description: "Infer types for untyped variables".to_string(),
                category: "analysis".to_string(),
            },
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        match rule_id {
            0 => self.apply_dead_code_elimination(graph),
            1 => self.apply_constant_folding(graph),
            2 => self.apply_inline_expansion(graph),
            3 => self.apply_loop_unrolling(graph),
            4 => self.apply_type_inference(graph),
            _ => Err(grapheme_core::DomainError::InvalidInput(format!(
                "Unknown rule ID: {}",
                rule_id
            ))),
        }
    }

    fn generate_examples(&self, count: usize) -> Vec<DomainExample> {
        let mut examples = Vec::with_capacity(count);

        let patterns = [
            ("1 + 2", "3"),
            ("x * 2", "x * 2"),
            ("true", "true"),
            ("5 - 3", "2"),
            ("10 / 2", "5"),
        ];

        for i in 0..count {
            let (input, output) = patterns[i % patterns.len()];

            if let (Ok(input_graph), Ok(output_graph)) =
                (DagNN::from_text(input), DagNN::from_text(output))
            {
                examples.push(DomainExample {
                    input: input_graph,
                    output: output_graph,
                    domain: "code".to_string(),
                    difficulty: ((i % 5) + 1) as u8,
                });
            }
        }

        examples
    }
}

// ============================================================================
// Tree-sitter Integration (Optional Feature)
// ============================================================================

/// Tree-sitter based multi-language parser for converting source code to CodeGraph.
///
/// This module is only available when the `tree-sitter-parsing` feature is enabled.
/// It provides production-grade parsing for multiple programming languages.
#[cfg(feature = "tree-sitter-parsing")]
pub mod tree_sitter_parser {
    use super::*;

    /// Tree-sitter based parser that converts source code ASTs to CodeGraph.
    pub struct TreeSitterParser {
        parser: tree_sitter::Parser,
        language: Language,
    }

    /// Helper function to create a parse error
    fn parse_error(message: &str) -> CodeGraphError {
        CodeGraphError::ParseError {
            line: 0,
            column: 0,
            message: message.to_string(),
        }
    }

    /// Get tree-sitter language from our Language enum
    fn get_ts_language(lang: Language) -> tree_sitter::Language {
        match lang {
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
            Language::Python => tree_sitter_python::LANGUAGE.into(),
            Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Language::C => tree_sitter_c::LANGUAGE.into(),
            Language::Generic => tree_sitter_rust::LANGUAGE.into(), // Default to Rust
        }
    }

    impl TreeSitterParser {
        /// Create a new parser for the specified language.
        pub fn new(language: Language) -> CodeGraphResult<Self> {
            let mut parser = tree_sitter::Parser::new();
            parser
                .set_language(&get_ts_language(language))
                .map_err(|e| parse_error(&format!("Failed to set language: {}", e)))?;
            Ok(Self { parser, language })
        }

        /// Parse Rust source code into a CodeGraph.
        pub fn parse_rust(code: &str) -> CodeGraphResult<CodeGraph> {
            let mut parser = Self::new(Language::Rust)?;
            parser.parse(code)
        }

        /// Parse Python source code into a CodeGraph.
        pub fn parse_python(code: &str) -> CodeGraphResult<CodeGraph> {
            let mut parser = Self::new(Language::Python)?;
            parser.parse(code)
        }

        /// Parse JavaScript source code into a CodeGraph.
        pub fn parse_javascript(code: &str) -> CodeGraphResult<CodeGraph> {
            let mut parser = Self::new(Language::JavaScript)?;
            parser.parse(code)
        }

        /// Parse C source code into a CodeGraph.
        pub fn parse_c(code: &str) -> CodeGraphResult<CodeGraph> {
            let mut parser = Self::new(Language::C)?;
            parser.parse(code)
        }

        /// Parse source code into a CodeGraph.
        pub fn parse(&mut self, code: &str) -> CodeGraphResult<CodeGraph> {
            let tree = self
                .parser
                .parse(code, None)
                .ok_or_else(|| parse_error("Failed to parse code"))?;

            self.convert_tree_to_code_graph(tree.root_node(), code)
        }

        /// Convert a tree-sitter AST to a CodeGraph.
        fn convert_tree_to_code_graph(
            &self,
            root: tree_sitter::Node,
            source: &str,
        ) -> CodeGraphResult<CodeGraph> {
            let mut code_graph = CodeGraph::with_language(self.language);
            let mut child_index = 0;

            // Recursively convert nodes
            if let Some(root_idx) =
                self.convert_node(&mut code_graph, root, source, &mut child_index)?
            {
                code_graph.root = Some(root_idx);
            }

            Ok(code_graph)
        }

        /// Recursively convert a tree-sitter node to CodeGraph nodes.
        fn convert_node(
            &self,
            graph: &mut CodeGraph,
            node: tree_sitter::Node,
            source: &str,
            child_index: &mut usize,
        ) -> CodeGraphResult<Option<NodeIndex>> {
            let node_text = node
                .utf8_text(source.as_bytes())
                .map_err(|e| parse_error(&format!("UTF-8 error: {}", e)))?;
            let kind = node.kind();

            // Map tree-sitter node kinds to CodeNode types
            let code_node = self.map_node_kind(kind, node_text);

            // Add the node to the graph
            let node_idx = graph.graph.add_node(code_node);

            // Process children and add edges
            let mut cursor = node.walk();
            let mut local_child_idx = 0usize;
            for child in node.children(&mut cursor) {
                // Skip comment and whitespace nodes for cleaner graphs
                if child.kind() == "comment"
                    || child.kind() == "line_comment"
                    || child.kind() == "block_comment"
                {
                    continue;
                }

                if let Some(child_node_idx) =
                    self.convert_node(graph, child, source, &mut local_child_idx)?
                {
                    graph.graph.add_edge(
                        node_idx,
                        child_node_idx,
                        CodeEdge::Child(local_child_idx),
                    );
                    local_child_idx += 1;
                }
            }

            *child_index += 1;
            Ok(Some(node_idx))
        }

        /// Map tree-sitter node kinds to CodeNode types.
        fn map_node_kind(&self, kind: &str, text: &str) -> CodeNode {
            match self.language {
                Language::Rust => self.map_rust_node(kind, text),
                Language::Python => self.map_python_node(kind, text),
                Language::JavaScript => self.map_javascript_node(kind, text),
                Language::C => self.map_c_node(kind, text),
                Language::Generic => self.map_rust_node(kind, text),
            }
        }

        /// Map Rust-specific AST nodes.
        fn map_rust_node(&self, kind: &str, text: &str) -> CodeNode {
            let node_type = match kind {
                "identifier" | "type_identifier" | "field_identifier" => {
                    CodeNodeType::Identifier(text.to_string())
                }
                "integer_literal" => {
                    CodeNodeType::Literal(LiteralValue::Integer(text.parse().unwrap_or(0)))
                }
                "float_literal" => {
                    CodeNodeType::Literal(LiteralValue::Float(text.parse().unwrap_or(0.0)))
                }
                "string_literal" | "raw_string_literal" => {
                    CodeNodeType::Literal(LiteralValue::String(text.trim_matches('"').to_string()))
                }
                "char_literal" => CodeNodeType::Literal(LiteralValue::String(
                    text.chars()
                        .nth(1)
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                )),
                "boolean_literal" | "true" | "false" => {
                    CodeNodeType::Literal(LiteralValue::Boolean(text == "true"))
                }
                "function_item" | "function_signature_item" => CodeNodeType::Function {
                    name: String::new(),
                    params: vec![],
                    return_type: None,
                },
                "call_expression" => CodeNodeType::Call {
                    function: String::new(),
                    arg_count: 0,
                },
                "binary_expression" => CodeNodeType::BinaryOp(BinaryOperator::Add),
                "unary_expression" => CodeNodeType::UnaryOp(UnaryOperator::Neg),
                "if_expression" => CodeNodeType::If,
                "loop_expression" | "while_expression" | "for_expression" => CodeNodeType::Loop {
                    kind: LoopKind::While,
                },
                "let_declaration" => CodeNodeType::Variable {
                    name: String::new(),
                    var_type: None,
                },
                "return_expression" => CodeNodeType::Return,
                "block" => CodeNodeType::Block,
                _ => CodeNodeType::Comment(format!(
                    "{}:{}",
                    kind,
                    text.chars().take(50).collect::<String>()
                )),
            };
            CodeNode::new(node_type)
        }

        /// Map Python-specific AST nodes.
        fn map_python_node(&self, kind: &str, text: &str) -> CodeNode {
            let node_type = match kind {
                "identifier" => CodeNodeType::Identifier(text.to_string()),
                "integer" => CodeNodeType::Literal(LiteralValue::Integer(text.parse().unwrap_or(0))),
                "float" => CodeNodeType::Literal(LiteralValue::Float(text.parse().unwrap_or(0.0))),
                "string" | "concatenated_string" => CodeNodeType::Literal(LiteralValue::String(
                    text.trim_matches(|c| c == '"' || c == '\'').to_string(),
                )),
                "true" | "false" => {
                    CodeNodeType::Literal(LiteralValue::Boolean(text == "True" || text == "true"))
                }
                "none" => CodeNodeType::Literal(LiteralValue::Null),
                "function_definition" => CodeNodeType::Function {
                    name: String::new(),
                    params: vec![],
                    return_type: None,
                },
                "call" => CodeNodeType::Call {
                    function: String::new(),
                    arg_count: 0,
                },
                "binary_operator" => CodeNodeType::BinaryOp(BinaryOperator::Add),
                "unary_operator" => CodeNodeType::UnaryOp(UnaryOperator::Neg),
                "if_statement" => CodeNodeType::If,
                "while_statement" | "for_statement" => CodeNodeType::Loop {
                    kind: LoopKind::While,
                },
                "assignment" | "augmented_assignment" => CodeNodeType::Assignment,
                "return_statement" => CodeNodeType::Return,
                "block" => CodeNodeType::Block,
                _ => CodeNodeType::Comment(format!(
                    "{}:{}",
                    kind,
                    text.chars().take(50).collect::<String>()
                )),
            };
            CodeNode::new(node_type)
        }

        /// Map JavaScript-specific AST nodes.
        fn map_javascript_node(&self, kind: &str, text: &str) -> CodeNode {
            let node_type = match kind {
                "identifier" | "property_identifier" => CodeNodeType::Identifier(text.to_string()),
                "number" => {
                    if text.contains('.') {
                        CodeNodeType::Literal(LiteralValue::Float(text.parse().unwrap_or(0.0)))
                    } else {
                        CodeNodeType::Literal(LiteralValue::Integer(text.parse().unwrap_or(0)))
                    }
                }
                "string" | "template_string" => CodeNodeType::Literal(LiteralValue::String(
                    text.trim_matches(|c| c == '"' || c == '\'' || c == '`')
                        .to_string(),
                )),
                "true" | "false" => CodeNodeType::Literal(LiteralValue::Boolean(text == "true")),
                "null" | "undefined" => CodeNodeType::Literal(LiteralValue::Null),
                "function_declaration" | "function" | "arrow_function" => CodeNodeType::Function {
                    name: String::new(),
                    params: vec![],
                    return_type: None,
                },
                "call_expression" => CodeNodeType::Call {
                    function: String::new(),
                    arg_count: 0,
                },
                "binary_expression" => CodeNodeType::BinaryOp(BinaryOperator::Add),
                "unary_expression" => CodeNodeType::UnaryOp(UnaryOperator::Neg),
                "if_statement" => CodeNodeType::If,
                "while_statement" | "for_statement" | "for_in_statement" | "do_statement" => {
                    CodeNodeType::Loop {
                        kind: LoopKind::While,
                    }
                }
                "variable_declaration" | "lexical_declaration" => CodeNodeType::Variable {
                    name: String::new(),
                    var_type: None,
                },
                "return_statement" => CodeNodeType::Return,
                "statement_block" => CodeNodeType::Block,
                _ => CodeNodeType::Comment(format!(
                    "{}:{}",
                    kind,
                    text.chars().take(50).collect::<String>()
                )),
            };
            CodeNode::new(node_type)
        }

        /// Map C-specific AST nodes.
        fn map_c_node(&self, kind: &str, text: &str) -> CodeNode {
            let node_type = match kind {
                "identifier" | "type_identifier" | "field_identifier" => {
                    CodeNodeType::Identifier(text.to_string())
                }
                "number_literal" => {
                    if text.contains('.') {
                        CodeNodeType::Literal(LiteralValue::Float(text.parse().unwrap_or(0.0)))
                    } else {
                        CodeNodeType::Literal(LiteralValue::Integer(
                            i64::from_str_radix(
                                text.trim_start_matches("0x").trim_start_matches("0X"),
                                if text.starts_with("0x") || text.starts_with("0X") {
                                    16
                                } else {
                                    10
                                },
                            )
                            .unwrap_or(0),
                        ))
                    }
                }
                "string_literal" => {
                    CodeNodeType::Literal(LiteralValue::String(text.trim_matches('"').to_string()))
                }
                "char_literal" => CodeNodeType::Literal(LiteralValue::String(
                    text.chars()
                        .nth(1)
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                )),
                "true" | "false" => CodeNodeType::Literal(LiteralValue::Boolean(text == "true")),
                "null" | "NULL" => CodeNodeType::Literal(LiteralValue::Null),
                "function_definition" | "function_declarator" => CodeNodeType::Function {
                    name: String::new(),
                    params: vec![],
                    return_type: None,
                },
                "call_expression" => CodeNodeType::Call {
                    function: String::new(),
                    arg_count: 0,
                },
                "binary_expression" => CodeNodeType::BinaryOp(BinaryOperator::Add),
                "unary_expression" => CodeNodeType::UnaryOp(UnaryOperator::Neg),
                "if_statement" => CodeNodeType::If,
                "while_statement" | "for_statement" | "do_statement" => CodeNodeType::Loop {
                    kind: LoopKind::While,
                },
                "declaration" | "init_declarator" => CodeNodeType::Variable {
                    name: String::new(),
                    var_type: None,
                },
                "return_statement" => CodeNodeType::Return,
                "compound_statement" => CodeNodeType::Block,
                _ => CodeNodeType::Comment(format!(
                    "{}:{}",
                    kind,
                    text.chars().take(50).collect::<String>()
                )),
            };
            CodeNode::new(node_type)
        }
    }
}

// Re-export tree-sitter types when feature is enabled
#[cfg(feature = "tree-sitter-parsing")]
pub use tree_sitter_parser::TreeSitterParser;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_graph_from_literal() {
        let graph = CodeGraph::from_simple_expr("42").unwrap();
        assert_eq!(graph.node_count(), 1);
        assert!(matches!(
            &graph.graph[graph.root.unwrap()],
            CodeNode { node_type: CodeNodeType::Literal(LiteralValue::Integer(42)), .. }
        ));
    }

    #[test]
    fn test_code_graph_from_boolean() {
        let graph = CodeGraph::from_simple_expr("true").unwrap();
        assert!(matches!(
            &graph.graph[graph.root.unwrap()],
            CodeNode { node_type: CodeNodeType::Literal(LiteralValue::Boolean(true)), .. }
        ));
    }

    #[test]
    fn test_code_graph_from_binary_op() {
        let graph = CodeGraph::from_simple_expr("1 + 2").unwrap();
        assert!(graph.node_count() >= 1);
        assert!(matches!(
            &graph.graph[graph.root.unwrap()],
            CodeNode { node_type: CodeNodeType::BinaryOp(BinaryOperator::Add), .. }
        ));
    }

    #[test]
    fn test_code_brain_creation() {
        let brain = CodeBrain::new();
        assert_eq!(brain.domain_id(), "code");
        assert_eq!(brain.domain_name(), "Source Code");
    }

    #[test]
    fn test_code_brain_can_process() {
        let brain = CodeBrain::new();
        assert!(brain.can_process("fn main() {}"));
        assert!(brain.can_process("def foo():"));
        assert!(brain.can_process("let x = 5;"));
        assert!(!brain.can_process("hello world"));
    }

    #[test]
    fn test_detect_language() {
        let brain = CodeBrain::new();
        assert_eq!(brain.detect_language("fn main() -> i32 {}"), Language::Rust);
        assert_eq!(brain.detect_language("def foo():"), Language::Python);
        assert_eq!(
            brain.detect_language("function bar() {}"),
            Language::JavaScript
        );
        assert_eq!(brain.detect_language("#include <stdio.h>"), Language::C);
    }

    #[test]
    fn test_code_brain_get_rules() {
        let brain = CodeBrain::new();
        let rules = brain.get_rules();
        assert_eq!(rules.len(), 5);
        assert_eq!(rules[0].domain, "code");
    }

    #[test]
    fn test_code_brain_generate_examples() {
        let brain = CodeBrain::new();
        let examples = brain.generate_examples(10);
        assert_eq!(examples.len(), 10);
        for example in &examples {
            assert_eq!(example.domain, "code");
        }
    }

    #[test]
    fn test_validate_code() {
        let brain = CodeBrain::new();
        let mut graph = CodeGraph::new();
        let issues = brain.validate_code(&graph);
        assert_eq!(issues.len(), 1); // Empty graph warning

        // Add a node to make it non-empty
        graph.add_node(CodeNode::new(CodeNodeType::Literal(LiteralValue::Integer(42))));
        let issues = brain.validate_code(&graph);
        assert!(issues.is_empty());
    }

    // ========================================================================
    // Type Inference Tests
    // ========================================================================

    #[test]
    fn test_inferred_type_display() {
        assert_eq!(format!("{}", InferredType::Int), "Int");
        assert_eq!(format!("{}", InferredType::Float), "Float");
        assert_eq!(format!("{}", InferredType::Bool), "Bool");
        assert_eq!(format!("{}", InferredType::String), "String");
        assert_eq!(format!("{}", InferredType::Unit), "()");
        assert_eq!(format!("{}", InferredType::Null), "Null");
        assert_eq!(format!("{}", InferredType::Unknown(0)), "?T0");
        assert_eq!(
            format!("{}", InferredType::Array(Box::new(InferredType::Int))),
            "[Int]"
        );
        assert_eq!(
            format!(
                "{}",
                InferredType::Function {
                    params: vec![InferredType::Int, InferredType::Int],
                    ret: Box::new(InferredType::Int),
                }
            ),
            "(Int, Int) -> Int"
        );
    }

    #[test]
    fn test_inferred_type_from_string() {
        assert_eq!(InferredType::from_type_string("i32"), InferredType::Int);
        assert_eq!(InferredType::from_type_string("i64"), InferredType::Int);
        assert_eq!(InferredType::from_type_string("f64"), InferredType::Float);
        assert_eq!(InferredType::from_type_string("bool"), InferredType::Bool);
        assert_eq!(
            InferredType::from_type_string("String"),
            InferredType::String
        );
        assert_eq!(InferredType::from_type_string("()"), InferredType::Unit);
        assert_eq!(
            InferredType::from_type_string("Vec<i32>"),
            InferredType::Array(Box::new(InferredType::Int))
        );
        assert_eq!(
            InferredType::from_type_string("&i32"),
            InferredType::Ref(Box::new(InferredType::Int))
        );
    }

    #[test]
    fn test_type_contains_var() {
        let t0 = InferredType::Unknown(0);
        assert!(t0.contains_var(0));
        assert!(!t0.contains_var(1));

        let arr = InferredType::Array(Box::new(InferredType::Unknown(0)));
        assert!(arr.contains_var(0));

        let func = InferredType::Function {
            params: vec![InferredType::Unknown(1)],
            ret: Box::new(InferredType::Unknown(2)),
        };
        assert!(func.contains_var(1));
        assert!(func.contains_var(2));
        assert!(!func.contains_var(0));
    }

    #[test]
    fn test_type_substitution() {
        let t0 = InferredType::Unknown(0);
        let result = t0.substitute(0, &InferredType::Int);
        assert_eq!(result, InferredType::Int);

        let arr = InferredType::Array(Box::new(InferredType::Unknown(0)));
        let result = arr.substitute(0, &InferredType::Float);
        assert_eq!(result, InferredType::Array(Box::new(InferredType::Float)));
    }

    #[test]
    fn test_type_inference_engine_fresh_var() {
        let mut engine = TypeInferenceEngine::new();
        let v1 = engine.fresh_var();
        let v2 = engine.fresh_var();
        assert_eq!(v1, InferredType::Unknown(0));
        assert_eq!(v2, InferredType::Unknown(1));
    }

    #[test]
    fn test_type_inference_engine_unify_same_types() {
        let mut engine = TypeInferenceEngine::new();
        assert!(engine.unify(&InferredType::Int, &InferredType::Int).is_ok());
        assert!(engine
            .unify(&InferredType::Bool, &InferredType::Bool)
            .is_ok());
    }

    #[test]
    fn test_type_inference_engine_unify_type_var() {
        let mut engine = TypeInferenceEngine::new();
        let t0 = engine.fresh_var();
        assert!(engine.unify(&t0, &InferredType::Int).is_ok());
        // After unification, t0 should resolve to Int
        assert_eq!(engine.apply_substitutions(&t0), InferredType::Int);
    }

    #[test]
    fn test_type_inference_engine_unify_numeric_coercion() {
        let mut engine = TypeInferenceEngine::new();
        // Int and Float can coerce
        assert!(engine
            .unify(&InferredType::Int, &InferredType::Float)
            .is_ok());
    }

    #[test]
    fn test_type_inference_engine_unify_fails() {
        let mut engine = TypeInferenceEngine::new();
        // Int and Bool cannot unify
        assert!(engine
            .unify(&InferredType::Int, &InferredType::Bool)
            .is_err());
        // String and Int cannot unify
        assert!(engine
            .unify(&InferredType::String, &InferredType::Int)
            .is_err());
    }

    #[test]
    fn test_type_inference_engine_unify_functions() {
        let mut engine = TypeInferenceEngine::new();
        let f1 = InferredType::Function {
            params: vec![InferredType::Int],
            ret: Box::new(InferredType::Int),
        };
        let f2 = InferredType::Function {
            params: vec![InferredType::Int],
            ret: Box::new(InferredType::Int),
        };
        assert!(engine.unify(&f1, &f2).is_ok());

        // Arity mismatch
        let f3 = InferredType::Function {
            params: vec![InferredType::Int, InferredType::Int],
            ret: Box::new(InferredType::Int),
        };
        assert!(engine.unify(&f1, &f3).is_err());
    }

    #[test]
    fn test_type_inference_engine_occurs_check() {
        let mut engine = TypeInferenceEngine::new();
        let t0 = engine.fresh_var();
        // Try to unify ?T0 with [?T0] - should fail (infinite type)
        let arr = InferredType::Array(Box::new(t0.clone()));
        assert!(engine.unify(&t0, &arr).is_err());
    }

    #[test]
    fn test_infer_integer_literal() {
        let graph = CodeGraph::from_simple_expr("42").unwrap();
        let types = graph.infer_types();
        let root_type = types.get(&graph.root.unwrap()).unwrap();
        assert_eq!(*root_type, InferredType::Int);
    }

    #[test]
    fn test_infer_float_literal() {
        let graph = CodeGraph::from_simple_expr("3.14").unwrap();
        let types = graph.infer_types();
        let root_type = types.get(&graph.root.unwrap()).unwrap();
        assert_eq!(*root_type, InferredType::Float);
    }

    #[test]
    fn test_infer_boolean_literal() {
        let graph = CodeGraph::from_simple_expr("true").unwrap();
        let types = graph.infer_types();
        let root_type = types.get(&graph.root.unwrap()).unwrap();
        assert_eq!(*root_type, InferredType::Bool);
    }

    #[test]
    fn test_infer_binary_add() {
        let graph = CodeGraph::from_simple_expr("1 + 2").unwrap();
        let types = graph.infer_types();
        let root_type = types.get(&graph.root.unwrap()).unwrap();
        // Addition of integers should be Int
        assert_eq!(*root_type, InferredType::Int);
    }

    #[test]
    fn test_infer_comparison_returns_bool() {
        let graph = CodeGraph::from_simple_expr("1 < 2").unwrap();
        let types = graph.infer_types();
        let root_type = types.get(&graph.root.unwrap()).unwrap();
        assert_eq!(*root_type, InferredType::Bool);
    }

    #[test]
    fn test_infer_equality_returns_bool() {
        let graph = CodeGraph::from_simple_expr("x == y").unwrap();
        let types = graph.infer_types();
        let root_type = types.get(&graph.root.unwrap()).unwrap();
        assert_eq!(*root_type, InferredType::Bool);
    }

    #[test]
    fn test_code_graph_is_well_typed() {
        let graph = CodeGraph::from_simple_expr("1 + 2").unwrap();
        assert!(graph.is_well_typed());
    }

    #[test]
    fn test_code_graph_infer_types_with_errors() {
        let graph = CodeGraph::from_simple_expr("42").unwrap();
        let (types, errors) = graph.infer_types_with_errors();
        assert!(errors.is_empty());
        assert!(!types.is_empty());
    }

    #[test]
    fn test_type_inference_result() {
        let graph = CodeGraph::from_simple_expr("42").unwrap();
        let types = graph.infer_types();
        let result = TypeInferenceResult {
            node_types: types.clone(),
            errors: vec![],
            success: true,
        };
        assert!(result.success);
        assert!(result.has_concrete_type(graph.root.unwrap()));
    }

    #[test]
    fn test_code_graph_annotate_type() {
        let mut graph = CodeGraph::new();
        let var_node = graph.add_node(CodeNode::new(CodeNodeType::Variable {
            name: "x".to_string(),
            var_type: None,
        }));
        let type_node = graph.annotate_type(var_node, "i32");
        // Check that the type node was added (node_count should be 2)
        assert_eq!(graph.node_count(), 2);
        // Check the type node is a Type node
        assert!(matches!(&graph.graph[type_node], CodeNode { node_type: CodeNodeType::Type(s), .. } if s == "i32"));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_code_brain_type_inference_transform() {
        use grapheme_core::DomainBrain;

        let brain = CodeBrain::new();
        let input = grapheme_core::DagNN::from_text("42").unwrap();
        let result = brain.transform(&input, 4).unwrap(); // Rule 4 is Type Inference
        let text = result.to_text();
        assert!(text.contains("Int"), "Expected type annotation: {}", text);
    }

    #[test]
    fn test_code_brain_type_inference_bool() {
        use grapheme_core::DomainBrain;

        let brain = CodeBrain::new();
        let input = grapheme_core::DagNN::from_text("true").unwrap();
        let result = brain.transform(&input, 4).unwrap();
        let text = result.to_text();
        assert!(text.contains("Bool"), "Expected Bool type: {}", text);
    }

    #[test]
    fn test_code_brain_type_inference_float() {
        use grapheme_core::DomainBrain;

        let brain = CodeBrain::new();
        let input = grapheme_core::DagNN::from_text("3.14").unwrap();
        let result = brain.transform(&input, 4).unwrap();
        let text = result.to_text();
        assert!(text.contains("Float"), "Expected Float type: {}", text);
    }

    #[test]
    fn test_environment_binding() {
        let mut engine = TypeInferenceEngine::new();
        engine.bind_var("x", InferredType::Int);
        assert_eq!(engine.lookup_var("x"), Some(&InferredType::Int));
        assert_eq!(engine.lookup_var("y"), None);
    }

    #[test]
    fn test_infer_variable_with_type() {
        let mut graph = CodeGraph::new();
        let var = graph.add_node(CodeNode::new(CodeNodeType::Variable {
            name: "x".to_string(),
            var_type: Some("i32".to_string()),
        }));
        graph.root = Some(var);

        let types = graph.infer_types();
        assert_eq!(types.get(&var), Some(&InferredType::Int));
    }

    #[test]
    fn test_infer_function_type() {
        let mut graph = CodeGraph::new();
        let func = graph.add_node(CodeNode::new(CodeNodeType::Function {
            name: "add".to_string(),
            params: vec!["a".to_string(), "b".to_string()],
            return_type: Some("i32".to_string()),
        }));
        graph.root = Some(func);

        let types = graph.infer_types();
        let func_type = types.get(&func).unwrap();
        match func_type {
            InferredType::Function { params, ret } => {
                assert_eq!(params.len(), 2);
                assert_eq!(**ret, InferredType::Int);
            }
            _ => panic!("Expected function type"),
        }
    }
}

// Tree-sitter tests (only when feature is enabled)
#[cfg(all(test, feature = "tree-sitter-parsing"))]
mod tree_sitter_tests {
    use super::*;

    #[test]
    fn test_parse_rust_simple() {
        let code = "fn main() { let x = 42; }";
        let graph = TreeSitterParser::parse_rust(code).unwrap();
        assert!(graph.root.is_some());
        assert!(graph.node_count() > 0);
        assert_eq!(graph.language, Language::Rust);
    }

    #[test]
    fn test_parse_rust_function() {
        let code = r#"
            fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        "#;
        let graph = TreeSitterParser::parse_rust(code).unwrap();
        assert!(graph.root.is_some());
        // Should have function node, identifiers, binary op, etc.
        assert!(graph.node_count() >= 5);
    }

    #[test]
    fn test_parse_python_simple() {
        let code = "def hello(): return 42";
        let graph = TreeSitterParser::parse_python(code).unwrap();
        assert!(graph.root.is_some());
        assert!(graph.node_count() > 0);
        assert_eq!(graph.language, Language::Python);
    }

    #[test]
    fn test_parse_javascript_simple() {
        let code = "function add(a, b) { return a + b; }";
        let graph = TreeSitterParser::parse_javascript(code).unwrap();
        assert!(graph.root.is_some());
        assert!(graph.node_count() > 0);
        assert_eq!(graph.language, Language::JavaScript);
    }

    #[test]
    fn test_parse_c_simple() {
        let code = "int main() { return 0; }";
        let graph = TreeSitterParser::parse_c(code).unwrap();
        assert!(graph.root.is_some());
        assert!(graph.node_count() > 0);
        assert_eq!(graph.language, Language::C);
    }

    #[test]
    fn test_parser_creates_edges() {
        let code = "fn main() { 1 + 2 }";
        let graph = TreeSitterParser::parse_rust(code).unwrap();
        assert!(graph.edge_count() > 0);
    }
}
