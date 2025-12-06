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
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule,
    ExecutionResult, ValidationIssue, ValidationSeverity,
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

/// AST node types for code representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CodeNode {
    /// Module/file root
    Module { name: String, language: Language },
    /// Function definition
    Function {
        name: String,
        params: Vec<String>,
        return_type: Option<String>,
    },
    /// Variable declaration
    Variable { name: String, var_type: Option<String> },
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
            let node = graph.add_node(CodeNode::Literal(LiteralValue::Integer(n)));
            graph.root = Some(node);
            return Ok(graph);
        }

        if let Ok(n) = trimmed.parse::<f64>() {
            let node = graph.add_node(CodeNode::Literal(LiteralValue::Float(n)));
            graph.root = Some(node);
            return Ok(graph);
        }

        // Try to parse as a boolean
        if trimmed == "true" {
            let node = graph.add_node(CodeNode::Literal(LiteralValue::Boolean(true)));
            graph.root = Some(node);
            return Ok(graph);
        }
        if trimmed == "false" {
            let node = graph.add_node(CodeNode::Literal(LiteralValue::Boolean(false)));
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
                    let op_node = graph.add_node(CodeNode::BinaryOp(op));

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
        let node = graph.add_node(CodeNode::Identifier(trimmed.to_string()));
        graph.root = Some(node);
        Ok(graph)
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
            "fn ", "def ", "function ", "func ",
            "let ", "var ", "const ",
            "if ", "else ", "while ", "for ",
            "return ", "class ", "struct ",
            "import ", "from ", "use ",
            "->", "=>", "::",
            "{", "}", "(", ")",
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
        let lines: Vec<&str> = normalized
            .lines()
            .map(|line| line.trim_end())
            .collect();

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
            if let CodeNode::Identifier(name) = &graph.graph[node_idx] {
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
                if let (Ok(a), Ok(b)) = (parts[0].trim().parse::<i64>(), parts[1].trim().parse::<i64>()) {
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
    fn apply_type_inference(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // For DagNN, we can annotate based on content patterns
        // This is a placeholder - real type inference requires full parsing
        Ok(graph.clone())
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
            _ => Err(grapheme_core::DomainError::InvalidInput(
                format!("Unknown rule ID: {}", rule_id)
            )),
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

            if let (Ok(input_graph), Ok(output_graph)) = (
                DagNN::from_text(input),
                DagNN::from_text(output),
            ) {
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
            CodeNode::Literal(LiteralValue::Integer(42))
        ));
    }

    #[test]
    fn test_code_graph_from_boolean() {
        let graph = CodeGraph::from_simple_expr("true").unwrap();
        assert!(matches!(
            &graph.graph[graph.root.unwrap()],
            CodeNode::Literal(LiteralValue::Boolean(true))
        ));
    }

    #[test]
    fn test_code_graph_from_binary_op() {
        let graph = CodeGraph::from_simple_expr("1 + 2").unwrap();
        assert!(graph.node_count() >= 1);
        assert!(matches!(
            &graph.graph[graph.root.unwrap()],
            CodeNode::BinaryOp(BinaryOperator::Add)
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
        assert_eq!(brain.detect_language("function bar() {}"), Language::JavaScript);
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
        graph.add_node(CodeNode::Literal(LiteralValue::Integer(42)));
        let issues = brain.validate_code(&graph);
        assert!(issues.is_empty());
    }
}
