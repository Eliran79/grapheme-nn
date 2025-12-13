//! Tree-sitter integration for multi-language parsing
//!
//! Provides robust AST parsing using tree-sitter for:
//! - Rust
//! - Python
//! - JavaScript/TypeScript
//! - C/C++
//!
//! Tree-sitter provides:
//! - Incremental parsing for editor integration
//! - Error-tolerant parsing (partial results for syntax errors)
//! - Consistent AST structure across languages

use crate::{BinaryOperator, CodeEdge, CodeGraph, CodeGraphError, CodeGraphResult, CodeNode, Language, LiteralValue, LoopKind, UnaryOperator};
use petgraph::graph::NodeIndex;
use tree_sitter::{Node, Parser, Tree};

/// Tree-sitter based parser for multiple languages
pub struct TreeSitterParser {
    /// Internal parser (reused across calls)
    parser: Parser,
}

impl Default for TreeSitterParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeSitterParser {
    /// Create a new tree-sitter parser
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
        }
    }

    /// Parse code with automatic language detection
    pub fn parse(&mut self, code: &str, language: Language) -> CodeGraphResult<CodeGraph> {
        match language {
            Language::Rust => self.parse_rust(code),
            Language::Python => self.parse_python(code),
            Language::JavaScript => self.parse_javascript(code),
            Language::C => self.parse_c(code),
            Language::Generic => self.parse_generic(code),
        }
    }

    /// Parse Rust code
    pub fn parse_rust(&mut self, code: &str) -> CodeGraphResult<CodeGraph> {
        self.parser.set_language(&tree_sitter_rust::language())
            .map_err(|e| CodeGraphError::InvalidSyntax(format!("Failed to set Rust language: {}", e)))?;

        let tree = self.parse_with_tree(code)?;
        self.convert_tree_to_code_graph(tree.root_node(), code, Language::Rust)
    }

    /// Parse Python code
    pub fn parse_python(&mut self, code: &str) -> CodeGraphResult<CodeGraph> {
        self.parser.set_language(&tree_sitter_python::language())
            .map_err(|e| CodeGraphError::InvalidSyntax(format!("Failed to set Python language: {}", e)))?;

        let tree = self.parse_with_tree(code)?;
        self.convert_tree_to_code_graph(tree.root_node(), code, Language::Python)
    }

    /// Parse JavaScript code
    pub fn parse_javascript(&mut self, code: &str) -> CodeGraphResult<CodeGraph> {
        self.parser.set_language(&tree_sitter_javascript::language())
            .map_err(|e| CodeGraphError::InvalidSyntax(format!("Failed to set JavaScript language: {}", e)))?;

        let tree = self.parse_with_tree(code)?;
        self.convert_tree_to_code_graph(tree.root_node(), code, Language::JavaScript)
    }

    /// Parse C code
    pub fn parse_c(&mut self, code: &str) -> CodeGraphResult<CodeGraph> {
        self.parser.set_language(&tree_sitter_c::language())
            .map_err(|e| CodeGraphError::InvalidSyntax(format!("Failed to set C language: {}", e)))?;

        let tree = self.parse_with_tree(code)?;
        self.convert_tree_to_code_graph(tree.root_node(), code, Language::C)
    }

    /// Parse with generic fallback (uses simple expression parser)
    pub fn parse_generic(&mut self, code: &str) -> CodeGraphResult<CodeGraph> {
        // Fallback to simple expression parser for generic code
        CodeGraph::from_simple_expr(code)
    }

    /// Parse and return the tree-sitter tree
    fn parse_with_tree(&mut self, code: &str) -> CodeGraphResult<Tree> {
        self.parser.parse(code, None)
            .ok_or_else(|| CodeGraphError::InvalidSyntax("Failed to parse code".to_string()))
    }

    /// Convert a tree-sitter tree to a CodeGraph
    fn convert_tree_to_code_graph(
        &self,
        root: Node,
        source: &str,
        language: Language,
    ) -> CodeGraphResult<CodeGraph> {
        let mut graph = CodeGraph::with_language(language);

        // Create root module node
        let module_name = match language {
            Language::Rust => "rust_module",
            Language::Python => "python_module",
            Language::JavaScript => "js_module",
            Language::C => "c_module",
            Language::Generic => "module",
        };

        let root_node = graph.add_node(CodeNode::Module {
            name: module_name.to_string(),
            language,
        });
        graph.root = Some(root_node);

        // Recursively convert children
        self.convert_node(&mut graph, root, source, root_node, 0);

        Ok(graph)
    }

    /// Recursively convert a tree-sitter node to CodeGraph nodes
    fn convert_node(
        &self,
        graph: &mut CodeGraph,
        ts_node: Node,
        source: &str,
        parent: NodeIndex,
        child_index: usize,
    ) -> Option<NodeIndex> {
        let kind = ts_node.kind();

        // Skip certain node types
        if kind == "source_file" || kind == "program" || kind == "translation_unit" {
            // Just process children for root container nodes
            let mut cursor = ts_node.walk();
            for (i, child) in ts_node.children(&mut cursor).enumerate() {
                if let Some(child_node) = self.convert_node(graph, child, source, parent, i) {
                    graph.add_edge(parent, child_node, CodeEdge::Child(i));
                }
            }
            return None;
        }

        // Get the text for this node
        let text = ts_node.utf8_text(source.as_bytes()).unwrap_or("");

        // Convert node based on kind
        let code_node = self.ts_kind_to_code_node(kind, text, ts_node, source);

        if let Some(node) = code_node {
            let node_idx = graph.add_node(node);

            // Add edge from parent
            graph.add_edge(parent, node_idx, CodeEdge::Child(child_index));

            // Process children
            let mut cursor = ts_node.walk();
            let mut prev_child: Option<NodeIndex> = None;

            for (i, child) in ts_node.named_children(&mut cursor).enumerate() {
                if let Some(child_node) = self.convert_node(graph, child, source, node_idx, i) {
                    // Add Next edge for sequential children
                    if let Some(prev) = prev_child {
                        graph.add_edge(prev, child_node, CodeEdge::Next);
                    }
                    prev_child = Some(child_node);
                }
            }

            Some(node_idx)
        } else {
            // Skip this node but process children
            let mut cursor = ts_node.walk();
            for (i, child) in ts_node.named_children(&mut cursor).enumerate() {
                if let Some(child_node) = self.convert_node(graph, child, source, parent, i) {
                    graph.add_edge(parent, child_node, CodeEdge::Child(i));
                }
            }
            None
        }
    }

    /// Convert tree-sitter node kind to CodeNode
    fn ts_kind_to_code_node(&self, kind: &str, text: &str, ts_node: Node, source: &str) -> Option<CodeNode> {
        match kind {
            // Function definitions
            "function_item" | "function_definition" | "function_declaration" |
            "method_definition" | "arrow_function" | "lambda" => {
                let name = self.extract_function_name(ts_node, source).unwrap_or_else(|| "anonymous".to_string());
                let params = self.extract_function_params(ts_node, source);
                let return_type = self.extract_return_type(ts_node, source);
                Some(CodeNode::Function { name, params, return_type })
            }

            // Variable declarations
            "let_declaration" | "variable_declaration" | "const_declaration" |
            "assignment_expression" | "augmented_assignment" => {
                let name = self.extract_var_name(ts_node, source).unwrap_or_else(|| text.to_string());
                Some(CodeNode::Variable { name, var_type: None })
            }

            // Literals
            "integer_literal" | "number" | "integer" => {
                if let Ok(n) = text.parse::<i64>() {
                    Some(CodeNode::Literal(LiteralValue::Integer(n)))
                } else {
                    Some(CodeNode::Literal(LiteralValue::String(text.to_string())))
                }
            }

            "float_literal" | "float" => {
                if let Ok(n) = text.parse::<f64>() {
                    Some(CodeNode::Literal(LiteralValue::Float(n)))
                } else {
                    Some(CodeNode::Literal(LiteralValue::String(text.to_string())))
                }
            }

            "string_literal" | "string" | "raw_string_literal" | "string_content" => {
                Some(CodeNode::Literal(LiteralValue::String(text.to_string())))
            }

            "true" | "false" => {
                Some(CodeNode::Literal(LiteralValue::Boolean(text == "true")))
            }

            "none" | "null" | "nil" => {
                Some(CodeNode::Literal(LiteralValue::Null))
            }

            // Binary operators
            "binary_expression" | "comparison_operator" | "boolean_operator" => {
                let op = self.extract_binary_op(ts_node, source);
                op.map(CodeNode::BinaryOp)
            }

            // Unary operators
            "unary_expression" | "not_operator" => {
                let op = self.extract_unary_op(ts_node, source);
                op.map(CodeNode::UnaryOp)
            }

            // Control flow
            "if_statement" | "if_expression" | "if_let_expression" => {
                Some(CodeNode::If)
            }

            "for_statement" | "for_expression" | "for_in_clause" => {
                Some(CodeNode::Loop { kind: LoopKind::For })
            }

            "while_statement" | "while_expression" => {
                Some(CodeNode::Loop { kind: LoopKind::While })
            }

            "loop_expression" => {
                Some(CodeNode::Loop { kind: LoopKind::Loop })
            }

            "do_statement" => {
                Some(CodeNode::Loop { kind: LoopKind::DoWhile })
            }

            // Function calls
            "call_expression" | "function_call" => {
                let func_name = self.extract_call_name(ts_node, source).unwrap_or_else(|| "call".to_string());
                let arg_count = self.count_call_args(ts_node, source);
                Some(CodeNode::Call { function: func_name, arg_count })
            }

            // Return statements
            "return_statement" | "return_expression" => {
                Some(CodeNode::Return)
            }

            // Blocks
            "block" | "compound_statement" | "statement_block" | "suite" => {
                Some(CodeNode::Block)
            }

            // Identifiers
            "identifier" | "field_identifier" | "property_identifier" => {
                Some(CodeNode::Identifier(text.to_string()))
            }

            // Type annotations
            "type_identifier" | "primitive_type" | "type_annotation" => {
                Some(CodeNode::Type(text.to_string()))
            }

            // Comments
            "comment" | "line_comment" | "block_comment" => {
                Some(CodeNode::Comment(text.to_string()))
            }

            // Expression statements
            "expression_statement" => {
                Some(CodeNode::ExprStmt)
            }

            // Assignment
            "assignment" | "compound_assignment" => {
                Some(CodeNode::Assignment)
            }

            // Skip these node types
            "(" | ")" | "{" | "}" | "[" | "]" | "," | ";" | ":" | "=" |
            "ERROR" | "MISSING" | "keyword_argument" | "attribute" => None,

            // Default: try to create an identifier
            _ => {
                if text.len() <= 50 && !text.contains('\n') {
                    Some(CodeNode::Identifier(format!("{}:{}", kind, text)))
                } else {
                    None
                }
            }
        }
    }

    /// Extract function name from a function node
    fn extract_function_name(&self, node: Node, source: &str) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let kind = child.kind();
            if kind == "identifier" || kind == "name" || kind == "property_identifier" {
                return child.utf8_text(source.as_bytes()).ok().map(|s| s.to_string());
            }
        }
        None
    }

    /// Extract function parameters
    fn extract_function_params(&self, node: Node, source: &str) -> Vec<String> {
        let mut params = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind().contains("parameter") {
                let mut param_cursor = child.walk();
                for param_child in child.children(&mut param_cursor) {
                    if param_child.kind() == "identifier" {
                        if let Ok(name) = param_child.utf8_text(source.as_bytes()) {
                            params.push(name.to_string());
                        }
                    }
                }
            }
        }

        params
    }

    /// Extract return type from function
    fn extract_return_type(&self, node: Node, source: &str) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind().contains("return_type") || child.kind().contains("type") {
                return child.utf8_text(source.as_bytes()).ok().map(|s| s.to_string());
            }
        }
        None
    }

    /// Extract variable name
    fn extract_var_name(&self, node: Node, source: &str) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" || child.kind() == "pattern" {
                return child.utf8_text(source.as_bytes()).ok().map(|s| s.to_string());
            }
        }
        None
    }

    /// Extract binary operator from expression
    fn extract_binary_op(&self, node: Node, source: &str) -> Option<BinaryOperator> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let text = child.utf8_text(source.as_bytes()).unwrap_or("");
            match text {
                "+" => return Some(BinaryOperator::Add),
                "-" => return Some(BinaryOperator::Sub),
                "*" => return Some(BinaryOperator::Mul),
                "/" => return Some(BinaryOperator::Div),
                "%" => return Some(BinaryOperator::Mod),
                "==" => return Some(BinaryOperator::Eq),
                "!=" => return Some(BinaryOperator::Ne),
                "<" => return Some(BinaryOperator::Lt),
                "<=" => return Some(BinaryOperator::Le),
                ">" => return Some(BinaryOperator::Gt),
                ">=" => return Some(BinaryOperator::Ge),
                "&&" | "and" => return Some(BinaryOperator::And),
                "||" | "or" => return Some(BinaryOperator::Or),
                "&" => return Some(BinaryOperator::BitAnd),
                "|" => return Some(BinaryOperator::BitOr),
                "^" => return Some(BinaryOperator::BitXor),
                "<<" => return Some(BinaryOperator::Shl),
                ">>" => return Some(BinaryOperator::Shr),
                _ => {}
            }
        }
        Some(BinaryOperator::Add) // Default fallback
    }

    /// Extract unary operator from expression
    fn extract_unary_op(&self, node: Node, source: &str) -> Option<UnaryOperator> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let text = child.utf8_text(source.as_bytes()).unwrap_or("");
            match text {
                "-" => return Some(UnaryOperator::Neg),
                "!" | "not" => return Some(UnaryOperator::Not),
                "~" => return Some(UnaryOperator::BitNot),
                "*" => return Some(UnaryOperator::Deref),
                "&" => return Some(UnaryOperator::Ref),
                _ => {}
            }
        }
        Some(UnaryOperator::Neg) // Default fallback
    }

    /// Extract function name from call expression
    fn extract_call_name(&self, node: Node, source: &str) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let kind = child.kind();
            if kind == "identifier" || kind == "field_expression" || kind == "member_expression" {
                return child.utf8_text(source.as_bytes()).ok().map(|s| s.to_string());
            }
        }
        None
    }

    /// Count arguments in function call
    fn count_call_args(&self, node: Node, _source: &str) -> usize {
        let mut count = 0;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind().contains("argument") {
                count += child.named_child_count();
            }
        }
        count.max(1) // At least 1 if we found arguments block
    }

    /// Check if the tree has syntax errors
    pub fn has_errors(&mut self, code: &str, language: Language) -> bool {
        let result = match language {
            Language::Rust => {
                let _ = self.parser.set_language(&tree_sitter_rust::language());
                self.parser.parse(code, None)
            }
            Language::Python => {
                let _ = self.parser.set_language(&tree_sitter_python::language());
                self.parser.parse(code, None)
            }
            Language::JavaScript => {
                let _ = self.parser.set_language(&tree_sitter_javascript::language());
                self.parser.parse(code, None)
            }
            Language::C => {
                let _ = self.parser.set_language(&tree_sitter_c::language());
                self.parser.parse(code, None)
            }
            Language::Generic => return false,
        };

        match result {
            Some(tree) => tree.root_node().has_error(),
            None => true,
        }
    }

    /// Get error positions in the code
    pub fn get_error_positions(&mut self, code: &str, language: Language) -> Vec<(usize, usize)> {
        let tree = match language {
            Language::Rust => {
                let _ = self.parser.set_language(&tree_sitter_rust::language());
                self.parser.parse(code, None)
            }
            Language::Python => {
                let _ = self.parser.set_language(&tree_sitter_python::language());
                self.parser.parse(code, None)
            }
            Language::JavaScript => {
                let _ = self.parser.set_language(&tree_sitter_javascript::language());
                self.parser.parse(code, None)
            }
            Language::C => {
                let _ = self.parser.set_language(&tree_sitter_c::language());
                self.parser.parse(code, None)
            }
            Language::Generic => return vec![],
        };

        let mut errors = Vec::new();
        if let Some(tree) = tree {
            Self::collect_errors(tree.root_node(), &mut errors);
        }
        errors
    }

    /// Recursively collect error positions
    fn collect_errors(node: Node, errors: &mut Vec<(usize, usize)>) {
        if node.is_error() || node.is_missing() {
            let start = node.start_position();
            errors.push((start.row, start.column));
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::collect_errors(child, errors);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = TreeSitterParser::new();
        assert!(parser.parser.language().is_none());
    }

    #[test]
    fn test_parse_rust_function() {
        let mut parser = TreeSitterParser::new();
        let code = "fn main() { println!(\"Hello\"); }";
        let graph = parser.parse_rust(code).unwrap();

        assert!(graph.node_count() > 0);
        assert_eq!(graph.language, Language::Rust);
    }

    #[test]
    fn test_parse_python_function() {
        let mut parser = TreeSitterParser::new();
        let code = "def hello():\n    print('Hello')";
        let graph = parser.parse_python(code).unwrap();

        assert!(graph.node_count() > 0);
        assert_eq!(graph.language, Language::Python);
    }

    #[test]
    fn test_parse_javascript_function() {
        let mut parser = TreeSitterParser::new();
        let code = "function hello() { console.log('Hello'); }";
        let graph = parser.parse_javascript(code).unwrap();

        assert!(graph.node_count() > 0);
        assert_eq!(graph.language, Language::JavaScript);
    }

    #[test]
    fn test_parse_c_function() {
        let mut parser = TreeSitterParser::new();
        let code = "int main() { return 0; }";
        let graph = parser.parse_c(code).unwrap();

        assert!(graph.node_count() > 0);
        assert_eq!(graph.language, Language::C);
    }

    #[test]
    fn test_parse_with_language() {
        let mut parser = TreeSitterParser::new();
        let code = "x = 1 + 2";

        let graph = parser.parse(code, Language::Python).unwrap();
        assert_eq!(graph.language, Language::Python);
    }

    #[test]
    fn test_error_detection() {
        let mut parser = TreeSitterParser::new();

        // Valid Rust
        assert!(!parser.has_errors("fn main() {}", Language::Rust));

        // Invalid Rust (unclosed brace)
        assert!(parser.has_errors("fn main() {", Language::Rust));
    }

    #[test]
    fn test_error_positions() {
        let mut parser = TreeSitterParser::new();
        let code = "fn main() { let x = ; }"; // Missing expression

        let errors = parser.get_error_positions(code, Language::Rust);
        // Should find at least one error
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_parse_rust_with_variables() {
        let mut parser = TreeSitterParser::new();
        let code = r#"
            fn add(a: i32, b: i32) -> i32 {
                let sum = a + b;
                sum
            }
        "#;
        let graph = parser.parse_rust(code).unwrap();

        // Should have function, variable, binary op, etc.
        assert!(graph.node_count() > 5);
    }

    #[test]
    fn test_parse_python_with_loop() {
        let mut parser = TreeSitterParser::new();
        let code = r#"
def count():
    for i in range(10):
        print(i)
"#;
        let graph = parser.parse_python(code).unwrap();
        assert!(graph.node_count() > 3);
    }

    #[test]
    fn test_parse_javascript_arrow() {
        let mut parser = TreeSitterParser::new();
        let code = "const add = (a, b) => a + b;";
        let graph = parser.parse_javascript(code).unwrap();

        assert!(graph.node_count() > 0);
    }

    #[test]
    fn test_generic_fallback() {
        let mut parser = TreeSitterParser::new();
        let code = "1 + 2";
        let graph = parser.parse(code, Language::Generic).unwrap();

        assert!(graph.node_count() > 0);
    }
}
