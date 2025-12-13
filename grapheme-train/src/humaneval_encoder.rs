//! HumanEval Pre-Encoder (backend-228)
//!
//! Pre-encodes HumanEval problems to graph pairs for pure graph-to-graph training.
//! This eliminates text processing from the training loop.
//!
//! Workflow:
//! 1. Load HumanEval problems (JSONL format)
//! 2. Parse prompt → CodeGraph (input)
//! 3. Parse canonical_solution → CodeGraph (output)
//! 4. Store as GraphPair in GraphDataset
//! 5. Save to binary format for graph-only training

use crate::graph_data::{GraphDataResult, GraphDataset, GraphDataError, GraphPair, GraphPairBuilder};
use grapheme_code::CodeBrain;
use grapheme_core::DagNN;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// ============================================================================
// HumanEval Problem Format
// ============================================================================

/// HumanEval problem from JSONL
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HumanEvalProblem {
    /// Task identifier (e.g., "HumanEval/0")
    pub task_id: String,
    /// Function signature and docstring
    pub prompt: String,
    /// Reference solution
    pub canonical_solution: String,
    /// Test cases (Python code)
    pub test: String,
    /// Function name
    pub entry_point: String,
    /// Optional description
    #[serde(default)]
    pub description: String,
}

impl HumanEvalProblem {
    /// Get the full function code (prompt + solution)
    pub fn full_code(&self) -> String {
        format!("{}{}", self.prompt, self.canonical_solution)
    }

    /// Compute a complexity level based on solution length and structure
    pub fn complexity_level(&self) -> u8 {
        let line_count = self.canonical_solution.lines().count();
        let char_count = self.canonical_solution.len();

        // Simple heuristic based on solution size
        match (line_count, char_count) {
            (0..=5, 0..=100) => 1,   // Very simple
            (0..=10, 0..=250) => 2,  // Simple
            (0..=20, 0..=500) => 3,  // Medium
            (0..=40, 0..=1000) => 4, // Complex
            _ => 5,                   // Very complex
        }
    }
}

// ============================================================================
// HumanEval Encoder
// ============================================================================

/// Encoder for HumanEval problems to graph pairs
pub struct HumanEvalEncoder {
    code_brain: CodeBrain,
}

impl HumanEvalEncoder {
    /// Create a new HumanEval encoder
    pub fn new() -> Self {
        Self {
            code_brain: CodeBrain::new(),
        }
    }

    /// Load HumanEval problems from JSONL file
    pub fn load_problems<P: AsRef<Path>>(&self, path: P) -> GraphDataResult<Vec<HumanEvalProblem>> {
        let file = File::open(path.as_ref()).map_err(|e| {
            GraphDataError::Io(e)
        })?;
        let reader = BufReader::new(file);
        let mut problems = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(GraphDataError::Io)?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str(&line) {
                Ok(problem) => problems.push(problem),
                Err(e) => {
                    return Err(GraphDataError::Serialization(
                        format!("Line {}: {}", line_num + 1, e)
                    ));
                }
            }
        }

        Ok(problems)
    }

    /// Encode a single problem to a graph pair
    pub fn encode_problem(&self, problem: &HumanEvalProblem) -> GraphDataResult<GraphPair> {
        // Parse prompt to graph (input)
        let input_graph = self.parse_to_dagnn(&problem.prompt)?;

        // Parse full code to graph (output = prompt + solution)
        let output_graph = self.parse_to_dagnn(&problem.full_code())?;

        // Build the graph pair
        let pair = GraphPairBuilder::new(&problem.task_id)
            .input(input_graph)
            .output(output_graph)
            .level(problem.complexity_level())
            .domain("humaneval")
            .meta("entry_point", &problem.entry_point)
            .meta("prompt_len", problem.prompt.len().to_string())
            .meta("solution_len", problem.canonical_solution.len().to_string())
            .build();

        Ok(pair)
    }

    /// Encode all problems to a dataset
    pub fn encode_dataset<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> GraphDataResult<HumanEvalEncodingResult> {
        let problems = self.load_problems(&path)?;
        let mut dataset = GraphDataset::new("humaneval");
        let mut successes = 0;
        let mut failures = Vec::new();

        for problem in &problems {
            match self.encode_problem(problem) {
                Ok(pair) => {
                    dataset.add(pair);
                    successes += 1;
                }
                Err(e) => {
                    failures.push(EncodingFailure {
                        task_id: problem.task_id.clone(),
                        error: e.to_string(),
                    });
                }
            }
        }

        // Update metadata
        dataset.metadata.domain = "humaneval".to_string();
        dataset.metadata.source = path.as_ref().display().to_string();
        dataset.metadata.properties.insert(
            "total_problems".to_string(),
            problems.len().to_string(),
        );
        dataset.metadata.properties.insert(
            "encoded_successfully".to_string(),
            successes.to_string(),
        );

        Ok(HumanEvalEncodingResult {
            dataset,
            total: problems.len(),
            successes,
            failures,
        })
    }

    /// Parse code string to DagNN graph
    fn parse_to_dagnn(&self, code: &str) -> GraphDataResult<DagNN> {
        // Use CodeBrain to parse code
        let code_graph = self.code_brain.parse_code(code).map_err(|e| {
            GraphDataError::InvalidData(format!("Failed to parse code: {}", e))
        })?;

        // Convert CodeGraph to DagNN by walking the structure
        let dag = code_graph_to_dagnn(&code_graph);
        Ok(dag)
    }
}

impl Default for HumanEvalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of encoding HumanEval problems
#[derive(Debug)]
pub struct HumanEvalEncodingResult {
    /// Encoded dataset
    pub dataset: GraphDataset,
    /// Total number of problems
    pub total: usize,
    /// Number of successfully encoded problems
    pub successes: usize,
    /// Failed encodings
    pub failures: Vec<EncodingFailure>,
}

impl HumanEvalEncodingResult {
    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.successes as f64 / self.total as f64) * 100.0
        }
    }
}

/// Details about a failed encoding
#[derive(Debug, Clone)]
pub struct EncodingFailure {
    /// Task ID that failed
    pub task_id: String,
    /// Error message
    pub error: String,
}

// ============================================================================
// CodeGraph to DagNN Conversion
// ============================================================================

use grapheme_code::CodeGraph;
use grapheme_core::{Node, Edge};

/// Convert a CodeGraph to a DagNN for training
fn code_graph_to_dagnn(code_graph: &CodeGraph) -> DagNN {
    let mut dag = DagNN::new();

    // Map CodeGraph node indices to DagNN node indices
    let mut node_map = std::collections::HashMap::new();

    // Add all nodes
    for node_idx in code_graph.graph.node_indices() {
        let code_node = &code_graph.graph[node_idx];

        // Create a hidden node with activation based on node type
        let mut node = Node::hidden();
        // Set activation to encode node type information
        node.activation = code_node_activation(code_node);

        let dag_idx = dag.graph.add_node(node);
        node_map.insert(node_idx, dag_idx);
    }

    // Add all edges
    for edge_idx in code_graph.graph.edge_indices() {
        if let Some((source, target)) = code_graph.graph.edge_endpoints(edge_idx) {
            if let (Some(&dag_source), Some(&dag_target)) = (node_map.get(&source), node_map.get(&target)) {
                dag.graph.add_edge(dag_source, dag_target, Edge::sequential());
            }
        }
    }

    // Set root as input if available
    if let Some(root_idx) = code_graph.root {
        if let Some(&dag_root) = node_map.get(&root_idx) {
            // Mark root as a special node
            if let Some(node) = dag.graph.node_weight_mut(dag_root) {
                node.activation = 1.0; // Mark as root
            }
        }
    }

    dag
}

use grapheme_code::CodeNode;

/// Compute activation value for a code node (encodes node type)
fn code_node_activation(node: &CodeNode) -> f32 {
    // Use different activation values for different node types
    // This provides a simple encoding of node semantics
    match node {
        CodeNode::Module { .. } => 0.95,
        CodeNode::Function { .. } => 0.9,
        CodeNode::Variable { .. } => 0.7,
        CodeNode::Return => 0.75,
        CodeNode::If => 0.6,
        CodeNode::Loop { .. } => 0.65,
        CodeNode::BinaryOp(_) => 0.5,
        CodeNode::UnaryOp(_) => 0.48,
        CodeNode::Literal(_) => 0.3,
        CodeNode::Identifier(_) => 0.4,
        CodeNode::Call { .. } => 0.8,
        CodeNode::Assignment => 0.55,
        CodeNode::Block => 0.2,
        CodeNode::ExprStmt => 0.35,
        CodeNode::Type(_) => 0.25,
        CodeNode::Comment(_) => 0.05,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use grapheme_core::DomainBrain;
    use tempfile::NamedTempFile;
    use std::io::Write;

    fn create_test_problem() -> HumanEvalProblem {
        HumanEvalProblem {
            task_id: "HumanEval/0".to_string(),
            prompt: "def has_close_elements(numbers, threshold):\n    \"\"\"Check if any two numbers are close.\"\"\"\n".to_string(),
            canonical_solution: "    for i, n1 in enumerate(numbers):\n        for n2 in numbers[i+1:]:\n            if abs(n1 - n2) < threshold:\n                return True\n    return False\n".to_string(),
            test: "assert has_close_elements([1, 2, 3], 0.5) == False".to_string(),
            entry_point: "has_close_elements".to_string(),
            description: "Test problem".to_string(),
        }
    }

    #[test]
    fn test_humaneval_problem_full_code() {
        let problem = create_test_problem();
        let full = problem.full_code();
        assert!(full.contains("def has_close_elements"));
        assert!(full.contains("return False"));
    }

    #[test]
    fn test_humaneval_problem_complexity() {
        let simple = HumanEvalProblem {
            task_id: "test".to_string(),
            prompt: "def f():".to_string(),
            canonical_solution: "  return 1".to_string(),
            test: "".to_string(),
            entry_point: "f".to_string(),
            description: "".to_string(),
        };
        assert!(simple.complexity_level() <= 2);

        let complex = HumanEvalProblem {
            task_id: "test".to_string(),
            prompt: "def f():".to_string(),
            canonical_solution: "x\n".repeat(50),
            test: "".to_string(),
            entry_point: "f".to_string(),
            description: "".to_string(),
        };
        assert!(complex.complexity_level() >= 4);
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = HumanEvalEncoder::new();
        assert!(encoder.code_brain.domain_name() == "Source Code");
    }

    #[test]
    fn test_load_problems_from_jsonl() {
        let mut file = NamedTempFile::new().unwrap();
        let problem = create_test_problem();
        let json = serde_json::to_string(&problem).unwrap();
        writeln!(file, "{}", json).unwrap();

        let encoder = HumanEvalEncoder::new();
        let problems = encoder.load_problems(file.path()).unwrap();
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].task_id, "HumanEval/0");
    }

    #[test]
    fn test_encode_problem() {
        let encoder = HumanEvalEncoder::new();
        let problem = create_test_problem();

        let pair = encoder.encode_problem(&problem).unwrap();
        assert_eq!(pair.id, "HumanEval/0");
        assert_eq!(pair.domain, "humaneval");
        assert!(pair.input.node_count() > 0);
        assert!(pair.output.node_count() > 0);
        // Output should have more nodes (includes solution)
        assert!(pair.output.node_count() >= pair.input.node_count());
    }

    #[test]
    fn test_encode_dataset() {
        let mut file = NamedTempFile::new().unwrap();

        // Write multiple problems
        for i in 0..3 {
            let problem = HumanEvalProblem {
                task_id: format!("HumanEval/{}", i),
                prompt: format!("def f{}():\n", i),
                canonical_solution: "  return 1\n".to_string(),
                test: "".to_string(),
                entry_point: format!("f{}", i),
                description: "".to_string(),
            };
            let json = serde_json::to_string(&problem).unwrap();
            writeln!(file, "{}", json).unwrap();
        }

        let encoder = HumanEvalEncoder::new();
        let result = encoder.encode_dataset(file.path()).unwrap();

        assert_eq!(result.total, 3);
        assert_eq!(result.successes, 3);
        assert!(result.failures.is_empty());
        assert_eq!(result.dataset.len(), 3);
    }

    #[test]
    fn test_encoding_result_success_rate() {
        let result = HumanEvalEncodingResult {
            dataset: GraphDataset::new("test"),
            total: 100,
            successes: 95,
            failures: vec![],
        };
        assert!((result.success_rate() - 95.0).abs() < 0.001);
    }

    #[test]
    fn test_code_node_activation() {
        let func_node = CodeNode::Function {
            name: "test".to_string(),
            params: vec![],
            return_type: None,
        };
        assert!(code_node_activation(&func_node) > 0.8);

        let lit_node = CodeNode::Literal(grapheme_code::LiteralValue::Integer(42));
        assert!(code_node_activation(&lit_node) < 0.5);
    }

    #[test]
    fn test_code_graph_to_dagnn() {
        let code = "def f(): return 1";
        let brain = CodeBrain::new();
        let code_graph = brain.parse_code(code).unwrap();
        let dag = code_graph_to_dagnn(&code_graph);

        // Should have nodes for function, return, literal
        assert!(dag.node_count() > 0);
    }
}
