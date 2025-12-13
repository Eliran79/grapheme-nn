//! Graph-LLM Translation Module
//!
//! Integration-002: LLM response to DagNN graph translation
//! Integration-003: DagNN graph to LLM prompt translation
//!
//! Provides bidirectional conversion between GRAPHEME graphs and LLM interactions.

use grapheme_core::DagNN;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Node representation for LLM communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMNode {
    pub id: usize,
    pub label: String,
    pub node_type: String,
    pub value: f32,
}

/// Edge representation for LLM communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMEdge {
    pub source: usize,
    pub target: usize,
    pub weight: f32,
    pub edge_type: String,
}

/// Graph representation for LLM communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMGraph {
    pub nodes: Vec<LLMNode>,
    pub edges: Vec<LLMEdge>,
    pub metadata: HashMap<String, String>,
}

/// Graph-to-LLM prompt translator (Integration-003)
pub struct GraphToPrompt {
    /// Include node values in output
    pub include_values: bool,
    /// Include edge weights
    pub include_weights: bool,
    /// Maximum nodes to include
    pub max_nodes: usize,
    /// Format style
    pub format: PromptFormat,
}

/// Prompt format options
#[derive(Debug, Clone, Copy)]
pub enum PromptFormat {
    /// Plain text description
    Text,
    /// JSON format
    Json,
    /// Mermaid graph notation
    Mermaid,
    /// DOT graph format
    Dot,
}

impl Default for GraphToPrompt {
    fn default() -> Self {
        Self {
            include_values: true,
            include_weights: false,
            max_nodes: 100,
            format: PromptFormat::Text,
        }
    }
}

impl GraphToPrompt {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_format(mut self, format: PromptFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_max_nodes(mut self, max: usize) -> Self {
        self.max_nodes = max;
        self
    }

    /// Convert DagNN to LLM-friendly prompt
    pub fn translate(&self, dag: &DagNN) -> String {
        match self.format {
            PromptFormat::Text => self.to_text(dag),
            PromptFormat::Json => self.to_json(dag),
            PromptFormat::Mermaid => self.to_mermaid(dag),
            PromptFormat::Dot => self.to_dot(dag),
        }
    }

    /// Convert to plain text description
    fn to_text(&self, dag: &DagNN) -> String {
        let mut output = String::new();
        output.push_str("Graph Structure:\n");

        // Get nodes
        let nodes = dag.input_nodes();
        let node_count = nodes.len().min(self.max_nodes);

        output.push_str(&format!("- {} nodes\n", node_count));

        // Describe input nodes
        output.push_str("\nInput nodes:\n");
        for (i, _node) in nodes.iter().take(self.max_nodes).enumerate() {
            output.push_str(&format!("  [{}] node\n", i));
        }

        output
    }

    /// Convert to JSON format
    fn to_json(&self, dag: &DagNN) -> String {
        let llm_graph = self.to_llm_graph(dag);
        serde_json::to_string_pretty(&llm_graph).unwrap_or_else(|_| "{}".to_string())
    }

    /// Convert to Mermaid diagram notation
    fn to_mermaid(&self, dag: &DagNN) -> String {
        let mut output = String::from("graph TD\n");

        let nodes = dag.input_nodes();
        for (i, _node) in nodes.iter().take(self.max_nodes).enumerate() {
            output.push_str(&format!("    N{}[\"node_{}\"]\n", i, i));
        }

        output
    }

    /// Convert to DOT graph format
    fn to_dot(&self, dag: &DagNN) -> String {
        let mut output = String::from("digraph G {\n");

        let nodes = dag.input_nodes();
        for (i, _node) in nodes.iter().take(self.max_nodes).enumerate() {
            output.push_str(&format!("    n{} [label=\"node_{}\"];\n", i, i));
        }

        output.push_str("}\n");
        output
    }

    /// Convert to LLMGraph structure
    pub fn to_llm_graph(&self, dag: &DagNN) -> LLMGraph {
        let mut nodes = Vec::new();
        let edges = Vec::new();

        let input_nodes = dag.input_nodes();
        for (i, _node) in input_nodes.iter().take(self.max_nodes).enumerate() {
            nodes.push(LLMNode {
                id: i,
                label: format!("node_{}", i),
                node_type: "Input".to_string(),
                value: 0.0,
            });
        }

        let mut metadata = HashMap::new();
        metadata.insert("node_count".to_string(), nodes.len().to_string());
        metadata.insert("edge_count".to_string(), edges.len().to_string());

        LLMGraph { nodes, edges, metadata }
    }
}

/// LLM response to graph translator (Integration-002)
pub struct PromptToGraph {
    /// Whether to create edges between sequential nodes
    pub create_sequential_edges: bool,
}

impl Default for PromptToGraph {
    fn default() -> Self {
        Self {
            create_sequential_edges: true,
        }
    }
}

impl PromptToGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse LLM response into graph modifications
    pub fn translate(&self, response: &str) -> Result<GraphModification, TranslationError> {
        // Try JSON parse first
        if let Ok(json_mod) = self.parse_json(response) {
            return Ok(json_mod);
        }

        // Try structured text format
        if let Ok(text_mod) = self.parse_text(response) {
            return Ok(text_mod);
        }

        // Fall back to simple text extraction
        Ok(self.extract_simple(response))
    }

    /// Parse JSON-formatted graph modification
    fn parse_json(&self, response: &str) -> Result<GraphModification, TranslationError> {
        // Find JSON block in response
        let json_start = response.find('{')
            .ok_or(TranslationError::NoJsonFound)?;
        let json_end = response.rfind('}')
            .ok_or(TranslationError::NoJsonFound)?;

        if json_end <= json_start {
            return Err(TranslationError::InvalidJson);
        }

        let json_str = &response[json_start..=json_end];

        // Try parsing as LLMGraph
        if let Ok(graph) = serde_json::from_str::<LLMGraph>(json_str) {
            return Ok(GraphModification::ReplaceGraph(graph));
        }

        // Try parsing as modification command
        if let Ok(cmd) = serde_json::from_str::<ModificationCommand>(json_str) {
            return Ok(GraphModification::Command(cmd));
        }

        Err(TranslationError::InvalidJson)
    }

    /// Parse structured text format
    fn parse_text(&self, response: &str) -> Result<GraphModification, TranslationError> {
        let mut add_nodes: Vec<String> = Vec::new();
        let mut add_edges: Vec<(String, String)> = Vec::new();

        for line in response.lines() {
            let line = line.trim().to_lowercase();

            // Parse "add node: label" format
            if line.starts_with("add node:") {
                let label = line.strip_prefix("add node:").unwrap().trim();
                add_nodes.push(label.to_string());
            }
            // Parse "add edge: a -> b" format
            else if line.starts_with("add edge:") {
                let edge_part = line.strip_prefix("add edge:").unwrap().trim();
                if let Some((src, tgt)) = edge_part.split_once("->") {
                    add_edges.push((src.trim().to_string(), tgt.trim().to_string()));
                }
            }
        }

        if add_nodes.is_empty() && add_edges.is_empty() {
            return Err(TranslationError::NoModificationsFound);
        }

        Ok(GraphModification::Incremental {
            add_nodes,
            add_edges,
            remove_nodes: Vec::new(),
        })
    }

    /// Extract simple text content as nodes
    fn extract_simple(&self, response: &str) -> GraphModification {
        // Split response into meaningful chunks
        let words: Vec<String> = response
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .take(50)
            .map(|s| s.to_string())
            .collect();

        GraphModification::Incremental {
            add_nodes: words,
            add_edges: Vec::new(),
            remove_nodes: Vec::new(),
        }
    }

    /// Apply modification to a DagNN - creates a new graph from the modification
    pub fn apply_to_dag(&self, _dag: &mut DagNN, modification: &GraphModification) -> Result<DagNN, TranslationError> {
        match modification {
            GraphModification::ReplaceGraph(llm_graph) => {
                // Create new graph from node labels
                let text = llm_graph.nodes.iter()
                    .map(|n| n.label.chars().next().unwrap_or('?'))
                    .collect::<String>();
                DagNN::from_text(&text).map_err(|e| TranslationError::GraphError(e.to_string()))
            }
            GraphModification::Command(cmd) => {
                match cmd.action.as_str() {
                    "add_node" => {
                        if let Some(label) = &cmd.label {
                            let ch = label.chars().next().unwrap_or('?');
                            DagNN::from_text(&ch.to_string())
                                .map_err(|e| TranslationError::GraphError(e.to_string()))
                        } else {
                            DagNN::from_text("?").map_err(|e| TranslationError::GraphError(e.to_string()))
                        }
                    }
                    _ => DagNN::from_text("").map_err(|e| TranslationError::GraphError(e.to_string()))
                }
            }
            GraphModification::Incremental { add_nodes, add_edges: _, remove_nodes: _ } => {
                // Create graph from labels (first char of each)
                let text = add_nodes.iter()
                    .map(|l| l.chars().next().unwrap_or('?'))
                    .collect::<String>();
                DagNN::from_text(&text).map_err(|e| TranslationError::GraphError(e.to_string()))
            }
        }
    }
}

/// Graph modification types
#[derive(Debug, Clone)]
pub enum GraphModification {
    /// Replace entire graph
    ReplaceGraph(LLMGraph),
    /// Single modification command
    Command(ModificationCommand),
    /// Incremental changes
    Incremental {
        add_nodes: Vec<String>,
        add_edges: Vec<(String, String)>,
        remove_nodes: Vec<String>,
    },
}

/// Single modification command from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationCommand {
    pub action: String,
    pub node_id: Option<usize>,
    pub label: Option<String>,
    pub value: Option<f32>,
    pub source: Option<usize>,
    pub target: Option<usize>,
}

/// Translation errors
#[derive(Debug, Clone)]
pub enum TranslationError {
    NoJsonFound,
    InvalidJson,
    NoModificationsFound,
    InvalidNodeReference,
    GraphError(String),
}

impl std::fmt::Display for TranslationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoJsonFound => write!(f, "No JSON found in response"),
            Self::InvalidJson => write!(f, "Invalid JSON format"),
            Self::NoModificationsFound => write!(f, "No modifications found"),
            Self::InvalidNodeReference => write!(f, "Invalid node reference"),
            Self::GraphError(e) => write!(f, "Graph error: {}", e),
        }
    }
}

impl std::error::Error for TranslationError {}

/// Prompt templates for graph-related LLM queries
pub struct PromptTemplates;

impl PromptTemplates {
    /// Generate a prompt for graph analysis
    pub fn analyze_graph(graph_desc: &str) -> String {
        format!(
            "Analyze the following graph structure and describe its key properties:\n\n{}\n\n\
            Provide analysis of:\n\
            1. Graph topology (linear, tree, DAG, cyclic)\n\
            2. Key nodes and their roles\n\
            3. Information flow patterns\n\
            4. Potential optimizations",
            graph_desc
        )
    }

    /// Generate a prompt for graph modification suggestions
    pub fn suggest_modifications(graph_desc: &str, goal: &str) -> String {
        format!(
            "Given this graph structure:\n\n{}\n\n\
            Goal: {}\n\n\
            Suggest modifications to achieve the goal. Format your response as:\n\
            ADD NODE: <label>\n\
            ADD EDGE: <source> -> <target>\n\
            REMOVE NODE: <label>",
            graph_desc, goal
        )
    }

    /// Generate a prompt for knowledge extraction
    pub fn extract_knowledge(text: &str) -> String {
        format!(
            "Extract entities and relationships from the following text. \
            Format as JSON with 'nodes' and 'edges' arrays:\n\n{}\n\n\
            Output format:\n\
            {{\n  \"nodes\": [{{\"id\": 0, \"label\": \"entity1\", \"node_type\": \"concept\"}}],\n  \
            \"edges\": [{{\"source\": 0, \"target\": 1, \"weight\": 1.0, \"edge_type\": \"relates_to\"}}]\n}}",
            text
        )
    }

    /// Generate a prompt for graph-to-text translation
    pub fn graph_to_natural_language(graph_desc: &str) -> String {
        format!(
            "Convert the following graph structure into a natural language description:\n\n{}\n\n\
            Write a clear, concise explanation of what this graph represents.",
            graph_desc
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_to_prompt_text() {
        let dag = DagNN::from_text("test").unwrap();
        let translator = GraphToPrompt::new();
        let prompt = translator.translate(&dag);
        assert!(prompt.contains("Graph Structure"));
    }

    #[test]
    fn test_graph_to_prompt_json() {
        let dag = DagNN::from_text("ab").unwrap();
        let translator = GraphToPrompt::new().with_format(PromptFormat::Json);
        let json = translator.translate(&dag);
        assert!(json.contains("nodes"));
    }

    #[test]
    fn test_prompt_to_graph_json() {
        let translator = PromptToGraph::new();
        let response = r#"Here's the modification: {"action": "add_node", "label": "new_concept"}"#;
        let result = translator.translate(response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prompt_to_graph_text() {
        let translator = PromptToGraph::new();
        let response = "ADD NODE: concept1\nADD NODE: concept2\nADD EDGE: concept1 -> concept2";
        let result = translator.translate(response);
        assert!(result.is_ok());
        if let Ok(GraphModification::Incremental { add_nodes, add_edges, .. }) = result {
            assert_eq!(add_nodes.len(), 2);
            assert_eq!(add_edges.len(), 1);
        }
    }

    #[test]
    fn test_prompt_templates() {
        let analyze = PromptTemplates::analyze_graph("Test graph");
        assert!(analyze.contains("Analyze"));

        let suggest = PromptTemplates::suggest_modifications("Graph", "Goal");
        assert!(suggest.contains("modifications"));

        let extract = PromptTemplates::extract_knowledge("Some text");
        assert!(extract.contains("entities"));
    }
}
