//! # grapheme-core
//!
//! Layer 4: The universal interface - character-level natural language processing.
//!
//! This crate provides:
//! - Pure GRAPHEME processing (character-level, vocabulary-free)
//! - Natural language understanding
//! - Support for any human language input
//! - Intent and parameter extraction
//!
//! Key innovations:
//! - No tokenization or vocabulary
//! - Direct character-to-node mapping
//! - Dynamic graph growth with input
//! - Universal Unicode support

use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors in core GRAPHEME processing
#[derive(Error, Debug)]
pub enum GraphemeError {
    #[error("Invalid character: {0}")]
    InvalidCharacter(char),
    #[error("Graph construction error: {0}")]
    GraphError(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Result type for GRAPHEME operations
pub type GraphemeResult<T> = Result<T, GraphemeError>;

/// Type of node in the GRAPHEME graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    /// Input character node
    Input(char),
    /// Hidden processing node
    Hidden,
    /// Output node
    Output,
    /// Clique (compressed concept)
    Clique(Vec<usize>),
    /// Pattern node (compressed repeated sequence)
    Pattern(Vec<u8>),
    /// Compressed region
    Compressed,
}

/// A node in the GRAPHEME graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// The raw character value (if applicable)
    pub value: Option<u8>,
    /// Current activation level
    pub activation: f32,
    /// Type of this node
    pub node_type: NodeType,
    /// Position in original text (if input node)
    pub position: Option<usize>,
}

impl Node {
    /// Create a new input node from a character
    pub fn input(ch: char, position: usize) -> Self {
        Self {
            value: if ch.is_ascii() {
                Some(ch as u8)
            } else {
                None
            },
            activation: 1.0,
            node_type: NodeType::Input(ch),
            position: Some(position),
        }
    }

    /// Create a new hidden node
    pub fn hidden() -> Self {
        Self {
            value: None,
            activation: 0.0,
            node_type: NodeType::Hidden,
            position: None,
        }
    }
}

/// Type of edge in the GRAPHEME graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Sequential character connection
    Sequential,
    /// Semantic relationship
    Semantic,
    /// Structural/syntactic connection
    Structural,
    /// Within-clique connection
    Clique,
    /// Long-range dependency (skip connection)
    Skip,
}

/// An edge in the GRAPHEME graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Edge weight (learnable)
    pub weight: f32,
    /// Type of edge
    pub edge_type: EdgeType,
}

impl Edge {
    /// Create a new edge
    pub fn new(weight: f32, edge_type: EdgeType) -> Self {
        Self { weight, edge_type }
    }

    /// Create a sequential edge
    pub fn sequential() -> Self {
        Self::new(1.0, EdgeType::Sequential)
    }
}

/// The main GRAPHEME graph structure
#[derive(Debug)]
pub struct GraphemeGraph {
    /// The underlying directed acyclic graph
    pub graph: DiGraph<Node, Edge>,
    /// Input nodes in order
    pub input_nodes: Vec<NodeIndex>,
    /// Detected cliques
    pub cliques: Vec<Vec<NodeIndex>>,
}

impl Default for GraphemeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphemeGraph {
    /// Create a new empty GRAPHEME graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            input_nodes: Vec::new(),
            cliques: Vec::new(),
        }
    }

    /// Build a graph from text (character by character, NO tokenization)
    pub fn from_text(text: &str) -> Self {
        let mut graph = Self::new();

        let mut prev_node: Option<NodeIndex> = None;

        for (position, ch) in text.chars().enumerate() {
            let node = graph.graph.add_node(Node::input(ch, position));
            graph.input_nodes.push(node);

            // Connect to previous character
            if let Some(prev) = prev_node {
                graph.graph.add_edge(prev, node, Edge::sequential());
            }

            prev_node = Some(node);
        }

        graph
    }

    /// Add a character to the graph
    pub fn add_character(&mut self, ch: char, position: usize) -> NodeIndex {
        let node = self.graph.add_node(Node::input(ch, position));
        self.input_nodes.push(node);

        // Connect to previous if exists
        if self.input_nodes.len() > 1 {
            let prev = self.input_nodes[self.input_nodes.len() - 2];
            self.graph.add_edge(prev, node, Edge::sequential());
        }

        node
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Convert graph back to text
    pub fn to_text(&self) -> String {
        self.input_nodes
            .iter()
            .filter_map(|&idx| {
                if let NodeType::Input(ch) = self.graph[idx].node_type {
                    Some(ch)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute processing depth based on character complexity
    /// Simple characters → shallow processing
    /// Complex characters → deeper processing
    pub fn compute_processing_depth(ch: char, _context: &[char]) -> usize {
        match ch {
            // Common ASCII letters - shallow
            'a'..='z' | 'A'..='Z' => 2,
            // Digits
            '0'..='9' => 2,
            // Common punctuation
            ' ' | '.' | ',' | '!' | '?' => 1,
            // Mathematical symbols
            '+' | '-' | '*' | '/' | '=' | '<' | '>' => 3,
            // Complex Unicode (non-ASCII)
            _ if !ch.is_ascii() => {
                // More complex characters need deeper processing
                if ch.len_utf8() > 2 {
                    5
                } else {
                    4
                }
            }
            // Default
            _ => 2,
        }
    }
}

/// Text processor trait for GRAPHEME
pub trait TextProcessor {
    /// Convert text to graph (CHARACTER BY CHARACTER, NO TOKENIZATION)
    fn text_to_graph(&mut self, text: &str) -> GraphemeGraph;

    /// Convert graph back to text
    fn graph_to_text(&self, graph: &GraphemeGraph) -> String;

    /// Process any Unicode text without configuration
    fn process_universal(&mut self, text: &str) -> GraphemeResult<GraphemeGraph>;
}

/// Basic implementation of TextProcessor
#[derive(Debug, Default)]
pub struct BasicTextProcessor;

impl BasicTextProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl TextProcessor for BasicTextProcessor {
    fn text_to_graph(&mut self, text: &str) -> GraphemeGraph {
        GraphemeGraph::from_text(text)
    }

    fn graph_to_text(&self, graph: &GraphemeGraph) -> String {
        graph.to_text()
    }

    fn process_universal(&mut self, text: &str) -> GraphemeResult<GraphemeGraph> {
        // Works with any Unicode: "Hello", "你好", "مرحبا", "🚀", "∫dx"
        Ok(self.text_to_graph(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_graph() {
        let graph = GraphemeGraph::from_text("Hello");
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4); // 4 sequential edges
    }

    #[test]
    fn test_roundtrip() {
        let original = "Hello, World!";
        let graph = GraphemeGraph::from_text(original);
        let roundtrip = graph.to_text();
        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_unicode() {
        // Test universal language support
        let texts = vec![
            "Hello",
            "你好",
            "مرحبا",
            "🚀🎉",
            "∫x²dx",
            "Hello你好🚀",
        ];

        let mut processor = BasicTextProcessor::new();

        for text in texts {
            let graph = processor.process_universal(text).unwrap();
            let roundtrip = graph.to_text();
            assert_eq!(text, roundtrip, "Failed for: {}", text);
        }
    }

    #[test]
    fn test_processing_depth() {
        assert_eq!(GraphemeGraph::compute_processing_depth('a', &[]), 2);
        assert_eq!(GraphemeGraph::compute_processing_depth('+', &[]), 3);
        // '你' has 3-byte UTF-8 encoding, so depth = 5
        assert_eq!(GraphemeGraph::compute_processing_depth('你', &[]), 5);
    }
}
