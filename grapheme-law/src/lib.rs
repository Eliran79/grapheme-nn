//! # grapheme-law
//!
//! Law Brain: Legal reasoning and case law analysis for GRAPHEME.
//!
//! This crate provides:
//! - Legal document node types (Citation, Statute, Precedent, Argument)
//! - Case law graph construction
//! - Legal reasoning and precedent analysis
//! - Argument structure representation

use grapheme_core::{
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule,
    ExecutionResult, ValidationIssue, ValidationSeverity,
};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors in legal graph processing
#[derive(Error, Debug)]
pub enum LawGraphError {
    #[error("Invalid citation format: {0}")]
    InvalidCitation(String),
    #[error("Missing precedent: {0}")]
    MissingPrecedent(String),
    #[error("Argument structure error: {0}")]
    ArgumentError(String),
}

/// Result type for law graph operations
pub type LawGraphResult<T> = Result<T, LawGraphError>;

/// Legal jurisdiction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Jurisdiction {
    /// US Federal law
    USFederal,
    /// US State law (generic)
    USState,
    /// UK law
    UK,
    /// EU law
    EU,
    /// Generic/unknown jurisdiction
    #[default]
    Generic,
}

/// Legal document types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LegalNode {
    /// Case citation
    Citation {
        case_name: String,
        year: Option<u16>,
        volume: Option<String>,
        page: Option<u32>,
    },
    /// Statute reference
    Statute {
        title: String,
        section: String,
        subsection: Option<String>,
    },
    /// Legal principle or holding
    Holding(String),
    /// Legal argument
    Argument {
        premise: String,
        conclusion: String,
    },
    /// Party in a case
    Party { name: String, role: PartyRole },
    /// Legal issue or question
    Issue(String),
    /// Factual finding
    Fact(String),
    /// Legal rule or test
    Rule(String),
    /// Application of rule to facts
    Application(String),
    /// Conclusion or outcome
    Conclusion(String),
    /// Concurrence or dissent
    Opinion { author: String, kind: OpinionKind },
}

/// Role of a party in legal proceedings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartyRole {
    Plaintiff,
    Defendant,
    Appellant,
    Appellee,
    Petitioner,
    Respondent,
}

/// Type of judicial opinion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpinionKind {
    Majority,
    Concurrence,
    Dissent,
}

/// Edge types in legal graphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegalEdge {
    /// Cites another case/statute
    Cites,
    /// Follows precedent
    Follows,
    /// Distinguishes from precedent
    Distinguishes,
    /// Overrules precedent
    Overrules,
    /// Supports an argument
    Supports,
    /// Contradicts an argument
    Contradicts,
    /// Applies rule to facts
    AppliesTo,
    /// Leads to conclusion
    LeadsTo,
    /// Party relationship
    PartyOf,
}

/// A legal document represented as a graph
#[derive(Debug)]
pub struct LegalGraph {
    /// The underlying directed graph
    pub graph: DiGraph<LegalNode, LegalEdge>,
    /// Root node (main holding or issue)
    pub root: Option<NodeIndex>,
    /// Jurisdiction
    pub jurisdiction: Jurisdiction,
}

impl Default for LegalGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl LegalGraph {
    /// Create a new empty legal graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            root: None,
            jurisdiction: Jurisdiction::Generic,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: LegalNode) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: LegalEdge) {
        self.graph.add_edge(from, to, edge);
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Parse a simple legal citation
    pub fn parse_citation(citation: &str) -> LawGraphResult<Self> {
        let mut graph = Self::new();

        // Simple parsing for format like "Brown v. Board of Education, 347 U.S. 483 (1954)"
        let trimmed = citation.trim();

        // Check for year in parentheses
        let year = if let Some(start) = trimmed.rfind('(') {
            if let Some(end) = trimmed.rfind(')') {
                trimmed[start + 1..end].parse::<u16>().ok()
            } else {
                None
            }
        } else {
            None
        };

        let node = graph.add_node(LegalNode::Citation {
            case_name: trimmed.to_string(),
            year,
            volume: None,
            page: None,
        });
        graph.root = Some(node);

        Ok(graph)
    }
}

// ============================================================================
// Law Brain
// ============================================================================

/// The Law Brain for legal reasoning
pub struct LawBrain {
    /// Supported jurisdictions
    jurisdictions: Vec<Jurisdiction>,
}

impl Default for LawBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LawBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LawBrain")
            .field("domain", &"law")
            .field("jurisdictions", &self.jurisdictions)
            .finish()
    }
}

impl LawBrain {
    /// Create a new law brain
    pub fn new() -> Self {
        Self {
            jurisdictions: vec![
                Jurisdiction::USFederal,
                Jurisdiction::USState,
                Jurisdiction::UK,
                Jurisdiction::EU,
            ],
        }
    }

    /// Check if text looks like legal content
    fn looks_like_legal(&self, input: &str) -> bool {
        let legal_patterns = [
            " v. ", " vs. ", "plaintiff", "defendant",
            "appellant", "appellee", "statute", "§",
            "U.S.C.", "U.S. ", "F.2d", "F.3d", "S.Ct.",
            "court", "judge", "ruling", "precedent",
            "holding", "dissent", "concur", "jurisdiction",
            "liable", "damages", "tort", "contract",
            "constitutional", "amendment", "rights",
        ];
        let lower = input.to_lowercase();
        legal_patterns.iter().any(|p| lower.contains(&p.to_lowercase()))
    }

    /// Validate a legal graph
    pub fn validate_legal(&self, graph: &LegalGraph) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        if graph.node_count() == 0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Empty legal graph".to_string(),
                location: None,
            });
        }

        // Check for orphan citations (citations with no connections)
        for node_idx in graph.graph.node_indices() {
            if let LegalNode::Citation { case_name, .. } = &graph.graph[node_idx] {
                let incoming = graph.graph.edges_directed(node_idx, petgraph::Direction::Incoming).count();
                let outgoing = graph.graph.edges_directed(node_idx, petgraph::Direction::Outgoing).count();
                if incoming == 0 && outgoing == 0 && graph.node_count() > 1 {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Info,
                        message: format!("Orphan citation: {}", case_name),
                        location: Some(node_idx.index()),
                    });
                }
            }
        }

        issues
    }

    // ========================================================================
    // Graph Transform Helper Methods
    // ========================================================================

    /// Stare Decisis: Strengthen connections to nodes with high activation.
    /// In legal reasoning, precedent (high-activation nodes) should have stronger influence.
    fn stare_decisis(&self, graph: &DagNN) -> DomainResult<DagNN> {
        use petgraph::visit::EdgeRef;
        let mut result = graph.clone();

        // Find high-activation nodes (established precedents)
        let precedent_nodes = result.get_nodes_by_activation(0.6);

        // Strengthen edges connected to precedent nodes
        let edges_to_strengthen: Vec<_> = result.graph.edge_references()
            .filter_map(|edge| {
                let boost = if precedent_nodes.contains(&edge.source()) {
                    1.3 // Outgoing from precedent
                } else if precedent_nodes.contains(&edge.target()) {
                    1.2 // Incoming to precedent
                } else {
                    1.0
                };
                if boost > 1.0 {
                    Some((edge.source(), edge.target(), edge.weight().weight * boost))
                } else {
                    None
                }
            })
            .collect();

        for (src, tgt, new_weight) in edges_to_strengthen {
            if let Some(edge) = result.graph.find_edge(src, tgt) {
                result.graph[edge].weight = new_weight.min(2.0);
            }
        }

        let _ = result.update_topology();
        Ok(result)
    }

    /// Distinguish Precedent: Identify nodes that differ from common patterns.
    /// Weakens edges to low-activation nodes (distinguishable cases).
    fn distinguish_precedent(&self, graph: &DagNN) -> DomainResult<DagNN> {
        use petgraph::visit::EdgeRef;
        let mut result = graph.clone();

        // Weaken edges to low-activation nodes (cases that can be distinguished)
        let edges_to_weaken: Vec<_> = result.graph.edge_references()
            .filter_map(|edge| {
                let tgt_act = result.graph[edge.target()].activation;
                // Low activation = weak precedent, can be distinguished
                if tgt_act < 0.3 {
                    Some((edge.source(), edge.target(), edge.weight().weight * 0.7))
                } else {
                    None
                }
            })
            .collect();

        for (src, tgt, new_weight) in edges_to_weaken {
            if let Some(edge) = result.graph.find_edge(src, tgt) {
                result.graph[edge].weight = new_weight.max(0.1);
            }
        }

        let _ = result.update_topology();
        Ok(result)
    }

    /// IRAC Analysis: Structure for Issue, Rule, Application, Conclusion.
    /// Forms cliques from connected high-activation node groups.
    fn irac_analysis(&self, graph: &DagNN) -> DomainResult<DagNN> {
        use petgraph::visit::EdgeRef;
        let mut result = graph.clone();

        // Find clusters of connected nodes with moderate-to-high activation
        // These represent the IRAC components
        let strong_nodes = result.get_nodes_by_activation(0.5);

        // Strengthen edges within the strong node set (IRAC structure)
        let edges_to_strengthen: Vec<_> = result.graph.edge_references()
            .filter_map(|edge| {
                if strong_nodes.contains(&edge.source()) && strong_nodes.contains(&edge.target()) {
                    Some((edge.source(), edge.target(), edge.weight().weight * 1.2))
                } else {
                    None
                }
            })
            .collect();

        for (src, tgt, new_weight) in edges_to_strengthen {
            if let Some(edge) = result.graph.find_edge(src, tgt) {
                result.graph[edge].weight = new_weight.min(2.0);
            }
        }

        // Form a clique if we have enough strong nodes
        if strong_nodes.len() >= 3 {
            result.form_clique(strong_nodes, Some("IRAC".to_string()));
        }

        let _ = result.update_topology();
        Ok(result)
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

impl DomainBrain for LawBrain {
    fn domain_id(&self) -> &str {
        "law"
    }

    fn domain_name(&self) -> &str {
        "Legal Reasoning"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn can_process(&self, input: &str) -> bool {
        self.looks_like_legal(input)
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        DagNN::from_text(input).map_err(|e| e.into())
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        use petgraph::visit::EdgeRef;
        let mut result = graph.clone();

        // Convert core graph to legal domain representation
        // Strengthen structural edges (legal arguments are structural)
        let edges_to_strengthen: Vec<_> = result.graph.edge_references()
            .filter_map(|edge| {
                match edge.weight().edge_type {
                    grapheme_core::EdgeType::Structural => {
                        Some((edge.source(), edge.target(), edge.weight().weight * 1.2))
                    }
                    grapheme_core::EdgeType::Semantic => {
                        Some((edge.source(), edge.target(), edge.weight().weight * 1.1))
                    }
                    _ => None
                }
            })
            .collect();

        for (src, tgt, new_weight) in edges_to_strengthen {
            if let Some(edge) = result.graph.find_edge(src, tgt) {
                result.graph[edge].weight = new_weight.min(2.0);
            }
        }

        let _ = result.update_topology();
        Ok(result)
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let mut result = graph.clone();

        // Convert legal domain back to core representation
        // Preserve structure and normalize
        let _ = result.update_topology();
        Ok(result)
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        if graph.input_nodes().is_empty() {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Empty legal document graph".to_string(),
                location: None,
            });
        }

        Ok(issues)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        let text = graph.to_text();
        Ok(ExecutionResult::Text(format!("Legal analysis: {}", text)))
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        vec![
            DomainRule::new(0, "Stare Decisis", "Apply precedent from similar cases"),
            DomainRule::new(1, "Distinguish Precedent", "Identify material differences from prior cases"),
            DomainRule::new(2, "IRAC Analysis", "Issue, Rule, Application, Conclusion"),
            DomainRule::new(3, "Citation Validation", "Verify citation format and existence"),
            DomainRule::new(4, "Hierarchy of Authority", "Rank sources by binding authority"),
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        // Legal reasoning transforms are learned through training on legal corpora.
        // These provide structural graph operations as scaffolding.
        match rule_id {
            0 => self.stare_decisis(graph),       // Apply precedent
            1 => self.distinguish_precedent(graph), // Identify differences
            2 => self.irac_analysis(graph),       // IRAC structure
            3 => Ok(graph.clone()),               // Citation validation - requires external DB
            4 => Ok(graph.clone()),               // Hierarchy - requires authority ranking
            _ => Err(grapheme_core::DomainError::InvalidInput(
                format!("Unknown rule ID: {}", rule_id)
            )),
        }
    }

    fn generate_examples(&self, count: usize) -> Vec<DomainExample> {
        let mut examples = Vec::with_capacity(count);

        let patterns = [
            ("Brown v. Board", "desegregation"),
            ("Marbury v. Madison", "judicial review"),
            ("Miranda v. Arizona", "rights warning"),
            ("Roe v. Wade", "privacy rights"),
            ("Gideon v. Wainwright", "right to counsel"),
        ];

        for i in 0..count {
            let (input, output) = patterns[i % patterns.len()];

            if let (Ok(input_graph), Ok(output_graph)) = (
                DagNN::from_text(input),
                DagNN::from_text(output),
            ) {
                let difficulty = ((i % 5) + 1) as u8;
                examples.push(
                    DomainExample::new(
                        serde_json::to_string(&input_graph).unwrap_or_default(),
                        serde_json::to_string(&output_graph).unwrap_or_default()
                    )
                    .with_metadata("domain", "law")
                    .with_metadata("difficulty", format!("{}", difficulty))
                );
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
    fn test_legal_graph_creation() {
        let graph = LegalGraph::new();
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_parse_citation() {
        let graph = LegalGraph::parse_citation("Brown v. Board of Education (1954)").unwrap();
        assert_eq!(graph.node_count(), 1);
        if let LegalNode::Citation { year, .. } = &graph.graph[graph.root.unwrap()] {
            assert_eq!(*year, Some(1954));
        } else {
            panic!("Expected Citation node");
        }
    }

    #[test]
    fn test_law_brain_creation() {
        let brain = LawBrain::new();
        assert_eq!(brain.domain_id(), "law");
        assert_eq!(brain.domain_name(), "Legal Reasoning");
    }

    #[test]
    fn test_law_brain_can_process() {
        let brain = LawBrain::new();
        assert!(brain.can_process("plaintiff vs defendant"));
        assert!(brain.can_process("Brown v. Board of Education"));
        assert!(brain.can_process("pursuant to statute"));
        assert!(!brain.can_process("hello world"));
    }

    #[test]
    fn test_law_brain_get_rules() {
        let brain = LawBrain::new();
        let rules = brain.get_rules();
        assert_eq!(rules.len(), 5);
        assert_eq!(rules[0].name, "Stare Decisis");
    }

    #[test]
    fn test_law_brain_generate_examples() {
        let brain = LawBrain::new();
        let examples = brain.generate_examples(10);
        assert_eq!(examples.len(), 10);
        for example in &examples {
            assert_eq!(example.metadata.get("domain").map(|s| s.as_str()), Some("law"));
        }
    }

    #[test]
    fn test_validate_legal_empty() {
        let brain = LawBrain::new();
        let graph = LegalGraph::new();
        let issues = brain.validate_legal(&graph);
        assert_eq!(issues.len(), 1);
    }
}
