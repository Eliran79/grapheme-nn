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
                node: None,
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
                        node: Some(node_idx),
                    });
                }
            }
        }

        issues
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
        Ok(graph.clone())
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        Ok(graph.clone())
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        if graph.input_nodes().is_empty() {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Empty legal document graph".to_string(),
                node: None,
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
            DomainRule {
                id: 0,
                domain: "law".to_string(),
                name: "Stare Decisis".to_string(),
                description: "Apply precedent from similar cases".to_string(),
                category: "reasoning".to_string(),
            },
            DomainRule {
                id: 1,
                domain: "law".to_string(),
                name: "Distinguish Precedent".to_string(),
                description: "Identify material differences from prior cases".to_string(),
                category: "reasoning".to_string(),
            },
            DomainRule {
                id: 2,
                domain: "law".to_string(),
                name: "IRAC Analysis".to_string(),
                description: "Issue, Rule, Application, Conclusion".to_string(),
                category: "structure".to_string(),
            },
            DomainRule {
                id: 3,
                domain: "law".to_string(),
                name: "Citation Validation".to_string(),
                description: "Verify citation format and existence".to_string(),
                category: "validation".to_string(),
            },
            DomainRule {
                id: 4,
                domain: "law".to_string(),
                name: "Hierarchy of Authority".to_string(),
                description: "Rank sources by binding authority".to_string(),
                category: "analysis".to_string(),
            },
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        match rule_id {
            0..=4 => Ok(graph.clone()),
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
                examples.push(DomainExample {
                    input: input_graph,
                    output: output_graph,
                    domain: "law".to_string(),
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
        assert_eq!(rules[0].domain, "law");
        assert_eq!(rules[0].name, "Stare Decisis");
    }

    #[test]
    fn test_law_brain_generate_examples() {
        let brain = LawBrain::new();
        let examples = brain.generate_examples(10);
        assert_eq!(examples.len(), 10);
        for example in &examples {
            assert_eq!(example.domain, "law");
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
