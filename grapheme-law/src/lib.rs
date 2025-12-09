//! # grapheme-law
//!
//! Law Brain: Legal reasoning and case law analysis for GRAPHEME.
//!
//! This crate provides:
//! - Legal document node types (Citation, Statute, Precedent, Argument)
//! - Case law graph construction
//! - Legal reasoning and precedent analysis
//! - Argument structure representation
//!
//! ## Migration to brain-common
//!
//! This crate uses shared abstractions from `grapheme-brain-common`:
//! - `ActivatedNode<LegalNodeType>` - Generic node wrapper (aliased as `LegalNode`)
//! - `BaseDomainBrain` - Default implementations for DomainBrain methods
//! - `DomainConfig` - Domain configuration (keywords, normalizer, etc.)

use grapheme_brain_common::{ActivatedNode, BaseDomainBrain, DomainConfig, TextNormalizer};
use grapheme_core::{
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule, ExecutionResult, ValidationIssue,
    ValidationSeverity,
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
pub enum LegalNodeType {
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
    Argument { premise: String, conclusion: String },
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

/// Get the default activation value for a legal node type.
/// Higher values indicate more important legal elements.
///
/// Used by `new_legal_node()` to compute initial activation from type.
pub fn legal_type_activation(node_type: &LegalNodeType) -> f32 {
    match node_type {
        LegalNodeType::Citation { .. } => 0.8,      // Citations are foundational
        LegalNodeType::Statute { .. } => 0.9,       // Statutes are authoritative
        LegalNodeType::Holding(_) => 0.85,          // Holdings are key outcomes
        LegalNodeType::Argument { .. } => 0.7,      // Arguments support conclusions
        LegalNodeType::Party { .. } => 0.4,         // Parties are contextual
        LegalNodeType::Issue(_) => 0.75,            // Issues frame the case
        LegalNodeType::Fact(_) => 0.5,              // Facts are evidence
        LegalNodeType::Rule(_) => 0.8,              // Rules are authoritative
        LegalNodeType::Application(_) => 0.6,       // Application connects rule to facts
        LegalNodeType::Conclusion(_) => 0.85,       // Conclusions are key outcomes
        LegalNodeType::Opinion { .. } => 0.7,       // Opinions provide reasoning
    }
}

/// Legal node with activation for gradient flow (Backend-118)
///
/// This is a type alias for `ActivatedNode<LegalNodeType>` from brain-common.
pub type LegalNode = ActivatedNode<LegalNodeType>;

/// Create a new legal node with default activation based on type.
pub fn new_legal_node(node_type: LegalNodeType) -> LegalNode {
    ActivatedNode::with_type_activation(node_type, legal_type_activation)
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

        let node = graph.add_node(new_legal_node(LegalNodeType::Citation {
            case_name: trimmed.to_string(),
            year,
            volume: None,
            page: None,
        }));
        graph.root = Some(node);

        Ok(graph)
    }
}

// ============================================================================
// Law Brain
// ============================================================================

/// Create the law domain configuration.
fn create_law_config() -> DomainConfig {
    // Legal keywords for can_process detection
    let keywords = vec![
        " v. ", " vs. ", "plaintiff", "defendant", "appellant", "appellee",
        "statute", "§", "U.S.C.", "U.S. ", "F.2d", "F.3d", "S.Ct.",
        "court", "judge", "ruling", "precedent", "holding", "dissent",
        "concur", "jurisdiction", "liable", "damages", "tort", "contract",
        "constitutional", "amendment", "rights",
    ];

    // Create normalizer for legal citations
    let normalizer = TextNormalizer::new()
        .add_replacements(vec![
            (" vs. ", " v. "),
            (" vs ", " v. "),
            ("U.S.C", "U.S.C."),
            ("S. Ct.", "S.Ct."),
            ("F. 2d", "F.2d"),
            ("F. 3d", "F.3d"),
        ])
        .trim_whitespace(true);

    DomainConfig::new("law", "Legal Reasoning", keywords)
        .with_version("0.1.0")
        .with_normalizer(normalizer)
        .with_annotation_prefix("[legal:")
}

/// The Law Brain for legal reasoning.
///
/// Uses DomainConfig from brain-common for keyword detection and normalization.
pub struct LawBrain {
    /// Domain configuration
    config: DomainConfig,
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
            config: create_law_config(),
            jurisdictions: vec![
                Jurisdiction::USFederal,
                Jurisdiction::USState,
                Jurisdiction::UK,
                Jurisdiction::EU,
            ],
        }
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
            if let LegalNode { node_type: LegalNodeType::Citation { case_name, .. }, .. } = &graph.graph[node_idx] {
                let incoming = graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .count();
                let outgoing = graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Outgoing)
                    .count();
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
// BaseDomainBrain Implementation
// ============================================================================

impl BaseDomainBrain for LawBrain {
    fn config(&self) -> &DomainConfig {
        &self.config
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

impl DomainBrain for LawBrain {
    fn domain_id(&self) -> &str {
        &self.config.domain_id
    }

    fn domain_name(&self) -> &str {
        &self.config.domain_name
    }

    fn version(&self) -> &str {
        &self.config.version
    }

    fn can_process(&self, input: &str) -> bool {
        self.default_can_process(input)
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        self.default_parse(input)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        self.default_from_core(graph)
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        self.default_to_core(graph)
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
            0 => self.apply_stare_decisis(graph),
            1 => self.apply_distinguish_precedent(graph),
            2 => self.apply_irac_analysis(graph),
            3 => self.apply_citation_validation(graph),
            4 => self.apply_hierarchy_of_authority(graph),
            _ => Err(grapheme_core::DomainError::InvalidInput(format!(
                "Unknown rule ID: {}",
                rule_id
            ))),
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

            if let (Ok(input_graph), Ok(output_graph)) =
                (DagNN::from_text(input), DagNN::from_text(output))
            {
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
// Transform Helper Methods
// ============================================================================

impl LawBrain {
    /// Rule 0: Stare Decisis - Apply precedent from similar cases
    /// Normalizes case citation format for consistent precedent references
    fn apply_stare_decisis(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize "vs." to "v." for consistent citation format
        let normalized = text.replace(" vs. ", " v. ").replace(" vs ", " v. ");

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 1: Distinguish Precedent - Identify material differences
    /// Cleans up comparative language for clearer analysis
    fn apply_distinguish_precedent(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize distinguishing language
        let normalized = text
            .replace("differs from", "is distinguished from")
            .replace("unlike in", "distinguished from");

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 2: IRAC Analysis - Issue, Rule, Application, Conclusion
    /// Trims and normalizes whitespace for cleaner structure
    fn apply_irac_analysis(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize whitespace for cleaner IRAC structure
        let normalized = text.trim().to_string();

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 3: Citation Validation - Verify citation format and existence
    /// Normalizes common citation format inconsistencies
    fn apply_citation_validation(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize citation abbreviations
        let normalized = text
            .replace("U.S.C", "U.S.C.")
            .replace("S. Ct.", "S.Ct.")
            .replace("S.Ct", "S.Ct.")
            .replace("F. 2d", "F.2d")
            .replace("F. 3d", "F.3d");

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 4: Hierarchy of Authority - Rank sources by binding authority
    /// No transformation needed at character graph level
    fn apply_hierarchy_of_authority(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Hierarchy analysis doesn't modify the text representation
        Ok(graph.clone())
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
        if let LegalNode { node_type: LegalNodeType::Citation { year, .. }, .. } = &graph.graph[graph.root.unwrap()] {
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
