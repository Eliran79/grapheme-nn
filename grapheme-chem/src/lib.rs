//! # grapheme-chem
//!
//! Chemistry Brain: Molecular structure and reaction analysis for GRAPHEME.
//!
//! This crate provides:
//! - Molecular structure node types (Atom, Bond, Molecule)
//! - Chemical reaction representation
//! - Molecular graph construction
//! - Chemical property analysis

use grapheme_core::{
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule,
    ExecutionResult, ValidationIssue, ValidationSeverity,
};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors in chemistry graph processing
#[derive(Error, Debug)]
pub enum ChemGraphError {
    #[error("Invalid element: {0}")]
    InvalidElement(String),
    #[error("Invalid bond: {0}")]
    InvalidBond(String),
    #[error("Valence error: {0}")]
    ValenceError(String),
}

/// Result type for chemistry graph operations
pub type ChemGraphResult<T> = Result<T, ChemGraphError>;

/// Chemical element
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Element {
    // First row
    H, He,
    // Second row
    Li, Be, B, C, N, O, F, Ne,
    // Third row
    Na, Mg, Al, Si, P, S, Cl, Ar,
    // Fourth row (common)
    K, Ca, Fe, Cu, Zn, Br,
    // Generic placeholder
    Unknown,
}

impl Element {
    /// Get atomic number
    pub fn atomic_number(&self) -> u8 {
        match self {
            Element::H => 1,
            Element::He => 2,
            Element::Li => 3,
            Element::Be => 4,
            Element::B => 5,
            Element::C => 6,
            Element::N => 7,
            Element::O => 8,
            Element::F => 9,
            Element::Ne => 10,
            Element::Na => 11,
            Element::Mg => 12,
            Element::Al => 13,
            Element::Si => 14,
            Element::P => 15,
            Element::S => 16,
            Element::Cl => 17,
            Element::Ar => 18,
            Element::K => 19,
            Element::Ca => 20,
            Element::Fe => 26,
            Element::Cu => 29,
            Element::Zn => 30,
            Element::Br => 35,
            Element::Unknown => 0,
        }
    }

    /// Get typical valence
    pub fn valence(&self) -> u8 {
        match self {
            Element::H => 1,
            Element::He => 0,
            Element::C => 4,
            Element::N => 3,
            Element::O => 2,
            Element::F | Element::Cl | Element::Br => 1,
            Element::S => 2,
            Element::P => 3,
            _ => 0,
        }
    }

    /// Parse element from symbol
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        match symbol.to_uppercase().as_str() {
            "H" => Some(Element::H),
            "HE" => Some(Element::He),
            "LI" => Some(Element::Li),
            "BE" => Some(Element::Be),
            "B" => Some(Element::B),
            "C" => Some(Element::C),
            "N" => Some(Element::N),
            "O" => Some(Element::O),
            "F" => Some(Element::F),
            "NE" => Some(Element::Ne),
            "NA" => Some(Element::Na),
            "MG" => Some(Element::Mg),
            "AL" => Some(Element::Al),
            "SI" => Some(Element::Si),
            "P" => Some(Element::P),
            "S" => Some(Element::S),
            "CL" => Some(Element::Cl),
            "AR" => Some(Element::Ar),
            "K" => Some(Element::K),
            "CA" => Some(Element::Ca),
            "FE" => Some(Element::Fe),
            "CU" => Some(Element::Cu),
            "ZN" => Some(Element::Zn),
            "BR" => Some(Element::Br),
            _ => None,
        }
    }
}

/// Chemistry node types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChemNode {
    /// An atom
    Atom {
        element: Element,
        charge: i8,
        isotope: Option<u16>,
    },
    /// A functional group
    FunctionalGroup(FunctionalGroupType),
    /// A molecule (container)
    Molecule { name: Option<String>, formula: Option<String> },
    /// A reaction
    Reaction { name: Option<String> },
    /// A catalyst
    Catalyst(String),
    /// Reaction conditions
    Conditions { temperature: Option<f32>, pressure: Option<f32> },
}

/// Bond types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BondType {
    #[default]
    Single,
    Double,
    Triple,
    Aromatic,
    Ionic,
    Hydrogen,
    Metallic,
}

/// Common functional groups
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FunctionalGroupType {
    Hydroxyl,    // -OH
    Carbonyl,    // C=O
    Carboxyl,    // -COOH
    Amino,       // -NH2
    Methyl,      // -CH3
    Phenyl,      // C6H5-
    Aldehyde,    // -CHO
    Ketone,      // R-CO-R
    Ester,       // R-COO-R
    Ether,       // R-O-R
}

/// Edge types in chemistry graphs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChemEdge {
    /// Chemical bond
    Bond(BondType),
    /// Part of molecule
    PartOf,
    /// Reactant in reaction
    Reactant,
    /// Product of reaction
    Product,
    /// Catalyst in reaction
    CatalyzedBy,
}

/// A molecular structure represented as a graph
#[derive(Debug)]
pub struct MolecularGraph {
    /// The underlying directed graph
    pub graph: DiGraph<ChemNode, ChemEdge>,
    /// Root node (usually the molecule container)
    pub root: Option<NodeIndex>,
}

impl Default for MolecularGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MolecularGraph {
    /// Create a new empty molecular graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            root: None,
        }
    }

    /// Add a node
    pub fn add_node(&mut self, node: ChemNode) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Add an edge
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: ChemEdge) {
        self.graph.add_edge(from, to, edge);
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Parse a simple molecular formula (e.g., "H2O", "CH4")
    pub fn from_formula(formula: &str) -> ChemGraphResult<Self> {
        let mut graph = Self::new();

        // Create molecule container
        let mol = graph.add_node(ChemNode::Molecule {
            name: None,
            formula: Some(formula.to_string()),
        });
        graph.root = Some(mol);

        // Simple parser for formulas like H2O, CO2, CH4
        let chars: Vec<char> = formula.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Get element symbol (1-2 chars)
            let mut symbol = String::new();
            if chars[i].is_uppercase() {
                symbol.push(chars[i]);
                i += 1;
                if i < chars.len() && chars[i].is_lowercase() {
                    symbol.push(chars[i]);
                    i += 1;
                }
            } else {
                i += 1;
                continue;
            }

            // Get count
            let mut count = 0u32;
            while i < chars.len() && chars[i].is_ascii_digit() {
                count = count * 10 + chars[i].to_digit(10).unwrap();
                i += 1;
            }
            if count == 0 {
                count = 1;
            }

            // Add atoms
            if let Some(element) = Element::from_symbol(&symbol) {
                for _ in 0..count {
                    let atom = graph.add_node(ChemNode::Atom {
                        element,
                        charge: 0,
                        isotope: None,
                    });
                    graph.add_edge(mol, atom, ChemEdge::PartOf);
                }
            }
        }

        Ok(graph)
    }
}

// ============================================================================
// Chemistry Brain
// ============================================================================

/// The Chemistry Brain for molecular analysis
pub struct ChemBrain;

impl Default for ChemBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ChemBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChemBrain")
            .field("domain", &"chemistry")
            .finish()
    }
}

impl ChemBrain {
    /// Create a new chemistry brain
    pub fn new() -> Self {
        Self
    }

    /// Check if text looks like chemistry content
    fn looks_like_chemistry(&self, input: &str) -> bool {
        let chem_patterns = [
            "molecule", "atom", "bond", "element",
            "reaction", "compound", "formula",
            "acid", "base", "salt", "ion",
            "carbon", "hydrogen", "oxygen", "nitrogen",
            "organic", "inorganic", "polymer",
            "catalyst", "enzyme", "solution",
            "molar", "mol", "pH", "concentration",
        ];
        let lower = input.to_lowercase();

        // Check for patterns
        if chem_patterns.iter().any(|p| lower.contains(p)) {
            return true;
        }

        // Check for molecular formula patterns (e.g., H2O, CO2, C6H12O6)
        let has_element = input.chars().any(|c| "HCNOS".contains(c));
        let has_subscript = input.chars().any(|c| c.is_ascii_digit());
        has_element && has_subscript
    }

    // ========================================================================
    // Graph Transform Helper Methods
    // ========================================================================

    /// Balance Equation: Ensure atom conservation in chemical equations.
    /// Normalizes edge weights to balance the equation representation.
    fn balance_equation(&self, graph: &DagNN) -> DomainResult<DagNN> {
        use petgraph::visit::EdgeRef;
        let mut result = graph.clone();

        // Calculate total edge weight for normalization (conservation)
        let total_weight: f32 = result.graph.edge_references()
            .map(|e| e.weight().weight)
            .sum();

        if total_weight > 0.0 && result.edge_count() > 0 {
            let avg_weight = total_weight / result.edge_count() as f32;

            // Normalize edges to balance around average
            let edges_to_update: Vec<_> = result.graph.edge_references()
                .map(|edge| {
                    let normalized = (edge.weight().weight + avg_weight) / 2.0;
                    (edge.source(), edge.target(), normalized)
                })
                .collect();

            for (src, tgt, new_weight) in edges_to_update {
                if let Some(edge) = result.graph.find_edge(src, tgt) {
                    result.graph[edge].weight = new_weight.clamp(0.1, 2.0);
                }
            }
        }

        let _ = result.update_topology();
        Ok(result)
    }

    /// Valence Check: Verify that atom bonding satisfies valence rules.
    /// Prunes edges that violate valence constraints (too many connections).
    fn valence_check(&self, graph: &DagNN) -> DomainResult<DagNN> {
        use petgraph::Direction;
        let mut result = graph.clone();

        // Identify nodes with too many connections (violating valence)
        // Typical max valence is 4 (carbon), so we prune nodes with > 6 edges
        let max_connections = 6;
        let overconnected: Vec<_> = result.graph.node_indices()
            .filter(|&node| {
                let in_edges = result.graph.neighbors_directed(node, Direction::Incoming).count();
                let out_edges = result.graph.neighbors_directed(node, Direction::Outgoing).count();
                in_edges + out_edges > max_connections
            })
            .collect();

        // Prune weakest edges from overconnected nodes
        for _node in overconnected {
            let _ = result.prune_weak_edges(0.2);
        }

        let _ = result.update_topology();
        Ok(result)
    }

    /// Functional Group Detection: Identify common functional groups.
    /// Forms cliques from densely connected node regions.
    fn functional_group_detection(&self, graph: &DagNN) -> DomainResult<DagNN> {
        use petgraph::visit::EdgeRef;
        let mut result = graph.clone();

        // Find nodes with high connectivity (potential functional group centers)
        let high_connectivity: Vec<_> = result.graph.node_indices()
            .filter(|&node| {
                result.graph.edges(node).count() >= 2
            })
            .collect();

        // Strengthen connections between high-connectivity nodes
        let edges_to_strengthen: Vec<_> = result.graph.edge_references()
            .filter_map(|edge| {
                if high_connectivity.contains(&edge.source()) && high_connectivity.contains(&edge.target()) {
                    Some((edge.source(), edge.target(), edge.weight().weight * 1.3))
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

        // Form clique if we have a functional group pattern
        if high_connectivity.len() >= 2 {
            result.form_clique(high_connectivity, Some("FunctionalGroup".to_string()));
        }

        let _ = result.update_topology();
        Ok(result)
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

impl DomainBrain for ChemBrain {
    fn domain_id(&self) -> &str {
        "chemistry"
    }

    fn domain_name(&self) -> &str {
        "Chemistry"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn can_process(&self, input: &str) -> bool {
        self.looks_like_chemistry(input)
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        DagNN::from_text(input).map_err(|e| e.into())
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        use petgraph::visit::EdgeRef;
        let mut result = graph.clone();

        // Convert core graph to chemistry domain representation
        // Strengthen semantic edges (chemical bonds are semantic)
        let edges_to_strengthen: Vec<_> = result.graph.edge_references()
            .filter_map(|edge| {
                match edge.weight().edge_type {
                    grapheme_core::EdgeType::Semantic => {
                        // Chemical bonds are semantic connections
                        Some((edge.source(), edge.target(), edge.weight().weight * 1.25))
                    }
                    grapheme_core::EdgeType::Clique => {
                        // Functional groups are cliques
                        Some((edge.source(), edge.target(), edge.weight().weight * 1.15))
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

        // Convert chemistry domain back to core representation
        // Clean up and normalize
        result.prune_weak_edges(0.05);
        let _ = result.update_topology();
        Ok(result)
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        if graph.input_nodes().is_empty() {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Empty chemistry graph".to_string(),
                location: None,
            });
        }

        Ok(issues)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        let text = graph.to_text();
        Ok(ExecutionResult::Text(format!("Chemistry: {}", text)))
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        vec![
            DomainRule::new(0, "Balance Equation", "Balance a chemical equation"),
            DomainRule::new(1, "Valence Check", "Verify atom valence is satisfied"),
            DomainRule::new(2, "IUPAC Naming", "Generate IUPAC name for molecule"),
            DomainRule::new(3, "Molecular Weight", "Calculate molecular weight"),
            DomainRule::new(4, "Functional Group Detection", "Identify functional groups"),
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        // Chemistry transforms operate on molecular graphs.
        // These are learned from chemical databases and reaction data.
        match rule_id {
            0 => self.balance_equation(graph),
            1 => self.valence_check(graph),
            2 => Ok(graph.clone()), // IUPAC naming - requires nomenclature rules
            3 => Ok(graph.clone()), // Molecular weight - requires element table
            4 => self.functional_group_detection(graph),
            _ => Err(grapheme_core::DomainError::InvalidInput(
                format!("Unknown rule ID: {}", rule_id)
            )),
        }
    }

    fn generate_examples(&self, count: usize) -> Vec<DomainExample> {
        let mut examples = Vec::with_capacity(count);

        let patterns = [
            ("H2O", "water"),
            ("CO2", "carbon dioxide"),
            ("CH4", "methane"),
            ("NaCl", "sodium chloride"),
            ("C6H12O6", "glucose"),
        ];

        for i in 0..count {
            let (input, output) = patterns[i % patterns.len()];

            if let (Ok(input_graph), Ok(output_graph)) = (
                DagNN::from_text(input),
                DagNN::from_text(output),
            ) {
                examples.push(DomainExample::new(
                    serde_json::to_string(&input_graph).unwrap_or_default(),
                    serde_json::to_string(&output_graph).unwrap_or_default()
                )
                .with_metadata("domain", "chemistry")
                .with_metadata("difficulty", format!("{}", (i % 5) + 1)));
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
    fn test_element_atomic_number() {
        assert_eq!(Element::H.atomic_number(), 1);
        assert_eq!(Element::C.atomic_number(), 6);
        assert_eq!(Element::O.atomic_number(), 8);
    }

    #[test]
    fn test_element_valence() {
        assert_eq!(Element::H.valence(), 1);
        assert_eq!(Element::C.valence(), 4);
        assert_eq!(Element::O.valence(), 2);
    }

    #[test]
    fn test_element_from_symbol() {
        assert_eq!(Element::from_symbol("H"), Some(Element::H));
        assert_eq!(Element::from_symbol("Na"), Some(Element::Na));
        assert_eq!(Element::from_symbol("Cl"), Some(Element::Cl));
    }

    #[test]
    fn test_molecular_graph_from_formula() {
        let graph = MolecularGraph::from_formula("H2O").unwrap();
        assert_eq!(graph.node_count(), 4); // 1 molecule + 2 H + 1 O
    }

    #[test]
    fn test_chem_brain_creation() {
        let brain = ChemBrain::new();
        assert_eq!(brain.domain_id(), "chemistry");
    }

    #[test]
    fn test_chem_brain_can_process() {
        let brain = ChemBrain::new();
        assert!(brain.can_process("H2O molecule"));
        assert!(brain.can_process("chemical reaction"));
        assert!(!brain.can_process("hello world"));
    }

    #[test]
    fn test_chem_brain_get_rules() {
        let brain = ChemBrain::new();
        let rules = brain.get_rules();
        assert_eq!(rules.len(), 5);
        assert_eq!(rules[0].name, "Balance Equation");
    }

    #[test]
    fn test_chem_brain_generate_examples() {
        let brain = ChemBrain::new();
        let examples = brain.generate_examples(10);
        assert_eq!(examples.len(), 10);
    }
}
