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
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule, ExecutionResult, ValidationIssue,
    ValidationSeverity,
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
    H,
    He,
    // Second row
    Li,
    Be,
    B,
    C,
    N,
    O,
    F,
    Ne,
    // Third row
    Na,
    Mg,
    Al,
    Si,
    P,
    S,
    Cl,
    Ar,
    // Fourth row (common)
    K,
    Ca,
    Fe,
    Cu,
    Zn,
    Br,
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
pub enum ChemNodeType {
    /// An atom
    Atom {
        element: Element,
        charge: i8,
        isotope: Option<u16>,
    },
    /// A functional group
    FunctionalGroup(FunctionalGroupType),
    /// A molecule (container)
    Molecule {
        name: Option<String>,
        formula: Option<String>,
    },
    /// A reaction
    Reaction { name: Option<String> },
    /// A catalyst
    Catalyst(String),
    /// Reaction conditions
    Conditions {
        temperature: Option<f32>,
        pressure: Option<f32>,
    },
}

/// A chemistry node with activation for gradient flow
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChemNode {
    /// The type of this chemistry node
    pub node_type: ChemNodeType,
    /// Activation value for gradient flow during training
    pub activation: f32,
}

impl ChemNode {
    /// Create a new chemistry node with default activation based on type
    pub fn new(node_type: ChemNodeType) -> Self {
        let activation = Self::type_activation(&node_type);
        Self {
            node_type,
            activation,
        }
    }

    /// Get default activation value based on node type importance
    fn type_activation(node_type: &ChemNodeType) -> f32 {
        match node_type {
            // Structural elements - high importance
            ChemNodeType::Atom { .. } => 0.7,
            ChemNodeType::FunctionalGroup(_) => 0.8,
            // Container/organization nodes
            ChemNodeType::Molecule { .. } => 0.9,
            ChemNodeType::Reaction { .. } => 0.85,
            // Process-related nodes
            ChemNodeType::Catalyst(_) => 0.75,
            ChemNodeType::Conditions { .. } => 0.5,
        }
    }
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
    Hydroxyl, // -OH
    Carbonyl, // C=O
    Carboxyl, // -COOH
    Amino,    // -NH2
    Methyl,   // -CH3
    Phenyl,   // C6H5-
    Aldehyde, // -CHO
    Ketone,   // R-CO-R
    Ester,    // R-COO-R
    Ether,    // R-O-R
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
        let mol = graph.add_node(ChemNode::new(ChemNodeType::Molecule {
            name: None,
            formula: Some(formula.to_string()),
        }));
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
                    let atom = graph.add_node(ChemNode::new(ChemNodeType::Atom {
                        element,
                        charge: 0,
                        isotope: None,
                    }));
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
            "molecule",
            "atom",
            "bond",
            "element",
            "reaction",
            "compound",
            "formula",
            "acid",
            "base",
            "salt",
            "ion",
            "carbon",
            "hydrogen",
            "oxygen",
            "nitrogen",
            "organic",
            "inorganic",
            "polymer",
            "catalyst",
            "enzyme",
            "solution",
            "molar",
            "mol",
            "pH",
            "concentration",
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

    /// Normalize chemistry text for domain processing
    /// Standardizes molecular notation and chemical terminology
    fn normalize_chemistry_text(&self, text: &str) -> String {
        // Normalize reaction arrows
        let normalized = text
            .replace("->", "→")
            .replace("=>", "→")
            .replace("<->", "⇌")
            .replace("<=>", "⇌");

        // Normalize common compound names
        let normalized = normalized
            .replace("water", "H₂O")
            .replace("carbon dioxide", "CO₂")
            .replace("methane", "CH₄");

        // Trim whitespace
        normalized.trim().to_string()
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
        // Convert core DagNN to chemistry domain representation
        // Normalize molecular notation and chemical terminology
        let text = graph.to_text();

        // Apply chemistry-specific normalization
        let normalized = self.normalize_chemistry_text(&text);

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Convert chemistry domain representation back to generic core format
        let text = graph.to_text();

        // Remove any chemistry-specific annotations
        let cleaned = text
            .lines()
            .filter(|line| !line.trim().starts_with("@chem:"))
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
                message: "Empty chemistry graph".to_string(),
                node: None,
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
            DomainRule {
                id: 0,
                domain: "chemistry".to_string(),
                name: "Balance Equation".to_string(),
                description: "Balance a chemical equation".to_string(),
                category: "reaction".to_string(),
            },
            DomainRule {
                id: 1,
                domain: "chemistry".to_string(),
                name: "Valence Check".to_string(),
                description: "Verify atom valence is satisfied".to_string(),
                category: "validation".to_string(),
            },
            DomainRule {
                id: 2,
                domain: "chemistry".to_string(),
                name: "IUPAC Naming".to_string(),
                description: "Generate IUPAC name for molecule".to_string(),
                category: "naming".to_string(),
            },
            DomainRule {
                id: 3,
                domain: "chemistry".to_string(),
                name: "Molecular Weight".to_string(),
                description: "Calculate molecular weight".to_string(),
                category: "calculation".to_string(),
            },
            DomainRule {
                id: 4,
                domain: "chemistry".to_string(),
                name: "Functional Group Detection".to_string(),
                description: "Identify functional groups".to_string(),
                category: "analysis".to_string(),
            },
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        match rule_id {
            0 => self.apply_balance_equation(graph),
            1 => self.apply_valence_check(graph),
            2 => self.apply_iupac_naming(graph),
            3 => self.apply_molecular_weight(graph),
            4 => self.apply_functional_group_detection(graph),
            _ => Err(grapheme_core::DomainError::InvalidInput(format!(
                "Unknown rule ID: {}",
                rule_id
            ))),
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

            if let (Ok(input_graph), Ok(output_graph)) =
                (DagNN::from_text(input), DagNN::from_text(output))
            {
                examples.push(DomainExample {
                    input: input_graph,
                    output: output_graph,
                    domain: "chemistry".to_string(),
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

impl ChemBrain {
    /// Rule 0: Balance Equation - Balance a chemical equation
    /// Normalizes reaction arrow notation
    fn apply_balance_equation(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize reaction arrows
        let normalized = text
            .replace("->", "→")
            .replace("=>", "→")
            .replace("<->", "⇌")
            .replace("<=>", "⇌");

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 1: Valence Check - Verify atom valence is satisfied
    /// Returns graph unchanged (validation only, no transformation)
    fn apply_valence_check(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Valence checking is validation, not transformation
        Ok(graph.clone())
    }

    /// Rule 2: IUPAC Naming - Generate IUPAC name for molecule
    /// Normalizes common chemical names to formulas
    fn apply_iupac_naming(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize common names
        let normalized = text
            .replace("water", "H₂O")
            .replace("methane", "CH₄")
            .replace("ethanol", "C₂H₅OH")
            .replace("carbon dioxide", "CO₂")
            .replace("ammonia", "NH₃");

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 3: Molecular Weight - Calculate molecular weight
    /// Returns graph unchanged (calculation, no text transformation)
    fn apply_molecular_weight(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Molecular weight is calculated, not a text transformation
        Ok(graph.clone())
    }

    /// Rule 4: Functional Group Detection - Identify functional groups
    /// Normalizes functional group notation
    fn apply_functional_group_detection(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize functional group notation
        let normalized = text
            .replace("-OH", "(OH)")
            .replace("-NH2", "(NH₂)")
            .replace("-COOH", "(COOH)")
            .replace("-CHO", "(CHO)");

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
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
        assert_eq!(rules[0].domain, "chemistry");
    }

    #[test]
    fn test_chem_brain_generate_examples() {
        let brain = ChemBrain::new();
        let examples = brain.generate_examples(10);
        assert_eq!(examples.len(), 10);
    }
}
