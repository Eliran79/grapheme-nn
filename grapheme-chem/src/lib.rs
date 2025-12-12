//! # grapheme-chem
//!
//! Chemistry Brain: Molecular structure and reaction analysis for GRAPHEME.
//!
//! This crate provides:
//! - Molecular structure node types (Atom, Bond, Molecule)
//! - Chemical reaction representation
//! - Molecular graph construction
//! - Chemical property analysis
//!
//! ## Migration to brain-common
//!
//! This crate uses shared abstractions from `grapheme-brain-common`:
//! - `ActivatedNode<ChemNodeType>` - Generic node wrapper (aliased as `ChemNode`)
//! - `BaseDomainBrain` - Default implementations for DomainBrain methods
//! - `DomainConfig` - Domain configuration (keywords, normalizer, etc.)

use grapheme_brain_common::{ActivatedNode, BaseDomainBrain, DomainConfig, TextNormalizer};
use grapheme_core::{
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule, ExecutionResult, NodeType,
    ValidationIssue,
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

/// Get default activation value based on chemistry node type importance.
///
/// Used by `new_chem_node()` to compute initial activation from type.
pub fn chem_type_activation(node_type: &ChemNodeType) -> f32 {
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

/// A chemistry node with activation for gradient flow.
///
/// This is a type alias for `ActivatedNode<ChemNodeType>` from brain-common.
pub type ChemNode = ActivatedNode<ChemNodeType>;

/// Create a new chemistry node with default activation based on type.
pub fn new_chem_node(node_type: ChemNodeType) -> ChemNode {
    ActivatedNode::with_type_activation(node_type, chem_type_activation)
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
        let mol = graph.add_node(new_chem_node(ChemNodeType::Molecule {
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
                if let Some(digit) = chars[i].to_digit(10) {
                    count = count * 10 + digit;
                }
                i += 1;
            }
            if count == 0 {
                count = 1;
            }

            // Add atoms
            if let Some(element) = Element::from_symbol(&symbol) {
                for _ in 0..count {
                    let atom = graph.add_node(new_chem_node(ChemNodeType::Atom {
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

/// Create the chemistry domain configuration.
fn create_chem_config() -> DomainConfig {
    // Chemistry keywords for can_process detection
    let keywords = vec![
        "molecule", "atom", "bond", "element", "reaction", "compound",
        "formula", "acid", "base", "salt", "ion", "carbon", "hydrogen",
        "oxygen", "nitrogen", "organic", "inorganic", "polymer",
        "catalyst", "enzyme", "solution", "molar", "mol", "pH", "concentration",
    ];

    // Create normalizer for chemistry notation
    let normalizer = TextNormalizer::new()
        .add_replacements(vec![
            ("->", "→"),
            ("=>", "→"),
            ("<->", "⇌"),
            ("<=>", "⇌"),
            ("water", "H₂O"),
            ("carbon dioxide", "CO₂"),
            ("methane", "CH₄"),
        ])
        .trim_whitespace(true);

    DomainConfig::new("chemistry", "Chemistry", keywords)
        .with_version("0.1.0")
        .with_normalizer(normalizer)
        .with_annotation_prefix("@chem:")
}

/// The Chemistry Brain for molecular analysis.
///
/// Uses DomainConfig from brain-common for keyword detection and normalization.
pub struct ChemBrain {
    /// Domain configuration
    config: DomainConfig,
}

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
        Self {
            config: create_chem_config(),
        }
    }

    /// Check for molecular formula patterns (e.g., H2O, CO2, C6H12O6)
    fn has_formula_pattern(&self, input: &str) -> bool {
        let has_element = input.chars().any(|c| "HCNOS".contains(c));
        let has_subscript = input.chars().any(|c| c.is_ascii_digit());
        has_element && has_subscript
    }
}

// ============================================================================
// BaseDomainBrain Implementation
// ============================================================================

impl BaseDomainBrain for ChemBrain {
    fn config(&self) -> &DomainConfig {
        &self.config
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

impl DomainBrain for ChemBrain {
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
        // Use default keyword-based detection, plus chemistry-specific formula patterns
        self.default_can_process(input) || self.has_formula_pattern(input)
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
        self.default_validate(graph)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        self.default_execute(graph)
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

    /// Returns all semantic node types that ChemBrain can produce.
    ///
    /// Chemistry node types are character-based for now, covering:
    /// - Element symbols (periodic table)
    /// - Subscript digits for molecular formulas
    /// - Bond symbols and reaction arrows
    fn node_types(&self) -> Vec<NodeType> {
        let mut types = Vec::new();

        // Element symbols (uppercase letters)
        for c in 'A'..='Z' {
            types.push(NodeType::Input(c));
        }

        // Lowercase for element symbols (second letter, e.g., Na, Cl, Fe)
        for c in 'a'..='z' {
            types.push(NodeType::Input(c));
        }

        // Digits for subscripts and coefficients
        for c in '0'..='9' {
            types.push(NodeType::Input(c));
        }

        // Chemical symbols
        for c in ['+', '-', '→', '⇌', '(', ')', '[', ']', '·', '↑', '↓'] {
            types.push(NodeType::Input(c));
        }

        // Space for separation
        types.push(NodeType::Input(' '));

        types
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
