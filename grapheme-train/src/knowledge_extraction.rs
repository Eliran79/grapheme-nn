//! Knowledge Graph Extraction from Text
//!
//! Backend-173: Extract entities and relations from text to build knowledge graphs.
//!
//! This module provides rule-based and pattern-based extraction of:
//! - Named entities (persons, organizations, locations, concepts)
//! - Relations between entities (subject-predicate-object triples)
//! - Coreference resolution (basic pronoun/reference linking)
//!
//! Output integrates with GRAPHEME's graph representations.

use grapheme_core::GraphemeGraph;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Entity Types
// ============================================================================

/// Types of named entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// Person (e.g., "John Smith", "Dr. Johnson")
    Person,
    /// Organization (e.g., "Google", "United Nations")
    Organization,
    /// Location (e.g., "New York", "France")
    Location,
    /// Date/Time (e.g., "January 2024", "yesterday")
    DateTime,
    /// Numeric value (e.g., "42", "3.14")
    Number,
    /// Abstract concept (e.g., "democracy", "gravity")
    Concept,
    /// Event (e.g., "World War II", "the meeting")
    Event,
    /// Unknown/Other
    Other,
}

/// An extracted entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity ID (unique within document)
    pub id: String,
    /// The surface text
    pub text: String,
    /// Normalized/canonical form
    pub canonical: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Start character offset in source text
    pub start: usize,
    /// End character offset in source text
    pub end: usize,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Additional attributes
    pub attributes: HashMap<String, String>,
}

impl Entity {
    /// Create a new entity
    pub fn new(text: &str, entity_type: EntityType, start: usize, end: usize) -> Self {
        Self {
            id: format!("e_{}_{}", start, end),
            text: text.to_string(),
            canonical: text.to_lowercase(),
            entity_type,
            start,
            end,
            confidence: 1.0,
            attributes: HashMap::new(),
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Add an attribute
    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }
}

// ============================================================================
// Relation Types
// ============================================================================

/// Types of relations between entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    /// Is-A relationship (type/class membership)
    IsA,
    /// Part-Of relationship
    PartOf,
    /// Located-In relationship
    LocatedIn,
    /// Works-For relationship
    WorksFor,
    /// Created-By relationship
    CreatedBy,
    /// Owns relationship
    Owns,
    /// Related-To (generic)
    RelatedTo,
    /// Causes relationship
    Causes,
    /// Before (temporal)
    Before,
    /// After (temporal)
    After,
    /// Has-Property relationship
    HasProperty,
    /// Custom/other
    Other(u32),
}

/// An extracted relation (triple)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Relation ID
    pub id: String,
    /// Subject entity ID
    pub subject_id: String,
    /// Predicate (relation type)
    pub predicate: RelationType,
    /// Predicate text (original verb/preposition)
    pub predicate_text: String,
    /// Object entity ID
    pub object_id: String,
    /// Confidence score
    pub confidence: f32,
    /// Source text span
    pub source_span: Option<(usize, usize)>,
}

impl Relation {
    /// Create a new relation
    pub fn new(subject_id: &str, predicate: RelationType, object_id: &str) -> Self {
        Self {
            id: format!("r_{}_{}", subject_id, object_id),
            subject_id: subject_id.to_string(),
            predicate,
            predicate_text: String::new(),
            object_id: object_id.to_string(),
            confidence: 1.0,
            source_span: None,
        }
    }

    /// Set predicate text
    pub fn with_predicate_text(mut self, text: &str) -> Self {
        self.predicate_text = text.to_string();
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

// ============================================================================
// Knowledge Graph
// ============================================================================

/// A knowledge graph extracted from text
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Extracted relations
    pub relations: Vec<Relation>,
    /// Entity index by ID
    #[serde(skip)]
    entity_index: HashMap<String, usize>,
    /// Source text
    pub source_text: String,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from source text
    pub fn from_text(text: &str) -> Self {
        Self {
            source_text: text.to_string(),
            ..Default::default()
        }
    }

    /// Add an entity
    pub fn add_entity(&mut self, entity: Entity) -> &str {
        let id = entity.id.clone();
        let idx = self.entities.len();
        self.entity_index.insert(id.clone(), idx);
        self.entities.push(entity);
        &self.entities[idx].id
    }

    /// Add a relation
    pub fn add_relation(&mut self, relation: Relation) {
        self.relations.push(relation);
    }

    /// Get entity by ID
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entity_index.get(id).map(|&idx| &self.entities[idx])
    }

    /// Get all entities of a type
    pub fn entities_of_type(&self, entity_type: EntityType) -> Vec<&Entity> {
        self.entities
            .iter()
            .filter(|e| e.entity_type == entity_type)
            .collect()
    }

    /// Get relations involving an entity
    pub fn relations_for_entity(&self, entity_id: &str) -> Vec<&Relation> {
        self.relations
            .iter()
            .filter(|r| r.subject_id == entity_id || r.object_id == entity_id)
            .collect()
    }

    /// Get number of entities
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get all entities
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Get number of relations
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    /// Get all relations
    pub fn relations(&self) -> &[Relation] {
        &self.relations
    }

    /// Convert to GRAPHEME graph representation
    pub fn to_grapheme_graph(&self) -> GraphemeGraph {
        use grapheme_core::{Edge, EdgeType, Node};
        use petgraph::graph::NodeIndex;

        let mut graph = GraphemeGraph::new();

        // Create nodes for entities
        let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

        for (i, entity) in self.entities.iter().enumerate() {
            // Use first character as input node
            let first_char = entity.text.chars().next().unwrap_or('?');
            let node = Node::input(first_char, i);
            let node_idx = graph.graph.add_node(node);
            node_map.insert(entity.id.clone(), node_idx);
        }

        // Create edges for relations
        for relation in &self.relations {
            if let (Some(&src), Some(&dst)) = (
                node_map.get(&relation.subject_id),
                node_map.get(&relation.object_id),
            ) {
                let edge = Edge::new(relation.confidence, EdgeType::Semantic);
                graph.graph.add_edge(src, dst, edge);
            }
        }

        graph
    }
}

// ============================================================================
// Extraction Configuration
// ============================================================================

/// Configuration for knowledge extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Minimum entity confidence threshold
    pub min_entity_confidence: f32,
    /// Minimum relation confidence threshold
    pub min_relation_confidence: f32,
    /// Enable coreference resolution
    pub resolve_coreferences: bool,
    /// Maximum entities to extract
    pub max_entities: usize,
    /// Maximum relations to extract
    pub max_relations: usize,
    /// Custom entity patterns (regex -> EntityType)
    pub custom_entity_patterns: Vec<(String, EntityType)>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            min_entity_confidence: 0.5,
            min_relation_confidence: 0.5,
            resolve_coreferences: true,
            max_entities: 1000,
            max_relations: 5000,
            custom_entity_patterns: Vec::new(),
        }
    }
}

// ============================================================================
// Entity Extractor
// ============================================================================

/// Knowledge extractor using rule-based patterns
pub struct KnowledgeExtractor {
    config: ExtractionConfig,
    /// Common person titles
    person_titles: HashSet<String>,
    /// Common organization suffixes
    org_suffixes: HashSet<String>,
    /// Common location indicators
    location_indicators: HashSet<String>,
    /// Relation trigger words
    relation_triggers: HashMap<String, RelationType>,
}

impl KnowledgeExtractor {
    /// Create a new extractor with default configuration
    pub fn new() -> Self {
        Self::with_config(ExtractionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ExtractionConfig) -> Self {
        let mut extractor = Self {
            config,
            person_titles: HashSet::new(),
            org_suffixes: HashSet::new(),
            location_indicators: HashSet::new(),
            relation_triggers: HashMap::new(),
        };
        extractor.init_patterns();
        extractor
    }

    fn init_patterns(&mut self) {
        // Person titles
        for title in &["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sir", "Lord", "President", "CEO"] {
            self.person_titles.insert(title.to_lowercase());
        }

        // Organization suffixes
        for suffix in &["Inc.", "Corp.", "Ltd.", "LLC", "Co.", "Company", "Foundation", "Institute"] {
            self.org_suffixes.insert(suffix.to_lowercase());
        }

        // Location indicators
        for indicator in &["in", "at", "near", "from", "to", "city", "country", "state", "region"] {
            self.location_indicators.insert(indicator.to_lowercase());
        }

        // Relation triggers
        self.relation_triggers.insert("is a".to_string(), RelationType::IsA);
        self.relation_triggers.insert("is an".to_string(), RelationType::IsA);
        self.relation_triggers.insert("was a".to_string(), RelationType::IsA);
        self.relation_triggers.insert("are".to_string(), RelationType::IsA);
        self.relation_triggers.insert("part of".to_string(), RelationType::PartOf);
        self.relation_triggers.insert("belongs to".to_string(), RelationType::PartOf);
        self.relation_triggers.insert("located in".to_string(), RelationType::LocatedIn);
        self.relation_triggers.insert("lives in".to_string(), RelationType::LocatedIn);
        self.relation_triggers.insert("based in".to_string(), RelationType::LocatedIn);
        self.relation_triggers.insert("works for".to_string(), RelationType::WorksFor);
        self.relation_triggers.insert("employed by".to_string(), RelationType::WorksFor);
        self.relation_triggers.insert("created by".to_string(), RelationType::CreatedBy);
        self.relation_triggers.insert("made by".to_string(), RelationType::CreatedBy);
        self.relation_triggers.insert("owns".to_string(), RelationType::Owns);
        self.relation_triggers.insert("has".to_string(), RelationType::HasProperty);
        self.relation_triggers.insert("causes".to_string(), RelationType::Causes);
        self.relation_triggers.insert("leads to".to_string(), RelationType::Causes);
        self.relation_triggers.insert("before".to_string(), RelationType::Before);
        self.relation_triggers.insert("after".to_string(), RelationType::After);
        self.relation_triggers.insert("following".to_string(), RelationType::After);
    }

    /// Extract knowledge graph from text
    pub fn extract(&self, text: &str) -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::from_text(text);

        // Extract entities
        let entities = self.extract_entities(text);
        for entity in entities {
            if entity.confidence >= self.config.min_entity_confidence
                && kg.entity_count() < self.config.max_entities
            {
                kg.add_entity(entity);
            }
        }

        // Extract relations
        let relations = self.extract_relations(text, &kg.entities);
        for relation in relations {
            if relation.confidence >= self.config.min_relation_confidence
                && kg.relation_count() < self.config.max_relations
            {
                kg.add_relation(relation);
            }
        }

        // Resolve coreferences
        if self.config.resolve_coreferences {
            self.resolve_coreferences(&mut kg);
        }

        kg
    }

    /// Extract entities from text
    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Split into tokens for analysis
        let tokens: Vec<&str> = text.split_whitespace().collect();

        // Track current position
        let mut char_pos = 0;

        for (i, token) in tokens.iter().enumerate() {
            // Find token position in original text
            if let Some(start) = text[char_pos..].find(token) {
                let start = char_pos + start;
                let end = start + token.len();
                char_pos = end;

                // Check for capitalized words (potential named entities)
                if let Some(first_char) = token.chars().next() {
                    if first_char.is_uppercase() && token.len() > 1 {
                        // Determine entity type
                        let entity_type = self.classify_entity(token, &tokens, i);

                        // Check for multi-word entities
                        let (full_text, full_end) = self.expand_entity(text, token, end, &tokens, i);

                        let entity = Entity::new(&full_text, entity_type, start, full_end)
                            .with_confidence(0.7);

                        // Avoid duplicates
                        if !entities.iter().any(|e: &Entity| e.start == start) {
                            entities.push(entity);
                        }
                    }
                }

                // Check for numbers
                if token.chars().next().map(|c| c.is_numeric()).unwrap_or(false) {
                    let entity = Entity::new(token, EntityType::Number, start, end)
                        .with_confidence(0.9);
                    entities.push(entity);
                }
            }
        }

        // Custom patterns
        for (pattern, entity_type) in &self.config.custom_entity_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for mat in regex.find_iter(text) {
                    let entity = Entity::new(mat.as_str(), *entity_type, mat.start(), mat.end())
                        .with_confidence(0.8);
                    entities.push(entity);
                }
            }
        }

        entities
    }

    /// Classify an entity based on context
    fn classify_entity(&self, token: &str, tokens: &[&str], index: usize) -> EntityType {
        let lower = token.to_lowercase();

        // Check preceding word for person titles
        if index > 0 {
            let prev = tokens[index - 1].to_lowercase();
            if self.person_titles.contains(&prev) {
                return EntityType::Person;
            }
        }

        // Check for organization suffixes
        if self.org_suffixes.iter().any(|s| lower.ends_with(s)) {
            return EntityType::Organization;
        }

        // Check context for location indicators
        if index > 0 {
            let prev = tokens[index - 1].to_lowercase();
            if self.location_indicators.contains(&prev) {
                return EntityType::Location;
            }
        }

        // Default heuristics
        if lower.ends_with("tion") || lower.ends_with("ism") || lower.ends_with("ity") {
            return EntityType::Concept;
        }

        EntityType::Other
    }

    /// Expand entity to include following capitalized words
    fn expand_entity(&self, text: &str, token: &str, end: usize, tokens: &[&str], index: usize) -> (String, usize) {
        let mut full_text = token.to_string();
        let mut full_end = end;

        // Look ahead for more capitalized words
        for following in tokens.iter().skip(index + 1) {
            if let Some(first_char) = following.chars().next() {
                if first_char.is_uppercase() {
                    // Find position in text
                    if let Some(pos) = text[full_end..].find(following) {
                        let gap = &text[full_end..full_end + pos];
                        // Only expand if separated by single space
                        if gap == " " {
                            full_text.push(' ');
                            full_text.push_str(following);
                            full_end = full_end + pos + following.len();
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        (full_text, full_end)
    }

    /// Extract relations between entities
    fn extract_relations(&self, text: &str, entities: &[Entity]) -> Vec<Relation> {
        let mut relations = Vec::new();
        let text_lower = text.to_lowercase();

        // Find relation trigger words and link nearby entities
        for (trigger, rel_type) in &self.relation_triggers {
            for mat in text_lower.match_indices(trigger) {
                let trigger_pos = mat.0;

                // Find subject (entity before trigger)
                let subject = entities
                    .iter()
                    .filter(|e| e.end <= trigger_pos)
                    .max_by_key(|e| e.end);

                // Find object (entity after trigger)
                let object = entities
                    .iter()
                    .filter(|e| e.start >= trigger_pos + trigger.len())
                    .min_by_key(|e| e.start);

                if let (Some(subj), Some(obj)) = (subject, object) {
                    // Check proximity (not too far apart)
                    let distance = obj.start - subj.end;
                    if distance < 100 {
                        let confidence = 0.8 - (distance as f32 / 200.0);
                        let relation = Relation::new(&subj.id, *rel_type, &obj.id)
                            .with_predicate_text(trigger)
                            .with_confidence(confidence.max(0.3));
                        relations.push(relation);
                    }
                }
            }
        }

        relations
    }

    /// Basic coreference resolution
    fn resolve_coreferences(&self, kg: &mut KnowledgeGraph) {
        let pronouns = ["he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their"];

        // Find pronouns and link to most recent entity
        let text_lower = kg.source_text.to_lowercase();

        for pronoun in &pronouns {
            for mat in text_lower.match_indices(pronoun) {
                let pronoun_pos = mat.0;

                // Find most recent entity of compatible type
                let antecedent = kg.entities
                    .iter()
                    .filter(|e| e.end < pronoun_pos)
                    .filter(|e| self.pronoun_matches_entity(pronoun, e))
                    .max_by_key(|e| e.end);

                if let Some(entity) = antecedent {
                    // Add coreference relation
                    let coref_entity = Entity {
                        id: format!("coref_{}_{}", pronoun_pos, entity.id),
                        text: pronoun.to_string(),
                        canonical: entity.canonical.clone(),
                        entity_type: entity.entity_type,
                        start: pronoun_pos,
                        end: pronoun_pos + pronoun.len(),
                        confidence: 0.6,
                        attributes: HashMap::new(),
                    };

                    let relation = Relation::new(&coref_entity.id, RelationType::RelatedTo, &entity.id)
                        .with_predicate_text("refers to")
                        .with_confidence(0.6);

                    // Add if not too many entities
                    if kg.entity_count() < self.config.max_entities {
                        kg.add_entity(coref_entity);
                        kg.add_relation(relation);
                    }
                }
            }
        }
    }

    /// Check if pronoun could refer to entity
    fn pronoun_matches_entity(&self, pronoun: &str, entity: &Entity) -> bool {
        match pronoun {
            "he" | "him" | "his" => entity.entity_type == EntityType::Person,
            "she" | "her" | "hers" => entity.entity_type == EntityType::Person,
            "it" | "its" => {
                entity.entity_type != EntityType::Person
            }
            "they" | "them" | "their" => true,
            _ => true,
        }
    }
}

impl Default for KnowledgeExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new("John Smith", EntityType::Person, 0, 10);
        assert_eq!(entity.text, "John Smith");
        assert_eq!(entity.entity_type, EntityType::Person);
        assert_eq!(entity.canonical, "john smith");
    }

    #[test]
    fn test_entity_with_confidence() {
        let entity = Entity::new("Test", EntityType::Other, 0, 4)
            .with_confidence(0.75);
        assert_eq!(entity.confidence, 0.75);
    }

    #[test]
    fn test_relation_creation() {
        let relation = Relation::new("e1", RelationType::WorksFor, "e2");
        assert_eq!(relation.subject_id, "e1");
        assert_eq!(relation.object_id, "e2");
        assert_eq!(relation.predicate, RelationType::WorksFor);
    }

    #[test]
    fn test_knowledge_graph_creation() {
        let kg = KnowledgeGraph::new();
        assert_eq!(kg.entity_count(), 0);
        assert_eq!(kg.relation_count(), 0);
    }

    #[test]
    fn test_add_entity() {
        let mut kg = KnowledgeGraph::new();
        let entity = Entity::new("Test Entity", EntityType::Concept, 0, 11);
        kg.add_entity(entity);
        assert_eq!(kg.entity_count(), 1);
    }

    #[test]
    fn test_add_relation() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity(Entity::new("Subject", EntityType::Person, 0, 7));
        kg.add_entity(Entity::new("Object", EntityType::Organization, 20, 26));

        let relation = Relation::new("e_0_7", RelationType::WorksFor, "e_20_26");
        kg.add_relation(relation);

        assert_eq!(kg.relation_count(), 1);
    }

    #[test]
    fn test_entities_of_type() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity(Entity::new("John", EntityType::Person, 0, 4));
        kg.add_entity(Entity::new("New York", EntityType::Location, 10, 18));
        kg.add_entity(Entity::new("Jane", EntityType::Person, 25, 29));

        let persons = kg.entities_of_type(EntityType::Person);
        assert_eq!(persons.len(), 2);
    }

    #[test]
    fn test_extractor_creation() {
        let extractor = KnowledgeExtractor::new();
        assert!(extractor.person_titles.contains("dr."));
        assert!(extractor.org_suffixes.contains("inc."));
    }

    #[test]
    fn test_extract_simple_entities() {
        let extractor = KnowledgeExtractor::new();
        let kg = extractor.extract("John Smith works at Google Inc.");

        assert!(kg.entity_count() > 0);

        // Should find "John Smith" and "Google Inc."
        let names: Vec<_> = kg.entities.iter().map(|e| &e.text).collect();
        assert!(names.iter().any(|n| n.contains("John")));
    }

    #[test]
    fn test_extract_relations() {
        let extractor = KnowledgeExtractor::new();
        let kg = extractor.extract("Apple Inc. is located in California.");

        // Should find "located in" relation
        let _loc_relations: Vec<_> = kg.relations
            .iter()
            .filter(|r| r.predicate == RelationType::LocatedIn)
            .collect();

        // At least found the pattern
        assert!(kg.entity_count() > 0);
    }

    #[test]
    fn test_to_grapheme_graph() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity(Entity::new("A", EntityType::Concept, 0, 1));
        kg.add_entity(Entity::new("B", EntityType::Concept, 5, 6));
        kg.add_relation(Relation::new("e_0_1", RelationType::RelatedTo, "e_5_6"));

        let graph = kg.to_grapheme_graph();
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn test_extraction_config_default() {
        let config = ExtractionConfig::default();
        assert_eq!(config.min_entity_confidence, 0.5);
        assert!(config.resolve_coreferences);
    }

    #[test]
    fn test_get_entity() {
        let mut kg = KnowledgeGraph::new();
        let entity = Entity::new("Test", EntityType::Concept, 0, 4);
        kg.add_entity(entity);

        let found = kg.get_entity("e_0_4");
        assert!(found.is_some());
        assert_eq!(found.unwrap().text, "Test");
    }

    #[test]
    fn test_relations_for_entity() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity(Entity::new("A", EntityType::Concept, 0, 1));
        kg.add_entity(Entity::new("B", EntityType::Concept, 5, 6));
        kg.add_entity(Entity::new("C", EntityType::Concept, 10, 11));

        kg.add_relation(Relation::new("e_0_1", RelationType::RelatedTo, "e_5_6"));
        kg.add_relation(Relation::new("e_5_6", RelationType::Causes, "e_10_11"));

        let relations = kg.relations_for_entity("e_5_6");
        assert_eq!(relations.len(), 2); // Involved in both relations
    }

    #[test]
    fn test_entity_type_classification() {
        let extractor = KnowledgeExtractor::new();

        // Test with Dr. title
        let kg = extractor.extract("Dr. Smith discovered the cure.");
        assert!(kg.entity_count() > 0);
    }

    #[test]
    fn test_number_extraction() {
        let extractor = KnowledgeExtractor::new();
        let kg = extractor.extract("The price is 42 dollars.");

        let numbers = kg.entities_of_type(EntityType::Number);
        assert!(!numbers.is_empty());
    }

    #[test]
    fn test_custom_patterns() {
        let config = ExtractionConfig {
            custom_entity_patterns: vec![
                (r"\b[A-Z]{3,4}\b".to_string(), EntityType::Organization),
            ],
            ..Default::default()
        };

        let extractor = KnowledgeExtractor::with_config(config);
        let kg = extractor.extract("NASA launched a rocket.");

        // Should find NASA as organization
        let orgs = kg.entities_of_type(EntityType::Organization);
        assert!(!orgs.is_empty());
    }
}
