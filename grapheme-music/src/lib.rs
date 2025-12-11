//! # grapheme-music
//!
//! Music Brain: Music theory and composition analysis for GRAPHEME.
//!
//! This crate provides:
//! - Music notation node types (Note, Chord, Scale, Rhythm)
//! - Music theory graph construction
//! - Harmonic analysis and chord progression
//! - Composition structure representation
//!
//! ## Migration to brain-common
//!
//! This crate uses shared abstractions from `grapheme-brain-common`:
//! - `ActivatedNode<MusicNodeType>` - Generic node wrapper (aliased as `MusicNode`)
//! - `BaseDomainBrain` - Default implementations for DomainBrain methods
//! - `DomainConfig` - Domain configuration (keywords, normalizer, etc.)

use grapheme_brain_common::{ActivatedNode, BaseDomainBrain, DomainConfig, TextNormalizer};
use grapheme_core::{
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule, ExecutionResult, ValidationIssue,
};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors in music graph processing
#[derive(Error, Debug)]
pub enum MusicGraphError {
    #[error("Invalid note: {0}")]
    InvalidNote(String),
    #[error("Invalid chord: {0}")]
    InvalidChord(String),
    #[error("Rhythm error: {0}")]
    RhythmError(String),
}

/// Result type for music graph operations
pub type MusicGraphResult<T> = Result<T, MusicGraphError>;

/// Musical note names
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NoteName {
    C,
    CSharp,
    D,
    DSharp,
    E,
    F,
    FSharp,
    G,
    GSharp,
    A,
    ASharp,
    B,
}

impl NoteName {
    /// Get semitone offset from C
    pub fn semitone(&self) -> u8 {
        match self {
            NoteName::C => 0,
            NoteName::CSharp => 1,
            NoteName::D => 2,
            NoteName::DSharp => 3,
            NoteName::E => 4,
            NoteName::F => 5,
            NoteName::FSharp => 6,
            NoteName::G => 7,
            NoteName::GSharp => 8,
            NoteName::A => 9,
            NoteName::ASharp => 10,
            NoteName::B => 11,
        }
    }
}

/// Music node types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MusicNodeType {
    /// A musical note
    Note {
        name: NoteName,
        octave: i8,
        duration: Duration,
    },
    /// A chord
    Chord {
        root: NoteName,
        quality: ChordQuality,
    },
    /// A scale
    Scale { root: NoteName, mode: ScaleMode },
    /// Time signature
    TimeSignature { numerator: u8, denominator: u8 },
    /// Key signature
    KeySignature { root: NoteName, mode: ScaleMode },
    /// Tempo marking
    Tempo(u16),
    /// Rest
    Rest(Duration),
    /// Measure/bar
    Measure(u32),
    /// Dynamic marking
    Dynamic(DynamicLevel),
    /// Articulation
    Articulation(ArticulationType),
}

/// Get default activation value based on music node type importance.
///
/// Used by `new_music_node()` to compute initial activation from type.
pub fn music_type_activation(node_type: &MusicNodeType) -> f32 {
    match node_type {
        // Melodic content - high importance
        MusicNodeType::Note { .. } => 0.7,
        MusicNodeType::Chord { .. } => 0.8,
        MusicNodeType::Scale { .. } => 0.75,
        // Structural markers - medium-high importance
        MusicNodeType::TimeSignature { .. } => 0.6,
        MusicNodeType::KeySignature { .. } => 0.85,
        MusicNodeType::Tempo(_) => 0.5,
        // Timing elements
        MusicNodeType::Rest(_) => 0.4,
        MusicNodeType::Measure(_) => 0.3,
        // Expression - medium importance
        MusicNodeType::Dynamic(_) => 0.55,
        MusicNodeType::Articulation(_) => 0.5,
    }
}

/// A music node with activation for gradient flow.
///
/// This is a type alias for `ActivatedNode<MusicNodeType>` from brain-common.
pub type MusicNode = ActivatedNode<MusicNodeType>;

/// Create a new music node with default activation based on type.
pub fn new_music_node(node_type: MusicNodeType) -> MusicNode {
    ActivatedNode::with_type_activation(node_type, music_type_activation)
}

/// Note/rest duration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Duration {
    Whole,
    Half,
    #[default]
    Quarter,
    Eighth,
    Sixteenth,
    ThirtySecond,
}

/// Chord quality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ChordQuality {
    #[default]
    Major,
    Minor,
    Diminished,
    Augmented,
    Dominant7,
    Major7,
    Minor7,
    Suspended2,
    Suspended4,
}

/// Scale mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ScaleMode {
    #[default]
    Major,
    Minor,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Locrian,
    Pentatonic,
    Blues,
}

/// Dynamic levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynamicLevel {
    Pianissimo,
    Piano,
    MezzoPiano,
    MezzoForte,
    Forte,
    Fortissimo,
}

/// Articulation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArticulationType {
    Staccato,
    Legato,
    Accent,
    Tenuto,
    Fermata,
}

/// Edge types in music graphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicEdge {
    /// Sequential (next in time)
    Next,
    /// Simultaneous (chord/harmony)
    Simultaneous,
    /// Part of (note in chord)
    PartOf,
    /// Resolves to
    ResolvesTo,
    /// Modulates to
    ModulatesTo,
    /// In measure
    InMeasure,
}

/// A music piece represented as a graph
#[derive(Debug)]
pub struct MusicGraph {
    /// The underlying directed graph
    pub graph: DiGraph<MusicNode, MusicEdge>,
    /// Root node
    pub root: Option<NodeIndex>,
}

impl Default for MusicGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MusicGraph {
    /// Create a new empty music graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            root: None,
        }
    }

    /// Add a node
    pub fn add_node(&mut self, node: MusicNode) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Add an edge
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: MusicEdge) {
        self.graph.add_edge(from, to, edge);
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Parse a simple note string (e.g., "C4", "D#5")
    pub fn parse_note(note_str: &str) -> MusicGraphResult<Self> {
        let mut graph = Self::new();
        let trimmed = note_str.trim().to_uppercase();

        if trimmed.is_empty() {
            return Err(MusicGraphError::InvalidNote("Empty note".to_string()));
        }

        // Parse note name
        let Some(first_char) = trimmed.chars().next() else {
            return Err(MusicGraphError::InvalidNote("Empty note".to_string()));
        };
        let (name, rest) = if trimmed.len() >= 2 && trimmed.chars().nth(1) == Some('#') {
            let name = match first_char {
                'C' => NoteName::CSharp,
                'D' => NoteName::DSharp,
                'F' => NoteName::FSharp,
                'G' => NoteName::GSharp,
                'A' => NoteName::ASharp,
                _ => {
                    return Err(MusicGraphError::InvalidNote(format!(
                        "Invalid sharp note: {}",
                        trimmed
                    )))
                }
            };
            (name, &trimmed[2..])
        } else {
            let name = match first_char {
                'C' => NoteName::C,
                'D' => NoteName::D,
                'E' => NoteName::E,
                'F' => NoteName::F,
                'G' => NoteName::G,
                'A' => NoteName::A,
                'B' => NoteName::B,
                _ => {
                    return Err(MusicGraphError::InvalidNote(format!(
                        "Invalid note: {}",
                        trimmed
                    )))
                }
            };
            (name, &trimmed[1..])
        };

        // Parse octave
        let octave = rest.parse::<i8>().unwrap_or(4);

        let node = graph.add_node(new_music_node(MusicNodeType::Note {
            name,
            octave,
            duration: Duration::Quarter,
        }));
        graph.root = Some(node);

        Ok(graph)
    }
}

// ============================================================================
// Music Brain
// ============================================================================

/// Create the music domain configuration.
fn create_music_config() -> DomainConfig {
    // Music keywords for can_process detection
    let keywords = vec![
        "note", "chord", "scale", "key", "major", "minor",
        "tempo", "bpm", "measure", "bar", "beat", "rhythm",
        "sharp", "flat", "natural", "piano", "forte",
        "crescendo", "staccato", "legato", "harmony",
    ];

    // Create normalizer for music notation
    let normalizer = TextNormalizer::new()
        .add_replacements(vec![
            (" maj ", " major "),
            (" min ", " minor "),
            ("maj7", "major7"),
            ("min7", "minor7"),
        ])
        .trim_whitespace(true);

    DomainConfig::new("music", "Music Theory", keywords)
        .with_version("0.1.0")
        .with_normalizer(normalizer)
        .with_annotation_prefix("@music:")
}

/// The Music Brain for music theory analysis.
///
/// Uses DomainConfig from brain-common for keyword detection and normalization.
pub struct MusicBrain {
    /// Domain configuration
    config: DomainConfig,
}

impl Default for MusicBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MusicBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MusicBrain")
            .field("domain", &"music")
            .finish()
    }
}

impl MusicBrain {
    /// Create a new music brain
    pub fn new() -> Self {
        Self {
            config: create_music_config(),
        }
    }

    /// Check if text contains note patterns like C4, D#5
    fn has_note_patterns(&self, input: &str) -> bool {
        input.chars().any(|c| "CDEFGAB".contains(c))
    }
}

// ============================================================================
// BaseDomainBrain Implementation
// ============================================================================

impl BaseDomainBrain for MusicBrain {
    fn config(&self) -> &DomainConfig {
        &self.config
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

impl DomainBrain for MusicBrain {
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
        // Use default keyword-based detection, plus music-specific note patterns
        self.default_can_process(input) || self.has_note_patterns(input)
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
                domain: "music".to_string(),
                name: "Voice Leading".to_string(),
                description: "Smooth melodic transitions between chords".to_string(),
                category: "harmony".to_string(),
            },
            DomainRule {
                id: 1,
                domain: "music".to_string(),
                name: "Chord Progression".to_string(),
                description: "Common chord progression patterns".to_string(),
                category: "harmony".to_string(),
            },
            DomainRule {
                id: 2,
                domain: "music".to_string(),
                name: "Key Detection".to_string(),
                description: "Identify the key of a piece".to_string(),
                category: "analysis".to_string(),
            },
            DomainRule {
                id: 3,
                domain: "music".to_string(),
                name: "Rhythm Quantization".to_string(),
                description: "Align notes to beat grid".to_string(),
                category: "rhythm".to_string(),
            },
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        match rule_id {
            0 => self.apply_voice_leading(graph),
            1 => self.apply_chord_progression(graph),
            2 => self.apply_key_detection(graph),
            3 => self.apply_rhythm_quantization(graph),
            _ => Err(grapheme_core::DomainError::InvalidInput(format!(
                "Unknown rule ID: {}",
                rule_id
            ))),
        }
    }

    fn generate_examples(&self, count: usize) -> Vec<DomainExample> {
        let mut examples = Vec::with_capacity(count);

        let patterns = [
            ("C major", "C E G"),
            ("D minor", "D F A"),
            ("G7", "G B D F"),
            ("Am", "A C E"),
        ];

        for i in 0..count {
            let (input, output) = patterns[i % patterns.len()];

            if let (Ok(input_graph), Ok(output_graph)) =
                (DagNN::from_text(input), DagNN::from_text(output))
            {
                examples.push(DomainExample {
                    input: input_graph,
                    output: output_graph,
                    domain: "music".to_string(),
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

impl MusicBrain {
    /// Rule 0: Voice Leading - Smooth melodic transitions between chords
    /// Normalizes note notation for consistent representation
    fn apply_voice_leading(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize note names to uppercase
        let mut normalized = text.clone();
        for note in ['c', 'd', 'e', 'f', 'g', 'a', 'b'] {
            normalized = normalized.replace(
                &format!("{} ", note),
                &format!("{} ", note.to_ascii_uppercase()),
            );
        }

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 1: Chord Progression - Common chord progression patterns
    /// Normalizes chord notation (e.g., "maj" -> "major")
    fn apply_chord_progression(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize chord quality abbreviations
        let normalized = text
            .replace(" maj ", " major ")
            .replace(" min ", " minor ")
            .replace("maj7", "major7")
            .replace("min7", "minor7")
            .replace("dim", "diminished")
            .replace("aug", "augmented");

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 2: Key Detection - Identify the key of a piece
    /// Normalizes key signature notation
    fn apply_key_detection(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize key notation
        let normalized = text
            .replace("key of ", "Key: ")
            .replace("in the key", "Key:");

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Rule 3: Rhythm Quantization - Align notes to beat grid
    /// Normalizes rhythm/timing notation
    fn apply_rhythm_quantization(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        // Normalize beat/rhythm notation
        let normalized = text.trim().to_string();

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
    fn test_note_semitones() {
        assert_eq!(NoteName::C.semitone(), 0);
        assert_eq!(NoteName::A.semitone(), 9);
    }

    #[test]
    fn test_parse_note() {
        let graph = MusicGraph::parse_note("C4").unwrap();
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_parse_sharp_note() {
        let graph = MusicGraph::parse_note("F#5").unwrap();
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_music_brain_creation() {
        let brain = MusicBrain::new();
        assert_eq!(brain.domain_id(), "music");
    }

    #[test]
    fn test_music_brain_can_process() {
        let brain = MusicBrain::new();
        assert!(brain.can_process("C major chord"));
        assert!(brain.can_process("tempo 120 bpm"));
        assert!(!brain.can_process("hello world"));
    }

    #[test]
    fn test_music_brain_get_rules() {
        let brain = MusicBrain::new();
        let rules = brain.get_rules();
        assert_eq!(rules.len(), 4);
        assert_eq!(rules[0].domain, "music");
    }

    #[test]
    fn test_music_brain_generate_examples() {
        let brain = MusicBrain::new();
        let examples = brain.generate_examples(10);
        assert_eq!(examples.len(), 10);
    }
}
