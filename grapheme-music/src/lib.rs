//! # grapheme-music
//!
//! Music Brain: Music theory and composition analysis for GRAPHEME.
//!
//! This crate provides:
//! - Music notation node types (Note, Chord, Scale, Rhythm)
//! - Music theory graph construction
//! - Harmonic analysis and chord progression
//! - Composition structure representation

use grapheme_core::{
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule, ExecutionResult, ValidationIssue,
    ValidationSeverity,
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

/// A music node with activation for gradient flow
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicNode {
    /// The type of this music node
    pub node_type: MusicNodeType,
    /// Activation value for gradient flow during training
    pub activation: f32,
}

impl MusicNode {
    /// Create a new music node with default activation based on type
    pub fn new(node_type: MusicNodeType) -> Self {
        let activation = Self::type_activation(&node_type);
        Self {
            node_type,
            activation,
        }
    }

    /// Get default activation value based on node type importance
    fn type_activation(node_type: &MusicNodeType) -> f32 {
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
        let (name, rest) = if trimmed.len() >= 2 && trimmed.chars().nth(1) == Some('#') {
            let name = match trimmed.chars().next().unwrap() {
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
            let name = match trimmed.chars().next().unwrap() {
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

        let node = graph.add_node(MusicNode::new(MusicNodeType::Note {
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

/// The Music Brain for music theory analysis
pub struct MusicBrain;

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
        Self
    }

    /// Check if text looks like music notation
    fn looks_like_music(&self, input: &str) -> bool {
        let music_patterns = [
            "note",
            "chord",
            "scale",
            "key",
            "major",
            "minor",
            "tempo",
            "bpm",
            "measure",
            "bar",
            "beat",
            "rhythm",
            "sharp",
            "flat",
            "natural",
            "piano",
            "forte",
            "crescendo",
            "staccato",
            "legato",
            "harmony",
        ];
        let lower = input.to_lowercase();
        music_patterns.iter().any(|p| lower.contains(p))
            // Also check for note patterns like C4, D#5
            || input.chars().any(|c| "CDEFGAB".contains(c))
    }

    /// Normalize music text for domain processing
    /// Standardizes note notation and music terminology
    fn normalize_music_text(&self, text: &str) -> String {
        let mut normalized = text.to_string();

        // Normalize note names to uppercase
        for note in ['c', 'd', 'e', 'f', 'g', 'a', 'b'] {
            normalized = normalized.replace(
                &format!("{} ", note),
                &format!("{} ", note.to_ascii_uppercase()),
            );
        }

        // Normalize chord quality abbreviations
        let normalized = normalized
            .replace(" maj ", " major ")
            .replace(" min ", " minor ")
            .replace("maj7", "major7")
            .replace("min7", "minor7");

        // Trim whitespace
        normalized.trim().to_string()
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

impl DomainBrain for MusicBrain {
    fn domain_id(&self) -> &str {
        "music"
    }

    fn domain_name(&self) -> &str {
        "Music Theory"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn can_process(&self, input: &str) -> bool {
        self.looks_like_music(input)
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        DagNN::from_text(input).map_err(|e| e.into())
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Convert core DagNN to music domain representation
        // Normalize note notation and music terminology
        let text = graph.to_text();

        // Apply music-specific normalization
        let normalized = self.normalize_music_text(&text);

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // Convert music domain representation back to generic core format
        let text = graph.to_text();

        // Remove any music-specific annotations
        let cleaned = text
            .lines()
            .filter(|line| !line.trim().starts_with("@music:"))
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
                message: "Empty music graph".to_string(),
                node: None,
            });
        }

        Ok(issues)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        let text = graph.to_text();
        Ok(ExecutionResult::Text(format!("Music: {}", text)))
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
