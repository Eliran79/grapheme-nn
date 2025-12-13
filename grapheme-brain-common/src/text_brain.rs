//! TextBrain - The fundamental text-to-graph brain for GRAPHEME.
//!
//! TextBrain is the simplest brain implementation that converts raw text
//! directly to character-level graph representations. This is the foundation
//! of GRAPHEME's vocabulary-free approach.
//!
//! ## Key Properties
//!
//! - **Character-level**: Each character becomes a node (no tokenization)
//! - **Sequential edges**: Adjacent characters are connected
//! - **Perfect reconstruction**: encode(decode(text)) == text (lossless)
//! - **Universal**: Works with any Unicode text without configuration

use crate::{GraphAutoencoder, LatentGraph, AutoencoderError};
use grapheme_core::{
    DagNN, DomainBrain, DomainError, DomainResult, DomainRule, DomainExample,
    ExecutionResult, ValidationIssue, ValidationSeverity, NodeType,
};

/// TextBrain - The fundamental brain for text processing.
///
/// This brain converts raw text into GRAPHEME's character-level graph
/// representation. It's the foundation that other brains can build upon.
///
/// Unlike word-based tokenizers, TextBrain:
/// - Handles ANY Unicode character
/// - Requires no vocabulary
/// - Produces consistent, learnable representations
#[derive(Debug, Default)]
pub struct TextBrain {
    /// Maximum text length to process (0 = unlimited)
    max_length: usize,
}

impl TextBrain {
    /// Create a new TextBrain with default settings.
    pub fn new() -> Self {
        Self { max_length: 0 }
    }

    /// Create a TextBrain with a maximum text length.
    pub fn with_max_length(max_length: usize) -> Self {
        Self { max_length }
    }

    /// Check if input text is valid for processing.
    fn validate_input(&self, input: &str) -> Result<(), DomainError> {
        if self.max_length > 0 && input.len() > self.max_length {
            return Err(DomainError::InvalidInput(format!(
                "Text length {} exceeds maximum {}",
                input.len(),
                self.max_length
            )));
        }
        Ok(())
    }
}

impl DomainBrain for TextBrain {
    fn domain_id(&self) -> &str {
        "text"
    }

    fn domain_name(&self) -> &str {
        "Text Processing"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn can_process(&self, input: &str) -> bool {
        // TextBrain can process any text
        !input.is_empty() && (self.max_length == 0 || input.len() <= self.max_length)
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        self.validate_input(input)?;
        DagNN::from_text(input).map_err(|e| e.into())
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // TextBrain is the core representation - no transformation needed
        Ok(graph.clone())
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        // TextBrain is the core representation - no transformation needed
        Ok(graph.clone())
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        if graph.node_count() == 0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Empty text graph".to_string(),
                location: None,
            });
        }

        // Check for disconnected nodes (should all be connected in text)
        let input_nodes = graph.input_nodes();
        if input_nodes.len() > 1 {
            // There should be sequential edges connecting all input nodes
            let edge_count = graph.graph.edge_count();
            if edge_count < input_nodes.len().saturating_sub(1) {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    message: "Text graph has missing sequential edges".to_string(),
                    location: None,
                });
            }
        }

        Ok(issues)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        // For text, "execution" just returns the text
        let text = graph.to_text();
        Ok(ExecutionResult::Text(text))
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        vec![
            DomainRule::new(0, "Lowercase", "Convert all characters to lowercase"),
            DomainRule::new(1, "Uppercase", "Convert all characters to uppercase"),
            DomainRule::new(2, "Trim", "Remove leading and trailing whitespace"),
            DomainRule::new(3, "Normalize Whitespace", "Collapse multiple spaces to single space"),
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        let text = graph.to_text();

        let transformed = match rule_id {
            0 => text.to_lowercase(),
            1 => text.to_uppercase(),
            2 => text.trim().to_string(),
            3 => {
                // Collapse multiple whitespace to single space
                let mut result = String::with_capacity(text.len());
                let mut prev_was_space = false;
                for ch in text.chars() {
                    if ch.is_whitespace() {
                        if !prev_was_space {
                            result.push(' ');
                            prev_was_space = true;
                        }
                    } else {
                        result.push(ch);
                        prev_was_space = false;
                    }
                }
                result
            }
            _ => return Err(DomainError::InvalidInput(format!("Unknown rule ID: {}", rule_id))),
        };

        DagNN::from_text(&transformed).map_err(|e| e.into())
    }

    fn generate_examples(&self, count: usize) -> Vec<DomainExample> {
        let examples = [
            ("hello", "HELLO"),
            ("  trim  ", "trim"),
            ("Hello World", "hello world"),
            ("multiple   spaces", "multiple spaces"),
            ("Unicode: 你好 مرحبا", "Unicode: 你好 مرحبا"),
        ];

        (0..count)
            .map(|i| {
                let (input, output) = examples[i % examples.len()];
                DomainExample::new(input, output).with_metadata("domain", "text")
            })
            .collect()
    }

    fn node_types(&self) -> Vec<NodeType> {
        // TextBrain primarily uses Input nodes (one per character)
        vec![
            NodeType::Input(' '),  // Placeholder for any char
            NodeType::Hidden,
            NodeType::Output,
        ]
    }
}

impl GraphAutoencoder for TextBrain {
    fn encode(&self, input: &str) -> Result<LatentGraph, AutoencoderError> {
        let graph = self.parse(input).map_err(|e| AutoencoderError::EncodingError(e.to_string()))?;
        Ok(LatentGraph::new("text", graph))
    }

    fn decode(&self, latent: &LatentGraph) -> Result<String, AutoencoderError> {
        self.validate_latent(latent)?;
        Ok(latent.graph.to_text())
    }

    fn reconstruction_loss(&self, original: &str, reconstructed: &str) -> f32 {
        // Perfect match = 0 loss
        if original == reconstructed {
            return 0.0;
        }

        // For text, use exact character-level comparison
        let max_len = original.len().max(reconstructed.len()).max(1);
        let matching: usize = original
            .chars()
            .zip(reconstructed.chars())
            .filter(|(a, b)| a == b)
            .count();

        let len_diff = (original.len() as isize - reconstructed.len() as isize).unsigned_abs();
        let accuracy = matching as f32 / max_len as f32;
        let length_penalty = len_diff as f32 / max_len as f32;

        (1.0 - accuracy + length_penalty * 0.5).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_brain_domain_id() {
        let brain = TextBrain::new();
        assert_eq!(brain.domain_id(), "text");
        assert_eq!(brain.domain_name(), "Text Processing");
    }

    #[test]
    fn test_text_brain_can_process() {
        let brain = TextBrain::new();
        assert!(brain.can_process("hello"));
        assert!(brain.can_process("你好"));
        assert!(!brain.can_process(""));
    }

    #[test]
    fn test_text_brain_parse_roundtrip() {
        let brain = TextBrain::new();
        let text = "Hello, World!";

        let graph = brain.parse(text).unwrap();
        let result = graph.to_text();

        assert_eq!(text, result);
    }

    #[test]
    fn test_text_brain_unicode() {
        let brain = TextBrain::new();
        let texts = ["你好", "مرحبا", "🎉", "∫dx"];

        for text in texts {
            let graph = brain.parse(text).unwrap();
            let result = graph.to_text();
            assert_eq!(text, result, "Failed for: {}", text);
        }
    }

    #[test]
    fn test_text_brain_transforms() {
        let brain = TextBrain::new();

        // Lowercase
        let graph = brain.parse("HELLO").unwrap();
        let result = brain.transform(&graph, 0).unwrap();
        assert_eq!(result.to_text(), "hello");

        // Uppercase
        let graph = brain.parse("hello").unwrap();
        let result = brain.transform(&graph, 1).unwrap();
        assert_eq!(result.to_text(), "HELLO");

        // Trim
        let graph = brain.parse("  hello  ").unwrap();
        let result = brain.transform(&graph, 2).unwrap();
        assert_eq!(result.to_text(), "hello");
    }

    #[test]
    fn test_text_brain_autoencoder() {
        let brain = TextBrain::new();

        let (output, loss) = brain.roundtrip("Hello, World!").unwrap();
        assert_eq!(output, "Hello, World!");
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_text_brain_reconstruction_loss() {
        let brain = TextBrain::new();

        // Perfect match
        assert_eq!(brain.reconstruction_loss("hello", "hello"), 0.0);

        // Completely different
        let loss = brain.reconstruction_loss("aaa", "bbb");
        assert!(loss > 0.5);

        // Partial match
        let loss = brain.reconstruction_loss("hello", "hallo");
        assert!(loss > 0.0 && loss < 0.5);
    }

    #[test]
    fn test_text_brain_max_length() {
        let brain = TextBrain::with_max_length(5);

        assert!(brain.can_process("hello"));
        assert!(!brain.can_process("hello world"));

        assert!(brain.parse("hello").is_ok());
        assert!(brain.parse("hello world").is_err());
    }
}
