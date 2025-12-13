//! Base trait for domain brain implementations with default behaviors.
//!
//! This module provides `BaseDomainBrain`, a trait extension that provides
//! default implementations for common DomainBrain methods to reduce code
//! duplication across cognitive brain implementations.

use grapheme_core::{
    DagNN, DomainExample, DomainResult, DomainRule, ExecutionResult,
    ValidationIssue, ValidationSeverity,
};

use crate::{KeywordCapabilityDetector, TextNormalizer};

/// Configuration for a domain brain implementation.
///
/// This struct provides the domain-specific data needed by BaseDomainBrain
/// to implement common functionality.
#[derive(Debug, Clone)]
pub struct DomainConfig {
    /// Unique domain identifier (e.g., "math", "code", "law")
    pub domain_id: String,
    /// Human-readable domain name
    pub domain_name: String,
    /// Version string (defaults to "0.1.0")
    pub version: String,
    /// Keyword detector for can_process()
    pub capability_detector: KeywordCapabilityDetector,
    /// Normalizer for input preprocessing
    pub normalizer: TextNormalizer,
    /// Domain-specific annotation prefix for to_core filtering
    /// (e.g., "// @code:", "[legal:", "@music:", "@chem:")
    pub annotation_prefix: Option<String>,
}

impl DomainConfig {
    /// Create a new domain configuration with required fields.
    ///
    /// # Arguments
    /// * `domain_id` - Unique domain identifier
    /// * `domain_name` - Human-readable name
    /// * `keywords` - Keywords for capability detection
    pub fn new<I, S>(domain_id: impl Into<String>, domain_name: impl Into<String>, keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            domain_id: domain_id.into(),
            domain_name: domain_name.into(),
            version: "0.1.0".to_string(),
            capability_detector: KeywordCapabilityDetector::new(keywords),
            normalizer: TextNormalizer::new(),
            annotation_prefix: None,
        }
    }

    /// Set the version string
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the normalizer
    pub fn with_normalizer(mut self, normalizer: TextNormalizer) -> Self {
        self.normalizer = normalizer;
        self
    }

    /// Set the annotation prefix for to_core filtering
    pub fn with_annotation_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.annotation_prefix = Some(prefix.into());
        self
    }

    /// Add additional keywords to the capability detector
    pub fn add_keywords<I, S>(mut self, keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.capability_detector = self.capability_detector.add_keywords(keywords);
        self
    }

    /// Configure the capability detector's minimum match count
    pub fn with_min_keyword_matches(mut self, count: usize) -> Self {
        self.capability_detector = self.capability_detector.min_matches(count);
        self
    }
}

/// Extension trait providing default implementations for common DomainBrain methods.
///
/// This trait works alongside `DomainBrain` to provide sensible defaults
/// that most brain implementations share. By implementing this trait along
/// with providing a `DomainConfig`, brains can reduce boilerplate significantly.
///
/// # Example
///
/// ```ignore
/// use grapheme_brain_common::{BaseDomainBrain, DomainConfig};
/// use grapheme_core::DomainBrain;
///
/// struct MyBrain {
///     config: DomainConfig,
/// }
///
/// impl BaseDomainBrain for MyBrain {
///     fn config(&self) -> &DomainConfig {
///         &self.config
///     }
/// }
///
/// impl DomainBrain for MyBrain {
///     fn domain_id(&self) -> &str { self.config.domain_id.as_str() }
///     fn domain_name(&self) -> &str { self.config.domain_name.as_str() }
///     fn version(&self) -> &str { self.config.version.as_str() }
///     fn can_process(&self, input: &str) -> bool { self.default_can_process(input) }
///     fn parse(&self, input: &str) -> DomainResult<DagNN> { self.default_parse(input) }
///     // ... use other default_* methods as needed
/// }
/// ```
pub trait BaseDomainBrain: Send + Sync + std::fmt::Debug {
    /// Get the domain configuration
    fn config(&self) -> &DomainConfig;

    /// Default implementation for can_process using keyword detection.
    fn default_can_process(&self, input: &str) -> bool {
        self.config().capability_detector.can_process(input)
    }

    /// Default implementation for parse using DagNN::from_text.
    fn default_parse(&self, input: &str) -> DomainResult<DagNN> {
        DagNN::from_text(input).map_err(|e| e.into())
    }

    /// Default implementation for from_core with normalization.
    ///
    /// Gets text from graph, applies normalization, and recreates if changed.
    fn default_from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();
        let normalized = self.config().normalizer.normalize(&text);

        if normalized != text {
            DagNN::from_text(&normalized).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Default implementation for to_core with annotation filtering.
    ///
    /// Removes lines starting with the domain's annotation prefix.
    fn default_to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        let text = graph.to_text();

        let cleaned = if let Some(ref prefix) = self.config().annotation_prefix {
            text.lines()
                .filter(|line| !line.trim().starts_with(prefix))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            text.clone()
        };

        if cleaned != text {
            DagNN::from_text(&cleaned).map_err(|e| e.into())
        } else {
            Ok(graph.clone())
        }
    }

    /// Default implementation for validate checking for empty input.
    fn default_validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        if graph.input_nodes().is_empty() {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Graph has no input nodes".to_string(),
                location: None,
            });
        }

        Ok(issues)
    }

    /// Default implementation for execute returning text representation.
    fn default_execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        let text = graph.to_text();
        Ok(ExecutionResult::Text(format!("{}: {}", self.config().domain_name, text)))
    }

    /// Default implementation for get_rules returning empty rules.
    fn default_get_rules(&self) -> Vec<DomainRule> {
        Vec::new()
    }

    /// Default implementation for transform that does nothing.
    fn default_transform(&self, graph: &DagNN, _rule_id: usize) -> DomainResult<DagNN> {
        Ok(graph.clone())
    }

    /// Default implementation for generate_examples returning empty list.
    fn default_generate_examples(&self, _count: usize) -> Vec<DomainExample> {
        Vec::new()
    }
}

/// Helper macro to implement DomainBrain using BaseDomainBrain defaults.
///
/// This macro generates a DomainBrain implementation that delegates to
/// the corresponding default_* methods from BaseDomainBrain.
///
/// # Example
///
/// ```ignore
/// impl_domain_brain_defaults!(MyBrain);
/// ```
#[macro_export]
macro_rules! impl_domain_brain_defaults {
    ($brain_type:ty) => {
        impl grapheme_core::DomainBrain for $brain_type {
            fn domain_id(&self) -> &str {
                &self.config().domain_id
            }

            fn domain_name(&self) -> &str {
                &self.config().domain_name
            }

            fn version(&self) -> &str {
                &self.config().version
            }

            fn can_process(&self, input: &str) -> bool {
                $crate::BaseDomainBrain::default_can_process(self, input)
            }

            fn parse(&self, input: &str) -> grapheme_core::DomainResult<grapheme_core::DagNN> {
                $crate::BaseDomainBrain::default_parse(self, input)
            }

            #[allow(clippy::wrong_self_convention)]
            fn from_core(&self, graph: &grapheme_core::DagNN) -> grapheme_core::DomainResult<grapheme_core::DagNN> {
                $crate::BaseDomainBrain::default_from_core(self, graph)
            }

            fn to_core(&self, graph: &grapheme_core::DagNN) -> grapheme_core::DomainResult<grapheme_core::DagNN> {
                $crate::BaseDomainBrain::default_to_core(self, graph)
            }

            fn validate(&self, graph: &grapheme_core::DagNN) -> grapheme_core::DomainResult<Vec<grapheme_core::ValidationIssue>> {
                $crate::BaseDomainBrain::default_validate(self, graph)
            }

            fn execute(&self, graph: &grapheme_core::DagNN) -> grapheme_core::DomainResult<grapheme_core::ExecutionResult> {
                $crate::BaseDomainBrain::default_execute(self, graph)
            }

            fn get_rules(&self) -> Vec<grapheme_core::DomainRule> {
                $crate::BaseDomainBrain::default_get_rules(self)
            }

            fn transform(&self, graph: &grapheme_core::DagNN, rule_id: usize) -> grapheme_core::DomainResult<grapheme_core::DagNN> {
                $crate::BaseDomainBrain::default_transform(self, graph, rule_id)
            }

            fn generate_examples(&self, count: usize) -> Vec<grapheme_core::DomainExample> {
                $crate::BaseDomainBrain::default_generate_examples(self, count)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestBrain {
        config: DomainConfig,
    }

    impl TestBrain {
        fn new() -> Self {
            Self {
                config: DomainConfig::new("test", "Test Domain", vec!["keyword1", "keyword2"])
                    .with_annotation_prefix("@test:"),
            }
        }
    }

    impl BaseDomainBrain for TestBrain {
        fn config(&self) -> &DomainConfig {
            &self.config
        }
    }

    #[test]
    fn test_domain_config_new() {
        let config = DomainConfig::new("math", "Mathematics", vec!["solve", "calculate"]);
        assert_eq!(config.domain_id, "math");
        assert_eq!(config.domain_name, "Mathematics");
        assert_eq!(config.version, "0.1.0");
    }

    #[test]
    fn test_domain_config_with_version() {
        let config = DomainConfig::new("math", "Mathematics", vec!["solve"])
            .with_version("1.0.0");
        assert_eq!(config.version, "1.0.0");
    }

    #[test]
    fn test_domain_config_with_annotation_prefix() {
        let config = DomainConfig::new("code", "Source Code", vec!["function"])
            .with_annotation_prefix("// @code:");
        assert_eq!(config.annotation_prefix, Some("// @code:".to_string()));
    }

    #[test]
    fn test_domain_config_add_keywords() {
        let config = DomainConfig::new("math", "Mathematics", vec!["solve"])
            .add_keywords(vec!["calculate", "compute"]);
        assert_eq!(config.capability_detector.keywords().len(), 3);
    }

    #[test]
    fn test_default_can_process() {
        let brain = TestBrain::new();
        assert!(brain.default_can_process("contains keyword1 here"));
        assert!(!brain.default_can_process("no keywords"));
    }

    #[test]
    fn test_default_parse() {
        let brain = TestBrain::new();
        let result = brain.default_parse("test input");
        assert!(result.is_ok());
    }

    #[test]
    fn test_default_execute() {
        let brain = TestBrain::new();
        let graph = DagNN::from_text("test").unwrap();
        let result = brain.default_execute(&graph);
        assert!(result.is_ok());
        if let Ok(ExecutionResult::Text(text)) = result {
            assert!(text.starts_with("Test Domain:"));
        }
    }

    #[test]
    fn test_default_validate_empty_graph() {
        let brain = TestBrain::new();
        let graph = DagNN::new();
        let result = brain.default_validate(&graph);
        assert!(result.is_ok());
        let issues = result.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, ValidationSeverity::Warning);
    }

    #[test]
    fn test_default_get_rules_empty() {
        let brain = TestBrain::new();
        assert!(brain.default_get_rules().is_empty());
    }

    #[test]
    fn test_default_generate_examples_empty() {
        let brain = TestBrain::new();
        assert!(brain.default_generate_examples(10).is_empty());
    }
}
