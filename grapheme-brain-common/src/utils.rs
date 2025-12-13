//! Utility types for domain detection and text normalization.

use serde::{Deserialize, Serialize};

/// Keyword-based capability detector for domain identification.
///
/// Many cognitive brains use simple keyword matching to determine if they
/// can process a given input. This struct provides a reusable implementation
/// of this pattern.
///
/// # Example
///
/// ```
/// use grapheme_brain_common::KeywordCapabilityDetector;
///
/// let detector = KeywordCapabilityDetector::new(vec![
///     "function", "class", "def", "return", "import"
/// ]);
///
/// assert!(detector.can_process("def hello(): return 42"));
/// assert!(!detector.can_process("The quick brown fox"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordCapabilityDetector {
    /// Keywords that indicate this domain can process the input
    keywords: Vec<String>,
    /// Whether to use case-insensitive matching
    case_insensitive: bool,
    /// Minimum number of keywords that must match
    min_matches: usize,
}

impl KeywordCapabilityDetector {
    /// Create a new detector with the given keywords
    pub fn new<I, S>(keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            keywords: keywords.into_iter().map(|s| s.into()).collect(),
            case_insensitive: true,
            min_matches: 1,
        }
    }

    /// Set whether to use case-insensitive matching (default: true)
    pub fn case_insensitive(mut self, value: bool) -> Self {
        self.case_insensitive = value;
        self
    }

    /// Set minimum number of keywords that must match (default: 1)
    pub fn min_matches(mut self, count: usize) -> Self {
        self.min_matches = count;
        self
    }

    /// Add additional keywords
    pub fn add_keywords<I, S>(mut self, keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.keywords.extend(keywords.into_iter().map(|s| s.into()));
        self
    }

    /// Check if the input can be processed by this domain
    pub fn can_process(&self, input: &str) -> bool {
        let input_normalized = if self.case_insensitive {
            input.to_lowercase()
        } else {
            input.to_string()
        };

        let match_count = self.keywords.iter().filter(|kw| {
            let kw_normalized = if self.case_insensitive {
                kw.to_lowercase()
            } else {
                kw.to_string()
            };
            input_normalized.contains(&kw_normalized)
        }).count();

        match_count >= self.min_matches
    }

    /// Count how many keywords match
    pub fn match_count(&self, input: &str) -> usize {
        let input_normalized = if self.case_insensitive {
            input.to_lowercase()
        } else {
            input.to_string()
        };

        self.keywords.iter().filter(|kw| {
            let kw_normalized = if self.case_insensitive {
                kw.to_lowercase()
            } else {
                kw.to_string()
            };
            input_normalized.contains(&kw_normalized)
        }).count()
    }

    /// Get all matching keywords
    pub fn matching_keywords(&self, input: &str) -> Vec<&str> {
        let input_normalized = if self.case_insensitive {
            input.to_lowercase()
        } else {
            input.to_string()
        };

        self.keywords.iter().filter(|kw| {
            let kw_normalized = if self.case_insensitive {
                kw.to_lowercase()
            } else {
                kw.to_string()
            };
            input_normalized.contains(&kw_normalized)
        }).map(|s| s.as_str()).collect()
    }

    /// Get the keywords
    pub fn keywords(&self) -> &[String] {
        &self.keywords
    }
}

/// Text normalizer for input preprocessing.
///
/// Applies a series of string replacements to normalize input text before
/// processing. This is commonly used to standardize notation, expand
/// abbreviations, and clean up whitespace.
///
/// # Example
///
/// ```
/// use grapheme_brain_common::TextNormalizer;
///
/// let normalizer = TextNormalizer::new()
///     .add_replacement("isn't", "is not")
///     .add_replacement("don't", "do not")
///     .trim_whitespace(true);
///
/// let normalized = normalizer.normalize("  I don't know  ");
/// assert_eq!(normalized, "I do not know");
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextNormalizer {
    /// Replacement pairs (pattern, replacement)
    replacements: Vec<(String, String)>,
    /// Whether to trim leading/trailing whitespace
    trim: bool,
    /// Whether to collapse multiple spaces into one
    collapse_spaces: bool,
    /// Whether to convert to lowercase
    lowercase: bool,
}

impl TextNormalizer {
    /// Create a new empty normalizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a replacement pattern
    pub fn add_replacement(mut self, pattern: impl Into<String>, replacement: impl Into<String>) -> Self {
        self.replacements.push((pattern.into(), replacement.into()));
        self
    }

    /// Add multiple replacement patterns at once
    pub fn add_replacements<I, S1, S2>(mut self, replacements: I) -> Self
    where
        I: IntoIterator<Item = (S1, S2)>,
        S1: Into<String>,
        S2: Into<String>,
    {
        for (pattern, replacement) in replacements {
            self.replacements.push((pattern.into(), replacement.into()));
        }
        self
    }

    /// Set whether to trim whitespace (default: false)
    pub fn trim_whitespace(mut self, value: bool) -> Self {
        self.trim = value;
        self
    }

    /// Set whether to collapse multiple spaces (default: false)
    pub fn collapse_spaces(mut self, value: bool) -> Self {
        self.collapse_spaces = value;
        self
    }

    /// Set whether to convert to lowercase (default: false)
    pub fn lowercase(mut self, value: bool) -> Self {
        self.lowercase = value;
        self
    }

    /// Normalize the input text
    pub fn normalize(&self, input: &str) -> String {
        let mut result = input.to_string();

        // Apply replacements
        for (pattern, replacement) in &self.replacements {
            result = result.replace(pattern.as_str(), replacement.as_str());
        }

        // Collapse spaces
        if self.collapse_spaces {
            while result.contains("  ") {
                result = result.replace("  ", " ");
            }
        }

        // Trim
        if self.trim {
            result = result.trim().to_string();
        }

        // Lowercase
        if self.lowercase {
            result = result.to_lowercase();
        }

        result
    }

    /// Check if normalization would change the input
    pub fn would_change(&self, input: &str) -> bool {
        self.normalize(input) != input
    }
}

/// Common math notation normalizer
pub fn math_normalizer() -> TextNormalizer {
    TextNormalizer::new()
        .add_replacement("×", "*")
        .add_replacement("÷", "/")
        .add_replacement("−", "-")
        .add_replacement("π", "pi")
        .add_replacement("√", "sqrt")
        .trim_whitespace(true)
        .collapse_spaces(true)
}

/// Common code normalizer
pub fn code_normalizer() -> TextNormalizer {
    TextNormalizer::new()
        .add_replacement("\t", "    ")  // Tabs to spaces
        .add_replacement("\r\n", "\n")  // Windows line endings
        .add_replacement("\r", "\n")    // Old Mac line endings
        .trim_whitespace(true)
}

/// Common legal text normalizer
pub fn legal_normalizer() -> TextNormalizer {
    TextNormalizer::new()
        .add_replacement("v.", "versus")
        .add_replacement("Inc.", "Incorporated")
        .add_replacement("Corp.", "Corporation")
        .add_replacement("Ltd.", "Limited")
        .trim_whitespace(true)
        .collapse_spaces(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_detector_basic() {
        let detector = KeywordCapabilityDetector::new(vec!["foo", "bar"]);

        assert!(detector.can_process("this has foo in it"));
        assert!(detector.can_process("this has bar in it"));
        assert!(!detector.can_process("no keywords here"));
    }

    #[test]
    fn test_keyword_detector_case_insensitive() {
        let detector = KeywordCapabilityDetector::new(vec!["Function"])
            .case_insensitive(true);

        assert!(detector.can_process("FUNCTION test"));
        assert!(detector.can_process("function test"));
    }

    #[test]
    fn test_keyword_detector_case_sensitive() {
        let detector = KeywordCapabilityDetector::new(vec!["Function"])
            .case_insensitive(false);

        assert!(detector.can_process("Function test"));
        assert!(!detector.can_process("function test"));
    }

    #[test]
    fn test_keyword_detector_min_matches() {
        let detector = KeywordCapabilityDetector::new(vec!["a", "b", "c"])
            .min_matches(2);

        assert!(!detector.can_process("only a"));
        assert!(detector.can_process("has a and b"));
        assert!(detector.can_process("has a, b, and c"));
    }

    #[test]
    fn test_keyword_detector_match_count() {
        let detector = KeywordCapabilityDetector::new(vec!["a", "b", "c"]);

        assert_eq!(detector.match_count("a b c"), 3);
        assert_eq!(detector.match_count("a b"), 2);
        assert_eq!(detector.match_count("x y z"), 0);
    }

    #[test]
    fn test_keyword_detector_matching_keywords() {
        let detector = KeywordCapabilityDetector::new(vec!["foo", "bar", "baz"]);

        let matches = detector.matching_keywords("foo and baz");
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&"foo"));
        assert!(matches.contains(&"baz"));
    }

    #[test]
    fn test_normalizer_replacements() {
        let normalizer = TextNormalizer::new()
            .add_replacement("old", "new");

        assert_eq!(normalizer.normalize("old text"), "new text");
    }

    #[test]
    fn test_normalizer_trim() {
        let normalizer = TextNormalizer::new().trim_whitespace(true);
        assert_eq!(normalizer.normalize("  hello  "), "hello");
    }

    #[test]
    fn test_normalizer_collapse_spaces() {
        let normalizer = TextNormalizer::new().collapse_spaces(true);
        assert_eq!(normalizer.normalize("hello    world"), "hello world");
    }

    #[test]
    fn test_normalizer_lowercase() {
        let normalizer = TextNormalizer::new().lowercase(true);
        assert_eq!(normalizer.normalize("HELLO World"), "hello world");
    }

    #[test]
    fn test_normalizer_combined() {
        let normalizer = TextNormalizer::new()
            .add_replacement("foo", "bar")
            .trim_whitespace(true)
            .collapse_spaces(true)
            .lowercase(true);

        // Note: lowercase is applied last, so "FOO" -> "foo" doesn't match the replacement "foo"->"bar"
        // The replacement happens first on the original case, then lowercase is applied
        assert_eq!(normalizer.normalize("  foo   BAZ  "), "bar baz");
        assert_eq!(normalizer.normalize("  FOO   BAZ  "), "foo baz"); // FOO stays as foo
    }

    #[test]
    fn test_math_normalizer() {
        let normalizer = math_normalizer();
        assert_eq!(normalizer.normalize(" 2 × 3 ÷ 4 "), "2 * 3 / 4");
    }

    #[test]
    fn test_would_change() {
        let normalizer = TextNormalizer::new()
            .add_replacement("a", "b");

        assert!(normalizer.would_change("a"));
        assert!(!normalizer.would_change("c"));
    }
}
