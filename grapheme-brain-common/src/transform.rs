//! Text transformation rules for domain-specific brain implementations.

use serde::{Deserialize, Serialize};

/// A reusable text-based transformation rule.
///
/// Many cognitive brain transformations follow the same pattern:
/// 1. Get text representation of graph
/// 2. Apply string replacements
/// 3. Parse back to graph
///
/// This struct encapsulates this pattern to eliminate duplication across
/// brain implementations.
///
/// # Example
///
/// ```
/// use grapheme_brain_common::TextTransformRule;
///
/// // Create a rule for mathematical simplification
/// let rule = TextTransformRule::new("zero_addition")
///     .add_replacement("+ 0", "")
///     .add_replacement("0 +", "");
///
/// let input = "x + 0";
/// if let Some(output) = rule.apply(input) {
///     assert_eq!(output, "x ");
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextTransformRule {
    /// Rule identifier/name
    pub name: String,
    /// Description of what this rule does
    pub description: Option<String>,
    /// List of (pattern, replacement) pairs
    pub replacements: Vec<(String, String)>,
}

impl TextTransformRule {
    /// Create a new transformation rule with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            replacements: Vec::new(),
        }
    }

    /// Set the description for this rule
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
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

    /// Apply the transformation rule to input text.
    ///
    /// Returns `Some(transformed)` if any replacement was made,
    /// `None` if the text was unchanged.
    pub fn apply(&self, input: &str) -> Option<String> {
        let mut result = input.to_string();
        let mut changed = false;

        for (pattern, replacement) in &self.replacements {
            if result.contains(pattern.as_str()) {
                result = result.replace(pattern.as_str(), replacement.as_str());
                changed = true;
            }
        }

        if changed {
            Some(result)
        } else {
            None
        }
    }

    /// Apply the transformation rule, returning the original if unchanged
    pub fn apply_or_original<'a>(&self, input: &'a str) -> std::borrow::Cow<'a, str> {
        match self.apply(input) {
            Some(transformed) => std::borrow::Cow::Owned(transformed),
            None => std::borrow::Cow::Borrowed(input),
        }
    }

    /// Check if this rule would modify the input
    pub fn matches(&self, input: &str) -> bool {
        self.replacements.iter().any(|(pattern, _)| input.contains(pattern.as_str()))
    }

    /// Get the rule name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the rule description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// A collection of transformation rules that can be applied in sequence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransformRuleSet {
    rules: Vec<TextTransformRule>,
}

impl TransformRuleSet {
    /// Create a new empty rule set
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule to the set
    pub fn add_rule(mut self, rule: TextTransformRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Get rules by index
    pub fn get(&self, index: usize) -> Option<&TextTransformRule> {
        self.rules.get(index)
    }

    /// Get all rules
    pub fn rules(&self) -> &[TextTransformRule] {
        &self.rules
    }

    /// Number of rules in the set
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Apply all rules in sequence, returning the final result
    pub fn apply_all(&self, input: &str) -> String {
        let mut result = input.to_string();
        for rule in &self.rules {
            if let Some(transformed) = rule.apply(&result) {
                result = transformed;
            }
        }
        result
    }

    /// Apply rules until no more changes occur (fixed point)
    pub fn apply_until_fixed(&self, input: &str, max_iterations: usize) -> String {
        let mut result = input.to_string();
        for _ in 0..max_iterations {
            let new_result = self.apply_all(&result);
            if new_result == result {
                break;
            }
            result = new_result;
        }
        result
    }

    /// Find all rules that match the input
    pub fn matching_rules(&self, input: &str) -> Vec<usize> {
        self.rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| rule.matches(input))
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_replacement() {
        let rule = TextTransformRule::new("test")
            .add_replacement("foo", "bar");

        assert_eq!(rule.apply("hello foo world"), Some("hello bar world".into()));
        assert_eq!(rule.apply("no match"), None);
    }

    #[test]
    fn test_multiple_replacements() {
        let rule = TextTransformRule::new("test")
            .add_replacement("a", "1")
            .add_replacement("b", "2");

        assert_eq!(rule.apply("abc"), Some("12c".into()));
    }

    #[test]
    fn test_math_rule() {
        let rule = TextTransformRule::new("zero_addition")
            .with_description("Remove addition of zero")
            .add_replacement(" + 0", "")
            .add_replacement("0 + ", "");

        assert_eq!(rule.apply("x + 0"), Some("x".into()));
        assert_eq!(rule.apply("0 + y"), Some("y".into()));
        assert_eq!(rule.apply("x + y"), None);
    }

    #[test]
    fn test_matches() {
        let rule = TextTransformRule::new("test")
            .add_replacement("pattern", "");

        assert!(rule.matches("contains pattern here"));
        assert!(!rule.matches("no match"));
    }

    #[test]
    fn test_apply_or_original() {
        let rule = TextTransformRule::new("test")
            .add_replacement("old", "new");

        let result1 = rule.apply_or_original("old text");
        assert_eq!(result1.as_ref(), "new text");

        let result2 = rule.apply_or_original("unchanged");
        assert_eq!(result2.as_ref(), "unchanged");
    }

    #[test]
    fn test_rule_set() {
        let rules = TransformRuleSet::new()
            .add_rule(TextTransformRule::new("r1").add_replacement("a", "b"))
            .add_rule(TextTransformRule::new("r2").add_replacement("b", "c"));

        assert_eq!(rules.len(), 2);
        assert_eq!(rules.apply_all("a"), "c"); // a -> b -> c
    }

    #[test]
    fn test_apply_until_fixed() {
        let rules = TransformRuleSet::new()
            .add_rule(TextTransformRule::new("double").add_replacement("xx", "x"));

        assert_eq!(rules.apply_until_fixed("xxxx", 10), "x");
    }

    #[test]
    fn test_matching_rules() {
        let rules = TransformRuleSet::new()
            .add_rule(TextTransformRule::new("r1").add_replacement("foo", ""))
            .add_rule(TextTransformRule::new("r2").add_replacement("bar", ""))
            .add_rule(TextTransformRule::new("r3").add_replacement("baz", ""));

        let matches = rules.matching_rules("foo and bar");
        assert_eq!(matches, vec![0, 1]);
    }
}
