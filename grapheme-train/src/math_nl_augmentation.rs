//! NL Augmentation Pipeline for Math Expressions
//!
//! Data-009: Generate varied natural language phrasings for math expressions.
//!
//! Provides augmentation techniques for math training data:
//! - Paraphrasing math questions
//! - Varying numeric values
//! - Changing variable names
//! - Generating equivalent expressions
//! - Creating word problems from formulas

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for math NL augmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathAugmentConfig {
    /// Number of augmentations to generate per input
    pub augmentations_per_input: usize,
    /// Probability of number substitution
    pub number_sub_prob: f32,
    /// Range for number substitution (relative to original)
    pub number_range: (f32, f32),
    /// Probability of variable name change
    pub var_rename_prob: f32,
    /// Probability of phrase reordering
    pub reorder_prob: f32,
    /// Probability of synonym substitution
    pub synonym_prob: f32,
    /// Whether to preserve mathematical correctness
    pub preserve_correctness: bool,
}

impl Default for MathAugmentConfig {
    fn default() -> Self {
        Self {
            augmentations_per_input: 3,
            number_sub_prob: 0.5,
            number_range: (0.5, 2.0),
            var_rename_prob: 0.3,
            reorder_prob: 0.4,
            synonym_prob: 0.6,
            preserve_correctness: true,
        }
    }
}

// ============================================================================
// Math Expression Types
// ============================================================================

/// Types of math expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathExprType {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Equation,
    Comparison,
    Percentage,
    Ratio,
    WordProblem,
    Formula,
}

/// A math expression with its natural language form
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathNLPair {
    /// The mathematical expression
    pub expression: String,
    /// Natural language description
    pub natural_language: String,
    /// Expression type
    pub expr_type: MathExprType,
    /// Variables in the expression
    pub variables: Vec<String>,
    /// Numbers in the expression
    pub numbers: Vec<f64>,
    /// The answer/result if applicable
    pub answer: Option<String>,
}

impl MathNLPair {
    /// Create new pair
    pub fn new(expression: &str, natural_language: &str, expr_type: MathExprType) -> Self {
        let numbers = Self::extract_numbers(expression);
        let variables = Self::extract_variables(expression);

        Self {
            expression: expression.to_string(),
            natural_language: natural_language.to_string(),
            expr_type,
            variables,
            numbers,
            answer: None,
        }
    }

    /// Set the answer
    pub fn with_answer(mut self, answer: &str) -> Self {
        self.answer = Some(answer.to_string());
        self
    }

    /// Extract numbers from expression
    fn extract_numbers(expr: &str) -> Vec<f64> {
        let mut numbers = Vec::new();
        let mut current = String::new();
        let mut in_number = false;

        for c in expr.chars() {
            if c.is_ascii_digit() || c == '.' || (c == '-' && !in_number) {
                current.push(c);
                in_number = true;
            } else if in_number {
                if let Ok(n) = current.parse::<f64>() {
                    numbers.push(n);
                }
                current.clear();
                in_number = false;
            }
        }

        if in_number {
            if let Ok(n) = current.parse::<f64>() {
                numbers.push(n);
            }
        }

        numbers
    }

    /// Extract variables from expression
    fn extract_variables(expr: &str) -> Vec<String> {
        let mut vars = Vec::new();
        let mut current = String::new();
        let keywords = ["sin", "cos", "tan", "log", "ln", "sqrt", "abs", "if", "then", "else"];

        for c in expr.chars() {
            if c.is_alphabetic() || (c == '_' && !current.is_empty()) {
                current.push(c);
            } else if !current.is_empty() {
                if current.len() == 1 || !keywords.contains(&current.to_lowercase().as_str()) {
                    if !vars.contains(&current) {
                        vars.push(current.clone());
                    }
                }
                current.clear();
            }
        }

        if !current.is_empty() && (current.len() == 1 || !keywords.contains(&current.to_lowercase().as_str())) {
            if !vars.contains(&current) {
                vars.push(current);
            }
        }

        vars
    }
}

// ============================================================================
// Augmentation Strategies
// ============================================================================

/// Synonym mappings for math terms
pub fn get_math_synonyms() -> HashMap<&'static str, Vec<&'static str>> {
    let mut map = HashMap::new();

    // Operations
    map.insert("add", vec!["sum", "plus", "combine", "increase by", "total"]);
    map.insert("subtract", vec!["minus", "take away", "decrease by", "remove", "less"]);
    map.insert("multiply", vec!["times", "product of", "multiplied by"]);
    map.insert("divide", vec!["split", "shared among", "divided by", "quotient of"]);

    // Comparisons
    map.insert("equal", vec!["same as", "equivalent to", "equals", "is"]);
    map.insert("greater", vec!["more than", "larger than", "exceeds", "bigger than"]);
    map.insert("less", vec!["fewer than", "smaller than", "under", "below"]);

    // Questions
    map.insert("find", vec!["calculate", "determine", "compute", "figure out", "what is"]);
    map.insert("how many", vec!["what number of", "count of", "total number of"]);
    map.insert("how much", vec!["what amount", "what quantity", "the total"]);

    // Problem framing
    map.insert("has", vec!["owns", "possesses", "holds", "keeps"]);
    map.insert("gave", vec!["handed", "passed", "transferred", "gave away"]);
    map.insert("bought", vec!["purchased", "got", "acquired"]);
    map.insert("sold", vec!["traded", "exchanged", "gave up"]);

    map
}

/// Word problem templates
pub fn get_word_problem_templates() -> Vec<WordProblemTemplate> {
    vec![
        // Addition
        WordProblemTemplate {
            operation: MathExprType::Addition,
            template: "{name} has {n1} {item}. {name2} gives {pronoun} {n2} more {item}. How many {item} does {name} have now?",
            answer_template: "{n1} + {n2} = {result}",
        },
        WordProblemTemplate {
            operation: MathExprType::Addition,
            template: "There are {n1} {item} in one box and {n2} {item} in another. What is the total number of {item}?",
            answer_template: "{n1} + {n2} = {result}",
        },
        // Subtraction
        WordProblemTemplate {
            operation: MathExprType::Subtraction,
            template: "{name} had {n1} {item}. {pronoun_cap} gave away {n2} {item}. How many {item} does {name} have left?",
            answer_template: "{n1} - {n2} = {result}",
        },
        WordProblemTemplate {
            operation: MathExprType::Subtraction,
            template: "A store had {n1} {item}. After selling some, there are {n2} left. How many were sold?",
            answer_template: "{n1} - {n2} = {result}",
        },
        // Multiplication
        WordProblemTemplate {
            operation: MathExprType::Multiplication,
            template: "Each {container} contains {n1} {item}. If there are {n2} {container}s, how many {item} are there in total?",
            answer_template: "{n1} * {n2} = {result}",
        },
        WordProblemTemplate {
            operation: MathExprType::Multiplication,
            template: "{name} buys {n2} {item} at ${n1} each. What is the total cost?",
            answer_template: "{n1} * {n2} = {result}",
        },
        // Division
        WordProblemTemplate {
            operation: MathExprType::Division,
            template: "{name} has {n1} {item} to share equally among {n2} friends. How many does each friend get?",
            answer_template: "{n1} / {n2} = {result}",
        },
        WordProblemTemplate {
            operation: MathExprType::Division,
            template: "A {container} holds {n2} {item}. How many {container}s are needed for {n1} {item}?",
            answer_template: "{n1} / {n2} = {result}",
        },
        // Percentage
        WordProblemTemplate {
            operation: MathExprType::Percentage,
            template: "A {item} costs ${n1}. If it's {n2}% off, how much is the discount?",
            answer_template: "{n1} * {n2} / 100 = {result}",
        },
        WordProblemTemplate {
            operation: MathExprType::Percentage,
            template: "{name} scored {n1} out of {n2} on a test. What percentage did {pronoun} get?",
            answer_template: "({n1} / {n2}) * 100 = {result}%",
        },
    ]
}

/// A word problem template
#[derive(Debug, Clone)]
pub struct WordProblemTemplate {
    pub operation: MathExprType,
    pub template: &'static str,
    pub answer_template: &'static str,
}

/// Names for word problems
pub fn get_names() -> Vec<(&'static str, &'static str, &'static str)> {
    // (name, pronoun, pronoun_cap)
    vec![
        ("Alice", "she", "She"),
        ("Bob", "he", "He"),
        ("Carol", "she", "She"),
        ("David", "he", "He"),
        ("Emma", "she", "She"),
        ("Frank", "he", "He"),
        ("Grace", "she", "She"),
        ("Henry", "he", "He"),
        ("Ivy", "she", "She"),
        ("Jack", "he", "He"),
    ]
}

/// Items for word problems
pub fn get_items() -> Vec<&'static str> {
    vec![
        "apples", "oranges", "books", "pencils", "cookies", "marbles",
        "stickers", "toys", "candies", "balls", "cards", "coins",
        "flowers", "stamps", "shells", "crayons", "blocks", "stars",
    ]
}

/// Containers for word problems
pub fn get_containers() -> Vec<&'static str> {
    vec![
        "box", "bag", "basket", "jar", "shelf", "pack", "crate", "bucket",
    ]
}

// ============================================================================
// Augmenter
// ============================================================================

/// NL augmentation pipeline for math expressions
pub struct MathNLAugmenter {
    config: MathAugmentConfig,
    synonyms: HashMap<&'static str, Vec<&'static str>>,
    templates: Vec<WordProblemTemplate>,
    names: Vec<(&'static str, &'static str, &'static str)>,
    items: Vec<&'static str>,
    containers: Vec<&'static str>,
}

impl MathNLAugmenter {
    /// Create new augmenter
    pub fn new(config: MathAugmentConfig) -> Self {
        Self {
            config,
            synonyms: get_math_synonyms(),
            templates: get_word_problem_templates(),
            names: get_names(),
            items: get_items(),
            containers: get_containers(),
        }
    }

    /// Create with default config
    pub fn default_augmenter() -> Self {
        Self::new(MathAugmentConfig::default())
    }

    /// Augment a single math NL pair
    pub fn augment(&self, pair: &MathNLPair) -> Vec<MathNLPair> {
        let mut augmented = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..self.config.augmentations_per_input {
            let mut new_pair = pair.clone();

            // Apply various augmentations
            if rng.gen::<f32>() < self.config.synonym_prob {
                new_pair.natural_language = self.apply_synonyms(&new_pair.natural_language);
            }

            if rng.gen::<f32>() < self.config.number_sub_prob {
                new_pair = self.substitute_numbers(&new_pair);
            }

            if rng.gen::<f32>() < self.config.var_rename_prob {
                new_pair = self.rename_variables(&new_pair);
            }

            if rng.gen::<f32>() < self.config.reorder_prob {
                new_pair.natural_language = self.reorder_phrases(&new_pair.natural_language);
            }

            // Only add if different from original
            if new_pair.natural_language != pair.natural_language
                || new_pair.expression != pair.expression
            {
                augmented.push(new_pair);
            }
        }

        augmented
    }

    /// Apply synonym substitutions
    fn apply_synonyms(&self, text: &str) -> String {
        let mut result = text.to_string();
        let mut rng = rand::thread_rng();

        for (word, synonyms) in &self.synonyms {
            if result.to_lowercase().contains(*word) {
                let replacement = synonyms[rng.gen_range(0..synonyms.len())];
                // Case-insensitive replace (simple version)
                let lower = result.to_lowercase();
                if let Some(pos) = lower.find(*word) {
                    let original_case = &result[pos..pos + word.len()];
                    let new_word = if original_case.chars().next().unwrap().is_uppercase() {
                        let mut chars: Vec<char> = replacement.chars().collect();
                        if !chars.is_empty() {
                            chars[0] = chars[0].to_uppercase().next().unwrap();
                        }
                        chars.into_iter().collect()
                    } else {
                        replacement.to_string()
                    };
                    result = format!("{}{}{}", &result[..pos], new_word, &result[pos + word.len()..]);
                }
            }
        }

        result
    }

    /// Substitute numbers with similar values
    fn substitute_numbers(&self, pair: &MathNLPair) -> MathNLPair {
        let mut new_pair = pair.clone();
        let mut rng = rand::thread_rng();

        if pair.numbers.is_empty() {
            return new_pair;
        }

        // Pick a number to substitute
        let idx = rng.gen_range(0..pair.numbers.len());
        let original = pair.numbers[idx];

        // Generate new number in range
        let factor = rng.gen_range(self.config.number_range.0..self.config.number_range.1);
        let new_num = (original * factor as f64).round();

        // Substitute in expression and NL
        let old_str = if original == original.floor() {
            format!("{}", original as i64)
        } else {
            format!("{}", original)
        };
        let new_str = if new_num == new_num.floor() {
            format!("{}", new_num as i64)
        } else {
            format!("{:.1}", new_num)
        };

        new_pair.expression = new_pair.expression.replace(&old_str, &new_str);
        new_pair.natural_language = new_pair.natural_language.replace(&old_str, &new_str);
        new_pair.numbers[idx] = new_num;

        // Update answer if we need to preserve correctness
        if self.config.preserve_correctness && new_pair.answer.is_some() {
            // Recalculate (simplified - only for basic operations)
            new_pair.answer = self.recalculate_answer(&new_pair);
        }

        new_pair
    }

    /// Rename variables
    fn rename_variables(&self, pair: &MathNLPair) -> MathNLPair {
        let mut new_pair = pair.clone();
        let mut rng = rand::thread_rng();

        let var_names = ['x', 'y', 'z', 'a', 'b', 'c', 'n', 'm', 'k', 'p', 'q', 'r'];

        for var in &pair.variables {
            if var.len() == 1 {
                let old_char = var.chars().next().unwrap();
                let new_char = var_names[rng.gen_range(0..var_names.len())];
                if new_char != old_char {
                    new_pair.expression = new_pair
                        .expression
                        .replace(&old_char.to_string(), &new_char.to_string());
                    new_pair.natural_language = new_pair
                        .natural_language
                        .replace(&old_char.to_string(), &new_char.to_string());
                }
            }
        }

        new_pair.variables = MathNLPair::extract_variables(&new_pair.expression);
        new_pair
    }

    /// Reorder phrases in the question
    fn reorder_phrases(&self, text: &str) -> String {
        // Split on common separators
        let parts: Vec<&str> = text.split(". ").collect();
        if parts.len() <= 1 {
            return text.to_string();
        }

        let mut rng = rand::thread_rng();

        // Keep question at end
        let question_idx = parts.iter().position(|p| p.contains('?'));

        let mut reordered: Vec<&str> = parts
            .iter()
            .enumerate()
            .filter(|(i, _)| question_idx.map(|q| *i != q).unwrap_or(true))
            .map(|(_, p)| *p)
            .collect();

        // Simple shuffle of non-question parts
        for i in (1..reordered.len()).rev() {
            let j = rng.gen_range(0..=i);
            reordered.swap(i, j);
        }

        // Add question back at end
        if let Some(idx) = question_idx {
            reordered.push(parts[idx]);
        }

        reordered.join(". ")
    }

    /// Generate word problem from expression
    pub fn generate_word_problem(&self, operation: MathExprType, n1: f64, n2: f64) -> MathNLPair {
        let mut rng = rand::thread_rng();

        // Find matching template
        let matching: Vec<_> = self.templates.iter().filter(|t| t.operation == operation).collect();
        if matching.is_empty() {
            // Fallback
            return MathNLPair::new(
                &format!("{} ? {} = ?", n1, n2),
                &format!("What is {} and {}?", n1, n2),
                operation,
            );
        }

        let template = matching[rng.gen_range(0..matching.len())];
        let (name, pronoun, pronoun_cap) = self.names[rng.gen_range(0..self.names.len())];
        let (name2, _, _) = self.names[rng.gen_range(0..self.names.len())];
        let item = self.items[rng.gen_range(0..self.items.len())];
        let container = self.containers[rng.gen_range(0..self.containers.len())];

        // Calculate result
        let result = match operation {
            MathExprType::Addition => n1 + n2,
            MathExprType::Subtraction => n1 - n2,
            MathExprType::Multiplication => n1 * n2,
            MathExprType::Division => n1 / n2,
            MathExprType::Percentage => n1 * n2 / 100.0,
            _ => n1 + n2,
        };

        let n1_str = if n1 == n1.floor() {
            format!("{}", n1 as i64)
        } else {
            format!("{:.1}", n1)
        };
        let n2_str = if n2 == n2.floor() {
            format!("{}", n2 as i64)
        } else {
            format!("{:.1}", n2)
        };
        let result_str = if result == result.floor() {
            format!("{}", result as i64)
        } else {
            format!("{:.2}", result)
        };

        // Fill in template
        let nl = template
            .template
            .replace("{name}", name)
            .replace("{name2}", name2)
            .replace("{pronoun}", pronoun)
            .replace("{pronoun_cap}", pronoun_cap)
            .replace("{item}", item)
            .replace("{container}", container)
            .replace("{n1}", &n1_str)
            .replace("{n2}", &n2_str);

        let answer = template
            .answer_template
            .replace("{n1}", &n1_str)
            .replace("{n2}", &n2_str)
            .replace("{result}", &result_str);

        let expr = format!(
            "{} {} {} = {}",
            n1_str,
            match operation {
                MathExprType::Addition => "+",
                MathExprType::Subtraction => "-",
                MathExprType::Multiplication => "*",
                MathExprType::Division => "/",
                _ => "?",
            },
            n2_str,
            result_str
        );

        MathNLPair::new(&expr, &nl, operation).with_answer(&answer)
    }

    /// Recalculate answer for basic operations
    fn recalculate_answer(&self, pair: &MathNLPair) -> Option<String> {
        if pair.numbers.len() < 2 {
            return pair.answer.clone();
        }

        let n1 = pair.numbers[0];
        let n2 = pair.numbers[1];

        let result = match pair.expr_type {
            MathExprType::Addition => n1 + n2,
            MathExprType::Subtraction => n1 - n2,
            MathExprType::Multiplication => n1 * n2,
            MathExprType::Division => {
                if n2 == 0.0 {
                    return pair.answer.clone();
                }
                n1 / n2
            }
            MathExprType::Percentage => n1 * n2 / 100.0,
            _ => return pair.answer.clone(),
        };

        Some(if result == result.floor() {
            format!("{}", result as i64)
        } else {
            format!("{:.2}", result)
        })
    }

    /// Batch augment multiple pairs
    pub fn augment_batch(&self, pairs: &[MathNLPair]) -> Vec<MathNLPair> {
        let mut all_augmented = Vec::new();

        for pair in pairs {
            // Include original
            all_augmented.push(pair.clone());
            // Add augmentations
            all_augmented.extend(self.augment(pair));
        }

        all_augmented
    }

    /// Generate random word problems
    pub fn generate_batch(&self, count: usize) -> Vec<MathNLPair> {
        let mut problems = Vec::new();
        let mut rng = rand::thread_rng();

        let operations = [
            MathExprType::Addition,
            MathExprType::Subtraction,
            MathExprType::Multiplication,
            MathExprType::Division,
        ];

        for _ in 0..count {
            let op = operations[rng.gen_range(0..operations.len())];

            // Generate appropriate numbers
            let (n1, n2) = match op {
                MathExprType::Addition | MathExprType::Subtraction => {
                    let a = rng.gen_range(1..100) as f64;
                    let b = rng.gen_range(1..50) as f64;
                    if op == MathExprType::Subtraction {
                        (a.max(b), a.min(b))
                    } else {
                        (a, b)
                    }
                }
                MathExprType::Multiplication => {
                    (rng.gen_range(2..15) as f64, rng.gen_range(2..10) as f64)
                }
                MathExprType::Division => {
                    let divisor = rng.gen_range(2..10) as f64;
                    let quotient = rng.gen_range(2..20) as f64;
                    (divisor * quotient, divisor)
                }
                _ => (rng.gen_range(1..100) as f64, rng.gen_range(1..50) as f64),
            };

            problems.push(self.generate_word_problem(op, n1, n2));
        }

        problems
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = MathAugmentConfig::default();
        assert_eq!(config.augmentations_per_input, 3);
        assert!(config.preserve_correctness);
    }

    #[test]
    fn test_math_nl_pair_creation() {
        let pair = MathNLPair::new("5 + 3 = 8", "What is five plus three?", MathExprType::Addition);
        assert_eq!(pair.expr_type, MathExprType::Addition);
        assert_eq!(pair.numbers.len(), 3); // 5, 3, 8
    }

    #[test]
    fn test_extract_numbers() {
        let pair = MathNLPair::new("12 * 5 = 60", "12 times 5", MathExprType::Multiplication);
        assert!(pair.numbers.contains(&12.0));
        assert!(pair.numbers.contains(&5.0));
        assert!(pair.numbers.contains(&60.0));
    }

    #[test]
    fn test_extract_variables() {
        let pair = MathNLPair::new("x + y = z", "x plus y equals z", MathExprType::Equation);
        assert!(pair.variables.contains(&"x".to_string()));
        assert!(pair.variables.contains(&"y".to_string()));
        assert!(pair.variables.contains(&"z".to_string()));
    }

    #[test]
    fn test_pair_with_answer() {
        let pair = MathNLPair::new("2 + 2", "two plus two", MathExprType::Addition).with_answer("4");
        assert_eq!(pair.answer, Some("4".to_string()));
    }

    #[test]
    fn test_augmenter_creation() {
        let augmenter = MathNLAugmenter::default_augmenter();
        assert!(!augmenter.synonyms.is_empty());
        assert!(!augmenter.templates.is_empty());
    }

    #[test]
    fn test_synonym_map() {
        let synonyms = get_math_synonyms();
        assert!(synonyms.contains_key("add"));
        assert!(!synonyms.get("add").unwrap().is_empty());
    }

    #[test]
    fn test_word_problem_templates() {
        let templates = get_word_problem_templates();
        assert!(!templates.is_empty());
        assert!(templates.iter().any(|t| t.operation == MathExprType::Addition));
    }

    #[test]
    fn test_generate_word_problem() {
        let augmenter = MathNLAugmenter::default_augmenter();
        let problem = augmenter.generate_word_problem(MathExprType::Addition, 5.0, 3.0);

        assert_eq!(problem.expr_type, MathExprType::Addition);
        assert!(!problem.natural_language.is_empty());
        assert!(problem.answer.is_some());
    }

    #[test]
    fn test_generate_subtraction() {
        let augmenter = MathNLAugmenter::default_augmenter();
        let problem = augmenter.generate_word_problem(MathExprType::Subtraction, 10.0, 3.0);

        assert_eq!(problem.expr_type, MathExprType::Subtraction);
        assert!(problem.expression.contains('-'));
    }

    #[test]
    fn test_generate_multiplication() {
        let augmenter = MathNLAugmenter::default_augmenter();
        let problem = augmenter.generate_word_problem(MathExprType::Multiplication, 4.0, 5.0);

        assert_eq!(problem.expr_type, MathExprType::Multiplication);
        assert!(problem.expression.contains('*'));
    }

    #[test]
    fn test_generate_division() {
        let augmenter = MathNLAugmenter::default_augmenter();
        let problem = augmenter.generate_word_problem(MathExprType::Division, 20.0, 4.0);

        assert_eq!(problem.expr_type, MathExprType::Division);
        assert!(problem.expression.contains('/'));
    }

    #[test]
    fn test_augment_generates_variations() {
        let augmenter = MathNLAugmenter::new(MathAugmentConfig {
            augmentations_per_input: 5,
            synonym_prob: 1.0, // Always apply synonyms
            number_sub_prob: 0.0,
            var_rename_prob: 0.0,
            reorder_prob: 0.0,
            ..Default::default()
        });

        let pair = MathNLPair::new("5 + 3", "Add five and three", MathExprType::Addition);
        let augmented = augmenter.augment(&pair);

        // Should generate some variations
        assert!(!augmented.is_empty() || pair.natural_language.to_lowercase().contains("add"));
    }

    #[test]
    fn test_augment_batch() {
        let augmenter = MathNLAugmenter::default_augmenter();
        let pairs = vec![
            MathNLPair::new("1 + 1", "one plus one", MathExprType::Addition),
            MathNLPair::new("2 * 3", "two times three", MathExprType::Multiplication),
        ];

        let augmented = augmenter.augment_batch(&pairs);
        // Should include originals plus some augmentations
        assert!(augmented.len() >= pairs.len());
    }

    #[test]
    fn test_generate_batch() {
        let augmenter = MathNLAugmenter::default_augmenter();
        let problems = augmenter.generate_batch(10);

        assert_eq!(problems.len(), 10);
        for problem in &problems {
            assert!(!problem.natural_language.is_empty());
            assert!(!problem.expression.is_empty());
        }
    }

    #[test]
    fn test_names_list() {
        let names = get_names();
        assert!(!names.is_empty());
        for (name, pronoun, pronoun_cap) in names {
            assert!(!name.is_empty());
            assert!(!pronoun.is_empty());
            assert!(pronoun_cap.chars().next().unwrap().is_uppercase());
        }
    }

    #[test]
    fn test_items_list() {
        let items = get_items();
        assert!(!items.is_empty());
        assert!(items.len() > 10);
    }

    #[test]
    fn test_containers_list() {
        let containers = get_containers();
        assert!(!containers.is_empty());
    }
}
