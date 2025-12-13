//! MATH Competition Dataset Loader
//!
//! Downloads and parses the MATH dataset (Hendrycks et al.)
//! for advanced mathematical reasoning training.
//!
//! Dataset: https://github.com/hendrycks/math
//! Format: JSON files per problem with LaTeX solutions

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// MATH problem categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathCategory {
    Algebra,
    CountingAndProbability,
    Geometry,
    IntermediateAlgebra,
    NumberTheory,
    Prealgebra,
    Precalculus,
}

impl MathCategory {
    pub fn from_dir_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "algebra" => Some(Self::Algebra),
            "counting_and_probability" => Some(Self::CountingAndProbability),
            "geometry" => Some(Self::Geometry),
            "intermediate_algebra" => Some(Self::IntermediateAlgebra),
            "number_theory" => Some(Self::NumberTheory),
            "prealgebra" => Some(Self::Prealgebra),
            "precalculus" => Some(Self::Precalculus),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Algebra => "algebra",
            Self::CountingAndProbability => "counting_and_probability",
            Self::Geometry => "geometry",
            Self::IntermediateAlgebra => "intermediate_algebra",
            Self::NumberTheory => "number_theory",
            Self::Prealgebra => "prealgebra",
            Self::Precalculus => "precalculus",
        }
    }
}

/// Raw MATH problem from dataset
#[derive(Debug, Clone, Deserialize)]
pub struct MathRaw {
    pub problem: String,
    #[serde(default)]
    pub level: String,
    #[serde(rename = "type", default)]
    pub problem_type: String,
    pub solution: String,
}

/// Processed MATH example for GRAPHEME training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathExample {
    pub id: String,
    pub category: String,
    pub level: u8,
    pub problem: String,
    pub solution: String,
    pub final_answer: String,
    pub latex_content: bool,
}

/// MATH dataset loader
pub struct MathLoader {
    data_dir: PathBuf,
}

impl MathLoader {
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
        }
    }

    /// Load MATH dataset from directory structure
    pub fn load(&self, split: &str) -> Result<Vec<MathExample>, Box<dyn std::error::Error>> {
        let split_dir = self.data_dir.join(split);

        if !split_dir.exists() {
            return Err(format!(
                "MATH {} directory not found: {:?}. Download from https://github.com/hendrycks/math",
                split, split_dir
            ).into());
        }

        let mut examples = Vec::new();
        let mut idx = 0;

        // Iterate over category directories
        for entry in fs::read_dir(&split_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let category_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");

                let _category = MathCategory::from_dir_name(category_name);

                // Load problems from this category
                for problem_file in fs::read_dir(&path)? {
                    let problem_file = problem_file?;
                    let problem_path = problem_file.path();

                    if problem_path.extension().map_or(false, |e| e == "json") {
                        match self.load_problem(&problem_path, idx, category_name) {
                            Ok(example) => {
                                examples.push(example);
                                idx += 1;
                            }
                            Err(e) => {
                                eprintln!("Warning: Failed to load {:?}: {}", problem_path, e);
                            }
                        }
                    }
                }
            }
        }

        Ok(examples)
    }

    /// Load a single problem from JSON file
    fn load_problem(
        &self,
        path: &Path,
        idx: usize,
        category: &str,
    ) -> Result<MathExample, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let raw: MathRaw = serde_json::from_reader(reader)?;

        // Parse level from string like "Level 1" to "Level 5"
        let level = raw.level
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<u8>()
            .unwrap_or(1);

        // Extract final answer (usually in \boxed{...} or at end of solution)
        let final_answer = self.extract_answer(&raw.solution);

        // Check if contains LaTeX
        let latex_content = raw.problem.contains('\\') || raw.solution.contains('\\');

        Ok(MathExample {
            id: format!("math_{:06}", idx),
            category: category.to_string(),
            level,
            problem: raw.problem,
            solution: raw.solution,
            final_answer,
            latex_content,
        })
    }

    /// Extract final answer from solution (typically in \boxed{})
    fn extract_answer(&self, solution: &str) -> String {
        // Look for \boxed{...}
        if let Some(start) = solution.rfind("\\boxed{") {
            let rest = &solution[start + 7..];
            let mut depth = 1;
            let mut end = 0;

            for (i, c) in rest.chars().enumerate() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = i;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if end > 0 {
                return rest[..end].to_string();
            }
        }

        // Fallback: try to extract last number or expression
        solution
            .lines()
            .rev()
            .find(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && (
                    trimmed.chars().any(|c| c.is_ascii_digit()) ||
                    trimmed.starts_with('$')
                )
            })
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| solution.lines().last().unwrap_or("").trim().to_string())
    }

    /// Convert to GRAPHEME training format (JSONL)
    pub fn to_grapheme_format<P: AsRef<Path>>(
        &self,
        examples: &[MathExample],
        output_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        for example in examples {
            let training_example = serde_json::json!({
                "id": example.id,
                "domain": "math",
                "input_type": "text",
                "input": {
                    "text": example.problem
                },
                "expected_output": example.final_answer,
                "metadata": {
                    "difficulty": example.level,
                    "category": example.category,
                    "tags": ["math", "competition", example.category.as_str()],
                    "latex": example.latex_content
                }
            });

            writeln!(writer, "{}", serde_json::to_string(&training_example)?)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Filter examples by category
    pub fn filter_by_category(&self, examples: &[MathExample], category: &str) -> Vec<MathExample> {
        examples
            .iter()
            .filter(|e| e.category.to_lowercase() == category.to_lowercase())
            .cloned()
            .collect()
    }

    /// Filter examples by difficulty level
    pub fn filter_by_level(&self, examples: &[MathExample], min_level: u8, max_level: u8) -> Vec<MathExample> {
        examples
            .iter()
            .filter(|e| e.level >= min_level && e.level <= max_level)
            .cloned()
            .collect()
    }

    /// Get dataset statistics
    pub fn stats(&self, examples: &[MathExample]) -> MathStats {
        let total = examples.len();

        let mut by_category: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut by_level: [usize; 6] = [0; 6]; // Levels 0-5

        let mut latex_count = 0;
        let mut total_problem_len = 0;
        let mut total_solution_len = 0;

        for example in examples {
            *by_category.entry(example.category.clone()).or_insert(0) += 1;

            let level_idx = (example.level as usize).min(5);
            by_level[level_idx] += 1;

            if example.latex_content {
                latex_count += 1;
            }

            total_problem_len += example.problem.len();
            total_solution_len += example.solution.len();
        }

        MathStats {
            total_examples: total,
            by_category,
            by_level,
            latex_percentage: if total > 0 { latex_count as f32 / total as f32 * 100.0 } else { 0.0 },
            avg_problem_length: if total > 0 { total_problem_len as f32 / total as f32 } else { 0.0 },
            avg_solution_length: if total > 0 { total_solution_len as f32 / total as f32 } else { 0.0 },
        }
    }
}

/// MATH dataset statistics
#[derive(Debug, Clone)]
pub struct MathStats {
    pub total_examples: usize,
    pub by_category: std::collections::HashMap<String, usize>,
    pub by_level: [usize; 6],
    pub latex_percentage: f32,
    pub avg_problem_length: f32,
    pub avg_solution_length: f32,
}

impl std::fmt::Display for MathStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MATH Competition Dataset Statistics:")?;
        writeln!(f, "  Total examples: {}", self.total_examples)?;
        writeln!(f, "  By category:")?;
        for (cat, count) in &self.by_category {
            writeln!(f, "    {}: {}", cat, count)?;
        }
        writeln!(f, "  By level:")?;
        for (level, count) in self.by_level.iter().enumerate() {
            if *count > 0 {
                writeln!(f, "    Level {}: {}", level, count)?;
            }
        }
        writeln!(f, "  LaTeX content: {:.1}%", self.latex_percentage)?;
        writeln!(f, "  Avg problem length: {:.1} chars", self.avg_problem_length)?;
        writeln!(f, "  Avg solution length: {:.1} chars", self.avg_solution_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_boxed_answer() {
        let loader = MathLoader::new("data/external/math");

        let solution = r"We have $x^2 + 2x + 1 = 0$. Solving gives $x = -1$. Thus, the answer is $\boxed{-1}$.";
        assert_eq!(loader.extract_answer(solution), "-1");

        let solution2 = r"The answer is $\boxed{\frac{1}{2}}$ after simplification.";
        assert_eq!(loader.extract_answer(solution2), r"\frac{1}{2}");
    }

    #[test]
    fn test_extract_answer_no_boxed() {
        let loader = MathLoader::new("data/external/math");

        let solution = "The answer is 42.";
        assert_eq!(loader.extract_answer(solution), "The answer is 42.");
    }

    #[test]
    fn test_category_from_dir_name() {
        assert_eq!(MathCategory::from_dir_name("algebra"), Some(MathCategory::Algebra));
        assert_eq!(MathCategory::from_dir_name("Algebra"), Some(MathCategory::Algebra));
        assert_eq!(MathCategory::from_dir_name("number_theory"), Some(MathCategory::NumberTheory));
        assert_eq!(MathCategory::from_dir_name("unknown"), None);
    }
}
