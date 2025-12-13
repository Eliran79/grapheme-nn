//! GSM8K Dataset Loader
//!
//! Downloads and parses the GSM8K (Grade School Math 8K) dataset
//! for natural language to math training.
//!
//! Dataset: https://github.com/openai/grade-school-math
//! Format: JSONL with question/answer pairs

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// GSM8K example from the original dataset
#[derive(Debug, Clone, Deserialize)]
pub struct Gsm8kRaw {
    pub question: String,
    pub answer: String,
}

/// Processed GSM8K example for GRAPHEME training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gsm8kExample {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub final_answer: String,
    pub reasoning_steps: Vec<String>,
}

/// GSM8K dataset loader
pub struct Gsm8kLoader {
    data_dir: PathBuf,
}

impl Gsm8kLoader {
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
        }
    }

    /// Load GSM8K from JSONL file
    pub fn load(&self, split: &str) -> Result<Vec<Gsm8kExample>, Box<dyn std::error::Error>> {
        let path = self.data_dir.join(format!("{}.jsonl", split));

        if !path.exists() {
            return Err(format!("GSM8K file not found: {:?}. Download from https://github.com/openai/grade-school-math", path).into());
        }

        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut examples = Vec::new();

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let raw: Gsm8kRaw = serde_json::from_str(&line)?;
            let processed = self.process_example(idx, raw);
            examples.push(processed);
        }

        Ok(examples)
    }

    /// Process raw GSM8K example into training format
    fn process_example(&self, idx: usize, raw: Gsm8kRaw) -> Gsm8kExample {
        // Extract final answer (usually after "####")
        let (reasoning, final_answer) = if let Some(pos) = raw.answer.find("####") {
            let reasoning = raw.answer[..pos].trim().to_string();
            let answer = raw.answer[pos + 4..].trim().to_string();
            (reasoning, answer)
        } else {
            (raw.answer.clone(), raw.answer.clone())
        };

        // Split reasoning into steps (usually separated by newlines)
        let reasoning_steps: Vec<String> = reasoning
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        Gsm8kExample {
            id: format!("gsm8k_{:06}", idx),
            question: raw.question,
            answer: raw.answer,
            final_answer,
            reasoning_steps,
        }
    }

    /// Convert to GRAPHEME training format (JSONL)
    pub fn to_grapheme_format<P: AsRef<Path>>(
        &self,
        examples: &[Gsm8kExample],
        output_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        for example in examples {
            // Convert to simplified format for math brain training
            let training_example = serde_json::json!({
                "id": example.id,
                "domain": "math",
                "input_type": "text",
                "input": {
                    "text": example.question
                },
                "expected_output": example.final_answer,
                "metadata": {
                    "difficulty": example.reasoning_steps.len().min(5) as u8,
                    "tags": ["gsm8k", "word_problem", "nl_math"],
                    "reasoning_steps": example.reasoning_steps.len()
                }
            });

            writeln!(writer, "{}", serde_json::to_string(&training_example)?)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Get dataset statistics
    pub fn stats(&self, examples: &[Gsm8kExample]) -> Gsm8kStats {
        let total = examples.len();
        let avg_question_len = examples.iter().map(|e| e.question.len()).sum::<usize>() as f32 / total as f32;
        let avg_steps = examples.iter().map(|e| e.reasoning_steps.len()).sum::<usize>() as f32 / total as f32;
        let max_steps = examples.iter().map(|e| e.reasoning_steps.len()).max().unwrap_or(0);

        Gsm8kStats {
            total_examples: total,
            avg_question_length: avg_question_len,
            avg_reasoning_steps: avg_steps,
            max_reasoning_steps: max_steps,
        }
    }
}

/// GSM8K dataset statistics
#[derive(Debug, Clone)]
pub struct Gsm8kStats {
    pub total_examples: usize,
    pub avg_question_length: f32,
    pub avg_reasoning_steps: f32,
    pub max_reasoning_steps: usize,
}

impl std::fmt::Display for Gsm8kStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GSM8K Dataset Statistics:")?;
        writeln!(f, "  Total examples: {}", self.total_examples)?;
        writeln!(f, "  Avg question length: {:.1} chars", self.avg_question_length)?;
        writeln!(f, "  Avg reasoning steps: {:.1}", self.avg_reasoning_steps)?;
        writeln!(f, "  Max reasoning steps: {}", self.max_reasoning_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_example() {
        let loader = Gsm8kLoader::new("data/external/gsm8k");

        let raw = Gsm8kRaw {
            question: "Janet has 3 apples. She buys 2 more. How many does she have?".to_string(),
            answer: "Janet starts with 3 apples.\nShe buys 2 more.\n3 + 2 = 5\n#### 5".to_string(),
        };

        let processed = loader.process_example(0, raw);

        assert_eq!(processed.final_answer, "5");
        assert_eq!(processed.reasoning_steps.len(), 3);
    }

    #[test]
    fn test_process_example_no_separator() {
        let loader = Gsm8kLoader::new("data/external/gsm8k");

        let raw = Gsm8kRaw {
            question: "What is 2 + 2?".to_string(),
            answer: "4".to_string(),
        };

        let processed = loader.process_example(0, raw);

        assert_eq!(processed.final_answer, "4");
    }
}
