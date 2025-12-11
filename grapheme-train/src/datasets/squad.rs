//! SQuAD Dataset Loader
//!
//! Downloads and parses the SQuAD (Stanford Question Answering Dataset)
//! for question-answering training.
//!
//! Dataset: https://rajpurkar.github.io/SQuAD-explorer/
//! Format: JSON with paragraphs, questions, and answers

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// SQuAD answer span
#[derive(Debug, Clone, Deserialize)]
pub struct SquadAnswer {
    pub text: String,
    pub answer_start: usize,
}

/// SQuAD question-answer pair
#[derive(Debug, Clone, Deserialize)]
pub struct SquadQA {
    pub id: String,
    pub question: String,
    pub answers: Vec<SquadAnswer>,
    #[serde(default)]
    pub is_impossible: bool,
}

/// SQuAD paragraph with context
#[derive(Debug, Clone, Deserialize)]
pub struct SquadParagraph {
    pub context: String,
    pub qas: Vec<SquadQA>,
}

/// SQuAD article
#[derive(Debug, Clone, Deserialize)]
pub struct SquadArticle {
    pub title: String,
    pub paragraphs: Vec<SquadParagraph>,
}

/// SQuAD dataset root
#[derive(Debug, Clone, Deserialize)]
pub struct SquadDataset {
    pub version: String,
    pub data: Vec<SquadArticle>,
}

/// Processed SQuAD example for GRAPHEME training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquadExample {
    pub id: String,
    pub context: String,
    pub question: String,
    pub answer: String,
    pub answer_start: usize,
    pub is_impossible: bool,
    pub title: String,
}

/// SQuAD dataset loader
pub struct SquadLoader {
    data_dir: PathBuf,
}

impl SquadLoader {
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
        }
    }

    /// Load SQuAD dataset from JSON file
    pub fn load(&self, split: &str) -> Result<Vec<SquadExample>, Box<dyn std::error::Error>> {
        // Try different file naming conventions
        let filename = match split {
            "train" => "train-v2.0.json",
            "dev" | "val" => "dev-v2.0.json",
            _ => &format!("{}.json", split),
        };

        let path = self.data_dir.join(filename);

        // Also try v1.1 format
        let path = if path.exists() {
            path
        } else {
            let v1_filename = match split {
                "train" => "train-v1.1.json",
                "dev" | "val" => "dev-v1.1.json",
                _ => &format!("{}.json", split),
            };
            let v1_path = self.data_dir.join(v1_filename);
            if v1_path.exists() {
                v1_path
            } else {
                return Err(format!(
                    "SQuAD file not found: {:?}. Download from https://rajpurkar.github.io/SQuAD-explorer/",
                    self.data_dir
                ).into());
            }
        };

        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let dataset: SquadDataset = serde_json::from_reader(reader)?;

        let mut examples = Vec::new();

        for article in dataset.data {
            for paragraph in article.paragraphs {
                for qa in paragraph.qas {
                    let (answer, answer_start) = if qa.is_impossible || qa.answers.is_empty() {
                        (String::new(), 0)
                    } else {
                        (
                            qa.answers[0].text.clone(),
                            qa.answers[0].answer_start,
                        )
                    };

                    examples.push(SquadExample {
                        id: qa.id,
                        context: paragraph.context.clone(),
                        question: qa.question,
                        answer,
                        answer_start,
                        is_impossible: qa.is_impossible,
                        title: article.title.clone(),
                    });
                }
            }
        }

        Ok(examples)
    }

    /// Convert to GRAPHEME training format (JSONL)
    pub fn to_grapheme_format<P: AsRef<Path>>(
        &self,
        examples: &[SquadExample],
        output_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        for example in examples {
            // Skip impossible questions for basic training
            if example.is_impossible {
                continue;
            }

            let training_example = serde_json::json!({
                "id": example.id,
                "domain": "text",
                "input_type": "text",
                "input": {
                    "text": format!("Context: {}\n\nQuestion: {}",
                        truncate_text(&example.context, 500),
                        &example.question
                    )
                },
                "expected_output": example.answer,
                "metadata": {
                    "difficulty": calculate_difficulty(&example),
                    "tags": ["squad", "qa", "reading_comprehension"],
                    "context_length": example.context.len(),
                    "question_length": example.question.len(),
                    "title": example.title
                }
            });

            writeln!(writer, "{}", serde_json::to_string(&training_example)?)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Filter examples by context length
    pub fn filter_by_context_length(
        &self,
        examples: &[SquadExample],
        min_len: usize,
        max_len: usize,
    ) -> Vec<SquadExample> {
        examples
            .iter()
            .filter(|e| e.context.len() >= min_len && e.context.len() <= max_len)
            .cloned()
            .collect()
    }

    /// Filter out impossible questions
    pub fn filter_possible(&self, examples: &[SquadExample]) -> Vec<SquadExample> {
        examples
            .iter()
            .filter(|e| !e.is_impossible)
            .cloned()
            .collect()
    }

    /// Get dataset statistics
    pub fn stats(&self, examples: &[SquadExample]) -> SquadStats {
        let total = examples.len();
        let impossible = examples.iter().filter(|e| e.is_impossible).count();

        let total_context_len: usize = examples.iter().map(|e| e.context.len()).sum();
        let total_question_len: usize = examples.iter().map(|e| e.question.len()).sum();
        let total_answer_len: usize = examples.iter().filter(|e| !e.is_impossible).map(|e| e.answer.len()).sum();

        let unique_titles: std::collections::HashSet<_> = examples.iter().map(|e| &e.title).collect();

        SquadStats {
            total_examples: total,
            impossible_questions: impossible,
            possible_questions: total - impossible,
            unique_articles: unique_titles.len(),
            avg_context_length: if total > 0 { total_context_len as f32 / total as f32 } else { 0.0 },
            avg_question_length: if total > 0 { total_question_len as f32 / total as f32 } else { 0.0 },
            avg_answer_length: if total > impossible {
                total_answer_len as f32 / (total - impossible) as f32
            } else {
                0.0
            },
        }
    }
}

/// Truncate text to max length while preserving word boundaries
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }

    // Find last space before max_len
    let truncated = &text[..max_len];
    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &text[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

/// Calculate difficulty based on various factors
fn calculate_difficulty(example: &SquadExample) -> u8 {
    let mut difficulty = 1u8;

    // Longer context = harder
    if example.context.len() > 500 {
        difficulty += 1;
    }
    if example.context.len() > 1000 {
        difficulty += 1;
    }

    // Longer answer = harder
    if example.answer.len() > 20 {
        difficulty += 1;
    }

    // Answer not at beginning = harder
    if example.answer_start > 200 {
        difficulty += 1;
    }

    difficulty.min(5)
}

/// SQuAD dataset statistics
#[derive(Debug, Clone)]
pub struct SquadStats {
    pub total_examples: usize,
    pub impossible_questions: usize,
    pub possible_questions: usize,
    pub unique_articles: usize,
    pub avg_context_length: f32,
    pub avg_question_length: f32,
    pub avg_answer_length: f32,
}

impl std::fmt::Display for SquadStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SQuAD Dataset Statistics:")?;
        writeln!(f, "  Total examples: {}", self.total_examples)?;
        writeln!(f, "  Possible questions: {}", self.possible_questions)?;
        writeln!(f, "  Impossible questions: {}", self.impossible_questions)?;
        writeln!(f, "  Unique articles: {}", self.unique_articles)?;
        writeln!(f, "  Avg context length: {:.1} chars", self.avg_context_length)?;
        writeln!(f, "  Avg question length: {:.1} chars", self.avg_question_length)?;
        writeln!(f, "  Avg answer length: {:.1} chars", self.avg_answer_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_text() {
        let text = "This is a long text that needs to be truncated at some point.";
        let truncated = truncate_text(text, 30);
        assert!(truncated.len() <= 33); // 30 + "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_truncate_short_text() {
        let text = "Short text";
        let truncated = truncate_text(text, 100);
        assert_eq!(truncated, text);
    }

    #[test]
    fn test_calculate_difficulty() {
        let easy = SquadExample {
            id: "1".to_string(),
            context: "Short context.".to_string(),
            question: "What?".to_string(),
            answer: "Answer".to_string(),
            answer_start: 0,
            is_impossible: false,
            title: "Test".to_string(),
        };
        assert!(calculate_difficulty(&easy) <= 2);

        let hard = SquadExample {
            id: "2".to_string(),
            context: "A".repeat(1500),
            question: "What is this very long question about?".to_string(),
            answer: "This is a much longer answer that spans multiple words".to_string(),
            answer_start: 500,
            is_impossible: false,
            title: "Test".to_string(),
        };
        assert!(calculate_difficulty(&hard) >= 3);
    }
}
