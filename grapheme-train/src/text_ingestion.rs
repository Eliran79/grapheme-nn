//! Text File Ingestion Module
//!
//! Supports reading training data from various text formats:
//! - TXT: Plain text files
//! - MD: Markdown files
//! - JSON: Structured JSON data
//! - CSV: Comma-separated values
//!
//! Backend-169: Text file ingestion for training

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Supported text file formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TextFormat {
    /// Plain text (.txt)
    PlainText,
    /// Markdown (.md)
    Markdown,
    /// JSON (.json, .jsonl)
    Json,
    /// CSV (.csv)
    Csv,
    /// Unknown format
    Unknown,
}

impl TextFormat {
    /// Detect format from file extension
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("txt") => Self::PlainText,
            Some("md") => Self::Markdown,
            Some("json") | Some("jsonl") => Self::Json,
            Some("csv") => Self::Csv,
            _ => Self::Unknown,
        }
    }
}

/// A text document loaded from a file
#[derive(Debug, Clone)]
pub struct TextDocument {
    /// Source file path
    pub path: String,
    /// Detected format
    pub format: TextFormat,
    /// Raw content
    pub content: String,
    /// Extracted text chunks (paragraphs, lines, etc.)
    pub chunks: Vec<String>,
    /// Metadata extracted from the document
    pub metadata: DocumentMetadata,
}

/// Document metadata
#[derive(Debug, Clone, Default)]
pub struct DocumentMetadata {
    /// Total character count
    pub char_count: usize,
    /// Total word count
    pub word_count: usize,
    /// Total line count
    pub line_count: usize,
    /// Number of chunks extracted
    pub chunk_count: usize,
}

/// Text ingestion configuration
#[derive(Debug, Clone)]
pub struct IngestionConfig {
    /// Minimum chunk size in characters
    pub min_chunk_size: usize,
    /// Maximum chunk size in characters
    pub max_chunk_size: usize,
    /// Whether to strip markdown formatting
    pub strip_markdown: bool,
    /// Whether to skip empty lines
    pub skip_empty: bool,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 10,
            max_chunk_size: 1000,
            strip_markdown: true,
            skip_empty: true,
        }
    }
}

/// Text file ingestion engine
pub struct TextIngestion {
    config: IngestionConfig,
}

impl TextIngestion {
    /// Create a new text ingestion engine with default config
    pub fn new() -> Self {
        Self {
            config: IngestionConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: IngestionConfig) -> Self {
        Self { config }
    }

    /// Load a single text file
    pub fn load_file(&self, path: &Path) -> Result<TextDocument, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let format = TextFormat::from_path(path);
        let chunks = self.extract_chunks(&content, format);

        let metadata = DocumentMetadata {
            char_count: content.len(),
            word_count: content.split_whitespace().count(),
            line_count: content.lines().count(),
            chunk_count: chunks.len(),
        };

        Ok(TextDocument {
            path: path.to_string_lossy().to_string(),
            format,
            content,
            chunks,
            metadata,
        })
    }

    /// Load all text files from a directory
    pub fn load_directory(&self, dir: &Path) -> Result<Vec<TextDocument>, String> {
        let mut documents = Vec::new();

        let entries = fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory: {}", e))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let format = TextFormat::from_path(&path);
                if format != TextFormat::Unknown {
                    if let Ok(doc) = self.load_file(&path) {
                        documents.push(doc);
                    }
                }
            }
        }

        Ok(documents)
    }

    /// Load JSONL file (one JSON object per line)
    pub fn load_jsonl(&self, path: &Path) -> Result<Vec<serde_json::Value>, String> {
        let file = fs::File::open(path)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = BufReader::new(file);

        let mut records = Vec::new();
        for line in reader.lines().flatten() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(value) = serde_json::from_str(&line) {
                records.push(value);
            }
        }

        Ok(records)
    }

    /// Load CSV file
    pub fn load_csv(&self, path: &Path) -> Result<Vec<Vec<String>>, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let mut rows = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<String> = line
                .split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .collect();
            rows.push(fields);
        }

        Ok(rows)
    }

    /// Extract text chunks from content
    fn extract_chunks(&self, content: &str, format: TextFormat) -> Vec<String> {
        let processed = if self.config.strip_markdown && format == TextFormat::Markdown {
            self.strip_markdown_formatting(content)
        } else {
            content.to_string()
        };

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for line in processed.lines() {
            let line = line.trim();

            if self.config.skip_empty && line.is_empty() {
                // End current chunk on empty line
                if current_chunk.len() >= self.config.min_chunk_size {
                    chunks.push(current_chunk.trim().to_string());
                }
                current_chunk.clear();
                continue;
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(line);

            // Split if chunk is too large
            if current_chunk.len() >= self.config.max_chunk_size {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
            }
        }

        // Don't forget the last chunk
        if current_chunk.len() >= self.config.min_chunk_size {
            chunks.push(current_chunk.trim().to_string());
        }

        chunks
    }

    /// Strip markdown formatting
    fn strip_markdown_formatting(&self, content: &str) -> String {
        let mut result = String::with_capacity(content.len());

        for line in content.lines() {
            let line = line.trim();

            // Skip code blocks
            if line.starts_with("```") {
                continue;
            }

            // Remove headers (# ## ### etc)
            let line = if line.starts_with('#') {
                line.trim_start_matches('#').trim()
            } else {
                line
            };

            // Remove bold/italic markers
            let line = line
                .replace("**", "")
                .replace("__", "")
                .replace('*', "")
                .replace('_', "");

            // Remove link syntax [text](url) -> text
            let mut cleaned = String::new();
            let mut chars = line.chars().peekable();
            while let Some(c) = chars.next() {
                if c == '[' {
                    // Capture link text
                    let mut link_text = String::new();
                    while let Some(&next) = chars.peek() {
                        if next == ']' {
                            chars.next();
                            break;
                        }
                        link_text.push(chars.next().unwrap());
                    }
                    // Skip (url) part
                    if chars.peek() == Some(&'(') {
                        chars.next();
                        while let Some(&next) = chars.peek() {
                            if next == ')' {
                                chars.next();
                                break;
                            }
                            chars.next();
                        }
                    }
                    cleaned.push_str(&link_text);
                } else {
                    cleaned.push(c);
                }
            }

            result.push_str(&cleaned);
            result.push('\n');
        }

        result
    }
}

impl Default for TextIngestion {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(TextFormat::from_path(Path::new("file.txt")), TextFormat::PlainText);
        assert_eq!(TextFormat::from_path(Path::new("file.md")), TextFormat::Markdown);
        assert_eq!(TextFormat::from_path(Path::new("file.json")), TextFormat::Json);
        assert_eq!(TextFormat::from_path(Path::new("file.jsonl")), TextFormat::Json);
        assert_eq!(TextFormat::from_path(Path::new("file.csv")), TextFormat::Csv);
        assert_eq!(TextFormat::from_path(Path::new("file.xyz")), TextFormat::Unknown);
    }

    #[test]
    fn test_chunk_extraction() {
        let ingestion = TextIngestion::new();
        let content = "This is the first paragraph.\n\nThis is the second paragraph.\n\nAnd a third one.";
        let chunks = ingestion.extract_chunks(content, TextFormat::PlainText);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_markdown_stripping() {
        let ingestion = TextIngestion::new();
        let md = "# Header\n\n**Bold** and *italic* text.\n\n[Link](http://example.com)";
        let stripped = ingestion.strip_markdown_formatting(md);
        assert!(!stripped.contains('#'));
        assert!(!stripped.contains("**"));
        assert!(!stripped.contains("http://"));
    }

    #[test]
    fn test_metadata() {
        let ingestion = TextIngestion::new();
        let content = "Hello world.\n\nThis is a test.";
        let chunks = ingestion.extract_chunks(content, TextFormat::PlainText);

        let metadata = DocumentMetadata {
            char_count: content.len(),
            word_count: content.split_whitespace().count(),
            line_count: content.lines().count(),
            chunk_count: chunks.len(),
        };

        assert_eq!(metadata.word_count, 6);
        assert_eq!(metadata.line_count, 3);
    }
}
