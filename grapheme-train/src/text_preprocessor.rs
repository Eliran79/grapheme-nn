//! Text Preprocessing Pipeline Module
//!
//! Provides text cleaning, tokenization, and chunking for training data.
//! Data-001: Text preprocessing pipeline.

use std::collections::HashSet;

/// Text preprocessing configuration
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Convert to lowercase
    pub lowercase: bool,
    /// Remove punctuation
    pub remove_punctuation: bool,
    /// Remove numbers
    pub remove_numbers: bool,
    /// Remove extra whitespace
    pub normalize_whitespace: bool,
    /// Remove URLs
    pub remove_urls: bool,
    /// Remove email addresses
    pub remove_emails: bool,
    /// Remove HTML tags
    pub remove_html_tags: bool,
    /// Minimum word length to keep
    pub min_word_length: usize,
    /// Maximum word length to keep
    pub max_word_length: usize,
    /// Custom stopwords to remove
    pub stopwords: HashSet<String>,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: false,
            remove_numbers: false,
            normalize_whitespace: true,
            remove_urls: true,
            remove_emails: true,
            remove_html_tags: true,
            min_word_length: 1,
            max_word_length: 100,
            stopwords: HashSet::new(),
        }
    }
}

/// Chunking configuration for splitting text
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size in characters
    pub chunk_size: usize,
    /// Overlap between chunks in characters
    pub overlap: usize,
    /// Split on sentence boundaries when possible
    pub respect_sentences: bool,
    /// Split on paragraph boundaries when possible
    pub respect_paragraphs: bool,
    /// Minimum chunk size (won't create chunks smaller than this)
    pub min_chunk_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 50,
            respect_sentences: true,
            respect_paragraphs: true,
            min_chunk_size: 100,
        }
    }
}

/// A preprocessed text chunk with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The chunk text content
    pub text: String,
    /// Character offset from start of original text
    pub start_offset: usize,
    /// Character offset to end of chunk in original text
    pub end_offset: usize,
    /// Chunk index in sequence
    pub index: usize,
    /// Word count in this chunk
    pub word_count: usize,
}

/// Text preprocessor for cleaning and preparing text for training
pub struct TextPreprocessor {
    config: PreprocessConfig,
}

impl TextPreprocessor {
    /// Create a new preprocessor with default config
    pub fn new() -> Self {
        Self {
            config: PreprocessConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: PreprocessConfig) -> Self {
        Self { config }
    }

    /// Clean and preprocess text
    pub fn clean(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Remove HTML tags
        if self.config.remove_html_tags {
            result = self.remove_html_tags(&result);
        }

        // Remove URLs
        if self.config.remove_urls {
            result = self.remove_urls(&result);
        }

        // Remove emails
        if self.config.remove_emails {
            result = self.remove_emails(&result);
        }

        // Convert to lowercase
        if self.config.lowercase {
            result = result.to_lowercase();
        }

        // Remove punctuation
        if self.config.remove_punctuation {
            result = result.chars()
                .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
                .collect();
        }

        // Remove numbers
        if self.config.remove_numbers {
            result = result.chars()
                .filter(|c| !c.is_numeric())
                .collect();
        }

        // Normalize whitespace
        if self.config.normalize_whitespace {
            result = self.normalize_whitespace(&result);
        }

        // Apply stopwords and word length filters
        if !self.config.stopwords.is_empty()
            || self.config.min_word_length > 1
            || self.config.max_word_length < 100
        {
            result = self.filter_words(&result);
        }

        result
    }

    /// Tokenize text into words
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }

    /// Tokenize into sentences
    pub fn sentence_tokenize(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            current.push(c);
            if c == '.' || c == '!' || c == '?' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        // Don't forget remaining text
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }

    // Remove HTML tags (simple regex-free implementation)
    fn remove_html_tags(&self, text: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;

        for c in text.chars() {
            if c == '<' {
                in_tag = true;
            } else if c == '>' {
                in_tag = false;
                result.push(' '); // Replace tag with space
            } else if !in_tag {
                result.push(c);
            }
        }

        result
    }

    // Remove URLs (simple pattern matching)
    fn remove_urls(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let filtered: Vec<&str> = words
            .into_iter()
            .filter(|w| {
                !w.starts_with("http://")
                    && !w.starts_with("https://")
                    && !w.starts_with("www.")
            })
            .collect();
        filtered.join(" ")
    }

    // Remove email addresses (simple @ detection)
    fn remove_emails(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let filtered: Vec<&str> = words
            .into_iter()
            .filter(|w| !w.contains('@') || !w.contains('.'))
            .collect();
        filtered.join(" ")
    }

    // Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    // Filter words by stopwords and length
    fn filter_words(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let filtered: Vec<&str> = words
            .into_iter()
            .filter(|w| {
                let len = w.len();
                len >= self.config.min_word_length
                    && len <= self.config.max_word_length
                    && !self.config.stopwords.contains(*w)
            })
            .collect();
        filtered.join(" ")
    }
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Text chunker for splitting text into training-sized pieces
pub struct TextChunker {
    config: ChunkConfig,
}

impl TextChunker {
    /// Create a new chunker with default config
    pub fn new() -> Self {
        Self {
            config: ChunkConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ChunkConfig) -> Self {
        Self { config }
    }

    /// Split text into chunks
    pub fn chunk(&self, text: &str) -> Vec<TextChunk> {
        if text.len() <= self.config.chunk_size {
            return vec![TextChunk {
                text: text.to_string(),
                start_offset: 0,
                end_offset: text.len(),
                index: 0,
                word_count: text.split_whitespace().count(),
            }];
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut index = 0;

        while start < text.len() {
            // Ensure start is on a character boundary
            while start < text.len() && !text.is_char_boundary(start) {
                start += 1;
            }
            if start >= text.len() {
                break;
            }

            let end = self.find_chunk_end(text, start);
            let chunk_text = &text[start..end];

            if chunk_text.len() >= self.config.min_chunk_size || start == 0 {
                chunks.push(TextChunk {
                    text: chunk_text.trim().to_string(),
                    start_offset: start,
                    end_offset: end,
                    index,
                    word_count: chunk_text.split_whitespace().count(),
                });
                index += 1;
            }

            // Move start, accounting for overlap
            let advance = if self.config.overlap > 0 && end > start + self.config.overlap {
                end - start - self.config.overlap
            } else {
                end - start
            };

            start += advance.max(1);
        }

        chunks
    }

    /// Chunk with context windows (includes previous/next chunk overlap)
    pub fn chunk_with_context(&self, text: &str) -> Vec<(TextChunk, Option<String>, Option<String>)> {
        let chunks = self.chunk(text);
        let mut result = Vec::new();

        for i in 0..chunks.len() {
            let prev_context = if i > 0 {
                Some(chunks[i - 1].text.clone())
            } else {
                None
            };

            let next_context = if i + 1 < chunks.len() {
                Some(chunks[i + 1].text.clone())
            } else {
                None
            };

            result.push((chunks[i].clone(), prev_context, next_context));
        }

        result
    }

    // Find the best end position for a chunk
    fn find_chunk_end(&self, text: &str, start: usize) -> usize {
        let raw_max_end = (start + self.config.chunk_size).min(text.len());

        // Adjust to valid character boundary (find next valid boundary)
        let max_end = if raw_max_end >= text.len() {
            text.len()
        } else {
            // Find the nearest character boundary at or before raw_max_end
            let mut end = raw_max_end;
            while end > start && !text.is_char_boundary(end) {
                end -= 1;
            }
            end
        };

        if max_end >= text.len() {
            return text.len();
        }

        // Try to find paragraph boundary
        if self.config.respect_paragraphs {
            if let Some(para_end) = self.find_paragraph_boundary(text, start, max_end) {
                return para_end;
            }
        }

        // Try to find sentence boundary
        if self.config.respect_sentences {
            if let Some(sent_end) = self.find_sentence_boundary(text, start, max_end) {
                return sent_end;
            }
        }

        // Try to find word boundary
        if let Some(word_end) = self.find_word_boundary(text, max_end) {
            return word_end;
        }

        max_end
    }

    fn find_paragraph_boundary(&self, text: &str, start: usize, max_end: usize) -> Option<usize> {
        let search_text = &text[start..max_end];
        if let Some(pos) = search_text.rfind("\n\n") {
            let boundary = start + pos + 2;
            if boundary - start >= self.config.min_chunk_size {
                return Some(boundary);
            }
        }
        None
    }

    fn find_sentence_boundary(&self, text: &str, start: usize, max_end: usize) -> Option<usize> {
        let search_text = &text[start..max_end];
        for (i, c) in search_text.char_indices().rev() {
            if c == '.' || c == '!' || c == '?' {
                let boundary = start + i + 1;
                if boundary - start >= self.config.min_chunk_size {
                    return Some(boundary);
                }
            }
        }
        None
    }

    fn find_word_boundary(&self, text: &str, max_end: usize) -> Option<usize> {
        // Look backwards from max_end for whitespace
        let mut search_start = max_end.saturating_sub(50);
        // Ensure search_start is on a character boundary
        while search_start > 0 && !text.is_char_boundary(search_start) {
            search_start -= 1;
        }
        for (i, c) in text[search_start..max_end].char_indices().rev() {
            if c.is_whitespace() {
                return Some(search_start + i);
            }
        }
        None
    }
}

impl Default for TextChunker {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined preprocessing pipeline
pub struct TextPipeline {
    preprocessor: TextPreprocessor,
    chunker: TextChunker,
}

impl TextPipeline {
    /// Create a new pipeline with default configs
    pub fn new() -> Self {
        Self {
            preprocessor: TextPreprocessor::new(),
            chunker: TextChunker::new(),
        }
    }

    /// Create with custom configs
    pub fn with_configs(preprocess_config: PreprocessConfig, chunk_config: ChunkConfig) -> Self {
        Self {
            preprocessor: TextPreprocessor::with_config(preprocess_config),
            chunker: TextChunker::with_config(chunk_config),
        }
    }

    /// Process text through the full pipeline
    pub fn process(&self, text: &str) -> Vec<TextChunk> {
        let cleaned = self.preprocessor.clean(text);
        self.chunker.chunk(&cleaned)
    }

    /// Process with context windows
    pub fn process_with_context(&self, text: &str) -> Vec<(TextChunk, Option<String>, Option<String>)> {
        let cleaned = self.preprocessor.clean(text);
        self.chunker.chunk_with_context(&cleaned)
    }
}

impl Default for TextPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cleaning() {
        let preprocessor = TextPreprocessor::new();
        let text = "Hello   World!  How are  you?";
        let cleaned = preprocessor.clean(text);
        assert_eq!(cleaned, "hello world! how are you?");
    }

    #[test]
    fn test_url_removal() {
        let preprocessor = TextPreprocessor::new();
        let text = "Check out https://example.com for more info";
        let cleaned = preprocessor.clean(text);
        assert!(!cleaned.contains("https://"));
    }

    #[test]
    fn test_html_removal() {
        let preprocessor = TextPreprocessor::new();
        let text = "<p>Hello <b>World</b></p>";
        let cleaned = preprocessor.clean(text);
        assert!(!cleaned.contains('<'));
        assert!(cleaned.contains("hello"));
        assert!(cleaned.contains("world"));
    }

    #[test]
    fn test_tokenization() {
        let preprocessor = TextPreprocessor::new();
        let tokens = preprocessor.tokenize("Hello world how are you");
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], "Hello");
    }

    #[test]
    fn test_sentence_tokenization() {
        let preprocessor = TextPreprocessor::new();
        let sentences = preprocessor.sentence_tokenize("Hello world. How are you? I am fine!");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_chunking_small_text() {
        let chunker = TextChunker::new();
        let text = "Short text.";
        let chunks = chunker.chunk(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short text.");
    }

    #[test]
    fn test_chunking_with_overlap() {
        let config = ChunkConfig {
            chunk_size: 50,
            overlap: 10,
            min_chunk_size: 10,
            ..Default::default()
        };
        let chunker = TextChunker::with_config(config);
        let text = "This is a longer text that should be split into multiple chunks for testing purposes.";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_chunk_metadata() {
        let chunker = TextChunker::new();
        let text = "Hello world.";
        let chunks = chunker.chunk(text);
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].start_offset, 0);
        assert!(chunks[0].word_count > 0);
    }

    #[test]
    fn test_pipeline() {
        let pipeline = TextPipeline::new();
        let text = "<p>Hello   World!</p> Check https://test.com for more.";
        let chunks = pipeline.process(text);
        assert!(!chunks.is_empty());
        assert!(!chunks[0].text.contains('<'));
        assert!(!chunks[0].text.contains("https://"));
    }

    #[test]
    fn test_stopwords() {
        let mut config = PreprocessConfig::default();
        config.stopwords.insert("the".to_string());
        config.stopwords.insert("a".to_string());
        let preprocessor = TextPreprocessor::with_config(config);
        let cleaned = preprocessor.clean("The cat sat on a mat");
        assert!(!cleaned.contains("the "));
        assert!(!cleaned.contains(" a "));
    }
}
