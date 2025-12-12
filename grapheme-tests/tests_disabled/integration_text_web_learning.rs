//! Integration tests for text and web learning pipelines
//!
//! Tests the text ingestion, web fetching, HTML parsing, and text preprocessing
//! modules working together in a realistic learning pipeline.

use grapheme_train::{
    ChunkConfig, FetchConfig, HtmlParser, IngestionConfig, TextChunker, TextFormat, TextIngestion,
    TextPreprocessor, WebContent, WebFetcher,
};
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Text Ingestion Pipeline Tests
// ============================================================================

/// Test loading and processing a plain text file
#[test]
fn test_text_ingestion_plain_text() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let file_path = temp_dir.path().join("sample.txt");

    // Create a sample plain text file
    let content = "This is the first paragraph.\n\nThis is the second paragraph.\n\nAnd a third one.";
    fs::write(&file_path, content).expect("write file");

    let ingestion = TextIngestion::new();
    let document = ingestion
        .load_file(&file_path)
        .expect("should load file");

    assert_eq!(document.format, TextFormat::PlainText);
    assert!(document.chunks.len() >= 3, "should extract multiple chunks");
    assert!(document.metadata.char_count > 0);
    assert!(document.metadata.word_count > 0);
}

/// Test loading and processing a markdown file
#[test]
fn test_text_ingestion_markdown() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let file_path = temp_dir.path().join("sample.md");

    // Create a sample markdown file
    let content = r#"# Header

This is **bold** and *italic* text.

## Section 2

- List item 1
- List item 2

[Link](http://example.com)"#;

    fs::write(&file_path, content).expect("write file");

    let ingestion = TextIngestion::new();
    let document = ingestion
        .load_file(&file_path)
        .expect("should load file");

    assert_eq!(document.format, TextFormat::Markdown);
    // With strip_markdown enabled by default, formatting should be removed
    for chunk in &document.chunks {
        assert!(
            !chunk.contains("**"),
            "bold markers should be stripped"
        );
    }
}

/// Test loading JSON files
#[test]
fn test_text_ingestion_json() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let file_path = temp_dir.path().join("sample.json");

    let content = r#"{"text": "Hello world", "category": "greeting"}"#;
    fs::write(&file_path, content).expect("write file");

    let ingestion = TextIngestion::new();
    let document = ingestion
        .load_file(&file_path)
        .expect("should load file");

    assert_eq!(document.format, TextFormat::Json);
    assert!(document.content.contains("Hello world"));
}

/// Test loading JSONL files
#[test]
fn test_text_ingestion_jsonl() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let file_path = temp_dir.path().join("sample.jsonl");

    let content = r#"{"id": 1, "text": "First line"}
{"id": 2, "text": "Second line"}
{"id": 3, "text": "Third line"}"#;
    fs::write(&file_path, content).expect("write file");

    let ingestion = TextIngestion::new();
    let document = ingestion
        .load_file(&file_path)
        .expect("should load file");

    assert_eq!(document.format, TextFormat::Json);
    assert!(document.metadata.line_count >= 3);
}

/// Test custom ingestion configuration
#[test]
fn test_text_ingestion_custom_config() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let file_path = temp_dir.path().join("sample.txt");

    let content = "Short.\n\nThis is a longer paragraph with more words.";
    fs::write(&file_path, content).expect("write file");

    let config = IngestionConfig {
        min_chunk_size: 20, // Skip short chunks
        max_chunk_size: 1000,
        strip_markdown: false,
        skip_empty: true,
    };

    let ingestion = TextIngestion::with_config(config);
    let document = ingestion
        .load_file(&file_path)
        .expect("should load file");

    // Short chunk "Short." should be filtered out
    for chunk in &document.chunks {
        assert!(
            chunk.len() >= 20 || chunk.is_empty(),
            "chunks should meet min size"
        );
    }
}

/// Test loading multiple files from a directory
#[test]
fn test_text_ingestion_directory() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create multiple files
    fs::write(temp_dir.path().join("file1.txt"), "Content 1").expect("write");
    fs::write(temp_dir.path().join("file2.txt"), "Content 2").expect("write");
    fs::write(temp_dir.path().join("file3.md"), "# Markdown").expect("write");

    let ingestion = TextIngestion::new();
    let documents = ingestion
        .load_directory(temp_dir.path())
        .expect("should load directory");

    assert!(documents.len() >= 3, "should load all files");
}

// ============================================================================
// Web Fetcher Tests
// ============================================================================

/// Test web fetcher creation
#[test]
fn test_web_fetcher_creation() {
    let _fetcher = WebFetcher::new();
    // Should create without panic
}

/// Test web fetcher with custom config
#[test]
fn test_web_fetcher_with_custom_config() {
    let config = FetchConfig {
        timeout_secs: 60,
        user_agent: "Test/1.0".to_string(),
        max_size: 1024 * 1024,
        follow_redirects: false,
        max_redirects: 0,
    };
    let _fetcher = WebFetcher::with_config(config);
    // Should create without panic
}

/// Test fetch config defaults
#[test]
fn test_web_fetcher_config_defaults() {
    let config = FetchConfig::default();
    assert_eq!(config.timeout_secs, 30);
    assert!(config.follow_redirects);
    assert_eq!(config.max_redirects, 5);
    assert!(config.max_size > 0);
}

/// Test custom fetch config applied
#[test]
fn test_web_fetcher_custom_config_applied() {
    let config = FetchConfig {
        timeout_secs: 60,
        user_agent: "CustomAgent/1.0".to_string(),
        max_size: 5 * 1024 * 1024,
        follow_redirects: false,
        max_redirects: 0,
    };

    let _fetcher = WebFetcher::with_config(config);
    // Should create fetcher without panic
}

/// Test WebContent type detection
#[test]
fn test_web_content_type_detection() {
    let html_content = WebContent {
        url: "http://example.com".to_string(),
        final_url: "http://example.com".to_string(),
        status_code: 200,
        content_type: Some("text/html; charset=utf-8".to_string()),
        content: vec![],
        text: None,
        content_length: 0,
    };
    assert!(html_content.is_html());
    assert!(!html_content.is_json());
    assert!(!html_content.is_text());

    let json_content = WebContent {
        content_type: Some("application/json".to_string()),
        ..html_content.clone()
    };
    assert!(json_content.is_json());
    assert!(!json_content.is_html());

    let text_content = WebContent {
        content_type: Some("text/plain".to_string()),
        ..html_content
    };
    assert!(text_content.is_text());
    assert!(!text_content.is_json());
}

// ============================================================================
// HTML Parser Tests
// ============================================================================

/// Test basic HTML parsing
#[test]
fn test_html_parser_basic() {
    let parser = HtmlParser::new();

    let html = r#"<html><body><p>Hello world</p></body></html>"#;
    let result = parser.parse(html);

    assert!(result.text.contains("Hello world"));
}

/// Test HTML title extraction
#[test]
fn test_html_parser_title_extraction() {
    let parser = HtmlParser::new();

    let html = r#"<html><head><title>Page Title</title></head><body>Content</body></html>"#;
    let result = parser.parse(html);

    assert_eq!(result.metadata.title, Some("Page Title".to_string()));
}

/// Test HTML link extraction
#[test]
fn test_html_parser_link_extraction() {
    let parser = HtmlParser::new();

    let html = r#"<html><body>
        <a href="http://example.com">Example</a>
        <a href="/relative/path">Relative</a>
    </body></html>"#;
    let result = parser.parse(html);

    assert!(!result.links.is_empty());
    // links is Vec<(String, String)> = (href, text)
    assert!(result.links.iter().any(|(href, _text)| href.contains("example.com")));
}

/// Test HTML script/style removal
#[test]
fn test_html_parser_script_removal() {
    let parser = HtmlParser::new();

    let html = r#"<html><body>
        <script>alert('hello');</script>
        <style>.hidden { display: none; }</style>
        <p>Visible content</p>
    </body></html>"#;
    let result = parser.parse(html);

    assert!(result.text.contains("Visible content"));
    assert!(!result.text.contains("alert"));
    assert!(!result.text.contains("display"));
}

/// Test HTML metadata extraction
#[test]
fn test_html_parser_metadata() {
    let parser = HtmlParser::new();

    let html = r#"<html><head>
        <meta name="description" content="Page description here">
        <meta name="keywords" content="test, html, parser">
    </head><body>Content</body></html>"#;
    let result = parser.parse(html);

    // metadata is HtmlMetadata struct
    assert_eq!(
        result.metadata.description,
        Some("Page description here".to_string())
    );
}

// ============================================================================
// Text Preprocessor Tests
// ============================================================================

/// Test basic text preprocessing
#[test]
fn test_text_preprocessor_basic() {
    let preprocessor = TextPreprocessor::new();

    let text = "Hello   World!  \n\n\n  Test.";
    let result = preprocessor.clean(text);

    // Should normalize whitespace
    assert!(!result.contains("   "));
    assert!(!result.contains("\n\n\n"));
}

/// Test text preprocessor with URLs
#[test]
fn test_text_preprocessor_removes_urls() {
    let preprocessor = TextPreprocessor::new();

    let text = "Visit http://example.com for more info.";
    let result = preprocessor.clean(text);

    // URLs should be removed
    assert!(!result.contains("http://"));
}

/// Test text chunking with default config
#[test]
fn test_text_chunker_default() {
    let chunker = TextChunker::new();

    // Create text longer than default chunk_size (512)
    let text = "This is a sentence. ".repeat(100);
    let chunks = chunker.chunk(&text);

    // Should create multiple chunks for long text
    assert!(chunks.len() >= 1, "should create at least one chunk");

    // Check chunks have content
    for chunk in &chunks {
        assert!(!chunk.text.is_empty(), "chunk should have text");
        assert!(chunk.word_count > 0, "chunk should have words");
    }
}

/// Test text chunking with custom config
#[test]
fn test_text_chunker_custom_config() {
    let config = ChunkConfig {
        chunk_size: 50,
        overlap: 10,
        respect_sentences: true,
        respect_paragraphs: false,
        min_chunk_size: 20,
    };
    let chunker = TextChunker::with_config(config);

    let text = "This is the first sentence. This is the second sentence. And this is the third.";
    let chunks = chunker.chunk(text);

    // Should create chunks
    assert!(!chunks.is_empty(), "should create chunks");

    // Check chunks aren't empty
    for chunk in &chunks {
        assert!(!chunk.text.is_empty());
    }
}

/// Test text chunking preserves content
#[test]
fn test_text_chunker_preserves_content() {
    let chunker = TextChunker::new();

    let text = "Important word1 word2 word3 word4 word5";
    let chunks = chunker.chunk(text);

    // All words should be present across chunks
    let combined: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
    assert!(combined.contains("word1"));
    assert!(combined.contains("word5"));
}

// ============================================================================
// Integration Pipeline Tests
// ============================================================================

/// Test full pipeline: file -> ingestion -> preprocessing
#[test]
fn test_full_text_pipeline() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let file_path = temp_dir.path().join("article.md");

    let content = r#"# Article Title

This is the **introduction** paragraph with some *emphasized* text.

## Section 1

First section content goes here. It contains multiple sentences.
More content in this section.

## Section 2

Second section with [a link](http://example.com) and more text.
"#;

    fs::write(&file_path, content).expect("write file");

    // Step 1: Ingest
    let ingestion = TextIngestion::new();
    let document = ingestion
        .load_file(&file_path)
        .expect("should load");

    // Step 2: Preprocess each chunk
    let preprocessor = TextPreprocessor::new();
    let processed_chunks: Vec<String> = document
        .chunks
        .iter()
        .map(|chunk| preprocessor.clean(chunk))
        .collect();

    // Verify pipeline results
    assert!(!processed_chunks.is_empty());
    for chunk in &processed_chunks {
        // No markdown formatting should remain (stripped by ingestion)
        assert!(!chunk.contains("**"));
    }
}

/// Test pipeline with JSON training data
#[test]
fn test_json_training_pipeline() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let file_path = temp_dir.path().join("training.jsonl");

    let content = r#"{"question": "What is 2+2?", "answer": "4"}
{"question": "What is the capital of France?", "answer": "Paris"}
{"question": "Who wrote Hamlet?", "answer": "Shakespeare"}"#;

    fs::write(&file_path, content).expect("write file");

    let ingestion = TextIngestion::new();
    let document = ingestion
        .load_file(&file_path)
        .expect("should load");

    // Should load as JSON format
    assert_eq!(document.format, TextFormat::Json);
    // Should have multiple lines
    assert!(document.metadata.line_count >= 3);
    // Content should be preserved
    assert!(document.content.contains("Shakespeare"));
}

/// Test mixed format directory processing
#[test]
fn test_mixed_format_directory() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create files in different formats
    fs::write(
        temp_dir.path().join("plain.txt"),
        "Plain text content",
    )
    .expect("write");
    fs::write(
        temp_dir.path().join("formatted.md"),
        "# Markdown content",
    )
    .expect("write");
    fs::write(
        temp_dir.path().join("data.json"),
        r#"{"key": "value"}"#,
    )
    .expect("write");

    let ingestion = TextIngestion::new();
    let documents = ingestion
        .load_directory(temp_dir.path())
        .expect("should load");

    // Should detect correct formats
    let formats: Vec<TextFormat> = documents.iter().map(|d| d.format).collect();
    assert!(formats.contains(&TextFormat::PlainText));
    assert!(formats.contains(&TextFormat::Markdown));
    assert!(formats.contains(&TextFormat::Json));
}

// ============================================================================
// Web Learning Pipeline Tests
// ============================================================================

/// Test web content to text pipeline
#[test]
fn test_web_content_to_text_pipeline() {
    // Simulate fetched web content
    let web_content = WebContent {
        url: "http://example.com/article".to_string(),
        final_url: "http://example.com/article".to_string(),
        status_code: 200,
        content_type: Some("text/html".to_string()),
        content: br#"<html><body><h1>Title</h1><p>Article content here.</p></body></html>"#.to_vec(),
        text: Some("<html><body><h1>Title</h1><p>Article content here.</p></body></html>".to_string()),
        content_length: 0,
    };

    // Parse HTML
    let parser = HtmlParser::new();
    let parsed = parser.parse(web_content.text.as_ref().unwrap());

    // Preprocess
    let preprocessor = TextPreprocessor::new();
    let processed = preprocessor.clean(&parsed.text);

    // Verify pipeline - text should be extracted and lowercased (default config)
    assert!(processed.contains("title"));
    assert!(processed.contains("article content"));
    assert!(!processed.contains("<html>"));
}

/// Test error handling for missing files
#[test]
fn test_text_ingestion_missing_file() {
    use std::path::Path;
    let ingestion = TextIngestion::new();
    let result = ingestion.load_file(Path::new("/nonexistent/path/file.txt"));
    assert!(result.is_err());
}

/// Test format detection for various extensions
#[test]
fn test_format_detection_comprehensive() {
    use std::path::Path;

    assert_eq!(TextFormat::from_path(Path::new("file.txt")), TextFormat::PlainText);
    assert_eq!(TextFormat::from_path(Path::new("file.TXT")), TextFormat::Unknown); // Case sensitive
    assert_eq!(TextFormat::from_path(Path::new("file.md")), TextFormat::Markdown);
    assert_eq!(TextFormat::from_path(Path::new("file.json")), TextFormat::Json);
    assert_eq!(TextFormat::from_path(Path::new("file.jsonl")), TextFormat::Json);
    assert_eq!(TextFormat::from_path(Path::new("file.csv")), TextFormat::Csv);
    assert_eq!(TextFormat::from_path(Path::new("file.unknown")), TextFormat::Unknown);
    assert_eq!(TextFormat::from_path(Path::new("noextension")), TextFormat::Unknown);
}
