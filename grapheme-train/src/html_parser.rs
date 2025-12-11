//! HTML Parser Module
//!
//! Parses HTML content and extracts clean text for training.
//! Data-002: HTML/web content parser.

use std::collections::HashSet;

/// HTML parsing configuration
#[derive(Debug, Clone)]
pub struct HtmlParseConfig {
    /// Extract text from these tags only (empty = all)
    pub include_tags: HashSet<String>,
    /// Skip these tags entirely
    pub exclude_tags: HashSet<String>,
    /// Preserve whitespace formatting
    pub preserve_whitespace: bool,
    /// Include alt text from images
    pub include_alt_text: bool,
    /// Include link text
    pub include_link_text: bool,
    /// Include title attributes
    pub include_titles: bool,
    /// Extract metadata (title, description, etc.)
    pub extract_metadata: bool,
}

impl Default for HtmlParseConfig {
    fn default() -> Self {
        let mut exclude = HashSet::new();
        exclude.insert("script".to_string());
        exclude.insert("style".to_string());
        exclude.insert("noscript".to_string());
        exclude.insert("nav".to_string());
        exclude.insert("footer".to_string());
        exclude.insert("header".to_string());
        exclude.insert("aside".to_string());

        Self {
            include_tags: HashSet::new(),
            exclude_tags: exclude,
            preserve_whitespace: false,
            include_alt_text: true,
            include_link_text: true,
            include_titles: false,
            extract_metadata: true,
        }
    }
}

/// Extracted HTML metadata
#[derive(Debug, Clone, Default)]
pub struct HtmlMetadata {
    /// Page title
    pub title: Option<String>,
    /// Meta description
    pub description: Option<String>,
    /// Meta keywords
    pub keywords: Vec<String>,
    /// Author
    pub author: Option<String>,
    /// Language
    pub language: Option<String>,
    /// Canonical URL
    pub canonical: Option<String>,
    /// Open Graph title
    pub og_title: Option<String>,
    /// Open Graph description
    pub og_description: Option<String>,
}

/// Parsed HTML result
#[derive(Debug, Clone)]
pub struct ParsedHtml {
    /// Extracted plain text content
    pub text: String,
    /// Page metadata
    pub metadata: HtmlMetadata,
    /// Extracted links (href, text)
    pub links: Vec<(String, String)>,
    /// Extracted images (src, alt)
    pub images: Vec<(String, String)>,
    /// Extracted headings by level
    pub headings: Vec<(u8, String)>,
    /// Extracted paragraphs
    pub paragraphs: Vec<String>,
}

/// HTML parser for extracting training content from web pages
pub struct HtmlParser {
    config: HtmlParseConfig,
}

impl HtmlParser {
    /// Create a new parser with default config
    pub fn new() -> Self {
        Self {
            config: HtmlParseConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: HtmlParseConfig) -> Self {
        Self { config }
    }

    /// Parse HTML content
    pub fn parse(&self, html: &str) -> ParsedHtml {
        let mut result = ParsedHtml {
            text: String::new(),
            metadata: HtmlMetadata::default(),
            links: Vec::new(),
            images: Vec::new(),
            headings: Vec::new(),
            paragraphs: Vec::new(),
        };

        // Extract metadata
        if self.config.extract_metadata {
            result.metadata = self.extract_metadata(html);
        }

        // Extract text content
        result.text = self.extract_text(html);

        // Extract structured elements
        result.links = self.extract_links(html);
        result.images = self.extract_images(html);
        result.headings = self.extract_headings(html);
        result.paragraphs = self.extract_paragraphs(html);

        result
    }

    /// Extract clean text from HTML
    pub fn extract_text(&self, html: &str) -> String {
        let mut text = String::new();
        let mut in_excluded_tag: usize = 0;
        let mut i = 0;
        let bytes = html.as_bytes();

        while i < bytes.len() {
            if bytes[i] == b'<' {
                // Find end of tag
                let tag_start = i + 1;
                let mut tag_end = tag_start;
                while tag_end < bytes.len() && bytes[tag_end] != b'>' {
                    tag_end += 1;
                }

                if tag_start < bytes.len() {
                    // Extract tag name (before space or >)
                    let tag_content = &html[tag_start..tag_end.min(bytes.len())];

                    // Check if closing tag (starts with /)
                    let is_closing = tag_content.starts_with('/');
                    let tag_search = if is_closing { &tag_content[1..] } else { tag_content };

                    let tag_name_end = tag_search.find(|c: char| c.is_whitespace() || c == '/' || c == '>').unwrap_or(tag_search.len());
                    let tag_name = tag_search[..tag_name_end].to_lowercase();

                    // Check excluded tags
                    if self.config.exclude_tags.contains(&tag_name) {
                        if is_closing {
                            in_excluded_tag = in_excluded_tag.saturating_sub(1);
                        } else {
                            in_excluded_tag += 1;
                        }
                    }

                    // Add space/newline for block elements
                    if in_excluded_tag == 0 && self.is_block_element(&tag_name) && !text.ends_with(' ') && !text.ends_with('\n') {
                        if tag_name == "p" || tag_name == "div" || tag_name == "br" {
                            text.push('\n');
                        } else {
                            text.push(' ');
                        }
                    }
                }

                // Skip past closing >
                i = if tag_end < bytes.len() { tag_end + 1 } else { bytes.len() };
            } else if in_excluded_tag == 0 {
                let c = html[i..].chars().next().unwrap();
                if c == '&' {
                    // Parse entity
                    let entity_end = html[i..].find(';').map(|pos| i + pos + 1).unwrap_or(i + 1);
                    let entity = &html[i+1..entity_end.saturating_sub(1)];
                    text.push_str(&self.decode_entity(entity));
                    i = entity_end;
                } else {
                    text.push(c);
                    i += c.len_utf8();
                }
            } else {
                // Skip character in excluded tag
                let c = html[i..].chars().next().unwrap();
                i += c.len_utf8();
            }
        }

        // Clean up whitespace
        if !self.config.preserve_whitespace {
            text = self.normalize_whitespace(&text);
        }

        text.trim().to_string()
    }

    // Helper: Decode a single entity
    fn decode_entity(&self, entity: &str) -> String {
        match entity {
            "nbsp" | "#160" => " ".to_string(),
            "amp" | "#38" => "&".to_string(),
            "lt" | "#60" => "<".to_string(),
            "gt" | "#62" => ">".to_string(),
            "quot" | "#34" => "\"".to_string(),
            "apos" | "#39" => "'".to_string(),
            "copy" | "#169" => "\u{00A9}".to_string(),
            "mdash" | "#8212" => "\u{2014}".to_string(),
            "ndash" | "#8211" => "\u{2013}".to_string(),
            _ => format!("&{};", entity),
        }
    }

    /// Extract page metadata
    fn extract_metadata(&self, html: &str) -> HtmlMetadata {
        let mut meta = HtmlMetadata::default();

        // Extract title
        if let Some(title) = self.extract_tag_content(html, "title") {
            meta.title = Some(self.decode_entities(&title));
        }

        // Extract meta tags
        let html_lower = html.to_lowercase();
        for meta_match in self.find_meta_tags(&html_lower, html) {
            let (name, content) = meta_match;
            match name.as_str() {
                "description" => meta.description = Some(content),
                "keywords" => {
                    meta.keywords = content.split(',').map(|s| s.trim().to_string()).collect();
                }
                "author" => meta.author = Some(content),
                "og:title" => meta.og_title = Some(content),
                "og:description" => meta.og_description = Some(content),
                _ => {}
            }
        }

        // Extract language
        if let Some(pos) = html_lower.find("lang=") {
            let start = pos + 6;
            if let Some(end) = html[start..].find(|c| c == '"' || c == '\'') {
                meta.language = Some(html[start..start + end].to_string());
            }
        }

        meta
    }

    /// Extract all links
    fn extract_links(&self, html: &str) -> Vec<(String, String)> {
        let mut links = Vec::new();
        let html_lower = html.to_lowercase();
        let mut pos = 0;

        while let Some(a_pos) = html_lower[pos..].find("<a ") {
            let tag_start = pos + a_pos;

            // Find href
            if let Some(href_pos) = html_lower[tag_start..].find("href=") {
                let href_start = tag_start + href_pos + 6;
                let quote = html.chars().nth(href_start - 1).unwrap_or('"');
                if let Some(href_end) = html[href_start..].find(quote) {
                    let href = html[href_start..href_start + href_end].to_string();

                    // Find closing tag and extract text
                    if let Some(close_pos) = html_lower[tag_start..].find("</a>") {
                        let tag_end = html[tag_start..].find('>').unwrap_or(0) + 1;
                        let link_text = self.extract_text(&html[tag_start + tag_end..tag_start + close_pos]);
                        links.push((href, link_text));
                    }
                }
            }

            pos = tag_start + 3;
        }

        links
    }

    /// Extract all images
    fn extract_images(&self, html: &str) -> Vec<(String, String)> {
        let mut images = Vec::new();
        let html_lower = html.to_lowercase();
        let mut pos = 0;

        while let Some(img_pos) = html_lower[pos..].find("<img ") {
            let tag_start = pos + img_pos;

            // Find src
            let mut src = String::new();
            if let Some(src_pos) = html_lower[tag_start..].find("src=") {
                let src_start = tag_start + src_pos + 5;
                let quote = html.chars().nth(src_start - 1).unwrap_or('"');
                if let Some(src_end) = html[src_start..].find(quote) {
                    src = html[src_start..src_start + src_end].to_string();
                }
            }

            // Find alt
            let mut alt = String::new();
            if let Some(alt_pos) = html_lower[tag_start..].find("alt=") {
                let alt_start = tag_start + alt_pos + 5;
                let quote = html.chars().nth(alt_start - 1).unwrap_or('"');
                if let Some(alt_end) = html[alt_start..].find(quote) {
                    alt = html[alt_start..alt_start + alt_end].to_string();
                }
            }

            if !src.is_empty() {
                images.push((src, alt));
            }

            pos = tag_start + 5;
        }

        images
    }

    /// Extract headings
    fn extract_headings(&self, html: &str) -> Vec<(u8, String)> {
        let mut headings = Vec::new();

        for level in 1..=6 {
            let open_tag = format!("<h{}", level);
            let close_tag = format!("</h{}>", level);
            let html_lower = html.to_lowercase();
            let mut pos = 0;

            while let Some(h_pos) = html_lower[pos..].find(&open_tag) {
                let tag_start = pos + h_pos;
                if let Some(content_start) = html[tag_start..].find('>') {
                    let start = tag_start + content_start + 1;
                    if let Some(end) = html_lower[start..].find(&close_tag) {
                        let heading_text = self.extract_text(&html[start..start + end]);
                        if !heading_text.is_empty() {
                            headings.push((level, heading_text));
                        }
                    }
                }
                pos = tag_start + 3;
            }
        }

        headings
    }

    /// Extract paragraphs
    fn extract_paragraphs(&self, html: &str) -> Vec<String> {
        let mut paragraphs = Vec::new();
        let html_lower = html.to_lowercase();
        let mut pos = 0;

        while let Some(p_pos) = html_lower[pos..].find("<p") {
            let tag_start = pos + p_pos;
            if let Some(content_start) = html[tag_start..].find('>') {
                let start = tag_start + content_start + 1;
                if let Some(end) = html_lower[start..].find("</p>") {
                    let para_text = self.extract_text(&html[start..start + end]);
                    if !para_text.is_empty() {
                        paragraphs.push(para_text);
                    }
                }
            }
            pos = tag_start + 2;
        }

        paragraphs
    }

    // Helper: Extract content between tags
    fn extract_tag_content(&self, html: &str, tag: &str) -> Option<String> {
        let html_lower = html.to_lowercase();
        let open_tag = format!("<{}", tag);
        let close_tag = format!("</{}>", tag);

        if let Some(start_pos) = html_lower.find(&open_tag) {
            if let Some(content_start) = html[start_pos..].find('>') {
                let start = start_pos + content_start + 1;
                if let Some(end) = html_lower[start..].find(&close_tag) {
                    return Some(html[start..start + end].to_string());
                }
            }
        }
        None
    }

    // Helper: Find meta tags
    fn find_meta_tags(&self, html_lower: &str, html: &str) -> Vec<(String, String)> {
        let mut metas = Vec::new();
        let mut pos = 0;

        while let Some(meta_pos) = html_lower[pos..].find("<meta ") {
            let tag_start = pos + meta_pos;
            let tag_end = html[tag_start..].find('>').unwrap_or(html.len() - tag_start);
            let tag = &html[tag_start..tag_start + tag_end];

            // Extract name/property and content
            let name = self.extract_attribute(tag, "name")
                .or_else(|| self.extract_attribute(tag, "property"));
            let content = self.extract_attribute(tag, "content");

            if let (Some(n), Some(c)) = (name, content) {
                metas.push((n.to_lowercase(), c));
            }

            pos = tag_start + 6;
        }

        metas
    }

    // Helper: Extract attribute value
    fn extract_attribute(&self, tag: &str, attr: &str) -> Option<String> {
        let pattern = format!("{}=", attr);
        let tag_lower = tag.to_lowercase();
        if let Some(pos) = tag_lower.find(&pattern) {
            let start = pos + pattern.len();
            let quote = tag.chars().nth(start)?;
            if quote == '"' || quote == '\'' {
                if let Some(end) = tag[start + 1..].find(quote) {
                    return Some(tag[start + 1..start + 1 + end].to_string());
                }
            }
        }
        None
    }

    // Helper: Check if block element
    fn is_block_element(&self, tag: &str) -> bool {
        matches!(
            tag,
            "p" | "div" | "br" | "hr" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
            | "ul" | "ol" | "li" | "table" | "tr" | "td" | "th"
            | "blockquote" | "pre" | "section" | "article"
        )
    }

    // Helper: Parse HTML entity
    fn parse_entity(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
        let mut entity = String::new();
        for c in chars.by_ref() {
            if c == ';' {
                break;
            }
            entity.push(c);
            if entity.len() > 10 {
                return format!("&{}", entity);
            }
        }

        match entity.as_str() {
            "nbsp" | "#160" => " ".to_string(),
            "amp" | "#38" => "&".to_string(),
            "lt" | "#60" => "<".to_string(),
            "gt" | "#62" => ">".to_string(),
            "quot" | "#34" => "\"".to_string(),
            "apos" | "#39" => "'".to_string(),
            "copy" | "#169" => "\u{00A9}".to_string(),
            "mdash" | "#8212" => "\u{2014}".to_string(),
            "ndash" | "#8211" => "\u{2013}".to_string(),
            _ => format!("&{};", entity),
        }
    }

    // Helper: Decode HTML entities in string
    fn decode_entities(&self, text: &str) -> String {
        let mut result = String::new();
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '&' {
                result.push_str(&self.parse_entity(&mut chars));
            } else {
                result.push(c);
            }
        }

        result
    }

    // Helper: Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> String {
        let mut result = String::new();
        let mut last_was_space = true;

        for c in text.chars() {
            if c.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(c);
                last_was_space = false;
            }
        }

        result
    }
}

impl Default for HtmlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_parsing() {
        let parser = HtmlParser::new();
        let html = "<html><body><p>Hello World</p></body></html>";
        let result = parser.parse(html);
        assert!(result.text.contains("Hello World"));
    }

    #[test]
    fn test_script_removal() {
        let parser = HtmlParser::new();
        let html = "<html><script>alert('bad');</script><p>Good content</p></html>";
        let result = parser.parse(html);
        // Debug: eprintln!("text: '{}'", result.text);
        assert!(!result.text.contains("alert"), "Should not contain script content, got: '{}'", result.text);
        assert!(result.text.contains("Good content"), "Should contain paragraph content, got: '{}'", result.text);
    }

    #[test]
    fn test_metadata_extraction() {
        let parser = HtmlParser::new();
        let html = r#"<html><head>
            <title>Test Page</title>
            <meta name="description" content="A test description">
            <meta name="keywords" content="test, page, html">
        </head><body>Content</body></html>"#;
        let result = parser.parse(html);
        assert_eq!(result.metadata.title, Some("Test Page".to_string()));
        assert_eq!(result.metadata.description, Some("A test description".to_string()));
        assert_eq!(result.metadata.keywords.len(), 3);
    }

    #[test]
    fn test_link_extraction() {
        let parser = HtmlParser::new();
        let html = r#"<a href="https://example.com">Example Link</a>"#;
        let result = parser.parse(html);
        assert_eq!(result.links.len(), 1);
        assert_eq!(result.links[0].0, "https://example.com");
        assert_eq!(result.links[0].1, "Example Link");
    }

    #[test]
    fn test_heading_extraction() {
        let parser = HtmlParser::new();
        let html = "<h1>Main Title</h1><h2>Subtitle</h2><p>Content</p>";
        let result = parser.parse(html);
        assert_eq!(result.headings.len(), 2);
        assert_eq!(result.headings[0], (1, "Main Title".to_string()));
        assert_eq!(result.headings[1], (2, "Subtitle".to_string()));
    }

    #[test]
    fn test_entity_decoding() {
        let parser = HtmlParser::new();
        let html = "<p>Hello &amp; World &lt;test&gt;</p>";
        let result = parser.parse(html);
        assert!(result.text.contains("Hello & World <test>"));
    }

    #[test]
    fn test_paragraph_extraction() {
        let parser = HtmlParser::new();
        let html = "<p>First paragraph.</p><p>Second paragraph.</p>";
        let result = parser.parse(html);
        assert_eq!(result.paragraphs.len(), 2);
    }

    #[test]
    fn test_nested_tags() {
        let parser = HtmlParser::new();
        let html = "<div><p>Hello <strong>bold</strong> text</p></div>";
        let result = parser.parse(html);
        assert!(result.text.contains("Hello bold text"));
    }
}
