//! Web Content Fetcher Module
//!
//! Fetches training content from HTTP/HTTPS URLs.
//! Backend-170: Web content fetcher for training.
//!
//! Uses ureq for HTTP/HTTPS with native-tls support.

use std::io::Read;
use std::time::Duration;

/// HTTP client configuration
#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// User agent string
    pub user_agent: String,
    /// Maximum response size in bytes
    pub max_size: usize,
    /// Follow redirects
    pub follow_redirects: bool,
    /// Maximum number of redirects
    pub max_redirects: u32,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            user_agent: "GRAPHEME-Train/1.0 (+https://github.com/grapheme-nn)".to_string(),
            max_size: 10 * 1024 * 1024, // 10MB
            follow_redirects: true,
            max_redirects: 5,
        }
    }
}

/// Fetched web content
#[derive(Debug, Clone)]
pub struct WebContent {
    /// Original URL
    pub url: String,
    /// Final URL after redirects
    pub final_url: String,
    /// HTTP status code
    pub status_code: u16,
    /// Content type (MIME)
    pub content_type: Option<String>,
    /// Raw content bytes
    pub content: Vec<u8>,
    /// Content as UTF-8 string (if valid)
    pub text: Option<String>,
    /// Content length
    pub content_length: usize,
}

impl WebContent {
    /// Check if content is HTML
    pub fn is_html(&self) -> bool {
        self.content_type
            .as_ref()
            .map(|ct| ct.contains("text/html"))
            .unwrap_or(false)
    }

    /// Check if content is plain text
    pub fn is_text(&self) -> bool {
        self.content_type
            .as_ref()
            .map(|ct| ct.contains("text/plain"))
            .unwrap_or(false)
    }

    /// Check if content is JSON
    pub fn is_json(&self) -> bool {
        self.content_type
            .as_ref()
            .map(|ct| ct.contains("application/json"))
            .unwrap_or(false)
    }
}

/// Web content fetcher using ureq with HTTPS support
pub struct WebFetcher {
    config: FetchConfig,
    agent: ureq::Agent,
}

impl WebFetcher {
    /// Create a new web fetcher with default config
    pub fn new() -> Self {
        Self::with_config(FetchConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: FetchConfig) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(config.timeout_secs))
            .user_agent(&config.user_agent)
            .redirects(if config.follow_redirects { config.max_redirects } else { 0 })
            .build();

        Self { config, agent }
    }

    /// Fetch content from a URL (supports both HTTP and HTTPS)
    pub fn fetch(&self, url: &str) -> Result<WebContent, String> {
        let url = url.trim();

        // Validate URL scheme
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err("URL must start with http:// or https://".to_string());
        }

        // Make the request
        let response = self.agent
            .get(url)
            .call()
            .map_err(|e| format!("Request failed: {}", e))?;

        let status_code = response.status();
        let content_type = response.header("content-type").map(|s| s.to_string());
        let final_url = response.get_url().to_string();

        // Read response body with size limit
        let mut content = Vec::new();
        let mut reader = response.into_reader();

        // Read in chunks to enforce size limit
        let mut buffer = [0u8; 8192];
        loop {
            let bytes_read = reader.read(&mut buffer)
                .map_err(|e| format!("Read failed: {}", e))?;

            if bytes_read == 0 {
                break;
            }

            content.extend_from_slice(&buffer[..bytes_read]);

            if content.len() > self.config.max_size {
                return Err(format!(
                    "Content too large: exceeded {} bytes limit",
                    self.config.max_size
                ));
            }
        }

        // Try to decode as UTF-8
        let text = String::from_utf8(content.clone()).ok();
        let content_length = content.len();

        Ok(WebContent {
            url: url.to_string(),
            final_url,
            status_code,
            content_type,
            content,
            text,
            content_length,
        })
    }

    /// Fetch multiple URLs
    pub fn fetch_all(&self, urls: &[&str]) -> Vec<Result<WebContent, String>> {
        urls.iter().map(|url| self.fetch(url)).collect()
    }

    /// Fetch JSON content and parse it
    pub fn fetch_json<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T, String> {
        let content = self.fetch(url)?;

        let text = content.text
            .ok_or("Response is not valid UTF-8")?;

        serde_json::from_str(&text)
            .map_err(|e| format!("JSON parse error: {}", e))
    }

    /// Get configuration
    pub fn config(&self) -> &FetchConfig {
        &self.config
    }
}

impl Default for WebFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fetch_config_defaults() {
        let config = FetchConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert!(config.follow_redirects);
        assert_eq!(config.max_redirects, 5);
    }

    #[test]
    fn test_web_content_type_checks() {
        let content = WebContent {
            url: "http://example.com".to_string(),
            final_url: "http://example.com".to_string(),
            status_code: 200,
            content_type: Some("text/html; charset=utf-8".to_string()),
            content: vec![],
            text: None,
            content_length: 0,
        };
        assert!(content.is_html());
        assert!(!content.is_json());
        assert!(!content.is_text());
    }

    #[test]
    fn test_web_content_json_type() {
        let content = WebContent {
            url: "http://api.example.com".to_string(),
            final_url: "http://api.example.com".to_string(),
            status_code: 200,
            content_type: Some("application/json".to_string()),
            content: vec![],
            text: Some("{}".to_string()),
            content_length: 2,
        };
        assert!(content.is_json());
        assert!(!content.is_html());
    }

    #[test]
    fn test_invalid_url_scheme() {
        let fetcher = WebFetcher::new();
        assert!(fetcher.fetch("ftp://example.com").is_err());
        assert!(fetcher.fetch("example.com").is_err());
    }

    #[test]
    fn test_fetcher_config() {
        let config = FetchConfig {
            timeout_secs: 60,
            user_agent: "Test/1.0".to_string(),
            max_size: 1024,
            follow_redirects: false,
            max_redirects: 0,
        };
        let fetcher = WebFetcher::with_config(config.clone());
        assert_eq!(fetcher.config().timeout_secs, 60);
        assert_eq!(fetcher.config().max_size, 1024);
    }
}
