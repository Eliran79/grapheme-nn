//! Web Content Fetcher Module
//!
//! Fetches training content from HTTP/HTTPS URLs.
//! Backend-170: Web content fetcher for training.

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
            user_agent: "GRAPHEME-Train/1.0".to_string(),
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

/// Web content fetcher
pub struct WebFetcher {
    config: FetchConfig,
}

impl WebFetcher {
    /// Create a new web fetcher with default config
    pub fn new() -> Self {
        Self {
            config: FetchConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: FetchConfig) -> Self {
        Self { config }
    }

    /// Fetch content from a URL using pure Rust (no external dependencies)
    /// This is a minimal implementation - for production use, consider ureq or reqwest
    pub fn fetch(&self, url: &str) -> Result<WebContent, String> {
        // Parse URL
        let parsed = self.parse_url(url)?;

        // For now, we implement a simple HTTP/1.1 client
        // In production, you'd want to use ureq, reqwest, or similar
        self.fetch_http(&parsed, url, 0)
    }

    /// Fetch multiple URLs
    pub fn fetch_all(&self, urls: &[&str]) -> Vec<Result<WebContent, String>> {
        urls.iter().map(|url| self.fetch(url)).collect()
    }

    // Internal URL parsing
    fn parse_url(&self, url: &str) -> Result<ParsedUrl, String> {
        let url = url.trim();

        let (scheme, rest) = if url.starts_with("https://") {
            ("https", &url[8..])
        } else if url.starts_with("http://") {
            ("http", &url[7..])
        } else {
            return Err("URL must start with http:// or https://".to_string());
        };

        let (host_port, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };

        let (host, port) = match host_port.find(':') {
            Some(i) => {
                let port_str = &host_port[i+1..];
                let port = port_str.parse::<u16>()
                    .map_err(|_| format!("Invalid port: {}", port_str))?;
                (&host_port[..i], port)
            }
            None => {
                let default_port = if scheme == "https" { 443 } else { 80 };
                (host_port, default_port)
            }
        };

        Ok(ParsedUrl {
            scheme: scheme.to_string(),
            host: host.to_string(),
            port,
            path: path.to_string(),
        })
    }

    // Simple HTTP fetch implementation
    fn fetch_http(&self, parsed: &ParsedUrl, original_url: &str, redirect_count: u32) -> Result<WebContent, String> {
        if redirect_count > self.config.max_redirects {
            return Err("Too many redirects".to_string());
        }

        use std::net::TcpStream;

        // Connect with timeout
        let addr = format!("{}:{}", parsed.host, parsed.port);
        let mut stream = TcpStream::connect_timeout(
            &addr.parse().map_err(|e| format!("Invalid address: {}", e))?,
            Duration::from_secs(self.config.timeout_secs),
        ).map_err(|e| format!("Connection failed: {}", e))?;

        stream.set_read_timeout(Some(Duration::from_secs(self.config.timeout_secs)))
            .map_err(|e| format!("Failed to set timeout: {}", e))?;

        // For HTTPS, we need TLS - for now, only support HTTP
        if parsed.scheme == "https" {
            return Err("HTTPS requires TLS support. Use the http_client feature or add native-tls/rustls dependency.".to_string());
        }

        // Send HTTP request
        let request = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: {}\r\nConnection: close\r\n\r\n",
            parsed.path,
            parsed.host,
            self.config.user_agent
        );

        use std::io::Write;
        stream.write_all(request.as_bytes())
            .map_err(|e| format!("Write failed: {}", e))?;

        // Read response
        let mut response = Vec::new();
        stream.read_to_end(&mut response)
            .map_err(|e| format!("Read failed: {}", e))?;

        // Parse response
        self.parse_http_response(&response, original_url, parsed, redirect_count)
    }

    fn parse_http_response(
        &self,
        response: &[u8],
        original_url: &str,
        parsed: &ParsedUrl,
        redirect_count: u32,
    ) -> Result<WebContent, String> {
        // Find header/body separator
        let header_end = response.windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or("Invalid HTTP response: no header terminator")?;

        let header_bytes = &response[..header_end];
        let body = &response[header_end + 4..];

        // Parse header
        let header_str = String::from_utf8_lossy(header_bytes);
        let lines: Vec<&str> = header_str.lines().collect();

        if lines.is_empty() {
            return Err("Empty HTTP response".to_string());
        }

        // Parse status line
        let status_parts: Vec<&str> = lines[0].splitn(3, ' ').collect();
        if status_parts.len() < 2 {
            return Err("Invalid HTTP status line".to_string());
        }

        let status_code: u16 = status_parts[1]
            .parse()
            .map_err(|_| "Invalid status code")?;

        // Parse headers
        let mut content_type = None;
        let mut location = None;

        for line in &lines[1..] {
            let parts: Vec<&str> = line.splitn(2, ':').collect();
            if parts.len() == 2 {
                let key = parts[0].trim().to_lowercase();
                let value = parts[1].trim();
                match key.as_str() {
                    "content-type" => content_type = Some(value.to_string()),
                    "location" => location = Some(value.to_string()),
                    _ => {}
                }
            }
        }

        // Handle redirects
        if self.config.follow_redirects && (status_code == 301 || status_code == 302 || status_code == 307 || status_code == 308) {
            if let Some(loc) = location {
                let redirect_url = if loc.starts_with("http") {
                    loc
                } else if loc.starts_with('/') {
                    format!("{}://{}:{}{}", parsed.scheme, parsed.host, parsed.port, loc)
                } else {
                    return Err(format!("Invalid redirect location: {}", loc));
                };
                let new_parsed = self.parse_url(&redirect_url)?;
                return self.fetch_http(&new_parsed, original_url, redirect_count + 1);
            }
        }

        // Check size limit
        if body.len() > self.config.max_size {
            return Err(format!("Content too large: {} bytes (max {})", body.len(), self.config.max_size));
        }

        // Try to decode as UTF-8
        let text = String::from_utf8(body.to_vec()).ok();

        Ok(WebContent {
            url: original_url.to_string(),
            final_url: format!("{}://{}:{}{}", parsed.scheme, parsed.host, parsed.port, parsed.path),
            status_code,
            content_type,
            content: body.to_vec(),
            text,
            content_length: body.len(),
        })
    }
}

impl Default for WebFetcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed URL components
#[derive(Debug)]
struct ParsedUrl {
    scheme: String,
    host: String,
    port: u16,
    path: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_url_http() {
        let fetcher = WebFetcher::new();
        let parsed = fetcher.parse_url("http://example.com/path").unwrap();
        assert_eq!(parsed.scheme, "http");
        assert_eq!(parsed.host, "example.com");
        assert_eq!(parsed.port, 80);
        assert_eq!(parsed.path, "/path");
    }

    #[test]
    fn test_parse_url_https() {
        let fetcher = WebFetcher::new();
        let parsed = fetcher.parse_url("https://example.com/").unwrap();
        assert_eq!(parsed.scheme, "https");
        assert_eq!(parsed.host, "example.com");
        assert_eq!(parsed.port, 443);
        assert_eq!(parsed.path, "/");
    }

    #[test]
    fn test_parse_url_with_port() {
        let fetcher = WebFetcher::new();
        let parsed = fetcher.parse_url("http://localhost:8080/api/data").unwrap();
        assert_eq!(parsed.host, "localhost");
        assert_eq!(parsed.port, 8080);
        assert_eq!(parsed.path, "/api/data");
    }

    #[test]
    fn test_parse_url_no_path() {
        let fetcher = WebFetcher::new();
        let parsed = fetcher.parse_url("http://example.com").unwrap();
        assert_eq!(parsed.path, "/");
    }

    #[test]
    fn test_invalid_url() {
        let fetcher = WebFetcher::new();
        assert!(fetcher.parse_url("ftp://example.com").is_err());
        assert!(fetcher.parse_url("example.com").is_err());
    }

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
}
