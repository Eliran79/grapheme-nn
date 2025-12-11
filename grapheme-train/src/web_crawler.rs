//! Web Crawler Module
//!
//! Backend-174: Web crawler with rate limiting and robots.txt support.
//!
//! Extends web_fetcher with:
//! - Rate limiting per domain
//! - robots.txt parsing and compliance
//! - URL queue management
//! - Crawl depth control
//! - Link extraction

use crate::web_fetcher::{FetchConfig, WebContent, WebFetcher};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

// ============================================================================
// Crawler Configuration
// ============================================================================

/// Configuration for web crawler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlerConfig {
    /// Minimum delay between requests to same domain (ms)
    pub min_delay_ms: u64,
    /// Maximum crawl depth from seed URLs
    pub max_depth: u32,
    /// Maximum pages to crawl total
    pub max_pages: usize,
    /// Maximum pages per domain
    pub max_pages_per_domain: usize,
    /// Whether to respect robots.txt
    pub respect_robots_txt: bool,
    /// Whether to follow links to external domains
    pub follow_external: bool,
    /// Allowed URL patterns (regex)
    pub allowed_patterns: Vec<String>,
    /// Blocked URL patterns (regex)
    pub blocked_patterns: Vec<String>,
    /// Request timeout (seconds)
    pub timeout_secs: u64,
    /// User agent string
    pub user_agent: String,
}

impl Default for CrawlerConfig {
    fn default() -> Self {
        Self {
            min_delay_ms: 1000,
            max_depth: 3,
            max_pages: 100,
            max_pages_per_domain: 20,
            respect_robots_txt: true,
            follow_external: false,
            allowed_patterns: Vec::new(),
            blocked_patterns: vec![
                r".*\.(jpg|jpeg|png|gif|ico|css|js|pdf|zip|tar|gz)$".to_string(),
                r".*/wp-admin/.*".to_string(),
                r".*/wp-includes/.*".to_string(),
            ],
            timeout_secs: 30,
            user_agent: "GRAPHEME-Crawler/1.0 (+https://github.com/grapheme-nn)".to_string(),
        }
    }
}

impl CrawlerConfig {
    /// Create config for gentle crawling
    pub fn gentle() -> Self {
        Self {
            min_delay_ms: 2000,
            max_depth: 2,
            max_pages: 50,
            max_pages_per_domain: 10,
            ..Default::default()
        }
    }

    /// Create config for aggressive crawling
    pub fn aggressive() -> Self {
        Self {
            min_delay_ms: 500,
            max_depth: 5,
            max_pages: 1000,
            max_pages_per_domain: 100,
            ..Default::default()
        }
    }
}

// ============================================================================
// Robots.txt Support
// ============================================================================

/// Parsed robots.txt file
#[derive(Debug, Clone, Default)]
pub struct RobotsTxt {
    /// Disallowed paths for our user agent
    pub disallowed: Vec<String>,
    /// Allowed paths (overrides disallowed)
    pub allowed: Vec<String>,
    /// Crawl delay in seconds (if specified)
    pub crawl_delay: Option<f64>,
    /// Sitemap URLs
    pub sitemaps: Vec<String>,
}

impl RobotsTxt {
    /// Parse robots.txt content
    pub fn parse(content: &str, user_agent: &str) -> Self {
        let mut result = RobotsTxt::default();
        let mut in_matching_block = false;
        let mut found_specific_ua = false;

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let lower = line.to_lowercase();

            // Check for user-agent directive
            if lower.starts_with("user-agent:") {
                let ua = line[11..].trim().to_lowercase();
                // Check if this applies to us
                if ua == "*" && !found_specific_ua {
                    in_matching_block = true;
                } else if user_agent.to_lowercase().contains(&ua) || ua.contains("grapheme") {
                    in_matching_block = true;
                    found_specific_ua = true;
                    // Clear previous wildcards if we found specific match
                    if found_specific_ua {
                        result.disallowed.clear();
                        result.allowed.clear();
                        result.crawl_delay = None;
                    }
                } else {
                    in_matching_block = false;
                }
                continue;
            }

            if !in_matching_block {
                continue;
            }

            // Parse directives
            if lower.starts_with("disallow:") {
                let path = line[9..].trim();
                if !path.is_empty() {
                    result.disallowed.push(path.to_string());
                }
            } else if lower.starts_with("allow:") {
                let path = line[6..].trim();
                if !path.is_empty() {
                    result.allowed.push(path.to_string());
                }
            } else if lower.starts_with("crawl-delay:") {
                if let Ok(delay) = line[12..].trim().parse::<f64>() {
                    result.crawl_delay = Some(delay);
                }
            } else if lower.starts_with("sitemap:") {
                result.sitemaps.push(line[8..].trim().to_string());
            }
        }

        result
    }

    /// Check if a path is allowed
    pub fn is_allowed(&self, path: &str) -> bool {
        // Check allowed first (they override disallowed)
        for pattern in &self.allowed {
            if Self::path_matches(path, pattern) {
                return true;
            }
        }

        // Check disallowed
        for pattern in &self.disallowed {
            if Self::path_matches(path, pattern) {
                return false;
            }
        }

        // Default allow
        true
    }

    /// Check if path matches a robots.txt pattern
    fn path_matches(path: &str, pattern: &str) -> bool {
        if pattern.is_empty() {
            return false;
        }

        // Simple prefix matching (robots.txt standard)
        if pattern.ends_with('*') {
            path.starts_with(&pattern[..pattern.len() - 1])
        } else if pattern.ends_with('$') {
            path == &pattern[..pattern.len() - 1]
        } else {
            path.starts_with(pattern)
        }
    }
}

// ============================================================================
// Rate Limiter
// ============================================================================

/// Per-domain rate limiter
#[derive(Debug)]
pub struct RateLimiter {
    /// Minimum delay between requests
    min_delay: Duration,
    /// Last request time per domain
    last_request: HashMap<String, Instant>,
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(min_delay_ms: u64) -> Self {
        Self {
            min_delay: Duration::from_millis(min_delay_ms),
            last_request: HashMap::new(),
        }
    }

    /// Get delay needed before next request to domain
    pub fn delay_for(&self, domain: &str) -> Duration {
        if let Some(last) = self.last_request.get(domain) {
            let elapsed = last.elapsed();
            if elapsed < self.min_delay {
                return self.min_delay - elapsed;
            }
        }
        Duration::ZERO
    }

    /// Record a request to a domain
    pub fn record_request(&mut self, domain: &str) {
        self.last_request.insert(domain.to_string(), Instant::now());
    }

    /// Wait for rate limit (blocking)
    pub fn wait_if_needed(&mut self, domain: &str) {
        let delay = self.delay_for(domain);
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }
        self.record_request(domain);
    }

    /// Update minimum delay (e.g., from crawl-delay directive)
    pub fn set_min_delay(&mut self, _domain: &str, delay_secs: f64) {
        // We could store per-domain delays, but for simplicity use max
        let new_delay = Duration::from_secs_f64(delay_secs);
        if new_delay > self.min_delay {
            self.min_delay = new_delay;
        }
    }
}

// ============================================================================
// URL Queue
// ============================================================================

/// A URL in the crawl queue
#[derive(Debug, Clone)]
pub struct QueuedUrl {
    /// URL to crawl
    pub url: String,
    /// Crawl depth from seed
    pub depth: u32,
    /// Source URL that linked to this
    pub source: Option<String>,
}

impl QueuedUrl {
    /// Create new queued URL
    pub fn new(url: &str, depth: u32) -> Self {
        Self {
            url: url.to_string(),
            depth,
            source: None,
        }
    }

    /// Set source URL
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }
}

/// URL queue for crawling
pub struct UrlQueue {
    /// Pending URLs
    queue: VecDeque<QueuedUrl>,
    /// Visited URLs
    visited: HashSet<String>,
    /// URLs per domain
    domain_counts: HashMap<String, usize>,
    /// Maximum pages per domain
    max_per_domain: usize,
}

impl UrlQueue {
    /// Create new URL queue
    pub fn new(max_per_domain: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            visited: HashSet::new(),
            domain_counts: HashMap::new(),
            max_per_domain,
        }
    }

    /// Add URL to queue
    pub fn push(&mut self, queued: QueuedUrl) -> bool {
        let normalized = Self::normalize_url(&queued.url);

        // Skip if already visited
        if self.visited.contains(&normalized) {
            return false;
        }

        // Check domain limit
        if let Some(domain) = Self::extract_domain(&normalized) {
            let count = self.domain_counts.get(&domain).copied().unwrap_or(0);
            if count >= self.max_per_domain {
                return false;
            }
        }

        self.visited.insert(normalized);
        self.queue.push_back(queued);
        true
    }

    /// Pop next URL from queue
    pub fn pop(&mut self) -> Option<QueuedUrl> {
        while let Some(queued) = self.queue.pop_front() {
            // Update domain count
            if let Some(domain) = Self::extract_domain(&queued.url) {
                *self.domain_counts.entry(domain).or_insert(0) += 1;
            }
            return Some(queued);
        }
        None
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Get visited count
    pub fn visited_count(&self) -> usize {
        self.visited.len()
    }

    /// Check if URL was already visited
    pub fn was_visited(&self, url: &str) -> bool {
        self.visited.contains(&Self::normalize_url(url))
    }

    /// Normalize URL for deduplication
    fn normalize_url(url: &str) -> String {
        let mut url = url.to_string();
        // Remove trailing slash
        if url.ends_with('/') && url.len() > 1 {
            url.pop();
        }
        // Remove fragment
        if let Some(idx) = url.find('#') {
            url.truncate(idx);
        }
        url.to_lowercase()
    }

    /// Extract domain from URL
    fn extract_domain(url: &str) -> Option<String> {
        // Simple domain extraction
        let url = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://"))?;
        let end = url.find('/').unwrap_or(url.len());
        Some(url[..end].to_string())
    }
}

// ============================================================================
// Link Extractor
// ============================================================================

/// Extract links from HTML content
pub fn extract_links(html: &str, base_url: &str) -> Vec<String> {
    let mut links = Vec::new();
    let _base_domain = extract_domain(base_url);

    // Simple regex-like extraction for href attributes
    let mut pos = 0;
    while let Some(href_start) = html[pos..].find("href=") {
        let start = pos + href_start + 5;
        pos = start;

        // Get quote character
        let quote = html[start..].chars().next();
        let quote = match quote {
            Some('"') | Some('\'') => quote.unwrap(),
            _ => continue,
        };

        // Find end of URL
        let url_start = start + 1;
        if let Some(url_end) = html[url_start..].find(quote) {
            let url = &html[url_start..url_start + url_end];

            // Resolve relative URLs
            let resolved = resolve_url(base_url, url);
            if let Some(resolved) = resolved {
                links.push(resolved);
            }
        }
    }

    links
}

/// Resolve relative URL against base
fn resolve_url(base: &str, url: &str) -> Option<String> {
    // Skip javascript, mailto, etc.
    if url.starts_with("javascript:") || url.starts_with("mailto:") || url.starts_with("#") {
        return None;
    }

    // Absolute URL
    if url.starts_with("http://") || url.starts_with("https://") {
        return Some(url.to_string());
    }

    // Protocol-relative
    if url.starts_with("//") {
        if base.starts_with("https://") {
            return Some(format!("https:{}", url));
        }
        return Some(format!("http:{}", url));
    }

    // Parse base URL
    let (scheme, rest) = if base.starts_with("https://") {
        ("https://", &base[8..])
    } else if base.starts_with("http://") {
        ("http://", &base[7..])
    } else {
        return None;
    };

    let domain_end = rest.find('/').unwrap_or(rest.len());
    let domain = &rest[..domain_end];
    let path = if domain_end < rest.len() {
        &rest[domain_end..]
    } else {
        "/"
    };

    // Root-relative
    if url.starts_with('/') {
        return Some(format!("{}{}{}", scheme, domain, url));
    }

    // Path-relative
    let base_dir = if let Some(last_slash) = path.rfind('/') {
        &path[..last_slash + 1]
    } else {
        "/"
    };

    Some(format!("{}{}{}{}", scheme, domain, base_dir, url))
}

/// Extract domain from URL
fn extract_domain(url: &str) -> Option<String> {
    let url = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://"))?;
    let end = url.find('/').unwrap_or(url.len());
    Some(url[..end].to_string())
}

// ============================================================================
// Web Crawler
// ============================================================================

/// Crawled page result
#[derive(Debug, Clone)]
pub struct CrawledPage {
    /// Page URL
    pub url: String,
    /// Final URL after redirects
    pub final_url: String,
    /// Page content
    pub content: WebContent,
    /// Crawl depth
    pub depth: u32,
    /// Links found on page
    pub links: Vec<String>,
}

/// Web crawler with rate limiting and robots.txt support
pub struct WebCrawler {
    /// Configuration
    config: CrawlerConfig,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// URL queue
    queue: UrlQueue,
    /// Robots.txt cache
    robots_cache: HashMap<String, RobotsTxt>,
    /// Statistics
    stats: CrawlerStats,
}

/// Crawler statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrawlerStats {
    /// Pages crawled
    pub pages_crawled: usize,
    /// Pages skipped (rate limit, robots.txt, etc.)
    pub pages_skipped: usize,
    /// Errors encountered
    pub errors: usize,
    /// Total bytes downloaded
    pub bytes_downloaded: usize,
    /// Links discovered
    pub links_discovered: usize,
}

impl WebCrawler {
    /// Create new crawler
    pub fn new(config: CrawlerConfig) -> Self {
        let rate_limiter = RateLimiter::new(config.min_delay_ms);
        let queue = UrlQueue::new(config.max_pages_per_domain);

        Self {
            config,
            rate_limiter,
            queue,
            robots_cache: HashMap::new(),
            stats: CrawlerStats::default(),
        }
    }

    /// Create with default config
    pub fn default_crawler() -> Self {
        Self::new(CrawlerConfig::default())
    }

    /// Get statistics
    pub fn stats(&self) -> &CrawlerStats {
        &self.stats
    }

    /// Get queue length
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Add seed URLs
    pub fn add_seeds(&mut self, urls: &[&str]) {
        for url in urls {
            self.queue.push(QueuedUrl::new(url, 0));
        }
    }

    /// Add a single URL
    pub fn add_url(&mut self, url: &str, depth: u32) {
        self.queue.push(QueuedUrl::new(url, depth));
    }

    /// Check if URL is allowed by robots.txt
    fn is_allowed_by_robots(&mut self, url: &str) -> bool {
        if !self.config.respect_robots_txt {
            return true;
        }

        let domain = match extract_domain(url) {
            Some(d) => d,
            None => return true,
        };

        // Check cache
        if !self.robots_cache.contains_key(&domain) {
            // Fetch robots.txt
            let robots_url = format!(
                "{}://{}/robots.txt",
                if url.starts_with("https://") { "https" } else { "http" },
                domain
            );

            let fetcher = WebFetcher::with_config(FetchConfig {
                timeout_secs: 10,
                user_agent: self.config.user_agent.clone(),
                ..Default::default()
            });

            let robots = match fetcher.fetch(&robots_url) {
                Ok(content) => {
                    if let Some(text) = &content.text {
                        let parsed = RobotsTxt::parse(text, &self.config.user_agent);
                        // Update rate limiter if crawl-delay specified
                        if let Some(delay) = parsed.crawl_delay {
                            self.rate_limiter.set_min_delay(&domain, delay);
                        }
                        parsed
                    } else {
                        RobotsTxt::default()
                    }
                }
                Err(_) => RobotsTxt::default(), // Assume allowed if can't fetch
            };

            self.robots_cache.insert(domain.clone(), robots);
        }

        // Check path against robots.txt
        if let Some(robots) = self.robots_cache.get(&domain) {
            let path = url
                .strip_prefix("https://")
                .or_else(|| url.strip_prefix("http://"))
                .and_then(|u| u.find('/').map(|i| &u[i..]))
                .unwrap_or("/");

            return robots.is_allowed(path);
        }

        true
    }

    /// Check if URL matches blocked patterns
    fn is_blocked(&self, url: &str) -> bool {
        for pattern in &self.config.blocked_patterns {
            // Simple glob matching
            if url.contains(pattern.trim_start_matches(".*").trim_end_matches(".*").trim_matches('$')) {
                return true;
            }
        }
        false
    }

    /// Crawl next URL in queue
    pub fn crawl_next(&mut self) -> Option<Result<CrawledPage, CrawlError>> {
        // Check limits
        if self.stats.pages_crawled >= self.config.max_pages {
            return None;
        }

        // Get next URL
        let queued = self.queue.pop()?;

        // Check depth limit
        if queued.depth > self.config.max_depth {
            self.stats.pages_skipped += 1;
            return Some(Err(CrawlError::DepthExceeded));
        }

        // Check blocked patterns
        if self.is_blocked(&queued.url) {
            self.stats.pages_skipped += 1;
            return Some(Err(CrawlError::BlockedPattern));
        }

        // Check robots.txt
        if !self.is_allowed_by_robots(&queued.url) {
            self.stats.pages_skipped += 1;
            return Some(Err(CrawlError::RobotsTxt));
        }

        // Wait for rate limit
        if let Some(domain) = extract_domain(&queued.url) {
            self.rate_limiter.wait_if_needed(&domain);
        }

        // Fetch page
        let fetcher = WebFetcher::with_config(FetchConfig {
            timeout_secs: self.config.timeout_secs,
            user_agent: self.config.user_agent.clone(),
            ..Default::default()
        });

        match fetcher.fetch(&queued.url) {
            Ok(content) => {
                self.stats.pages_crawled += 1;
                self.stats.bytes_downloaded += content.content_length;

                // Extract links
                let links = if content.is_html() {
                    if let Some(text) = &content.text {
                        let extracted = extract_links(text, &content.final_url);
                        self.stats.links_discovered += extracted.len();

                        // Add to queue
                        for link in &extracted {
                            // Check external links
                            let link_domain = extract_domain(link);
                            let base_domain = extract_domain(&queued.url);

                            if !self.config.follow_external {
                                if link_domain != base_domain {
                                    continue;
                                }
                            }

                            self.queue.push(
                                QueuedUrl::new(link, queued.depth + 1)
                                    .with_source(&queued.url)
                            );
                        }

                        extracted
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                Some(Ok(CrawledPage {
                    url: queued.url,
                    final_url: content.final_url.clone(),
                    content,
                    depth: queued.depth,
                    links,
                }))
            }
            Err(e) => {
                self.stats.errors += 1;
                Some(Err(CrawlError::Fetch(format!("{:?}", e))))
            }
        }
    }

    /// Crawl all URLs up to limits
    pub fn crawl_all(&mut self) -> Vec<CrawledPage> {
        let mut pages = Vec::new();

        while let Some(result) = self.crawl_next() {
            if let Ok(page) = result {
                pages.push(page);
            }
        }

        pages
    }
}

/// Crawl errors
#[derive(Debug, Clone)]
pub enum CrawlError {
    /// Depth limit exceeded
    DepthExceeded,
    /// Blocked by robots.txt
    RobotsTxt,
    /// Blocked by pattern
    BlockedPattern,
    /// Fetch error
    Fetch(String),
    /// Rate limited
    RateLimited,
}

impl std::fmt::Display for CrawlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DepthExceeded => write!(f, "Depth limit exceeded"),
            Self::RobotsTxt => write!(f, "Blocked by robots.txt"),
            Self::BlockedPattern => write!(f, "URL matches blocked pattern"),
            Self::Fetch(e) => write!(f, "Fetch error: {}", e),
            Self::RateLimited => write!(f, "Rate limited"),
        }
    }
}

impl std::error::Error for CrawlError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crawler_config_default() {
        let config = CrawlerConfig::default();
        assert_eq!(config.min_delay_ms, 1000);
        assert_eq!(config.max_depth, 3);
        assert!(config.respect_robots_txt);
    }

    #[test]
    fn test_crawler_config_gentle() {
        let config = CrawlerConfig::gentle();
        assert!(config.min_delay_ms > CrawlerConfig::default().min_delay_ms);
    }

    #[test]
    fn test_crawler_config_aggressive() {
        let config = CrawlerConfig::aggressive();
        assert!(config.min_delay_ms < CrawlerConfig::default().min_delay_ms);
    }

    #[test]
    fn test_robots_txt_parse_simple() {
        let content = r#"
User-agent: *
Disallow: /admin
Disallow: /private/
Allow: /admin/public

User-agent: googlebot
Disallow: /
"#;
        let robots = RobotsTxt::parse(content, "GRAPHEME-Crawler");
        assert!(robots.disallowed.contains(&"/admin".to_string()));
        assert!(robots.disallowed.contains(&"/private/".to_string()));
        assert!(robots.allowed.contains(&"/admin/public".to_string()));
    }

    #[test]
    fn test_robots_txt_is_allowed() {
        let mut robots = RobotsTxt::default();
        robots.disallowed.push("/admin".to_string());
        robots.disallowed.push("/private/".to_string());
        robots.allowed.push("/admin/public".to_string());

        assert!(!robots.is_allowed("/admin"));
        assert!(!robots.is_allowed("/admin/secret"));
        assert!(robots.is_allowed("/admin/public"));
        assert!(!robots.is_allowed("/private/data"));
        assert!(robots.is_allowed("/public"));
    }

    #[test]
    fn test_robots_txt_crawl_delay() {
        let content = r#"
User-agent: *
Crawl-delay: 5
Disallow: /admin
"#;
        let robots = RobotsTxt::parse(content, "GRAPHEME-Crawler");
        assert_eq!(robots.crawl_delay, Some(5.0));
    }

    #[test]
    fn test_robots_txt_sitemap() {
        let content = r#"
User-agent: *
Disallow:
Sitemap: https://example.com/sitemap.xml
"#;
        let robots = RobotsTxt::parse(content, "GRAPHEME-Crawler");
        assert!(robots.sitemaps.contains(&"https://example.com/sitemap.xml".to_string()));
    }

    #[test]
    fn test_rate_limiter_creation() {
        let limiter = RateLimiter::new(1000);
        assert_eq!(limiter.delay_for("example.com"), Duration::ZERO);
    }

    #[test]
    fn test_rate_limiter_delay() {
        let mut limiter = RateLimiter::new(100);
        limiter.record_request("example.com");

        // Should have some delay now
        let delay = limiter.delay_for("example.com");
        assert!(delay <= Duration::from_millis(100));

        // Different domain should have no delay
        assert_eq!(limiter.delay_for("other.com"), Duration::ZERO);
    }

    #[test]
    fn test_url_queue_creation() {
        let queue = UrlQueue::new(10);
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_url_queue_push_pop() {
        let mut queue = UrlQueue::new(10);

        assert!(queue.push(QueuedUrl::new("http://example.com", 0)));
        assert_eq!(queue.len(), 1);

        let popped = queue.pop().unwrap();
        assert_eq!(popped.url, "http://example.com");
        assert!(queue.is_empty());
    }

    #[test]
    fn test_url_queue_dedup() {
        let mut queue = UrlQueue::new(10);

        assert!(queue.push(QueuedUrl::new("http://example.com/page", 0)));
        // Same URL shouldn't be added again
        assert!(!queue.push(QueuedUrl::new("http://example.com/page", 0)));
        // With trailing slash (normalized)
        assert!(!queue.push(QueuedUrl::new("http://example.com/page/", 0)));

        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_url_queue_domain_limit() {
        let mut queue = UrlQueue::new(2);

        assert!(queue.push(QueuedUrl::new("http://example.com/1", 0)));
        queue.pop(); // Consume to count domain
        assert!(queue.push(QueuedUrl::new("http://example.com/2", 0)));
        queue.pop();
        // Third should be blocked
        assert!(!queue.push(QueuedUrl::new("http://example.com/3", 0)));

        // Different domain should work
        assert!(queue.push(QueuedUrl::new("http://other.com/1", 0)));
    }

    #[test]
    fn test_extract_links_absolute() {
        let html = r#"<a href="http://example.com/page1">Link 1</a>"#;
        let links = extract_links(html, "http://base.com");
        assert!(links.contains(&"http://example.com/page1".to_string()));
    }

    #[test]
    fn test_extract_links_relative() {
        let html = r#"<a href="/page1">Link 1</a><a href="page2">Link 2</a>"#;
        let links = extract_links(html, "http://example.com/dir/");
        assert!(links.contains(&"http://example.com/page1".to_string()));
        assert!(links.contains(&"http://example.com/dir/page2".to_string()));
    }

    #[test]
    fn test_extract_links_skip_javascript() {
        let html = r#"<a href="javascript:void(0)">Skip</a><a href="/real">Real</a>"#;
        let links = extract_links(html, "http://example.com");
        assert_eq!(links.len(), 1);
        assert!(links.contains(&"http://example.com/real".to_string()));
    }

    #[test]
    fn test_resolve_url_absolute() {
        assert_eq!(
            resolve_url("http://base.com", "http://other.com/page"),
            Some("http://other.com/page".to_string())
        );
    }

    #[test]
    fn test_resolve_url_root_relative() {
        assert_eq!(
            resolve_url("http://example.com/dir/page", "/root"),
            Some("http://example.com/root".to_string())
        );
    }

    #[test]
    fn test_resolve_url_path_relative() {
        assert_eq!(
            resolve_url("http://example.com/dir/page", "sibling"),
            Some("http://example.com/dir/sibling".to_string())
        );
    }

    #[test]
    fn test_crawler_creation() {
        let crawler = WebCrawler::default_crawler();
        assert_eq!(crawler.stats().pages_crawled, 0);
        assert_eq!(crawler.queue_len(), 0);
    }

    #[test]
    fn test_crawler_add_seeds() {
        let mut crawler = WebCrawler::default_crawler();
        crawler.add_seeds(&["http://example.com", "http://test.com"]);
        assert_eq!(crawler.queue_len(), 2);
    }

    #[test]
    fn test_crawl_error_display() {
        assert_eq!(
            format!("{}", CrawlError::DepthExceeded),
            "Depth limit exceeded"
        );
        assert_eq!(
            format!("{}", CrawlError::RobotsTxt),
            "Blocked by robots.txt"
        );
    }

    #[test]
    fn test_queued_url_with_source() {
        let queued = QueuedUrl::new("http://example.com", 1)
            .with_source("http://parent.com");
        assert_eq!(queued.source, Some("http://parent.com".to_string()));
        assert_eq!(queued.depth, 1);
    }
}
