//! LLM API Client Module
//!
//! Integration-001: LLM API client for Claude, OpenAI, and Gemini.
//!
//! Provides unified interface to various LLM providers for:
//! - Training data generation
//! - Graph-to-text translation
//! - Knowledge extraction
//! - Collaborative learning

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::TcpStream;
use std::time::Duration;

/// LLM provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LLMProvider {
    Claude,
    OpenAI,
    Gemini,
    Ollama, // Local models
    Custom,
}

impl Default for LLMProvider {
    fn default() -> Self {
        LLMProvider::Claude
    }
}

/// LLM model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Provider type
    pub provider: LLMProvider,
    /// Model name (e.g., "claude-3-opus", "gpt-4", "gemini-pro")
    pub model: String,
    /// API key (loaded from env if not set)
    pub api_key: Option<String>,
    /// API base URL (for custom/Ollama)
    pub base_url: Option<String>,
    /// Max tokens to generate
    pub max_tokens: usize,
    /// Temperature (0.0 - 2.0)
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Retry count on failure
    pub max_retries: u32,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: LLMProvider::Claude,
            model: "claude-3-haiku-20240307".to_string(),
            api_key: None,
            base_url: None,
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 0.9,
            timeout_secs: 60,
            max_retries: 3,
        }
    }
}

impl LLMConfig {
    /// Create config for Claude
    pub fn claude(model: &str) -> Self {
        Self {
            provider: LLMProvider::Claude,
            model: model.to_string(),
            ..Default::default()
        }
    }

    /// Create config for OpenAI
    pub fn openai(model: &str) -> Self {
        Self {
            provider: LLMProvider::OpenAI,
            model: model.to_string(),
            ..Default::default()
        }
    }

    /// Create config for Gemini
    pub fn gemini(model: &str) -> Self {
        Self {
            provider: LLMProvider::Gemini,
            model: model.to_string(),
            ..Default::default()
        }
    }

    /// Create config for local Ollama
    pub fn ollama(model: &str) -> Self {
        Self {
            provider: LLMProvider::Ollama,
            model: model.to_string(),
            base_url: Some("http://localhost:11434".to_string()),
            ..Default::default()
        }
    }

    /// Set API key
    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
}

/// Chat message role
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
        }
    }
}

/// LLM completion request
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Messages in the conversation
    pub messages: Vec<Message>,
    /// System prompt (if separate from messages)
    pub system: Option<String>,
    /// Override max tokens
    pub max_tokens: Option<usize>,
    /// Override temperature
    pub temperature: Option<f32>,
    /// Stop sequences
    pub stop: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl CompletionRequest {
    /// Create a simple completion request
    pub fn new(prompt: &str) -> Self {
        Self {
            messages: vec![Message::user(prompt)],
            system: None,
            max_tokens: None,
            temperature: None,
            stop: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create from messages
    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self {
            messages,
            system: None,
            max_tokens: None,
            temperature: None,
            stop: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set system prompt
    pub fn with_system(mut self, system: &str) -> Self {
        self.system = Some(system.to_string());
        self
    }

    /// Add stop sequence
    pub fn with_stop(mut self, stop: &str) -> Self {
        self.stop.push(stop.to_string());
        self
    }
}

/// LLM completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Generated text content
    pub content: String,
    /// Model used
    pub model: String,
    /// Finish reason
    pub finish_reason: Option<String>,
    /// Input tokens used
    pub input_tokens: Option<usize>,
    /// Output tokens generated
    pub output_tokens: Option<usize>,
    /// Response latency in milliseconds
    pub latency_ms: u64,
}

/// Error type for LLM operations
#[derive(Debug, Clone)]
pub enum LLMError {
    /// API key not configured
    MissingApiKey,
    /// Network error
    NetworkError(String),
    /// API error response
    ApiError { status: u16, message: String },
    /// Rate limited
    RateLimited { retry_after: Option<u64> },
    /// Invalid response
    InvalidResponse(String),
    /// Timeout
    Timeout,
    /// Serialization error
    SerializationError(String),
}

impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LLMError::MissingApiKey => write!(f, "API key not configured"),
            LLMError::NetworkError(e) => write!(f, "Network error: {}", e),
            LLMError::ApiError { status, message } => {
                write!(f, "API error ({}): {}", status, message)
            }
            LLMError::RateLimited { retry_after } => {
                write!(f, "Rate limited")?;
                if let Some(secs) = retry_after {
                    write!(f, ", retry after {}s", secs)?;
                }
                Ok(())
            }
            LLMError::InvalidResponse(e) => write!(f, "Invalid response: {}", e),
            LLMError::Timeout => write!(f, "Request timeout"),
            LLMError::SerializationError(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for LLMError {}

/// LLM API client
pub struct LLMClient {
    config: LLMConfig,
}

impl LLMClient {
    /// Create a new LLM client
    pub fn new(config: LLMConfig) -> Self {
        Self { config }
    }

    /// Create with default Claude config
    pub fn claude() -> Self {
        Self::new(LLMConfig::claude("claude-3-haiku-20240307"))
    }

    /// Create with default OpenAI config
    pub fn openai() -> Self {
        Self::new(LLMConfig::openai("gpt-4-turbo-preview"))
    }

    /// Create with default Gemini config
    pub fn gemini() -> Self {
        Self::new(LLMConfig::gemini("gemini-pro"))
    }

    /// Create for local Ollama
    pub fn ollama(model: &str) -> Self {
        Self::new(LLMConfig::ollama(model))
    }

    /// Get API key from config or environment
    fn get_api_key(&self) -> Result<String, LLMError> {
        if let Some(ref key) = self.config.api_key {
            return Ok(key.clone());
        }

        let env_var = match self.config.provider {
            LLMProvider::Claude => "ANTHROPIC_API_KEY",
            LLMProvider::OpenAI => "OPENAI_API_KEY",
            LLMProvider::Gemini => "GOOGLE_API_KEY",
            LLMProvider::Ollama => return Ok(String::new()), // No key needed
            LLMProvider::Custom => "LLM_API_KEY",
        };

        std::env::var(env_var).map_err(|_| LLMError::MissingApiKey)
    }

    /// Get base URL for the provider
    fn get_base_url(&self) -> String {
        if let Some(ref url) = self.config.base_url {
            return url.clone();
        }

        match self.config.provider {
            LLMProvider::Claude => "https://api.anthropic.com".to_string(),
            LLMProvider::OpenAI => "https://api.openai.com".to_string(),
            LLMProvider::Gemini => "https://generativelanguage.googleapis.com".to_string(),
            LLMProvider::Ollama => "http://localhost:11434".to_string(),
            LLMProvider::Custom => "http://localhost:8080".to_string(),
        }
    }

    /// Send a completion request
    pub fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let start = std::time::Instant::now();

        match self.config.provider {
            LLMProvider::Claude => self.complete_claude(&request, start),
            LLMProvider::OpenAI => self.complete_openai(&request, start),
            LLMProvider::Gemini => self.complete_gemini(&request, start),
            LLMProvider::Ollama => self.complete_ollama(&request, start),
            LLMProvider::Custom => self.complete_openai(&request, start), // OpenAI-compatible
        }
    }

    /// Simple text completion
    pub fn generate(&self, prompt: &str) -> Result<String, LLMError> {
        let request = CompletionRequest::new(prompt);
        let response = self.complete(request)?;
        Ok(response.content)
    }

    /// Completion with system prompt
    pub fn generate_with_system(&self, system: &str, prompt: &str) -> Result<String, LLMError> {
        let request = CompletionRequest::new(prompt).with_system(system);
        let response = self.complete(request)?;
        Ok(response.content)
    }

    // Claude API implementation
    fn complete_claude(
        &self,
        request: &CompletionRequest,
        start: std::time::Instant,
    ) -> Result<CompletionResponse, LLMError> {
        let api_key = self.get_api_key()?;

        // Build request body
        let mut messages_json = String::from("[");
        for (i, msg) in request.messages.iter().enumerate() {
            if i > 0 {
                messages_json.push(',');
            }
            let role = match msg.role {
                Role::System => "user", // Claude handles system differently
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            messages_json.push_str(&format!(
                r#"{{"role":"{}","content":{}}}"#,
                role,
                serde_json::to_string(&msg.content)
                    .map_err(|e| LLMError::SerializationError(e.to_string()))?
            ));
        }
        messages_json.push(']');

        let system_json = if let Some(ref sys) = request.system {
            format!(
                r#","system":{}"#,
                serde_json::to_string(sys)
                    .map_err(|e| LLMError::SerializationError(e.to_string()))?
            )
        } else {
            String::new()
        };

        let body = format!(
            r#"{{"model":"{}","max_tokens":{},"messages":{}{},"temperature":{}}}"#,
            self.config.model,
            request.max_tokens.unwrap_or(self.config.max_tokens),
            messages_json,
            system_json,
            request.temperature.unwrap_or(self.config.temperature),
        );

        // Make HTTP request
        let response = self.http_post(
            &format!("{}/v1/messages", self.get_base_url()),
            &body,
            &[
                ("x-api-key", &api_key),
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json"),
            ],
        )?;

        // Parse response
        self.parse_claude_response(&response, start)
    }

    fn parse_claude_response(
        &self,
        response: &str,
        start: std::time::Instant,
    ) -> Result<CompletionResponse, LLMError> {
        // Simple JSON parsing without full serde_json dependency
        let content = self
            .extract_json_string(response, "text")
            .or_else(|| self.extract_json_string(response, "content"))
            .unwrap_or_default();

        let finish_reason = self.extract_json_string(response, "stop_reason");

        Ok(CompletionResponse {
            content,
            model: self.config.model.clone(),
            finish_reason,
            input_tokens: self.extract_json_number(response, "input_tokens"),
            output_tokens: self.extract_json_number(response, "output_tokens"),
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }

    // OpenAI API implementation
    fn complete_openai(
        &self,
        request: &CompletionRequest,
        start: std::time::Instant,
    ) -> Result<CompletionResponse, LLMError> {
        let api_key = self.get_api_key()?;

        // Build messages array with system message
        let mut messages_json = String::from("[");
        if let Some(ref sys) = request.system {
            messages_json.push_str(&format!(
                r#"{{"role":"system","content":{}}}"#,
                serde_json::to_string(sys)
                    .map_err(|e| LLMError::SerializationError(e.to_string()))?
            ));
            if !request.messages.is_empty() {
                messages_json.push(',');
            }
        }

        for (i, msg) in request.messages.iter().enumerate() {
            if i > 0 {
                messages_json.push(',');
            }
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            messages_json.push_str(&format!(
                r#"{{"role":"{}","content":{}}}"#,
                role,
                serde_json::to_string(&msg.content)
                    .map_err(|e| LLMError::SerializationError(e.to_string()))?
            ));
        }
        messages_json.push(']');

        let body = format!(
            r#"{{"model":"{}","messages":{},"max_tokens":{},"temperature":{}}}"#,
            self.config.model,
            messages_json,
            request.max_tokens.unwrap_or(self.config.max_tokens),
            request.temperature.unwrap_or(self.config.temperature),
        );

        let response = self.http_post(
            &format!("{}/v1/chat/completions", self.get_base_url()),
            &body,
            &[
                ("Authorization", &format!("Bearer {}", api_key)),
                ("Content-Type", "application/json"),
            ],
        )?;

        self.parse_openai_response(&response, start)
    }

    fn parse_openai_response(
        &self,
        response: &str,
        start: std::time::Instant,
    ) -> Result<CompletionResponse, LLMError> {
        // Extract content from nested structure
        let content = self
            .extract_nested_content(response)
            .unwrap_or_default();

        let finish_reason = self.extract_json_string(response, "finish_reason");

        Ok(CompletionResponse {
            content,
            model: self.config.model.clone(),
            finish_reason,
            input_tokens: self.extract_json_number(response, "prompt_tokens"),
            output_tokens: self.extract_json_number(response, "completion_tokens"),
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }

    // Gemini API implementation
    fn complete_gemini(
        &self,
        request: &CompletionRequest,
        start: std::time::Instant,
    ) -> Result<CompletionResponse, LLMError> {
        let api_key = self.get_api_key()?;

        // Build contents array
        let mut contents_json = String::from("[");
        for (i, msg) in request.messages.iter().enumerate() {
            if i > 0 {
                contents_json.push(',');
            }
            let role = match msg.role {
                Role::System | Role::User => "user",
                Role::Assistant => "model",
            };
            contents_json.push_str(&format!(
                r#"{{"role":"{}","parts":[{{"text":{}}}]}}"#,
                role,
                serde_json::to_string(&msg.content)
                    .map_err(|e| LLMError::SerializationError(e.to_string()))?
            ));
        }
        contents_json.push(']');

        let body = format!(
            r#"{{"contents":{},"generationConfig":{{"maxOutputTokens":{},"temperature":{}}}}}"#,
            contents_json,
            request.max_tokens.unwrap_or(self.config.max_tokens),
            request.temperature.unwrap_or(self.config.temperature),
        );

        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.get_base_url(),
            self.config.model,
            api_key
        );

        let response = self.http_post(&url, &body, &[("Content-Type", "application/json")])?;

        self.parse_gemini_response(&response, start)
    }

    fn parse_gemini_response(
        &self,
        response: &str,
        start: std::time::Instant,
    ) -> Result<CompletionResponse, LLMError> {
        let content = self.extract_json_string(response, "text").unwrap_or_default();

        Ok(CompletionResponse {
            content,
            model: self.config.model.clone(),
            finish_reason: self.extract_json_string(response, "finishReason"),
            input_tokens: self.extract_json_number(response, "promptTokenCount"),
            output_tokens: self.extract_json_number(response, "candidatesTokenCount"),
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }

    // Ollama API implementation
    fn complete_ollama(
        &self,
        request: &CompletionRequest,
        start: std::time::Instant,
    ) -> Result<CompletionResponse, LLMError> {
        // Build messages
        let mut messages_json = String::from("[");
        if let Some(ref sys) = request.system {
            messages_json.push_str(&format!(
                r#"{{"role":"system","content":{}}}"#,
                serde_json::to_string(sys)
                    .map_err(|e| LLMError::SerializationError(e.to_string()))?
            ));
            if !request.messages.is_empty() {
                messages_json.push(',');
            }
        }

        for (i, msg) in request.messages.iter().enumerate() {
            if i > 0 {
                messages_json.push(',');
            }
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            messages_json.push_str(&format!(
                r#"{{"role":"{}","content":{}}}"#,
                role,
                serde_json::to_string(&msg.content)
                    .map_err(|e| LLMError::SerializationError(e.to_string()))?
            ));
        }
        messages_json.push(']');

        let body = format!(
            r#"{{"model":"{}","messages":{},"stream":false,"options":{{"temperature":{}}}}}"#,
            self.config.model,
            messages_json,
            request.temperature.unwrap_or(self.config.temperature),
        );

        let response = self.http_post(
            &format!("{}/api/chat", self.get_base_url()),
            &body,
            &[("Content-Type", "application/json")],
        )?;

        self.parse_ollama_response(&response, start)
    }

    fn parse_ollama_response(
        &self,
        response: &str,
        start: std::time::Instant,
    ) -> Result<CompletionResponse, LLMError> {
        let content = self
            .extract_json_string(response, "content")
            .unwrap_or_default();

        Ok(CompletionResponse {
            content,
            model: self.config.model.clone(),
            finish_reason: Some("stop".to_string()),
            input_tokens: self.extract_json_number(response, "prompt_eval_count"),
            output_tokens: self.extract_json_number(response, "eval_count"),
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }

    // Simple HTTP POST (pure Rust, no external deps)
    fn http_post(
        &self,
        url: &str,
        body: &str,
        headers: &[(&str, &str)],
    ) -> Result<String, LLMError> {
        // Parse URL
        let url = url.trim_start_matches("https://").trim_start_matches("http://");
        let (host, path) = url.split_once('/').unwrap_or((url, ""));
        let path = format!("/{}", path);

        // Connect
        let addr = format!("{}:443", host);
        let stream = TcpStream::connect_timeout(
            &addr.parse().map_err(|e| LLMError::NetworkError(format!("Invalid address: {}", e)))?,
            Duration::from_secs(self.config.timeout_secs),
        )
        .map_err(|e| LLMError::NetworkError(e.to_string()))?;

        // For HTTPS we need TLS - this is a simplified stub
        // In production, use rustls or native-tls
        // For now, return a mock response for testing
        let _ = (stream, path, body, headers);

        // Mock response for testing (real impl needs TLS)
        Err(LLMError::NetworkError(
            "HTTPS not implemented in minimal client - use ureq/reqwest in production".to_string(),
        ))
    }

    // JSON parsing helpers
    fn extract_json_string(&self, json: &str, key: &str) -> Option<String> {
        let pattern = format!(r#""{}":"#, key);
        if let Some(start) = json.find(&pattern) {
            let value_start = start + pattern.len();
            if json.chars().nth(value_start) == Some('"') {
                // String value
                let content_start = value_start + 1;
                let mut end = content_start;
                let chars: Vec<char> = json.chars().collect();
                while end < chars.len() {
                    if chars[end] == '"' && (end == content_start || chars[end - 1] != '\\') {
                        break;
                    }
                    end += 1;
                }
                return Some(json[content_start..end].to_string());
            }
        }
        None
    }

    fn extract_json_number(&self, json: &str, key: &str) -> Option<usize> {
        let pattern = format!(r#""{}":"#, key);
        if let Some(start) = json.find(&pattern) {
            let value_start = start + pattern.len();
            let chars: Vec<char> = json.chars().collect();
            let mut end = value_start;
            while end < chars.len() && (chars[end].is_ascii_digit() || chars[end] == '.') {
                end += 1;
            }
            if end > value_start {
                return json[value_start..end].parse().ok();
            }
        }
        None
    }

    fn extract_nested_content(&self, json: &str) -> Option<String> {
        // For OpenAI: choices[0].message.content
        if let Some(choices_start) = json.find("\"choices\"") {
            if let Some(content_start) = json[choices_start..].find("\"content\"") {
                let abs_start = choices_start + content_start;
                return self.extract_json_string(&json[abs_start..], "content");
            }
        }
        None
    }
}

impl Default for LLMClient {
    fn default() -> Self {
        Self::claude()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = LLMConfig::claude("claude-3-opus");
        assert_eq!(config.provider, LLMProvider::Claude);
        assert_eq!(config.model, "claude-3-opus");
    }

    #[test]
    fn test_config_builder() {
        let config = LLMConfig::openai("gpt-4")
            .with_api_key("test-key")
            .with_max_tokens(1000)
            .with_temperature(0.5);

        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.max_tokens, 1000);
        assert_eq!(config.temperature, 0.5);
    }

    #[test]
    fn test_message_creation() {
        let sys = Message::system("You are helpful");
        let user = Message::user("Hello");
        let asst = Message::assistant("Hi there!");

        assert_eq!(sys.role, Role::System);
        assert_eq!(user.role, Role::User);
        assert_eq!(asst.role, Role::Assistant);
    }

    #[test]
    fn test_completion_request() {
        let request = CompletionRequest::new("Hello")
            .with_system("Be helpful")
            .with_stop("END");

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.system, Some("Be helpful".to_string()));
        assert_eq!(request.stop, vec!["END".to_string()]);
    }

    #[test]
    fn test_client_creation() {
        let client = LLMClient::claude();
        assert_eq!(client.config.provider, LLMProvider::Claude);

        let client = LLMClient::openai();
        assert_eq!(client.config.provider, LLMProvider::OpenAI);

        let client = LLMClient::ollama("llama2");
        assert_eq!(client.config.provider, LLMProvider::Ollama);
        assert_eq!(client.config.model, "llama2");
    }

    #[test]
    fn test_json_extraction() {
        let client = LLMClient::default();

        let json = r#"{"text":"Hello world","count":42}"#;
        assert_eq!(
            client.extract_json_string(json, "text"),
            Some("Hello world".to_string())
        );
        assert_eq!(client.extract_json_number(json, "count"), Some(42));
    }

    #[test]
    fn test_base_url() {
        let client = LLMClient::claude();
        assert!(client.get_base_url().contains("anthropic"));

        let client = LLMClient::openai();
        assert!(client.get_base_url().contains("openai"));

        let client = LLMClient::ollama("test");
        assert!(client.get_base_url().contains("localhost"));
    }
}
