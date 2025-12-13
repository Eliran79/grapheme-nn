//! MCP (Model Context Protocol) Client for GRAPHEME
//!
//! Integration-004: Implements MCP client for connecting to external tool servers.
//!
//! This module enables GRAPHEME to consume tools from external MCP servers,
//! allowing integration with various AI tool providers and services.
//!
//! Specification: https://modelcontextprotocol.io/specification

use crate::mcp_server::{
    error_codes, JsonRpcError, JsonRpcRequest, JsonRpcResponse, ServerCapabilities,
    ServerInfo, Tool, ToolContent, ToolResult, JSONRPC_VERSION, MCP_VERSION,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// MCP Client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPClientConfig {
    /// Connection timeout in milliseconds
    pub timeout_ms: u64,
    /// Auto-reconnect on failure
    pub auto_reconnect: bool,
    /// Maximum reconnection attempts
    pub max_reconnect_attempts: u32,
    /// Request retry count
    pub retry_count: u32,
}

impl Default for MCPClientConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 30000,
            auto_reconnect: true,
            max_reconnect_attempts: 3,
            retry_count: 2,
        }
    }
}

/// Client information sent during initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

impl Default for ClientInfo {
    fn default() -> Self {
        Self {
            name: "grapheme-mcp-client".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Client capabilities (what this client supports)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClientCapabilities {
    /// Roots capability (file system roots)
    pub roots: Option<RootsCapability>,
    /// Sampling capability (for LLM sampling)
    pub sampling: Option<SamplingCapability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootsCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingCapability {}

/// Connection state
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Error(String),
}

/// MCP Client error types
#[derive(Debug, Clone)]
pub enum MCPClientError {
    /// Connection failed
    ConnectionFailed(String),
    /// Request timeout
    Timeout,
    /// Protocol error (invalid response)
    ProtocolError(String),
    /// Server returned an error
    ServerError(JsonRpcError),
    /// Tool not found
    ToolNotFound(String),
    /// Invalid parameters
    InvalidParams(String),
    /// IO error
    IoError(String),
}

impl std::fmt::Display for MCPClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            Self::Timeout => write!(f, "Request timeout"),
            Self::ProtocolError(msg) => write!(f, "Protocol error: {}", msg),
            Self::ServerError(err) => write!(f, "Server error {}: {}", err.code, err.message),
            Self::ToolNotFound(name) => write!(f, "Tool not found: {}", name),
            Self::InvalidParams(msg) => write!(f, "Invalid params: {}", msg),
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

/// Transport for MCP communication
pub trait MCPTransport: Send {
    /// Send a request and receive a response
    fn send_request(&mut self, request: &JsonRpcRequest) -> Result<JsonRpcResponse, MCPClientError>;

    /// Send a notification (no response expected)
    fn send_notification(&mut self, request: &JsonRpcRequest) -> Result<(), MCPClientError>;

    /// Close the transport
    fn close(&mut self) -> Result<(), MCPClientError>;
}

/// Stdio transport for subprocess-based MCP servers
pub struct StdioTransport {
    process: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout_reader: Option<BufReader<ChildStdout>>,
    #[allow(dead_code)]
    request_id: u64,
}

impl StdioTransport {
    /// Create a new stdio transport by spawning a command
    pub fn spawn(command: &str, args: &[&str]) -> Result<Self, MCPClientError> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| MCPClientError::ConnectionFailed(e.to_string()))?;

        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stdout_reader = stdout.map(BufReader::new);

        Ok(Self {
            process: Some(child),
            stdin,
            stdout_reader,
            request_id: 0,
        })
    }

    #[allow(dead_code)]
    fn next_id(&mut self) -> u64 {
        self.request_id += 1;
        self.request_id
    }
}

impl MCPTransport for StdioTransport {
    fn send_request(&mut self, request: &JsonRpcRequest) -> Result<JsonRpcResponse, MCPClientError> {
        let stdin = self.stdin.as_mut()
            .ok_or_else(|| MCPClientError::IoError("stdin not available".to_string()))?;
        let reader = self.stdout_reader.as_mut()
            .ok_or_else(|| MCPClientError::IoError("stdout not available".to_string()))?;

        // Send request
        let request_json = serde_json::to_string(request)
            .map_err(|e| MCPClientError::ProtocolError(e.to_string()))?;
        writeln!(stdin, "{}", request_json)
            .map_err(|e| MCPClientError::IoError(e.to_string()))?;
        stdin.flush()
            .map_err(|e| MCPClientError::IoError(e.to_string()))?;

        // Read response
        let mut response_line = String::new();
        reader.read_line(&mut response_line)
            .map_err(|e| MCPClientError::IoError(e.to_string()))?;

        // Parse response
        let response: JsonRpcResponse = serde_json::from_str(&response_line)
            .map_err(|e| MCPClientError::ProtocolError(format!("Invalid response: {}", e)))?;

        Ok(response)
    }

    fn send_notification(&mut self, request: &JsonRpcRequest) -> Result<(), MCPClientError> {
        let stdin = self.stdin.as_mut()
            .ok_or_else(|| MCPClientError::IoError("stdin not available".to_string()))?;

        let request_json = serde_json::to_string(request)
            .map_err(|e| MCPClientError::ProtocolError(e.to_string()))?;
        writeln!(stdin, "{}", request_json)
            .map_err(|e| MCPClientError::IoError(e.to_string()))?;
        stdin.flush()
            .map_err(|e| MCPClientError::IoError(e.to_string()))?;

        Ok(())
    }

    fn close(&mut self) -> Result<(), MCPClientError> {
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
        self.stdin = None;
        self.stdout_reader = None;
        Ok(())
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// In-memory transport for testing (direct function calls)
pub struct InMemoryTransport {
    handler: Box<dyn FnMut(&JsonRpcRequest) -> JsonRpcResponse + Send>,
}

impl InMemoryTransport {
    /// Create a new in-memory transport with a request handler
    pub fn new<F>(handler: F) -> Self
    where
        F: FnMut(&JsonRpcRequest) -> JsonRpcResponse + Send + 'static,
    {
        Self {
            handler: Box::new(handler),
        }
    }
}

impl MCPTransport for InMemoryTransport {
    fn send_request(&mut self, request: &JsonRpcRequest) -> Result<JsonRpcResponse, MCPClientError> {
        Ok((self.handler)(request))
    }

    fn send_notification(&mut self, request: &JsonRpcRequest) -> Result<(), MCPClientError> {
        let _ = (self.handler)(request);
        Ok(())
    }

    fn close(&mut self) -> Result<(), MCPClientError> {
        Ok(())
    }
}

/// MCP Client for connecting to external tool servers
pub struct MCPClient {
    #[allow(dead_code)]
    config: MCPClientConfig,
    client_info: ClientInfo,
    transport: Option<Box<dyn MCPTransport>>,
    state: ConnectionState,
    server_info: Option<ServerInfo>,
    server_capabilities: Option<ServerCapabilities>,
    available_tools: Vec<Tool>,
    request_id: u64,
}

impl MCPClient {
    /// Create a new MCP client with default configuration
    pub fn new() -> Self {
        Self::with_config(MCPClientConfig::default())
    }

    /// Create a new MCP client with custom configuration
    pub fn with_config(config: MCPClientConfig) -> Self {
        Self {
            config,
            client_info: ClientInfo::default(),
            transport: None,
            state: ConnectionState::Disconnected,
            server_info: None,
            server_capabilities: None,
            available_tools: Vec::new(),
            request_id: 0,
        }
    }

    /// Get the current connection state
    pub fn state(&self) -> &ConnectionState {
        &self.state
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.state == ConnectionState::Connected
    }

    /// Get server info (after successful connection)
    pub fn server_info(&self) -> Option<&ServerInfo> {
        self.server_info.as_ref()
    }

    /// Get server capabilities (after successful connection)
    pub fn server_capabilities(&self) -> Option<&ServerCapabilities> {
        self.server_capabilities.as_ref()
    }

    /// Get available tools (after successful connection)
    pub fn available_tools(&self) -> &[Tool] {
        &self.available_tools
    }

    fn next_id(&mut self) -> Value {
        self.request_id += 1;
        json!(self.request_id)
    }

    /// Connect to an MCP server via subprocess
    pub fn connect_stdio(&mut self, command: &str, args: &[&str]) -> Result<(), MCPClientError> {
        self.state = ConnectionState::Connecting;

        // Spawn the subprocess
        let transport = StdioTransport::spawn(command, args)?;
        self.transport = Some(Box::new(transport));

        // Initialize the connection
        self.initialize()
    }

    /// Connect using a custom transport
    pub fn connect_transport(&mut self, transport: Box<dyn MCPTransport>) -> Result<(), MCPClientError> {
        self.state = ConnectionState::Connecting;
        self.transport = Some(transport);
        self.initialize()
    }

    /// Initialize the MCP connection
    fn initialize(&mut self) -> Result<(), MCPClientError> {
        // Send initialize request
        let init_request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(self.next_id()),
            method: "initialize".to_string(),
            params: Some(json!({
                "protocolVersion": MCP_VERSION,
                "capabilities": ClientCapabilities::default(),
                "clientInfo": self.client_info
            })),
        };

        let response = self.send_request(&init_request)?;

        if let Some(error) = response.error {
            self.state = ConnectionState::Error(error.message.clone());
            return Err(MCPClientError::ServerError(error));
        }

        // Parse initialization result
        if let Some(result) = response.result {
            if let Some(server_info) = result.get("serverInfo") {
                self.server_info = serde_json::from_value(server_info.clone()).ok();
            }
            if let Some(capabilities) = result.get("capabilities") {
                self.server_capabilities = serde_json::from_value(capabilities.clone()).ok();
            }
        }

        // Send initialized notification
        let initialized_notification = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: None,
            method: "initialized".to_string(),
            params: None,
        };
        self.send_notification(&initialized_notification)?;

        // Fetch available tools
        self.refresh_tools()?;

        self.state = ConnectionState::Connected;
        Ok(())
    }

    /// Refresh the list of available tools
    pub fn refresh_tools(&mut self) -> Result<(), MCPClientError> {
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(self.next_id()),
            method: "tools/list".to_string(),
            params: None,
        };

        let response = self.send_request(&request)?;

        if let Some(error) = response.error {
            return Err(MCPClientError::ServerError(error));
        }

        if let Some(result) = response.result {
            if let Some(tools) = result.get("tools") {
                self.available_tools = serde_json::from_value(tools.clone())
                    .unwrap_or_default();
            }
        }

        Ok(())
    }

    /// Call a tool on the connected server
    pub fn call_tool(&mut self, name: &str, arguments: Value) -> Result<ToolResult, MCPClientError> {
        if !self.is_connected() {
            return Err(MCPClientError::ConnectionFailed("Not connected".to_string()));
        }

        // Verify tool exists
        if !self.available_tools.iter().any(|t| t.name == name) {
            return Err(MCPClientError::ToolNotFound(name.to_string()));
        }

        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(self.next_id()),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": name,
                "arguments": arguments
            })),
        };

        let response = self.send_request(&request)?;

        if let Some(error) = response.error {
            return Err(MCPClientError::ServerError(error));
        }

        // Parse tool result
        if let Some(result) = response.result {
            let tool_result: ToolResult = serde_json::from_value(result)
                .map_err(|e| MCPClientError::ProtocolError(e.to_string()))?;
            return Ok(tool_result);
        }

        Err(MCPClientError::ProtocolError("Empty response".to_string()))
    }

    /// Call a tool and extract text result
    pub fn call_tool_text(&mut self, name: &str, arguments: Value) -> Result<String, MCPClientError> {
        let result = self.call_tool(name, arguments)?;

        // Check for error
        if result.is_error == Some(true) {
            if let Some(ToolContent::Text { text }) = result.content.first() {
                return Err(MCPClientError::ServerError(JsonRpcError {
                    code: error_codes::INTERNAL_ERROR,
                    message: text.clone(),
                    data: None,
                }));
            }
        }

        // Extract text content
        for content in result.content {
            if let ToolContent::Text { text } = content {
                return Ok(text);
            }
        }

        Err(MCPClientError::ProtocolError("No text content in response".to_string()))
    }

    /// Send a ping to check connection
    pub fn ping(&mut self) -> Result<(), MCPClientError> {
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(self.next_id()),
            method: "ping".to_string(),
            params: None,
        };

        let response = self.send_request(&request)?;

        if response.error.is_some() {
            return Err(MCPClientError::ServerError(response.error.unwrap()));
        }

        Ok(())
    }

    /// Disconnect from the server
    pub fn disconnect(&mut self) -> Result<(), MCPClientError> {
        if let Some(mut transport) = self.transport.take() {
            transport.close()?;
        }
        self.state = ConnectionState::Disconnected;
        self.server_info = None;
        self.server_capabilities = None;
        self.available_tools.clear();
        Ok(())
    }

    fn send_request(&mut self, request: &JsonRpcRequest) -> Result<JsonRpcResponse, MCPClientError> {
        let transport = self.transport.as_mut()
            .ok_or_else(|| MCPClientError::ConnectionFailed("No transport".to_string()))?;
        transport.send_request(request)
    }

    fn send_notification(&mut self, request: &JsonRpcRequest) -> Result<(), MCPClientError> {
        let transport = self.transport.as_mut()
            .ok_or_else(|| MCPClientError::ConnectionFailed("No transport".to_string()))?;
        transport.send_notification(request)
    }
}

impl Default for MCPClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating MCP clients with specific configurations
pub struct MCPClientBuilder {
    config: MCPClientConfig,
    client_info: ClientInfo,
}

impl MCPClientBuilder {
    pub fn new() -> Self {
        Self {
            config: MCPClientConfig::default(),
            client_info: ClientInfo::default(),
        }
    }

    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.config.timeout_ms = ms;
        self
    }

    pub fn auto_reconnect(mut self, enable: bool) -> Self {
        self.config.auto_reconnect = enable;
        self
    }

    pub fn client_name(mut self, name: &str) -> Self {
        self.client_info.name = name.to_string();
        self
    }

    pub fn client_version(mut self, version: &str) -> Self {
        self.client_info.version = version.to_string();
        self
    }

    pub fn build(self) -> MCPClient {
        let mut client = MCPClient::with_config(self.config);
        client.client_info = self.client_info;
        client
    }
}

impl Default for MCPClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry for managing multiple MCP server connections
pub struct MCPServerRegistry {
    servers: HashMap<String, MCPClient>,
}

impl MCPServerRegistry {
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(),
        }
    }

    /// Register a server by name
    pub fn register(&mut self, name: &str, client: MCPClient) {
        self.servers.insert(name.to_string(), client);
    }

    /// Get a server by name
    pub fn get(&self, name: &str) -> Option<&MCPClient> {
        self.servers.get(name)
    }

    /// Get a mutable reference to a server
    pub fn get_mut(&mut self, name: &str) -> Option<&mut MCPClient> {
        self.servers.get_mut(name)
    }

    /// Remove a server
    pub fn remove(&mut self, name: &str) -> Option<MCPClient> {
        self.servers.remove(name)
    }

    /// List all registered server names
    pub fn list_servers(&self) -> Vec<&str> {
        self.servers.keys().map(|s| s.as_str()).collect()
    }

    /// Get all tools from all connected servers
    pub fn all_tools(&self) -> Vec<(&str, &Tool)> {
        let mut tools = Vec::new();
        for (server_name, client) in &self.servers {
            for tool in client.available_tools() {
                tools.push((server_name.as_str(), tool));
            }
        }
        tools
    }

    /// Find which server has a specific tool
    pub fn find_tool(&self, tool_name: &str) -> Option<&str> {
        for (server_name, client) in &self.servers {
            if client.available_tools().iter().any(|t| t.name == tool_name) {
                return Some(server_name);
            }
        }
        None
    }

    /// Call a tool by name, automatically routing to the right server
    pub fn call_tool(&mut self, tool_name: &str, arguments: Value) -> Result<ToolResult, MCPClientError> {
        // Find the server
        let server_name = self.find_tool(tool_name)
            .ok_or_else(|| MCPClientError::ToolNotFound(tool_name.to_string()))?
            .to_string();

        // Call the tool
        let client = self.servers.get_mut(&server_name)
            .ok_or_else(|| MCPClientError::ToolNotFound(tool_name.to_string()))?;

        client.call_tool(tool_name, arguments)
    }

    /// Disconnect all servers
    pub fn disconnect_all(&mut self) -> Result<(), MCPClientError> {
        for client in self.servers.values_mut() {
            client.disconnect()?;
        }
        Ok(())
    }
}

impl Default for MCPServerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_default() {
        let config = MCPClientConfig::default();
        assert_eq!(config.timeout_ms, 30000);
        assert!(config.auto_reconnect);
    }

    #[test]
    fn test_client_info_default() {
        let info = ClientInfo::default();
        assert_eq!(info.name, "grapheme-mcp-client");
    }

    #[test]
    fn test_client_creation() {
        let client = MCPClient::new();
        assert_eq!(*client.state(), ConnectionState::Disconnected);
        assert!(!client.is_connected());
    }

    #[test]
    fn test_client_builder() {
        let client = MCPClientBuilder::new()
            .timeout_ms(5000)
            .auto_reconnect(false)
            .client_name("test-client")
            .build();

        assert_eq!(client.config.timeout_ms, 5000);
        assert!(!client.config.auto_reconnect);
        assert_eq!(client.client_info.name, "test-client");
    }

    #[test]
    fn test_in_memory_transport() {
        let mut transport = InMemoryTransport::new(|req| {
            JsonRpcResponse::success(
                req.id.clone(),
                json!({"echo": req.method}),
            )
        });

        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(json!(1)),
            method: "test".to_string(),
            params: None,
        };

        let response = transport.send_request(&request).unwrap();
        assert!(response.result.is_some());
    }

    #[test]
    fn test_connect_with_mock_server() {
        use crate::mcp_server::MCPServer;

        let mut server = MCPServer::new();

        // Create in-memory transport that routes to server
        let transport = InMemoryTransport::new(move |req| {
            server.handle_request(req)
        });

        let mut client = MCPClient::new();
        let result = client.connect_transport(Box::new(transport));

        assert!(result.is_ok());
        assert!(client.is_connected());
        assert!(client.server_info().is_some());
        assert!(!client.available_tools().is_empty());
    }

    #[test]
    fn test_call_tool_with_mock() {
        use crate::mcp_server::MCPServer;

        let mut server = MCPServer::new();

        let transport = InMemoryTransport::new(move |req| {
            server.handle_request(req)
        });

        let mut client = MCPClient::new();
        client.connect_transport(Box::new(transport)).unwrap();

        // Call graph_from_text tool
        let result = client.call_tool("graph_from_text", json!({
            "text": "Hello world"
        }));

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.is_error.is_none());
    }

    #[test]
    fn test_call_tool_not_found() {
        use crate::mcp_server::MCPServer;

        let mut server = MCPServer::new();

        let transport = InMemoryTransport::new(move |req| {
            server.handle_request(req)
        });

        let mut client = MCPClient::new();
        client.connect_transport(Box::new(transport)).unwrap();

        // Try to call non-existent tool
        let result = client.call_tool("nonexistent_tool", json!({}));

        assert!(result.is_err());
        match result {
            Err(MCPClientError::ToolNotFound(name)) => assert_eq!(name, "nonexistent_tool"),
            _ => panic!("Expected ToolNotFound error"),
        }
    }

    #[test]
    fn test_server_registry() {
        let registry = MCPServerRegistry::new();
        assert!(registry.list_servers().is_empty());
    }

    #[test]
    fn test_registry_with_mock_server() {
        use crate::mcp_server::MCPServer;

        let mut server1 = MCPServer::new();
        let mut server2 = MCPServer::new();

        let transport1 = InMemoryTransport::new(move |req| {
            server1.handle_request(req)
        });
        let transport2 = InMemoryTransport::new(move |req| {
            server2.handle_request(req)
        });

        let mut client1 = MCPClient::new();
        let mut client2 = MCPClient::new();

        client1.connect_transport(Box::new(transport1)).unwrap();
        client2.connect_transport(Box::new(transport2)).unwrap();

        let mut registry = MCPServerRegistry::new();
        registry.register("server1", client1);
        registry.register("server2", client2);

        assert_eq!(registry.list_servers().len(), 2);

        // Find tool
        let server = registry.find_tool("graph_from_text");
        assert!(server.is_some());
    }

    #[test]
    fn test_ping() {
        use crate::mcp_server::MCPServer;

        let mut server = MCPServer::new();

        let transport = InMemoryTransport::new(move |req| {
            server.handle_request(req)
        });

        let mut client = MCPClient::new();
        client.connect_transport(Box::new(transport)).unwrap();

        let result = client.ping();
        assert!(result.is_ok());
    }

    #[test]
    fn test_disconnect() {
        use crate::mcp_server::MCPServer;

        let mut server = MCPServer::new();

        let transport = InMemoryTransport::new(move |req| {
            server.handle_request(req)
        });

        let mut client = MCPClient::new();
        client.connect_transport(Box::new(transport)).unwrap();
        assert!(client.is_connected());

        client.disconnect().unwrap();
        assert!(!client.is_connected());
        assert!(client.available_tools().is_empty());
    }

    #[test]
    fn test_mcp_client_error_display() {
        let error = MCPClientError::ToolNotFound("test".to_string());
        assert_eq!(format!("{}", error), "Tool not found: test");

        let error = MCPClientError::Timeout;
        assert_eq!(format!("{}", error), "Request timeout");
    }
}
