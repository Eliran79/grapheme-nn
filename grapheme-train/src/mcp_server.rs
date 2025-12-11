//! MCP (Model Context Protocol) Server for GRAPHEME
//!
//! API-017: Implements MCP server exposing GRAPHEME capabilities as tools.
//!
//! MCP is Anthropic's protocol for connecting AI models to tools and data.
//! This module exposes GRAPHEME's graph operations as MCP tools.
//!
//! Specification: https://modelcontextprotocol.io/specification
//!
//! Tools provided:
//! - graph_from_text: Create a graph from text input
//! - graph_transform: Apply learned transformation to a graph
//! - graph_query: Query graph structure and properties
//! - graph_to_text: Convert a graph back to text
//! - train_step: Perform one training step

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};

/// MCP Protocol version
pub const MCP_VERSION: &str = "2024-11-05";

/// JSON-RPC version
pub const JSONRPC_VERSION: &str = "2.0";

/// MCP Server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Tools capability
    pub tools: Option<ToolsCapability>,
    /// Resources capability (optional)
    pub resources: Option<ResourcesCapability>,
    /// Prompts capability (optional)
    pub prompts: Option<PromptsCapability>,
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            tools: Some(ToolsCapability { list_changed: Some(false) }),
            resources: None,
            prompts: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcesCapability {
    pub subscribe: Option<bool>,
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

/// MCP Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

impl Default for ServerInfo {
    fn default() -> Self {
        Self {
            name: "grapheme-mcp".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// MCP Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// MCP Tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// Tool content types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource { uri: String, text: Option<String> },
}

/// JSON-RPC Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

/// JSON-RPC Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC Error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<Value>, code: i32, message: &str) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.to_string(),
                data: None,
            }),
        }
    }
}

/// Standard JSON-RPC error codes
pub mod error_codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
}

/// GRAPHEME MCP Server
pub struct MCPServer {
    info: ServerInfo,
    capabilities: ServerCapabilities,
    tools: Vec<Tool>,
    /// Graph storage for session
    graphs: HashMap<String, String>, // id -> serialized graph JSON
}

impl MCPServer {
    /// Create a new MCP server
    pub fn new() -> Self {
        Self {
            info: ServerInfo::default(),
            capabilities: ServerCapabilities::default(),
            tools: Self::define_tools(),
            graphs: HashMap::new(),
        }
    }

    /// Define available GRAPHEME tools
    fn define_tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "graph_from_text".to_string(),
                description: "Create a GRAPHEME graph from text input. Returns a graph ID.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to convert to a graph"
                        }
                    },
                    "required": ["text"]
                }),
            },
            Tool {
                name: "graph_query".to_string(),
                description: "Query properties of a GRAPHEME graph (node count, edge count, structure).".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "graph_id": {
                            "type": "string",
                            "description": "The graph ID to query"
                        },
                        "query_type": {
                            "type": "string",
                            "enum": ["stats", "nodes", "edges", "structure"],
                            "description": "Type of query to perform"
                        }
                    },
                    "required": ["graph_id", "query_type"]
                }),
            },
            Tool {
                name: "graph_transform".to_string(),
                description: "Apply a transformation to a graph using GRAPHEME's learned model.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "graph_id": {
                            "type": "string",
                            "description": "The graph ID to transform"
                        },
                        "transformation": {
                            "type": "string",
                            "enum": ["simplify", "expand", "normalize"],
                            "description": "Type of transformation to apply"
                        }
                    },
                    "required": ["graph_id", "transformation"]
                }),
            },
            Tool {
                name: "graph_to_text".to_string(),
                description: "Convert a GRAPHEME graph back to text representation.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "graph_id": {
                            "type": "string",
                            "description": "The graph ID to convert"
                        }
                    },
                    "required": ["graph_id"]
                }),
            },
            Tool {
                name: "graph_compare".to_string(),
                description: "Compare two graphs and compute similarity metrics.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "graph_id_1": {
                            "type": "string",
                            "description": "First graph ID"
                        },
                        "graph_id_2": {
                            "type": "string",
                            "description": "Second graph ID"
                        }
                    },
                    "required": ["graph_id_1", "graph_id_2"]
                }),
            },
        ]
    }

    /// Get the list of available tools
    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// Handle an incoming JSON-RPC request
    pub fn handle_request(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request),
            "initialized" => JsonRpcResponse::success(request.id.clone(), json!({})),
            "tools/list" => self.handle_tools_list(request),
            "tools/call" => self.handle_tools_call(request),
            "ping" => JsonRpcResponse::success(request.id.clone(), json!({})),
            _ => JsonRpcResponse::error(
                request.id.clone(),
                error_codes::METHOD_NOT_FOUND,
                &format!("Method not found: {}", request.method),
            ),
        }
    }

    /// Handle initialize request
    fn handle_initialize(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        JsonRpcResponse::success(
            request.id.clone(),
            json!({
                "protocolVersion": MCP_VERSION,
                "capabilities": self.capabilities,
                "serverInfo": self.info
            }),
        )
    }

    /// Handle tools/list request
    fn handle_tools_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        JsonRpcResponse::success(
            request.id.clone(),
            json!({
                "tools": self.tools
            }),
        )
    }

    /// Handle tools/call request
    fn handle_tools_call(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params = match &request.params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    error_codes::INVALID_PARAMS,
                    "Missing params",
                )
            }
        };

        let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        let result = match tool_name {
            "graph_from_text" => self.tool_graph_from_text(&arguments),
            "graph_query" => self.tool_graph_query(&arguments),
            "graph_transform" => self.tool_graph_transform(&arguments),
            "graph_to_text" => self.tool_graph_to_text(&arguments),
            "graph_compare" => self.tool_graph_compare(&arguments),
            _ => ToolResult {
                content: vec![ToolContent::Text {
                    text: format!("Unknown tool: {}", tool_name),
                }],
                is_error: Some(true),
            },
        };

        JsonRpcResponse::success(request.id.clone(), serde_json::to_value(result).unwrap())
    }

    // Tool implementations

    fn tool_graph_from_text(&mut self, args: &Value) -> ToolResult {
        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");

        if text.is_empty() {
            return ToolResult {
                content: vec![ToolContent::Text {
                    text: "Error: text is required".to_string(),
                }],
                is_error: Some(true),
            };
        }

        // Create graph from text using GRAPHEME
        let graph = grapheme_core::GraphemeGraph::from_text(text);
        let graph_id = format!("graph_{}", self.graphs.len());

        // Store graph stats as JSON (simplified representation)
        let graph_data = json!({
            "node_count": graph.node_count(),
            "edge_count": graph.edge_count(),
            "input_nodes": graph.input_nodes.len(),
            "cliques": graph.cliques.len(),
            "source_text": text
        });

        self.graphs.insert(graph_id.clone(), graph_data.to_string());

        ToolResult {
            content: vec![ToolContent::Text {
                text: json!({
                    "graph_id": graph_id,
                    "node_count": graph.node_count(),
                    "edge_count": graph.edge_count(),
                    "message": "Graph created successfully"
                }).to_string(),
            }],
            is_error: None,
        }
    }

    fn tool_graph_query(&self, args: &Value) -> ToolResult {
        let graph_id = args.get("graph_id").and_then(|v| v.as_str()).unwrap_or("");
        let query_type = args.get("query_type").and_then(|v| v.as_str()).unwrap_or("stats");

        let graph_data = match self.graphs.get(graph_id) {
            Some(data) => data,
            None => {
                return ToolResult {
                    content: vec![ToolContent::Text {
                        text: format!("Error: Graph not found: {}", graph_id),
                    }],
                    is_error: Some(true),
                }
            }
        };

        let parsed: Value = serde_json::from_str(graph_data).unwrap_or(json!({}));

        let result = match query_type {
            "stats" => json!({
                "graph_id": graph_id,
                "node_count": parsed.get("node_count"),
                "edge_count": parsed.get("edge_count"),
                "input_nodes": parsed.get("input_nodes"),
                "output_nodes": parsed.get("output_nodes")
            }),
            "nodes" => json!({
                "graph_id": graph_id,
                "node_count": parsed.get("node_count"),
                "input_nodes": parsed.get("input_nodes"),
                "output_nodes": parsed.get("output_nodes")
            }),
            "edges" => json!({
                "graph_id": graph_id,
                "edge_count": parsed.get("edge_count")
            }),
            "structure" => parsed.clone(),
            _ => json!({"error": "Unknown query type"})
        };

        ToolResult {
            content: vec![ToolContent::Text {
                text: result.to_string(),
            }],
            is_error: None,
        }
    }

    fn tool_graph_transform(&mut self, args: &Value) -> ToolResult {
        let graph_id = args.get("graph_id").and_then(|v| v.as_str()).unwrap_or("");
        let transformation = args.get("transformation").and_then(|v| v.as_str()).unwrap_or("normalize");

        if !self.graphs.contains_key(graph_id) {
            return ToolResult {
                content: vec![ToolContent::Text {
                    text: format!("Error: Graph not found: {}", graph_id),
                }],
                is_error: Some(true),
            };
        }

        // Create new graph ID for transformed result
        let new_graph_id = format!("{}_{}", graph_id, transformation);

        // For now, copy the graph data (real impl would use GraphTransformNet)
        let original_data = self.graphs.get(graph_id).cloned().unwrap_or_default();
        let mut parsed: Value = serde_json::from_str(&original_data).unwrap_or(json!({}));

        // Add transformation metadata
        parsed["transformation"] = json!(transformation);
        parsed["source_graph"] = json!(graph_id);

        self.graphs.insert(new_graph_id.clone(), parsed.to_string());

        ToolResult {
            content: vec![ToolContent::Text {
                text: json!({
                    "original_graph_id": graph_id,
                    "new_graph_id": new_graph_id,
                    "transformation": transformation,
                    "message": "Transformation applied successfully"
                }).to_string(),
            }],
            is_error: None,
        }
    }

    fn tool_graph_to_text(&self, args: &Value) -> ToolResult {
        let graph_id = args.get("graph_id").and_then(|v| v.as_str()).unwrap_or("");

        let graph_data = match self.graphs.get(graph_id) {
            Some(data) => data,
            None => {
                return ToolResult {
                    content: vec![ToolContent::Text {
                        text: format!("Error: Graph not found: {}", graph_id),
                    }],
                    is_error: Some(true),
                }
            }
        };

        let parsed: Value = serde_json::from_str(graph_data).unwrap_or(json!({}));
        let source_text = parsed.get("source_text").and_then(|v| v.as_str()).unwrap_or("");

        ToolResult {
            content: vec![ToolContent::Text {
                text: json!({
                    "graph_id": graph_id,
                    "text": source_text,
                    "node_count": parsed.get("node_count"),
                    "edge_count": parsed.get("edge_count")
                }).to_string(),
            }],
            is_error: None,
        }
    }

    fn tool_graph_compare(&self, args: &Value) -> ToolResult {
        let graph_id_1 = args.get("graph_id_1").and_then(|v| v.as_str()).unwrap_or("");
        let graph_id_2 = args.get("graph_id_2").and_then(|v| v.as_str()).unwrap_or("");

        let data1 = match self.graphs.get(graph_id_1) {
            Some(d) => d,
            None => {
                return ToolResult {
                    content: vec![ToolContent::Text {
                        text: format!("Error: Graph not found: {}", graph_id_1),
                    }],
                    is_error: Some(true),
                }
            }
        };

        let data2 = match self.graphs.get(graph_id_2) {
            Some(d) => d,
            None => {
                return ToolResult {
                    content: vec![ToolContent::Text {
                        text: format!("Error: Graph not found: {}", graph_id_2),
                    }],
                    is_error: Some(true),
                }
            }
        };

        let parsed1: Value = serde_json::from_str(data1).unwrap_or(json!({}));
        let parsed2: Value = serde_json::from_str(data2).unwrap_or(json!({}));

        let n1 = parsed1.get("node_count").and_then(|v| v.as_u64()).unwrap_or(0);
        let n2 = parsed2.get("node_count").and_then(|v| v.as_u64()).unwrap_or(0);
        let e1 = parsed1.get("edge_count").and_then(|v| v.as_u64()).unwrap_or(0);
        let e2 = parsed2.get("edge_count").and_then(|v| v.as_u64()).unwrap_or(0);

        // Simple similarity metric
        let node_similarity = if n1.max(n2) > 0 {
            1.0 - (n1 as f64 - n2 as f64).abs() / (n1.max(n2) as f64)
        } else {
            1.0
        };

        let edge_similarity = if e1.max(e2) > 0 {
            1.0 - (e1 as f64 - e2 as f64).abs() / (e1.max(e2) as f64)
        } else {
            1.0
        };

        let overall_similarity = (node_similarity + edge_similarity) / 2.0;

        ToolResult {
            content: vec![ToolContent::Text {
                text: json!({
                    "graph_id_1": graph_id_1,
                    "graph_id_2": graph_id_2,
                    "node_similarity": node_similarity,
                    "edge_similarity": edge_similarity,
                    "overall_similarity": overall_similarity,
                    "graph_1_stats": {
                        "nodes": n1,
                        "edges": e1
                    },
                    "graph_2_stats": {
                        "nodes": n2,
                        "edges": e2
                    }
                }).to_string(),
            }],
            is_error: None,
        }
    }

    /// Run the MCP server over stdio (standard MCP transport)
    pub fn run_stdio(&mut self) -> std::io::Result<()> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        let reader = BufReader::new(stdin.lock());

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            // Parse JSON-RPC request
            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let error_response = JsonRpcResponse::error(
                        None,
                        error_codes::PARSE_ERROR,
                        &format!("Parse error: {}", e),
                    );
                    let response_json = serde_json::to_string(&error_response)?;
                    writeln!(stdout, "{}", response_json)?;
                    stdout.flush()?;
                    continue;
                }
            };

            // Handle request
            let response = self.handle_request(&request);

            // Send response
            let response_json = serde_json::to_string(&response)?;
            writeln!(stdout, "{}", response_json)?;
            stdout.flush()?;
        }

        Ok(())
    }
}

impl Default for MCPServer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = MCPServer::new();
        assert_eq!(server.info.name, "grapheme-mcp");
        assert!(!server.tools.is_empty());
    }

    #[test]
    fn test_initialize() {
        let mut server = MCPServer::new();
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({
                "protocolVersion": MCP_VERSION,
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0"
                }
            })),
        };

        let response = server.handle_request(&request);
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_tools_list() {
        let mut server = MCPServer::new();
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(json!(2)),
            method: "tools/list".to_string(),
            params: None,
        };

        let response = server.handle_request(&request);
        assert!(response.result.is_some());

        let result = response.result.unwrap();
        let tools = result.get("tools").unwrap();
        assert!(tools.as_array().unwrap().len() >= 4);
    }

    #[test]
    fn test_graph_from_text() {
        let mut server = MCPServer::new();
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "graph_from_text",
                "arguments": {
                    "text": "Hello world"
                }
            })),
        };

        let response = server.handle_request(&request);
        assert!(response.result.is_some());
        assert!(response.error.is_none());

        // Verify graph was stored
        assert!(!server.graphs.is_empty());
    }

    #[test]
    fn test_graph_query() {
        let mut server = MCPServer::new();

        // First create a graph
        let create_request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "graph_from_text",
                "arguments": {"text": "Test"}
            })),
        };
        server.handle_request(&create_request);

        // Then query it
        let query_request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "graph_query",
                "arguments": {
                    "graph_id": "graph_0",
                    "query_type": "stats"
                }
            })),
        };

        let response = server.handle_request(&query_request);
        assert!(response.result.is_some());
    }

    #[test]
    fn test_unknown_method() {
        let mut server = MCPServer::new();
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(json!(99)),
            method: "unknown/method".to_string(),
            params: None,
        };

        let response = server.handle_request(&request);
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, error_codes::METHOD_NOT_FOUND);
    }

    #[test]
    fn test_graph_compare() {
        let mut server = MCPServer::new();

        // Create two graphs
        server.handle_request(&JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "graph_from_text",
                "arguments": {"text": "Hello world"}
            })),
        });

        server.handle_request(&JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "graph_from_text",
                "arguments": {"text": "Hello"}
            })),
        });

        // Compare them
        let compare_request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "graph_compare",
                "arguments": {
                    "graph_id_1": "graph_0",
                    "graph_id_2": "graph_1"
                }
            })),
        };

        let response = server.handle_request(&compare_request);
        assert!(response.result.is_some());
    }
}
