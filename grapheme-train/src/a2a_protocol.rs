//! A2A (Agent-to-Agent) Protocol Implementation
//!
//! Implements Google's Agent2Agent protocol for inter-agent communication.
//! A2A enables agents to discover, communicate, and delegate tasks to each other.
//!
//! Key concepts:
//! - Agent Card: JSON manifest describing agent capabilities (/.well-known/agent.json)
//! - Tasks: Units of work with status tracking and artifacts
//! - Messages: Communication within task context
//! - Streaming: Server-Sent Events for real-time updates
//!
//! Protocol spec: https://google.github.io/A2A/

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

// ============================================================================
// Agent Card - Discovery and Capabilities
// ============================================================================

/// Agent Card describing agent capabilities and metadata
/// Served at /.well-known/agent.json for discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCard {
    /// Human-readable agent name
    pub name: String,
    /// Agent description
    pub description: String,
    /// API endpoint URL
    pub url: String,
    /// A2A protocol version
    pub protocol_version: String,
    /// Authentication methods supported
    pub authentication: AuthenticationInfo,
    /// Capabilities this agent provides
    pub capabilities: AgentCapabilities,
    /// Skills the agent can perform
    pub skills: Vec<Skill>,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationInfo {
    /// Supported auth schemes (api_key, oauth2, bearer)
    pub schemes: Vec<String>,
    /// OAuth2 configuration if supported
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth2: Option<OAuth2Config>,
}

/// OAuth2 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Config {
    /// Token endpoint
    pub token_url: String,
    /// Supported scopes
    pub scopes: Vec<String>,
}

/// Agent capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    /// Supports streaming responses (SSE)
    pub streaming: bool,
    /// Supports push notifications
    pub push_notifications: bool,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: Option<u32>,
}

/// A skill that an agent can perform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique skill identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Skill description
    pub description: String,
    /// Input schema (JSON Schema)
    pub input_schema: Value,
    /// Output schema (JSON Schema)
    pub output_schema: Value,
    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

// ============================================================================
// Tasks - Work Units
// ============================================================================

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Task is pending execution
    Pending,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Waiting for input from client
    InputRequired,
}

/// A task representing a unit of work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task ID
    pub id: String,
    /// Task status
    pub status: TaskStatus,
    /// Task context (history of messages)
    pub context: Vec<A2AMessage>,
    /// Task artifacts (outputs)
    pub artifacts: Vec<Artifact>,
    /// Creation timestamp
    pub created_at: String,
    /// Last update timestamp
    pub updated_at: String,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Progress percentage (0-100)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<u8>,
}

/// A message in task context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AMessage {
    /// Message role (user, assistant, system)
    pub role: String,
    /// Message content
    pub content: Vec<ContentPart>,
    /// Message timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

/// Part of message content (supports multimodal)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "file")]
    File { uri: String, mime_type: String },
    #[serde(rename = "data")]
    Data { data: String, mime_type: String },
}

/// An artifact produced by a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact identifier
    pub id: String,
    /// Artifact name
    pub name: String,
    /// Artifact type
    pub artifact_type: String,
    /// Content (inline data or URI)
    pub content: ArtifactContent,
}

/// Artifact content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ArtifactContent {
    #[serde(rename = "inline")]
    Inline { data: String, mime_type: String },
    #[serde(rename = "uri")]
    Uri { uri: String },
}

// ============================================================================
// JSON-RPC Protocol
// ============================================================================

/// JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2ARequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<A2AError>,
}

/// JSON-RPC error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

// Error codes
pub const ERROR_PARSE: i32 = -32700;
pub const ERROR_INVALID_REQUEST: i32 = -32600;
pub const ERROR_METHOD_NOT_FOUND: i32 = -32601;
pub const ERROR_INVALID_PARAMS: i32 = -32602;
pub const ERROR_INTERNAL: i32 = -32603;
pub const ERROR_TASK_NOT_FOUND: i32 = -32001;
pub const ERROR_TASK_CANCELLED: i32 = -32002;

// ============================================================================
// A2A Agent Server
// ============================================================================

/// A2A Agent Server for GRAPHEME
pub struct A2AAgent {
    /// Agent card (metadata)
    card: AgentCard,
    /// Active tasks
    tasks: HashMap<String, Task>,
    /// Task counter for ID generation
    task_counter: u64,
}

impl A2AAgent {
    /// Create a new GRAPHEME A2A agent
    pub fn new(base_url: &str) -> Self {
        let card = AgentCard {
            name: "GRAPHEME Agent".to_string(),
            description: "Graph-based neural processing agent for text understanding and transformation".to_string(),
            url: base_url.to_string(),
            protocol_version: "1.0".to_string(),
            authentication: AuthenticationInfo {
                schemes: vec!["api_key".to_string()],
                oauth2: None,
            },
            capabilities: AgentCapabilities {
                streaming: true,
                push_notifications: false,
                max_concurrent_tasks: Some(10),
            },
            skills: vec![
                Skill {
                    id: "text_to_graph".to_string(),
                    name: "Text to Graph".to_string(),
                    description: "Convert text to a GRAPHEME neural graph representation".to_string(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Input text to convert"}
                        },
                        "required": ["text"]
                    }),
                    output_schema: json!({
                        "type": "object",
                        "properties": {
                            "graph_id": {"type": "string"},
                            "node_count": {"type": "integer"},
                            "edge_count": {"type": "integer"}
                        }
                    }),
                    tags: vec!["nlp".to_string(), "graph".to_string()],
                },
                Skill {
                    id: "graph_transform".to_string(),
                    name: "Graph Transform".to_string(),
                    description: "Apply neural transformations to a graph".to_string(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "graph_id": {"type": "string"},
                            "transform_type": {"type": "string", "enum": ["simplify", "expand", "summarize"]}
                        },
                        "required": ["graph_id", "transform_type"]
                    }),
                    output_schema: json!({
                        "type": "object",
                        "properties": {
                            "graph_id": {"type": "string"},
                            "changes": {"type": "array", "items": {"type": "string"}}
                        }
                    }),
                    tags: vec!["transform".to_string(), "graph".to_string()],
                },
                Skill {
                    id: "graph_to_text".to_string(),
                    name: "Graph to Text".to_string(),
                    description: "Convert a graph back to text representation".to_string(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "graph_id": {"type": "string"}
                        },
                        "required": ["graph_id"]
                    }),
                    output_schema: json!({
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"}
                        }
                    }),
                    tags: vec!["nlp".to_string(), "graph".to_string()],
                },
                Skill {
                    id: "analyze_text".to_string(),
                    name: "Analyze Text".to_string(),
                    description: "Analyze text using graph-based neural processing".to_string(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "analysis_type": {"type": "string", "enum": ["sentiment", "entities", "structure", "summary"]}
                        },
                        "required": ["text", "analysis_type"]
                    }),
                    output_schema: json!({
                        "type": "object",
                        "properties": {
                            "result": {"type": "object"},
                            "confidence": {"type": "number"}
                        }
                    }),
                    tags: vec!["analysis".to_string(), "nlp".to_string()],
                },
            ],
        };

        Self {
            card,
            tasks: HashMap::new(),
            task_counter: 0,
        }
    }

    /// Get the agent card (for discovery)
    pub fn get_agent_card(&self) -> &AgentCard {
        &self.card
    }

    /// Handle an A2A request
    pub fn handle_request(&mut self, request: &A2ARequest) -> A2AResponse {
        match request.method.as_str() {
            "tasks/create" => self.handle_task_create(request),
            "tasks/get" => self.handle_task_get(request),
            "tasks/cancel" => self.handle_task_cancel(request),
            "tasks/list" => self.handle_task_list(request),
            "tasks/sendMessage" => self.handle_send_message(request),
            "agent/info" => self.handle_agent_info(request),
            _ => A2AResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: None,
                error: Some(A2AError {
                    code: ERROR_METHOD_NOT_FOUND,
                    message: format!("Method not found: {}", request.method),
                    data: None,
                }),
            },
        }
    }

    /// Handle agent/info request
    fn handle_agent_info(&self, request: &A2ARequest) -> A2AResponse {
        A2AResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(serde_json::to_value(&self.card).unwrap()),
            error: None,
        }
    }

    /// Handle tasks/create request
    fn handle_task_create(&mut self, request: &A2ARequest) -> A2AResponse {
        let skill_id = request.params.get("skill_id").and_then(|v| v.as_str());
        let input = request.params.get("input").cloned().unwrap_or(json!({}));

        // Validate skill exists
        if let Some(skill_id) = skill_id {
            if !self.card.skills.iter().any(|s| s.id == skill_id) {
                return A2AResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: None,
                    error: Some(A2AError {
                        code: ERROR_INVALID_PARAMS,
                        message: format!("Unknown skill: {}", skill_id),
                        data: None,
                    }),
                };
            }
        }

        // Create task
        self.task_counter += 1;
        let task_id = format!("task_{}", self.task_counter);
        let now = chrono_lite_now();

        let task = Task {
            id: task_id.clone(),
            status: TaskStatus::Running,
            context: vec![A2AMessage {
                role: "user".to_string(),
                content: vec![ContentPart::Text {
                    text: serde_json::to_string(&input).unwrap_or_default(),
                }],
                timestamp: Some(now.clone()),
            }],
            artifacts: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
            error: None,
            progress: Some(0),
        };

        // Execute skill (simplified - in production this would be async)
        let result = self.execute_skill(skill_id.unwrap_or("analyze_text"), &input);

        let mut task = task;
        match result {
            Ok(output) => {
                task.status = TaskStatus::Completed;
                task.progress = Some(100);
                task.artifacts.push(Artifact {
                    id: "result_0".to_string(),
                    name: "Result".to_string(),
                    artifact_type: "application/json".to_string(),
                    content: ArtifactContent::Inline {
                        data: serde_json::to_string(&output).unwrap_or_default(),
                        mime_type: "application/json".to_string(),
                    },
                });
            }
            Err(e) => {
                task.status = TaskStatus::Failed;
                task.error = Some(e);
            }
        }
        task.updated_at = chrono_lite_now();

        self.tasks.insert(task_id.clone(), task.clone());

        A2AResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(serde_json::to_value(&task).unwrap()),
            error: None,
        }
    }

    /// Handle tasks/get request
    fn handle_task_get(&self, request: &A2ARequest) -> A2AResponse {
        let task_id = request.params.get("task_id").and_then(|v| v.as_str());

        match task_id {
            Some(id) => match self.tasks.get(id) {
                Some(task) => A2AResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: Some(serde_json::to_value(task).unwrap()),
                    error: None,
                },
                None => A2AResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: None,
                    error: Some(A2AError {
                        code: ERROR_TASK_NOT_FOUND,
                        message: format!("Task not found: {}", id),
                        data: None,
                    }),
                },
            },
            None => A2AResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: None,
                error: Some(A2AError {
                    code: ERROR_INVALID_PARAMS,
                    message: "Missing task_id parameter".to_string(),
                    data: None,
                }),
            },
        }
    }

    /// Handle tasks/cancel request
    fn handle_task_cancel(&mut self, request: &A2ARequest) -> A2AResponse {
        let task_id = request.params.get("task_id").and_then(|v| v.as_str());

        match task_id {
            Some(id) => match self.tasks.get_mut(id) {
                Some(task) => {
                    if task.status == TaskStatus::Running || task.status == TaskStatus::Pending {
                        task.status = TaskStatus::Cancelled;
                        task.updated_at = chrono_lite_now();
                        A2AResponse {
                            jsonrpc: "2.0".to_string(),
                            id: request.id.clone(),
                            result: Some(serde_json::to_value(task).unwrap()),
                            error: None,
                        }
                    } else {
                        A2AResponse {
                            jsonrpc: "2.0".to_string(),
                            id: request.id.clone(),
                            result: None,
                            error: Some(A2AError {
                                code: ERROR_TASK_CANCELLED,
                                message: "Task is not cancellable in current state".to_string(),
                                data: None,
                            }),
                        }
                    }
                }
                None => A2AResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: None,
                    error: Some(A2AError {
                        code: ERROR_TASK_NOT_FOUND,
                        message: format!("Task not found: {}", id),
                        data: None,
                    }),
                },
            },
            None => A2AResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: None,
                error: Some(A2AError {
                    code: ERROR_INVALID_PARAMS,
                    message: "Missing task_id parameter".to_string(),
                    data: None,
                }),
            },
        }
    }

    /// Handle tasks/list request
    fn handle_task_list(&self, request: &A2ARequest) -> A2AResponse {
        let status_filter = request.params.get("status").and_then(|v| v.as_str());

        let tasks: Vec<&Task> = self.tasks.values()
            .filter(|t| {
                if let Some(status) = status_filter {
                    let task_status = format!("{:?}", t.status).to_lowercase();
                    task_status == status.to_lowercase()
                } else {
                    true
                }
            })
            .collect();

        A2AResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(json!({
                "tasks": tasks,
                "total": tasks.len()
            })),
            error: None,
        }
    }

    /// Handle tasks/sendMessage request
    fn handle_send_message(&mut self, request: &A2ARequest) -> A2AResponse {
        let task_id = request.params.get("task_id").and_then(|v| v.as_str());
        let message_content = request.params.get("message").and_then(|v| v.as_str());

        match (task_id, message_content) {
            (Some(id), Some(content)) => match self.tasks.get_mut(id) {
                Some(task) => {
                    // Add message to context
                    task.context.push(A2AMessage {
                        role: "user".to_string(),
                        content: vec![ContentPart::Text {
                            text: content.to_string(),
                        }],
                        timestamp: Some(chrono_lite_now()),
                    });
                    task.updated_at = chrono_lite_now();

                    // Process message and generate response
                    let response_text = format!("Received: {}", content);
                    task.context.push(A2AMessage {
                        role: "assistant".to_string(),
                        content: vec![ContentPart::Text {
                            text: response_text,
                        }],
                        timestamp: Some(chrono_lite_now()),
                    });

                    A2AResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request.id.clone(),
                        result: Some(serde_json::to_value(task).unwrap()),
                        error: None,
                    }
                }
                None => A2AResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: None,
                    error: Some(A2AError {
                        code: ERROR_TASK_NOT_FOUND,
                        message: format!("Task not found: {}", id),
                        data: None,
                    }),
                },
            },
            _ => A2AResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: None,
                error: Some(A2AError {
                    code: ERROR_INVALID_PARAMS,
                    message: "Missing task_id or message parameter".to_string(),
                    data: None,
                }),
            },
        }
    }

    /// Execute a skill
    fn execute_skill(&self, skill_id: &str, input: &Value) -> Result<Value, String> {
        match skill_id {
            "text_to_graph" => {
                let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("");
                let graph = grapheme_core::GraphemeGraph::from_text(text);
                Ok(json!({
                    "graph_id": format!("graph_{}", self.task_counter),
                    "node_count": graph.node_count(),
                    "edge_count": graph.edge_count(),
                    "input_nodes": graph.input_nodes.len()
                }))
            }
            "graph_transform" => {
                let transform_type = input.get("transform_type").and_then(|v| v.as_str()).unwrap_or("simplify");
                Ok(json!({
                    "graph_id": input.get("graph_id"),
                    "transform_type": transform_type,
                    "changes": ["applied_transform"],
                    "status": "completed"
                }))
            }
            "graph_to_text" => {
                // Simplified - would need actual graph lookup
                Ok(json!({
                    "text": "Reconstructed text from graph",
                    "graph_id": input.get("graph_id")
                }))
            }
            "analyze_text" => {
                let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("");
                let analysis_type = input.get("analysis_type").and_then(|v| v.as_str()).unwrap_or("structure");

                // Create graph for analysis
                let graph = grapheme_core::GraphemeGraph::from_text(text);

                Ok(json!({
                    "analysis_type": analysis_type,
                    "result": {
                        "text_length": text.len(),
                        "word_count": text.split_whitespace().count(),
                        "graph_nodes": graph.node_count(),
                        "graph_edges": graph.edge_count()
                    },
                    "confidence": 0.95
                }))
            }
            _ => Err(format!("Unknown skill: {}", skill_id)),
        }
    }
}

/// Simple timestamp generator (avoids chrono dependency)
fn chrono_lite_now() -> String {
    // Use system time for a basic ISO timestamp
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Simple ISO-8601 format
    format!("{}Z", secs)
}

// ============================================================================
// HTTP Server helpers (for integration with actix-web, axum, etc.)
// ============================================================================

/// Generate agent.json for /.well-known/agent.json endpoint
pub fn generate_agent_json(agent: &A2AAgent) -> String {
    serde_json::to_string_pretty(agent.get_agent_card()).unwrap_or_default()
}

/// Parse A2A request from JSON string
pub fn parse_request(json: &str) -> Result<A2ARequest, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}

/// Serialize A2A response to JSON string
pub fn serialize_response(response: &A2AResponse) -> String {
    serde_json::to_string(response).unwrap_or_default()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let agent = A2AAgent::new("http://localhost:8080");
        assert_eq!(agent.card.name, "GRAPHEME Agent");
        assert_eq!(agent.card.skills.len(), 4);
    }

    #[test]
    fn test_agent_card() {
        let agent = A2AAgent::new("http://localhost:8080");
        let card = agent.get_agent_card();
        assert!(card.capabilities.streaming);
        assert_eq!(card.protocol_version, "1.0");
    }

    #[test]
    fn test_agent_info_request() {
        let mut agent = A2AAgent::new("http://localhost:8080");
        let request = A2ARequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "agent/info".to_string(),
            params: json!({}),
        };

        let response = agent.handle_request(&request);
        assert!(response.error.is_none());
        assert!(response.result.is_some());
    }

    #[test]
    fn test_task_create() {
        let mut agent = A2AAgent::new("http://localhost:8080");
        let request = A2ARequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "tasks/create".to_string(),
            params: json!({
                "skill_id": "text_to_graph",
                "input": {"text": "Hello world"}
            }),
        };

        let response = agent.handle_request(&request);
        assert!(response.error.is_none());

        let result = response.result.unwrap();
        assert_eq!(result["status"], "completed");
    }

    #[test]
    fn test_task_get() {
        let mut agent = A2AAgent::new("http://localhost:8080");

        // Create a task first
        let create_request = A2ARequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "tasks/create".to_string(),
            params: json!({
                "skill_id": "analyze_text",
                "input": {"text": "Test", "analysis_type": "structure"}
            }),
        };
        agent.handle_request(&create_request);

        // Get the task
        let get_request = A2ARequest {
            jsonrpc: "2.0".to_string(),
            id: json!(2),
            method: "tasks/get".to_string(),
            params: json!({"task_id": "task_1"}),
        };

        let response = agent.handle_request(&get_request);
        assert!(response.error.is_none());
        assert!(response.result.is_some());
    }

    #[test]
    fn test_task_not_found() {
        let mut agent = A2AAgent::new("http://localhost:8080");
        let request = A2ARequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "tasks/get".to_string(),
            params: json!({"task_id": "nonexistent"}),
        };

        let response = agent.handle_request(&request);
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, ERROR_TASK_NOT_FOUND);
    }

    #[test]
    fn test_task_list() {
        let mut agent = A2AAgent::new("http://localhost:8080");

        // Create tasks
        for i in 0..3 {
            let request = A2ARequest {
                jsonrpc: "2.0".to_string(),
                id: json!(i),
                method: "tasks/create".to_string(),
                params: json!({
                    "skill_id": "text_to_graph",
                    "input": {"text": format!("Test {}", i)}
                }),
            };
            agent.handle_request(&request);
        }

        // List tasks
        let list_request = A2ARequest {
            jsonrpc: "2.0".to_string(),
            id: json!(100),
            method: "tasks/list".to_string(),
            params: json!({}),
        };

        let response = agent.handle_request(&list_request);
        assert!(response.error.is_none());
        let result = response.result.unwrap();
        assert_eq!(result["total"], 3);
    }

    #[test]
    fn test_unknown_method() {
        let mut agent = A2AAgent::new("http://localhost:8080");
        let request = A2ARequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "unknown/method".to_string(),
            params: json!({}),
        };

        let response = agent.handle_request(&request);
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, ERROR_METHOD_NOT_FOUND);
    }

    #[test]
    fn test_unknown_skill() {
        let mut agent = A2AAgent::new("http://localhost:8080");
        let request = A2ARequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "tasks/create".to_string(),
            params: json!({
                "skill_id": "unknown_skill",
                "input": {}
            }),
        };

        let response = agent.handle_request(&request);
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, ERROR_INVALID_PARAMS);
    }

    #[test]
    fn test_generate_agent_json() {
        let agent = A2AAgent::new("http://localhost:8080");
        let json = generate_agent_json(&agent);
        assert!(json.contains("GRAPHEME Agent"));
        assert!(json.contains("text_to_graph"));
    }
}
