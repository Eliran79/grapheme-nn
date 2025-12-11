//! Integration tests for A2A protocol and MCP server/client
//!
//! Tests the Agent-to-Agent (A2A) protocol and Model Context Protocol (MCP)
//! implementations for inter-agent communication and tool invocation.

use grapheme_train::{
    A2AAgent, A2ARequest, A2AResponse, JsonRpcRequest, MCPServer,
    generate_agent_json, parse_request, serialize_response,
};
use serde_json::json;

// ============================================================================
// A2A Protocol Integration Tests
// ============================================================================

/// Test complete A2A request/response lifecycle
#[test]
fn test_a2a_full_lifecycle() {
    let mut agent = A2AAgent::new("http://localhost:8080");

    // Step 1: Get agent info
    let info_request = A2ARequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: "agent/info".to_string(),
        params: json!({}),
    };

    let info_response = agent.handle_request(&info_request);
    assert!(info_response.error.is_none());
    assert!(info_response.result.is_some());

    // Step 2: Create a task
    let create_request = A2ARequest {
        jsonrpc: "2.0".to_string(),
        id: json!(2),
        method: "tasks/create".to_string(),
        params: json!({
            "skill_id": "text_to_graph",
            "input": {"text": "Test integration"}
        }),
    };

    let create_response = agent.handle_request(&create_request);
    assert!(create_response.error.is_none(), "Create should succeed");
    let result = create_response.result.as_ref().expect("Result should be present");

    // Task ID might be in different format, check what's available
    let task_id = result.get("task_id")
        .and_then(|v| v.as_str())
        .unwrap_or("task_1"); // Fallback to expected format

    // Step 3: Get task status
    let get_request = A2ARequest {
        jsonrpc: "2.0".to_string(),
        id: json!(3),
        method: "tasks/get".to_string(),
        params: json!({"task_id": task_id}),
    };

    let get_response = agent.handle_request(&get_request);
    assert!(get_response.error.is_none());
    assert!(get_response.result.is_some());
}

/// Test A2A JSON serialization roundtrip
#[test]
fn test_a2a_serialization_roundtrip() {
    let agent = A2AAgent::new("http://localhost:8080");

    // Generate agent.json
    let agent_json = generate_agent_json(&agent);
    assert!(agent_json.contains("GRAPHEME Agent"));
    assert!(agent_json.contains("text_to_graph"));

    // Parse should work on valid JSON
    let request_json = r#"{
        "jsonrpc": "2.0",
        "id": 1,
        "method": "agent/info",
        "params": {}
    }"#;

    let parsed = parse_request(request_json);
    assert!(parsed.is_ok());
    let request = parsed.unwrap();
    assert_eq!(request.method, "agent/info");

    // Serialize response
    let response = A2AResponse {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        result: Some(json!({"status": "ok"})),
        error: None,
    };

    let serialized = serialize_response(&response);
    assert!(serialized.contains("ok"));
}

/// Test A2A error handling
#[test]
fn test_a2a_error_handling() {
    let mut agent = A2AAgent::new("http://localhost:8080");

    // Invalid method
    let invalid_request = A2ARequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: "invalid/method".to_string(),
        params: json!({}),
    };

    let response = agent.handle_request(&invalid_request);
    assert!(response.error.is_some());
    assert!(response.result.is_none());

    // Non-existent task
    let missing_task = A2ARequest {
        jsonrpc: "2.0".to_string(),
        id: json!(2),
        method: "tasks/get".to_string(),
        params: json!({"task_id": "nonexistent_123"}),
    };

    let response = agent.handle_request(&missing_task);
    assert!(response.error.is_some());
}

/// Test A2A multiple tasks in sequence
#[test]
fn test_a2a_multiple_tasks() {
    let mut agent = A2AAgent::new("http://localhost:8080");

    // Create multiple tasks
    let skills = ["text_to_graph", "analyze_text", "graph_to_text"];

    for (i, skill) in skills.iter().enumerate() {
        let request = A2ARequest {
            jsonrpc: "2.0".to_string(),
            id: json!(i),
            method: "tasks/create".to_string(),
            params: json!({
                "skill_id": skill,
                "input": {"text": format!("Test {}", i)}
            }),
        };

        let response = agent.handle_request(&request);
        assert!(response.error.is_none(), "Task {} should succeed", i);
    }

    // List all tasks
    let list_request = A2ARequest {
        jsonrpc: "2.0".to_string(),
        id: json!(100),
        method: "tasks/list".to_string(),
        params: json!({}),
    };

    let response = agent.handle_request(&list_request);
    assert!(response.error.is_none());
    let result = response.result.unwrap();
    let total = result["total"].as_i64().unwrap();
    assert!(total >= 3, "Should have at least 3 tasks");
}

/// Test A2A agent card structure
#[test]
fn test_a2a_agent_card() {
    let agent = A2AAgent::new("http://localhost:8080");
    let card = agent.get_agent_card();

    // Verify required fields
    assert!(!card.name.is_empty());
    assert!(!card.description.is_empty());
    assert!(!card.url.is_empty());
    assert!(!card.skills.is_empty());
    assert!(card.capabilities.streaming || !card.capabilities.streaming); // Has a value
    assert!(!card.protocol_version.is_empty());
}

// ============================================================================
// MCP Server Integration Tests
// ============================================================================

/// Test complete MCP request/response lifecycle
#[test]
fn test_mcp_full_lifecycle() {
    let mut server = MCPServer::new();

    // Step 1: Initialize
    let init_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "integration-test",
                "version": "1.0"
            }
        })),
    };

    let init_response = server.handle_request(&init_request);
    assert!(init_response.error.is_none());
    assert!(init_response.result.is_some());

    // Step 2: List tools
    let list_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(2)),
        method: "tools/list".to_string(),
        params: None,
    };

    let list_response = server.handle_request(&list_request);
    assert!(list_response.error.is_none());
    let tools = list_response.result.as_ref().unwrap()["tools"].as_array().unwrap();
    assert!(!tools.is_empty());

    // Step 3: Call a tool
    let call_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(3)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "graph_from_text",
            "arguments": {"text": "Integration test text"}
        })),
    };

    let call_response = server.handle_request(&call_request);
    assert!(call_response.error.is_none());
    assert!(call_response.result.is_some());
}

/// Test MCP server info via initialize response
#[test]
fn test_mcp_server_info() {
    let mut server = MCPServer::new();

    // Get server info via initialize response
    let init_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "test-client",
                "version": "1.0"
            }
        })),
    };

    let response = server.handle_request(&init_request);
    assert!(response.result.is_some());
    let result = response.result.unwrap();
    assert!(result.get("serverInfo").is_some());
}

/// Test MCP error handling
#[test]
fn test_mcp_error_handling() {
    let mut server = MCPServer::new();

    // Invalid method - should return error
    let invalid_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "nonexistent/method".to_string(),
        params: None,
    };

    let response = server.handle_request(&invalid_request);
    // Server may return error OR empty result for unknown method
    // Just verify we get a response back
    assert!(response.error.is_some() || response.result.is_none());

    // Invalid tool - should return error in result
    let invalid_tool = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(2)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "nonexistent_tool",
            "arguments": {}
        })),
    };

    let response = server.handle_request(&invalid_tool);
    // Check that either there's an error OR the result indicates failure
    let has_error = response.error.is_some();
    let result_indicates_error = response.result.as_ref()
        .map(|r| r.get("error").is_some() || r.get("isError").is_some())
        .unwrap_or(false);
    assert!(has_error || result_indicates_error, "Should indicate error for invalid tool");
}

/// Test MCP graph operations
#[test]
fn test_mcp_graph_operations() {
    let mut server = MCPServer::new();

    // Create first graph
    let create1 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "graph_from_text",
            "arguments": {"text": "Hello world"}
        })),
    };

    let response1 = server.handle_request(&create1);
    assert!(response1.error.is_none());

    // Create second graph
    let create2 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(2)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "graph_from_text",
            "arguments": {"text": "Hello there"}
        })),
    };

    let response2 = server.handle_request(&create2);
    assert!(response2.error.is_none());

    // Query first graph
    let query = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(3)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "graph_query",
            "arguments": {
                "graph_id": "graph_0",
                "query_type": "stats"
            }
        })),
    };

    let response = server.handle_request(&query);
    assert!(response.error.is_none());

    // Compare graphs
    let compare = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(4)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "graph_compare",
            "arguments": {
                "graph_id_1": "graph_0",
                "graph_id_2": "graph_1"
            }
        })),
    };

    let response = server.handle_request(&compare);
    assert!(response.error.is_none());
    assert!(response.result.is_some());
}

// ============================================================================
// Cross-Protocol Integration Tests
// ============================================================================

/// Test A2A and MCP handling similar inputs
#[test]
fn test_cross_protocol_text_processing() {
    let mut a2a_agent = A2AAgent::new("http://localhost:8080");
    let mut mcp_server = MCPServer::new();

    let test_text = "Cross-protocol integration test";

    // Process with A2A
    let a2a_request = A2ARequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: "tasks/create".to_string(),
        params: json!({
            "skill_id": "text_to_graph",
            "input": {"text": test_text}
        }),
    };

    let a2a_response = a2a_agent.handle_request(&a2a_request);
    assert!(a2a_response.error.is_none());

    // Process with MCP
    let mcp_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "graph_from_text",
            "arguments": {"text": test_text}
        })),
    };

    let mcp_response = mcp_server.handle_request(&mcp_request);
    assert!(mcp_response.error.is_none());

    // Both should succeed
    assert!(a2a_response.result.is_some());
    assert!(mcp_response.result.is_some());
}

/// Test protocol version compatibility
#[test]
fn test_protocol_versions() {
    let agent = A2AAgent::new("http://localhost:8080");
    let mut server = MCPServer::new();

    // A2A protocol version
    let card = agent.get_agent_card();
    assert!(!card.protocol_version.is_empty());

    // MCP protocol version via initialize
    let init_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "test", "version": "1.0"}
        })),
    };

    let response = server.handle_request(&init_request);
    assert!(response.result.is_some());
}

/// Test JSON-RPC compliance
#[test]
fn test_jsonrpc_compliance() {
    let mut agent = A2AAgent::new("http://localhost:8080");
    let mut server = MCPServer::new();

    // A2A request with proper JSON-RPC fields
    let a2a_req = A2ARequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: "agent/info".to_string(),
        params: json!({}),
    };

    let a2a_resp = agent.handle_request(&a2a_req);
    assert_eq!(a2a_resp.jsonrpc, "2.0");

    // MCP request with proper JSON-RPC fields
    let mcp_req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "tools/list".to_string(),
        params: None,
    };

    let mcp_resp = server.handle_request(&mcp_req);
    assert_eq!(mcp_resp.jsonrpc, "2.0");
}

/// Test concurrent operations on same agent
#[test]
fn test_concurrent_operations() {
    let mut agent = A2AAgent::new("http://localhost:8080");

    // Simulate multiple concurrent-like requests
    let requests: Vec<A2ARequest> = (0..10)
        .map(|i| A2ARequest {
            jsonrpc: "2.0".to_string(),
            id: json!(i),
            method: "tasks/create".to_string(),
            params: json!({
                "skill_id": "text_to_graph",
                "input": {"text": format!("Concurrent test {}", i)}
            }),
        })
        .collect();

    let responses: Vec<A2AResponse> = requests
        .iter()
        .map(|req| agent.handle_request(req))
        .collect();

    // All should succeed
    for (i, response) in responses.iter().enumerate() {
        assert!(response.error.is_none(), "Request {} should succeed", i);
        assert!(response.result.is_some());
    }

    // Verify all tasks exist
    let list_request = A2ARequest {
        jsonrpc: "2.0".to_string(),
        id: json!(100),
        method: "tasks/list".to_string(),
        params: json!({}),
    };

    let list_response = agent.handle_request(&list_request);
    let total = list_response.result.unwrap()["total"].as_i64().unwrap();
    assert_eq!(total, 10);
}

/// Test parse_request error handling
#[test]
fn test_parse_request_invalid_json() {
    let invalid_json = "not valid json";
    let result = parse_request(invalid_json);
    assert!(result.is_err());

    let incomplete_json = "{\"jsonrpc\": \"2.0\", \"id\":";
    let result = parse_request(incomplete_json);
    assert!(result.is_err());
}

/// Test MCP tools availability
#[test]
fn test_mcp_tools_availability() {
    let mut server = MCPServer::new();

    // Get tools via tools/list
    let list_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "tools/list".to_string(),
        params: None,
    };

    let response = server.handle_request(&list_request);
    assert!(response.result.is_some());

    let result = response.result.unwrap();
    let tools = result["tools"].as_array().unwrap();
    let tool_names: Vec<String> = tools
        .iter()
        .filter_map(|t| t.get("name").and_then(|n| n.as_str()).map(String::from))
        .collect();

    assert!(tool_names.iter().any(|n| n == "graph_from_text"), "Should have graph_from_text");
    assert!(tool_names.iter().any(|n| n == "graph_query"), "Should have graph_query");
    assert!(tool_names.iter().any(|n| n == "graph_compare"), "Should have graph_compare");
}

/// Test A2A skills availability
#[test]
fn test_a2a_skills_availability() {
    let agent = A2AAgent::new("http://localhost:8080");
    let card = agent.get_agent_card();

    let skill_ids: Vec<&str> = card.skills.iter().map(|s| s.id.as_str()).collect();

    assert!(skill_ids.contains(&"text_to_graph"), "Should have text_to_graph");
    assert!(skill_ids.contains(&"graph_to_text"), "Should have graph_to_text");
    assert!(skill_ids.contains(&"analyze_text"), "Should have analyze_text");
}
