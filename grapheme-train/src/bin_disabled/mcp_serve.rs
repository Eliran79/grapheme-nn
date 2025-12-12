//! MCP Server Binary for GRAPHEME
//!
//! Backend-187: Expose GRAPHEME capabilities via Model Context Protocol.
//!
//! Usage:
//!   mcp_serve                    # Run with stdio transport (for Claude Code)
//!   mcp_serve --mode http        # Run with HTTP transport on port 8080
//!   mcp_serve --port 3000        # Custom port for HTTP mode

use clap::Parser;
use grapheme_train::{MCPServer, JsonRpcRequest};
use std::io::{self, BufRead, Write};

#[derive(Parser, Debug)]
#[command(name = "mcp_serve")]
#[command(about = "GRAPHEME MCP Server - Expose graph tools via Model Context Protocol")]
struct Args {
    /// Transport mode: stdio (default) or http
    #[arg(short, long, default_value = "stdio")]
    mode: String,

    /// Port for HTTP mode
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    eprintln!("GRAPHEME MCP Server");
    eprintln!("===================");
    eprintln!("Mode: {}", args.mode);

    let mut server = MCPServer::new();

    match args.mode.as_str() {
        "stdio" => run_stdio_server(&mut server, args.verbose),
        "http" => {
            eprintln!("HTTP mode not yet implemented - use stdio mode");
            eprintln!("For HTTP, integrate with actix-web or axum");
            std::process::exit(1);
        }
        _ => {
            eprintln!("Unknown mode: {}. Use 'stdio' or 'http'", args.mode);
            std::process::exit(1);
        }
    }
}

/// Run MCP server with stdio transport (for Claude Code integration)
fn run_stdio_server(server: &mut MCPServer, verbose: bool) -> ! {
    eprintln!("Running in stdio mode - reading JSON-RPC from stdin");
    eprintln!("Send JSON-RPC requests, one per line");
    eprintln!("Example: {{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{{}}}}");
    eprintln!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading stdin: {}", e);
                continue;
            }
        };

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        if verbose {
            eprintln!(">>> {}", line);
        }

        // Parse JSON-RPC request
        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                let error_response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {}", e)
                    }
                });
                let _ = writeln!(stdout, "{}", error_response);
                let _ = stdout.flush();
                continue;
            }
        };

        // Handle request
        let response = server.handle_request(&request);

        // Write response
        let response_json = serde_json::to_string(&response).unwrap_or_else(|e| {
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {
                    "code": -32603,
                    "message": format!("Internal error: {}", e)
                }
            }).to_string()
        });

        if verbose {
            eprintln!("<<< {}", response_json);
        }

        let _ = writeln!(stdout, "{}", response_json);
        let _ = stdout.flush();
    }

    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = MCPServer::new();
        assert!(server.tools().len() >= 5);
    }
}
