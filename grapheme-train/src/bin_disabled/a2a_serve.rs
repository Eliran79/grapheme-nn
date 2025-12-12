//! A2A Agent HTTP Server Binary for GRAPHEME
//!
//! Backend-188: Expose GRAPHEME as an A2A-compatible agent.
//!
//! Usage:
//!   a2a_serve                    # Run HTTP server on port 8080
//!   a2a_serve --port 3000        # Custom port
//!   a2a_serve --verbose          # Enable request logging
//!
//! Endpoints:
//!   GET  /.well-known/agent.json  - Agent discovery card
//!   POST /a2a                     - JSON-RPC endpoint for tasks

use clap::Parser;
use grapheme_train::{A2AAgent, A2ARequest, generate_agent_json, serialize_response};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

#[derive(Parser, Debug)]
#[command(name = "a2a_serve")]
#[command(about = "GRAPHEME A2A Agent Server - Inter-agent communication via Agent2Agent protocol")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Bind address
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let bind_addr = format!("{}:{}", args.bind, args.port);
    let base_url = format!("http://{}:{}", args.bind, args.port);

    eprintln!("GRAPHEME A2A Agent Server");
    eprintln!("=========================");
    eprintln!("Base URL: {}", base_url);
    eprintln!("Agent Card: {}/.well-known/agent.json", base_url);
    eprintln!("A2A Endpoint: {}/a2a", base_url);
    eprintln!();

    let agent = Arc::new(Mutex::new(A2AAgent::new(&base_url)));
    let listener = TcpListener::bind(&bind_addr)?;

    eprintln!("Listening on {}", bind_addr);
    eprintln!("Press Ctrl+C to stop");
    eprintln!();

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let agent = Arc::clone(&agent);
                let verbose = args.verbose;
                std::thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, &agent, verbose) {
                        eprintln!("Connection error: {}", e);
                    }
                });
            }
            Err(e) => {
                eprintln!("Accept error: {}", e);
            }
        }
    }

    Ok(())
}

fn handle_connection(
    mut stream: TcpStream,
    agent: &Arc<Mutex<A2AAgent>>,
    verbose: bool,
) -> anyhow::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;

    if verbose {
        eprintln!(">>> {}", request_line.trim());
    }

    // Parse HTTP request line
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return send_response(&mut stream, 400, "Bad Request", "Invalid request");
    }

    let method = parts[0];
    let path = parts[1];

    // Read headers
    let mut content_length = 0;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        if header.trim().is_empty() {
            break;
        }
        if header.to_lowercase().starts_with("content-length:") {
            if let Some(len) = header.split(':').nth(1) {
                content_length = len.trim().parse().unwrap_or(0);
            }
        }
    }

    // Handle routes
    match (method, path) {
        ("GET", "/.well-known/agent.json") => {
            let agent = agent.lock().unwrap();
            let card_json = generate_agent_json(&agent);
            send_json_response(&mut stream, 200, &card_json)
        }
        ("POST", "/a2a") => {
            // Read body
            let mut body = vec![0u8; content_length];
            if content_length > 0 {
                std::io::Read::read_exact(&mut reader, &mut body)?;
            }
            let body_str = String::from_utf8_lossy(&body);

            if verbose {
                eprintln!("Body: {}", body_str);
            }

            // Parse JSON-RPC request
            let request: A2ARequest = match serde_json::from_str(&body_str) {
                Ok(req) => req,
                Err(e) => {
                    let error_json = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": null,
                        "error": {
                            "code": -32700,
                            "message": format!("Parse error: {}", e)
                        }
                    });
                    return send_json_response(&mut stream, 200, &error_json.to_string());
                }
            };

            // Handle request
            let mut agent = agent.lock().unwrap();
            let response = agent.handle_request(&request);
            let response_json = serialize_response(&response);

            if verbose {
                eprintln!("<<< {}", response_json);
            }

            send_json_response(&mut stream, 200, &response_json)
        }
        ("GET", "/health") => {
            send_json_response(&mut stream, 200, r#"{"status":"ok"}"#)
        }
        ("OPTIONS", _) => {
            // CORS preflight
            send_cors_response(&mut stream)
        }
        _ => {
            send_response(&mut stream, 404, "Not Found", "Unknown endpoint")
        }
    }
}

fn send_response(
    stream: &mut TcpStream,
    status: u16,
    status_text: &str,
    body: &str,
) -> anyhow::Result<()> {
    let response = format!(
        "HTTP/1.1 {} {}\r\n\
         Content-Type: text/plain\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        status,
        status_text,
        body.len(),
        body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn send_json_response(stream: &mut TcpStream, status: u16, body: &str) -> anyhow::Result<()> {
    let response = format!(
        "HTTP/1.1 {} OK\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        status,
        body.len(),
        body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn send_cors_response(stream: &mut TcpStream) -> anyhow::Result<()> {
    let response = "HTTP/1.1 204 No Content\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Connection: close\r\n\
         \r\n";
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let agent = A2AAgent::new("http://localhost:8080");
        let card = agent.get_agent_card();
        assert_eq!(card.name, "GRAPHEME Agent");
    }

    #[test]
    fn test_agent_json() {
        let agent = A2AAgent::new("http://localhost:8080");
        let json = generate_agent_json(&agent);
        assert!(json.contains("GRAPHEME Agent"));
        assert!(json.contains("text_to_graph"));
    }
}
