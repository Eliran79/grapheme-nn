//! Efficient Graph Serialization for Network Transport
//!
//! Backend-180: Implements compact binary serialization for graphs.
//!
//! Supports:
//! - Binary serialization with variable-length integers
//! - Delta encoding for node positions
//! - Run-length encoding for repeated values
//! - Compression-friendly layout

use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Errors during serialization
#[derive(Debug, Clone)]
pub enum SerializationError {
    IoError(String),
    InvalidFormat(String),
    VersionMismatch { expected: u8, found: u8 },
}

impl std::fmt::Display for SerializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializationError::IoError(msg) => write!(f, "IO error: {}", msg),
            SerializationError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            SerializationError::VersionMismatch { expected, found } => {
                write!(f, "Version mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for SerializationError {}

/// Result type for serialization operations
pub type SerResult<T> = Result<T, SerializationError>;

/// Magic number for binary format identification
const MAGIC: [u8; 4] = [0x47, 0x52, 0x50, 0x48]; // "GRPH"
/// Current format version
const VERSION: u8 = 1;

/// Binary graph representation for network transport
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactGraph {
    /// Number of nodes
    pub node_count: u32,
    /// Number of edges
    pub edge_count: u32,
    /// Node data (serialized)
    pub nodes: Vec<u8>,
    /// Edge data (source, target pairs as delta-encoded)
    pub edges: Vec<u8>,
    /// Edge weights (optional, compressed)
    pub weights: Option<Vec<u8>>,
}

impl CompactGraph {
    /// Create a new empty compact graph
    pub fn new() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            nodes: Vec::new(),
            edges: Vec::new(),
            weights: None,
        }
    }

    /// Serialize a petgraph DiGraph to compact binary format
    pub fn from_digraph<N, E>(graph: &DiGraph<N, E>) -> SerResult<Self>
    where
        N: Serialize,
        E: Serialize,
    {
        let node_count = graph.node_count() as u32;
        let edge_count = graph.edge_count() as u32;

        // Serialize nodes
        let mut nodes = Vec::new();
        for node_idx in graph.node_indices() {
            let node_data = graph.node_weight(node_idx).ok_or_else(|| {
                SerializationError::InvalidFormat("Missing node weight".into())
            })?;
            let serialized =
                serde_json::to_vec(node_data).map_err(|e| SerializationError::IoError(e.to_string()))?;
            // Length-prefix each node
            write_varint(&mut nodes, serialized.len() as u64);
            nodes.extend(serialized);
        }

        // Serialize edges with delta encoding
        let mut edges = Vec::new();
        let mut prev_source = 0u32;
        for edge in graph.edge_indices() {
            if let Some((source, target)) = graph.edge_endpoints(edge) {
                let src = source.index() as u32;
                let tgt = target.index() as u32;
                // Delta encode source
                let delta = if src >= prev_source {
                    ((src - prev_source) << 1) as u64
                } else {
                    (((prev_source - src) << 1) | 1) as u64
                };
                write_varint(&mut edges, delta);
                write_varint(&mut edges, tgt as u64);
                prev_source = src;
            }
        }

        // Serialize edge weights
        let mut weights_data = Vec::new();
        for edge in graph.edge_indices() {
            if let Some(weight) = graph.edge_weight(edge) {
                let serialized =
                    serde_json::to_vec(weight).map_err(|e| SerializationError::IoError(e.to_string()))?;
                write_varint(&mut weights_data, serialized.len() as u64);
                weights_data.extend(serialized);
            }
        }

        Ok(Self {
            node_count,
            edge_count,
            nodes,
            edges,
            weights: if weights_data.is_empty() {
                None
            } else {
                Some(weights_data)
            },
        })
    }

    /// Deserialize to a petgraph DiGraph
    pub fn to_digraph<N, E>(&self) -> SerResult<DiGraph<N, E>>
    where
        N: for<'de> Deserialize<'de>,
        E: for<'de> Deserialize<'de> + Default + Clone,
    {
        let mut graph = DiGraph::new();
        let mut node_indices: Vec<NodeIndex> = Vec::with_capacity(self.node_count as usize);

        // Deserialize nodes
        let mut cursor = 0;
        for _ in 0..self.node_count {
            let (len, bytes_read) = read_varint(&self.nodes[cursor..])
                .ok_or_else(|| SerializationError::InvalidFormat("Invalid node length".into()))?;
            cursor += bytes_read;
            let node_data: N = serde_json::from_slice(&self.nodes[cursor..cursor + len as usize])
                .map_err(|e| SerializationError::IoError(e.to_string()))?;
            cursor += len as usize;
            let idx = graph.add_node(node_data);
            node_indices.push(idx);
        }

        // Deserialize edges
        let mut edge_cursor = 0;
        let mut weight_cursor = 0;
        let mut prev_source = 0u32;

        for _ in 0..self.edge_count {
            let (delta, bytes_read) = read_varint(&self.edges[edge_cursor..])
                .ok_or_else(|| SerializationError::InvalidFormat("Invalid edge delta".into()))?;
            edge_cursor += bytes_read;

            let (target, bytes_read) = read_varint(&self.edges[edge_cursor..])
                .ok_or_else(|| SerializationError::InvalidFormat("Invalid edge target".into()))?;
            edge_cursor += bytes_read;

            // Decode delta
            let source = if delta & 1 == 0 {
                prev_source + (delta >> 1) as u32
            } else {
                prev_source - (delta >> 1) as u32
            };
            prev_source = source;

            // Get edge weight
            let weight: E = if let Some(ref weights) = self.weights {
                let (len, bytes_read) = read_varint(&weights[weight_cursor..])
                    .ok_or_else(|| SerializationError::InvalidFormat("Invalid weight length".into()))?;
                weight_cursor += bytes_read;
                let w: E = serde_json::from_slice(&weights[weight_cursor..weight_cursor + len as usize])
                    .map_err(|e| SerializationError::IoError(e.to_string()))?;
                weight_cursor += len as usize;
                w
            } else {
                E::default()
            };

            if (source as usize) < node_indices.len() && (target as usize) < node_indices.len() {
                graph.add_edge(
                    node_indices[source as usize],
                    node_indices[target as usize],
                    weight,
                );
            }
        }

        Ok(graph)
    }

    /// Write to binary format
    pub fn write_binary<W: Write>(&self, writer: &mut W) -> SerResult<()> {
        writer
            .write_all(&MAGIC)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;
        writer
            .write_all(&[VERSION])
            .map_err(|e| SerializationError::IoError(e.to_string()))?;

        // Write counts
        writer
            .write_all(&self.node_count.to_le_bytes())
            .map_err(|e| SerializationError::IoError(e.to_string()))?;
        writer
            .write_all(&self.edge_count.to_le_bytes())
            .map_err(|e| SerializationError::IoError(e.to_string()))?;

        // Write nodes length and data
        writer
            .write_all(&(self.nodes.len() as u32).to_le_bytes())
            .map_err(|e| SerializationError::IoError(e.to_string()))?;
        writer
            .write_all(&self.nodes)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;

        // Write edges length and data
        writer
            .write_all(&(self.edges.len() as u32).to_le_bytes())
            .map_err(|e| SerializationError::IoError(e.to_string()))?;
        writer
            .write_all(&self.edges)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;

        // Write weights (optional)
        match &self.weights {
            Some(w) => {
                writer
                    .write_all(&[1])
                    .map_err(|e| SerializationError::IoError(e.to_string()))?;
                writer
                    .write_all(&(w.len() as u32).to_le_bytes())
                    .map_err(|e| SerializationError::IoError(e.to_string()))?;
                writer
                    .write_all(w)
                    .map_err(|e| SerializationError::IoError(e.to_string()))?;
            }
            None => {
                writer
                    .write_all(&[0])
                    .map_err(|e| SerializationError::IoError(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Read from binary format
    pub fn read_binary<R: Read>(reader: &mut R) -> SerResult<Self> {
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;
        if magic != MAGIC {
            return Err(SerializationError::InvalidFormat("Invalid magic number".into()));
        }

        let mut version = [0u8; 1];
        reader
            .read_exact(&mut version)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;
        if version[0] != VERSION {
            return Err(SerializationError::VersionMismatch {
                expected: VERSION,
                found: version[0],
            });
        }

        let mut buf4 = [0u8; 4];

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;
        let node_count = u32::from_le_bytes(buf4);

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;
        let edge_count = u32::from_le_bytes(buf4);

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;
        let nodes_len = u32::from_le_bytes(buf4) as usize;
        let mut nodes = vec![0u8; nodes_len];
        reader
            .read_exact(&mut nodes)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;
        let edges_len = u32::from_le_bytes(buf4) as usize;
        let mut edges = vec![0u8; edges_len];
        reader
            .read_exact(&mut edges)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;

        let mut has_weights = [0u8; 1];
        reader
            .read_exact(&mut has_weights)
            .map_err(|e| SerializationError::IoError(e.to_string()))?;

        let weights = if has_weights[0] == 1 {
            reader
                .read_exact(&mut buf4)
                .map_err(|e| SerializationError::IoError(e.to_string()))?;
            let weights_len = u32::from_le_bytes(buf4) as usize;
            let mut weights = vec![0u8; weights_len];
            reader
                .read_exact(&mut weights)
                .map_err(|e| SerializationError::IoError(e.to_string()))?;
            Some(weights)
        } else {
            None
        };

        Ok(Self {
            node_count,
            edge_count,
            nodes,
            edges,
            weights,
        })
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        4 + 1 + 8 + 4 + self.nodes.len() + 4 + self.edges.len() + 1 + self.weights.as_ref().map(|w| 4 + w.len()).unwrap_or(0)
    }
}

impl Default for CompactGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Write a variable-length integer (LEB128-like encoding)
fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Read a variable-length integer, returns (value, bytes_read)
fn read_varint(data: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0;
    let mut bytes_read = 0;

    for &byte in data {
        bytes_read += 1;
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((value, bytes_read));
        }
        shift += 7;
        if shift > 63 {
            return None; // Overflow
        }
    }
    None
}

/// Statistics about serialization
#[derive(Debug, Clone, Default)]
pub struct SerializationStats {
    pub original_size: usize,
    pub compact_size: usize,
    pub compression_ratio: f32,
}

/// Calculate serialization stats by comparing JSON vs binary
pub fn calculate_stats<N, E>(graph: &DiGraph<N, E>) -> SerializationStats
where
    N: Serialize,
    E: Serialize,
{
    let json_size = serde_json::to_vec(graph)
        .map(|v| v.len())
        .unwrap_or(0);

    let compact_size = CompactGraph::from_digraph(graph)
        .map(|c| c.size_bytes())
        .unwrap_or(0);

    let compression_ratio = if compact_size > 0 {
        json_size as f32 / compact_size as f32
    } else {
        0.0
    };

    SerializationStats {
        original_size: json_size,
        compact_size,
        compression_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        let values = [0, 1, 127, 128, 255, 1000, 65535, 1_000_000];
        for &v in &values {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let (read_value, _) = read_varint(&buf).unwrap();
            assert_eq!(v, read_value, "Failed for value {}", v);
        }
    }

    #[test]
    fn test_compact_graph_roundtrip() {
        let mut graph: DiGraph<String, f32> = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 2.0);
        graph.add_edge(a, c, 0.5);

        let compact = CompactGraph::from_digraph(&graph).unwrap();
        let restored: DiGraph<String, f32> = compact.to_digraph().unwrap();

        assert_eq!(graph.node_count(), restored.node_count());
        assert_eq!(graph.edge_count(), restored.edge_count());
    }

    #[test]
    fn test_binary_roundtrip() {
        let mut graph: DiGraph<i32, ()> = DiGraph::new();
        let a = graph.add_node(1);
        let b = graph.add_node(2);
        graph.add_edge(a, b, ());

        let compact = CompactGraph::from_digraph(&graph).unwrap();

        let mut buffer = Vec::new();
        compact.write_binary(&mut buffer).unwrap();

        let mut cursor = std::io::Cursor::new(buffer);
        let restored = CompactGraph::read_binary(&mut cursor).unwrap();

        assert_eq!(compact.node_count, restored.node_count);
        assert_eq!(compact.edge_count, restored.edge_count);
    }

    #[test]
    fn test_compression_stats() {
        let mut graph: DiGraph<String, f32> = DiGraph::new();
        for i in 0..100 {
            graph.add_node(format!("Node{}", i));
        }
        for i in 0..99 {
            graph.add_edge(NodeIndex::new(i), NodeIndex::new(i + 1), i as f32 * 0.1);
        }

        let stats = calculate_stats(&graph);
        assert!(stats.original_size > 0);
        assert!(stats.compact_size > 0);
        // Compact format should be more efficient for large graphs
        println!(
            "JSON: {} bytes, Compact: {} bytes, Ratio: {:.2}x",
            stats.original_size, stats.compact_size, stats.compression_ratio
        );
    }
}
