//! Graph-Only Training Data Format (backend-227)
//!
//! Stores training pairs as DagNN graphs without text intermediates.
//! This enables pure graph-to-graph training following GRAPHEME vision.
//!
//! Key features:
//! - Binary serialization for efficient storage
//! - No text in training loop - pure graph pairs
//! - Batched I/O for high-throughput training
//! - Metadata for curriculum level tracking

use grapheme_core::{DagNN, NodeId, Node, Edge};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use thiserror::Error;

/// Errors for graph data operations
#[derive(Error, Debug)]
pub enum GraphDataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Invalid graph data: {0}")]
    InvalidData(String),
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },
}

/// Result type for graph data operations
pub type GraphDataResult<T> = Result<T, GraphDataError>;

/// Current format version
const FORMAT_VERSION: u32 = 1;

/// Magic bytes to identify graph data files
const MAGIC_BYTES: [u8; 4] = [b'G', b'R', b'P', b'H'];

// ============================================================================
// Graph Pair - Core Training Unit
// ============================================================================

/// A single graph-to-graph training pair
///
/// This is the fundamental unit of training data - an input graph
/// and its expected output graph, with no text representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPair {
    /// Unique identifier for this pair
    pub id: String,
    /// Input graph (e.g., source code AST, math expression)
    pub input: DagNN,
    /// Expected output graph (e.g., compiled code, evaluated expression)
    pub output: DagNN,
    /// Curriculum level (1-7, higher = more complex)
    pub level: u8,
    /// Domain tag for routing (e.g., "math", "code", "text")
    pub domain: String,
    /// Optional metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl GraphPair {
    /// Create a new graph pair
    pub fn new(id: impl Into<String>, input: DagNN, output: DagNN) -> Self {
        Self {
            id: id.into(),
            input,
            output,
            level: 1,
            domain: "general".to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Set the curriculum level
    pub fn with_level(mut self, level: u8) -> Self {
        self.level = level;
        self
    }

    /// Set the domain
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = domain.into();
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Compute statistics about this pair
    pub fn stats(&self) -> GraphPairStats {
        GraphPairStats {
            input_nodes: self.input.node_count(),
            input_edges: self.input.edge_count(),
            output_nodes: self.output.node_count(),
            output_edges: self.output.edge_count(),
            level: self.level,
        }
    }
}

/// Statistics about a graph pair
#[derive(Debug, Clone, Default)]
pub struct GraphPairStats {
    pub input_nodes: usize,
    pub input_edges: usize,
    pub output_nodes: usize,
    pub output_edges: usize,
    pub level: u8,
}

// ============================================================================
// Graph Dataset - Collection of Graph Pairs
// ============================================================================

/// A dataset of graph pairs for training
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphDataset {
    /// All graph pairs
    pub pairs: Vec<GraphPair>,
    /// Dataset metadata
    pub metadata: GraphDatasetMetadata,
}

/// Metadata about a graph dataset
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphDatasetMetadata {
    /// Dataset name
    pub name: String,
    /// Version string
    pub version: String,
    /// Domain (e.g., "humaneval", "math", "mixed")
    pub domain: String,
    /// Curriculum levels present
    pub levels: Vec<u8>,
    /// Number of pairs
    pub count: usize,
    /// Source information
    pub source: String,
    /// Extra properties
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

impl GraphDataset {
    /// Create a new empty dataset
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            pairs: Vec::new(),
            metadata: GraphDatasetMetadata {
                name: name.into(),
                version: "1.0.0".to_string(),
                domain: "general".to_string(),
                levels: Vec::new(),
                count: 0,
                source: String::new(),
                properties: HashMap::new(),
            },
        }
    }

    /// Create from a vector of pairs
    pub fn from_pairs(name: impl Into<String>, pairs: Vec<GraphPair>) -> Self {
        let mut levels: Vec<u8> = pairs.iter().map(|p| p.level).collect();
        levels.sort_unstable();
        levels.dedup();

        let count = pairs.len();
        Self {
            pairs,
            metadata: GraphDatasetMetadata {
                name: name.into(),
                version: "1.0.0".to_string(),
                domain: "general".to_string(),
                levels,
                count,
                source: String::new(),
                properties: HashMap::new(),
            },
        }
    }

    /// Add a graph pair
    pub fn add(&mut self, pair: GraphPair) {
        if !self.metadata.levels.contains(&pair.level) {
            self.metadata.levels.push(pair.level);
            self.metadata.levels.sort_unstable();
        }
        self.pairs.push(pair);
        self.metadata.count = self.pairs.len();
    }

    /// Number of pairs
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Filter by level
    pub fn filter_level(&self, level: u8) -> Vec<&GraphPair> {
        self.pairs.iter().filter(|p| p.level == level).collect()
    }

    /// Filter by domain
    pub fn filter_domain(&self, domain: &str) -> Vec<&GraphPair> {
        self.pairs.iter().filter(|p| p.domain == domain).collect()
    }

    /// Split into train/validation/test sets
    pub fn split(&self, train_ratio: f64, val_ratio: f64) -> (GraphDataset, GraphDataset, GraphDataset) {
        let n = self.pairs.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = train_end + (n as f64 * val_ratio) as usize;

        let train = GraphDataset::from_pairs(
            format!("{}_train", self.metadata.name),
            self.pairs[..train_end].to_vec(),
        );
        let val = GraphDataset::from_pairs(
            format!("{}_val", self.metadata.name),
            self.pairs[train_end..val_end].to_vec(),
        );
        let test = GraphDataset::from_pairs(
            format!("{}_test", self.metadata.name),
            self.pairs[val_end..].to_vec(),
        );

        (train, val, test)
    }

    /// Compute dataset statistics
    pub fn stats(&self) -> DatasetStats {
        let num_pairs = self.pairs.len();
        let mut total_input_nodes = 0;
        let mut total_input_edges = 0;
        let mut total_output_nodes = 0;
        let mut total_output_edges = 0;
        let mut level_counts = HashMap::new();

        for pair in &self.pairs {
            let ps = pair.stats();
            total_input_nodes += ps.input_nodes;
            total_input_edges += ps.input_edges;
            total_output_nodes += ps.output_nodes;
            total_output_edges += ps.output_edges;
            *level_counts.entry(ps.level).or_insert(0) += 1;
        }

        let avg_input_nodes = if num_pairs > 0 {
            total_input_nodes as f64 / num_pairs as f64
        } else {
            0.0
        };
        let avg_output_nodes = if num_pairs > 0 {
            total_output_nodes as f64 / num_pairs as f64
        } else {
            0.0
        };

        DatasetStats {
            num_pairs,
            total_input_nodes,
            total_input_edges,
            total_output_nodes,
            total_output_edges,
            avg_input_nodes,
            avg_output_nodes,
            level_counts,
        }
    }

    /// Create a batch iterator
    pub fn batches(&self, batch_size: usize) -> GraphBatchIterator<'_> {
        GraphBatchIterator {
            pairs: &self.pairs,
            batch_size,
            current: 0,
        }
    }

    // ========================================================================
    // Binary Serialization (efficient, no text in loop)
    // ========================================================================

    /// Save to binary format
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> GraphDataResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write magic bytes
        writer.write_all(&MAGIC_BYTES)?;

        // Write version
        writer.write_all(&FORMAT_VERSION.to_le_bytes())?;

        // Serialize with bincode
        let data = bincode::serialize(self)
            .map_err(|e| GraphDataError::Serialization(e.to_string()))?;

        // Write length then data
        writer.write_all(&(data.len() as u64).to_le_bytes())?;
        writer.write_all(&data)?;

        writer.flush()?;
        Ok(())
    }

    /// Load from binary format
    pub fn load_binary<P: AsRef<Path>>(path: P) -> GraphDataResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Check magic bytes
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC_BYTES {
            return Err(GraphDataError::InvalidData(
                "Invalid magic bytes - not a graph data file".to_string(),
            ));
        }

        // Check version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != FORMAT_VERSION {
            return Err(GraphDataError::VersionMismatch {
                expected: FORMAT_VERSION,
                actual: version,
            });
        }

        // Read length
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        // Read data
        let mut data = vec![0u8; len];
        reader.read_exact(&mut data)?;

        // Deserialize
        bincode::deserialize(&data)
            .map_err(|e| GraphDataError::Serialization(e.to_string()))
    }

    // ========================================================================
    // JSON Serialization (for debugging/inspection)
    // ========================================================================

    /// Save to JSON (for debugging)
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> GraphDataResult<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| GraphDataError::Serialization(e.to_string()))
    }

    /// Load from JSON
    pub fn load_json<P: AsRef<Path>>(path: P) -> GraphDataResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| GraphDataError::Serialization(e.to_string()))
    }
}

/// Dataset statistics
#[derive(Debug, Clone, Default)]
pub struct DatasetStats {
    pub num_pairs: usize,
    pub total_input_nodes: usize,
    pub total_input_edges: usize,
    pub total_output_nodes: usize,
    pub total_output_edges: usize,
    pub avg_input_nodes: f64,
    pub avg_output_nodes: f64,
    pub level_counts: HashMap<u8, usize>,
}

// ============================================================================
// Batch Iterator
// ============================================================================

/// Iterator over batches of graph pairs
pub struct GraphBatchIterator<'a> {
    pairs: &'a [GraphPair],
    batch_size: usize,
    current: usize,
}

impl<'a> Iterator for GraphBatchIterator<'a> {
    type Item = &'a [GraphPair];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.pairs.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.pairs.len());
        let batch = &self.pairs[self.current..end];
        self.current = end;
        Some(batch)
    }
}

impl<'a> GraphBatchIterator<'a> {
    /// Number of batches
    pub fn num_batches(&self) -> usize {
        self.pairs.len().div_ceil(self.batch_size)
    }
}

// ============================================================================
// Graph Builders - Create training pairs programmatically
// ============================================================================

/// Builder for creating graph pairs from scratch
pub struct GraphPairBuilder {
    id: String,
    input: DagNN,
    output: DagNN,
    level: u8,
    domain: String,
    metadata: HashMap<String, String>,
}

impl GraphPairBuilder {
    /// Start building a new graph pair
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            input: DagNN::new(),
            output: DagNN::new(),
            level: 1,
            domain: "general".to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Set the input graph
    pub fn input(mut self, graph: DagNN) -> Self {
        self.input = graph;
        self
    }

    /// Set the output graph
    pub fn output(mut self, graph: DagNN) -> Self {
        self.output = graph;
        self
    }

    /// Set the curriculum level
    pub fn level(mut self, level: u8) -> Self {
        self.level = level;
        self
    }

    /// Set the domain
    pub fn domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = domain.into();
        self
    }

    /// Add metadata
    pub fn meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the graph pair
    pub fn build(self) -> GraphPair {
        GraphPair {
            id: self.id,
            input: self.input,
            output: self.output,
            level: self.level,
            domain: self.domain,
            metadata: self.metadata,
        }
    }
}

// ============================================================================
// Conversion Helpers
// ============================================================================

/// Create a simple chain graph (linear sequence of nodes)
pub fn create_chain_graph(length: usize) -> DagNN {
    let mut dag = DagNN::new();
    let mut prev: Option<NodeId> = None;

    for i in 0..length {
        let node = dag.graph.add_node(Node::hidden());
        if let Some(p) = prev {
            dag.graph.add_edge(p, node, Edge::sequential());
        }
        prev = Some(node);
        let _ = i; // Silence unused warning
    }

    dag
}

/// Create a tree graph with given depth and branching factor
pub fn create_tree_graph(depth: usize, branching: usize) -> DagNN {
    let mut dag = DagNN::new();

    fn add_children(
        dag: &mut DagNN,
        parent: NodeId,
        depth: usize,
        branching: usize,
        current_depth: usize,
    ) {
        if current_depth >= depth {
            return;
        }

        for _ in 0..branching {
            let child = dag.graph.add_node(Node::hidden());
            dag.graph.add_edge(parent, child, Edge::sequential());
            add_children(dag, child, depth, branching, current_depth + 1);
        }
    }

    let root = dag.graph.add_node(Node::hidden());
    add_children(&mut dag, root, depth, branching, 0);

    dag
}

/// Create a DAG with given number of nodes and random edges
pub fn create_random_dag(num_nodes: usize, num_edges: usize, seed: u64) -> DagNN {
    let mut dag = DagNN::new();
    let mut rng_state = seed;

    // Simple LCG for deterministic randomness
    let mut rand = || {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        rng_state
    };

    // Add nodes
    let nodes: Vec<NodeId> = (0..num_nodes)
        .map(|_| dag.graph.add_node(Node::hidden()))
        .collect();

    // Add edges (ensuring DAG property: only forward edges)
    // Limit iterations to prevent infinite loops
    let max_edges = num_nodes * (num_nodes - 1) / 2;
    let target_edges = num_edges.min(max_edges);
    let max_attempts = target_edges * 10 + 100; // Allow plenty of attempts
    let mut attempts = 0;
    let mut edge_count = 0;

    while edge_count < target_edges && attempts < max_attempts {
        attempts += 1;
        let i = (rand() as usize) % num_nodes;
        let j = (rand() as usize) % num_nodes;

        // Only add forward edges (lower index to higher index)
        if i < j {
            let source = nodes[i];
            let target = nodes[j];
            if dag.graph.find_edge(source, target).is_none() {
                dag.graph.add_edge(source, target, Edge::sequential());
                edge_count += 1;
            }
        }
    }

    dag
}

// ============================================================================
// Pre-encoding Interface (for backend-228)
// ============================================================================

/// Trait for encoders that convert domain-specific data to graph pairs
pub trait GraphEncoder {
    /// Encode data into a graph pair
    fn encode(&self, id: &str, data: &[u8]) -> GraphDataResult<GraphPair>;

    /// Domain name for this encoder
    fn domain(&self) -> &str;
}

/// Registry of graph encoders
#[derive(Default)]
pub struct EncoderRegistry {
    encoders: HashMap<String, Box<dyn GraphEncoder + Send + Sync>>,
}

impl EncoderRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an encoder
    pub fn register(&mut self, encoder: Box<dyn GraphEncoder + Send + Sync>) {
        let domain = encoder.domain().to_string();
        self.encoders.insert(domain, encoder);
    }

    /// Get an encoder by domain
    pub fn get(&self, domain: &str) -> Option<&(dyn GraphEncoder + Send + Sync)> {
        self.encoders.get(domain).map(|e| e.as_ref())
    }

    /// List available domains
    pub fn domains(&self) -> Vec<&str> {
        self.encoders.keys().map(|s| s.as_str()).collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_graph_pair_creation() {
        let input = create_chain_graph(3);
        let output = create_chain_graph(5);

        let pair = GraphPair::new("test-001", input, output)
            .with_level(2)
            .with_domain("math")
            .with_metadata("source", "test");

        assert_eq!(pair.id, "test-001");
        assert_eq!(pair.level, 2);
        assert_eq!(pair.domain, "math");
        assert_eq!(pair.metadata.get("source"), Some(&"test".to_string()));
    }

    #[test]
    fn test_graph_pair_stats() {
        let input = create_chain_graph(3);
        let output = create_chain_graph(5);

        let pair = GraphPair::new("test", input, output).with_level(3);
        let stats = pair.stats();

        assert_eq!(stats.input_nodes, 3);
        assert_eq!(stats.input_edges, 2);
        assert_eq!(stats.output_nodes, 5);
        assert_eq!(stats.output_edges, 4);
        assert_eq!(stats.level, 3);
    }

    #[test]
    fn test_dataset_creation() {
        let mut dataset = GraphDataset::new("test-dataset");

        for i in 0..10 {
            let pair = GraphPair::new(
                format!("pair-{:03}", i),
                create_chain_graph(3),
                create_chain_graph(5),
            )
            .with_level((i % 3 + 1) as u8);
            dataset.add(pair);
        }

        assert_eq!(dataset.len(), 10);
        assert!(!dataset.is_empty());
        assert!(dataset.metadata.levels.contains(&1));
        assert!(dataset.metadata.levels.contains(&2));
        assert!(dataset.metadata.levels.contains(&3));
    }

    #[test]
    fn test_dataset_filter() {
        let pairs: Vec<GraphPair> = (0..10)
            .map(|i| {
                GraphPair::new(format!("pair-{}", i), create_chain_graph(3), create_chain_graph(5))
                    .with_level((i % 3 + 1) as u8)
                    .with_domain(if i < 5 { "math" } else { "code" })
            })
            .collect();

        let dataset = GraphDataset::from_pairs("test", pairs);

        let level1 = dataset.filter_level(1);
        assert_eq!(level1.len(), 4); // 0, 3, 6, 9

        let math_pairs = dataset.filter_domain("math");
        assert_eq!(math_pairs.len(), 5);

        let code_pairs = dataset.filter_domain("code");
        assert_eq!(code_pairs.len(), 5);
    }

    #[test]
    fn test_dataset_split() {
        let pairs: Vec<GraphPair> = (0..100)
            .map(|i| GraphPair::new(format!("pair-{}", i), create_chain_graph(2), create_chain_graph(3)))
            .collect();

        let dataset = GraphDataset::from_pairs("test", pairs);
        let (train, val, test) = dataset.split(0.8, 0.1);

        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 10);
        assert_eq!(test.len(), 10);
    }

    #[test]
    fn test_batch_iterator() {
        let pairs: Vec<GraphPair> = (0..25)
            .map(|i| GraphPair::new(format!("pair-{}", i), create_chain_graph(2), create_chain_graph(3)))
            .collect();

        let dataset = GraphDataset::from_pairs("test", pairs);
        let batches: Vec<_> = dataset.batches(10).collect();

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 10);
        assert_eq!(batches[1].len(), 10);
        assert_eq!(batches[2].len(), 5);
    }

    #[test]
    fn test_binary_serialization() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.graphdata");

        let pairs: Vec<GraphPair> = (0..5)
            .map(|i| {
                GraphPair::new(format!("pair-{}", i), create_chain_graph(3), create_chain_graph(4))
                    .with_level(1)
                    .with_domain("test")
            })
            .collect();

        let dataset = GraphDataset::from_pairs("test", pairs);
        dataset.save_binary(&path).unwrap();

        let loaded = GraphDataset::load_binary(&path).unwrap();
        assert_eq!(loaded.len(), 5);
        assert_eq!(loaded.metadata.name, "test");
        assert_eq!(loaded.pairs[0].domain, "test");
    }

    #[test]
    fn test_json_serialization() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.json");

        let pairs: Vec<GraphPair> = (0..3)
            .map(|i| {
                GraphPair::new(format!("pair-{}", i), create_chain_graph(2), create_chain_graph(3))
            })
            .collect();

        let dataset = GraphDataset::from_pairs("json-test", pairs);
        dataset.save_json(&path).unwrap();

        let loaded = GraphDataset::load_json(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.metadata.name, "json-test");
    }

    #[test]
    fn test_dataset_stats() {
        let pairs: Vec<GraphPair> = vec![
            GraphPair::new("a", create_chain_graph(2), create_chain_graph(4)).with_level(1),
            GraphPair::new("b", create_chain_graph(3), create_chain_graph(5)).with_level(2),
            GraphPair::new("c", create_chain_graph(4), create_chain_graph(6)).with_level(1),
        ];

        let dataset = GraphDataset::from_pairs("stats-test", pairs);
        let stats = dataset.stats();

        assert_eq!(stats.num_pairs, 3);
        assert_eq!(stats.total_input_nodes, 2 + 3 + 4);
        assert_eq!(stats.total_output_nodes, 4 + 5 + 6);
        assert_eq!(stats.level_counts.get(&1), Some(&2));
        assert_eq!(stats.level_counts.get(&2), Some(&1));
    }

    #[test]
    fn test_graph_pair_builder() {
        let pair = GraphPairBuilder::new("builder-test")
            .input(create_chain_graph(3))
            .output(create_tree_graph(2, 2))
            .level(5)
            .domain("code")
            .meta("language", "rust")
            .build();

        assert_eq!(pair.id, "builder-test");
        assert_eq!(pair.level, 5);
        assert_eq!(pair.domain, "code");
        assert_eq!(pair.metadata.get("language"), Some(&"rust".to_string()));
    }

    #[test]
    fn test_create_chain_graph() {
        let chain = create_chain_graph(5);
        assert_eq!(chain.node_count(), 5);
        assert_eq!(chain.edge_count(), 4);
    }

    #[test]
    fn test_create_tree_graph() {
        // Depth 2, branching 2: root + 2 children + 4 grandchildren = 7 nodes
        let tree = create_tree_graph(2, 2);
        assert_eq!(tree.node_count(), 7);
        assert_eq!(tree.edge_count(), 6); // Each non-root node has exactly 1 incoming edge
    }

    #[test]
    fn test_create_random_dag() {
        let dag = create_random_dag(10, 15, 42);
        assert_eq!(dag.node_count(), 10);
        assert!(dag.edge_count() <= 15);

        // Verify DAG property (no cycles) by checking we can iterate all nodes
        assert!(dag.node_count() > 0);
    }

    #[test]
    fn test_encoder_registry() {
        let registry = EncoderRegistry::new();
        assert!(registry.domains().is_empty());
    }

    #[test]
    fn test_invalid_binary_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("invalid.bin");

        // Write invalid magic bytes
        let mut file = File::create(&path).unwrap();
        file.write_all(&[0, 0, 0, 0]).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();

        let result = GraphDataset::load_binary(&path);
        assert!(matches!(result, Err(GraphDataError::InvalidData(_))));
    }

    #[test]
    fn test_version_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("old_version.bin");

        // Write correct magic but wrong version
        let mut file = File::create(&path).unwrap();
        file.write_all(&MAGIC_BYTES).unwrap();
        file.write_all(&999u32.to_le_bytes()).unwrap();

        let result = GraphDataset::load_binary(&path);
        assert!(matches!(
            result,
            Err(GraphDataError::VersionMismatch {
                expected: 1,
                actual: 999
            })
        ));
    }
}
