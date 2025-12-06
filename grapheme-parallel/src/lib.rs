//! # grapheme-parallel
//!
//! Parallel-first graph processing for GRAPHEME neural network.
//!
//! This crate provides parallelism infrastructure:
//! - **ParallelGraph**: Parallel iteration over nodes and edges
//! - **BatchProcessor**: Data parallelism for batch processing
//! - **ShardedGraph**: Graph partitioning for scale
//! - **ConcurrentGraph**: Thread-safe concurrent access wrapper
//!
//! ## Design Principles
//!
//! 1. **Parallel by default**: All operations should scale with cores
//! 2. **Sequential fallback**: Simple API for single-threaded use
//! 3. **Opt-in concurrency**: Explicit synchronization when needed
//! 4. **Zero-copy where possible**: Minimize data movement

use grapheme_core::DagNN;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use thiserror::Error;

// ============================================================================
// Type Aliases
// ============================================================================

/// Graph type
pub type Graph = DagNN;

/// Node index
pub type NodeIndex = usize;

// ============================================================================
// Re-exports
// ============================================================================

pub use rayon;

// ============================================================================
// Error Types
// ============================================================================

/// Errors in parallel operations
#[derive(Error, Debug)]
pub enum ParallelError {
    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),
    #[error("Shard index out of bounds: {0} >= {1}")]
    ShardOutOfBounds(usize, usize),
    #[error("Empty batch")]
    EmptyBatch,
    #[error("Operation cancelled")]
    Cancelled,
}

/// Result type for parallel operations
pub type ParallelResult<T> = Result<T, ParallelError>;

// ============================================================================
// ParallelGraph Trait
// ============================================================================

/// Configuration for parallel operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Minimum items before parallelizing
    pub min_parallel_size: usize,
    /// Chunk size for parallel iteration
    pub chunk_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_parallel_size: 100,
            chunk_size: rayon::current_num_threads() * 4,
        }
    }
}

/// Trait for graphs that support parallel operations
///
/// All graph types implementing this trait must be Send + Sync.
pub trait ParallelGraph: Send + Sync + Debug {
    /// Get node count
    fn node_count(&self) -> usize;

    /// Get edge count
    fn edge_count(&self) -> usize;

    /// Get node indices as a parallel iterator
    fn par_node_indices(&self) -> impl ParallelIterator<Item = NodeIndex>;

    /// Parallel map over node indices
    fn par_map<F, R>(&self, f: F) -> Vec<R>
    where
        F: Fn(NodeIndex) -> R + Send + Sync,
        R: Send,
    {
        self.par_node_indices().map(f).collect()
    }

    /// Parallel filter node indices
    fn par_filter<F>(&self, predicate: F) -> Vec<NodeIndex>
    where
        F: Fn(&NodeIndex) -> bool + Send + Sync,
    {
        self.par_node_indices().filter(predicate).collect()
    }

    /// Parallel fold over node indices
    fn par_fold<R, ID, F, M>(&self, identity: ID, fold: F, merge: M) -> R
    where
        R: Send,
        ID: Fn() -> R + Send + Sync + Clone,
        F: Fn(R, NodeIndex) -> R + Send + Sync,
        M: Fn(R, R) -> R + Send + Sync,
    {
        self.par_node_indices()
            .fold(identity.clone(), fold)
            .reduce(identity, merge)
    }

    /// Check if parallel execution is beneficial
    fn should_parallelize(&self, config: &ParallelConfig) -> bool {
        self.node_count() >= config.min_parallel_size
    }
}

/// Extension trait providing additional parallel operations
pub trait ParallelGraphExt: ParallelGraph {
    /// Parallel find first matching node
    fn par_find<F>(&self, predicate: F) -> Option<NodeIndex>
    where
        F: Fn(&NodeIndex) -> bool + Send + Sync,
    {
        self.par_node_indices().find_any(predicate)
    }

    /// Parallel any - check if any node matches predicate
    fn par_any<F>(&self, predicate: F) -> bool
    where
        F: Fn(NodeIndex) -> bool + Send + Sync,
    {
        self.par_node_indices().any(predicate)
    }

    /// Parallel all - check if all nodes match predicate
    fn par_all<F>(&self, predicate: F) -> bool
    where
        F: Fn(NodeIndex) -> bool + Send + Sync,
    {
        self.par_node_indices().all(predicate)
    }

    /// Parallel count matching nodes
    fn par_count<F>(&self, predicate: F) -> usize
    where
        F: Fn(&NodeIndex) -> bool + Send + Sync,
    {
        self.par_node_indices().filter(predicate).count()
    }
}

// Implement extension for all ParallelGraph
impl<T: ParallelGraph> ParallelGraphExt for T {}

// ============================================================================
// BatchProcessor Trait
// ============================================================================

/// Trait for batch data parallelism
pub trait BatchProcessor<T>: Send + Sync {
    /// Output type for each item
    type Output: Send;

    /// Process a single item
    fn process_one(&self, item: &T) -> Self::Output;

    /// Process batch in parallel
    fn process_batch(&self, batch: &[T]) -> Vec<Self::Output>
    where
        T: Sync,
    {
        batch.par_iter().map(|item| self.process_one(item)).collect()
    }

    /// Process batch with progress callback
    fn process_batch_with_progress<F>(&self, batch: &[T], on_progress: F) -> Vec<Self::Output>
    where
        T: Sync,
        F: Fn(usize, usize) + Send + Sync,
    {
        let total = batch.len();
        let processed = std::sync::atomic::AtomicUsize::new(0);

        batch
            .par_iter()
            .map(|item| {
                let result = self.process_one(item);
                let current = processed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                on_progress(current + 1, total);
                result
            })
            .collect()
    }

    /// Suggested batch size for current hardware
    fn optimal_batch_size(&self) -> usize {
        rayon::current_num_threads() * 4
    }
}

/// Batch processor that applies a function to graphs
#[derive(Debug)]
pub struct GraphBatchProcessor<F, R>
where
    F: Fn(&Graph) -> R + Send + Sync,
    R: Send + Sync,
{
    processor: F,
    _phantom: std::marker::PhantomData<R>,
}

impl<F, R> GraphBatchProcessor<F, R>
where
    F: Fn(&Graph) -> R + Send + Sync,
    R: Send + Sync,
{
    pub fn new(processor: F) -> Self {
        Self {
            processor,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, R> BatchProcessor<Graph> for GraphBatchProcessor<F, R>
where
    F: Fn(&Graph) -> R + Send + Sync,
    R: Send + Sync,
{
    type Output = R;

    fn process_one(&self, item: &Graph) -> Self::Output {
        (self.processor)(item)
    }
}

// ============================================================================
// ConcurrentGraph Wrapper
// ============================================================================

/// Thread-safe wrapper for concurrent graph access
#[derive(Debug)]
pub struct ConcurrentGraph<G> {
    inner: Arc<RwLock<G>>,
}

impl<G> ConcurrentGraph<G> {
    /// Create a new concurrent graph wrapper
    pub fn new(graph: G) -> Self {
        Self {
            inner: Arc::new(RwLock::new(graph)),
        }
    }

    /// Get read access to the graph
    pub fn read(&self) -> ParallelResult<RwLockReadGuard<'_, G>> {
        self.inner
            .read()
            .map_err(|e| ParallelError::LockPoisoned(e.to_string()))
    }

    /// Get write access to the graph
    pub fn write(&self) -> ParallelResult<RwLockWriteGuard<'_, G>> {
        self.inner
            .write()
            .map_err(|e| ParallelError::LockPoisoned(e.to_string()))
    }

    /// Try to get read access (non-blocking)
    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, G>> {
        self.inner.try_read().ok()
    }

    /// Try to get write access (non-blocking)
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, G>> {
        self.inner.try_write().ok()
    }

    /// Execute a read operation
    pub fn with_read<F, R>(&self, f: F) -> ParallelResult<R>
    where
        F: FnOnce(&G) -> R,
    {
        let guard = self.read()?;
        Ok(f(&*guard))
    }

    /// Execute a write operation
    pub fn with_write<F, R>(&self, f: F) -> ParallelResult<R>
    where
        F: FnOnce(&mut G) -> R,
    {
        let mut guard = self.write()?;
        Ok(f(&mut *guard))
    }
}

impl<G> Clone for ConcurrentGraph<G> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<G: Default> Default for ConcurrentGraph<G> {
    fn default() -> Self {
        Self::new(G::default())
    }
}

// ============================================================================
// ShardedGraph
// ============================================================================

/// Hash function type for sharding
pub type ShardHashFn = Box<dyn Fn(NodeIndex) -> usize + Send + Sync>;

/// Graph distributed across multiple shards
#[derive(Debug)]
pub struct ShardedGraph<G> {
    shards: Vec<G>,
    num_shards: usize,
}

impl<G> ShardedGraph<G> {
    /// Create a sharded graph with the given number of shards
    pub fn new(num_shards: usize) -> Self
    where
        G: Default,
    {
        let shards = (0..num_shards).map(|_| G::default()).collect();
        Self { shards, num_shards }
    }

    /// Create from existing graphs
    pub fn from_shards(shards: Vec<G>) -> Self {
        let num_shards = shards.len();
        Self { shards, num_shards }
    }

    /// Get shard for a node index (simple modulo hashing)
    pub fn shard_index(&self, node: NodeIndex) -> usize {
        node % self.num_shards
    }

    /// Get reference to a shard
    pub fn get_shard(&self, shard_idx: usize) -> Option<&G> {
        self.shards.get(shard_idx)
    }

    /// Get mutable reference to a shard
    pub fn get_shard_mut(&mut self, shard_idx: usize) -> Option<&mut G> {
        self.shards.get_mut(shard_idx)
    }

    /// Get shard for a specific node
    pub fn shard_for(&self, node: NodeIndex) -> &G {
        &self.shards[self.shard_index(node)]
    }

    /// Get mutable shard for a specific node
    pub fn shard_for_mut(&mut self, node: NodeIndex) -> &mut G {
        let idx = self.shard_index(node);
        &mut self.shards[idx]
    }

    /// Number of shards
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Iterate over shards
    pub fn shards(&self) -> impl Iterator<Item = &G> {
        self.shards.iter()
    }

    /// Iterate over shards mutably
    pub fn shards_mut(&mut self) -> impl Iterator<Item = &mut G> {
        self.shards.iter_mut()
    }
}

impl<G: Send + Sync> ShardedGraph<G> {
    /// Parallel iterate over shards
    pub fn par_shards(&self) -> impl ParallelIterator<Item = &G> {
        self.shards.par_iter()
    }

    /// Parallel map over all shards
    pub fn par_map_shards<F, R>(&self, f: F) -> Vec<R>
    where
        F: Fn(&G) -> R + Send + Sync,
        R: Send,
    {
        self.shards.par_iter().map(f).collect()
    }

    /// Parallel fold over shards
    pub fn par_fold_shards<R, ID, F, M>(&self, identity: ID, fold: F, merge: M) -> R
    where
        R: Send,
        ID: Fn() -> R + Send + Sync + Clone,
        F: Fn(R, &G) -> R + Send + Sync,
        M: Fn(R, R) -> R + Send + Sync,
    {
        self.shards
            .par_iter()
            .fold(identity.clone(), fold)
            .reduce(identity, merge)
    }
}

// ============================================================================
// ParallelGraph Implementation for DagNN
// ============================================================================

/// Wrapper that implements ParallelGraph for DagNN
#[derive(Debug)]
pub struct ParallelDagNN {
    graph: Graph,
}

impl ParallelDagNN {
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    pub fn inner(&self) -> &Graph {
        &self.graph
    }

    pub fn into_inner(self) -> Graph {
        self.graph
    }
}

impl ParallelGraph for ParallelDagNN {
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    fn par_node_indices(&self) -> impl ParallelIterator<Item = NodeIndex> {
        (0..self.node_count()).into_par_iter()
    }
}

impl From<Graph> for ParallelDagNN {
    fn from(graph: Graph) -> Self {
        Self::new(graph)
    }
}

// ============================================================================
// Parallel Clique Operations
// ============================================================================

/// Configuration for clique operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliqueConfig {
    /// Maximum clique size to find
    pub max_clique_size: usize,
    /// Minimum clique size
    pub min_clique_size: usize,
    /// Maximum number of cliques to return
    pub max_results: usize,
    /// Timeout per shard (in milliseconds)
    pub timeout_ms: u64,
}

impl Default for CliqueConfig {
    fn default() -> Self {
        Self {
            max_clique_size: 10,
            min_clique_size: 3,
            max_results: 1000,
            timeout_ms: 5000,
        }
    }
}

/// A clique (complete subgraph)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clique {
    /// Node indices in this clique
    pub nodes: Vec<NodeIndex>,
}

impl Clique {
    pub fn new(nodes: Vec<NodeIndex>) -> Self {
        let mut sorted = nodes;
        sorted.sort();
        Self { nodes: sorted }
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains(&self, node: NodeIndex) -> bool {
        self.nodes.binary_search(&node).is_ok()
    }
}

/// Parallel clique finder
#[derive(Debug)]
pub struct ParallelCliqueFinder {
    config: CliqueConfig,
}

impl ParallelCliqueFinder {
    pub fn new(config: CliqueConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self::new(CliqueConfig::default())
    }

    /// Find cliques of a specific size in parallel
    /// Note: This is a simplified implementation
    pub fn find_k_cliques(&self, graph: &ParallelDagNN, k: usize) -> Vec<Clique> {
        if k < 2 || k > self.config.max_clique_size {
            return Vec::new();
        }

        // For small graphs or k, use sequential
        if graph.node_count() < 100 || k <= 2 {
            return self.find_k_cliques_sequential(graph, k);
        }

        // Parallel: partition nodes and search in parallel
        let node_count = graph.node_count();
        let chunk_size = (node_count / rayon::current_num_threads()).max(1);

        (0..node_count)
            .into_par_iter()
            .step_by(chunk_size)
            .flat_map(|start| {
                let end = (start + chunk_size).min(node_count);
                self.find_cliques_in_range(graph, k, start, end)
            })
            .take_any(self.config.max_results)
            .collect()
    }

    fn find_k_cliques_sequential(&self, _graph: &ParallelDagNN, k: usize) -> Vec<Clique> {
        // Simplified: return empty for now
        // Real implementation would use Bron-Kerbosch
        let _ = k;
        Vec::new()
    }

    fn find_cliques_in_range(
        &self,
        _graph: &ParallelDagNN,
        k: usize,
        start: usize,
        end: usize,
    ) -> Vec<Clique> {
        // Simplified: return single-node "cliques" if k=1
        if k == 1 {
            return (start..end).map(|i| Clique::new(vec![i])).collect();
        }
        Vec::new()
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a parallel graph wrapper
pub fn make_parallel(graph: Graph) -> ParallelDagNN {
    ParallelDagNN::new(graph)
}

/// Create a concurrent graph wrapper
pub fn make_concurrent<G>(graph: G) -> ConcurrentGraph<G> {
    ConcurrentGraph::new(graph)
}

/// Create a sharded graph
pub fn make_sharded<G: Default>(num_shards: usize) -> ShardedGraph<G> {
    ShardedGraph::new(num_shards)
}

/// Get optimal number of threads
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph(text: &str) -> Graph {
        DagNN::from_text(text).unwrap()
    }

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::default();
        assert!(config.min_parallel_size > 0);
        assert!(config.chunk_size > 0);
    }

    #[test]
    fn test_parallel_dagnn() {
        let graph = make_graph("hello world");
        let parallel = ParallelDagNN::new(graph);

        assert!(parallel.node_count() > 0);
        let indices: Vec<_> = parallel.par_node_indices().collect();
        assert_eq!(indices.len(), parallel.node_count());
    }

    #[test]
    fn test_parallel_map() {
        let graph = make_graph("test");
        let parallel = ParallelDagNN::new(graph);

        let doubled: Vec<_> = parallel.par_map(|i| i * 2);
        assert_eq!(doubled.len(), parallel.node_count());
    }

    #[test]
    fn test_parallel_filter() {
        let graph = make_graph("hello");
        let parallel = ParallelDagNN::new(graph);

        let evens: Vec<_> = parallel.par_filter(|&i| i % 2 == 0);
        assert!(evens.iter().all(|&i| i % 2 == 0));
    }

    #[test]
    fn test_parallel_find() {
        let graph = make_graph("test");
        let parallel = ParallelDagNN::new(graph);

        let found = parallel.par_find(|&i| i > 0);
        assert!(found.is_some());
    }

    #[test]
    fn test_parallel_any_all() {
        let graph = make_graph("hello");
        let parallel = ParallelDagNN::new(graph);

        assert!(parallel.par_any(|i| i < 100));
        assert!(parallel.par_all(|i| i < 1000));
    }

    #[test]
    fn test_parallel_count() {
        let graph = make_graph("test");
        let parallel = ParallelDagNN::new(graph);

        let evens = parallel.par_count(|&i| i % 2 == 0);
        assert!(evens <= parallel.node_count());
    }

    #[test]
    fn test_should_parallelize() {
        let small_graph = make_graph("hi");
        let parallel_small = ParallelDagNN::new(small_graph);

        let config = ParallelConfig {
            min_parallel_size: 1000,
            chunk_size: 4,
        };

        assert!(!parallel_small.should_parallelize(&config));
    }

    #[test]
    fn test_batch_processor() {
        let graphs: Vec<Graph> = vec![
            make_graph("hello"),
            make_graph("world"),
            make_graph("test"),
        ];

        let processor = GraphBatchProcessor::new(|g: &Graph| g.node_count());
        let counts = processor.process_batch(&graphs);

        assert_eq!(counts.len(), 3);
        assert!(counts.iter().all(|&c| c > 0));
    }

    #[test]
    fn test_concurrent_graph() {
        let graph = make_graph("test");
        let concurrent = ConcurrentGraph::new(graph);

        // Test read
        let count = concurrent.with_read(|g| g.node_count()).unwrap();
        assert!(count > 0);

        // Test clone
        let cloned = concurrent.clone();
        let count2 = cloned.with_read(|g| g.node_count()).unwrap();
        assert_eq!(count, count2);
    }

    #[test]
    fn test_concurrent_write() {
        let graph = make_graph("test");
        let concurrent = ConcurrentGraph::new(graph);

        // Write doesn't actually modify in this test
        concurrent.with_write(|_g| {
            // Would modify graph here
        }).unwrap();

        let count = concurrent.with_read(|g| g.node_count()).unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_sharded_graph() {
        let sharded: ShardedGraph<Graph> = ShardedGraph::new(4);

        assert_eq!(sharded.num_shards(), 4);
        assert!(sharded.get_shard(0).is_some());
        assert!(sharded.get_shard(4).is_none());
    }

    #[test]
    fn test_sharded_routing() {
        let graphs = vec![
            make_graph("shard0"),
            make_graph("shard1"),
            make_graph("shard2"),
            make_graph("shard3"),
        ];
        let sharded = ShardedGraph::from_shards(graphs);

        // Node 0 goes to shard 0, node 5 goes to shard 1
        assert_eq!(sharded.shard_index(0), 0);
        assert_eq!(sharded.shard_index(5), 1);
        assert_eq!(sharded.shard_index(7), 3);
    }

    #[test]
    fn test_sharded_parallel() {
        let graphs = vec![
            make_graph("one"),
            make_graph("two"),
            make_graph("three"),
        ];
        let sharded = ShardedGraph::from_shards(graphs);

        let counts: Vec<_> = sharded.par_map_shards(|g| g.node_count());
        assert_eq!(counts.len(), 3);
        assert!(counts.iter().all(|&c| c > 0));
    }

    #[test]
    fn test_clique() {
        let clique = Clique::new(vec![3, 1, 2]);
        assert_eq!(clique.nodes, vec![1, 2, 3]); // Sorted
        assert_eq!(clique.size(), 3);
        assert!(clique.contains(2));
        assert!(!clique.contains(4));
    }

    #[test]
    fn test_clique_config() {
        let config = CliqueConfig::default();
        assert!(config.max_clique_size >= config.min_clique_size);
        assert!(config.max_results > 0);
    }

    #[test]
    fn test_parallel_clique_finder() {
        let graph = make_graph("test clique");
        let parallel = ParallelDagNN::new(graph);
        let finder = ParallelCliqueFinder::with_default_config();

        // K=1 is below minimum (2), should return empty
        let cliques_k1 = finder.find_k_cliques(&parallel, 1);
        assert!(cliques_k1.is_empty());

        // K=3 (typical clique search) - works without panic
        let cliques_k3 = finder.find_k_cliques(&parallel, 3);
        // Small graph may not have 3-cliques, just check it doesn't panic
        let _ = cliques_k3;
    }

    #[test]
    fn test_factory_functions() {
        let graph = make_graph("test");

        let parallel = make_parallel(graph);
        assert!(parallel.node_count() > 0);

        let concurrent = make_concurrent(DagNN::new());
        assert!(concurrent.try_read().is_some());

        let sharded: ShardedGraph<DagNN> = make_sharded(2);
        assert_eq!(sharded.num_shards(), 2);

        assert!(num_threads() > 0);
    }

    #[test]
    fn test_batch_optimal_size() {
        let processor = GraphBatchProcessor::new(|g: &Graph| g.node_count());
        assert!(processor.optimal_batch_size() > 0);
    }
}
