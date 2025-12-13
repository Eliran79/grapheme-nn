//! Generic typed graph wrapper for domain-specific brain implementations.

use petgraph::graph::{DiGraph, NodeIndex, EdgeIndex};
use petgraph::Direction;
use serde::{Deserialize, Serialize};

/// Generic graph wrapper for domain-specific graphs.
///
/// This struct wraps petgraph's `DiGraph` and provides common operations
/// needed by all cognitive brain implementations. It eliminates the need
/// for each brain to define its own graph wrapper with identical methods.
///
/// # Type Parameters
///
/// * `N` - Node type (typically `ActivatedNode<T>` for some domain type `T`)
/// * `E` - Edge type (defaults to `()` for unweighted edges)
///
/// # Example
///
/// ```ignore
/// use grapheme_brain_common::{TypedGraph, ActivatedNode};
///
/// #[derive(Clone, Debug)]
/// enum MyNodeType { A, B }
///
/// let mut graph: TypedGraph<ActivatedNode<MyNodeType>> = TypedGraph::new();
/// let a = graph.add_node(ActivatedNode::new(MyNodeType::A));
/// let b = graph.add_node(ActivatedNode::new(MyNodeType::B));
/// graph.add_edge(a, b, ());
///
/// assert_eq!(graph.node_count(), 2);
/// assert_eq!(graph.edge_count(), 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedGraph<N, E = ()> {
    /// The underlying directed graph
    pub graph: DiGraph<N, E>,
    /// Optional root node for tree-like structures
    pub root: Option<NodeIndex>,
}

impl<N, E> TypedGraph<N, E> {
    /// Create a new empty typed graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            root: None,
        }
    }

    /// Create a new typed graph with a specified root node
    pub fn with_root(root_node: N) -> Self
    where
        E: Default,
    {
        let mut graph = DiGraph::new();
        let root = graph.add_node(root_node);
        Self {
            graph,
            root: Some(root),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: N) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: E) -> EdgeIndex {
        self.graph.add_edge(from, to, edge)
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Get a reference to a node by index
    pub fn node(&self, idx: NodeIndex) -> Option<&N> {
        self.graph.node_weight(idx)
    }

    /// Get a mutable reference to a node by index
    pub fn node_mut(&mut self, idx: NodeIndex) -> Option<&mut N> {
        self.graph.node_weight_mut(idx)
    }

    /// Get a reference to an edge by index
    pub fn edge(&self, idx: EdgeIndex) -> Option<&E> {
        self.graph.edge_weight(idx)
    }

    /// Get a mutable reference to an edge by index
    pub fn edge_mut(&mut self, idx: EdgeIndex) -> Option<&mut E> {
        self.graph.edge_weight_mut(idx)
    }

    /// Get the root node reference
    pub fn root_node(&self) -> Option<&N> {
        self.root.and_then(|idx| self.graph.node_weight(idx))
    }

    /// Get the root node index
    pub fn root_index(&self) -> Option<NodeIndex> {
        self.root
    }

    /// Set the root node
    pub fn set_root(&mut self, idx: NodeIndex) {
        self.root = Some(idx);
    }

    /// Iterator over all node indices
    pub fn node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph.node_indices()
    }

    /// Iterator over all edge indices
    pub fn edge_indices(&self) -> impl Iterator<Item = EdgeIndex> + '_ {
        self.graph.edge_indices()
    }

    /// Get neighbors of a node (outgoing edges)
    pub fn neighbors(&self, idx: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph.neighbors(idx)
    }

    /// Get predecessors of a node (incoming edges)
    pub fn predecessors(&self, idx: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph.neighbors_directed(idx, Direction::Incoming)
    }

    /// Get successors of a node (outgoing edges)
    pub fn successors(&self, idx: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph.neighbors_directed(idx, Direction::Outgoing)
    }

    /// Check if there's an edge from `a` to `b`
    pub fn has_edge(&self, from: NodeIndex, to: NodeIndex) -> bool {
        self.graph.find_edge(from, to).is_some()
    }

    /// Find edge between two nodes
    pub fn find_edge(&self, from: NodeIndex, to: NodeIndex) -> Option<EdgeIndex> {
        self.graph.find_edge(from, to)
    }

    /// Remove a node from the graph (also removes connected edges)
    pub fn remove_node(&mut self, idx: NodeIndex) -> Option<N> {
        if self.root == Some(idx) {
            self.root = None;
        }
        self.graph.remove_node(idx)
    }

    /// Remove an edge from the graph
    pub fn remove_edge(&mut self, idx: EdgeIndex) -> Option<E> {
        self.graph.remove_edge(idx)
    }

    /// Clear all nodes and edges
    pub fn clear(&mut self) {
        self.graph.clear();
        self.root = None;
    }
}

impl<N, E: Default> TypedGraph<N, E> {
    /// Add an edge with default edge weight
    pub fn add_default_edge(&mut self, from: NodeIndex, to: NodeIndex) -> EdgeIndex {
        self.graph.add_edge(from, to, E::default())
    }
}

impl<N, E> Default for TypedGraph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct TestNode(String);

    #[derive(Debug, Clone, PartialEq, Default)]
    struct TestEdge(f32);

    #[test]
    fn test_new() {
        let graph: TypedGraph<TestNode> = TypedGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.root.is_none());
    }

    #[test]
    fn test_with_root() {
        let graph: TypedGraph<TestNode, TestEdge> = TypedGraph::with_root(TestNode("root".into()));
        assert_eq!(graph.node_count(), 1);
        assert!(graph.root.is_some());
        assert_eq!(graph.root_node().unwrap().0, "root");
    }

    #[test]
    fn test_add_nodes_and_edges() {
        let mut graph: TypedGraph<TestNode, TestEdge> = TypedGraph::new();

        let a = graph.add_node(TestNode("a".into()));
        let b = graph.add_node(TestNode("b".into()));
        let c = graph.add_node(TestNode("c".into()));

        graph.add_edge(a, b, TestEdge(1.0));
        graph.add_edge(b, c, TestEdge(2.0));

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_neighbors() {
        let mut graph: TypedGraph<TestNode> = TypedGraph::new();

        let a = graph.add_node(TestNode("a".into()));
        let b = graph.add_node(TestNode("b".into()));
        let c = graph.add_node(TestNode("c".into()));

        graph.add_default_edge(a, b);
        graph.add_default_edge(a, c);

        let neighbors: Vec<_> = graph.neighbors(a).collect();
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_predecessors_successors() {
        let mut graph: TypedGraph<TestNode> = TypedGraph::new();

        let a = graph.add_node(TestNode("a".into()));
        let b = graph.add_node(TestNode("b".into()));
        let c = graph.add_node(TestNode("c".into()));

        graph.add_default_edge(a, b);
        graph.add_default_edge(c, b);

        let preds: Vec<_> = graph.predecessors(b).collect();
        assert_eq!(preds.len(), 2);

        let succs: Vec<_> = graph.successors(a).collect();
        assert_eq!(succs.len(), 1);
    }

    #[test]
    fn test_has_edge() {
        let mut graph: TypedGraph<TestNode> = TypedGraph::new();

        let a = graph.add_node(TestNode("a".into()));
        let b = graph.add_node(TestNode("b".into()));

        assert!(!graph.has_edge(a, b));
        graph.add_default_edge(a, b);
        assert!(graph.has_edge(a, b));
        assert!(!graph.has_edge(b, a)); // Directed
    }

    #[test]
    fn test_remove_node() {
        let mut graph: TypedGraph<TestNode> = TypedGraph::new();

        let a = graph.add_node(TestNode("a".into()));
        let b = graph.add_node(TestNode("b".into()));
        graph.add_default_edge(a, b);

        graph.set_root(a);
        assert!(graph.root.is_some());

        let removed = graph.remove_node(a);
        assert_eq!(removed.unwrap().0, "a");
        assert!(graph.root.is_none());
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut graph: TypedGraph<TestNode> = TypedGraph::new();

        let a = graph.add_node(TestNode("a".into()));
        let b = graph.add_node(TestNode("b".into()));
        graph.add_default_edge(a, b);
        graph.set_root(a);

        graph.clear();
        assert!(graph.is_empty());
        assert!(graph.root.is_none());
    }

    #[test]
    fn test_node_access() {
        let mut graph: TypedGraph<TestNode> = TypedGraph::new();
        let a = graph.add_node(TestNode("a".into()));

        assert_eq!(graph.node(a).unwrap().0, "a");

        graph.node_mut(a).unwrap().0 = "modified".into();
        assert_eq!(graph.node(a).unwrap().0, "modified");
    }
}
