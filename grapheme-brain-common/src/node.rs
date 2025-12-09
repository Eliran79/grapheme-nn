//! Generic activated node wrapper for domain-specific brain implementations.

use serde::{Deserialize, Serialize};

/// Generic node wrapper with activation field.
///
/// This struct wraps any domain-specific node type enum and adds a learnable
/// activation value. This eliminates the need for each brain to define its own
/// `XxxNode` struct with identical activation handling.
///
/// # Type Parameters
///
/// * `T` - The domain-specific node type enum (e.g., `MathNodeType`, `CodeNodeType`)
///
/// # Example
///
/// ```ignore
/// use grapheme_brain_common::ActivatedNode;
///
/// #[derive(Clone, Debug)]
/// enum MathNodeType {
///     Number(f64),
///     Operator(char),
///     Variable(String),
/// }
///
/// fn type_activation(node_type: &MathNodeType) -> f32 {
///     match node_type {
///         MathNodeType::Number(_) => 0.8,
///         MathNodeType::Operator(_) => 0.6,
///         MathNodeType::Variable(_) => 0.7,
///     }
/// }
///
/// let node = ActivatedNode::with_type_activation(
///     MathNodeType::Number(42.0),
///     type_activation,
/// );
/// assert_eq!(node.activation, 0.8);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedNode<T> {
    /// The domain-specific node type
    pub node_type: T,
    /// Current activation level (0.0 to 1.0 typically)
    pub activation: f32,
}

impl<T> ActivatedNode<T> {
    /// Create a new activated node with default activation of 0.0
    pub fn new(node_type: T) -> Self {
        Self {
            node_type,
            activation: 0.0,
        }
    }

    /// Create a new activated node with specified activation
    pub fn with_activation(node_type: T, activation: f32) -> Self {
        Self {
            node_type,
            activation,
        }
    }

    /// Create a new activated node using a function to compute initial activation
    /// from the node type.
    ///
    /// This is useful when different node types should have different default
    /// activation levels.
    pub fn with_type_activation<F>(node_type: T, activation_fn: F) -> Self
    where
        F: FnOnce(&T) -> f32,
    {
        let activation = activation_fn(&node_type);
        Self {
            node_type,
            activation,
        }
    }

    /// Get a reference to the underlying node type
    pub fn node_type(&self) -> &T {
        &self.node_type
    }

    /// Get the current activation value
    pub fn activation(&self) -> f32 {
        self.activation
    }

    /// Set the activation value
    pub fn set_activation(&mut self, activation: f32) {
        self.activation = activation;
    }

    /// Map the node type to a new type while preserving activation
    pub fn map<U, F>(self, f: F) -> ActivatedNode<U>
    where
        F: FnOnce(T) -> U,
    {
        ActivatedNode {
            node_type: f(self.node_type),
            activation: self.activation,
        }
    }
}

impl<T: Default> Default for ActivatedNode<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: PartialEq> PartialEq for ActivatedNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.node_type == other.node_type
        // Note: activation is intentionally excluded from equality
        // Two nodes with the same type are considered equal regardless of activation
    }
}

impl<T: Eq> Eq for ActivatedNode<T> {}

impl<T: std::hash::Hash> std::hash::Hash for ActivatedNode<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.node_type.hash(state);
        // Note: activation is intentionally excluded from hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
    enum TestNodeType {
        #[default]
        A,
        B(i32),
        C(String),
    }

    #[test]
    fn test_new() {
        let node: ActivatedNode<TestNodeType> = ActivatedNode::new(TestNodeType::A);
        assert_eq!(node.node_type, TestNodeType::A);
        assert_eq!(node.activation, 0.0);
    }

    #[test]
    fn test_with_activation() {
        let node = ActivatedNode::with_activation(TestNodeType::B(42), 0.75);
        assert_eq!(node.node_type, TestNodeType::B(42));
        assert_eq!(node.activation, 0.75);
    }

    #[test]
    fn test_with_type_activation() {
        let node = ActivatedNode::with_type_activation(TestNodeType::B(10), |t| match t {
            TestNodeType::A => 0.1,
            TestNodeType::B(n) => *n as f32 / 100.0,
            TestNodeType::C(_) => 0.5,
        });
        assert_eq!(node.activation, 0.1); // 10 / 100.0
    }

    #[test]
    fn test_set_activation() {
        let mut node = ActivatedNode::new(TestNodeType::A);
        node.set_activation(0.9);
        assert_eq!(node.activation, 0.9);
    }

    #[test]
    fn test_map() {
        let node = ActivatedNode::with_activation(TestNodeType::B(5), 0.5);
        let mapped = node.map(|t| match t {
            TestNodeType::B(n) => n.to_string(),
            _ => "other".to_string(),
        });
        assert_eq!(mapped.node_type, "5");
        assert_eq!(mapped.activation, 0.5);
    }

    #[test]
    fn test_equality_ignores_activation() {
        let node1 = ActivatedNode::with_activation(TestNodeType::A, 0.1);
        let node2 = ActivatedNode::with_activation(TestNodeType::A, 0.9);
        assert_eq!(node1, node2);
    }

    #[test]
    fn test_default() {
        let node: ActivatedNode<TestNodeType> = ActivatedNode::default();
        assert_eq!(node.node_type, TestNodeType::A);
        assert_eq!(node.activation, 0.0);
    }
}
