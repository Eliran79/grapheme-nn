//! Backpropagation through graph structures
//!
//! This module implements reverse-mode autodiff for graph neural network operations.
//! Key features:
//! - Tape for recording operations during forward pass
//! - Backward pass respecting DAG topological order
//! - Gradient accumulation at nodes
//! - Chain rule through edges
//! - Gradient clipping utilities
//!
//! Backend-027: Backpropagation through graph structures

use ndarray::Array1;
use std::collections::{HashMap, VecDeque};

// ============================================================================
// Tape for recording operations
// ============================================================================

/// An operation recorded on the tape
#[derive(Debug, Clone)]
pub enum TapeOp {
    /// Node embedding lookup: node_id -> embedding
    NodeEmbed {
        node_id: usize,
        output_idx: usize,
    },
    /// Edge transformation: (source_embed, target_embed) -> message
    EdgeTransform {
        source_idx: usize,
        target_idx: usize,
        output_idx: usize,
    },
    /// Node aggregation: [messages] -> aggregated
    Aggregate {
        input_indices: Vec<usize>,
        output_idx: usize,
        agg_type: AggregationType,
    },
    /// Linear transformation: input @ weights + bias
    Linear {
        input_idx: usize,
        output_idx: usize,
        weight_id: usize,
    },
    /// Activation function
    Activation {
        input_idx: usize,
        output_idx: usize,
        act_type: ActivationType,
    },
    /// Combine operation: element-wise operation on two inputs
    Combine {
        left_idx: usize,
        right_idx: usize,
        output_idx: usize,
        combine_type: CombineType,
    },
}

/// Aggregation types for message passing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationType {
    Sum,
    Mean,
    Max,
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    ReLU,
    Tanh,
    Sigmoid,
    LeakyReLU(f32),
}

/// Combination operation types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CombineType {
    Add,
    Multiply,
    Concat,
}

/// Tape for recording operations during forward pass
#[derive(Debug)]
pub struct Tape {
    /// Recorded operations in forward order
    ops: Vec<TapeOp>,
    /// Cached intermediate values (index -> value)
    values: HashMap<usize, Array1<f32>>,
    /// Gradients accumulated at each index
    gradients: HashMap<usize, Array1<f32>>,
    /// Counter for generating unique indices
    next_idx: usize,
    /// Whether tape is recording
    recording: bool,
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Tape {
    /// Create a new empty tape
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            values: HashMap::new(),
            gradients: HashMap::new(),
            next_idx: 0,
            recording: true,
        }
    }

    /// Start recording operations
    pub fn start_recording(&mut self) {
        self.recording = true;
    }

    /// Stop recording operations
    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    /// Clear the tape (keeping values, clearing gradients)
    pub fn zero_grad(&mut self) {
        self.gradients.clear();
    }

    /// Clear all recorded operations and values
    pub fn clear(&mut self) {
        self.ops.clear();
        self.values.clear();
        self.gradients.clear();
        self.next_idx = 0;
    }

    /// Allocate a new index and store a value
    pub fn store_value(&mut self, value: Array1<f32>) -> usize {
        let idx = self.next_idx;
        self.next_idx += 1;
        self.values.insert(idx, value);
        idx
    }

    /// Get a stored value
    pub fn get_value(&self, idx: usize) -> Option<&Array1<f32>> {
        self.values.get(&idx)
    }

    /// Get a gradient
    pub fn get_grad(&self, idx: usize) -> Option<&Array1<f32>> {
        self.gradients.get(&idx)
    }

    /// Record a node embedding operation
    pub fn record_node_embed(&mut self, node_id: usize, embedding: Array1<f32>) -> usize {
        let output_idx = self.store_value(embedding);
        if self.recording {
            self.ops.push(TapeOp::NodeEmbed { node_id, output_idx });
        }
        output_idx
    }

    /// Record an edge transform operation
    pub fn record_edge_transform(
        &mut self,
        source_idx: usize,
        target_idx: usize,
        output: Array1<f32>,
    ) -> usize {
        let output_idx = self.store_value(output);
        if self.recording {
            self.ops.push(TapeOp::EdgeTransform {
                source_idx,
                target_idx,
                output_idx,
            });
        }
        output_idx
    }

    /// Record an aggregation operation
    pub fn record_aggregate(
        &mut self,
        input_indices: Vec<usize>,
        output: Array1<f32>,
        agg_type: AggregationType,
    ) -> usize {
        let output_idx = self.store_value(output);
        if self.recording {
            self.ops.push(TapeOp::Aggregate {
                input_indices,
                output_idx,
                agg_type,
            });
        }
        output_idx
    }

    /// Record a linear transformation
    pub fn record_linear(
        &mut self,
        input_idx: usize,
        output: Array1<f32>,
        weight_id: usize,
    ) -> usize {
        let output_idx = self.store_value(output);
        if self.recording {
            self.ops.push(TapeOp::Linear {
                input_idx,
                output_idx,
                weight_id,
            });
        }
        output_idx
    }

    /// Record an activation function
    pub fn record_activation(
        &mut self,
        input_idx: usize,
        output: Array1<f32>,
        act_type: ActivationType,
    ) -> usize {
        let output_idx = self.store_value(output);
        if self.recording {
            self.ops.push(TapeOp::Activation {
                input_idx,
                output_idx,
                act_type,
            });
        }
        output_idx
    }

    /// Record a combine operation
    pub fn record_combine(
        &mut self,
        left_idx: usize,
        right_idx: usize,
        output: Array1<f32>,
        combine_type: CombineType,
    ) -> usize {
        let output_idx = self.store_value(output);
        if self.recording {
            self.ops.push(TapeOp::Combine {
                left_idx,
                right_idx,
                output_idx,
                combine_type,
            });
        }
        output_idx
    }

    /// Get number of recorded operations
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Run backward pass from a given output index
    ///
    /// This implements reverse-mode autodiff through the recorded operations.
    /// Gradients flow from output to inputs respecting the computation order.
    pub fn backward(&mut self, output_idx: usize, grad_output: Array1<f32>) -> BackwardResult {
        // Seed the gradient at the output
        self.gradients.insert(output_idx, grad_output);

        // Clone ops to avoid borrow conflict
        let ops: Vec<TapeOp> = self.ops.iter().rev().cloned().collect();

        // Process operations in reverse order
        for op in &ops {
            self.backward_op(op);
        }

        // Collect results
        BackwardResult {
            node_gradients: self.collect_node_gradients(),
            weight_gradients: self.collect_weight_gradients(),
        }
    }

    /// Backward pass for a single operation
    fn backward_op(&mut self, op: &TapeOp) {
        match op {
            TapeOp::NodeEmbed { node_id: _, output_idx } => {
                // Node embeddings are leaf nodes - gradient just accumulates
                // The gradient is already stored, will be collected later
                let _ = output_idx; // Gradient stays at output_idx
            }
            TapeOp::EdgeTransform {
                source_idx,
                target_idx,
                output_idx,
            } => {
                // Edge transform: backward splits gradient to source and target
                if let Some(grad) = self.gradients.get(output_idx).cloned() {
                    // For simplicity, we split equally - in real impl this depends on transform
                    let half_grad = &grad * 0.5;
                    self.accumulate_grad(*source_idx, half_grad.clone());
                    self.accumulate_grad(*target_idx, half_grad);
                }
            }
            TapeOp::Aggregate {
                input_indices,
                output_idx,
                agg_type,
            } => {
                if let Some(grad) = self.gradients.get(output_idx).cloned() {
                    match agg_type {
                        AggregationType::Sum => {
                            // Sum: gradient flows equally to all inputs
                            for idx in input_indices {
                                self.accumulate_grad(*idx, grad.clone());
                            }
                        }
                        AggregationType::Mean => {
                            // Mean: gradient divided by count
                            let n = input_indices.len() as f32;
                            let scaled_grad = &grad / n;
                            for idx in input_indices {
                                self.accumulate_grad(*idx, scaled_grad.clone());
                            }
                        }
                        AggregationType::Max => {
                            // Max: gradient only flows to the max element
                            // This requires knowing which element was max during forward
                            // For now, distribute equally (simplified)
                            let n = input_indices.len() as f32;
                            let scaled_grad = &grad / n;
                            for idx in input_indices {
                                self.accumulate_grad(*idx, scaled_grad.clone());
                            }
                        }
                    }
                }
            }
            TapeOp::Linear {
                input_idx,
                output_idx,
                weight_id: _,
            } => {
                // Linear: gradient flows through (simplified - assumes identity weights)
                if let Some(grad) = self.gradients.get(output_idx).cloned() {
                    self.accumulate_grad(*input_idx, grad);
                }
            }
            TapeOp::Activation {
                input_idx,
                output_idx,
                act_type,
            } => {
                if let (Some(grad), Some(input)) = (
                    self.gradients.get(output_idx).cloned(),
                    self.values.get(input_idx).cloned(),
                ) {
                    let local_grad = match act_type {
                        ActivationType::ReLU => {
                            // ReLU: grad * (input > 0)
                            input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
                        }
                        ActivationType::Tanh => {
                            // Tanh: grad * (1 - tanh(x)^2)
                            input.mapv(|x| 1.0 - x.tanh().powi(2))
                        }
                        ActivationType::Sigmoid => {
                            // Sigmoid: grad * sig(x) * (1 - sig(x))
                            input.mapv(|x| {
                                let s = 1.0 / (1.0 + (-x).exp());
                                s * (1.0 - s)
                            })
                        }
                        ActivationType::LeakyReLU(alpha) => {
                            // LeakyReLU: grad * (1 if x > 0 else alpha)
                            input.mapv(|x| if x > 0.0 { 1.0 } else { *alpha })
                        }
                    };
                    let combined_grad = &grad * &local_grad;
                    self.accumulate_grad(*input_idx, combined_grad);
                }
            }
            TapeOp::Combine {
                left_idx,
                right_idx,
                output_idx,
                combine_type,
            } => {
                if let Some(grad) = self.gradients.get(output_idx).cloned() {
                    match combine_type {
                        CombineType::Add => {
                            // Add: gradient flows to both inputs
                            self.accumulate_grad(*left_idx, grad.clone());
                            self.accumulate_grad(*right_idx, grad);
                        }
                        CombineType::Multiply => {
                            // Multiply: gradient * other input
                            if let (Some(left), Some(right)) = (
                                self.values.get(left_idx).cloned(),
                                self.values.get(right_idx).cloned(),
                            ) {
                                self.accumulate_grad(*left_idx, &grad * &right);
                                self.accumulate_grad(*right_idx, &grad * &left);
                            }
                        }
                        CombineType::Concat => {
                            // Concat: split gradient (simplified)
                            let half = grad.len() / 2;
                            if half > 0 {
                                self.accumulate_grad(*left_idx, grad.slice(ndarray::s![..half]).to_owned());
                                self.accumulate_grad(*right_idx, grad.slice(ndarray::s![half..]).to_owned());
                            }
                        }
                    }
                }
            }
        }
    }

    /// Accumulate gradient at an index
    fn accumulate_grad(&mut self, idx: usize, grad: Array1<f32>) {
        self.gradients
            .entry(idx)
            .and_modify(|existing| *existing = &*existing + &grad)
            .or_insert(grad);
    }

    /// Collect gradients for node embeddings
    fn collect_node_gradients(&self) -> HashMap<usize, Array1<f32>> {
        let mut node_grads = HashMap::new();
        for op in &self.ops {
            if let TapeOp::NodeEmbed { node_id, output_idx } = op {
                if let Some(grad) = self.gradients.get(output_idx) {
                    node_grads.insert(*node_id, grad.clone());
                }
            }
        }
        node_grads
    }

    /// Collect gradients for weight matrices
    fn collect_weight_gradients(&self) -> HashMap<usize, Array1<f32>> {
        // Simplified: in real implementation, would compute proper weight gradients
        HashMap::new()
    }
}

/// Result of backward pass
#[derive(Debug)]
pub struct BackwardResult {
    /// Gradients at node embeddings (node_id -> gradient)
    pub node_gradients: HashMap<usize, Array1<f32>>,
    /// Gradients for weight matrices (weight_id -> gradient)
    pub weight_gradients: HashMap<usize, Array1<f32>>,
}

impl BackwardResult {
    /// Get gradient for a node
    pub fn node_grad(&self, node_id: usize) -> Option<&Array1<f32>> {
        self.node_gradients.get(&node_id)
    }

    /// Get total gradient norm
    pub fn total_norm(&self) -> f32 {
        let mut total = 0.0;
        for grad in self.node_gradients.values() {
            total += grad.iter().map(|x| x * x).sum::<f32>();
        }
        for grad in self.weight_gradients.values() {
            total += grad.iter().map(|x| x * x).sum::<f32>();
        }
        total.sqrt()
    }
}

// ============================================================================
// Topological ordering for graphs
// ============================================================================

/// Compute topological ordering of a DAG
///
/// Returns nodes in order such that all predecessors come before successors.
/// Returns None if the graph contains cycles.
pub fn topological_sort(
    num_nodes: usize,
    edges: &[(usize, usize)],
) -> Option<Vec<usize>> {
    // Build adjacency list and in-degree count
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    let mut in_degree = vec![0usize; num_nodes];

    for &(from, to) in edges {
        if from < num_nodes && to < num_nodes {
            adj[from].push(to);
            in_degree[to] += 1;
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (node, &degree) in in_degree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(node);
        }
    }

    let mut result = Vec::with_capacity(num_nodes);
    while let Some(node) = queue.pop_front() {
        result.push(node);
        for &neighbor in &adj[node] {
            in_degree[neighbor] -= 1;
            if in_degree[neighbor] == 0 {
                queue.push_back(neighbor);
            }
        }
    }

    if result.len() == num_nodes {
        Some(result)
    } else {
        None // Cycle detected
    }
}

/// Get reverse topological order (for backward pass)
pub fn reverse_topological_sort(
    num_nodes: usize,
    edges: &[(usize, usize)],
) -> Option<Vec<usize>> {
    topological_sort(num_nodes, edges).map(|mut order| {
        order.reverse();
        order
    })
}

// ============================================================================
// Gradient utilities
// ============================================================================

/// Clip gradients by global norm
pub fn clip_grad_norm(
    gradients: &mut HashMap<usize, Array1<f32>>,
    max_norm: f32,
) -> f32 {
    // Compute total norm
    let mut total_norm_sq = 0.0;
    for grad in gradients.values() {
        total_norm_sq += grad.iter().map(|x| x * x).sum::<f32>();
    }
    let total_norm = total_norm_sq.sqrt();

    // Scale if necessary
    if total_norm > max_norm && total_norm > 1e-8 {
        let scale = max_norm / total_norm;
        for grad in gradients.values_mut() {
            grad.mapv_inplace(|x| x * scale);
        }
    }

    total_norm
}

/// Clip gradients by value
pub fn clip_grad_value(
    gradients: &mut HashMap<usize, Array1<f32>>,
    max_value: f32,
) {
    for grad in gradients.values_mut() {
        grad.mapv_inplace(|x| x.clamp(-max_value, max_value));
    }
}

// ============================================================================
// Numerical gradient checking
// ============================================================================

/// Numerical gradient check for verifying analytical gradients
///
/// Computes numerical gradient using finite differences and compares
/// with the provided analytical gradient.
pub fn gradient_check(
    f: impl Fn(&Array1<f32>) -> f32,
    x: &Array1<f32>,
    analytical_grad: &Array1<f32>,
    epsilon: f32,
) -> GradientCheckResult {
    let mut numerical_grad = Array1::zeros(x.len());

    for i in 0..x.len() {
        // f(x + e) - f(x - e) / (2*e)
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[i] += epsilon;
        x_minus[i] -= epsilon;

        numerical_grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * epsilon);
    }

    // Compute relative error
    let diff = &numerical_grad - analytical_grad;
    let diff_norm = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
    let numer_norm = numerical_grad.iter().map(|x| x * x).sum::<f32>().sqrt();
    let anal_norm = analytical_grad.iter().map(|x| x * x).sum::<f32>().sqrt();

    let relative_error = if numer_norm + anal_norm > 1e-10 {
        diff_norm / (numer_norm + anal_norm)
    } else {
        diff_norm
    };

    GradientCheckResult {
        numerical_grad,
        analytical_grad: analytical_grad.clone(),
        relative_error,
        passed: relative_error < 1e-3, // Threshold for f32 precision
    }
}

/// Result of gradient check
#[derive(Debug)]
pub struct GradientCheckResult {
    /// Numerically computed gradient
    pub numerical_grad: Array1<f32>,
    /// Analytically computed gradient
    pub analytical_grad: Array1<f32>,
    /// Relative error between gradients
    pub relative_error: f32,
    /// Whether check passed (relative_error < threshold)
    pub passed: bool,
}

// ============================================================================
// Graph-specific backward utilities
// ============================================================================

/// Backward pass through message passing layer
pub struct MessagePassingBackward {
    /// Gradients for each node's embedding
    pub node_grads: HashMap<usize, Array1<f32>>,
    /// Gradients for edge weights
    pub edge_grads: HashMap<(usize, usize), f32>,
}

impl MessagePassingBackward {
    /// Create empty backward result
    pub fn new() -> Self {
        Self {
            node_grads: HashMap::new(),
            edge_grads: HashMap::new(),
        }
    }

    /// Accumulate gradient for a node
    pub fn accumulate_node_grad(&mut self, node_id: usize, grad: Array1<f32>) {
        self.node_grads
            .entry(node_id)
            .and_modify(|existing| *existing = &*existing + &grad)
            .or_insert(grad);
    }

    /// Accumulate gradient for an edge
    pub fn accumulate_edge_grad(&mut self, src: usize, dst: usize, grad: f32) {
        *self.edge_grads.entry((src, dst)).or_insert(0.0) += grad;
    }
}

impl Default for MessagePassingBackward {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_basic() {
        let mut tape = Tape::new();

        // Record some operations
        let embed1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let embed2 = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let idx1 = tape.record_node_embed(0, embed1);
        let idx2 = tape.record_node_embed(1, embed2);

        assert_eq!(tape.op_count(), 2);
        assert!(tape.get_value(idx1).is_some());
        assert!(tape.get_value(idx2).is_some());
    }

    #[test]
    fn test_tape_backward_simple() {
        let mut tape = Tape::new();

        // Simulate: output = input1 + input2
        let input1 = Array1::from_vec(vec![1.0, 2.0]);
        let input2 = Array1::from_vec(vec![3.0, 4.0]);
        let output = &input1 + &input2;

        let idx1 = tape.record_node_embed(0, input1);
        let idx2 = tape.record_node_embed(1, input2);
        let out_idx = tape.record_combine(idx1, idx2, output, CombineType::Add);

        // Backward with gradient of 1s
        let grad_output = Array1::from_vec(vec![1.0, 1.0]);
        let result = tape.backward(out_idx, grad_output);

        // For addition, gradient should flow to both inputs
        assert!(result.node_gradients.contains_key(&0));
        assert!(result.node_gradients.contains_key(&1));
    }

    #[test]
    fn test_topological_sort() {
        // DAG: 0 -> 1 -> 2
        let edges = vec![(0, 1), (1, 2)];
        let order = topological_sort(3, &edges);
        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn test_topological_sort_cycle() {
        // Cycle: 0 -> 1 -> 2 -> 0
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let order = topological_sort(3, &edges);
        assert!(order.is_none()); // Should detect cycle
    }

    #[test]
    fn test_clip_grad_norm() {
        let mut grads = HashMap::new();
        grads.insert(0, Array1::from_vec(vec![3.0, 4.0])); // norm = 5

        let norm = clip_grad_norm(&mut grads, 2.5); // clip to half

        assert!((norm - 5.0).abs() < 1e-6);
        let clipped_norm: f32 = grads[&0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((clipped_norm - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_check() {
        // Simple quadratic: f(x) = x^2, grad = 2x
        let f = |x: &Array1<f32>| x.iter().map(|xi| xi * xi).sum();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let analytical = x.mapv(|xi| 2.0 * xi);

        let result = gradient_check(f, &x, &analytical, 1e-5);
        assert!(result.passed, "Gradient check failed: relative error = {}", result.relative_error);
    }

    #[test]
    fn test_activation_backward_relu() {
        let mut tape = Tape::new();

        let input: Array1<f32> = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
        let output = input.mapv(|x: f32| x.max(0.0)); // ReLU forward

        let input_idx = tape.store_value(input);
        let output_idx = tape.record_activation(input_idx, output, ActivationType::ReLU);

        // Backward
        let grad_output = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let _ = tape.backward(output_idx, grad_output);

        // Check gradient
        if let Some(grad) = tape.get_grad(input_idx) {
            // ReLU: gradient is 0 for negative inputs, 1 for positive
            assert!((grad[0] - 0.0).abs() < 1e-6); // x=-1, grad=0
            assert!((grad[2] - 1.0).abs() < 1e-6); // x=1, grad=1
            assert!((grad[3] - 1.0).abs() < 1e-6); // x=2, grad=1
        }
    }

    #[test]
    fn test_aggregation_backward_mean() {
        let mut tape = Tape::new();

        let v1 = Array1::from_vec(vec![1.0, 2.0]);
        let v2 = Array1::from_vec(vec![3.0, 4.0]);
        let v3 = Array1::from_vec(vec![5.0, 6.0]);

        let idx1 = tape.store_value(v1);
        let idx2 = tape.store_value(v2);
        let idx3 = tape.store_value(v3);

        // Mean aggregation
        let mean = Array1::from_vec(vec![3.0, 4.0]); // (1+3+5)/3, (2+4+6)/3
        let out_idx = tape.record_aggregate(
            vec![idx1, idx2, idx3],
            mean,
            AggregationType::Mean,
        );

        let grad_output = Array1::from_vec(vec![1.0, 1.0]);
        let _ = tape.backward(out_idx, grad_output);

        // Mean: gradient divided by count
        if let Some(grad) = tape.get_grad(idx1) {
            let expected = 1.0 / 3.0;
            assert!((grad[0] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_message_passing_backward() {
        let mut mp_backward = MessagePassingBackward::new();

        // Accumulate gradients
        mp_backward.accumulate_node_grad(0, Array1::from_vec(vec![1.0, 2.0]));
        mp_backward.accumulate_node_grad(0, Array1::from_vec(vec![0.5, 0.5]));

        // Should sum gradients
        if let Some(grad) = mp_backward.node_grads.get(&0) {
            assert!((grad[0] - 1.5).abs() < 1e-6);
            assert!((grad[1] - 2.5).abs() < 1e-6);
        }
    }
}
