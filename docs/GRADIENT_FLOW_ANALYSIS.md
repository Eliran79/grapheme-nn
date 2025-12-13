# Gradient Flow Analysis (Backend-100)

## Executive Summary

**Status:** ✅ **Problem identified**, ⚠️ **Partial fixes applied**, 🚧 **Full solution requires new task**

**Core Issue:** Backward pass doesn't route gradients through graph morphing operations.

**Impact:** Embeddings cannot learn - loss increases instead of decreasing.

## The Problem

### Current Flow (Broken)

```
Forward Pass:
  Input "abc" → nodes [a, b, c]
       ↓
  Embeddings computed → activations set
       ↓
  Merge b+c based on similarity → nodes [a, bc]
       ↓
  Structural loss compares [a, bc] vs target
       ↓
  Returns gradients for nodes [a, bc]

Backward Pass (BROKEN):
  Gradients for [a, bc] ← from loss
       ↓
  Try to map to input nodes [a, b, c] ❌
       ↓
  No tracking of which input nodes → which output nodes!
       ↓
  Gradient routing fails → embeddings don't learn
```

### Root Cause

**Graph structure changes during forward pass**, but backward pass assumes 1:1 mapping.

When node_j merges into node_i:
- Forward: Updates activation, transfers edges ✓
- Backward: No gradient split - node_j's gradient is lost! ✗

## Experimental Evidence

### Test 1: Gradient Descent Direction

```bash
cargo run --release --bin test_gradient_descent
```

**Results:**
```
Training: 'abc' → 'ab', 'xyz' → 'xy', '123' → '12'

Epoch  0: loss=4.157602
Epoch  5: loss=4.158530  (+0.001 increase!)
Epoch 10: loss=4.158614
Epoch 15: loss=4.158614

Total change: +0.001231 (+0.03%)
Decreases: 0
Increases: 20
✗ FAIL: Not converging
```

**Conclusion:** Gradients point in WRONG direction → loss increases.

### Test 2: Activation Weight Experiment

Changed structural loss weights:
- Before: `0.6 * type_cost + 0.2 * activation_cost`
- After: `0.1 * type_cost + 0.7 * activation_cost`

**Result:** Loss still increases - weight doesn't fix routing problem.

### Test 3: Activation Merging

Added activation averaging during node merge (grapheme-core/src/lib.rs:4649-4652):
```rust
let act_i = morphed_graph.graph[node_i].activation;
let act_j = morphed_graph.graph[node_j].activation;
morphed_graph.graph[node_i].activation = (act_i + act_j) / 2.0;
```

**Result:** Still doesn't converge - backward pass is the issue.

## Partial Fixes Applied

### ✅ Fix 1: Increased Activation Weight in Loss

**File:** `grapheme-train/src/lib.rs:2080`

**Change:**
```rust
// OLD: 0.6 * type_cost + 0.2 * activation_cost + 0.2 * degree_cost
// NEW: 0.1 * type_cost + 0.7 * activation_cost + 0.2 * degree_cost
```

**Rationale:** Activation is LEARNED (from embeddings), type/degree are fixed. This emphasizes the learnable component.

**Impact:** Loss now depends more on activations, but gradients still don't flow correctly.

### ✅ Fix 2: Activation Merging During Node Merge

**File:** `grapheme-core/src/lib.rs:4649-4652`

**Change:**
```rust
// Average activations when merging nodes
let act_i = morphed_graph.graph[node_i].activation;
let act_j = morphed_graph.graph[node_j].activation;
morphed_graph.graph[node_i].activation = (act_i + act_j) / 2.0;
```

**Rationale:** Merged node should represent both original nodes.

**Impact:** Forward pass is more principled, but backward still broken.

## The Real Solution (Requires New Task)

### Option 1: Track Merge History (Recommended)

Store which input nodes contributed to each output node:

```rust
pub struct MergeHistory {
    // output_node_id → Vec<(input_node_id, weight)>
    node_mapping: HashMap<NodeId, Vec<(NodeId, f32)>>,
}

impl GraphTransformNet {
    pub fn forward(&self, input: &Graph) -> (Graph, MergeHistory) {
        // ... morphing ...

        // When merging node_j into node_i:
        history.add_merge(node_i, vec![
            (orig_node_i, 0.5),  // Equal contribution
            (orig_node_j, 0.5),
        ]);

        (morphed_graph, history)
    }

    pub fn backward(&mut self, history: &MergeHistory, output_grads: &[f32]) {
        // Route gradients back through merge history
        for (output_node, input_contributors) in &history.node_mapping {
            let grad = output_grads[output_node];
            for (input_node, weight) in input_contributors {
                // Split gradient proportionally
                input_grads[input_node] += grad * weight;
            }
        }

        // Now backprop to embeddings
        for (input_node, grad) in input_grads {
            let ch = get_char(input_node);
            self.embedding.backward(ch, grad);
        }
    }
}
```

**Complexity:** O(n) tracking, O(n) gradient routing - still polynomial! ✓

### Option 2: Straight-Through Estimator

Treat merge as continuous during backward:

```rust
// Forward: Hard merge (discrete)
if similarity > threshold {
    merge(node_i, node_j);
}

// Backward: Pretend it was soft (continuous approximation)
let merge_weight = sigmoid((similarity - threshold) / temperature);
grad_i = (1 - merge_weight) * output_grad;
grad_j = merge_weight * output_grad;
```

**Pros:** Simple, well-studied technique
**Cons:** Approximation - not exact gradient

### Option 3: Policy Gradient (REINFORCE)

Treat merging as discrete action, use policy gradient:

```rust
// Probability of merging
let p_merge = sigmoid((similarity - threshold) / temp);

// Sample action
let action = if rand() < p_merge { Merge } else { Keep };

// Backward: Policy gradient
let advantage = baseline_loss - current_loss;
threshold_grad = advantage * p_merge * (1 - p_merge) * similarity_grad;
```

**Pros:** Theoretically sound for discrete decisions
**Cons:** High variance, needs baseline

## Comparison to Early Prototypes

From exploration of `/data/git/DagNeuralNetwork/`:

**Early prototypes stored activations:**
```python
# Forward pass (trainer.py:56)
node_attrs['input_value'] = z  # Pre-activation
node_attrs['output'] = activate(z)  # Post-activation

# Backward pass (trainer.py:85)
grad = error * deriv_func(node['input_value'])  # Uses stored value!
```

**Key insight:** They stored intermediate values during forward, used them in backward!

**GRAPHEME needs similar tracking:**
- Forward: Store merge decisions and activation contributions
- Backward: Use stored info to route gradients correctly

## DAG Advantages (Still No NP-Hard!)

All proposed solutions maintain polynomial complexity:

**Option 1 (Merge History):**
- Storage: O(n) mappings
- Gradient routing: O(n) per backward pass
- Total: O(n) - **linear time!** ✓

**Option 2 (Straight-Through):**
- No extra storage needed
- Gradient computation: O(n²) (same as current)
- Total: O(n²) - polynomial ✓

**Option 3 (REINFORCE):**
- Baseline computation: O(1) per example
- Policy gradient: O(1) per merge decision
- Total: O(m) where m = merge decisions ✓

**DAG properties help:**
- Topological order: Process nodes sequentially
- No cycles: Gradient flow is acyclic (no loops)
- Sparse edges: O(n) edges typically, not O(n²)

## Recommendations

### Immediate (Backend-104): Implement Merge History

1. Add `MergeHistory` struct to track mappings
2. Update `forward()` to return `(graph, history)`
3. Update `backward()` to use history for gradient routing
4. Test with gradient descent direction test

**Expected result:** Loss should decrease!

### Medium-term: Improve Gradient Estimation

1. Implement straight-through estimator as fallback
2. Add finite difference tests to verify correctness
3. Compare different gradient estimators

### Long-term: Advanced Morphing

1. Add edge insertion/deletion (not just node merging)
2. Implement node splitting (opposite of merging)
3. Multi-layer message passing before morphing

## Testing Strategy

After implementing merge history:

```bash
# 1. Gradient descent direction
cargo run --release --bin test_gradient_descent
# Expected: Loss DECREASES monotonically

# 2. Finite difference check
cargo run --release --bin test_gradient_check
# Expected: Analytical ≈ Numerical (within 1%)

# 3. Threshold learning
cargo run --release --bin train_with_threshold_tracking
# Expected: Threshold adapts, loss decreases

# 4. Full training
cargo run --release --bin train -- --data data/generated --epochs 100
# Expected: Sustained loss decrease
```

## Key Takeaways

✅ **Problem identified:** Backward pass doesn't route gradients through morphing

✅ **Partial fixes applied:** Activation weight increased, merging improved

⚠️ **Core issue remains:** Need merge history tracking

✅ **Solution is tractable:** O(n) complexity, no NP-hard operations

✅ **DAG structure helps:** Acyclic gradient flow, sparse connectivity

🚧 **Next task:** Backend-104 - Implement gradient routing with merge history

## Files Modified

### grapheme-train/src/lib.rs:2080
```rust
// Increased activation weight from 0.2 to 0.7
let cost = 0.1 * type_cost + 0.7 * activation_cost + 0.2 * degree_cost;
```

### grapheme-core/src/lib.rs:4649-4652
```rust
// Average activations when merging nodes
let act_i = morphed_graph.graph[node_i].activation;
let act_j = morphed_graph.graph[node_j].activation;
morphed_graph.graph[node_i].activation = (act_i + act_j) / 2.0;
```

### New test binaries:
- `test_gradient_check.rs` - Finite difference verification
- `test_gradient_descent.rs` - Descent direction test

## References

- Backend-099: Graph morphing implementation
- Backend-096, 097, 098: Structural loss
- `/data/git/DagNeuralNetwork/trainer.py`: Early prototype activation storage
- README.md FAQ: Hard vs soft merging (still correct!)
