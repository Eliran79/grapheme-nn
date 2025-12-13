---
id: backend-100
title: Implement finite difference gradient checking for structural loss
status: done
priority: medium
tags:
- backend
dependencies:
- backend-099
assignee: developer
created: 2025-12-07T17:47:05.438976765Z
estimate: ~
complexity: 3
area: backend
---

# Implement finite difference gradient checking for structural loss

> **⚠️ SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**
>
> When you mark this task as `done`, you MUST:
> 1. Fill the "Session Handoff" section at the bottom with complete implementation details
> 2. Document what was changed, what runtime behavior to expect, and what dependencies were affected
> 3. Create a clear handoff for the developer/next AI agent working on dependent tasks
>
> **If this task has dependents,** the next task will be handled in a NEW session and depends on your handoff for context.

## Context

Backend-099 implemented graph morphing with learnable threshold. This task validates that gradients flow correctly using finite difference checking.

**Finding:** Discovered critical gradient routing issue - backward pass doesn't account for graph morphing.

## Objectives

- [x] Implement finite difference gradient checker
- [x] Test embedding gradients
- [x] Test threshold gradients
- [x] Identify gradient flow issues
- [x] Apply partial fixes
- [x] Document findings and solution path

## Tasks

### Phase 1: Gradient Checking Implementation
- [x] Create finite difference test (`test_gradient_check.rs`)
- [x] Test threshold gradient with numerical approximation
- [x] Test embedding gradients with numerical approximation
- [x] Identify mismatches

### Phase 2: Problem Diagnosis
- [x] Create gradient descent direction test (`test_gradient_descent.rs`)
- [x] Verify loss should decrease with correct gradients
- [x] Discovered: Loss increases → gradients are wrong!
- [x] Root cause: Backward doesn't route through morphing

### Phase 3: Partial Fixes
- [x] Increase activation weight in structural loss (0.7 vs 0.2)
- [x] Add activation merging during node merge
- [x] Test fixes (still doesn't converge - core issue remains)

### Phase 4: Documentation
- [x] Document gradient flow problem in detail
- [x] Propose solutions (merge history tracking)
- [x] Create `GRADIENT_FLOW_ANALYSIS.md`

## Acceptance Criteria

✅ **Gradient Checker Implemented:**
- Finite difference test compares analytical vs numerical gradients
- Tests both threshold and embedding gradients
- Identifies mismatches correctly

✅ **Problem Identified:**
- Root cause found: Backward pass doesn't route through morphing
- Loss increases instead of decreases (experimental proof)
- Solution path documented (merge history tracking)

⚠️ **Full Convergence (Deferred to Backend-104):**
- Current implementation doesn't converge
- Requires merge history implementation (new task)
- Partial fixes applied as interim improvements

## Technical Notes

### The Core Problem

**Graph morphing breaks gradient flow:**

```
Forward: [a, b, c] → merge b+c → [a, bc]
         ↓ embeddings            ↓ activations

Backward: gradients for [a, bc] ← from loss
          How to map to [a, b, c]? ❌ No tracking!
```

### Why Hard Merging is Still Correct

Despite gradient issues, hard merging remains theoretically sound (graph theory):
- Discrete topology (valid quotient graphs)
- Learnable threshold (policy gradient style)
- Problem is implementation, not theory

### Solution: Merge History Tracking

```rust
pub struct MergeHistory {
    node_mapping: HashMap<NodeId, Vec<(NodeId, f32)>>,
}

// Forward: Track which input nodes → output nodes
// Backward: Route gradients using mapping
```

**Complexity:** O(n) - still polynomial! ✓

### DAG Advantages Maintained

All operations stay polynomial:
- Merge history storage: O(n)
- Gradient routing: O(n)
- No NP-hard graph enumeration

## Testing

**Test binaries created:**

1. `test_gradient_check.rs` - Finite difference validation
   - Threshold gradient: FAIL (heuristic vs numerical)
   - Embedding gradient: FAIL (wrong routing)

2. `test_gradient_descent.rs` - Descent direction test
   - Single step: Loss increases +0.000219
   - 20 epochs: Loss increases +0.001231
   - Conclusion: Gradients point wrong direction

**Partial fixes tested:**
- Activation weight 0.7: No convergence
- Activation merging: No convergence
- Core routing issue must be fixed

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [ ] **BEFORE committing**: Build, test, AND run the code to verify it works
  - Run `cargo build --release` (or `cargo build` for debug)
  - Run `cargo test` to ensure tests pass
  - **Actually run/execute the code** to verify runtime behavior
  - Fix all errors, warnings, and runtime issues
- [ ] Commit changes incrementally with clear messages
- [ ] Use descriptive commit messages that explain the "why"
- [ ] Consider creating a feature branch for complex changes
- [ ] Review changes before committing

**Testing requirements by change type:**
- Code changes: Build + test + **run the actual program/command** to verify behavior
- Bug fixes: Verify the bug is actually fixed by running the code, not just compiling
- New features: Test the feature works as intended by executing it
- Minor changes: At minimum build, check warnings, and run basic functionality

## Updates
- 2025-12-07: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed

**New test binaries:**
1. `grapheme-train/src/bin/test_gradient_check.rs` - Finite difference checker
2. `grapheme-train/src/bin/test_gradient_descent.rs` - Descent direction test

**Code fixes (partial):**
1. `grapheme-train/src/lib.rs:2080` - Increased activation weight to 0.7
2. `grapheme-core/src/lib.rs:4649-4652` - Average activations during merge

**Documentation:**
1. `docs/GRADIENT_FLOW_ANALYSIS.md` - Complete analysis and solution path

**Runtime behavior:**
- Loss now depends more on activations (70% weight vs 10% type)
- Merged nodes have averaged activations
- **But training still doesn't converge** (gradients wrong)

### Causality Impact

**Broken causal chain identified:**
```
Input text → Embeddings → Morphed graph → Loss
                ↓              ↑
                X  (gradient flow broken!)
```

**The problem:**
- Forward morphs graph: [a, b, c] → [a, bc]
- Backward gets gradients for [a, bc]
- Cannot map back to [a, b, c] without tracking!

**No async flows** - all synchronous operations.

### Dependencies & Integration

**No new external dependencies.**

**Modified files:**
- `grapheme-train/src/lib.rs` - Activation weight changed
- `grapheme-core/src/lib.rs` - Activation merging added

**New files:**
- Two test binaries (gradient checking)
- One analysis document

**Backward compatibility:**
- API unchanged
- All existing tests pass (595 tests)
- Training runs but doesn't converge

### Verification & Testing

**How to verify the problem:**
```bash
# Should show loss INCREASING (proves gradients wrong)
cargo run --release --bin test_gradient_descent

# Should show analytical ≠ numerical
cargo run --release --bin test_gradient_check
```

**What to test when building solution:**
```bash
# After implementing merge history (Backend-104):
1. test_gradient_descent → Loss should DECREASE
2. test_gradient_check → Gradients should match
3. train_with_threshold_tracking → Threshold should adapt meaningfully
```

**Known limitations:**
- Gradients don't flow correctly (documented)
- Loss increases instead of decreases
- Cannot train until Backend-104 completes

### Context for Next Task (Backend-104)

**What to implement:**

Merge history tracking to route gradients correctly:

```rust
pub struct MergeHistory {
    // Maps output_node → Vec<(input_node, contribution_weight)>
    node_mapping: HashMap<NodeId, Vec<(NodeId, f32)>>,
}

impl GraphTransformNet {
    pub fn forward(&self, input: &Graph) -> (Graph, MergeHistory) {
        let mut history = MergeHistory::new();

        // During merging:
        when merge node_j into node_i:
            history.record_merge(node_i, vec![
                (original_node_i, 0.5),
                (original_node_j, 0.5),
            ]);

        (morphed_graph, history)
    }

    pub fn backward(&mut self, history: &MergeHistory, ...) {
        // Route gradients using history
        for (output_node, contributors) in history.mappings {
            let grad = output_grads[output_node];
            for (input_node, weight) in contributors {
                input_grads[input_node] += grad * weight;
            }
        }

        // Then backprop to embeddings as usual
    }
}
```

**Why this works:**
- O(n) complexity - polynomial! ✓
- Tracks exact merge decisions
- Enables correct gradient routing
- Preserves DAG advantages

**Important decisions:**

1. **Hard merging is CORRECT** - Don't change to soft merging!
   - Graph theory requires discrete topology
   - Problem is gradient routing, not the merge operation
   - See README.md FAQ for full justification

2. **Activation weight matters** - Keep at 0.7
   - Emphasizes learnable component
   - Type/degree are fixed, activation is learned

3. **Early prototypes had it right** - They stored intermediate values
   - We need similar tracking for merges
   - See `/data/git/DagNeuralNetwork/trainer.py` for reference

**Gotchas:**

- Don't try soft merging - violates graph theory
- Don't decrease activation weight - needs to be high for learning
- Forward pass signature will change: `fn forward() -> (Graph, History)`
- Backward pass needs history parameter
- All calling code (train.rs) must be updated

**Success criteria for Backend-104:**
- `test_gradient_descent` shows DECREASING loss
- `test_gradient_check` shows matching gradients (< 1% error)
- Sustained convergence over 100+ epochs