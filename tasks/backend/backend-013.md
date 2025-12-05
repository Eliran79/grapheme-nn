---
id: backend-013
title: Optimize connect_relevant context window
status: todo
priority: low
tags:
- backend
- performance
- optimization
dependencies: []
assignee: developer
created: 2025-12-05T21:53:53.172320109Z
estimate: ~
complexity: 2
area: backend
---

# Optimize connect_relevant context window

## Context
The `connect_relevant()` method iterates over all input nodes to find nodes within a context window. When called for every node during graph construction, this results in O(n²) total complexity.

**Current code** (`grapheme-core/src/lib.rs:739-761`):
```rust
fn connect_relevant(&mut self, node: NodeId, context_window: usize) {
    let node_pos = self.graph[node].position.unwrap_or(0);

    for &other in &self.input_nodes {  // O(n) iteration
        if other == node { continue; }
        if let Some(other_pos) = self.graph[other].position {
            let distance = /* compute distance */;
            if distance <= context_window && distance > 1 {
                self.graph.add_edge(other, node, Edge::skip(weight));
            }
        }
    }
}
```

## Objectives
- Reduce complexity when called repeatedly
- Use position-indexed data structure for O(1) neighbor lookup
- Or batch the operation to avoid repeated iterations

## Tasks
- [ ] Analyze call patterns for `connect_relevant()`
- [ ] Option A: Add position-indexed lookup (e.g., `BTreeMap<usize, NodeId>`)
- [ ] Option B: Batch all skip connections in single pass
- [ ] Option C: Limit context_window to small constant (already bounded?)
- [ ] Implement chosen optimization
- [ ] Add benchmark to verify improvement
- [ ] Update tests

## Acceptance Criteria
✅ **Correctness:**
- Same skip connections created as before
- All tests pass

✅ **Performance:**
- Reduce from O(n²) to O(n × window_size) or O(n log n)
- Benchmark shows measurable improvement for large graphs

## Technical Notes

### Option A: Position-Indexed Lookup
```rust
pub struct GraphemeGraph {
    // Add index for position-based queries
    position_index: BTreeMap<usize, NodeId>,
    // ...
}

fn connect_relevant(&mut self, node: NodeId, context_window: usize) {
    let node_pos = self.graph[node].position.unwrap_or(0);
    let start = node_pos.saturating_sub(context_window);
    let end = node_pos + context_window;

    // O(window_size) iteration using range query
    for (&pos, &other) in self.position_index.range(start..=end) {
        if other != node && pos != node_pos {
            let distance = (node_pos as i64 - pos as i64).abs() as usize;
            if distance > 1 {
                let weight = 1.0 / (distance as f32);
                self.graph.add_edge(other, node, Edge::skip(weight));
            }
        }
    }
}
```

### Option B: Batch Processing
```rust
fn connect_all_relevant(&mut self, context_window: usize) {
    // Sort nodes by position once
    let mut nodes_by_pos: Vec<_> = self.input_nodes.iter()
        .filter_map(|&n| self.graph[n].position.map(|p| (p, n)))
        .collect();
    nodes_by_pos.sort_by_key(|&(p, _)| p);

    // Sliding window approach - O(n × window_size)
    for i in 0..nodes_by_pos.len() {
        let (pos_i, node_i) = nodes_by_pos[i];
        for j in (i+1)..nodes_by_pos.len() {
            let (pos_j, node_j) = nodes_by_pos[j];
            let distance = pos_j - pos_i;
            if distance > context_window { break; }
            if distance > 1 {
                let weight = 1.0 / (distance as f32);
                self.graph.add_edge(node_j, node_i, Edge::skip(weight));
            }
        }
    }
}
```

### Key Design Decision
- If `context_window` is always small (e.g., 5-10), O(n²) with early break is acceptable
- If graphs grow large (>10k nodes), position index is worth the overhead

### Files to Modify
- `grapheme-core/src/lib.rs`: Add position index or batch method

## Testing
- [ ] Verify skip connections match previous behavior
- [ ] Benchmark on graphs of size 100, 1000, 10000 nodes
- [ ] Test with various context_window sizes

## Updates
- 2025-12-05: Task created from NP-hard gap analysis

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [Document code changes, new files, modified functions]

### Causality Impact
- Performance improvement for large graph construction
- No functional changes to skip connections

### Dependencies & Integration
- May need BTreeMap import
- Internal optimization, no API changes

### Verification & Testing
- Run `cargo test -p grapheme-core`
- Run scaling benchmark
