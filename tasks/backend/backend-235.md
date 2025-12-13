---
id: backend-235
title: Fix O(n) contains checks - Use HashSet for node lookups
status: todo
priority: critical
tags:
- backend
- performance
- dag-optimization
dependencies: []
assignee: developer
created: 2025-12-13T00:00:00Z
estimate: 1h
complexity: 2
area: backend
---

# Fix O(n) contains checks - Use HashSet for node lookups

## Context
Multiple hot paths use `Vec::contains()` which is O(n). For large graphs, this makes forward pass O(V·n) instead of O(V).

## Problem Locations
- `grapheme-core/src/lib.rs:485` - `add_output_node()`
- `grapheme-core/src/lib.rs:499` - `neuromorphic_forward()`
- `grapheme-core/src/lib.rs:1176` - `gc_disconnected()`

## Current Code
```rust
if self.input_nodes.contains(&node_id) { ... }
if self.output_nodes.contains(&node_id) { ... }
```

## Fix
Change `input_nodes` and `output_nodes` from `Vec<NodeId>` to `HashSet<NodeId>`:
```rust
pub struct DagNN {
    input_nodes: HashSet<NodeId>,  // Was Vec<NodeId>
    output_nodes: HashSet<NodeId>, // Was Vec<NodeId>
    input_nodes_order: Vec<NodeId>, // Keep order if needed
}
```

## DAG Impact
- Forward pass: O(V) instead of O(V·n)
- GC: O(V) instead of O(V·n)
- Critical for large dynamic graphs

## Acceptance Criteria
- [ ] Change `input_nodes` to `HashSet<NodeId>`
- [ ] Change `output_nodes` to `HashSet<NodeId>`
- [ ] Add `input_nodes_order: Vec<NodeId>` if ordering matters
- [ ] Update all `.push()` to `.insert()`
- [ ] Update all `.contains()` calls
- [ ] Verify tests pass
- [ ] Benchmark improvement
