---
id: backend-024
title: Complete TODO implementations in memory and training
status: todo
priority: low
tags:
- backend
- enhancement
dependencies: []
assignee: developer
created: 2025-12-06T00:00:00Z
estimate: ~
complexity: 1
area: backend
---

# Complete TODO implementations in memory and training

## Context
Code review identified 2 TODO comments for incomplete implementations. These are minor enhancements that don't affect core functionality but would improve accuracy.

## Objectives
- Complete node_types population in GraphFingerprint
- Implement clique comparison in GraphEditDistance

## Tasks
- [ ] Populate `node_types` array in `GraphFingerprint::from_graph()` (grapheme-memory/src/lib.rs:637)
- [ ] Implement clique comparison for `clique_mismatch` field (grapheme-train/src/lib.rs:841)
- [ ] Update `GraphFingerprint::similarity()` to use node_types in calculation
- [ ] Add tests for enhanced functionality

## Acceptance Criteria
✅ **GraphFingerprint Enhancement:**
- node_types array populated based on node characteristics
- Similarity function uses node_types for more accurate matching

✅ **Clique Mismatch Enhancement:**
- clique_mismatch computed based on clique structure differences
- Weighted contribution to overall GED

## Technical Notes

### node_types Enhancement
```rust
// In GraphFingerprint::from_graph()
// Bucket nodes by type (activation level, degree, position)
for node in graph.graph.node_indices() {
    let node_data = &graph.graph[node];
    let bucket = determine_node_bucket(node_data); // 0-7
    node_types[bucket] += 1;
}
```

### clique_mismatch Enhancement
```rust
// In GraphEditDistance::compute()
let pred_cliques = predicted.cliques.len();
let target_cliques = target.cliques.len();
let clique_diff = (pred_cliques as i32 - target_cliques as i32).abs() as f32;
clique_mismatch = clique_diff * 0.2;
```

## Testing
- [ ] Test GraphFingerprint similarity with node_types
- [ ] Test clique_mismatch calculation
- [ ] Verify existing tests still pass

## Updates
- 2025-12-06: Task created from code review

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [Document code changes]

### Verification & Testing
- [How to verify this works]
