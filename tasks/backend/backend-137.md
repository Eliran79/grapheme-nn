---
id: backend-137
title: Implement spatial relationship graph from detected features
status: done
priority: high
tags:
- backend
- vision
- graph-construction
dependencies:
- backend-136
assignee: developer
created: 2025-12-09T19:29:21.259324232Z
estimate: 4h
complexity: 6
area: backend
---

# Implement spatial relationship graph from detected features

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
Build spatial relationship edges between detected blobs to capture directional and proximity information in the vision graph. This enables GRAPHEME to understand spatial layout of visual features.

## Objectives
- [x] Add directional edge types (Above, Below, LeftOf, RightOf)
- [x] Add proximity edges with distance weights
- [x] Implement spatial relationship computation from blob centers
- [x] Integrate spatial graph building into image_to_graph()
- [x] Test on real MNIST data to verify spatial relationships

## Tasks
- [x] Add VisionEdge variants: Above, Below, LeftOf, RightOf, Proximity(f32)
- [x] Implement SpatialRelationship struct with direction detection
- [x] Implement compute_spatial_relationships() function
- [x] Implement build_spatial_graph() function
- [x] Add FeatureConfig fields: adjacency_threshold, build_spatial_graph
- [x] Update image_to_graph() to call build_spatial_graph()
- [x] Add unit tests for spatial relationships
- [x] Add integration tests on real MNIST data
- [x] Verify determinism: same image = same graph

## Acceptance Criteria
✅ **Criteria 1:**
- Directional edges based on blob center positions (Above/Below for Y, LeftOf/RightOf for X)

✅ **Criteria 2:**
- Proximity edges with normalized distance for blobs within threshold

✅ **Criteria 3:**
- Integration tests pass on real MNIST data showing spatial relationships

✅ **Criteria 4:**
- Deterministic: same image always produces identical graph embedding

## Technical Notes
- Direction determined by angle from blob A to blob B:
  - Above: dy < 0 and |dy| > |dx|
  - Below: dy > 0 and |dy| > |dx|
  - LeftOf: dx < 0 and |dx| >= |dy|
  - RightOf: dx > 0 and |dx| >= |dy|
- Proximity threshold is normalized by image diagonal
- Same-level blobs get directional edges; cross-level use hierarchy edges
- Flood-fill blob extraction is deterministic (same pixels = same blobs)

## Testing
- [x] test_spatial_relationship_directions - 8 direction cases
- [x] test_spatial_relationship_primary_direction
- [x] test_compute_spatial_relationships
- [x] test_build_spatial_graph
- [x] test_image_to_graph_with_spatial
- [x] test_vision_edge_directional
- [x] 8 MNIST integration tests including determinism

## Version Control
- [x] Build passes
- [x] All 42 tests pass (33 unit + 8 integration + 1 doc)
- [x] Committed with spatial relationship implementation

## Updates
- 2025-12-09: Task created
- 2025-12-09: Implementation complete, MNIST integration tests added

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-vision/src/lib.rs** (~150 new lines):
  - `VisionEdge` extended: Added Above, Below, LeftOf, RightOf, Proximity(f32) variants
  - `SpatialRelationship` struct: Captures dx, dy, distance, angle between blobs
  - `compute_spatial_relationships()`: Computes all pairwise relationships within threshold
  - `build_spatial_graph()`: Adds directional and proximity edges to VisionGraph
  - `FeatureConfig`: Added adjacency_threshold (0.15) and build_spatial_graph (true)
  - `image_to_graph()`: Now calls build_spatial_graph() when enabled

- **grapheme-vision/tests/mnist_integration.rs** (new file, ~335 lines):
  - 8 integration tests on real MNIST data from data/mnist/
  - Tests blob extraction, hierarchical detection, graph construction
  - Tests spatial relationships and **determinism** (same image = same graph)

- **grapheme-vision/Cargo.toml**: Added `mnist = "0.6"` dev-dependency

### Causality Impact
- Spatial edges are added AFTER hierarchical edges in image_to_graph()
- Direction computed from normalized blob centers (center / image_size)
- Proximity edges only created for blobs within adjacency_threshold distance
- Deterministic: same pixel values → same blobs → same edges → same graph

### Dependencies & Integration
- Uses: HierarchicalBlob and BlobHierarchy from backend-136
- Exports: SpatialRelationship, compute_spatial_relationships, build_spatial_graph
- FeatureConfig.build_spatial_graph controls whether spatial edges are added
- Unblocks: backend-139 (MNIST pipeline needs complete vision graph)

### Verification & Testing
```bash
# Run all vision tests (42 total)
cargo test -p grapheme-vision

# Run only spatial relationship tests
cargo test -p grapheme-vision spatial

# Run MNIST integration tests
cargo test -p grapheme-vision --test mnist_integration

# Run determinism test specifically
cargo test -p grapheme-vision test_mnist_determinism
```

### Context for Next Task
- **backend-139 (MNIST Pipeline)**: VisionBrain now produces complete graphs with:
  - Hierarchical blob nodes (root → level 0 → level 1 → ...)
  - Contains edges (root to top-level blobs)
  - Hierarchy edges (parent to child blobs)
  - Directional edges (Above/Below/LeftOf/RightOf between same-level blobs)
  - Proximity edges (between nearby blobs within threshold)
- **Determinism verified**: test_mnist_determinism passes - same image always gives same graph
- Example digit "4" produces: 4 nodes, 7 edges including LeftOf, RightOf, Adjacent
- MNIST data expected at: ../data/mnist/ (relative to grapheme-vision crate)
