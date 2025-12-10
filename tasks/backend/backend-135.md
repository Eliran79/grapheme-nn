---
id: backend-135
title: Create grapheme-vision crate with VisionBrain for image-to-graph embedding
status: done
priority: high
tags:
- backend
- vision
- mnist
- cognitive-brain
dependencies:
- backend-123
- backend-126
assignee: developer
created: 2025-12-09T19:29:11.343924315Z
estimate: 8h
complexity: 8
area: backend
---

# Create grapheme-vision crate with VisionBrain for image-to-graph embedding

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
Create the grapheme-vision crate implementing VisionBrain for GRAPHEME-native image processing.
Per GRAPHEME Vision: Images become graphs through deterministic blob detection (no CNN).
Same image → same input graph (always).

## Objectives
- [x] Create grapheme-vision crate with Cargo.toml
- [x] Implement RawImage struct for any image size (raw pixels, no image library)
- [x] Implement VisionBrain implementing DomainBrain trait
- [x] Implement blob detection via flood-fill connected components
- [x] Implement image_to_graph() for deterministic conversion
- [x] Add benchmarks for performance testing

## Tasks
- [x] Create grapheme-vision/Cargo.toml with dependencies
- [x] Implement RawImage struct (pixels: Vec<f32>, width, height, channels)
- [x] Implement VisionError types
- [x] Implement Blob struct and extraction via flood-fill
- [x] Implement VisionNodeType enum (Blob, Edge, Corner, Region, ImageRoot)
- [x] Implement VisionGraph with petgraph DiGraph
- [x] Implement VisionBrain with to_graph() and to_dagnn()
- [x] Implement DomainBrain trait for VisionBrain
- [x] Add 10 unit tests
- [x] Add benchmarks (vision_bench.rs)
- [x] Add to workspace Cargo.toml

## Acceptance Criteria
✅ **Criteria 1:**
- RawImage supports any dimensions (not just 28x28)

✅ **Criteria 2:**
- Same image always produces same VisionGraph (deterministic)

✅ **Criteria 3:**
- VisionBrain implements DomainBrain trait from grapheme-brain-common

✅ **Criteria 4:**
- All 10 unit tests pass, benchmarks run

## Technical Notes
- No image library needed - just Vec<f32> with dimensions
- Blob detection: flood-fill with intensity threshold (default 0.3)
- Spatial grouping: blobs within distance threshold considered adjacent
- Blob features: centroid (cx, cy), size, mean intensity
- VisionGraph stores hierarchical structure: blobs → regions → root

## Testing
- [x] test_raw_image_grayscale
- [x] test_raw_image_from_mnist
- [x] test_raw_image_get_pixel
- [x] test_extract_blobs_simple
- [x] test_extract_blobs_multiple
- [x] test_blobs_adjacent
- [x] test_image_to_graph
- [x] test_vision_brain_mnist
- [x] test_vision_brain_deterministic
- [x] test_vision_node_activation

## Version Control
- [x] Build passes
- [x] All 10 grapheme-vision tests pass
- [x] All workspace tests pass

## Updates
- 2025-12-09: Task created
- 2025-12-09: Implementation complete
- 2025-12-10: **Refactored**: Removed MNIST-specific methods (`from_mnist()`, `VisionBrain::mnist()`, `mnist_to_graph()`). Now fully generic for any image size/format

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-vision/Cargo.toml**: New crate configuration
  - Dependencies: grapheme-core, grapheme-brain-common, petgraph, ndarray, serde, thiserror

- **grapheme-vision/src/lib.rs** (~900 lines): Complete VisionBrain implementation
  - `RawImage`: Simple pixel container (Vec<f32>, width, height, channels)
  - `VisionError`: Error types for vision processing
  - `Blob`: Connected component with centroid, size, intensity
  - `VisionNodeType`: Blob, Edge, Corner, Region, ImageRoot variants
  - `VisionNode`, `VisionEdge`: Graph node/edge types
  - `VisionGraph`: petgraph DiGraph with vision nodes
  - `FeatureConfig`: Blob detection parameters (threshold, min_size, etc.)
  - `VisionBrain`: Main brain implementing DomainBrain trait
  - `extract_blobs()`: Flood-fill connected component detection
  - `image_to_graph()`: Deterministic image-to-graph conversion
  - `blobs_adjacent()`: Spatial relationship detection

- **grapheme-vision/benches/vision_bench.rs**: Criterion benchmarks
  - bench_blob_extraction, bench_image_to_graph, bench_vision_brain

- **Cargo.toml (workspace)**: Added grapheme-vision to members list

### Causality Impact
- VisionBrain.to_graph() is deterministic: same pixels → same VisionGraph
- Blob detection is single-pass flood-fill (O(n) pixels)
- to_dagnn() converts VisionGraph back to DagNN using DagNN::from_image()
- Blobs become high-activation pixels at their centroid positions

### Dependencies & Integration
- Depends on: grapheme-core (DagNN), grapheme-brain-common (DomainBrain trait)
- Exports: `RawImage`, `VisionBrain`, `VisionGraph`, `VisionNodeType`, `Blob`, `extract_blobs`, `image_to_graph`, `FeatureConfig`
- **Generic API (2025-12-10)**: Use `RawImage::grayscale(w, h, &pixels)` or `RawImage::rgb(w, h, &pixels)`, `VisionBrain::new()`, `brain.to_graph(&image)`
- Unblocks: backend-136 (hierarchical blob detection), backend-137 (spatial relationships)

### Verification & Testing
```bash
# Run vision tests
cargo test -p grapheme-vision

# Run benchmarks
cargo bench -p grapheme-vision -- --test

# Test determinism
# The test_vision_brain_deterministic test verifies same input → same output
```

### Context for Next Task
- **backend-136 (Hierarchical Blob Detection)**: Current blob detection is single-level. Next step is multi-scale detection (coarse → fine)
- **backend-137 (Spatial Relationships)**: `blobs_adjacent()` exists but spatial graph construction needs more work
- Current to_dagnn() creates pixel-grid DagNN from blob positions - may want blob-node DagNN instead
- FeatureConfig is tunable: intensity_threshold (0.3), min_blob_size (3), adjacency_threshold (0.15)
- RawImage supports any dimensions - not limited to MNIST 28x28