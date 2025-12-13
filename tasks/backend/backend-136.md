---
id: backend-136
title: Implement hierarchical blob detection for VisionBrain.to_graph()
status: done
priority: high
tags:
- backend
- vision
- feature-extraction
dependencies:
- backend-135
assignee: developer
created: 2025-12-09T19:29:17.308793223Z
estimate: 6h
complexity: 7
area: backend
---

# Implement hierarchical blob detection for VisionBrain.to_graph()

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
Implement multi-scale hierarchical blob detection for VisionBrain.
Detect blobs at multiple intensity thresholds (coarse → fine) and build parent-child relationships.

## Objectives
- [x] Implement HierarchicalBlob struct with parent/child links
- [x] Implement extract_hierarchical_blobs() for multi-scale detection
- [x] Update image_to_graph() to use hierarchical detection
- [x] Maintain backward compatibility with single-level mode

## Tasks
- [x] Add HierarchicalBlob struct with level, scale, parent, children
- [x] Add BlobHierarchy result struct
- [x] Implement extract_blobs_at_threshold() helper
- [x] Implement extract_hierarchical_blobs() with threshold scaling
- [x] Implement parent-child linking based on spatial containment
- [x] Update image_to_graph() to branch on max_hierarchy_levels
- [x] Add 7 unit tests for hierarchical detection
- [x] Verify all existing tests still pass

## Acceptance Criteria
✅ **Criteria 1:**
- Multi-scale detection at different thresholds (coarse = low, fine = high)

✅ **Criteria 2:**
- Parent-child relationships based on spatial containment

✅ **Criteria 3:**
- image_to_graph() creates hierarchy edges in the graph

✅ **Criteria 4:**
- Backward compatible: max_hierarchy_levels=1 uses single-level detection

## Technical Notes
- Coarse levels use lower thresholds (more permissive, larger blobs)
- Fine levels use higher thresholds (more selective, smaller blobs)
- Parent contains child if child's center is within parent's bounding box
- VisionEdge::Hierarchy used for parent-child links
- VisionEdge::Contains used for root to top-level blobs

## Testing
- [x] test_hierarchical_blob_new
- [x] test_hierarchical_blob_contains
- [x] test_extract_hierarchical_blobs_single_blob
- [x] test_extract_hierarchical_blobs_multi_scale
- [x] test_blob_hierarchy_parent_child_links
- [x] test_image_to_graph_hierarchical
- [x] test_image_to_graph_single_level_fallback

## Version Control
- [x] Build passes
- [x] All 27 grapheme-vision tests pass
- [x] All workspace tests pass

## Updates
- 2025-12-09: Task created
- 2025-12-09: Implementation complete

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-vision/src/lib.rs** (~200 new lines): Hierarchical blob detection
  - `HierarchicalBlob`: Struct with blob, level, children, parent, scale
  - `BlobHierarchy`: Result struct with all blobs, num_levels, roots
  - `extract_blobs_at_threshold()`: Helper for single-threshold extraction
  - `extract_hierarchical_blobs()`: Multi-scale detection with parent-child linking
  - Updated `image_to_graph()`: Uses hierarchical mode when max_hierarchy_levels > 1

### Causality Impact
- Hierarchical detection runs at multiple thresholds (coarse → fine)
- Parent-child relationships determined by spatial containment
- Level 0 = finest (highest threshold), Level N-1 = coarsest (lowest threshold)
- Root blobs (no parent) connect to ImageRoot node
- Children connect to parents via VisionEdge::Hierarchy

### Dependencies & Integration
- Uses: FeatureConfig.max_hierarchy_levels and build_hierarchy flags
- Exports: `HierarchicalBlob`, `BlobHierarchy`, `extract_hierarchical_blobs`
- Unblocks: backend-137 (spatial relationship graph needs hierarchical structure)

### Verification & Testing
```bash
# Run hierarchical tests
cargo test -p grapheme-vision test_hierarchical

# Test all vision tests (27 total)
cargo test -p grapheme-vision
```

### Context for Next Task
- **backend-137 (Spatial Relationships)**: Can now use BlobHierarchy for richer spatial analysis
- The hierarchy is in the VisionGraph: root → top-level blobs → child blobs
- Same-level blobs have VisionEdge::Adjacent if their bounding boxes touch
- Threshold scaling: `threshold = base + step * (num_levels - level - 1)`
- For MNIST: default is 2 levels (blob_threshold=0.2, max_hierarchy_levels=2)