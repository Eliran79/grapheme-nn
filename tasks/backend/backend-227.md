---
id: backend-227
title: Graph-only training data format
status: done
priority: high
tags:
- backend
- stage2
- graph
- data
dependencies:
- backend-226
assignee: developer
created: 2025-12-12T17:29:43.623539405Z
estimate: 2h
complexity: 4
area: backend
---

# Graph-only training data format

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
Stores training pairs as DagNN graphs without text intermediates, enabling pure graph-to-graph training following the GRAPHEME vision.

## Objectives
- Create binary serialization for efficient graph storage
- Remove text from training loop - pure graph pairs
- Provide batched I/O for high-throughput training
- Track metadata for curriculum level management

## Tasks
- [x] Create `GraphPair` structure for input/output graph pairs
- [x] Implement `GraphDataset` for collections of graph pairs
- [x] Add binary serialization (bincode) for efficient storage
- [x] Add JSON serialization for debugging/inspection
- [x] Create batch iterator for training loops
- [x] Implement graph builders (chain, tree, random DAG)
- [x] Add `GraphEncoder` trait for domain-specific encoders
- [x] Create `EncoderRegistry` for encoder management
- [x] Write comprehensive unit tests (16 tests)

## Acceptance Criteria
✅ **Binary Format:**
- Magic bytes for file identification
- Version number for compatibility
- Efficient bincode serialization

✅ **Dataset API:**
- Add/filter/split operations
- Batch iteration support
- Statistics computation

## Technical Notes
- Uses DagNN from grapheme-core as underlying graph representation
- Binary format: GRPH magic + version + bincode data
- GraphPair contains: id, input DagNN, output DagNN, level, domain, metadata
- GraphDataset supports train/val/test splitting
- Helper functions: create_chain_graph, create_tree_graph, create_random_dag

## Testing
- [x] Write unit tests for new functionality (16 tests)
- [x] Write integration tests if applicable
- [x] Ensure all tests pass before marking task complete
- [x] Consider edge cases and error conditions

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
- [x] Commit changes incrementally with clear messages
- [x] Use descriptive commit messages that explain the "why"

## Updates
- 2025-12-12: Task created
- 2025-12-13: Task completed

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created new file: `grapheme-train/src/graph_data.rs`
- Updated `grapheme-train/src/lib.rs` with module and re-exports
- Updated `grapheme-train/Cargo.toml` with bincode and tempfile deps
- Key structures:
  - `GraphPair`: Input/output DagNN pair with metadata
  - `GraphDataset`: Collection with split/filter/batch operations
  - `GraphBatchIterator`: Efficient batch iteration
  - `GraphEncoder`: Trait for domain-specific encoders
  - `EncoderRegistry`: Encoder management

### Causality Impact
- Training flow: load GraphDataset → batches() → train on GraphPair
- Serialization: save_binary/load_binary for persistent storage
- No text intermediates in training loop - pure graph-to-graph

### Dependencies & Integration
- Added `bincode = "1.3"` for binary serialization
- Added `tempfile = "3.10"` (dev-dependency) for tests
- Re-exports from lib.rs: GraphPair, GraphDataset, GraphBatchIterator, etc.
- Integrates with existing DagNN from grapheme-core

### Verification & Testing
- Run: `cargo test -p grapheme-train graph_data` - 16 tests pass
- Run: `cargo test -p grapheme-train --lib` - 48 total tests pass
- Clippy: `cargo clippy -p grapheme-train -- -D warnings` - 0 warnings

### Context for Next Task
- GraphPair.input/output are DagNN graphs (serializable with serde)
- Binary format uses magic bytes [G,R,P,H] and version 1
- GraphEncoder trait for backend-228 (pre-encoding HumanEval)
- EncoderRegistry.register() accepts Box<dyn GraphEncoder + Send + Sync>
- create_random_dag uses max_attempts to prevent infinite loops
