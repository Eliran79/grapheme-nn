---
id: backend-221
title: Implement GraphAutoencoder trait
status: done
priority: high
tags:
- backend
- autoencoder
- stage1
- brains
dependencies: []
assignee: developer
created: 2025-12-12T17:29:24.582986039Z
estimate: 3h
complexity: 5
area: backend
---

# Implement GraphAutoencoder trait

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
Part of the **Two-Stage Training Paradigm** (see VISION_ARCHITECTURE.md):
- **Stage 1**: Train brains to be perfect encoders/decoders (autoencoders)
- **Stage 2**: Train graph transformations on pre-encoded graphs (no text in loop)

This task creates the `GraphAutoencoder` trait that all domain brains will implement for Stage 1 training.

## Objectives
- Define `GraphAutoencoder` trait in `grapheme-brain-common`
- Enable perfect reconstruction: `text → graph → text` with zero information loss
- Support learnable encoding/decoding parameters
- Integrate with existing `DomainBrain` trait

## Tasks
- [x] Define `GraphAutoencoder` trait with `encode()`, `decode()`, `reconstruction_loss()`
- [x] Add `LatentGraph` type for intermediate representation
- [x] Add serialization support for pre-encoded graphs
- [x] Write default implementations where possible
- [x] Add documentation with usage examples

## Acceptance Criteria
✅ **Trait Definition:**
- `encode(&self, input: &str) -> LatentGraph` converts domain input to latent graph
- `decode(&self, graph: &LatentGraph) -> String` converts graph back to text
- `reconstruction_loss(&self, original: &str, reconstructed: &str) -> f32`

✅ **Integration:**
- Works alongside existing `DomainBrain` trait
- No breaking changes to existing code
- Builds with zero errors

## Technical Notes
```rust
/// Trait for brain autoencoders (Stage 1 training)
pub trait GraphAutoencoder: DomainBrain {
    /// Encode domain input to latent graph representation
    fn encode(&self, input: &str) -> Result<LatentGraph, AutoencoderError>;

    /// Decode latent graph back to domain output
    fn decode(&self, graph: &LatentGraph) -> Result<String, AutoencoderError>;

    /// Compute reconstruction loss (0.0 = perfect reconstruction)
    fn reconstruction_loss(&self, original: &str, reconstructed: &str) -> f32;

    /// Full roundtrip: encode → decode, returns (output, loss)
    fn roundtrip(&self, input: &str) -> Result<(String, f32), AutoencoderError> {
        let graph = self.encode(input)?;
        let output = self.decode(&graph)?;
        let loss = self.reconstruction_loss(input, &output);
        Ok((output, loss))
    }
}

/// Latent graph representation for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentGraph {
    pub domain: String,           // e.g., "code", "math", "text"
    pub graph: DagNN,             // The actual graph structure
    pub metadata: HashMap<String, String>,
}
```

- Location: `grapheme-brain-common/src/autoencoder.rs`
- Re-export from `grapheme-brain-common/src/lib.rs`

## Testing
- [ ] Write unit tests for new functionality
- [ ] Write integration tests if applicable
- [ ] Ensure all tests pass before marking task complete
- [ ] Consider edge cases and error conditions

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
- 2025-12-12: Task created
- 2025-12-12: Task completed - GraphAutoencoder trait, LatentGraph, EncodedPair implemented

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **New File**: `grapheme-brain-common/src/autoencoder.rs` (~350 lines)
  - `GraphAutoencoder` trait: `encode()`, `decode()`, `reconstruction_loss()`, `roundtrip()`, `encode_batch()`, `decode_batch()`, `validate_latent()`
  - `LatentGraph` struct: Serializable latent graph with domain, graph, metadata
  - `EncodedPair` struct: Pre-encoded (input, output) pair for Stage 2 training
  - `AutoencoderError` enum: EncodingError, DecodingError, ValidationError, DomainMismatch
- **Modified**: `grapheme-brain-common/src/lib.rs` - Added autoencoder module and exports
- **8 new unit tests** for LatentGraph, EncodedPair, and reconstruction_loss

### Causality Impact
- All methods are pure/functional - no side effects
- `reconstruction_loss()` has a default implementation using normalized character accuracy
- Brains can override with domain-specific loss functions
- Thread-safe (no shared mutable state)

### Dependencies & Integration
- Uses existing dependencies: `grapheme_core::DagNN`, `serde`, `thiserror`
- Extends `DomainBrain` trait - implementors must also implement DomainBrain
- No breaking changes to existing code
- Exported from `grapheme_brain_common`: `GraphAutoencoder`, `LatentGraph`, `EncodedPair`, `AutoencoderError`

### Verification & Testing
- Build: `cargo build --release -p grapheme-brain-common` - zero errors
- Tests: `cargo test --release -p grapheme-brain-common` - 55 tests pass (8 new autoencoder tests)
- To verify implementation: Import and implement the trait on a brain, call `roundtrip()` on test inputs

### Context for Next Task
- **For backend-222/223/224/225 (brain implementations)**: Each brain must implement `encode()` and `decode()`. The default `reconstruction_loss()` can be used initially, then customized for domain-specific equivalence (e.g., ignore whitespace in code).
- **Important**: `encode()` should use the brain's `parse()` method internally, `decode()` should convert the graph back to text
- **Pattern**: `impl GraphAutoencoder for MyBrain { ... }` after implementing `DomainBrain`
- **Gotcha**: `validate_latent()` checks domain match - don't decode a "code" graph with MathBrain