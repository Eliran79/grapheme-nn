---
id: backend-224
title: TextBrain autoencoder (text to TextGraph)
status: done
priority: medium
tags:
- backend
- autoencoder
- stage1
- text
dependencies:
- backend-221
assignee: developer
created: 2025-12-12T17:29:31.890751862Z
estimate: 2h
complexity: 3
area: backend
---

# TextBrain autoencoder (text to TextGraph)

> **SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**

## Context
TextBrain implements the GraphAutoencoder trait for Stage 1 training, enabling
perfect encode/decode of text to character-level graph representations.

## Objectives
- Implement GraphAutoencoder for TextBrain
- Enable perfect text <-> graph roundtrip with zero information loss
- Support Unicode text without vocabulary

## Tasks
- [x] Implement `encode()` to convert text to LatentGraph
- [x] Implement `decode()` to convert LatentGraph back to text
- [x] Implement `reconstruction_loss()` for character-level comparison
- [x] Add unit tests for autoencoder functionality

## Acceptance Criteria
**Encoding:**
- Text input converted to LatentGraph with domain="text"
- Uses DagNN.from_text() for character-level graph creation

**Decoding:**
- LatentGraph converted back to text
- Domain validation ensures correct brain is used

**Reconstruction:**
- Perfect roundtrip: encode(decode(text)) == text
- Loss = 0.0 for perfect reconstruction

## Technical Notes
- Location: `grapheme-brain-common/src/text_brain.rs`
- Implements trait from `grapheme-brain-common/src/autoencoder.rs`
- Character-level: each character becomes a node (no tokenization)
- Works with any Unicode text without configuration
- Uses default reconstruction_loss with character accuracy + length penalty

## Testing
- [x] Write unit tests for new functionality (8 tests)
- [x] Ensure all tests pass
- Tests: `test_text_brain_autoencoder`, `test_text_brain_reconstruction_loss`

## Version Control
- [x] Build, test verified working

## Updates
- 2025-12-12: Task created
- 2025-12-13: Task verified complete - TextBrain already implements GraphAutoencoder

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- TextBrain already implements GraphAutoencoder in `grapheme-brain-common/src/text_brain.rs`
- Implementation at lines 193-224
- 8 tests verify functionality

### Causality Impact
- encode(): text -> DagNN.from_text() -> LatentGraph(domain="text")
- decode(): LatentGraph -> graph.to_text() -> String
- Perfect reconstruction for all Unicode text

### Dependencies & Integration
- Uses grapheme_core::DagNN for graph representation
- Imports GraphAutoencoder, LatentGraph, AutoencoderError from crate
- Integrates with DomainBrain trait (TextBrain implements both)

### Verification & Testing
- Run: `cargo test -p grapheme-brain-common text_brain` - 8 tests pass
- Key tests: test_text_brain_autoencoder, test_text_brain_reconstruction_loss

### Context for Next Task
- TextBrain is the foundation for other brains
- Character-level means no vocabulary needed
- Loss = 0.0 for perfect match
- Unicode fully supported
