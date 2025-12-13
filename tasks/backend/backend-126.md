---
id: backend-126
title: Implement BaseDomainBrain trait with default implementations
status: done
priority: high
tags:
- backend
- refactoring
- traits
dependencies:
- backend-123
- backend-124
assignee: developer
created: 2025-12-09T11:44:11.510863676Z
estimate: 3h
complexity: 6
area: backend
---

# Implement BaseDomainBrain trait with default implementations

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
Many cognitive brain implementations (math, code, law, music, chem) share identical patterns for DomainBrain trait methods. This task creates a BaseDomainBrain trait with default implementations to reduce code duplication.

## Objectives
- Create BaseDomainBrain trait with sensible defaults
- Provide DomainConfig struct for common configuration
- Reduce boilerplate in brain implementations
- Add macro for easy DomainBrain implementation

## Tasks
- [x] Analyze existing brain implementations for common patterns
- [x] Design DomainConfig struct for common configuration
- [x] Implement BaseDomainBrain trait with default_* methods
- [x] Implement impl_domain_brain_defaults! macro
- [x] Add comprehensive unit tests (10 tests)

## Acceptance Criteria
✅ **BaseDomainBrain trait:**
- Provides default implementations for all DomainBrain methods
- Works with KeywordCapabilityDetector and TextNormalizer

✅ **DomainConfig struct:**
- Stores domain_id, domain_name, version
- Supports keyword configuration
- Supports annotation prefix for to_core filtering

## Technical Notes
- BaseDomainBrain is an extension trait, not a replacement for DomainBrain
- Brain implementations can override individual default_* methods
- impl_domain_brain_defaults! macro generates full DomainBrain impl

## Testing
- [x] Write unit tests for DomainConfig (5 tests)
- [x] Write unit tests for BaseDomainBrain defaults (5 tests)
- [x] All 47 tests pass in grapheme-brain-common

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
- 2025-12-09: Task created
- 2025-12-09: Completed - BaseDomainBrain trait with 10 tests

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **New file**: `grapheme-brain-common/src/traits.rs`
- **DomainConfig struct**: Configuration holder for domain brains with domain_id, domain_name, version, capability_detector, normalizer, annotation_prefix
- **BaseDomainBrain trait**: Extension trait providing default_* methods for all DomainBrain methods
- **impl_domain_brain_defaults! macro**: Generates full DomainBrain implementation from BaseDomainBrain

### Causality Impact
- No runtime effects yet - brains need to adopt these traits
- Brain implementations can gradually migrate to use BaseDomainBrain
- Existing DomainBrain implementations remain unchanged

### Dependencies & Integration
- Uses KeywordCapabilityDetector and TextNormalizer from utils.rs
- Uses grapheme_core types (DagNN, DomainResult, etc.)
- Brain crates (backend-129-133) will migrate to use this

### Verification & Testing
- Run `cargo test --package grapheme-brain-common` - 47 tests pass
- 10 new tests for traits.rs specifically
- Doc tests ignored (require ignore attribute due to trait visibility)

### Context for Next Task
- **Design decision**: BaseDomainBrain is opt-in via default_* methods, not automatic inheritance
- **Macro**: impl_domain_brain_defaults! provides full automation for brains using all defaults
- **Customization**: Brains can implement DomainBrain manually and call specific default_* methods
- **Pattern**: parse() always uses DagNN::from_text, from_core() applies normalizer, to_core() filters annotation lines