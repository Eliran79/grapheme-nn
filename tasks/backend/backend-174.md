---
id: backend-174
title: Implement web crawler with rate limiting and robots.txt support
status: done
priority: medium
tags:
- backend
- crawler
- web
- rate-limiting
- robots
dependencies:
- backend-170
assignee: developer
created: 2025-12-11T07:42:01.609323137Z
estimate: 5h
complexity: 7
area: backend
---

# Implement web crawler with rate limiting and robots.txt support

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
Brief description of what needs to be done and why.

## Objectives
- Clear, actionable objectives
- Measurable outcomes
- Success criteria

## Tasks
- [ ] Break down the work into specific tasks
- [ ] Each task should be clear and actionable
- [ ] Mark tasks as completed when done

## Acceptance Criteria
✅ **Criteria 1:**
- Specific, testable criteria

✅ **Criteria 2:**
- Additional criteria as needed

## Technical Notes
- Implementation details
- Architecture considerations
- Dependencies and constraints

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
- 2025-12-11: Task created

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created `grapheme-train/src/web_crawler.rs` (~800 lines)
- Key types: CrawlerConfig, RobotsTxt, RateLimiter, UrlQueue, QueuedUrl, WebCrawler, CrawledPage, CrawlError
- Added module export in lib.rs

### Causality Impact
- WebCrawler.add_seeds() initializes URL queue
- crawl_next() processes one URL:
  1. Check depth/pattern/robots.txt limits
  2. Wait for rate limit (per-domain delay)
  3. Fetch page via WebFetcher
  4. Extract links and add to queue
- crawl_all() processes entire queue up to max_pages

### Dependencies & Integration
- Uses web_fetcher module for HTTP requests
- RobotsTxt.parse() extracts directives from robots.txt
- RateLimiter tracks per-domain request timing
- UrlQueue deduplicates and limits per-domain

### Verification & Testing
- Run: `cargo test -p grapheme-train web_crawler::`
- Expected: 23 tests pass
- Key tests: test_robots_txt_parse_simple, test_rate_limiter_delay, test_url_queue_dedup

### Context for Next Task
- CrawlerConfig::gentle() for respectful crawling (2s delay)
- CrawlerConfig::aggressive() for faster crawling (500ms delay)
- robots.txt crawl-delay directive updates rate limiter
- Block patterns exclude common non-content URLs (images, admin, etc.)
- Link extraction is regex-based (simple href= parsing)