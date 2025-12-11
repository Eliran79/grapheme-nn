//! A2A Agent Discovery and Registry Service
//!
//! API-019: Implements agent discovery and registry for A2A protocol.
//!
//! This module provides:
//! - Agent registration and discovery
//! - Skill-based agent lookup
//! - Health monitoring and status tracking
//! - Agent metadata management
//!
//! Registry enables agents to find each other based on capabilities and skills.

use crate::a2a_protocol::{AgentCard, AgentCapabilities, AuthenticationInfo, Skill};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Registry Configuration
// ============================================================================

/// Configuration for the agent registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// How often to check agent health (seconds)
    pub health_check_interval_secs: u64,
    /// Timeout for health checks (milliseconds)
    pub health_check_timeout_ms: u64,
    /// Remove agents after this many failed health checks
    pub max_failed_health_checks: u32,
    /// Maximum agents in registry
    pub max_agents: usize,
    /// Enable automatic health checking
    pub auto_health_check: bool,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            health_check_interval_secs: 30,
            health_check_timeout_ms: 5000,
            max_failed_health_checks: 3,
            max_agents: 1000,
            auto_health_check: true,
        }
    }
}

// ============================================================================
// Agent Registration Entry
// ============================================================================

/// Registration status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    /// Agent is active and healthy
    Active,
    /// Agent is temporarily unavailable
    Unavailable,
    /// Agent is being checked
    Checking,
    /// Agent has been deregistered
    Deregistered,
    /// Unknown status (never checked)
    Unknown,
}

/// Entry in the agent registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEntry {
    /// The agent card (capabilities, skills, etc.)
    pub card: AgentCard,
    /// Current status
    pub status: AgentStatus,
    /// Registration timestamp (ISO 8601)
    pub registered_at: String,
    /// Last health check timestamp
    pub last_health_check: Option<String>,
    /// Last successful health check
    pub last_healthy_at: Option<String>,
    /// Consecutive failed health checks
    pub failed_health_checks: u32,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl AgentEntry {
    /// Create a new agent entry from a card
    pub fn new(card: AgentCard) -> Self {
        Self {
            card,
            status: AgentStatus::Unknown,
            registered_at: chrono_now(),
            last_health_check: None,
            last_healthy_at: None,
            failed_health_checks: 0,
            metadata: HashMap::new(),
            tags: Vec::new(),
        }
    }

    /// Mark as healthy
    pub fn mark_healthy(&mut self) {
        self.status = AgentStatus::Active;
        self.failed_health_checks = 0;
        let now = chrono_now();
        self.last_health_check = Some(now.clone());
        self.last_healthy_at = Some(now);
    }

    /// Mark as unhealthy
    pub fn mark_unhealthy(&mut self) {
        self.failed_health_checks += 1;
        self.status = AgentStatus::Unavailable;
        self.last_health_check = Some(chrono_now());
    }

    /// Check if agent should be removed
    pub fn should_remove(&self, max_failures: u32) -> bool {
        self.failed_health_checks >= max_failures
    }
}

// ============================================================================
// Discovery Query
// ============================================================================

/// Query for discovering agents
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoveryQuery {
    /// Filter by skill ID
    pub skill_id: Option<String>,
    /// Filter by skill tags
    pub skill_tags: Vec<String>,
    /// Filter by agent name (partial match)
    pub name_contains: Option<String>,
    /// Only return active agents
    pub active_only: bool,
    /// Filter by capabilities
    pub requires_streaming: Option<bool>,
    /// Filter by authentication scheme
    pub auth_scheme: Option<String>,
    /// Maximum results to return
    pub limit: Option<usize>,
}

impl DiscoveryQuery {
    /// Create a query for agents with a specific skill
    pub fn with_skill(skill_id: &str) -> Self {
        Self {
            skill_id: Some(skill_id.to_string()),
            active_only: true,
            ..Default::default()
        }
    }

    /// Create a query for agents with specific tags
    pub fn with_tags(tags: Vec<String>) -> Self {
        Self {
            skill_tags: tags,
            active_only: true,
            ..Default::default()
        }
    }

    /// Set limit on results
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Require streaming support
    pub fn require_streaming(mut self) -> Self {
        self.requires_streaming = Some(true);
        self
    }
}

/// Discovery result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResult {
    /// Matching agents
    pub agents: Vec<AgentEntry>,
    /// Total matching (before limit)
    pub total_matches: usize,
    /// Query that was executed
    pub query: DiscoveryQuery,
}

// ============================================================================
// Agent Registry
// ============================================================================

/// Agent Discovery and Registry Service
pub struct AgentRegistry {
    config: RegistryConfig,
    /// Registered agents (keyed by URL)
    agents: HashMap<String, AgentEntry>,
    /// Skill index (skill_id -> [agent URLs])
    skill_index: HashMap<String, Vec<String>>,
    /// Tag index (tag -> [agent URLs])
    tag_index: HashMap<String, Vec<String>>,
}

impl AgentRegistry {
    /// Create a new registry with default config
    pub fn new() -> Self {
        Self::with_config(RegistryConfig::default())
    }

    /// Create a registry with custom config
    pub fn with_config(config: RegistryConfig) -> Self {
        Self {
            config,
            agents: HashMap::new(),
            skill_index: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Get registry configuration
    pub fn config(&self) -> &RegistryConfig {
        &self.config
    }

    /// Get number of registered agents
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Register an agent
    pub fn register(&mut self, card: AgentCard) -> Result<(), RegistryError> {
        // Check capacity
        if self.agents.len() >= self.config.max_agents {
            return Err(RegistryError::CapacityExceeded);
        }

        // Check for duplicate
        if self.agents.contains_key(&card.url) {
            return Err(RegistryError::AlreadyRegistered(card.url.clone()));
        }

        let url = card.url.clone();
        let entry = AgentEntry::new(card.clone());

        // Index skills
        for skill in &card.skills {
            self.skill_index
                .entry(skill.id.clone())
                .or_default()
                .push(url.clone());
        }

        // Index tags (collect unique tags first to avoid duplicates)
        let mut all_tags = std::collections::HashSet::new();
        for skill in &card.skills {
            for tag in &skill.tags {
                all_tags.insert(tag.clone());
            }
        }
        for tag in all_tags {
            self.tag_index
                .entry(tag)
                .or_default()
                .push(url.clone());
        }

        self.agents.insert(url, entry);
        Ok(())
    }

    /// Update an existing agent registration
    pub fn update(&mut self, card: AgentCard) -> Result<(), RegistryError> {
        let url = card.url.clone();

        if !self.agents.contains_key(&url) {
            return Err(RegistryError::NotFound(url));
        }

        // Remove old indexes
        self.remove_from_indexes(&url);

        // Re-index skills
        for skill in &card.skills {
            self.skill_index
                .entry(skill.id.clone())
                .or_default()
                .push(url.clone());
        }

        // Re-index tags (collect unique tags first to avoid duplicates)
        let mut all_tags = std::collections::HashSet::new();
        for skill in &card.skills {
            for tag in &skill.tags {
                all_tags.insert(tag.clone());
            }
        }
        for tag in all_tags {
            self.tag_index
                .entry(tag)
                .or_default()
                .push(url.clone());
        }

        // Update entry
        if let Some(entry) = self.agents.get_mut(&url) {
            entry.card = card;
        }

        Ok(())
    }

    /// Deregister an agent
    pub fn deregister(&mut self, url: &str) -> Result<AgentEntry, RegistryError> {
        // Remove from indexes
        self.remove_from_indexes(url);

        // Remove agent
        self.agents
            .remove(url)
            .ok_or_else(|| RegistryError::NotFound(url.to_string()))
    }

    /// Get an agent by URL
    pub fn get(&self, url: &str) -> Option<&AgentEntry> {
        self.agents.get(url)
    }

    /// Get mutable reference to an agent
    pub fn get_mut(&mut self, url: &str) -> Option<&mut AgentEntry> {
        self.agents.get_mut(url)
    }

    /// List all agents
    pub fn list_all(&self) -> Vec<&AgentEntry> {
        self.agents.values().collect()
    }

    /// List active agents only
    pub fn list_active(&self) -> Vec<&AgentEntry> {
        self.agents
            .values()
            .filter(|e| e.status == AgentStatus::Active)
            .collect()
    }

    /// Discover agents matching a query
    pub fn discover(&self, query: &DiscoveryQuery) -> DiscoveryResult {
        let mut candidates: Vec<&AgentEntry> = if let Some(skill_id) = &query.skill_id {
            // Start with skill index
            self.skill_index
                .get(skill_id)
                .map(|urls| urls.iter().filter_map(|u| self.agents.get(u)).collect())
                .unwrap_or_default()
        } else if !query.skill_tags.is_empty() {
            // Start with tag index
            let mut urls = Vec::new();
            for tag in &query.skill_tags {
                if let Some(tagged) = self.tag_index.get(tag) {
                    urls.extend(tagged.iter().cloned());
                }
            }
            urls.sort();
            urls.dedup();
            urls.iter().filter_map(|u| self.agents.get(u)).collect()
        } else {
            // All agents
            self.agents.values().collect()
        };

        // Apply filters
        if query.active_only {
            candidates.retain(|e| e.status == AgentStatus::Active || e.status == AgentStatus::Unknown);
        }

        if let Some(name) = &query.name_contains {
            let name_lower = name.to_lowercase();
            candidates.retain(|e| e.card.name.to_lowercase().contains(&name_lower));
        }

        if let Some(requires_streaming) = query.requires_streaming {
            candidates.retain(|e| e.card.capabilities.streaming == requires_streaming);
        }

        if let Some(auth_scheme) = &query.auth_scheme {
            candidates.retain(|e| e.card.authentication.schemes.contains(auth_scheme));
        }

        let total_matches = candidates.len();

        // Apply limit
        if let Some(limit) = query.limit {
            candidates.truncate(limit);
        }

        DiscoveryResult {
            agents: candidates.into_iter().cloned().collect(),
            total_matches,
            query: query.clone(),
        }
    }

    /// Find agents with a specific skill
    pub fn find_by_skill(&self, skill_id: &str) -> Vec<&AgentEntry> {
        self.skill_index
            .get(skill_id)
            .map(|urls| {
                urls.iter()
                    .filter_map(|u| self.agents.get(u))
                    .filter(|e| e.status == AgentStatus::Active || e.status == AgentStatus::Unknown)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find agents with a specific tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<&AgentEntry> {
        self.tag_index
            .get(tag)
            .map(|urls| {
                urls.iter()
                    .filter_map(|u| self.agents.get(u))
                    .filter(|e| e.status == AgentStatus::Active || e.status == AgentStatus::Unknown)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Update agent health status
    pub fn update_health(&mut self, url: &str, healthy: bool) -> Result<(), RegistryError> {
        let entry = self.agents
            .get_mut(url)
            .ok_or_else(|| RegistryError::NotFound(url.to_string()))?;

        if healthy {
            entry.mark_healthy();
        } else {
            entry.mark_unhealthy();

            // Auto-remove if too many failures
            if entry.should_remove(self.config.max_failed_health_checks) {
                let url = url.to_string();
                self.remove_from_indexes(&url);
                self.agents.remove(&url);
            }
        }

        Ok(())
    }

    /// Get all registered skill IDs
    pub fn list_skills(&self) -> Vec<&str> {
        self.skill_index.keys().map(|s| s.as_str()).collect()
    }

    /// Get all registered tags
    pub fn list_tags(&self) -> Vec<&str> {
        self.tag_index.keys().map(|s| s.as_str()).collect()
    }

    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        let mut active = 0;
        let mut unavailable = 0;
        let mut unknown = 0;

        for entry in self.agents.values() {
            match entry.status {
                AgentStatus::Active => active += 1,
                AgentStatus::Unavailable => unavailable += 1,
                AgentStatus::Unknown => unknown += 1,
                _ => {}
            }
        }

        RegistryStats {
            total_agents: self.agents.len(),
            active_agents: active,
            unavailable_agents: unavailable,
            unknown_agents: unknown,
            total_skills: self.skill_index.len(),
            total_tags: self.tag_index.len(),
        }
    }

    // Helper: Remove agent from all indexes
    fn remove_from_indexes(&mut self, url: &str) {
        for urls in self.skill_index.values_mut() {
            urls.retain(|u| u != url);
        }
        for urls in self.tag_index.values_mut() {
            urls.retain(|u| u != url);
        }

        // Clean up empty entries
        self.skill_index.retain(|_, v| !v.is_empty());
        self.tag_index.retain(|_, v| !v.is_empty());
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStats {
    /// Total registered agents
    pub total_agents: usize,
    /// Active agents
    pub active_agents: usize,
    /// Unavailable agents
    pub unavailable_agents: usize,
    /// Unknown status agents
    pub unknown_agents: usize,
    /// Total unique skills
    pub total_skills: usize,
    /// Total unique tags
    pub total_tags: usize,
}

// ============================================================================
// Errors
// ============================================================================

/// Registry error types
#[derive(Debug, Clone)]
pub enum RegistryError {
    /// Agent already registered
    AlreadyRegistered(String),
    /// Agent not found
    NotFound(String),
    /// Registry capacity exceeded
    CapacityExceeded,
    /// Invalid agent card
    InvalidCard(String),
    /// Health check failed
    HealthCheckFailed(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRegistered(url) => write!(f, "Agent already registered: {}", url),
            Self::NotFound(url) => write!(f, "Agent not found: {}", url),
            Self::CapacityExceeded => write!(f, "Registry capacity exceeded"),
            Self::InvalidCard(msg) => write!(f, "Invalid agent card: {}", msg),
            Self::HealthCheckFailed(msg) => write!(f, "Health check failed: {}", msg),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get current timestamp as ISO 8601 string
fn chrono_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}Z", now)
}

// ============================================================================
// Builder for Agent Cards
// ============================================================================

/// Builder for creating AgentCard instances
pub struct AgentCardBuilder {
    name: String,
    description: String,
    url: String,
    skills: Vec<Skill>,
    auth_schemes: Vec<String>,
    streaming: bool,
    push_notifications: bool,
    max_concurrent: Option<u32>,
}

impl AgentCardBuilder {
    /// Create a new builder
    pub fn new(name: &str, url: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            url: url.to_string(),
            skills: Vec::new(),
            auth_schemes: vec!["none".to_string()],
            streaming: false,
            push_notifications: false,
            max_concurrent: None,
        }
    }

    /// Set description
    pub fn description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a skill
    pub fn skill(mut self, skill: Skill) -> Self {
        self.skills.push(skill);
        self
    }

    /// Set authentication schemes
    pub fn auth(mut self, schemes: Vec<String>) -> Self {
        self.auth_schemes = schemes;
        self
    }

    /// Enable streaming
    pub fn streaming(mut self, enable: bool) -> Self {
        self.streaming = enable;
        self
    }

    /// Enable push notifications
    pub fn push_notifications(mut self, enable: bool) -> Self {
        self.push_notifications = enable;
        self
    }

    /// Set max concurrent tasks
    pub fn max_concurrent(mut self, max: u32) -> Self {
        self.max_concurrent = Some(max);
        self
    }

    /// Build the AgentCard
    pub fn build(self) -> AgentCard {
        AgentCard {
            name: self.name,
            description: self.description,
            url: self.url,
            protocol_version: "1.0".to_string(),
            authentication: AuthenticationInfo {
                schemes: self.auth_schemes,
                oauth2: None,
            },
            capabilities: AgentCapabilities {
                streaming: self.streaming,
                push_notifications: self.push_notifications,
                max_concurrent_tasks: self.max_concurrent,
            },
            skills: self.skills,
        }
    }
}

/// Builder for creating Skill instances
pub struct SkillBuilder {
    id: String,
    name: String,
    description: String,
    tags: Vec<String>,
}

impl SkillBuilder {
    /// Create a new builder
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            tags: Vec::new(),
        }
    }

    /// Set description
    pub fn description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a tag
    pub fn tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Add multiple tags
    pub fn tags(mut self, tags: Vec<&str>) -> Self {
        self.tags.extend(tags.into_iter().map(String::from));
        self
    }

    /// Build the Skill
    pub fn build(self) -> Skill {
        use serde_json::json;
        Skill {
            id: self.id,
            name: self.name,
            description: self.description,
            input_schema: json!({"type": "object"}),
            output_schema: json!({"type": "object"}),
            tags: self.tags,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use serde_json::json;

    fn create_test_skill(id: &str, tags: Vec<&str>) -> Skill {
        SkillBuilder::new(id, &format!("{} Skill", id))
            .description("Test skill")
            .tags(tags)
            .build()
    }

    fn create_test_card(name: &str, url: &str, skills: Vec<Skill>) -> AgentCard {
        AgentCardBuilder::new(name, url)
            .description("Test agent")
            .streaming(true)
            .build()
            .with_skills(skills)
    }

    // Extension trait for adding skills to AgentCard
    trait AgentCardExt {
        fn with_skills(self, skills: Vec<Skill>) -> Self;
    }

    impl AgentCardExt for AgentCard {
        fn with_skills(mut self, skills: Vec<Skill>) -> Self {
            self.skills = skills;
            self
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = AgentRegistry::new();
        assert_eq!(registry.agent_count(), 0);
    }

    #[test]
    fn test_register_agent() {
        let mut registry = AgentRegistry::new();
        let card = create_test_card(
            "TestAgent",
            "http://localhost:8080",
            vec![create_test_skill("graph", vec!["ml", "graphs"])],
        );

        let result = registry.register(card);
        assert!(result.is_ok());
        assert_eq!(registry.agent_count(), 1);
    }

    #[test]
    fn test_duplicate_registration() {
        let mut registry = AgentRegistry::new();
        let card = create_test_card("TestAgent", "http://localhost:8080", vec![]);

        registry.register(card.clone()).unwrap();
        let result = registry.register(card);

        assert!(matches!(result, Err(RegistryError::AlreadyRegistered(_))));
    }

    #[test]
    fn test_deregister_agent() {
        let mut registry = AgentRegistry::new();
        let card = create_test_card("TestAgent", "http://localhost:8080", vec![]);

        registry.register(card).unwrap();
        let result = registry.deregister("http://localhost:8080");

        assert!(result.is_ok());
        assert_eq!(registry.agent_count(), 0);
    }

    #[test]
    fn test_deregister_not_found() {
        let mut registry = AgentRegistry::new();
        let result = registry.deregister("http://nonexistent");

        assert!(matches!(result, Err(RegistryError::NotFound(_))));
    }

    #[test]
    fn test_find_by_skill() {
        let mut registry = AgentRegistry::new();

        let card1 = create_test_card(
            "Agent1",
            "http://agent1.local",
            vec![create_test_skill("text_gen", vec!["nlp"])],
        );
        let card2 = create_test_card(
            "Agent2",
            "http://agent2.local",
            vec![create_test_skill("image_gen", vec!["vision"])],
        );

        registry.register(card1).unwrap();
        registry.register(card2).unwrap();

        let text_agents = registry.find_by_skill("text_gen");
        assert_eq!(text_agents.len(), 1);
        assert_eq!(text_agents[0].card.name, "Agent1");

        let image_agents = registry.find_by_skill("image_gen");
        assert_eq!(image_agents.len(), 1);
        assert_eq!(image_agents[0].card.name, "Agent2");
    }

    #[test]
    fn test_find_by_tag() {
        let mut registry = AgentRegistry::new();

        let card = create_test_card(
            "MLAgent",
            "http://ml.local",
            vec![
                create_test_skill("train", vec!["ml", "gpu"]),
                create_test_skill("predict", vec!["ml", "inference"]),
            ],
        );

        registry.register(card).unwrap();

        let ml_agents = registry.find_by_tag("ml");
        assert_eq!(ml_agents.len(), 1);

        let gpu_agents = registry.find_by_tag("gpu");
        assert_eq!(gpu_agents.len(), 1);
    }

    #[test]
    fn test_discovery_query() {
        let mut registry = AgentRegistry::new();

        let card1 = create_test_card(
            "TextBot",
            "http://text.local",
            vec![create_test_skill("text_gen", vec!["nlp"])],
        );
        let card2 = create_test_card(
            "ImageBot",
            "http://image.local",
            vec![create_test_skill("image_gen", vec!["vision"])],
        );

        registry.register(card1).unwrap();
        registry.register(card2).unwrap();

        // Query by skill
        let query = DiscoveryQuery::with_skill("text_gen");
        let result = registry.discover(&query);
        assert_eq!(result.total_matches, 1);
        assert_eq!(result.agents[0].card.name, "TextBot");
    }

    #[test]
    fn test_discovery_with_name_filter() {
        let mut registry = AgentRegistry::new();

        let card1 = create_test_card("TextBot", "http://text.local", vec![]);
        let card2 = create_test_card("ImageBot", "http://image.local", vec![]);

        registry.register(card1).unwrap();
        registry.register(card2).unwrap();

        let mut query = DiscoveryQuery::default();
        query.name_contains = Some("Text".to_string());

        let result = registry.discover(&query);
        assert_eq!(result.total_matches, 1);
        assert_eq!(result.agents[0].card.name, "TextBot");
    }

    #[test]
    fn test_health_update() {
        let mut registry = AgentRegistry::new();
        let card = create_test_card("TestAgent", "http://test.local", vec![]);

        registry.register(card).unwrap();

        // Mark healthy
        registry.update_health("http://test.local", true).unwrap();
        let entry = registry.get("http://test.local").unwrap();
        assert_eq!(entry.status, AgentStatus::Active);
        assert_eq!(entry.failed_health_checks, 0);

        // Mark unhealthy
        registry.update_health("http://test.local", false).unwrap();
        let entry = registry.get("http://test.local").unwrap();
        assert_eq!(entry.status, AgentStatus::Unavailable);
        assert_eq!(entry.failed_health_checks, 1);
    }

    #[test]
    fn test_auto_remove_unhealthy() {
        let config = RegistryConfig {
            max_failed_health_checks: 2,
            ..Default::default()
        };
        let mut registry = AgentRegistry::with_config(config);

        let card = create_test_card("UnhealthyAgent", "http://unhealthy.local", vec![]);
        registry.register(card).unwrap();

        // First failure
        registry.update_health("http://unhealthy.local", false).unwrap();
        assert!(registry.get("http://unhealthy.local").is_some());

        // Second failure - should auto-remove
        registry.update_health("http://unhealthy.local", false).unwrap();
        assert!(registry.get("http://unhealthy.local").is_none());
    }

    #[test]
    fn test_registry_stats() {
        let mut registry = AgentRegistry::new();

        let card1 = create_test_card(
            "Agent1",
            "http://agent1.local",
            vec![create_test_skill("skill1", vec!["tag1"])],
        );
        let card2 = create_test_card(
            "Agent2",
            "http://agent2.local",
            vec![create_test_skill("skill2", vec!["tag2"])],
        );

        registry.register(card1).unwrap();
        registry.register(card2).unwrap();

        registry.update_health("http://agent1.local", true).unwrap();

        let stats = registry.stats();
        assert_eq!(stats.total_agents, 2);
        assert_eq!(stats.active_agents, 1);
        assert_eq!(stats.unknown_agents, 1);
        assert_eq!(stats.total_skills, 2);
    }

    #[test]
    fn test_capacity_limit() {
        let config = RegistryConfig {
            max_agents: 2,
            ..Default::default()
        };
        let mut registry = AgentRegistry::with_config(config);

        registry.register(create_test_card("A1", "http://a1", vec![])).unwrap();
        registry.register(create_test_card("A2", "http://a2", vec![])).unwrap();

        let result = registry.register(create_test_card("A3", "http://a3", vec![]));
        assert!(matches!(result, Err(RegistryError::CapacityExceeded)));
    }

    #[test]
    fn test_agent_card_builder() {
        let skill = SkillBuilder::new("graph_transform", "Graph Transform")
            .description("Transform graphs")
            .tags(vec!["ml", "graphs"])
            .build();

        let card = AgentCardBuilder::new("GraphBot", "http://graph.local")
            .description("Graph processing agent")
            .skill(skill)
            .streaming(true)
            .max_concurrent(5)
            .build();

        assert_eq!(card.name, "GraphBot");
        assert!(card.capabilities.streaming);
        assert_eq!(card.capabilities.max_concurrent_tasks, Some(5));
        assert_eq!(card.skills.len(), 1);
    }

    #[test]
    fn test_list_skills_and_tags() {
        let mut registry = AgentRegistry::new();

        let card = create_test_card(
            "MultiSkill",
            "http://multi.local",
            vec![
                create_test_skill("s1", vec!["t1", "t2"]),
                create_test_skill("s2", vec!["t2", "t3"]),
            ],
        );

        registry.register(card).unwrap();

        let skills = registry.list_skills();
        assert_eq!(skills.len(), 2);

        let tags = registry.list_tags();
        assert_eq!(tags.len(), 3);
    }

    #[test]
    fn test_update_agent() {
        let mut registry = AgentRegistry::new();

        let card1 = create_test_card(
            "Agent",
            "http://agent.local",
            vec![create_test_skill("old_skill", vec![])],
        );
        registry.register(card1).unwrap();

        // Update with new skills
        let card2 = create_test_card(
            "UpdatedAgent",
            "http://agent.local",
            vec![create_test_skill("new_skill", vec!["new_tag"])],
        );
        registry.update(card2).unwrap();

        let entry = registry.get("http://agent.local").unwrap();
        assert_eq!(entry.card.name, "UpdatedAgent");
        assert_eq!(entry.card.skills[0].id, "new_skill");

        // Old skill should be gone
        assert!(registry.find_by_skill("old_skill").is_empty());
        assert_eq!(registry.find_by_skill("new_skill").len(), 1);
    }

    #[test]
    fn test_discovery_limit() {
        let mut registry = AgentRegistry::new();

        for i in 0..10 {
            let card = create_test_card(
                &format!("Agent{}", i),
                &format!("http://agent{}.local", i),
                vec![],
            );
            registry.register(card).unwrap();
        }

        let query = DiscoveryQuery::default().limit(3);
        let result = registry.discover(&query);

        assert_eq!(result.total_matches, 10);
        assert_eq!(result.agents.len(), 3);
    }

    #[test]
    fn test_registry_error_display() {
        let error = RegistryError::NotFound("test".to_string());
        assert_eq!(format!("{}", error), "Agent not found: test");

        let error = RegistryError::CapacityExceeded;
        assert_eq!(format!("{}", error), "Registry capacity exceeded");
    }
}
