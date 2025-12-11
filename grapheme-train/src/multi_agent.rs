//! Multi-Agent Orchestration Module
//!
//! Backend-177: Multi-agent orchestration with GRAPHEME coordinator.
//!
//! Implements orchestration patterns for coordinating multiple AI agents:
//! - Agent registration and discovery
//! - Task distribution and load balancing
//! - Result aggregation and consensus
//! - Fault tolerance and retry handling

use crate::a2a_protocol::AgentCard;
use crate::a2a_registry::AgentStatus;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Orchestration Configuration
// ============================================================================

/// Configuration for multi-agent orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationConfig {
    /// Maximum number of agents to use per task
    pub max_agents_per_task: usize,
    /// Timeout for agent responses (seconds)
    pub agent_timeout_secs: u64,
    /// Number of retries on failure
    pub max_retries: u32,
    /// Strategy for selecting agents
    pub selection_strategy: AgentSelectionStrategy,
    /// Strategy for aggregating results
    pub aggregation_strategy: AggregationStrategy,
    /// Enable parallel execution
    pub parallel_execution: bool,
    /// Minimum agents required for consensus
    pub min_consensus_agents: usize,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            max_agents_per_task: 5,
            agent_timeout_secs: 30,
            max_retries: 3,
            selection_strategy: AgentSelectionStrategy::RoundRobin,
            aggregation_strategy: AggregationStrategy::FirstSuccess,
            parallel_execution: true,
            min_consensus_agents: 3,
        }
    }
}

/// Strategy for selecting agents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentSelectionStrategy {
    /// Select agents in round-robin order
    RoundRobin,
    /// Select least loaded agents first
    LeastLoaded,
    /// Random selection
    Random,
    /// Select by skill match score
    SkillMatch,
    /// Use all available agents
    Broadcast,
}

/// Strategy for aggregating results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Use first successful result
    FirstSuccess,
    /// Require majority consensus
    Consensus,
    /// Merge all results
    Merge,
    /// Use best result by score
    BestScore,
    /// Collect all results
    CollectAll,
}

// ============================================================================
// Agent Wrapper
// ============================================================================

/// Wrapper for an agent with orchestration metadata
#[derive(Debug, Clone)]
pub struct ManagedAgent {
    /// Agent card
    pub card: AgentCard,
    /// Current status
    pub status: AgentStatus,
    /// Active task count
    pub active_tasks: usize,
    /// Total tasks processed
    pub total_processed: usize,
    /// Success count
    pub success_count: usize,
    /// Failure count
    pub failure_count: usize,
    /// Average response time (ms)
    pub avg_response_time_ms: f64,
}

impl ManagedAgent {
    /// Create from agent card
    pub fn new(card: AgentCard) -> Self {
        Self {
            card,
            status: AgentStatus::Active,
            active_tasks: 0,
            total_processed: 0,
            success_count: 0,
            failure_count: 0,
            avg_response_time_ms: 0.0,
        }
    }

    /// Get agent URL
    pub fn url(&self) -> &str {
        &self.card.url
    }

    /// Get agent name
    pub fn name(&self) -> &str {
        &self.card.name
    }

    /// Check if agent has a specific skill
    pub fn has_skill(&self, skill_id: &str) -> bool {
        self.card.skills.iter().any(|s| s.id == skill_id)
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_processed == 0 {
            1.0 // Default to perfect for new agents
        } else {
            self.success_count as f64 / self.total_processed as f64
        }
    }

    /// Record task start
    pub fn task_started(&mut self) {
        self.active_tasks += 1;
    }

    /// Record task completion
    pub fn task_completed(&mut self, success: bool, response_time_ms: u64) {
        self.active_tasks = self.active_tasks.saturating_sub(1);
        self.total_processed += 1;
        if success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
        // Update moving average
        let n = self.total_processed as f64;
        self.avg_response_time_ms =
            (self.avg_response_time_ms * (n - 1.0) + response_time_ms as f64) / n;
    }
}

// ============================================================================
// Orchestrated Task
// ============================================================================

/// A task distributed across multiple agents
#[derive(Debug, Clone)]
pub struct OrchestratedTask {
    /// Task ID
    pub id: String,
    /// Task description
    pub description: String,
    /// Required skill
    pub required_skill: Option<String>,
    /// Task input data
    pub input: serde_json::Value,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Task status
    pub status: OrchestratedTaskStatus,
    /// Assigned agents
    pub assigned_agents: Vec<String>,
    /// Results from agents
    pub results: Vec<AgentResult>,
    /// Creation time
    pub created_at: u64,
    /// Retry count
    pub retry_count: u32,
}

/// Status of an orchestrated task
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrchestratedTaskStatus {
    /// Waiting to be assigned
    Pending,
    /// Assigned to agents
    Assigned,
    /// Currently executing
    Running,
    /// Completed successfully
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
}

/// Result from a single agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Agent URL
    pub agent_url: String,
    /// Result status
    pub status: AgentResultStatus,
    /// Result data
    pub data: Option<serde_json::Value>,
    /// Error message if failed
    pub error: Option<String>,
    /// Response time (ms)
    pub response_time_ms: u64,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Status of an agent result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentResultStatus {
    Success,
    Error,
    Timeout,
    Rejected,
}

impl OrchestratedTask {
    /// Create new task
    pub fn new(description: &str, input: serde_json::Value) -> Self {
        Self {
            id: format!("ot_{}", uuid_simple()),
            description: description.to_string(),
            required_skill: None,
            input,
            priority: 0,
            status: OrchestratedTaskStatus::Pending,
            assigned_agents: Vec::new(),
            results: Vec::new(),
            created_at: current_timestamp(),
            retry_count: 0,
        }
    }

    /// Set required skill
    pub fn with_skill(mut self, skill_id: &str) -> Self {
        self.required_skill = Some(skill_id.to_string());
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Add result
    pub fn add_result(&mut self, result: AgentResult) {
        self.results.push(result);
    }

    /// Get successful results
    pub fn successful_results(&self) -> Vec<&AgentResult> {
        self.results
            .iter()
            .filter(|r| r.status == AgentResultStatus::Success)
            .collect()
    }

    /// Check if has any success
    pub fn has_success(&self) -> bool {
        self.results.iter().any(|r| r.status == AgentResultStatus::Success)
    }
}

// ============================================================================
// Coordinator
// ============================================================================

/// Central coordinator for multi-agent orchestration
pub struct AgentCoordinator {
    /// Configuration
    config: OrchestrationConfig,
    /// Managed agents
    agents: HashMap<String, ManagedAgent>,
    /// Pending tasks
    pending_tasks: Vec<OrchestratedTask>,
    /// Active tasks
    active_tasks: HashMap<String, OrchestratedTask>,
    /// Completed tasks
    completed_tasks: Vec<OrchestratedTask>,
    /// Round-robin index
    rr_index: usize,
    /// Statistics
    stats: CoordinatorStats,
}

/// Coordinator statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoordinatorStats {
    /// Total tasks submitted
    pub tasks_submitted: usize,
    /// Tasks completed successfully
    pub tasks_succeeded: usize,
    /// Tasks failed
    pub tasks_failed: usize,
    /// Total agent invocations
    pub agent_invocations: usize,
    /// Average task completion time (ms)
    pub avg_completion_time_ms: f64,
}

impl AgentCoordinator {
    /// Create new coordinator
    pub fn new(config: OrchestrationConfig) -> Self {
        Self {
            config,
            agents: HashMap::new(),
            pending_tasks: Vec::new(),
            active_tasks: HashMap::new(),
            completed_tasks: Vec::new(),
            rr_index: 0,
            stats: CoordinatorStats::default(),
        }
    }

    /// Create with default config
    pub fn default_coordinator() -> Self {
        Self::new(OrchestrationConfig::default())
    }

    /// Get configuration
    pub fn config(&self) -> &OrchestrationConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &CoordinatorStats {
        &self.stats
    }

    /// Register an agent
    pub fn register_agent(&mut self, card: AgentCard) {
        let url = card.url.clone();
        self.agents.insert(url, ManagedAgent::new(card));
    }

    /// Unregister an agent
    pub fn unregister_agent(&mut self, url: &str) {
        self.agents.remove(url);
    }

    /// Get agent by URL
    pub fn get_agent(&self, url: &str) -> Option<&ManagedAgent> {
        self.agents.get(url)
    }

    /// List all agents
    pub fn list_agents(&self) -> Vec<&ManagedAgent> {
        self.agents.values().collect()
    }

    /// List active agents
    pub fn active_agents(&self) -> Vec<&ManagedAgent> {
        self.agents
            .values()
            .filter(|a| a.status == AgentStatus::Active)
            .collect()
    }

    /// Submit a task for orchestration
    pub fn submit_task(&mut self, task: OrchestratedTask) -> String {
        let task_id = task.id.clone();
        self.pending_tasks.push(task);
        self.stats.tasks_submitted += 1;
        task_id
    }

    /// Select agents for a task based on strategy
    pub fn select_agents(&mut self, task: &OrchestratedTask) -> Vec<String> {
        let available: Vec<&ManagedAgent> = self
            .agents
            .values()
            .filter(|a| {
                a.status == AgentStatus::Active
                    && task
                        .required_skill
                        .as_ref()
                        .map(|s| a.has_skill(s))
                        .unwrap_or(true)
            })
            .collect();

        if available.is_empty() {
            return Vec::new();
        }

        let max_agents = self.config.max_agents_per_task.min(available.len());

        match self.config.selection_strategy {
            AgentSelectionStrategy::RoundRobin => {
                let mut selected = Vec::new();
                for i in 0..max_agents {
                    let idx = (self.rr_index + i) % available.len();
                    selected.push(available[idx].url().to_string());
                }
                self.rr_index = (self.rr_index + max_agents) % available.len();
                selected
            }
            AgentSelectionStrategy::LeastLoaded => {
                let mut sorted: Vec<_> = available.iter().collect();
                sorted.sort_by_key(|a| a.active_tasks);
                sorted
                    .iter()
                    .take(max_agents)
                    .map(|a| a.url().to_string())
                    .collect()
            }
            AgentSelectionStrategy::Random => {
                // Simple pseudo-random using task ID hash
                let hash = task.id.bytes().fold(0usize, |acc, b| acc.wrapping_add(b as usize));
                let mut selected = Vec::new();
                for i in 0..max_agents {
                    let idx = (hash + i) % available.len();
                    let url = available[idx].url().to_string();
                    if !selected.contains(&url) {
                        selected.push(url);
                    }
                }
                selected
            }
            AgentSelectionStrategy::SkillMatch => {
                // Prioritize agents with exact skill match
                let mut scored: Vec<(&ManagedAgent, f64)> = available
                    .iter()
                    .map(|a| {
                        let skill_score = if let Some(ref skill) = task.required_skill {
                            if a.has_skill(skill) {
                                1.0
                            } else {
                                0.5
                            }
                        } else {
                            0.8
                        };
                        let success_score = a.success_rate();
                        (*a, skill_score * 0.6 + success_score * 0.4)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored
                    .iter()
                    .take(max_agents)
                    .map(|(a, _)| a.url().to_string())
                    .collect()
            }
            AgentSelectionStrategy::Broadcast => {
                available.iter().map(|a| a.url().to_string()).collect()
            }
        }
    }

    /// Process pending tasks (assign to agents)
    pub fn process_pending(&mut self) -> Vec<(String, Vec<String>)> {
        let mut assignments = Vec::new();

        // Sort by priority (highest first)
        self.pending_tasks.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Process pending tasks
        let mut tasks_to_assign = Vec::new();
        std::mem::swap(&mut self.pending_tasks, &mut tasks_to_assign);

        for mut task in tasks_to_assign {
            let agents = self.select_agents(&task);
            if agents.is_empty() {
                // No agents available, return to pending
                self.pending_tasks.push(task);
            } else {
                // Assign to agents
                task.assigned_agents = agents.clone();
                task.status = OrchestratedTaskStatus::Assigned;
                let task_id = task.id.clone();

                // Update agent stats
                for url in &agents {
                    if let Some(agent) = self.agents.get_mut(url) {
                        agent.task_started();
                    }
                }

                self.active_tasks.insert(task_id.clone(), task);
                assignments.push((task_id, agents));
            }
        }

        self.stats.agent_invocations += assignments.iter().map(|(_, a)| a.len()).sum::<usize>();
        assignments
    }

    /// Record a result from an agent
    pub fn record_result(&mut self, task_id: &str, result: AgentResult) {
        if let Some(task) = self.active_tasks.get_mut(task_id) {
            // Update agent stats
            if let Some(agent) = self.agents.get_mut(&result.agent_url) {
                agent.task_completed(
                    result.status == AgentResultStatus::Success,
                    result.response_time_ms,
                );
            }

            task.add_result(result);
            task.status = OrchestratedTaskStatus::Running;
        }
    }

    /// Aggregate results for a task
    pub fn aggregate_results(&self, task: &OrchestratedTask) -> Option<AggregatedResult> {
        if task.results.is_empty() {
            return None;
        }

        match self.config.aggregation_strategy {
            AggregationStrategy::FirstSuccess => {
                task.results
                    .iter()
                    .find(|r| r.status == AgentResultStatus::Success)
                    .map(|r| AggregatedResult {
                        data: r.data.clone(),
                        confidence: r.confidence,
                        source_agents: vec![r.agent_url.clone()],
                        strategy_used: AggregationStrategy::FirstSuccess,
                    })
            }
            AggregationStrategy::Consensus => {
                let successful: Vec<_> = task
                    .results
                    .iter()
                    .filter(|r| r.status == AgentResultStatus::Success)
                    .collect();

                if successful.len() < self.config.min_consensus_agents {
                    return None;
                }

                // Simple consensus: use most common result (by JSON serialization)
                let mut result_counts: HashMap<String, (usize, &AgentResult)> = HashMap::new();
                for r in &successful {
                    let key = r.data.as_ref().map(|d| d.to_string()).unwrap_or_default();
                    result_counts
                        .entry(key)
                        .and_modify(|(count, _)| *count += 1)
                        .or_insert((1, r));
                }

                result_counts
                    .into_iter()
                    .max_by_key(|(_, (count, _))| *count)
                    .map(|(_, (count, r))| AggregatedResult {
                        data: r.data.clone(),
                        confidence: count as f32 / successful.len() as f32,
                        source_agents: successful.iter().map(|r| r.agent_url.clone()).collect(),
                        strategy_used: AggregationStrategy::Consensus,
                    })
            }
            AggregationStrategy::Merge => {
                let successful: Vec<_> = task
                    .results
                    .iter()
                    .filter(|r| r.status == AgentResultStatus::Success)
                    .collect();

                if successful.is_empty() {
                    return None;
                }

                // Merge all results into array
                let merged: Vec<_> = successful
                    .iter()
                    .filter_map(|r| r.data.clone())
                    .collect();

                let avg_confidence =
                    successful.iter().map(|r| r.confidence).sum::<f32>() / successful.len() as f32;

                Some(AggregatedResult {
                    data: Some(serde_json::Value::Array(merged)),
                    confidence: avg_confidence,
                    source_agents: successful.iter().map(|r| r.agent_url.clone()).collect(),
                    strategy_used: AggregationStrategy::Merge,
                })
            }
            AggregationStrategy::BestScore => {
                task.results
                    .iter()
                    .filter(|r| r.status == AgentResultStatus::Success)
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                    .map(|r| AggregatedResult {
                        data: r.data.clone(),
                        confidence: r.confidence,
                        source_agents: vec![r.agent_url.clone()],
                        strategy_used: AggregationStrategy::BestScore,
                    })
            }
            AggregationStrategy::CollectAll => {
                let all_results: Vec<_> = task
                    .results
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "agent": r.agent_url,
                            "status": format!("{:?}", r.status),
                            "data": r.data,
                            "confidence": r.confidence
                        })
                    })
                    .collect();

                let avg_confidence = task
                    .results
                    .iter()
                    .filter(|r| r.status == AgentResultStatus::Success)
                    .map(|r| r.confidence)
                    .sum::<f32>()
                    / task.results.len().max(1) as f32;

                Some(AggregatedResult {
                    data: Some(serde_json::Value::Array(all_results)),
                    confidence: avg_confidence,
                    source_agents: task.results.iter().map(|r| r.agent_url.clone()).collect(),
                    strategy_used: AggregationStrategy::CollectAll,
                })
            }
        }
    }

    /// Complete a task
    pub fn complete_task(&mut self, task_id: &str, success: bool) -> Option<OrchestratedTask> {
        if let Some(mut task) = self.active_tasks.remove(task_id) {
            task.status = if success {
                self.stats.tasks_succeeded += 1;
                OrchestratedTaskStatus::Completed
            } else {
                self.stats.tasks_failed += 1;
                OrchestratedTaskStatus::Failed
            };

            // Update avg completion time
            let completion_time = current_timestamp() - task.created_at;
            let n = self.stats.tasks_succeeded + self.stats.tasks_failed;
            self.stats.avg_completion_time_ms = (self.stats.avg_completion_time_ms
                * (n - 1) as f64
                + completion_time as f64)
                / n as f64;

            self.completed_tasks.push(task.clone());
            Some(task)
        } else {
            None
        }
    }

    /// Retry a failed task
    pub fn retry_task(&mut self, task_id: &str) -> bool {
        if let Some(mut task) = self.active_tasks.remove(task_id) {
            if task.retry_count < self.config.max_retries {
                task.retry_count += 1;
                task.status = OrchestratedTaskStatus::Pending;
                task.results.clear();
                task.assigned_agents.clear();
                self.pending_tasks.push(task);
                return true;
            }
        }
        false
    }

    /// Get task by ID
    pub fn get_task(&self, task_id: &str) -> Option<&OrchestratedTask> {
        self.active_tasks
            .get(task_id)
            .or_else(|| self.pending_tasks.iter().find(|t| t.id == task_id))
            .or_else(|| self.completed_tasks.iter().find(|t| t.id == task_id))
    }

    /// Get pending task count
    pub fn pending_count(&self) -> usize {
        self.pending_tasks.len()
    }

    /// Get active task count
    pub fn active_count(&self) -> usize {
        self.active_tasks.len()
    }
}

/// Aggregated result from multiple agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedResult {
    /// Final aggregated data
    pub data: Option<serde_json::Value>,
    /// Confidence in the result
    pub confidence: f32,
    /// Agents that contributed
    pub source_agents: Vec<String>,
    /// Strategy used
    pub strategy_used: AggregationStrategy,
}

// ============================================================================
// Task Builder
// ============================================================================

/// Builder for creating orchestrated tasks
pub struct TaskBuilder {
    task: OrchestratedTask,
}

impl TaskBuilder {
    /// Create new builder
    pub fn new(description: &str) -> Self {
        Self {
            task: OrchestratedTask::new(description, serde_json::Value::Null),
        }
    }

    /// Set input data
    pub fn input(mut self, input: serde_json::Value) -> Self {
        self.task.input = input;
        self
    }

    /// Set required skill
    pub fn skill(mut self, skill_id: &str) -> Self {
        self.task.required_skill = Some(skill_id.to_string());
        self
    }

    /// Set priority
    pub fn priority(mut self, priority: u32) -> Self {
        self.task.priority = priority;
        self
    }

    /// Build the task
    pub fn build(self) -> OrchestratedTask {
        self.task
    }
}

// ============================================================================
// Utilities
// ============================================================================

/// Generate simple UUID-like string
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:016x}", now)
}

/// Get current timestamp (ms since epoch)
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a2a_protocol::{AgentCapabilities, AuthenticationInfo, Skill};

    fn create_test_card(url: &str, name: &str) -> AgentCard {
        AgentCard {
            name: name.to_string(),
            description: format!("Test agent {}", name),
            url: url.to_string(),
            protocol_version: "1.0".to_string(),
            authentication: AuthenticationInfo {
                schemes: vec!["api_key".to_string()],
                oauth2: None,
            },
            capabilities: AgentCapabilities {
                streaming: true,
                push_notifications: false,
                max_concurrent_tasks: Some(10),
            },
            skills: vec![Skill {
                id: "test_skill".to_string(),
                name: "Test Skill".to_string(),
                description: "A test skill".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: serde_json::json!({}),
                tags: vec!["test".to_string()],
            }],
        }
    }

    #[test]
    fn test_orchestration_config_default() {
        let config = OrchestrationConfig::default();
        assert_eq!(config.max_agents_per_task, 5);
        assert_eq!(config.max_retries, 3);
        assert!(config.parallel_execution);
    }

    #[test]
    fn test_managed_agent_creation() {
        let card = create_test_card("http://agent1.local", "Agent1");
        let agent = ManagedAgent::new(card);
        assert_eq!(agent.name(), "Agent1");
        assert_eq!(agent.status, AgentStatus::Active);
        assert_eq!(agent.active_tasks, 0);
    }

    #[test]
    fn test_managed_agent_has_skill() {
        let card = create_test_card("http://agent1.local", "Agent1");
        let agent = ManagedAgent::new(card);
        assert!(agent.has_skill("test_skill"));
        assert!(!agent.has_skill("unknown_skill"));
    }

    #[test]
    fn test_managed_agent_task_tracking() {
        let card = create_test_card("http://agent1.local", "Agent1");
        let mut agent = ManagedAgent::new(card);

        agent.task_started();
        assert_eq!(agent.active_tasks, 1);

        agent.task_completed(true, 100);
        assert_eq!(agent.active_tasks, 0);
        assert_eq!(agent.total_processed, 1);
        assert_eq!(agent.success_count, 1);
        assert_eq!(agent.avg_response_time_ms, 100.0);
    }

    #[test]
    fn test_managed_agent_success_rate() {
        let card = create_test_card("http://agent1.local", "Agent1");
        let mut agent = ManagedAgent::new(card);

        // Default success rate for new agent
        assert_eq!(agent.success_rate(), 1.0);

        // After some tasks
        agent.task_started();
        agent.task_completed(true, 100);
        agent.task_started();
        agent.task_completed(false, 100);

        assert_eq!(agent.success_rate(), 0.5);
    }

    #[test]
    fn test_orchestrated_task_creation() {
        let task = OrchestratedTask::new("Test task", serde_json::json!({"key": "value"}));
        assert!(task.id.starts_with("ot_"));
        assert_eq!(task.description, "Test task");
        assert_eq!(task.status, OrchestratedTaskStatus::Pending);
    }

    #[test]
    fn test_task_builder() {
        let task = TaskBuilder::new("Build test")
            .input(serde_json::json!({"data": "test"}))
            .skill("my_skill")
            .priority(10)
            .build();

        assert_eq!(task.description, "Build test");
        assert_eq!(task.required_skill, Some("my_skill".to_string()));
        assert_eq!(task.priority, 10);
    }

    #[test]
    fn test_coordinator_creation() {
        let coord = AgentCoordinator::default_coordinator();
        assert_eq!(coord.pending_count(), 0);
        assert_eq!(coord.active_count(), 0);
    }

    #[test]
    fn test_coordinator_register_agent() {
        let mut coord = AgentCoordinator::default_coordinator();
        let card = create_test_card("http://agent1.local", "Agent1");

        coord.register_agent(card);
        assert_eq!(coord.list_agents().len(), 1);
        assert!(coord.get_agent("http://agent1.local").is_some());
    }

    #[test]
    fn test_coordinator_unregister_agent() {
        let mut coord = AgentCoordinator::default_coordinator();
        let card = create_test_card("http://agent1.local", "Agent1");

        coord.register_agent(card);
        coord.unregister_agent("http://agent1.local");
        assert_eq!(coord.list_agents().len(), 0);
    }

    #[test]
    fn test_coordinator_submit_task() {
        let mut coord = AgentCoordinator::default_coordinator();
        let task = OrchestratedTask::new("Test", serde_json::json!({}));

        let task_id = coord.submit_task(task);
        assert!(task_id.starts_with("ot_"));
        assert_eq!(coord.pending_count(), 1);
        assert_eq!(coord.stats().tasks_submitted, 1);
    }

    #[test]
    fn test_coordinator_process_pending_no_agents() {
        let mut coord = AgentCoordinator::default_coordinator();
        let task = OrchestratedTask::new("Test", serde_json::json!({}));

        coord.submit_task(task);
        let assignments = coord.process_pending();

        // No agents, task stays pending
        assert!(assignments.is_empty());
        assert_eq!(coord.pending_count(), 1);
    }

    #[test]
    fn test_coordinator_process_pending_with_agents() {
        let mut coord = AgentCoordinator::default_coordinator();

        // Register agents
        coord.register_agent(create_test_card("http://agent1.local", "Agent1"));
        coord.register_agent(create_test_card("http://agent2.local", "Agent2"));

        // Submit task
        let task = OrchestratedTask::new("Test", serde_json::json!({}));
        coord.submit_task(task);

        let assignments = coord.process_pending();
        assert_eq!(assignments.len(), 1);
        assert!(!assignments[0].1.is_empty());
        assert_eq!(coord.active_count(), 1);
        assert_eq!(coord.pending_count(), 0);
    }

    #[test]
    fn test_coordinator_record_result() {
        let mut coord = AgentCoordinator::default_coordinator();
        coord.register_agent(create_test_card("http://agent1.local", "Agent1"));

        let task = OrchestratedTask::new("Test", serde_json::json!({}));
        let task_id = coord.submit_task(task);
        coord.process_pending();

        let result = AgentResult {
            agent_url: "http://agent1.local".to_string(),
            status: AgentResultStatus::Success,
            data: Some(serde_json::json!({"result": "ok"})),
            error: None,
            response_time_ms: 50,
            confidence: 0.95,
        };

        coord.record_result(&task_id, result);

        let task = coord.get_task(&task_id).unwrap();
        assert_eq!(task.results.len(), 1);
    }

    #[test]
    fn test_coordinator_complete_task() {
        let mut coord = AgentCoordinator::default_coordinator();
        coord.register_agent(create_test_card("http://agent1.local", "Agent1"));

        let task = OrchestratedTask::new("Test", serde_json::json!({}));
        let task_id = coord.submit_task(task);
        coord.process_pending();

        let completed = coord.complete_task(&task_id, true);
        assert!(completed.is_some());
        assert_eq!(coord.stats().tasks_succeeded, 1);
        assert_eq!(coord.active_count(), 0);
    }

    #[test]
    fn test_aggregation_first_success() {
        let config = OrchestrationConfig {
            aggregation_strategy: AggregationStrategy::FirstSuccess,
            ..Default::default()
        };
        let coord = AgentCoordinator::new(config);

        let mut task = OrchestratedTask::new("Test", serde_json::json!({}));
        task.add_result(AgentResult {
            agent_url: "agent1".to_string(),
            status: AgentResultStatus::Error,
            data: None,
            error: Some("error".to_string()),
            response_time_ms: 100,
            confidence: 0.0,
        });
        task.add_result(AgentResult {
            agent_url: "agent2".to_string(),
            status: AgentResultStatus::Success,
            data: Some(serde_json::json!({"answer": 42})),
            error: None,
            response_time_ms: 50,
            confidence: 0.9,
        });

        let result = coord.aggregate_results(&task);
        assert!(result.is_some());
        let agg = result.unwrap();
        assert_eq!(agg.source_agents.len(), 1);
        assert_eq!(agg.source_agents[0], "agent2");
    }

    #[test]
    fn test_aggregation_best_score() {
        let config = OrchestrationConfig {
            aggregation_strategy: AggregationStrategy::BestScore,
            ..Default::default()
        };
        let coord = AgentCoordinator::new(config);

        let mut task = OrchestratedTask::new("Test", serde_json::json!({}));
        task.add_result(AgentResult {
            agent_url: "agent1".to_string(),
            status: AgentResultStatus::Success,
            data: Some(serde_json::json!({"a": 1})),
            error: None,
            response_time_ms: 100,
            confidence: 0.7,
        });
        task.add_result(AgentResult {
            agent_url: "agent2".to_string(),
            status: AgentResultStatus::Success,
            data: Some(serde_json::json!({"b": 2})),
            error: None,
            response_time_ms: 50,
            confidence: 0.95,
        });

        let result = coord.aggregate_results(&task);
        assert!(result.is_some());
        let agg = result.unwrap();
        assert_eq!(agg.confidence, 0.95);
        assert_eq!(agg.source_agents[0], "agent2");
    }

    #[test]
    fn test_selection_round_robin() {
        let config = OrchestrationConfig {
            selection_strategy: AgentSelectionStrategy::RoundRobin,
            max_agents_per_task: 1,
            ..Default::default()
        };
        let mut coord = AgentCoordinator::new(config);

        coord.register_agent(create_test_card("http://agent1.local", "Agent1"));
        coord.register_agent(create_test_card("http://agent2.local", "Agent2"));

        let task1 = OrchestratedTask::new("T1", serde_json::json!({}));
        let agents1 = coord.select_agents(&task1);

        let task2 = OrchestratedTask::new("T2", serde_json::json!({}));
        let agents2 = coord.select_agents(&task2);

        // Should select different agents in round-robin
        assert_ne!(agents1[0], agents2[0]);
    }

    #[test]
    fn test_retry_task() {
        let mut coord = AgentCoordinator::default_coordinator();
        coord.register_agent(create_test_card("http://agent1.local", "Agent1"));

        let task = OrchestratedTask::new("Test", serde_json::json!({}));
        let task_id = coord.submit_task(task);
        coord.process_pending();

        // Retry should work
        let retried = coord.retry_task(&task_id);
        assert!(retried);
        assert_eq!(coord.pending_count(), 1);
        assert_eq!(coord.active_count(), 0);

        // Check retry count increased
        let task = coord.get_task(&task_id).unwrap();
        assert_eq!(task.retry_count, 1);
    }

    #[test]
    fn test_selection_least_loaded() {
        let config = OrchestrationConfig {
            selection_strategy: AgentSelectionStrategy::LeastLoaded,
            max_agents_per_task: 1,
            ..Default::default()
        };
        let mut coord = AgentCoordinator::new(config);

        // Register two agents
        coord.register_agent(create_test_card("http://agent1.local", "Agent1"));
        coord.register_agent(create_test_card("http://agent2.local", "Agent2"));

        // Load agent1
        if let Some(agent) = coord.agents.get_mut("http://agent1.local") {
            agent.task_started();
            agent.task_started();
        }

        let task = OrchestratedTask::new("Test", serde_json::json!({}));
        let agents = coord.select_agents(&task);

        // Should select agent2 (less loaded)
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0], "http://agent2.local");
    }
}
