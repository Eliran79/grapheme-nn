//! Collaborative Learning Module
//!
//! Backend-178: Collaborative learning from LLM interactions.
//!
//! Enables GRAPHEME to learn from LLM interactions by:
//! - Converting LLM completions into graph structures
//! - Using LLM feedback to refine graph generation
//! - Building a knowledge base from LLM interactions
//! - Supporting iterative improvement cycles

use crate::llm_client::{CompletionRequest, LLMClient, LLMConfig};
use grapheme_core::DagNN;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for collaborative learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeLearningConfig {
    /// Maximum interactions per session
    pub max_interactions: usize,
    /// Learning rate for feedback integration
    pub learning_rate: f32,
    /// Minimum confidence threshold for accepting LLM feedback
    pub confidence_threshold: f32,
    /// Enable automatic refinement cycles
    pub auto_refine: bool,
    /// Maximum refinement iterations
    pub max_refinement_cycles: u32,
    /// Store interaction history
    pub store_history: bool,
}

impl Default for CollaborativeLearningConfig {
    fn default() -> Self {
        Self {
            max_interactions: 100,
            learning_rate: 0.01,
            confidence_threshold: 0.7,
            auto_refine: true,
            max_refinement_cycles: 3,
            store_history: true,
        }
    }
}

/// A learning interaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningInteraction {
    /// Unique interaction ID
    pub id: String,
    /// Input text or prompt
    pub input: String,
    /// LLM response
    pub llm_response: String,
    /// Graph representation of the input
    pub input_graph_id: Option<String>,
    /// Graph representation of the output
    pub output_graph_id: Option<String>,
    /// Feedback score (0.0 - 1.0)
    pub feedback_score: Option<f32>,
    /// Timestamp
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Feedback from evaluating graph quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphFeedback {
    /// Overall quality score (0.0 - 1.0)
    pub quality_score: f32,
    /// Structural coherence score
    pub structural_score: f32,
    /// Semantic accuracy score
    pub semantic_score: f32,
    /// Suggested improvements
    pub suggestions: Vec<String>,
    /// Confidence in the feedback
    pub confidence: f32,
}

impl GraphFeedback {
    /// Create positive feedback
    pub fn positive() -> Self {
        Self {
            quality_score: 0.9,
            structural_score: 0.9,
            semantic_score: 0.9,
            suggestions: vec![],
            confidence: 0.95,
        }
    }

    /// Create negative feedback with suggestions
    pub fn needs_improvement(suggestions: Vec<String>) -> Self {
        Self {
            quality_score: 0.4,
            structural_score: 0.5,
            semantic_score: 0.4,
            suggestions,
            confidence: 0.8,
        }
    }
}

/// Learning session state
#[derive(Debug, Clone)]
pub struct LearningSession {
    /// Session ID
    pub id: String,
    /// Interaction history
    pub interactions: Vec<LearningInteraction>,
    /// Total interactions in this session
    pub total_interactions: usize,
    /// Average feedback score
    pub avg_feedback_score: f32,
    /// Learning progress metrics
    pub metrics: LearningMetrics,
}

/// Metrics tracking learning progress
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningMetrics {
    /// Total successful interactions
    pub successful_interactions: usize,
    /// Total failed interactions
    pub failed_interactions: usize,
    /// Average response quality
    pub avg_quality: f32,
    /// Improvement rate over time
    pub improvement_rate: f32,
    /// Knowledge items learned
    pub knowledge_items: usize,
}

/// Collaborative learning engine
pub struct CollaborativeLearner {
    config: CollaborativeLearningConfig,
    llm_client: LLMClient,
    sessions: HashMap<String, LearningSession>,
    knowledge_base: Vec<LearnedKnowledge>,
    interaction_count: usize,
}

/// Knowledge item learned from interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedKnowledge {
    /// Knowledge ID
    pub id: String,
    /// Source interaction IDs
    pub source_interactions: Vec<String>,
    /// Pattern or rule learned
    pub pattern: String,
    /// Confidence in this knowledge
    pub confidence: f32,
    /// Application count
    pub applications: usize,
}

impl CollaborativeLearner {
    /// Create a new collaborative learner
    pub fn new(llm_config: LLMConfig) -> Self {
        Self {
            config: CollaborativeLearningConfig::default(),
            llm_client: LLMClient::new(llm_config),
            sessions: HashMap::new(),
            knowledge_base: Vec::new(),
            interaction_count: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(llm_config: LLMConfig, learning_config: CollaborativeLearningConfig) -> Self {
        Self {
            config: learning_config,
            llm_client: LLMClient::new(llm_config),
            sessions: HashMap::new(),
            knowledge_base: Vec::new(),
            interaction_count: 0,
        }
    }

    /// Start a new learning session
    pub fn start_session(&mut self) -> String {
        let session_id = format!("session_{}", self.sessions.len());
        let session = LearningSession {
            id: session_id.clone(),
            interactions: Vec::new(),
            total_interactions: 0,
            avg_feedback_score: 0.0,
            metrics: LearningMetrics::default(),
        };
        self.sessions.insert(session_id.clone(), session);
        session_id
    }

    /// End a learning session and compute final metrics
    pub fn end_session(&mut self, session_id: &str) -> Option<LearningMetrics> {
        self.sessions.get(session_id).map(|s| s.metrics.clone())
    }

    /// Learn from text by interacting with LLM
    pub fn learn_from_text(&mut self, session_id: &str, text: &str) -> Result<LearningInteraction, String> {
        self.interaction_count += 1;
        let interaction_id = format!("interaction_{}", self.interaction_count);

        // Build prompt for knowledge extraction
        let prompt = format!(
            "Analyze the following text and extract key concepts, relationships, and patterns:\n\n{}\n\n\
             Provide a structured analysis with:\n\
             1. Main concepts/entities\n\
             2. Relationships between concepts\n\
             3. Key patterns or rules\n\
             4. Suggested graph structure",
            text
        );

        // Request completion from LLM
        let mut request = CompletionRequest::new(&prompt);
        request.max_tokens = Some(2048);
        request.temperature = Some(0.3);

        let response = self.llm_client.complete(request)
            .map_err(|e| format!("LLM error: {:?}", e))?;

        let llm_response = response.content;

        // Create interaction record
        let interaction = LearningInteraction {
            id: interaction_id.clone(),
            input: text.to_string(),
            llm_response,
            input_graph_id: None,
            output_graph_id: None,
            feedback_score: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            metadata: HashMap::new(),
        };

        // Store in session
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.interactions.push(interaction.clone());
            session.total_interactions += 1;
            session.metrics.successful_interactions += 1;
        }

        Ok(interaction)
    }

    /// Get feedback on a graph from LLM
    pub fn get_graph_feedback(&mut self, graph: &DagNN, context: &str) -> Result<GraphFeedback, String> {
        // Build prompt for graph evaluation
        let prompt = format!(
            "Evaluate the quality of a graph representation for the following context:\n\n\
             Context: {}\n\n\
             Graph Statistics:\n\
             - Node count: {}\n\n\
             Rate the following aspects (0.0 to 1.0):\n\
             1. Structural coherence\n\
             2. Semantic accuracy\n\
             3. Overall quality\n\n\
             Also provide specific suggestions for improvement.",
            context,
            graph.node_count()
        );

        let mut request = CompletionRequest::new(&prompt);
        request.max_tokens = Some(1024);
        request.temperature = Some(0.2);

        let response = self.llm_client.complete(request)
            .map_err(|e| format!("LLM error: {:?}", e))?;

        // Parse feedback from response
        let feedback = self.parse_feedback(&response.content);
        Ok(feedback)
    }

    /// Refine a graph based on LLM feedback
    pub fn refine_graph(&mut self, graph: &mut DagNN, feedback: &GraphFeedback) -> Result<bool, String> {
        if feedback.quality_score >= self.config.confidence_threshold {
            return Ok(false); // No refinement needed
        }

        // Apply simple refinements based on suggestions
        for suggestion in &feedback.suggestions {
            self.apply_refinement(graph, suggestion)?;
        }

        Ok(true)
    }

    /// Run iterative refinement cycle
    pub fn iterative_refine(&mut self, graph: &mut DagNN, context: &str) -> Result<u32, String> {
        let mut cycles = 0;

        for _ in 0..self.config.max_refinement_cycles {
            let feedback = self.get_graph_feedback(graph, context)?;

            if feedback.quality_score >= self.config.confidence_threshold {
                break;
            }

            if self.refine_graph(graph, &feedback)? {
                cycles += 1;
            } else {
                break;
            }
        }

        Ok(cycles)
    }

    /// Extract knowledge from successful interactions
    pub fn extract_knowledge(&mut self, session_id: &str) -> Vec<LearnedKnowledge> {
        let mut knowledge = Vec::new();

        if let Some(session) = self.sessions.get(session_id) {
            // Find patterns in successful interactions
            for interaction in &session.interactions {
                if interaction.feedback_score.unwrap_or(0.5) > self.config.confidence_threshold {
                    let item = LearnedKnowledge {
                        id: format!("knowledge_{}", self.knowledge_base.len()),
                        source_interactions: vec![interaction.id.clone()],
                        pattern: self.extract_pattern(&interaction.llm_response),
                        confidence: interaction.feedback_score.unwrap_or(0.5),
                        applications: 0,
                    };
                    knowledge.push(item);
                }
            }
        }

        // Store in knowledge base
        for item in &knowledge {
            self.knowledge_base.push(item.clone());
        }

        knowledge
    }

    /// Apply learned knowledge to new input
    pub fn apply_knowledge(&self, input: &str) -> Vec<&LearnedKnowledge> {
        let input_lower = input.to_lowercase();
        self.knowledge_base
            .iter()
            .filter(|k| {
                // Simple pattern matching
                k.confidence > self.config.confidence_threshold
                    && k.pattern.to_lowercase().split_whitespace()
                        .any(|word| input_lower.contains(word))
            })
            .collect()
    }

    /// Get session statistics
    pub fn get_session_stats(&self, session_id: &str) -> Option<&LearningMetrics> {
        self.sessions.get(session_id).map(|s| &s.metrics)
    }

    /// Get total knowledge base size
    pub fn knowledge_base_size(&self) -> usize {
        self.knowledge_base.len()
    }

    /// Get all sessions
    pub fn sessions(&self) -> &HashMap<String, LearningSession> {
        &self.sessions
    }

    // Helper: Parse feedback from LLM response
    fn parse_feedback(&self, response: &str) -> GraphFeedback {
        let response_lower = response.to_lowercase();

        // Extract scores (simple heuristic parsing)
        let quality_score = self.extract_score(&response_lower, "quality");
        let structural_score = self.extract_score(&response_lower, "structural");
        let semantic_score = self.extract_score(&response_lower, "semantic");

        // Extract suggestions
        let suggestions: Vec<String> = response
            .lines()
            .filter(|line| {
                let l = line.to_lowercase();
                l.contains("suggest") || l.contains("improve") || l.contains("recommend")
            })
            .map(|s| s.trim().to_string())
            .collect();

        GraphFeedback {
            quality_score: quality_score.unwrap_or(0.5),
            structural_score: structural_score.unwrap_or(0.5),
            semantic_score: semantic_score.unwrap_or(0.5),
            suggestions,
            confidence: 0.7,
        }
    }

    // Helper: Extract numeric score from text
    fn extract_score(&self, text: &str, keyword: &str) -> Option<f32> {
        if let Some(pos) = text.find(keyword) {
            let after = &text[pos..];
            // Look for a number (0.X format)
            for word in after.split_whitespace().take(10) {
                if let Ok(score) = word.trim_matches(|c: char| !c.is_numeric() && c != '.').parse::<f32>() {
                    if score >= 0.0 && score <= 1.0 {
                        return Some(score);
                    }
                }
            }
        }
        None
    }

    // Helper: Apply a refinement suggestion
    fn apply_refinement(&self, _graph: &mut DagNN, _suggestion: &str) -> Result<(), String> {
        // Placeholder for graph refinement logic
        // In a full implementation, this would modify the graph based on the suggestion
        Ok(())
    }

    // Helper: Extract pattern from LLM response
    fn extract_pattern(&self, response: &str) -> String {
        // Extract first meaningful line as pattern
        response
            .lines()
            .find(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && trimmed.len() > 10
            })
            .unwrap_or("unknown pattern")
            .to_string()
    }
}

impl Default for CollaborativeLearner {
    fn default() -> Self {
        Self::new(LLMConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = CollaborativeLearningConfig::default();
        assert_eq!(config.max_interactions, 100);
        assert!(config.auto_refine);
    }

    #[test]
    fn test_start_session() {
        let mut learner = CollaborativeLearner::default();
        let session_id = learner.start_session();
        assert!(session_id.starts_with("session_"));
        assert!(learner.sessions.contains_key(&session_id));
    }

    #[test]
    fn test_end_session() {
        let mut learner = CollaborativeLearner::default();
        let session_id = learner.start_session();
        let metrics = learner.end_session(&session_id);
        assert!(metrics.is_some());
    }

    #[test]
    fn test_feedback_positive() {
        let feedback = GraphFeedback::positive();
        assert!(feedback.quality_score > 0.8);
        assert!(feedback.suggestions.is_empty());
    }

    #[test]
    fn test_feedback_needs_improvement() {
        let suggestions = vec!["Add more nodes".to_string()];
        let feedback = GraphFeedback::needs_improvement(suggestions);
        assert!(feedback.quality_score < 0.5);
        assert!(!feedback.suggestions.is_empty());
    }

    #[test]
    fn test_knowledge_base_size() {
        let learner = CollaborativeLearner::default();
        assert_eq!(learner.knowledge_base_size(), 0);
    }

    #[test]
    fn test_apply_knowledge_empty() {
        let learner = CollaborativeLearner::default();
        let applicable = learner.apply_knowledge("test input");
        assert!(applicable.is_empty());
    }

    #[test]
    fn test_learning_interaction_creation() {
        let interaction = LearningInteraction {
            id: "test_1".to_string(),
            input: "Hello".to_string(),
            llm_response: "World".to_string(),
            input_graph_id: None,
            output_graph_id: None,
            feedback_score: Some(0.8),
            timestamp: 0,
            metadata: HashMap::new(),
        };
        assert_eq!(interaction.id, "test_1");
        assert_eq!(interaction.feedback_score, Some(0.8));
    }

    #[test]
    fn test_learning_metrics_default() {
        let metrics = LearningMetrics::default();
        assert_eq!(metrics.successful_interactions, 0);
        assert_eq!(metrics.failed_interactions, 0);
    }

    #[test]
    fn test_extract_score() {
        let learner = CollaborativeLearner::default();
        let text = "quality: 0.8, structural: 0.7";
        assert!(learner.extract_score(text, "quality").is_some());
    }
}
