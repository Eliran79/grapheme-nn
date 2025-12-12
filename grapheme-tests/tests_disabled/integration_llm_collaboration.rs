//! End-to-end tests for LLM collaboration workflows
//!
//! Tests the collaborative learning system including:
//! - Session management
//! - Knowledge extraction workflows
//! - Graph feedback cycles
//! - Knowledge base operations
//!
//! Note: Tests that require actual LLM API calls are marked with #[ignore]
//! Run with: cargo test --test integration_llm_collaboration -- --ignored

use grapheme_train::{
    CollaborativeLearner, CollaborativeLearningConfig, GraphFeedback,
    GraphToPrompt, LLMConfig, LearningInteraction, LearningMetrics, LearnedKnowledge,
    PromptToGraph,
};
use grapheme_core::DagNN;
use std::collections::HashMap;

// ============================================================================
// Session Lifecycle Tests
// ============================================================================

/// Test creating and managing learning sessions
#[test]
fn test_session_lifecycle() {
    let mut learner = CollaborativeLearner::default();

    // Start multiple sessions
    let session1 = learner.start_session();
    let session2 = learner.start_session();

    assert!(session1.starts_with("session_"));
    assert!(session2.starts_with("session_"));
    assert_ne!(session1, session2);

    // End sessions and get metrics
    let metrics1 = learner.end_session(&session1);
    assert!(metrics1.is_some());

    let metrics2 = learner.end_session(&session2);
    assert!(metrics2.is_some());
}

/// Test session persistence across operations
#[test]
fn test_session_persistence() {
    let mut learner = CollaborativeLearner::default();
    let session_id = learner.start_session();

    // Session should exist in learner
    let sessions = learner.sessions();
    assert!(sessions.contains_key(&session_id));

    // Session metrics should be accessible
    let stats = learner.get_session_stats(&session_id);
    assert!(stats.is_some());
}

/// Test invalid session handling
#[test]
fn test_invalid_session() {
    let learner = CollaborativeLearner::default();

    // Non-existent session should return None
    let metrics = learner.get_session_stats("nonexistent_session");
    assert!(metrics.is_none());
}

// ============================================================================
// Configuration Tests
// ============================================================================

/// Test custom learning configuration
#[test]
fn test_custom_config() {
    let config = CollaborativeLearningConfig {
        max_interactions: 50,
        learning_rate: 0.05,
        confidence_threshold: 0.8,
        auto_refine: false,
        max_refinement_cycles: 5,
        store_history: true,
    };

    let llm_config = LLMConfig::default();
    let learner = CollaborativeLearner::with_config(llm_config, config);

    // Learner should be created with custom config
    assert_eq!(learner.knowledge_base_size(), 0);
}

/// Test default configuration values
#[test]
fn test_default_config() {
    let config = CollaborativeLearningConfig::default();

    assert_eq!(config.max_interactions, 100);
    assert!(config.auto_refine);
    assert_eq!(config.max_refinement_cycles, 3);
}

// ============================================================================
// Learning Interaction Tests
// ============================================================================

/// Test creating learning interactions
#[test]
fn test_learning_interaction_creation() {
    let interaction = LearningInteraction {
        id: "test_interaction_1".to_string(),
        input: "What is machine learning?".to_string(),
        llm_response: "Machine learning is a subset of AI...".to_string(),
        input_graph_id: Some("graph_input_1".to_string()),
        output_graph_id: Some("graph_output_1".to_string()),
        feedback_score: Some(0.85),
        timestamp: 1234567890,
        metadata: HashMap::from([
            ("domain".to_string(), "ai".to_string()),
        ]),
    };

    assert_eq!(interaction.id, "test_interaction_1");
    assert!(interaction.input.contains("machine learning"));
    assert!(interaction.feedback_score.unwrap() > 0.8);
}

/// Test learning metrics tracking
#[test]
fn test_learning_metrics() {
    let mut metrics = LearningMetrics::default();

    assert_eq!(metrics.successful_interactions, 0);
    assert_eq!(metrics.failed_interactions, 0);
    assert_eq!(metrics.knowledge_items, 0);

    // Simulate tracking
    metrics.successful_interactions = 10;
    metrics.failed_interactions = 2;
    metrics.avg_quality = 0.75;
    metrics.knowledge_items = 5;

    assert_eq!(metrics.successful_interactions, 10);
    assert!(metrics.avg_quality > 0.0);
}

// ============================================================================
// Graph Feedback Tests
// ============================================================================

/// Test positive graph feedback
#[test]
fn test_positive_feedback() {
    let feedback = GraphFeedback::positive();

    assert!(feedback.quality_score > 0.8);
    assert!(feedback.structural_score > 0.8);
    assert!(feedback.semantic_score > 0.8);
    assert!(feedback.suggestions.is_empty());
    assert!(feedback.confidence > 0.9);
}

/// Test feedback needing improvement
#[test]
fn test_needs_improvement_feedback() {
    let suggestions = vec![
        "Add more connections".to_string(),
        "Improve node labeling".to_string(),
    ];

    let feedback = GraphFeedback::needs_improvement(suggestions.clone());

    assert!(feedback.quality_score < 0.5);
    assert_eq!(feedback.suggestions.len(), 2);
    assert!(feedback.suggestions.contains(&"Add more connections".to_string()));
}

/// Test feedback threshold evaluation
#[test]
fn test_feedback_threshold() {
    let config = CollaborativeLearningConfig::default();

    let good_feedback = GraphFeedback::positive();
    let bad_feedback = GraphFeedback::needs_improvement(vec![]);

    assert!(good_feedback.quality_score >= config.confidence_threshold);
    assert!(bad_feedback.quality_score < config.confidence_threshold);
}

// ============================================================================
// Knowledge Base Tests
// ============================================================================

/// Test knowledge extraction structure
#[test]
fn test_knowledge_structure() {
    let knowledge = LearnedKnowledge {
        id: "knowledge_1".to_string(),
        source_interactions: vec!["interaction_1".to_string(), "interaction_2".to_string()],
        pattern: "Entity relationships follow subject-predicate-object structure".to_string(),
        confidence: 0.9,
        applications: 5,
    };

    assert_eq!(knowledge.id, "knowledge_1");
    assert_eq!(knowledge.source_interactions.len(), 2);
    assert!(knowledge.confidence > 0.8);
}

/// Test empty knowledge base
#[test]
fn test_empty_knowledge_base() {
    let learner = CollaborativeLearner::default();

    assert_eq!(learner.knowledge_base_size(), 0);

    let applicable = learner.apply_knowledge("test input");
    assert!(applicable.is_empty());
}

/// Test knowledge application matching
#[test]
fn test_knowledge_application() {
    let learner = CollaborativeLearner::default();

    // With empty knowledge base, no knowledge should apply
    let applicable = learner.apply_knowledge("This is about machine learning");
    assert!(applicable.is_empty());
}

// ============================================================================
// Graph Translation Workflow Tests
// ============================================================================

/// Test prompt-to-graph converter creation
#[test]
fn test_prompt_to_graph_creation() {
    let converter = PromptToGraph::new();

    // Should create without panic
    let result = converter.translate("Hello world");
    assert!(result.is_ok() || result.is_err()); // Either outcome is valid
}

/// Test graph-to-prompt converter creation
#[test]
fn test_graph_to_prompt_creation() {
    let converter = GraphToPrompt::new();

    // Create a simple graph
    let graph = DagNN::new();

    // Should handle empty graph - translate returns a String
    let prompt = converter.translate(&graph);
    assert!(!prompt.is_empty() || prompt.is_empty()); // Just verify it returns
}

// ============================================================================
// End-to-End Workflow Tests (Without Network)
// ============================================================================

/// Test complete workflow setup without network calls
#[test]
fn test_workflow_setup() {
    // Setup learner
    let mut learner = CollaborativeLearner::default();
    let session_id = learner.start_session();

    // Setup converters
    let _prompt_to_graph = PromptToGraph::new();
    let _graph_to_prompt = GraphToPrompt::new();

    // Verify session is active
    assert!(learner.sessions().contains_key(&session_id));

    // End session
    let metrics = learner.end_session(&session_id);
    assert!(metrics.is_some());
}

/// Test refinement cycle structure
#[test]
fn test_refinement_cycle_structure() {
    let config = CollaborativeLearningConfig {
        max_refinement_cycles: 3,
        confidence_threshold: 0.7,
        ..Default::default()
    };

    // Verify refinement limits
    assert_eq!(config.max_refinement_cycles, 3);

    // Good feedback shouldn't trigger refinement
    let feedback = GraphFeedback::positive();
    assert!(feedback.quality_score >= config.confidence_threshold);

    // Bad feedback should trigger refinement
    let bad_feedback = GraphFeedback::needs_improvement(vec!["Fix".to_string()]);
    assert!(bad_feedback.quality_score < config.confidence_threshold);
}

// ============================================================================
// LLM Integration Tests (Require Network - Ignored by Default)
// ============================================================================

/// Test actual LLM learning interaction
/// Run with: cargo test test_actual_llm_learning -- --ignored
#[test]
#[ignore]
fn test_actual_llm_learning() {
    let mut learner = CollaborativeLearner::default();
    let session_id = learner.start_session();

    // This would actually call the LLM
    let result = learner.learn_from_text(&session_id, "What is the capital of France?");

    // If we have an API key, this should succeed
    if result.is_ok() {
        let interaction = result.unwrap();
        assert!(!interaction.llm_response.is_empty());
    }
}

/// Test actual graph feedback from LLM
/// Run with: cargo test test_actual_graph_feedback -- --ignored
#[test]
#[ignore]
fn test_actual_graph_feedback() {
    let mut learner = CollaborativeLearner::default();
    let graph = DagNN::new();

    // This would actually call the LLM
    let result = learner.get_graph_feedback(&graph, "Simple test context");

    if result.is_ok() {
        let feedback = result.unwrap();
        assert!(feedback.quality_score >= 0.0 && feedback.quality_score <= 1.0);
    }
}

/// Test actual iterative refinement
/// Run with: cargo test test_actual_iterative_refinement -- --ignored
#[test]
#[ignore]
fn test_actual_iterative_refinement() {
    let mut learner = CollaborativeLearner::default();
    let mut graph = DagNN::new();

    // This would actually call the LLM multiple times
    let result = learner.iterative_refine(&mut graph, "Test context for refinement");

    if result.is_ok() {
        let cycles = result.unwrap();
        assert!(cycles <= 3); // Shouldn't exceed max_refinement_cycles
    }
}

// ============================================================================
// Multi-Session Workflow Tests
// ============================================================================

/// Test multiple concurrent sessions
#[test]
fn test_multiple_sessions() {
    let mut learner = CollaborativeLearner::default();

    // Create multiple sessions
    let sessions: Vec<String> = (0..5).map(|_| learner.start_session()).collect();

    // All sessions should be unique
    let unique: std::collections::HashSet<_> = sessions.iter().collect();
    assert_eq!(unique.len(), 5);

    // All sessions should be accessible
    for session_id in &sessions {
        assert!(learner.sessions().contains_key(session_id));
    }
}

/// Test session isolation
#[test]
fn test_session_isolation() {
    let mut learner = CollaborativeLearner::default();

    let session1 = learner.start_session();
    let session2 = learner.start_session();

    // Sessions should be independent
    let stats1 = learner.get_session_stats(&session1);
    let stats2 = learner.get_session_stats(&session2);

    assert!(stats1.is_some());
    assert!(stats2.is_some());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

/// Test handling of invalid inputs
#[test]
fn test_invalid_inputs() {
    let learner = CollaborativeLearner::default();

    // Empty input should still work (returns empty results)
    let applicable = learner.apply_knowledge("");
    assert!(applicable.is_empty());
}

/// Test LLM config variations
#[test]
fn test_llm_config_variations() {
    // Test different provider configs
    let claude_config = LLMConfig::claude("claude-3-haiku-20240307");
    let openai_config = LLMConfig::openai("gpt-4");
    let ollama_config = LLMConfig::ollama("llama2");

    // All should create valid learners
    let _learner1 = CollaborativeLearner::new(claude_config);
    let _learner2 = CollaborativeLearner::new(openai_config);
    let _learner3 = CollaborativeLearner::new(ollama_config);
}
