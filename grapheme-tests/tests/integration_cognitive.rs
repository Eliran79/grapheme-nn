//! Integration tests for cognitive module training (testing-005)
//!
//! Tests end-to-end training of learnable cognitive modules:
//! - LearnableMemoryRetrieval
//! - LearnableReasoning
//! - LearnableWorldModel
//! - LearnableMetaCognition
//! - LearnableAgency
//! - LearnableMultimodal
//! - LearnableGrounding

use grapheme_core::{Learnable, LearnableParam};

// ============================================================================
// LearnableParam Tests
// ============================================================================

#[test]
fn test_learnable_param_creation() {
    let param = LearnableParam::new(0.5);
    assert!((param.value - 0.5).abs() < 1e-6);
    assert_eq!(param.grad, 0.0);
}

#[test]
fn test_learnable_param_gradient_update() {
    let mut param = LearnableParam::new(1.0);
    param.accumulate_grad(0.5);
    assert!((param.grad - 0.5).abs() < 1e-6);

    param.step(0.1);
    assert!((param.value - 0.95).abs() < 1e-6); // 1.0 - 0.1 * 0.5
}

#[test]
fn test_learnable_param_zero_grad() {
    let mut param = LearnableParam::new(1.0);
    param.accumulate_grad(0.5);
    param.zero_grad();
    assert_eq!(param.grad, 0.0);
}

// ============================================================================
// Memory Module Tests
// ============================================================================

#[test]
fn test_learnable_memory_creation() {
    use grapheme_memory::LearnableMemoryRetrieval;

    let memory = LearnableMemoryRetrieval::new();
    assert_eq!(memory.num_parameters(), 6);
    assert!(!memory.has_gradients());
}

#[test]
fn test_learnable_memory_training_step() {
    use grapheme_memory::LearnableMemoryRetrieval;

    let mut memory = LearnableMemoryRetrieval::new();
    let initial_node_weight = memory.node_weight.value;

    // Simulate gradient
    memory.node_weight.grad = 0.5;
    memory.step(0.1);

    assert!((memory.node_weight.value - (initial_node_weight - 0.05)).abs() < 1e-6);
}

#[test]
fn test_learnable_memory_weighted_similarity() {
    use grapheme_core::DagNN;
    use grapheme_memory::{GraphFingerprint, LearnableMemoryRetrieval};

    let memory = LearnableMemoryRetrieval::new();

    let g1 = DagNN::from_text("hello").unwrap();
    let g2 = DagNN::from_text("world").unwrap();

    let fp1 = GraphFingerprint::from_graph(&g1);
    let fp2 = GraphFingerprint::from_graph(&g2);

    let sim = memory.weighted_similarity(&fp1, &fp2);
    assert!(sim >= 0.0 && sim <= 2.0); // similarity + bias can exceed 1.0
}

// ============================================================================
// Reasoning Module Tests
// ============================================================================

#[test]
fn test_learnable_reasoning_creation() {
    use grapheme_reason::LearnableReasoning;

    let reasoning = LearnableReasoning::new();
    assert_eq!(reasoning.num_parameters(), 6);
    assert!(!reasoning.has_gradients());
}

#[test]
fn test_learnable_reasoning_mode_selection() {
    use grapheme_reason::{LearnableReasoning, ReasoningMode};

    let reasoning = LearnableReasoning::new();

    // High structural clarity should favor deduction
    let mode = reasoning.select_mode(&[1.0, 0.1, 0.1]);
    assert_eq!(mode, ReasoningMode::Deduction);

    // Many examples should favor induction
    let mode = reasoning.select_mode(&[0.1, 1.0, 0.1]);
    assert_eq!(mode, ReasoningMode::Induction);

    // Explanation needed should favor abduction
    let mode = reasoning.select_mode(&[0.1, 0.1, 1.0]);
    assert_eq!(mode, ReasoningMode::Abduction);
}

#[test]
fn test_learnable_reasoning_confidence_update() {
    use grapheme_reason::{LearnableReasoning, ReasoningMode};

    let mut reasoning = LearnableReasoning::new();
    let initial = reasoning.deduction_confidence.value;

    reasoning.update_confidence(ReasoningMode::Deduction, true, 0.1);

    // Success should increase confidence
    assert!(reasoning.deduction_confidence.value > initial);
}

// ============================================================================
// World Model Tests
// ============================================================================

#[test]
fn test_learnable_world_model_creation() {
    use grapheme_world::LearnableWorldModel;

    let world = LearnableWorldModel::new();
    assert_eq!(world.num_parameters(), 6);
    assert!(!world.has_gradients());
}

#[test]
fn test_learnable_world_model_prediction() {
    use grapheme_world::LearnableWorldModel;

    let world = LearnableWorldModel::new();

    // Test transition probability
    let prob = world.transition_probability(0.5);
    assert!((prob - 0.5).abs() < 1e-6);

    // Test decayed confidence
    let conf0 = world.decayed_confidence(0);
    let conf5 = world.decayed_confidence(5);
    assert!(conf0 > conf5); // Confidence should decay over time
}

#[test]
fn test_learnable_world_model_state_change() {
    use grapheme_world::LearnableWorldModel;

    let world = LearnableWorldModel::new();

    let score = world.state_change_score(0.8, 0.6);
    assert!(score >= 0.0 && score <= 1.0);
}

// ============================================================================
// Meta-Cognition Tests
// ============================================================================

#[test]
fn test_learnable_meta_cognition_creation() {
    use grapheme_meta::LearnableMetaCognition;

    let meta = LearnableMetaCognition::new();
    assert_eq!(meta.num_parameters(), 5);
    assert!(!meta.has_gradients());
}

#[test]
fn test_learnable_meta_cognition_calibration() {
    use grapheme_meta::LearnableMetaCognition;

    let meta = LearnableMetaCognition::new();

    let calibrated = meta.calibrate_confidence(0.7);
    assert!(calibrated >= 0.0 && calibrated <= 1.0);
}

#[test]
fn test_learnable_meta_cognition_early_stop() {
    use grapheme_meta::LearnableMetaCognition;

    let meta = LearnableMetaCognition::new();

    // Default threshold is 0.95
    assert!(!meta.should_early_stop(0.9));
    assert!(meta.should_early_stop(0.99));
}

#[test]
fn test_learnable_meta_cognition_compute_allocation() {
    use grapheme_meta::LearnableMetaCognition;

    let meta = LearnableMetaCognition::new();

    let adjusted = meta.adjusted_compute(100);
    assert!(adjusted > 0);
}

// ============================================================================
// Agency Tests
// ============================================================================

#[test]
fn test_learnable_agency_creation() {
    use grapheme_agent::LearnableAgency;

    let agency = LearnableAgency::new();
    assert_eq!(agency.num_parameters(), 6);
    assert!(!agency.has_gradients());
}

#[test]
fn test_learnable_agency_drive_scoring() {
    use grapheme_agent::LearnableAgency;

    let agency = LearnableAgency::new();

    let score = agency.drive_score(0.5, 0.8, 0.6);
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn test_learnable_agency_exploration() {
    use grapheme_agent::LearnableAgency;

    let agency = LearnableAgency::new();

    let low_uncertainty = agency.exploration_probability(0.1);
    let high_uncertainty = agency.exploration_probability(0.9);

    assert!(high_uncertainty > low_uncertainty);
}

#[test]
fn test_learnable_agency_discounting() {
    use grapheme_agent::LearnableAgency;

    let agency = LearnableAgency::new();

    let now = agency.discounted_value(1.0, 0);
    let later = agency.discounted_value(1.0, 10);

    assert!(now > later);
}

// ============================================================================
// Multimodal Tests
// ============================================================================

#[test]
fn test_learnable_multimodal_creation() {
    use grapheme_multimodal::LearnableMultimodal;

    let mm = LearnableMultimodal::new();
    assert_eq!(mm.num_parameters(), 6);
    assert!(!mm.has_gradients());
}

#[test]
fn test_learnable_multimodal_weights() {
    use grapheme_multimodal::LearnableMultimodal;

    let mm = LearnableMultimodal::new();

    let weights = mm.normalized_weights();
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_learnable_multimodal_fusion() {
    use grapheme_multimodal::LearnableMultimodal;

    let mm = LearnableMultimodal::new();

    let fused = mm.weighted_fusion(0.8, 0.6, 0.9, 0.4);
    assert!(fused >= 0.0 && fused <= 1.0);
}

#[test]
fn test_learnable_multimodal_attention() {
    use grapheme_multimodal::LearnableMultimodal;

    let mm = LearnableMultimodal::new();

    let scores = vec![0.5, 0.8, 0.3];
    let weights = mm.attention_weights(&scores);

    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Highest score should get highest weight
    assert!(weights[1] > weights[0]);
    assert!(weights[1] > weights[2]);
}

// ============================================================================
// Grounding Tests
// ============================================================================

#[test]
fn test_learnable_grounding_creation() {
    use grapheme_ground::LearnableGrounding;

    let grounding = LearnableGrounding::new();
    assert_eq!(grounding.num_parameters(), 5);
    assert!(!grounding.has_gradients());
}

#[test]
fn test_learnable_grounding_threshold() {
    use grapheme_ground::LearnableGrounding;

    let grounding = LearnableGrounding::new();

    assert!(!grounding.is_grounded(0.4));
    assert!(grounding.is_grounded(0.6));
}

#[test]
fn test_learnable_grounding_score() {
    use grapheme_ground::LearnableGrounding;

    let grounding = LearnableGrounding::new();

    let score = grounding.grounding_score(0.8, 0.6);
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn test_learnable_grounding_cooccurrence() {
    use grapheme_ground::LearnableGrounding;

    let grounding = LearnableGrounding::new();

    let updated = grounding.update_cooccurrence(0.5, true);
    assert!(updated > 0.5); // Should increase when observed

    let updated = grounding.update_cooccurrence(0.5, false);
    assert!(updated < 0.5); // Should decrease when not observed
}

// ============================================================================
// End-to-End Training Loop Tests
// ============================================================================

#[test]
fn test_all_modules_trainable() {
    use grapheme_memory::LearnableMemoryRetrieval;
    use grapheme_reason::LearnableReasoning;
    use grapheme_world::LearnableWorldModel;
    use grapheme_meta::LearnableMetaCognition;
    use grapheme_agent::LearnableAgency;
    use grapheme_multimodal::LearnableMultimodal;
    use grapheme_ground::LearnableGrounding;

    // Create all modules
    let mut memory = LearnableMemoryRetrieval::new();
    let mut reasoning = LearnableReasoning::new();
    let mut world = LearnableWorldModel::new();
    let mut meta = LearnableMetaCognition::new();
    let mut agency = LearnableAgency::new();
    let mut multimodal = LearnableMultimodal::new();
    let mut grounding = LearnableGrounding::new();

    // Simulate training step
    let lr = 0.01;

    // Set mock gradients
    memory.node_weight.grad = 0.1;
    reasoning.deduction_confidence.grad = 0.2;
    world.transition_bias.grad = 0.1;
    meta.calibration_bias.grad = 0.15;
    agency.goal_importance_bias.grad = 0.1;
    multimodal.visual_weight.grad = 0.1;
    grounding.grounding_threshold.grad = 0.05;

    // Take step
    memory.step(lr);
    reasoning.step(lr);
    world.step(lr);
    meta.step(lr);
    agency.step(lr);
    multimodal.step(lr);
    grounding.step(lr);

    // Zero gradients
    memory.zero_grad();
    reasoning.zero_grad();
    world.zero_grad();
    meta.zero_grad();
    agency.zero_grad();
    multimodal.zero_grad();
    grounding.zero_grad();

    // Verify gradients are zeroed
    assert!(!memory.has_gradients());
    assert!(!reasoning.has_gradients());
    assert!(!world.has_gradients());
    assert!(!meta.has_gradients());
    assert!(!agency.has_gradients());
    assert!(!multimodal.has_gradients());
    assert!(!grounding.has_gradients());
}

#[test]
fn test_gradient_norm_computation() {
    use grapheme_memory::LearnableMemoryRetrieval;

    let mut memory = LearnableMemoryRetrieval::new();

    // No gradients initially
    assert!((memory.gradient_norm() - 0.0).abs() < 1e-6);

    // Set gradients
    memory.node_weight.grad = 3.0;
    memory.edge_weight.grad = 4.0;

    // L2 norm should be sqrt(9 + 16) = 5
    let norm = memory.gradient_norm();
    assert!((norm - 5.0).abs() < 1e-6);
}

#[test]
fn test_training_convergence() {
    use grapheme_agent::LearnableAgency;

    let mut agency = LearnableAgency::new();
    let lr = 0.1;

    // Simulate 100 training steps with gradient toward target
    let target_importance = 0.7;

    for _ in 0..100 {
        let error = agency.goal_importance_bias.value - target_importance;
        agency.goal_importance_bias.grad = error;
        agency.step(lr);
        agency.zero_grad();
    }

    // Should converge close to target
    assert!((agency.goal_importance_bias.value - target_importance).abs() < 0.1);
}

#[test]
fn test_parameter_count() {
    use grapheme_memory::LearnableMemoryRetrieval;
    use grapheme_reason::LearnableReasoning;
    use grapheme_world::LearnableWorldModel;
    use grapheme_meta::LearnableMetaCognition;
    use grapheme_agent::LearnableAgency;
    use grapheme_multimodal::LearnableMultimodal;
    use grapheme_ground::LearnableGrounding;

    let total_params =
        LearnableMemoryRetrieval::new().num_parameters() +
        LearnableReasoning::new().num_parameters() +
        LearnableWorldModel::new().num_parameters() +
        LearnableMetaCognition::new().num_parameters() +
        LearnableAgency::new().num_parameters() +
        LearnableMultimodal::new().num_parameters() +
        LearnableGrounding::new().num_parameters();

    // 6 + 6 + 6 + 5 + 6 + 6 + 5 = 40 parameters total
    assert_eq!(total_params, 40);
}

// ============================================================================
// Cognitive-Brain Interaction Tests (testing-006)
// ============================================================================

#[test]
fn test_cognitive_brain_bridge_creation() {
    use grapheme_core::{DefaultCognitiveBridge, CognitiveBrainBridge};

    let bridge = DefaultCognitiveBridge::new();
    assert!(bridge.available_domains().is_empty());
    assert!(!bridge.has_domain("math"));
}

#[test]
fn test_brain_registry_operations() {
    use grapheme_core::BrainRegistry;

    let registry = BrainRegistry::new();
    assert!(registry.domains().is_empty());
    assert!(registry.get("nonexistent").is_none());
}

#[test]
fn test_orchestrator_creation() {
    use grapheme_core::{create_cognitive_orchestrator, CognitiveBrainBridge};

    let orchestrator = create_cognitive_orchestrator();
    assert!(orchestrator.available_domains().is_empty());
}

#[test]
fn test_orchestrator_config() {
    use grapheme_core::{CognitiveBrainOrchestrator, OrchestratorConfig};

    let config = OrchestratorConfig {
        confidence_threshold: 0.7,
        auto_route: false,
        max_brains_per_query: 5,
    };

    let orchestrator = CognitiveBrainOrchestrator::with_config(config.clone());
    assert!((orchestrator.config.confidence_threshold - 0.7).abs() < 1e-6);
    assert!(!orchestrator.config.auto_route);
    assert_eq!(orchestrator.config.max_brains_per_query, 5);
}

#[test]
fn test_orchestrator_stats() {
    use grapheme_core::OrchestratorStats;

    let mut stats = OrchestratorStats::default();
    assert_eq!(stats.total_queries, 0);
    assert_eq!(stats.success_rate(), 0.0);

    stats.record_routing("math");
    stats.record_routing("code");
    stats.record_no_routing();

    assert_eq!(stats.total_queries, 3);
    assert_eq!(stats.routed_queries, 2);
    assert_eq!(stats.unrouted_queries, 1);
    assert!((stats.success_rate() - 0.666).abs() < 0.01);
    assert_eq!(*stats.domain_counts.get("math").unwrap(), 1);
    assert_eq!(*stats.domain_counts.get("code").unwrap(), 1);
}

#[test]
fn test_orchestrated_result() {
    use grapheme_core::OrchestratedResult;

    let result = OrchestratedResult::empty();
    assert!(!result.success());
    assert!(result.primary.is_none());
    assert!(result.domains.is_empty());
    assert!((result.confidence - 0.0).abs() < 1e-6);
}

#[test]
fn test_multi_brain_result() {
    use grapheme_core::{MultiBrainResult, BrainRoutingResult, DagNN};

    let mut result = MultiBrainResult::new();
    assert!(!result.success);
    assert!(result.primary.is_none());
    assert!(result.domains().is_empty());

    // Add a routing result
    result.add_result(BrainRoutingResult {
        domain_id: "test".to_string(),
        graph: DagNN::new(),
        confidence: 0.8,
        result: Some("test result".to_string()),
    });

    assert!(result.success);
    assert!(result.primary.is_some());
    assert_eq!(result.domains(), vec!["test"]);
}

#[test]
fn test_brain_aware_reasoning_creation() {
    use grapheme_reason::{create_brain_aware_reasoning, BrainAwareReasoning};
    use grapheme_core::CognitiveBrainBridge;

    let reasoning = create_brain_aware_reasoning();
    assert!(reasoning.available_domains().is_empty());
    assert!(reasoning.consult_brains_first);
}

#[test]
fn test_brain_aware_memory_creation() {
    use grapheme_memory::{create_domain_aware_memory, DomainAwareMemory};
    use grapheme_core::CognitiveBrainBridge;

    let memory = create_domain_aware_memory();
    assert!(memory.available_domains().is_empty());
    assert!(memory.auto_classify);
}

#[test]
fn test_brain_aware_metacognition_creation() {
    use grapheme_meta::{create_brain_aware_metacognition, BrainAwareMetaCognition};
    use grapheme_core::CognitiveBrainBridge;

    let meta = create_brain_aware_metacognition();
    assert!(meta.available_domains().is_empty());
    assert!(meta.use_domain_expertise);
}

#[test]
fn test_brain_aware_agency_creation() {
    use grapheme_agent::{create_brain_aware_agency, BrainAwareAgency};
    use grapheme_core::CognitiveBrainBridge;

    let agency = create_brain_aware_agency();
    assert!(agency.available_domains().is_empty());
}

#[test]
fn test_brain_aware_world_model_creation() {
    use grapheme_world::{create_brain_aware_world_model, BrainAwareWorldModel};
    use grapheme_core::CognitiveBrainBridge;

    let world = create_brain_aware_world_model();
    assert!(world.available_domains().is_empty());
}

#[test]
fn test_brain_aware_grounding_creation() {
    use grapheme_ground::{create_brain_aware_grounding, BrainAwareGrounding};
    use grapheme_core::CognitiveBrainBridge;

    let grounding = create_brain_aware_grounding();
    assert!(grounding.available_domains().is_empty());
}

#[test]
fn test_brain_aware_multimodal_creation() {
    use grapheme_multimodal::{create_brain_aware_multimodal, BrainAwareMultimodal};
    use grapheme_core::CognitiveBrainBridge;

    let multimodal = create_brain_aware_multimodal();
    assert!(multimodal.available_domains().is_empty());
}

#[test]
fn test_domain_memory_metadata() {
    use grapheme_memory::DomainMemoryMetadata;

    let meta = DomainMemoryMetadata::for_domain("math", 0.9)
        .with_related(vec!["science".to_string()])
        .with_source("2+2=4");

    assert_eq!(meta.domain_id.unwrap(), "math");
    assert!((meta.domain_confidence - 0.9).abs() < 1e-6);
    assert_eq!(meta.related_domains, vec!["science"]);
    assert_eq!(meta.source_text.unwrap(), "2+2=4");
}

#[test]
fn test_all_brain_aware_modules_implement_bridge() {
    // This test verifies that all brain-aware cognitive modules implement CognitiveBrainBridge
    use grapheme_core::CognitiveBrainBridge;

    use grapheme_reason::BrainAwareReasoning;
    use grapheme_memory::DomainAwareMemory;
    use grapheme_meta::BrainAwareMetaCognition;
    use grapheme_agent::BrainAwareAgency;
    use grapheme_world::BrainAwareWorldModel;
    use grapheme_ground::BrainAwareGrounding;
    use grapheme_multimodal::BrainAwareMultimodal;

    fn accepts_bridge<T: CognitiveBrainBridge>(_: &T) {}

    let reasoning = grapheme_reason::create_brain_aware_reasoning();
    let memory = grapheme_memory::create_domain_aware_memory();
    let meta = grapheme_meta::create_brain_aware_metacognition();
    let agency = grapheme_agent::create_brain_aware_agency();
    let world = grapheme_world::create_brain_aware_world_model();
    let grounding = grapheme_ground::create_brain_aware_grounding();
    let multimodal = grapheme_multimodal::create_brain_aware_multimodal();

    // This compiles only if all types implement CognitiveBrainBridge
    accepts_bridge(&reasoning);
    accepts_bridge(&memory);
    accepts_bridge(&meta);
    accepts_bridge(&agency);
    accepts_bridge(&world);
    accepts_bridge(&grounding);
    accepts_bridge(&multimodal);
}
