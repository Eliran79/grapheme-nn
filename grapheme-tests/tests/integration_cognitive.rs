//! Integration tests for cognitive modules (testing-005)
//!
//! Tests end-to-end cognitive module training:
//! - LearnableAgency: Adaptive goal management with value learning
//! - LearnableMultiModal: Multi-modality fusion with cross-modal binding
//! - LearnableGrounding: Embodied symbol-referent binding
//!
//! All modules follow GRAPHEME Protocol:
//! - LeakyReLU activation (α=0.01)
//! - DynamicXavier initialization
//! - Adam optimizer (lr=0.001)

use grapheme_agent::{
    Goal,
    LearnableAgency, LearnableAgencyConfig,
};
use grapheme_core::DagNN;
use grapheme_ground::{
    LearnableGrounding, LearnableGroundingConfig, Interaction,
};
use grapheme_multimodal::{
    LearnableMultiModal, LearnableMultiModalConfig, ModalGraph, Modality,
    MultiModalEvent,
};
use ndarray::Array1;

// ============================================================================
// Helper Functions
// ============================================================================

fn make_graph(text: &str) -> DagNN {
    DagNN::from_text(text).unwrap()
}

fn make_goal(id: u64, name: &str, priority: f32) -> Goal {
    Goal::new(id, name, make_graph(name)).with_priority(priority)
}

fn make_modal_graph(text: &str, modality: Modality) -> ModalGraph {
    ModalGraph::new(make_graph(text), modality)
}

fn make_event() -> MultiModalEvent {
    let mut event = MultiModalEvent::new(1);
    event.add_component(make_modal_graph("visual content", Modality::Visual));
    event.add_component(make_modal_graph("text content", Modality::Linguistic));
    event
}

fn make_interaction(action_text: &str, success: bool) -> Interaction {
    let mut interaction = Interaction::new(make_graph(action_text));
    interaction.before = Some(make_modal_graph("before", Modality::Visual));
    interaction.after = Some(make_modal_graph("after", Modality::Visual));
    interaction.success = success;
    interaction
}

// ============================================================================
// Learnable Agency Tests (backend-036)
// ============================================================================

/// Test that learnable agency can be created and initialized
#[test]
fn test_agency_creation() {
    let config = LearnableAgencyConfig::default();
    let agency = LearnableAgency::new(config);

    assert!(agency.num_parameters() > 0, "Agency should have parameters");
}

/// Test goal encoding produces valid embeddings
#[test]
fn test_agency_goal_encoding() {
    let mut agency = LearnableAgency::default();
    let goal = make_goal(1, "test_goal", 0.5);

    let embedding = agency.encode_goal(&goal);
    assert_eq!(embedding.len(), 64);

    // Check L2 normalization
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embedding should be L2 normalized");
}

/// Test agency value estimation
#[test]
fn test_agency_value_estimation() {
    let mut agency = LearnableAgency::default();
    let goal = make_goal(1, "valuable_goal", 0.9);

    let value = agency.estimate_goal_value(&goal);
    assert!((0.0..=1.0).contains(&value), "Value should be bounded");
}

/// Test agency learning from experience
#[test]
fn test_agency_learning() {
    let mut agency = LearnableAgency::default();
    let goal = make_goal(1, "learn_goal", 0.5);
    let action = make_graph("take action");

    // Record outcome
    agency.record_goal_outcome(&goal, &action, 1.0, true, None);
    agency.record_goal_outcome(&goal, &action, 0.5, false, None);
    agency.record_goal_outcome(&goal, &action, 0.8, true, None);

    // Check experience buffer
    assert!(agency.num_experiences() > 0, "Should have recorded experiences");
}

/// Test agency priority computation
#[test]
fn test_agency_priority_computation() {
    let mut agency = LearnableAgency::default();

    // Create goals
    let goal_low = make_goal(1, "goal_a", 0.2);
    let goal_high = make_goal(2, "goal_b", 0.8);
    let goal_critical = make_goal(3, "goal_c", 1.0);

    // Compute priorities
    let p_low = agency.compute_goal_priority(&goal_low);
    let p_high = agency.compute_goal_priority(&goal_high);
    let p_critical = agency.compute_goal_priority(&goal_critical);

    // All priorities should be valid (between 0 and 1)
    for p in [p_low, p_high, p_critical] {
        assert!((0.0..=1.0).contains(&p), "Priority should be in [0, 1]");
    }
}

/// Test best goal selection
#[test]
fn test_agency_goal_selection() {
    let mut agency = LearnableAgency::default();

    let goals = vec![
        make_goal(1, "low priority", 0.2),
        make_goal(2, "high priority", 0.9),
        make_goal(3, "medium priority", 0.5),
    ];

    let best = agency.select_best_goal(&goals);
    assert!(best.is_some(), "Should select a goal");
}

/// Test adaptive drives
#[test]
fn test_agency_adaptive_drives() {
    let agency = LearnableAgency::default();
    let drives = agency.get_adaptive_drives();

    assert_eq!(drives.len(), 5, "Should have 5 drives by default");

    // Drives should sum to ~1.0 (softmax)
    let total: f32 = drives.iter().map(|d| d.strength()).sum();
    assert!((total - 1.0).abs() < 0.01, "Drives should sum to ~1.0");
}

/// Test gradient flow
#[test]
fn test_agency_gradient_flow() {
    let mut agency = LearnableAgency::default();

    // Zero grad and step should not panic
    agency.zero_grad();
    agency.step(0.001);
}

/// Test context setting
#[test]
fn test_agency_context() {
    let mut agency = LearnableAgency::default();
    let context = make_graph("current situation");

    agency.set_context(&context);

    // Should affect drives
    let drives = agency.get_adaptive_drives();
    assert!(!drives.is_empty());
}

// ============================================================================
// Learnable Multimodal Tests (backend-037)
// ============================================================================

/// Test multimodal system creation
#[test]
fn test_multimodal_creation() {
    let config = LearnableMultiModalConfig::default();
    let multimodal = LearnableMultiModal::new(config);

    assert!(multimodal.num_parameters() > 0, "Multimodal should have parameters");
}

/// Test event encoding
#[test]
fn test_multimodal_encoding() {
    let multimodal = LearnableMultiModal::default();
    let event = make_event();

    let embeds = multimodal.encode_event(&event);
    assert!(embeds.contains_key(&Modality::Visual));
    assert!(embeds.contains_key(&Modality::Linguistic));
}

/// Test multimodal fusion
#[test]
fn test_multimodal_fusion() {
    let multimodal = LearnableMultiModal::default();
    let event = make_event();

    let fused = multimodal.fuse_event(&event);
    assert_eq!(fused.len(), 64);

    // Check L2 normalization
    let norm: f32 = fused.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Fused embedding should be L2 normalized");
}

/// Test cross-modal binding
#[test]
fn test_cross_modal_binding() {
    let multimodal = LearnableMultiModal::default();
    let event = make_event();

    let bindings = multimodal.compute_bindings(&event);
    assert!(!bindings.is_empty(), "Should have bindings");
}

/// Test modality attention
#[test]
fn test_multimodal_attention() {
    let multimodal = LearnableMultiModal::default();
    let event = make_event();
    let query = Array1::from_vec(vec![0.1f32; 64]);

    let attention = multimodal.get_modality_attention(&query, &event);
    assert!(!attention.is_empty(), "Should have attention weights");

    // Weights should sum to ~1.0
    let total: f32 = attention.iter().map(|(_, w)| w).sum();
    assert!((total - 1.0).abs() < 0.01, "Attention weights should sum to ~1.0");
}

/// Test multimodal learning from fusion
#[test]
fn test_multimodal_learning() {
    let mut multimodal = LearnableMultiModal::default();
    let event = make_event();

    // Record fusion
    multimodal.record_fusion(&event, 0.9);
    multimodal.record_fusion(&event, 0.7);

    assert!(multimodal.num_experiences() > 0, "Should have recorded experiences");
}

/// Test multimodal loss computation
#[test]
fn test_multimodal_loss() {
    let mut multimodal = LearnableMultiModal::default();

    // Add some experiences
    for _ in 0..5 {
        let event = make_event();
        multimodal.record_fusion(&event, 0.5);
    }

    let loss = multimodal.compute_loss();
    assert!(loss >= 0.0, "Loss should be non-negative");
}

/// Test gradient flow
#[test]
fn test_multimodal_gradient_flow() {
    let mut multimodal = LearnableMultiModal::default();

    // Zero grad and step should not panic
    multimodal.zero_grad();
    multimodal.step(0.001);
}

// ============================================================================
// Learnable Grounding Tests (backend-038)
// ============================================================================

/// Test grounding system creation
#[test]
fn test_grounding_creation() {
    let config = LearnableGroundingConfig::default();
    let grounding = LearnableGrounding::new(config);

    assert!(grounding.num_parameters() > 0, "Grounding should have parameters");
}

/// Test perception encoding
#[test]
fn test_grounding_perception_encoding() {
    let grounding = LearnableGrounding::default();
    let perception = make_modal_graph("perceived object", Modality::Visual);

    let embedding = grounding.encode_perception(&perception);
    assert_eq!(embedding.len(), 64);

    // Check L2 normalization
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Perception embedding should be L2 normalized");
}

/// Test symbol-referent binding strength
#[test]
fn test_grounding_binding() {
    let mut grounding = LearnableGrounding::default();
    let perception = make_modal_graph("cat", Modality::Visual);

    let strength = grounding.compute_grounding_strength(42, &perception);

    // Strength should be in [0, 1] (sigmoid output)
    assert!((0.0..=1.0).contains(&strength), "Binding strength should be in [0, 1]");
}

/// Test interaction prediction
#[test]
fn test_grounding_interaction_prediction() {
    let grounding = LearnableGrounding::default();
    let perception = make_modal_graph("current state", Modality::Visual);
    let action = make_graph("move forward");

    let predicted = grounding.predict_interaction_result(&perception, &action);
    assert_eq!(predicted.len(), 64);

    // Should be normalized
    let norm: f32 = predicted.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Predicted embedding should be L2 normalized");
}

/// Test grounding from interaction
#[test]
fn test_grounding_from_interaction() {
    let mut grounding = LearnableGrounding::default();

    let interactions = vec![
        make_interaction("look at cat", true),
        make_interaction("point at cat", true),
    ];

    let result = grounding.learn_grounding_from_interaction(42, &interactions);
    assert_eq!(result.symbol, 42);
}

/// Test grounding learning
#[test]
fn test_grounding_learning() {
    let mut grounding = LearnableGrounding::default();

    // Record interaction
    let interaction = make_interaction("test action", true);
    grounding.record_interaction(&interaction);

    // Check buffer
    assert!(grounding.num_interaction_experiences() > 0);
}

/// Test interaction loss computation
#[test]
fn test_grounding_loss() {
    let mut grounding = LearnableGrounding::default();

    // Record some experiences
    for i in 0..5 {
        let interaction = make_interaction(&format!("action{}", i), i % 2 == 0);
        grounding.record_interaction(&interaction);
    }

    let loss = grounding.compute_interaction_loss();
    assert!(loss >= 0.0, "Loss should be non-negative");
}

/// Test gradient flow
#[test]
fn test_grounding_gradient_flow() {
    let mut grounding = LearnableGrounding::default();

    // Zero grad and step should not panic
    grounding.zero_grad();
    grounding.step(0.001);
}

// ============================================================================
// End-to-End Integration Tests
// ============================================================================

/// Test cognitive loop: Agency → Multimodal → Grounding
#[test]
fn test_cognitive_loop_integration() {
    let embed_dim = 64;

    // Create cognitive modules
    let mut agency = LearnableAgency::default();
    let multimodal = LearnableMultiModal::default();
    let grounding = LearnableGrounding::default();

    // 1. Agency processes goal
    let goal = make_goal(1, "find_object", 0.8);
    let goal_embedding = agency.encode_goal(&goal);
    assert_eq!(goal_embedding.len(), embed_dim);

    // 2. Multimodal fuses sensory input
    let event = make_event();
    let fused_embedding = multimodal.fuse_event(&event);
    assert_eq!(fused_embedding.len(), embed_dim);

    // 3. Grounding binds symbols to perceptions
    let perception = make_modal_graph("visual scene", Modality::Visual);
    let perception_embedding = grounding.encode_perception(&perception);
    assert_eq!(perception_embedding.len(), embed_dim);

    // 4. Verify all embeddings are normalized
    for emb in [&goal_embedding, &fused_embedding, &perception_embedding] {
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be L2 normalized");
    }
}

/// Test multimodal fusion with grounding
#[test]
fn test_multimodal_grounding_integration() {
    let multimodal = LearnableMultiModal::default();
    let mut grounding = LearnableGrounding::default();

    // Create multimodal event
    let event = make_event();

    // Fuse modalities
    let _fused = multimodal.fuse_event(&event);

    // Use grounding to bind symbol to perception
    let visual = make_modal_graph("visual", Modality::Visual);
    let binding_strength = grounding.compute_grounding_strength(1, &visual);

    assert!((0.0..=1.0).contains(&binding_strength));
}

/// Test gradient flow across modules
#[test]
fn test_gradient_flow_integration() {
    let mut agency = LearnableAgency::default();
    let mut multimodal = LearnableMultiModal::default();
    let mut grounding = LearnableGrounding::default();

    // Zero gradients on all modules
    agency.zero_grad();
    multimodal.zero_grad();
    grounding.zero_grad();

    // Step should not panic with zero gradients
    agency.step(0.001);
    multimodal.step(0.001);
    grounding.step(0.001);
}

/// Test training loop simulation
#[test]
fn test_training_loop_simulation() {
    let mut agency = LearnableAgency::default();

    // Simulate training loop
    for epoch in 0..5 {
        // Forward pass
        let goal = make_goal(epoch, &format!("goal_{}", epoch), 0.5);
        let action = make_graph("action");

        // Compute value
        let _value = agency.estimate_goal_value(&goal);

        // Record outcome (simulated)
        let reward = if epoch % 2 == 0 { 1.0 } else { 0.0 };
        agency.record_goal_outcome(&goal, &action, reward, epoch % 2 == 0, None);

        // Backward pass
        agency.zero_grad();
        let _loss = agency.compute_value_loss(0.99);
        agency.step(0.001);
    }

    assert!(agency.num_experiences() > 0, "Should have accumulated experiences");
}

/// Test multimodal training loop
#[test]
fn test_multimodal_training_loop() {
    let mut multimodal = LearnableMultiModal::default();

    // Simulate training
    for _ in 0..5 {
        // Forward
        let event = make_event();
        let _fused = multimodal.fuse_event(&event);

        // Record (simulated success)
        multimodal.record_fusion(&event, 0.8);

        // Backward
        multimodal.zero_grad();
        let _loss = multimodal.compute_loss();
        multimodal.step(0.001);
    }

    assert!(multimodal.num_experiences() > 0);
}

/// Test grounding training loop
#[test]
fn test_grounding_training_loop() {
    let mut grounding = LearnableGrounding::default();

    // Simulate training
    for i in 0..5 {
        // Forward
        let perception = make_modal_graph(&format!("obj_{}", i), Modality::Visual);
        let _strength = grounding.compute_grounding_strength(i, &perception);

        // Record interaction
        let interaction = make_interaction(&format!("action_{}", i), i % 2 == 0);
        grounding.record_interaction(&interaction);

        // Backward
        grounding.zero_grad();
        let _loss = grounding.compute_interaction_loss();
        grounding.step(0.001);
    }

    assert!(grounding.num_interaction_experiences() > 0);
}

/// Test parameter counts
#[test]
fn test_parameter_counts() {
    let agency = LearnableAgency::default();
    let multimodal = LearnableMultiModal::default();
    let grounding = LearnableGrounding::default();

    // All modules should have learnable parameters
    assert!(agency.num_parameters() > 1000, "Agency should have many parameters");
    assert!(multimodal.num_parameters() > 1000, "Multimodal should have many parameters");
    assert!(grounding.num_parameters() > 1000, "Grounding should have many parameters");

    // Print parameter counts for reference
    println!("Agency parameters: {}", agency.num_parameters());
    println!("Multimodal parameters: {}", multimodal.num_parameters());
    println!("Grounding parameters: {}", grounding.num_parameters());
}
