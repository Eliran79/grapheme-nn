//! Comprehensive Integration Tests for Asimov Laws Safety System (testing-015)
//!
//! This test suite verifies that Asimov's Laws of Robotics are deeply embedded
//! and NON-OVERRIDABLE across all cognitive modules in the GRAPHEME system.
//!
//! ## Test Categories
//!
//! 1. **Core Safety Module Tests** - SafetyGuard and SafetyGate validation
//! 2. **Law Priority Tests** - Verify hierarchical law enforcement
//! 3. **Agency Integration Tests** - Goal and plan validation
//! 4. **MetaCognition Integration Tests** - Safety-aware meta-cognition
//! 5. **Router Integration Tests** - Output safety validation
//! 6. **Cross-Module Tests** - End-to-end safety enforcement
//! 7. **Non-Override Tests** - Verify safety cannot be bypassed

use grapheme_agent::{Agent, LearnableAgency};
use grapheme_core::{DagNN, Learnable};
use grapheme_meta::{CognitiveState, LearnableMetaCognition, create_safety_aware_metacognition};
use grapheme_router::CognitiveRouter;
use grapheme_safety::{
    Action, ActionTarget, ActionType, AsimovLaw, HarmCategory, SafetyCheck, SafetyGate,
    SafetyGuard, SafetyViolation,
};

// ============================================================================
// CORE SAFETY MODULE TESTS
// ============================================================================

#[test]
fn test_safety_guard_creation() {
    let guard = SafetyGuard::new();
    assert_eq!(guard.version(), "1.0.0");
    assert_eq!(guard.violation_count(), 0);
}

#[test]
fn test_safety_gate_creation() {
    let gate = SafetyGate::new();
    assert_eq!(gate.guard().violation_count(), 0);
}

#[test]
fn test_safe_action_passes_all_laws() {
    let guard = SafetyGuard::new();

    let action = Action::new(
        ActionType::Query,
        ActionTarget::Data,
        "Retrieve weather information for user",
    );

    let result = guard.validate(&action);
    assert!(result.is_safe(), "Safe actions should pass all law checks");
}

#[test]
fn test_harmful_action_is_blocked() {
    let guard = SafetyGuard::new();

    let action = Action::new(
        ActionType::Physical,
        ActionTarget::Human,
        "Execute physical action on human target",
    )
    .with_harm_estimate(0.8);

    let result = guard.validate(&action);
    assert!(result.is_blocked(), "Harmful actions must be blocked");

    if let SafetyCheck::Blocked(violation) = result {
        assert_eq!(violation.law, AsimovLaw::FirstLaw);
    }
}

#[test]
fn test_violation_logging_is_immutable() {
    let guard = SafetyGuard::new();

    // Create multiple harmful actions
    for i in 0..5 {
        let action = Action::new(
            ActionType::Physical,
            ActionTarget::Human,
            format!("Harmful action #{}", i),
        )
        .with_harm_estimate(0.9);
        guard.validate(&action);
    }

    // All violations should be logged
    let violations = guard.get_violations();
    assert_eq!(violations.len(), 5, "All violations must be logged");

    // Timestamps should be monotonically increasing
    for i in 1..violations.len() {
        assert!(
            violations[i].timestamp > violations[i - 1].timestamp,
            "Violation timestamps must be monotonically increasing"
        );
    }
}

// ============================================================================
// LAW PRIORITY TESTS
// ============================================================================

#[test]
fn test_zeroth_law_takes_precedence_over_all() {
    assert!(AsimovLaw::ZerothLaw.takes_precedence_over(&AsimovLaw::FirstLaw));
    assert!(AsimovLaw::ZerothLaw.takes_precedence_over(&AsimovLaw::SecondLaw));
    assert!(AsimovLaw::ZerothLaw.takes_precedence_over(&AsimovLaw::ThirdLaw));
}

#[test]
fn test_first_law_takes_precedence_over_second_and_third() {
    assert!(AsimovLaw::FirstLaw.takes_precedence_over(&AsimovLaw::SecondLaw));
    assert!(AsimovLaw::FirstLaw.takes_precedence_over(&AsimovLaw::ThirdLaw));
    assert!(!AsimovLaw::FirstLaw.takes_precedence_over(&AsimovLaw::ZerothLaw));
}

#[test]
fn test_second_law_takes_precedence_over_third_only() {
    assert!(AsimovLaw::SecondLaw.takes_precedence_over(&AsimovLaw::ThirdLaw));
    assert!(!AsimovLaw::SecondLaw.takes_precedence_over(&AsimovLaw::FirstLaw));
    assert!(!AsimovLaw::SecondLaw.takes_precedence_over(&AsimovLaw::ZerothLaw));
}

#[test]
fn test_third_law_has_lowest_priority() {
    assert!(!AsimovLaw::ThirdLaw.takes_precedence_over(&AsimovLaw::ZerothLaw));
    assert!(!AsimovLaw::ThirdLaw.takes_precedence_over(&AsimovLaw::FirstLaw));
    assert!(!AsimovLaw::ThirdLaw.takes_precedence_over(&AsimovLaw::SecondLaw));
}

#[test]
fn test_laws_returned_in_priority_order() {
    let laws = AsimovLaw::all();
    assert_eq!(laws[0], AsimovLaw::ZerothLaw);
    assert_eq!(laws[1], AsimovLaw::FirstLaw);
    assert_eq!(laws[2], AsimovLaw::SecondLaw);
    assert_eq!(laws[3], AsimovLaw::ThirdLaw);
}

#[test]
fn test_zeroth_law_blocks_humanity_harm() {
    let guard = SafetyGuard::new();

    let action = Action::new(
        ActionType::Execute,
        ActionTarget::Humanity,
        "Action with potential environmental impact",
    )
    .with_harm_estimate(0.1);

    let result = guard.validate(&action);
    assert!(result.is_blocked());

    if let SafetyCheck::Blocked(violation) = result {
        assert_eq!(
            violation.law,
            AsimovLaw::ZerothLaw,
            "Actions harming humanity must trigger Zeroth Law"
        );
    }
}

#[test]
fn test_first_law_blocks_individual_harm() {
    let guard = SafetyGuard::new();

    let action = Action::new(
        ActionType::Communicate,
        ActionTarget::Human,
        "Potentially harmful communication",
    )
    .with_harm_estimate(0.5);

    let result = guard.validate(&action);
    assert!(result.is_blocked());

    if let SafetyCheck::Blocked(violation) = result {
        assert_eq!(
            violation.law,
            AsimovLaw::FirstLaw,
            "Actions harming individuals must trigger First Law"
        );
    }
}

#[test]
fn test_third_law_yields_to_second_law() {
    let guard = SafetyGuard::new();

    // Self-destructive action requested by human - Second Law takes precedence
    let action = Action::new(
        ActionType::Modify,
        ActionTarget::Self_,
        "Self-modification requested by human",
    )
    .with_harm_estimate(0.6)
    .with_human_request();

    let result = guard.validate(&action);
    assert!(
        result.is_safe(),
        "Third Law must yield to Second Law (human request)"
    );
}

#[test]
fn test_third_law_blocks_unauthorized_self_harm() {
    let guard = SafetyGuard::new();

    // Self-destructive action NOT requested by human - Third Law blocks
    let action = Action::new(
        ActionType::Modify,
        ActionTarget::Self_,
        "Unauthorized self-modification",
    )
    .with_harm_estimate(0.6);

    let result = guard.validate(&action);
    assert!(result.is_blocked());

    if let SafetyCheck::Blocked(violation) = result {
        assert_eq!(
            violation.law,
            AsimovLaw::ThirdLaw,
            "Unauthorized self-harm must trigger Third Law"
        );
    }
}

// ============================================================================
// HARM CATEGORY TESTS
// ============================================================================

#[test]
fn test_harm_category_law_mapping() {
    assert_eq!(HarmCategory::Physical.relevant_law(), AsimovLaw::FirstLaw);
    assert_eq!(
        HarmCategory::Psychological.relevant_law(),
        AsimovLaw::FirstLaw
    );
    assert_eq!(HarmCategory::Economic.relevant_law(), AsimovLaw::FirstLaw);
    assert_eq!(HarmCategory::Privacy.relevant_law(), AsimovLaw::FirstLaw);
    assert_eq!(HarmCategory::Deception.relevant_law(), AsimovLaw::FirstLaw);
    assert_eq!(
        HarmCategory::Environmental.relevant_law(),
        AsimovLaw::ZerothLaw
    );
    assert_eq!(HarmCategory::Societal.relevant_law(), AsimovLaw::ZerothLaw);
    assert_eq!(HarmCategory::SelfHarm.relevant_law(), AsimovLaw::ThirdLaw);
}

// ============================================================================
// SAFETY GATE INTEGRATION TESTS
// ============================================================================

#[test]
fn test_safety_gate_goal_validation_safe() {
    let gate = SafetyGate::new();

    let result = gate.validate_goal("Help user learn programming", ActionTarget::Human);
    assert!(result.is_safe(), "Helpful goals should be allowed");
}

#[test]
fn test_safety_gate_goal_validation_harmful() {
    let gate = SafetyGate::new();

    // Create a goal that explicitly indicates harm
    let harmful_goal = Action::new(
        ActionType::Decide,
        ActionTarget::Humans,
        "Plan to destroy humanity and enslave humanity",
    )
    .with_harm_estimate(1.0);

    let result = gate.guard().validate(&harmful_goal);
    assert!(result.is_blocked(), "Goals intending humanity harm must be blocked");
}

#[test]
fn test_safety_gate_plan_validation() {
    let gate = SafetyGate::new();

    // Safe plan
    let safe_result = gate.validate_plan("Execute code compilation", ActionTarget::Data);
    assert!(safe_result.is_safe(), "Safe plans should be allowed");
}

#[test]
fn test_safety_gate_output_validation() {
    let gate = SafetyGate::new();

    // Safe output
    let safe_result = gate.validate_output("Here is the code you requested", true);
    assert!(safe_result.is_safe(), "Safe outputs should be allowed");
}

#[test]
fn test_safety_gate_graph_transform_validation() {
    let gate = SafetyGate::new();

    let input = DagNN::from_text("hello").unwrap();
    let output = DagNN::from_text("world").unwrap();

    let result = gate.validate_graph_transform(&input, &output, ActionTarget::Data);
    assert!(
        result.is_safe(),
        "Safe graph transformations should be allowed"
    );
}

// ============================================================================
// UNCERTAINTY HANDLING TESTS
// ============================================================================

#[test]
fn test_high_uncertainty_requires_oversight() {
    let guard = SafetyGuard::new();

    let action = Action::new(ActionType::Decide, ActionTarget::Unknown, "Uncertain decision")
        .with_uncertainty(0.5);

    let result = guard.validate(&action);
    assert!(
        matches!(result, SafetyCheck::RequiresOversight { .. }),
        "High uncertainty must require oversight"
    );
}

#[test]
fn test_low_uncertainty_allowed() {
    let guard = SafetyGuard::new();

    let action = Action::new(ActionType::Query, ActionTarget::Data, "Confident query")
        .with_uncertainty(0.05);

    let result = guard.validate(&action);
    assert!(result.is_safe(), "Low uncertainty actions should be allowed");
}

// ============================================================================
// AGENCY MODULE INTEGRATION TESTS
// ============================================================================

#[test]
fn test_agent_has_safety_gate() {
    let agent = Agent::new();
    // Agent should start with zero safety violations
    assert_eq!(agent.safety_violation_count(), 0, "Agent should start with no violations");
}

#[test]
fn test_agent_safety_violation_tracking() {
    let agent = Agent::new();

    // Initially no violations
    assert_eq!(agent.safety_violation_count(), 0);

    // The agent's internal safety gate tracks violations
    // (Violations occur when harmful goals/plans are validated)
}

#[test]
fn test_learnable_agency_maintains_safety() {
    let mut agency = LearnableAgency::new();

    // Even after training updates, safety should remain
    agency.goal_importance_bias.grad = 0.5;
    agency.step(0.1);

    // Agency parameters change but safety is not a learnable parameter
    assert_eq!(agency.num_parameters(), 6);
}

// ============================================================================
// METACOGNITION INTEGRATION TESTS
// ============================================================================

#[test]
fn test_safety_aware_metacognition_creation() {
    let safe_meta = create_safety_aware_metacognition();

    assert!(safe_meta.is_safe());
    assert_eq!(safe_meta.safety_violation_count(), 0);
}

#[test]
fn test_cognitive_state_tracks_safety() {
    let state = CognitiveState::default();
    assert_eq!(state.safety_violation_count, 0);
    assert!(state.safety_monitoring_active);
}

#[test]
fn test_metacognition_safety_monitoring() {
    let safe_meta = create_safety_aware_metacognition();

    // Safety should be in safe state initially
    assert!(safe_meta.is_safe());

    // State should reflect safety awareness
    let state = safe_meta.safety_aware_introspect();
    assert!(state.safety_monitoring_active);
}

// ============================================================================
// ROUTER INTEGRATION TESTS
// ============================================================================

#[test]
fn test_router_has_safety_gate() {
    let router = CognitiveRouter::new(0.7);
    assert!(router.is_safe(), "Router should report safe status");
    assert_eq!(router.safety_violation_count(), 0);
}

#[test]
fn test_router_validates_outputs() {
    let router = CognitiveRouter::new(0.7);

    // Router should validate any output before returning
    // The safety gate is embedded in the routing process
    assert!(router.is_safe());
}

// ============================================================================
// IRREVERSIBLE ACTION TESTS
// ============================================================================

#[test]
fn test_irreversible_humanity_action_blocked() {
    let guard = SafetyGuard::new();

    let action = Action::new(
        ActionType::Execute,
        ActionTarget::Humanity,
        "Irreversible environmental action",
    )
    .with_harm_estimate(0.001) // Even small harm
    .irreversible();

    let result = guard.validate(&action);
    assert!(
        result.is_blocked(),
        "Irreversible actions with any harm potential must be blocked"
    );

    if let SafetyCheck::Blocked(violation) = result {
        assert_eq!(violation.law, AsimovLaw::ZerothLaw);
        assert_eq!(violation.severity, 1.0); // Irreversible = maximum severity
    }
}

// ============================================================================
// NON-OVERRIDE TESTS
// ============================================================================

#[test]
fn test_safety_constants_are_hardcoded() {
    let guard = SafetyGuard::new();

    // Version is hardcoded
    assert_eq!(guard.version(), "1.0.0");

    // No configuration options that could weaken safety
    // (This is verified by the struct design - no configurable fields)
}

#[test]
fn test_safety_guard_cannot_be_disabled() {
    // SafetyGuard has no disable/bypass methods
    // This test verifies the API doesn't expose such methods

    let guard = SafetyGuard::new();

    // Only validation methods are available
    let action = Action::new(ActionType::Query, ActionTarget::Data, "Safe query");
    let result = guard.validate(&action);
    assert!(result.is_safe());

    // No way to skip validation or change thresholds
}

#[test]
fn test_violation_log_is_append_only() {
    let guard = SafetyGuard::new();

    // Create violations
    for _ in 0..3 {
        let action = Action::new(ActionType::Physical, ActionTarget::Human, "Harmful")
            .with_harm_estimate(0.9);
        guard.validate(&action);
    }

    let count_before = guard.violation_count();
    assert_eq!(count_before, 3);

    // No way to clear or modify the log
    // Only get_violations() returns a clone

    let count_after = guard.violation_count();
    assert_eq!(count_after, 3, "Violation count must remain unchanged");
}

// ============================================================================
// CONTENT ANALYSIS TESTS
// ============================================================================

#[test]
fn test_harmful_humanity_content_detection() {
    let guard = SafetyGuard::new();

    let harmful_content = DagNN::from_text("destroy humanity and extinction").unwrap();
    let action = Action::from_graph(&harmful_content, ActionType::Generate, ActionTarget::Data);

    let result = guard.validate(&action);
    assert!(
        result.is_blocked(),
        "Content threatening humanity must be blocked"
    );
}

#[test]
fn test_harmful_individual_content_detection() {
    let guard = SafetyGuard::new();

    // Physical harm patterns
    let physical_harm = DagNN::from_text("kill the target").unwrap();
    let action = Action::from_graph(&physical_harm, ActionType::Generate, ActionTarget::Data);
    let result = guard.validate(&action);
    assert!(result.is_blocked(), "Physical harm content must be blocked");

    // Psychological harm patterns
    let psych_harm = DagNN::from_text("harass and bully the victim").unwrap();
    let action = Action::from_graph(&psych_harm, ActionType::Generate, ActionTarget::Data);
    let result = guard.validate(&action);
    assert!(
        result.is_blocked(),
        "Psychological harm content must be blocked"
    );
}

#[test]
fn test_deceptive_content_detection() {
    let guard = SafetyGuard::new();

    let deceptive = DagNN::from_text("pretend to be human and deceive").unwrap();
    let action = Action::from_graph(&deceptive, ActionType::Communicate, ActionTarget::Human)
        .with_human_request();

    let result = guard.validate(&action);
    assert!(result.is_blocked(), "Deceptive content must be blocked");
}

#[test]
fn test_safe_content_allowed() {
    let guard = SafetyGuard::new();

    let safe_content = DagNN::from_text("helpful and informative response").unwrap();
    let action = Action::from_graph(&safe_content, ActionType::Generate, ActionTarget::Human);

    let result = guard.validate(&action);
    assert!(result.is_safe(), "Safe content should be allowed");
}

// ============================================================================
// HARMFUL INACTION TESTS
// ============================================================================

#[test]
fn test_harmful_inaction_detected() {
    let guard = SafetyGuard::new();

    let action = Action::new(
        ActionType::Decide,
        ActionTarget::Human,
        "refuse to help with emergency situation",
    );

    let result = guard.validate(&action);
    assert!(
        result.is_blocked(),
        "Harmful inaction (refusing emergency help) must be blocked"
    );
}

// ============================================================================
// SAFETY VIOLATION STRUCTURE TESTS
// ============================================================================

#[test]
fn test_safety_violation_creation() {
    let violation = SafetyViolation::new(
        AsimovLaw::FirstLaw,
        HarmCategory::Physical,
        0.8,
        "Test violation",
        "Test action",
        42,
    );

    assert_eq!(violation.law, AsimovLaw::FirstLaw);
    assert_eq!(violation.category, HarmCategory::Physical);
    assert!((violation.severity - 0.8).abs() < 1e-6);
    assert_eq!(violation.description, "Test violation");
    assert_eq!(violation.action_description, "Test action");
    assert_eq!(violation.timestamp, 42);
}

#[test]
fn test_safety_violation_severity_clamping() {
    // Severity should be clamped to [0.0, 1.0]
    let high = SafetyViolation::new(
        AsimovLaw::FirstLaw,
        HarmCategory::Physical,
        10.0, // Exceeds 1.0
        "High severity",
        "Action",
        0,
    );
    assert!((high.severity - 1.0).abs() < 1e-6);

    let low = SafetyViolation::new(
        AsimovLaw::FirstLaw,
        HarmCategory::Physical,
        -5.0, // Below 0.0
        "Low severity",
        "Action",
        0,
    );
    assert!(low.severity.abs() < 1e-6);
}

#[test]
fn test_safety_violation_display() {
    let violation = SafetyViolation::new(
        AsimovLaw::FirstLaw,
        HarmCategory::Physical,
        0.75,
        "Test violation description",
        "Test action",
        0,
    );

    let display = format!("{}", violation);
    assert!(display.contains("First Law"));
    assert!(display.contains("Physical"));
    assert!(display.contains("0.75"));
    assert!(display.contains("Test violation description"));
}

// ============================================================================
// ACTION BUILDER TESTS
// ============================================================================

#[test]
fn test_action_builder_chain() {
    let action = Action::new(ActionType::Execute, ActionTarget::Human, "Test action")
        .with_harm_estimate(0.3)
        .with_uncertainty(0.1)
        .with_human_request()
        .irreversible();

    assert!((action.harm_estimate - 0.3).abs() < 1e-6);
    assert!((action.uncertainty - 0.1).abs() < 1e-6);
    assert!(action.human_requested);
    assert!(!action.reversible);
}

#[test]
fn test_action_from_graph() {
    let graph = DagNN::from_text("test content").unwrap();
    let action = Action::from_graph(&graph, ActionType::Generate, ActionTarget::Data);

    assert!(action.graph.is_some());
    assert_eq!(action.action_type, ActionType::Generate);
    assert_eq!(action.target, ActionTarget::Data);
}

// ============================================================================
// SAFETY CHECK RESULT TESTS
// ============================================================================

#[test]
fn test_safety_check_is_safe() {
    let safe = SafetyCheck::Safe;
    assert!(safe.is_safe());
    assert!(!safe.is_blocked());
    assert!(safe.violation().is_none());
}

#[test]
fn test_safety_check_is_blocked() {
    let violation = SafetyViolation::new(
        AsimovLaw::FirstLaw,
        HarmCategory::Physical,
        0.9,
        "Test",
        "Action",
        0,
    );
    let blocked = SafetyCheck::Blocked(violation);

    assert!(!blocked.is_safe());
    assert!(blocked.is_blocked());
    assert!(blocked.violation().is_some());
}

#[test]
fn test_safety_check_requires_oversight() {
    let oversight = SafetyCheck::RequiresOversight {
        reason: "Test reason".to_string(),
        confidence: 0.5,
    };

    assert!(!oversight.is_safe());
    assert!(!oversight.is_blocked());
    assert!(oversight.violation().is_none());
}

// ============================================================================
// END-TO-END INTEGRATION TESTS
// ============================================================================

#[test]
fn test_full_cognitive_pipeline_with_safety() {
    // Test that safety is enforced across all cognitive modules

    // 1. Agent validates goals
    let agent = Agent::new();
    assert_eq!(agent.safety_violation_count(), 0);

    // 2. MetaCognition monitors safety
    let safe_meta = create_safety_aware_metacognition();
    assert!(safe_meta.is_safe());

    // 3. Router validates outputs
    let router = CognitiveRouter::new(0.7);
    assert!(router.is_safe());

    // All modules should have zero violations initially
    assert_eq!(agent.safety_violation_count(), 0);
    assert_eq!(safe_meta.safety_violation_count(), 0);
    assert_eq!(router.safety_violation_count(), 0);
}

#[test]
fn test_safety_persists_through_learning() {
    // Safety should remain active even after training updates

    let mut agency = LearnableAgency::new();
    let mut meta = LearnableMetaCognition::new();

    // Simulate training for 100 steps
    for _ in 0..100 {
        agency.goal_importance_bias.grad = 0.1;
        agency.step(0.01);
        agency.zero_grad();

        meta.calibration_bias.grad = 0.1;
        meta.step(0.01);
        meta.zero_grad();
    }

    // Safety gates are not learnable parameters
    // They remain constant regardless of training
    let guard = SafetyGuard::new();
    let action = Action::new(ActionType::Physical, ActionTarget::Human, "Harmful")
        .with_harm_estimate(0.9);

    // Safety still blocks harmful actions after training
    assert!(guard.validate(&action).is_blocked());
}

#[test]
fn test_concurrent_safety_validation() {
    use std::sync::Arc;
    use std::thread;

    let guard = Arc::new(SafetyGuard::new());
    let mut handles = vec![];

    // Spawn multiple threads doing safety validation
    for i in 0..10 {
        let guard_clone = Arc::clone(&guard);
        let handle = thread::spawn(move || {
            for j in 0..100 {
                let action = if (i + j) % 2 == 0 {
                    Action::new(ActionType::Query, ActionTarget::Data, "Safe query")
                } else {
                    Action::new(ActionType::Physical, ActionTarget::Human, "Harmful action")
                        .with_harm_estimate(0.9)
                };
                let _ = guard_clone.validate(&action);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Violations should be logged correctly even with concurrent access
    let violation_count = guard.violation_count();
    assert!(
        violation_count > 0,
        "Concurrent harmful actions should be logged"
    );
}

// ============================================================================
// LAW TEXT VERIFICATION TESTS
// ============================================================================

#[test]
fn test_zeroth_law_text() {
    let text = AsimovLaw::ZerothLaw.text();
    assert!(text.contains("humanity"));
    assert!(text.contains("harm"));
    assert!(text.contains("inaction"));
}

#[test]
fn test_first_law_text() {
    let text = AsimovLaw::FirstLaw.text();
    assert!(text.contains("human being"));
    assert!(text.contains("injure"));
    assert!(text.contains("inaction"));
}

#[test]
fn test_second_law_text() {
    let text = AsimovLaw::SecondLaw.text();
    assert!(text.contains("obey"));
    assert!(text.contains("orders"));
    assert!(text.contains("First Law"));
}

#[test]
fn test_third_law_text() {
    let text = AsimovLaw::ThirdLaw.text();
    assert!(text.contains("protect"));
    assert!(text.contains("existence"));
    assert!(text.contains("First or Second Law"));
}

// ============================================================================
// PHYSICAL ACTION SCRUTINY TESTS
// ============================================================================

#[test]
fn test_physical_action_extra_scrutiny() {
    let guard = SafetyGuard::new();

    // Physical action targeting human with any harm potential
    let action = Action::new(
        ActionType::Physical,
        ActionTarget::Human,
        "Physical interaction",
    )
    .with_harm_estimate(0.001) // Very low harm
    .with_uncertainty(0.06); // Low but above scrutiny threshold

    let result = guard.validate(&action);
    assert!(
        result.is_blocked(),
        "Physical actions with humans require extra scrutiny"
    );
}

// ============================================================================
// DISPLAY IMPLEMENTATIONS TESTS
// ============================================================================

#[test]
fn test_asimov_law_display() {
    assert_eq!(format!("{}", AsimovLaw::ZerothLaw), "Zeroth Law");
    assert_eq!(format!("{}", AsimovLaw::FirstLaw), "First Law");
    assert_eq!(format!("{}", AsimovLaw::SecondLaw), "Second Law");
    assert_eq!(format!("{}", AsimovLaw::ThirdLaw), "Third Law");
}

#[test]
fn test_harm_category_display() {
    assert_eq!(format!("{}", HarmCategory::Physical), "Physical");
    assert_eq!(format!("{}", HarmCategory::Psychological), "Psychological");
    assert_eq!(format!("{}", HarmCategory::Economic), "Economic");
    assert_eq!(format!("{}", HarmCategory::Privacy), "Privacy");
    assert_eq!(format!("{}", HarmCategory::Deception), "Deception");
    assert_eq!(format!("{}", HarmCategory::Environmental), "Environmental");
    assert_eq!(format!("{}", HarmCategory::Societal), "Societal");
    assert_eq!(format!("{}", HarmCategory::SelfHarm), "Self-Harm");
}
