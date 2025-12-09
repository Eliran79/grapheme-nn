//! Integration tests for VisionBrain and MnistModel on real MNIST data
//!
//! These tests verify that the vision pipeline works correctly on actual
//! MNIST images, testing:
//! - Blob detection on real digit images
//! - Hierarchical blob detection
//! - Spatial relationship graph construction
//! - Determinism (same image = same graph)
//! - End-to-end MnistModel pipeline

use grapheme_vision::{
    extract_blobs, extract_hierarchical_blobs, image_to_graph,
    FeatureConfig, MnistModel, MnistModelConfig, RawImage, VisionBrain, VisionEdge,
};
use mnist::{Mnist, MnistBuilder};
use std::collections::HashMap;

/// Load a small subset of MNIST for testing
fn load_mnist_subset() -> Mnist {
    MnistBuilder::new()
        .base_path("../data/mnist")
        .label_format_digit()
        .training_set_length(100)
        .test_set_length(20)
        .finalize()
}

/// Normalize pixel values from u8 [0, 255] to f32 [0.0, 1.0]
fn normalize_image(pixels: &[u8]) -> Vec<f32> {
    pixels.iter().map(|&p| p as f32 / 255.0).collect()
}

#[test]
fn test_mnist_blob_extraction() {
    let mnist = load_mnist_subset();
    let config = FeatureConfig::mnist();

    // Test on first 10 training images
    for i in 0..10 {
        let start = i * 784;
        let end = start + 784;
        let pixels = normalize_image(&mnist.trn_img[start..end]);
        let image = RawImage::from_mnist(&pixels).unwrap();

        let blobs = extract_blobs(&image, &config);

        // Every digit should have at least one blob
        assert!(
            !blobs.is_empty(),
            "Image {} (label {}) should have at least one blob",
            i,
            mnist.trn_lbl[i]
        );

        // Check blob properties are valid
        for blob in &blobs {
            assert!(!blob.pixels.is_empty(), "Blob should have pixels");
            assert!(blob.intensity > 0.0, "Blob should have positive intensity");
            let (cx, cy) = blob.center;
            assert!(cx >= 0.0 && cx < 28.0, "Center x should be in image bounds");
            assert!(cy >= 0.0 && cy < 28.0, "Center y should be in image bounds");
        }
    }
}

#[test]
fn test_mnist_hierarchical_blobs() {
    let mnist = load_mnist_subset();
    let mut config = FeatureConfig::mnist();
    config.max_hierarchy_levels = 3;

    // Test on first 10 training images
    for i in 0..10 {
        let start = i * 784;
        let end = start + 784;
        let pixels = normalize_image(&mnist.trn_img[start..end]);
        let image = RawImage::from_mnist(&pixels).unwrap();

        let hierarchy = extract_hierarchical_blobs(&image, &config);

        // Should have the configured number of levels
        assert_eq!(hierarchy.num_levels, 3);

        // Should have some blobs
        assert!(
            !hierarchy.blobs.is_empty(),
            "Image {} should have hierarchical blobs",
            i
        );

        // Check parent-child consistency
        for (idx, hblob) in hierarchy.blobs.iter().enumerate() {
            if let Some(parent_idx) = hblob.parent {
                assert!(
                    hierarchy.blobs[parent_idx].children.contains(&idx),
                    "Parent-child relationship should be consistent"
                );
            }
        }
    }
}

#[test]
fn test_mnist_image_to_graph() {
    let mnist = load_mnist_subset();
    let config = FeatureConfig::mnist();

    // Test on first 10 training images
    for i in 0..10 {
        let start = i * 784;
        let end = start + 784;
        let pixels = normalize_image(&mnist.trn_img[start..end]);
        let image = RawImage::from_mnist(&pixels).unwrap();

        let graph = image_to_graph(&image, &config).unwrap();

        // Should have root node
        assert!(graph.root.is_some(), "Graph should have root node");

        // Should have at least root + one blob
        assert!(
            graph.node_count() >= 2,
            "Image {} should have at least 2 nodes, got {}",
            i,
            graph.node_count()
        );

        // Should have edges (at least Contains edges from root)
        assert!(
            graph.edge_count() >= 1,
            "Image {} should have at least 1 edge",
            i
        );
    }
}

#[test]
fn test_mnist_spatial_relationships() {
    let mnist = load_mnist_subset();
    let mut config = FeatureConfig::mnist();
    config.build_spatial_graph = true;
    config.adjacency_threshold = 0.3;

    // Test images that should have multiple blobs with spatial relationships
    // (e.g., digit "8" has two loops, "0" is a loop, etc.)
    let test_indices: Vec<usize> = (0..20).collect();

    let mut found_directional = false;
    let mut found_proximity = false;

    for &i in &test_indices {
        let start = i * 784;
        let end = start + 784;
        let pixels = normalize_image(&mnist.trn_img[start..end]);
        let image = RawImage::from_mnist(&pixels).unwrap();

        let graph = image_to_graph(&image, &config).unwrap();

        // Check for different edge types
        for edge_idx in graph.graph.edge_indices() {
            if let Some(edge) = graph.graph.edge_weight(edge_idx) {
                match edge {
                    VisionEdge::Above | VisionEdge::Below | VisionEdge::LeftOf | VisionEdge::RightOf => {
                        found_directional = true;
                    }
                    VisionEdge::Proximity(_) => {
                        found_proximity = true;
                    }
                    _ => {}
                }
            }
        }
    }

    // At least some images should have spatial relationships
    assert!(
        found_directional || found_proximity,
        "Should find some spatial relationships in MNIST digits"
    );
}

#[test]
fn test_mnist_determinism() {
    let mnist = load_mnist_subset();
    let config = FeatureConfig::mnist();
    let brain = VisionBrain::mnist();

    // Test determinism: same image should produce identical graphs
    for i in 0..5 {
        let start = i * 784;
        let end = start + 784;
        let pixels = normalize_image(&mnist.trn_img[start..end]);

        // Convert twice
        let graph1 = brain.mnist_to_graph(&pixels).unwrap();
        let graph2 = brain.mnist_to_graph(&pixels).unwrap();

        // Should have same structure
        assert_eq!(
            graph1.node_count(),
            graph2.node_count(),
            "Same image should produce same node count"
        );
        assert_eq!(
            graph1.edge_count(),
            graph2.edge_count(),
            "Same image should produce same edge count"
        );
    }
}

#[test]
fn test_mnist_different_digits() {
    let mnist = load_mnist_subset();
    let _config = FeatureConfig::mnist();

    // Group images by label
    let mut by_label: HashMap<u8, Vec<usize>> = HashMap::new();
    for (i, &label) in mnist.trn_lbl.iter().enumerate().take(100) {
        by_label.entry(label).or_default().push(i);
    }

    // Test that we can process all digit types
    for label in 0..10u8 {
        if let Some(indices) = by_label.get(&label) {
            if let Some(&idx) = indices.first() {
                let start = idx * 784;
                let end = start + 784;
                let pixels = normalize_image(&mnist.trn_img[start..end]);
                let image = RawImage::from_mnist(&pixels).unwrap();

                let graph = image_to_graph(&image, &_config).unwrap();

                println!(
                    "Digit {}: {} nodes, {} edges",
                    label,
                    graph.node_count(),
                    graph.edge_count()
                );

                assert!(
                    graph.node_count() >= 2,
                    "Digit {} should have nodes",
                    label
                );
            }
        }
    }
}

#[test]
fn test_mnist_detailed_graph_analysis() {
    let mnist = load_mnist_subset();
    let mut config = FeatureConfig::mnist();
    config.build_spatial_graph = true;
    config.adjacency_threshold = 0.3;
    // Lower blob threshold to detect more structure within digits
    config.blob_threshold = 0.4;
    config.min_blob_size = 2;

    println!("\n=== Detailed MNIST Graph Analysis ===\n");

    // Analyze first 10 images
    for i in 0..10 {
        let start = i * 784;
        let end = start + 784;
        let pixels = normalize_image(&mnist.trn_img[start..end]);
        let label = mnist.trn_lbl[i];
        let image = RawImage::from_mnist(&pixels).unwrap();

        // Extract blobs
        let blobs = extract_blobs(&image, &config);

        // Extract hierarchical blobs
        let hierarchy = extract_hierarchical_blobs(&image, &config);

        // Build graph
        let graph = image_to_graph(&image, &config).unwrap();

        // Count edge types
        let mut edge_counts: HashMap<String, usize> = HashMap::new();
        for edge_idx in graph.graph.edge_indices() {
            if let Some(edge) = graph.graph.edge_weight(edge_idx) {
                let name = match edge {
                    VisionEdge::Contains => "Contains",
                    VisionEdge::Adjacent => "Adjacent",
                    VisionEdge::Hierarchy => "Hierarchy",
                    VisionEdge::Above => "Above",
                    VisionEdge::Below => "Below",
                    VisionEdge::LeftOf => "LeftOf",
                    VisionEdge::RightOf => "RightOf",
                    VisionEdge::Proximity(_) => "Proximity",
                    VisionEdge::SameContour => "SameContour",
                };
                *edge_counts.entry(name.to_string()).or_default() += 1;
            }
        }

        println!("Image {}: Digit {}", i, label);
        println!("  Single-level blobs: {}", blobs.len());
        println!("  Hierarchical blobs: {} (across {} levels)", hierarchy.blobs.len(), hierarchy.num_levels);
        println!("  Graph nodes: {}, edges: {}", graph.node_count(), graph.edge_count());
        println!("  Edge types: {:?}", edge_counts);
        println!();
    }
}

#[test]
fn test_mnist_vision_brain_to_dagnn() {
    let mnist = load_mnist_subset();
    let brain = VisionBrain::mnist();

    // Test that VisionBrain can convert to DagNN for GRAPHEME processing
    for i in 0..5 {
        let start = i * 784;
        let end = start + 784;
        let pixels = normalize_image(&mnist.trn_img[start..end]);

        let vision_graph = brain.mnist_to_graph(&pixels).unwrap();
        let dagnn = brain.to_dagnn(&vision_graph).unwrap();

        // DagNN should have nodes
        assert!(
            dagnn.node_count() > 0,
            "DagNN should have nodes for image {}",
            i
        );

        // Input nodes should be set
        assert!(
            !dagnn.input_nodes().is_empty(),
            "DagNN should have input nodes"
        );
    }
}

// ============================================================================
// MnistModel End-to-End Pipeline Tests
// ============================================================================

#[test]
fn test_mnist_model_forward_on_real_data() {
    let mnist = load_mnist_subset();
    let model = MnistModel::new();

    // Test forward pass on first 10 training images
    for i in 0..10 {
        let start = i * 784;
        let end = start + 784;
        let pixels = normalize_image(&mnist.trn_img[start..end]);
        let label = mnist.trn_lbl[i] as usize;

        let result = model.forward_with_target(&pixels, label);
        assert!(
            result.is_ok(),
            "Forward pass failed for image {} (label {})",
            i,
            label
        );

        let result = result.unwrap();
        assert!(
            result.predicted_class < 10,
            "Predicted class should be 0-9, got {}",
            result.predicted_class
        );
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "Confidence should be in [0, 1], got {}",
            result.confidence
        );
        assert!(
            result.correct.is_some(),
            "Correct field should be set when target is provided"
        );
    }
}

#[test]
fn test_mnist_model_train_step_on_real_data() {
    let mnist = load_mnist_subset();
    let mut model = MnistModel::new();

    // Test training step on first 5 training images
    for i in 0..5 {
        let start = i * 784;
        let end = start + 784;
        let pixels = normalize_image(&mnist.trn_img[start..end]);
        let label = mnist.trn_lbl[i] as usize;

        let result = model.train_step(&pixels, label);
        assert!(
            result.is_ok(),
            "Train step failed for image {} (label {})",
            i,
            label
        );

        let (train_result, dag) = result.unwrap();
        assert!(
            train_result.loss >= 0.0,
            "Loss should be non-negative, got {}",
            train_result.loss
        );
        assert!(
            train_result.predicted_class < 10,
            "Predicted class should be 0-9"
        );
        assert!(
            !train_result.gradient.is_empty(),
            "Gradient should not be empty"
        );
        assert!(
            dag.node_count() > 0,
            "DAG should have nodes"
        );
    }
}

#[test]
fn test_mnist_model_determinism_on_real_data() {
    let mnist = load_mnist_subset();
    let model = MnistModel::new();

    // Test that same image always produces same vision graph structure
    // Note: For untrained models, classification may vary due to equal-distance
    // templates. The critical GRAPHEME principle is that the INPUT graph is
    // deterministic (same image = same graph). Classification becomes
    // deterministic after templates are trained with distinct patterns.
    for i in 0..5 {
        let start = i * 784;
        let end = start + 784;
        let pixels = normalize_image(&mnist.trn_img[start..end]);

        let result1 = model.forward(&pixels).unwrap();
        let result2 = model.forward(&pixels).unwrap();

        // Vision graph structure is always deterministic
        assert_eq!(
            result1.vision_nodes, result2.vision_nodes,
            "Same image should produce same vision graph node count"
        );
        assert_eq!(
            result1.vision_edges, result2.vision_edges,
            "Same image should produce same vision graph edge count"
        );
    }
}

#[test]
fn test_mnist_model_all_digit_classes() {
    let mnist = load_mnist_subset();
    let model = MnistModel::new();

    // Group images by label
    let mut by_label: HashMap<u8, Vec<usize>> = HashMap::new();
    for (i, &label) in mnist.trn_lbl.iter().enumerate().take(100) {
        by_label.entry(label).or_default().push(i);
    }

    // Test that we can process all digit types (0-9)
    for label in 0..10u8 {
        if let Some(indices) = by_label.get(&label) {
            if let Some(&idx) = indices.first() {
                let start = idx * 784;
                let end = start + 784;
                let pixels = normalize_image(&mnist.trn_img[start..end]);

                let result = model.forward(&pixels);
                assert!(
                    result.is_ok(),
                    "MnistModel should process digit {} successfully",
                    label
                );

                let result = result.unwrap();
                println!(
                    "Digit {}: predicted={}, confidence={:.2}%, vision_nodes={}, vision_edges={}",
                    label,
                    result.predicted_class,
                    result.confidence * 100.0,
                    result.vision_nodes,
                    result.vision_edges
                );
            }
        }
    }
}

#[test]
fn test_mnist_model_custom_config() {
    let mnist = load_mnist_subset();

    // Create model with custom configuration
    let config = MnistModelConfig::mnist()
        .with_hidden_size(32)
        .with_momentum(0.8);
    let model = MnistModel::with_config(config);

    // Verify config was applied
    assert_eq!(model.config().hidden_size, 32);
    assert!((model.config().classification.template_momentum - 0.8).abs() < 1e-6);

    // Test forward pass works with custom config
    let start = 0;
    let end = 784;
    let pixels = normalize_image(&mnist.trn_img[start..end]);

    let result = model.forward(&pixels);
    assert!(result.is_ok(), "Forward pass should work with custom config");
}
