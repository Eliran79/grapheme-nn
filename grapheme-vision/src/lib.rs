//! # grapheme-vision
//!
//! Vision Brain: Image-to-graph embedding for GRAPHEME (no CNN).
//!
//! This crate provides:
//! - `RawImage` - Raw image representation (any size, grayscale or RGB)
//! - `VisionBrain` - Hierarchical feature extraction (blob detection, edge detection)
//! - `ClassificationBrain` - Output graph to class label conversion
//! - Deterministic: same image = same graph
//!
//! ## GRAPHEME Vision Principle
//!
//! ```text
//! Image → VisionBrain → Input Graph → GRAPHEME Core → Output Graph → ClassificationBrain → Class
//!       (deterministic)                  (learns)                    (structural matching)
//! ```
//!
//! No CNN. No learned feature extraction. Pure signal processing to graph.

use grapheme_brain_common::{ActivatedNode, BaseDomainBrain, DomainConfig, TextNormalizer};
use grapheme_core::{
    DagNN, DomainBrain, DomainExample, DomainResult, DomainRule, ExecutionResult, ValidationIssue,
};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
// HashSet removed - not currently needed
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Errors in vision graph processing
#[derive(Error, Debug)]
pub enum VisionError {
    #[error("Invalid image dimensions: {0}x{1}")]
    InvalidDimensions(usize, usize),
    #[error("Pixel count mismatch: expected {expected}, got {actual}")]
    PixelCountMismatch { expected: usize, actual: usize },
    #[error("Invalid channel count: {0} (expected 1 or 3)")]
    InvalidChannels(usize),
    #[error("Empty image")]
    EmptyImage,
    #[error("Feature extraction error: {0}")]
    FeatureError(String),
}

/// Result type for vision operations
pub type VisionResult<T> = Result<T, VisionError>;

// ============================================================================
// Raw Image (Simple BMP-like format)
// ============================================================================

/// Raw image - just pixels and dimensions.
///
/// This is the universal input format for VisionBrain.
/// No compression, no color profiles, just raw pixel values.
///
/// # Example
/// ```
/// use grapheme_vision::RawImage;
///
/// // Create a 28x28 grayscale image (like MNIST)
/// let pixels = vec![0.0f32; 28 * 28];
/// let image = RawImage::grayscale(28, 28, pixels).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawImage {
    /// Pixel values normalized to 0.0-1.0
    pub pixels: Vec<f32>,
    /// Image width in pixels
    pub width: usize,
    /// Image height in pixels
    pub height: usize,
    /// Number of channels (1=grayscale, 3=RGB)
    pub channels: usize,
}

impl RawImage {
    /// Create a grayscale image (1 channel)
    pub fn grayscale(width: usize, height: usize, pixels: Vec<f32>) -> VisionResult<Self> {
        let expected = width * height;
        if pixels.len() != expected {
            return Err(VisionError::PixelCountMismatch {
                expected,
                actual: pixels.len(),
            });
        }
        if width == 0 || height == 0 {
            return Err(VisionError::InvalidDimensions(width, height));
        }
        Ok(Self {
            pixels,
            width,
            height,
            channels: 1,
        })
    }

    /// Create an RGB image (3 channels)
    pub fn rgb(width: usize, height: usize, pixels: Vec<f32>) -> VisionResult<Self> {
        let expected = width * height * 3;
        if pixels.len() != expected {
            return Err(VisionError::PixelCountMismatch {
                expected,
                actual: pixels.len(),
            });
        }
        if width == 0 || height == 0 {
            return Err(VisionError::InvalidDimensions(width, height));
        }
        Ok(Self {
            pixels,
            width,
            height,
            channels: 3,
        })
    }

    /// Create from MNIST format (784 pixels, 28x28)
    pub fn from_mnist(pixels: &[f32]) -> VisionResult<Self> {
        if pixels.len() != 784 {
            return Err(VisionError::PixelCountMismatch {
                expected: 784,
                actual: pixels.len(),
            });
        }
        Self::grayscale(28, 28, pixels.to_vec())
    }

    /// Get pixel value at (x, y) for grayscale, or (x, y, channel) for RGB
    pub fn get_pixel(&self, x: usize, y: usize) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = y * self.width + x;
        if self.channels == 1 {
            self.pixels.get(idx).copied().unwrap_or(0.0)
        } else {
            // For RGB, return grayscale average
            let base = idx * 3;
            let r = self.pixels.get(base).copied().unwrap_or(0.0);
            let g = self.pixels.get(base + 1).copied().unwrap_or(0.0);
            let b = self.pixels.get(base + 2).copied().unwrap_or(0.0);
            (r + g + b) / 3.0
        }
    }

    /// Convert to grayscale if RGB
    pub fn to_grayscale(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }

        let mut gray_pixels = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                gray_pixels.push(self.get_pixel(x, y));
            }
        }

        Self {
            pixels: gray_pixels,
            width: self.width,
            height: self.height,
            channels: 1,
        }
    }

    /// Get total pixel count
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

// ============================================================================
// Vision Node Types
// ============================================================================

/// Vision-specific node types for the image graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VisionNodeType {
    /// A detected blob/region
    Blob {
        /// Center x coordinate (normalized 0-1)
        cx: f32,
        /// Center y coordinate (normalized 0-1)
        cy: f32,
        /// Blob size (pixel count)
        size: usize,
        /// Average intensity
        intensity: f32,
    },
    /// An edge/contour point
    Edge {
        /// X coordinate (normalized 0-1)
        x: f32,
        /// Y coordinate (normalized 0-1)
        y: f32,
        /// Edge strength
        strength: f32,
    },
    /// A corner/keypoint
    Corner {
        /// X coordinate (normalized 0-1)
        x: f32,
        /// Y coordinate (normalized 0-1)
        y: f32,
        /// Corner response
        response: f32,
    },
    /// Hierarchical region (groups of blobs)
    Region {
        /// Child blob indices
        children: Vec<usize>,
        /// Region level in hierarchy
        level: usize,
    },
    /// Root node for the image
    ImageRoot {
        width: usize,
        height: usize,
    },
}

/// Get default activation based on vision node type
pub fn vision_type_activation(node_type: &VisionNodeType) -> f32 {
    match node_type {
        VisionNodeType::Blob { intensity, .. } => *intensity,
        VisionNodeType::Edge { strength, .. } => *strength,
        VisionNodeType::Corner { response, .. } => *response,
        VisionNodeType::Region { .. } => 0.8,
        VisionNodeType::ImageRoot { .. } => 1.0,
    }
}

/// A vision node with activation
pub type VisionNode = ActivatedNode<VisionNodeType>;

/// Create a new vision node with activation based on type
pub fn new_vision_node(node_type: VisionNodeType) -> VisionNode {
    ActivatedNode::with_type_activation(node_type, vision_type_activation)
}

// ============================================================================
// Vision Edge Types
// ============================================================================

/// Edge types in vision graphs
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VisionEdge {
    /// Spatial adjacency (blobs touching/near each other)
    Adjacent,
    /// Containment (parent region contains child)
    Contains,
    /// Same contour (edge points on same boundary)
    SameContour,
    /// Hierarchical (lower level to higher level)
    Hierarchy,
    /// Directional: source is above target
    Above,
    /// Directional: source is below target
    Below,
    /// Directional: source is left of target
    LeftOf,
    /// Directional: source is right of target
    RightOf,
    /// Proximity with distance (normalized 0.0-1.0)
    Proximity(f32),
}

// Note: Eq is implemented manually for VisionEdge because Proximity contains f32
// Two Proximity edges are considered equal if their distances are within epsilon
impl Eq for VisionEdge {}

impl std::hash::Hash for VisionEdge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        if let VisionEdge::Proximity(d) = self {
            // Quantize to 3 decimal places for consistent hashing
            ((d * 1000.0).round() as i32).hash(state);
        }
    }
}

/// Spatial relationship between two blobs
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialRelationship {
    /// Horizontal direction (-1.0 = left, 1.0 = right)
    pub dx: f32,
    /// Vertical direction (-1.0 = up, 1.0 = down)
    pub dy: f32,
    /// Distance (normalized 0.0-1.0 relative to image diagonal)
    pub distance: f32,
    /// Angle in radians (0 = right, π/2 = down)
    pub angle: f32,
}

impl SpatialRelationship {
    /// Compute spatial relationship between two blobs
    pub fn from_blobs(a: &Blob, b: &Blob, image_width: usize, image_height: usize) -> Self {
        let (ax, ay) = a.center;
        let (bx, by) = b.center;

        let dx = bx - ax;
        let dy = by - ay;

        // Normalize distance to image diagonal
        let diag = ((image_width * image_width + image_height * image_height) as f32).sqrt();
        let dist = (dx * dx + dy * dy).sqrt();
        let distance = (dist / diag).min(1.0);

        // Compute angle (0 = right, π/2 = down, π = left, -π/2 = up)
        let angle = dy.atan2(dx);

        Self {
            dx: dx / image_width as f32,
            dy: dy / image_height as f32,
            distance,
            angle,
        }
    }

    /// Check if source is predominantly above target
    pub fn is_above(&self) -> bool {
        self.dy < -0.1 && self.dy.abs() > self.dx.abs()
    }

    /// Check if source is predominantly below target
    pub fn is_below(&self) -> bool {
        self.dy > 0.1 && self.dy.abs() > self.dx.abs()
    }

    /// Check if source is predominantly left of target
    pub fn is_left_of(&self) -> bool {
        self.dx < -0.1 && self.dx.abs() > self.dy.abs()
    }

    /// Check if source is predominantly right of target
    pub fn is_right_of(&self) -> bool {
        self.dx > 0.1 && self.dx.abs() > self.dy.abs()
    }

    /// Get the primary directional edge type
    pub fn primary_direction(&self) -> Option<VisionEdge> {
        if self.is_above() {
            Some(VisionEdge::Above)
        } else if self.is_below() {
            Some(VisionEdge::Below)
        } else if self.is_left_of() {
            Some(VisionEdge::LeftOf)
        } else if self.is_right_of() {
            Some(VisionEdge::RightOf)
        } else {
            None // Too close to determine direction
        }
    }
}

// ============================================================================
// Vision Graph
// ============================================================================

/// An image represented as a graph
#[derive(Debug)]
pub struct VisionGraph {
    /// The underlying directed graph
    pub graph: DiGraph<VisionNode, VisionEdge>,
    /// Root node index
    pub root: Option<NodeIndex>,
    /// Source image dimensions
    pub width: usize,
    pub height: usize,
}

impl Default for VisionGraph {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl VisionGraph {
    /// Create a new empty vision graph
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            graph: DiGraph::new(),
            root: None,
            width,
            height,
        }
    }

    /// Add a node
    pub fn add_node(&mut self, node: VisionNode) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Add an edge
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: VisionEdge) {
        self.graph.add_edge(from, to, edge);
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

// ============================================================================
// Feature Extraction (No CNN - Pure Signal Processing)
// ============================================================================

/// Configuration for feature extraction
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Threshold for blob detection (pixels above this are foreground)
    pub blob_threshold: f32,
    /// Minimum blob size in pixels
    pub min_blob_size: usize,
    /// Maximum number of blobs to extract
    pub max_blobs: usize,
    /// Enable edge detection
    pub detect_edges: bool,
    /// Edge detection threshold
    pub edge_threshold: f32,
    /// Enable corner detection
    pub detect_corners: bool,
    /// Build hierarchical regions
    pub build_hierarchy: bool,
    /// Maximum hierarchy levels
    pub max_hierarchy_levels: usize,
    /// Threshold for spatial adjacency (normalized distance 0.0-1.0)
    pub adjacency_threshold: f32,
    /// Enable rich spatial relationships (directional edges, proximity)
    pub build_spatial_graph: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            blob_threshold: 0.3,
            min_blob_size: 4,
            max_blobs: 100,
            detect_edges: false,
            edge_threshold: 0.2,
            detect_corners: false,
            build_hierarchy: true,
            max_hierarchy_levels: 3,
            adjacency_threshold: 0.15,
            build_spatial_graph: true,
        }
    }
}

impl FeatureConfig {
    /// Config optimized for MNIST digits
    pub fn mnist() -> Self {
        Self {
            blob_threshold: 0.2,
            min_blob_size: 3,
            max_blobs: 50,
            detect_edges: false,
            edge_threshold: 0.3,
            detect_corners: false,
            build_hierarchy: true,
            max_hierarchy_levels: 2,
            adjacency_threshold: 0.2,
            build_spatial_graph: true,
        }
    }
}

/// A detected blob (connected component)
#[derive(Debug, Clone)]
pub struct Blob {
    /// Pixel coordinates (x, y)
    pub pixels: Vec<(usize, usize)>,
    /// Center of mass
    pub center: (f32, f32),
    /// Average intensity
    pub intensity: f32,
    /// Bounding box (x, y, w, h)
    pub bbox: (usize, usize, usize, usize),
}

/// A hierarchical blob with parent-child relationships
#[derive(Debug, Clone)]
pub struct HierarchicalBlob {
    /// The blob data
    pub blob: Blob,
    /// Hierarchy level (0 = finest, higher = coarser)
    pub level: usize,
    /// Indices of child blobs (finer scale)
    pub children: Vec<usize>,
    /// Index of parent blob (coarser scale), if any
    pub parent: Option<usize>,
    /// Scale at which this blob was detected
    pub scale: f32,
}

impl HierarchicalBlob {
    /// Create a new hierarchical blob at a given level
    pub fn new(blob: Blob, level: usize, scale: f32) -> Self {
        Self {
            blob,
            level,
            children: Vec::new(),
            parent: None,
            scale,
        }
    }

    /// Check if this blob contains another blob
    pub fn contains(&self, other: &Blob) -> bool {
        // Check if other's center is within this blob's bounding box
        let (bx, by, bw, bh) = self.blob.bbox;
        let (ox, oy) = other.center;
        ox >= bx as f32 && ox < (bx + bw) as f32 &&
        oy >= by as f32 && oy < (by + bh) as f32
    }
}

/// Result of hierarchical blob detection
#[derive(Debug, Clone)]
pub struct BlobHierarchy {
    /// All blobs across all levels
    pub blobs: Vec<HierarchicalBlob>,
    /// Number of hierarchy levels
    pub num_levels: usize,
    /// Indices of root blobs (no parent)
    pub roots: Vec<usize>,
}

/// Extract blobs (connected components) from image
pub fn extract_blobs(image: &RawImage, config: &FeatureConfig) -> Vec<Blob> {
    let gray = image.to_grayscale();
    let mut visited = vec![false; gray.pixel_count()];
    let mut blobs = Vec::new();

    for y in 0..gray.height {
        for x in 0..gray.width {
            let idx = y * gray.width + x;
            if visited[idx] || gray.pixels[idx] < config.blob_threshold {
                continue;
            }

            // Flood fill to find connected component
            let mut pixels = Vec::new();
            let mut stack = vec![(x, y)];
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sum_intensity = 0.0f32;
            let mut min_x = x;
            let mut max_x = x;
            let mut min_y = y;
            let mut max_y = y;

            while let Some((px, py)) = stack.pop() {
                let pidx = py * gray.width + px;
                if visited[pidx] || gray.pixels[pidx] < config.blob_threshold {
                    continue;
                }
                visited[pidx] = true;

                let intensity = gray.pixels[pidx];
                pixels.push((px, py));
                sum_x += px as f32 * intensity;
                sum_y += py as f32 * intensity;
                sum_intensity += intensity;

                min_x = min_x.min(px);
                max_x = max_x.max(px);
                min_y = min_y.min(py);
                max_y = max_y.max(py);

                // 4-connectivity neighbors
                if px > 0 {
                    stack.push((px - 1, py));
                }
                if px + 1 < gray.width {
                    stack.push((px + 1, py));
                }
                if py > 0 {
                    stack.push((px, py - 1));
                }
                if py + 1 < gray.height {
                    stack.push((px, py + 1));
                }
            }

            if pixels.len() >= config.min_blob_size {
                let center = if sum_intensity > 0.0 {
                    (sum_x / sum_intensity, sum_y / sum_intensity)
                } else {
                    (
                        pixels.iter().map(|p| p.0).sum::<usize>() as f32 / pixels.len() as f32,
                        pixels.iter().map(|p| p.1).sum::<usize>() as f32 / pixels.len() as f32,
                    )
                };

                blobs.push(Blob {
                    intensity: sum_intensity / pixels.len() as f32,
                    center,
                    bbox: (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                    pixels,
                });

                if blobs.len() >= config.max_blobs {
                    break;
                }
            }
        }
        if blobs.len() >= config.max_blobs {
            break;
        }
    }

    // Sort by size (largest first)
    blobs.sort_by(|a, b| b.pixels.len().cmp(&a.pixels.len()));
    blobs
}

/// Extract blobs at a specific threshold
fn extract_blobs_at_threshold(image: &RawImage, threshold: f32, min_size: usize, max_blobs: usize) -> Vec<Blob> {
    let gray = image.to_grayscale();
    let mut visited = vec![false; gray.pixel_count()];
    let mut blobs = Vec::new();

    for y in 0..gray.height {
        for x in 0..gray.width {
            let idx = y * gray.width + x;
            if visited[idx] || gray.pixels[idx] < threshold {
                continue;
            }

            // Flood fill to find connected component
            let mut pixels = Vec::new();
            let mut stack = vec![(x, y)];
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sum_intensity = 0.0f32;
            let mut min_x = x;
            let mut max_x = x;
            let mut min_y = y;
            let mut max_y = y;

            while let Some((px, py)) = stack.pop() {
                let pidx = py * gray.width + px;
                if visited[pidx] || gray.pixels[pidx] < threshold {
                    continue;
                }
                visited[pidx] = true;

                let intensity = gray.pixels[pidx];
                pixels.push((px, py));
                sum_x += px as f32 * intensity;
                sum_y += py as f32 * intensity;
                sum_intensity += intensity;

                min_x = min_x.min(px);
                max_x = max_x.max(px);
                min_y = min_y.min(py);
                max_y = max_y.max(py);

                // 4-connectivity neighbors
                if px > 0 {
                    stack.push((px - 1, py));
                }
                if px + 1 < gray.width {
                    stack.push((px + 1, py));
                }
                if py > 0 {
                    stack.push((px, py - 1));
                }
                if py + 1 < gray.height {
                    stack.push((px, py + 1));
                }
            }

            if pixels.len() >= min_size {
                let center = if sum_intensity > 0.0 {
                    (sum_x / sum_intensity, sum_y / sum_intensity)
                } else {
                    (
                        pixels.iter().map(|p| p.0).sum::<usize>() as f32 / pixels.len() as f32,
                        pixels.iter().map(|p| p.1).sum::<usize>() as f32 / pixels.len() as f32,
                    )
                };

                blobs.push(Blob {
                    intensity: sum_intensity / pixels.len() as f32,
                    center,
                    bbox: (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                    pixels,
                });

                if blobs.len() >= max_blobs {
                    break;
                }
            }
        }
        if blobs.len() >= max_blobs {
            break;
        }
    }

    blobs.sort_by(|a, b| b.pixels.len().cmp(&a.pixels.len()));
    blobs
}

/// Extract hierarchical blobs at multiple scales.
///
/// This function detects blobs at multiple intensity thresholds to create
/// a hierarchical structure. Coarser scales (lower thresholds) capture
/// larger structures, while finer scales (higher thresholds) capture details.
///
/// # Algorithm
/// 1. Detect blobs at multiple thresholds (coarse → fine)
/// 2. Link child blobs to parent blobs based on spatial containment
/// 3. Return hierarchy with parent-child relationships
///
/// # Arguments
/// * `image` - The input image
/// * `config` - Feature configuration (uses max_hierarchy_levels)
///
/// # Returns
/// A BlobHierarchy with all detected blobs and their relationships
pub fn extract_hierarchical_blobs(image: &RawImage, config: &FeatureConfig) -> BlobHierarchy {
    let num_levels = config.max_hierarchy_levels.max(1);
    let mut all_blobs: Vec<HierarchicalBlob> = Vec::new();

    // Generate threshold levels from coarse (low) to fine (high)
    // Coarse = low threshold (more permissive, larger blobs)
    // Fine = high threshold (more selective, smaller blobs)
    let base_threshold = config.blob_threshold;
    let threshold_step = (1.0 - base_threshold) / (num_levels as f32 + 1.0);

    for level in 0..num_levels {
        // Level 0 = finest (highest threshold), Level N-1 = coarsest (lowest)
        let threshold = base_threshold + threshold_step * ((num_levels - level - 1) as f32);
        let scale = 1.0 / (level as f32 + 1.0);

        // Adjust min_blob_size for scale (larger at coarser scales)
        let min_size = (config.min_blob_size as f32 * (level as f32 + 1.0).sqrt()) as usize;
        let max_blobs = config.max_blobs / num_levels.max(1);

        let blobs = extract_blobs_at_threshold(image, threshold, min_size, max_blobs.max(10));

        let level_start_idx = all_blobs.len();
        for blob in blobs {
            all_blobs.push(HierarchicalBlob::new(blob, level, scale));
        }

        // Link to parent level (coarser, higher level number)
        if level > 0 {
            let parent_level = level - 1;
            // Find parent blobs from the coarser level
            for child_idx in level_start_idx..all_blobs.len() {
                let child_center = all_blobs[child_idx].blob.center;

                // Find best parent (smallest containing blob at parent level)
                let mut best_parent: Option<(usize, usize)> = None; // (index, size)

                for (parent_idx, parent_blob) in all_blobs.iter().enumerate() {
                    if parent_blob.level != parent_level {
                        continue;
                    }

                    // Check if parent contains child's center
                    let (px, py, pw, ph) = parent_blob.blob.bbox;
                    let (cx, cy) = child_center;

                    if cx >= px as f32 && cx < (px + pw) as f32 &&
                       cy >= py as f32 && cy < (py + ph) as f32 {
                        let parent_size = parent_blob.blob.pixels.len();
                        if best_parent.is_none() || parent_size < best_parent.unwrap().1 {
                            best_parent = Some((parent_idx, parent_size));
                        }
                    }
                }

                if let Some((parent_idx, _)) = best_parent {
                    all_blobs[child_idx].parent = Some(parent_idx);
                    all_blobs[parent_idx].children.push(child_idx);
                }
            }
        }
    }

    // Find root blobs (no parent)
    let roots: Vec<usize> = all_blobs
        .iter()
        .enumerate()
        .filter(|(_, b)| b.parent.is_none())
        .map(|(i, _)| i)
        .collect();

    BlobHierarchy {
        blobs: all_blobs,
        num_levels,
        roots,
    }
}

/// Compute all spatial relationships between blobs.
///
/// Returns a vector of (source_idx, target_idx, relationship) tuples
/// for all pairs of blobs within the proximity threshold.
pub fn compute_spatial_relationships(
    blobs: &[Blob],
    image_width: usize,
    image_height: usize,
    max_distance: f32,
) -> Vec<(usize, usize, SpatialRelationship)> {
    let mut relationships = Vec::new();

    for i in 0..blobs.len() {
        for j in (i + 1)..blobs.len() {
            let rel = SpatialRelationship::from_blobs(&blobs[i], &blobs[j], image_width, image_height);

            // Only include relationships within threshold
            if rel.distance <= max_distance {
                relationships.push((i, j, rel));
            }
        }
    }

    relationships
}

/// Build a complete spatial relationship graph from blobs.
///
/// This creates edges for:
/// - Adjacency (blobs touching)
/// - Directional relationships (above, below, left, right)
/// - Proximity with distance
pub fn build_spatial_graph(
    graph: &mut VisionGraph,
    blobs: &[Blob],
    blob_nodes: &[NodeIndex],
    config: &FeatureConfig,
) {
    let max_distance = config.adjacency_threshold.max(0.5); // At least 50% of diagonal

    let relationships = compute_spatial_relationships(
        blobs,
        graph.width,
        graph.height,
        max_distance,
    );

    for (i, j, rel) in relationships {
        if i >= blob_nodes.len() || j >= blob_nodes.len() {
            continue;
        }

        let node_i = blob_nodes[i];
        let node_j = blob_nodes[j];

        // Add adjacency edge if blobs are touching
        if blobs_adjacent(&blobs[i], &blobs[j]) {
            graph.add_edge(node_i, node_j, VisionEdge::Adjacent);
            graph.add_edge(node_j, node_i, VisionEdge::Adjacent);
        }

        // Add directional edges
        if let Some(dir) = rel.primary_direction() {
            graph.add_edge(node_i, node_j, dir);
            // Add inverse direction
            let inverse = match dir {
                VisionEdge::Above => VisionEdge::Below,
                VisionEdge::Below => VisionEdge::Above,
                VisionEdge::LeftOf => VisionEdge::RightOf,
                VisionEdge::RightOf => VisionEdge::LeftOf,
                _ => continue,
            };
            graph.add_edge(node_j, node_i, inverse);
        }

        // Add proximity edge if within threshold
        if rel.distance <= config.adjacency_threshold {
            graph.add_edge(node_i, node_j, VisionEdge::Proximity(rel.distance));
            graph.add_edge(node_j, node_i, VisionEdge::Proximity(rel.distance));
        }
    }
}

/// Check if two blobs are adjacent (bounding boxes touch or overlap)
pub fn blobs_adjacent(a: &Blob, b: &Blob) -> bool {
    let (ax, ay, aw, ah) = a.bbox;
    let (bx, by, bw, bh) = b.bbox;

    // Expand bounding boxes by 1 pixel and check overlap
    let a_left = ax.saturating_sub(1);
    let a_right = ax + aw + 1;
    let a_top = ay.saturating_sub(1);
    let a_bottom = ay + ah + 1;

    let b_left = bx;
    let b_right = bx + bw;
    let b_top = by;
    let b_bottom = by + bh;

    !(a_right < b_left || b_right < a_left || a_bottom < b_top || b_bottom < a_top)
}

// ============================================================================
// Image to Graph Conversion
// ============================================================================

/// Convert an image to a GRAPHEME graph.
///
/// This is the core VisionBrain functionality:
/// - Deterministic: same image always produces same graph
/// - Hierarchical: multi-scale blob detection with parent-child relationships
/// - No CNN: pure signal processing
pub fn image_to_graph(image: &RawImage, config: &FeatureConfig) -> VisionResult<VisionGraph> {
    if image.pixels.is_empty() {
        return Err(VisionError::EmptyImage);
    }

    let mut graph = VisionGraph::new(image.width, image.height);
    let w = image.width as f32;
    let h = image.height as f32;

    // Create root node
    let root = graph.add_node(new_vision_node(VisionNodeType::ImageRoot {
        width: image.width,
        height: image.height,
    }));
    graph.root = Some(root);

    // Use hierarchical blob detection if enabled
    if config.build_hierarchy && config.max_hierarchy_levels > 1 {
        let hierarchy = extract_hierarchical_blobs(image, config);

        if hierarchy.blobs.is_empty() {
            return Ok(graph);
        }

        // Create nodes for all hierarchical blobs
        let mut blob_nodes: Vec<NodeIndex> = Vec::with_capacity(hierarchy.blobs.len());

        for hblob in &hierarchy.blobs {
            let blob = &hblob.blob;
            let node = graph.add_node(new_vision_node(VisionNodeType::Blob {
                cx: blob.center.0 / w,
                cy: blob.center.1 / h,
                size: blob.pixels.len(),
                intensity: blob.intensity,
            }));
            blob_nodes.push(node);
        }

        // Connect root to top-level blobs (roots in hierarchy)
        for &root_idx in &hierarchy.roots {
            if root_idx < blob_nodes.len() {
                graph.add_edge(root, blob_nodes[root_idx], VisionEdge::Contains);
            }
        }

        // Connect parent-child relationships within hierarchy
        for (idx, hblob) in hierarchy.blobs.iter().enumerate() {
            // Connect to children
            for &child_idx in &hblob.children {
                if child_idx < blob_nodes.len() {
                    graph.add_edge(blob_nodes[idx], blob_nodes[child_idx], VisionEdge::Hierarchy);
                }
            }
        }

        // Build spatial relationships between same-level blobs
        if config.build_spatial_graph {
            // Group blobs by level and build spatial graphs for each level
            for level in 0..hierarchy.num_levels {
                let level_blobs: Vec<&Blob> = hierarchy.blobs
                    .iter()
                    .filter(|h| h.level == level)
                    .map(|h| &h.blob)
                    .collect();
                let level_indices: Vec<usize> = hierarchy.blobs
                    .iter()
                    .enumerate()
                    .filter(|(_, h)| h.level == level)
                    .map(|(i, _)| i)
                    .collect();

                // Build spatial relationships within this level
                if level_blobs.len() > 1 {
                    let owned_blobs: Vec<Blob> = level_blobs.iter().map(|b| (*b).clone()).collect();
                    let level_nodes: Vec<NodeIndex> = level_indices.iter()
                        .filter_map(|&i| blob_nodes.get(i).copied())
                        .collect();
                    build_spatial_graph(&mut graph, &owned_blobs, &level_nodes, config);
                }
            }
        } else {
            // Simple adjacency (old behavior)
            for i in 0..hierarchy.blobs.len() {
                for j in (i + 1)..hierarchy.blobs.len() {
                    if hierarchy.blobs[i].level == hierarchy.blobs[j].level {
                        if blobs_adjacent(&hierarchy.blobs[i].blob, &hierarchy.blobs[j].blob) {
                            graph.add_edge(blob_nodes[i], blob_nodes[j], VisionEdge::Adjacent);
                            graph.add_edge(blob_nodes[j], blob_nodes[i], VisionEdge::Adjacent);
                        }
                    }
                }
            }
        }
    } else {
        // Single-level blob detection (original behavior)
        let blobs = extract_blobs(image, config);

        if blobs.is_empty() {
            return Ok(graph);
        }

        // Create blob nodes
        let mut blob_nodes = Vec::with_capacity(blobs.len());

        for blob in &blobs {
            let node = graph.add_node(new_vision_node(VisionNodeType::Blob {
                cx: blob.center.0 / w,
                cy: blob.center.1 / h,
                size: blob.pixels.len(),
                intensity: blob.intensity,
            }));
            blob_nodes.push(node);

            // Connect blob to root
            graph.add_edge(root, node, VisionEdge::Contains);
        }

        // Build spatial relationships
        if config.build_spatial_graph && blobs.len() > 1 {
            build_spatial_graph(&mut graph, &blobs, &blob_nodes, config);
        } else {
            // Simple adjacency (old behavior)
            for i in 0..blobs.len() {
                for j in (i + 1)..blobs.len() {
                    if blobs_adjacent(&blobs[i], &blobs[j]) {
                        graph.add_edge(blob_nodes[i], blob_nodes[j], VisionEdge::Adjacent);
                        graph.add_edge(blob_nodes[j], blob_nodes[i], VisionEdge::Adjacent);
                    }
                }
            }
        }

        // Build legacy hierarchy if enabled (spatial clustering)
        if config.build_hierarchy && blobs.len() > 1 {
            build_blob_hierarchy(&mut graph, &blobs, &blob_nodes, root, config);
        }
    }

    Ok(graph)
}

/// Build hierarchical regions from blobs using spatial clustering
fn build_blob_hierarchy(
    graph: &mut VisionGraph,
    blobs: &[Blob],
    blob_nodes: &[NodeIndex],
    root: NodeIndex,
    config: &FeatureConfig,
) {
    if blobs.len() < 2 || config.max_hierarchy_levels == 0 {
        return;
    }

    // Simple spatial clustering: group blobs that are close together
    let mut used = vec![false; blobs.len()];
    let mut regions: Vec<Vec<usize>> = Vec::new();

    // Distance threshold for grouping (relative to image size)
    let threshold = 0.2; // 20% of image diagonal

    for i in 0..blobs.len() {
        if used[i] {
            continue;
        }

        let mut region = vec![i];
        used[i] = true;

        for j in (i + 1)..blobs.len() {
            if used[j] {
                continue;
            }

            // Check distance between blob centers
            let dx = blobs[i].center.0 - blobs[j].center.0;
            let dy = blobs[i].center.1 - blobs[j].center.1;
            let dist = (dx * dx + dy * dy).sqrt()
                / (graph.width as f32 * graph.width as f32
                    + graph.height as f32 * graph.height as f32)
                    .sqrt();

            if dist < threshold {
                region.push(j);
                used[j] = true;
            }
        }

        if region.len() > 1 {
            regions.push(region);
        }
    }

    // Create region nodes
    for (level, region) in regions.iter().enumerate() {
        if level >= config.max_hierarchy_levels {
            break;
        }

        let region_node = graph.add_node(new_vision_node(VisionNodeType::Region {
            children: region.clone(),
            level: level + 1,
        }));

        // Connect root to region
        graph.add_edge(root, region_node, VisionEdge::Hierarchy);

        // Connect region to its blobs
        for &blob_idx in region {
            if blob_idx < blob_nodes.len() {
                graph.add_edge(region_node, blob_nodes[blob_idx], VisionEdge::Contains);
            }
        }
    }
}

// ============================================================================
// Vision Brain
// ============================================================================

/// Create the vision domain configuration
fn create_vision_config() -> DomainConfig {
    let keywords = vec![
        "image", "pixel", "blob", "region", "edge", "corner", "visual", "picture", "photo",
    ];

    let normalizer = TextNormalizer::new().trim_whitespace(true);

    DomainConfig::new("vision", "Computer Vision", keywords)
        .with_version("0.1.0")
        .with_normalizer(normalizer)
        .with_annotation_prefix("@vision:")
}

/// The Vision Brain for image-to-graph embedding.
///
/// Implements GRAPHEME's universal principle:
/// - Same image → Same graph (deterministic)
/// - No CNN, no learned features
/// - Hierarchical structure from signal processing
pub struct VisionBrain {
    /// Domain configuration
    config: DomainConfig,
    /// Feature extraction configuration
    feature_config: FeatureConfig,
}

impl Default for VisionBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for VisionBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionBrain")
            .field("domain", &"vision")
            .finish()
    }
}

impl VisionBrain {
    /// Create a new vision brain with default config
    pub fn new() -> Self {
        Self {
            config: create_vision_config(),
            feature_config: FeatureConfig::default(),
        }
    }

    /// Create vision brain optimized for MNIST
    pub fn mnist() -> Self {
        Self {
            config: create_vision_config(),
            feature_config: FeatureConfig::mnist(),
        }
    }

    /// Set feature extraction configuration
    pub fn with_feature_config(mut self, config: FeatureConfig) -> Self {
        self.feature_config = config;
        self
    }

    /// Convert image to graph (core VisionBrain operation)
    ///
    /// This is deterministic: same image always produces same graph.
    pub fn to_graph(&self, image: &RawImage) -> VisionResult<VisionGraph> {
        image_to_graph(image, &self.feature_config)
    }

    /// Convert MNIST pixels to graph
    pub fn mnist_to_graph(&self, pixels: &[f32]) -> VisionResult<VisionGraph> {
        let image = RawImage::from_mnist(pixels)?;
        self.to_graph(&image)
    }

    /// Convert VisionGraph to DagNN for GRAPHEME core processing
    ///
    /// This converts the blob-based vision graph into a DagNN using from_image
    /// with blob-weighted pixel values. Blobs that are detected become
    /// high-activation regions.
    pub fn to_dagnn(&self, vision_graph: &VisionGraph) -> DomainResult<DagNN> {
        // Get image dimensions from the root node
        let (width, height) = vision_graph.graph.node_weights()
            .find_map(|n| match &n.node_type {
                VisionNodeType::ImageRoot { width, height } => Some((*width, *height)),
                _ => None,
            })
            .unwrap_or((28, 28)); // Default to MNIST dimensions

        // Create a pixel array with blob activations
        // Blobs contribute their intensity to their center position
        let mut pixels = vec![0.0f32; width * height];

        for node in vision_graph.graph.node_weights() {
            if let VisionNodeType::Blob { cx, cy, size, intensity } = &node.node_type {
                // Convert normalized coords back to pixel coords
                let px = (*cx * width as f32).round() as usize;
                let py = (*cy * height as f32).round() as usize;

                if px < width && py < height {
                    let idx = py * width + px;
                    // Combine intensity with size for richer activation
                    let activation = intensity * (1.0 + (*size as f32).log2().max(0.0) / 10.0);
                    pixels[idx] = pixels[idx].max(activation.clamp(0.0, 1.0));
                }
            }
        }

        // Use DagNN's from_image to create proper structure with input nodes
        DagNN::from_image(&pixels, width, height)
            .map_err(|e| grapheme_core::DomainError::InvalidInput(e.to_string()))
    }
}

// ============================================================================
// BaseDomainBrain Implementation
// ============================================================================

impl BaseDomainBrain for VisionBrain {
    fn config(&self) -> &DomainConfig {
        &self.config
    }
}

// ============================================================================
// DomainBrain Implementation
// ============================================================================

impl DomainBrain for VisionBrain {
    fn domain_id(&self) -> &str {
        &self.config.domain_id
    }

    fn domain_name(&self) -> &str {
        &self.config.domain_name
    }

    fn version(&self) -> &str {
        &self.config.version
    }

    fn can_process(&self, input: &str) -> bool {
        self.default_can_process(input)
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        self.default_parse(input)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        self.default_from_core(graph)
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        self.default_to_core(graph)
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        self.default_validate(graph)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        self.default_execute(graph)
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        vec![
            DomainRule {
                id: 0,
                domain: "vision".to_string(),
                name: "Blob Detection".to_string(),
                description: "Extract connected components from image".to_string(),
                category: "feature".to_string(),
            },
            DomainRule {
                id: 1,
                domain: "vision".to_string(),
                name: "Spatial Grouping".to_string(),
                description: "Group nearby blobs into regions".to_string(),
                category: "hierarchy".to_string(),
            },
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        // Vision transforms are applied during to_graph, not as separate rules
        match rule_id {
            0 | 1 => Ok(graph.clone()),
            _ => Err(grapheme_core::DomainError::InvalidInput(format!(
                "Unknown rule ID: {}",
                rule_id
            ))),
        }
    }

    fn generate_examples(&self, _count: usize) -> Vec<DomainExample> {
        // Vision examples would be image-graph pairs
        // For now, return empty (images are loaded externally)
        Vec::new()
    }
}

// ============================================================================
// ClassificationBrain - Output graph to class label conversion
// ============================================================================

/// Configuration for classification brain.
#[derive(Debug, Clone)]
pub struct ClassificationConfig {
    /// Number of classes (e.g., 10 for MNIST digits)
    pub num_classes: usize,
    /// Number of output nodes in the GRAPHEME graph
    pub num_outputs: usize,
    /// Template update momentum (higher = slower adaptation)
    pub template_momentum: f32,
    /// Whether to use structural loss (vs cross-entropy)
    pub use_structural: bool,
}

impl ClassificationConfig {
    /// Create MNIST configuration (10 classes)
    pub fn mnist() -> Self {
        Self {
            num_classes: 10,
            num_outputs: 10,
            template_momentum: 0.9,
            use_structural: true,
        }
    }

    /// Create custom configuration
    pub fn new(num_classes: usize, num_outputs: usize) -> Self {
        Self {
            num_classes,
            num_outputs,
            template_momentum: 0.9,
            use_structural: true,
        }
    }

    /// Set template momentum
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.template_momentum = momentum;
        self
    }

    /// Enable/disable structural loss
    pub fn with_structural(mut self, use_structural: bool) -> Self {
        self.use_structural = use_structural;
        self
    }
}

impl Default for ClassificationConfig {
    fn default() -> Self {
        Self::mnist()
    }
}

/// Classification result from ClassificationBrain.
#[derive(Debug, Clone)]
pub struct ClassificationOutput {
    /// Predicted class index
    pub predicted_class: usize,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Probabilities for each class
    pub probabilities: Vec<f32>,
    /// Optional: class label string
    pub label: Option<String>,
}

impl ClassificationOutput {
    /// Create a new classification output
    pub fn new(predicted_class: usize, confidence: f32, probabilities: Vec<f32>) -> Self {
        Self {
            predicted_class,
            confidence,
            probabilities,
            label: None,
        }
    }

    /// Add label for the predicted class
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// ClassificationBrain converts GRAPHEME output graphs to class labels.
///
/// This brain implements the output side of the MNIST pipeline:
/// ```text
/// VisionBrain → Input Graph → GRAPHEME Core → Output Graph → ClassificationBrain → Class
/// ```
///
/// Uses StructuralClassifier from grapheme-core for template-based classification
/// instead of softmax/cross-entropy (GRAPHEME-native approach).
#[derive(Debug)]
pub struct ClassificationBrain {
    config: DomainConfig,
    classification_config: ClassificationConfig,
    classifier: grapheme_core::StructuralClassifier,
    /// Optional class labels (e.g., ["0", "1", ..., "9"] for MNIST)
    labels: Vec<String>,
}

impl ClassificationBrain {
    /// Create a new ClassificationBrain with the given configuration.
    pub fn new(classification_config: ClassificationConfig) -> Self {
        let classifier = grapheme_core::StructuralClassifier::new(
            classification_config.num_classes,
            classification_config.num_outputs,
        ).with_momentum(classification_config.template_momentum);

        // Generate default numeric labels
        let labels: Vec<String> = (0..classification_config.num_classes)
            .map(|i| i.to_string())
            .collect();

        let config = DomainConfig::new(
            "classification",
            "Classification Brain",
            vec!["classify", "predict", "label", "class"],
        );

        Self {
            config,
            classification_config,
            classifier,
            labels,
        }
    }

    /// Create a ClassificationBrain for MNIST (10 digit classes).
    pub fn mnist() -> Self {
        let mut brain = Self::new(ClassificationConfig::mnist());
        brain.labels = (0..10).map(|i| i.to_string()).collect();
        brain
    }

    /// Create a ClassificationBrain with custom class labels.
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        if labels.len() == self.classification_config.num_classes {
            self.labels = labels;
        }
        self
    }

    /// Get the structural classifier (for training).
    pub fn classifier(&self) -> &grapheme_core::StructuralClassifier {
        &self.classifier
    }

    /// Get mutable access to the classifier (for template updates during training).
    pub fn classifier_mut(&mut self) -> &mut grapheme_core::StructuralClassifier {
        &mut self.classifier
    }

    /// Classify a GRAPHEME output graph.
    ///
    /// Extracts output node activations and uses structural matching
    /// to find the closest class template.
    pub fn classify(&self, graph: &DagNN) -> ClassificationOutput {
        let (predicted_class, distance) = graph.structural_classify(&self.classifier);
        let probabilities = self.classifier.distance_to_probs(&graph.get_classification_logits());

        // Convert distance to confidence (smaller distance = higher confidence)
        let confidence = (-distance).exp().min(1.0);

        let mut output = ClassificationOutput::new(predicted_class, confidence, probabilities);
        if let Some(label) = self.labels.get(predicted_class) {
            output = output.with_label(label.clone());
        }
        output
    }

    /// Get loss and gradient for training.
    ///
    /// Returns the structural loss and gradient with respect to output activations.
    pub fn loss_and_gradient(
        &self,
        graph: &DagNN,
        target_class: usize,
    ) -> grapheme_core::StructuralClassificationResult {
        graph.structural_classification_step(&self.classifier, target_class)
    }

    /// Update templates based on observed activations (call during training).
    pub fn update_templates(&mut self, activations: &[f32], true_class: usize) {
        self.classifier.update_template(true_class, activations);
    }

    /// Get the class label for an index.
    pub fn get_label(&self, class_idx: usize) -> Option<&str> {
        self.labels.get(class_idx).map(|s| s.as_str())
    }

    /// Get all class labels.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Get number of classes.
    pub fn num_classes(&self) -> usize {
        self.classification_config.num_classes
    }
}

// ============================================================================
// BaseDomainBrain Implementation for ClassificationBrain
// ============================================================================

impl BaseDomainBrain for ClassificationBrain {
    fn config(&self) -> &DomainConfig {
        &self.config
    }
}

// ============================================================================
// DomainBrain Implementation for ClassificationBrain
// ============================================================================

impl DomainBrain for ClassificationBrain {
    fn domain_id(&self) -> &str {
        &self.config.domain_id
    }

    fn domain_name(&self) -> &str {
        &self.config.domain_name
    }

    fn version(&self) -> &str {
        &self.config.version
    }

    fn can_process(&self, input: &str) -> bool {
        self.default_can_process(input)
    }

    fn parse(&self, input: &str) -> DomainResult<DagNN> {
        self.default_parse(input)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        self.default_from_core(graph)
    }

    fn to_core(&self, graph: &DagNN) -> DomainResult<DagNN> {
        self.default_to_core(graph)
    }

    fn validate(&self, graph: &DagNN) -> DomainResult<Vec<ValidationIssue>> {
        self.default_validate(graph)
    }

    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult> {
        // Classification-specific execution: return predicted class
        let result = self.classify(graph);
        Ok(ExecutionResult::Text(format!(
            "Predicted class: {} ({}), confidence: {:.2}%",
            result.predicted_class,
            result.label.unwrap_or_default(),
            result.confidence * 100.0
        )))
    }

    fn get_rules(&self) -> Vec<DomainRule> {
        vec![
            DomainRule {
                id: 0,
                domain: "classification".to_string(),
                name: "Structural Matching".to_string(),
                description: "Match output to class templates".to_string(),
                category: "classification".to_string(),
            },
        ]
    }

    fn transform(&self, graph: &DagNN, rule_id: usize) -> DomainResult<DagNN> {
        match rule_id {
            0 => Ok(graph.clone()),
            _ => Err(grapheme_core::DomainError::InvalidInput(format!(
                "Unknown rule ID: {}",
                rule_id
            ))),
        }
    }

    fn generate_examples(&self, _count: usize) -> Vec<DomainExample> {
        Vec::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_image_grayscale() {
        let pixels = vec![0.5f32; 100];
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        assert_eq!(image.width, 10);
        assert_eq!(image.height, 10);
        assert_eq!(image.channels, 1);
        assert_eq!(image.pixel_count(), 100);
    }

    #[test]
    fn test_raw_image_from_mnist() {
        let pixels = vec![0.0f32; 784];
        let image = RawImage::from_mnist(&pixels).unwrap();
        assert_eq!(image.width, 28);
        assert_eq!(image.height, 28);
    }

    #[test]
    fn test_raw_image_get_pixel() {
        let mut pixels = vec![0.0f32; 9];
        pixels[4] = 1.0; // Center pixel
        let image = RawImage::grayscale(3, 3, pixels).unwrap();
        assert!((image.get_pixel(1, 1) - 1.0).abs() < 1e-6);
        assert!((image.get_pixel(0, 0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_extract_blobs_simple() {
        // Create a simple image with one blob in the center
        let mut pixels = vec![0.0f32; 100];
        // 3x3 blob in center
        for y in 4..7 {
            for x in 4..7 {
                pixels[y * 10 + x] = 1.0;
            }
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let config = FeatureConfig::default();
        let blobs = extract_blobs(&image, &config);

        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].pixels.len(), 9);
    }

    #[test]
    fn test_extract_blobs_multiple() {
        // Create image with two separate blobs
        let mut pixels = vec![0.0f32; 100];
        // Blob 1: top-left
        for y in 0..3 {
            for x in 0..3 {
                pixels[y * 10 + x] = 1.0;
            }
        }
        // Blob 2: bottom-right
        for y in 7..10 {
            for x in 7..10 {
                pixels[y * 10 + x] = 1.0;
            }
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let config = FeatureConfig::default();
        let blobs = extract_blobs(&image, &config);

        assert_eq!(blobs.len(), 2);
    }

    #[test]
    fn test_image_to_graph() {
        let mut pixels = vec![0.0f32; 100];
        // Create a blob
        for y in 4..7 {
            for x in 4..7 {
                pixels[y * 10 + x] = 0.8;
            }
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let config = FeatureConfig::default();
        let graph = image_to_graph(&image, &config).unwrap();

        assert!(graph.root.is_some());
        assert!(graph.node_count() >= 2); // Root + at least one blob
    }

    #[test]
    fn test_vision_brain_mnist() {
        let brain = VisionBrain::mnist();
        assert_eq!(brain.domain_id(), "vision");

        // Test with blank MNIST image
        let pixels = vec![0.0f32; 784];
        let graph = brain.mnist_to_graph(&pixels).unwrap();
        assert!(graph.root.is_some());
    }

    #[test]
    fn test_vision_brain_deterministic() {
        let brain = VisionBrain::mnist();

        // Create a test image with some structure
        let mut pixels = vec![0.0f32; 784];
        for i in 100..200 {
            pixels[i] = 0.8;
        }

        // Convert twice - should produce identical graphs
        let graph1 = brain.mnist_to_graph(&pixels).unwrap();
        let graph2 = brain.mnist_to_graph(&pixels).unwrap();

        assert_eq!(graph1.node_count(), graph2.node_count());
        assert_eq!(graph1.edge_count(), graph2.edge_count());
    }

    #[test]
    fn test_vision_node_activation() {
        let blob_node = new_vision_node(VisionNodeType::Blob {
            cx: 0.5,
            cy: 0.5,
            size: 10,
            intensity: 0.7,
        });
        assert!((blob_node.activation - 0.7).abs() < 1e-6);

        let root_node = new_vision_node(VisionNodeType::ImageRoot {
            width: 28,
            height: 28,
        });
        assert!((root_node.activation - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blobs_adjacent() {
        let blob1 = Blob {
            pixels: vec![(0, 0), (1, 0), (0, 1), (1, 1)],
            center: (0.5, 0.5),
            intensity: 1.0,
            bbox: (0, 0, 2, 2),
        };
        let blob2 = Blob {
            pixels: vec![(3, 0), (4, 0)],
            center: (3.5, 0.0),
            intensity: 1.0,
            bbox: (3, 0, 2, 1),
        };
        let blob3 = Blob {
            pixels: vec![(10, 10)],
            center: (10.0, 10.0),
            intensity: 1.0,
            bbox: (10, 10, 1, 1),
        };

        assert!(blobs_adjacent(&blob1, &blob2)); // Close enough
        assert!(!blobs_adjacent(&blob1, &blob3)); // Far apart
    }

    // ========================================================================
    // Spatial Relationship Tests
    // ========================================================================

    #[test]
    fn test_spatial_relationship_from_blobs() {
        let blob_a = Blob {
            pixels: vec![(5, 5)],
            center: (5.0, 5.0),
            intensity: 1.0,
            bbox: (5, 5, 1, 1),
        };
        let blob_b = Blob {
            pixels: vec![(15, 5)],
            center: (15.0, 5.0),
            intensity: 1.0,
            bbox: (15, 5, 1, 1),
        };

        let rel = SpatialRelationship::from_blobs(&blob_a, &blob_b, 20, 20);

        // B is to the right of A
        assert!(rel.dx > 0.0);
        assert!(rel.dy.abs() < 0.01); // Same row
        assert!(rel.is_right_of());
        assert_eq!(rel.primary_direction(), Some(VisionEdge::RightOf));
    }

    #[test]
    fn test_spatial_relationship_above_below() {
        let blob_top = Blob {
            pixels: vec![(10, 2)],
            center: (10.0, 2.0),
            intensity: 1.0,
            bbox: (10, 2, 1, 1),
        };
        let blob_bottom = Blob {
            pixels: vec![(10, 18)],
            center: (10.0, 18.0),
            intensity: 1.0,
            bbox: (10, 18, 1, 1),
        };

        let rel = SpatialRelationship::from_blobs(&blob_top, &blob_bottom, 20, 20);

        // Bottom is below Top
        assert!(rel.dy > 0.0);
        assert!(rel.is_below());
        assert_eq!(rel.primary_direction(), Some(VisionEdge::Below));

        // Reverse: Top is above Bottom
        let rel_reverse = SpatialRelationship::from_blobs(&blob_bottom, &blob_top, 20, 20);
        assert!(rel_reverse.dy < 0.0);
        assert!(rel_reverse.is_above());
        assert_eq!(rel_reverse.primary_direction(), Some(VisionEdge::Above));
    }

    #[test]
    fn test_spatial_relationship_distance() {
        let blob_a = Blob {
            pixels: vec![(0, 0)],
            center: (0.0, 0.0),
            intensity: 1.0,
            bbox: (0, 0, 1, 1),
        };
        let blob_b = Blob {
            pixels: vec![(10, 0)],
            center: (10.0, 0.0),
            intensity: 1.0,
            bbox: (10, 0, 1, 1),
        };

        let rel = SpatialRelationship::from_blobs(&blob_a, &blob_b, 10, 10);

        // Distance should be normalized to diagonal (sqrt(100+100) = 14.14)
        // Distance 10 / 14.14 ≈ 0.707
        assert!(rel.distance > 0.5 && rel.distance < 0.8);
    }

    #[test]
    fn test_compute_spatial_relationships() {
        let blobs = vec![
            Blob {
                pixels: vec![(2, 2)],
                center: (2.0, 2.0),
                intensity: 1.0,
                bbox: (2, 2, 1, 1),
            },
            Blob {
                pixels: vec![(8, 2)],
                center: (8.0, 2.0),
                intensity: 1.0,
                bbox: (8, 2, 1, 1),
            },
            Blob {
                pixels: vec![(2, 8)],
                center: (2.0, 8.0),
                intensity: 1.0,
                bbox: (2, 8, 1, 1),
            },
        ];

        let relationships = compute_spatial_relationships(&blobs, 10, 10, 1.0);

        // Should have 3 pairs: (0,1), (0,2), (1,2)
        assert_eq!(relationships.len(), 3);

        // Check that relationships are computed
        for (i, j, rel) in &relationships {
            assert!(*i < *j);
            assert!(rel.distance > 0.0);
        }
    }

    #[test]
    fn test_vision_edge_directional() {
        assert_eq!(VisionEdge::Above, VisionEdge::Above);
        assert_ne!(VisionEdge::Above, VisionEdge::Below);
        assert_eq!(VisionEdge::Proximity(0.5), VisionEdge::Proximity(0.5));
    }

    #[test]
    fn test_image_to_graph_with_spatial() {
        // Create two blobs in different positions (closer together for spatial edges)
        let mut pixels = vec![0.0f32; 400]; // 20x20
        // Blob 1: left side
        for y in 8..12 {
            for x in 3..7 {
                pixels[y * 20 + x] = 0.8;
            }
        }
        // Blob 2: right side (close enough for directional relationship)
        for y in 8..12 {
            for x in 13..17 {
                pixels[y * 20 + x] = 0.8;
            }
        }

        let image = RawImage::grayscale(20, 20, pixels).unwrap();
        let mut config = FeatureConfig::default();
        config.build_spatial_graph = true;
        config.build_hierarchy = false;
        config.max_hierarchy_levels = 1;
        config.adjacency_threshold = 0.5; // Allow more distant relationships

        let graph = image_to_graph(&image, &config).unwrap();

        // Should have root + 2 blobs = 3 nodes
        assert_eq!(graph.node_count(), 3);
        // Should have:
        // - 2 Contains edges (root → blob1, root → blob2)
        // - Directional edges (blob1 → blob2 RightOf, blob2 → blob1 LeftOf)
        // - Proximity edges (if within threshold)
        assert!(graph.edge_count() >= 4, "Expected at least 4 edges (2 contains + 2 directional), got {}", graph.edge_count());
    }

    // ========================================================================
    // Hierarchical Blob Detection Tests
    // ========================================================================

    #[test]
    fn test_hierarchical_blob_new() {
        let blob = Blob {
            pixels: vec![(0, 0), (1, 0)],
            center: (0.5, 0.0),
            intensity: 0.8,
            bbox: (0, 0, 2, 1),
        };
        let hblob = HierarchicalBlob::new(blob.clone(), 1, 0.5);
        assert_eq!(hblob.level, 1);
        assert!((hblob.scale - 0.5).abs() < 1e-6);
        assert!(hblob.parent.is_none());
        assert!(hblob.children.is_empty());
    }

    #[test]
    fn test_hierarchical_blob_contains() {
        let parent = HierarchicalBlob::new(
            Blob {
                pixels: vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
                center: (1.0, 0.5),
                intensity: 0.8,
                bbox: (0, 0, 3, 2),
            },
            0,
            1.0,
        );
        let child_inside = Blob {
            pixels: vec![(1, 0)],
            center: (1.0, 0.0),
            intensity: 0.9,
            bbox: (1, 0, 1, 1),
        };
        let child_outside = Blob {
            pixels: vec![(5, 5)],
            center: (5.0, 5.0),
            intensity: 0.7,
            bbox: (5, 5, 1, 1),
        };

        assert!(parent.contains(&child_inside));
        assert!(!parent.contains(&child_outside));
    }

    #[test]
    fn test_extract_hierarchical_blobs_single_blob() {
        // Single blob should result in single-element hierarchy
        let mut pixels = vec![0.0f32; 100];
        for y in 4..7 {
            for x in 4..7 {
                pixels[y * 10 + x] = 0.8;
            }
        }
        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let mut config = FeatureConfig::default();
        config.max_hierarchy_levels = 2;

        let hierarchy = extract_hierarchical_blobs(&image, &config);
        // Should find blobs at multiple scales
        assert!(hierarchy.num_levels == 2);
    }

    #[test]
    fn test_extract_hierarchical_blobs_multi_scale() {
        // Create image with varying intensities (should produce hierarchy)
        let mut pixels = vec![0.0f32; 400]; // 20x20
        // Large low-intensity region
        for y in 2..18 {
            for x in 2..18 {
                pixels[y * 20 + x] = 0.4;
            }
        }
        // Smaller high-intensity region inside
        for y in 6..14 {
            for x in 6..14 {
                pixels[y * 20 + x] = 0.7;
            }
        }
        // Even smaller very high intensity core
        for y in 8..12 {
            for x in 8..12 {
                pixels[y * 20 + x] = 0.95;
            }
        }

        let image = RawImage::grayscale(20, 20, pixels).unwrap();
        let mut config = FeatureConfig::default();
        config.max_hierarchy_levels = 3;
        config.blob_threshold = 0.3;

        let hierarchy = extract_hierarchical_blobs(&image, &config);

        // Should have multiple levels
        assert_eq!(hierarchy.num_levels, 3);
        // Should have some blobs
        assert!(!hierarchy.blobs.is_empty());
    }

    #[test]
    fn test_blob_hierarchy_parent_child_links() {
        // Create nested blobs
        let mut pixels = vec![0.0f32; 400]; // 20x20
        // Outer region (lower intensity)
        for y in 2..18 {
            for x in 2..18 {
                pixels[y * 20 + x] = 0.4;
            }
        }
        // Inner region (higher intensity)
        for y in 6..14 {
            for x in 6..14 {
                pixels[y * 20 + x] = 0.8;
            }
        }

        let image = RawImage::grayscale(20, 20, pixels).unwrap();
        let mut config = FeatureConfig::default();
        config.max_hierarchy_levels = 2;
        config.blob_threshold = 0.3;

        let hierarchy = extract_hierarchical_blobs(&image, &config);

        // Check that parent-child relationships are consistent
        for (idx, hblob) in hierarchy.blobs.iter().enumerate() {
            // If has parent, parent should have this as child
            if let Some(parent_idx) = hblob.parent {
                assert!(hierarchy.blobs[parent_idx].children.contains(&idx));
            }
            // If has children, children should have this as parent
            for &child_idx in &hblob.children {
                assert_eq!(hierarchy.blobs[child_idx].parent, Some(idx));
            }
        }
    }

    #[test]
    fn test_image_to_graph_hierarchical() {
        // Create nested structure
        let mut pixels = vec![0.0f32; 400]; // 20x20
        for y in 2..18 {
            for x in 2..18 {
                pixels[y * 20 + x] = 0.5;
            }
        }
        for y in 6..14 {
            for x in 6..14 {
                pixels[y * 20 + x] = 0.9;
            }
        }

        let image = RawImage::grayscale(20, 20, pixels).unwrap();
        let mut config = FeatureConfig::default();
        config.build_hierarchy = true;
        config.max_hierarchy_levels = 2;
        config.blob_threshold = 0.3;

        let graph = image_to_graph(&image, &config).unwrap();

        // Should have root node
        assert!(graph.root.is_some());
        // Should have blob nodes (at least one from each scale)
        assert!(graph.node_count() > 1);
        // Should have hierarchy edges
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_image_to_graph_single_level_fallback() {
        // With max_hierarchy_levels = 1, should use single-level detection
        let mut pixels = vec![0.0f32; 100];
        for y in 4..7 {
            for x in 4..7 {
                pixels[y * 10 + x] = 0.8;
            }
        }

        let image = RawImage::grayscale(10, 10, pixels).unwrap();
        let mut config = FeatureConfig::default();
        config.build_hierarchy = true;
        config.max_hierarchy_levels = 1; // Single level

        let graph = image_to_graph(&image, &config).unwrap();

        assert!(graph.root.is_some());
        assert!(graph.node_count() >= 2); // Root + at least one blob
    }

    // ========================================================================
    // ClassificationBrain Tests
    // ========================================================================

    #[test]
    fn test_classification_config_mnist() {
        let config = ClassificationConfig::mnist();
        assert_eq!(config.num_classes, 10);
        assert_eq!(config.num_outputs, 10);
        assert!((config.template_momentum - 0.9).abs() < 1e-6);
        assert!(config.use_structural);
    }

    #[test]
    fn test_classification_config_custom() {
        let config = ClassificationConfig::new(5, 8)
            .with_momentum(0.8)
            .with_structural(false);
        assert_eq!(config.num_classes, 5);
        assert_eq!(config.num_outputs, 8);
        assert!((config.template_momentum - 0.8).abs() < 1e-6);
        assert!(!config.use_structural);
    }

    #[test]
    fn test_classification_brain_mnist() {
        let brain = ClassificationBrain::mnist();
        assert_eq!(brain.domain_id(), "classification");
        assert_eq!(brain.num_classes(), 10);
        assert_eq!(brain.labels().len(), 10);
        assert_eq!(brain.get_label(0), Some("0"));
        assert_eq!(brain.get_label(9), Some("9"));
    }

    #[test]
    fn test_classification_brain_with_labels() {
        let labels = vec![
            "cat".to_string(),
            "dog".to_string(),
            "bird".to_string(),
        ];
        let brain = ClassificationBrain::new(ClassificationConfig::new(3, 3))
            .with_labels(labels);
        assert_eq!(brain.get_label(0), Some("cat"));
        assert_eq!(brain.get_label(1), Some("dog"));
        assert_eq!(brain.get_label(2), Some("bird"));
    }

    #[test]
    fn test_classification_output_new() {
        let probs = vec![0.8, 0.1, 0.1];
        let output = ClassificationOutput::new(0, 0.8, probs.clone());
        assert_eq!(output.predicted_class, 0);
        assert!((output.confidence - 0.8).abs() < 1e-6);
        assert_eq!(output.probabilities, probs);
        assert!(output.label.is_none());
    }

    #[test]
    fn test_classification_output_with_label() {
        let output = ClassificationOutput::new(0, 0.8, vec![0.8, 0.2])
            .with_label("cat");
        assert_eq!(output.label, Some("cat".to_string()));
    }

    #[test]
    fn test_classification_brain_classify() {
        let brain = ClassificationBrain::mnist();

        // Create a MNIST DagNN with hidden layer and output nodes
        let pixels = vec![0.0f32; 784];
        let mut dag = DagNN::from_mnist_with_classifier(&pixels, 32).unwrap();
        dag.neuromorphic_forward().unwrap();

        let result = brain.classify(&dag);
        assert!(result.predicted_class < 10);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.probabilities.len(), 10);
    }

    #[test]
    fn test_classification_brain_domain_brain_trait() {
        let brain = ClassificationBrain::mnist();

        // Test DomainBrain trait methods
        assert_eq!(brain.domain_id(), "classification");
        assert_eq!(brain.domain_name(), "Classification Brain");
        assert!(brain.can_process("classify this"));
        assert!(!brain.can_process("hello world"));

        let rules = brain.get_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].name, "Structural Matching");
    }

    #[test]
    fn test_classification_brain_classifier_access() {
        let mut brain = ClassificationBrain::mnist();

        // Test read access
        let classifier = brain.classifier();
        assert_eq!(classifier.templates.len(), 10);

        // Test mutable access
        let classifier_mut = brain.classifier_mut();
        assert_eq!(classifier_mut.templates.len(), 10);
    }

    #[test]
    fn test_classification_brain_execute() {
        let brain = ClassificationBrain::mnist();

        let pixels = vec![0.0f32; 784];
        let mut dag = DagNN::from_mnist_with_classifier(&pixels, 32).unwrap();
        dag.neuromorphic_forward().unwrap();

        let result = brain.execute(&dag);
        assert!(result.is_ok());
        if let Ok(grapheme_core::ExecutionResult::Text(text)) = result {
            assert!(text.contains("Predicted class:"));
            assert!(text.contains("confidence:"));
        }
    }
}
