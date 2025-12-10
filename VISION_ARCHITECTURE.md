# GRAPHEME Vision Architecture

## Component Distinctions

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           GRAPHEME VISION ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. VisionGraph (INTERNAL to VisionBrain)                                       │
│     - Blob/edge/corner detection results                                        │
│     - Domain-specific: knows about pixels, spatial relationships                │
│     - NOT exposed to DagNN - internal representation only                       │
│                                                                                 │
│  2. Input Graph (GENERIC - output of VisionBrain)                               │
│     - Generic nodes with activations (no domain knowledge)                      │
│     - What VisionBrain produces for DagNN consumption                           │
│     - Domain-agnostic: DagNN sees only nodes/edges/weights                      │
│                                                                                 │
│  3. DagNN (GRAPHEME Core - LEARNS)                                              │
│     - Transforms Input Graph → Output Graph                                     │
│     - Domain-agnostic: only knows graph structure                               │
│     - The learning happens here                                                 │
│                                                                                 │
│  4. ClassificationBrain (LEARNABLE - interprets Output Graph)                   │
│     - Converts Output Graph → Class Labels                                      │
│     - Generic: any number of classes                                            │
│     - Learnable templates/weights                                               │
│                                                                                 │
│  5. ImageClassificationModel (GENERIC pipeline)                                 │
│     - Combines: VisionBrain + DagNN + ClassificationBrain                       │
│     - Works with ANY image size, ANY number of classes                          │
│     - Configuration-driven, no hardcoded values                                 │
│                                                                                 │
│  6. MnistModel (SPECIFIC - belongs in grapheme-train)                           │
│     - ImageClassificationModel configured for MNIST                             │
│     - 28x28 images, 10 classes, specific hyperparameters                        │
│     - Just a configuration, not a separate type                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌──────────────────┐
│   RawImage  │ ──►  │ VisionBrain │ ──►  │    DagNN    │ ──►  │ClassificationBrain│
│  (any size) │      │             │      │   (learns)  │      │   (learnable)    │
└─────────────┘      └─────────────┘      └─────────────┘      └──────────────────┘
                            │                    │                      │
                            ▼                    ▼                      ▼
                     ┌─────────────┐      ┌─────────────┐      ┌──────────────────┐
                     │ VisionGraph │      │ Output Graph│      │   Class Label    │
                     │ (internal)  │      │  (generic)  │      │                  │
                     └─────────────┘      └─────────────┘      └──────────────────┘
```

## Key Principles

### 1. VisionGraph is INTERNAL
- Used only within VisionBrain for feature extraction
- Contains domain-specific types: Blob, Edge, Corner, Region
- NOT passed to DagNN

### 2. DagNN sees ONLY generic graphs
- Input: Generic nodes with activations
- Output: Generic nodes with activations
- No knowledge of images, pixels, or classes
- This is where learning happens

### 3. DagNN can REQUEST more features from VisionBrain
- During training, DagNN may need more information
- DagNN can ask VisionBrain for: more blobs, finer edges, different thresholds
- VisionBrain parameters are tunable by DagNN feedback
- Example: "I need more blobs" → VisionBrain lowers blob_threshold
- This creates a feedback loop for optimal feature extraction

### 4. ClassificationBrain is LEARNABLE
- Not just template matching
- Has learnable parameters
- Interprets output graph as class probabilities

### 5. ImageClassificationModel is GENERIC
- Configuration object specifies:
  - Image dimensions (width, height, channels)
  - Number of classes
  - Feature extraction parameters
  - Learning hyperparameters
- No hardcoded values

### 6. Dataset-specific configs belong in training crates
- MNIST config → grapheme-train
- CIFAR config → grapheme-train
- Custom datasets → user code

## Feedback Loop: DagNN ↔ VisionBrain

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE FEATURE EXTRACTION                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   VisionBrain                    DagNN                                          │
│   ───────────                    ─────                                          │
│                                                                                 │
│   Parameters:                    During training:                               │
│   - blob_threshold    ◄────────  "Need more blobs" (poor discrimination)        │
│   - min_blob_size     ◄────────  "Blobs too small" (noise)                      │
│   - max_blobs         ◄────────  "Need more detail" (underfitting)              │
│   - edge_threshold    ◄────────  "Need edge info" (shape matters)               │
│   - hierarchy_levels  ◄────────  "Need structure" (complex patterns)            │
│                                                                                 │
│   VisionBrain adapts its feature extraction based on DagNN learning signals     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## VisionGraph Internal Types

VisionGraph is the internal representation used by VisionBrain during feature extraction. It contains domain-specific types that should NOT be exposed to DagNN:

```rust
// INTERNAL to VisionBrain - not part of public API
pub(crate) struct VisionGraph {
    blobs: Vec<Blob>,       // Detected regions
    edges: Vec<EdgeFeature>, // Detected edges
    corners: Vec<Corner>,    // Detected corners
    hierarchy: HierarchyTree, // Spatial relationships
}

pub(crate) struct Blob {
    centroid: (f32, f32),   // Position
    size: usize,            // Pixel count
    intensity: f32,         // Average brightness
    bounding_box: Rect,     // Extent
}

// VisionBrain converts VisionGraph → generic Graph for DagNN
impl VisionBrain {
    pub fn to_graph(&self, image: &RawImage) -> Graph {
        // 1. Build internal VisionGraph (domain-specific)
        let vision_graph = self.extract_features(image);

        // 2. Convert to generic Graph (domain-agnostic)
        self.vision_graph_to_generic(&vision_graph)
    }
}
```

## DomainBrain Trait Implementation

VisionBrain implements the `DomainBrain` trait from GRAPHEME_Vision.md:

```rust
impl DomainBrain for VisionBrain {
    /// Convert image to graph (DETERMINISTIC for fixed parameters)
    fn to_graph(&self, input: &DomainInput) -> Graph;

    /// Convert graph back to image (not typically used for classification)
    fn from_graph(&self, graph: &Graph) -> DomainOutput;

    /// Domain-specific transformation rules (e.g., augmentation)
    fn transform(&self, graph: &Graph, rule: &Rule) -> Graph;
}
```

The `transform()` method can be used for:
- **Data augmentation**: Apply rotation/translation to the graph
- **Feature enhancement**: Add edge/corner features on demand
- **Resolution changes**: Re-extract features at different scales

## Refactoring TODO

1. [ ] VisionBrain.to_graph() returns generic Graph (not VisionGraph)
2. [ ] VisionGraph becomes private/internal type (pub(crate))
3. [ ] ClassificationBrain gets learnable parameters
4. [ ] MnistModel → ImageClassificationModel (generic)
5. [ ] Move MNIST-specific code to grapheme-train
6. [ ] Remove all hardcoded 28, 784, 10 values from grapheme-vision
