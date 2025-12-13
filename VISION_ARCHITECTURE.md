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

## Refactoring Status (December 2025)

1. [x] VisionBrain.to_graph() returns generic Graph (not VisionGraph)
2. [x] VisionGraph becomes private/internal type (pub(crate))
3. [x] ClassificationBrain gets learnable parameters (template matching with Adam)
4. [x] MnistModel → ImageClassificationModel (generic)
5. [x] Move MNIST-specific code to grapheme-train (train_mnist.rs)
6. [x] Remove all hardcoded 28, 784, 10 values from grapheme-vision
7. [x] Router-to-training integration (process_to_graph, input_to_graph)

**All refactoring complete** - grapheme-vision is now generic and configuration-driven.

## Router Integration (December 2025)

The `CognitiveRouter` now supports training graph generation:

```rust
/// Route input and return (input_graph, output_graph) pair for training
pub fn route_for_training(&self, input: &Input) -> RouterResult<TrainingPair>;

/// Generate training examples from a batch of inputs
pub fn generate_training_batch(&self, inputs: &[Input]) -> Vec<RouterResult<TrainingPair>>;

pub struct TrainingPair {
    pub module_id: ModuleId,
    pub confidence: f32,
    pub input_graph: DagNN,
    pub output_graph: DagNN,
}
```

This enables multi-modal AGI training with structural loss across all domain brains.

## Unified AGI Training (December 2025)

The `train_unified_agi` binary implements shared DagNN training with brain slices:

```rust
/// Shared DagNN model with brain slice allocation
struct SharedAGIModel {
    dag: DagNN,                              // Single shared network
    slices: HashMap<String, BrainSlice>,     // Per-domain node ownership
    learning_rate: f32,
    gradients: HashMap<String, Vec<f32>>,    // Per-slice gradient accumulation
}

// Brain slice allocation
let brain_requests = vec![
    ("math".to_string(), 32, 16),       // Math: 32 input, 16 output
    ("text".to_string(), 64, 32),       // Text: 64 input, 32 output
    ("timeseries".to_string(), 16, 8),  // TimeSeries: 16 input, 8 output
    ("vision".to_string(), 48, 16),     // Vision: 48 input, 16 output
];
```

The `generate_mixed_agi` binary creates multi-modal datasets:
- **Math**: Arithmetic expressions
- **Text**: QA factoid pairs
- **TimeSeries**: Sequence prediction (linear, doubling, Fibonacci)
- **Vision**: 4x4 pattern classification (8 pattern types)

## AGI Roadmap (December 2025)

### Planned: Graph-to-Graph (G2G) Learning
- `backend-175`: G2G transformation learning
- `backend-176`: Graph morphism detection and alignment
- `backend-180`: Efficient graph serialization for network transport

### Planned: A2A (Agent-to-Agent) Protocol
- `api-015`: A2A communication protocol
- `api-016`: Graph-based message format
- `api-019`: Agent discovery and registry
- `backend-177`: Multi-agent orchestration with GRAPHEME coordinator

### Planned: LLM Collaboration
- `integration-001`: LLM API client (Claude, OpenAI, Gemini)
- `integration-002`: LLM response → DagNN graph translation
- `integration-003`: DagNN graph → LLM prompt translation
- `backend-178`: Collaborative learning from LLM interactions
- `backend-179`: Knowledge distillation from LLMs to graphs

### Planned: MCP Integration
- `api-017`: MCP server for GRAPHEME
- `api-018`: MCP tools (graph_query, graph_transform, train_step)
- `integration-004`: MCP client for external tool servers

## Two-Stage Training Paradigm (December 2025)

### The Vision: Train on Graphs, Not Text

Traditional training flow has a critical flaw:
```
Text → Graph → Transform → Graph → Text
         ↑                      ↓
    (lossy encoding)      (lossy decoding)
```

The new two-stage paradigm eliminates text from the training loop:

### Stage 1: Graph Autoencoder (Perfect the Brains)

Each cognitive brain learns to be a **perfect encoder/decoder** for its domain:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: BRAIN AUTOENCODER TRAINING                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   For each domain brain (Code, Math, Text, Vision, etc.):                    │
│                                                                              │
│   Domain Input ──► Brain.encode() ──► Latent Graph ──► Brain.decode() ──►   │
│        ↓                                    ↓                          ↓     │
│   "def foo():"                        Graph repr.               "def foo():" │
│                                                                              │
│   Loss: Reconstruction (input_text == output_text)                           │
│   Goal: Zero information loss in encoding/decoding                           │
│                                                                              │
│   CodeBrain:   "fn main() {}" ←──► CodeGraph  (perfect reconstruction)       │
│   MathBrain:   "2 + 3 * 4"    ←──► MathGraph  (perfect reconstruction)       │
│   TextBrain:   "Hello world"  ←──► TextGraph  (perfect reconstruction)       │
│   VisionBrain: [pixels]       ←──► VisionGraph (feature preservation)        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 2: Graph-Only Transformation (Pure Graph Learning)

Once brains can perfectly encode/decode, we train **purely on graphs**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: GRAPH-TO-GRAPH TRANSFORMATION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   NO TEXT IN THIS STAGE - Pure graph operations                              │
│                                                                              │
│   Problem Graph ──────────► GraphTransformNet ──────────► Solution Graph     │
│        ↓                          ↓                            ↓             │
│   (pre-encoded)            (learns mapping)             (pre-encoded)        │
│                                                                              │
│   Loss: Structural (graph topology, node types, edge weights)                │
│   Goal: Learn graph→graph transformations without text interference          │
│                                                                              │
│   Training Data Format:                                                      │
│   {                                                                          │
│     "input_graph": <serialized CodeGraph from docstring>,                    │
│     "output_graph": <serialized CodeGraph from solution>                     │
│   }                                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters for HumanEval

```
Current approach:  Text → Graph → Transform → Graph → Text  (lossy at both ends)
New approach:      Graph ────────► Transform ────────► Graph (lossless)
```

Benefits:
1. **No tokenization artifacts** - Graphs preserve code structure natively
2. **Compositional generalization** - Graph operations compose naturally
3. **Smaller model needed** - Learning graph transforms is simpler than text→text
4. **Interpretable** - Can inspect intermediate graphs for debugging

### Brains as Perception (Eyes and Ears)

After Stage 1 training, brains become like **sensory organs**:
- They **perceive** raw input (text, images, audio)
- They **convert** to the universal graph language
- The **thinking** happens entirely in graph space
- They **translate** back to the output modality

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BRAINS AS SENSORY ORGANS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   PERCEPTION (Stage 1 trained)        COGNITION (Stage 2 trained)            │
│   ─────────────────────────────       ────────────────────────────           │
│                                                                              │
│   CodeBrain   ──┐                                                            │
│   (eyes for    │                                                             │
│    code)       │                     ┌─────────────────┐                     │
│                ├──► Input Graph ──►  │  GraphTransform  │ ──► Output Graph   │
│   MathBrain   ──┤                    │     (learns)     │                    │
│   (eyes for    │                     └─────────────────┘                     │
│    math)       │                                                             │
│                │                                                             │
│   VisionBrain ──┘                                                            │
│   (eyes for                                                                  │
│    images)                                                                   │
│                                                                              │
│   Brains are FROZEN after Stage 1 - only GraphTransform learns in Stage 2   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

**Stage 1: Brain Autoencoder Tasks**
- `backend-221`: Implement GraphAutoencoder trait for all brains
- `backend-222`: CodeBrain autoencoder (code ↔ CodeGraph)
- `backend-223`: MathBrain autoencoder (expression ↔ MathGraph)
- `backend-224`: TextBrain autoencoder (text ↔ TextGraph)
- `backend-225`: VisionBrain autoencoder (image ↔ VisionGraph)
- `backend-226`: Autoencoder reconstruction loss and metrics

**Stage 2: Graph-Only Training Tasks**
- `backend-227`: Graph-only training data format
- `backend-228`: Pre-encode HumanEval to graph pairs
- `backend-229`: Graph-only trainer (no text in training loop)
- `backend-230`: Structural loss improvements for code graphs
- `backend-231`: HumanEval benchmark integration
