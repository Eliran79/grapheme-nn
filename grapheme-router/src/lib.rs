//! AGI-Ready Cognitive Router for GRAPHEME
//!
//! This crate implements an intelligent routing system that automatically
//! selects which cognitive module to use based on input analysis.
//!
//! # Architecture
//!
//! The router analyzes incoming inputs and routes them to appropriate modules:
//! - Text inputs → NLP modules (kindergarten, math, code)
//! - Image inputs → Vision modules (MNIST, object detection)
//! - Time series inputs → Forecasting modules
//!
//! # Example
//!
//! ```ignore
//! use grapheme_router::{CognitiveRouter, Input};
//!
//! let mut router = CognitiveRouter::new(0.5);
//! router.register_module(Box::new(TextModule::new()));
//! router.register_module(Box::new(TimeSeriesModule::new()));
//!
//! let result = router.route(&Input::Text("Hello world".to_string()))?;
//! println!("Module: {:?}, Confidence: {}", result.module_id, result.confidence);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Error types for routing operations
#[derive(Error, Debug)]
pub enum RouterError {
    #[error("No modules registered")]
    NoModulesRegistered,

    #[error("No suitable module found for input type: {0}")]
    NoSuitableModule(String),

    #[error("Confidence too low: {confidence:.2} < threshold {threshold:.2}")]
    ConfidenceTooLow { confidence: f32, threshold: f32 },

    #[error("Input analysis failed: {0}")]
    AnalysisFailed(String),

    #[error("Module execution failed: {0}")]
    ExecutionFailed(String),
}

/// Result type for routing operations
pub type RouterResult<T> = Result<T, RouterError>;

/// Input types that the router can handle
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Input {
    /// Text input (natural language, code, math expressions)
    Text(String),

    /// Numeric sequence (time series data)
    NumericSequence(Vec<f32>),

    /// 2D array of pixel values (grayscale image)
    Image {
        width: usize,
        height: usize,
        pixels: Vec<f32>,
    },

    /// Comma-separated numeric values (as string)
    CsvNumeric(String),

    /// Raw bytes (unknown format)
    Raw(Vec<u8>),
}

impl Input {
    /// Create a text input
    pub fn text(s: impl Into<String>) -> Self {
        Input::Text(s.into())
    }

    /// Create a numeric sequence input
    pub fn sequence(values: Vec<f32>) -> Self {
        Input::NumericSequence(values)
    }

    /// Create an image input
    pub fn image(width: usize, height: usize, pixels: Vec<f32>) -> Self {
        Input::Image { width, height, pixels }
    }

    /// Check if this is a text input
    pub fn is_text(&self) -> bool {
        matches!(self, Input::Text(_))
    }

    /// Check if this is a numeric sequence
    pub fn is_sequence(&self) -> bool {
        matches!(self, Input::NumericSequence(_) | Input::CsvNumeric(_))
    }

    /// Check if this is an image
    pub fn is_image(&self) -> bool {
        matches!(self, Input::Image { .. })
    }
}

/// Detected input type with confidence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputType {
    /// Plain text (kindergarten, reading)
    Text,
    /// Mathematical expression
    Math,
    /// Source code
    Code,
    /// Time series data
    TimeSeries,
    /// Image data
    Image,
    /// Chemical formula
    Chemical,
    /// Musical notation
    Music,
    /// Unknown input type
    Unknown,
}

/// Module identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModuleId {
    Text,
    Math,
    Code,
    TimeSeries,
    Vision,
    Chemical,
    Music,
    Custom(u32),
}

/// Routing result with module selection and confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingResult {
    /// Selected module ID
    pub module_id: ModuleId,
    /// Detected input type
    pub input_type: InputType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Output from module execution
    pub output: Option<String>,
    /// Additional modules that could handle this input
    pub alternatives: Vec<(ModuleId, f32)>,
}

/// Cognitive module trait for pluggable modules
pub trait CognitiveModule: Send + Sync {
    /// Get the module's unique identifier
    fn module_id(&self) -> ModuleId;

    /// Get the module's display name
    fn name(&self) -> &str;

    /// Check if this module can handle the given input
    fn can_handle(&self, input: &Input) -> f32;

    /// Get the input types this module prefers
    fn preferred_input_types(&self) -> Vec<InputType>;

    /// Process the input and return output
    fn process(&self, input: &Input) -> RouterResult<String>;
}

/// Cognitive router that automatically routes inputs to appropriate modules
pub struct CognitiveRouter {
    modules: HashMap<ModuleId, Box<dyn CognitiveModule>>,
    confidence_threshold: f32,
    /// Enable multi-module routing (combine outputs from multiple modules)
    multi_module: bool,
}

impl Default for CognitiveRouter {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl CognitiveRouter {
    /// Create a new router with the given confidence threshold
    pub fn new(confidence_threshold: f32) -> Self {
        Self {
            modules: HashMap::new(),
            confidence_threshold,
            multi_module: false,
        }
    }

    /// Set the confidence threshold for routing
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Enable multi-module routing
    pub fn with_multi_module(mut self, enable: bool) -> Self {
        self.multi_module = enable;
        self
    }

    /// Register a cognitive module
    pub fn register_module(&mut self, module: Box<dyn CognitiveModule>) {
        let id = module.module_id();
        self.modules.insert(id, module);
    }

    /// Get the number of registered modules
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Analyze input and detect type
    pub fn analyze_input(&self, input: &Input) -> (InputType, f32) {
        match input {
            Input::Text(text) => self.analyze_text(text),
            Input::NumericSequence(values) => {
                if values.len() >= 2 {
                    (InputType::TimeSeries, 0.95)
                } else {
                    (InputType::Unknown, 0.3)
                }
            }
            Input::Image { .. } => (InputType::Image, 0.98),
            Input::CsvNumeric(csv) => {
                if csv.contains(',') && csv.split(',').all(|s| s.trim().parse::<f32>().is_ok()) {
                    (InputType::TimeSeries, 0.90)
                } else {
                    (InputType::Unknown, 0.3)
                }
            }
            Input::Raw(_) => (InputType::Unknown, 0.1),
        }
    }

    /// Analyze text input to determine type
    fn analyze_text(&self, text: &str) -> (InputType, f32) {
        let text = text.trim();

        // Check for math patterns
        let has_math_ops = text.chars().any(|c| "+-*/=^".contains(c));
        let has_digits = text.chars().any(|c| c.is_ascii_digit());
        let has_parens = text.contains('(') || text.contains(')');
        let math_score = (has_math_ops as u8 + has_digits as u8 + has_parens as u8) as f32 / 3.0;

        // Check for code patterns
        let has_braces = text.contains('{') || text.contains('}');
        let has_semicolon = text.contains(';');
        let has_fn_keyword = text.contains("fn ") || text.contains("def ") || text.contains("function ");
        let code_score = (has_braces as u8 + has_semicolon as u8 + has_fn_keyword as u8) as f32 / 3.0;

        // Check for chemical patterns
        let has_element = ["H", "O", "C", "N", "Na", "Cl", "Fe"]
            .iter()
            .any(|e| text.contains(e));
        let chemical_score = if has_element && has_digits { 0.7 } else { 0.0 };

        // Check for music patterns
        let has_notes = ["C", "D", "E", "F", "G", "A", "B"]
            .iter()
            .any(|n| text.contains(n));
        let music_score = if has_notes && text.len() < 20 { 0.3 } else { 0.0 };

        // Determine highest scoring type
        let mut best_type = InputType::Text;
        let mut best_score = 0.6; // Default text confidence

        if math_score > 0.3 && math_score > code_score {
            best_type = InputType::Math;
            best_score = 0.7 + math_score * 0.25;
        } else if code_score > 0.3 {
            best_type = InputType::Code;
            best_score = 0.7 + code_score * 0.25;
        } else if chemical_score > 0.5 {
            best_type = InputType::Chemical;
            best_score = chemical_score;
        } else if music_score > 0.3 && text.len() < 10 {
            best_type = InputType::Music;
            best_score = music_score;
        }

        (best_type, best_score.min(0.99))
    }

    /// Route input to appropriate module(s)
    pub fn route(&self, input: &Input) -> RouterResult<RoutingResult> {
        if self.modules.is_empty() {
            return Err(RouterError::NoModulesRegistered);
        }

        // Step 1: Analyze input
        let (input_type, type_confidence) = self.analyze_input(input);

        // Step 2: Find best matching module(s)
        let mut module_scores: Vec<(ModuleId, f32)> = self
            .modules
            .values()
            .map(|m| (m.module_id(), m.can_handle(input)))
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // Sort by score descending
        module_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Step 3: Check if any module can handle this input
        let Some((best_module_id, best_score)) = module_scores.first().copied() else {
            return Err(RouterError::NoSuitableModule(format!("{:?}", input_type)));
        };

        // Step 4: Compute overall confidence
        let confidence = (best_score + type_confidence) / 2.0;

        // Step 5: Check confidence threshold
        if confidence < self.confidence_threshold {
            return Err(RouterError::ConfidenceTooLow {
                confidence,
                threshold: self.confidence_threshold,
            });
        }

        // Step 6: Execute module
        let output = if let Some(module) = self.modules.get(&best_module_id) {
            match module.process(input) {
                Ok(out) => Some(out),
                Err(e) => {
                    // Module execution failed, but we still return routing info
                    Some(format!("Error: {}", e))
                }
            }
        } else {
            None
        };

        // Step 7: Get alternatives (for multi-module support)
        let alternatives: Vec<(ModuleId, f32)> = module_scores
            .into_iter()
            .skip(1)
            .take(3)
            .collect();

        Ok(RoutingResult {
            module_id: best_module_id,
            input_type,
            confidence,
            output,
            alternatives,
        })
    }

    /// Get routing decision without executing the module
    pub fn get_routing_decision(&self, input: &Input) -> RouterResult<(ModuleId, f32)> {
        if self.modules.is_empty() {
            return Err(RouterError::NoModulesRegistered);
        }

        let (input_type, type_confidence) = self.analyze_input(input);

        let mut module_scores: Vec<(ModuleId, f32)> = self
            .modules
            .values()
            .map(|m| (m.module_id(), m.can_handle(input)))
            .filter(|(_, score)| *score > 0.0)
            .collect();

        module_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let Some((best_module_id, best_score)) = module_scores.first().copied() else {
            return Err(RouterError::NoSuitableModule(format!("{:?}", input_type)));
        };

        let confidence = (best_score + type_confidence) / 2.0;

        Ok((best_module_id, confidence))
    }
}

// ============================================================================
// Built-in Modules
// ============================================================================

/// Text module for natural language processing
pub struct TextModule;

impl TextModule {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TextModule {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveModule for TextModule {
    fn module_id(&self) -> ModuleId {
        ModuleId::Text
    }

    fn name(&self) -> &str {
        "Text"
    }

    fn can_handle(&self, input: &Input) -> f32 {
        match input {
            Input::Text(text) => {
                // Prefer pure alphabetic text
                let alpha_ratio = text.chars().filter(|c| c.is_alphabetic() || c.is_whitespace()).count() as f32
                    / text.len().max(1) as f32;
                alpha_ratio * 0.9
            }
            _ => 0.0,
        }
    }

    fn preferred_input_types(&self) -> Vec<InputType> {
        vec![InputType::Text]
    }

    fn process(&self, input: &Input) -> RouterResult<String> {
        match input {
            Input::Text(text) => {
                // Simple text processing: return length and word count
                let word_count = text.split_whitespace().count();
                Ok(format!("Text processed: {} chars, {} words", text.len(), word_count))
            }
            _ => Err(RouterError::AnalysisFailed("Not a text input".to_string())),
        }
    }
}

/// Math module for mathematical expressions
pub struct MathModule;

impl MathModule {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MathModule {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveModule for MathModule {
    fn module_id(&self) -> ModuleId {
        ModuleId::Math
    }

    fn name(&self) -> &str {
        "Math"
    }

    fn can_handle(&self, input: &Input) -> f32 {
        match input {
            Input::Text(text) => {
                let has_ops = text.chars().any(|c| "+-*/=^".contains(c));
                let has_digits = text.chars().any(|c| c.is_ascii_digit());
                if has_ops && has_digits {
                    0.85
                } else if has_digits {
                    0.5
                } else {
                    0.0
                }
            }
            Input::NumericSequence(_) => 0.3, // Could be math, but probably time series
            _ => 0.0,
        }
    }

    fn preferred_input_types(&self) -> Vec<InputType> {
        vec![InputType::Math]
    }

    fn process(&self, input: &Input) -> RouterResult<String> {
        match input {
            Input::Text(text) => {
                // Try to evaluate simple expressions
                let result = self.evaluate_simple(text);
                Ok(format!("Math result: {}", result))
            }
            _ => Err(RouterError::AnalysisFailed("Not a math input".to_string())),
        }
    }
}

impl MathModule {
    fn evaluate_simple(&self, expr: &str) -> String {
        // Very simple evaluation for demo purposes
        let expr = expr.trim().replace(' ', "");

        // Try simple addition
        if let Some(idx) = expr.find('+') {
            let (a, b) = expr.split_at(idx);
            let b = &b[1..]; // Skip the '+'
            if let (Ok(a), Ok(b)) = (a.parse::<f64>(), b.parse::<f64>()) {
                return format!("{}", a + b);
            }
        }

        // Try simple multiplication
        if let Some(idx) = expr.find('*') {
            let (a, b) = expr.split_at(idx);
            let b = &b[1..];
            if let (Ok(a), Ok(b)) = (a.parse::<f64>(), b.parse::<f64>()) {
                return format!("{}", a * b);
            }
        }

        // Try simple subtraction
        if let Some(idx) = expr.rfind('-') {
            if idx > 0 {
                let (a, b) = expr.split_at(idx);
                let b = &b[1..];
                if let (Ok(a), Ok(b)) = (a.parse::<f64>(), b.parse::<f64>()) {
                    return format!("{}", a - b);
                }
            }
        }

        format!("Could not evaluate: {}", expr)
    }
}

/// Time series module for forecasting
pub struct TimeSeriesModule {
    window_size: usize,
}

impl TimeSeriesModule {
    pub fn new() -> Self {
        Self { window_size: 10 }
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }
}

impl Default for TimeSeriesModule {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveModule for TimeSeriesModule {
    fn module_id(&self) -> ModuleId {
        ModuleId::TimeSeries
    }

    fn name(&self) -> &str {
        "Time Series"
    }

    fn can_handle(&self, input: &Input) -> f32 {
        match input {
            Input::NumericSequence(values) => {
                if values.len() >= 2 {
                    0.95
                } else {
                    0.3
                }
            }
            Input::CsvNumeric(csv) => {
                if csv.contains(',') {
                    0.85
                } else {
                    0.2
                }
            }
            Input::Text(text) if text.contains(',') && text.chars().filter(|c| c.is_ascii_digit()).count() > 5 => {
                0.7
            }
            _ => 0.0,
        }
    }

    fn preferred_input_types(&self) -> Vec<InputType> {
        vec![InputType::TimeSeries]
    }

    fn process(&self, input: &Input) -> RouterResult<String> {
        let values = match input {
            Input::NumericSequence(v) => v.clone(),
            Input::CsvNumeric(csv) => {
                csv.split(',')
                    .filter_map(|s| s.trim().parse::<f32>().ok())
                    .collect()
            }
            Input::Text(text) => {
                text.split(',')
                    .filter_map(|s| s.trim().parse::<f32>().ok())
                    .collect()
            }
            _ => return Err(RouterError::AnalysisFailed("Not a time series input".to_string())),
        };

        if values.len() < 2 {
            return Err(RouterError::AnalysisFailed("Need at least 2 values".to_string()));
        }

        // Simple prediction: linear extrapolation
        let n = values.len();
        let last = values[n - 1];
        let prev = values[n - 2];
        let trend = last - prev;
        let prediction = last + trend;

        Ok(format!(
            "Time series: {} values, trend={:.4}, next={:.4}",
            n, trend, prediction
        ))
    }
}

/// Vision module for image processing
pub struct VisionModule;

impl VisionModule {
    pub fn new() -> Self {
        Self
    }
}

impl Default for VisionModule {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveModule for VisionModule {
    fn module_id(&self) -> ModuleId {
        ModuleId::Vision
    }

    fn name(&self) -> &str {
        "Vision"
    }

    fn can_handle(&self, input: &Input) -> f32 {
        match input {
            Input::Image { width, height, .. } => {
                // Prefer standard image sizes
                if *width == 28 && *height == 28 {
                    0.99 // MNIST
                } else if *width > 0 && *height > 0 {
                    0.85
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    fn preferred_input_types(&self) -> Vec<InputType> {
        vec![InputType::Image]
    }

    fn process(&self, input: &Input) -> RouterResult<String> {
        match input {
            Input::Image { width, height, pixels } => {
                // Simple image analysis: compute mean and std
                let mean: f32 = pixels.iter().sum::<f32>() / pixels.len() as f32;
                let variance: f32 = pixels.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / pixels.len() as f32;
                let std = variance.sqrt();

                Ok(format!(
                    "Image {}x{}: mean={:.4}, std={:.4}",
                    width, height, mean, std
                ))
            }
            _ => Err(RouterError::AnalysisFailed("Not an image input".to_string())),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let router = CognitiveRouter::new(0.5);
        assert_eq!(router.module_count(), 0);
    }

    #[test]
    fn test_register_module() {
        let mut router = CognitiveRouter::new(0.5);
        router.register_module(Box::new(TextModule::new()));
        assert_eq!(router.module_count(), 1);
    }

    #[test]
    fn test_text_analysis() {
        let router = CognitiveRouter::new(0.5);
        let (input_type, confidence) = router.analyze_input(&Input::text("Hello world"));
        assert_eq!(input_type, InputType::Text);
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_math_analysis() {
        let router = CognitiveRouter::new(0.5);
        let (input_type, confidence) = router.analyze_input(&Input::text("2 + 3 = 5"));
        assert_eq!(input_type, InputType::Math);
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_time_series_analysis() {
        let router = CognitiveRouter::new(0.5);
        let (input_type, confidence) = router.analyze_input(&Input::sequence(vec![1.0, 2.0, 3.0]));
        assert_eq!(input_type, InputType::TimeSeries);
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_image_analysis() {
        let router = CognitiveRouter::new(0.5);
        let (input_type, confidence) = router.analyze_input(&Input::image(28, 28, vec![0.5; 784]));
        assert_eq!(input_type, InputType::Image);
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_route_text() {
        let mut router = CognitiveRouter::new(0.3);
        router.register_module(Box::new(TextModule::new()));

        let result = router.route(&Input::text("Hello world")).unwrap();
        assert_eq!(result.module_id, ModuleId::Text);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_route_math() {
        let mut router = CognitiveRouter::new(0.3);
        router.register_module(Box::new(MathModule::new()));

        let result = router.route(&Input::text("2 + 3")).unwrap();
        assert_eq!(result.module_id, ModuleId::Math);
        assert!(result.output.is_some());
    }

    #[test]
    fn test_route_time_series() {
        let mut router = CognitiveRouter::new(0.3);
        router.register_module(Box::new(TimeSeriesModule::new()));

        let result = router.route(&Input::sequence(vec![1.0, 2.0, 3.0, 4.0, 5.0])).unwrap();
        assert_eq!(result.module_id, ModuleId::TimeSeries);
        assert!(result.output.is_some());
    }

    #[test]
    fn test_route_image() {
        let mut router = CognitiveRouter::new(0.3);
        router.register_module(Box::new(VisionModule::new()));

        let result = router.route(&Input::image(28, 28, vec![0.5; 784])).unwrap();
        assert_eq!(result.module_id, ModuleId::Vision);
        assert!(result.output.is_some());
    }

    #[test]
    fn test_multi_module_selection() {
        let mut router = CognitiveRouter::new(0.3);
        router.register_module(Box::new(TextModule::new()));
        router.register_module(Box::new(MathModule::new()));
        router.register_module(Box::new(TimeSeriesModule::new()));
        router.register_module(Box::new(VisionModule::new()));

        // Text should route to Text module
        let result = router.route(&Input::text("Hello world")).unwrap();
        assert_eq!(result.module_id, ModuleId::Text);

        // Math should route to Math module
        let result = router.route(&Input::text("2 + 3")).unwrap();
        assert_eq!(result.module_id, ModuleId::Math);

        // Sequence should route to TimeSeries module
        let result = router.route(&Input::sequence(vec![1.0, 2.0, 3.0])).unwrap();
        assert_eq!(result.module_id, ModuleId::TimeSeries);

        // Image should route to Vision module
        let result = router.route(&Input::image(28, 28, vec![0.5; 784])).unwrap();
        assert_eq!(result.module_id, ModuleId::Vision);
    }

    #[test]
    fn test_confidence_threshold() {
        let router = CognitiveRouter::new(0.99); // Very high threshold
        // No modules registered
        let result = router.route(&Input::text("test"));
        assert!(result.is_err());
    }

    #[test]
    fn test_no_modules_error() {
        let router = CognitiveRouter::new(0.5);
        let result = router.route(&Input::text("test"));
        assert!(matches!(result, Err(RouterError::NoModulesRegistered)));
    }

    #[test]
    fn test_math_module_evaluation() {
        let module = MathModule::new();
        let result = module.process(&Input::text("2 + 3")).unwrap();
        assert!(result.contains("5"));
    }

    #[test]
    fn test_time_series_prediction() {
        let module = TimeSeriesModule::new();
        let result = module.process(&Input::sequence(vec![1.0, 2.0, 3.0, 4.0, 5.0])).unwrap();
        assert!(result.contains("next=6")); // Linear extrapolation: 5 + 1 = 6
    }
}
