//! Optimizer Module
//!
//! Provides SGD, Adam, gradient clipping, and learning rate schedulers for training.
//! Backend-028: Optimizer implementations.
//! Backend-193: Gradient clipping for training stability.

use grapheme_core::{Persistable, PersistenceError};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

// ============================================================================
// GRADIENT CLIPPING (backend-193)
// ============================================================================

/// Gradient clipping configuration for training stability
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GradientClipping {
    /// Maximum gradient norm (for norm clipping)
    pub max_norm: Option<f32>,
    /// Maximum absolute gradient value (for value clipping)
    pub max_value: Option<f32>,
}

impl Default for GradientClipping {
    fn default() -> Self {
        Self {
            max_norm: None,
            max_value: None,
        }
    }
}

impl GradientClipping {
    /// Create gradient clipping with max norm (most common)
    pub fn by_norm(max_norm: f32) -> Self {
        Self {
            max_norm: Some(max_norm),
            max_value: None,
        }
    }

    /// Create gradient clipping with max absolute value
    pub fn by_value(max_value: f32) -> Self {
        Self {
            max_norm: None,
            max_value: Some(max_value),
        }
    }

    /// Create gradient clipping with both norm and value clipping
    pub fn combined(max_norm: f32, max_value: f32) -> Self {
        Self {
            max_norm: Some(max_norm),
            max_value: Some(max_value),
        }
    }

    /// Apply gradient clipping to a gradient array
    pub fn clip(&self, grads: &Array2<f32>) -> Array2<f32> {
        let mut clipped = grads.clone();

        // First apply value clipping (element-wise)
        if let Some(max_val) = self.max_value {
            clipped.mapv_inplace(|g| g.clamp(-max_val, max_val));
        }

        // Then apply norm clipping (scales entire gradient)
        if let Some(max_norm) = self.max_norm {
            let norm = compute_gradient_norm(&clipped);
            if norm > max_norm && norm > 1e-8 {
                let scale = max_norm / norm;
                clipped.mapv_inplace(|g| g * scale);
            }
        }

        clipped
    }

    /// Check if any clipping is enabled
    pub fn is_enabled(&self) -> bool {
        self.max_norm.is_some() || self.max_value.is_some()
    }
}

/// Compute L2 norm of gradient array
pub fn compute_gradient_norm(grads: &Array2<f32>) -> f32 {
    grads.iter().map(|&g| g * g).sum::<f32>().sqrt()
}

/// Compute gradient statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct GradientStats {
    /// L2 norm of gradients
    pub norm: f32,
    /// Maximum absolute gradient value
    pub max_abs: f32,
    /// Minimum absolute gradient value (non-zero)
    pub min_abs: f32,
    /// Mean gradient value
    pub mean: f32,
    /// Whether gradients were clipped
    pub was_clipped: bool,
}

impl GradientStats {
    /// Compute gradient statistics
    pub fn compute(grads: &Array2<f32>) -> Self {
        let n = grads.len() as f32;
        let norm = compute_gradient_norm(grads);
        let max_abs = grads.iter().map(|&g| g.abs()).fold(0.0f32, f32::max);
        let min_abs = grads
            .iter()
            .map(|&g| g.abs())
            .filter(|&g| g > 1e-10)
            .fold(f32::MAX, f32::min);
        let mean = grads.iter().sum::<f32>() / n;

        Self {
            norm,
            max_abs,
            min_abs: if min_abs == f32::MAX { 0.0 } else { min_abs },
            mean,
            was_clipped: false,
        }
    }

    /// Compute stats and check if clipping occurred
    pub fn compute_with_clipping(
        original: &Array2<f32>,
        clipped: &Array2<f32>,
        clip_config: &GradientClipping,
    ) -> Self {
        let mut stats = Self::compute(clipped);

        // Check if clipping occurred
        if clip_config.is_enabled() {
            let orig_norm = compute_gradient_norm(original);
            let clipped_norm = compute_gradient_norm(clipped);
            stats.was_clipped = (orig_norm - clipped_norm).abs() > 1e-6;
        }

        stats
    }
}

/// Trait for optimizers that update parameters
pub trait Optimizer {
    /// Perform a single optimization step
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>);

    /// Zero out gradients (called at start of each iteration)
    fn zero_grad(&mut self);

    /// Get current learning rate
    fn get_lr(&self) -> f32;

    /// Set learning rate
    fn set_lr(&mut self, lr: f32);
}

/// Stochastic Gradient Descent with optional momentum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGD {
    /// Learning rate
    pub lr: f32,
    /// Momentum coefficient (0 = no momentum)
    pub momentum: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Velocity buffer for momentum
    velocity: Option<Array2<f32>>,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocity: None,
        }
    }

    /// Add momentum to the optimizer
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Add weight decay (L2 regularization)
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) {
        // Apply weight decay
        let mut adjusted_grads = if self.weight_decay > 0.0 {
            grads + &(params.clone() * self.weight_decay)
        } else {
            grads.clone()
        };

        // Apply momentum if enabled
        if self.momentum > 0.0 {
            if self.velocity.is_none() {
                self.velocity = Some(Array2::zeros(params.dim()));
            }

            if let Some(ref mut v) = self.velocity {
                *v = &*v * self.momentum + &adjusted_grads;
                adjusted_grads = v.clone();
            }
        }

        // Update parameters: params -= lr * grads
        *params = &*params - &(&adjusted_grads * self.lr);
    }

    fn zero_grad(&mut self) {
        // SGD doesn't need to zero anything, momentum is preserved
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Adam optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam {
    /// Learning rate
    pub lr: f32,
    /// Beta1 (exponential decay rate for first moment)
    pub beta1: f32,
    /// Beta2 (exponential decay rate for second moment)
    pub beta2: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// First moment estimate
    m: Option<Array2<f32>>,
    /// Second moment estimate
    v: Option<Array2<f32>>,
    /// Timestep
    t: usize,
}

impl Adam {
    /// Create a new Adam optimizer with default parameters
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: None,
            v: None,
            t: 0,
        }
    }

    /// Set beta1
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Add weight decay
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Get the current timestep (number of optimizer steps taken)
    pub fn timestep(&self) -> usize {
        self.t
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) {
        self.t += 1;

        // Initialize moment estimates if needed
        if self.m.is_none() {
            self.m = Some(Array2::zeros(params.dim()));
        }
        if self.v.is_none() {
            self.v = Some(Array2::zeros(params.dim()));
        }

        // Apply weight decay (decoupled, as in AdamW)
        if self.weight_decay > 0.0 {
            *params = &*params - &(params.clone() * (self.lr * self.weight_decay));
        }

        // Update biased first moment estimate
        if let Some(ref mut m) = self.m {
            *m = &*m * self.beta1 + &(grads * (1.0 - self.beta1));
        }

        // Update biased second moment estimate
        if let Some(ref mut v) = self.v {
            let grads_sq = grads.mapv(|x| x * x);
            *v = &*v * self.beta2 + &(grads_sq * (1.0 - self.beta2));
        }

        // Compute bias-corrected estimates
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        if let (Some(ref m), Some(ref v)) = (&self.m, &self.v) {
            let m_hat = m / bias_correction1;
            let v_hat = v / bias_correction2;

            // Update parameters
            let denom = v_hat.mapv(|x| x.sqrt() + self.epsilon);
            let update = m_hat / denom;
            *params = &*params - &(update * self.lr);
        }
    }

    fn zero_grad(&mut self) {
        // Adam keeps momentum, but this resets for fresh training
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Learning rate scheduler types
#[derive(Debug, Clone)]
pub enum LRScheduler {
    /// Constant learning rate (no decay)
    Constant,
    /// Step decay: lr = lr * gamma every step_size epochs
    StepLR { step_size: usize, gamma: f32 },
    /// Exponential decay: lr = lr * gamma^epoch
    ExponentialLR { gamma: f32 },
    /// Cosine annealing: lr oscillates from max to min
    CosineAnnealingLR { t_max: usize, eta_min: f32 },
    /// Linear warmup: lr increases linearly for warmup_steps, then constant
    WarmupLR { warmup_steps: usize },
    /// Warmup then cosine decay
    WarmupCosineDecay {
        warmup_steps: usize,
        total_steps: usize,
        eta_min: f32,
    },
}

impl LRScheduler {
    /// Compute learning rate for given epoch
    pub fn get_lr(&self, base_lr: f32, epoch: usize) -> f32 {
        match self {
            LRScheduler::Constant => base_lr,

            LRScheduler::StepLR { step_size, gamma } => {
                let num_decays = epoch / step_size;
                base_lr * gamma.powi(num_decays as i32)
            }

            LRScheduler::ExponentialLR { gamma } => base_lr * gamma.powi(epoch as i32),

            LRScheduler::CosineAnnealingLR { t_max, eta_min } => {
                let t = (epoch % t_max) as f32;
                let t_max = *t_max as f32;
                eta_min
                    + (base_lr - eta_min) * (1.0 + (std::f32::consts::PI * t / t_max).cos()) / 2.0
            }

            LRScheduler::WarmupLR { warmup_steps } => {
                if epoch < *warmup_steps {
                    base_lr * (epoch + 1) as f32 / *warmup_steps as f32
                } else {
                    base_lr
                }
            }

            LRScheduler::WarmupCosineDecay {
                warmup_steps,
                total_steps,
                eta_min,
            } => {
                if epoch < *warmup_steps {
                    base_lr * (epoch + 1) as f32 / *warmup_steps as f32
                } else {
                    let t = (epoch - warmup_steps) as f32;
                    let t_max = (total_steps - warmup_steps) as f32;
                    eta_min
                        + (base_lr - eta_min) * (1.0 + (std::f32::consts::PI * t / t_max).cos())
                            / 2.0
                }
            }
        }
    }
}

// ============================================================================
// GRADIENT ACCUMULATION (backend-197)
// ============================================================================

/// Configuration for gradient accumulation
/// Allows training with effectively larger batch sizes by accumulating gradients
/// over multiple micro-batches before performing an optimizer step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAccumulator {
    /// Number of micro-batches to accumulate before updating
    pub accumulation_steps: usize,
    /// Current accumulated gradients
    accumulated_grads: Option<Array2<f32>>,
    /// Number of micro-batches accumulated so far
    current_step: usize,
    /// Normalize gradients by number of accumulation steps
    pub normalize: bool,
}

impl GradientAccumulator {
    /// Create a new gradient accumulator
    ///
    /// # Arguments
    /// * `accumulation_steps` - Number of micro-batches to accumulate (effective batch = actual batch * accumulation_steps)
    pub fn new(accumulation_steps: usize) -> Self {
        Self {
            accumulation_steps: accumulation_steps.max(1),
            accumulated_grads: None,
            current_step: 0,
            normalize: true,
        }
    }

    /// Create accumulator without gradient normalization
    pub fn without_normalization(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Accumulate gradients from a micro-batch
    /// Returns true if optimizer step should be performed (all micro-batches accumulated)
    pub fn accumulate(&mut self, grads: &Array2<f32>) -> bool {
        self.current_step += 1;

        match &mut self.accumulated_grads {
            Some(acc) => {
                *acc = &*acc + grads;
            }
            None => {
                self.accumulated_grads = Some(grads.clone());
            }
        }

        self.current_step >= self.accumulation_steps
    }

    /// Get accumulated gradients for optimizer step
    /// Optionally normalizes by number of accumulation steps
    /// Returns None if no gradients have been accumulated
    pub fn get_gradients(&self) -> Option<Array2<f32>> {
        self.accumulated_grads.as_ref().map(|grads| {
            if self.normalize && self.accumulation_steps > 1 {
                grads / self.accumulation_steps as f32
            } else {
                grads.clone()
            }
        })
    }

    /// Reset accumulator for next round of micro-batches
    pub fn reset(&mut self) {
        self.accumulated_grads = None;
        self.current_step = 0;
    }

    /// Check if ready to perform optimizer step
    pub fn is_ready(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Get current accumulation progress
    pub fn progress(&self) -> (usize, usize) {
        (self.current_step, self.accumulation_steps)
    }

    /// Get effective batch size multiplier
    pub fn effective_batch_multiplier(&self) -> usize {
        self.accumulation_steps
    }
}

impl Default for GradientAccumulator {
    fn default() -> Self {
        Self::new(1) // No accumulation by default
    }
}

/// Wrapper that combines an optimizer with gradient accumulation
#[derive(Debug)]
pub struct AccumulatedOptimizer<O: Optimizer> {
    /// Inner optimizer
    pub optimizer: O,
    /// Gradient accumulator
    pub accumulator: GradientAccumulator,
    /// Gradient clipping configuration
    pub clipping: GradientClipping,
}

impl<O: Optimizer> AccumulatedOptimizer<O> {
    /// Create a new accumulated optimizer
    pub fn new(optimizer: O, accumulation_steps: usize) -> Self {
        Self {
            optimizer,
            accumulator: GradientAccumulator::new(accumulation_steps),
            clipping: GradientClipping::default(),
        }
    }

    /// Add gradient clipping
    pub fn with_clipping(mut self, clipping: GradientClipping) -> Self {
        self.clipping = clipping;
        self
    }

    /// Perform a micro-batch step (accumulate gradients)
    /// Returns true if optimizer step was performed
    pub fn micro_step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> bool {
        // Apply clipping to each micro-batch gradient
        let clipped = if self.clipping.is_enabled() {
            self.clipping.clip(grads)
        } else {
            grads.clone()
        };

        // Accumulate gradients
        if self.accumulator.accumulate(&clipped) {
            // All micro-batches accumulated, perform optimizer step
            if let Some(accumulated) = self.accumulator.get_gradients() {
                self.optimizer.step(params, &accumulated);
            }
            self.accumulator.reset();
            true
        } else {
            false
        }
    }

    /// Force an optimizer step with current accumulated gradients
    /// Useful for handling incomplete batches at end of epoch
    pub fn force_step(&mut self, params: &mut Array2<f32>) {
        if let Some(accumulated) = self.accumulator.get_gradients() {
            self.optimizer.step(params, &accumulated);
        }
        self.accumulator.reset();
    }

    /// Get gradient statistics from current accumulation
    pub fn accumulated_stats(&self) -> Option<GradientStats> {
        self.accumulator.get_gradients().map(|g| GradientStats::compute(&g))
    }
}

impl<O: Optimizer> Optimizer for AccumulatedOptimizer<O> {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) {
        // Direct step without accumulation (for compatibility)
        self.optimizer.step(params, grads);
    }

    fn zero_grad(&mut self) {
        self.optimizer.zero_grad();
        self.accumulator.reset();
    }

    fn get_lr(&self) -> f32 {
        self.optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }
}

// Persistence implementations

impl Persistable for SGD {
    fn persist_type_id() -> &'static str {
        "SGD_optimizer"
    }

    fn persist_version() -> u32 {
        2
    }

    fn validate(&self) -> Result<(), PersistenceError> {
        if self.lr < 0.0 {
            return Err(PersistenceError::ValidationFailed(
                "Learning rate cannot be negative".to_string(),
            ));
        }
        Ok(())
    }
}

impl Persistable for Adam {
    fn persist_type_id() -> &'static str {
        "Adam_optimizer"
    }

    fn persist_version() -> u32 {
        2
    }

    fn validate(&self) -> Result<(), PersistenceError> {
        if self.lr < 0.0 {
            return Err(PersistenceError::ValidationFailed(
                "Learning rate cannot be negative".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_sgd_basic() {
        let mut sgd = SGD::new(0.1);
        let mut params = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let grads = arr2(&[[0.1, 0.2], [0.3, 0.4]]);

        sgd.step(&mut params, &grads);

        // params = params - lr * grads = [[1-0.01, 2-0.02], [3-0.03, 4-0.04]]
        assert!((params[[0, 0]] - 0.99).abs() < 1e-6);
        assert!((params[[0, 1]] - 1.98).abs() < 1e-6);
    }

    #[test]
    fn test_adam_basic() {
        let mut adam = Adam::new(0.001);
        let mut params = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let grads = arr2(&[[0.1, 0.2], [0.3, 0.4]]);

        adam.step(&mut params, &grads);
        assert_eq!(adam.timestep(), 1);

        // Adam should have modified params
        assert!(params[[0, 0]] < 1.0);
    }

    #[test]
    fn test_lr_scheduler_constant() {
        let scheduler = LRScheduler::Constant;
        assert_eq!(scheduler.get_lr(0.1, 0), 0.1);
        assert_eq!(scheduler.get_lr(0.1, 100), 0.1);
    }

    #[test]
    fn test_lr_scheduler_step() {
        let scheduler = LRScheduler::StepLR {
            step_size: 10,
            gamma: 0.1,
        };
        assert!((scheduler.get_lr(1.0, 0) - 1.0).abs() < 1e-6);
        assert!((scheduler.get_lr(1.0, 10) - 0.1).abs() < 1e-6);
        assert!((scheduler.get_lr(1.0, 20) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_lr_scheduler_warmup() {
        let scheduler = LRScheduler::WarmupLR { warmup_steps: 5 };
        assert!((scheduler.get_lr(1.0, 0) - 0.2).abs() < 1e-6);
        assert!((scheduler.get_lr(1.0, 4) - 1.0).abs() < 1e-6);
        assert_eq!(scheduler.get_lr(1.0, 10), 1.0);
    }

    // ============================================================================
    // GRADIENT CLIPPING TESTS (backend-193)
    // ============================================================================

    #[test]
    fn test_gradient_clipping_by_norm() {
        let clip = GradientClipping::by_norm(1.0);

        // Create gradients with norm > 1.0
        let grads = arr2(&[[3.0, 4.0]]); // norm = 5.0
        let clipped = clip.clip(&grads);

        // After clipping, norm should be 1.0
        let clipped_norm = compute_gradient_norm(&clipped);
        assert!((clipped_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_clipping_by_value() {
        let clip = GradientClipping::by_value(0.5);

        let grads = arr2(&[[1.0, -2.0], [0.3, 3.0]]);
        let clipped = clip.clip(&grads);

        // All values should be in [-0.5, 0.5]
        assert!((clipped[[0, 0]] - 0.5).abs() < 1e-6);
        assert!((clipped[[0, 1]] - (-0.5)).abs() < 1e-6);
        assert!((clipped[[1, 0]] - 0.3).abs() < 1e-6);
        assert!((clipped[[1, 1]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_clipping_combined() {
        let clip = GradientClipping::combined(2.0, 0.8);

        let grads = arr2(&[[1.0, -2.0], [3.0, 4.0]]);
        let clipped = clip.clip(&grads);

        // First value clipping, then norm clipping
        let clipped_norm = compute_gradient_norm(&clipped);
        assert!(clipped_norm <= 2.0 + 1e-5);

        // No individual value should exceed 0.8 after value clipping
        // (though norm clipping may reduce them further)
    }

    #[test]
    fn test_gradient_clipping_no_clip_needed() {
        let clip = GradientClipping::by_norm(10.0);

        let grads = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
        let original_norm = compute_gradient_norm(&grads);
        let clipped = clip.clip(&grads);
        let clipped_norm = compute_gradient_norm(&clipped);

        // No clipping should occur
        assert!((original_norm - clipped_norm).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_clipping_disabled() {
        let clip = GradientClipping::default();
        assert!(!clip.is_enabled());

        let clip_norm = GradientClipping::by_norm(1.0);
        assert!(clip_norm.is_enabled());

        let clip_value = GradientClipping::by_value(1.0);
        assert!(clip_value.is_enabled());
    }

    #[test]
    fn test_gradient_stats() {
        let grads = arr2(&[[1.0, -2.0], [3.0, 4.0]]);
        let stats = GradientStats::compute(&grads);

        // norm = sqrt(1 + 4 + 9 + 16) = sqrt(30) ≈ 5.477
        assert!((stats.norm - 30.0_f32.sqrt()).abs() < 1e-4);
        assert!((stats.max_abs - 4.0).abs() < 1e-6);
        assert!((stats.min_abs - 1.0).abs() < 1e-6);
        assert!((stats.mean - 1.5).abs() < 1e-6); // (1-2+3+4)/4 = 1.5
    }

    #[test]
    fn test_gradient_stats_with_clipping() {
        let grads = arr2(&[[3.0, 4.0]]); // norm = 5.0
        let clip = GradientClipping::by_norm(1.0);
        let clipped = clip.clip(&grads);

        let stats = GradientStats::compute_with_clipping(&grads, &clipped, &clip);
        assert!(stats.was_clipped);
        assert!((stats.norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_gradient_norm() {
        let grads = arr2(&[[3.0, 4.0]]);
        let norm = compute_gradient_norm(&grads);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    // ============================================================================
    // GRADIENT ACCUMULATION TESTS (backend-197)
    // ============================================================================

    #[test]
    fn test_gradient_accumulator_basic() {
        let mut accumulator = GradientAccumulator::new(4);
        let grads = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        // First 3 steps should not be ready
        assert!(!accumulator.accumulate(&grads));
        assert!(!accumulator.is_ready());
        assert!(!accumulator.accumulate(&grads));
        assert!(!accumulator.accumulate(&grads));

        // Fourth step should trigger readiness
        assert!(accumulator.accumulate(&grads));
        assert!(accumulator.is_ready());
    }

    #[test]
    fn test_gradient_accumulator_sum() {
        let mut accumulator = GradientAccumulator::new(4).without_normalization();
        let grads = arr2(&[[1.0, 2.0]]);

        for _ in 0..4 {
            accumulator.accumulate(&grads);
        }

        let accumulated = accumulator.get_gradients().unwrap();
        // Sum of 4 identical gradients
        assert!((accumulated[[0, 0]] - 4.0).abs() < 1e-6);
        assert!((accumulated[[0, 1]] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_normalized() {
        let mut accumulator = GradientAccumulator::new(4); // normalize = true by default
        let grads = arr2(&[[4.0, 8.0]]);

        for _ in 0..4 {
            accumulator.accumulate(&grads);
        }

        let accumulated = accumulator.get_gradients().unwrap();
        // Average of 4 identical gradients (= original)
        assert!((accumulated[[0, 0]] - 4.0).abs() < 1e-6);
        assert!((accumulated[[0, 1]] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_reset() {
        let mut accumulator = GradientAccumulator::new(2);
        let grads = arr2(&[[1.0, 2.0]]);

        accumulator.accumulate(&grads);
        accumulator.accumulate(&grads);
        assert!(accumulator.is_ready());

        accumulator.reset();
        assert!(!accumulator.is_ready());
        assert_eq!(accumulator.progress(), (0, 2));
    }

    #[test]
    fn test_gradient_accumulator_progress() {
        let mut accumulator = GradientAccumulator::new(4);
        let grads = arr2(&[[1.0, 2.0]]);

        assert_eq!(accumulator.progress(), (0, 4));
        accumulator.accumulate(&grads);
        assert_eq!(accumulator.progress(), (1, 4));
        accumulator.accumulate(&grads);
        assert_eq!(accumulator.progress(), (2, 4));
    }

    #[test]
    fn test_accumulated_optimizer() {
        let sgd = SGD::new(0.1);
        let mut acc_opt = AccumulatedOptimizer::new(sgd, 4);
        let mut params = arr2(&[[10.0, 20.0]]);
        let grads = arr2(&[[1.0, 2.0]]);

        // First 3 micro-steps should not update params
        assert!(!acc_opt.micro_step(&mut params, &grads));
        assert_eq!(params[[0, 0]], 10.0);

        assert!(!acc_opt.micro_step(&mut params, &grads));
        assert!(!acc_opt.micro_step(&mut params, &grads));

        // Fourth micro-step should update params
        assert!(acc_opt.micro_step(&mut params, &grads));
        // params = params - lr * avg_grads = 10 - 0.1 * 1 = 9.9
        assert!((params[[0, 0]] - 9.9).abs() < 1e-6);
    }

    #[test]
    fn test_accumulated_optimizer_with_clipping() {
        let sgd = SGD::new(1.0);
        let mut acc_opt =
            AccumulatedOptimizer::new(sgd, 2).with_clipping(GradientClipping::by_value(0.5));
        let mut params = arr2(&[[0.0, 0.0]]);
        let grads = arr2(&[[10.0, 20.0]]); // Will be clipped to [0.5, 0.5]

        acc_opt.micro_step(&mut params, &grads);
        acc_opt.micro_step(&mut params, &grads);

        // Accumulated gradient = avg([0.5, 0.5], [0.5, 0.5]) = [0.5, 0.5]
        // params = 0 - 1.0 * 0.5 = -0.5
        assert!((params[[0, 0]] - (-0.5)).abs() < 1e-6);
        assert!((params[[0, 1]] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_accumulated_optimizer_force_step() {
        let sgd = SGD::new(0.1);
        let mut acc_opt = AccumulatedOptimizer::new(sgd, 4);
        let mut params = arr2(&[[10.0, 20.0]]);
        let grads = arr2(&[[1.0, 2.0]]);

        // Only accumulate 2 steps (partial accumulation)
        acc_opt.micro_step(&mut params, &grads);
        acc_opt.micro_step(&mut params, &grads);

        // Force step with partial accumulation
        acc_opt.force_step(&mut params);

        // accumulated = 2.0, normalized by 4 (not 2) = 0.5
        // params = 10 - 0.1 * 0.5 = 9.95
        assert!((params[[0, 0]] - 9.95).abs() < 1e-6);
    }

    #[test]
    fn test_effective_batch_multiplier() {
        let accumulator = GradientAccumulator::new(8);
        assert_eq!(accumulator.effective_batch_multiplier(), 8);
    }
}
