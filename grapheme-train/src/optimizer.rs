//! Optimizer Module
//!
//! Provides SGD, Adam, and learning rate schedulers for training.
//! Backend-028: Optimizer implementations.

use grapheme_core::{Persistable, PersistenceError};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

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

// Persistence implementations

impl Persistable for SGD {
    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PersistenceError> {
        serde_json::from_slice(bytes).map_err(|e| PersistenceError::DeserializationError {
            message: e.to_string(),
        })
    }
}

impl Persistable for Adam {
    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PersistenceError> {
        serde_json::from_slice(bytes).map_err(|e| PersistenceError::DeserializationError {
            message: e.to_string(),
        })
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
        assert_eq!(scheduler.get_lr(1.0, 0), 1.0);
        assert_eq!(scheduler.get_lr(1.0, 10), 0.1);
        assert_eq!(scheduler.get_lr(1.0, 20), 0.01);
    }

    #[test]
    fn test_lr_scheduler_warmup() {
        let scheduler = LRScheduler::WarmupLR { warmup_steps: 5 };
        assert!((scheduler.get_lr(1.0, 0) - 0.2).abs() < 1e-6);
        assert!((scheduler.get_lr(1.0, 4) - 1.0).abs() < 1e-6);
        assert_eq!(scheduler.get_lr(1.0, 10), 1.0);
    }
}
