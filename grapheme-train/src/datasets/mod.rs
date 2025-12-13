//! Dataset loaders for GRAPHEME training
//!
//! This module provides loaders for various external and internal datasets
//! used in training different cognitive brains.

pub mod gsm8k;
pub mod math_comp;
pub mod squad;
pub mod fashion_mnist;

pub use gsm8k::{Gsm8kExample, Gsm8kLoader, Gsm8kStats};
pub use math_comp::{MathCategory, MathExample, MathLoader, MathStats};
pub use squad::{SquadExample, SquadLoader, SquadStats};
pub use fashion_mnist::{FashionMnistExample, FashionMnistLoader, FashionMnistStats, FASHION_LABELS};
