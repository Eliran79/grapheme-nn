//! Curriculum Learning Module
//!
//! Defines curriculum levels for progressive training from simple to complex.
//! Per GRAPHEME_Math_Dataset.md specification.

use grapheme_engine::{MathFn, MathOp};
use serde::{Deserialize, Serialize};

/// Output type for training examples
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// Numeric result only (e.g., 5)
    Numeric,
    /// Symbolic result only (e.g., (* 2 x))
    Symbolic,
    /// Both numeric and symbolic
    Both,
}

/// Dataset format enumeration for auto-detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DatasetFormat {
    /// Math curriculum format: input_polish, expected_symbolic/expected_result
    #[default]
    MathCurriculum,
    /// Text pairs format: input, target (for QA, kindergarten, etc.)
    TextPairs,
}

/// Specification for a curriculum level
#[derive(Debug, Clone)]
pub struct LevelSpec {
    /// Level number (1-7)
    pub level: u8,
    /// Allowed operations
    pub ops: Vec<MathOp>,
    /// Allowed functions
    pub functions: Vec<MathFn>,
    /// Maximum expression depth
    pub max_depth: usize,
    /// Whether symbols are allowed
    pub allow_symbols: bool,
    /// Output type
    pub output: OutputType,
    /// Number of samples to generate
    pub samples: usize,
}

impl LevelSpec {
    /// Level 1: Basic arithmetic
    pub fn level_1() -> Self {
        Self {
            level: 1,
            ops: vec![MathOp::Add, MathOp::Sub],
            functions: vec![],
            max_depth: 1,
            allow_symbols: false,
            output: OutputType::Numeric,
            samples: 10_000,
        }
    }

    /// Level 2: Nested arithmetic
    pub fn level_2() -> Self {
        Self {
            level: 2,
            ops: vec![MathOp::Add, MathOp::Sub, MathOp::Mul, MathOp::Div],
            functions: vec![],
            max_depth: 3,
            allow_symbols: false,
            output: OutputType::Numeric,
            samples: 50_000,
        }
    }

    /// Level 3: Symbolic substitution
    pub fn level_3() -> Self {
        Self {
            level: 3,
            ops: vec![
                MathOp::Add,
                MathOp::Sub,
                MathOp::Mul,
                MathOp::Div,
                MathOp::Pow,
            ],
            functions: vec![],
            max_depth: 3,
            allow_symbols: true,
            output: OutputType::Numeric,
            samples: 50_000,
        }
    }

    /// Level 4: Functions
    pub fn level_4() -> Self {
        Self {
            level: 4,
            ops: vec![
                MathOp::Add,
                MathOp::Sub,
                MathOp::Mul,
                MathOp::Div,
                MathOp::Pow,
            ],
            functions: vec![
                MathFn::Sin,
                MathFn::Cos,
                MathFn::Tan,
                MathFn::Log,
                MathFn::Exp,
                MathFn::Sqrt,
            ],
            max_depth: 3,
            allow_symbols: true,
            output: OutputType::Numeric,
            samples: 100_000,
        }
    }

    /// Level 5: Symbolic differentiation
    pub fn level_5() -> Self {
        Self {
            level: 5,
            ops: vec![
                MathOp::Add,
                MathOp::Sub,
                MathOp::Mul,
                MathOp::Div,
                MathOp::Pow,
            ],
            functions: vec![MathFn::Derive],
            max_depth: 4,
            allow_symbols: true,
            output: OutputType::Symbolic,
            samples: 100_000,
        }
    }

    /// Level 6: Integration
    pub fn level_6() -> Self {
        Self {
            level: 6,
            ops: vec![
                MathOp::Add,
                MathOp::Sub,
                MathOp::Mul,
                MathOp::Div,
                MathOp::Pow,
            ],
            functions: vec![MathFn::Integrate],
            max_depth: 4,
            allow_symbols: true,
            output: OutputType::Both,
            samples: 100_000,
        }
    }

    /// Level 7: Equation solving
    pub fn level_7() -> Self {
        Self {
            level: 7,
            ops: vec![
                MathOp::Add,
                MathOp::Sub,
                MathOp::Mul,
                MathOp::Div,
                MathOp::Pow,
            ],
            functions: vec![],
            max_depth: 4,
            allow_symbols: true,
            output: OutputType::Numeric,
            samples: 100_000,
        }
    }

    /// Get all level specs
    pub fn all_levels() -> Vec<Self> {
        vec![
            Self::level_1(),
            Self::level_2(),
            Self::level_3(),
            Self::level_4(),
            Self::level_5(),
            Self::level_6(),
            Self::level_7(),
        ]
    }

    /// Get a level spec by number
    pub fn by_level(level: u8) -> Option<Self> {
        match level {
            1 => Some(Self::level_1()),
            2 => Some(Self::level_2()),
            3 => Some(Self::level_3()),
            4 => Some(Self::level_4()),
            5 => Some(Self::level_5()),
            6 => Some(Self::level_6()),
            7 => Some(Self::level_7()),
            _ => None,
        }
    }
}

/// Curriculum level enum for runtime selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurriculumLevel {
    /// Level 1: Basic arithmetic (add, sub)
    Level1,
    /// Level 2: All arithmetic, nested
    Level2,
    /// Level 3: Symbolic variables
    Level3,
    /// Level 4: Transcendental functions
    Level4,
    /// Level 5: Differentiation
    Level5,
    /// Level 6: Integration
    Level6,
    /// Level 7: Equation solving
    Level7,
}

impl CurriculumLevel {
    /// Get level number
    pub fn level_number(&self) -> u8 {
        match self {
            Self::Level1 => 1,
            Self::Level2 => 2,
            Self::Level3 => 3,
            Self::Level4 => 4,
            Self::Level5 => 5,
            Self::Level6 => 6,
            Self::Level7 => 7,
        }
    }

    /// Get level spec
    pub fn spec(&self) -> LevelSpec {
        LevelSpec::by_level(self.level_number()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_specs() {
        let levels = LevelSpec::all_levels();
        assert_eq!(levels.len(), 7);
        assert_eq!(levels[0].level, 1);
        assert_eq!(levels[6].level, 7);
    }

    #[test]
    fn test_level_by_number() {
        assert!(LevelSpec::by_level(1).is_some());
        assert!(LevelSpec::by_level(7).is_some());
        assert!(LevelSpec::by_level(8).is_none());
    }

    #[test]
    fn test_curriculum_level() {
        assert_eq!(CurriculumLevel::Level1.level_number(), 1);
        assert_eq!(CurriculumLevel::Level7.level_number(), 7);
    }
}
