//! # grapheme-polish
//!
//! Layer 2: Polish notation intermediate representation.
//!
//! This crate provides:
//! - Unambiguous Polish (prefix) notation
//! - Bidirectional conversion: text <-> graph
//! - Direct graph mapping for expressions
//! - Optimization passes
//!
//! Polish notation is ideal for graph representation because:
//! - No parentheses needed (unambiguous)
//! - Natural tree/graph structure
//! - Easy to parse and generate

use grapheme_engine::{Expr, MathFn, MathOp, Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors during Polish notation processing
#[derive(Error, Debug)]
pub enum PolishError {
    #[error("Parse error at position {position}: {message}")]
    ParseError { position: usize, message: String },
    #[error("Unexpected end of input")]
    UnexpectedEof,
    #[error("Unknown operator: {0}")]
    UnknownOperator(String),
    #[error("Unknown function: {0}")]
    UnknownFunction(String),
    #[error("Invalid token: {0}")]
    InvalidToken(String),
}

/// Result type for Polish operations
pub type PolishResult<T> = Result<T, PolishError>;

/// A token in Polish notation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Token {
    Number(f64),
    Symbol(String),
    Operator(MathOp),
    Function(MathFn),
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
}

/// Parser for Polish notation expressions
#[derive(Debug, Default)]
pub struct PolishParser {
    tokens: Vec<Token>,
    position: usize,
}

impl PolishParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokenize a Polish notation string
    pub fn tokenize(&mut self, input: &str) -> PolishResult<Vec<Token>> {
        let mut tokens = Vec::new();
        let mut chars = input.chars().peekable();

        while let Some(&ch) = chars.peek() {
            match ch {
                ' ' | '\t' | '\n' => {
                    chars.next();
                }
                '(' => {
                    tokens.push(Token::OpenParen);
                    chars.next();
                }
                ')' => {
                    tokens.push(Token::CloseParen);
                    chars.next();
                }
                '[' => {
                    tokens.push(Token::OpenBracket);
                    chars.next();
                }
                ']' => {
                    tokens.push(Token::CloseBracket);
                    chars.next();
                }
                '+' => {
                    tokens.push(Token::Operator(MathOp::Add));
                    chars.next();
                }
                '-' => {
                    chars.next();
                    // Check if it's a negative number
                    if chars.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        let num = self.read_number(&mut chars, true)?;
                        tokens.push(Token::Number(num));
                    } else {
                        tokens.push(Token::Operator(MathOp::Sub));
                    }
                }
                '*' => {
                    tokens.push(Token::Operator(MathOp::Mul));
                    chars.next();
                }
                '/' => {
                    tokens.push(Token::Operator(MathOp::Div));
                    chars.next();
                }
                '^' => {
                    tokens.push(Token::Operator(MathOp::Pow));
                    chars.next();
                }
                '%' => {
                    tokens.push(Token::Operator(MathOp::Mod));
                    chars.next();
                }
                '0'..='9' | '.' => {
                    let num = self.read_number(&mut chars, false)?;
                    tokens.push(Token::Number(num));
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    let ident = self.read_identifier(&mut chars);
                    tokens.push(self.classify_identifier(&ident)?);
                }
                _ => {
                    return Err(PolishError::InvalidToken(ch.to_string()));
                }
            }
        }

        self.tokens = tokens.clone();
        Ok(tokens)
    }

    fn read_number(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
        negative: bool,
    ) -> PolishResult<f64> {
        let mut num_str = if negative {
            "-".to_string()
        } else {
            String::new()
        };

        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                num_str.push(ch);
                chars.next();
            } else {
                break;
            }
        }

        num_str
            .parse()
            .map_err(|_| PolishError::InvalidToken(num_str))
    }

    fn read_identifier(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
        let mut ident = String::new();

        while let Some(&ch) = chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                chars.next();
            } else {
                break;
            }
        }

        ident
    }

    fn classify_identifier(&self, ident: &str) -> PolishResult<Token> {
        match ident.to_lowercase().as_str() {
            "sin" => Ok(Token::Function(MathFn::Sin)),
            "cos" => Ok(Token::Function(MathFn::Cos)),
            "tan" => Ok(Token::Function(MathFn::Tan)),
            "log" => Ok(Token::Function(MathFn::Log)),
            "ln" => Ok(Token::Function(MathFn::Ln)),
            "exp" => Ok(Token::Function(MathFn::Exp)),
            "sqrt" => Ok(Token::Function(MathFn::Sqrt)),
            "abs" => Ok(Token::Function(MathFn::Abs)),
            "floor" => Ok(Token::Function(MathFn::Floor)),
            "ceil" => Ok(Token::Function(MathFn::Ceil)),
            "derive" => Ok(Token::Function(MathFn::Derive)),
            "integrate" => Ok(Token::Function(MathFn::Integrate)),
            _ => Ok(Token::Symbol(ident.to_string())),
        }
    }

    /// Parse tokens into an expression tree
    pub fn parse(&mut self, input: &str) -> PolishResult<Expr> {
        self.tokenize(input)?;
        self.position = 0;
        self.parse_expr()
    }

    fn parse_expr(&mut self) -> PolishResult<Expr> {
        let token = self.next_token()?;

        match token {
            Token::Number(n) => Ok(Expr::Value(Value::Float(n))),
            Token::Symbol(s) => Ok(Expr::Value(Value::Symbol(s))),
            Token::OpenParen => {
                let expr = self.parse_sexp()?;
                self.expect_token(&Token::CloseParen)?;
                Ok(expr)
            }
            _ => Err(PolishError::ParseError {
                position: self.position,
                message: format!("Unexpected token: {:?}", token),
            }),
        }
    }

    fn parse_sexp(&mut self) -> PolishResult<Expr> {
        let token = self.next_token()?;

        match token {
            Token::Operator(op) => {
                let left = self.parse_expr()?;
                let right = self.parse_expr()?;
                Ok(Expr::BinOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }
            Token::Function(func) => {
                let mut args = Vec::new();
                while self.peek_token() != Some(&Token::CloseParen) {
                    args.push(self.parse_expr()?);
                }
                Ok(Expr::Function { func, args })
            }
            _ => Err(PolishError::ParseError {
                position: self.position,
                message: format!("Expected operator or function, got {:?}", token),
            }),
        }
    }

    fn next_token(&mut self) -> PolishResult<Token> {
        if self.position >= self.tokens.len() {
            return Err(PolishError::UnexpectedEof);
        }
        let token = self.tokens[self.position].clone();
        self.position += 1;
        Ok(token)
    }

    fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn expect_token(&mut self, expected: &Token) -> PolishResult<()> {
        let token = self.next_token()?;
        if &token == expected {
            Ok(())
        } else {
            Err(PolishError::ParseError {
                position: self.position,
                message: format!("Expected {:?}, got {:?}", expected, token),
            })
        }
    }
}

/// Convert an expression to Polish notation string
pub fn expr_to_polish(expr: &Expr) -> String {
    match expr {
        Expr::Value(Value::Integer(i)) => i.to_string(),
        Expr::Value(Value::Float(f)) => f.to_string(),
        Expr::Value(Value::Symbol(s)) => s.clone(),
        Expr::Value(Value::Rational(n, d)) => format!("(/ {} {})", n, d),
        Expr::BinOp { op, left, right } => {
            let op_str = match op {
                MathOp::Add => "+",
                MathOp::Sub => "-",
                MathOp::Mul => "*",
                MathOp::Div => "/",
                MathOp::Pow => "^",
                MathOp::Mod => "%",
                MathOp::Neg => "-",
            };
            format!(
                "({} {} {})",
                op_str,
                expr_to_polish(left),
                expr_to_polish(right)
            )
        }
        Expr::UnaryOp { op, operand } => {
            let op_str = match op {
                MathOp::Neg => "-",
                _ => "?",
            };
            format!("({} {})", op_str, expr_to_polish(operand))
        }
        Expr::Function { func, args } => {
            let func_str = match func {
                MathFn::Sin => "sin",
                MathFn::Cos => "cos",
                MathFn::Tan => "tan",
                MathFn::Log => "log",
                MathFn::Ln => "ln",
                MathFn::Exp => "exp",
                MathFn::Sqrt => "sqrt",
                MathFn::Abs => "abs",
                MathFn::Floor => "floor",
                MathFn::Ceil => "ceil",
                MathFn::Derive => "derive",
                MathFn::Integrate => "integrate",
            };
            let args_str: Vec<String> = args.iter().map(expr_to_polish).collect();
            format!("({} {})", func_str, args_str.join(" "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let mut parser = PolishParser::new();
        let tokens = parser.tokenize("(+ 1 2)").unwrap();

        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], Token::OpenParen);
        assert_eq!(tokens[1], Token::Operator(MathOp::Add));
        assert_eq!(tokens[2], Token::Number(1.0));
        assert_eq!(tokens[3], Token::Number(2.0));
        assert_eq!(tokens[4], Token::CloseParen);
    }

    #[test]
    fn test_parse_simple() {
        let mut parser = PolishParser::new();
        let expr = parser.parse("(+ 2 3)").unwrap();

        match expr {
            Expr::BinOp { op, left, right } => {
                assert_eq!(op, MathOp::Add);
                assert_eq!(*left, Expr::Value(Value::Float(2.0)));
                assert_eq!(*right, Expr::Value(Value::Float(3.0)));
            }
            _ => panic!("Expected BinOp"),
        }
    }

    #[test]
    fn test_roundtrip() {
        let expr = Expr::BinOp {
            op: MathOp::Mul,
            left: Box::new(Expr::BinOp {
                op: MathOp::Add,
                left: Box::new(Expr::Value(Value::Integer(2))),
                right: Box::new(Expr::Value(Value::Integer(3))),
            }),
            right: Box::new(Expr::Value(Value::Integer(4))),
        };

        let polish = expr_to_polish(&expr);
        assert_eq!(polish, "(* (+ 2 3) 4)");
    }
}
