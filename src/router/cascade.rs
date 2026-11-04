// src/router/cascade.rs
//
// Backend identity is a dispatch concern. Classification and task resolution
// are handled by pipeline::CascadeRunner.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Jev,
    Llm,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backend::Jev => write!(f, "jev"),
            Backend::Llm => write!(f, "llm"),
        }
    }
}
