// src/pipeline/mod.rs
//
// Multi-task inference substrate. Each Task owns its ordered Cascade; the
// CascadeGraph compiles their shared tier nodes and ordering constraints.

pub mod cascade;
pub mod config;
pub mod registry;
pub mod runner;
pub mod tagger;
pub mod task;
pub mod tier;

pub use registry::TierRegistry;
pub use runner::CascadeRunner;
