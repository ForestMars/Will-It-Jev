// src/pipeline/mod.rs
//
// Multi-task inference substrate. Each Task owns its ordered Cascade. The
// CascadeGraph compiles the global union of shared tier instances and records
// task-labelled progression edges; runtime scheduling advances task paths
// independently over those shared nodes.

pub mod cascade;
pub mod config;
pub mod registry;
pub mod runner;
pub mod tagger;
pub mod task;
pub mod tier;

pub use registry::TierRegistry;
pub use runner::CascadeRunner;
