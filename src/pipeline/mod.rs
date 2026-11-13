// src/pipeline/mod.rs
//
// Task-independent inference cascade. Task definitions, tier/task resolution
// policies, and execution-tier implementations are separate interfaces.

pub mod cascade;
pub mod identity;
pub mod config;
pub mod registry;
pub mod runner;
pub mod tagger;
pub mod task;
pub mod tier;

pub use registry::TierRegistry;
pub use runner::CascadeRunner;
pub use task::Task;
