// src/pipeline/mod.rs
//
// Task-independent inference cascade. Task definitions, tier/task resolution
// policies, and execution-tier implementations are separate interfaces.

pub mod config;
pub mod runner;
pub mod tagger;
pub mod task;
pub mod tier;

pub use runner::CascadeRunner;
pub use task::TaskId;
