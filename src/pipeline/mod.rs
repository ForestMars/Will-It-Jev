// src/pipeline/mod.rs
//
// Task-agnostic cascade engine.
//
// The six-tier cascade used to be welded to one question (jev_capable vs needs_llm):
// each tier applied its own thresholds and returned a route-shaped outcome. This
// module separates the two halves.
//
//   * A `Tier` is a scorer. Given a prompt and the set of tasks still open, it returns
//     a raw reading per task and applies no thresholds. It does not know which use
//     case is calling it.
//   * A `TaskProfile` is a use case. It says which tiers the task may exit at, the
//     confidence bar per tier and label, the allowed label set, and what happens if
//     no tier resolves it.
//   * The `CascadeRunner` walks the tiers once, in order, over the *union* of tasks
//     still open. A tier is called at most once per run, so two tasks share one
//     substrate execution; a task leaves the active set the moment its own profile
//     says a tier is confident enough, and later tiers refine only what is left.
//
// Whether two tasks cost one backend call or two is a property of the tier, not of
// the runner: a tier that implements joint scoring (see `tier::Tier3Tier`) serves
// all open tasks from a single shared encoding.

#![allow(dead_code)] // the route path still uses router::cascade::route; see README note

pub mod config;
pub mod runner;
pub mod tagger;
pub mod task;
pub mod tier;

pub use runner::CascadeRunner;
pub use task::TaskId;
