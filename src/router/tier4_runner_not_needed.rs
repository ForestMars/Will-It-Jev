//! Tier 4 intentionally reuses Tier3Runner.
//!
//! Tier 3 and Tier 4 are distinct cascade execution stages, but share
//! the same gRPC wire protocol and outcome structure.
//!
//! Tier 4 is instantiated as a separate Tier3Runner with its own
//! endpoint and configuration.
//!
//! A dedicated Tier4Runner implementation is therefore unnecessary.
//! This file is retained to document that architectural decision.
