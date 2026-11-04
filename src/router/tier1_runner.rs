// src/router/tier1_runner.rs

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use arc_swap::ArcSwap;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use strsim::damerau_levenshtein;
use tracing::{info, instrument, Span};

pub const BINARY_FASTPATH_MAX_CHARS: usize = 250;
pub const AMBIGUITY_LOWER_BOUND: f32 = 0.35;
pub const AMBIGUITY_UPPER_BOUND: f32 = 0.78;

#[derive(Debug, Clone, Deserialize)]
pub struct PatternGroup {
    #[serde(default)]
    pub exact: Vec<String>,
    #[serde(default)]
    pub fuzzy: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FuzzyConfig {
    #[serde(default = "default_max_distance")]
    pub max_distance: usize,
    #[serde(default = "default_imperative_zone_bytes")]
    pub imperative_zone_bytes: usize,
}

fn default_max_distance() -> usize {
    1
}
fn default_imperative_zone_bytes() -> usize {
    48
}

#[derive(Debug, Clone, Deserialize)]
pub struct Tier1Config {
    pub fuzzy: FuzzyConfig,
    pub jev_strong: PatternGroup,
    pub jev_moderate: PatternGroup,
    pub llm_strong: PatternGroup,
    pub llm_prefixes: PatternGroup,
    pub polar_starts: PatternGroup,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tier1Result {
    pub confidence: f32,
    pub reason: String,
    pub decided: bool,
}

/// Holds compiled Aho-Corasick automata and active pattern configurations.
pub struct Tier1Automaton {
    pub config: Tier1Config,
    pub jev_strong: AhoCorasick,
    pub jev_moderate: AhoCorasick,
    pub llm_strong: AhoCorasick,
    pub llm_prefixes: AhoCorasick,
    pub polar_starts: AhoCorasick,
}

pub struct Tier1Engine {
    automaton: ArcSwap<Tier1Automaton>,
}

fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut boundary = max_bytes;
    while boundary > 0 && !s.is_char_boundary(boundary) {
        boundary -= 1;
    }
    &s[..boundary]
}

fn is_prefix_match(text: &str, start_idx: usize) -> bool {
    if start_idx == 0 {
        return true;
    }
    text[..start_idx]
        .chars()
        .all(|c| c.is_whitespace() || c.is_ascii_punctuation())
}

const PURE_ARITHMETIC_PREFIXES: [&str; 4] = ["what is ", "what's ", "calculate ", "compute "];
const PURE_ARITHMETIC_ALLOWED_CHARS: &str = "0123456789.+-*/%() ?";

fn is_pure_arithmetic(prompt: &str) -> bool {
    let text = prompt.trim();
    if text.is_empty() {
        return false;
    }

    let lowered = text.to_lowercase();
    let expr = PURE_ARITHMETIC_PREFIXES
        .iter()
        .find_map(|prefix| lowered.strip_prefix(prefix))
        .unwrap_or(&lowered);

    if expr.is_empty() || expr.trim().is_empty() {
        return false;
    }

    let has_number = expr.chars().any(|c| c.is_ascii_digit());
    let has_operator = expr.contains('+')
        || expr.contains('-')
        || expr.contains('*')
        || expr.contains('/')
        || expr.contains('%');

    if !has_number || !has_operator {
        return false;
    }

    expr.chars()
        .all(|c| PURE_ARITHMETIC_ALLOWED_CHARS.contains(c))
}

impl Tier1Automaton {
    pub fn from_toml_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Tier1Config, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Tier1Config = toml::from_str(&content)?;
        Ok(config)
    }
}

impl Tier1Engine {
    pub fn new(config: Tier1Config) -> Result<Self, aho_corasick::BuildError> {
        let automaton = Self::build_automaton(config)?;
        Ok(Self {
            automaton: ArcSwap::from_pointee(automaton),
        })
    }

    #[allow(dead_code)]
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Tier1Config = serde_json::from_str(&content)?;
        Ok(Self::new(config)?)
    }

    fn build_automaton(config: Tier1Config) -> Result<Tier1Automaton, aho_corasick::BuildError> {
        let build_ac = |patterns: &[String]| {
            AhoCorasickBuilder::new()
                .ascii_case_insensitive(true)
                .match_kind(MatchKind::Standard)
                .build(patterns)
        };

        Ok(Tier1Automaton {
            jev_strong: build_ac(&config.jev_strong.exact)?,
            jev_moderate: build_ac(&config.jev_moderate.exact)?,
            llm_strong: build_ac(&config.llm_strong.exact)?,
            llm_prefixes: build_ac(&config.llm_prefixes.exact)?,
            polar_starts: build_ac(&config.polar_starts.exact)?,
            config,
        })
    }

    #[allow(dead_code)]
    pub fn update_config(&self, new_config: Tier1Config) -> Result<(), aho_corasick::BuildError> {
        let new_automaton = Self::build_automaton(new_config)?;
        self.automaton.store(Arc::new(new_automaton));
        Ok(())
    }

    #[instrument(
        name = "tier1_classify",
        skip(self, prompt),
        fields(
            tier = 1,
            engine = "aho_corasick_hybrid",
            prompt_len = prompt.len(),
            jev_strong_hits,
            jev_mod_hits,
            llm_strong_hits,
            matches_llm_prefix,
            starts_polar,
            is_binary_question,
            confidence,
            decided,
            outcome,
            reason
        )
    )]
    pub fn classify(&self, prompt: &str) -> Tier1Result {
        let text = prompt.trim();
        let current_span = Span::current();

        if text.is_empty() {
            current_span.record("confidence", 0.0f32);
            current_span.record("decided", true);
            current_span.record("outcome", "empty_input");
            current_span.record("reason", "empty prompt — defaulting to LLM");

            return Tier1Result {
                confidence: 0.0,
                reason: "empty prompt — defaulting to LLM".into(),
                decided: true,
            };
        }

        let guard = self.automaton.load();
        let max_bytes = guard.config.fuzzy.imperative_zone_bytes;
        let max_dist = guard.config.fuzzy.max_distance;

        // 1. Exact Aho-Corasick Positional Scans
        let matches_llm_prefix = guard
            .llm_prefixes
            .find_iter(text)
            .any(|m| is_prefix_match(text, m.start()));

        let starts_polar = guard
            .polar_starts
            .find_iter(text)
            .any(|m| is_prefix_match(text, m.start()));

        let mut jev_strong_hits = guard
            .jev_strong
            .find_iter(text)
            .filter(|m| m.start() <= max_bytes)
            .count();

        let jev_mod_hits = guard
            .jev_moderate
            .find_iter(text)
            .filter(|m| m.start() <= max_bytes)
            .count();

        let mut llm_strong_hits = guard
            .llm_strong
            .find_iter(text)
            .filter(|m| m.start() <= max_bytes)
            .count();

        // 2. Fuzzy Token Matcher over Imperative Zone
        if jev_strong_hits == 0 || llm_strong_hits == 0 {
            let imperative_slice = truncate_utf8(text, max_bytes);

            for token in imperative_slice.split_whitespace() {
                let clean_token = token
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();

                if clean_token.len() <= 3 {
                    continue;
                }

                if jev_strong_hits == 0 {
                    for target in &guard.config.jev_strong.fuzzy {
                        if damerau_levenshtein(&clean_token, target) <= max_dist {
                            jev_strong_hits += 1;
                            break;
                        }
                    }
                }

                if llm_strong_hits == 0 {
                    for target in &guard.config.llm_strong.fuzzy {
                        if damerau_levenshtein(&clean_token, target) <= max_dist {
                            llm_strong_hits += 1;
                            break;
                        }
                    }
                }
            }
        }

        let is_binary_question = starts_polar
            && text.ends_with('?')
            && text.len() <= BINARY_FASTPATH_MAX_CHARS
            && !matches_llm_prefix
            && llm_strong_hits == 0;

        current_span.record("jev_strong_hits", jev_strong_hits);
        current_span.record("jev_mod_hits", jev_mod_hits);
        current_span.record("llm_strong_hits", llm_strong_hits);
        current_span.record("matches_llm_prefix", matches_llm_prefix);
        current_span.record("starts_polar", starts_polar);
        current_span.record("is_binary_question", is_binary_question);

        // 3. Routing Policy Evaluation
        let result = if is_pure_arithmetic(text) {
            Tier1Result {
                confidence: 0.99,
                reason: "simple arithmetic question".into(),
                decided: true,
            }
        } else if matches_llm_prefix || llm_strong_hits >= 2 {
            Tier1Result {
                confidence: 0.1,
                reason: format!(
                    "LLM imperative signal (prefix={matches_llm_prefix}, hits={llm_strong_hits})"
                ),
                decided: true,
            }
        } else if llm_strong_hits >= 1 && jev_strong_hits == 0 && jev_mod_hits == 0 {
            Tier1Result {
                confidence: 0.2,
                reason: "LLM directive in imperative zone with zero Jev matches".into(),
                decided: true,
            }
        } else if llm_strong_hits >= 1 && (jev_strong_hits > 0 || jev_mod_hits > 0) {
            Tier1Result {
                confidence: 0.5,
                reason: "conflicting strong routing signals".into(),
                decided: false,
            }
        } else if is_binary_question {
            Tier1Result {
                confidence: 0.9,
                reason: "short polar yes/no question without generative markers".into(),
                decided: true,
            }
        } else if jev_strong_hits >= 2 {
            Tier1Result {
                confidence: 0.85,
                reason: format!("{jev_strong_hits} strong Jev matches in imperative zone"),
                decided: true,
            }
        } else {
            let combined_jev = (jev_strong_hits as f32 * 0.35) + (jev_mod_hits as f32 * 0.15);
            let combined_llm = llm_strong_hits as f32 * 0.3;

            let raw_confidence = 0.5 + combined_jev - combined_llm;
            let confidence = raw_confidence.clamp(0.0, 1.0);
            let decided = !(AMBIGUITY_LOWER_BOUND..=AMBIGUITY_UPPER_BOUND).contains(&confidence);

            Tier1Result {
                confidence,
                reason: if decided {
                    format!("heuristic confident: jev={jev_strong_hits}+{jev_mod_hits}, llm={llm_strong_hits}")
                } else {
                    format!("heuristic ambiguous: jev={jev_strong_hits}+{jev_mod_hits}, llm={llm_strong_hits}")
                },
                decided,
            }
        };

        current_span.record("confidence", result.confidence);
        current_span.record("decided", result.decided);
        current_span.record("reason", result.reason.as_str());

        if result.decided {
            current_span.record("outcome", "resolved");
            info!(
                confidence = result.confidence,
                reason = %result.reason,
                "Tier 1 short-circuited cascade"
            );
        } else {
            current_span.record("outcome", "ambiguous_passthrough");
            info!(
                confidence = result.confidence,
                reason = %result.reason,
                "Tier 1 ambiguous heuristic; passing through to Tier 2"
            );
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn sample_config() -> Tier1Config {
        Tier1Config {
            fuzzy: FuzzyConfig {
                max_distance: 1,
                imperative_zone_bytes: 48,
            },
            jev_strong: PatternGroup {
                exact: vec!["find".to_string(), "search".to_string()],
                fuzzy: vec!["search".to_string()],
            },
            jev_moderate: PatternGroup {
                exact: vec!["filter".to_string()],
                fuzzy: vec![],
            },
            llm_strong: PatternGroup {
                exact: vec!["explain".to_string(), "write".to_string()],
                fuzzy: vec!["explain".to_string()],
            },
            llm_prefixes: PatternGroup {
                exact: vec!["please write".to_string()],
                fuzzy: vec![],
            },
            polar_starts: PatternGroup {
                exact: vec!["is".to_string(), "can".to_string()],
                fuzzy: vec![],
            },
        }
    }

    #[test]
    fn test_truncate_utf8() {
        assert_eq!(truncate_utf8("hello", 10), "hello");
        assert_eq!(truncate_utf8("hello world", 5), "hello");

        let s = "hello 🦀 world";
        assert_eq!(truncate_utf8(s, 8), "hello ");
        assert_eq!(truncate_utf8(s, 10), "hello 🦀");
    }

    #[test]
    fn test_classify_empty() {
        let engine = Tier1Engine::new(sample_config()).unwrap();
        let res = engine.classify("");
        assert!(res.decided);
        assert_eq!(res.confidence, 0.0);
    }

    #[test]
    fn test_classify_binary_fastpath() {
        let engine = Tier1Engine::new(sample_config()).unwrap();
        let res = engine.classify("Is this working?");
        assert!(res.decided);
        assert_eq!(res.confidence, 0.9);
    }

    #[test]
    fn arithmetic_prompts_short_circuit_to_jev() {
        let config = Tier1Automaton::from_toml_file("config/tier1.toml").unwrap();
        let engine = Tier1Engine::new(config).unwrap();

        let result = engine.classify("What is 2+2?");
        assert!(result.decided, "2+2 should resolve at tier 1: {result:?}");
        assert!(
            result.confidence > 0.9,
            "expected a high-confidence Jev exit: {result:?}"
        );
    }

    #[test]
    fn test_is_pure_arithmetic_accepts_expression() {
        assert!(is_pure_arithmetic("What is 2+2?"));
    }

    #[test]
    fn test_is_pure_arithmetic_rejects_natural_language() {
        assert!(!is_pure_arithmetic(
            "What is a good 5-step plan for learning Rust?"
        ));
    }

    #[test]
    fn generative_prompts_defer_from_tier1() {
        let config = Tier1Automaton::from_toml_file("config/tier1.toml").unwrap();
        let engine = Tier1Engine::new(config).unwrap();

        for prompt in [
            "convert this paragraph into a bulleted list",
            "What's the weather like in Paris in spring, and what should I pack?",
        ] {
            let result = engine.classify(prompt);
            assert!(
                !result.decided || result.confidence <= AMBIGUITY_UPPER_BOUND,
                "Tier 1 incorrectly committed to Jev for {prompt:?}: {result:?}"
            );
        }
    }

    proptest! {
        #[test]
        fn test_truncate_utf8_never_panics(s in "\\PC*", max_bytes in 0..100usize) {
            let res = truncate_utf8(&s, max_bytes);
            prop_assert!(res.len() <= max_bytes);
        }
    }
}
