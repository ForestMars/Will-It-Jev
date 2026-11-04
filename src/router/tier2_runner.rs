// src/router/tier2_runner.rs

use fasttext::FastText;
use std::path::Path;

#[allow(unused_imports)]
use tracing::{info, instrument};

/// Labels the model must carry, as they appear after stripping `__label__`.
const REQUIRED_LABELS: [&str; 2] = ["jev", "llm"];
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.95;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier2Route {
    Jev,
    Llm,
}

#[derive(Debug, Clone)]
pub enum Tier2Outcome {
    Resolved {
        route: Tier2Route,
        confidence: f32,
        calibrated_score: f32,
        p_jev: f32,
        p_llm: f32,
        raw_probabilities: Vec<f32>,
    },
    PassThrough {
        top_guess: Tier2Route,
        score: f32,
        reason: &'static str,
        p_jev: f32,
        p_llm: f32,
        raw_probabilities: Vec<f32>,
    },
}

pub struct Tier2Runner {
    model: FastText,
    confidence_threshold: f32,
}

/// Pure scoring step. Returns (route, top_prob, calibrated_score).
///
/// calibrated_score is the margin between the two class probabilities, so it
/// stays strictly increasing in model certainty and only reaches 1.0 when the
/// model puts all mass on one label.
pub fn calibrate(p_jev: f32, p_llm: f32) -> (Tier2Route, f32, f32) {
    let p_jev = p_jev.clamp(0.0, 1.0);
    let p_llm = p_llm.clamp(0.0, 1.0);

    let margin = (p_jev - p_llm).abs();

    let (route, top_prob) = if p_jev >= p_llm {
        (Tier2Route::Jev, p_jev)
    } else {
        (Tier2Route::Llm, p_llm)
    };

    (route, top_prob, margin.clamp(0.0, 1.0))
}

/// Pure label check. Accepts labels with or without the `__label__` prefix and
/// fails with the list of what the model actually contains.
pub fn check_labels(labels: &[String]) -> Result<(), String> {
    let names: Vec<&str> = labels
        .iter()
        .map(|l| l.trim_start_matches("__label__"))
        .collect();

    let missing: Vec<&str> = REQUIRED_LABELS
        .iter()
        .copied()
        .filter(|req| !names.contains(req))
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Tier2 model is missing required label(s) {missing:?}; model contains {names:?}"
        ))
    }
}

impl Tier2Runner {
    pub fn normalize_input(input: &str) -> String {
        input
            .trim()
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn new<P: AsRef<Path>>(model_path: P, confidence_threshold: f32) -> Result<Self, String> {
        let model = FastText::load_model(&model_path)
            .map_err(|e| format!("Failed to load FastText model: {e}"))?;

        let (labels, _freqs) = model.get_labels();
        check_labels(&labels)?;

        Ok(Self {
            model,
            confidence_threshold,
        })
    }

    #[inline]
    pub fn evaluate(&self, prompt: &str) -> Tier2Outcome {
        // 1. Input Normalization pre-pass
        let normalized = Self::normalize_input(prompt);

        if normalized.is_empty() {
            return Tier2Outcome::PassThrough {
                top_guess: Tier2Route::Llm,
                score: 0.0,
                reason: "empty_input",
                p_jev: 0.0,
                p_llm: 0.0,
                raw_probabilities: Vec::new(),
            };
        }

        let predictions = self.model.predict(&normalized, 2, 0.0);
        if predictions.is_empty() {
            return Tier2Outcome::PassThrough {
                top_guess: Tier2Route::Llm,
                score: 0.0,
                reason: "no_predictions",
                p_jev: 0.0,
                p_llm: 0.0,
                raw_probabilities: Vec::new(),
            };
        }

        let raw_probabilities = predictions
            .iter()
            .map(|prediction| prediction.prob)
            .collect();

        // Ground truth trace: every label the model returned, with its raw probability.
        tracing::info!(
            "tier2 raw predictions: {:?}",
            predictions
                .iter()
                .map(|p| (p.label.as_str(), p.prob))
                .collect::<Vec<_>>()
        );

        // 2. Explicit Class Extraction by label name
        let mut p_jev = 0.0f32;
        let mut p_llm = 0.0f32;

        for pred in &predictions {
            let label_name = pred.label.trim_start_matches("__label__");
            match label_name {
                "jev" => p_jev = pred.prob,
                "llm" => p_llm = pred.prob,
                _ => {}
            }
        }

        // 3. Margin-based calibration with defensive clamping
        let (route, top_prob, calibrated_score) = calibrate(p_jev, p_llm);

        if route == Tier2Route::Jev && calibrated_score >= self.confidence_threshold {
            Tier2Outcome::Resolved {
                route,
                confidence: top_prob,
                calibrated_score,
                p_jev,
                p_llm,
                raw_probabilities,
            }
        } else {
            Tier2Outcome::PassThrough {
                top_guess: route,
                score: calibrated_score,
                reason: if route == Tier2Route::Jev {
                    "below_confidence_threshold"
                } else {
                    "not_jev"
                },
                p_jev,
                p_llm,
                raw_probabilities,
            }
        }
    }
}

//  ========== tests ==========

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // 1. Invariant: normalize_input never produces leading/trailing whitespace
        // or consecutive spaces, regardless of input string noise.
        #[test]
        fn prop_normalize_input_invariants(s in ".*") {
            let normalized = Tier2Runner::normalize_input(&s);

            prop_assert_eq!(&normalized, &normalized.to_lowercase());
            prop_assert_eq!(&normalized, normalized.trim());
            prop_assert!(!normalized.contains("  "));
        }

        // 2. Invariant: the real calibrate() output always lies in [0.0, 1.0]
        // even with out-of-bounds float inputs from bindings.
        #[test]
        fn prop_calibrated_score_bounds(
            p_jev in -100.0f32..100.0f32,
            p_llm in -100.0f32..100.0f32
        ) {
            let (_, top_prob, calibrated) = calibrate(p_jev, p_llm);

            prop_assert!((0.0..=1.0).contains(&top_prob));
            prop_assert!((0.0..=1.0).contains(&calibrated));
        }

        // 3. Invariant: swapping the two class probabilities leaves the score unchanged.
        #[test]
        fn prop_score_is_symmetric(a in 0.0f32..1.0f32, b in 0.0f32..1.0f32) {
            let (_, _, s1) = calibrate(a, b);
            let (_, _, s2) = calibrate(b, a);
            prop_assert_eq!(s1, s2);
        }

        // 4. Invariant: more certainty never lowers the score.
        #[test]
        fn prop_score_monotonic_in_certainty(a in 0.5f32..1.0f32, b in 0.5f32..1.0f32) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let (_, _, s_lo) = calibrate(lo, 1.0 - lo);
            let (_, _, s_hi) = calibrate(hi, 1.0 - hi);
            prop_assert!(s_lo <= s_hi);
        }
    }

    #[test]
    fn test_scores_discriminate_mid_range() {
        let (_, _, weak) = calibrate(0.60, 0.40);
        let (_, _, strong) = calibrate(0.90, 0.10);
        assert!(weak < strong);
        assert!(strong < 1.0);
    }

    #[test]
    fn test_whitespace_and_casing_normalization() {
        let raw = "  Hello   WORLD \t\n Test  ";
        assert_eq!(Tier2Runner::normalize_input(raw), "hello world test");
    }

    #[test]
    fn test_check_labels_accepts_both_labels() {
        let labels = vec!["__label__jev".to_string(), "__label__llm".to_string()];
        assert!(check_labels(&labels).is_ok());
    }

    #[test]
    fn test_check_labels_accepts_unprefixed_labels() {
        let labels = vec!["llm".to_string(), "jev".to_string()];
        assert!(check_labels(&labels).is_ok());
    }

    #[test]
    fn test_check_labels_rejects_single_label_model() {
        let labels = vec!["__label__jev".to_string()];
        let err = check_labels(&labels).unwrap_err();
        assert!(err.contains("llm"));
    }

    #[test]
    fn test_check_labels_rejects_wrong_label_names() {
        let labels = vec![
            "__label__jev_capable".to_string(),
            "__label__needs_llm".to_string(),
        ];
        assert!(check_labels(&labels).is_err());
    }

    #[test]
    fn test_check_labels_rejects_empty() {
        assert!(check_labels(&[]).is_err());
    }
}
