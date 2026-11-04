// src/router/tier2_runner.rs
//
// FastText scorer for the existing binary route task. It produces raw class
// evidence only; the pipeline's task/tier policy decides whether that evidence
// is sufficient to resolve the task.

use fasttext::FastText;
use std::path::Path;

const REQUIRED_LABELS: [&str; 2] = ["jev", "llm"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier2Route {
    Jev,
    Llm,
}

#[derive(Debug, Clone)]
pub struct Tier2Score {
    pub route: Tier2Route,
    /// Absolute difference between the two class probabilities.
    pub calibrated_score: f32,
    pub p_jev: f32,
    pub p_llm: f32,
}

#[derive(Debug, Clone)]
pub enum Tier2Outcome {
    Scored(Tier2Score),
    Abstain { reason: &'static str },
}

pub struct Tier2Runner {
    model: FastText,
}

/// Pure scoring step. Returns (top class, top probability, class-probability
/// margin). The margin measures separation; it is not an acceptance decision.
pub fn calibrate(p_jev: f32, p_llm: f32) -> (Tier2Route, f32, f32) {
    let p_jev = finite_probability(p_jev);
    let p_llm = finite_probability(p_llm);

    let margin = (p_jev - p_llm).abs();
    let (route, top_probability) = if p_jev >= p_llm {
        (Tier2Route::Jev, p_jev)
    } else {
        (Tier2Route::Llm, p_llm)
    };

    (route, top_probability, margin.clamp(0.0, 1.0))
}

fn finite_probability(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Pure model-schema check. Accepts labels with or without the `__label__`
/// prefix and reports the labels the model actually contains.
pub fn check_labels(labels: &[String]) -> Result<(), String> {
    let names: Vec<&str> = labels
        .iter()
        .map(|label| label.trim_start_matches("__label__"))
        .collect();
    let missing: Vec<&str> = REQUIRED_LABELS
        .iter()
        .copied()
        .filter(|required| !names.contains(required))
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Tier 2 model is missing required label(s) {missing:?}; model contains {names:?}"
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

    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, String> {
        let model = FastText::load_model(&model_path)
            .map_err(|error| format!("Failed to load FastText model: {error}"))?;
        let (labels, _frequencies) = model.get_labels();
        check_labels(&labels)?;

        Ok(Self { model })
    }

    /// Return the model's raw prediction. No confidence threshold or routing
    /// decision is applied here.
    #[inline]
    pub fn evaluate(&self, prompt: &str) -> Tier2Outcome {
        let normalized = Self::normalize_input(prompt);
        if normalized.is_empty() {
            return Tier2Outcome::Abstain {
                reason: "empty_input",
            };
        }

        let predictions = self.model.predict(&normalized, 2, 0.0);
        if predictions.is_empty() {
            return Tier2Outcome::Abstain {
                reason: "no_predictions",
            };
        }

        tracing::info!(
            "tier2 raw predictions: {:?}",
            predictions
                .iter()
                .map(|prediction| (prediction.label.as_str(), prediction.prob))
                .collect::<Vec<_>>()
        );

        let mut p_jev = 0.0f32;
        let mut p_llm = 0.0f32;
        for prediction in &predictions {
            match prediction.label.trim_start_matches("__label__") {
                "jev" => p_jev = prediction.prob,
                "llm" => p_llm = prediction.prob,
                _ => {}
            }
        }

        let (route, _top_probability, calibrated_score) = calibrate(p_jev, p_llm);
        Tier2Outcome::Scored(Tier2Score {
            route,
            calibrated_score,
            p_jev: finite_probability(p_jev),
            p_llm: finite_probability(p_llm),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_normalize_input_invariants(s in ".*") {
            let normalized = Tier2Runner::normalize_input(&s);
            prop_assert_eq!(&normalized, &normalized.to_lowercase());
            prop_assert_eq!(&normalized, normalized.trim());
            prop_assert!(!normalized.contains("  "));
        }

        #[test]
        fn prop_calibrated_score_bounds(
            p_jev in -100.0f32..100.0f32,
            p_llm in -100.0f32..100.0f32
        ) {
            let (_, top_probability, calibrated) = calibrate(p_jev, p_llm);
            prop_assert!((0.0..=1.0).contains(&top_probability));
            prop_assert!((0.0..=1.0).contains(&calibrated));
        }

        #[test]
        fn prop_score_is_symmetric(a in 0.0f32..1.0f32, b in 0.0f32..1.0f32) {
            let (_, _, score_ab) = calibrate(a, b);
            let (_, _, score_ba) = calibrate(b, a);
            prop_assert_eq!(score_ab, score_ba);
        }

        #[test]
        fn prop_score_monotonic_in_certainty(a in 0.5f32..1.0f32, b in 0.5f32..1.0f32) {
            let (lower, upper) = if a <= b { (a, b) } else { (b, a) };
            let (_, _, score_lower) = calibrate(lower, 1.0 - lower);
            let (_, _, score_upper) = calibrate(upper, 1.0 - upper);
            prop_assert!(score_lower <= score_upper);
        }
    }

    #[test]
    fn calibration_discriminates_mid_range() {
        let (_, _, weak) = calibrate(0.60, 0.40);
        let (_, _, strong) = calibrate(0.90, 0.10);
        assert!(weak < strong);
        assert!(strong < 1.0);
    }

    #[test]
    fn normalize_input_handles_whitespace_and_casing() {
        let raw = "  Hello   WORLD \t\n Test  ";
        assert_eq!(Tier2Runner::normalize_input(raw), "hello world test");
    }

    #[test]
    fn check_labels_accepts_prefixed_and_unprefixed_labels() {
        assert!(check_labels(&[
            "__label__jev".to_string(),
            "__label__llm".to_string(),
        ]).is_ok());
        assert!(check_labels(&["llm".to_string(), "jev".to_string()]).is_ok());
    }

    #[test]
    fn check_labels_rejects_wrong_or_missing_labels() {
        assert!(check_labels(&["__label__jev".to_string()]).is_err());
        assert!(check_labels(&[
            "__label__jev_capable".to_string(),
            "__label__needs_llm".to_string(),
        ]).is_err());
        assert!(check_labels(&[]).is_err());
    }

    #[test]
    fn calibration_treats_non_finite_values_as_zero() {
        let (_, top_probability, score) = calibrate(f32::NAN, f32::INFINITY);
        assert_eq!(top_probability, 0.0);
        assert_eq!(score, 0.0);
    }
}
