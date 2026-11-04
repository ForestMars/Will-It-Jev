// Evaluates Tier 2's Jev commit threshold on the held-out FastText split.
//
// Usage:
//     cargo run --example eval_tier2
//     cargo run --example eval_tier2 -- --model models/tier2.bin --cases calibration/ft_valid.txt

#[allow(dead_code)]
#[path = "../src/router/tier2_runner.rs"]
mod tier2_runner;

use std::collections::HashMap;
use std::fs;
use tier2_runner::{Tier2Outcome, Tier2Route, Tier2Runner, DEFAULT_CONFIDENCE_THRESHOLD};

struct Case {
    line: usize,
    label: Tier2Route,
    prompt: String,
}

struct ScoredCase {
    case: Case,
    guess: Tier2Route,
    score: f32,
}

fn argument(args: &[String], name: &str, default: &str) -> String {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn load_cases(path: &str) -> Result<Vec<Case>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let mut cases = Vec::new();

    for (index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let line_number = index + 1;
        let (label, prompt) = line
            .split_once(' ')
            .ok_or_else(|| format!("{path}:{line_number}: expected label and prompt"))?;
        let label = match label {
            "__label__jev" => Tier2Route::Jev,
            "__label__llm" => Tier2Route::Llm,
            _ => return Err(format!("{path}:{line_number}: unknown label {label:?}").into()),
        };
        let prompt = prompt.trim();
        if prompt.is_empty() {
            return Err(format!("{path}:{line_number}: prompt is empty").into());
        }

        cases.push(Case {
            line: line_number,
            label,
            prompt: prompt.to_string(),
        });
    }

    if cases.is_empty() {
        return Err(format!("{path}: no cases").into());
    }

    Ok(cases)
}

fn score_case(runner: &Tier2Runner, case: Case) -> ScoredCase {
    let (guess, score) = match runner.evaluate(&case.prompt) {
        Tier2Outcome::Resolved {
            route,
            calibrated_score,
            ..
        } => (route, calibrated_score),
        Tier2Outcome::PassThrough {
            top_guess, score, ..
        } => (top_guess, score),
    };

    ScoredCase { case, guess, score }
}

fn commits_jev(case: &ScoredCase, threshold: f32) -> bool {
    case.guess == Tier2Route::Jev && case.score >= threshold
}

fn validate_independent_cases(evaluation: &[Case], training: &[Case]) -> Result<(), String> {
    let training_lines: HashMap<_, _> = training
        .iter()
        .map(|case| (Tier2Runner::normalize_input(&case.prompt), case.line))
        .collect();
    let overlaps: Vec<_> = evaluation
        .iter()
        .filter_map(|case| {
            training_lines
                .get(&Tier2Runner::normalize_input(&case.prompt))
                .map(|training_line| (case.line, *training_line))
        })
        .collect();

    if overlaps.is_empty() {
        Ok(())
    } else {
        let examples = overlaps
            .iter()
            .take(5)
            .map(|(evaluation_line, training_line)| {
                format!("evaluation line {evaluation_line} matches training line {training_line}")
            })
            .collect::<Vec<_>>()
            .join("; ");
        Err(format!(
            "evaluation data overlaps training data in {} prompt(s): {examples}",
            overlaps.len()
        ))
    }
}

fn wilson_interval(successes: usize, trials: usize) -> Option<(f64, f64)> {
    if trials == 0 {
        return None;
    }

    let z = 1.96;
    let n = trials as f64;
    let proportion = successes as f64 / n;
    let denominator = 1.0 + z * z / n;
    let center = (proportion + z * z / (2.0 * n)) / denominator;
    let half_width =
        z * (proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)).sqrt() / denominator;
    Some((center - half_width, center + half_width))
}

fn threshold_candidates(scored: &[ScoredCase]) -> Vec<f32> {
    let mut thresholds = vec![0.0, 1.0];
    thresholds.extend(
        scored
            .iter()
            .filter(|result| result.guess == Tier2Route::Jev)
            .map(|result| result.score),
    );
    thresholds.sort_by(f32::total_cmp);
    thresholds.dedup_by(|left, right| *left == *right);
    thresholds
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = argument(&args, "--model", "models/tier2.bin");
    let cases_path = argument(&args, "--cases", "calibration/ft_valid.txt");
    let training_path = argument(&args, "--training-cases", "calibration/ft_train.txt");
    let cases = load_cases(&cases_path)?;
    let training_cases = load_cases(&training_path)?;
    validate_independent_cases(&cases, &training_cases)?;
    let runner = Tier2Runner::new(&model_path, DEFAULT_CONFIDENCE_THRESHOLD)?;
    let scored: Vec<_> = cases
        .into_iter()
        .map(|case| score_case(&runner, case))
        .collect();

    let jev_total = scored
        .iter()
        .filter(|result| result.case.label == Tier2Route::Jev)
        .count();
    let llm_total = scored.len() - jev_total;
    if jev_total == 0 || llm_total == 0 {
        return Err(format!(
            "{cases_path}: held-out data must contain both labels (Jev={jev_total}, LLM={llm_total})"
        )
        .into());
    }

    println!("model: {model_path}");
    println!(
        "evaluation cases: {cases_path} ({} rows; Jev={jev_total}, LLM={llm_total})",
        scored.len(),
    );
    println!("checked for prompt overlap with training cases: {training_path}");
    println!(
        "\n{:<10} {:>10} {:>29} {:>12} {:>12}",
        "threshold", "Jev commits", "Jev precision (95% Wilson CI)", "Jev coverage", "false Jev"
    );

    for threshold in threshold_candidates(&scored) {
        let committed = scored
            .iter()
            .filter(|result| commits_jev(result, threshold))
            .count();
        let correct = scored
            .iter()
            .filter(|result| commits_jev(result, threshold) && result.case.label == Tier2Route::Jev)
            .count();
        let false_jev = committed - correct;
        let precision = if committed == 0 {
            0.0
        } else {
            correct as f32 / committed as f32
        };
        let precision_interval = wilson_interval(correct, committed)
            .map(|(lower, upper)| format!("{precision:.3} [{lower:.3}, {upper:.3}]"))
            .unwrap_or_else(|| "n/a".to_string());
        let coverage = correct as f32 / jev_total as f32;

        println!(
            "{threshold:<12.7} {committed:>5}/{:<5} {precision_interval:>29} {coverage:>11.3} {false_jev:>12}",
            scored.len(),
        );
    }

    let committed: Vec<_> = scored
        .iter()
        .filter(|result| commits_jev(result, DEFAULT_CONFIDENCE_THRESHOLD))
        .collect();
    let false_jev = committed
        .iter()
        .filter(|result| result.case.label != Tier2Route::Jev)
        .count();
    let true_jev = committed.len() - false_jev;
    let precision_interval = wilson_interval(true_jev, committed.len())
        .map(|(lower, upper)| format!("[{lower:.3}, {upper:.3}]"))
        .unwrap_or_else(|| "n/a (no Jev commits)".to_string());
    println!(
        "\nAt the current {DEFAULT_CONFIDENCE_THRESHOLD:.2} threshold: Jev precision={:.3} {precision_interval} ({true_jev}/{}), Jev coverage={:.3} ({true_jev}/{jev_total}), false Jev commits={false_jev}",
        if committed.is_empty() {
            0.0
        } else {
            true_jev as f32 / committed.len() as f32
        },
        committed.len(),
        true_jev as f32 / jev_total as f32,
    );

    for result in scored.iter().filter(|result| {
        commits_jev(result, DEFAULT_CONFIDENCE_THRESHOLD) && result.case.label != Tier2Route::Jev
    }) {
        println!(
            "[FALSE JEV] line={} score={:.4} prompt={:?}",
            result.case.line, result.score, result.case.prompt
        );
    }
    for result in scored.iter().filter(|result| {
        result.case.label == Tier2Route::Jev && !commits_jev(result, DEFAULT_CONFIDENCE_THRESHOLD)
    }) {
        println!(
            "[DEFERRED JEV] line={} score={:.4} prompt={:?}",
            result.case.line, result.score, result.case.prompt
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored_case(label: Tier2Route, guess: Tier2Route, score: f32) -> ScoredCase {
        ScoredCase {
            case: Case {
                line: 1,
                label,
                prompt: "test prompt".to_string(),
            },
            guess,
            score,
        }
    }

    #[test]
    fn jev_commit_includes_threshold_boundary() {
        let result = scored_case(
            Tier2Route::Jev,
            Tier2Route::Jev,
            DEFAULT_CONFIDENCE_THRESHOLD,
        );
        assert!(commits_jev(&result, DEFAULT_CONFIDENCE_THRESHOLD));
    }

    #[test]
    fn llm_guess_never_commits_to_jev() {
        let result = scored_case(Tier2Route::Jev, Tier2Route::Llm, 1.0);
        assert!(!commits_jev(&result, DEFAULT_CONFIDENCE_THRESHOLD));
    }

    #[test]
    fn independent_case_check_rejects_normalized_prompt_overlap() {
        let evaluation = [Case {
            line: 3,
            label: Tier2Route::Jev,
            prompt: "  WHAT is 2+2? ".to_string(),
        }];
        let training = [Case {
            line: 8,
            label: Tier2Route::Jev,
            prompt: "what is 2+2?".to_string(),
        }];

        let error = validate_independent_cases(&evaluation, &training).unwrap_err();
        assert!(error.contains("evaluation line 3 matches training line 8"));
    }

    #[test]
    fn independent_case_check_accepts_disjoint_prompts() {
        let evaluation = [Case {
            line: 3,
            label: Tier2Route::Jev,
            prompt: "what is 2+2?".to_string(),
        }];
        let training = [Case {
            line: 8,
            label: Tier2Route::Jev,
            prompt: "what is 3+3?".to_string(),
        }];

        assert!(validate_independent_cases(&evaluation, &training).is_ok());
    }

    #[test]
    fn wilson_interval_shows_uncertainty_for_perfect_small_sample() {
        let (lower, upper) = wilson_interval(12, 12).unwrap();
        assert!(lower < 1.0);
        assert_eq!(upper, 1.0);
    }

    #[test]
    fn wilson_interval_is_unavailable_without_commits() {
        assert_eq!(wilson_interval(0, 0), None);
    }

    #[test]
    fn threshold_candidates_cover_distinct_jev_scores() {
        let scored = [
            scored_case(Tier2Route::Jev, Tier2Route::Jev, 0.73),
            scored_case(Tier2Route::Jev, Tier2Route::Jev, 0.41),
            scored_case(Tier2Route::Jev, Tier2Route::Jev, 0.73),
            scored_case(Tier2Route::Llm, Tier2Route::Llm, 0.99),
        ];

        assert_eq!(threshold_candidates(&scored), vec![0.0, 0.41, 0.73, 1.0]);
    }
}
