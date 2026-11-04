// examples/eval_tier1.rs
//
// Runs Tier1Engine::classify over a labeled JSONL file and reports routing quality.
//
// Usage (from the project root):
//     cargo run --example eval_tier1
//     cargo run --example eval_tier1 -- --config config/tier1.toml --cases evals/tier1/cases.jsonl
//
// Case format, one JSON object per line (same labels as the training exemplars):
//     {"id": "optional", "prompt": "What is 2+2?", "label": "jev_capable"}
// "text" is accepted in place of "prompt".

#[allow(dead_code)]
#[path = "../src/router/tier1_runner.rs"]
mod tier1_runner;

use std::collections::BTreeMap;
use std::fs;

use serde::Deserialize;
use tier1_runner::{Tier1Automaton, Tier1Engine, Tier1Result, AMBIGUITY_UPPER_BOUND};

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    text: Option<String>,
    label: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    Jev,
    Llm,
}

/// What Tier 1 did with the prompt. Mirrors the cascade: only a decided result
/// above AMBIGUITY_UPPER_BOUND resolves as jev; a decided result below the
/// lower bound is an LLM verdict that still escalates; anything else defers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pred {
    Jev,
    Llm,
    Defer,
}

struct Row {
    id: String,
    prompt: String,
    gold: Route,
    pred: Pred,
    result: Tier1Result,
}

fn gold_of(label: &str) -> Option<Route> {
    match label {
        "jev_capable" => Some(Route::Jev),
        "needs_llm" => Some(Route::Llm),
        _ => None,
    }
}

fn predict(r: &Tier1Result) -> Pred {
    if !r.decided {
        Pred::Defer
    } else if r.confidence > AMBIGUITY_UPPER_BOUND {
        Pred::Jev
    } else {
        Pred::Llm
    }
}

/// Groups results by which rule fired. The reason strings carry hit counts,
/// so strip leading digits and anything after the first '(' or ':'.
fn rule_of(reason: &str) -> String {
    let stripped = reason.trim_start_matches(|c: char| c.is_ascii_digit() || c == ' ');
    stripped
        .split(['(', ':'])
        .next()
        .unwrap_or(stripped)
        .trim()
        .to_string()
}

fn ratio(num: usize, den: usize) -> String {
    if den == 0 {
        "n/a".to_string()
    } else {
        format!("{:.3}  ({num}/{den})", num as f64 / den as f64)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str, default: &str| -> String {
        args.windows(2)
            .find(|w| w[0] == name)
            .map(|w| w[1].clone())
            .unwrap_or_else(|| default.to_string())
    };
    let config_path = flag("--config", "config/tier1.toml");
    let cases_path = flag("--cases", "evals/tier1/cases.jsonl");

    let engine = Tier1Engine::new(Tier1Automaton::from_toml_file(&config_path)?)?;

    let mut rows: Vec<Row> = Vec::new();
    for (i, line) in fs::read_to_string(&cases_path)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let at = format!("{cases_path}:{}", i + 1);
        let case: Case = serde_json::from_str(line).map_err(|e| format!("{at}: {e}"))?;
        let gold =
            gold_of(&case.label).ok_or_else(|| format!("{at}: unknown label {:?}", case.label))?;
        let prompt = case
            .prompt
            .or(case.text)
            .filter(|p| !p.is_empty())
            .ok_or_else(|| format!("{at}: missing prompt/text"))?;
        let id = case.id.unwrap_or_else(|| format!("row{}", i + 1));

        let result = engine.classify(&prompt);
        rows.push(Row {
            id,
            prompt,
            gold,
            pred: predict(&result),
            result,
        });
    }

    if rows.is_empty() {
        return Err(format!("{cases_path}: no cases").into());
    }

    let count = |g: Route, p: Pred| rows.iter().filter(|r| r.gold == g && r.pred == p).count();
    let gold_total = |g: Route| rows.iter().filter(|r| r.gold == g).count();

    println!("config: {config_path}");
    println!("cases:  {cases_path}  ({} rows)\n", rows.len());

    println!(
        "{:<10} {:>9} {:>9} {:>7} {:>7}",
        "", "pred JEV", "pred LLM", "DEFER", "total"
    );
    for (name, g) in [("gold JEV", Route::Jev), ("gold LLM", Route::Llm)] {
        println!(
            "{:<10} {:>9} {:>9} {:>7} {:>7}",
            name,
            count(g, Pred::Jev),
            count(g, Pred::Llm),
            count(g, Pred::Defer),
            gold_total(g)
        );
    }

    let tp_jev = count(Route::Jev, Pred::Jev);
    let fp_jev = count(Route::Llm, Pred::Jev);
    let tp_llm = count(Route::Llm, Pred::Llm);
    let fp_llm = count(Route::Jev, Pred::Llm);
    let decided = rows.iter().filter(|r| r.pred != Pred::Defer).count();

    println!();
    println!(
        "committed-jev precision : {}",
        ratio(tp_jev, tp_jev + fp_jev)
    );
    println!(
        "jev coverage            : {}",
        ratio(tp_jev, gold_total(Route::Jev))
    );
    println!(
        "llm-verdict precision   : {}",
        ratio(tp_llm, tp_llm + fp_llm)
    );
    println!("tier-1 decided rate     : {}", ratio(decided, rows.len()));

    // Per-rule breakdown: which rules commit, and how often they are right.
    let mut by_rule: BTreeMap<String, (usize, usize, usize, usize)> = BTreeMap::new();
    for r in &rows {
        let e = by_rule.entry(rule_of(&r.result.reason)).or_default();
        e.0 += 1;
        match (r.pred, r.gold) {
            (Pred::Defer, _) => e.3 += 1,
            (Pred::Jev, Route::Jev) | (Pred::Llm, Route::Llm) => e.1 += 1,
            _ => e.2 += 1,
        }
    }

    println!(
        "\n{:<48} {:>4} {:>6} {:>6} {:>6}",
        "rule", "n", "right", "wrong", "defer"
    );
    for (rule, (n, ok, wrong, defer)) in &by_rule {
        println!("{rule:<48} {n:>4} {ok:>6} {wrong:>6} {defer:>6}");
    }

    // Cases worth reading: expensive errors and jev prompts Tier 1 failed to commit.
    println!();
    let mut flagged = 0;
    for r in &rows {
        let kind = match (r.gold, r.pred) {
            (Route::Llm, Pred::Jev) => "FALSE JEV",
            (Route::Jev, Pred::Llm) => "FALSE LLM",
            (Route::Jev, Pred::Defer) => "JEV NOT COMMITTED",
            _ => continue,
        };
        flagged += 1;
        println!(
            "[{kind}] {}  conf={:.3}  rule={}  prompt={:?}",
            r.id,
            r.result.confidence,
            rule_of(&r.result.reason),
            r.prompt
        );
    }
    if flagged == 0 {
        println!("no flagged cases");
    }

    Ok(())
}
