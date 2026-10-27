use super::tier3_runner::Tier3Outcome;

pub struct ModernBertRunner;

impl ModernBertRunner {
    pub fn new() -> Self {
        Self
    }

    pub fn evaluate(&self, _prompt: &str) -> Tier3Outcome {
        Tier3Outcome::PassThrough {
            top_guess: None,
            score: 0.0,
            reason: "not_implemented",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_always_passes_through_to_harrier() {
        assert!(matches!(
            ModernBertRunner::new().evaluate("prompt"),
            Tier3Outcome::PassThrough {
                score: 0.0,
                reason: "not_implemented",
                ..
            }
        ));
    }
}