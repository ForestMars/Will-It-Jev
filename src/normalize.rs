use std::ops::Range;

use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

/// Single normalized view of a request that every tier reads.
///
/// `text` is NFC, with every whitespace run folded to one ASCII space and
/// leading/trailing whitespace stripped. For each byte of `text`, the map
/// records the byte span of the raw input it came from, so any span found in
/// normalized space (a regex hit, a token offset, an n-gram) maps back to the
/// raw string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedText {
    text: String,
    orig_start: Vec<usize>,
    orig_end: Vec<usize>,
}

impl NormalizedText {
    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Maps a byte range of the normalized text to the byte range of the raw
    /// input it covers. A range that ends inside a composed character or a
    /// folded whitespace run widens to cover the whole source. Returns `None`
    /// for an empty or out-of-bounds range.
    pub fn original_span(&self, range: Range<usize>) -> Option<Range<usize>> {
        if range.start >= range.end || range.end > self.text.len() {
            return None;
        }
        Some(self.orig_start[range.start]..self.orig_end[range.end - 1])
    }
}

fn push(out: &mut NormalizedText, s: &str, start: usize, end: usize) {
    out.text.push_str(s);
    for _ in 0..s.len() {
        out.orig_start.push(start);
        out.orig_end.push(end);
    }
}

pub fn normalize(raw: &str) -> NormalizedText {
    let mut out = NormalizedText {
        text: String::with_capacity(raw.len()),
        orig_start: Vec::with_capacity(raw.len()),
        orig_end: Vec::with_capacity(raw.len()),
    };
    let mut pending_ws: Option<(usize, usize)> = None;

    // Composition never crosses a grapheme cluster boundary, so NFC runs per
    // cluster and each output byte maps to exactly one source cluster.
    for (idx, cluster) in raw.grapheme_indices(true) {
        let end = idx + cluster.len();
        if cluster.chars().all(char::is_whitespace) {
            pending_ws = Some(match pending_ws {
                Some((s, _)) => (s, end),
                None => (idx, end),
            });
            continue;
        }
        if let Some((s, e)) = pending_ws.take() {
            if !out.text.is_empty() {
                push(&mut out, " ", s, e);
            }
        }
        let composed: String = cluster.nfc().collect();
        push(&mut out, &composed, idx, end);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLES: [&str; 6] = [
        "plain ascii",
        "caf\u{e9}",
        "cafe\u{301}",
        "\u{1100}\u{1161}\u{11a8}",
        "e\u{301}\u{323}",
        "\u{212b}ngstr\u{f6}m",
    ];

    #[test]
    fn composed_and_decomposed_inputs_normalize_identically() {
        assert_eq!(
            normalize("caf\u{e9}").as_str(),
            normalize("cafe\u{301}").as_str()
        );
        assert_eq!(normalize("cafe\u{301}").as_str(), "caf\u{e9}");
    }

    #[test]
    fn matches_whole_string_nfc_when_no_whitespace_edits_apply() {
        for s in SAMPLES {
            let expected: String = s.nfc().collect();
            assert_eq!(normalize(s).as_str(), expected, "sample {s:?}");
        }
    }

    #[test]
    fn is_idempotent() {
        for s in SAMPLES {
            let once = normalize(s);
            let twice = normalize(once.as_str());
            assert_eq!(once.as_str(), twice.as_str(), "sample {s:?}");
        }
    }

    #[test]
    fn folds_whitespace_runs_and_strips_edges() {
        let n = normalize("  reset \t my\n\n password  ");
        assert_eq!(n.as_str(), "reset my password");
    }

    #[test]
    fn whitespace_only_input_is_empty() {
        let n = normalize("  \t\n ");
        assert!(n.is_empty());
        assert_eq!(n.original_span(0..1), None);
        assert!(normalize("").is_empty());
    }

    #[test]
    fn span_maps_back_to_raw_bytes() {
        let raw = "  my  pw expired ";
        let n = normalize(raw);
        assert_eq!(n.as_str(), "my pw expired");
        let span = n.original_span(3..5).unwrap();
        assert_eq!(&raw[span], "pw");
    }

    #[test]
    fn span_covers_whole_cluster_for_decomposed_input() {
        let raw = "cafe\u{301}";
        let n = normalize(raw);
        // composed e-acute occupies normalized bytes 3..5
        let full = n.original_span(3..5).unwrap();
        assert_eq!(&raw[full], "e\u{301}");
        let partial = n.original_span(3..4).unwrap();
        assert_eq!(&raw[partial], "e\u{301}");
    }

    #[test]
    fn folded_whitespace_run_maps_to_the_whole_run() {
        let raw = "a \n b";
        let n = normalize(raw);
        assert_eq!(n.as_str(), "a b");
        let span = n.original_span(1..2).unwrap();
        assert_eq!(&raw[span], " \n ");
    }

    #[test]
    fn offsets_are_monotonic_and_in_bounds() {
        for s in SAMPLES {
            let raw = format!("  {s}  {s} ");
            let n = normalize(&raw);
            assert_eq!(n.orig_start.len(), n.len());
            assert_eq!(n.orig_end.len(), n.len());
            for w in n.orig_start.windows(2) {
                assert!(w[0] <= w[1]);
            }
            assert!(n.orig_end.iter().all(|&e| e <= raw.len()));
        }
    }

    #[test]
    fn rejects_empty_and_out_of_bounds_ranges() {
        let n = normalize("abc");
        assert_eq!(n.original_span(1..1), None);
        assert_eq!(n.original_span(0..4), None);
        assert_eq!(n.original_span(0..3), Some(0..3));
    }
}
