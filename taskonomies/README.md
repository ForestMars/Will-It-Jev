# Taskonomy authoring contract

A taskonomy category has three distinct kinds of content:

- `description` defines the category's scope for maintainers and other consumers.
- `classifier_criterion` is the concise, contrastive rule used as that category's value in a closed-set classifier choice.
- `examples` are representative utterances. They illustrate the category but do not enumerate every valid phrasing.

Keep those roles separate. Do not shorten the canonical description merely to make a model prompt smaller, and do not treat examples as an exhaustive keyword list. Write classifier criteria to distinguish neighboring categories, including explicit exclusions or cross-references where overlap is likely.

`classification_guidance` describes how to select the task being pursued. `classifier_criterion_guidance` describes how model-facing prompts should use the category criteria. In particular, taskonomy classification is independent of the binary Jev-eligibility decision: a category must never imply `jev_eligible` or `needs_llm` by itself.

## Versioning

- Increment `version` when category definitions, classifier criteria, or taskonomy semantics change.
- Increment `schema_version` when the JSON authoring contract changes, such as adding a required field.
- Preserve category `id` and `name` values when refining wording. They are identifiers consumed by evaluations and telemetry, not prose labels.

The E2E router test reads the category keys and `classifier_criterion` values directly from `help_desk.json`. Its fixture expectations (for example, `account_pw` and `nw_vpn`) remain explicit gold labels because changing a taxonomy definition should not silently change what the test considers correct.
