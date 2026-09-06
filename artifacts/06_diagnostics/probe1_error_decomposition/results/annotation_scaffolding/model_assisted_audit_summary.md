# Probe 1 model-assisted audit

This is a rubric-constrained model-assisted audit, not a human audit and not ground truth.
The ambiguity sample remained blinded: `audit_key.csv` was neither read nor joined during labelling.

## Ambiguity labels

- `multiple_plausible`: 5
- `no_match`: 5
- `underspecified`: 10
- `unique`: 80

## Failure labels

- `caption_underspecified`: 3
- `fine_grained_miss`: 27
- `label_defensible`: 8
- `positive_atypical`: 2
- `retrieved_also_correct`: 10

## Interpretation boundary

These labels are structured judgements under the frozen rubric. They support descriptive sensitivity analysis, but must not be presented as independently verified human ground truth.
