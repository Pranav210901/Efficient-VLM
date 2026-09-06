# Alignment v4 Wave 0 — recipe lock

Status: **COMPLETE**

## Locked recipe / stop record

```json
{
  "status": "SELECTED_WAVE0_RECIPE",
  "proceed": true,
  "recipe": {
    "adapter": false,
    "distillation": false,
    "lora": false,
    "loss_type": "infonce_no_queue",
    "captions_per_image": null,
    "memory_queue_size": 0,
    "batch_size": 1024,
    "sweep_multiplier": 3.0
  },
  "experiment_id": "infonce_no_queue__captions_all__b1024__lr_3p0",
  "mean_3seed": 0.37321177621682483,
  "mean_2seed": 0.37183182686567307,
  "mean_2seed_seed_order": [
    43,
    44
  ],
  "mean_2seed_semantics": "Confirmation mean over seeds 43 and 44 only. This is not a winner's-curse correction: the corrected queue-free configuration was not selected from the original 18-cell screening ranking.",
  "winner_curse_correction_applicable": false,
  "seed_values": [
    0.3759716749191284,
    0.37063227593898773,
    0.3730313777923584
  ],
  "seed_order": [
    42,
    43,
    44
  ],
  "sigmoid_mean_3seed": 0.3652395159006119,
  "corrected_minus_sigmoid_mean_3seed": 0.007972260316212954,
  "practical_equivalence_margin": 0.005,
  "winner_on_mean_3seed": true,
  "tie_break_triggered": false,
  "tie_break_rule": "fewer_changes_from_v4_recipe",
  "tie_break_outcome_if_needed": "infonce_no_queue",
  "selection_basis": "Higher three-seed mean; the difference exceeded the practical equivalence margin, so the tie-break was not needed.",
  "historical_wave0_screen": {
    "replication_anchor_R@1": 0.17016710340976715,
    "replication_anchor_target_R@1": 0.1702,
    "replication_anchor_margin": -3.289659023283931e-05,
    "replication_anchor_absolute_margin": 3.289659023283931e-05,
    "replication_support": "STRONG",
    "screening_spread": {
      "min": 0.13321583718061447,
      "max": 0.3660529553890228,
      "std": 0.09485667801988704
    },
    "batch_effect_2048_minus_512": 0.008441619575023651
  },
  "prior_selection_artifact": "selection/finalists.json",
  "prior_selection_superseded": true,
  "loss_family_finding": false,
  "dominant_finding": "memory_queue_staleness",
  "wave1_unblocked": true
}
```

## Prediction outcomes

| id                 | statement                                                                                                                                           | observed                                      | outcome   | if_wrong                                                                                                                                                                                                                          |
|:-------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------|:----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| replication_anchor | Lands within 2pp of 17.02% but below it (15.02-17.02%).                                                                                             | 0.17016710340976715                           | hit       |                                                                                                                                                                                                                                   |
| wave0_winner       | sigmoid / captions=all, at either batch=1024 or batch=2048. Counts as a hit if the winner is sigmoid / captions=all at either of those batch sizes. | infonce_no_queue__captions_all__b1024__lr_3p0 | miss      | If an infonce arm wins, the v3/v4 recipe was already near-optimal on the loss axis and the ceiling is less likely to be a recipe artifact. If sigmoid wins at batch=512, the batch effect is weaker than the v3/v4 gap suggested. |
| batch_effect       | Positive, between 0 and 2pp.                                                                                                                        | 0.008441619575023651                          | hit       |                                                                                                                                                                                                                                   |

## Limitations

- **Tie-break was not invoked:** queue-free InfoNCE exceeded sigmoid's three-seed mean by 0.797pp, beyond the pre-registered 0.5pp practical-equivalence margin.
- **Historical LR asymmetry:** sigmoid received the original per-batch LR sweep. Subsequent matched-LR controls tested queue-free InfoNCE at x3 and x6 before recipe lock.
- **Compounding sigmoid/captions-all limitation:** sigmoid multipliers are tuned at captions=2 and applied to captions=all, whose per-image normalization already increases loss magnitude with text-row count.

The controls reduce but do not erase the fact that the recipe was selected on the development split.

## Queue audit and revised finding

- The dominant observed effect is the memory queue (~19pp at captions=all, batch=1024), not the loss family (~1pp before matched-LR controls).
- Queue entries are detached, FIFO eviction at 16,384 entries is active, and same-image captions are masked as positives rather than false negatives.
- Embeddings are unit-normalized when produced, but are enqueued after the optimizer step from the pre-update forward pass. They are stale at insertion and are never recomputed in the updated projection space; no momentum key encoder is used.
- The batch effect is queue-conditional: +3.7/+4.6pp with the queue, -2.0/-2.2pp without it, and approximately flat for sigmoid. The v3/v4 gap is therefore better explained by queue staleness ratio than batch size alone: 16,384 historical image entries versus 512 current images is 32:1, versus 2,048 current images is 8:1.
- Cell 0 (`infonce_queue`, captions=2, batch=512; 13.32%) reproduces the v4 baseline, and cell 2 (17.02%) reproduces v3.
- Scope note for prior chapters: every reported v2, v3, and v4 result used the memory queue. Conclusions from those chapters therefore apply to the queued training recipe, not queue-free frozen alignment in general.

This audit found no detach, eviction, or positive-mask implementation defect. It did find a methodological design mismatch between a changing student projector and historical detached keys. The staleness explanation is strongly consistent with the screen, but remains an inference rather than an isolated causal result.

## Sigmoid LR provenance

The three captions=2 calibration cells ran at multiplier x1.0, but their screening metrics were later replaced by aliases to the selected x3.0 sweep jobs. Their original screening checkpoint directories still contain the obsolete x1.0 checkpoints. The three captions=all screening cells and every seed-43/44 finalist rerun genuinely ran at x3.0. Thus all six sigmoid values used for finalist selection were x3.0 results, but the captions=2 screening artifact directories have mixed provenance.

The LR multiplier is material, not noise: at captions=2, x1.0 produced 33.12%/33.35%/30.70% for batches 512/1024/2048, while x3.0 produced 35.65%/36.05%/36.20%. This is an approximately 3–5pp tuning effect and must qualify every loss-family comparison.

Matched-LR control C confirmed queue-free InfoNCE at x3 (37.58%) exceeds the locked sigmoid candidate (36.61%). Its three-seed mean is 37.32%, versus 36.52% for sigmoid. The difference exceeds the 0.5pp equivalence margin, so the tie-break was not required; it would also have favoured queue-free InfoNCE as the fewer-change recipe.

## Hardware and precision split

- Controls C/D and the accepted queue-capacity ablation tasks 0–6 from array 2223439 ran on the teaching partition's RTX PRO 6000 Blackwell GPUs with native BF16.
- The two winner-confirmation seeds and all six pair re-screen jobs run on the teaching partition's RTX PRO 6000 Blackwell GPUs with native BF16, keeping the corrected-recipe comparison hardware matched.
- No follow-up training is run in FP16.

## Corrected-recipe result

- Queue dose response (training-time best dev mean bidirectional R@1): 0 → 35.76%, 1,024 → 30.06%, 4,096 → 24.78%, 8,192 → 21.21%, 16,384 → 16.77%.
- Decomposition from the v4 baseline (13.32%) to the seed-42 corrected recipe (37.58%): queue removal ≈19.0pp, LR ×3 ≈1.8pp, all captions ≈0.9pp, and batch 512→1024 ≈0.3pp.
- The loss family is **not** the finding: sigmoid and queue-free InfoNCE differ by roughly 1pp, while the queue accounts for roughly 19pp.
- The seven recovered values have completed result-level evaluation. The anchor train/eval discrepancy was 0.00004pp.
- The queue stores post-projection embeddings. A frozen backbone does not prevent staleness because the projector—the component defining that embedding space—is the component being updated.

## Pre-registered interpretation

If the replication anchor is within 1pp of 17.02%, batch size is the remaining explanation for the v3/v4 baseline gap without qualification. At 1–2pp, support is qualified and the residual is unexplained. Outside 2pp, the wave stops.

A higher locked baseline mechanically lowers capture fraction for fixed teacher performance and roughly fixed distillation gain because the capture denominator grows; this is algebra, not a new empirical result.

The v3 batch 2048 value came from throughput probing; Wave 0 pins 2048 as a factorial level, so matching values are not an independent rediscovery.
