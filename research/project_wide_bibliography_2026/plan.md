# Plan — Project-wide relevance-ranked bibliography

- **Slug:** `project_wide_bibliography_2026`
- **Run date:** 2026-08-06
- **Genre:** landscape (ranked bibliography with novelty assessment)
- **Depth:** deep
- **Request:** "find all the papers relevant to my entire project; rank them by relevancy"

## Phase 0 — Existing-work check (mandatory, completed)

Two prior deep-research folders already exist and were read in full before any
search was issued:

| Folder | Date | Sources | Scope |
|---|---|---|---|
| `docs/research/alignment_queue_prior_art` | 2026-07-26 | 33 | Frozen alignment, contrastive losses, memory queues, distillation, multi-encoder routing, reproducibility |
| `research/efficient_frozen_vlm_2026` | 2026-07-28 | 5 grouped | Novelty boundary for token routing / conditional computation |

**Consequence:** this run does *not* re-search those areas. It inherits their
38 sources, re-ranks them against the project as it stands today, and spends
its search budget exclusively on subject areas the code entered *after*
2026-07-28.

## Reframe

The literal request ("all papers relevant to the project") is under-specified —
"relevant" is only meaningful against a decision. The decision this feeds is:

> **Which literature must the dissertation cite, position against, and
> differentiate from — and does anything in it threaten the novelty claim of
> the work now in flight?**

Ranking is therefore by **threat-and-obligation to the contribution**, not by
topical similarity or citation count. A highly-cited paper that is merely
background ranks below an obscure 2025 preprint that occupies the same object.

## Scope gap identified

The prior folders were scoped to alignment/queue/routing. Since 2026-07-28 the
codebase has moved into areas neither folder covers:

| Project area | Evidence in repo | Prior coverage |
|---|---|---|
| Image token aggregation | `LearnedQueryPatchPool`, `SpatialTokenAdapter`, `token_aggregator_scale_*` | **none** |
| Text token aggregation | `LearnedQueryTextPool`, `text_aggregation_study` | **none** |
| Gated residual adaptation | `gate = Parameter(tensor(-2.0))` throughout `model.py` | **none** |
| Text encoder choice (E5 arm) | `text_aggregation_study` E-cells | **none** |
| Data scaling | `cc3m_acquisition`, `mixed_data_training`, `data_scale_pilot` | **none** |
| Error decomposition | `probe1_error_decomposition` | **none** |
| Efficiency frontier | `efficiency_frontier`, `flop_correction` | partial |

Search budget was spent entirely on these seven.

## Hypotheses (falsifiable, committed before synthesis)

- **H1** — The learned-query pooling used in the current aggregation study is
  an established primitive, not a novel mechanism.
  *Falsified if* no prior work applies learnable-query cross-attention pooling
  to frozen encoder features in a contrastive retrieval setting.
- **H2** — No existing paper occupies the exact combination of frozen
  heterogeneous encoders + pre-registered budget-gated aggregation selection.
  *Falsified if* such a paper is found.
- **H3** — The gate-at-−2.0 residual construction is standard practice with
  established precedent, not a contribution.
  *Falsified if* zero/negative-initialised residual gating has no precedent.
- **H4** — The project's mechanism has drifted from the direction the
  2026-07-28 landscape recommended (token *routing*) toward a more crowded
  object (token *aggregation*).
  *Falsified if* the current code implements conditional//dynamic computation.

## Sourcing strategy

Channels: arXiv (primary), CVPR/NeurIPS/ICLR proceedings, Semantic Scholar,
HuggingFace/OpenCLIP implementation sources for architectural ground truth.

Opposition queries (deliberately searching for disconfirmation):
- "attentive probing efficiency frozen encoder parameters"
- "attention pooling contrastive image text MAP head"
- "learned query pooling sentence embedding beats mean pooling"

## Risk register

| Risk | Mitigation |
|---|---|
| Confirming novelty by not looking hard enough | Opposition queries run explicitly; adversarial pass mandatory |
| Citing arXiv IDs from memory | Every ID recorded here appeared in a live search result or fetch; `verification_level` column records which |
| Over-trusting the prior 38 sources | Re-ranked, not re-asserted; their scores carried but relevance recomputed |
| Sub-agent fan-out unavailable | Session config prohibits agent spawning; searches were run inline in parallel batches instead. No loss of coverage, some loss of wall-clock speed. |

## Stop criteria

Stop when each of the seven gap areas has ≥1 verified primary source and the
novelty-critical area (token aggregation) has ≥3 independent, differently-typed
sources. Met.

## Changelog

- 2026-08-06 — initial run.
