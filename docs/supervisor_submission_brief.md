# Supervisor submission brief

**Prepared:** 20 August 2026  
**Decision requested:** approve the bounded scientific claims and confirm the
programme-specific declaration, citation, word-count, and PDF requirements.

## Bottom line

The registered experimental programme is complete. The dissertation does not
claim a new VLM architecture, state-of-the-art retrieval, or compact-model
parity. Its main scientific contribution is narrower: in this frozen
cross-modal alignment setting, stale negatives became harmful after one update,
the two modalities showed different damage sensitivity, and matched embedding
drift did not imply matched retrieval damage. The engineering endpoint then
quantifies the boundary: dual FreezeShift reached 62.20 ± 0.45% Flickr30k-test
mean bidirectional R@1 with 4.862M locally trainable inference parameters in a
49.163M-parameter deployed stack. [Evidence: `docs/dissertation.md`, Sections
4.3, 4.13, 5.2, and 8; `artifacts/07_final_evaluation/zero_shot/report/aggregate_results.csv`.]

## Claims recommended for approval

1. **Scientific lead.** Cross-batch memory staleness was modality asymmetric:
   harm began at age one, matched-drift conditions differed by up to 4.74pp,
   and the overlap-restricted text response was about 7.2 times steeper than the
   image response. This establishes a dissociation, not a complete causal
   mechanism. [Evidence: `docs/dissertation.md`, Sections 4.3, 5.2, 6.4.]
2. **Strict-freezing result.** M_T1 reached 52.90 ± 0.87% on the frozen-roster
   Flickr30k test across seeds 42/43/44. [Evidence: final-evaluation aggregate
   table and `docs/dissertation.md`, Section 4.13.]
3. **Bounded-adaptation result.** Dual FreezeShift reached 62.20 ± 0.45%; the
   9.30pp difference from M_T1 is the measured cost of strict freezing in this
   system. [Evidence: same locations.]
4. **Reference boundary.** FreezeShift retained 91.18% of OpenCLIP ViT-B/32's
   test score at 32.5% of its total size, but only 79.49% of MobileCLIP2-S0's
   score at 65.7% of its size. The OpenCLIP comparison is a field-anchor result;
   MobileCLIP2-S0 remains the primary compact reference. [Evidence:
   `docs/dissertation.md`, Sections 4.13 and 5.1.]
5. **Negative efficiency result.** Fixed TokenShift merging saved 0.98--1.61ms
   but lost 5.69--15.95pp across the completed four-arm, three-seed study; no
   merge was promoted. [Evidence: `docs/dissertation.md`, Section 4.10.]

## Claims that should remain out

- state of the art or compact-reference parity;
- a new attention, pooling, frozen-alignment, or LoRA architecture;
- “one third the size of MobileCLIP2-S0”;
- a claim that drift fully explains stale-memory damage;
- broad CLIP-like generalisation, especially given weak Oxford-IIIT Pets
  zero-shot results;
- a universal claim that CC3M, hard negatives, or token merging cannot help.

[Evidence: `docs/completion_status.md`, Claim boundary; `docs/dissertation.md`,
Sections 5 and 6; final-evaluation report, classification diagnostics.]

## Literature position after refresh

Freeze-Align, SAIL, ShareLock, STRUCTURE, and SOTAlign establish frozen or
lightweight unimodal alignment as prior art. Their published results use
different encoders, data volumes, objectives, and evaluation protocols, so the
dissertation reports them with conspicuous caveats and makes no scalar rank
claim. A 20 August refresh found no source that supersedes the stated novelty
boundary. [Evidence: `docs/literature_review.tex`, Sections 3.1--3.2 and 7;
`docs/literature_gap_closure_20260810.md`, submission-date refresh.]

| Source class | Credibility | Evidence | Recency | Objectivity | Use |
|---|---|---|---|---|---|
| Final repository result tables | High | High | High | High | Primary numerical source of record |
| Preregistered configs and experiment registry | High | High | High | High | Protocol and provenance |
| CVPR/NeurIPS peer papers | High | Medium--High | High | Medium--High | Methodological positioning; protocols differ |
| 2026 arXiv peers | Medium | Medium | High | Medium | Recency check and forward-looking context |

## Remaining submission actions

1. Supervisor: approve or amend the claim set above.
2. Author and programme: confirm declaration wording, citation style, required
   word-count method, and PDF/template rules.
3. Author: inspect the rebuilt 43-page PDF visually, especially wide tables and
   appendix paths. The automated build reports zero errors and zero undefined
   citations/references, but retains several overfull-box warnings.
4. Author: submit without adding test-led model selection or unregistered
   parity-chasing experiments.
