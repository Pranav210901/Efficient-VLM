# Refresh targets

Re-run before implementation freeze and again immediately before submission.
H2 (no exact occupant) is the claim most likely to decay.

## Priority 0 — the run this folder does not contain

**Commission a separate landscape scoped to queue staleness and modality
asymmetry.** This folder's searches covered token aggregation and text pooling
only; the queue/drift work was never assessed against prior art despite the
original report generalising as though it had been.

Scope that run to:

- queue staleness and cross-batch memory pathology in contrastive learning
- projector drift / cross-modal feature staleness
- **modality-asymmetric degradation** — matched cumulative drift failing to
  predict matched degradation across the two towers
- image-only vs text-only memory banks in bidirectional retrieval

Known going in: MoCo, XBM and EfficientCLIP establish queue staleness as a known
phenomenon. Nothing was found on the modality asymmetry, but that is an
**unverified gap** from quick searching, not an established one. That run decides
the publishable claim; this one cannot.

## Priority 1 — novelty-critical watches

- Citations of **arXiv:2506.10178** (Attention, Please!) — any follow-up that
  extends attentive probing to *cross-modal retrieval* would collapse the main
  remaining gap.
- New work pairing **attention pooling + frozen heterogeneous encoders +
  contrastive retrieval** in one paper.
- **SigLIP 3** or successor: if it adds a parameter-budgeted or gated MAP head,
  the differentiator narrows further.
- Any paper adding **measured latency gating** (not parameter count) to a
  probing or pooling study.

## Priority 2 — inherited watches, still live

Carried forward from `docs/research/alignment_queue_prior_art` and
`research/efficient_frozen_vlm_2026`:

- New papers citing Cross-Batch Memory with image-text objectives.
- Queue-capacity / queue-age ablations in SAIL, Freeze-Align, ShareLock follow-ups.
- Final publication status for SCOPE, MOVE, MobileCLIP2.
- Dual-frozen DINOv3 + text-encoder alignment work.
- Phrases: "projector drift", "cross-modal feature staleness", "dual-frozen queue".
- SaMer follow-up versions; ICAR final venue.
- CVPR / NeurIPS / ICLR 2026 efficient-retrieval papers.

## Priority 3 — only if routing is resumed

Dormant unless the F4 decision goes back to conditional computation:
MoDE CLIP, SCOPE, MADTP, Patch Ranking, ToMe/ATS/DiffRate, ICAR.

## Verification debt (do before any submitted bibliography)

Entries in `sources.csv` with `verification_level = search_summary` have IDs that
appeared in live search results but were **not individually fetched**. Fetch and
confirm title/authors/venue for every Tier 1 and Tier 2 row.

Unconfirmed and deliberately excluded from the CSV — resolve IDs if cited:
Perceiver, Perceiver-IO, BLIP-2 / Q-Former, DINOv2, ToMe.

## Redesign triggers

Trigger a contribution rethink if a paper appears that combines:

- frozen heterogeneous encoders, **and**
- a global dual-encoder retrieval space, **and**
- learned token aggregation on either tower, **and**
- an explicit deployment budget (latency or parameters) as a selection gate.

Three of four already exist in separate papers. A single paper with all four
ends the current framing.

## Experimental refresh triggers

- Replace retrospective one-seed queue values with multi-seed evaluation-only results.
- Add matched-age batch-512 / batch-1024 evidence.
- Add image-only and text-only queue evidence.
- Check `LearnedQueryPatchPool` seed variance against the 196-token instability
  warning in `arXiv:2601.09322`.
