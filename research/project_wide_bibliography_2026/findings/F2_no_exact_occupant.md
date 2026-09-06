# F2 — No paper occupies the exact protocol setting

**Status:** H2 not falsified. Confidence: **medium** (absence of evidence).

No source in the 44-item corpus combines all of:

1. frozen heterogeneous encoders (SSL vision + independent sentence encoder),
2. a single global dual-encoder retrieval space,
3. learned token aggregation on **both** towers,
4. a **pre-registered** hierarchical selection protocol with hash-chained
   provenance and a sealed test split,
5. **measured latency** as a hard admission gate alongside a <5M parameter
   ceiling.

The nearest occupants each miss at least two:

| Work | Misses |
|---|---|
| Attention, Please! (2506.10178) | cross-modal setting; latency gate; pre-registration |
| SigLIP 2 | frozen heterogeneous pair; parameter budget; pre-registration |
| SAIL / Freeze-Align / ShareLock | token aggregation; latency gating |
| CoCa | frozen encoders; budget |

## Important caveat

This is an **absence-of-evidence** finding, and it is the weakest kind of claim
in this report. It rests on the union of three search rounds (2026-07-26,
2026-07-28, 2026-08-06). It is *not* a novelty guarantee. The prior folder
already recorded the same caution and it still stands:

> "The exact claim requires another targeted search immediately before paper
> submission because July 2026 work is still appearing."

## Consequence

The contribution is **methodological, not architectural**. Frame it that way and
it survives; frame it as a new module and F1 defeats it.

Related: [[F1_aggregation_not_novel]]
