# F4 — Routing was scoped out on measurement, not lost to drift

**Status:** original H4 ("the project drifted") **refuted**. Confidence: high.
**Supersedes:** `F4_mechanism_drift.md`, deleted 2026-08-06.

## What the original finding claimed, and why it was wrong

The first version of this finding asserted that the project abandoned the token
*routing* direction recommended on 2026-07-28 and built static token
*aggregation* instead, without deciding to. It called this drift and treated the
earlier novelty argument as stranded.

That was an inference from repo contents without checking chronology. Experiment
timestamps refute it.

## What actually happened

The 2026-07-28 landscape prescribed **Stage A controls before any router**. Those
controls ran, in the prescribed order:

| Timestamp | Experiment | Stage A step |
|---|---|---|
| 07-29 15:09 | `resolution_arm` | A.1 — 224px retrain |
| 07-29 21:13 | `efficiency_frontier` | A.2 — dynamic padding measurement |
| 07-29 21:45 → 07-30 03:41 | `resolution_distillation_224*`, `teacher_extension_224` | Stage D — accuracy recovery |
| 07-30 09:17 → 13:54 | `token_projection_*`, `token_aggregator_scale_*` | capacity curves |
| 07-30 → 08-01 | `mixed_data`, `e5_c4_224`, `text_aggregation`, `probe1` | headroom diagnosis |

## The measurement that redirected the work

The **pre-training/random-graph profile** recorded in
`token_aggregator_scale_training/manifests/design.json`:

> "C4 added 1,252,225 parameters and only 0.204 ms over the contemporaneous CLS
> baseline. At these scales the frozen image tower dominates full-stack latency;
> the binding declared aggregator constraint is the 5M parameter budget, not
> latency."

The later **trained-weight controlled profile** in
`token_aggregator_scale_training/report/trained_latency.json` measured C4 at
8.9866 ms versus a same-allocation CLS baseline of 8.5486 ms: a **0.4380 ms**
median delta (Q3 9.1670 ms). Both values are valid, but answer different
questions. The 0.204 ms figure was the pre-training admission measurement; the
0.438 ms figure is the post-training result and is the value used when reporting
the completed model. Neither measurement changes the conclusion that C4 remains
below the 9.48 ms latency ceiling.

Parameter accounting must likewise remain explicit: C4's token aggregator adds
**1,252,225 parameters (1.25M)** within a **2,730,628-parameter (2.73M)
trainable inference stack**. The remaining 1,478,403 trainable parameters are
the two residual projection heads. C4 is not a 2.73M-parameter aggregator.

Corroborated by `efficiency_frontier/dynamic_padding_report.json`: text latency
is **1.49 ms of a 10.75 ms** full-stack median. The image tower is ~86% of cost;
aggregation capacity is close to free.

**Consequence:** the efficiency motivation for routing collapsed on measurement.
The binding constraint moved to parameters, and the work correctly repivoted to
the accuracy gap and to locating headroom — which is what
`probe1_error_decomposition` exists to do. Its conclusion then gated the text
study rather than the reverse:

> "C4 has no excess directional asymmetry relative to OpenCLIP; text aggregation
> is a symmetry and fine-ranking mechanism probe, not a response to an
> established text-tower bottleneck."

That is evidence-driven sequencing, not drift.

## The genuine remaining gap

Stage A steps **4 and 5** were never executed:

> 4. Apply training-free ToMe/ATS and random token removal at matched realised latency.
> 5. Establish fixed token-budget curves before training a dynamic router.

`grep` over `src/` finds no `token_budget`, `prune`, `router`, `keep_ratio` or
ToMe implementation. So routing was **bypassed, not falsified** — the gate that
would have justified or killed it never ran.

This is a defensible position but it must be *stated* rather than left implicit.

## Recommended disposition

**Keep aggregation. Document routing as an explicit scoping decision.**

Write it as: *routing was scoped out under the measured latency and parameter
budget; static aggregation was selected instead.* Place routing in future work,
beside the queue-asymmetry probes.

Building a router now would reopen a closed architecture search, require its own
preregistration, and produce a thin final chapter. The measurement that closed it
is itself reportable — the 2026-07-28 landscape named "FLOP/token reduction does
not translate into batched latency for frozen SSL encoders" as a useful negative
result, and the evidence for it largely exists already.

Related: [[F1_aggregation_not_novel]], [[F2_no_exact_occupant]]
