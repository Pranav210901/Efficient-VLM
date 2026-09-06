# Dissertation completion status

**Assessed:** 20 August 2026

## Bottom line

The registered scientific programme and its authorised sealed evaluation are
complete. All planned TokenShift computation has closed, its negative accuracy
result has been incorporated, and the final candidates and references now have
split-matched Flickr30k-test evidence. There is no remaining architecture,
training, profiling, evaluation, or analysis experiment required by the
registered study.

The scientific report has been placed in the Surrey template with a title page,
declaration, word count, bibliography and appendices. Formal submission still
requires author/supervisor approval and any programme-specific checks not
represented in the repository, especially declaration wording and final
bibliographic/style requirements.

## Completed

- Integrated dissertation with abstract, research question, related work,
  methodology, results, discussion, limitations, reproducibility, ethics,
  conclusion, bibliography, and experiment inventory.
- Twenty-nine registered experiment families: 26 `COMPLETE`, one
  `STOPPED_BY_GATE`, one `INCOMPLETE_ARCHIVED`, and one `SUPERSEDED`.
- Fully frozen line through M_T1 and bounded dual-LoRA FreezeShift upper bound.
- Same-allocation OpenCLIP comparison and declared latency amendment.
- TokenShift profiling and four-arm, three-seed accuracy closure; no merge arm
  was promoted because each lost 5.69--15.95pp.
- Literature positioning that treats frozen alignment and learned-query pooling
  as prior art and limits originality to the empirical/mechanism findings.
- A 25-area literature-gap disposition backed by 20 decomposed searches, a
  globally deduplicated 273-record targeted export, primary-source checks, and
  a 37-entry submission bibliography.
- Recoverable GitHub cleanup retaining source, configs, tests, manuscript, and
  compact evidence while excluding raw data, environments, checkpoints,
  tensors, caches, and logs.
- A dissertation-flow narrative explaining how every experiment responds to
  the uncertainty created by the previous result.

## Final selected endpoints

The strictly frozen endpoint is M_T1: 54.487% three-seed Flickr30k-validation
mean bidirectional R@1, 2,896,389 trainable inference parameters, and 0.119 ms
faster than the local OpenCLIP reference in the same allocation.

The bounded-adaptation endpoint is dual FreezeShift: 63.416% validation mean
R@1, 4,862,469 trainable inference parameters, and 0.101 ms faster than
OpenCLIP (paired 95% bootstrap CI -0.145 to -0.016 ms). It retains 90.67% of
OpenCLIP validation R@1 but remains 6.525pp behind. These are development
results, not final-test claims.

Its full deployed stack has 49,162,629 parameters versus 151,277,313 for the
OpenCLIP ViT-B/32 field anchor: 32.5% as many total parameters. That is the only
defensible “about one third” comparison. MobileCLIP2-S0 is the primary compact
jointly pretrained reference and teacher; dual FreezeShift has 65.7% of its
74,835,073 total parameters. The final split-matched test comparison is 62.20%
versus 78.25%, or 79.49% retention. This does not establish compact-reference
parity.

## Deliberately outstanding

1. Obtain author/supervisor approval of the final scientific claims and report.
2. Confirm the university's required declaration wording, citation style and
   submission-format checks against the rebuilt PDF.
3. Obtain supervisor judgement on the disclosed post-observation latency
   amendment and conservative modality-gap interpretation.
4. Refresh the literature search immediately before submission because the
   efficient/frozen-VLM field is moving quickly.
5. Treat compact-reference parity as an unmet result, not an outstanding run:
   the split-matched test shows 62.20% for FreezeShift versus 78.25% for
   MobileCLIP2-S0.

## Claim boundary

Safe headline:

> Under a fixed same-allocation inference comparison, learned aggregation over
> frozen unimodal encoders produced a 54.487% Flickr30k-validation model with
> 2.90M trainable parameters; bounded dual-tower LoRA raised this to 63.416%
> with 4.86M parameters while remaining faster than the local OpenCLIP
> reference. In the frozen-roster Flickr30k test, the two endpoints reached
> 52.90 ± 0.87% and 62.20 ± 0.45% across three seeds. The adapted endpoint
> retained 91.18% of OpenCLIP's test R@1 at 32.5% of its total size, but only
> 79.49% of MobileCLIP2-S0's score at 65.7% of its size.

Unsafe claims include “state of the art,” “test-confirmed parity,” “new attention
architecture,” “queue drift fully explains the mechanism,” “CC3M never helps,”
“one third the size of MobileCLIP2,” or “token merging improves the frontier.”

## Verification record

Verification completed after GitHub cleanup:

- experiment registry: **PASS**, 29 entries and 29 result roots;
- status totals: **26 complete**, one stopped, one incomplete-archived, one
  superseded;
- test suite: **210 passed, 3 skipped, 0 failed** using the quarantined verified
  environment so no dependency tree was reintroduced into the submission;
- retained package: approximately **90 MB** excluding ignored quarantine;
- individual retained files over 50 MB: **none**;
- JSON/YAML parsing: **2,447 JSON** and **115 YAML** files, zero errors;
- local Markdown links: **zero broken** after compact legacy evidence repair;
- retained symlinks: **zero broken** and no result link depends on quarantine;
- post-test Python bytecode and test caches: moved into manifested quarantine
  passes, most recently `github_submission_cleanup_20260810_litverify` after the
  literature-gap verification run.
