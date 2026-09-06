# Evaluation protocol

> **Historical protocol, superseded for current status.** This document records
> the rules in force while the final test was sealed. The authorised frozen-roster
> evaluation was run on 11 August 2026. Current results and claim boundaries are
> in `artifacts/07_final_evaluation/zero_shot/report/report.md` and
> `docs/completion_status.md`; the historical instructions below are preserved
> unchanged as provenance.

The machine-readable policy is frozen in `configs/evaluation_protocol.yaml`. Generated copies and the data-usage table live under `results/phase15/methodology/`.

| Dataset | Development and selection data | Reserved final evidence | Current historical status |
|---|---|---|---|
| COCO retrieval | val2017 | Flickr30k or another external retrieval benchmark, if installed | COCO validation is development evidence, not untouched final evidence |
| CIFAR-100 | deterministic development partition of official train | official test | existing official-test results are exploratory |
| Oxford-IIIT Pets | deterministic development partition of trainval | official test | existing official-test results are exploratory |
| EuroSAT | stored deterministic stratified development split | stored stratified test split | existing whole-dataset results are exploratory |
| Winoground | evaluation-only | evaluation-only | unavailable or evaluation-only |
| SugarCrepe | evaluation-only | evaluation-only | unavailable or evaluation-only |

The EuroSAT split manifest is written once to `data/splits/eurosat_split_manifest.csv` with a fixed seed. It is reused and validated rather than regenerated. Current held-out prediction caches are preserved for reproducibility, complementarity diagnostics, and error inspection, but they cannot provide training examples or a final confirmatory claim.

The notebook's Step 13 exposes a separately gated `RUN_DEVELOPMENT_EVALUATION` path. It sends the three classification development datasets through the resumable multi-GPU evaluator under `results/phase1_multitask_development/`, then combines those rows with the already-complete COCO validation evidence. The combination is strict: all configurations must cover all four datasets, and no historical classification-test row is copied.

## Current final-evaluation state

Flickr30k validation is installed and has been used for development selection.
Flickr30k test is installed but remains sealed. After the same-allocation
latency amendment, the frozen final development candidate is dual-tower
FreezeShift; M_T1 remains the final fully frozen-backbone candidate. No result
in this repository may be described as a confirmatory Flickr30k-test result.

The test seal should be opened once, only after author/supervisor approval of
the model-freeze record in `docs/final_model_freeze.md`. Until then, the project
must state that fully untouched external retrieval test evidence is unavailable.
