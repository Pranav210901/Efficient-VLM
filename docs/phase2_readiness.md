# Phase 2 readiness

`src.phase15.phase2_readiness.run_phase2_readiness` evaluates the evidence state and writes CSV, JSON, and Markdown reports under `results/phase15/phase2_readiness/`. It does not implement Phase 2.

The gate requires safe notebook defaults; canonical retrieval, seed, multi-task development, prediction-coverage, complementarity, oracle, cross-task, and efficiency outputs; valid seed mapping; a schema-valid diversity-constrained `selected_experts.yaml`; existing selected checkpoints and registered encoders; explicit diagnostic-negative restrictions; and train/development hard negatives with zero protected-test-ID overlap.

The status is:

- `READY`: every required check and optional status check passes.
- `READY_WITH_WARNINGS`: required checks pass but an optional benchmark/status warning remains. Warnings still require explicit review before future Phase 2 work.
- `NOT_READY`: at least one required check fails.

Historical CIFAR-100 test, Pets test, and complete-EuroSAT predictions are exploratory. They do not satisfy the development-output check. This is deliberate leakage prevention, not a request to delete them. Future development-split evaluation and training-negative mining must complete before readiness can pass.

For the final allocated Phase 1.5 execution, set `RUN_MODE="full"`, `RESULTS_ONLY=False`, `RUN_PHASE15=True`, and `RUN_DEVELOPMENT_EVALUATION=True`. Run Step 13, then reset the development switch after the resumable evaluator and evidence assembly finish. Training-negative mining has its own independent switch and requires `pyarrow` plus a GPU; it must not be conflated with the existing diagnostic-negative files.
