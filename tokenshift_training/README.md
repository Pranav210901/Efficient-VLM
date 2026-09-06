# TokenShift training: frozen versus dual-LoRA

This isolated package completed the two latency-validated TokenShift locations
against two parent architectures:

| Parent | block 8 | block 6 | Existing no-merge baseline |
|---|---:|---:|---|
| strictly frozen M_T1/C4 | 3 seeds | 3 seeds | reused |
| FreezeShift dual LoRA | 3 seeds | 3 seeds | reused |

The input remains 224×224. A fixed 2×2 internal spatial average reduces 196
patch tokens to 49 after DINOv3 block 8 or block 6, while preserving CLS and
four register tokens. TokenShift adds no parameters.

The package does not modify `tokenshift/` or `freezeshift/`. It writes only to
`tokenshift_training/results`, `tokenshift_training/checkpoints`, and
`tokenshift_training/logs`.

## Submission

No job is submitted during build or validation. From the AI Surrey login node:

```bash
cd /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm
source .venv/bin/activate

bash tokenshift_training/submit.sh \
  --partition teaching \
  --max-total-gpus 8 \
  --resume
```

One launcher queues the entire dependency graph. The 12 training tasks and 12
seed-level evaluation tasks are arrays capped at the requested concurrency.
Training resumes from each run's `latest.pt`; completed epoch evaluations are
reused.

## Outputs

- `results/manifests/design.json`: frozen package/config receipt
- `results/<arm>/report/report.json`: per-arm 24-epoch trajectory result
- `results/profile/report.json`: same-allocation repeated latency result
- `results/report/report.json`: combined frozen-vs-LoRA comparison

Flickr30k test is never read. No winner is selected automatically.

## Closed outcome

All four three-seed arms were faster but less accurate than their matched
no-merge parent. The losses were 12.581pp and 15.950pp for the frozen block-8
and block-6 arms, and 5.690pp and 10.861pp for the corresponding dual-LoRA
arms. The frozen decision in `docs/final_model_freeze.md` therefore retains
both no-merge parents. No additional training is pending.
