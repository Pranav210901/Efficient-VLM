# TokenShift

TokenShift tests whether the 224×224 FreezeShift dual model can retain its
input resolution while reducing its internal DINOv3 patch sequence.

The intervention is a fixed, non-overlapping 2×2 spatial average after either
the sixth or eighth DINOv3 block. The 14×14 patch grid becomes 7×7 (196 to 49
patch tokens). CLS and all four register tokens remain untouched. Subsequent
blocks receive a regenerated 7×7 DINOv3 rotary-position grid. The operation
adds zero parameters and is not content-adaptive ToMe.

## Why profile first

The point of the study is measured wall-clock efficiency, not nominal FLOPs.
The first stage therefore profiles three graphs in one allocation:

1. the unmodified 224px dual FreezeShift model;
2. merge after block 8 (four reduced-token blocks);
3. merge after block 6 (six reduced-token blocks).

Each graph is independently warmed and timed ten times, and order is shuffled
within each repetition. No candidate is trained or promoted automatically.
After the profile is reviewed, only latency-viable candidates may receive a
separately frozen accuracy study.

## Run the profile

```bash
cd /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm
source .venv/bin/activate
bash tokenshift/submit_profile.sh --partition teaching
```

This submits exactly one teaching-partition GPU job and no training jobs.

Inspect progress:

```bash
squeue -u pp01184
tail -F tokenshift/logs/slurm/*.out tokenshift/logs/slurm/*.err
```

When complete:

```bash
cat tokenshift/results/profile/report.md
```

## Result

The profile completed on an RTX PRO 6000 Blackwell with ten independently
warmed repetitions per arm. Relative to the no-merge mean Q3 of 9.122 ms,
merging after block 8 reduced mean Q3 by 0.984 ms and merging after block 6
reduced it by 1.606 ms. Both paired 95% bootstrap intervals were wholly below
zero. This package is therefore a latency-only result. The subsequent
three-seed accuracy closure is reported in
`tokenshift_training/results/report/report.md`; every merge arm lost accuracy,
so no training decision was promoted from this profile.
