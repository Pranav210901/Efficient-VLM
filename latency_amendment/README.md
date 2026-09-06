# FreezeShift latency amendment

The original FreezeShift gate compared a contemporaneous student allocation
against an OpenCLIP-derived `9.480 ms` number from another session. In the
FreezeShift allocation, the unchanged M_T1 control itself measured Q3
`9.4970 ms`. The original negative verdict is retained, but its absolute
cross-session eligibility classification is not treated as a reliable headline.

This post-observation measurement amendment was frozen before corrective
measurements. It profiles OpenCLIP, M_T1, and all three FreezeShift arms in one
allocation. Ten independently warmed repetitions are interleaved in a fixed
random order. The primary comparison is paired Q3 latency against OpenCLIP;
paired differences from M_T1 are secondary adaptation-overhead diagnostics.

No training, evaluation, checkpoint selection, or Flickr test access occurs.

## Completed result

Slurm job 2297019 completed ten same-allocation repetitions on an RTX PRO 6000
Blackwell. OpenCLIP's mean Q3 was 9.267 ms. M_T1, vision, text, and dual measured
9.148, 9.168, 9.173, and 9.166 ms respectively. Every candidate's paired 95%
bootstrap interval versus OpenCLIP lay entirely below zero. The corrective
measurement therefore classifies all four as faster than the reference while
retaining the original cross-session verdict as historical evidence.

## Reproduce

```bash
cd /mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm
bash latency_amendment/submit.sh --partition teaching
```

Monitor:

```bash
squeue -u pp01184
tail -F latency_amendment/logs/slurm/*.out latency_amendment/logs/slurm/*.err
```

Result:

```bash
cat latency_amendment/results/report/report.md
```
