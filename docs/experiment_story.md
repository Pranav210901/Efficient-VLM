# How the experiments build into one dissertation

The dissertation is not the story of repeatedly trying architectures until one
worked. It is the story of turning an initially weak and unfairly measured
frozen-alignment system into a controlled efficiency frontier. Each experiment
answers the uncertainty exposed by the one before it; negative results narrow
the explanation or close an unproductive branch rather than disappearing.

| Stage | What the previous stage left unresolved | Experiment response | What became fixed next |
|---|---|---|---|
| Foundations | Historical scores were not protocol-matched | Rebuild local OpenCLIP, MobileCLIP2/SigLIP2 references and frozen baselines | Use local, split-aware comparisons |
| Compatibility screen | Was the weak result just a bad encoder pair? | Screen pairs under a frozen gate | Preserve failure; diagnose training |
| Recipe diagnosis | Was the ceiling caused by loss, batch, captions, LR, or memory? | Wave-0 recipe factorial | Queue-free InfoNCE, all captions, batch 1024 |
| Queue mechanism | Why did removing memory recover about 19pp? | Capacity, age, batch, and modality studies | No queue; drift alone is not a safety rule |
| Data scale | Was the remaining gap simply too little paired data? | COCO scale, CC3M-only, then mixed training | Distribution and coverage matter, not volume alone |
| Efficiency measurement | Which model is efficient under a fair deployment protocol? | Correct padding, fusion, FLOPs, and native transforms | Use paired end-to-end measurement |
| Distillation and resolution | Can accuracy rise without inference cost, and can latency fall safely? | Teacher distillation and resolution curve | MobileCLIP2 teacher; 224-pixel input |
| Aggregation | Are useful frozen features hidden below global summaries? | Mean, query, and transformer token readers | C4 vision reader and M_T1 text reader |
| Final tuning and diagnostics | Is the remaining gap a simple LR or negative-selection problem? | LR retune, rank decomposition, guarded mining | Keep LR 1.0; stop unsafe hard-negative branch |
| Bounded adaptation | How much is strict freezing itself costing? | Vision-, text-, and dual-tower LoRA | Dual FreezeShift is the adaptation upper bound |
| Timing repair | Did that model genuinely satisfy the latency question? | Same-allocation paired amendment | Both final endpoints are faster than local OpenCLIP |
| Token reduction closure | Could internal token removal move the frontier further? | TokenShift profile, then four-arm accuracy study | No merge: speed gain costs 5.69--15.95pp |
| Sealed final evaluation | Did the development ordering survive without further model selection? | Frozen-roster Flickr30k test and zero-shot diagnostics | Strict freezing costs 9.30pp; compact-reference parity remains unmet |

## The narrative

The project begins by refusing to treat an old CLIP number as a trustworthy
baseline. The first experiments reconstruct the reference models locally and
show that checkpoint, split, transform, and deployment details change the
apparent gap. This establishes the comparison logic used throughout: compact
MobileCLIP2-S0 is the primary jointly pretrained reference and distillation
teacher, OpenCLIP ViT-B/32 is the widely recognised field anchor, and SigLIP2
provides a different jointly pretrained objective. Frozen-alignment systems
such as Freeze-Align, SAIL, ShareLock, and STRUCTURE define the methodological
peer group; the present work does not claim to invent frozen alignment.

With the measurement problem bounded, the next question is whether the frozen
vision and text encoders are inherently incompatible. The formal compatibility
screen fails its preregistered gate. Instead of lowering the gate, the project
asks why. The Wave-0 factorial shows that changing the contrastive loss family
matters little compared with removing the cross-batch memory queue: queue
removal recovers roughly 19 percentage points. That finding transforms an
apparent architecture ceiling into a training-recipe problem and locks the
queue-free, all-caption, batch-1024 recipe for everything that follows.

Because the queue effect is unexpectedly large, it becomes a research question
in its own right. The dose experiment shows monotonic deterioration as capacity
increases. The factorial then shows that harm begins at age one and that the two
modalities respond differently. Finally, matched-drift conditions differ by as
much as 4.74pp, while the overlap-restricted text response is about 7.2 times
steeper than the image response. XBM motivates memory with slow feature drift,
and ACBN already shows that stale memory can sometimes be worse than none; the
new contribution here is narrower and more defensible: in frozen cross-modal
alignment, drift magnitude does not predict the damage and tower sensitivity is
asymmetric. This justifies removing the queue without pretending the causal
mechanism is fully identified. Existing diagnostics narrow one alternative:
same-image queue recurrences were recognised by ID and treated as multi-positive,
so only unannotated cross-image semantic false negatives remain plausible.

Once the recipe is repaired, the dissertation tests whether the remaining gap
is a data problem. More unique COCO coverage helps strongly at matched updates,
so broader data is a plausible next move. Yet the qualified CC3M mirror performs
worse than the COCO control within the fixed budget, and extra passes do not
close the gap. Mixing COCO back into CC3M recovers much of the loss. The sequence
therefore rejects the simple conclusion that more pairs automatically transfer:
caption and source distribution matter alongside scale, consistent with
contrastive scaling literature that treats data distribution as part of the
scaling law.

The efficiency track then repairs the measurement axis before optimising it.
Native transforms, deployed MobileCLIP fusion, dynamic text padding, full-stack
parameters, and operator-level FLOPs replace nominal or mismatched quantities.
Against that corrected protocol, distillation is tested again and still helps,
showing it was not merely masking the broken queue recipe. A resolution sweep
then identifies 224 pixels as the lowest-cost setting that meets the latency
budget without creating a new positional-encoding problem. Teacher experiments
show that the strongest teacher benchmark does not necessarily produce the
strongest student, and an untuned teacher mixture does not beat MobileCLIP2
alone. MobileCLIP2 is therefore retained as a training-only teacher, not an
inference dependency.

The next experiments ask where the remaining frozen information is being lost.
Mean pooling over image patches produces only a small gain; learned-query and
transformer readers produce much larger gains. The vision scaling grid shows
that both width and depth help but saturate sub-additively, selecting the
two-block, 256-dimensional C4 reader below the five-million-parameter budget. A
small learned-query text reader then yields M_T1, the best strictly frozen
endpoint at 54.487% validation R@1 and 2.896M trainable inference parameters.
This is not an architectural novelty claim—Set Transformer pooling and attentive
probing establish the relevant primitives—but an empirical result about how
much useful information global summaries hide in this constrained system.

Final tuning is deliberately allowed to fail. The LR retune loses to factor
1.0, so the existing setting stays fixed. Rank analysis shows that many errors
are fine-ordering failures rather than complete retrieval misses, which makes
hard-negative training tempting. The guarded miner cannot satisfy its frozen
safety constraints, so that branch stops before training. Together these
results prevent post-hoc optimisation from quietly changing the final model.

FreezeShift then measures the price of strict freezing. Vision-only LoRA helps,
text-only LoRA helps more, and dual-tower LoRA helps most, reaching 63.416% with
4.862M trainable inference parameters. This is reported separately as a bounded
adaptation upper bound, not relabelled as a frozen model. An initial latency gate
rejects even its unchanged M_T1 control, revealing session contamination. The
declared same-allocation repair measures OpenCLIP, M_T1, and all LoRA arms
together and confirms that M_T1 and dual FreezeShift are faster than the local
OpenCLIP anchor. The correction is part of the scientific story because it
changes eligibility without rewriting the failed first measurement.

The final branch asks whether reducing internal vision tokens can move the
frontier further. TokenShift profiling shows a genuine 0.98--1.61 ms speedup,
so a frozen four-arm accuracy experiment is warranted. The closure is decisive:
all block-6 and block-8 variants are faster, but every one loses accuracy; even
the best dual block-8 trade gives up 5.690pp for about 0.934 ms. This links back
to the aggregation finding. Patch tokens are computationally expensive, but
they also contain information the learned reader uses. The correct final models
are therefore the no-merge M_T1 and no-merge dual FreezeShift.

## The dissertation-level answer

The experiments support a qualified answer. The sealed test confirms the
development ordering: M_T1 reaches 52.90 ± 0.87%, dual FreezeShift 62.20 ±
0.45%, OpenCLIP 68.22%, and MobileCLIP2-S0 78.25% mean bidirectional R@1. Dual
FreezeShift therefore retains 91.18% of OpenCLIP at 32.5% of its total size,
but only 79.49% of MobileCLIP2-S0 at 65.7% of its size. The fully frozen result
shows what learned readers can recover without changing the towers; the 9.30pp
test difference quantifies what bounded adaptation adds; and TokenShift shows
that a nominal inference shortcut is not a frontier improvement unless accuracy
survives. The strongest original insight is the queue mechanism boundary—equal
drift does not imply equal damage, and modality sensitivity differs sharply—
while the broader contribution is a transparent chain in which failed screens,
negative data results, invalid timing, stopped mining, rejected token merging,
and weak zero-shot transfer all constrain the final claim.
