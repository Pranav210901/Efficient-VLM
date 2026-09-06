#!/usr/bin/env python3
"""Add captions/labels to the converted body tables and fix over-wide layouts."""
import sys

CAPS = [
    ("tab:compatibility", r"Best local encoder pairing by development task in the compatibility screen."),
    ("tab:queue", r"Retrieval degradation as stale cross-batch memory capacity increases."),
    ("tab:aggregation", r"Effect of vision-token aggregation strategy, measured against a matched baseline."),
    ("tab:mt1", r"Properties of \MTone{}, the strongest fully frozen-backbone model."),
    ("tab:freezeshift", r"FreezeShift low-rank adaptation arms against the frozen \MTone{} baseline."),
    ("tab:tokenshift-latency", r"Measured latency effect of fixed internal token merging."),
    ("tab:tokenshift-accuracy", r"Accuracy and latency of token merging against matched no-merge parents."),
    ("tab:references", r"Final candidates against the compact reference, field anchor and "
                       r"alternative-objective reference. Protocol blocks differ: candidate rows use "
                       r"Flickr30k validation; the compact and SigLIP2 rows use Flickr30k test."),
    ("tab:final-test", r"Frozen-roster Flickr30k test results for the selected endpoints and fixed references."),
]

# tables whose default l/r columns do not fit \textwidth
WIDTH_FIX = {
    "tab:compatibility":       (r"\begin{tabular}{llr}",    r"\begin{tabular}{L{4.0cm}L{5.0cm}r}"),
    "tab:queue":               (r"\begin{tabular}{lr}",    r"\begin{tabular}{L{7.5cm}r}"),
    "tab:aggregation":         (r"\begin{tabular}{lr}",    r"\begin{tabular}{L{7.5cm}r}"),
    "tab:mt1":                 (r"\begin{tabular}{lr}",    r"\begin{tabular}{L{8.5cm}r}"),
    "tab:freezeshift":         (r"\begin{tabular}{lrrr}",  r"\begin{tabular}{L{4.5cm}rrr}"),
    "tab:tokenshift-latency":  (r"\begin{tabular}{lrr}",   r"\begin{tabular}{L{5.5cm}rr}"),
    "tab:tokenshift-accuracy": (r"\begin{tabular}{lrrr}",  r"\begin{tabular}{L{4.5cm}rrr}"),
    "tab:references":          (r"\begin{tabular}{llrrl}",
                                "\\small\n\\begin{tabular}{L{2.1cm}L{2.9cm}R{1.9cm}R{2.3cm}L{3.3cm}}"),
    "tab:final-test":          (r"\begin{tabular}{lrrr}",  r"\begin{tabular}{L{4.5cm}rrr}"),
}

path = sys.argv[1]
lines = open(path).read().split("\n")
out, n = [], 0
for ln in lines:
    if ln.strip().startswith(r"\begin{tabular}") and n < len(CAPS):
        lab, cap = CAPS[n]
        old, new = WIDTH_FIX.get(lab, (None, None))
        out.append(r"\caption{%s}" % cap)
        out.append(r"\label{%s}" % lab)
        out.append(new if old and ln.strip() == old else ln)
        n += 1
        continue
    out.append(ln)
open(path, "w").write("\n".join(out))
print("captioned %d tables" % n)
