#!/bin/bash
# Regenerate docs/Dissertation.tex (single self-contained file) and its PDF from:
#   docs/dissertation.md        -> Chapters 1, 3-8 and Appendices A, B
#   docs/literature_review.tex  -> Chapter 2 and the bibliography; Appendix C
#   docs/abstract.md            -> Abstract
# Usage:  docs/tools/rebuild.sh
set -e
TOOLS="$(cd "$(dirname "$0")" && pwd)"
DOCS="$(dirname "$TOOLS")"
AUX="$TOOLS/aux"
mkdir -p "$AUX"

# --- 1. body chapters from markdown -----------------------------------------
python3 "$TOOLS/md2tex.py" "$DOCS/dissertation.md" > "$AUX/body.tex"
python3 "$TOOLS/addcites.py" "$AUX/body.tex"      # \cite at first mention
python3 "$TOOLS/tables.py"   "$AUX/body.tex"      # captions, labels, widths

# --- 2. split body: ch1 | ch3-8 | appendices ---------------------------------
python3 - "$AUX" "$TOOLS" <<'PY'
import sys
AUX, TOOLS = sys.argv[1], sys.argv[2]
b = open(f"{AUX}/body.tex").read()
k, a = b.index("\\chapter{Methodology}"), b.index("\\appendix")
open(f"{AUX}/ch1.tex", "w").write(b[:k])
open(f"{AUX}/chrest.tex", "w").write(b[k:a])
open(f"{AUX}/appx.tex", "w").write(b[a:] + "\n" + open(f"{TOOLS}/appx_c.tex").read())
PY

# --- 3. literature review chapter + bibliography, from the standalone article -
python3 - "$DOCS" "$AUX" <<'PY'
import sys
DOCS, AUX = sys.argv[1], sys.argv[2]
src = open(f"{DOCS}/literature_review.tex").read().split("\n")
start = next(i for i, l in enumerate(src) if l.startswith(r"\section{Introduction and review question}"))
bib   = next(i for i, l in enumerate(src) if l.startswith(r"\begin{thebibliography}"))
# the article's own \appendix (search audit) is lifted out to Appendix C
app   = next(i for i, l in enumerate(src) if l.strip() == r"\appendix")
open(f"{AUX}/litrev.tex", "w").write("\n".join(src[start:app]))
end   = next(i for i, l in enumerate(src) if l.startswith(r"\end{thebibliography}"))
open(f"{AUX}/bib.tex", "w").write("\n".join(src[bib:end + 1]))
PY

# --- 4. abstract -------------------------------------------------------------
python3 - "$TOOLS" "$DOCS" "$AUX" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from md2tex import inline
paras, cur = [], []
for ln in open(f"{sys.argv[2]}/abstract.md"):
    if ln.startswith("# "):
        continue
    if ln.strip():
        cur.append(ln.strip())
    elif cur:
        paras.append(" ".join(cur)); cur = []
if cur:
    paras.append(" ".join(cur))
# the bold line repeats the title, which is already on the title page
out = [inline(p) for p in paras if not (p.startswith("**") and p.endswith("**"))]
open(f"{sys.argv[3]}/abstract.tex", "w").write("\n\n".join(out) + "\n")
PY

# --- 5. concatenate into ONE file -------------------------------------------
python3 - "$TOOLS" "$AUX" "$DOCS" <<'PY'
import sys
TOOLS, AUX, DOCS = sys.argv[1], sys.argv[2], sys.argv[3]
R = lambda p: open(p).read()
doc = R(f"{TOOLS}/preamble.tex").replace("ABSTRACT_PLACEHOLDER", R(f"{AUX}/abstract.tex").rstrip())
doc += "\n\n%%%%%%%%%%%%%%%%CHAPTERS%%%%%%%%%%%%%%%%%%%%%\n\n"
doc += R(f"{AUX}/ch1.tex")
doc += "\\chapter{Literature Review}\n\n" + R(f"{AUX}/litrev.tex") + "\n"
doc += R(f"{AUX}/chrest.tex")
doc += ("%%%%%%%%%%%%%%%%%BIBLIOGRAPHY%%%%%%%%%%%%%%%%\n"
        "\\clearpage\n\\addcontentsline{toc}{chapter}{Bibliography}\n") + R(f"{AUX}/bib.tex")
doc += ("\n\n%%%%%%%%%%%%%%%%%APPENDICES%%%%%%%%%%%%%%%%%%\n"
        "\\clearpage\n\\addcontentsline{toc}{chapter}{Appendices}\n") + R(f"{AUX}/appx.tex")
doc += "\n\\end{document}\n"
open(f"{DOCS}/Dissertation.tex", "w").write(doc)
PY

# --- 6. body word count, substituted into the Word Count page ----------------
WC=$(python3 - "$AUX" <<'PY'
import re, sys
AUX = sys.argv[1]
t = "".join(open(f"{AUX}/{f}").read() for f in ("ch1.tex", "litrev.tex", "chrest.tex"))
t = re.sub(r"\\begin\{(table|longtable|tabular|tabularx)\}.*?\\end\{\1\}", " ", t, flags=re.S)
t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", t)
print(len(t.split()))
PY
)
sed -i "s/WORDCOUNTBODY/$WC/" "$DOCS/Dissertation.tex"

# --- 7. build (aux files stay out of docs/) ---------------------------------
cd "$DOCS"
for i in 1 2 3; do
  pdflatex -interaction=nonstopmode -output-directory="$AUX" Dissertation.tex >/dev/null 2>&1 || true
done
cp "$AUX/Dissertation.pdf" "$DOCS/Dissertation.pdf"

echo "=== $DOCS/Dissertation.tex -> Dissertation.pdf ==="
grep -oE 'Output written on .*\([0-9]+ pages' "$AUX/Dissertation.log" || echo "NO PDF"
echo "body word count:  $WC"
echo "errors:           $(grep -cE '^! ' "$AUX/Dissertation.log" || true)"
echo "undefined refs:   $(grep -icE 'undefined (citation|reference)' "$AUX/Dissertation.log" || true)"
echo "overfull >20pt:   $(grep 'Overfull \\hbox' "$AUX/Dissertation.log" | grep -cE '\([0-9]{2,}\.[0-9]+pt too wide\)' || true)"
