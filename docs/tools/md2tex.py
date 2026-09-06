#!/usr/bin/env python3
"""Convert dissertation.md body chapters to LaTeX for the Surrey template."""
import re, sys

URL2KEY = {
    "https://arxiv.org/abs/2303.17127": "ajanthan2023acbn",
    "https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Assessing_and_Learning_Alignment_of_Unimodal_Vision_and_Language_Models_CVPR_2025_paper.html": "zhang2025sail",
    "https://arxiv.org/abs/2506.10178": "psomas2026attention",
    "https://doi.org/10.1145/3805622.3810607": "xu2026bhfa",
    "https://proceedings.mlr.press/v202/li23q.html": "li2023blip2",
    "https://openreview.net/forum?id=wqBHJNqeQJ": "ruthardt2026sharelock",
    "https://openaccess.thecvf.com/content/CVPR2024/html/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.html": "yang2024clipkd",
    "https://openreview.net/forum?id=Ee277P3AYC": "yu2022coca",
    "https://arxiv.org/abs/1912.06798": "wang2020xbm",
    "https://arxiv.org/abs/2304.14108": "gadre2023datacomp",
    "https://arxiv.org/abs/2606.28551": "farina2026datacompvlm",
    "https://openaccess.thecvf.com/content/CVPR2026F/html/Rubab_Dyna-ViT_Parameter-Free_Pre-Encoder_Token_Pruning_for_Efficient_Vision_Transformers_CVPRF_2026_paper.html": "rubab2026dynavit",
    "https://proceedings.nips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html": "rao2021dynamicvit",
    "https://openaccess.thecvf.com/content/CVPR2026/html/Kim_FALCON_False-Negative_Aware_Learning_of_Contrastive_Negatives_in_Vision-Language_Alignment_CVPR_2026_paper.html": "kim2026falcon",
    "https://ojs.aaai.org/index.php/AAAI/article/view/40056": "zhang2026halora",
    "https://openaccess.thecvf.com/content/CVPR2025/html/Maniparambil_Harnessing_Frozen_Unimodal_Encoders_for_Flexible_Multimodal_Alignment_CVPR_2025_paper.html": "maniparambil2025freezealign",
    "https://arxiv.org/abs/2111.07991": "zhai2022lit",
    "https://arxiv.org/abs/2103.15686": "zhao2021meel",
    "https://arxiv.org/abs/2203.02053": "liang2022gap",
    "https://openaccess.thecvf.com/content/CVPR2024/html/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.html": "vasu2024mobileclip",
    "https://machinelearning.apple.com/research/mobileclip2": "faghri2025mobileclip2",
    "https://openaccess.thecvf.com/content/CVPR2021/html/Chun_Probabilistic_Embeddings_for_Cross-Modal_Retrieval_CVPR_2021_paper.html": "chun2021pcme",
    "https://aclanthology.org/2023.acl-long.721/": "cao2023pumer",
    "https://arxiv.org/abs/2212.07143": "cherti2023scaling",
    "https://arxiv.org/abs/2602.23353": "roschmann2026sotalign",
    "https://papers.nips.cc/paper_files/paper/2025/hash/dee8f820d86aca28ab0328a9243020f9-Abstract-Conference.html": "groger2025structure",
    "https://arxiv.org/abs/1810.00825": "lee2019settransformer",
    "https://arxiv.org/abs/2502.14786": "tschannen2025siglip2",
    "https://openaccess.thecvf.com/content/ICCV2023/html/Wu_TinyCLIP_CLIP_Distillation_via_Affinity_Mimicking_and_Weight_Inheritance_ICCV_2023_paper.html": "wu2023tinyclip",
    "https://arxiv.org/abs/2312.01026": "kim2023tokenfusion",
    "https://arxiv.org/abs/2210.09461": "bolya2022tome",
    "https://openaccess.thecvf.com/content/CVPR2022/html/Sung_VL-Adapter_Parameter-Efficient_Transfer_Learning_for_Vision-and-Language_Tasks_CVPR_2022_paper.html": "sung2022vladapter",
    "https://proceedings.mlr.press/v280/yaras25a.html": "yaras2025modalitygap",
    "https://openaccess.thecvf.com/content/CVPR2026/html/Xie_Saliency-Driven_Token_Merging_for_Vision_Transformers_CVPR_2026_paper.html": "xie2026sadtm",
}

UNI = [
    ("\u2014", "---"), ("\u2013", "--"), ("\u2212", "-"),
    ("\u201c", "``"), ("\u201d", "''"), ("\u2018", "`"), ("\u2019", "'"),
    ("\u00d7", r"$\times$"), ("\u00b1", r"$\pm$"), ("\u2264", r"$\le$"),
    ("\u2265", r"$\ge$"), ("\u2192", r"$\rightarrow$"), ("\u2026", r"\ldots"),
    ("\u00a0", "~"),
]

def esc(t):
    """Escape LaTeX specials in plain prose (run before markup expansion)."""
    for a, b in UNI:
        t = t.replace(a, b)
    t = re.sub(r"([%&#])", r"\\\1", t)
    return t

def inline(t):
    """Markdown inline markup -> LaTeX.

    Code spans are stashed BEFORE escaping: esc() turns curly quotes into
    `` and '', which would otherwise be picked up as code-span delimiters.
    """
    holds = []
    def stash(m):
        holds.append(m.group(1))
        return "\x00%d\x00" % (len(holds) - 1)
    t = re.sub(r"`([^`]+)`", stash, t)
    t = esc(t)
    # links
    def link(m):
        text, url = m.group(1), m.group(2)
        key = URL2KEY.get(url)
        if key:
            return "%s~\\cite{%s}" % (text, key)
        if url.startswith("http"):
            return "%s\\footnote{\\url{%s}}" % (text, url)
        return texttt(url)  # internal repo path
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
    # M_T1 -> macro; then escape any remaining bare underscores
    t = t.replace("M_T1", r"\MTone")
    t = re.sub(r"(?<![\\\{])_", r"\\_", t)
    t = t.replace(r"\MTone", r"\MTone{}")
    # restore code spans
    for i, s in enumerate(holds):
        t = t.replace("\x00%d\x00" % i, texttt(s))
    return t

def texttt(s):
    """\\texttt{} for a literal string, with break opportunities after path
    separators and underscores so long repository paths do not overrun the
    text block (the template loads no line-breaking URL package)."""
    s = (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
          .replace("$", r"\$").replace("%", r"\%").replace("&", r"\&")
          .replace("#", r"\#"))
    for sep in ("/", r"\_", "."):
        s = s.replace(sep, sep + r"\linebreak[1]")
    return r"\texttt{%s}" % s

def cell(t):
    return inline(t.strip())

def convert(md):
    lines = md.split("\n")
    out, i = [], 0
    in_appendix = False
    while i < len(lines):
        ln = lines[i]

        # ---- headings ----
        m = re.match(r"^## Appendix ([AB])\. (.+)$", ln)
        if m:
            if not in_appendix:
                out.append(r"\appendix")
                in_appendix = True
            out.append(r"\chapter{%s}" % cell(m.group(2)))
            i += 1; continue
        m = re.match(r"^## (?:\d+\. )?(.+)$", ln)
        if m:
            out.append(r"\chapter{%s}" % cell(m.group(1)))
            i += 1; continue
        m = re.match(r"^### (?:[\d.]+ )?(.+)$", ln)
        if m:
            out.append(r"\section{%s}" % cell(m.group(1)))
            i += 1; continue
        m = re.match(r"^#### (?:[\d.]+ )?(.+)$", ln)
        if m:
            out.append(r"\subsection{%s}" % cell(m.group(1)))
            i += 1; continue

        # ---- display math (already LaTeX in source) ----
        if ln.strip() == r"\[":
            block = []
            i += 1
            while lines[i].strip() != r"\]":
                block.append(lines[i]); i += 1
            i += 1
            out.append(r"\begin{equation*}")
            out.extend(block)
            out.append(r"\end{equation*}")
            continue

        # ---- tables ----
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            spec_row = [c.strip() for c in lines[i + 1].strip().strip("|").split("|")]
            colspec = "".join("r" if s.endswith(":") and not s.startswith(":") else "l"
                              for s in spec_row)
            # widen first column for readability
            colspec = "l" + colspec[1:]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append(r"\begin{table}[htbp]")
            out.append(r"\centering")
            out.append(r"\begin{tabular}{%s}" % colspec)
            out.append(r"\toprule")
            out.append(" & ".join(cell(c) for c in header) + r" \\")
            out.append(r"\midrule")
            for r in rows:
                out.append(" & ".join(cell(c) for c in r) + r" \\")
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}")
            out.append(r"\end{table}")
            continue

        # ---- blockquote (research question) ----
        if ln.startswith("> "):
            block = []
            while i < len(lines) and lines[i].startswith(">"):
                block.append(lines[i].lstrip(">").strip()); i += 1
            out.append(r"\begin{quote}")
            out.append(r"\itshape")
            out.append(inline(" ".join(block)))
            out.append(r"\end{quote}")
            continue

        # ---- ordered list ----
        if re.match(r"^\d+\. ", ln):
            items = []
            while i < len(lines):
                if re.match(r"^\d+\. ", lines[i]):
                    items.append(re.sub(r"^\d+\. ", "", lines[i]).strip())
                elif lines[i].startswith("   ") and items:
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            out.append(r"\begin{enumerate}")
            for it in items:
                out.append(r"\item %s" % inline(it))
            out.append(r"\end{enumerate}")
            continue

        # ---- bullet list ----
        if ln.startswith("- "):
            items = []
            while i < len(lines):
                if lines[i].startswith("- "):
                    items.append(lines[i][2:].strip())
                elif lines[i].startswith("  ") and lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            out.append(r"\begin{itemize}")
            for it in items:
                out.append(r"\item %s" % inline(it))
            out.append(r"\end{itemize}")
            continue

        # ---- paragraph ----
        if ln.strip():
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ", "> ")) \
                    and not re.match(r"^\d+\. ", lines[i]) and lines[i].strip() != r"\[":
                para.append(lines[i].strip()); i += 1
            # Unknown nonblank structural lines (for example, a standalone
            # pipe outside a Markdown table) must still consume input.
            if not para:
                para.append(ln.strip()); i += 1
            out.append(inline(" ".join(para)))
            out.append("")
            continue

        i += 1
    return "\n".join(out)

if __name__ == "__main__":
    src = open(sys.argv[1]).read()
    # keep only the chapters we want: from "## 1. Introduction" onward,
    # dropping "## 2. Related work" (replaced by the literature review chapter)
    # and "## References" (replaced by thebibliography).
    start = src.index("## 1. Introduction")
    body = src[start:]
    rel_a = body.index("## 2. Related work")
    rel_b = body.index("## 3. Methodology")
    body = body[:rel_a] + body[rel_b:]
    ref_a = body.index("## References")
    ref_b = body.index("## Appendix A.")
    body = body[:ref_a] + body[ref_b:]
    print(convert(body))
