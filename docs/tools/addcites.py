import re, sys
# first-mention citations for external works named in the body chapters
M = [("Freeze-Align","maniparambil2025freezealign"),("MobileCLIP2","faghri2025mobileclip2"),
     ("OpenCLIP","cherti2023scaling"),("SigLIP2","tschannen2025siglip2"),
     ("DINOv3","simeoni2025dinov3"),("DINOv2","oquab2023dinov2"),
     ("SAIL","zhang2025sail"),("HALoRA","zhang2026halora"),("FALCON","kim2026falcon"),
     ("XBM","wang2020xbm"),("ShareLock","ruthardt2026sharelock"),
     ("STRUCTURE","groger2025structure"),("SOTAlign","roschmann2026sotalign"),
     ("SAD-TM","xie2026sadtm"),("PCME","chun2021pcme"),("Dyna-ViT","rubab2026dynavit"),
     ("Flickr30k","young2014flickr30k"),("COCO","lin2014coco"),("LoRA","hu2021lora")]
txt = open(sys.argv[1]).read()
lines = txt.split("\n")
done = set()
for i, ln in enumerate(lines):
    if ln.startswith(("\\chapter", "\\section", "\\begin", "\\end", "\\item", "\\texttt")):
        continue
    for name, key in M:
        if key in done:
            continue
        # word-boundary match not already followed by a cite, not inside \texttt{}
        m = re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])(?!~\\cite)", ln)
        if m and "\\texttt{" not in ln[:m.start()].split("}")[-1]:
            lines[i] = ln[:m.end()] + "~\\cite{%s}" % key + ln[m.end():]
            ln = lines[i]
            done.add(key)
open(sys.argv[1], "w").write("\n".join(lines))
print("cited:", len(done), "of", len(M))
print("missing:", [k for _, k in M if k not in done])
