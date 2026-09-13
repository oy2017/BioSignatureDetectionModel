"""Turn note_src.md (citations as {key} or {key1,key2}) into manuscript.md: numbers references by first
citation in the body, renders grouped citations as (1, 3-5), and writes the numbered reference list in place
of the <!-- references --> marker. Fails on unknown keys and reports keys in REFS that were never cited.
Usage: python render_note.py"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from refs_note import REFS

src = open(os.path.join(HERE, "note_src.md"), encoding="utf-8").read()
order = []
for m in re.finditer(r"\{([a-z0-9_,\s]+)\}", src):
    for k in [x.strip() for x in m.group(1).split(",")]:
        if k not in REFS:
            sys.exit(f"unknown reference key: {k}")
        if k not in order:
            order.append(k)
num = {k: i + 1 for i, k in enumerate(order)}


def fmt(keys):
    ns = sorted({num[k] for k in keys}); out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(f"{ns[i]}-{ns[j]}" if j - i >= 2 else ", ".join(str(n) for n in ns[i:j + 1]))
        i = j + 1
    return "(" + ", ".join(out) + ")"


body = re.sub(r"\{([a-z0-9_,\s]+)\}", lambda m: fmt([x.strip() for x in m.group(1).split(",")]), src)
refs = "\n".join(f"{num[k]}. {REFS[k]}" for k in order)
body = body.replace("<!-- references -->", refs)
open(os.path.join(HERE, "manuscript.md"), "w", encoding="utf-8").write(body)
unused = [k for k in REFS if k not in num]
print(f"{len(order)} references cited; unused keys: {unused if unused else 'none'}")
