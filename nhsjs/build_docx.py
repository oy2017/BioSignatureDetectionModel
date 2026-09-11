"""Build the two NHSJS submission files from manuscript.md.

  <title>.docx            Standard citations (superscript numbers before punctuation,
                          no space), blind: no authors, no acknowledgments.
  <title> - online.docx   Online citations: the complete reference in double
                          parentheses at every use, a space before "((", and a
                          superscript comma between adjacent citations; author
                          block from authors.md and acknowledgments from
                          acknowledgments.md.

Both are poured into the official templates in templates/ so the journal's
paragraph styles (NHSJS Title / Section / Subsection, Normal = Times New Roman
12 pt single-spaced) apply. Citations in the source are [n] or [n, m] and are
renumbered by first mention. Figures are inserted at 6.0 in from the paths in
the ![Figure N](path) lines; captions are the following **Figure N | ...** line.
Markdown tables become Word tables; the **Table N | ...** caption precedes them.

Run: ~/tfenv/bin/python build_docx.py
"""
import re
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

FIG_WIDTH_IN = 5.0

HERE = Path(__file__).resolve().parent
MD = HERE / "manuscript.md"
TPL_STD = HERE / "templates" / "NHSJS-Manuscript-Template-Standard-Citations.docx"
TPL_ONL = HERE / "templates" / "NHSJS-Manuscript-Template-Online-Citations.docx"
CITE_RE = re.compile(r"\s*\[(\d+(?:,\s*\d+)*)\]")
SPAN_RE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*|" + CITE_RE.pattern)


def load():
    raw = MD.read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S).strip()
    body, refs_txt = raw.split("## References", 1)
    refs = {int(m.group(1)): m.group(2).strip() for m in re.finditer(r"^(\d+)\.\s+(.*)$", refs_txt, re.M)}
    return body, refs


def mapping_by_first_mention(body):
    order = []
    for m in CITE_RE.finditer(body):
        for n in [int(x) for x in m.group(1).split(",")]:
            if n not in order:
                order.append(n)
    return {old: i + 1 for i, old in enumerate(order)}


def plain_ref(text):
    """Reference text without markdown italics markers (for the online citation)."""
    return re.sub(r"\*(.+?)\*", r"\1", text)


def add_text(par, text, mapping, refs, online, bold=False, italic=False):
    """Add text with **bold**, *italic* spans and citations rendered per mode."""
    pos = 0
    for m in SPAN_RE.finditer(text):
        if m.start() > pos:
            r = par.add_run(text[pos:m.start()]); r.bold = bold or None; r.italic = italic or None
        if m.group(1) is not None:
            add_text(par, m.group(1), mapping, refs, online, bold=True, italic=italic)
        elif m.group(2) is not None:
            add_text(par, m.group(2), mapping, refs, online, bold=bold, italic=True)
        else:
            nums = sorted(mapping[int(x)] for x in m.group(3).split(","))
            if online:
                inv = {v: k for k, v in mapping.items()}
                for j, n in enumerate(nums):
                    if j == 0:
                        par.add_run(" ")
                    else:
                        rc = par.add_run(","); rc.font.superscript = True
                    r = par.add_run(f"(({plain_ref(refs[inv[n]])}))"); r.bold = bold or None
            else:
                r = par.add_run(",".join(str(n) for n in nums)); r.font.superscript = True
        pos = m.end()
    if pos < len(text):
        r = par.add_run(text[pos:]); r.bold = bold or None; r.italic = italic or None


def md_table(doc, lines, mapping, refs, online):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            p = t.cell(i, j).paragraphs[0]
            add_text(p, cell, mapping, refs, online, bold=(i == 0))
            for r in p.runs:
                r.font.size = Pt(9)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(0)
    sp.add_run().font.size = Pt(6)


def build(online, out_path):
    body, refs = load()
    mapping = mapping_by_first_mention(body)
    doc = Document(str(TPL_ONL if online else TPL_STD))
    el_body = doc.element.body
    for el in list(el_body):
        if not el.tag.endswith("}sectPr"):
            el_body.remove(el)
    lines = body.splitlines()
    i = 0
    pending_fig = None
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln:
            i += 1; continue
        if ln.startswith("# "):
            doc.add_paragraph(ln[2:], style="NHSJS Title")
            # author block
            if online:
                for para in (HERE / "authors.md").read_text(encoding="utf-8").strip().split("\n\n"):
                    p = doc.add_paragraph()
                    for k, sub in enumerate(para.split("\n")):
                        if k:
                            p.add_run().add_break()
                        # superscript digits / asterisks that follow a name or precede an affiliation
                        for m in re.finditer(r"([¹²³*]+)|([^¹²³*]+)", sub):
                            if m.group(1):
                                rr = p.add_run(m.group(1).replace("¹", "1").replace("²", "2").replace("³", "3"))
                                rr.font.superscript = True
                            else:
                                p.add_run(m.group(2))
            else:
                p = doc.add_paragraph()
                add_text(p, "**Authors and affiliations:** omitted for blind review.", mapping, refs, online)
        elif ln.startswith("**Authors and affiliations:**"):
            pass
        elif ln.startswith("## Acknowledgments"):
            if online:
                doc.add_paragraph("Acknowledgments", style="NHSJS Section")
                for para in (HERE / "acknowledgments.md").read_text(encoding="utf-8").strip().split("\n\n"):
                    p = doc.add_paragraph(); add_text(p, para, mapping, refs, online)
            # skip the placeholder line(s) under the heading in either mode
            while i + 1 < len(lines) and not lines[i + 1].startswith("## "):
                i += 1
        elif ln.startswith("## "):
            doc.add_paragraph(ln[3:], style="NHSJS Section")
        elif ln.startswith("### "):
            doc.add_paragraph(ln[4:], style="NHSJS Subsection")
        elif ln.startswith("|"):
            tbl = [ln]
            while i + 1 < len(lines) and lines[i + 1].startswith("|"):
                i += 1; tbl.append(lines[i])
            md_table(doc, tbl, mapping, refs, online)
        elif ln.startswith("!["):
            pending_fig = re.search(r"\((.+?)\)", ln).group(1)
        elif re.match(r"^\*\*Figure \d+ \|", ln):
            if pending_fig:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str((HERE / pending_fig).resolve()), width=Inches(FIG_WIDTH_IN))
                pending_fig = None
            p = doc.add_paragraph(); add_text(p, ln, mapping, refs, online)
        else:
            p = doc.add_paragraph(); add_text(p, ln, mapping, refs, online)
        i += 1
    # references, renumbered
    doc.add_paragraph("References", style="NHSJS Section")
    inv = {v: k for k, v in mapping.items()}
    for new in range(1, len(inv) + 1):
        p = doc.add_paragraph(); add_text(p, f"{new}. {refs[inv[new]]}", mapping, refs, online)
    doc.save(str(out_path))
    return mapping, len(re.sub(r"\|.*\|", "", body).split())


def main():
    body, _ = load()
    title = re.search(r"^# (.+)$", body, re.M).group(1).strip()
    std = HERE / f"{title}.docx"
    onl = HERE / f"{title} - online.docx"
    mapping, words = build(False, std)
    build(True, onl)
    # every figure the manuscript embeds, plus the supporting ones it points at;
    # anything not referenced anywhere is left out rather than shipped orphaned
    SUPP = {"fig3_margin_amplitude.png", "fig5_threshold_transfer.png"}
    embedded = {Path(p).name for p in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body)}
    figs = sorted(p for p in (HERE.parent / "v2" / "results" / "figures").glob("fig*.png")
                  if p.name in embedded | SUPP)
    with zipfile.ZipFile(HERE / "supplementary_figures.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in figs:
            z.write(f, f.name)
    n_fig = len(re.findall(r"^\*\*Figure \d+ \|", body, re.M)); n_tab = len(re.findall(r"^\*\*Table \d+ \|", body, re.M))
    # figures cost what they actually occupy, not a flat half page, plus an allowance
    # for the gap each one leaves when it will not fit the space left on a page
    from PIL import Image
    fig_in = 0.0
    for m in re.finditer(r"^!\[[^\]]*\]\(([^)]+)\)", body, re.M):
        f = (HERE / m.group(1)).resolve()
        if f.exists():
            w, h = Image.open(f).size
            fig_in += FIG_WIDTH_IN * h / w
    text_h = 9.0
    est = (words / 600 + fig_in / text_h + n_fig * 0.18 + n_tab * 0.45
           + len(mapping) * 3 / 46 + 0.7)
    print(f"wrote:\n  {std.name}\n  {onl.name}\n  supplementary_figures.zip ({len(figs)} files)")
    print(f"citations: {len(mapping)}; body words (excl. tables): {words}; figures {n_fig}, tables {n_tab}")
    print(f"estimated pages at 12 pt single-spaced: {est:.1f} (confirm in Word; limit 20)")


if __name__ == "__main__":
    main()
