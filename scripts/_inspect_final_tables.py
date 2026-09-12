from docx import Document

d = Document(r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx")
for i in (5, 6, 7):
    t = d.tables[i]
    print(f"== table {i} ({len(t.rows)}r x {len(t.columns)}c)")
    for r in t.rows:
        print("   " + " | ".join(c.text.strip()[:30] for c in r.cells))
