from docx import Document

d = Document(r"D:\Dataset\pp3\NRGA-Net_paper.docx")
for i in (1, 2, 3, 5, 6):
    t = d.tables[i]
    print("== table", i)
    for r in t.rows[:12]:
        print("   " + " | ".join(c.text.strip()[:38] for c in r.cells))
