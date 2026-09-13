from docx import Document

d = Document(r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx")
print(f"total tables: {len(d.tables)}")
for i, t in enumerate(d.tables):
    first = t.rows[0].cells[0].text.strip()[:28] if t.rows else "?"
    second = t.rows[0].cells[1].text.strip()[:24] if t.rows and len(t.columns) > 1 else ""
    print(f"table {i}: {len(t.rows)}r x {len(t.columns)}c | {first} | {second}")

# show the last 3 tables fully if they look like the new ones
for i in range(max(0, len(d.tables) - 3), len(d.tables)):
    t = d.tables[i]
    print(f"== table {i}")
    for r in t.rows[:6]:
        print("   " + " | ".join(c.text.strip()[:22] for c in r.cells))
