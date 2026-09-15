# -*- coding: utf-8 -*-
"""Verify IJIES formatting of the revised manuscript."""
import re
from docx import Document
from docx.text.paragraph import Paragraph

REV = r"D:\Dataset\pp3\NRGA-Net_paper_revised.docx"
LET = r"D:\Dataset\pp3\NRGA-Net_response_letter.docx"

ok = True
def check(name, cond, detail=''):
    global ok
    print(('PASS' if cond else 'FAIL'), '|', name, ('| ' + str(detail) if detail else ''))
    if not cond:
        ok = False

r = Document(REV)

# 1. table numbering: captions end with "." after the number
caps = [p.text.strip() for p in r.paragraphs if re.match(r'^Table\s+\d+\.', p.text.strip())]
nums = sorted(int(re.match(r'^Table\s+(\d+)', c).group(1)) for c in caps)
check('table captions sequential 1..10', nums == list(range(1, 11)), str(nums))
check('no Table 2a remains', not any('2a' in c for c in caps))

# 2. in-text refs
refs = set()
for p in r.paragraphs:
    for m in re.finditer(r'Table\s+(\d+)(?![0-9a-zA-Z])', p.text):
        refs.add(int(m.group(1)))
for t in r.tables:
    for row in t.rows:
        for cell in row.cells:
            for m in re.finditer(r'Table\s+(\d+)(?![0-9a-zA-Z])', cell.text):
                refs.add(int(m.group(1)))
check('in-text table refs within 1-10', all(1 <= n <= 10 for n in refs), str(sorted(refs)))

# 3. figure captions now "Fig. N" (captions contain ". " after label; prose excluded by check 4)
figcaps = [p.text.strip()[:30] for p in r.paragraphs if re.match(r'^Figure\.', p.text.strip())]
check('no "Figure." captions remain', len(figcaps) == 0)
n_fig = len([p for p in r.paragraphs if re.match(r'^Fig\.\s*\d+\s+[A-Z]', p.text.strip())
             and not p.text.strip().startswith('Fig. 13 shows')])
check('figure captions use Fig. N', n_fig == 15, f'{n_fig} captions')

# 4. first section single column
first_sect = None
for child in r.element.body:
    if child.tag.endswith('}p') and child.pPr is not None and child.pPr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr') is not None:
        first_sect = child
        break
cols = first_sect.pPr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr').find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}cols')
check('first section single column', cols is None or cols.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}num') is None,
      dict(cols.attrib) if cols is not None else 'no cols elem')

# 5. equations: indent + italic + no "*"
eqs = [p for p in r.paragraphs if re.search(r'\(\d+\)\s*$', p.text) and '=' in p.text]
check('equation count', len(eqs) == 9, f'{len(eqs)}')
star_eqs = [p.text[:60] for p in eqs if '*' in p.text]
check('no "*" in equations', not star_eqs, str(star_eqs))
bad_indent = [p.text[:40] for p in eqs if p.paragraph_format.left_indent is None or abs(p.paragraph_format.left_indent.cm - 0.5) > 0.01]
check('equations indented 5mm', not bad_indent, str(bad_indent))

# 6. headings bold/sizes
h_ok = True
for p in r.paragraphs:
    t = p.text.strip()
    if re.match(r'^\d+\.\s+\S', t) and len(t) < 60:
        if not all(rr.bold and rr.font.size and abs(rr.font.size.pt - 12.0) < 0.01 for rr in p.runs if rr.text.strip()):
            h_ok = False
            print('   heading issue:', t, [(rr.font.size, rr.bold) for rr in p.runs if rr.text.strip()])
check('first-order headings 12pt bold', h_ok)

# 7. title block sizes
title = r.paragraphs[0]
check('title 14pt bold', title.runs[0].font.size.pt == 14 and title.runs[0].bold)
abstract = [p for p in r.paragraphs if p.text.strip().startswith('Abstract:')]
check('abstract 10pt', abstract and abstract[0].runs[0].font.size.pt == 10)

# 8. references
ref_texts = {}
grab = False
for p in r.paragraphs:
    t = p.text.strip()
    if t == 'References':
        grab = True
        continue
    if grab and re.match(r'^\[\d+\]', t):
        ref_texts[t[:4]] = p
check('ref [2] has Vol/No/pp', 'Vol.53, No.4, pp.465-478' in ref_texts['[2] '].text.replace('  ', ' '), ref_texts['[2] '].text[-80:])
check('ref [13] e42', 'e42' in ref_texts['[13]'].text, ref_texts['[13] '].text[-60:] if '[13] ' in ref_texts else ref_texts['[13]'].text[-60:])
check('ref [10] Proc. of', 'Proc. of International Conference on Pattern Recognition' in ref_texts['[10]'].text)
straight = [k for k, p in ref_texts.items() if '"' in p.text]
check('no straight quotes in refs', not straight, str(straight))

# 9. letter: reviewer quotes untouched, responses renumbered
let = Document(LET)
com_t4 = [p.text for p in let.paragraphs if p.text.strip().startswith('Comment') and 'Table 4' in p.text]
check('letter keeps reviewer "Table 4" quotes', len(com_t4) >= 2, f'{len(com_t4)} quotes')
resp_2a = [p.text for p in let.paragraphs if p.text.strip().startswith('Response') and 'Table 2a' in p.text]
check('letter responses have no 2a', not resp_2a)
resp_t3 = [p.text for p in let.paragraphs if p.text.strip().startswith('Response') and 'Table 3' in p.text]
check('letter responses reference Table 3 (split)', len(resp_t3) >= 1)

# 10. table cell font sizes still 10pt
szs = set()
for t in r.tables:
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    if run.text.strip() and run.font.size:
                        szs.add(run.font.size.pt)
check('table fonts 10pt', szs <= {10.0}, str(szs))

print()
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
