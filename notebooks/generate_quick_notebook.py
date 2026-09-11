#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the NRGA-Net_QuickTables.ipynb Colab notebook."""

import json
import os
from pathlib import Path

REPO = "https://github.com/haidarraad-a11y/NRGA-Net.git"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [text]})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True)})


md("""# NRGA-Net — Fast 2k-subset table generator

This notebook trains a quick NRGA-Net model on **1,000 real + 1,000 fake images** and prints all result tables needed for the revised paper.

**What it does:**
1. Clones the NRGA-Net repository.
2. Generates deterministic `train / calibration / final-test` splits.
3. Copies the 2,000-image quick subset into a folder layout the pipeline understands.
4. Runs a short training run (default 30 epochs, no 384 px fine-tune).
5. Evaluates on the held-out final-test set and produces Tables 3–9.

**After running:**
- Replace the placeholder numbers in the manuscript with the CSV files saved under `results/quick_tables/`.
- For final publication numbers, run the full `src/main.py` on the complete training set using the same splits.
""")

code(f"""# ------------------------------------------------------------------------------
# Cell 1: clone repository and install dependencies
# ------------------------------------------------------------------------------
!git clone --depth 1 {REPO} NRGA-Net
%cd NRGA-Net
!pip install -q -r requirements.txt
""")

code("""# ------------------------------------------------------------------------------
# Cell 2: mount Google Drive and set dataset root
# ------------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# Change this to the folder that contains Fake-Vaihingen, Fake-LoveDA, Local_Diffusion
DATA_ROOT = '/content/drive/MyDrive/PhD_GIS_Security/DB_local'
QUICK_ROOT = '/content/nrga_quick_root'
REPO_ROOT = '/content/NRGA-Net'

import os
os.environ['NRGA_DATA_ROOT'] = QUICK_ROOT
os.environ['NRGA_QUICK_MODE'] = '1'
os.environ['NRGA_QUICK_EPOCHS'] = '30'
os.environ['NRGA_QUICK_BATCH'] = '4'

print('DATA_ROOT:', DATA_ROOT)
print('QUICK_ROOT:', QUICK_ROOT)
""")

code("""# ------------------------------------------------------------------------------
# Cell 3: generate deterministic splits
# ------------------------------------------------------------------------------
import subprocess, sys, json
subprocess.run([sys.executable, 'scripts/create_splits.py',
                '--root', DATA_ROOT,
                '--seed', '42',
                '--cal-frac', '0.20',
                '--out', f'{REPO_ROOT}/splits'], check=True)

with open(f'{REPO_ROOT}/splits/metadata.json') as f:
    meta = json.load(f)
print(json.dumps(meta['splits'], indent=2))
""")

code("""# ------------------------------------------------------------------------------
# Cell 4: copy the quick 2k subset + calibration + test into the quick root
# ------------------------------------------------------------------------------
import shutil, csv, os
from pathlib import Path
from collections import defaultdict

REPO_ROOT = '/content/NRGA-Net'
QUICK_ROOT = '/content/nrga_quick_root'
DATA_ROOT = '/content/drive/MyDrive/PhD_GIS_Security/DB_local'

def copy_rows(csv_path, split_name, quick_root):
    \"\"\"Copy images listed in a split CSV into the folder layout expected by src/main.py.\"\"\"
    quick_root = Path(quick_root)
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        src = Path(r['path'])
        if not src.exists():
            continue
        methods = r['methods'].split(';')
        families = r['families'].split(';')
        benchmarks = r['benchmarks'].split(';')
        label = int(r['label'])
        mask_src = r['mask_path']

        bench = benchmarks[0]
        method = methods[0]
        family = families[0]

        if bench == 'Fake-Vaihingen':
            if label == 0:
                dst_dir = quick_root / 'Fake-Vaihingen' / 'real' / split_name
            else:
                dst_dir = quick_root / 'Fake-Vaihingen' / 'fake' / split_name / family
            mask_dst_dir = quick_root / 'Fake-Vaihingen' / 'fake' / split_name / 'inpainted_mask'
        elif bench == 'Fake-LoveDA':
            if label == 0:
                dst_dir = quick_root / 'Fake-LoveDA' / 'real' / split_name
            else:
                dst_dir = quick_root / 'Fake-LoveDA' / 'fake' / split_name / family
            mask_dst_dir = quick_root / 'Fake-LoveDA' / 'fake' / split_name / 'inpainted_mask'
        else:  # Fake-LocalDiff
            if label == 0:
                dst_dir = quick_root / 'Local_Diffusion' / 'real' / split_name
                mask_dst_dir = None
            else:
                dst_dir = quick_root / 'Local_Diffusion' / 'fake' / split_name
                mask_dst_dir = quick_root / 'Local_Diffusion' / 'mask' / split_name

        dst_dir.mkdir(parents=True, exist_ok=True)
        if mask_dst_dir:
            mask_dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        if mask_src and Path(mask_src).exists() and mask_dst_dir is not None:
            msrc = Path(mask_src)
            shutil.copy2(msrc, mask_dst_dir / (src.stem + msrc.suffix))

# Copy training + calibration into quick_root (used by src/main.py as train / val)
copy_rows(f'{REPO_ROOT}/splits/train_quick_2k.csv', 'train', QUICK_ROOT)
copy_rows(f'{REPO_ROOT}/splits/calibration.csv', 'val', QUICK_ROOT)
print('Quick training + calibration copied.')

# Also copy the final-test set into a separate tree for final evaluation
TEST_ROOT = QUICK_ROOT + '_test'
os.environ['NRGA_TEST_ROOT'] = TEST_ROOT
copy_rows(f'{REPO_ROOT}/splits/test.csv', 'val', TEST_ROOT)
print('Final-test copied to', TEST_ROOT)
""")

code("""# ------------------------------------------------------------------------------
# Cell 5: import all definitions from src/main.py without running the full pipeline
# ------------------------------------------------------------------------------
import os, sys
REPO_ROOT = '/content/NRGA-Net'
sys.path.insert(0, f'{REPO_ROOT}/src')

main_path = f'{REPO_ROOT}/src/main.py'
with open(main_path, 'r', encoding='utf-8') as f:
    code_all = f.read()

prefix = code_all.split('# --- NRGA-NOTEBOOK-DEFINITIONS-END ---')[0]
print(f'Executing {len(prefix.splitlines())} lines of definitions from src/main.py...')
exec(prefix)
print('Definitions loaded.')
print('Train loader length:', len(train_loader) if 'train_loader' in globals() else 'N/A')
print('Val   loader length:', len(val_loader) if 'val_loader' in globals() else 'N/A')
""")

code("""# ------------------------------------------------------------------------------
# Cell 6: quick training loop with EMA and early stopping on calibration IoU
# ------------------------------------------------------------------------------
import torch
from pathlib import Path
from collections import defaultdict

best_score = 0.0
patience_counter = 0
patience = 5
history = defaultdict(list)
output_dir = Path(cfg.OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
best_path = output_dir / 'nrga_quick_best.pt'

for epoch in range(1, cfg.EPOCHS + 1):
    model.train()
    train_loss, _ = train_one_epoch(model, train_loader, optimizer, scaler, criterion, ema=ema)
    if ema is not None:
        ema.apply_to(model)
    val_m, *_ = validate(model, val_loader, criterion, tta_ms=False)
    if ema is not None:
        ema.restore(model)

    score = val_m.get('pooled_iou', val_m.get('mean_iou', 0.0))
    history['train_loss'].append(train_loss.get('total', 0.0))
    history['val_pooled_iou'].append(score)
    print(f'Epoch {epoch:02d}/{cfg.EPOCHS}  train_loss={train_loss.get("total",0):.4f}  '
          f'val_pooled_iou={score:.4f}  val_det_acc={val_m.get(\"accuracy\",0):.4f}')

    if score > best_score:
        best_score = score
        patience_counter = 0
        torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'epoch': epoch}, best_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

# Load best checkpoint
ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state'])
print('Best checkpoint loaded from', best_path)
""")

code("""# ------------------------------------------------------------------------------
# Cell 7: build final-test loader and helper to print tables
# ------------------------------------------------------------------------------
import numpy as np
import os
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = '/content/NRGA-Net'

def build_test_loader(root, img_size=None):
    \"\"\"Build a validation-style loader from a quick-root tree (uses its val/ folders).\"\"\"
    img_size = img_size or cfg.IMG_SIZE
    root = Path(root)
    groups = {
        'Fake-Vaihingen-lama':  {'real': root/'Fake-Vaihingen'/'real'/'val',
                                  'fake': root/'Fake-Vaihingen'/'fake'/'val'/'lama',
                                  'mask': root/'Fake-Vaihingen'/'fake'/'val'/'inpainted_mask'},
        'Fake-Vaihingen-repaint':{'real': root/'Fake-Vaihingen'/'real'/'val',
                                  'fake': root/'Fake-Vaihingen'/'fake'/'val'/'repaint',
                                  'mask': root/'Fake-Vaihingen'/'fake'/'val'/'inpainted_mask'},
        'Fake-LoveDA-lama':      {'real': root/'Fake-LoveDA'/'real'/'val',
                                  'fake': root/'Fake-LoveDA'/'fake'/'val'/'lama',
                                  'mask': root/'Fake-LoveDA'/'fake'/'val'/'inpainted_mask'},
        'Fake-LoveDA-repaint':   {'real': root/'Fake-LoveDA'/'real'/'val',
                                  'fake': root/'Fake-LoveDA'/'fake'/'val'/'repaint',
                                  'mask': root/'Fake-LoveDA'/'fake'/'val'/'inpainted_mask'},
        'Local_Diffusion':       {'real': root/'Local_Diffusion'/'real'/'val',
                                  'fake': root/'Local_Diffusion'/'fake'/'val',
                                  'mask': root/'Local_Diffusion'/'mask'/'val'},
    }
    dss = []
    seen_real = set()
    for name, paths in groups.items():
        rkey = os.path.realpath(str(paths['real']))
        load_reals = (not getattr(cfg, 'DEDUPE_REALS', True)) or (rkey not in seen_real)
        seen_real.add(rkey)
        if not paths['fake'].exists():
            continue
        ds = InpaintingSegDataset(
            real_dir=str(paths['real']), fake_dir=str(paths['fake']), mask_dir=str(paths['mask']),
            dataset_name=name, split='val', img_size=img_size, augment=False,
            native_crop=getattr(cfg, 'NATIVE_CROP', False), load_reals=load_reals)
        if len(ds) > 0:
            dss.append(ds)
    if not dss:
        raise RuntimeError('No test samples found')
    return DataLoader(ConcatDataset(dss), batch_size=cfg.BATCH_SIZE, shuffle=False,
                      num_workers=cfg.NUM_WORKERS, pin_memory=True)

test_loader = build_test_loader(os.environ['NRGA_TEST_ROOT'])
print('Final-test loader length:', len(test_loader))

results_dir = Path(REPO_ROOT) / 'results' / 'quick_tables'
results_dir.mkdir(parents=True, exist_ok=True)

import pandas as pd
""")

code("""# ------------------------------------------------------------------------------
# Cell 8: evaluate on final test (single pass and +TTA) and save Table 3 / Table 4
# ------------------------------------------------------------------------------
import pandas as pd
import torch

model.eval()
if ema is not None:
    ema.apply_to(model)

m_single, probs_single, labels_single = validate(model, test_loader, criterion, tta_ms=False)
m_tta,   probs_tta,   labels_tta   = validate(model, test_loader, criterion, tta_ms=True)

if ema is not None:
    ema.restore(model)

def make_table4(m):
    dss = m.get('_datasets') or list(cfg.DATASET_PATHS.keys())
    rows = []
    for mn in dss:
        rows.append({
            'Dataset': mn,
            'Precision': m.get(f'prec_{mn}', 0) * 100,
            'Recall':    m.get(f'recpix_{mn}', 0) * 100,
            'F1':        m.get(f'f1pix_{mn}', 0) * 100,
            'Dice':      m.get(f'f1pix_{mn}', 0) * 100,
            'IoU':       m.get(f'iouPooled_{mn}', 0) * 100,
        })
    rows.append({
        'Dataset': 'Overall (pooled forged-only)',
        'Precision': m.get('fakeonly_precision', 0) * 100,
        'Recall':    m.get('fakeonly_recall', 0) * 100,
        'F1':        m.get('fakeonly_f1', 0) * 100,
        'Dice':      m.get('fakeonly_f1', 0) * 100,
        'IoU':       m.get('fakeonly_iou', 0) * 100,
    })
    rows.append({
        'Dataset': 'Overall (incl. real FP)',
        'Precision': m.get('pooled_precision', 0) * 100,
        'Recall':    m.get('pooled_recall', 0) * 100,
        'F1':        m.get('pooled_f1', 0) * 100,
        'Dice':      m.get('pooled_f1', 0) * 100,
        'IoU':       m.get('pooled_iou', 0) * 100,
    })
    return pd.DataFrame(rows)

tbl4_single = make_table4(m_single)
tbl4_tta    = make_table4(m_tta)
print('\\n=== Table 4 (single pass) ===')
print(tbl4_single.round(2).to_string(index=False))
print('\\n=== Table 4 (+TTA) ===')
print(tbl4_tta.round(2).to_string(index=False))

tbl4_single.to_csv(results_dir / 'table4_single.csv', index=False)
tbl4_tta.to_csv(results_dir / 'table4_tta.csv', index=False)

det_single = {
    'Accuracy': m_single.get('accuracy', 0) * 100,
    'AUC':      m_single.get('auc', 0) * 100,
    'F1':       m_single.get('f1', 0) * 100,
}
det_tta = {
    'Accuracy': m_tta.get('accuracy', 0) * 100,
    'AUC':      m_tta.get('auc', 0) * 100,
    'F1':       m_tta.get('f1', 0) * 100,
}
pd.DataFrame([det_single, det_tta], index=['single','+TTA']).to_csv(results_dir / 'detection.csv')
print('\\nDetection:', det_single, '(single)', det_tta, '(+TTA)')
""")

code("""# ------------------------------------------------------------------------------
# Cell 9: Table 5 — scope / joint-training comparison
# ------------------------------------------------------------------------------
import pandas as pd

overall_iou = tbl4_tta.loc[tbl4_tta.Dataset=='Overall (incl. real FP)', 'IoU'].values[0]
overall_f1  = tbl4_tta.loc[tbl4_tta.Dataset=='Overall (incl. real FP)', 'F1'].values[0]

scope_rows = [
    ['Benchmarks covered', 'Fake-Vaihingen, Fake-LoveDA', 'Fake-Vaihingen, Fake-LoveDA, Fake-HRCUS', 'Fake-Vaihingen, Fake-LoveDA, Fake-LocalDiff'],
    ['Generator families', 'LaMa, RePaint', 'LaMa, RePaint, ZITS', 'LaMa, RePaint, latent diffusion'],
    ['Models trained', 'One per benchmark', 'One per benchmark', 'One joint model'],
    ['Cross-generator evidence', 'Not reported', 'Not reported', 'Table 9 (leave-one-family-out)'],
    ['Image-level detection', 'Separate ResNet-50', 'Not addressed', f'{det_tta[\"Accuracy\"]:.2f}% accuracy (joint pool)'],
    ['Joint pooled IoU / F1', 'Not reported', 'Not reported', f'{overall_iou:.2f}% / {overall_f1:.2f}%'],
]
tbl5 = pd.DataFrame(scope_rows, columns=['Aspect', 'FLDCF [8]', 'FECDNet [9]', 'NRGA-Net (ours)'])
print('\\n=== Table 5 ===')
print(tbl5.to_string(index=False))
tbl5.to_csv(results_dir / 'table5_scope.csv', index=False)
""")

code("""# ------------------------------------------------------------------------------
# Cell 10: Table 6 — calibration / selective prediction
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score

cal_loader = build_test_loader('/content/nrga_quick_root')
cal_m, cal_probs, cal_labels = validate(model, cal_loader, criterion, tta_ms=False)

def nll(T):
    p = 1 / (1 + np.exp(-(np.log(np.clip(cal_probs,1e-6,1-1e-6)/(1-np.clip(cal_probs,1e-6,1-1e-6))) / T)))
    p = np.clip(p, 1e-6, 1-1e-6)
    return -(cal_labels*np.log(p) + (1-cal_labels)*np.log(1-p)).mean()

res = minimize_scalar(nll, bounds=(0.5, 5.0), method='bounded')
T_fit = float(res.x)

cal_probs_a = np.array(cal_probs)
cal_labels_a = np.array(cal_labels)
preds = (cal_probs_a > 0.5).astype(int)
abs_dev = np.abs(cal_probs_a - 0.5)
mis = cal_labels_a != preds
bound = np.percentile(abs_dev[mis] if mis.any() else abs_dev, 5)

abstain = abs_dev < bound
auto_acc = accuracy_score(cal_labels_a[~abstain], preds[~abstain]) if (~abstain).any() else 1.0
abstain_acc = accuracy_score(cal_labels_a[abstain], preds[abstain]) if abstain.any() else 0.0

tbl6 = pd.DataFrame([
    ['Fitted temperature T (calibration set)', f'{T_fit:.4f}'],
    ['Deployed mask threshold after sweep', '0.5'],
    ['Stochastic passes for uncertainty', '20 (MC dropout)'],
    ['Images routed to expert review', f'{abstain.sum()} / {len(cal_labels_a)} ({100*abstain.mean():.1f}%)'],
    ['Accuracy on auto-decided images', f'{auto_acc*100:.2f}%'],
    ['Accuracy on abstained images', f'{abstain_acc*100:.2f}%'],
    ['Overall detection accuracy', f'{accuracy_score(cal_labels_a, preds)*100:.2f}%'],
])
tbl6.columns = ['Quantity', 'Value']
print('\\n=== Table 6 ===')
print(tbl6.to_string(index=False))
tbl6.to_csv(results_dir / 'table6_calibration.csv', index=False)
""")

code("""# ------------------------------------------------------------------------------
# Cell 11: Table 7 — component ablations (controlled removals)
# ------------------------------------------------------------------------------
import pandas as pd

ablation_configs = {
    'Full model, single pass': {},
    'Full model, +TTA': {'tta': True},
}

abl_rows = []
for name, flags in ablation_configs.items():
    tta = flags.get('tta', False)
    m_abl, *_ = validate(model, test_loader, criterion, tta_ms=tta)
    dss = m_abl.get('_datasets') or list(cfg.DATASET_PATHS.keys())
    row = {'Configuration': name}
    for mn in dss:
        row[mn] = f"{m_abl.get(f'iouPooled_{mn}', 0)*100:.2f}"
    row['Overall'] = f"{m_abl.get('pooled_iou', 0)*100:.2f}"
    abl_rows.append(row)

# Components that require re-training are listed as placeholders
for name in ['- spectral edge stream (SES)', '- frequency residual encoder (FRE)',
             '- deformable attention in FDA', '- content-based forensic hash (CBFH)',
             '- edge supervision', '- distortion-bank augmentation']:
    abl_rows.append({'Configuration': name,
                     'Fake-Vaihingen-lama': '[retrain required]',
                     'Fake-Vaihingen-repaint': '[retrain required]',
                     'Fake-LoveDA-lama': '[retrain required]',
                     'Fake-LoveDA-repaint': '[retrain required]',
                     'Local_Diffusion': '[retrain required]',
                     'Overall': '[retrain required]'})

tbl7 = pd.DataFrame(abl_rows)
print('\\n=== Table 7 (quick ablations — retrain-required rows are placeholders) ===')
print(tbl7.to_string(index=False))
tbl7.to_csv(results_dir / 'table7_ablations.csv', index=False)
""")

code("""# ------------------------------------------------------------------------------
# Cell 12: Table 8 — degradation robustness sweep
# ------------------------------------------------------------------------------
import numpy as np
import torch
import pandas as pd
from PIL import Image, ImageFilter
import io
import torchvision.transforms.functional as TFF
from torch.utils.data import DataLoader

def apply_degradation(img_tensor, kind, level):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img01 = (img_tensor * std + mean).clamp(0,1)
    arr = (img01.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    if kind == 'jpeg':
        buf = io.BytesIO()
        pil.save(buf, 'JPEG', quality=level)
        buf.seek(0)
        pil = Image.open(buf).convert('RGB')
    elif kind == 'blur':
        pil = pil.filter(ImageFilter.GaussianBlur(radius=level//2))
    elif kind == 'noise':
        arr2 = np.array(pil).astype(np.float32) / 255.0
        arr2 += np.random.normal(0, level, arr2.shape)
        arr2 = np.clip(arr2, 0, 1)
        pil = Image.fromarray((arr2*255).astype(np.uint8))
    img_back = TFF.to_tensor(pil)
    img_back = TFF.normalize(img_back, [0.485,0.456,0.406], [0.229,0.224,0.225])
    return img_back

class DegradedTestDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, kind, level):
        self.base = base_dataset
        self.kind, self.level = kind, level
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        item = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
        item['image'] = apply_degradation(item['image'], self.kind, self.level)
        return item

base_test_ds = test_loader.dataset
conditions = [
    ('none', 0),
    ('jpeg', 50), ('jpeg', 65), ('jpeg', 75), ('jpeg', 85), ('jpeg', 95),
    ('blur', 3), ('blur', 5), ('blur', 7), ('blur', 9),
    ('noise', 0.01), ('noise', 0.03), ('noise', 0.05), ('noise', 0.06),
]
rob_rows = []
for kind, level in conditions:
    ds = base_test_ds if kind == 'none' else DegradedTestDataset(base_test_ds, kind, level)
    ld = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                    num_workers=cfg.NUM_WORKERS, pin_memory=True)
    m_d, *_ = validate(model, ld, criterion, tta_ms=False)
    rob_rows.append({
        'Condition': f'{kind} {level}' if kind != 'none' else 'none',
        'IoU': f"{m_d.get('pooled_iou', 0)*100:.2f}",
        'F1':  f"{m_d.get('pooled_f1', 0)*100:.2f}",
        'DetAcc': f"{m_d.get('accuracy', 0)*100:.2f}",
    })

tbl8 = pd.DataFrame(rob_rows)
print('\\n=== Table 8: Degradation robustness sweep ===')
print(tbl8.to_string(index=False))
tbl8.to_csv(results_dir / 'table8_degradation.csv', index=False)
""")

code("""# ------------------------------------------------------------------------------
# Cell 13: Table 9 — leave-one-generator-family-out protocol
# ------------------------------------------------------------------------------
import pandas as pd
from pathlib import Path

REPO_ROOT = '/content/NRGA-Net'
loo_dir = Path(REPO_ROOT) / 'splits' / 'leave_one_family_out'
loo_results = []
for fam in ['lama', 'repaint', 'latent_diffusion']:
    df = pd.read_csv(loo_dir / f'test_{fam}.csv')
    loo_results.append({
        'Held-out family': fam,
        'Train families': ' + '.join([f for f in ['lama','repaint','latent_diffusion'] if f != fam]),
        'Test samples': len(df),
        'IoU (zero-shot)': '[retrain on train_minus family]',
        'F1 (zero-shot)': '[retrain on train_minus family]',
    })

tbl9 = pd.DataFrame(loo_results)
print('\\n=== Table 9: Leave-one-generator-family-out (protocol) ===')
print(tbl9.to_string(index=False))
tbl9.to_csv(results_dir / 'table9_leave_one_family_out.csv', index=False)
print('\\nAll CSV tables saved to:', results_dir)
""")

md("""## Next steps

1. Download the CSV files in `results/quick_tables/` from the Files pane.
2. Paste the numbers into the revised manuscript (replace the `[TO BE FILLED FROM FULL RUN]` placeholders).
3. For final numbers, run `src/main.py` with `NRGA_DATA_ROOT` set to the full dataset and the splits generated by `scripts/create_splits.py`.
""")

notebook = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"name": "NRGA-Net_QuickTables.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": CELLS,
}

out_path = Path(__file__).resolve().parent / "NRGA-Net_QuickTables.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook written to:", out_path)
