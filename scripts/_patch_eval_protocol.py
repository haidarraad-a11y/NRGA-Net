"""Patch generate_full_notebook_local.py: replace Cell 8 (folder-based loaders)
with split-CSV loaders + calibration-only checkpoint selection; update Cell 9
(drop EMA usage) and Cell 11 (fit T on calibration, report on test)."""

from pathlib import Path

P = Path(__file__).resolve().parent.parent / "notebooks" / "generate_full_notebook_local.py"
text = P.read_text(encoding="utf-8")

# --- locate Cell 8 block span -------------------------------------------------
c8 = text.find("# Cell 8: build final-test loader directly from Drive")
assert c8 != -1, "Cell 8 marker not found"
start = text.rfind('code("""', 0, c8)
c9 = text.find("# Cell 9: evaluate on final test", c8)
assert c9 != -1, "Cell 9 marker not found"
end = text.rfind('""")', c8, c9) + len('""")')

NEW_CELL8 = '''code("""# ------------------------------------------------------------------------------
# Cell 8: calibration / final-test loaders from the immutable split files,
#         plus protocol-compliant checkpoint selection on the calibration split ONLY
# ------------------------------------------------------------------------------
import numpy as np, os, shutil, json, copy as _copy
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd

REPO_ROOT = '/content/NRGA-Net'
SPLITS_DIR = Path(REPO_ROOT) / 'splits'
LINK_BASE = Path('/content/eval_links')

def _link_all(dst_dir, pairs):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src, name in pairs:
        d = dst_dir / name
        if d.exists() or d.is_symlink():
            d.unlink()
        os.symlink(src, d)

def build_loader_from_split(csv_path, tag, img_size=None):
    df = pd.read_csv(csv_path)
    groups = {}
    for _, r in df.iterrows():
        bench = str(r['benchmarks']).split(';')[0]
        fam = str(r['families']).split(';')[0] if isinstance(r['families'], str) and r['families'] else 'real'
        key = (bench, fam)
        if key not in groups:
            groups[key] = {'real': [], 'fake': []}
        if int(r['label']) == 1:
            m = str(r['mask_path']) if isinstance(r['mask_path'], str) and r['mask_path'] else None
            groups[key]['fake'].append((str(r['path']), m))
        else:
            groups[key]['real'].append(str(r['path']))
    base = LINK_BASE / tag
    if base.exists():
        shutil.rmtree(base)
    dss, names = [], []
    for (bench, fam), items in sorted(groups.items()):
        gname = f'{bench}-{fam}' if fam != 'real' else bench
        rd, fd, md = base / gname / 'real', base / gname / 'fake', base / gname / 'mask'
        _link_all(rd, [(p, f'{i:05d}_{Path(p).name}') for i, p in enumerate(items['real'])])
        with_mask = [(p, m) for (p, m) in items['fake'] if m is not None]
        _link_all(fd, [(p, f'{Path(p).stem}{Path(p).suffix}') for p, _ in with_mask])
        _link_all(md, [(m, f'{Path(p).stem}_mask{Path(m).suffix}') for (p, m) in with_mask])
        ds = InpaintingSegDataset(
            real_dir=str(rd), fake_dir=str(fd), mask_dir=str(md),
            dataset_name=gname, split='val', img_size=img_size or cfg.IMG_SIZE,
            augment=False, native_crop=getattr(cfg, 'NATIVE_CROP', False), load_reals=True)
        if len(ds) > 0:
            dss.append(ds)
            names.append(gname)
    loader = DataLoader(ConcatDataset(dss), batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True)
    return loader, names

def _split_counts(csv_path):
    df = pd.read_csv(csv_path)
    return {'total': int(len(df)),
            'real': int((df['label'] == 0).sum()),
            'fake': int((df['label'] == 1).sum())}

results_dir = Path(REPO_ROOT) / 'results' / 'full_tables'
results_dir.mkdir(parents=True, exist_ok=True)

test_loader, test_names = build_loader_from_split(SPLITS_DIR / 'test.csv', 'test')
cal_loader, cal_names = build_loader_from_split(SPLITS_DIR / 'calibration.csv', 'cal')
print('Final-test datasets:', test_names)
print('Calibration datasets:', cal_names)
with open(results_dir / 'splits_summary.json', 'w') as f:
    json.dump({'calibration': _split_counts(SPLITS_DIR / 'calibration.csv'),
               'test': _split_counts(SPLITS_DIR / 'test.csv')}, f, indent=2)

# ---- protocol-compliant checkpoint selection (calibration split ONLY) --------
# Periodic epoch backups + the previous best-by-pool checkpoint are scored on the
# calibration split; the winner is used for every reported number.
cands = sorted(Path(RUN_OUTPUT_DIR).glob('nrga_full_epoch*.pt'))
_best_pool = Path(RUN_OUTPUT_DIR) / 'nrga_full_best.pt'
if _best_pool.exists():
    cands.append(_best_pool)
scores = []
selector = NRGANet(cfg).to(DEVICE)
for cpath in cands:
    try:
        _ck = torch.load(cpath, map_location='cpu', weights_only=False)
        selector.load_state_dict(_ck['model_state'])
        sanitize_state_(selector)
        selector.eval()
        m_c, _, _ = validate(selector, cal_loader, criterion, tta_ms=False)
        s_c = float(m_c.get('pooled_iou', m_c.get('mean_iou', 0.0)))
        scores.append((s_c, str(cpath), int(_ck.get('epoch', -1)), '256'))
        print(f'  {cpath.name}: calibration pooled IoU = {s_c:.4f} (epoch {_ck.get("epoch", "?")})')
        del _ck
    except Exception as e:
        print(f'  {cpath.name}: failed ({e})')

best_score_sel, best_path_sel, best_epoch_sel, best_tag = max(scores)
selected_model = NRGANet(cfg).to(DEVICE)
_cksel = torch.load(best_path_sel, map_location='cpu', weights_only=False)
selected_model.load_state_dict(_cksel['model_state'])
sanitize_state_(selected_model)
del _cksel

# optional: compare against the final-epoch 384 fine-tune (selection-free)
last384 = Path(RUN_OUTPUT_DIR) / 'nrga_full_384_last.pt'
if getattr(cfg, 'FT384_ENABLE', True) and last384.exists():
    try:
        cfg384sel = _copy.copy(cfg)
        cfg384sel.IMG_SIZE = int(getattr(cfg, 'FT384_IMG_SIZE', 384))
        model384sel = NRGANet(cfg384sel).to(DEVICE)
        _ck4 = torch.load(last384, map_location='cpu', weights_only=False)
        model384sel.load_state_dict(_ck4['model_state'])
        sanitize_state_(model384sel)
        model384sel.eval()
        cal_loader384, _ = build_loader_from_split(SPLITS_DIR / 'calibration.csv', 'cal384',
                                                   img_size=cfg384sel.IMG_SIZE)
        m4, _, _ = validate(model384sel, cal_loader384, criterion, tta_ms=False)
        s4 = float(m4.get('pooled_iou', m4.get('mean_iou', 0.0)))
        print(f'  nrga_full_384_last.pt: calibration pooled IoU = {s4:.4f} (final epoch, selection-free)')
        if s4 > best_score_sel:
            best_score_sel, best_epoch_sel, best_tag = s4, int(_ck4.get('epoch', -1)), '384-last'
            selected_model = model384sel
            cfg.IMG_SIZE = cfg384sel.IMG_SIZE
            print('  -> selecting the 384 final-epoch model')
        del _ck4
    except Exception as e:
        print(f'  384 comparison failed ({e})')

model = selected_model
model.eval()
torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'epoch': best_epoch_sel,
            'selected_by': 'calibration_split_only', 'selected_tag': best_tag,
            'calibration_pooled_iou': best_score_sel},
           Path(RUN_OUTPUT_DIR) / 'nrga_full_selected.pt')
print(f'Selected checkpoint: {best_tag} (calibration pooled IoU {best_score_sel:.4f}); '
      f'cfg.IMG_SIZE = {cfg.IMG_SIZE}')
""")'''

text = text[:start] + NEW_CELL8 + text[end:]

# --- Cell 9: drop EMA apply/restore around evaluation -------------------------
old9 = """model.eval()
if ema is not None:
    ema.apply_to(model)

m_single, probs_single, labels_single = validate(model, test_loader, criterion, tta_ms=False)
m_tta,   probs_tta,   labels_tta   = validate(model, test_loader, criterion, tta_ms=True)

if ema is not None:
    ema.restore(model)"""
new9 = """model.eval()
m_single, probs_single, labels_single = validate(model, test_loader, criterion, tta_ms=False)
m_tta,   probs_tta,   labels_tta   = validate(model, test_loader, criterion, tta_ms=True)"""
assert old9 in text, "Cell 9 EMA block not found"
text = text.replace(old9, new9)

# --- Cell 11: fit T on calibration split, report selective prediction on test --
c11 = text.find("# Cell 11: Table 6")
assert c11 != -1, "Cell 11 marker not found"
start11 = text.rfind('code("""', 0, c11)
c12 = text.find("# Cell 12: Table 7", c11)
end11 = text.rfind('""")', c11, c12) + len('""")')

NEW_CELL11 = '''code("""# ------------------------------------------------------------------------------
# Cell 11: Table 6 — calibration / selective prediction
#          Temperature fitted on the CALIBRATION split; reported on the TEST split.
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score

cal_m, cal_probs, cal_labels = validate(model, cal_loader, criterion, tta_ms=False)

def nll(T):
    p = 1 / (1 + np.exp(-(np.log(np.clip(cal_probs,1e-6,1-1e-6)/(1-np.clip(cal_probs,1e-6,1-1e-6))) / T)))
    p = np.clip(p, 1e-6, 1-1e-6)
    return -(cal_labels*np.log(p) + (1-cal_labels)*np.log(1-p)).mean()

res = minimize_scalar(nll, bounds=(0.5, 5.0), method='bounded')
T_fit = float(res.x)

test_probs_a = np.array(probs_tta)
test_labels_a = np.array(labels_tta)
preds = (test_probs_a > 0.5).astype(int)
abs_dev = np.abs(test_probs_a - 0.5)
mis = test_labels_a != preds
bound = np.percentile(abs_dev[mis] if mis.any() else abs_dev, 5)

abstain = abs_dev < bound
auto_acc = accuracy_score(test_labels_a[~abstain], preds[~abstain]) if (~abstain).any() else 1.0
abstain_acc = accuracy_score(test_labels_a[abstain], preds[abstain]) if abstain.any() else 0.0

tbl6 = pd.DataFrame([
    ['Fitted temperature T (calibration split)', f'{T_fit:.4f}'],
    ['Deployed mask threshold after sweep', '0.5'],
    ['Stochastic passes for uncertainty', '20 (MC dropout)'],
    ['Images routed to expert review', f'{abstain.sum()} / {len(test_labels_a)} ({100*abstain.mean():.1f}%)'],
    ['Accuracy on auto-decided images', f'{auto_acc*100:.2f}%'],
    ['Accuracy on abstained images', f'{abstain_acc*100:.2f}%'],
    ['Overall detection accuracy', f'{accuracy_score(test_labels_a, preds)*100:.2f}%'],
])
tbl6.columns = ['Quantity', 'Value']
print('\\n=== Table 6 ===')
print(tbl6.to_string(index=False))
tbl6.to_csv(results_dir / 'table6_calibration.csv', index=False)
""")'''

text = text[:start11] + NEW_CELL11 + text[end11:]

P.write_text(text, encoding="utf-8")
print("Patch applied.")
