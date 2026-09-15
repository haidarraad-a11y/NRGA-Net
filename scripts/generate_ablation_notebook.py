#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate notebooks/NRGA-Net_Ablations_LOFO.ipynb.

Self-contained Colab notebook (reads data from Google Drive only, saves all
results to Google Drive) that produces:
  - Table 7: component-ablation retrains (SES / FRE / FDA-deform / CBFH /
    edge-supervision / distortion-bank removals) + full-model control at the
    same reduced epoch budget, with optional multi-seed variability.
  - Table 9: leave-one-generator-family-out zero-shot runs.

Cells 1-5 (env, Drive mount, splits, definitions) are copied verbatim from the
generated NRGA-Net_FullTables_local.ipynb so the two notebooks stay in sync.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "notebooks" / "NRGA-Net_FullTables_local.ipynb"
DST_NB = ROOT / "notebooks" / "NRGA-Net_Ablations_LOFO.ipynb"

# ---------------------------------------------------------------------------
# load the full-run notebook and extract Cells 1..5
# ---------------------------------------------------------------------------
src = json.loads(SRC_NB.read_text(encoding="utf-8"))
copied = {}
for cell in src["cells"]:
    if cell["cell_type"] != "code":
        continue
    text = "".join(cell["source"])
    m = re.search(r"#\s*Cell\s+(\d+(?:\.\d+)?):", text)
    if m and m.group(1) in {"1", "2", "3", "4", "5"}:
        copied.setdefault(m.group(1), text)
missing = {"1", "2", "3", "4", "5"} - set(copied)
assert not missing, f"missing cells in source notebook: {missing}"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


md("""# NRGA-Net — Table 7 (component ablations) + Table 9 (leave-one-family-out)

**Drive-direct, self-contained retrains.** No dataset upload; reads images from
`DB_local` in Google Drive and writes every checkpoint/result to
`DB_local/NRGA_Net_revised/ablation_results/`.

What it runs (each a full training run at a reduced, identical budget):
- **Table 7**: full-model control + 6 controlled removals
  (`SES`, `FRE`, `FDA deformable attention`, `CBFH`, `edge supervision`, `distortion bank`)
- **Table 9**: 3 leave-one-family-out runs (train on 2 families, zero-shot on the held-out one)

Protocol per run: checkpoint selection on the **calibration** split only;
final numbers on the **final-test** split (single pass + TTA). Every run
resumes from its last checkpoint, so disconnects are safe; completed runs are
skipped on re-run (`SKIP_COMPLETED = True`).

Typical cost on an A100 runtime: ~7–8 h per Table 7 run (~11 min/epoch x 40
epochs with early stopping), ~6–7 h per LOFO run. Run over several sessions;
progress is preserved in Drive.""")

for key in ("1", "2", "3", "4", "5"):
    code(copied[key])

code("""# ------------------------------------------------------------------------------
# Cell A1: ablation / LOFO configuration
# ------------------------------------------------------------------------------
import copy as _copy, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from torch.cuda.amp import GradScaler

# Reduced but identical budget for ALL runs so the comparison is fair.
# (The published 100-epoch numbers are NOT used for these comparisons.)
ABL_EPOCHS   = 40
ABL_PATIENCE = 6
ABL_SEEDS    = [42]        # add 43, 44 for multi-seed variability (mean +/- std)
RUN_TABLE7   = True
RUN_TABLE9   = True
SKIP_COMPLETED = True      # skip runs whose result JSON already exists in Drive

ABL_OUT = Path(RUN_OUTPUT_DIR) / 'ablation_results'
ABL_OUT.mkdir(parents=True, exist_ok=True)
print('Ablation results ->', ABL_OUT)
print(f'Budget: {ABL_EPOCHS} epochs, patience {ABL_PATIENCE}, seeds {ABL_SEEDS}')""")

code("""# ------------------------------------------------------------------------------
# Cell A2: loaders from the immutable split CSVs (train / calibration / test)
# ------------------------------------------------------------------------------
import shutil
SPLITS_DIR = Path(REPO_ROOT) / 'splits'
LINK_BASE = Path('/content/abl_links')

def _link_all(dst_dir, pairs):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src, name in pairs:
        d = dst_dir / name
        if d.exists() or d.is_symlink():
            d.unlink()
        os.symlink(src, d)

def build_split_loader(df, tag, img_size=None, augment=False, shuffle=False,
                       batch=None, drop_last=False):
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
    _seen = set()
    for (bench, fam), items in sorted(groups.items()):
        gname = 'Local_Diffusion' if bench == 'Fake-LocalDiff' else (
            f'{bench}-{fam}' if fam != 'real' else bench)
        rd, fd, md_ = base / gname / 'real', base / gname / 'fake', base / gname / 'mask'
        _rk = str(rd.resolve()) if rd.exists() else str(rd)
        _lr = True
        if getattr(cfg, 'DEDUPE_REALS', True) and items['real']:
            _lr = _rk not in _seen
            _seen.add(_rk)
        _link_all(rd, [(p, f'{i:05d}_{Path(p).name}') for i, p in enumerate(items['real'])])
        with_mask = [(p, m) for (p, m) in items['fake'] if m is not None]
        _link_all(fd, [(p, f'{Path(p).stem}{Path(p).suffix}') for p, _ in with_mask])
        _link_all(md_, [(m, f'{Path(p).stem}_mask{Path(m).suffix}') for (p, m) in with_mask])
        ds = InpaintingSegDataset(
            real_dir=str(rd), fake_dir=str(fd), mask_dir=str(md_),
            dataset_name=gname, split='train' if augment else 'val',
            img_size=img_size or cfg.IMG_SIZE,
            augment=augment, native_crop=getattr(cfg, 'NATIVE_CROP', False),
            load_reals=_lr)
        if len(ds) > 0:
            dss.append(ds)
            names.append(gname)
    from torch.utils.data import DataLoader, ConcatDataset
    loader = DataLoader(ConcatDataset(dss), batch_size=batch or cfg.BATCH_SIZE,
                        shuffle=shuffle, num_workers=cfg.NUM_WORKERS,
                        pin_memory=True, drop_last=drop_last)
    return loader, names

train_df = pd.read_csv(SPLITS_DIR / 'train_full.csv')
cal_df   = pd.read_csv(SPLITS_DIR / 'calibration.csv')
test_df  = pd.read_csv(SPLITS_DIR / 'test.csv')

train_loader_csv, tr_names = build_split_loader(
    train_df, 'train_full', augment=True, shuffle=True, drop_last=True)
cal_loader,  cal_names = build_split_loader(cal_df,  'cal')
test_loader, tst_names = build_split_loader(test_df, 'test')
print('Train groups:', tr_names, '| batches:', len(train_loader_csv))
print('Calibration groups:', cal_names)
print('Test groups:', tst_names)""")

code("""# ------------------------------------------------------------------------------
# Cell A3: per-run trainer (resume-safe, calibration-selected, NaN-guarded)
# ------------------------------------------------------------------------------
def _set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _metrics_floats(m):
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating))}

def train_eval_run(run_name, overrides, train_ld, test_ld, sel_ld,
                   epochs=None, seed=None):
    run_dir = Path(RUN_OUTPUT_DIR) / 'ablations' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_p, last_p = run_dir / 'best.pt', run_dir / 'last.pt'
    n_ep = int(epochs or ABL_EPOCHS)

    cfg_run = _copy.copy(cfg)
    for k, v in overrides.items():
        setattr(cfg_run, k, v)
    cfg_run.EPOCHS = n_ep
    cfg_run.PATIENCE = ABL_PATIENCE

    _set_seed(seed if seed is not None else ABL_SEEDS[0])
    model_r = NRGANet(cfg_run).to(DEVICE)
    opt = optim.AdamW(build_param_groups(model_r, cfg_run),
                      lr=float(cfg.LR_DECODER), weight_decay=cfg.WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-6)
    scaler = GradScaler()
    ema = EMA(model_r, cfg.EMA_DECAY) if cfg.EMA_ENABLE else None
    crit = NRGALoss(cfg_run).to(DEVICE)

    best_s, patience, start = -1.0, 0, 1
    if getattr(cfg, 'RESUME', True) and last_p.exists():
        ck = torch.load(last_p, map_location=DEVICE, weights_only=False)
        _bad = [k for k, v in ck['model_state'].items()
                if torch.is_floating_point(v) and not bool(torch.isfinite(v).all())]
        if _bad:
            print(f'>>> {run_name}: last checkpoint has {len(_bad)} non-finite tensors -> fresh start')
        else:
            model_r.load_state_dict(ck['model_state'])
            opt.load_state_dict(ck['optim_state'])
            sched.load_state_dict(ck['sched_state'])
            scaler.load_state_dict(ck['scaler_state'])
            if ema is not None and ck.get('ema_shadow') is not None:
                ema.shadow = {k: v.to(DEVICE) for k, v in ck['ema_shadow'].items()}
                ema.updates = int(ck.get('ema_updates', 0))
                sanitize_ema_(ema)
            start = int(ck['epoch']) + 1
            best_s = float(ck.get('best_score', -1.0))
            patience = int(ck.get('patience', 0))
            print(f'>>> {run_name}: resumed at epoch {start} (best {best_s:.4f})')
        del ck

    print(f'[{run_name}] overrides={overrides} | {n_ep} epochs | '
          f'train batches {len(train_ld)} | {time.strftime("%H:%M:%S")}')
    for epoch in range(start, n_ep + 1):
        t0 = time.time()
        train_loss, _ = train_one_epoch(model_r, train_ld, opt, scaler, crit, ema)
        ema_active = ema is not None and ema.updates >= cfg.EMA_WARMUP_STEPS
        if ema_active:
            ema.apply_to(model_r)
        m_sel, _, _ = validate(model_r, sel_ld, crit, tta_ms=False)
        s = float(m_sel.get('pooled_iou', m_sel.get('mean_iou', 0.0)))
        if not np.isfinite(s) or s <= 0.0:
            print(f'[{run_name}] ep {epoch}: non-finite/zero selection score {s} -- not saved')
            s = -1.0
        else:
            print(f'[{run_name}] ep {epoch}/{n_ep} ({time.time()-t0:.0f}s) '
                  f'loss={train_loss.get("total",0):.4f} cal_IoU={s:.4f}')
        if s > best_s:
            best_s, patience = s, 0
            torch.save({'epoch': epoch, 'model_state': model_r.state_dict(),
                        'metrics': _metrics_floats(m_sel)}, best_p)
            print(f'[{run_name}]   -> new best (calibration pooled IoU {best_s:.4f})')
        else:
            patience += 1
            print(f'[{run_name}]   no improvement ({patience}/{ABL_PATIENCE})')
        if ema_active:
            ema.restore(model_r)
        sched.step()
        torch.save({'epoch': epoch, 'model_state': model_r.state_dict(),
                    'optim_state': opt.state_dict(), 'sched_state': sched.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'ema_shadow': (ema.shadow if ema is not None else None),
                    'ema_updates': (ema.updates if ema is not None else 0),
                    'best_score': best_s, 'patience': patience}, last_p)
        if patience >= ABL_PATIENCE:
            print(f'[{run_name}] early stopping.')
            break

    # final evaluation of the calibration-selected best on the FINAL-TEST split
    assert best_p.exists(), f'[{run_name}] no best checkpoint was saved; cannot evaluate'
    bk = torch.load(best_p, map_location=DEVICE, weights_only=False)
    model_r.load_state_dict(bk['model_state'])
    model_r.eval()
    m_single, _, _ = validate(model_r, test_ld, crit, tta_ms=False)
    m_tta, _, _ = validate(model_r, test_ld, crit, tta_ms=True)
    out = {'run': run_name, 'overrides': {k: str(v) for k, v in overrides.items()},
           'epochs': n_ep, 'seed': (seed if seed is not None else ABL_SEEDS[0]),
           'sel_epoch': int(bk.get('epoch', -1)),
           'calibration_pooled_iou': best_s,
           'test_single': _metrics_floats(m_single),
           'test_tta': _metrics_floats(m_tta)}
    with open(Path(RUN_OUTPUT_DIR) / 'ablation_results' / f'{run_name}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[{run_name}] DONE | test IoU single={out["test_single"].get("pooled_iou",0)*100:.2f} '
          f'TTA={out["test_tta"].get("pooled_iou",0)*100:.2f} | saved.')
    del model_r, bk
    torch.cuda.empty_cache()
    return out""")

code("""# ------------------------------------------------------------------------------
# Cell A4: Table 7 queue — full-model control + 6 controlled removals
# ------------------------------------------------------------------------------
ABLATION_RUNS = [
    ('full_model_ctrl', {}),
    ('no_ses',          {'SES_ENABLE': False}),
    ('no_fre',          {'FRE_ENABLE': False}),
    ('no_fda_deform',   {'FDA_DEFORM_ENABLE': False}),
    ('no_cbfh',         {'CBFH_ENABLE': False}),
    ('no_edge_sup',     {'LAMBDA_EDGESUP': 0.0}),
    ('no_distortion',   {'ROBUST_ENABLE': False}),
]
results7 = []
if RUN_TABLE7:
    for seed in ABL_SEEDS:
        for name, ov in ABLATION_RUNS:
            rn = f'{name}_s{seed}' if len(ABL_SEEDS) > 1 else name
            rj = ABL_OUT / f'{rn}.json'
            if SKIP_COMPLETED and rj.exists():
                with open(rj) as f:
                    results7.append(json.load(f))
                print(f'[skip] {rn} already complete')
                continue
            results7.append(train_eval_run(rn, ov, train_loader_csv,
                                           test_loader, cal_loader, seed=seed))
    print(f'Table 7 runs finished: {len(results7)}')""")

code("""# ------------------------------------------------------------------------------
# Cell A5: Table 9 queue — leave-one-generator-family-out (zero-shot)
# ------------------------------------------------------------------------------
FAMILIES = ['lama', 'repaint', 'latent_diffusion']
results9 = []
if RUN_TABLE9:
    for fam in FAMILIES:
        rn = f'lofo_minus_{fam}'
        rj = ABL_OUT / f'{rn}.json'
        if SKIP_COMPLETED and rj.exists():
            with open(rj) as f:
                results9.append(json.load(f))
            print(f'[skip] {rn} already complete')
            continue
        tr_lofo = train_df[~((train_df['label'] == 1) &
                             (train_df['families'].astype(str) == fam))]
        te_lofo = test_df[(test_df['label'] == 1) &
                          (test_df['families'].astype(str) == fam)]
        print(f'--- LOFO -{fam}: train {len(tr_lofo)} rows '
              f'({int((tr_lofo.label==1).sum())} fakes) | zero-shot test {len(te_lofo)} fakes')
        lofo_train_ld, _ = build_split_loader(tr_lofo, f'train_minus_{fam}',
                                              augment=True, shuffle=True, drop_last=True)
        lofo_test_ld, _ = build_split_loader(te_lofo, f'test_{fam}')
        results9.append(train_eval_run(rn, {}, lofo_train_ld, lofo_test_ld, cal_loader))
        del lofo_train_ld, lofo_test_ld
    print(f'Table 9 runs finished: {len(results9)}')""")

code("""# ------------------------------------------------------------------------------
# Cell A6: aggregate -> table7_ablations.csv + table9_leave_one_family_out.csv
# ------------------------------------------------------------------------------
import statistics
import re as _re

DS_COLS = ['Fake-Vaihingen-lama', 'Fake-Vaihingen-repaint',
           'Fake-LoveDA-lama', 'Fake-LoveDA-repaint', 'Local_Diffusion']
LABELS = {
    'full_model_ctrl': 'Full model (control)',
    'no_ses': '- spectral edge stream (SES)',
    'no_fre': '- frequency residual encoder (FRE)',
    'no_fda_deform': '- deformable attention in FDA',
    'no_cbfh': '- content-based forensic hash (CBFH)',
    'no_edge_sup': '- edge supervision',
    'no_distortion': '- distortion-bank augmentation',
}

def _base_name(run):
    return _re.sub(r'_s\d+$', '', run)

def _fmt_cell(results, name, split, ds_key):
    sel = [r for r in results if _base_name(r['run']) == name]
    if not sel:
        return 'pending'
    if ds_key == 'OVERALL':
        vals = [r[split].get('pooled_iou', 0.0) * 100 for r in sel]
    else:
        vals = [r[split].get(f'iouPooled_{ds_key}', 0.0) * 100 for r in sel]
    m = statistics.mean(vals)
    if len(vals) > 1:
        return f'{m:.2f} +/- {statistics.stdev(vals):.2f}'
    return f'{m:.2f}'

rows7 = []
for name, label, split in [('full_model_ctrl', 'Full model (control), single pass', 'test_single'),
                           ('full_model_ctrl', 'Full model (control), +TTA', 'test_tta')] + \\
                          [(n, LABELS[n], 'test_single') for n, _ in ABLATION_RUNS
                           if n != 'full_model_ctrl']:
    rows7.append([label] + [_fmt_cell(results7, name, split, c) for c in DS_COLS]
                 + [_fmt_cell(results7, name, split, 'OVERALL')])
tbl7 = pd.DataFrame(rows7, columns=['Configuration'] + DS_COLS + ['Overall (pooled)'])
print('=== Table 7 (same-budget control vs removals; single pass unless noted) ===')
print(tbl7.to_string(index=False))
tbl7.to_csv(ABL_OUT / 'table7_ablations.csv', index=False)

rows9 = []
for r in results9:
    fam = r['run'].replace('lofo_minus_', '')
    n_test = int(((test_df['label'] == 1) & (test_df['families'].astype(str) == fam)).sum())
    rows9.append([fam, ' + '.join(sorted(set(FAMILIES) - {fam})), n_test,
                  f"{r['test_single'].get('pooled_iou', 0)*100:.2f}",
                  f"{r['test_single'].get('pooled_f1', 0)*100:.2f}"])
tbl9 = pd.DataFrame(rows9, columns=['Held-out family', 'Train families',
                                    'Test samples (fakes)', 'IoU (zero-shot)', 'F1 (zero-shot)'])
print('\\n=== Table 9 (zero-shot, single pass) ===')
print(tbl9.to_string(index=False))
tbl9.to_csv(ABL_OUT / 'table9_leave_one_family_out.csv', index=False)

print('\\nAll CSVs saved to:', ABL_OUT)""")

md("""## Notes

- **Fair comparison**: every Table 7 run (control and removals) uses the same
  `ABL_EPOCHS` budget, seed, loaders, and evaluation protocol. Do not compare
  these 40-epoch numbers against the 100-epoch headline results.
- **Multi-seed**: set `ABL_SEEDS = [42, 43, 44]` in Cell A1 and re-run to get
  mean ± std for every row (each extra seed adds one run per configuration).
- **Resume**: if the runtime disconnects, just re-run all cells — completed
  runs are skipped (`SKIP_COMPLETED`), interrupted ones continue from
  `ablations/<run>/last.pt` in Drive.
- The merged CSVs land in `DB_local/NRGA_Net_revised/ablation_results/`;
  download them and share them back so the manuscript tables can be filled.""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
DST_NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Notebook written to: {DST_NB} ({len(cells)} cells)")
