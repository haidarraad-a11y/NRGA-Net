#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""create_splits.py

Deterministic train / calibration / final-test split generator for NRGA-Net.

Design goals (reviewer comments #1, #2, #6):
  * The final-test set is never used for temperature scaling, threshold selection,
    uncertainty-bound fitting, checkpoint selection, or any post-hoc decision.
  * Splits are reproducible from a single random seed and are saved as CSVs.
  * A small 2,000-image quick subset (1,000 real + 1,000 fake) is produced for
    fast Colab table generation.

Expected dataset layout is the same as src/main.py:

    DATA_ROOT/
    ├── Fake-Vaihingen/{real,fake}/{train,val}/...
    ├── Fake-LoveDA/{real,fake}/{train,val}/...
    └── Local_Diffusion/{real,fake,mask}/{train,val}/...

Usage:
    set NRGA_DATA_ROOT=D:\path\to\DATA_ROOT       (Windows)
    python scripts/create_splits.py --root %NRGA_DATA_ROOT% --seed 42

Outputs (under <repo>/splits/):
    train_full.csv
    train_quick_2k.csv
    calibration.csv
    test.csv
    metadata.json
    leave_one_family_out/{train_minus_<family>.csv, test_<family>.csv}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Dataset layout: keep this in sync with src/main.py _DATASET_GROUPS['ALL']
# ---------------------------------------------------------------------------
def _build_layout(base: Path) -> dict:
    b = str(base)
    return {
        "Fake-Vaihingen-lama": {
            "family": "lama",
            "benchmark": "Fake-Vaihingen",
            "train": {
                "real": f"{b}/Fake-Vaihingen/real/train",
                "fake": f"{b}/Fake-Vaihingen/fake/train/lama",
                "mask": f"{b}/Fake-Vaihingen/fake/train/inpainted_mask",
            },
            "val": {
                "real": f"{b}/Fake-Vaihingen/real/val",
                "fake": f"{b}/Fake-Vaihingen/fake/val/lama",
                "mask": f"{b}/Fake-Vaihingen/fake/val/inpainted_mask",
            },
        },
        "Fake-Vaihingen-repaint": {
            "family": "repaint",
            "benchmark": "Fake-Vaihingen",
            "train": {
                "real": f"{b}/Fake-Vaihingen/real/train",
                "fake": f"{b}/Fake-Vaihingen/fake/train/repaint",
                "mask": f"{b}/Fake-Vaihingen/fake/train/inpainted_mask",
            },
            "val": {
                "real": f"{b}/Fake-Vaihingen/real/val",
                "fake": f"{b}/Fake-Vaihingen/fake/val/repaint",
                "mask": f"{b}/Fake-Vaihingen/fake/val/inpainted_mask",
            },
        },
        "Fake-LoveDA-lama": {
            "family": "lama",
            "benchmark": "Fake-LoveDA",
            "train": {
                "real": f"{b}/Fake-LoveDA/real/train",
                "fake": f"{b}/Fake-LoveDA/fake/train/lama",
                "mask": f"{b}/Fake-LoveDA/fake/train/inpainted_mask",
            },
            "val": {
                "real": f"{b}/Fake-LoveDA/real/val",
                "fake": f"{b}/Fake-LoveDA/fake/val/lama",
                "mask": f"{b}/Fake-LoveDA/fake/val/inpainted_mask",
            },
        },
        "Fake-LoveDA-repaint": {
            "family": "repaint",
            "benchmark": "Fake-LoveDA",
            "train": {
                "real": f"{b}/Fake-LoveDA/real/train",
                "fake": f"{b}/Fake-LoveDA/fake/train/repaint",
                "mask": f"{b}/Fake-LoveDA/fake/train/inpainted_mask",
            },
            "val": {
                "real": f"{b}/Fake-LoveDA/real/val",
                "fake": f"{b}/Fake-LoveDA/fake/val/repaint",
                "mask": f"{b}/Fake-LoveDA/fake/val/inpainted_mask",
            },
        },
        "Local_Diffusion": {
            "family": "latent_diffusion",
            "benchmark": "Fake-LocalDiff",
            "train": {
                "real": f"{b}/Local_Diffusion/real/train",
                "fake": f"{b}/Local_Diffusion/fake/train",
                "mask": f"{b}/Local_Diffusion/mask/train",
            },
            "val": {
                "real": f"{b}/Local_Diffusion/real/val",
                "fake": f"{b}/Local_Diffusion/fake/val",
                "mask": f"{b}/Local_Diffusion/mask/val",
            },
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def _build_mask_index(mask_dir: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    if not mask_dir.exists():
        return idx
    for p in _list_images(mask_dir):
        stem = p.stem
        idx.setdefault(stem, p)
        for suf in ("_mask", "-mask", "_gt", "-gt", ".mask", "_label"):
            if stem.endswith(suf):
                idx.setdefault(stem[: -len(suf)], p)
    return idx


def _resolve_mask(fake_path: Path, mask_index: dict[str, Path]) -> Path | None:
    for key in (fake_path.stem, fake_path.stem + "_mask", fake_path.stem + "-mask"):
        if key in mask_index:
            return mask_index[key]
    return None


def _collect_unique_images(layout: dict, split: str) -> tuple[list[dict], list[dict]]:
    """Return (train_samples, val_samples) as list of dicts keyed by canonical path.

    Real folders are de-duplicated across methods that share the same physical
    directory (e.g. Fake-Vaihingen-lama and Fake-Vaihingen-repaint share one real
    set).  Each unique image appears exactly once, but we keep all generator labels
    it carries so downstream stratification is accurate.
    """
    by_split: dict[str, dict[str, dict]] = {"train": {}, "val": {}}

    for method, meta in layout.items():
        family = meta["family"]
        benchmark = meta["benchmark"]
        sp = meta[split]

        # real images: de-duplicate by realpath
        seen_real = set()
        for p in _list_images(Path(sp["real"])):
            rp = os.path.realpath(p)
            if rp in seen_real:
                continue
            seen_real.add(rp)
            by_split[split].setdefault(
                rp,
                {
                    "path": str(p),
                    "label": 0,
                    "methods": [],
                    "families": [],
                    "benchmarks": [],
                    "mask_path": "",
                },
            )
            rec = by_split[split][rp]
            rec["methods"].append(method)
            rec["families"].append(family)
            rec["benchmarks"].append(benchmark)

        # fake images: one sample per fake file
        mask_index = _build_mask_index(Path(sp["mask"]))
        for p in _list_images(Path(sp["fake"])):
            rp = os.path.realpath(p)
            mask = _resolve_mask(p, mask_index)
            by_split[split][rp] = {
                "path": str(p),
                "label": 1,
                "methods": [method],
                "families": [family],
                "benchmarks": [benchmark],
                "mask_path": str(mask) if mask else "",
            }

    return list(by_split["train"].values()), list(by_split["val"].values())


def _stratified_sample(items: list[dict], n_total: int, rng: np.random.Generator) -> list[dict]:
    """Sample n_total items proportionally by (benchmark, family), deterministic."""
    if len(items) <= n_total:
        return items
    groups = defaultdict(list)
    for it in items:
        key = (it["benchmarks"][0], it["families"][0] if it["families"] else "real")
        groups[key].append(it)

    # number of items per group proportional to group size
    counts = {k: len(v) for k, v in groups.items()}
    total = sum(counts.values())
    per_group = {k: max(1, int(round(n_total * c / total))) for k, c in counts.items()}

    # trim/expand to exactly n_total
    while sum(per_group.values()) > n_total:
        per_group[max(per_group, key=lambda k: per_group[k] - counts[k])] -= 1
    while sum(per_group.values()) < n_total:
        per_group[max(per_group, key=lambda k: counts[k] - per_group[k])] += 1

    out = []
    for k, v in groups.items():
        n = min(per_group[k], len(v))
        idx = rng.choice(len(v), size=n, replace=False)
        out.extend([v[i] for i in idx])
    return out


def _split_calibration_test(
    val_items: list[dict], cal_frac: float, rng: np.random.Generator
) -> tuple[list[dict], list[dict]]:
    """Stratified split of unique val images by (benchmark, label)."""
    strata = defaultdict(list)
    for it in val_items:
        strata[(it["benchmarks"][0], it["label"])].append(it)

    cal, test = [], []
    for key, items in strata.items():
        items = items[:]
        rng.shuffle(items)
        n_cal = max(1, int(round(len(items) * cal_frac))) if len(items) >= 5 else 0
        # keep at least one test sample when possible
        if n_cal >= len(items):
            n_cal = max(0, len(items) - 1)
        cal.extend(items[:n_cal])
        test.extend(items[n_cal:])
    return cal, test


def _flatten_fields(items: list[dict]) -> list[dict]:
    """Convert list-of-methods fields to comma-separated strings for CSV."""
    out = []
    for it in items:
        out.append(
            {
                "path": it["path"],
                "label": it["label"],
                "methods": ";".join(sorted(set(it["methods"]))),
                "families": ";".join(sorted(set(it["families"]))),
                "benchmarks": ";".join(sorted(set(it["benchmarks"]))),
                "mask_path": it.get("mask_path", ""),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _counts(rows: list[dict]) -> dict:
    real = sum(1 for r in rows if int(r["label"]) == 0)
    fake = sum(1 for r in rows if int(r["label"]) == 1)
    by_bench = defaultdict(lambda: {"real": 0, "fake": 0})
    for r in rows:
        bench = r["benchmarks"].split(";")[0]
        lbl = "real" if int(r["label"]) == 0 else "fake"
        by_bench[bench][lbl] += 1
    return {"total": len(rows), "real": real, "fake": fake, "by_benchmark": dict(by_bench)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate deterministic NRGA-Net splits")
    parser.add_argument("--root", default=os.environ.get("NRGA_DATA_ROOT", ""), help="Dataset root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cal-frac", type=float, default=0.20, help="Fraction of val used for calibration")
    parser.add_argument("--out", default=None, help="Output directory (default: repo/splits)")
    args = parser.parse_args()

    if not args.root:
        raise SystemExit("ERROR: pass --root or set NRGA_DATA_ROOT")

    root = Path(args.root)
    repo = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out) if args.out else repo / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    layout = _build_layout(root)
    train_items, _ = _collect_unique_images(layout, split="train")
    _, val_items = _collect_unique_images(layout, split="val")

    # Quick subset: 1,000 real + 1,000 fake from training pool
    real_train = [it for it in train_items if it["label"] == 0]
    fake_train = [it for it in train_items if it["label"] == 1]
    quick_real = _stratified_sample(real_train, 1000, rng)
    quick_fake = _stratified_sample(fake_train, 1000, rng)
    quick_items = quick_real + quick_fake
    rng.shuffle(quick_items)

    # Calibration / final test split
    cal_items, test_items = _split_calibration_test(val_items, args.cal_frac, rng)

    # Leave-one-generator-family-out splits
    families = ["lama", "repaint", "latent_diffusion"]
    loo_dir = out_dir / "leave_one_family_out"
    loo_dir.mkdir(parents=True, exist_ok=True)
    loo_files = {}
    for fam in families:
        # train on all train images whose family != fam
        train_minus = [it for it in train_items if fam not in set(it["families"])]
        # test on held-out family's val forgeries + the corresponding val reals
        test_fam = [it for it in val_items if fam in set(it["families"])]
        _write_csv(loo_dir / f"train_minus_{fam}.csv", _flatten_fields(train_minus))
        _write_csv(loo_dir / f"test_{fam}.csv", _flatten_fields(test_fam))
        loo_files[fam] = {
            "train_minus": str(loo_dir / f"train_minus_{fam}.csv"),
            "test": str(loo_dir / f"test_{fam}.csv"),
        }

    # Write main splits
    _write_csv(out_dir / "train_full.csv", _flatten_fields(train_items))
    _write_csv(out_dir / "train_quick_2k.csv", _flatten_fields(quick_items))
    _write_csv(out_dir / "calibration.csv", _flatten_fields(cal_items))
    _write_csv(out_dir / "test.csv", _flatten_fields(test_items))

    # Metadata
    metadata = {
        "seed": args.seed,
        "cal_frac": args.cal_frac,
        "splits": {
            "train_full": _counts(_flatten_fields(train_items)),
            "train_quick_2k": _counts(_flatten_fields(quick_items)),
            "calibration": _counts(_flatten_fields(cal_items)),
            "test": _counts(_flatten_fields(test_items)),
        },
        "leave_one_family_out": {
            fam: {
                "train_minus": _counts(_flatten_fields([it for it in train_items if fam not in set(it["families"])])),
                "test": _counts(_flatten_fields([it for it in val_items if fam in set(it["families"])])),
            }
            for fam in families
        },
        "files": {
            "train_full": str(out_dir / "train_full.csv"),
            "train_quick_2k": str(out_dir / "train_quick_2k.csv"),
            "calibration": str(out_dir / "calibration.csv"),
            "test": str(out_dir / "test.csv"),
            "loo": loo_files,
        },
        "note": "Calibration set is used for temperature/threshold/uncertainty-bound fitting; final-test set is used only for reported metrics.",
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Verify no overlap
    cal_paths = {os.path.realpath(r["path"]) for r in _flatten_fields(cal_items)}
    test_paths = {os.path.realpath(r["path"]) for r in _flatten_fields(test_items)}
    overlap = cal_paths & test_paths
    if overlap:
        raise RuntimeError(f"FATAL: calibration and test overlap ({len(overlap)} paths)")

    print("Splits written to:", out_dir)
    print(json.dumps(metadata["splits"], indent=2))
    print("Calibration/Test overlap:", len(overlap), "paths")


if __name__ == "__main__":
    main()
