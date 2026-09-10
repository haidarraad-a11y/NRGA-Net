#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""metrics.py

Reconciled pooled metrics for NRGA-Net.

Reviewer comments #1 and #2 point out that the headline pooled numbers must be
algebraically consistent.  For a binary forged-class confusion matrix:

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)
    IoU       = TP / (TP + FP + FN)
    Dice      = 2 * TP / (2 * TP + FP + FN)

For the forged class, Dice and F1 are identical.  Also:

    F1 = 2 * IoU / (1 + IoU)   ->   IoU = F1 / (2 - F1)

All functions below return Python scalars so they can be used from the Colab
notebook and from the main training script.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def pooled_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-9):
    """Return precision, recall, F1, Dice, IoU from pooled counts."""
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def reconcile_precision_recall(precision: float, recall: float, eps: float = 1e-9):
    """Given precision and recall, return the consistent F1, Dice and IoU."""
    precision = float(precision)
    recall = float(recall)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = f1 / (2.0 - f1 + eps)
    dice = f1
    return {"precision": precision, "recall": recall, "f1": f1, "dice": dice, "iou": iou}


def pooled_from_prob(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    include_real_fp: bool = True,
):
    """Compute pooled TP/FP/FN from flattened pixel arrays.

    y_true: 0/1 or float array of ground-truth labels.
    y_prob: predicted probabilities.
    include_real_fp: if True, authentic images contribute pure false positives.
    """
    y_true = np.asarray(y_true).astype(np.float32).ravel()
    y_prob = np.asarray(y_prob).astype(np.float32).ravel()
    y_pred = (y_prob > threshold).astype(np.float32)

    if not include_real_fp:
        # restrict to forged pixels only
        mask = y_true > 0.5
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    tp = float(np.sum(y_pred * y_true))
    fp = float(np.sum(y_pred * (1.0 - y_true)))
    fn = float(np.sum((1.0 - y_pred) * y_true))
    return pooled_from_counts(tp, fp, fn)


def per_image_iou_dice(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, eps: float = 1e-8):
    """Return per-image IoU and Dice (each image is a sample)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim == 2:
        y_true = y_true[None, ...]
        y_prob = y_prob[None, ...]
    y_pred = (y_prob > threshold).astype(np.float32)
    inter = (y_pred * y_true).sum(axis=(-2, -1))
    union = ((y_pred + y_true) > 0.5).sum(axis=(-2, -1))
    iou = inter / (union + eps)
    dice = 2 * inter / (y_pred.sum(axis=(-2, -1)) + y_true.sum(axis=(-2, -1)) + eps)
    return iou.mean(), dice.mean()


def detection_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    """Image-level detection accuracy, AUC, F1, precision, recall."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auc"] = 0.0
    return out


def assert_consistent(precision: float, recall: float, f1: float, iou: float, tol: float = 1e-4):
    """Raise if the four reported numbers cannot come from the same confusion matrix."""
    rec = reconcile_precision_recall(precision, recall)
    if abs(rec["f1"] - f1) > tol or abs(rec["iou"] - iou) > tol:
        raise ValueError(
            f"Inconsistent metrics: P={precision:.6f} R={recall:.6f} "
            f"=> F1={rec['f1']:.6f}, IoU={rec['iou']:.6f}; "
            f"reported F1={f1:.6f}, IoU={iou:.6f}"
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example from reviewer comment #2:
    # P=0.9769, R=0.9902 gives F1=0.9835 and IoU≈0.9675.
    rec = reconcile_precision_recall(0.9769, 0.9902)
    print("Reconciled:", {k: round(v, 4) for k, v in rec.items()})
    assert abs(rec["f1"] - 0.9835) < 1e-4
    assert abs(rec["iou"] - 0.9675) < 1e-4

    # Verify Dice = F1 for forged class.
    m = pooled_from_counts(tp=970, fp=30, fn=10)
    assert abs(m["dice"] - m["f1"]) < 1e-9
    print("Dice=F1 consistency OK")
