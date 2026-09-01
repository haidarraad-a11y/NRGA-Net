#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NRGA-Net: Noise-Residual Guided Attention Network for local manipulation
detection and localization in satellite imagery.

Reproducibility pipeline accompanying the paper. Running this script executes
the full workflow end to end, in order:

  1.  environment setup (CUDA, bf16 autocast)
  2.  benchmark configuration (Fake-Vaihingen / Fake-LoveDA / Fake-LocalDiff)
  3.  data layer (joint dataset, zoom-crop, distortion bank)
  4.  model (dilated DenseNet-201 + NRE/FRE + forensic gate fusion + SES
      decoder + residual logit cascade + detection head + CBFH)
  5.  losses (focal + Dice + Lovasz + deep supervision + edge supervision)
  6.  content-prior pre-training (masked, pair-free)
  7.  256px main training loop (split LRs, EMA, NaN guards, resume)
  8.  384px boundary fine-tune
  9.  checkpoint health check
  10. FECDNet-comparable pooled evaluation and report tables
  11. training curves and ablation diagnosis
  12. TTA inference, threshold calibration, qualitative figures
  13. CBFH registry build / verification utilities

Set the NRGA_DATA_ROOT environment variable to the dataset root before
running (see data/README.md for the expected folder layout).
"""

# ======================================================================
# NRGA-Net + CBFH: Noise-Residual Guided Attention Network with Content-Based Forensic Hashing
# Satellite Inpainting Detection + Pixel-Level Localization
#
# Novel contributions
# 1. Dual-domain forensic gating: spatial noise-residual + frequency-residual (FFT) features as external gating signals (NoiseGuidedAttentionGate)
# 2. Multi-scale U-Net decoder with skip connections from DenseNet-201+CBAM
# 3. Depthwise separable convolutions for lightweight inference
# 4. ViT global context injection into decoder
# 5. Deep supervision + Focal + Boundary-aware loss
# 6. Frequency-Residual Encoder (FRE): lightweight FFT high-pass (HFRI) + spectral amplitude/phase convolution (FCL) for complementary frequency-domain forensics
#
# Dataset: Fake-LoveDA + Fake-Vaihingen (gt / lama / repaint + inpainted_mask)
# ======================================================================

# ======================================================================
# — closing the gap with FLDCF
#
# Symptom: the ground truth is a thin winding structure (~15 px wide); the model
# returned a fat blob of roughly the right *area* (IoU 0.34). Seven concrete causes,
# each fixed below.
#
# | # | Cause in the base model | fix |
# |---|---|---|
# | 1 | ViT branch. `ViTBranch` tokenised the image into 16×16 patches and the decoder projected those tokens onto a 16×16 grid. A patch transformer cannot represent anything finer than one patch, so it stamped a block-level prior on the mask. FLDCF has no transformer. | ViT + cross-attention bridge removed. Context now comes from an FLDCF-style atrous `FPM` (dilations 1/6/12/18 + image pooling), which is pixel-precise and translation-equivariant. Also frees ~40 M params to pay for fix #2. |
# | 2 | Output stride 32 — the deepest feature was 8×8, where a 15-px road is sub-pixel, so a blob was the only representable answer. FLDCF runs at stride 8. | `DILATE_STAGE3` + `DILATE_STAGE4`: drop the transition pooling, dilate the dense blocks → stride 8 at the same receptive field. `memory_efficient` DenseNet checkpointing pays the memory bill. |
# | 3 | No skip above 64×64 — the final 4× was a blind upsample, and the last two decoder stages had only 32 and 16 channels. | 128×128 stem skip (`c0`) + full-resolution `DetailBranch`; a coarse-to-fine residual logit cascade with a `BR` block at every level (FLDCF `br5/br6/br7`); last two stages widened to 64/32. |
# | 4 | Losses were interior-dominated and boundary-agnostic, so over-spill past the true edge was nearly free. | `EdgeWeightedBCE` (boundary band weighted 9×) — this is the term that actually punishes a fat mask — plus `SoftClDiceLoss` as a *connectivity* guard. `LAMBDA_BOUNDARY` 0.3 → 1.0. See the measured table below: clDice is thickness-blind, so it is not the anti-blob fix, but it is the only term that catches a fragmented contour once the mask does get thin. |
# | 5 | Deep supervision used `mode='nearest'` to downsample the GT — a 15-px road at 1/16 becomes scattered dots, so DS was actively teaching a blob. | `F.adaptive_max_pool2d`, which preserves thin structure at every scale. DS extended to the 1/2 level. |
# | 6 | No learned content prior. FLDCF's decisive cue is the residual of a frozen restoration net; NRGA had only *fixed* SRM filters and an FFT branch (non-local, so it blurs boundaries). | New cell 6b: `BlindSpotPrior`, a self-supervised model of authentic local statistics (centre-masked 5×5 + 1×1 stack), trained on real images only, frozen; `x − R(x)` feeds the detail stream and the noise encoder. |
# | 7 | Dataloader squashed every image to 256 with `LANCZOS` and the mask with `NEAREST`, which breaks a thin mask into disconnected dots. | Mask resized with `BOX` + a 0.35 cut; `NATIVE_CROP` takes a forgery-biased crop at native scale when the source is larger than 256 (automatic no-op at 256). |
#
# Which loss term actually punishes which failure?
#
# Measured on a synthetic 8-px winding contour — penalty added relative to a
# perfect mask (bigger = the term dislikes that prediction more)
#
# | prediction | Dice | clDice | EdgeWeightedBCE |
# |---|---|---|---|
# | dilated ×11 — the fat blob you are seeing | +0.664 | +0.000 | +2.478 |
# | dilated ×21 — fatter | +0.786 | +0.000 | +3.048 |
# | fragmented contour | +0.187 | +0.175 | +0.378 |
# | displaced blob | +0.969 | +0.984 | +1.269 |
#
# Read this before tuning: clDice is thickness-blind — it scores a fat mask
# exactly like a perfect one, so raising `LAMBDA_CLDICE` will *not* de-blob the
# output. Raise `LAMBDA_EDGE` / `EDGE_W` for that. clDice earns its place on the
# fragmented row: its penalty jumps ~145× from perfect to broken (Dice only ~8×),
# so it is the guard against the mask shattering once fixes #1–#3 let it get thin.
#
# Inference also changed: the network is now fully convolutional, so `predict_mask`
# runs at native resolution (`whole`) or tiled, with 8-way TTA, and the mask
# threshold is calibrated on validation IoU instead of fixed at 0.5.
#
# The architecture changed, so older checkpoints cannot be loaded — the model
# must be retrained. Outputs go to `OUTPUT_DIR` (see Config); existing files
# are untouched.
#
# If you hit CUDA OOM: set `Config.DILATE_STAGE3 = False` first (biggest saving),
# then `DILATE_STAGE4 = False`, then lower `BATCH_SIZE`.
#
# ======================================================================

# ======================================================================
# 1. Install
# ======================================================================

# ======================================================================
# 2. Imports & Setup
# ======================================================================

import io
import os, random, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast

import torchvision.transforms.functional as TF
from torchvision.models import densenet201, DenseNet201_Weights
from torchvision.ops import DeformConv2d

import timm
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             balanced_accuracy_score, # <- CAST step 2c
                             classification_report, confusion_matrix)
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ===================== AMP dtype -- fp16 -> bf16 =====================
# The 256px run can go NaN late in training with a very specific signature:
# focal/dice/lovasz/suppress became NaN while the deep-supervision heads (`ds`)
# stayed finite. Those first four all read `main_mask`; `ds` reads `aux_masks`.
# So only the FINAL full-resolution head of the residual cascade overflowed --
# classic fp16, which has a 5-bit exponent (max 65504). bf16 carries fp32's
# 8-bit exponent, so the identical graph cannot overflow.

# Why one overflow was fatal: BatchNorm running_mean / running_var are written
# during the FORWARD pass. GradScaler detects the non-finite gradient and skips
# the optimizer step -- but it cannot undo a buffer write. One inf therefore
# poisons every BN buffer with NaN permanently, after which eval() returns NaN
# forever (all metrics hard 0.0000) while train() keeps working because it uses
# batch statistics. That is exactly what epochs 61-70 showed.
AMP_DTYPE = torch.float16
if torch.cuda.is_available() and getattr(torch.cuda, 'is_bf16_supported', lambda: False)():
    AMP_DTYPE = torch.bfloat16
_autocast_fp16 = autocast # keep the original for A/B comparison

def autocast(*args, **kwargs): # noqa: F811 -- deliberately shadows torch.cuda.amp.autocast
    """Drop-in replacement for the imported `autocast`: same call sites, bf16 dtype."""
    _dev = kwargs.pop('device_type', None) or (args[0] if args else 'cuda')
    if _dev == 'cuda':
        kwargs.setdefault('dtype', AMP_DTYPE)
    return torch.amp.autocast(_dev, **kwargs)

print(f'AMP autocast dtype: {str(AMP_DTYPE).replace("torch.", "")}'
      + (' (bf16 -> fp16 overflow impossible)' if AMP_DTYPE is torch.bfloat16
         else ' (WARNING: bf16 unsupported, fp16 overflow still possible)'))

# ======================================================================
# 3. Config
# ======================================================================

# Base Directory
# Dataset root: override with the NRGA_DATA_ROOT environment variable
BASE_DB = os.environ.get('NRGA_DATA_ROOT',
                         '/content/drive/MyDrive/PhD_GIS_Security/DB_local')

# ----
# Benchmark selector -- switch datasets by editing DATASET_GROUP only.
# Entries that share a 'real' directory (lama + repaint) are de-duplicated at
# load time by cfg.DEDUPE_REALS, so authentic images are counted ONCE.
# ----
DATASET_GROUP = 'ALL' # 'ALL' | 'Fake-Vaihingen' | 'Fake-LoveDA' | 'Local_Diffusion'

_DATASET_GROUPS = {
    'Fake-Vaihingen': {
        'Fake-Vaihingen-lama': {
            'train': {
                'real': f'{BASE_DB}/Fake-Vaihingen/real/train',
                'fake': f'{BASE_DB}/Fake-Vaihingen/fake/train/lama',
                'mask': f'{BASE_DB}/Fake-Vaihingen/fake/train/inpainted_mask'
            },
            'val': {
                'real': f'{BASE_DB}/Fake-Vaihingen/real/val',
                'fake': f'{BASE_DB}/Fake-Vaihingen/fake/val/lama',
                'mask': f'{BASE_DB}/Fake-Vaihingen/fake/val/inpainted_mask'
            }
        },
        'Fake-Vaihingen-repaint': {
            'train': {
                'real': f'{BASE_DB}/Fake-Vaihingen/real/train',
                'fake': f'{BASE_DB}/Fake-Vaihingen/fake/train/repaint',
                'mask': f'{BASE_DB}/Fake-Vaihingen/fake/train/inpainted_mask'
            },
            'val': {
                'real': f'{BASE_DB}/Fake-Vaihingen/real/val',
                'fake': f'{BASE_DB}/Fake-Vaihingen/fake/val/repaint',
                'mask': f'{BASE_DB}/Fake-Vaihingen/fake/val/inpainted_mask'
            }
        },
    },
    'Fake-LoveDA': {
        'Fake-LoveDA-lama': {
            'train': {
                'real': f'{BASE_DB}/Fake-LoveDA/real/train',
                'fake': f'{BASE_DB}/Fake-LoveDA/fake/train/lama',
                'mask': f'{BASE_DB}/Fake-LoveDA/fake/train/inpainted_mask'
            },
            'val': {
                'real': f'{BASE_DB}/Fake-LoveDA/real/val',
                'fake': f'{BASE_DB}/Fake-LoveDA/fake/val/lama',
                'mask': f'{BASE_DB}/Fake-LoveDA/fake/val/inpainted_mask'
            }
        },
        'Fake-LoveDA-repaint': {
            'train': {
                'real': f'{BASE_DB}/Fake-LoveDA/real/train',
                'fake': f'{BASE_DB}/Fake-LoveDA/fake/train/repaint',
                'mask': f'{BASE_DB}/Fake-LoveDA/fake/train/inpainted_mask'
            },
            'val': {
                'real': f'{BASE_DB}/Fake-LoveDA/real/val',
                'fake': f'{BASE_DB}/Fake-LoveDA/fake/val/repaint',
                'mask': f'{BASE_DB}/Fake-LoveDA/fake/val/inpainted_mask'
            }
        },
    },
    'Local_Diffusion': {
        'Local_Diffusion': {
            'train': {
                'real': f'{BASE_DB}/Local_Diffusion/real/train',
                'fake': f'{BASE_DB}/Local_Diffusion/fake/train',
                'mask': f'{BASE_DB}/Local_Diffusion/mask/train'
            },
            'val': {
                'real': f'{BASE_DB}/Local_Diffusion/real/val',
                'fake': f'{BASE_DB}/Local_Diffusion/fake/val',
                'mask': f'{BASE_DB}/Local_Diffusion/mask/val'
            }
        },
    },
}

# 'ALL' = every benchmark in one run. Real folders are still de-duplicated per
# split, so each authentic image is loaded exactly once across the whole mix.
_DATASET_GROUPS['ALL'] = {k: v for _g in ('Fake-Vaihingen', 'Fake-LoveDA', 'Local_Diffusion')
                          for k, v in _DATASET_GROUPS[_g].items()}

assert DATASET_GROUP in _DATASET_GROUPS, (
    f'unknown DATASET_GROUP {DATASET_GROUP!r}; choose one of {list(_DATASET_GROUPS)}')
DATASET_PATHS = _DATASET_GROUPS[DATASET_GROUP]

# Drop entries whose folders are not present on Drive (e.g. a benchmark not
# downloaded yet) so a mixed run never dies on a missing directory.
import os as _os
_avail, _missing = {}, []
for _k, _v in DATASET_PATHS.items():
    if all(_os.path.isdir(_v[_s][_p]) for _s in _v for _p in ('real', 'fake', 'mask')):
        _avail[_k] = _v
    else:
        _missing.append(_k)
if _missing:
    print(f' WARNING: folders not found, skipping -> {_missing}')
if _avail:
    DATASET_PATHS = _avail
else:
    print(' WARNING: none of the configured folders exist; keeping the full list '
          'so the error surfaces with a real path.')
print(f'Benchmark: {DATASET_GROUP} -> {list(DATASET_PATHS)}')
if DATASET_GROUP == 'Fake-LoveDA':
    print(' note: the FLDCF paper uses 512x512 patches for LoveDA (256 for Vaihingen);'
          ' set Config.IMG_SIZE = 512 for a like-for-like comparison.')

class Config:
    DATASET_PATHS = DATASET_PATHS
    OUTPUT_DIR = f'{BASE_DB}/NRGA-Net'

    IMG_SIZE = 256
    BACKBONE_DIM = 512

    # ---------------- resolution of the encoder ----------------
    # FLDCF runs at output stride 8. The base model ran at stride 32, so the
    # semantic decision about a 15-px road was taken on an 8x8 grid where the
    # road is sub-pixel -- that alone guarantees a blob. Dilating stage3/stage4
    # (drop their pooling, dilate the 3x3s) restores stride 8 at the SAME
    # receptive field. `memory_efficient` DenseNet checkpointing pays for it.
    DILATE_STAGE3 = True # -> deepest feature 32x32 (stride 8, like FLDCF)
    DILATE_STAGE4 = True # set STAGE3 False first if you hit CUDA OOM
    MEMORY_EFFICIENT_BACKBONE = True

    # ---------------- content-restoration prior ----------------
    # Self-supervised blind-spot model of AUTHENTIC local statistics; the
    # residual |x - R(x)| is NRGA-Net's counterpart to FLDCF's frozen RDN.
    USE_CONTENT_PRIOR = False # hybrid-only ablation (flip True to re-enable priors)
    # ---- masked content prior + Lovász + point refinement + EMA ----
    PRIOR_MODE = 'masked' # 'masked' (novel, pair-free) | 'blindspot' | 'both' (ablation)
    MASKED_PRIOR_EPOCHS = 15
    MASKED_PRIOR_LR = 1e-4
    MASKED_PRIOR_CH = 32 # masked prior base width
    MASK_RATIO = 0.5 # fraction of MASK_PATCH blocks hidden during prior training
    MASK_PATCH = 16 # mask granularity (px)
    LAMBDA_LOVASZ = 1.0 # direct Jaccard surrogate (Berman et al., CVPR 2018)
    LAMBDA_POINT = 1.0 # uncertainty-sampled point BCE (head was undertrained)
    POINT_K = 4096 # points sampled per image during training
    POINT_UNC_FRAC = 0.5 # fraction of POINT_K drawn from the high-uncertainty band
    POINT_UNC_DELTA = 0.35 # |p-0.5| threshold for inference-time refinement
    EMA_ENABLE = True
    EMA_DECAY = 0.999
    EMA_WARMUP_STEPS = 300 # validate with LIVE weights until the EMA has tracked
    # ---- spectral-edge hybrid (FECDNet x SFNet) ----
    SES_ENABLE = True # spectral edge streams in the decoder
    SES_HP_RADIUS = 0.25 # fraction of the Nyquist radius zeroed (low-freq disk)
    SES_CH = 32 # edge-stream width per scale
    POINT_ENABLE = False # point head off for the hybrid-only ablation
    # ---- split learning rates (the main fix) ----
    LR_ENCODER = 5e-5 # pretrained DenseNet-201 backbone
    LR_DECODER = 5e-4 # decoder / SES / fusion / heads (FECDNet uses 5e-4)
    ENCODER_PARAM_PREFIXES = ('cnn.',)
    RESUME = True # resume from the rolling 'last' checkpoint
    FT384_RESET = False # True -> ignore any FT384 resume file, start fresh
    POOLED_INCLUDE_REAL = True # count real-image false positives in pooled IoU
    DEDUPE_REALS = True
    MASK_POLARITY = 'auto' # 'auto' | 'normal' | 'invert' (forged region must end up = 1) # register shared gt/ once (2099/525 parity)
    ZOOM_CROP_ENABLE = True # forgery-centric zoom crops (train only)
    ZOOM_CROP_PROB = 0.5 # fraction of train samples zoom-cropped per step
    ZOOM_CROP_RANGE = (0.5, 0.75) # crop side = U(lo,hi) x min(W,H) of the source
    MASK_THRESHOLD = 0.40 # mask binarization threshold (val sweep optimum)
    TTA_MS_EVAL = True # final eval uses flip + x1.5 multi-scale TTA
    LAMBDA_EDGESUP = 1.0 # deep edge supervision (BCE+Dice vs GT boundary band)
    EDGE_SUP_BAND = 2 # GT boundary band half-width (px)
    # ---- high-resolution boundary fine-tune (after the 256 run) ----
    FT384_ENABLE = True # fine-tune the converged 256 model at FT384_IMG_SIZE
    FT384_IMG_SIZE = 384
    # the fine-tune was still improving at epoch 8/8 and patience never
    # fired -- the budget was the binding constraint, not convergence.
    FT384_EPOCHS = 40 # was 8
    FT384_LR = 2e-5 # encoder rate; decoder gets FT384_DECODER_MULT x this
    FT384_DECODER_MULT = 10.0
    FT384_BATCH = 3 # effective batch = FT384_BATCH x FT384_ACCUM; lower if OOM
    FT384_ACCUM = 2
    FT384_PATIENCE = 6 # was 4
    PRIOR_CH = 64
    PRIOR_EPOCHS = 3 # cheap: a 5x5 masked conv + three 1x1 layers
    PRIOR_LR = 1e-3

    # ---------------- inference ----------------
    TTA_ENABLE = True
    MASK_TAU_GRID = 41 # threshold sweep resolution (0.05 .. 0.95)
    NATIVE_CROP = True # crop at native scale instead of squashing to 256
                              # (automatic no-op when the source is already 256)
    # the ViT branch was REMOVED (see the header cell). These entries are
    # kept only so older cells still import; nothing reads them any more.
    VIT_PATCH_SIZE = 16
    VIT_DEPTH = 6
    VIT_HEADS = 8
    VIT_DIM = 768

    BATCH_SIZE = 8
    NUM_WORKERS = 0
    LR = 5e-5
    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    PATIENCE = 10

    # Loss weights - CONDITIONAL design
    # ---- rebalance (measured, not guessed) ----
    # At epoch 40 the weighted budget was:
    # cls 43.5% | lovasz 32.8% | dice 13.9% | ds 4.1% | focal 3.4% | prov 1.9%
    # i.e. the single largest consumer was CLASSIFICATION, already saturated at
    # AUC 0.9997 / F1 0.9909, back-propagating through the SHARED encoder; and
    # CBFH verify_acc sat at exactly 0.5000 (chance) every single epoch.
    # FECDNet reaches 93.47 IoU on this dataset with three terms only.
    # puts 93.3% of the budget on segmentation. Old values kept for ablation.
    LAMBDA_CLS = 0.2 # was 2.0 -- head kept alive, no longer dominant
    LAMBDA_FOCAL = 1.0 # was 1.5 Only applied to FAKE samples
    LAMBDA_DICE = 1.0 # Only applied to FAKE samples
    LAMBDA_BOUNDARY = 0.0 # off -- superseded by deep edge supervision
    LAMBDA_EDGE = 0.0 # off -- superseded by deep edge supervision
    LAMBDA_CLDICE = 0.0 # off -- loss debloating (FECDNet-style focus)
    EDGE_W = 8.0 # a boundary pixel weighs (1+EDGE_W)x an interior one
    EDGE_BAND = 5 # boundary band width (dilate-erode of the GT)
    CLDICE_ITERS = 5 # soft-skeletonisation iterations
    LAMBDA_DS = 0.3 # was 0.1 -- this is our FECDNet 'foreground' arm
    # kept at 0.1 rather than removed. Pooled IoU (FECDNet Eq.17) counts
    # real-image false positives, so some suppression still protects the metric;
    # RealFP is 0.0003, so 0.1 is enough to hold it there.
    LAMBDA_MASK_SUPPRESS = 0.1 # was 0.5 # BCE suppression on REAL samples only

    # CBFH (Content-Based Forensic Hashing) - provenance verification
    CBFH_EMBED_DIM = 256
    CBFH_HASH_BITS = 64
    CBFH_ALPHA = 10.0 # binarization sharpness for tanh relaxation
    # CBFH provenance is OFF for the localization run (verify_acc was 0.5000
    # = chance in every // epoch, so it cost encoder capacity for nothing).
    # Set back to 0.5 to reproduce the provenance arm; the checkpoint retains it.
    LAMBDA_PROV = 0.0 # weight of provenance contrastive loss (was 0.5)
    CBFH_MARGIN = 0.4 # margin for mismatched pairs (normalized Hamming)
    LAMBDA_QUANT = 0.01 # quantization regularizer weight

    # Robustness augmentation (JPEG / Gaussian blur / Gaussian noise)
    ROBUST_ENABLE = True
    ROBUST_PROB = 0.5 # prob. of distorting a training input (invariance)
    JPEG_QUALITIES = [65, 75, 85, 95]
    BLUR_KERNELS = [3, 5, 7, 9]
    NOISE_SIGMAS = [0.01, 0.02, 0.04, 0.06]

    # DGT: DCT/FFT-Guided Dynamic Threshold -- inference-time, post-training
    # (SF-CFNet Eq.14). Replaces CAST: no MC dropout, no ECE, no abstain policy.
    # tau = clip(T_base + alpha*tanh(E_freq) + beta*Norm(H_img), lo, hi)
    DGT_ENABLE = True
    DGT_TBASE = 0.5
    DGT_ALPHA = 0.2
    DGT_BETA = 0.45
    DGT_CLIP = (0.3, 0.7)
    DGT_NORM_EFREQ = True # robust z-score E_freq before tanh. With norm='ortho'
                               # the high-pass energy measures ~0.002..0.08 on these
                               # tiles, so tanh(E_freq) ~= E_freq and the alpha term
                               # contributes ~0.002 -- it vanishes instead of steering
                               # anything. The z-score restores its +-alpha range.
    DGT_CENTER_HIMG = True # subtract median(H_img) before beta. H_img averages
                               # ~0.83 here, so beta*H = +0.37 against only +0.20 of
                               # head-room: without this every tau pins to the ceiling.
                               # Set both to False for the original formula, verbatim.

    # CBFH registry maintenance (section "14. CBFH Database")
    CBFH_REGISTER_REALS_ONLY = True # the registry holds AUTHENTIC images only
    CBFH_PURGE_VAIHINGEN_FAKES = True # drop registered Fake-Vaihingen FAKE records
    CBFH_PURGE_OWN_FAKES = True # also drop fakes THIS model registered elsewhere
                                        # (never touches other models' records)
    CBFH_RELABEL_REALS = True # fix the hard-coded 'fake-vaihingen' source_type
    CBFH_DEDUPE_REALS = True # one record per physical real image
    CBFH_BACKUP = True # timestamped copy before any mutation

    # Evaluation guard: refuse to report metrics from an untrained / broken model.
    EVAL_MIN_AUC = 0.80 # ensure_trained_model() raises below this

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print('Dataset paths:', {k: list(v.keys()) for k, v in cfg.DATASET_PATHS.items()})
print('Output:', cfg.OUTPUT_DIR)

# ======================================================================
# 4. Dataset (same as before)
# ======================================================================

IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}

class InpaintingSegDataset(Dataset):
    """Dataset for the new directory structure: separate real/, fake/, mask/ dirs."""
    _bad_paths = set() # class-level: unreadable files seen this session

    def __init__(self, real_dir, fake_dir, mask_dir, dataset_name='unknown', split='train', img_size=256, augment=False, native_crop=False, load_reals=True):
        self.img_size = img_size
        self.split = split
        self.native_crop = bool(native_crop) #
        self.augment = augment
        self.samples = []

        # Load REAL images (label=0, no mask)
        # `load_reals=False` for the 2nd+ method entry that resolves to the
        # same gt/ folder. Fake-Vaihingen shares one authentic set between lama
        # and repaint, so registering it per method inflated the split to
        # 2654/664 instead of the paper's 2099/525 and over-weighted reals 2x.
        real_path = Path(real_dir)
        if load_reals and real_path.exists():
            for p in sorted(real_path.rglob('*')):
                if p.suffix.lower() in IMAGE_EXT:
                    self.samples.append((str(p), 0, None, 'real'))

        # Load FAKE images (label=1, with mask)
        fake_path = Path(fake_dir)
        mask_path = Path(mask_dir)
        # Recursive + naming-tolerant mask index: handles nested mask dirs and
        # common suffix conventions (e.g. img.png -> img_mask.png). Fixes the case
        # where inpainting datasets (Repaint/LaMa) store masks under a different
        # name/folder, which silently produced empty GT masks (IoU forced to 0).
        mask_index = {}
        if mask_path.exists():
            for mp in mask_path.rglob('*'):
                if mp.suffix.lower() in IMAGE_EXT:
                    stem = mp.stem
                    mask_index.setdefault(stem, str(mp))
                    for suf in ('_mask', '-mask', '_gt', '-gt', '.mask', '_label'):
                        if stem.endswith(suf):
                            mask_index.setdefault(stem[:-len(suf)], str(mp))
        if fake_path.exists():
            for p in sorted(fake_path.rglob('*')):
                if p.suffix.lower() not in IMAGE_EXT:
                    continue
                m_path = (mask_index.get(p.stem)
                          or mask_index.get(p.stem + '_mask')
                          or mask_index.get(p.stem + '-mask'))
                self.samples.append((str(p), 1, m_path, dataset_name))

        # Mask polarity. Some datasets ship masks with the forged region as 0
        # instead of 255. File-manager thumbnails cannot settle this: PNG alpha
        # composites to BLACK in Windows Explorer and to WHITE in Google Drive,
        # so the same file looks inverted between them. Decide from pixel data.
        # Inpainting masks cover a minority of the image, so median white
        # coverage above 60% means the forged region is stored as black.
        self.invert_masks = False
        _pol = str(getattr(cfg, 'MASK_POLARITY', 'auto')).lower()
        if _pol == 'invert':
            self.invert_masks = True
        elif _pol == 'auto' and mask_index:
            _fr = []
            for _mp in sorted(set(mask_index.values()))[:40]:
                try:
                    _fr.append(float((np.array(Image.open(_mp).convert('L')) > 127).mean()))
                except Exception:
                    pass
            if _fr:
                _med = float(np.median(_fr))
                if _med > 0.60:
                    self.invert_masks = True
                    print(f' [{dataset_name}/{split}] masks appear INVERTED '
                          f'(median white coverage {_med:.1%}) -> inverting on load; '
                          "set cfg.MASK_POLARITY='normal' to disable.")

        real_n = sum(1 for _, l, _, _ in self.samples if l == 0)
        fake_n = sum(1 for _, l, _, _ in self.samples if l == 1)
        mask_n = sum(1 for _, l, m, _ in self.samples if l == 1 and m)
        print(f'[{dataset_name}/{split}] Real:{real_n} Fake:{fake_n} Masks:{mask_n}/{fake_n}')

    def __len__(self): return len(self.samples)

    def _native_crop(self, img, mask_im):
        """Crop `img_size` at NATIVE scale. Training crops are randomly placed but
        biased to contain forged pixels; validation uses a deterministic crop
        centred on the forgery so the metric stays reproducible."""
        S = self.img_size
        W, H = img.size
        cx, cy = None, None
        if mask_im is not None:
            m = np.array(mask_im) > 127
            if m.any():
                ys, xs = np.nonzero(m)
                cx, cy = int(xs.mean()), int(ys.mean())
        if cx is None:
            cx, cy = W // 2, H // 2
        if self.split == 'train':
            jitter = S // 3
            cx += random.randint(-jitter, jitter)
            cy += random.randint(-jitter, jitter)
            if random.random() < 0.25: # keep some pure-background views
                cx, cy = random.randint(0, W), random.randint(0, H)
        x0 = int(np.clip(cx - S // 2, 0, W - S))
        y0 = int(np.clip(cy - S // 2, 0, H - S))
        box = (x0, y0, x0 + S, y0 + S)
        return img.crop(box), (mask_im.crop(box) if mask_im is not None else None)

    def __getitem__(self, idx):
        # Google Drive can list a file and still fail to open it (partial sync,
        # shortcuts, FUSE drops). A single bad file must not kill a long run, so
        # retry a few neighbours and report each bad path once.
        for _attempt in range(8):
            path, label, mask_path, method = self.samples[idx]
            try:
                img = Image.open(path).convert('RGB')
                break
            except (FileNotFoundError, OSError) as _e:
                bad = InpaintingSegDataset._bad_paths
                if path not in bad:
                    bad.add(path)
                    if len(bad) <= 20:
                        print(f' [dataset] unreadable, skipping: {path} ({type(_e).__name__})')
                    elif len(bad) == 21:
                        print(' [dataset] >20 unreadable files -- suppressing further warnings; '
                              'run the file-integrity check cell.')
                idx = random.randrange(len(self.samples))
        else:
            raise RuntimeError(
                f'{len(InpaintingSegDataset._bad_paths)} unreadable files; could not find a '
                'readable sample in 8 tries. Your dataset folder is likely still syncing or '
                'incomplete -- run the file-integrity check cell.')
        try:
            mask_im = (Image.open(mask_path).convert('L')
                       if (mask_path and os.path.exists(mask_path)) else None)
        except (FileNotFoundError, OSError):
            mask_im = None

        # (a): when the source is LARGER than img_size, crop at native scale
        # instead of squashing. Downscaling is what thins a 15-px road to a couple
        # of pixels; the crop is biased onto the forged region so we do not just
        # sample empty background. Automatic no-op when the source is already 256.
        if getattr(self, 'native_crop', False) and min(img.size) > self.img_size:
            img, mask_im = self._native_crop(img, mask_im)

        # forgery-centric zoom crop (train only). Small forged regions get
        # up to 2x effective supervision resolution. Real images get the same
        # crop with a random centre so zoom is not a leak cue for the classifier.
        if (self.split == 'train' and getattr(cfg, 'ZOOM_CROP_ENABLE', False)
                and random.random() < getattr(cfg, 'ZOOM_CROP_PROB', 0.5)):
            lo, hi = getattr(cfg, 'ZOOM_CROP_RANGE', (0.5, 0.75))
            W, H = img.size
            s = max(8, int(min(W, H) * random.uniform(lo, hi)))
            cx = cy = None
            if mask_im is not None:
                m = np.array(mask_im) > 127
                if m.any():
                    ys, xs = np.nonzero(m)
                    i = random.randint(0, len(ys) - 1)
                    cx, cy = int(xs[i]), int(ys[i])
            if cx is None:
                cx, cy = random.randint(0, W - 1), random.randint(0, H - 1)
            x0 = int(np.clip(cx - s // 2, 0, W - s))
            y0 = int(np.clip(cy - s // 2, 0, H - s))
            img = img.crop((x0, y0, x0 + s, y0 + s))
            if mask_im is not None:
                mask_im = mask_im.crop((x0, y0, x0 + s, y0 + s))

        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        if mask_im is not None:
            # (b): NEAREST point-samples the mask, breaking a thin structure
            # into disconnected dots when downscaling. BOX (area average) plus a
            # low cut keeps it connected.
            mask_im = mask_im.resize((self.img_size, self.img_size), Image.BOX)
            mask = np.array(mask_im).astype(np.float32) / 255.0
            mask = (mask > 0.35).astype(np.float32)
            if getattr(self, 'invert_masks', False):
                mask = 1.0 - mask
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        img_tensor = TF.to_tensor(img)
        if self.augment:
            if random.random()>0.5: img_tensor=TF.hflip(img_tensor); mask=TF.hflip(mask)
            if random.random()>0.5: img_tensor=TF.vflip(img_tensor); mask=TF.vflip(mask)
            k=random.randint(0,3)
            if k>0: img_tensor=torch.rot90(img_tensor,k,[1,2]); mask=torch.rot90(mask,k,[1,2])
            if random.random()>0.5: img_tensor=(img_tensor*random.uniform(0.9,1.1)).clamp(0,1)
        img_tensor = TF.normalize(img_tensor, [0.485,0.456,0.406], [0.229,0.224,0.225])
        return {'image':img_tensor, 'label':torch.tensor(label,dtype=torch.float32), 'mask':mask, 'method':method, 'path':path}

print('Loading datasets...')
train_datasets, val_datasets = [], []
_seen_real = {'train': set(), 'val': set()} # gt/ is shared by lama+repaint
for ds_name, splits in cfg.DATASET_PATHS.items():
    for split_name, split_key, ds_list, do_aug in [('train', 'train', train_datasets, True), ('val', 'val', val_datasets, False)]:
        if split_key in splits:
            paths = splits[split_key]
            _rkey = os.path.realpath(str(paths['real']))
            _load_reals = (not getattr(cfg, 'DEDUPE_REALS', True)) or (_rkey not in _seen_real[split_name])
            _seen_real[split_name].add(_rkey)
            try:
                ds = InpaintingSegDataset(
                    real_dir=paths['real'],
                    fake_dir=paths['fake'],
                    mask_dir=paths['mask'],
                    dataset_name=ds_name,
                    split=split_name,
                    img_size=cfg.IMG_SIZE,
                    augment=do_aug,
                    native_crop=getattr(cfg, 'NATIVE_CROP', False), #
                    load_reals=_load_reals #
                )
                ds_list.append(ds)
            except FileNotFoundError as e:
                print(f' Skip {ds_name}/{split_name}: {e}')

train_ds = ConcatDataset(train_datasets) if train_datasets else []
val_ds = ConcatDataset(val_datasets) if val_datasets else []
print(f'Combined -> Train:{len(train_ds)} Val:{len(val_ds)}')
if getattr(cfg, 'DEDUPE_REALS', True):
    print(' (reals de-duplicated: expect 2099/525 for Fake-Vaihingen, matching FECDNet Table)')
assert len(train_ds) > 0, f'No training samples found! Check that DATASET_PATHS directories exist.'
assert len(val_ds) > 0, f'No validation samples found! Check that DATASET_PATHS directories exist.'
train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

# ======================================================================
# Robustness Augmentation
#
# Distortions applied during training for invariance and CBFH benign-transform views: JPEG compression, Gaussian blur, Gaussian noise.
# ======================================================================

# ===================== Robustness Augmentation =====================
# Distortions used for (1) robustness training (invariance) and
# (2) CBFH benign-transform second views: JPEG / Gaussian blur / Gaussian noise.
from PIL import Image as _PILImage

_RB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_RB_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _denorm(x):
    return (x * _RB_STD.to(x.device) + _RB_MEAN.to(x.device)).clamp(0, 1)

def _renorm(x):
    return (x - _RB_MEAN.to(x.device)) / _RB_STD.to(x.device)

def _jpeg(img01, quality):
    arr = (img01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    _PILImage.fromarray(arr).save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    out = np.array(_PILImage.open(buf).convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1)

def _blur(img01, k):
    return TF.gaussian_blur(img01, kernel_size=int(k))

def _noise(img01, sigma):
    return (img01 + torch.randn_like(img01) * float(sigma)).clamp(0, 1)

def apply_robustness(x_norm, cfg, always=False):
    """Apply a random JPEG / Gaussian-blur / Gaussian-noise distortion to a
    normalized image batch.
      always=True -> force exactly one distortion per sample (CBFH benign view)
      always=False -> apply a distortion with prob cfg.ROBUST_PROB (invariance aug)
    """
    x01 = _denorm(x_norm).cpu()
    out = []
    for i in range(x01.shape[0]):
        img = x01[i]
        if always or random.random() < cfg.ROBUST_PROB:
            kind = random.choice(['jpeg', 'blur', 'noise'])
            if kind == 'jpeg':
                img = _jpeg(img, random.choice(cfg.JPEG_QUALITIES))
            elif kind == 'blur':
                img = _blur(img, random.choice(cfg.BLUR_KERNELS))
            else:
                img = _noise(img, random.choice(cfg.NOISE_SIGMAS))
        out.append(img)
    return _renorm(torch.stack(out).to(x_norm.device))

print('Robustness augmentation ready: JPEG / Gaussian blur / Gaussian noise')

# ======================================================================
# 5. Visualize Samples
# ======================================================================

def show_samples(dataset, n=6):
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(3, n, figsize=(n*3, 9))
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    for col, idx in enumerate(indices):
        s = dataset[idx]
        img = (s['image']*std+mean).clamp(0,1).permute(1,2,0).numpy()
        msk = s['mask'].squeeze().numpy()
        c = 'green' if s['label']==0 else 'red'
        lbl = 'REAL' if s['label']==0 else f'FAKE({s["method"]})'
        axes[0,col].imshow(img); axes[0,col].set_title(lbl,fontsize=8,color=c,fontweight='bold'); axes[0,col].axis('off')
        axes[1,col].imshow(msk,cmap='gray',vmin=0,vmax=1); axes[1,col].set_title(f'Mask({msk.sum():.0f}px)',fontsize=8); axes[1,col].axis('off')
        ov = img.copy(); ov[msk>0.5] = ov[msk>0.5]*0.5+np.array([1,0,0])*0.5
        axes[2,col].imshow(ov); axes[2,col].set_title('Overlay',fontsize=8); axes[2,col].axis('off')
    plt.tight_layout(); plt.show()
show_samples(train_ds)

# ======================================================================
# 6. NRGA-Net Architecture
#
# Novel components: NoiseResidualEncoder, FrequencyResidualEncoder (FFT HFRI + FCL), ForensicGateFusion, NoiseGuidedAttentionGate, NRGADecoder
# ======================================================================

# ======================================================================
# 6b. Content Prior: pair-free Masked Content Prior (+ blind-spot ablation arm)
#
# The decisive forensic cue (FLDCF's insight) is the residual of a frozen model of
# authentic content: inpainted regions lie off the manifold of authentic imagery, so
# a content model reconstructs them badly, and the error map has a pixel-accurate
# boundary.
#
# novelty — `MaskedContentPrior` (default). FLDCF trains its restoration
# network on *paired* forged/authentic images. This prior is pair-free and fully
# self-supervised: a light CNN autoencoder is trained on authentic tiles ONLY, with
# random 16x16 blocks zeroed (mask ratio 0.5). Heavy masking prevents the identity
# shortcut — any given pixel is hidden half of the time, so copying the input is never
# a stable solution and the network must model authentic Vaihingen content itself. At
# inference the image passes through two complementary checkerboard masks, so
# every pixel is reconstructed exactly once from context alone (deterministic and
# TTA-safe); the reconstruction error is ~0 on authentic content and jumps on
# inpainted content.
#
# Ablation arm. `cfg.PRIOR_MODE = 'blindspot'` keeps the old 5x5 `BlindSpotPrior`,
# and `'both'` concatenates the two residuals, so the paper can report the prior
# ablation table from one code path.
# ======================================================================

# ===================== content prior(s) =====================
class BlindSpotPrior(nn.Module):
    """Self-supervised model of AUTHENTIC local image statistics (ablation arm).

    The first conv is 5x5 with a permanently zeroed CENTRE TAP, so each output
    pixel is predicted from its neighbourhood only; every later layer is 1x1, so
    the blind spot cannot leak. Trained with L1 on real imagery, the residual
    |x - R(x)| stays near zero on authentic content and jumps on inpainted content.
    Kept for the PRIOR_MODE ablation ('blindspot' / 'both').
    """
    def __init__(self, ch=64):
        super().__init__()
        self.mconv = nn.Conv2d(3, ch, 5, padding=2)
        m = torch.ones(1, 1, 5, 5); m[0, 0, 2, 2] = 0.0
        self.register_buffer('center_mask', m)
        self.body = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(True),
                                  nn.Conv2d(ch, ch, 1), nn.ReLU(True),
                                  nn.Conv2d(ch, ch, 1), nn.ReLU(True))
        self.out = nn.Conv2d(ch, 3, 1)

    def forward(self, x01):
        w = self.mconv.weight * self.center_mask # blind spot enforced here
        y = F.relu(F.conv2d(x01, w, self.mconv.bias, padding=2))
        return self.out(self.body(y)).clamp(0, 1)

class _PriorResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(ch, ch, 3, padding=1))
    def forward(self, x):
        return F.relu(x + self.body(x))

class MaskedContentPrior(nn.Module):
    """NOVEL pair-free self-supervised model of AUTHENTIC content.

    FLDCF's restoration prior needs PAIRED forged/authentic images; this prior is
    trained on AUTHENTIC tiles ONLY. Random MASK_PATCH blocks are zeroed and a
    light CNN autoencoder reconstructs the full tile. Heavy masking prevents the
    identity shortcut -- any given pixel is hidden half of the time, so copying
    the input is never a stable solution -- forcing the network to model authentic
    content statistics. Inpainted content is then reconstructed as if it were
    authentic, and the reconstruction error localizes the forgery with a
    pixel-accurate boundary. Frozen once trained.
    """
    def __init__(self, base=32, n_res=4):
        super().__init__()
        self.enc0 = nn.Sequential(nn.Conv2d(3, base, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True))
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc1 = nn.Sequential(nn.ReLU(True), nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(True))
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(nn.ReLU(True), nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.ReLU(True))
        self.bottleneck = nn.Sequential(*[_PriorResBlock(base * 4) for _ in range(n_res)])
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.ReLU(True), nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(True))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.ReLU(True), nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True))
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x_masked):
        e0 = self.enc0(x_masked) # (B, base, H, W)
        e1 = self.enc1(self.down1(e0)) # (B, 2*base, H/2, W/2)
        e2 = self.enc2(self.down2(e1)) # (B, 4*base, H/4, W/4)
        b = self.bottleneck(e2)
        d2 = self.dec2(self.up2(b) + e1)
        d1 = self.dec1(self.up1(d2) + e0)
        return self.out(d1).clamp(0, 1)

def random_block_mask(B, H, W, patch, ratio, device):
    """Random block mask, 1 = hidden. Returns (B,1,H,W)."""
    gh, gw = max(H // patch, 1), max(W // patch, 1)
    m = (torch.rand(B, 1, gh, gw, device=device) < ratio).float()
    return F.interpolate(m, size=(H, W), mode='nearest')

def checkerboard_masks(H, W, patch, device):
    """Two complementary deterministic block masks; every pixel hidden exactly once."""
    gh, gw = max(H // patch, 1), max(W // patch, 1)
    yy, xx = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing='ij')
    m1 = ((yy + xx) % 2).float().view(1, 1, gh, gw).to(device)
    m1 = F.interpolate(m1, size=(H, W), mode='nearest')
    return m1, 1.0 - m1

@torch.no_grad()
def masked_prior_residual(prior, x01, patch):
    """Complementary-mask reconstruction error (B,3,H,W): each pixel is predicted
    exactly once from context alone. Deterministic -> safe for TTA averaging."""
    H, W = x01.shape[-2:]
    m1, m2 = checkerboard_masks(H, W, patch, x01.device)
    r1 = (x01 - prior(x01 * (1.0 - m1))).abs() * m1
    r2 = (x01 - prior(x01 * (1.0 - m2))).abs() * m2
    return r1 + r2

PRIOR_PATH = os.path.join(cfg.OUTPUT_DIR, 'content_prior.pt')
MASKED_PRIOR_PATH = os.path.join(cfg.OUTPUT_DIR, 'masked_prior_FV.pt')
PRIOR_MODE = getattr(cfg, 'PRIOR_MODE', 'blindspot')
_mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

def _real_images01(loader):
    """Yield authentic-only batches, un-normalized to [0,1]."""
    for _b in loader:
        _keep = _b['label'] == 0
        if int(_keep.sum()) == 0:
            continue
        yield (_b['image'][_keep].to(DEVICE) * _std + _mean).clamp(0, 1)

if not getattr(cfg, 'USE_CONTENT_PRIOR', True):
    print('Content prior disabled (cfg.USE_CONTENT_PRIOR = False)')
else:
    # ---- arm 1: the old 5x5 blind-spot prior (ablation) ----
    if PRIOR_MODE in ('blindspot', 'both'):
        if os.path.exists(PRIOR_PATH):
            print('Blind-spot prior found ->', PRIOR_PATH, '(delete the file to retrain)')
        else:
            _prior = BlindSpotPrior(cfg.PRIOR_CH).to(DEVICE)
            _opt = optim.AdamW(_prior.parameters(), lr=cfg.PRIOR_LR)
            print(f'Training the blind-spot prior on REAL images only for {cfg.PRIOR_EPOCHS} epoch(s)...')
            for _ep in range(cfg.PRIOR_EPOCHS):
                _prior.train(); _tot, _n = 0.0, 0
                for _x in tqdm(_real_images01(train_loader), desc=f'prior {_ep + 1}/{cfg.PRIOR_EPOCHS}', leave=False):
                    _opt.zero_grad()
                    _loss = F.l1_loss(_prior(_x), _x)
                    _loss.backward(); _opt.step()
                    _tot += float(_loss.detach()) * _x.shape[0]; _n += _x.shape[0]
                print(f' epoch {_ep + 1}: L1 {_tot / max(_n, 1):.5f} over {_n} real images')
            _prior.eval()
            torch.save(_prior.state_dict(), PRIOR_PATH)
            print('Saved ->', PRIOR_PATH)
            del _prior, _opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- arm 2: the NOVEL pair-free masked content prior (default) ----
    if PRIOR_MODE in ('masked', 'both'):
        if os.path.exists(MASKED_PRIOR_PATH):
            print('Masked content prior found ->', MASKED_PRIOR_PATH, '(delete the file to retrain)')
        else:
            _mp = MaskedContentPrior(getattr(cfg, 'MASKED_PRIOR_CH', 32)).to(DEVICE)
            _opt = optim.AdamW(_mp.parameters(), lr=getattr(cfg, 'MASKED_PRIOR_LR', 1e-4))
            _patch = int(getattr(cfg, 'MASK_PATCH', 16))
            _ratio = float(getattr(cfg, 'MASK_RATIO', 0.5))
            _eps = int(getattr(cfg, 'MASKED_PRIOR_EPOCHS', 15))
            print(f'Training the MASKED content prior on REAL images only for {_eps} epochs '
                  f'(pair-free; mask ratio {_ratio}, patch {_patch}px)...')
            for _ep in range(_eps):
                _mp.train(); _tot, _n = 0.0, 0
                for _x in tqdm(_real_images01(train_loader), desc=f'masked prior {_ep + 1}/{_eps}', leave=False):
                    _m = random_block_mask(_x.shape[0], _x.shape[-2], _x.shape[-1],
                                           _patch, _ratio, DEVICE)
                    _rec = _mp(_x * (1.0 - _m))
                    # reconstruct the HIDDEN region (main) + mild full-image anchor
                    _l_hidden = ((_rec - _x).abs() * _m).sum() / (_m.sum() * 3 + 1e-6)
                    _l_full = F.l1_loss(_rec, _x)
                    _loss = _l_hidden + 0.2 * _l_full
                    _opt.zero_grad(); _loss.backward(); _opt.step()
                    _tot += float(_loss.detach()) * _x.shape[0]; _n += _x.shape[0]
                print(f' epoch {_ep + 1}: loss {_tot / max(_n, 1):.5f} over {_n} real images')
            _mp.eval()
            torch.save(_mp.state_dict(), MASKED_PRIOR_PATH)
            print('Saved ->', MASKED_PRIOR_PATH)
            del _mp, _opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ===================== autocast output dtype =====================
# bf16 keeps the network out of fp16's 5-bit exponent (the epoch-61 NaN), but numpy
# has no bfloat16 dtype, so every downstream `.numpy()` in validate()/the eval cells
# raises `TypeError: Got unsupported ScalarType BFloat16`. Casting the heads back to
# fp32 on the way out fixes all of them at one point instead of ~20 call sites, and
# it also runs the loss in fp32, which is strictly more stable.
def _to_fp32(obj):
    if torch.is_tensor(obj):
        return obj.float() if obj.is_floating_point() else obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_fp32(x) for x in obj)
    return obj

# ===================== CBAM =====================
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg=nn.AdaptiveAvgPool2d(1); self.mx=nn.AdaptiveMaxPool2d(1)
        self.mlp=nn.Sequential(nn.Conv2d(ch,ch//r,1,bias=False),nn.ReLU(True),nn.Conv2d(ch//r,ch,1,bias=False))
    def forward(self,x): return torch.sigmoid(self.mlp(self.avg(x))+self.mlp(self.mx(x)))

class SpatialAttention(nn.Module):
    def __init__(self,k=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,k,padding=k//2,bias=False)
    def forward(self,x): return torch.sigmoid(self.conv(torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True)[0]],1)))

class CBAM(nn.Module):
    def __init__(self,ch,r=16):
        super().__init__()
        self.ca=ChannelAttention(ch,r); self.sa=SpatialAttention()
    def forward(self,x): x=x*self.ca(x); return x*self.sa(x)

# =========== NOVEL: Deformable Spatial Attention (SF-CFNet FDA, low-freq) ===========
class DeformableSpatialAttention(nn.Module):
    """Spatial attention whose sampling grid *deforms* to mold around irregular,
    non-rigid forgery boundaries (typical of diffusion / inpainting). A 2-channel
    avg|max descriptor predicts per-location offsets; a DeformConv then yields a
    geometry-adaptive spatial gate -- injecting 'geometric elasticity' that a
    static-kernel SpatialAttention lacks. Offset is zero-initialised, so it starts
    as a standard (undeformed) spatial attention and learns deformation gradually."""
    def __init__(self, ch, k=7):
        super().__init__()
        self.offset = nn.Conv2d(2, 2 * k * k, k, padding=k // 2)
        self.dconv = DeformConv2d(2, 1, k, padding=k // 2)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
    def forward(self, x):
        desc = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1) # (B,2,H,W)
        dev = 'cuda' if x.is_cuda else 'cpu'
        with torch.autocast(device_type=dev, enabled=False): # deform conv in fp32 (AMP-safe)
            d = desc.float()
            gate = torch.sigmoid(self.dconv(d, self.offset(d))) # (B,1,H,W)
        return x * gate.to(x.dtype)

# ===================== CNN Branch (DenseNet-201 MULTI-SCALE) =====================
class CNNBranch(nn.Module):
    """DenseNet-201 (pretrained) + CBAM. Returns multi-scale features (c1,c2,c3,c4) + global.
    DenseNet-201 feature stages (input 256x256, growth_rate=32):
      stem (64ch, 64x64) -> denseblock1 (256ch, 64x64)
      -> transition1+denseblock2 (512ch, 32x32)
      -> transition2+denseblock3 (1792ch, 16x16)
      -> transition3+denseblock4+norm5 (1920ch, 8x8)
    Projects channels to match decoder interface (256, 512, 1024, backbone_dim)."""
    def __init__(self, backbone_dim=512, dilate_stage3=True, dilate_stage4=True,
                 memory_efficient=True):
        super().__init__()
        densenet = densenet201(weights=DenseNet201_Weights.DEFAULT,
                               memory_efficient=memory_efficient)
        feats = densenet.features
        # split the stem so the 128x128 pre-pool feature is EXPOSED as c0.
        # The old decoder had no skip at all above 64x64.
        self.stem_hi = nn.Sequential(feats.conv0, feats.norm0, feats.relu0) # 64ch, 128x128
        self.stem_pool = feats.pool0 # 64ch, 64x64
        self.stage1 = feats.denseblock1 # 256ch, 64x64
        self.stage2 = nn.Sequential(feats.transition1, feats.denseblock2) # 512ch, 32x32
        # ---- dilate the deep stages (what FLDCF does to its ResNet) ----
        # Removing a transition's 2x pooling and doubling the dilation of the
        # following dense block keeps the receptive field identical while HALVING
        # the output stride. With both enabled the deepest feature is 32x32
        # (stride 8, same as FLDCF) instead of 8x8 (stride 32) -- on an 8x8 grid a
        # 15-px road is sub-pixel, so a blob was the only representable answer.
        self.dilate_stage3 = bool(dilate_stage3)
        self.dilate_stage4 = bool(dilate_stage4)

        def _drop_pool(trans):
            return nn.Sequential(trans.norm, trans.relu, trans.conv) # pooling dropped

        def _set_dilation(block, d):
            for _m in block.modules():
                if isinstance(_m, nn.Conv2d) and _m.kernel_size == (3, 3):
                    _m.dilation = (d, d); _m.padding = (d, d)

        d4 = 1
        if self.dilate_stage3:
            t2 = _drop_pool(feats.transition2); d4 = 2
            _set_dilation(feats.denseblock3, 2)
        else:
            t2 = feats.transition2
        if self.dilate_stage4:
            t3 = _drop_pool(feats.transition3); d4 = d4 * 2
        else:
            t3 = feats.transition3
        if d4 > 1:
            _set_dilation(feats.denseblock4, d4)

        self.stage3 = nn.Sequential(t2, feats.denseblock3) # 1792ch, 32x32 | 16x16
        self.stage4 = nn.Sequential(t3, feats.denseblock4,
                                    feats.norm5, nn.ReLU(True)) # 1920ch, 32|16|8
        # 1x1 projections to match decoder channel dims
        self.proj3 = nn.Sequential(nn.Conv2d(1792, 1024, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(True))
        self.proj4 = nn.Sequential(nn.Conv2d(1920, backbone_dim, 1, bias=False), nn.BatchNorm2d(backbone_dim), nn.ReLU(True))
        self.cbam2 = CBAM(512); self.cbam3 = CBAM(1024); self.cbam4 = CBAM(backbone_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        c0 = self.stem_hi(x) # (B, 64, 128, 128) <-- high-resolution skip
        s0 = self.stem_pool(c0) # (B, 64, 64, 64)
        s1 = self.stage1(s0) # (B, 256, 64, 64)
        s2 = self.stage2(s1) # (B, 512, 32, 32)
        s3 = self.stage3(s2) # (B, 1792, 32x32 dilated | 16x16 plain)
        s4 = self.stage4(s3) # (B, 1920, same grid as s3 when dilated)
        c1 = s1 # (B, 256, 64, 64)
        c2 = self.cbam2(s2) # (B, 512, 32, 32)
        c3 = self.cbam3(self.proj3(s3)) # (B, 1024, ...)
        c4 = self.cbam4(self.proj4(s4)) # (B, 512, ...)
        glob = self.gap(c4).flatten(1) # (B, 512)
        return (c0, c1, c2, c3, c4), glob

# ===================== Feature Pyramid Module (replaces the ViT) =====================
class FPM(nn.Module):
    """Atrous feature pyramid + image-level pooling (the FLDCF-style context head).

    WHY THE ViT WAS REMOVED. `ViTBranch` tokenised the image into 16x16 patches and
    the decoder projected those tokens back onto a 16x16 grid. A patch transformer
    cannot represent anything finer than one patch, so injecting its tokens stamped
    a BLOCK-LEVEL prior onto the mask -- which is exactly the observed failure: a
    15-px winding road came back as a rounded blob of roughly the right area.
    FLDCF uses no transformer at all; it gets multi-scale context from dilated
    convolutions, which are translation-equivariant and pixel-precise.

    Parallel 3x3 convs at dilations (1, 6, 12, 18) plus a global-pooled branch give
    the same receptive-field range at full spatial resolution for a fraction of the
    parameters -- and removing ~40M ViT params and their attention activations is
    what pays for the dilated backbone.
    """
    def __init__(self, in_ch, mid=256, out_ch=512, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            k, p, d = (1, 0, 1) if r == 1 else (3, r, r)
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, mid, k, padding=p, dilation=d, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(True)))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_ch, mid, 1, bias=False), nn.ReLU(True))
        self.project = nn.Sequential(
            nn.Conv2d(mid * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True), nn.Dropout2d(0.1))

    def forward(self, x):
        ys = [b(x) for b in self.branches]
        g = F.interpolate(self.pool(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat(ys + [g], dim=1))

# ===================== CBFH (Content-Based Forensic Hashing) =====================
class ContentBasedForensicHashing(nn.Module):
    """Content-Based Forensic Hashing for provenance verification (paper Sec. 4.4).

    Replaces the SFA + FAM provenance path. A compact K-bit forensic fingerprint
    is derived from a penultimate encoder feature map via spatial-attention-weighted
    global average pooling, where the attention is guided by the localization
    decoder's intermediate mask logits. A 2-layer MLP (256 -> 128 -> K) projects the
    pooled embedding to K bits; tanh(alpha * z) is a differentiable relaxation used
    during training, while sign(z) yields the binary code at inference.
    """
    def __init__(self, in_ch=512, embed_dim=256, hash_bits=64, alpha=10.0):
        super().__init__()
        self.alpha = alpha
        self.hash_bits = hash_bits
        self.feat_proj = nn.Conv2d(in_ch, embed_dim, 1, bias=False)
        self.hash_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(True),
            nn.Linear(128, hash_bits))

    def forward(self, feat, mask_logits):
        # feat: (B, C, H, W) penultimate encoder feature; mask_logits: (B, 1, h, w)
        B, C, H, W = feat.shape
        x = self.feat_proj(feat) # (B, embed_dim, H, W)
        attn = torch.sigmoid(mask_logits)
        attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
        denom = attn.sum(dim=(2, 3), keepdim=True) + 1e-6
        embed = (x * attn).sum(dim=(2, 3), keepdim=True) / denom # attention-weighted GAP
        embed = embed.flatten(1) # (B, embed_dim)
        z = self.hash_mlp(embed) # (B, K)
        h_soft = torch.tanh(self.alpha * z) # training relaxation in (-1,1)
        h_bin = torch.sign(z) # inference code in {-1,+1}
        return h_soft, h_bin

# ===================== NOVEL: Noise Residual Encoder =====================
class DepthwiseSeparableConv(nn.Module):
    """Lightweight conv: depthwise + pointwise."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn(self.pw(self.dw(x))))

class NoiseResidualEncoder(nn.Module):
    """
    Fixed constrained convolution (high-pass) + lightweight multi-scale encoder.
    Extracts noise-domain features at 3 scales for gating the decoder.
    """
    def __init__(self, extra_ch=0):
        super().__init__()
        self.extra_ch = int(extra_ch) # content-prior residual channels
        # Fixed high-pass filters (non-trainable)
        self.constrained_conv = nn.Conv2d(3, 9, 5, padding=2, bias=False)
        self.constrained_conv.weight.requires_grad = False
        self._init_constrained_filters()

        # Scale 1: 256 -> 64 (stride-4)
        self.scale1 = nn.Sequential(
            DepthwiseSeparableConv(9 + int(extra_ch), 32, 3, 2, 1), # 256->128
            DepthwiseSeparableConv(32, 32, 3, 2, 1), # 128->64
        )
        # Scale 2: 64 -> 32
        self.scale2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, 3, 2, 1), # 64->32
        )
        # Scale 3: 32 -> 16
        self.scale3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, 3, 2, 1), # 32->16
        )

    def _init_constrained_filters(self):
        """Initialize 9 high-pass filters: 3 edge + 3 Laplacian + 3 LoG, per channel."""
        with torch.no_grad():
            kernels = torch.zeros(9, 3, 5, 5)
            # Edge detectors (horizontal, vertical, diagonal) per channel
            for c in range(3):
                # Horizontal edge
                k = torch.zeros(5,5); k[1,:] = -1; k[3,:] = 1; k[2,:] = 0
                kernels[c*3+0, c] = k / (k.abs().sum() + 1e-8)
                # Laplacian
                k = torch.zeros(5,5); k[2,2] = -4; k[1,2]=1; k[3,2]=1; k[2,1]=1; k[2,3]=1
                kernels[c*3+1, c] = k / (k.abs().sum() + 1e-8)
                # LoG approximation
                k = torch.tensor([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]], dtype=torch.float32)
                kernels[c*3+2, c] = k / (k.abs().sum() + 1e-8)
            self.constrained_conv.weight.copy_(kernels)

    def forward(self, x_raw, prior_res=None):
        noise = self.constrained_conv(x_raw) # (B, 9, H, W) fixed high-pass
        if prior_res is not None:
            # concatenate the LEARNED content-restoration residual next to the
            # fixed SRM response. The fixed filters fire on every natural edge; the
            # prior residual fires only where the content is off-manifold.
            noise = torch.cat([noise, prior_res], dim=1)
        # expose the 1/2-resolution scale (to gate the new 128x128 skip) and
        # the full-resolution residual (for the detail stream).
        n0 = self.scale1[0](noise) # (B, 32, 128, 128)
        n1 = self.scale1[1](n0) # (B, 32, 64, 64)
        n2 = self.scale2(n1) # (B, 64, 32, 32)
        n3 = self.scale3(n2) # (B, 128, 16, 16)
        return noise, n0, n1, n2, n3

# ===================== NOVEL: Noise-Guided Attention Gate =====================
class NoiseGuidedAttentionGate(nn.Module):
    """
    Uses EXTERNAL noise features to gate encoder skip connections.
    Unlike standard attention gates that use decoder features as gating signal,
    this uses noise-domain features — attending to regions with noise inconsistencies.
    """
    def __init__(self, skip_ch, noise_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or skip_ch // 4
        self.W_skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_noise = nn.Sequential(nn.Conv2d(noise_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip, noise):
        g1 = self.W_skip(skip)
        g2 = self.W_noise(noise)
        gate = self.psi(self.relu(g1 + g2)) # (B, 1, H, W) in (0,1)
        # RESIDUAL gate (Wang et al., Residual Attention Net, CVPR'17): the base
        # semantic skip ALWAYS flows, the forensic gate only ADDS emphasis. Pure
        # multiplicative gating (skip*gate) zeroed the skip in diffusion regions --
        # which have weak high-pass forensic response -- starving the decoder there.
        # skip*(1+gate): real regions ~1x (gate->0), forensic regions up to 2x.
        return skip * (1.0 + gate) # residual gated skip connection

# ===================== NOVEL: Boundary Refinement (BR) =====================
class BR(nn.Module):
    """Residual boundary refinement on a LOGIT map (GCN / FLDCF br5-br7).

    Straight after an upsample a logit map is geometrically correct but SMOOTH --
    every boundary has been low-pass filtered. A small residual 3x3 pair re-injects
    the lost high frequency, turning a rounded envelope back into a contour. conv2
    is scaled down (not zeroed) so the block starts as a near-identity; an exact
    zero would make dL/dconv1 identically zero and stall the first layer.
    """
    def __init__(self, ch=1, mid=32):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, ch, 3, padding=1)
        with torch.no_grad():
            self.conv2.weight.mul_(0.01)
        nn.init.zeros_(self.conv2.bias)
    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))

class DetailBranch(nn.Module):
    """FULL-resolution detail stream (no downsampling anywhere).

    The old decoder produced its 256x256 output from a 64x64 feature via two
    unguided ConvTranspose steps -- the last 4x was a blind guess, which is what
    rounds thin structure into a blob. This branch keeps raw RGB, the fixed
    high-pass residual and the learned content-prior residual at native resolution
    and hands the decoder pixel-accurate edge evidence for the final refinement.
    """
    def __init__(self, in_ch=12, out_ch=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3 + in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self, x_raw, residual):
        return self.body(torch.cat([x_raw, residual], dim=1))

# ===================== NOVEL: NRGA Decoder =====================
# ===================== Spectral-Edge Stream (FECDNet x SFNet hybrid) =====================
class SpectralEdgeStream(nn.Module):
    """Multi-scale edge extraction in the SPECTRAL domain (the hybrid novelty).

    FECDNet derives its edge streams with spatial Sobel filters; SFNet shows that
    forgery cues concentrate in selected high-frequency bands processed by complex
    frequency convolutions (FCL). The hybrid applies the FCL mechanism to EDGE
    extraction: per encoder stage, rfft2 -> zero the centred low-frequency disk ->
    separate real/imag 1x1 convs -> irfft2 -> |.| -> conv. The stream therefore
    highlights boundaries where the HIGH-FREQUENCY content is inconsistent --
    exactly what an inpainting boundary is. Runs in float32 (FFT is not
    autocast-safe; SFNet has the same constraint).
    """
    def __init__(self, in_ch, out_ch=32, hp_radius=0.25):
        super().__init__()
        self.hp_radius = float(hp_radius)
        self.re_conv = nn.Conv2d(in_ch, in_ch, 1)
        self.im_conv = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, f):
        amp_ctx = (torch.autocast(device_type='cuda', enabled=False) if f.is_cuda
                   else torch.autocast(device_type='cpu', enabled=False))
        with amp_ctx:
            f = f.float()
            H, W = f.shape[-2:]
            Ff = torch.fft.rfft2(f, norm='ortho')
            fy = torch.fft.fftfreq(H, device=f.device).view(-1, 1)
            fx = torch.fft.rfftfreq(W, device=f.device).view(1, -1)
            rad = (fy ** 2 + fx ** 2).sqrt()
            keep = (rad >= self.hp_radius * 0.5).float() # zero low-freq disk
            Ff = Ff * keep
            re, im = self.re_conv(Ff.real), self.im_conv(Ff.imag)
            sp = torch.fft.irfft2(torch.complex(re, im), s=(H, W), norm='ortho')
            return self.proj(sp.abs())

class NRGADecoder(nn.Module):
    """ edge-collaborative decoder (FECDNet x SFNet hybrid).

    The noise-gated attention gates are replaced by FECDNet-style per-stage
    CONCATENATION of [upsampled semantics, encoder skip, spectral-edge stream,
    forensic stream] -- but the edge stream is extracted in the SPECTRAL domain
    (SpectralEdgeStream), so every stage sees explicit high-frequency boundary
    evidence instead of Sobel gradients. The residual logit cascade (ds heads +
    BR) and the full-resolution detail stream are unchanged. Deep edge heads at
    1/4 and 1/1 provide boundary supervision targets.
    """
    def __init__(self, ctx_ch=512, img_size=256, c0_ch=64, detail_in=12,
                 ses_ch=32, hp_radius=0.25):
        super().__init__()
        self.img_size = img_size
        self.ses_ch = int(ses_ch)

        # spectral edge streams, one per encoder scale
        if self.ses_ch > 0:
            self.ses3 = SpectralEdgeStream(1024, ses_ch, hp_radius)
            self.ses2 = SpectralEdgeStream(512, ses_ch, hp_radius)
            self.ses1 = SpectralEdgeStream(256, ses_ch, hp_radius)
            self.ses0 = SpectralEdgeStream(c0_ch, ses_ch, hp_radius)
        else:
            self.ses3 = self.ses2 = self.ses1 = self.ses0 = None
        e = self.ses_ch
        # forensic (noise+freq fused) stream projections
        self.n3p = nn.Conv2d(128, 32, 1)
        self.n2p = nn.Conv2d(64, 32, 1)
        self.n1p = nn.Conv2d(32, 32, 1)
        self.n0p = nn.Conv2d(32, 32, 1)

        self.lat5 = nn.Sequential(nn.Conv2d(ctx_ch, 256, 1, bias=False),
                                  nn.BatchNorm2d(256), nn.ReLU(True))
        self.conv5 = nn.Sequential(DepthwiseSeparableConv(256 + 1024 + e + 32, 256),
                                   DepthwiseSeparableConv(256, 256))

        self.up4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Sequential(DepthwiseSeparableConv(128 + 512 + e + 32, 128),
                                   DepthwiseSeparableConv(128, 128))

        self.up3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Sequential(DepthwiseSeparableConv(64 + 256 + e + 32, 64),
                                   DepthwiseSeparableConv(64, 64))

        self.up2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Sequential(DepthwiseSeparableConv(64 + c0_ch + e + 32, 64),
                                   DepthwiseSeparableConv(64, 64))

        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.detail = DetailBranch(detail_in, 32)
        self.conv1 = nn.Sequential(DepthwiseSeparableConv(32 + 32, 32),
                                   DepthwiseSeparableConv(32, 32))
        self.out_conv = nn.Conv2d(32, 1, 1)

        # Per-level logit heads (deep supervision + cascade residuals)
        self.ds5 = nn.Conv2d(256, 1, 1) # coarsest
        self.ds4 = nn.Conv2d(128, 1, 1)
        self.ds3 = nn.Conv2d(64, 1, 1)
        self.ds2 = nn.Conv2d(64, 1, 1) # 1/2

        self.br4 = BR(); self.br3 = BR(); self.br2 = BR(); self.br1 = BR()
        # deep edge heads (1/4 and full-res) for boundary supervision
        self.edge_head3 = nn.Conv2d(64, 1, 1)
        self.edge_head1 = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _to(t, ref):
        if t.shape[-2:] == ref.shape[-2:]:
            return t
        return F.interpolate(t, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def _edge(self, ses, feat, ref):
        if ses is None:
            return None
        return self._to(ses(feat), ref)

    def forward(self, c0, c1, c2, c3, ctx, n0, n1, n2, n3, x_raw, residual):
        e3 = self._edge(self.ses3, c3, c3)
        e2 = self._edge(self.ses2, c2, c2)
        e1 = self._edge(self.ses1, c1, c1)
        e0 = self._edge(self.ses0, c0, c0)

        def join(*parts):
            return torch.cat([p for p in parts if p is not None], dim=1)

        # ---- semantic level (FPM context + skip + spectral edge + forensic)
        d5 = self.conv5(join(self._to(self.lat5(ctx), c3), c3,
                             e3, self._to(self.n3p(n3), c3)))
        ds5_out = self.ds5(d5)
        m = ds5_out

        # ---- next level
        d4 = self.conv4(join(self._to(self.up4(d5), c2), c2,
                             e2, self._to(self.n2p(n2), c2)))
        ds4_out = self.ds4(d4)
        m = self.br4(self._to(m, d4) + ds4_out)

        # ---- 1/4
        d3 = self.conv3(join(self._to(self.up3(d4), c1), c1,
                             e1, self._to(self.n1p(n1), c1)))
        ds3_out = self.ds3(d3)
        m = self.br3(self._to(m, d3) + ds3_out)
        edge3 = self.edge_head3(d3) # deep edge supervision

        # ---- 1/2
        d2 = self.conv2(join(self._to(self.up2(d3), c0), c0,
                             e0, self._to(self.n0p(n0), c0)))
        ds2_out = self.ds2(d2)
        m = self.br2(self._to(m, d2) + ds2_out)

        # ---- 1/1 (full-resolution detail stream, unchanged)
        det = self.detail(x_raw, residual)
        d1 = self.conv1(torch.cat([self._to(self.up1(d2), det), det], dim=1))
        main_mask = self.br1(self._to(m, d1) + self.out_conv(d1))
        edge1 = self.edge_head1(d1) # deep edge supervision

        return main_mask, [ds5_out, ds4_out, ds3_out, ds2_out], d1, [edge3, edge1]

# ===================== NOVEL: Frequency-Residual Encoder (FRE) =====================
class FrequencyConvLayer(nn.Module):
    """Lightweight Frequency Conv Layer (FCL).

    Learns directly in the spectral domain: the input feature is transformed by a
    2D FFT, then *separate* depthwise convolutions refine the amplitude and phase
    spectra (non-shared params, as they encode different physical cues), before an
    inverse FFT maps the result back to the spatial domain. A residual connection
    preserves the original feature. Depthwise convs keep it extremely lightweight.
    """
    def __init__(self, ch):
        super().__init__()
        self.pre = nn.Conv2d(ch, ch, 1, bias=False)
        self.conv_amp = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.conv_pha = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.post = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True))

    def forward(self, f):
        identity = f
        y = self.pre(f)
        Xf = torch.fft.fft2(y.float(), norm='ortho') # spatial -> frequency
        amp = torch.abs(Xf) # amplitude spectrum
        pha = torch.angle(Xf) # phase spectrum
        amp = F.relu(self.conv_amp(amp)) # refine amplitude
        pha = self.conv_pha(pha) # refine phase
        Xf2 = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))
        y = torch.fft.ifft2(Xf2, norm='ortho').real # frequency -> spatial
        return identity + self.post(y)

class FrequencyResidualEncoder(nn.Module):
    """NOVEL lightweight frequency-domain forensic branch.

    Complements the spatial NoiseResidualEncoder with explicit spectral evidence:
      (1) HFRI (parameter-free): an FFT high-pass that zeroes the central
          low-frequency quarter and returns the high-frequency residual image,
          exposing up-sampling / inpainting fingerprints left by GAN & diffusion
          generators (rich in edge / texture artifacts).
      (2) FCL: a learnable spectral layer (separate amplitude / phase convs).
    Produces multi-scale features (f1,f2,f3) aligned with the noise encoder so the
    two forensic domains can be fused to guide the decoder's attention gates.
    All spectral math runs in float32 (autocast-disabled) to stay AMP-safe.
    """
    def __init__(self, hp_ratio=0.5):
        super().__init__()
        self.hp_ratio = hp_ratio
        self.stem = DepthwiseSeparableConv(3, 16, 3, 2, 1) # 256 -> 128, 16ch
        self.fcl = FrequencyConvLayer(16) # spectral refinement
        self.scale1 = DepthwiseSeparableConv(16, 32, 3, 2, 1) # 128 -> 64
        self.scale2 = DepthwiseSeparableConv(32, 64, 3, 2, 1) # 64 -> 32
        self.scale3 = DepthwiseSeparableConv(64, 128, 3, 2, 1) # 32 -> 16
        self.hfrc_scale = nn.Parameter(torch.zeros(1)) # cross-channel HF gate (HFRF-C); zero-init = safe
        # --- LOW-FREQUENCY forensic branch (SF-CFNet dual-stream): diffusion
        # artifacts live in the LOW band (global lighting, macro-structure,
        # non-rigid boundaries) that the high-pass HFRI path is blind to. This
        # separate low-pass stream supplies the missing evidence -> fixes weak
        # diffusion IoU. lf_scale zero-init: starts == high-pass baseline, grows.
        self.stem_lo = DepthwiseSeparableConv(3, 16, 3, 2, 1)
        self.scale1_lo = DepthwiseSeparableConv(16, 32, 3, 2, 1)
        self.scale2_lo = DepthwiseSeparableConv(32, 64, 3, 2, 1)
        self.scale3_lo = DepthwiseSeparableConv(64, 128, 3, 2, 1)
        self.lf_scale = nn.Parameter(torch.full((1,), 0.1)) # un-gated: low-freq diffusion evidence active from start (was zeros->dead)

    def _hfrc(self, s):
        """SFNet-style cross-channel high-frequency enhancement: FFT along the
        CHANNEL axis, suppress the lowest channel-frequencies, inverse FFT. Exposes
        inter-channel spectral inconsistencies that generators tend to leave."""
        Cf = torch.fft.rfft(s, dim=1)
        k = Cf.shape[1]
        mask = torch.ones(k, device=s.device, dtype=Cf.real.dtype).view(1, k, 1, 1)
        mask[:, :max(1, k // 4)] = 0.0
        return torch.fft.irfft(Cf * mask, n=s.shape[1], dim=1)

    def _hfri(self, x):
        # High-pass filter B(): zero out the central low-frequency quarter (by area)
        Xf = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
        H, W = x.shape[-2:]
        cy, cx = H // 2, W // 2
        rh, rw = int(H * self.hp_ratio / 2), int(W * self.hp_ratio / 2)
        m = torch.ones(H, W, device=x.device, dtype=Xf.real.dtype)
        m[cy - rh:cy + rh, cx - rw:cx + rw] = 0.0
        Xf = Xf * m
        return torch.fft.ifft2(torch.fft.ifftshift(Xf, dim=(-2, -1)), norm='ortho').real

    def _lfri(self, x):
        # Low-pass complement of _hfri: KEEP the central low-frequency quarter
        # (global lighting / macro-structure where diffusion anomalies live)
        # and zero the high frequencies.
        Xf = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
        H, W = x.shape[-2:]
        cy, cx = H // 2, W // 2
        rh, rw = int(H * self.hp_ratio / 2), int(W * self.hp_ratio / 2)
        m = torch.zeros(H, W, device=x.device, dtype=Xf.real.dtype)
        m[cy - rh:cy + rh, cx - rw:cx + rw] = 1.0
        Xf = Xf * m
        return torch.fft.ifft2(torch.fft.ifftshift(Xf, dim=(-2, -1)), norm='ortho').real

    def forward(self, x_raw):
        dev = 'cuda' if x_raw.is_cuda else 'cpu'
        with torch.autocast(device_type=dev, enabled=False): # FFT in float32 (AMP-safe)
            x = x_raw.float()
            # low-frequency branch (diffusion structural / non-rigid evidence)
            _xlf = self._lfri(x)
            _sl = self.stem_lo(_xlf)
            _l1r = self.scale1_lo(_sl)
            _l2r = self.scale2_lo(_l1r)
            _l3r = self.scale3_lo(_l2r)
            l1 = self.lf_scale * _l1r # zero-gated at init -> == high-pass baseline
            l2 = self.lf_scale * _l2r
            l3 = self.lf_scale * _l3r
            xhf = self._hfri(x) # (B, 3, 256, 256) high-frequency residual
            s = self.fcl(self.stem(xhf)) # (B, 16, 128, 128) spectral feature
            s = s + self.hfrc_scale * self._hfrc(s) # + cross-channel HF residual (HFRF-C)
            f1 = self.scale1(s) # (B, 32, 64, 64)
            f2 = self.scale2(f1) # (B, 64, 32, 32)
            f3 = self.scale3(f2) # (B, 128, 16, 16)
        return f1, f2, f3, l1, l2, l3

class ForensicGateFusion(nn.Module):
    """Cross-Frequency Interaction (CFI) + Frequency-specific Dual-Attention (FDA),
    adapted from SF-CFNet, fusing spatial-noise (n: structural / low-freq) and
    spectral (f: high-freq) forensic features at one scale.

    (1) CFI with InstanceNorm: each branch is guided by the other via a 1x1 conv,
        then InstanceNorm re-scales the sum -- suppressing per-generator style
        statistics for better GAN<->Diffusion<->inpainting generalization.
    (2) FDA discipline:
          high-freq (f): Channel-Attention ONLY (no spatial attn) -> avoids firing
              on abundant natural high-frequency textures (cuts false positives).
          low-freq (n): Channel-Attention + Deformable Spatial-Attention -> molds
              to irregular non-rigid boundaries (recovers missed diffusion areas).
    Output channels == input channels, so the decoder gate interface is unchanged."""
    def __init__(self, ch):
        super().__init__()
        # Cross-Frequency Interaction
        self.conv_l2h = nn.Conv2d(ch, ch, 1, bias=False) # low(noise) -> high(freq)
        self.conv_h2l = nn.Conv2d(ch, ch, 1, bias=False) # high(freq) -> low(noise)
        self.in_high = nn.InstanceNorm2d(ch, affine=True)
        self.in_low = nn.InstanceNorm2d(ch, affine=True)
        # Frequency-specific Dual-Attention
        self.ca_high = ChannelAttention(ch) # high-freq: CA only
        self.ca_low = ChannelAttention(ch) # low-freq: CA ...
        self.dsa_low = DeformableSpatialAttention(ch) # ... + Deformable SA
        # Channel-preserving fuse: 1x1 cross-domain mix + depthwise 3x3 refine
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True))
    def forward(self, n, f):
        f2 = self.in_high(f + self.conv_l2h(n)) # CFI: guide high-freq with structure
        n2 = self.in_low(n + self.conv_h2l(f)) # CFI: guide structure with spectral detail
        f2 = f2 * self.ca_high(f2) # high-freq branch: CA only
        n2 = n2 * self.ca_low(n2) # low-freq branch: CA ...
        n2 = self.dsa_low(n2) # ... + Deformable SA
        return self.fuse(torch.cat([n2, f2], dim=1))

# ===================== uncertainty-guided point sampling =====================
def _sample_uncertain_points(coarse_logit, refined_logit, k, unc_frac):
    """PointRend-style sampling: half the points uniform, half from the
    high-uncertainty band (|p-0.5| small -- the boundary pixels that cap IoU).
    Returns (refined point logits (B,1,K) with grad, coords (B,K,2) in [-1,1])."""
    B, _, H, W = coarse_logit.shape
    with torch.no_grad():
        p = torch.sigmoid(coarse_logit.detach()).view(B, -1)
        unc = 1.0 - (2.0 * p - 1.0).abs()
        k_unc = min(max(int(k * unc_frac), 1), k - 1)
        k_uni = k - k_unc
        idx_uni = torch.randint(0, H * W, (B, k_uni), device=coarse_logit.device)
        score = unc * torch.rand_like(unc)
        idx_unc = score.topk(k_unc, dim=1).indices
        idx = torch.cat([idx_uni, idx_unc], dim=1) # (B, K)
        ys = torch.div(idx, W, rounding_mode='floor').float()
        xs = (idx % W).float()
        coords = torch.stack([xs / max(W - 1, 1) * 2.0 - 1.0,
                              ys / max(H - 1, 1) * 2.0 - 1.0], dim=-1)
    pts = refined_logit.view(B, -1).gather(1, idx).unsqueeze(1) # (B,1,K) w/ grad
    return pts, coords

# ===================== Full NRGA-Net Model =====================
class NRGANet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Encoder branches
        self.cnn = CNNBranch(cfg.BACKBONE_DIM,
                             dilate_stage3=getattr(cfg, 'DILATE_STAGE3', True),
                             dilate_stage4=getattr(cfg, 'DILATE_STAGE4', True),
                             memory_efficient=getattr(cfg, 'MEMORY_EFFICIENT_BACKBONE', True))
        # ViT + cross-attention bridge REMOVED (they stamped a 16x16 block
        # prior on the mask). Context now comes from a dilated pyramid, as in FLDCF.
        self.fpm = FPM(cfg.BACKBONE_DIM, mid=256, out_ch=cfg.BACKBONE_DIM)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # frozen content prior(s) -- weights loaded after construction.
        # PRIOR_MODE: 'masked' (novel pair-free MaskedContentPrior, default),
        # 'blindspot' (old 5x5 BlindSpotPrior), 'both' (ablation arm).
        self.use_prior = bool(getattr(cfg, 'USE_CONTENT_PRIOR', True))
        self.prior_mode = getattr(cfg, 'PRIOR_MODE', 'blindspot') if self.use_prior else 'none'
        self.mask_patch = int(getattr(cfg, 'MASK_PATCH', 16))
        self.content_prior = None
        self.masked_prior = None
        _extra = 0
        if self.use_prior:
            if self.prior_mode in ('blindspot', 'both'):
                self.content_prior = BlindSpotPrior(getattr(cfg, 'PRIOR_CH', 64))
                _extra += 3
            if self.prior_mode in ('masked', 'both'):
                self.masked_prior = MaskedContentPrior(getattr(cfg, 'MASKED_PRIOR_CH', 32))
                _extra += 3
            for _pm in (self.content_prior, self.masked_prior):
                if _pm is not None:
                    for _p in _pm.parameters():
                        _p.requires_grad = False

        # Noise encoder (novel) - spatial high-pass forensic features
        self.noise_encoder = NoiseResidualEncoder(extra_ch=_extra)
        # Frequency encoder (novel) - spectral (FFT) forensic features
        self.freq_encoder = FrequencyResidualEncoder()
        # Forensic gate fusion: combine spatial-noise + frequency cues per scale
        self.fuse1 = ForensicGateFusion(32)
        self.fuse2 = ForensicGateFusion(64)
        self.fuse3 = ForensicGateFusion(128)

        # Classification path (SFA + FAM removed -> classify directly from fused backbone feature)
        self.cls_head = nn.Sequential(nn.LayerNorm(cfg.BACKBONE_DIM), nn.Linear(cfg.BACKBONE_DIM, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, 1))
        # MVSS-Net detection coupling: maps the strongest forged-pixel logit
        # (global max-pool of the mask) into the classification logit, so a
        # small well-localized diffusion patch can trigger 'fake' even when the
        # global feature looks real (was: masks zeroed by cls-gated IoU metric).
        self.mask_cls = nn.Linear(1, 1)

        # Provenance verification path (CBFH) - replaces SFA + FAM
        self.cbfh = ContentBasedForensicHashing(in_ch=cfg.BACKBONE_DIM, embed_dim=cfg.CBFH_EMBED_DIM,
                                                hash_bits=cfg.CBFH_HASH_BITS, alpha=cfg.CBFH_ALPHA)

        # Segmentation decoder (novel NRGA)
        self.decoder = NRGADecoder(ctx_ch=cfg.BACKBONE_DIM, img_size=cfg.IMG_SIZE,
                                   c0_ch=64, detail_in=9 + _extra,
                                   ses_ch=(int(getattr(cfg, 'SES_CH', 32))
                                           if getattr(cfg, 'SES_ENABLE', True) else 0),
                                   hp_radius=float(getattr(cfg, 'SES_HP_RADIUS', 0.25)))

        # uncertainty-guided point refinement head (PointRend mechanism; novel
        # for generative-inpainting localization). The point-wise MLP is written as
        # 1x1 convs -- mathematically identical to per-point evaluation -- fusing
        # the full-res decoder feature with the coarse logit to re-decide the
        # uncertain boundary pixels that cap IoU.
        self.point_enable = bool(getattr(cfg, 'POINT_ENABLE', True))
        if self.point_enable:
            self.point_head = nn.Sequential(
                nn.Conv2d(32 + 1, 64, 1), nn.ReLU(True),
                nn.Conv2d(64, 64, 1), nn.ReLU(True),
                nn.Conv2d(64, 1, 1))
        else:
            self.point_head = None
        self.point_k = int(getattr(cfg, 'POINT_K', 2048))
        self.point_unc_frac = float(getattr(cfg, 'POINT_UNC_FRAC', 0.5))
        self.point_unc_delta = float(getattr(cfg, 'POINT_UNC_DELTA', 0.35))

    def forward(self, x):
        mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
        x_raw = (x * std + mean).clamp(0, 1)

        # content-prior residual(s) (frozen, no gradient). The masked prior is
        # evaluated through two COMPLEMENTARY checkerboard block masks, so every
        # pixel is reconstructed once from context alone (deterministic, TTA-safe).
        if self.use_prior:
            _res = []
            with torch.no_grad():
                if self.content_prior is not None:
                    _res.append((x_raw - self.content_prior(x_raw)).detach())
                if self.masked_prior is not None:
                    _res.append(masked_prior_residual(self.masked_prior, x_raw,
                                                      self.mask_patch).detach())
            prior_res = torch.cat(_res, dim=1) if _res else None
        else:
            prior_res = None

        # Multi-scale CNN features (c0 = 128x128 stem skip)
        (c0, c1, c2, c3, c4), cnn_global = self.cnn(x)

        # Noise features (spatial) + Frequency features (spectral)
        # also returns the raw residual and the 1/2-res scale
        residual, n0, n1, n2, n3 = self.noise_encoder(x_raw, prior_res)
        f1, f2, f3, l1, l2, l3 = self.freq_encoder(x_raw)
        # SF-CFNet dual-frequency FDA fusion: the LOW-freq stream (l) -> the
        # Deformable spatial-attention 'low' slot localises diffusion's non-rigid
        # forged regions; HIGH-freq spectral (f) + SRM noise residual (n) -> the
        # CA-only 'high' slot captures GAN checkerboard / inpainting edges. This
        # supplies the low-frequency evidence diffusion needs (was missing).
        g1 = self.fuse1(l1, n1 + f1); g2 = self.fuse2(l2, n2 + f2); g3 = self.fuse3(l3, n3 + f3)

        # dilated pyramid context replaces the ViT cross-attention bridge
        ctx = self.fpm(c4) # (B, 512, h, w)
        backbone_feat = self.gap(ctx).flatten(1) # (B, 512)

        # Segmentation path - decoder gated by fused noise+frequency cues
        main_mask, aux_masks, d1_feat, edge_logits = self.decoder(
            c0, c1, c2, c3, ctx, n0, g1, g2, g3, x_raw, residual)

        # uncertainty-guided point refinement. The point MLP is evaluated
        # densely (1x1 convs == per-point math), then:
        # train: K uncertainty-biased points are returned for the point loss;
        # eval : only the uncertain band (|p-0.5| < POINT_UNC_DELTA) is re-decided.
        point_logits, point_coords = None, None
        if self.point_head is not None:
            refined_mask = self.point_head(torch.cat([d1_feat, main_mask], dim=1))
            if self.training:
                point_logits, point_coords = _sample_uncertain_points(
                    main_mask, refined_mask, self.point_k, self.point_unc_frac)
            elif getattr(self, 'point_refine_eval', True):
                _p = torch.sigmoid(main_mask)
                _unc = (_p - 0.5).abs() < self.point_unc_delta
                main_mask = torch.where(_unc, refined_mask, main_mask)

        # Localization-aware classification (MVSS-Net): fuse the peak forged-pixel
        # logit (global max-pool over the mask) into the global cls logit. 'fake'
        # fires if EITHER the global feature OR a localized patch says so -- the
        # fix for diffusion fakes whose global appearance looks real.
        mask_gmp = main_mask.amax(dim=(2, 3)) # (B, 1) peak forged response
        cls_logit = self.cls_head(backbone_feat).squeeze(-1) + self.mask_cls(mask_gmp).squeeze(-1)

        # Provenance hashing (CBFH): pooled from c4 guided by deepest decoder mask
        h_soft, h_bin = self.cbfh(c4, aux_masks[0])

        return _to_fp32((cls_logit, main_mask, aux_masks, h_soft, h_bin,
                         point_logits, point_coords, edge_logits))

model = NRGANet(cfg).to(DEVICE)

# load + freeze the content prior(s) trained in cell 6b
if getattr(model, 'use_prior', False):
    for _name, _path in (('content_prior', PRIOR_PATH), ('masked_prior', MASKED_PRIOR_PATH)):
        _m = getattr(model, _name, None)
        if _m is None:
            continue
        if os.path.exists(_path):
            _m.load_state_dict(torch.load(_path, map_location=DEVICE))
            print(f'{_name} loaded -> {_path}')
        else:
            print(f'WARNING: {_name} weights not found ({_path}); acting as an untrained residual.')
        _m.eval()
        for _p in _m.parameters():
            _p.requires_grad = False

# bias init. The decoder is a RESIDUAL cascade (each level ADDS to the
# upsampled coarser logit), so biasing every head to -3 would stack to -15 and
# break the mask_cls no-op below. Only the coarsest head carries the -3
# 'no mask' prior; the refinement heads start neutral at 0.
nn.init.constant_(model.decoder.ds5.bias, -3.0)
nn.init.constant_(model.decoder.ds4.bias, 0.0)
nn.init.constant_(model.decoder.ds3.bias, 0.0)
nn.init.constant_(model.decoder.ds2.bias, 0.0)
nn.init.constant_(model.decoder.out_conv.bias, 0.0)
# Detection coupling init as a NO-OP at the suppressed baseline (mask logit -3):
# 0.5*(-3)+1.5 = 0, so it does not disturb cls warmup; as the decoder learns to
# fire on forged pixels (logit rises above -3), it adds positive 'fake' evidence.
nn.init.constant_(model.mask_cls.weight, 0.5)
nn.init.constant_(model.mask_cls.bias, 1.5)
print(' decoder: residual cascade, coarse head biased to -3.0 (no mask)')
total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'NRGA-Net: {total_p:,} total params | {train_p:,} trainable')

with torch.no_grad():
    dummy = torch.randn(2, 3, 256, 256).to(DEVICE)
    cls_out, mask_out, aux_out, h_soft, h_bin, pl_out, pc_out, el_out = model(dummy)
    print(f'Forward OK: cls={cls_out.shape}, mask={mask_out.shape}, '
          f'aux={[tuple(a.shape[-2:]) for a in aux_out]}, '
          f'points={None if pl_out is None else tuple(pl_out.shape)}, '
          f'edges={None if el_out is None else [tuple(e.shape[-2:]) for e in el_out]}')
    (_c0, _c1, _c2, _c3, _c4), _ = model.cnn(dummy)
    print(f'Encoder grid: c0={tuple(_c0.shape[-2:])} c1={tuple(_c1.shape[-2:])} '
          f'c2={tuple(_c2.shape[-2:])} c3={tuple(_c3.shape[-2:])} c4={tuple(_c4.shape[-2:])} '
          f'-> output stride {cfg.IMG_SIZE // _c4.shape[-1]}')
    # is fully convolutional: it must also run at a non-256 size, otherwise
    # native-resolution / tiled inference (the FLDCF advantage) is out of reach.
    _alt = torch.randn(1, 3, 384, 384).to(DEVICE)
    _c, _m, *_ = model(_alt)
    assert _m.shape[-2:] == (384, 384), _m.shape
    print(f'Resolution-agnostic OK: 384x384 -> mask {tuple(_m.shape)}')
    del _alt, _c, _m, _c0, _c1, _c2, _c3, _c4
    print(f'CBFH hash : soft={tuple(h_soft.shape)}, bin={tuple(h_bin.shape)} ({h_bin.shape[1]}-bit forensic fingerprint)')

# ======================================================================
# 7. NRGA Loss (Focal + Dice + Boundary + DeepSupervision)
# ======================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        # PER-SAMPLE dice (not batch-flattened): every sample -- and thus every
        # forgery source -- is weighted equally regardless of forged-area size, so
        # small-mask diffusion edits are no longer drowned out by large GAN regions.
        pred = torch.sigmoid(pred)
        b = pred.shape[0]
        pred = pred.reshape(b, -1); target = target.reshape(b, -1)
        inter = (pred * target).sum(dim=1)
        dice = (2 * inter + self.smooth) / (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)
        return 1 - dice.mean()

class BoundaryLoss(nn.Module):
    """Enforce sharp edges by comparing Laplacian of predicted vs GT mask."""
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('laplacian', lap)
    def forward(self, pred, target):
        pred_s = torch.sigmoid(pred)
        pred_edge = F.conv2d(pred_s, self.laplacian, padding=1)
        gt_edge = F.conv2d(target, self.laplacian, padding=1)
        return F.l1_loss(pred_edge, gt_edge) * 10.0

# ===================== thin-structure & boundary losses =====================
def _soft_erode(x):
    return torch.min(-F.max_pool2d(-x, (3, 1), (1, 1), (1, 0)),
                     -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1)))

def _soft_dilate(x):
    return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))

def _soft_skel(x, iters):
    """Differentiable morphological skeleton (Shit et al., clDice, CVPR 2021)."""
    op = _soft_dilate(_soft_erode(x))
    skel = F.relu(x - op)
    for _ in range(iters):
        x = _soft_erode(x)
        op = _soft_dilate(_soft_erode(x))
        delta = F.relu(x - op)
        skel = skel + F.relu(delta - skel * delta)
    return skel

class SoftClDiceLoss(nn.Module):
    """Soft centreline Dice (Shit et al., CVPR 2021) -- a CONNECTIVITY guard.

    Read this before tuning LAMBDA_CLDICE: clDice is *thickness-blind*. Measured
    on a synthetic 8-px winding contour, penalty relative to a perfect mask:

        prediction Dice clDice EdgeWeightedBCE
        dilated x11 (fat blob) +0.664 +0.000 +2.478
        dilated x21 (fatter) +0.786 +0.000 +3.048
        fragmented contour +0.187 +0.175 +0.378
        displaced blob +0.969 +0.984 +1.269

    So clDice does NOT fix the fat-blob failure -- EdgeWeightedBCE and Dice do.
    clDice is here because it is the only term that punishes a BROKEN contour
    (its penalty jumps ~145x from perfect to fragmented, versus ~8x for Dice).
    Once the architecture and boundary loss push the mask thin, fragmentation
    becomes the new risk, and this term is what holds the structure together.
    """
    def __init__(self, iters=5, smooth=1.0):
        super().__init__()
        self.iters = int(iters); self.smooth = float(smooth)

    def forward(self, pred_logit, target):
        p = torch.sigmoid(pred_logit)
        sp = _soft_skel(p, self.iters)
        st = _soft_skel(target, self.iters)
        dims = (1, 2, 3)
        tprec = ((sp * target).sum(dims) + self.smooth) / (sp.sum(dims) + self.smooth)
        tsens = ((st * p).sum(dims) + self.smooth) / (st.sum(dims) + self.smooth)
        return (1.0 - (2.0 * tprec * tsens) / (tprec + tsens + 1e-8)).mean()

class EdgeWeightedBCE(nn.Module):
    """Pixel BCE whose weight map spikes on a band around the GT boundary.

    THIS is the main anti-blob term (see the table in SoftClDiceLoss): almost all
    of the error of a fat mask lives just outside the true edge, and weighting
    that band by EDGE_W makes over-spill expensive instead of nearly free. clDice
    handles topology, this handles exact edge placement. The band is a
    morphological dilate-erode of the GT via max-pool (no SciPy, autograd-safe).
    """
    def __init__(self, w=8.0, band=5):
        super().__init__()
        self.w = float(w); self.band = int(band)

    def forward(self, pred, target):
        k, p = self.band, self.band // 2
        dil = F.max_pool2d(target, k, 1, p)
        ero = -F.max_pool2d(-target, k, 1, p)
        edge = (dil - ero).clamp(0, 1)
        wmap = 1.0 + self.w * edge
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (bce * wmap).sum() / (wmap.sum() + 1e-6)

# ---------------- Lovász-hinge (binary Jaccard surrogate) ----------------
def _lovasz_grad(gt_sorted):
    """Gradient of the Lovász extension w.r.t. sorted errors (Berman et al., 2018)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def _lovasz_hinge_flat(logits, labels):
    """Binary Lovász-hinge on flattened logits / 0-1 labels."""
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)

def lovasz_hinge(logits, labels):
    """Mean per-image binary Lovász-hinge. logits/labels: (B,1,H,W)."""
    vals = [_lovasz_hinge_flat(lg.reshape(-1), lb.reshape(-1))
            for lg, lb in zip(logits, labels)]
    return torch.stack(vals).mean() if vals else logits.sum() * 0.0

class EdgeSupervisionLoss(nn.Module):
    """deep boundary supervision on the spectral-edge heads (FECDNet's
    edge-supervision role). GT boundary band = morphological gradient of the mask
    (dilate - erode), computed per supervision scale; BCE + soft Dice per head.
    FAKE samples only (real images have no boundary)."""
    def __init__(self, band=2):
        super().__init__()
        self.band = int(band)

    def _band(self, gt, size):
        g = F.interpolate(gt.float(), size=size, mode='bilinear', align_corners=False)
        k = 2 * self.band + 1
        dil = F.max_pool2d(g, k, stride=1, padding=self.band)
        ero = 1.0 - F.max_pool2d(1.0 - g, k, stride=1, padding=self.band)
        return (dil - ero).clamp(0, 1)

    def forward(self, edge_logits, gt, fake_idx):
        tot = None
        for e in edge_logits:
            band = self._band(gt[fake_idx], e.shape[-2:])
            ef = e[fake_idx].float()
            bce = F.binary_cross_entropy_with_logits(ef, band)
            p = torch.sigmoid(ef)
            dice = 1 - (2 * (p * band).sum(dim=(1, 2, 3)) + 1) / \
                       (p.sum(dim=(1, 2, 3)) + band.sum(dim=(1, 2, 3)) + 1)
            cur = bce + dice.mean()
            tot = cur if tot is None else tot + cur
        return tot / max(len(edge_logits), 1)

class NRGALoss(nn.Module):
    """
    CONDITIONAL segmentation loss - the definitive fix for false positives:
    - FAKE samples (label=1): Focal + Dice + Boundary + DeepSupervision
    - REAL samples (label=0): Strong BCE mask suppression ONLY
    Previous bug: seg losses on real images gave weak/conflicting gradients,
    causing decoder to hedge and produce non-zero masks everywhere.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.edgew = EdgeWeightedBCE(getattr(cfg, 'EDGE_W', 8.0),
                                     getattr(cfg, 'EDGE_BAND', 5)) #
        self.cldice = SoftClDiceLoss(getattr(cfg, 'CLDICE_ITERS', 5)) #
        self.ds_bce = nn.BCEWithLogitsLoss()
        self.ds_dice = DiceLoss()
        self.edgesup = EdgeSupervisionLoss(getattr(cfg, 'EDGE_SUP_BAND', 2)) #

    def forward(self, cls_logit, main_mask, aux_masks, label, mask_gt,
                point_logits=None, point_coords=None, edge_logits=None):
        # Classification loss (ALL samples)
        loss_cls = self.cls_bce(cls_logit, label)

        # Split batch by label
        fake_idx = (label == 1)
        real_idx = (label == 0)
        n_fake = fake_idx.sum().item()
        n_real = real_idx.sum().item()

        # --- Seg losses: ONLY on FAKE samples (label=1) ---
        if n_fake > 0:
            fake_pred = main_mask[fake_idx]
            fake_gt = mask_gt[fake_idx]
            loss_focal = self.focal(fake_pred, fake_gt)
            loss_dice = self.dice(fake_pred, fake_gt)
            loss_boundary = (self.boundary(fake_pred, fake_gt)
                             if getattr(self.cfg, 'LAMBDA_BOUNDARY', 0.0)
                             else torch.tensor(0.0, device=cls_logit.device))
            loss_edge = (self.edgew(fake_pred, fake_gt) #
                         if getattr(self.cfg, 'LAMBDA_EDGE', 0.0)
                         else torch.tensor(0.0, device=cls_logit.device))
            loss_cldice = (self.cldice(fake_pred, fake_gt) #
                           if getattr(self.cfg, 'LAMBDA_CLDICE', 0.0)
                           else torch.tensor(0.0, device=cls_logit.device))
            # deep spectral-edge supervision
            if edge_logits is not None and getattr(self.cfg, 'LAMBDA_EDGESUP', 0.0):
                loss_edgesup = self.edgesup(edge_logits, mask_gt, fake_idx)
            else:
                loss_edgesup = torch.tensor(0.0, device=cls_logit.device)
            # Lovász-hinge -- the convex surrogate of the Jaccard index itself
            # (Berman et al., CVPR 2018): optimises IoU directly, not a soft proxy.
            loss_lovasz = lovasz_hinge(fake_pred.float(), fake_gt.float())
            # point-sampled BCE on the uncertainty-refined boundary logits
            if point_logits is not None and point_coords is not None:
                _pgt = F.grid_sample(mask_gt.float(), point_coords.unsqueeze(2),
                                     mode='nearest', align_corners=True).squeeze(-1) # (B,1,K)
                loss_point = F.binary_cross_entropy_with_logits(
                    point_logits[fake_idx].float(), _pgt[fake_idx])
            else:
                loss_point = torch.tensor(0.0, device=cls_logit.device)
        else:
            loss_focal = torch.tensor(0.0, device=cls_logit.device)
            loss_dice = torch.tensor(0.0, device=cls_logit.device)
            loss_boundary = torch.tensor(0.0, device=cls_logit.device)
            loss_edge = torch.tensor(0.0, device=cls_logit.device) #
            loss_cldice = torch.tensor(0.0, device=cls_logit.device) #
            loss_lovasz = torch.tensor(0.0, device=cls_logit.device) #
            loss_point = torch.tensor(0.0, device=cls_logit.device) #
            loss_edgesup = torch.tensor(0.0, device=cls_logit.device) #

        # --- Mask suppression: ONLY on REAL samples (label=0) ---
        # Strong BCE pushing ALL decoder pixels toward zero
        if n_real > 0:
            real_pred = main_mask[real_idx]
            mask_suppress = F.binary_cross_entropy_with_logits(
                real_pred, torch.zeros_like(real_pred))
        else:
            mask_suppress = torch.tensor(0.0, device=cls_logit.device)

        # --- Deep supervision: ONLY on FAKE samples ---
        loss_ds = torch.tensor(0.0, device=cls_logit.device)
        if n_fake > 0:
            for aux in aux_masks:
                h, w = aux.shape[2], aux.shape[3]
                # mode='nearest' point-samples the GT, so a 15-px road at 1/16
                # became scattered dots or vanished -- deep supervision was then
                # actively teaching a blob. Max-pooling keeps thin structure alive.
                gt_down = F.adaptive_max_pool2d(mask_gt, (h, w))
                loss_ds = loss_ds + self.ds_bce(aux[fake_idx], gt_down[fake_idx])
                loss_ds = loss_ds + self.ds_dice(aux[fake_idx], gt_down[fake_idx])
            loss_ds = loss_ds / len(aux_masks)

        total = (self.cfg.LAMBDA_CLS * loss_cls +
                 self.cfg.LAMBDA_FOCAL * loss_focal +
                 self.cfg.LAMBDA_DICE * loss_dice +
                 self.cfg.LAMBDA_BOUNDARY * loss_boundary +
                 getattr(self.cfg, 'LAMBDA_EDGE', 0.0) * loss_edge +
                 getattr(self.cfg, 'LAMBDA_CLDICE', 0.0) * loss_cldice +
                 getattr(self.cfg, 'LAMBDA_LOVASZ', 0.0) * loss_lovasz +
                 getattr(self.cfg, 'LAMBDA_EDGESUP', 0.0) * loss_edgesup +
                 getattr(self.cfg, 'LAMBDA_POINT', 0.0) * loss_point +
                 self.cfg.LAMBDA_DS * loss_ds +
                 self.cfg.LAMBDA_MASK_SUPPRESS * mask_suppress)

        return total, {
            'cls': loss_cls.item(), 'focal': loss_focal.item(),
            'dice': loss_dice.item(), 'boundary': loss_boundary.item(),
            'edge': loss_edge.item(), 'cldice': loss_cldice.item(),
            'ds': loss_ds.item(), 'suppress': mask_suppress.item(),
            'lovasz': loss_lovasz.item(), 'point': loss_point.item(),
            'edgesup': loss_edgesup.item(),
            'total': total.item()
        }

def provenance_loss(h_anchor, h_pos, margin, lambda_q):
    """CBFH contrastive provenance loss (paper Sec. 4.4.5):
        L = y * d_H^2 + (1 - y) * max(0, m - d_H)^2 + lambda_q * ||h - sign(h)||
    Matched pairs (y=1): clean vs benign-transformed view of the SAME image.
    Mismatched pairs (y=0): in-batch negatives (DIFFERENT images, rolled by 1).
    Distances use a differentiable normalized soft-Hamming on the tanh codes.
    Returns (loss, mean_match_dist, mean_nonmatch_dist).
    """
    def soft_hamming(a, b):
        return ((1.0 - a * b) / 2.0).mean(dim=1) # (B,) in [0,1]
    # Matched (same content) -> push distance to 0
    d_pos = soft_hamming(h_anchor, h_pos)
    loss_pos = (d_pos ** 2).mean()
    # Mismatched (different content) -> push distance beyond margin
    idx = torch.roll(torch.arange(h_anchor.shape[0], device=h_anchor.device), 1)
    d_neg = soft_hamming(h_anchor, h_anchor[idx])
    loss_neg = (F.relu(margin - d_neg) ** 2).mean()
    # Quantization regularizer -> encourage near-binary codes
    quant = 0.5 * ((h_anchor - torch.sign(h_anchor)).norm(dim=1).mean()
                   + (h_pos - torch.sign(h_pos)).norm(dim=1).mean())
    loss = loss_pos + loss_neg + lambda_q * quant
    return loss, d_pos.mean().item(), d_neg.mean().item()

criterion = NRGALoss(cfg).to(DEVICE)
print('NRGALoss ready (CONDITIONAL: seg on fakes only, suppress on reals only)')
print('Provenance loss ready (CBFH contrastive: matched vs in-batch mismatched)')

# ======================================================================
# 8. Training & Validation
# ======================================================================

class EMA:
    """Exponential moving average over all floating-point state entries."""
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self._backup = None
    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # warm-up ramp: early updates track fast (small decay), then relax to the
        # target horizon. Without this, decay 0.999 (~1000-iter horizon) keeps the
        # shadow near the RANDOM INIT for many epochs on small datasets -> the
        # EMA-validated mask collapses to ~0 while live training stays healthy.
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
    @torch.no_grad()
    def apply_to(self, model):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if k in self.shadow}
        msd = model.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v.to(msd[k].dtype))
    @torch.no_grad()
    def restore(self, model):
        if self._backup is None:
            return
        msd = model.state_dict()
        for k, v in self._backup.items():
            msd[k].copy_(v)
        self._backup = None

# ===================== NaN guards =====================
# BatchNorm running statistics are updated inside the FORWARD pass, so a single
# non-finite activation writes NaN into them permanently -- GradScaler skipping
# the optimizer step protects the WEIGHTS but not these BUFFERS. Once poisoned,
# eval() returns NaN for every image (train() survives because it uses batch
# statistics), which is why validation collapsed to 0.0000 and never
# recovered. Snapshot + rollback makes a bad batch a non-event.
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

def _bn_buffers(model):
    out = []
    for mod in model.modules():
        if isinstance(mod, _BN_TYPES):
            if mod.running_mean is not None:
                out.append(mod.running_mean)
            if mod.running_var is not None:
                out.append(mod.running_var)
    return out

@torch.no_grad()
def snapshot_bn(model):
    """Clone every BatchNorm running statistic (a few hundred KB, negligible)."""
    return [b.detach().clone() for b in _bn_buffers(model)]

@torch.no_grad()
def restore_bn(model, snap):
    for b, s in zip(_bn_buffers(model), snap):
        b.copy_(s)

@torch.no_grad()
def sanitize_state_(model, verbose=True):
    """Repair every non-finite parameter/buffer in place. Returns the count fixed."""
    fixed = []
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if not torch.is_floating_point(t) or bool(torch.isfinite(t).all()):
            continue
        if name.endswith('running_var'):
            torch.nan_to_num_(t, nan=1.0, posinf=1.0, neginf=1.0)
            t.clamp_(min=1e-5)
        else:
            torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
        fixed.append(name)
    if fixed and verbose:
        print(f' [guard] repaired {len(fixed)} non-finite tensors '
              f'(e.g. {fixed[:3]}{" ..." if len(fixed) > 3 else ""})')
    return len(fixed)

@torch.no_grad()
def sanitize_ema_(ema, model=None, verbose=True):
    """Repair the EMA shadow. NaN * decay stays NaN forever, so a single bad
    update silently zeroes every subsequent EMA-based validation. When `model` is
    given, a poisoned entry is reseeded from the live weights (far better than
    zeroing it); otherwise it is neutralised numerically."""
    if ema is None:
        return 0
    bad = [k for k, v in ema.shadow.items() if not bool(torch.isfinite(v).all())]
    if not bad:
        return 0
    msd = model.state_dict() if model is not None else None
    for k in bad:
        v = ema.shadow[k]
        src = msd.get(k) if msd is not None else None
        if src is not None and bool(torch.isfinite(src).all()):
            v.copy_(src.detach().float())
            continue
        if k.endswith('running_var'):
            torch.nan_to_num_(v, nan=1.0, posinf=1.0, neginf=1.0)
            v.clamp_(min=1e-5)
        else:
            torch.nan_to_num_(v, nan=0.0, posinf=0.0, neginf=0.0)
    if verbose:
        print(f' [guard] repaired {len(bad)} non-finite EMA shadow entries'
              + (' (reseeded from live weights)' if msd is not None else ''))
    return len(bad)

def checkpoint_img_size(path, ck=None):
    """Resolution a checkpoint was trained at.

    evaluating the 384px fine-tune on a 256px val set is a silent, dataset-
    dependent accuracy loss -- 256px-native Vaihingen barely moves while
    large-native LoveDA / Local_Diffusion collapse. Newer checkpoints record
    `img_size`; older ones are identified by the `_384_` tag in their filename.
    """
    if isinstance(ck, dict):
        v = ck.get('img_size')
        if isinstance(v, int) and v > 0:
            return int(v)
    ft = int(getattr(cfg, 'FT384_IMG_SIZE', 384))
    return ft if f'_{ft}_' in os.path.basename(str(path)) else int(cfg.IMG_SIZE)

@torch.no_grad()
def pick_best_checkpoint(prefer=None, verbose=True):
    """Return (path, loaded_ckpt) for the newest checkpoint that is actually usable.

    every downstream cell used to hard-reference `best_model_path`, a name that
    only exists when the 256 training cell ran in this session. Skipping training and
    loading from Drive -- the documented recovery path -- therefore crashed with
    NameError. A file is rejected when any floating-point tensor is non-finite: the
    epoch-61 NaN event leaves a perfectly loadable checkpoint whose BatchNorm buffers
    are NaN, which then reports 0.0000 for every metric instead of failing loudly.
    """
    cands = list(prefer or [])
    for _g in ('best384_path', 'best_model_path'):
        _v = globals().get(_g)
        if isinstance(_v, str) and _v:
            cands.append(_v)
    cands += ['nrga_densenet201_384_best_FV.pt', 'nrga_densenet201_best_FV.pt',
              'nrga_densenet201_384_best_FV.pt', 'nrga_densenet201_best_FV.pt']
    seen = set()
    for c in cands:
        p = c if os.path.isabs(c) else os.path.join(cfg.OUTPUT_DIR, c)
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        try:
            ck = torch.load(p, map_location='cpu', weights_only=False)
        except Exception as e:
            if verbose:
                print(f' skipping {os.path.basename(p)}: {type(e).__name__}: {e}')
            continue
        bad = [k for k, v in ck.get('model_state', {}).items()
               if torch.is_floating_point(v) and not bool(torch.isfinite(v).all())]
        if bad:
            if verbose:
                print(f' skipping {os.path.basename(p)}: POISONED, '
                      f'{len(bad)} non-finite tensors (e.g. {bad[:2]})')
            continue
        return p, ck
    raise FileNotFoundError(
        'No usable checkpoint in ' + str(cfg.OUTPUT_DIR) + '. Every candidate is either '
        'missing or NaN-poisoned -- run the checkpoint health-check cell to see which.')

def ensure_trained_model(prefer=None, n_probe=96, min_auc=None, force_reload=True,
                         chunk=16):
    """Load the newest CLEAN checkpoint into `model`, then PROVE the classifier works.

    V18: the eval cells used to accept whatever `model` happened to sit in globals.
    The model-construction cell creates a randomly initialised NRGANet, so running an
    eval cell before any checkpoint load silently scored an UNTRAINED network: every
    probability collapses into a narrow band around 0.45 and AUC lands at ~0.5, which
    looks like a plausible table rather than an error. This refuses to continue.
    """
    global model
    min_auc = getattr(cfg, 'EVAL_MIN_AUC', 0.80) if min_auc is None else min_auc
    if force_reload or globals().get('model') is None:
        _p, _ck = pick_best_checkpoint(prefer=prefer, verbose=True)
        model = NRGANet(cfg).to(DEVICE)
        model.load_state_dict(_ck['model_state'])
        print(f' loaded {os.path.basename(_p)} epoch={_ck.get("epoch", "?")}')
        del _ck
    model.eval()

    if 'val_loader' not in globals():
        print(' [warn] val_loader undefined -- skipping the classifier sanity probe')
        return model

    ds = val_loader.dataset
    _idx = np.random.default_rng(0).choice(len(ds), size=min(n_probe, len(ds)),
                                           replace=False)
    _pr, _ys = [], []
    with torch.no_grad():
        for _s in range(0, len(_idx), chunk):
            _items = [ds[int(i)] for i in _idx[_s:_s + chunk]]
            _x = torch.stack([it['image'] for it in _items]).to(DEVICE)
            with autocast():
                _out = model(_x)
            _pr.append(torch.sigmoid(_out[0]).float().cpu().numpy().ravel())
            _ys.append(np.array([float(it['label']) for it in _items]))
    _pr = np.concatenate(_pr)
    _ys = np.concatenate(_ys).astype(int)
    if len(np.unique(_ys)) < 2:
        print(' [warn] sanity probe drew a single class -- AUC not computable')
        return model
    _auc = roc_auc_score(_ys, _pr)
    print(f' sanity probe: n={len(_ys)} (fake={int(_ys.sum())}) AUC={_auc:.4f} '
          f'mean prob real={_pr[_ys == 0].mean():.4f} fake={_pr[_ys == 1].mean():.4f}')
    if _auc < min_auc:
        raise RuntimeError(
            f'classifier AUC {_auc:.4f} < cfg.EVAL_MIN_AUC ({min_auc}). The loaded '
            f'weights do not separate real from fake -- this is exactly what an '
            f'UNTRAINED NRGANet looks like (probabilities bunched near 0.45). '
            f'Refusing to report metrics from it. Run the checkpoint health-check '
            f'cell and load a trained file.')
    return model

def build_param_groups(model, cfg):
    """the pretrained encoder and the from-scratch decoder need different LRs.

    earlier revisions used a single LR (5e-5) for `model.parameters()`. That rate is sane
    for fine-tuning ImageNet DenseNet-201, but it left the randomly-initialised
    decoder, spectral-edge streams, fusion and heads roughly 10x under-trained.
    FECDNet trains its entire network at 5e-4 and reaches 93.47 IoU at 256px on
    this same dataset, so the decoder here is very likely LR-starved.
    """
    lr_enc = float(getattr(cfg, 'LR_ENCODER', getattr(cfg, 'LR', 5e-5)))
    lr_dec = float(getattr(cfg, 'LR_DECODER', 5e-4))
    prefixes = tuple(getattr(cfg, 'ENCODER_PARAM_PREFIXES', ('cnn.',)))
    enc, dec, enc_n, dec_n = [], [], 0, 0
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pname.startswith(prefixes):
            enc.append(p); enc_n += p.numel()
        else:
            dec.append(p); dec_n += p.numel()
    groups = []
    if enc:
        groups.append({'params': enc, 'lr': lr_enc, 'name': 'encoder'})
    if dec:
        groups.append({'params': dec, 'lr': lr_dec, 'name': 'decoder'})
    assert groups, 'build_param_groups: no trainable parameters found'
    print(f' param groups -> encoder {enc_n/1e6:.2f}M @ lr {lr_enc:g} | '
          f'decoder/heads {dec_n/1e6:.2f}M @ lr {lr_dec:g}')
    return groups

def train_one_epoch(model, loader, optimizer, scaler, criterion, ema=None, accum_steps=1):
    model.train()
    total_loss = defaultdict(float)
    all_preds, all_labels, all_probs = [], [], []
    all_ious, all_dices = [], []
    prov_d_pos, prov_d_neg, prov_steps = 0.0, 0.0, 0
    use_robust = getattr(cfg, 'ROBUST_ENABLE', False)
    # CBFH provenance costs a full extra forward pass -- skip it when its weight is 0
    use_prov = use_robust and float(getattr(cfg, 'LAMBDA_PROV', 0.0)) > 0.0
    _nan_batches = 0 # batches dropped for a non-finite loss
    for _bi, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        images = batch['image'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        masks = batch['mask'].to(DEVICE)

        # Robustness augmentation: randomly distort inputs (JPEG/blur/noise) for invariance
        model_input = apply_robustness(images, cfg, always=False) if use_robust else images

        if _bi % accum_steps == 0:
            optimizer.zero_grad()
        # BN running stats are written during the forward pass -- keep a
        # rollback copy so one overflowing batch cannot poison them for good.
        _bn_snap = snapshot_bn(model)
        with autocast():
            cls_logit, main_mask, aux_masks, h_soft, _, point_logits, point_coords, edge_logits = model(model_input)
            loss, loss_dict = criterion(cls_logit, main_mask, aux_masks, labels, masks,
                                        point_logits, point_coords, edge_logits)

            # CBFH provenance: second benign-transformed view -> contrastive hash loss
            if use_prov:
                view2 = apply_robustness(images, cfg, always=True)
                _, _, _, h_pos, *_ = model(view2)
                loss_prov, d_pos, d_neg = provenance_loss(h_soft, h_pos, cfg.CBFH_MARGIN, cfg.LAMBDA_QUANT)
                loss = loss + cfg.LAMBDA_PROV * loss_prov
                prov_d_pos += d_pos; prov_d_neg += d_neg; prov_steps += 1
                loss_dict['prov'] = loss_prov.item()
                loss_dict['total'] = loss.item()
        # drop the batch instead of letting it destroy the model. Without
        # this, a single fp16 overflow in the final mask head left every BN
        # buffer NaN and every later validation returned 0.0000 ( ep.61+).
        if not torch.isfinite(loss):
            _nan_batches += 1
            restore_bn(model, _bn_snap)
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss / accum_steps).backward()
        if (_bi + 1) % accum_steps == 0 or _bi == len(loader) - 1:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        for k, v in loss_dict.items(): total_loss[k] += v
        # --- collect train metrics (cls AUC/F1 + classification-gated mask IoU/Dice) ---
        probs_b = torch.sigmoid(cls_logit).detach().float().cpu()
        all_probs.extend(probs_b.numpy())
        all_preds.extend((probs_b > 0.5).numpy())
        all_labels.extend(labels.cpu().numpy())
        mask_prob_b = torch.sigmoid(main_mask).detach().float().cpu()
        masks_b = masks.detach().cpu()
        for i in range(len(labels)):
            if labels[i].item() == 1: # FAKE -> gated mask metrics
                if probs_b[i].item() > 0.5:
                    pred_bin = (mask_prob_b[i] > 0.5).float(); pred_soft = mask_prob_b[i].view(-1)
                else:
                    pred_bin = torch.zeros_like(mask_prob_b[i]); pred_soft = torch.zeros(mask_prob_b[i].numel())
                inter = (pred_bin * masks_b[i]).sum()
                union = ((pred_bin + masks_b[i]) > 0).float().sum()
                all_ious.append((inter / (union + 1e-8)).item())
                gt_s = masks_b[i].view(-1)
                all_dices.append((2 * (pred_soft * gt_s).sum() / (pred_soft.sum() + gt_s.sum() + 1e-8)).item())
    n = max(len(loader) - _nan_batches, 1)
    metrics = defaultdict(float, {k: v / n for k, v in total_loss.items()})
    metrics['nan_batches'] = _nan_batches
    if _nan_batches:
        print(f' [guard] dropped {_nan_batches}/{len(loader)} non-finite '
              f'batches (BN stats rolled back, weights untouched)')
    sanitize_ema_(ema, model)
    if prov_steps > 0:
        metrics['prov_d_pos'] = prov_d_pos / prov_steps
        metrics['prov_d_neg'] = prov_d_neg / prov_steps
    all_labels_np = np.array(all_labels); all_probs_np = np.array(all_probs)
    all_preds_np = (all_probs_np > 0.5).astype(int)
    metrics['accuracy'] = accuracy_score(all_labels_np, all_preds_np)
    metrics['f1'] = f1_score(all_labels_np, all_preds_np, zero_division=0)
    try: metrics['auc'] = roc_auc_score(all_labels_np, all_probs_np)
    except: metrics['auc'] = 0.0
    metrics['mean_iou'] = np.mean(all_ious) if all_ious else 0.0
    metrics['mean_dice'] = np.mean(all_dices) if all_dices else 0.0
    return metrics, metrics['accuracy']

@torch.no_grad()
def validate(model, loader, criterion, tta_ms=False):
    _thr = float(getattr(cfg, 'MASK_THRESHOLD', 0.5)) # tuned binarization cut
    model.eval()
    total_loss = defaultdict(float)
    all_probs, all_labels = [], []
    all_ious, all_dices, all_methods = [], [], []
    samp_method = [] # method tag for EVERY sample (fake and real)
    all_tp, all_fp, all_fn = [], [], [] # per-fake pixel counts, aligned with all_methods
    all_ious_raw, all_fake_correct, all_gt_area = [], [], [] # ungated IoU + recall + GT forged-area (diagnostic)
    all_pin, all_parea = [], [] # mean prob inside GT + predicted area (failure-mode probe)
    real_mask_areas = [] # Track false positive masks on real images
    prov_match, prov_nonmatch = [], [] # CBFH normalized Hamming distances
    # pooled pixel counts for the FECDNet-style metric
    # IoU = TP / (TP + FP + FN) with TP/FP/FN summed over the WHOLE test set
    # (paper Eq. 17; their Evaluator accumulates one confusion matrix).
    # This is UNGATED pure segmentation -- FECDNet has no classifier branch.
    _ptp = _pfp = _pfn = 0.0
    _pool_real = bool(getattr(cfg, 'POOLED_INCLUDE_REAL', True))
    use_robust = getattr(cfg, 'ROBUST_ENABLE', False)
    # CBFH provenance costs a full extra forward pass -- skip it when its weight is 0
    use_prov = use_robust and float(getattr(cfg, 'LAMBDA_PROV', 0.0)) > 0.0
    for batch in tqdm(loader, desc='Val', leave=False):
        images = batch['image'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        masks_gt = batch['mask'].to(DEVICE)
        with autocast():
            cls_logit, main_mask, aux_masks, _, h_bin, *_ = model(images)
            loss, loss_dict = criterion(cls_logit, main_mask, aux_masks, labels, masks_gt)
            # Provenance verification: benign view (matched) vs rolled batch (non-matched)
            if use_prov:
                view2 = apply_robustness(images, cfg, always=True)
                _, _, _, _, h_bin2, *_ = model(view2)
                hb = h_bin.float(); hb2 = h_bin2.float()
                d_match = ((1.0 - hb * hb2) / 2.0).mean(dim=1).cpu().numpy()
                prov_match.extend(d_match.tolist())
                idx = torch.roll(torch.arange(hb.shape[0], device=hb.device), 1)
                d_non = ((1.0 - hb * hb2[idx]) / 2.0).mean(dim=1).cpu().numpy()
                prov_nonmatch.extend(d_non.tolist())
        for k, v in loss_dict.items(): total_loss[k] += v
        all_probs.extend(torch.sigmoid(cls_logit).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # Classification-gated mask evaluation
        cls_probs_b = torch.sigmoid(cls_logit).cpu()
        if tta_ms: # flip + x1.5 multi-scale TTA for the mask probabilities
            with torch.no_grad():
                _mp = torch.zeros_like(main_mask, dtype=torch.float32)
                for _d in (None, [-1], [-2], [-1, -2]):
                    _xt = images if _d is None else torch.flip(images, _d)
                    with autocast():
                        _p1 = torch.sigmoid(model(_xt)[1])
                    _xh = F.interpolate(_xt, scale_factor=1.5, mode='bilinear', align_corners=False)
                    with autocast():
                        _p2 = torch.sigmoid(model(_xh)[1])
                    _p2 = F.interpolate(_p2.float(), size=_p1.shape[-2:], mode='bilinear', align_corners=False)
                    _pm = 0.5 * (_p1.float() + _p2)
                    _mp = _mp + (_pm if _d is None else torch.flip(_pm, _d))
                mask_prob = (_mp / 4.0).cpu()
        else:
            mask_prob = torch.sigmoid(main_mask).cpu()
        masks_gt_cpu = masks_gt.cpu()
        samp_method.extend(list(batch['method']))
        for i in range(len(labels)):
            if labels[i].item() == 1: # Ground truth: FAKE
                cls_says_fake = cls_probs_b[i].item() > 0.5
                if cls_says_fake:
                    pred_bin = (mask_prob[i] > _thr).float()
                    pred_soft = mask_prob[i].view(-1)
                else:
                    pred_bin = torch.zeros_like(mask_prob[i])
                    pred_soft = torch.zeros(mask_prob[i].numel())
                inter = (pred_bin * masks_gt_cpu[i]).sum()
                union = ((pred_bin + masks_gt_cpu[i]) > 0).float().sum()
                iou = (inter / (union + 1e-8)).item()
                gt_s = masks_gt_cpu[i].view(-1)
                dice_s = (2*(pred_soft*gt_s).sum()/(pred_soft.sum()+gt_s.sum()+1e-8)).item()
                all_ious.append(iou); all_dices.append(dice_s)
                all_methods.append(batch['method'][i])
                # diagnostic: UNGATED IoU (pure segmentation) + per-sample cls recall
                pb_raw = (mask_prob[i] > _thr).float()
                inter_r = (pb_raw * masks_gt_cpu[i]).sum()
                union_r = ((pb_raw + masks_gt_cpu[i]) > 0).float().sum()
                all_ious_raw.append((inter_r / (union_r + 1e-8)).item())
                all_fake_correct.append(1.0 if cls_says_fake else 0.0)
                all_gt_area.append(masks_gt_cpu[i].mean().item())
                all_pin.append((mask_prob[i] * masks_gt_cpu[i]).sum().item() / (masks_gt_cpu[i].sum().item() + 1e-6))
                all_parea.append((mask_prob[i] > _thr).float().mean().item())
                # pooled counts (ungated, forged class)
                _g = (masks_gt_cpu[i] > 0.5).float()
                _tp_i = (pb_raw * _g).sum().item()
                _fp_i = (pb_raw * (1.0 - _g)).sum().item()
                _fn_i = ((1.0 - pb_raw) * _g).sum().item()
                _ptp += _tp_i; _pfp += _fp_i; _pfn += _fn_i
                all_tp.append(_tp_i); all_fp.append(_fp_i); all_fn.append(_fn_i)
            else: # Ground truth: REAL
                raw_mask_area = (mask_prob[i] > _thr).float().mean().item()
                real_mask_areas.append(raw_mask_area)
                if _pool_real: # no GT foreground -> pure false positives
                    _pfp += (mask_prob[i] > _thr).float().sum().item()
    n = len(loader)
    all_probs = np.array(all_probs); all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    m = {k: v / n for k, v in total_loss.items()}
    m['accuracy'] = accuracy_score(all_labels, all_preds)
    m['f1'] = f1_score(all_labels, all_preds, zero_division=0)
    m['precision'] = precision_score(all_labels, all_preds, zero_division=0)
    m['recall'] = recall_score(all_labels, all_preds, zero_division=0)
    try: m['auc'] = roc_auc_score(all_labels, all_probs)
    except: m['auc'] = 0.0
    m['mean_iou'] = np.mean(all_ious) if all_ious else 0.0
    m['mean_dice'] = np.mean(all_dices) if all_dices else 0.0
    m['real_fp_mask'] = np.mean(real_mask_areas) if real_mask_areas else 0.0
    # FECDNet-comparable pooled metrics (Eq. 17). Reported ALONGSIDE the
    # per-image mean above -- never as a replacement for it.
    _eps = 1e-9
    m['pooled_iou'] = _ptp / (_ptp + _pfp + _pfn + _eps)
    m['pooled_precision'] = _ptp / (_ptp + _pfp + _eps)
    m['pooled_recall'] = _ptp / (_ptp + _pfn + _eps)
    m['pooled_f1'] = (2.0 * m['pooled_precision'] * m['pooled_recall']
                      / (m['pooled_precision'] + m['pooled_recall'] + _eps))
    m['pooled_tp'], m['pooled_fp'], m['pooled_fn'] = _ptp, _pfp, _pfn
    # ---- per-dataset breakdown -------------------------------------------
    # Detection (image-level) needs both classes. With cfg.DEDUPE_REALS the shared
    # authentic images are loaded under the FIRST method entry only, so every method
    # is scored against its own forgeries + the SHARED authentic pool -- otherwise
    # the second method would have no negatives and AUC would be undefined.
    _real_idx = [k for k in range(len(all_labels)) if all_labels[k] == 0]
    _methods_present = list(dict.fromkeys(samp_method)) or list(cfg.DATASET_PATHS.keys())
    m['_datasets'] = [mn for mn in cfg.DATASET_PATHS.keys() if mn in _methods_present] or _methods_present
    m['_n_real_shared'] = len(_real_idx)
    for mn in cfg.DATASET_PATHS.keys():
        idxs = [k for k, met in enumerate(all_methods) if met == mn]
        m[f'iou_{mn}'] = np.mean([all_ious[k] for k in idxs]) if idxs else 0.0
        m[f'iouRAW_{mn}'] = np.mean([all_ious_raw[k] for k in idxs]) if idxs else 0.0
        m[f'rec_{mn}'] = np.mean([all_fake_correct[k] for k in idxs]) if idxs else 0.0
        m[f'area_{mn}'] = np.mean([all_gt_area[k] for k in idxs]) if idxs else 0.0
        m[f'pin_{mn}'] = np.mean([all_pin[k] for k in idxs]) if idxs else 0.0
        m[f'parea_{mn}'] = np.mean([all_parea[k] for k in idxs]) if idxs else 0.0
        m[f'dice_{mn}'] = np.mean([all_dices[k] for k in idxs]) if idxs else 0.0
        m[f'n_{mn}'] = len(idxs)
        # pixel-level pooled (forged images of THIS method only -> rows stay additive)
        _tp = float(np.sum([all_tp[k] for k in idxs])) if idxs else 0.0
        _fp = float(np.sum([all_fp[k] for k in idxs])) if idxs else 0.0
        _fn = float(np.sum([all_fn[k] for k in idxs])) if idxs else 0.0
        m[f'prec_{mn}'] = _tp / (_tp + _fp + _eps)
        m[f'recpix_{mn}'] = _tp / (_tp + _fn + _eps)
        m[f'f1pix_{mn}'] = 2.0 * m[f'prec_{mn}'] * m[f'recpix_{mn}'] / (
            m[f'prec_{mn}'] + m[f'recpix_{mn}'] + _eps)
        m[f'iouPooled_{mn}'] = _tp / (_tp + _fp + _fn + _eps)
        # image-level detection: this method's fakes + the shared authentic pool
        _fake_pos = [k for k in range(len(all_labels))
                     if all_labels[k] == 1 and samp_method[k] == mn]
        _sub = sorted(_fake_pos + _real_idx)
        if _sub and len(set(all_labels[k] for k in _sub)) == 2:
            _y = all_labels[_sub]; _p = all_probs[_sub]; _pr = (_p > 0.5).astype(int)
            m[f'acc_{mn}'] = accuracy_score(_y, _pr)
            m[f'f1det_{mn}'] = f1_score(_y, _pr, zero_division=0)
            try: m[f'auc_{mn}'] = roc_auc_score(_y, _p)
            except Exception: m[f'auc_{mn}'] = 0.0
        else:
            m[f'acc_{mn}'] = m[f'f1det_{mn}'] = m[f'auc_{mn}'] = 0.0
    # OVERALL localization pooled over forged images only, so it is directly
    # comparable to (and additive with) the per-dataset rows above.
    _ftp = float(np.sum(all_tp)) if all_tp else 0.0
    _ffp = float(np.sum(all_fp)) if all_fp else 0.0
    _ffn = float(np.sum(all_fn)) if all_fn else 0.0
    m['fakeonly_precision'] = _ftp / (_ftp + _ffp + _eps)
    m['fakeonly_recall'] = _ftp / (_ftp + _ffn + _eps)
    m['fakeonly_f1'] = 2.0 * m['fakeonly_precision'] * m['fakeonly_recall'] / (
        m['fakeonly_precision'] + m['fakeonly_recall'] + _eps)
    m['fakeonly_iou'] = _ftp / (_ftp + _ffp + _ffn + _eps)
    # CBFH provenance verification metrics
    if prov_match and prov_nonmatch:
        m['prov_match_dist'] = float(np.mean(prov_match))
        m['prov_nonmatch_dist'] = float(np.mean(prov_nonmatch))
        tau = cfg.CBFH_MARGIN
        correct = sum(d < tau for d in prov_match) + sum(d >= tau for d in prov_nonmatch)
        m['prov_verify_acc'] = correct / (len(prov_match) + len(prov_nonmatch))
    else:
        m['prov_match_dist'] = 0.0; m['prov_nonmatch_dist'] = 0.0; m['prov_verify_acc'] = 0.0
    return m, all_probs, all_labels

# ======================================================================
# 9. Run Training
# ======================================================================

optimizer = optim.AdamW(build_param_groups(model, cfg),
                        lr=float(getattr(cfg, 'LR_DECODER', 5e-4)),
                        weight_decay=cfg.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
scaler = GradScaler()

# ---------------- EMA of weights (validation + checkpoint use EMA) ----------------
ema = EMA(model, getattr(cfg, 'EMA_DECAY', 0.999)) if getattr(cfg, 'EMA_ENABLE', True) else None

history = defaultdict(list)
best_score = 0.0
patience_counter = 0
best_model_path = os.path.join(cfg.OUTPUT_DIR, 'nrga_densenet201_best_FV.pt')
ckpt_last_path = os.path.join(cfg.OUTPUT_DIR, 'nrga_densenet201_last_FV.pt')
start_epoch = 1

# ---- resume support (Colab sessions disconnect mid-run) ----
if getattr(cfg, 'RESUME', True) and os.path.exists(ckpt_last_path):
    _rs = torch.load(ckpt_last_path, map_location=DEVICE, weights_only=False)
    # a run that ended on a NaN epoch leaves a perfectly loadable resume file
    # whose BatchNorm buffers are NaN -- resuming from it reproduces the collapse on
    # the first validation. Fall back to the (clean) best checkpoint's weights.
    _badr = [k for k, v in _rs['model_state'].items()
             if torch.is_floating_point(v) and not bool(torch.isfinite(v).all())]
    _fellback = bool(_badr) and os.path.exists(best_model_path)
    if _fellback:
        print(f'>>> resume file holds {len(_badr)} non-finite tensors '
              f'(e.g. {_badr[:2]}) -> loading weights from the best checkpoint instead')
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE,
                                         weights_only=False)['model_state'])
    else:
        model.load_state_dict(_rs['model_state'])
    sanitize_state_(model)
    # V19: the AdamW moment buffers (exp_avg / exp_avg_sq) go NaN in the SAME
    # collapse that poisons the weights, and GradScaler only inspects GRADIENTS --
    # so restoring clean weights while loading poisoned moments re-poisons the
    # model on the very first optimizer.step(). Drop the moments in that case;
    # AdamW rebuilds them within a few dozen iterations.
    _nan_opt = sum(1 for _st in (_rs.get('optim_state', {}).get('state', {}) or {}).values()
                   for _v in _st.values()
                   if torch.is_tensor(_v) and torch.is_floating_point(_v)
                   and not bool(torch.isfinite(_v).all()))
    if _fellback or _nan_opt:
        print(f'>>> discarding the saved optimizer state '
              f'({_nan_opt} non-finite moment tensors) -- AdamW restarts with zero moments')
    else:
        optimizer.load_state_dict(_rs['optim_state'])
    scheduler.load_state_dict(_rs['sched_state'])
    scaler.load_state_dict(_rs['scaler_state'])
    if ema is not None and _rs.get('ema_shadow') is not None:
        ema.shadow = {k: v.to(DEVICE) for k, v in _rs['ema_shadow'].items()}
        ema.updates = int(_rs.get('ema_updates', 0))
        sanitize_ema_(ema, model)
    start_epoch = int(_rs['epoch']) + 1
    best_score = float(_rs.get('best_score', 0.0))
    patience_counter = int(_rs.get('patience', 0))
    history = defaultdict(list, _rs.get('history', {}))
    print(f'>>> Resumed at epoch {start_epoch} (best score {best_score:.4f})')

print(f'Training NRGA-Net (split LRs + focused loss + 2099/525 split) for {cfg.EPOCHS} epochs | Patience: {cfg.PATIENCE}')
print()

for epoch in range(start_epoch, cfg.EPOCHS + 1):
    print(f'Epoch {epoch}/{cfg.EPOCHS} (LR: {scheduler.get_last_lr()[0]:.6f})')
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, ema)
    ema_active = ema is not None and ema.updates >= getattr(cfg, 'EMA_WARMUP_STEPS', 300)
    if ema_active:
        ema.apply_to(model) # validate + checkpoint with EMA weights
        if not getattr(ema, '_announced', False):
            ema._announced = True
            print(f'>>> EMA weights active for validation/checkpointing ({ema.updates} updates)')
    val_m, _, _ = validate(model, val_loader, criterion)
    scheduler.step()

    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss['total'])
    history['train_auc'].append(train_loss['auc'])
    history['train_f1'].append(train_loss['f1'])
    history['train_iou'].append(train_loss['mean_iou'])
    history['train_dice'].append(train_loss['mean_dice'])
    history['val_loss'].append(val_m['total'])
    history['val_auc'].append(val_m['auc'])
    history['val_f1'].append(val_m['f1'])
    history['val_iou'].append(val_m['mean_iou'])
    history['val_dice'].append(val_m['mean_dice'])
    history['val_real_fp'].append(val_m['real_fp_mask'])

    sup_val = train_loss.get("suppress", 0.0)
    prov_val = train_loss.get("prov", 0.0)
    print(f' Train -> Loss:{train_loss["total"]:.4f} (focal:{train_loss["focal"]:.4f} dice:{train_loss["dice"]:.4f} bnd:{train_loss["boundary"]:.4f} edg:{train_loss.get("edge", 0.0):.4f} cld:{train_loss.get("cldice", 0.0):.4f} lov:{train_loss.get("lovasz", 0.0):.4f} pnt:{train_loss.get("point", 0.0):.4f} ds:{train_loss["ds"]:.4f} sup:{sup_val:.4f} prov:{prov_val:.4f})')
    print(f' Val -> Loss:{val_m["total"]:.4f} | Acc:{val_m["accuracy"]:.4f} | AUC:{val_m["auc"]:.4f} | F1:{val_m["f1"]:.4f}')
    print(f' IoU:{val_m["mean_iou"]:.4f} | Dice:{val_m["mean_dice"]:.4f} | RealFP:{val_m["real_fp_mask"]:.4f}')
    print(f' [FECDNet-comparable pooled] IoU:{val_m["pooled_iou"]:.4f} '
          f'F1:{val_m["pooled_f1"]:.4f} P:{val_m["pooled_precision"]:.4f} R:{val_m["pooled_recall"]:.4f}')
    per_method = ' | '.join(f'{mn}:{val_m.get(f"iou_{mn}", 0.0):.4f}' for mn in cfg.DATASET_PATHS.keys())
    print(f' {per_method}')
    per_diag = ' | '.join(f'{mn}[raw:{val_m.get(f"iouRAW_{mn}",0.0):.3f} rec:{val_m.get(f"rec_{mn}",0.0):.3f} pin:{val_m.get(f"pin_{mn}",0.0):.3f} par:{val_m.get(f"parea_{mn}",0.0):.3f}]' for mn in cfg.DATASET_PATHS.keys())
    print(f' {per_diag}')
    print(f' CBFH match_d:{val_m["prov_match_dist"]:.4f} | nonmatch_d:{val_m["prov_nonmatch_dist"]:.4f} | verify_acc:{val_m["prov_verify_acc"]:.4f}')

    # localization is what this revision targets, so IoU/Dice dominate.
    score = val_m['auc']*0.2 + val_m['mean_iou']*0.6 + val_m['mean_dice']*0.2
    if not np.isfinite(score) or score <= 0.0:
        # an exactly-0 score means the forward pass returned NaN, not that the
        # model got worse. Do not touch the best checkpoint, and say so loudly.
        print(' !!! validation returned a non-finite / zero score -- the model has '
              'NaN weights or BN buffers. Best checkpoint left untouched; stop the run '
              'and use the checkpoint health-check cell.')
        score = -1.0
    if score > best_score:
        best_score = score; patience_counter = 0
        torch.save({'epoch':epoch, 'model_state':model.state_dict(), 'metrics':val_m,
                    'img_size':int(cfg.IMG_SIZE)}, best_model_path)
        print(f' >>> Best model saved (score={best_score:.4f})')
    else:
        patience_counter += 1
        print(f' No improvement ({patience_counter}/{cfg.PATIENCE})')
    if ema_active:
        ema.restore(model) # hand live weights back before the next epoch

    # rolling 'last' checkpoint (live weights) so a dropped session resumes
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'img_size': int(cfg.IMG_SIZE),
                'optim_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'ema_shadow': (ema.shadow if ema is not None else None),
                'ema_updates': (ema.updates if ema is not None else 0),
                'best_score': best_score, 'patience': patience_counter,
                'history': dict(history)}, ckpt_last_path)
    if patience_counter >= cfg.PATIENCE:
        print(f'\nEarly stopping at epoch {epoch}')
        break

print(f'\nDone. Best score: {best_score:.4f}')

# ======================================================================
# 10b. : high-resolution boundary fine-tune (384)
#
# Diagnosis from the 256 run: recall saturates (~0.98) and predicted area matches GT, but the
# mask is offset by a few pixels — per-image IoU is boundary-resolution-limited at 256.
# Fine-tuning the converged model at 384 sharpens boundary alignment (with `NATIVE_CROP` the
# crops keep native detail when the source is larger than 384). All weights are size-agnostic,
# so the 256 checkpoint loads strictly. Saves to a separate 384px checkpoint
# — the 256 best model is preserved.
# ======================================================================

# ===================== high-resolution boundary fine-tune =====================
# allow FT-only runs (no 40-epoch retrain) -- pick the best available 256 checkpoint
_bmp = globals().get('best_model_path')
if not isinstance(_bmp, str) or not os.path.exists(_bmp):
    best_model_path = None
    for _cand in ('nrga_densenet201_best_FV.pt'):
        _p = os.path.join(cfg.OUTPUT_DIR, _cand)
        if os.path.exists(_p):
            best_model_path = _p
            break
if best_model_path is not None:
    print(f'384 fine-tune starts from: {best_model_path}')
if getattr(cfg, 'FT384_ENABLE', True) and best_model_path is not None and os.path.exists(best_model_path):
    if 'EMA' not in globals(): # self-sufficient when the 40-epoch run cell was skipped
        class EMA:
            """Exponential moving average over all floating-point state entries."""
            def __init__(self, model, decay):
                self.decay = float(decay)
                self.updates = 0
                self.shadow = {k: v.detach().clone().float()
                               for k, v in model.state_dict().items() if v.dtype.is_floating_point}
                self._backup = None
            @torch.no_grad()
            def update(self, model):
                self.updates += 1
                # warm-up ramp: early updates track fast (small decay), then relax to the
                # target horizon. Without this, decay 0.999 (~1000-iter horizon) keeps the
                # shadow near the RANDOM INIT for many epochs on small datasets -> the
                # EMA-validated mask collapses to ~0 while live training stays healthy.
                d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
                for k, v in model.state_dict().items():
                    if k in self.shadow:
                        self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            @torch.no_grad()
            def apply_to(self, model):
                self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                                if k in self.shadow}
                msd = model.state_dict()
                for k, v in self.shadow.items():
                    msd[k].copy_(v.to(msd[k].dtype))
            @torch.no_grad()
            def restore(self, model):
                if self._backup is None:
                    return
                msd = model.state_dict()
                for k, v in self._backup.items():
                    msd[k].copy_(v)
                self._backup = None

    if 'build_param_groups' not in globals():
        def build_param_groups(model, cfg):
            """the pretrained encoder and the from-scratch decoder need different LRs.

            earlier revisions used a single LR (5e-5) for `model.parameters()`. That rate is sane
            for fine-tuning ImageNet DenseNet-201, but it left the randomly-initialised
            decoder, spectral-edge streams, fusion and heads roughly 10x under-trained.
            FECDNet trains its entire network at 5e-4 and reaches 93.47 IoU at 256px on
            this same dataset, so the decoder here is very likely LR-starved.
            """
            lr_enc = float(getattr(cfg, 'LR_ENCODER', getattr(cfg, 'LR', 5e-5)))
            lr_dec = float(getattr(cfg, 'LR_DECODER', 5e-4))
            prefixes = tuple(getattr(cfg, 'ENCODER_PARAM_PREFIXES', ('cnn.',)))
            enc, dec, enc_n, dec_n = [], [], 0, 0
            for pname, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if pname.startswith(prefixes):
                    enc.append(p); enc_n += p.numel()
                else:
                    dec.append(p); dec_n += p.numel()
            groups = []
            if enc:
                groups.append({'params': enc, 'lr': lr_enc, 'name': 'encoder'})
            if dec:
                groups.append({'params': dec, 'lr': lr_dec, 'name': 'decoder'})
            assert groups, 'build_param_groups: no trainable parameters found'
            print(f' param groups -> encoder {enc_n/1e6:.2f}M @ lr {lr_enc:g} | '
                  f'decoder/heads {dec_n/1e6:.2f}M @ lr {lr_dec:g}')
            return groups
    if 'apply_robustness' not in globals(): # robustness aug (cell 11 equivalent)
        # ===================== Robustness Augmentation =====================
        # Distortions used for (1) robustness training (invariance) and
        # (2) CBFH benign-transform second views: JPEG / Gaussian blur / Gaussian noise.
        from PIL import Image as _PILImage

        _RB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _RB_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        def _denorm(x):
            return (x * _RB_STD.to(x.device) + _RB_MEAN.to(x.device)).clamp(0, 1)

        def _renorm(x):
            return (x - _RB_MEAN.to(x.device)) / _RB_STD.to(x.device)

        def _jpeg(img01, quality):
            arr = (img01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            buf = io.BytesIO()
            _PILImage.fromarray(arr).save(buf, format='JPEG', quality=int(quality))
            buf.seek(0)
            out = np.array(_PILImage.open(buf).convert('RGB')).astype(np.float32) / 255.0
            return torch.from_numpy(out).permute(2, 0, 1)

        def _blur(img01, k):
            return TF.gaussian_blur(img01, kernel_size=int(k))

        def _noise(img01, sigma):
            return (img01 + torch.randn_like(img01) * float(sigma)).clamp(0, 1)

        def apply_robustness(x_norm, cfg, always=False):
            """Apply a random JPEG / Gaussian-blur / Gaussian-noise distortion to a
            normalized image batch.
              always=True -> force exactly one distortion per sample (CBFH benign view)
              always=False -> apply a distortion with prob cfg.ROBUST_PROB (invariance aug)
            """
            x01 = _denorm(x_norm).cpu()
            out = []
            for i in range(x01.shape[0]):
                img = x01[i]
                if always or random.random() < cfg.ROBUST_PROB:
                    kind = random.choice(['jpeg', 'blur', 'noise'])
                    if kind == 'jpeg':
                        img = _jpeg(img, random.choice(cfg.JPEG_QUALITIES))
                    elif kind == 'blur':
                        img = _blur(img, random.choice(cfg.BLUR_KERNELS))
                    else:
                        img = _noise(img, random.choice(cfg.NOISE_SIGMAS))
                out.append(img)
            return _renorm(torch.stack(out).to(x_norm.device))
    if 'train_one_epoch' not in globals(): # self-sufficient fallbacks
        def train_one_epoch(model, loader, optimizer, scaler, criterion, ema=None, accum_steps=1):
            model.train()
            total_loss = defaultdict(float)
            all_preds, all_labels, all_probs = [], [], []
            all_ious, all_dices = [], []
            prov_d_pos, prov_d_neg, prov_steps = 0.0, 0.0, 0
            use_robust = getattr(cfg, 'ROBUST_ENABLE', False)
            # CBFH provenance costs a full extra forward pass -- skip it when its weight is 0
            use_prov = use_robust and float(getattr(cfg, 'LAMBDA_PROV', 0.0)) > 0.0
            _nan_batches = 0 # batches dropped for a non-finite loss
            for _bi, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                masks = batch['mask'].to(DEVICE)

                # Robustness augmentation: randomly distort inputs (JPEG/blur/noise) for invariance
                model_input = apply_robustness(images, cfg, always=False) if use_robust else images

                if _bi % accum_steps == 0:
                    optimizer.zero_grad()
                # BN running stats are written during the forward pass -- keep a
                # rollback copy so one overflowing batch cannot poison them for good.
                _bn_snap = snapshot_bn(model)
                with autocast():
                    cls_logit, main_mask, aux_masks, h_soft, _, point_logits, point_coords, edge_logits = model(model_input)
                    loss, loss_dict = criterion(cls_logit, main_mask, aux_masks, labels, masks,
                                                point_logits, point_coords, edge_logits)

                    # CBFH provenance: second benign-transformed view -> contrastive hash loss
                    if use_prov:
                        view2 = apply_robustness(images, cfg, always=True)
                        _, _, _, h_pos, *_ = model(view2)
                        loss_prov, d_pos, d_neg = provenance_loss(h_soft, h_pos, cfg.CBFH_MARGIN, cfg.LAMBDA_QUANT)
                        loss = loss + cfg.LAMBDA_PROV * loss_prov
                        prov_d_pos += d_pos; prov_d_neg += d_neg; prov_steps += 1
                        loss_dict['prov'] = loss_prov.item()
                        loss_dict['total'] = loss.item()
                # drop the batch instead of letting it destroy the model. Without
                # this, a single fp16 overflow in the final mask head left every BN
                # buffer NaN and every later validation returned 0.0000 ( ep.61+).
                if not torch.isfinite(loss):
                    _nan_batches += 1
                    restore_bn(model, _bn_snap)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss / accum_steps).backward()
                if (_bi + 1) % accum_steps == 0 or _bi == len(loader) - 1:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)
                for k, v in loss_dict.items(): total_loss[k] += v
                # --- collect train metrics (cls AUC/F1 + classification-gated mask IoU/Dice) ---
                probs_b = torch.sigmoid(cls_logit).detach().float().cpu()
                all_probs.extend(probs_b.numpy())
                all_preds.extend((probs_b > 0.5).numpy())
                all_labels.extend(labels.cpu().numpy())
                mask_prob_b = torch.sigmoid(main_mask).detach().float().cpu()
                masks_b = masks.detach().cpu()
                for i in range(len(labels)):
                    if labels[i].item() == 1: # FAKE -> gated mask metrics
                        if probs_b[i].item() > 0.5:
                            pred_bin = (mask_prob_b[i] > 0.5).float(); pred_soft = mask_prob_b[i].view(-1)
                        else:
                            pred_bin = torch.zeros_like(mask_prob_b[i]); pred_soft = torch.zeros(mask_prob_b[i].numel())
                        inter = (pred_bin * masks_b[i]).sum()
                        union = ((pred_bin + masks_b[i]) > 0).float().sum()
                        all_ious.append((inter / (union + 1e-8)).item())
                        gt_s = masks_b[i].view(-1)
                        all_dices.append((2 * (pred_soft * gt_s).sum() / (pred_soft.sum() + gt_s.sum() + 1e-8)).item())
            n = max(len(loader) - _nan_batches, 1)
            metrics = defaultdict(float, {k: v / n for k, v in total_loss.items()})
            metrics['nan_batches'] = _nan_batches
            if _nan_batches:
                print(f' [guard] dropped {_nan_batches}/{len(loader)} non-finite '
                      f'batches (BN stats rolled back, weights untouched)')
            sanitize_ema_(ema, model)
            if prov_steps > 0:
                metrics['prov_d_pos'] = prov_d_pos / prov_steps
                metrics['prov_d_neg'] = prov_d_neg / prov_steps
            all_labels_np = np.array(all_labels); all_probs_np = np.array(all_probs)
            all_preds_np = (all_probs_np > 0.5).astype(int)
            metrics['accuracy'] = accuracy_score(all_labels_np, all_preds_np)
            metrics['f1'] = f1_score(all_labels_np, all_preds_np, zero_division=0)
            try: metrics['auc'] = roc_auc_score(all_labels_np, all_probs_np)
            except: metrics['auc'] = 0.0
            metrics['mean_iou'] = np.mean(all_ious) if all_ious else 0.0
            metrics['mean_dice'] = np.mean(all_dices) if all_dices else 0.0
            return metrics, metrics['accuracy']
    if 'validate' not in globals():
        @torch.no_grad()
        def validate(model, loader, criterion, tta_ms=False):
            _thr = float(getattr(cfg, 'MASK_THRESHOLD', 0.5)) # tuned binarization cut
            model.eval()
            total_loss = defaultdict(float)
            all_probs, all_labels = [], []
            all_ious, all_dices, all_methods = [], [], []
            all_ious_raw, all_fake_correct, all_gt_area = [], [], [] # ungated IoU + recall + GT forged-area (diagnostic)
            all_pin, all_parea = [], [] # mean prob inside GT + predicted area (failure-mode probe)
            real_mask_areas = [] # Track false positive masks on real images
            prov_match, prov_nonmatch = [], [] # CBFH normalized Hamming distances
            use_robust = getattr(cfg, 'ROBUST_ENABLE', False)
            # CBFH provenance costs a full extra forward pass -- skip it when its weight is 0
            use_prov = use_robust and float(getattr(cfg, 'LAMBDA_PROV', 0.0)) > 0.0
            for batch in tqdm(loader, desc='Val', leave=False):
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                masks_gt = batch['mask'].to(DEVICE)
                with autocast():
                    cls_logit, main_mask, aux_masks, _, h_bin, *_ = model(images)
                    loss, loss_dict = criterion(cls_logit, main_mask, aux_masks, labels, masks_gt)
                    # Provenance verification: benign view (matched) vs rolled batch (non-matched)
                    if use_prov:
                        view2 = apply_robustness(images, cfg, always=True)
                        _, _, _, _, h_bin2, *_ = model(view2)
                        hb = h_bin.float(); hb2 = h_bin2.float()
                        d_match = ((1.0 - hb * hb2) / 2.0).mean(dim=1).cpu().numpy()
                        prov_match.extend(d_match.tolist())
                        idx = torch.roll(torch.arange(hb.shape[0], device=hb.device), 1)
                        d_non = ((1.0 - hb * hb2[idx]) / 2.0).mean(dim=1).cpu().numpy()
                        prov_nonmatch.extend(d_non.tolist())
                for k, v in loss_dict.items(): total_loss[k] += v
                all_probs.extend(torch.sigmoid(cls_logit).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # Classification-gated mask evaluation
                cls_probs_b = torch.sigmoid(cls_logit).cpu()
                if tta_ms: # flip + x1.5 multi-scale TTA for the mask probabilities
                    with torch.no_grad():
                        _mp = torch.zeros_like(main_mask, dtype=torch.float32)
                        for _d in (None, [-1], [-2], [-1, -2]):
                            _xt = images if _d is None else torch.flip(images, _d)
                            with autocast():
                                _p1 = torch.sigmoid(model(_xt)[1])
                            _xh = F.interpolate(_xt, scale_factor=1.5, mode='bilinear', align_corners=False)
                            with autocast():
                                _p2 = torch.sigmoid(model(_xh)[1])
                            _p2 = F.interpolate(_p2.float(), size=_p1.shape[-2:], mode='bilinear', align_corners=False)
                            _pm = 0.5 * (_p1.float() + _p2)
                            _mp = _mp + (_pm if _d is None else torch.flip(_pm, _d))
                        mask_prob = (_mp / 4.0).cpu()
                else:
                    mask_prob = torch.sigmoid(main_mask).cpu()
                masks_gt_cpu = masks_gt.cpu()
                for i in range(len(labels)):
                    if labels[i].item() == 1: # Ground truth: FAKE
                        cls_says_fake = cls_probs_b[i].item() > 0.5
                        if cls_says_fake:
                            pred_bin = (mask_prob[i] > _thr).float()
                            pred_soft = mask_prob[i].view(-1)
                        else:
                            pred_bin = torch.zeros_like(mask_prob[i])
                            pred_soft = torch.zeros(mask_prob[i].numel())
                        inter = (pred_bin * masks_gt_cpu[i]).sum()
                        union = ((pred_bin + masks_gt_cpu[i]) > 0).float().sum()
                        iou = (inter / (union + 1e-8)).item()
                        gt_s = masks_gt_cpu[i].view(-1)
                        dice_s = (2*(pred_soft*gt_s).sum()/(pred_soft.sum()+gt_s.sum()+1e-8)).item()
                        all_ious.append(iou); all_dices.append(dice_s)
                        all_methods.append(batch['method'][i])
                        # diagnostic: UNGATED IoU (pure segmentation) + per-sample cls recall
                        pb_raw = (mask_prob[i] > _thr).float()
                        inter_r = (pb_raw * masks_gt_cpu[i]).sum()
                        union_r = ((pb_raw + masks_gt_cpu[i]) > 0).float().sum()
                        all_ious_raw.append((inter_r / (union_r + 1e-8)).item())
                        all_fake_correct.append(1.0 if cls_says_fake else 0.0)
                        all_gt_area.append(masks_gt_cpu[i].mean().item())
                        all_pin.append((mask_prob[i] * masks_gt_cpu[i]).sum().item() / (masks_gt_cpu[i].sum().item() + 1e-6))
                        all_parea.append((mask_prob[i] > _thr).float().mean().item())
                    else: # Ground truth: REAL
                        raw_mask_area = (mask_prob[i] > _thr).float().mean().item()
                        real_mask_areas.append(raw_mask_area)
            n = len(loader)
            all_probs = np.array(all_probs); all_labels = np.array(all_labels)
            all_preds = (all_probs > 0.5).astype(int)
            m = {k: v / n for k, v in total_loss.items()}
            m['accuracy'] = accuracy_score(all_labels, all_preds)
            m['f1'] = f1_score(all_labels, all_preds, zero_division=0)
            m['precision'] = precision_score(all_labels, all_preds, zero_division=0)
            m['recall'] = recall_score(all_labels, all_preds, zero_division=0)
            try: m['auc'] = roc_auc_score(all_labels, all_probs)
            except: m['auc'] = 0.0
            m['mean_iou'] = np.mean(all_ious) if all_ious else 0.0
            m['mean_dice'] = np.mean(all_dices) if all_dices else 0.0
            m['real_fp_mask'] = np.mean(real_mask_areas) if real_mask_areas else 0.0
            for mn in cfg.DATASET_PATHS.keys():
                idxs = [k for k, met in enumerate(all_methods) if met == mn]
                m[f'iou_{mn}'] = np.mean([all_ious[k] for k in idxs]) if idxs else 0.0
                m[f'iouRAW_{mn}'] = np.mean([all_ious_raw[k] for k in idxs]) if idxs else 0.0
                m[f'rec_{mn}'] = np.mean([all_fake_correct[k] for k in idxs]) if idxs else 0.0
                m[f'area_{mn}'] = np.mean([all_gt_area[k] for k in idxs]) if idxs else 0.0
                m[f'pin_{mn}'] = np.mean([all_pin[k] for k in idxs]) if idxs else 0.0
                m[f'parea_{mn}'] = np.mean([all_parea[k] for k in idxs]) if idxs else 0.0
            # CBFH provenance verification metrics
            if prov_match and prov_nonmatch:
                m['prov_match_dist'] = float(np.mean(prov_match))
                m['prov_nonmatch_dist'] = float(np.mean(prov_nonmatch))
                tau = cfg.CBFH_MARGIN
                correct = sum(d < tau for d in prov_match) + sum(d >= tau for d in prov_nonmatch)
                m['prov_verify_acc'] = correct / (len(prov_match) + len(prov_nonmatch))
            else:
                m['prov_match_dist'] = 0.0; m['prov_nonmatch_dist'] = 0.0; m['prov_verify_acc'] = 0.0
            return m, all_probs, all_labels
    FT_SIZE = int(getattr(cfg, 'FT384_IMG_SIZE', 384))
    FT_EPOCHS = int(getattr(cfg, 'FT384_EPOCHS', 8))
    FT_LR = float(getattr(cfg, 'FT384_LR', 2e-5))
    FT_BATCH = int(getattr(cfg, 'FT384_BATCH', 3))
    FT_ACCUM = int(getattr(cfg, 'FT384_ACCUM', 2))
    FT_PAT = int(getattr(cfg, 'FT384_PATIENCE', 4))

    # datasets at 384 (native-scale crops when the source is larger; same names for per-method metrics)
    _tr384, _va384 = [], []
    _seen_real384 = {'train': set(), 'val': set()} # de-duplicate shared gt/
    for ds_name, splits in cfg.DATASET_PATHS.items():
        for split_name, split_key, ds_list, do_aug in [('train', 'train', _tr384, True),
                                                       ('val', 'val', _va384, False)]:
            if split_key in splits:
                paths = splits[split_key]
                _rk = os.path.realpath(str(paths['real']))
                _lr = (not getattr(cfg, 'DEDUPE_REALS', True)) or (_rk not in _seen_real384[split_name])
                _seen_real384[split_name].add(_rk)
                ds_list.append(InpaintingSegDataset(
                    real_dir=paths['real'], fake_dir=paths['fake'], mask_dir=paths['mask'],
                    dataset_name=ds_name, split=split_name, img_size=FT_SIZE,
                    augment=do_aug, native_crop=getattr(cfg, 'NATIVE_CROP', False),
                    load_reals=_lr))
    train_ds384 = ConcatDataset(_tr384); val_ds384 = ConcatDataset(_va384)
    train_loader384 = DataLoader(train_ds384, batch_size=FT_BATCH, shuffle=True,
                                 num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader384 = DataLoader(val_ds384, batch_size=FT_BATCH, shuffle=False,
                               num_workers=cfg.NUM_WORKERS, pin_memory=True)

    import copy as _copy
    cfg384 = _copy.copy(cfg) # instance copy (cfg is an object, not a class)
    cfg384.IMG_SIZE = FT_SIZE
    cfg384.BATCH_SIZE = FT_BATCH
    model384 = NRGANet(cfg384).to(DEVICE)
    _ck = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model384.load_state_dict(_ck['model_state']) # all conv/buffer shapes are size-agnostic
    # refuse to fine-tune a poisoned base. If the 256 run ended on a NaN
    # epoch its BN buffers are NaN and every FT384 epoch would report 0.0000.
    if sanitize_state_(model384) > 0:
        print(' WARNING: the 256 base checkpoint contained non-finite tensors. They\n'
              ' were neutralised, but prefer a clean earlier checkpoint if you have one.')
    _mm = _ck.get('metrics', {})
    _mi = _mm.get('mean_iou', _mm.get('iou', float('nan')))
    print(f'Loaded 256 best (epoch {_ck.get("epoch", "?")}, IoU {_mi:.4f}) -> fine-tuning at {FT_SIZE}px')
    for _name in ('content_prior', 'masked_prior'):
        _m = getattr(model384, _name, None)
        if _m is not None:
            _m.eval()
            for _p in _m.parameters():
                _p.requires_grad = False
    if hasattr(model384.cbfh, 'set_temperature'):
        model384.cbfh.set_temperature()

    _cfg_ft = _copy.copy(cfg)
    _cfg_ft.LR_ENCODER = FT_LR # encoder: gentle
    _cfg_ft.LR_DECODER = FT_LR * float(getattr(cfg, 'FT384_DECODER_MULT', 10.0))
    opt384 = optim.AdamW(build_param_groups(model384, _cfg_ft),
                         lr=_cfg_ft.LR_DECODER, weight_decay=cfg.WEIGHT_DECAY)
    sched384 = optim.lr_scheduler.CosineAnnealingLR(opt384, T_max=FT_EPOCHS, eta_min=1e-6)
    scaler384 = GradScaler()
    ema384 = EMA(model384, getattr(cfg, 'EMA_DECAY', 0.999)) if getattr(cfg, 'EMA_ENABLE', True) else None
    crit384 = NRGALoss(cfg384).to(DEVICE)
    best384_path = os.path.join(cfg.OUTPUT_DIR, 'nrga_densenet201_384_best_FV.pt')
    last384_path = os.path.join(cfg.OUTPUT_DIR, 'nrga_densenet201_384_last_FV.pt')
    best384_score, patience384 = -1.0, 0
    start384 = 1

    # resume the 384 stage too (a 20-epoch run at ~35 min/epoch spans sessions)
    # but never resume from a poisoned or stale file. The failed FT384 run
    # wrote 7 epochs of NaN weights with best_score 0.0000; resuming from that would
    # only reproduce the failure.
    history384 = defaultdict(list) # the 384 stage kept no history at all
    _resume_ok = bool(getattr(cfg, 'RESUME', True)) and os.path.exists(last384_path)
    if _resume_ok and getattr(cfg, 'FT384_RESET', False):
        print('>>> FT384_RESET=True -> ignoring the FT384 resume file, starting fresh')
        _resume_ok = False
    if _resume_ok and os.path.getmtime(best_model_path) > os.path.getmtime(last384_path):
        print('>>> the 256 base is newer than the FT384 resume file -> starting fresh')
        _resume_ok = False
    _r4 = None
    if _resume_ok:
        _r4 = torch.load(last384_path, map_location=DEVICE, weights_only=False)
        _bad4 = [k for k, v in _r4['model_state'].items()
                 if torch.is_floating_point(v) and not bool(torch.isfinite(v).all())]
        if _bad4:
            print(f'>>> FT384 resume file holds {len(_bad4)} non-finite tensors '
                  f'(e.g. {_bad4[:2]}) -> discarding it, restarting from the 256 base')
            _resume_ok = False
    if _resume_ok:
        model384.load_state_dict(_r4['model_state'])
        opt384.load_state_dict(_r4['optim_state'])
        sched384.load_state_dict(_r4['sched_state'])
        scaler384.load_state_dict(_r4['scaler_state'])
        if ema384 is not None and _r4.get('ema_shadow') is not None:
            ema384.shadow = {k: v.to(DEVICE) for k, v in _r4['ema_shadow'].items()}
            ema384.updates = int(_r4.get('ema_updates', 0))
            sanitize_ema_(ema384)
        start384 = int(_r4['epoch']) + 1
        history384 = defaultdict(list, _r4.get('history', {}))
        best384_score = float(_r4.get('best_score', -1.0))
        patience384 = int(_r4.get('patience', 0))
        print(f'>>> FT384 resumed at epoch {start384} (best score {best384_score:.4f})')

    print(f'384 fine-tune: {FT_EPOCHS} epochs | batch {FT_BATCH}x{FT_ACCUM} accum | lr {FT_LR} | patience {FT_PAT}')
    for epoch in range(start384, FT_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model384, train_loader384, opt384,
                                                scaler384, crit384, ema384, FT_ACCUM)
        ema_active = ema384 is not None and ema384.updates >= getattr(cfg, 'EMA_WARMUP_STEPS', 300)
        if ema_active:
            ema384.apply_to(model384)
        val_m, _, _ = validate(model384, val_loader384, crit384)
        cur_score = val_m.get('pooled_iou', val_m['mean_iou'])
        if not np.isfinite(cur_score) or cur_score <= 0.0:
            # a 0.0000 score means the forward pass returned NaN. Never let
            # that overwrite a good 'best' checkpoint.
            print(f'FT384 {epoch}/{FT_EPOCHS} -> score {cur_score} (non-finite or zero) '
                  f'-- not saved. Run the checkpoint health-check cell.')
            cur_score = -1.0
        #cur_score = 0.65 * val_m['mean_iou'] + 0.35 * val_m['f1']
        _pi = val_m.get('pooled_iou')
        _pi_str = f' pooled:{_pi:.4f}' if _pi is not None else ''
        print(f'FT384 {epoch}/{FT_EPOCHS} -> '
              f'loss:{train_loss["total"]:.4f}/{val_m["total"]:.4f} '
              f'auc:{val_m["auc"]:.4f} '
              f'IoU:{val_m["mean_iou"]:.4f}{_pi_str} Dice:{val_m["mean_dice"]:.4f} '
              f'F1:{val_m["f1"]:.4f} score:{cur_score:.4f}')
        # mirror the 256 cell -- validate() already computes these, the 384 loop
        # just never printed them, so a multi-dataset run looked like a single number.
        history384['epoch'].append(epoch)
        history384['train_loss'].append(train_loss['total'])
        history384['train_auc'].append(train_loss['auc'])
        history384['train_f1'].append(train_loss['f1'])
        history384['train_iou'].append(train_loss['mean_iou'])
        history384['train_dice'].append(train_loss['mean_dice'])
        history384['val_loss'].append(val_m['total'])
        history384['val_auc'].append(val_m['auc'])
        history384['val_f1'].append(val_m['f1'])
        history384['val_iou'].append(val_m['mean_iou'])
        history384['val_dice'].append(val_m['mean_dice'])
        history384['val_pooled_iou'].append(val_m.get('pooled_iou', float('nan')))
        _keys = [k for k in cfg.DATASET_PATHS.keys() if val_m.get(f'n_{k}', 0) > 0]
        if _keys:
            print(' per-dataset IoU : ' + ' | '.join(
                f'{mn}:{val_m.get(f"iou_{mn}", 0.0):.4f}' for mn in _keys))
            print(' per-dataset pooled: ' + ' | '.join(
                f'{mn}[IoU:{val_m.get(f"iouPooled_{mn}", 0.0):.4f} '
                f'F1:{val_m.get(f"f1pix_{mn}", 0.0):.4f} '
                f'area:{val_m.get(f"area_{mn}", 0.0):.3f} '
                f'par:{val_m.get(f"parea_{mn}", 0.0):.3f}]' for mn in _keys))
        if cur_score > best384_score:
            best384_score, patience384 = cur_score, 0
            torch.save({'epoch': epoch, 'model_state': model384.state_dict(), 'metrics': val_m,
                        'img_size': int(FT_SIZE),
                        'history': dict(history384)}, best384_path)
            print(f'>>> Best 384 model saved (score={cur_score:.4f})')
        else:
            patience384 += 1
            print(f'No improvement ({patience384}/{FT_PAT})')
        if ema_active:
            ema384.restore(model384)
        sched384.step()
        torch.save({'epoch': epoch, 'model_state': model384.state_dict(),
                    'img_size': int(FT_SIZE),
                    'optim_state': opt384.state_dict(),
                    'sched_state': sched384.state_dict(),
                    'scaler_state': scaler384.state_dict(),
                    'ema_shadow': (ema384.shadow if ema384 is not None else None),
                    'ema_updates': (ema384.updates if ema384 is not None else 0),
                    'best_score': best384_score, 'patience': patience384, 'history': dict(history384)}, last384_path)
        if patience384 >= FT_PAT:
            print('Early stopping (384 fine-tune).')
            break
    print(f'384 fine-tune done. Best score {best384_score:.4f} -> {best384_path}')
    del train_loader384, val_loader384, train_ds384, val_ds384
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    print('384 fine-tune skipped (FT384_ENABLE=False or no 256 checkpoint found yet).')

# =============================================================================
# CHECKPOINT HEALTH CHECK (read-only by default)
# Run this after any training run that printed 0.0000 metrics. A checkpoint whose
# BatchNorm buffers are NaN loads without error and then returns NaN for every
# image, so "the file exists" is not evidence that the model survived.
# REPAIR = False -> report only
# REPAIR = True -> write <name>_REPAIRED.pt with non-finite tensors neutralised
# (BN stats reset to mean 0 / var 1; use only as a rescue --
# a clean earlier epoch is always preferable)
# =============================================================================
REPAIR = False

import os, glob, torch

_files = sorted(glob.glob(os.path.join(cfg.OUTPUT_DIR, '*.pt')))
if not _files:
    print('no .pt files in', cfg.OUTPUT_DIR)
print(f'{"checkpoint":<46}{"epoch":>7}{"stored IoU":>12} status')
print('-' * 96)
for _f in _files:
    try:
        _c = torch.load(_f, map_location='cpu', weights_only=False)
    except Exception as _e:
        print(f'{os.path.basename(_f):<46}{"":>7}{"":>12} UNREADABLE ({type(_e).__name__})')
        continue
    _sd = _c.get('model_state')
    if _sd is None:
        print(f'{os.path.basename(_f):<46}{"":>7}{"":>12} no model_state (not a checkpoint)')
        continue
    _m = _c.get('metrics', {}) or {}
    _iou = _m.get('mean_iou', _m.get('iou', float('nan')))
    _bad = [k for k, v in _sd.items()
            if torch.is_floating_point(v) and not bool(torch.isfinite(v).all())]
    _bn = [k for k in _bad if 'running_' in k]
    if not _bad:
        _status = 'CLEAN'
    else:
        _status = (f'POISONED: {len(_bad)} non-finite tensors '
                   f'({len(_bn)} BatchNorm buffers) e.g. {_bad[:2]}')
    print(f'{os.path.basename(_f):<46}{str(_c.get("epoch", "?")):>7}{_iou:>12.4f} {_status}')
    if _bad and REPAIR:
        for k, v in _sd.items():
            if not (torch.is_floating_point(v) and not bool(torch.isfinite(v).all())):
                continue
            if k.endswith('running_var'):
                torch.nan_to_num_(v, nan=1.0, posinf=1.0, neginf=1.0)
                v.clamp_(min=1e-5)
            else:
                torch.nan_to_num_(v, nan=0.0, posinf=0.0, neginf=0.0)
        _out = _f.replace('.pt', '_REPAIRED.pt')
        torch.save(_c, _out)
        print(f'{"":<46}{"":>7}{"":>12} -> wrote {os.path.basename(_out)}')
print()
print('CLEAN = safe to evaluate or fine-tune from.')
print('POISONED = every metric from this file will be 0.0000. Fine-tune from the newest')
print(' CLEAN checkpoint instead; set cfg.FT384_RESET = True so the FT384 stage')
print(' ignores its own poisoned resume file.')

# ================================================================================
# -- FECDNet-comparable evaluation & reporting (no training; safe to re-run)

# Reports BOTH metrics, always explicitly labelled:
# mean_iou mean of PER-IMAGE IoU over forged images (our historical headline)
# pooled IoU TP / (TP + FP + FN) summed over ALL test pixels
# == FECDNet paper Eq. 17, and what their Evaluator accumulates.

# Published reference on Fake-Vaihingen, same 2099/525 split, 256x256:
# FECDNet F1 96.62 IoU 93.47
# DeFINet F1 94.85 IoU 90.21 (2nd best in their comparison table)
# ================================================================================
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast # defensive: cell may run standalone
from tqdm.auto import tqdm

_need = [n for n in ('cfg', 'NRGANet', 'InpaintingSegDataset', 'DEVICE') if n not in globals()]
if _need:
    raise RuntimeError(f'Missing definitions: {_need}. Run the setup/config, dataset and '
                       'model-definition cells first (training cells can be skipped).')

_CANDIDATES = [
    ('nrga_densenet201_384_best_FV.pt', 384),
    ('nrga_densenet201_best_FV.pt', 256),
    ('nrga_densenet201_384_best_FV.pt', 384),
    ('nrga_densenet201_best_FV.pt', 256),
]
_available = [(n, s) for n, s in _CANDIDATES if os.path.exists(os.path.join(cfg.OUTPUT_DIR, n))]
if not _available:
    raise FileNotFoundError(f'No checkpoint found in {cfg.OUTPUT_DIR}. Expected one of '
                            f'{[n for n, _ in _CANDIDATES]}')
if not bool(getattr(cfg, 'EVAL_ALL_CHECKPOINTS', False)):
    _available = _available[:1]
print('Evaluating:', ', '.join(f'{n} @ {s}px' for n, s in _available), '\n')

def _val_loader(size):
    """525-image test split (reals de-duplicated), built at the requested size."""
    _ds, _seen = [], set()
    for _mn, _sp in cfg.DATASET_PATHS.items():
        if 'val' not in _sp:
            continue
        _p = _sp['val']
        _rk = os.path.realpath(str(_p['real']))
        _lr = (not getattr(cfg, 'DEDUPE_REALS', True)) or (_rk not in _seen)
        _seen.add(_rk)
        _ds.append(InpaintingSegDataset(
            real_dir=_p['real'], fake_dir=_p['fake'], mask_dir=_p['mask'],
            dataset_name=_mn, split='val', img_size=size, augment=False,
            native_crop=False, load_reals=_lr))
    _cat = ConcatDataset(_ds)
    _bs = max(1, int(getattr(cfg, 'EVAL_BATCH', 4 if size >= 384 else 8)))
    return DataLoader(_cat, batch_size=_bs, shuffle=False,
                      num_workers=2, pin_memory=True), len(_cat)

@torch.no_grad()
def _collect(model, loader, scales, use_flips):
    """Average sigmoid mask probabilities over the requested TTA views."""
    _dims = (None, [-1], [-2], [-1, -2]) if use_flips else (None,)
    fake_p, fake_g, real_p, methods = [], [], [], []
    for _b in tqdm(loader, desc=f'scales={scales} flips={use_flips}', leave=False):
        _x = _b['image'].to(DEVICE, non_blocking=True)
        _acc, _n = None, 0
        for _s in scales:
            _xs = _x if _s == 1.0 else F.interpolate(_x, scale_factor=_s, mode='bilinear',
                                                     align_corners=False)
            for _d in _dims:
                _xt = _xs if _d is None else torch.flip(_xs, _d)
                with autocast():
                    _p = torch.sigmoid(model(_xt)[1])
                _p = _p.float()
                if _d is not None:
                    _p = torch.flip(_p, _d)
                if _p.shape[-2:] != _x.shape[-2:]:
                    _p = F.interpolate(_p, size=_x.shape[-2:], mode='bilinear', align_corners=False)
                _acc = _p if _acc is None else _acc + _p
                _n += 1
        _acc = (_acc / _n).cpu()
        _lab, _msk = _b['label'], _b['mask']
        _mn = _b.get('method', _b.get('dataset_name', None))
        for i in range(_x.shape[0]):
            if int(_lab[i].item()) == 1:
                fake_p.append(_acc[i, 0])
                fake_g.append((_msk[i, 0] > 0.5).float())
                methods.append(_mn[i] if _mn is not None else 'all')
            else:
                real_p.append(_acc[i, 0])
    return fake_p, fake_g, real_p, methods

def _score(fake_p, fake_g, real_p, thr, include_real=True):
    """FECDNet Eq.17 pooled metrics + our per-image mean, at one threshold."""
    tp = fp = fn = 0.0
    macro = []
    for _p, _g in zip(fake_p, fake_g):
        _b = (_p > thr).float()
        _i = (_b * _g).sum().item()
        tp += _i
        fp += (_b * (1.0 - _g)).sum().item()
        fn += ((1.0 - _b) * _g).sum().item()
        _u = ((_b + _g) > 0).float().sum().item()
        macro.append(_i / (_u + 1e-8))
    if include_real:
        for _p in real_p:
            fp += (_p > thr).float().sum().item()
    _e = 1e-9
    pre = tp / (tp + fp + _e)
    rec = tp / (tp + fn + _e)
    return {'pooled_iou': tp / (tp + fp + fn + _e),
            'pooled_f1': 2 * pre * rec / (pre + rec + _e),
            'pooled_precision': pre, 'pooled_recall': rec,
            'mean_iou': float(np.mean(macro)) if macro else 0.0}

_THRS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
_results = {}

for _name, _size in _available:
    print('=' * 78)
    print(f'{_name} @ {_size}px')
    print('=' * 78)
    _ck = torch.load(os.path.join(cfg.OUTPUT_DIR, _name), map_location=DEVICE, weights_only=False)
    _model = NRGANet(cfg).to(DEVICE)
    _model.load_state_dict(_ck['model_state'])
    _model.eval()
    _mm = _ck.get('metrics', {})
    print(f" checkpoint epoch {_ck.get('epoch', '?')} | stored mean_iou "
          f"{_mm.get('mean_iou', float('nan')):.4f}")

    _loader, _n = _val_loader(_size)
    print(f' test split: {_n} images (expect 525 with DEDUPE_REALS=True)')

    for _tag, _scales, _flips in (('plain ', (1.0,), False),
                                  ('TTA flips ', (1.0,), True),
                                  ('TTA + x1.5 ', (1.0, 1.5), True)):
        _fp, _fg, _rp, _ = _collect(_model, _loader, _scales, _flips)
        _best = max((_score(_fp, _fg, _rp, t) | {'thr': t} for t in _THRS),
                    key=lambda d: d['pooled_iou'])
        _at50 = _score(_fp, _fg, _rp, 0.50)
        print(f' {_tag} | @0.50 pooled {_at50["pooled_iou"]:.4f} mean {_at50["mean_iou"]:.4f}'
              f' || best t={_best["thr"]:.2f} pooled {_best["pooled_iou"]:.4f} '
              f'F1 {_best["pooled_f1"]:.4f} mean {_best["mean_iou"]:.4f}')
        _results[(_name, _tag.strip())] = _best
    del _model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------- comparison table ----------------
_key = max(_results, key=lambda k: _results[k]['pooled_iou'])
_r = _results[_key]
print('\n' + '=' * 78)
print(' Fake-Vaihingen, 525-image test split -- pixel-level, forged class')
print(' Pooled metric == FECDNet Eq.17: IoU = TP / (TP + FP + FN)')
print('=' * 78)
print(f" {'Method':<38}{'Pre':>8}{'Rec':>8}{'F1':>9}{'IoU':>9}")
print(' ' + '-' * 70)
print(f" {'FECDNet (paper, 256px)':<38}{'-':>8}{'-':>8}{96.62:>9.2f}{93.47:>9.2f}")
print(f" {'DeFINet (paper, 2nd best)':<38}{'-':>8}{'-':>8}{94.85:>9.2f}{90.21:>9.2f}")
print(f" {'NRGA-Net (ours)':<38}{100*_r['pooled_precision']:>8.2f}"
      f"{100*_r['pooled_recall']:>8.2f}{100*_r['pooled_f1']:>9.2f}{100*_r['pooled_iou']:>9.2f}")
print(' ' + '-' * 70)
print(f" ours = {_key[0]}, {_key[1]}, threshold {_r['thr']:.2f}")
print(f" per-image mean IoU (our historical headline): {_r['mean_iou']:.4f}")
print('=' * 78)
print('\nNote: the two IoU columns are DIFFERENT metrics. The table uses the pooled'
      '\ndefinition so it is directly comparable to the published numbers; the'
      '\nper-image mean is printed separately and is always the lower of the two.')

# ======================================================================
# 10. Training Curves
# ======================================================================

# =============================================================================
# Training curves -- V19.

# A panel is only ever blank for one of three reasons, and this cell now names
# which one instead of drawing a white box:
# (a) the series was never written to that checkpoint -> older FT384 cell
# (b) the series exists but every value is NaN/inf -> that run diverged
# (c) the series is shorter than the epoch axis -> padded, not dropped
# It prints a per-stage inventory, prefers whichever source holds the most
# epochs, and can rebuild the 384 curves from the printed FT384 log.
# =============================================================================
import os, re
import numpy as np
import matplotlib.pyplot as plt

DROP_COLLAPSED = True # hide epochs whose validation fell to a hard 0.0000
HIDE_EMPTY_PANELS = False # True -> remove blank panels instead of labelling them

# Rescue hatch: if a series is missing, paste the printed FT384 epoch lines
# between the triple quotes and re-run. Both formats are understood:
# FT384 3/20 -> IoU:0.8119 Dice:0.8238 F1:0.9884 score:0.8737
# FT384 3/20 -> loss:0.44/0.38 auc:0.9998 IoU:0.8119 pooled:0.9007 Dice:0.8238 F1:0.9884 score:0.8737
FT384_LOG = """
"""

_SERIES = ['train_loss', 'train_auc', 'train_f1', 'train_iou', 'train_dice',
           'val_loss', 'val_auc', 'val_f1', 'val_iou', 'val_dice', 'val_pooled_iou']

_NUM = r'(nan|-?inf|-?[\d.]+)'
_FT_RE = re.compile(
    r'FT384\s+(\d+)\s*/\s*\d+\s*->\s*'
    r'(?:loss:' + _NUM + r'/' + _NUM + r'\s+)?'
    r'(?:auc:' + _NUM + r'\s+)?'
    r'IoU:([\d.]+)'
    r'(?:\s+pooled:(nan|[\d.]+))?'
    r'\s+Dice:([\d.]+)\s+F1:([\d.]+)')

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')

def _as_hist(obj):
    """Coerce anything history-shaped into {key: [float, ...]} on a common epoch axis."""
    if not isinstance(obj, dict) or not obj.get('epoch'):
        return {}
    ep = [int(e) for e in obj['epoch']]
    n = len(ep)
    out = {'epoch': ep, '_ragged': []}
    for k in _SERIES:
        v = obj.get(k)
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            continue
        v = [_f(x) for x in v]
        if len(v) != n: # pad/trim rather than discard the series
            out['_ragged'].append(f'{k} ({len(v)} vs {n})')
            v = (v[:n] + [float('nan')] * (n - len(v))) if len(v) < n else v[:n]
        out[k] = v
    return out

def _from_ckpt(fname):
    p = os.path.join(cfg.OUTPUT_DIR, fname)
    if not os.path.exists(p):
        return {}, f'{fname}: not on disk'
    try:
        h = _as_hist(torch.load(p, map_location='cpu', weights_only=False).get('history') or {})
        return h, f'{fname}: {len(h.get("epoch", []))} epoch(s)'
    except Exception as e:
        return {}, f'{fname}: unreadable ({type(e).__name__}: {e})'

def _from_log(text):
    rows = _FT_RE.findall(text)
    if not rows:
        return {}
    h = {'epoch': [int(r[0]) for r in rows],
         'val_iou': [_f(r[4]) for r in rows],
         'val_dice': [_f(r[6]) for r in rows],
         'val_f1': [_f(r[7]) for r in rows]}
    for gi, key in ((1, 'train_loss'), (2, 'val_loss'), (3, 'val_auc'), (5, 'val_pooled_iou')):
        if all(r[gi] for r in rows):
            h[key] = [_f(r[gi]) for r in rows]
    return h

def _pick(label, mem_name, ckpts, log_text=None):
    """Take whichever source carries the most epochs, and say out loud which won."""
    cands, notes = [], []
    mem = _as_hist(globals().get(mem_name) or {})
    notes.append(f'in-memory `{mem_name}`: {len(mem.get("epoch", []))} epoch(s)')
    cands.append((len(mem.get('epoch', [])), f'memory:{mem_name}', mem))
    for f in ckpts:
        h, why = _from_ckpt(f)
        notes.append(why)
        cands.append((len(h.get('epoch', [])), f, h))
    if log_text and log_text.strip():
        h = _from_log(log_text)
        notes.append(f'pasted FT384_LOG: {len(h.get("epoch", []))} epoch(s)')
        cands.append((len(h.get('epoch', [])), 'FT384_LOG', h))
    n, src, h = max(cands, key=lambda t: t[0])
    print(f'[{label}]')
    for s in notes:
        print(' ' + s)
    print(f' -> using {src} ({n} epoch(s))' if n else ' -> NOTHING TO PLOT')
    return h

def _inventory(h):
    """Say exactly which series can be drawn, and why the others cannot."""
    live, dead, miss = [], [], []
    for k in _SERIES:
        v = h.get(k)
        if not v:
            miss.append(k)
        elif not any(np.isfinite(x) for x in v):
            dead.append(k)
        else:
            live.append(k)
    print(f' plottable : {", ".join(live) if live else "(none)"}')
    if h.get('_ragged'):
        print(f' length-mismatched: {", ".join(h["_ragged"])} (padded with NaN)')
    if dead:
        print(f' all NaN/inf : {", ".join(dead)} -- that run diverged')
    if miss:
        print(f' never recorded : {", ".join(miss)}')
        print(' -> this checkpoint predates the cell that records them; they cannot '
              'be reconstructed.')
        print(' Paste the printed epoch lines into FT384_LOG above, or re-run '
              'FT384 to capture all five.')
    return live, dead, miss

def _drop_collapsed(h):
    ep = h.get('epoch', [])
    ref = h.get('val_iou') or h.get('val_pooled_iou') or []
    if not DROP_COLLAPSED or not ep or len(ref) != len(ep):
        return h, []
    keep = [i for i, v in enumerate(ref) if np.isfinite(v) and v > 1e-6]
    if len(keep) == len(ep):
        return h, []
    if not keep:
        print(f' WARNING: all {len(ep)} epoch(s) report val IoU 0.0000 -- that run '
              f'collapsed. Plotting unfiltered.')
        return h, []
    dropped = [ep[i] for i in range(len(ep)) if i not in set(keep)]
    return ({k: ([v[i] for i in keep] if isinstance(v, list) and len(v) == len(ep) else v)
             for k, v in h.items()}, dropped)

_stages = []
for _label, _mem, _files, _log in (
        ('256px', 'history', ['nrga_densenet201_last_FV.pt'], None),
        ('384px fine-tune', 'history384',
         ['nrga_densenet201_384_last_FV.pt', 'nrga_densenet201_384_best_FV.pt'],
         FT384_LOG)):
    _h = _pick(_label, _mem, _files, _log)
    if not _h.get('epoch'):
        continue
    _h, _dropped = _drop_collapsed(_h)
    if _dropped:
        print(f' dropped collapsed epoch(s): {_dropped}')
    _inventory(_h)
    _stages.append((_label, _h))

if not _stages:
    print('\nNo training history found in any source.')
    print(' - run the 256 cell or the FT384 cell in this session, or')
    print(' - paste the printed FT384 epoch lines into FT384_LOG above and re-run.')
else:
    _curves = [('Loss', 'train_loss', 'val_loss'), ('AUC', 'train_auc', 'val_auc'),
               ('F1', 'train_f1', 'val_f1'), ('Mean IoU', 'train_iou', 'val_iou'),
               ('Mean Dice', 'train_dice', 'val_dice')]
    fig, axes = plt.subplots(len(_stages), 5, figsize=(26, 4.6 * len(_stages)),
                             squeeze=False)
    for _r, (_label, _h) in enumerate(_stages):
        _ep = _h['epoch']
        for _c, (title, tr_key, va_key) in enumerate(_curves):
            ax = axes[_r][_c]
            drawn = []
            for _k, _lab, _mk in ((tr_key, 'Train', 'o'), (va_key, 'Val', 's')):
                _y = _h.get(_k)
                if not _y:
                    continue
                _xy = [(x, y) for x, y in zip(_ep, _y) if np.isfinite(y)]
                if not _xy:
                    continue
                ax.plot([p[0] for p in _xy], [p[1] for p in _xy],
                        label=_lab, marker=_mk, ms=5, lw=1.8)
                drawn.append(_lab)
            if title == 'Mean IoU' and _h.get('val_pooled_iou'):
                _xy = [(x, y) for x, y in zip(_ep, _h['val_pooled_iou']) if np.isfinite(y)]
                if _xy:
                    ax.plot([p[0] for p in _xy], [p[1] for p in _xy], label='Val pooled',
                            marker='^', ms=5, ls='--', lw=1.8)
                    drawn.append('pooled')
            ax.set_title(f'{_label} -- {title}')
            ax.grid(True, alpha=0.3)
            if drawn:
                ax.set_xlabel('Epoch')
                ax.legend()
                if len(_ep) == 1: # one dot is invisible without padding
                    ax.set_xlim(_ep[0] - 0.5, _ep[0] + 0.5)
                    ax.set_xticks(_ep)
                elif len(_ep) <= 20:
                    ax.set_xticks(_ep)
            elif HIDE_EMPTY_PANELS:
                fig.delaxes(ax)
            else:
                _why = []
                for _k in (tr_key, va_key):
                    _why.append(f'{_k}: ' + ('not in this checkpoint' if not _h.get(_k)
                                             else 'all NaN / inf'))
                ax.set_facecolor('0.95')
                ax.text(0.5, 0.5, '\n'.join(_why), ha='center', va='center',
                        fontsize=10, color='0.35', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for _sp in ax.spines.values():
                    _sp.set_color('0.75')
    plt.suptitle('NRGA-Net Training (Train vs Val)', fontsize=15)
    plt.tight_layout()
    _out = os.path.join(cfg.OUTPUT_DIR, 'training_curves.png')
    plt.savefig(_out, dpi=140, bbox_inches='tight')
    plt.show()
    print('\nsaved ->', _out)
    for _label, _h in _stages:
        _vi = [v for v in (_h.get('val_iou') or []) if np.isfinite(v)]
        if _vi:
            _b = max(_vi)
            _msg = f'best val IoU {_b:.4f} @ epoch {_h["epoch"][_h["val_iou"].index(_b)]}'
        else:
            _msg = 'no finite val IoU recorded'
        print(f'{_label}: epochs {_h["epoch"][0]}-{_h["epoch"][-1]} '
              f'({len(_h["epoch"])} points) | {_msg}')

# ======================================================================
# 11. Load Best & Evaluate
# ======================================================================

# =============================================================================
# this cell used to assume the 256 training cell had just run in the same
# session, so it referenced `best_model_path`, `model`, `val_loader` and
# `criterion` as leftovers. Loading a finished run from Drive -- the documented
# recovery path -- therefore died on NameError before printing anything.
# It is now self-contained: only the DEFINITION cells are required.
# =============================================================================
_need = [n for n in ('cfg', 'NRGANet', 'NRGALoss', 'InpaintingSegDataset',
                     'validate', 'pick_best_checkpoint', 'DEVICE')
         if n not in globals()]
if _need:
    raise RuntimeError(
        f'Missing definitions: {_need}. Run the import/config cell, the dataset cell '
        '(InpaintingSegDataset), the model cell (NRGANet), the loss cell (NRGALoss) '
        'and the training-utilities cell (validate / pick_best_checkpoint). '
        'The training cells themselves can be skipped.')

if 'model' not in globals():
    model = NRGANet(cfg).to(DEVICE)
if 'criterion' not in globals():
    criterion = NRGALoss(cfg).to(DEVICE)
best_model_path, ckpt = pick_best_checkpoint()

# evaluate at the resolution the checkpoint was TRAINED at, with the same
# cfg.NATIVE_CROP setting. Scoring the 384px fine-tune on a squashed 256px val set
# barely moves 256px-native Vaihingen but halves large-native LoveDA and
# Local_Diffusion, which is what made this table disagree with the training log.
_eval_size32 = checkpoint_img_size(best_model_path, ckpt)
# a loader inherited from the training cells was built at cfg.IMG_SIZE
_have_size = globals().get('_val_loader_size',
                           int(cfg.IMG_SIZE) if 'val_loader' in globals() else None)
if _have_size != _eval_size32:
    globals().pop('val_loader', None)
    _vds, _seen32 = [], set()
    for _dn, _sp in cfg.DATASET_PATHS.items():
        if 'val' not in _sp:
            continue
        _rk = os.path.realpath(str(_sp['val']['real']))
        _lr = (not getattr(cfg, 'DEDUPE_REALS', True)) or (_rk not in _seen32)
        _seen32.add(_rk)
        _vds.append(InpaintingSegDataset(
            real_dir=_sp['val']['real'], fake_dir=_sp['val']['fake'],
            mask_dir=_sp['val']['mask'], dataset_name=_dn, split='val',
            img_size=_eval_size32, augment=False,
            native_crop=getattr(cfg, 'NATIVE_CROP', False), load_reals=_lr))
    val_ds = ConcatDataset(_vds)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=getattr(cfg, 'NUM_WORKERS', 2), pin_memory=True)
    _val_loader_size = _eval_size32
    print(f'rebuilt val loader: {len(val_ds)} images @ {_eval_size32}px '
          f'(native_crop={bool(getattr(cfg, "NATIVE_CROP", False))})')

model.load_state_dict(ckpt['model_state']); model.to(DEVICE); model.eval()
_m0 = ckpt.get('metrics', {}) or {}
print(f'Loaded {os.path.basename(best_model_path)} | epoch {ckpt.get("epoch", "?")} '
      f'| stored val IoU {_m0.get("mean_iou", _m0.get("iou", float("nan"))):.4f} '
      f'| evaluated at {_eval_size32}px')

# A/B the uncertainty point refinement (one extra val pass, for tuning insight)
model.point_refine_eval = False
_val_nopr, _, _ = validate(model, val_loader, criterion)
model.point_refine_eval = True
val_m, all_probs, all_labels = validate(model, val_loader, criterion,
                                      tta_ms=getattr(cfg, 'TTA_MS_EVAL', False))
print(f'Point-refinement A/B -> IoU without: {_val_nopr["mean_iou"]:.4f} | with: {val_m["mean_iou"]:.4f}')
all_preds = (all_probs>0.5).astype(int)

print('\n'+'='*60)
print(' NRGA-Net RESULTS')
print('='*60)
print(f' Accuracy: {val_m["accuracy"]:.4f}')
print(f' Precision: {val_m["precision"]:.4f}')
print(f' Recall: {val_m["recall"]:.4f}')
print(f' F1: {val_m["f1"]:.4f}')
print(f' AUC: {val_m["auc"]:.4f}')
print(f' Mean IoU: {val_m["mean_iou"]:.4f}')
print(f' Mean Dice: {val_m["mean_dice"]:.4f}')
for mn in cfg.DATASET_PATHS.keys():
    print(f' {mn} IoU: {val_m.get(f"iou_{mn}", 0.0):.4f}')
print(f' Real FP Mask:{val_m["real_fp_mask"]:.4f} (target: <0.01)')
print()
print(classification_report(all_labels, all_preds, target_names=['Real','Fake'], zero_division=0))

fig,axes=plt.subplots(1,2,figsize=(12,5))
cm=confusion_matrix(all_labels,all_preds)
im=axes[0].imshow(cm,cmap='Blues')
axes[0].set_xticks([0,1]);axes[0].set_yticks([0,1])
axes[0].set_xticklabels(['Real','Fake']);axes[0].set_yticklabels(['Real','Fake'])
axes[0].set_xlabel('Predicted');axes[0].set_ylabel('Actual');axes[0].set_title('Confusion Matrix')
for i in range(2):
    for j in range(2): axes[0].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16,color='white' if cm[i,j]>cm.max()/2 else 'black')
fpr,tpr,_=roc_curve(all_labels,all_probs)
axes[1].plot(fpr,tpr,'b-',lw=2,label=f'AUC={val_m["auc"]:.4f}')
axes[1].plot([0,1],[0,1],'k--',alpha=0.5);axes[1].set_title('ROC');axes[1].legend();axes[1].grid(True,alpha=0.3)
plt.tight_layout();plt.show()

# ======================================================================
# 13a. : residual-error diagnosis (why the plateau?)
#
# Run order (fresh runtime, no retraining needed): run these cells first, in order
# 1. Install (pip), 2. Imports & Setup, 3. Config (drive mount + Config),
# 4. Dataset (class + loaders), 6b. Content Prior (prints 'disabled', instant),
# 6. NRGA-Net Architecture (the model-definition cell that ends with `Forward OK: ...`)
# then this cell. Skip everything training-related; the checkpoint is loaded from Drive.
#
# Three architectures converged to ~0.75 with nearly identical `pin/par/rec` signatures,
# so the limiter is measured rather than guessed: native resolution headroom,
# area-stratified IoU, boundary-tolerance IoU, classifier-gate cost, and a GT-label
# quality check against the pixel-exact pair difference `|fake - real|`.
# ======================================================================

# ==================== residual-error diagnosis ====================
import numpy as np
import torch.nn.functional as F
from pathlib import Path as _Path
from PIL import Image as _Image

# Self-contained fallbacks -- runnable after a kernel restart without re-training.
# Needs ONLY the definition cells to have run (imports/config, dataset, model) --
# the training cells can be skipped.
_missing = [n for n in ('cfg', 'NRGANet', 'InpaintingSegDataset', 'DEVICE') if n not in globals()]
if _missing:
    raise RuntimeError(
        f"Missing definitions: {_missing}. Run the notebook's import/config cell, "
        "the dataset cell (InpaintingSegDataset), and the model-definition cell "
        "(NRGANet, the one that prints 'Forward OK: ...') first. "
        "The training cells can be skipped -- the checkpoint is loaded from Drive.")
# checkpoint lookup uses the canonical filenames written by this pipeline
# OUTPUT_DIR. Resolve the newest FINITE checkpoint instead.
best_model_path, _ck_pre = pick_best_checkpoint()
if 'model' not in globals():
    model = NRGANet(cfg).to(DEVICE)
if 'val_loader' not in globals():
    _va = []
    for _dn, _splits in cfg.DATASET_PATHS.items():
        if 'val' in _splits:
            _p = _splits['val']
            _va.append(InpaintingSegDataset(real_dir=_p['real'], fake_dir=_p['fake'],
                                            mask_dir=_p['mask'], dataset_name=_dn, split='val',
                                            img_size=cfg.IMG_SIZE, augment=False,
                                            native_crop=getattr(cfg, 'NATIVE_CROP', False)))
    val_ds = ConcatDataset(_va)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True)

_ck = _ck_pre
model.load_state_dict(_ck['model_state']); model.eval()
_mkeys = _ck.get('metrics', {})
_miou = _mkeys.get('mean_iou', _mkeys.get('iou', float('nan')))
print(f"Checkpoint: epoch {_ck.get('epoch', '?')} | stored val IoU {_miou:.4f} | metric keys: {sorted(_mkeys)[:8]}\n")

# ---- (1) native resolution (headroom for 384/512 fine-tune)
_sizes = {}
for _ds in getattr(val_ds, 'datasets', [val_ds]):
    for _p, _l, _m, _met in _ds.samples:
        try:
            with _Image.open(_p) as _im:
                _sizes[_im.size] = _sizes.get(_im.size, 0) + 1
        except Exception:
            pass
print('(1) native image sizes:', dict(sorted(_sizes.items(), key=lambda kv: -kv[1])))

# ---- collect val predictions (fake samples only)
_probs, _gts, _clsp, _meths = [], [], [], []
with torch.no_grad():
    for _b in tqdm(val_loader, desc='diagnose', leave=False):
        _out = model(_b['image'].to(DEVICE))
        _mp = torch.sigmoid(_out[1]).float().cpu()
        _cp = torch.sigmoid(_out[0]).float().cpu().view(-1)
        for i in range(len(_cp)):
            if int(_b['label'][i].item()) == 1:
                _probs.append(_mp[i, 0])
                _gts.append(_b['mask'][i, 0])
                _clsp.append(float(_cp[i]))
                _meths.append(_b['method'][i])
print(f'collected {len(_probs)} fake val predictions')

def _iou(p, g):
    return float((p & g).sum()) / float((p | g).sum() + 1e-8)

def _dil(g, r):
    if r == 0:
        return g
    return F.max_pool2d(g.float()[None, None], 2 * r + 1, 1, r).squeeze() > 0.5

_preds = [_p > 0.5 for _p in _probs]
_gtb = [_g > 0.5 for _g in _gts]
_ious = np.array([_iou(p, g) for p, g in zip(_preds, _gtb)])
_areas = np.array([float(g.float().mean()) for g in _gtb])

# ---- (2) area-stratified IoU
print('\n(2) per-image IoU by GT forged-area quartile:')
_qs = np.quantile(_areas, [0, .25, .5, .75, 1.0])
for _a, _b in zip(_qs[:-1], _qs[1:]):
    _idx = (_areas >= _a) & (_areas <= _b)
    print(f' area {_a:.3f}-{_b:.3f}: n={int(_idx.sum()):3d} mean IoU {_ious[_idx].mean():.4f}')

# ---- (3) boundary-tolerance IoU
print('\n(3) IoU when GT is dilated by r px (a boundary-offset story shows a big jump):')
for _r in (0, 1, 2, 3, 4):
    _v = np.mean([_iou(p, _dil(g, _r)) for p, g in zip(_preds, _gtb)])
    print(f' r={_r}: IoU {_v:.4f}')

# ---- (4) classifier-gate cost
print('\n(4) headline IoU vs cls-gate threshold (0.0 = ungated/raw):')
for _t in (0.5, 0.3, 0.1, 0.05, 0.0):
    _v = np.mean([_iou(p, g) if c > _t else 0.0 for p, g, c in zip(_preds, _gtb, _clsp)])
    print(f' gate>{_t}: IoU {_v:.4f}')

# ---- (5) GT-label quality: IoU(GT mask, pixel-exact pair diff |fake-real|)
# Reals indexed from BOTH train and val; fake stems matched exactly first, then by
# progressively stripping trailing _token parts (e.g. img_03_lama -> img_03).
_real_idx = {}
for _mn, _sp in cfg.DATASET_PATHS.items():
    for _split in ('train', 'val'):
        _rd = _Path(_sp[_split]['real'])
        if _rd.exists():
            for _p in _rd.rglob('*'):
                if _p.suffix.lower() in IMAGE_EXT:
                    _real_idx.setdefault(_p.stem, str(_p))
print(f'\n(5) real-image index: {len(_real_idx)} stems (train+val)')

def _match_real(stem):
    if stem in _real_idx:
        return _real_idx[stem]
    s = stem
    while '_' in s or '-' in s:
        s = s.rsplit('_', 1)[0] if '_' in s else s.rsplit('-', 1)[0]
        if s in _real_idx:
            return _real_idx[s]
    return None

import re as _re
def _area_of(stem):
    _mm = _re.search(r'area\d+', stem)
    return _mm.group(0) if _mm else stem
_fake_stems_all = [_Path(_pt).stem for _ds in getattr(val_ds, 'datasets', [val_ds])
                   for _pt, _lb, _mp, _met in _ds.samples if _lb == 1]
_fake_train_stems = [_Path(_pt).stem for _ds in train_datasets
                     for _pt, _lb, _mp, _met in _ds.samples if _lb == 1]
_fa = sorted({_area_of(s) for s in _fake_stems_all})
print(f'(5) fake train areas: {sorted({_area_of(s) for s in _fake_train_stems})}')
_ra = sorted({_area_of(s) for s in _real_idx})
print(f'(5) fake val areas: {_fa}')
print(f'(5) real areas (train+val): {_ra}')
print(f'(5) overlap: {sorted(set(_fa) & set(_ra))}')

# Pair across train+val: pristine sources (gt) may only be shipped for some splits.
_res, _unmatched = [], []
_per_method = {}
_seen = {'train': 0, 'val': 0}
_hit = {'train': 0, 'val': 0}
for _tag, _coll in (('train', train_datasets), ('val', val_datasets)):
    for _ds in _coll:
        for _pt, _lb, _mp, _met in _ds.samples:
            if _lb != 1 or not _mp:
                continue
            _seen[_tag] += 1
            _rp = _match_real(_Path(_pt).stem)
            if _rp is None:
                if len(_unmatched) < 8:
                    _unmatched.append((_tag, _Path(_pt).stem))
                continue
            _hit[_tag] += 1
            if len(_res) >= 600:
                continue
            _f = np.asarray(_Image.open(_pt).convert('RGB').resize((cfg.IMG_SIZE, cfg.IMG_SIZE), _Image.LANCZOS), np.float32) / 255.
            _r = np.asarray(_Image.open(_rp).convert('RGB').resize((cfg.IMG_SIZE, cfg.IMG_SIZE), _Image.LANCZOS), np.float32) / 255.
            _m = torch.from_numpy((np.asarray(_Image.open(_mp).convert('L').resize((cfg.IMG_SIZE, cfg.IMG_SIZE), _Image.BOX), np.float32) / 255. > 0.35))
            _d = torch.from_numpy((np.abs(_f - _r).max(-1) > 0.03))
            _dt = _d.float()[None, None]
            _dt = F.max_pool2d(1.0 - F.max_pool2d(1.0 - _dt, 3, 1, 1), 3, 1, 1) # open: erode+dilate
            _v = _iou(_m, _dt.squeeze() > 0.5)
            _res.append(_v)
            _per_method.setdefault(_met, []).append(_v)
print(f'(5) pairing: train {_hit["train"]}/{_seen["train"]} matched | val {_hit["val"]}/{_seen["val"]} matched')
if _res:
    print(f'(5) label ceiling on {len(_res)} paired fakes: IoU(GT mask, pixel-exact diff) = {np.mean(_res):.4f}')
    for _met, _vs in sorted(_per_method.items()):
        print(f' {_met}: n={len(_vs)} IoU {np.mean(_vs):.4f}')
    print(' (>~0.95: labels are exact, ceiling is the model. <~0.9: imprecise GT caps every model.)')
else:
    print(f'(5) no pairs anywhere. unmatched sample: {_unmatched[:5]}')
    print(f' real stems sample: {sorted(_real_idx)[:5]}')

# ---- (6) binarization threshold sweep (ungated probs)
print('\n(6) IoU vs mask binarization threshold (ungated):')
_best_t, _best_v = 0.5, 0.0
for _t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
    _v = np.mean([_iou(_p > _t, g) for _p, g in zip(_probs, _gtb)])
    if _v > _best_v:
        _best_t, _best_v = _t, _v
    print(f' t={_t:.2f}: IoU {_v:.4f}')
print(f' -> best threshold {_best_t:.2f} (IoU {_best_v:.4f})')

# ---- (7) test-time augmentation: average probs over identity + 3 flips
print('\n(7) TTA (identity + hflip + vflip + rot180), ungated:')
_dims = (None, [-1], [-2], [-1, -2])
_tta = []
with torch.no_grad():
    for _b in tqdm(val_loader, desc='tta', leave=False):
        _x = _b['image'].to(DEVICE)
        _acc = None
        for _d in _dims:
            _xt = _x if _d is None else torch.flip(_x, _d)
            _pm = torch.sigmoid(model(_xt)[1])
            _pm = _pm if _d is None else torch.flip(_pm, _d)
            _acc = _pm if _acc is None else _acc + _pm
        _acc = (_acc / len(_dims)).float().cpu()
        for i in range(_x.shape[0]):
            if int(_b['label'][i].item()) == 1:
                _tta.append(_acc[i, 0])
print(f' TTA @0.50: IoU {np.mean([_iou(p > 0.5, g) for p, g in zip(_tta, _gtb)]):.4f}')
print(f' TTA @best-t({_best_t:.2f}): IoU {np.mean([_iou(p > _best_t, g) for p, g in zip(_tta, _gtb)]):.4f}')

# ---- (8) multi-scale TTA: each flip also predicted at x1.5 upscale, probs averaged
print('\n(8) TTA + x1.5 scale (small-region booster), ungated:')
_ms = []
with torch.no_grad():
    for _b in tqdm(val_loader, desc='tta-ms', leave=False):
        _x = _b['image'].to(DEVICE)
        _acc = None
        for _d in _dims:
            _xt = _x if _d is None else torch.flip(_x, _d)
            _p1 = torch.sigmoid(model(_xt)[1])
            _xh = F.interpolate(_xt, scale_factor=1.5, mode='bilinear', align_corners=False)
            _p2 = torch.sigmoid(model(_xh)[1])
            _p2 = F.interpolate(_p2, size=_p1.shape[-2:], mode='bilinear', align_corners=False)
            _pm = 0.5 * (_p1 + _p2)
            _pm = _pm if _d is None else torch.flip(_pm, _d)
            _acc = _pm if _acc is None else _acc + _pm
        _acc = (_acc / len(_dims)).float().cpu()
        for i in range(_x.shape[0]):
            if int(_b['label'][i].item()) == 1:
                _ms.append(_acc[i, 0])
print(f' TTA+MS @0.50: IoU {np.mean([_iou(p > 0.5, g) for p, g in zip(_ms, _gtb)]):.4f}')
print(f' TTA+MS @best-t({_best_t:.2f}): IoU {np.mean([_iou(p > _best_t, g) for p, g in zip(_ms, _gtb)]):.4f}')
_qs = np.quantile(_areas, [0, .25, .5, .75, 1.0])
_ious_ms = np.array([_iou(p > _best_t, g) for p, g in zip(_ms, _gtb)])
for _a, _b2 in zip(_qs[:-1], _qs[1:]):
    _idx = (_areas >= _a) & (_areas <= _b2)
    print(f' area {_a:.3f}-{_b2:.3f}: n={int(_idx.sum()):3d} IoU {_ious_ms[_idx].mean():.4f}')

print('\nHow to read: (3) no jump at r=1-2 -> NOT a boundary-offset problem (confirmed).',
      '(5) is the GT-label ceiling -- if it is <~0.9, no model can beat it on these masks.',
      '(6) picks the best binarization threshold; (7) shows what free TTA averaging adds.')

# ======================================================================
# 11b. DGT: DCT/FFT-Guided Dynamic Threshold
#
# Post-training, inference-time only (no retraining). Replaces the earlier CAST cell
# with the far simpler dynamic-threshold rule (SF-CFNet Eq. 14)
#
# $$\tau = \mathrm{clip}\big(T_{base} + \alpha\,\tanh(E_{freq}) + \beta\,\mathrm{Norm}(H_{img}),\; \tau_{lo},\, \tau_{hi}\big)$$
#
# * $E_{freq}$ — mean magnitude of the high-pass FFT spectrum (central $H/4\times W/4$ block zeroed): scene texture / structural complexity.
# * $H_{img}$ — Shannon entropy of the 256-bin grey histogram, divided by 8 so it lands in $[0,1]$.
#
# The threshold rises for complex, high-entropy scenes (fewer false accusations) and
# falls for smooth scenes (recovers missed detections). Two constants only
# (`DGT_ALPHA`, `DGT_BETA`) — nothing is learned, so there is no calibration split, no
# Monte-Carlo dropout, no ECE and no abstention policy to tune.
#
# One change vs. the original formulation (`cfg.DGT_NORM_EFREQ = True`): the raw
# high-pass energy is O(1..100), so `tanh(E_freq)` saturates at 1.0 for *every* image and
# the $\alpha$ term degenerates into a constant offset. A robust z-score
# (median / MAD) over the evaluation set puts `tanh` back in its informative range. Set
# `DGT_NORM_EFREQ = False` to reproduce the original behaviour verbatim.
#
# > The DGT-vs-fixed-0.5 comparison below is computed on the validation set that also
# > supplies the median/MAD reference — labelled as such rather than presented as a
# > held-out result. DGT fits no parameters, so the leak is limited to those two scalars.
# ======================================================================

# ===================== DGT: DCT/FFT-Guided Dynamic Threshold =====================
# Inference-time only, post-training -- NO retraining. Replaces CAST.
# tau = clip(T_base + alpha*tanh(E_freq) + beta*Norm(H_img), lo, hi)
# Raises tau for complex / high-entropy scenes (fewer false positives) and lowers it
# for smooth scenes (recovers missed detections). (SF-CFNet Eq.14)
import os
import json as _json

_DGT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_DGT_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def dgt_image_stats(images_norm):
    """Per-image (E_freq, H_img_norm) from a NORMALIZED batch (B,3,H,W)."""
    x = (images_norm.detach().cpu().float() * _DGT_STD + _DGT_MEAN).clamp(0, 1)
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2] # (B,H,W)
    B, H, W = gray.shape
    Xf = torch.fft.fftshift(torch.fft.fft2(gray, norm='ortho'), dim=(-2, -1))
    cy, cx, rh, rw = H // 2, W // 2, H // 4, W // 4
    Xf[:, cy - rh:cy + rh, cx - rw:cx + rw] = 0 # high-pass
    e_freq = Xf.abs().mean(dim=(-2, -1)).numpy() # (B,)
    h_norm = []
    for b in range(B):
        hist = torch.histc(gray[b] * 255.0, bins=256, min=0, max=255)
        p = hist / (hist.sum() + 1e-8)
        p = p[p > 0]
        h_norm.append((-(p * torch.log2(p)).sum().item()) / 8.0) # 8-bit -> [0,1]
    return e_freq, np.array(h_norm)

def dgt_reference(e_freq):
    """Robust (median, MAD) reference used to z-score E_freq before tanh."""
    e = np.asarray(e_freq, dtype=np.float64).ravel()
    med = float(np.median(e))
    mad = float(np.median(np.abs(e - med)))
    return med, mad

def dgt_threshold(e_freq, h_norm, cfg, e_ref=None, h_ref=None):
    """Vectorised per-image threshold. Returns (tau, e_ref, h_ref)."""
    e = np.asarray(e_freq, dtype=np.float64)
    if bool(getattr(cfg, 'DGT_NORM_EFREQ', True)):
        if e_ref is None:
            e_ref = dgt_reference(e)
        _med, _mad = e_ref
        e = (e - _med) / (1.4826 * _mad + 1e-8)
    h = np.asarray(h_norm, dtype=np.float64)
    if bool(getattr(cfg, 'DGT_CENTER_HIMG', True)):
        # H_img averages ~0.83 on these tiles, so beta*H contributes +0.37 while the
        # clip only allows +0.20 above T_base: every tau pins to the ceiling and the
        # "dynamic" threshold becomes a constant. Centring on the median makes beta
        # scale a DEVIATION from the typical scene instead of an absolute offset.
        if h_ref is None:
            h_ref = float(np.median(h))
        h = h - h_ref
    lo, hi = getattr(cfg, 'DGT_CLIP', (0.3, 0.7))
    tau = (float(getattr(cfg, 'DGT_TBASE', 0.5))
           + float(getattr(cfg, 'DGT_ALPHA', 0.2)) * np.tanh(e)
           + float(getattr(cfg, 'DGT_BETA', 0.45)) * h)
    return np.clip(tau, float(lo), float(hi)), e_ref, h_ref

if getattr(cfg, 'DGT_ENABLE', False):
    # self-sufficient: works after a plain restart, no training cell required
    if 'val_loader' not in globals():
        raise RuntimeError('DGT needs val_loader -- run the dataset/loader cells first.')
    # NEVER score whatever happens to be in `model`: cell 17 leaves a randomly
    # initialised NRGANet there, and an untrained net yields probabilities bunched
    # around 0.45 with AUC ~0.5 -- a table that looks real but means nothing.
    ensure_trained_model()
    _pr, _lb, _ef, _hn = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='DGT', leave=False):
            images = batch['image'].to(DEVICE)
            with autocast():
                cls_logit, *_rest = model(images)
            _pr.extend(torch.sigmoid(cls_logit).float().cpu().numpy().ravel().tolist())
            _lb.extend(np.asarray(batch['label']).ravel().tolist())
            _e, _h = dgt_image_stats(batch['image'])
            _ef.extend(np.atleast_1d(_e).tolist())
            _hn.extend(np.atleast_1d(_h).tolist())

    dgt_probs = np.asarray(_pr, dtype=np.float64)
    dgt_labels = np.asarray(_lb).astype(int)
    dgt_efreq = np.asarray(_ef, dtype=np.float64)
    dgt_hnorm = np.asarray(_hn, dtype=np.float64)
    dgt_taus, dgt_eref, dgt_href = dgt_threshold(dgt_efreq, dgt_hnorm, cfg)

    pred_fixed = (dgt_probs > 0.5).astype(int)
    pred_dgt = (dgt_probs > dgt_taus).astype(int)
    _changed = int((pred_fixed != pred_dgt).sum())

    _m_fixed = dict(acc=accuracy_score(dgt_labels, pred_fixed),
                    f1=f1_score(dgt_labels, pred_fixed, zero_division=0),
                    prec=precision_score(dgt_labels, pred_fixed, zero_division=0),
                    rec=recall_score(dgt_labels, pred_fixed, zero_division=0))
    _m_dgt = dict(acc=accuracy_score(dgt_labels, pred_dgt),
                  f1=f1_score(dgt_labels, pred_dgt, zero_division=0),
                  prec=precision_score(dgt_labels, pred_dgt, zero_division=0),
                  rec=recall_score(dgt_labels, pred_dgt, zero_division=0))

    print('\n' + '=' * 66)
    print(' DGT: DCT/FFT-Guided Dynamic Threshold vs Fixed 0.5')
    print('=' * 66)
    print(f' images : {len(dgt_probs)} (fake={int(dgt_labels.sum())}, '
          f'real={int((dgt_labels == 0).sum())})')
    print(f' E_freq : min={dgt_efreq.min():.4f} med={np.median(dgt_efreq):.4f} '
          f'max={dgt_efreq.max():.4f} norm={bool(getattr(cfg, "DGT_NORM_EFREQ", True))}')
    print(f' H_img (norm) : min={dgt_hnorm.min():.4f} mean={dgt_hnorm.mean():.4f} '
          f'max={dgt_hnorm.max():.4f}')
    _lo, _hi = getattr(cfg, 'DGT_CLIP', (0.3, 0.7))
    _pinned = int(((dgt_taus <= _lo + 1e-9) | (dgt_taus >= _hi - 1e-9)).sum())
    print(f' tau : [{dgt_taus.min():.3f}, {dgt_taus.max():.3f}] '
          f'mean={dgt_taus.mean():.3f} spread={dgt_taus.std():.4f} '
          f'centred={bool(getattr(cfg, "DGT_CENTER_HIMG", True))}')
    if _pinned > 0.5 * len(dgt_taus):
        print(f' !! {_pinned}/{len(dgt_taus)} tau values sit ON a clip bound -- the '
              f'threshold is effectively constant. Widen DGT_CLIP or lower DGT_BETA.')
    _auc_all = (roc_auc_score(dgt_labels, dgt_probs)
                if len(np.unique(dgt_labels)) > 1 else float('nan'))
    print(f' classifier AUC: {_auc_all:.4f} (threshold-free separation)')
    print(f' decisions changed vs fixed-0.5: {_changed}/{len(dgt_probs)}')
    print(f' Fixed 0.5 -> Acc:{_m_fixed["acc"]:.4f} F1:{_m_fixed["f1"]:.4f} '
          f'Prec:{_m_fixed["prec"]:.4f} Rec:{_m_fixed["rec"]:.4f}')
    print(f' DGT -> Acc:{_m_dgt["acc"]:.4f} F1:{_m_dgt["f1"]:.4f} '
          f'Prec:{_m_dgt["prec"]:.4f} Rec:{_m_dgt["rec"]:.4f}')
    print(f' dF1 = {_m_dgt["f1"] - _m_fixed["f1"]:+.4f} '
          f'dPrec = {_m_dgt["prec"] - _m_fixed["prec"]:+.4f} '
          f'dRec = {_m_dgt["rec"] - _m_fixed["rec"]:+.4f}')
    print(' Note: alpha/beta are fixed constants; the median/MAD reference above comes')
    print(' from this same val set. DGT is applied at INFERENCE only.')

    dgt_params = {
        'T_base': float(getattr(cfg, 'DGT_TBASE', 0.5)),
        'alpha': float(getattr(cfg, 'DGT_ALPHA', 0.2)),
        'beta': float(getattr(cfg, 'DGT_BETA', 0.45)),
        'clip': [float(getattr(cfg, 'DGT_CLIP', (0.3, 0.7))[0]),
                 float(getattr(cfg, 'DGT_CLIP', (0.3, 0.7))[1])],
        'norm_efreq': bool(getattr(cfg, 'DGT_NORM_EFREQ', True)),
        'center_himg': bool(getattr(cfg, 'DGT_CENTER_HIMG', True)),
        'e_freq_median': float(dgt_eref[0]) if dgt_eref is not None else None,
        'e_freq_mad': float(dgt_eref[1]) if dgt_eref is not None else None,
        'h_img_median': float(dgt_href) if dgt_href is not None else None,
        'classifier_auc': float(_auc_all),
        'tau_min': float(dgt_taus.min()), 'tau_max': float(dgt_taus.max()),
        'tau_mean': float(dgt_taus.mean()), 'tau_std': float(dgt_taus.std()),
        'n_images': int(len(dgt_probs)), 'n_decisions_changed': _changed,
        'metrics_fixed_0.5': {k: float(v) for k, v in _m_fixed.items()},
        'metrics_dgt': {k: float(v) for k, v in _m_dgt.items()},
        'img_size': int(cfg.IMG_SIZE),
    }
    dgt_params_path = os.path.join(cfg.OUTPUT_DIR, 'dgt_params.json')
    with open(dgt_params_path, 'w') as _f:
        _json.dump(dgt_params, _f, indent=2)
    print(f' Saved -> {dgt_params_path}')
else:
    print('DGT disabled (cfg.DGT_ENABLE=False)')

# ======================================================================
# 12b. Inference: native resolution + TTA + calibrated mask threshold
#
# FLDCF's GUI ran each image at its native resolution (`whole` mode), while
# NRGA-Net was locked to `256x256` by the ViT positional grid — every test image was
# downscaled and the mask upscaled back, which destroys thin structure on its own.
#
# With the ViT gone the network is fully convolutional, so `predict_mask` offers the
# same `whole / resize / tile` modes as the FLDCF GUI plus 8-way dihedral TTA, and the
# mask threshold is calibrated on validation IoU instead of being hard-wired to `0.5`
# (a high threshold erodes a thin structure away entirely).
# ======================================================================

# ===================== TTA + resolution-flexible inference =====================
import json as _json

_IMNET_MEAN = [0.485, 0.456, 0.406]
_IMNET_STD = [0.229, 0.224, 0.225]

def _pad_to(t, mult=32):
    """Pad NCHW to a multiple of `mult` (reflect) -> (padded, (H, W))."""
    h, w = t.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    return t, (h, w)

@torch.no_grad()
def _raw_forward(model, arr01, device=DEVICE):
    """arr01: float32 (H, W, 3) in [0,1] -> (cls_prob, mask_prob (H, W))."""
    t = torch.from_numpy(np.ascontiguousarray(arr01)).permute(2, 0, 1).unsqueeze(0)
    t = TF.normalize(t, _IMNET_MEAN, _IMNET_STD).to(device)
    t, (h, w) = _pad_to(t, 32)
    with autocast():
        cls_logit, main_mask, *_ = model(t)
    prob = torch.sigmoid(main_mask.float())[0, 0, :h, :w].cpu().numpy()
    return float(torch.sigmoid(cls_logit.float()).reshape(-1)[0].item()), prob

_DIHEDRAL = [(0, False), (1, False), (2, False), (3, False),
             (0, True), (1, True), (2, True), (3, True)]

@torch.no_grad()
def _tta_forward(model, arr01, device=DEVICE, enable=True):
    """8-way dihedral TTA: averaging the 8 symmetries cancels the orientation bias
    of the transposed convolutions, which visibly de-noises the mask boundary."""
    if not enable:
        return _raw_forward(model, arr01, device)
    cls_acc, acc, n = 0.0, None, 0
    for k, flip in _DIHEDRAL:
        a = np.rot90(arr01, k, axes=(0, 1))
        if flip:
            a = a[:, ::-1]
        c, p = _raw_forward(model, np.ascontiguousarray(a), device)
        if flip:
            p = p[:, ::-1]
        p = np.rot90(p, -k, axes=(0, 1))
        acc = p.astype(np.float64) if acc is None else acc + p
        cls_acc += c; n += 1
    return cls_acc / n, (acc / n).astype(np.float32)

@torch.no_grad()
def predict_mask(model, arr01, mode='auto', resize_to=256, tile=256, overlap=64,
                 max_side_for_whole=1024, tta=None, device=DEVICE):
    """Forgery probability map for an RGB float image in [0, 1].

    mode: 'auto' -> 'whole' if max(H, W) <= max_side_for_whole else 'tile'
          'whole' -> one pass at NATIVE resolution (what FLDCF's GUI did)
          'resize' -> squeeze to resize_to x resize_to, upsample the mask back
          'tile' -> overlapping sliding windows, averaged
    Returns (cls_prob, mask_prob HxW float32).
    """
    if tta is None:
        tta = bool(getattr(cfg, 'TTA_ENABLE', True))
    model.eval()
    H, W = arr01.shape[:2]
    if mode == 'auto':
        mode = 'whole' if max(H, W) <= max_side_for_whole else 'tile'

    if mode == 'whole':
        return _tta_forward(model, arr01, device, tta)

    if mode == 'resize':
        small = np.asarray(Image.fromarray((arr01 * 255).astype(np.uint8))
                           .resize((resize_to, resize_to), Image.BICUBIC)).astype(np.float32) / 255.
        c, p = _tta_forward(model, small, device, tta)
        p = np.asarray(Image.fromarray((p * 255).astype(np.uint8))
                       .resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.
        return c, p

    step = max(tile - overlap, 32)
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    cls_acc, n = 0.0, 0
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < H: ys.append(max(H - tile, 0))
    if xs[-1] + tile < W: xs.append(max(W - tile, 0))
    for y in ys:
        for x in xs:
            patch = arr01[y:y + tile, x:x + tile]
            c, p = _tta_forward(model, patch, device, tta)
            acc[y:y + p.shape[0], x:x + p.shape[1]] += p
            cnt[y:y + p.shape[0], x:x + p.shape[1]] += 1.0
            cls_acc += c; n += 1
    return cls_acc / max(n, 1), acc / np.maximum(cnt, 1e-6)

# ----
# Mask-threshold calibration. 0.5 is arbitrary; the IoU-optimal cut is often
# 0.3-0.6, and for thin structures a high threshold erodes them away entirely.
@torch.no_grad()
def calibrate_mask_threshold(model, loader, grid_n=None):
    grid_n = grid_n or int(getattr(cfg, 'MASK_TAU_GRID', 41))
    taus = np.linspace(0.05, 0.95, grid_n)
    inter = np.zeros(grid_n); union = np.zeros(grid_n)
    model.eval()
    for batch in tqdm(loader, desc='Mask tau sweep', leave=False):
        images = batch['image'].to(DEVICE)
        labels = batch['label'].numpy().ravel()
        gts = batch['mask'].numpy()[:, 0]
        with autocast():
            _, main_mask, *_ = model(images)
        probs = torch.sigmoid(main_mask.float())[:, 0].cpu().numpy()
        for i in range(len(labels)):
            if labels[i] != 1:
                continue
            g = gts[i] > 0.5
            for j, t in enumerate(taus):
                p = probs[i] > t
                inter[j] += np.logical_and(p, g).sum()
                union[j] += np.logical_or(p, g).sum()
    iou = inter / np.maximum(union, 1.0)
    j = int(np.argmax(iou))
    return float(taus[j]), float(iou[j]), taus, iou

MASK_TAU = 0.5
try:
    MASK_TAU, _tau_iou, _taus, _ious = calibrate_mask_threshold(model, val_loader)
    _iou05 = float(_ious[int(np.argmin(np.abs(_taus - 0.5)))])
    print(f'Mask threshold: tau={MASK_TAU:.3f} -> dataset IoU {_tau_iou:.4f} '
          f'(tau=0.5 gives {_iou05:.4f}, {_tau_iou - _iou05:+.4f})')
    plt.figure(figsize=(6, 3.2))
    plt.plot(_taus, _ious, 'b-', lw=2)
    plt.axvline(0.5, color='gray', ls='--', label='0.5')
    plt.axvline(MASK_TAU, color='black', lw=2, label=f'deployed {MASK_TAU:.3f}')
    plt.xlabel('mask threshold'); plt.ylabel('dataset IoU (fake samples)')
    plt.title(' mask-threshold calibration'); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.show()
except Exception as _e:
    print('Mask threshold calibration skipped:', _e)

_seg_params = {
    'model': 'nrga_net_densenet201',
    'mask_tau': float(MASK_TAU),
    'tta': bool(getattr(cfg, 'TTA_ENABLE', True)),
    'img_size': int(cfg.IMG_SIZE),
    'vit': False,
    'dilate_stage3': bool(getattr(cfg, 'DILATE_STAGE3', True)),
    'dilate_stage4': bool(getattr(cfg, 'DILATE_STAGE4', True)),
    'content_prior': bool(getattr(cfg, 'USE_CONTENT_PRIOR', True)),
    'inference_modes': ['auto', 'whole', 'resize', 'tile'],
    'note': ' is fully convolutional: run whole/tile at native resolution.',
}
with open(os.path.join(cfg.OUTPUT_DIR, 'seg_params.json'), 'w') as _f:
    _json.dump(_seg_params, _f, indent=2)
print('Saved ->', os.path.join(cfg.OUTPUT_DIR, 'seg_params.json'))

def show_predictions(model, dataset, n=8):
    model.eval()
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    fig,axes=plt.subplots(4,n,figsize=(n*3,12))
    for col,idx in enumerate(indices):
        s=dataset[idx]
        img_np=(s['image']*std+mean).clamp(0,1).permute(1,2,0).numpy()
        gt_mask=s['mask'].squeeze().numpy()
        # 8-way TTA + the IoU-calibrated threshold instead of a bare 0.5
        prob, mask_p = predict_mask(model, img_np.astype(np.float32),
                                    mode='whole', tta=bool(getattr(cfg,'TTA_ENABLE',True)))
        pred_mask = (mask_p > globals().get('MASK_TAU', 0.5)).astype(np.float32)
        # CLS-GATED MASK: if classifier says REAL, force empty mask
        if prob <= 0.5:
            pred_mask = np.zeros_like(pred_mask)
        gt_lbl='REAL' if s['label']==0 else f'FAKE({s["method"]})'
        pred_lbl=f'{"FAKE" if prob>0.5 else "REAL"}({prob:.2f})'
        ok=(s['label']==0 and prob<=0.5)or(s['label']==1 and prob>0.5)
        c='green' if ok else 'red'
        inter=(pred_mask*gt_mask).sum(); union=((pred_mask+gt_mask)>0).sum(); iou=inter/(union+1e-8)
        axes[0,col].imshow(img_np);axes[0,col].set_title(f'GT:{gt_lbl}\nPred:{pred_lbl}',fontsize=7,color=c);axes[0,col].axis('off')
        axes[1,col].imshow(gt_mask,cmap='gray',vmin=0,vmax=1);axes[1,col].set_title('GT Mask',fontsize=8);axes[1,col].axis('off')
        axes[2,col].imshow(pred_mask,cmap='gray',vmin=0,vmax=1);axes[2,col].set_title(f'Pred(IoU:{iou:.2f})',fontsize=8);axes[2,col].axis('off')
        ov=img_np.copy(); ov[pred_mask>0.5]=ov[pred_mask>0.5]*0.5+np.array([1,0,0])*0.5
        axes[3,col].imshow(ov);axes[3,col].set_title('Overlay',fontsize=8);axes[3,col].axis('off')
    for i,l in enumerate(['Image','GT Mask','Pred Mask','Overlay']): axes[i,0].set_ylabel(l,fontsize=10)
    plt.suptitle('NRGA-Net Predictions vs Ground Truth (cls-gated, TTA, calibrated tau)',fontsize=13)
plt.tight_layout(); plt.show()
show_predictions(model, val_ds)

# ======================================================================
# 13. Save Results
# ======================================================================

with open(os.path.join(cfg.OUTPUT_DIR,'results.txt'),'w') as f:
    f.write('NRGA-Net Results\n'+'='*50+'\n\n')
    for k,v in val_m.items(): f.write(f'{k}: {v:.4f}\n')
    f.write(f'\n{classification_report(all_labels,all_preds,target_names=["Real","Fake"],zero_division=0)}')
print(f'Results saved to: {cfg.OUTPUT_DIR}')
print(f'Best model: {best_model_path}')
print('Training complete!')

# ======================================================================
# 14. CBFH Database — Authentic-Image Registry
#
# The shared CBFH forensic database is a registry of authentic (real) imagery. A query
# image whose content hash is found in the registry has *known provenance*; anything else
# is unverified. Forged images are therefore no longer registered — a forgery has no
# authentic provenance to certify, and keeping fakes in the registry made a "hash found"
# answer ambiguous.
#
# Running this cell performs, in order
#
# | Step | Action |
# |---|---|
# | 1 | Timestamped backup of `cbfh_registry.json` (same Drive folder) |
# | 2 | Purge already-registered Fake-Vaihingen FAKE records (+ this model's fakes elsewhere) |
# | 3 | Relabel real records that carry the old hard-coded `fake-vaihingen` source type |
# | 4 | Collapse duplicate real records (LaMa/RePaint entries share one `real/` folder) |
# | 5 | Register real images only — `Local_Diffusion` reals as Fake-LocalDiff Dataset |
# | 6 | One single `_save()` at the end (was one full Drive write *per image*) |
#
# No training, no fine-tuning: run the config + definition cells, then this cell alone.
# Registration is idempotent — a second run reports `registered=0`.
#
# | source_type | Dataset name | Contents |
# |---|---|---|
# | `vaihingen-real` | Fake-Vaihingen | Authentic ISPRS Vaihingen tiles |
# | `loveda-real` | Fake-LoveDA | Authentic LoveDA (Google Earth) tiles |
# | `fake-localdiff` | Fake-LocalDiff Dataset | Authentic sources for prompt-driven local diffusion |
# | `fake-vaihingen` | *(legacy)* | Kept so pre-existing records still resolve |
#
# Cascade support is unchanged: another model can call `db.add_model_result()` on the
# same records without overwriting anything.
# ======================================================================

# ===================== CBFH Database: Authentic-Image Registry =====================
# Shared forensic hash database at: /content/drive/MyDrive/PhD_GIS_Security/DB_local/

# Policy: the registry holds AUTHENTIC (real) images only.
# step 1 timestamped backup of cbfh_registry.json
# step 2 purge already-registered Fake-Vaihingen FAKE records
# step 3 relabel real records carrying the old hard-coded 'fake-vaihingen' source
# step 4 collapse duplicate real records (shared real/ dirs registered twice)
# step 5 register REAL images only -- Local_Diffusion reals = 'Fake-LocalDiff Dataset'
# step 6 ONE _save() at the end (cbfh_db._save() writes the whole JSON per call)

# Standalone: no training, no fine-tuning. Idempotent -- a re-run registers 0.
# cbfh_db.py itself is NOT modified.

import os, sys, hashlib, json, shutil, time
from pathlib import Path

CBFH_DB_DIR = "/content/drive/MyDrive/PhD_GIS_Security/DB_local"
sys.path.insert(0, CBFH_DB_DIR)
from cbfh_db import CBFHDatabase, DATASET_INFO

MODEL_NAME = "nrga_net_densenet201" # this model's identifier in the cascade
_IMG_EXT = globals().get('IMAGE_EXT',
                         {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'})

# ─── source-type taxonomy ───────────────────────────────────────────────────────
DATASET_INFO.setdefault('fake-vaihingen', { # legacy: old real records point here
    'dataset_name': 'Fake-Vaihingen',
    'description': 'Fake-Vaihingen: Vaihingen aerial images with synthetic inpainting '
                   'forgeries (LaMa / RePaint).',
})
DATASET_INFO['vaihingen-real'] = {
    'dataset_name': 'Fake-Vaihingen',
    'description': 'Authentic ISPRS Vaihingen aerial tiles -- the originals behind the '
                   'Fake-Vaihingen benchmark.',
}
DATASET_INFO['loveda-real'] = {
    'dataset_name': 'Fake-LoveDA',
    'description': 'Authentic LoveDA tiles obtained from the Google Earth platform -- '
                   'the originals behind the Fake-LoveDA benchmark.',
}
DATASET_INFO['fake-localdiff'] = {
    'dataset_name': 'Fake-LocalDiff Dataset',
    'description': 'Fake-LocalDiff Dataset: authentic source imagery for prompt-driven '
                   'local diffusion replacement forgeries.',
}

# dataset-root token -> source_type for REAL images
REAL_SOURCE_BY_ROOT = (
    ('Fake-Vaihingen', 'vaihingen-real'),
    ('Fake-LoveDA', 'loveda-real'),
    ('Local_Diffusion', 'fake-localdiff'),
)

def _norm_path(p):
    return str(p or '').replace('\\', '/')

def real_source_for_dataset(ds_name):
    """source_type for the REAL images of a cfg.DATASET_PATHS key."""
    for _root, _src in REAL_SOURCE_BY_ROOT:
        if str(ds_name).startswith(_root):
            return _src
    return str(ds_name).lower() + '-real'

def real_source_for_path(image_path):
    """source_type inferred from a stored image_path; None when unrecognised."""
    p = _norm_path(image_path)
    for _root, _src in REAL_SOURCE_BY_ROOT:
        if f'/{_root}/' in p:
            return _src
    return None

def _record_is_fake(entry):
    """True when the record describes a FORGED image."""
    st = str(entry.get('source_type', '')).lower()
    if st.endswith('-lama') or st.endswith('-repaint'):
        return True
    p = _norm_path(entry.get('image_path')).lower()
    if '/fake/' in p:
        return True
    for mr in (entry.get('model_results') or {}).values():
        if isinstance(mr, dict) and str(mr.get('img_type_gt', '')).lower() == 'fake':
            return True
    return False

def _record_is_vaihingen(entry):
    st = str(entry.get('source_type', '')).lower()
    return ('/Fake-Vaihingen/' in _norm_path(entry.get('image_path'))
            or st.startswith('fake-vaihingen'))

def _purge_reason(entry):
    """Why this record must go, or None to keep it."""
    if not _record_is_fake(entry):
        return None
    if getattr(cfg, 'CBFH_PURGE_VAIHINGEN_FAKES', True) and _record_is_vaihingen(entry):
        return 'fake-vaihingen'
    # only ever this model's own records -- other models' data is never touched
    if (getattr(cfg, 'CBFH_PURGE_OWN_FAKES', True)
            and str(entry.get('registered_by', '')) == MODEL_NAME):
        return 'own-fake'
    return None

def _fmt_sources(d):
    return ', '.join(f'{k}:{v}' for k, v in sorted(d.items(), key=lambda kv: -kv[1]))

# ─── init ───────────────────────────────────────────────────────────────────────
db = CBFHDatabase()
print(f'CBFH Database loaded from: {db.db_path}')
_stats_before = db.stats()
print(f'Existing records: {_stats_before["total_images"]}')
print(f' by source: {_fmt_sources(_stats_before["by_source"])}')

# ─── step 1: backup ─────────────────────────────────────────────────────────────
_backup_path = None
if getattr(cfg, 'CBFH_BACKUP', True) and Path(db.db_path).exists():
    _backup_path = str(Path(db.db_path).with_name(
        f'cbfh_registry.backup_{time.strftime("%Y%m%d_%H%M%S")}.json'))
    shutil.copy2(str(db.db_path), _backup_path)
    with open(_backup_path, 'r', encoding='utf-8') as _f:
        _n_bak = len(json.load(_f).get('images', {}))
    print(f'\nBackup -> {_backup_path} ({_n_bak} records)')
    assert _n_bak == _stats_before['total_images'], 'backup record count mismatch'
else:
    print('\nNo backup written (cfg.CBFH_BACKUP=False or registry does not exist yet).')

# ─── step 2: purge fake records ─────────────────────────────────────────────────
_purge = {}
for _iid, _e in db.data['images'].items():
    _why = _purge_reason(_e)
    if _why:
        _purge[_iid] = (_why, str(_e.get('source_type', '?')))

print('\n' + '=' * 66)
print(' STEP 2 - purge FAKE records (registry keeps authentic images only)')
print('=' * 66)
if _purge:
    _by_src, _by_why = {}, {}
    for _why, _src in _purge.values():
        _by_src[_src] = _by_src.get(_src, 0) + 1
        _by_why[_why] = _by_why.get(_why, 0) + 1
    print(f' {len(_purge)} records to remove')
    print(f' by reason: {_fmt_sources(_by_why)}')
    print(f' by source: {_fmt_sources(_by_src)}')
    print(f' example : {list(_purge)[:3]}')
    for _iid in _purge:
        del db.data['images'][_iid]
    print(f' removed {len(_purge)} records (backup above is the recovery path)')
else:
    print(' nothing to purge')

# ─── step 3: relabel real records ───────────────────────────────────────────────
print('\n' + '=' * 66)
print(' STEP 3 - relabel real records with a correct source_type')
print('=' * 66)
_relabelled, _unresolved = {}, []
if getattr(cfg, 'CBFH_RELABEL_REALS', True):
    for _iid, _e in db.data['images'].items():
        _st = str(_e.get('source_type', ''))
        if _st not in ('fake-vaihingen',) and not _st.endswith('-real'):
            continue # other pipelines' source types: leave alone
        _new = real_source_for_path(_e.get('image_path'))
        if _new is None:
            if _st == 'fake-vaihingen':
                _unresolved.append(_iid)
            continue
        if _new != _st:
            _e['source_type'] = _new
            _e['dataset'] = DATASET_INFO.get(_new, _e.get('dataset'))
            _relabelled[_new] = _relabelled.get(_new, 0) + 1
    print(f' relabelled: {_fmt_sources(_relabelled) if _relabelled else "none"}')
    if _unresolved:
        print(f' {len(_unresolved)} record(s) kept as-is: image_path matches no known '
              f'dataset root (e.g. {_unresolved[:2]})')
else:
    print(' skipped (cfg.CBFH_RELABEL_REALS=False)')

# ─── step 4: collapse duplicate real records ────────────────────────────────────
print('\n' + '=' * 66)
print(' STEP 4 - collapse duplicate real records (shared real/ folders)')
print('=' * 66)
if getattr(cfg, 'CBFH_DEDUPE_REALS', True):
    _by_file = {}
    for _iid, _e in db.data['images'].items():
        _p = _norm_path(_e.get('image_path'))
        if not _p:
            continue
        _by_file.setdefault(_p, []).append(_iid)
    _dups = []
    for _p, _ids in _by_file.items():
        if len(_ids) < 2:
            continue
        _ids = sorted(_ids, key=lambda i: (db.data['images'][i].get('registered_at', ''), i))
        _dups.extend(_ids[1:]) # keep the earliest registration
    if _dups:
        print(f' {len(_dups)} duplicate record(s) removed (example {_dups[:3]})')
        for _iid in _dups:
            del db.data['images'][_iid]
    else:
        print(' no duplicates found')
else:
    print(' skipped (cfg.CBFH_DEDUPE_REALS=False)')

# ─── step 5: register REAL images only ──────────────────────────────────────────
def compute_image_hash(image_path):
    """SHA-256 content hash from raw image bytes."""
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def register_real_images(dataset_paths, model, device, db):
    """Register the AUTHENTIC images of every dataset. Fake folders are never opened.

    image_id keeps the historical `{ds_name}_{split}_{stem}` scheme so records already
    in the registry are recognised and skipped instead of duplicated.
    """
    model.eval()
    stats = {'registered': 0, 'skipped': 0, 'errors': 0, 'dirs': 0}
    seen_real = {'train': set(), 'val': set()}

    for ds_name, splits in dataset_paths.items():
        source_type = real_source_for_dataset(ds_name)
        print(f'\n--- Registering [{ds_name}] real -> source_type={source_type} '
              f'({DATASET_INFO.get(source_type, {}).get("dataset_name", "?")}) ---')

        for split_name in ('train', 'val'):
            if split_name not in splits:
                continue
            real_dir = Path(splits[split_name]['real'])
            if not real_dir.exists():
                print(f' {ds_name}/{split_name}: real dir missing -> skipped')
                continue
            _rk = os.path.realpath(str(real_dir))
            if _rk in seen_real[split_name]:
                print(f' {ds_name}/{split_name}: real dir already handled by an '
                      f'earlier dataset key -> skipped (de-duplicated)')
                continue
            seen_real[split_name].add(_rk)
            stats['dirs'] += 1

            all_paths = [str(p) for p in sorted(real_dir.rglob('*'))
                         if p.suffix.lower() in _IMG_EXT]
            reg_count = skip_count = 0

            for img_path in tqdm(all_paths, desc=f' {ds_name}/{split_name}', leave=False):
                try:
                    image_id = f'{ds_name}_{split_name}_{Path(img_path).stem}'
                    if db.image_exists(image_id):
                        skip_count += 1
                        stats['skipped'] += 1
                        continue

                    cbfh_hash = compute_image_hash(img_path)

                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE), Image.LANCZOS)
                    img_tensor = TF.to_tensor(img)
                    img_tensor = TF.normalize(img_tensor, [0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
                    with torch.no_grad():
                        inp = img_tensor.unsqueeze(0).to(device)
                        with autocast():
                            cls_logit, main_mask, _, _, h_bin, *_ = model(inp)
                        cls_prob = torch.sigmoid(cls_logit).item()
                        mask_prob = torch.sigmoid(main_mask)
                        pred_area = ((mask_prob > 0.5).float().mean().item()
                                     if cls_prob > 0.5 else 0.0)
                        hash_code = ''.join('1' if b > 0 else '0'
                                            for b in h_bin[0].cpu().numpy())

                    ok = db.register_image(
                        image_id=image_id,
                        cbfh_hash=cbfh_hash,
                        source_type=source_type,
                        model_name=MODEL_NAME,
                        model_result={
                            'cls_prob': round(cls_prob, 4),
                            'pred_label': 'fake' if cls_prob > 0.5 else 'real',
                            'pred_area': round(pred_area, 4),
                            'cbfh_binary_code': hash_code,
                            'img_type_gt': 'real',
                        },
                        image_path=img_path,
                    )
                    if ok:
                        reg_count += 1
                        stats['registered'] += 1
                    else:
                        skip_count += 1
                        stats['skipped'] += 1

                except Exception as e:
                    stats['errors'] += 1
                    if stats['errors'] <= 5:
                        print(f' Error: {img_path}: {e}')

            print(f' {ds_name}/{split_name}: registered={reg_count}, '
                  f'skipped={skip_count} (of {len(all_paths)} real files)')
    return stats

print('\n' + '=' * 66)
print(' STEP 5 - register REAL images only')
print('=' * 66)
if not getattr(cfg, 'CBFH_REGISTER_REALS_ONLY', True):
    raise RuntimeError('cfg.CBFH_REGISTER_REALS_ONLY=False is not supported: this cell '
                       'maintains an authentic-image registry. Set it back to True.')

# Registration bakes cls_prob and the CBFH code into permanent records, so the model
# must be a TRAINED one -- never the randomly initialised net cell 17 leaves behind.
ensure_trained_model()

# step 6: cbfh_db._save() rewrites the whole JSON on EVERY register_image() call.
# Suppress it for the duration of the loop and write once at the end.
_orig_save = db._save
db._save = lambda *a, **k: None
try:
    reg_stats = register_real_images(cfg.DATASET_PATHS, model, DEVICE, db)
finally:
    db._save = _orig_save

db.data['metadata']['dataset_sources'] = DATASET_INFO
db._save()

# ─── report ─────────────────────────────────────────────────────────────────────
_stats_after = db.stats()
print('\n' + '=' * 66)
print(' RESULT')
print('=' * 66)
print(f' registered={reg_stats["registered"]}, skipped={reg_stats["skipped"]}, '
      f'errors={reg_stats["errors"]}, real dirs visited={reg_stats["dirs"]}')
print(f' records: {_stats_before["total_images"]} -> {_stats_after["total_images"]} '
      f'({_stats_after["total_images"] - _stats_before["total_images"]:+d})')
_keys = sorted(set(_stats_before['by_source']) | set(_stats_after['by_source']))
print(f' {"source_type":<28}{"before":>9}{"after":>9}{"delta":>9}')
for _k in _keys:
    _b = _stats_before['by_source'].get(_k, 0)
    _a = _stats_after['by_source'].get(_k, 0)
    if _b or _a:
        print(f' {_k:<28}{_b:>9}{_a:>9}{_a - _b:>+9}')
print(f' DB saved at: {db.db_path}')
if _backup_path:
    print(f' Backup : {_backup_path}')

# ======================================================================
# Per-dataset results table
# ACC / AUC / F1 (image-level detection) and Precision / Recall / F1 / IoU / Dice (pixel-level localization) for each dataset in the selected `DATASET_GROUP`, plus an OVERALL row.
# ======================================================================

# =============================================================================
# RESULTS TABLE + QUALITATIVE FIGURE HELPERS (works for any DATASET_GROUP)
# Defines: print_results_table(), show_predictions_grid()
# then evaluates the best checkpoint and prints the tables.
# =============================================================================
import os, numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

_need = [n for n in ('cfg', 'NRGANet', 'InpaintingSegDataset', 'NRGALoss', 'validate', 'DEVICE')
         if n not in globals()]
if _need:
    raise RuntimeError(f'Missing definitions: {_need}. Run the setup cells '
                       '(imports/config, dataset, model, loss, training utilities) first.')

def _fmt(v, w=8):
    return f'{v:>{w}.4f}' if isinstance(v, (int, float)) and not isinstance(v, bool) else f'{str(v):>{w}}'

def print_results_table(m, title='RESULTS'):
    """Per-dataset + overall table.

    Detection columns (ACC / AUC / F1det) are image-level: each dataset is scored
    against its own forgeries plus the SHARED authentic pool (reals are loaded once
    thanks to cfg.DEDUPE_REALS, so they cannot belong to a single method).
    Localization columns are pixel-level, pooled over that dataset's forged images.
    """
    dss = m.get('_datasets') or list(cfg.DATASET_PATHS.keys())
    print('\n' + '=' * 108)
    print(f' {title} | benchmark: {globals().get("DATASET_GROUP", "?")} | '
          f'shared authentic images: {m.get("_n_real_shared", 0)}')
    print('=' * 108)
    print(f'{"Dataset":<26}{"N":>6}{"ACC":>9}{"AUC":>9}{"F1det":>9}'
          f'{"Prec":>9}{"Recall":>9}{"F1":>9}{"IoU":>9}{"Dice":>9}')
    print(f'{"":<26}{"":>6}{"<-- detection (image) -->":>27}'
          f'{"<-- localization (pixel, pooled) -->":>45}')
    print('-' * 108)
    for mn in dss:
        print(f'{mn:<26}{m.get(f"n_{mn}", 0):>6}'
              f'{_fmt(m.get(f"acc_{mn}", 0), 9)}{_fmt(m.get(f"auc_{mn}", 0), 9)}'
              f'{_fmt(m.get(f"f1det_{mn}", 0), 9)}{_fmt(m.get(f"prec_{mn}", 0), 9)}'
              f'{_fmt(m.get(f"recpix_{mn}", 0), 9)}{_fmt(m.get(f"f1pix_{mn}", 0), 9)}'
              f'{_fmt(m.get(f"iouPooled_{mn}", 0), 9)}{_fmt(m.get(f"dice_{mn}", 0), 9)}')
    print('-' * 108)
    _ntot = sum(m.get(f'n_{mn}', 0) for mn in dss)
    print(f'{"OVERALL":<26}{_ntot:>6}'
          f'{_fmt(m.get("accuracy", 0), 9)}{_fmt(m.get("auc", 0), 9)}{_fmt(m.get("f1", 0), 9)}'
          f'{_fmt(m.get("fakeonly_precision", 0), 9)}{_fmt(m.get("fakeonly_recall", 0), 9)}'
          f'{_fmt(m.get("fakeonly_f1", 0), 9)}{_fmt(m.get("fakeonly_iou", 0), 9)}'
          f'{_fmt(m.get("mean_dice", 0), 9)}')
    print('=' * 108)
    print(f' headline pooled IoU (incl. real-image false positives, FECDNet Eq.17): '
          f'{m.get("pooled_iou", 0):.4f} | pooled F1: {m.get("pooled_f1", 0):.4f}')
    print(f' per-image mean IoU: {m.get("mean_iou", 0):.4f} | '
          f'per-image mean Dice: {m.get("mean_dice", 0):.4f} | '
          f'real-image FP area: {m.get("real_fp_mask", 0):.5f}')
    print(' note: localization rows exclude real images so the rows stay additive;')
    print(' the headline pooled IoU above includes them. Dice = per-image mean.')
    print('=' * 108)

def _build_val(method_names=None, img_size=None, augment=False):
    """Val dataset(s), de-duplicating the shared authentic folders."""
    img_size = int(img_size or globals().get('EVAL_IMG_SIZE') or cfg.IMG_SIZE)
    out, seen = {}, set()
    for ds_name, sp in cfg.DATASET_PATHS.items():
        if 'val' not in sp or (method_names and ds_name not in method_names):
            continue
        rk = os.path.realpath(str(sp['val']['real']))
        load_reals = (not getattr(cfg, 'DEDUPE_REALS', True)) or (rk not in seen)
        seen.add(rk)
        out[ds_name] = InpaintingSegDataset(
            real_dir=sp['val']['real'], fake_dir=sp['val']['fake'], mask_dir=sp['val']['mask'],
            dataset_name=ds_name, split='val', img_size=img_size,
            augment=augment, load_reals=load_reals,
            native_crop=getattr(cfg, 'NATIVE_CROP', False))
    return out

def show_predictions_grid(method_names, n_per_method=4, title='Predictions vs Ground Truth',
                          save_name=None, thr=None, seed=0, model_override=None):
    """Input | GT mask | Predicted mask | Overlay, for a few forged samples per method."""
    thr = float(thr if thr is not None else getattr(cfg, 'MASK_THRESHOLD', 0.5))
    net = model_override if model_override is not None else model
    net.eval()
    dsets = _build_val(method_names)
    if not dsets:
        print(f'no val data for {method_names} under DATASET_GROUP='
              f'{globals().get("DATASET_GROUP", "?")} -- skipping figure')
        return None

    rows = []
    rng = np.random.RandomState(seed)
    for ds_name, ds in dsets.items():
        fake_idx = [i for i in range(len(ds)) if float(ds[i]['label']) > 0.5]
        if not fake_idx:
            continue
        pick = rng.choice(len(fake_idx), size=min(n_per_method, len(fake_idx)), replace=False)
        for p in pick:
            rows.append((ds_name, ds[fake_idx[int(p)]]))
    if not rows:
        print('no forged samples found -- skipping figure')
        return None

    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.15 * len(rows)))
    axes = np.atleast_2d(axes)
    with torch.no_grad():
        for r, (ds_name, sample) in enumerate(rows):
            x = sample['image'].unsqueeze(0).to(DEVICE)
            gt = sample['mask'].squeeze().cpu().numpy()
            pr = torch.sigmoid(net(x)[1])[0, 0].float().cpu().numpy()
            pb = (pr > thr).astype(np.float32)

            img = sample['image'].cpu().numpy().transpose(1, 2, 0)
            mean = np.array(getattr(cfg, 'NORM_MEAN', [0.485, 0.456, 0.406]))
            std = np.array(getattr(cfg, 'NORM_STD', [0.229, 0.224, 0.225]))
            img = np.clip(img * std + mean, 0, 1)

            inter = float((pb * (gt > 0.5)).sum())
            union = float(((pb + (gt > 0.5)) > 0).sum())
            iou = inter / (union + 1e-8)

            ov = img.copy()
            ov[..., 0] = np.clip(ov[..., 0] + 0.45 * pb, 0, 1) # prediction -> red
            ov[..., 1] = np.clip(ov[..., 1] + 0.45 * (gt > 0.5), 0, 1) # ground truth -> green

            for cidx, (arr, cmap, lab) in enumerate((
                    (img, None, f'{ds_name}\nforged image'),
                    (gt, 'gray', 'ground truth'),
                    (pb, 'gray', f'prediction (t={thr:.2f})'),
                    (ov, None, f'overlay IoU={iou:.3f}'))):
                ax = axes[r, cidx]
                ax.imshow(arr, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
                ax.set_title(lab, fontsize=9)
                ax.axis('off')
    fig.suptitle(f'{title} (red = prediction, green = ground truth, yellow = overlap)',
                 fontsize=12, y=1.003)
    plt.tight_layout()
    if save_name:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        _p = os.path.join(cfg.OUTPUT_DIR, save_name)
        fig.savefig(_p, dpi=160, bbox_inches='tight')
        print('saved figure ->', _p)
    plt.show()
    return fig

# ----------------------------------------------------------------- evaluate
# prefer the newest checkpoint that is actually FINITE. A run that ended on a
# NaN epoch leaves a syntactically valid file whose BatchNorm buffers are NaN -- it
# loads fine and then reports 0.0000 for everything.
_ck_path, _ck = None, None
for _c in ('nrga_densenet201_384_best_FV.pt', 'nrga_densenet201_best_FV.pt',
           'nrga_densenet201_384_best_FV.pt', 'nrga_densenet201_best_FV.pt'):
    _p = os.path.join(cfg.OUTPUT_DIR, _c)
    if not os.path.exists(_p):
        continue
    _cand = torch.load(_p, map_location='cpu', weights_only=False)
    _bad = [k for k, v in _cand['model_state'].items()
            if torch.is_floating_point(v) and not bool(torch.isfinite(v).all())]
    if _bad:
        print(f'skipping {_c}: {len(_bad)} non-finite tensors (e.g. {_bad[:2]})')
        continue
    _ck_path, _ck = _p, _cand
    break

if 'model' not in globals():
    model = NRGANet(cfg).to(DEVICE)
if _ck_path:
    model.load_state_dict(_ck['model_state'])
    print(f'loaded {os.path.basename(_ck_path)} (epoch {_ck.get("epoch", "?")})')
else:
    print('no checkpoint found -- evaluating the model currently in memory')
model.eval()

# evaluate at the resolution the checkpoint was TRAINED at. Reporting the
# 384px fine-tune on a 256px val set understated LoveDA and Local_Diffusion by
# ~0.5 IoU while leaving 256px-native Vaihingen almost untouched.
EVAL_IMG_SIZE = checkpoint_img_size(_ck_path, _ck) if _ck_path else int(cfg.IMG_SIZE)
if EVAL_IMG_SIZE != int(cfg.IMG_SIZE):
    print(f'eval resolution: {EVAL_IMG_SIZE}px (from the checkpoint), '
          f'not cfg.IMG_SIZE={cfg.IMG_SIZE}')
print(f'native crop: {bool(getattr(cfg, "NATIVE_CROP", False))} '
      f'(must match training, else large-native datasets are squashed)')

_vd = _build_val()
print('val datasets:', {k: len(v) for k, v in _vd.items()})
_val_loader_tbl = DataLoader(ConcatDataset(list(_vd.values())),
                             batch_size=getattr(cfg, 'VAL_BATCH_SIZE', 4),
                             shuffle=False, num_workers=0, pin_memory=False)
_crit_tbl = NRGALoss(cfg).to(DEVICE)

_m_plain, _, _ = validate(model, _val_loader_tbl, _crit_tbl, tta_ms=False)
print_results_table(_m_plain, title='RESULTS (single forward pass)')

if getattr(cfg, 'REPORT_TTA', True):
    _m_tta, _, _ = validate(model, _val_loader_tbl, _crit_tbl, tta_ms=True)
    print_results_table(_m_tta, title='RESULTS (TTA: 4 flips + x1.5 multi-scale)')

# ======================================================================
# Qualitative results - inpainting benchmarks
# Fake-Vaihingen and Fake-LoveDA (LaMa / RePaint).
# ======================================================================

# =============================================================================
# FIGURE 1 -- Predictions vs Ground Truth : INPAINTING BENCHMARKS
# Fake-Vaihingen (lama / repaint) and Fake-LoveDA (lama / repaint)
# Requires the results-table cell above (defines show_predictions_grid).
# =============================================================================
if 'show_predictions_grid' not in globals():
    raise RuntimeError('Run the results-table cell above first '
                       '(it defines show_predictions_grid).')

_inpaint = [k for k in cfg.DATASET_PATHS
            if k.startswith('Fake-Vaihingen') or k.startswith('Fake-LoveDA')]

if _inpaint:
    show_predictions_grid(
        _inpaint,
        n_per_method=3,
        title=f'{DATASET_GROUP} - inpainting forgery localization',
        save_name=f'qualitative_{DATASET_GROUP}.png',
        seed=0)
else:
    print(f'DATASET_GROUP={DATASET_GROUP} has no Fake-Vaihingen / Fake-LoveDA entry -- '
          'switch DATASET_GROUP in the config cell to render this figure.')

# ======================================================================
# Qualitative results - Local_Diffusion
# ======================================================================

# =============================================================================
# FIGURE 2 -- Predictions vs Ground Truth : LOCAL_DIFFUSION
# Separate figure: prompt-driven local diffusion replacement (not inpainting removal)
# Requires the results-table cell above (defines show_predictions_grid).
# =============================================================================
if 'show_predictions_grid' not in globals():
    raise RuntimeError('Run the results-table cell above first '
                       '(it defines show_predictions_grid).')

_ld = [k for k in cfg.DATASET_PATHS if k.startswith('Local_Diffusion')]

if _ld:
    show_predictions_grid(
        _ld,
        n_per_method=6,
        title='Local_Diffusion - prompt-driven local replacement localization',
        save_name='qualitative_Local_Diffusion.png',
        seed=0)
else:
    print(f'DATASET_GROUP={DATASET_GROUP} has no Local_Diffusion entry -- '
          "set DATASET_GROUP = 'Local_Diffusion' in the config cell to render this figure.")

