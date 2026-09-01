# Datasets

`src/main.py` expects the three benchmarks under one root folder, pointed to by
the `NRGA_DATA_ROOT` environment variable (or by editing `BASE_DB` near the top
of `src/main.py`):

```
DATA_ROOT/
├── Fake-Vaihingen/
│   ├── real/{train,val}/
│   └── fake/{train,val}/{lama,repaint}/  +  fake/{train,val}/inpainted_mask/
├── Fake-LoveDA/
│   ├── real/{train,val}/
│   └── fake/{train,val}/{lama,repaint}/  +  fake/{train,val}/inpainted_mask/
└── Local_Diffusion/
    ├── real/{train,val}/
    ├── fake/{train,val}/
    └── mask/{train,val}/
```

## Sources

| Benchmark | Where to get it |
| --- | --- |
| Fake-Vaihingen | Released with FLDCF (Sui et al., TGRS 2024) — LaMa + RePaint forgeries of ISPRS Vaihingen |
| Fake-LoveDA | Released with FLDCF (Sui et al., TGRS 2024) — LaMa + RePaint forgeries of the LoveDA dataset |
| Fake-LocalDiff | Introduced in our paper — prompt-driven latent-diffusion (SD-2 inpainting) local replacements of PRDLC-PRO and RRSIS scenes; 4,800 forged + 4,800 untouched originals at 256×256 PNG |

Entries whose folders are missing are skipped with a warning, so the pipeline
can run on any subset (set `DATASET_GROUP` in `src/main.py` to
`'Fake-Vaihingen'`, `'Fake-LoveDA'`, `'Local_Diffusion'`, or `'ALL'`).

Authentic folders shared between generator variants are de-duplicated at load
time (`Config.DEDUPE_REALS = True`), matching the paper splits.
