# -*- coding: utf-8 -*-
"""CBFH Shared Database Manager — Cascade Model Support.

Shared forensic hash database for multi-model cascade pipeline.
Location: /content/drive/MyDrive/PhD_GIS_Security/DB_local/

Usage (Model 1 — NRGA-Net Localization):
    from cbfh_db import CBFHDatabase
    db = CBFHDatabase()
    db.register_image(
        image_id="img_001",
        cbfh_hash="a3f2b1...",
        source_type="diffusion",
        model_name="nrga_net_loc",
        model_result={"pred_mask_iou": 0.58, "cls_prob": 0.99, "source_pred": "diffusion"}
    )

Usage (Model 2 — Full Fake Detection):
    from cbfh_db import CBFHDatabase
    db = CBFHDatabase()
    db.add_model_result(
        image_id="img_001",
        model_name="full_fake_det",
        model_result={"is_fake": True, "confidence": 0.97, "method": "diffusion"}
    )
"""

import json
import os
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any

# ─── Configuration ───────────────────────────────────────────────────────────
DB_DIR = "/content/drive/MyDrive/PhD_GIS_Security/DB_local"
DB_FILENAME = "cbfh_registry.json"

# Dataset descriptions per source type
DATASET_INFO = {
    "diffusion": {
        "dataset_name": "PRDLC-PRO",
        "description": "PRDLC-PRO: Annotated Semantic Segmentation Dataset",
    },
    "gan": {
        "dataset_name": "PRDLC-PRO",
        "description": "PRDLC-PRO: Annotated Semantic Segmentation Dataset",
    },
    "repaint": {
        "dataset_name": "LoveDA",
        "description": "LoveDA dataset, images were obtained from the Google Earth platform.",
    },
    "lama": {
        "dataset_name": "LoveDA",
        "description": "LoveDA dataset, images were obtained from the Google Earth platform.",
    },
    "fake-vaihingen": {
        "dataset_name": "Fake-Vaihingen",
        "description": "Fake-Vaihingen: Vaihingen aerial images with synthetic inpainting forgeries (LaMa / RePaint).",
    },
}


class CBFHDatabase:
    """Content-Based Forensic Hashing shared database.

    Supports cascade model pipeline:
      1. First model registers images with hash + source info
      2. Subsequent models add their results without overwriting existing data
    """

    def __init__(self, db_dir: str = DB_DIR, db_filename: str = DB_FILENAME):
        self.db_dir = Path(db_dir)
        self.db_path = self.db_dir / db_filename
        self._ensure_dir()
        self.data = self._load()

    def _ensure_dir(self):
        """Create DB directory if it doesn't exist."""
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        """Load database from disk."""
        if self.db_path.exists():
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"metadata": self._init_metadata(), "images": {}}

    def _save(self):
        """Persist database to disk."""
        self.data["metadata"]["last_modified"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data["metadata"]["total_images"] = len(self.data["images"])
        # Write to temp then rename for atomicity
        tmp_path = self.db_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.db_path)

    def _init_metadata(self) -> dict:
        return {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_modified": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": 0,
            "models_registered": [],
            "dataset_sources": DATASET_INFO,
        }

    # ─── Core API ────────────────────────────────────────────────────────────

    def image_exists(self, image_id: str) -> bool:
        """Check if an image is already registered."""
        return image_id in self.data["images"]

    def register_image(
        self,
        image_id: str,
        cbfh_hash: str,
        source_type: str,
        model_name: str,
        model_result: Dict[str, Any],
        image_path: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register a new image. Returns False if already exists (no overwrite).

        Args:
            image_id: Unique identifier (filename or generated ID)
            cbfh_hash: Content-based forensic hash of the image
            source_type: One of 'diffusion', 'gan', 'repaint', 'lama', 'real'
            model_name: Name of the model registering (e.g. 'nrga_net_loc')
            model_result: Dict of model outputs (predictions, scores, etc.)
            image_path: Optional original file path
            extra_meta: Optional additional metadata

        Returns:
            True if registered, False if already exists (skipped)
        """
        if self.image_exists(image_id):
            return False  # Do NOT overwrite

        # Get dataset description
        dataset_info = DATASET_INFO.get(source_type, {"description": "Unknown source"})

        entry = {
            "image_id": image_id,
            "cbfh_hash": cbfh_hash,
            "source_type": source_type,
            "dataset": dataset_info,
            "image_path": image_path,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "registered_by": model_name,
            "model_results": {model_name: model_result},
        }
        if extra_meta:
            entry["extra"] = extra_meta

        self.data["images"][image_id] = entry

        # Track model in metadata
        if model_name not in self.data["metadata"]["models_registered"]:
            self.data["metadata"]["models_registered"].append(model_name)

        self._save()
        return True

    def add_model_result(
        self,
        image_id: str,
        model_name: str,
        model_result: Dict[str, Any],
    ) -> bool:
        """Add results from a subsequent model in the cascade. No overwrite.

        Args:
            image_id: Must already exist in the database
            model_name: Name of the model adding results
            model_result: Dict of model outputs

        Returns:
            True if added, False if image not found or model already wrote results
        """
        if not self.image_exists(image_id):
            return False

        entry = self.data["images"][image_id]

        # Do not overwrite existing model results
        if model_name in entry["model_results"]:
            return False

        entry["model_results"][model_name] = model_result
        entry[f"last_updated_by"] = model_name
        entry[f"last_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Track model in metadata
        if model_name not in self.data["metadata"]["models_registered"]:
            self.data["metadata"]["models_registered"].append(model_name)

        self._save()
        return True

    def get_image(self, image_id: str) -> Optional[dict]:
        """Get full record for an image."""
        return self.data["images"].get(image_id, None)

    def get_all_by_source(self, source_type: str) -> Dict[str, dict]:
        """Get all images of a specific source type."""
        return {
            k: v for k, v in self.data["images"].items()
            if v.get("source_type") == source_type
        }

    def get_unprocessed_by(self, model_name: str) -> Dict[str, dict]:
        """Get images that haven't been processed by a specific model yet."""
        return {
            k: v for k, v in self.data["images"].items()
            if model_name not in v.get("model_results", {})
        }

    def stats(self) -> dict:
        """Get database statistics."""
        images = self.data["images"]
        sources = {}
        for v in images.values():
            src = v.get("source_type", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {
            "total_images": len(images),
            "by_source": sources,
            "models_registered": self.data["metadata"]["models_registered"],
            "db_path": str(self.db_path),
        }

    # ─── Utility ─────────────────────────────────────────────────────────────

    @staticmethod
    def compute_cbfh_hash(image_bytes: bytes) -> str:
        """Compute content-based forensic hash from raw image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()

    def bulk_register(
        self,
        entries: list,
        model_name: str,
    ) -> dict:
        """Register multiple images at once. Skips existing.

        Args:
            entries: List of dicts with keys:
                image_id, cbfh_hash, source_type, model_result
                Optional: image_path, extra_meta
            model_name: Registering model name

        Returns:
            {"registered": N, "skipped": N}
        """
        registered = 0
        skipped = 0
        for e in entries:
            ok = self.register_image(
                image_id=e["image_id"],
                cbfh_hash=e["cbfh_hash"],
                source_type=e["source_type"],
                model_name=model_name,
                model_result=e.get("model_result", {}),
                image_path=e.get("image_path"),
                extra_meta=e.get("extra_meta"),
            )
            if ok:
                registered += 1
            else:
                skipped += 1
        return {"registered": registered, "skipped": skipped}


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo with local temp dir
    import tempfile
    tmp = tempfile.mkdtemp()
    db = CBFHDatabase(db_dir=tmp)

    # Model 1 (NRGA-Net Localization) registers images
    print("=== Model 1: NRGA-Net Localization ===")
    db.register_image(
        image_id="diff_001",
        cbfh_hash="abc123def456",
        source_type="diffusion",
        model_name="nrga_net_loc",
        model_result={"pred_mask_iou": 0.58, "cls_prob": 0.99, "source_pred": "diffusion"},
    )
    db.register_image(
        image_id="gan_001",
        cbfh_hash="789ghi012jkl",
        source_type="gan",
        model_name="nrga_net_loc",
        model_result={"pred_mask_iou": 0.92, "cls_prob": 1.00, "source_pred": "gan"},
    )
    db.register_image(
        image_id="lama_001",
        cbfh_hash="mno345pqr678",
        source_type="lama",
        model_name="nrga_net_loc",
        model_result={"pred_mask_iou": 0.45, "cls_prob": 0.87, "source_pred": "lama"},
    )

    # Try duplicate — should skip
    ok = db.register_image(
        image_id="diff_001",
        cbfh_hash="DIFFERENT_HASH",
        source_type="diffusion",
        model_name="nrga_net_loc",
        model_result={"overwritten": True},
    )
    print(f"  Duplicate register attempt: {'SKIPPED (correct)' if not ok else 'ERROR: overwrote!'}")

    # Model 2 (Full Fake Detection) adds its results
    print("\n=== Model 2: Full Fake Detection ===")
    db.add_model_result(
        image_id="diff_001",
        model_name="full_fake_det",
        model_result={"is_fake": True, "confidence": 0.97, "method_pred": "diffusion"},
    )
    db.add_model_result(
        image_id="gan_001",
        model_name="full_fake_det",
        model_result={"is_fake": True, "confidence": 0.99, "method_pred": "gan"},
    )

    # Try duplicate model result — should skip
    ok = db.add_model_result(
        image_id="diff_001",
        model_name="full_fake_det",
        model_result={"overwritten": True},
    )
    print(f"  Duplicate model result: {'SKIPPED (correct)' if not ok else 'ERROR: overwrote!'}")

    # Query
    print(f"\n=== Stats ===")
    print(json.dumps(db.stats(), indent=2))

    print(f"\n=== Image diff_001 full record ===")
    print(json.dumps(db.get_image("diff_001"), indent=2))

    print(f"\n=== Unprocessed by full_fake_det ===")
    unprocessed = db.get_unprocessed_by("full_fake_det")
    print(f"  {list(unprocessed.keys())}")

    print(f"\nDB saved at: {db.db_path}")
