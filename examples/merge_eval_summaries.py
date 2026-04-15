#!/usr/bin/env python3
"""
Merge per-worker DreamDojo eval summaries into one global summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge all_summary_rank*.json files.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="all_summary_rank*.json")
    parser.add_argument("--save-name", type=str, default="all_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_files = sorted(args.output_root.glob(args.pattern))
    if not summary_files:
        raise RuntimeError(f"No summary files matched pattern {args.pattern} under {args.output_root}")

    items: list[dict] = []
    for p in summary_files:
        with open(p, encoding="utf-8") as f:
            items.append(json.load(f))

    counts = np.array([float(x.get("num_samples", 0)) for x in items], dtype=np.float64)
    total = int(counts.sum())
    if total <= 0:
        raise RuntimeError("Total num_samples is 0, cannot merge summary.")

    psnr = float(np.sum(counts * np.array([x["psnr"] for x in items], dtype=np.float64)) / total)
    ssim = float(np.sum(counts * np.array([x["ssim"] for x in items], dtype=np.float64)) / total)
    lpips = float(np.sum(counts * np.array([x["lpips"] for x in items], dtype=np.float64)) / total)

    merged = {
        "num_workers": len(items),
        "num_samples": total,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
    }

    save_path = args.output_root / args.save_name
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(json.dumps(merged, indent=2))
    print(f"Saved merged summary to: {save_path}")


if __name__ == "__main__":
    main()

