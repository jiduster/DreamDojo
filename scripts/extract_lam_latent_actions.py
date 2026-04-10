#!/usr/bin/env python3
"""
Offline latent action extraction with DreamDojo LAM.

This script recursively scans input videos, extracts one latent action (32-d) for
each consecutive frame transition, and saves outputs as .npy files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from external.lam.model import LAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract latent actions using DreamDojo LAM checkpoint.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/mnt/ceph2/EgoDex"),
        help="Root directory to recursively search for video files.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=Path("/mnt/ceph2/EgoDex/LAM_400k.ckpt"),
        help="LAM checkpoint path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/ceph2/EgoDex/latent_actions_lam400k"),
        help="Output root for extracted latent action files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.mp4",
        help="Video glob pattern under input root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of frame-pairs per inference batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--save-index",
        action="store_true",
        help="Save an index json with extraction metadata.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of data shards (for multi-process parallel extraction).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Current shard index in [0, num_shards).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=-1,
        help="CUDA device index to use. If negative, auto-select from shard index.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress every N processed videos.",
    )
    return parser.parse_args()


def center_crop_to_ratio(frame_rgb: np.ndarray, target_ratio: float = 640.0 / 480.0) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    ratio = w / h
    if ratio > target_ratio:
        target_w = int(h * target_ratio)
        offset = (w - target_w) // 2
        return frame_rgb[:, offset : offset + target_w]
    if ratio < target_ratio:
        target_h = int(w / target_ratio)
        offset = (h - target_h) // 2
        return frame_rgb[offset : offset + target_h, :]
    return frame_rgb


def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = center_crop_to_ratio(frame_rgb)
    frame_rgb = cv2.resize(frame_rgb, (320, 240), interpolation=cv2.INTER_LINEAR)
    frame = torch.from_numpy(frame_rgb).float() / 255.0
    return frame


def flush_batch(model: LAM, pair_batch: List[torch.Tensor], device: torch.device) -> np.ndarray:
    videos = torch.stack(pair_batch, dim=0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = model.lam.encode(videos)
    # z_rep shape: (B, 1, 1, 32) because each sample has 2 frames.
    latent = outputs["z_rep"][:, 0, 0, :].detach().cpu().numpy().astype(np.float32)
    return latent


def extract_video_latents(
    model: LAM,
    video_path: Path,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ok, prev_frame = cap.read()
    if not ok:
        cap.release()
        return np.zeros((0, 32), dtype=np.float32), 0

    prev = preprocess_frame(prev_frame)
    pair_batch: List[torch.Tensor] = []
    all_latents: List[np.ndarray] = []
    frame_count = 1

    while True:
        ok, cur_frame = cap.read()
        if not ok:
            break
        frame_count += 1
        cur = preprocess_frame(cur_frame)
        pair_batch.append(torch.stack([prev, cur], dim=0))
        prev = cur

        if len(pair_batch) >= batch_size:
            all_latents.append(flush_batch(model, pair_batch, device))
            pair_batch.clear()

    cap.release()

    if pair_batch:
        all_latents.append(flush_batch(model, pair_batch, device))

    if not all_latents:
        return np.zeros((0, 32), dtype=np.float32), frame_count
    return np.concatenate(all_latents, axis=0), frame_count


def build_model(ckpt_path: Path, device: torch.device) -> LAM:
    model = LAM(
        image_channels=3,
        lam_model_dim=1024,
        lam_latent_dim=32,
        lam_patch_size=16,
        lam_enc_blocks=24,
        lam_dec_blocks=24,
        lam_num_heads=16,
        ckpt_path=str(ckpt_path),
    )
    model.eval()
    model.to(device)
    return model


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    ckpt_path = args.ckpt_path.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")

    output_root.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count <= 0:
            raise RuntimeError("No CUDA device is visible.")
        if args.gpu_id >= 0:
            if args.gpu_id >= visible_gpu_count:
                raise ValueError(
                    f"--gpu-id={args.gpu_id} out of range for visible CUDA devices ({visible_gpu_count})."
                )
            gpu_id = args.gpu_id
        else:
            gpu_id = args.shard_index % visible_gpu_count
        device = torch.device(f"cuda:{gpu_id}")
    else:
        gpu_id = -1
        device = torch.device("cpu")
    model = build_model(ckpt_path, device)

    video_paths = sorted(input_root.rglob(args.glob))
    if not video_paths:
        raise RuntimeError(f"No videos found under {input_root} with pattern {args.glob}")
    video_paths = video_paths[args.shard_index :: args.num_shards]
    if not video_paths:
        print(
            f"[DONE] shard={args.shard_index}/{args.num_shards} no assigned videos. "
            f"input_root={input_root}"
        )
        return

    index_rows = []
    processed = 0
    skipped = 0
    failed = 0

    print(
        f"[INFO] shard={args.shard_index}/{args.num_shards} gpu_id={gpu_id} "
        f"device={device} assigned_videos={len(video_paths)}"
    )

    for i, video_path in enumerate(video_paths, start=1):
        rel = video_path.relative_to(input_root)
        out_path = (output_root / rel).with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            latents, n_frames = extract_video_latents(
                model=model,
                video_path=video_path,
                batch_size=args.batch_size,
                device=device,
            )
            np.save(out_path, latents)
            processed += 1
            if args.save_index:
                index_rows.append(
                    {
                        "video": str(video_path),
                        "latent_path": str(out_path),
                        "num_frames": n_frames,
                        "num_latent_actions": int(latents.shape[0]),
                    }
                )
            print(
                f"[OK] {video_path} -> {out_path} "
                f"({latents.shape[0]} x {latents.shape[1] if latents.size else 32})"
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {video_path}: {exc}")

        if i % args.log_every == 0:
            print(
                f"[PROGRESS] shard={args.shard_index} done={i}/{len(video_paths)} "
                f"processed={processed} skipped={skipped} failed={failed}"
            )

    if args.save_index:
        if args.num_shards > 1:
            index_path = output_root / f"latent_action_index.shard{args.shard_index:02d}.json"
        else:
            index_path = output_root / "latent_action_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_rows, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Index saved to {index_path}")

    print(
        f"[DONE] shard={args.shard_index}/{args.num_shards} total={len(video_paths)} "
        f"processed={processed} skipped={skipped} failed={failed} output_root={output_root}"
    )


if __name__ == "__main__":
    main()
