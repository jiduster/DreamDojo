#!/usr/bin/env python3
"""
Evaluate DreamDojo action-conditioned generation with precomputed EgoDex latent actions.

Input:
  - videos under --video-root (recursive *.mp4)
  - latent action npy under --latent-root with mirrored relative paths
    e.g. video:  video-root/foo/bar/0001.mp4
         latent: latent-root/foo/bar/0001.npy

Output:
  - *_pred.mp4, *_gt.mp4, *_merged.mp4, *_actions.npy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mediapy
import numpy as np
import piq
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EgoDex latent-action conditioned evaluation with DreamDojo action-conditioned model."
    )
    parser.add_argument("--video-root", type=Path, default=Path("/mnt/ceph2/EgoDex"))
    parser.add_argument("--latent-root", type=Path, default=Path("/mnt/ceph2/EgoDex/latent_actions_lam400k"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/ceph2/EgoDex/latent_action_eval"))
    parser.add_argument("--glob", type=str, default="*.mp4")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--guidance", type=int, default=0)
    parser.add_argument("--save-fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-latent-conditional-frames", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--resolution", type=str, default="480,640")
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--config-file",
        type=str,
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    parser.add_argument("--context-parallel-size", type=int, default=1)
    return parser.parse_args()


def resize_video_uint8(video_t_h_w_c: np.ndarray, h: int = 480, w: int = 640) -> np.ndarray:
    video = torch.from_numpy(video_t_h_w_c).permute(0, 3, 1, 2).float()
    video = F.interpolate(video, size=(h, w), mode="bilinear", align_corners=False)
    video = torch.clamp(video, 0, 255).to(torch.uint8)
    return video.permute(0, 2, 3, 1).cpu().numpy()


def build_lam_video(video_t_h_w_c: np.ndarray) -> torch.Tensor:
    # Reproduce training/inference convention used in groot_dreams datasets:
    # lam_video length is 2*(T-1), where each action step uses 2 frames.
    video = torch.from_numpy(video_t_h_w_c).permute(3, 0, 1, 2).float()  # C, T, H, W
    lam = F.interpolate(video, size=(240, 320), mode="bilinear", align_corners=False)
    lam = torch.clamp(lam / 255.0, 0, 1)
    lam = torch.repeat_interleave(lam, 2, dim=1)[:, 1:-1, :, :]
    lam = rearrange(lam, "c t h w -> t h w c")
    return lam


def run_one_video(
    video2world_cli: Video2WorldInference,
    video_path: Path,
    latent_path: Path,
    save_dir: Path,
    chunk_size: int,
    guidance: int,
    save_fps: int,
    seed: int,
    num_latent_conditional_frames: int,
    resolution: str,
    zero_actions: bool,
) -> dict:
    gt_video = mediapy.read_video(str(video_path))
    gt_video = resize_video_uint8(gt_video, h=480, w=640)
    latents = np.load(latent_path).astype(np.float32)

    if gt_video.shape[0] < 2 or latents.shape[0] < chunk_size:
        raise ValueError(f"Too short for chunking: frames={gt_video.shape[0]}, latents={latents.shape[0]}")

    action = np.concatenate(
        [
            np.zeros((latents.shape[0], 352), dtype=np.float32),
            latents,
        ],
        axis=1,
    )
    if zero_actions:
        action = np.zeros_like(action)

    lam_video = build_lam_video(gt_video)

    img_array = gt_video[0]
    first_round = True
    chunk_video = []

    for i in range(0, len(action), chunk_size):
        actions_chunk = action[i : i + chunk_size]
        if actions_chunk.shape[0] != chunk_size:
            break

        current_lam_video = lam_video[i * 2 : (i + chunk_size) * 2]
        if current_lam_video.shape[0] != chunk_size * 2:
            break

        if first_round:
            img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0) * 255.0
            first_round = False
        else:
            img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0) * 255.0

        num_video_frames = actions_chunk.shape[0] + 1
        vid_input = torch.cat([img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
        vid_input = vid_input.to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)

        video = video2world_cli.generate_vid2world(
            prompt="",
            input_path=vid_input,
            action=torch.from_numpy(actions_chunk).float(),
            guidance=guidance,
            num_video_frames=num_video_frames,
            num_latent_conditional_frames=num_latent_conditional_frames,
            resolution=resolution,
            seed=seed + i,
            negative_prompt="",
            lam_video=current_lam_video,
        )
        video_norm = (video - (-1)) / (1 - (-1))
        video_uint8 = (torch.clamp(video_norm[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        img_array = video_uint8[-1]
        chunk_video.append(video_uint8)

    if not chunk_video:
        raise RuntimeError("No valid chunk generated.")

    pred_video = np.concatenate([chunk_video[0]] + [chunk_video[i][:chunk_size] for i in range(1, len(chunk_video))], axis=0)
    gt_crop = gt_video[: pred_video.shape[0]]
    merged = np.concatenate([gt_crop, pred_video], axis=2)

    save_dir.mkdir(parents=True, exist_ok=True)
    pred_path = save_dir / f"{video_path.stem}_pred.mp4"
    gt_path = save_dir / f"{video_path.stem}_gt.mp4"
    merged_path = save_dir / f"{video_path.stem}_merged.mp4"
    action_path = save_dir / f"{video_path.stem}_actions.npy"

    mediapy.write_video(str(pred_path), pred_video, fps=save_fps)
    mediapy.write_video(str(gt_path), gt_crop, fps=save_fps)
    mediapy.write_video(str(merged_path), merged, fps=save_fps)
    np.save(action_path, action[: pred_video.shape[0] - 1])

    x_batch = torch.clamp(torch.from_numpy(pred_video) / 255.0, 0, 1).permute(0, 3, 1, 2)
    y_batch = torch.clamp(torch.from_numpy(gt_crop) / 255.0, 0, 1).permute(0, 3, 1, 2)
    psnr = float(piq.psnr(x_batch, y_batch).mean().item())
    ssim = float(piq.ssim(x_batch, y_batch).mean().item())
    lpips = float(piq.LPIPS()(x_batch, y_batch).mean().item())

    metrics = {
        "video": str(video_path),
        "latent": str(latent_path),
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "merged_path": str(merged_path),
        "num_frames_pred": int(pred_video.shape[0]),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
    }
    with open(save_dir / f"{video_path.stem}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    # Avoid remote checkpoint resolution during config import.
    # The eval script passes an explicit local --checkpoint-path anyway.
    os.environ.setdefault("COSMOS_DISABLE_REMOTE_CKPT_RESOLVE", "1")
    # Prefer local tokenizer checkpoint if available.
    os.environ.setdefault("COSMOS_LOCAL_WAN2PT1_TOKENIZER_PTH", "/mnt/ceph2/ckpt/Cosmos-Predict2.5-2B/tokenizer.pth")
    # Prefer local Reason1 checkpoint directory if available.
    os.environ.setdefault("COSMOS_LOCAL_REASON1_CKPT_PATH", "/mnt/ceph2/ckpt/Cosmos-Reason1-7B")
    # Prefer local Reason1 tokenizer/processor files.
    os.environ.setdefault("COSMOS_LOCAL_REASON1_TOKENIZER_DIR", "/mnt/ceph2/ckpt/Cosmos-Reason1-7B")
    args = parse_args()
    if not args.video_root.exists():
        raise FileNotFoundError(f"Missing video root: {args.video_root}")
    if not args.latent_root.exists():
        raise FileNotFoundError(f"Missing latent root: {args.latent_root}")

    video_paths = sorted(args.video_root.rglob(args.glob))
    pairs = []
    for vp in video_paths:
        rel = vp.relative_to(args.video_root)
        lp = (args.latent_root / rel).with_suffix(".npy")
        if lp.exists():
            pairs.append((vp, lp))
    if not pairs:
        raise RuntimeError("No matched (video, latent) pair found.")

    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    video2world_cli = Video2WorldInference(
        experiment_name=args.experiment_name,
        ckpt_path=args.checkpoint_path,
        s3_credential_path="",
        context_parallel_size=args.context_parallel_size,
        config_file=args.config_file,
    )

    metrics_all = []
    for idx, (vp, lp) in enumerate(pairs):
        save_dir = args.output_root / vp.parent.relative_to(args.video_root)
        try:
            metrics = run_one_video(
                video2world_cli=video2world_cli,
                video_path=vp,
                latent_path=lp,
                save_dir=save_dir,
                chunk_size=args.chunk_size,
                guidance=args.guidance,
                save_fps=args.save_fps,
                seed=args.seed,
                num_latent_conditional_frames=args.num_latent_conditional_frames,
                resolution=args.resolution,
                zero_actions=args.zero_actions,
            )
            metrics_all.append(metrics)
            print(
                f"[{idx + 1}/{len(pairs)}] OK {vp} "
                f"PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx + 1}/{len(pairs)}] FAIL {vp}: {exc}")

    if metrics_all:
        summary = {
            "num_samples": len(metrics_all),
            "psnr": float(np.mean([m["psnr"] for m in metrics_all])),
            "ssim": float(np.mean([m["ssim"] for m in metrics_all])),
            "lpips": float(np.mean([m["lpips"] for m in metrics_all])),
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        with open(args.output_root / "all_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, indent=2))

    video2world_cli.cleanup()


if __name__ == "__main__":
    main()

