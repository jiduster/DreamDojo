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

By default this script uses --max-pred-frames=48 to evaluate a single forward pass per clip
without multi-chunk stitching.
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


def ensure_mediapy_ffmpeg() -> None:
    """
    Ensure mediapy can find an ffmpeg executable.

    Prefer system ffmpeg when available; otherwise fall back to imageio-ffmpeg's bundled binary.
    """
    if mediapy.video_is_available():
        return
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        mediapy.set_ffmpeg(ffmpeg_exe)
        print(f"[eval] mediapy ffmpeg fallback: {ffmpeg_exe}")
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] Warning: failed to configure ffmpeg fallback: {exc}")


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
    parser.add_argument(
        "--max-pred-frames",
        type=int,
        default=48,
        help=(
            "Truncate each clip to exactly this many pixel frames and run a single "
            "generate_vid2world call (no script-level chunk stitching / pseudo-AR). "
            "Default: 48. "
            "Requires at least this many GT frames and max_pred_frames-1 latent rows."
        ),
    )
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
    parser.add_argument(
        "--experiment-opt",
        action="append",
        default=[],
        help=(
            "Extra experiment override passed through to Video2WorldInference, "
            "e.g. --experiment-opt model.config.text_encoder_config.ckpt_path=/local/path"
        ),
    )
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument(
        "--offload-diffusion-model",
        action="store_true",
        help="Offload diffusion network to CPU between stages to reduce GPU memory.",
    )
    parser.add_argument(
        "--offload-text-encoder",
        action="store_true",
        help="Offload text encoder to CPU when possible to reduce GPU memory.",
    )
    parser.add_argument(
        "--offload-tokenizer",
        action="store_true",
        help="Offload tokenizer encoder/decoder to CPU when possible to reduce GPU memory.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of independent workers for data-parallel sample sharding.",
    )
    parser.add_argument(
        "--worker-rank",
        type=int,
        default=0,
        help="Current worker rank in [0, num_workers).",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Optional CUDA device id for this worker. If omitted, keep current default device.",
    )
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
    max_pred_frames: int | None,
) -> dict:
    gt_video = mediapy.read_video(str(video_path))
    gt_video = resize_video_uint8(gt_video, h=480, w=640)
    latents = np.load(latent_path).astype(np.float32)
    gt_video_eval = gt_video

    if max_pred_frames is not None:
        if max_pred_frames < 2:
            raise ValueError(f"--max-pred-frames must be >= 2, got {max_pred_frames}")
        # Action-conditioned network groups actions by latent temporal compression (4 for this model).
        # Align action length to a multiple of 4 for model forward, then crop output back to max_pred_frames.
        num_action_per_latent = 4
        n_action_target = max_pred_frames - 1
        n_action = ((n_action_target + num_action_per_latent - 1) // num_action_per_latent) * num_action_per_latent
        model_num_frames = n_action + 1
        if gt_video.shape[0] < model_num_frames or latents.shape[0] < n_action:
            raise ValueError(
                f"Too short for --max-pred-frames={max_pred_frames}: "
                f"gt_frames={gt_video.shape[0]}, latent_rows={latents.shape[0]} "
                f"(need >= {model_num_frames} frames and >= {n_action} latent rows)"
            )
        gt_video = gt_video[:model_num_frames].copy()
        gt_video_eval = gt_video[:max_pred_frames].copy()
        latents = latents[:n_action]
    elif gt_video.shape[0] < 2 or latents.shape[0] < chunk_size:
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

    if max_pred_frames is not None:
        n_action = max_pred_frames - 1
        actions_chunk = action
        current_lam_video = lam_video[0 : n_action * 2]
        if current_lam_video.shape[0] != n_action * 2:
            raise RuntimeError(
                f"lam_video length mismatch: got {current_lam_video.shape[0]}, expected {n_action * 2}"
            )
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
            seed=seed,
            negative_prompt="",
            lam_video=current_lam_video,
        )
        video_norm = (video - (-1)) / (1 - (-1))
        video_uint8 = (torch.clamp(video_norm[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        chunk_video.append(video_uint8)
    else:
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
            vid_input = torch.cat(
                [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0
            )
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

    if max_pred_frames is not None:
        pred_video = chunk_video[0]
        if pred_video.shape[0] != max_pred_frames:
            pred_video = pred_video[:max_pred_frames]
    else:
        pred_video = np.concatenate(
            [chunk_video[0]] + [chunk_video[i][:chunk_size] for i in range(1, len(chunk_video))], axis=0
        )
    gt_crop = gt_video_eval[: pred_video.shape[0]]
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
        "max_pred_frames": max_pred_frames,
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
    ensure_mediapy_ffmpeg()
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError(f"--num-workers must be >= 1, got {args.num_workers}")
    if args.worker_rank < 0 or args.worker_rank >= args.num_workers:
        raise ValueError(
            f"--worker-rank must be in [0, {args.num_workers}), got {args.worker_rank}"
        )

    # Allow torchrun-based launch without explicitly passing sharding args.
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = int(os.environ.get("RANK", "0"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size_env > 1 and args.num_workers == 1 and args.worker_rank == 0:
        args.num_workers = world_size_env
        args.worker_rank = rank_env
        if args.device_id is None:
            args.device_id = local_rank_env

    if args.device_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--device-id is set but CUDA is not available.")
        torch.cuda.set_device(args.device_id)
        print(f"[worker {args.worker_rank}] Using CUDA device {args.device_id}")

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
    if args.num_workers > 1:
        pairs = [p for i, p in enumerate(pairs) if i % args.num_workers == args.worker_rank]
        print(
            f"[worker {args.worker_rank}/{args.num_workers}] Assigned {len(pairs)} samples "
            f"(max_samples={args.max_samples})"
        )
    if not pairs:
        print(f"[worker {args.worker_rank}] No samples assigned after sharding. Exiting.")
        return

    video2world_cli = Video2WorldInference(
        experiment_name=args.experiment_name,
        ckpt_path=args.checkpoint_path,
        s3_credential_path="",
        context_parallel_size=args.context_parallel_size,
        config_file=args.config_file,
        experiment_opts=args.experiment_opt,
        offload_diffusion_model=args.offload_diffusion_model,
        offload_text_encoder=args.offload_text_encoder,
        offload_tokenizer=args.offload_tokenizer,
    )

    metrics_all = []
    failures = []
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
                max_pred_frames=args.max_pred_frames,
            )
            metrics_all.append(metrics)
            print(
                f"[{idx + 1}/{len(pairs)}] OK {vp} "
                f"PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((str(vp), str(exc)))
            print(f"[{idx + 1}/{len(pairs)}] FAIL {vp}: {exc}")

    if metrics_all:
        summary = {
            "worker_rank": args.worker_rank,
            "num_workers": args.num_workers,
            "num_samples": len(metrics_all),
            "psnr": float(np.mean([m["psnr"] for m in metrics_all])),
            "ssim": float(np.mean([m["ssim"] for m in metrics_all])),
            "lpips": float(np.mean([m["lpips"] for m in metrics_all])),
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        summary_name = "all_summary.json"
        if args.num_workers > 1:
            summary_name = f"all_summary_rank{args.worker_rank}.json"
        with open(args.output_root / summary_name, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, indent=2))
    else:
        preview = "\n".join([f"  - {vp}: {err}" for vp, err in failures[:5]])
        raise RuntimeError(
            f"No successful samples on worker {args.worker_rank}. "
            f"Observed {len(failures)} failures. Example errors:\n{preview}"
        )

    video2world_cli.cleanup()


if __name__ == "__main__":
    main()
