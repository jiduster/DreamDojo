#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mediapy as mp
import numpy as np


def center_crop_to_4_3_then_resize(image: np.ndarray, target_h: int = 480, target_w: int = 640) -> np.ndarray:
    # Match GR1 eval preprocessing in groot_dreams:
    # 1) center-crop to 4:3 aspect ratio
    # 2) resize to 480x640
    h, w = image.shape[:2]
    target_ratio = target_w / target_h  # 640/480 = 4/3
    cur_ratio = w / h

    if cur_ratio > target_ratio:
        crop_h = h
        crop_w = int(round(h * target_ratio))
    elif cur_ratio < target_ratio:
        crop_w = w
        crop_h = int(round(w / target_ratio))
    else:
        crop_h, crop_w = h, w

    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    cropped = image[top : top + crop_h, left : left + crop_w]
    return mp.resize_image(cropped, (target_h, target_w))


def pad_or_truncate_action(action: np.ndarray, target_action_dim: int) -> np.ndarray:
    if action.ndim != 2:
        raise ValueError(f"Expected action to be 2D (T, D), got shape {action.shape}")
    current_dim = action.shape[1]
    if current_dim == target_action_dim:
        return action
    if current_dim > target_action_dim:
        return action[:, :target_action_dim]

    padded = np.zeros((action.shape[0], target_action_dim), dtype=action.dtype)
    padded[:, :current_dim] = action
    return padded


def configure_local_reason1_paths() -> None:
    # Make smoke test robust to missing env exports: prefer local checkpoints by default.
    local_reason1_ckpt = "/mnt/ceph2/ckpt/Cosmos-Reason1-7B"
    local_qwen_model = "/mnt/ceph2/ckpt/Qwen2.5-VL-7B-Instruct"

    if not os.environ.get("COSMOS_LOCAL_REASON1_CKPT_PATH") and Path(local_reason1_ckpt).exists():
        os.environ["COSMOS_LOCAL_REASON1_CKPT_PATH"] = local_reason1_ckpt
    if not os.environ.get("COSMOS_LOCAL_REASON1_TOKENIZER_DIR") and Path(local_reason1_ckpt).exists():
        os.environ["COSMOS_LOCAL_REASON1_TOKENIZER_DIR"] = local_reason1_ckpt
    if not os.environ.get("COSMOS_LOCAL_REASON1_MODEL_DIR") and Path(local_qwen_model).exists():
        os.environ["COSMOS_LOCAL_REASON1_MODEL_DIR"] = local_qwen_model

    os.environ.setdefault("COSMOS_DISABLE_REMOTE_CKPT_RESOLVE", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/dd_gr1_rollout_smoke"))
    parser.add_argument("--ckpt", type=Path, default=Path("/mnt/ceph/ckpt/2B_GR1_post-train/2B_GR1_post-train/iter_000050000/model_ema_bf16.pt"))
    parser.add_argument("--experiment", type=str, default="dreamdojo_2b_480_640_gr1")
    parser.add_argument("--action-dim", type=int, default=384)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-latent-conditional-frames", type=int, default=1)
    args = parser.parse_args()

    configure_local_reason1_paths()
    from cosmos_predict2._src.predict2.action.inference.inference_pipeline import ActionVideo2WorldInference

    args.out_dir.mkdir(parents=True, exist_ok=True)

    video = mp.read_video(str(args.video))
    first_raw = video[0]
    first = center_crop_to_4_3_then_resize(first_raw, 480, 640)
    action = np.load(args.action).astype(np.float32)
    action = pad_or_truncate_action(action, args.action_dim)

    mp.write_image(str(args.out_dir / "first_frame_raw.png"), first_raw)
    mp.write_image(str(args.out_dir / "first_frame.png"), first)
    np.save(args.out_dir / "first_frame.npy", first)
    np.save(args.out_dir / "action_padded.npy", action)
    print(f"action shape after pad/truncate: {action.shape}")

    inf = ActionVideo2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=str(args.ckpt),
        s3_credential_path="",
        context_parallel_size=1,
    )

    next_frame, video_clamped = inf.step_inference(
        img_array=first,
        action=action,
        guidance=args.guidance,
        seed=args.seed,
        num_latent_conditional_frames=args.num_latent_conditional_frames,
    )

    mp.write_video(str(args.out_dir / "pred.mp4"), video_clamped, fps=10)
    mp.write_image(str(args.out_dir / "next_frame.png"), next_frame)
    print(f"saved to {args.out_dir}")
    print(f"video shape: {video_clamped.shape}")

    inf.cleanup()


if __name__ == "__main__":
    main()
