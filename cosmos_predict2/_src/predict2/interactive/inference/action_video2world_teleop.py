"""
Real-time action-conditioned video generation with streaming visualization.

This script generates video frames on-the-fly as actions come in (from LeapMotion, 
random source, or any real-time input). Frames are visualized with low latency.

Example:
python -m cosmos_predict2._src.predict2.interactive.inference.action_video2world_teleop \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
    --experiment=cosmos_predict2p5_2B_action_gr00t_gr1_self_forcing_no_s3 \
    --ckpt_path /path/to/checkpoint \
    --input_video /path/to/conditioning_video.mp4 \
    --action_source random \
    --max_latent_frames 100
"""

import argparse
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import mediapy
import numpy as np
import torch
from loguru import logger

try:
    from megatron.core import parallel_state
except Exception:
    class _DummyParallelState:
        def is_initialized(self):
            return False
        def get_context_parallel_group(self):
            return None
        def initialize_model_parallel(self, **kwargs):
            return None
        def destroy_model_parallel(self):
            return None
    parallel_state = _DummyParallelState()

from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.interactive.inference.action_video2world import (
    ActionStreamingInference,
)
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time action-conditioned Video2World inference")
    # === Same as original streaming script ===
    parser.add_argument(
        "--config", type=str, default="cosmos_predict2/_src/predict2/interactive/configs/config.py", help="Config file"
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config name")
    parser.add_argument("--ckpt_path", type=str, default="", help="S3/local checkpoint path")
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument("--input_video", type=str, default=None, help="Conditioning video (first frame used)")
    parser.add_argument("--input_frame", type=str, default=None, help="Conditioning frame (PNG/JPG image, alternative to --input_video)")
    parser.add_argument("--resolution", type=str, default="none", help="Optional resolution H,W (e.g. 256,320)")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS for output")
    parser.add_argument("--num_steps", type=int, default=4, help="Student steps per frame")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--start_frame_idx", type=int, default=0, help="Start frame index (only for --input_video)")
    parser.add_argument(
        "--cr1_embeddings_path",
        type=str,
        default="datasets/cr1_dreamdojo_text_embeddings.pt",
        help="Local path to CR1 empty-string text embeddings (.pt)",
    )
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel size")

    # === New args for realtime mode ===
    parser.add_argument("--action_source", type=str, default="random", choices=["random", "leapmotion", "file"],
                        help="Source of actions: random, file, or leapmotion")
    parser.add_argument("--action_file", type=str, default=None, help="Action file path (for --action_source file)")
    parser.add_argument("--action_dim", type=int, default=384, help="Action dimension")
    parser.add_argument("--max_latent_frames", type=int, default=100, help="Max latent frames to generate")
    parser.add_argument("--save_output", type=str, default=None, help="Path to save output video")
    # Display settings
    parser.add_argument("--display_width", type=int, default=640, help="Display window width")
    parser.add_argument("--display_height", type=int, default=480, help="Display window height")
    parser.add_argument("--no_display", action="store_true", help="Disable display (headless mode)")
    # === Performance options ===
    parser.add_argument("--compile_model", action="store_true", help="Use torch.compile() for faster inference")
    parser.add_argument("--skip_decode_display", type=int, default=1, help="Only decode every N chunks for display (1=all)")
    parser.add_argument("--warmup", action="store_true", help="Run warmup iteration before timing")
    parser.add_argument("--fast_mode", action="store_true", help="Enable all speed optimizations")
    return parser.parse_args()


class ActionSource:
    """Base class for action sources."""
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    
    def get_actions(self, num_actions: int) -> np.ndarray:
        """Get next batch of actions. Returns shape (num_actions, action_dim)."""
        raise NotImplementedError


class RandomActionSource(ActionSource):
    """Generates random actions (for testing)."""
    def __init__(self, action_dim: int, seed: int = 42):
        super().__init__(action_dim)
        self.rng = np.random.default_rng(seed)
        # Small random actions to simulate gentle movements
        self.scale = 0.05
    
    def get_actions(self, num_actions: int) -> np.ndarray:
        # Generate smooth random walk actions
        actions = self.rng.normal(0, self.scale, (num_actions, self.action_dim)).astype(np.float32)
        return actions


class FileActionSource(ActionSource):
    """Reads actions from a file sequentially."""
    def __init__(self, action_dim: int, file_path: str):
        super().__init__(action_dim)
        self.actions = np.load(file_path).astype(np.float32)
        self.idx = 0
        logger.info(f"Loaded {len(self.actions)} actions from {file_path}")
    
    def get_actions(self, num_actions: int) -> np.ndarray:
        if self.idx + num_actions > len(self.actions):
            # Wrap around or repeat last action
            remaining = len(self.actions) - self.idx
            actions = np.zeros((num_actions, self.action_dim), dtype=np.float32)
            if remaining > 0:
                actions[:remaining] = self.actions[self.idx:]
            actions[remaining:] = self.actions[-1]  # Repeat last action
            self.idx = len(self.actions)
        else:
            actions = self.actions[self.idx:self.idx + num_actions]
            self.idx += num_actions
        return actions


class LeapMotionActionSource(ActionSource):
    """
    Placeholder for LeapMotion integration.
    Replace this with actual LeapMotion SDK integration.
    """
    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        logger.warning("LeapMotion source not implemented - using random fallback")
        self._fallback = RandomActionSource(action_dim)
    
    def get_actions(self, num_actions: int) -> np.ndarray:
        # TODO: Implement actual LeapMotion integration
        return self._fallback.get_actions(num_actions)


class RealtimeVisualizer:
    """Real-time video frame visualizer using OpenCV."""
    
    def __init__(self, width: int, height: int, window_name: str = "Real-time Generation"):
        self.width = width
        self.height = height
        self.window_name = window_name
        self.frame_buffer = deque(maxlen=30)  # Buffer for smoother display
        self.running = True
        self.current_frame = None
        self.frame_count = 0
        self.start_time = None
        self.lock = threading.Lock()
        
        # Initialize OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)
    
    def update_frame(self, frame: np.ndarray):
        """
        Update the current frame. Frame should be HWC, uint8, RGB.
        """
        with self.lock:
            # Convert RGB to BGR for OpenCV
            if frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Resize if needed
            if frame_bgr.shape[:2] != (self.height, self.width):
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
            
            self.current_frame = frame_bgr
            self.frame_count += 1
            
            if self.start_time is None:
                self.start_time = time.time()
    
    def display(self) -> bool:
        """
        Display current frame. Returns False if window is closed.
        Call this in main loop.
        """
        with self.lock:
            if self.current_frame is not None:
                # Add FPS overlay
                if self.start_time is not None and self.frame_count > 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    frame_display = self.current_frame.copy()
                    cv2.putText(
                        frame_display, 
                        f"FPS: {fps:.1f} | Frame: {self.frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                else:
                    frame_display = self.current_frame
                
                cv2.imshow(self.window_name, frame_display)
        
        # Check for window close or 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            self.running = False
            return False
        return True
    
    def close(self):
        """Clean up resources."""
        self.running = False
        cv2.destroyAllWindows()


class RealtimeStreamingInference(ActionStreamingInference):
    """
    Real-time action-conditioned video generation with streaming output.

    Inherits from ``ActionStreamingInference`` to reuse its comprehensive
    ``torch.compile`` setup (TE op replacement, activation-checkpointing
    unwrapping, tensor-based stateless KV cache, per-block DiT compilation,
    and VAE encode/decode compilation).  Only the real-time loop and
    conditioning-frame helpers are added here.
    """

    def __init__(
        self,
        config_path: str,
        experiment_name: str,
        ckpt_path: str,
        s3_credential_path: str,
        cr1_embeddings_path: str,
        context_parallel_size: int = 1,
        compile_model: bool = False,
    ):
        # CUDA performance flags (set before model load)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Parent handles: model loading, TE replacements, activation-checkpointing
        # unwrapping, tensor KV cache install, per-block torch.compile, VAE
        # encode/decode compilation, and CR1 embedding loading.
        super().__init__(
            config_path=config_path,
            experiment_name=experiment_name,
            ckpt_path=ckpt_path,
            s3_credential_path=s3_credential_path,
            cr1_embeddings_path=cr1_embeddings_path,
            context_parallel_size=context_parallel_size,
            enable_fsdp=False,
            torch_compile=compile_model,
        )

        self._warmed_up = False
        # Number of pixel frames per latent frame (temporal compression ratio)
        self.actions_per_latent = 4

        # Pre-cache T5 embeddings on GPU for faster access during generation
        self.t5_text_embeddings_gpu = self.t5_text_embeddings_cpu.to(
            device=self.model.tensor_kwargs["device"], dtype=torch.bfloat16,
        )
        self.t5_text_mask_gpu = torch.ones(
            (self.t5_text_embeddings_gpu.shape[0], self.t5_text_embeddings_gpu.shape[1]),
            device=self.model.tensor_kwargs["device"],
            dtype=torch.bfloat16,
        )

    def _prepare_conditioning_frame(
        self, 
        video_path: str, 
        resolution_hw: Optional[Tuple[int, int]] = None,
        start_frame_idx: int = 0
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Load and prepare the conditioning frame from video."""
        video_array = mediapy.read_video(video_path)
        if video_array.ndim == 3:
            video_array = video_array[None, ...]
        
        # mediapy.read_video returns float [0, 1] - convert to [0, 255]
        if video_array.dtype != np.uint8:
            video_array = (video_array * 255).astype(np.uint8)
        
        Ht, Wt = video_array.shape[1], video_array.shape[2]
        if resolution_hw is not None:
            Ht, Wt = resolution_hw
        
        frame = video_array[start_frame_idx]
        if (frame.shape[0], frame.shape[1]) != (Ht, Wt):
            # Resize requires float input, then convert back to uint8
            frame_float = frame.astype(np.float32) / 255.0
            frame_float = mediapy.resize_image(frame_float, (Ht, Wt))
            frame = (frame_float * 255).astype(np.uint8)
        
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        frame_tensor = torch.from_numpy(frame_uint8).permute(2, 0, 1).unsqueeze(0).contiguous()
        
        return frame_tensor, (Ht, Wt)

    def _prepare_conditioning_frame_from_image(
        self,
        image_path: str,
        resolution_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Load and prepare the conditioning frame from an image file (PNG/JPG)."""
        from PIL import Image
        
        img = Image.open(image_path).convert("RGB")
        frame = np.array(img)  # [H, W, 3] uint8
        
        Ht, Wt = frame.shape[0], frame.shape[1]
        if resolution_hw is not None:
            Ht, Wt = resolution_hw
        
        if (frame.shape[0], frame.shape[1]) != (Ht, Wt):
            # Resize
            frame_float = frame.astype(np.float32) / 255.0
            frame_float = mediapy.resize_image(frame_float, (Ht, Wt))
            frame = (frame_float * 255).astype(np.uint8)
        
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        frame_tensor = torch.from_numpy(frame_uint8).permute(2, 0, 1).unsqueeze(0).contiguous()
        
        logger.info(f"Loaded conditioning frame from image: {image_path}, shape: {frame_tensor.shape}")
        return frame_tensor, (Ht, Wt)

    def _decode_latent_to_frame(self, latent: torch.Tensor) -> np.ndarray:
        """Decode a single latent frame to pixel space (uses compiled VAE)."""
        video = self._decode(latent)
        video = video.clip(min=-1, max=1)
        # Convert from [-1, 1] to [0, 255]
        frame = ((video[0, :, 0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        return frame

    def _run_single_chunk(
        self,
        cond_frames: torch.Tensor,
        actions_np: np.ndarray,
        chunk_idx: int,
        seed: int,
        num_steps: int,
        start_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single chunk of generation. Returns (latents, video).

        Args:
            cond_frames: Conditioning frames, shape ``[B, C, T, H, W]`` (uint8).
            actions_np: Actions for this chunk, shape ``[B, chunk_size, A]``.
            chunk_idx: Current chunk index (used to vary the random seed).
            seed: Base random seed.
            num_steps: Number of student denoising steps.
            start_idx: Latent start index for the KV cache.

        Returns:
            Tuple of ``(latents, video)`` where *video* is the decoded pixel
            output in ``[-1, 1]`` range with shape ``[B, C, T_out, H, W]``.
        """
        cache_frame_size = self.config.model.config.cache_frame_size
        chunk_size = cache_frame_size * self.actions_per_latent

        # Prepare video input: cond frames + zero-padded tail
        first_stack = cond_frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        zeros_tail = torch.zeros_like(first_stack[:, :1]).repeat(
            1, chunk_size - first_stack.shape[1] + 1, 1, 1, 1,
        )
        vid_chw_t = torch.cat([first_stack, zeros_tail], dim=1)
        video_b_c_t_h_w = vid_chw_t.permute(0, 2, 1, 3, 4).contiguous()

        # Prepare data batch (inherited from parent)
        data_batch = self._prepare_data_batch(
            video_b_c_t_h_w=video_b_c_t_h_w,
            actions_np=actions_np,
            fps=4,
            num_latent_conditional_frames=0,
        )

        # Normalize and augment
        self.model._normalize_video_databatch_inplace(data_batch)
        self.model._augment_image_dim_inplace(data_batch)

        # Inject pre-cached GPU T5 embeddings and mask
        data_batch["t5_text_embeddings"] = self.t5_text_embeddings_gpu
        data_batch["t5_text_mask"] = self.t5_text_mask_gpu

        # Ensure bf16
        for k, v in list(data_batch.items()):
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.to(dtype=self.model.tensor_kwargs["dtype"])

        # Construct condition
        _, x0, condition, _ = self.model.get_data_and_condition(data_batch)
        x0 = x0.to(dtype=self.model.tensor_kwargs["dtype"])
        condition = condition.edit_data_type(DataType.VIDEO)
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=None,
            random_max_num_conditional_frames=None,
            num_conditional_frames=data_batch[NUM_CONDITIONAL_FRAMES_KEY],
        )

        # Init noise with correct latent shape
        _T, _H, _W = data_batch[self.model.input_data_key].shape[-3:]
        state_shape = (
            self.model.config.state_ch,
            int(self.model.tokenizer.get_latent_num_frames(_T)),
            _H // self.model.tokenizer.spatial_compression_factor,
            _W // self.model.tokenizer.spatial_compression_factor,
        )
        noise = misc.arch_invariant_rand(
            (1, *state_shape),
            torch.float32,
            self.model.tensor_kwargs["device"],
            seed + chunk_idx,
        )

        # Clamp steps to configured student schedule length
        K = len(self.model.config.selected_sampling_time)
        n_steps = max(1, min(int(num_steps), K))

        # Generate latents (stateless KV cache when compiled for zero graph breaks)
        use_stateless_kv = self.torch_compile
        latents = self.model.generate_streaming_video(
            condition, noise,
            n_steps=n_steps,
            cache_frame_size=cache_frame_size,
            start_idx=start_idx,
            stateless_kv=use_stateless_kv,
        )

        # Decode using compiled VAE
        video = self._decode(latents)
        video = video.clip(min=-1, max=1)

        return latents, video

    @torch.inference_mode()
    def run_realtime_generation(
        self,
        conditioning_frame: torch.Tensor,
        resolution_hw: Tuple[int, int],
        action_source: ActionSource,
        visualizer: Optional[RealtimeVisualizer],
        num_steps: int = 4,
        seed: int = 1,
        max_latent_frames: int = 100,
        fps: float = 10.0,
        on_frame_callback: Optional[Callable[[np.ndarray, int], None]] = None,
        skip_decode_display: int = 1,
        do_warmup: bool = False,
    ):
        """Run real-time generation loop using chunk-based approach.

        The generation pipeline mirrors ``generate_action_streaming`` from
        ``action_video2world.py`` including:

        * Stateless KV cache (when ``torch_compile`` is enabled).
        * Compiled VAE encode/decode.
        * O(1) video-chunk accumulation (concatenate once at end).
        * Proper ``torch.compile`` warmup (2 full pipeline passes).

        Args:
            conditioning_frame: First frame tensor ``[1, 3, H, W]``, uint8.
            resolution_hw: Target resolution ``(H, W)``.
            action_source: Source of real-time actions.
            visualizer: Optional :class:`RealtimeVisualizer` for display.
            num_steps: Denoising steps per chunk.
            seed: Random seed.
            max_latent_frames: Maximum number of latent *chunks* to generate.
            fps: Target FPS (used for final stat reporting).
            on_frame_callback: ``callback(frame, frame_idx)`` per generated frame.
            skip_decode_display: Only decode every N chunks for display (1=all).
            do_warmup: Run warmup before timing (ignored if torch_compile already
                triggers automatic warmup).
        """
        Ht, Wt = resolution_hw
        cache_frame_size = self.config.model.config.cache_frame_size
        chunk_size = cache_frame_size * self.actions_per_latent  # actions per chunk

        # ---- conditioning frame: add temporal dim [B, C, 1, H, W] ----
        cond_frames = conditioning_frame.unsqueeze(2)  # [1, 3, 1, H, W]

        # Display the conditioning frame first
        cond_frame_display = conditioning_frame[0].permute(1, 2, 0).numpy()
        if visualizer is not None:
            visualizer.update_frame(cond_frame_display)
            if not visualizer.display():
                return []

        # Collect video chunks in a list; concatenate once at end (O(1) per iter).
        _video_chunks: list[torch.Tensor] = [
            conditioning_frame.float().unsqueeze(2) / 128 - 1,
        ]
        all_frames: list[np.ndarray] = [cond_frame_display]
        _new_frames_per_chunk: list[int] = []

        # ---- torch.compile warmup: 2 full pipeline passes ----
        if (self.torch_compile or do_warmup) and not self._warmed_up:
            logger.info("Running torch.compile warmup (2 passes, may take ~2 min on first run)...")
            warmup_actions = action_source.get_actions(chunk_size)
            if warmup_actions is not None:
                warmup_actions = warmup_actions[np.newaxis, ...]
                start_idx = (cond_frames.shape[2] - 1) // 4 + 1
                for _ in range(2):
                    self._run_single_chunk(
                        cond_frames, warmup_actions, 0, seed, num_steps, start_idx,
                    )
                torch.cuda.synchronize()
                self._warmed_up = True
                # Reset file action source index after warmup consumed actions
                if hasattr(action_source, "idx"):
                    action_source.idx = 0
                logger.info("Warmup complete!")

        logger.info(
            f"Starting real-time generation (max {max_latent_frames} latent chunks, "
            f"skip_decode={skip_decode_display})..."
        )

        chunk_idx = 0
        total_latents_generated = 0
        _chunk_timings: list[float] = []
        start_time = time.time()

        while total_latents_generated < max_latent_frames - 1:
            if visualizer is not None and not visualizer.running:
                logger.info("Visualization closed, stopping generation")
                break

            # Get actions from source
            actions_np = action_source.get_actions(chunk_size)
            if actions_np is None:
                logger.info("No more actions available")
                break
            actions_np = actions_np[np.newaxis, ...]

            start_idx = (cond_frames.shape[2] - 1) // 4 + 1

            # ---- Generate & decode (timed) ----
            torch.cuda.synchronize()
            t0 = time.time()

            latents, video = self._run_single_chunk(
                cond_frames, actions_np, chunk_idx, seed, num_steps, start_idx,
            )

            torch.cuda.synchronize()
            t1 = time.time()
            _chunk_timings.append(t1 - t0)

            # ---- Extract and display new frames ----
            new_frames = video[:, :, ((start_idx - 1) * 4 + 1) :]
            _nf = new_frames.shape[2]
            _video_chunks.append(new_frames.cpu())
            _new_frames_per_chunk.append(_nf)

            should_decode = (skip_decode_display == 1) or ((chunk_idx + 1) % skip_decode_display == 0)
            if should_decode:
                for t in range(new_frames.shape[2]):
                    frame_float = new_frames[0, :, t].float().permute(1, 2, 0).cpu().numpy()
                    frame_uint8 = ((frame_float + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                    all_frames.append(frame_uint8)

                    if visualizer is not None:
                        visualizer.update_frame(frame_uint8)
                        if not visualizer.display():
                            return all_frames

                    if on_frame_callback is not None:
                        on_frame_callback(frame_uint8, len(all_frames) - 1)

            # Update conditioning frames for next chunk (keep on GPU)
            cond_frames = (
                (1.0 + video[:, :, -((cache_frame_size - 1) * 4 + 1) :]) / 2 * 255.0
            ).to(torch.uint8)

            total_latents_generated += 1
            chunk_idx += 1

            # Log timing every 10 chunks
            if chunk_idx % 10 == 0:
                elapsed = time.time() - start_time
                total_pixel_frames = sum(_new_frames_per_chunk)
                pixel_fps = total_pixel_frames / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Chunk {chunk_idx}: {total_pixel_frames} frames in "
                    f"{elapsed:.1f}s ({pixel_fps:.1f} FPS)"
                )

        # ---- Final stats ----
        end_time = time.time()
        total_time = end_time - start_time
        final_video = torch.cat(_video_chunks, dim=2)
        total_generated = final_video.shape[2] - 1  # exclude conditioning frame

        # Steady-state timing (skip chunk 0 which includes extra initial frames)
        ss_timings = _chunk_timings[1:] if len(_chunk_timings) > 1 else _chunk_timings
        ss_nf = _new_frames_per_chunk[1:] if len(_new_frames_per_chunk) > 1 else _new_frames_per_chunk
        new_frames_ss = ss_nf[0] if ss_nf else (_new_frames_per_chunk[0] if _new_frames_per_chunk else 4)
        avg_chunk_ms = (sum(ss_timings) / max(len(ss_timings), 1)) * 1000

        logger.info("=" * 60)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total video: {total_generated} generated + 1 conditioning = {final_video.shape[2]} frames")
        logger.info(f"Video generation time (end-to-end): {total_time:.3f}s")
        logger.info(f"Overall FPS: {total_generated / total_time:.2f}" if total_time > 0 else "N/A")
        logger.info(f"Steady-state per chunk: {avg_chunk_ms:.0f}ms → {new_frames_ss} new frames")
        playback_budget_ms = new_frames_ss / fps * 1000
        if avg_chunk_ms <= playback_budget_ms:
            logger.info(
                f"  ✓ {avg_chunk_ms:.0f}ms < {playback_budget_ms:.0f}ms — "
                f"REAL-TIME ({avg_chunk_ms / playback_budget_ms:.2f}x)"
            )
        else:
            logger.info(
                f"  ✗ {avg_chunk_ms:.0f}ms > {playback_budget_ms:.0f}ms — "
                f"{avg_chunk_ms / playback_budget_ms:.2f}x real-time"
            )
        logger.info("=" * 60)

        return all_frames

    # _prepare_data_batch and cleanup are inherited from ActionStreamingInference.


@torch.inference_mode()
def main():
    args = parse_arguments()

    # The KV cache's current_idx is a Python int that changes each chunk,
    # causing torch.compile guard specialisation and recompilations.
    # Raise the limit so all unique idx values get compiled without warnings.
    torch._dynamo.config.recompile_limit = 64  # type: ignore[attr-defined]

    # === Fast mode enables all optimizations ===
    if args.fast_mode:
        args.compile_model = True
        args.warmup = True
        args.skip_decode_display = 1  # Decode all (VAE decode is fast when compiled)
        logger.info("FAST MODE enabled: compile + warmup")
    
    # Set up CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.error("CUDA not available!")
        return
    
    # Create action source
    if args.action_source == "random":
        action_source = RandomActionSource(args.action_dim, seed=args.seed)
        logger.info("Using random action source")
    elif args.action_source == "file":
        if args.action_file is None:
            raise ValueError("--action_file required for file source")
        action_source = FileActionSource(args.action_dim, args.action_file)
        logger.info(f"Using file action source: {args.action_file}")
    elif args.action_source == "leapmotion":
        action_source = LeapMotionActionSource(args.action_dim)
        logger.info("Using LeapMotion action source")
    else:
        raise ValueError(f"Unknown action source: {args.action_source}")
    
    # Create visualizer (or dummy if no_display)
    if args.no_display:
        visualizer = None
        logger.info("Running in headless mode (no display)")
    else:
        visualizer = RealtimeVisualizer(
            width=args.display_width,
            height=args.display_height,
            window_name="Real-time Video Generation"
        )
    
    try:
        # Initialize inference engine
        logger.info("Initializing inference engine...")
        infer = RealtimeStreamingInference(
            config_path=args.config,
            experiment_name=args.experiment,
            ckpt_path=args.ckpt_path,
            s3_credential_path=args.s3_cred,
            cr1_embeddings_path=args.cr1_embeddings_path,
            context_parallel_size=args.context_parallel_size,
            compile_model=args.compile_model,
        )
        
        # Parse resolution
        resolution_hw = None
        if args.resolution != "none":
            try:
                h, w = map(int, args.resolution.split(","))
                resolution_hw = (h, w)
            except:
                pass
        
        # Prepare conditioning frame (from image or video)
        if args.input_frame:
            logger.info(f"Loading conditioning frame from image: {args.input_frame}")
            conditioning_frame, resolution_hw = infer._prepare_conditioning_frame_from_image(
                args.input_frame, resolution_hw
            )
        elif args.input_video:
            logger.info(f"Loading conditioning frame from video: {args.input_video}")
            conditioning_frame, resolution_hw = infer._prepare_conditioning_frame(
                args.input_video, resolution_hw, start_frame_idx=args.start_frame_idx
            )
        else:
            raise ValueError("Either --input_video or --input_frame must be provided")
        
        # Run real-time generation
        all_frames = infer.run_realtime_generation(
            conditioning_frame=conditioning_frame,
            resolution_hw=resolution_hw,
            action_source=action_source,
            visualizer=visualizer,
            num_steps=args.num_steps,
            seed=args.seed,
            max_latent_frames=args.max_latent_frames,
            fps=args.fps,
            skip_decode_display=args.skip_decode_display,
            do_warmup=args.warmup,
        )
        
        # Save output if requested
        if args.save_output and len(all_frames) > 0:
            logger.info(f"Saving output to {args.save_output}")
            # Stack frames and save
            video_array = np.stack(all_frames, axis=0)  # [T, H, W, C]
            mediapy.write_video(args.save_output, video_array, fps=args.fps)
            logger.info(f"Saved {len(all_frames)} frames at {args.fps} FPS")
        
        # Cleanup
        infer.cleanup()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if visualizer is not None:
            visualizer.close()


if __name__ == "__main__":
    main()
