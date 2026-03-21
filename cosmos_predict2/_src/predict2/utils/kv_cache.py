# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn

# Per-layer KV cache state: (k_cache_list, v_cache_list).
# Used in stateless mode to pass cache state as function arguments
# rather than storing it on the module (required for torch.compile).
KVCacheLayerState = tuple[list[torch.Tensor | None], list[torch.Tensor | None]]

# Tensor-based KV cache state: (k_cache_tensor, v_cache_tensor).
# Each tensor has shape [max_frames, B, tokens_per_frame, num_heads, head_dim].
# Used in stateless+tensor mode for full torch.compile traceability (no graph breaks).
TensorKVCacheLayerState = tuple[torch.Tensor, torch.Tensor]


@dataclass
class KVCacheConfig:
    run_with_kv: bool = False
    store_kv: bool = False
    current_idx: int = 0
    recompute_cross_attn_kv: bool = False


class AttentionOpWithKVCache(nn.Module):
    """A thin wrapper that adds K/V caching to an existing attention op.

    This wrapper expects the wrapped op to accept (q, k, v, attn_mask=None)
    and return attention outputs with heads already flattened on the last dim.

    Cache semantics:
    - Cache entries are stored as per-chunk tensors, where each chunk corresponds
      to one latent frame composed of HxW tokens (after patchify).
    - The `max_cache_size` capacity therefore refers to the number of latent
      frames (chunks), NOT the number of individual tokens.
    - When `max_cache_size` is None, the cache grows without an automatic
      rolling window; otherwise, it acts as a rolling window of at most
      `max_cache_size` frames. Upon overflow, the oldest frames are dropped.

    Modes:
    - **Stateful** (default, ``stateless=False``): K/V tensors are stored on the
      module itself (``self.k_cache`` / ``self.v_cache``).  ``forward`` accepts
      ``kv_state=None`` and returns only the attention output tensor.
    - **Stateless** (``stateless=True``): No internal cache lists are allocated.
      Instead, the caller passes the cache state via ``kv_state`` and receives
      the (possibly updated) state back as a second return value.  This keeps
      all mutable state outside the compiled region so that ``torch.compile``
      can trace the attention op without side-effects.
    """

    def __init__(
        self,
        attn_op: nn.Module | Any,
        max_cache_size: Optional[int] = None,
        stateless: bool = False,
        use_tensor_state: bool = False,
    ):
        """Initialize the KV cache wrapper.

        Args:
            attn_op: The underlying attention operation (q, k, v[, attn_mask]) -> out.
            max_cache_size: Optional capacity measured in number of latent frames
                (chunks). Each chunk is a single frame worth of HxW tokens. If None,
                the cache does not enforce a rolling capacity.
            stateless: If True, the wrapper does **not** allocate internal cache
                lists.  The caller must supply ``kv_state`` on every ``forward``
                call and use the returned state for subsequent calls.
            use_tensor_state: If True (requires ``stateless=True``), use pre-allocated
                tensors instead of Python lists for the cache state.  This makes
                ``forward`` fully traceable by ``torch.compile`` (no graph breaks).
        """
        super().__init__()
        self.attn_op = attn_op
        self.stateless = stateless
        self.use_tensor_state = use_tensor_state
        self.max_cache_size = max_cache_size
        self.pg: Optional[Any] = None
        self.stream: Optional[Any] = None
        if not stateless:
            self.reset_kv_cache(max_cache_size=max_cache_size)

    def reset_kv_cache(self, max_cache_size: Optional[int] = None) -> None:
        """Reset/initialize the KV caches.

        Only has effect in stateful mode.  In stateless mode this is a no-op.

        Args:
            max_cache_size: Optional capacity measured in number of latent frames
                (chunks). Each chunk is a single frame worth of HxW tokens. If None,
                the cache does not enforce a rolling capacity.
        """
        if self.stateless:
            self.max_cache_size = max_cache_size if max_cache_size is not None else self.max_cache_size
            return
        # Initialize list-based caches and optionally set capacity in chunks
        self.k_cache: list[torch.Tensor | None] = [None] * 99999
        self.v_cache: list[torch.Tensor | None] = [None] * 99999
        self.max_cache_size = max_cache_size

    @staticmethod
    def create_empty_state() -> KVCacheLayerState:
        """Create a fresh empty KV cache state for a single layer.

        Returns a ``(k_cache, v_cache)`` tuple of None-filled lists, suitable
        for passing as ``kv_state`` in stateless mode.
        """
        return ([None] * 99999, [None] * 99999)

    @staticmethod
    def create_empty_tensor_state(
        max_frames: int,
        batch_size: int,
        tokens_per_frame: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TensorKVCacheLayerState:
        """Create a fresh pre-allocated tensor KV cache state for a single layer.

        Returns a ``(k_cache, v_cache)`` tuple of zero-filled tensors with shape
        ``[max_frames, batch_size, tokens_per_frame, num_heads, head_dim]``.
        """
        shape = (max_frames, batch_size, tokens_per_frame, num_heads, head_dim)
        k_cache = torch.zeros(shape, device=device, dtype=dtype)
        v_cache = torch.zeros(shape, device=device, dtype=dtype)
        return (k_cache, v_cache)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache_cfg: KVCacheConfig,
        kv_state: Optional[Union[KVCacheLayerState, TensorKVCacheLayerState]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Union[KVCacheLayerState, TensorKVCacheLayerState]]]:
        """Run attention with optional KV caching.

        Dispatches to ``_forward_tensor`` (fully torch.compile-traceable, no
        graph breaks) when ``use_tensor_state=True``, or to
        ``_forward_list`` (with ``@torch.compiler.disable`` graph break)
        otherwise.

        Returns:
            * **Stateful mode** (``stateless=False``): a single output tensor.
            * **Stateless mode** (``stateless=True``): a ``(output, new_kv_state)``
              tuple so the caller can thread state through subsequent calls.
        """
        if self.use_tensor_state:
            return self._forward_tensor(q, k, v, kv_cache_cfg=kv_cache_cfg, kv_state=kv_state, **kwargs)
        return self._forward_list(q, k, v, kv_cache_cfg=kv_cache_cfg, kv_state=kv_state, **kwargs)

    @torch.compiler.disable
    def _forward_list(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache_cfg: KVCacheConfig,
        kv_state: Optional[KVCacheLayerState] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, KVCacheLayerState]]:
        """List-based KV cache forward (original implementation).

        Decorated with ``@torch.compiler.disable`` to create a graph break.
        """
        if self.stateless:
            assert kv_state is not None, (
                "kv_state must be provided when running in stateless mode. "
                "Use AttentionOpWithKVCache.create_empty_state() to create initial state."
            )
            k_cache, v_cache = kv_state
        else:
            assert self.k_cache is not None and self.v_cache is not None, (
                "KV cache is not initialized. Call reset_kv_cache() first."
            )
            k_cache = self.k_cache
            v_cache = self.v_cache

        # Store into cache at start_idx location (list-based)
        if kv_cache_cfg.store_kv:
            index = int(kv_cache_cfg.current_idx)
            k_cache[index] = k.detach()
            v_cache[index] = v.detach()

        if kv_cache_cfg.run_with_kv and self.max_cache_size is not None:
            rolling_start_idx = max(0, int(kv_cache_cfg.current_idx) - self.max_cache_size)
        else:
            rolling_start_idx = 0

        # Prepend cached prefix up to start_idx (list-based)
        if kv_cache_cfg.run_with_kv and kv_cache_cfg.current_idx > 0:
            history_k = k_cache[rolling_start_idx : kv_cache_cfg.current_idx]
            history_v = v_cache[rolling_start_idx : kv_cache_cfg.current_idx]
            assert not any(x is None for x in history_k)
            assert not any(x is None for x in history_v)
            k_out = torch.cat(history_k + [k], dim=1)  # type: ignore
            v_out = torch.cat(history_v + [v], dim=1)  # type: ignore
        else:
            k_out = k
            v_out = v

        output = self.attn_op(q, k_out, v_out, **kwargs)

        if self.stateless:
            return output, (k_cache, v_cache)
        return output

    def _forward_tensor(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache_cfg: KVCacheConfig,
        kv_state: Optional[TensorKVCacheLayerState] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, TensorKVCacheLayerState]:
        """Tensor-based KV cache forward – fully traceable by ``torch.compile``.

        No ``@torch.compiler.disable`` decorator: all operations are pure
        tensor ops that ``torch._dynamo`` can trace end-to-end.  This
        eliminates graph breaks within each DiT block.

        The cache tensors are mutated **in-place** (``k_cache[idx] = k``)
        which is compatible with ``torch.compile`` in default mode.
        """
        assert kv_state is not None, (
            "kv_state must be provided when use_tensor_state=True. "
            "Use create_empty_tensor_state() or create_network_kv_tensor_state()."
        )
        k_cache, v_cache = kv_state  # Each: [max_frames, B, HW, H, D]

        idx = kv_cache_cfg.current_idx  # Python int – specialized by torch.compile

        # Store into tensor cache (in-place)
        if kv_cache_cfg.store_kv:
            k_cache[idx] = k
            v_cache[idx] = v

        # Read cached prefix + current frame
        if kv_cache_cfg.run_with_kv and idx > 0:
            if self.max_cache_size is not None:
                rolling_start = max(0, idx - self.max_cache_size)
            else:
                rolling_start = 0

            # Gather cached frames: [num_cached, B, HW, H, D]
            cached_k = k_cache[rolling_start:idx]
            cached_v = v_cache[rolling_start:idx]

            # Reshape to [B, num_cached*HW, H, D] for concatenation
            B = q.shape[0]
            cached_k = cached_k.transpose(0, 1).reshape(B, -1, q.shape[-2], q.shape[-1])
            cached_v = cached_v.transpose(0, 1).reshape(B, -1, v.shape[-2], v.shape[-1])

            k_out = torch.cat([cached_k, k], dim=1)
            v_out = torch.cat([cached_v, v], dim=1)
        else:
            k_out = k
            v_out = v

        output = self.attn_op(q, k_out, v_out, **kwargs)
        return output, (k_cache, v_cache)

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        self.attn_op.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)  # type: ignore


class VideoSeqPos:
    """Flattened 3D grid positions for a video clip.

    Stores flattened t/h/w indices of length L = T*H*W to enable constructing
    RoPE frequencies aligned with global positions across sequential chunks.

    Attributes:
        rope_grid_t, rope_grid_h, rope_grid_w:
            Plain Python ints that specify the *full* grid size needed to
            generate RoPE embeddings.  For a complete ``VideoSeqPos`` these
            equal ``T``, ``H``, ``W``.  For a single-frame view produced by
            :meth:`frame`, ``rope_grid_t`` equals ``t_idx + 1`` so that the
            RoPE table covers all positions up to the current frame.

            Using pre-computed ints avoids ``Tensor.item()`` calls in the
            forward path, which would otherwise cause graph breaks with
            ``torch.compile``.
    """

    def __init__(
        self,
        T: int,
        H: int,
        W: int,
        pos_h=None,
        pos_w=None,
        pos_t=None,
        rope_grid_t: int | None = None,
    ) -> None:
        self.T = T
        self.H = H
        self.W = W

        # RoPE grid dimensions – plain ints, safe for torch.compile.
        self.rope_grid_t: int = rope_grid_t if rope_grid_t is not None else T
        self.rope_grid_h: int = H
        self.rope_grid_w: int = W

        if pos_h is not None and pos_w is not None and pos_t is not None:
            self.pos_h = pos_h.to(dtype=torch.long)
            self.pos_w = pos_w.to(dtype=torch.long)
            self.pos_t = pos_t.to(dtype=torch.long)
            return

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        t = torch.arange(self.T, device=device, dtype=torch.long)
        h = torch.arange(self.H, device=device, dtype=torch.long)
        w = torch.arange(self.W, device=device, dtype=torch.long)
        pos_t, pos_h, pos_w = torch.meshgrid(t, h, w, indexing="ij")
        self.pos_t = pos_t.reshape(-1)
        self.pos_h = pos_h.reshape(-1)
        self.pos_w = pos_w.reshape(-1)

    def size(self) -> int:
        return int(self.pos_h.numel())

    def frame(self, t_idx: int) -> "VideoSeqPos":
        """Return a `VideoSeqPos` view for a single frame at absolute index `t_idx`.

        This is useful for streaming / KV-cache inference where the model is run on
        one frame at a time but RoPE positions must reflect global video indices.
        """
        t_idx = int(t_idx)
        if t_idx < 0 or t_idx >= int(self.T):
            raise IndexError(f"t_idx out of range: {t_idx} (valid: [0, {self.T}))")
        tokens_per_frame = int(self.H) * int(self.W)
        start = t_idx * tokens_per_frame
        end = start + tokens_per_frame
        return VideoSeqPos(
            T=1,
            H=int(self.H),
            W=int(self.W),
            pos_h=self.pos_h[start:end],
            pos_w=self.pos_w[start:end],
            pos_t=self.pos_t[start:end],
            rope_grid_t=t_idx + 1,
        )
