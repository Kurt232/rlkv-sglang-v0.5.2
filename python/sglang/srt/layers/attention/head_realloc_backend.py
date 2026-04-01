"""
Head Reallocation Attention Backend.

Implements head-level KV cache reallocation: full-attention heads retain
complete KV cache, while compressed heads use only sink + recent window.

Supports two modes:
- Single pool (V1): All heads share one KV pool, head indexing at read time.
- Dual pool (V2): Separate full/comp pools via HeadReallocKVPool, no head
  indexing at read time. set_kv_buffer splits heads automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.attention.mixed_triton_backend import (
    update_streaming_window_buffer,
    update_streaming_window_buffer_cuda_graph,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import HeadReallocKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import get_device_core_count, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class HeadReallocForwardMetadata:
    # Full attention metadata
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    # Streaming window metadata (for compressed heads)
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    # Comp pool kv indices (for dual pool extend)
    comp_kv_indices: torch.Tensor
    # Extend metadata
    qo_indptr: torch.Tensor
    max_extend_len: int


class HeadReallocAttnBackend(AttentionBackend):
    """
    Attention backend for RLKV inference.

    Per layer, heads are statically classified as full or compressed.
    Full heads use standard full-context attention.
    Compressed heads use streaming attention (sink + recent window).

    Supports both single-pool (V1) and dual-pool (V2) modes.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        head_masks: Dict[int, torch.Tensor],
    ):
        """
        Args:
            model_runner: The model runner instance.
            head_masks: Dict mapping layer_id -> binary tensor of shape [num_kv_heads].
                        1 = full attention head, 0 = compressed head.
        """
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        max_bs = model_runner.req_to_token_pool.size

        self.sink_window_size = model_runner.server_args.sink_window_size
        self.local_window_size = model_runner.server_args.recent_window_size
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.num_kv_groups = self.num_head // self.num_kv_head

        # Detect dual pool mode
        self.use_dual_pool = isinstance(
            model_runner.token_to_kv_pool, HeadReallocKVPool
        )

        # Per-layer head masks and index tensors
        self.head_masks = head_masks
        self._precompute_head_indices(model_runner.device)

        # Buffers for kv indexing
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        self.window_kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )

        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        self.max_context_len = model_runner.model_config.context_len

        self.device = model_runner.device
        self.device_core_count = get_device_core_count(model_runner.gpu_id)

        self.forward_metadata: HeadReallocForwardMetadata = None

    def _precompute_head_indices(self, device: str):
        """Precompute full/compressed head indices and Q-head expansion for each layer."""
        self.full_kv_head_indices: Dict[int, torch.Tensor] = {}
        self.comp_kv_head_indices: Dict[int, torch.Tensor] = {}
        # Q-head indices (expanded from KV head indices via GQA groups)
        self.full_q_head_indices: Dict[int, torch.Tensor] = {}
        self.comp_q_head_indices: Dict[int, torch.Tensor] = {}
        # For restoring original head order after split attention
        self.restore_indices: Dict[int, torch.Tensor] = {}

        for layer_id, mask in self.head_masks.items():
            # KV head indices
            full_kv = torch.where(mask == 1)[0].to(device)
            comp_kv = torch.where(mask == 0)[0].to(device)
            self.full_kv_head_indices[layer_id] = full_kv
            self.comp_kv_head_indices[layer_id] = comp_kv

            # Expand to Q head indices (each KV head maps to num_kv_groups Q heads)
            full_q = torch.cat([
                torch.arange(
                    kv_idx * self.num_kv_groups,
                    (kv_idx + 1) * self.num_kv_groups,
                    device=device,
                )
                for kv_idx in full_kv
            ]) if len(full_kv) > 0 else torch.tensor([], dtype=torch.long, device=device)

            comp_q = torch.cat([
                torch.arange(
                    kv_idx * self.num_kv_groups,
                    (kv_idx + 1) * self.num_kv_groups,
                    device=device,
                )
                for kv_idx in comp_kv
            ]) if len(comp_kv) > 0 else torch.tensor([], dtype=torch.long, device=device)

            self.full_q_head_indices[layer_id] = full_q.long()
            self.comp_q_head_indices[layer_id] = comp_q.long()

            # Restore index: maps [full_q..., comp_q...] back to original order
            combined = torch.cat([full_q, comp_q])
            restore = torch.empty_like(combined)
            restore[combined] = torch.arange(len(combined), device=device)
            self.restore_indices[layer_id] = restore.long()

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ):
        """Compute optimal number of KV splits for decode attention."""
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        num_group = num_token // num_seq

        if self.device_core_count <= 0:
            num_kv_splits.fill_(self.max_kv_splits)
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        _get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            num_heads,
            num_kv_heads,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        kv_pool = forward_batch.token_to_kv_pool

        if forward_batch.forward_mode.is_decode_or_idle():
            # Full attention indices (pointing to full pool locations)
            kv_indptr = self.kv_indptr
            kv_indptr[1: bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            # Streaming window indices (for compressed heads)
            window_kv_indptr, window_kv_indices, window_kv_lens = (
                update_streaming_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sink_window_size,
                    self.local_window_size,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                )
            )

            # For dual pool: translate window indices to comp pool locations
            if self.use_dual_pool:
                window_kv_indices = kv_pool.translate_loc_full_to_comp(
                    window_kv_indices
                )

            # Allocate logits/lse buffers (using full num_head for max size)
            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )

            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            self.get_num_kv_splits(
                num_kv_splits, forward_batch.seq_lens,
                self.num_head, self.num_kv_head,
            )

            window_num_kv_splits = torch.empty(
                (bs,), dtype=torch.int32, device=self.device
            )
            self.get_num_kv_splits(
                window_num_kv_splits, window_kv_lens,
                self.num_head, self.num_kv_head,
            )

            qo_indptr = None
            max_extend_len = None
            comp_kv_indices = None

        elif forward_batch.forward_mode.is_extend():
            # Extend mode (prefill)
            kv_indptr = self.kv_indptr
            kv_indptr[1: bs + 1] = torch.cumsum(
                forward_batch.extend_prefix_lens, dim=0
            )
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                forward_batch.extend_prefix_lens.sum().item(),
                dtype=torch.int32,
                device=self.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.extend_prefix_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            # For dual pool: translate prefix indices to comp pool locations
            comp_kv_indices = None
            if self.use_dual_pool and kv_indices.numel() > 0:
                comp_kv_indices = kv_pool.translate_loc_full_to_comp(kv_indices)

            qo_indptr = self.qo_indptr
            qo_indptr[1: bs + 1] = torch.cumsum(
                forward_batch.extend_seq_lens, dim=0
            )
            qo_indptr = qo_indptr[: bs + 1]

            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()

            attn_logits = None
            attn_lse = None
            num_kv_splits = None
            window_kv_indptr = None
            window_kv_indices = None
            window_num_kv_splits = None
        else:
            raise ValueError(
                f"HeadReallocAttnBackend does not support forward mode: "
                f"{forward_batch.forward_mode}"
            )

        self.forward_metadata = HeadReallocForwardMetadata(
            attn_logits=attn_logits,
            attn_lse=attn_lse,
            num_kv_splits=num_kv_splits,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            window_kv_indptr=window_kv_indptr,
            window_kv_indices=window_kv_indices,
            window_num_kv_splits=window_num_kv_splits,
            comp_kv_indices=comp_kv_indices,
            qo_indptr=qo_indptr,
            max_extend_len=max_extend_len,
        )

    def _get_kv_buffers(self, kv_pool, layer_id, group: str):
        """Get K/V buffers for a head group.

        In dual pool mode, buffers already contain only the relevant heads.
        In single pool mode, we index-select from the shared buffer.
        """
        if self.use_dual_pool:
            if group == "full":
                return kv_pool.get_key_buffer(layer_id), kv_pool.get_value_buffer(layer_id)
            else:
                return kv_pool.get_comp_key_buffer(layer_id), kv_pool.get_comp_value_buffer(layer_id)
        else:
            # Single pool: index select by head
            k_all = kv_pool.get_key_buffer(layer_id)
            v_all = kv_pool.get_value_buffer(layer_id)
            if group == "full":
                idx = self.full_kv_head_indices[layer_id]
            else:
                idx = self.comp_kv_head_indices[layer_id]
            return k_all[:, idx, :].contiguous(), v_all[:, idx, :].contiguous()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        bs = q.shape[0]

        # Save KV cache (HeadReallocKVPool.set_kv_buffer splits heads internally)
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        layer_id = layer.layer_id
        full_q_idx = self.full_q_head_indices[layer_id]
        comp_q_idx = self.comp_q_head_indices[layer_id]

        num_full_q = len(full_q_idx)
        num_comp_q = len(comp_q_idx)

        q_3d = q.view(bs, layer.tp_q_head_num, layer.qk_head_dim)

        # Output buffer
        o_full = q.new_empty((bs, layer.tp_q_head_num, layer.v_head_dim))

        # --- Full attention heads ---
        if num_full_q > 0:
            q_full = q_3d[:, full_q_idx, :].contiguous()
            k_buf, v_buf = self._get_kv_buffers(
                forward_batch.token_to_kv_pool, layer_id, "full"
            )

            o_part = torch.empty(
                (bs, num_full_q, layer.v_head_dim),
                dtype=q.dtype, device=q.device,
            )
            attn_logits = torch.empty(
                (bs, num_full_q, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32, device=self.device,
            )
            attn_lse = torch.empty(
                (bs, num_full_q, self.max_kv_splits),
                dtype=torch.float32, device=self.device,
            )
            num_kv_splits = self.forward_metadata.num_kv_splits

            self.decode_attention_fwd(
                q_full,
                k_buf,
                v_buf,
                o_part,
                self.forward_metadata.kv_indptr,
                self.forward_metadata.kv_indices,
                attn_logits,
                attn_lse,
                num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                layer.logit_cap,
            )
            o_full[:, full_q_idx, :] = o_part

        # --- Compressed attention heads (streaming window) ---
        if num_comp_q > 0:
            q_comp = q_3d[:, comp_q_idx, :].contiguous()
            k_buf, v_buf = self._get_kv_buffers(
                forward_batch.token_to_kv_pool, layer_id, "comp"
            )

            o_part = torch.empty(
                (bs, num_comp_q, layer.v_head_dim),
                dtype=q.dtype, device=q.device,
            )
            attn_logits = torch.empty(
                (bs, num_comp_q, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32, device=self.device,
            )
            attn_lse = torch.empty(
                (bs, num_comp_q, self.max_kv_splits),
                dtype=torch.float32, device=self.device,
            )
            window_num_kv_splits = self.forward_metadata.window_num_kv_splits

            self.decode_attention_fwd(
                q_comp,
                k_buf,
                v_buf,
                o_part,
                self.forward_metadata.window_kv_indptr,
                self.forward_metadata.window_kv_indices,
                attn_logits,
                attn_lse,
                window_num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                layer.logit_cap,
            )
            o_full[:, comp_q_idx, :] = o_part

        return o_full.view(bs, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        num_tokens = q.shape[0]

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        layer_id = layer.layer_id
        full_q_idx = self.full_q_head_indices[layer_id]
        comp_q_idx = self.comp_q_head_indices[layer_id]

        num_full_q = len(full_q_idx)
        num_comp_q = len(comp_q_idx)

        full_kv_idx = self.full_kv_head_indices[layer_id]
        comp_kv_idx = self.comp_kv_head_indices[layer_id]

        q_3d = q.view(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)
        k_3d = k.contiguous()
        v_3d = v.contiguous()

        o_full = q.new_empty((num_tokens, layer.tp_q_head_num, layer.v_head_dim))

        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices
        qo_indptr = self.forward_metadata.qo_indptr

        # --- Full attention heads: causal, no mask ---
        if num_full_q > 0:
            q_part = q_3d[:, full_q_idx, :].contiguous()
            k_part = k_3d[:, full_kv_idx, :].contiguous()
            v_part = v_3d[:, full_kv_idx, :].contiguous()
            k_buf, v_buf = self._get_kv_buffers(
                forward_batch.token_to_kv_pool, layer_id, "full"
            )

            o_part = torch.empty(
                (num_tokens, num_full_q, layer.v_head_dim),
                dtype=q.dtype, device=q.device,
            )
            self.extend_attention_fwd(
                q_part,
                k_part,
                v_part,
                o_part,
                k_buf,
                v_buf,
                qo_indptr,
                kv_indptr,
                kv_indices,
                None,  # custom_mask
                True,  # causal
                None,  # mask_indptr
                self.forward_metadata.max_extend_len,
                layer.scaling,
                layer.logit_cap,
                False,
                -1,  # sliding_window_size
            )
            o_full[:, full_q_idx, :] = o_part

        # --- Compressed heads: causal, full attention during prefill ---
        if num_comp_q > 0:
            q_part = q_3d[:, comp_q_idx, :].contiguous()
            k_part = k_3d[:, comp_kv_idx, :].contiguous()
            v_part = v_3d[:, comp_kv_idx, :].contiguous()
            k_buf, v_buf = self._get_kv_buffers(
                forward_batch.token_to_kv_pool, layer_id, "comp"
            )

            # For dual pool, use translated comp indices for prefix buffer
            extend_kv_indices = kv_indices
            if self.use_dual_pool and self.forward_metadata.comp_kv_indices is not None:
                extend_kv_indices = self.forward_metadata.comp_kv_indices

            o_part = torch.empty(
                (num_tokens, num_comp_q, layer.v_head_dim),
                dtype=q.dtype, device=q.device,
            )
            self.extend_attention_fwd(
                q_part,
                k_part,
                v_part,
                o_part,
                k_buf,
                v_buf,
                qo_indptr,
                kv_indptr,
                extend_kv_indices,
                None,  # custom_mask
                True,  # causal
                None,  # mask_indptr
                self.forward_metadata.max_extend_len,
                layer.scaling,
                layer.logit_cap,
                False,
                -1,  # sliding_window_size
            )
            o_full[:, comp_q_idx, :] = o_part

        return o_full.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)


# Reuse the triton kernel from mixed_triton_backend for kv_splits calculation
@triton.jit
def _get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)
