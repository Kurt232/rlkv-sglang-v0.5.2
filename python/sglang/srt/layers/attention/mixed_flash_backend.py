from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from block_sparse_attn import block_streaming_attn_func
from flash_attn import flash_attn_varlen_func


@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)

    # streaming attention metadata
    streaming_mask: torch.Tensor = None
    head_mask_type: torch.Tensor = None


class MixedFlashAttnBackend(AttentionBackend):
    """MixedFlashAttn backend implementation."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()

        assert (
            not model_runner.model_config.is_encoder_decoder
        ), "Cross attention is not supported"

        self.sink_window_size = model_runner.server_args.sink_window_size
        self.recent_window_size = model_runner.server_args.recent_window_size

        self.forward_metadata: FlashAttentionMetadata = None
        # Not support Speculative Decoding
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.skip_prefill = skip_prefill

        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.num_kv_groups = self.num_head // self.num_kv_head

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # Normal Decode
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
        elif forward_batch.forward_mode.is_extend():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )

            metadata.max_seq_len_q = metadata.max_seq_len_k
            metadata.cu_seqlens_q = metadata.cu_seqlens_k

        # For block streaming attention
        num_sink_blocks = (self.sink_window_size + 127) // 128
        num_recent_blocks = (self.sink_window_size + 127) // 128
        streaming_mask = torch.tensor(
            [num_sink_blocks, num_recent_blocks] * self.num_head,
            device=device,
            dtype=torch.int32,
        )
        metadata.streaming_mask = streaming_mask
        metadata.head_mask_type = torch.tensor(
            [-1] * self.num_head, device=device, dtype=torch.int32
        )
        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        adapter: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = forward_batch.out_cache_loc
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (-1, -1)
        causal = True

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seq_len_q
        max_seqlen_k = metadata.max_seq_len_k
        cu_seqlens_k = metadata.cu_seqlens_k

        # Use Flash Attention for prefill
        # Do multi-head attention
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(-1, layer.tp_k_head_num, layer.head_dim)
        value_cache = value_cache.view(-1, layer.tp_v_head_num, layer.head_dim)

        o = flash_attn_varlen_func(
            q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=window_size,
            softcap=layer.logit_cap,
        )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        adapter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = forward_batch.out_cache_loc
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (-1, -1)
        causal = True

        # Do multi-head attention
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(-1, layer.tp_k_head_num, layer.head_dim)
        value_cache = value_cache.view(-1, layer.tp_v_head_num, layer.head_dim)

        cache_seqlens = metadata.cache_seqlens_int32
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        max_seqlen_q = metadata.max_seq_len_q
        max_seqlen_k = metadata.max_seq_len_k
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        # Default: single-token self-attention
        o = flash_attn_varlen_func(
            q=q_reshaped,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=window_size,
            softcap=layer.logit_cap,
        )
        o = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        o_streaming = block_streaming_attn_func(
            q=q_reshaped,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q_=max_seqlen_q,
            max_seqlen_k_=max_seqlen_k,
            softmax_scale=layer.scaling,
            p_dropout=0.0,
            is_causal=True,
            head_mask_type=metadata.head_mask_type,
            streaming_info=metadata.streaming_mask,
        )

        adapter = adapter.repeat_interleave(self.num_kv_groups).view(1, -1, 1)

        o = o * adapter + o_streaming * (1.0 - adapter)

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        # This is being used by normal decode and draft decode when topk == 1
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        """Initialize forward metadata for capturing CUDA graph."""
        metadata = FlashAttentionMetadata()

        if spec_info is not None:
            raise ValueError(
                f"Speculative decoding is not supported with MixedTritonAttnBackend."
            )

        device = seq_lens.device
        if forward_mode.is_decode_or_idle():
            # Normal Decode
            # Get sequence information
            metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
            batch_size = len(seq_lens)
            device = seq_lens.device
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            # Precompute maximum sequence length
            metadata.max_seq_len_k = seq_lens.max().item()
            # Precompute cumulative sequence lengths
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            self.decode_cuda_graph_metadata[bs] = metadata

        # For block streaming attention
        num_sink_blocks = (self.sink_window_size + 127) // 128
        num_recent_blocks = (self.sink_window_size + 127) // 128
        streaming_mask = torch.tensor(
            [num_sink_blocks, num_recent_blocks] * self.num_head,
            device=device,
            dtype=torch.int32,
        )
        metadata.streaming_mask = streaming_mask
        metadata.head_mask_type = torch.tensor(
            [-1] * self.num_head, device=device, dtype=torch.int32
        )

        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]
        metadata = None

        if spec_info is not None:
            raise ValueError(
                f"Speculative decoding is not supported with MixedTritonAttnBackend."
            )

        if forward_mode.is_decode_or_idle():
            # Normal Decode
            metadata = self.decode_cuda_graph_metadata[bs]
            max_len = seq_lens_cpu.max().item()
            metadata.max_seq_len_k = max_len

            normal_decode_set_medadata(
                metadata.cache_seqlens_int32,
                metadata.cu_seqlens_k,
                seq_lens,
                0,
            )

        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1


# @torch.compile(dynamic=True, backend=get_compiler_backend())
# TODO: fuse these kernels
# NOTE: torch.compile makes it slower in speculative decoding
def normal_decode_set_medadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len_delta: int,
):
    cache_seqlens_int32.copy_(seq_lens + seq_len_delta)
    cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32))
