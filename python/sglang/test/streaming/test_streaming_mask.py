# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
import math

import pytest
import torch
from block_sparse_attn import block_sparse_attn_func
from einops import repeat
from utils import (
    MockModelRunner,
    attention_blocksparse_ref,
    convert_flash_attn_S_to_softmax,
    generate_base_sparsity_mask,
    generate_qkv,
    generate_random_padding_mask,
    generate_streaming_mask,
    get_dropout_fraction,
    normalize_flash_attn_S,
    prepare_mixed_exact_mask,
    prepare_mixed_mask,
)

from sglang.srt.layers.attention.mixed_triton_backend import MixedTritonAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

MAX_HEADDIM_SM8x = 192
block_size = 128
is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


@pytest.mark.parametrize(
    "dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16])
)
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize(
    "causal, sink_num, local_num",
    [
        (True, 1, 3),
        (True, 64, 256),
        (True, 1, 3),
        (False, 64, 256),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nheads", [16, 32])
def test_streaming_mask(
    seqlen_q,
    seqlen_k,
    d,
    causal,
    sink_num,
    local_num,
    mha_type,
    dtype,
    batch_size,
    nheads,
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    device = "cuda:0"
    # set seed
    torch.random.manual_seed(42)
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 8)
    assert nheads % nheads_k == 0
    window_size = (-1, -1)
    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch_size, device, mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        seqlen_k, batch_size, device, mode="random"
    )

    alibi_slopes, attn_bias = None, None
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    num_streaming_heads = nheads
    head_mask_type = torch.tensor(
        [-1] * num_streaming_heads, device=device, dtype=torch.int32
    )
    base_blockmask = generate_base_sparsity_mask(
        max_seqlen_q,
        max_seqlen_k,
        block_size,
        block_size,
        block_size,
        batch_size,
        0,
        [],
        causal=causal,
        device=device,
    )

    streaming_info = torch.tensor(
        [sink_num, local_num] * nheads, device=device, dtype=torch.int32
    )
    streaming_mask = generate_streaming_mask(
        max_seqlen_q,
        max_seqlen_k,
        batch_size,
        nheads,
        cu_seqlens_q,
        cu_seqlens_k,
        block_size,
        block_size,
        block_size,
        streaming_info,
        causal=causal,
        device=device,
    )

    mixed_mask = prepare_mixed_mask(
        base_blockmask,
        streaming_mask,
        head_mask_type,
        batch_size,
        nheads,
        block_size,
        block_size,
        block_size,
        max_seqlen_q,
        max_seqlen_k,
        q.shape[1],
        k.shape[1],
        device=device,
    )

    assert len(cu_seqlens_q) == len(cu_seqlens_k)
    streaming_mask_ref_list = []
    for bs in range(batch_size):
        # TODO: this loop process a sequence per iter, this is inefficient.
        # Need optimize the performance later.

        seq_len_q = cu_seqlens_q[bs + 1] - cu_seqlens_q[bs]
        seq_len_kv = cu_seqlens_k[bs + 1] - cu_seqlens_k[bs]

        streaming_mask_ref_list.append(
            mixed_mask[bs, 0][:seq_len_q, :seq_len_kv]
            .logical_not()
            .contiguous()
            .view(-1)
        )

    streaming_mask_ref = torch.cat(streaming_mask_ref_list, dim=0)

    streaming_mask = create_streaming_mask(
        cu_seqlens_q,
        cu_seqlens_k,
        sink_num * 128,
        local_num * 128,
        causal=causal,
        block_dim=block_size,
        device=device,
    )

    assert torch.equal(streaming_mask, streaming_mask_ref)


def create_streaming_mask(
    cu_seqlens_q,
    cu_seqlens_k,
    sink_window_size,
    local_window_size,
    causal=True,
    block_dim=128,
    device=None,
):

    assert len(cu_seqlens_q) == len(cu_seqlens_k)
    batch_size = len(cu_seqlens_q) - 1
    qo_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    kv_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

    mask_list = []

    sink_block_num, local_block_num = (
        (sink_window_size + block_dim - 1) // block_dim,
        (local_window_size + block_dim - 1) // block_dim,
    )

    for bs in range(batch_size):
        q_len = qo_lens[bs].item()
        k_len = kv_lens[bs].item()
        nrow = (q_len + block_dim - 1) // block_dim
        ncol = (k_len + block_dim - 1) // block_dim

        start_row_idx = max((q_len - k_len) // block_dim, 0) if causal else 0
        mask = torch.zeros(nrow, ncol, device=device, dtype=torch.bool)
        for i in range(start_row_idx, nrow):
            if causal:
                max_row_block_num = (
                    (max(k_len - q_len, 0) + block_dim - 1) // block_dim
                    + 1
                    + i
                    - start_row_idx
                )
            else:
                max_row_block_num = ncol

            mask[
                i,
                min(max(max_row_block_num - local_block_num, 0), ncol) : min(
                    max_row_block_num, ncol
                ),
            ] = True
            mask[i, :sink_block_num] = True

        mask_list.append(
            repeat(
                mask, "s_m s_n -> (s_m d_m) (s_n d_n)", d_m=block_dim, d_n=block_dim
            )[:q_len, :k_len]
            .contiguous()
            .view(-1)
        )

    return torch.cat(mask_list, dim=0)
