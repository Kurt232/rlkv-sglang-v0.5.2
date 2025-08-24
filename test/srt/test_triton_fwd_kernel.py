import unittest

import torch

# Import the unified attention function to be tested
from sglang.srt.layers.attention.triton_ops.attention import attention_fwd
from sglang.test.test_utils import CustomTestCase


class TestUnifiedAttention(CustomTestCase):
    """Test suite for the unified attention kernel."""

    def _test_unified_attention_once(
        self,
        batch_size: int,
        seq_lens_prefix: list,
        seq_lens_q: list,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        is_causal: bool,
        sliding_window_size: int,
        dtype: torch.dtype = torch.float16,
        seed: int = 42,
    ):
        """A generic function to test the attention_fwd kernel against a torch reference implementation."""
        torch.manual_seed(seed)

        # 1. Prepare inputs for the Triton kernel
        seq_lens_kv = [p + q for p, q in zip(seq_lens_prefix, seq_lens_q)]
        max_len_q = max(seq_lens_q)

        # Create Q, K, V tensors for the whole sequences for later reference
        all_q_list = [
            torch.randn(s, num_heads_q, head_dim, dtype=dtype, device="cuda")
            for s in seq_lens_kv
        ]
        all_k_list = [
            torch.randn(s, num_heads_kv, head_dim, dtype=dtype, device="cuda")
            for s in seq_lens_kv
        ]
        all_v_list = [
            torch.randn(s, num_heads_kv, head_dim, dtype=dtype, device="cuda")
            for s in seq_lens_kv
        ]

        # Create contiguous buffers for k and v, which contain all tokens (prefix + new)
        k_buffer = torch.cat(all_k_list, dim=0)
        v_buffer = torch.cat(all_v_list, dim=0)

        # The 'q' tensor for our function only contains the queries for the new tokens
        q_to_compute_list = []
        for i in range(batch_size):
            prefix_len = seq_lens_prefix[i]
            q_len = seq_lens_q[i]
            q_to_compute_list.append(all_q_list[i][prefix_len : prefix_len + q_len])
        q = torch.cat(q_to_compute_list, dim=0)

        o_triton = torch.empty_like(q)

        # Create indptrs and indices for the kernel
        qo_indptr = torch.cumsum(
            torch.tensor([0] + seq_lens_q, device="cuda"), dim=0, dtype=torch.int32
        )
        kv_indptr = torch.cumsum(
            torch.tensor([0] + seq_lens_kv, device="cuda"), dim=0, dtype=torch.int32
        )
        kv_indices = torch.arange(
            kv_indptr[-1].item(), dtype=torch.int32, device="cuda"
        )

        # 2. Call the function to be tested
        attention_fwd(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            o=o_triton,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            is_causal=is_causal,
            max_len_q=max_len_q,
            sliding_window_size=sliding_window_size,
        )

        # 3. Calculate the reference output using torch's native scaled_dot_product_attention
        o_torch_list = []
        for i in range(batch_size):
            q_full = all_q_list[i].permute(1, 0, 2)  # H, L, D
            k_full = all_k_list[i].permute(1, 0, 2)
            v_full = all_v_list[i].permute(1, 0, 2)

            # Repeat K,V heads for GQA/MQA
            if num_heads_q != num_heads_kv:
                k_full = k_full.repeat(num_heads_q // num_heads_kv, 1, 1)
                v_full = v_full.repeat(num_heads_q // num_heads_kv, 1, 1)

            # Create attention mask for torch. This is the most complex part.
            seq_len_kv = seq_lens_kv[i]
            seq_len_q = seq_lens_q[i]
            prefix_len = seq_lens_prefix[i]

            # Create a mask of shape (L_q, L_kv)
            q_indices = torch.arange(seq_len_q, device="cuda") + prefix_len
            k_indices = torch.arange(seq_len_kv, device="cuda")

            mask = torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device="cuda")
            if is_causal:
                mask &= q_indices[:, None] >= k_indices[None, :]
            if sliding_window_size > 0:
                mask &= (q_indices[:, None] - k_indices[None, :]) < sliding_window_size

            attn_mask = torch.where(mask, 0.0, -torch.inf)

            # We only need to compute attention for the new queries
            q_to_compute_torch = q_full[:, prefix_len : prefix_len + seq_len_q, :]

            output_full = torch.nn.functional.scaled_dot_product_attention(
                q_to_compute_torch, k_full, v_full, attn_mask=attn_mask
            ).permute(
                1, 0, 2
            )  # L_q, H, D

            o_torch_list.append(output_full)

        o_torch = torch.cat(o_torch_list, dim=0)

        # 4. Compare Triton kernel output with the torch reference
        self.assertTrue(
            torch.allclose(o_triton, o_torch, atol=1e-2, rtol=1e-2),
            f"mean absolute error: {torch.mean(torch.abs(o_triton - o_torch))}, "
            f"std deviation: {torch.std(torch.abs(o_triton - o_torch))}",
        )

    def test_attention_prefill_hma(self):
        """Test case: pure prefill (no prefix)."""
        self._test_unified_attention_once(
            batch_size=2,
            seq_lens_prefix=[0, 0],
            seq_lens_q=[16, 24],
            num_heads_q=8,
            num_heads_kv=8,
            head_dim=64,
            is_causal=True,
            sliding_window_size=-1,
        )

    def test_attention_decode_hma(self):
        """Test case: decode (query length is 1)."""
        self._test_unified_attention_once(
            batch_size=4,
            seq_lens_prefix=[10, 20, 30, 40],
            seq_lens_q=[1, 1, 1, 1],
            num_heads_q=8,
            num_heads_kv=8,
            head_dim=80,
            is_causal=True,
            sliding_window_size=-1,
        )

    def test_attention_extend_hma(self):
        """Test case: extend (has prefix, query length > 1)."""
        self._test_unified_attention_once(
            batch_size=2,
            seq_lens_prefix=[32, 48],
            seq_lens_q=[16, 24],
            num_heads_q=8,
            num_heads_kv=8,
            head_dim=128,
            is_causal=True,
            sliding_window_size=-1,
        )

    def test_attention_sliding_window_hma(self):
        """Test case: sliding window attention."""
        self._test_unified_attention_once(
            batch_size=2,
            seq_lens_prefix=[100, 120],
            seq_lens_q=[16, 8],
            num_heads_q=8,
            num_heads_kv=8,
            head_dim=64,
            is_causal=True,
            sliding_window_size=64,
        )

    def test_attention_mixed_batch_hma(self):
        """Test case: a mixed batch of prefill, decode, and extend."""
        self._test_unified_attention_once(
            batch_size=3,
            seq_lens_prefix=[0, 64, 128],  # Prefill, Extend, Decode
            seq_lens_q=[48, 32, 1],
            num_heads_q=8,
            num_heads_kv=8,
            head_dim=64,
            is_causal=True,
            sliding_window_size=-1,
        )

    def test_attention_prefill_gpa(self):
        """Test case: pure prefill (no prefix)."""
        self._test_unified_attention_once(
            batch_size=2,
            seq_lens_prefix=[0, 0],
            seq_lens_q=[16, 24],
            num_heads_q=8,
            num_heads_kv=8,
            head_dim=64,
            is_causal=True,
            sliding_window_size=-1,
        )

    def test_attention_decode_gqa(self):
        """Test case: decode (query length is 1)."""
        self._test_unified_attention_once(
            batch_size=4,
            seq_lens_prefix=[10, 20, 30, 40],
            seq_lens_q=[1, 1, 1, 1],
            num_heads_q=16,
            num_heads_kv=8,
            head_dim=80,
            is_causal=True,
            sliding_window_size=-1,
        )

    def test_attention_extend_gqa(self):
        """Test case: extend (has prefix, query length > 1)."""
        self._test_unified_attention_once(
            batch_size=2,
            seq_lens_prefix=[32, 48],
            seq_lens_q=[16, 24],
            num_heads_q=16,
            num_heads_kv=8,
            head_dim=128,
            is_causal=True,
            sliding_window_size=-1,
        )

    def test_attention_sliding_window_gqa(self):
        """Test case: sliding window attention."""
        self._test_unified_attention_once(
            batch_size=2,
            seq_lens_prefix=[100, 120],
            seq_lens_q=[16, 8],
            num_heads_q=16,
            num_heads_kv=8,
            head_dim=64,
            is_causal=True,
            sliding_window_size=64,
        )

    def test_attention_mixed_batch_gqa(self):
        """Test case: a mixed batch of prefill, decode, and extend."""
        self._test_unified_attention_once(
            batch_size=3,
            seq_lens_prefix=[0, 64, 128],  # Prefill, Extend, Decode
            seq_lens_q=[48, 32, 1],
            num_heads_q=16,
            num_heads_kv=8,
            head_dim=64,
            is_causal=True,
            sliding_window_size=-1,
        )


if __name__ == "__main__":
    unittest.main()
