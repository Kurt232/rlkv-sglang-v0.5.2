"""
Unit tests for Head Reallocation Attention Backend with Dual KV Pool.

Tests that head-level split attention with dual pool produces correct results
by comparing full heads' output against standard full attention and verifying
compressed heads produce valid output.

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m pytest test_head_realloc_backend.py -v
"""

import unittest
from unittest import mock

import torch
import torch.nn as nn

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.head_realloc_backend import HeadReallocAttnBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import HeadReallocAllocator
from sglang.srt.mem_cache.memory_pool import HeadReallocKVPool, MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ServerArgs


class MockModelRunner:
    """Mock model runner for unit testing with dual KV pool."""

    def __init__(
        self,
        num_heads=8,
        head_dim=128,
        sink_window_size=128,
        recent_window_size=256,
        head_masks=None,
    ):
        self.device = "cuda"
        self.dtype = torch.float16
        self.gpu_id = 0
        attention_arch = AttentionArch.MHA
        max_batch_size = 32
        max_context_len = 2048
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": max_context_len,
                "is_multimodal": False,
                "attention_arch": attention_arch,
                "is_encoder_decoder": False,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_heads,
                "get_num_kv_heads": lambda x: num_heads,
            },
        )
        self.sliding_window_size = None
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": max_batch_size,
                "req_to_token": torch.zeros(
                    max_batch_size,
                    max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            },
        )
        max_total_num_tokens = max_batch_size * max_context_len
        self.server_args = ServerArgs(
            model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            enable_rlkv_inference=True,
            sink_window_size=sink_window_size,
            recent_window_size=recent_window_size,
        )
        self.kv_cache_dtype = self.dtype

        if head_masks is not None:
            # Dual pool mode
            self.token_to_kv_pool = HeadReallocKVPool(
                size_full=max_total_num_tokens,
                size_comp=max_total_num_tokens,
                head_masks=head_masks,
                head_dim=head_dim,
                layer_num=1,
                dtype=self.dtype,
                device=self.device,
                enable_memory_saver=False,
            )
            # Create allocator and set up mapping
            self.token_to_kv_pool_allocator = HeadReallocAllocator(
                size_full=max_total_num_tokens,
                size_comp=max_total_num_tokens,
                dtype=self.dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
                need_sort=False,
            )
        else:
            # Single pool mode (for reference backend)
            self.token_to_kv_pool = MHATokenToKVPool(
                size=max_total_num_tokens,
                page_size=1,
                dtype=self.dtype,
                head_num=num_heads,
                head_dim=head_dim,
                layer_num=1,
                device=self.device,
                enable_memory_saver=False,
            )
            self.token_to_kv_pool_allocator = None


def _mock_write_req_to_token(model_runner, batch_size, seq_len):
    """Write sequential token indices to req_to_token_pool."""
    req_to_token = (
        torch.arange(0, batch_size, dtype=torch.int32, device="cuda")[:, None]
        * seq_len
        + torch.arange(0, seq_len, dtype=torch.int32, device="cuda")[None, :]
    )
    model_runner.req_to_token_pool.req_to_token[:batch_size, :seq_len] = req_to_token


def _setup_kv_cache_dual_pool(model_runner, layer, batch_size, cache_len):
    """Write random KV cache for dual pool testing.

    Allocates through the HeadReallocAllocator to set up proper
    full_to_comp_mapping, then writes KV via set_kv_buffer which
    splits heads internally.
    """
    dtype = model_runner.dtype
    device = model_runner.device
    num_heads = model_runner.model_config.num_attention_heads
    head_dim = 128
    total_tokens = batch_size * cache_len

    # Allocate through the allocator to set up full_to_comp_mapping
    alloc = model_runner.token_to_kv_pool_allocator
    loc = alloc.alloc(total_tokens)
    assert loc is not None, "Failed to allocate tokens"

    # Create KV data with all heads
    cache_k = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    cache_v = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)

    # set_kv_buffer splits heads internally
    model_runner.token_to_kv_pool.set_kv_buffer(layer, loc, cache_k, cache_v)

    return cache_k, cache_v, loc


def _setup_kv_cache_single_pool(model_runner, layer, batch_size, cache_len):
    """Write random KV cache for single pool testing (reference backend)."""
    dtype = model_runner.dtype
    device = model_runner.device
    num_heads = model_runner.model_config.num_attention_heads
    head_dim = 128

    cache_k = torch.randn(
        batch_size * cache_len, num_heads, head_dim,
        dtype=dtype, device=device,
    )
    cache_v = torch.randn(
        batch_size * cache_len, num_heads, head_dim,
        dtype=dtype, device=device,
    )
    loc = torch.arange(batch_size * cache_len, device=device)
    model_runner.token_to_kv_pool.set_kv_buffer(layer, loc, cache_k, cache_v)

    return cache_k, cache_v, loc


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestHeadReallocDualPoolDecode(unittest.TestCase):
    """Test decode with dual pool RLKV backend vs reference (TorchNative full attention)."""

    @classmethod
    def setUpClass(cls):
        cls._patcher = mock.patch(
            "sglang.srt.layers.attention.head_realloc_backend.get_attention_tp_size",
            return_value=1,
        )
        cls.mock_tp_size = cls._patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 512
        self.num_heads = 8
        self.head_dim = 128
        self.device = "cuda"
        self.dtype = torch.float16

    def test_all_full_heads_dual_pool_matches_reference(self):
        """When all heads are full, dual pool output should match full attention."""
        # All heads are full (mask = 1)
        head_masks = {0: torch.ones(self.num_heads, device=self.device)}

        model_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
            head_masks=head_masks,
        )
        ref_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
            head_masks=None,  # single pool for reference
        )

        backend = HeadReallocAttnBackend(model_runner, head_masks)
        ref_backend = TorchNativeAttnBackend(ref_runner)

        layer = RadixAttention(
            num_heads=self.num_heads, head_dim=self.head_dim,
            scaling=1.0, num_kv_heads=self.num_heads, layer_id=0,
        )

        # Setup KV cache (same random data)
        torch.manual_seed(123)
        cache_k, cache_v, _ = _setup_kv_cache_dual_pool(
            model_runner, layer, self.batch_size, self.seq_len
        )
        # Write same data to reference pool
        ref_loc = torch.arange(self.batch_size * self.seq_len, device=self.device)
        ref_runner.token_to_kv_pool.set_kv_buffer(layer, ref_loc, cache_k, cache_v)

        _mock_write_req_to_token(model_runner, self.batch_size, self.seq_len + 1)
        _mock_write_req_to_token(ref_runner, self.batch_size, self.seq_len + 1)

        q = torch.randn(
            self.batch_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        k = torch.randn(
            self.batch_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        v = torch.randn(
            self.batch_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )

        total_len = self.seq_len + 1
        out_cache_start = self.batch_size * self.seq_len
        out_cache_end = self.batch_size * total_len

        # Allocate out_cache_loc through the allocator for dual pool
        out_cache_loc = model_runner.token_to_kv_pool_allocator.alloc(self.batch_size)
        ref_out_cache_loc = torch.arange(out_cache_start, out_cache_end, device=self.device)

        fb_kwargs = dict(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 100, (self.batch_size, 1), device=self.device),
            seq_lens_sum=self.batch_size * total_len,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=torch.tensor([total_len] * self.batch_size, device=self.device),
            seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
        )

        fb = ForwardBatch(**fb_kwargs, out_cache_loc=out_cache_loc, attn_backend=backend)
        fb.req_to_token_pool = model_runner.req_to_token_pool
        fb.token_to_kv_pool = model_runner.token_to_kv_pool

        fb_ref = ForwardBatch(**fb_kwargs, out_cache_loc=ref_out_cache_loc, attn_backend=ref_backend)
        fb_ref.req_to_token_pool = ref_runner.req_to_token_pool
        fb_ref.token_to_kv_pool = ref_runner.token_to_kv_pool

        backend.init_forward_metadata(fb)
        ref_backend.init_forward_metadata(fb_ref)

        output = backend.forward_decode(q.clone(), k.clone(), v.clone(), layer, fb)
        output_ref = ref_backend.forward_decode(q.clone(), k.clone(), v.clone(), layer, fb_ref)

        output = output.view(self.batch_size, self.num_heads * self.head_dim)
        output_ref = output_ref.view(self.batch_size, self.num_heads * self.head_dim)

        self.assertFalse(
            torch.isnan(output).any(), "Dual pool output contains NaN"
        )
        max_diff = (output - output_ref).abs().max().item()
        self.assertLess(
            max_diff, 0.1,
            f"All-full-heads dual pool should match reference, max_diff={max_diff}",
        )

    def test_mixed_heads_dual_pool_no_nan(self):
        """With mixed full/compressed heads in dual pool, output should be valid."""
        # Half full, half compressed
        mask = torch.zeros(self.num_heads, device=self.device)
        mask[:self.num_heads // 2] = 1.0
        head_masks = {0: mask}

        model_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
            head_masks=head_masks,
        )

        backend = HeadReallocAttnBackend(model_runner, head_masks)

        layer = RadixAttention(
            num_heads=self.num_heads, head_dim=self.head_dim,
            scaling=1.0, num_kv_heads=self.num_heads, layer_id=0,
        )

        _setup_kv_cache_dual_pool(model_runner, layer, self.batch_size, self.seq_len)
        _mock_write_req_to_token(model_runner, self.batch_size, self.seq_len + 1)

        q = torch.randn(
            self.batch_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        k = torch.randn(
            self.batch_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        v = torch.randn(
            self.batch_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )

        total_len = self.seq_len + 1
        out_cache_loc = model_runner.token_to_kv_pool_allocator.alloc(self.batch_size)

        fb = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 100, (self.batch_size, 1), device=self.device),
            out_cache_loc=out_cache_loc,
            seq_lens_sum=self.batch_size * total_len,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=torch.tensor([total_len] * self.batch_size, device=self.device),
            seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
            attn_backend=backend,
        )
        fb.req_to_token_pool = model_runner.req_to_token_pool
        fb.token_to_kv_pool = model_runner.token_to_kv_pool

        backend.init_forward_metadata(fb)
        output = backend.forward_decode(q, k, v, layer, fb)

        expected_shape = (self.batch_size, self.num_heads * self.head_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")
        self.assertEqual(output.dtype, self.dtype)

    def test_all_compressed_heads_dual_pool_no_nan(self):
        """When all heads are compressed in dual pool, output should still be valid."""
        # All compressed
        head_masks = {0: torch.zeros(self.num_heads, device=self.device)}

        model_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
            head_masks=head_masks,
        )

        backend = HeadReallocAttnBackend(model_runner, head_masks)

        layer = RadixAttention(
            num_heads=self.num_heads, head_dim=self.head_dim,
            scaling=1.0, num_kv_heads=self.num_heads, layer_id=0,
        )

        _setup_kv_cache_dual_pool(model_runner, layer, self.batch_size, self.seq_len)
        _mock_write_req_to_token(model_runner, self.batch_size, self.seq_len + 1)

        q = torch.randn(
            self.batch_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        k = torch.randn(
            self.batch_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        v = torch.randn(
            self.batch_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )

        total_len = self.seq_len + 1
        out_cache_loc = model_runner.token_to_kv_pool_allocator.alloc(self.batch_size)

        fb = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 100, (self.batch_size, 1), device=self.device),
            out_cache_loc=out_cache_loc,
            seq_lens_sum=self.batch_size * total_len,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=torch.tensor([total_len] * self.batch_size, device=self.device),
            seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
            attn_backend=backend,
        )
        fb.req_to_token_pool = model_runner.req_to_token_pool
        fb.token_to_kv_pool = model_runner.token_to_kv_pool

        backend.init_forward_metadata(fb)
        output = backend.forward_decode(q, k, v, layer, fb)

        expected_shape = (self.batch_size, self.num_heads * self.head_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")

    def test_dual_pool_kv_split_correctness(self):
        """Verify KV data is correctly split between full and comp pools."""
        mask = torch.zeros(self.num_heads, device=self.device)
        mask[:4] = 1.0  # First 4 heads full, last 4 compressed
        head_masks = {0: mask}

        model_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
            head_masks=head_masks,
        )

        layer = RadixAttention(
            num_heads=self.num_heads, head_dim=self.head_dim,
            scaling=1.0, num_kv_heads=self.num_heads, layer_id=0,
        )

        cache_k, cache_v, loc = _setup_kv_cache_dual_pool(
            model_runner, layer, 1, 10,  # 1 batch, 10 tokens
        )
        kv_pool = model_runner.token_to_kv_pool

        # Verify full pool has only full heads (first 4)
        full_k = kv_pool.get_key_buffer(0)  # [size, 4, head_dim]
        self.assertEqual(full_k.shape[1], 4, "Full pool should have 4 heads")

        # Verify comp pool has only comp heads (last 4)
        comp_k = kv_pool.get_comp_key_buffer(0)  # [size, 4, head_dim]
        self.assertEqual(comp_k.shape[1], 4, "Comp pool should have 4 heads")

        # Verify the stored data matches the original
        full_idx = kv_pool.full_head_indices[0]
        comp_idx = kv_pool.comp_head_indices[0]

        # Check full pool data
        stored_full_k = full_k[loc].float()
        expected_full_k = cache_k[:, full_idx, :].float()
        max_diff_full = (stored_full_k - expected_full_k).abs().max().item()
        self.assertLess(max_diff_full, 1e-3, f"Full pool data mismatch: {max_diff_full}")

        # Check comp pool data
        comp_loc = kv_pool.translate_loc_full_to_comp(loc)
        stored_comp_k = comp_k[comp_loc].float()
        expected_comp_k = cache_k[:, comp_idx, :].float()
        max_diff_comp = (stored_comp_k - expected_comp_k).abs().max().item()
        self.assertLess(max_diff_comp, 1e-3, f"Comp pool data mismatch: {max_diff_comp}")

    def test_output_shape_correct(self):
        """Verify output shape is [batch_size, num_heads * head_dim]."""
        mask = torch.zeros(self.num_heads, device=self.device)
        mask[0] = 1.0  # Only 1 full head
        head_masks = {0: mask}

        model_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
            head_masks=head_masks,
        )

        backend = HeadReallocAttnBackend(model_runner, head_masks)

        layer = RadixAttention(
            num_heads=self.num_heads, head_dim=self.head_dim,
            scaling=1.0, num_kv_heads=self.num_heads, layer_id=0,
        )

        _setup_kv_cache_dual_pool(model_runner, layer, self.batch_size, self.seq_len)
        _mock_write_req_to_token(model_runner, self.batch_size, self.seq_len + 1)

        q = torch.randn(self.batch_size, self.num_heads, self.head_dim,
                         dtype=self.dtype, device=self.device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        total_len = self.seq_len + 1
        out_cache_loc = model_runner.token_to_kv_pool_allocator.alloc(self.batch_size)

        fb = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 100, (self.batch_size, 1), device=self.device),
            out_cache_loc=out_cache_loc,
            seq_lens_sum=self.batch_size * total_len,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=torch.tensor([total_len] * self.batch_size, device=self.device),
            seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
            attn_backend=backend,
        )
        fb.req_to_token_pool = model_runner.req_to_token_pool
        fb.token_to_kv_pool = model_runner.token_to_kv_pool

        backend.init_forward_metadata(fb)
        output = backend.forward_decode(q, k, v, layer, fb)

        self.assertEqual(output.shape, (self.batch_size, self.num_heads * self.head_dim))


if __name__ == "__main__":
    unittest.main()
