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
        max_context_len = 16384
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


def profile_decode_breakdown():
    """Profile the breakdown of decode latency for head realloc vs standard attention.

    Usage:
        CUDA_VISIBLE_DEVICES=5 python test_head_realloc_backend.py --profile
    """
    import time

    torch.manual_seed(42)
    batch_size = 4
    seq_len = 8192
    num_heads = 8  # MHA: 8 KV heads = 8 Q heads (no GQA in this test)
    head_dim = 128
    device = "cuda"
    dtype = torch.float16
    num_layers = 1
    warmup = 5
    repeats = 50

    with mock.patch(
        "sglang.srt.layers.attention.head_realloc_backend.get_attention_tp_size",
        return_value=1,
    ):
        # --- Setup: HeadRealloc backend (half full, half comp) ---
        mask = torch.zeros(num_heads, device=device)
        mask[:num_heads // 2] = 1.0
        head_masks = {0: mask}

        runner_hr = MockModelRunner(
            num_heads=num_heads, head_dim=head_dim, head_masks=head_masks,
        )
        backend_hr = HeadReallocAttnBackend(runner_hr, head_masks)

        layer = RadixAttention(
            num_heads=num_heads, head_dim=head_dim,
            scaling=1.0, num_kv_heads=num_heads, layer_id=0,
        )

        _setup_kv_cache_dual_pool(runner_hr, layer, batch_size, seq_len)
        _mock_write_req_to_token(runner_hr, batch_size, seq_len + 1)

        total_len = seq_len + 1
        out_loc_hr = runner_hr.token_to_kv_pool_allocator.alloc(batch_size)

        fb_hr = ForwardBatch(
            batch_size=batch_size,
            input_ids=torch.randint(0, 100, (batch_size, 1), device=device),
            out_cache_loc=out_loc_hr,
            seq_lens_sum=batch_size * total_len,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(batch_size, device=device),
            seq_lens=torch.tensor([total_len] * batch_size, device=device),
            seq_lens_cpu=torch.tensor([total_len] * batch_size, device="cpu"),
            attn_backend=backend_hr,
        )
        fb_hr.req_to_token_pool = runner_hr.req_to_token_pool
        fb_hr.token_to_kv_pool = runner_hr.token_to_kv_pool

        # --- Setup: Reference backend (standard full attention) ---
        runner_ref = MockModelRunner(
            num_heads=num_heads, head_dim=head_dim, head_masks=None,
        )
        ref_backend = TorchNativeAttnBackend(runner_ref)

        cache_k = torch.randn(batch_size * seq_len, num_heads, head_dim, dtype=dtype, device=device)
        cache_v = torch.randn_like(cache_k)
        ref_loc = torch.arange(batch_size * seq_len, device=device)
        runner_ref.token_to_kv_pool.set_kv_buffer(layer, ref_loc, cache_k, cache_v)
        _mock_write_req_to_token(runner_ref, batch_size, seq_len + 1)

        ref_out_loc = torch.arange(batch_size * seq_len, batch_size * total_len, device=device)
        fb_ref = ForwardBatch(
            batch_size=batch_size,
            input_ids=torch.randint(0, 100, (batch_size, 1), device=device),
            out_cache_loc=ref_out_loc,
            seq_lens_sum=batch_size * total_len,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(batch_size, device=device),
            seq_lens=torch.tensor([total_len] * batch_size, device=device),
            seq_lens_cpu=torch.tensor([total_len] * batch_size, device="cpu"),
            attn_backend=ref_backend,
        )
        fb_ref.req_to_token_pool = runner_ref.req_to_token_pool
        fb_ref.token_to_kv_pool = runner_ref.token_to_kv_pool

        q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # ======= Profile: init_forward_metadata =======
        def bench(fn, label):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / repeats * 1000
            return elapsed

        t_meta_hr = bench(lambda: backend_hr.init_forward_metadata(fb_hr), "HR init_meta")
        t_meta_ref = bench(lambda: ref_backend.init_forward_metadata(fb_ref), "Ref init_meta")

        # ======= Profile: forward_decode breakdown =======
        # HeadRealloc: step-by-step profiling
        backend_hr.init_forward_metadata(fb_hr)
        ref_backend.init_forward_metadata(fb_ref)

        layer_id = 0
        full_q_idx = backend_hr.full_q_head_indices[layer_id]
        comp_q_idx = backend_hr.comp_q_head_indices[layer_id]

        def time_op(fn, label):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / repeats * 1000

        # 1. KV write
        t_kv_hr = time_op(
            lambda: runner_hr.token_to_kv_pool.set_kv_buffer(layer, out_loc_hr, k, v),
            "HR set_kv_buffer",
        )
        t_kv_ref = time_op(
            lambda: runner_ref.token_to_kv_pool.set_kv_buffer(layer, ref_out_loc, cache_k[:batch_size], cache_v[:batch_size]),
            "Ref set_kv_buffer",
        )

        # 2. Q gather
        q_3d = q.view(batch_size, num_heads, head_dim)
        t_q_gather = time_op(
            lambda: (q_3d[:, full_q_idx, :].contiguous(), q_3d[:, comp_q_idx, :].contiguous()),
            "Q gather x2",
        )

        # 3. Full heads attention kernel
        q_full = q_3d[:, full_q_idx, :].contiguous()
        k_buf_f, v_buf_f = backend_hr._get_kv_buffers(runner_hr.token_to_kv_pool, 0, "full")
        o_part_f = torch.empty(batch_size, len(full_q_idx), head_dim, dtype=dtype, device=device)
        al_f = torch.empty(batch_size, len(full_q_idx), backend_hr.max_kv_splits, head_dim, dtype=torch.float32, device=device)
        alse_f = torch.empty(batch_size, len(full_q_idx), backend_hr.max_kv_splits, dtype=torch.float32, device=device)

        t_attn_full = time_op(
            lambda: backend_hr.decode_attention_fwd(
                q_full, k_buf_f, v_buf_f, o_part_f,
                backend_hr.forward_metadata.kv_indptr,
                backend_hr.forward_metadata.kv_indices,
                al_f, alse_f,
                backend_hr.forward_metadata.num_kv_splits,
                backend_hr.max_kv_splits, 1.0, 0.0,
            ),
            "Attn full heads",
        )

        # 4. Comp heads attention kernel
        q_comp = q_3d[:, comp_q_idx, :].contiguous()
        k_buf_c, v_buf_c = backend_hr._get_kv_buffers(runner_hr.token_to_kv_pool, 0, "comp")
        o_part_c = torch.empty(batch_size, len(comp_q_idx), head_dim, dtype=dtype, device=device)
        al_c = torch.empty(batch_size, len(comp_q_idx), backend_hr.max_kv_splits, head_dim, dtype=torch.float32, device=device)
        alse_c = torch.empty(batch_size, len(comp_q_idx), backend_hr.max_kv_splits, dtype=torch.float32, device=device)

        t_attn_comp = time_op(
            lambda: backend_hr.decode_attention_fwd(
                q_comp, k_buf_c, v_buf_c, o_part_c,
                backend_hr.forward_metadata.window_kv_indptr,
                backend_hr.forward_metadata.window_kv_indices,
                al_c, alse_c,
                backend_hr.forward_metadata.window_num_kv_splits,
                backend_hr.max_kv_splits, 1.0, 0.0,
            ),
            "Attn comp heads",
        )

        # 5. Output scatter
        o_full = torch.empty(batch_size, num_heads, head_dim, dtype=dtype, device=device)
        t_scatter = time_op(
            lambda: (o_full.__setitem__((slice(None), full_q_idx, slice(None)), o_part_f),
                     o_full.__setitem__((slice(None), comp_q_idx, slice(None)), o_part_c)),
            "Output scatter x2",
        )

        # 6. Reference full forward_decode (end-to-end)
        t_ref_e2e = time_op(
            lambda: ref_backend.forward_decode(q.clone(), k.clone(), v.clone(), layer, fb_ref),
            "Ref forward_decode",
        )
        t_hr_e2e = time_op(
            lambda: backend_hr.forward_decode(q.clone(), k.clone(), v.clone(), layer, fb_hr),
            "HR forward_decode",
        )

        # ======= Report =======
        print(f"\n{'='*65}")
        print(f"Decode Latency Breakdown (bs={batch_size}, seq={seq_len}, heads={num_heads}, avg of {repeats} runs)")
        print(f"{'='*65}")
        print(f"{'Operation':<35} {'HeadRealloc':>12} {'Reference':>12}")
        print(f"{'-'*65}")
        print(f"{'init_forward_metadata':<35} {t_meta_hr:>10.3f}ms {t_meta_ref:>10.3f}ms")
        print(f"{'set_kv_buffer':<35} {t_kv_hr:>10.3f}ms {t_kv_ref:>10.3f}ms")
        print(f"{'Q gather (x2 contiguous)':<35} {t_q_gather:>10.3f}ms {'N/A':>12}")
        print(f"{'Attention (full heads)':<35} {t_attn_full:>10.3f}ms {'':>12}")
        print(f"{'Attention (comp heads)':<35} {t_attn_comp:>10.3f}ms {'':>12}")
        print(f"{'Output scatter (x2)':<35} {t_scatter:>10.3f}ms {'N/A':>12}")
        print(f"{'-'*65}")
        t_hr_parts = t_kv_hr + t_q_gather + t_attn_full + t_attn_comp + t_scatter
        print(f"{'Sum of parts':<35} {t_hr_parts:>10.3f}ms {'':>12}")
        print(f"{'End-to-end forward_decode':<35} {t_hr_e2e:>10.3f}ms {t_ref_e2e:>10.3f}ms")
        print(f"{'Overhead':<35} {t_hr_e2e - t_ref_e2e:>10.3f}ms {t_hr_e2e/t_ref_e2e:>11.2f}x")
        print(f"{'='*65}")


if __name__ == "__main__":
    import sys
    if "--profile" in sys.argv:
        sys.argv.remove("--profile")
        profile_decode_breakdown()
    else:
        unittest.main()
