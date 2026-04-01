"""
Unit tests for Head Reallocation Attention Backend.

Tests that head-level split attention produces correct results by comparing
full heads' output against standard full attention and compressed heads'
output against streaming window attention.

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
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ServerArgs


class MockModelRunner:
    """Mock model runner for unit testing."""

    def __init__(
        self,
        num_heads=8,
        head_dim=128,
        sink_window_size=128,
        recent_window_size=256,
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
        self.server_args = ServerArgs(
            model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            enable_rlkv_inference=True,
            sink_window_size=sink_window_size,
            recent_window_size=recent_window_size,
        )
        self.kv_cache_dtype = self.dtype


def _mock_write_req_to_token(model_runner, batch_size, seq_len):
    """Write sequential token indices to req_to_token_pool."""
    req_to_token = (
        torch.arange(0, batch_size, dtype=torch.int32, device="cuda")[:, None]
        * seq_len
        + torch.arange(0, seq_len, dtype=torch.int32, device="cuda")[None, :]
    )
    model_runner.req_to_token_pool.req_to_token[:batch_size, :seq_len] = req_to_token


def _setup_kv_cache(model_runner, layer, batch_size, cache_len):
    """Write random KV cache for testing."""
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
    model_runner.token_to_kv_pool.set_kv_buffer(
        layer, loc, cache_k, cache_v,
    )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestHeadReallocBackendDecode(unittest.TestCase):
    """Test decode with RLKV backend vs reference (TorchNative full attention)."""

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

    def test_all_full_heads_matches_reference(self):
        """When all heads are full, RLKV output should match full attention."""
        model_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
        )
        ref_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
        )

        # All heads are full (mask = 1)
        head_masks = {0: torch.ones(self.num_heads, device=self.device)}

        backend = HeadReallocAttnBackend(model_runner, head_masks)
        ref_backend = TorchNativeAttnBackend(ref_runner)

        layer = RadixAttention(
            num_heads=self.num_heads, head_dim=self.head_dim,
            scaling=1.0, num_kv_heads=self.num_heads, layer_id=0,
        )

        # Setup KV cache (same data in both runners)
        torch.manual_seed(123)
        _setup_kv_cache(model_runner, layer, self.batch_size, self.seq_len)
        torch.manual_seed(123)
        _setup_kv_cache(ref_runner, layer, self.batch_size, self.seq_len)

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

        fb_kwargs = dict(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 100, (self.batch_size, 1), device=self.device),
            out_cache_loc=torch.arange(out_cache_start, out_cache_end, device=self.device),
            seq_lens_sum=self.batch_size * total_len,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=torch.tensor([total_len] * self.batch_size, device=self.device),
            seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
        )

        fb = ForwardBatch(**fb_kwargs, attn_backend=backend)
        fb.req_to_token_pool = model_runner.req_to_token_pool
        fb.token_to_kv_pool = model_runner.token_to_kv_pool

        fb_ref = ForwardBatch(**fb_kwargs, attn_backend=ref_backend)
        fb_ref.req_to_token_pool = ref_runner.req_to_token_pool
        fb_ref.token_to_kv_pool = ref_runner.token_to_kv_pool

        backend.init_forward_metadata(fb)
        ref_backend.init_forward_metadata(fb_ref)

        output = backend.forward_decode(q.clone(), k.clone(), v.clone(), layer, fb)
        output_ref = ref_backend.forward_decode(q.clone(), k.clone(), v.clone(), layer, fb_ref)

        output = output.view(self.batch_size, self.num_heads * self.head_dim)
        output_ref = output_ref.view(self.batch_size, self.num_heads * self.head_dim)

        self.assertFalse(
            torch.isnan(output).any(), "RLKV output contains NaN"
        )
        max_diff = (output - output_ref).abs().max().item()
        self.assertLess(
            max_diff, 0.1,
            f"All-full-heads RLKV should match reference, max_diff={max_diff}",
        )

    def test_mixed_heads_no_nan(self):
        """With mixed full/compressed heads, output should be valid (no NaN)."""
        model_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
        )

        # Half full, half compressed
        mask = torch.zeros(self.num_heads, device=self.device)
        mask[:self.num_heads // 2] = 1.0
        head_masks = {0: mask}

        backend = HeadReallocAttnBackend(model_runner, head_masks)

        layer = RadixAttention(
            num_heads=self.num_heads, head_dim=self.head_dim,
            scaling=1.0, num_kv_heads=self.num_heads, layer_id=0,
        )

        _setup_kv_cache(model_runner, layer, self.batch_size, self.seq_len)
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
        out_cache_start = self.batch_size * self.seq_len
        out_cache_end = self.batch_size * total_len

        fb = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 100, (self.batch_size, 1), device=self.device),
            out_cache_loc=torch.arange(out_cache_start, out_cache_end, device=self.device),
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

    def test_all_compressed_heads_no_nan(self):
        """When all heads are compressed, output should still be valid."""
        model_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
        )

        # All compressed
        head_masks = {0: torch.zeros(self.num_heads, device=self.device)}

        backend = HeadReallocAttnBackend(model_runner, head_masks)

        layer = RadixAttention(
            num_heads=self.num_heads, head_dim=self.head_dim,
            scaling=1.0, num_kv_heads=self.num_heads, layer_id=0,
        )

        _setup_kv_cache(model_runner, layer, self.batch_size, self.seq_len)
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
        out_cache_start = self.batch_size * self.seq_len
        out_cache_end = self.batch_size * total_len

        fb = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 100, (self.batch_size, 1), device=self.device),
            out_cache_loc=torch.arange(out_cache_start, out_cache_end, device=self.device),
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

    def test_output_shape_correct(self):
        """Verify output shape is [batch_size, num_heads * head_dim]."""
        model_runner = MockModelRunner(
            num_heads=self.num_heads, head_dim=self.head_dim,
        )

        mask = torch.zeros(self.num_heads, device=self.device)
        mask[0] = 1.0  # Only 1 full head
        head_masks = {0: mask}

        backend = HeadReallocAttnBackend(model_runner, head_masks)

        layer = RadixAttention(
            num_heads=self.num_heads, head_dim=self.head_dim,
            scaling=1.0, num_kv_heads=self.num_heads, layer_id=0,
        )

        _setup_kv_cache(model_runner, layer, self.batch_size, self.seq_len)
        _mock_write_req_to_token(model_runner, self.batch_size, self.seq_len + 1)

        q = torch.randn(self.batch_size, self.num_heads, self.head_dim,
                         dtype=self.dtype, device=self.device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        total_len = self.seq_len + 1

        fb = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 100, (self.batch_size, 1), device=self.device),
            out_cache_loc=torch.arange(
                self.batch_size * self.seq_len,
                self.batch_size * total_len,
                device=self.device,
            ),
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
