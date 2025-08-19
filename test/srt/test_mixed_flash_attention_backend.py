"""
Usage:
python3 -m unittest test_mixed_flash_attention_backend.TestMixedFlashAttnBackend.test_mmlu_with_full
python3 -m unittest test_mixed_flash_attention_backend.TestMixedFlashAttnBackend.test_mmlu_with_adapter
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.mixed_flash_backend import MixedFlashAttnBackend
from sglang.srt.layers.attention.mixed_triton_backend import MixedTritonAttnBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)


# modified from python/sglang/srt/layers/attention/mixed_triton_backend.py
class MockModelRunner:
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
        # Max batch size for the test.
        max_batch_size = 160
        # Total tokens(prefix + extend + decode) in the test should not exceed this length.
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
        self.device = self.device
        # Create a large enough req_to_token_pool to fit the test usage.
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                # A typical max_bs * max_context_len for cuda graph decode
                "size": max_batch_size,
                # Add req_to_token attribute
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
            page_size=1,  # page_size=1 means each token has its own page
            dtype=self.dtype,
            head_num=num_heads,
            head_dim=head_dim,
            layer_num=1,  # only consider layer=1 for unit test
            device=self.device,
            enable_memory_saver=False,
        )
        # Required by torch native backend
        self.server_args = ServerArgs(
            model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            enable_mixed_attention=True,
            sink_window_size=sink_window_size,
            recent_window_size=recent_window_size,
        )
        self.kv_cache_dtype = self.dtype


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestMixedFlashAttnWithFullAttn(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._patcher = mock.patch(
            "sglang.srt.layers.attention.mixed_flash_backend.get_attention_tp_size",
            return_value=1,
        )
        cls.mock_tp_size = cls._patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()

    def setUp(self):
        # Test parameters
        self.batch_size = 2
        self.seq_len = 512
        self.num_heads = 8
        self.head_dim = 128
        self.device = "cuda"
        self.dtype = torch.float16

    def _init_model_runner(self):
        self.model_runner = MockModelRunner(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        self.backend = MixedFlashAttnBackend(self.model_runner)
        self.ref_backend = TorchNativeAttnBackend(self.model_runner)

    def _mock_write_to_req_to_token_pool(self, batch_size, seq_len):
        # if page_size > 1, the token pool stores the index to the page.
        # so we need to multiply the index by page_size.
        self.req_to_token = (
            torch.arange(0, batch_size, dtype=torch.int32, device=self.device)[:, None]
            * seq_len
            + torch.arange(0, seq_len, dtype=torch.int32, device=self.device)[None, :]
        )
        self.model_runner.req_to_token_pool.req_to_token[:batch_size, :seq_len] = (
            self.req_to_token
        )

    def _create_attention_layer(self):
        """Create attention layer for testing."""
        return RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_heads,
            layer_id=0,
        )

    def _create_qkv_tensors(self, tokens_len):
        """Create q, k, v tensors for testing."""
        shape = (tokens_len, self.num_heads, self.head_dim)
        return (
            torch.randn(shape, dtype=self.dtype, device=self.device),
            torch.randn(shape, dtype=self.dtype, device=self.device),
            torch.randn(shape, dtype=self.dtype, device=self.device),
        )

    def _run_reference_forward(
        self, mode, q, k, v, layer, forward_batch, expected_shape
    ):
        """Run reference forward pass using native backend."""
        if mode == ForwardMode.EXTEND:
            output = self.ref_backend.forward_extend(q, k, v, layer, forward_batch)
        else:  # ForwardMode.DECODE
            output = self.ref_backend.forward_decode(q, k, v, layer, forward_batch)
        return output.view(expected_shape)

    def _verify_output(self, output, expected_shape, output_ref):
        """Verify output tensor shape, dtype, and values."""
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {output.shape}",
        )
        self.assertEqual(output.dtype, self.dtype)
        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(
            torch.isnan(output).sum().item(), 0, "Output contains NaN values"
        )

        if not torch.allclose(output, output_ref, atol=1e-1, rtol=0.0):
            # Check where the values differ beyond the given tolerances
            diff_mask = ~torch.isclose(output, output_ref, atol=1e-1, rtol=0.0)

            # Find the first index where the difference occurs
            if diff_mask.any():
                first_mismatch_idx = diff_mask.nonzero()[0]
                print("First mismatch at index:", tuple(first_mismatch_idx.tolist()))
                print("output:", output[tuple(first_mismatch_idx.tolist())])
                print("output_ref:", output_ref[tuple(first_mismatch_idx.tolist())])
            raise AssertionError(
                "the backend output is not close to the ref backend output"
            )

    def _create_forward_batch(self, mode, q_len=None, prefix_len=0):
        """Create a forward batch for testing based on mode and lengths."""
        self._init_model_runner()

        # Default to self.seq_len if not specified
        q_len = q_len or self.seq_len

        if mode == ForwardMode.EXTEND:
            total_len = prefix_len + q_len
            out_cache_start = prefix_len * self.batch_size
            out_cache_end = total_len * self.batch_size

            forward_batch = ForwardBatch(
                batch_size=self.batch_size,
                input_ids=torch.randint(
                    0, 100, (self.batch_size, q_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    out_cache_start, out_cache_end, device=self.device
                ),
                seq_lens_sum=self.batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(self.batch_size, device=self.device),
                seq_lens=torch.tensor(
                    [total_len] * self.batch_size, device=self.device
                ),
                seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
                extend_prefix_lens=torch.tensor(
                    [prefix_len] * self.batch_size, device=self.device
                ),
                extend_prefix_lens_cpu=torch.tensor(
                    [prefix_len] * self.batch_size, device="cpu"
                ),
                extend_seq_lens=torch.tensor(
                    [q_len] * self.batch_size, device=self.device
                ),
                extend_seq_lens_cpu=torch.tensor(
                    [q_len] * self.batch_size, device="cpu"
                ),
                attn_backend=self.backend,
            )
        else:  # ForwardMode.DECODE
            decode_len = q_len  # Assuming 1 for decode testing
            total_len = self.seq_len + decode_len
            out_cache_start = self.batch_size * self.seq_len
            out_cache_end = self.batch_size * total_len

            forward_batch = ForwardBatch(
                batch_size=self.batch_size,
                input_ids=torch.randint(
                    0, 100, (self.batch_size, decode_len), device=self.device
                ),
                out_cache_loc=torch.tensor(
                    [out_cache_start, out_cache_end], device=self.device
                ),
                seq_lens_sum=self.batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(self.batch_size, device=self.device),
                seq_lens=torch.tensor(
                    [total_len] * self.batch_size, device=self.device
                ),
                seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
                attn_backend=self.backend,
            )

        # Add token pool
        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool

        # Write current batch's req_to_token to req_to_token_pool
        self._mock_write_to_req_to_token_pool(self.batch_size, total_len)
        # Add kv pool for this forward batch
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool

        return forward_batch

    def _setup_kv_cache(self, forward_batch, layer, cache_len):
        # Create constant values for the prefix cache for easy debugging
        cache_k = torch.ones(
            self.batch_size * cache_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        cache_v = (
            torch.ones(
                self.batch_size * cache_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            * 2
        )

        # Set the prefix KV cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            torch.arange(self.batch_size * cache_len, device=self.device),
            cache_k,
            cache_v,
            layer.k_scale,
            layer.v_scale,
        )

    def _run_attention_test(self, mode, q_len, prefix_len=0):
        """
            Run an attention test with the specified parameters.
        Args:
            mode: ForwardMode.EXTEND or ForwardMode.DECODE
            q_len: Length of the query sequence. For decode mode, q_len is 1.
            prefix_len: Length of the prefix sequence for extend mode
        """
        layer = self._create_attention_layer()
        layer.alpha_weight = nn.Parameter(
            torch.ones(
                self.num_heads,
                device=self.device,
                dtype=self.dtype,
            )
            * 1
        )

        # Create forward batch and set up
        forward_batch = self._create_forward_batch(mode, q_len, prefix_len)

        # Create QKV tensors for the input
        q, k, v = self._create_qkv_tensors(self.batch_size * q_len)

        # KV cache for prefixed extend is prefix_len
        # KV cache for decode is same as seq_len
        # No KV cache for extend without prefix
        if mode == ForwardMode.EXTEND:
            if prefix_len > 0:
                self._setup_kv_cache(forward_batch, layer, prefix_len)
        else:
            self._setup_kv_cache(forward_batch, layer, self.seq_len)

        self.backend.init_forward_metadata(forward_batch)

        if mode == ForwardMode.EXTEND:
            expected_shape = (
                self.batch_size * q_len,
                self.num_heads * self.head_dim,
            )
            output = self.backend.forward_extend(q, k, v, layer, forward_batch)
        else:
            expected_shape = (self.batch_size, self.num_heads * self.head_dim)
            output = self.backend.forward_decode(q, k, v, layer, forward_batch)

        output_ref = self._run_reference_forward(
            mode, q, k, v, layer, forward_batch, expected_shape
        )

        self._verify_output(output, expected_shape, output_ref)

        return output

    def test_forward_extend(self):
        """Test the standard extend operation."""
        self._run_attention_test(ForwardMode.EXTEND, q_len=self.seq_len)

    def test_forward_decode(self):
        """Test the decode operation with cached tokens."""
        self._run_attention_test(ForwardMode.DECODE, q_len=1)

    def test_forward_extend_with_prefix(self):
        """Test extending from cached prefix tokens."""
        prefix_len = self.seq_len // 2
        extend_len = self.seq_len - prefix_len
        self._run_attention_test(
            ForwardMode.EXTEND, q_len=extend_len, prefix_len=prefix_len
        )
        # ! can pass this one


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestMixedFlashAttnWithMixedAttn(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._patcher = mock.patch(
            "sglang.srt.layers.attention.mixed_flash_backend.get_attention_tp_size",
            return_value=1,
        )
        cls.mock_tp_size = cls._patcher.start()
        cls._patcher1 = mock.patch(
            "sglang.srt.layers.attention.mixed_triton_backend.get_attention_tp_size",
            return_value=1,
        )
        cls.mock_tp_size1 = cls._patcher1.start()

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()
        cls._patcher1.stop()

    def setUp(self):
        # torch.manual_seed(42)
        # Test parameters
        self.batch_size = 2
        self.seq_len = 512
        self.num_heads = 8
        self.head_dim = 128
        self.device = "cuda"
        self.dtype = torch.float16

    def _init_model_runner(self):
        self.model_runner = MockModelRunner(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        self.ref_model_runner = MockModelRunner(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        self.backend = MixedFlashAttnBackend(self.model_runner)
        self.ref_backend = MixedTritonAttnBackend(self.ref_model_runner)

    def _mock_write_to_req_to_token_pool(self, batch_size, seq_len):
        # if page_size > 1, the token pool stores the index to the page.
        # so we need to multiply the index by page_size.
        self.req_to_token = (
            torch.arange(0, batch_size, dtype=torch.int32, device=self.device)[:, None]
            * seq_len
            + torch.arange(0, seq_len, dtype=torch.int32, device=self.device)[None, :]
        )
        self.model_runner.req_to_token_pool.req_to_token[:batch_size, :seq_len] = (
            self.req_to_token
        )
        self.req_to_token = (
            torch.arange(0, batch_size, dtype=torch.int32, device=self.device)[:, None]
            * seq_len
            + torch.arange(0, seq_len, dtype=torch.int32, device=self.device)[None, :]
        )
        self.ref_model_runner.req_to_token_pool.req_to_token[:batch_size, :seq_len] = (
            self.req_to_token
        )

    def _create_attention_layer(self):
        """Create attention layer for testing."""
        return RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_heads,
            layer_id=0,
        )

    def _create_qkv_tensors(self, tokens_len):
        """Create q, k, v tensors for testing."""
        shape = (tokens_len, self.num_heads, self.head_dim)
        return (
            torch.randn(shape, dtype=self.dtype, device=self.device),
            torch.randn(shape, dtype=self.dtype, device=self.device),
            torch.randn(shape, dtype=self.dtype, device=self.device),
        )

    def _verify_output(self, output, expected_shape, output_ref):
        """Verify output tensor shape, dtype, and values."""
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {output.shape}",
        )
        self.assertEqual(output.dtype, self.dtype)
        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(
            torch.isnan(output).sum().item(), 0, "Output contains NaN values"
        )

        if not torch.allclose(output, output_ref, atol=1e-1, rtol=0.0):
            total_elements = output.numel()
            print(
                f"Matrix shape ->  {tuple(output.shape)}  (total elements: {total_elements})"
            )
            diff_mask = ~torch.isclose(output, output_ref, atol=1e-1, rtol=0.0)
            if diff_mask.any():
                first_mismatch_idx = diff_mask.nonzero()[0]
                print("First mismatch at index:", tuple(first_mismatch_idx.tolist()))
                print("output:", output[tuple(first_mismatch_idx.tolist())])
                print("output_ref:", output_ref[tuple(first_mismatch_idx.tolist())])

            error = output - output_ref
            error_mean = error.float().mean().item()
            error_var = error.float().var(unbiased=False).item()
            print(f"Error stats  ->  mean: {error_mean:.6e}, var: {error_var:.6e}")

            diff_count = diff_mask.sum().item()
            diff_ratio = diff_count / total_elements
            print(
                f"Different elements: {diff_count} / {total_elements}  ({100.0 * diff_ratio:.2f} %)"
            )

            raise AssertionError(
                "the backend output is not close to the ref backend output"
            )

    def _create_forward_batch(self, mode, q_len=None, prefix_len=0):
        """Create a forward batch for testing based on mode and lengths."""
        self._init_model_runner()

        # Default to self.seq_len if not specified
        q_len = q_len or self.seq_len

        if mode == ForwardMode.EXTEND:
            total_len = prefix_len + q_len
            out_cache_start = prefix_len * self.batch_size
            out_cache_end = total_len * self.batch_size

            forward_batch = ForwardBatch(
                batch_size=self.batch_size,
                input_ids=torch.randint(
                    0, 100, (self.batch_size, q_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    out_cache_start, out_cache_end, device=self.device
                ),
                seq_lens_sum=self.batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(self.batch_size, device=self.device),
                seq_lens=torch.tensor(
                    [total_len] * self.batch_size, device=self.device
                ),
                seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
                extend_prefix_lens=torch.tensor(
                    [prefix_len] * self.batch_size, device=self.device
                ),
                extend_prefix_lens_cpu=torch.tensor(
                    [prefix_len] * self.batch_size, device="cpu"
                ),
                extend_seq_lens=torch.tensor(
                    [q_len] * self.batch_size, device=self.device
                ),
                extend_seq_lens_cpu=torch.tensor(
                    [q_len] * self.batch_size, device="cpu"
                ),
                attn_backend=self.backend,
            )

            forward_batch_ref = ForwardBatch(
                batch_size=forward_batch.batch_size,
                input_ids=forward_batch.input_ids.clone(),
                out_cache_loc=forward_batch.out_cache_loc.clone(),
                seq_lens_sum=forward_batch.seq_lens_sum,
                forward_mode=forward_batch.forward_mode,
                req_pool_indices=forward_batch.req_pool_indices.clone(),
                seq_lens=forward_batch.seq_lens.clone(),
                seq_lens_cpu=forward_batch.seq_lens_cpu.clone(),
                extend_prefix_lens=forward_batch.extend_prefix_lens.clone(),
                extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu.clone(),
                extend_seq_lens=forward_batch.extend_seq_lens.clone(),
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu.clone(),
                attn_backend=self.ref_backend,
            )

        else:  # ForwardMode.DECODE
            decode_len = q_len  # Assuming 1 for decode testing
            total_len = self.seq_len + decode_len
            out_cache_start = self.batch_size * self.seq_len
            out_cache_end = self.batch_size * total_len

            forward_batch = ForwardBatch(
                batch_size=self.batch_size,
                input_ids=torch.randint(
                    0, 100, (self.batch_size, decode_len), device=self.device
                ),
                out_cache_loc=torch.tensor(
                    [out_cache_start, out_cache_end], device=self.device
                ),
                seq_lens_sum=self.batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(self.batch_size, device=self.device),
                seq_lens=torch.tensor(
                    [total_len] * self.batch_size, device=self.device
                ),
                seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
                attn_backend=self.backend,
            )

            forward_batch_ref = ForwardBatch(
                batch_size=forward_batch.batch_size,
                input_ids=forward_batch.input_ids.clone(),
                out_cache_loc=forward_batch.out_cache_loc.clone(),
                seq_lens_sum=forward_batch.seq_lens_sum,
                forward_mode=forward_batch.forward_mode,
                req_pool_indices=forward_batch.req_pool_indices.clone(),
                seq_lens=forward_batch.seq_lens.clone(),
                seq_lens_cpu=forward_batch.seq_lens_cpu.clone(),
                attn_backend=self.ref_backend,
            )

        # Add token pool
        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool
        forward_batch_ref.req_to_token_pool = self.ref_model_runner.req_to_token_pool

        # Write current batch's req_to_token to req_to_token_pool
        self._mock_write_to_req_to_token_pool(self.batch_size, total_len)
        # Add kv pool for this forward batch
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
        forward_batch_ref.token_to_kv_pool = self.ref_model_runner.token_to_kv_pool

        return forward_batch, forward_batch_ref

    def _setup_kv_cache(self, forward_batch, layer, cache_len):
        # Create constant values for the prefix cache for easy debugging
        cache_k = torch.ones(
            self.batch_size * cache_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        cache_v = (
            torch.ones(
                self.batch_size * cache_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            * 2
        )

        # Set the prefix KV cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            torch.arange(self.batch_size * cache_len, device=self.device),
            cache_k,
            cache_v,
            layer.k_scale,
            layer.v_scale,
        )

    def test_forward_decode_compare_with_native(self):
        """Test the decode against MixedNativeAttnBackend."""
        self._init_model_runner()

        layer = self._create_attention_layer()

        # Set alpha_weight to 0.5 for mixed attention
        layer.alpha_weight = nn.Parameter(
            torch.ones(
                self.num_heads,
                device=self.device,
                dtype=self.dtype,
            )
            * 0.5
        )

        # Test for DECODE mode
        q_len_decode = 1
        forward_batch, forward_batch_ref = self._create_forward_batch(
            ForwardMode.DECODE, q_len=q_len_decode
        )
        q_decode, k_decode, v_decode = self._create_qkv_tensors(
            self.batch_size * q_len_decode
        )

        self._setup_kv_cache(forward_batch, layer, self.seq_len)
        self._setup_kv_cache(forward_batch_ref, layer, self.seq_len)

        self.backend.init_forward_metadata(forward_batch)
        self.ref_backend.init_forward_metadata(forward_batch_ref)

        output_flash_decode = self.backend.forward_decode(
            q_decode, k_decode, v_decode, layer, forward_batch
        )

        output_native_decode = self.ref_backend.forward_decode(
            q_decode, k_decode, v_decode, layer, forward_batch_ref
        )

        except_shape = (
            self.batch_size,
            self.num_heads * self.head_dim,
        )

        output_native_decode = output_native_decode.view(except_shape)

        self._verify_output(output_flash_decode, except_shape, output_native_decode)


class TestMixedFlashAttnBackend(CustomTestCase):
    def test_latency(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST,
            [
                "--attention-backend",
                "fa3",
                "--enable-torch-compile",
                "--cuda-graph-max-bs",
                4,
                "--enable-mixed-attention",
                "--sink-window-size",
                128,
                "--recent-window-size",
                256,
            ],
        )

        print(f"{output_throughput=}")

        if is_in_ci():
            self.assertGreater(output_throughput, 153)

    def test_mmlu_with_full(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "fa3",
                "--enable-mixed-attention",
                "--sink-window-size",
                128,
                "--recent-window-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], 0.65)
        finally:
            kill_process_tree(process.pid)

    def test_mmlu_with_adapter(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        model_name = model.split("/")[-1]
        adapter_path = "mixed-adapter-weight-Llama-3.1-8B-Instruct.pt"
        assert os.path.exists(
            adapter_path
        ), f"Adapter path {adapter_path} does not exist"
        assert (
            model_name in adapter_path
        ), f"Adapter path {adapter_path} is NOT for {model_name}"
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "fa3",
                "--enable-mixed-attention",
                "--sink-window-size",
                128,
                "--recent-window-size",
                256,
                "--adapter-load-path",
                adapter_path,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], 0.65)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
