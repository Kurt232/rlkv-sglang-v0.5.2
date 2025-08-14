import unittest

import torch

from sglang.srt.layers.attention.utils import create_streaming_window_kv_indices_triton
from sglang.test.test_utils import CustomTestCase


# copy from python/sglang/srt/layers/attention/mixed_triton_backend.py
def update_streaming_window_buffer(
    window_kv_indptr,
    req_to_token,
    sink_window_size,
    recent_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    device,
):
    # if seq_len <= sink + recent, then window_len is seq_len
    # otherwise, it is sink + recent
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(
            sink_window_size + recent_window_size, dtype=torch.long, device=device
        ),
    )

    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_indices = torch.empty(
        window_kv_indptr[-1], dtype=torch.int32, device=device
    )

    create_streaming_window_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        window_kv_indptr,
        window_kv_indices,
        sink_window_size,
        recent_window_size,
        req_to_token.stride(0),
    )
    return window_kv_indptr, window_kv_indices, window_kv_lens


class TestStreamingAttentionMask(CustomTestCase):
    def _run_test_case(self, B, max_ctx, sink, recent, seq_lens_list):

        device = "cuda"
        req_to_token = (
            torch.arange(max_ctx, dtype=torch.int32, device=device)
            .expand(B, max_ctx)
            .contiguous()
        )
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(B, dtype=torch.int32, device=device)
        indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)

        indptr, indices, lens = update_streaming_window_buffer(
            indptr, req_to_token, sink, recent, seq_lens, req_pool_indices, B, device
        )

        # Reference implementation
        ref = []
        for l in seq_lens.tolist():
            current_ref = []
            # Sink window
            current_ref.extend(list(range(min(l, sink))))
            # Recent window
            if l > sink:
                start_recent = max(sink, l - recent)
                current_ref.extend(list(range(start_recent, l)))
            ref.extend(current_ref)

        ref = torch.tensor(ref, dtype=torch.int32, device=device)
        self.assertTrue(
            torch.equal(indices, ref),
            f"Indices mismatch for B={B}, sink={sink}, recent={recent}, seq_lens={seq_lens_list}\nExpected: {ref}\nGot: {indices}",
        )
        print(
            f"PASS: B={B}, max_ctx={max_ctx}, sink={sink}, recent={recent}, seq_lens={seq_lens_list}"
        )

    def test_update_streaming_window_buffer(self):
        # Original case
        self._run_test_case(B=2, max_ctx=20, sink=2, recent=0, seq_lens_list=[6, 9])

        # No recent window
        self._run_test_case(B=2, max_ctx=20, sink=5, recent=0, seq_lens_list=[6, 9])

        # Only recent window (no sink)
        self._run_test_case(B=2, max_ctx=20, sink=0, recent=5, seq_lens_list=[6, 9])

        # Both sink and recent
        self._run_test_case(B=2, max_ctx=20, sink=3, recent=4, seq_lens_list=[6, 9])

        # Sequence length smaller than window size
        self._run_test_case(B=2, max_ctx=10, sink=5, recent=5, seq_lens_list=[3, 7])

        # Larger batch size
        self._run_test_case(
            B=4, max_ctx=20, sink=2, recent=3, seq_lens_list=[5, 7, 10, 12]
        )

        # Sequence length exactly sink + recent
        self._run_test_case(B=1, max_ctx=10, sink=3, recent=2, seq_lens_list=[5])

        # All zero sequence lengths
        self._run_test_case(B=3, max_ctx=10, sink=3, recent=2, seq_lens_list=[0, 0, 0])

        # All sequence lengths are 1
        self._run_test_case(
            B=4, max_ctx=10, sink=2, recent=3, seq_lens_list=[1, 1, 1, 1]
        )

        # Sequence lengths equal to sink and recent equals 0
        self._run_test_case(B=2, max_ctx=20, sink=4, recent=0, seq_lens_list=[4, 4])

        # Sequence lengths equal to recent and sink equals 0
        self._run_test_case(B=2, max_ctx=20, sink=0, recent=4, seq_lens_list=[4, 4])

        # sink + recent far greater than max_ctx
        self._run_test_case(B=2, max_ctx=8, sink=10, recent=10, seq_lens_list=[6, 8])

        # Very large batch size but each sequence is short (stress test)
        self._run_test_case(
            B=512, max_ctx=20, sink=2, recent=3, seq_lens_list=[3] * 256 + [5] * 256
        )

        # Increasing sequence lengths to maximum
        self._run_test_case(
            B=5, max_ctx=20, sink=3, recent=4, seq_lens_list=[3, 7, 8, 15, 20]
        )

        # Decreasing sequence lengths
        self._run_test_case(
            B=5, max_ctx=20, sink=3, recent=4, seq_lens_list=[20, 15, 8, 7, 3]
        )

        # Extreme combinations when sink or recent is 0
        self._run_test_case(B=3, max_ctx=20, sink=0, recent=0, seq_lens_list=[5, 6, 7])

        # Super long sequence (close to GPU memory boundary, ensure not OOM)
        self._run_test_case(
            B=1, max_ctx=8192, sink=128, recent=128, seq_lens_list=[4096]
        )


if __name__ == "__main__":
    unittest.main()
