import triton
import triton.language as tl

# Keep this in sync with the Triton kernel inside `create_flashmla_kv_indices_triton`.
# Number of pages that the kernel writes per iteration.
# Exposed here so other Python modules can import it instead of hard-coding 64.
TRITON_PAD_NUM_PAGE_PER_BLOCK = 64


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0  # logical index
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(
        tl.int32
    )  # += min(seq_len, sliding_window_size + 1)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start  # physical index
            + offset,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + kv_indices_offset + offset, data, mask=mask
        )  # logical index


@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_indices_ptr_stride: tl.constexpr,
    NUM_PAGE_PER_BLOCK: tl.constexpr = TRITON_PAD_NUM_PAGE_PER_BLOCK,
    PAGED_SIZE: tl.constexpr = 64,
):
    BLOCK_SIZE: tl.constexpr = 4096
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start

    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)

    for i in range(num_pages_loop):
        # index into req_to_token_ptr needs to be int64
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        mask = paged_offset < num_paged * PAGED_SIZE
        mask_out = paged_offset_out < num_paged

        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + paged_offset,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            data // PAGED_SIZE,
            mask=mask_out,
        )


@triton.jit
def create_streaming_window_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    local_kv_start_idx,  # [bs]
    kv_indices_ptr,
    sink_window_size: tl.constexpr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)  # logical start index for this req

    local_kv_start = tl.load(local_kv_start_idx + pid).to(tl.int32)
    window_len = tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    if sink_window_size < local_kv_start:
        # [0, sink_window_size)
        # sink_kv_start = 0
        # sink_kv_end = sink_window_size
        num_loop = tl.cdiv(sink_window_size, BLOCK_SIZE)
        for i in range(num_loop):
            # index into req_to_token_ptr needs to be int64
            offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
            mask = offset < sink_window_size
            data = tl.load(
                req_to_token_ptr
                + req_pool_index * req_to_token_ptr_stride
                + offset,  # physical index
                mask=mask,
            )
            tl.store(
                kv_indices_ptr + kv_indices_offset + offset, data, mask=mask
            )  # logical index

        # [-local_window_size, seq_len)
        local_kv_end = local_kv_start + window_len - sink_window_size
        num_loop = tl.cdiv(local_kv_end - local_kv_start, BLOCK_SIZE)
        for i in range(num_loop):
            # index into req_to_token_ptr needs to be int64
            offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
            mask = offset < local_kv_end - local_kv_start
            data = tl.load(
                req_to_token_ptr
                + req_pool_index * req_to_token_ptr_stride
                + local_kv_start  # physical index
                + offset,
                mask=mask,
            )
            tl.store(
                kv_indices_ptr + kv_indices_offset + sink_window_size + offset,
                data,
                mask=mask,
            )  # logical index
    else:  # window_len == seq_len
        kv_start = 0
        kv_end = window_len

        num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
        for i in range(num_loop):
            # index into req_to_token_ptr needs to be int64
            offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
            mask = offset < kv_end - kv_start
            data = tl.load(
                req_to_token_ptr
                + req_pool_index * req_to_token_ptr_stride
                + kv_start  # physical index
                + offset,
                mask=mask,
            )
            tl.store(
                kv_indices_ptr + kv_indices_offset + offset, data, mask=mask
            )  # logical index


@triton.jit
def create_streaming_mask_triton(
    mask_ptr,  # [batch, seqlen_q, seqlen_k]
    mask_stride_ptr,  # [batch]
    seqlen_q_ptr,  # [batch]
    seqlen_k_ptr,  # [batch]
    prefix_q_ptr,  # [batch]
    sink_size: tl.constexpr,
    local_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 512,
):
    # todo::
    # Program IDs
    pid_batch = tl.program_id(0)  # which batch
    pid_q = tl.program_id(1)  # which q-block
    pid_k = tl.program_id(2)  # which k-block

    # Load seq lengths etc.
    seqlen_q = tl.load(seqlen_q_ptr + pid_batch)
    seqlen_k = tl.load(seqlen_k_ptr + pid_batch)
    prefix_q = tl.load(prefix_q_ptr + pid_batch)
    mask_stride = tl.load(mask_stride_ptr + pid_batch)

    # Q/K indices this block is responsible for
    q_offs = pid_q * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # (BLOCK_SIZE,)
    k_offs = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # (BLOCK_SIZE,)

    # Valid mask (avoid writing OOB)
    q_mask = q_offs < seqlen_q
    k_mask = k_offs < seqlen_k

    # Broadcast to (Q,K)
    q_idx = q_offs[:, None]  # (BLOCK,1)
    k_idx = k_offs[None, :]  # (1,BLOCK)

    # sink tokens: always visible
    sink_mask = k_idx < sink_size

    # local causal mask
    left = prefix_q + q_idx - local_size - 1
    right = prefix_q + q_idx
    local_mask = (k_idx >= left) & (k_idx <= right)

    # final mask
    final_mask = sink_mask | local_mask

    # Global pointer for this batch
    base_ptr = mask_ptr + pid_batch * mask_stride

    # Compute memory offsets (row-major [q,k])
    offs = q_idx * seqlen_k + k_idx
    ptrs = base_ptr + offs

    # Only store for valid (q,k)
    store_mask = q_mask[:, None] & k_mask[None, :]

    tl.store(ptrs, final_mask.to(tl.int8), mask=store_mask)
