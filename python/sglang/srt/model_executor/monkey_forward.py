import torch


def monkey_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(
        q, k, v, forward_batch, save_kv_cache=True, adapter=self.adapter
    )
    output, _ = self.o_proj(attn_output)
    return output


def monkey_qwen3_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self._apply_qk_norm(q, k)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(
        q, k, v, forward_batch, save_kv_cache=True, adapter=self.adapter
    )
    output, _ = self.o_proj(attn_output)
    return output
