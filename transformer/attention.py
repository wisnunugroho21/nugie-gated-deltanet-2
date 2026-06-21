import flax.nnx as nnx
import jax
import jax.numpy as jnp
from rope import apply_rope

# ═══════════════════════════════════════════════════════════════════════════════
# Sliding-Window Attention
# ═══════════════════════════════════════════════════════════════════════════════


class SlidingWindowAttention(nnx.Module):
    """
    Standard multi-head self-attention with:
      • Causal sliding-window mask: token i attends to j ∈ [i−W+1, i]
      • RoPE positional encoding on q and k
    Corresponds to the SWA sub-layer in the hybrid cell (Fig. 1, left).

    Note: For production use, replace the explicit O(L²) attention matrix
    with a Flash Attention kernel (e.g., jax.nn.dot_product_attention with
    is_causal=True and a custom local-window implementation).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim**-0.5

        self.q_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        """x: (B, L, dim) — already pre-normed by the calling block."""
        B, L, _ = x.shape
        H, D_h = self.num_heads, self.head_dim

        # Project and reshape → (B, H, L, D_h)
        def _proj(lin: nnx.Linear) -> jax.Array:
            return lin(x).reshape(B, L, H, D_h).swapaxes(1, 2)

        q = apply_rope(_proj(self.q_proj), cos, sin)  # (B,H,L,D_h)
        k = apply_rope(_proj(self.k_proj), cos, sin)
        v = _proj(self.v_proj)  # (B,H,L,D_h)

        # Scaled dot-product  (B, H, L, L)
        scores = jnp.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # Causal sliding-window mask
        #   mask[i,j] = True  iff  j ≤ i  AND  j ≥ i − W + 1
        i_idx = jnp.arange(L)[:, None]
        j_idx = jnp.arange(L)[None, :]
        mask = (j_idx <= i_idx) & (j_idx >= i_idx - self.window_size + 1)
        neg_inf = jnp.finfo(scores.dtype).min
        scores = jnp.where(mask[None, None], scores, neg_inf)

        # Softmax in fp32 for stability, cast back for downstream
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
            x.dtype
        )  # (B,H,L,L)

        # Weighted values
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)  # (B,H,L,D_h)
        out = out.swapaxes(1, 2).reshape(B, L, H * D_h)  # (B,L,dim)
        return self.o_proj(out)
