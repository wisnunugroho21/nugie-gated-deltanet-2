import flax.nnx as nnx
import jax
import jax.numpy as jnp

from conv.nnx_conv import ShortCausalConv
from gated_delta_rule.chunked_optimized import chunked_forward_optimized

# ---------------------------------------------------------------------------
# Gated DeltaNet-2 token mixer  (GDN-2 block, Fig. 1)
# ---------------------------------------------------------------------------


class GatedDeltaNet2Layer(nnx.Module):
    """
    Full GDN-2 token mixer  (paper Fig. 1, right side).

    q, k paths:  Linear → ShortCausalConv → SiLU → L2-norm  (per head)
    v    path:   Linear → ShortCausalConv → SiLU
    α  (decay):  Linear(bias=True) + learnable per-channel log-scale a
                 → g_t = −exp(a) ⊙ softplus(proj(x) + bias)
                 → α_t = exp(g_t)  ∈ (0,1]^{d_k}          (paper Eq.12)
    b  (erase):  Linear → sigmoid  ∈ [0,1]^{d_k}
    w  (write):  Linear → sigmoid  ∈ [0,1]^{d_v}
    output:      RMSNorm(recurrent_out) ⊙ SiLU(gate(x)) → o_proj
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        chunk_size: int,
        conv_kernel: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        # ── q / k / v projections + depthwise short convolutions ────────
        self.q_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.q_conv = ShortCausalConv(dim, conv_kernel, rngs=rngs)
        self.k_conv = ShortCausalConv(dim, conv_kernel, rngs=rngs)
        self.v_conv = ShortCausalConv(dim, conv_kernel, rngs=rngs)

        # ── Channel-wise gates ──────────────────────────────────────────
        self.b_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)  # erase
        self.w_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)  # write

        # ── Decay branch (paper Eq.12) ───────────────────────────────────
        # decay_proj: W_f x + bias  (bias=True provides the δ offset)
        self.decay_proj = nnx.Linear(dim, dim, use_bias=True, rngs=rngs)
        # decay_log_scale: learnable per-head-channel log-scale  `a`
        # Initialised to 0 → exp(a)=1 at startup (neutral timescale)
        self.decay_log_scale = nnx.Param(jnp.zeros((num_heads, self.head_dim)))

        # ── Output branch ───────────────────────────────────────────────
        self.out_norm = nnx.RMSNorm(dim, rngs=rngs)
        self.out_gate_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _l2(x: jax.Array, eps: float = 1e-6) -> jax.Array:
        """L2-normalise the last dimension (per head)."""
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)

    def _flat(self, x: jax.Array, B: int, L: int) -> jax.Array:
        """(B, L, dim) → (B·H, L, d_h)  for the per-head recurrence."""
        return (
            x.reshape(B, L, self.num_heads, self.head_dim)
            .swapaxes(1, 2)  # (B, H, L, d_h)
            .reshape(B * self.num_heads, L, self.head_dim)
        )

    def _unflat(self, x: jax.Array, B: int, L: int) -> jax.Array:
        """(B·H, L, d_h) → (B, L, dim)"""
        return (
            x.reshape(B, self.num_heads, L, self.head_dim)
            .swapaxes(1, 2)  # (B, L, H, d_h)
            .reshape(B, L, self.num_heads * self.head_dim)
        )

    # ── forward ────────────────────────────────────────────────────────────

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (B, L, dim) — already pre-normed by the calling block."""
        B, L, _ = x.shape

        # q, k:  Linear → Conv → SiLU → L2-norm (per head)
        q_h = self._l2(
            jax.nn.silu(self.q_conv(self.q_proj(x))).reshape(
                B, L, self.num_heads, self.head_dim
            )
        )
        k_h = self._l2(
            jax.nn.silu(self.k_conv(self.k_proj(x))).reshape(
                B, L, self.num_heads, self.head_dim
            )
        )
        # v:  Linear → Conv → SiLU  (no L2-norm on value)
        v_h = jax.nn.silu(self.v_conv(self.v_proj(x))).reshape(
            B, L, self.num_heads, self.head_dim
        )

        # Erase gate  b_t ∈ [0,1]^{d_k}
        b_h = jax.nn.sigmoid(self.b_proj(x)).reshape(
            B, L, self.num_heads, self.head_dim
        )
        # Write gate  w_t ∈ [0,1]^{d_v}
        w_h = jax.nn.sigmoid(self.w_proj(x)).reshape(
            B, L, self.num_heads, self.head_dim
        )

        # Decay  α_t ∈ (0,1]^{d_k}  (paper Eq. 12, computed in fp32)
        # g_t = −exp(a) ⊙ softplus( W_f x_t + bias )
        # α_t = exp(g_t)
        raw = self.decay_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        scale = jnp.exp(self.decay_log_scale[...])  # (H, d_h)
        g = -(scale * jax.nn.softplus(raw.astype(jnp.float32)))
        alpha = jnp.exp(g).astype(x.dtype)  # (B,L,H,d_h)

        # Flatten all (B,L,H,d_h) → (B·H, L, d_h) for the recurrence
        o_flat = chunked_forward_optimized(
            self._flat(q_h, B, L),
            self._flat(k_h, B, L),
            self._flat(v_h, B, L),
            self._flat(b_h, B, L),
            self._flat(w_h, B, L),
            self._flat(alpha, B, L),
            self.chunk_size,
        )  # (B·H, L, d_h)

        # Recombine: (B·H, L, d_h) → (B, L, dim)
        o = self._unflat(o_flat, B, L)

        # Output gate + norm + projection  (paper Sec 3.5)
        # output = o_proj( RMSNorm(o) ⊙ SiLU(gate(x)) )
        return self.o_proj(self.out_norm(o) * jax.nn.silu(self.out_gate_proj(x)))
