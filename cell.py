import flax.nnx as nnx
import jax

from gated_deltanet_2.layer import GatedDeltaNet2Layer
from mlp import SwiGLUMLP
from transformer.attention import SlidingWindowAttention

# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid cell and full model
# ═══════════════════════════════════════════════════════════════════════════════


class HybridGDN2Cell(nnx.Module):
    """
    One repeated cell of the hybrid model (paper Fig. 1, left).

    The residual stream is split into TWO paired blocks, each wrapping a
    token mixer together with its following MLP under a single shared
    skip connection (confirmed by tracing Fig. 1's two ⊕ symbols against
    its dashed line / nested-box skip-connection paths):

        y = x + MLP1( norm_mlp1( GDN2(norm_gdn2(x)) ) )       — lower ⊕
        z = y + MLP2( norm_mlp2( SWA(norm_swa(y), …) ) )      — upper ⊕

    This is NOT the more common "one residual per sublayer" Transformer
    pattern (which would use four separate skip connections, one each for
    GDN2, MLP1, SWA, MLP2). Each [mixer, MLP] pair is instead treated as
    one computational unit — the MLP "finishes" what its preceding mixer
    started before the result rejoins the stream. This halves the number
    of residual-add / RMSNorm round-trips per cell (2 instead of 4), and
    still reduces to an exact identity map when all sublayer weights are
    zero, preserving the trainability benefit of standard residual nets.
    """

    def __init__(
        self,
        dim: int = 2_048,
        num_heads: int = 16,
        mlp_dim: int = 8_192,
        chunk_size: int = 64,
        conv_kernel: int = 4,
        window_size: int = 2_048,
        *,
        rngs: nnx.Rngs,
    ):
        # One pre-norm per sub-layer (standard pre-LN design)
        self.norm_gdn2 = nnx.RMSNorm(dim, rngs=rngs)
        self.norm_mlp1 = nnx.RMSNorm(dim, rngs=rngs)
        self.norm_swa = nnx.RMSNorm(dim, rngs=rngs)
        self.norm_mlp2 = nnx.RMSNorm(dim, rngs=rngs)

        self.gdn2 = GatedDeltaNet2Layer(
            dim, num_heads, chunk_size, conv_kernel, rngs=rngs
        )
        self.mlp1 = SwiGLUMLP(dim, mlp_dim, rngs=rngs)
        self.swa = SlidingWindowAttention(dim, num_heads, window_size, rngs=rngs)
        self.mlp2 = SwiGLUMLP(dim, mlp_dim, rngs=rngs)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        # Pair 1: GDN-2 mixer → MLP, ONE shared residual (Fig. 1 lower ⊕).
        # The skip connects the cell's original input straight to the
        # output of MLP1, bypassing both transformations as a single unit.
        y = x + self.mlp1(self.norm_mlp1(self.gdn2(self.norm_gdn2(x))))

        # Pair 2: SWA → MLP, ONE shared residual (Fig. 1 upper ⊕).
        # The skip connects y (pair 1's output) to the output of MLP2.
        z = y + self.mlp2(self.norm_mlp2(self.swa(self.norm_swa(y), cos, sin)))

        return z
