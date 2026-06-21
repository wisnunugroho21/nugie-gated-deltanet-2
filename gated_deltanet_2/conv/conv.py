import flax.nnx as nnx
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Short causal convolution helper  [B3]
# Paper Fig.1: q, k, v each pass through a short causal conv + SiLU
# ---------------------------------------------------------------------------


class ShortCausalConv(nnx.Module):
    """
    Depthwise causal convolution with kernel size 4 (as used in GDN-2 / GDN).
    Equivalent to FLA's ShortConvolution in casual mode.
    """

    def __init__(self, dim: int, kernel_size: int, rngs: nnx.Rngs):
        # Depthwise: groups = dim
        self.kernel_size = kernel_size
        self.dim = dim
        # Weight: (dim, 1, kernel_size)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (dim, kernel_size)) * 0.02
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, L, dim)
        B, L, D = x.shape
        k = self.kernel_size
        # Causal padding: pad k-1 zeros at the start of the time axis
        x_pad = jnp.pad(x, ((0, 0), (k - 1, 0), (0, 0)))  # (B, L+k-1, D)
        # Depthwise conv via sliding window
        # weight: (D, k),  x_windows: (B, L, D, k)
        x_windows = jnp.stack(
            [x_pad[:, i : i + L, :] for i in range(k)], axis=-1
        )  # (B, L, D, k)
        out = jnp.einsum("bldk,dk->bld", x_windows, self.weight)
        return out
