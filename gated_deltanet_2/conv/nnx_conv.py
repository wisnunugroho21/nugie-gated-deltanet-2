import flax.nnx as nnx
import jax

# ---------------------------------------------------------------------------
# Short causal convolution helper  [B3]
# Paper Fig.1: q, k, v each pass through a short causal conv + SiLU
# ---------------------------------------------------------------------------


class ShortCausalConv(nnx.Module):
    """
    Depthwise causal 1-D convolution with kernel size k  (default 4).
    Paper Fig.1: q / k / v each pass through this before SiLU.

    Implemented via nnx.Conv with two settings combined:
      • padding='CAUSAL'        — left-pads (k−1) zeros, no future leakage
                                   (verified by impulse-response test: changing
                                   x[t] never changes y[t'] for t' < t)
      • feature_group_count=dim — depthwise: each channel gets its own
                                   independent k-tap filter, no cross-channel
                                   mixing (verified: perturbing channel c only
                                   changes output channel c)

    Kernel shape is (kernel_size, in_features // feature_group_count, out_features)
    = (k, 1, dim) for the depthwise case. Tap order: kernel[0] is the oldest
    tap (lag k−1), kernel[k−1] is the most recent tap (lag 0) — standard JAX/
    Flax convolution convention, confirmed via impulse-response test.
    """

    def __init__(self, dim: int, kernel_size: int = 4, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=(kernel_size,),
            padding="CAUSAL",
            feature_group_count=dim,  # depthwise
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (B, L, dim)  →  (B, L, dim)"""
        return self.conv(x)
