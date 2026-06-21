import jax
import jax.numpy as jnp

# ═══════════════════════════════════════════════════════════════════════════════
# Positional encoding utilities (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10_000.0):
    """
    Precomputes (cos, sin) tables for Rotary Position Embedding.
    Returns two arrays of shape (max_seq_len, head_dim // 2).
    Frequencies: θ_i = 1 / (theta ^ (2i / head_dim))
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    table = jnp.outer(t, freqs)  # (max_seq_len, head_dim // 2)
    return jnp.cos(table), jnp.sin(table)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """
    Apply RoPE to x (shape: ..., L, head_dim).
    cos, sin: (max_seq_len, head_dim // 2) — sliced to L inside.

    Uses the "split-half" convention (first half and second half rotated):
        [x1, x2] → [x1·cos − x2·sin,  x1·sin + x2·cos]
    which is mathematically equivalent to LLaMA / Mistral RoPE.
    """
    L = x.shape[-2]
    half = x.shape[-1] // 2
    cos_L = cos[:L, :]  # (L, half)
    sin_L = sin[:L, :]

    x1 = x[..., :half]  # (..., L, half)
    x2 = x[..., half:]  # (..., L, half)

    return jnp.concatenate(
        [x1 * cos_L - x2 * sin_L, x1 * sin_L + x2 * cos_L],
        axis=-1,
    )
