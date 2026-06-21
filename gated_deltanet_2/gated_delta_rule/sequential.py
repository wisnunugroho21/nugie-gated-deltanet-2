"""
Gated Delta Rule-2
Paper: https://arxiv.org/abs/2605.22791  (Eq. 9-10)

Two equivalent formulations of:

    S_bar_t = Diag(alpha_t) S_{t-1}                     -- per-channel decay
    r_t     = S_bar_t^T e_t,  e_t = b_t o k_t           -- read along erase dir
    S_t     = S_bar_t + k_t (z_t - r_t)^T,  z_t = w_t o v_t  -- write
    o_t     = S_t^T q_t                                  -- output

Equivalently (compact matrix form, Eq. 10):
    S_t = (I - k_t (b_t o k_t)^T) Diag(alpha_t) S_{t-1} + k_t (w_t o v_t)^T

Argument convention (both functions identical):
  beta   = b_t  in [0,1]^{d_k}  erase gate
  gamma  = w_t  in [0,1]^{d_v}  write gate                   [C1]
  delta  = a_t  in (0,1]^{d_k}  per-channel decay            [C1]

[C1] Paper uses gamma for cumulative log-decay and 'w' for the write gate.
     Consider renaming: beta->b, gamma->w, delta->alpha for paper alignment.
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Token-by-token sequential reference
# ---------------------------------------------------------------------------


def sequential_forward(
    query: jax.Array,  # (B, L, d_k)
    key: jax.Array,  # (B, L, d_k)
    value: jax.Array,  # (B, L, d_v)
    beta: jax.Array,  # (B, L, d_k)  erase gate b_t
    gamma: jax.Array,  # (B, L, d_v)  write gate w_t  [C1]
    delta: jax.Array,  # (B, L, d_k)  per-channel decay alpha_t  [C1]
) -> jax.Array:  # (B, L, d_v)
    """
    Token-by-token implementation of Gated Delta Rule-2 (paper Eq. 10).
    O(L * d_k^2 * d_v) -- use only for verification / learning.
    """
    batch_size, seq_len, query_dim = query.shape
    value_dim = value.shape[-1]

    S_t = jnp.zeros((batch_size, query_dim, value_dim), dtype=query.dtype)
    Id = jnp.eye(query_dim, dtype=query.dtype)  # (d_k, d_k); broadcasts over B

    outputs: list[jax.Array] = []

    for t in range(seq_len):
        # Slice token t and add a unit "seq" axis for matmul compatibility.
        q_t = jnp.expand_dims(query[:, t, :], axis=1)  # (B, 1, d_k)
        k_t = jnp.expand_dims(key[:, t, :], axis=1)  # (B, 1, d_k)
        v_t = jnp.expand_dims(value[:, t, :], axis=1)  # (B, 1, d_v)
        b_t = jnp.expand_dims(beta[:, t, :], axis=1)  # (B, 1, d_k)
        w_t = jnp.expand_dims(gamma[:, t, :], axis=1)  # (B, 1, d_v)
        d_t = jnp.expand_dims(delta[:, t, :], axis=1)  # (B, 1, d_k)

        # ── Transition matrix  A_t = (I - k_t e_t^T) Diag(alpha_t) ─────
        #   k_t.swapaxes(1,2)  : (B, d_k, 1)   column vector
        #   b_t * k_t          : (B, 1,  d_k)   row vector e_t^T = (b_t o k_t)^T
        #   outer product      : (B, d_k, d_k)  k_t e_t^T
        #   * d_t (broadcast)  : (B, 1,  d_k) -> column-wise scale by alpha_t
        #   A_t[i,j] = (delta_{ij} - k_i * e_j) * alpha_j          (Eq. 10)
        A_t = (Id - k_t.swapaxes(1, 2) * (b_t * k_t)) * d_t  # (B, d_k, d_k)

        # ── Write term  B_t = k_t z_t^T,  z_t = w_t o v_t ─────────────
        #   k_t.swapaxes(1,2) : (B, d_k, 1)
        #   w_t * v_t         : (B,  1, d_v)
        #   outer product     : (B, d_k, d_v)
        B_t = k_t.swapaxes(1, 2) * (w_t * v_t)  # (B, d_k, d_v)

        # ── State update and output ──────────────────────────────────────
        S_t = A_t @ S_t + B_t  # (B, d_k, d_v)   Eq. 10
        o_t = (q_t @ S_t).squeeze(1)  # (B, d_v)        o_t = S_t^T q_t

        outputs.append(o_t)

    return jnp.stack(outputs, axis=1)  # (B, L, d_v)
