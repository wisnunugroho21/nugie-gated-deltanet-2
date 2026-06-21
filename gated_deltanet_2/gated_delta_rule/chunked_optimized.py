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
# Core recurrence (mathematically correct in the original; kept intact)
# Parameter names are now aligned with paper notation:
#   b_t = erase gate  (paper: b_t ∈ [0,1]^{d_k})
#   w_t = write gate  (paper: w_t ∈ [0,1]^{d_v})
#   alpha_t = per-channel decay  (paper: α_t ∈ (0,1]^{d_k})
# ---------------------------------------------------------------------------


def chunked_forward_optimized(
    query: jax.Array,  # (B, L, d_k)
    key: jax.Array,  # (B, L, d_k)
    value: jax.Array,  # (B, L, d_v)
    b: jax.Array,  # (B, L, d_k) — erase gate
    w: jax.Array,  # (B, L, d_v) — write gate
    alpha: jax.Array,  # (B, L, d_k) — per-channel decay  ← was "delta"
    chunk_size: int,
) -> jax.Array:
    """
    Implements the Gated Delta Rule-2 recurrence (paper Eq. 10):

        S_t = (I − k_t (b_t ⊙ k_t)ᵀ) Diag(α_t) S_{t-1} + k_t (w_t ⊙ v_t)ᵀ
        o_t = Sₜᵀ q_t

    Uses a two-level loop:
      outer — jax.lax.scan across chunks (sequential across chunk boundaries)
      inner — jax.lax.associative_scan inside each chunk (parallel prefix)
    """
    batch_size, seq_len, dk = query.shape
    dv = value.shape[-1]
    num_chunks = seq_len // chunk_size
    assert seq_len % chunk_size == 0, (
        f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
    )

    # Reshape → (num_chunks, batch, chunk_size, dim) so scan iterates axis-0
    def to_scan(x):
        return x.reshape(batch_size, num_chunks, chunk_size, -1).swapaxes(0, 1)

    q_s = to_scan(query)
    k_s = to_scan(key)
    v_s = to_scan(value)
    b_s = to_scan(b)
    w_s = to_scan(w)
    a_s = to_scan(alpha)  # renamed from d_s

    Id = jnp.eye(dk, dtype=query.dtype)

    # ── Associative scan combiner ──────────────────────────────────────────
    # Encodes the linear recurrence S_t = A_t S_{t-1} + B_t
    # (A1,B1) = prefix ending at t1; (A2,B2) = update from t1+1 to t2
    def combine(
        left: tuple[jax.Array, jax.Array], right: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        A1, B1 = left
        A2, B2 = right
        return A2 @ A1, A2 @ B1 + B2

    # ── Outer scan: one step = one chunk ──────────────────────────────────
    def chunk_step(
        S_prev: jax.Array,
        xs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        q_c, k_c, v_c, b_c, w_c, a_c = xs

        # ── Build per-token A and B matrices (paper Eq. 10) ──────────────
        # erase factor: e_t = b_t ⊙ k_t
        # A_t = (I − k_t eₜᵀ) Diag(α_t)
        #     = (I − k_t (b_t⊙k_t)ᵀ) Diag(α_t)
        # [i,j] = (δ_{ij} − k_i·(b⊙k)_j) · α_j   ← column-wise scaling by α
        e_c = b_c * k_c  # (B, C, dk)
        outer_ke = jnp.einsum("bci,bcj->bcij", k_c, e_c)  # k eᵀ
        # Expand α to (B,C,1,dk) so it broadcasts column-wise
        A_c = (Id - outer_ke) * jnp.expand_dims(a_c, axis=-2)  # (B,C,dk,dk)

        # write term: B_t = k_t (w_t ⊙ v_t)ᵀ = k_t zₜᵀ
        z_c = w_c * v_c  # (B, C, dv)
        B_c = jnp.einsum("bci,bcj->bcij", k_c, z_c)  # (B,C,dk,dv)

        # ── Associative scan over the chunk (parallel prefix) ─────────────
        A_ct = A_c.swapaxes(0, 1)  # (C, B, dk, dk)
        B_ct = B_c.swapaxes(0, 1)  # (C, B, dk, dv)
        A_cum_t, B_cum_t = jax.lax.associative_scan(combine, (A_ct, B_ct))
        A_cum = A_cum_t.swapaxes(0, 1)  # (B, C, dk, dk)
        B_cum = B_cum_t.swapaxes(0, 1)  # (B, C, dk, dv)

        # ── Compute all hidden states in this chunk ───────────────────────
        # S_r = A_cum_r S_prev + B_cum_r
        S_all = jnp.einsum("bcij,bjk->bcik", A_cum, S_prev) + B_cum
        # (B, C, dk, dv)

        # ── Outputs: o_r = Sᵀ_r q_r ─────────────────────────────────────
        o_c = jnp.einsum("bci,bcij->bcj", q_c, S_all)  # (B, C, dv)

        S_next = S_all[:, -1, :]  # last token's state
        return S_next, o_c

    S_init = jnp.zeros((batch_size, dk, dv), dtype=query.dtype)
    _, o_chunks = jax.lax.scan(
        chunk_step, S_init, (q_s, k_s, v_s, b_s, w_s, a_s)
    )  # o_chunks: (num_chunks, batch, chunk_size, dv)

    o = o_chunks.swapaxes(0, 1)  # (batch, num_chunks, chunk_size, dv)
    return o.reshape(batch_size, num_chunks * chunk_size, dv)
