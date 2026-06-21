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
# Chunk-parallel formulation (naive O(C^3) per chunk, for learning only)
# ---------------------------------------------------------------------------


def chunked_forward(
    query: jax.Array,  # (B, L, d_k)
    key: jax.Array,  # (B, L, d_k)
    value: jax.Array,  # (B, L, d_v)
    beta: jax.Array,  # (B, L, d_k)  erase gate b_t
    gamma: jax.Array,  # (B, L, d_v)  write gate w_t  [C1]
    delta: jax.Array,  # (B, L, d_k)  per-channel decay alpha_t  [C1]
    chunk_size: int,
) -> jax.Array:  # (B, L, d_v)
    """
    Chunk-parallel implementation.

    For position r inside a chunk the state is expressed in closed form:

        S_r = (A_r A_{r-1} ... A_0) S_prev
            + sum_{i=0}^{r} (A_r ... A_{i+1}) B_i

    where S_prev is the state carried in from the previous chunk.
    All token states share the same S_prev, enabling parallel computation.

    Complexity: O((L/C) * C^3 * d_k^2)  -- very slow for large C.
    Exists only to make the math of the chunked recurrence transparent.
    """
    batch_size, seq_len, query_dim = query.shape
    value_dim = value.shape[-1]

    assert seq_len % chunk_size == 0, (  # [F3]
        f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
    )

    # Explicitly batched identity for matmuls inside the chunk loop.
    Id = jnp.broadcast_to(
        jnp.eye(query_dim, dtype=query.dtype),
        (batch_size, query_dim, query_dim),
    )

    S_t = jnp.zeros((batch_size, query_dim, value_dim), dtype=query.dtype)
    outputs: list[jax.Array] = []
    num_chunks = seq_len // chunk_size

    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_size

        # ── Slice the chunk, add a "1" axis at position 2 for matmul ────
        # Resulting shape: (B, C, 1, d_*)                            [F1]
        # (original comments said "(B, 1, d_*)" -- incorrect)
        q_c = jnp.expand_dims(
            query[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_k)
        k_c = jnp.expand_dims(
            key[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_k)
        v_c = jnp.expand_dims(
            value[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_v)
        b_c = jnp.expand_dims(
            beta[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_k)
        w_c = jnp.expand_dims(
            gamma[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_v)
        d_c = jnp.expand_dims(
            delta[:, start : start + chunk_size, :], axis=2
        )  # (B,C,1,d_k)

        # ── Build per-token A_r and B_r for r in [0, C) ─────────────────
        A_c: list[jax.Array] = []
        B_c: list[jax.Array] = []

        for r in range(chunk_size):
            # Index along the chunk axis (axis 1) to get single-token tensors.
            k_r = k_c[:, r, :, :]  # (B, 1, d_k)  [F5]
            v_r = v_c[:, r, :, :]  # (B, 1, d_v)
            b_r = b_c[:, r, :, :]  # (B, 1, d_k)
            w_r = w_c[:, r, :, :]  # (B, 1, d_v)
            d_r = d_c[:, r, :, :]  # (B, 1, d_k)

            A_r = (Id - k_r.swapaxes(1, 2) * (b_r * k_r)) * d_r  # (B, d_k, d_k)
            B_r = k_r.swapaxes(1, 2) * (w_r * v_r)  # (B, d_k, d_v)

            A_c.append(A_r)
            B_c.append(B_r)

        # ── Compute S_r and o_r for each position in the chunk ───────────
        o_c: list[jax.Array] = []
        S_last = S_t  # will be overwritten; init avoids unbound-variable risk

        for r in range(chunk_size):
            # Prefix product P_r = A_r A_{r-1} ... A_0
            # Built left-to-right: A_c[i] @ prefix  =>  A_i(A_{i-1}(... I))
            prefix = Id
            for i in range(r + 1):
                prefix = A_c[i] @ prefix  # (B, d_k, d_k)

            # Suffix-weighted write accumulation:
            #   accum = sum_{i=0}^{r} (A_r ... A_{i+1}) B_i
            # When i == r the suffix is I (no matrices to the right of B_r).
            accum = jnp.zeros(
                (batch_size, query_dim, value_dim), dtype=query.dtype
            )  # [F2]
            for i in range(r + 1):
                suffix = Id
                for j in range(i + 1, r + 1):
                    suffix = A_c[j] @ suffix  # (B, d_k, d_k)
                accum = accum + suffix @ B_c[i]

            # Closed-form state at position r
            S_r = prefix @ S_t + accum  # (B, d_k, d_v)

            # Output token r
            o_r = (q_c[:, r, :, :] @ S_r).squeeze(1)  # (B, d_v)
            o_c.append(o_r)
            S_last = S_r  # track the last state in the chunk  [F4]

        S_t = S_last  # carry to the next chunk  [F4]
        outputs.extend(o_c)

    return jnp.stack(outputs, axis=1)  # (B, L, d_v)
