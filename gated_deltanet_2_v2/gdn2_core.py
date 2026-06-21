"""
Gated DeltaNet-2 — chunkwise parallel training core (JAX).

Implements Section 3.3 / Appendix A of Hatamizadeh, Choi, Kautz,
"Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention" (arXiv:2605.22791).

State orientation follows the paper: S in R^{dk x dv}, output o_t = S_t^T q_t.

Per-head recurrence (Eq. 29):
    S_r = (I - k_r e_r^T) diag(alpha_r) S_{r-1} + k_r z_r^T,
    e_r = b_r ⊙ k_r,   z_r = w_r ⊙ v_r,   alpha_r = exp(g_r).

Chunkwise form (Eqs. 18-25 / 30-44):
    G_r   = cumsum(g)             (inclusive, within chunk)
    gamma = exp(G),  gamma_C = gamma[-1]
    Kbar  = gamma^{-1} ⊙ K        (decay-normalized keys)
    Ebar  = gamma     ⊙ (B ⊙ K)   (decay-absorbed erase factor)
    Z     = W ⊙ V
    T     = tril(Ebar Kbar^T, -1)
    A     = (I + T)^{-1}          (unit lower-triangular solve)
    Y, U  = A Ebar, A Z           (WY auxiliaries; share the same inverse)
    R     = U - Y S0
    O     = Qgamma S0 + Aqk R,     Qgamma = gamma ⊙ Q, Aqk = tril(Qgamma Kbar^T)
    S_C   = diag(gamma_C) S0 + Ktail^T R,   Ktail_r = (gamma_C / gamma_r) ⊙ k_r
"""

import jax
import jax.numpy as jnp
from jax import lax


# --------------------------------------------------------------------------- #
#  Single (batch, head) sequence — the actual algorithm.
#  Everything else is vmap over (B, H) on top of this.
# --------------------------------------------------------------------------- #
def _chunkwise_single(q, k, v, g, b, w, S0, chunk_size):
    """q,k,g,b: [L, dk]  v,w: [L, dv]  S0: [dk, dv]  ->  (O: [L, dv], S_final: [dk, dv])."""
    L, dk = k.shape
    dv = v.shape[-1]
    C = chunk_size
    N = L // C
    cdtype = jnp.float32  # chunk math runs in fp32 (paper App. D)

    def to_chunks(x):
        return x.reshape(N, C, x.shape[-1]).astype(cdtype)

    q, k, v = to_chunks(q), to_chunks(k), to_chunks(v)
    g, b, w = to_chunks(g), to_chunks(b), to_chunks(w)
    eye = jnp.eye(C, dtype=cdtype)

    def chunk_step(S, inp):
        qc, kc, vc, gc, bc, wc = inp                      # each [C, d*]

        G = jnp.cumsum(gc, axis=0)                        # [C, dk] cumulative log-decay
        gamma = jnp.exp(G)                                # [C, dk]
        gamma_C = gamma[-1]                               # [dk]

        Kbar = kc * jnp.exp(-G)                           # gamma^{-1} ⊙ K
        Ebar = gamma * (bc * kc)                          # gamma ⊙ (B ⊙ K)   (Eq. 33)
        Z = wc * vc                                       # W ⊙ V
        Qg = gamma * qc                                   # gamma ⊙ Q

        T = jnp.tril(Ebar @ Kbar.T, k=-1)                 # strictly lower (Eq. 34)
        A = jax.scipy.linalg.solve_triangular(            # A = (I + T)^{-1}
            eye + T, eye, lower=True, unit_diagonal=True)

        Y = A @ Ebar                                      # erase-side aux [C, dk]
        U = A @ Z                                         # write-side aux [C, dv]
        R = U - Y @ S                                     # [C, dv]   (Eq. 35)

        Aqk = jnp.tril(Qg @ Kbar.T)                       # lower incl. diag (Eq. 43)
        O = Qg @ S + Aqk @ R                              # [C, dv]   (Eq. 44)

        Ktail = kc * (gamma_C[None, :] / gamma)           # row r: (gamma_C/gamma_r) ⊙ k_r
        S_new = gamma_C[:, None] * S + Ktail.T @ R        # [dk, dv]  (Eq. 40)
        return S_new, O

    S_final, O = lax.scan(chunk_step, S0.astype(cdtype), (q, k, v, g, b, w))
    return O.reshape(L, dv), S_final


def _recurrent_single(q, k, v, g, b, w, S0):
    """Token-by-token reference (Eq. 9 / 29). Same signature as the chunkwise core."""
    alpha = jnp.exp(g.astype(jnp.float32))
    e = (b * k).astype(jnp.float32)
    z = (w * v).astype(jnp.float32)
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)

    def step(S, inp):
        qt, kt, at, et, zt = inp
        S_bar = at[:, None] * S                           # D_t S_{t-1}  (decay rows)
        r_t = S_bar.T @ et                                # [dv] read along erase dir
        S_new = S_bar + kt[:, None] * (zt - r_t)[None, :] # write delta residual
        o_t = S_new.T @ qt                                # [dv]
        return S_new, o_t

    S_final, O = lax.scan(step, S0.astype(jnp.float32), (q, k, alpha, e, z))
    return O, S_final


# --------------------------------------------------------------------------- #
#  Batched public entry points: inputs are [B, H, L, d].
# --------------------------------------------------------------------------- #
def _batchify(fn):
    # vmap over heads (axis 1) then batch (axis 0); S0 has no L axis.
    over_heads = jax.vmap(fn, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))
    return jax.vmap(over_heads, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))


def chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, chunk_size=64):
    """Parallel chunkwise forward.

    q, k, g, b : [B, H, L, dk]      v, w : [B, H, L, dv]      S0 : [B, H, dk, dv]
    returns (O : [B, H, L, dv], S_final : [B, H, dk, dv]).
    """
    f = lambda *a: _chunkwise_single(*a, chunk_size=chunk_size)
    return _batchify(f)(q, k, v, g, b, w, S0)


def recurrent_gated_delta_rule_2(q, k, v, g, b, w, S0):
    """Token-by-token reference forward, same I/O as the chunkwise version."""
    return _batchify(_recurrent_single)(q, k, v, g, b, w, S0)
