"""
Gated DeltaNet-2 token-mixer layer in Flax NNX (block design of Fig. 1 / App. C).

Pipeline per the paper:
  q,k = L2norm(SiLU(ShortConv(Linear(x))))      # key-side, normalized
  v   =        SiLU(ShortConv(Linear(x)))        # value-side
  g   = -exp(a) ⊙ softplus(Linear_f(x) + delta)  # log-decay (fp32)
  b   = sigmoid(Linear_b(x))   (optionally x2 for the negative-eigenvalue variant)
  w   = sigmoid(Linear_w(x))
  O   = chunkwise_gated_delta_rule_2(q,k,v,g,b,w, state)
  out = Linear_o( RMSNorm(O) * SiLU(gate) )

Grouped value heads (GQA): with num_v_heads = G * num_heads, the key-side tensors
q, k, g, b are repeated across the G value-head groups (App. C.1); v, w live on the
value-head axis.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from gdn2_core import chunkwise_gated_delta_rule_2


class ShortConv(nnx.Module):
    """Causal depthwise 1-D convolution (the 'Conv' boxes in Fig. 1)."""

    def __init__(self, channels: int, kernel_size: int = 4, *, rngs: nnx.Rngs):
        self.channels = channels
        self.kernel_size = kernel_size
        key = rngs.params()
        w = jax.random.normal(key, (channels, 1, kernel_size)) * (kernel_size ** -0.5)
        self.weight = nnx.Param(w)
        self.bias = nnx.Param(jnp.zeros((channels,)))

    def __call__(self, x):                      # x: [B, L, C]
        xt = jnp.transpose(x, (0, 2, 1))        # [B, C, L]
        xt = jnp.pad(xt, ((0, 0), (0, 0), (self.kernel_size - 1, 0)))  # causal pad
        y = jax.lax.conv_general_dilated(
            xt, self.weight.value,
            window_strides=(1,), padding="VALID",
            feature_group_count=self.channels,   # depthwise
            dimension_numbers=("NCW", "OIW", "NCW"),
        )
        y = y + self.bias.value[None, :, None]
        return jnp.transpose(y, (0, 2, 1))       # [B, L, C]


class GatedRMSNorm(nnx.Module):
    """RMSNorm followed by a SiLU output gate (the 'Norm' + gate path in Fig. 1)."""

    def __init__(self, dim: int, *, eps: float = 1e-5, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x, gate):                 # both [..., dim]
        x = x.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        x = x * rms * self.weight.value
        return x * jax.nn.silu(gate.astype(jnp.float32))


class GatedDeltaNet2(nnx.Module):
    """Gated DeltaNet-2 recurrent token mixer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 16,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        num_v_heads: int | None = None,   # GQA: defaults to num_heads
        chunk_size: int = 64,
        conv_size: int = 4,
        expanded_erase: bool = False,     # erase gate in [0,2] (negative-eigenvalue variant)
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.H = num_heads
        self.Hv = num_v_heads or num_heads
        assert self.Hv % self.H == 0, "num_v_heads must be a multiple of num_heads"
        self.group = self.Hv // self.H
        self.dk = head_k_dim
        self.dv = head_v_dim
        self.chunk_size = chunk_size
        self.expanded_erase = expanded_erase

        k_proj_dim = self.H * self.dk     # q, k, b live on key-head axis
        v_proj_dim = self.Hv * self.dv    # v, w live on value-head axis

        self.q_proj = nnx.Linear(d_model, k_proj_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, k_proj_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, v_proj_dim, use_bias=False, rngs=rngs)
        self.b_proj = nnx.Linear(d_model, k_proj_dim, use_bias=True, rngs=rngs)   # erase gate
        self.w_proj = nnx.Linear(d_model, v_proj_dim, use_bias=True, rngs=rngs)   # write gate
        self.f_proj = nnx.Linear(d_model, k_proj_dim, use_bias=True, rngs=rngs)   # log-decay

        self.q_conv = ShortConv(k_proj_dim, conv_size, rngs=rngs)
        self.k_conv = ShortConv(k_proj_dim, conv_size, rngs=rngs)
        self.v_conv = ShortConv(v_proj_dim, conv_size, rngs=rngs)

        # log-decay parameters (Eq. 12 / 86): a per key-head, delta per key-channel.
        self.A_log = nnx.Param(jnp.zeros((self.H, self.dk)))      # 'a' in -exp(a)*softplus(.)
        # dt_bias init negative -> mild per-token decay at start (alpha near 1),
        # which keeps the cumulative decay (and gamma^{-1}) in a safe fp32 range.
        self.dt_bias = nnx.Param(jnp.full((self.H * self.dk,), -4.0))  # 'delta'

        self.gate_proj = nnx.Linear(d_model, v_proj_dim, use_bias=False, rngs=rngs)
        self.o_norm = GatedRMSNorm(self.dv, rngs=rngs)
        self.o_proj = nnx.Linear(v_proj_dim, d_model, use_bias=False, rngs=rngs)

    def _split_k(self, x, B, L):
        return x.reshape(B, L, self.H, self.dk).transpose(0, 2, 1, 3)   # [B,H,L,dk]

    def _split_v(self, x, B, L):
        return x.reshape(B, L, self.Hv, self.dv).transpose(0, 2, 1, 3)  # [B,Hv,L,dv]

    def __call__(self, x, initial_state=None):
        """x: [B, L, d_model]. Returns (out: [B, L, d_model], final_state: [B,Hv,dk,dv])."""
        B, L, _ = x.shape

        q = jax.nn.silu(self.q_conv(self.q_proj(x)))
        k = jax.nn.silu(self.k_conv(self.k_proj(x)))
        v = jax.nn.silu(self.v_conv(self.v_proj(x)))

        q = self._split_k(q, B, L)
        k = self._split_k(k, B, L)
        v = self._split_v(v, B, L)

        # L2 normalize q, k per head (App. D.2).
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

        # Log-decay in fp32 (Eq. 12 / App. D.1).
        f = self.f_proj(x).astype(jnp.float32) + self.dt_bias.value.astype(jnp.float32)
        f = self._split_k(f, B, L)
        a = jnp.exp(self.A_log.value.astype(jnp.float32))[None, :, None, :]  # [1,H,1,dk]
        g = -a * jax.nn.softplus(f)                                          # [B,H,L,dk] <= 0

        # Channel-wise gates.
        b = jax.nn.sigmoid(self.b_proj(x))
        b = self._split_k(b, B, L)
        if self.expanded_erase:
            b = 2.0 * b                                   # erase range [0,2]
        w = jax.nn.sigmoid(self.w_proj(x))
        w = self._split_v(w, B, L)

        # GQA: repeat key-side tensors across value-head groups (App. C.1).
        if self.group > 1:
            rep = lambda t: jnp.repeat(t, self.group, axis=1)
            q, k, g, b = rep(q), rep(k), rep(g), rep(b)

        if initial_state is None:
            initial_state = jnp.zeros((B, self.Hv, self.dk, self.dv), jnp.float32)

        O, final_state = chunkwise_gated_delta_rule_2(
            q, k, v, g, b, w, initial_state, chunk_size=self.chunk_size)

        # Gated RMSNorm + output projection.
        O = O.transpose(0, 2, 1, 3)                       # [B,L,Hv,dv]
        gate = self.gate_proj(x).reshape(B, L, self.Hv, self.dv)
        O = self.o_norm(O, gate).reshape(B, L, self.Hv * self.dv)
        out = self.o_proj(O.astype(x.dtype))
        return out, final_state
