"""
gated_deltanet2.py — Minimal PyTorch implementation of Gated DeltaNet-2.

Paper : "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
        https://arxiv.org/abs/2605.22791
        Authors: Ali Hatamizadeh, Yejin Choi, Jan Kautz (NVIDIA, 2026)
Official code: https://github.com/NVlabs/GatedDeltaNet-2

──────────────────────────────────────────────────────────────────────────────
Background: Linear Attention and the Delta Rule
──────────────────────────────────────────────────────────────────────────────
Standard softmax attention is O(T²) because it keeps the full KV cache.
Linear attention replaces that with a *fixed-size* recurrent state matrix
S ∈ R^{d_k × d_v}. Each new token (k_t, v_t) performs a rank-1 write:

    S ← S + k_t ⊗ v_t

and the output for query q_t is a rank-1 read:

    o_t = Sᵀ q_t

The *delta rule* (DeltaNet) first *erases* whatever was stored at key k_t
before writing new content, preventing interference:

    S ← S + k_t (v_t − k_tᵀ S)ᵀ = S + k_t ⊗ (v_t − k_tᵀ S)

Gated DeltaNet added a scalar gate β_t ∈ (0,1) shared between erase and write.
Kimi Delta Attention (KDA) further added channel-wise decay α_t.

──────────────────────────────────────────────────────────────────────────────
GDN-2 Core Idea: Decouple Erase and Write
──────────────────────────────────────────────────────────────────────────────
GDN-2 replaces the single shared scalar β_t with *two* independent
channel-wise gates — one for erasing, one for writing:

    S_t = (I − k_t (b_t ⊙ k_t)ᵀ) · Diag(α_t) · S_{t-1}  +  k_t (w_t ⊙ v_t)ᵀ

where:
  α_t ∈ (0,1)^{d_k}  — channel-wise DECAY applied to each key-dim of S
  b_t ∈ [0,1]^{d_k}  — channel-wise ERASE gate (key axis)
                         controls *which key coordinates* of S to read/erase
  w_t ∈ [0,1]^{d_v}  — channel-wise WRITE gate (value axis)
                         controls *which value coordinates* to commit

Expanded into a sequential update rule (per time step t):
  (1)  S ← Diag(α_t) · S              ← channel-wise decay (row-scale S)
  (2)  e_t = (b_t ⊙ k_t)ᵀ S           ← read old value via gated key → [d_v]
  (3)  v̂_t = (w_t ⊙ v_t) − e_t        ← delta: write new minus erased old
  (4)  S ← S + k_t ⊗ v̂_t              ← rank-1 update
  (5)  o_t = Sᵀ q_t                    ← read output → [d_v]

Special cases (strict generalisations):
  • b_t = β_t · 1, w_t = β_t · 1 (same scalar) → recovers KDA
  • additionally collapse α_t to a scalar             → recovers Gated DeltaNet
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Core recurrence
# ─────────────────────────────────────────────────────────────────────────────

def gated_delta_rule_2(
    q: torch.Tensor,                           # [B, T, H, d_k]  queries
    k: torch.Tensor,                           # [B, T, H, d_k]  keys
    v: torch.Tensor,                           # [B, T, H, d_v]  values
    g: torch.Tensor,                           # [B, T, H, d_k]  log-decay (<0)
    b: torch.Tensor,                           # [B, T, H, d_k]  erase gate ∈(0,1)
    w: torch.Tensor,                           # [B, T, H, d_v]  write gate ∈(0,1)
    initial_state: torch.Tensor | None = None, # [B, H, d_k, d_v]
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sequential (token-by-token) implementation of the GDN-2 recurrence.

    This is the reference implementation for clarity; the official code
    uses hardware-fused Triton kernels (fused_recurrent_gdn2 / chunk_gdn2)
    for speed. The math is identical.

    Returns:
        o           : [B, T, H, d_v]  output sequence
        final_state : [B, H, d_k, d_v]  last value of the recurrent state S
    """
    B, T, H, d_k = q.shape
    d_v = v.shape[-1]

    if scale is None:
        scale = d_k ** -0.5

    # L2-normalise q and k (stabilises the state magnitude; matches official code).
    q = F.normalize(q, p=2, dim=-1) * scale   # unit-norm then scaled
    k = F.normalize(k, p=2, dim=-1)            # unit-norm

    # Initialise the recurrent state S ∈ R^{d_k × d_v}.
    # We always keep S in float32 for numerical stability.
    if initial_state is not None:
        S = initial_state.to(torch.float32).clone()
    else:
        S = torch.zeros(B, H, d_k, d_v, device=q.device, dtype=torch.float32)

    outputs = []

    for t in range(T):
        # Slice token t; cast to float32 for state arithmetic.
        q_t = q[:, t].float()   # [B, H, d_k]
        k_t = k[:, t].float()   # [B, H, d_k]
        v_t = v[:, t].float()   # [B, H, d_v]
        g_t = g[:, t].float()   # [B, H, d_k]  log-decay < 0
        b_t = b[:, t].float()   # [B, H, d_k]  erase gate
        w_t = w[:, t].float()   # [B, H, d_v]  write gate

        # ── Step 1: Channel-wise decay ───────────────────────────────────────
        # α_t = exp(g_t) ∈ (0,1) because g_t < 0.
        # Scale each *row* of S (each row corresponds to one key dimension).
        alpha = g_t.exp()                          # [B, H, d_k]
        S = S * alpha.unsqueeze(-1)                # [B, H, d_k, d_v]

        # ── Step 2: Erase — read what is stored at the gated key ─────────────
        # Gated key selects which key-axis components to erase from S.
        bk_t = b_t * k_t                           # [B, H, d_k]  b ⊙ k

        # e_t = (b_t ⊙ k_t)ᵀ S  →  d_v-dimensional "old value" at that key
        # Equivalent to: for each value dim v, e[v] = sum_k bk_t[k] * S[k, v]
        e_t = torch.einsum("bhk, bhkv -> bhv", bk_t, S)   # [B, H, d_v]

        # ── Step 3: Delta write — new gated value minus erased old value ──────
        # Compared with the plain delta rule (v̂ = v_t − e_t), GDN-2 applies
        # a channel-wise write gate on the value axis before writing.
        v_hat = w_t * v_t - e_t                    # [B, H, d_v]   (w ⊙ v) − e

        # ── Step 4: Rank-1 state update ──────────────────────────────────────
        # Add the outer product k_t ⊗ v̂_t to S.
        # k_t[b,h,k] * v_hat[b,h,v] → contributes to S[b,h,k,v]
        S = S + torch.einsum("bhk, bhv -> bhkv", k_t, v_hat)

        # ── Step 5: Read output with query ────────────────────────────────────
        # o_t = Sᵀ q_t  →  for each value dim v: o[v] = sum_k S[k,v] * q_t[k]
        o_t = torch.einsum("bhkv, bhk -> bhv", S, q_t)    # [B, H, d_v]

        outputs.append(o_t)

    # Reassemble the time axis: list of [B,H,d_v] → [B,T,H,d_v]
    o = torch.stack(outputs, dim=1).to(v.dtype)

    return o, S


# ─────────────────────────────────────────────────────────────────────────────
# Full GDN-2 Layer (nn.Module)
# ─────────────────────────────────────────────────────────────────────────────

class GatedDeltaNet2(nn.Module):
    """
    Gated DeltaNet-2 token-mixing layer.

    Drop-in replacement for multi-head attention in a transformer block.
    Given input [B, T, d_model]:
      1. Project → q, k, v, decay g, erase gate b, write gate w.
      2. Run the GDN-2 recurrence (O(T) instead of O(T²)).
      3. Apply gated RMSNorm on the output.
      4. Project back to [B, T, d_model].

    Args:
        d_model   : model dimension (input and output width)
        num_heads : number of parallel recurrent heads
        d_k       : key/query dimension per head
        d_v       : value dimension per head (defaults to d_k)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_k: int = 64,
        d_v: int | None = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v if d_v is not None else d_k

        total_k = num_heads * d_k       # total key/query dim (all heads)
        total_v = num_heads * self.d_v  # total value dim (all heads)

        # ── q / k / v projections ─────────────────────────────────────────────
        self.q_proj = nn.Linear(d_model, total_k, bias=False)
        self.k_proj = nn.Linear(d_model, total_k, bias=False)
        self.v_proj = nn.Linear(d_model, total_v, bias=False)

        # ── Channel-wise decay (parameterisation from official code) ──────────
        # We want a *negative* per-channel log-decay g_t ∈ (-∞, 0) so that
        # α_t = exp(g_t) ∈ (0,1) is a proper decay factor.
        #
        # Formula: g_t = -exp(A_log) · softplus(f_proj(x) + dt_bias)
        #   • A_log (per head, learnable): sets a base decay rate per head.
        #     Initialised in [1, 16] so exp(A_log) gives a moderate rate.
        #   • f_proj(x): input-dependent time step Δt (per channel).
        #   • dt_bias (per channel, learnable): shifts Δt; initialised so
        #     the initial decay is slow (Δt ≈ 0.001 to 0.1).
        #   • softplus ensures the product is strictly positive → g_t < 0.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_heads).uniform_(1, 16))
        )
        self.A_log._no_weight_decay = True    # keep out of weight-decay group

        self.f_proj = nn.Linear(d_model, total_k, bias=False)

        # Initialise dt_bias via the softplus inverse so initial Δt ∈ (0.001, 0.1)
        dt_init = torch.exp(
            torch.rand(total_k) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        # softplus_inv(x) = x + log(1 - exp(-x))  for x > 0
        self.dt_bias = nn.Parameter(dt_init + torch.log(-torch.expm1(-dt_init)))
        self.dt_bias._no_weight_decay = True

        # ── GDN-2 decoupled erase and write gates ─────────────────────────────
        # b_proj → erase gate b_t ∈ (0,1)^{total_k}: key-axis selectivity
        # w_proj → write gate w_t ∈ (0,1)^{total_v}: value-axis selectivity
        # Both use sigmoid to land in (0,1). This is the GDN-2 contribution:
        # prior methods tied these two roles to one scalar β_t.
        self.b_proj = nn.Linear(d_model, total_k, bias=False)
        self.w_proj = nn.Linear(d_model, total_v, bias=False)

        # ── Gated output: RMSNorm(o) × SiLU(gate) ────────────────────────────
        # Matches FusedRMSNormSwishGate in the official code.
        # RMSNorm stabilises the recurrent output; the SiLU gate adds
        # input-dependent amplitude control before the final projection.
        self.g_proj = nn.Linear(d_model, total_v, bias=True)
        self.o_norm = nn.RMSNorm(self.d_v, eps=1e-5)
        self.o_proj = nn.Linear(total_v, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform initialisation for all linear layers (as in official code)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=2 ** -2.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,                            # [B, T, d_model]
        initial_state: torch.Tensor | None = None,  # [B, H, d_k, d_v]
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x             : input token sequence [B, T, d_model]
            initial_state : optional prior recurrent state [B, H, d_k, d_v]
                            (used for incremental / stateful decoding)
            return_state  : also return the final state if True

        Returns:
            o            : output sequence [B, T, d_model]
            final_state  : (only if return_state=True) [B, H, d_k, d_v]
        """
        B, T, _ = x.shape
        H, d_k, d_v = self.num_heads, self.d_k, self.d_v

        # ── Project and activate q, k, v ─────────────────────────────────────
        # SiLU activation on the projections (as in the official no-conv path).
        q = F.silu(self.q_proj(x)).view(B, T, H, d_k)
        k = F.silu(self.k_proj(x)).view(B, T, H, d_k)
        v = F.silu(self.v_proj(x)).view(B, T, H, d_v)

        # ── Channel-wise log-decay g ──────────────────────────────────────────
        # Compute in float32 for stability. A per-head rate is broadcast to all
        # d_k channels of that head via repeat_interleave.
        A = self.A_log.float().exp()                        # [H]  base decay rate
        A_per_ch = A.repeat_interleave(d_k)                 # [H*d_k]

        # Input-dependent Δt: softplus keeps it positive; dt_bias shifts init.
        delta_t = F.softplus(
            self.f_proj(x).float() + self.dt_bias           # [B, T, H*d_k]
        )

        # g_t < 0 so exp(g_t) ∈ (0,1) is the per-channel decay multiplier.
        g = -(A_per_ch * delta_t)                           # [B, T, H*d_k]
        g = g.view(B, T, H, d_k)

        # ── GDN-2 decoupled gates ─────────────────────────────────────────────
        # sigmoid maps to (0,1): 0 = "gate closed", 1 = "gate fully open"
        b = self.b_proj(x).sigmoid().view(B, T, H, d_k)    # erase gate
        w = self.w_proj(x).sigmoid().view(B, T, H, d_v)    # write gate

        # ── Run the GDN-2 recurrence ──────────────────────────────────────────
        o, final_state = gated_delta_rule_2(
            q=q, k=k, v=v, g=g, b=b, w=w,
            initial_state=initial_state,
        )
        # o: [B, T, H, d_v]

        # ── Gated output normalisation ────────────────────────────────────────
        # RMSNorm stabilises the recurrent output per head; the SiLU gate
        # applies learned input-dependent scaling. Equivalent to the
        # FusedRMSNormSwishGate in the official code.
        gate = self.g_proj(x).view(B, T, H, d_v)           # [B, T, H, d_v]
        o = self.o_norm(o) * F.silu(gate)                   # [B, T, H, d_v]

        # ── Project back to model dimension ───────────────────────────────────
        o = o.reshape(B, T, H * d_v)
        o = self.o_proj(o)                                  # [B, T, d_model]

        return (o, final_state) if return_state else o


# ─────────────────────────────────────────────────────────────────────────────
# GDN-2 Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class GDN2Block(nn.Module):
    """
    One transformer-style block with GDN-2 as the token mixer.

    Layout: Pre-norm → GDN-2 → residual  →  Pre-norm → FFN → residual
    (same structure as GPT-2, with GDN-2 replacing multi-head attention)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int | None = None,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = GatedDeltaNet2(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))   # token mixing with residual
        x = x + self.ffn(self.norm2(x))     # channel mixing with residual
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity checks
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T = 2, 16         # batch size, sequence length
    d_model = 256
    num_heads, d_k, d_v = 4, 32, 32

    print("=" * 60)
    print("1. Testing core recurrence: gated_delta_rule_2")
    print("=" * 60)

    q = torch.randn(B, T, num_heads, d_k)
    k = torch.randn(B, T, num_heads, d_k)
    v = torch.randn(B, T, num_heads, d_v)
    # g must be < 0 (log-decay); create via -softplus so the constraint is obvious
    g = -F.softplus(torch.randn(B, T, num_heads, d_k))
    b = torch.rand(B, T, num_heads, d_k)    # erase gate ∈ (0,1)
    w = torch.rand(B, T, num_heads, d_v)    # write gate ∈ (0,1)

    o_full, state_full = gated_delta_rule_2(q, k, v, g, b, w)
    print(f"  Batch output shape : {o_full.shape}")    # [B, T, H, d_v]
    print(f"  Final state  shape : {state_full.shape}")  # [B, H, d_k, d_v]

    # Verify incremental (token-by-token) decoding matches batch output.
    # This is how you would run it at inference time: pass `initial_state`
    # from the previous step to continue the recurrence.
    S = None
    for t in range(T):
        _, S = gated_delta_rule_2(
            q[:, t:t+1], k[:, t:t+1], v[:, t:t+1],
            g[:, t:t+1], b[:, t:t+1], w[:, t:t+1],
            initial_state=S,
        )
    assert torch.allclose(state_full, S, atol=1e-5), "Incremental state mismatch!"
    print("  Incremental decoding matches batch recurrence ✓")

    print()
    print("=" * 60)
    print("2. Testing GatedDeltaNet2 layer")
    print("=" * 60)

    layer = GatedDeltaNet2(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v)
    x = torch.randn(B, T, d_model)

    out = layer(x)
    print(f"  Input  shape : {x.shape}")
    print(f"  Output shape : {out.shape}")
    assert out.shape == (B, T, d_model), "Output shape mismatch!"
    print("  Shape check ✓")

    # Test stateful inference: process the same input in two halves.
    out1_first, state1 = layer(x[:, :T//2], return_state=True)
    out1_second, _ = layer(x[:, T//2:], initial_state=state1, return_state=True)
    out1_incremental = torch.cat([out1_first, out1_second], dim=1)
    out1_full = layer(x)
    assert torch.allclose(out1_incremental, out1_full, atol=1e-5), "Stateful inference mismatch!"
    print("  Stateful (split) inference matches single-pass ✓")

    print()
    print("=" * 60)
    print("3. Testing GDN2Block")
    print("=" * 60)

    block = GDN2Block(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v)
    out = block(x)
    print(f"  Output shape : {out.shape}")
    assert out.shape == x.shape, "Block output shape mismatch!"
    print("  Block test ✓")

    print()
    print("=" * 60)
    print("Parameter counts")
    print("=" * 60)
    layer_params = sum(p.numel() for p in layer.parameters())
    block_params = sum(p.numel() for p in block.parameters())
    print(f"  GatedDeltaNet2 : {layer_params:,} parameters")
    print(f"  GDN2Block      : {block_params:,} parameters")

    print()
    print("All tests passed ✓")
