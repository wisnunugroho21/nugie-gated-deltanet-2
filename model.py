import flax.nnx as nnx
import jax

from cell import HybridGDN2Cell
from transformer.rope import precompute_rope


class HybridGDN2LM(nnx.Module):
    """
    Hybrid Gated DeltaNet-2 Language Model.

    Quick start
    -----------
    >>> cfg   = config_debug()
    >>> model = HybridGDN2LM(cfg, rngs=nnx.Rngs(0))
    >>> ids   = jnp.ones((2, 64), dtype=jnp.int32)
    >>> logits = model(ids)          # (2, 64, vocab_size)
    >>> n     = model.num_params
    """

    def __init__(
        self,
        vocab_size: int = 32_000,
        dim: int = 2_048,
        num_heads: int = 16,
        num_cells: int = 8,
        mlp_dim: int = 8_192,
        chunk_size: int = 64,
        conv_kernel: int = 4,
        window_size: int = 2_048,
        max_seq_len: int = 4_096,
        rope_theta: float = 10_000.0,
        tie_embeddings: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        # ── Token embedding ────────────────────────────────────────────
        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs)

        # ── Repeated hybrid cells ──────────────────────────────────────
        # nnx.List required: plain Python lists are not tracked by NNX
        self.cells = nnx.List(
            [
                HybridGDN2Cell(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    chunk_size=chunk_size,
                    conv_kernel=conv_kernel,
                    window_size=window_size,
                    rngs=rngs,
                )
                for _ in range(num_cells)
            ]
        )

        # ── Final norm and LM head ─────────────────────────────────────
        self.norm_f = nnx.RMSNorm(dim, rngs=rngs)
        # If not weight-tied, create a separate linear head
        self.tie_embeddings = tie_embeddings
        if not tie_embeddings:
            self.lm_head = nnx.Linear(dim, vocab_size, use_bias=False, rngs=rngs)

        # ── RoPE buffers (non-trainable, excluded from optimizer) ──────
        # Precomputed once; sliced to actual seq_len at runtime.
        head_dim = dim // num_heads
        cos, sin = precompute_rope(head_dim, max_seq_len, rope_theta)
        self.rope_cos = nnx.Variable(cos)  # (max_seq_len, head_dim // 2)
        self.rope_sin = nnx.Variable(sin)

    # ── forward ──────────────────────────────────────────────────────────

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        """
        Args:
            input_ids: (B, L) int32 token ids.
                       L must be divisible by chunk_size.
        Returns:
            logits: (B, L, vocab_size)
        """
        x = self.embed(input_ids)  # (B, L, dim)
        cos = self.rope_cos[...]  # (max_seq_len, head_dim // 2)
        sin = self.rope_sin[...]

        for cell in self.cells:
            x = cell(x, cos, sin)

        x = self.norm_f(x)  # (B, L, dim)

        if self.tie_embeddings:
            # Share weights: logits = x @ embedding_matrixᵀ
            logits = x @ self.embed.embedding[...].T  # (B, L, vocab_size)
        else:
            logits = self.lm_head(x)

        return logits

    # ── utilities ────────────────────────────────────────────────────────

    @property
    def num_params(self) -> int:
        """Total trainable parameter count (excludes Buffer)."""
        state = nnx.state(self, nnx.Param)
        return int(sum(v.size for v in jax.tree.leaves(state)))

    def param_breakdown(self) -> dict:
        """Per-module trainable param counts for quick inspection."""
        state = nnx.state(self, nnx.Param)
        leaves_with_paths = jax.tree_util.tree_leaves_with_path(state)
        counts: dict[str, int] = {}
        for path, v in leaves_with_paths:
            top = str(path[0]) if path else "root"
            counts[top] = counts.get(top, 0) + v.size
        return counts
