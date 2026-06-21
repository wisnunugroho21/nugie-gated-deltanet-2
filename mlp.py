import flax.nnx as nnx
import jax

# ---------------------------------------------------------------------------
# Multi-Layer Perceptron
# ---------------------------------------------------------------------------


class SwiGLUMLP(nnx.Module):
    """
    SwiGLU feed-forward network used after both the GDN-2 mixer and SWA.
    output = down( silu(gate(x)) ⊙ up(x) )
    """

    def __init__(self, dim: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.gate = nnx.Linear(dim, mlp_dim, use_bias=False, rngs=rngs)
        self.up = nnx.Linear(dim, mlp_dim, use_bias=False, rngs=rngs)
        self.down = nnx.Linear(mlp_dim, dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))
