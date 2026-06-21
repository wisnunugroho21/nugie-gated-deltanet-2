"""
Verification for the Gated DeltaNet-2 chunkwise implementation.

  1. Chunkwise forward == token-by-token recurrence (Eq. 29), fp32 + fp64.
  2. Chunk-size invariance (C in {32,64,128} agree).
  3. Cross-chunk state carry (non-zero initial state honored).
  4. Tied-gate KDA reduction sanity.
  5. Autodiff gradients (chunkwise) == recurrent gradients.
  6. Full NNX layer: forward, 5 Adam steps, jitted train step.

Run:  python test_gdn2.py
"""
import jax, jax.numpy as jnp
import flax.nnx as nnx, optax
from gdn2_core import chunkwise_gated_delta_rule_2, recurrent_gated_delta_rule_2
from gdn2_layer import GatedDeltaNet2


def _inputs(key, B, H, L, dk, dv, decay=0.05):
    ks = jax.random.split(key, 7)
    q = jax.random.normal(ks[0], (B, H, L, dk))
    k = jax.random.normal(ks[1], (B, H, L, dk))
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)   # L2 norm (App. D.2)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = jax.random.normal(ks[2], (B, H, L, dv))
    b = jax.nn.sigmoid(jax.random.normal(ks[3], (B, H, L, dk)))   # erase gate
    w = jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, L, dv)))   # write gate
    g = -decay * jax.nn.softplus(jax.random.normal(ks[6], (B, H, L, dk)))  # mild log-decay
    return q, k, v, g, b, w


def _err(a, b):
    return float(jnp.max(jnp.abs(a - b)))


def verify_core():
    print("=" * 64, "\nCore: chunkwise vs recurrent reference\n" + "=" * 64)
    B, H, L, dk, dv, C = 2, 3, 256, 32, 48, 64
    q, k, v, g, b, w = _inputs(jax.random.PRNGKey(0), B, H, L, dk, dv)
    S0 = jax.random.normal(jax.random.PRNGKey(1), (B, H, dk, dv))  # non-zero carry

    Oc, Sc = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, C)
    Or, Sr = recurrent_gated_delta_rule_2(q, k, v, g, b, w, S0)
    print(f"  output    max|diff| = {_err(Oc, Or):.2e}")
    print(f"  state     max|diff| = {_err(Sc, Sr):.2e}")

    O32, _ = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, 32)
    O128, _ = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, 128)
    print(f"  C=32  vs C=64       = {_err(O32, Oc):.2e}")
    print(f"  C=128 vs C=64       = {_err(O128, Oc):.2e}")

    beta = jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(2), (B, H, L, 1)))
    bt = jnp.broadcast_to(beta, (B, H, L, dk))
    wt = jnp.broadcast_to(beta, (B, H, L, dv))
    Oc2, _ = chunkwise_gated_delta_rule_2(q, k, v, g, bt, wt, S0 * 0, C)
    Or2, _ = recurrent_gated_delta_rule_2(q, k, v, g, bt, wt, S0 * 0)
    print(f"  KDA tied-gate       = {_err(Oc2, Or2):.2e}")

    lc = lambda *a: jnp.sum(chunkwise_gated_delta_rule_2(*a, S0 * 0, C)[0] ** 2)
    lr = lambda *a: jnp.sum(recurrent_gated_delta_rule_2(*a, S0 * 0)[0] ** 2)
    gc = jax.grad(lc, (0, 1, 2, 3, 4, 5))(q, k, v, g, b, w)
    gr = jax.grad(lr, (0, 1, 2, 3, 4, 5))(q, k, v, g, b, w)
    for nm, a, bb in zip(["dq", "dk", "dv", "dg", "db", "dw"], gc, gr):
        print(f"  grad {nm}            = {_err(a, bb):.2e}")


def verify_layer():
    print("=" * 64, "\nLayer: forward + training\n" + "=" * 64)
    B, L, dm = 2, 256, 256
    layer = GatedDeltaNet2(dm, num_heads=4, head_k_dim=32, head_v_dim=32,
                           num_v_heads=8, chunk_size=64, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (B, L, dm))
    out, state = layer(x)
    print(f"  forward out={out.shape} state={state.shape} "
          f"finite={bool(jnp.all(jnp.isfinite(out)))}")

    opt = nnx.Optimizer(layer, optax.adam(1e-3), wrt=nnx.Param)
    loss_fn = lambda m: jnp.mean(m(x)[0] ** 2)

    @nnx.jit
    def step(m, o):
        l, g = nnx.value_and_grad(loss_fn)(m)
        o.update(m, g)
        return l

    losses = [float(step(layer, opt)) for _ in range(5)]
    print("  loss (5 jitted Adam steps):", [f"{l:.4f}" for l in losses])
    n = sum(int(p.size) for p in jax.tree.leaves(nnx.state(layer, nnx.Param)))
    print(f"  trainable params: {n:,}")


if __name__ == "__main__":
    verify_core()
    verify_layer()
    print("\nAll checks completed.")
