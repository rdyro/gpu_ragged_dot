### GPU ragged_dot

Example implementation of GPU ragged dot in [JAX
Pallas](https://docs.jax.dev/en/latest/pallas/quickstart.html) (conforming to
[`jax.lax.ragged_dot`](https://github.com/jax-ml/jax/blob/713ea3caa17b506a6b485224c88f35e74ff6a297/jax/_src/lax/lax.py#L2531))
with hyperparameter auto-tuning via
[tune-jax](https://github.com/rdyro/tune-jax). Defaults chosen for Nvidia H100.
Includes backwards pass via [jax.custom_vjp](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html).

```python
@partial(jax.jit, static_argnums=list(range(4, 13)))
def ragged_dot(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    A_scale: jax.Array | None = None,  # [g, n] or None
    block_m: int = DEFAULT_BLOCK_M,  # unused, but used by the backwards pass
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    block_c: int = DEFAULT_BLOCK_C,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    """Ragged dot corresponding to jax.lax.ragged_dot (m, k) x (g, k, n) -> (m, n)"""
    ...
```
