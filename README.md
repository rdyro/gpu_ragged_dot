### GPU ragged_dot

Example implementation of GPU ragged dot in [JAX
Pallas](https://docs.jax.dev/en/latest/pallas/quickstart.html) (conforming to
[`jax.lax.ragged_dot`](https://github.com/jax-ml/jax/blob/713ea3caa17b506a6b485224c88f35e74ff6a297/jax/_src/lax/lax.py#L2531))
with hyperparameter auto-tuning via
[tune-jax](https://github.com/rdyro/tune-jax).

Defaults chosen to accomodate most post-Ampere GPUs (but could benefit from tuning for your input shape examples, see [tuning.py](./tuning.py)).

Includes backwards pass via [jax.custom_vjp](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html).

```python
from gpu_ragged_dot import ragged_dot

y = gpu_ragged_dot(lhs, rhs, group_sizes)
# or
y = gpu_ragged_dot(lhs, rhs, group_sizes, block_m=128, block_k=128, block_n=128)

# derivatives fully supported
y, vjp_fn = jax.vjp(partial(gpu_ragged_dot, block_m=128, block_k=128, block_n=128, group_sizes=group_sizes))(
    lhs, rhs
)
dlhs, drhs = vjp_fn(dout)
```

### Hyperparameters and Tuning

For a product of the form:
```
ragged_dot(lhs: bf16[m, k], rhs: bf16[g, k, n], group_sizes: int32[g]): bf16[m, n]
```

There are three main hyperparameters to tune:
```py
block_m: int   # tiling along rows of lhs
block_k: int   # tiling along the reduction/contraction axis in the matrix product in gmm
block_n: int   # tiling along the output columns
```

You can find optimal hyperparameters for you GPU by running:
```bash
pip install tune-jax
python3 tuning.py
```

### The signature

The main function signature of `ragged_dot` is:

```python
def ragged_dot(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,  # unused, but used by the backwards pass
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    trans_rhs: bool = False,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    """Ragged dot corresponding to jax.lax.ragged_dot (m, k) x (g, k, n) -> (m, n)"""
    ...
```

### Testing

There are some tests available as:
```bash
pytest tests.py
```
