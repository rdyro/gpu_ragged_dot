# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import jax
import jax.numpy as jnp
import tune_jax
from jax import random

from gpu_ragged_dot import gmm, tgmm

# a simple tuninig example #############################################################################################


def main():
    tune_jax.logger.setLevel("DEBUG")

    keys = iter(random.split(random.key(time.time_ns() % 2**31), 1024))
    m, k, n = (2 * 4096, 7168, 2048)
    g = 32
    print(f"FLOPS bound:  {2 * m * k * n / 35e12:.4e} s")
    print(f"HBM BW bound: {2 * (m * k + k * n + m * n) / 936e9:.4e} s")
    dtype = jnp.bfloat16
    lhs = random.normal(next(keys), (m, k), dtype=dtype)
    rhs = random.normal(next(keys), (g, k, n), dtype=dtype)
    dout = random.normal(next(keys), (m, n), dtype=dtype)


    group_sizes = jnp.round((m - 2) * jax.nn.softmax(1e0 * random.normal(next(keys), (g,)), -1)).astype(jnp.int32)
    while jnp.sum(group_sizes) != m:
        idx = jnp.argmax(group_sizes)
        group_sizes = group_sizes.at[idx].set(group_sizes[idx] + (m - jnp.sum(group_sizes)))
    assert jnp.sum(group_sizes) <= m

    gmm(lhs, rhs, group_sizes)

    hyperparams = dict(
        block_m=[32, 64, 128, 256],
        block_k=[32, 64, 128, 256],
        block_n=[32, 64, 128, 256],
    )

    err_fn = lambda x, y, axis=-1: jnp.linalg.norm(x - y, axis=axis) / jnp.maximum(jnp.linalg.norm(y, axis=axis), 1e-7)

    # tune the fwd function ############################################################################################
    fn = tune_jax.tune(jax.jit(gmm, static_argnames=hyperparams.keys()), hyperparams=hyperparams)
    o_ref = jax.lax.ragged_dot(lhs, rhs, group_sizes)
    o = jax.block_until_ready(fn(lhs, rhs, group_sizes))
    with jax.profiler.trace("/tmp/profiles"):
        for _ in range(3):
            jax.block_until_ready(fn(lhs, rhs, group_sizes))
    o_err = err_fn(o, o_ref)
    print("#" * 80)
    print(f"gmm err: {float(jnp.max(o_err)):.4e}")
    print("#" * 80)

    gmm_optimal_hyperparams = dict(fn.optimal_hyperparams)

    # tune the combined trans_ragged_dot ###############################################################################
    fn = tune_jax.tune(jax.jit(tgmm, static_argnames=hyperparams.keys()), hyperparams=hyperparams)
    drhs = jax.block_until_ready(fn(lhs, dout, group_sizes))
    drhs_ref = jax.vjp(lambda rhs: jax.lax.ragged_dot(lhs, rhs, group_sizes), rhs)[1](dout)[0]
    drhs_err = err_fn(drhs, drhs_ref, axis=(-1, -2))

    with jax.profiler.trace("/tmp/profiles"):
        for _ in range(3):
            jax.block_until_ready(fn(lhs, dout, group_sizes))
    print("#" * 80)
    print(f"tgmm err: {float(jnp.max(drhs_err)):.4e}")
    print("#" * 80)

    # tune combined ####################################################################################################
    def combined_fn(lhs, rhs, group_sizes, **kw):
        y = gmm(lhs, rhs, group_sizes, **kw)
        dlhs, drhs = jax.grad(lambda lhs_, rhs_, gs_: jnp.sum(gmm(lhs_, rhs_, gs_, **kw)), argnums=(0, 1))(
            lhs, rhs, group_sizes
        )
        return jnp.sum(drhs.astype(jnp.float32)) + jnp.sum(dlhs.astype(jnp.float32)) + jnp.sum(y.astype(jnp.float32))

    fn = tune_jax.tune(jax.jit(combined_fn, static_argnames=("block_m", "block_k", "block_n")), hyperparams=hyperparams)
    jax.block_until_ready(fn(lhs, rhs, group_sizes))
    with jax.profiler.trace("/tmp/profiles"):
        for _ in range(3):
            jax.block_until_ready(fn(lhs, rhs, group_sizes))

    combined_optimal_hyperparams = dict(fn.optimal_hyperparams)

    print("-" * 80)
    print("-" * 80)
    print("-" * 80)
    print("Summary:")
    print(f"gmm optimal hyperparameters:                     {gmm_optimal_hyperparams}")
    print(f"gmm and its derivatives optimal hyperparameters: {combined_optimal_hyperparams}")
    print("-" * 80)
    print("-" * 80)
    print("-" * 80)


if __name__ == "__main__":
    main()
