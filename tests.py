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

import itertools
import math
import os
import random as pyrandom
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import random
from jax.lax import RaggedDotDimensionNumbers
from scipy.special import softmax

from gpu_ragged_dot import ragged_dot, trans_ragged_dot

GMM_DIM_NUMS = RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())), lhs_ragged_dimensions=(0,), rhs_group_dimensions=(0,)
)
TGMM_DIM_NUMS = RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((0,), (0,)), ((), ())), lhs_ragged_dimensions=(0,), rhs_group_dimensions=()
)


def ragged_dot_ref(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, **kw):
    assert lhs.ndim == 2 and rhs.ndim == 3
    del kw
    return jax.lax.ragged_dot_general(lhs, rhs, group_sizes, ragged_dot_dimension_numbers=GMM_DIM_NUMS)


def trans_ragged_dot_ref(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, **kw):
    assert lhs.ndim == rhs.ndim == 2
    del kw
    return jax.lax.ragged_dot_general(lhs, rhs, group_sizes, ragged_dot_dimension_numbers=TGMM_DIM_NUMS)


def generate_group_sizes(key: jax.Array, g: int, target_m: int):
    seed = sum(int(x) << (32 * i) for i, x in enumerate(np.array(jax.random.key_data(key))))
    group_sizes = np.round(target_m * softmax(1e0 * np.random.default_rng(seed).normal(size=(g,)))).astype(np.int32)
    while np.sum(group_sizes) != target_m:
        idx = np.argmax(group_sizes)
        group_sizes[idx] = max(group_sizes[idx] + (target_m - np.sum(group_sizes)), 0)
    return jnp.array(group_sizes)


def _normal(key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype):
    seed = sum(int(x) << (32 * i) for i, x in enumerate(np.array(jax.random.key_data(key))))
    return jnp.array(np.random.default_rng(seed).normal(size=shape).astype(dtype))


def generate_inputs(key: jax.Array, m: int, k: int, n: int, g: int, dtype: jnp.dtype):
    keys = iter(jax.random.split(key, 1024))
    normal = _normal
    lhs = normal(next(keys), (m, k), dtype=dtype)
    rhs = normal(next(keys), (g, k, n), dtype=dtype)
    dout = normal(next(keys), (m, n), dtype=dtype)
    return lhs, rhs, dout


@partial(jax.jit, static_argnames=("axis",))
def err_fn(a: jax.Array, b: jax.Array, axis: int = -1, *, mask: jax.Array | None = None):
    err = jnp.linalg.norm(a - b, axis=axis) / jnp.maximum(jnp.linalg.norm(b, axis=axis), 1e-7)
    if mask is not None:
        return jnp.where(jnp.expand_dims(mask, range(mask.ndim, err.ndim)), err, 0)
    return err


def _sampled_product(samples, **kw):
    keys, vals = list(kw.keys()), list(kw.values())
    idx = list(range(math.prod(len(v) for v in vals)))
    pyrandom.shuffle(idx)
    params = [v for i, v in enumerate(itertools.product(*vals)) if i in idx[:samples]]
    params = [dict(zip(keys, vals)) for vals in params]
    return parameterized.parameters(*params)


cases_wrapper = _sampled_product(
    os.getenv("TEST_SAMPLES", 32),
    dtype=[jnp.bfloat16, jnp.float32],
    m=[128, 256, 171],
    k=[256, 512, 58],
    n=[64, 128, 77],
    g=[8, 16],
    block_m=[16, 32, 64],
    block_k=[16, 32, 64],
    block_n=[16, 32, 64],
    fill_in=[1.0, 0.7, 0.5],
    seed=[0, 1],
)


class GMMTests(parameterized.TestCase):
    def generate_inputs(self, seed, m, k, n, g, fill_in, dtype):
        keys = iter(jax.random.split(jax.random.key(seed), 1024))
        lhs, rhs, dout = generate_inputs(next(keys), m=m, k=k, n=n, g=g, dtype=dtype)
        if "gpu" not in list(lhs.devices())[0].platform.lower():
            self.skipTest("Test requires a GPU")
        target_m = max(min(round(m * fill_in), m), 0)
        group_sizes = generate_group_sizes(next(keys), g=g, target_m=target_m)
        assert group_sizes.size == rhs.shape[0]
        return lhs, rhs, dout, group_sizes, target_m

    @cases_wrapper
    def test_ragged_dot(self, dtype, m, k, n, g, block_m, block_k, block_n, fill_in, seed):
        lhs, rhs, dout, group_sizes, target_m = self.generate_inputs(seed, m, k, n, g, fill_in, dtype)
        y = jax.jit(partial(ragged_dot, block_m=block_m, block_k=block_k, block_n=block_n))(lhs, rhs, group_sizes)
        y_ref = jax.jit(ragged_dot_ref)(lhs, rhs, group_sizes)
        err = err_fn(y, y_ref, axis=-1, mask=jnp.arange(m) < target_m)
        eps = 5e-3 if jnp.dtype(dtype).name == "bfloat16" else 5e-4
        self.assertLess(jnp.max(err), eps)

    @cases_wrapper
    def test_ragged_dot_diff(self, dtype, m, k, n, g, block_m, block_k, block_n, fill_in, seed):
        lhs, rhs, dout, group_sizes, target_m = self.generate_inputs(seed, m, k, n, g, fill_in, dtype)
        o, vjp_fn = jax.vjp(
            partial(ragged_dot, block_m=block_m, block_k=block_k, block_n=block_n, group_sizes=group_sizes), lhs, rhs
        )
        o_ref, vjp_ref_fn = jax.vjp(partial(ragged_dot_ref, group_sizes=group_sizes), lhs, rhs)
        dlhs, drhs = vjp_fn(dout)
        dlhs_ref, drhs_ref = vjp_ref_fn(dout)
        o_err = err_fn(o, o_ref, axis=-1, mask=jnp.arange(m) < target_m)
        dlhs_err = err_fn(dlhs, dlhs_ref, axis=-1, mask=jnp.arange(m) < target_m)
        drhs_err = err_fn(drhs, drhs_ref, axis=(-1, -2), mask=group_sizes == 0)
        eps = 5e-3 if jnp.dtype(dtype).name == "bfloat16" else 5e-4
        self.assertLess(jnp.max(o_err), eps)
        self.assertLess(jnp.max(dlhs_err), eps)
        self.assertLess(jnp.max(drhs_err), eps)

    @cases_wrapper
    def test_ragged_dot_diff_with_trans_A(self, dtype, m, k, n, g, block_m, block_k, block_n, fill_in, seed):
        lhs, rhs, dout, group_sizes, target_m = self.generate_inputs(seed, m, k, n, g, fill_in, dtype)
        rhs = rhs.mT
        o, vjp_fn = jax.vjp(
            partial(
                ragged_dot, block_m=block_m, block_k=block_k, block_n=block_n, group_sizes=group_sizes, trans_rhs=True
            ),
            lhs,
            rhs,
        )
        o_ref, vjp_ref_fn = jax.vjp(partial(ragged_dot_ref, group_sizes=group_sizes), lhs, rhs.mT)
        dlhs, drhs = vjp_fn(dout)
        dlhs_ref, drhs_ref = vjp_ref_fn(dout)
        drhs_ref = drhs_ref.mT
        o_err = err_fn(o, o_ref, axis=-1, mask=jnp.arange(m) < target_m)
        dlhs_err = err_fn(dlhs, dlhs_ref, axis=-1, mask=jnp.arange(m) < target_m)
        drhs_err = err_fn(drhs, drhs_ref, axis=(-1, -2), mask=group_sizes == 0)
        eps = 5e-3 if jnp.dtype(dtype).name == "bfloat16" else 5e-4
        self.assertLess(jnp.max(o_err), eps)
        self.assertLess(jnp.max(dlhs_err), eps)
        self.assertLess(jnp.max(drhs_err), eps)

    @cases_wrapper
    def test_trans_ragged_dot(self, dtype, m, k, n, g, block_m, block_k, block_n, fill_in, seed):
        lhs, rhs, dout, group_sizes, target_m = self.generate_inputs(seed, m, k, n, g, fill_in, dtype)
        y = jax.jit(partial(trans_ragged_dot, block_m=block_m, block_k=block_k, block_n=block_n))(
            lhs, dout, group_sizes
        )
        y_ref = jax.jit(trans_ragged_dot_ref)(lhs, dout, group_sizes)
        err = err_fn(y, y_ref, axis=(-1, -2), mask=group_sizes == 0)
        eps = 2e-3 if jnp.dtype(dtype).name == "bfloat16" else 1e-5
        self.assertLess(jnp.max(err), eps)


if __name__ == "__main__":
    absltest.main()
