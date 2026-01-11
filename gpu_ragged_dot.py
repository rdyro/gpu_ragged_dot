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
from collections import namedtuple
from functools import lru_cache, partial, wraps
from typing import Optional

import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

CompilerParams = getattr(plgpu, "CompilerParams", getattr(plgpu, "TritonCompilerParams", None))
if CompilerParams is None:
    raise RuntimeError("Could not find CompilerParams in jax.experimental.pallas.triton. "
                       "Upgrade to an newer JAX version please.")

__all__ = ["ragged_dot", "ragged_dot_ref", "trans_ragged_dot", "trans_ragged_dot_ref"]

# kernel ###############################################################################################################

DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 64

_cdiv = lambda a, b: pl.cdiv(a, jnp.array(b, jnp.int32))

def _gpu_ragged_dot_kernel(
    # inputs
    x_ref,  # [m, k]
    A_ref,  # [k, n]
    group_sizes_ref,  # [g]
    group_offset_ref,  # [g]
    A_scale_ref,  # [n]
    # outputs
    y_ref,  # [k, n]
    # static problem shapes
    m: int,
    k: int,
    n: int,
    g: int,
    # hyperparameters
    block_m: int,
    block_k: int,
    block_n: int,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: "jnp.dtype" = jnp.float32,
):
    pid = namedtuple("pid", ["i", "j"])(pl.program_id(0), pl.program_id(1))
    size = namedtuple("size", ["m", "k", "n"])(m, k, n)  # pack into named tuple to not lose indices later
    group_sz = group_sizes_ref[pid.i]
    compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

    dim_nums = (((1,), (0,)), ((), ()))
    _dot_fn = partial(jax.lax.dot_general, dimension_numbers=dim_nums, preferred_element_type= acc_dtype)

    @pl.when(group_sz > 0)
    def _():
        # row index into lhs and output
        start_ridx = jnp.where(pid.i == 0, 0, group_offset_ref[jnp.maximum(pid.i - 1, 0)])

        def outer_compute(r_offset, _):
            ridx = start_ridx + r_offset * block_m  # r_offset is 0,1,2,... need to map it to actual row indices
            lhs_rows_mask = (r_offset * block_m + jnp.arange(block_m)) < group_sz
            lhs_rows_idx = pl.ds(ridx, block_m)
            rhs_cols_idx = pl.ds(0, block_n)
            rhs_cols_mask = (block_n * pid.j + jnp.arange(block_n)) < size.n

            def inner_compute(k, acc):
                inner_idx = pl.ds(k * block_k, block_k)
                inner_mask = (k * block_k + jnp.arange(block_k)) < size.k
                x = plgpu.load(
                    x_ref.at[lhs_rows_idx, inner_idx], mask=lhs_rows_mask[:, None] & inner_mask[None, :], other=0
                )
                A = plgpu.load(
                    A_ref.at[inner_idx, rhs_cols_idx], mask=inner_mask[:, None] & rhs_cols_mask[None, :], other=0
                )
                return acc + _dot_fn(x.astype(compute_dtype), A.astype(compute_dtype)).astype(acc.dtype)

            acc = jnp.zeros((block_m, block_n), dtype=acc_dtype)
            acc = jax.lax.fori_loop(0, _cdiv(size.k, block_k), inner_compute, acc)
            if A_scale_ref is not None:
                acc = acc * plgpu.load(A_scale_ref.at[rhs_cols_idx], mask=rhs_cols_mask, other=0).astype(acc.dtype)
            acc = acc.astype(y_ref.dtype)
            plgpu.store(y_ref.at[lhs_rows_idx, rhs_cols_idx], acc, mask=lhs_rows_mask[:, None] & rhs_cols_mask[None, :])
            return None

        jax.lax.fori_loop(0, _cdiv(group_sz, block_m), outer_compute, None)

    @pl.when(pid.i == group_sizes_ref.size)
    def _():
        last_offset = group_offset_ref[-1]
        col_mask = (block_n * pid.j + jnp.arange(block_n)) < size.n

        def set_zero(i, _):
            row_mask = (last_offset + i * block_m + jnp.arange(block_m)) < size.m
            idx = (pl.ds(last_offset + i * block_m, block_m), pl.ds(0, block_n))
            mask = row_mask[:, None] & col_mask[None, :]
            plgpu.store(y_ref.at[*idx], jnp.zeros((block_m, block_n), dtype=y_ref.dtype), mask=mask)

        jax.lax.fori_loop(0, _cdiv(size.m - last_offset, block_m), set_zero, None)


# main routine #########################################################################################################


@partial(jax.jit, static_argnums=list(range(4, 12)))
def _gpu_ragged_dot(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    A_scale: jax.Array | None = None,  # [g, n] or None
    block_m: int = DEFAULT_BLOCK_M,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    """Compute grouped matmul on GPU via a Pallas lowering."""
    assert A.ndim == 3 and x.ndim == 2
    assert A.shape[:1] == group_sizes.shape
    if A_scale is not None:
        assert A_scale.shape == (A.shape[0], A.shape[-1])  # one scale per A column
    size = namedtuple("size", ["m", "k", "n", "g"])(x.shape[0], x.shape[1], A.shape[-1], A.shape[0])

    # normalize the block sizes for GPU
    block_m, block_k, block_n = [
        pl.next_power_of_2(min(b, s)) for b, s in zip([block_m, block_k, block_n], [size.m, size.k, size.n])
    ]
    block_k, block_n = max(block_k, 16), max(block_n, 16)

    group_offsets = jnp.cumsum(group_sizes, -1)  # we'll read 1 down always
    in_specs = [
        pl.BlockSpec((size.m, size.k), lambda i, j: (0, 0)),
        pl.BlockSpec((None, size.k, block_n), lambda i, j: (i, 0, j)),
        pl.BlockSpec((group_sizes.size,), lambda i, j: (0,)),
        pl.BlockSpec((group_offsets.size,), lambda i, j: (0,)),
        pl.BlockSpec((None, block_n), lambda i, j: (i, j)) if A_scale is not None else None,
    ]

    out_shape = jax.ShapeDtypeStruct((size.m, size.n), dtype=x.dtype)
    out_specs = pl.BlockSpec((size.m, block_n), lambda i, j: (0, j))
    grid = (size.g, pl.cdiv(size.n, block_n))
    block_sizes = dict(block_m=block_m, block_k=block_k, block_n=block_n)
    dtype_spec = dict(compute_dtype=compute_dtype, acc_dtype=acc_dtype)
    with jax.named_scope("gpu_ragged_dot"):
        y = pl.pallas_call(
            partial(_gpu_ragged_dot_kernel, **size._asdict(), **block_sizes, **dtype_spec),
            out_shape=out_shape,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            interpret=interpret,
            compiler_params=CompilerParams(num_warps=num_warps, num_stages=num_stages),
            name="gpu_ragged_dot",
        )(x, A, group_sizes, group_offsets, A_scale)
    return y


# reference implementation #############################################################################################


@partial(jax.jit, static_argnums=list(range(4, 12)))
def _gpu_ragged_dot_ref(
    x: jax.Array,
    A: jax.Array,
    group_sizes: jax.Array,
    A_scale: jax.Array | None = None,
    block_m: int = DEFAULT_BLOCK_M,  # unused, but used by the backwards pass
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    del block_m, block_n, block_k, interpret, compute_dtype, acc_dtype, num_warps, num_stages
    ret = jax.lax.ragged_dot(x, A, group_sizes)
    if A_scale is not None:
        indices = jnp.repeat(jnp.arange(A.shape[0]), group_sizes, total_repeat_length=x.shape[0])
        A_scale = jnp.take_along_axis(A_scale, indices[:, None], 0)
        ret = ret * A_scale
    return ret


# tests ################################################################################################################


def _gpu_trans_ragged_dot_kernel(
    # inputs
    x_ref,  # [m, k]
    y_ref,  # [k, n]
    group_sizes_ref,  # [g]
    group_offset_ref,  # [g]
    # outputs
    A_bar_ref,  # [g, k, n]
    # static problem shapes
    m: int,
    k: int,
    n: int,
    g: int,
    # hyperparameters
    block_m: int,
    block_n: int,
    block_k: int,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: "jnp.dtype" = jnp.float32,
):
    assert A_bar_ref.shape == (block_k, block_n)
    del g
    pid = namedtuple("pid", ["i", "r", "c"])(pl.program_id(0), pl.program_id(1), pl.program_id(2))
    size = namedtuple("size", ["m", "k", "n"])(m, k, n)  # pack into named tuple to not lose indices later
    group_sz = group_sizes_ref[pid.i]
    compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

    dim_nums = (((0,), (0,)), ((), ()))
    _dot_fn = partial(jax.lax.dot_general, dimension_numbers=dim_nums, preferred_element_type=acc_dtype)

    @pl.when(group_sz > 0)
    def _():
        # row index into lhs and output
        start_ridx = jnp.where(pid.i == 0, 0, group_offset_ref[jnp.maximum(pid.i - 1, 0)])

        k_idx = pl.ds(0, block_k)
        k_mask = (pid.r * block_m + jnp.arange(block_k)) < size.k
        cols_idx = pl.ds(0, block_n)
        cols_mask = (pid.c * block_n + jnp.arange(block_n)) < size.n

        def inner_compute(r_offset, acc):
            ridx = start_ridx + r_offset * block_m  # r_offset is 0,1,2,... need to map it to actual row indices
            xy_rows_mask = (r_offset * block_m + jnp.arange(block_m)) < group_sz
            xy_rows_idx = pl.ds(ridx, block_m)

            x = plgpu.load(x_ref.at[xy_rows_idx, k_idx], mask=xy_rows_mask[:, None] & k_mask[None, :], other=0)
            y = plgpu.load(y_ref.at[xy_rows_idx, cols_idx], mask=xy_rows_mask[:, None] & cols_mask[None, :], other=0)
            return acc + _dot_fn(x.astype(compute_dtype), y.astype(compute_dtype)).astype(acc.dtype)

        acc = jnp.zeros((block_k, block_n), dtype=acc_dtype)
        acc = jax.lax.fori_loop(0, _cdiv(group_sz, block_m), inner_compute, acc)
        acc = acc.astype(y_ref.dtype)
        plgpu.store(A_bar_ref.at[k_idx, cols_idx], acc, mask=k_mask[:, None] & cols_mask[None, :])


    @pl.when(group_sz == 0)
    def _():
        rmask = (pid.r * block_k + jnp.arange(block_k)) < size.k
        cmask = (pid.c * block_n + jnp.arange(block_n)) < size.n
        plgpu.store(A_bar_ref, jnp.zeros_like(A_bar_ref), mask=rmask[:, None] & cmask[None, :])


# main routine #########################################################################################################


@partial(jax.jit, static_argnums=list(range(3, 11)))
def _gpu_trans_ragged_dot(
    x: jax.Array,  # [m, k]
    y: jax.Array,  # [m, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,  # shape[0] of A_i tile (block_m, block_n)
    block_n: int = DEFAULT_BLOCK_N,  # shape[1] of A_i tile (block_m, block_n)
    block_k: int = DEFAULT_BLOCK_K,  # how many rows in the accumulation loop over block_m
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    """Compute grouped matmul on GPU via a Pallas lowering."""
    assert y.ndim == 2 and x.ndim == 2 and x.shape[0] == y.shape[0]
    (m, k), n = x.shape, y.shape[-1]
    size = namedtuple("size", ["m", "k", "n", "g"])(m, k, n, group_sizes.size)

    block_m, block_n = min(block_m, m), min(block_n, n)

    # normalize the block sizes for GPU
    block_m, block_k, block_n = [
        max(pl.next_power_of_2(min(b, s)), 16)
        for b, s in zip([block_m, block_k, block_n, block_k], [size.m, size.k, size.n])
    ]

    group_offsets = jnp.cumsum(group_sizes, -1)  # we'll read 1 down always
    in_specs = [
        pl.BlockSpec((size.m, block_k), lambda i, r, c: (0, r)),
        pl.BlockSpec((size.m, block_n), lambda i, r, c: (0, c)),
        pl.BlockSpec((size.g,), lambda i, r, c: (0,)),
        pl.BlockSpec((size.g,), lambda i, r, c: (0,)),
    ]

    out_shape = jax.ShapeDtypeStruct((size.g, size.k, size.n), dtype=x.dtype)
    out_specs = pl.BlockSpec((None, block_k, block_n), lambda i, r, c: (i, r, c))
    grid = (size.g, pl.cdiv(size.k, block_k), pl.cdiv(size.n, block_n))

    block_sizes = dict(block_m=block_m, block_k=block_k, block_n=block_n)
    dtype_spec = dict(compute_dtype=compute_dtype, acc_dtype=acc_dtype)
    y = pl.pallas_call(
        partial(_gpu_trans_ragged_dot_kernel, **size._asdict(), **block_sizes, **dtype_spec),
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=interpret,
        compiler_params=CompilerParams(num_warps=num_warps, num_stages=num_stages),
    )(x, y, group_sizes, group_offsets)
    return y


# reference implementation #############################################################################################


@partial(jax.jit, static_argnums=list(range(3, 11)))
def _gpu_trans_ragged_dot_ref(
    x: jax.Array,
    y: jax.Array,
    group_sizes: jax.Array,
    block_m: int = DEFAULT_BLOCK_M,  # shape[0] of A_i tile (block_m, block_n)
    block_n: int = DEFAULT_BLOCK_N,  # shape[1] of A_i tile (block_m, block_n)
    block_k: int = DEFAULT_BLOCK_K,  # how many rows in the accumulation loop over block_m
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    del block_m, block_n, block_k, interpret, compute_dtype, acc_dtype, num_warps, num_stages

    def scan_fn(i_offset, _):
        i, offset = i_offset
        accumulate = lambda j, acc: acc + x[j, :][:, None] @ y[j, :][None, :]
        zero = jnp.zeros((x.shape[-1], y.shape[-1]), dtype=x.dtype)
        Ai = jax.lax.fori_loop(offset, offset + group_sizes[i], accumulate, zero)
        return (i + 1, offset + group_sizes[i]), Ai

    return jax.lax.scan(scan_fn, (0, 0), None, length=group_sizes.shape[0])[1]


# autodiff rules #######################################################################################################


@lru_cache
def _get_ragged_dot(ref: bool = False, **kw):
    @wraps(_gpu_ragged_dot)
    @jax.custom_vjp
    def ragged_dot(x, A, group_sizes, A_scale):
        return (_gpu_ragged_dot_ref if ref else _gpu_ragged_dot)(x, A, group_sizes, A_scale, **kw)

    def ragged_dot_fwd(x, A, group_sizes, A_scale):
        return ragged_dot(x, A, group_sizes, A_scale), (x, A, group_sizes, A_scale)

    def ragged_dot_bwd(res, g):
        (x, A, group_sizes, A_scale), dy = res, g
        assert A_scale is None, "Differentiating ragged_dot with respect to A_scale is not supported"
        dx = ragged_dot(dy, A.swapaxes(-1, -2), group_sizes, None)
        dA = trans_ragged_dot(x, dy, group_sizes)
        return dx, dA, None, None

    ragged_dot.defvjp(ragged_dot_fwd, ragged_dot_bwd)

    # --------------------------------------------------------------------- #

    @wraps(_gpu_trans_ragged_dot)
    @jax.custom_vjp
    def trans_ragged_dot(x, A, group_sizes):
        return (_gpu_trans_ragged_dot_ref if ref else _gpu_trans_ragged_dot)(x, A, group_sizes, **kw)

    def trans_ragged_dot_fwd(x, y, group_sizes):
        return trans_ragged_dot(x, y, group_sizes), (x, y, group_sizes)

    def trans_ragged_dot_bwd(res, g):
        (x, y, group_sizes), dA = res, g
        dy = ragged_dot(x, dA, group_sizes)
        dx = ragged_dot(y, dA.swapaxes(-1, -2), group_sizes)
        return dx, dy, None

    trans_ragged_dot.defvjp(trans_ragged_dot_fwd, trans_ragged_dot_bwd)
    return ragged_dot, trans_ragged_dot


# exported methods #####################################################################################################


@partial(jax.jit, static_argnums=list(range(4, 11)))
def ragged_dot(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    A_scale: jax.Array | None = None,  # [g, n] or None
    block_m: int = DEFAULT_BLOCK_M,  # unused, but used by the backwards pass
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    """Ragged dot corresponding to jax.lax.ragged_dot (m, k) x (g, k, n) -> (m, n)"""
    kw = dict(block_m=block_m, block_k=block_k, block_n=block_n, interpret=interpret)
    kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
    _ragged_dot = _get_ragged_dot(ref=False, **kw)[0]
    return _ragged_dot(x, A, group_sizes, A_scale)


@partial(jax.jit, static_argnums=list(range(3, 11)))
def trans_ragged_dot(
    x: jax.Array,  # [m, k]
    y: jax.Array,  # [m, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,  # shape[0] of A_i tile (block_m, block_n)
    block_n: int = DEFAULT_BLOCK_N,  # shape[1] of A_i tile (block_m, block_n)
    block_k: int = DEFAULT_BLOCK_K,  # how many rows in the accumulation loop over block_m
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    """Tranposed ragged dot corresponding to transpose of ragged dot wrt A argument (m, k) x (m, n) -> (g, k, n)"""
    kw = dict(block_m=block_m, block_n=block_n, block_k=block_k, interpret=interpret)
    kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
    _trans_ragged_dot = _get_ragged_dot(ref=False, **kw)[1]
    return _trans_ragged_dot(x, y, group_sizes)


@partial(jax.jit, static_argnums=list(range(4, 12)))
def ragged_dot_ref(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    A_scale: jax.Array | None = None,  # [g, n] or None
    block_m: int = DEFAULT_BLOCK_M,  # unused, but used by the backwards pass
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    kw = dict(block_m=block_m, block_k=block_k, block_n=block_n, interpret=interpret)
    kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
    _ragged_dot = _get_ragged_dot(ref=True, **kw)[0]
    return _ragged_dot(x, A, group_sizes, A_scale)


@partial(jax.jit, static_argnums=list(range(3, 11)))
def trans_ragged_dot_ref(
    x: jax.Array,  # [m, k]
    y: jax.Array,  # [m, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,  # shape[0] of A_i tile (block_m, block_n)
    block_k: int = DEFAULT_BLOCK_K,  # how many rows in the accumulation loop over block_m
    block_n: int = DEFAULT_BLOCK_N,  # shape[1] of A_i tile (block_m, block_n)
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    kw = dict(block_m=block_m,  block_k=block_k, block_n=block_n, interpret=interpret)
    kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
    _trans_ragged_dot = _get_ragged_dot(ref=True, **kw)[1]
    return _trans_ragged_dot(x, y, group_sizes)


# a simple tuninig example #############################################################################################

if __name__ == "__main__":
    import tune_jax

    tune_jax.logger.setLevel("DEBUG")

    keys = iter(random.split(random.key(time.time_ns() % 2**31), 1024))
    m, k, n = (2 * 4096, 7168, 2048)
    g = 32
    print(f"FLOPS bound:  {2 * m * k * n / 35e12:.4e} s")
    print(f"HBM BW bound: {2 * (m * k + k * n + m * n) / 936e9:.4e} s")
    dtype = jnp.bfloat16
    x = random.normal(next(keys), (m, k), dtype=dtype)
    A = random.normal(next(keys), (g, k, n), dtype=dtype)

    group_sizes = jnp.round((m - 2) * jax.nn.softmax(1e0 * random.normal(next(keys), g), -1)).astype(jnp.int32)
    while jnp.sum(group_sizes) != m:
        idx = jnp.argmax(group_sizes)
        group_sizes = group_sizes.at[idx].set(group_sizes[idx] + (m - jnp.sum(group_sizes)))
    assert jnp.sum(group_sizes) <= m

    hyperparams = dict(
        block_m=[16, 32, 64, 128, 256],
        block_k=[16, 32, 64, 128, 256],
        block_n=[16, 32, 64, 128, 256],
        # block_m=32,
        # block_n=128,
        # block_k=64,
    )

    jax.config.update("jax_traceback_filtering", "off")

    def combined_fn(x, A, group_sizes, **kw):
        y = ragged_dot(x, A, group_sizes, **kw)
        dx, dA = jax.grad(lambda x_, A_, gs_: jnp.sum(ragged_dot(x_, A_, gs_, **kw)), argnums=(0, 1))(x, A, group_sizes)
        return jnp.sum(dA) + jnp.sum(dx) + jnp.sum(y)

    def combined_ref_fn(x, A, group_sizes, **kw):
        del kw
        y = jax.lax.ragged_dot(x, A, group_sizes)
        dx, dA = jax.grad(lambda x_, A_, gs_: jnp.sum(jax.lax.ragged_dot(x_, A_, gs_)), argnums=(0, 1))(
            x, A, group_sizes
        )
        return jnp.sum(dA) + jnp.sum(dx) + jnp.sum(y)

    r = jax.random.normal(jax.random.key(1), (m, n), dtype=dtype)
    dA = jax.vjp(lambda A: ragged_dot(x, A, group_sizes, block_m=32, block_k=128, block_n=64), A)[1](r)[0]
    # dA_ref = jax.vjp(lambda A: jax.lax.ragged_dot(x, A, group_sizes), A)[1](r)[0]
    #breakpoint()

    # fn = tune_jax.tune(jax.jit(ragged_dot, static_argnames=hyperparams.keys()), hyperparams=hyperparams, sample_num=1e9)
    # jax.block_until_ready(fn(x, A, group_sizes))
    # with jax.profiler.trace("/tmp/profiles"):
    #     for _ in range(3):
    #         jax.block_until_ready(fn(x, A, group_sizes))

    # combined_fn(x, A, group_sizes, **hyperparams)
    rhs = ragged_dot(x, A, group_sizes)
    trans_ragged_dot(x, rhs, group_sizes, block_m=32, block_k=128, block_n=64)

    ref_fn = jax.jit(lambda x, rhs, group_sizes: jax.vjp(lambda A: jax.lax.ragged_dot(x, A, group_sizes), A)[1](rhs)[0])
    dA_ref = ref_fn(x, rhs, group_sizes)

    fn = tune_jax.tune(jax.jit(trans_ragged_dot, static_argnames=hyperparams.keys()), hyperparams=hyperparams, sample_num=1e9)
    dA = jax.block_until_ready(fn(x, rhs, group_sizes))
    with jax.profiler.trace("/tmp/profiles"):
        for _ in range(3):
            jax.block_until_ready(fn(x, rhs, group_sizes))
    breakpoint()

    _ = jax.jit(combined_fn)(x, A, group_sizes)
    fn = tune_jax.tune(jax.jit(combined_fn, static_argnames=hyperparams.keys()), hyperparams=hyperparams, sample_num=1e9)
    o = jax.block_until_ready(fn(x, A, group_sizes))
    o_ref = jax.block_until_ready(jax.jit(combined_ref_fn)(x, A, group_sizes))

    breakpoint()

    with jax.profiler.trace("/tmp/profiles"):
        for _ in range(3):
            jax.block_until_ready(fn(x, A, group_sizes))
    breakpoint()

    fn_ = jax.jit(partial(ragged_dot, **fn.optimal_hyperparams))
    jax.block_until_ready(fn_(x, A, group_sizes))
    with jax.profiler.trace("/tmp/profiles"):
        for _ in range(3):
            jax.block_until_ready(fn_(x, A, group_sizes))
    # o = ragged_dot(x, A, group_sizes, **fn.optimal_hyperparams)
    # o2 = jax.lax.ragged_dot(x, A, group_sizes)
    # err = jnp.linalg.norm(o - o2, axis=-1) / jnp.maximum(jnp.linalg.norm(o2, axis=-1), 1e-7)
    # print(err)
    # print(jnp.max(err))
