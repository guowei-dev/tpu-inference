# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark per-DMA latency of the fabrics a kernel can move data over.

Each row issues N serially dependent DMAs inside one Pallas kernel and fits
total time against N, so the slope is the per-DMA latency while the kernel
launch, host dispatch and in-kernel barrier all sit in the intercept. Chain
lengths are sized per row (see latency_us) to land where this wall clock tracks
the profiler's module device time within about a percent.

A link is measured by exchanging a tile between two devices every iteration and
waiting on both directions, so iteration i+1 cannot start before iteration i
has landed: the slope is the one-way latency. Which link that is follows from
the device pair alone -- same chip is the on-chip core bridge, adjacent chips
one ICI hop, diagonal chips two.

Run on a TPU host:
    python scripts/benchmarking/kernels/benchmark_link_latency.py
"""

import argparse
import functools
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

AXIS = "pair"
TILE = (8, 128)  # 4 KiB in f32: latency-, not bandwidth-bound
HBM_ROWS = (256 << 20) // (TILE[0] * TILE[1] * 4)


def link_pairs():
    """(label, device_a, device_b) per fabric, from the live topology."""
    chips = {}
    for d in jax.devices():
        chips.setdefault(tuple(d.coords), {})[d.core_on_chip] = d
    origin, *others = sorted(chips)
    extent = [max(c[i] for c in chips) + 1 for i in range(len(origin))]

    def hops(a, b):
        return sum(
            min(abs(a[i] - b[i]), extent[i] - abs(a[i] - b[i]))
            for i in range(len(a)))

    pairs = [("core bridge  same chip, core 0 -> 1", chips[origin][0],
              chips[origin][1])]
    near = [c for c in others if hops(origin, c) == 1]
    far = [c for c in others if hops(origin, c) == 2]
    if near:
        pairs.append((f"ICI 1 hop    {origin} -> {near[0]}", chips[origin][0],
                      chips[near[0]][0]))
        pairs.append((f"ICI 1 hop    {origin} -> {near[0]}, other core",
                      chips[origin][0], chips[near[0]][1]))
    if far:
        pairs.append((f"ICI 2 hops   {origin} -> {far[0]}, diagonal",
                      chips[origin][0], chips[far[0]][0]))
    return pairs


def _pair_barrier(partner):
    """Line the two devices up before the timed chain. The second round stops
    a device that has already re-entered the kernel from releasing the next
    call's barrier."""

    @functools.partial(pl.run_scoped,
                       first=pltpu.SemaphoreType.REGULAR,
                       second=pltpu.SemaphoreType.REGULAR)
    def _(first, second):
        for sem in (first, second):
            pl.semaphore_signal(sem,
                                inc=1,
                                device_id=(partner, ),
                                device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(sem, 1)


def _exchange_kernel(x_ref, o_ref, send, recv, send_sem, recv_sem, *, n_iters):
    partner = jnp.int32(1) - lax.axis_index(AXIS)
    send[...] = x_ref[...]
    _pair_barrier(partner)

    @pl.loop(0, n_iters)
    def _(_i):
        copy = pltpu.make_async_remote_copy(
            send,
            recv,
            send_sem,
            recv_sem,
            device_id=(partner, ),
            device_id_type=pl.DeviceIdType.MESH)
        copy.start()
        copy.wait()  # both directions, so the next iteration cannot overlap

    o_ref[...] = recv[...]


def build_exchange(mesh, n_iters):
    """N chained tile exchanges between the mesh's two devices."""
    call = pl.pallas_call(
        functools.partial(_exchange_kernel, n_iters=n_iters),
        out_shape=jax.ShapeDtypeStruct(TILE, jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
        scratch_shapes=(
            pltpu.VMEM(TILE, jnp.float32),  # send
            pltpu.VMEM(TILE, jnp.float32),  # recv
            pltpu.SemaphoreType.DMA,  # send_sem
            pltpu.SemaphoreType.DMA,  # recv_sem
        ),
    )
    return jax.jit(
        jax.shard_map(call,
                      mesh=mesh,
                      in_specs=P(AXIS),
                      out_specs=P(AXIS),
                      check_vma=False))


def _hbm_read_kernel(x_ref, o_ref, buf, sem, *, n_iters):

    def body(i, acc):
        copy = pltpu.make_async_copy(x_ref.at[lax.rem(i, jnp.int32(HBM_ROWS))],
                                     buf, sem)
        copy.start()
        copy.wait()
        # Reading the row keeps the copy alive: with no consumer, every copy
        # but the last is a dead store and the loop times itself, not HBM.
        return acc + buf[0, 0]

    total = pl.loop(0, n_iters, init_carry=jnp.float32(0))(body)
    o_ref[...] = jnp.full(o_ref.shape, total, jnp.float32)


def build_hbm_read(n_iters):
    """N chained HBM -> VMEM copies of one tile.

    The source spans HBM_ROWS so it cannot fit VMEM: a small one is prefetched
    there and the kernel then measures VMEM, since a ref's memory space says
    where the kernel reads from, not where the operand lives.
    """
    return jax.jit(
        pl.pallas_call(
            functools.partial(_hbm_read_kernel, n_iters=n_iters),
            out_shape=jax.ShapeDtypeStruct(TILE, jnp.float32),
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
            scratch_shapes=(
                pltpu.VMEM(TILE, jnp.float32),  # buf
                pltpu.SemaphoreType.DMA,  # sem
            ),
        ))


def _hbm_write_kernel(x_ref, o_ref, buf, sem, *, n_iters):
    buf[...] = x_ref[...]

    @pl.loop(0, n_iters)
    def _(i):
        copy = pltpu.make_async_copy(buf,
                                     o_ref.at[lax.rem(i, jnp.int32(HBM_ROWS))],
                                     sem)
        copy.start()
        copy.wait()  # waits for the data to land in HBM


def build_hbm_write(n_iters):
    """N chained VMEM -> HBM copies of one tile."""
    return jax.jit(
        pl.pallas_call(
            functools.partial(_hbm_write_kernel, n_iters=n_iters),
            out_shape=jax.ShapeDtypeStruct((HBM_ROWS, ) + TILE, jnp.float32),
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.VMEM(TILE, jnp.float32),  # buf
                pltpu.SemaphoreType.DMA,  # sem
            ),
        ))


def _median_us(fn, operand, *, reps, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(operand))
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn(operand))
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def latency_us(build, operand, *, reps, warmup, target_us, check=None):
    """Fit time against chain length; returns (us per DMA, longest chain).

    A chain too short is swamped by the constant per-call overhead, and one too
    long is timed through a host-side wait whose cost stops being constant, so
    the sweep is sized from a coarse two-point probe to land between the two
    whatever the latency of the fabric under it.
    """
    probe = [(n, _median_us(build(n), operand, reps=3, warmup=1))
             for n in (64, 512)]
    per_dma = max((probe[1][1] - probe[0][1]) / (probe[1][0] - probe[0][0]),
                  1e-3)
    n_max = int(min(4096, max(256, target_us / per_dma)))
    lengths = (n_max // 8, n_max // 2, n_max)

    times = []
    for n in lengths:
        fn = build(n)
        out = jax.block_until_ready(fn(operand))
        if check is not None:
            np.testing.assert_array_equal(np.asarray(out), check(n))
        times.append(_median_us(fn, operand, reps=reps, warmup=warmup))
    slope, _ = np.polyfit(np.asarray(lengths, float), np.asarray(times), 1)
    return slope, n_max


def hbm_source():
    """HBM_ROWS tiles whose row r holds the value r."""
    rows = np.broadcast_to(
        np.arange(HBM_ROWS, dtype=np.float32)[:, None, None],
        (HBM_ROWS, ) + TILE)
    return jax.device_put(np.ascontiguousarray(rows), jax.devices()[0])


def hbm_read_total(n_iters):
    """What the read chain must accumulate -- exact in f32 at these lengths, so
    an elided copy shows up as a mismatch rather than a fast row."""
    total = float(sum(i % HBM_ROWS for i in range(n_iters)))
    return np.full(TILE, total, np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps",
                        type=int,
                        default=20,
                        help="timed calls per chain length (median)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--target-us",
                        type=float,
                        default=2000.0,
                        help="run time of the longest chain in each fit")
    args = parser.parse_args()

    assert jax.devices()[0].platform == "tpu", "requires a TPU host"
    if jax.device_count() < 2:
        parser.error(f"needs >= 2 devices, found {jax.device_count()}")

    cells = []
    for label, device_a, device_b in link_pairs():
        mesh = Mesh(np.asarray([device_a, device_b]), (AXIS, ))
        tile = jax.device_put(jnp.zeros((2 * TILE[0], TILE[1]), jnp.float32),
                              NamedSharding(mesh, P(AXIS)))
        cells.append((label, functools.partial(build_exchange,
                                               mesh), tile, None))
    cells.append(("HBM read     HBM -> VMEM", build_hbm_read, hbm_source(),
                  hbm_read_total))
    cells.append(("HBM write    VMEM -> HBM", build_hbm_write,
                  jax.device_put(jnp.ones(TILE, jnp.float32)), None))

    print(f"devices={jax.device_count()} ({jax.devices()[0].device_kind}), "
          f"tile={TILE[0] * TILE[1] * 4 // 1024} KiB, "
          f"target={args.target_us:g} us")
    cols = (f"{'link':<48}", f"{'one-way us':>10}", f"{'chain':>6}")
    print(" | ".join(cols))
    for label, build, operand, check in cells:
        us, n_max = latency_us(build,
                               operand,
                               reps=args.reps,
                               warmup=args.warmup,
                               target_us=args.target_us,
                               check=check)
        print(" | ".join((f"{label:<48}", f"{us:>10.3f}", f"{n_max:>6}")))


if __name__ == "__main__":
    main()
