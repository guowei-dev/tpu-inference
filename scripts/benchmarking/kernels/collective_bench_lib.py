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
"""Shared harness for the collective-matmul microbenchmarks.

Holds what the benchmarks share — mesh construction (with selectable physical
device order), per-call trace timing, HLO classification, a numeric check, and
the sweep/report driver — so each benchmark script is just its own collective:
input shardings, the einsum, and an IMPLEMENTATIONS registry. Not runnable on
its own.
"""

import argparse
import bisect
import functools
import glob
import hashlib
import os
import shutil
import statistics
import tempfile
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P  # noqa: F401  (re-export)

AXIS = "x"

# Hamiltonian chip cycle for the 4-chip (2x2) x 2-core topology: every ring
# hop between consecutive chips is a single ICI link (the natural order rings
# the torus with two 2-hop diagonal edges instead).
_GRAY_8 = [0, 1, 2, 3, 6, 7, 4, 5]


def _chip_cycle(devices):
    """Consecutive-device chip coords, deduplicated, in ring order."""
    cycle = []
    for d in devices:
        c = tuple(d.coords)
        if not cycle or cycle[-1] != c:
            cycle.append(c)
    return cycle


def _assert_single_hop_ring(devices):
    """Every consecutive chip pair (incl. the wraparound) must be 1 hop."""
    cycle = _chip_cycle(devices)
    if sorted(cycle) != sorted(set(cycle)):
        raise ValueError(f"device order revisits a chip: {cycle}")
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        hops = sum(abs(x - y) for x, y in zip(a, b))
        if hops != 1:
            raise ValueError(
                f"chip ring edge {a}->{b} is {hops} hops (want 1); "
                f"the hardcoded order does not fit this topology")


def build_mesh(kind="natural"):
    """Auto-axis mesh (not explicit axes) so the auto-partitioner inserts the
    collective for the sharded einsum, as the serving path does.

    kind: 'natural' = jax.devices() order; 'optimized' = the serving path's
    make_optimized_mesh; 'gray' = the single-hop Hamiltonian chip cycle
    (validated against the live topology, fails loudly elsewhere).
    """
    devs = jax.devices()
    if kind == "natural":
        return Mesh(np.asarray(devs), (AXIS, ))
    if kind == "optimized":
        from tpu_inference import utils
        return utils.make_optimized_mesh((len(devs), ), (AXIS, ))
    if kind == "gray":
        if len(devs) != len(_GRAY_8):
            raise ValueError(f"gray order is defined for {len(_GRAY_8)} "
                             f"devices, got {len(devs)}")
        ordered = [devs[i] for i in _GRAY_8]
        _assert_single_hop_ring(ordered)
        return Mesh(np.asarray(ordered), (AXIS, ))
    raise ValueError(f"unknown mesh kind {kind!r}")


class Impl(NamedTuple):
    """One benchmarked implementation: builder + the mesh order it runs on."""
    build: Callable[[Any, tuple], Callable]  # (mesh, inputs) -> callable
    mesh: str = "natural"


def xla_einsum_builder(einsum, out_spec, options=None):
    """Builder for an auto-partitioned einsum compiled with the given
    compiler options ('serve' resolves the real serving step-fn options on
    the TPU host; None compiles plain)."""

    def build(mesh, inputs):
        opts = options
        if opts == "serve":
            from tpu_inference.models.common.compiler_options import \
                get_step_fn_compiler_options
            opts = get_step_fn_compiler_options()
        fn = jax.jit(functools.partial(jnp.einsum, einsum),
                     out_shardings=NamedSharding(mesh, out_spec))
        return fn.lower(*inputs).compile(compiler_options=opts or None)

    return build


_HLO_MARKS = ("collective-permute", "while(", "all-gather(",
              "all-gather-start", "reduce-scatter(", "all-reduce(",
              "dynamic-slice(", "custom-call")


def classify_hlo(fn):
    """Label the lowering a built implementation actually got (never trust a
    flag name): op counts + a short fingerprint of the compiled HLO text.
    Returns None for builders that don't expose the compiled module."""
    as_text = getattr(fn, "as_text", None)
    if as_text is None:
        return None
    txt = as_text()
    c = {m: txt.count(m) for m in _HLO_MARKS}
    if c["collective-permute"] and c["while("]:
        label = "windowed"
    elif c["all-gather("] or c["all-gather-start"]:
        label = "serial_ag"
    elif c["reduce-scatter("]:
        label = "serial_rs"
    elif c["all-reduce("]:
        label = "ar_ds"
    elif c["collective-permute"]:
        label = "cp"
    elif c["custom-call"]:
        label = "kernel"
    else:
        label = "local"
    return dict(label=label,
                fp=hashlib.sha1(txt.encode()).hexdigest()[:12],
                counts=c)


def golden_output(einsum, out_spec, mesh, inputs):
    """fp32 reference output for the numeric check (same shardings)."""
    f32 = tuple(v.astype(jnp.float32) for v in inputs)
    fn = jax.jit(functools.partial(jnp.einsum, einsum),
                 out_shardings=NamedSharding(mesh, out_spec))
    return jax.block_until_ready(fn(*f32))


def rel_err(out, golden):
    """max|out - golden| / max|golden| (global normalized max error).
    Reshards `out` onto the golden's mesh first — an implementation may run
    on a different device order."""
    out = jax.device_put(out.astype(jnp.float32), golden.sharding)
    d = jnp.max(jnp.abs(out - golden))
    return float(d / jnp.max(jnp.abs(golden)))


def _event_spans(plane, line_name, event_name=None):
    """(start_ns, device_duration_ps) per event, sorted by start time."""

    def dur(event):
        for name, value in event.stats:
            if name == "device_duration_ps":
                return value
        return 0.0

    return sorted((e.start_ns, dur(e)) for line in plane.lines
                  if line.name == line_name for e in line.events
                  if event_name is None or e.name == event_name)


def device_time(fn, inputs, *, reps, warmup, save_dir=None):
    """Per-call kernel latency from the JAX profiler.

    Each `barrier-cores` TraceMe span (the collective's cross-core wait) is
    assigned to the call whose module span it falls in — by start time, so a
    call may own 0, 1, or N spans (windowed einsum has none, a Pallas kernel
    brings its own) — and subtracted from that call's module device time;
    this cancels the sync jitter the wait absorbs. Returns
    dict(lat_ms = median per-call barrier-subtracted latency (the timing
    authority), module_ms = median raw module device time, cv = %CV over the
    interquartile core of per-call latencies, barriers_per_call). Traces
    `reps` calls into save_dir if given (kept for inspection).
    """
    for _ in range(warmup):
        fn(*inputs)
    jax.block_until_ready(fn(*inputs))
    trace_dir = save_dir or tempfile.mkdtemp()
    try:
        with jax.profiler.trace(trace_dir):
            out = None
            for _ in range(reps):
                out = fn(*inputs)
            jax.block_until_ready(out)
        pbs = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"),
                        recursive=True)
        if not pbs:
            raise RuntimeError("profiler wrote no trace")
        newest = max(pbs,
                     key=os.path.getmtime)  # this run's, if save_dir reused
        plane = jax.profiler.ProfileData.from_file(
            newest).find_plane_with_name("/device:TPU:0")
        if plane is None:
            raise RuntimeError("no /device:TPU:0 plane in the trace")
        modules = _event_spans(plane, "XLA Modules")
        if not modules:
            raise RuntimeError("no XLA-module device time in the profile")
        barriers = _event_spans(plane, "XLA TraceMe", "barrier-cores")
        starts = [s for s, _ in modules]
        barrier_ps = [0.0] * len(modules)
        for start, dur in barriers:
            call = bisect.bisect_right(starts, start) - 1
            if call < 0:
                raise RuntimeError("barrier span precedes the first module")
            barrier_ps[call] += dur
        per_call = [m - b for (_, m), b in zip(modules, barrier_ps)]
        latency = statistics.median(per_call)
        if latency <= 0:
            raise RuntimeError("non-positive per-call latency (barrier span "
                               ">= module, or empty module time)")
        core = sorted(per_call)[len(per_call) //
                                4:-(len(per_call) // 4) or None]
        cv = (statistics.pstdev(core) / statistics.mean(core) *
              100 if len(core) > 1 else 0.0)
        return dict(lat_ms=latency / 1e9,
                    module_ms=statistics.median(m for _, m in modules) / 1e9,
                    cv=cv,
                    barriers_per_call=len(barriers) / len(modules))
    finally:
        if save_dir is None:
            shutil.rmtree(trace_dir, ignore_errors=True)


def run_impl(impl, mesh, inputs, *, reps, warmup, golden, rtol, save_dir=None):
    """Build + classify + numeric-check + time one implementation.
    Returns a result dict; failures become status rows, never sweep aborts."""
    try:
        fn = impl.build(mesh, inputs)
        hlo = classify_hlo(fn)
        out = jax.block_until_ready(fn(*inputs))
        err = None if golden is None else rel_err(out, golden)
        del out
        t = device_time(fn,
                        inputs,
                        reps=reps,
                        warmup=warmup,
                        save_dir=save_dir)
        status = "ok" if err is None or err <= rtol else "numeric_fail"
        return dict(status=status, err_rel=err, hlo=hlo, **t)
    except Exception as e:  # a narrow-domain impl yields a FAIL cell
        return dict(status=f"FAIL {type(e).__name__}: "
                    f"{' '.join(str(e).split())[:160]}")


def run(*, make_inputs, einsum, out_spec, implementations, default_n):
    """Sweep M for one collective and print a per-M table.

    make_inputs(mesh, m, k, n, dtype) -> operands; implementations maps
    name -> Impl, first entry the reference; einsum/out_spec define the
    pattern (they build the fp32 numeric golden and the XLA baselines);
    default_n is the --n default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",
                        default="16,32,64,128,256,512,1024,2048,4096,8192",
                        help="comma-separated token counts to sweep")
    parser.add_argument("--k", type=int, default=8192, help="hidden dim")
    parser.add_argument("--n", type=int, default=default_n)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--reps",
                        type=int,
                        default=20,
                        help="traced calls per cell (median)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rtol",
                        type=float,
                        default=2e-2,
                        help="numeric gate vs the fp32 golden")
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="if set, keep each cell's xprof trace under this dir "
        "(named <impl>_m<M>) for inspection; otherwise discarded")
    args = parser.parse_args()

    assert jax.devices()[0].platform == "tpu", "requires a TPU host"
    tp = jax.device_count()
    if args.n % tp != 0:
        parser.error(f"--n ({args.n}) must be divisible by {tp} devices")
    dtype = jnp.dtype(args.dtype)
    names = list(implementations)
    ref = names[0]
    meshes = {
        impl.mesh: build_mesh(impl.mesh)
        for impl in implementations.values()
    }

    print(f"devices={tp} ({jax.devices()[0].device_kind}), k={args.k}, "
          f"n={args.n} (n/tp={args.n // tp}), dtype={dtype.name}")
    cols = (["M", "TFLOP/s/c"] + [f"{n} us" for n in names] +
            [f"{n} x" for n in names[1:]])
    print(" | ".join(f"{c:>9}" for c in cols))

    prev_labels = None
    for m in [int(v) for v in args.m.split(",")]:
        if m % tp != 0:
            print(f"{m:>9} | (skip: not divisible by {tp} devices)")
            continue
        inputs_by_kind = {
            kind: make_inputs(mesh, m, args.k, args.n, dtype)
            for kind, mesh in meshes.items()
        }
        ref_kind = implementations[ref].mesh
        golden = golden_output(einsum, out_spec, meshes[ref_kind],
                               inputs_by_kind[ref_kind])
        results = {}
        for name in names:
            impl = implementations[name]
            save_dir = (os.path.join(args.profile_dir, f"{name}_m{m}")
                        if args.profile_dir else None)
            results[name] = run_impl(impl,
                                     meshes[impl.mesh],
                                     inputs_by_kind[impl.mesh],
                                     reps=args.reps,
                                     warmup=args.warmup,
                                     golden=golden,
                                     rtol=args.rtol,
                                     save_dir=save_dir)
        del golden
        times = {
            n: r["lat_ms"]
            for n, r in results.items() if r["status"] == "ok"
        }
        base = times.get(ref)
        # Per-core matmul FLOPs = 2 * M * K * (N // tp), the same for both
        # patterns (AG: [M, K] @ [K, N//tp]; RS: [M, N//tp] @ [N//tp, K]).
        tflops = (2 * m * args.k * (args.n // tp) / (base * 1e-3) /
                  1e12 if base else None)
        cells = [f"{m:>9}", f"{tflops:>9.1f}" if tflops else f"{'--':>9}"]
        cells += [
            f"{times[n] * 1e3:>9.1f}" if n in times else f"{'FAIL':>9}"
            for n in names
        ]
        cells += [
            f"{base / times[n]:>9.2f}" if base and n in times else f"{'--':>9}"
            for n in names[1:]
        ]
        print(" | ".join(cells))
        labels = {
            n: (r.get("hlo") or {}).get("label", "?")
            for n, r in results.items()
        }
        if labels != prev_labels:
            print("          | lowering: " +
                  ", ".join(f"{n}={v}" for n, v in labels.items()))
            prev_labels = labels
        for n, r in results.items():
            if r["status"] != "ok":
                print(f"          | {n}: {r['status']}")

    if args.profile_dir:
        print(f"xprof traces kept under {args.profile_dir}")
