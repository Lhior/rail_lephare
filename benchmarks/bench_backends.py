#!/usr/bin/env python
"""Accuracy-gated + max-batch throughput benchmarks for LePHARE backends.

Usage (set the thread count to the cores you want the C++ baseline to use):
  export OMP_NUM_THREADS=<n_cores>
  export JAX_ENABLE_X64=1
  pip install -e '.[jax,cuda]'
  python benchmarks/bench_backends.py
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import lephare as lp
import numpy as np
import tables_io

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "tests" / "data"
EX_DATA = ROOT / "examples" / "data"
GOLDEN = 99.66482986408954


def _device_info():
    info = {"omp_threads": os.environ.get("OMP_NUM_THREADS"), "jax": None, "cuda": None}
    try:
        import jax
        from lephare.accel.jax_backend import ensure_gpu_backend

        plat = ensure_gpu_backend()
        info["jax"] = f"{plat} devices={jax.devices()}"
    except Exception as exc:
        info["jax"] = f"unavailable ({exc})"
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = (
            props["name"].decode()
            if isinstance(props["name"], bytes)
            else props["name"]
        )
        info["cuda"] = name
    except Exception as exc:
        info["cuda"] = f"unavailable ({exc})"
    return info


class GpuUtilSampler:
    """Background nvidia-smi util/memory sampler."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._stop = threading.Event()
        self._thread = None
        self.utils: list[float] = []
        self.mems: list[float] = []

    def start(self):
        self.utils.clear()
        self.mems.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self.utils:
            return {
                "mean_util": 0.0,
                "peak_util": 0.0,
                "peak_mem_mib": 0.0,
                "nsamples": 0,
            }
        return {
            "mean_util": float(np.mean(self.utils)),
            "peak_util": float(np.max(self.utils)),
            "peak_mem_mib": float(np.max(self.mems)) if self.mems else 0.0,
            "nsamples": len(self.utils),
        }

    def _run(self):
        while not self._stop.is_set():
            try:
                out = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    )
                    .strip()
                    .splitlines()[0]
                )
                util_s, mem_s = [p.strip() for p in out.split(",")]
                self.utils.append(float(util_s))
                self.mems.append(float(mem_s))
            except Exception:
                pass
            self._stop.wait(self.interval_ms / 1000.0)


def _tile_table(table: dict, n_target: int) -> dict:
    """Repeat rows until ``n_target`` objects (dict-of-arrays / tables_io style)."""
    # Determine length from first array-like column
    n0 = None
    for v in table.values():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.dtype.kind != "O":
            n0 = len(arr)
            break
    if n0 is None or n0 == 0:
        raise ValueError("cannot tile empty table")
    reps = int(np.ceil(n_target / n0))
    out = {}
    for k, v in table.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and len(arr) == n0:
            tiled = np.concatenate([arr] * reps, axis=0)[:n_target]
            out[k] = tiled
        else:
            out[k] = v
    return out


def _table_nrows(table: dict) -> int:
    for v in table.values():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.dtype.kind != "O":
            return len(arr)
    raise ValueError("cannot determine table length")


def _slice_table(table: dict, start: int, end: int) -> dict:
    """Row-slice a dict-of-arrays catalog (in-memory RAIL chunk)."""
    n0 = _table_nrows(table)
    out = {}
    for k, v in table.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and len(arr) == n0:
            out[k] = arr[start:end]
        else:
            out[k] = v
    return out


def _run_parity(backends, cfg, train, test):
    from rail.estimation.algos.lephare import LephareEstimator, LephareInformer

    rows = []
    print("\n=== Parity fixture (~1.5k objs, accuracy-gated FP64) ===")
    for backend in backends:
        dtypes = ["float64"] if backend == "lephare" else ["float64", "float32"]
        for dtype in dtypes:
            t0 = time.perf_counter()
            inform = LephareInformer.make_stage(
                name=f"bench_{backend}_{dtype}",
                nondetect_val=np.nan,
                model=f"bench_{backend}_{dtype}.pkl",
                hdf5_groupname="",
                **{f"lephare.{k}": v for k, v in cfg.items()},
                zmin=0,
                zmax=5,
                nzbins=6,
                backend=backend,
                dtype=dtype,
            )
            t_prep0 = time.perf_counter()
            inform.inform(train)
            t_inform = time.perf_counter() - t_prep0

            est = LephareEstimator.make_stage(
                name=f"bench_est_{backend}_{dtype}",
                nondetect_val=np.nan,
                model=inform.get_handle("model"),
                hdf5_groupname="",
                aliases=dict(input="test_data", output="out"),
                use_inform_offsets=True,
                backend=backend,
                dtype=dtype,
            )
            sampler = GpuUtilSampler()
            sampler.start()
            t_e0 = time.perf_counter()
            out = est.estimate(test)
            t_est = time.perf_counter() - t_e0
            util = sampler.stop()
            total = time.perf_counter() - t0
            pdf_sum = float(np.sum(out.data[0].pdf(np.linspace(0, 3, 300))))
            ok = bool(np.isclose(pdf_sum, GOLDEN)) if dtype == "float64" else None
            err = abs(pdf_sum - GOLDEN) if np.isfinite(pdf_sum) else float("nan")
            n_obj = len(np.asarray(test[list(test.keys())[0]]))
            rows.append(
                dict(
                    phase="parity",
                    backend=backend,
                    dtype=dtype,
                    n_obj=n_obj,
                    t_inform=t_inform,
                    t_estimate=t_est,
                    t_total=total,
                    pdf_sum=pdf_sum,
                    abs_err=err,
                    accuracy_ok=ok,
                    **util,
                    objs_per_s=n_obj / t_est if t_est > 0 else 0.0,
                    model=inform.get_handle("model"),
                )
            )
            print(
                f"{backend:12s} {dtype:8s} n={n_obj:6d} "
                f"inform={t_inform:7.3f}s est={t_est:7.3f}s "
                f"util_mean={util['mean_util']:5.1f}% peak={util['peak_util']:5.1f}% "
                f"mem={util['peak_mem_mib']:.0f}MiB "
                f"pdf_sum={pdf_sum:.6f} ok={ok}"
            )
            if dtype == "float64" and not ok:
                raise SystemExit(f"FP64 accuracy gate failed for {backend}")
    return rows


def _collect_models(parity_rows):
    models = {}
    for r in parity_rows:
        if r["dtype"] == "float64" and r["backend"] not in models:
            models[r["backend"]] = r["model"]
    return models


def _ensure_model(backend, dtype, models, cfg, train):
    from rail.estimation.algos.lephare import LephareInformer

    if backend in models and dtype == "float64":
        return models[backend]
    inform = LephareInformer.make_stage(
        name=f"thr_{backend}_{dtype}",
        nondetect_val=np.nan,
        model=f"thr_{backend}_{dtype}.pkl",
        hdf5_groupname="",
        **{f"lephare.{k}": v for k, v in cfg.items()},
        zmin=0,
        zmax=5,
        nzbins=6,
        backend=backend,
        dtype=dtype,
    )
    inform.inform(train)
    model_handle = inform.get_handle("model")
    if dtype == "float64":
        models[backend] = model_handle
    return model_handle


def _run_throughput(backends, cfg, train, test, parity_rows):
    """Short VRAM-saturation probe (one GPU-sized tile)."""
    from lephare.accel.gpu_util import suggest_chi2_chunk_size
    from lephare.accel.maglib import load_maglib

    from rail.estimation.algos.lephare import LephareEstimator

    print("\n=== Throughput probe (one GPU-sized tile) ===")
    models = _collect_models(parity_rows)

    n_t = 9194
    for backend, handle in models.items():
        if backend == "lephare":
            continue
        try:
            model = handle.read() if hasattr(handle, "read") else handle
            if isinstance(model, dict) and model.get("maglib_path"):
                tensors = load_maglib(model["maglib_path"])
                n_t = int(tensors.flux.shape[0])
                break
        except Exception:
            pass

    n_fp64 = suggest_chi2_chunk_size(n_t, dtype=np.float64, mem_fraction=0.60)
    n_fp32 = suggest_chi2_chunk_size(n_t, dtype=np.float32, mem_fraction=0.60)
    n_fp64 = min(max(n_fp64, 4096), 20000)
    n_fp32 = min(max(n_fp32, 8192), 40000)
    print(f"GPU tile targets: FP64 n_obj={n_fp64}, FP32 n_obj={n_fp32} (N_t={n_t})")

    rows = []
    for backend in backends:
        dtypes = ["float64"] if backend == "lephare" else ["float64", "float32"]
        for dtype in dtypes:
            n_target = n_fp64 if dtype == "float64" else n_fp32
            big = _tile_table(test, n_target)
            model_handle = _ensure_model(backend, dtype, models, cfg, train)

            est_w = LephareEstimator.make_stage(
                name=f"thr_w_{backend}_{dtype}",
                nondetect_val=np.nan,
                model=model_handle,
                hdf5_groupname="",
                aliases=dict(input="test_data", output="out_w"),
                use_inform_offsets=True,
                backend=backend,
                dtype=dtype,
            )
            _ = est_w.estimate(_tile_table(test, min(64, n_target)))

            est = LephareEstimator.make_stage(
                name=f"thr_est_{backend}_{dtype}",
                nondetect_val=np.nan,
                model=model_handle,
                hdf5_groupname="",
                aliases=dict(input="test_data", output="out"),
                use_inform_offsets=True,
                backend=backend,
                dtype=dtype,
            )
            sampler = GpuUtilSampler(interval_ms=50)
            sampler.start()
            t0 = time.perf_counter()
            _out = est.estimate(big)
            t_est = time.perf_counter() - t0
            util = sampler.stop()
            row = dict(
                phase="throughput",
                backend=backend,
                dtype=dtype,
                n_obj=n_target,
                chunk_size=n_target,
                t_estimate=t_est,
                objs_per_s=n_target / t_est if t_est > 0 else 0.0,
                **util,
            )
            rows.append(row)
            print(
                f"{backend:12s} {dtype:8s} n={n_target:6d} "
                f"est={t_est:7.3f}s  {row['objs_per_s']:8.1f} obj/s  "
                f"util_mean={util['mean_util']:5.1f}% peak={util['peak_util']:5.1f}% "
                f"mem={util['peak_mem_mib']:.0f}MiB"
            )
    return rows


def _run_large_throughput(
    backends,
    cfg,
    train,
    test,
    parity_rows,
    n_total: int = 1_000_000,
    chunk_size: int = 100_000,
    dtypes: tuple[str, ...] = ("float64",),
):
    """Longer estimate bench: in-memory clones of the fixture to ``n_total``.

    RAIL's ``input_iterator`` always yields a single chunk for in-memory tables
    (ignores ``chunk_size``), so this bench **manually** slices the tiled catalog
    into ``chunk_size`` row blocks. GPU backends further sub-chunk each block via
    ``suggest_chi2_chunk_size`` to fit VRAM. ``lephare`` is skipped (hours at 1M).
    """
    from rail.estimation.algos.lephare import LephareEstimator

    gpu_backends = [b for b in backends if b != "lephare"]
    if not gpu_backends:
        print("\n=== Large throughput skipped (no GPU backends) ===")
        return []

    n_chunks = int(np.ceil(n_total / chunk_size))
    print(
        f"\n=== Large throughput (in-memory clones: n={n_total:,}, "
        f"{n_chunks} × {chunk_size:,}-row slices) ==="
    )
    print(
        "Note: RAIL ignores chunk_size for in-memory inputs; "
        "bench slices the tiled table explicitly.",
        flush=True,
    )
    models = _collect_models(parity_rows)
    print(f"Building tiled catalog ({n_total:,} rows)…", flush=True)
    t_tile0 = time.perf_counter()
    big = _tile_table(test, n_total)
    print(f"  tiled in {time.perf_counter() - t_tile0:.2f}s", flush=True)

    rows = []
    for backend in gpu_backends:
        for dtype in dtypes:
            if backend == "lephare" and dtype != "float64":
                continue
            model_handle = _ensure_model(backend, dtype, models, cfg, train)

            est_w = LephareEstimator.make_stage(
                name=f"large_w_{backend}_{dtype}",
                nondetect_val=np.nan,
                model=model_handle,
                hdf5_groupname="",
                aliases=dict(input="test_data", output="large_w"),
                use_inform_offsets=True,
                backend=backend,
                dtype=dtype,
            )
            _ = est_w.estimate(_tile_table(test, min(1024, n_total)))

            sampler = GpuUtilSampler(interval_ms=100)
            sampler.start()
            t0 = time.perf_counter()
            for start in range(0, n_total, chunk_size):
                end = min(start + chunk_size, n_total)
                print(
                    f"  {backend:12s} {dtype:8s} slice {start:,} - {end:,}",
                    flush=True,
                )
                est = LephareEstimator.make_stage(
                    name=f"large_est_{backend}_{dtype}_{start}",
                    nondetect_val=np.nan,
                    model=model_handle,
                    hdf5_groupname="",
                    aliases=dict(input="test_data", output=f"large_out_{start}"),
                    use_inform_offsets=True,
                    backend=backend,
                    dtype=dtype,
                )
                _ = est.estimate(_slice_table(big, start, end))
            t_est = time.perf_counter() - t0
            util = sampler.stop()
            row = dict(
                phase="large_throughput",
                backend=backend,
                dtype=dtype,
                n_obj=n_total,
                chunk_size=chunk_size,
                n_chunks=n_chunks,
                t_estimate=t_est,
                objs_per_s=n_total / t_est if t_est > 0 else 0.0,
                **util,
            )
            rows.append(row)
            print(
                f"{backend:12s} {dtype:8s} n={n_total:,} chunk={chunk_size:,} "
                f"est={t_est:8.2f}s  {row['objs_per_s']:8.1f} obj/s  "
                f"util_mean={util['mean_util']:5.1f}% peak={util['peak_util']:5.1f}% "
                f"mem={util['peak_mem_mib']:.0f}MiB",
                flush=True,
            )
    return rows


def _run_compute_only(
    backends,
    cfg,
    train,
    test,
    parity_rows,
    n_total: int = 1_000_000,
    mean_util_min: float = 40.0,
    gate_backend: str = "jax_hybrid",
):
    """Fused-API util gate: sample nvidia-smi only during ``process_batch_*``.

    Skips Estimator/HDF5/qp so mean util reflects device-native compute.
    Fails the process if ``gate_backend`` FP64 mean util is below ``mean_util_min``.
    """
    from lephare.accel.dispatch import process_with_backend
    from lephare.accel.maglib import load_maglib

    gpu_backends = [b for b in backends if b != "lephare"]
    if not gpu_backends:
        print("\n=== Compute-only util gate skipped (no GPU backends) ===")
        return []

    print(
        f"\n=== Compute-only util gate (n={n_total:,}, "
        f"require {gate_backend} mean util >= {mean_util_min:.0f}%) ===",
        flush=True,
    )
    models = _collect_models(parity_rows)
    rows = []

    # Resolve bands from first available model / informer config
    from rail.estimation.algos.lephare import LephareInformer

    inform = LephareInformer.make_stage(
        name="compute_bands",
        nondetect_val=np.nan,
        model="compute_bands.pkl",
        hdf5_groupname="",
        **{f"lephare.{k}": v for k, v in cfg.items()},
        zmin=0,
        zmax=5,
        nzbins=6,
        backend="lephare",
    )
    inform.validate()
    bands = list(inform.config["bands"])
    err_bands = list(inform.config["err_bands"])

    big = _tile_table(test, n_total)
    mags = np.column_stack([np.asarray(big[b], dtype=np.float64) for b in bands])
    magerrs = np.column_stack([np.asarray(big[b], dtype=np.float64) for b in err_bands])
    context = big["context"] if "context" in big else None

    for backend in gpu_backends:
        model_handle = _ensure_model(backend, "float64", models, cfg, train)
        model = model_handle.read() if hasattr(model_handle, "read") else model_handle
        if not isinstance(model, dict) or not model.get("maglib_path"):
            print(f"  {backend}: skip (no maglib_path)")
            continue
        tensors = load_maglib(model["maglib_path"])
        offsets = model.get("offsets")

        # Warmup small tile (compile / cache)
        _ = process_with_backend(
            backend,
            mags[:256],
            magerrs[:256],
            tensors,
            context=None if context is None else context[:256],
            offsets=offsets,
            dtype=np.float64,
        )

        sampler = GpuUtilSampler(interval_ms=50)
        sampler.start()
        t0 = time.perf_counter()
        result = process_with_backend(
            backend,
            mags,
            magerrs,
            tensors,
            context=context,
            offsets=offsets,
            dtype=np.float64,
        )
        t_est = time.perf_counter() - t0
        util = sampler.stop()
        pdfs = result[0]
        row = dict(
            phase="compute_only",
            backend=backend,
            dtype="float64",
            n_obj=n_total,
            t_estimate=t_est,
            objs_per_s=n_total / t_est if t_est > 0 else 0.0,
            pdf_shape=tuple(pdfs.shape),
            **util,
        )
        rows.append(row)
        status = "PASS" if util["mean_util"] >= mean_util_min else "LOW"
        print(
            f"{backend:12s} float64  n={n_total:,} "
            f"est={t_est:8.2f}s  {row['objs_per_s']:8.1f} obj/s  "
            f"util_mean={util['mean_util']:5.1f}% peak={util['peak_util']:5.1f}% "
            f"mem={util['peak_mem_mib']:.0f}MiB [{status}]",
            flush=True,
        )

    gate = next(
        (r for r in rows if r["backend"] == gate_backend and r["dtype"] == "float64"),
        None,
    )
    if gate is None and gate_backend in gpu_backends:
        raise SystemExit(f"compute-only gate: no result for {gate_backend}")
    if gate is not None and gate["mean_util"] < mean_util_min:
        raise SystemExit(
            f"compute-only util gate FAILED: {gate_backend} mean_util="
            f"{gate['mean_util']:.1f}% < {mean_util_min:.0f}%"
        )
    if gate is not None:
        print(
            f"\nUtil gate PASS: {gate_backend} mean_util={gate['mean_util']:.1f}% "
            f"(threshold {mean_util_min:.0f}%)",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None):
    import argparse

    from lephare.accel.dispatch import available_backends

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip ~1.5k accuracy-gated parity (reuse requires prior models on disk)",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip short one-tile GPU throughput probe",
    )
    parser.add_argument(
        "--large-n",
        type=int,
        default=1_000_000,
        help="In-memory tiled catalog size for large throughput (0 to skip)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="RAIL catalog chunk size for large throughput",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Subset of backends (default: all available)",
    )
    parser.add_argument(
        "--large-dtypes",
        nargs="*",
        default=["float64"],
        help="Dtypes for large throughput (default: float64 only)",
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Run fused process_batch_* util gate (skip Estimator I/O)",
    )
    parser.add_argument(
        "--compute-n",
        type=int,
        default=1_000_000,
        help="Object count for --compute-only util gate",
    )
    parser.add_argument(
        "--mean-util-min",
        type=float,
        default=40.0,
        help="Fail if jax_hybrid mean util is below this (compute-only)",
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip large Estimator throughput even if --large-n > 0",
    )
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    print("Devices:", _device_info())
    cfg = lp.keymap_to_string_dict(lp.read_config(str(TEST_DATA / "lsst.para")))
    lp.data_retrieval.get_auxiliary_data(
        keymap=lp.read_config(str(TEST_DATA / "lsst.para")),
        additional_files=["examples/output.para"],
    )
    train = tables_io.read(str(EX_DATA / "output_table_conv_train.hdf5"))
    test = tables_io.read(str(EX_DATA / "output_table_conv_test.hdf5"))

    backends = args.backends or available_backends()
    parity = []
    if not args.skip_parity:
        parity = _run_parity(backends, cfg, train, test)

        print("\nFP64/FP32 penalty (parity estimate time):")
        by = {(r["backend"], r["dtype"]): r for r in parity}
        for backend in backends:
            if backend == "lephare":
                continue
            a = by.get((backend, "float64"))
            b = by.get((backend, "float32"))
            if a and b and b["t_estimate"] > 0:
                print(
                    f"  {backend:12s}  t64/t32 = {a['t_estimate'] / b['t_estimate']:.3f} "
                    f"(t64={a['t_estimate']:.3f}s t32={b['t_estimate']:.3f}s)"
                )
    else:
        # Inform once per GPU backend so probe/large have models
        models: dict = {}
        parity = []
        for backend in backends:
            if backend == "lephare":
                continue
            handle = _ensure_model(backend, "float64", models, cfg, train)
            parity.append(dict(dtype="float64", backend=backend, model=handle))

    thr = []
    if not args.skip_probe:
        thr = _run_throughput(backends, cfg, train, test, parity)
        lep = next(
            (r for r in thr if r["backend"] == "lephare" and r["dtype"] == "float64"),
            None,
        )
        print("\nThroughput speedup vs lephare FP64 (same n_obj for FP64 rows):")
        if lep:
            for r in thr:
                if r["backend"] == "lephare":
                    continue
                if (
                    r["dtype"] == "float64"
                    and r["n_obj"] == lep["n_obj"]
                    and r["t_estimate"] > 0
                ):
                    print(
                        f"  {r['backend']:12s} FP64  "
                        f"speedup={lep['t_estimate'] / r['t_estimate']:.2f}x  "
                        f"util_peak={r['peak_util']:.1f}%"
                    )
                elif r["dtype"] == "float32" and r["t_estimate"] > 0:
                    t_lep = lep["t_estimate"] * (r["n_obj"] / lep["n_obj"])
                    print(
                        f"  {r['backend']:12s} FP32  "
                        f"speedup≈{t_lep / r['t_estimate']:.2f}x (vs scaled lephare)  "
                        f"util_peak={r['peak_util']:.1f}%"
                    )

        print("\nGPU engagement (probe peak util ≳ 50% target):")
        for r in thr:
            if r["backend"] == "lephare":
                continue
            status = "PASS" if r["peak_util"] >= 50.0 else "LOW"
            print(
                f"  {r['backend']:12s} {r['dtype']:8s} peak_util={r['peak_util']:5.1f}% [{status}]"
            )

    if args.compute_only:
        _run_compute_only(
            backends,
            cfg,
            train,
            test,
            parity,
            n_total=args.compute_n,
            mean_util_min=args.mean_util_min,
        )

    if args.large_n > 0 and not args.skip_large:
        large = _run_large_throughput(
            backends,
            cfg,
            train,
            test,
            parity,
            n_total=args.large_n,
            chunk_size=args.chunk_size,
            dtypes=tuple(args.large_dtypes),
        )
        if large:
            print("\nLarge-throughput summary:")
            for r in large:
                print(
                    f"  {r['backend']:12s} {r['dtype']:8s} "
                    f"{r['objs_per_s']:8.1f} obj/s  "
                    f"({r['n_obj']:,} objs / {r['t_estimate']:.1f}s)  "
                    f"mean_util={r['mean_util']:.1f}% peak_util={r['peak_util']:.1f}%"
                )


if __name__ == "__main__":
    main()
