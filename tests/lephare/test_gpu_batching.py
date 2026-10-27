"""RAIL-side batching: PDF evaluation on the stored grid and asynchronous chunk output."""

import numpy as np
import pytest


def test_pdf_on_grid_matches_qp():
    import qp

    from rail.estimation.algos.lephare import _pdf_on_grid

    z = np.arange(301) * 0.01
    y = np.random.default_rng(0).random((20, 301))
    ens = qp.Ensemble(qp.interp, data=dict(xvals=z, yvals=y, norm=False, warn=False))
    grid = np.linspace(0.0, 3.0, 301)
    # stored node values; scipy's interpolation at the nodes may differ by an ulp
    np.testing.assert_allclose(
        _pdf_on_grid(ens, grid), ens.pdf(grid), rtol=1e-12, atol=0
    )
    off = np.linspace(0.005, 2.995, 300)  # not the stored grid: falls back to qp
    np.testing.assert_array_equal(_pdf_on_grid(ens, off), ens.pdf(off))


@pytest.mark.skipif(
    not __import__("lephare.accel.dispatch", fromlist=["x"]).cuda_available(),
    reason="cuda/cupy unavailable",
)
def test_async_output_matches_sync(test_data_dir, tmp_path):
    """Writing chunk k on the output thread while chunk k+1 is fitted gives the same file."""
    import os

    import h5py
    import tables_io

    from rail.estimation.algos.lephare import LephareEstimator, LephareInformer

    train = tables_io.read(os.path.join(test_data_dir, "output_table_conv_train.hdf5"))
    test = tables_io.read(os.path.join(test_data_dir, "output_table_conv_test.hdf5"))
    inform = LephareInformer.make_stage(
        name="inform_async",
        nondetect_val=np.nan,
        model=str(tmp_path / "async.pkl"),
        hdf5_groupname="",
        zmin=0,
        zmax=3,
        nzbins=31,
        backend="cuda_hybrid",
        dtype="float64",
    )
    inform.inform(train)
    n = len(next(iter(test.values())))
    chunk = max(1, n // 5 + 1)  # several chunks, short last one
    files = {}
    for mode in (False, True):
        path = tmp_path / f"est_async_{mode}.hdf5"
        est = LephareEstimator.make_stage(
            name=f"est_async_{mode}",
            nondetect_val=np.nan,
            model=inform.get_handle("model"),
            hdf5_groupname="",
            output=str(path),
            backend="cuda_hybrid",
            dtype="float64",
            chunk_size=chunk,
            async_output=mode,
        )
        est.estimate(test)
        est.finalize()
        files[mode] = path
    with h5py.File(files[False], "r") as a, h5py.File(files[True], "r") as b:
        names = []
        a.visit(lambda k: names.append(k) if isinstance(a[k], h5py.Dataset) else None)
        assert names
        for k in names:
            np.testing.assert_array_equal(a[k][()], b[k][()], err_msg=k)
