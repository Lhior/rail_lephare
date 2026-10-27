"""RAIL-side batching: PDF evaluation on the stored grid."""

import numpy as np


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
