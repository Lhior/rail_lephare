"""LephareEstimator routes RAIL zbest to the device for accelerated backends."""

from __future__ import annotations

import numpy as np
import pytest
import qp

pytest.importorskip("lephare.accel", reason="lephare without lephare.accel")


def _rail_zbest(pdf_vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """RAIL's PointEstimationMixin._calculate_best_point_estimate loop, verbatim math."""
    from scipy.integrate import simpson
    from scipy.optimize import minimize_scalar

    out = np.zeros(pdf_vals.shape[0])
    for i, pz in enumerate(pdf_vals):

        def risk(zx, pz=pz):
            dz = (zx - grid) / (1 + grid)
            return simpson(pz * (1 - 1 / (1 + (dz / 0.15) ** 2)), grid)

        out[i] = minimize_scalar(risk, bounds=(grid[0], grid[-1]), method="bounded").x
    return out


def _pdfs(n=400, nz=301, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 3.0, nz)
    pdfs = np.zeros((n, nz))
    for i in range(n):
        kind = i % 5
        if kind == 0:  # narrow
            pdfs[i] = np.exp(
                -0.5 * ((grid - rng.uniform(0, 3)) / rng.uniform(0.005, 0.05)) ** 2
            )
        elif kind == 1:  # bimodal (several local risk minima)
            for _ in range(2):
                pdfs[i] += rng.uniform(0.3, 1) * np.exp(
                    -0.5 * ((grid - rng.uniform(0, 3)) / rng.uniform(0.03, 0.2)) ** 2
                )
        elif kind == 2:  # peaked at an edge
            pdfs[i] = (
                np.exp(-grid / rng.uniform(0.05, 0.5))
                if rng.random() < 0.5
                else np.exp((grid - 3) / 0.1)
            )
        elif kind == 3:  # broad, noisy
            pdfs[i] = np.exp(
                -0.5 * ((grid - rng.uniform(0.5, 2.5)) / 0.6) ** 2
            ) * rng.uniform(0.5, 1.5, nz)
        else:  # flat
            pdfs[i] = 1.0
    return grid, pdfs


def test_estimator_zbest_override_matches_rail():
    """LephareEstimator routes zbest to the device path only for accel backends."""
    from rail.estimation.algos.lephare import LephareEstimator

    grid, pdfs = _pdfs(n=60)
    ens = qp.Ensemble(qp.interp, data=dict(xvals=grid, yvals=pdfs))
    est = LephareEstimator.make_stage(
        name="est_zbest_override", zmin=0.0, zmax=3.0, nzbins=301, backend="jax_hybrid"
    )
    est.model = {"backend": "lephare"}
    ref = _rail_zbest(ens.pdf(grid), grid)
    np.testing.assert_allclose(
        est._calculate_best_point_estimate(ens), ref, rtol=0, atol=1e-9
    )
