"""RAIL stage tests for the optional ``lephare.accel`` backends.

Kernel-level parity tests live with the backends in lephare (tests/lephare/accel).
"""

import os

import lephare as lp
import numpy as np
import pytest
import tables_io

from rail.estimation.algos.lephare import LephareEstimator, LephareInformer

accel = pytest.importorskip(
    "lephare.accel.dispatch", reason="lephare without lephare.accel"
)

GOLDEN_PDF_SUM = 99.66482986408954
JAX = pytest.mark.skipif(not accel.jax_available(), reason="jax not installed")
CUDA = pytest.mark.skipif(not accel.cuda_available(), reason="cuda/cupy unavailable")


@pytest.fixture(scope="module")
def lsst_inputs():
    test_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train = tables_io.read(os.path.join(test_data_dir, "output_table_conv_train.hdf5"))
    test = tables_io.read(os.path.join(test_data_dir, "output_table_conv_test.hdf5"))
    lephare_config = lp.keymap_to_string_dict(
        lp.read_config(os.path.join(test_data_dir, "lsst.para"))
    )
    lp.data_retrieval.get_auxiliary_data(
        keymap=lephare_config,
        additional_files=["examples/output.para"],
    )
    return lephare_config, train, test


def _inform_and_estimate(lsst_inputs, backend, name, **estimate_kwargs):
    lephare_config, train, test = lsst_inputs
    inform = LephareInformer.make_stage(
        name=f"inform_{name}",
        nondetect_val=np.nan,
        model=f"lephare_{name}.pkl",
        hdf5_groupname="",
        **{f"lephare.{k}": v for k, v in lephare_config.items()},
        # Very sparse redshift grid to speed up the test
        zmin=0,
        zmax=5,
        nzbins=6,
        backend=backend,
    )
    inform.inform(train)
    estimate = LephareEstimator.make_stage(
        name=f"estimate_{name}",
        nondetect_val=np.nan,
        model=inform.get_handle("model"),
        hdf5_groupname="",
        aliases=dict(input=f"test_data_{name}", output=f"lephare_estim_{name}"),
        use_inform_offsets=True,
        backend=backend,
        **estimate_kwargs,
    )
    return inform, estimate.estimate(test).data


def test_unknown_backend_rejected():
    inform = LephareInformer.make_stage(
        name="inform_bad_backend", model="bad.pkl", hdf5_groupname="", backend="jax"
    )
    with pytest.raises(ValueError, match="Unknown backend"):
        inform.validate()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("jax_hybrid", marks=JAX),
        pytest.param("jax_full", marks=JAX),
        pytest.param("cuda_hybrid", marks=CUDA),
        pytest.param("cuda_full", marks=CUDA),
    ],
)
def test_informer_and_estimator_backends(lsst_inputs, backend):
    """Every accel backend reproduces the C++ golden PDF through the RAIL stages."""
    inform, ens = _inform_and_estimate(lsst_inputs, backend, backend)
    assert inform.model["backend"] == backend
    assert os.path.isfile(inform.model["maglib_path"])
    assert np.isclose(np.sum(ens[0].pdf(np.linspace(0, 3, 300))), GOLDEN_PDF_SUM)


@JAX
def test_jax_hybrid_matches_classic(lsst_inputs):
    """Object-by-object PDFs and ancil columns: jax_hybrid vs classic lephare."""
    no_summary = dict(calculated_point_estimates=[], recompute_point_estimates=False)
    _, ens_cpp = _inform_and_estimate(
        lsst_inputs, "lephare", "parity_cpp", **no_summary
    )
    _, ens_jax = _inform_and_estimate(
        lsst_inputs, "jax_hybrid", "parity_jax", **no_summary
    )

    z = np.linspace(0, 3, 300)
    pdf_cpp = np.stack([ens_cpp[i].pdf(z) for i in range(len(ens_cpp))])
    pdf_jax = np.stack([ens_jax[i].pdf(z) for i in range(len(ens_jax))])
    assert np.max(np.abs(pdf_cpp - pdf_jax)) < 1e-4

    for key in (
        "zmode",
        "zmean",
        "Z_BEST",
        "CHI_BEST",
        "ZQ_BEST",
        "CHI_QSO",
        "MOD_STAR",
        "CHI_STAR",
    ):
        a = np.asarray(ens_cpp.ancil[key], dtype=np.float64)
        b = np.asarray(ens_jax.ancil[key], dtype=np.float64)
        atol, rtol = (1e-3, 1e-6) if "CHI" in key else (1e-4, 1e-5)
        ok = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        assert np.all(
            ok
        ), f"{key}: {(~ok).sum()} mismatches, cpp={a[~ok][:3]} jax={b[~ok][:3]}"
