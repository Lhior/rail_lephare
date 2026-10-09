"""Galaxy properties through the RAIL stages: accelerated backends vs C++ LePhare."""

from __future__ import annotations

import os

import lephare as lp
import numpy as np
import pytest
import tables_io
from lephare.accel.dispatch import cuda_available, jax_available
from lephare.accel.physical import (
    GALAXY_PROPERTY_KEYS,
)

from rail.estimation.algos.lephare import LephareEstimator, LephareInformer


def _bc03_overrides(tmp_path) -> dict:
    """Stage kwargs for a small BC03 setup (physical parameters defined, dust on all
    models) with RF colours and MABS method 1; PARA_OUT lists every compared key."""
    from rail.estimation.algos.lephare import lsst_default_config

    para = tmp_path / "output.para"
    para.write_text(
        "\n".join(("IDENT", "Z_BEST", "CHI_BEST") + GALAXY_PROPERTY_KEYS) + "\n"
    )
    lephare = dict(
        GAL_SED="sed/GAL/BC03_CHAB/BC03COMB_MOD.list",
        SEL_AGE=os.path.join(lp.LEPHAREDIR, "sed/GAL/BC03_CHAB/AGE_BC03COMB.dat"),
        GAL_LIB="LSST_GAL_BC03_BIN",
        GAL_LIB_IN="LSST_GAL_BC03_BIN",
        GAL_LIB_OUT="LSST_GAL_BC03_MAG",
        ZPHOTLIB="LSST_STAR_MAG,LSST_GAL_BC03_MAG,LSST_QSO_MAG",
        EB_V="0.,0.1,0.3",
        RF_COLORS="1,3,3,5",
        M_REF="4",
        MABS_METHOD="1",
        MABS_CONTEXT="0",
        PARA_OUT=str(para),
    )
    keymap = lp.all_types_to_keymap({**lsst_default_config, **lephare})
    lp.data_retrieval.get_auxiliary_data(
        keymap=keymap,
        additional_files=["examples/output.para", "sed/GAL/BC03_CHAB/AGE_BC03COMB.dat"],
    )
    gal = {"MOD_EXTINC": "1,12,1,12,1,12,1,12", "EM_DISPERSION": "0.5,1.,2."}
    return {
        **{f"lephare.{k}": v for k, v in lephare.items()},
        **{f"gal.{k}": v for k, v in gal.items()},
    }


def _inform_and_estimate(backend, overrides, train, test, keys, tmp_path):
    inform = LephareInformer.make_stage(
        name=f"inform_phys_{backend}",
        nondetect_val=np.nan,
        model=str(tmp_path / f"phys_{backend}.pkl"),
        hdf5_groupname="",
        zmin=0,
        zmax=3,
        nzbins=31,
        backend=backend,
        dtype="float64",
        **overrides,
    )
    inform.inform(train)
    estimate = LephareEstimator.make_stage(
        name=f"est_phys_{backend}",
        nondetect_val=np.nan,
        model=inform.get_handle("model"),
        hdf5_groupname="",
        output=str(tmp_path / f"est_phys_{backend}.hdf5"),
        output_keys=keys,
        backend=backend,
        dtype="float64",
    )
    return inform.model, estimate.estimate(test).data.ancil


def _load_data(test_data_dir, n_test=60):
    train = tables_io.read(os.path.join(test_data_dir, "output_table_conv_train.hdf5"))
    test = tables_io.read(os.path.join(test_data_dir, "output_table_conv_test.hdf5"))
    return train, {k: np.asarray(v)[:n_test] for k, v in test.items()}


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "jax_hybrid",
            marks=pytest.mark.skipif(not jax_available(), reason="jax not installed"),
        ),
        pytest.param(
            "cuda_hybrid",
            marks=pytest.mark.skipif(
                not cuda_available(), reason="cuda/cupy unavailable"
            ),
        ),
        pytest.param(
            "cuda_full",
            marks=pytest.mark.skipif(
                not cuda_available(), reason="cuda/cupy unavailable"
            ),
        ),
    ],
)
def test_estimator_galaxy_properties_match_cpp(test_data_dir, tmp_path, backend):
    """RAIL accel estimator ancil vs the C++ ``lp.process`` columns (same output_keys)."""
    overrides = _bc03_overrides(tmp_path)
    train, test = _load_data(test_data_dir)
    keys = list(GALAXY_PROPERTY_KEYS)
    _, ref = _inform_and_estimate("lephare", overrides, train, test, keys, tmp_path)
    _, got = _inform_and_estimate(backend, overrides, train, test, keys, tmp_path)
    n_obj = len(test["redshift"])
    for key in keys:
        a = np.asarray(got[key], dtype=np.float64).reshape(n_obj, -1)
        b = np.asarray(ref[key], dtype=np.float64).reshape(a.shape)
        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-6, equal_nan=True, err_msg=key
        )


@pytest.mark.slow
@pytest.mark.skipif(not jax_available(), reason="jax not installed")
def test_jax_full_maglib_metadata_matches_cpp(test_data_dir, tmp_path):
    """jax_full rebuilds MagGal on GPU: its per-template SED metadata must match the
    C++ library (GAL block order is the same; STAR/QSO blocks may be ordered differently).

    ``mag`` / ``kcorr`` follow the jax_full fluxes and are not compared here.
    """
    from lephare.accel.maglib import load_maglib

    overrides = _bc03_overrides(tmp_path)
    train, test = _load_data(test_data_dir, n_test=20)
    keys = list(GALAXY_PROPERTY_KEYS)
    model_c, _ = _inform_and_estimate(
        "jax_hybrid", overrides, train, test, keys[:1], tmp_path
    )
    model_j, ancil_j = _inform_and_estimate(
        "jax_full", overrides, train, test, keys, tmp_path
    )
    ref = load_maglib(model_c["maglib_path"])
    jx = load_maglib(model_j["maglib_path"])
    g0 = np.flatnonzero(ref.types == 0)
    g1 = np.flatnonzero(jx.types == 0)
    assert g0.size == g1.size
    for field in (
        "z",
        "nummod",
        "age",
        "ebv",
        "extlaw",
        "luv",
        "lopt",
        "lnir",
        "mass",
        "sfr",
    ):
        np.testing.assert_array_equal(
            getattr(jx, field)[g1], getattr(ref, field)[g0], err_msg=field
        )
    np.testing.assert_array_equal(jx.index_z0[g1] - g1[0], ref.index_z0[g0] - g0[0])
    # dust luminosity uses the Python extinction port (C++ 1.0 interpolates differently)
    np.testing.assert_allclose(jx.ltir[g1], ref.ltir[g0], rtol=0, atol=1e-3)
    assert np.any(ref.ltir[g0] > 0)  # dust energy balance actually exercised
    for key in keys:  # every requested key is computed (NaN means "not provided")
        assert not np.all(np.isnan(np.asarray(ancil_j[key], dtype=np.float64))), key
