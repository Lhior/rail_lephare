import importlib
import os
from typing import Any

import lephare as lp
import numpy as np
import qp
from astropy.table import Table
from ceci.config import StageParameter as Param
from ceci.config import StageConfig
from rail.core.common_params import SHARED_PARAMS
from rail.estimation.estimator import CatEstimator, CatInformer

# We start with the COSMOS default and override with LSST specific values.
lsst_default_config = lp.default_cosmos_config.copy()
lsst_default_config.update(
    {
        "VERBOSE": "NO",
        "CAT_IN": "undefined",
        "ERR_SCALE": "0.02",
        "FILTER_CALIB": "0",
        "FILTER_FILE": "filter_lsst",
        "FILTER_LIST": "lsst/total_u.pb,lsst/total_g.pb,lsst/total_r.pb,lsst/total_i.pb,lsst/total_z.pb,lsst/total_y.pb",
        "GAL_LIB": "LSST_GAL_BIN",
        "GAL_LIB_IN": "LSST_GAL_BIN",
        "GAL_LIB_OUT": "LSST_GAL_MAG",
        "GLB_CONTEXT": "0",
        "INP_TYPE": "M",
        "MABS_CONTEXT": "0",
        "MABS_REF": "1",
        "MAG_REF": "2",
        "MAG_ABS_QSO": "-30,-20.5",
        "QSO_LIB": "LSST_QSO_BIN",
        "QSO_LIB_IN": "LSST_QSO_BIN",
        "QSO_LIB_OUT": "LSST_QSO_MAG",
        "STAR_LIB": "LSST_STAR_BIN",
        "STAR_LIB_IN": "LSST_STAR_BIN",
        "STAR_LIB_OUT": "LSST_STAR_MAG",
        "ZPHOTLIB": "LSST_STAR_MAG,LSST_GAL_MAG,LSST_QSO_MAG",
        "Z_STEP": "0.02,0.,6.",
    }
)

lsst_phys_config = lsst_default_config.copy()

lsst_phys_config.update(
    {
        # In order to get the physical parameters you need to use
        # Composite Stellar Population synthesis models. Here Bruzual & Charlot (2003).
        # This can be done only for galaxies.
        "GAL_SED": "$LEPHAREDIR/sed/GAL/BC03_CHAB/BC03COMB_MOD.list",
        "GAL_LIB": "LIB_BC03",
        "SEL_AGE": "$LEPHAREDIR/sed/GAL/BC03_CHAB/AGE_BC03COMB.dat",
        # Library
        "GAL_LIB_IN":"LIB_BC03",
        "GAL_LIB_OUT":"BC03_LSST",
        "MOD_EXTINC": "0,12,0,12",
        "EXTINC_LAW": "SB_calzetti.dat,SMC_prevot.dat",
        "EM_LINES": "PHYS",
        "EM_DISPERSION": "1.",
        # FIT
        # We set the redshift to the spec-z value
        "ZPHOTLIB":"BC03_LSST",
        "ZFIX": "YES",
        "SPEC_OUT": "spec_phys",  # We would like to see the output
    }
)

star_default_config = dict(LIB_ASCII="YES")

gal_default_config = dict(
    LIB_ASCII="YES",
    MOD_EXTINC="18,26,26,33,26,33,26,33",
    EXTINC_LAW="SMC_prevot.dat,SB_calzetti.dat,SB_calzetti_bump1.dat,SB_calzetti_bump2.dat",
    EM_LINES="EMP_UV",
    EM_DISPERSION="0.5,0.75,1.,1.5,2.",
)

qso_default_config = dict(
    LIB_ASCII="YES",
    MOD_EXTINC="0,1000",
    EB_V="0.,0.1,0.2,0.3",
    EXTINC_LAW="SB_calzetti.dat",
)


ACCEL_BACKENDS = ("jax_hybrid", "cuda_hybrid", "jax_full", "cuda_full")


def _backend_params() -> dict[str, Param]:
    """Config options selecting the compute backend (see ``lephare.accel``)."""
    return dict(
        backend=Param(
            str,
            "lephare",
            msg=(
                "Compute backend: 'lephare' (C++ default), or one of "
                f"{', '.join(ACCEL_BACKENDS)} from lephare.accel. '*_hybrid' "
                "builds the template library with C++ and fits on JAX/CUDA; "
                "'*_full' also builds the library on the GPU."
            ),
        ),
        dtype=Param(
            str,
            "float64",
            msg="Floating dtype for accelerated backends (float32 is faster, float64 matches C++).",
        ),
    )


def _add_sub_config(
    config: dict[str, Any],
    sub_config: dict[str, Any],
    prefix: str,
) -> None:
    """Add all sub-config parameters to the stage config with
    the requested prefix.  This will make the correct parameter types
    and defaults
    """
    for key, val in sub_config.items():
        dtype = type(val)
        default = val
        param = Param(dtype=dtype, default=default)
        config[f"{prefix}{key}"] = param


def _get_sub_config(config: StageConfig, prefix: str) -> dict[str, Any]:
    """Extract all config parameters that start with a
    particular prefix into a dict"""
    out_dict = {
        key[len(prefix) :]: val for key, val in config.items() if key.find(prefix) == 0
    }
    return out_dict


class LephareInformer(CatInformer):
    """Inform stage for LephareEstimator

    This class will set templates and filters required for photoz estimation.
    """

    name = "LephareInformer"
    entrypoint_function = "inform"  # the user-facing science function for this class
    interactive_function = "lephare_informer"
    config_options = CatInformer.config_options.copy()
    config_options.update(
        zmin=SHARED_PARAMS,
        zmax=SHARED_PARAMS,
        nzbins=SHARED_PARAMS,
        nondetect_val=SHARED_PARAMS,
        mag_limits=SHARED_PARAMS,
        bands=SHARED_PARAMS,
        err_bands=SHARED_PARAMS,
        ref_band=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
        **_backend_params(),
    )
    _add_sub_config(config_options, lsst_default_config, "lephare.")
    _add_sub_config(config_options, star_default_config, "star.")
    _add_sub_config(config_options, gal_default_config, "gal.")
    _add_sub_config(config_options, qso_default_config, "qso.")

    def __init__(self, args, **kwargs):
        """Init function, init config stuff (COPIED from rail_bpz)"""
        super().__init__(args, **kwargs)

    def validate(self):
        if self.config["backend"] not in ("lephare",) + ACCEL_BACKENDS:
            raise ValueError(
                f"Unknown backend {self.config['backend']!r}; "
                f"choose 'lephare' or one of {ACCEL_BACKENDS}"
            )
        # Put something in place to allow for not rerunning the prepare stage
        try:
            self.do_prepare = self.config["do_prepare"]
            if self.do_prepare.__class__ is not bool:
                raise RuntimeError("do_prepare argument must be a bool")
        except KeyError:
            self.do_prepare = True

        # We need to ensure the requested redshift grid is propagated
        self.zmin = self.config["zmin"]
        self.zmax = self.config["zmax"]
        self.nzbins = self.config["nzbins"]
        self.dz = (self.zmax - self.zmin) / (self.nzbins - 1)
        Z_STEP = f"{self.dz},{self.zmin},{self.zmax}"
        self.log.info(
            f"rail_lephare is setting the Z_STEP config to {Z_STEP} based on the informer params."
        )
        self.config["lephare.Z_STEP"] = Z_STEP
        # Extract the lephare sub-config only now, so that it contains the
        # redshift grid set above rather than the lephare default Z_STEP.
        self.lephare_config = _get_sub_config(self.config, "lephare.")
        # We create a run directory with the informer name
        self.run_dir = _set_run_dir(self.config["name"])

    def _set_config(self, lephare_config):
        """Update the lephare config

        Parameters
        ==========
        lephare_config : `dict`
            A dictionary of the lephare config keywords.
        """
        self.lephare_config = lephare_config

    def run(self):
        """Run rail_lephare inform stage.

        This informer takes the config and templates and makes the inputs
        required for the run.

        In addition to the three lephare stages making the filter, sed, and
        magnitude libraries we also do some tasks required by all rail inform
        stages.
        """
        # Set training data required for all informers?
        if self.config.hdf5_groupname:
            training_data = self.get_data("input")[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data("input")

        # Get number of sources
        ngal = len(training_data[self.config.ref_band])

        star_config = _get_sub_config(self.config, "star.")
        gal_config = _get_sub_config(self.config, "gal.")
        qso_config = _get_sub_config(self.config, "qso.")

        backend = self.config["backend"]
        tensors = None
        maglib_path = None
        # The three main lephare specific inform tasks
        if self.do_prepare and backend == "lephare":
            lp.prepare(
                self.lephare_config,
                star_config=star_config,
                gal_config=gal_config,
                qso_config=qso_config,
            )
        elif self.do_prepare:
            from lephare.accel.maglib import save_maglib

            tensors = _build_maglib(
                backend, self.lephare_config, star_config, gal_config, qso_config
            )
            maglib_path = os.path.join(self.run_dir, "maglib_tensors.npz")
            save_maglib(tensors, maglib_path)
        else:
            self.log.info(
                f"do_prepare set to False, using precomputed files in {self.run_dir}."
            )

        # Spectroscopic redshifts
        self.szs = training_data[self.config.redshift_col]

        # Run auto adapt on training sample
        input_table = _rail_to_lephare_input(
            training_data, self.config.bands, self.config.err_bands
        )
        if backend in ("jax_full", "cuda_hybrid", "cuda_full"):
            # jax_full has no C++ magnitude library to adapt against
            offsets = _accel_offsets(
                backend,
                self.lephare_config,
                training_data,
                self.config.bands,
                self.config.err_bands,
                self.config.redshift_col,
                tensors,
                self.run_dir,
            )
        else:
            # This will return zeros if AUTO_ADAPT is NO
            offsets = lp.calculate_offsets_from_input(self.lephare_config, input_table)
        # Give principle inform config 'model' to instance.
        self.model = dict(
            lephare_version=lp.__version__,
            lephare_config=self.lephare_config,
            offsets=offsets,
            run_dir=self.run_dir,
            star_config=star_config,
            gal_config=gal_config,
            qso_config=qso_config,
            backend=backend,
            maglib_path=maglib_path,
            dtype=self.config["dtype"],
        )
        self.add_data("model", self.model)


class LephareEstimator(CatEstimator):
    """LePhare-base CatEstimator"""

    name = "LephareEstimator"
    entrypoint_function = "estimate"  # the user-facing science function for this class
    interactive_function = "lephare_estimator"
    config_options = CatEstimator.config_options.copy()

    # Add Lephare-specific configuration options here
    config_options.update(
        nondetect_val=SHARED_PARAMS,
        mag_limits=SHARED_PARAMS,
        bands=SHARED_PARAMS,
        ref_band=SHARED_PARAMS,
        err_bands=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
        lephare_config_from_model=Param(
            bool,
            True,
            "Load lephare config keymap from model",
        ),
        use_inform_offsets=Param(
            bool,
            True,
            msg="Use the zero point offsets computed in the inform stage.",
        ),
        posterior_output=Param(
            int,
            11,
            msg=(
                "Which posterior distribution to output."
                "MASS: 0"
                "SFR: 1"
                "SSFR: 2"
                "LDUST: 3"
                "LIR: 4"
                "AGE: 5"
                "COL1: 6"
                "COL2: 7"
                "MREF: 8"
                "MIN_ZG: 9"
                "MIN_ZQ: 10"
                "BAY_ZG: 11"
                "BAY_ZQ: 12"
            ),
        ),
        output_keys=Param(
            list,
            ["Z_BEST", "CHI_BEST", "ZQ_BEST", "CHI_QSO", "MOD_STAR", "CHI_STAR"],
            msg=(
                "The output keys to add to ancil. These must be in the "
                "output para file. By default we include the best galaxy "
                "and QSO redshift and best star alongside their respective "
                "chi squared. Accelerated backends also support the galaxy "
                "properties MOD_BEST, SCALE_BEST, {MASS,SFR,SSFR,LDUST,AGE}_BEST, "
                "EBV_BEST, EXTLAW_BEST, LUM_{NUV,R,K}_BEST, "
                "{MASS,SFR,SSFR,LDUST,AGE,COL1,COL2,MREF}_{MED,INF,SUP} and "
                "MAG_ABS(), EMAG_ABS(), K_COR(), MABS_FILT(), MAG_MOD() "
                "(per-band 2D arrays); other keys are NaN."
            ),
        ),
        run_dir=Param(
            str,
            "None",
            msg=(
                "Override for the LEPHAREWORK directory. If None we load it "
                "from the model which is set during the inform stage. This "
                "is to facilitate manually moving intermediate files."
            ),
        ),
        write_outputs=Param(
            bool,
            False,
            msg="Whether to write the output files.",
        ),
        **_backend_params(),
    )
    _add_sub_config(config_options, lsst_default_config, "lephare.")

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.lephare_config: dict = {}
        self.zmin: float | None = None
        self.zmax: float | None = None
        self.nzbins: int | None = None
        self._tensors = None

    def _resolved_backend(self) -> str:
        """The configured backend, or the informer's when left at 'lephare'."""
        backend = self.config["backend"]
        if backend == "lephare" and isinstance(self.model, dict):
            if self.model.get("backend") in ACCEL_BACKENDS:
                return self.model["backend"]
        return backend

    def run(self) -> None:
        """Estimate loop. Accel backends skip the per-chunk ``gc.collect`` (it stalls the GPU)."""
        if self.config["backend"] == "lephare":
            super().run()
            return

        self.open_model(**self.config)
        iterator = self.input_iterator("input")
        first = True
        self._initialize_run()
        self._output_handle = None
        for s, e, test_data in iterator:
            self.log.info(
                f"Process {self.rank} running estimator on chunk {s:,} - {e:,}"
            )
            self._process_chunk(s, e, test_data, first)
            first = False
        self._finalize_run()

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.config["lephare_config_from_model"]:
            self.lephare_config = self.model["lephare_config"]
        else:
            self.lephare_config = _get_sub_config(self.config, "lephare.")
        # String values, in case a keymap was passed to the estimate stage
        self.lephare_config = lp.keymap_to_string_dict(
            lp.all_types_to_keymap(self.lephare_config)
        )

        Z_STEP = self.model["lephare_config"]["Z_STEP"]
        self.lephare_config["Z_STEP"] = Z_STEP
        self.dz = float(Z_STEP.split(",")[0])
        self.zmin = float(Z_STEP.split(",")[1])
        self.zmax = float(Z_STEP.split(",")[2])
        self.nzbins = int((self.zmax - self.zmin) / self.dz)
        if self.config["run_dir"] == "None":
            self.run_dir = self.model["run_dir"]
        else:
            self.run_dir = self.config["run_dir"]
        _update_lephare_env(None, self.run_dir)

    def _process_chunk(self, start, end, data, first):
        """Process an individual chunk of sources using lephare.

        Run the equivalent of zphota and get the PDF for every source.
        """
        if self._resolved_backend() != "lephare":
            self._process_chunk_accel(start, end, data, first)
            return
        # Create the lephare input table
        input_table = _rail_to_lephare_input(
            data, self.config.bands, self.config.err_bands
        )
        # Set the desired offsets estimate config overide lephare config overide inform offsets
        if self.config["use_inform_offsets"] and self.model["offsets"] is not None:
            offsets = self.model["offsets"]
            self.lephare_config["APPLY_SYSSHIFT"] = ",".join([str(o) for o in offsets])

        output, photozlist = lp.process(
            self.lephare_config,
            input_table,
            write_outputs=self.config["write_outputs"],
        )

        ng = data[self.config.bands[0]].shape[0]
        # Unpack the pdfs for galaxies
        pdfs = []
        posterior_output = self.config["posterior_output"]
        for i in range(ng):
            pdf = np.array(photozlist[i].pdfmap[posterior_output].vPDF)
            pdfs.append(pdf)
        zgrid = np.array(photozlist[i].pdfmap[posterior_output].xaxis)
        pdfs = np.array(pdfs)
        self.zgrid = zgrid
        zmode = np.zeros(ng)
        zmean = np.zeros(ng)

        for i in range(ng):
            # Take median in case multiple probability densities are equal
            zmode[i] = np.median(
                zgrid[np.where(pdfs[i] == np.max(pdfs[i]))[0].astype(int)]
            )
            zmean[i] = (zgrid * pdfs[i]).sum() / pdfs[i].sum()

        qp_dstn = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=np.array(pdfs)))
        ancil = dict(zmode=zmode, zmean=zmean)
        # Add the requested outputs.
        for c in self.config["output_keys"]:
            if output[c].dtype.kind in ["U", "O"]:
                output[c] = output[c].astype("S")
            ancil[c] = np.array(output[c])
        qp_dstn.set_ancil(ancil)
        self._do_chunk_output(qp_dstn, start, end, first, data=data)

    def _process_chunk_accel(self, start, end, data, first):
        """Process a chunk with a ``lephare.accel`` backend (BAY_ZG posterior only)."""
        from lephare.accel.dispatch import process_with_backend
        from lephare.accel.kernels_ref import pdf_zmode_zmean

        dtype_name = self.config["dtype"]
        if dtype_name == "float64" and self.model.get("dtype"):
            dtype_name = self.model["dtype"]
        offsets = None
        if self.config["use_inform_offsets"] and self.model["offsets"] is not None:
            offsets = np.asarray(self.model["offsets"], dtype=np.float64)

        mags = np.column_stack(
            [np.asarray(data[b], dtype=np.float64) for b in self.config.bands]
        )
        magerrs = np.column_stack(
            [np.asarray(data[b], dtype=np.float64) for b in self.config.err_bands]
        )
        backend = self._resolved_backend()
        tensors = self._accel_tensors()
        context = data["context"] if "context" in data else None
        np_dtype = np.float64 if dtype_name == "float64" else np.float32
        result = process_with_backend(
            backend,
            mags,
            magerrs,
            tensors,
            context=context,
            offsets=offsets,
            dtype=np_dtype,
        )
        # Backends return (pdfs, zgrid[, zmode, zmean[, ancil]])
        pdfs, zgrid = result[0], result[1]
        if len(result) >= 4:
            zmode, zmean = result[2], result[3]
        else:
            zmode, zmean = pdf_zmode_zmean(pdfs, zgrid)
        backend_ancil = result[4] if len(result) == 5 else {}
        output_keys = list(self.config["output_keys"])
        if backend_ancil:
            from lephare.accel.physical import galaxy_properties

            # Galaxy properties (masses, SFRs, absolute magnitudes, ...) from the best fit
            backend_ancil = dict(backend_ancil)
            backend_ancil.update(
                galaxy_properties(
                    output_keys,
                    mags,
                    magerrs,
                    tensors,
                    backend_ancil,
                    self.lephare_config,
                    context=context,
                    offsets=offsets,
                    backend=backend,
                    dtype=np_dtype,
                )
            )
        self.zgrid = zgrid

        ng = mags.shape[0]
        ancil = dict(zmode=zmode, zmean=zmean)
        for c in output_keys:
            if c in backend_ancil and not c.startswith("_"):
                ancil[c] = np.asarray(backend_ancil[c], dtype=np.float64)
            else:
                ancil[c] = np.full(ng, np.nan)
        # PDFs are already normalized on the device
        qp_dstn = qp.Ensemble(
            qp.interp,
            data=dict(xvals=zgrid, yvals=np.asarray(pdfs), norm=False, warn=False),
        )
        qp_dstn.set_ancil(ancil)
        self._do_chunk_output(qp_dstn, start, end, first, data=data)

    def _accel_tensors(self):
        """Template tensors saved by the informer, or rebuilt from the C++ library."""
        if self._tensors is None:
            from lephare.accel.mag_builder import build_maglib_from_prepare
            from lephare.accel.maglib import load_maglib

            maglib_path = self.model.get("maglib_path")
            if maglib_path and os.path.isfile(maglib_path):
                self._tensors = load_maglib(maglib_path)
            else:
                self._tensors = build_maglib_from_prepare(self.lephare_config)
        return self._tensors


def _rail_to_lephare_input(data, mag_cols, mag_err_cols):
    """Take the rail data input and convert it to that expected by lephare

    Parameters
    ==========
    data : OrderedDict
        The RAIL input data chunk being and ordered dictionary of magnitude arrays.
    mag_cols : list
        The names of the magnitude or flux columns.
    mag_err_cols : list
        The names of the magnitude or flux error columns.

    Returns
    =======
    input_table : astropy.table.Table
        The lephare input


    """
    ng = data[mag_cols[0]].shape[0]
    # Make input catalogue in standard lephare format
    input_table = Table()
    try:
        input_table["id"] = data["id"]
    except KeyError:
        input_table["id"] = np.arange(ng)
    # Add all available magnitudes

    context = np.full(ng, 0)
    for n in np.arange(len(mag_cols)):
        input_table[mag_cols[n]] = data[mag_cols[n]].T
        input_table[mag_err_cols[n]] = data[mag_err_cols[n]].T
        # Shall we allow negative fluxes?
        mask = input_table[mag_cols[n]] > 0
        mask &= ~np.isnan(input_table[mag_cols[n]])
        mask &= input_table[mag_err_cols[n]] > 0
        mask &= ~np.isnan(input_table[mag_err_cols[n]])
        context += mask * 2**n
    # Set context to data value if set or else exclude all negative and nan values
    try:
        input_table["context"] = data["context"]
    except KeyError:
        input_table["context"] = context
    try:
        input_table["zspec"] = data["redshift"]
    except KeyError:
        input_table["zspec"] = np.full(ng, -99.0)
    input_table["string_data"] = " "
    return input_table


def _build_maglib(backend, lephare_config, star_config, gal_config, qso_config):
    """Build the template tensors for an accel backend.

    ``*_hybrid`` and ``cuda_full`` run the classic ``lp.prepare`` first;
    ``cuda_full`` then recomputes the magnitudes on the GPU and checks them
    against C++. ``jax_full`` computes the magnitudes with JAX only.
    """
    sub_configs = dict(
        star_config=star_config, gal_config=gal_config, qso_config=qso_config
    )
    if backend == "jax_full":
        from lephare.accel.jax_maggal import build_maglib_jax_from_lib_bin

        return build_maglib_jax_from_lib_bin(
            lephare_config, integrator="jax", **sub_configs
        )

    from lephare.accel.mag_builder import (
        assert_maglib_parity,
        build_maglib_from_prepare,
        rebuild_maglib_cuda,
    )

    tensors = build_maglib_from_prepare(lephare_config, **sub_configs)
    if backend == "cuda_full":
        rebuilt = rebuild_maglib_cuda(tensors, config=lephare_config)
        assert_maglib_parity(tensors, rebuilt)
        tensors = rebuilt
    return tensors


def _accel_offsets(
    backend,
    lephare_config,
    training_data,
    mag_cols,
    mag_err_cols,
    redshift_col,
    tensors,
    run_dir,
):
    """AUTO_ADAPT zero-point offsets on the accel template tensors (C++ parity)."""
    from lephare.accel.autoadapt import calculate_offsets
    from lephare.accel.maglib import export_maglib_from_photoz, load_maglib

    if tensors is None:
        path = os.path.join(run_dir, "maglib_tensors.npz")
        if os.path.isfile(path):
            tensors = load_maglib(path)
        else:
            photz = lp.PhotoZ(lp.all_types_to_keymap(lephare_config))
            tensors = export_maglib_from_photoz(photz, lephare_config)

    mags = np.column_stack(
        [np.asarray(training_data[c], dtype=np.float64) for c in mag_cols]
    )
    magerrs = np.column_stack(
        [np.asarray(training_data[c], dtype=np.float64) for c in mag_err_cols]
    )
    zs = np.asarray(training_data[redshift_col], dtype=np.float64)
    context = training_data["context"] if "context" in training_data else None
    args = (mags, magerrs, zs, tensors, lephare_config)
    if backend.startswith("cuda"):
        try:
            from lephare.accel.cuda_autoadapt import calculate_offsets_cuda

            return np.asarray(
                calculate_offsets_cuda(*args, context=context, verbose=False)
            )
        except ImportError:
            pass
    return np.asarray(calculate_offsets(*args, context=context, verbose=False))


def _update_lephare_env(lepharedir, lepharework):
    """Update the environment variables and reset the lephare package.

    We may be using the same Python session to run inform with different
    settings. These produce intermediate files which are distinct and we must
    use different runs.

    This simple function updates the environment variables and reloads lephare
    to ensure they are properly used.
    """
    if lepharedir is not None:
        os.environ["LEPHAREDIR"] = lepharedir
    if lepharework is not None:
        os.environ["LEPHAREWORK"] = lepharework
    importlib.reload(lp)


def _set_run_dir(name=None, full_path=None):
    """Create a named run if it doesn't exist otherwise set it to existing.

    lephare has the functionality to set a timed or named run. In general we
    must ensure that each inform run has a distinct run to ensure that
    intermediate files are not overwritten.

    Parameters
    ==========
    name : str
        The name to set the run. If not set we use a default timestamped run.
    full_path : str
        If set we create a run directory wherever the user sets it.
    """
    if name is None:
        run_directory = lp.dm.create_new_run()
    elif full_path:
        run_directory = full_path
    else:
        run_directory = os.path.realpath(f"{lp.dm.lephare_work_dir}/../{name}")
    _update_lephare_env(lp.LEPHAREDIR, run_directory)
    return run_directory
