"""
End-to-end smoke test for the harp_input.py <-> HarP calling convention.

The other tests only exercise coordinate-transform math in utils.py; none of
them actually invoke a real HarP simulation, so a regression in how we invoke
HarP.py (e.g. the namelist-by-path vs. namelist-by-module-name calling
convention) is invisible to CI. This test runs one real, deliberately tiny
HarP simulation to catch exactly that class of regression.

Skipped (not failed) when no real HarP installation is available at
~/HarP-main, since a physical HarP checkout isn't a hard requirement for
running the rest of the test suite.
"""

import math
from pathlib import Path

import pytest

from harp_input import check_harp_installation, run_harp

HARP_WORKDIR = Path("~/HarP-main").expanduser().resolve()

# Deliberately tiny grid (production runs use nr=1024, nz=256) - keeps this
# test to a couple of seconds while still exercising a real simulation.
CONFIG = {
    "harp_workdir": HARP_WORKDIR,
    "nr": 64,
    "nz": 32,
    "lr_factor": 3.0,
    "lt_factor": 3.0,
    "dt_si": 1.2890625e-16,
    "temperature_K": 293.0,
    "wavelength_m": 0.81e-6,
    "zR_mm": 18.6,
    "harmonic_order": 23,
    "objective_scale": 1.0e-6,
}

# One fixed, mid-range physical point - no need to explore the space here,
# just confirm HarP runs and returns a physically sane result.
VALUES = {"Lzzr": 0.5, "pzr": 150.0, "intensity_Wcm2": 2.5e14, "tau_fs": 22.0}


def test_run_harp_smoke(tmp_path) -> None:
    try:
        paths = check_harp_installation(HARP_WORKDIR)
    except FileNotFoundError:
        pytest.skip(f"No real HarP installation found at {HARP_WORKDIR}")

    archive_path = tmp_path / "diag_smoke.h5"
    result = run_harp(values=VALUES, config=CONFIG, paths=paths, archive_path=archive_path)

    assert math.isfinite(result["conversion_efficiency"])
    assert result["conversion_efficiency"] > 0.0
    assert result["harmonic_energy_J"] > 0.0
    assert archive_path.exists()
