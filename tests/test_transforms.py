import torch
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

from utils import (
    physical_to_x, batched_x_to_physical,
    get_physical_constraints
)

CONFIG = {
    "optimization_space": "log_p_over_L__log_pL__I__inv_sqrt_tau",
    "Lzzr_bounds": (0.015, 1.5),
    "pzr_bounds": (3.0, 300.0),
    "intensity_Wcm2_bounds": (0.01e14, 10e14),
    "tau_fs_bounds": (11.0, 88.0)
}

def test_batched_x_to_physical() -> None:
    physical_input = {
        "Lzzr": 0.5,
        "pzr": 150.0,
        "intensity_Wcm2": 2.5e14,
        "tau_fs": 22.0
    }

    # 1D Vector Test
    x_norm = physical_to_x(physical_input, CONFIG)
    L_recovered, p_recovered, _ = batched_x_to_physical(x_norm, CONFIG)

    assert pytest.approx(L_recovered.item(), rel=1e-5) == physical_input["Lzzr"]
    assert pytest.approx(p_recovered.item(), rel=1e-5) == physical_input["pzr"]

    # Batched Tensor Test
    x_batched = x_norm.repeat(512, 1, 1)
    L_batch, p_batch, _ = batched_x_to_physical(x_batched, CONFIG)

    assert L_batch.shape == (512, 1)
    assert p_batch.shape == (512, 1)

# --- HYPOTHESIS PROPERTY TEST ---

@given(
    Lzzr=st.floats(min_value=0.015, max_value=1.5),
    pzr=st.floats(min_value=3.0, max_value=300.0),
    intensity=st.floats(min_value=0.01e14, max_value=10e14),
    tau=st.floats(min_value=11.0, max_value=88.0)
)
@settings(deadline=None) # PyTorch operations can sometimes exceed the default 200ms deadline
def test_transform_roundtrip_property(Lzzr: float, pzr: float, intensity: float, tau: float) -> None:
    """
    Property: Mapping any valid physical point to normalized space
    and back must yield the original physical values.
    """
    physical_input = {
        "Lzzr": Lzzr,
        "pzr": pzr,
        "intensity_Wcm2": intensity,
        "tau_fs": tau
    }

    x_norm = physical_to_x(physical_input, CONFIG)
    L_rec, p_rec, _ = batched_x_to_physical(x_norm, CONFIG)

    assert pytest.approx(L_rec.item(), rel=1e-4) == Lzzr
    assert pytest.approx(p_rec.item(), rel=1e-4) == pzr
