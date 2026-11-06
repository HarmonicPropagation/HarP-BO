import torch
from utils import get_physical_ic_generator

def test_ic_generator() -> None:
    configs = [
        {
            "optimization_space": "L__p__I__tau",
            "Lzzr_bounds": (0.015, 1.5),
            "pzr_bounds": (3.0, 300.0),
            "intensity_Wcm2_bounds": (0.01e14, 10e14),
            "tau_fs_bounds": (11.0, 88.0)
        },
        {
            "optimization_space": "log_p_over_L__log_pL__I__inv_sqrt_tau",
            "Lzzr_bounds": (0.015, 1.5),
            "pzr_bounds": (3.0, 300.0),
            "intensity_Wcm2_bounds": (0.01e14, 10e14),
            "tau_fs_bounds": (11.0, 88.0)
        }
    ]

    for cfg in configs:
        ic_gen = get_physical_ic_generator(cfg, dtype=torch.float64)
        ics = ic_gen(bounds=None, q=1, num_samples=512)

        assert ics.shape == (512, 1, 4), f"Unexpected shape: {ics.shape}"
        assert ics.min().item() >= 0.0, "Values strictly below BO bounds"
        assert ics.max().item() <= 1.0, "Values strictly above BO bounds"

