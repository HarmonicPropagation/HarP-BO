"""Load the saved 4D GP and plot paper-style slices through BO iteration 119.
Checkpoint k contains observations through k but stores the GP fitted before
simulation k, i.e. using observations through k-1. No simulations by default.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize

from utils import (
    confirm_surrogate_optimum,
    find_surrogate_maximum,
    get_objective_transform,
    physical_to_x,
    make_kernel,
)
from slice_plot_utils import plot_gp_slices


dtype             = torch.double
dim               = 4
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
results_root      = PROJECT_ROOT / "results"
plots_root        = PROJECT_ROOT / "figures"
_DEFAULT_RUN_NAME = (
    "bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_"
    "straddle100per_max3_matern52_sqrtCE"
)
RUN_DIR           = (
    Path(os.environ["RUN_DIR"]) if "RUN_DIR" in os.environ
    else results_root / _DEFAULT_RUN_NAME
)
print(f"Using run directory: {RUN_DIR}")
DATA_CSV           = RUN_DIR / "data.csv"
CHECKPOINT_TO_PLOT = 120  # None -> newest existing checkpoint

REFERENCE_POINT = {
    "Lzzr": 0.08,
    "pzr": 300.0,
    "intensity_Wcm2": 2.5e14,
    "tau_fs": 22.0,
}

config = {
    "dimension": 4,
    "optimization_space": "log_p_over_L__log_pL__I__inv_sqrt_tau",
    "Lzzr_bounds": (1.5 / 100.0, 1.5),
    "pzr_bounds": (300.0 / 100.0, 300.0),
    "intensity_Wcm2_bounds": (0.01e14, 10e14),
    "tau_fs_bounds": (22.0 / 2.0, 4.0 * 22.0),
}

# No additional HarP simulations unless explicitly requested.
CONFIRM_OPTIMUM         = os.environ.get("CONFIRM_OPTIMUM", "0") == "1"
CONFIRM_REFERENCE_POINT = os.environ.get("CONFIRM_REFERENCE_POINT", "0") == "1"

HARP_CONFIG = {
    "harp_workdir": Path("~/HarP-main").expanduser().resolve(),
    "nr": 1024,
    "nz": 256,
    "lr_factor": 3.0,
    "lt_factor": 3.0,
    "dt_si": 1.2890625e-16,
    "temperature_K": 293.0,
    "wavelength_m": 0.81e-6,
    "zR_mm": 18.6,
    "harmonic_order": 23,
    "objective_scale": 1.0e-6,
}


def _predict_physical_ce(model, f_inv, physical_values, run_config, dtype=torch.double):
    """Predict CE at one physical point using the saved GP."""
    X = torch.stack([physical_to_x(physical_values, run_config)]).to(dtype=dtype)
    with torch.no_grad():
        pred = f_inv(model.posterior(X).mean)
    return float(pred.reshape(-1)[0].detach().cpu().item())


def main():
    plots_root.mkdir(parents=True, exist_ok=True)
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Cannot find {DATA_CSV}")

    available = sorted((RUN_DIR / "models").glob("model_iter_*.pt"))
    if not available:
        raise FileNotFoundError(f"No saved GP checkpoints in {RUN_DIR / 'models'}")
    checkpoint_iteration = (
        int(available[-1].stem.rsplit("_", 1)[-1])
        if CHECKPOINT_TO_PLOT is None else CHECKPOINT_TO_PLOT
    )
    if checkpoint_iteration < 1:
        raise ValueError("A fitted GP requires a checkpoint numbered at least 1")
    model_path = RUN_DIR / "models" / f"model_iter_{checkpoint_iteration:03d}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing requested checkpoint: {model_path}")

    # The stored GP is fitted through k-1, not through k.
    gp_iteration = checkpoint_iteration - 1
    df = pd.read_csv(DATA_CSV)
    df_filtered = df.loc[df["iteration"] <= gp_iteration].copy()
    if df_filtered.empty:
        raise ValueError(f"No BO observations through iteration {gp_iteration}")
    if not (df_filtered["iteration"] == gp_iteration).any():
        raise ValueError(f"Missing observation at iteration {gp_iteration} in {DATA_CSV}")

    print(f"Loading checkpoint {checkpoint_iteration}; GP trained through iteration {gp_iteration}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    X_saved = checkpoint["X_normalized"].to(dtype=dtype)
    Y_saved = checkpoint["Y_objective"].to(dtype=dtype)
    if len(X_saved) != checkpoint_iteration + 1 or len(Y_saved) != len(X_saved):
        raise ValueError(
            f"Unexpected checkpoint size: X={len(X_saved)}, Y={len(Y_saved)}, "
            f"checkpoint={checkpoint_iteration}"
        )
    # Final stored observation is evaluation k, not used in the fitted state.
    X_train = X_saved[:-1]
    Y_obj = Y_saved[:-1]

    checkpoint_config = checkpoint.get("config") or {}
    run_config = {**config, **checkpoint_config}
    if "optimization_space" not in checkpoint_config:
        name = RUN_DIR.name
        if "L__p__I__tau" in name:
            run_config["optimization_space"] = "L__p__I__tau"
        elif "log_p_over_L__log_pL__I__tau" in name:
            run_config["optimization_space"] = "log_p_over_L__log_pL__I__tau"
        elif "log_p_over_L__log_pL__I__inv_sqrt_tau" in name:
            run_config["optimization_space"] = "log_p_over_L__log_pL__I__inv_sqrt_tau"
    print(f"Optimization space: {run_config['optimization_space']}")

    if checkpoint.get("use_log_objective", False):
        transform_name = "log10"
    else:
        transform_name = checkpoint_config.get("objective_transform") or "id"
    f_fwd, f_inv = get_objective_transform(transform_name)
    Y_obj = f_fwd(torch.clamp(Y_obj, min=1e-15))

    kernel = make_kernel(run_config.get("kernel", "matern52"), dim)
    model = SingleTaskGP(
        train_X=X_train,
        train_Y=Y_obj,
        covar_module=kernel,
        outcome_transform=Standardize(m=1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_phys, max_val_transformed, std_transformed = find_surrogate_maximum(
        model, run_config, dtype=dtype
    )
    best_ce_unscaled = float(f_inv(max_val_transformed.item()))
    objective_scale = 1e-6
    lower_transformed = max(0.0, max_val_transformed.item() - std_transformed)
    upper_transformed = max_val_transformed.item() + std_transformed
    ce_lower = float(f_inv(lower_transformed)) * objective_scale
    ce_upper = float(f_inv(upper_transformed)) * objective_scale
    predicted_ce = best_ce_unscaled * objective_scale

    print(
        f"Surrogate-predicted global optimum: CE = {predicted_ce:.4e} "
        f"(1-sigma band [{ce_lower:.4e}, {ce_upper:.4e}]) at "
        f"Lzzr={best_phys['Lzzr']:.4f}, pzr={best_phys['pzr']:.4f}, "
        fr"I={best_phys['intensity_Wcm2']:.4e} $W cm^{-2}$, "
        f"tau={best_phys['tau_fs']:.2f} fs"
    )
    print(f"GP prediction: {predicted_ce / 1e-6:.4f}")
    print(f"Lower error:   {(predicted_ce - ce_lower) / 1e-6:.4f}")
    print(f"Upper error:   {(ce_upper - predicted_ce) / 1e-6:.4f}")

    confirmed_ce = None
    if CONFIRM_OPTIMUM:
        print("Running one confirmatory HarP simulation at the predicted optimum...")
        confirmed_ce = confirm_surrogate_optimum(best_phys, HARP_CONFIG)
        in_band = ce_lower <= confirmed_ce <= ce_upper
        print(
            f"Confirmatory HarP-simulated CE = {confirmed_ce:.4e} "
            f"({confirmed_ce / predicted_ce - 1.0:+.2%} vs. prediction, "
            f"{'within' if in_band else 'OUTSIDE'} the 1-sigma band)"
        )

    reference_point = {
        **REFERENCE_POINT,
        "Lzzr": best_phys["Lzzr"],
        "pzr": best_phys["pzr"],
    }
    reference_predicted_ce = _predict_physical_ce(
        model, f_inv, reference_point, run_config, dtype=dtype
    ) * objective_scale
    print(
        "Intersection point: "
        f"Lzzr={reference_point['Lzzr']:.4f}, pzr={reference_point['pzr']:.2f}, "
        f"I={reference_point['intensity_Wcm2']:.4e} W cm^{-2}, "
        f"tau={reference_point['tau_fs']:.2f} fs"
    )
    print(f"Intersection-point GP CE = {reference_predicted_ce:.4e}")

    reference_confirmed_ce = None
    if CONFIRM_REFERENCE_POINT:
        print("Running one confirmatory HarP simulation at the intersection point...")
        reference_confirmed_ce = confirm_surrogate_optimum(reference_point, HARP_CONFIG)
        print(
            f"Intersection-point HarP CE = {reference_confirmed_ce:.4e} "
            f"({reference_confirmed_ce / reference_predicted_ce - 1.0:+.2%} vs. GP prediction)"
        )

    best_row = pd.Series({
        "Lzzr": best_phys["Lzzr"],
        "pzr": best_phys["pzr"],
        "intensity_Wcm2": best_phys["intensity_Wcm2"],
        "tau_fs": best_phys["tau_fs"],
        "conversion_efficiency": predicted_ce,
    })

    out_path_base = plots_root / f"slices_4d_iter_{gp_iteration:03d}"
    plot_gp_slices(
        model=model,
        f_inv=f_inv,
        best_row=best_row,
        config=run_config,
        out_path_base=out_path_base,
        dtype=dtype,
        df_valid=df_filtered,
        objective_scale=objective_scale,
        confirmed_ce=confirmed_ce,
        ce_band=(ce_lower, ce_upper),
        reference_i=reference_point["intensity_Wcm2"],
        reference_tau=reference_point["tau_fs"],
        reference_l=reference_point["Lzzr"],
        reference_p=reference_point["pzr"],
        reference_confirmed_ce=reference_confirmed_ce,
    )
    print(f"Saved slices for fitted GP iteration {gp_iteration} (checkpoint {checkpoint_iteration})")


if __name__ == "__main__":
    main()

