import torch
import numpy as np
import pandas as pd
from pathlib import Path
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib
matplotlib.use("Agg")
from utils import physical_to_x, get_objective_transform, x_to_physical, get_physical_constraints, get_physical_ic_generator, make_kernel, make_likelihood, DEFAULT_NOISE_MODE
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf
from slice_plot_utils import plot_gp_slices

dtype        = torch.double
dim          = 4
KERNEL       = "matern52"  # baseline GP: same kernel and noise model as the BO runs (bo.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_CSV  = PROJECT_ROOT / "data" / "holdout_4d" / "holdout.csv"
PLOTS_ROOT   = PROJECT_ROOT / "figures"

# We intentionally use the log-invariant space here to train a smarter surrogate,
# even if the data was generated via uniform physical sampling.
config = {
    "optimization_space": "log_p_over_L__log_pL__I__inv_sqrt_tau",
    "Lzzr_bounds": (1.5 / 100.0, 1.5),
    "pzr_bounds": (300.0 / 100.0, 300.0),
    "intensity_Wcm2_bounds": (0.01e14, 10e14),
    "tau_fs_bounds": (22.0 / 2.0, 4.0 * 22.0)
}


def main():
    if not HOLDOUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find holdout data at {HOLDOUT_CSV}")

    df = pd.read_csv(HOLDOUT_CSV)

    # We don't filter the new valid dataset because the Cartesian generation
    # guarantees they are already in bounds. But we keep it for safety.
    valid_mask = (
        (df["Lzzr"] >= config["Lzzr_bounds"][0]) & (df["Lzzr"] <= config["Lzzr_bounds"][1]) &
        (df["pzr"] >= config["pzr_bounds"][0]) & (df["pzr"] <= config["pzr_bounds"][1])
    )
    df_valid = df[valid_mask].copy()
    print(f"Loaded {len(df_valid)} valid points.")

    # 1. Transform physical features to normalized training inputs X
    X_list = []
    for _, row in df_valid.iterrows():
        values = {
            "Lzzr": row["Lzzr"],
            "pzr": row["pzr"],
            "intensity_Wcm2": row["intensity_Wcm2"],
            "tau_fs": row["tau_fs"],
        }
        x = physical_to_x(values, config)
        X_list.append(x)
    X_train = torch.stack(X_list).to(dtype=dtype)

    # Scale objective by 1e-6
    objective_scale = 1e-6
    Y_raw           = torch.tensor(df_valid["conversion_efficiency"].values / objective_scale, dtype=dtype,).unsqueeze(-1)

    # Transform
    transform_name  = "sqrt"
    f_fwd, f_inv    = get_objective_transform(transform_name)
    Y_train         = f_fwd(torch.clamp(Y_raw, min=1e-15))

    # =========================================================================
    # SANITY CHECKS (Pre-Fit)
    # =========================================================================
    print("\n--- SANITY CHECKS ---")
    print(f"1. X_train bounds: Min={X_train.min().item():.4f}, Max={X_train.max().item():.4f} (Should be in [0, 1])")
    print(f"2. Y_raw scale:    Max={Y_raw.max().item():.4f} (Should be around 1 to 10)")
    print(f"3. Y_train scale:  Max={Y_train.max().item():.4f} (Should be sqrt of Y_raw max)")
    assert X_train.min() >= -1e-5 and X_train.max() <= 1.0 + 1e-5, "X_train is out of bounds!"

    # Fit GP
    print("\nFitting SingleTaskGP model...")
    kernel = make_kernel(KERNEL, dim)
    model = SingleTaskGP(
        train_X=X_train,
        train_Y=Y_train,
        covar_module=kernel,
        likelihood=make_likelihood(DEFAULT_NOISE_MODE),
        outcome_transform=Standardize(m=1)
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    # =========================================================================
    # SANITY CHECKS (Post-Fit)
    # =========================================================================
    with torch.no_grad():
        Y_pred_train_transformed = model.posterior(X_train).mean
        # Invert the sqrt transform to get back to the 1e-6 scale
        Y_pred_train_scaled = f_inv(Y_pred_train_transformed)

    mse = torch.nn.functional.mse_loss(Y_pred_train_scaled, Y_raw)
    best_idx = Y_raw.argmax()
    print("4. GP Self-Consistency:")
    print(f"   Overall MSE on training data: {mse.item():.4e}")
    print(f"   True Max CE (scaled):      {Y_raw[best_idx].item():.4f}")
    print(f"   Predicted Max CE (scaled): {Y_pred_train_scaled[best_idx].item():.4f}")
    print("---------------------\n")

    # Plotting
    # Find the 4D Global Optimum of the surrogate model
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=dtype)
    constraints = get_physical_constraints(config)
    candidate, max_val_transformed = optimize_acqf(
        acq_function=PosteriorMean(model),
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        nonlinear_inequality_constraints=constraints,
        ic_generator=get_physical_ic_generator(config, dtype=dtype)
    )
    best_phys        = x_to_physical(candidate.detach().cpu().squeeze(0), config)
    best_ce_unscaled = float(f_inv(max_val_transformed.item()))

    # Override best_row to represent the true surrogate optimum
    best_row = pd.Series({
        "Lzzr": best_phys["Lzzr"],
        "pzr": best_phys["pzr"],
        "intensity_Wcm2": best_phys["intensity_Wcm2"],
        "tau_fs": best_phys["tau_fs"],
        "conversion_efficiency": best_ce_unscaled * objective_scale
    })

    out_path_base = PLOTS_ROOT / "slices_sobol_4d"
    plot_gp_slices(
        model=model,
        f_inv=f_inv,
        best_row=best_row,
        config=config,
        out_path_base=out_path_base,
        dtype=dtype,
        df_valid=df_valid,
        objective_scale=objective_scale,
        # Plotting only: use this Sobol GP's optimum to fix panel (b).
        reference_l=best_phys["Lzzr"],
        reference_p=best_phys["pzr"],
    )


if __name__ == "__main__":
    main()

