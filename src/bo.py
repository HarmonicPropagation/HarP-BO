"""
This module implements the main Bayesian optimization workflow. It generates
the initial random sample, fits the Gaussian-process surrogate model,
constructs and optimizes the selected acquisition function, executes HarP
simulations at the proposed points, and stores the resulting data, models,
and configuration files. The optimization itself is independent of the physical
parameterization and supports both 2D and 4D optimization through the shared configuration
dictionary.
This is file 4 of 5 in the BO workflow:
    1. run_bo_2d.py   - Defines the setup and starts the 2D Bayesian optimization
    2. run_bo_4d.py   - Defines the setup and starts the 4D Bayesian optimization
    3. harp_input.py  - Prepares, runs, and evaluates one HarP simulation
    4. bo.py          - Bayesian optimization algorithm and acquisition strategy
    5. utils.py       - Shared helper functions and numerical utilities
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from harp_input import check_harp_installation, run_harp
from utils import (
    get_physical_constraints,
    inside_physical_box,
    get_objective_transform,
    make_acquisition,
    make_kernel,
    make_likelihood,
    DEFAULT_NOISE_MODE,
    DEFAULT_GP_FIT,
    fit_gp,
    gp_hyperparameters,
    prepare_run_folder,
    physical_to_x,
    save_run_configuration,
    set_random_seed,
    x_to_physical,
)
# Shared Bayesian optimization loop
########################################################################################################
def _archive_filename(prefix, iteration, values):
    # Create a descriptive filename for the archived HarP diagnostic file
    # The filename includes the iteration number and physical parameters
    return (
        f"DiagEnergy_{prefix}_{iteration:03d}_"
        f"L_{values['Lzzr']:.6f}_"
        f"p_{values['pzr']:.6f}_"
        f"I_{values['intensity_Wcm2']:.6e}_"
        f"tau_{values['tau_fs']:.6f}.h5"
    )
def _result_row(
    # The dictionary is assembled from information coming from several parts of the optimization workflow
    # _result_row() acts as a collector that gathers everything into one place
    optimization_space,
    iteration,
    phase,
    x,
    values,
    result,
    bo_overhead_s,
    cumulative_wall_time_s,
    diag_file,
    acq_value,
):
    pL = values["pzr"] * values["Lzzr"]
    p_over_L = values["pzr"] / values["Lzzr"]
    iteration_wall_time_s = (
        bo_overhead_s + result["simulation_time_s"]
    )
    row = {
        "optimization_space": optimization_space,
        "iteration": iteration,
        "phase": phase,
        "Lzzr": values["Lzzr"],
        "pzr": values["pzr"],
        "pL": pL,
        "p_over_L": p_over_L,
        "intensity_Wcm2": values["intensity_Wcm2"],
        "tau_fs": values["tau_fs"],
        "input_laser_energy_J": result["input_laser_energy_J"],
        "harmonic_energy_J": result["harmonic_energy_J"],
        "conversion_efficiency": result["conversion_efficiency"],
        "objective_scaled": result["objective"],
        "inside_physical_box": inside_physical_box(values, CONFIG_HOLDER),
        "nt": result["nt"],
        "simulation_time_s": result["simulation_time_s"],
        "bo_overhead_s": bo_overhead_s,
        "iteration_wall_time_s": iteration_wall_time_s,
        "cumulative_wall_time_s": cumulative_wall_time_s,
        "acq_value": acq_value,
        "diag_file": str(diag_file),
    }
    for index in range(x.shape[-1]):
        row[f"x{index + 1}_norm"] = float(x[index])
    return row
# Small module level holder used only by _result_row to avoid repeating
# a long list of arguments in every call
CONFIG_HOLDER = None
def _make_feasible_initial_conditions(
    config,
    iteration,
    num_restarts,
    dtype,
):
    # Generate feasible starting points for constrained acquisition optimization.
    # Points are sampled inside the physical parameter box and then transformed
    # into the normalized coordinates used by the selected BO parameterization.
    rng = np.random.default_rng([config["seed"], iteration])  # SeedSequence over the pair (see POOL_SEEDING)
    initial_conditions = []
    for _ in range(num_restarts):
        physical_values = {
            "Lzzr": rng.uniform(
                *config["Lzzr_bounds"]
            ),
            "pzr": rng.uniform(
                *config["pzr_bounds"]
            ),
            "intensity_Wcm2": rng.uniform(
                *config["intensity_Wcm2_bounds"]
            ),
            "tau_fs": rng.uniform(
                *config["tau_fs_bounds"]
            ),
        }
        x_initial = physical_to_x(
            physical_values,
            config,
        ).to(dtype=dtype)
        initial_conditions.append(x_initial)
    # Shape:
    # num_restarts x q x dimension
    # where q = 1
    return torch.stack(
        initial_conditions,
        dim=0,
    ).unsqueeze(1)
def _make_discrete_candidate_pool(
    config,
    iteration,
    num_candidates,
    dtype,
):
    """
    Generate acquisition candidates uniformly in the active transformed BO
    space while enforcing the original physical parameter bounds.
    Fresh candidates are drawn directly in normalized BO coordinates. Each
    point is transformed back to physical variables and rejected if it lies
    outside the allowed physical box. Sampling continues until exactly
    num_candidates valid points have been collected.
    """
    rng = np.random.default_rng([config["seed"], iteration])  # SeedSequence over the pair (see POOL_SEEDING)
    dim = int(config["dimension"])
    candidates = []
    max_attempts = max(1000 * num_candidates, 100000)
    attempts = 0
    while len(candidates) < num_candidates:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Could not generate enough feasible acquisition candidates "
                "in the transformed BO space. "
                f"Accepted {len(candidates)} of {num_candidates} candidates "
                f"after {attempts - 1} attempts. "
                f"Optimization space: {config['optimization_space']}"
            )
        x_candidate = torch.tensor(
            rng.uniform(
                low=0.0,
                high=1.0,
                size=dim,
            ),
            dtype=dtype,
        )
        physical_values = x_to_physical(
            x_candidate,
            config,
        )
        if not inside_physical_box(
            physical_values,
            config,
        ):
            continue
        candidates.append(x_candidate)
    return torch.stack(
        candidates,
        dim=0,
    )
# Per-iteration randomness (candidate pool, legacy restart points) is drawn from
# np.random.default_rng([seed, iteration]), i.e. a SeedSequence over the pair.
# The previous default_rng(seed + iteration) made seed s at iteration i share its
# candidates with seed s+k at iteration i-k (and with seed s+i's initial point), so
# "independent" seeds overlapped. Recorded in run_config.txt as pool_seeding.
POOL_SEEDING = "seedsequence_pair"


# Choose point, Run HarP, Read CE, Update GP, Choose next point
def run_optimization(config):
    # Check paths
    global CONFIG_HOLDER
    CONFIG_HOLDER = config
    set_random_seed(config["seed"])
    paths = check_harp_installation(
        config["harp_workdir"]
    )
    print("HarP workdir:", paths["workdir"])
    print("HarP script:", paths["script"])
    print("HarP input:", paths["input"])
    print("DiagEnergy file:", paths["diag"])
    # Set dimension and paths
    dim = int(config["dimension"])
    dtype = torch.double
    bounds = torch.tensor(
        [[0.0] * dim, [1.0] * dim],
        dtype=dtype,
    )
    run_name, output_folder = prepare_run_folder(config)
    model_folder = output_folder / "models"
    diag_folder = output_folder / "diagnostics"
    model_folder.mkdir()
    diag_folder.mkdir()
    config.setdefault("pool_seeding", POOL_SEEDING)
    # GP hyperparameter fit (see utils.fit_gp); "cold" is the fit of runs without this key.
    config.setdefault("gp_fit", DEFAULT_GP_FIT)
    save_run_configuration(
        config,
        output_folder,
    )
    data_csv = output_folder / "data.csv"
    summary_txt = output_folder / "summary.txt"
    f_fwd, f_inv = get_objective_transform(
        config.get("objective_transform", "id")
    )
    # Initial point
    ####################################################################################################
    print("\n" + "=" * 80)
    print("Preparing initial physical point")
    print("=" * 80)
    if config["n_initial"] != 1:
        raise ValueError(
            "This workflow requires n_initial = 1 because the runner "
            "provides one fixed physical initial point."
        )
    values = {
        "Lzzr": float(
            config["initial_physical_values"]["Lzzr"]
        ),
        "pzr": float(
            config["initial_physical_values"]["pzr"]
        ),
        "intensity_Wcm2": float(
            config["initial_physical_values"]["intensity_Wcm2"]
        ),
        "tau_fs": float(
            config["initial_physical_values"]["tau_fs"]
        ),
    }
    if not inside_physical_box(
        values,
        config,
    ):
        raise ValueError(
            "The initial physical point lies outside the configured bounds.\n"
            f"Initial point: {values}"
        )
    # Convert the physical point into the normalized coordinates of the
    # selected optimization parameterization.
    x_initial = physical_to_x(
        values,
        config,
    ).to(dtype=dtype)
    # Check that the coordinate transformation preserves the physical point.
    reconstructed_values = x_to_physical(
        x_initial,
        config,
    )
    for variable_name in (
        "Lzzr",
        "pzr",
        "intensity_Wcm2",
        "tau_fs",
    ):
        if not np.isclose(
            reconstructed_values[variable_name],
            values[variable_name],
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "The initial point changed during the coordinate "
                "transformation.\n"
                f"Variable: {variable_name}\n"
                f"Original: "
                f"{values[variable_name]:.16e}\n"
                f"Reconstructed: "
                f"{reconstructed_values[variable_name]:.16e}\n"
                f"Optimization space: "
                f"{config['optimization_space']}"
            )
    print(
        f"Optimization space: "
        f"{config['optimization_space']}"
    )
    print(
        f"Initial Lzzr: "
        f"{values['Lzzr']:.16e}"
    )
    print(
        f"Initial pzr:  "
        f"{values['pzr']:.16e}"
    )
    print(
        f"Initial intensity: "
        f"{values['intensity_Wcm2']:.16e}"
    )
    print(
        f"Initial tau:  "
        f"{values['tau_fs']:.16e}"
    )
    print(
        f"Normalized coordinates: "
        f"{x_initial.tolist()}"
    )
    rows = []
    cumulative_wall_time_s = 0.0
    archive_path = (
        diag_folder
        / _archive_filename(
            "initial",
            0,
            values,
        )
    )
    print(
        f"Running initial point 000 | "
        f"L={values['Lzzr']:.6f}, "
        f"p={values['pzr']:.6f}, "
        f"I={values['intensity_Wcm2']:.6e}, "
        f"tau={values['tau_fs']:.6f}"
    )
    result = run_harp(
        values=values,
        config=config,
        paths=paths,
        archive_path=archive_path,
    )
    cumulative_wall_time_s += result["simulation_time_s"]
    rows.append(
        _result_row(
            optimization_space=config["optimization_space"],
            iteration=0,
            phase="initial",
            x=x_initial,
            values=values,
            result=result,
            bo_overhead_s=0.0,
            cumulative_wall_time_s=cumulative_wall_time_s,
            acq_value=np.nan,
            diag_file=archive_path,
        )
    )
    pd.DataFrame(rows).to_csv(
        data_csv,
        index=False,
    )
    # Initial GP training data
    X = x_initial.unsqueeze(0)
    Y_objective = torch.tensor(
        [[result["objective"]]],
        dtype=dtype,
    )
    # Track the peak utility to ensure we only stop after exploration has occurred
    max_utility_seen = -float("inf")
    previous_hyperparameters = None  # warm start for gp_fit = "warm"
    # BO loop
    ####################################################################################################
    for bo_step in range(
        config["n_iterations"]
    ):
        iteration = (
            config["n_initial"]
            + bo_step
        )
        bo_start = time.perf_counter()
        kernel = make_kernel(
            config["kernel"],
            dim,
        )
        # Apply objective transform
        # (clamping at 1e-15 to avoid -inf for log transforms)
        Y_train = f_fwd(
            torch.clamp(
                Y_objective,
                min=1e-15,
            )
        )
        # Standardize the objective values internally before fitting the model
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y_train,
            covar_module=kernel,
            likelihood=make_likelihood(config.get("noise_mode", DEFAULT_NOISE_MODE)),
            outcome_transform=Standardize(m=1),
        )
        gp_fit_info = fit_gp(
            model,
            config["gp_fit"],
            previous=previous_hyperparameters,
        )
        previous_hyperparameters = gp_hyperparameters(model)
        print(f"GP fit ({gp_fit_info['gp_fit']}): kept {gp_fit_info['start']} fit, log marginal likelihood "
              f"cold {gp_fit_info['mll_cold']}, warm {gp_fit_info['mll_warm']}"
              + (f" (cold fit failed: {gp_fit_info['cold_error']})" if "cold_error" in gp_fit_info else ""))
        # Pass Y_train so the acquisition function knows the current scale
        acquisition_function, target_info = make_acquisition(
            model=model,
            Y_objective=Y_train,
            config=config,
            bounds=bounds,
        )
        ################################################################################################
        # DISCRETE ACQUISITION SELECTION TEST
        #
        # Instead of continuously optimizing the acquisition function from
        # multiple restart points, generate a finite pool of physically valid
        # candidates, evaluate the acquisition function directly at each one,
        # and select the candidate with the largest acquisition value.
        ################################################################################################
        num_candidates = 512
        candidate_pool = _make_discrete_candidate_pool(
            config=config,
            iteration=iteration,
            num_candidates=num_candidates,
            dtype=dtype,
        )
        # Analytic acquisition functions expect:
        #
        # batch x q x dimension
        #
        # Here q = 1.
        candidate_pool_q = candidate_pool.unsqueeze(1)
        with torch.no_grad():
            acquisition_values = acquisition_function(
                candidate_pool_q
            )
        # Pick the single candidate with the largest acquisition value.
        best_candidate_index = torch.argmax(
            acquisition_values
        )
        candidate = candidate_pool[
            best_candidate_index
        ].unsqueeze(0)
        acquisition_value = acquisition_values[
            best_candidate_index
        ]
        ################################################################################################
        acq_val_scalar = float(
            acquisition_value
            .detach()
            .reshape(-1)[0]
        )
        # Evaluate utility
        # (exponentiate if operating in log-space)
        utility = (
            np.exp(acq_val_scalar)
            if config["acquisition"] == "log_ei"
            else acq_val_scalar
        )
        # Update our state tracker
        max_utility_seen = max(
            max_utility_seen,
            utility,
        )
        threshold = config.get(
            "acq_stopping_threshold"
        )
        grace_period = config.get(
            "acq_stopping_grace_period",
            15,
        )
        # Break the loop if convergence is reached.
        # Only terminate if we have successfully entered Phase 2
        # (utility > threshold) and have now settled back down
        # into Phase 3 (utility < threshold).
        if (
            threshold is not None
            and bo_step > grace_period
        ):
            if (
                max_utility_seen > threshold
                and utility < threshold
            ):
                print(
                    f"\nConvergence reached at "
                    f"iteration {iteration:03d}."
                )
                print(
                    f"Maximum acquisition utility "
                    f"({utility:.2e}) dropped below "
                    f"threshold ({threshold:.2e})."
                )
                break
        # Candidate already has shape q x dimension.
        # Here q = 1.
        x_next = candidate.detach().to(
            dtype=dtype
        )
        values = x_to_physical(
            x_next[0],
            config,
        )
        # All candidate-pool points were generated directly inside the
        # physical parameter box, so this should always pass.
        if not inside_physical_box(
            values,
            config,
        ):
            print(
                "Invalid candidate after coordinate transformation:"
            )
            print(
                f"L    = {values['Lzzr']:.16e}, "
                f"bounds = {config['Lzzr_bounds']}"
            )
            print(
                f"p    = {values['pzr']:.16e}, "
                f"bounds = {config['pzr_bounds']}"
            )
            print(
                f"I    = {values['intensity_Wcm2']:.16e}, "
                f"bounds = {config['intensity_Wcm2_bounds']}"
            )
            print(
                f"tau  = {values['tau_fs']:.16e}, "
                f"bounds = {config['tau_fs_bounds']}"
            )
            raise RuntimeError(
                "A candidate generated inside the physical box "
                "became invalid after the coordinate transformation."
            )
        bo_overhead_s = (
            time.perf_counter()
            - bo_start
        )
        best_index = torch.argmax(
            Y_objective
        )
        best_ce = float(
            Y_objective[best_index]
            * config["objective_scale"]
        )
        best_values = x_to_physical(
            X[best_index],
            config,
        )
        print(
            f"{config['optimization_space']} | "
            f"iter {iteration:03d} | "
            f"points = {len(X):03d} | "
            f"best CE = {best_ce:.6e} | "
            f"best L = {best_values['Lzzr']:.6f} | "
            f"best p = {best_values['pzr']:.6f} | "
            f"acq = {utility:.4e}"
        )
        print(
            f"Running BO point {iteration:03d} | "
            f"L={values['Lzzr']:.6f}, "
            f"p={values['pzr']:.6f}, "
            f"I={values['intensity_Wcm2']:.6e}, "
            f"tau={values['tau_fs']:.6f}"
        )
        archive_path = (
            diag_folder
            / _archive_filename(
                "iter",
                iteration,
                values,
            )
        )
        result = run_harp(
            values=values,
            config=config,
            paths=paths,
            archive_path=archive_path,
        )
        iteration_wall_time_s = (
            bo_overhead_s
            + result["simulation_time_s"]
        )
        cumulative_wall_time_s += (
            iteration_wall_time_s
        )
        # Add the new point and its objective value to the BO dataset
        X = torch.cat(
            [
                X,
                x_next,
            ],
            dim=0,
        )
        Y_objective = torch.cat(
            [
                Y_objective,
                torch.tensor(
                    [[result["objective"]]],
                    dtype=dtype,
                ),
            ],
            dim=0,
        )
        rows.append(
            _result_row(
                optimization_space=config["optimization_space"],
                iteration=iteration,
                phase="bo",
                x=x_next[0],
                values=values,
                result=result,
                bo_overhead_s=bo_overhead_s,
                cumulative_wall_time_s=cumulative_wall_time_s,
                acq_value=acq_val_scalar,
                diag_file=archive_path,
            )
        )
        # Write the CSV file with all results collected so far
        pd.DataFrame(
            rows
        ).to_csv(
            data_csv,
            index=False,
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "X_normalized": X,
            "Y_objective": Y_objective,
            "iteration": iteration,
            "config": {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in config.items()
                if key != "runner_file"
            },
            "straddle_target_info": target_info,
            "gp_fit_info": gp_fit_info,
        }
        torch.save(
            checkpoint,
            model_folder
            / f"model_iter_{iteration:03d}.pt",
        )
    # Final summary
    ####################################################################################################
    df = pd.DataFrame(rows)
    best_row = df.loc[
        df["conversion_efficiency"].idxmax()
    ]
    # This is only for Straddle acq,
    # to store if maximum was exceeded
    reference_exceeded = False
    if (
        config["acquisition"] == "straddle"
        and config["straddle_reference_mode"] == "fixed"
    ):
        reference_exceeded = (
            best_row["conversion_efficiency"]
            > config[
                "straddle_reference_maximum_ce"
            ]
        )
    summary = {
        "run_name": run_name,
        "optimization_space": config["optimization_space"],
        "acquisition": config["acquisition"],
        "kernel": config["kernel"],
        "gp_fit": config["gp_fit"],
        "best_conversion_efficiency": best_row[
            "conversion_efficiency"
        ],
        "best_Lzzr": best_row["Lzzr"],
        "best_pzr": best_row["pzr"],
        "best_intensity_Wcm2": best_row[
            "intensity_Wcm2"
        ],
        "best_tau_fs": best_row["tau_fs"],
        "maximum_observed_ce": df[
            "conversion_efficiency"
        ].max(),
        "reference_maximum_exceeded": reference_exceeded,
        "n_points": len(df),
        "n_initial": config["n_initial"],
        "n_iterations": config["n_iterations"],
        "total_simulation_time_s": df[
            "simulation_time_s"
        ].sum(),
        "total_bo_overhead_s": df[
            "bo_overhead_s"
        ].sum(),
        "total_wall_time_s": df[
            "iteration_wall_time_s"
        ].sum(),
    }
    summary_txt.write_text(
        "\n".join(
            f"{key} = {value}"
            for key, value in summary.items()
        )
        + "\n"
    )
    torch.save(
        {
            "X_normalized": X,
            "Y_objective": Y_objective,
            "summary": summary,
        },
        model_folder / "final_data.pt",
    )
    print("\nDone.")
    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )
    return summary
