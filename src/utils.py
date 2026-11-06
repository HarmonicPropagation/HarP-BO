"""
This module provides the shared mathematical tools and utility functions used
throughout the Bayesian optimization workflow. It includes coordinate
transformations between physical and optimization variables, Gaussian pulse
calculations, GP kernel and acquisition function definitions, and helper
functions for managing optimization runs and their configuration.
The utilities are designed to support both 2D and 4D optimizations while
keeping the main optimization scripts concise and independent of the
implementation details.

This is file 5 of 5 in the BO workflow:
    1. run_bo_2d.py   - Defines the setup and starts the 2D Bayesian optimization.
    2. run_bo_4d.py   - Defines the setup and starts the 4D Bayesian optimization.
    3. harp_input.py  - Prepares, runs, and evaluates one HarP simulation.
    4. bo.py          - Bayesian optimization algorithm and acquisition strategy.
    5. utils.py       - Shared mathematical functions and helper utilities.
"""

import math
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from botorch.acquisition import AnalyticAcquisitionFunction, LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import LogNormalPrior

########################################################################################################
# General mathematical helpers
########################################################################################################
def norm(value, value_min, value_max): # Normalization of variables
    return (value - value_min) / (value_max - value_min)
def denorm(x, value_min, value_max):   # Denormalization of variables
    return value_min + x * (value_max - value_min)

def transformed_range(transform, physical_min, physical_max):
    # Transform the endpoints of a physical interval
    # Return the transformed bounds in increasing order
    # Valid for monotonic transformations, including decreasing transformations, e.g., 1/sqrt(tau),
    # for which the physical lower bound maps to the transformed upper bound.
    a = transform(physical_min)
    b = transform(physical_max)
    return min(a, b), max(a, b)

def gaussian_pulse_energy(peak_intensity_w_cm2, waist_m, duration_fwhm_s):
    # Calculate energy of a pulse with Gaussian intensity profiles in radius and time
    # peak_intensity_w_cm2 : on-axis peak intensity in W/cm^2
    # waist_m              : 1/e^2 intensity radius w0 in metres
    # duration_fwhm_s      : intensity FWHM in seconds
    peak_intensity_w_m2 = peak_intensity_w_cm2 * 1.0e4
    # Integral of exp(-2r^2/w0^2) over the transverse plane
    spatial_integral    = np.pi * waist_m**2 / 2.0
    # Integral of exp(-4ln(2)t^2/tau_FWHM^2) over time
    temporal_integral   = (duration_fwhm_s*np.sqrt(np.pi)/(2.0 * np.sqrt(np.log(2.0))))
    return float(peak_intensity_w_m2 * spatial_integral * temporal_integral)

def calculate_nt(tau_fs, lt_factor, dt_si):
    # Calculate number of grid points in the temporal direction from timestep dt_si
    # Keep the temporal spacing approximately fixed when tau changes
    # Rounding upward = temporal spacing is always less coarser (rather than downward)
    return int(math.ceil(lt_factor * tau_fs * 1.0e-15 / dt_si))

def points_per_optical_cycle(wavelength_m, dt_si, speed_of_light_m_s=299792458.0):
    # Diagnostic only - NOT how dt_si is chosen. The runner scripts' DT_SI
    # (currently 1.2890625e-16 s = 165/128 * 1e-15, exactly representable in
    # binary) was reverse-engineered to target Nt = LT_FACTOR*tau/dt_si as a
    # power of two at the reference pulse durations (512 at tau=22fs, 256 at
    # tau=11fs, 2048 at tau=88fs) - an FFT-grid-size choice, not a
    # wavelength-driven one. In the actual floating-point computation this
    # target is missed by one: 3.0*22.0*1e-15 isn't exactly representable in
    # double precision, so the division lands at 512.0000000000001 rather
    # than exactly 512, and calculate_nt()'s ceil() then rounds that up to
    # 513 (257/2049 for the other two) - a real, harmless-but-real quirk,
    # not something introduced here. At the current WAVELENGTH_M=0.81um this
    # dt gives ~21 points per optical carrier cycle, which is why runtime
    # tracks tau (Nt, hence simulation cost, scales with the pulse duration
    # at fixed dt). If WAVELENGTH_M is ever changed for a different laser,
    # call this to check the carrier is still adequately resolved - don't
    # assume it still is, and don't just recompute dt_si from a target
    # points-per-cycle without re-checking the power-of-two Nt property too.
    period_s = wavelength_m / speed_of_light_m_s
    return period_s / dt_si

########################################################################################################
# Coordinate transformations
########################################################################################################
def physical_bounds(config): # Reads boundaries from input
    return {
        "Lzzr": tuple(config["Lzzr_bounds"]),
        "pzr": tuple(config["pzr_bounds"]),
        "intensity_Wcm2": tuple(config["intensity_Wcm2_bounds"]),
        "tau_fs": tuple(config["tau_fs_bounds"]),
    }

def _space_names(optimization_space):
    # "__" in the names of parameter spaces separates variables
    # By using this function, we obtain the list of variables in the optimization space
    return optimization_space.split("__")

def physical_to_x(values, config):
    # Convert physical variables into normalized BO coordinates according to the chosen parameter space.
    # The physical variables remain in the convenient units used in the HarP study:
    # Lzzr: length in the units of zr, pzr in mbar cm, intensity in W/cm^2, and tau in fs.
    names               = _space_names(config["optimization_space"])
    bounds              = physical_bounds(config)
    L                   = values["Lzzr"]
    p                   = values["pzr"]
    I                   = values["intensity_Wcm2"]
    tau                 = values["tau_fs"]
    transformed         = {
        "L": L,
        "p": p,
        "log_L": np.log10(L),
        "log_p": np.log10(p),
        "pL": p*L,
        "p_over_L": p/L,
        "log_pL": np.log10(p*L),
        "log_p_over_L": np.log10(p/L),
        "sqrt_pL": np.sqrt(p*L),
        "sqrt_p_over_L": np.sqrt(p/L),
        "I": I,
        "tau": tau,
        "inv_sqrt_tau": 1.0/np.sqrt(tau),
    }
    transformed_bounds = make_transformed_bounds(config)
    x                  = [norm(transformed[name], *transformed_bounds[name]) for name in names]
    return torch.tensor(x, dtype=torch.double)

def batched_x_to_physical(X, config):
    """
    Converts normalized BO coordinates back to physical L and p, returning
    the denormalized dictionary as well.
    Supports both 1D arrays and batched PyTorch tensors (e.g., from optimize_acqf).
    """
    X = torch.as_tensor(X)
    names = _space_names(config["optimization_space"])
    tb = make_transformed_bounds(config)
    transformed = {}

    # Denormalize
    for index, name in enumerate(names):
        vmin, vmax = tb[name]
        transformed[name] = vmin + X[..., index] * (vmax - vmin)

    # Recover physical L and p
    if "L" in transformed and "p" in transformed:
        L, p = transformed["L"], transformed["p"]
    elif "log_L" in transformed and "log_p" in transformed:
        L, p = 10.0 ** transformed["log_L"], 10.0 ** transformed["log_p"]
    elif "pL" in transformed and "p_over_L" in transformed:
        p = torch.sqrt(transformed["pL"] * transformed["p_over_L"])
        L = torch.sqrt(transformed["pL"] / transformed["p_over_L"])
    elif "log_pL" in transformed and "log_p_over_L" in transformed:
        pL = 10.0 ** transformed["log_pL"]
        p_over_L = 10.0 ** transformed["log_p_over_L"]
        p = torch.sqrt(pL * p_over_L)
        L = torch.sqrt(pL / p_over_L)
    elif "sqrt_pL" in transformed and "sqrt_p_over_L" in transformed:
        pL = transformed["sqrt_pL"] ** 2
        p_over_L = transformed["sqrt_p_over_L"] ** 2
        p = torch.sqrt(pL * p_over_L)
        L = torch.sqrt(pL / p_over_L)
    else:
        raise ValueError("Unsupported optimization space variables for L, p recovery.")

    return L, p, transformed

def x_to_physical(x, config):
    """
    Converts a single normalized coordinate vector from BoTorch into the
    physical quantities expected by harp_input.py.
    """
    L, p, transformed = batched_x_to_physical(x, config)
    bounds = physical_bounds(config)

    # Recover intensity (use fixed bound if not in 2D space)
    if "I" in transformed:
        intensity = transformed["I"]
    else:
        intensity = bounds["intensity_Wcm2"][0]

    # Recover tau (use fixed bound if not in 2D space)
    if "tau" in transformed:
        tau = transformed["tau"]
    elif "inv_sqrt_tau" in transformed:
        tau = 1.0 / (transformed["inv_sqrt_tau"] ** 2)
    else:
        tau = bounds["tau_fs"][0]

    return {
        "Lzzr": float(L),
        "pzr": float(p),
        "intensity_Wcm2": float(intensity),
        "tau_fs": float(tau),
    }

def make_transformed_bounds(config):
    # This function creates the minimum and maximum values for every coordinate system that BO may use
    L_min, L_max                       = config["Lzzr_bounds"]
    p_min, p_max                       = config["pzr_bounds"]
    I_min, I_max                       = config["intensity_Wcm2_bounds"]
    tau_min, tau_max                   = config["tau_fs_bounds"]
    pL_min                             = p_min*L_min
    pL_max                             = p_max*L_max
    p_over_L_min                       = p_min/L_max
    p_over_L_max                       = p_max/L_min
    inv_sqrt_tau_min, inv_sqrt_tau_max = transformed_range(lambda t: 1.0 / np.sqrt(t), tau_min, tau_max)
    return {
        "L": (L_min, L_max),
        "p": (p_min, p_max),
        "log_L": transformed_range(np.log10, L_min, L_max),
        "log_p": transformed_range(np.log10, p_min, p_max),
        "pL": (pL_min, pL_max),
        "p_over_L": (p_over_L_min, p_over_L_max),
        "log_pL": transformed_range(np.log10, pL_min, pL_max),
        "log_p_over_L": transformed_range(np.log10, p_over_L_min, p_over_L_max),
        "sqrt_pL": transformed_range(np.sqrt, pL_min, pL_max),
        "sqrt_p_over_L": transformed_range(np.sqrt, p_over_L_min, p_over_L_max),
        "I": (I_min, I_max),
        "tau": (tau_min, tau_max),
        "inv_sqrt_tau": (inv_sqrt_tau_min, inv_sqrt_tau_max),
    }

def inside_physical_box(values, config, tolerance=1e-5):
    bounds = physical_bounds(config)
    return (
        bounds["Lzzr"][0] - tolerance
        <= values["Lzzr"]
        <= bounds["Lzzr"][1] + tolerance
        and bounds["pzr"][0] - tolerance
        <= values["pzr"]
        <= bounds["pzr"][1] + tolerance
        and bounds["intensity_Wcm2"][0] - tolerance
        <= values["intensity_Wcm2"]
        <= bounds["intensity_Wcm2"][1] + tolerance
        and bounds["tau_fs"][0] - tolerance
        <= values["tau_fs"]
        <= bounds["tau_fs"][1] + tolerance
    )

def get_physical_constraints(config):
    """
    Creates nonlinear constraints for BoTorch to ensure proposed candidates
    reconstruct to physically allowed L and p bounds.
    """
    L_min, L_max = config["Lzzr_bounds"]
    p_min, p_max = config["pzr_bounds"]

    def c1(X): return batched_x_to_physical(X, config)[0] - L_min
    def c2(X): return L_max - batched_x_to_physical(X, config)[0]
    def c3(X): return batched_x_to_physical(X, config)[1] - p_min
    def c4(X): return p_max - batched_x_to_physical(X, config)[1]

    return [(c1, True), (c2, True), (c3, True), (c4, True)]

########################################################################################################
# GP kernel and acquisition functions
########################################################################################################
def make_kernel(name, dim):
    # ARD kernel with a free outputscale (ScaleKernel) and no hyperparameter
    # priors, fitted by marginal likelihood: the kernel the paper describes. It is
    # combined with a fixed, tiny observation noise (DEFAULT_NOISE_MODE below):
    # HarP is deterministic, and with the noise fixed the fit is stable against
    # ulp-level input differences. A lengthscale prior over-smoothed the 4D surrogate.
    if name == "matern12":
        return ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=dim))
    if name == "matern32":
        return ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=dim))
    if name == "matern52":
        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim))
    if name == "rbf":
        return ScaleKernel(RBFKernel(ard_num_dims=dim))
    raise ValueError(f"Unknown kernel: {name}")

# Observation noise used when a run config does not set noise_mode (see make_likelihood).
DEFAULT_NOISE_MODE = "fixed:1e-5"


def make_likelihood(noise_mode=None):
    # Observation-noise model for the GP, in the standardized objective units
    # (SingleTaskGP uses Standardize). HarP is deterministic, so the "noise" is
    # only numerical jitter; a freely learned noise lets the fit explain a
    # narrow ridge away as noise instead of a short lengthscale.
    #   None / "learned":      SingleTaskGP's default (LogNormal(-4, 1) prior, >= 1e-4)
    #   "fixed:<v>":           noise fixed at v (not fitted)
    #   "prior:<median>,<s>":  learned with a tight LogNormal(ln median, s) prior, >= 1e-6
    # Both custom variants keep the default likelihood's structure (untransformed GreaterThan
    # constraint + LogNormal noise prior), so their saved state_dicts load into a
    # default SingleTaskGP (as the plotting scripts build it) with identical noise.
    if noise_mode in (None, "learned"):
        return None
    kind, _, arg = noise_mode.partition(":")
    if kind == "fixed":
        value = float(arg)
        likelihood = GaussianLikelihood(noise_prior=LogNormalPrior(loc=-4.0, scale=1.0),
                                        noise_constraint=GreaterThan(value / 10.0, transform=None))
        likelihood.noise = value
        likelihood.raw_noise.requires_grad_(False)
        return likelihood
    if kind == "prior":
        median, scale = (float(v) for v in arg.split(","))
        likelihood = GaussianLikelihood(noise_prior=LogNormalPrior(loc=math.log(median), scale=scale),
                                        noise_constraint=GreaterThan(1e-6, transform=None))
        likelihood.noise = median
        return likelihood
    raise ValueError(f"Unknown noise_mode: {noise_mode}")


# How the GP hyperparameters are fitted in each BO iteration (run config key gp_fit):
#   "cold": one fit_gpytorch_mll from gpytorch's initial values (lengthscales = outputscale
#           = softplus(0) = 0.693, constant mean 0). Runs whose run_config.txt has no gp_fit
#           line were made this way.
#   "warm": the cold fit, then a second fit from the previous iteration's fitted
#           hyperparameters; the warm fit is kept only if its log marginal likelihood is
#           higher by more than GP_FIT_WARM_TOLERANCE, otherwise the cold fit is kept
#           unchanged (bit for bit).
# The model (kernel, noise, no priors) is the same in both modes. With "cold", the 2D L__p
# seed-0 run (fixed noise 1e-5) fell at iterations 54-61 into a flat optimum with
# lengthscales -> 0 (an iid-noise model, about 49 nats below the warm fit from the
# iteration-53 hyperparameters); see tests/test_gp_fit.py.
DEFAULT_GP_FIT = "warm"
GP_FIT_WARM_TOLERANCE = 1e-3  # nats


def gp_hyperparameters(model):
    # What a warm start copies: lengthscales, outputscale and constant mean (the noise is
    # either fixed by make_likelihood or fitted again from its initial value).
    return {
        "lengthscale": model.covar_module.base_kernel.lengthscale.detach().clone(),
        "outputscale": model.covar_module.outputscale.detach().clone(),
        "constant": model.mean_module.constant.detach().clone(),
    }


def _set_gp_hyperparameters(model, hyperparameters):
    model.covar_module.base_kernel.lengthscale = hyperparameters["lengthscale"]
    model.covar_module.outputscale = hyperparameters["outputscale"]
    model.mean_module.constant = hyperparameters["constant"]


def gp_log_marginal_likelihood(model):
    # Exact log marginal likelihood of the training data, summed over the points, plus the
    # log density of the registered priors (for fixed noise only the constant noise prior).
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    with torch.no_grad():
        value = mll(model(*model.train_inputs), model.train_targets) * model.train_targets.shape[-1]
    model.eval()
    return float(value)


def fit_gp(model, gp_fit, previous=None):
    """Fit the hyperparameters of a freshly built SingleTaskGP in place and return fit info.
    previous: gp_hyperparameters() of the previous iteration's fitted model, or None."""
    if gp_fit not in ("cold", "warm"):
        raise ValueError(f"Unknown gp_fit: {gp_fit}")
    initial_state = {key: value.clone() for key, value in model.state_dict().items()}
    try:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    except Exception as exc:  # e.g. ModelFittingError (seen with Matern 1/2): fall back to the warm fit
        if gp_fit == "cold" or previous is None:
            raise
        model.load_state_dict(initial_state)
        _set_gp_hyperparameters(model, previous)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        return {"gp_fit": gp_fit, "start": "warm (cold failed)", "mll_cold": None,
                "mll_warm": gp_log_marginal_likelihood(model), "cold_error": repr(exc)}
    info = {"gp_fit": gp_fit, "start": "cold", "mll_cold": gp_log_marginal_likelihood(model), "mll_warm": None}
    if gp_fit == "cold" or previous is None:
        return info
    cold_state = {key: value.clone() for key, value in model.state_dict().items()}
    model.load_state_dict(initial_state)
    _set_gp_hyperparameters(model, previous)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    info["mll_warm"] = gp_log_marginal_likelihood(model)
    if info["mll_warm"] > info["mll_cold"] + GP_FIT_WARM_TOLERANCE:
        info["start"] = "warm"
    else:
        model.load_state_dict(cold_state)
    model.eval()
    return info

class Straddle(AnalyticAcquisitionFunction):
    # Acquisition function used to identify the region where
    # the conversion efficiency is above a selected target value.
    # It primarily samples uncertain points near the boundary CE = target,
    # because locating this boundary allows the GP to determine the complete
    # region satisfying CE >= target
    def __init__(self, model, target, beta=1.96):
        super().__init__(model=model)
        self.register_buffer("target", torch.as_tensor(target, dtype=torch.double),)
        self.beta = float(beta)
    # Before calculating Straddle, make sure X has the standard BoTorch candidate shape
    # and verify that every proposed candidate set contains exactly one point
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        posterior = self.model.posterior(X)
        mean      = posterior.mean.squeeze(-1).squeeze(-1)
        sigma     = (posterior.variance.clamp_min(1.0e-12).sqrt().squeeze(-1).squeeze(-1))
        return self.beta * sigma - torch.abs(mean - self.target)

def estimate_gp_maximum(model, bounds, config):
    """
    Estimate the current maximum of the GP posterior mean within physical bounds.
    This is used only for the optional Straddle mode "gp".
    """
    from botorch.acquisition import PosteriorMean
    from botorch.optim import optimize_acqf

    constraints = get_physical_constraints(config)
    ic_gen = get_physical_ic_generator(config, dtype=bounds.dtype)

    candidate, value = optimize_acqf(
        acq_function=PosteriorMean(model),
        bounds=bounds,
        q=1,
        num_restarts=25,
        raw_samples=512,
        nonlinear_inequality_constraints=constraints,
        ic_generator=ic_gen
    )
    return float(value.detach().reshape(-1)[0])

def make_acquisition(model, Y_objective, config, bounds):
    acquisition    = config["acquisition"]
    if acquisition == "log_ei":
        return LogExpectedImprovement(
            model  = model,
            best_f = Y_objective.max(),
        ), None
    if acquisition != "straddle":
        raise ValueError(f"Unknown acquisition function: {acquisition}")
    percentage = config["straddle_percentage"]
    if percentage is None or not 0.0 < percentage <= 100.0:
        raise ValueError("Straddle requires straddle_percentage in (0, 100].")
    mode             = config["straddle_reference_mode"]
    objective_scale  = config["objective_scale"]

    transform_name   = config.get("objective_transform", "id")
    f_fwd, f_inv     = get_objective_transform(transform_name)

    if mode == "fixed":
        reference_ce = config["straddle_reference_maximum_ce"]
        if reference_ce is None or reference_ce <= 0.0:
            raise ValueError(
                "Fixed Straddle mode requires a positive "
                "straddle_reference_maximum_ce."
            )
        reference_objective = reference_ce/objective_scale # Never used, but kept for completeness
    elif mode == "gp":
        # Estimate the current maximum directly from the GP posterior mean
        reference_objective = estimate_gp_maximum(model=model, bounds=bounds, config=config)
        # Inverse-transform to get the physical CE, handling any active objective mapping
        reference_ce = float(f_inv(reference_objective)) * objective_scale

    else:
        raise ValueError("Straddle reference mode must be 'fixed' or 'gp'.")
    target_ce        = percentage/100.0*reference_ce
    target_objective = target_ce/objective_scale

    # Transform the target into the surrogate space
    target_objective = float(f_fwd(target_objective))

    acqf             = Straddle(model=model, target=target_objective, beta=config["straddle_beta"],)
    target_info      = {
        "straddle_reference_mode": mode,
        "straddle_reference_maximum_ce": reference_ce,
        "straddle_percentage": percentage,
        "straddle_target_ce": target_ce,
    }
    return acqf, target_info

########################################################################################################
# Run folders and configuration snapshots
########################################################################################################
def make_run_name(config):  # Generate a folder name for an optimization run
    acq = config["acquisition"]
    if acq == "straddle":
        acq = f"straddle{config['straddle_percentage']:g}per"
        if config["straddle_reference_mode"] == "fixed":
            reference_max = (config["straddle_reference_maximum_ce"] / 1.0e-6)
            acq += f"_max{reference_max:g}"
        elif config["straddle_reference_mode"] == "gp":
            acq += "_gp"

    run_name = (
        f"bo_{config['dimension']}d_"
        f"{config['optimization_space']}_"
        f"{acq}_{config['kernel']}"
    )


    # Append a tag to identify the objective transform
    if config.get("objective_transform", "id") != "id":
        run_name += f"_{config.get('objective_transform')}CE"

    return run_name

def prepare_run_folder(config):
    # Create a clean output folder for this optimization run.
    # If a folder with the same run name already exists,
    # remove it completely before starting a new optimization.
    # This avoids mixing files from different runs.
    run_name      = make_run_name(config)
    output_folder = Path(config["results_root"]) / run_name
    if output_folder.exists():
        print(f"Removing existing run folder:\n  {output_folder}")
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True)
    return run_name, output_folder

def save_run_configuration(config, output_folder):
    # Save both the exact run script and a simple human readable text file
    # Records exactly how an optimization was run
    output_folder.mkdir(parents=True, exist_ok=True)
    runner_file   = Path(config["runner_file"]).resolve()
    shutil_target = output_folder / f"{runner_file.stem}_snapshot.py"
    shutil.copy2(runner_file, shutil_target)
    lines         = []
    for key in sorted(config):
        if key == "runner_file":
            continue
        lines.append(f"{key} = {config[key]}")
    (output_folder / "run_config.txt").write_text(
        "\n".join(lines) + "\n"
    )

def resolve_run_dir(run_name, results_root):
    """Locate a BO run's output folder by name, for the plotting scripts.

    Prefers data/archived_runs/<run_name> (the durable, git-tracked copy the
    paper's figures are actually built from) over results_root/<run_name> (the
    ephemeral, most-recently-run copy - see prepare_run_folder/README). This
    means regenerating a paper figure from already-archived data doesn't
    require rerunning the full BO loop; results_root is still tried as a
    fallback so runs that haven't been archived yet (or were just freshly
    regenerated there) keep working. data/archived_runs is assumed to be a
    sibling of results_root, i.e. results_root.parent/data/archived_runs -
    true for every results_root in this repo (always PROJECT_ROOT/"results").

    Raises FileNotFoundError if the run is in neither location.
    """
    results_root = Path(results_root)
    archived_dir = results_root.parent / "data" / "archived_runs" / run_name
    if archived_dir.exists():
        return archived_dir
    live_dir = results_root / run_name
    if live_dir.exists():
        return live_dir
    raise FileNotFoundError(
        f"Run '{run_name}' not found in {archived_dir} or {live_dir}. "
        "Regenerate it first - see the README's 'Reproducing the paper's figures' section."
    )

########################################################################################################
# Shared helpers for the 2D BO paper-figure scripts (plot_fig1_fig2_paper.py,
# plot_straddle.py). Coordinate transforms/metrics live here per
# plotting_utils.py's own stated boundary; pure plotting/styling stays there.
########################################################################################################

# Fixed physical domain (gas-cell length L/z_R, pressure p*z_R) used by every
# 2D BO paper figure - matches plotting_utils.L_MIN/L_MAX/P_MIN/P_MAX, kept as
# a separate copy here rather than importing plotting_utils (wrong direction:
# this is the lower-level module plotting_utils itself depends on).
LP_L_MIN = 1.5 / 100
LP_L_MAX = 1.5
LP_P_MIN = 300 / 100
LP_P_MAX = 300.0

def clean_space_name(space):
    # Current optimization-space names used by run_bo_2d.py.
    return space

def L_to_plot(L):
    return np.asarray(L)

def p_to_plot(p):
    return np.asarray(p)

def Lp_to_plot_units(L, p):
    return L_to_plot(L), p_to_plot(p) # return L and p as numpy arrays

def best_so_far(df): # cummulative maximum for the best so far point
    return df["objective_scaled_ratio"].cummax().to_numpy()

def get_bo_iteration_range(all_data, plot_iteration):
    # what is the largest number of actual BO points in any of my runs
    bo_counts = []
    for df in all_data.values():
        d    = df[df["iteration"] <= plot_iteration].copy()
        d_bo = d[d["phase"] == "bo"]
        if len(d_bo) > 0:
            bo_counts.append(len(d_bo))
    if len(bo_counts) == 0:
        return 0
    return int(max(bo_counts))

def Lp_to_Xnorm(L, p, optimization_space, *, L_min=LP_L_MIN, L_max=LP_L_MAX, p_min=LP_P_MIN, p_max=LP_P_MAX):
    # Take L,p, transform them into the chosen optimization parameterization
    # Normalize them to the BO coordinate system
    # Return them in the shape expected by the GP
    base_space = clean_space_name(optimization_space) # just gets rid of the filename suffix
    log_L_min, log_L_max     = np.log10(L_min), np.log10(L_max)
    log_p_min, log_p_max     = np.log10(p_min), np.log10(p_max)
    pL_min                   = p_min * L_min
    pL_max                   = p_max * L_max
    p_over_L_min             = p_min / L_max
    p_over_L_max             = p_max / L_min
    log_pL_min, log_pL_max   = np.log10(pL_min), np.log10(pL_max)
    log_p_over_L_min         = np.log10(p_over_L_min)
    log_p_over_L_max         = np.log10(p_over_L_max)
    sqrt_pL_min, sqrt_pL_max = np.sqrt(pL_min), np.sqrt(pL_max)
    sqrt_p_over_L_min        = np.sqrt(p_over_L_min)
    sqrt_p_over_L_max        = np.sqrt(p_over_L_max)
    if base_space == "L__p":
        x0 = (L - L_min) / (L_max - L_min)
        x1 = (p - p_min) / (p_max - p_min)
    elif base_space == "log_L__log_p":
        x0 = (np.log10(L) - log_L_min) / (log_L_max - log_L_min)
        x1 = (np.log10(p) - log_p_min) / (log_p_max - log_p_min)
    elif base_space == "p_over_L__pL":
        pL = p * L
        p_over_L = p / L
        x0 = (p_over_L - p_over_L_min) / (p_over_L_max - p_over_L_min)
        x1 = (pL - pL_min) / (pL_max - pL_min)
    elif base_space == "log_p_over_L__log_pL":
        pL = p * L
        p_over_L = p / L
        x0 = (np.log10(p_over_L) - log_p_over_L_min) / (log_p_over_L_max - log_p_over_L_min)
        x1 = (np.log10(pL) - log_pL_min) / (log_pL_max - log_pL_min)
    elif base_space == "sqrt_p_over_L__sqrt_pL":
        pL = p * L
        p_over_L = p / L
        x0 = (np.sqrt(p_over_L) - sqrt_p_over_L_min) / (sqrt_p_over_L_max - sqrt_p_over_L_min)
        x1 = (np.sqrt(pL) - sqrt_pL_min) / (sqrt_pL_max - sqrt_pL_min)
    else:
        raise ValueError(f"Unknown optimization space: {optimization_space}")
    return np.stack([x0.reshape(-1), x1.reshape(-1)], axis=1) #reshape to the expected shape

########################################################################################################
# Define random seeds through all the libraries
########################################################################################################
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

########################################################################################################
# Convergence Metrics
########################################################################################################

def normalized_rmse(Z_ref, Z_pred, ref_range=None):
    # normalized root mean squared error (NRMSE) between the direct grid Z_ref and GP prediction Z_pred
    valid = np.isfinite(Z_ref) & np.isfinite(Z_pred)
    if np.sum(valid) < 2:
        return np.nan
    rmse = np.sqrt(np.mean((Z_pred[valid] - Z_ref[valid]) ** 2))

    # Use the fixed reference range if provided, otherwise fall back to reference data range
    scale = ref_range if ref_range is not None else (np.nanmax(Z_ref[valid]) - np.nanmin(Z_ref[valid]))

    if scale == 0:
        return np.nan
    return rmse / scale


def global_ssim(Z_ref, Z_pred):
    # single global SSIM-like similarity score between the direct grid Z_ref and GP prediction Z_pred
    valid = np.isfinite(Z_ref) & np.isfinite(Z_pred)
    if np.sum(valid) < 2:
        return np.nan
    x = Z_ref[valid].reshape(-1)
    y = Z_pred[valid].reshape(-1)
    data_range = np.nanmax(x) - np.nanmin(x)
    if data_range == 0:
        return np.nan
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * cov_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
    if denominator == 0:
        return np.nan
    return numerator / denominator

def get_objective_transform(name):
    """
    Returns a tuple of (forward_transform, inverse_transform).
    Functions should be capable of handling both NumPy arrays and PyTorch tensors.
    """
    registry = {
        "id": (
            lambda x: x,
            lambda x: x
        ),
        "log10": (
            lambda x: torch.log10(x) if isinstance(x, torch.Tensor) else np.log10(x),
            lambda x: 10.0 ** x
        ),
        "sqrt": (
            lambda x: torch.sqrt(x) if isinstance(x, torch.Tensor) else np.sqrt(x),
            lambda x: torch.square(x) if isinstance(x, torch.Tensor) else np.square(x)
        )
    }

    if name not in registry:
        raise ValueError(f"Unknown objective transform: {name}")

    return registry[name]

def get_physical_ic_generator(config, dtype=torch.float64):
    """
    Returns a custom initial condition generator for BoTorch's optimize_acqf
    that dynamically maps uniformly sampled physical spaces into the active BO space.
    """
    def generate_physical_ics(bounds, q, num_samples=512, **kwargs):
        tb = make_transformed_bounds(config)

        # 1. Sample uniformly in physical space
        L = torch.rand(num_samples, q, dtype=dtype) * (config["Lzzr_bounds"][1] - config["Lzzr_bounds"][0]) + config["Lzzr_bounds"][0]
        p = torch.rand(num_samples, q, dtype=dtype) * (config["pzr_bounds"][1] - config["pzr_bounds"][0]) + config["pzr_bounds"][0]
        I = torch.rand(num_samples, q, dtype=dtype) * (config["intensity_Wcm2_bounds"][1] - config["intensity_Wcm2_bounds"][0]) + config["intensity_Wcm2_bounds"][0]
        tau = torch.rand(num_samples, q, dtype=dtype) * (config["tau_fs_bounds"][1] - config["tau_fs_bounds"][0]) + config["tau_fs_bounds"][0]

        # 2. Compute all possible transformations natively in PyTorch
        transformed = {
            "L": L,
            "p": p,
            "log_L": torch.log10(L),
            "log_p": torch.log10(p),
            "pL": p * L,
            "p_over_L": p / L,
            "log_pL": torch.log10(p * L),
            "log_p_over_L": torch.log10(p / L),
            "sqrt_pL": torch.sqrt(p * L),
            "sqrt_p_over_L": torch.sqrt(p / L),
            "I": I,
            "tau": tau,
            "inv_sqrt_tau": 1.0 / torch.sqrt(tau),
        }

        # 3. Stack only the variables requested by the optimization space
        names = _space_names(config["optimization_space"])
        x_list = []
        for name in names:
            val = transformed[name]
            vmin, vmax = tb[name]
            x_norm = (val - vmin) / (vmax - vmin)
            x_list.append(x_norm)

        return torch.stack(x_list, dim=-1)

    return generate_physical_ics
def find_surrogate_maximum(model, config, dtype=torch.double):
    """Finds the continuous global optimum of the surrogate model (GP posterior mean),
    via constrained acquisition optimization. Also returns the GP's own posterior std
    at that point (in the same fitted/transformed objective space as max_val_transformed),
    so callers can turn it into a physical-CE uncertainty band via the objective
    transform's inverse, the same way max_val_transformed itself is turned into a CE."""
    from botorch.acquisition import PosteriorMean
    from botorch.optim import optimize_acqf

    dim = config.get("dimension", 4)
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=dtype)
    constraints = get_physical_constraints(config)
    ic_gen = get_physical_ic_generator(config, dtype=dtype)

    candidate, max_val_transformed = optimize_acqf(
        acq_function=PosteriorMean(model),
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        nonlinear_inequality_constraints=constraints,
        ic_generator=ic_gen
    )
    best_phys = x_to_physical(candidate.detach().cpu().squeeze(0), config)
    with torch.no_grad():
        std_transformed = model.posterior(candidate).variance.sqrt().item()
    return best_phys, max_val_transformed, std_transformed


def confirm_surrogate_optimum(best_phys, harp_config):
    """Runs one real HarP simulation at a surrogate-predicted optimum (as returned by
    find_surrogate_maximum) to check it against the surrogate's own prediction.
    harp_config must carry the physical/numerical simulation settings expected by
    harp_input.run_harp (nr, nz, lr_factor, lt_factor, dt_si, temperature_K,
    wavelength_m, zR_mm, harmonic_order, objective_scale, harp_workdir) - i.e. the
    same fixed constants used by every run_bo_*.py / generate_*.py script.
    Returns the confirmatory conversion_efficiency (physical units)."""
    from harp_input import check_harp_installation, run_harp

    paths = check_harp_installation(harp_config["harp_workdir"])
    values = {
        "Lzzr": float(best_phys["Lzzr"]),
        "pzr": float(best_phys["pzr"]),
        "intensity_Wcm2": float(best_phys["intensity_Wcm2"]),
        "tau_fs": float(best_phys["tau_fs"]),
    }
    result = run_harp(values=values, config=harp_config, paths=paths)
    return result["conversion_efficiency"]
