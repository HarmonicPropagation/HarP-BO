# Changelog

Development history of HarP-BO. All conversion-efficiency values and other physical results are
reported in the paper, not here.

## 1.0.0 (2026-11)

The code and data behind the paper "Physics-informed Bayesian optimization of high-harmonic
generation in gases". It corresponds to the tag `public-v1.0.0` of the development repository
HarmonicPropagation/HarP-BO-dev (its `v1.0.0`, with which the paper's runs were made, plus the
changes below).

- HarP: requires HarP v1.0.0 (HarmonicPropagation/HarP, tag v1.0.0), which is public and pinned in
  CI and in `make init-harp`. `harp_input.py` writes the v1.0.0 input convention
  (`rate_unit_SI = 1`, `statistical_factor = 2`) and refuses to run with an older HarP, which would
  silently ignore `statistical_factor` and give a different CE. With the matching input, results are
  bit-identical to those of the HarP version the paper's runs used.
- GP fit (`gp_fit = "warm"`): Matérn 5/2 kernel with fixed observation noise 1e-5; each GP is fitted
  from a cold start and refitted from the previous iteration's hyperparameters, keeping the better
  fit; if the cold fit fails, the refit is used.
- Reproducing the paper: one generator script per figure family (`make repro-2d-ei`,
  `make repro-2d-straddle`, `make repro-4d`, `src/generate_2d_runs_for_kernels.py`), the reference-data
  generators (2D grid scan, 4D Sobol holdout), and a README table mapping each figure to its script
  and data.
- Results in the repository: the seed-0 runs shown in the paper figures (`results/bo_*`), the
  five-seed study with mean and standard deviation of every configuration (`results/seed_study/`),
  and the numerical-convergence tests in 2D and at 4D points (`results/numerical_convergence*/`).
  The supplementary notes on the kernel comparison and on numerical convergence are in
  `docs/supplementary/`.
- Data record: the raw HarP diagnostics (`*.h5`) and the GP checkpoints of the seed study are not in
  git but in a separate data archive; `make data-bundle`, `make data-verify` and `make data-unpack`
  (`scripts/make_data_bundle.py`) build, check and unpack it. Local paths in the committed run data
  are shortened to `~/` or made run-relative.
- Dependencies: pinned with pip-compile; pillow 13.0.0 and filelock 4.2.1 for their security fixes.
  `requirements-mac.txt` is derived from `requirements.txt` (`make mac-lock`) and CI checks that the
  two agree. Dependabot configuration with grouped updates.
- CI: Linux and macOS, Python 3.13 and 3.14, against public HarP v1.0.0 (no access token needed).
- MIT licence (`LICENSE`) and citation metadata (`CITATION.cff`).

## Development history before 1.0.0
### 2026-07: Initial optimization script

- First working Bayesian optimization script interfacing with the HarP HHG
  simulation code: Expected-Improvement acquisition, Matérn 5/2 kernel, search
  over the physics-informed `log(pL)` / `log(p/L)` coordinates.
- HarP (physics solver) and HarP-BO (optimization) kept as separate
  repositories with independent dependency stacks, so the lightweight physics
  code doesn't have to carry the heavier ML dependencies (PyTorch/BoTorch).

### 2026-07: Modularization and 2D optimization

- Refactored the original single-script prototype into separate modules:
  `utils.py` (coordinate transforms, kernels, acquisition functions),
  `harp_input.py` (HarP interface), `bo.py` (optimization loop), and
  `run_bo_2d.py` (2D run configuration).
- Generalized the HarP install path (symlink-based `~/HarP-main` convention)
  so the code isn't tied to one machine's directory layout.
- Added the `Straddle` acquisition function for locating a target
  conversion-efficiency level set, alongside standard Expected Improvement.
- Added nonlinear physical-box constraints to `optimize_acqf` so proposed
  candidates always reconstruct to physically valid parameter values.

### 2026-08: Extension to 4D optimization and validation

- Extended the search space to 4D (gas cell length, pressure, laser intensity,
  and pulse duration) via `run_bo_4d.py`.
- Generated a reproducible 1024-point Sobol hold-out set
  (`generate_holdout_4d.py`, `data/holdout_4d/holdout.csv`) to evaluate
  surrogate model fidelity (NRMSE, global SSIM) without requiring an
  exhaustive 4D grid scan.
- Introduced a `sqrt(CE)` objective transform to enforce strict positivity and
  better handle the conversion efficiency's multi-decade dynamic range.
- Added an early-stopping criterion based on acquisition-utility thresholding,
  avoiding the numerical instability that occurs when the acquisition
  function repeatedly samples near-identical points late in a run.
- Added a "naive" Cartesian-coordinate 4D baseline (`run_bo_4d_naive.py`,
  raw `L, p, I, tau` with no physics-informed reparameterization) as the
  comparison case for the physics-informed methodology.
- Added paper-figure generation scripts for convergence metrics
  (`plot_4Dbo_convergence.py`, `plot_fig1_fig2_paper.py`) and GP surrogate
  cross-sections (`plot_slices_4d.py`, `plot_sobol_slices_4d.py`,
  `slice_plot_utils.py`).

### 2026-08: Reproducibility and testing

- Added a GitHub Actions CI workflow that checks out HarP at a fixed commit,
  builds a Python 3.13/3.14 matrix, and runs the test suite from a clean
  environment.
- Pinned all Python dependencies via `pip-compile` (base and dev
  requirements) so runs are reproducible across machines.
- Added `pytest`/`hypothesis` tests for the physical-to-normalized coordinate
  transforms, including a round-trip property test.
- Added a README covering installation, environment setup, and usage
  instructions for the 2D and 4D optimization runs.

### 2026-08: Calling-convention fix and CI hardening

- Updated `harp_input.py` to match a change in how HarP loads its namelist
  (by file path instead of by module name), so it no longer needs to be
  invoked from inside HarP's own directory.
- Added an end-to-end smoke test (`tests/test_harp_smoke.py`) that runs one
  real, minimal-resolution HarP simulation to catch calling-convention
  regressions between this repo and HarP that the other, HarP-independent
  unit tests can't see.
- Pinned CI to the HarP commit containing that fix.
- Added a `RUN_DIR` environment-variable override for the plotting/polling
  scripts, which previously required manually editing a hardcoded path
  before each run.

### 2026-09: macOS support and 4D optimum confirmation

- Added macOS installation support: `requirements-mac.txt` (dependency set without
  CUDA/`nvidia-*` packages) and a `make init-mac` target, documented in the README
  alongside the existing Linux `make init` path.
- Added `generate_2d_runs_for_fig1_fig2.py` and `generate_2d_runs_for_straddle_fig.py`,
  reproducibility scripts that regenerate the 2D optimization runs behind the
  corresponding paper figures.
- `plot_slices_4d.py`/`slice_plot_utils.py`: the reported 4D global optimum is now, by
  default, confirmed by running one real HarP simulation at the surrogate's predicted
  optimum rather than relying on the surrogate prediction alone; the figure now shows
  both the confirmed value and the surrogate's own prediction with its uncertainty band.
- Minor dead-code cleanup in the paper-figure plotting scripts.
