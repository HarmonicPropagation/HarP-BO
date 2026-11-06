# HarP-BO: Bayesian optimization for HHG

This repository contains a Bayesian Optimization (BO) framework for optimizing High-Order Harmonic Generation (HHG). It is designed to interface directly with the [HarP simulation code](https://github.com/HarmonicPropagation/HarP).

## Overview

HarP-BO leverages `BoTorch` and `GPyTorch` to efficiently explore the macroscopic parameter space of HHG. It maps physical parameters (such as medium length, gas pressure, driving intensity, and pulse duration) into a physics-informed, normalized optimization space. This guides the HarP physical simulator toward optimal conditions for macroscopic harmonic yield and phase matching while minimizing computationally expensive evaluations.

## Prerequisites

*   Python 3.13+
*   [PyTorch](https://pytorch.org/)
*   [BoTorch](https://botorch.org/) & [GPyTorch](https://gpytorch.ai/)
*   [HarP](https://github.com/HarmonicPropagation/HarP) v1.0.0 (the core physics simulation engine)

## Installation

The project uses a standard `Makefile` for environment setup and dependency management via `pip-tools`.

```bash
git clone https://github.com/HarmonicPropagation/HarP-BO.git
cd HarP-BO
make init
source .venv/bin/activate
```

`requirements.txt` is a `pip-compile`'d lock file generated on Linux, so it pins a number of
CUDA/`nvidia-*` packages that aren't available on macOS. On macOS, use `make init-mac` instead of
`make init` — it installs from `requirements-mac.txt`, the same dependency set without those
packages. `requirements-mac.txt` is derived from `requirements.txt` by `make mac-lock`; run it after
changing `requirements.txt` (`make compile` does it for you). CI checks that the two files agree.

To install development and testing dependencies (including `pytest` and `hypothesis`), run:
```bash
make dev
```

## Testing

The unit tests cover the coordinate transformations (including a property-based round trip), the
initial-point generator, the reproducible candidate pools, and the GP hyperparameter fit
(`tests/test_gp_fit.py`, on stored training data). Three end-to-end tests need a real HarP installation at
`~/HarP-main` and are skipped without one: a smoke test of the HarP-BO/HarP calling convention with one
tiny simulation (`tests/test_harp_smoke.py`), and short runs of the paper's 2D LogEI and Straddle
configurations on a tiny HarP grid (`tests/test_logei_2d_repro.py`, `tests/test_straddle_2d_repro.py`).
With HarP, `make test` takes about 10 minutes.

```bash
make test
```

## Usage

The optimization framework is modular and relies on file-system side-effects to bridge the Bayesian optimizer and the physical HarP simulation.

Because executing the physical simulations requires significant computational resources, the usage runs are deliberately kept separate from the standard `make test` suite.

### 1. Linking the simulator
The runner scripts expect the HarP simulation directory to be available at `~/HarP-main`. The easiest way to configure this without modifying the Python scripts is to create a symbolic link to your local HarP clone:

```bash
ln -s /path/to/your/HarP ~/HarP-main
```

If you don't already have a local HarP clone, `make init-harp` will fetch one for you — it clones
[HarP](https://github.com/HarmonicPropagation/HarP) straight to `~/HarP-main`, pinned to the same commit CI
tests against (release v1.0.0). It refuses to run if `~/HarP-main` already exists in any form (directory or
symlink), so it never overwrites an existing checkout.

### 2. 2D optimization (fixed laser parameters)
To optimize the medium length $L/z_R$ and the pressure $p z_R$ at fixed laser intensity and pulse
duration, run the 2D runner:

```bash
make run-2d
```
This runs `src/run_bo_2d.py`. The settings block at the top of the script selects the acquisition function,
the optimization space, the kernel and the number of iterations. The defaults are the Straddle acquisition
with a fixed CE target of $3.0 \times 10^{-6}$ (`STRADDLE_PERCENTAGE = 100` of
`STRADDLE_REFERENCE_MAXIMUM_CE = 3e-6`), the $(\log(p/L), \log(pL))$ space, the Matérn 5/2 kernel and
101 BO iterations after one initial point, at $I = 2.5 \times 10^{14}$ W/cm² and $\tau = 22$ fs. The run is
written to `results/bo_2d_log_p_over_L__log_pL_straddle100per_max3_matern52/`.

Cost: one simulation takes about 40 s on a laptop, so the default run takes about 1–1.5 hours. The paper's
2D runs and their settings are in section 4.

### 3. 4D Optimization (Global Search)
To optimize the medium length and pressure together with the laser intensity $I$ and pulse duration
$\tau$, run the 4D runner:

```bash
make run-4d
```

This runs `src/run_bo_4d.py`, which has the same kind of settings block. The defaults are LogEI, the
$(\log(p/L), \log(pL), I, 1/\sqrt{\tau})$ space with a square-root transformation of CE, the Matérn 5/2
kernel and 256 BO iterations after one initial point. There is no early stopping by default; setting
`ACQ_STOPPING_THRESHOLD` to an acquisition-utility threshold stops the run once the utility has fallen
below it. The run is written to `results/bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_log_ei_matern52_sqrtCE/`.

Cost: a simulation takes about 25–100 s depending on the laser parameters, so the default run takes
several hours on a laptop. The paper's 4D runs (Straddle, 120 iterations) and their settings are in
section 4.

### 4. Reproducing the paper

The paper's figures are made from the runs in `results/` (seed 0, one folder per configuration) and the
reference data in `data/`. The plotting scripts read only these folders and write to `figures/`, so
every figure can be regenerated without running HarP. Run them from the repository root with the virtual
environment active:

| Paper figure | Script | Data | Output |
|---|---|---|---|
| Fig. 1: BO scheme | (drawn in the manuscript) | – | – |
| Fig. 2: GP maps, five 2D parameterizations | `python src/plot_fig1_fig2_paper.py` | `results/bo_2d_*_log_ei_matern52/`, `data/scan_results.csv` | `figures/figure1.{pdf,png}` |
| Fig. 3: 2D convergence, NRMSE, SSIM | (same script) | (same) | `figures/figure2.{pdf,png}` |
| Fig. 4: Straddle GP maps, three CE targets | `python src/plot_straddle.py` | `results/bo_2d_log_p_over_L__log_pL_straddle{25,50,90}per_max3_matern52/`, `data/scan_results.csv` | `figures/STRADDLEfigure1.{pdf,png}` |
| Fig. 5: Straddle convergence, NRMSE, SSIM | (same script) | (same) | `figures/STRADDLEfigure2.{pdf,png}` |
| Fig. 6: 4D GP slices | `python src/plot_slices_4d.py` | `results/bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_straddle100per_max3_matern52_sqrtCE/` | `figures/slices_4d_iter_119.{pdf,png}` |
| Fig. 7: 4D convergence, NRMSE | `python src/plot_4Dbo_convergence.py` | `results/bo_4d_*_straddle100per_max3_matern52[_sqrtCE]/`, `data/holdout_4d/holdout.csv` | `figures/figure_4Dbo_convergence.{pdf,png}` |

The supplementary material the paper refers to:

| Material | Document | Data and scripts |
|---|---|---|
| Kernel comparison | [`docs/supplementary/kernels.pdf`](docs/supplementary/kernels.pdf) ([source](docs/supplementary/kernels.tex)) | `results/bo_2d_log_p_over_L__log_pL_log_ei_{matern12,matern32,matern52,rbf}/`; figure: `python src/plot_kernels.py` → `figures/KERNELfigure2.{pdf,png}`; runs: `python src/generate_2d_runs_for_kernels.py` |
| Numerical convergence | [`docs/supplementary/numerical_convergence.pdf`](docs/supplementary/numerical_convergence.pdf) ([source](docs/supplementary/numerical_convergence.tex)) | 2D: `results/numerical_convergence/` (`python src/run_numerical_tests.py`); 4D: [`results/numerical_convergence_4d/`](results/numerical_convergence_4d/README.md) |
| Mean and standard deviation over five seeds | [`results/seed_study/README.md`](results/seed_study/README.md) | `results/seed_study/` |

The git repository holds everything the figures and tables need: each run's `data.csv`, summaries
and GP checkpoints, the reference scans and the seed-study curves. The raw files are in a separate
data record (DOI to be added): the HarP energy diagnostic (`*.h5`) of every simulation, and the GP
checkpoints of the seed study. To put them in place in a clone, download the archive and run

```bash
make data-unpack DATA_BUNDLE=HarP-BO-data-<version>.tar.gz
```

This checks every file against the archive's checksums and never overwrites a file that differs.
`make data-verify DATA_BUNDLE=...` only checks, and `make data-bundle DATA_VERSION=...` builds the
archive from `scripts/data_bundle_manifest.txt` (see `scripts/make_data_bundle.py`).

To rerun the BO itself, the generator scripts run each figure's configurations with the paper's exact
settings:

```bash
make repro-2d-ei         # the five 2D parameterizations (Figs. 2-3), 60 iterations each
make repro-2d-straddle   # the three Straddle CE targets (Figs. 4-5), 60 iterations each
make repro-4d            # the four 4D configurations (Figs. 6-7), 120 iterations each
python src/generate_2d_runs_for_kernels.py   # the kernel comparison
```

Each script's `--help` lists options for running a subset or resuming (`--skip-existing`). On a laptop, a
2D run (60 iterations) takes about 45 minutes and a 4D run (120 iterations) about 1.5–2 hours. The reference data come from
`src/generate_2d_grid_scan.py` (the 100 × 100 grid scan) and `src/generate_holdout_4d.py` (the 1024-point
4D Sobol holdout).

⚠️ **Rerunning overwrites the published runs.** `bo.prepare_run_folder()` deletes and recreates a run
folder in `results/` with the same name, without asking. Restore the published runs with
`git checkout -- results/`, or copy them elsewhere first.

On a given machine, a run is reproducible with a fixed seed. Across machines it can differ slightly:
round-off differs between platforms (BLAS, PyTorch builds), and where the acquisition function has two
almost equal maxima, a round-off difference can change the chosen point. The runs then follow
different trajectories. One example is the seed-0 Straddle runs, which choose different points at
iteration 2 on macOS and on Linux. The seed study (`results/seed_study/`) shows the spread over five
seeds.

### 5. Monitoring and plotting
As the optimizer evaluates new physical parameters, it saves the Gaussian Process surrogate models and evaluation metrics to the `results/` directory.

Before running the plotting or polling scripts, either open the respective python file (e.g., `src/plot_slices_4d.py` or `src/poll_convergence_4d.py`) and update the `RUN_DIR` variable to match the exact folder name generated inside your `results/` directory, or override it without editing the file via the `RUN_DIR` environment variable, e.g.:
```bash
RUN_DIR=results/bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_log_ei_matern52_sqrtCE python src/poll_convergence_4d.py
```

Once configured, you can monitor convergence or visualize the surrogate slices post-run using the provided scripts (run these directly via Python while your virtual environment is active):
```bash
# Poll active 4D convergence metrics
python src/poll_convergence_4d.py

# Generate 2D cross-section plots of the BO surrogate
python src/plot_slices_4d.py

# Generate paper-ready comparative convergence figures
python src/plot_fig1_fig2_paper.py
```


## Citation

If you use this code in your research, please cite the paper (citation details will be added once it is
available) and the software, see [`CITATION.cff`](CITATION.cff).

## License

MIT, see [`LICENSE`](LICENSE).
