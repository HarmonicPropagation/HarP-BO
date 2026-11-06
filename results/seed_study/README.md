# Seed study: five seeds per configuration

The paper reports the mean and standard deviation over five seeds (0–4) for every BO configuration it
shows. This folder holds those runs and the evaluation behind the numbers.

All runs used HarP-BO v1.0.0 with HarP 0ad9c77. HarP 0ad9c77 has the same physics as the public HarP
v1.0.0, which this repository now pins: with the matching input, the results are bit-identical. Settings:
Matérn 5/2 kernel, fixed observation noise 1e-5, `gp_fit = warm`, one start point, one OpenMP thread,
and full-length runs (no early stopping). The machine and Python environment are in `env/`.

| Family | Runs | Iterations |
|---|---|---|
| 2D LogEI | `bo_2d_*_log_ei_matern52`: (L, p), (log L, log p), (p/L, pL), (√(p/L), √(pL)), (log(p/L), log(pL)) | 60 |
| 2D Straddle | `bo_2d_log_p_over_L__log_pL_straddle{25,50,90}per_max3_matern52` (target as % of 3e-6) | 60 |
| 4D Straddle | `bo_4d_*_straddle100per_max3_matern52[_sqrtCE]`, the four configurations of the 4D convergence figure | 120 |

A seed sets the start point and the candidate pools. Everything else is the same across seeds.

## Files

- `runs/seed<k>/<run>/`: one run, as written by the run scripts. Each folder has:
  - `data.csv`, with every evaluated point;
  - `run_config.txt` and `summary.txt`;
  - the run script as it was used (`*_snapshot.py`).

  Local paths in `run_config.txt` are shortened to `~/`. `diag_file` in `data.csv` is relative to the
  run folder.
- `seed_study_curves.py`: evaluates every run and seed against the same reference data as the paper
  figures. Those are the 2D grid scan `data/scan_results.csv` and the 4D Sobol holdout
  `data/holdout_4d/holdout.csv`. It writes:
  - `seed_study_curves.csv`: best observed CE, NRMSE and SSIM for each iteration, run and seed;
  - `seed_study_per_seed.csv`: the same, at the iterations the paper reports (2D: 60; 4D: 30, 60, 120);
  - `seed_study_summary.csv`: mean, standard deviation (ddof = 1), minimum and maximum over the five seeds;
  - `seed_study_{2d,4d}_convergence.{pdf,png}`: mean ± std curves.

  Run it from the repository root with `PYTHONPATH=src python results/seed_study/seed_study_curves.py`.
  It needs the GP checkpoints (`models/`) of each run, which are in the data record (see below).

## Raw data

Each run also has a `models/` folder with the GP after every iteration, and a `diagnostics/` folder with
the HarP energy diagnostic of every evaluation. Together they are about 100 MB, so they are not in
the git repository. They are in the data record; `make data-unpack DATA_BUNDLE=...` puts them in
place (see the main README).

## Platform differences

The paper figures come from separate seed-0 runs (`results/`), made on another machine with the same
code and settings. Runs on different machines can differ slightly, because round-off differs between
platforms (BLAS, PyTorch builds). Usually the differences stay at round-off level. Where the
acquisition function has two almost equal maxima, though, a round-off difference can change the point
that is chosen, and from then on the runs follow different trajectories.

One example is the 2D Straddle runs with seed 0. They agree bit for bit until iteration 1 and choose
different points at iteration 2, where the two best acquisition values differ by about 1e-7. Both runs
are valid realisations. The seed study shows the spread such differences cause.
