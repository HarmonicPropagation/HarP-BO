# Numerical convergence at 4D conditions

These runs use the tests of `src/run_numerical_tests.py` and `results/numerical_convergence/`:
- `final`: the production grid;
- `res`: higher resolution;
- `box`: a larger radial and temporal box.

Here they are run at points of the 4D optimization, where the intensity and pulse duration vary. The
supplementary note `docs/supplementary/numerical_convergence.pdf` discusses the results.

| Point | L/z_R | p·z_R [mbar cm] | I [W/cm²] | τ [fs] |
|---|---|---|---|---|
| a: 4D GP optimum | 0.0526 | 300 | 3.71e14 | 11 |
| b: best sampled point, seed-0 4D run (iteration 24) | 0.0864 | 265.0 | 4.01e14 | 12.48 |
| c: short pulse, high intensity | 0.05 | 300 | 1e15 | 11 |
| d: long pulse at a's L, p, I | 0.0526 | 300 | 3.71e14 | 88 |

"res" changes three things at once: Nz ×8, dt ÷4 and Nr ×2. The tests `res_nz`, `res_dt` and `res_nr`
each apply one of them, which shows which numerical parameter dominates. They were also run at the
32 Sobol points of the 2D tests (ids `s00`–`s31`; I 2.5e14 W/cm², τ 22 fs), for comparison with 2D.

Results, as CE in units of 1e-6 and the signed difference to `final`:

| Point | final | res | box | Nz ×8 only | dt ÷4 only | Nr ×2 only |
|---|---|---|---|---|---|---|
| a | 6.909 | 6.999 (+1.3%) | 6.910 (+0.01%) | +0.7% | +1.6% | −1.2% |
| b | 6.456 | 6.421 (−0.5%) | 6.448 (−0.1%) | | | |
| c | 2.574 | 2.451 (−5.0%) | 2.574 (+0.03%) | +1.5% | +1.8% | −6.4% |
| d | 1.437 | 1.541 (+6.7%) | 1.437 (0.00%) | +2.3% | −0.1% | +5.0% |

In 2D, all 32 points meet |ΔCE| < 5e-8 in every test. At the best 2D point (Sobol point 13) the
single changes give dt +0.62% and Nr −0.64%, the same pattern as at the 4D optimum but about 2.5×
smaller.

## Files

- `conv4d.py`: runs the tests (`worker` mode, one private HarP copy per worker) and builds
  `numerical_convergence_table.csv` (`table` mode).
- `table2d.py`: builds `numerical_convergence_table_2d.csv` for the 32 2D points. It takes `res` and
  `box` from `../numerical_convergence/simulations.csv`.
- `simulations.csv`: all runs, one row each. `settings.json` holds the settings.
- `diagnostics/`: the HarP energy diagnostic of each run (in the data record, not in git).

The runs were made in 2026-10 with HarP-dev 0ad9c77, on Linux, one thread per run. HarP-dev 0ad9c77 has
the same physics as HarP v1.0.0, with the matching input: at point a, `final` gives
6.908684031930973e-06 with HarP v1.0.0 and the current `src/harp_input.py`.
