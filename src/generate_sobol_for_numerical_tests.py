from pathlib import Path
import numpy as np
from scipy.stats import qmc

# User settings
N_POINTS     = 32
SEED         = 20260711
L_MIN, L_MAX = 0.015, 1.5
P_MIN, P_MAX = 3.0, 300.0
OUTPUT_FILE  = Path("sobol_32_points.txt")


# Generate Sobol points
m            = int(np.log2(N_POINTS))
sampler      = qmc.Sobol(d=2, scramble=True, seed=SEED)
points_unit  = sampler.random_base2(m=m)
points       = qmc.scale(points_unit,[L_MIN, P_MIN],[L_MAX, P_MAX])
Lzzr         = points[:, 0]
pzr          = points[:, 1]

# Save
data         = np.column_stack((np.arange(N_POINTS),Lzzr,pzr))

np.savetxt(
    OUTPUT_FILE,
    data,
    fmt=["%d", "%.16e", "%.16e"],
    header="point_index Lzzr pzr",
    comments="# ",
)

print(f"Generated {N_POINTS} Sobol points.")
print(f"Saved to {OUTPUT_FILE}")
print("\nGenerated points:")
for i, (L, p) in enumerate(zip(Lzzr, pzr)):
    print(f"{i:2d}: L/z_R = {L:.6f},   p z_R = {p:.6f}")
