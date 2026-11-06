"""
This file connects the Bayesian optimizer with the HarP simulation. For every
proposed point, it generates a new BOinput.py file, runs HarP, reads the final
harmonic energy, and calculates the conversion efficiency used as the objective
function. It also checks the HarP installation and can archive the diagnostic
output from individual simulations.

This is file 3 of 5 in the BO workflow:
    1. run_bo_2d.py   - Starts the 2D Bayesian optimization loop.
    2. run_bo_4d.py   - Starts the 4D Bayesian optimization loop.
    3. harp_input.py  - Runs one HarP simulation and evaluates the objective.
    4. bo.py          - Implements the Bayesian optimization algorithm.
    5. utils.py       - Shared numerical functions and helper utilities.
"""

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
import h5py
import numpy as np
from scipy import constants
from utils import calculate_nt, gaussian_pulse_energy

# HarP paths and checks if they exist
########################################################################################################
def check_harp_installation(harp_workdir):
    harp_workdir = Path(harp_workdir).expanduser().resolve()
    harp_script  = harp_workdir / "HarP.py"
    tables_path  = harp_workdir / "tables"
    if not harp_workdir.is_dir():
        raise FileNotFoundError(f"HarP workdir does not exist: {harp_workdir}")
    if not harp_script.is_file():
        raise FileNotFoundError(f"HarP.py was not found: {harp_script}")
    if not tables_path.is_dir():
        raise FileNotFoundError(f"HarP tables directory was not found: {tables_path}")
    check_harp_convention(harp_workdir)
    # NOTE: the generated namelist's tables_path is still the relative string "tables/"
    # (see write_boinput below), so HarP still needs to be run with cwd=harp_workdir
    # for the medium/ionisation/harmonic table lookups to resolve. Making tables_path
    # absolute would need a coordinated change here and on the HarP side; not done yet.
    return {"workdir": harp_workdir, "script": harp_script, "input": harp_workdir / "BOinput.py", "diag": harp_workdir / "DiagEnergy.h5"}

# The generated input uses the convention of HarP v1.0.0, where the SAE spin factor of 2 is given as
# statistical_factor = 2 on both the ionisation rate and the dipole (rate_unit_SI = 1). Older HarP
# versions (e.g. HarP-dev 0ad9c77) do not know statistical_factor and silently ignore it, which would
# halve the ionisation rate and change the CE without any error. Refuse to run in that case.
def check_harp_convention(harp_workdir):
    missing = []
    for module in ("ionisation.py", "harmonic.py"):
        source = Path(harp_workdir) / "src" / module
        text = source.read_text() if source.is_file() else ""
        if not re.search(r"^\s+statistical_factor\s*:", text, re.M):
            missing.append(str(source))
    if missing:
        raise RuntimeError(
            "This HarP installation does not support statistical_factor (missing in "
            + ", ".join(missing) + "). HarP-BO needs HarP v1.0.0 or later "
            "(https://github.com/HarmonicPropagation/HarP, tag v1.0.0); with an older HarP the "
            "ionisation rate would be wrong by a factor of 2 without any error message."
        )

# Write the HarP input file
########################################################################################################
def write_boinput(values, config, paths):
    Lzzr            = values["Lzzr"]
    pzr             = values["pzr"]
    intensity_w_cm2 = values["intensity_Wcm2"]
    tau_fs          = values["tau_fs"]
    nt              = calculate_nt(tau_fs=tau_fs, lt_factor=config["lt_factor"], dt_si=config["dt_si"])
    text = f"""
from components import Main
from pulse import Pulse
from medium import Medium
from ionisation import Ionisation
from diagnostic import Diagnostics, DiagEnergy
from harmonic import Harmonic
import numpy as np
import os
from scipy import constants

for _component in (Ionisation, Harmonic):  # see check_harp_convention in harp_input.py
    if not hasattr(_component, "statistical_factor"):
        raise RuntimeError("HarP-BO needs HarP v1.0.0 or later (statistical_factor is not supported by this HarP)")

### BO variables
Lzzr        = {Lzzr:.16e}
pzr         = {pzr:.16e}
I_input     = {intensity_w_cm2:.16e}
tau_input   = {tau_fs:.16e}

### Physical constants
Temperature = {config["temperature_K"]:.16e}
lambda0     = {config["wavelength_m"]:.16e}
c           = constants.c
omega0      = 2*np.pi*c/lambda0
eps0        = constants.epsilon_0
e           = constants.e
me          = constants.m_e
kb          = constants.Boltzmann

### Unit conversions
T_r         = 1.0/omega0
L_r         = lambda0/2.0/np.pi
n_r         = eps0*omega0**2*me/e**2
E_r         = me*omega0*c/e
I_r         = eps0*c*E_r**2/2.0*1e-4
Pol_r       = me*omega0**2/e**2
mm          = 1e-3/L_r # conversion factor from mm to HarP normalized units
fs          = 1e-15/T_r

### Laser parameters
z_R         = {config["zR_mm"]:.16e}*mm
w0          = np.sqrt(z_R*(lambda0/L_r)/np.pi)
tfwhm       = tau_input*fs
I0          = I_input/I_r
E0          = np.sqrt(I0)

### Gas parameters
z_R_cm      = z_R/mm*1e-1
p_in_mbar   = pzr/z_R_cm
p           = p_in_mbar*1e2 # mbar to Pa
n0          = p/kb/Temperature/n_r
Lgaz        = Lzzr*z_R

def gazprofile(r, z):
    return n0*np.heaviside(z-0.0, 1.0)*np.heaviside(Lgaz-z, 1.0)

### Simulation parameters
nr          = {int(config["nr"])}
nz          = {int(config["nz"])}
nt          = {nt}
Lr          = {config["lr_factor"]:.16e}*w0
Lz          = Lgaz
Tsim        = {config["lt_factor"]:.16e}*tfwhm
dr          = Lr/nr
dz          = Lz/nz
dt          = Tsim/nt

### Laser profile
tfocus      = Tsim/2.0
zfocus      = Lz/2.0

def tenv(t):
    return np.exp(-2*np.log(2)*(t-tfocus)**2/tfwhm**2)

def eprofile(r):
    wz       = w0*np.sqrt(1.0+(zfocus/z_R)**2)
    R        = zfocus*(1.0+(z_R/zfocus)**2)
    Phi_gouy = np.arctan2(zfocus, z_R)
    return (w0/wz * np.exp(-(r/wz)**2) * np.exp(-1j*(r**2/2.0/R-Phi_gouy)))

### Paths
tables_path  = "tables/"  # relative to cwd (still needs cwd=HarP workdir - see check_harp_installation)
# Written next to this namelist file, matching HarP's own input.py convention.
# Since this namelist is written directly into the HarP workdir, that's still
# workdir/DiagEnergy.h5, matching paths["diag"] in harp_input.py.
diag_path    = os.path.dirname(os.path.abspath(__file__)) + os.sep

### Main
Main(
    wavelenght          = lambda0,
    dr                  = dr,
    dz                  = dz,
    Lr                  = Lr,
    Lz                  = Lz,
    dt                  = dt,
    simulation_time     = Tsim,
    solver              = "CrankNicolson",
    anne_eq             = True,
    predictor_corrector = False,
    print_every         = 10000,
)

Pulse(
    amplitude           = E0,
    radial_profile      = eprofile,
    time_envelope       = tenv,
)

Medium(
    gaz_profile         = gazprofile,
    polarizability_0    = 1.8247e-40,
    gamma_kerr          = 1.2e-62,
    table_path          = tables_path+"form_factor/f12_Ar_henkegov.txt",
    unit_ev             = 1.0,
    unit_f1             = 1.0,   # HarP >= 849edfe convention (f1 as tabulated); was -1.0 with HarP 4dadb4b, same physics
    unit_f2             = 1.0,
    skiprows            = 1,
)

Ionisation(
    table_path          = tables_path+"ionisation/ionisation_Ar_810_tdse.txt",
    skiprows            = 2,
    intensity_unit_Wcm2 = 1.0,
    rate_unit_SI        = 1.0,
    statistical_factor  = 2.0,   # SAE spin factor (HarP v1.0.0 convention; with HarP-dev 0ad9c77 this was rate_unit_SI = 2.0)
)

order = np.array([{int(config["harmonic_order"])}], dtype=int)

Harmonic(
    order               = order,
    table_path          = tables_path+"tdse/dipoles_Ar_810.txt",
    intensity_unit_Wcm2 = 1e14,
    statistical_factor  = 2.0,   # SAE spin factor on the dipole (HarP v1.0.0 default, set explicitly)
)

DiagEnergy(
    diag_path           = diag_path,
    every_z             = 1,
    order               = order,
    SI                  = True,
)

print(f"BOinput.py: recovered pressure = {{p_in_mbar:.8f}} mbar")
print(f"BOinput.py: recovered p*z_R    = {{pzr:.8f}} mbar*cm")
print(f"BOinput.py: recovered Lgaz/z_R = {{Lzzr:.8f}}")
print(f"BOinput.py: recovered Lgaz     = {{Lgaz/mm:.8f}} mm")
print(f"BOinput.py: intensity          = {{I_input:.8e}} W/cm^2")
print(f"BOinput.py: tau FWHM           = {{tau_input:.8f}} fs")
print(f"BOinput.py: Nt                 = {{nt}}")
print(f"BOinput.py: dt                 = {{dt*T_r:.8e}} s")
"""

    paths["input"].write_text(text)
    return nt


# Run HarP and calculate the objective for one optimization point
########################################################################################################
def run_harp(values, config, paths, archive_path=None, max_attempts=3):
    # Print the parameters of the simulation that is about to start
    print(
        "Running HarP | "
        f"Lzzr={values['Lzzr']:.6f}, "
        f"pzr={values['pzr']:.6f}, "
        f"I={values['intensity_Wcm2']:.6e} W/cm^2, "
        f"tau={values['tau_fs']:.6f} fs"
    )

    # Generate the HarP input file and store the chosen temporal grid size in the local variable nt.
    # nt changes according to the box size
    nt = write_boinput(values, config, paths)

    # Retry a small number of times on failure - seen in practice: a rare
    # (~1-in-300) HarP-internal HDF5 "dataset already exists" collision that
    # crashes the HarP subprocess for no repeatable reason tied to these
    # physical parameters. Each attempt deletes any stale/partial diagnostic
    # file first, so a retry starts from a clean slate. A persistent (not
    # transient) failure still raises after max_attempts, rather than being
    # silently swallowed.
    #
    # The read of the resulting diagnostic file is inside this same retry
    # loop (not just the subprocess call): the same underlying HDF5 race can
    # also manifest as a diag file that exists but is missing the expected
    # dataset (KeyError from h5py), not just a subprocess crash or a missing
    # file.
    dataset_name = f"order_{int(config['harmonic_order'])}"
    for attempt in range(1, max_attempts + 1):
        # If the diagnostic from the previous simulation (or a failed attempt) exists, delete it first.
        if paths["diag"].exists():
            paths["diag"].unlink()

        sim_start = time.perf_counter()
        try:
            # Run HarP with absolute paths for both the script and the namelist (HarP.py
            # loads the namelist by file path, not by module name, so this no longer
            # relies on cwd to find it). cwd is still set to the HarP workdir because
            # tables_path in the generated namelist is still relative (see the NOTE in
            # check_harp_installation above).
            # Suppress the stdout of HarP in every simulation due to extensive console output
            # Standard error is left unchanged so that any HarP or Python errors remain visible
            subprocess.run([sys.executable, str(paths["script"]), str(paths["input"])], check=True, cwd=str(paths["workdir"]),stdout=subprocess.DEVNULL,)
            simulation_time_s = time.perf_counter() - sim_start
            if not paths["diag"].exists():
                raise FileNotFoundError(f"Diagnostic file missing: {paths['diag']}")
            with h5py.File(paths["diag"], "r") as h5:
                harmonic_energy = float(np.asarray(h5[dataset_name])[-1])
            break
        except (subprocess.CalledProcessError, FileNotFoundError, KeyError) as e:
            if attempt < max_attempts:
                print(f"HarP simulation attempt {attempt}/{max_attempts} failed ({e}); retrying...")
            else:
                raise

    # Recover the physical waist from z_R and lambda0.
    # From this, calculate laser energy and subsequently CE
    zR_m                  = config["zR_mm"] * 1.0e-3
    waist_m               = np.sqrt(zR_m * config["wavelength_m"] / np.pi)
    input_laser_energy    = gaussian_pulse_energy(
        peak_intensity_w_cm2=values["intensity_Wcm2"],
        waist_m=waist_m,
        duration_fwhm_s=values["tau_fs"] * 1.0e-15,
    )
    conversion_efficiency = harmonic_energy / input_laser_energy
    # The final objective is scaled so that values are normally around one.
    objective             = conversion_efficiency / config["objective_scale"]

    # Copy the HarP diagnostic file contents
    # Otherwise the next HarP run would overwrite the same DiagEnergy.h5 file,
    # and the previous result would be lost
    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths["diag"], archive_path)

    return {
        "objective": objective,
        "conversion_efficiency": conversion_efficiency,
        "input_laser_energy_J": input_laser_energy,
        "harmonic_energy_J": harmonic_energy,
        "simulation_time_s": simulation_time_s,
        "nt": nt,
    }
