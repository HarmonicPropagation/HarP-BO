# Reusable Python environment Makefile

VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
PIP_COMPILE := $(VENV)/bin/pip-compile
PIP_SYNC := $(VENV)/bin/pip-sync

export PYTHONPATH := ./src

# Auto-detect macOS so targets that depend on a dev environment (test,
# typecheck, run-2d/4d, repro-*) pick requirements-mac.txt automatically.
# Needed because `make` prerequisites aren't sticky across separate `make`
# invocations: even if a CI step already ran `make dev-mac` by name, a later
# step's plain `make test` would otherwise re-run its own hardcoded `dev`
# prerequisite (pip-syncing requirements.txt's CUDA pins, which don't exist
# on macOS) regardless - found via a real macos-latest CI failure.
ifeq ($(shell uname -s),Darwin)
DEV_TARGET := dev-mac
else
DEV_TARGET := dev
endif

# Where the runner scripts expect the HarP simulator (see "Linking the simulator"
# in the README). init-harp clones straight into this path - it never touches an
# existing checkout or symlink there (see init-harp below).
HARP_MAIN := $(HOME)/HarP-main
# The public HarP repository, pinned to its release v1.0.0.
HARP_REPO := https://github.com/HarmonicPropagation/HarP.git
# Pinned to the same commit CI checks out (.github/workflows/ci.yaml), for a
# reproducible install: tag v1.0.0. Update both together.
HARP_REF := 7880bace90d5bec3fd0667b389e39caabcc70de5

.PHONY: help init init-mac compile update compile-all dev dev-mac typecheck test clean clean-tests run-2d run-4d repro-2d-ei repro-2d-straddle repro-4d init-harp data-bundle data-verify data-unpack mac-lock check-mac-lock

help:
	@echo "Available commands:"
	@echo "  make init        - Create .venv and install dependencies"
	@echo "  make init-mac    - Same, but using requirements-mac.txt (no CUDA/nvidia packages)"
	@echo "  make dev         - Create/sync .venv with development and testing dependencies"
	@echo "  make dev-mac     - Same, but using requirements-mac.txt (no CUDA/nvidia packages)"
	@echo "  make compile     - Generate requirements.txt from requirements.in"
	@echo "  make compile-all - Generate both base and dev requirements"
	@echo "  make mac-lock    - Derive requirements-mac.txt from requirements.txt (run after compile)"
	@echo "  make check-mac-lock - Check that requirements-mac.txt is up to date (as CI does)"
	@echo "  make update      - Sync .venv to match requirements.txt"
	@echo "  make test        - Run standard correctness tests"
	@echo "  make typecheck   - Run static type checking via mypy"
	@echo "  make run-2d      - Execute the 2D optimization usage example"
	@echo "  make run-4d      - Execute the 4D optimization usage example"
	@echo "  make repro-2d-ei       - Regenerate the 5-parameterization 2D comparison (paper figures)"
	@echo "  make repro-2d-straddle - Regenerate the 3-threshold 2D Straddle comparison (paper figures)"
	@echo "  make repro-4d          - Regenerate the 4 4D configs (paper figures)"
	@echo "  make data-bundle DATA_VERSION=x.y.z - Build dist/HarP-BO-data-x.y.z.tar.gz (the data record)"
	@echo "  make data-verify DATA_BUNDLE=file   - Check a data bundle's checksums and its git-tracked files"
	@echo "  make data-unpack DATA_BUNDLE=file   - Put the raw files of a data bundle in place"
	@echo "  make init-harp   - Clone the HarP simulator to \$$HOME/HarP-main (refuses to touch an existing one)"
	@echo "  make clean       - Remove .venv, caches, and test artifacts"

# ---------------------------------------------------------
# Base environment & compilation
# ---------------------------------------------------------

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip pip-tools

requirements.txt: requirements.in | $(VENV)/bin/activate
	$(PIP_COMPILE) --strip-extras requirements.in

init: requirements.txt | $(VENV)/bin/activate
	$(PIP_SYNC) requirements.txt
	@echo "\nEnvironment ready. Run 'source $(VENV)/bin/activate' to enter."

# requirements.txt is pip-compile'd on Linux and pins CUDA/nvidia packages that
# don't exist on macOS. requirements-mac.txt is the same lock without them,
# derived by scripts/make_mac_lock.py (make mac-lock) - never edit it by hand.
# CI checks that the two agree (make check-mac-lock).
init-mac: | $(VENV)/bin/activate
	$(PIP_SYNC) requirements-mac.txt
	@echo "\nEnvironment ready (macOS variant, no CUDA/nvidia packages). Run 'source $(VENV)/bin/activate' to enter."

compile: requirements.txt
	python3 scripts/make_mac_lock.py

# Standard library only, so these work without .venv (CI, Dependabot branches).
mac-lock:
	python3 scripts/make_mac_lock.py

check-mac-lock:
	python3 scripts/make_mac_lock.py --check
update: init

# ---------------------------------------------------------
# Development & analysis environment
# ---------------------------------------------------------

requirements-dev.txt: requirements-dev.in requirements.txt | $(VENV)/bin/activate
	$(PIP_COMPILE) --strip-extras requirements-dev.in

compile-all: requirements.txt requirements-dev.txt

dev: requirements-dev.txt | $(VENV)/bin/activate
	$(PIP_SYNC) requirements.txt requirements-dev.txt
	@echo "\nDev environment ready. Analysis dependencies installed."

# macOS variant of dev - requirements-dev.txt itself has no CUDA/nvidia pins
# (checked), so syncing it alongside requirements-mac.txt instead of
# requirements.txt should be safe, but this hasn't been exercised on real
# macOS outside of CI - see the ci.yaml macos-latest matrix entry.
dev-mac: requirements-dev.txt | $(VENV)/bin/activate
	$(PIP_SYNC) requirements-mac.txt requirements-dev.txt
	@echo "\nDev environment ready (macOS variant). Analysis dependencies installed."

# ---------------------------------------------------------
# Execution (usage examples)
# ---------------------------------------------------------

run-2d: $(DEV_TARGET)
	$(VENV)/bin/python3 src/run_bo_2d.py

run-4d: $(DEV_TARGET)
	$(VENV)/bin/python3 src/run_bo_4d.py

# ---------------------------------------------------------
# HarP simulator setup
# ---------------------------------------------------------

# Clones HarP (public) to $(HARP_MAIN), pinned to HARP_REF above.
#
# Refuses to run if $(HARP_MAIN) already exists in any form (real directory,
# empty directory, or symlink) rather than overwriting it - many setups
# already point ~/HarP-main at an existing HarP checkout elsewhere, and this
# must never clobber that. Remove or rename
# the existing path yourself first if you really want a fresh clone here.
init-harp:
	@if [ -e "$(HARP_MAIN)" ] || [ -L "$(HARP_MAIN)" ]; then \
		echo "Refusing to run: $(HARP_MAIN) already exists (directory or symlink)."; \
		echo "This target never overwrites an existing HarP checkout or symlink."; \
		echo "Remove or rename it yourself first if you want a fresh clone here."; \
		exit 1; \
	fi
	git clone $(HARP_REPO) $(HARP_MAIN)
	cd $(HARP_MAIN) && git checkout $(HARP_REF)
	@echo "\nHarP cloned to $(HARP_MAIN) at $(HARP_REF)."

# Regenerates the exact named configs behind the paper's comparison figures
# (see the README's "Reproducing the paper's figures" section for cost estimates).
repro-2d-ei: $(DEV_TARGET)
	$(VENV)/bin/python3 src/generate_2d_runs_for_fig1_fig2.py

repro-2d-straddle: $(DEV_TARGET)
	$(VENV)/bin/python3 src/generate_2d_runs_for_straddle_fig.py

repro-4d: $(DEV_TARGET)
	$(VENV)/bin/python3 src/generate_4d_runs.py

# The data record (Zenodo): GP checkpoints and HarP diagnostics that are not in git.
# See scripts/make_data_bundle.py and scripts/data_bundle_manifest.txt. Standard library only.
data-bundle:
	@test -n "$(DATA_VERSION)" || { echo "usage: make data-bundle DATA_VERSION=x.y.z"; exit 2; }
	python3 scripts/make_data_bundle.py build --version $(DATA_VERSION)

data-verify:
	@test -n "$(DATA_BUNDLE)" || { echo "usage: make data-verify DATA_BUNDLE=path/to/HarP-BO-data-x.y.z.tar.gz"; exit 2; }
	python3 scripts/make_data_bundle.py verify $(DATA_BUNDLE)

data-unpack:
	@test -n "$(DATA_BUNDLE)" || { echo "usage: make data-unpack DATA_BUNDLE=path/to/HarP-BO-data-x.y.z.tar.gz"; exit 2; }
	python3 scripts/make_data_bundle.py unpack $(DATA_BUNDLE)

# ---------------------------------------------------------
# Testing and type checking
# ---------------------------------------------------------

typecheck: $(DEV_TARGET)
	$(VENV)/bin/mypy src/ tests/ --ignore-missing-imports --explicit-package-bases --strict

test: $(DEV_TARGET)
	$(VENV)/bin/pytest -v tests/

# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

clean: clean-tests
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-tests:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
