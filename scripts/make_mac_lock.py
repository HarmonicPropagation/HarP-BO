#!/usr/bin/env python3
"""Derive requirements-mac.txt from the pip-compile'd requirements.txt.

requirements.txt is compiled on Linux and pins CUDA packages (nvidia-*, cuda-*, triton) that do not
exist on macOS. requirements-mac.txt is the same lock without them: every pin of requirements.txt
except those packages and any package that only they need (its "# via" entries are all dropped
packages). Everything else, versions included, is copied unchanged.

Usage (from the repository root):
  python3 scripts/make_mac_lock.py           # write requirements-mac.txt
  python3 scripts/make_mac_lock.py --check   # exit 1 and show a diff if it is out of date

Standard library only, so it runs without the virtual environment (CI, Dependabot branches).
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "requirements.txt", ROOT / "requirements-mac.txt"
LINUX_ONLY = re.compile(r"^(nvidia-|cuda-|triton$)")
HEADER = ("# Generated from requirements.txt by scripts/make_mac_lock.py (make mac-lock); do not edit.\n"
          "# The same pins without the Linux-only CUDA packages (nvidia-*, cuda-*, triton).\n")
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)")


def parse(text: str) -> list[tuple[str, str, set[str]]]:
    """(name, pin line, via set) for each pin in a pip-compile output."""
    pins: list[tuple[str, str, set[str]]] = []
    for line in text.splitlines():
        m = PIN.match(line)
        if m:
            pins.append((m.group(1).lower(), f"{m.group(1)}=={m.group(2)}", set()))
            continue
        via = line.strip().removeprefix("#").strip()
        if pins and line.startswith("    #") and via:
            via = via.removeprefix("via").strip()
            if via and not via.startswith("-r"):
                pins[-1][2].add(via.split()[0].lower())
            elif via.startswith("-r"):
                pins[-1][2].add("-r")
    return pins


def derive(text: str) -> str:
    pins = parse(text)
    dropped = {n for n, _, _ in pins if LINUX_ONLY.match(n)}
    changed = True
    while changed:  # drop packages whose every dependent is dropped
        changed = False
        for n, _, via in pins:
            if n not in dropped and via and via <= dropped:
                dropped.add(n)
                changed = True
    return HEADER + "".join(line + "\n" for n, line, _ in pins if n not in dropped)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check that requirements-mac.txt is up to date")
    args = ap.parse_args()
    want = derive(SRC.read_text())
    have = DST.read_text() if DST.exists() else ""
    if args.check:
        if have == want:
            print("requirements-mac.txt is up to date with requirements.txt")
            return 0
        sys.stdout.writelines(difflib.unified_diff(have.splitlines(True), want.splitlines(True),
                                                   "requirements-mac.txt", "derived from requirements.txt"))
        print("\nrequirements-mac.txt is out of date: run `make mac-lock` and commit the result.")
        return 1
    if have != want:
        DST.write_text(want)
        print("wrote requirements-mac.txt")
    else:
        print("requirements-mac.txt already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
