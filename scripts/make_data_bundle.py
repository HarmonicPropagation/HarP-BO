"""Build, verify and unpack the HarP-BO data bundle (the data record on Zenodo).

The bundle is a .tar.gz with one top folder, HarP-BO-data-<version>/, holding
- the raw files listed in scripts/data_bundle_manifest.txt (not in git),
- every file git tracks under results/ and data/ (so the record is complete on its own),
- README.md, FILES.tsv (path, size, kind) and SHA256SUMS (for `sha256sum -c`).
Paths inside the top folder are the paths in the repository.

Usage, from the repository root:
    python scripts/make_data_bundle.py build --version 1.0.0     # -> dist/HarP-BO-data-1.0.0.tar.gz
    python scripts/make_data_bundle.py verify dist/HarP-BO-data-1.0.0.tar.gz
    python scripts/make_data_bundle.py unpack HarP-BO-data-1.0.0.tar.gz [--force]

`verify` checks every checksum in the archive and that the files git tracks are identical to
the copies in the bundle. `unpack` verifies the archive and then puts the raw files in place in
the repository. It never overwrites a file that differs, unless --force is given.
"""

import argparse
import gzip
import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "data_bundle_manifest.txt"
FORBIDDEN = (b"/Users/", b"/home/")  # local paths must not reach the data record


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def collect() -> dict[str, str]:
    """Map each bundle path (relative to the repository root) to its kind: 'raw' or 'git'."""
    files = {p: "git" for p in git("ls-files", "results", "data").splitlines()}
    for line in MANIFEST.read_text().splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        matches = sorted(ROOT.glob(pattern))
        if not matches:
            sys.exit(f"manifest pattern matches no file: {pattern}")
        for path in matches:
            rel = path.relative_to(ROOT).as_posix()
            files.setdefault(rel, "raw")
    return dict(sorted(files.items()))


def readme(version: str, commit: str, files: dict[str, str], sizes: dict[str, int]) -> str:
    groups: dict[tuple[str, str], list[int]] = {}
    for path, kind in files.items():
        parts = PurePosixPath(path).parts
        folder = "/".join(parts[:2]) + "/" if len(parts) > 2 else parts[0] + "/"
        if folder.startswith("results/bo_"):
            folder = "results/bo_*/"
        n_size = groups.setdefault((folder, kind), [0, 0])
        n_size[0] += 1
        n_size[1] += sizes[path]
    rows = "\n".join(f"| `{folder}` | {kind} | {n} | {s / 1e6:.1f} MB |"
                     for (folder, kind), (n, s) in sorted(groups.items()))
    return f"""# HarP-BO data, version {version}

The run data behind the paper's figures and tables, made with HarP-BO (git commit `{commit}`) and
HarP v1.0.0. The folder layout is the layout of the HarP-BO repository: unpack it into a clone with

    python scripts/make_data_bundle.py unpack HarP-BO-data-{version}.tar.gz

"git" files are also in the HarP-BO repository at that commit; "raw" files (GP checkpoints `*.pt`, HarP
energy diagnostics `*.h5`) are only here. `FILES.tsv` lists every file, `SHA256SUMS` their checksums
(`sha256sum -c SHA256SUMS` inside this folder).

| Folder | Kind | Files | Size |
|---|---|---|---|
{rows}

Authors: Dominika Maslarova, Patrik Jansson, Mattys Pouyez. Licence: Creative Commons Attribution 4.0
International (CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/). The HarP-BO code itself is
under the MIT licence.
"""


def build(version: str, out_dir: Path) -> None:
    if git("status", "--porcelain", "--untracked-files=no", "results", "data").strip():
        sys.exit("tracked files under results/ or data/ have local changes; commit or restore them first")
    commit = git("rev-parse", "HEAD").strip()
    files = collect()
    top = f"HarP-BO-data-{version}"
    blobs = {p: (ROOT / p).read_bytes() for p in files}
    bad = [p for p, b in blobs.items() if any(f in b for f in FORBIDDEN)]
    if bad:
        sys.exit(f"{len(bad)} files contain a local path, e.g. {bad[0]}")
    sizes = {p: len(b) for p, b in blobs.items()}
    extra = {
        "README.md": readme(version, commit, files, sizes).encode(),
        "FILES.tsv": ("path\tsize\tkind\n" + "".join(f"{p}\t{sizes[p]}\t{k}\n" for p, k in files.items())).encode(),
        "SHA256SUMS": "".join(f"{sha256(b)}  {p}\n" for p, b in blobs.items()).encode(),
    }
    mtime = int(git("show", "-s", "--format=%ct", "HEAD"))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{top}.tar.gz"
    # Reproducible: fixed order, owner and times, and no time stamp in the gzip header.
    with open(target, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, data in [*extra.items(), *blobs.items()]:
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size, info.mtime, info.mode = len(data), mtime, 0o644
            tar.addfile(info, io.BytesIO(data))
    n_raw = sum(k == "raw" for k in files.values())
    print(f"{target}: {len(files)} files ({n_raw} raw, {len(files) - n_raw} git), "
          f"{sum(sizes.values()) / 1e6:.1f} MB unpacked, {target.stat().st_size / 1e6:.1f} MB packed")


def read_bundle(archive: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """Read and checksum-verify an archive. Returns (files by repository path, kinds)."""
    members: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            parts = PurePosixPath(m.name).parts
            if not m.isfile() or len(parts) < 2 or m.name.startswith("/") or ".." in parts:
                sys.exit(f"unexpected archive member: {m.name}")
            members["/".join(parts[1:])] = tar.extractfile(m).read()
    sums = {}
    for line in members.pop("SHA256SUMS").decode().splitlines():
        digest, path = line.split("  ", 1)
        sums[path] = digest
    kinds = {}
    for line in members.pop("FILES.tsv").decode().splitlines()[1:]:
        path, _size, kind = line.split("\t")
        kinds[path] = kind
    members.pop("README.md")
    if set(sums) != set(members):
        sys.exit("SHA256SUMS does not list exactly the files in the archive")
    bad = [p for p, b in members.items() if sha256(b) != sums[p]]
    if bad:
        sys.exit(f"{len(bad)} checksum mismatches, e.g. {bad[0]}")
    return members, kinds


def verify(archive: Path) -> None:
    members, kinds = read_bundle(archive)
    tracked = set(git("ls-files", "results", "data").splitlines())
    differ = [p for p in tracked if p in members and (ROOT / p).read_bytes() != members[p]]
    missing = sorted(tracked - set(members))
    print(f"{archive}: {len(members)} files, checksums OK")
    print(f"git-tracked files: {len(tracked) - len(missing) - len(differ)} identical, "
          f"{len(differ)} differ, {len(missing)} not in the bundle")
    for p in sorted(differ)[:10] + missing[:10]:
        print("  ", p)
    if differ or missing:
        sys.exit(1)


def unpack(archive: Path, force: bool) -> None:
    members, kinds = read_bundle(archive)
    written = same = 0
    conflicts = []
    for path, data in members.items():
        target = ROOT / path
        if target.exists():
            if target.read_bytes() == data:
                same += 1
                continue
            if not force:
                conflicts.append(path)
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written += 1
    print(f"{written} files written, {same} already identical")
    if conflicts:
        print(f"{len(conflicts)} files differ and were left as they are (use --force), e.g.:")
        for p in conflicts[:10]:
            print("  ", p)
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--version", required=True)
    b.add_argument("--out", type=Path, default=ROOT / "dist")
    sub.add_parser("verify").add_argument("archive", type=Path)
    u = sub.add_parser("unpack")
    u.add_argument("archive", type=Path)
    u.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.version, args.out)
    elif args.cmd == "verify":
        verify(args.archive)
    else:
        unpack(args.archive, args.force)


if __name__ == "__main__":
    main()
