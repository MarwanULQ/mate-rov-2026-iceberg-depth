#!/usr/bin/env python3
"""Automate Linux dependency setup for this repository.

What this script installs:
- APT prerequisites required by the C++ project and examples.
- udev rule installation for ZED Open Capture sensors/video access.
- Optional LibTorch download + extraction to third_party/libtorch.
- Exports directory for model files.

Usage examples:
  python3 install_prereqs.py
  python3 install_prereqs.py --skip-udev
  python3 install_prereqs.py --install-libtorch \
      --libtorch-url https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

APT_PACKAGES = [
    "build-essential",
    "cmake",
    "pkg-config",
    "zlib1g-dev",
    "libusb-1.0-0-dev",
    "libhidapi-libusb0",
    "libhidapi-dev",
    "libopencv-dev",
    "libopencv-viz-dev",
    "nlohmann-json3-dev",
    "gstreamer1.0-tools",
    "gstreamer1.0-libav",
    "gstreamer1.0-plugins-base",
    "gstreamer1.0-plugins-good",
    "gstreamer1.0-plugins-bad",
    "gstreamer1.0-plugins-ugly",
    "libgstreamer1.0-dev",
    "libgstreamer-plugins-base1.0-dev",
]


def run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def get_priv_prefix() -> list[str]:
    if os.geteuid() == 0:
        return []
    sudo = shutil.which("sudo")
    if not sudo:
        print("Error: this script needs root privileges and 'sudo' was not found.", file=sys.stderr)
        print("Run as root or install sudo.", file=sys.stderr)
        sys.exit(1)
    return [sudo]


def install_apt_packages(*, dry_run: bool) -> None:
    if shutil.which("apt-get") is None:
        print("Error: apt-get not found. This script currently supports Debian/Ubuntu-style systems.", file=sys.stderr)
        sys.exit(1)

    priv = get_priv_prefix()
    run(priv + ["apt-get", "update"], dry_run=dry_run)
    run(priv + ["apt-get", "install", "-y", *APT_PACKAGES], dry_run=dry_run)


def install_udev_rule(repo_root: Path, *, dry_run: bool) -> None:
    udev_script = repo_root / "src" / "iceberg_depth" / "udev" / "install_udev_rule.sh"
    if not udev_script.exists():
        print(f"Warning: udev script not found at {udev_script}; skipping udev installation.")
        return

    run(["bash", str(udev_script)], cwd=udev_script.parent, dry_run=dry_run)


def _extract_archive(archive_path: Path, out_dir: Path) -> None:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
        return

    # Handles .tar, .tar.gz, .tgz, and other tar-compatible formats.
    with tarfile.open(archive_path, "r:*") as tf:
        tf.extractall(out_dir)





def _find_libtorch_root(search_root: Path) -> Path | None:
    for cfg in search_root.rglob("TorchConfig.cmake"):
        if cfg.as_posix().endswith("share/cmake/Torch/TorchConfig.cmake"):
            return cfg.parents[3]
    return None


def install_libtorch(repo_root: Path, libtorch_url: str, *, force: bool, dry_run: bool) -> None:
    target = repo_root / "third_party" / "libtorch"

    if target.exists():
        if not force:
            print(f"LibTorch already exists at {target}. Use --force-libtorch to replace it.")
            return
        print(f"Replacing existing LibTorch at {target}")
        if not dry_run:
            shutil.rmtree(target)

    print(f"Downloading LibTorch from: {libtorch_url}")
    if dry_run:
        print("[dry-run] Skipping download/extract.")
        return

    with tempfile.TemporaryDirectory(prefix="libtorch_dl_") as tmp:
        tmp_path = Path(tmp)
        archive_name = libtorch_url.rstrip("/").split("/")[-1] or "libtorch_archive"
        archive_path = tmp_path / archive_name

        urllib.request.urlretrieve(libtorch_url, archive_path)

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive_path, extract_dir)

        found_root = _find_libtorch_root(extract_dir)
        if not found_root:
            raise RuntimeError(
                "Could not find share/cmake/Torch/TorchConfig.cmake in the downloaded archive."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(found_root), str(target))

    print(f"Installed LibTorch to: {target}")


def create_exports_dir(repo_root: Path, *, dry_run: bool) -> None:
    exports_dir = repo_root / "exports"
    if not dry_run:
        exports_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created exports directory: {exports_dir}")
    else:
        print(f"[dry-run] Would create exports directory: {exports_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install dependencies for the Iceberg Depth repository.")
    parser.add_argument("--skip-apt", action="store_true", help="Skip apt package installation.")
    parser.add_argument("--skip-udev", action="store_true", help="Skip udev rule installation.")
    parser.add_argument(
        "--install-libtorch",
        action="store_true",
        help="Download and install LibTorch into third_party/libtorch.",
    )
    parser.add_argument(
        "--libtorch-url",
        type=str,
        default="",
        help="LibTorch archive URL (.zip or .tar.*). Required with --install-libtorch.",
    )
    parser.add_argument(
        "--force-libtorch",
        action="store_true",
        help="Replace existing third_party/libtorch when installing LibTorch.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    if not args.skip_apt:
        install_apt_packages(dry_run=args.dry_run)
    else:
        print("Skipping apt package installation.")

    if not args.skip_udev:
        install_udev_rule(repo_root, dry_run=args.dry_run)
    else:
        print("Skipping udev rule installation.")

    if args.install_libtorch:
        if not args.libtorch_url:
            print("Error: --libtorch-url is required when --install-libtorch is set.", file=sys.stderr)
            return 2
        install_libtorch(
            repo_root,
            args.libtorch_url,
            force=args.force_libtorch,
            dry_run=args.dry_run,
        )
    else:
        print("LibTorch auto-install not requested.")

    create_exports_dir(repo_root, dry_run=args.dry_run)

    print("All requested installation steps completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
