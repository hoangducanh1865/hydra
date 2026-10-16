# SPDX-FileCopyrightText: Contributors to Hydra
# SPDX-License-Identifier: MIT
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def test_antlr_jar_license_in_distributions(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for name in ("ATTRIBUTION", "build_helpers", "hydra", "requirements"):
        shutil.copytree(project_root / name, source_dir / name)
    for name in ("LICENSE", "MANIFEST.in", "README.md", "pyproject.toml", "setup.py"):
        shutil.copy2(project_root / name, source_dir / name)

    sdist_dir = tmp_path / "sdist"
    subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(sdist_dir)],
        cwd=source_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    sdist = next(sdist_dir.glob("*.tar.gz"))
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        pkg_info = next(name for name in names if name.endswith("/PKG-INFO"))
        metadata_file = archive.extractfile(pkg_info)
        assert metadata_file is not None
        metadata = metadata_file.read().decode("utf-8")
    assert any(
        name.endswith("/build_helpers/bin/antlr-4.11.1-complete.jar") for name in names
    )
    assert any(name.endswith("/ATTRIBUTION/LICENSE-antlr4") for name in names)
    assert "License-File: ATTRIBUTION/LICENSE-antlr4" in metadata

    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(sdist),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with zipfile.ZipFile(next(wheel_dir.glob("*.whl"))) as archive:
        names = archive.namelist()
        assert not any(name.endswith("antlr-4.11.1-complete.jar") for name in names)
        assert any(
            name.endswith("/licenses/ATTRIBUTION/LICENSE-antlr4") for name in names
        )
