# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pathlib import Path

from hydra.test_utils.test_utils import chdir_hydra_root, run_python_script

chdir_hydra_root()


def test_env_defaults(tmpdir: Path) -> None:
    cmd = [
        "tests/test_apps/custom_env_defaults/my_app.py",
        f'hydra.run.dir="{str(tmpdir)}"',
        "hydra.job.chdir=True",
    ]
    run_python_script(cmd)


def test_named_env_defaults() -> None:
    stdout, _ = run_python_script(
        [
            "tests/test_apps/custom_env_defaults/my_app.py",
            "hydra/env=campus",
            "--cfg",
            "hydra",
        ]
    )
    assert "FOO: campus" in stdout
    assert "dir: ./campus_outputs" in stdout


def test_named_env_defaults_from_application_config() -> None:
    stdout, _ = run_python_script(
        [
            "tests/test_apps/custom_env_defaults/my_app.py",
            "--config-path=conf",
            "--config-name=config",
            "--cfg",
            "hydra",
        ]
    )
    assert "FOO: campus" in stdout


def test_application_overrides_env_defaults() -> None:
    stdout, _ = run_python_script(
        [
            "tests/test_apps/custom_env_defaults/my_app.py",
            "--config-path=conf",
            "--config-name=app_override",
            "--cfg",
            "hydra",
        ]
    )
    assert "FOO: app" in stdout
