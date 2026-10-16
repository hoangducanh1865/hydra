# SPDX-FileCopyrightText: Contributors to Hydra
# SPDX-License-Identifier: MIT
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        raise ValueError(f"submitit remote cause {cfg.job}")
    except ValueError as cause:
        raise RuntimeError(f"submitit remote failure {cfg.job}") from cause


if __name__ == "__main__":
    main()
