import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    out = HydraConfig.get().runtime.output_dir
    print("cwd       :", os.getcwd())
    print("output_dir:", out)
    log.info("lr=%s", cfg.lr)

    with open(f"{out}/result.txt", "w") as f:
        f.write(f"lr={cfg.lr}\n")


if __name__ == "__main__":
    main()
