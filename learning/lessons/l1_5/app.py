import hydra
from omegaconf import DictConfig, OmegaConf

# My answer: If there is no "register_new_resolver" line, there will have bugs, need to have this line before or after main(), both are the same

# OmegaConf.register_new_resolver("mul", lambda a, b: a * b)   # register BEFORE main runs


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("run_name:", cfg.run_name)


OmegaConf.register_new_resolver("mul", lambda a, b: a * b)   # register BEFORE main runs
OmegaConf.register_new_resolver("add", lambda a, b: a + b)  


if __name__ == "__main__":
    main()
