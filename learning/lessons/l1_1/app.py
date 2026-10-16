import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("lr is", cfg.lr, "| also cfg['epochs'] =", cfg["epochs"])
    
    
if __name__=="__main__":
    main()