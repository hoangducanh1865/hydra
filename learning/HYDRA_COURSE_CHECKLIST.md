# Hydra Mastery Course — Checklist (Google Colab, GPU)

Goal: be able to start **every future research project** with a professional Hydra layout
(composable configs, sweeps, reproducible runs, GPU training on Colab).

**28 lessons · 10 modules · basic → advanced.** Each lesson = 1 Colab section (~10–30 min).

## How to use

1. Colab → *Runtime → Change runtime type → T4 GPU* (or better).
2. One notebook per module (`hydra_m1.ipynb`, ...) or one big notebook. Each lesson below is a
   sequence of **Colab cells**: `%%writefile ...` creates a file, `!cmd` runs a Linux command.
3. Colab disk is wiped when the runtime resets → save important runs to Google Drive (Lesson 0.1).
4. When you finish a lesson, tell me *"Lesson X.Y done"* and I will tick it below
   (`- [ ] ⬜ TODO` → `- [x] ✅ DONE`). You can also edit the file yourself.

> Version note: lessons use `hydra-core` **1.3.x** (stable, `pip install hydra-core`).
> Every `@hydra.main` uses `version_base="1.3"`. Lesson 9.2 has one optional 1.4-only feature.

---

## ✅ Master checklist

**Progress: 3 / 28**

### Module 0 — Setup
- [x] ✅ DONE — **0.1** Colab + GPU + install Hydra

### Module 1 — Basics
- [x] ✅ DONE — **1.1** Your first Hydra app (`@hydra.main`, `cfg`)
- [x] ✅ DONE — **1.2** Command-line overrides (`key=v`, `+key`, `++key`, `~key`)
- [ ] ⬜ TODO — **1.3** Output directories, working dir, `.hydra/` snapshot
- [ ] ⬜ TODO — **1.4** Config groups & the defaults list
- [ ] ⬜ TODO — **1.5** Interpolation & resolvers
- [ ] ⬜ TODO — **1.6** Mandatory values, struct mode & inspecting configs

### Module 2 — Multirun (sweeps)
- [ ] ⬜ TODO — **2.1** Grid sweeps with `--multirun`
- [ ] ⬜ TODO — **2.2** Sweep directories & `override_dirname`

### Module 3 — Structured Configs
- [ ] ⬜ TODO — **3.1** Dataclass configs + `ConfigStore` + type validation
- [ ] ⬜ TODO — **3.2** Config groups in `ConfigStore` (polymorphic optimizers)
- [ ] ⬜ TODO — **3.3** YAML files validated by a dataclass schema

### Module 4 — Instantiating objects
- [ ] ⬜ TODO — **4.1** `instantiate`: `_target_` and `_partial_`
- [ ] ⬜ TODO — **4.2** Nested/recursive instantiate, `_convert_`, runtime kwargs

### Module 5 — Composition patterns
- [ ] ⬜ TODO — **5.1** Experiment configs (`# @package _global_`, `+experiment=`)
- [ ] ⬜ TODO — **5.2** Extending/inheriting configs & `_self_` ordering
- [ ] ⬜ TODO — **5.3** Package overrides (`optimizer@opt_g`)

### Module 6 — Customizing Hydra itself
- [ ] ⬜ TODO — **6.1** Logging, job name, colored logs
- [ ] ⬜ TODO — **6.2** Callbacks (hooks on job start/end)

### Module 7 — Compose API & testing
- [ ] ⬜ TODO — **7.1** Compose API inside a notebook
- [ ] ⬜ TODO — **7.2** Testing your configs with `pytest`

### Module 8 — Launchers & sweepers
- [ ] ⬜ TODO — **8.1** Parallel jobs with the Joblib launcher
- [ ] ⬜ TODO — **8.2** Hyperparameter optimization with the Optuna sweeper
- [ ] ⬜ TODO — **8.3** Submitit launcher (local now, Slurm on your cluster)

### Module 9 — Professional research workflow
- [ ] ⬜ TODO — **9.1** Experiment tracking: TensorBoard / Weights & Biases
- [ ] ⬜ TODO — **9.2** Reproducibility: seeds, snapshots, re-running a job
- [ ] ⬜ TODO — **9.3** The research-project template (installable package, GPU training)
- [ ] ⬜ TODO — **9.4** Capstone: sweep + Optuna + Drive on the template

---

# Module 0 — Setup

## Lesson 0.1 — Colab + GPU + install Hydra

**Learn:** verify the GPU, install Hydra, mount Drive.

```python
!nvidia-smi
!pip install -q hydra-core
import hydra, torch
print("hydra", hydra.__version__, "| cuda:", torch.cuda.is_available())
```

Optional (persist results):

```python
from google.colab import drive
drive.mount("/content/drive")
```

**Done when:** `cuda: True` and a Hydra version is printed.

---

# Module 1 — Basics

## Lesson 1.1 — Your first Hydra app

**Learn:** a Hydra app = a YAML config + a function decorated with `@hydra.main`.

```python
%mkdir -p /content/L1/conf
%cd /content/L1
```

```python
%%writefile conf/config.yaml
lr: 0.001
epochs: 3
batch_size: 64
```

```python
%%writefile app.py
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("lr is", cfg.lr, "| also cfg['epochs'] =", cfg["epochs"])


if __name__ == "__main__":
    main()
```

```python
!python app.py
```

**Exercise:** add a nested key `optimizer: {name: adam}` and read `cfg.optimizer.name`.
**Done when:** you can explain what `config_path` and `config_name` mean.

> Lessons 1.2 – 1.3 reuse this `/content/L1` folder.

## Lesson 1.2 — Command-line overrides

**Learn:** change any value without editing files. The 4 prefixes:

| Syntax | Meaning |
|---|---|
| `lr=0.01` | override an existing key |
| `+seed=42` | **add** a key that is not in the config |
| `++epochs=10` | add *or* override |
| `~batch_size` | **delete** a key |

```python
%cd /content/L1
!python app.py lr=0.01
!python app.py +seed=42 ++epochs=10 '~batch_size'
!python app.py foo=1          # error on purpose: read the hint
!python app.py --help         # auto-generated help
```

**Exercise:** override a list (`'layers=[64,128]'`) and a string with spaces (`'name="my run"'`).
**Done when:** you understand why `foo=1` fails but `+foo=1` works.

## Lesson 1.3 — Output directories, working dir, `.hydra/`

**Learn:** every run gets its own folder containing a **snapshot of the exact config** used.
This is the #1 reason Hydra makes research reproducible.

```python
%%writefile app.py
import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("cwd       :", os.getcwd())
    print("output_dir:", HydraConfig.get().runtime.output_dir)
    log.info("lr=%s", cfg.lr)


if __name__ == "__main__":
    main()
```

```python
!python app.py lr=0.01
!ls -a outputs/*/*/            # app.log and .hydra/
!cat outputs/*/*/.hydra/overrides.yaml
!python app.py hydra.run.dir=runs/exp1          # choose the folder yourself
!python app.py hydra.run.dir=runs/exp2 hydra.job.chdir=True   # also cd into it
```

**Key facts:** with `version_base="1.3"` Hydra does **not** change the working directory
(`hydra.job.chdir=False`); write your files to `HydraConfig.get().runtime.output_dir`.
`.hydra/` holds `config.yaml` (final config), `overrides.yaml` (your CLI args), `hydra.yaml`.

**Exercise:** save a file `result.txt` into the output dir from `main`.
**Done when:** you can find the config used by any past run.

## Lesson 1.4 — Config groups & the defaults list

**Learn:** a *config group* is a folder of interchangeable options (`model/resnet.yaml`,
`model/vit.yaml`). The `defaults` list picks one.

```python
%mkdir -p /content/L4/conf/model
%cd /content/L4
```

```python
%%bash
cat > conf/config.yaml <<'EOF'
defaults:
  - model: resnet
  - _self_

lr: 0.001
epochs: 3
EOF
printf 'name: resnet18\nhidden: 512\n' > conf/model/resnet.yaml
printf 'name: vit_b16\nhidden: 768\n' > conf/model/vit.yaml
cp /content/L1/app.py .
```

```python
!python app.py
!python app.py model=vit
!python app.py model=vit model.hidden=256
!python app.py --info defaults-tree      # how the final config is assembled
```

**Exercise:** add a `dataset` group (`mnist.yaml`, `cifar10.yaml`) and select it from the CLI.
**Done when:** you can swap whole components with one CLI word.

## Lesson 1.5 — Interpolation & resolvers

**Learn:** values can reference other values; built-in and custom *resolvers* compute values.

```python
%mkdir -p /content/L5i/conf
%cd /content/L5i
```

```python
%%writefile conf/config.yaml
project: demo
lr: 0.001
epochs: 3
run_name: ${project}_lr${lr}_ep${epochs}        # interpolation
data_dir: ${oc.env:DATA_DIR,/content/data}      # env var with default
total_steps: ${mul:${epochs},100}               # custom resolver
stamp: ${now:%Y-%m-%d_%H-%M-%S}                 # Hydra's built-in resolver
```

```python
%%writefile app.py
import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)   # register BEFORE main()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
```

```python
!python app.py
!python app.py epochs=10 project=cifar
!DATA_DIR=/tmp/mydata python app.py
```

**Exercise:** make `run_name` include the model name from Lesson 1.4.
**Done when:** changing `epochs` automatically changes `run_name` and `total_steps`.

## Lesson 1.6 — Mandatory values, struct mode & inspecting configs

**Learn:** `???` marks a value the user *must* provide; Hydra configs are **struct**
(typos in keys raise errors instead of silently creating keys).

```python
%cd /content/L5i
!echo 'ckpt: ???' >> conf/config.yaml
```

```python
%%writefile app.py
import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("ckpt =", cfg.ckpt)          # MissingMandatoryValue if not provided


if __name__ == "__main__":
    main()
```

```python
!python app.py                 # fails: ckpt is missing
!python app.py ckpt=a.pt       # works
!python app.py ckpt=a.pt --cfg job --resolve   # print the final config, don't run
!python app.py ckpt=a.pt --cfg hydra           # print Hydra's own settings
!python app.py ckpt=a.pt cpkt=1                # typo → caught
```

**Done when:** you use `--cfg job` before long runs to check what will be executed.

---

# Module 2 — Multirun (sweeps)

## Lesson 2.1 — Grid sweeps with `--multirun`

```python
%mkdir -p /content/L2/conf
%cd /content/L2
```

```python
%%writefile conf/config.yaml
lr: 0.01
batch_size: 32
epochs: 2
seed: 0
```

```python
%%writefile train.py
import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    fake_loss = abs(cfg.lr - 0.01) + cfg.batch_size / 1000   # stand-in for real training
    log.info("lr=%s bs=%s epochs=%s -> loss=%.4f", cfg.lr, cfg.batch_size, cfg.epochs, fake_loss)
    return fake_loss


if __name__ == "__main__":
    main()
```

```python
!python train.py -m lr=0.1,0.01,0.001 batch_size=32,64          # 3 x 2 = 6 jobs
!python train.py -m 'epochs=range(1,4)' 'lr=choice(0.1,0.01)'   # sweep functions (quote them!)
```

**Exercise:** sweep over 3 seeds × 2 learning rates (6 jobs).
**Done when:** you can predict how many jobs a sweep launches.

## Lesson 2.2 — Sweep directories & `override_dirname`

**Learn:** name each job's folder after its overrides, so results are self-describing.

```python
%%writefile conf/config.yaml
lr: 0.01
batch_size: 32
epochs: 2
seed: 0

hydra:
  sweep:
    dir: multirun/${now:%Y-%m-%d}
    subdir: ${hydra.job.override_dirname}
  job:
    config:
      override_dirname:
        exclude_keys: [seed]      # don't put the seed in the folder name
```

```python
!python train.py -m lr=0.1,0.01 batch_size=32,64 seed=0,1
!ls multirun/*/
!cat multirun/*/multirun.yaml | head -20
```

**Done when:** you see folders like `batch_size=32,lr=0.01` containing `.hydra/` and logs.

---

# Module 3 — Structured Configs

## Lesson 3.1 — Dataclass configs + `ConfigStore` + type validation

**Learn:** define the config as a typed `dataclass` → IDE autocompletion and **type checking
at override time**.

```python
%mkdir -p /content/L31
%cd /content/L31
```

```python
%%writefile app.py
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 64
    device: str = "cuda"


cs = ConfigStore.instance()
cs.store(name="config", node=TrainConfig)


@hydra.main(version_base="1.3", config_path=None, config_name="config")
def main(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
```

```python
!python app.py epochs=10
!python app.py lr=oops           # ValidationError: not a float
!python app.py epochs=1.5        # error: not an int
!python app.py learning_rate=1   # error: key does not exist
```

**Done when:** you understand how this catches typos *before* a 6-hour GPU job starts.

## Lesson 3.2 — Config groups in `ConfigStore` (polymorphic optimizers)

```python
%%writefile app.py
from dataclasses import dataclass, field
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


@dataclass
class OptimizerConfig:
    name: str = MISSING
    lr: float = 1e-3


@dataclass
class AdamConfig(OptimizerConfig):
    name: str = "adam"
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SGDConfig(OptimizerConfig):
    name: str = "sgd"
    lr: float = 1e-2
    momentum: float = 0.9


@dataclass
class TrainConfig:
    defaults: list[Any] = field(default_factory=lambda: [{"optimizer": "adam"}, "_self_"])
    optimizer: OptimizerConfig = MISSING
    epochs: int = 3


cs = ConfigStore.instance()
cs.store(name="config", node=TrainConfig)
cs.store(group="optimizer", name="adam", node=AdamConfig)
cs.store(group="optimizer", name="sgd", node=SGDConfig)


@hydra.main(version_base="1.3", config_path=None, config_name="config")
def main(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
```

```python
!python app.py
!python app.py optimizer=sgd optimizer.momentum=0.5
!python app.py optimizer.momentum=0.5      # error: Adam has no momentum
```

**Done when:** you see that the *selected* option decides which fields are legal.

## Lesson 3.3 — YAML files validated by a dataclass schema

**Learn:** the most common professional pattern — humans edit YAML, a dataclass **schema**
validates it.

```python
%mkdir -p /content/L33/conf
%cd /content/L33
```

```python
%%writefile conf/config.yaml
defaults:
  - base_config        # the schema registered in ConfigStore
  - _self_

lr: 0.01
epochs: 5
```

```python
%%writefile app.py
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 3
    seed: int = 0


ConfigStore.instance().store(name="base_config", node=TrainConfig)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
```

```python
!python app.py
!python app.py lr=oops          # schema catches the wrong type
!python app.py typo_lr=1        # schema catches the unknown key
```

Now edit `conf/config.yaml` to say `epochs: many` and re-run → validation error.
**Done when:** YAML typos fail immediately with a clear message.

---

# Module 4 — Instantiating objects

> Needs PyTorch (preinstalled on Colab). Use the GPU runtime.

## Lesson 4.1 — `instantiate`: `_target_` and `_partial_`

**Learn:** build any Python object from config. Swap model/optimizer **from the CLI, with no code change**.

```python
%mkdir -p /content/L41/conf
%cd /content/L41
```

```python
%%writefile conf/config.yaml
model:
  _target_: torch.nn.Linear
  in_features: 784
  out_features: 10
optimizer:
  _target_: torch.optim.Adam
  _partial_: true        # params aren't known yet → returns functools.partial
  lr: 0.001
```

```python
%%writefile app.py
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.optimizer)(model.parameters())
    out = model(torch.randn(8, 784, device=device))
    print(out.shape, type(optimizer).__name__, device)


if __name__ == "__main__":
    main()
```

```python
!python app.py
!python app.py optimizer._target_=torch.optim.SGD optimizer.lr=0.1
!python app.py model.out_features=100
```

**Done when:** you changed the optimizer class from the terminal.

## Lesson 4.2 — Nested/recursive instantiate, `_convert_`, runtime kwargs

**Learn:** nested `_target_`s are built bottom-up; `_convert_: all` hands plain `list`/`dict`
(not `ListConfig`) to your class; extra kwargs can be passed at call time.

```python
%mkdir -p /content/L42/conf
%cd /content/L42
```

```python
%%writefile my_models.py
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, sizes: list, act: nn.Module):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

```python
%%writefile conf/config.yaml
model:
  _target_: my_models.MLP
  _convert_: all
  sizes: [784, 256, 10]
  act:
    _target_: torch.nn.ReLU
```

```python
%%writefile app.py
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    model = instantiate(cfg.model).cuda() if torch.cuda.is_available() else instantiate(cfg.model)
    print(model)


if __name__ == "__main__":
    main()
```

```python
!python app.py
!python app.py model.act._target_=torch.nn.GELU 'model.sizes=[784,512,256,10]'
```

**Exercise:** add a `dropout` module config under `model` and pass it to `MLP`.
**Done when:** a new architecture needs only a YAML file + class, no edits to `app.py`.

---

# Module 5 — Composition patterns

## Lesson 5.1 — Experiment configs

**Learn:** an *experiment* = a named bundle of overrides (`+experiment=big`). Keeps the paper's
runs in version control.

```python
%mkdir -p /content/L51/conf/model /content/L51/conf/optimizer /content/L51/conf/experiment
%cd /content/L51
```

```python
%%bash
cat > conf/config.yaml <<'EOF'
defaults:
  - model: mlp
  - optimizer: adam
  - _self_

seed: 0
epochs: 3
EOF
printf '_target_: builtins.dict\nhidden: 256\n' > conf/model/mlp.yaml
printf '_target_: builtins.dict\nhidden: 1024\nlayers: 4\n' > conf/model/big_mlp.yaml
printf 'name: adam\nlr: 0.001\n' > conf/optimizer/adam.yaml
printf 'name: sgd\nlr: 0.01\nmomentum: 0.9\n' > conf/optimizer/sgd.yaml
printf '# @package _global_\nepochs: 3\n' > conf/experiment/baseline.yaml
cat > conf/experiment/big.yaml <<'EOF'
# @package _global_
defaults:
  - override /model: big_mlp
  - override /optimizer: sgd

epochs: 20
optimizer:
  lr: 0.05
EOF
cp /content/L1/app.py .
```

```python
!python app.py +experiment=big
!python app.py -m +experiment=baseline,big        # run both experiments
```

`# @package _global_` makes the file's keys land at the config root; `override /model:` swaps
group choices.
**Done when:** `python train.py +experiment=<name>` reproduces a whole experiment.

## Lesson 5.2 — Extending/inheriting configs & `_self_` ordering

**Learn:** a config can inherit another in the same group; `_self_` decides who wins.

```python
%cd /content/L51
```

```python
%%writefile conf/model/big_mlp2.yaml
defaults:
  - mlp          # inherit from model/mlp.yaml
  - _self_       # then apply this file's values on top

hidden: 2048
layers: 6
```

```python
!python app.py model=big_mlp2
!python app.py model=big_mlp2 --info defaults-tree | head -30
```

**Experiment:** move `_self_` before `- mlp` and see how `hidden` changes.
**Done when:** you can explain "later entries override earlier ones".

## Lesson 5.3 — Package overrides (`optimizer@opt_g`)

**Learn:** use the **same** group twice under different names (GANs, teacher/student,
multi-optimizer setups).

```python
%mkdir -p /content/L53/conf/optimizer
%cd /content/L53
```

```python
%%bash
cp /content/L51/conf/optimizer/*.yaml conf/optimizer/
cp /content/L1/app.py .
cat > conf/config.yaml <<'EOF'
defaults:
  - optimizer@opt_g: adam
  - optimizer@opt_d: sgd
  - _self_
EOF
```

```python
!python app.py
!python app.py opt_g.lr=0.0002 optimizer@opt_d=adam
```

**Done when:** two components share one config group with independent values.

---

# Module 6 — Customizing Hydra itself

## Lesson 6.1 — Logging, job name, colored logs

```python
!pip install -q hydra-colorlog
%cd /content/L2
```

```python
!python train.py                                   # default: console + train.log in output dir
!python train.py hydra.verbose=__main__ hydra.job_logging.root.level=DEBUG
!python train.py hydra/job_logging=colorlog hydra/hydra_logging=colorlog
!python train.py hydra/job_logging=disabled        # silence Hydra's logging config
!python train.py hydra.job.name=my_experiment      # changes the log file name
!ls outputs/*/*/
```

Use `log = logging.getLogger(__name__)` + `log.info(...)` instead of `print` in all research code.
**Done when:** your logs show in the console *and* in a file inside the run folder.

## Lesson 6.2 — Callbacks

**Learn:** run custom code on job start/end (upload results, send a notification, log time) —
without touching the training code.

```python
%mkdir -p /content/L62/conf
%cd /content/L62
```

```python
%%writefile callbacks.py
import logging
import time

from hydra.experimental.callback import Callback

log = logging.getLogger(__name__)


class Timer(Callback):
    def on_job_start(self, config, **kwargs):
        self.t0 = time.time()
        log.info("job started")

    def on_job_end(self, config, job_return, **kwargs):
        log.info("job ended in %.2fs, return=%s", time.time() - self.t0, job_return.return_value)
```

```python
%%writefile conf/config.yaml
x: 1
hydra:
  callbacks:
    timer:
      _target_: callbacks.Timer
```

```python
!cp /content/L1/app.py .
!python app.py
!python app.py -m x=1,2
```

(Callbacks are marked *experimental* in Hydra, the API may change.)
**Exercise:** implement `on_job_end` to copy `.hydra/config.yaml` to Google Drive.
**Done when:** the timer message appears for every job in a sweep.

---

# Module 7 — Compose API & testing

## Lesson 7.1 — Compose API inside a notebook

**Learn:** `@hydra.main` is for scripts. In a notebook/Colab cell use `compose` to build the
same config interactively (great for debugging and exploration).

```python
%cd /content/L51
```

```python
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

GlobalHydra.instance().clear()          # needed if you re-run this cell
with initialize_config_dir(version_base="1.3", config_dir="/content/L51/conf"):
    cfg = compose(config_name="config", overrides=["model=big_mlp", "epochs=2"])
print(OmegaConf.to_yaml(cfg))
```

**Exercise:** `compose` the Lesson 4.2 config and call `instantiate(cfg.model)` in the notebook.
**Done when:** you can load any config from a notebook and build the model interactively.

## Lesson 7.2 — Testing your configs with `pytest`

**Learn:** catch broken config combos in seconds, before burning GPU hours.

```python
!pip install -q pytest
%cd /content/L51
```

```python
%%writefile test_config.py
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

CONF = str(Path(__file__).parent / "conf")


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()


@pytest.mark.parametrize("model", ["mlp", "big_mlp"])
def test_models_compose(model):
    with initialize_config_dir(version_base="1.3", config_dir=CONF):
        cfg = compose(config_name="config", overrides=[f"model={model}"])
    assert cfg.model.hidden > 0


def test_experiment_overrides():
    with initialize_config_dir(version_base="1.3", config_dir=CONF):
        cfg = compose(config_name="config", overrides=["+experiment=big"])
    assert cfg.optimizer.name == "sgd" and cfg.epochs == 20
```

```python
!python -m pytest -q test_config.py
```

**Done when:** all tests pass, and breaking a YAML key makes one fail.

---

# Module 8 — Launchers & sweepers

> Plugins are separate pip packages selected by config group (`hydra/launcher=...`,
> `hydra/sweeper=...`).

## Lesson 8.1 — Parallel jobs with the Joblib launcher

```python
!pip install -q hydra-joblib-launcher
%cd /content/L2
```

```python
!python train.py -m hydra/launcher=joblib hydra.launcher.n_jobs=2 lr=0.1,0.01,0.001
```

> Colab has **one GPU**. Parallel jobs help for CPU work (data prep, evaluation, small models);
> for GPU training keep `n_jobs` small or the jobs will fight over GPU memory.

**Done when:** you see 3 jobs run concurrently.

## Lesson 8.2 — Hyperparameter optimization with the Optuna sweeper

**Learn:** instead of a grid, let Optuna (Bayesian/TPE) search. Your `main` **returns the
objective** (loss or accuracy).

```python
!pip install -q hydra-optuna-sweeper
%mkdir -p /content/L82/conf
%cd /content/L82
!cp /content/L2/train.py .
```

```python
%%writefile conf/config.yaml
defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    sampler:
      seed: 42
    direction: minimize         # use "maximize" for accuracy
    n_trials: 20
    n_jobs: 1
    params:
      lr: tag(log, interval(0.0001, 0.1))
      batch_size: choice(32, 64, 128)

lr: 0.01
batch_size: 64
epochs: 2
```

```python
!python train.py -m
```

Read the last lines: `Best parameters` and `Best value`.
**Exercise:** change `direction` to `maximize` and make `train.py` return an accuracy-like value.
**Done when:** Optuna finds a `lr` near 0.01 (the fake loss's minimum).

## Lesson 8.3 — Submitit launcher (local now, Slurm on your cluster)

**Learn:** the same code runs on a Slurm cluster by switching one config group. Colab has no
Slurm, so run `submitit_local` and only *inspect* the Slurm config.

```python
!pip install -q hydra-submitit-launcher
%cd /content/L2
```

```python
!python train.py -m hydra/launcher=submitit_local hydra.launcher.timeout_min=5 lr=0.1,0.01
# What you would run on a cluster (inspect only, do not launch in Colab):
!python train.py hydra/launcher=submitit_slurm hydra.launcher.gpus_per_node=1 hydra.launcher.partition=gpu hydra.launcher.timeout_min=120 --cfg hydra | grep -A12 "launcher:"
```

**Done when:** you can write the command that would submit this sweep to Slurm.

---

# Module 9 — Professional research workflow

## Lesson 9.1 — Experiment tracking: TensorBoard / Weights & Biases

**Learn:** write logs *inside the Hydra run folder* so every run's config and metrics live together.

```python
%load_ext tensorboard
%tensorboard --logdir /content/proj/multirun   # after Lesson 9.3/9.4 produce runs
```

Pattern for your training script (used in Lesson 9.3):

```python
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter

out = HydraConfig.get().runtime.output_dir
writer = SummaryWriter(out)
writer.add_scalar("test/acc", acc, epoch)
```

Weights & Biases (optional, needs a free account; `wandb.login()` first):

```python
import wandb
from omegaconf import OmegaConf

wandb.init(
    project=cfg.project,
    config=OmegaConf.to_container(cfg, resolve=True),   # the full Hydra config as a dict
    name=HydraConfig.get().job.override_dirname or None,
)
```

**Done when:** you know where to put the TensorBoard writer and how to send `cfg` to W&B.

## Lesson 9.2 — Reproducibility: seeds, snapshots, re-running a job

**Learn:** a run is reproducible if you can rebuild its exact config and seed.

```python
import os, random
import numpy as np, torch

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
```

Call `seed_everything(cfg.seed)` first in `main`, with `seed` in the config.

Re-run any old job from the snapshot Hydra saved:

```python
%cd /content/L33
!python app.py epochs=9 hydra.run.dir=outputs/exp_a
!cat outputs/exp_a/.hydra/overrides.yaml
!python app.py --config-path outputs/exp_a/.hydra --config-name config hydra.run.dir=outputs/exp_a_rerun
```

*Optional, Hydra 1.4 only* (experimental): `!pip install -q --pre hydra-core`, then
`!python app.py --experimental-rerun outputs/exp_a` (rebuilds from the stored overrides).

**Done when:** the re-run prints exactly the same config as the original run.

## Lesson 9.3 — The research-project template

**This is the layout to copy for every future project.**

```
proj/
├── pyproject.toml
├── conf/
│   ├── config.yaml          # defaults list + global settings
│   ├── model/    mlp.yaml  cnn.yaml
│   ├── optimizer/ adam.yaml sgd.yaml
│   └── experiment/ fast_debug.yaml
├── src/proj/
│   ├── __init__.py
│   ├── models.py
│   └── train.py             # @hydra.main entry point
└── tests/                    # config tests (Lesson 7.2)
```

```python
%mkdir -p /content/proj/src/proj /content/proj/conf/model /content/proj/conf/optimizer /content/proj/conf/experiment
%cd /content/proj
```

```python
%%bash
touch src/proj/__init__.py

cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "proj"
version = "0.1.0"
dependencies = ["hydra-core"]

[tool.setuptools.packages.find]
where = ["src"]
EOF

cat > conf/config.yaml <<'EOF'
defaults:
  - model: mlp
  - optimizer: adam
  - _self_

project: fashion_mnist
seed: 0
epochs: 3
batch_size: 128
data_dir: /content/data
EOF

cat > conf/model/mlp.yaml <<'EOF'
_target_: proj.models.MLP
hidden: 256
EOF
cat > conf/model/cnn.yaml <<'EOF'
_target_: proj.models.CNN
channels: 32
EOF
cat > conf/optimizer/adam.yaml <<'EOF'
_target_: torch.optim.Adam
_partial_: true
lr: 0.001
EOF
cat > conf/optimizer/sgd.yaml <<'EOF'
_target_: torch.optim.SGD
_partial_: true
lr: 0.05
momentum: 0.9
EOF
cat > conf/experiment/fast_debug.yaml <<'EOF'
# @package _global_
epochs: 1
batch_size: 512
EOF
```

```python
%%writefile src/proj/models.py
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 10))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(channels * 2 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)
```

```python
%%writefile src/proj/train.py
import logging
import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(out)

    tf = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(cfg.data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST(cfg.data_dir, train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=1024)

    model = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.optimizer)(model.parameters())

    acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        acc = correct / len(test_ds)
        log.info("epoch=%d test_acc=%.4f", epoch, acc)
        writer.add_scalar("test/acc", acc, epoch)

    torch.save(model.state_dict(), f"{out}/model.pt")
    return acc


if __name__ == "__main__":
    main()
```

```python
!pip install -q -e .
!python -m proj.train +experiment=fast_debug                 # quick smoke test on GPU
!python -m proj.train model=cnn optimizer=sgd epochs=3       # a real run
!python -m proj.train -m model=mlp,cnn optimizer=adam,sgd +experiment=fast_debug   # 4 jobs
!ls outputs/*/*/ multirun/*/*/
```

**Done when:** the 4-job sweep finishes and each run folder has `.hydra/`, a log, TensorBoard
events and `model.pt`.

## Lesson 9.4 — Capstone: sweep + Optuna + Drive on the template

Extend the template **without touching `train.py`'s logic** (only config + commands):

1. Mount Drive and send all outputs there:
   ```python
   from google.colab import drive; drive.mount("/content/drive")
   ```
   ```python
   !python -m proj.train hydra.run.dir=/content/drive/MyDrive/hydra_runs/single_run model=cnn
   ```
2. Add `hydra.sweep.dir` / `subdir: ${hydra.job.override_dirname}` (Lesson 2.2) to `conf/config.yaml`,
   pointing at Drive.
3. Add an Optuna config (Lesson 8.2) searching `optimizer.lr` (log scale) and `batch_size`,
   `direction: maximize` (the script returns test accuracy), `n_trials: 10`, `epochs=2`.
   `!pip install -q hydra-optuna-sweeper` first.
4. Add a `model/cnn_wide.yaml` that extends `cnn` (Lesson 5.2) with `channels: 64`.
5. Add a pytest that composes every `model × optimizer` pair (Lesson 7.2).
6. Open TensorBoard on the sweep folder (Lesson 9.1) and pick the best run.
7. Re-run the best run from its `.hydra/` snapshot (Lesson 9.2) and confirm the same accuracy
   (± GPU nondeterminism).

**Done when:** all 7 steps work and you could hand `proj/` to a colleague who reproduces your
best result with one command.

---

## Cheat sheet

| Task | Command |
|---|---|
| Override a value | `python -m proj.train lr=0.01` |
| Add / delete key | `+seed=1` / `'~batch_size'` |
| Pick a group option | `model=cnn` |
| Run an experiment | `+experiment=fast_debug` |
| Sweep | `-m lr=0.1,0.01 model=mlp,cnn` |
| Show final config only | `--cfg job --resolve` |
| Show Hydra's config | `--cfg hydra` |
| Show defaults tree | `--info defaults-tree` |
| Choose output folder | `hydra.run.dir=...` / `hydra.sweep.dir=...` |
| Change working dir | `hydra.job.chdir=True` |
| Use a plugin | `hydra/launcher=joblib`, `hydra/sweeper=optuna` |
| Help | `--help`, `--hydra-help` |
