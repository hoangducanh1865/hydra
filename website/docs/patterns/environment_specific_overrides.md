---
id: environment_specific_overrides
title: Environment-specific overrides
---

import {ExampleGithubLink} from "@site/src/components/GithubLink"

An organization, such as a company or university, may want Hydra applications
to share defaults that are specific to its environment. Hydra reserves the
`hydra/env` config group for this purpose. Its built-in `default` option is
empty, so it has no effect unless another config source supplies an override.

This changes Hydra's configuration, not an application's primary config.
Application settings and command-line overrides can still override the values
provided here. Typical uses include choosing a default launcher, setting
environment variables for jobs, and changing the output directory for runs or
multiruns.

## Provide defaults automatically

Package a `hydra/env/default.yaml` file in an importable Python package:

```yaml title="my_org/hydra_defaults/conf/hydra/env/default.yaml"
# @package _global_
defaults:
  - override /hydra/launcher: joblib

hydra:
  job:
    env_set:
      ORG: example
  run:
    dir: ./org_outputs/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: ./org_multirun/${now:%Y-%m-%d_%H-%M-%S}
```

This example selects Joblib for multiruns (assuming its launcher plugin is
installed), sets `ORG=example` in Hydra jobs, and changes the output paths.
The absolute `/hydra/launcher` path is needed because the Defaults List is
inside the `hydra/env` group. The `# @package _global_` directive puts the
values at the root of Hydra's configuration. Include `__init__.py` in each
package directory used by the `pkg://` path.

Use a `SearchPathPlugin` to place the package **before** Hydra's built-in
config source. See the <ExampleGithubLink text="example SearchPathPlugin" to="examples/plugins/example_searchpath_plugin/"/> for a complete plugin layout.

```python title="my_org/hydra_defaults/plugin.py"
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class OrgDefaultsSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.prepend(
            provider="org-defaults",
            path="pkg://my_org.hydra_defaults.conf",
            anchor="hydra",
        )
```

Register the plugin using a `hydra.plugins` entry point in the package's
`pyproject.toml`:

```toml
[project.entry-points."hydra.plugins"]
org_defaults = "my_org.hydra_defaults.plugin:OrgDefaultsSearchPathPlugin"
```

Package `hydra/env/default.yaml` with the plugin distribution, under the
`conf` package referenced by its `pkg://` path. Ensure the YAML file is
included in the installed wheel, not only in the source checkout.

Once the package is installed, Hydra selects this `default.yaml` for every
application that uses the plugin. `anchor="hydra"` matters: placing the source
after Hydra's own source leaves Hydra's empty `default.yaml` as the first
match. See [plugin development](../advanced/plugins/develop.md) for more on
packaging and registration.

:::caution
An override named `hydra/env/default.yaml` affects every Hydra application
where the plugin is installed. Keep its contents limited to settings that are
appropriate across that environment.
:::

## Make the override opt-in

If applications should choose the environment explicitly, package a named
option instead, such as `hydra/env/campus.yaml`, with the same plugin setup.
Select it on the command line:

```shell
python my_app.py hydra/env=campus
```

Or select it from an application's Defaults List:

```yaml title="config.yaml"
defaults:
  - override hydra/env: campus
  - _self_
```

The named option avoids changing applications that do not select it. In
either mode, you can inspect the result with `--cfg hydra` and the source
order with `--info searchpath`. See [Config Search Path](../advanced/search_path.md)
for how Hydra chooses the first matching config.
