---
id: develop
title: Plugin development
sidebar_label: Plugin development
---

:::info
If you develop plugins, please join the <a href="https://hydra-framework.zulipchat.com/#narrow/stream/233935-Hydra-plugin.20dev.20announcements">Plugin developer announcement chat channel</a>.
:::


import GithubLink from "@site/src/components/GithubLink"

Hydra plugins must be registered before they can be used. Declare an entry
point in the `hydra.plugins` group for each plugin class, or call the
`register` method on Hydra's `Plugins` singleton. Hydra 1.4 also discovers
plugins in the `hydra_plugins` namespace for compatibility, but this method is
deprecated. See the [plugin discovery migration guide](/docs/upgrades/1.3_to_1.4/plugin_discovery).

## Entry point discovery

For example, in `setup.py`:

```python
setup(
    # ...
    entry_points={
        "hydra.plugins": [
            "my_launcher = my_package.my_launcher:MyLauncher",
        ]
    },
)
```

The target must be a concrete Hydra plugin class. Its module should also
register any associated Hydra configuration when imported. Plugin classes no
longer need to live in the `hydra_plugins` namespace. Keep expensive optional
dependencies out of module-level imports.

## Legacy namespace discovery

If you create a Plugin and want it to be discovered automatically by Hydra, keep the following things in mind:
- Namespace-discovered plugins can be either standalone packages or part of an
  existing package. They must be in the top-level `hydra_plugins` namespace.
- Do __NOT__ place an `__init__.py` file in `hydra_plugins` (doing so may break other installed Hydra plugins).
  
The plugin discovery process runs whenever Hydra starts. During plugin discovery, Hydra scans for plugins in all the submodules of `hydra_plugins`. Hydra will import each module and look for plugins defined in that module.
Any module under `hydra_plugins` that is slow to import will slow down the startup of __ALL__ Hydra applications.
Plugins with expensive imports can exclude individual files from Hydra's plugin discovery process by prefixing them with `_` (but not `__`).
For example, the file `_my_plugin_lib.py` would not be imported and scanned, while `my_plugin_lib.py` would be.

## Plugin registration via the `Plugins.register` method

Plugins can be manually registered by calling the `register` method on the instance of Hydra's `Plugins` singleton class.
```python
from hydra.core.plugins import Plugins
from hydra.plugins.plugin import Plugin

class MyPlugin(Plugin):
  ...

def register_my_plugin() -> None:
    """Hydra users should call this function before invoking @hydra.main"""
    Plugins.instance().register(MyPlugin)
```

## Getting started

The best way to get started developing a Hydra plugin is to base your new plugin on one of the example plugins:
- Copy the subtree of the relevant <GithubLink to="examples/plugins">example plugin</GithubLink> into a standalone project.
- Edit `setup.py`, rename the plugin module, and declare its entry point.
- Install the new plugin (Run this in the plugin directory: `pip install -e .`)
- Run the included example app and make sure that the plugin is discovered:
```shell
$ python example/my_app.py --info plugins
Installed Hydra Plugins
***********************
        ...
        Launcher:
        ---------
                MyLauncher
        ...
```
- Run the example application to see that that your plugin is doing something.
- *[Optional]* Embed the plugin in your existing application or library. An
  entry-point plugin can use any importable package name.
- Hack on your plugin, Ensure that the recommended tests and any tests you want to add are passing.

For a `SearchPathPlugin` that provides organization-specific Hydra defaults,
see [Environment-specific overrides](../../patterns/environment_specific_overrides.md).
