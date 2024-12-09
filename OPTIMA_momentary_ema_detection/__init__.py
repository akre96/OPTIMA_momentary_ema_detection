import pkgutil
import yaml

config_raw = pkgutil.get_data(__name__, "../lib/config.yaml")
if config_raw is not None:
    config = yaml.safe_load(config_raw)
else:
    config = {}

definitions_raw = pkgutil.get_data(__name__, "../lib/definitions.yaml")
if definitions_raw is not None:
    definitions = yaml.safe_load(definitions_raw)
else:
    definitions = {}
