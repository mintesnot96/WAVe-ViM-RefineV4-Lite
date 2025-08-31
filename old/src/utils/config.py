# src/utils/config.py
# (project root)/src/utils/config.py

import yaml
from dataclasses import dataclass
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
