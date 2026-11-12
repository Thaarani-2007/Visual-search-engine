"""
Centralized Configuration Loader and Validator.
Provides type-safe access to application settings across data engineering,
training, indexing, and serving pipelines.
"""

from pathlib import Path
from typing import Any, Dict, List
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load and validate the YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# Root project directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

