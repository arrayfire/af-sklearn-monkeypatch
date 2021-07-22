from pathlib import Path
from typing import Any

import yaml

app_dir = Path(__file__).resolve().parent
__version__ = '0.1.0'


def load_yaml_file(name: str, directory: Path = app_dir) -> Any:
    path = directory / name
    with path.open() as f:
        return yaml.safe_load(f)


patches_info = load_yaml_file("patched_modules.yml")

from .patcher import Patcher

def patch_sklearn():
    Patcher.patch_all()

def unpatch_sklearn():
    Patcher.rollback_all()

__all__ = ['Patcher', 'patch_sklearn', 'unpatch_sklearn']
