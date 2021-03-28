from pathlib import Path
from typing import Any

import yaml

app_dir = Path(__file__).resolve().parent


def load_yaml_file(name: str, directory: Path = app_dir) -> Any:
    path = directory / name
    with path.open() as f:
        return yaml.safe_load(f)


patches_info = load_yaml_file("patched_modules.yml")
