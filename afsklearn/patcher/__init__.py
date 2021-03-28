__all__ = ["Patcher"]

import importlib
from pathlib import Path
from typing import Any, Optional

from .. import patches_info


class Patcher:
    @staticmethod
    def patch(module_name: str) -> None:
        patch_config = patches_info[module_name]
        _apply_patch(patch_config["module"], patch_config["name"], patch_config["patched_module"])

    @staticmethod
    def rollback(module_name: str) -> None:
        patch_config = patches_info[module_name]
        _apply_patch(patch_config["module"], patch_config["name"], None)

    def patch_all():
        raise NotImplemented

    def rollback_all():
        raise NotImplemented


def _load_module(module_path: str) -> Any:
    return importlib.import_module(module_path)


def _load_instance(module_path: str, instance_name: str) -> Any:
    return getattr(_load_module(module_path), instance_name)


def _apply_patch(parent_module: str, target_instance_name: str, patched_module: Optional[str]) -> None:
    if patched_module is None:
        patched_module = parent_module

    loaded_parent_module = _load_module(parent_module)
    loaded_patched_instance = _load_instance(patched_module, target_instance_name)
    setattr(loaded_parent_module, target_instance_name, loaded_patched_instance)
